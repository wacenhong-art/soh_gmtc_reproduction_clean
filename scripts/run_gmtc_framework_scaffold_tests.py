"""Deterministic synthetic-only structural tests for the Phase 4 G-MTC scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from gmtc_framework_scaffold import (  # noqa: E402
    BranchPrediction,
    ComponentInput,
    DeterministicMockBranch,
    GMTCFrameworkScaffold,
    GMTCScaffoldError,
    GMTCScaffoldProvenanceError,
    StructuralResidualBranchAdapter,
    SeasonalModelPort,
    TrendModelPort,
)
from itransformer_structural import (  # noqa: E402
    ITransformerStructuralConfig,
    ParameterizedITransformerStructural,
)


FIXTURE_SEED = 20260912
SEQ_LEN = 10
PRED_LEN = 1
N_VARS = 3
BATCH_SIZE = 2
SAMPLE_PROVENANCE = "SYNTHETIC_GMTC_COMPONENT_FIXTURE"
PARTITION_PROVENANCE = "SYNTHETIC_GMTC_PARTITION_FIXTURE"
BATCH_IDS = ("SYNTHETIC_BATCH_0", "SYNTHETIC_BATCH_1")
TIME_ALIGNMENT = "SYNTHETIC_ONE_STEP_ALIGNMENT"
FORECAST_HORIZON = "SYNTHETIC_PRED_LEN_1"


def set_seed() -> None:
    random.seed(FIXTURE_SEED)
    np.random.seed(FIXTURE_SEED)
    torch.manual_seed(FIXTURE_SEED)
    torch.use_deterministic_algorithms(True)


def make_component(component_type: str, tensor: torch.Tensor, **overrides) -> ComponentInput:
    values = {
        "component_type": component_type,
        "input_tensor": tensor,
        "seq_len": SEQ_LEN,
        "batch_ids": BATCH_IDS,
        "time_alignment": TIME_ALIGNMENT,
        "forecast_horizon": FORECAST_HORIZON,
        "sample_provenance": SAMPLE_PROVENANCE,
        "partition_provenance": PARTITION_PROVENANCE,
        "fidelity_status": "SYNTHETIC_STL_FIXTURE_ONLY",
        "synthetic_only": True,
    }
    values.update(overrides)
    return ComponentInput(**values)


def fixture_components() -> dict[str, ComponentInput]:
    time = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, SEQ_LEN, 1)
    variable = torch.arange(N_VARS, dtype=torch.float32).view(1, 1, N_VARS)
    batch = torch.arange(BATCH_SIZE, dtype=torch.float32).view(BATCH_SIZE, 1, 1)
    trend = 1.0 - 0.01 * time + 0.02 * batch + 0.001 * variable
    seasonal = 0.05 * torch.sin(2.0 * torch.pi * time / 4.0) + 0.002 * variable + 0.0 * batch
    residual = 0.01 * torch.cos(2.0 * torch.pi * time / 3.0) + 0.003 * batch + 0.001 * variable
    return {
        "TREND": make_component("TREND", trend),
        "SEASONAL": make_component("SEASONAL", seasonal),
        "RESIDUAL": make_component("RESIDUAL", residual),
    }


def residual_model() -> ParameterizedITransformerStructural:
    return ParameterizedITransformerStructural(
        ITransformerStructuralConfig(SEQ_LEN, PRED_LEN, N_VARS, 8, 2, 1, 16, 0.0, "gelu", True, True)
    )


def build_predictions() -> tuple[BranchPrediction, BranchPrediction, BranchPrediction]:
    components = fixture_components()
    trend = DeterministicMockBranch("TREND").predict(components["TREND"])
    seasonal = DeterministicMockBranch("SEASONAL").predict(components["SEASONAL"])
    residual = StructuralResidualBranchAdapter(residual_model()).predict(components["RESIDUAL"])
    return trend, seasonal, residual


def expect_failure(callback, error_type=GMTCScaffoldError) -> bool:
    try:
        callback()
    except error_type:
        return True
    return False


def main() -> None:
    set_seed()
    predictions = build_predictions()
    scaffold = GMTCFrameworkScaffold()
    payload = scaffold.build_future_fusion_payload(predictions)
    expected_components = {"TREND", "SEASONAL", "RESIDUAL"}
    integration = {
        "three_component_inputs_accepted": True,
        "fusion_payload_component_names": sorted(payload["branch_predictions"].keys()),
        "fusion_payload_has_exactly_three_predictions": set(payload["branch_predictions"].keys()) == expected_components,
        "fusion_executed": payload["fusion_executed"],
        "branch_shapes": {name: list(item.output_prediction.shape) for name, item in payload["branch_predictions"].items()},
        "residual_branch_fidelity": payload["branch_predictions"]["RESIDUAL"].model_fidelity_status,
        "trend_branch_fidelity": payload["branch_predictions"]["TREND"].model_fidelity_status,
        "seasonal_branch_fidelity": payload["branch_predictions"]["SEASONAL"].model_fidelity_status,
    }
    trend, seasonal, residual = predictions
    wrong_alignment = BranchPrediction(**{**residual.__dict__, "time_alignment": "SYNTHETIC_MISMATCH"})
    wrong_batch = BranchPrediction(**{**residual.__dict__, "batch_ids": ("SYNTHETIC_OTHER_0", "SYNTHETIC_OTHER_1")})
    components = fixture_components()
    fail_closed = {
        "trend_timesnet_port_unimplemented": expect_failure(
            lambda: TrendModelPort().predict(components["TREND"])
        ),
        "seasonal_timesnet_port_unimplemented": expect_failure(
            lambda: SeasonalModelPort().predict(components["SEASONAL"])
        ),
        "missing_component": expect_failure(lambda: scaffold.build_future_fusion_payload((trend, seasonal))),
        "duplicate_component": expect_failure(lambda: scaffold.build_future_fusion_payload((trend, seasonal, trend))),
        "mismatched_time_alignment": expect_failure(lambda: scaffold.build_future_fusion_payload((trend, seasonal, wrong_alignment))),
        "mismatched_batch_ids": expect_failure(lambda: scaffold.build_future_fusion_payload((trend, seasonal, wrong_batch))),
        "wrong_seq_len": expect_failure(lambda: DeterministicMockBranch("TREND").predict(make_component("TREND", components["TREND"].input_tensor, seq_len=9))),
        "real_data_provenance": expect_failure(lambda: DeterministicMockBranch("TREND").predict(make_component("TREND", components["TREND"].input_tensor, sample_provenance="NASA_REAL")), GMTCScaffoldProvenanceError),
        "test_provenance": expect_failure(lambda: DeterministicMockBranch("TREND").predict(make_component("TREND", components["TREND"].input_tensor, partition_provenance="SYNTHETIC_TEST")), GMTCScaffoldProvenanceError),
    }
    all_passed = integration["three_component_inputs_accepted"] and integration["fusion_payload_has_exactly_three_predictions"] and not integration["fusion_executed"] and all(fail_closed.values())
    audit = {
        "artifact_type": "SYNTHETIC_ONLY_GMTC_FRAMEWORK_SCAFFOLD_STRUCTURAL_AUDIT",
        "real_battery_data_used": False,
        "real_stl_components_used": False,
        "real_timesnet_used": False,
        "real_itransformer_used": False,
        "validation_values_used": False,
        "test_loaded": False,
        "model_trained": False,
        "model_metrics_used": False,
        "grid_search_run": False,
        "seq_len": SEQ_LEN,
        "seq_len_provenance": "PAPER_EXACT",
        "real_pred_len_inference": False,
        "synthetic_pred_len": PRED_LEN,
        "component_ratio_definition_status": "UNRESOLVED",
        "fusion_feature_d_status": "UNRESOLVED",
        "integration": integration,
        "fail_closed": fail_closed,
        "all_structural_tests_passed": all_passed,
    }
    destination = PROJECT_ROOT / "output" / "framework" / "gmtc_scaffold_structural_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not all_passed:
        raise SystemExit("G-MTC scaffold structural tests failed")
    print(destination)


if __name__ == "__main__":
    main()
