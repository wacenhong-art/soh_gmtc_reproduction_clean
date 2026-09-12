"""Pre-registered synthetic-only structural tests for the generic TimesNet core."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from gmtc_framework_scaffold import (  # noqa: E402
    ComponentInput,
    GMTCFrameworkScaffold,
    StructuralResidualBranchAdapter,
)
from itransformer_structural import (  # noqa: E402
    ITransformerStructuralConfig,
    ParameterizedITransformerStructural,
)
from timesnet_structural import (  # noqa: E402
    ParameterizedStandardTimesNet,
    SyntheticStandardTimesNetAdapter,
    TimesNetConfigError,
    TimesNetInputError,
    TimesNetProvenanceError,
    TimesNetStructuralConfig,
    reshape_periodic_2d,
    restore_periodic_2d,
)


FIXTURE_SEED = 20260913
SEQ_LEN = 10
PRED_LEN = 1
BATCH_SIZE = 2
SYNTHETIC_N_VARS = (1, 3)
SYNTHETIC_TOPK_VALUES = (1, 2)
OVERFIT_MAX_STEPS = 32
OVERFIT_RULE = "final_loss < initial_loss"
SYNTHETIC_PROVENANCE = "SYNTHETIC_TIMESNET_COMPONENT_FIXTURE"
SYNTHETIC_PARTITION = "SYNTHETIC_TIMESNET_PARTITION_FIXTURE"


def set_seed(seed: int = FIXTURE_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def config(n_vars: int, top_k: int) -> TimesNetStructuralConfig:
    return TimesNetStructuralConfig(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        n_vars=n_vars,
        top_k=top_k,
        d_model=8,
        d_ff=16,
        e_layers=1,
        num_kernels=2,
        dropout=0.0,
        strict_paper_mode=True,
        synthetic_only_mode=True,
        branch_variant="STANDARD_REFERENCE_CORE",
    )


def periodic_fixture(n_vars: int, periods: tuple[int, ...]) -> torch.Tensor:
    time = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, SEQ_LEN, 1)
    batch = torch.arange(BATCH_SIZE, dtype=torch.float32).view(BATCH_SIZE, 1, 1)
    variable = torch.arange(n_vars, dtype=torch.float32).view(1, 1, n_vars)
    signal = sum(torch.sin(2.0 * torch.pi * time / float(period)) for period in periods)
    return signal + 0.02 * batch + 0.01 * variable


def synthetic_target(x: torch.Tensor) -> torch.Tensor:
    return 0.4 * x[:, -1:, :] + 0.05


def expect_failure(callback, exception_type) -> bool:
    try:
        callback()
    except exception_type:
        return True
    return False


def shape_and_backward(n_vars: int, top_k: int, periods: tuple[int, ...]) -> dict[str, object]:
    set_seed(FIXTURE_SEED + n_vars + top_k)
    model = ParameterizedStandardTimesNet(config(n_vars, top_k))
    x = periodic_fixture(n_vars, periods)
    output, trace = model.forward_with_trace(x, provenance=SYNTHETIC_PROVENANCE)
    loss = nn.MSELoss()(output, synthetic_target(x))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    return {
        "n_vars": n_vars,
        "top_k": top_k,
        "known_synthetic_periods": list(periods),
        "detected_periods": trace["blocks"][0]["detected_periods"],
        "period_detection_status": set(periods).issubset(set(trace["blocks"][0]["detected_periods"])),
        "input_shape": trace["input"],
        "embedded_shape": trace["embedded"],
        "output_shape": trace["output"],
        "forward_pass": bool(torch.isfinite(output).all()),
        "backward_pass": True,
        "finite_loss": bool(torch.isfinite(loss)),
        "nonzero_gradient": any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
        "finite_gradients": all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients),
    }


def padding_test() -> dict[str, object]:
    source = periodic_fixture(1, (5,))
    reshaped, padding = reshape_periodic_2d(source, 3)
    restored = restore_periodic_2d(reshaped, original_length=SEQ_LEN)
    return {
        "period": 3,
        "padding_rule": "RIGHT_ZERO_PAD_TO_CEIL_T_OVER_PERIOD_MULTIPLE_THEN_CROP_BACK",
        "padding": padding,
        "reshaped_shape": list(reshaped.shape),
        "crop_back_shape": list(restored.shape),
        "input_preserved_after_reshape_restore": bool(torch.equal(source, restored)),
    }


def determinism_test() -> bool:
    x = periodic_fixture(3, (5, 2))
    set_seed()
    first = ParameterizedStandardTimesNet(config(3, 2)).eval()
    first_output = first(x, provenance=SYNTHETIC_PROVENANCE)
    set_seed()
    second = ParameterizedStandardTimesNet(config(3, 2)).eval()
    second_output = second(x, provenance=SYNTHETIC_PROVENANCE)
    return bool(torch.equal(first_output, second_output))


def tiny_overfit_test() -> dict[str, object]:
    set_seed()
    model = ParameterizedStandardTimesNet(config(1, 1))
    x = periodic_fixture(1, (5,))
    target = synthetic_target(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    initial_loss = None
    final_loss = None
    for _ in range(OVERFIT_MAX_STEPS):
        optimizer.zero_grad()
        loss = loss_function(model(x, provenance=SYNTHETIC_PROVENANCE), target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return {"max_steps": OVERFIT_MAX_STEPS, "rule": OVERFIT_RULE, "initial_loss": initial_loss, "final_loss": final_loss, "pass": bool(final_loss < initial_loss)}


def framework_adapter_test() -> bool:
    x = periodic_fixture(3, (5, 2))
    common = {
        "input_tensor": x,
        "seq_len": SEQ_LEN,
        "batch_ids": ("SYNTHETIC_B0", "SYNTHETIC_B1"),
        "time_alignment": "SYNTHETIC_ONE_STEP_ALIGNMENT",
        "forecast_horizon": "SYNTHETIC_PRED_LEN_1",
        "sample_provenance": SYNTHETIC_PROVENANCE,
        "partition_provenance": SYNTHETIC_PARTITION,
        "fidelity_status": "SYNTHETIC_STL_FIXTURE_ONLY",
        "synthetic_only": True,
    }
    trend = SyntheticStandardTimesNetAdapter(component_type="TREND", model=ParameterizedStandardTimesNet(config(3, 2))).predict(ComponentInput(component_type="TREND", **common))
    seasonal = SyntheticStandardTimesNetAdapter(component_type="SEASONAL", model=ParameterizedStandardTimesNet(config(3, 2))).predict(ComponentInput(component_type="SEASONAL", **common))
    residual_model = ParameterizedITransformerStructural(
        ITransformerStructuralConfig(SEQ_LEN, PRED_LEN, 3, 8, 2, 1, 16, 0.0, "gelu", True, True)
    )
    residual = StructuralResidualBranchAdapter(residual_model).predict(ComponentInput(component_type="RESIDUAL", **common))
    payload = GMTCFrameworkScaffold().build_future_fusion_payload((trend, seasonal, residual))
    return set(payload["branch_predictions"]) == {"TREND", "SEASONAL", "RESIDUAL"} and not payload["fusion_executed"]


def main() -> None:
    results = [
        shape_and_backward(1, 1, (5,)),
        shape_and_backward(3, 2, (5, 2)),
    ]
    padding = padding_test()
    deterministic = determinism_test()
    overfit = tiny_overfit_test()
    fixture = periodic_fixture(1, (5,))
    fail_closed = {
        "strict_seq_len_not_10": expect_failure(lambda: ParameterizedStandardTimesNet(TimesNetStructuralConfig(9, 1, 1, 1, 8, 16, 1, 2, 0.0, True, True, "STANDARD_REFERENCE_CORE")), TimesNetConfigError),
        "missing_config": expect_failure(lambda: ParameterizedStandardTimesNet(None), TimesNetConfigError),
        "invalid_top_k": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 6)), TimesNetConfigError),
        "wrong_tensor_rank": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(torch.zeros(2, 10), provenance=SYNTHETIC_PROVENANCE), TimesNetInputError),
        "nan_input": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(torch.full((2, 10, 1), float("nan")), provenance=SYNTHETIC_PROVENANCE), TimesNetInputError),
        "inf_input": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(torch.full((2, 10, 1), float("inf")), provenance=SYNTHETIC_PROVENANCE), TimesNetInputError),
        "real_data_provenance": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(fixture, provenance="NASA_REAL"), TimesNetProvenanceError),
        "validation_provenance": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(fixture, provenance="SYNTHETIC_VALIDATION"), TimesNetProvenanceError),
        "test_provenance": expect_failure(lambda: ParameterizedStandardTimesNet(config(1, 1))(fixture, provenance="SYNTHETIC_TEST"), TimesNetProvenanceError),
        "unknown_branch_variant": expect_failure(lambda: ParameterizedStandardTimesNet(TimesNetStructuralConfig(10, 1, 1, 1, 8, 16, 1, 2, 0.0, True, True, "TREND_DEEP_EXACT")), TimesNetConfigError),
    }
    all_shapes = all(item["forward_pass"] and item["backward_pass"] and item["finite_loss"] and item["nonzero_gradient"] and item["finite_gradients"] for item in results)
    all_passed = all_shapes and padding["input_preserved_after_reshape_restore"] and deterministic and overfit["pass"] and all(fail_closed.values()) and framework_adapter_test()
    audit = {
        "artifact_type": "SYNTHETIC_ONLY_PARAMETERIZED_STANDARD_TIMESNET_STRUCTURAL_AUDIT",
        "real_stl_components_used": False,
        "real_battery_data_used": False,
        "validation_values_used": False,
        "test_loaded": False,
        "model_trained": False,
        "model_metrics_used": False,
        "grid_search_run": False,
        "target_paper_topk_inference": False,
        "real_target_alignment_inference": False,
        "trend_real_variant_implemented": False,
        "seasonal_real_variant_implemented": False,
        "synthetic_topk_values_tested": list(SYNTHETIC_TOPK_VALUES),
        "shape_backward_period_results": results,
        "padding": padding,
        "determinism": deterministic,
        "tiny_overfit": overfit,
        "framework_adapter_integration": framework_adapter_test(),
        "fail_closed": fail_closed,
        "all_structural_tests_passed": all_passed,
    }
    destination = PROJECT_ROOT / "output" / "timesnet" / "timesnet_structural_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not all_passed:
        raise SystemExit("Synthetic TimesNet structural tests failed")
    print(destination)


if __name__ == "__main__":
    main()
