"""Pre-registered synthetic-only structural tests for adaptive fusion."""

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

from adaptive_fusion_structural import (  # noqa: E402
    AdaptiveFusionConfigError,
    AdaptiveFusionInputError,
    AdaptiveFusionProvenanceError,
    AdaptiveFusionStructural,
    AdaptiveFusionStructuralConfig,
)
from gmtc_framework_scaffold import BranchPrediction, GMTCFrameworkScaffold  # noqa: E402


FIXTURE_SEED = 20260913
BATCH_SIZE = 4
SYNTHETIC_TIME_STEPS = 4
OVERFIT_MAX_STEPS = 32
OVERFIT_RULE = "final_loss < initial_loss"
SYNTHETIC_PROVENANCE = "SYNTHETIC_PRECONSTRUCTED_FUSION_FEATURE_FIXTURE"


def set_seed(seed: int = FIXTURE_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def config() -> AdaptiveFusionStructuralConfig:
    return AdaptiveFusionStructuralConfig(
        input_dim=9,
        hidden_size=16,
        num_layers=1,
        paper_dropout=0.3,
        leaky_relu_slope=0.1,
        strict_input_dim=True,
        synthetic_only_mode=True,
        component_attention_axis="FEATURE_AXIS_ENGINEERING_FIXTURE_ONLY",
        weight_network_placement="LINEAR_BN_LEAKYRELU_DROPOUT__ENGINEERING_FIXTURE_ONLY",
        final_activation_placement="HIDDEN_RELU_ONLY_ENGINEERING_FIXTURE_ONLY",
    )


def preconstructed_fixture() -> torch.Tensor:
    """Three prediction-like, three error-like, and three ratio-like placeholders."""
    time = torch.arange(SYNTHETIC_TIME_STEPS, dtype=torch.float32).view(1, SYNTHETIC_TIME_STEPS, 1)
    batch = torch.arange(BATCH_SIZE, dtype=torch.float32).view(BATCH_SIZE, 1, 1)
    feature = torch.arange(9, dtype=torch.float32).view(1, 1, 9)
    return 0.1 * batch + 0.02 * time + 0.01 * feature


def synthetic_target(features: torch.Tensor) -> torch.Tensor:
    return (0.25 * features[:, -1, :3].sum(dim=-1, keepdim=True)) + 0.1


def expect_failure(callback, exception_type) -> bool:
    try:
        callback()
    except exception_type:
        return True
    return False


def forward_backward() -> dict[str, object]:
    set_seed()
    model = AdaptiveFusionStructural(config())
    features = preconstructed_fixture()
    output, trace = model.forward_with_trace(features, provenance=SYNTHETIC_PROVENANCE)
    loss = nn.MSELoss()(output, synthetic_target(features))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    return {
        "input_shape": trace["input_shape"],
        "component_attention_shape": trace["component_attention_shape"],
        "bilstm_output_shape": trace["bilstm_output_shape"],
        "temporal_attention_shape": trace["temporal_attention_shape"],
        "fusion_weight_shape": trace["fusion_weight_shape"],
        "weighted_component_shape": trace["weighted_component_shape"],
        "output_shape": trace["output_shape"],
        "component_attention_normalized": abs(trace["component_attention_sum"] - 1.0) < 1e-6,
        "temporal_attention_normalized": all(abs(value - 1.0) < 1e-6 for value in trace["temporal_attention_sums"]),
        "fusion_weight_normalized": all(abs(value - 1.0) < 1e-6 for value in trace["fusion_weight_sums"]),
        "fusion_weights_nonnegative": trace["fusion_weights_nonnegative"],
        "forward_pass": bool(torch.isfinite(output).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "backward_pass": True,
        "nonzero_gradient": any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
        "finite_gradients": all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients),
        "synthetic_wiring_loss": "MSE__NOT_PAPER_LOSS",
    }


def determinism_test() -> bool:
    features = preconstructed_fixture()
    set_seed()
    first = AdaptiveFusionStructural(config()).eval()
    first_output = first(features, provenance=SYNTHETIC_PROVENANCE)
    set_seed()
    second = AdaptiveFusionStructural(config()).eval()
    second_output = second(features, provenance=SYNTHETIC_PROVENANCE)
    return bool(torch.equal(first_output, second_output))


def tiny_overfit_test() -> dict[str, object]:
    set_seed()
    model = AdaptiveFusionStructural(config()).train()
    features = preconstructed_fixture()
    target = synthetic_target(features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    initial_loss = None
    final_loss = None
    for _ in range(OVERFIT_MAX_STEPS):
        optimizer.zero_grad()
        loss = loss_function(model(features, provenance=SYNTHETIC_PROVENANCE), target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return {"max_steps": OVERFIT_MAX_STEPS, "rule": OVERFIT_RULE, "initial_loss": initial_loss, "final_loss": final_loss, "pass": bool(final_loss < initial_loss)}


def scaffold_schema_integration() -> bool:
    """Validate only synthetic lineage before consuming an already-built fixture."""
    common = {
        "seq_len": 10,
        "pred_len": 1,
        "batch_ids": tuple(f"SYNTHETIC_B{index}" for index in range(BATCH_SIZE)),
        "time_alignment": "SYNTHETIC_FIXTURE_ALIGNMENT",
        "forecast_horizon": "SYNTHETIC_FIXTURE_PRED_LEN_1",
        "sample_provenance": "SYNTHETIC_BRANCH_PREDICTION_FIXTURE",
        "partition_provenance": "SYNTHETIC_BRANCH_PARTITION_FIXTURE",
        "model_fidelity_status": "SYNTHETIC_PLACEHOLDER__NOT_REAL_BRANCH_PREDICTION",
        "synthetic_only": True,
    }
    predictions = tuple(
        BranchPrediction(
            branch_name=f"SYNTHETIC_{component}_PREDICTION_PORT",
            component_type=component,
            output_prediction=torch.zeros(BATCH_SIZE, 1, 1),
            **common,
        )
        for component in ("TREND", "SEASONAL", "RESIDUAL")
    )
    payload = GMTCFrameworkScaffold().build_future_fusion_payload(predictions)
    output = AdaptiveFusionStructural(config())(preconstructed_fixture(), provenance=SYNTHETIC_PROVENANCE)
    return bool(not payload["fusion_executed"] and set(payload["branch_predictions"]) == {"TREND", "SEASONAL", "RESIDUAL"} and torch.isfinite(output).all())


def main() -> None:
    set_seed()
    fixture = preconstructed_fixture()
    outcomes = forward_backward()
    deterministic = determinism_test()
    overfit = tiny_overfit_test()
    scaffold_integration = scaffold_schema_integration()
    fail_closed = {
        "wrong_feature_dimension": expect_failure(lambda: AdaptiveFusionStructural(config())(torch.zeros(BATCH_SIZE, SYNTHETIC_TIME_STEPS, 8), provenance=SYNTHETIC_PROVENANCE), AdaptiveFusionInputError),
        "wrong_tensor_rank": expect_failure(lambda: AdaptiveFusionStructural(config())(torch.zeros(BATCH_SIZE, 9), provenance=SYNTHETIC_PROVENANCE), AdaptiveFusionInputError),
        "nan_input": expect_failure(lambda: AdaptiveFusionStructural(config())(torch.full_like(fixture, float("nan")), provenance=SYNTHETIC_PROVENANCE), AdaptiveFusionInputError),
        "inf_input": expect_failure(lambda: AdaptiveFusionStructural(config())(torch.full_like(fixture, float("inf")), provenance=SYNTHETIC_PROVENANCE), AdaptiveFusionInputError),
        "real_data_provenance": expect_failure(lambda: AdaptiveFusionStructural(config())(fixture, provenance="NASA_REAL"), AdaptiveFusionProvenanceError),
        "validation_provenance": expect_failure(lambda: AdaptiveFusionStructural(config())(fixture, provenance="SYNTHETIC_VALIDATION"), AdaptiveFusionProvenanceError),
        "test_provenance": expect_failure(lambda: AdaptiveFusionStructural(config())(fixture, provenance="SYNTHETIC_TEST"), AdaptiveFusionProvenanceError),
        "missing_config": expect_failure(lambda: AdaptiveFusionStructural(None), AdaptiveFusionConfigError),
        "real_mode_configuration": expect_failure(lambda: AdaptiveFusionStructural(AdaptiveFusionStructuralConfig(9, 16, 1, 0.3, 0.1, True, False, "FEATURE_AXIS_ENGINEERING_FIXTURE_ONLY", "LINEAR_BN_LEAKYRELU_DROPOUT__ENGINEERING_FIXTURE_ONLY", "HIDDEN_RELU_ONLY_ENGINEERING_FIXTURE_ONLY")), AdaptiveFusionConfigError),
    }
    passed = (
        outcomes["forward_pass"] and outcomes["backward_pass"] and outcomes["finite_loss"]
        and outcomes["nonzero_gradient"] and outcomes["finite_gradients"]
        and outcomes["component_attention_normalized"] and outcomes["temporal_attention_normalized"]
        and outcomes["fusion_weight_normalized"] and outcomes["fusion_weights_nonnegative"]
        and deterministic and overfit["pass"] and scaffold_integration and all(fail_closed.values())
    )
    audit = {
        "artifact_type": "SYNTHETIC_ONLY_PARAMETERIZED_ADAPTIVE_FUSION_STRUCTURAL_AUDIT",
        "real_battery_data_used": False,
        "real_stl_components_used": False,
        "real_branch_predictions_used": False,
        "real_fusion_feature_builder_implemented": False,
        "validation_values_used": False,
        "test_loaded": False,
        "model_trained": False,
        "model_metrics_used": False,
        "grid_search_run": False,
        "synthetic_feature_fixture": {"shape": list(fixture.shape), "semantics": "THREE_PREDICTION_LIKE__THREE_ERROR_LIKE__THREE_RATIO_LIKE_PLACEHOLDERS"},
        "known_table4_parameters": {"input_dim": 9, "hidden_size": 16, "bidirectional_combined": 32, "num_layers": 1, "paper_dropout": 0.3, "weight_network": [32, 128, 64, 32, 3], "leaky_relu_alpha": 0.1, "final_prediction_network": [3, 16, 1]},
        "library_single_layer_dropout_effect": "PYTORCH_LSTM_INTERNAL_DROPOUT_NOT_APPLIED_WHEN_NUM_LAYERS_1",
        "fixture_only_choices": config().as_dict(),
        "forward_backward": outcomes,
        "determinism": deterministic,
        "tiny_overfit": overfit,
        "framework_scaffold_schema_integration": scaffold_integration,
        "fail_closed": fail_closed,
        "all_structural_tests_passed": passed,
    }
    destination = PROJECT_ROOT / "output" / "fusion" / "adaptive_fusion_structural_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Synthetic adaptive-fusion structural tests failed")
    print(destination)


if __name__ == "__main__":
    main()
