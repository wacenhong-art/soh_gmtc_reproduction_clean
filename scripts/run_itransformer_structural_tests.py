"""Pre-registered synthetic-only structural tests for the iTransformer core."""

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

from itransformer_structural import (  # noqa: E402
    ITransformerConfigError,
    ITransformerInputError,
    ITransformerProvenanceError,
    ITransformerStructuralConfig,
    ParameterizedITransformerStructural,
)


FIXTURE_SEED = 20260912
FIXTURE_BATCH_SIZE = 2
FIXTURE_SEQ_LEN = 10
FIXTURE_PRED_LEN = 1
FIXTURE_N_VARS = (1, 3)
SYNTHETIC_TARGET_ALIGNMENT = "ONE_STEP_SYNTHETIC_PROJECTION_ONLY"
SYNTHETIC_TARGET_ALIGNMENT_IS_REAL_INFERENCE = False
OVERFIT_MAX_STEPS = 64
OVERFIT_LOSS_DECREASE_RULE = "final_loss < initial_loss"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fixture_input(n_vars: int) -> torch.Tensor:
    time = torch.arange(FIXTURE_SEQ_LEN, dtype=torch.float32).view(1, FIXTURE_SEQ_LEN, 1)
    variable = torch.arange(n_vars, dtype=torch.float32).view(1, 1, n_vars)
    batch = torch.arange(FIXTURE_BATCH_SIZE, dtype=torch.float32).view(FIXTURE_BATCH_SIZE, 1, 1)
    return 0.1 * time + 0.2 * variable + 0.05 * batch


def fixture_target(x: torch.Tensor) -> torch.Tensor:
    """Synthetic one-step target for wiring only; not a real forecast convention."""
    return 0.5 * x[:, -1:, :] + 0.1


def explicit_config(n_vars: int) -> ITransformerStructuralConfig:
    return ITransformerStructuralConfig(
        seq_len=FIXTURE_SEQ_LEN,
        pred_len=FIXTURE_PRED_LEN,
        n_vars=n_vars,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
        strict_paper_mode=True,
        synthetic_only_mode=True,
    )


def expect_failure(callable_object, exception_type) -> bool:
    try:
        callable_object()
    except exception_type:
        return True
    return False


def run_shape_and_backward_test(n_vars: int) -> dict[str, object]:
    set_seed(FIXTURE_SEED + n_vars)
    config = explicit_config(n_vars)
    model = ParameterizedITransformerStructural(config)
    x = fixture_input(n_vars)
    target = fixture_target(x)
    output, shapes = model.forward_with_trace(x, provenance="SYNTHETIC_ITRANSFORMER_UNIT_FIXTURE")
    loss = nn.MSELoss()(output, target)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    nonzero_gradient = any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    finite_gradients = all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    expected_shapes = {
        "input": (FIXTURE_BATCH_SIZE, FIXTURE_SEQ_LEN, n_vars),
        "inverted_tokens": (FIXTURE_BATCH_SIZE, n_vars, 10),
        "embedded": (FIXTURE_BATCH_SIZE, n_vars, 8),
        "attention_output": (FIXTURE_BATCH_SIZE, n_vars, 8),
        "output": (FIXTURE_BATCH_SIZE, FIXTURE_PRED_LEN, n_vars),
    }
    return {
        "n_vars": n_vars,
        "config": config.as_dict(),
        "input_shape": list(x.shape),
        "target_shape": list(target.shape),
        "trace_shapes": {key: list(value) for key, value in shapes.items()},
        "expected_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "forward_pass": bool(torch.isfinite(output).all()),
        "output_shape_pass": shapes == expected_shapes,
        "backward_pass": True,
        "finite_loss": bool(torch.isfinite(loss)),
        "nonzero_gradient": nonzero_gradient,
        "finite_gradients": finite_gradients,
    }


def determinism_test() -> bool:
    x = fixture_input(3)
    set_seed(FIXTURE_SEED)
    first = ParameterizedITransformerStructural(explicit_config(3))
    first.eval()
    first_output = first(x, provenance="SYNTHETIC_ITRANSFORMER_DETERMINISM")
    set_seed(FIXTURE_SEED)
    second = ParameterizedITransformerStructural(explicit_config(3))
    second.eval()
    second_output = second(x, provenance="SYNTHETIC_ITRANSFORMER_DETERMINISM")
    return bool(torch.equal(first_output, second_output))


def tiny_overfit_test() -> dict[str, object]:
    set_seed(FIXTURE_SEED)
    model = ParameterizedITransformerStructural(explicit_config(1))
    x = fixture_input(1)
    target = fixture_target(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    initial_loss = None
    final_loss = None
    for _ in range(OVERFIT_MAX_STEPS):
        optimizer.zero_grad()
        prediction = model(x, provenance="SYNTHETIC_ITRANSFORMER_TINY_OVERFIT")
        loss = loss_function(prediction, target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return {
        "max_steps": OVERFIT_MAX_STEPS,
        "loss_decrease_rule": OVERFIT_LOSS_DECREASE_RULE,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "pass": bool(final_loss is not None and initial_loss is not None and final_loss < initial_loss),
        "real_performance_evaluation": False,
    }


def main() -> None:
    shape_results = [run_shape_and_backward_test(n_vars) for n_vars in FIXTURE_N_VARS]
    fail_closed = {
        "strict_seq_len_not_10": expect_failure(
            lambda: ParameterizedITransformerStructural(
                ITransformerStructuralConfig(9, 1, 1, 8, 2, 1, 16, 0.0, "gelu", True, True)
            ),
            ITransformerConfigError,
        ),
        "missing_config": expect_failure(lambda: ParameterizedITransformerStructural(None), ITransformerConfigError),
        "missing_pred_len": expect_failure(
            lambda: ITransformerStructuralConfig(
                seq_len=10,
                n_vars=1,
                d_model=8,
                n_heads=2,
                e_layers=1,
                d_ff=16,
                dropout=0.0,
                activation="gelu",
                strict_paper_mode=True,
                synthetic_only_mode=True,
            ),
            TypeError,
        ),
        "d_model_head_incompatibility": expect_failure(
            lambda: ParameterizedITransformerStructural(
                ITransformerStructuralConfig(10, 1, 1, 7, 2, 1, 16, 0.0, "gelu", True, True)
            ),
            ITransformerConfigError,
        ),
        "wrong_tensor_rank": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                torch.zeros(2, 10), provenance="SYNTHETIC_RANK"
            ),
            ITransformerInputError,
        ),
        "nan_input": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                torch.full((2, 10, 1), float("nan")), provenance="SYNTHETIC_NAN"
            ),
            ITransformerInputError,
        ),
        "inf_input": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                torch.full((2, 10, 1), float("inf")), provenance="SYNTHETIC_INF"
            ),
            ITransformerInputError,
        ),
        "real_data_provenance": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                fixture_input(1), provenance="NASA_TRAIN_REAL"
            ),
            ITransformerProvenanceError,
        ),
        "calce_provenance": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                fixture_input(1), provenance="CALCE_SYNTHETIC_LOOKALIKE"
            ),
            ITransformerProvenanceError,
        ),
        "validation_provenance": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                fixture_input(1), provenance="SYNTHETIC_VALIDATION"
            ),
            ITransformerProvenanceError,
        ),
        "test_provenance": expect_failure(
            lambda: ParameterizedITransformerStructural(explicit_config(1))(
                fixture_input(1), provenance="SYNTHETIC_TEST"
            ),
            ITransformerProvenanceError,
        ),
    }
    deterministic = determinism_test()
    overfit = tiny_overfit_test()
    all_shape = all(
        item["forward_pass"]
        and item["output_shape_pass"]
        and item["backward_pass"]
        and item["finite_loss"]
        and item["nonzero_gradient"]
        and item["finite_gradients"]
        for item in shape_results
    )
    all_passed = all_shape and all(fail_closed.values()) and deterministic and overfit["pass"]
    audit = {
        "artifact_type": "SYNTHETIC_ONLY_PARAMETERIZED_ITRANSFORMER_STRUCTURAL_AUDIT",
        "real_input_semantics_inference": False,
        "real_target_alignment_inference": False,
        "real_residual_data_used": False,
        "real_window_manifest_created": False,
        "validation_values_used": False,
        "test_values_used": False,
        "model_metrics_used": False,
        "grid_search_run": False,
        "custom_graph_module_implemented": False,
        "standard_itransformer_core_implemented": True,
        "synthetic_target_alignment": SYNTHETIC_TARGET_ALIGNMENT,
        "synthetic_target_alignment_is_real_inference": SYNTHETIC_TARGET_ALIGNMENT_IS_REAL_INFERENCE,
        "shape_and_backward_results": shape_results,
        "determinism_test": deterministic,
        "tiny_overfit": overfit,
        "fail_closed_tests": fail_closed,
        "real_data_guard_test": all(
            fail_closed[key]
            for key in ("real_data_provenance", "calce_provenance", "validation_provenance", "test_provenance")
        ),
        "all_structural_tests_passed": all_passed,
    }
    output_path = PROJECT_ROOT / "output" / "itransformer" / "itransformer_structural_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not all_passed:
        raise SystemExit("Synthetic iTransformer structural tests failed")
    print(output_path)


if __name__ == "__main__":
    main()
