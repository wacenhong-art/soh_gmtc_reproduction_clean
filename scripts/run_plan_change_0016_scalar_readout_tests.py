"""Synthetic-only contract tests for PLAN-CHANGE-0016.

No raw battery data is opened. No model quality comparison is performed.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from itransformer_structural import ITransformerStructuralConfig, ParameterizedITransformerStructural
from tier2_scalar_component_readout import (
    LinearScalarComponentReadout,
    ScalarComponentReadoutConfig,
    ScalarComponentReadoutError,
    strict_scalar_mse,
)
from timesnet_structural import ParameterizedStandardTimesNet, TimesNetStructuralConfig


SEED = 16016
OUT = ROOT / "output" / "protocol" / "plan_change_0016_scalar_readout_audit.json"


def timesnet_core() -> ParameterizedStandardTimesNet:
    return ParameterizedStandardTimesNet(TimesNetStructuralConfig(
        seq_len=10, pred_len=1, n_vars=3, top_k=2, d_model=64, d_ff=128,
        e_layers=1, num_kernels=2, dropout=0.1, strict_paper_mode=True,
        synthetic_only_mode=False, branch_variant="STANDARD_REFERENCE_CORE",
    ))


def itransformer_core() -> ParameterizedITransformerStructural:
    return ParameterizedITransformerStructural(ITransformerStructuralConfig(
        seq_len=10, pred_len=1, n_vars=3, d_model=64, n_heads=4, e_layers=1,
        d_ff=128, dropout=0.1, activation="gelu", strict_paper_mode=True,
        synthetic_only_mode=False,
    ))


def nonzero_finite_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in module.parameters()
    )


def assert_shape_failure(readout: LinearScalarComponentReadout) -> bool:
    try:
        strict_scalar_mse(torch.zeros(2, 1, 1), torch.zeros(2, 1, 3))
    except ScalarComponentReadoutError:
        return True
    return False


def tiny_readout_overfit() -> dict[str, float | bool]:
    """Pre-registered 20-step synthetic wiring check with a frozen core output."""
    torch.manual_seed(SEED + 1)
    core_output = torch.randn(8, 1, 3)
    target = 0.25 * core_output[..., :1] - 0.50 * core_output[..., 1:2] + 0.75 * core_output[..., 2:3] + 0.10
    readout = LinearScalarComponentReadout(ScalarComponentReadoutConfig(3, 1, 1, True, "NONE"))
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
    initial = float(strict_scalar_mse(readout(core_output), target).detach())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = strict_scalar_mse(readout(core_output), target)
        loss.backward()
        optimizer.step()
    final = float(strict_scalar_mse(readout(core_output), target).detach())
    return {"steps": 20, "initial_loss": initial, "final_loss": final, "loss_decreased": final < initial}


def main() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x = torch.linspace(-1.0, 1.0, steps=4 * 10 * 3, dtype=torch.float32).reshape(4, 10, 3)
    target = torch.tensor([[[0.10]], [[-0.20]], [[0.30]], [[-0.40]]], dtype=torch.float32)
    cores = {
        "trend": timesnet_core(),
        "seasonal": timesnet_core(),
        "residual": itransformer_core(),
    }
    readouts = {
        name: LinearScalarComponentReadout(ScalarComponentReadoutConfig(3, 1, 1, True, "NONE"))
        for name in cores
    }
    assert all(sum(parameter.numel() for parameter in module.parameters()) == 4 for module in readouts.values())

    outputs: dict[str, torch.Tensor] = {}
    losses: dict[str, torch.Tensor] = {}
    broadcasting_detected = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, core in cores.items():
            core.train()
            readouts[name].train()
            core_output = core(x, provenance="SYNTHETIC_PLAN_CHANGE_0016")
            outputs[name] = readouts[name](core_output)
            losses[name] = strict_scalar_mse(outputs[name], target)
        total_loss = sum(losses.values())
        total_loss.backward()
        broadcasting_detected = any("broadcast" in str(item.message).lower() for item in caught)

    fusion = outputs["trend"] + outputs["seasonal"] + outputs["residual"]
    result = {
        "audit_scope": "SYNTHETIC_ONLY",
        "seed": SEED,
        "real_hf_values_used": False,
        "real_numerical_target_values_used": False,
        "model_trained": False,
        "test_loaded": False,
        "core_output_shape": [4, 1, 3],
        "trend_output_shape": list(outputs["trend"].shape),
        "seasonal_output_shape": list(outputs["seasonal"].shape),
        "residual_output_shape": list(outputs["residual"].shape),
        "fusion_output_shape": list(fusion.shape),
        "readout_parameter_count_per_branch": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in readouts.items()
        },
        "strict_scalar_shape_equality": all(tuple(value.shape) == tuple(target.shape) for value in outputs.values()),
        "finite_outputs": all(bool(torch.isfinite(value).all()) for value in outputs.values()),
        "finite_losses": all(bool(torch.isfinite(value)) for value in losses.values()),
        "readout_gradients_nonzero_finite": {name: nonzero_finite_gradient(module) for name, module in readouts.items()},
        "core_gradients_nonzero_finite": {name: nonzero_finite_gradient(module) for name, module in cores.items()},
        "broadcasting_detected": broadcasting_detected,
        "mismatched_loss_shape_rejected": assert_shape_failure(readouts["trend"]),
        "tiny_readout_overfit": tiny_readout_overfit(),
    }
    result["gate"] = "PASS" if (
        result["finite_outputs"]
        and result["finite_losses"]
        and all(result["readout_gradients_nonzero_finite"].values())
        and all(result["core_gradients_nonzero_finite"].values())
        and all(count == 4 for count in result["readout_parameter_count_per_branch"].values())
        and result["strict_scalar_shape_equality"]
        and not result["broadcasting_detected"]
        and result["mismatched_loss_shape_rejected"]
        and result["tiny_readout_overfit"]["loss_decreased"]
    ) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SCALAR_COMPONENT_READOUT_GATE={result['gate']}")
    print(f"REPORT={OUT.relative_to(ROOT)}")
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
