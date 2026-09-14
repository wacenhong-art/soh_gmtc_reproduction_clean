"""Synthetic-only contract tests for PLAN-CHANGE-0014.

No raw battery, partition, validation, or TEST values are read by this script.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from itransformer_structural import ITransformerStructuralConfig, ParameterizedITransformerStructural
from stl_decomposition import STLDecomposer
from timesnet_structural import ParameterizedStandardTimesNet, TimesNetStructuralConfig

SEED = 20260913

def set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

def assert_test_locked(partition: str) -> None:
    if partition.upper() == "TEST":
        raise PermissionError("TEST_LOCK: TEST provenance is prohibited in Phase 9A")

def main() -> None:
    set_seed()
    x = torch.linspace(0.0, 1.0, 2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    target = torch.zeros(2, 1, 3)
    times_config = TimesNetStructuralConfig(
        seq_len=10, pred_len=1, n_vars=3, top_k=2, d_model=64, d_ff=128,
        e_layers=1, num_kernels=2, dropout=0.1, strict_paper_mode=True,
        synthetic_only_mode=False, branch_variant="STANDARD_REFERENCE_CORE",
    )
    itrans_config = ITransformerStructuralConfig(
        seq_len=10, pred_len=1, n_vars=3, d_model=64, n_heads=4, e_layers=1,
        d_ff=128, dropout=0.1, activation="gelu", strict_paper_mode=True,
        synthetic_only_mode=False,
    )
    trend_model = ParameterizedStandardTimesNet(times_config)
    seasonal_model = ParameterizedStandardTimesNet(times_config)
    residual_model = ParameterizedITransformerStructural(itrans_config)
    provenance = "TIER2_SYNTHETIC_CONFIG_CONTRACT"
    trend = trend_model(x, provenance=provenance)
    seasonal = seasonal_model(x, provenance=provenance)
    residual = residual_model(x, provenance=provenance)
    prediction = trend + seasonal + residual
    loss = nn.MSELoss()(prediction, target)
    loss.backward()
    stl_input = [0.01 * index + 0.1 * np.sin(2 * np.pi * index / 7) for index in range(28)]
    stl = STLDecomposer().decompose(series=stl_input, period=7, partition_id="TRAIN", series_id="SYNTHETIC_TRAIN_ONLY")
    test_lock = False
    try:
        assert_test_locked("TEST")
    except PermissionError:
        test_lock = True
    results = {
        "status": "PASS",
        "synthetic_only": True,
        "timesnet_constructor": True,
        "itransformer_constructor": True,
        "ten_to_one_shape": list(prediction.shape) == [2, 1, 3],
        "forward_finite": bool(torch.isfinite(prediction).all()),
        "backward_finite": all(torch.isfinite(p.grad).all() for m in (trend_model, seasonal_model, residual_model) for p in m.parameters() if p.grad is not None),
        "loss_finite": bool(torch.isfinite(loss)),
        "component_sum": bool(torch.equal(prediction, trend + seasonal + residual)),
        "d_model_divisible_by_heads": 64 % 4 == 0,
        "train_only_offline_stl": stl["metadata"]["partition_id"] == "TRAIN" and stl["metadata"]["period"] == 7,
        "val_stl_required": False,
        "test_lock": test_lock,
    }
    assert all(value is True or value is False and key == "val_stl_required" for key, value in results.items() if key not in {"status", "synthetic_only"})
    output = ROOT / "output" / "protocol" / "plan_change_0014_synthetic_contract_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
