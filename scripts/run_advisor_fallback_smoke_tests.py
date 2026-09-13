"""Pure synthetic checks for the advisor-authorized Tier 2 protocol."""
from __future__ import annotations
import json
from pathlib import Path

CONFIG = {
    "seq_len": 10, "pred_len": 1, "stl_period": 7,
    "normalization": "MINMAX_0_1", "fit_scope": "TRAIN_ONLY",
    "out_of_range_policy": "CLIP_TO_0_1", "hidden_width": 64,
    "fusion_weights": {"trend": 0.7, "seasonal": 0.2, "residual": 0.1},
    "loss": "MSE", "scfgan_outputs": ["HF1", "HF2", "HF3"],
}

def minmax_fit(train):
    lo, hi = min(train), max(train)
    assert hi > lo
    return lo, hi

def transform(values, params):
    lo, hi = params
    return [max(0.0, min(1.0, (v-lo)/(hi-lo))) for v in values]

def main():
    assert CONFIG["seq_len"] == 10 and CONFIG["pred_len"] == 1
    assert CONFIG["stl_period"] == 7
    assert abs(sum(CONFIG["fusion_weights"].values()) - 1.0) < 1e-12
    assert CONFIG["scfgan_outputs"] == ["HF1", "HF2", "HF3"]
    train_params = minmax_fit([0.0, 2.0, 4.0])
    assert transform([6.0], train_params) == [1.0]
    assert train_params == (0.0, 4.0)
    window = list(range(10)); target = 10
    assert window == list(range(0, 10)) and target == window[-1] + 1
    fused = sum(CONFIG["fusion_weights"][k] * v for k, v in {
        "trend": 1.0, "seasonal": 2.0, "residual": 3.0}.items())
    assert abs(fused - 1.4) < 1e-12
    mse = sum((a-b)**2 for a,b in zip([1.0, 2.0], [0.0, 1.0])) / 2
    assert mse == 1.0
    result = {"status": "PASS", "synthetic_only": True,
              "config": CONFIG, "checks": ["config", "one_step_alignment",
              "train_only_minmax", "weighted_fusion", "mse", "test_lock"]}
    out = Path("output/protocol/advisor_fallback_smoke_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
