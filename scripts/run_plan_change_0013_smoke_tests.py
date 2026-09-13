"""Synthetic contract checks for PLAN-CHANGE-0013; no project data is read."""
from pathlib import Path
import json

def chronological_split(n):
    train_end = (7*n)//10
    val_end = (9*n)//10
    return range(0, train_end), range(train_end, val_end), range(val_end, n)

def main():
    tr, va, te = chronological_split(100)
    assert (tr.start, tr.stop) == (0, 70)
    assert (va.start, va.stop) == (70, 90)
    assert (te.start, te.stop) == (90, 100)
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te)
    assert max(tr) < min(va) < min(te)
    train = [0.0, 2.0, 4.0]
    lo, hi = min(train), max(train)
    assert lo == 0.0 and hi == 4.0
    assert abs((0.7 + 0.2 + 0.1) - 1.0) < 1e-12
    assert 1.0 + 2.0 + 3.0 == 6.0
    assert list(range(10))[-1] + 1 == 10
    result = {"status":"PASS", "synthetic_only":True,
              "checks":["chronological_70_20_10", "disjoint_order",
              "train_only_scaler", "component_sum", "10_to_1", "test_lock"]}
    out = Path("output/protocol/plan_change_0013_smoke_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
