"""Run the existing deterministic CALCE full-row parser and persist a new artifact.

No split labels, target definition, model code, metrics, or raw-file writes occur.
"""
from __future__ import annotations

import json
from pathlib import Path

from prepare_split_independent_data import ROSTER, calce_inventory


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "data_readiness"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batteries, rows = [], []
    for battery in ROSTER["CALCE"]:
        report, canonical = calce_inventory(battery)
        batteries.append(report)
        rows.extend(canonical)
        print(f"COMPLETED_BATTERY={battery}", flush=True)
    payload = {
        "scope": "CALCE_FULL_ROW_DETERMINISTIC_INVENTORY",
        "batteries": batteries,
        "canonical_row_count": len(rows),
        "flags": {
            "TRAIN_SET_ASSIGNED": False,
            "VALIDATION_SET_ASSIGNED": False,
            "TEST_SET_ASSIGNED": False,
            "TARGET_FIRST20_ASSIGNED": False,
            "GAN_TARGET_POPULATION_ASSIGNED": False,
            "FINAL_SOH_TARGET_ASSIGNED": False,
            "PA004_UNCHANGED": True,
        },
        "limitations": [
            "Cycle identity is the source workbook plus its recorded Cycle_Index; identical numeric IDs across workbooks are reported rather than silently merged.",
            "Final SOH target construction remains blocked by PA-001.",
        ],
        "status": "COMPLETE",
    }
    (OUT / "calce_full_row_inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
