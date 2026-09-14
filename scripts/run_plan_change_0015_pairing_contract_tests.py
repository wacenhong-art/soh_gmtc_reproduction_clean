"""Metadata-only contract tests for PLAN-CHANGE-0015.

This script reads the pre-existing structural pairing manifest only.  It never
opens raw NASA MAT files and never accesses Capacity, HF, SOH, validation, or
test values.  The split checks use synthetic pair ordinals.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "NASA_DISCHARGE_CENTERED_PAIRING_MANIFEST.json"
OUT = ROOT / "output" / "protocol" / "plan_change_0015_pairing_contract_audit.json"
EXPECTED = {
    "B0005": {"pairs": 168, "unpaired_charges": 2},
    "B0006": {"pairs": 168, "unpaired_charges": 2},
    "B0007": {"pairs": 168, "unpaired_charges": 2},
    "B0018": {"pairs": 132, "unpaired_charges": 2},
}


def split_ordinals(n_pairs: int) -> tuple[list[int], list[int], list[int]]:
    """Frozen half-open chronological 70/20/10 partition on pair identities."""
    train_end = (7 * n_pairs) // 10
    val_end = (9 * n_pairs) // 10
    return list(range(0, train_end)), list(range(train_end, val_end)), list(range(val_end, n_pairs))


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results: dict[str, object] = {
        "audit_scope": "METADATA_AND_SYNTHETIC_CONTRACT_ONLY",
        "real_hf_values_used": False,
        "real_numerical_target_values_used": False,
        "test_values_loaded": False,
        "pairing_mode": "DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE",
        "batteries": {},
    }
    observed_ids = {row["battery_id"] for row in data["batteries"]}
    assert observed_ids == set(EXPECTED), f"Unexpected battery roster: {observed_ids}"

    for row in data["batteries"]:
        battery_id = row["battery_id"]
        expected = EXPECTED[battery_id]
        pairs = row["pair_identities"]
        charges = [pair["charge_record_index"] for pair in pairs]
        discharges = [pair["discharge_record_index"] for pair in pairs]
        assert row["n_valid_pairs"] == expected["pairs"] == len(pairs)
        assert row["unpaired_charges"] == expected["unpaired_charges"]
        assert len(charges) == len(set(charges)), f"Charge reuse in {battery_id}"
        assert len(discharges) == len(set(discharges)), f"Discharge reuse in {battery_id}"
        assert all(c < d for c, d in zip(charges, discharges)), f"Order violation in {battery_id}"
        assert row["one_to_one"] is True
        assert row["one_to_many_collisions"] == 0
        assert row["many_to_one_collisions"] == 0
        assert row["pair_temporal_order_valid"] is True
        assert row["impedance_between_pair_count"] > 0, f"No impedance-between coverage in {battery_id}"

        train, val, test = split_ordinals(len(pairs))
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        assert max(train) < min(val) < min(test)
        results["batteries"][battery_id] = {
            "valid_pairs": len(pairs),
            "unpaired_charges": row["unpaired_charges"],
            "charge_reuse": False,
            "discharge_reuse": False,
            "temporal_order_valid": True,
            "impedance_between_pair_count": row["impedance_between_pair_count"],
            "synthetic_pair_split": {
                "train_count": len(train),
                "validation_count": len(val),
                "test_membership_count": len(test),
                "disjoint": True,
                "chronological": True,
            },
        }

    results["one_to_one_contract_status"] = "PASS"
    results["split_unit_contract"] = "PASS__VALID_PAIRED_SUPERVISED_CYCLE"
    results["test_lock_contract"] = "PASS__NO_TEST_VALUES_MATERIALIZED"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("PLAN_CHANGE_0015_PAIRING_CONTRACT_TESTS=PASS")
    print(f"REPORT={OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
