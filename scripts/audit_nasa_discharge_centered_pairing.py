"""Metadata-only discharge-centered NASA charge/discharge pairing audit."""
from __future__ import annotations

import json
from pathlib import Path

from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "nasa_pcoe"
BATTERIES = ("B0005", "B0006", "B0007", "B0018")
OUTPUT = ROOT / "NASA_DISCHARGE_CENTERED_PAIRING_MANIFEST.json"


def operation_metadata(battery: str) -> list[dict]:
    path = next(RAW.rglob(f"{battery}.mat"), None)
    if path is None:
        raise FileNotFoundError(f"official raw file not found: {battery}.mat")
    root = loadmat(path, squeeze_me=True, struct_as_record=False)[battery]
    cycles = root.cycle
    if not hasattr(cycles, "__len__"):
        cycles = [cycles]
    rows = []
    for index, cycle in enumerate(cycles):
        fields = set(getattr(cycle.data, "_fieldnames", []))
        rows.append({
            "record_index": index,
            "operation_type": str(cycle.type).strip().lower(),
            "capacity_field_present": "Capacity" in fields,
        })
    return rows


def consecutive_charge_stats(ops: list[dict]) -> tuple[int, int]:
    blocks = dropped = 0
    index = 0
    while index < len(ops):
        if ops[index]["operation_type"] != "charge":
            index += 1
            continue
        end = index
        while end + 1 < len(ops) and ops[end + 1]["operation_type"] == "charge":
            end += 1
        if end > index:
            blocks += 1
            # Only the charge nearest a later discharge is eligible under E.
            dropped += end - index
        index = end + 1
    return blocks, dropped


def candidate_e(ops: list[dict]) -> tuple[list[dict], int, int]:
    """Pair each discharge to its nearest earlier *unconsumed* charge."""
    consumed_charges: set[int] = set()
    pairs: list[dict] = []
    impedance_between = 0
    for discharge in ops:
        if discharge["operation_type"] != "discharge" or not discharge["capacity_field_present"]:
            continue
        earlier = [row for row in ops[:discharge["record_index"]]
                   if row["operation_type"] == "charge" and row["record_index"] not in consumed_charges]
        if not earlier:
            continue
        charge = earlier[-1]
        consumed_charges.add(charge["record_index"])
        between = ops[charge["record_index"] + 1:discharge["record_index"]]
        has_impedance = any(row["operation_type"] == "impedance" for row in between)
        impedance_between += int(has_impedance)
        pairs.append({
            "charge_record_index": charge["record_index"],
            "charge_operation_type": "charge",
            "discharge_record_index": discharge["record_index"],
            "discharge_operation_type": "discharge",
            "impedance_between": has_impedance,
        })
    return pairs, impedance_between, len(consumed_charges)


def audit(battery: str) -> dict:
    ops = operation_metadata(battery)
    charges = {row["record_index"] for row in ops if row["operation_type"] == "charge"}
    discharges = {row["record_index"] for row in ops
                  if row["operation_type"] == "discharge" and row["capacity_field_present"]}
    pairs, impedance_between, _ = candidate_e(ops)
    paired_charges = [row["charge_record_index"] for row in pairs]
    paired_discharges = [row["discharge_record_index"] for row in pairs]
    blocks, dropped = consecutive_charge_stats(ops)
    return {
        "battery_id": battery,
        "n_charge": len(charges),
        "n_discharge": len(discharges),
        "n_valid_pairs": len(pairs),
        "unpaired_charges": len(charges - set(paired_charges)),
        "unpaired_discharges": len(discharges - set(paired_discharges)),
        "one_to_one": len(paired_charges) == len(set(paired_charges)) and len(paired_discharges) == len(set(paired_discharges)),
        "one_to_many_collisions": len(paired_charges) - len(set(paired_charges)),
        "many_to_one_collisions": len(paired_discharges) - len(set(paired_discharges)),
        "ambiguous_pairs": 0,
        "pair_temporal_order_valid": all(row["charge_record_index"] < row["discharge_record_index"] for row in pairs),
        "impedance_between_pair_count": impedance_between,
        "consecutive_charge_block_count": blocks,
        "consecutive_charge_dropped_count": dropped,
        "pair_identities": pairs,
    }


def main() -> None:
    audits = [audit(battery) for battery in BATTERIES]
    output = {
        "audit_type": "STRUCTURAL_METADATA_ONLY",
        "candidate": "E_DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE",
        "real_data_metadata_used": True,
        "real_numerical_target_values_used": False,
        "real_hf_values_used": False,
        "test_numerical_values_used": False,
        "batteries": audits,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
