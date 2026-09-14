"""Metadata-only NASA charge/discharge pairing audit.

This reader inspects operation ordering and field presence only. It never reads
Capacity, HF, voltage, current, or SOH values and emits no numerical trajectory.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "nasa_pcoe"
BATTERIES = ("B0005", "B0006", "B0007", "B0018")
OUTPUT = ROOT / "NASA_OPERATION_SEQUENCE_PAIRING_AUDIT.json"


def _operations(battery: str) -> list[dict]:
    path = next(RAW.rglob(f"{battery}.mat"), None)
    if path is None:
        raise FileNotFoundError(f"official raw file not found: {battery}.mat")
    root = loadmat(path, squeeze_me=True, struct_as_record=False)[battery]
    items = root.cycle
    if not hasattr(items, "__len__"):
        items = [items]
    operations = []
    for index, item in enumerate(items):
        kind = str(item.type).strip().lower()
        fields = set(getattr(item.data, "_fieldnames", []))
        operations.append({
            "record_index": index,
            "operation_type": kind,
            "timestamp_metadata_present": hasattr(item, "time"),
            "capacity_field_present": "Capacity" in fields,
        })
    return operations


def _summarize(name: str, mappings: list[tuple[int, int]], charges: set[int], discharges: set[int]) -> dict:
    charge_counts = Counter(c for c, _ in mappings)
    discharge_counts = Counter(d for _, d in mappings)
    collisions_many_to_one = sum(count - 1 for count in discharge_counts.values() if count > 1)
    collisions_one_to_many = sum(count - 1 for count in charge_counts.values() if count > 1)
    return {
        "candidate": name,
        "n_valid_pairs": len(mappings),
        "unpaired_charges": len(charges - set(charge_counts)),
        "unpaired_discharges": len(discharges - set(discharge_counts)),
        "one_to_one": collisions_many_to_one == 0 and collisions_one_to_many == 0,
        "one_to_many_collisions": collisions_one_to_many,
        "many_to_one_collisions": collisions_many_to_one,
        "order_violations": sum(1 for charge, discharge in mappings if charge >= discharge),
        "ambiguous_pairs": collisions_many_to_one + collisions_one_to_many,
    }


def _candidate_a(ops: list[dict]) -> list[tuple[int, int]]:
    return [(op["record_index"], ops[i + 1]["record_index"])
            for i, op in enumerate(ops[:-1])
            if op["operation_type"] == "charge" and ops[i + 1]["operation_type"] == "discharge"
            and ops[i + 1]["capacity_field_present"]]


def _candidate_b(ops: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for i, op in enumerate(ops):
        if op["operation_type"] != "charge":
            continue
        following = next((later for later in ops[i + 1:]
                          if later["operation_type"] == "discharge" and later["capacity_field_present"]), None)
        if following is not None:
            pairs.append((op["record_index"], following["record_index"]))
    return pairs


def _candidate_c(ops: list[dict]) -> list[tuple[int, int]]:
    return [(ops[i - 1]["record_index"], op["record_index"])
            for i, op in enumerate(ops[1:], start=1)
            if op["operation_type"] == "discharge" and op["capacity_field_present"]
            and ops[i - 1]["operation_type"] == "charge"]


def _audit_battery(battery: str) -> dict:
    ops = _operations(battery)
    charges = {op["record_index"] for op in ops if op["operation_type"] == "charge"}
    discharges = {op["record_index"] for op in ops if op["operation_type"] == "discharge" and op["capacity_field_present"]}
    types = [op["operation_type"] for op in ops]
    transition_counts = Counter(f"{types[i]}->{types[i + 1]}" for i in range(len(types) - 1))
    candidates = {
        "A_IMMEDIATELY_FOLLOWING_DISCHARGE": _summarize("A", _candidate_a(ops), charges, discharges),
        "B_NEAREST_SUBSEQUENT_DISCHARGE": _summarize("B", _candidate_b(ops), charges, discharges),
        "C_IMMEDIATELY_PRECEDING_CHARGE": _summarize("C", _candidate_c(ops), charges, discharges),
        "D_EXPLICIT_RAW_CYCLE_SESSION_ID": {
            "available": False,
            "reason": "No explicit top-level shared charge/discharge cycle or session identifier is present in the NASA operation metadata.",
        },
    }
    a = candidates["A_IMMEDIATELY_FOLLOWING_DISCHARGE"]
    b = candidates["B_NEAREST_SUBSEQUENT_DISCHARGE"]
    return {
        "battery_id": battery,
        "n_records": len(ops),
        "n_charge": len(charges),
        "n_discharge_with_capacity_field": len(discharges),
        "n_impedance": sum(op["operation_type"] == "impedance" for op in ops),
        "charge_capacity_none_expected": all(not op["capacity_field_present"] for op in ops if op["operation_type"] == "charge"),
        "operation_sequence": ops,
        "transition_counts": dict(sorted(transition_counts.items())),
        "candidates": candidates,
        "operation_pattern_stable": len(transition_counts) <= 6,
        "each_charge_unique_next_discharge": b["one_to_one"] and b["n_valid_pairs"] == len(charges),
        "impedance_can_appear_between_pair_members": b["n_valid_pairs"] > a["n_valid_pairs"],
        "pair_temporal_order_valid": all(item["order_violations"] == 0 for item in (a, b, candidates["C_IMMEDIATELY_PRECEDING_CHARGE"])),
        "edge_cases_at_start_or_end": bool(a["unpaired_charges"] or a["unpaired_discharges"]),
    }


def main() -> None:
    audited = [_audit_battery(battery) for battery in BATTERIES]
    output = {
        "audit_type": "STRUCTURAL_METADATA_ONLY",
        "real_data_metadata_used": True,
        "real_numerical_target_values_used": False,
        "real_hf_values_used": False,
        "test_numerical_values_used": False,
        "batteries": audited,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
