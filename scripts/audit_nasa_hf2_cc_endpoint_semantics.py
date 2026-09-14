"""TRAIN-charge-only HF2 CC-endpoint semantics audit.

The script accesses no discharge Capacity, no validation charge trajectory, and
no TEST numeric field.  It reports only aggregate ranges, counts, and flags.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from audit_nasa_discharge_centered_pairing import BATTERIES, RAW, audit


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "protocol" / "nasa_hf2_sign_endpoint_audit.json"
EXPECTED = {"B0005": 117, "B0006": 117, "B0007": 117, "B0018": 92}


class AuditFailure(RuntimeError):
    pass


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise AuditFailure(f"{name} empty or non-finite")
    return array


def _cycles(battery: str) -> list[Any]:
    path = next(RAW.rglob(f"{battery}.mat"), None)
    if path is None:
        raise AuditFailure(f"missing official raw file for {battery}")
    root = loadmat(path, squeeze_me=True, struct_as_record=False)[battery]
    cycles = root.cycle
    return list(cycles) if hasattr(cycles, "__len__") else [cycles]


def _sign(values: np.ndarray) -> str:
    positive, negative = int((values > 0).sum()), int((values < 0).sum())
    if positive > negative:
        return "POSITIVE"
    if negative > positive:
        return "NEGATIVE"
    return "TIED_OR_ZERO"


def _range(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "max": float(values.max()), "median": float(np.median(values))}


def _existing_hf2_available(time: np.ndarray, voltage_charge: np.ndarray, current_charge: np.ndarray) -> bool:
    upper = np.flatnonzero(voltage_charge >= 4.2)
    if upper.size == 0:
        return False
    return bool(np.flatnonzero((current_charge[int(upper[0]) :] >= 0) & (current_charge[int(upper[0]) :] <= 0.02)).size)


def charge_summary(cycle: Any) -> dict[str, Any]:
    data = cycle.data
    time = _vector(getattr(data, "Time"), "Time")
    voltage_measured = _vector(getattr(data, "Voltage_measured"), "Voltage_measured")
    voltage_charge = _vector(getattr(data, "Voltage_charge"), "Voltage_charge")
    current_measured = _vector(getattr(data, "Current_measured"), "Current_measured")
    current_charge = _vector(getattr(data, "Current_charge"), "Current_charge")
    if len({time.size, voltage_measured.size, voltage_charge.size, current_measured.size, current_charge.size}) != 1:
        raise AuditFailure("charge field lengths differ")
    vmeasured_cross = np.flatnonzero(voltage_measured >= 4.2)
    vcharge_cross = np.flatnonzero(voltage_charge >= 4.2)
    early = slice(0, max(1, int(math.ceil(time.size * 0.10))))
    cc_abs_early = float(np.median(np.abs(current_charge[early])))
    transition = int(vmeasured_cross[0]) if vmeasured_cross.size else None
    later = np.abs(current_charge[transition:]) if transition is not None else np.asarray([], dtype=float)
    current_decays = bool(later.size and float(np.median(later)) < cc_abs_early)
    return {
        "n_samples": int(time.size),
        "time": {"start": float(time[0]), "end": float(time[-1])},
        "voltage_measured": _range(voltage_measured),
        "voltage_charge": _range(voltage_charge),
        "current_measured": _range(current_measured),
        "current_charge": _range(current_charge),
        "current_measured_dominant_sign": _sign(current_measured),
        "current_charge_dominant_sign": _sign(current_charge),
        "voltage_measured_reaches_4p2": bool(vmeasured_cross.size),
        "voltage_charge_reaches_4p2": bool(vcharge_cross.size),
        "current_charge_early_abs_median": cc_abs_early,
        "current_charge_early_abs_is_approximately_1p5a": bool(1.4 <= cc_abs_early <= 1.6),
        "current_decay_after_vmeasured_transition": current_decays,
        "existing_hf2_condition_available": _existing_hf2_available(time, voltage_charge, current_charge),
    }


def summarize(battery: str) -> dict[str, Any]:
    structure = audit(battery)
    n_train = math.floor(0.70 * structure["n_valid_pairs"])
    if n_train != EXPECTED[battery]:
        raise AuditFailure(f"unexpected frozen TRAIN count for {battery}")
    cycles = _cycles(battery)
    rows = [charge_summary(cycles[int(pair["charge_record_index"])]) for pair in structure["pair_identities"][:n_train]]
    hf2_failure_pair_indices = [index for index, row in enumerate(rows) if not row["existing_hf2_condition_available"]]
    sign_m = {key: sum(row["current_measured_dominant_sign"] == key for row in rows) for key in ("POSITIVE", "NEGATIVE", "TIED_OR_ZERO")}
    sign_c = {key: sum(row["current_charge_dominant_sign"] == key for row in rows) for key in ("POSITIVE", "NEGATIVE", "TIED_OR_ZERO")}
    return {
        "train_charge_count_audited": n_train,
        "current_measured_sign_counts": sign_m,
        "current_charge_sign_counts": sign_c,
        "current_charge_cc_level_1p5a_count": sum(row["current_charge_early_abs_is_approximately_1p5a"] for row in rows),
        "voltage_measured_4p2_endpoint_count": sum(row["voltage_measured_reaches_4p2"] for row in rows),
        "voltage_charge_4p2_endpoint_count": sum(row["voltage_charge_reaches_4p2"] for row in rows),
        "current_decay_after_vmeasured_transition_count": sum(row["current_decay_after_vmeasured_transition"] for row in rows),
        "existing_hf2_failure_count": len(hf2_failure_pair_indices),
        "hf2_failure_pair_indices": hf2_failure_pair_indices,
        "first_hf2_failure_summary": rows[hf2_failure_pair_indices[0]] if hf2_failure_pair_indices else None,
        "first_train_charge_summary": rows[0],
    }


def main() -> None:
    summary = {battery: summarize(battery) for battery in BATTERIES}
    all_vmeasured = all(item["voltage_measured_4p2_endpoint_count"] == item["train_charge_count_audited"] for item in summary.values())
    all_vcharge = all(item["voltage_charge_4p2_endpoint_count"] == item["train_charge_count_audited"] for item in summary.values())
    all_cc = all(item["current_charge_cc_level_1p5a_count"] == item["train_charge_count_audited"] for item in summary.values())
    all_decay = all(item["current_decay_after_vmeasured_transition_count"] == item["train_charge_count_audited"] for item in summary.values())
    all_existing_fail = all(item["existing_hf2_failure_count"] == item["train_charge_count_audited"] for item in summary.values())
    # The physical endpoint gate is the all-cycle terminal-voltage transition.
    # Charger-current statistics remain evidence only: they must not silently
    # redefine a CC-duration endpoint into a CV-cutoff feature.
    gate = "PASS__CC_DURATION_ENDPOINT_IDENTIFIED" if all_vmeasured else "BLOCKED__ENDPOINT_NOT_STABLE"
    output = {
        "audit_scope": "LEGAL_PAIRED_TRAIN_CHARGE_OPERATIONS_ONLY",
        "real_numerical_target_values_used": False,
        "validation_data_used": False,
        "test_loaded": False,
        "test_numerical_value_access_count": 0,
        "batteries": summary,
        "aggregate": {
            "voltage_measured_4p2_endpoint_identifiable_all": all_vmeasured,
            "voltage_charge_4p2_endpoint_identifiable_all": all_vcharge,
            "current_charge_cc_level_1p5a_consistent_all": all_cc,
            "current_decay_after_transition_consistent_all": all_decay,
            "existing_current_extractor_fails_all": all_existing_fail,
            "hf2_semantics_audit_gate": gate,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
