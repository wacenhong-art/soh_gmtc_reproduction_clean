"""Contract/unit tests for PLAN-CHANGE-0017_REVISED.

The only raw numerical access is to legal TRAIN charge trajectories for the
HF endpoint-coverage audit.  It never reads a discharge Capacity, validation
charge, or TEST charge.  Raw MAT parsing is permitted by the revised human
authorization; post-split TEST numerical record access is still rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from audit_nasa_discharge_centered_pairing import BATTERIES, RAW, audit
from tier2_hf_extraction import HFExtractionError, extract_nasa_hf_from_arrays
from tier2_phase9a_access import PostSplitPairAccessor, TestLockViolation
from tier2_phase9a_config import FROZEN_TIER2_PHASE9A_CONFIG
from tier2_soh_bridge import NASA_RATED_CAPACITY_AH, capacity_to_soh


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output" / "protocol" / "plan_change_0017_contract_tests.json"
EXPECTED_TRAIN = {"B0005": 117, "B0006": 117, "B0007": 117, "B0018": 92}


def expect_failure(callback) -> bool:
    try:
        callback()
    except (HFExtractionError, ValueError):
        return True
    return False


def endpoint_unit_tests() -> dict[str, bool]:
    exact = extract_nasa_hf_from_arrays(
        [0.0, 1.0, 2.0, 3.0], [3.7, 3.8, 4.2, 4.2], [1.5, 1.5, 0.5, 0.02]
    )
    fallback = extract_nasa_hf_from_arrays(
        [0.0, 1.0, 2.0, 3.0], [3.7, 3.8, 4.2, 4.2], [1.5, 1.5, 0.5, 0.1]
    )
    return {
        "exact_voltage_endpoints": (exact.hf1_seconds, exact.hf2_seconds, exact.hf3_seconds) == (2.0, 1.0, 1.0),
        "first_current_leq_threshold_after_vmax": exact.t_imin_seconds == 3.0,
        "final_timestamp_fallback_provenance": fallback.t_imin_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK",
        "final_timestamp_fallback_duration": fallback.hf2_seconds == 1.0,
        "nonmonotonic_time_rejected": expect_failure(lambda: extract_nasa_hf_from_arrays([0.0, 2.0, 1.0], [3.8, 4.0, 4.2], [1.5, 1.0, 0.01])),
        "absent_voltage_endpoint_rejected": expect_failure(lambda: extract_nasa_hf_from_arrays([0.0, 1.0], [3.7, 4.0], [1.5, 0.1])),
    }


class SentinelCycles:
    def __init__(self) -> None:
        self.accessed: list[int] = []
        self.values = ["TRAIN_CHARGE", "TRAIN_DISCHARGE", "TEST_CHARGE", "TEST_DISCHARGE"]

    def __getitem__(self, index: int):
        self.accessed.append(index)
        return self.values[index]


def test_lock_unit_test() -> dict[str, bool | int]:
    cycles = SentinelCycles()
    accessor = PostSplitPairAccessor(cycles, train_end=1, val_end=2)
    train_pair = {"charge_record_index": 0, "discharge_record_index": 1}
    test_pair = {"charge_record_index": 2, "discharge_record_index": 3}
    assert accessor.records_for_pair(0, train_pair) == ("TRAIN_CHARGE", "TRAIN_DISCHARGE")
    before = list(cycles.accessed)
    denied = False
    try:
        accessor.records_for_pair(2, test_pair)
    except TestLockViolation:
        denied = True
    return {
        "train_pair_access_allowed": before == [0, 1],
        "test_pair_rejected_before_dereference": denied and cycles.accessed == before,
        "post_split_test_numerical_materialization_count": accessor.post_split_test_numerical_materialization_count,
    }


def _cycles(battery: str):
    path = next(RAW.rglob(f"{battery}.mat"), None)
    if path is None:
        raise FileNotFoundError(f"missing official raw NASA file: {battery}")
    root = loadmat(path, squeeze_me=True, struct_as_record=False)[battery]
    return list(root.cycle) if hasattr(root.cycle, "__len__") else [root.cycle]


def train_only_hf_coverage() -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for battery in BATTERIES:
        pair_audit = audit(battery)
        n_train = EXPECTED_TRAIN[battery]
        if pair_audit["n_valid_pairs"] < n_train:
            raise AssertionError(f"{battery}: paired TRAIN membership exceeds valid pair count")
        cycles = _cycles(battery)  # Full raw parse is authorized; only following TRAIN charges are dereferenced.
        successes = failures = final_timestamp_fallbacks = 0
        for pair in pair_audit["pair_identities"][:n_train]:
            try:
                result = extract_nasa_hf_from_arrays(
                    getattr(cycles[int(pair["charge_record_index"])].data, "Time"),
                    getattr(cycles[int(pair["charge_record_index"])].data, "Voltage_measured"),
                    getattr(cycles[int(pair["charge_record_index"])].data, "Current_charge"),
                )
            except HFExtractionError:
                failures += 1
                continue
            successes += 1
            final_timestamp_fallbacks += int(result.t_imin_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
        if successes != n_train or failures != 0:
            raise AssertionError(f"{battery}: unresolved TRAIN HF endpoint failure")
        coverage[battery] = {
            "train_charge_count": n_train,
            "hf1_valid_count": successes,
            "hf2_valid_count": successes,
            "hf3_valid_count": successes,
            "failure_count": failures,
            "t_imin_final_timestamp_fallback_count": final_timestamp_fallbacks,
        }
    return coverage


def scale_and_soh_unit_tests() -> dict[str, bool]:
    train = np.asarray([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], dtype=np.float64)
    low, high = train.min(axis=0), train.max(axis=0)
    transformed = (train - low) / (high - low)
    restored = transformed * (high - low) + low
    soh = capacity_to_soh(np.asarray([0.0, 1.0, 2.0]))
    return {
        "minmax_round_trip": bool(np.allclose(restored, train)),
        "soh_formula_capacity_over_2ah": bool(np.allclose(soh, [0.0, 0.5, 1.0])),
        "soh_denominator_is_2ah": NASA_RATED_CAPACITY_AH == 2.0,
    }


def main() -> None:
    FROZEN_TIER2_PHASE9A_CONFIG.validate()
    endpoints = endpoint_unit_tests()
    lock = test_lock_unit_test()
    coverage = train_only_hf_coverage()
    scale_soh = scale_and_soh_unit_tests()
    passed = all(endpoints.values()) and lock["train_pair_access_allowed"] and lock["test_pair_rejected_before_dereference"] and lock["post_split_test_numerical_materialization_count"] == 0 and all(
        item["failure_count"] == 0 and item["hf1_valid_count"] == item["train_charge_count"]
        for item in coverage.values()
    ) and all(scale_soh.values())
    output = {
        "plan_change": "PLAN-CHANGE-0017_REVISED",
        "raw_mat_full_parse_allowed": True,
        "validation_data_used": False,
        "test_loaded": False,
        "test_numerical_materialization_count": 0,
        "model_trained": False,
        "grid_search_run": False,
        "endpoint_unit_tests": endpoints,
        "test_lock_unit_test": lock,
        "train_only_hf_coverage": coverage,
        "scale_and_soh_unit_tests": scale_soh,
        "all_contract_tests_passed": passed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit("PLAN-CHANGE-0017_REVISED contract tests failed")
    print("PLAN_CHANGE_0017_REVISED_CONTRACT_TESTS=PASS")


if __name__ == "__main__":
    main()
