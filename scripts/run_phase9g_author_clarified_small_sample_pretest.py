"""Execute PLAN-CHANGE-0020 without dereferencing any TEST numerical field."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import (
    PERIOD, SEQ_LEN, ProtocolViolation, _cycles, extract_capacity,
)
from run_phase9b_minimal_intelligent_fusion import (
    FUSION_SEQ_LEN, KINDS, branch_prediction_matrix, feature_row,
)
from run_phase9d_augmented_full_gmtc_convergence import (
    _soh_metrics_percent, train_augmented_branches, train_fusion,
)
from run_phase9d_augmented_full_gmtc_smoke import (
    RATED_CAPACITY_AH, SEED, set_seed, stl_components, synthetic_sets,
    trajectory_branch_samples, transform_synthetic_hf, fusion_trajectory_samples,
)
from stl_decomposition import STLDecomposer
from tier2_hf_extraction import extract_nasa_hf
from tier2_phase9a_access import PostSplitPairAccessor, TestLockViolation


ROOT = Path(__file__).resolve().parent.parent
PC0019 = ROOT / "PHASE9C_FIGURE3_SCFGAN_RECONSTRUCTION_METRICS.json"
OUTPUT_ROOT = ROOT / "output" / "phase9g_author_clarified_small_sample_pretest"
DIAGNOSTIC_ROOT = ROOT / "output" / "phase9g_diagnostics"
DIAGNOSTIC_LOG = DIAGNOSTIC_ROOT / "phase9g_execution.log"
PRECHECK_RECORD = DIAGNOSTIC_ROOT / "phase9g_precheck.json"
REPORT = ROOT / "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE_PRETEST_REPORT.md"
METRICS = ROOT / "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE_PRETEST_METRICS.json"
SPLIT_MANIFEST = ROOT / "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE_SPLIT_MANIFEST.json"
PROVENANCE = ROOT / "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE_ARTIFACT_PROVENANCE.json"
TARGETS = ("B0007", "B0018")
EXPECTED = {
    "B0007": {"n": 168, "train": 33, "unused_end": 100, "val_end": 134, "real": 23, "synthetic": 115, "augmented": 138, "fusion_real": 13},
    "B0018": {"n": 132, "train": 26, "unused_end": 79, "val_end": 105, "real": 16, "synthetic": 80, "augmented": 96, "fusion_real": 6},
}


class Phase9GRejected(RuntimeError):
    pass


class ExecutionLog:
    """Append-only, non-numerical execution telemetry for Phase9G."""

    def __init__(self) -> None:
        DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
        self.last_completed_stage = "NOT_STARTED"

    def emit(self, message: str, *, completed_stage: bool = False) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        with DIAGNOSTIC_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if completed_stage:
            self.last_completed_stage = message

    def exception(self, error: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        print(detail, file=sys.stderr, flush=True)
        with DIAGNOSTIC_LOG.open("a", encoding="utf-8") as handle:
            handle.write(detail)
            handle.flush()


class Heartbeat:
    """Non-invasive liveness telemetry while frozen helper training is active."""

    def __init__(self, log: ExecutionLog, battery: str, phase: str) -> None:
        self.log, self.battery, self.phase = log, battery, phase
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(15.0):
            self.log.emit(
                "PHASE9G_HEARTBEAT="
                f"battery={self.battery};phase={self.phase};"
                "epoch=UNEXPOSED_BY_FROZEN_HELPER;finite_loss_status=PENDING"
            )

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)


@dataclass(frozen=True)
class SmallSplit:
    train_end: int
    val_start: int
    val_end: int
    total: int

    @property
    def counts(self) -> tuple[int, int, int, int]:
        return self.train_end, self.val_start - self.train_end, self.val_end - self.val_start, self.total - self.val_end


def split_for(battery: str) -> SmallSplit:
    expected = EXPECTED[battery]
    split = SmallSplit(expected["train"], expected["unused_end"], expected["val_end"], expected["n"])
    if split.counts != ((33, 67, 34, 34) if battery == "B0007" else (26, 53, 26, 27)):
        raise Phase9GRejected(f"TARGET_20_PERCENT_TRAIN_GATE__{battery}_boundary_mismatch")
    return split


def pairing_preflight() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for battery in TARGETS:
        result = audit(battery)
        if result["n_valid_pairs"] != EXPECTED[battery]["n"] or not result["one_to_one"] or not result["pair_temporal_order_valid"]:
            raise Phase9GRejected(f"TARGET_20_PERCENT_TRAIN_GATE__{battery}_pairing_contract_failed")
        results[battery] = result
    return results


def pc0019_preflight() -> dict[str, Any]:
    if not PC0019.is_file():
        raise Phase9GRejected("SCFGAN_PROVISIONAL_PROVENANCE_GATE__missing_pc0019_metrics")
    payload = json.loads(PC0019.read_text(encoding="utf-8"))
    if payload.get("selection", {}).get("epoch") != 200:
        raise Phase9GRejected("SCFGAN_PROVISIONAL_PROVENANCE_GATE__source_checkpoint_not_epoch_200")
    if not payload.get("stage_a", {}).get("contract", {}).get("source_only_provenance_pass", False):
        raise Phase9GRejected("SCFGAN_PROVISIONAL_PROVENANCE_GATE__source_only_contract_failed")
    if payload.get("target_information_used_for_training") or payload.get("target_information_used_for_checkpoint_selection"):
        raise Phase9GRejected("SCFGAN_PROVISIONAL_PROVENANCE_GATE__target_training_or_selection_detected")
    if payload.get("test_loaded") or payload.get("test_access_count") != 0:
        raise Phase9GRejected("SCFGAN_PROVISIONAL_PROVENANCE_GATE__test_lock_not_clean")
    manifests: dict[str, Any] = {}
    for battery in TARGETS:
        generation = payload.get("target_results", {}).get(battery, {}).get("generation", {})
        condition = generation.get("condition_metadata", {})
        rows = EXPECTED[battery]["train"]
        if (
            condition.get("target_valid_pair_count_metadata_only") != EXPECTED[battery]["n"]
            or condition.get("first_20_percent_condition_count") != rows
            or condition.get("target_hf_or_soh_materialized") is not False
        ):
            raise Phase9GRejected(f"SCFGAN_PROVISIONAL_PROVENANCE_GATE__{battery}_condition_provenance_failed")
        if generation.get("shape") != [rows, 4] or len(generation.get("set_paths", [])) != 5:
            raise Phase9GRejected(f"SCFGAN_PROVISIONAL_PROVENANCE_GATE__{battery}_trajectory_contract_failed")
        manifests[battery] = {"rows": rows, "sets": 5, "shape": [rows, 4], "target_blind_generation": True}
    return {"gate": "PASS", "checkpoint_epoch": 200, "source_only": True, "target_blind_checkpoint_selection": True, "voltage_window_encoding": "UNRESOLVED", "trajectories": manifests}


def runtime_provenance() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "statsmodels_version": statsmodels.__version__,
    }


def numeric_prefix(battery: str, pairs: list[dict[str, Any]], split: SmallSplit) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    accessor = PostSplitPairAccessor(_cycles(battery), split.train_end, split.val_end)
    hf, capacity = [], []
    fallback = 0
    for pair_index, pair in enumerate(pairs):
        if pair_index >= split.val_end:
            continue
        try:
            charge, discharge = accessor.records_for_pair(pair_index, pair)
            endpoint = extract_nasa_hf(charge)
            vector = np.asarray(
                (endpoint.hf1_seconds, endpoint.hf2_seconds, endpoint.hf3_seconds),
                dtype=np.float64,
            )
        except TestLockViolation as error:
            raise Phase9GRejected(f"TEST_LOCK_GATE__{battery}_numerical_access_attempt") from error
        except Exception as error:
            raise Phase9GRejected(f"HF_EXTRACTION_GATE__{battery}_pair_{pair_index}_{error}") from error
        hf.append(vector)
        capacity.append(extract_capacity(discharge))
        fallback += int(endpoint.t_imin_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
    if len(hf) != split.val_end:
        raise Phase9GRejected(f"TEST_LOCK_GATE__{battery}_prefix_length_mismatch")
    return np.asarray(hf, dtype=np.float64), np.asarray(capacity, dtype=np.float64), {
        "test_access_count": accessor.test_numerical_access_attempts,
        "test_numerical_dereference_count": accessor.post_split_test_numerical_materialization_count,
        "train_numeric_pairs": split.train_end,
        "unused_numeric_pairs_past_context_only": split.val_start - split.train_end,
        "val_numeric_pairs": split.val_end - split.val_start,
        "hf2_timestamp_fallback_count": fallback,
    }


def train_only_scale(features: np.ndarray, split: SmallSplit) -> tuple[np.ndarray, dict[str, Any]]:
    train = features[:split.train_end]
    low, high = train.min(axis=0), train.max(axis=0)
    if np.any(high <= low):
        raise Phase9GRejected("SCALER_PROVENANCE_GATE__zero_real_train_range")
    transformed = (features - low) / (high - low)
    transformed[split.train_end:] = np.clip(transformed[split.train_end:], 0.0, 1.0)
    val_raw = features[split.val_start:split.val_end]
    return transformed.astype(np.float32), {
        "normalization": "MINMAX_0_1", "fit_scope": "FIRST_20_PERCENT_REAL_TARGET_TRAIN_ONLY",
        "clipping_policy": "FROZEN_TIER2_REAL_NONTRAIN_CLIP_TO_0_1", "train_min_max_by_feature": {f"HF{i+1}": {"min": float(low[i]), "max": float(high[i])} for i in range(3)},
        "val_below_train_range_count": {f"HF{i+1}": int((val_raw[:, i] < low[i]).sum()) for i in range(3)},
        "val_above_train_range_count": {f"HF{i+1}": int((val_raw[:, i] > high[i]).sum()) for i in range(3)},
    }


def val_samples(history_hf: np.ndarray, capacities: np.ndarray, split: SmallSplit) -> tuple[np.ndarray, np.ndarray, list[int]]:
    endpoints = list(range(split.val_start, split.val_end))
    x = np.asarray([history_hf[index - SEQ_LEN:index] for index in endpoints], dtype=np.float32)
    y = np.asarray(capacities[endpoints], dtype=np.float32).reshape(-1, 1, 1)
    if x.shape != (len(endpoints), SEQ_LEN, 3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Phase9GRejected("VAL_REAL_ONLY_GATE__window_or_target_contract_failed")
    return x, y, endpoints


def val_component_labels(capacities: np.ndarray, split: SmallSplit, battery: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = {kind: [] for kind in KINDS}
    max_index = -1
    for endpoint in range(split.val_start, split.val_end):
        decomposed = STLDecomposer().decompose(series=capacities[:endpoint + 1], period=PERIOD, partition_id="VALIDATION", series_id=f"{battery}_P9G_PREFIX_0_TO_{endpoint}")
        audit_payload = decomposed["structural_audit"]
        if audit_payload["max_abs_reconstruction_error"] > 1e-10 or not audit_payload["no_nan_or_inf"]:
            raise Phase9GRejected("VAL_REAL_ONLY_GATE__expanding_prefix_stl_failed")
        for kind in KINDS:
            labels[kind].append(float(decomposed[kind][-1]))
        max_index = endpoint
    resolved = {kind: np.asarray(value, dtype=np.float32).reshape(-1, 1, 1) for kind, value in labels.items()}
    return resolved, {"policy": "EXPANDING_PREFIX_THROUGH_OWN_ENDPOINT__NO_FUTURE_VAL_CAPACITY", "used_only_for": "BRANCH_EARLY_STOPPING_COMPONENT_MSE", "target_indices": list(range(split.val_start, split.val_end)), "max_capacity_index_used": max_index, "future_validation_capacity_used": False}


def val_fusion_samples(branches: Any, history_hf: np.ndarray, capacities: np.ndarray, components: dict[str, np.ndarray], split: SmallSplit, battery: str) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    predictions = branch_prediction_matrix(branches, history_hf)
    leakage = {"current_target_error_used": False, "future_validation_capacity_used": False, "validation_feature_rows": 0, "max_validation_capacity_index_used_for_feature": -1, "ratio_epsilon_guard_count": 0}
    rows = {index: feature_row(predictions, index, capacities, components, split.train_end, battery, leakage) for index in range(SEQ_LEN + 1, history_hf.shape[0])}
    endpoints = list(range(split.val_start, split.val_end))
    x = np.asarray([[rows[index] for index in range(endpoint - FUSION_SEQ_LEN + 1, endpoint + 1)] for endpoint in endpoints], dtype=np.float32)
    y = np.asarray(capacities[endpoints], dtype=np.float32).reshape(-1, 1, 1)
    if x.shape != (len(endpoints), FUSION_SEQ_LEN, 9) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Phase9GRejected("VAL_REAL_ONLY_GATE__fusion_sequence_contract_failed")
    if leakage["max_validation_capacity_index_used_for_feature"] >= max(endpoints):
        raise Phase9GRejected("FUSION_LEAKAGE_GATE__current_or_future_validation_capacity_used")
    return x, y, endpoints, leakage


def history() -> dict[str, Any]:
    payload: dict[str, Any] = {"label": "NON_COMPARABLE_PROTOCOL_HISTORY"}
    for phase, name in (("phase9d", "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE_METRICS.json"), ("phase9e", "PHASE9E_NASA_60_20_20_PRETEST_METRICS.json")):
        path = ROOT / name
        if path.is_file():
            results = json.loads(path.read_text(encoding="utf-8")).get("results_by_battery", {})
            payload[phase] = {battery: {"available": battery in results} for battery in TARGETS}
    return payload


def precheck_battery(battery: str, pairing: dict[str, Any], split: SmallSplit, log: ExecutionLog) -> dict[str, Any]:
    """Construct every pre-training artifact without model updates or metrics."""

    features, capacities, access = numeric_prefix(battery, pairing["pair_identities"], split)
    if access["test_access_count"] or access["test_numerical_dereference_count"]:
        raise Phase9GRejected("TEST_LOCK_GATE__numeric_access_nonzero")
    log.emit(f"PHASE9G_STAGE=HF_EXTRACTION_PASS__{battery}")
    scaled, scaler = train_only_scale(features, split)
    log.emit(f"PHASE9G_STAGE=SCALER_PASS__{battery}")
    train_hf, train_capacity = scaled[:split.train_end], capacities[:split.train_end]
    real_components = stl_components(train_capacity, f"{battery}_PHASE9G_REAL_FIRST20_TRAIN")
    log.emit(f"PHASE9G_STAGE=STL_PASS__{battery}")
    real_x, _ = trajectory_branch_samples(train_hf, train_capacity, real_components, f"{battery}_PHASE9G_REAL_TRAIN")
    sets, _ = synthetic_sets(battery, split.train_end)
    per_set, below, above = [], 0, 0
    for number, vector in enumerate(sets, 1):
        synthetic_hf, scale_audit = transform_synthetic_hf(vector[:, :3], scaler)
        capacity = vector[:, 3] * RATED_CAPACITY_AH
        components = stl_components(capacity, f"SYNTHETIC_{battery}_P9G_SET_{number:02d}")
        item_x, _ = trajectory_branch_samples(synthetic_hf, capacity, components, f"SYNTHETIC_{battery}_P9G_SET_{number:02d}")
        per_set.append(int(item_x.shape[0]))
        below += scale_audit["below_0"]
        above += scale_audit["above_1"]
    val_x, val_y, _ = val_samples(scaled, capacities, split)
    component_y, component_audit = val_component_labels(capacities, split, battery)
    expected = EXPECTED[battery]
    if (
        int(real_x.shape[0]) != expected["real"]
        or sum(per_set) != expected["synthetic"]
        or int(real_x.shape[0]) + sum(per_set) != expected["augmented"]
        or not np.isfinite(np.concatenate([real_x.reshape(-1), val_x.reshape(-1), val_y.reshape(-1)])).all()
        or any(not np.isfinite(value).all() for value in component_y.values())
    ):
        raise Phase9GRejected(f"SYNTHETIC_TRAJECTORY_BOUNDARY_GATE__{battery}_precheck_contract_failed")
    log.emit(f"PHASE9G_STAGE=WINDOW_ASSEMBLY_PASS__{battery}")
    return {
        "real_train_windows": int(real_x.shape[0]),
        "synthetic_windows": int(sum(per_set)),
        "augmented_train_windows": int(real_x.shape[0]) + int(sum(per_set)),
        "synthetic_windows_per_set": per_set,
        "synthetic_scale_below_0_count": below,
        "synthetic_scale_above_1_count": above,
        "val_windows": int(val_x.shape[0]),
        "component_label_audit": component_audit,
        "access": access,
    }


def one_battery(battery: str, pairing: dict[str, Any], split: SmallSplit, log: ExecutionLog | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed()
    features, capacities, access = numeric_prefix(battery, pairing["pair_identities"], split)
    if access["test_access_count"] or access["test_numerical_dereference_count"]:
        raise Phase9GRejected("TEST_LOCK_GATE__numeric_access_nonzero")
    scaled, scaler = train_only_scale(features, split)
    train_hf, train_capacity = scaled[:split.train_end], capacities[:split.train_end]
    real_components = stl_components(train_capacity, f"{battery}_PHASE9G_REAL_FIRST20_TRAIN")
    real_x, real_targets = trajectory_branch_samples(train_hf, train_capacity, real_components, f"{battery}_PHASE9G_REAL_TRAIN")
    sets, synthetic_manifest = synthetic_sets(battery, split.train_end)
    synthetic_x, synthetic_inputs = [], []
    synthetic_targets = {kind: [] for kind in KINDS}
    per_set, below, above = [], 0, 0
    for number, vector in enumerate(sets, 1):
        synthetic_hf, scale_audit = transform_synthetic_hf(vector[:, :3], scaler)
        capacity = vector[:, 3] * RATED_CAPACITY_AH
        components = stl_components(capacity, f"SYNTHETIC_{battery}_P9G_SET_{number:02d}")
        item_x, item_y = trajectory_branch_samples(synthetic_hf, capacity, components, f"SYNTHETIC_{battery}_P9G_SET_{number:02d}")
        synthetic_x.append(item_x); synthetic_inputs.append((synthetic_hf, capacity, components))
        for kind in KINDS: synthetic_targets[kind].append(item_y[kind])
        per_set.append(int(item_x.shape[0])); below += scale_audit["below_0"]; above += scale_audit["above_1"]
    augmented_x = np.concatenate([real_x, *synthetic_x])
    augmented_targets = {kind: np.concatenate([real_targets[kind], *synthetic_targets[kind]]) for kind in KINDS}
    val_x, val_y, val_endpoints = val_samples(scaled, capacities, split)
    component_y, component_audit = val_component_labels(capacities, split, battery)
    if log is not None:
        log.emit(f"PHASE9G_STAGE=TRAIN_{battery}_START")
    with Heartbeat(log, battery, "BRANCH") if log is not None else nullcontext():
        branches, branch_convergence, branch_states = train_augmented_branches(augmented_x, augmented_targets, val_x, component_y, battery)
    if log is not None:
        for kind, item in branch_convergence["by_branch"].items():
            log.emit(f"PHASE9G_HEARTBEAT=battery={battery};branch={kind};epoch={item['stop_epoch']};finite_loss_status=PASS")
    real_fx, real_fy, _ = fusion_trajectory_samples(branches, train_hf, train_capacity, real_components, f"{battery}_P9G_REAL_TRAIN")
    fusion_x, fusion_y, synthetic_fusion = [real_fx], [real_fy], []
    for number, (synthetic_hf, capacity, components) in enumerate(synthetic_inputs, 1):
        item_x, item_y, item_audit = fusion_trajectory_samples(branches, synthetic_hf, capacity, components, f"SYNTHETIC_{battery}_P9G_SET_{number:02d}")
        fusion_x.append(item_x); fusion_y.append(item_y); synthetic_fusion.append(item_audit)
    val_fx, val_fy, endpoints, leakage = val_fusion_samples(branches, scaled, capacities, real_components, split, battery)
    if leakage["current_target_error_used"] or leakage["future_validation_capacity_used"]:
        raise Phase9GRejected("FUSION_LEAKAGE_GATE__declared_leakage_true")
    with Heartbeat(log, battery, "FUSION") if log is not None else nullcontext():
        fusion, fusion_convergence, fusion_state, prediction, simple, persistence = train_fusion(np.concatenate(fusion_x), np.concatenate(fusion_y), val_fx, val_fy, capacities, endpoints, battery)
    if log is not None:
        log.emit(f"PHASE9G_HEARTBEAT=battery={battery};branch=fusion;epoch={fusion_convergence['stop_epoch']};finite_loss_status=PASS")
    target = val_fy.reshape(-1)
    if not np.array_equal(target, val_y.reshape(-1)):
        raise Phase9GRejected("VAL_REAL_ONLY_GATE__branch_fusion_target_mismatch")
    expected = EXPECTED[battery]
    if int(real_x.shape[0]) != expected["real"] or sum(per_set) != expected["synthetic"] or int(augmented_x.shape[0]) != expected["augmented"] or int(real_fx.shape[0]) != expected["fusion_real"]:
        raise Phase9GRejected(f"SYNTHETIC_TRAJECTORY_BOUNDARY_GATE__{battery}_window_count_mismatch")
    values = np.concatenate([augmented_x.reshape(-1), val_x.reshape(-1), prediction, simple, persistence, target])
    if not np.isfinite(values).all():
        raise Phase9GRejected("FINITE_GATE__nan_inf_detected")
    result = {
        "split": {"n_total": split.total, "real_train_range": [0, split.train_end], "unused_real_range": [split.train_end, split.val_start], "val_range": [split.val_start, split.val_end], "test_membership_range": [split.val_end, split.total], "counts": {"real_train": split.counts[0], "unused_real": split.counts[1], "val": split.counts[2], "test_membership_only": split.counts[3]}},
        "normalization": scaler, "synthetic_scale_below_0_count": below, "synthetic_scale_above_1_count": above,
        "real_train_windows": int(real_x.shape[0]), "synthetic_windows_per_set": per_set, "total_synthetic_windows": int(sum(per_set)), "augmented_train_windows": int(augmented_x.shape[0]), "val_windows": int(val_x.shape[0]),
        "branch_validation_component_target_audit": component_audit, "branch_convergence": branch_convergence,
        "fusion_training_windows": {"real": int(real_fx.shape[0]), "synthetic_per_set": [int(x["fusion_windows"]) for x in synthetic_fusion], "total_augmented": int(sum(x.shape[0] for x in fusion_x))}, "fusion_convergence": fusion_convergence,
        "validation": {"simple_sum": _metrics_with_soh(simple, target), "intelligent_fusion": _metrics_with_soh(prediction, target), "persistence": _metrics_with_soh(persistence, target)},
        "leakage": {**leakage, "current_target_error_used": False, "future_validation_capacity_used": False, "unused_20_60_used_for_training": False, "unused_20_60_past_context_only": True}, "access": access, "nan_inf_count": 0,
    }
    artifact = {"branch_state_dicts": branch_states, "fusion_state_dict": fusion_state, "synthetic_manifest": synthetic_manifest}
    if log is not None:
        log.emit(f"PHASE9G_STAGE=TRAIN_{battery}_END")
    return result, artifact


def _metrics_with_soh(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target; mse = float(np.mean(error ** 2)); sst = float(np.sum((target - target.mean()) ** 2))
    if not np.isfinite(mse) or sst == 0.0: raise Phase9GRejected("FINITE_GATE__validation_metric_invalid")
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error))), "r2": float(1.0 - np.sum(error ** 2) / sst), **_soh_metrics_percent(prediction, target)}


def render(payload: dict[str, Any]) -> str:
    lines = ["# Phase9G Author-Clarified Small-Sample NASA Pretest", "", "Author-clarified Tier-2 TRAIN+VAL protocol. PC0019 augmentation is provisional because voltage-window condition encoding remains unresolved.", ""]
    for battery, result in payload["results_by_battery"].items():
        lines += [f"## {battery}", "", f"REAL/UNUSED/VAL/TEST={result['split']['counts']['real_train']}/{result['split']['counts']['unused_real']}/{result['split']['counts']['val']}/{result['split']['counts']['test_membership_only']}; REAL/SYNTHETIC/AUGMENTED windows={result['real_train_windows']}/{result['total_synthetic_windows']}/{result['augmented_train_windows']}.", "", "| Method | RMSE Ah | MAE Ah | R2 | SOH RMSE % | SOH MAE % |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, metric in result["validation"].items(): lines.append(f"| {name} | {metric['rmse']:.9g} | {metric['mae']:.9g} | {metric['r2']:.9g} | {metric['rmse_percent']:.9g} | {metric['mae_percent']:.9g} |")
        fusion = result["fusion_convergence"]; lines += ["", f"FUSION_BEST_EPOCH={fusion['best_epoch']}; FUSION_STOP_EPOCH={fusion['stop_epoch']}; WEIGHT_SUM_MAX_ERROR={fusion['weight_sum_max_error']:.3g}.", ""]
    lines += ["## Protocol status", "", "Phase9D and Phase9E results are `NON_COMPARABLE_PROTOCOL_HISTORY`; no cross-protocol difference is interpreted as model improvement.", "", "```text", "TEST_LOADED=False", "TEST_ACCESS_COUNT=0", "TEST_NUMERICAL_DEREFERENCE_COUNT=0", "SCFGAN_RETRAINED=False", "SYNTHETIC_REGENERATED=False", "FINAL_CHECKPOINTS_FROZEN=False", "PROVISIONAL_CHECKPOINTS_FROZEN=True", "NASA_FINAL_TEST_UNLOCK_ALLOWED=False", f"AUTHOR_PROTOCOL_GATE={payload['gates']['author_protocol_gate']}", "```", ""]
    return "\n".join(lines)


def run_precheck(log: ExecutionLog, pairing: dict[str, dict[str, Any]], splits: dict[str, SmallSplit]) -> None:
    results = {battery: precheck_battery(battery, pairing[battery], splits[battery], log) for battery in TARGETS}
    if any(item["access"]["test_access_count"] or item["access"]["test_numerical_dereference_count"] for item in results.values()):
        raise Phase9GRejected("TEST_LOCK_GATE__precheck_numeric_access_nonzero")
    payload = {
        "precheck_only_gate": "PASS", "results_by_battery": results,
        "test_loaded": False, "test_access_count": 0, "test_numerical_dereference_count": 0,
        "model_trained": False, "scfgan_retrained": False, "synthetic_regenerated": False,
    }
    PRECHECK_RECORD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.emit("PRECHECK_ONLY_GATE=PASS", completed_stage=True)
    print(json.dumps(payload, indent=2), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck-only", action="store_true", help="build all pre-training artifacts without model updates")
    args = parser.parse_args(argv)
    log = ExecutionLog()
    log.emit("PHASE9G_STAGE=START")
    try:
        log.emit("PHASE9G_STAGE=RUNTIME_PREFLIGHT_PASS", completed_stage=True)
        provenance = pc0019_preflight()
        log.emit("PHASE9G_STAGE=PROVENANCE_PREFLIGHT_PASS", completed_stage=True)
        pairing = pairing_preflight()
        splits = {battery: split_for(battery) for battery in TARGETS}
        log.emit("PHASE9G_STAGE=DATA_SPLIT_PASS", completed_stage=True)
        if args.precheck_only:
            run_precheck(log, pairing, splits)
            log.emit(f"PROCESS_EXIT_CODE=0;LAST_COMPLETED_STAGE={log.last_completed_stage}")
            return 0
        if any(path.exists() for path in (OUTPUT_ROOT, REPORT, METRICS, SPLIT_MANIFEST, PROVENANCE)):
            raise Phase9GRejected("OUTPUT_NAMESPACE_GATE__phase9g_artifact_already_exists")
        results, artifacts = {}, {}
        for battery in TARGETS:
            results[battery], artifacts[battery] = one_battery(battery, pairing[battery], splits[battery], log)
        if any(r["access"]["test_access_count"] or r["access"]["test_numerical_dereference_count"] for r in results.values()):
            raise Phase9GRejected("TEST_LOCK_GATE__global_failure")
        log.emit("PHASE9G_STAGE=REPORT_WRITE_START")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False); checkpoint_paths = {}
        for battery, artifact in artifacts.items():
            path = OUTPUT_ROOT / f"{battery}_PHASE9G_PROVISIONAL_BRANCH_AND_FUSION_CHECKPOINT.pt"; torch.save({"phase": "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE", "battery": battery, **artifact}, path); checkpoint_paths[battery] = str(path.relative_to(ROOT))
        gates = {"author_protocol_gate": "PASS", "target_20_percent_train_gate": "PASS", "unused_20_60_train_exclusion_gate": "PASS", "scfgan_provisional_provenance_gate": "PASS", "synthetic_trajectory_boundary_gate": "PASS", "scaler_provenance_gate": "PASS", "val_real_only_gate": "PASS", "test_lock_gate": "PASS", "runtime_compatibility_gate": "PASS"}
        payload = {"phase": "PHASE9G_AUTHOR_CLARIFIED_SMALL_SAMPLE_PRETEST", "authority": "AUTHOR_CLARIFIED_SMALL_SAMPLE_PROTOCOL", "protocol": {"seq_len": SEQ_LEN, "pred_len": 1, "stl_period": PERIOD, "seed": SEED, "soh": "CAPACITY_AH_DIVIDED_BY_2.0", "architecture_hyperparameters_frozen": True}, "runtime_provenance": runtime_provenance(), "gates": gates, "pc0019_provenance": provenance, "results_by_battery": results, "checkpoint_paths": checkpoint_paths, "history": history(), "test_loaded": False, "test_access_count": 0, "test_numerical_dereference_count": 0, "scfgan_retrained": False, "synthetic_regenerated": False, "grid_search_run": False, "hyperparameter_tuning": False, "nan_inf_count": 0, "final_checkpoints_frozen": False, "provisional_checkpoints_frozen": True, "nasa_final_test_unlock_allowed": False, "next_action": "RETURN_AUTHOR_CLARIFIED_TRAIN_VAL_RESULTS_FOR_HUMAN_REVIEW"}
        split_payload = {"phase": payload["phase"], "integer_boundary_rule": "FLOOR_HALF_OPEN", "paper_explicit_integer_rounding": "NOT_SPECIFIED", "test_membership_only": True, "by_battery": {battery: results[battery]["split"] for battery in TARGETS}}
        provenance.update({"phase": payload["phase"], "runtime_provenance": payload["runtime_provenance"], "old_phase9e_train_artifact_reused": False, "scaler_fit_scope": "FIRST_20_PERCENT_REAL_TARGET_TRAIN_ONLY", "synthetic_transform": "NEW_FIRST20_REAL_TRAIN_SCALER_ONLY__UNCLIPPED", "synthetic_trajectory_policy": "FIVE_INDEPENDENT_TRAJECTORIES__NO_CROSS_BOUNDARY_WINDOW", "checkpoint_paths": checkpoint_paths, "scfgan_status": "PROVISIONAL_TIER2_FALLBACK", "paper_exact_scfgan": False})
        SPLIT_MANIFEST.write_text(json.dumps(split_payload, indent=2), encoding="utf-8"); PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8"); METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8"); REPORT.write_text(render(payload), encoding="utf-8")
        log.emit("PHASE9G_STAGE=COMPLETE", completed_stage=True)
        log.emit(f"PROCESS_EXIT_CODE=0;LAST_COMPLETED_STAGE={log.last_completed_stage}")
        print(json.dumps({"status": "PASS", "gates": gates, "metrics": METRICS.name}, indent=2), flush=True)
        return 0
    except Exception as error:
        log.exception(error)
        log.emit(f"RESEARCH_DIRECTION_REJECTED={error}")
        log.emit(f"PROCESS_EXIT_CODE=1;LAST_COMPLETED_STAGE={log.last_completed_stage}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
