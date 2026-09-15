"""Execute the isolated Tier-2 paper-closest NASA 60/20/20 pretest.

This runner deliberately leaves Phase9A/9D development artifacts untouched.
Raw MAT containers may be parsed for pairing metadata, but a post-split guard
rejects every TEST numerical dereference before it reaches any feature, scaler,
STL, window, model, checkpoint, prediction, or metric payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import (
    PERIOD,
    SEQ_LEN,
    Split,
    build_val_samples,
    capacity_to_soh,
    metrics,
    numeric_pairs_for_train_and_val,
    scale_train_val,
)
from run_phase9b_minimal_intelligent_fusion import KINDS
from run_phase9d_augmented_full_gmtc_convergence import (
    _fusion_weights,
    _soh_metrics_percent,
    current_validation_components,
    train_augmented_branches,
    train_fusion,
)
from run_phase9d_augmented_full_gmtc_smoke import (
    BATCH_SIZE,
    LEARNING_RATE,
    RATED_CAPACITY_AH,
    SEED,
    fusion_trajectory_samples,
    set_seed,
    stl_components,
    synthetic_sets,
    trajectory_branch_samples,
    transform_synthetic_hf,
    validation_fusion_samples,
)


ROOT = Path(__file__).resolve().parent.parent
PC0019_METRICS = ROOT / "PHASE9C_FIGURE3_SCFGAN_RECONSTRUCTION_METRICS.json"
SYNTHETIC_ROOT = ROOT / "output" / "phase9c_figure3_reconstruction"
OUTPUT_ROOT = ROOT / "output" / "phase9e_nasa_60_20_20_pretest"
REPORT_PATH = ROOT / "PHASE9E_NASA_60_20_20_PRETEST_REPORT.md"
METRICS_PATH = ROOT / "PHASE9E_NASA_60_20_20_PRETEST_METRICS.json"
SPLIT_MANIFEST_PATH = ROOT / "PHASE9E_NASA_60_20_20_SPLIT_MANIFEST.json"
PROVENANCE_PATH = ROOT / "PHASE9E_NASA_60_20_20_ARTIFACT_PROVENANCE.json"

TARGET_BATTERIES = ("B0007", "B0018")
EXPECTED_PAIRS = {"B0007": 168, "B0018": 132}
EXPECTED_SPLITS = {"B0007": (100, 34, 34), "B0018": (79, 26, 27)}
EXPECTED_REAL_BRANCH_WINDOWS = {"B0007": 90, "B0018": 69}
EXPECTED_SYNTHETIC_ROWS = {"B0007": 33, "B0018": 26}
EXPECTED_SYNTHETIC_BRANCH_WINDOWS = {"B0007": 115, "B0018": 80}
EXPECTED_REAL_FUSION_WINDOWS = {"B0007": 80, "B0018": 59}
EXPECTED_SYNTHETIC_FUSION_WINDOWS = {"B0007": 65, "B0018": 30}
SYNTHETIC_SET_COUNT = 5


class Phase9ERejected(RuntimeError):
    """Fail closed before any Phase9E artifact can be published."""


def split_60_20_20(n: int) -> Split:
    """Frozen Tier-2 floor, half-open implementation convention."""

    if n <= 0:
        raise Phase9ERejected("SPLIT_GATE__non_positive_pair_count")
    train_end = math.floor(0.60 * n)
    val_end = math.floor(0.80 * n)
    result = Split(train_end=train_end, val_end=val_end, total=n)
    if result.train_end <= SEQ_LEN or result.val_end >= n or any(value <= 0 for value in result.counts):
        raise Phase9ERejected("SPLIT_GATE__invalid_floor_half_open_boundaries")
    return result


def validate_phase9e_pairs_and_splits() -> tuple[dict[str, dict[str, Any]], dict[str, Split]]:
    audits: dict[str, dict[str, Any]] = {}
    splits: dict[str, Split] = {}
    for battery in TARGET_BATTERIES:
        result = audit(battery)
        if result["n_valid_pairs"] != EXPECTED_PAIRS[battery] or not result["one_to_one"]:
            raise Phase9ERejected(f"PAIRING_GATE__{battery}_unexpected_pair_count_or_nonunique_mapping")
        if not result["pair_temporal_order_valid"]:
            raise Phase9ERejected(f"PAIRING_GATE__{battery}_temporal_order_invalid")
        split = split_60_20_20(result["n_valid_pairs"])
        if split.counts != EXPECTED_SPLITS[battery]:
            raise Phase9ERejected(f"SPLIT_GATE__{battery}_unexpected_60_20_20_counts")
        audits[battery], splits[battery] = result, split
    return audits, splits


def pc0019_preflight() -> dict[str, Any]:
    """Prove source-only generation before touching Phase9E real values."""

    if not PC0019_METRICS.is_file() or not SYNTHETIC_ROOT.is_dir():
        raise Phase9ERejected("PC0019_PROVENANCE_GATE__missing_persisted_source_or_synthetic_artifact")
    payload = json.loads(PC0019_METRICS.read_text(encoding="utf-8"))
    if payload.get("selection", {}).get("epoch") != 200:
        raise Phase9ERejected("PC0019_PROVENANCE_GATE__selected_epoch_not_200")
    if not payload.get("stage_a", {}).get("contract", {}).get("source_only_provenance_pass", False):
        raise Phase9ERejected("PC0019_PROVENANCE_GATE__source_only_contract_not_passed")
    if payload.get("target_information_used_for_training") or payload.get("target_information_used_for_checkpoint_selection"):
        raise Phase9ERejected("PC0019_PROVENANCE_GATE__target_data_used_for_training_or_selection")
    if payload.get("test_loaded") or payload.get("test_access_count") != 0:
        raise Phase9ERejected("PC0019_PROVENANCE_GATE__test_lock_not_clean")
    target_manifest: dict[str, Any] = {}
    for battery in TARGET_BATTERIES:
        generation = payload.get("target_results", {}).get(battery, {}).get("generation", {})
        condition = generation.get("condition_metadata", {})
        expected = EXPECTED_SYNTHETIC_ROWS[battery]
        if condition.get("target_valid_pair_count_metadata_only") != EXPECTED_PAIRS[battery]:
            raise Phase9ERejected(f"PC0019_PROVENANCE_GATE__{battery}_target_pair_metadata_mismatch")
        if condition.get("first_20_percent_condition_count") != expected:
            raise Phase9ERejected(f"PC0019_PROVENANCE_GATE__{battery}_first20_condition_count_mismatch")
        if condition.get("target_hf_or_soh_materialized") is not False:
            raise Phase9ERejected(f"PC0019_PROVENANCE_GATE__{battery}_target_values_materialized_for_generation")
        if generation.get("shape") != [expected, 4] or len(generation.get("set_paths", [])) != SYNTHETIC_SET_COUNT:
            raise Phase9ERejected(f"PC0019_PROVENANCE_GATE__{battery}_synthetic_shape_or_set_count_mismatch")
        target_manifest[battery] = {
            "target_valid_pair_count_metadata_only": EXPECTED_PAIRS[battery],
            "first_20_percent_condition_count": expected,
            "target_hf_or_soh_materialized_for_generation": False,
            "set_count": SYNTHETIC_SET_COUNT,
            "shape_per_set": [expected, 4],
        }
    return {
        "gate": "PASS",
        "selected_source_epoch": 200,
        "source_only_training_and_checkpoint_selection": True,
        "scfgan_retrained": False,
        "synthetic_regenerated": False,
        "no_old_downstream_scaler_dependency": True,
        "target_generation_conditions": target_manifest,
        "historical_post_generation_target_quality_sections_used_for_execution": False,
    }


def development_history() -> dict[str, Any]:
    """Read existing metrics only to label them as non-comparable history."""

    path = ROOT / "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE_METRICS.json"
    if not path.is_file():
        return {"available": False, "label": "70_20_10_DEVELOPMENT_RESULTS__NOT_COMPARABLE"}
    results = json.loads(path.read_text(encoding="utf-8"))["results_by_battery"]
    return {
        "available": True,
        "label": "70_20_10_DEVELOPMENT_RESULTS__NOT_COMPARABLE_TO_PHASE9E_60_20_20",
        "by_battery": {
            battery: {
                "simple_sum_rmse": item["validation"]["augmented_simple_sum"]["rmse"],
                "intelligent_fusion_rmse": item["validation"]["augmented_intelligent_fusion"]["rmse"],
            }
            for battery, item in results.items()
            if battery in TARGET_BATTERIES
        },
    }


def _assert_expected_windows(battery: str, result: dict[str, Any]) -> None:
    if result["real_train_windows"] != EXPECTED_REAL_BRANCH_WINDOWS[battery]:
        raise Phase9ERejected(f"WINDOW_GATE__{battery}_real_branch_window_count_mismatch")
    if result["total_synthetic_windows"] != EXPECTED_SYNTHETIC_BRANCH_WINDOWS[battery]:
        raise Phase9ERejected(f"WINDOW_GATE__{battery}_synthetic_branch_window_count_mismatch")
    if result["total_augmented_train_windows"] != sum((EXPECTED_REAL_BRANCH_WINDOWS[battery], EXPECTED_SYNTHETIC_BRANCH_WINDOWS[battery])):
        raise Phase9ERejected(f"WINDOW_GATE__{battery}_augmented_branch_window_count_mismatch")
    fusion = result["fusion_training_windows"]
    if fusion["real"] != EXPECTED_REAL_FUSION_WINDOWS[battery]:
        raise Phase9ERejected(f"WINDOW_GATE__{battery}_real_fusion_window_count_mismatch")
    if sum(fusion["synthetic_per_set"]) != EXPECTED_SYNTHETIC_FUSION_WINDOWS[battery]:
        raise Phase9ERejected(f"WINDOW_GATE__{battery}_synthetic_fusion_window_count_mismatch")


def one_battery(battery: str, audit_payload: dict[str, Any], split: Split) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    set_seed()
    features, capacities, _, access = numeric_pairs_for_train_and_val(battery, audit_payload["pair_identities"], split)
    if access["post_split_test_numerical_access_attempts"] or access["post_split_test_numerical_materialization_count"]:
        raise Phase9ERejected(f"TEST_LOCK_GATE__{battery}_test_numerical_access_attempted")
    train_hf, val_hf, scaler = scale_train_val(features, split)
    train_capacity = capacities[:split.train_end]
    val_capacity = capacities[split.train_end:split.val_end]
    real_components = stl_components(train_capacity, f"{battery}_PHASE9E_REAL_TRAIN_CAPACITY")
    real_x, real_targets = trajectory_branch_samples(train_hf, train_capacity, real_components, f"{battery}_PHASE9E_REAL_TRAIN")

    sets, synthetic_manifest = synthetic_sets(battery, EXPECTED_SYNTHETIC_ROWS[battery])
    synthetic_x: list[np.ndarray] = []
    synthetic_targets = {kind: [] for kind in KINDS}
    synthetic_fusion_inputs: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    windows_per_set: list[int] = []
    below_0 = above_1 = 0
    for set_number, vector in enumerate(sets, 1):
        scaled_hf, range_count = transform_synthetic_hf(vector[:, :3], scaler)
        below_0 += range_count["below_0"]
        above_1 += range_count["above_1"]
        synthetic_capacity = vector[:, 3] * RATED_CAPACITY_AH
        synthetic_components = stl_components(synthetic_capacity, f"SYNTHETIC_{battery}_PHASE9E_SET_{set_number:02d}")
        item_x, item_targets = trajectory_branch_samples(
            scaled_hf, synthetic_capacity, synthetic_components, f"SYNTHETIC_{battery}_PHASE9E_SET_{set_number:02d}"
        )
        synthetic_x.append(item_x)
        for kind in KINDS:
            synthetic_targets[kind].append(item_targets[kind])
        windows_per_set.append(int(item_x.shape[0]))
        synthetic_fusion_inputs.append((scaled_hf, synthetic_capacity, synthetic_components))

    augmented_x = np.concatenate([real_x, *synthetic_x], axis=0)
    augmented_targets = {kind: np.concatenate([real_targets[kind], *synthetic_targets[kind]], axis=0) for kind in KINDS}
    val_x, val_capacity_target = build_val_samples(train_hf, val_hf, val_capacity)
    val_component_targets, val_component_audit = current_validation_components(capacities, real_components, split.train_end, battery)
    branches, branch_convergence, branch_states = train_augmented_branches(
        augmented_x, augmented_targets, val_x, val_component_targets, battery
    )

    real_fusion_x, real_fusion_y, _ = fusion_trajectory_samples(
        branches, train_hf, train_capacity, real_components, f"{battery}_PHASE9E_REAL_TRAIN"
    )
    fusion_x, fusion_y = [real_fusion_x], [real_fusion_y]
    synthetic_fusion_audits: list[dict[str, Any]] = []
    for set_number, (scaled_hf, synthetic_capacity, synthetic_components) in enumerate(synthetic_fusion_inputs, 1):
        item_x, item_y, item_audit = fusion_trajectory_samples(
            branches, scaled_hf, synthetic_capacity, synthetic_components, f"SYNTHETIC_{battery}_PHASE9E_SET_{set_number:02d}"
        )
        fusion_x.append(item_x)
        fusion_y.append(item_y)
        synthetic_fusion_audits.append(item_audit)
    augmented_fusion_x, augmented_fusion_y = np.concatenate(fusion_x), np.concatenate(fusion_y)
    history_hf = np.concatenate((train_hf, val_hf), axis=0)
    val_fusion_x, val_fusion_y, val_endpoints, leakage = validation_fusion_samples(
        branches, history_hf, capacities, real_components, split.train_end, battery
    )
    if leakage["current_target_error_used"] or leakage["future_validation_capacity_used"]:
        raise Phase9ERejected(f"FUSION_LEAKAGE_GATE__{battery}_current_target_or_future_validation_capacity_used")
    fusion, fusion_convergence, fusion_state, prediction, simple_sum, persistence = train_fusion(
        augmented_fusion_x, augmented_fusion_y, val_fusion_x, val_fusion_y, capacities, val_endpoints, battery
    )
    target = val_fusion_y.reshape(-1)
    if not np.array_equal(target, val_capacity_target.reshape(-1)):
        raise Phase9ERejected(f"ALIGNMENT_GATE__{battery}_branch_fusion_validation_targets_differ")
    all_numeric = np.concatenate((augmented_x.reshape(-1), augmented_fusion_x.reshape(-1), prediction, simple_sum, persistence, target))
    if not np.isfinite(all_numeric).all():
        raise Phase9ERejected(f"FINITE_GATE__{battery}_nan_or_inf_detected")
    result = {
        "split": {
            "n_total": split.total, "n_train": split.counts[0], "n_val": split.counts[1], "n_test_membership_only": split.counts[2],
            "train_range": [0, split.train_end], "val_range": [split.train_end, split.val_end], "test_membership_range": [split.val_end, split.total],
            "integer_boundary_rule": "FLOOR_HALF_OPEN",
        },
        "normalization": {**scaler, "synthetic_transform": "PHASE9E_REAL_60_PERCENT_TRAIN_SCALER_ONLY__UNCLIPPED"},
        "synthetic_scale_below_0_count": below_0,
        "synthetic_scale_above_1_count": above_1,
        "real_train_windows": int(real_x.shape[0]),
        "synthetic_windows_per_set": windows_per_set,
        "total_synthetic_windows": int(sum(windows_per_set)),
        "total_augmented_train_windows": int(augmented_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "branch_validation_component_target_audit": val_component_audit,
        "branch_convergence": branch_convergence,
        "fusion_training_windows": {
            "real": int(real_fusion_x.shape[0]),
            "synthetic_per_set": [int(item["fusion_windows"]) for item in synthetic_fusion_audits],
            "total_augmented": int(augmented_fusion_x.shape[0]),
        },
        "fusion_convergence": fusion_convergence,
        "leakage_gate": {**leakage, "current_target_error_used": False, "future_validation_capacity_used": False, "test_access_count": 0, "status": "PASS"},
        "validation": {
            "label_domain": "CAPACITY_AH",
            "augmented_simple_sum": metrics(simple_sum, target),
            "augmented_intelligent_fusion": metrics(prediction, target),
            "persistence": metrics(persistence, target),
            "augmented_simple_sum_soh_metrics_percent": _soh_metrics_percent(simple_sum, target),
            "augmented_intelligent_fusion_soh_metrics_percent": _soh_metrics_percent(prediction, target),
        },
        "stl": {"period": PERIOD, "real_train": "OFFLINE_TRAIN_ONLY", "synthetic": "ISOLATED_PER_TRAJECTORY"},
        "access": access,
        "nan_inf_count": 0,
    }
    _assert_expected_windows(battery, result)
    rows = [
        {
            "sample_index": int(index), "target_capacity": float(value), "predicted_capacity": float(pred),
            "simple_sum_capacity": float(simple), "persistence_capacity": float(persist), "error": float(pred - value),
            "weight_trend": float(weight[0]), "weight_seasonal": float(weight[1]), "weight_residual": float(weight[2]),
        }
        for index, value, pred, simple, persist, weight in zip(
            val_endpoints, target, prediction, simple_sum, persistence, _fusion_weights(fusion, val_fusion_x), strict=True
        )
    ]
    return result, {"branch_state_dicts": branch_states, "fusion_state_dict": fusion_state, "synthetic_manifest": synthetic_manifest}, rows


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase9E NASA Paper-Aligned 60/20/20 Pretest Retrain", "",
        "Tier-2 paper-closest numerical pretest. The paper specifies 60/20/20 but not integer rounding; floor/half-open is an implementation convention. This is not a paper-exact full-protocol claim.", "",
    ]
    for battery, result in payload["results_by_battery"].items():
        branch, fusion, val = result["branch_convergence"], result["fusion_convergence"], result["validation"]
        lines += [f"## {battery}", "", f"SPLIT={result['split']['n_train']}/{result['split']['n_val']}/{result['split']['n_test_membership_only']}; REAL/SYNTHETIC/AUGMENTED/VAL_BRANCH_WINDOWS={result['real_train_windows']}/{result['total_synthetic_windows']}/{result['total_augmented_train_windows']}/{result['val_windows']}.", "", "| Branch | Best epoch | Stop epoch | Best real-VAL component MSE |", "| --- | ---: | ---: | ---: |"]
        for kind in KINDS:
            item = branch["by_branch"][kind]
            lines.append(f"| {kind} | {item['best_epoch']} | {item['stop_epoch']} | {item['best_val_component_mse']:.12g} |")
        lines += ["", "| Validation method | RMSE | MAE | R2 |", "| --- | ---: | ---: | ---: |"]
        for name, item in (("Augmented simple sum", val["augmented_simple_sum"]), ("Augmented intelligent fusion", val["augmented_intelligent_fusion"]), ("Persistence", val["persistence"])):
            lines.append(f"| {name} | {item['rmse']:.12g} | {item['mae']:.12g} | {item['r2']:.12g} |")
        lines += ["", f"FUSION_BEST_EPOCH={fusion['best_epoch']}; FUSION_STOP_EPOCH={fusion['stop_epoch']}; BEST_VAL_CAPACITY_MSE={fusion['best_val_capacity_mse']:.12g}.", f"MEAN_WEIGHT_T/S/R={fusion['mean_weights']['trend']:.12g}/{fusion['mean_weights']['seasonal']:.12g}/{fusion['mean_weights']['residual']:.12g}; STD={fusion['std_weights']['trend']:.12g}/{fusion['std_weights']['seasonal']:.12g}/{fusion['std_weights']['residual']:.12g}; WEIGHT_SUM_MAX_ERROR={fusion['weight_sum_max_error']:.12g}.", ""]
    lines += ["## Development-history boundary", "", "The retained 70/20/10 development results have a different split and are recorded only as non-comparable historical context; no cross-split value is interpreted as a performance improvement.", "", "```text", "TEST_LOADED=False", "TEST_ACCESS_COUNT=0", "SCFGAN_RETRAINED=False", "SYNTHETIC_REGENERATED=False", "CURRENT_TARGET_ERROR_USED=False", "FUTURE_VALIDATION_CAPACITY_USED=False", "GRID_SEARCH_RUN=False", "HYPERPARAMETER_TUNING=False", f"PAPER_ALIGNED_PRETEST_TRAINING_GATE={payload['paper_aligned_pretest_training_gate']}", "```", ""]
    return "\n".join(lines)


def main() -> None:
    if any(path.exists() for path in (OUTPUT_ROOT, REPORT_PATH, METRICS_PATH, SPLIT_MANIFEST_PATH, PROVENANCE_PATH)):
        raise Phase9ERejected("OUTPUT_NAMESPACE_GATE__refusing_to_overwrite_existing_phase9e_artifact")
    try:
        provenance = pc0019_preflight()
        audits, splits = validate_phase9e_pairs_and_splits()
        results, checkpoint_payloads, rows_by_battery = {}, {}, {}
        for battery in TARGET_BATTERIES:
            results[battery], checkpoint_payloads[battery], rows_by_battery[battery] = one_battery(battery, audits[battery], splits[battery])
        if any(item["access"]["post_split_test_numerical_access_attempts"] or item["access"]["post_split_test_numerical_materialization_count"] for item in results.values()):
            raise Phase9ERejected("TEST_LOCK_GATE__global_post_split_test_access")
        if any(item["leakage_gate"]["current_target_error_used"] or item["leakage_gate"]["future_validation_capacity_used"] for item in results.values()):
            raise Phase9ERejected("FUSION_LEAKAGE_GATE__global_failure")
        if any(item["nan_inf_count"] != 0 for item in results.values()):
            raise Phase9ERejected("FINITE_GATE__global_nan_inf_count_nonzero")
    except Exception as error:
        label = str(error)
        print(json.dumps({"status": "RESEARCH_DIRECTION_REJECTED", "failed_gate": label}, indent=2))
        raise

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    checkpoint_paths: dict[str, str] = {}
    for battery, artifact in checkpoint_payloads.items():
        path = OUTPUT_ROOT / f"{battery}_PHASE9E_BEST_BRANCH_AND_FUSION_CHECKPOINT.pt"
        torch.save({"phase": "PHASE9E_NASA_60_20_20_PRETEST", "battery": battery, **artifact}, path)
        checkpoint_paths[battery] = str(path.relative_to(ROOT))
    split_manifest = {
        "phase": "PHASE9E_NASA_60_20_20_PRETEST", "split_ratio_paper_aligned": True,
        "integer_boundary_rule": "FLOOR_HALF_OPEN", "integer_boundary_rule_source": "TIER2_IMPLEMENTATION_CONVENTION", "paper_explicit_integer_rounding": "NOT_SPECIFIED",
        "test_membership_only": True,
        "by_battery": {battery: results[battery]["split"] for battery in TARGET_BATTERIES},
    }
    provenance.update({"phase": "PHASE9E_NASA_60_20_20_PRETEST", "old_split_artifact_reused": False, "new_real_train_scaler_fit_scope": "PHASE9E_60_PERCENT_REAL_TRAIN_ONLY", "synthetic_hf_transform": "NEW_PHASE9E_REAL_TRAIN_SCALER_ONLY__UNCLIPPED", "synthetic_trajectory_policy": "FIVE_INDEPENDENT_TRAJECTORIES__NO_CROSS_BOUNDARY_WINDOW", "checkpoint_paths": checkpoint_paths})
    payload = {
        "phase": "PHASE9E_NASA_60_20_20_PRETEST", "authority": "EXPLICIT_HUMAN_APPROVAL__PAPER_CLOSEST_TIER2_PRETEST", "paper_aligned_split": True, "paper_exact_full_protocol": False, "reproduction_tier": "TIER2_PAPER_CLOSEST_NUMERICAL",
        "protocol": {"split": "CHRONOLOGICAL_60_20_20", "integer_boundary_rule": "FLOOR_HALF_OPEN", "integer_boundary_rule_source": "TIER2_IMPLEMENTATION_CONVENTION", "paper_explicit_integer_rounding": "NOT_SPECIFIED", "seq_len": SEQ_LEN, "pred_len": 1, "stl_period": PERIOD, "architecture_frozen": True, "hyperparameter_tuning_allowed": False, "seed": SEED, "branch_optimizer": "Adam", "branch_lr": LEARNING_RATE, "branch_batch_size": BATCH_SIZE, "branch_max_epochs": 150, "branch_patience": 50, "branch_min_delta": 0.0, "fusion_optimizer": "Adam", "fusion_lr": LEARNING_RATE, "fusion_weight_decay": 1e-5, "fusion_batch_size": BATCH_SIZE, "fusion_max_epochs": 150, "fusion_patience": 50, "fusion_min_delta": 0.0},
        "old_split_artifact_reused": False, "pc0019_provenance": provenance, "results_by_battery": results, "checkpoint_paths": checkpoint_paths,
        "development_history": development_history(), "test_loaded": False, "test_access_count": 0, "current_target_error_used": False, "future_validation_capacity_used": False, "scfgan_retrained": False, "synthetic_regenerated": False, "grid_search_run": False, "hyperparameter_tuning": False, "nan_inf_count": 0,
        "paper_aligned_pretest_training_gate": "PASS", "final_checkpoints_frozen": True, "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    SPLIT_MANIFEST_PATH.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "paper_aligned_pretest_training_gate": "PASS", "metrics": METRICS_PATH.name}, indent=2))


if __name__ == "__main__":
    main()
