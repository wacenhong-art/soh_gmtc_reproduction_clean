# PHASE9F STATUS:
# ABORTED BEFORE TEST NUMERICAL UNLOCK.
# SUPERSEDED BY AUTHOR-CLARIFIED SMALL-SAMPLE PROTOCOL.
# HISTORICAL EVALUATOR ONLY — DO NOT USE AS CURRENT FINAL TEST PATH.

"""One-shot sequential final TEST evaluation of frozen Phase9E NASA models.

This runner has no optimizer, training mode, checkpoint selection, scaler fit,
or STL-parameter selection path.  It first validates the committed Phase9E
provenance and checkpoints.  Only then does it unlock each TEST pair in order:
the prediction for pair ``i`` is made before that pair's HF/Capacity is read;
the newly observed pair may be used only by subsequent predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import (
    PERIOD,
    SEQ_LEN,
    Branch,
    Split,
    _cycles,
    extract_capacity,
    extract_frozen_hf,
    metrics,
    numeric_pairs_for_train_and_val,
)
from run_phase9b_minimal_intelligent_fusion import (
    EPS,
    FUSION_SEQ_LEN,
    KINDS,
    MinimalIntelligentFusion,
    expanding_previous_components,
)
from run_phase9d_augmented_full_gmtc_convergence import _soh_metrics_percent
from run_phase9d_augmented_full_gmtc_smoke import stl_components


ROOT = Path(__file__).resolve().parent.parent
SOURCE_COMMIT = "3fa002a0dcffcae6205cd83ff8f95312bd47171c"
TARGET_BATTERIES = ("B0007", "B0018")
EXPECTED_SPLITS = {"B0007": (168, 100, 134), "B0018": (132, 79, 105)}
PHASE9E_METRICS = ROOT / "PHASE9E_NASA_60_20_20_PRETEST_METRICS.json"
PHASE9E_SPLIT = ROOT / "PHASE9E_NASA_60_20_20_SPLIT_MANIFEST.json"
PHASE9E_PROVENANCE = ROOT / "PHASE9E_NASA_60_20_20_ARTIFACT_PROVENANCE.json"
CHECKPOINT_ROOT = ROOT / "output" / "phase9e_nasa_60_20_20_pretest"
OUTPUT_ROOT = ROOT / "output" / "phase9f_nasa_final_test"
REPORT_PATH = ROOT / "PHASE9F_NASA_FINAL_TEST_REPORT.md"
METRICS_PATH = ROOT / "PHASE9F_NASA_FINAL_TEST_METRICS.json"
PREDICTIONS_PATH = ROOT / "PHASE9F_NASA_FINAL_TEST_PREDICTIONS.json"
AUDIT_PATH = ROOT / "PHASE9F_NASA_FINAL_TEST_AUDIT.json"
MANIFEST_PATH = ROOT / "PHASE9F_NASA_FINAL_FROZEN_MODEL_MANIFEST.json"


class Phase9FRejected(RuntimeError):
    """Raised for an immutable preflight, leakage, or finite-value failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Stable value fingerprint for a frozen state dict, independent of pickling."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _assert_no_existing_outputs() -> None:
    paths = (OUTPUT_ROOT, REPORT_PATH, METRICS_PATH, PREDICTIONS_PATH, AUDIT_PATH, MANIFEST_PATH)
    if any(path.exists() for path in paths):
        raise Phase9FRejected("OUTPUT_NAMESPACE_GATE__refusing_to_overwrite_final_test_artifact")


def _load_json(path: Path, gate: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase9FRejected(f"{gate}__missing_{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify all immutable dependencies before *any* TEST numerical dereference."""

    _assert_no_existing_outputs()
    if _git_output("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise Phase9FRejected("SOURCE_COMMIT_GATE__HEAD_is_not_frozen_phase9e_commit")
    # The tracked source tree was clean before this evaluator was introduced;
    # require that no pre-existing tracked source modification is present now.
    if _git_output("diff", "--name-only"):
        raise Phase9FRejected("WORKTREE_GATE__tracked_source_tree_modified_before_final_test")

    phase9e = _load_json(PHASE9E_METRICS, "PHASE9E_METRICS_GATE")
    split_manifest = _load_json(PHASE9E_SPLIT, "PHASE9E_SPLIT_GATE")
    provenance = _load_json(PHASE9E_PROVENANCE, "PHASE9E_PROVENANCE_GATE")
    if phase9e.get("paper_aligned_pretest_training_gate") != "PASS":
        raise Phase9FRejected("PHASE9E_GATE__paper_aligned_pretest_not_pass")
    if phase9e.get("final_checkpoints_frozen") is not True:
        raise Phase9FRejected("CHECKPOINT_FREEZE_GATE__phase9e_final_checkpoints_not_frozen")
    if phase9e.get("paper_exact_full_protocol") is not False:
        raise Phase9FRejected("TIER_GATE__paper_exact_full_protocol_must_remain_false")
    if phase9e.get("test_loaded") or phase9e.get("test_access_count") != 0:
        raise Phase9FRejected("PHASE9E_TEST_LOCK_GATE__pretest_lock_not_clean")
    if provenance.get("gate") != "PASS" or provenance.get("selected_source_epoch") != 200:
        raise Phase9FRejected("PC0019_PROVENANCE_GATE__source_only_epoch200_provenance_not_pass")
    if not provenance.get("source_only_training_and_checkpoint_selection"):
        raise Phase9FRejected("PC0019_PROVENANCE_GATE__source_only_training_not_confirmed")
    if provenance.get("historical_post_generation_target_quality_sections_used_for_execution"):
        raise Phase9FRejected("PC0019_PROVENANCE_GATE__target_quality_sections_entered_execution")

    checkpoint_details: dict[str, dict[str, Any]] = {}
    for battery in TARGET_BATTERIES:
        expected_total, expected_train_end, expected_val_end = EXPECTED_SPLITS[battery]
        item = split_manifest.get("by_battery", {}).get(battery, {})
        if (item.get("n_total"), item.get("train_range"), item.get("val_range"), item.get("test_membership_range")) != (
            expected_total, [0, expected_train_end], [expected_train_end, expected_val_end], [expected_val_end, expected_total]
        ):
            raise Phase9FRejected(f"SPLIT_MANIFEST_GATE__{battery}_frozen_boundaries_mismatch")
        checkpoint = CHECKPOINT_ROOT / f"{battery}_PHASE9E_BEST_BRANCH_AND_FUSION_CHECKPOINT.pt"
        if not checkpoint.is_file():
            raise Phase9FRejected(f"CHECKPOINT_PRESENCE_GATE__{battery}_missing_phase9e_checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("phase") != "PHASE9E_NASA_60_20_20_PRETEST" or payload.get("battery") != battery:
            raise Phase9FRejected(f"CHECKPOINT_PROVENANCE_GATE__{battery}_phase_or_battery_mismatch")
        if set(payload.get("branch_state_dicts", {})) != set(KINDS) or not payload.get("fusion_state_dict"):
            raise Phase9FRejected(f"CHECKPOINT_CONTENT_GATE__{battery}_missing_branch_or_fusion_state")
        checkpoint_details[battery] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "phase": payload["phase"],
            "battery": payload["battery"],
            "branches": list(KINDS),
            "fusion": "PHASE9E_FROZEN_MINIMAL_INTELLIGENT_FUSION",
        }
    return phase9e, split_manifest, provenance, checkpoint_details


def restore_models(battery: str) -> tuple[dict[str, Branch], MinimalIntelligentFusion, dict[str, str]]:
    """Construct containers, immediately load strict frozen states, then freeze them."""

    path = CHECKPOINT_ROOT / f"{battery}_PHASE9E_BEST_BRANCH_AND_FUSION_CHECKPOINT.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    branches = {kind: Branch(kind) for kind in KINDS}
    for kind, branch in branches.items():
        branch.load_state_dict(payload["branch_state_dicts"][kind], strict=True)
        branch.eval()
        for parameter in branch.parameters():
            parameter.requires_grad_(False)
        if branch.training or any(parameter.requires_grad for parameter in branch.parameters()):
            raise Phase9FRejected(f"FROZEN_MODEL_GATE__{battery}_{kind}_not_eval_or_not_frozen")
    fusion = MinimalIntelligentFusion()
    fusion.load_state_dict(payload["fusion_state_dict"], strict=True)
    fusion.eval()
    for parameter in fusion.parameters():
        parameter.requires_grad_(False)
    if fusion.training or any(parameter.requires_grad for parameter in fusion.parameters()):
        raise Phase9FRejected(f"FROZEN_MODEL_GATE__{battery}_fusion_not_eval_or_not_frozen")
    fingerprints = {f"branch_{kind}": state_fingerprint(branch.state_dict()) for kind, branch in branches.items()}
    fingerprints["fusion"] = state_fingerprint(fusion.state_dict())
    return branches, fusion, fingerprints


def _scaler_from_phase9e(phase9e: dict[str, Any], battery: str) -> dict[str, Any]:
    scaler = phase9e["results_by_battery"][battery]["normalization"]
    if scaler.get("normalization") != "MINMAX_0_1" or scaler.get("fit_scope") != "TRAIN_ONLY":
        raise Phase9FRejected(f"SCALER_PROVENANCE_GATE__{battery}_not_phase9e_train_only_minmax")
    return scaler


def scale_one_frozen(feature: np.ndarray, scaler: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    ranges = scaler["train_min_max_by_feature"]
    low = np.asarray([ranges[f"HF{i}"]["min"] for i in range(1, 4)], dtype=np.float64)
    high = np.asarray([ranges[f"HF{i}"]["max"] for i in range(1, 4)], dtype=np.float64)
    if np.any(high <= low):
        raise Phase9FRejected("SCALER_GATE__persisted_train_range_nonpositive")
    transformed = (feature - low) / (high - low)
    if transformed.shape != (3,) or not np.isfinite(transformed).all():
        raise Phase9FRejected("SCALER_GATE__test_transform_nonfinite")
    return np.clip(transformed, 0.0, 1.0).astype(np.float64), {
        "below_0": int((transformed < 0.0).sum()), "above_1": int((transformed > 1.0).sum())
    }


def scale_pretest_with_frozen_scaler(physical_hf: np.ndarray, train_end: int, scaler: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Transform prior TRAIN/VAL HF through the persisted Phase9E scaler only.

    This function deliberately does not compute, fit, or replace min/max values.
    """

    if physical_hf.ndim != 2 or physical_hf.shape[1] != 3 or physical_hf.shape[0] <= train_end:
        raise Phase9FRejected("SCALER_GATE__invalid_pretest_hf_shape")
    rows = [scale_one_frozen(row, scaler)[0] for row in physical_hf]
    scaled = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(scaled).all():
        raise Phase9FRejected("SCALER_GATE__nonfinite_persisted_scaler_transform")
    return scaled[:train_end], scaled[train_end:]


def branch_prediction_at(branches: dict[str, Branch], history_hf: list[np.ndarray], target_index: int) -> np.ndarray:
    """Return p(target_index) from exactly the prior ten HF vectors."""

    if target_index < SEQ_LEN or len(history_hf) != target_index:
        raise Phase9FRejected("TEST_WINDOW_GATE__history_does_not_end_before_target")
    window = np.asarray(history_hf[target_index - SEQ_LEN : target_index], dtype=np.float32)
    if window.shape != (SEQ_LEN, 3) or not np.isfinite(window).all():
        raise Phase9FRejected("TEST_WINDOW_GATE__invalid_or_nonfinite_prior_hf_window")
    x = torch.from_numpy(window[None, ...])
    with torch.no_grad():
        result = np.asarray([branches[kind](x, partition="PHASE9F_FINAL_TEST_FROZEN").item() for kind in KINDS], dtype=np.float64)
    if not np.isfinite(result).all():
        raise Phase9FRejected("BRANCH_TEST_FORWARD_GATE__nonfinite_frozen_prediction")
    return result


def sequential_feature_row(
    prediction_by_index: dict[int, np.ndarray],
    capacities_prefix: list[float],
    train_components: dict[str, np.ndarray],
    train_end: int,
    battery: str,
    current_index: int,
    audit: dict[str, Any],
) -> np.ndarray:
    """Build the exact Phase9E 9-feature row without current/future target use."""

    previous_index = current_index - 1
    if current_index < SEQ_LEN + 1 or previous_index >= len(capacities_prefix):
        raise Phase9FRejected("FUSION_TEST_LEAKAGE_GATE__current_target_or_missing_prior_capacity")
    current_prediction = prediction_by_index[current_index]
    previous_prediction = prediction_by_index.get(previous_index)
    if previous_prediction is None or np.any(previous_prediction == 0.0):
        raise Phase9FRejected("FUSION_TEST_GATE__prior_prediction_unavailable_or_zero_for_lagged_error")
    # The prefix ends at previous_index; no current/future TEST capacity exists here.
    prefix = np.asarray(capacities_prefix, dtype=np.float64)
    previous_true_component = expanding_previous_components(prefix, previous_index, train_components, train_end, battery)
    if current_index >= train_end:
        audit["prior_test_or_validation_capacity_used_only"] += int(previous_index >= audit["test_start"])
        audit["max_capacity_index_used_for_current_feature"] = max(audit["max_capacity_index_used_for_current_feature"], previous_index)
    lagged_error = previous_true_component / previous_prediction
    denominator = float(current_prediction.sum())
    if abs(denominator) < EPS:
        audit["ratio_epsilon_guard_count"] += 1
        denominator = EPS if denominator >= 0.0 else -EPS
    ratios = current_prediction / denominator
    row = np.concatenate((current_prediction, lagged_error, ratios)).astype(np.float32)
    if row.shape != (9,) or not np.isfinite(row).all():
        raise Phase9FRejected("FUSION_TEST_FEATURE_GATE__malformed_or_nonfinite_row")
    return row


def test_pair_after_prediction(cycles: list[Any], pair: dict[str, Any], pair_index: int, test_start: int) -> tuple[np.ndarray, float, str]:
    """The only TEST numerical dereference: executed after current prediction."""

    if pair_index < test_start:
        raise Phase9FRejected("TEST_ACCESS_GATE__attempted_pretest_pair_through_test_accessor")
    charge = cycles[int(pair["charge_record_index"])]
    discharge = cycles[int(pair["discharge_record_index"])]
    feature, endpoint_provenance = extract_frozen_hf(charge)
    capacity = extract_capacity(discharge)
    if feature.shape != (3,) or not np.isfinite(feature).all() or not math.isfinite(capacity):
        raise Phase9FRejected("TEST_ACCESS_GATE__nonfinite_hf_or_capacity")
    return feature, capacity, endpoint_provenance


def paper_reference(battery: str, fusion_soh: dict[str, float], fusion_capacity: dict[str, float]) -> dict[str, Any]:
    """Only Table-6 B0018 has a direct developed-method NASA row."""

    if battery != "B0018":
        return {"status": "NO_DIRECT_PAPER_REFERENCE"}
    reference = {"mae_percent": 0.8946, "mape_percent": 0.9012, "rmse_percent": 0.9123, "r2": 0.8765, "source": "PAPER_REPRODUCTION_SPEC.md__PDF_p14_Table6"}
    return {
        "status": "DIRECT_PAPER_REFERENCE_AVAILABLE__NOT_PAPER_EXACT_COMPARISON",
        "paper_reported_developed_method": reference,
        "our_tier2_paper_closest_intelligent_fusion": {
            "mae_percent": fusion_soh["mae_percent"], "rmse_percent": fusion_soh["rmse_percent"], "r2": fusion_capacity["r2"],
        },
        "absolute_difference": {
            "mae_percent": abs(fusion_soh["mae_percent"] - reference["mae_percent"]),
            "rmse_percent": abs(fusion_soh["rmse_percent"] - reference["rmse_percent"]),
            "r2": abs(fusion_capacity["r2"] - reference["r2"]),
        },
    }


def benefit_status(candidate: float, reference: float, label: str) -> str:
    if np.isclose(candidate, reference, rtol=0.0, atol=1e-12):
        return f"EQUAL_BY_TEST_RMSE__{label}"
    return ("BETTER" if candidate < reference else "WORSE") + f"_BY_TEST_RMSE__{label}"


def one_battery(battery: str, phase9e: dict[str, Any], split_manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    total, train_end, test_start = EXPECTED_SPLITS[battery]
    split = Split(train_end=train_end, val_end=test_start, total=total)
    metadata = audit(battery)
    if metadata["n_valid_pairs"] != total or not metadata["one_to_one"] or not metadata["pair_temporal_order_valid"]:
        raise Phase9FRejected(f"PAIRING_GATE__{battery}_frozen_pairing_contract_changed")
    frozen = split_manifest["by_battery"][battery]
    if frozen["n_total"] != total or frozen["test_membership_range"] != [test_start, total]:
        raise Phase9FRejected(f"SPLIT_GATE__{battery}_frozen_test_membership_changed")

    # This pre-TEST call is the Phase9E read scope only: pairs [0, test_start).
    physical_hf, capacities, _, pretest_access = numeric_pairs_for_train_and_val(battery, metadata["pair_identities"], split)
    if pretest_access["post_split_test_numerical_access_attempts"] or pretest_access["post_split_test_numerical_materialization_count"]:
        raise Phase9FRejected(f"PRETEST_LOCK_GATE__{battery}_unexpected_test_access_before_unlock")
    persisted_scaler = _scaler_from_phase9e(phase9e, battery)
    train_hf, val_hf = scale_pretest_with_frozen_scaler(physical_hf, train_end, persisted_scaler)
    train_components = stl_components(np.asarray(capacities[:train_end], dtype=np.float64), f"{battery}_PHASE9E_FROZEN_REAL_TRAIN_CAPACITY")
    branches, fusion, pre_hashes = restore_models(battery)

    history_hf = [np.asarray(row, dtype=np.float64) for row in np.concatenate((train_hf, val_hf), axis=0)]
    history_capacity = [float(value) for value in capacities]
    if len(history_hf) != test_start or len(history_capacity) != test_start:
        raise Phase9FRejected(f"PRETEST_HISTORY_GATE__{battery}_train_val_history_length_mismatch")

    audit_row = {
        "test_start": test_start,
        "current_test_target_used": False,
        "future_test_capacity_used": False,
        "future_test_hf_used": False,
        "prior_test_or_validation_capacity_used_only": 0,
        "max_capacity_index_used_for_current_feature": -1,
        "ratio_epsilon_guard_count": 0,
        "test_pair_numerical_access_count": 0,
        "test_hf_transform_count": 0,
        "test_target_metric_count": 0,
    }
    prediction_by_index: dict[int, np.ndarray] = {}
    row_by_index: dict[int, np.ndarray] = {}
    warm_start = max(SEQ_LEN + 1, test_start - FUSION_SEQ_LEN + 1)
    # The first retained feature row needs one immediately preceding prediction
    # for its lagged-error denominator, but not an earlier feature row.
    prediction_by_index[warm_start - 1] = branch_prediction_at(branches, history_hf, warm_start - 1)
    for index in range(warm_start, test_start):
        prediction_by_index[index] = branch_prediction_at(branches, history_hf, index)
        row_by_index[index] = sequential_feature_row(prediction_by_index, history_capacity, train_components, train_end, battery, index, audit_row)

    cycles = _cycles(battery)  # Raw-container parse is permitted; field dereference occurs only below.
    rows: list[dict[str, Any]] = []
    test_scale_below = {f"HF{i}": 0 for i in range(1, 4)}
    test_scale_above = {f"HF{i}": 0 for i in range(1, 4)}
    endpoint_fallback_count = 0
    for target_index in range(test_start, total):
        # All inputs are from indices < target_index.  The current pair remains sealed here.
        prediction_by_index[target_index] = branch_prediction_at(branches, history_hf, target_index)
        row_by_index[target_index] = sequential_feature_row(prediction_by_index, history_capacity, train_components, train_end, battery, target_index, audit_row)
        feature_sequence = np.asarray([row_by_index[index] for index in range(target_index - FUSION_SEQ_LEN + 1, target_index + 1)], dtype=np.float32)
        if feature_sequence.shape != (FUSION_SEQ_LEN, 9) or not np.isfinite(feature_sequence).all():
            raise Phase9FRejected(f"FUSION_SEQUENCE_GATE__{battery}_invalid_test_sequence")
        with torch.no_grad():
            fusion_prediction, weights, _, _ = fusion(torch.from_numpy(feature_sequence[None, ...]))
        fusion_value = float(fusion_prediction.item())
        weight = weights.cpu().numpy().reshape(3)
        simple_value = float(prediction_by_index[target_index].sum())
        persistence_value = float(history_capacity[-1])
        if not np.isfinite([fusion_value, simple_value, persistence_value, *weight]).all() or abs(float(weight.sum()) - 1.0) > 1e-6:
            raise Phase9FRejected(f"FROZEN_FORWARD_GATE__{battery}_nonfinite_output_or_weight_sum_failure")

        # Current TEST pair is unlocked only after every feature and prediction above exists.
        hf_current, target_capacity, endpoint_provenance = test_pair_after_prediction(
            cycles, metadata["pair_identities"][target_index], target_index, test_start
        )
        audit_row["test_pair_numerical_access_count"] += 1
        scaled_current, range_count = scale_one_frozen(hf_current, persisted_scaler)
        audit_row["test_hf_transform_count"] += 1
        audit_row["test_target_metric_count"] += 1
        for offset, name in enumerate(("HF1", "HF2", "HF3")):
            test_scale_below[name] += int((hf_current[offset] < persisted_scaler["train_min_max_by_feature"][name]["min"]))
            test_scale_above[name] += int((hf_current[offset] > persisted_scaler["train_min_max_by_feature"][name]["max"]))
        endpoint_fallback_count += int(endpoint_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
        rows.append({
            "sample_index": target_index,
            "target_capacity_ah": target_capacity,
            "simple_sum_capacity_ah": simple_value,
            "intelligent_fusion_capacity_ah": fusion_value,
            "persistence_capacity_ah": persistence_value,
            "simple_sum_error_ah": simple_value - target_capacity,
            "intelligent_fusion_error_ah": fusion_value - target_capacity,
            "persistence_error_ah": persistence_value - target_capacity,
            "weight_trend": float(weight[0]), "weight_seasonal": float(weight[1]), "weight_residual": float(weight[2]),
        })
        # After the prediction and target-only evaluation, this pair becomes past history.
        history_hf.append(scaled_current)
        history_capacity.append(target_capacity)

    if len(rows) != total - test_start or len(history_hf) != total or len(history_capacity) != total:
        raise Phase9FRejected(f"TEST_SEQUENCE_GATE__{battery}_final_count_or_history_mismatch")
    targets = np.asarray([row["target_capacity_ah"] for row in rows], dtype=np.float64)
    simple = np.asarray([row["simple_sum_capacity_ah"] for row in rows], dtype=np.float64)
    fusion_pred = np.asarray([row["intelligent_fusion_capacity_ah"] for row in rows], dtype=np.float64)
    persistence = np.asarray([row["persistence_capacity_ah"] for row in rows], dtype=np.float64)
    weights = np.asarray([[row["weight_trend"], row["weight_seasonal"], row["weight_residual"]] for row in rows], dtype=np.float64)
    numeric = np.concatenate((targets, simple, fusion_pred, persistence, weights.reshape(-1)))
    if not np.isfinite(numeric).all():
        raise Phase9FRejected(f"FINITE_GATE__{battery}_nan_or_inf_detected")
    post_hashes = {f"branch_{kind}": state_fingerprint(branch.state_dict()) for kind, branch in branches.items()}
    post_hashes["fusion"] = state_fingerprint(fusion.state_dict())
    if pre_hashes != post_hashes:
        raise Phase9FRejected(f"FROZEN_MODEL_GATE__{battery}_parameter_fingerprint_changed_during_test")
    test_metrics = {"simple_sum": metrics(simple, targets), "intelligent_fusion": metrics(fusion_pred, targets), "persistence": metrics(persistence, targets)}
    test_soh = {"simple_sum": _soh_metrics_percent(simple, targets), "intelligent_fusion": _soh_metrics_percent(fusion_pred, targets), "persistence": _soh_metrics_percent(persistence, targets)}
    result = {
        "test_n": len(rows), "prediction_domain": "CAPACITY_AH", "capacity_to_soh": "SOH=Capacity_Ah/2.0__FROZEN_TIER2_BRIDGE",
        "test_capacity_metrics": test_metrics, "test_soh_metrics_percent": test_soh,
        "fusion_weights": {"mean": {kind: float(weights[:, i].mean()) for i, kind in enumerate(KINDS)}, "std": {kind: float(weights[:, i].std(ddof=0)) for i, kind in enumerate(KINDS)}, "weight_sum_max_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0)))},
        "frozen_model_parameter_fingerprints_before": pre_hashes, "frozen_model_parameter_fingerprints_after": post_hashes,
        "test_scaler_range_counts": {"below_train_range": test_scale_below, "above_train_range": test_scale_above, "clipping_policy": persisted_scaler["clipping_policy"]},
        "hf2_final_timestamp_fallback_count": endpoint_fallback_count,
        "sequential_leakage_audit": audit_row,
        "simple_vs_persistence_rmse_status": benefit_status(test_metrics["simple_sum"]["rmse"], test_metrics["persistence"]["rmse"], "SIMPLE_VS_PERSISTENCE"),
        "fusion_vs_persistence_rmse_status": benefit_status(test_metrics["intelligent_fusion"]["rmse"], test_metrics["persistence"]["rmse"], "FUSION_VS_PERSISTENCE"),
        "fusion_vs_simple_rmse_status": benefit_status(test_metrics["intelligent_fusion"]["rmse"], test_metrics["simple_sum"]["rmse"], "FUSION_VS_SIMPLE"),
        "augmentation_numerical_benefit_status": "NOT_IDENTIFIABLE__NO_FROZEN_NONAUGMENTED_TEST_COUNTERFACTUAL",
        "intelligent_fusion_numerical_benefit_status": benefit_status(test_metrics["intelligent_fusion"]["rmse"], test_metrics["simple_sum"]["rmse"], "FUSION_VS_SIMPLE_SUM"),
        "persistence_baseline_status": "REPORTED__FROZEN_PREVIOUS_OBSERVED_CAPACITY_DEFINITION",
    }
    result["paper_comparison"] = paper_reference(battery, test_soh["intelligent_fusion"], test_metrics["intelligent_fusion"])
    return result, rows, {"pretest_access": pretest_access, "post_unlock_test_pair_access_count": len(rows)}


def reproduction_status(results: dict[str, dict[str, Any]]) -> str:
    reference = results["B0018"]["paper_comparison"]
    if reference["status"].startswith("DIRECT"):
        difference = reference["absolute_difference"]
        if all(np.isclose(value, 0.0, rtol=0.0, atol=1e-12) for value in difference.values()):
            return "PARTIALLY_MATCHED"  # Paper-exact protocol remains explicitly unresolved.
    return "NOT_MATCHED"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase9F NASA Final One-Shot TEST Evaluation", "",
        "This is one frozen-model, chronological final evaluation. It is Tier-2 paper-closest numerical work, not a paper-exact full-protocol claim. TEST results were not used for any update, selection, scaling, STL parameter choice, or tuning.", "",
        "## Pre-test immutability", "", "```text",
        f"SOURCE_COMMIT={SOURCE_COMMIT}", "WORKTREE_CLEAN_BEFORE_PHASE9F_IMPLEMENTATION=True", "FINAL_TEST_PREFLIGHT_GATE=PASS",
        "FINAL_TEST_UNLOCKED=True", "FINAL_TEST_UNLOCK_REASON=FROZEN_MODEL_ONE_SHOT_EVALUATION", "```", "",
    ]
    for battery, result in payload["results_by_battery"].items():
        cap, soh, weights = result["test_capacity_metrics"], result["test_soh_metrics_percent"], result["fusion_weights"]
        lines += [f"## {battery}: Phase9F TEST", "", f"TEST_N={result['test_n']}", "", "| Frozen method | RMSE Ah | MAE Ah | R2 | SOH RMSE % | SOH MAE % |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, key in (("Simple sum", "simple_sum"), ("Intelligent fusion", "intelligent_fusion"), ("Persistence", "persistence")):
            lines.append(f"| {label} | {cap[key]['rmse']:.12g} | {cap[key]['mae']:.12g} | {cap[key]['r2']:.12g} | {soh[key]['rmse_percent']:.12g} | {soh[key]['mae_percent']:.12g} |")
        lines += ["", f"MEAN_WEIGHT_T/S/R={weights['mean']['trend']:.12g}/{weights['mean']['seasonal']:.12g}/{weights['mean']['residual']:.12g}; STD={weights['std']['trend']:.12g}/{weights['std']['seasonal']:.12g}/{weights['std']['residual']:.12g}; WEIGHT_SUM_MAX_ERROR={weights['weight_sum_max_error']:.12g}.", f"SIMPLE_VS_PERSISTENCE={result['simple_vs_persistence_rmse_status']}; FUSION_VS_PERSISTENCE={result['fusion_vs_persistence_rmse_status']}; FUSION_VS_SIMPLE={result['fusion_vs_simple_rmse_status']}.", ""]
        paper = result["paper_comparison"]
        lines += ["### Paper-reported comparison", ""]
        if paper["status"] == "NO_DIRECT_PAPER_REFERENCE":
            lines.append("NO_DIRECT_PAPER_REFERENCE")
        else:
            reported = paper["paper_reported_developed_method"]
            ours = paper["our_tier2_paper_closest_intelligent_fusion"]
            lines += ["| Item | Paper developed method | Our Tier2 frozen intelligent fusion |", "| --- | ---: | ---: |", f"| MAE % | {reported['mae_percent']:.12g} | {ours['mae_percent']:.12g} |", f"| RMSE % | {reported['rmse_percent']:.12g} | {ours['rmse_percent']:.12g} |", f"| R2 | {reported['r2']:.12g} | {ours['r2']:.12g} |", ""]
    lines += [
        "## Phase9E validation history (separate, non-comparable)", "",
        "The retained Phase9E validation results use the preceding 60/20/20 validation partition. They are preserved separately in `PHASE9E_NASA_60_20_20_PRETEST_METRICS.json` and are not merged with, selected by, or numerically compared as final TEST results.", "",
        "## Tier-2 / paper-ambiguity limitations", "",
        "`PAPER_EXACT_FULL_PROTOCOL=False`. The evaluation preserves the documented Tier-2 pairing, HF endpoint, scalar readout, synthetic-provenance, and fusion fallbacks; it does not establish paper-exact reproduction.", "",
        "## Mandatory audit", "", "```text",
        "MODEL_PARAMETERS_UPDATED_DURING_TEST=False", "CHECKPOINT_SELECTED_USING_TEST=False", "SCALER_FIT_USING_TEST=False", "STL_PARAMETER_SELECTED_USING_TEST=False", "HYPERPARAMETER_TUNING_USING_TEST=False", "GRID_SEARCH_RUN=False", "SCFGAN_RETRAINED=False", "SYNTHETIC_REGENERATED=False", "NaN_INF_COUNT=0",
        f"NASA_FINAL_TEST_EXECUTION_GATE={payload['nasa_final_test_execution_gate']}", f"NUMERICAL_REPRODUCTION_STATUS={payload['numerical_reproduction_status']}", f"NEXT_ACTION={payload['next_action']}", "```", "",
    ]
    return "\n".join(lines)


def main() -> None:
    try:
        phase9e, split_manifest, provenance, checkpoint_details = preflight()
        results: dict[str, dict[str, Any]] = {}
        rows_by_battery: dict[str, list[dict[str, Any]]] = {}
        access_by_battery: dict[str, dict[str, Any]] = {}
        # Explicit one-shot unlock occurs only after preflight returns PASS.
        for battery in TARGET_BATTERIES:
            results[battery], rows_by_battery[battery], access_by_battery[battery] = one_battery(battery, phase9e, split_manifest)
        if any(item["sequential_leakage_audit"][key] for item in results.values() for key in ("current_test_target_used", "future_test_capacity_used", "future_test_hf_used")):
            raise Phase9FRejected("FINAL_TEST_LEAKAGE_GATE__current_or_future_test_data_used")
        if any(item["sequential_leakage_audit"]["test_pair_numerical_access_count"] != item["test_n"] for item in results.values()):
            raise Phase9FRejected("FINAL_TEST_ACCESS_GATE__test_access_count_not_exactly_one_pair_per_prediction")
    except Exception as error:
        print(json.dumps({"status": "RESEARCH_DIRECTION_REJECTED", "failed_gate": str(error)}, indent=2))
        raise

    total_test_accesses = sum(item["sequential_leakage_audit"]["test_pair_numerical_access_count"] for item in results.values())
    payload = {
        "phase": "PHASE9F_NASA_FINAL_ONE_SHOT_TEST", "source_commit": SOURCE_COMMIT,
        "authority": "EXPLICIT_HUMAN_AUTHORIZATION__FROZEN_MODEL_ONE_SHOT_FINAL_TEST", "final_test_preflight_gate": "PASS", "final_test_unlocked": True,
        "final_test_unlock_reason": "FROZEN_MODEL_ONE_SHOT_EVALUATION", "test_loaded": True, "test_access_count": total_test_accesses,
        "test_access_count_definition": "PERMITTED_SEQUENTIAL_TEST_PAIR_NUMERICAL_DEREFERENCES__ONE_AFTER_EACH_CURRENT_PREDICTION",
        "paper_exact_full_protocol": False, "reproduction_tier": "TIER2_PAPER_CLOSEST_NUMERICAL", "phase9e_provenance": provenance,
        "frozen_model_manifest": checkpoint_details, "results_by_battery": results, "test_access_audit_by_battery": access_by_battery,
        "model_parameters_updated_during_test": False, "checkpoint_selected_using_test": False, "scaler_fit_using_test": False,
        "stl_parameter_selected_using_test": False, "hyperparameter_tuning_using_test": False, "grid_search_run": False,
        "scfgan_retrained": False, "synthetic_regenerated": False, "current_test_target_used": False,
        "future_test_capacity_used": False, "future_test_hf_used": False, "nan_inf_count": 0,
        "nasa_final_test_execution_gate": "PASS", "numerical_reproduction_status": reproduction_status(results),
        "next_action": "RETURN_FINAL_NASA_RESULTS_FOR_HUMAN_REVIEW",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    MANIFEST_PATH.write_text(json.dumps(checkpoint_details, indent=2), encoding="utf-8")
    AUDIT_PATH.write_text(json.dumps({key: payload[key] for key in ("phase", "source_commit", "final_test_preflight_gate", "final_test_unlocked", "test_loaded", "test_access_count", "test_access_count_definition", "model_parameters_updated_during_test", "checkpoint_selected_using_test", "scaler_fit_using_test", "stl_parameter_selected_using_test", "hyperparameter_tuning_using_test", "grid_search_run", "scfgan_retrained", "synthetic_regenerated", "current_test_target_used", "future_test_capacity_used", "future_test_hf_used", "nan_inf_count", "nasa_final_test_execution_gate") } | {"test_access_audit_by_battery": access_by_battery}, indent=2), encoding="utf-8")
    PREDICTIONS_PATH.write_text(json.dumps({"phase": payload["phase"], "prediction_domain": "CAPACITY_AH", "rows_by_battery": rows_by_battery}, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "nasa_final_test_execution_gate": "PASS", "test_access_count": total_test_accesses}, indent=2))


if __name__ == "__main__":
    main()
