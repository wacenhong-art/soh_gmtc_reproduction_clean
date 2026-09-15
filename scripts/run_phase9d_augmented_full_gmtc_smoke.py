"""Tier-2 Phase 9D augmented end-to-end G-MTC integration smoke.

This script is intentionally a three-epoch integration check, not a paper
numerical-reproduction claim.  It consumes the already persisted PC0019
Figure-3 synthetic four-vectors; it never retrains or reselects SCF-GAN and it
does not dereference any TEST numerical record.
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_phase9a_tier2_paired_train_val_dry_run import (
    PERIOD,
    SEQ_LEN,
    Branch,
    ProtocolViolation,
    Split,
    build_train_samples,
    build_val_samples,
    capacity_to_soh,
    metrics,
    numeric_pairs_for_train_and_val,
    scale_train_val,
    validate_pairing_and_splits,
)
from run_phase9b_minimal_intelligent_fusion import (
    EPS,
    FUSION_SEQ_LEN,
    KINDS,
    MinimalIntelligentFusion,
    branch_prediction_matrix,
    feature_row,
)
from stl_decomposition import STLDecomposer
from tier2_phase9a_config import FROZEN_TIER2_PHASE9A_CONFIG
from tier2_scalar_component_readout import strict_scalar_mse


ROOT = Path(__file__).resolve().parent.parent
PC0019_METRICS = ROOT / "PHASE9C_FIGURE3_SCFGAN_RECONSTRUCTION_METRICS.json"
SYNTHETIC_ROOT = ROOT / "output" / "phase9c_figure3_reconstruction"
OUTPUT_ROOT = ROOT / "output" / "phase9d_augmented_full_gmtc_smoke"
METRICS_PATH = ROOT / "PHASE9D_AUGMENTED_FULL_GMTC_SMOKE_METRICS.json"
REPORT_PATH = ROOT / "PHASE9D_AUGMENTED_FULL_GMTC_SMOKE_REPORT.md"
MANIFEST_PATH = ROOT / "PHASE9D_AUGMENTATION_MANIFEST.json"
TARGET_BATTERIES = ("B0007", "B0018")
SYNTHETIC_SET_COUNT = 5
SYNTHETIC_FEATURE_DIM = 4
EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SEED = 42
RATED_CAPACITY_AH = 2.0


class Phase9DProtocolViolation(ProtocolViolation):
    """Raised before final artifacts are written when an integration gate fails."""


def set_seed() -> None:
    FROZEN_TIER2_PHASE9A_CONFIG.validate()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def source_preflight() -> dict[str, Any]:
    if not PC0019_METRICS.is_file() or not SYNTHETIC_ROOT.is_dir():
        raise Phase9DProtocolViolation("missing PC0019 persisted artifacts")
    metrics_payload = json.loads(PC0019_METRICS.read_text(encoding="utf-8"))
    if metrics_payload["selection"]["epoch"] != 200:
        raise Phase9DProtocolViolation("PC0019 selected source epoch is not frozen epoch 200")
    if metrics_payload["target_information_used_for_training"] or metrics_payload["target_information_used_for_checkpoint_selection"]:
        raise Phase9DProtocolViolation("PC0019 provenance gate is not target-blind")
    if metrics_payload["test_access_count"] != 0 or metrics_payload["test_loaded"]:
        raise Phase9DProtocolViolation("PC0019 TEST lock is not clean")
    return {
        "selected_source_epoch": 200,
        "scfgan_retrained": False,
        "synthetic_regenerated": False,
        "pc0019_architecture_gate": metrics_payload["figure3_architecture_contract_gate"],
        "pc0019_target_blind_training": True,
    }


def synthetic_sets(battery: str, expected_rows: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    sets: list[np.ndarray] = []
    set_manifest: list[dict[str, Any]] = []
    for number in range(1, SYNTHETIC_SET_COUNT + 1):
        path = SYNTHETIC_ROOT / f"{battery}_figure3_synthetic_set_{number:02d}.npy"
        if not path.is_file():
            raise Phase9DProtocolViolation(f"missing persisted PC0019 synthetic set: {path.name}")
        value = np.load(path, allow_pickle=False)
        if value.shape != (expected_rows, SYNTHETIC_FEATURE_DIM) or not np.isfinite(value).all():
            raise Phase9DProtocolViolation(f"illegal synthetic shape or finite contract for {path.name}")
        capacity = value[:, 3] * RATED_CAPACITY_AH
        if not np.isfinite(capacity).all():
            raise Phase9DProtocolViolation(f"synthetic SOH-to-capacity bridge is non-finite for {path.name}")
        sets.append(value.astype(np.float64, copy=False))
        set_manifest.append(
            {
                "set": number,
                "path": str(path.relative_to(ROOT)),
                "shape": list(value.shape),
                "trajectory_id": f"{battery}_PC0019_SET_{number:02d}",
                "soh_stats": stats(value[:, 3]),
                "capacity_ah_stats": stats(capacity),
            }
        )
    return sets, {"trajectory_count": SYNTHETIC_SET_COUNT, "sets": set_manifest}


def stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise Phase9DProtocolViolation("summary statistic input is empty or non-finite")
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}


def stl_components(capacity: np.ndarray, trajectory_id: str) -> dict[str, np.ndarray]:
    result = STLDecomposer().decompose(
        series=capacity,
        period=PERIOD,
        partition_id="TIER2_SYNTHETIC_TRAJECTORY" if trajectory_id.startswith("SYNTHETIC") else "TRAIN",
        series_id=trajectory_id,
    )
    audit = result["structural_audit"]
    if not audit["length_preserved"] or not audit["no_nan_or_inf"] or audit["max_abs_reconstruction_error"] > 1e-10:
        raise Phase9DProtocolViolation(f"STL reconstruction contract failed for {trajectory_id}")
    return {kind: np.asarray(result[kind], dtype=np.float64) for kind in KINDS}


def transform_synthetic_hf(physical_hf: np.ndarray, scaler: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    ranges = scaler["train_min_max_by_feature"]
    low = np.asarray([ranges[f"HF{index}"]["min"] for index in range(1, 4)], dtype=np.float64)
    high = np.asarray([ranges[f"HF{index}"]["max"] for index in range(1, 4)], dtype=np.float64)
    if np.any(high <= low):
        raise Phase9DProtocolViolation("persisted real TRAIN scaler has zero range")
    transformed = (physical_hf - low) / (high - low)
    if transformed.shape[1] != 3 or not np.isfinite(transformed).all():
        raise Phase9DProtocolViolation("synthetic scaler transform contract failed")
    return transformed.astype(np.float32), {
        "below_0": int((transformed < 0.0).sum()),
        "above_1": int((transformed > 1.0).sum()),
    }


def trajectory_branch_samples(hf: np.ndarray, capacity: np.ndarray, components: dict[str, np.ndarray], trajectory_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x, y = build_train_samples(hf, components)
    if x.shape[1:] != (SEQ_LEN, 3) or any(value.shape != (x.shape[0], 1, 1) for value in y.values()):
        raise Phase9DProtocolViolation(f"branch window shape failed for {trajectory_id}")
    if not np.isfinite(x).all() or not np.isfinite(capacity).all() or any(not np.isfinite(value).all() for value in y.values()):
        raise Phase9DProtocolViolation(f"branch window finite contract failed for {trajectory_id}")
    return x, y


def train_augmented_branches(train_x: np.ndarray, targets: dict[str, np.ndarray], val_x: np.ndarray, val_y: np.ndarray) -> tuple[dict[str, Branch], dict[str, Any]]:
    branches = {kind: Branch(kind) for kind in KINDS}
    optimizers = {kind: torch.optim.Adam(branches[kind].parameters(), lr=LEARNING_RATE) for kind in KINDS}
    x = torch.from_numpy(train_x)
    y = {kind: torch.from_numpy(targets[kind]) for kind in KINDS}
    trace = {kind: [] for kind in KINDS}
    audits = {kind: {"finite_loss": True, "finite_nonzero_core_gradient": False, "finite_nonzero_readout_gradient": False} for kind in KINDS}
    for _epoch in range(1, EPOCHS + 1):
        for kind in KINDS:
            branch = branches[kind]
            branch.train()
            total, count = 0.0, 0
            for start in range(0, x.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, x.shape[0])
                optimizers[kind].zero_grad(set_to_none=True)
                prediction = branch(x[start:end], partition="PHASE9D_AUGMENTED_TRAIN")
                loss = strict_scalar_mse(prediction, y[kind][start:end])
                if not torch.isfinite(loss):
                    raise Phase9DProtocolViolation(f"{kind} augmented branch loss is non-finite")
                loss.backward()
                for label, module in (("core", branch.core), ("readout", branch.readout)):
                    valid = any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and parameter.grad.abs().sum() > 0
                        for parameter in module.parameters()
                    )
                    audits[kind][f"finite_nonzero_{label}_gradient"] |= valid
                optimizers[kind].step()
                total += float(loss.detach()) * (end - start)
                count += end - start
            trace[kind].append(total / count)
    for branch in branches.values():
        branch.eval()
        for parameter in branch.parameters():
            parameter.requires_grad_(False)
        if branch.training or any(parameter.requires_grad for parameter in branch.parameters()):
            raise Phase9DProtocolViolation("branch freeze contract failed")
    with torch.no_grad():
        parts = {kind: branch(torch.from_numpy(val_x), partition="PHASE9D_VALIDATION") for kind, branch in branches.items()}
    if any(value.shape != val_y.shape or not torch.isfinite(value).all() for value in parts.values()):
        raise Phase9DProtocolViolation("augmented branch validation shape or finite contract failed")
    if any(not values["finite_nonzero_core_gradient"] or not values["finite_nonzero_readout_gradient"] for values in audits.values()):
        raise Phase9DProtocolViolation("augmented branch gradient contract failed")
    return branches, {"epochs": EPOCHS, "loss_trace": trace, "gradient_audit": audits, "validation_component_shapes": {kind: list(value.shape) for kind, value in parts.items()}, "validation_forward_finite": True, "branches_frozen_eval": True}


def fusion_trajectory_samples(branches: dict[str, Branch], hf: np.ndarray, capacity: np.ndarray, components: dict[str, np.ndarray], trajectory_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prediction_matrix = branch_prediction_matrix(branches, hf)
    audit = {"current_target_error_used": False, "future_validation_capacity_used": False, "validation_feature_rows": 0, "max_validation_capacity_index_used_for_feature": -1, "ratio_epsilon_guard_count": 0}
    rows = {index: feature_row(prediction_matrix, index, capacity, components, capacity.shape[0], trajectory_id, audit) for index in range(SEQ_LEN + 1, hf.shape[0])}
    endpoints = list(range(SEQ_LEN + FUSION_SEQ_LEN, hf.shape[0]))
    x = np.asarray([np.asarray([rows[index] for index in range(endpoint - FUSION_SEQ_LEN + 1, endpoint + 1)], dtype=np.float32) for endpoint in endpoints], dtype=np.float32)
    y = np.asarray(capacity[endpoints], dtype=np.float32).reshape(-1, 1, 1)
    if x.shape[1:] != (FUSION_SEQ_LEN, 9) or x.shape[0] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Phase9DProtocolViolation(f"fusion trajectory contract failed for {trajectory_id}")
    return x, y, {"trajectory_id": trajectory_id, "fusion_windows": int(x.shape[0]), "ratio_epsilon_guard_count": audit["ratio_epsilon_guard_count"], "cross_trajectory_access": False}


def validation_fusion_samples(branches: dict[str, Branch], history_hf: np.ndarray, capacities: np.ndarray, components: dict[str, np.ndarray], train_end: int, battery: str) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    prediction_matrix = branch_prediction_matrix(branches, history_hf)
    audit = {"current_target_error_used": False, "future_validation_capacity_used": False, "validation_feature_rows": 0, "max_validation_capacity_index_used_for_feature": -1, "ratio_epsilon_guard_count": 0}
    rows = {index: feature_row(prediction_matrix, index, capacities, components, train_end, battery, audit) for index in range(SEQ_LEN + 1, history_hf.shape[0])}
    endpoints = list(range(train_end, history_hf.shape[0]))
    x = np.asarray([np.asarray([rows[index] for index in range(endpoint - FUSION_SEQ_LEN + 1, endpoint + 1)], dtype=np.float32) for endpoint in endpoints], dtype=np.float32)
    y = np.asarray(capacities[endpoints], dtype=np.float32).reshape(-1, 1, 1)
    if x.shape[1:] != (FUSION_SEQ_LEN, 9) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Phase9DProtocolViolation("validation fusion feature contract failed")
    if audit["current_target_error_used"] or audit["future_validation_capacity_used"]:
        raise Phase9DProtocolViolation("validation fusion leakage contract failed")
    return x, y, endpoints, audit


def train_fusion(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, capacities: np.ndarray, val_endpoints: list[int]) -> tuple[MinimalIntelligentFusion, dict[str, Any], np.ndarray, np.ndarray]:
    fusion = MinimalIntelligentFusion()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    trace: list[float] = []
    for _epoch in range(1, EPOCHS + 1):
        fusion.train()
        total, count = 0.0, 0
        for start in range(0, x.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            prediction, weights, _, _ = fusion(x[start:end])
            loss = strict_scalar_mse(prediction, y[start:end])
            if not torch.isfinite(loss) or not torch.isfinite(weights).all():
                raise Phase9DProtocolViolation("augmented fusion loss/weight is non-finite")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in fusion.parameters()):
                raise Phase9DProtocolViolation("augmented fusion gradient is non-finite")
            optimizer.step()
            total += float(loss.detach()) * (end - start)
            count += end - start
        trace.append(total / count)
    fusion.eval()
    with torch.no_grad():
        prediction, weights, component_attention, time_attention = fusion(torch.from_numpy(val_x))
    if prediction.shape != val_y.shape or not torch.isfinite(prediction).all() or not torch.isfinite(weights).all():
        raise Phase9DProtocolViolation("augmented fusion validation contract failed")
    weight_error = float(np.max(np.abs(weights.numpy().sum(axis=1) - 1.0)))
    if weight_error > 1e-6:
        raise Phase9DProtocolViolation("augmented fusion weights do not sum to one")
    simple_sum = val_x[:, -1, :3].sum(axis=1)
    persistence = capacities[np.asarray(val_endpoints) - 1]
    target = val_y.reshape(-1)
    return fusion, {"epochs": EPOCHS, "loss_trace": trace, "validation_forward_finite": True, "weight_sum_max_error": weight_error, "mean_weights": {kind: float(weights.numpy()[:, index].mean()) for index, kind in enumerate(KINDS)}, "std_weights": {kind: float(weights.numpy()[:, index].std()) for index, kind in enumerate(KINDS)}, "component_attention": component_attention.numpy().reshape(-1).tolist(), "time_attention_shape": list(time_attention.shape)}, prediction.numpy().reshape(-1), np.column_stack((simple_sum, persistence, target))


def one_battery(battery: str, audits: dict[str, Any], splits: dict[str, Split], expected_synthetic_rows: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    set_seed()
    split = splits[battery]
    features, capacities, _, access = numeric_pairs_for_train_and_val(battery, audits[battery]["pair_identities"], split)
    if access["post_split_test_numerical_access_attempts"] or access["post_split_test_numerical_materialization_count"]:
        raise Phase9DProtocolViolation("TEST numerical access occurred during real TRAIN/VAL extraction")
    train_hf, val_hf, scaler = scale_train_val(features, split)
    train_capacity = capacities[:split.train_end]
    val_capacity = capacities[split.train_end:split.val_end]
    real_components = stl_components(train_capacity, f"{battery}_REAL_TRAIN_CAPACITY")
    real_x, real_targets = trajectory_branch_samples(train_hf, train_capacity, real_components, f"{battery}_REAL_TRAIN")
    sets, synthetic_manifest = synthetic_sets(battery, expected_synthetic_rows)
    synthetic_x: list[np.ndarray] = []
    synthetic_targets = {kind: [] for kind in KINDS}
    synthetic_fusion_inputs: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    per_set_windows: list[int] = []
    below_0 = above_1 = 0
    for set_number, vector in enumerate(sets, 1):
        scaled_hf, range_count = transform_synthetic_hf(vector[:, :3], scaler)
        below_0 += range_count["below_0"]
        above_1 += range_count["above_1"]
        capacity = vector[:, 3] * RATED_CAPACITY_AH
        components = stl_components(capacity, f"SYNTHETIC_{battery}_SET_{set_number:02d}")
        x, targets = trajectory_branch_samples(scaled_hf, capacity, components, f"SYNTHETIC_{battery}_SET_{set_number:02d}")
        synthetic_x.append(x)
        for kind in KINDS:
            synthetic_targets[kind].append(targets[kind])
        per_set_windows.append(int(x.shape[0]))
        synthetic_fusion_inputs.append((scaled_hf, capacity, components))
    augmented_x = np.concatenate([real_x, *synthetic_x], axis=0)
    augmented_targets = {kind: np.concatenate([real_targets[kind], *synthetic_targets[kind]], axis=0) for kind in KINDS}
    val_x, val_y = build_val_samples(train_hf, val_hf, val_capacity)
    branches, branch_result = train_augmented_branches(augmented_x, augmented_targets, val_x, val_y)
    real_fusion_x, real_fusion_y, real_fusion_audit = fusion_trajectory_samples(branches, train_hf, train_capacity, real_components, f"{battery}_REAL_TRAIN")
    fusion_x = [real_fusion_x]
    fusion_y = [real_fusion_y]
    fusion_set_audits = []
    for set_number, (scaled_hf, capacity, components) in enumerate(synthetic_fusion_inputs, 1):
        x, y, audit = fusion_trajectory_samples(branches, scaled_hf, capacity, components, f"SYNTHETIC_{battery}_SET_{set_number:02d}")
        fusion_x.append(x)
        fusion_y.append(y)
        fusion_set_audits.append(audit)
    augmented_fusion_x, augmented_fusion_y = np.concatenate(fusion_x), np.concatenate(fusion_y)
    history_hf, history_capacity = np.concatenate((train_hf, val_hf)), capacities
    val_fusion_x, val_fusion_y, val_endpoints, leakage = validation_fusion_samples(branches, history_hf, history_capacity, real_components, split.train_end, battery)
    fusion, fusion_result, prediction, baselines = train_fusion(augmented_fusion_x, augmented_fusion_y, val_fusion_x, val_fusion_y, history_capacity, val_endpoints)
    simple_sum, persistence, target = baselines[:, 0], baselines[:, 1], baselines[:, 2]
    intelligent_metric, simple_metric, persistence_metric = metrics(prediction, target), metrics(simple_sum, target), metrics(persistence, target)
    output = {
        "split": {"n_total": split.total, "n_train": split.counts[0], "n_val": split.counts[1], "n_test_membership_only": split.counts[2]},
        "real_train_windows": int(real_x.shape[0]),
        "synthetic_windows_per_set": per_set_windows,
        "total_synthetic_windows": int(sum(per_set_windows)),
        "total_augmented_train_windows": int(augmented_x.shape[0]),
        "synthetic_scale_below_0_count": below_0,
        "synthetic_scale_above_1_count": above_1,
        "normalization": {**scaler, "synthetic_transform": "REAL_TRAIN_SCALER_ONLY__UNCLIPPED"},
        "stl": {"real_train": {"period": PERIOD, "mode": "OFFLINE_TRAIN_ONLY"}, "synthetic": "TIER2_SYNTHETIC_TRAJECTORY_STL=True", "paper_exact_synthetic_stl_protocol": "UNRESOLVED"},
        "branch_smoke": branch_result,
        "fusion_training_windows": {"real": int(real_fusion_x.shape[0]), "synthetic_per_set": [int(item["fusion_windows"]) for item in fusion_set_audits], "total_augmented": int(augmented_fusion_x.shape[0])},
        "fusion_smoke": fusion_result,
        "leakage_gate": {**leakage, "current_target_error_used": False, "future_validation_capacity_used": False, "test_access_count": 0, "status": "PASS"},
        "validation": {
            "label_domain": "CAPACITY_AH",
            "intelligent_fusion": intelligent_metric,
            "simple_sum": simple_metric,
            "persistence": persistence_metric,
            "soh_metrics_percent": {"rmse_percent": 100.0 * metrics(capacity_to_soh(prediction), capacity_to_soh(target))["rmse"], "mae_percent": 100.0 * metrics(capacity_to_soh(prediction), capacity_to_soh(target))["mae"]},
        },
        "access": access,
        "nan_inf_count": int(sum((~np.isfinite(value)).sum() for value in [augmented_x, *augmented_targets.values(), val_x, val_y, augmented_fusion_x, augmented_fusion_y, val_fusion_x, val_fusion_y, prediction])),
    }
    if output["nan_inf_count"] != 0:
        raise Phase9DProtocolViolation("Phase9D aggregate NaN/Inf count is non-zero")
    checkpoints = {"branch_state_dicts": {kind: deepcopy(branch.state_dict()) for kind, branch in branches.items()}, "fusion_state_dict": deepcopy(fusion.state_dict())}
    manifest = {"synthetic_trajectory_manifest": synthetic_manifest, "real_train_window_count": int(real_x.shape[0]), "synthetic_window_counts": per_set_windows, "no_cross_trajectory_windows": True, "synthetic_capacity_ah_from_soh_times_2": True}
    return output, checkpoints, manifest


def report(payload: dict[str, Any]) -> str:
    lines = ["# Phase 9D Augmented Full G-MTC Integration Smoke", "", "Tier-2 integration-smoke diagnostic only; it is not a final paper-aligned numerical reproduction.", ""]
    for battery, result in payload["results_by_battery"].items():
        validation = result["validation"]
        lines.extend([f"## {battery}", "", f"REAL_TRAIN_WINDOWS={result['real_train_windows']}; SYNTHETIC_WINDOWS_PER_SET={result['synthetic_windows_per_set']}; TOTAL_SYNTHETIC_WINDOWS={result['total_synthetic_windows']}; TOTAL_AUGMENTED_TRAIN_WINDOWS={result['total_augmented_train_windows']}", "", "| Diagnostic | Capacity RMSE | Capacity MAE | R2 |", "| --- | ---: | ---: | ---: |"])
        for name, values in (("Intelligent fusion", validation["intelligent_fusion"]), ("Simple sum", validation["simple_sum"]), ("Persistence", validation["persistence"])):
            lines.append(f"| {name} | {values['rmse']:.12g} | {values['mae']:.12g} | {values['r2']:.12g} |")
        lines.extend(["", f"VAL_SOH_RMSE_PERCENT={validation['soh_metrics_percent']['rmse_percent']:.12g}; VAL_SOH_MAE_PERCENT={validation['soh_metrics_percent']['mae_percent']:.12g}.", ""])
    lines.extend(["```text", "SCFGAN_RETRAINED=False", "SYNTHETIC_REGENERATED=False", "TEST_LOADED=False", "TEST_ACCESS_COUNT=0", "GRID_SEARCH_RUN=False", "HYPERPARAMETER_TUNING=False", f"PHASE9D_AUGMENTED_FULL_GMTC_SMOKE_GATE={payload['phase9d_augmented_full_gmtc_smoke_gate']}", "```", ""])
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_ROOT.exists() or METRICS_PATH.exists() or REPORT_PATH.exists() or MANIFEST_PATH.exists():
        raise Phase9DProtocolViolation("refusing to overwrite an existing Phase9D artifact")
    preflight = source_preflight()
    audits, splits = validate_pairing_and_splits()
    expected_rows = {"B0007": 33, "B0018": 26}
    results, checkpoint_payloads, manifests = {}, {}, {}
    for battery in TARGET_BATTERIES:
        results[battery], checkpoint_payloads[battery], manifests[battery] = one_battery(battery, audits, splits, expected_rows[battery])
    if any(result["access"]["post_split_test_numerical_access_attempts"] or result["access"]["post_split_test_numerical_materialization_count"] for result in results.values()):
        raise Phase9DProtocolViolation("global TEST lock failed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    checkpoint_paths = {}
    for battery, checkpoint in checkpoint_payloads.items():
        path = OUTPUT_ROOT / f"{battery}_AUGMENTED_BRANCH_AND_FUSION_SMOKE_ONLY.pt"
        torch.save({"phase": "PHASE9D_AUGMENTED_FULL_GMTC_INTEGRATION_SMOKE", "battery": battery, **checkpoint}, path)
        checkpoint_paths[battery] = str(path.relative_to(ROOT))
    payload = {
        "phase": "PHASE9D_AUGMENTED_FULL_GMTC_INTEGRATION_SMOKE",
        "authority": "EXPLICIT_HUMAN_AUTHORIZATION__TIER2_INTEGRATION_SMOKE_ONLY",
        "protocol": {"development_split_70_20_10": True, "paper_aligned_60_20_20": False, "seq_len": SEQ_LEN, "pred_len": 1, "stl_period": PERIOD, "branch_epochs": EPOCHS, "fusion_epochs": EPOCHS, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "optimizer": "Adam", "seed": SEED, "fusion": "PHASE9B_MINIMAL_INTELLIGENT_FUSION_UNMODIFIED"},
        "preflight": preflight,
        "results_by_battery": results,
        "checkpoint_paths": checkpoint_paths,
        "scfgan_retrained": False,
        "synthetic_regenerated": False,
        "augmentation_assembly_gate": "PASS",
        "synthetic_trajectory_boundary_gate": "PASS",
        "scaler_provenance_gate": "PASS",
        "branch_augmented_smoke_gate": "PASS",
        "fusion_augmented_smoke_gate": "PASS",
        "test_loaded": False,
        "test_access_count": 0,
        "grid_search_run": False,
        "hyperparameter_tuning": False,
        "phase9d_augmented_full_gmtc_smoke_gate": "PASS",
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    manifest = {"phase": payload["phase"], "scope": "AUGMENTED_TRAIN_ONLY__REAL_VALIDATION_ONLY__TEST_SEALED", "pc0019_selected_source_epoch": 200, "by_battery": manifests}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": str(METRICS_PATH), "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
