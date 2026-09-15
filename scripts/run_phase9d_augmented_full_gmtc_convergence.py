"""Converge the frozen Tier-2 Phase 9D augmented G-MTC path.

This runner consumes only the persisted PC0019 synthetic trajectories and the
already authorized B0007/B0018 TRAIN/VALIDATION records.  It has no TEST data
path.  Unlike the three-epoch Phase 9D smoke, each branch is selected against
an expanding-prefix *current* validation component label.  That label uses
the observed capacity at its own validation endpoint and no later capacity;
it is used only for branch early stopping, never as a model/fusion input.
"""

from __future__ import annotations

import json
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
    build_train_samples,
    build_val_samples,
    capacity_to_soh,
    metrics,
    numeric_pairs_for_train_and_val,
    scale_train_val,
    validate_pairing_and_splits,
)
from run_phase9b_minimal_intelligent_fusion import (
    FUSION_SEQ_LEN,
    KINDS,
    MinimalIntelligentFusion,
    branch_prediction_matrix,
    feature_row,
)
from run_phase9d_augmented_full_gmtc_smoke import (
    BATCH_SIZE,
    LEARNING_RATE,
    RATED_CAPACITY_AH,
    SEED,
    TARGET_BATTERIES,
    Phase9DProtocolViolation,
    fusion_trajectory_samples,
    set_seed,
    source_preflight,
    stl_components,
    synthetic_sets,
    trajectory_branch_samples,
    transform_synthetic_hf,
    validation_fusion_samples,
)
from stl_decomposition import STLDecomposer
from tier2_scalar_component_readout import strict_scalar_mse


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "phase9d_augmented_full_gmtc_convergence"
METRICS_PATH = ROOT / "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE_METRICS.json"
REPORT_PATH = ROOT / "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE_REPORT.md"
PREDICTIONS_PATH = ROOT / "PHASE9D_AUGMENTED_VALIDATION_PREDICTIONS.json"
UN_AUGMENTED_SIMPLE_PATH = ROOT / "PHASE9A_CONVERGENCE_DIAGNOSTIC_METRICS.json"
UN_AUGMENTED_FUSION_PATH = ROOT / "PHASE9B_MINIMAL_INTELLIGENT_FUSION_METRICS.json"

MAX_EPOCHS = 150
PATIENCE = 50
MIN_DELTA = 0.0
WEIGHT_DECAY = 1e-5
EXPECTED_SYNTHETIC_ROWS = {"B0007": 33, "B0018": 26}


class ConvergenceProtocolViolation(Phase9DProtocolViolation):
    """Raised whenever the frozen Phase 9D convergence contract is violated."""


def _finite_scalar_mse(prediction: torch.Tensor, target: torch.Tensor, label: str) -> float:
    loss = strict_scalar_mse(prediction, target)
    if not torch.isfinite(loss):
        raise ConvergenceProtocolViolation(f"{label} MSE is non-finite")
    return float(loss.detach().cpu())


def current_validation_components(
    capacities: np.ndarray,
    train_components: dict[str, np.ndarray],
    train_end: int,
    battery: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Make legal endpoint labels without future validation capacity.

    For validation target i, STL sees capacities[:i+1], so the current observed
    validation target is permitted as a label while i+1...VAL/TEST are absent.
    This function is deliberately separate from the Phase 9A final-target
    evaluation path, whose historical contract remains ``VAL_STL_REQUIRED=False``.
    """

    if capacities.ndim != 1 or train_end < 2 * PERIOD or train_end >= capacities.shape[0]:
        raise ConvergenceProtocolViolation("validation component-label prefix dimensions are illegal")
    labels = {kind: [] for kind in KINDS}
    target_indices = list(range(train_end, capacities.shape[0]))
    max_source_index = -1
    reconstruction_error = 0.0
    for target_index in target_indices:
        prefix = capacities[: target_index + 1]
        decomposition = STLDecomposer().decompose(
            series=prefix,
            period=PERIOD,
            partition_id="VALIDATION",
            series_id=f"{battery}_VALIDATION_PREFIX_0_TO_{target_index}",
        )
        audit = decomposition["structural_audit"]
        if not audit["length_preserved"] or not audit["no_nan_or_inf"] or audit["max_abs_reconstruction_error"] > 1e-10:
            raise ConvergenceProtocolViolation(f"{battery} validation-prefix STL structural contract failed")
        for kind in KINDS:
            labels[kind].append(float(decomposition[kind][-1]))
        max_source_index = max(max_source_index, target_index)
        reconstruction_error = max(reconstruction_error, float(audit["max_abs_reconstruction_error"]))
    resolved = {
        kind: np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
        for kind, values in labels.items()
    }
    expected_shape = (len(target_indices), 1, 1)
    if any(value.shape != expected_shape or not np.isfinite(value).all() for value in resolved.values()):
        raise ConvergenceProtocolViolation(f"{battery} validation component label shape/finite contract failed")
    return resolved, {
        "policy": "EXPANDING_PREFIX_STL__CURRENT_OBSERVED_TARGET_ONLY__NO_FUTURE_CAPACITY",
        "used_only_for": "BRANCH_EARLY_STOPPING_COMPONENT_LOSS",
        "target_indices": target_indices,
        "current_target_component_label_used": True,
        "future_validation_capacity_used": False,
        "test_capacity_used": False,
        "max_capacity_index_used": max_source_index,
        "max_reconstruction_error": reconstruction_error,
        "validation_component_shapes": {kind: list(value.shape) for kind, value in resolved.items()},
    }


def train_augmented_branches(
    train_x: np.ndarray,
    targets: dict[str, np.ndarray],
    val_x: np.ndarray,
    val_component_targets: dict[str, np.ndarray],
    battery: str,
) -> tuple[dict[str, Branch], dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """Converge independent frozen branches with per-component real VAL loss."""

    branches = {kind: Branch(kind) for kind in KINDS}
    optimizers = {kind: torch.optim.Adam(branches[kind].parameters(), lr=LEARNING_RATE) for kind in KINDS}
    x = torch.from_numpy(train_x)
    y = {kind: torch.from_numpy(targets[kind]) for kind in KINDS}
    vx = torch.from_numpy(val_x)
    vy = {kind: torch.from_numpy(val_component_targets[kind]) for kind in KINDS}
    active = set(KINDS)
    train_trace = {kind: [] for kind in KINDS}
    val_trace = {kind: [] for kind in KINDS}
    best_mse = {kind: float("inf") for kind in KINDS}
    best_epoch = {kind: 0 for kind in KINDS}
    since_best = {kind: 0 for kind in KINDS}
    stop_epoch = {kind: None for kind in KINDS}
    status = {kind: "RUNNING" for kind in KINDS}
    best_state: dict[str, dict[str, torch.Tensor] | None] = {kind: None for kind in KINDS}
    gradient_audit = {
        kind: {"finite_nonzero_core_gradient": False, "finite_nonzero_readout_gradient": False}
        for kind in KINDS
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        for kind in KINDS:
            if kind not in active:
                continue
            branch = branches[kind]
            branch.train()
            total, seen = 0.0, 0
            for start in range(0, x.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, x.shape[0])
                optimizers[kind].zero_grad(set_to_none=True)
                prediction = branch(x[start:end], partition="PHASE9D_AUGMENTED_TRAIN")
                loss = strict_scalar_mse(prediction, y[kind][start:end])
                if not torch.isfinite(loss):
                    raise ConvergenceProtocolViolation(f"{battery} {kind} augmented TRAIN MSE is non-finite at epoch {epoch}")
                loss.backward()
                for label, module in (("core", branch.core), ("readout", branch.readout)):
                    gradient_audit[kind][f"finite_nonzero_{label}_gradient"] |= any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and parameter.grad.abs().sum() > 0
                        for parameter in module.parameters()
                    )
                optimizers[kind].step()
                total += float(loss.detach()) * (end - start)
                seen += end - start
            train_loss = total / seen
            if not np.isfinite(train_loss):
                raise ConvergenceProtocolViolation(f"{battery} {kind} aggregate augmented TRAIN MSE is non-finite")
            train_trace[kind].append(train_loss)

        for kind in KINDS:
            if kind not in active:
                continue
            branch = branches[kind]
            branch.eval()
            with torch.no_grad():
                prediction = branch(vx, partition="PHASE9D_VALIDATION_COMPONENT_SELECTION")
                val_loss = _finite_scalar_mse(prediction, vy[kind], f"{battery} {kind} validation component")
            val_trace[kind].append(val_loss)
            if val_loss < best_mse[kind] - MIN_DELTA:
                best_mse[kind] = val_loss
                best_epoch[kind] = epoch
                since_best[kind] = 0
                best_state[kind] = deepcopy(branch.state_dict())
            else:
                since_best[kind] += 1
            if since_best[kind] >= PATIENCE:
                status[kind] = "EARLY_STOPPED"
                stop_epoch[kind] = epoch
                active.remove(kind)
        if not active:
            break
    else:
        for kind in active:
            status[kind] = "INCONCLUSIVE_MAX_EPOCH_CAP"
            stop_epoch[kind] = MAX_EPOCHS

    restored_states: dict[str, dict[str, torch.Tensor]] = {}
    restored_shapes: dict[str, list[int]] = {}
    restored_mse: dict[str, float] = {}
    for kind in KINDS:
        state = best_state[kind]
        if state is None or best_epoch[kind] == 0 or stop_epoch[kind] is None:
            raise ConvergenceProtocolViolation(f"{battery} {kind} did not produce an early-stopping checkpoint")
        branches[kind].load_state_dict(state)
        branches[kind].eval()
        with torch.no_grad():
            restored = branches[kind](vx, partition="PHASE9D_VALIDATION_COMPONENT_SELECTION")
            restored_mse[kind] = _finite_scalar_mse(restored, vy[kind], f"{battery} {kind} restored validation component")
        if not np.isclose(restored_mse[kind], best_mse[kind], rtol=0.0, atol=1e-12):
            raise ConvergenceProtocolViolation(f"{battery} {kind} restored checkpoint differs from its best validation component MSE")
        restored_shapes[kind] = list(restored.shape)
        restored_states[kind] = {name: value.detach().cpu().clone() for name, value in state.items()}
        for parameter in branches[kind].parameters():
            parameter.requires_grad_(False)
        if branches[kind].training or any(parameter.requires_grad for parameter in branches[kind].parameters()):
            raise ConvergenceProtocolViolation(f"{battery} {kind} branch freeze contract failed")
        if not gradient_audit[kind]["finite_nonzero_core_gradient"] or not gradient_audit[kind]["finite_nonzero_readout_gradient"]:
            raise ConvergenceProtocolViolation(f"{battery} {kind} core/readout gradient path was inactive")

    return branches, {
        "selection_metric": "REAL_VALIDATION_EXPANDING_PREFIX_COMPONENT_MSE",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "restore_best_checkpoint": True,
        "by_branch": {
            kind: {
                "best_epoch": best_epoch[kind],
                "stop_epoch": stop_epoch[kind],
                "best_val_component_mse": best_mse[kind],
                "restored_val_component_mse": restored_mse[kind],
                "epochs_since_best_at_stop": since_best[kind],
                "status": status[kind],
                "train_component_mse_by_epoch": train_trace[kind],
                "val_component_mse_by_epoch": val_trace[kind],
                "output_shape": restored_shapes[kind],
                "gradient_audit": gradient_audit[kind],
            }
            for kind in KINDS
        },
    }, restored_states


def train_fusion(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    capacities: np.ndarray,
    val_endpoints: list[int],
    battery: str,
) -> tuple[MinimalIntelligentFusion, dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """Converge the frozen Phase9B fusion topology and restore its best state."""

    # The goal explicitly requires a fresh seed-42 fusion initialization.
    set_seed()
    fusion = MinimalIntelligentFusion()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(val_x), torch.from_numpy(val_y)
    best_mse, best_epoch, since_best = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    train_trace: list[float] = []
    val_trace: list[float] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        fusion.train()
        total, seen = 0.0, 0
        for start in range(0, x.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            prediction, weights, _, _ = fusion(x[start:end])
            loss = strict_scalar_mse(prediction, y[start:end])
            if not torch.isfinite(loss) or not torch.isfinite(weights).all():
                raise ConvergenceProtocolViolation(f"{battery} fusion TRAIN loss/weight is non-finite")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in fusion.parameters()):
                raise ConvergenceProtocolViolation(f"{battery} fusion gradient is non-finite")
            optimizer.step()
            total += float(loss.detach()) * (end - start)
            seen += end - start
        train_trace.append(total / seen)
        fusion.eval()
        with torch.no_grad():
            prediction, weights, _, _ = fusion(vx)
            val_mse = _finite_scalar_mse(prediction, vy, f"{battery} fusion validation capacity")
        val_trace.append(val_mse)
        if val_mse < best_mse - MIN_DELTA:
            best_mse, best_epoch, since_best = val_mse, epoch, 0
            best_state = deepcopy(fusion.state_dict())
        else:
            since_best += 1
        if since_best >= PATIENCE:
            status, stop_epoch = "EARLY_STOPPED", epoch
            break
    else:
        status, stop_epoch = "INCONCLUSIVE_MAX_EPOCH_CAP", MAX_EPOCHS
    if best_state is None or best_epoch == 0:
        raise ConvergenceProtocolViolation(f"{battery} fusion did not produce a best checkpoint")
    fusion.load_state_dict(best_state)
    fusion.eval()
    with torch.no_grad():
        prediction, weights, component_attention, time_attention = fusion(vx)
        restored_mse = _finite_scalar_mse(prediction, vy, f"{battery} restored fusion validation capacity")
    if not np.isclose(restored_mse, best_mse, rtol=0.0, atol=1e-12):
        raise ConvergenceProtocolViolation(f"{battery} restored fusion checkpoint differs from best validation MSE")
    weight_array = weights.cpu().numpy()
    sum_error = float(np.max(np.abs(weight_array.sum(axis=1) - 1.0)))
    if sum_error > 1e-6:
        raise ConvergenceProtocolViolation(f"{battery} fusion component weights do not sum to one")
    target = val_y.reshape(-1)
    simple_sum = val_x[:, -1, :3].sum(axis=1)
    persistence = capacities[np.asarray(val_endpoints) - 1]
    if not np.isfinite(simple_sum).all() or not np.isfinite(persistence).all():
        raise ConvergenceProtocolViolation(f"{battery} baseline construction is non-finite")
    return fusion, {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "status": status,
        "best_val_capacity_mse": best_mse,
        "restored_val_capacity_mse": restored_mse,
        "epochs_since_best_at_stop": since_best,
        "train_capacity_mse_by_epoch": train_trace,
        "val_capacity_mse_by_epoch": val_trace,
        "weight_sum_max_error": sum_error,
        "mean_weights": {kind: float(weight_array[:, index].mean()) for index, kind in enumerate(KINDS)},
        "std_weights": {kind: float(weight_array[:, index].std(ddof=0)) for index, kind in enumerate(KINDS)},
        "component_attention": component_attention.cpu().numpy().reshape(-1).tolist(),
        "time_attention_shape": list(time_attention.shape),
    }, {name: value.detach().cpu().clone() for name, value in best_state.items()}, prediction.cpu().numpy().reshape(-1), simple_sum, persistence


def _unaugmented_metrics(battery: str) -> dict[str, float]:
    if not UN_AUGMENTED_SIMPLE_PATH.is_file() or not UN_AUGMENTED_FUSION_PATH.is_file():
        raise ConvergenceProtocolViolation("missing persisted unaugmented convergence/fusion evidence")
    simple = json.loads(UN_AUGMENTED_SIMPLE_PATH.read_text(encoding="utf-8"))["results_by_battery"][battery]["validation_metrics_capacity"]
    fusion = json.loads(UN_AUGMENTED_FUSION_PATH.read_text(encoding="utf-8"))["results_by_battery"][battery]["validation_capacity_metrics"]
    return {"simple_sum_rmse": float(simple["rmse"]), "intelligent_fusion_rmse": float(fusion["rmse"])}


def one_battery(battery: str, audits: dict[str, Any], splits: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    set_seed()
    split = splits[battery]
    features, capacities, _, access = numeric_pairs_for_train_and_val(battery, audits[battery]["pair_identities"], split)
    if access["post_split_test_numerical_access_attempts"] or access["post_split_test_numerical_materialization_count"]:
        raise ConvergenceProtocolViolation(f"{battery} TEST numerical access occurred during TRAIN/VAL extraction")
    train_hf, val_hf, scaler = scale_train_val(features, split)
    train_capacity = capacities[:split.train_end]
    val_capacity = capacities[split.train_end:split.val_end]
    real_components = stl_components(train_capacity, f"{battery}_REAL_TRAIN_CAPACITY")
    real_x, real_targets = trajectory_branch_samples(train_hf, train_capacity, real_components, f"{battery}_REAL_TRAIN")
    if real_x.shape[0] != {"B0007": 107, "B0018": 82}[battery]:
        raise ConvergenceProtocolViolation(f"{battery} frozen real TRAIN window count changed")

    sets, synthetic_manifest = synthetic_sets(battery, EXPECTED_SYNTHETIC_ROWS[battery])
    synthetic_x: list[np.ndarray] = []
    synthetic_targets = {kind: [] for kind in KINDS}
    synthetic_fusion_inputs: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    branch_windows_per_set: list[int] = []
    below_0 = above_1 = 0
    for set_number, vector in enumerate(sets, 1):
        scaled_hf, range_count = transform_synthetic_hf(vector[:, :3], scaler)
        below_0 += range_count["below_0"]
        above_1 += range_count["above_1"]
        synthetic_capacity = vector[:, 3] * RATED_CAPACITY_AH
        synthetic_components = stl_components(synthetic_capacity, f"SYNTHETIC_{battery}_SET_{set_number:02d}")
        item_x, item_targets = trajectory_branch_samples(scaled_hf, synthetic_capacity, synthetic_components, f"SYNTHETIC_{battery}_SET_{set_number:02d}")
        synthetic_x.append(item_x)
        for kind in KINDS:
            synthetic_targets[kind].append(item_targets[kind])
        branch_windows_per_set.append(int(item_x.shape[0]))
        synthetic_fusion_inputs.append((scaled_hf, synthetic_capacity, synthetic_components))

    augmented_x = np.concatenate([real_x, *synthetic_x], axis=0)
    augmented_targets = {kind: np.concatenate([real_targets[kind], *synthetic_targets[kind]], axis=0) for kind in KINDS}
    expected_synthetic_windows = {"B0007": 115, "B0018": 80}[battery]
    if sum(branch_windows_per_set) != expected_synthetic_windows or augmented_x.shape[0] != real_x.shape[0] + expected_synthetic_windows:
        raise ConvergenceProtocolViolation(f"{battery} frozen augmented branch window count changed")
    val_x, val_capacity_target = build_val_samples(train_hf, val_hf, val_capacity)
    val_component_targets, val_component_audit = current_validation_components(capacities, real_components, split.train_end, battery)
    branches, branch_result, branch_states = train_augmented_branches(
        augmented_x, augmented_targets, val_x, val_component_targets, battery
    )

    real_fusion_x, real_fusion_y, real_fusion_audit = fusion_trajectory_samples(
        branches, train_hf, train_capacity, real_components, f"{battery}_REAL_TRAIN"
    )
    fusion_x, fusion_y = [real_fusion_x], [real_fusion_y]
    synthetic_fusion_audits: list[dict[str, Any]] = []
    for set_number, (scaled_hf, synthetic_capacity, synthetic_components) in enumerate(synthetic_fusion_inputs, 1):
        item_x, item_y, item_audit = fusion_trajectory_samples(
            branches, scaled_hf, synthetic_capacity, synthetic_components, f"SYNTHETIC_{battery}_SET_{set_number:02d}"
        )
        fusion_x.append(item_x)
        fusion_y.append(item_y)
        synthetic_fusion_audits.append(item_audit)
    augmented_fusion_x, augmented_fusion_y = np.concatenate(fusion_x), np.concatenate(fusion_y)
    expected_fusion_synthetic = {"B0007": 65, "B0018": 30}[battery]
    if sum(int(item["fusion_windows"]) for item in synthetic_fusion_audits) != expected_fusion_synthetic:
        raise ConvergenceProtocolViolation(f"{battery} frozen augmented fusion window count changed")
    history_hf = np.concatenate((train_hf, val_hf), axis=0)
    val_fusion_x, val_fusion_y, val_endpoints, leakage = validation_fusion_samples(
        branches, history_hf, capacities, real_components, split.train_end, battery
    )
    if leakage["current_target_error_used"] or leakage["future_validation_capacity_used"]:
        raise ConvergenceProtocolViolation(f"{battery} validation fusion leakage gate failed")
    fusion, fusion_result, fusion_state, prediction, simple_sum, persistence = train_fusion(
        augmented_fusion_x, augmented_fusion_y, val_fusion_x, val_fusion_y, capacities, val_endpoints, battery
    )
    target = val_fusion_y.reshape(-1)
    if not np.array_equal(target, val_capacity_target.reshape(-1)):
        raise ConvergenceProtocolViolation(f"{battery} branch/fusion validation target alignment differs")
    intelligent_metrics = metrics(prediction, target)
    simple_metrics = metrics(simple_sum, target)
    persistence_metrics = metrics(persistence, target)
    unaugmented = _unaugmented_metrics(battery)
    if not np.isfinite(np.concatenate((augmented_x.reshape(-1), augmented_fusion_x.reshape(-1), prediction, simple_sum, persistence))).all():
        raise ConvergenceProtocolViolation(f"{battery} aggregate Phase9D tensor contains NaN/Inf")
    rows = [
        {
            "sample_index": int(index), "target_capacity": float(value), "predicted_capacity": float(pred),
            "simple_sum_capacity": float(simple), "persistence_capacity": float(persist), "error": float(pred - value),
            "weight_trend": float(weight[0]), "weight_seasonal": float(weight[1]), "weight_residual": float(weight[2]),
        }
        for index, value, pred, simple, persist, weight in zip(
            val_endpoints, target, prediction, simple_sum, persistence,
            _fusion_weights(fusion, val_fusion_x), strict=True
        )
    ]
    result = {
        "split": {"n_total": split.total, "n_train": split.counts[0], "n_val": split.counts[1], "n_test_membership_only": split.counts[2]},
        "normalization": {**scaler, "synthetic_transform": "REAL_TRAIN_SCALER_ONLY__UNCLIPPED"},
        "synthetic_scale_below_0_count": below_0, "synthetic_scale_above_1_count": above_1,
        "real_train_windows": int(real_x.shape[0]), "synthetic_windows_per_set": branch_windows_per_set,
        "total_synthetic_windows": int(sum(branch_windows_per_set)), "total_augmented_train_windows": int(augmented_x.shape[0]),
        "branch_validation_component_target_audit": val_component_audit,
        "branch_convergence": branch_result,
        "fusion_training_windows": {"real": int(real_fusion_x.shape[0]), "synthetic_per_set": [int(item["fusion_windows"]) for item in synthetic_fusion_audits], "total_augmented": int(augmented_fusion_x.shape[0])},
        "fusion_convergence": fusion_result,
        "leakage_gate": {**leakage, "current_target_error_used": False, "future_validation_capacity_used": False, "test_access_count": 0, "status": "PASS"},
        "validation": {
            "label_domain": "CAPACITY_AH",
            "augmented_simple_sum": simple_metrics,
            "augmented_intelligent_fusion": intelligent_metrics,
            "persistence": persistence_metrics,
            "augmented_simple_sum_soh_metrics_percent": _soh_metrics_percent(simple_sum, target),
            "augmented_intelligent_fusion_soh_metrics_percent": _soh_metrics_percent(prediction, target),
            "unaugmented": unaugmented,
            "augmented_vs_unaugmented_simple_rmse_change": float(simple_metrics["rmse"] - unaugmented["simple_sum_rmse"]),
            "augmented_vs_unaugmented_fusion_rmse_change": float(intelligent_metrics["rmse"] - unaugmented["intelligent_fusion_rmse"]),
            "augmented_fusion_vs_persistence_rmse_change": float(intelligent_metrics["rmse"] - persistence_metrics["rmse"]),
        },
        "stl": {"period": PERIOD, "real_train": "OFFLINE_TRAIN_ONLY", "synthetic": "ISOLATED_PER_TRAJECTORY", "paper_exact_synthetic_stl_protocol": "UNRESOLVED"},
        "access": access,
        "nan_inf_count": 0,
    }
    artifact = {
        "branch_state_dicts": branch_states,
        "fusion_state_dict": fusion_state,
        "synthetic_manifest": synthetic_manifest,
    }
    return result, artifact, rows


def _fusion_weights(fusion: MinimalIntelligentFusion, val_x: np.ndarray) -> np.ndarray:
    fusion.eval()
    with torch.no_grad():
        _, weights, _, _ = fusion(torch.from_numpy(val_x))
    return weights.cpu().numpy()


def _soh_metrics_percent(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result = metrics(capacity_to_soh(prediction), capacity_to_soh(target))
    return {"rmse_percent": 100.0 * result["rmse"], "mae_percent": 100.0 * result["mae"]}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 9D Augmented Full G-MTC Convergence",
        "",
        "Tier-2 augmented convergence diagnostic only. It is not a paper-aligned final result and no outcome was used to alter a frozen configuration.",
        "",
    ]
    for battery, result in payload["results_by_battery"].items():
        val, branch, fusion = result["validation"], result["branch_convergence"], result["fusion_convergence"]
        lines += [f"## {battery}", "", "### Branch early stopping", "", "| Branch | Best epoch | Stop epoch | Best real-VAL component MSE |", "| --- | ---: | ---: | ---: |"]
        for kind in KINDS:
            item = branch["by_branch"][kind]
            lines.append(f"| {kind} | {item['best_epoch']} | {item['stop_epoch']} | {item['best_val_component_mse']:.12g} |")
        lines += ["", "### Validation capacity", "", "| Method | RMSE | MAE | R2 |", "| --- | ---: | ---: | ---: |"]
        for name, value in (("Augmented simple sum", val["augmented_simple_sum"]), ("Augmented intelligent fusion", val["augmented_intelligent_fusion"]), ("Persistence", val["persistence"])):
            lines.append(f"| {name} | {value['rmse']:.12g} | {value['mae']:.12g} | {value['r2']:.12g} |")
        lines += [
            "",
            f"FUSION_BEST_EPOCH={fusion['best_epoch']}; FUSION_STOP_EPOCH={fusion['stop_epoch']}; BEST_VAL_CAPACITY_MSE={fusion['best_val_capacity_mse']:.12g}.",
            f"AUGMENTED_VS_UNAUGMENTED_SIMPLE_RMSE_CHANGE={val['augmented_vs_unaugmented_simple_rmse_change']:.12g}; AUGMENTED_VS_UNAUGMENTED_FUSION_RMSE_CHANGE={val['augmented_vs_unaugmented_fusion_rmse_change']:.12g}; AUGMENTED_FUSION_VS_PERSISTENCE_RMSE_CHANGE={val['augmented_fusion_vs_persistence_rmse_change']:.12g}.",
            f"Mean fusion weights T/S/R={fusion['mean_weights']['trend']:.12g}/{fusion['mean_weights']['seasonal']:.12g}/{fusion['mean_weights']['residual']:.12g}; std={fusion['std_weights']['trend']:.12g}/{fusion['std_weights']['seasonal']:.12g}/{fusion['std_weights']['residual']:.12g}; max sum error={fusion['weight_sum_max_error']:.12g}.",
            "",
        ]
    lines += [
        "```text",
        "SCFGAN_RETRAINED=False",
        "SYNTHETIC_REGENERATED=False",
        "TEST_LOADED=False",
        "TEST_ACCESS_COUNT=0",
        "CURRENT_TARGET_ERROR_USED=False",
        "FUTURE_VALIDATION_CAPACITY_USED=False",
        "GRID_SEARCH_RUN=False",
        "HYPERPARAMETER_TUNING=False",
        f"PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE_GATE={payload['phase9d_augmented_full_gmtc_convergence_gate']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_ROOT.exists() or METRICS_PATH.exists() or REPORT_PATH.exists() or PREDICTIONS_PATH.exists():
        raise ConvergenceProtocolViolation("refusing to overwrite an existing Phase9D convergence artifact")
    preflight = source_preflight()
    audits, splits = validate_pairing_and_splits()
    results, artifacts, rows_by_battery = {}, {}, {}
    for battery in TARGET_BATTERIES:
        results[battery], artifacts[battery], rows_by_battery[battery] = one_battery(battery, audits, splits)
    if any(result["access"]["post_split_test_numerical_access_attempts"] or result["access"]["post_split_test_numerical_materialization_count"] for result in results.values()):
        raise ConvergenceProtocolViolation("global TEST lock failed")
    if any(result["leakage_gate"]["current_target_error_used"] or result["leakage_gate"]["future_validation_capacity_used"] for result in results.values()):
        raise ConvergenceProtocolViolation("global fusion leakage gate failed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    checkpoint_paths: dict[str, str] = {}
    for battery, artifact in artifacts.items():
        path = OUTPUT_ROOT / f"{battery}_AUGMENTED_BRANCH_AND_FUSION_BEST_CHECKPOINT.pt"
        torch.save({"phase": "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE", "battery": battery, **artifact}, path)
        checkpoint_paths[battery] = str(path.relative_to(ROOT))
    predictions_payload = {"phase": "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE", "scope": "REAL_VALIDATION_ONLY__NO_TEST_VALUES", "rows_by_battery": rows_by_battery}
    payload = {
        "phase": "PHASE9D_AUGMENTED_FULL_GMTC_CONVERGENCE",
        "authority": "EXPLICIT_HUMAN_AUTHORIZATION__TIER2_AUGMENTED_CONVERGENCE",
        "paper_exact": False,
        "protocol": {"development_split": "CHRONOLOGICAL_70_20_10", "seq_len": SEQ_LEN, "pred_len": 1, "stl_period": PERIOD, "branch_config": "PHASE9A_FROZEN", "fusion_config": "PHASE9B_MINIMAL_INTELLIGENT_FUSION_UNMODIFIED", "seed": SEED, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "optimizer": "Adam", "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "weight_decay_fusion": WEIGHT_DECAY},
        "preflight": preflight,
        "results_by_battery": results,
        "checkpoint_paths": checkpoint_paths,
        "branch_validation_selection": "REAL_VAL_EXPANDING_PREFIX_COMPONENT_MSE__CURRENT_TARGET_LABEL_ONLY__NO_FUTURE_CAPACITY",
        "scfgan_retrained": False,
        "synthetic_regenerated": False,
        "test_loaded": False,
        "test_access_count": 0,
        "current_target_error_used": False,
        "future_validation_capacity_used": False,
        "grid_search_run": False,
        "hyperparameter_tuning": False,
        "nan_inf_count": 0,
        "phase9d_augmented_full_gmtc_convergence_gate": "PASS",
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    PREDICTIONS_PATH.write_text(json.dumps(predictions_payload, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": METRICS_PATH.name, "report": REPORT_PATH.name}, indent=2))


if __name__ == "__main__":
    main()
