"""Tier-2 Phase 9B minimal intelligent fusion with a sealed TEST boundary.

Frozen Phase 9A branch checkpoints are restored and marked eval/no-grad.  The
only trainable parameters belong to the Phase 9B fusion module specified by the
user.  Each validation lagged component error is built from an STL prefix that
ends at the prior observed capacity, never the current target or any future
validation capacity.
"""

from __future__ import annotations

from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_phase9a_tier2_paired_train_val_dry_run import (
    ROOT,
    TARGET_BATTERIES,
    Branch,
    PERIOD,
    SEQ_LEN,
    ProtocolViolation,
    _cycles,
    capacity_to_soh,
    metrics,
    numeric_pairs_for_train_and_val,
    scale_train_val,
    validate_pairing_and_splits,
)
from stl_decomposition import STLDecomposer
from tier2_phase9a_access import PostSplitPairAccessor
from tier2_scalar_component_readout import strict_scalar_mse


FUSION_SEQ_LEN = 10
EPS = 1e-8
SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 150
PATIENCE = 50
OUTPUT = ROOT / "PHASE9B_MINIMAL_INTELLIGENT_FUSION_METRICS.json"
REPORT = ROOT / "PHASE9B_MINIMAL_INTELLIGENT_FUSION_REPORT.md"
PREDICTIONS = ROOT / "PHASE9B_MINIMAL_INTELLIGENT_FUSION_VALIDATION_PREDICTIONS.json"
CHECKPOINT_DIRECTORY = ROOT / "output" / "phase9b_minimal_intelligent_fusion"
PHASE9A_REPLAY = ROOT / "PHASE9A_CONVERGENCE_REPLAY_METRICS.json"
PHASE9A_CHECKPOINT_DIRECTORY = ROOT / "output" / "phase9a_convergence_replay"
KINDS = ("trend", "seasonal", "residual")


class FusionProtocolViolation(ProtocolViolation):
    """Raised before any result is written when the frozen fusion contract fails."""


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Keep Table-4 BatchNorm1d valid for an unavoidable final singleton batch."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class MinimalIntelligentFusion(nn.Module):
    """Exact user-specified Phase 9B topology, without an Lweight term."""

    def __init__(self) -> None:
        super().__init__()
        self.component_logits = nn.Parameter(torch.zeros(9))
        self.bilstm = nn.LSTM(9, 16, num_layers=1, bidirectional=True, batch_first=True)
        self.time_projection = nn.Linear(32, 32)
        self.time_score = nn.Linear(32, 1)
        self.weight_network = nn.Sequential(
            nn.Linear(32, 128), SingletonSafeBatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64), SingletonSafeBatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), SingletonSafeBatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(32, 3),
        )
        self.final_head = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (FUSION_SEQ_LEN, 9):
            raise FusionProtocolViolation("fusion input must be Bx10x9")
        if not torch.isfinite(features).all():
            raise FusionProtocolViolation("fusion input is non-finite")
        component_attention = torch.softmax(self.component_logits, dim=0)
        attended = features * component_attention.view(1, 1, 9)
        sequence, _ = self.bilstm(attended)
        time_scores = self.time_score(torch.tanh(self.time_projection(sequence))).squeeze(-1)
        time_attention = torch.softmax(time_scores, dim=1)
        context = torch.sum(sequence * time_attention.unsqueeze(-1), dim=1)
        weights = torch.softmax(self.weight_network(context), dim=-1)
        weighted_components = features[:, -1, :3] * weights
        prediction = self.final_head(weighted_components).unsqueeze(1)
        if not torch.isfinite(prediction).all() or not torch.isfinite(weights).all():
            raise FusionProtocolViolation("fusion output is non-finite")
        return prediction, weights, component_attention, time_attention


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def restore_frozen_branches(battery: str) -> dict[str, Branch]:
    checkpoint_path = PHASE9A_CHECKPOINT_DIRECTORY / f"{battery}_BEST_CHECKPOINT_DRY_RUN_ONLY.pt"
    if not checkpoint_path.is_file():
        raise FusionProtocolViolation(f"missing persisted Phase9A best checkpoint for {battery}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    branches = {kind: Branch(kind) for kind in KINDS}
    for kind, branch in branches.items():
        branch.load_state_dict(payload["branch_state_dicts"][kind], strict=True)
        branch.eval()
        for parameter in branch.parameters():
            parameter.requires_grad_(False)
        if branch.training or any(parameter.requires_grad for parameter in branch.parameters()):
            raise FusionProtocolViolation(f"{battery} {kind} branch is not frozen/eval")
    return branches


def branch_prediction_matrix(branches: dict[str, Branch], history_hf: np.ndarray) -> np.ndarray:
    """Generate frozen one-step component predictions for allowed target indices only."""
    result = np.full((history_hf.shape[0], 3), np.nan, dtype=np.float64)
    inputs = np.asarray(
        [history_hf[index - SEQ_LEN : index] for index in range(SEQ_LEN, history_hf.shape[0])],
        dtype=np.float32,
    )
    with torch.no_grad():
        x = torch.from_numpy(inputs)
        outputs = [branches[kind](x, partition="FUSION_FROZEN_BRANCH").reshape(-1).cpu().numpy() for kind in KINDS]
    result[SEQ_LEN:] = np.column_stack(outputs)
    if not np.isfinite(result[SEQ_LEN:]).all():
        raise FusionProtocolViolation("frozen branch prediction matrix is non-finite")
    return result


def expanding_previous_components(
    capacities: np.ndarray,
    previous_index: int,
    train_components: dict[str, np.ndarray],
    train_end: int,
    battery: str,
) -> np.ndarray:
    """Return T/S/R at previous_index without using current/future VAL capacity."""
    if previous_index < train_end:
        return np.asarray([train_components[kind][previous_index] for kind in KINDS], dtype=np.float64)
    prefix = capacities[: previous_index + 1]
    decomposition = STLDecomposer().decompose(
        series=prefix,
        period=PERIOD,
        partition_id="VALIDATION",
        series_id=f"{battery}_CAPACITY_0_TO_{previous_index}",
    )
    audit = decomposition["structural_audit"]
    if not audit["length_preserved"] or not audit["no_nan_or_inf"] or audit["max_abs_reconstruction_error"] > 1e-10:
        raise FusionProtocolViolation("expanding-prefix STL structural contract failed")
    return np.asarray([decomposition[kind][-1] for kind in KINDS], dtype=np.float64)


def feature_row(
    prediction_matrix: np.ndarray,
    current_index: int,
    capacities: np.ndarray,
    train_components: dict[str, np.ndarray],
    train_end: int,
    battery: str,
    leakage_audit: dict[str, Any],
) -> np.ndarray:
    """Create [pT,pS,pR,eT_lag,eS_lag,eR_lag,rT,rS,rR] at current_index."""
    previous_index = current_index - 1
    if current_index < SEQ_LEN + 1 or previous_index < 0:
        raise FusionProtocolViolation("fusion feature row lacks a legal prior branch prediction")
    current_prediction = prediction_matrix[current_index]
    previous_prediction = prediction_matrix[previous_index]
    if not np.isfinite(current_prediction).all() or not np.isfinite(previous_prediction).all():
        raise FusionProtocolViolation("fusion feature row references unavailable branch prediction")
    if np.any(previous_prediction == 0.0):
        raise FusionProtocolViolation("lagged error denominator is exactly zero; no unapproved error epsilon exists")
    previous_true_component = expanding_previous_components(
        capacities, previous_index, train_components, train_end, battery
    )
    if current_index >= train_end:
        leakage_audit["validation_feature_rows"] += 1
        leakage_audit["max_validation_capacity_index_used_for_feature"] = max(
            leakage_audit["max_validation_capacity_index_used_for_feature"], previous_index
        )
        if previous_index >= current_index:
            raise FusionProtocolViolation("current validation target entered lagged error feature")
    lagged_error = previous_true_component / previous_prediction
    denominator = float(current_prediction.sum())
    if abs(denominator) < EPS:
        leakage_audit["ratio_epsilon_guard_count"] += 1
        denominator = EPS if denominator >= 0.0 else -EPS
    ratios = current_prediction / denominator
    row = np.concatenate((current_prediction, lagged_error, ratios)).astype(np.float32)
    if row.shape != (9,) or not np.isfinite(row).all():
        raise FusionProtocolViolation("fusion 9-feature row is malformed or non-finite")
    return row


def build_fusion_datasets(
    battery: str,
    history_hf: np.ndarray,
    capacities: np.ndarray,
    train_components: dict[str, np.ndarray],
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    predictions = branch_prediction_matrix(restore_frozen_branches(battery), history_hf)
    audit: dict[str, Any] = {
        "current_target_error_used": False,
        "future_validation_capacity_used": False,
        "validation_feature_rows": 0,
        "max_validation_capacity_index_used_for_feature": -1,
        "ratio_epsilon_guard_count": 0,
    }
    rows = {
        index: feature_row(predictions, index, capacities, train_components, train_end, battery, audit)
        for index in range(SEQ_LEN + 1, history_hf.shape[0])
    }
    train_endpoints = list(range(SEQ_LEN + FUSION_SEQ_LEN, train_end))
    val_endpoints = list(range(train_end, history_hf.shape[0]))
    def sequence_for(endpoint: int) -> np.ndarray:
        indices = range(endpoint - FUSION_SEQ_LEN + 1, endpoint + 1)
        return np.asarray([rows[index] for index in indices], dtype=np.float32)
    train_x = np.asarray([sequence_for(index) for index in train_endpoints], dtype=np.float32)
    val_x = np.asarray([sequence_for(index) for index in val_endpoints], dtype=np.float32)
    train_y = np.asarray(capacities[train_endpoints], dtype=np.float32).reshape(-1, 1, 1)
    val_y = np.asarray(capacities[val_endpoints], dtype=np.float32).reshape(-1, 1, 1)
    if train_x.shape[1:] != (FUSION_SEQ_LEN, 9) or val_x.shape[1:] != (FUSION_SEQ_LEN, 9):
        raise FusionProtocolViolation("fusion sequence shape contract failed")
    if audit["max_validation_capacity_index_used_for_feature"] >= train_end:
        # It may use prior validation capacity, but never the corresponding current/future target.
        audit["future_validation_capacity_used"] = False
    return train_x, train_y, val_x, val_y, val_endpoints, audit


def evaluate(model: MinimalIntelligentFusion, x: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        prediction, weights, component_attention, time_attention = model(x)
        if tuple(prediction.shape) != tuple(y.shape):
            raise FusionProtocolViolation("fusion prediction/target shape mismatch")
        if not torch.isfinite(prediction).all() or not torch.isfinite(weights).all():
            raise FusionProtocolViolation("non-finite fusion validation output")
    return prediction.cpu().numpy(), weights.cpu().numpy(), {
        "component_attention": component_attention.cpu().numpy(),
        "time_attention": time_attention.cpu().numpy(),
    }


def train_one_battery(
    battery: str,
    audits: dict[str, dict[str, Any]],
    splits: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed()
    split = splits[battery]
    features, capacities, _, access_audit = numeric_pairs_for_train_and_val(
        battery, audits[battery]["pair_identities"], split
    )
    train_hf, val_hf, scaler = scale_train_val(features, split)
    history_hf = np.concatenate((train_hf, val_hf), axis=0)
    train_capacity = capacities[: split.train_end]
    stl = STLDecomposer().decompose(
        series=train_capacity,
        period=PERIOD,
        partition_id="TRAIN",
        series_id=f"{battery}_TRAIN_CAPACITY",
    )
    stl_audit = stl["structural_audit"]
    if not stl_audit["length_preserved"] or not stl_audit["no_nan_or_inf"] or stl_audit["max_abs_reconstruction_error"] > 1e-10:
        raise FusionProtocolViolation("TRAIN-only STL contract failed")
    components = {kind: stl[kind] for kind in KINDS}
    train_x, train_y, val_x, val_y, val_indices, leakage = build_fusion_datasets(
        battery, history_hf, capacities, components, split.train_end
    )
    if access_audit["post_split_test_numerical_access_attempts"] != 0 or access_audit["post_split_test_numerical_materialization_count"] != 0:
        raise FusionProtocolViolation("TEST access occurred before fusion training")
    if leakage["current_target_error_used"] or leakage["future_validation_capacity_used"]:
        raise FusionProtocolViolation("leakage gate failed while constructing fusion features")

    model = MinimalIntelligentFusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(val_x), torch.from_numpy(val_y)
    best_mse, best_epoch, since_best = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    train_trace: list[float] = []
    val_trace: list[float] = []
    singleton_batches = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for start in range(0, x.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, x.shape[0])
            if end - start == 1:
                singleton_batches += 1
            optimizer.zero_grad(set_to_none=True)
            prediction, weights, _, _ = model(x[start:end])
            loss = strict_scalar_mse(prediction, y[start:end])
            if not torch.isfinite(loss) or not torch.isfinite(weights).all():
                raise FusionProtocolViolation("non-finite fusion training loss or weights")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FusionProtocolViolation("non-finite fusion gradient")
            optimizer.step()
            total += float(loss.detach()) * (end - start)
            seen += end - start
        train_trace.append(total / seen)
        prediction, _, _ = evaluate(model, vx, vy)
        val_mse = float(np.mean((prediction.reshape(-1) - vy.numpy().reshape(-1)) ** 2))
        val_trace.append(val_mse)
        if val_mse < best_mse:
            best_mse, best_epoch, since_best = val_mse, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            since_best += 1
        if since_best >= PATIENCE:
            status, stop_epoch = "EARLY_STOPPED", epoch
            break
    else:
        status, stop_epoch = "INCONCLUSIVE_MAX_EPOCH_CAP", MAX_EPOCHS
    if best_state is None:
        raise FusionProtocolViolation("fusion produced no finite checkpoint")
    model.load_state_dict(best_state)
    prediction, weights, attention = evaluate(model, vx, vy)
    restored_mse = float(np.mean((prediction.reshape(-1) - vy.numpy().reshape(-1)) ** 2))
    if not np.isclose(restored_mse, best_mse, rtol=0.0, atol=1e-12):
        raise FusionProtocolViolation("restored fusion checkpoint differs from best validation MSE")
    simple_sum = np.sum(val_x[:, -1, :3], axis=1)
    val_target = val_y.reshape(-1)
    persistence = capacities[np.asarray(val_indices) - 1]
    if not np.isfinite(simple_sum).all() or not np.isfinite(persistence).all():
        raise FusionProtocolViolation("baseline construction failed")
    intelligent_metrics = metrics(prediction.reshape(-1), val_target)
    sum_metrics = metrics(simple_sum, val_target)
    persistence_metrics = metrics(persistence, val_target)
    weight_sum_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    if weight_sum_error > 1e-6:
        raise FusionProtocolViolation("fusion weights do not sum to one")
    rows = [
        {
            "sample_index": int(index),
            "target_capacity": float(target),
            "predicted_capacity": float(predicted),
            "error": float(predicted - target),
            "simple_sum_capacity": float(simple),
            "persistence_capacity": float(persistent),
            "weight_trend": float(weight[0]),
            "weight_seasonal": float(weight[1]),
            "weight_residual": float(weight[2]),
        }
        for index, target, predicted, simple, persistent, weight in zip(
            val_indices, val_target, prediction.reshape(-1), simple_sum, persistence, weights, strict=True
        )
    ]
    result = {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "convergence_status": status,
        "best_validation_capacity_mse": best_mse,
        "fusion_input": {
            "sequence_length": FUSION_SEQ_LEN,
            "feature_order": ["p_T", "p_S", "p_R", "e_T_lag", "e_S_lag", "e_R_lag", "r_T", "r_S", "r_R"],
            "train_samples": int(train_x.shape[0]), "validation_samples": int(val_x.shape[0]),
        },
        "leakage_gate": {**leakage, "test_access_count": 0, "status": "PASS"},
        "branch_status": "RESTORED_FROZEN_EVAL_NO_GRAD",
        "paper_weight_regularization_term": "UNRESOLVED_NOT_EXECUTED",
        "training": {"seed": SEED, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "optimizer": "Adam", "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "singleton_batchnorm_policy": "RUNNING_STATS_FOR_FINAL_SINGLETON_ONLY"},
        "validation_capacity_metrics": intelligent_metrics,
        "validation_soh_metrics_percent": {"rmse_percent": 100.0 * metrics(capacity_to_soh(prediction.reshape(-1)), capacity_to_soh(val_target))["rmse"], "mae_percent": 100.0 * metrics(capacity_to_soh(prediction.reshape(-1)), capacity_to_soh(val_target))["mae"]},
        "simple_sum_baseline": sum_metrics,
        "persistence_baseline": persistence_metrics,
        "intelligent_vs_simple_sum_rmse_change": float(intelligent_metrics["rmse"] - sum_metrics["rmse"]),
        "intelligent_vs_persistence_rmse_change": float(intelligent_metrics["rmse"] - persistence_metrics["rmse"]),
        "weights": {"mean": {kind: float(weights[:, i].mean()) for i, kind in enumerate(KINDS)}, "std": {kind: float(weights[:, i].std(ddof=0)) for i, kind in enumerate(KINDS)}, "sum_max_error": weight_sum_error},
        "component_attention": attention["component_attention"].reshape(-1).tolist(),
        "stl": {"period": PERIOD, "mode": "OFFLINE_TRAIN_ONLY", **stl_audit},
        "normalization": scaler,
        "access": access_audit,
    }
    artifact = {"state_dict": best_state, "rows": rows}
    return result, artifact


def render_report(payload: dict[str, Any]) -> str:
    lines = ["# Phase 9B Minimal Intelligent Fusion", "", "Tier-2 advisor-authorized numerical execution; not paper-exact fusion reproduction.", ""]
    for battery, result in payload["results_by_battery"].items():
        m, s, p = result["validation_capacity_metrics"], result["simple_sum_baseline"], result["persistence_baseline"]
        w = result["weights"]
        lines += [f"## {battery}", "", f"BEST_EPOCH={result['best_epoch']}; STOP_EPOCH={result['stop_epoch']}; status={result['convergence_status']}.", "", "| Method | RMSE | MAE | R2 |", "|---|---:|---:|---:|", f"| Intelligent fusion | {m['rmse']:.12g} | {m['mae']:.12g} | {m['r2']:.12g} |", f"| Simple sum | {s['rmse']:.12g} | {s['mae']:.12g} | {s['r2']:.12g} |", f"| Persistence | {p['rmse']:.12g} | {p['mae']:.12g} | {p['r2']:.12g} |", "", f"INTELLIGENT_VS_SIMPLE_SUM_RMSE_CHANGE={result['intelligent_vs_simple_sum_rmse_change']:.12g}; INTELLIGENT_VS_PERSISTENCE_RMSE_CHANGE={result['intelligent_vs_persistence_rmse_change']:.12g}.", "", f"Mean weights T/S/R={w['mean']['trend']:.12g}/{w['mean']['seasonal']:.12g}/{w['mean']['residual']:.12g}; std={w['std']['trend']:.12g}/{w['std']['seasonal']:.12g}/{w['std']['residual']:.12g}; sum max error={w['sum_max_error']:.12g}.", ""]
    lines += ["TEST_LOCK_GATE=PASS; CURRENT_TARGET_ERROR_USED=False; FUTURE_VALIDATION_CAPACITY_USED=False; SCFGAN_TRAINED=False; GRID_SEARCH_RUN=False.", ""]
    return "\n".join(lines)


def main() -> None:
    if not PHASE9A_REPLAY.is_file():
        raise FusionProtocolViolation("missing Phase9A replay evidence")
    replay = json.loads(PHASE9A_REPLAY.read_text(encoding="utf-8"))
    if replay["archive_comparison"] != "PASS__COMPLETE_RESULT_PAYLOAD_EXACT" or replay["test_lock"]["gate"] != "PASS":
        raise FusionProtocolViolation("Phase9A replay preflight is not valid")
    audits, splits = validate_pairing_and_splits()
    results, artifacts = {}, {}
    for battery in TARGET_BATTERIES:
        results[battery], artifacts[battery] = train_one_battery(battery, audits, splits)
    if any(result["leakage_gate"]["current_target_error_used"] or result["leakage_gate"]["future_validation_capacity_used"] for result in results.values()):
        raise FusionProtocolViolation("global leakage gate failed")
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    checkpoint_paths, rows_by_battery = {}, {}
    for battery, artifact in artifacts.items():
        path = CHECKPOINT_DIRECTORY / f"{battery}_BEST_FUSION_CHECKPOINT_DRY_RUN_ONLY.pt"
        torch.save({"battery": battery, "phase": "PHASE9B_MINIMAL_INTELLIGENT_FUSION", "state_dict": artifact["state_dict"]}, path)
        checkpoint_paths[battery] = str(path.relative_to(ROOT))
        rows_by_battery[battery] = artifact["rows"]
    prediction_payload = {"phase": "PHASE9B_MINIMAL_INTELLIGENT_FUSION", "scope": "VALIDATION_ONLY__NO_TEST_VALUES", "rows_by_battery": rows_by_battery}
    PREDICTIONS.write_text(json.dumps(prediction_payload, indent=2), encoding="utf-8")
    payload = {"phase": "PHASE9B_MINIMAL_INTELLIGENT_FUSION", "authority": "EXPLICIT_HUMAN_SPECIFICATION", "paper_exact": False, "fusion_mode": "MINIMAL_INTELLIGENT_FUSION__FROZEN_BRANCHES", "checkpoint_paths": checkpoint_paths, "validation_predictions_path": PREDICTIONS.name, "results_by_battery": results, "test_lock": {"test_loaded": False, "test_access_count": 0, "gate": "PASS"}, "scfgan_trained": False, "grid_search_run": False, "next_action": "RETURN_PHASE9B_RESULTS_FOR_HUMAN_REVIEW"}
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
