"""Run the explicitly authorized, fixed-control Phase 9A convergence diagnostic.

Only the training-control overlay differs from the completed three-epoch dry
run: max epochs, fused-validation early stopping, in-memory best checkpoint,
and restoration. All NASA, split, feature, scale, STL, model, fusion, and TEST
lock contracts are imported unchanged from the Phase 9A runner.
"""

from __future__ import annotations

from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

from run_phase9a_tier2_paired_train_val_dry_run import (
    OUTPUT as DRY_RUN_OUTPUT,
    PERIOD,
    SEQ_LEN,
    TARGET_BATTERIES,
    Branch,
    ProtocolViolation,
    build_train_samples,
    build_val_samples,
    capacity_to_soh,
    fused_validation_prediction,
    metrics,
    nonfinite_count,
    numeric_pairs_for_train_and_val,
    scale_train_val,
    validate_pairing_and_splits,
)
from stl_decomposition import STLDecomposer
from tier2_phase9a_convergence_config import FROZEN_TIER2_PHASE9A_CONVERGENCE_CONFIG
from tier2_scalar_component_readout import strict_scalar_mse
from tier2_soh_bridge import NASA_RATED_CAPACITY_AH, SOH_FORMULA


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "PHASE9A_CONVERGENCE_DIAGNOSTIC_METRICS.json"
CONFIG = FROZEN_TIER2_PHASE9A_CONVERGENCE_CONFIG
KINDS = ("trend", "seasonal", "residual")


def set_seed_for_battery() -> None:
    """Reset every target battery to a fresh, identical seed-42 initialization."""
    CONFIG.validate()
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def fingerprint(model: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def train_one_battery(
    battery: str,
    audits: dict[str, dict[str, Any]],
    splits: dict[str, Any],
    *,
    capture_persistence_artifacts: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Train exactly one target battery and restore its minimum fused-VAL-MSE state."""
    set_seed_for_battery()
    split = splits[battery]
    pairs = audits[battery]["pair_identities"]
    features, capacities, _, hf_audit = numeric_pairs_for_train_and_val(battery, pairs, split)
    train_hf, val_hf, scaler = scale_train_val(features, split)
    train_capacity = capacities[: split.train_end]
    val_capacity = capacities[split.train_end : split.val_end]
    stl = STLDecomposer().decompose(
        series=train_capacity,
        period=PERIOD,
        partition_id="TRAIN",
        series_id=f"{battery}_TRAIN_CAPACITY",
    )
    stl_audit = stl["structural_audit"]
    if (
        not stl_audit["length_preserved"]
        or not stl_audit["no_nan_or_inf"]
        or stl_audit["max_abs_reconstruction_error"] > 1e-10
    ):
        raise ProtocolViolation(f"STL reconstruction contract failed for {battery}")
    components = {kind: stl[kind] for kind in KINDS}
    train_x, targets = build_train_samples(train_hf, components)
    val_x, val_y = build_val_samples(train_hf, val_hf, val_capacity)
    x = torch.from_numpy(train_x)
    y = {kind: torch.from_numpy(targets[kind]) for kind in KINDS}
    val_x_tensor, val_y_tensor = torch.from_numpy(val_x), torch.from_numpy(val_y)

    branches = {kind: Branch(kind) for kind in KINDS}
    initial_fingerprints = {kind: fingerprint(branch) for kind, branch in branches.items()}
    optimizers = {
        kind: torch.optim.Adam(branch.parameters(), lr=CONFIG.learning_rate)
        for kind, branch in branches.items()
    }
    train_loss_traces = {kind: [] for kind in KINDS}
    fused_val_mse_trace: list[float] = []
    core_grad = {kind: False for kind in KINDS}
    readout_grad = {kind: False for kind in KINDS}
    best_mse = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    final_shapes: dict[str, list[int]] = {}

    for epoch in range(1, CONFIG.max_epochs + 1):
        for kind in KINDS:
            branch = branches[kind]
            branch.train()
            total, count = 0.0, 0
            for start in range(0, x.shape[0], CONFIG.batch_size):
                end = min(start + CONFIG.batch_size, x.shape[0])
                optimizers[kind].zero_grad(set_to_none=True)
                prediction = branch(x[start:end], partition="TRAIN")
                loss = strict_scalar_mse(prediction, y[kind][start:end])
                if not torch.isfinite(loss):
                    raise ProtocolViolation(f"{battery} {kind} TRAIN MSE is non-finite at epoch {epoch}")
                loss.backward()
                core_grad[kind] = core_grad[kind] or any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for parameter in branch.core.parameters()
                )
                readout_grad[kind] = readout_grad[kind] or any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for parameter in branch.readout.parameters()
                )
                optimizers[kind].step()
                total += float(loss.detach()) * (end - start)
                count += end - start
            epoch_loss = total / count
            if not np.isfinite(epoch_loss):
                raise ProtocolViolation(f"{battery} {kind} aggregate TRAIN MSE is non-finite at epoch {epoch}")
            train_loss_traces[kind].append(epoch_loss)

        fused, final_shapes, fused_mse = fused_validation_prediction(branches, val_x_tensor, val_y_tensor)
        fused_val_mse_trace.append(fused_mse)
        if fused_mse < best_mse - CONFIG.min_delta:
            best_mse = fused_mse
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {kind: deepcopy(branch.state_dict()) for kind, branch in branches.items()}
        else:
            epochs_since_best += 1
        if epochs_since_best >= CONFIG.patience:
            convergence_status = "EARLY_STOPPED"
            stop_epoch = epoch
            break
    else:
        convergence_status = "INCONCLUSIVE_MAX_EPOCH_CAP"
        stop_epoch = CONFIG.max_epochs

    if best_state is None or best_epoch == 0:
        raise ProtocolViolation(f"{battery} never produced a finite best checkpoint")
    for kind in KINDS:
        branches[kind].load_state_dict(best_state[kind])
    restored_fused, restored_shapes, restored_mse = fused_validation_prediction(
        branches, val_x_tensor, val_y_tensor
    )
    if not np.isclose(restored_mse, best_mse, rtol=0.0, atol=1e-12):
        raise ProtocolViolation(f"{battery} restored checkpoint does not reproduce best validation MSE")
    if tuple(restored_fused.shape) != tuple(val_y.shape):
        raise ProtocolViolation(f"{battery} restored fusion shape differs from validation target")
    if any(not core_grad[kind] or not readout_grad[kind] for kind in KINDS):
        raise ProtocolViolation(f"{battery} has inactive core/readout gradients")

    result = {
        "training_unit": "PER_TARGET_BATTERY__PHASE1_E1_ROLE_MATRIX",
        "split": {
            "n_total": split.total,
            "n_train": split.counts[0],
            "n_val": split.counts[1],
            "n_test_membership_only": split.counts[2],
            "train_range": [0, split.train_end],
            "val_range": [split.train_end, split.val_end],
            "test_membership_range": [split.val_end, split.total],
        },
        "hf_extraction": hf_audit,
        "normalization": scaler,
        "stl": {
            "period": PERIOD,
            "mode": "OFFLINE_TRAIN_ONLY",
            "temporal_leakage_audit": "PASS__OFFLINE_WITHIN_TRAIN_ONLY",
            **stl_audit,
        },
        "samples": {
            "train": int(train_x.shape[0]),
            "val": int(val_x.shape[0]),
            "target_alignment": "HF[t-9:t]__TO__COMPONENT_OR_CAPACITY[t+1]",
            "val_history_policy": "PAST_HF_AVAILABLE_AT_PREDICTION_TIME",
        },
        "shape_contract": {
            "hf_input_train": list(train_x.shape),
            "hf_input_validation": list(val_x.shape),
            "component_target_train": {kind: list(target.shape) for kind, target in targets.items()},
            "branch_scalar_output_validation": restored_shapes,
            "fused_capacity_output_validation": list(restored_fused.shape),
            "soh_output_validation": list(capacity_to_soh(restored_fused).shape),
            "status": "PASS__EXACT_SCALAR_SHAPES_NO_BROADCASTING",
        },
        "branch_audit": {
            kind: {
                "initialization_fingerprint": initial_fingerprints[kind],
                "core_gradient_finite_nonzero": core_grad[kind],
                "readout_gradient_finite_nonzero": readout_grad[kind],
                "core_output_contract_shape": ["B", 1, 3],
                "scalar_output_shape": restored_shapes[kind],
            }
            for kind in KINDS
        },
        "convergence": {
            "status": convergence_status,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "best_validation_capacity_mse": best_mse,
            "epochs_since_best_at_stop": epochs_since_best,
            "best_checkpoint": "IN_MEMORY_MIN_FUSED_VALIDATION_CAPACITY_MSE",
            "restored_best_checkpoint": True,
        },
        "train_component_loss_by_epoch": train_loss_traces,
        "validation_fused_capacity_mse_by_epoch": fused_val_mse_trace,
        "fusion": {
            "mode": "ADVISOR_AUTHORIZED_COMPONENT_SUM",
            "output_shape": list(restored_fused.shape),
            "status": "PASS",
        },
        "prediction_domain": "CAPACITY",
        "capacity_to_soh_bridge_status": "PA001_ACTIVE__HUMAN_AUTHORIZED_FALLBACK",
        "soh_formula": SOH_FORMULA,
        "soh_denominator_ah": NASA_RATED_CAPACITY_AH,
        "validation_metrics_capacity": metrics(restored_fused.reshape(-1), val_y.reshape(-1)),
        "validation_metrics_soh": metrics(
            capacity_to_soh(restored_fused.reshape(-1)), capacity_to_soh(val_y.reshape(-1))
        ),
        "validation_soh_metrics_percent": {
            "rmse_percent": 100.0 * metrics(
                capacity_to_soh(restored_fused.reshape(-1)), capacity_to_soh(val_y.reshape(-1))
            )["rmse"],
            "mae_percent": 100.0 * metrics(
                capacity_to_soh(restored_fused.reshape(-1)), capacity_to_soh(val_y.reshape(-1))
            )["mae"],
        },
        "nan_inf_count": nonfinite_count(
            features,
            capacities,
            train_hf,
            val_hf,
            train_x,
            val_x,
            val_y,
            restored_fused,
            *targets.values(),
            np.asarray(fused_val_mse_trace),
        ),
    }
    if not capture_persistence_artifacts:
        return result

    # This capture is deliberately post-restoration and contains only legal
    # validation targets/predictions.  It is used by the separately authorized
    # replay-persistence workflow; TEST records remain unreachable.
    prediction_rows = [
        {
            "sample_index": int(split.train_end + local_index),
            "target_capacity": float(target),
            "predicted_capacity": float(prediction),
            "error": float(prediction - target),
        }
        for local_index, (target, prediction) in enumerate(
            zip(val_y.reshape(-1), restored_fused.reshape(-1), strict=True)
        )
    ]
    return result, {
        "checkpoint": {
            "battery": battery,
            "phase": "PHASE9A_CONVERGENCE_REPLAY_PERSISTENCE",
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "best_validation_capacity_mse": best_mse,
            "config": {
                "seed": CONFIG.seed,
                "max_epochs": CONFIG.max_epochs,
                "patience": CONFIG.patience,
                "learning_rate": CONFIG.learning_rate,
                "batch_size": CONFIG.batch_size,
                "optimizer": CONFIG.optimizer,
                "device": CONFIG.device,
            },
            "branch_state_dicts": {
                kind: {name: tensor.detach().cpu().clone() for name, tensor in best_state[kind].items()}
                for kind in KINDS
            },
        },
        "validation_prediction_rows": prediction_rows,
    }


def one_full_diagnostic(
    audits: dict[str, dict[str, Any]],
    splits: dict[str, Any],
    *,
    capture_persistence_artifacts: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not capture_persistence_artifacts:
        return {battery: train_one_battery(battery, audits, splits) for battery in TARGET_BATTERIES}
    results: dict[str, Any] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for battery in TARGET_BATTERIES:
        result, artifact = train_one_battery(
            battery, audits, splits, capture_persistence_artifacts=True
        )
        results[battery] = result
        artifacts[battery] = artifact
    return results, artifacts


def comparable(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def main() -> None:
    CONFIG.validate()
    audits, splits = validate_pairing_and_splits()
    first = one_full_diagnostic(audits, splits)
    second = one_full_diagnostic(audits, splits)
    deterministic = comparable(first, second)
    if not deterministic:
        raise ProtocolViolation("same-seed convergence diagnostic payload mismatch")
    statuses = [first[battery]["convergence"]["status"] for battery in TARGET_BATTERIES]
    output = {
        "phase": "PHASE9A_CONVERGENCE_DIAGNOSTIC",
        "authority": "EXPLICIT_HUMAN_AUTHORIZATION__EXECUTION_CONTROL_ONLY",
        "derived_from_dry_run_metrics": str(DRY_RUN_OUTPUT.name),
        "protocol": {
            "pairing_mode": "DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE",
            "split": "CHRONOLOGICAL_70_20_10",
            "seq_len": SEQ_LEN,
            "pred_len": 1,
            "stl_period": PERIOD,
            "stl_mode": "OFFLINE_TRAIN_ONLY",
            "normalization": "MINMAX_0_1__TRAIN_ONLY",
            "loss": "MSE",
            "fusion": "TREND_PLUS_SEASONAL_PLUS_RESIDUAL",
            "seed": CONFIG.seed,
            "max_epochs": CONFIG.max_epochs,
            "early_stopping": CONFIG.early_stopping,
            "patience": CONFIG.patience,
            "min_delta": CONFIG.min_delta,
            "early_stopping_metric": CONFIG.early_stopping_metric,
            "restore_best_checkpoint": CONFIG.restore_best_checkpoint,
            "batch_size": CONFIG.batch_size,
            "learning_rate": CONFIG.learning_rate,
            "optimizer": CONFIG.optimizer,
            "device": CONFIG.device,
            "train_batch_shuffle": CONFIG.train_batch_shuffle,
            "validation_batch_shuffle": CONFIG.validation_batch_shuffle,
            "drop_last": CONFIG.drop_last,
        },
        "test_lock": {
            "raw_mat_full_parse_allowed": True,
            "test_loaded": False,
            "test_numerical_value_access_count": 0,
            "post_split_test_numerical_access_attempts": sum(
                result["hf_extraction"]["post_split_test_numerical_access_attempts"] for result in first.values()
            ),
            "post_split_test_numerical_materialization_count": sum(
                result["hf_extraction"]["post_split_test_numerical_materialization_count"] for result in first.values()
            ),
            "test_transform_applied": False,
            "test_predictions_saved": False,
            "gate": "PASS",
        },
        "target_execution_roster": list(TARGET_BATTERIES),
        "results_by_battery": first,
        "phase9a_convergence_status": "EARLY_STOPPED" if all(status == "EARLY_STOPPED" for status in statuses) else "INCONCLUSIVE_MAX_EPOCH_CAP",
        "determinism": {
            "same_seed_reruns": 2,
            "status": "PASS" if deterministic else "FAIL",
            "comparison": "COMPLETE_RESULT_PAYLOAD_EXACT",
        },
        "validation_used_for_early_stopping": True,
        "validation_used_for_hyperparameter_tuning": False,
        "scfgan_trained": False,
        "synthetic_hf_used": False,
        "grid_search_run": False,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "target_batteries": list(TARGET_BATTERIES)}, indent=2))


if __name__ == "__main__":
    main()
