"""Fail-closed Tier 2 Phase 9A paired TRAIN/VAL numerical dry-run.

This runner implements only the frozen, advisor-authorized fallback protocol.
It deliberately accesses numerical fields only for paired TRAIN and VALIDATION
records.  TEST pair membership is calculated from metadata, but no TEST numeric
field is read, transformed, predicted, or reported.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

from audit_nasa_discharge_centered_pairing import BATTERIES, RAW, audit
from itransformer_structural import ITransformerStructuralConfig, ParameterizedITransformerStructural
from stl_decomposition import STLDecomposer
from tier2_hf_extraction import HFExtractionError, extract_nasa_hf
from tier2_phase9a_access import PostSplitPairAccessor, TestLockViolation
from tier2_phase9a_config import FROZEN_TIER2_PHASE9A_CONFIG
from tier2_scalar_component_readout import (
    LinearScalarComponentReadout,
    ScalarComponentReadoutConfig,
    strict_scalar_mse,
)
from timesnet_structural import ParameterizedStandardTimesNet, TimesNetStructuralConfig
from tier2_soh_bridge import NASA_RATED_CAPACITY_AH, SOH_FORMULA, capacity_to_soh


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "PHASE9A_DRY_RUN_METRICS.json"
TARGET_BATTERIES = ("B0007", "B0018")  # Phase 1 E1 target SOH-estimation roles.
EXPECTED_PAIR_COUNTS = {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
EXPECTED_SPLITS = {"B0005": (117, 34, 17), "B0006": (117, 34, 17), "B0007": (117, 34, 17), "B0018": (92, 26, 14)}
SEQ_LEN, PRED_LEN, PERIOD = 10, 1, 7
RUN_CONFIG = FROZEN_TIER2_PHASE9A_CONFIG


class ProtocolViolation(RuntimeError):
    """Raised for any frozen-contract or TEST-lock violation."""


@dataclass(frozen=True)
class Split:
    train_end: int
    val_end: int
    total: int

    @property
    def counts(self) -> tuple[int, int, int]:
        return self.train_end, self.val_end - self.train_end, self.total - self.val_end


def set_seed() -> None:
    RUN_CONFIG.validate()
    random.seed(RUN_CONFIG.seed)
    np.random.seed(RUN_CONFIG.seed)
    torch.manual_seed(RUN_CONFIG.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def split_for(n: int) -> Split:
    train_end = math.floor(0.70 * n)
    val_end = math.floor(0.90 * n)
    return Split(train_end, val_end, n)


def _cycles(battery: str) -> list[Any]:
    """Authorize only raw MAT parsing; post-split field access is guarded separately."""
    path = next(RAW.rglob(f"{battery}.mat"), None)
    if path is None:
        raise ProtocolViolation(f"official raw file missing for {battery}")
    root = loadmat(path, squeeze_me=True, struct_as_record=False)[battery]
    cycles = root.cycle
    return list(cycles) if hasattr(cycles, "__len__") else [cycles]


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ProtocolViolation(f"{name} is empty or non-finite")
    return values


def extract_frozen_hf(charge: Any) -> tuple[np.ndarray, str]:
    """Use the authorized Eq.(1)–Eq.(3) NASA endpoint map."""
    try:
        result = extract_nasa_hf(charge)
    except HFExtractionError as error:
        raise ProtocolViolation(str(error)) from error
    return np.asarray((result.hf1_seconds, result.hf2_seconds, result.hf3_seconds), dtype=np.float64), result.t_imin_provenance


def extract_capacity(discharge: Any) -> float:
    capacity = _finite_vector(getattr(discharge.data, "Capacity"), name="Capacity")
    if capacity.size != 1:
        raise ProtocolViolation("paired discharge Capacity is not scalar")
    return float(capacity[0])


def validate_pairing_and_splits() -> tuple[dict[str, dict[str, Any]], dict[str, Split]]:
    audits: dict[str, dict[str, Any]] = {}
    splits: dict[str, Split] = {}
    for battery in BATTERIES:
        result = audit(battery)
        if result["n_valid_pairs"] != EXPECTED_PAIR_COUNTS[battery] or not result["one_to_one"]:
            raise ProtocolViolation(f"pairing contract violation for {battery}")
        split = split_for(result["n_valid_pairs"])
        if split.counts != EXPECTED_SPLITS[battery]:
            raise ProtocolViolation(f"chronological split contract violation for {battery}")
        audits[battery] = result
        splits[battery] = split
    return audits, splits


def numeric_pairs_for_train_and_val(battery: str, pairs: list[dict[str, Any]], split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Extract only TRAIN/VAL pair numeric fields; TEST indices are rejected before access."""
    accessor = PostSplitPairAccessor(_cycles(battery), split.train_end, split.val_end)
    hf, capacity, pair_indices = [], [], []
    t_imin_fallback_count = 0
    for pair_index, pair in enumerate(pairs):
        if pair_index >= split.val_end:
            continue  # TEST membership is never dereferenced numerically.
        charge_index = int(pair["charge_record_index"])
        try:
            charge, discharge = accessor.records_for_pair(pair_index, pair)
            feature_vector, endpoint_provenance = extract_frozen_hf(charge)
            hf.append(feature_vector)
        except ProtocolViolation as error:
            partition = "TRAIN" if pair_index < split.train_end else "VALIDATION"
            raise ProtocolViolation(
                f"{battery} {partition} pair_index={pair_index} charge_record_index={charge_index}: {error}"
            ) from error
        except TestLockViolation as error:
            raise ProtocolViolation(f"{battery} attempted TEST numerical access") from error
        t_imin_fallback_count += int(endpoint_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
        capacity.append(extract_capacity(discharge))
        pair_indices.append(pair_index)
    expected = split.val_end
    if len(hf) != expected or pair_indices != list(range(expected)):
        raise ProtocolViolation(f"TRAIN/VAL numerical pair extraction mismatch for {battery}")
    return np.asarray(hf), np.asarray(capacity), np.asarray(pair_indices), {
        "train_hf_pair_count": split.train_end,
        "val_hf_pair_count": split.val_end - split.train_end,
        "hf_extraction_failure_count": 0,
        "t_imin_final_timestamp_fallback_count": t_imin_fallback_count,
        "raw_mat_full_parse_allowed": True,
        "post_split_test_numerical_materialization_count": accessor.post_split_test_numerical_materialization_count,
    }


def scale_train_val(features: np.ndarray, split: Split) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = features[: split.train_end]
    val = features[split.train_end : split.val_end]
    low, high = train.min(axis=0), train.max(axis=0)
    if np.any(high <= low):
        raise ProtocolViolation("TRAIN-only MinMax scaler has a zero range")
    train_scaled = (train - low) / (high - low)
    val_unclipped = (val - low) / (high - low)
    val_scaled = np.clip(val_unclipped, 0.0, 1.0)
    return train_scaled, val_scaled, {
        "normalization": "MINMAX_0_1",
        "fit_scope": "TRAIN_ONLY",
        "clipping_policy": "CLIP_TO_0_1__RESULT_BLIND_ENGINEERING_RULE",
        "train_min_max_by_feature": {
            f"HF{index + 1}": {"min": float(low[index]), "max": float(high[index])} for index in range(3)
        },
        "val_below_train_range_count": {f"HF{index + 1}": int((val[:, index] < low[index]).sum()) for index in range(3)},
        "val_above_train_range_count": {f"HF{index + 1}": int((val[:, index] > high[index]).sum()) for index in range(3)},
    }


def build_train_samples(train_hf: np.ndarray, components: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    inputs = np.asarray([train_hf[target - SEQ_LEN : target] for target in range(SEQ_LEN, train_hf.shape[0])], dtype=np.float32)
    targets = {
        name: np.asarray(component[SEQ_LEN:], dtype=np.float32).reshape(-1, 1, 1)
        for name, component in components.items()
    }
    if inputs.shape[0] == 0 or any(target.shape[0] != inputs.shape[0] for target in targets.values()):
        raise ProtocolViolation("TRAIN 10-to-1 sample construction failed")
    return inputs, targets


def build_val_samples(train_hf: np.ndarray, val_hf: np.ndarray, val_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    history = np.concatenate((train_hf, val_hf), axis=0)
    start = train_hf.shape[0]
    inputs = np.asarray([history[target - SEQ_LEN : target] for target in range(start, start + val_hf.shape[0])], dtype=np.float32)
    targets = np.asarray(val_capacity, dtype=np.float32).reshape(-1, 1, 1)
    if inputs.shape[0] != targets.shape[0] or inputs.shape[1:] != (SEQ_LEN, 3):
        raise ProtocolViolation("VAL rolling one-step construction failed")
    return inputs, targets


class Branch(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind in {"trend", "seasonal"}:
            self.core = ParameterizedStandardTimesNet(
                TimesNetStructuralConfig(SEQ_LEN, PRED_LEN, 3, 2, 64, 128, 1, 2, 0.1, False, False, "STANDARD_REFERENCE_CORE")
            )
        elif kind == "residual":
            self.core = ParameterizedITransformerStructural(
                ITransformerStructuralConfig(SEQ_LEN, PRED_LEN, 3, 64, 4, 1, 128, 0.1, "gelu", False, False)
            )
        else:
            raise ProtocolViolation(f"unknown branch {kind}")
        self.readout = LinearScalarComponentReadout(ScalarComponentReadoutConfig(3, 1, 1, True, "NONE"))
        self.kind = kind

    def forward(self, x: torch.Tensor, *, partition: str) -> torch.Tensor:
        core = self.core(x, provenance=f"TIER2_{partition}_HF_WINDOW")
        return self.readout(core)


def fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def train_branch(kind: str, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[nn.Module, list[float], dict[str, Any]]:
    model = Branch(kind)
    initial = fingerprint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=RUN_CONFIG.learning_rate)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    losses: list[float] = []
    core_grad = readout_grad = False
    model.train()
    for _ in range(RUN_CONFIG.dry_run_epochs):
        total, count = 0.0, 0
        for start in range(0, x.shape[0], RUN_CONFIG.batch_size):
            end = min(start + RUN_CONFIG.batch_size, x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[start:end], partition="TRAIN")
            loss = strict_scalar_mse(prediction, y[start:end])
            if not torch.isfinite(loss):
                raise ProtocolViolation(f"{kind} training loss is non-finite")
            loss.backward()
            core_grad = core_grad or any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.core.parameters())
            readout_grad = readout_grad or any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.readout.parameters())
            optimizer.step()
            total += float(loss.detach()) * (end - start)
            count += end - start
        losses.append(total / count)
    if not (core_grad and readout_grad):
        raise ProtocolViolation(f"{kind} core/readout gradient path is inactive")
    model.eval()
    with torch.no_grad():
        val_prediction = model(torch.from_numpy(val_x), partition="VALIDATION")
    if not torch.isfinite(val_prediction).all() or tuple(val_prediction.shape) != (val_x.shape[0], 1, 1):
        raise ProtocolViolation(f"{kind} validation prediction contract failed")
    return model, losses, {"initialization_fingerprint": initial, "core_gradient_finite_nonzero": core_grad, "readout_gradient_finite_nonzero": readout_grad}


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error**2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    if not np.isfinite(mse) or sst == 0.0:
        raise ProtocolViolation("validation metric is non-finite or R2 is undefined")
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error))), "r2": float(1.0 - np.sum(error**2) / sst)}


def one_run(audits: dict[str, dict[str, Any]], splits: dict[str, Split]) -> dict[str, Any]:
    set_seed()
    results: dict[str, Any] = {}
    for battery in TARGET_BATTERIES:
        split = splits[battery]
        pairs = audits[battery]["pair_identities"]
        features, capacities, _, hf_audit = numeric_pairs_for_train_and_val(battery, pairs, split)
        train_hf, val_hf, scaler = scale_train_val(features, split)
        train_capacity, val_capacity = capacities[: split.train_end], capacities[split.train_end : split.val_end]
        stl = STLDecomposer().decompose(series=train_capacity, period=PERIOD, partition_id="TRAIN", series_id=f"{battery}_TRAIN_CAPACITY")
        stl_audit = stl["structural_audit"]
        if not stl_audit["length_preserved"] or not stl_audit["no_nan_or_inf"] or stl_audit["max_abs_reconstruction_error"] > 1e-10:
            raise ProtocolViolation(f"STL reconstruction contract failed for {battery}")
        components = {"trend": stl["trend"], "seasonal": stl["seasonal"], "residual": stl["residual"]}
        train_x, targets = build_train_samples(train_hf, components)
        val_x, val_y = build_val_samples(train_hf, val_hf, val_capacity)
        branches, loss_traces, branch_audit, val_parts = {}, {}, {}, []
        for kind in ("trend", "seasonal", "residual"):
            branch, losses, audit_info = train_branch(kind, train_x, targets[kind], val_x)
            branches[kind] = branch
            loss_traces[kind] = losses
            branch_audit[kind] = audit_info
            with torch.no_grad():
                val_parts.append(branch(torch.from_numpy(val_x), partition="VALIDATION").numpy())
        fused = val_parts[0] + val_parts[1] + val_parts[2]
        if tuple(fused.shape) != tuple(val_y.shape) or not np.isfinite(fused).all():
            raise ProtocolViolation(f"component-sum fusion contract failed for {battery}")
        results[battery] = {
            "training_unit": "PER_TARGET_BATTERY__PHASE1_E1_ROLE_MATRIX",
            "split": {"n_total": split.total, "n_train": split.counts[0], "n_val": split.counts[1], "n_test_membership_only": split.counts[2], "train_range": [0, split.train_end], "val_range": [split.train_end, split.val_end], "test_membership_range": [split.val_end, split.total]},
            "hf_extraction": hf_audit,
            "normalization": scaler,
            "stl": {"period": PERIOD, "mode": "OFFLINE_TRAIN_ONLY", "temporal_leakage_audit": "PASS__OFFLINE_WITHIN_TRAIN_ONLY", **stl_audit},
            "samples": {"train": int(train_x.shape[0]), "val": int(val_x.shape[0]), "target_alignment": "HF[t-9:t]__TO__COMPONENT_OR_CAPACITY[t+1]", "val_history_policy": "PAST_HF_AVAILABLE_AT_PREDICTION_TIME"},
            "branch_audit": branch_audit,
            "train_loss_traces": loss_traces,
            "fusion": {"mode": "ADVISOR_AUTHORIZED_COMPONENT_SUM", "output_shape": list(fused.shape), "status": "PASS"},
            "prediction_domain": "CAPACITY",
            "capacity_to_soh_bridge_status": "PA001_ACTIVE__HUMAN_AUTHORIZED_FALLBACK",
            "soh_formula": SOH_FORMULA,
            "soh_denominator_ah": NASA_RATED_CAPACITY_AH,
            "validation_metrics_capacity": metrics(fused.reshape(-1), val_y.reshape(-1)),
            "validation_metrics_soh": metrics(
                capacity_to_soh(fused.reshape(-1)), capacity_to_soh(val_y.reshape(-1))
            ),
        }
    return results


def comparable(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def main() -> None:
    audits, splits = validate_pairing_and_splits()
    first = one_run(audits, splits)
    second = one_run(audits, splits)
    deterministic = comparable(first, second)
    if not deterministic:
        raise ProtocolViolation("fixed-seed deterministic rerun mismatch")
    output = {
        "phase": "PHASE9A_REAL_TRAIN_VAL_DRY_RUN_AFTER_ALL_INTERFACE_FREEZES",
        "protocol": {"pairing_mode": "DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE", "split": "CHRONOLOGICAL_70_20_10", "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "stl_period": PERIOD, "stl_mode": "OFFLINE_TRAIN_ONLY", "normalization": "MINMAX_0_1__TRAIN_ONLY", "loss": "MSE", "fusion": "TREND_PLUS_SEASONAL_PLUS_RESIDUAL", "epochs": RUN_CONFIG.dry_run_epochs, "seed": RUN_CONFIG.seed, "batch_size": RUN_CONFIG.batch_size, "learning_rate": RUN_CONFIG.learning_rate, "optimizer": RUN_CONFIG.optimizer, "device": RUN_CONFIG.device, "train_batch_shuffle": RUN_CONFIG.train_batch_shuffle, "validation_batch_shuffle": RUN_CONFIG.validation_batch_shuffle, "drop_last": RUN_CONFIG.drop_last},
        "test_lock": {"raw_mat_full_parse_allowed": True, "test_loaded": False, "test_numerical_value_access_count": 0, "test_transform_applied": False, "test_predictions_saved": False},
        "pairing_metadata": {battery: {key: value for key, value in audits[battery].items() if key != "pair_identities"} for battery in BATTERIES},
        "target_execution_roster": list(TARGET_BATTERIES),
        "results_by_battery": first,
        "determinism": {"same_seed_reruns": 2, "status": "PASS" if deterministic else "FAIL", "comparison": "complete_result_payload_exact"},
        "validation_used_for_model_selection": False,
        "scfgan_trained": False,
        "synthetic_hf_used": False,
        "grid_search_run": False,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "target_batteries": list(TARGET_BATTERIES)}, indent=2))


if __name__ == "__main__":
    main()
