"""Tier 2 Phase 9C source-only SCF-GAN real-data smoke execution.

PLAN-CHANGE-0018 freezes every execution choice in this runner.  It permits
real HF extraction only from B0005/B0006.  B0007/B0018 are accessed only by
the metadata-only pairing audit to obtain first-20%-life condition lengths.
No capacity, SOH, validation, or TEST numeric field is dereferenced here.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import _cycles, extract_frozen_hf
from scfgan_structural import ParameterizedSCFGANStructural, SCFGANStructuralConfig


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "PHASE9C_SCFGAN_REAL_DATA_SMOKE_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_SCFGAN_REAL_DATA_SMOKE_REPORT.md"
OUTPUT_ROOT = ROOT / "output" / "phase9c_scfgan_real_data_smoke"

SOURCE_BATTERIES = ("B0005", "B0006")
TARGET_BATTERIES = ("B0007", "B0018")
NOISE_DIM, CONDITION_DIM, FEATURE_DIM = 16, 1, 3
BATCH_SIZE, SEED, SMOKE_EPOCHS = 32, 42, 20
LEARNING_RATE, BETAS = 0.001, (0.9, 0.999)
FIRST_LIFE_FRACTION = 0.20
EPS = 1e-12


class ProtocolViolation(RuntimeError):
    """Raised before a forbidden provenance path can complete."""


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def normalized_cycle_position(index: int, count: int) -> float:
    if count < 1:
        raise ProtocolViolation("condition lifecycle is empty")
    return 0.0 if count == 1 else float(index / (count - 1))


def source_hf_and_condition(battery: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract real source-HF values only; never dereference discharge data."""
    result = audit(battery)  # metadata-only audit; capacity values are not read.
    pairs = result["pair_identities"]
    if battery not in SOURCE_BATTERIES or not result["one_to_one"]:
        raise ProtocolViolation(f"invalid source battery or pairing for {battery}")
    cycles = _cycles(battery)
    features: list[np.ndarray] = []
    conditions: list[list[float]] = []
    t_imin_fallback_count = 0
    for pair_index, pair in enumerate(pairs):
        # This intentionally uses only the paired *charge* operation.
        charge = cycles[int(pair["charge_record_index"])]
        hf, endpoint_provenance = extract_frozen_hf(charge)
        features.append(hf)
        conditions.append([normalized_cycle_position(pair_index, len(pairs))])
        t_imin_fallback_count += int(endpoint_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
    array = np.asarray(features, dtype=np.float64)
    condition = np.asarray(conditions, dtype=np.float64)
    if array.shape != (len(pairs), FEATURE_DIM) or condition.shape != (len(pairs), CONDITION_DIM):
        raise ProtocolViolation(f"source shape contract failed for {battery}")
    if not np.isfinite(array).all() or not np.isfinite(condition).all() or (condition < 0.0).any() or (condition > 1.0).any():
        raise ProtocolViolation(f"source finite/range contract failed for {battery}")
    return array, condition, {
        "valid_source_hf_count": int(array.shape[0]),
        "condition_min": float(condition.min()),
        "condition_max": float(condition.max()),
        "hf2_final_timestamp_fallback_count": t_imin_fallback_count,
        "capacity_or_soh_accessed": False,
    }


def target_condition_only(battery: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Use metadata-only target pair identity count; never parse target cycles."""
    if battery not in TARGET_BATTERIES:
        raise ProtocolViolation(f"unapproved target battery {battery}")
    result = audit(battery)
    total = int(result["n_valid_pairs"])
    count = math.floor(total * FIRST_LIFE_FRACTION)
    if count < 1:
        raise ProtocolViolation(f"target first-20% condition sequence is empty for {battery}")
    condition = np.asarray([[normalized_cycle_position(index, total)] for index in range(count)], dtype=np.float32)
    if condition.shape != (count, CONDITION_DIM) or not np.isfinite(condition).all():
        raise ProtocolViolation(f"target metadata condition shape/finite contract failed for {battery}")
    return condition, {
        "target_valid_pair_count_metadata_only": total,
        "first_20_percent_condition_count": count,
        "condition_min": float(condition.min()),
        "condition_max": float(condition.max()),
        "target_hf_values_materialized": False,
        "target_capacity_or_soh_materialized": False,
    }


def fit_source_tanh_scaler(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    low = features.min(axis=0)
    high = features.max(axis=0)
    if np.any(~np.isfinite(low)) or np.any(high - low <= EPS):
        raise ProtocolViolation("source-only GAN scaler has invalid feature range")
    normalized = 2.0 * (features - low) / (high - low) - 1.0
    if not np.isfinite(normalized).all() or normalized.min() < -1.0 - 1e-7 or normalized.max() > 1.0 + 1e-7:
        raise ProtocolViolation("source-only GAN scaler output violates [-1,1]")
    return normalized.astype(np.float32), {
        "normalization": "MINMAX_MINUS1_TO_PLUS1",
        "fit_scope": "B0005_B0006_SOURCE_HF_ONLY",
        "feature_min": [float(value) for value in low],
        "feature_max": [float(value) for value in high],
    }


def inverse_source_tanh_scaler(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    low = np.asarray(scaler["feature_min"], dtype=np.float64)
    high = np.asarray(scaler["feature_max"], dtype=np.float64)
    physical = 0.5 * (values + 1.0) * (high - low) + low
    if physical.ndim != 2 or physical.shape[1] != FEATURE_DIM or not np.isfinite(physical).all():
        raise ProtocolViolation("generated-HF inverse transform failed")
    return physical


def stat_summary(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.isfinite(values).all():
        raise ProtocolViolation("diagnostic tensor is not finite [N,3]")
    return {
        "mean": [float(value) for value in values.mean(axis=0)],
        "std": [float(value) for value in values.std(axis=0)],
        "min": [float(value) for value in values.min(axis=0)],
        "max": [float(value) for value in values.max(axis=0)],
    }


def model_config() -> SCFGANStructuralConfig:
    return SCFGANStructuralConfig(
        noise_dim=NOISE_DIM,
        condition_dim=CONDITION_DIM,
        feature_dim=FEATURE_DIM,
        generator_hidden_dims=(64, 128, 64),
        generator_activation="relu",
        generator_norm_mode="conditional_affine_bn_structural",
        generator_use_sn=False,
        discriminator_hidden_dims=(64, 128, 64, 32),
        discriminator_use_sn=True,
        feature_match_layer_index=2,
        leaky_relu_slope=0.2,
        synthetic_only_mode=False,
    )


def _finite_nonempty_gradients(module: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(value).all() for value in gradients) and any(torch.count_nonzero(value).item() for value in gradients)


def train_source_only(model: ParameterizedSCFGANStructural, features: np.ndarray, conditions: np.ndarray) -> tuple[list[dict[str, float]], dict[str, bool]]:
    real = torch.tensor(features, dtype=torch.float32)
    cond = torch.tensor(conditions, dtype=torch.float32)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    trace: list[dict[str, float]] = []
    gradient_status = {"generator": False, "discriminator": False, "feature_match": False}
    model.train()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        sums = {"discriminator_loss": 0.0, "generator_loss": 0.0, "feature_match_loss": 0.0}
        batches = 0
        for start in range(0, real.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, real.shape[0])
            real_batch, condition_batch = real[start:stop], cond[start:stop]
            noise = torch.randn(real_batch.shape[0], NOISE_DIM)

            discriminator_optimizer.zero_grad(set_to_none=True)
            fake_batch = model.generate(noise, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            real_score, _, _ = model.discriminate(real_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            fake_score, _, _ = model.discriminate(fake_batch.detach(), condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            discriminator_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
            if not torch.isfinite(discriminator_loss):
                raise ProtocolViolation("non-finite discriminator loss")
            discriminator_loss.backward()
            gradient_status["discriminator"] = gradient_status["discriminator"] or _finite_nonempty_gradients(model.discriminator)
            discriminator_optimizer.step()

            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(real_batch.shape[0], NOISE_DIM)
            fake_batch = model.generate(noise, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            with torch.no_grad():
                _, real_features, _ = model.discriminate(real_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            fake_score, fake_features, _ = model.discriminate(fake_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            adversarial_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
            feature_match_loss = F.mse_loss(fake_features, real_features)
            generator_loss = adversarial_loss + model.feature_match_lambda * feature_match_loss
            if not torch.isfinite(generator_loss) or not torch.isfinite(feature_match_loss):
                raise ProtocolViolation("non-finite generator or feature-match loss")
            generator_loss.backward()
            gradient_status["generator"] = gradient_status["generator"] or _finite_nonempty_gradients(model.generator)
            gradient_status["feature_match"] = gradient_status["feature_match"] or any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad).item()
                for parameter in model.generator.parameters()
            )
            generator_optimizer.step()
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(True)

            sums["discriminator_loss"] += float(discriminator_loss.detach())
            sums["generator_loss"] += float(generator_loss.detach())
            sums["feature_match_loss"] += float(feature_match_loss.detach())
            batches += 1
        if batches == 0:
            raise ProtocolViolation("source-only GAN has no batches")
        trace.append({"epoch": epoch, **{key: value / batches for key, value in sums.items()}})
    return trace, gradient_status


def generate_target_sets(model: ParameterizedSCFGANStructural, target: str, condition: np.ndarray, scaler: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, list[float]]]]:
    model.eval()
    condition_tensor = torch.tensor(condition, dtype=torch.float32)
    records: list[dict[str, Any]] = []
    statistics: list[dict[str, list[float]]] = []
    for set_index in range(1, 6):
        with torch.no_grad():
            noise = torch.randn(condition_tensor.shape[0], NOISE_DIM)
            normalized = model.generate(noise, condition_tensor, provenance="TARGET_CYCLE_IDENTITY_METADATA_ONLY", request_kind="TARGET_CONDITION_GENERATION")
        normalized_array = normalized.detach().cpu().numpy().astype(np.float64)
        if normalized_array.shape != (condition.shape[0], FEATURE_DIM) or not np.isfinite(normalized_array).all():
            raise ProtocolViolation(f"generated normalized shape/finite contract failed for {target} set {set_index}")
        physical = inverse_source_tanh_scaler(normalized_array, scaler)
        path = OUTPUT_ROOT / f"{target}_synthetic_hf_set_{set_index:02d}.npy"
        np.save(path, physical.astype(np.float64), allow_pickle=False)
        records.append({"set_index": set_index, "path": str(path.relative_to(ROOT)), "shape": list(physical.shape), "finite": True, "inverse_transform": "PASS"})
        statistics.append(stat_summary(physical))
    return records, statistics


def write_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 9C SCF-GAN real-data smoke report",
        "",
        "## Result",
        "",
        f"`PHASE9C_SCFGAN_SMOKE_GATE={metrics['phase9c_scfgan_smoke_gate']}`  ",
        f"`SCFGAN_SOURCE_ONLY_GATE={metrics['scfgan_source_only_gate']}`  ",
        f"`TEST_ACCESS_COUNT={metrics['test_access_count']}`",
        "",
        "## Frozen Tier 2 status",
        "",
        "The real execution configuration is a human-authorized, result-blind Tier 2 fallback. It is not paper exact. Generator activation, generator spectral-normalization scope, discriminator depth, and condition semantics remain recorded paper contradictions.",
        "",
        "## Data boundary",
        "",
        "Training materialized HF only from B0005/B0006 paired charge operations. B0007/B0018 were read only through structural pair metadata to form first-20%-life cycle-position conditions. No target HF, capacity, SOH, validation, or TEST value entered GAN training or generation.",
        "",
        "## Runtime",
        "",
        f"`seed={SEED}`, `epochs={SMOKE_EPOCHS}`, `batch_size={BATCH_SIZE}`, `Adam lr={LEARNING_RATE}`, `betas={BETAS}`. No early stopping, grid search, or quality-driven selection was performed.",
        "",
        "## Smoke diagnostics",
        "",
        "The JSON companion contains source-real and generated physical-HF mean, standard deviation, minimum, and maximum. These are structural smoke diagnostics only and were not used to modify the frozen configuration.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise ProtocolViolation(f"refusing to overwrite existing local smoke output: {OUTPUT_ROOT}")
    set_seed()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    source_hf, source_condition, source_detail = [], [], {}
    for battery in SOURCE_BATTERIES:
        hf, condition, detail = source_hf_and_condition(battery)
        source_hf.append(hf)
        source_condition.append(condition)
        source_detail[battery] = detail
    source_hf_array = np.concatenate(source_hf, axis=0)
    source_condition_array = np.concatenate(source_condition, axis=0)
    source_normalized, scaler = fit_source_tanh_scaler(source_hf_array)

    config = model_config()
    model = ParameterizedSCFGANStructural(config)
    sn_layers = model.spectral_norm_layer_names()
    if sn_layers["generator"] or len(sn_layers["discriminator"]) != 4:
        raise ProtocolViolation("spectral-normalization configuration does not match PLAN-CHANGE-0018")
    trace, gradients = train_source_only(model, source_normalized, source_condition_array.astype(np.float32))
    if not all(gradients.values()):
        raise ProtocolViolation("generator/discriminator/feature-match gradient gate failed")

    checkpoint_path = OUTPUT_ROOT / "SCFGAN_SOURCE_ONLY_SMOKE_20_EPOCHS.pt"
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "scaler": scaler, "epoch_count": SMOKE_EPOCHS}, checkpoint_path)

    target_generation: dict[str, Any] = {}
    for target in TARGET_BATTERIES:
        condition, target_detail = target_condition_only(target)
        sets, set_statistics = generate_target_sets(model, target, condition, scaler)
        target_generation[target] = {"condition_metadata": target_detail, "sets": sets, "physical_hf_statistics_by_set": set_statistics}

    metrics: dict[str, Any] = {
        "task": "PHASE_9C_SCFGAN_REAL_DATA_SMOKE_TEST",
        "plan_change": "PLAN-CHANGE-0018",
        "paper_generator_activation_contradiction": True,
        "paper_generator_sn_scope_contradiction": True,
        "paper_discriminator_depth_contradiction": True,
        "paper_scfgan_condition_contradiction": True,
        "paper_exact_scfgan_execution_status": "UNRESOLVED",
        "tier2_scfgan_config_status": "HUMAN_AUTHORIZED_RESULT_BLIND_FALLBACK",
        "config": asdict(config),
        "condition_definition": "PER_BATTERY_NORMALIZED_PAIRED_CYCLE_POSITION_0_TO_1",
        "noise_distribution": "STANDARD_NORMAL",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(BETAS), "batch_size": BATCH_SIZE, "seed": SEED, "device": "CPU", "epochs": SMOKE_EPOCHS},
        "scfgan_source_only_gate": "PASS",
        "scfgan_train_source_only": True,
        "target_hf_used_for_gan_training": False,
        "target_capacity_or_soh_used_for_gan_training": False,
        "validation_access_count": 0,
        "test_access_count": 0,
        "test_loaded": False,
        "scaler": scaler,
        "downstream_train_only_0_1_scaler_reused_or_fitted": False,
        "source_hf_counts_and_condition_details": source_detail,
        "source_physical_hf_statistics": stat_summary(source_hf_array),
        "spectral_norm_layers": sn_layers,
        "loss_trace": trace,
        "finite_loss": True,
        "gradient_gate": gradients,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "target_generation": target_generation,
        "generated_hf_contains_capacity_or_soh": False,
        "grid_search_run": False,
        "early_stopping_used": False,
        "generation_quality_used_for_selection": False,
        "phase9c_scfgan_smoke_gate": "PASS",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({key: metrics[key] for key in ("phase9c_scfgan_smoke_gate", "scfgan_source_only_gate", "test_access_count", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
