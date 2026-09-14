"""Fixed-config, source-only SCF-GAN convergence diagnostic for Phase 9C.

This runner is deliberately isolated from target data. It trains only on the
already authorized B0005/B0006 HF values, evaluates fixed noise against that
same source reference, and writes local model snapshots every 20 epochs. It
does not load target, validation, or TEST records and does not select a model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_phase9c_scfgan_generation_quality_diagnostic import dataset_metrics
from run_phase9c_scfgan_real_data_smoke import (
    BATCH_SIZE,
    BETAS,
    FEATURE_DIM,
    LEARNING_RATE,
    NOISE_DIM,
    ProtocolViolation,
    fit_source_tanh_scaler,
    model_config,
    source_hf_and_condition,
)
from scfgan_structural import ParameterizedSCFGANStructural


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "phase9c_scfgan_source_only_convergence"
METRICS_PATH = ROOT / "PHASE9C_SCFGAN_SOURCE_ONLY_CONVERGENCE_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_SCFGAN_SOURCE_ONLY_CONVERGENCE_REPORT.md"
SOURCE_BATTERIES = ("B0005", "B0006")
SEED = 42
EVALUATION_NOISE_SEED = 424242
MAX_EPOCHS = 300
SNAPSHOT_INTERVAL = 20
EPSILON = 1e-12
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def set_training_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def source_data() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features, conditions, details = [], [], {}
    for battery in SOURCE_BATTERIES:
        hf, condition, detail = source_hf_and_condition(battery)
        features.append(hf)
        conditions.append(condition)
        details[battery] = detail
    hf_array = np.concatenate(features, axis=0)
    condition_array = np.concatenate(conditions, axis=0).astype(np.float32)
    if hf_array.shape != (336, FEATURE_DIM) or condition_array.shape != (336, 1):
        raise ProtocolViolation("source-only source-HF shape contract failed")
    if not np.isfinite(hf_array).all() or not np.isfinite(condition_array).all():
        raise ProtocolViolation("source-only source-HF finite contract failed")
    return hf_array, condition_array, details


def static_evaluation_noise(rows: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(EVALUATION_NOISE_SEED)
    return torch.randn(rows, NOISE_DIM, generator=generator)


def bin_mask(condition: np.ndarray, index: int) -> np.ndarray:
    low, high = BIN_EDGES[index], BIN_EDGES[index + 1]
    return (condition >= low) & ((condition < high) if index < 4 else (condition <= high))


def condition_bin_statistics(source_hf: np.ndarray, generated_hf: np.ndarray, condition: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    flat_condition = condition.reshape(-1)
    for index in range(5):
        mask = bin_mask(flat_condition, index)
        source, generated = source_hf[mask], generated_hf[mask]
        if source.shape[0] == 0 or generated.shape[0] == 0:
            raise ProtocolViolation("source condition bin unexpectedly has no samples")
        output[f"bin_{index + 1}"] = {
            "interval": "[0.8,1.0]" if index == 4 else f"[{BIN_EDGES[index]:.1f},{BIN_EDGES[index + 1]:.1f})",
            "source": {
                "count": int(source.shape[0]),
                "mean": {f"HF{feature + 1}": float(source[:, feature].mean()) for feature in range(FEATURE_DIM)},
                "std": {f"HF{feature + 1}": float(source[:, feature].std()) for feature in range(FEATURE_DIM)},
            },
            "generated": {
                "count": int(generated.shape[0]),
                "mean": {f"HF{feature + 1}": float(generated[:, feature].mean()) for feature in range(FEATURE_DIM)},
                "std": {f"HF{feature + 1}": float(generated[:, feature].std()) for feature in range(FEATURE_DIM)},
            },
        }
    return output


def summary_values(metric: dict[str, Any]) -> dict[str, float]:
    values = list(metric["feature_metrics"].values())
    return {
        "mean_diff": float(np.mean([item["mean_percentage_difference"] for item in values])),
        "std_diff": float(np.mean([item["std_percentage_difference"] for item in values])),
        "kl": float(np.mean([item["kl_divergence_real_to_synthetic"] for item in values])),
        "wasserstein": float(np.mean([item["wasserstein_distance_physical_domain"] for item in values])),
        "outlier_ratio": float(np.mean([item["synthetic_outlier_ratio"] for item in values])),
    }


def snapshot_diagnostic(model: ParameterizedSCFGANStructural, source_physical: np.ndarray, source_condition: np.ndarray, scaler: dict[str, Any], noise: torch.Tensor, epoch: int) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        generated_normalized = model.generate(
            noise,
            torch.tensor(source_condition, dtype=torch.float32),
            provenance="NASA_SOURCE_FIXED_EVALUATION_HF",
            request_kind="REAL_TRAINING",
        ).detach().cpu().numpy().astype(np.float64)
    low = np.asarray(scaler["feature_min"], dtype=np.float64)
    high = np.asarray(scaler["feature_max"], dtype=np.float64)
    generated_physical = 0.5 * (generated_normalized + 1.0) * (high - low) + low
    if generated_physical.shape != source_physical.shape or not np.isfinite(generated_physical).all():
        raise ProtocolViolation(f"non-finite or malformed source evaluation at epoch {epoch}")
    metric = dataset_metrics(source_physical, generated_physical)
    path = OUTPUT_ROOT / f"SCFGAN_SOURCE_ONLY_EPOCH_{epoch:03d}.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "evaluation_noise_seed": EVALUATION_NOISE_SEED}, path)
    return {
        "epoch": epoch,
        "snapshot": str(path.relative_to(ROOT)),
        "source_vs_generated": metric,
        "trajectory_summary_mean_over_hf": summary_values(metric),
        "generated_variance_by_feature": metric["synthetic_variance_by_feature"],
        "generated_nan_inf_count": metric["synthetic_nan_inf_count"],
        "condition_bin_mean_std": condition_bin_statistics(source_physical, generated_physical, source_condition),
    }


def train(model: ParameterizedSCFGANStructural, normalized_hf: np.ndarray, condition: np.ndarray, source_physical: np.ndarray, scaler: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, Any]], dict[str, bool]]:
    real = torch.tensor(normalized_hf, dtype=torch.float32)
    cond = torch.tensor(condition, dtype=torch.float32)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    evaluation_noise = static_evaluation_noise(real.shape[0])
    losses: list[dict[str, float]] = []
    snapshots: list[dict[str, Any]] = []
    gradient_gate = {"generator": False, "discriminator": False, "feature_matching": False}
    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        totals = {"generator_adversarial_loss": 0.0, "feature_matching_loss": 0.0, "generator_total_loss": 0.0, "discriminator_loss": 0.0}
        batches = 0
        for start in range(0, real.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, real.shape[0])
            real_batch, condition_batch = real[start:stop], cond[start:stop]
            discriminator_optimizer.zero_grad(set_to_none=True)
            fake_batch = model.generate(torch.randn(real_batch.shape[0], NOISE_DIM), condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            real_score, _, _ = model.discriminate(real_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            fake_score, _, _ = model.discriminate(fake_batch.detach(), condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            discriminator_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
            if not torch.isfinite(discriminator_loss):
                raise ProtocolViolation(f"non-finite discriminator loss at epoch {epoch}")
            discriminator_loss.backward()
            discriminator_grads = [value.grad for value in model.discriminator.parameters() if value.grad is not None]
            gradient_gate["discriminator"] = gradient_gate["discriminator"] or bool(discriminator_grads) and all(torch.isfinite(value).all() for value in discriminator_grads)
            discriminator_optimizer.step()

            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            fake_batch = model.generate(torch.randn(real_batch.shape[0], NOISE_DIM), condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            with torch.no_grad():
                _, real_hidden, _ = model.discriminate(real_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            fake_score, fake_hidden, _ = model.discriminate(fake_batch, condition_batch, provenance="NASA_SOURCE_TRAIN_HF", request_kind="REAL_TRAINING")
            generator_adversarial_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
            feature_matching_loss = F.mse_loss(fake_hidden, real_hidden)
            generator_total_loss = generator_adversarial_loss + model.feature_match_lambda * feature_matching_loss
            if not torch.isfinite(generator_total_loss) or not torch.isfinite(feature_matching_loss):
                raise ProtocolViolation(f"non-finite generator loss at epoch {epoch}")
            generator_total_loss.backward()
            generator_grads = [value.grad for value in model.generator.parameters() if value.grad is not None]
            gradient_gate["generator"] = gradient_gate["generator"] or bool(generator_grads) and all(torch.isfinite(value).all() for value in generator_grads)
            gradient_gate["feature_matching"] = gradient_gate["feature_matching"] or bool(generator_grads) and any(torch.count_nonzero(value).item() for value in generator_grads)
            generator_optimizer.step()
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(True)

            totals["generator_adversarial_loss"] += float(generator_adversarial_loss.detach())
            totals["feature_matching_loss"] += float(feature_matching_loss.detach())
            totals["generator_total_loss"] += float(generator_total_loss.detach())
            totals["discriminator_loss"] += float(discriminator_loss.detach())
            batches += 1
        losses.append({"epoch": epoch, **{name: value / batches for name, value in totals.items()}})
        if epoch % SNAPSHOT_INTERVAL == 0:
            snapshot = snapshot_diagnostic(model, source_physical, condition, scaler, evaluation_noise, epoch)
            snapshots.append(snapshot)
            summary = snapshot["trajectory_summary_mean_over_hf"]
            print(f"snapshot epoch={epoch} mean_diff={summary['mean_diff']:.6f} std_diff={summary['std_diff']:.6f} kl={summary['kl']:.6f} wasserstein={summary['wasserstein']:.6f} outlier={summary['outlier_ratio']:.6f}", flush=True)
            model.train()
    return losses, snapshots, gradient_gate


def late_analysis(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    summary = [item["trajectory_summary_mean_over_hf"] for item in snapshots]
    first, last = summary[0], summary[-1]
    fields = ("mean_diff", "std_diff", "kl", "wasserstein", "outlier_ratio")
    improvement_by_metric = {field: bool(last[field] < first[field]) for field in fields}
    source_alignment_improved = bool(
        all(last[field] <= first[field] for field in fields)
        and any(improvement_by_metric.values())
    )
    late = [(snapshot["epoch"], snapshot["trajectory_summary_mean_over_hf"]) for snapshot in snapshots if snapshot["epoch"] >= 200]
    late_ranges = {field: {"min": float(min(item[field] for _, item in late)), "max": float(max(item[field] for _, item in late))} for field in fields}
    reversal_counts: dict[str, int] = {}
    for field in fields:
        changes = np.sign(np.diff(np.asarray([item[field] for _, item in late], dtype=np.float64)))
        nonzero = changes[changes != 0]
        reversal_counts[field] = int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0
    any_reversal = any(value > 0 for value in reversal_counts.values())
    all_nonzero_variance = all(
        all(value > 0.0 for value in snapshot["generated_variance_by_feature"].values())
        for snapshot in snapshots
    )
    return {
        "improvement_definition": "EPOCH_300_NO_HIGHER_THAN_EPOCH_20_FOR_ALL_FIVE_UNWEIGHTED_MEAN_OVER_HF_METRICS__AND_STRICTLY_LOWER_FOR_AT_LEAST_ONE",
        "improvement_by_metric": improvement_by_metric,
        "source_alignment_improved": source_alignment_improved,
        "epoch_20_summary": first,
        "epoch_300_summary": last,
        "late_epoch_range": "200_TO_300",
        "late_metric_ranges": late_ranges,
        "late_direction_reversal_counts": reversal_counts,
        "late_training_stability_status": "OSCILLATORY__DIRECTION_REVERSALS_OBSERVED" if any_reversal else "MONOTONIC__NO_DIRECTION_REVERSALS_OBSERVED",
        "epoch_200_to_300_stable_platform": not any_reversal,
        "mode_collapse_status": "NO_ZERO_VARIANCE_FEATURE_COLLAPSE_OBSERVED_AT_SNAPSHOTS" if all_nonzero_variance else "ZERO_VARIANCE_FEATURE_OBSERVED",
    }


def write_report(metrics: dict[str, Any]) -> None:
    analysis = metrics["convergence_analysis"]
    lines = [
        "# Phase 9C SCF-GAN source-only convergence diagnostic",
        "",
        "## Scope controls",
        "",
        "```text",
        "TARGET_INFORMATION_USED_FOR_TRAINING=False",
        "TARGET_INFORMATION_USED_FOR_MODEL_SELECTION=False",
        "TEST_ACCESS_COUNT=0",
        "SCFGAN_RETRAINED=True",
        "DOWNSTREAM_MODEL_TRAINED=False",
        "EARLY_STOPPING=False",
        "GRID_SEARCH=False",
        "```",
        "",
        "The model was initialized anew with seed 42 and trained for exactly 300 epochs using B0005/B0006 source HF only. Snapshot comparison uses a single fixed evaluation noise draw and the source cycle-position conditions. No snapshot was selected as best.",
        "",
        "## Source-alignment trajectory",
        "",
        "| epoch | mean diff | std diff | KL | Wasserstein | outlier ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for snapshot in metrics["snapshots"]:
        values = snapshot["trajectory_summary_mean_over_hf"]
        lines.append(f"| {snapshot['epoch']} | {values['mean_diff']:.6f} | {values['std_diff']:.6f} | {values['kl']:.6f} | {values['wasserstein']:.6f} | {values['outlier_ratio']:.6f} |")
    lines.extend([
        "",
        "## Descriptive convergence facts",
        "",
        f"`SOURCE_ALIGNMENT_IMPROVED={analysis['source_alignment_improved']}` using the registered all-five-metric non-worsening plus at-least-one-improvement Epoch-20-to-Epoch-300 definition.",
        f"`LATE_TRAINING_STABILITY_STATUS={analysis['late_training_stability_status']}`  ",
        f"`EPOCH_200_TO_300_STABLE_PLATFORM={analysis['epoch_200_to_300_stable_platform']}`  ",
        f"`MODE_COLLAPSE_STATUS={analysis['mode_collapse_status']}`",
        "",
        "Per-HF snapshot metrics, loss traces, source/generated condition-bin means and standard deviations, NaN/Inf counts, generated variance, late ranges, and direction-reversal counts are contained in the JSON companion. These facts were not used to alter any frozen parameter.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise ProtocolViolation(f"refusing to overwrite local convergence snapshots: {OUTPUT_ROOT}")
    set_training_seed()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    source_physical, source_condition, source_details = source_data()
    normalized_hf, scaler = fit_source_tanh_scaler(source_physical)
    model = ParameterizedSCFGANStructural(model_config())
    losses, snapshots, gradient_gate = train(model, normalized_hf, source_condition, source_physical, scaler)
    if len(snapshots) != MAX_EPOCHS // SNAPSHOT_INTERVAL or not all(gradient_gate.values()):
        raise ProtocolViolation("snapshot or gradient contract failed")
    analysis = late_analysis(snapshots)
    metrics = {
        "task": "PHASE9C_SCFGAN_SOURCE_ONLY_CONVERGENCE_DIAGNOSTIC",
        "plan_change": "PLAN-CHANGE-0018__UNCHANGED",
        "source_batteries": list(SOURCE_BATTERIES),
        "target_information_used_for_training": False,
        "target_information_used_for_model_selection": False,
        "validation_information_used": False,
        "test_loaded": False,
        "test_access_count": 0,
        "scfgan_retrained": True,
        "downstream_model_trained": False,
        "synthetic_regenerated_for_target": False,
        "seed": SEED,
        "evaluation_noise_seed": EVALUATION_NOISE_SEED,
        "max_epochs": MAX_EPOCHS,
        "snapshot_interval_epochs": SNAPSHOT_INTERVAL,
        "early_stopping": False,
        "grid_search": False,
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(BETAS), "batch_size": BATCH_SIZE},
        "source_hf_scaler": scaler,
        "source_details": source_details,
        "loss_trace": losses,
        "snapshots": snapshots,
        "gradient_gate": gradient_gate,
        "convergence_analysis": analysis,
        "test_lock_gate": "PASS",
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({"source_alignment_at_epoch_20": analysis["epoch_20_summary"], "source_alignment_at_epoch_300": analysis["epoch_300_summary"], "source_alignment_improved": analysis["source_alignment_improved"], "late_training_stability_status": analysis["late_training_stability_status"], "mode_collapse_status": analysis["mode_collapse_status"], "test_lock_gate": metrics["test_lock_gate"]}, indent=2))


if __name__ == "__main__":
    main()
