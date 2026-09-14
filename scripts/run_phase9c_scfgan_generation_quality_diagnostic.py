"""Post-hoc Tier 2 quality diagnostic for already persisted Phase 9C HF sets.

This runner never loads a GAN checkpoint, trains a model, or accesses TEST.
It extracts only each target battery's first-20%-life paired-charge HF as a
reference and compares it with the already persisted HF-only synthetic sets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import _cycles, extract_frozen_hf, split_for


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "phase9c_scfgan_real_data_smoke"
METRICS_PATH = ROOT / "PHASE9C_SCFGAN_GENERATION_QUALITY_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_SCFGAN_GENERATION_QUALITY_REPORT.md"
TARGETS = ("B0007", "B0018")
FEATURE_NAMES = ("HF1", "HF2", "HF3")
FIRST_LIFE_FRACTION = 0.20
HISTOGRAM_BINS = 20
EPSILON = 1e-8


class ProtocolViolation(RuntimeError):
    """Raised before a forbidden TEST or incomplete artifact path can continue."""


def first_20_reference_hf(battery: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize permitted target TRAIN-reference charge HF only.

    Pair membership is established before charge-field dereference. The
    reference range is first 20% of valid paired cycles and is asserted to be
    contained fully in the frozen 70% chronological TRAIN partition.
    """
    pairing = audit(battery)  # metadata only: no capacity values are read.
    pairs = pairing["pair_identities"]
    total = int(pairing["n_valid_pairs"])
    reference_count = math.floor(total * FIRST_LIFE_FRACTION)
    split = split_for(total)
    if not pairing["one_to_one"] or reference_count < 1 or reference_count > split.train_end:
        raise ProtocolViolation(f"first-20% reference is not fully in TRAIN for {battery}")
    cycles = _cycles(battery)  # authorized full parse; post-split charge access is gated below.
    values: list[np.ndarray] = []
    hf2_fallback_count = 0
    for pair_index in range(reference_count):
        if pair_index >= split.train_end:
            raise ProtocolViolation("reference attempt crossed the frozen TRAIN boundary")
        charge = cycles[int(pairs[pair_index]["charge_record_index"])]
        hf, endpoint_provenance = extract_frozen_hf(charge)
        values.append(hf)
        hf2_fallback_count += int(endpoint_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
    reference = np.asarray(values, dtype=np.float64)
    if reference.shape != (reference_count, 3) or not np.isfinite(reference).all():
        raise ProtocolViolation(f"invalid target HF reference tensor for {battery}")
    return reference, {
        "reference_scope": "TARGET_FIRST_20_PERCENT_LIFE__PAIRED_CHARGE_HF__POST_HOC_QUALITY_EVALUATION_ONLY",
        "valid_pair_count_metadata": total,
        "reference_pair_count": reference_count,
        "frozen_train_pair_count": split.train_end,
        "reference_fully_within_train": True,
        "capacity_or_soh_values_accessed": False,
        "hf2_final_timestamp_fallback_count": hf2_fallback_count,
    }


def checked_synthetic_sets(battery: str, expected_rows: int) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for set_index in range(1, 6):
        path = OUTPUT_ROOT / f"{battery}_synthetic_hf_set_{set_index:02d}.npy"
        if not path.is_file():
            raise ProtocolViolation(f"missing persisted synthetic set: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (expected_rows, 3) or not np.isfinite(array).all():
            raise ProtocolViolation(f"persisted synthetic shape/finite contract failed: {path}")
        arrays.append(np.asarray(array, dtype=np.float64))
    return arrays


def kl_real_to_synthetic(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, dict[str, float]]:
    lower = float(min(real.min(), synthetic.min()))
    upper = float(max(real.max(), synthetic.max()))
    if upper <= lower:
        # Deterministic range widening is needed only for a degenerate feature.
        lower, upper = lower - 0.5, upper + 0.5
    real_hist, _ = np.histogram(real, bins=HISTOGRAM_BINS, range=(lower, upper))
    synthetic_hist, _ = np.histogram(synthetic, bins=HISTOGRAM_BINS, range=(lower, upper))
    p = (real_hist.astype(np.float64) + EPSILON)
    q = (synthetic_hist.astype(np.float64) + EPSILON)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q))), {"common_min": lower, "common_max": upper}


def feature_metric(real: np.ndarray, synthetic: np.ndarray) -> dict[str, Any]:
    mean_real, mean_synthetic = float(real.mean()), float(synthetic.mean())
    std_real, std_synthetic = float(real.std()), float(synthetic.std())
    q1, q3 = (float(value) for value in np.percentile(real, (25.0, 75.0)))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kl, histogram_range = kl_real_to_synthetic(real, synthetic)
    outlier_mask = (synthetic < lower) | (synthetic > upper)
    return {
        "mean_real": mean_real,
        "mean_synthetic": mean_synthetic,
        "std_real": std_real,
        "std_synthetic": std_synthetic,
        "mean_percentage_difference": abs(mean_synthetic - mean_real) / max(abs(mean_real), EPSILON) * 100.0,
        "std_percentage_difference": abs(std_synthetic - std_real) / max(std_real, EPSILON) * 100.0,
        "kl_divergence_real_to_synthetic": kl,
        "kl_histogram": {"bins": HISTOGRAM_BINS, "epsilon": EPSILON, **histogram_range},
        "wasserstein_distance_physical_domain": float(wasserstein_distance(real, synthetic)),
        "real_q1": q1,
        "real_q3": q3,
        "real_iqr": iqr,
        "outlier_lower_bound": lower,
        "outlier_upper_bound": upper,
        "synthetic_outlier_ratio": float(outlier_mask.mean()),
        "real_min": float(real.min()),
        "real_max": float(real.max()),
        "synthetic_min": float(synthetic.min()),
        "synthetic_max": float(synthetic.max()),
        "real_nan_inf_count": int((~np.isfinite(real)).sum()),
        "synthetic_nan_inf_count": int((~np.isfinite(synthetic)).sum()),
        "real_variance": float(real.var()),
        "synthetic_variance": float(synthetic.var()),
        "synthetic_to_real_variance_ratio": float(synthetic.var() / max(real.var(), EPSILON)),
    }


def dataset_metrics(real: np.ndarray, synthetic: np.ndarray) -> dict[str, Any]:
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != 3 or synthetic.shape[1] != 3:
        raise ProtocolViolation("quality metric requires [N,3] physical-HF tensors")
    return {
        "rows_real": int(real.shape[0]),
        "rows_synthetic": int(synthetic.shape[0]),
        "feature_metrics": {name: feature_metric(real[:, index], synthetic[:, index]) for index, name in enumerate(FEATURE_NAMES)},
        "synthetic_variance_by_feature": {name: float(synthetic[:, index].var()) for index, name in enumerate(FEATURE_NAMES)},
        "synthetic_min_by_feature": {name: float(synthetic[:, index].min()) for index, name in enumerate(FEATURE_NAMES)},
        "synthetic_max_by_feature": {name: float(synthetic[:, index].max()) for index, name in enumerate(FEATURE_NAMES)},
        "synthetic_nan_inf_count": int((~np.isfinite(synthetic)).sum()),
        "zero_variance_feature_count": int(sum(float(synthetic[:, index].var()) == 0.0 for index in range(3))),
    }


def set_variability(sets: list[np.ndarray]) -> dict[str, Any]:
    means = np.asarray([item.mean(axis=0) for item in sets], dtype=np.float64)
    stds = np.asarray([item.std(axis=0) for item in sets], dtype=np.float64)
    return {
        "set_count": len(sets),
        "set_mean_mean_by_feature": {name: float(means[:, index].mean()) for index, name in enumerate(FEATURE_NAMES)},
        "set_mean_std_by_feature": {name: float(means[:, index].std()) for index, name in enumerate(FEATURE_NAMES)},
        "set_std_mean_by_feature": {name: float(stds[:, index].mean()) for index, name in enumerate(FEATURE_NAMES)},
        "set_std_std_by_feature": {name: float(stds[:, index].std()) for index, name in enumerate(FEATURE_NAMES)},
        "all_sets_have_nonzero_variance": bool(all(np.all(item.var(axis=0) > 0.0) for item in sets)),
    }


def maxima(battery_metrics: dict[str, Any]) -> dict[str, float]:
    values: list[dict[str, Any]] = []
    for battery in battery_metrics.values():
        values.extend(item["feature_metrics"] for item in battery["per_set"].values())
        values.append(battery["pooled"]["feature_metrics"])
    flattened = [feature for dataset in values for feature in dataset.values()]
    return {
        "max_mean_pct_diff": float(max(item["mean_percentage_difference"] for item in flattened)),
        "max_std_pct_diff": float(max(item["std_percentage_difference"] for item in flattened)),
        "max_kl": float(max(item["kl_divergence_real_to_synthetic"] for item in flattened)),
        "max_wasserstein": float(max(item["wasserstein_distance_physical_domain"] for item in flattened)),
        "max_outlier_ratio": float(max(item["synthetic_outlier_ratio"] for item in flattened)),
    }


def write_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 9C SCF-GAN generation-quality diagnostic",
        "",
        "## Scope and controls",
        "",
        "This post-hoc diagnostic compared persisted HF-only synthetic sets with target first-20%-life paired-charge HF references. Target HF was used only for evaluation after GAN training; no GAN or downstream model was trained or changed. TEST numerical access was zero.",
        "",
        "```text",
        "SCFGAN_RETRAINED=False",
        "DOWNSTREAM_MODEL_TRAINED=False",
        "TEST_LOADED=False",
        "TEST_ACCESS_COUNT=0",
        "KL_IMPLEMENTATION=TIER2_DETERMINISTIC_DIAGNOSTIC",
        "OUTLIER_IMPLEMENTATION=TIER2_DETERMINISTIC_DIAGNOSTIC",
        "```",
        "",
        "## Metric definitions",
        "",
        "KL is `D_KL(real || synthetic)` using 20 bins over the real/synthetic common physical-HF range, with epsilon `1e-8` added before normalization. Wasserstein distance is calculated in physical HF units. Outlier ratio uses the real reference `[Q1-1.5IQR, Q3+1.5IQR]` interval. These deterministic Tier 2 diagnostics are not claimed to reproduce an unreported paper implementation.",
        "",
        "## Aggregate maxima",
        "",
    ]
    for name, value in metrics["aggregate_maxima"].items():
        lines.append(f"- `{name}={value}`")
    lines.extend([
        "",
        "## Pooled physical-HF results",
        "",
        "| Battery | HF | mean pct diff | std pct diff | KL | Wasserstein | outlier ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for battery, result in metrics["battery_results"].items():
        for feature, values in result["pooled"]["feature_metrics"].items():
            lines.append(
                f"| {battery} | {feature} | {values['mean_percentage_difference']:.6f} | "
                f"{values['std_percentage_difference']:.6f} | {values['kl_divergence_real_to_synthetic']:.6f} | "
                f"{values['wasserstein_distance_physical_domain']:.6f} | {values['synthetic_outlier_ratio']:.6f} |"
            )
    lines.extend([
        "",
        "## Paper reference values",
        "",
        "`PAPER_MAX_MEAN_PCT_DIFF_REF=0.52`; `PAPER_MAX_STD_PCT_DIFF_REF=2.86`; `PAPER_MAX_KL_REF=0.54`; `PAPER_MAX_WASSERSTEIN_REF=58.73`.",
        "They are recorded for context only. Differences in unreported metric implementation do not by themselves establish paper-reproduction failure.",
        "",
        "## Objective variance facts",
        "",
    ])
    for battery, result in metrics["battery_results"].items():
        variation = result["five_set_variability"]
        lines.append(f"- `{battery}`: `all_sets_have_nonzero_variance={variation['all_sets_have_nonzero_variance']}`; set-level mean and standard-deviation variability is recorded feature-wise in the JSON companion.")
    lines.extend([
        "",
        "The JSON companion contains every per-set and pooled HF1/HF2/HF3 calculation, including means, standard deviations, min/max, NaN/Inf counts, variance, KL, Wasserstein, and IQR-based outlier ratio. This report makes no parameter-change recommendation.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not OUTPUT_ROOT.is_dir():
        raise ProtocolViolation("persisted Phase9C output directory is missing")
    results: dict[str, Any] = {}
    for battery in TARGETS:
        real, reference_detail = first_20_reference_hf(battery)
        sets = checked_synthetic_sets(battery, real.shape[0])
        per_set = {f"set_{index:02d}": dataset_metrics(real, generated) for index, generated in enumerate(sets, start=1)}
        pooled = np.concatenate(sets, axis=0)
        results[battery] = {
            "reference": reference_detail,
            "per_set": per_set,
            "pooled": dataset_metrics(real, pooled),
            "five_set_variability": set_variability(sets),
        }
    metrics = {
        "task": "PHASE_9C_SCFGAN_GENERATION_QUALITY_DIAGNOSTIC",
        "reference_use": "TARGET_FIRST_20_PERCENT_HF__POST_HOC_QUALITY_EVALUATION_ONLY__NOT_GAN_TRAINING",
        "scfgan_retrained": False,
        "downstream_model_trained": False,
        "test_loaded": False,
        "test_access_count": 0,
        "kl_implementation": "TIER2_DETERMINISTIC_DIAGNOSTIC",
        "outlier_implementation": "TIER2_DETERMINISTIC_DIAGNOSTIC",
        "histogram_bins": HISTOGRAM_BINS,
        "histogram_epsilon": EPSILON,
        "paper_reference_values": {
            "paper_max_mean_pct_diff_ref": 0.52,
            "paper_max_std_pct_diff_ref": 2.86,
            "paper_max_kl_ref": 0.54,
            "paper_max_wasserstein_ref": 58.73,
        },
        "battery_results": results,
        "aggregate_maxima": maxima(results),
        "automatic_parameter_change_performed": False,
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({"test_access_count": 0, "aggregate_maxima": metrics["aggregate_maxima"], "next_action": metrics["next_action"]}, indent=2))


if __name__ == "__main__":
    main()
