"""Purely diagnostic Phase 9C source/target alignment audit.

No GAN checkpoint is loaded, no synthetic HF is regenerated, and no model is
trained. The only target values materialized are first-20%-life paired-charge
HF reference values for post-hoc diagnostics; capacity, SOH, validation, and
TEST numeric values remain outside this runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from run_phase9c_scfgan_generation_quality_diagnostic import (
    FEATURE_NAMES,
    TARGETS,
    checked_synthetic_sets,
    dataset_metrics,
    first_20_reference_hf,
)
from run_phase9c_scfgan_real_data_smoke import source_hf_and_condition


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "PHASE9C_SCFGAN_SOURCE_TARGET_ALIGNMENT_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_SCFGAN_SOURCE_TARGET_ALIGNMENT_REPORT.md"
SOURCE_BATTERIES = ("B0005", "B0006")
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class ProtocolViolation(RuntimeError):
    """Raised when the frozen provenance or artifact contract is not met."""


def source_reference() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hfs, conditions, details = [], [], {}
    for battery in SOURCE_BATTERIES:
        hf, condition, detail = source_hf_and_condition(battery)
        hfs.append(hf)
        conditions.append(condition)
        details[battery] = detail
    source_hf = np.concatenate(hfs, axis=0)
    source_condition = np.concatenate(conditions, axis=0).reshape(-1)
    if source_hf.shape != (336, 3) or source_condition.shape != (336,) or not np.isfinite(source_hf).all():
        raise ProtocolViolation("source-HF / condition audit tensor is invalid")
    return source_hf, source_condition, details


def bin_mask(condition: np.ndarray, index: int) -> np.ndarray:
    lower, upper = BIN_EDGES[index], BIN_EDGES[index + 1]
    return (condition >= lower) & ((condition < upper) if index < 4 else (condition <= upper))


def condition_bin_statistics(source_hf: np.ndarray, source_condition: np.ndarray, synthetic_hf: np.ndarray, synthetic_condition: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index in range(5):
        source_values = source_hf[bin_mask(source_condition, index)]
        synthetic_values = synthetic_hf[bin_mask(synthetic_condition, index)]
        def values_summary(values: np.ndarray) -> dict[str, Any]:
            if values.shape[0] == 0:
                return {"count": 0, "mean": None, "std": None}
            return {
                "count": int(values.shape[0]),
                "mean": {name: float(values[:, feature].mean()) for feature, name in enumerate(FEATURE_NAMES)},
                "std": {name: float(values[:, feature].std()) for feature, name in enumerate(FEATURE_NAMES)},
            }
        rows[f"bin_{index + 1}"] = {
            "interval": "[0.8,1.0]" if index == 4 else f"[{BIN_EDGES[index]:.1f},{BIN_EDGES[index + 1]:.1f})",
            "source": values_summary(source_values),
            "synthetic": values_summary(synthetic_values),
        }
    return rows


def max_fields(metrics: dict[str, Any]) -> dict[str, float]:
    features = list(metrics["feature_metrics"].values())
    return {
        "mean_percentage_difference": float(max(value["mean_percentage_difference"] for value in features)),
        "std_percentage_difference": float(max(value["std_percentage_difference"] for value in features)),
        "kl_divergence": float(max(value["kl_divergence_real_to_synthetic"] for value in features)),
        "wasserstein_distance": float(max(value["wasserstein_distance_physical_domain"] for value in features)),
        "outlier_ratio": float(max(value["synthetic_outlier_ratio"] for value in features)),
    }


def all_generated_source_not_worse_than_generated_target(generated_source: dict[str, Any], generated_target: dict[str, Any]) -> bool:
    # A value is comparable only within the same battery, HF, and metric.
    pairs = ((source, target) for source, target in zip(generated_source["feature_metrics"].values(), generated_target["feature_metrics"].values()))
    fields = ("mean_percentage_difference", "std_percentage_difference", "kl_divergence_real_to_synthetic", "wasserstein_distance_physical_domain", "synthetic_outlier_ratio")
    return all(source[field] <= target[field] for source, target in pairs for field in fields)


def write_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 9C SCF-GAN source-target alignment diagnostic",
        "",
        "## Scope",
        "",
        "This is a post-hoc diagnostic only. It did not retrain the GAN, regenerate synthetic HF, train a downstream model, change PLAN-CHANGE-0018, or access TEST. B0005/B0006 provides the permitted real source-HF reference. B0007/B0018 first-20%-life paired-charge HF provides a target-quality reference only; no target reference entered GAN training.",
        "",
        "## Condition provenance audit",
        "",
        "```text",
        "CURRENT_CONDITION_IMPLEMENTATION=NORMALIZED_CYCLE_POSITION",
        "PAPER_EQ5_CONDITION_TEXT=CYCLE_PERIOD",
        "PAPER_EXPERIMENT_CONDITION_TEXT=VOLTAGE_RANGE",
        "PAPER_CONDITION_SEMANTICS_CONTRADICTION=True",
        "```",
        "",
        "## Alignment classifications",
        "",
        f"`GEN_SOURCE_ALIGNMENT_STATUS={metrics['gen_source_alignment_status']}`  ",
        f"`SOURCE_TARGET_DOMAIN_SHIFT_STATUS={metrics['source_target_domain_shift_status']}`  ",
        f"`CONDITIONWISE_ALIGNMENT_STATUS={metrics['conditionwise_alignment_status']}`",
        "",
        "## Pooled maxima by comparison",
        "",
        "| Target | comparison | max mean pct diff | max std pct diff | max KL | max Wasserstein | max outlier ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for battery, result in metrics["battery_results"].items():
        for name, comparison in result["comparisons"].items():
            maxima = max_fields(comparison["pooled"])
            lines.append(f"| {battery} | {name} | {maxima['mean_percentage_difference']:.6f} | {maxima['std_percentage_difference']:.6f} | {maxima['kl_divergence']:.6f} | {maxima['wasserstein_distance']:.6f} | {maxima['outlier_ratio']:.6f} |")
    lines.extend([
        "",
        "## Condition-bin observability",
        "",
        "Source condition positions span the full lifecycle. Existing target synthetic sets were generated only for target first-20%-life positions; therefore only `[0,.2)` has synthetic support. Empty bins are reported as empty rather than filled or extrapolated.",
        "",
        "The JSON companion contains all per-HF, per-set, pooled, and five-bin source/synthetic mean and standard-deviation statistics. It records the deterministic KL, physical-domain Wasserstein, and reference-IQR outlier definitions. No automatic parameter change is implied.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_hf, source_condition, source_details = source_reference()
    results: dict[str, Any] = {}
    source_better_flags: list[bool] = []
    for battery in TARGETS:
        target_hf, target_detail = first_20_reference_hf(battery)
        sets = checked_synthetic_sets(battery, target_hf.shape[0])
        pooled_generated = np.concatenate(sets, axis=0)
        generated_condition = np.tile(np.linspace(0.0, (target_hf.shape[0] - 1) / (target_detail["valid_pair_count_metadata"] - 1), target_hf.shape[0]), len(sets))
        generated_source_sets = {f"set_{index:02d}": dataset_metrics(source_hf, generated) for index, generated in enumerate(sets, start=1)}
        generated_target_sets = {f"set_{index:02d}": dataset_metrics(target_hf, generated) for index, generated in enumerate(sets, start=1)}
        generated_source_pooled = dataset_metrics(source_hf, pooled_generated)
        source_target = dataset_metrics(target_hf, source_hf)  # target defines the IQR boundary for comparison B.
        generated_target_pooled = dataset_metrics(target_hf, pooled_generated)
        source_better_flags.append(all_generated_source_not_worse_than_generated_target(generated_source_pooled, generated_target_pooled))
        results[battery] = {
            "target_reference": target_detail,
            "comparisons": {
                "A_GENERATED_VS_SOURCE": {"outlier_boundary_reference": "SOURCE_REAL_HF", "per_set": generated_source_sets, "pooled": generated_source_pooled},
                "B_SOURCE_VS_TARGET": {"outlier_boundary_reference": "TARGET_REFERENCE_HF", "pooled": source_target},
                "C_GENERATED_VS_TARGET": {"outlier_boundary_reference": "TARGET_REFERENCE_HF", "per_set": generated_target_sets, "pooled": generated_target_pooled},
            },
            "condition_bin_source_vs_generated": condition_bin_statistics(source_hf, source_condition, pooled_generated, generated_condition),
        }
    # These labels are descriptive classifications, not quality gates or tuning rules.
    generated_source_max_kl = max(max_fields(item["comparisons"]["A_GENERATED_VS_SOURCE"]["pooled"])["kl_divergence"] for item in results.values())
    generated_source_status = "POOR__DESCRIPTIVE_MULTI_METRIC_MISMATCH_OBSERVED" if generated_source_max_kl > 0.0 else "EXACT_MATCH__DEGENERATE_CASE"
    source_target_status = "PRESENT__DESCRIPTIVE_SOURCE_TARGET_DIFFERENCES_OBSERVED"
    condition_status = "LIMITED__SYNTHETIC_TARGET_SUPPORT_ONLY_IN_FIRST_20_PERCENT_BIN"
    metrics = {
        "task": "PHASE9C_SCFGAN_SOURCE_TARGET_ALIGNMENT_DIAGNOSTIC",
        "source_batteries": list(SOURCE_BATTERIES),
        "target_batteries": list(TARGETS),
        "source_real_hf_training_scope": "B0005_B0006__EXISTING_LEGAL_SCFGAN_SOURCE_RANGE",
        "target_reference_hf_scope": "FIRST_20_PERCENT_LIFE__POST_HOC_DIAGNOSTIC_ONLY",
        "scfgan_retrained": False,
        "synthetic_regenerated": False,
        "downstream_model_trained": False,
        "test_loaded": False,
        "test_access_count": 0,
        "condition_provenance_audit": {
            "current_condition_implementation": "NORMALIZED_CYCLE_POSITION",
            "paper_eq5_condition_text": "CYCLE_PERIOD",
            "paper_experiment_condition_text": "VOLTAGE_RANGE",
            "paper_condition_semantics_contradiction": True,
        },
        "metric_implementation": "SAME_AS_PHASE9C_GENERATION_QUALITY_DIAGNOSTIC__20_BIN_COMMON_RANGE_KL__EPSILON_1E_MINUS_8__PHYSICAL_WASSERSTEIN__REFERENCE_IQR_OUTLIER_RATIO",
        "source_detail": source_details,
        "battery_results": results,
        "gen_source_alignment_status": generated_source_status,
        "source_target_domain_shift_status": source_target_status,
        "generated_vs_source_clearly_better_than_generated_vs_target_all_metrics": bool(all(source_better_flags)),
        "conditionwise_alignment_status": condition_status,
        "automatic_parameter_change_performed": False,
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({key: metrics[key] for key in ("gen_source_alignment_status", "source_target_domain_shift_status", "conditionwise_alignment_status", "test_access_count", "next_action")}, indent=2))


if __name__ == "__main__":
    main()
