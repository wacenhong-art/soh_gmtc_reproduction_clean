"""Target-blind source checkpoint selection followed by target HF regeneration.

The selection phase reads only the source-only convergence JSON. Target data
is inaccessible until after selection is persisted in memory. Target charge HF
is then used solely for post-generation diagnostic metrics, never for GAN
training or checkpoint choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_phase9c_scfgan_generation_quality_diagnostic import (
    TARGETS,
    checked_synthetic_sets,
    dataset_metrics,
    first_20_reference_hf,
)
from run_phase9c_scfgan_real_data_smoke import (
    FEATURE_DIM,
    NOISE_DIM,
    ProtocolViolation,
    model_config,
    source_hf_and_condition,
    target_condition_only,
)
from scfgan_structural import ParameterizedSCFGANStructural


ROOT = Path(__file__).resolve().parent.parent
CONVERGENCE_METRICS = ROOT / "PHASE9C_SCFGAN_SOURCE_ONLY_CONVERGENCE_METRICS.json"
OUTPUT_ROOT = ROOT / "output" / "phase9c_source_selected_target_regen"
METRICS_PATH = ROOT / "PHASE9C_SOURCE_CHECKPOINT_TARGET_REGEN_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_SOURCE_CHECKPOINT_TARGET_REGEN_REPORT.md"
SOURCE_BATTERIES = ("B0005", "B0006")
GENERATION_SEED = 42
SCORE_FIELDS = ("mean_diff", "std_diff", "kl", "wasserstein")


def select_source_checkpoint() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Selection-only phase: reads source alignment summaries, never target data."""
    payload = json.loads(CONVERGENCE_METRICS.read_text(encoding="utf-8"))
    snapshots = payload["snapshots"]
    if [item["epoch"] for item in snapshots] != list(range(20, 301, 20)):
        raise ProtocolViolation("incomplete source-only snapshot trajectory")
    baseline = snapshots[0]["trajectory_summary_mean_over_hf"]
    if snapshots[0]["epoch"] != 20 or any(float(baseline[field]) <= 0.0 for field in SCORE_FIELDS):
        raise ProtocolViolation("invalid epoch-20 source-only score baseline")
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        values = snapshot["trajectory_summary_mean_over_hf"]
        normalized = {field: float(values[field]) / float(baseline[field]) for field in SCORE_FIELDS}
        score = float(np.mean(list(normalized.values())))
        rows.append({
            "epoch": int(snapshot["epoch"]),
            "source_snapshot": snapshot["snapshot"],
            "raw_source_metrics": {field: float(values[field]) for field in SCORE_FIELDS},
            "normalized_metrics": normalized,
            "source_alignment_score": score,
            "outlier_ratio_report_only": float(values["outlier_ratio"]),
        })
    selected = min(rows, key=lambda item: (item["source_alignment_score"], item["epoch"]))
    return selected, rows


def source_reference() -> np.ndarray:
    rows: list[np.ndarray] = []
    for battery in SOURCE_BATTERIES:
        hf, _, _ = source_hf_and_condition(battery)
        rows.append(hf)
    source = np.concatenate(rows, axis=0)
    if source.shape != (336, FEATURE_DIM) or not np.isfinite(source).all():
        raise ProtocolViolation("source reference HF is invalid")
    return source


def load_selected_model(snapshot_relative_path: str) -> ParameterizedSCFGANStructural:
    path = ROOT / snapshot_relative_path
    if not path.is_file():
        raise ProtocolViolation(f"selected source snapshot missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise ProtocolViolation("selected checkpoint lacks a state dict")
    model = ParameterizedSCFGANStructural(model_config())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def source_scaler() -> dict[str, Any]:
    """Read the frozen source scaler metadata only; do not fit a new scaler."""
    smoke_metrics = json.loads((ROOT / "PHASE9C_SCFGAN_REAL_DATA_SMOKE_METRICS.json").read_text(encoding="utf-8"))
    scaler = smoke_metrics["scaler"]
    if scaler["normalization"] != "MINMAX_MINUS1_TO_PLUS1" or scaler["fit_scope"] != "B0005_B0006_SOURCE_HF_ONLY":
        raise ProtocolViolation("PLAN-CHANGE-0018 source-only scaler contract mismatch")
    return scaler


def generate_target_sets(model: ParameterizedSCFGANStructural, target: str, scaler: dict[str, Any], generator: torch.Generator) -> tuple[list[np.ndarray], dict[str, Any]]:
    # This function is intentionally condition-metadata-only until all five sets exist.
    condition, detail = target_condition_only(target)
    condition_tensor = torch.tensor(condition, dtype=torch.float32)
    low = np.asarray(scaler["feature_min"], dtype=np.float64)
    high = np.asarray(scaler["feature_max"], dtype=np.float64)
    arrays: list[np.ndarray] = []
    paths: list[str] = []
    for set_index in range(1, 6):
        noise = torch.randn(condition_tensor.shape[0], NOISE_DIM, generator=generator)
        with torch.no_grad():
            generated_normalized = model.generate(
                noise,
                condition_tensor,
                provenance="TARGET_CYCLE_IDENTITY_METADATA_ONLY",
                request_kind="TARGET_CONDITION_GENERATION",
            ).detach().cpu().numpy().astype(np.float64)
        physical = 0.5 * (generated_normalized + 1.0) * (high - low) + low
        if physical.shape != (condition.shape[0], FEATURE_DIM) or not np.isfinite(physical).all():
            raise ProtocolViolation(f"generated target set invalid: {target} set {set_index}")
        path = OUTPUT_ROOT / f"{target}_selected_source_checkpoint_synthetic_hf_set_{set_index:02d}.npy"
        np.save(path, physical, allow_pickle=False)
        arrays.append(physical)
        paths.append(str(path.relative_to(ROOT)))
    return arrays, {"condition_metadata": detail, "set_paths": paths, "shape": [int(condition.shape[0]), FEATURE_DIM]}


def comparison_metrics(source: np.ndarray, target: np.ndarray, generated_sets: list[np.ndarray]) -> dict[str, Any]:
    pooled = np.concatenate(generated_sets, axis=0)
    return {
        "A_GENERATED_VS_SOURCE": {
            "outlier_boundary_reference": "SOURCE_REAL_HF",
            "per_set": {f"set_{index:02d}": dataset_metrics(source, generated) for index, generated in enumerate(generated_sets, start=1)},
            "pooled": dataset_metrics(source, pooled),
        },
        "B_SOURCE_VS_TARGET": {
            "outlier_boundary_reference": "TARGET_REFERENCE_HF",
            "pooled": dataset_metrics(target, source),
        },
        "C_GENERATED_VS_TARGET": {
            "outlier_boundary_reference": "TARGET_REFERENCE_HF",
            "per_set": {f"set_{index:02d}": dataset_metrics(target, generated) for index, generated in enumerate(generated_sets, start=1)},
            "pooled": dataset_metrics(target, pooled),
        },
    }


def target_mismatch_persists(comparisons: dict[str, Any]) -> bool:
    """Descriptive flag, not a causal attribution or model-selection rule."""
    values = list(comparisons["C_GENERATED_VS_TARGET"]["pooled"]["feature_metrics"].values())
    # This classification captures persistent spread/distribution mismatch even
    # when means happen to be close. It neither chooses a checkpoint nor assigns
    # a causal mechanism.
    affected_features = sum(
        item["std_percentage_difference"] >= 40.0
        or item["kl_divergence_real_to_synthetic"] >= 1.0
        for item in values
    )
    return bool(affected_features >= 2)


def pooled_maximum(comparison: dict[str, Any]) -> dict[str, float]:
    values = list(comparison["pooled"]["feature_metrics"].values())
    return {
        "max_mean_percentage_difference": float(max(item["mean_percentage_difference"] for item in values)),
        "max_std_percentage_difference": float(max(item["std_percentage_difference"] for item in values)),
        "max_kl": float(max(item["kl_divergence_real_to_synthetic"] for item in values)),
        "max_wasserstein": float(max(item["wasserstein_distance_physical_domain"] for item in values)),
        "max_outlier_ratio": float(max(item["synthetic_outlier_ratio"] for item in values)),
    }


def write_report(metrics: dict[str, Any]) -> None:
    selected = metrics["selection"]
    lines = [
        "# Phase 9C source-only checkpoint selection and target regeneration",
        "",
        "## Source-only selection",
        "",
        "Checkpoint scoring was completed before any target HF reference was materialized. The score is the unweighted mean of epoch metric / epoch-20 metric for mean difference, standard-deviation difference, KL, and Wasserstein. Outlier ratio was reported but excluded. Ties resolve to the earlier epoch.",
        "",
        f"`SELECTED_EPOCH={selected['epoch']}`  ",
        f"`SELECTED_SOURCE_ALIGNMENT_SCORE={selected['source_alignment_score']}`  ",
        "`CHECKPOINT_SELECTION_TARGET_BLIND=True`",
        "",
        "| epoch | source alignment score | outlier ratio (excluded) |",
        "| ---: | ---: | ---: |",
    ]
    for row in metrics["source_alignment_score_by_epoch"]:
        lines.append(f"| {row['epoch']} | {row['source_alignment_score']:.9f} | {row['outlier_ratio_report_only']:.9f} |")
    lines.extend([
        "",
        "## Regeneration and post-hoc diagnostics",
        "",
        "The selected source-only checkpoint generated five new HF-only sets per target in a separate local directory; existing smoke sets were not overwritten. Target first-20%-life HF was accessed only after generation for the diagnostic comparisons below. Capacity, SOH, validation, and TEST were not accessed.",
        "",
        "| Target | comparison | max mean pct diff | max std pct diff | max KL | max Wasserstein | max outlier ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for target, result in metrics["target_results"].items():
        for name, comparison in result["comparisons"].items():
            maximum = pooled_maximum(comparison)
            lines.append(f"| {target} | {name} | {maximum['max_mean_percentage_difference']:.6f} | {maximum['max_std_percentage_difference']:.6f} | {maximum['max_kl']:.6f} | {maximum['max_wasserstein']:.6f} | {maximum['max_outlier_ratio']:.6f} |")
    lines.extend([
        "",
        "## Objective conclusion fields",
        "",
        f"`SELECTED_SOURCE_ALIGNMENT_STATUS={metrics['selected_source_alignment_status']}`  ",
        f"`TARGET_ALIGNMENT_AFTER_SOURCE_CONVERGENCE={metrics['target_alignment_after_source_convergence']}`  ",
        f"`TARGET_MISMATCH_PERSISTS_AFTER_SOURCE_CONVERGENCE={metrics['target_mismatch_persists_after_source_convergence']}`",
        "",
        "The target-mismatch field is a descriptive result from the registered post-generation statistics. It does not assign a cause to conditioning, voltage semantics, domain shift, or architecture, and it triggered no model/protocol modification.",
        "",
        "```text",
        "TARGET_INFORMATION_USED_FOR_CHECKPOINT_SELECTION=False",
        "TARGET_INFORMATION_USED_FOR_TRAINING=False",
        "TEST_LOADED=False",
        "TEST_ACCESS_COUNT=0",
        "DOWNSTREAM_MODEL_TRAINED=False",
        "GRID_SEARCH_RUN=False",
        "NEXT_ACTION=RETURN_RESULTS_FOR_HUMAN_REVIEW",
        "```",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise ProtocolViolation(f"refusing to overwrite existing regenerated HF output: {OUTPUT_ROOT}")
    # Phase 1: source-only and target-blind checkpoint selection.
    selected, score_rows = select_source_checkpoint()
    model = load_selected_model(selected["source_snapshot"])
    scaler = source_scaler()
    source = source_reference()

    # Phase 2: selected checkpoint generation uses target cycle identity metadata only.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(GENERATION_SEED)
    regenerated: dict[str, tuple[list[np.ndarray], dict[str, Any]]] = {}
    for target in TARGETS:
        regenerated[target] = generate_target_sets(model, target, scaler, generator)

    # Phase 3: target HF becomes available only as a post-generation reference.
    target_results: dict[str, Any] = {}
    mismatch_flags: list[bool] = []
    for target in TARGETS:
        target_hf, target_detail = first_20_reference_hf(target)
        generated_sets, generation_detail = regenerated[target]
        if generated_sets[0].shape[0] != target_hf.shape[0]:
            raise ProtocolViolation("target diagnostic row count mismatch")
        comparisons = comparison_metrics(source, target_hf, generated_sets)
        mismatch_flags.append(target_mismatch_persists(comparisons))
        target_results[target] = {
            "generation": generation_detail,
            "target_reference": target_detail,
            "comparisons": comparisons,
        }
    metrics = {
        "task": "PHASE9C_SOURCE_ONLY_CHECKPOINT_SELECTION_AND_TARGET_REGEN",
        "plan_change": "PLAN-CHANGE-0018__UNCHANGED",
        "selection": selected,
        "source_alignment_score_by_epoch": score_rows,
        "checkpoint_selection_target_blind": True,
        "target_information_used_for_checkpoint_selection": False,
        "target_information_used_for_training": False,
        "selected_source_alignment_status": "IMPROVED_VS_EPOCH_20__SOURCE_ONLY_SCORE_MINIMUM_SELECTED",
        "generation_seed": GENERATION_SEED,
        "target_results": target_results,
        "target_alignment_after_source_convergence": "POST_HOC_TARGET_DIAGNOSTIC__NO_CAUSAL_ATTRIBUTION",
        "target_mismatch_persistence_definition": "AT_LEAST_TWO_POOLED_HF_HAVE_STD_PERCENT_DIFFERENCE_GE_40_OR_KL_GE_1__DESCRIPTIVE_ONLY",
        "target_mismatch_persists_after_source_convergence": bool(all(mismatch_flags)),
        "test_loaded": False,
        "test_access_count": 0,
        "downstream_model_trained": False,
        "grid_search_run": False,
        "scfgan_retrained": False,
        "synthetic_regenerated": True,
        "next_action": "RETURN_RESULTS_FOR_HUMAN_REVIEW",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({"selected_epoch": selected["epoch"], "source_alignment_score": selected["source_alignment_score"], "target_mismatch_persists_after_source_convergence": metrics["target_mismatch_persists_after_source_convergence"], "test_access_count": 0}, indent=2))


if __name__ == "__main__":
    main()
