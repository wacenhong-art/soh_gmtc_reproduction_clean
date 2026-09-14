"""Complete Phase 9A validation error-geometry/baseline diagnostic.

This script consumes the separately verified replay-persisted best checkpoints
and VALIDATION prediction rows. It retains the existing post-split TEST guard
while calculating only pre-authorized descriptive statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from run_phase9a_tier2_paired_train_val_dry_run import (
    ROOT,
    TARGET_BATTERIES,
    _cycles,
    extract_capacity,
    metrics,
    validate_pairing_and_splits,
)
from tier2_phase9a_access import PostSplitPairAccessor, TestLockViolation


OUTPUT = ROOT / "PHASE9A_ERROR_GEOMETRY_BASELINE_METRICS.json"
CONVERGENCE_METRICS = ROOT / "PHASE9A_CONVERGENCE_DIAGNOSTIC_METRICS.json"
REPLAY_METRICS = ROOT / "PHASE9A_CONVERGENCE_REPLAY_METRICS.json"
PREDICTIONS = ROOT / "PHASE9A_CONVERGENCE_VALIDATION_PREDICTIONS.json"
REPORT = ROOT / "PHASE9A_ERROR_GEOMETRY_BASELINE_DIAGNOSTIC.md"


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def capacity_train_val_only(
    battery: str, pairs: list[dict[str, Any]], split: Any
) -> tuple[np.ndarray, dict[str, int]]:
    """Materialize Capacity only for legal TRAIN/VALIDATION pair identities."""
    accessor = PostSplitPairAccessor(_cycles(battery), split.train_end, split.val_end)
    capacities: list[float] = []
    for pair_index, pair in enumerate(pairs):
        if pair_index >= split.val_end:
            continue
        try:
            _, discharge = accessor.records_for_pair(pair_index, pair)
        except TestLockViolation as error:
            raise RuntimeError(f"TEST_LOCK_VIOLATION for {battery}") from error
        capacities.append(extract_capacity(discharge))
    if len(capacities) != split.val_end:
        raise RuntimeError(f"TRAIN/VALIDATION capacity count mismatch for {battery}")
    return np.asarray(capacities, dtype=np.float64), {
        "post_split_test_numerical_access_attempts": accessor.test_numerical_access_attempts,
        "post_split_test_numerical_materialization_count": accessor.post_split_test_numerical_materialization_count,
    }


def baseline_diagnostic(
    battery: str,
    pairs: list[dict[str, Any]],
    split: Any,
    saved: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    capacities, access = capacity_train_val_only(battery, pairs, split)
    train, val = capacities[: split.train_end], capacities[split.train_end : split.val_end]
    train_mean_prediction = np.full_like(val, train.mean())
    persistence_prediction = capacities[split.train_end - 1 : split.val_end - 1]
    if persistence_prediction.shape != val.shape:
        raise RuntimeError(f"persistence alignment mismatch for {battery}")
    expected_indices = list(range(split.train_end, split.val_end))
    observed_indices = [int(row["sample_index"]) for row in prediction_rows]
    if observed_indices != expected_indices:
        raise RuntimeError(f"validation sample-index sequence mismatch for {battery}")
    target = np.asarray([row["target_capacity"] for row in prediction_rows], dtype=np.float64)
    prediction = np.asarray([row["predicted_capacity"] for row in prediction_rows], dtype=np.float64)
    error = np.asarray([row["error"] for row in prediction_rows], dtype=np.float64)
    if not np.allclose(target, val, rtol=0.0, atol=1e-7) or not np.allclose(error, prediction - target, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"persisted validation rows do not reconcile with legal VAL targets for {battery}")
    saved_capacity = saved["validation_metrics_capacity"]
    model_metrics = metrics(prediction, target)
    # Persisted prediction rows are serialized from float32 model/target tensors;
    # the archived payload holds the original scalar reduction.  This tolerance
    # is solely a serialization check, not a result-selection tolerance.
    if not np.isclose(model_metrics["mse"], float(saved_capacity["mse"]), rtol=1e-6, atol=1e-12):
        raise RuntimeError(f"persisted model MSE does not match archived convergence result for {battery}")
    model_sse = float(np.sum(error**2))
    model_sst = float(np.sum((target - np.mean(target)) ** 2))
    target_sst = float(np.sum((val - np.mean(val)) ** 2))
    if not np.isclose(model_sst, target_sst, rtol=1e-5, atol=1e-12):
        raise RuntimeError(f"saved model metrics do not reconcile with current VAL target geometry for {battery}")
    persistence_metrics = metrics(persistence_prediction, target)
    table = [
        {
            "sample_index": int(row["sample_index"]),
            "target_capacity": float(row["target_capacity"]),
            "predicted_capacity": float(row["predicted_capacity"]),
            "error": float(row["error"]),
        }
        for row in prediction_rows
    ]
    return {
        "target_geometry": describe(target),
        "prediction_geometry": describe(prediction),
        "error_geometry": describe(error),
        "model_r2_sse": model_sse,
        "model_r2_sst": model_sst,
        "target_sst_direct": target_sst,
        "baseline_metrics": {
            "train_mean": metrics(train_mean_prediction, val),
            "persistence": persistence_metrics,
        },
        "model_metrics": model_metrics,
        "model_vs_persistence": {
            "rmse_improvement_vs_persistence": float(persistence_metrics["rmse"] - model_metrics["rmse"]),
            "mae_improvement_vs_persistence": float(persistence_metrics["mae"] - model_metrics["mae"]),
            "definition": "PERSISTENCE_METRIC_MINUS_MODEL_METRIC__POSITIVE_MEANS_LOWER_MODEL_ERROR",
        },
        "objective_variance_bias_level_statistics": {
            "target_variance": float(np.var(target, ddof=0)),
            "prediction_variance": float(np.var(prediction, ddof=0)),
            "prediction_to_target_std_ratio": float(np.std(prediction, ddof=0) / np.std(target, ddof=0)),
            "prediction_to_target_range_ratio": float(np.ptp(prediction) / np.ptp(target)),
            "mean_bias_prediction_minus_target": float(np.mean(error)),
            "absolute_mean_bias_over_target_std": float(abs(np.mean(error)) / np.std(target, ddof=0)),
            "sse_over_sst": float(model_sse / model_sst),
            "r2_is_negative": bool(model_metrics["r2"] < 0),
        },
        "validation_time_order_table": table,
        "test_access": access,
    }


def render_report(output: dict[str, Any]) -> str:
    lines = [
        "# Phase 9A Error Geometry and Baseline Diagnostic",
        "",
        "Scope: replay-persisted VALIDATION predictions only. No TEST numerical values were accessed.",
        "",
    ]
    for battery, result in output["results_by_battery"].items():
        target, prediction, error = result["target_geometry"], result["prediction_geometry"], result["error_geometry"]
        baseline, model = result["baseline_metrics"], result["model_metrics"]
        relative = result["model_vs_persistence"]
        lines.extend([
            f"## {battery}",
            "",
            "| Quantity | Mean | Std | Min | Max | Range |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Target capacity | {target['mean']:.12g} | {target['std']:.12g} | {target['min']:.12g} | {target['max']:.12g} | {target['range']:.12g} |",
            f"| Predicted capacity | {prediction['mean']:.12g} | {prediction['std']:.12g} | {prediction['min']:.12g} | {prediction['max']:.12g} | {prediction['range']:.12g} |",
            f"| Error (prediction - target) | {error['mean']:.12g} | {error['std']:.12g} | {error['min']:.12g} | {error['max']:.12g} | n/a |",
            "",
            f"R2_SSE={result['model_r2_sse']:.12g}; R2_SST={result['model_r2_sst']:.12g}; model R2={model['r2']:.12g}.",
            "",
            "| Method | RMSE | MAE | R2 |",
            "|---|---:|---:|---:|",
            f"| Model | {model['rmse']:.12g} | {model['mae']:.12g} | {model['r2']:.12g} |",
            f"| TRAIN mean baseline | {baseline['train_mean']['rmse']:.12g} | {baseline['train_mean']['mae']:.12g} | {baseline['train_mean']['r2']:.12g} |",
            f"| Persistence baseline | {baseline['persistence']['rmse']:.12g} | {baseline['persistence']['mae']:.12g} | {baseline['persistence']['r2']:.12g} |",
            "",
            f"Model minus persistence improvement (positive means lower model error): RMSE={relative['rmse_improvement_vs_persistence']:.12g}; MAE={relative['mae_improvement_vs_persistence']:.12g}.",
            "",
            "Objective variance/bias/level statistics: " + json.dumps(result["objective_variance_bias_level_statistics"], sort_keys=True),
            "",
            "Chronological validation rows are stored in the JSON artifact under `validation_time_order_table` with sample index, target capacity, predicted capacity, and error only.",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    saved = json.loads(CONVERGENCE_METRICS.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_METRICS.read_text(encoding="utf-8"))
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if replay["archive_comparison"] != "PASS__COMPLETE_RESULT_PAYLOAD_EXACT":
        raise RuntimeError("replay archive comparison is not PASS")
    audits, splits = validate_pairing_and_splits()
    results = {
        battery: baseline_diagnostic(
            battery,
            audits[battery]["pair_identities"],
            splits[battery],
            saved["results_by_battery"][battery],
            predictions["rows_by_battery"][battery],
        )
        for battery in TARGET_BATTERIES
    }
    output = {
        "phase": "PHASE9A_B0018_ERROR_GEOMETRY_AND_BASELINE_DIAGNOSTIC",
        "scope": "REPLAY_PERSISTED_VALIDATION_ONLY_ERROR_GEOMETRY_AND_BASELINE_ANALYSIS",
        "model_reexecution": True,
        "model_training": "FROZEN_CONFIGURATION_REPLAY_ONLY",
        "saved_best_checkpoint_available": True,
        "saved_validation_model_outputs_available": True,
        "replay_metrics": str(REPLAY_METRICS.name),
        "validation_predictions": str(PREDICTIONS.name),
        "checkpoint_paths": replay["checkpoint_paths"],
        "test_lock": {
            "test_loaded": False,
            "test_numerical_value_access_count": 0,
            "post_split_test_numerical_access_attempts": sum(
                result["test_access"]["post_split_test_numerical_access_attempts"] for result in results.values()
            ),
            "post_split_test_numerical_materialization_count": sum(
                result["test_access"]["post_split_test_numerical_materialization_count"] for result in results.values()
            ),
            "gate": "PASS",
        },
        "results_by_battery": results,
        "next_action": "RETURN_DIAGNOSTICS_FOR_HUMAN_REVIEW",
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    REPORT.write_text(render_report(output), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
