"""Replay the frozen Phase 9A convergence diagnostic and persist legal VAL artifacts.

The replay is fail-closed: it persists no checkpoint or prediction until both
same-seed executions agree exactly and agree exactly with the archived
convergence payload.  Only TRAIN/VALIDATION values are accessed through the
existing guarded runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from run_phase9a_convergence_diagnostic import (
    CONFIG,
    KINDS,
    OUTPUT as ARCHIVED_OUTPUT,
    ROOT,
    ProtocolViolation,
    comparable,
    one_full_diagnostic,
    validate_pairing_and_splits,
)


REPLAY_OUTPUT = ROOT / "PHASE9A_CONVERGENCE_REPLAY_METRICS.json"
PREDICTIONS_OUTPUT = ROOT / "PHASE9A_CONVERGENCE_VALIDATION_PREDICTIONS.json"
CHECKPOINT_DIRECTORY = ROOT / "output" / "phase9a_convergence_replay"


def persist(artifacts: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Persist only restored best states and legal validation prediction rows."""
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[str, str] = {}
    prediction_payload: dict[str, Any] = {
        "phase": "PHASE9A_CONVERGENCE_REPLAY_VALIDATION_PREDICTIONS",
        "scope": "VALIDATION_ONLY__NO_TEST_VALUES",
        "columns": ["sample_index", "target_capacity", "predicted_capacity", "error"],
        "rows_by_battery": {},
    }
    for battery, artifact in artifacts.items():
        checkpoint_path = CHECKPOINT_DIRECTORY / f"{battery}_BEST_CHECKPOINT_DRY_RUN_ONLY.pt"
        torch.save(artifact["checkpoint"], checkpoint_path)
        checkpoint_paths[battery] = str(checkpoint_path.relative_to(ROOT))
        prediction_payload["rows_by_battery"][battery] = artifact["validation_prediction_rows"]
    PREDICTIONS_OUTPUT.write_text(json.dumps(prediction_payload, indent=2), encoding="utf-8")
    return checkpoint_paths


def main() -> None:
    CONFIG.validate()
    archived = json.loads(ARCHIVED_OUTPUT.read_text(encoding="utf-8"))
    audits, splits = validate_pairing_and_splits()
    first, artifacts = one_full_diagnostic(audits, splits, capture_persistence_artifacts=True)
    second = one_full_diagnostic(audits, splits)
    if not comparable(first, second):
        raise ProtocolViolation("same-seed replay payload mismatch; persistence blocked")
    if not comparable(first, archived["results_by_battery"]):
        raise ProtocolViolation("replay differs from archived convergence payload; persistence blocked")
    checkpoint_paths = persist(artifacts)
    output = {
        "phase": "PHASE9A_CONVERGENCE_REPLAY_PERSISTENCE",
        "authority": "EXPLICIT_HUMAN_AUTHORIZATION__SAME_FROZEN_CONFIGURATION",
        "archive_comparison": "PASS__COMPLETE_RESULT_PAYLOAD_EXACT",
        "same_seed_replay_comparison": "PASS__COMPLETE_RESULT_PAYLOAD_EXACT",
        "best_checkpoint_restored_and_persisted": True,
        "checkpoint_paths": checkpoint_paths,
        "validation_predictions_path": str(PREDICTIONS_OUTPUT.relative_to(ROOT)),
        "results_by_battery": first,
        "test_lock": {
            "test_loaded": False,
            "test_numerical_value_access_count": 0,
            "post_split_test_numerical_access_attempts": 0,
            "post_split_test_numerical_materialization_count": 0,
            "gate": "PASS",
        },
        "scfgan_trained": False,
        "grid_search_run": False,
    }
    REPLAY_OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(REPLAY_OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
