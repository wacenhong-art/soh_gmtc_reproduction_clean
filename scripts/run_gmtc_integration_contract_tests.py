"""Synthetic metadata-only tests for the Phase 8A integration validator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from gmtc_integration_contract import (  # noqa: E402
    BranchPredictionMetadata,
    FusionReadinessMetadata,
    GMTCIntegrationContractError,
    validate_contract_integration,
    validate_numeric_integration,
)


def branch(branch_name: str, **changes: object) -> BranchPredictionMetadata:
    values: dict[str, object] = {
        "branch_name": branch_name,
        "battery_id": "SYNTHETIC_BATTERY_METADATA",
        "cycle_identity": "SYNTHETIC_CYCLE_001",
        "source_target_role": "TARGET_METADATA_ONLY",
        "real_synthetic_indicator": "SYNTHETIC_METADATA_ONLY",
        "synthetic_set_id": "NOT_MATERIALIZED",
        "protocol": "PROTOCOL_B_METADATA_ONLY",
        "partition": "TRAIN",
        "window_start_cycle_identity": "SYNTHETIC_CYCLE_000",
        "window_end_cycle_identity": "SYNTHETIC_CYCLE_001",
        "target_cycle_identity": "SYNTHETIC_CYCLE_001",
        "feature_order": ("HF1", "HF2", "HF3"),
        "feature_dim": 3,
        "feature_scale_status": "UNRESOLVED",
        "target_alignment_status": "UNRESOLVED",
        "prediction_scale_status": "UNRESOLVED",
        "model_fidelity_status": "STRUCTURAL_ONLY",
        "provenance": "SYNTHETIC_GMTC_CONTRACT_FIXTURE",
    }
    values.update(changes)
    return BranchPredictionMetadata(**values)


def fusion(**changes: object) -> FusionReadinessMetadata:
    values: dict[str, str] = {
        "component_error_definition_status": "UNRESOLVED",
        "component_ratio_definition_status": "UNRESOLVED",
        "fusion_d_status": "UNRESOLVED",
        "weighted_feature_selection_status": "UNRESOLVED",
        "final_output_domain_status": "UNRESOLVED",
        "full_loss_composition_status": "UNRESOLVED",
        "provenance": "SYNTHETIC_GMTC_CONTRACT_FIXTURE",
    }
    values.update(changes)
    return FusionReadinessMetadata(**values)


def expected_failure(callback) -> bool:
    try:
        callback()
    except GMTCIntegrationContractError:
        return True
    return False


def main() -> None:
    valid = (branch("TREND"), branch("SEASONAL"), branch("RESIDUAL"))
    fusion_metadata = fusion()
    validate_contract_integration(valid, fusion_metadata)
    fail_closed = {
        "missing_scale_metadata": expected_failure(lambda: validate_contract_integration((branch("TREND", feature_scale_status=""), valid[1], valid[2]), fusion_metadata)),
        "unknown_target_alignment_numeric": expected_failure(lambda: validate_numeric_integration(valid, fusion_metadata)),
        "mismatched_cycle_identity": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", cycle_identity="SYNTHETIC_CYCLE_002"), valid[2]), fusion_metadata)),
        "mixed_partition": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", partition="VALIDATION"), valid[2]), fusion_metadata)),
        "validation_provenance": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", provenance="SYNTHETIC_VALIDATION"), valid[2]), fusion_metadata)),
        "test_provenance": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", provenance="SYNTHETIC_TEST"), valid[2]), fusion_metadata)),
        "feature_dim_not_3": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", feature_dim=4, feature_order=("HF1", "HF2", "HF3", "GENERIC_FEATURE_4")), valid[2]), fusion_metadata)),
        "synthetic_soh_request": expected_failure(lambda: validate_contract_integration((valid[0], branch("SEASONAL", provenance="SYNTHETIC_SOH_REQUEST"), valid[2]), fusion_metadata)),
        "fusion_semantics_unresolved_numeric": expected_failure(lambda: validate_numeric_integration(valid, fusion_metadata)),
    }
    audit = {
        "artifact_type": "GMTC_LEVEL1_METADATA_ONLY_CONTRACT_AUDIT",
        "real_data_used": False,
        "real_synthetic_rows_created": False,
        "test_loaded": False,
        "validation_values_used": False,
        "model_trained": False,
        "grid_search_run": False,
        "numeric_full_pipeline_executed": False,
        "contract_integration_gate": "PASS",
        "synthetic_numeric_integration_gate": "BLOCKED_BY_UNRESOLVED_SCALE_ALIGNMENT_AND_FUSION_SEMANTICS",
        "real_data_integration_gate": "BLOCKED_BY_RETAINED_FIDELITY_BLOCKERS",
        "fail_closed_tests": fail_closed,
        "all_tests_passed": all(fail_closed.values()),
    }
    output_path = PROJECT_ROOT / "output" / "integration" / "gmtc_contract_integration_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["all_tests_passed"]:
        raise SystemExit("GMTC integration contract tests failed")
    print(output_path)


if __name__ == "__main__":
    main()
