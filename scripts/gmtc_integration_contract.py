"""Metadata-only validator for Level-1 G-MTC integration readiness.

It validates semantic/provenance metadata and deliberately refuses any numeric
pipeline when scale, alignment, or fusion-feature semantics are unresolved.
No battery signal, component tensor, model, loss, or generated feature is
accepted by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


class GMTCIntegrationContractError(ValueError):
    """Raised when a metadata-only integration contract is incomplete or unsafe."""


CANONICAL_FEATURE_ORDER = ("HF1", "HF2", "HF3")
BRANCHES = ("TREND", "SEASONAL", "RESIDUAL")


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^A-Z0-9]+", value.upper()) if token}


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GMTCIntegrationContractError(f"{name} must be an explicit non-empty string")


def _reject_provenance(provenance: str) -> None:
    _require_nonempty("provenance", provenance)
    forbidden = {"NASA", "CALCE", "VALIDATION", "TEST"}
    normalized = provenance.upper()
    if _tokens(provenance) & forbidden or "TRAIN_REAL" in normalized or "SYNTHETIC_SOH" in normalized:
        raise GMTCIntegrationContractError("metadata-only integration rejects real-battery, validation, and TEST provenance")
    if "SYNTHETIC" not in provenance.upper():
        raise GMTCIntegrationContractError("metadata-only integration requires explicit SYNTHETIC provenance")


@dataclass(frozen=True)
class BranchPredictionMetadata:
    """Non-numeric future prediction payload identity and semantic metadata."""

    branch_name: str
    battery_id: str
    cycle_identity: str
    source_target_role: str
    real_synthetic_indicator: str
    synthetic_set_id: str
    protocol: str
    partition: str
    window_start_cycle_identity: str
    window_end_cycle_identity: str
    target_cycle_identity: str
    feature_order: tuple[str, ...]
    feature_dim: int
    feature_scale_status: str
    target_alignment_status: str
    prediction_scale_status: str
    model_fidelity_status: str
    provenance: str

    def validate_contract_level(self) -> None:
        if self.branch_name not in BRANCHES:
            raise GMTCIntegrationContractError("branch_name must be exactly TREND, SEASONAL, or RESIDUAL")
        for name in (
            "battery_id", "cycle_identity", "source_target_role", "real_synthetic_indicator", "protocol", "partition",
            "window_start_cycle_identity", "window_end_cycle_identity", "target_cycle_identity", "feature_scale_status",
            "target_alignment_status", "prediction_scale_status", "model_fidelity_status", "provenance",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.real_synthetic_indicator != "SYNTHETIC_METADATA_ONLY":
            raise GMTCIntegrationContractError("Level-1 validator accepts only SYNTHETIC_METADATA_ONLY payloads")
        if self.synthetic_set_id != "NOT_MATERIALIZED":
            raise GMTCIntegrationContractError("synthetic_set_id must state NOT_MATERIALIZED at Level 1")
        if self.partition != "TRAIN":
            raise GMTCIntegrationContractError("integration contract accepts only TRAIN metadata")
        if self.feature_order != CANONICAL_FEATURE_ORDER or self.feature_dim != 3:
            raise GMTCIntegrationContractError("current Human fallback requires canonical [HF1, HF2, HF3] with feature_dim=3")
        _reject_provenance(self.provenance)

    def validate_numeric_level(self) -> None:
        self.validate_contract_level()
        unresolved = {
            "feature_scale_status": self.feature_scale_status,
            "target_alignment_status": self.target_alignment_status,
            "prediction_scale_status": self.prediction_scale_status,
        }
        if any(value != "RESOLVED" for value in unresolved.values()):
            raise GMTCIntegrationContractError("numeric integration is blocked by unresolved scale or target alignment")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionReadinessMetadata:
    """Non-numeric fusion seam metadata; no B×T×9 feature tensor is accepted."""

    component_error_definition_status: str
    component_ratio_definition_status: str
    fusion_d_status: str
    weighted_feature_selection_status: str
    final_output_domain_status: str
    full_loss_composition_status: str
    provenance: str

    def validate_contract_level(self) -> None:
        for name in (
            "component_error_definition_status", "component_ratio_definition_status", "fusion_d_status",
            "weighted_feature_selection_status", "final_output_domain_status", "full_loss_composition_status", "provenance",
        ):
            _require_nonempty(name, getattr(self, name))
        _reject_provenance(self.provenance)

    def validate_numeric_level(self) -> None:
        self.validate_contract_level()
        values = (
            self.component_error_definition_status,
            self.component_ratio_definition_status,
            self.fusion_d_status,
            self.weighted_feature_selection_status,
            self.final_output_domain_status,
            self.full_loss_composition_status,
        )
        if any(value != "RESOLVED" for value in values):
            raise GMTCIntegrationContractError("numeric integration is blocked by unresolved fusion semantics")


def validate_contract_integration(
    branches: tuple[BranchPredictionMetadata, ...], fusion: FusionReadinessMetadata
) -> None:
    """Validate Level-1 schemas without resolving any unknown semantic field."""
    if len(branches) != 3 or {branch.branch_name for branch in branches} != set(BRANCHES):
        raise GMTCIntegrationContractError("contract requires exactly one TREND, SEASONAL, and RESIDUAL metadata payload")
    for branch in branches:
        branch.validate_contract_level()
    shared = {
        "battery_id": {branch.battery_id for branch in branches},
        "cycle_identity": {branch.cycle_identity for branch in branches},
        "protocol": {branch.protocol for branch in branches},
        "partition": {branch.partition for branch in branches},
        "window_start_cycle_identity": {branch.window_start_cycle_identity for branch in branches},
        "window_end_cycle_identity": {branch.window_end_cycle_identity for branch in branches},
        "target_cycle_identity": {branch.target_cycle_identity for branch in branches},
    }
    if any(len(values) != 1 for values in shared.values()):
        raise GMTCIntegrationContractError("branch metadata must share battery, cycle, protocol, partition, window, and target identity")
    fusion.validate_contract_level()


def validate_numeric_integration(
    branches: tuple[BranchPredictionMetadata, ...], fusion: FusionReadinessMetadata
) -> None:
    """Fail closed until every numeric semantic seam is explicitly resolved."""
    validate_contract_integration(branches, fusion)
    for branch in branches:
        branch.validate_numeric_level()
    fusion.validate_numeric_level()
