"""Synthetic-only G-MTC routing scaffold; no TimesNet or fusion implementation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import torch
from torch import Tensor

from itransformer_structural import ParameterizedITransformerStructural


COMPONENTS = frozenset({"TREND", "SEASONAL", "RESIDUAL"})


class GMTCScaffoldError(ValueError):
    """Raised when a synthetic-only G-MTC contract is violated."""


class GMTCScaffoldProvenanceError(GMTCScaffoldError):
    """Raised when non-synthetic or locked-test provenance is supplied."""


def _validate_synthetic_provenance(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GMTCScaffoldProvenanceError("explicit provenance is required")
    normalized = value.strip().upper()
    tokens = set(token for token in re.split(r"[^A-Z0-9]+", normalized) if token)
    forbidden = {"NASA", "CALCE", "TEST", "VALIDATION", "TRAIN", "REAL"}
    if "SYNTHETIC" not in normalized or tokens & forbidden:
        raise GMTCScaffoldProvenanceError("synthetic-only scaffold rejects real, partition, and TEST provenance")
    return normalized


def _validate_tensor(tensor: Tensor, *, seq_len: int) -> None:
    if not isinstance(tensor, Tensor) or tensor.ndim != 3:
        raise GMTCScaffoldError("component input must be [batch, seq_len, n_vars]")
    if tensor.shape[1] != seq_len:
        raise GMTCScaffoldError("component input does not satisfy the explicit sequence contract")
    if not torch.isfinite(tensor).all():
        raise GMTCScaffoldError("component input contains NaN or Inf")


@dataclass(frozen=True)
class ComponentInput:
    component_type: str
    input_tensor: Tensor
    seq_len: int
    batch_ids: tuple[str, ...]
    time_alignment: str
    forecast_horizon: str
    sample_provenance: str
    partition_provenance: str
    fidelity_status: str
    synthetic_only: bool = True

    def validate(self, *, strict_seq_len: int = 10) -> None:
        component = self.component_type.upper()
        if component not in COMPONENTS:
            raise GMTCScaffoldError("component_type must be TREND, SEASONAL, or RESIDUAL")
        if self.seq_len != strict_seq_len:
            raise GMTCScaffoldError("strict paper scaffold requires seq_len=10")
        _validate_tensor(self.input_tensor, seq_len=self.seq_len)
        if len(self.batch_ids) != self.input_tensor.shape[0] or not all(self.batch_ids):
            raise GMTCScaffoldError("batch_ids must uniquely describe every batch row")
        if len(set(self.batch_ids)) != len(self.batch_ids):
            raise GMTCScaffoldError("batch_ids may not repeat")
        if not self.synthetic_only:
            raise GMTCScaffoldProvenanceError("synthetic-only scaffold requires synthetic_only=True")
        _validate_synthetic_provenance(self.sample_provenance)
        _validate_synthetic_provenance(self.partition_provenance)
        if not self.time_alignment or not self.forecast_horizon:
            raise GMTCScaffoldError("explicit time alignment and horizon metadata are required")


@dataclass(frozen=True)
class BranchPrediction:
    branch_name: str
    component_type: str
    output_prediction: Tensor
    seq_len: int
    pred_len: int
    batch_ids: tuple[str, ...]
    time_alignment: str
    forecast_horizon: str
    sample_provenance: str
    partition_provenance: str
    model_fidelity_status: str
    synthetic_only: bool = True

    def validate(self) -> None:
        if self.component_type.upper() not in COMPONENTS:
            raise GMTCScaffoldError("prediction component is invalid")
        if not isinstance(self.output_prediction, Tensor) or self.output_prediction.ndim != 3:
            raise GMTCScaffoldError("branch output must be [batch, pred_len, n_vars]")
        if self.output_prediction.shape[0] != len(self.batch_ids):
            raise GMTCScaffoldError("branch output batch shape does not match batch_ids")
        if self.output_prediction.shape[1] != self.pred_len or self.pred_len < 1:
            raise GMTCScaffoldError("branch output prediction horizon is invalid")
        if not torch.isfinite(self.output_prediction).all():
            raise GMTCScaffoldError("branch output contains NaN or Inf")
        if not self.synthetic_only:
            raise GMTCScaffoldProvenanceError("branch prediction is not marked synthetic-only")
        _validate_synthetic_provenance(self.sample_provenance)
        _validate_synthetic_provenance(self.partition_provenance)


class TrendModelPort:
    """Required TimesNet port; deliberately unavailable until Phase 5."""

    implementation = "NOT_AVAILABLE_YET"
    required_phase = "Phase_5"

    def predict(self, component: ComponentInput) -> BranchPrediction:
        raise GMTCScaffoldError("TrendModelPort is an unimplemented TimesNet port, not a substitute model")


class SeasonalModelPort:
    """Required TimesNet-variant port; deliberately unavailable until Phase 5."""

    implementation = "NOT_AVAILABLE_YET"
    required_phase = "Phase_5"

    def predict(self, component: ComponentInput) -> BranchPrediction:
        raise GMTCScaffoldError("SeasonalModelPort is an unimplemented TimesNet port, not a substitute model")


class DeterministicMockBranch:
    """Routing-only mock. This is explicitly not a model-reproduction branch."""

    mock_is_not_model_reproduction = True

    def __init__(self, component_type: str) -> None:
        self.component_type = component_type.upper()
        if self.component_type not in {"TREND", "SEASONAL"}:
            raise GMTCScaffoldError("mocks are permitted only for Trend and Seasonal routing tests")
        self.branch_name = f"MOCK_{self.component_type}_BRANCH"

    def predict(self, component: ComponentInput) -> BranchPrediction:
        component.validate()
        if component.component_type.upper() != self.component_type:
            raise GMTCScaffoldError("mock branch received an incorrect component")
        prediction = component.input_tensor[:, -1:, :].clone()
        return BranchPrediction(
            branch_name=self.branch_name,
            component_type=self.component_type,
            output_prediction=prediction,
            seq_len=component.seq_len,
            pred_len=1,
            batch_ids=component.batch_ids,
            time_alignment=component.time_alignment,
            forecast_horizon=component.forecast_horizon,
            sample_provenance=component.sample_provenance,
            partition_provenance=component.partition_provenance,
            model_fidelity_status="MOCK__NOT_MODEL_REPRODUCTION",
            synthetic_only=True,
        )


class StructuralResidualBranchAdapter:
    """Adapter for the Phase 3B synthetic-only standard iTransformer core."""

    def __init__(self, model: ParameterizedITransformerStructural) -> None:
        self.model = model

    def predict(self, component: ComponentInput) -> BranchPrediction:
        component.validate()
        if component.component_type.upper() != "RESIDUAL":
            raise GMTCScaffoldError("residual adapter received a non-residual component")
        output, _ = self.model.forward_with_trace(
            component.input_tensor,
            provenance=component.sample_provenance,
        )
        return BranchPrediction(
            branch_name="STRUCTURAL_RESIDUAL_ITRANSFORMER_ADAPTER",
            component_type="RESIDUAL",
            output_prediction=output,
            seq_len=component.seq_len,
            pred_len=int(output.shape[1]),
            batch_ids=component.batch_ids,
            time_alignment=component.time_alignment,
            forecast_horizon=component.forecast_horizon,
            sample_provenance=component.sample_provenance,
            partition_provenance=component.partition_provenance,
            model_fidelity_status="STRUCTURALLY_IMPLEMENTED_SYNTHETIC_ONLY",
            synthetic_only=True,
        )


class GMTCFrameworkScaffold:
    """Build a future-fusion payload without executing any fusion computation."""

    fusion_executed = False

    def build_future_fusion_payload(self, predictions: Iterable[BranchPrediction]) -> dict[str, object]:
        items = tuple(predictions)
        if len(items) != 3:
            raise GMTCScaffoldError("exactly three branch predictions are required")
        for item in items:
            item.validate()
        component_names = tuple(item.component_type.upper() for item in items)
        if set(component_names) != COMPONENTS or len(set(component_names)) != 3:
            raise GMTCScaffoldError("payload requires exactly one TREND, SEASONAL, and RESIDUAL prediction")
        reference = items[0]
        for item in items[1:]:
            attributes = ("batch_ids", "time_alignment", "forecast_horizon", "sample_provenance", "partition_provenance", "pred_len")
            if any(getattr(item, name) != getattr(reference, name) for name in attributes):
                raise GMTCScaffoldError("branch predictions are misaligned; coercion is prohibited")
        by_component = {item.component_type.upper(): item for item in items}
        return {
            "artifact_type": "FUTURE_FUSION_PAYLOAD_SCHEMA_ONLY",
            "fusion_executed": False,
            "branch_predictions": by_component,
            "feature_slots": {
                "predictions": tuple(sorted(COMPONENTS)),
                "errors": tuple(sorted(COMPONENTS)),
                "ratios": tuple(sorted(COMPONENTS)),
            },
            "component_ratio_definition_status": "UNRESOLVED",
            "fusion_feature_d_status": "UNRESOLVED",
            "alignment": {
                "batch_ids": reference.batch_ids,
                "time_alignment": reference.time_alignment,
                "forecast_horizon": reference.forecast_horizon,
                "sample_provenance": reference.sample_provenance,
                "partition_provenance": reference.partition_provenance,
            },
        }
