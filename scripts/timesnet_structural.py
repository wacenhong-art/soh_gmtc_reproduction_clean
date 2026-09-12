"""Parameterized, synthetic-only standard TimesNet core.

This module is a reference-architecture structural fixture. It does not select
the target paper's unknown TimesNet branch variants, input tensor, target
alignment, or hyperparameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import torch
from torch import Tensor, nn

from gmtc_framework_scaffold import BranchPrediction, ComponentInput, GMTCScaffoldError


class TimesNetConfigError(ValueError):
    """Raised when an explicit synthetic structural configuration is invalid."""


class TimesNetInputError(ValueError):
    """Raised when an input tensor violates the structural contract."""


class TimesNetProvenanceError(ValueError):
    """Raised when synthetic-only mode receives forbidden provenance."""


@dataclass(frozen=True)
class TimesNetStructuralConfig:
    """All values are explicit; none is a hidden target-paper default."""

    seq_len: int
    pred_len: int
    n_vars: int
    top_k: int
    d_model: int
    d_ff: int
    e_layers: int
    num_kernels: int
    dropout: float
    strict_paper_mode: bool
    synthetic_only_mode: bool
    branch_variant: str

    def validate(self) -> None:
        fields = {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "n_vars": self.n_vars,
            "top_k": self.top_k,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "num_kernels": self.num_kernels,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in fields.values()):
            raise TimesNetConfigError("all integer configuration fields must be positive integers")
        if self.strict_paper_mode and self.seq_len != 10:
            raise TimesNetConfigError("strict-paper mode requires seq_len=10")
        if self.top_k > self.seq_len // 2:
            raise TimesNetConfigError("top_k exceeds the available positive FFT frequencies")
        if not isinstance(self.dropout, (float, int)) or not 0.0 <= float(self.dropout) < 1.0:
            raise TimesNetConfigError("dropout must be within [0, 1)")
        if not isinstance(self.strict_paper_mode, bool) or not isinstance(self.synthetic_only_mode, bool):
            raise TimesNetConfigError("strict_paper_mode and synthetic_only_mode must be bool")
        if self.branch_variant != "STANDARD_REFERENCE_CORE":
            raise TimesNetConfigError("only STANDARD_REFERENCE_CORE is permitted in this synthetic-only module")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_provenance(provenance: str, synthetic_only_mode: bool) -> str:
    if not isinstance(provenance, str) or not provenance.strip():
        raise TimesNetProvenanceError("an explicit non-empty provenance string is required")
    normalized = provenance.strip().upper()
    tokens = set(token for token in re.split(r"[^A-Z0-9]+", normalized) if token)
    forbidden_tokens = {"NASA", "CALCE", "VALIDATION", "TEST", "TRAIN", "REAL"}
    if synthetic_only_mode and (tokens & forbidden_tokens or "SYNTHETIC" not in normalized):
        raise TimesNetProvenanceError("synthetic-only mode rejects real, partition, and TEST provenance")
    return normalized


def _validate_input(x: Tensor, config: TimesNetStructuralConfig) -> None:
    if not isinstance(x, Tensor) or x.ndim != 3:
        raise TimesNetInputError("input must be [batch, seq_len, n_vars]")
    if tuple(x.shape[1:]) != (config.seq_len, config.n_vars):
        raise TimesNetInputError("input must match configured seq_len and n_vars")
    if not torch.isfinite(x).all():
        raise TimesNetInputError("input contains NaN or Inf")


@dataclass(frozen=True)
class PeriodDetection:
    frequency_indices: tuple[int, ...]
    periods: tuple[int, ...]
    batch_amplitudes: Tensor


def detect_periods(x: Tensor, top_k: int) -> PeriodDetection:
    """Standard reference FFT-amplitude period discovery on a synthetic tensor."""
    if x.ndim != 3:
        raise TimesNetInputError("FFT period detection requires [batch, time, channels]")
    length = int(x.shape[1])
    spectrum = torch.fft.rfft(x, dim=1)
    amplitudes = spectrum.abs()
    average_amplitude = amplitudes.mean(dim=0).mean(dim=-1)
    positive = average_amplitude.clone()
    positive[0] = 0.0
    if top_k < 1 or top_k > length // 2:
        raise TimesNetConfigError("top_k is invalid for this sequence length")
    _, index = torch.topk(positive, top_k)
    frequency_indices = tuple(int(value) for value in index.detach().cpu().tolist())
    periods = tuple(int(math.ceil(length / frequency)) for frequency in frequency_indices)
    batch_amplitudes = amplitudes.mean(dim=-1)[:, index]
    return PeriodDetection(frequency_indices, periods, batch_amplitudes)


def reshape_periodic_2d(x: Tensor, period: int) -> tuple[Tensor, int]:
    """Right-pad time with zeros, reshape to [B,C,groups,period], and report padding."""
    if not isinstance(period, int) or period < 1:
        raise TimesNetInputError("period must be a positive integer")
    batch, length, channels = x.shape
    target_length = int(math.ceil(length / period) * period)
    padding = target_length - length
    if padding:
        x = torch.cat((x, torch.zeros(batch, padding, channels, device=x.device, dtype=x.dtype)), dim=1)
    reshaped = x.permute(0, 2, 1).reshape(batch, channels, target_length // period, period)
    return reshaped, padding


def restore_periodic_2d(x: Tensor, *, original_length: int) -> Tensor:
    """Restore [B,C,groups,period] to [B,T,C] and crop deterministic right padding."""
    if x.ndim != 4:
        raise TimesNetInputError("2D periodic tensor must have rank 4")
    batch, channels, groups, period = x.shape
    restored = x.reshape(batch, channels, groups * period).permute(0, 2, 1)
    return restored[:, :original_length, :]


class ParameterSharedInception(nn.Module):
    """Reference Inception-style 2D block; all kernel counts are explicit fixture settings."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * index + 1, padding=index)
            for index in range(num_kernels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(dim=-1)


class TimesBlockStructural(nn.Module):
    """One generic standard-core multi-period block, not a target-paper branch variant."""

    def __init__(self, config: TimesNetStructuralConfig) -> None:
        super().__init__()
        self.config = config
        self.inception = nn.Sequential(
            ParameterSharedInception(config.d_model, config.d_ff, config.num_kernels),
            nn.GELU(),
            nn.Dropout2d(float(config.dropout)),
            ParameterSharedInception(config.d_ff, config.d_model, config.num_kernels),
        )

    def forward_with_trace(self, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
        detection = detect_periods(x, self.config.top_k)
        representations: list[Tensor] = []
        padding_by_period: dict[str, int] = {}
        for period in detection.periods:
            tensor_2d, padding = reshape_periodic_2d(x, period)
            processed = self.inception(tensor_2d)
            representations.append(restore_periodic_2d(processed, original_length=x.shape[1]))
            padding_by_period[str(period)] = padding
        stacked = torch.stack(representations, dim=-1)
        weights = torch.softmax(detection.batch_amplitudes, dim=1).unsqueeze(1).unsqueeze(1)
        merged = (stacked * weights).sum(dim=-1)
        return x + merged, {
            "frequency_indices": list(detection.frequency_indices),
            "detected_periods": list(detection.periods),
            "padding_by_period": padding_by_period,
            "adaptive_merge_weight_shape": list(weights.shape),
            "adaptive_merge_reference_only": True,
        }


class ParameterizedStandardTimesNet(nn.Module):
    """Synthetic-only standard TimesNet core with explicit projection and output contract."""

    standard_timesnet_core_implemented = True
    target_paper_branch_variant_implemented = False

    def __init__(self, config: TimesNetStructuralConfig | None) -> None:
        super().__init__()
        if config is None:
            raise TimesNetConfigError("an explicit TimesNetStructuralConfig is required")
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.n_vars, config.d_model)
        self.blocks = nn.ModuleList(TimesBlockStructural(config) for _ in range(config.e_layers))
        self.dropout = nn.Dropout(float(config.dropout))
        self.output_projection = nn.Linear(config.d_model, config.n_vars)

    def forward_with_trace(self, x: Tensor, *, provenance: str) -> tuple[Tensor, dict[str, Any]]:
        _validate_input(x, self.config)
        _validate_provenance(provenance, self.config.synthetic_only_mode)
        embedded = self.input_projection(x)
        block_traces: list[dict[str, Any]] = []
        encoded = embedded
        for block in self.blocks:
            encoded, trace = block.forward_with_trace(encoded)
            block_traces.append(trace)
        encoded = self.dropout(encoded)
        output = self.output_projection(encoded[:, -self.config.pred_len :, :])
        if not torch.isfinite(output).all():
            raise TimesNetInputError("output contains NaN or Inf")
        return output, {
            "input": list(x.shape),
            "embedded": list(embedded.shape),
            "encoded": list(encoded.shape),
            "output": list(output.shape),
            "blocks": block_traces,
        }

    def forward(self, x: Tensor, *, provenance: str) -> Tensor:
        output, _ = self.forward_with_trace(x, provenance=provenance)
        return output


class SyntheticStandardTimesNetAdapter:
    """Connect the generic core to a synthetic G-MTC branch port without claiming a real variant."""

    def __init__(self, *, component_type: str, model: ParameterizedStandardTimesNet) -> None:
        self.component_type = component_type.upper()
        if self.component_type not in {"TREND", "SEASONAL"}:
            raise GMTCScaffoldError("TimesNet adapter supports only TREND or SEASONAL")
        self.model = model

    def predict(self, component: ComponentInput) -> BranchPrediction:
        component.validate()
        if component.component_type.upper() != self.component_type:
            raise GMTCScaffoldError("TimesNet adapter received a mismatched component")
        output, _ = self.model.forward_with_trace(component.input_tensor, provenance=component.sample_provenance)
        return BranchPrediction(
            branch_name=f"SYNTHETIC_STANDARD_TIMESNET_{self.component_type}_ADAPTER",
            component_type=self.component_type,
            output_prediction=output,
            seq_len=component.seq_len,
            pred_len=int(output.shape[1]),
            batch_ids=component.batch_ids,
            time_alignment=component.time_alignment,
            forecast_horizon=component.forecast_horizon,
            sample_provenance=component.sample_provenance,
            partition_provenance=component.partition_provenance,
            model_fidelity_status="STANDARD_REFERENCE_CORE__SYNTHETIC_ONLY__NOT_TARGET_VARIANT",
            synthetic_only=True,
        )
