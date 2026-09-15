"""Explicit-period, partition-provenance STL structural module.

This module intentionally does *not* discover a period, read raw battery files,
concatenate partitions, train a model, or compute prediction metrics. It is
safe to exercise only on deterministic synthetic fixtures until a paper-
defensible real-battery period is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Iterable

import numpy as np
from statsmodels.tsa.seasonal import STL


class STLPeriodNotSpecified(ValueError):
    """Raised when a caller omits the mandatory, explicit STL period."""


class STLDecompositionInputError(ValueError):
    """Raised when a series or period cannot be safely decomposed."""


class STLPartitionProvenanceError(ValueError):
    """Raised when a caller declares combined, locked, or unknown partitions."""


ALLOWED_PARTITIONS = {
    "TRAIN",
    "VALIDATION",
    "SYNTHETIC_UNIT_TEST",
    # Explicitly isolated synthetic trajectories authorized for Phase 9D
    # augmentation smoke work.  This is not a real-data partition and may
    # never be concatenated with TRAIN, VALIDATION, or TEST trajectories.
    "TIER2_SYNTHETIC_TRAJECTORY",
}
FORBIDDEN_PARTITION_IDS = {
    "TEST",
    "FULL_LIFECYCLE",
    "ALL_PARTITIONS",
    "TRAIN+VALIDATION",
    "TRAIN+TEST",
    "VALIDATION+TEST",
}


@dataclass(frozen=True)
class STLContract:
    """All resolved statsmodels options, including library defaults."""

    period: int
    seasonal: int
    trend: int | None
    low_pass: int | None
    seasonal_deg: int
    trend_deg: int
    low_pass_deg: int
    robust: bool
    seasonal_jump: int
    trend_jump: int
    low_pass_jump: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "seasonal": self.seasonal,
            "trend": self.trend,
            "low_pass": self.low_pass,
            "seasonal_deg": self.seasonal_deg,
            "trend_deg": self.trend_deg,
            "low_pass_deg": self.low_pass_deg,
            "robust": self.robust,
            "seasonal_jump": self.seasonal_jump,
            "trend_jump": self.trend_jump,
            "low_pass_jump": self.low_pass_jump,
        }


def _as_finite_vector(series: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(series), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise STLDecompositionInputError("series must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all():
        raise STLDecompositionInputError("series contains NaN or Inf")
    return values


def _validate_period(period: int | None, input_length: int) -> int:
    if period is None:
        raise STLPeriodNotSpecified("STLPeriodNotSpecified: explicit period is required")
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)):
        raise STLDecompositionInputError("period must be an integer")
    normalized = int(period)
    if normalized < 2:
        raise STLDecompositionInputError("period must be an integer greater than or equal to 2")
    if input_length < 2 * normalized:
        raise STLDecompositionInputError(
            "series is too short for the explicit period; require at least two full periods"
        )
    return normalized


def _validate_partition(partition_id: str) -> str:
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise STLPartitionProvenanceError("partition_id must be an explicit non-empty string")
    normalized = partition_id.strip().upper()
    if normalized in FORBIDDEN_PARTITION_IDS or "+" in normalized:
        raise STLPartitionProvenanceError(
            "combined, full-lifecycle, and TEST partitions are prohibited by the STL leakage contract"
        )
    if normalized not in ALLOWED_PARTITIONS:
        raise STLPartitionProvenanceError(
            "partition_id must be TRAIN, VALIDATION, SYNTHETIC_UNIT_TEST, or TIER2_SYNTHETIC_TRAJECTORY"
        )
    return normalized


def _validate_series_id(series_id: str) -> str:
    if not isinstance(series_id, str) or not series_id.strip():
        raise STLDecompositionInputError("series_id must be an explicit non-empty string")
    return series_id.strip()


def resolved_default_contract(series: Iterable[float], period: int | None) -> STLContract:
    """Materialize statsmodels defaults for an explicit period without fitting."""

    values = _as_finite_vector(series)
    normalized_period = _validate_period(period, int(values.size))
    config = STL(values, period=normalized_period).config
    return STLContract(
        period=int(config["period"]),
        seasonal=int(config["seasonal"]),
        trend=None if config["trend"] is None else int(config["trend"]),
        low_pass=None if config["low_pass"] is None else int(config["low_pass"]),
        seasonal_deg=int(config["seasonal_deg"]),
        trend_deg=int(config["trend_deg"]),
        low_pass_deg=int(config["low_pass_deg"]),
        robust=bool(config["robust"]),
        seasonal_jump=int(config["seasonal_jump"]),
        trend_jump=int(config["trend_jump"]),
        low_pass_jump=int(config["low_pass_jump"]),
    )


class STLDecomposer:
    """Decompose one explicit, already-isolated series with no hidden defaults."""

    library = "statsmodels.tsa.seasonal.STL"

    def decompose(
        self,
        *,
        series: Iterable[float],
        period: int | None,
        partition_id: str,
        series_id: str,
    ) -> dict[str, Any]:
        """Return STL components and a fully auditable software contract."""

        values = _as_finite_vector(series)
        normalized_partition = _validate_partition(partition_id)
        normalized_series_id = _validate_series_id(series_id)
        normalized_period = _validate_period(period, int(values.size))
        contract = resolved_default_contract(values, normalized_period)
        result = STL(values, **contract.as_dict()).fit()
        reconstructed = result.trend + result.seasonal + result.resid
        error = np.abs(values - reconstructed)

        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "metadata": {
                "period": normalized_period,
                "input_length": int(values.size),
                "partition_id": normalized_partition,
                "series_id": normalized_series_id,
                "library": self.library,
                "library_version": version("statsmodels"),
                "parameter_config": contract.as_dict(),
                "execution_mode": "PARTITION_ISOLATED_OFFLINE_STL",
                "causality_status": "NON_CAUSAL_OFFLINE_DECOMPOSITION",
            },
            "structural_audit": {
                "length_preserved": bool(
                    len(result.trend) == len(values)
                    and len(result.seasonal) == len(values)
                    and len(result.resid) == len(values)
                ),
                "no_nan_or_inf": bool(
                    np.isfinite(result.trend).all()
                    and np.isfinite(result.seasonal).all()
                    and np.isfinite(result.resid).all()
                ),
                "max_abs_reconstruction_error": float(error.max()),
                "mean_abs_reconstruction_error": float(error.mean()),
            },
        }
