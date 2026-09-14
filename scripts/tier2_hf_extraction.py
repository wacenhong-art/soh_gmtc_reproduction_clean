"""Frozen Tier 2 NASA health-feature endpoint extraction.

This module implements PLAN-CHANGE-0017_REVISED only.  It preserves the
paper equations rather than their contradictory prose labels:

    HF1 = t_Vmax - t_0
    HF2 = t_Imin - t_Vmax
    HF3 = t_4p0 - t_3p8

The execution protocol is a disclosed Tier 2 fallback, not a paper-exact
claim.  It intentionally contains no capacity, SOH, split, model, or metric
logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


NASA_CANONICAL_VOLTAGE_FIELD = "Voltage_measured"
T_IMIN_FIELD = "Current_charge"
T_IMIN_SIGN_CONVENTION = "RAW_CHARGER_CURRENT_SIGN__NO_ABS_OR_SIGN_INVERSION"
T_IMIN_THRESHOLD_A = 0.02
V_MAX = 4.2
V_START = 3.8
V_END = 4.0
VOLTAGE_ENDPOINT_RULE = "FIRST_SAMPLED_VALUE_AT_OR_ABOVE_THRESHOLD"
T_IMIN_ENDPOINT_RULE = "FIRST_RAW_CURRENT_CHARGE_LEQ_0P02_STRICTLY_AFTER_T_VMAX__ELSE_FINAL_VALID_TIMESTAMP"


class HFExtractionError(ValueError):
    """Raised when a charge operation cannot support a frozen endpoint."""


@dataclass(frozen=True)
class NASAHFExtraction:
    hf1_seconds: float
    hf2_seconds: float
    hf3_seconds: float
    t0_seconds: float
    t_vmax_seconds: float
    t_imin_seconds: float
    t_3p8_seconds: float
    t_4p0_seconds: float
    t_imin_provenance: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise HFExtractionError(f"{name} is empty or non-finite")
    return values


def _first_at_or_above(values: np.ndarray, threshold: float, *, name: str) -> int:
    indices = np.flatnonzero(values >= threshold)
    if indices.size == 0:
        raise HFExtractionError(f"{name} never reaches {threshold}")
    return int(indices[0])


def extract_nasa_hf_from_arrays(
    time: Any,
    voltage_measured: Any,
    current_charge: Any,
) -> NASAHFExtraction:
    """Extract the three frozen NASA Tier 2 HFs from one charge operation.

    The final-timestamp fallback is explicitly authorized for a charge that
    never contains a sampled charger-current point <= 0.02 A after ``t_Vmax``.
    It is recorded in ``t_imin_provenance`` rather than hidden or imputed.
    """

    t = _finite_vector(time, name="Time")
    voltage = _finite_vector(voltage_measured, name=NASA_CANONICAL_VOLTAGE_FIELD)
    current = _finite_vector(current_charge, name=T_IMIN_FIELD)
    if len({t.size, voltage.size, current.size}) != 1:
        raise HFExtractionError("Time, Voltage_measured, and Current_charge lengths differ")
    if t.size < 2:
        raise HFExtractionError("charge operation has fewer than two samples")
    if not np.all(np.diff(t) > 0.0):
        raise HFExtractionError("Time is not strictly increasing")

    i_vmax = _first_at_or_above(voltage, V_MAX, name=NASA_CANONICAL_VOLTAGE_FIELD)
    i_3p8 = _first_at_or_above(voltage, V_START, name=NASA_CANONICAL_VOLTAGE_FIELD)
    i_4p0 = _first_at_or_above(voltage, V_END, name=NASA_CANONICAL_VOLTAGE_FIELD)
    if i_4p0 < i_3p8:
        raise HFExtractionError("4.0 V endpoint precedes 3.8 V endpoint")

    cutoff = np.flatnonzero(current[i_vmax + 1 :] <= T_IMIN_THRESHOLD_A)
    if cutoff.size:
        i_imin = i_vmax + 1 + int(cutoff[0])
        imin_provenance = "SAMPLED_CURRENT_CHARGE_LEQ_0P02_A"
    else:
        i_imin = t.size - 1
        imin_provenance = "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK"

    result = NASAHFExtraction(
        hf1_seconds=float(t[i_vmax] - t[0]),
        hf2_seconds=float(t[i_imin] - t[i_vmax]),
        hf3_seconds=float(t[i_4p0] - t[i_3p8]),
        t0_seconds=float(t[0]),
        t_vmax_seconds=float(t[i_vmax]),
        t_imin_seconds=float(t[i_imin]),
        t_3p8_seconds=float(t[i_3p8]),
        t_4p0_seconds=float(t[i_4p0]),
        t_imin_provenance=imin_provenance,
    )
    values = np.asarray((result.hf1_seconds, result.hf2_seconds, result.hf3_seconds))
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise HFExtractionError("HF extraction produced non-finite or negative duration")
    return result


def extract_nasa_hf(charge: Any) -> NASAHFExtraction:
    """Extract frozen HFs from a NASA charge record only."""

    data = charge.data
    return extract_nasa_hf_from_arrays(
        getattr(data, "Time"),
        getattr(data, NASA_CANONICAL_VOLTAGE_FIELD),
        getattr(data, T_IMIN_FIELD),
    )
