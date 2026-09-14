"""Frozen NASA Tier 2 capacity-to-SOH bridge (PA001)."""

from __future__ import annotations

from typing import Any

import numpy as np


NASA_RATED_CAPACITY_AH = 2.0
SOH_FORMULA = "SOH=CAPACITY_AH/2.0"


class SOHBridgeError(ValueError):
    pass


def capacity_to_soh(capacity_ah: Any) -> np.ndarray:
    values = np.asarray(capacity_ah, dtype=np.float64)
    if not np.isfinite(values).all():
        raise SOHBridgeError("capacity input is non-finite")
    return values / NASA_RATED_CAPACITY_AH
