"""Post-split numerical-field guard for the authorized Phase 9A MAT boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TestLockViolation(PermissionError):
    """Raised before a TEST pair can dereference a numerical record."""


@dataclass
class PostSplitPairAccessor:
    """Allow TRAIN/VALIDATION records only after a metadata split has been frozen.

    Full raw MAT parsing is expressly authorized by PLAN-CHANGE-0017_REVISED.
    This object enforces the separate, post-split rule: a TEST pair may not
    dereference charge/discharge numerical fields for HF, scaler, STL, model,
    metric, or any downstream payload.
    """

    cycles: list[Any]
    train_end: int
    val_end: int
    numerical_record_accesses: list[tuple[int, str]] = field(default_factory=list)
    test_numerical_access_attempts: int = 0

    def records_for_pair(self, pair_index: int, pair: dict[str, Any]) -> tuple[Any, Any]:
        if pair_index >= self.val_end:
            self.test_numerical_access_attempts += 1
            raise TestLockViolation("TEST_LOCK: post-split TEST numerical field access is prohibited")
        if pair_index < 0:
            raise ValueError("pair_index must be non-negative")
        partition = "TRAIN" if pair_index < self.train_end else "VALIDATION"
        charge_index = int(pair["charge_record_index"])
        discharge_index = int(pair["discharge_record_index"])
        self.numerical_record_accesses.append((pair_index, partition))
        return self.cycles[charge_index], self.cycles[discharge_index]

    @property
    def post_split_test_numerical_materialization_count(self) -> int:
        # An attempted access is rejected before a record dereference.
        return 0
