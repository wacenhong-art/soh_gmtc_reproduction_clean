"""Frozen, result-blind execution configuration for the Tier 2 Phase 9A dry run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier2Phase9ARunConfig:
    seed: int = 42
    dry_run_epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "ADAM"
    device: str = "CPU"
    deterministic_algorithms: bool = True
    train_batch_shuffle: bool = False
    validation_batch_shuffle: bool = False
    drop_last: bool = False
    early_stopping: bool = False
    checkpoint_policy: str = "DISABLED_FOR_DRY_RUN"

    def validate(self) -> None:
        if (self.seed, self.dry_run_epochs, self.batch_size) != (42, 3, 32):
            raise ValueError("Tier 2 dry-run seed, epochs, or batch size differs from PLAN-CHANGE-0017_REVISED")
        if self.learning_rate != 0.001 or self.optimizer != "ADAM" or self.device != "CPU":
            raise ValueError("Tier 2 optimizer configuration differs from PLAN-CHANGE-0017_REVISED")
        if not self.deterministic_algorithms or self.train_batch_shuffle or self.validation_batch_shuffle or self.drop_last or self.early_stopping:
            raise ValueError("Tier 2 batch/determinism policy differs from the frozen no-shuffle dry-run contract")


FROZEN_TIER2_PHASE9A_CONFIG = Tier2Phase9ARunConfig()
