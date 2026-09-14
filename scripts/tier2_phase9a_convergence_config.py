"""Explicit, human-authorized execution control for the Phase 9A diagnostic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier2Phase9AConvergenceConfig:
    seed: int = 42
    max_epochs: int = 150
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.0
    early_stopping_metric: str = "FUSED_VALIDATION_CAPACITY_MSE"
    restore_best_checkpoint: bool = True
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "ADAM"
    device: str = "CPU"
    train_batch_shuffle: bool = False
    validation_batch_shuffle: bool = False
    drop_last: bool = False

    def validate(self) -> None:
        if (self.seed, self.max_epochs, self.patience, self.min_delta) != (42, 150, 50, 0.0):
            raise ValueError("Phase 9A convergence control differs from the human-authorized fixed diagnostic")
        if not self.early_stopping or not self.restore_best_checkpoint:
            raise ValueError("Phase 9A convergence diagnostic requires early stopping and best-checkpoint restoration")
        if self.early_stopping_metric != "FUSED_VALIDATION_CAPACITY_MSE":
            raise ValueError("Phase 9A convergence metric must be fused validation capacity MSE")
        if (self.learning_rate, self.batch_size, self.optimizer, self.device) != (0.001, 32, "ADAM", "CPU"):
            raise ValueError("Phase 9A convergence diagnostic changes a frozen optimizer setting")
        if self.train_batch_shuffle or self.validation_batch_shuffle or self.drop_last:
            raise ValueError("Phase 9A convergence diagnostic changes the frozen batch-order policy")


FROZEN_TIER2_PHASE9A_CONVERGENCE_CONFIG = Tier2Phase9AConvergenceConfig()
