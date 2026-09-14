"""Frozen Tier 2 scalar component adapter for synthetic structural verification.

The adapter is intentionally separate from the TimesNet and iTransformer cores.
It maps their learned multivariate intermediate forecast [B, 1, 3] to one
scalar component prediction [B, 1, 1] without imposing an activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


class ScalarComponentReadoutError(ValueError):
    """Raised when the frozen scalar-readout contract is violated."""


@dataclass(frozen=True)
class ScalarComponentReadoutConfig:
    input_dim: int
    output_dim: int
    pred_len: int
    bias: bool
    activation: str

    def validate(self) -> None:
        if self.input_dim != 3:
            raise ScalarComponentReadoutError("Tier 2 scalar readout requires input_dim=3")
        if self.output_dim != 1:
            raise ScalarComponentReadoutError("Tier 2 scalar readout requires output_dim=1")
        if self.pred_len != 1:
            raise ScalarComponentReadoutError("Tier 2 scalar readout requires pred_len=1")
        if self.bias is not True:
            raise ScalarComponentReadoutError("Tier 2 scalar readout requires bias=True")
        if self.activation != "NONE":
            raise ScalarComponentReadoutError("Tier 2 scalar readout permits no activation")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinearScalarComponentReadout(nn.Module):
    """Exactly one trainable 3-to-1 linear adapter, with no post-activation."""

    def __init__(self, config: ScalarComponentReadoutConfig | None) -> None:
        super().__init__()
        if config is None:
            raise ScalarComponentReadoutError("an explicit ScalarComponentReadoutConfig is required")
        config.validate()
        self.config = config
        self.linear = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)

    def forward(self, core_output: Tensor) -> Tensor:
        if not isinstance(core_output, Tensor) or core_output.ndim != 3:
            raise ScalarComponentReadoutError("core output must be rank-3 [B,1,3]")
        if tuple(core_output.shape[1:]) != (self.config.pred_len, self.config.input_dim):
            raise ScalarComponentReadoutError("core output must exactly match [B,1,3]")
        if not torch.isfinite(core_output).all():
            raise ScalarComponentReadoutError("core output contains NaN or Inf")
        scalar = self.linear(core_output)
        if tuple(scalar.shape[1:]) != (self.config.pred_len, self.config.output_dim):
            raise ScalarComponentReadoutError("scalar output must exactly match [B,1,1]")
        if not torch.isfinite(scalar).all():
            raise ScalarComponentReadoutError("scalar output contains NaN or Inf")
        return scalar


def strict_scalar_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Compute MSE only after exact scalar-component shape validation."""
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise ScalarComponentReadoutError("prediction and target must be tensors")
    if prediction.ndim != 3 or target.ndim != 3:
        raise ScalarComponentReadoutError("prediction and target must both be rank-3")
    if tuple(prediction.shape) != tuple(target.shape):
        raise ScalarComponentReadoutError("prediction and target shapes must match exactly; broadcasting is prohibited")
    if prediction.shape[1:] != (1, 1):
        raise ScalarComponentReadoutError("Tier 2 scalar prediction and target must be [B,1,1]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ScalarComponentReadoutError("prediction or target contains NaN or Inf")
    return torch.mean((prediction - target) ** 2)
