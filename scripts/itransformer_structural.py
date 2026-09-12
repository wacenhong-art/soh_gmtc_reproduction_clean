"""Synthetic-only, explicit-config standard iTransformer structural core.

This module implements only the common Eq.(21)-(23) core: time-history
embedding, encoder attention across variate tokens, and projection to a declared
synthetic forecast horizon. It contains no graph mechanism and cannot accept
real-battery provenance while synthetic_only mode is enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import torch
from torch import Tensor, nn


class ITransformerConfigError(ValueError):
    """Raised when an explicit structural configuration is invalid or absent."""


class ITransformerInputError(ValueError):
    """Raised when a supplied tensor violates the explicit contract."""


class ITransformerProvenanceError(ValueError):
    """Raised when synthetic-only mode receives non-synthetic provenance."""


@dataclass(frozen=True)
class ITransformerStructuralConfig:
    """All architecture values are explicit fixture-only inputs."""

    seq_len: int
    pred_len: int
    n_vars: int
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int
    dropout: float
    activation: str
    strict_paper_mode: bool
    synthetic_only_mode: bool

    def validate(self) -> None:
        integer_fields = {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "n_vars": self.n_vars,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_ff": self.d_ff,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_fields.values()):
            raise ITransformerConfigError("all integer configuration fields must be positive integers")
        if self.strict_paper_mode and self.seq_len != 10:
            raise ITransformerConfigError("strict-paper mode requires seq_len=10")
        if self.d_model % self.n_heads != 0:
            raise ITransformerConfigError("d_model must be divisible by n_heads")
        if not isinstance(self.dropout, (float, int)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ITransformerConfigError("dropout must be within [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ITransformerConfigError("activation must be explicitly relu or gelu")
        if not isinstance(self.strict_paper_mode, bool) or not isinstance(self.synthetic_only_mode, bool):
            raise ITransformerConfigError("strict_paper_mode and synthetic_only_mode must be bool")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_provenance(provenance: str, synthetic_only_mode: bool) -> str:
    if not isinstance(provenance, str) or not provenance.strip():
        raise ITransformerProvenanceError("an explicit non-empty provenance string is required")
    normalized = provenance.strip().upper()
    provenance_tokens = set(token for token in re.split(r"[^A-Z0-9]+", normalized) if token)
    forbidden_tokens = {"NASA", "CALCE", "VALIDATION", "TEST"}
    forbidden_phrases = {"TRAIN_REAL", "REAL_RESIDUAL", "REAL_CAPACITY", "REAL_HF"}
    if synthetic_only_mode and (
        bool(provenance_tokens & forbidden_tokens)
        or any(phrase in normalized for phrase in forbidden_phrases)
    ):
        raise ITransformerProvenanceError(
            "synthetic_only_mode rejects real-battery, validation, and TEST provenance"
        )
    if synthetic_only_mode and "SYNTHETIC" not in normalized:
        raise ITransformerProvenanceError("synthetic_only_mode requires SYNTHETIC provenance")
    return normalized


class ParameterizedITransformerStructural(nn.Module):
    """Standard inverted-token Transformer core with explicit input contract."""

    standard_itransformer_core = True
    custom_graph_module_implemented = False

    def __init__(self, config: ITransformerStructuralConfig | None) -> None:
        super().__init__()
        if config is None:
            raise ITransformerConfigError("an explicit ITransformerStructuralConfig is required")
        config.validate()
        self.config = config
        self.embedding = nn.Linear(config.seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=float(config.dropout),
            activation=config.activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        self.projection = nn.Linear(config.d_model, config.pred_len)

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor) or x.ndim != 3:
            raise ITransformerInputError("input must be a rank-3 tensor shaped [batch, seq_len, n_vars]")
        if x.shape[1] != self.config.seq_len or x.shape[2] != self.config.n_vars:
            raise ITransformerInputError(
                "input shape must exactly match configured [batch, seq_len, n_vars]"
            )
        if not torch.isfinite(x).all():
            raise ITransformerInputError("input contains NaN or Inf")

    def forward_with_trace(self, x: Tensor, *, provenance: str) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        """Apply inversion, attention over variables, and projection with traceable shapes."""
        self._validate_input(x)
        _validate_provenance(provenance, self.config.synthetic_only_mode)
        inverted = x.transpose(1, 2)  # [B, N, L]: variables are attention tokens.
        embedded = self.embedding(inverted)  # [B, N, D]
        encoded = self.encoder(embedded)  # [B, N, D]
        projected = self.projection(encoded)  # [B, N, P]
        output = projected.transpose(1, 2)  # [B, P, N]
        return output, {
            "input": tuple(x.shape),
            "inverted_tokens": tuple(inverted.shape),
            "embedded": tuple(embedded.shape),
            "attention_output": tuple(encoded.shape),
            "output": tuple(output.shape),
        }

    def forward(self, x: Tensor, *, provenance: str) -> Tensor:
        output, _ = self.forward_with_trace(x, provenance=provenance)
        return output
