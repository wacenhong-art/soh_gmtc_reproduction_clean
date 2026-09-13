"""Synthetic-only structural contract for the target paper's fusion topology.

This module consumes a *preconstructed* ``[B, T, 9]`` feature tensor.  It
does not define or derive the paper's real component errors or ratios, and it
rejects all battery and partition provenance.  Consequently it is only a
structural test of the paper-supported attention/Bi-LSTM/weight-head topology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import torch
from torch import Tensor, nn


class AdaptiveFusionConfigError(ValueError):
    """Raised when an explicit synthetic structural configuration is invalid."""


class AdaptiveFusionInputError(ValueError):
    """Raised when a preconstructed feature tensor violates the contract."""


class AdaptiveFusionProvenanceError(ValueError):
    """Raised when synthetic-only mode receives real or partition provenance."""


@dataclass(frozen=True)
class AdaptiveFusionStructuralConfig:
    """Explicit structural configuration; no target-paper missing value is hidden."""

    input_dim: int
    hidden_size: int
    num_layers: int
    paper_dropout: float
    leaky_relu_slope: float
    strict_input_dim: bool
    synthetic_only_mode: bool
    component_attention_axis: str
    weight_network_placement: str
    final_activation_placement: str

    def validate(self) -> None:
        if self.input_dim != 9:
            raise AdaptiveFusionConfigError("the synthetic structural contract requires explicit input_dim=9")
        if self.hidden_size != 16 or self.num_layers != 1:
            raise AdaptiveFusionConfigError("the encoded Table 4 structural configuration requires hidden_size=16 and num_layers=1")
        if not isinstance(self.paper_dropout, (float, int)) or not 0.0 <= float(self.paper_dropout) < 1.0:
            raise AdaptiveFusionConfigError("paper_dropout must be within [0, 1)")
        if not isinstance(self.leaky_relu_slope, (float, int)) or float(self.leaky_relu_slope) < 0.0:
            raise AdaptiveFusionConfigError("leaky_relu_slope must be non-negative")
        if not self.strict_input_dim or not self.synthetic_only_mode:
            raise AdaptiveFusionConfigError("this module is restricted to strict synthetic-only structural mode")
        if self.component_attention_axis != "FEATURE_AXIS_ENGINEERING_FIXTURE_ONLY":
            raise AdaptiveFusionConfigError("component-attention axis must remain explicitly fixture-only")
        if self.weight_network_placement != "LINEAR_BN_LEAKYRELU_DROPOUT__ENGINEERING_FIXTURE_ONLY":
            raise AdaptiveFusionConfigError("weight-network placement must remain explicitly fixture-only")
        if self.final_activation_placement != "HIDDEN_RELU_ONLY_ENGINEERING_FIXTURE_ONLY":
            raise AdaptiveFusionConfigError("final activation placement must remain explicitly fixture-only")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_synthetic_provenance(provenance: str) -> str:
    if not isinstance(provenance, str) or not provenance.strip():
        raise AdaptiveFusionProvenanceError("explicit provenance is required")
    normalized = provenance.strip().upper()
    tokens = {token for token in re.split(r"[^A-Z0-9]+", normalized) if token}
    forbidden = {"NASA", "CALCE", "REAL", "TRAIN", "VALIDATION", "TEST"}
    if "SYNTHETIC" not in normalized or tokens & forbidden:
        raise AdaptiveFusionProvenanceError("synthetic-only fusion rejects real, partition, and TEST provenance")
    return normalized


def _validate_features(x: Tensor, config: AdaptiveFusionStructuralConfig) -> None:
    if not isinstance(x, Tensor) or x.ndim != 3:
        raise AdaptiveFusionInputError("features must be a rank-3 [batch, time, 9] tensor")
    if x.shape[-1] != config.input_dim:
        raise AdaptiveFusionInputError("features do not satisfy the explicit 9-feature contract")
    if x.shape[0] < 2:
        raise AdaptiveFusionInputError("batch size must be at least 2 for the structural BatchNorm contract")
    if not torch.isfinite(x).all():
        raise AdaptiveFusionInputError("features contain NaN or Inf")


class AdaptiveFusionStructural(nn.Module):
    """Paper-supported topology with deliberately explicit fixture-only choices."""

    real_fusion_feature_builder_implemented = False
    target_paper_exact_fusion_implementation = False

    def __init__(self, config: AdaptiveFusionStructuralConfig | None) -> None:
        super().__init__()
        if config is None:
            raise AdaptiveFusionConfigError("an explicit AdaptiveFusionStructuralConfig is required")
        config.validate()
        self.config = config

        # Eq. (25) does not define this axis.  The feature-axis softmax is an
        # explicit synthetic-fixture decision, never a paper-exact claim.
        self.component_attention_logits = nn.Parameter(torch.zeros(config.input_dim))

        # Table 4: input 9; hidden 16 each direction; one layer; batch_first.
        # PyTorch suppresses internal LSTM dropout for one layer, so external
        # dropout below is a documented fixture placement rather than a silent
        # substitution for the reported dropout value.
        self.bilstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.post_lstm_dropout = nn.Dropout(float(config.paper_dropout))

        # Eq. (26) specifies tanh + softmax but leaves h_t -> a_t unspecified.
        self.temporal_projection = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.temporal_vector = nn.Linear(config.hidden_size * 2, 1, bias=False)

        # Table 4: 32 -> 128 -> 64 -> 32 -> 3; BN/LeakyReLU/dropout placement
        # is explicitly an engineering fixture choice.
        self.weight_hidden_layers = nn.ModuleList(
            [
                nn.Linear(32, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
            ]
        )
        self.weight_batch_norms = nn.ModuleList([nn.BatchNorm1d(128), nn.BatchNorm1d(64), nn.BatchNorm1d(32)])
        self.weight_activation = nn.LeakyReLU(float(config.leaky_relu_slope))
        self.weight_dropout = nn.Dropout(float(config.paper_dropout))
        self.weight_output = nn.Linear(32, 3)

        # Table 4 final head 3 -> 16 -> 1.  The table does not localize ReLU;
        # this explicitly chooses hidden-only activation for structural wiring.
        self.final_hidden = nn.Linear(3, 16)
        self.final_activation = nn.ReLU()
        self.final_output = nn.Linear(16, 1)

    def forward_with_trace(self, features: Tensor, *, provenance: str) -> tuple[Tensor, dict[str, Any]]:
        _validate_features(features, self.config)
        _validate_synthetic_provenance(provenance)

        component_attention = torch.softmax(self.component_attention_logits, dim=0)
        attended = features * component_attention.view(1, 1, -1)
        lstm_output, _ = self.bilstm(attended)
        lstm_output = self.post_lstm_dropout(lstm_output)

        attention_hidden = torch.tanh(self.temporal_projection(lstm_output))
        temporal_scores = self.temporal_vector(attention_hidden).squeeze(-1)
        temporal_weights = torch.softmax(temporal_scores, dim=1)
        temporal_context = torch.sum(lstm_output * temporal_weights.unsqueeze(-1), dim=1)

        hidden = temporal_context
        for layer, batch_norm in zip(self.weight_hidden_layers, self.weight_batch_norms):
            hidden = self.weight_dropout(self.weight_activation(batch_norm(layer(hidden))))
        fusion_weights = torch.softmax(self.weight_output(hidden), dim=-1)

        # First 3d is ambiguous in Table 4.  These are explicitly the first
        # three scalar slots of a *synthetic preconstructed* fixture only.
        synthetic_component_predictions = features[:, -1, :3]
        weighted_components = fusion_weights * synthetic_component_predictions
        prediction = self.final_output(self.final_activation(self.final_hidden(weighted_components)))

        trace = {
            "input_shape": list(features.shape),
            "component_attention_shape": list(component_attention.shape),
            "attended_shape": list(attended.shape),
            "bilstm_output_shape": list(lstm_output.shape),
            "temporal_attention_shape": list(temporal_weights.shape),
            "temporal_context_shape": list(temporal_context.shape),
            "fusion_weight_shape": list(fusion_weights.shape),
            "weighted_component_shape": list(weighted_components.shape),
            "output_shape": list(prediction.shape),
            "component_attention_sum": float(component_attention.detach().sum()),
            "temporal_attention_sums": temporal_weights.detach().sum(dim=1).cpu().tolist(),
            "fusion_weight_sums": fusion_weights.detach().sum(dim=1).cpu().tolist(),
            "fusion_weights_nonnegative": bool((fusion_weights.detach() >= 0).all()),
            "synthetic_feature_selection_only": True,
        }
        return prediction, trace

    def forward(self, features: Tensor, *, provenance: str) -> Tensor:
        output, _ = self.forward_with_trace(features, provenance=provenance)
        return output
