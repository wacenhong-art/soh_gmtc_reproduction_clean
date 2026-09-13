"""Ambiguity-preserving, synthetic-only parameterized SCF-GAN structural core.

The target paper contains unresolved internal contradictions.  This module
therefore exposes—not resolves—the relevant architecture options.  It accepts
only abstract synthetic tensors and fails closed for real-battery, validation,
test, real-augmentation, or synthetic-SOH requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


FEATURE_MATCH_LAMBDA = 0.1
NOISE_DISTRIBUTION = "STANDARD_NORMAL"


class SCFGANConfigError(ValueError):
    """Raised when a required, explicit structural configuration is invalid."""


class SCFGANInputError(ValueError):
    """Raised when a tensor violates the explicit synthetic-only contract."""


class SCFGANProvenanceError(ValueError):
    """Raised when synthetic-only code receives prohibited provenance/request."""


@dataclass(frozen=True)
class SCFGANStructuralConfig:
    """Explicit ambiguity-space configuration; no paper defaults are supplied."""

    noise_dim: int
    condition_dim: int
    feature_dim: int
    generator_hidden_dims: tuple[int, ...]
    generator_activation: Literal["relu", "leaky_relu"]
    generator_norm_mode: Literal["bn_structural", "conditional_affine_bn_structural"]
    generator_use_sn: bool
    discriminator_hidden_dims: tuple[int, ...]
    discriminator_use_sn: bool
    feature_match_layer_index: int
    leaky_relu_slope: float
    synthetic_only_mode: bool

    def validate(self) -> None:
        scalar_dims = {
            "noise_dim": self.noise_dim,
            "condition_dim": self.condition_dim,
            "feature_dim": self.feature_dim,
            "feature_match_layer_index": self.feature_match_layer_index,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in scalar_dims.values()):
            raise SCFGANConfigError("dimension fields and feature_match_layer_index must be integers")
        if any(self.__dict__[name] < 1 for name in ("noise_dim", "condition_dim", "feature_dim")):
            raise SCFGANConfigError("noise_dim, condition_dim, and feature_dim must be positive")
        for field_name, dims in {
            "generator_hidden_dims": self.generator_hidden_dims,
            "discriminator_hidden_dims": self.discriminator_hidden_dims,
        }.items():
            if not isinstance(dims, tuple) or not dims or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dims
            ):
                raise SCFGANConfigError(f"{field_name} must be a non-empty tuple of positive integers")
        if len(self.discriminator_hidden_dims) not in {3, 4}:
            raise SCFGANConfigError("synthetic ambiguity coverage supports exactly 3 or 4 discriminator hidden layers")
        if self.feature_match_layer_index != 2:
            raise SCFGANConfigError("paper-supported feature matching uses the explicit third hidden layer index 2")
        if self.feature_match_layer_index >= len(self.discriminator_hidden_dims):
            raise SCFGANConfigError("feature_match_layer_index must exist in discriminator hidden layers")
        if self.generator_activation not in {"relu", "leaky_relu"}:
            raise SCFGANConfigError("generator_activation must explicitly be relu or leaky_relu")
        if self.generator_norm_mode not in {"bn_structural", "conditional_affine_bn_structural"}:
            raise SCFGANConfigError("generator_norm_mode must explicitly be a supported structural mode")
        if not isinstance(self.generator_use_sn, bool) or not isinstance(self.discriminator_use_sn, bool):
            raise SCFGANConfigError("spectral-normalization flags must be bool")
        if not isinstance(self.synthetic_only_mode, bool):
            raise SCFGANConfigError("synthetic_only_mode must be bool")
        if not isinstance(self.leaky_relu_slope, (int, float)) or not 0.0 < float(self.leaky_relu_slope) <= 1.0:
            raise SCFGANConfigError("leaky_relu_slope must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_provenance(provenance: str, synthetic_only_mode: bool, request_kind: str) -> str:
    if not isinstance(provenance, str) or not provenance.strip():
        raise SCFGANProvenanceError("an explicit non-empty provenance string is required")
    if not isinstance(request_kind, str) or not request_kind.strip():
        raise SCFGANProvenanceError("an explicit request_kind is required")
    normalized = provenance.strip().upper()
    request = request_kind.strip().upper()
    tokens = {token for token in re.split(r"[^A-Z0-9]+", normalized) if token}
    forbidden_tokens = {"NASA", "CALCE", "VALIDATION", "TEST"}
    forbidden_phrases = {"TRAIN_REAL", "REAL_HF", "REAL_CAPACITY", "REAL_AUGMENTATION", "SYNTHETIC_SOH"}
    if synthetic_only_mode and (
        bool(tokens & forbidden_tokens)
        or any(phrase in normalized for phrase in forbidden_phrases)
        or request in {"REAL_AUGMENTATION", "SYNTHETIC_SOH_GENERATION", "REAL_TRAINING"}
    ):
        raise SCFGANProvenanceError("synthetic-only mode rejects real, validation, TEST, and synthetic-SOH requests")
    if synthetic_only_mode and "SYNTHETIC" not in normalized:
        raise SCFGANProvenanceError("synthetic-only mode requires SYNTHETIC provenance")
    return normalized


def _validate_rank2(name: str, value: Tensor, width: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise SCFGANInputError(f"{name} must be a finite rank-2 tensor with configured width {width}")
    if not torch.isfinite(value).all():
        raise SCFGANInputError(f"{name} contains NaN or Inf")


def _linear(in_features: int, out_features: int, use_sn: bool) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    return spectral_norm(layer) if use_sn else layer


class ConditionalAffineBatchNorm(nn.Module):
    """Synthetic engineering conditional-affine BN, not target-paper-exact CBN."""

    provenance = "SYNTHETIC_ENGINEERING_CBN_IMPLEMENTATION"

    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.base_bn = nn.BatchNorm1d(width, affine=False)
        self.condition_affine = nn.Linear(condition_dim, 2 * width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        normalized = self.base_bn(x)
        gamma, beta = self.condition_affine(condition).chunk(2, dim=1)
        return normalized * (1.0 + gamma) + beta


class StructuralGenerator(nn.Module):
    """Generator supporting paper-contradiction options through explicit config."""

    output_activation = "TANH"

    def __init__(self, config: SCFGANStructuralConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = (config.noise_dim + config.condition_dim, *config.generator_hidden_dims)
        self.hidden_layers = nn.ModuleList(
            _linear(dimensions[index], dimensions[index + 1], config.generator_use_sn)
            for index in range(len(dimensions) - 1)
        )
        if config.generator_norm_mode == "bn_structural":
            self.norm_layers: nn.ModuleList = nn.ModuleList(nn.BatchNorm1d(width) for width in config.generator_hidden_dims)
        else:
            self.norm_layers = nn.ModuleList(
                ConditionalAffineBatchNorm(width, config.condition_dim) for width in config.generator_hidden_dims
            )
        self.output_layer = _linear(config.generator_hidden_dims[-1], config.feature_dim, config.generator_use_sn)

    def _activate(self, x: Tensor) -> Tensor:
        if self.config.generator_activation == "relu":
            return F.relu(x)
        return F.leaky_relu(x, negative_slope=float(self.config.leaky_relu_slope))

    def forward(self, noise: Tensor, condition: Tensor) -> Tensor:
        x = torch.cat((noise, condition), dim=1)
        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            x = layer(x)
            x = norm(x, condition) if isinstance(norm, ConditionalAffineBatchNorm) else norm(x)
            x = self._activate(x)
        return torch.tanh(self.output_layer(x))


class StructuralDiscriminator(nn.Module):
    """Conditional discriminator with explicit 3- or 4-hidden ambiguity coverage."""

    output_activation = "SIGMOID"

    def __init__(self, config: SCFGANStructuralConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = (config.feature_dim + config.condition_dim, *config.discriminator_hidden_dims)
        self.hidden_layers = nn.ModuleList(
            _linear(dimensions[index], dimensions[index + 1], config.discriminator_use_sn)
            for index in range(len(dimensions) - 1)
        )
        self.output_layer = nn.Linear(config.discriminator_hidden_dims[-1], 1)

    def forward_with_features(self, features: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        x = torch.cat((features, condition), dim=1)
        hidden_outputs: list[Tensor] = []
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=float(self.config.leaky_relu_slope))
            hidden_outputs.append(x)
        score = torch.sigmoid(self.output_layer(x))
        return score, hidden_outputs[self.config.feature_match_layer_index], hidden_outputs


class ParameterizedSCFGANStructural(nn.Module):
    """Synthetic-only conditional GAN routing with explicit ambiguity configuration."""

    feature_match_lambda = FEATURE_MATCH_LAMBDA
    noise_distribution = NOISE_DISTRIBUTION

    def __init__(self, config: SCFGANStructuralConfig | None) -> None:
        super().__init__()
        if config is None:
            raise SCFGANConfigError("an explicit SCFGANStructuralConfig is required")
        config.validate()
        self.config = config
        self.generator = StructuralGenerator(config)
        self.discriminator = StructuralDiscriminator(config)

    def _validate_noise_condition(self, noise: Tensor, condition: Tensor) -> None:
        _validate_rank2("noise", noise, self.config.noise_dim)
        _validate_rank2("condition", condition, self.config.condition_dim)
        if noise.shape[0] != condition.shape[0]:
            raise SCFGANInputError("noise and condition batch sizes must match")

    def _validate_features_condition(self, features: Tensor, condition: Tensor) -> None:
        _validate_rank2("features", features, self.config.feature_dim)
        _validate_rank2("condition", condition, self.config.condition_dim)
        if features.shape[0] != condition.shape[0]:
            raise SCFGANInputError("features and condition batch sizes must match")

    def generate(self, noise: Tensor, condition: Tensor, *, provenance: str, request_kind: str = "STRUCTURAL_TEST") -> Tensor:
        _validate_provenance(provenance, self.config.synthetic_only_mode, request_kind)
        self._validate_noise_condition(noise, condition)
        return self.generator(noise, condition)

    def discriminate(self, features: Tensor, condition: Tensor, *, provenance: str, request_kind: str = "STRUCTURAL_TEST") -> tuple[Tensor, Tensor, list[Tensor]]:
        _validate_provenance(provenance, self.config.synthetic_only_mode, request_kind)
        self._validate_features_condition(features, condition)
        return self.discriminator.forward_with_features(features, condition)

    def losses(
        self,
        real_features: Tensor,
        noise: Tensor,
        condition: Tensor,
        *,
        provenance: str,
        request_kind: str = "STRUCTURAL_TEST",
    ) -> dict[str, Tensor]:
        _validate_provenance(provenance, self.config.synthetic_only_mode, request_kind)
        self._validate_features_condition(real_features, condition)
        self._validate_noise_condition(noise, condition)
        fake_features = self.generator(noise, condition)
        real_scores, real_match_features, _ = self.discriminator.forward_with_features(real_features, condition)
        fake_scores, fake_match_features, _ = self.discriminator.forward_with_features(fake_features, condition)
        ones = torch.ones_like(real_scores)
        zeros = torch.zeros_like(fake_scores)
        discriminator_loss = F.binary_cross_entropy(real_scores, ones) + F.binary_cross_entropy(fake_scores.detach(), zeros)
        generator_adversarial_loss = F.binary_cross_entropy(fake_scores, ones)
        feature_match_loss = F.mse_loss(fake_match_features, real_match_features.detach())
        generator_loss = generator_adversarial_loss + self.feature_match_lambda * feature_match_loss
        return {
            "fake_features": fake_features,
            "real_scores": real_scores,
            "fake_scores": fake_scores,
            "real_match_features": real_match_features,
            "fake_match_features": fake_match_features,
            "discriminator_loss": discriminator_loss,
            "generator_adversarial_loss": generator_adversarial_loss,
            "feature_match_loss": feature_match_loss,
            "generator_loss": generator_loss,
        }

    def spectral_norm_layer_names(self) -> dict[str, list[str]]:
        def names(module: nn.Module) -> list[str]:
            return [name for name, child in module.named_modules() if isinstance(child, nn.Linear) and hasattr(child, "weight_u")]

        return {"generator": names(self.generator), "discriminator": names(self.discriminator)}
