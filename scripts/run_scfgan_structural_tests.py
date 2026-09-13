"""Finite ambiguity-coverage tests for the synthetic-only SCF-GAN core.

This is a coverage test, not a configuration search: every conflicting option
is exercised at least once, and no loss, stability, or quality ranking is
computed or reported.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from scfgan_structural import (  # noqa: E402
    FEATURE_MATCH_LAMBDA,
    SCFGANConfigError,
    SCFGANInputError,
    SCFGANProvenanceError,
    SCFGANStructuralConfig,
    ParameterizedSCFGANStructural,
)


FIXTURE_SEED = 20260913
FIXTURE_BATCH = 5
SMOKE_STEPS = 3
SMOKE_CONFIG_NOTE = "SYNTHETIC_SMOKE_CONFIG_ONLY"
AMBIGUITY_SELECTION_PERFORMED = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def config_for(
    *, activation: str, norm_mode: str, generator_sn: bool, discriminator_depth: int, feature_dim: int
) -> SCFGANStructuralConfig:
    return SCFGANStructuralConfig(
        noise_dim=5,
        condition_dim=2,
        feature_dim=feature_dim,
        generator_hidden_dims=(7, 6),
        generator_activation=activation,
        generator_norm_mode=norm_mode,
        generator_use_sn=generator_sn,
        discriminator_hidden_dims=(8, 7, 6) if discriminator_depth == 3 else (8, 7, 6, 5),
        discriminator_use_sn=True,
        feature_match_layer_index=2,
        leaky_relu_slope=0.2,
        synthetic_only_mode=True,
    )


def synthetic_tensors(config: SCFGANStructuralConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn(FIXTURE_BATCH, config.noise_dim)
    condition = torch.linspace(-0.7, 0.7, FIXTURE_BATCH * config.condition_dim).view(FIXTURE_BATCH, config.condition_dim)
    real_features = torch.tanh(torch.randn(FIXTURE_BATCH, config.feature_dim))
    return noise, condition, real_features


def expect_failure(callable_object, exception_type) -> bool:
    try:
        callable_object()
    except exception_type:
        return True
    return False


def run_coverage_case(index: int, config: SCFGANStructuralConfig) -> dict[str, object]:
    set_seed(FIXTURE_SEED + index)
    model = ParameterizedSCFGANStructural(config)
    noise, condition, real_features = synthetic_tensors(config)
    fake_features = model.generate(noise, condition, provenance="SYNTHETIC_ABSTRACT_CONDITION_FIXTURE")
    real_score, real_match, _ = model.discriminate(real_features, condition, provenance="SYNTHETIC_ABSTRACT_CONDITION_FIXTURE")
    fake_score, fake_match, _ = model.discriminate(fake_features, condition, provenance="SYNTHETIC_ABSTRACT_CONDITION_FIXTURE")
    losses = model.losses(real_features, noise, condition, provenance="SYNTHETIC_ABSTRACT_CONDITION_FIXTURE")

    model.zero_grad(set_to_none=True)
    losses["discriminator_loss"].backward(retain_graph=True)
    discriminator_gradients = [parameter.grad for parameter in model.discriminator.parameters() if parameter.requires_grad]
    discriminator_backward = all(g is not None and torch.isfinite(g).all() for g in discriminator_gradients)
    model.zero_grad(set_to_none=True)
    losses["generator_loss"].backward()
    generator_gradients = [parameter.grad for parameter in model.generator.parameters() if parameter.requires_grad]
    generator_backward = all(g is not None and torch.isfinite(g).all() for g in generator_gradients)
    generator_nonzero = any(g is not None and torch.count_nonzero(g).item() > 0 for g in generator_gradients)
    sn_layers = model.spectral_norm_layer_names()
    return {
        "case_id": f"AMBIGUITY_COVERAGE_{index + 1}",
        "config": config.as_dict(),
        "feature_dimension_semantics": "GENERIC_FEATURE_4" if config.feature_dim == 4 else "GENERIC_FEATURE_VECTOR_3",
        "generator_forward": bool(torch.isfinite(fake_features).all() and fake_features.shape == (FIXTURE_BATCH, config.feature_dim)),
        "discriminator_forward": bool(
            torch.isfinite(real_score).all()
            and torch.isfinite(fake_score).all()
            and real_score.shape == (FIXTURE_BATCH, 1)
            and fake_score.shape == (FIXTURE_BATCH, 1)
        ),
        "tanh_bounds": bool(torch.all(fake_features <= 1.0 + 1e-6) and torch.all(fake_features >= -1.0 - 1e-6)),
        "sigmoid_bounds": bool(
            torch.all(real_score >= 0.0)
            and torch.all(real_score <= 1.0)
            and torch.all(fake_score >= 0.0)
            and torch.all(fake_score <= 1.0)
        ),
        "condition_concatenation": real_match.shape[0] == FIXTURE_BATCH and fake_match.shape[0] == FIXTURE_BATCH,
        "feature_match_extraction": real_match.shape == fake_match.shape,
        "finite_losses": all(torch.isfinite(losses[name]).item() for name in ("discriminator_loss", "generator_adversarial_loss", "feature_match_loss", "generator_loss")),
        "loss_scalar_shapes": all(losses[name].ndim == 0 for name in ("discriminator_loss", "generator_adversarial_loss", "feature_match_loss", "generator_loss")),
        "lambda_applied": model.feature_match_lambda == FEATURE_MATCH_LAMBDA,
        "generator_backward": generator_backward,
        "discriminator_backward": discriminator_backward,
        "feature_match_gradient_to_generator": generator_nonzero,
        "spectral_norm_layers": sn_layers,
        "generator_sn_support": bool(sn_layers["generator"]) if config.generator_use_sn else not bool(sn_layers["generator"]),
        "discriminator_sn_support": bool(sn_layers["discriminator"]),
    }


def condition_path_tests() -> dict[str, bool]:
    set_seed(FIXTURE_SEED + 99)
    config = config_for(
        activation="leaky_relu", norm_mode="conditional_affine_bn_structural", generator_sn=True, discriminator_depth=3, feature_dim=3
    )
    model = ParameterizedSCFGANStructural(config).eval()
    noise, condition, real_features = synthetic_tensors(config)
    alternate_condition = condition + 0.3
    with torch.no_grad():
        generator_first = model.generate(noise, condition, provenance="SYNTHETIC_CONDITION_PATH")
        generator_second = model.generate(noise, alternate_condition, provenance="SYNTHETIC_CONDITION_PATH")
        discriminator_first, _, _ = model.discriminate(real_features, condition, provenance="SYNTHETIC_CONDITION_PATH")
        discriminator_second, _, _ = model.discriminate(real_features, alternate_condition, provenance="SYNTHETIC_CONDITION_PATH")
    conditional_norm_modules = [module for module in model.generator.norm_layers if hasattr(module, "condition_affine")]
    return {
        "generator_condition_changes_output": not torch.equal(generator_first, generator_second),
        "discriminator_condition_changes_score": not torch.equal(discriminator_first, discriminator_second),
        "conditional_affine_modules_present": bool(conditional_norm_modules),
        "conditional_affine_parameters_present": all(any(parameter.requires_grad for parameter in module.parameters()) for module in conditional_norm_modules),
    }


def optimizer_smoke_test() -> dict[str, object]:
    set_seed(FIXTURE_SEED + 201)
    config = config_for(
        activation="relu", norm_mode="bn_structural", generator_sn=False, discriminator_depth=3, feature_dim=3
    )
    model = ParameterizedSCFGANStructural(config)
    noise, condition, real_features = synthetic_tensors(config)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=0.001)
    first_parameter = next(model.generator.parameters()).detach().clone()
    finite = True
    for _ in range(SMOKE_STEPS):
        losses = model.losses(real_features, noise, condition, provenance="SYNTHETIC_OPTIMIZER_SMOKE")
        discriminator_optimizer.zero_grad(set_to_none=True)
        losses["discriminator_loss"].backward(retain_graph=True)
        discriminator_optimizer.step()
        generator_optimizer.zero_grad(set_to_none=True)
        losses = model.losses(real_features, noise, condition, provenance="SYNTHETIC_OPTIMIZER_SMOKE")
        losses["generator_loss"].backward()
        generator_optimizer.step()
        finite = finite and all(torch.isfinite(losses[name]).item() for name in ("discriminator_loss", "generator_loss", "feature_match_loss"))
    changed = not torch.equal(first_parameter, next(model.generator.parameters()).detach())
    return {
        "optimizer": "Adam",
        "configuration_provenance": SMOKE_CONFIG_NOTE,
        "steps": SMOKE_STEPS,
        "losses_finite": finite,
        "parameters_changed": changed,
        "gan_convergence_evaluated": False,
        "generation_quality_evaluated": False,
        "pass": finite and changed,
    }


def determinism_test() -> bool:
    config = config_for(
        activation="relu", norm_mode="bn_structural", generator_sn=True, discriminator_depth=4, feature_dim=4
    )
    set_seed(FIXTURE_SEED + 400)
    first = ParameterizedSCFGANStructural(config).eval()
    noise, condition, _ = synthetic_tensors(config)
    first_output = first.generate(noise, condition, provenance="SYNTHETIC_DETERMINISM")
    set_seed(FIXTURE_SEED + 400)
    second = ParameterizedSCFGANStructural(config).eval()
    noise_again, condition_again, _ = synthetic_tensors(config)
    second_output = second.generate(noise_again, condition_again, provenance="SYNTHETIC_DETERMINISM")
    return bool(torch.equal(first_output, second_output))


def fail_closed_tests() -> dict[str, bool]:
    config = config_for(
        activation="relu", norm_mode="bn_structural", generator_sn=False, discriminator_depth=3, feature_dim=3
    )
    model = ParameterizedSCFGANStructural(config)
    noise, condition, features = synthetic_tensors(config)
    return {
        "missing_config": expect_failure(lambda: ParameterizedSCFGANStructural(None), SCFGANConfigError),
        "missing_noise_dim": expect_failure(
            lambda: SCFGANStructuralConfig(
                condition_dim=2, feature_dim=3, generator_hidden_dims=(7,), generator_activation="relu", generator_norm_mode="bn_structural", generator_use_sn=False, discriminator_hidden_dims=(7, 6, 5), discriminator_use_sn=True, feature_match_layer_index=2, leaky_relu_slope=0.2, synthetic_only_mode=True
            ), TypeError
        ),
        "missing_condition_dim": expect_failure(
            lambda: SCFGANStructuralConfig(
                noise_dim=5, feature_dim=3, generator_hidden_dims=(7,), generator_activation="relu", generator_norm_mode="bn_structural", generator_use_sn=False, discriminator_hidden_dims=(7, 6, 5), discriminator_use_sn=True, feature_match_layer_index=2, leaky_relu_slope=0.2, synthetic_only_mode=True
            ), TypeError
        ),
        "missing_feature_dim": expect_failure(
            lambda: SCFGANStructuralConfig(
                noise_dim=5, condition_dim=2, generator_hidden_dims=(7,), generator_activation="relu", generator_norm_mode="bn_structural", generator_use_sn=False, discriminator_hidden_dims=(7, 6, 5), discriminator_use_sn=True, feature_match_layer_index=2, leaky_relu_slope=0.2, synthetic_only_mode=True
            ), TypeError
        ),
        "unsupported_activation": expect_failure(
            lambda: ParameterizedSCFGANStructural(
                SCFGANStructuralConfig(5, 2, 3, (7,), "gelu", "bn_structural", False, (7, 6, 5), True, 2, 0.2, True)
            ), SCFGANConfigError
        ),
        "invalid_discriminator_depth": expect_failure(
            lambda: ParameterizedSCFGANStructural(
                SCFGANStructuralConfig(5, 2, 3, (7,), "relu", "bn_structural", False, (7, 6), True, 2, 0.2, True)
            ), SCFGANConfigError
        ),
        "wrong_noise_shape": expect_failure(lambda: model.generate(noise[:, :-1], condition, provenance="SYNTHETIC_BAD"), SCFGANInputError),
        "wrong_condition_shape": expect_failure(lambda: model.generate(noise, condition[:, :-1], provenance="SYNTHETIC_BAD"), SCFGANInputError),
        "wrong_feature_shape": expect_failure(lambda: model.discriminate(features[:, :-1], condition, provenance="SYNTHETIC_BAD"), SCFGANInputError),
        "nan_input": expect_failure(lambda: model.generate(torch.full_like(noise, float("nan")), condition, provenance="SYNTHETIC_NAN"), SCFGANInputError),
        "inf_input": expect_failure(lambda: model.generate(torch.full_like(noise, float("inf")), condition, provenance="SYNTHETIC_INF"), SCFGANInputError),
        "nasa_provenance": expect_failure(lambda: model.generate(noise, condition, provenance="NASA_TRAIN_REAL"), SCFGANProvenanceError),
        "calce_provenance": expect_failure(lambda: model.generate(noise, condition, provenance="CALCE_TRAIN_REAL"), SCFGANProvenanceError),
        "validation_provenance": expect_failure(lambda: model.generate(noise, condition, provenance="SYNTHETIC_VALIDATION"), SCFGANProvenanceError),
        "test_provenance": expect_failure(lambda: model.generate(noise, condition, provenance="SYNTHETIC_TEST"), SCFGANProvenanceError),
        "real_augmentation_request": expect_failure(lambda: model.generate(noise, condition, provenance="SYNTHETIC_REQUEST", request_kind="REAL_AUGMENTATION"), SCFGANProvenanceError),
        "synthetic_soh_request": expect_failure(lambda: model.generate(noise, condition, provenance="SYNTHETIC_REQUEST", request_kind="SYNTHETIC_SOH_GENERATION"), SCFGANProvenanceError),
    }


def main() -> None:
    coverage_configs = (
        config_for(activation="relu", norm_mode="bn_structural", generator_sn=False, discriminator_depth=3, feature_dim=3),
        config_for(activation="leaky_relu", norm_mode="conditional_affine_bn_structural", generator_sn=True, discriminator_depth=4, feature_dim=4),
    )
    coverage = [run_coverage_case(index, config) for index, config in enumerate(coverage_configs)]
    condition_paths = condition_path_tests()
    smoke = optimizer_smoke_test()
    deterministic = determinism_test()
    fail_closed = fail_closed_tests()
    coverage_pass = all(
        all(
            case[key]
            for key in (
                "generator_forward", "discriminator_forward", "tanh_bounds", "sigmoid_bounds", "condition_concatenation", "feature_match_extraction", "finite_losses", "loss_scalar_shapes", "lambda_applied", "generator_backward", "discriminator_backward", "feature_match_gradient_to_generator", "generator_sn_support", "discriminator_sn_support"
            )
        )
        for case in coverage
    )
    condition_pass = all(condition_paths.values())
    all_passed = coverage_pass and condition_pass and smoke["pass"] and deterministic and all(fail_closed.values())
    audit = {
        "artifact_type": "AMBIGUITY_PRESERVING_SYNTHETIC_SCFGAN_STRUCTURAL_AUDIT",
        "noise_distribution": "STANDARD_NORMAL",
        "noise_dimension_provenance": "SYNTHETIC_STRUCTURAL_FIXTURE_ONLY__PAPER_DIMENSION_MISSING",
        "condition_mode": "ABSTRACT_SYNTHETIC_CONDITION_VECTOR",
        "real_condition_semantics_inference": False,
        "real_condition_encoding_inference": False,
        "feature_dimensions_structurally_tested": [3, 4],
        "generic_feature_4_is_soh": False,
        "real_feature_dim_inference": False,
        "synthetic_soh_generation": False,
        "generator_activations_tested": ["relu", "leaky_relu"],
        "generator_normalization_modes_tested": ["bn_structural", "conditional_affine_bn_structural"],
        "generator_sn_options_tested": [False, True],
        "discriminator_depths_tested": [3, 4],
        "generator_output_activation": "TANH__PAPER_EXACT",
        "discriminator_output_activation": "SIGMOID__PAPER_EXACT",
        "feature_match_layer": "THIRD_HIDDEN_LAYER__PAPER_EXACT_WITH_DEPTH_CONTRADICTION_RETAINED",
        "feature_match_lambda": FEATURE_MATCH_LAMBDA,
        "feature_match_lambda_provenance": "PAPER_EXACT",
        "loss_implementation_form": "PAPER_EQUATION_EQUIVALENT_NUMERIC_STABLE_BINARY_CROSS_ENTROPY",
        "ambiguity_selection_performed": AMBIGUITY_SELECTION_PERFORMED,
        "generation_quality_metrics_run": False,
        "real_scfgan_training": False,
        "real_synthetic_rows_created": False,
        "real_battery_features_used": False,
        "validation_values_used": False,
        "test_loaded": False,
        "model_metrics_used": False,
        "grid_search_run": False,
        "coverage_matrix": coverage,
        "condition_path": condition_paths,
        "optimizer_smoke": smoke,
        "determinism_test": deterministic,
        "fail_closed_tests": fail_closed,
        "real_data_guard_test": all(fail_closed[key] for key in ("nasa_provenance", "calce_provenance", "validation_provenance", "test_provenance", "real_augmentation_request", "synthetic_soh_request")),
        "all_structural_tests_passed": all_passed,
    }
    output_path = PROJECT_ROOT / "output" / "scfgan" / "scfgan_structural_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not all_passed:
        raise SystemExit("Synthetic SCF-GAN structural tests failed")
    print(output_path)


if __name__ == "__main__":
    main()
