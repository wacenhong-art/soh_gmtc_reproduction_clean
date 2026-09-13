# SCF-GAN Parameterized Architecture Contract — Phase 7C

`SCOPE=AMBIGUITY_PRESERVING_SYNTHETIC_ONLY_STRUCTURAL_IMPLEMENTATION`

`PAPER_EXACT_VARIANT_SELECTION=False`

`AMBIGUITY_SELECTION_PERFORMED=False`

## Explicit required configuration

`scripts/scfgan_structural.py` requires every one of the following values through `SCFGANStructuralConfig`; no paper, feature, condition, noise, width, or depth default exists.

| Field | Contract | Provenance |
|---|---|---|
| `noise_dim` | positive explicit fixture value | `SYNTHETIC_STRUCTURAL_FIXTURE_ONLY`; paper dimension missing |
| `condition_dim` | positive explicit abstract-vector width | synthetic-only; real semantics/encoding unresolved |
| `feature_dim` | positive explicit generic vector width | 3 and 4 tested; no real-dimension inference |
| `generator_hidden_dims` / `discriminator_hidden_dims` | explicit small fixture widths | synthetic-only, not figure width selection |
| `generator_activation` | `relu` or `leaky_relu` | paper internal contradiction space |
| `generator_norm_mode` | `bn_structural` or `conditional_affine_bn_structural` | latter is a synthetic engineering CBN expression, not paper-exact CBN |
| `generator_use_sn` | explicit `True` or `False` | paper SN-scope contradiction space |
| `discriminator_use_sn` | explicit; coverage requires `True` | paper support for discriminator hidden-layer SN |
| `feature_match_layer_index` | explicit and must be `2` | third hidden layer, while D-depth contradiction remains |
| `leaky_relu_slope` | explicit synthetic value | paper-exact slope not established |

`NOISE_DISTRIBUTION=STANDARD_NORMAL__PAPER_EXACT`

`GENERATOR_OUTPUT_ACTIVATION=TANH__PAPER_EXACT`

`DISCRIMINATOR_OUTPUT_ACTIVATION=SIGMOID__PAPER_EXACT`

`FEATURE_MATCH_LAMBDA=0.1__PAPER_EXACT`

`LOSS_IMPLEMENTATION_FORM=PAPER_EQUATION_EQUIVALENT_NUMERIC_STABLE_BINARY_CROSS_ENTROPY`

## Synthetic-only guard

Every public forward/loss route requires `SYNTHETIC` provenance and an explicit request kind. It rejects NASA, CALCE, real-training, validation, TEST, real-augmentation, and synthetic-SOH requests before computation.

`REAL_CONDITION_SEMANTICS_INFERENCE=False`

`REAL_CONDITION_ENCODING_INFERENCE=False`

`REAL_FEATURE_DIM_INFERENCE=False`

`SYNTHETIC_SOH_GENERATION=False`

The optional fourth fixture dimension is named `GENERIC_FEATURE_4`; it has no SOH meaning.

## Tested ambiguity coverage

| Coverage case | Activation | Normalization | Generator SN | D hidden depth | Generic feature width |
|---|---:|---|---:|---:|---:|
| `AMBIGUITY_COVERAGE_1` | ReLU | BN structural mode | False | 3 | 3 |
| `AMBIGUITY_COVERAGE_2` | LeakyReLU | Conditional-affine BN structural mode | True | 4 | 4 |

This finite matrix covers every requested ambiguity option once. It is not a Cartesian search, ranking, quality assessment, or architecture selection procedure.
