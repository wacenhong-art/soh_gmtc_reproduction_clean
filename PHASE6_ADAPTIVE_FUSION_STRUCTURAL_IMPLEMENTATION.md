# Phase 6A Adaptive Fusion Specification and Synthetic Structural Implementation

## Scope and dual gates

`SCOPE=SYNTHETIC_PRECONSTRUCTED_BxTx9_FEATURES_ONLY`

`REAL_BATTERY_DATA_USED=False`

`REAL_STL_COMPONENTS_USED=False`

`REAL_BRANCH_PREDICTIONS_USED=False`

`REAL_FUSION_FEATURE_BUILDER_IMPLEMENTED=False`

`TEST_LOADED=False`

`MODEL_METRICS_USED=False`

`GRID_SEARCH_RUN=False`

The module implements the paper-supported topology only. It does not claim that fixture-only attention axis, temporal-score mapping, BatchNorm/dropout placement, final ReLU placement, MSE wiring loss, or first-three-slot selection is paper-exact.

`ADAPTIVE_FUSION_STRUCTURAL_GATE=PASS`

`ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_FEATURE_SEMANTICS_LOSS_AND_BRANCH_ALIGNMENT`

`PHASE_6_STATUS=STRUCTURAL_FUSION_IMPLEMENTED__REAL_DATA_FIDELITY_BLOCKED`

## Explicit implementation contract

`scripts/adaptive_fusion_structural.py` requires an explicit `AdaptiveFusionStructuralConfig`, a rank-3 finite `B×T×9` tensor, and explicit synthetic-only provenance. It has no hidden input-dimension, architecture, or provenance default.

| Contract field | Value | Provenance |
|---|---|---|
| Bi-LSTM | 9 input, 16 hidden/direction, 1 layer, bidirectional | PAPER_EXACT, Table 4 |
| Paper dropout metadata | 0.3 | PAPER_EXACT, Table 4 |
| PyTorch one-layer LSTM internal dropout | Not applied when `num_layers=1` | LIBRARY_BEHAVIOR |
| External post-LSTM dropout | used in synthetic test | ENGINEERING_FIXTURE_ONLY; placement absent from paper |
| Component-attention axis | feature axis | ENGINEERING_FIXTURE_ONLY; Eq.(25) axis missing |
| Time-attention mapping | `Linear(32,32) → Tanh → Linear(32,1) → Softmax(time)` | ENGINEERING_FIXTURE_ONLY; Eq.(26) mapping incomplete |
| Weight network | 32→128→64→32→3, BN, LeakyReLU .1, dropout .3, Softmax | widths/types PAPER_EXACT; placement ENGINEERING_FIXTURE_ONLY |
| Weighted inputs | first three scalar fixture slots at last timestep | ENGINEERING_FIXTURE_ONLY; `first 3d` ambiguous |
| Final head | 3→16→1 with hidden ReLU only | widths/ReLU PAPER_EXACT; placement ENGINEERING_FIXTURE_ONLY |
| Synthetic wiring loss | MSE | NOT_PAPER_LOSS |

`LIBRARY_SINGLE_LAYER_DROPOUT_EFFECT=PYTORCH_LSTM_INTERNAL_DROPOUT_NOT_APPLIED_WHEN_NUM_LAYERS_1`

## Synthetic verification

The deterministic fixture is `4×4×9`: three prediction-like, three error-like, and three ratio-like software placeholders. The pre-registered wiring check is exactly 32 optimizer steps; it passes only if final synthetic MSE is lower than initial synthetic MSE. No configuration comparison, validation value, real battery signal, or predictive metric is used.

| Check | Result |
|---|---|
| Forward, finite synthetic MSE, backward, nonzero finite gradient | PASS |
| Component/temporal/fusion-weight normalization | PASS |
| Three nonnegative fusion weights per sample | PASS |
| Fixed-seed determinism | PASS |
| 32-step synthetic wiring sanity | PASS (`0.1077307612 → 0.0060455883`) |
| Fail-closed shape/rank, NaN/Inf, missing config, real/validation/TEST provenance, real-mode configuration | PASS |

## Real-data fidelity blockers

`FUSION_ERROR_DEFINITION_MISSING`

`FUSION_RATIO_DEFINITION_MISSING`

`FUSION_D_SEMANTICS_MISSING`

`FUSION_FEATURE_SELECTION_AMBIGUOUS`

`FUSION_FINAL_OUTPUT_SEMANTICS_MISSING`

`FUSION_WEIGHT_REGULARIZATION_MISSING`

`FUSION_LAMBDA_MISSING`

Real fidelity additionally inherits unresolved real branch targets, iTransformer input/target/architecture gaps, TimesNet branch gaps, and the missing real-battery STL period. Structural pass does not establish framework, numerical, or paper reproduction.

## Next boundary

`ADVANCEMENT_TO_PHASE_7_STRUCTURAL_WORK=True`

This only permits a future, separately authorized synthetic SCF-GAN specification/structural audit. It does not start Phase 7 or permit real SCF-GAN training or an end-to-end G-MTC claim.
