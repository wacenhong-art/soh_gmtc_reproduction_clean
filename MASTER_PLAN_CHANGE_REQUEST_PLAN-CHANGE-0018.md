# PLAN-CHANGE-0018 — SCF-GAN real-execution configuration

## Authorization and scope

**Status:** APPLIED — human-authorized, result-blind Tier 2 execution fallback  
**Type:** `HUMAN_AUTHORIZED_RESULT_BLIND_EXECUTION_CONFIGURATION`  
**Scope:** Phase 9C source-only SCF-GAN real-data smoke test only.

This plan change does not revise the frozen Phase 9A/9B data protocol, branch
models, scalar readouts, fusion protocol, data split, or test lock. It does
not make the SCF-GAN implementation paper exact.

## Frozen Tier 2 configuration

| Item | Frozen value |
| --- | --- |
| noise dimension | 16 |
| condition dimension | 1 |
| condition | per-battery normalized paired-cycle position in `[0, 1]` |
| Generator hidden dimensions | `[64, 128, 64]` |
| Generator activation | ReLU |
| Generator normalization | Conditional Batch Normalization structural mode |
| Generator spectral normalization | False |
| Generator output | 3 HF dimensions, Tanh |
| Discriminator hidden dimensions | `[64, 128, 64, 32]` |
| Discriminator activation | LeakyReLU, slope 0.2 |
| Discriminator spectral normalization | True |
| Discriminator output | Sigmoid |
| Feature-match layer | third discriminator hidden-layer output |
| feature-match lambda | 0.1 |
| source batteries | B0005, B0006 only |
| target condition-only batteries | B0007, B0018 |
| GAN scaler | source-only HF MinMax to `[-1, 1]` |
| optimizer | Adam, lr 0.001, betas `(0.9, 0.999)` |
| batch size / seed / smoke epochs | 32 / 42 / 20 |

The configuration gives explicit execution values for the previously
unfrozen Phase 7 parameters. It uses the existing ambiguity-preserving core;
it does not replace its historical structural-test contract.

## Evidence and contradiction preservation

Configuration tables (Tables 2/3) are the Tier 2 evidence priority for these
explicit choices. The published record still contains unresolved conflicts:

```text
PAPER_GENERATOR_ACTIVATION_CONTRADICTION=True
PAPER_GENERATOR_SN_SCOPE_CONTRADICTION=True
PAPER_DISCRIMINATOR_DEPTH_CONTRADICTION=True
PAPER_SCFGAN_CONDITION_CONTRADICTION=True
PAPER_EXACT_SCFGAN_EXECUTION_STATUS=UNRESOLVED
TIER2_SCFGAN_CONFIG_STATUS=HUMAN_AUTHORIZED_RESULT_BLIND_FALLBACK
```

The cycle-period condition implements the explicit Eq. (5) interpretation for
Tier 2. It does not resolve the paper's separate voltage-range-condition text.

## Data and leakage boundaries

Training can materialize only B0005/B0006 charge-HF values and their
per-source-lifecycle condition. No B0007/B0018 HF, capacity, SOH, validation,
or test value may enter GAN training.

For generation, B0007/B0018 contribute only the metadata-derived first-20%
paired-cycle identity count and normalized condition sequence. Five `[N, 3]`
HF-only sets are generated for each target. Capacity and SOH are neither read
nor generated.

The GAN scaler is fit only on source HF and exists only for GAN Tanh-domain
training. Generated HF is inverse-transformed to physical HF. If a later
approved downstream step uses it, that step must separately apply the frozen
Train-only `[0, 1]` HF scaler; the two scalers must never be conflated.

## Non-goals and pass condition

There is no early stopping, hyperparameter selection, quality-based change,
downstream model training, fusion training, SCF-GAN augmentation experiment,
or test access in this smoke test.

The Phase 9C smoke gate may pass only if source-only training, finite
forward/backward losses, finite correctly shaped generated tensors,
inverse-transform completion, and the test lock all pass. Distribution summary
statistics are structural smoke diagnostics only.
