# Tier 2 SCF-GAN Figure 3 reconstruction protocol V1

This is a paper-closest Figure-3-priority Tier 2 reconstruction, not a
paper-exact claim. It is separate from PLAN-CHANGE-0018's cycle-only,
three-HF fallback.

## Frozen architecture

| Module | Contract |
| --- | --- |
| Condition | `[normalized_cycle_position, battery_type_constant_0.0]` |
| Generator | `18 -> 256 -> 512 -> 256 -> 4`; CBN + ReLU; Tanh output; no SN |
| Discriminator | `6 -> 256 -> 512 -> 256 -> 1`; hidden SN, LeakyReLU(0.2), Dropout(0.3); raw logit |
| Features | `[HF1, HF2, HF3, SOH]`, with `SOH=Capacity_Ah/2.0` |
| Loss | `BCEWithLogitsLoss`, plus feature matching at hidden layer 3, lambda 0.1 |
| Scaling | source B0005/B0006 four-vector MinMax `[-1,1]` only |
| Runtime | Adam, lr 0.001, betas `(0.9,0.999)`, batch 32, CPU, seed 42 |

## Safeguards

`c2=0.0` is a constant single-paper-battery-type code. It is not a battery ID.
Target numeric data is prohibited until post-generation diagnostics. Target
information cannot select a checkpoint. The original Phase 9A Train-only
`[0,1]` scaler is not reused or modified.

## Retained evidence status

```text
PAPER_EXACT_STATUS=UNRESOLVED_INTERNAL_CONTRADICTION
PAPER_GENERATOR_ACTIVATION_CONTRADICTION=True
PAPER_GENERATOR_SN_SCOPE_CONTRADICTION=True
TABLE3_DISCRIMINATOR_DEPTH_CONTRADICTION=True
FIGURE3_THREE_HIDDEN_LAYER_EXECUTION=True
VOLTAGE_RANGE_DIRECT_IMPLEMENTATION=BLOCKED_BY_HF1_HF2_PARTIAL_WINDOW_UNDEFINED
```
