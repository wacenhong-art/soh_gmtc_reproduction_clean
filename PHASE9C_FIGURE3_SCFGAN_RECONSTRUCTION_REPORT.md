# Phase 9C Figure 3 SCF-GAN reconstruction

## Frozen Figure 3 execution configuration

```text
CONDITION_MODE=CYCLE_PLUS_BATTERY_TYPE
CONDITION_DIM=2
CONDITION_VECTOR=[NORMALIZED_CYCLE_POSITION, BATTERY_TYPE_CONSTANT_0.0]
OUTPUT_VECTOR=[HF1, HF2, HF3, SOH]
SOH=Capacity_Ah/2.0
GENERATOR=18_TO_256_TO_512_TO_256_TO_4__CBN_RELU_TANH__NO_SN
DISCRIMINATOR=6_TO_256_TO_512_TO_256_TO_1__SN_LEAKY_RELU_0.2_DROPOUT_0.3__RAW_LOGIT
FEATURE_MATCHING=THIRD_HIDDEN_OUTPUT__LAMBDA_0.1
SCALER=SOURCE_ONLY_FOUR_VECTOR_MINMAX_MINUS1_TO_PLUS1
```

## Gates

`FIGURE3_ARCHITECTURE_CONTRACT_GATE=PASS`
`STAGE_A_STATUS=PASS`; epochs=20
`SOURCE_ONLY_CONVERGENCE_STATUS=PASS__300_EPOCHS__15_SNAPSHOTS`; snapshots=15
`SELECTED_SOURCE_EPOCH=200`
`SOURCE_ALIGNMENT_SCORE=0.06953425786741885`

## Safety

```text
TARGET_INFORMATION_USED_FOR_TRAINING=False
TARGET_INFORMATION_USED_FOR_CHECKPOINT_SELECTION=False
TEST_ACCESS_COUNT=0
DOWNSTREAM_MODEL_TRAINED=False
GRID_SEARCH_RUN=False
PLAN_CHANGE_0018_MODIFIED=False
```

## Post-hoc target status

`TARGET_ALIGNMENT_POST_HOC=RECORDED__NOT_USED_FOR_SELECTION`
`SOH_GENERATION_POST_HOC=RECORDED__FIGURE3_FOUR_DIMENSION_OUTPUT`

The selected checkpoint came solely from source-distribution score. Figure-3 generated-vs-source/generated-vs-target HF comparisons, the retained PLAN-CHANGE-0018 fallback comparisons, and SOH post-hoc statistics are reported in the JSON companion without checkpoint reselection.

### B0007 — five sets pooled

| Variable | Figure3 generated vs source mean % diff | Figure3 generated vs target mean % diff |
| --- | ---: | ---: |
| HF1 | 47.548514 | 4.835884 |
| HF2 | 15.635294 | 12.046590 |
| HF3 | 86.414209 | 3.238744 |
| SOH | 18.328940 | 1.187177 |

Generated sets: 5 x `[33, 4]`. The full deterministic 20-bin KL, Wasserstein, IQR-outlier, min/max, variance and PC0018 HF-only comparison records are in the JSON companion.

### B0018 — five sets pooled

| Variable | Figure3 generated vs source mean % diff | Figure3 generated vs target mean % diff |
| --- | ---: | ---: |
| HF1 | 46.437152 | 11.046088 |
| HF2 | 13.912697 | 4.421051 |
| HF3 | 85.228380 | 17.206508 |
| SOH | 18.192276 | 3.330570 |

Generated sets: 5 x `[26, 4]`. The full deterministic 20-bin KL, Wasserstein, IQR-outlier, min/max, variance and PC0018 HF-only comparison records are in the JSON companion.
