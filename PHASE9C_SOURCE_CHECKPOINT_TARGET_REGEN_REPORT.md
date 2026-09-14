# Phase 9C source-only checkpoint selection and target regeneration

## Source-only selection

Checkpoint scoring was completed before any target HF reference was materialized. The score is the unweighted mean of epoch metric / epoch-20 metric for mean difference, standard-deviation difference, KL, and Wasserstein. Outlier ratio was reported but excluded. Ties resolve to the earlier epoch.

`SELECTED_EPOCH=300`  
`SELECTED_SOURCE_ALIGNMENT_SCORE=0.10137477864297686`  
`CHECKPOINT_SELECTION_TARGET_BLIND=True`

| epoch | source alignment score | outlier ratio (excluded) |
| ---: | ---: | ---: |
| 20 | 1.000000000 | 0.000000000 |
| 40 | 0.505323494 | 0.000000000 |
| 60 | 0.824671271 | 0.000000000 |
| 80 | 0.670037820 | 0.047619048 |
| 100 | 0.398003737 | 0.025793651 |
| 120 | 0.127297293 | 0.002976190 |
| 140 | 0.292624596 | 0.000000000 |
| 160 | 0.174532565 | 0.000000000 |
| 180 | 0.139981049 | 0.000000000 |
| 200 | 0.316715945 | 0.000000000 |
| 220 | 0.178162048 | 0.000000000 |
| 240 | 0.137395162 | 0.000000000 |
| 260 | 0.188203037 | 0.001984127 |
| 280 | 0.163522221 | 0.000000000 |
| 300 | 0.101374779 | 0.000000000 |

## Regeneration and post-hoc diagnostics

The selected source-only checkpoint generated five new HF-only sets per target in a separate local directory; existing smoke sets were not overwritten. Target first-20%-life HF was accessed only after generation for the diagnostic comparisons below. Capacity, SOH, validation, and TEST were not accessed.

| Target | comparison | max mean pct diff | max std pct diff | max KL | max Wasserstein | max outlier ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0007 | A_GENERATED_VS_SOURCE | 85.841836 | 87.715370 | 14.832650 | 1587.891931 | 0.000000 |
| B0007 | B_SOURCE_VS_TARGET | 44.618629 | 34.653327 | 2.950288 | 1930.781297 | 0.952381 |
| B0007 | C_GENERATED_VS_TARGET | 6.998704 | 84.999915 | 2.653911 | 411.828010 | 0.654545 |
| B0018 | A_GENERATED_VS_SOURCE | 86.533349 | 86.974614 | 14.657018 | 1588.699090 | 0.000000 |
| B0018 | B_SOURCE_VS_TARGET | 36.723245 | 227.754052 | 1.843877 | 969.449527 | 0.672619 |
| B0018 | C_GENERATED_VS_TARGET | 18.032250 | 76.428924 | 11.210176 | 744.984031 | 0.130769 |

## Objective conclusion fields

`SELECTED_SOURCE_ALIGNMENT_STATUS=IMPROVED_VS_EPOCH_20__SOURCE_ONLY_SCORE_MINIMUM_SELECTED`  
`TARGET_ALIGNMENT_AFTER_SOURCE_CONVERGENCE=POST_HOC_TARGET_DIAGNOSTIC__NO_CAUSAL_ATTRIBUTION`  
`TARGET_MISMATCH_PERSISTS_AFTER_SOURCE_CONVERGENCE=True`

The target-mismatch field is a descriptive result from the registered post-generation statistics. It does not assign a cause to conditioning, voltage semantics, domain shift, or architecture, and it triggered no model/protocol modification.

```text
TARGET_INFORMATION_USED_FOR_CHECKPOINT_SELECTION=False
TARGET_INFORMATION_USED_FOR_TRAINING=False
TEST_LOADED=False
TEST_ACCESS_COUNT=0
DOWNSTREAM_MODEL_TRAINED=False
GRID_SEARCH_RUN=False
NEXT_ACTION=RETURN_RESULTS_FOR_HUMAN_REVIEW
```
