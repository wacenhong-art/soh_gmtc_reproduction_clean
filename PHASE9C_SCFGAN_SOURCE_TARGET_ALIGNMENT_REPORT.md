# Phase 9C SCF-GAN source-target alignment diagnostic

## Scope

This is a post-hoc diagnostic only. It did not retrain the GAN, regenerate synthetic HF, train a downstream model, change PLAN-CHANGE-0018, or access TEST. B0005/B0006 provides the permitted real source-HF reference. B0007/B0018 first-20%-life paired-charge HF provides a target-quality reference only; no target reference entered GAN training.

## Condition provenance audit

```text
CURRENT_CONDITION_IMPLEMENTATION=NORMALIZED_CYCLE_POSITION
PAPER_EQ5_CONDITION_TEXT=CYCLE_PERIOD
PAPER_EXPERIMENT_CONDITION_TEXT=VOLTAGE_RANGE
PAPER_CONDITION_SEMANTICS_CONTRADICTION=True
```

## Alignment classifications

`GEN_SOURCE_ALIGNMENT_STATUS=POOR__DESCRIPTIVE_MULTI_METRIC_MISMATCH_OBSERVED`  
`SOURCE_TARGET_DOMAIN_SHIFT_STATUS=PRESENT__DESCRIPTIVE_SOURCE_TARGET_DIFFERENCES_OBSERVED`  
`CONDITIONWISE_ALIGNMENT_STATUS=LIMITED__SYNTHETIC_TARGET_SUPPORT_ONLY_IN_FIRST_20_PERCENT_BIN`

## Pooled maxima by comparison

| Target | comparison | max mean pct diff | max std pct diff | max KL | max Wasserstein | max outlier ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0007 | A_GENERATED_VS_SOURCE | 50.865306 | 84.369736 | 16.392455 | 1133.277849 | 0.000000 |
| B0007 | B_SOURCE_VS_TARGET | 44.618629 | 34.653327 | 2.950288 | 1930.781297 | 0.952381 |
| B0007 | C_GENERATED_VS_TARGET | 59.151362 | 83.620222 | 23.089131 | 3059.862608 | 1.000000 |
| B0018 | A_GENERATED_VS_SOURCE | 51.331110 | 83.707964 | 16.227682 | 1169.110169 | 0.000000 |
| B0018 | B_SOURCE_VS_TARGET | 36.723245 | 227.754052 | 1.843877 | 969.449527 | 0.672619 |
| B0018 | C_GENERATED_VS_TARGET | 53.222470 | 99.381090 | 21.526756 | 1945.183363 | 0.984615 |

## Condition-bin observability

Source condition positions span the full lifecycle. Existing target synthetic sets were generated only for target first-20%-life positions; therefore only `[0,.2)` has synthetic support. Empty bins are reported as empty rather than filled or extrapolated.

The JSON companion contains all per-HF, per-set, pooled, and five-bin source/synthetic mean and standard-deviation statistics. It records the deterministic KL, physical-domain Wasserstein, and reference-IQR outlier definitions. No automatic parameter change is implied.
