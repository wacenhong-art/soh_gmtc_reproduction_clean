# Phase 9C SCF-GAN generation-quality diagnostic

## Scope and controls

This post-hoc diagnostic compared persisted HF-only synthetic sets with target first-20%-life paired-charge HF references. Target HF was used only for evaluation after GAN training; no GAN or downstream model was trained or changed. TEST numerical access was zero.

```text
SCFGAN_RETRAINED=False
DOWNSTREAM_MODEL_TRAINED=False
TEST_LOADED=False
TEST_ACCESS_COUNT=0
KL_IMPLEMENTATION=TIER2_DETERMINISTIC_DIAGNOSTIC
OUTLIER_IMPLEMENTATION=TIER2_DETERMINISTIC_DIAGNOSTIC
```

## Metric definitions

KL is `D_KL(real || synthetic)` using 20 bins over the real/synthetic common physical-HF range, with epsilon `1e-8` added before normalization. Wasserstein distance is calculated in physical HF units. Outlier ratio uses the real reference `[Q1-1.5IQR, Q3+1.5IQR]` interval. These deterministic Tier 2 diagnostics are not claimed to reproduce an unreported paper implementation.

## Aggregate maxima

- `max_mean_pct_diff=66.63942343481568`
- `max_std_pct_diff=121.28714204508901`
- `max_kl=23.089131117918246`
- `max_wasserstein=3120.088398525971`
- `max_outlier_ratio=1.0`

## Pooled physical-HF results

| Battery | HF | mean pct diff | std pct diff | KL | Wasserstein | outlier ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0007 | HF1 | 7.192525 | 83.620222 | 2.145569 | 241.120186 | 0.412121 |
| B0007 | HF2 | 52.000041 | 61.508369 | 23.089131 | 3059.862608 | 1.000000 |
| B0007 | HF3 | 59.151362 | 39.046506 | 2.027476 | 840.533699 | 0.903030 |
| B0018 | HF1 | 14.757270 | 74.421439 | 10.864723 | 433.578999 | 0.246154 |
| B0018 | HF2 | 27.634245 | 14.406435 | 21.526756 | 1945.183363 | 0.984615 |
| B0018 | HF3 | 53.222470 | 99.381090 | 2.439282 | 689.635476 | 0.761538 |

## Paper reference values

`PAPER_MAX_MEAN_PCT_DIFF_REF=0.52`; `PAPER_MAX_STD_PCT_DIFF_REF=2.86`; `PAPER_MAX_KL_REF=0.54`; `PAPER_MAX_WASSERSTEIN_REF=58.73`.
They are recorded for context only. Differences in unreported metric implementation do not by themselves establish paper-reproduction failure.

## Objective variance facts

- `B0007`: `all_sets_have_nonzero_variance=True`; set-level mean and standard-deviation variability is recorded feature-wise in the JSON companion.
- `B0018`: `all_sets_have_nonzero_variance=True`; set-level mean and standard-deviation variability is recorded feature-wise in the JSON companion.

The JSON companion contains every per-set and pooled HF1/HF2/HF3 calculation, including means, standard deviations, min/max, NaN/Inf counts, variance, KL, Wasserstein, and IQR-based outlier ratio. This report makes no parameter-change recommendation.
