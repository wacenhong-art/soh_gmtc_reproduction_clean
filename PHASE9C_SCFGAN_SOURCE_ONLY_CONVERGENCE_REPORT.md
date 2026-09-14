# Phase 9C SCF-GAN source-only convergence diagnostic

## Scope controls

```text
TARGET_INFORMATION_USED_FOR_TRAINING=False
TARGET_INFORMATION_USED_FOR_MODEL_SELECTION=False
TEST_ACCESS_COUNT=0
SCFGAN_RETRAINED=True
DOWNSTREAM_MODEL_TRAINED=False
EARLY_STOPPING=False
GRID_SEARCH=False
```

The model was initialized anew with seed 42 and trained for exactly 300 epochs using B0005/B0006 source HF only. Snapshot comparison uses a single fixed evaluation noise draw and the source cycle-position conditions. No snapshot was selected as best.

## Source-alignment trajectory

| epoch | mean diff | std diff | KL | Wasserstein | outlier ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 36.227945 | 50.878466 | 9.126222 | 954.778244 | 0.000000 |
| 40 | 14.031554 | 33.783253 | 2.870667 | 625.790657 | 0.000000 |
| 60 | 41.369095 | 32.228088 | 5.448761 | 884.407773 | 0.000000 |
| 80 | 22.646307 | 49.727035 | 0.887348 | 936.108225 | 0.047619 |
| 100 | 8.456610 | 33.076884 | 1.329216 | 537.371656 | 0.025794 |
| 120 | 3.259952 | 7.258391 | 1.121632 | 146.693316 | 0.002976 |
| 140 | 11.944058 | 22.547616 | 0.505330 | 326.791145 | 0.000000 |
| 160 | 5.350387 | 16.887430 | 0.279502 | 179.403073 | 0.000000 |
| 180 | 2.145466 | 13.321851 | 0.282926 | 198.464703 | 0.000000 |
| 200 | 10.923458 | 16.010871 | 2.787549 | 329.600002 | 0.000000 |
| 220 | 4.925850 | 14.250299 | 1.362443 | 140.644578 | 0.000000 |
| 240 | 1.576043 | 10.729736 | 0.969303 | 180.430999 | 0.000000 |
| 260 | 4.600680 | 12.863684 | 0.977246 | 253.882108 | 0.001984 |
| 280 | 3.325332 | 12.617625 | 0.973423 | 198.252221 | 0.000000 |
| 300 | 3.252330 | 7.478565 | 0.197743 | 140.417945 | 0.000000 |

## Descriptive convergence facts

`SOURCE_ALIGNMENT_IMPROVED=True` using the registered all-five-metric non-worsening plus at-least-one-improvement Epoch-20-to-Epoch-300 definition.
`LATE_TRAINING_STABILITY_STATUS=OSCILLATORY__DIRECTION_REVERSALS_OBSERVED`  
`EPOCH_200_TO_300_STABLE_PLATFORM=False`  
`MODE_COLLAPSE_STATUS=NO_ZERO_VARIANCE_FEATURE_COLLAPSE_OBSERVED_AT_SNAPSHOTS`

Per-HF snapshot metrics, loss traces, source/generated condition-bin means and standard deviations, NaN/Inf counts, generated variance, late ranges, and direction-reversal counts are contained in the JSON companion. These facts were not used to alter any frozen parameter.
