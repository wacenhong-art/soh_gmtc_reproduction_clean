# Phase 2C — STL Period Discovery Method Audit

## Scope and locked boundaries

This audit reuses only Protocol-A `TRAIN` and Protocol-B `TRAIN_REAL` capacity values. Validation **counts** set an executable lag ceiling, but no validation or test capacity values are read. No STL decomposition, component materialization, prediction model, predictive metric, grid search, or period sensitivity run occurs here.

```text
TRAIN_ONLY=True
TEST_LOADED=False
VALIDATION_VALUES_USED=False
TEST_VALUES_USED=False
MODEL_METRICS_USED=False
STL_EXECUTED=False
```

## Recovered Phase 2B algorithm

For a train sequence `x_t`, Phase 2B first computed the OLS residual `r_t=x_t-(a+bt)`. Its ACF score was

```text
ACF(k) = sum_t r_t r_(t+k) / sum_t r_t^2
```

over every integer `k` in `[2, min(floor(N_train/2), floor(N_validation/2))]`, and selected the global maximum. Lag 1 was excluded; lag 2 was included. There were no local-peak requirement, prominence threshold, first-zero-crossing rule, significance criterion, or persistence exclusion.

The spectral rule used `abs(rfft(r))^2` with an implicit sampling interval of one capacity observation, excluded the DC bin, retained bins whose `round(N/frequency_bin)` was in the same executable range, and selected the largest power. It had no window function, taper, peak-prominence rule, harmonic check, or low-frequency contamination exclusion.

`ACF_ORIGINAL_SELECTION_RULE=GLOBAL_ARGMAX_ACF_OVER_LAGS_2_TO_MAX_LAG`
`ACF_ARGMAX_PERIOD_RULE_VALIDITY=INVALID_FOR_PERSISTENT_SERIES`
`ACF_LAG2_ARTIFACT_STATUS=PERSISTENCE_OR_GLOBAL_ARGMAX_ARTIFACT__NOT_PERIOD_EVIDENCE`

The reason is methodological rather than numerical: after only linear detrending, a smooth degradation residual commonly remains highly correlated at the shortest admissible lag. A global short-lag maximum does not establish a recurrent period.

## Full-train structural evidence

| Track | ACF global argmax | OLS ACF local peaks | OLS spectral top / second | Prefix conclusion |
|---|---:|---|---|---|
| A CALCE CS2_36 | 2 | 50 | 53 / 58 | Local 50 and spectral 52–53 are stable; not frozen. |
| A CALCE CS2_38 | 2 | 50, 100 | 50 / 46 | Local 50 and spectral 47–50 are stable; not frozen. |
| A NASA B0007 | 2 | none | 8 / 10 | No stable ACF periodic peak. |
| A NASA B0018 | 2 | none | 5 / 11 | No stable ACF periodic peak. |
| B CALCE CS2_36 | 2 | 50, 94 | 49 / 24 | Local 50 persists; spectral top varies 44–52. |
| B CALCE CS2_38 | 2 | 50, 96 | 54 / 43 | Local 50 persists; spectral top varies 43–54. |
| B NASA B0007 | 2 | 11 | 17 / 11 | Short sequence; family is not stable. |
| B NASA B0018 | 2 | 5, 9 | 13 / 6 | Short sequence; family is not stable. |

The complete numeric ACF-by-lag, local-peak, prominence, approximate white-noise bound, first-zero-crossing, and top-two spectral evidence is stored in `output/stl/stl_period_method_audit.json`. No raw capacity array is stored.

## Preprocessing audit

The audit compared OLS-detrended capacity, first-difference capacity, and normalized OLS residual. Normalization leaves ACF/periodogram rankings invariant. First differences produced many weak, competing local extrema and low top1/top2 spectral ratios (about 1.01–1.15), so they do not independently confirm a periodic family. OLS detrending leaves an irreducible low-frequency contamination risk; this audit does not claim it has removed the complete nonlinear degradation trend.

`PERIOD_DISCOVERY_SIGNAL=OLS_DETRENDED_CAPACITY_PRIMARY_DIAGNOSTIC__FIRST_DIFFERENCE_SENSITIVITY_NONCONFIRMATORY`
`SPECTRAL_LOW_FREQUENCY_CONTAMINATION_STATUS=NOT_EXCLUDABLE_WITH_OLS_DETREND_ONLY__ELEVATED_FOR_SHORT_NASA_PROTOCOL_B_TRACKS`

## Revised detector policy — not frozen

Any future ACF periodicity detector must require a pre-registered minimum period grounded in sampling semantics, battery-cycle domain rationale, and detectability limits; it must then require a local ACF maximum, a pre-registered prominence/significance rule, and an agreement hierarchy:

1. ACF periodic local peak and spectral peak agree within a pre-registered tolerance.
2. ACF and spectral evidence have a simple harmonic/subharmonic relationship with order at most 3.
3. Spectral evidence only.
4. No reliable periodic evidence.

The paper supplies neither a battery-period range nor a minimum seasonal period. The project also cannot responsibly invent one from the downstream 10-step window. Therefore the revised detector is **not frozen**, no `MIN_PERIOD` is selected, and no candidate is promoted to a paper or engineering period.

`REVISED_ACF_PERIOD_RULE=UNFROZEN__LOCAL_PEAK_PROMINENCE_AND_DOMAIN_GROUNDED_MIN_PERIOD_REQUIRED`
`HARMONIC_ANALYSIS_STATUS=ORDER_LE_3_EXECUTED__CALCE_SUGGESTIVE_ONLY`
`CROSS_PROTOCOL_PERIOD_FAMILY_CONSISTENCY=MIXED__SUGGESTIVE_CALCE_AROUND_50__UNRESOLVED_NASA`
`PREFIX_STABILITY_STATUS=CALCE_LOCAL_PEAK_50_SUGGESTIVE__NO_ALL_TRACK_FAMILY_STABILITY`

## Gate conclusion

This is Outcome C. CALCE has diagnostic evidence around 50 observation intervals, but it is not sufficient to establish a project-wide or per-track frozen period without an authorized, pre-registered minimum-period rule. NASA has no stable periodic peak family. The available evidence therefore does not justify a bounded structural sensitivity study yet.

```text
STL_PERIOD_IDENTIFIABLE=False
BOUNDED_STRUCTURAL_CANDIDATES_AVAILABLE=False
STRUCTURAL_CANDIDATE_SET=NONE__CALCE_AROUND_50_IS_DIAGNOSTIC_ONLY
NEW_STL_BLOCKER_FOUND=False__EXISTING_BLOCKER_REFINED
BLOCKER_ID=STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE__ACF_GLOBAL_ARGMAX_INVALID_FOR_PERSISTENT_SERIES
PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE
NEXT_ACTION=HUMAN_FIXED_ENGINEERING_PERIOD_OR_PRIMARY_SOURCE_CLARIFICATION
```
