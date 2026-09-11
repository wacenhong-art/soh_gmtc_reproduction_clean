# Phase 2B — Train-only Structural Period Discovery

**Run scope:** Protocol-A `TRAIN` and Protocol-B `TRAIN_REAL` capacity sequences for the four Phase-1 target batteries.  Capacity was converted only for the selected early training members. Validation counts were used solely to set an executable upper bound; validation capacity values and all test capacity values were not selected, converted, retained, or analysed.

| Protocol | Dataset | Battery | N train | N validation (metadata) | ACF candidate | Spectral candidate | Difference | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | CALCE | CS2_36 | 585 | 195 | 2 | 53 | 51 | `UNRESOLVED` |
| A | CALCE | CS2_38 | 649 | 216 | 2 | 50 | 48 | `UNRESOLVED` |
| A | NASA | B0007 | 102 | 34 | 2 | 8 | 6 | `UNRESOLVED` |
| A | NASA | B0018 | 80 | 27 | 2 | 5 | 3 | `UNRESOLVED` |
| B | CALCE | CS2_36 | 195 | 195 | 2 | 49 | 47 | `UNRESOLVED` |
| B | CALCE | CS2_38 | 216 | 216 | 2 | 54 | 52 | `UNRESOLVED` |
| B | NASA | B0007 | 34 | 34 | 2 | 17 | 15 | `UNRESOLVED` |
| B | NASA | B0018 | 26 | 27 | 2 | 13 | 11 | `UNRESOLVED` |

The agreement tolerance was pre-registered as at most one cycle. Every full training sequence exceeds it; hence every 80%/90%/100% prefix assessment is necessarily `UNSTABLE_OR_UNRESOLVED`. This is a structural identifiability result, not a prediction-performance comparison.

`PERIOD_DISCOVERY_METHOD=ACF_AND_FFT_PERIODOGRAM_WITH_AGREEMENT_RULE`
`PERIOD_DISCOVERY_DETREND_METHOD=ORDINARY_LEAST_SQUARES_LINEAR_DETREND__NO_STL`
`PERIOD_SELECTION_USES_MODEL_METRICS=False`
`PERIOD_SELECTION_USES_VALIDATION_VALUES=False`
`PERIOD_SELECTION_USES_TEST_VALUES=False`
`PERIOD_STABILITY_STATUS=UNSTABLE_OR_UNRESOLVED_FOR_ALL_EIGHT_PROTOCOL_BATTERY_TRACKS`
`NEW_STL_BLOCKER_FOUND=True`
`BLOCKER_ID=STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`
`PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`

The exact reproducible calculation is recorded in [stl_period_discovery.json](output/stl/stl_period_discovery.json). It contains candidates and structural evidence only, not raw capacity arrays or component arrays.
