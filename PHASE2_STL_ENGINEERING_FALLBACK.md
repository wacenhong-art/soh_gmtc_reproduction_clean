# Phase 2B — Human-authorized STL Engineering Fallback Contract

**Task:** `PHASE_2B_STL_ENGINEERING_FALLBACK_FREEZE_AND_MINIMAL_IMPLEMENTATION`
**Scope:** structural STL preparation only; no predictive model, metric, search, synthetic rows, validation decomposition, or test capacity access.

## Provenance boundary

| Field | Frozen value | Provenance | Paper-exact? |
|---|---|---|---|
| STL input | capacity sequence | `PAPER_CROSS_EVIDENCE` | No |
| Period rule | train-only structural ACF/spectral agreement | `HUMAN_ENGINEERING_FALLBACK` | No |
| Fit mode | `PARTITION_ISOLATED_OFFLINE_STL` | `HUMAN_LEAKAGE_SAFE_FALLBACK` | No |
| Smoother settings | resolved `statsmodels` defaults after a period is frozen | `LIBRARY_DEFAULT_ENGINEERING_FALLBACK` | No |
| Validation component policy | independently decompose validation using train-frozen parameters | `HUMAN_LEAKAGE_SAFE_FALLBACK` | No |
| Test component policy | do not create before final model freeze | `HUMAN_LEAKAGE_SAFE_FALLBACK` | No |

`AUTHOR_OVERRIDE_PENDING=True`
`RESULT_BLIND=True`
`PERFORMANCE_BLIND=True`

## Execution and leakage contract

```text
STL_EXECUTION_MODE=PARTITION_ISOLATED_OFFLINE_STL
TRAIN_COMPONENT_TARGETS=STL(TRAIN_CAPACITY_ONLY)
VALIDATION_COMPONENT_TARGETS=STL(VALIDATION_CAPACITY_ONLY__TRAIN_FROZEN_PARAMETERS)
TEST_COMPONENT_TARGETS=NOT_CREATED_UNTIL_FINAL_MODEL_FREEZE
INTER_PARTITION_LEAKAGE=False
INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE=True
STL_CAUSALITY_STATUS=NON_CAUSAL_OFFLINE_DECOMPOSITION
FULL_LIFECYCLE_STL_BEFORE_SPLIT=PROHIBITED
TRAIN_PLUS_VALIDATION_STL=PROHIBITED
TRAIN_PLUS_VALIDATION_PLUS_TEST_STL=PROHIBITED
TEST_LOADED=False
TEST_STL_EXECUTED=False
```

Partition isolation prevents one partition from contributing values to another. It does **not** make standard LOESS/STL causal: an observation may use later observations within its own supplied partition. This fallback must never be described as online, causal, or real-time STL.

## Deterministic period discovery

The discovery input is only a declared Protocol-A `TRAIN` or Protocol-B `TRAIN_REAL` capacity sequence. Linear least-squares detrending is applied before both estimators; STL itself is never used for detrending.

1. Candidate range is every integer `p` in `[2, min(floor(N_train/2), floor(N_validation/2))]`; the validation count is metadata-only and no validation value is read.
2. ACF selects the largest positive normalized autocorrelation over that range.
3. The periodogram selects the largest non-DC FFT power whose rounded `N/frequency_bin` lies in that range.
4. A period is frozen only if the two candidates differ by at most one cycle; the selected value is their rounded arithmetic mean.
5. Prefixes containing 80%, 90%, and 100% of the same training sequence must each resolve and have a selected-period spread no larger than one.

This is `STRUCTURAL_PERIOD_DISCOVERY`, not hyperparameter optimization. It never reads validation/test capacity values and never uses RMSE, MAE, R2, TimesNet, iTransformer, fusion, or any predictive result.

`DOWNSTREAM_SEQUENCE_LENGTH=10`
`DOWNSTREAM_SEQUENCE_LENGTH_IS_STL_PERIOD=False`
`DOWNSTREAM_SEQUENCE_LENGTH_IS_STL_WINDOW=False`

## Software contract if, and only if, a period is frozen

`STL_LIBRARY=statsmodels.tsa.seasonal.STL`
`STL_LIBRARY_VERSION=0.14.6`
`PYTHON_RUNTIME=C:\\Users\\ROG\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`

The paper does not name a library or any smoother setting. For an allowed frozen period, the module must materialize the library-resolved configuration rather than call unrecorded defaults. In statsmodels 0.14.6 the resolved default family is `seasonal=7`, `seasonal_deg=1`, `trend_deg=1`, `low_pass_deg=1`, all jumps `=1`, and `robust=False`; `trend` and `low_pass` are period-dependent odd integers resolved by the library. No parameter is paper-exact and no setting may be changed because of later prediction performance.

## Current outcome

No target track met the pre-registered ACF/spectral agreement criterion. Therefore no period, period-dependent smoother contract, component decomposition, or STL module implementation was authorized. The reproducible audit utility `scripts/run_phase2_stl_structural_audit.py` contains only train-only period discovery; it does not instantiate STL.
