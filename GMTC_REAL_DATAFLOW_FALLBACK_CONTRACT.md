# G-MTC Real-Dataflow Fallback Contract — PLAN-CHANGE-0010

`STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_CROSS_PHASE_FALLBACK`

`CROSS_PHASE_EXECUTION_FALLBACK=HF_INPUT_TO_COMPONENT_TARGET_MODE`

`PROVENANCE=HUMAN_RESULT_BLIND_CROSS_PHASE_FALLBACK`

`MINIMUM_ADDITIONAL_ASSUMPTION=True`

`RESULT_BLIND=True`

`PERFORMANCE_BLIND=True`

`TEXT_EVIDENCE_PRIORITY=True`

`AUTHOR_OVERRIDE_PENDING=True`

This is an execution contract, not a claim that the paper unambiguously specifies this path.  It applies no condition encoding, GAN architecture choice, STL period, branch forecast horizon, final-domain conversion, synthetic SOH, real generated row, or learned model.

## Two deliberately separate real-data paths

```text
SOURCE BATTERIES (approved full-range source role; HF distribution only)
    │
    └── future SCF-GAN training on allowed TRAIN/source HF data only
            │
            └── target TRAIN synthetic HF1/HF2/HF3 only
                    │
                    │  [no synthetic SOH; no validation/test label]
                    ▼
TARGET TRAIN observed data
    ├── observed HF1/HF2/HF3 ───────────────────────────┐
    │                                                    │
    └── future synthetic HF1/HF2/HF3 (paired only to the │
        same observed TRAIN-cycle identity/label)        │
                                                         ▼
                                     multivariate HF1/HF2/HF3 input windows
                                          ├── Trend TimesNet ──────► T prediction
                                          ├── Seasonal TimesNet ───► S prediction
                                          └── iTransformer ───────► R prediction

TARGET TRAIN observed capacity
    │
    └── STL (capacity-domain, only after a separately valid real period)
            ├── Trend component T ─────────────────────────► Trend target
            ├── Seasonal component S ──────────────────────► Seasonal target
            └── Residual component R ──────────────────────► Residual target

T/S/R predictions ──► Adaptive Fusion ──► final capacity/SOH-domain estimate
                                            (final domain/normalization unresolved)
```

`INPUT_PATH=HF1_HF2_HF3_WINDOWS_TO_ALL_THREE_BRANCHES`

`TARGET_PATH=OBSERVED_TRAIN_CAPACITY_TO_STL_TO_T_S_R_COMPONENT_TARGETS`

`STL_INPUT_SOURCE=OBSERVED_TRAIN_CAPACITY`

`STL_OUTPUTS=TREND__SEASONAL__RESIDUAL`

`BRANCH_INPUT_DOMAIN=MULTIVARIATE_HEALTH_FEATURE_WINDOW__HF1_HF2_HF3`

`TREND_BRANCH_TARGET=STL_TREND_COMPONENT`

`SEASONAL_BRANCH_TARGET=STL_SEASONAL_COMPONENT`

`RESIDUAL_BRANCH_TARGET=STL_RESIDUAL_COMPONENT`

`BRANCH_TARGET_ALIGNMENT_STATUS=UNRESOLVED`

`FORECAST_ALIGNMENT_UNRESOLVED=True`

`FINAL_OUTPUT_SEMANTICS_UNRESOLVED=True`

## SCF-GAN output and pairing limits

`SCFGAN_REAL_OUTPUT_EXECUTION_POLICY=HF1_HF2_HF3_ONLY`

`PAPER_OUTPUT_SEMANTICS_STATUS=CONTRADICTORY`

`SYNTHETIC_SOH_GENERATION=PROHIBITED_UNDER_CURRENT_FALLBACK`

`SOH_GENERATED_BY_SCFGAN=NOT_ASSUMED`

`SYNTHETIC_FEATURE_LABEL_POLICY=REUSE_OBSERVED_TRAIN_CYCLE_LABEL_ONLY`

If future augmentation is separately authorized, each synthetic HF vector may be paired only with the observed label and component target belonging to the same TRAIN-cycle identity.  It must not generate a target/SOH label, use a validation/test label, infer a target from the full lifecycle, or be used to select any protocol.

`CURRENT_REAL_AUGMENTATION_EXECUTED=False`

## Retained unresolved fields and guards

`SCFGAN_CONDITION_SEMANTICS_STATUS=CONTRADICTORY`

`CYCLE_CONDITION_ENCODING_STATUS=MISSING`

`VOLTAGE_CONDITION_ENCODING_STATUS=MISSING`

`SCFGAN_ARCHITECTURE_CONTRADICTIONS_REMAIN=True`

`STL_REAL_BATTERY_PERIOD_MISSING=True`

`ITRANSFORMER_TARGET_ALIGNMENT_MISSING=True`

`TIMESNET_TARGET_ALIGNMENT_MISSING=True`

`TEST_LOADED=False`

`VALIDATION_VALUES_USED=False`

`REAL_DATA_USED=False`

`REAL_SCFGAN_TRAINING=False`

`REAL_SYNTHETIC_ROWS_CREATED=False`

`MODEL_TRAINED=False`

`GRID_SEARCH_RUN=False`

## Gate consequence

`GMTC_CROSS_PHASE_DATAFLOW_GATE=PASS_WITH_HUMAN_RESULT_BLIND_FALLBACK`

`FRAMEWORK_REPRODUCTION_DATAFLOW=EXECUTABLE_UNDER_HUMAN_FALLBACK`

`NUMERICAL_REPRODUCTION_DATAFLOW=NOT_PAPER_EXACT__AUTHOR_OVERRIDE_PENDING`

The gate validates only that the active contracts distinguish the input and target paths.  It does not clear any real-data fidelity gate and it does not authorize Phase 7C implementation in this run.
