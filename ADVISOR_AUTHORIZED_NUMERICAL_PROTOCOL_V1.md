# Advisor-Authorized Numerical Protocol V1

PROTOCOL_ID=ADVISOR_TIER2_V1  
PROVENANCE=ADVISOR_AUTHORIZED_FALLBACK  
PAPER_EXACT=False  
RESULT_BLIND_FREEZE=True  
PERFORMANCE_BLIND=True  
TEST_LOCKED=True

## Frozen configuration

```text
SEQ_LEN=10
PRED_LEN=1
TARGET_ALIGNMENT=[t-9:t] -> t+1
STL_PERIOD=7
STL_PERIOD_PROVENANCE=ADVISOR_RANGE_PLUS_RESULT_BLIND_MIDPOINT_FREEZE
NORMALIZATION=MINMAX_0_1
NORMALIZATION_FIT_SCOPE=TRAIN_ONLY
OUT_OF_RANGE_POLICY=CLIP_TO_0_1__RESULT_BLIND_ENGINEERING_RULE
SPLIT=CHRONOLOGICAL_TRAIN_VAL_TEST_7_2_1
TRAIN_FRACTION=0.70
VALIDATION_FRACTION=0.20
TEST_FRACTION=0.10
SPLIT_BOUNDARIES=HALF_OPEN_ZERO_BASED_FLOOR_RULE
PRIMARY_HIDDEN_WIDTH=64
HIDDEN_WIDTH_PROVENANCE=ADVISOR_RANGE_64_128_SMALL_SAMPLE_MIN_COMPLEXITY
FUSION_MODE=ADVISOR_AUTHORIZED_COMPONENT_SUM
FUSION_WEIGHTS=NONE
LOSS=MSE
SCFGAN_OUTPUTS=HF1_HF2_HF3_ONLY
SYNTHETIC_SOH_GENERATION=False
NO_9_FEATURE_ERROR_RATIO_BUILDER=True
STL_FIT_SCOPE=TRAIN_ONLY
TRAIN_STL_CAN_USE_INTRA_TRAIN_OFFLINE_CONTEXT=True
VAL_USED_IN_TRAIN_STL=False
TEST_USED_IN_TRAIN_STL=False
CROSS_SPLIT_CONTEXT=False
VAL_COMPONENT_TARGETS_REQUIRED=False
VAL_STL_REQUIRED=False
```

## Dataflow contract

`[HF1, HF2, HF3]` temporal windows are the three branch inputs. Observed TRAIN capacity is decomposed by the parameterized STL implementation using period 7 into trend, seasonal, and residual targets. Each branch uses the frozen one-step contract. The Tier 2 output is a direct weighted component prediction:

`y_hat = T_hat + S_hat + R_hat`.

HF normalization is fitted independently per feature on TRAIN only. Validation and future test, when separately authorized, use only those TRAIN-fitted min/max values. Values outside the fitted range are clipped to [0,1]; this is an explicit result-blind engineering rule, not paper evidence.

STL components remain in the capacity-domain target scale unless a later approved scale bridge says otherwise. Accordingly `TIER2_FINAL_OUTPUT_DOMAIN=CAPACITY_COMPONENT_DOMAIN` and no unapproved capacity-to-SOH conversion is applied. `PAPER_EXACT_FUSION_STATUS=UNRESOLVED`.

TRAIN capacity only is decomposed using offline STL(period=7); bilateral context within TRAIN is allowed. No validation or test capacity enters the TRAIN STL fit. Validation requires no STL/component labels: its reconstructed component sum is compared only to the observed final next-step target. The complete Tier 2 branch configuration is frozen in `TIER2_BRANCH_EXECUTION_CONFIG_V1.md`.

## Explicit exclusions

No period, width, component-weight, normalization, or architecture comparison is permitted. No paper-undefined current-target error or ratio feature is constructed. No synthetic SOH is generated. No formal training, validation experiment, test loading, or model-metric selection is authorized in this preparation step. Any change requires a new Plan Change. PLAN-CHANGE-0013 corrects PLAN-CHANGE-0012 before numerical execution.
