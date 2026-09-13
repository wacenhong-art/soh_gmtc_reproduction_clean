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
PRIMARY_HIDDEN_WIDTH=64
HIDDEN_WIDTH_PROVENANCE=ADVISOR_RANGE_64_128_SMALL_SAMPLE_MIN_COMPLEXITY
FUSION_MODE=ADVISOR_AUTHORIZED_WEIGHTED_FUSION
FUSION_TREND_WEIGHT=0.7
FUSION_SEASONAL_WEIGHT=0.2
FUSION_RESIDUAL_WEIGHT=0.1
LOSS=MSE
SCFGAN_OUTPUTS=HF1_HF2_HF3_ONLY
SYNTHETIC_SOH_GENERATION=False
NO_9_FEATURE_ERROR_RATIO_BUILDER=True
```

## Dataflow contract

`[HF1, HF2, HF3]` temporal windows are the three branch inputs. Observed TRAIN capacity is decomposed by the parameterized STL implementation using period 7 into trend, seasonal, and residual targets. Each branch uses the frozen one-step contract. The Tier 2 output is a direct weighted component prediction:

`y_hat = 0.7*T_hat + 0.2*S_hat + 0.1*R_hat`.

HF normalization is fitted independently per feature on TRAIN only. Validation and future test, when separately authorized, use only those TRAIN-fitted min/max values. Values outside the fitted range are clipped to [0,1]; this is an explicit result-blind engineering rule, not paper evidence.

STL components remain in the capacity-domain target scale unless a later approved scale bridge says otherwise. Accordingly `TIER2_FINAL_OUTPUT_DOMAIN=CAPACITY_COMPONENT_DOMAIN` and no unapproved capacity-to-SOH conversion is applied. `PAPER_EXACT_FUSION_STATUS=UNRESOLVED`.

## Explicit exclusions

No period, width, weight, normalization, or architecture comparison is permitted. No paper-undefined current-target error or ratio feature is constructed. No synthetic SOH is generated. No formal training, validation experiment, test loading, or model-metric selection is authorized in this preparation step. Any change requires a new Plan Change.
