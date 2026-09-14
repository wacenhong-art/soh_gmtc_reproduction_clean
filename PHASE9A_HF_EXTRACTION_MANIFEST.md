# Phase 9A HF Extraction Manifest

STATUS=BLOCKED__HF2_EXTRACTION_CONTRACT_UNSATISFIED  
RUN_SCOPE=FIRST_AUTHORIZED_B0007_TRAIN_PAIRED_CHARGE_ONLY  
REAL_HF_VALUES_USED=True  
REAL_NUMERICAL_TARGET_VALUES_USED=False  
VALIDATION_HF_VALUES_USED=False  
TEST_HF_VALUES_USED=False

The frozen Phase 1 HF formula was evaluated without alteration. HF2 requires the first post-4.2 V point satisfying `0 <= Current_charge <= 0.02`. That condition was unavailable in the first authorized B0007 TRAIN record. No feature values are recorded here.

HF_VALID_COUNTS_BY_BATTERY=NOT_MATERIALIZED__FAIL_CLOSED  
HF_EXTRACTION_FAILURE_COUNT=1__FIRST_AUTHORIZED_TRAIN_RECORD  
IMPUTATION_OR_DELETION_APPLIED=False  
NEXT_ACTION=RESULT_BLIND_HF2_SIGN_AND_ENDPOINT_SEMANTICS_AUDIT_OR_EXPLICIT_HUMAN_PROTOCOL_AUTHORIZATION

## Executed-run confirmation after PLAN-CHANGE-0017_REVISED — 2026-09-14

`STATUS=PASS__FROZEN_HF_ENDPOINT_PROTOCOL_APPLIED`
`HF_FORMULAS=HF1=t_Vmax-t_0; HF2=t_Imin-t_Vmax; HF3=t_4p0-t_3p8`
`VOLTAGE_FIELD=Voltage_measured`
`T_IMIN_FIELD=Current_charge`
`TEST_HF_VALUES_USED=False`

| Battery | TRAIN legal paired charges | VALIDATION legal paired charges | HF extraction failures | `t_Imin` final-timestamp fallbacks |
|---|---:|---:|---:|---:|
| B0007 | 117 | 34 | 0 | 0 |
| B0018 | 92 | 26 | 0 | 8 |

All values were finite. The eight B0018 fallbacks are the frozen,
explicitly-labelled `Current_charge<=0.02 A`-absent endpoint policy; they are
not imputation, deletion, capacity/SOH repair, or a model-driven choice.
