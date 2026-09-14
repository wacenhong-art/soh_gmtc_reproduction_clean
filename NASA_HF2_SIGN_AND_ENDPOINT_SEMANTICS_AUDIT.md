# NASA HF2 Sign and CC-Endpoint Semantics Audit

## Scope and lock

`AUDIT_SCOPE=LEGAL_PAIRED_TRAIN_CHARGE_OPERATIONS_ONLY`  
`TRAIN_CHARGE_TRAJECTORIES_USED=True`  
`REAL_NUMERICAL_TARGET_VALUES_USED=False`  
`VALIDATION_DATA_USED=False`  
`TEST_LOADED=False`  
`TEST_NUMERICAL_VALUE_ACCESS_COUNT=0`  
`MODEL_TRAINED=False`  
`GRID_SEARCH_RUN=False`

This audit inspected only `Time`, `Voltage_measured`, `Voltage_charge`,
`Current_measured`, and `Current_charge` from paired TRAIN charge operations.
It did not inspect a discharge Capacity, SOH, validation record, test record,
model prediction, or model metric.

## External evidence baseline

NASA documents the top-level charge/discharge/impedance operation structure and
defines `Voltage_measured` as terminal voltage, `Current_measured` as battery
output current, and `Voltage_charge`/`Current_charge` as charger measurements.
The official format also places `Capacity` on discharge records, not charge
records. [NASA Li-ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

For B0005/B0006/B0007, the publicly documented NASA protocol is CC charging at
1.5 A to 4.2 V, then CV charging at 4.2 V to a 20 mA cutoff. A published NASA
dataset analysis reports the same CC-CV protocol. [Published CC-CV description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10878932/)

The MathWorks NASA workflow removes incomplete/repeated-charge cycles before
aligning normal charge/discharge measurements and uses interpolated voltage-time
crossings for charging features. It is preprocessing precedent, not evidence of
the target paper's HF2 implementation. [MathWorks NASA SoH workflow](https://www.mathworks.com/help/predmaint/ug/battery-second-life-app-soh-estimation.html)

## Existing extractor and failure diagnosis

`CURRENT_HF2_FIELD=Voltage_charge__and__Current_charge`  
`CURRENT_HF2_SIGN_ASSUMPTION=Current_charge_must_contain_a_post_4p2_sample_in_[0,0.02]_A`  
`CURRENT_HF2_CURRENT_THRESHOLD=0<=Current_charge<=0.02`  
`CURRENT_HF2_START_RULE=first_Voltage_charge>=4.2_V`  
`CURRENT_HF2_END_RULE=first_post_start_Current_charge_in_[0,0.02]_A`

That rule computes a duration from a charger-voltage 4.2 V crossing to the CV
cutoff-current event. It is a CV-stage duration proxy, not the requested
constant-current duration from charge start to the CC-to-CV transition.

The original Phase 9A exception did not identify the battery or pair index. The
diagnostic rerun establishes that B0007 has zero existing-rule failures. The
first actual failure is B0018 TRAIN pair 14 (raw charge record index 40): both
voltage fields reach 4.2 V, the early charger-current magnitude is 1.507 A,
and charger current subsequently decays, but no sampled post-start point meets
the exact non-negative 0–0.02 A rule. No trajectory is stored in this artifact.

`B0007_FAILURE_REASON=NONE__117_OF_117_EXISTING_RULE_AVAILABLE`  
`ACTUAL_FIRST_FAILURE=B0018__TRAIN_PAIR_14__CHARGE_RECORD_40`  
`FAILURE_CLASS=D__ENDPOINT_DEFINITION_INCONSISTENT_WITH_CC_DURATION__PLUS_C__SAMPLED_CUTOFF_RULE_NOT_UNIVERSAL`

## TRAIN structural summary

| Battery | TRAIN charges | `Current_measured` dominant sign by cycle | `Current_charge` dominant sign by cycle | early `abs(Current_charge)` near 1.5 A | `Voltage_measured>=4.2` | `Voltage_charge>=4.2` | legacy HF2 failures |
|---|---:|---|---|---:|---:|---:|---:|
| B0005 | 117 | positive 116, negative 1 | positive 117 | 116 | 117 | 117 | 0 |
| B0006 | 117 | positive 117 | positive 117 | 116 | 117 | 117 | 97 |
| B0007 | 117 | positive 117 | positive 117 | 116 | 117 | 117 | 0 |
| B0018 | 92 | positive 92 | positive 92 | 90 | 92 | 92 | 4 |

Across all 443 legal TRAIN charge operations, `Voltage_measured` reaches or
crosses 4.2 V in 443/443 operations. `Voltage_charge` also reaches 4.2 V in
443/443. Charger-current early-stage magnitude is near 1.5 A in 438/443 and
later absolute-current decay after the terminal-voltage transition is observed
in 442/443. These latter current statistics are corroborating structural
evidence only; they do not redefine the physical end point.

`CURRENT_MEASURED_SIGN_CONSISTENT=NOT_UNIVERSAL__B0005_HAS_ONE_NEGATIVE_DOMINANT_CYCLE`  
`CURRENT_CHARGE_SIGN_CONSISTENT=POSITIVE_FOR_443_OF_443_TRAIN_CHARGES`  
`CURRENT_CHARGE_CC_LEVEL_STATUS=STRONG_STRUCTURAL_SUPPORT__438_OF_443_NEAR_1P5A`  
`VOLTAGE_MEASURED_4P2_ENDPOINT_STATUS=IDENTIFIABLE__443_OF_443`  
`VOLTAGE_CHARGE_4P2_ENDPOINT_STATUS=IDENTIFIABLE__443_OF_443`

## Candidate semantics (methodological executability only)

| Candidate | Definition | Status |
|---|---|---|
| A | `Time[0]` to first `Voltage_measured` 4.2 V crossing | `PASS__443_OF_443_ENDPOINTS_IDENTIFIABLE` |
| B | Candidate A, with `Current_charge` only as CC-consistency evidence | `SUPPORTED__AUXILIARY_CURRENT_CHECK_NOT_UNIVERSAL` |
| C | Existing `Voltage_charge` + exact sampled `Current_charge in [0,0.02]` rule | `FAIL__101_OF_443_TRAIN_CHARGES_UNAVAILABLE_AND_SEMANTICALLY_CV_DURATION` |

No endpoint tolerance or interpolation rule is frozen here. Although a linear
crossing-time interpolation is a plausible future candidate when adjacent
samples bracket 4.2 V, choosing it requires a separate result-blind Plan Change.

## Gate and provenance

`HF2_SEMANTICS_AUDIT_GATE=PASS__CC_DURATION_ENDPOINT_IDENTIFIED`  
`RECOMMENDED_FUTURE_SEMANTICS=charge_start_to_Voltage_measured_4p2_CC_TO_CV_transition`  
`PAPER_EXACT_HF2_EXTRACTION_STATUS=UNRESOLVED`  
`PLAN_CHANGE_REQUIRED=True__PLAN_CHANGE_0017_REQUIRED_BEFORE_IMPLEMENTATION`  
`NEXT_SCIENTIFIC_ACTION=PLAN_CHANGE_0017_FREEZE_WEB_SUPPORTED_HF2_CC_DURATION_SEMANTICS`

Evidence classes: NASA official protocol; NASA official field semantics;
published CC-CV/CCCT preprocessing precedent; and raw TRAIN structural evidence.
This does not establish the target paper's exact HF2 implementation.
