# NASA Discharge-Centered Pairing Audit

AUDIT_SCOPE=STRUCTURAL_METADATA_ONLY  
CANDIDATE_E=DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE  
REAL_DATA_METADATA_USED=True  
REAL_NUMERICAL_TARGET_VALUES_USED=False  
REAL_HF_VALUES_USED=False  
TEST_NUMERICAL_VALUES_USED=False

## Rule

For each discharge record with a Capacity field, search backward in operation order and assign the nearest earlier unconsumed charge. Impedance records may appear between members. Once assigned, neither record can be reused. The rule depends only on operation order and operation/field metadata.

## Structural results

| Battery | Charge | Discharge | Valid pairs | Unpaired charge | Unpaired discharge | Impedance-between pairs | Consecutive-charge blocks | Consecutive charge dropped | One-to-one |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0005 | 170 | 168 | 168 | 2 | 0 | 137 | 2 | 2 | Yes |
| B0006 | 170 | 168 | 168 | 2 | 0 | 137 | 2 | 2 | Yes |
| B0007 | 170 | 168 | 168 | 2 | 0 | 137 | 2 | 2 | Yes |
| B0018 | 134 | 132 | 132 | 2 | 0 | 23 | 1 | 1 | Yes |

All paired records satisfy `charge_record_index < discharge_record_index`. No charge or discharge is reused, and no candidate ambiguity remains under this deterministic consumption rule. The same rule and the same structural outcome hold across all four audited batteries.

## Evidence and decision boundary

`PAIRING_EVIDENCE_STRENGTH=STRONG_ENGINEERING_STRUCTURAL`  
`CANDIDATE_E_GATE=PASS__DETERMINISTIC_ONE_TO_ONE_STRUCTURAL_MAPPING`

This is `RAW_SEQUENCE_STRUCTURAL_EVIDENCE`, not author confirmation or paper-exact evidence. It proposes, but does not apply:

```text
PROPOSED_SAMPLE_IDENTITY=DISCHARGE_CENTERED_CHARGE_DISCHARGE_PAIR
input_HF=nearest_preceding_charge
target_capacity=current_discharge
PROPOSED_SPLIT_UNIT=VALID_PAIRED_SUPERVISED_CYCLES
```

NEXT_SCIENTIFIC_ACTION=ASK_ADVISOR_TO_CONFIRM_DISCHARGE_CENTERED_NEAREST_PRECEDING_CHARGE_POLICY
