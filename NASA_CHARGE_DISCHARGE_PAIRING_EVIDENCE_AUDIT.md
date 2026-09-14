# NASA Charge-to-Discharge Capacity Pairing Evidence Audit

AUDIT_SCOPE=STRUCTURAL_METADATA_ONLY  
REAL_DATA_METADATA_USED=True  
REAL_NUMERICAL_TARGET_VALUES_USED=False  
REAL_HF_VALUES_USED=False  
TEST_NUMERICAL_VALUES_USED=False

## Official-format and raw-sequence facts

The official NASA MAT operation structure records charge, discharge, and impedance operations in top-level chronological order. Charge operation data have no `Capacity` field; discharge operation data have a `Capacity` field. `CHARGE_CAPACITY_NONE_IS_EXPECTED=True`.

Only operation type, record order, timestamp-field presence, and field presence were inspected. No Capacity, HF, SOH, voltage, current, or performance value was read or emitted.

## Batteries audited

| Battery | Records | Charge | Discharge with Capacity field | Impedance | Operation-pattern conclusion |
|---|---:|---:|---:|---:|---|
| B0005 | 616 | 170 | 168 | 278 | Charge/discharge alternation is interrupted by impedance and occasional consecutive charge operations. |
| B0006 | 616 | 170 | 168 | 278 | Same structural pattern as B0005. |
| B0007 | 616 | 170 | 168 | 278 | Same structural pattern as B0005. |
| B0018 | 319 | 134 | 132 | 53 | Similar pattern, with fewer impedance records. |

## Candidate rule results

| Candidate | Structural result across audited batteries | Evidence class |
|---|---|---|
| A: charge → immediately following discharge | One-to-one whenever present, but yields 31/170 pairs for B0005–B0007 and 109/134 for B0018; it leaves many charge operations unpaired. | RAW_SEQUENCE_STRUCTURAL_EVIDENCE |
| B: charge → nearest subsequent discharge, skipping impedance | Covers 169/170 charges for B0005–B0007 and 134/134 for B0018, but creates two many-to-one collisions in every battery because consecutive charges can precede one discharge. | RAW_SEQUENCE_STRUCTURAL_EVIDENCE |
| C: discharge → immediately preceding charge | Equivalent to A on this sequence; same incompleteness. | RAW_SEQUENCE_STRUCTURAL_EVIDENCE |
| D: explicit shared cycle/session identity | Not available in the top-level raw operation metadata. | OFFICIAL_DATA_FORMAT_FACT |

All valid candidate pairs preserve operation order (charge record index < discharge record index). Impedance can occur between candidate members. Edge cases and missing immediate matches occur at both internal sequence positions and lifecycle boundaries.

## Decision

`ONE_TO_ONE_MAPPING_AVAILABLE=False`  
`AMBIGUOUS_PAIR_COUNT=2_PER_BATTERY_FOR_CANDIDATE_B`  
`PAIRING_EVIDENCE_STRENGTH=INSUFFICIENT_FOR_UNIQUE_MAPPING`  
`PAIRING_AUDIT_GATE=BLOCKED__NO_STABLE_ONE_TO_ONE_MAPPING`

The natural future supervised identity may be a charge–discharge pair, and if that identity is later authorized the logical split unit should be `PAIRED_SUPERVISED_CYCLE`, not raw operation records. This is an engineering inference, not an applied protocol change. No pairing winner, split-unit change, or training was made here.

NEXT_SCIENTIFIC_ACTION=ASK_ADVISOR_TO_SELECT_NASA_PAIRING_POLICY
