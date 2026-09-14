# NASA Pairing Protocol V1

PROTOCOL_ID=NASA_PAIRING_TIER2_V1
PLAN_CHANGE=PLAN-CHANGE-0015
PROVENANCE=WEB_LITERATURE_SUPPORTED_RESULT_BLIND_ENGINEERING_FALLBACK
PAPER_EXACT=False
PAPER_EXACT_PAIRING_STATUS=UNRESOLVED
TEST_LOCKED=True

## Frozen pairing contract

```text
PAIRING_MODE=DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE
ONE_CHARGE_MAX_ONE_DISCHARGE=True
ONE_DISCHARGE_MAX_ONE_CHARGE=True
NASA_SUPERVISED_SAMPLE_IDENTITY=VALID_CHARGE_DISCHARGE_PAIR
INPUT_SOURCE=PAIRED_CHARGE_OPERATION__HF1_HF2_HF3
TARGET_SOURCE=PAIRED_DISCHARGE_OPERATION__CAPACITY
PAIR_METADATA=battery_id,pair_index,charge_record_index,discharge_record_index
IMPEDANCE_BETWEEN_PAIR_MEMBERS=ALLOWED
SPLIT_UNIT=VALID_PAIRED_SUPERVISED_CYCLE
SPLIT_ORDER=PAIR_FIRST__THEN_CHRONOLOGICAL_70_20_10
```

For each discharge in original operation order, choose the nearest preceding eligible charge that is not consumed. A later discharge cannot reuse that charge. If several charges occur consecutively before one discharge, the nearest eligible one is used and the others remain unpaired. No numerical field participates in this decision.

## Structural count contract

| Battery | Expected valid pairs | Expected unpaired charges |
|---|---:|---:|
| B0005 | 168 | 2 |
| B0006 | 168 | 2 |
| B0007 | 168 | 2 |
| B0018 | 132 | 2 |

Before future numeric access, raw operation metadata must reproduce these counts and verify charge index < discharge index, no charge reuse, no discharge reuse, and permitted impedance-between cases. A mismatch is `FAIL_CLOSED`; no ad hoc repair or battery substitution is authorized.

## Split and lock

Only after valid pairs are created, their chronological pair order is split into first 70% TRAIN, next 20% VALIDATION, and final 10% TEST_LOCKED by the frozen half-open floor rule. The final partition's values must not be materialized during Phase 9A.

## Evidence boundary

`PAIRING_PROVENANCE=NASA_OFFICIAL_FORMAT__MATHWORKS_NASA_WORKFLOW__PUBLISHED_PREPROCESSING_PRECEDENT__RAW_SEQUENCE_STRUCTURAL_EVIDENCE`
`PAIRING_EVIDENCE_STRENGTH=STRONG_FOR_TIER2_ENGINEERING_REPRODUCTION`

External records justify data-format interpretation and incomplete-cycle removal as Tier 2 engineering practice. They do not establish that the target Energy paper used this exact code-level rule.
