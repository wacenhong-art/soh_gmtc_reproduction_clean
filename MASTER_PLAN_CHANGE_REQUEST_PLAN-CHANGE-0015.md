# PLAN-CHANGE-0015: Freeze Web-Supported NASA Pairing Policy

PLAN_CHANGE_ID=PLAN-CHANGE-0015
STATUS=APPLIED__HUMAN_AUTHORIZED__WEB_LITERATURE_SUPPORTED_RESULT_BLIND_ENGINEERING_FALLBACK
TYPE=WEB_LITERATURE_SUPPORTED_RESULT_BLIND_ENGINEERING_FALLBACK
HUMAN_AUTHORIZATION=True
ADVISOR_CONFIRMED=False
PAPER_EXACT=False
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
REAL_HF_VALUES_USED=False
REAL_NUMERICAL_TARGET_VALUES_USED=False
MODEL_TRAINED=False
TEST_LOADED=False
GRID_SEARCH_RUN=False

## Reason and boundary

The prior metadata audits found a deterministic one-to-one Candidate E but correctly left it unapplied. The Human now authorizes a Tier 2 engineering fallback after review of NASA's official format, MathWorks' incomplete-cycle handling, published usable-cycle precedent, and the local four-battery structural audit. No feature, capacity, SOH, validation, test, or performance result was used to choose this policy.

This change does not assert the target Energy paper's exact implementation. `PAPER_EXACT_PAIRING_STATUS=UNRESOLVED` remains active.

## Frozen policy

`PAIRING_MODE=DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE`

For each discharge operation, search backward in original operation order for the nearest eligible charge not already consumed by another discharge. Impedance operations may occur between the paired records. When a consecutive-charge block precedes a discharge, retain only the nearest eligible charge; other charges that cannot form a unique normal pair are `UNPAIRED` and excluded. A capacity label is never copied to multiple charge operations.

`ONE_CHARGE_MAX_ONE_DISCHARGE=True`

`ONE_DISCHARGE_MAX_ONE_CHARGE=True`

`NASA_SUPERVISED_SAMPLE_IDENTITY=VALID_CHARGE_DISCHARGE_PAIR`
`SPLIT_UNIT=VALID_PAIRED_SUPERVISED_CYCLE`

The input HF vector comes from the paired charge operation; the capacity target comes from its paired discharge operation. Pairing precedes the chronological 70/20/10 split.

## Expected structural verification before numerical access

| Battery | Valid pairs | Unpaired charges |
|---|---:|---:|
| B0005 | 168 | 2 |
| B0006 | 168 | 2 |
| B0007 | 168 | 2 |
| B0018 | 132 | 2 |

Any mismatch during the future raw-structure recheck is fail-closed: numerical Phase 9A work must stop rather than repair counts heuristically.

## Provenance and forward authorization

`PAIRING_PROVENANCE=NASA_OFFICIAL_FORMAT__MATHWORKS_NASA_WORKFLOW__PUBLISHED_PREPROCESSING_PRECEDENT__RAW_SEQUENCE_STRUCTURAL_EVIDENCE`

`PAIRING_EVIDENCE_STRENGTH=STRONG_FOR_TIER2_ENGINEERING_REPRODUCTION`
`PAPER_EXACT_PAIRING_STATUS=UNRESOLVED`

This change authorizes preparation for, but does not execute, `PHASE_9A_FALLBACK_NUMERICAL_PIPELINE_DRY_RUN_RETRY`. TEST numerical values remain locked.
