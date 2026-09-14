# Phase 9A Paired Chronological Split Manifest

PAIRING_MODE=DISCHARGE_CENTERED_NEAREST_PRECEDING_UNCONSUMED_CHARGE  
SPLIT_UNIT=VALID_PAIRED_SUPERVISED_CYCLE  
SPLIT_MODE=CHRONOLOGICAL__HALF_OPEN__ZERO_BASED_FLOOR  
TEST_NUMERICAL_VALUES_MATERIALIZED=False

| Battery | Valid pairs | TRAIN `[start,end)` | VALIDATION `[start,end)` | TEST membership only `[start,end)` |
|---|---:|---|---|---|
| B0005 | 168 | `[0,117)` | `[117,151)` | `[151,168)` |
| B0006 | 168 | `[0,117)` | `[117,151)` | `[151,168)` |
| B0007 | 168 | `[0,117)` | `[117,151)` | `[151,168)` |
| B0018 | 132 | `[0,92)` | `[92,118)` | `[118,132)` |

For every battery, the three half-open ranges are disjoint and ordered TRAIN < VALIDATION < TEST. This manifest records pairing and membership metadata only; it contains no HF, Capacity, SOH, target, prediction, or metric value.

Phase 1 E1 target-execution roster remains B0007 and B0018. B0005 and B0006 were revalidated for their frozen pairing/split structure but were not selected as substitute numerical execution targets.

## Executed-run confirmation — 2026-09-14

The real TRAIN+VALIDATION dry-run revalidated the full four-battery pairing
structure before numerical access and then executed only B0007 and B0018, the
frozen E1 target roster. The actual numerical membership remained exactly the
ranges above. TEST is represented only by its pair indices and counts; no TEST
charge/discharge numerical field was dereferenced after the split boundary.
