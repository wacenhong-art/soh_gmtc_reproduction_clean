# Phase 1 Exact Sample Manifest

`REAL_SAMPLE_MANIFEST_CREATED=True`

`ROW_ASSIGNMENT_EXECUTED=True__METADATA_ONLY`

`NEW_PHASE1_BLOCKER_FOUND=False`

`TEST_MEMBERSHIP_DEFINED=True`

`TEST_LOADED=False`

The deterministic body is local-only, contains source/order metadata rather than raw measurements, and is hashed by the versioned summary: `output/protocol/phase1_membership_manifest.csv`. It has **4,724** rows: the four E1 target-battery operational populations, each represented once for Protocol A and once for Protocol B. No health-feature value, raw capacity, SOH label, synthetic row, or model input is materialized.

## Identity and ordering

NASA identities combine dataset, battery, canonical raw source, original operation record, reported cycle index, and zero-based chronological index. CALCE identities use the audited `CANONICAL_CALCE_SOURCE_KEY`, which contains battery, workbook, sheet, source segment and reported cycle index; raw CALCE `Cycle_Index` is never treated as global. CALCE membership uses only audited `Channel_*` contiguous source segments, sorted by source timestamps.

`ROUNDING_RULE=zero_based_half_open__TRAIN20_END=floor(0.20*N)__TRAIN60_END=floor(0.60*N)__VALID80_END=floor(0.80*N)`

`ROUNDING_RULE_PROVENANCE=HUMAN_ENGINEERING_FALLBACK__RESULT_BLIND`

## Population map

| Dataset / target battery | N | Protocol A: Train / Validation / Test | Protocol B: Train real / Unused / Validation / Test |
|---|---:|---:|---:|
| NASA B0007 | 170 | 102 / 34 / 34 | 34 / 68 / 34 / 34 |
| NASA B0018 | 134 | 80 / 27 / 27 | 26 / 54 / 27 / 27 |
| CALCE CS2_36 | 976 | 585 / 195 / 196 | 195 / 390 / 195 / 196 |
| CALCE CS2_38 | 1,082 | 649 / 216 / 217 | 216 / 433 / 216 / 217 |

Every raw metadata record is conserved in exactly one role within each protocol. Protocol A and B are isolated by distinct `active_protocol` and `experiment_id`; reappearance of the same raw source record across these two separately declared tracks is reported as cross-track identity sharing, not pooled training or protocol contamination.

`PROTOCOL_A_ASSIGNMENT_STATUS=VALID`

`PROTOCOL_B_ASSIGNMENT_STATUS=VALID`

`PHASE_1_GATE=PASS__PENDING_ARCHIVE_VERIFICATION`
