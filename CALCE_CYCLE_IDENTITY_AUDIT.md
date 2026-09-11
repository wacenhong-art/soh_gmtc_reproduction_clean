# CALCE Cycle Identity and Boundary Audit

**Task:** `CALCE_DATA_READINESS_FINAL_AUDIT__CYCLE_IDENTITY_NULL_AND_CONTENT_DUPLICATES`
**Scope:** read-only, split-independent audit of all official CALCE XLSX workbooks for CS2-35/36/37/38. No raw file changed; no SOH target, data split, model, training, metric, or search operation occurred.

## Canonical CALCE source key

`BATTERY_ID|SOURCE_WORKBOOK|SOURCE_SHEET|segment=<contiguous ordinal>|cycle=<reported Cycle_Index>`

The source key is unique because the segment ordinal is assigned within the original workbook and worksheet. It preserves the raw `Cycle_Index`; it does **not** reinterpret it as a global life-cycle identifier.

## Cross-workbook Cycle_Index finding

| Battery | Workbooks | Source-keyed contiguous segments | Repeated reported indexes within workbooks | Index overlap across workbooks | Globally unique? |
|---|---:|---:|---:|---|---|
| CS2-35 | 25 | 1,563 | 627 | True | False |
| CS2-36 | 26 | 1,585 | 609 | True | False |
| CS2-37 | 27 | 1,700 | 657 | True | False |
| CS2-38 | 28 | 1,780 | 698 | True | False |

`IS_CYCLE_INDEX_GLOBALLY_UNIQUE=False`

The raw workbooks repeatedly begin or reuse numeric `Cycle_Index` ranges (for example, CS2-35's first two workbooks each report 1–50). This is a raw-data identity fact, not an error to repair. Cross-workbook chronological ordering must use source provenance plus actual timestamp evidence in a later, protocol-authorized construction; filename order alone is not a lifecycle order contract.

## Boundary inventory rule and status

A boundary is **EXACT** only for the deterministic, source-local definition: a maximal contiguous run of raw rows with the same reported `Cycle_Index` in one workbook and one worksheet. It carries its raw start/end row and timestamps in [calce_cycle_boundary_audit_v3.json](C:/Users/ROG/Documents/Python/new_soh/output/data_readiness/calce_cycle_boundary_audit_v3.json). The corrected complete identity view is [calce_cycle_identity_audit_v3.json](C:/Users/ROG/Documents/Python/new_soh/output/data_readiness/calce_cycle_identity_audit_v3.json).

`CALCE_CYCLE_BOUNDARIES_AUDITED=True`
`BOUNDARY_STATUS=EXACT__CONTIGUOUS_REPORTED_CYCLE_INDEX_RUN`
`GLOBAL_LIFECYCLE_CYCLE_MAPPING=UNRESOLVED__NOT_NEEDED_FOR_SPLIT_INDEPENDENT_READINESS`

No capacity/SOH trend, future data, or model result was used to establish a boundary.

## Timestamp-order audit

`TIMESTAMP_MONOTONIC_WITHIN_SEGMENT=True` because every source-local segment inherits a monotonic raw workbook traversal. All 272 timestamp-bearing worksheet traversals were monotonic. Across workbooks, actual timestamp intervals can overlap (for example, a channel sheet and a statistics sheet can cover the same acquisition period), so a filename-sorted global order is not asserted.

`TIMESTAMP_MONOTONIC_WITHIN_WORKBOOK=True`
`CROSS_WORKBOOK_TEMPORAL_ORDER=AVAILABLE_BY_TIMESTAMP__OVERLAPPING_SOURCE_WINDOWS__NO_FILENAME_ORDER_ASSUMPTION`
