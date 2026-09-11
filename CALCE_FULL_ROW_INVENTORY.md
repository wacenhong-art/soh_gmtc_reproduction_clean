# CALCE Full-row Deterministic Inventory

**Task:** `COMPLETE_REMOTE_ARCHIVE_THEN_CALCE_FULL_ROW_INVENTORY`  
**Raw source:** official CALCE archives listed and hashed in `DATA_SOURCE_MANIFEST.md`.  
**Scope:** read-only raw-file scanning. No raw file was modified, moved, renamed, or deleted; no split, target denominator, model, STL, GAN, metric, or search operation occurred.

## File map

| BATTERY_ID | ARCHIVE | SIZE_BYTES | SHA256 | SOURCE_MANIFEST_MATCH | FILE_READABLE |
|---|---|---:|---|---|---|
| CS2_35 | `raw_data/calce_cs2/CS2_35.zip` | 37056519 | `84D88CB98A43C02049F16C78806A78EC19084776EA5DF8A4B7C2F8F7BA4589E5` | True | True |
| CS2_36 | `raw_data/calce_cs2/CS2_36.zip` | 37881685 | `2E674F58BB6DC005C73BEF10E87EDBBD97AFD041DDE77A1D461A3C4ECAC1B027` | True | True |
| CS2_37 | `raw_data/calce_cs2/CS2_37.zip` | 41513048 | `8ED8AECBCBEA2BCB17F455923A974BE01EA21D7389210E1912D177DE5B6AF4A5` | True | True |
| CS2_38 | `raw_data/calce_cs2/CS2_38.zip` | 42811100 | `98787427C10478425EFD4F54D823805724DAFF1D2FCBD822B3A2D7006A1051D8` | True | True |

## Deterministic full-row result

The parser read every `sheet2` data row in every official workbook. The source schema provides `Data_Point`, `Test_Time(s)`, `Date_Time`, `Step_Time(s)`, `Step_Index`, `Cycle_Index`, `Current(A)`, `Voltage(V)`, `Charge_Capacity(Ah)`, `Charge_Energy(Wh)`, `Discharge_Capacity(Ah)`, `Discharge_Energy(Wh)`, `dV/dt(V/s)`, `Is_FC_Data`, `AC_Impedance(Ohm)`, and `ACI_Phase_Angle(Deg)`.

| BATTERY_ID | XLSX_FILES | TOTAL_ROWS | CHARGE_CYCLE_CANDIDATES | DISCHARGE_CYCLE_CANDIDATES | FEATURE_COMPLETE_CANDIDATES | MALFORMED_ROWS | DUPLICATE_RECORD_IDS | TIMESTAMP_STATUS |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CS2_35 | 25 | 266914 | 50 | 50 | 50 | 0 | 0 | `NON_MONOTONIC_OR_NO_RECORDS` in raw workbook traversal order |
| CS2_36 | 26 | 269656 | 50 | 50 | 50 | 0 | 0 | `NON_MONOTONIC_OR_NO_RECORDS` in raw workbook traversal order |
| CS2_37 | 27 | 290998 | 50 | 50 | 49 | 0 | 0 | `NON_MONOTONIC_OR_NO_RECORDS` in raw workbook traversal order |
| CS2_38 | 28 | 303295 | 50 | 50 | 50 | 0 | 0 | `NON_MONOTONIC_OR_NO_RECORDS` in raw workbook traversal order |

`CALCE_FULL_ROW_SCAN_STATUS=COMPLETE`

`CALCE_CYCLE_INVENTORY_STATUS=PARTIAL__CYCLE_INDEX_NUMBERS_WERE_OBSERVED_BUT_CROSS_WORKBOOK_GLOBAL_UNIQUENESS_AND_BOUNDARIES_ARE_NOT_YET_PROVEN`

`UNRESOLVED_SEGMENTS=All source-workbook boundaries whose numeric Cycle_Index may repeat are retained as unresolved rather than silently merged.`

`ROW_LEVEL_NULL_COUNT_AND_CONTENT_DUPLICATE_AUDIT=NOT_YET_PERSISTED_AS_A_SEPARATE_ARTIFACT`

The last line is a reporting gap, not permission to infer zero missing values or zero content duplicates. It blocks the split-independent readiness gate but does not invalidate the paper or select a PA-004 interpretation.

## Signal availability and feature constructability

| Signal / feature | CALCE status | Evidence / boundary |
|---|---|---|
| Time | AVAILABLE | `Test_Time(s)` and `Date_Time` are present in every inspected workbook schema. |
| Voltage | AVAILABLE | `Voltage(V)` is present. |
| Current | AVAILABLE | `Current(A)` is present. |
| Capacity | AVAILABLE | `Discharge_Capacity(Ah)` is present; this is raw capacity only, not an SOH label. |
| Temperature | MISSING_FROM_DATA_SHEET | No per-row temperature column occurs in the operational `sheet2` schema. |
| HF1 | PARTIAL | Required time/current/voltage fields exist; 50/50/49/50 candidate cycles met current parser boundaries for CS2_35/36/37/38. |
| HF2 | PARTIAL | Required time/current/voltage fields exist; exact source-safe cross-workbook cycle boundary audit remains pending. |
| HF3 | PARTIAL | Required time/voltage fields exist; exact source-safe cross-workbook cycle boundary audit remains pending. |

`FINAL_SOH_TARGET_ASSIGNED=False`

`TARGET_CONSTRUCTION_STATUS=BLOCKED_BY_PA001_SOH_DENOMINATOR_AMBIGUITY`

`TRAIN_SET_ASSIGNED=False`

`VALIDATION_SET_ASSIGNED=False`

`TEST_SET_ASSIGNED=False`

`PA004_STATUS=UNRESOLVED_BLOCKING`
