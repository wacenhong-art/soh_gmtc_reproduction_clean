# CALCE Column Null and Duplicate Audit

**Scope:** every row of every worksheet in 103 official CALCE XLSX workbooks (1,133,878 rows including Info sheets). Raw data were read only.

## Audit definitions

`EXACT_ROW_DUPLICATE_KEY` is the normalized sequence of every populated raw cell in a source row.
`CONTENT_DUPLICATE_KEY` contains: `Step_Index`, `Cycle_Index`, `Current(A)`, `Voltage(V)`, charge/discharge capacity and energy, `dV/dt(V/s)`, `Is_FC_Data`, AC impedance, and phase angle. It excludes `Data_Point`, `Test_Time(s)`, and `Date_Time` because those identify position/time rather than the electrochemical/step state. The complete machine-readable column-level counts are in [calce_null_duplicate_audit.json](C:/Users/ROG/Documents/Python/new_soh/output/data_readiness/calce_null_duplicate_audit.json).

| Battery | All-sheet rows | Exact row duplicates | Content duplicates | Same Cycle_Index, different content | Same timestamp, different content | Duplicate canonical source keys |
|---|---:|---:|---:|---:|---:|---:|
| CS2-35 | 267,641 | 0 | 120 | 265,933 | 2,216 | 0 |
| CS2-36 | 270,369 | 0 | 114 | 268,644 | 2,317 | 0 |
| CS2-37 | 291,763 | 0 | 127 | 289,909 | 2,526 | 0 |
| CS2-38 | 304,105 | 0 | 126 | 302,171 | 2,590 | 0 |
| **Total** | **1,133,878** | **0** | **487** | **1,126,657** | **9,649** | **0** |

The large “same Cycle_Index, different content” count is not treated as duplicated data: measurement rows within a reported cycle normally differ in voltage, current, capacity, or time. This count demonstrates why Cycle_Index cannot stand alone as a row or segment identity.

## Null and temperature finding

Every workbook/sheet/column has an explicit `TOTAL_ROWS`, `NULL_COUNT`, `NULL_RATE`, empty-string, nonnumeric, infinity, and NaN count in [calce_null_duplicate_audit_v3.json](C:/Users/ROG/Documents/Python/new_soh/output/data_readiness/calce_null_duplicate_audit_v3.json). The original full-row audit remains preserved alongside it. The only nonzero null pattern observed is one `TEST REPORT` null in each four-row `Info` sheet; this is metadata, not an unreported missing measurement in the operational data schema.

No worksheet exposes a temperature field. Therefore:

`CALCE_TEMPERATURE_SIGNAL_STATUS=MISSING`
`CALCE_FEATURE_CONSTRUCTION_FEASIBILITY=PASS_WITH_RECORDED_FEATURE_LIMITATION`

Time, voltage, current, and raw capacity are available. Paper HF1/HF2/HF3 are mechanically constructible from their required fields after later protocol/target authorization; any paper feature requiring temperature is not directly reconstructible from the official CALCE files currently in scope.
