# Tier 2 NASA HF endpoint protocol V1

`PROTOCOL_ID=TIER2_NASA_HF_ENDPOINT_V1`
`PLAN_CHANGE=PLAN-CHANGE-0017_REVISED`
`PROVENANCE=HUMAN_AUTHORIZED__RAW_TRAIN_EVIDENCE_SUPPORTED__RESULT_BLIND`
`PAPER_EXACT=False`

## Formula precedence

The target-paper prose names HF1/HF2 inconsistently with Eqs.(1)–(2). Tier 2
uses the equations, preserves the contradiction, and never swaps names:

```text
HF1=t_Vmax-t_0
HF2=t_Imin-t_Vmax
HF3=t_4p0-t_3p8
```

## NASA field and endpoint map

| Symbol | Field / rule | Status |
|---|---|---|
| `t_0` | first `Time` sample in the paired charge operation | frozen |
| `t_Vmax` | first `Voltage_measured >= 4.2` sample | frozen |
| `t_3p8` | first `Voltage_measured >= 3.8` sample | frozen |
| `t_4p0` | first `Voltage_measured >= 4.0` sample | frozen |
| `t_Imin` | first `Current_charge <= 0.02 A` sample strictly after `t_Vmax`; otherwise the last valid `Time` sample in the same charge operation | frozen, fallback-labelled |

`NASA_CANONICAL_VOLTAGE_FIELD=Voltage_measured`
`T_IMIN_SIGN_CONVENTION=RAW_CHARGER_CURRENT_SIGN__NO_ABS_OR_SIGN_INVERSION`
`VOLTAGE_ENDPOINT_INTERPOLATION=NONE__FIRST_SAMPLED_THRESHOLD_REACH`
`T_IMIN_INTERPOLATION=NONE__FIRST_SAMPLED_THRESHOLD_REACH`

The extractor fails closed for empty/nonfinite fields, unequal lengths,
non-increasing time, missing voltage threshold crossings, or negative/nonfinite
durations. It does not drop, copy, fabricate, or capacity/SOH-impute features.

## Coverage boundary

Only legal paired TRAIN charge operations may be inspected for the coverage
test. Raw MAT full parsing is authorized, but no validation or TEST trajectory
may be dereferenced after pair-first splitting. A `t_Imin` final-timestamp
fallback remains a valid frozen Tier 2 endpoint and is counted by battery.
