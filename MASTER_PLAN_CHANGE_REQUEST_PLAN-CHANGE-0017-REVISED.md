# PLAN-CHANGE-0017_REVISED — Unified Tier 2 pre-numerical closure

`STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_UNIFIED_CLOSURE`
`SUPERSEDES_UNAPPLIED_0017_PROPOSAL=True`
`RESULT_DRIVEN_CHANGE=False`
`VALIDATION_VALUES_USED=False`
`TEST_VALUES_USED=False`
`MODEL_TRAINED=False`
`GRID_SEARCH_RUN=False`

## Authority and scope

This single change applies the Human-authorized rules supplied on 2026-09-14.
It closes the Phase 9A Tier 2 execution blockers without changing the paper
equations, pairing, split, STL period, branch cores, scalar adapter, or
component-sum fusion. It does not claim paper-exact implementation.

## Applied endpoint contract

```text
HF1 = t_Vmax - t_0
HF2 = t_Imin - t_Vmax
HF3 = t_4p0 - t_3p8

NASA_CANONICAL_VOLTAGE_FIELD=Voltage_measured
VOLTAGE_ENDPOINT_RULE=FIRST_SAMPLED_VALUE_AT_OR_ABOVE_THRESHOLD
T_IMIN_FIELD=Current_charge
T_IMIN_SIGN_CONVENTION=RAW_CHARGER_CURRENT_SIGN__NO_ABS_OR_SIGN_INVERSION
T_IMIN_THRESHOLD_A=0.02
T_IMIN_ENDPOINT_RULE=FIRST_RAW_CURRENT_CHARGE_LEQ_0P02_STRICTLY_AFTER_T_VMAX__ELSE_FINAL_VALID_TIMESTAMP
```

The final-timestamp condition is a Human-authorized fallback. Each occurrence
is preserved as `HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK`; no
capacity, SOH, correlation, validation metric, copying, or interpolation is
used to choose it.

`PAPER_HF1_HF2_NAMING_FORMULA_CONTRADICTION=True`
`PAPER_EQUATIONS_TAKE_PRECEDENCE_FOR_TIER2_IMPLEMENTATION=True`

## TEST boundary

`RAW_MAT_FULL_PARSE_ALLOWED=True`
`TEST_MEMBERSHIP_METADATA_ALLOWED=True`
`POST_SPLIT_TEST_NUMERICAL_FIELD_ACCESS_ALLOWED=False`

The revised human authorization distinguishes parser-level MAT deserialization
from post-split numerical use. After the pair-first chronological split, TEST
pairs are rejected before any charge/discharge record dereference for HF,
scaling, STL, DataLoader, model, prediction, or metric payload.

## Scale and run configuration

```text
TIER2_SOH_BRIDGE=PA001_ACTIVE__HUMAN_AUTHORIZED_FALLBACK
SOH=CAPACITY_AH/2.0
NASA_RATED_CAPACITY_AH=2.0

SEED=42
DRY_RUN_EPOCHS=3
LEARNING_RATE=0.001
BATCH_SIZE=32
OPTIMIZER=Adam
DEVICE=CPU
LOSS=MSE
TRAIN_BATCH_SHUFFLE=False
VALIDATION_BATCH_SHUFFLE=False
DROP_LAST=False
EARLY_STOPPING=False
CHECKPOINT_POLICY=DISABLED_FOR_DRY_RUN
```

## Impact mapping

| Blocker | Applied resolution | Code / contract | Verification |
|---|---|---|---|
| PNC-001 | explicit `t_Imin` sampled-current rule plus final-timestamp fallback | `tier2_hf_extraction.py` | endpoint + 443 TRAIN coverage test |
| PNC-002 | `Voltage_measured` for 3.8/4.0/4.2 endpoints | `tier2_hf_extraction.py`, Phase 1 inventory | endpoint + coverage test |
| PNC-003 | full parse authorized; post-split sealed accessor | `tier2_phase9a_access.py`, dry-run runner | pre-dereference TEST-lock test |
| PNC-004 | PA001 NASA capacity-to-SOH bridge activated | `tier2_soh_bridge.py` | synthetic formula test |
| PNC-005 | fixed CPU Adam dry-run configuration | `tier2_phase9a_config.py` | parser/config test |

## Non-effects

CALCE global identity and HF execution remain future-dataset issues and are not
required for the NASA-only Phase 9A dry run. SCF-GAN remains disabled. Tier 3
paper-exact feature-field, endpoint, architecture, fusion, and augmentation
questions remain recorded rather than erased.
