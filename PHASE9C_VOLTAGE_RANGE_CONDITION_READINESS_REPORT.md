# Phase 9C voltage-range condition readiness audit

## Scope controls

```text
SOURCE_BATTERIES=B0005,B0006
MODEL_TRAINED=False
SYNTHETIC_HF_GENERATED=False
TEST_LOADED=False
TEST_ACCESS_COUNT=0
PLAN_CHANGE_0018_MODIFIED=False
```

## Source trace coverage

| Battery | legal paired charges | covers [3.6,4.2] V | covers [3.8,4.0] V |
| --- | ---: | ---: | ---: |
| B0005 | 168 | 167 | 167 |
| B0006 | 168 | 167 | 168 |

## Frozen HF voltage provenance

| HF | frozen endpoints / prerequisite | computable from [3.8,4.0] only |
| --- | --- | --- |
| HF1 | start=NOT_FIXED__CHARGE_OPERATION_START_t0; end=4.2 | False |
| HF2 | start=4.2; end=NOT_FIXED__CURRENT_CHARGE_CUTOFF_OR_FINAL_TIMESTAMP | False |
| HF3 | start=3.8; end=4.0 | True |

## Condition evidence and readiness

```text
PAPER_CONDITION_EVIDENCE:
Eq5=CYCLE_PERIOD
Fig3=CYCLE_PLUS_BATTERY_TYPE
METHODOLOGY_PROSE=VOLTAGE_RANGE
Fig2_VOLTAGE_RANGE_REPRESENTATION=START_VOLTAGE_PLUS_END_VOLTAGE
PAPER_CONDITION_SEMANTICS_CONTRADICTION=True
SOURCE_MULTI_RANGE_CONDITION_AVAILABLE=False
HF1_COMPUTABLE_FROM_3P8_4P0_ONLY=False
HF2_COMPUTABLE_FROM_3P8_4P0_ONLY=False
HF3_COMPUTABLE_FROM_3P8_4P0_ONLY=True
VOLTAGE_RANGE_CONDITION_IMPLEMENTATION_READY=False
```

## Missing scientific definitions

- An author- or protocol-supported rule that maps NASA raw trace voltage windows or observed endpoints to the SCF-GAN condition vector c=[V_start,V_end].
- A source-training sample policy defining which multiple voltage ranges are presented for a single charge cycle and how their labels/feature vectors are paired.
- A paper-supported replacement or restriction rule for HF1 and HF2 when only [3.8,4.0] V observations are available; their frozen equations require 4.2-V and/or post-4.2-V context.
- An executable reconciliation of Eq. (5) cycle-period, Fig. 3 Cycle + Battery Type, and methodology voltage-range condition semantics.

The full JSON manifest records observed start/end/min/max and both coverage flags for every legal source charge. No new voltage-window HF formula or condition encoding was created.
