# PLAN-CHANGE-0019 — SCF-GAN Figure 3 paper-closest reconstruction

**Status:** APPLIED — Human-authorized, result-blind Figure-3-priority Tier 2 path.
**Scope:** a new, separate SCF-GAN reconstruction. PLAN-CHANGE-0018 and all of
its three-HF artifacts remain intact.

## Frozen Figure 3 execution contract

```text
CONDITION_MODE=CYCLE_PLUS_BATTERY_TYPE
CONDITION_DIM=2
c1=NORMALIZED_CYCLE_POSITION_0_TO_1
c2=NASA_TABLE1_BATTERY_TYPE_CONSTANT_0P0
NASA_BATTERY_TYPE_CONDITION_IS_CONSTANT=True

OUTPUT_MODE=HF1_HF2_HF3_SOH
SOH=CAPACITY_AH_DIVIDED_BY_2P0

GENERATOR=18_TO_256_TO_512_TO_256_TO_4__RELU__CONDITIONAL_BATCH_NORM__TANH__NO_SN
DISCRIMINATOR=6_TO_256_TO_512_TO_256_TO_1__LEAKY_RELU_0P2__DROPOUT_0P3__SN_HIDDEN_LINEAR__RAW_LOGIT
LOSS=BCE_WITH_LOGITS_PLUS_0P1_FEATURE_MATCHING_AT_THIRD_HIDDEN_LAYER
```

The constant battery-type encoding is a single-class baseline value, not a
battery ID, target-selected value, or paper-exact category coding.

## Evidence boundary

```text
PAPER_EXACT_STATUS=UNRESOLVED_INTERNAL_CONTRADICTION
VOLTAGE_RANGE_DIRECT_IMPLEMENTATION=BLOCKED_BY_HF1_HF2_PARTIAL_WINDOW_UNDEFINED
TABLE3_DISCRIMINATOR_DEPTH_CONTRADICTION=True
FIGURE3_THREE_HIDDEN_LAYER_EXECUTION=True
```

Figure 3 is prioritized only for this new execution path. Its Cycle + Battery
Type condition and four-variable output do not delete the prior Table/prose/
equation ambiguity record.

## Source, scaler, training and target boundary

Only B0005/B0006 may provide training vectors. All four dimensions are scaled
from those source vectors alone to `[-1,1]`. B0007/B0018 numeric HF, capacity,
and SOH are prohibited from training and source-checkpoint selection.

Stage A is a 20-epoch contract/smoke test. Only after its pass may Stage B
create a fresh seed-42 initialization and run the fixed 300 source-only epochs
with twenty-epoch snapshots and source-only score selection. Target first-20%
cycle identities provide generation conditions only. Target HF/SOH can be read
only afterward for the disclosed post-hoc quality diagnostic.
