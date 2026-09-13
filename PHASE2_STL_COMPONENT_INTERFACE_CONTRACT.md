# Phase 2 STL Component Interface Contract

This is an interface contract, not a model implementation.

## Input

One already-isolated, chronologically ordered capacity sequence with explicit `period`, `partition_id`, and `series_id`.

## Output schema

- `trend`: float component array; future TimesNet input.
- `seasonal`: float component array; future TimesNet input.
- `residual`: float component array; future iTransformer input.
- `metadata`: period, input length, partition ID, series ID, STL library/version, resolved parameter configuration, and offline/non-causal execution status.

No output includes a prediction, training loss, SOH metric, split assignment, or model state.

## Partition and leakage contract

Only one declared partition may be supplied. Combined partitions, full lifecycle, and TEST are rejected. This prevents automatic cross-partition composition but does not make offline STL causal within an allowed partition.

INTER_PARTITION_LEAKAGE=False for a conforming invocation.

INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE=True for standard offline STL.

## Phase 7B real-data interface reconciliation — PLAN-CHANGE-0010

`PHASE2_REAL_INTERFACE_ROLE_REVISED=True`

`STRUCTURAL_MODULE_INVALIDATED=False`

The pre-existing synthetic structural interface remains valid only as a software test. For the authorized future real-data fallback, `OBSERVED_TRAIN_CAPACITY → STL → {TREND, SEASONAL, RESIDUAL}` is the **target path**. The components are supervision targets for future branch models; they are not the default real model inputs.

`REAL_STL_INPUT_SOURCE=OBSERVED_TRAIN_CAPACITY`

`REAL_STL_COMPONENT_ROLE=BRANCH_SUPERVISION_TARGETS`

`STL_REAL_BATTERY_PERIOD_MISSING=True`

No real STL run is authorized by this clarification.
