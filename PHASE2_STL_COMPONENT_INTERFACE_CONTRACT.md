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
