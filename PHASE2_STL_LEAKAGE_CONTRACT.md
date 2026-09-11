# Phase 2 STL Leakage Contract

**Status:** frozen contract; no STL decomposition has been executed.

## Two distinct risks

| Leakage level | Definition | Current policy | Status |
|---|---|---|---|
| `INTER_PARTITION_LEAKAGE` | A train component uses validation/test observations, or any fit derives statistics from a later partition. | Full-lifecycle decomposition before split is prohibited unless paper-exact evidence and a dedicated audit justify it. | `PROHIBITED` |
| `INTRA_TRAIN_FUTURE_CONTEXT` | A component at train time `t` uses later train observations through symmetric LOESS/STL smoothing. | Must be reported separately; restricting input to the train partition does not make STL causal. | `UNRESOLVED__IMPLEMENTATION_NOT_SELECTED` |

`FULL_LIFECYCLE_STL_BEFORE_SPLIT=PROHIBITED_UNLESS_PAPER_EXACTLY_REQUIRES_AND_LEAKAGE_AUDITED`

`TEST_LOADED=False`

`TEST_TARGET_ACCESSED_FOR_ANALYSIS=False`

`TEST_FEATURES_USED_FOR_FIT=False`

## Paper versus leakage-safe modes

`PAPER_FAITHFUL_STL_MODE=UNRESOLVED__PAPER_DOES_NOT_STATE_FIT_SCOPE_OR_BOUNDARY_POLICY`

`LEAKAGE_SAFE_STL_MODE=UNAUTHORIZED_PENDING_HUMAN_DECISION`

If the paper later proves it uses full-life, noncausal STL, that mode must be preserved separately as paper-faithful rather than silently converted into a causal filter. Conversely, a train-partition-only or rolling mode would be a leakage-safe engineering fallback, not a paper-exact reconstruction.

## Split order and component labels

`STL_SPLIT_ORDER_STATUS=UNRESOLVED`

The paper states chronological partitions and defines STL, but does not state whether decomposition happens before split, after split, per battery, per partition, or after windowing.

`COMPONENT_TARGET_CONSTRUCTION_STATUS=UNRESOLVED_BLOCKING`

The paper does not say how validation/test targets for trend, seasonal, and residual predictions are constructed. Candidate mechanisms—partition-wise independent STL, full-life STL, rolling/expanding decomposition, or final-SOH-only supervision—are not selected. Therefore no validation/test components may be constructed in Phase 2A.

## Preconditions for a minimal deterministic module

No module is authorized until all of the following are true:

```text
STL_INPUT_SEMANTICS_SUFFICIENTLY_FROZEN=True
STL_PARAMETER_POLICY_FROZEN=True
STL_SPLIT_POLICY_FROZEN=True
COMPONENT_TARGET_CONSTRUCTION_POLICY_FROZEN=True
NO_TEST_ACCESS=True
HUMAN_AUTHORIZATION_FOR_ANY_ENGINEERING_FALLBACK=True
```

The future module contract must test additive reconstruction, length preservation, finite outputs, chronological-order preservation, partition boundaries, deterministic repeatability, and absence of test-file access.
