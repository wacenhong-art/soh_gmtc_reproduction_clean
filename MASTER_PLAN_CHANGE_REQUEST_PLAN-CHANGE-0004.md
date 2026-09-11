# MASTER PLAN CHANGE REQUEST — PLAN-CHANGE-0004

`STATUS=APPLIED__HUMAN_AUTHORIZED`

`TRIGGER=Phase 1 exact-sample-manifest executability audit`

`CORE_ASSUMPTION_INVALIDATED=True`

`PROTOCOL_INVALIDATED=True`

`RESULT_DRIVEN=False`

`MODEL_RESULTS_SEEN=False`

`TEST_LOADED=False`

## New evidence

Protocol B defines only the first 20% of total life as its original small-sample training population. The paper does not define Protocol B validation/test populations or state that §4’s general 60/20/20 split is inherited by §4.2.

## Impact

The existing fallback can define a prospective Protocol B training construction, but cannot produce a valid, leakage-audited Protocol B evaluation manifest. The Phase 1 Gate therefore cannot pass. No row assignment has been made.

## Human-authorized decision

`HUMAN_AUTHORIZED=True`

`PLAN_DEVIATION_LEVEL=2`

`PROTOCOL_B_EVALUATION_FALLBACK=NESTED_SMALL_SAMPLE_WITH_GENERAL_HOLDOUT_BOUNDARIES`

For each target battery, using zero-based half-open intervals and result-blind deterministic rounding:

```text
TRAIN_REAL        = [0, floor(0.20*N))
UNUSED_PROTOCOL_B = [floor(0.20*N), floor(0.60*N))
VALIDATION        = [floor(0.60*N), floor(0.80*N))
TEST              = [floor(0.80*N), N)
```

**Rationale:** retain the paper’s stated general 60%/80% holdout boundaries while restricting Protocol B real training to the stated first 20% and retaining the intermediate records as visibly unused. The decision is `RESULT_DRIVEN=False`, `PERFORMANCE_DRIVEN=False`, and `TEST_LOADED=False`. It is a Human fallback, not a paper-exact claim; the active primary-source override policy remains controlling if the author replies.
