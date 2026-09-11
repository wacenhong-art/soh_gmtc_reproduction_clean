# MASTER PLAN CHANGE REQUEST — PLAN-CHANGE-0004

`STATUS=PENDING_HUMAN_AUTHORIZATION`

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

## Minimum Human decision

Choose one result-blind, pre-registered option:

1. Define Protocol B validation/test population and its relation to Protocol A.
2. Define Protocol B as a training-only augmentation/generation study with no numerical-evaluation population, and retain Protocol A as the only evaluated SOH-estimation protocol.
3. Wait for primary-source author clarification.

No option is selected by Codex.

