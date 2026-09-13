# PLAN-CHANGE-0012 — Advisor-Authorized Tier 2 Numerical Protocol

STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_FALLBACK_PROTOCOL_FREEZE__PREPARATION_ONLY  
LEVEL=2__NUMERICAL_FIDELITY_FALLBACK  
HUMAN_AUTHORIZATION=True  
PAPER_EXACT=False

## Scope

This change ends the wait-only state for preparation of a disclosed Tier 2 numerical path. It does not claim author confirmation, alter the research goal, replace the Tier 1 framework contracts, or authorize formal Phase 9 training.

CORE_ARCHITECTURE_CHANGED=False  
DATA_PROTOCOL_CHANGED=False  
RESEARCH_GOAL_CHANGED=False  
RESULT_DRIVEN=False  
PERFORMANCE_DRIVEN=False  
TEST_LOADED=False

## Frozen policy

The active preparation protocol is [`ADVISOR_AUTHORIZED_NUMERICAL_PROTOCOL_V1.md`](ADVISOR_AUTHORIZED_NUMERICAL_PROTOCOL_V1.md). It freezes sequence length 10, one-step horizon, STL period 7, TRAIN-only 0–1 normalization, hidden width 64, weighted fusion 0.7/0.2/0.1, and MSE. The HF_INPUT_TO_COMPONENT_TARGET_MODE from PLAN-CHANGE-0010 remains active. No 9-feature error/ratio builder is introduced.

## Gates

`ADVANCEMENT_TO_PHASE_9_PREPARATION=True`  
`FORMAL_PHASE9_TRAINING_STARTED=False`  
`TEST_LOCKED=True`  
`TIER3_PAPER_EXACT_STATUS=BLOCKED_PENDING_PRIMARY_SOURCE`

This change preserves all existing fidelity blockers, including the label-leaking current-target error finding. A future author response still requires `PRIMARY_SOURCE_RECONCILIATION_AUDIT` and does not automatically override this provenance.
