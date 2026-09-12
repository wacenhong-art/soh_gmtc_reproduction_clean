# PLAN-CHANGE-0006: iTransformer Structural and Real-Data Dual Gates

PLAN_CHANGE_ID=PLAN-CHANGE-0006

NAME=SEPARATE_ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_FROM_REAL_DATA_FIDELITY

LEVEL=1

STATUS=APPLIED__USER_AUTHORIZED

TRIGGER=PHASE_3A_SPECIFICATION_AND_DATAFLOW_AUDIT

CORE_ARCHITECTURE_CHANGED=False

DATA_PROTOCOL_CHANGED=False

RESEARCH_GOAL_CHANGED=False

RESULT_DRIVEN=False

PERFORMANCE_DRIVEN=False

TEST_LOADED=False

## Amendment

`ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE` may assess synthetic-only tensor routing, strict 10-step input contract, forward/backward wiring, determinism, and fail-closed provenance behavior only after a target-alignment contract is authorized.

`ITRANSFORMER_REAL_DATA_FIDELITY_GATE` may pass only when real residual construction, paper-defensible input semantics, target/horizon alignment, and architecture parameters are frozen. It remains blocked in the current project state.

No structural success may remove `STL_REAL_BATTERY_PERIOD_MISSING` or any iTransformer ambiguity from numerical-reproduction blockers.
