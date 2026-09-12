# PLAN-CHANGE-0006: Synthetic-Only Parameterized iTransformer Structural Implementation

PLAN_CHANGE_ID=PLAN-CHANGE-0006

NAME=ALLOW_SYNTHETIC_ONLY_PARAMETERIZED_ITRANSFORMER_STRUCTURAL_IMPLEMENTATION

LEVEL=1

STATUS=APPLIED__USER_AUTHORIZED__PHASE_3B_SCOPE_FROZEN

TRIGGER=PHASE_3A_SPECIFICATION_AND_DATAFLOW_AUDIT__PLUS_PHASE_3B_HUMAN_AUTHORIZATION

CORE_ARCHITECTURE_CHANGED=False

DATA_PROTOCOL_CHANGED=False

RESEARCH_GOAL_CHANGED=False

RESULT_DRIVEN=False

PERFORMANCE_DRIVEN=False

TEST_LOADED=False

## Amendment

`ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE` may assess synthetic-only tensor routing, strict 10-step input contract, forward/backward wiring, determinism, fail-closed provenance behavior, and a pre-registered generic synthetic target contract. The synthetic target is not an inference about target-paper horizon or real residual supervision.

`ITRANSFORMER_REAL_DATA_FIDELITY_GATE` may pass only when real residual construction, paper-defensible input semantics, target/horizon alignment, and architecture parameters are frozen. It remains blocked in the current project state.

No structural success may remove `STL_REAL_BATTERY_PERIOD_MISSING` or any iTransformer ambiguity from numerical-reproduction blockers.

`SYNTHETIC_TARGET_ALIGNMENT=ONE_STEP_SYNTHETIC_PROJECTION_ONLY`

`REAL_TARGET_ALIGNMENT_INFERENCE=False`
