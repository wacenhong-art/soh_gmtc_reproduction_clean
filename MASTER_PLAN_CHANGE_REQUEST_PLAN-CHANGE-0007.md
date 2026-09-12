# PLAN-CHANGE-0007 — synthetic-only G-MTC scaffold

`PLAN_CHANGE_ID=PLAN-CHANGE-0007`

`NAME=ALLOW_SYNTHETIC_ONLY_GMTC_FRAMEWORK_SCAFFOLD_WHILE_REAL_DATA_FIDELITY_REMAINS_BLOCKED`

`LEVEL=1`

`STATUS=APPLIED__USER_AUTHORIZED__PHASE_4A_SCOPE_FROZEN`

## Scope

The Human Researcher authorized a narrow structural side-track: define and test the G-MTC component-routing, branch-prediction, alignment, and future-fusion payload interfaces using deterministic synthetic fixtures only. This is interface engineering, not a new scientific method.

`CORE_ARCHITECTURE_CHANGED=False`

`DATA_PROTOCOL_CHANGED=False`

`RESEARCH_GOAL_CHANGED=False`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`TEST_LOADED=False`

## Permitted work

- synthetic STL-component fixture interfaces;
- unimplemented TimesNet ports and explicitly labelled deterministic mocks for routing tests;
- the existing synthetic-only standard iTransformer residual adapter;
- branch-lineage and time-alignment guards; and
- construction, but not execution, of the paper-derived future-fusion payload schema.

## Prohibited work

- TimesNet implementation or substitution;
- Bi-LSTM, attention, softmax weights, learned fusion, SCF-GAN, or adversarial-loss implementation;
- NASA/CALCE component input, validation/test access, real training, metrics, or search; and
- claims of framework, paper, or numerical reproduction.

## Gate consequence

`FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE` may pass when synthetic routing and fail-closed tests pass. `FRAMEWORK_REAL_DATA_FIDELITY_GATE` remains blocked until the inherited real-data blockers are resolved. This amendment does not authorize Phase 5 implementation.
