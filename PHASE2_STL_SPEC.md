# Phase 2A — STL Specification and Gate Decision

`CURRENT_PHASE=Phase 2`

`PHASE_2_SUBPHASE=2A_SPECIFICATION_AND_LEAKAGE_CONTRACT`

`MODEL_IMPLEMENTED=False`

`STL_MODULE_IMPLEMENTED=False`

`STL_DECOMPOSITION_EXECUTED=False`

## Recovered paper contract

| Field | Value | Provenance |
|---|---|---|
| Equation | `X_t=T_t+S_t+R_t` | PAPER_EXACT, PDF p.6 Eq.(17) |
| Trend meaning | Long-term monotonic capacity fade | PAPER_EXACT, PDF p.6 §3.3 |
| Seasonal meaning | Recurring short-term usage / operational fluctuation | PAPER_EXACT, PDF p.6 §3.3 |
| Residual meaning | Noise, regeneration, other stochastic effects | PAPER_EXACT, PDF p.6 §3.3 |
| Trend branch | TimesNet | PAPER_EXACT, PDF pp.6, 9 |
| Seasonal branch | TimesNet variant | PAPER_EXACT, PDF pp.6, 9 |
| Residual branch | iTransformer | PAPER_EXACT, PDF pp.6, 9 |
| Downstream window | 10 steps | PAPER_EXACT, PDF p.6 §3.3 |
| STL input | Capacity sequence | PAPER_CROSS_EVIDENCE, PDF p.6 §3.3 + Fig.9 Ah component units |

## Gate assessment

| Required condition | Status | Reason |
|---|---|---|
| STL input semantics | `RESOLVED_BY_CROSS_EVIDENCE_WITH_EXECUTABLE_SCALE_LIMITATION` | Capacity-domain evidence is strongest, but no complete label/component construction is given. |
| STL parameter policy | `UNRESOLVED` | Period/windows/robustness/boundaries/library absent. |
| STL split order | `UNRESOLVED` | No paper statement about split relative to STL. |
| Component target construction | `UNRESOLVED_BLOCKING` | Validation/test real component labels are not specified. |
| Test lock | `PASS` | No test value was loaded or analysed. |

`NEW_STL_BLOCKER_FOUND=True`

`BLOCKER_ID=PA-STL-COMPONENT-TARGET_AND_PM-002_PA-006`

`PHASE_2_GATE=BLOCKED_BY_STL_SPEC_AMBIGUITY`

`PHASE_2_STATUS=BLOCKED__SPECIFICATION_COMPLETE__IMPLEMENTATION_NOT_AUTHORIZED`

`NEXT_ACTION=HUMAN_REVIEW_OF_STL_ENGINEERING_FALLBACK_OPTIONS_OR_PRIMARY_SOURCE_CLARIFICATION`

No parameter sensitivity, STL run, structural reconstruction result, model work, or test access is authorized by this specification-only outcome.

## Phase 2B Human-authorized engineering fallback outcome

Human authorized a result-blind, train-only structural period-discovery rule and partition-isolated offline STL fallback. The rule was applied to all eight Protocol-A/Protocol-B target training tracks without validation/test capacity values or predictive metrics. Each track failed the pre-registered ACF-versus-spectral agreement rule; no period is frozen and no STL decomposition has been executed.

`STL_EXECUTION_MODE=PARTITION_ISOLATED_OFFLINE_STL`
`STL_CAUSALITY_STATUS=NON_CAUSAL_OFFLINE_DECOMPOSITION`
`INTER_PARTITION_LEAKAGE=False`
`INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE=True`
`PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`
`PHASE_2_STATUS=BLOCKED__STRUCTURAL_PERIOD_NOT_IDENTIFIABLE`
`NEXT_ACTION=HUMAN_DECISION_ON_FIXED_ENGINEERING_PERIOD_OR_SMALL_BOUNDED_STRUCTURAL_SENSITIVITY_STUDY`

See `PHASE2_STL_ENGINEERING_FALLBACK.md` and `PHASE2_STL_PERIOD_DISCOVERY.md`; these records do not convert any fallback into a paper-exact parameter.

## Phase 2C period-discovery method audit

Phase 2C invalidated the Phase 2B global-ACF-argmax candidate rule for this persistent degradation setting: it mechanically selected lag 2 in every track and supplied no periodic-peak evidence. The revised local-peak/prominence concept cannot yet be frozen because the paper supplies no domain-grounded minimum period. CALCE has only diagnostic evidence around 50 observations; NASA has no stable family. No STL was run.

`PHASE_2C_STATUS=METHODOLOGY_AUDIT_COMPLETE__OUTCOME_C`
`PLAN_MINOR_AMENDMENT=STL-PERIOD-0001`
`PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`
`NEXT_ACTION=HUMAN_FIXED_ENGINEERING_PERIOD_OR_PRIMARY_SOURCE_CLARIFICATION`
