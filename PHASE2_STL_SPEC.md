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
