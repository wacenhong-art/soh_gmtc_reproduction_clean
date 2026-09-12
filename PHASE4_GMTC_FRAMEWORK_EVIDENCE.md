# Phase 4 G-MTC framework evidence

**Authority:** project-local target-paper PDF, 17 pages. This is evidence recovery and a synthetic-only scaffold boundary, not a real-data model result.

## Branch topology

| Field | Value | Status | Paper location | Evidence summary |
|---|---|---|---|---|
| `GMTC_FRAMEWORK_ROLE` | Component-specific temporal estimation followed by attention-enhanced fusion for final SOH estimation. | EXACT | p.8 Fig.5; pp.8–9 §3.7 | The framework routes STL components to specialized predictors, then fuses their component information. |
| `TREND_BRANCH_ROLE` | Trend component → TimesNet. | EXACT | p.6 §3.3; p.9 §3.7 | Text assigns TimesNet to long-term trend extraction. |
| `SEASONAL_BRANCH_ROLE` | Seasonal component → TimesNet variant with adaptive cycle detection. | EXACT | p.6 §3.3; p.9 §3.7 | Text assigns a seasonal TimesNet variant. |
| `RESIDUAL_BRANCH_ROLE` | Residual component → iTransformer. | EXACT | p.6 §3.3; p.9 §3.7 | Text assigns iTransformer to residual processing. |
| `FUSION_ROLE` | Attention-enhanced Bi-LSTM learns temporal/component weights and generates final SOH estimation. | EXACT | p.7 §3.6 Eqs.(24)–(29); p.9 Table 4 | Equations define attention, Bi-LSTM features, Softmax weights, and prediction head. |
| `FINAL_OUTPUT_ROLE` | Final SOH estimation. | EXACT | p.7 Eq.(28); p.8 Fig.5 | Figure ends at “Final SOH”; Eq.(28) denotes final prediction. |

`CORE_BRANCH_TOPOLOGY_STATUS=PAPER_TEXT_SUPPORTED`

## Figure 5 versus text

`FIG5_BRANCH_MODEL_VISIBILITY=NOT_SHOWN`

The lower panel of Fig.5 visibly routes Trend, Seasonal, and Residual components into the attention-based Bi-LSTM network, but does not visually label TimesNet or iTransformer blocks between component and fusion. Section 3.3 and §3.7 explicitly provide those branch assignments.

`TEXTUAL_BRANCH_MODEL_EVIDENCE=EXACT`

`FIG5_TEXT_CONSISTENCY_STATUS=CONSISTENT_SCHEMATIC_OMISSION`

## Fusion boundary

Table 4 reports fusion input dimension `9d` and identifies predictions, errors, and ratios for Trend, Seasonal, and Residual components. The paper does not define the component-ratio formula or define `d` sufficiently for an executable real-data input tensor.

`COMPONENT_RATIO_DEFINITION_STATUS=UNRESOLVED`

`FUSION_FEATURE_D_STATUS=UNRESOLVED`

The Phase 4 scaffold therefore carries a future-fusion schema only; it does not calculate errors, ratios, attention, Bi-LSTM states, or weights.

## Training orchestration and adversarial claim

| Field | Status | Paper location | Finding |
|---|---|---|---|
| `GMTC_TRAINING_ORCHESTRATION_STATUS` | `AMBIGUOUS__END_TO_END_OR_STAGED_UNSPECIFIED` | p.9 §3.7 | The paper permits end-to-end or carefully staged training but does not state which orchestration was used for the reported experiments. |
| `SHARED_ADVERSARIAL_TRAINING_DETAIL` | `NOT_SPECIFIED` | p.6 §3.3; pp.7–10 | Shared adversarial training is mentioned for physical constraints, but no executable loss, discriminator, weight, graph, or update procedure is supplied. |

## Final-output domain audit

| Candidate | Evidence for | Evidence against / limitation | Verdict |
|---|---|---|---|
| SOH-domain output | Fig.5 and Eq.(28) call the result final SOH. | Fig.9 component axes and text discuss fading capacity in Ah. | Plausible. |
| Capacity-domain output | STL audit cross-evidence identifies capacity sequence; Fig.9 components are capacity-domain. | Final fusion figure/text calls output SOH. | Plausible. |
| Capacity then normalized to SOH | Consistent with both observations. | Conversion location/formula is not stated. | Plausible but not proven. |

`FINAL_OUTPUT_SEMANTICS_STATUS=AMBIGUOUS`

`FINAL_OUTPUT_SEMANTICS_INFERENCE=False`

## Structural-only implication

The paper supports the branch topology and a 10-step branch-window statement, but not a complete real-data fusion input/output construction. Synthetic routing can be tested without resolving this, provided no synthetic behavior is represented as a paper-exact fusion implementation.
