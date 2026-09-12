# iTransformer Input Semantics Audit — Phase 3A

## Candidate evaluation

| Candidate | Evidence for | Evidence against / missing link | Verdict |
|---|---|---|---|
| A. Univariate capacity residual only | STL capacity-domain cross-evidence; `R_t → iTransformer` routing; Fig.5 displays only residual branch input. | A one-channel input cannot realize the paper’s stated inter-channel voltage/temperature interaction; no conversion from capacity residual to those channels is specified. | Insufficient for paper-faithful real-data implementation. |
| B. Multivariate residual features | The paper says each architecture processes multivariate inputs and describes channel mixing. | No equation, figure, or preprocessing maps STL residuals for voltage, temperature, or HFs; Fig.5 shows no extra residual-branch inputs. | Unsupported. |
| C. STL per raw variable | “Multivariate inputs” could be read broadly. | Eq.(17) gives one `X_t`; component units/Phase-2 audit favor capacity; no per-variable STL flow is shown. | Unsupported. |
| D. Capacity residual plus voltage/temperature | Could make channel mixing non-degenerate. | No branch interface, time alignment, source, normalization, or leakage policy is reported. | Unsupported engineering addition. |
| E. Internal paper description is inconsistent | The prose is internally tense. | It does not explicitly state that voltage/temperature are simultaneous residual-branch inputs. | Not proven; retain ambiguity rather than declare contradiction. |

## Figure and prose reconciliation

`FIG5_RESIDUAL_BRANCH_INPUT=RESIDUAL_ONLY`

`VOLTAGE_TEMPERATURE_STATEMENT_STATUS=EXACT_WORDING__INPUT_MAPPING_UNRESOLVED`

`CAPACITY_RESIDUAL_ONLY_COMPATIBILITY=SUPPORTED_BY_ROUTING_AND_PHASE2_CROSS_EVIDENCE__INSUFFICIENT_FOR_UNIQUE_TENSOR_CONTRACT`

`CHANNEL_MIXING_INTERPRETATION=STANDARD_INVERTED_CROSS_VARIATE_ATTENTION_DESCRIPTION__NO_EXTRA_CHANNEL_MIXER_SPECIFIED`

`DYNAMIC_GRAPH_ATTENTION_IMPLEMENTATION_DETAIL=NOT_SPECIFIED`

No graph adjacency, edge weights, graph attention equation, graph layer, or iterative graph-update equation appears in the target-paper iTransformer section. No GAT or additional channel mixer is permitted as a paper-reproduction module.

## Implementation consequence

The phase may use a synthetic-only tensor contract only after the output/target alignment is independently frozen. That alignment is not supplied: the paper does not say whether a 10-step residual history predicts the same endpoint, the next residual, multiple future residuals, a component contribution, or SOH directly.

`SYNTHETIC_ONLY_INTERFACE_EXECUTABLE=False`

`REAL_DATA_FIDELITY_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING;ITRANSFORMER_INPUT_SEMANTICS_MISSING;ITRANSFORMER_TARGET_ALIGNMENT_MISSING`
