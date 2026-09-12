# iTransformer Paper Evidence — Phase 3A

This evidence record is restricted to the target paper. It does not import parameters from the standard iTransformer implementation.

| Field | Paper evidence | Status | Source location | Interpretation boundary |
|---|---|---|---|---|
| ITRANSFORMER_SECTION | Section 3.5, “itransformer” | EXACT | PDF p.7 | Generic architecture exposition. |
| ITRANSFORMER_PURPOSE | Predict future variables from historical data; strengthen long-range dependencies in multivariate time series. | EXACT | PDF p.7 §3.5 | Not a residual-branch tensor definition. |
| ITRANSFORMER_INPUT_STATEMENTS | `h_n^0=Embedding(X_:,n)`; `X` is historical data. | EXACT | PDF p.7 Eq.(21) | Number/type of variables and axes are not defined. |
| ITRANSFORMER_OUTPUT_STATEMENTS | `Y_:,n=Projection(h_n^L)`; `Y` is predicted data. | EXACT | PDF p.7 Eq.(23) | Horizon and residual target alignment are absent. |
| ITRANSFORMER_EQUATIONS | Eqs.(21)–(23): MLP embedding, transformer blocks, MLP projection. | EXACT | PDF p.7 | Only a generic X-to-Y formulation. |
| ITRANSFORMER_ATTENTION_DESCRIPTION | “dynamic graph attention mechanism” and “iterative feature update strategy.” | EXACT claim | PDF p.7 §3.5 | No construction, adjacency, graph layer, or update equation is supplied. |
| ITRANSFORMER_CHANNEL_MIXING | Nonlinear dependencies through channel mixing; §3.3 names voltage fluctuations and temperature variations. | EXACT wording | PDF pp.6, 9 | Input channels are not mapped to residual branch. |
| ITRANSFORMER_DOWNSTREAM_ROLE | Residual component `R_t` is modeled by iTransformer. | EXACT | PDF pp.6, 9, 16; Fig.5 | Does not uniquely fix tensor shape or target. |
| ITRANSFORMER_WINDOW | Each architecture processes multivariate inputs through dedicated 10-step sliding windows. | EXACT | PDF p.6 §3.3 | Input history length only; not STL period, horizon, variable count, or heads. |

## Equation dimensional audit

`n` is not explicitly defined in the target paper. The notation `X_:,n`, `Y_:,n`, and `H={h_1,...,h_n}∈R^(N×D)` is compatible with a variable/token index, but the target paper does not state this as a tensor contract. Therefore:

`EQUATION_N_SEMANTICS=AMBIGUOUS__LIKELY_VARIABLE_TOKEN_BY_GENERIC_NOTATION`

`TARGET_X_TIME_VARIABLE_AXES=NOT_SPECIFIED`

`TARGET_Y_FORECAST_VARIABLE_AXES=NOT_SPECIFIED`

`RESIDUAL_TARGET_ALIGNMENT=AMBIGUOUS_BLOCKING__PA-007`

## Figure 5 audit

The rendered source page is retained at `output/paper_specification/fig5_source_render_page8.png`. In the lower panel, STL routes only its residual component to the residual branch representation (`X_r`); no voltage, temperature, HF, or separate side-channel arrow is shown. The figure does not draw the internal TimesNet/iTransformer blocks.

`FIG5_RESIDUAL_BRANCH_INPUT=RESIDUAL_ONLY`

`FIG5_EXTRA_VOLTAGE_TEMPERATURE_HF_INPUT=NOT_SHOWN`

`FIG5_TOP_BOTTOM_CONNECTION=ABSENT`

## Evidence verdict

The paper establishes branch routing but not an executable residual-branch tensor. A capacity-domain univariate residual is supported by Phase 2 cross-evidence and Fig.5 routing, while the stated multivariate voltage/temperature channel-mixing purpose cannot be instantiated from the paper without adding unreported inputs or modules.

`ITRANSFORMER_INPUT_SEMANTICS_STATUS=AMBIGUOUS_BLOCKING_REAL_DATA`

`FORECAST_TARGET_ALIGNMENT_STATUS=MISSING_BLOCKING__PM-001_AND_PA-007`

`CORE_DATAFLOW_CONTRADICTION_FOUND=False__UNRESOLVED_TENSION_NOT_AN_EXPLICIT_PAPER_CONTRADICTION`
