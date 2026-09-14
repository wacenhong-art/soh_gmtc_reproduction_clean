# PAPER MISSING PARAMETER REGISTER

No entry below is eligible to become a paper parameter through framework defaults.

| PARAMETER | MODULE | STATUS | WHY_MISSING | IMPLEMENTATION_REQUIRED | SAFE_DEFAULT_EXISTS | DEFAULT_WOULD_BE_PAPER_FAITHFUL |
|---|---|---|---|---|---|---|
| SOH denominator | label | AMBIGUOUS | No dataset-level executable capacity equation | Yes | No | False |
| Forecast horizon / stride | sample construction | MISSING | 10-step window stated, target horizon and stride absent | Yes | No | False |
| STL period, seasonal/trend windows, robust, boundaries | STL | MISSING | Generic STL equation only | Yes | No | False |
| STL fit scope/timing | STL | AMBIGUOUS | Train-only/full-life and before/after windowing absent | Yes | No | False |
| TimesNet input/output dimensions, d_model, layers, top-k, kernels, dropout, normalization/head; Trend deep configuration; Seasonal lightweight mechanism; AdaptiveMerge formula | TimesNet | MISSING | Generic TimesNet exposition only; Phase 5A confirms branch-specific values and merge construction remain unreported | Yes | No | False |
| iTransformer d_model, heads, layers, d_ff, dropout, normalization, output head | iTransformer | MISSING | Generic equations only | Yes | No | False |
| iTransformer residual-input variables / tensor axes | iTransformer | AMBIGUOUS | Residual branch is explicit, but voltage/temperature channel-mixing prose has no branch-input mapping | Yes | No | False |
| Residual branch target | iTransformer | AMBIGUOUS | No concrete component-target tensor stated | Yes | No | False |
| Fusion L_pred/L_weight definitions and coefficient | Fusion | MISSING | Eq.(24) names terms without details; Phase 6A confirms lambda is not reported | Yes | No | False |
| Fusion errors/ratios definition, d semantics, first-3d feature ordering, final output domain, attention axes, and placement details | Fusion | AMBIGUOUS | 9d/9/3d table tension and no inference-safe feature construction; Phase 6A records exact missing items | Yes | No | False |
| GAN condition encoding | SCF-GAN | AMBIGUOUS | Cycle period, voltage range, and battery/usage descriptions differ | Yes | No | False |
| GAN condition set and generated-feature fields/dimension | SCF-GAN | CONTRADICTORY | Fig.3 labels Cycle+Battery type and 4D HF1/HF2/HF3/SOH; Fig.5 labels Cycle+Voltage; §4.2 explicitly names HF1/HF2/HF3 | Yes | No | False |
| GAN latent/hidden dimensions, epochs, batch, LR, Adam betas | SCF-GAN | MISSING | Not reported | Yes | No | False |
| GAN SN/activation/layer-count conflict | SCF-GAN | AMBIGUOUS | Equations, tables, and prose differ | Yes | No | False |
| SCF-GAN to STL/component branch interface | cross-phase G-MTC | AMBIGUOUS | No figure arrow/equation specifies generated fields, STL input domain, branch inputs, or component targets | Yes | No | False |
| Augmented real/synthetic ratio and placement | augmentation | AMBIGUOUS | Five datasets/fivefold wording not operationalized | Yes | No | False |
| Epoch count, initialization, scheduler, clipping, seed list | training | MISSING | Not reported | Yes | No | False |
| Metric aggregation across ten runs | evaluation | MISSING | “Ten parallel experiments” without aggregation rule | Yes | No | False |

**Rule:** where `DEFAULT_WOULD_BE_PAPER_FAITHFUL=False`, an engineering choice requires a `REPRODUCTION_DEVIATION_RECORD`; it cannot be labelled a paper-exact parameter.

## Phase 8A scale, alignment, and integration seams

`SCFGAN_OUTPUT_SCALE_STATUS=UNRESOLVED` — a Tanh generator output does not specify whether HF values are raw, normalized, or otherwise transformed.

`HF_NORMALIZATION_METHOD_STATUS=MISSING` — Fig. 5 shows “Normalize” but does not report the method, fit population, parameters, validation/test transform treatment, or inverse transform.

`BRANCH_INPUT_SCALE_STATUS=UNRESOLVED` and `REAL_SYNTHETIC_SCALE_COMPATIBILITY_STATUS=BLOCKED_BY_UNKNOWN_HF_NORMALIZATION_AND_GAN_OUTPUT_SCALE`.

`SYNTHETIC_TO_LABEL_CYCLE_ALIGNMENT_STATUS=BLOCKED_BY_UNRESOLVED_CONDITION_ENCODING_AND_TARGET_ALIGNMENT`.

These fields, together with the retained branch target alignment, STL-period, fusion feature-construction, final-domain, loss-composition, and orchestration rows, block both numeric synthetic integration and real-data Phase 9 work. They do not block metadata-only Level-1 contract validation.

## Phase 7B execution-policy note

`PLAN_CHANGE_ID=PLAN-CHANGE-0010`

The Human-authorized HF-input/component-target fallback operationalizes only the previously missing cross-phase bridge. It does not fill the GAN condition encoding, GAN fields/dimensions, GAN architecture, STL period, branch target alignment, or fusion semantics rows above. All remain non-paper-exact and subject to author override.

## PLAN-CHANGE-0014 Tier 2 execution note

The advisor-authorized Tier 2 fallback freezes a train-only offline STL policy and the standard-core branch execution values in `TIER2_BRANCH_EXECUTION_CONFIG_V1.md`. This is an execution fallback, not a resolution of the paper's missing parameters. Paper-exact rows remain missing or ambiguous.
