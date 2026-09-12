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
| Fusion L_pred/L_weight definitions and coefficient | Fusion | MISSING | Eq.(24) names terms without details | Yes | No | False |
| Fusion errors/ratios definition | Fusion | AMBIGUOUS | 9d listed without inference-safe construction | Yes | No | False |
| GAN condition encoding | SCF-GAN | AMBIGUOUS | Cycle period, voltage range, and battery/usage descriptions differ | Yes | No | False |
| GAN latent/hidden dimensions, epochs, batch, LR, Adam betas | SCF-GAN | MISSING | Not reported | Yes | No | False |
| GAN SN/activation/layer-count conflict | SCF-GAN | AMBIGUOUS | Equations, tables, and prose differ | Yes | No | False |
| Augmented real/synthetic ratio and placement | augmentation | AMBIGUOUS | Five datasets/fivefold wording not operationalized | Yes | No | False |
| Epoch count, initialization, scheduler, clipping, seed list | training | MISSING | Not reported | Yes | No | False |
| Metric aggregation across ten runs | evaluation | MISSING | “Ten parallel experiments” without aggregation rule | Yes | No | False |

**Rule:** where `DEFAULT_WOULD_BE_PAPER_FAITHFUL=False`, an engineering choice requires a `REPRODUCTION_DEVIATION_RECORD`; it cannot be labelled a paper-exact parameter.
