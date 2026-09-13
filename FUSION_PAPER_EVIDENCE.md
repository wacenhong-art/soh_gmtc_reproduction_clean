# Adaptive Fusion Paper Evidence — Phase 6A

**Primary authority:** the project-local 17-page target-paper PDF. This is a paper-evidence audit; missing fusion semantics are not engineering facts.

| Field | Value | Status | Evidence location | Evidence summary / boundary |
|---|---|---|---|---|
| Fusion section | §3.6, Bi-LSTM weight learner based on attention mechanism | PAPER_EXACT | PDF p.7 §3.6 | Defines component attention, temporal attention, Bi-LSTM, weights, and final prediction. |
| Fusion input | Predictions, errors, ratios for Trend/Seasonal/Residual; 9 features | PAPER_EXACT | PDF p.7 §3.6; p.9 Table 4 | Per-feature error and ratio formula is absent. |
| Component attention | `x_attn=x ⊙ Softmax(theta_c)` | PAPER_EXACT | PDF p.7 Eq.(25) | `theta_c` shape and Softmax axis are absent. |
| Temporal attention | `z=sum_t alpha_t h_t`; `alpha_t=softmax(v^T a_t)` | PAPER_EXACT | PDF p.7 Eq.(26) | The mapping from `h_t` to `a_t` is not supplied. |
| Weight generation | `w=Softmax(W4 z3+b4)`, `w∈R^3`, sum=1 | PAPER_EXACT | PDF p.7 Eq.(27) | Table 4 gives layer widths. |
| Component fusion | Weighted predictions `sum_i w_i X_comp,i` | PAPER_EXACT | PDF p.7 Eq.(28) | `X_comp,i` is component-i prediction. |
| Final prediction | `yhat=Wp·ReLU(Wf(sum_i w_i X_comp,i)+bf)+bp`; final SOH estimation | PAPER_EXACT | PDF p.7 Eq.(28) | Domain conversion from capacity components is not reported. |
| Objective | `L_total=L_pred+lambda·L_weight` | PAPER_EXACT equation / MISSING operational details | PDF p.7 Eq.(24) | Neither term nor `lambda` is defined. |
| Optimization | Adam-like update with `lambda_decay=1e-5` | PAPER_EXACT | PDF p.7 Eq.(29) | Weight decay is distinct from missing fusion loss coefficient. |
| Bi-LSTM | input 9; hidden 16 per direction/32 combined; one layer; dropout .3; batch-first | PAPER_EXACT | PDF p.9 Table 4 | One-layer internal dropout behavior is a library fact, not a paper change. |
| Temporal attention configuration | input 32; Tanh + Softmax | PAPER_EXACT / PARTIALLY_SPECIFIED | PDF p.9 Table 4 | Internal attention score mapping remains incomplete. |
| Weight network | `32→128→64→32→3`; LeakyReLU .1; BatchNorm1d; dropout .3; Softmax | PAPER_EXACT | PDF p.9 Table 4 | BatchNorm/dropout placement is not stated. |
| Weighted feature selection | first `3d` features of last timestep × component weights | PAPER_EXACT statement / AMBIGUOUS semantics | PDF p.9 Table 4 | Ordering is not explicitly tied to component predictions. |
| Final head | `3→16→1`; ReLU | PAPER_EXACT configuration / AMBIGUOUS placement | PDF p.9 Table 4 | ReLU placement is unspecified. |

## Paper-supported topology

```text
three component predictions plus feature slots
  -> component attention -> Bi-LSTM -> temporal attention
  -> 32→128→64→32→3 weight generator -> Softmax three weights
  -> weighted component predictions -> 3→16→1 final prediction
```

`FUSION_CORE_TOPOLOGY_STATUS=PAPER_SUPPORTED`

`NOMINAL_FEATURE_GROUPING_STATUS=SEMANTIC_GROUPING_ONLY__PAPER_EXACT_GROUPS`

`WEIGHT_DECAY_STATUS=PAPER_EXACT__1E-5`

`FUSION_PREDICTION_LOSS_STATUS=MISSING`

`FUSION_WEIGHT_REGULARIZATION_STATUS=MISSING`

`FUSION_LAMBDA_STATUS=MISSING`

`FINAL_OUTPUT_SEMANTICS_STATUS=UNRESOLVED`

The Phase 6A structural module deliberately consumes only a synthetic, preconstructed `B×T×9` tensor. It creates no real error, ratio, prediction, or component feature.
