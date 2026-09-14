# Tier 2 Phase 9B Minimal Intelligent Fusion Protocol V1

`AUTHORITY=EXPLICIT_HUMAN_SPECIFICATION`

`PAPER_EXACT=False`

Frozen Phase 9A branch checkpoints are restored, set to evaluation mode, and
have `requires_grad=False`. No Phase 9A branch, HF, STL, split, normalization,
scalar-readout, or branch hyperparameter is modified.

The fusion sequence has length 10. Each row is
`[p_T,p_S,p_R,e_T_lag,e_S_lag,e_R_lag,r_T,r_S,r_R]`. Lagged error uses the
previous component truth divided by the previous frozen component prediction.
TRAIN uses the existing TRAIN-only STL targets. VALIDATION uses an STL prefix
ending at the prior observed capacity only; current targets, future validation
capacity, and all TEST values are excluded. Ratios use the current prediction
sum, with `EPS=1e-8` only when its absolute value is smaller than EPS.

The fusion topology is: 9 feature logits with feature-axis softmax attention;
Bi-LSTM(9, hidden 16/direction, one layer); 32-dimensional Tanh/Softmax time
attention; `32→128→64→32→3` with BatchNorm1d, LeakyReLU(0.1), Dropout(0.3),
and final Softmax; then last-step `[p_T,p_S,p_R]` weighting and `3→16→1` with
hidden ReLU and linear scalar Capacity_Ah output. SOH is `Capacity_Ah/2.0`.

Only fusion parameters are trained with Adam, LR 0.001, batch size 32, weight
decay 1e-5, MSE, seed 42, maximum 150 epochs, validation-capacity-MSE early
stopping, patience 50, and restored best checkpoint. The undefined paper
weight-regularization term is not executed.

`TEST_LOCKED=True`

`SCFGAN_TRAINED=False`

`GRID_SEARCH_RUN=False`
