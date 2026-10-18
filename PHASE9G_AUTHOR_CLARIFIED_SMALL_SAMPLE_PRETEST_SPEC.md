# Phase9G Author-Clarified Small-Sample Pretest Specification

This isolated TRAIN+VAL execution applies PLAN-CHANGE-0020. It preserves all
architecture and hyperparameter freezes: `SEQ_LEN=10`, `PRED_LEN=1`,
`STL_PERIOD=7`, seed 42, Adam (`lr=0.001`, batch 32), branch/fusion maximum
150 epochs, patience 50, and no hyperparameter tuning.

The real target TRAIN is only the first 20% of valid paired cycles. PC0019
synthetic trajectories are transformed by this new real-TRAIN-only scaler
without clipping; each trajectory has independent STL/window construction.
Validation is real-only and contains no synthetic sample. TEST is represented
only as sealed membership metadata.

The Phase9B 9-feature fusion remains unchanged. Its validation rows preserve
the existing sequential semantics: a row may use only a prior observed
capacity/component for a lagged-error field, never its current target or a
future validation capacity. The 20–60% unused real interval is never a
training row or fitting source; it may serve only as past observed sequential
context before a later validation prediction.

All quality gates are protocol, leakage, provenance, shape, and finiteness
gates. Validation quality cannot alter the frozen configuration.
