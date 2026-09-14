# Tier 2 Phase 9A Convergence Diagnostic Configuration V1

`AUTHORITY=EXPLICIT_HUMAN_AUTHORIZATION__PHASE9A_CONVERGENCE_DIAGNOSTIC`
`PAPER_EXACT=False`
`RESULT_BLIND=True`
`HYPERPARAMETER_TUNING=False`

This is a separate execution-control overlay on the completed three-epoch
Phase 9A dry-run. It preserves all scientific protocol fields: NASA pairing,
70/20/10 pair-first split, HF endpoint formulas, TRAIN-only MinMax scaling,
STL period/mode, branch cores, scalar readout, component sum, MSE, optimizer,
learning rate, batch size, seed, CPU device, and TEST lock.

```text
SEED=42
MAX_EPOCHS=150
EARLY_STOPPING=True
PATIENCE=50
MIN_DELTA=0.0
EARLY_STOPPING_METRIC=FUSED_VALIDATION_CAPACITY_MSE
BEST_CHECKPOINT=MIN_FUSED_VALIDATION_CAPACITY_MSE
RESTORE_BEST_CHECKPOINT=True
BEST_CHECKPOINT_PERSISTENCE=MEMORY_ONLY__NO_SELECTION_ARTIFACT

LEARNING_RATE=0.001
BATCH_SIZE=32
OPTIMIZER=Adam
DEVICE=CPU
LOSS=MSE
TRAIN_BATCH_SHUFFLE=False
VALIDATION_BATCH_SHUFFLE=False
DROP_LAST=False
```

Each battery starts from a fresh seed-42 initialization. Validation is used
only for the pre-authorized stopping/checkpoint metric; it must not choose a
model architecture, feature, pairing, normalization, optimizer value, or any
other hyperparameter. TEST remains sealed after the pair-first split.

`CONVERGENCE_STATUS=EARLY_STOPPED` only when patience triggers. Otherwise,
after epoch 150, it is `INCONCLUSIVE_MAX_EPOCH_CAP` and no parameter may be
changed automatically.
