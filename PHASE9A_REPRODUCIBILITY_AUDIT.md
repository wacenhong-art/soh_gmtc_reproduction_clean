# Phase 9A Reproducibility Audit — Executed Dry-Run

`SEED=42`  
`DEVICE=CPU`  
`DETERMINISTIC_ALGORITHMS=True`  
`EPOCHS=3`  
`BATCH_SIZE=32`  
`SHUFFLE=False`  
`DROP_LAST=False`  
`EARLY_STOPPING=False`

The complete Phase 9A TRAIN+VALIDATION payload was run twice with the same
fixed seed. Pair identities, split membership, TRAIN-fitted scalers, STL
targets, initialization fingerprints, loss traces, fused validation losses,
and final metrics serialized identically.

`SAME_SEED_RERUNS=2`  
`COMPARISON=COMPLETE_RESULT_PAYLOAD_EXACT`  
`DETERMINISM_STATUS=PASS`

The validation metrics were not consulted for any choice. No additional seeds,
epochs, restarts, checkpoints, or parameter configurations were run.
