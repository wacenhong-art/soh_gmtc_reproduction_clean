# Phase 9A Convergence Diagnostic Report

`STATUS=PASS__FIXED_CONVERGENCE_DIAGNOSTIC_COMPLETE`
`AUTHORITY=EXPLICIT_HUMAN_AUTHORIZATION__EXECUTION_CONTROL_ONLY`
`PAPER_EXACT=False`

## Frozen scope retained

The run preserved NASA E1 B0007/B0018, discharge-centered paired-cycle
membership, chronological 70/20/10 split, the PLAN-CHANGE-0017_REVISED HF
endpoint contract, TRAIN-only MinMax scaling, TRAIN-only offline STL(period 7),
the frozen TimesNet/iTransformer cores, `Linear(3,1)` readouts, capacity-domain
`Trend + Seasonal + Residual` fusion, MSE, CPU Adam, learning rate 0.001,
batch size 32, seed 42, and chronological no-shuffle batching.

The explicit execution-control overlay was `MAX_EPOCHS=150`,
`PATIENCE=50`, `MIN_DELTA=0.0`, early stopping on fused validation capacity
MSE, in-memory best-state capture, and best-state restoration. Each battery
started from a new seed-42 initialization. No retained 3-epoch model state was
used.

## Convergence result

| Battery | Best epoch | Stop epoch | Best fused VAL capacity MSE | Epochs since best | Status |
|---|---:|---:|---:|---:|---|
| B0007 | 10 | 60 | 0.0001479668717365712 | 50 | EARLY_STOPPED |
| B0018 | 14 | 64 | 0.00105582969263196 | 50 | EARLY_STOPPED |

`PHASE9A_CONVERGENCE_STATUS=EARLY_STOPPED`

All per-epoch component TRAIN-MSE traces and fused validation-capacity-MSE
traces are retained verbatim in `PHASE9A_CONVERGENCE_DIAGNOSTIC_METRICS.json`.
Validation is used only for the pre-authorized stopping metric; it is not used
for hyperparameter tuning, architecture choice, feature choice, or model
selection beyond restoring the pre-defined minimum-metric checkpoint.

## Restored-best validation diagnostics

| Battery | Capacity RMSE | Capacity MAE | Capacity R2 | SOH RMSE | SOH MAE | SOH RMSE % | SOH MAE % |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0007 | 0.0121641634 | 0.0080611324 | 0.8194237351 | 0.0060820817 | 0.0040305660 | 0.6082081739 | 0.4030566005 |
| B0018 | 0.0324935331 | 0.0198455490 | -0.6020894051 | 0.0162467668 | 0.0099227749 | 1.6246766768 | 0.9922774938 |

Capacity is the primary prediction domain. SOH values use the already
authorized PA001 fallback `SOH=Capacity_Ah/2.0`; neither metric domain drove a
post-run protocol modification.

## Execution safety audit

| Check | Result |
|---|---|
| NaN/Inf | 0 for both batteries |
| Shape contract | PASS: all scalar targets/outputs and fused outputs are `[B,1,1]`; no broadcasting |
| Scale contract | PASS: feature-wise MinMax fit on TRAIN only; validation transform/clip only |
| TEST lock | PASS: loaded/access attempts/materializations/transforms/predictions/metrics all 0 |
| Same-seed reproducibility | PASS: two complete diagnostic payloads exactly identical |
| SCF-GAN / synthetic HF / grid search | False / False / False |

`VALIDATION_USED_FOR_EARLY_STOPPING=True`
`VALIDATION_USED_FOR_HYPERPARAMETER_TUNING=False`
`NEXT_SCIENTIFIC_ACTION=RETURN_RESULTS_FOR_HUMAN_REVIEW`

This report does not authorize Phase 9B, TEST loading, retraining, or a model
change.
