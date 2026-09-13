# Phase 9A Real-Data Numerical Pipeline Dry-Run Report

STATUS=BLOCKED_BEFORE_REAL_DATA_ACCESS  
REAL_DATA_USED=False  
TRAIN_DATA_USED=False  
VALIDATION_DATA_USED=False  
TEST_LOADED=False  
TEST_VALUE_ACCESS_COUNT=0  
GRID_SEARCH_RUN=False  
SCFGAN_TRAINED=False  
SYNTHETIC_HF_USED=False

## Preflight result

PLAN_READ_BEFORE_RUN=True  
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS__TIER2_PROTOCOL_CONSISTENT

PLAN-CHANGE-0013 establishes chronological 70/20/10 membership, component-sum fusion, 10-to-1 alignment, STL period 7, TRAIN-only min–max 0–1 normalization, hidden width 64, and MSE. The Phase 1 E1 role matrix uniquely identifies the four target batteries; no roster was changed.

## Fail-closed blockers

`PHASE9A_BLOCKED_BY_STL_TEMPORAL_LEAKAGE=True`  
`STL_TEMPORAL_LEAKAGE_AUDIT=FAIL__INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE`

The existing authorized STL mode is explicitly `PARTITION_ISOLATED_OFFLINE_STL` and `NON_CAUSAL_OFFLINE_DECOMPOSITION`. It prevents TRAIN/VALIDATION/TEST cross-partition contamination, but standard offline STL smoothing can use later observations within the declared partition. This violates Phase 9A's stricter requirement that target construction use no future observations. No alternative STL method was substituted.

`UNRESOLVED_EXECUTION_PARAMETER=TIMESNET_TOP_K,D_MODEL,D_FF,E_LAYERS,NUM_KERNELS,DROPOUT`  
`UNRESOLVED_EXECUTION_PARAMETER=ITRANSFORMER_D_MODEL,N_HEADS,E_LAYERS,D_FF,DROPOUT,ACTIVATION,NORMALIZATION`

The advisor protocol freezes only `PRIMARY_HIDDEN_WIDTH=64`; it does not freeze the remaining architecture parameters required by the existing real-data-incompatible structural modules. The modules are intentionally synthetic-only. No parameter was inferred, defaulted, searched, or copied from another project.

## Consequence

No split materialization, HF/capacity read, scaler fit, STL decomposition, window creation, branch forward/backward pass, checkpoint, metric, or prediction was performed. Therefore the requested split, normalization, temporal-leakage, and dry-run metric artifacts were not created; creating them would falsely imply real-data access or executable target construction.

NEXT_ACTION=HUMAN_OR_ADVISOR_AUTHORIZATION_FOR_CAUSAL_STL_TARGET_POLICY_AND_COMPLETE_BRANCH_EXECUTION_CONFIG

This is a protocol gate, not a scientific failure and not evidence about model performance.
