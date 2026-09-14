# PLAN-CHANGE-0014 — Train-only Offline STL and Tier 2 Branch Configuration Freeze

STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_EXECUTION_PROTOCOL_COMPLETION  
TYPE=HUMAN_AUTHORIZED_RESULT_BLIND_EXECUTION_PROTOCOL_COMPLETION  
REAL_DATA_ACCESSED=False  
RESULT_DRIVEN=False  
PERFORMANCE_DRIVEN=False  
TEST_LOADED=False

## STL policy correction

PLAN-CHANGE-0014 retracts Phase 9A's over-strict requirement that offline STL cannot use future observations within the TRAIN partition. Ordinary offline STL is retained as the Tier 2 target-engineering method because causal/online STL is not paper-supported and would be a different method.

`STL_FIT_SCOPE=TRAIN_ONLY`  
`TRAIN_STL_CAN_USE_INTRA_TRAIN_OFFLINE_CONTEXT=True`  
`VAL_USED_IN_TRAIN_STL=False`  
`TEST_USED_IN_TRAIN_STL=False`  
`CROSS_SPLIT_CONTEXT=False`

Validation does not require component-target construction or validation STL. Its primary comparison target is the observed final next-step capacity-domain target after component-sum reconstruction.

## Tier 2 branch configuration

The result-blind Tier 2 execution configuration is frozen in `TIER2_BRANCH_EXECUTION_CONFIG_V1.md`. It uses the existing standard structural cores as disclosed Tier 2 engineering implementations, not as target-paper exact deep/lightweight variants. No configuration was selected from real data, validation metrics, test metrics, old-project outputs, or a search.
