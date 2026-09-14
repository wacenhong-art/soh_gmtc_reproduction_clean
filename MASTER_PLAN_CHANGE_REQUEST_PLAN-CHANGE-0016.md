# PLAN-CHANGE-0016: Freeze Tier 2 Scalar Component Readout

PLAN_CHANGE_ID=PLAN-CHANGE-0016
STATUS=APPLIED__HUMAN_AUTHORIZED__WEB_SUPPORTED_RESULT_BLIND_TASK_ADAPTATION
TYPE=WEB_SUPPORTED_RESULT_BLIND_TASK_ADAPTATION
HUMAN_AUTHORIZATION=True
PAPER_EXACT=False
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
REAL_HF_VALUES_USED=False
REAL_NUMERICAL_TARGET_VALUES_USED=False
MODEL_TRAINED=False
TEST_LOADED=False
GRID_SEARCH_RUN=False

## Problem addressed

The Tier 2 branch cores preserve their multivariate HF output shape `[B,1,3]`, but each Trend, Seasonal, and Residual supervision target is one scalar component `[B,1,1]`. The previous Phase 9A preflight correctly failed closed because no target-compatible readout was frozen.

## Frozen adapter

```text
TIER2_SCALAR_COMPONENT_READOUT=LINEAR_3_TO_1
INPUT_SHAPE=[B,1,3]
READOUT=nn.Linear(in_features=3,out_features=1,bias=True)
OUTPUT_SHAPE=[B,1,1]
READOUT_ACTIVATION=NONE
READOUT_PARAMETER_COUNT_PER_BRANCH=4
TOTAL_READOUT_PARAMETER_COUNT=12
```

The core output is a `LEARNED_INTERMEDIATE_MULTIVARIATE_OUTPUT`; it is not interpreted as three physical component predictions or as HF targets. The readout output alone has the scalar component supervision meaning.

## Explicitly prohibited alternatives

No channel selection, mean/sum/max pooling, fixed weighted sum, attention pooling, multilayer head, nonlinear activation, or performance-driven readout selection is authorized.

## Preserved boundaries

TimesNet and iTransformer core configurations and internals remain unchanged. The Tier 2 fusion remains `T_hat + S_hat + R_hat`. Branch loss remains strict shape-equal MSE between `[B,1,1]` prediction and target; broadcasting is rejected. `PAPER_EXACT_SCALAR_READOUT_STATUS=UNRESOLVED` remains active.

## Authorization result

This Plan Change enables only the synthetic structural verification of the adapter in the present run and a future Phase 9A retry after its gate passes. It does not authorize real numerical access, training, TEST loading, SCF-GAN, or a parameter search.
