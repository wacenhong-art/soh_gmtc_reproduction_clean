# Tier 2 Scalar Component Readout Protocol V1

PROTOCOL_ID=TIER2_SCALAR_COMPONENT_READOUT_V1
PLAN_CHANGE=PLAN-CHANGE-0016
PROVENANCE=WEB_SUPPORTED_RESULT_BLIND_TASK_ADAPTATION
PAPER_EXACT=False
PAPER_EXACT_SCALAR_READOUT_STATUS=UNRESOLVED
TIER2_SCALAR_READOUT_STATUS=EXECUTABLE_BY_WEB_SUPPORTED_LINEAR_TASK_ADAPTER

## Contract

```text
CORE_OUTPUT_SHAPE=[B,1,3]
CORE_OUTPUT_SEMANTICS=LEARNED_INTERMEDIATE_MULTIVARIATE_OUTPUT
READOUT_TYPE=LINEAR_3_TO_1
READOUT_INPUT_DIM=3
READOUT_OUTPUT_DIM=1
READOUT_BIAS=True
READOUT_ACTIVATION=NONE
READOUT_PARAMETER_COUNT_PER_BRANCH=4
```

Each Trend, Seasonal, and Residual branch owns an independent `nn.Linear(3,1,bias=True)` adapter. It maps a core tensor `[B,1,3]` to `[B,1,1]`. No activation follows the adapter because seasonal and residual components may be negative.

## Branch routing

```text
HF window [B,10,3] -> Trend TimesNet core [B,1,3] -> Linear(3,1) -> Trend_hat [B,1,1]
HF window [B,10,3] -> Seasonal TimesNet core [B,1,3] -> Linear(3,1) -> Seasonal_hat [B,1,1]
HF window [B,10,3] -> Residual iTransformer core [B,1,3] -> Linear(3,1) -> Residual_hat [B,1,1]
Trend_hat + Seasonal_hat + Residual_hat -> Capacity_hat [B,1,1]
```

## Strict loss and exclusion contract

Branch loss is `MSE(prediction, target)` only when prediction and target have exactly equal shape `[B,1,1]` and finite values. Mismatched shapes fail closed before loss evaluation; broadcasting is prohibited.

No alternate reducer, nonlinearity, extra hidden layer, attention head, or component-weight search is permitted. The adapter does not modify either core’s `d_model`, `d_ff`, `e_layers`, `top_k`, `n_heads`, dropout, convolution, or attention implementation.
