# Phase9E NASA Paper-Aligned 60/20/20 Pretest Retrain

Tier-2 paper-closest numerical pretest. The paper specifies 60/20/20 but not integer rounding; floor/half-open is an implementation convention. This is not a paper-exact full-protocol claim.

## B0007

SPLIT=100/34/34; REAL/SYNTHETIC/AUGMENTED/VAL_BRANCH_WINDOWS=90/115/205/34.

| Branch | Best epoch | Stop epoch | Best real-VAL component MSE |
| --- | ---: | ---: | ---: |
| trend | 45 | 95 | 3.19239625242e-05 |
| seasonal | 81 | 131 | 5.11637263116e-05 |
| residual | 7 | 57 | 1.66499230545e-05 |

| Validation method | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Augmented simple sum | 0.0105437783877 | 0.00837143044919 | 0.853844881058 |
| Augmented intelligent fusion | 0.0278492349442 | 0.0234232060611 | -0.0196427106857 |
| Persistence | 0.00749272518948 | 0.00587225472057 | 0.926192403333 |

FUSION_BEST_EPOCH=81; FUSION_STOP_EPOCH=131; BEST_VAL_CAPACITY_MSE=0.000775579886977.
MEAN_WEIGHT_T/S/R=0.00380239146762/0.520790815353/0.475406825542; STD=0.00232906034216/0.368885308504/0.36995819211; WEIGHT_SUM_MAX_ERROR=1.19209289551e-07.

## B0018

SPLIT=79/26/27; REAL/SYNTHETIC/AUGMENTED/VAL_BRANCH_WINDOWS=69/80/149/26.

| Branch | Best epoch | Stop epoch | Best real-VAL component MSE |
| --- | ---: | ---: | ---: |
| trend | 9 | 59 | 0.00161839067005 |
| seasonal | 50 | 100 | 6.73921240377e-05 |
| residual | 23 | 73 | 3.31299415848e-05 |

| Validation method | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Augmented simple sum | 0.0424793574857 | 0.0386029109359 | -0.920712828636 |
| Augmented intelligent fusion | 0.0298148314202 | 0.0258997585624 | 0.0538254380226 |
| Persistence | 0.0155280786778 | 0.0116736760825 | 0.743349568726 |

FUSION_BEST_EPOCH=128; FUSION_STOP_EPOCH=150; BEST_VAL_CAPACITY_MSE=0.000888924172614.
MEAN_WEIGHT_T/S/R=0.0141685269773/0.164312273264/0.821519196033; STD=0.00938624981791/0.286986857653/0.29315072298; WEIGHT_SUM_MAX_ERROR=1.19209289551e-07.

## Development-history boundary

The retained 70/20/10 development results have a different split and are recorded only as non-comparable historical context; no cross-split value is interpreted as a performance improvement.

```text
TEST_LOADED=False
TEST_ACCESS_COUNT=0
SCFGAN_RETRAINED=False
SYNTHETIC_REGENERATED=False
CURRENT_TARGET_ERROR_USED=False
FUTURE_VALIDATION_CAPACITY_USED=False
GRID_SEARCH_RUN=False
HYPERPARAMETER_TUNING=False
PAPER_ALIGNED_PRETEST_TRAINING_GATE=PASS
```
