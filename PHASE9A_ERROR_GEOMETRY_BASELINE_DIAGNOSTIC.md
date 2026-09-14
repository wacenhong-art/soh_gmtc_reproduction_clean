# Phase 9A Error Geometry and Baseline Diagnostic

Scope: replay-persisted VALIDATION predictions only. No TEST numerical values were accessed.

## B0007

| Quantity | Mean | Std | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| Target capacity | 1.47903149268 | 0.0286254232228 | 1.43624567986 | 1.53913724422 | 0.102891564369 |
| Predicted capacity | 1.47628680748 | 0.0270005402799 | 1.42959904671 | 1.52095019817 | 0.0913511514664 |
| Error (prediction - target) | -0.00274468520108 | 0.0118504673443 | -0.0376073122025 | 0.0107687711716 | n/a |

R2_SSE=0.00503087368644; R2_SST=0.0278601050592; model R2=0.819423735993.

| Method | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Model | 0.0121641634785 | 0.00806113201029 | 0.819423735993 |
| TRAIN mean baseline | 0.247212090104 | 0.245549184262 | -73.5821989968 |
| Persistence baseline | 0.00843238369261 | 0.00600495587463 | 0.913224547574 |

Model minus persistence improvement (positive means lower model error): RMSE=-0.00373177978584; MAE=-0.00205617613567.

Objective variance/bias/level statistics: {"absolute_mean_bias_over_target_std": 0.09588278153040056, "mean_bias_prediction_minus_target": -0.0027446852010839127, "prediction_to_target_range_ratio": 0.887839076278187, "prediction_to_target_std_ratio": 0.9432363696340084, "prediction_variance": 0.0007290291754054582, "r2_is_negative": false, "sse_over_sst": 0.18057626400708468, "target_variance": 0.000819414854682582}

Chronological validation rows are stored in the JSON artifact under `validation_time_order_table` with sample index, target capacity, predicted capacity, and error only.

## B0018

| Quantity | Mean | Std | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| Target capacity | 1.39869794479 | 0.0256716367451 | 1.35671412945 | 1.46043097973 | 0.103716850281 |
| Predicted capacity | 1.39194056621 | 0.0205656646851 | 1.32592988014 | 1.42572164536 | 0.099791765213 |
| Error (prediction - target) | -0.00675737857819 | 0.0317831332053 | -0.124170064926 | 0.0273958444595 | n/a |

R2_SSE=0.0274515727614; R2_SST=0.0171348562625; model R2=-0.602089468446.

| Method | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Model | 0.0324935335351 | 0.0198455498769 | -0.602089468446 |
| TRAIN mean baseline | 0.233561849093 | 0.232146729039 | -81.7744467354 |
| Persistence baseline | 0.0221382552602 | 0.0115252918451 | 0.256331024905 |

Model minus persistence improvement (positive means lower model error): RMSE=-0.0103552782749; MAE=-0.0083202580318.

Objective variance/bias/level statistics: {"absolute_mean_bias_over_target_std": 0.26322351960977397, "mean_bias_prediction_minus_target": -0.006757378578186035, "prediction_to_target_range_ratio": 0.9621557629534274, "prediction_to_target_std_ratio": 0.8011045376375585, "prediction_variance": 0.0004229465639389084, "r2_is_negative": true, "sse_over_sst": 1.602089468446038, "target_variance": 0.0006590329331718247}

Chronological validation rows are stored in the JSON artifact under `validation_time_order_table` with sample index, target capacity, predicted capacity, and error only.
