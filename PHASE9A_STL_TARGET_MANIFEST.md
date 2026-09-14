# Phase 9A STL Target Manifest — Executed Dry-Run

`STL_PERIOD=7`  
`STL_MODE=OFFLINE_TRAIN_ONLY`  
`VAL_USED_IN_TRAIN_STL=False`  
`TEST_USED_IN_TRAIN_STL=False`  
`VAL_STL_REQUIRED=False`  
`COMPONENT_TARGET_DOMAIN=ORIGINAL_CAPACITY_DOMAIN`

For each target battery, only its paired TRAIN discharge-capacity sequence was
decomposed. Offline bilateral context inside that already-isolated TRAIN
sequence is the disclosed Tier 2 engineering policy; no validation or test
capacity entered STL.

| Battery | TRAIN pairs / STL length | Trend + Seasonal + Residual max absolute reconstruction error | Mean absolute reconstruction error | Leakage audit |
|---|---:|---:|---:|---|
| B0007 | 117 | 0.0 | 0.0 | PASS__OFFLINE_WITHIN_TRAIN_ONLY |
| B0018 | 92 | 2.220446049250313e-16 | 2.413528314402514e-18 | PASS__OFFLINE_WITHIN_TRAIN_ONLY |

No validation component target was created. Validation compares only the fused
capacity prediction against the observed next-step final capacity target.
