# Phase 9A Normalization Manifest — Executed Dry-Run

`NORMALIZATION=MINMAX_0_1`
`FIT_SCOPE=TRAIN_ONLY`
`VALIDATION_OPERATION=TRANSFORM_ONLY__NO_REFIT`
`TEST_TRANSFORM_APPLIED=False`
`OUT_OF_RANGE_POLICY=CLIP_TO_0_1__RESULT_BLIND_ENGINEERING_RULE`

Only paired TRAIN HF values fitted the three feature-wise extrema. Validation
was transformed once with those frozen extrema and never used to refit them.

| Battery | HF1 TRAIN min / max | HF2 TRAIN min / max | HF3 TRAIN min / max | VAL below-range counts HF1/HF2/HF3 | VAL above-range counts HF1/HF2/HF3 |
|---|---|---|---|---|---|
| B0007 | 0.0 / 3407.047 | 2.531 / 7102.359 | 0.0 / 1564.844 | 0 / 0 / 0 | 0 / 30 / 0 |
| B0018 | 6.859 / 3180.188 | 1429.922 / 8945.687 | 0.0 / 1411.797 | 0 / 0 / 0 | 0 / 1 / 0 |

Validation excursions were recorded, clipped under the pre-registered policy,
and did not trigger any change to scaling, features, or model configuration.
