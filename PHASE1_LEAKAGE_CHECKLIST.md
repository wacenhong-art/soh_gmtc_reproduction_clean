# Phase 1 Leakage Checklist

**Status:** policy frozen and source-keyed sample-level membership audit passed. The audit is metadata-only; it neither materialized nor analysed test features or targets.

| Control | Frozen rule | Status |
|---|---|---|
| Preprocessing fit scope | Standardization, normalization, imputation, scaling, data-driven thresholds, and learned transforms fit on training rows only. | `FIT_ON_TRAIN_ONLY=True` |
| Validation | Transform only; never used to fit preprocessing. | `VALIDATION_FIT=False` |
| Test | Transform only after final model freeze; no analysis-time access. | `TEST_LOADED=False` |
| STL | Full lifecycle STL before split is prohibited unless paper-exact and leakage-audited in Phase 2. | `FULL_LIFECYCLE_STL_BEFORE_SPLIT=PROHIBITED_UNLESS_PAPER_EXACTLY_REQUIRES_AND_LEAKAGE_AUDITED` |
| Windows | Input cycles and prediction target must remain inside one permitted partition; no window crosses Train→Validation or Validation→Test. | `WINDOW_CROSS_SPLIT=PROHIBITED` |
| GAN | Future GAN real-data training must contain only the protocol-specific source pool; no target validation/test cycle may enter it. | `GAN_TARGET_LEAKAGE=PROHIBITED` |
| Protocol isolation | A and B cannot be pooled or selected by model performance. | `PROTOCOL_CROSS_CONTAMINATION=PROHIBITED` |

## Mechanical membership audit

| Check | Result | Expected |
|---|---:|---:|
| Train / validation overlap | 0 | 0 |
| Train / test overlap | 0 | 0 |
| Validation / test overlap | 0 | 0 |
| Protocol B train-real / unused overlap | 0 | 0 |
| Protocol B unused / validation overlap | 0 | 0 |
| Protocol B unused / test overlap | 0 | 0 |
| Source-key collisions within a protocol | 0 | 0 |
| Duplicate `SAMPLE_UID` within a protocol | 0 | 0 |
| Chronological-order violations | 0 | 0 |
| Protocol cross-contamination | 0 | 0 |

`CROSS_PROTOCOL_SHARED_RAW_SAMPLE_COUNT=2362`

This nonzero value is expected because A and B are two explicitly separate protocol tracks that cover the same E1 raw target populations. It is not contamination: no manifest row has more than one `active_protocol`, and no `experiment_id` pools their roles.

`TEST_MEMBERSHIP_DEFINED=True`

`TEST_LOADED=False`

`TEST_TARGET_ACCESSED_FOR_ANALYSIS=False`

`TEST_FEATURES_USED_FOR_FIT=False`

`TEST_USED_FOR_HYPERPARAMETER_SELECTION=False`
