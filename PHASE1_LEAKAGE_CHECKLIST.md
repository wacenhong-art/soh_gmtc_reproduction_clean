# Phase 1 Leakage Checklist

**Status:** policy frozen; sample-level audit is `NOT_RUN__NO_ROW_ASSIGNMENT_AUTHORIZED`.

| Control | Frozen rule | Status |
|---|---|---|
| Preprocessing fit scope | Standardization, normalization, imputation, scaling, data-driven thresholds, and learned transforms fit on training rows only. | `FIT_ON_TRAIN_ONLY=True` |
| Validation | Transform only; never used to fit preprocessing. | `VALIDATION_FIT=False` |
| Test | Transform only after final model freeze; no analysis-time access. | `TEST_LOADED=False` |
| STL | Full lifecycle STL before split is prohibited unless paper-exact and leakage-audited in Phase 2. | `FULL_LIFECYCLE_STL_BEFORE_SPLIT=PROHIBITED_UNLESS_PAPER_EXACTLY_REQUIRES_AND_LEAKAGE_AUDITED` |
| Windows | Input cycles and prediction target must remain inside one permitted partition; no window crosses Train→Validation or Validation→Test. | `WINDOW_CROSS_SPLIT=PROHIBITED` |
| GAN | Future GAN real-data training must contain only the protocol-specific source pool; no target validation/test cycle may enter it. | `GAN_TARGET_LEAKAGE=PROHIBITED` |
| Protocol isolation | A and B cannot be pooled or selected by model performance. | `PROTOCOL_CROSS_CONTAMINATION=PROHIBITED` |

No count audit can truthfully report overlap values until the missing Protocol B evaluation population is resolved. Any later manifest must report zero expected duplicates, partition overlaps, source-key collisions, chronological violations, and protocol contamination.

