# G-MTC Sample Identity and Alignment Contract — Phase 8A

## Canonical future identity fields

Every future sample/window/prediction payload must retain:

`battery_id`

`cycle_identity`

`source_target_role`

`real_synthetic_indicator`

`synthetic_set_id` (required for synthetic, forbidden for observed real)

`protocol`

`partition`

`window_start_cycle_identity`

`window_end_cycle_identity`

`target_cycle_identity`

`branch_name`

`feature_order`

`feature_scale_status`

`target_alignment_status`

`model_fidelity_status`

`provenance`

## Frozen boundaries

`CANONICAL_FEATURE_ORDER=[HF1,HF2,HF3]`

`SEQ_LEN=10__PAPER_EXACT`

`WINDOW_INPUT_ALIGNMENT_STATUS=UNRESOLVED`

`TREND_TARGET_ALIGNMENT_STATUS=UNRESOLVED`

`SEASONAL_TARGET_ALIGNMENT_STATUS=UNRESOLVED`

`RESIDUAL_TARGET_ALIGNMENT_STATUS=UNRESOLVED`

`REAL_PRED_LEN_STATUS=UNRESOLVED`

`REAL_COMPONENT_TARGETS_MATERIALIZABLE=False`

The contract forbids interpreting the ten-step paper window as an implicit `t+1`, same-endpoint, multi-step, or sequence-to-sequence target convention.

## Synthetic-label pairing

`SYNTHETIC_FEATURE_LABEL_POLICY=REUSE_OBSERVED_TRAIN_CYCLE_LABEL_ONLY`

`SYNTHETIC_TO_LABEL_CYCLE_ALIGNMENT_STATUS=BLOCKED_BY_UNRESOLVED_CONDITION_ENCODING_AND_TARGET_ALIGNMENT`

No row-order pairing, full-lifecycle inference, validation/test label use, or synthetic-SOH label is permitted. A synthetic HF row would need an explicit unique target TRAIN-cycle identity before it could receive the observed label/component target for that identity. The current condition encoding does not supply such an identity.

## Provenance and partition guard

`PARTITION_ALLOWED_FOR_FUTURE_TRAINING=TRAIN_ONLY`

`VALIDATION_TEST_PROVENANCE=REJECT`

`MIXED_PARTITION_PAYLOAD=REJECT`

`MISMATCHED_CYCLE_IDENTITY=REJECT`

`SYNTHETIC_SOH_REQUEST=REJECT`

`FEATURE_DIM_NOT_3_IN_CURRENT_REAL_FALLBACK=REJECT`

The Phase 8 validator is metadata-only. It never pads, truncates, reorders, normalizes, or numerically pairs a payload to make it pass.
