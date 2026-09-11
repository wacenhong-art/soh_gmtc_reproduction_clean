# Phase 1 Augmentation Plan

`SCF_GAN_NOT_IMPLEMENTED=True`

`SYNTHETIC_ROWS_MATERIALIZED=False`

`PA005_PROVENANCE=HUMAN_FALLBACK`

`PA005_INTERNAL_WORDING_TENSION=True`

`AUTHOR_OVERRIDE_PENDING=True`

The Human fallback defines a future Protocol B construction after the evaluation fallback authorized in `PLAN-CHANGE-0004`:

```text
ORIGINAL_REAL_N = first-20%-of-total-life original small-sample count
EXPECTED_SYNTHETIC_SET_COUNT = 5
EXPECTED_ROWS_PER_SYNTHETIC_SET = ORIGINAL_REAL_N
EXPECTED_SYNTHETIC_TOTAL = 5N
FALLBACK_EXPECTED_AUGMENTED_TOTAL = 6N
```

`ELIGIBLE_TARGET_ROLE=TRAIN_REAL__PROTOCOL_B_ONLY`

`TARGET_TRAIN_REAL_COUNTS=NASA_B0007:34; NASA_B0018:26; CALCE_CS2_36:195; CALCE_CS2_38:216`

`VALIDATION_AUGMENTED=False`

`TEST_AUGMENTED=False`

`UNUSED_20_60_AUGMENTED=False`

`SYNTHETIC_INSERTION=UNMATERIALIZED__FUTURE_TRAINING_ONLY`

The paper’s wording “five complete synthetic datasets” conflicts in arithmetic interpretation with “expanded training sample size fivefold.” Thus `6N` is an engineering fallback, never a paper-exact total.
