# Phase 1 Augmentation Plan

`SCF_GAN_NOT_IMPLEMENTED=True`

`SYNTHETIC_ROWS_MATERIALIZED=False`

`PA005_PROVENANCE=HUMAN_FALLBACK`

`PA005_INTERNAL_WORDING_TENSION=True`

`AUTHOR_OVERRIDE_PENDING=True`

The Human fallback defines a future Protocol B construction only after Protocol B evaluation-population authority exists:

```text
ORIGINAL_REAL_N = first-20%-of-total-life original small-sample count
EXPECTED_SYNTHETIC_SET_COUNT = 5
EXPECTED_ROWS_PER_SYNTHETIC_SET = ORIGINAL_REAL_N
EXPECTED_SYNTHETIC_TOTAL = 5N
FALLBACK_EXPECTED_AUGMENTED_TOTAL = 6N
```

`TARGET_BATTERY=UNASSIGNED__BLOCKED_BEFORE_ROW_ASSIGNMENT`

`SYNTHETIC_INSERTION=UNMATERIALIZED__TRAINING_ONLY_IF_PROTOCOL_B_EVALUATION_POPULATION_IS_AUTHORIZED`

The paper’s wording “five complete synthetic datasets” conflicts in arithmetic interpretation with “expanded training sample size fivefold.” Thus `6N` is an engineering fallback, never a paper-exact total.

