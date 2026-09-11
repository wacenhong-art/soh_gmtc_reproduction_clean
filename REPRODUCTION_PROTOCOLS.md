# Dual-protocol Reproduction Freeze

`AUTHOR_REPLY_STATUS=CONTACTED_WAITING`
`PRIMARY_SOURCE_CONFIRMATION_PENDING=True`
`PRIMARY_SOURCE_CONFIRMED=False`
`RESULT_BLIND=True`
`PERFORMANCE_BLIND=True`

This is a Human-authorized engineering reproduction decision, not a claim that the paper unambiguously specifies one protocol. Protocol A and Protocol B are independent tracks. Results from one track may not select, validate, or overwrite the other.

## PA-001 fallback — frozen engineering assumption

`PA001_STATUS=FALLBACK_FROZEN__PRIMARY_SOURCE_PENDING`
`PA001_FALLBACK_INTERPRETATION=USE_DATASET_NOMINAL_OR_RATED_CAPACITY_AS_SOH_DENOMINATOR`

| Dataset | Fallback denominator | Paper evidence | Status |
|---|---:|---|---|
| NASA | 2.0 Ah | PDF p.3 Table 1; SOH framing on p.1 | ENGINEERING_REPRODUCTION_ASSUMPTION |
| CALCE | 1.1 Ah | PDF p.3 Table 1; SOH framing on p.1 | ENGINEERING_REPRODUCTION_ASSUMPTION |
| Tsinghua | 24 Ah | PDF p.3 Table 1; SOH framing on p.1 | ENGINEERING_REPRODUCTION_ASSUMPTION |

`PRIMARY_SOURCE_CONFIRMED=False`

An author reply that specifies another denominator requires an author-decision record and a review of every affected manifest, label, experiment, and reported result. Historical fallback work must remain visible rather than silently overwritten.

## PA-004 — dual protocol tracks

`PA004_STATUS=AMBIGUOUS__DUAL_TRACK_FROZEN__PRIMARY_SOURCE_PENDING`

### PROTOCOL_A_GENERAL_SOH_ESTIMATION

**Paper scope:** Section 4 general target-battery SOH-estimation statement.

```text
target battery chronological cycle population [0, N)
  training   = [0, 0.60N)
  validation = [0.60N, 0.80N)
  test       = [0.80N, N)
```

`PROTOCOL_A_STATUS=FROZEN__ASSIGNMENT_VALID`

No row may cross these chronological partitions. Synthetic samples belong to neither Protocol A nor its test/validation populations unless a paper-matched experiment explicitly specifies their use.

### PROTOCOL_B_SMALL_SAMPLE_AUGMENTATION

**Paper scope:** Section 4.2 original-versus-SCF-GAN-augmented small-sample experiment.

```text
target battery original small-sample population = [0, 0.20N)
SCF-GAN augmentation = five complete synthetic HF datasets per original target set
```

`PROTOCOL_B_STATUS=FROZEN__ASSIGNMENT_VALID`

Protocol B does **not** claim to be the sole protocol for every final G-MTC experiment. Under Human-authorized `PLAN-CHANGE-0004`, it uses the following result-blind evaluation fallback:

```text
target battery chronological population [0, N), zero-based half-open intervals
  TRAIN_REAL              = [0, floor(0.20*N))
  UNUSED_PROTOCOL_B       = [floor(0.20*N), floor(0.60*N))
  VALIDATION              = [floor(0.60*N), floor(0.80*N))
  TEST                    = [floor(0.80*N), N)
```

`PROTOCOL_B_EVALUATION_FALLBACK=NESTED_SMALL_SAMPLE_WITH_GENERAL_HOLDOUT_BOUNDARIES`

`PROTOCOL_B_EVALUATION_PROVENANCE=HUMAN_FALLBACK__RESULT_BLIND__PERFORMANCE_BLIND`

The `[20%,60%)` records remain as `UNUSED_PROTOCOL_B`; they are neither silently dropped nor used for Protocol B fitting, GAN training, scaling/imputation fitting, STL training decomposition, model training, or hyperparameter selection. This choice preserves the paper’s stated 60%/80% holdout boundaries while not importing unstated real training data into the 20%-small-sample regime.

## Isolation and manifest contract

Every future protocol manifest row must retain:

```text
battery_id
cycle_id
canonical_source_key
raw_source_provenance
chronological_index
active_protocol
protocol_role
target_definition_version
```

`ACTIVE_PROTOCOL` is exactly `A` or `B`. `protocol_role` is assigned only after a paper-scope roster and a chronologically ordered target population are frozen. No performance statistic may decide a protocol, roster, split, augmentation placement, or target definition.

## Author clarification override policy

`AUTHOR_CLARIFICATION_OVERRIDE_POLICY=ACTIVE`

If the author replies:

1. Save the reply and create a primary-source decision record.
2. Compare its meaning with each fallback assumption and protocol track.
3. If any difference exists, set `PLAN_CHANGE_REQUIRED=True` before affected future work advances.
4. Amend pending protocol artifacts; retain already executed artifacts/results as historical fallback evidence.
5. Never silently rewrite, relabel, or delete prior fallback work.

## Remaining Phase 1 blockers

`PA002_SOURCE_TARGET_ROSTER=FROZEN__EXPERIMENT_SPECIFIC`
`PA005_SYNTHETIC_COUNT_AND_INSERTION=FROZEN_WITH_LIMITATION__NO_SYNTHETIC_ROWS_MATERIALIZED`

The present document authorizes Phase 1 protocol-contract work only. It does not authorize STL, a model, training, metrics, grid search, or Phase 2 entry.
