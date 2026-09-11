# PA-004 Protocol Evidence Table

**Task:** `PHASE_0_PA004_PROTOCOL_AMBIGUITY_RESOLUTION`  
**Scope:** evidence-based reconstruction only. No model code, data processing, training, metric calculation, or performance-based selection was performed.  
**Primary source:** the project-local 17-page paper PDF, *Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach*, DOI `10.1016/j.energy.2025.139463`.

## Original ambiguity retained verbatim

`PA004_CURRENT_STATEMENT=§4 says target available cycles split chronological 60%/20%/20%. §4.2 says original target training set is first 20% of total life and is augmented with five complete synthetic datasets.`

`PA004_CURRENT_EVIDENCE=PDF p.10 §4 and §4.2.`

`PA004_CURRENT_INTERPRETATIONS=A: 60% is target training partition, but only its first 20%-of-life portion is original data before augmentation. B: Separate experiments use different small-sample protocols.`

`PA004_WHY_BLOCKING=Train/validation/test population and all numerical claims change.`

## Full-text protocol evidence

| EVIDENCE_ID | PAGE | SECTION | FIGURE_OR_TABLE | PAPER_TEXT_SUMMARY | MENTIONS_60_20_20 | MENTIONS_FIRST_20_PERCENT | MENTIONS_SOURCE_BATTERY | MENTIONS_TARGET_BATTERY | MENTIONS_TRAIN | MENTIONS_VALIDATION | MENTIONS_TEST | MENTIONS_GAN | INTERPRETIVE_IMPLICATION | CONFIDENCE |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA004-E01 | 5 | §3, opening / §3.1–3.2 | Fig. 3 | Generated data are stated to train the subsequent STL-Transformer-TimesNet model; the GAN text defines generic real/generated distributions but does not partition a target lifetime. | No | No | No | No | Yes, generic | No | No | Yes | Confirms augmentation precedes temporal-model training at a high level, but supplies no split relationship. | High |
| PA004-E02 | 7 | §3.7 | — | SCF-GAN is trained on source batteries with complete full-voltage data; it generates target-battery feature vectors using a cycle condition when target coverage is limited. | No | No | Yes | Yes | Yes, generic | No | No | Yes | Source-only GAN training is explicit; it does not state which target-cycle interval supplies temporal-model training or conditions. | High |
| PA004-E03 | 8 | §3.7 | Fig. 5; Table 2 | The figure shows raw battery data, health-feature extraction, GAN augmentation, then STL and the three-model fusion. It contains no target-cycle split boundaries, counts, or train/validation/test labels. | No | No | No | No | No | No | No | Yes | The architecture diagram cannot resolve whether the two percentage statements are nested or separate. | High |
| PA004-E04 | 9–10 | §4 opening | Table 3; Table 4; Fig. 6 | Source examples B0005/B0006 and CS2-35/CS2-37 are used exclusively to train SCF-GAN. Target examples B0007/B0018 and CS2-36/CS2-38 have available cycles split chronologically: first 60% training, next 20% validation, final 20% test for the subsequent SOH estimator. | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | This is the only explicit complete target-lifetime partition. It establishes `60/20/20` at the individual target-battery lifecycle level. | High |
| PA004-E05 | 10 | §4.2 | Table 5 | For target batteries, the “original training set” simulates initial partial-window BMS data; its original-cycle count is set to the first 20% of total cycle life. SCF-GAN generates five complete synthetic datasets per original target set, described as expanding training sample size fivefold. | No | Yes | No | Yes | Yes | No | No | Yes | This creates a second training-data construction but does not identify its relation to the earlier 60% target training partition, nor allocate cycles in `[0.2N,0.6N)`. | High |
| PA004-E06 | 10 | §4.2 / Table 5 | Table 5 | The table reports Original and SCF-GAN-augmented feature results for several batteries, but reports neither total cycle counts nor training/validation/test sample counts. Its battery roster also overlaps with examples called sources elsewhere. | No | Implicit context only | Indirect / roster overlap | Indirect / roster overlap | Indirect | No | No | Yes | Results table cannot mechanically select a protocol; roster changes reinforce that role labels are experiment-specific, not a split map. | High |
| PA004-E07 | 11 | §4.2 conditional-generation experiment | Fig. 7 caption begins below | Th5/CS-35 are target examples; Th6/CS-37 are sources. HFs from the sources form the comprehensive SCF-GAN training set. The trained GAN generates five HF datasets for each target battery. | No | No | Yes | Yes | Yes, GAN only | No | No | Yes | Independently confirms source-only GAN real-data training and five target outputs. It still does not state whether target first-20% cycles provide GAN conditions, temporal-model real training, or both. | High |
| PA004-E08 | 12–15 | §§4.3–4.5 | Figs. 7–11; Tables 6–8 | Performance/ablation discussion names batteries and compares models, but provides no cycle-level partition diagram, sample count, or statement reconciling 60/20/20 with first-20%-of-life. | No | No | Indirect | Indirect | No additional | No additional | No additional | Indirect | No figure, caption, table, or result text supplies omitted allocation details. | High |
| PA004-E09 | 16 | Data availability | — | “Data will be made available on request.” No official repository, code URL, configuration file, or supplementary-material pointer is given. | No | No | No | No | No | No | No | No | No paper-designated code/supplementary artifact is available in-project to disambiguate the protocol. | High |

## Figure and table protocol check

`FIGURE_PROTOCOL_EVIDENCE=Fig. 5 (p.8) was visually inspected: it gives the high-level GAN-to-STL-to-fusion architecture but no cycle split, count, or set labels. Fig. 6 (p.9) evaluates generation quality; Figs. 7–11 display results/attention, not a partition scheme. No inspected caption supplies the missing relationship.`

`TABLE_PROTOCOL_EVIDENCE=Tables 2–4 are architecture/configuration tables. Table 5 labels original versus SCF-GAN-augmented features but gives no sample count or cycle ranges. Tables 6–8 give metrics only. None states how 60/20/20 and first-20%-of-life coexist.`

## Protocol-layer reconstruction

| Level | Candidate meaning in the paper | Evidence status |
|---|---|---|
| A — source/target allocation | Whole batteries are assigned source or target roles for an experiment. | Exact at concept level; exact roster varies by experiment. |
| B — target lifecycle partition | A target battery’s available cycles are chronological `train=[0,0.6N)`, `validation=[0.6N,0.8N)`, `test=[0.8N,N)`. | Exact (PA004-E04). |
| C — training-data construction inside target lifecycle | “Original training set” is `first 20% of total life`, i.e. `[0,0.2N)`, only if this is a nested construction. | Unresolved: paper does not explicitly say it is nested inside Level B. |
| D — SCF-GAN real-data / condition population | SCF-GAN real-data training comes from full-range source batteries. Target condition/query population is not mapped to target-cycle intervals. | Source real-data exact; target condition population unresolved. |
| E — final prediction/testing population | Final 20% target cycles, `[0.8N,N)`, if the §4 split governs that experiment. | Exact for the §4 split, but connection to §4.2 experiment unresolved. |

`SIXTY_TWENTY_TWENTY_LEVEL=LEVEL_B__TARGET_BATTERY_LIFECYCLE_PARTITION`

`FIRST_20_PERCENT_LEVEL=UNRESOLVED__EITHER_LEVEL_C_NESTED_TRAINING_CONSTRUCTION_OR_A_SEPARATE_SMALL_SAMPLE_EXPERIMENT`

`RELATIONSHIP=UNRESOLVED__NESTED_OR_INDEPENDENT_EXPERIMENTS`

## Evidence-supported candidate protocols

### CANDIDATE_PROTOCOL_A — nested construction (not confirmed)

```text
Raw target battery cycles: [0, N)
  ├─ temporal-model train partition stated in §4: [0, 0.6N)
  │    └─ original small-sample set stated in §4.2: [0, 0.2N)
  │         └─ five SCF-GAN synthetic HF datasets (how they fill/combine with [0.2N,0.6N) is unstated)
  ├─ validation: [0.6N, 0.8N)
  └─ test: [0.8N, N)

SCF-GAN real-data training: full-range source-battery HFs
```

**Evidence support:** the two percentage statements could be read hierarchically; source-only GAN training is explicit.  
**Unresolved defect:** no text says that `[0,0.2N)` is a subset of `[0,0.6N)`, or explains the role of real target cycles `[0.2N,0.6N)`.

### CANDIDATE_PROTOCOL_B — separate experiment constructions (not confirmed)

```text
§4 target-estimator protocol:
  Raw target cycles [0,N) -> train [0,0.6N), validation [0.6N,0.8N), test [0.8N,N)

§4.2 small-sample / augmentation protocol:
  Raw target cycles [0,N) -> original set [0,0.2N) -> five synthetic HF datasets
  Validation and test population for this construction: not stated

SCF-GAN real-data training: full-range source-battery HFs
```

**Evidence support:** the two statements occur in differently scoped passages, and Table 5 uses a role roster not identical to the §4 examples.  
**Unresolved defect:** the paper never declares them separate experiments or specifies how §4.2 results use validation/test sets.

`NESTED_PROTOCOL_HYPOTHESIS_STATUS=UNRESOLVED`

## GAN population and count-consistency audit

`GAN_REAL_SAMPLE_POPULATION=EXACT__FULL_RANGE_SOURCE_BATTERY_HFs_ONLY__PDF_pp7_9_11`

`GAN_CONDITION_POPULATION=UNRESOLVED__THE_PAPER_CALLS_c_CYCLE_PERIOD_VOLTAGE_RANGE_AND_OTHER_CONDITION_LABELS_BUT_DOES_NOT_MAP_TARGET_CYCLE_INTERVALS_TO_c`

`GAN_SYNTHETIC_OUTPUT_USED_IN=EXACT_AT_HIGH_LEVEL__SUBSEQUENT_TEMPORAL_MODEL_TRAINING__PDF_p5_p10`

`TARGET_FIRST_20_PERCENT_USED_FOR_GAN=UNRESOLVED__THE_PAPER_EXPLICITLY_RESERVES_GAN_REAL_TRAINING_FOR_SOURCES_BUT_DOES_NOT_SAY_WHETHER_FIRST20_TARGET_CYCLES_ARE_ONLY_TEMPORAL_MODEL_REAL_SAMPLES_OR_ALSO_GENERATION_CONDITIONS`

`VALIDATION_OR_TEST_VISIBLE_TO_GAN=NOT_STATED__NO_PERMISSION_TO_ASSUME_VISIBILITY_OR_NON_VISIBILITY_BEYOND_THE_EXPLICIT_SOURCE_ONLY_REAL_DATA_STATEMENT`

`CANDIDATE_COUNT_CONSISTENCY=INDETERMINATE__THE_PAPER_PROVIDES_PERCENTAGES_AND_FIVE_GENERATED_DATASETS_BUT_NO_PER_TARGET_N_TRAIN_VALIDATION_TEST_COUNTS_OR_SYNTHETIC_SAMPLE_COUNTS. BOTH_CANDIDATES_CANNOT_BE_MECHANICALLY_CONFIRMED_OR_REJECTED_FROM_STATED_COUNTS.`

## Supplementary material and official implementation audit

`SUPPLEMENTARY_AVAILABLE=False`

`SUPPLEMENTARY_PROTOCOL_EVIDENCE=None__NO_SUPPLEMENTARY_APPENDIX_OR_SUPPORTING_INFORMATION_FILE_EXISTS_IN_THE_PROJECT_AND_THE_PAPER_DOES_NOT_POINT_TO_ONE.`

`OFFICIAL_CODE_AVAILABLE=False`

`OFFICIAL_CODE_PROTOCOL_EVIDENCE=None__THE_PAPER'S_DATA_AVAILABILITY_STATEMENT_SAYS_DATA_WILL_BE_MADE_AVAILABLE_ON_REQUEST_AND_DOES_NOT_PROVIDE_AN_OFFICIAL_CODE_OR_REPOSITORY_LINK.`

## Resolution

`PA004_STATUS=UNRESOLVED_BLOCKING`

`PA004_ROOT_CAUSE=The paper gives two explicit, incompatible-or-unlinked target-data constructions but omits the containment/experiment-scope statement needed to map each target cycle to real temporal training, GAN conditioning, validation, or test.`

`PA004_RESOLUTION_CONFIDENCE=HIGH_FOR_UNRESOLVED_STATUS`

`PAPER_PROTOCOL_AMBIGUITY_REMAINS=True`

`HUMAN_AUTHORIZATION_REQUIRED=True`

`PHASE_0_EXIT_GATE=BLOCKED_BY_PA004`

`ADVANCEMENT_ALLOWED=False`

### Minimum Human decision alternatives (result-blind)

1. **Authorize Candidate A (nested)**: treat the first `20%` as a deliberately scarce original subset inside the stated `60%` target training partition. This preserves a final `20%` temporal test but requires an explicit engineering rule for the unexplained `[0.2N,0.6N)` interval.
2. **Authorize Candidate B (separate constructions)**: treat the `60/20/20` protocol as the full target-estimator evaluation and the first-`20%` protocol as a distinct augmentation/small-sample experiment. This preserves each textual statement but requires separate declared validation/test rules for the small-sample experiment.
3. **Obtain author clarification / official implementation**: retain strict paper reproduction as blocked until an external primary-source clarification maps the two statements. This is the only option that avoids an engineering interpretation.

No candidate was selected. `RESULT_BLIND=True`; `PERFORMANCE_BLIND=True`; `TEST_NOT_LOADED=True`.
