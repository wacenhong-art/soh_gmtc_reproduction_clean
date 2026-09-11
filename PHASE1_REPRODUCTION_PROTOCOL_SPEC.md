# Phase 1 Reproduction Protocol Specification

**Scope:** Human-authorized fallback protocol freeze and source-keyed membership definition. It contains no model, STL, GAN training, metrics, feature values, target values, or synthetic rows. All provenance labels are one of `PAPER_EXACT`, `PAPER_CROSS_EVIDENCE`, `HUMAN_FALLBACK`, or `UNRESOLVED`.

`AUTHOR_CONTACT_STATUS=CONTACTED`
`AUTHOR_REPLY_STATUS=CONTACTED_WAITING`
`GMAIL_MESSAGE_ID=1a0906ec8a4ae625`
`PRIMARY_SOURCE_OVERRIDE=True`
`STRICT_NUMERICAL_REPRODUCTION_LIMITATION=True`

## SOH target — PA-001

| Field | Value | Provenance | Limitation |
|---|---|---|---|
| Formula | `SOH_i = measured_capacity_i / dataset_rated_capacity` | HUMAN_FALLBACK | Paper defines SOH relative to nominal capacity and gives rated capacities, but does not publish a dataset-level executable formula. |
| NASA denominator | 2.0 Ah | HUMAN_FALLBACK | PDF p.3 Table 1 is direct capacity evidence. |
| CALCE denominator | 1.1 Ah | HUMAN_FALLBACK | PDF p.3 Table 1 is direct capacity evidence. |
| Tsinghua denominator | 24 Ah | HUMAN_FALLBACK | PDF p.3 Table 1 is direct capacity evidence. |

`PA001_STATUS=FALLBACK_FROZEN_WITH_LIMITATION`
`PRIMARY_SOURCE_CONFIRMED=False`

## Experiment-level battery-role matrix — PA-002

| EXPERIMENT_ID | DATASET | BATTERY | ROLE | SOURCE_LOCATION | CONFIDENCE |
|---|---|---|---|---|---|
| E1_GENERAL_GMTC_PROTOCOL | NASA | B0005, B0006 | SCF-GAN source examples | PDF p.9 §4 | PAPER_CROSS_EVIDENCE |
| E1_GENERAL_GMTC_PROTOCOL | CALCE | CS2-35, CS2-37 | SCF-GAN source examples | PDF p.9 §4 | PAPER_CROSS_EVIDENCE |
| E1_GENERAL_GMTC_PROTOCOL | NASA | B0007, B0018 | target SOH-estimation examples | PDF pp.9–10 §4 | PAPER_CROSS_EVIDENCE |
| E1_GENERAL_GMTC_PROTOCOL | CALCE | CS2-36, CS2-38 | target SOH-estimation examples | PDF pp.9–10 §4 | PAPER_CROSS_EVIDENCE |
| E2_SECTION42_BASELINE_AUGMENTATION | NASA | B0005, B0006, B0018 | original/augmented feature evaluation rows | PDF p.10 Table 5 | PAPER_EXACT |
| E2_SECTION42_BASELINE_AUGMENTATION | CALCE | CS2-36, CS2-37, CS2-38 | original/augmented feature evaluation rows | PDF p.10 Table 5 | PAPER_EXACT |
| E2_SECTION42_BASELINE_AUGMENTATION | NASA/CALCE | Table-5 SCF-GAN real-data source roster | not mapped per Table-5 row | PDF pp.9–10 | UNRESOLVED |
| E3_CONDITIONAL_GENERATION | Tsinghua | Th6 | SCF-GAN source | PDF p.11 §4.2 | PAPER_EXACT |
| E3_CONDITIONAL_GENERATION | CALCE | CS2-37 | SCF-GAN source | PDF p.11 §4.2 | PAPER_EXACT |
| E3_CONDITIONAL_GENERATION | Tsinghua | Th5 | conditional-generation target | PDF p.11 §4.2 | PAPER_EXACT |
| E3_CONDITIONAL_GENERATION | CALCE | CS2-35 | conditional-generation target | PDF p.11 §4.2 | PAPER_EXACT |
| E4_FINAL_RESULT_TABLES | NASA/CALCE/Tsinghua | Tables 6–8 cells | reported-result population | PDF pp.14–15 Tables 6–8 | PAPER_EXACT_FOR_REPORTING_ONLY__ROLE_MAPPING_UNRESOLVED |

`PA002_STATUS=RESOLVED_AS_EXPERIMENT_SPECIFIC_ROSTER`

There is no global source/target label. Rows with an unmapped role cannot be used for a new protocol without a separate Human fallback or author clarification.

## Protocol A — general SOH estimation

| Field | Value | Provenance |
|---|---|---|
| Active protocol | `A` | HUMAN_FALLBACK_DUAL_TRACK_CONTEXT |
| Target population | E1 target batteries only | PAPER_EXACT |
| Train | first chronological 60% of available target cycles | PAPER_EXACT |
| Validation | next chronological 20% | PAPER_EXACT |
| Test | final chronological 20% | PAPER_EXACT |
| SCF-GAN real-data population | E1 source batteries exclusively | PAPER_EXACT |
| Synthetic insertion | excluded unless a matching paper experiment says otherwise | HUMAN_FALLBACK_ISOLATION |

`PROTOCOL_A_STATUS=FROZEN__ASSIGNMENT_VALID`

## Protocol B — small-sample augmentation

| Field | Value | Provenance |
|---|---|---|
| Active protocol | `B` | HUMAN_FALLBACK_DUAL_TRACK_CONTEXT |
| Original target population | first chronological 20% of total target life | PAPER_EXACT |
| SCF-GAN real-data population | source batteries with complete charging data | PAPER_EXACT |
| Synthetic dataset count | five complete synthetic HF datasets per original target set | PAPER_EXACT |
| Synthetic insertion fallback | preserve original set and concatenate five complete synthetic sets, yielding original `N` plus synthetic `5N` records (`6N` total) | HUMAN_FALLBACK |
| Evaluation fallback | `NESTED_SMALL_SAMPLE_WITH_GENERAL_HOLDOUT_BOUNDARIES` | HUMAN_FALLBACK: authorized in PLAN-CHANGE-0004 |
| TRAIN_REAL | zero-based interval `[0, floor(0.20*N))` | HUMAN_FALLBACK, retaining paper’s first-20% training statement |
| UNUSED_PROTOCOL_B | `[floor(0.20*N), floor(0.60*N))` | HUMAN_FALLBACK, preserved in the manifest and excluded from all B fitting/augmentation |
| Validation | `[floor(0.60*N), floor(0.80*N))` | HUMAN_FALLBACK, retaining §4’s 60%/80% holdout boundaries |
| Test | `[floor(0.80*N), N)` | HUMAN_FALLBACK, retaining §4’s 80% boundary |

`PA005_STATUS=FALLBACK_FROZEN_WITH_LIMITATION`

`PROTOCOL_B_STATUS=FROZEN__ASSIGNMENT_VALID`

`ROUNDING_RULE=zero_based__TRAIN20_END=floor(0.20*N)__TRAIN60_END=floor(0.60*N)__VALID80_END=floor(0.80*N)__all_intervals_half_open`

`ROUNDING_RULE_PROVENANCE=HUMAN_ENGINEERING_FALLBACK__RESULT_BLIND`

The text “five complete synthetic datasets” and “expanded training sample size fivefold” does not uniquely establish whether “fivefold” describes synthetic-only or final total count. The fallback follows the direct first phrase, retains original observations, and adds no unmentioned merging or performance-based selection. The evaluation fallback preserves the paper’s stated general holdout boundaries while preventing the unmentioned `[20%,60%)` interval from silently expanding the small-sample training population.

`RESULT_BLIND=True`
`PERFORMANCE_BLIND=True`
`TEST_NOT_LOADED=True`

## Mandatory row-level manifest contract

Every membership row includes `active_protocol`, `experiment_id`, `dataset`, `battery_id`, `sample_uid`, `source_key`, `chronological_index`, `role`, `role_provenance`, raw provenance, and PA-version fields. Protocol A and B remain physically separate. No model, validation, or test result may be used to select or merge them. Membership is metadata only: `TEST_LOADED=False`, `TARGET_VALUES_MATERIALIZED=False`, and no feature values are copied into the manifest.

## Author-reply override

When primary-source clarification arrives, compare it against PA-001, PA-002, PA-004, and PA-005. Any material difference sets `PLAN_CHANGE_REQUIRED=True`; affected manifests and future experiments must be re-marked, while historical fallback work remains intact.
