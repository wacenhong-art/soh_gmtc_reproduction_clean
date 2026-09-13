# REPRODUCTION MASTER PLAN

**Project:** faithful reproduction of Hanhua Yang et al., *Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach*, *Energy* 342 (2026) 139463, DOI: 10.1016/j.energy.2025.139463.

**Plan authority:** This is the sole highest-level roadmap for this repository. Every future Codex task must read this complete file before acting and must obey its gates. Its core architecture, phase order, reproduction definitions, and protocol rules may not be silently changed.

**Current plan state:** Phase 1 protocol membership and leakage freeze is complete under a Human-authorized, result-blind Protocol B evaluation fallback. Primary-source clarification remains pending and overrides the fallback if materially different. Phase 2 structural STL is complete only on deterministic synthetic fixtures; real NASA/CALCE STL remains blocked by its missing period. Phase 3 standard iTransformer core is structurally implemented and tested on a generic synthetic interface only; real residual input, target alignment, and target-paper architecture fidelity remain blocked. Phase 4 has a synthetic-only routing scaffold. Phase 5 has a Human-authorized synthetic standard TimesNet-core side-track, but neither target-paper Trend/Seasonal variant nor real-data TimesNet fidelity is established. Phase 6 has a synthetic-only preconstructed-feature fusion structural contract; it does not define any real error, ratio, 9d, loss, or final-output semantics.

`CURRENT_PLAN_VERSION=1.4.6`

`LAST_PLAN_CHANGE_ID=PLAN-CHANGE-0009`

## NON-NEGOTIABLE RULES

1. Reproduce first; innovate only after reproduction acceptance.
2. A simple baseline must never be represented as the paper framework.
3. TimesNet may be deferred for debugging, but it is mandatory for final reproduction.
4. SCF-GAN may be integrated later, but its paper-defined role must be validated for final reproduction.
5. Read and audit this file before and after every task.
6. Output the fixed progress block in every task.
7. A paper omission or ambiguity must never be represented as a paper fact.
8. Test/evaluation data must not be used for selection, tuning, normalization fitting, or early-stopping decisions.
9. Old projects are historical references only.
10. `PAPER_REPRODUCTION=PASS` requires Phase 10 acceptance; a runnable model is insufficient.
11. Every Codex task requires an audited, meaningful local Git commit and a verified private GitHub backup when infrastructure permits; an unsuccessful remote backup is an explicit governance blocker, never a reason to rewrite history.
12. Structural correctness precedes predictive claims; predictive validity precedes credibility claims; credibility and component contribution precede final reproduction acceptance; broad hyperparameter search is not a substitute for any of them.
13. `POOR_RESULT_ALONE IS_NOT A_PLAN_DEFECT`. Poor metrics trigger diagnosis before protocol, architecture, or plan changes.

## Evidence authority and initial paper status

`PAPER_FULLTEXT_STATUS=AVAILABLE`

The project-local authoritative paper is [Data-augmented SOH estimation for lithium-ion batteries under.pdf](<C:\Users\ROG\Documents\Python\new_soh\Data-augmented SOH estimation for lithium-ion batteries under.pdf>). It was found at repository initialization, is a 17-page PDF, and was read before this plan was written. The paper—not an abstract, old project, benchmark implementation, or generated code—is the authority for architecture, data protocol, parameters, and reported results.

`PAPER_FULLTEXT_REQUIRED_BEFORE_IMPLEMENTATION=False` because the local full text is available. This does **not** authorize implementation: Phase 0 remains incomplete until the executable paper specification is frozen and its ambiguities are resolved or recorded.

### Paper facts already observed (not an implementation specification)

| Item | Status | Full-text evidence captured for Phase 0 |
|---|---|---|
| Datasets | EXACT | NASA Ames PCoE, CALCE, and Tsinghua battery-aging datasets. |
| Health features | EXACT | Charging `HF1`: CC duration to upper-voltage threshold; `HF2`: CV duration from upper-voltage threshold to termination current; `HF3`: voltage-rise time from 3.8 V to 4.0 V. |
| SOH framing | AMBIGUOUS | Paper describes SOH as residual capacity relative to nominal/rated capacity; Phase 0 must freeze the executable label equation and each dataset’s capacity field. |
| Small-sample augmentation | EXACT | SCF-GAN: conditional GAN with spectral normalization, feature matching, and conditional batch normalization; feature-matching coefficient `lambda=0.1`. |
| Decomposition | EXACT | Additive STL: `X_t = T_t + S_t + R_t`. |
| Branch routing | EXACT | Trend -> TimesNet; Seasonal -> TimesNet variant; Residual -> iTransformer. |
| Branch-window statement | EXACT | Paper states dedicated 10-step sliding windows for the branch models. |
| Fusion | EXACT | Attention-enhanced Bi-LSTM weight learner, component and time-step attention; 9 input features (predictions, errors, ratios); softmax 3-component weights. |
| Fusion configuration | EXACT | Bi-LSTM input 9, hidden 16 per direction, one layer, dropout 0.3; weight network `32->128->64->32->3`, LeakyReLU alpha 0.1, BatchNorm1d, dropout 0.3; final `3->16->1`. |
| Temporal-model training configuration | EXACT | Learning rate 0.001, batch size 32, validation early stopping patience 50; ten parallel experiments; Python 3.9 / PyTorch 2.0.1+cu118 reported. |
| Target split | EXACT | Chronological first 60% train, next 20% validation, final 20% test for target-battery SOH estimation. |
| Small-sample regime | EXACT | Original target training set described as first 20% of total cycle life; five synthetic HF datasets per target set, described as fivefold expansion. Reconcile this with the 60/20/20 statement before execution. |
| Metrics | EXACT | RMSE, MAE, MAPE, R². |

### Architecture target

```text
Battery data / health features
        ↓
small-sample augmentation via SCF-GAN
        ↓
STL decomposition
 ┌──────┼──────┐
Trend Seasonal Residual
  ↓       ↓       ↓
TimesNet TimesNet iTransformer
 └───────┼────────┘
        ↓
adaptive attention-based weighted fusion
        ↓
SOH
```

The full text identifies the fusion learner specifically as an attention-enhanced Bi-LSTM. “Adaptive fusion” in all later work must preserve that requirement unless a formally approved deviation record says otherwise.

## Reproduction definitions

`FRAMEWORK_REPRODUCTION` means that the paper’s core computation has been established: STL, both TimesNet branches, iTransformer branch, attention-based adaptive fusion, and—where specified by the paper—SCF-GAN augmentation with its defined role.

`NUMERICAL_REPRODUCTION` means that the frozen paper-aligned dataset, inputs, label, chronological split, training protocol, and metrics produce results reasonably close to the paper’s reported results, with differences explicitly explained.

`FRAMEWORK_REPRODUCTION != NUMERICAL_REPRODUCTION`.

A model that runs, a partial scaffold, a classical baseline, or a proxy target does not establish either definition. `PAPER_REPRODUCTION=PASS` is possible only after Phase 10.

## REPRODUCTION VALIDATION HIERARCHY

The five levels below are ordered evidence claims, not a leaderboard. A later level cannot cure a failed earlier level. Results from a lower level must never be represented as proof of a higher level.

### LEVEL 1 — Structural Correctness

**Question:** does the code implement what the paper describes?

**Required evidence:** input/output shapes; tensor routing; STL reconstruction; Trend/Seasonal/Residual branch assignment; both TimesNet branches; iTransformer branch; attention-enhanced Bi-LSTM fusion; SCF-GAN’s paper-defined role; loss path; and gradient path. Every core module requires an independent minimal contract test before full integration.

`STRUCTURAL_VALIDATION_STATUS=NOT_STARTED`

If Level 1 fails, predictive performance may not be interpreted or compared to the paper.

### LEVEL 2 — Predictive Validity

**Question:** does the model actually predict under the frozen development protocol?

**Required metrics:** RMSE, MAE, and R² at minimum; MAPE when paper-aligned. Required sanity/reference baselines are Mean predictor, Persistence/previous-SOH predictor, and Ridge. Their purpose is `sanity check + predictive reference + complexity control`.

`PREDICTIVE_VALIDITY_STATUS=NOT_STARTED`

`BASELINE_MODEL != PAPER_REPRODUCTION_MODEL`

`RIDGE_HIGH_R2 DOES_NOT_MEAN PAPER_REPRODUCTION_SUCCESS`

If a complex model persistently fails to exceed simple baselines, first diagnose `CODE_BUG`, `DATA_PROTOCOL_ERROR`, `IMPLEMENTATION_MISMATCH`, `TRAINING_INSTABILITY`, `INSUFFICIENT_SAMPLE`, `PAPER_AMBIGUITY`, or `UNKNOWN`. Do not default to `NEEDS_MORE_GRID_SEARCH`. Conversely, `FULL_MODEL_LOWER_THAN_RIDGE` alone is not a reproduction failure; assess paper alignment, stability, protocol, sample count, and implementation fidelity first.

### LEVEL 3 — Credibility / Reproducibility

**Question:** is the result credible?

**Required evidence:** train/validation/test separation; preprocessing, future, STL, GAN, and standardization leakage controls; overlapping-window and duplicate-cycle audit; fixed evaluation population; and multi-seed reporting. The test set remains locked during development:

```text
TEST_SET_LOCKED=True
TestLoaded=False
```

Training learns parameters. Validation selects models, drives early stopping, and supports only permitted sensitivity work. Test may be loaded only after `FINAL_MODEL_FROZEN=True`; repeated test inspection followed by changes is prohibited.

`MULTI_SEED_EVALUATION=True`

Use five seeds by default for stochastic final models. Three are permitted only when resource constraints are recorded. Report individual seeds and mean ± standard deviation; a single best seed is never primary evidence.

`CREDIBILITY_STATUS=NOT_STARTED`

### LEVEL 4 — Component Contribution

**Question:** why is the model effective?

Use paper-logic-driven ablation, never arbitrary module deletion. The final acceptance evidence must consider the full model and suitable controls for: no SCF-GAN; no STL or an appropriate decomposition control; TimesNet replacement; iTransformer replacement; and adaptive-fusion versus simple-fusion control. Exact ablations must preserve the frozen evaluation population and be justified by the paper architecture.

`COMPONENT_CONTRIBUTION_STATUS=NOT_STARTED`

### LEVEL 5 — Controlled Optimization

**Question:** after correctness and credibility are established, can a small set of important choices be optimized under the frozen protocol?

Level 5 begins only after Levels 1–3 satisfy their declared entry conditions. Level 4 may be scheduled before or after limited optimization for practical reasons, but must be complete before final reproduction acceptance.

`HYPERPARAMETER_OPTIMIZATION_STATUS=NOT_STARTED`

## HYPERPARAMETER SEARCH POLICY

`LARGE_SCALE_GRID_SEARCH_BEFORE_FIRST_VALID_FULL_REPRODUCTION=PROHIBITED`

Before the first complete paper-aligned model has passed its structural checks, prohibit large-scale Grid Search, Random Search, Bayesian Optimization, massive Optuna searches, and exhaustive “800-grid” tuning. Grid search answers “which candidate is best on validation?” It does not establish whether the code follows the paper, whether the protocol is credible, or why a model works.

```text
GRID_SEARCH != STRUCTURAL_CORRECTNESS
GRID_SEARCH != CREDIBILITY
GRID_SEARCH != INTERPRETABILITY
GRID_SEARCH != REPRODUCTION_SUCCESS
```

Parameter priority is fixed:

1. `PAPER_EXACT_PARAMETER`: use it strictly.
2. `PAPER_AMBIGUOUS_PARAMETER`: use only a 3–5 value, physically/engineering-justified sensitivity analysis and record the ambiguity ID.
3. `LIMITED_SEARCH`: allowed only after model correctness, protocol correctness, and training stability are demonstrated; restrict to a few critical parameters with a validation-only rationale.
4. `LARGE_SEARCH`: requires explicit Human authorization.

Negative R², exploding loss, NaN, prediction collapse, constant prediction, severe overfit, or large baseline superiority must enter the diagnostic classification named in Level 2 before any search authorization.

## Module-level validation and small-subset sanity checks

`SMALL_BATCH_OVERFIT_TEST=REQUIRED_BEFORE_FORMAL_TRAINING`

For TimesNet, iTransformer, fusion, and the full predictor, first use a deliberately tiny **training-only** subset to verify that forward pass, backward pass, loss, optimizer, and gradients can substantially reduce training error. This is an implementation sanity check, not a performance experiment, numerical-reproduction result, or permission to load test data. Failure blocks formal large-scale training.

Minimum module evidence:

| Module | Required minimum validation before integration |
|---|---|
| STL | `Trend + Seasonal + Residual ≈ Original`; alignment, boundary, period, and shape audit. |
| iTransformer | input/output contract, loss, gradient, and small-subset overfit check. |
| TimesNet | input/output contract, period/tensor transformation, loss, gradient, and small-subset overfit check. |
| Fusion | weight shape, softmax normalization, gradients, and branch-contribution routing. |
| SCF-GAN | source-only training stability, condition usage, and generated-feature distribution checks. |

No module that lacks its minimum evidence may be hidden inside an end-to-end full-model claim.

## PLAN HEALTH AUDIT

`PLAN_HEALTH_AUDIT_REQUIRED_BEFORE_RUN=True`

`PLAN_HEALTH_AUDIT_REQUIRED_AFTER_RUN=True`

Every Codex task must perform `PLAN_HEALTH_AUDIT_BEFORE_RUN` immediately after reading this file and `PLAN_HEALTH_AUDIT_AFTER_RUN` after rereading it. Each audit must explicitly assess:

1. whether the research target remains clear;
2. whether the task still serves paper reproduction;
3. whether the current data protocol remains reasonable;
4. whether new evidence invalidates a downstream assumption;
5. whether new paper information requires plan correction;
6. whether model structure still matches the paper;
7. whether leakage risk exists;
8. whether sample count supports the proposed next step;
9. whether the current gate remains meaningful;
10. whether plan complexity exceeds the problem;
11. whether poor results are causing unprincipled module/tuning accumulation; and
12. whether “reproduce first, innovate later” is still being followed.

### Plan deviation levels and mandatory response

`PLAN_DEVIATION_LEVEL_0=NORMAL`: naming changes, file splits, small bug fixes, logging improvements, or other engineering changes that do not alter scientific protocol. Record normally; no core-plan change.

`PLAN_DEVIATION_LEVEL_1=MINOR`: a limited parameter ambiguity, implementation-detail correction, or minor execution-order optimization that does not alter core architecture, data protocol, research question, or evaluation logic. Append a `PLAN_MINOR_AMENDMENT` and continue only after it is recorded.

`PLAN_DEVIATION_LEVEL_2=MAJOR`: an incorrect paper-architecture interpretation; incorrect split or target definition; systemic leakage; invalid phase dependency; a core module materially unlike the paper; an inexecutably specified experiment; a route that would create substantial meaningless work; or a required redefinition of the reproduction target.

For Level 2, immediately set:

```text
PAUSE=True
PLAN_CHANGE_REQUIRED=True
ADVANCEMENT_ALLOWED=False
```

Create a `MASTER_PLAN_CHANGE_REQUEST`, await Human review, amend the plan only after authorization, and re-audit before continuing. `POOR_RESULT_ALONE IS_NOT A_PLAN_DEFECT`; a poor result may be a negative finding under a valid plan and must first enter the Level 2 diagnostic classification.

### Change-history contract

Every plan change must preserve history in the append-only `PLAN_CHANGE_LEDGER` using:

```text
PLAN_CHANGE_ID=
TRIGGER=
OLD_ASSUMPTION=
NEW_EVIDENCE=
IMPACT=
OLD_PLAN=
NEW_PLAN=
WHY_CHANGE_NECESSARY=
RESULT_DRIVEN=False/True
HUMAN_AUTHORIZED=True/False
COMMIT=
```

Core architecture, phase order, reproduction definition, or data protocol changes additionally require `MASTER_PLAN_CHANGE_REQUEST=True` and Human authorization. The paper has priority over this plan: evidence that SCF-GAN must occur earlier for faithful reproduction triggers the above change process rather than an unreviewed reorder.

## Mandatory run protocol

Every future Codex task must execute this protocol in order.

1. Read this entire `REPRODUCTION_MASTER_PLAN.md`.
2. Execute `PLAN_HEALTH_AUDIT_BEFORE_RUN`.
3. Read Git status and remote state.
4. Emit the current progress and Git/GitHub archive blocks.
5. Audit requested action and emit `PLAN_AUDIT_BEFORE_RUN` with `CURRENT_PHASE`, `CURRENT_GATE`, `LAST_COMPLETED_STEP`, `REQUESTED_ACTION`, `ACTION_ALLOWED=True/False`, and `PLAN_CONFLICT=True/False`.
6. If conflict or Level 2 deviation is found, stop; otherwise execute only current-Phase-permitted work.
7. Analyze task evidence without converting a poor result into a plan defect by default.
8. Re-read this file and execute `PLAN_HEALTH_AUDIT_AFTER_RUN`.
9. Determine `PLAN_DEVIATION_LEVEL`; for Level 1 append an amendment; for Level 2 pause and create a change request.
10. Update only permitted current status/progress/evidence/log fields.
11. Audit diff, create meaningful Git commit, push without force, and verify remote backup.
12. Emit final plan audit, progress, health, and archive blocks with all required status fields.

## Reproduction Master Progress

```text
========================================
REPRODUCTION MASTER PROGRESS
========================================

Phase 0  Paper Specification      [##########] 100%
Phase 1  Data Protocol            [##########] 100%
Phase 2  STL                      [#####.....] 50%
Phase 3  iTransformer             [#####.....] 50%
Phase 4  Framework Scaffold       [#####.....] 50%
Phase 5  TimesNet                 [#####.....] 50%
Phase 6  Adaptive Fusion          [#####.....] 50%
Phase 7  SCF-GAN                  [..........] 0%
Phase 8  Full G-MTC               [..........] 0%
Phase 9  Numerical Reproduction   [..........] 0%
Phase 10 Acceptance               [..........] 0%

OVERALL_PROGRESS=41%
CURRENT_PHASE=Phase 6
CURRENT_GATE=ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE: BLOCKED_BY_FEATURE_SEMANTICS_LOSS_AND_BRANCH_ALIGNMENT
LAST_COMPLETED=Phase 6A adaptive-fusion specification and synthetic preconstructed-feature structural implementation
NEXT_ACTION=PHASE_7_SCF_GAN_SPECIFICATION_AND_SYNTHETIC_ONLY_STRUCTURAL_WORK__OR_PRIMARY_SOURCE_CLARIFICATION; no real-data framework claim

PAPER_LOGIC_REPRODUCED=False
FRAMEWORK_REPRODUCTION_STATUS=NOT_STARTED
NUMERICAL_REPRODUCTION_STATUS=NOT_STARTED
STRUCTURAL_VALIDATION_STATUS=NOT_STARTED
PREDICTIVE_VALIDITY_STATUS=NOT_STARTED
CREDIBILITY_STATUS=NOT_STARTED
COMPONENT_CONTRIBUTION_STATUS=NOT_STARTED
HYPERPARAMETER_OPTIMIZATION_STATUS=NOT_STARTED
TEST_SET_LOCKED=True
TestLoaded=False
STL_STRUCTURAL_IMPLEMENTATION_GATE=PASS
STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD
FRAMEWORK_STL_IMPLEMENTATION_STATUS=STRUCTURALLY_REPRODUCED
PAPER_ALIGNED_REAL_STL_OUTPUTS_CREATED=False
NUMERICAL_REPRODUCTION_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING;ITRANSFORMER_INPUT_SEMANTICS_MISSING;ITRANSFORMER_TARGET_ALIGNMENT_MISSING;ITRANSFORMER_ARCH_PARAMETERS_MISSING;TIMESNET_INPUT_SEMANTICS_MISSING;TIMESNET_TARGET_ALIGNMENT_MISSING;TIMESNET_TOPK_MISSING;TREND_TIMESNET_CONFIGURATION_MISSING;SEASONAL_LIGHTWEIGHT_DEFINITION_MISSING;INCEPTION_CONFIGURATION_MISSING;ADAPTIVE_MERGE_DETAIL_MISSING;FUSION_INTERFACE_SEMANTICS_INCOMPLETE;FUSION_ERROR_DEFINITION_MISSING;FUSION_RATIO_DEFINITION_MISSING;FUSION_D_SEMANTICS_MISSING;FUSION_FEATURE_SELECTION_AMBIGUOUS;FUSION_FINAL_OUTPUT_SEMANTICS_MISSING;FUSION_WEIGHT_REGULARIZATION_MISSING;FUSION_LAMBDA_MISSING;GMTC_TRAINING_ORCHESTRATION_UNSPECIFIED;SHARED_ADVERSARIAL_TRAINING_UNSPECIFIED
ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=PASS
ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_TARGET_ALIGNMENT_AND_MISSING_ARCH_PARAMETERS
FRAMEWORK_ITRANSFORMER_STATUS=STRUCTURALLY_IMPLEMENTED_SYNTHETIC_ONLY
ITRANSFORMER_INPUT_SEMANTICS_STATUS=UNRESOLVED_BLOCKING_FOR_REAL_DATA
FORECAST_TARGET_ALIGNMENT_STATUS=UNRESOLVED_BLOCKING_FOR_REAL_DATA
REAL_FORECAST_ALIGNMENT=UNRESOLVED
REAL_RESIDUAL_TARGET=UNRESOLVED
REAL_COMPONENT_SUPERVISION=UNRESOLVED
FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE=PASS
FRAMEWORK_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_TIMESNET_AND_ITRANSFORMER_AND_FUSION_AND_TRAINING_ORCHESTRATION
FRAMEWORK_SCAFFOLD_STATUS=STRUCTURALLY_IMPLEMENTED_SYNTHETIC_ONLY
TIMESNET_STANDARD_CORE_STRUCTURAL_GATE=PASS
TIMESNET_BRANCH_VARIANT_FIDELITY_GATE=BLOCKED_BY_INPUT_TARGET_AND_VARIANT_PARAMETERS
FRAMEWORK_TIMESNET_STATUS=STANDARD_REFERENCE_CORE_STRUCTURALLY_IMPLEMENTED_SYNTHETIC_ONLY
========================================
```

## Phase 0 — Paper Specification Freeze

**OBJECTIVE:** convert the full paper into an executable, cited specification before writing formal model code.

**INPUT:** local full-text PDF, its tables/figures/equations, and lawful source-data documentation.

**ALLOWED_ACTIONS:** extract, visually verify, and cite paper text/equations/tables; inspect public dataset documentation; create a paper specification, ambiguity register, environment manifest template, and protocol design artifacts. Read-only forensic comparison to old code is permitted only after the paper specification is frozen.

**FORBIDDEN_ACTIONS:** formal model code; GAN/STL/TimesNet/Transformer training; experiments; performance results; locked-test loading; importing old model logic as a default.

**REQUIRED_ARTIFACTS:** `PAPER_SPECIFICATION.md`, `PAPER_AMBIGUITY_REGISTER.md`, `ENVIRONMENT_MANIFEST.md`, a source-to-paper evidence table, and a declared data-access/provenance plan.

**ACCEPTANCE_CRITERIA:** all following items are extracted from the PDF and marked `EXACT`, `AMBIGUOUS`, `MISSING`, or `INFERRED`: datasets; battery IDs; source and target cells; SOH definition; input HFs; charging windows; preprocessing; sequence construction; small-sample definition; split; SCF-GAN role; STL parameters; TimesNet configuration; iTransformer configuration; fusion; optimizer; learning rate; batch size; epoch/early stopping; metrics; reported results; and ablations. Every `INFERRED` item is visibly non-paper fact. All unresolved items receive a `PAPER_AMBIGUITY_ID`.

**FAILURE_CONDITIONS:** unverified text asserted as exact; inaccessible source data silently substituted; a protocol contradiction not entered in the ambiguity register; or model/experiment work started.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 0 acceptance recorded with Human-visible unresolved-item list and signed `PHASE_COMPLETION_COMMIT`.

### Authorized side-track — Split-independent data readiness while PA-004 is blocked

`SPLIT_INDEPENDENT_DATA_READINESS_AUTHORIZED=True`

This is an explicitly authorized Phase-0 side-track, not Phase 1 and not a protocol freeze. It exists solely to make lawful, provenance-preserving raw data available in an **unsplit** canonical form. It neither selects `PA-004` Candidate A/B nor changes the paper’s target definition, train/validation/test population, small-sample definition, GAN target-data allocation, model architecture, or evaluation population.

**OBJECTIVE:** acquire paper-roster NASA/CALCE raw files from official or provenance-verifiable sources; create deterministic operation/cycle inventory; verify chronological order and the feasibility of the three paper health features; retain raw capacity/provenance without deciding an SOH denominator.

**ALLOWED_ACTIONS:** official-source lookup and download; SHA-256, size, license/access and provenance records; deterministic raw-file parsing; battery and operation inventory; raw-record/cycle counts; duplicate, timestamp-order, missing-value, malformed-record and physical-range flags; feature-constructability checks; an unsplit canonical manifest with `battery_id`, chronological cycle identity/order, raw provenance, legally available raw capacity, and feature-extraction status.

**FORBIDDEN_ACTIONS:** assigning `train`, `validation`, `test`, `first20`, source/target-GAN population, or any split-derived role; fitting standardizers; constructing split-dependent windows; choosing an SOH denominator; deleting adverse cycles; STL; any neural/classical model; augmentation; training; metrics; hyperparameter search; or selecting either PA-004 candidate.

**REQUIRED_ARTIFACTS:** `DATA_SOURCE_MANIFEST.md`, `output/data_readiness/data_source_manifest.json`, an ignored raw-data directory, deterministic parser source, unsplit inventory/feature-feasibility manifests, and a data-readiness report. Large raw files must remain ignored.

**SPLIT_INDEPENDENT_DATA_READINESS_GATE:** pass only if `OFFICIAL_DATA_SOURCE_VERIFIED=True`, `BATTERY_ROSTER_VERIFIED=True`, `RAW_PARSING_VERIFIED=True`, `CYCLE_ORDER_VERIFIED=True`, `FEATURE_CONSTRUCTION_FEASIBILITY_VERIFIED=True`, `NO_TRAIN_VAL_TEST_ASSIGNMENT=True`, and `PA004_UNCHANGED=True`.

**NON-ADVANCEMENT RULE:** even when this side-track gate passes, retain `PHASE_0_SPECIFICATION_WORK=COMPLETE`, `PHASE_0_EXIT_GATE=BLOCKED_BY_PA004`, `PA004_STATUS=UNRESOLVED_BLOCKING`, and `ADVANCEMENT_TO_PROTOCOL_FREEZE=False`. Data readiness is not permission to enter Phase 1.

## Phase 1 — Data & Experimental Protocol

**OBJECTIVE:** establish traceable raw-data handling and a leakage-safe paper-aligned protocol.

**INPUT:** frozen Phase 0 specification, acquired datasets, dataset licenses/provenance, and approved ambiguity decisions.

**ALLOWED_ACTIONS:** acquire/parse data; define cycles; compute/validate SOH label and HF construction; define source/target cells; construct chronological train/validation/test protocol; create manifests and non-model validation checks.

**FORBIDDEN_ACTIONS:** STL, GAN, Transformer, TimesNet, prediction modeling, training, and performance reporting.

**REQUIRED_ARTIFACTS:** `DATA_MANIFEST`, `PROTOCOL_MANIFEST`, raw-to-sample provenance mapping, source/target roster, split manifest, and leakage-risk checklist.

**ACCEPTANCE_CRITERIA:** every input sample traces to immutable raw data and its cycle; label/feature code has unit and boundary checks; source-only GAN data and target chronological partitions are mechanically demonstrable; all fit-time statistics are train-only.

**FAILURE_CONDITIONS:** unknown origin; duplicate/cross-partition cycle; future/all-life normalization; target leakage into source-only GAN training; or undocumented proxy labels.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 1 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

### Authorized dual-protocol fallback while primary-source clarification is pending

`PLAN_CHANGE_ID=PLAN-CHANGE-0003`

Human-authorized fallback permits Phase 1 protocol-freeze work with two isolated tracks: `PROTOCOL_A_GENERAL_SOH_ESTIMATION` (Section 4 chronological 60/20/20) and `PROTOCOL_B_SMALL_SAMPLE_AUGMENTATION` (Section 4.2 first-20%-of-life plus five synthetic HF datasets). The PA-001 engineering fallback uses each dataset’s nominal/rated capacity as the SOH denominator. These are explicitly non-primary-source-confirmed engineering interpretations, recorded in `REPRODUCTION_PROTOCOLS.md`.

**Additional Phase 1 controls:** every future row must carry `ACTIVE_PROTOCOL=A/B` and source-key provenance. Protocol A real data and Protocol B real/synthetic data are isolated. Performance cannot select either track. PA-002 blocks row-level experiment-roster assignment; PA-005 blocks Protocol B augmentation execution. The author-clarification override policy requires a primary-source decision record and plan-change review if a reply differs.

## Phase 2 — STL Reproduction

**OBJECTIVE:** independently reproduce decomposition behavior before neural modeling.

**INPUT:** Phase 1 frozen manifest and exact/approved STL parameters. A synthetic-only parameterized software check may use explicit fixture parameters under PLAN-CHANGE-0005; it is not real-data parameter approval.

**ALLOWED_ACTIONS:** implement and test STL only; inspect raw degradation sequence, Trend, Seasonal, and Residual; create component manifests and reconstruction checks. Before real parameters are frozen, implementation tests are restricted to deterministic synthetic fixtures with an explicit period and no raw battery input.

**FORBIDDEN_ACTIONS:** GAN, TimesNet, iTransformer, fusion, end-to-end model, training, and numerical-performance claims.

**REQUIRED_ARTIFACTS:** STL configuration record, decomposition outputs with lineage, reconstruction/alignment/boundary tests, and period/window decision linked to paper evidence or ambiguity ID.

**ACCEPTANCE_CRITERIA:** `Trend + Seasonal + Residual ≈ original` under a predefined numerical tolerance; shape, time alignment, boundaries, window, period, and unavailable-cycle behavior are verified without future leakage.

**FAILURE_CONDITIONS:** mismatched indexes, unexplained reconstruction error, full-lifetime fitting that contaminates held-out evaluation, or unlogged parameter substitution.

**NEXT_PHASE_ENTRY_CONDITION:** the real-data fidelity gate passes and its `PHASE_COMPLETION_COMMIT` is recorded. A structural-only Phase 3 side-track may be authorized after the structural gate passes, but it must use synthetic components only and cannot claim real-data or paper-aligned reproduction.

### Phase 2 dual-gate contract — PLAN-CHANGE-0005

`STL_STRUCTURAL_IMPLEMENTATION_GATE` verifies a parameterized software contract on deterministic synthetic fixtures only: explicit period required; one declared partition; component and metadata schema; additive reconstruction; length preservation; finite outputs; determinism; and fail-closed rejection of missing/invalid periods or combined/full-lifecycle/TEST partitions.

`STL_REAL_DATA_FIDELITY_GATE` verifies paper-defensible real-battery period, smoother configuration, split order, and component-target construction. Until authoritative evidence or separately authorized scientific fallback is available, set `STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD`, `REAL_BATTERY_STL_AUTHORIZED=False`, and `PAPER_ALIGNED_REAL_STL_OUTPUTS_CREATED=False`.

Structural success is not Phase 2 paper reproduction or numerical reproduction success. It must not remove `STL_REAL_BATTERY_PERIOD_MISSING` from numerical-reproduction blockers.

## Phase 3 — Transformer / iTransformer Branch

**OBJECTIVE:** reproduce and validate the Residual -> iTransformer branch in isolation.

**INPUT:** frozen component manifest and paper-specified/approved iTransformer configuration.

**ALLOWED_ACTIONS:** implement iTransformer branch, tensor contracts, branch-local training checks, loss and prediction-target verification, and non-test development validation.

**FORBIDDEN_ACTIONS:** Ridge or any classical method as replacement; TimesNet substitute; fusion acceptance; SCF-GAN integration; paper-success claims.

**REQUIRED_ARTIFACTS:** iTransformer specification record, input/output tensor contract, branch-local configuration, seed/environment capture, and development-only verification log.

**ACCEPTANCE_CRITERIA:** residual tensor, 10-step sequence handling (unless Phase 0 corrects it with evidence), feature dimensions, target, loss, gradients, and non-test training behavior all match the frozen specification.

**FAILURE_CONDITIONS:** tensor mismatch, target leakage, implicit model substitution, or test-driven architecture selection.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 3 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

### Phase 3 dual-gate contract — PLAN-CHANGE-0006

`ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE` can assess a synthetic-only tensor contract, strict paper-exact 10-step input length, forward/backward wiring, determinism, and fail-closed behavior after a pre-registered generic synthetic target contract is frozen. A generic synthetic target does not infer the target paper’s horizon, offset, or real residual supervision. Structural work cannot use real residuals, validation/test values, model metrics, or target-paper parameter defaults.

`ITRANSFORMER_REAL_DATA_FIDELITY_GATE` can pass only after paper-defensible residual construction, input-channel semantics, target/horizon alignment, and architecture parameters are frozen. It inherits `STL_REAL_BATTERY_PERIOD_MISSING` and remains blocked by current iTransformer ambiguities.

No structural work can establish iTransformer paper reproduction, framework reproduction, or numerical reproduction.

## Phase 4 — G-MTC Framework Scaffold

**OBJECTIVE:** validate the routing graph `STL -> {Trend, Seasonal, Residual} -> temporal outputs -> fusion interface` without claiming complete paper architecture.

**INPUT:** accepted STL and iTransformer artifacts.

**ALLOWED_ACTIONS:** build interfaces and shape/lineage tests. Temporary Trend/Seasonal engineering placeholders are allowed only if explicitly labeled `TEMPORARY_ONLY=True`.

**FORBIDDEN_ACTIONS:** declaring framework or paper reproduction; presenting placeholder outputs as TimesNet; numerical reproduction; end-to-end SCF-GAN tuning.

**REQUIRED_ARTIFACTS:** branch routing contract, component-output schema, fusion-interface contract, and temporary-branch registry when applicable.

**ACCEPTANCE_CRITERIA:** component routing, temporal outputs, fusion input order, shapes, provenance, and gradients are verified. Any placeholder is visibly registered.

**FAILURE_CONDITIONS:** hidden placeholder, component swap, unsourced fusion interface, or any `PAPER_REPRODUCTION_STATUS=ESTABLISHED` claim.

**STATUS RULE:** before Phase 5, `PAPER_FRAMEWORK_REPRODUCTION=PASS` is prohibited and `PAPER_REPRODUCTION_STATUS=NOT_ESTABLISHED`.

**NEXT_PHASE_ENTRY_CONDITION:** the real-data fidelity gate passes and its `PHASE_COMPLETION_COMMIT` is recorded. A synthetic-only Phase 5 structural side-track may be authorized after the structural scaffold gate passes, but it cannot establish TimesNet, framework reproduction, or numerical reproduction.

### Phase 4 dual-gate contract — PLAN-CHANGE-0007

`FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE` may assess a synthetic-only component-routing graph, unimplemented TimesNet ports, an explicitly marked mock used only for routing tests, the synthetic-only Phase 3 iTransformer adapter, future-fusion payload schema, alignment, provenance guards, and fail-closed behavior. It must not implement TimesNet, Bi-LSTM fusion, attention, Softmax component weights, SCF-GAN, or any real-data workflow.

`FRAMEWORK_REAL_DATA_FIDELITY_GATE` can pass only after paper-defensible real STL components, TimesNet configuration/implementation, residual semantics and target alignment, fusion feature/output semantics, and training orchestration are frozen. It inherits all current numerical-reproduction blockers. A structural pass cannot establish framework, paper, or numerical reproduction.

## Phase 5 — TimesNet Reproduction

**OBJECTIVE:** formally implement Trend -> TimesNet and Seasonal -> TimesNet under the frozen paper specification.

**INPUT:** Phase 4 interfaces and paper/ambiguity-controlled TimesNet configuration.

**ALLOWED_ACTIONS:** implement TimesNet period detection, 1D-to-2D transformation, Inception-style processing, adaptive merge, Trend and Seasonal branch variants, and branch-local checks.

**FORBIDDEN_ACTIONS:** temporary branch retention; paper-success declaration before later gates; unapproved architecture choices.

**REQUIRED_ARTIFACTS:** two TimesNet branch specifications, tensor/period tests, source evidence map, and removal record for every temporary branch.

**ACCEPTANCE_CRITERIA:** both specified TimesNet branches pass contracts and `NO_TEMPORARY_BRANCHES=True`.

**FAILURE_CONDITIONS:** generic Transformer substitution, unverified hyperparameters represented as paper facts, or residual placeholders.

**NEXT_PHASE_ENTRY_CONDITION:** the branch-variant fidelity gate passes and its `PHASE_COMPLETION_COMMIT` is recorded. A synthetic-only Phase 6 structural side-track may be authorized after the standard-core structural gate passes, but it cannot establish adaptive-fusion, framework, paper, or numerical reproduction.

### Phase 5 dual-gate contract — PLAN-CHANGE-0008

`TIMESNET_STANDARD_CORE_STRUCTURAL_GATE` may assess only a generic synthetic input contract, explicit FFT top-k period discovery, deterministic padding/reshape/crop, parameter-shared Inception-style processing, standard-reference amplitude merge, forward/backward wiring, and fail-closed provenance/configuration behavior. Every unreported architecture value is an explicit synthetic fixture parameter, not a target-paper parameter.

`TIMESNET_BRANCH_VARIANT_FIDELITY_GATE` can pass only after target-paper-defensible Trend and Seasonal input semantics, forecast/component target alignment, top-k, deep versus lightweight configuration, Inception configuration, and AdaptiveMerge semantics are frozen. It inherits `STL_REAL_BATTERY_PERIOD_MISSING` and all current numerical-reproduction blockers. A standard-core structural pass cannot establish either TimesNet branch, framework, paper, or numerical reproduction.

## Phase 6 — Adaptive Fusion

**OBJECTIVE:** reproduce the paper’s attention-based Bi-LSTM adaptive fusion.

**INPUT:** three paper-specified branch outputs and fusion specification.

**ALLOWED_ACTIONS:** implement component attention, time-step attention, Bi-LSTM, softmax weights, weighted fusion, final prediction head, and contract tests.

**FORBIDDEN_ACTIONS:** direct sum represented as the paper fusion; test-selected fusion variants; omission of required 9-feature contract without a deviation record.

**REQUIRED_ARTIFACTS:** fusion configuration, weight/normalization/gradient tests, 9-feature provenance definition, and direct-sum comparator identified as an ablation only.

**ACCEPTANCE_CRITERIA:** weights have expected shape, normalize across the three components, are trainable, receive gradients, and map to the correct component predictions.

**FAILURE_CONDITIONS:** detached weight path, non-normalized weights, data leakage in error/ratio features, or non-paper fusion asserted as paper logic.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 6 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

### Phase 6 dual-gate contract — PLAN-CHANGE-0009

`ADAPTIVE_FUSION_STRUCTURAL_GATE` may assess only a synthetic, preconstructed `B×T×9` tensor; the paper-supported component-attention, Bi-LSTM, temporal-attention, 3-way Softmax-weight, weighted-component, and final-head topology; gradients; normalization; deterministic wiring; and fail-closed provenance/configuration behavior. Its fixture may not calculate a real component prediction, error, ratio, `d`, real loss, or real final SOH/capacity output. Every unreported implementation choice must be visibly marked synthetic-fixture-only.

`ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE` can pass only after an inference-safe error and ratio definition, `d` semantics, first-`3d` feature selection, final-output domain/conversion, `L_pred`, `L_weight`, fusion lambda, real branch targets, and their time alignment are frozen. It inherits all numerical-reproduction blockers. A structural pass cannot establish adaptive fusion, framework, paper, or numerical reproduction. A synthetic-only Phase 7 structural side-track may be authorized after the structural gate passes, but it cannot establish SCF-GAN or a full framework.

## Phase 7 — SCF-GAN Reproduction

**OBJECTIVE:** independently reproduce source-trained SCF-GAN and verify generated health-feature data before integration.

**INPUT:** frozen source/target/data manifests and SCF-GAN specification.

**ALLOWED_ACTIONS:** implement the spectral-normalized conditional GAN, conditional batch normalization, feature matching, source-only training, and generation-quality assessment.

**FORBIDDEN_ACTIONS:** joint opaque GAN/predictor tuning as first validation; target-test access; treating generated samples as raw observations; source/target redefinition after results.

**REQUIRED_ARTIFACTS:** GAN architecture/loss/config record, condition definition, source/target roster, real/generated provenance, generation manifest, and quality evaluation plan including the paper’s distribution metrics where reproducible.

**ACCEPTANCE_CRITERIA:** conditions, real samples, source batteries, generated samples, `lambda=0.1`, feature matching, CBN, and spectral-normalization placement are traceable to full text or ambiguity records; generation is evaluated independently.

**FAILURE_CONDITIONS:** GAN trained on target/test data contrary to frozen protocol, mode-collapse/unphysical outputs without a failed gate, or conflating data quality with downstream score.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 7 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 8 — Full G-MTC Integration

**OBJECTIVE:** construct the complete paper chain: `SCF-GAN -> augmented data -> STL -> TimesNet + TimesNet + iTransformer -> adaptive fusion -> SOH`.

**INPUT:** all accepted branch, fusion, GAN, and protocol artifacts.

**ALLOWED_ACTIONS:** controlled integration, end-to-end contract checks, frozen-development validation, and framework acceptance audit.

**FORBIDDEN_ACTIONS:** changing protocol for better results; numerical-reproduction claims before Phase 9; hidden fallback model.

**REQUIRED_ARTIFACTS:** integration graph, full configuration/seed/environment manifest, end-to-end provenance audit, no-temporary-branch check, and framework acceptance report.

**ACCEPTANCE_CRITERIA:** every core paper component is present and routed as specified; all inputs retain provenance; no temporary branches remain; full protocol has no known unrecorded deviation.

**FAILURE_CONDITIONS:** missing core component, unrecorded substitution, cross-split leakage, or integration that cannot trace predictions to raw data and configurations.

**NEXT_PHASE_ENTRY_CONDITION:** set `FRAMEWORK_REPRODUCTION_STATUS=PASS` only after this gate and record its `PHASE_COMPLETION_COMMIT`.

## Phase 9 — Paper-Aligned Numerical Reproduction

**OBJECTIVE:** evaluate the frozen complete framework against paper-aligned numerical evidence.

**INPUT:** Phase 8-accepted framework, frozen manifest, paper-reported metrics/results, and locked evaluation partitions.

**ALLOWED_ACTIONS:** protocol-locked runs; RMSE/MAE/MAPE/R² and paper-reported comparison; ten-run aggregation where full paper protocol supports it; variance reporting; discrepancy investigation without protocol mutation.

**FORBIDDEN_ACTIONS:** changing split, label, inputs, target population, future information, source/target roster, STL period, or model logic after observing results. No difficult-sample removal.

**REQUIRED_ARTIFACTS:** immutable run manifests, seed list, per-run and aggregate metric tables, paper-vs-ours table, evaluation-population declaration, and discrepancy log.

**ACCEPTANCE_CRITERIA:** results are computed only under the frozen protocol and compared metric-by-metric to the paper. Any difference is classified as `CODE_DIFFERENCE`, `PROTOCOL_AMBIGUITY`, `PARAMETER_MISSING`, `DATA_VERSION`, `TRAINING_VARIANCE`, or `UNRESOLVED`.

**FAILURE_CONDITIONS:** post-hoc protocol tuning, test-driven early stopping, result selection, or a numerical-success claim without lineage evidence.

**NEXT_PHASE_ENTRY_CONDITION:** a complete numerical evidence package exists and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 10 — Reproduction Acceptance

**OBJECTIVE:** issue the final, evidence-bounded reproduction verdict.

**INPUT:** all phase gates, commits, manifests, paper comparison, deviations, and unresolved ambiguity register.

**ALLOWED_ACTIONS:** final audit and transparent status declaration.

**FORBIDDEN_ACTIONS:** hiding discrepancies or declaring success from a single running model/baseline.

**REQUIRED_ARTIFACTS:** final acceptance report, complete provenance chain, signed phase ledger, paper-vs-ours metric report, and final deviation/ambiguity register.

**ACCEPTANCE_CRITERIA:** declare, with evidence: `PAPER_LOGIC_REPRODUCED=`, `FRAMEWORK_REPRODUCTION_STATUS=`, `NUMERICAL_REPRODUCTION_STATUS=`, and `PAPER_REPRODUCTION_STATUS=`. A numerical gap must remain assigned to one of the permitted classifications, never fixed secretly.

**FAILURE_CONDITIONS:** any unclosed core gate, insufficient evidence, protocol drift, or unsupported pass claim.

**NEXT_PHASE_ENTRY_CONDITION:** none; transition to innovation requires a Human-approved Master Plan change request.

## TimesNet deferral rule

`TIMESNET_IMPLEMENTATION_CAN_BE_DEFERRED=True`

`TIMESNET_REQUIRED_FOR_FINAL_REPRODUCTION=True`

Phases 3–4 may establish STL, iTransformer, and a clearly temporary branch scaffold to reduce debugging ambiguity. Before Phase 5 replaces every temporary Trend/Seasonal branch, `PAPER_FRAMEWORK_REPRODUCTION=PASS` is forbidden.

## SCF-GAN integration rule

SCF-GAN is a core paper component. A non-augmented temporal pipeline may be debugged first, because joint GAN/predictor failure is non-diagnostic. It is nevertheless not the complete paper framework. Final reproduction must restore and independently validate the paper-defined SCF-GAN role, including source battery, target battery, real/generated samples, and conditioning definition.

## Classical baseline prohibition

Ridge, linear regression, random forest, XGBoost, SVR, GPR, CNN, LSTM, and generic Transformer may be comparison baselines or paper-listed ablations only. None may be substituted for any required paper component.

`CLASSICAL_BASELINE_SUCCESS DOES_NOT_IMPLY PAPER_REPRODUCTION_SUCCESS`

`RIDGE_IS_REPRODUCTION_MODEL=False`

## Old-project isolation and forensic rule

Ridge ROOT, PTOC, old STL, old GAN, old fusion, and old experimental protocols are `HISTORICAL_REFERENCE_ONLY`. They must not be copied or treated as paper-correct by default.

After Phase 0 has frozen the paper specification, any consultation of old code must first create `OLD_CODE_VS_PAPER_SPEC_AUDIT` and classify each inspected element as `MATCH`, `MISMATCH`, or `UNKNOWN`. No old result, metric, split, or conclusion transfers into this project without fresh local evidence.

`OLD_PROJECT_IMPORTED=False`

`PTOC_ACTIVE=False`

## Paper ambiguity register contract

Every paper ambiguity must be appended to `PAPER_AMBIGUITY_REGISTER.md` using this exact record structure:

```text
PAPER_AMBIGUITY_ID=
PAPER_TEXT=
PAPER_LOCATION=
WHAT_IS_MISSING=
POSSIBLE_INTERPRETATIONS=
DEFAULT_ENGINEERING_CHOICE=
WHY_THIS_CHOICE=
NUMERICAL_REPRODUCTION_IMPACT=
HUMAN_APPROVAL_REQUIRED=True/False
RESOLUTION_STATUS=OPEN/APPROVED/RESOLVED_BY_NEW_EVIDENCE
```

Initial required entries include: the apparent tension between full target chronological 60/20/20 splitting and the “first 20% of total cycle life” small-sample description; exact source/target roster for every result table; exact partial-voltage handling; dataset-specific SOH capacity denominator; STL period/seasonal/trend windows; TimesNet widths/blocks/top-k periods; iTransformer widths/layers/heads; GAN latent and hidden dimensions/epochs; exact augmentation insertion/split behavior; and the paper’s differing conditioning descriptions (cycle-period condition versus voltage range/battery characteristics/usage conditions).

No `DEFAULT_ENGINEERING_CHOICE` is a paper parameter. Any choice affecting strict numerical reproduction must be reported separately.

## Leakage and deviation control

All split creation precedes fitting of scalers, decompositions, feature selection, augmentation, early stopping, or architecture selection. The test partition is held for the frozen evaluation protocol only. If a post-result change is needed, create a `REPRODUCTION_DEVIATION_RECORD` containing the original protocol, change, reason, affected evidence, and `THIS_IS_NOT_STRICT_REPRODUCTION=True`. It cannot overwrite strict-reproduction results.

## Git and reproducibility policy

This repository was initialized as Git during Master Plan creation. Its canonical local and GitHub repository name is `soh_gmtc_reproduction_clean`; the required GitHub visibility is `PRIVATE`. It must have a wholly new history and must never use, push to, or inherit an old `soh2_1` remote. Force push, history rewriting, destructive reset of recorded work, and deleting failed experiments to sanitize history are prohibited.

`.gitignore` excludes raw/derived data, copyrighted local PDFs, checkpoints, caches, temporary outputs, local environments, IDE cache, credentials, API keys, and tokens. Do not commit large raw data, model checkpoints, caches, secret material, or the local paper PDF. Instead, version `references/README.md` with paper title, DOI, local filename, source, and retrieval/availability date. Data provenance, manifests, processing code, and expected directory layout must be versioned so data can be lawfully reacquired.

Each Codex task requires a meaningful Git commit after a diff/sensitive-file audit, including a failed experiment or failed gate. Commit messages must state scientific/governance purpose (for example `plan: create strict GMTC reproduction master plan`); vague messages such as `update`, `fix`, or `changes` are prohibited. Each passed Phase additionally records `PHASE_START_COMMIT`, `PHASE_COMPLETION_COMMIT`, `PHASE_COMPLETION_PUSH_STATUS`, and `PHASE_REMOTE_BACKUP_CONFIRMED` in the phase ledger. Every runnable artifact must declare code revision, environment, dependencies, seed(s), data-manifest version, protocol-manifest version, and configuration hash.

Git initialization and plan creation are governance setup, not a completed scientific Phase. A missing or failed remote backup is not a scientific-gate failure, but it prevents the affected task from being called fully archived or from enabling a next step that depends on its formal archive.

## Mandatory Git/GitHub archive protocol

Before every task, read this plan and run a read-only Git audit: `git status`, current branch, recent commits, and `git remote -v`; check remote synchronization only when a remote is configured and reachable. Emit `GIT_AUDIT_BEFORE_RUN` with `CURRENT_BRANCH`, `CURRENT_HEAD`, `WORKTREE_STATUS`, `REMOTE_NAME`, `REMOTE_URL`, `REMOTE_REPOSITORY`, `REMOTE_PRIVATE_STATUS`, `UNPUSHED_COMMIT_COUNT`, and `PUSH_PENDING`. Uncommitted changes from an earlier task may not be silently mixed into a new task.

After a permitted task: re-read this plan; update only allowed status/progress/evidence/log content; audit the intended diff and sensitive-file exclusions; add only intended files; commit with a meaningful message; then push `main` to `origin` without force. Verify that local `HEAD` equals the remote backup commit. The first repository archive is complete only after the new private GitHub repo exists, `origin` points to it, the Master Plan commit has been pushed, and remote `main` has been verified.

If GitHub repository `soh_gmtc_reproduction_clean` already exists, verify it belongs to this project and is empty/current-project-only before use. If it contains unrelated history, stop with `GITHUB_REPOSITORY_NAME_CONFLICT`; never delete, overwrite, force-push, or rewrite it. If a push fails, preserve the local commit and output `LOCAL_COMMIT_SAFE=True`, `REMOTE_PUSH_STATUS=FAILED`, `PUSH_PENDING=True`, `BACKUP_BLOCKER=<exact error>`, and `STEP_ARCHIVE_STATUS=LOCAL_COMPLETE__REMOTE_PENDING`. Do not alter system certificates, security settings, VPN/firewall, authentication files, private keys, or tokens to work around the failure.

Every final task report must include `PLAN_READ_BEFORE_RUN`, `PLAN_AUDIT_BEFORE_RUN`, `CURRENT_PHASE`, `CURRENT_GATE`, `TASK_COMPLETED`, `GATE_RESULT`, `PLAN_READ_AFTER_RUN`, `PLAN_AUDIT_AFTER_RUN`, `MASTER_PLAN_PROGRESS_UPDATED`, local Git status/heads/current-run commit, remote repository/private/URL/push/head verification status, `PUSH_PENDING`, `STEP_ARCHIVE_STATUS`, and `NEXT_ACTION`.

Every displayed reproduction progress block must be immediately followed by this archive block:

```text
========================================
GIT / GITHUB ARCHIVE STATUS
========================================
LOCAL_REPOSITORY=True
CURRENT_BRANCH=main
CURRENT_HEAD=
CURRENT_RUN_COMMIT=
GITHUB_REPOSITORY=soh_gmtc_reproduction_clean
GITHUB_REPOSITORY_PRIVATE=True
REMOTE_NAME=origin
REMOTE_PUSH_STATUS=
PUSH_PENDING=
LAST_REMOTE_BACKUP_COMMIT=
STEP_ARCHIVE_STATUS=
========================================
```

If `CURRENT_RUN_COMMIT != LAST_REMOTE_BACKUP_COMMIT`, state `CLOUD_BACKUP_OUT_OF_SYNC=True` and never state `STEP_ARCHIVE_STATUS=COMPLETE`.

## Master Plan modification rule

Current status, progress, completion evidence, and the append-only log may be updated as work proceeds. Altering phase order, deleting a gate, changing a reproduction definition, changing the paper architecture, or changing data protocol requires:

```text
MASTER_PLAN_CHANGE_REQUEST=True
HUMAN_AUTHORIZATION_REQUIRED=True
```

Until authorized, preserve this plan and stop conflicting work.

## PLAN_CHANGE_LEDGER (append-only)

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0001
TRIGGER=Human-authorized methodology hardening and permanent plan-health governance
OLD_ASSUMPTION=Existing phase gates and generic before/after plan audit were sufficient without an explicit evidence hierarchy, controlled-search policy, or deviation-health mechanism.
NEW_EVIDENCE=Methodological review found missing/weak explicit requirements for structural-first validation, simple baseline sanity checks, multi-seed reporting, test lock, ablation, small-subset overfit tests, hyperparameter-search control, and plan-health escalation.
IMPACT=Governance and acceptance evidence are strengthened; Phase 0–10 order, paper architecture, data protocol, and reproduction definitions remain unchanged.
OLD_PLAN=Version 1.0.0
NEW_PLAN=Version 1.1.0
WHY_CHANGE_NECESSARY=Prevent optimization or favorable scores from being mistaken for paper fidelity or credible reproduction.
RESULT_DRIVEN=False
HUMAN_AUTHORIZED=True
COMMIT=dcb24dd841a2dc0917bb36f3c65c76dab790b7b0
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0002
NAME=ALLOW_SPLIT_INDEPENDENT_DATA_READINESS_WHILE_PA004_BLOCKED
TRIGGER=Human Researcher authorization to eliminate avoidable idle time while preserving PA-004 as a hard protocol blocker.
OLD_ASSUMPTION=Phase 0 ambiguity blocking prohibited all data handling before Phase 1.
NEW_EVIDENCE=Raw-data acquisition, provenance inventory, deterministic parsing, chronology checks, and feature constructability can be performed without choosing Candidate A/B or assigning any split-derived role.
IMPACT=Adds a constrained Phase-0 side-track only. It does not change the paper architecture, phase order, SOH target definition, train/validation/test population, small-sample definition, GAN target allocation, or evaluation population.
OLD_PLAN=Version 1.1.0
NEW_PLAN=Version 1.2.0
WHY_CHANGE_NECESSARY=Permit protocol-independent preparation while retaining an explicit non-advancement boundary and PA-004 hard block.
RESULT_DRIVEN=False
HUMAN_AUTHORIZED=True
COMMIT=5992f3050b90a813e1bc084fb013d233d3458a5d
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0003
NAME=ALLOW_DUAL_PROTOCOL_REPRODUCTION_WHILE_PRIMARY_SOURCE_CLARIFICATION_PENDING
TRIGGER=Human Researcher authorized a result-blind PA-001 nominal-capacity fallback and two isolated PA-004 protocol tracks after contacting the corresponding author.
OLD_ASSUMPTION=PA-001 and PA-004 blocked all Phase 1 entry until primary-source clarification.
NEW_EVIDENCE=Human decision: proceed transparently with nominal/rated-capacity SOH labels and separate Section 4 versus Section 4.2 protocol tracks while preserving author override priority.
IMPACT=Permits Phase 1 protocol-contract freeze only; does not alter architecture, Phase 2–10 order, final reproduction definition, or authorize model work. PA-002 and PA-005 remain explicit row-assignment/augmentation-execution blockers.
OLD_PLAN=Version 1.2.0
NEW_PLAN=Version 1.3.0
WHY_CHANGE_NECESSARY=Avoid indefinite research pause while keeping unconfirmed interpretations, isolation boundaries, and primary-source override visible.
RESULT_DRIVEN=False
HUMAN_AUTHORIZED=True
COMMIT=65ad49b098112a06e4ead0e43622644e3e8d471f
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0004
NAME=FREEZE_PROTOCOL_B_EVALUATION_POPULATION_WITH_GENERAL_HOLDOUT_BOUNDARIES
TRIGGER=PROTOCOL_B_VALIDATION_TEST_POPULATION_UNDEFINED
LEVEL=2
OLD_ASSUMPTION=The dual-track fallback was executable despite Protocol B lacking a stated validation/test population.
NEW_EVIDENCE=Human Researcher authorized a result-blind nested Protocol B evaluation fallback: [0,20%) train-real, [20,60%) unused, [60,80%) validation, [80,100%) test, with deterministic floor boundaries.
IMPACT=Completes Phase 1 source-keyed role assignment and leakage audit. It changes neither paper architecture nor phase order; it creates no model, synthetic data, target labels, feature values, or results. The fallback is explicitly non-paper-exact and remains subject to primary-source override.
OLD_PLAN=Version 1.3.0
NEW_PLAN=Version 1.4.0
WHY_CHANGE_NECESSARY=Protocol B could not be evaluated or leakage-audited honestly without explicit holdout semantics. The authorized choice preserves the paper’s 60%/80% general holdout boundaries and prevents [20%,60%) from silently expanding the small-sample training set.
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
TEST_LOADED=False
HUMAN_AUTHORIZED=True
COMMIT=36c7836fc2fa52cb3b1591f070b91b774dcaed85
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0007
NAME=ALLOW_SYNTHETIC_ONLY_GMTC_FRAMEWORK_SCAFFOLD_WHILE_REAL_DATA_FIDELITY_REMAINS_BLOCKED
LEVEL=1
TRIGGER=PHASE_4A_GMTC_SYNTHETIC_FRAMEWORK_SCAFFOLD_SPECIFICATION_AND_INTERFACE_CONTRACT
OLD_ASSUMPTION=Phase 4 had a single undifferentiated gate, risking either an unnecessary stop before interface verification or an overclaim from a scaffold.
NEW_EVIDENCE=Human authorized a result-blind, performance-blind synthetic-only routing and interface scaffold while retaining all real-data and paper-fidelity blockers.
IMPACT=Adds a Phase 4 structural-vs-real-data dual gate only. It does not change architecture, data protocol, phase order, reproduction definitions, model logic, or evaluation population.
OLD_PLAN=Version 1.4.3
NEW_PLAN=Version 1.4.4
WHY_CHANGE_NECESSARY=Separate contract correctness from unresolved real-data paper fidelity without allowing substitute branches, fusion, training, or claims of reproduction.
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
TEST_LOADED=False
HUMAN_AUTHORIZED=True
COMMIT=PENDING_CURRENT_RUN
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0008
NAME=ALLOW_SYNTHETIC_STANDARD_TIMESNET_CORE_WHILE_BRANCH_VARIANT_PARAMETERS_REMAIN_UNRESOLVED
LEVEL=1
TRIGGER=PHASE_5A_TIMESNET_SPECIFICATION_AND_SYNTHETIC_STANDARD_CORE_IMPLEMENTATION
OLD_ASSUMPTION=Phase 5 had a single undifferentiated gate, which could conflate a generic structural algorithm test with the paper's unspecified deep and lightweight branch variants.
NEW_EVIDENCE=Human authorized a result-blind, performance-blind standard-core synthetic side-track after evidence recovery confirmed the algorithmic core but not target-paper variant parameters or real-data semantics.
IMPACT=Adds a structural-vs-branch-fidelity dual gate only. It changes neither paper architecture, data protocol, phase order, research goal, reproduction definition, nor evaluation population.
OLD_PLAN=Version 1.4.4
NEW_PLAN=Version 1.4.5
WHY_CHANGE_NECESSARY=Allow isolated verification of standard FFT/2D/Inception routing without representing the generic implementation as the target paper's unspecified Trend or Seasonal model.
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
REAL_DATA_USED=False
TEST_LOADED=False
HUMAN_AUTHORIZED=True
COMMIT=PENDING_CURRENT_RUN
```

```text
PLAN_CHANGE_ID=PLAN-CHANGE-0009
NAME=ALLOW_SYNTHETIC_PRECONSTRUCTED_FUSION_FEATURE_STRUCTURAL_IMPLEMENTATION
LEVEL=1
TRIGGER=PHASE_6A_ADAPTIVE_FUSION_SPECIFICATION_AND_SYNTHETIC_STRUCTURAL_IMPLEMENTATION
OLD_ASSUMPTION=Phase 6 had a single undifferentiated gate that could conflate a generic fusion topology test with its missing real prediction/error/ratio semantics.
NEW_EVIDENCE=The paper specifies dual attention, Bi-LSTM and key Table 4 widths, but omits error/ratio definitions, d, feature ordering, real loss terms/lambda, and final-domain conversion.
IMPACT=Adds a synthetic-preconstructed-feature structural-vs-real-data-fidelity dual gate only. It changes neither core architecture, data protocol, research goal, phase order, reproduction definition, nor evaluation population.
OLD_PLAN=Version 1.4.5
NEW_PLAN=Version 1.4.6
WHY_CHANGE_NECESSARY=Allow contract testing without inventing real feature semantics or leaking targets.
RESULT_DRIVEN=False
PERFORMANCE_DRIVEN=False
REAL_DATA_USED=False
TEST_LOADED=False
HUMAN_AUTHORIZED=True
COMMIT=PENDING_CURRENT_RUN
```

## Phase completion ledger

| Phase | Gate status | Completion evidence | PHASE_START_COMMIT | PHASE_COMPLETION_COMMIT | PHASE_COMPLETION_PUSH_STATUS | PHASE_REMOTE_BACKUP_CONFIRMED |
|---|---|---|---|---|---|---|
| 0 | PASS_WITH_REGISTERED_PRIMARY_SOURCE_AMBIGUITY | Paper specification plus Human-authorized PA-001 fallback and PA-004 dual-track protocol record | f354af6 | 65ad49b | PASS | True |
| 1 | PASS__PRIMARY_SOURCE_OVERRIDE_PENDING | Frozen A/B protocols, source-keyed metadata membership manifest, population-conservation and leakage audits, and augmentation plan; no test values or synthetic rows materialized | 65ad49b | 36c7836 | PASS | True |
| 2 | STRUCTURAL_IMPLEMENTATION_COMPLETE__REAL_DATA_FIDELITY_BLOCKED | Capacity-domain STL input cross-evidence; Phase 2C period-method audit; and a parameterized, synthetic-only STL module with explicit-period and fail-closed structural tests. No real NASA/CALCE period or decomposition is authorized. | 27e4915 | pending Phase 2 real-fidelity evidence | `STL_STRUCTURAL_IMPLEMENTATION_GATE=PASS`; `STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD` | False |
| 3 | STRUCTURAL_IMPLEMENTATION_COMPLETE__REAL_DATA_FIDELITY_BLOCKED | Target-paper iTransformer evidence, Fig.5 residual-routing audit, standard-vs-target boundary, parameter evidence, and a parameterized standard core verified only on generic synthetic multivariate fixtures. No real residual input or paper target alignment was selected. | pending real-data evidence | pending real-data evidence | `ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=PASS`; real-data gate inherits STL period and iTransformer ambiguity blockers | False |
| 4 | STRUCTURAL_IMPLEMENTATION_COMPLETE__REAL_DATA_FIDELITY_BLOCKED | Synthetic-only component-routing/lineage interfaces, unimplemented TimesNet ports, a Phase 3 iTransformer residual adapter, future-fusion schema, and alignment/provenance tests. No TimesNet, fusion, SCF-GAN, real battery data, or training. | pending real-data evidence | pending Phase 4 real-fidelity evidence | `FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE=PASS`; real-data gate inherits STL, branch, fusion, and orchestration blockers | False |
| 5 | STANDARD_CORE_STRUCTURALLY_IMPLEMENTED__BRANCH_FIDELITY_BLOCKED | Target-paper evidence/semantics audit and a parameterized standard TimesNet reference core tested on synthetic periodic fixtures. No real STL component, Trend-deep variant, Seasonal-lightweight variant, fusion, or real-data training. | pending real-data evidence | pending Phase 5 branch-fidelity evidence | `TIMESNET_STANDARD_CORE_STRUCTURAL_GATE=PASS`; `TIMESNET_BRANCH_VARIANT_FIDELITY_GATE=BLOCKED_BY_INPUT_TARGET_AND_VARIANT_PARAMETERS` | False |
| 6 | STRUCTURAL_FUSION_IMPLEMENTED__REAL_DATA_FIDELITY_BLOCKED | Paper-evidence/dimension audit and a synthetic-only parameterized fusion topology tested on preconstructed B×T×9 feature fixtures. No real branch prediction, error/ratio builder, battery data, model metric, or test access. | pending real-data evidence | pending Phase 6 real-fidelity evidence | `ADAPTIVE_FUSION_STRUCTURAL_GATE=PASS`; `ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_FEATURE_SEMANTICS_LOSS_AND_BRANCH_ALIGNMENT` | False |
| 7 | NOT_STARTED | — | — | — | — | False |
| 8 | NOT_STARTED | — | — | — | — | False |
| 9 | NOT_STARTED | — | — | — | — | False |
| 10 | NOT_STARTED | — | — | — | — | False |

## Pre-write ten-audit record

These ten independent plan audits were completed before this Master Plan was written. They audit the plan, not an implementation.

| AUDIT_NAME | PASS/FAIL | PROBLEM_FOUND | REVISION_MADE |
|---|---|---|---|
| AUDIT_01_PAPER_FIDELITY | PASS | Public summary alone would under-specify the fusion and protocol. | Used the local full-text PDF; named Bi-LSTM fusion, 10-step branch statement, source/target examples, and explicit paper evidence authority. |
| AUDIT_02_DATA_PROTOCOL | PASS | The paper contains multiple dataset and small-sample descriptions requiring execution-level reconciliation. | Made Phase 0/1 manifests and ambiguity records mandatory before data/model work. |
| AUDIT_03_NO_ASSUMPTION | PASS | Core dimensions/periods and some condition definitions are not completely specified. | Marked unverified items as ambiguities; prohibited presenting defaults as paper facts. |
| AUDIT_04_MODULAR_DEBUGGABILITY | PASS | A full GAN-plus-three-branch build would obscure failure origin. | Ordered STL, iTransformer, scaffold, TimesNet, fusion, GAN, then integration. |
| AUDIT_05_LEAKAGE_SAFETY | PASS | Chronological target split and augmentation can leak future/test information. | Required split-first manifests, train-only fitting, source-only GAN constraints, and locked-test prohibition. |
| AUDIT_06_IMPLEMENTATION_ORDER | PASS | Full model first would make component defects non-identifiable. | Added phase-local tensor, reconstruction, routing, and independent GAN gates. |
| AUDIT_07_REPRODUCTION_DEFINITION | PASS | “Runs” could be confused with faithful numerical reproduction. | Defined Framework versus Numerical Reproduction and restricted pass status to Phase 10. |
| AUDIT_08_METRIC_ALIGNMENT | PASS | A different metric population/protocol could create misleading comparison. | Locked RMSE/MAE/MAPE/R² and paper-vs-ours comparison to Phase 9 frozen protocol. |
| AUDIT_09_REPRODUCIBILITY | PASS | Untracked environments, data versions, seeds, and commits prevent credible replication. | Required manifests, configuration hashes, environment capture, ten-run evidence where applicable, Git, and phase commits. |
| AUDIT_10_SCOPE_CONTROL | PASS | Legacy PTOC/Ridge work or early innovation could contaminate this project. | Added hard old-project isolation, baseline prohibition, deviation handling, and Human-authorized plan-change rule. |

`TEN_AUDIT_PASSES_COMPLETED=True`

## Append-only run log

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 0 | Plan did not yet exist; user-authorized governance initialization only. | Created this Master Plan, initialized Git, and created `.gitignore`; no data/model/experiment action. | Plan self-audit required below. | Phase 0 remains NOT_STARTED. | Local 17-page PDF inspected; this file. |
| 2026-09-11 | Phase 0 | Git/GitHub archive rules authorized; governance work is allowed in Phase 0. | Added mandatory Git/GitHub archive protocol, ignored local copyrighted PDF, created versioned paper metadata, created the new private GitHub repository, and pushed the initial plan commit; no data/model/experiment action. | Local/remote `main` commit `86a47e7` verified; current status-record update pending. | Phase 0 remains NOT_STARTED. | `origin=https://github.com/wacenhong-art/soh_gmtc_reproduction_clean.git`; private repository and remote `main` verified. |
| 2026-09-11 | Phase 0 | Full plan, Git state, project files, and local 17-page paper re-read; Plan Health Audit found governance gaps but no paper/protocol/architecture invalidation. | Added validation hierarchy, controlled-search policy, module and small-subset checks, baseline/multi-seed/test-lock/ablation rules, plan-health/deviation/change-history governance. No Phase 0 extraction, model code, training, search, or experiment. | Pending final self-audit and archive of this governance amendment. | Phase 0 remains NOT_STARTED. | `PLAN-CHANGE-0001`; Human-authorized; result-driven=False. |
| 2026-09-11 | Phase 0 | Full plan/Git/project/PDF audit and Plan Health Audit passed. | Extracted paper specification, ambiguity register, missing-parameter register, and machine-readable JSON. No code/model/training/search/metric run. | Specification complete; advancement blocked by PA-004 until Human/new-evidence resolution. | PASS_WITH_REGISTERED_AMBIGUITIES__ADVANCEMENT_BLOCKED. | PDF pp.1–16; `PA-001`–`PA-010`; `PM-001`–`PM-009`. |
| 2026-09-11 | Phase 0 | Full plan/specification/registers/PDF audit; task is Phase-0 evidence work and allowed. | Audited all PA-004-relevant full-text passages, visually inspected the adjacent architecture/results pages, checked tables/captions, project supplementary inventory, and paper-provided code/data availability. No data/model/training/search/metric run. | No new evidence maps the two target-data constructions; no core-plan change is warranted. | `PA004_STATUS=UNRESOLVED_BLOCKING`; Phase 0 exit remains blocked. | `PA004_PROTOCOL_EVIDENCE_TABLE.md`; PDF pp.5, 7–11, 12–16. |
| 2026-09-11 | Phase 0 side-track | Human authorized PLAN-CHANGE-0002; this task is split-independent raw-data readiness only. | Acquired and hashed official NASA/CALCE archives; parsed NASA fully; verified CALCE workbook presence/header schema and raw-record dimensions; created an unsplit NASA canonical artifact and complete provenance manifests. No split assignment, model, STL, GAN, training, metric, or search. | CALCE full-row parsing exceeds the bounded runtime and remains incomplete; no data or cycle was discarded. | `SPLIT_INDEPENDENT_DATA_READINESS_GATE=FAIL__PARTIAL`; `PA004_STATUS=UNRESOLVED_BLOCKING`. | `DATA_SOURCE_MANIFEST.md`; `output/data_readiness/`; `scripts/prepare_split_independent_data.py`. |
| 2026-09-11 | Phase 0 side-track | Previous remote archive verified before work. | Read every operational row in all four official CALCE archives and generated a separate full-row inventory. No raw-file change, split assignment, target choice, model, STL, GAN, training, metric, or search. | CALCE numeric Cycle_Index may span/ repeat across source workbooks; global cycle identity and persisted row-null/content-duplicate reporting remain unresolved and are not inferred. | `CALCE_FULL_ROW_SCAN_STATUS=COMPLETE`; readiness gate remains fail/partial; `PA004_STATUS=UNRESOLVED_BLOCKING`. | `CALCE_FULL_ROW_INVENTORY.md`; `output/data_readiness/calce_full_row_inventory.json`. |

## MASTER_PLAN_SELF_AUDIT

`MASTER_PLAN_SELF_AUDIT=PASS`

The plan contains: the local-paper authority and status; all 10 required audits; distinct framework and numerical definitions; a five-level structural-to-controlled-optimization validation hierarchy; Phases 0–10 with objective, input, allowed/forbidden actions, artifacts, acceptance, failures, and entry conditions; module-level and small-subset sanity gates; leakage/test-lock and multi-seed rules; paper-driven ablation policy; hyperparameter-search restrictions; mandatory before/after plan-health audits; deviation escalation and append-only plan-change history; baseline prohibition; Git/reproducibility policy; and append-only state tracking. It intentionally authorizes no Phase 0 model code.

## Initial status declaration

```text
MASTER_PLAN_CREATED=True
MASTER_PLAN_COMMITTED=True
MASTER_PLAN_PUSHED=True
GITHUB_CLOUD_BACKUP_CONFIRMED=True
PROJECT_INITIALIZATION_STATUS=PASS
PAPER_FULLTEXT_STATUS=AVAILABLE
TEN_AUDIT_PASSES_COMPLETED=True
MASTER_PLAN_SELF_AUDIT=PASS
PLAN_HEALTH_STATUS=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_DEFECT_FOUND=False
PLAN_CHANGE_REQUIRED=False
CORE_ASSUMPTIONS_STILL_VALID=True
PAPER_ALIGNMENT_STILL_VALID=True
DATA_PROTOCOL_STILL_VALID=True
LEAKAGE_STATUS=NOT_ASSESSED_NO_DATA_OR_EXECUTION
CURRENT_REPRODUCTION_STRATEGY_STILL_VALID=True
NEXT_STEP_STILL_OPTIMAL=True
CURRENT_PHASE=Phase 0
CURRENT_GATE=Phase 0 advancement review: PA-004
EXPERIMENT_RUN=False
MODEL_IMPLEMENTED=False
MODEL_TRAINED=False
OLD_PROJECT_IMPORTED=False
PTOC_ACTIVE=False
RIDGE_IS_REPRODUCTION_MODEL=False
TIMESNET_REQUIRED_FOR_FINAL_REPRODUCTION=True
SCF_GAN_REQUIRED_FOR_FINAL_REPRODUCTION=True
LARGE_GRID_SEARCH_AUTHORIZED=False
FIRST_FULL_REPRODUCTION_COMPLETED=False
TEST_LOADED=False
PHASE_0_STATUS=PASS_WITH_REGISTERED_AMBIGUITIES__ADVANCEMENT_BLOCKED
PHASE_0_PROGRESS=100%
ADVANCEMENT_ALLOWED=False
NEXT_ACTION=Complete bounded CALCE full-row inventory if resources permit, while Human obtains PA-004 primary-source clarification; do not enter Phase 1 or implement model code.
```

## Append-only run log — CALCE final readiness audit

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 0 side-track | Existing PLAN-CHANGE-0002 permits only split-independent preparation; PA-001 and PA-004 remain blocking. | Completed source-keyed CALCE cycle identity, contiguous boundary, all-sheet column-null, and layered duplicate audits. No raw-file mutation, split assignment, target choice, model, training, metric, or search occurred. | Paper HFs HF1/HF2/HF3 remain reconstructible from time/current/voltage. CALCE has no temperature field, but temperature is not a specified paper-HF input. Cross-workbook numeric Cycle_Index is not global and is retained with source keys. | `SPLIT_INDEPENDENT_DATA_READINESS_GATE=PASS`; Phase 0 exit remains blocked by PA-004. | `CALCE_CYCLE_IDENTITY_AUDIT.md`; `CALCE_NULL_DUPLICATE_AUDIT.md`; `output/data_readiness/data_readiness_final_status_v3.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
PA001_STATUS=AMBIGUOUS_BLOCKING_TARGET_CONSTRUCTION
PA004_STATUS=UNRESOLVED_BLOCKING
DATA_READINESS_GATE=PASS
DATA_READINESS_STATUS=PASS__PAPER_HF_FEATURES_FEASIBLE__TEMPERATURE_NOT_A_PAPER_HF_INPUT
ADVANCEMENT_TO_PROTOCOL_FREEZE=False
NEXT_ACTION=PRIMARY_SOURCE_PROTOCOL_CLARIFICATION for PA-001 and PA-004; do not enter Phase 1 or implement model code.
```

## Append-only run log — primary-source clarification package

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 0 blocked | PA-001 and PA-004 remain unresolved; documentation-only clarification preparation is allowed. | Re-read paper and registers; identified the corresponding author from the target-paper title page; prepared a two-question English draft, Chinese reference translation, evidence packet, and Human-only result-blind fallback framework. No contact, protocol selection, split, model, training, metric, or search occurred. | Waiting for Human review/send is the correct state. Additional Phase-1 blockers PA-002 and PA-005 remain visible, rather than deferred silently. | Phase 0 exit remains blocked. | `AUTHOR_CLARIFICATION_REQUEST.md`; `PRIMARY_SOURCE_CLARIFICATION_PACKET.md`; `PRIMARY_SOURCE_FALLBACK_POLICY.md`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
AUTHOR_REPLY_STATUS=NOT_CONTACTED
PA001_STATUS=AMBIGUOUS_BLOCKING_TARGET_CONSTRUCTION
PA004_STATUS=UNRESOLVED_BLOCKING
PHASE1_BLOCKER_SET=PA-001, PA-002, PA-004, PA-005
NEXT_ACTION=HUMAN_REVIEW_AND_SEND_AUTHOR_CLARIFICATION
```

## Append-only run log — dual-protocol fallback freeze

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 0 to Phase 1 entry | Human authorized Level-1 amendment; data readiness is pass; PA-001/PA-004 are retained as primary-source-pending interpretations. | Froze nominal/rated-capacity SOH fallback and two isolated, paper-section-specific protocol contracts. Created only declarative A/B manifests because PA-002 and PA-005 still preclude honest row-level role assignment/augmentation execution. No STL, model, training, metrics, or search. | Dual-track protocol freeze is coherent and result-blind. Author reply overrides fallback if materially different. | `PHASE_0_EXIT_GATE=PASS_WITH_REGISTERED_PRIMARY_SOURCE_AMBIGUITY`; Phase 1 protocol freeze is in progress. | `REPRODUCTION_PROTOCOLS.md`; `output/protocol/`. |

```text
AUTHOR_REPLY_STATUS=CONTACTED_WAITING
PA001_STATUS=FALLBACK_FROZEN__PRIMARY_SOURCE_PENDING
PA004_STATUS=AMBIGUOUS__DUAL_TRACK_FROZEN__PRIMARY_SOURCE_PENDING
PHASE_0_EXIT_GATE=PASS_WITH_REGISTERED_PRIMARY_SOURCE_AMBIGUITY
ADVANCEMENT_TO_PHASE_1=True
PHASE_1_STATUS=IN_PROGRESS__PROTOCOL_FREEZE
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=FREEZE_PA002_EXPERIMENT_ROSTER_AND_PA005_AUGMENTATION_INSERTION_BEFORE_ROW_ASSIGNMENT_OR_PHASE_1_ACCEPTANCE
```

## Append-only run log — Phase 1 blocker resolution and fallback protocol freeze

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 1 protocol freeze | Existing PLAN-CHANGE-0003 already includes PA-002/PA-005 in the Phase 1 blocker set; no duplicate plan change was needed. | Recorded Human-confirmed Gmail contact status; reconstructed experiment-specific paper rosters; froze PA-005’s explicit limitation with a result-blind original-plus-five-synthetic-set fallback; created field-provenanced Phase 1 protocol specification. No row split, STL, GAN training, model, metric, or search occurred. | All four named protocol blockers are resolved from paper context or frozen as limitations. Phase 1 remains incomplete pending exact source-keyed manifests and leakage checklist. | `PHASE_0_EXIT_GATE=PASS_WITH_REGISTERED_PRIMARY_SOURCE_AMBIGUITIES`; `PHASE_1_STATUS=PROTOCOL_FREEZE_IN_PROGRESS`. | `PHASE1_REPRODUCTION_PROTOCOL_SPEC.md`; `output/protocol/phase1_reproduction_protocol_spec.json`. |

```text
AUTHOR_CONTACT_STATUS=CONTACTED
AUTHOR_REPLY_STATUS=CONTACTED_WAITING
GMAIL_MESSAGE_ID=1a0906ec8a4ae625
PA001_STATUS=FALLBACK_FROZEN_WITH_LIMITATION
PA002_STATUS=RESOLVED_AS_EXPERIMENT_SPECIFIC_ROSTER
PA004_STATUS=DUAL_PROTOCOL_FALLBACK_FROZEN_WITH_LIMITATION
PA005_STATUS=FALLBACK_FROZEN_WITH_LIMITATION
PHASE1_BLOCKER_SET_AFTER=NONE__ALL_NAMED_BLOCKERS_RESOLVED_OR_FALLBACK_FROZEN
NEW_PHASE1_BLOCKER_FOUND=False
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=CREATE_EXACT_SOURCE_KEYED_SAMPLE_MANIFESTS_AND_LEAKAGE_CHECKLIST_WITHOUT_MODEL_WORK
```

## Append-only run log — Phase 1 exact sample manifests and leakage freeze

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 1 blocked | Existing fallback was audited for row-level executability before assignment. | Confirmed Protocol B defines its first-20% original training population but has no validation/test population; stopped prior to any membership assignment. Created blocked-manifest, leakage-policy, and augmentation-plan artifacts, plus a pending Human change request. No raw data mutation, split assignment, synthetic rows, STL, model, training, metrics, or search. | New protocol defect is material: Phase 1 cannot pass without a result-blind Human decision or author clarification. | `PHASE_1_GATE=FAIL__BLOCKED_BEFORE_ROW_ASSIGNMENT`; advancement is stopped. | `PHASE1_SAMPLE_MANIFEST.md`; `PHASE1_LEAKAGE_CHECKLIST.md`; `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0004.md`. |

```text
NEW_PHASE1_BLOCKER_FOUND=True
BLOCKER_ID=PROTOCOL_B_VALIDATION_TEST_POPULATION_UNDEFINED
PHASE_1_GATE=FAIL__BLOCKED_BEFORE_ROW_ASSIGNMENT
PHASE_1_STATUS=BLOCKED
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=FAIL__MAJOR_PROTOCOL_DEFECT
PLAN_DEVIATION_LEVEL=2
PLAN_CHANGE_REQUIRED=True
PLAN_CHANGE_ID=PLAN-CHANGE-0004
ADVANCEMENT_ALLOWED=False
NEXT_ACTION=AWAIT_HUMAN_RESULT_BLIND_DECISION_OR_PRIMARY_SOURCE_CLARIFICATION_FOR_PROTOCOL_B_EVALUATION_POPULATION
```

## Append-only run log — PLAN-CHANGE-0004 authorization and Phase 1 freeze

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 1 protocol freeze | Pending PLAN-CHANGE-0004 identified a material, result-blind Protocol B evaluation-population defect. | Applied Human-authorized Level-2 amendment: Protocol B is [0,20%) train-real, [20,60%) unused, [60,80%) validation, [80,100%) test. Built source-keyed metadata-only membership manifests for E1 NASA B0007/B0018 and CALCE CS2_36/CS2_38, froze augmentation boundaries, and mechanically audited conservation, overlap, ordering, and protocol isolation. No raw data modification, feature/target materialization, synthetic rows, STL, model, training, metric, or search occurred. | The fallback is internally coherent, result/performance blind, source-traceable, and test-locked. Primary-source override remains active. | `PHASE_1_GATE=PASS`; Phase 2 is next but not started. | `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0004.md`; `PHASE1_SAMPLE_MANIFEST.md`; `PHASE1_LEAKAGE_CHECKLIST.md`; `output/protocol/phase1_sample_manifest.json`. |

```text
AUTHOR_CONTACT_STATUS=CONTACTED
AUTHOR_REPLY_STATUS=CONTACTED_WAITING
GMAIL_MESSAGE_ID=1a0906ec8a4ae625
PLAN_CHANGE_ID=PLAN-CHANGE-0004
PLAN_CHANGE_0004_STATUS=APPLIED__HUMAN_AUTHORIZED
PLAN_CHANGE_HUMAN_AUTHORIZED=True
PROTOCOL_B_EVALUATION_FALLBACK=NESTED_SMALL_SAMPLE_WITH_GENERAL_HOLDOUT_BOUNDARIES
ROUNDING_RULE=zero_based__floor_boundaries__half_open_intervals
PHASE_1_GATE=PASS
PHASE_1_STATUS=COMPLETE
ADVANCEMENT_TO_PHASE_2=True
TEST_MEMBERSHIP_DEFINED=True
TEST_LOADED=False
SYNTHETIC_ROWS_MATERIALIZED=False
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_2_STL_REPRODUCTION__BEGIN_WITH_PARTITION_SAFE_DECOMPOSITION_SPEC_AND_CONTRACT_ONLY
```

## Append-only run log — Phase 2A STL specification and leakage contract

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 2A specification | Phase 1 is complete, test remains locked, and Phase 2 allows evidence recovery before implementation. | Re-read the full paper, visually inspected Fig.5 and Fig.9, audited STL input candidates, parameters, split order, causality, and component-target construction. Recorded a capacity-domain input conclusion only as cross-evidence; added no default period/window and did not execute STL. | Existing plan order and test-lock policy remain valid. Missing parameters, fit scope, and component-label mechanism prevent an honest module implementation. | `PHASE_2_GATE=BLOCKED_BY_STL_SPEC_AMBIGUITY`; no Phase 3 advancement. | `PHASE2_STL_SPEC.md`; `STL_INPUT_SEMANTICS_EVIDENCE.md`; `STL_PARAMETER_EVIDENCE.md`; `PHASE2_STL_LEAKAGE_CONTRACT.md`; `PA-STL-COMPONENT-TARGET`. |

```text
PHASE_2A_STATUS=SPECIFICATION_COMPLETE__IMPLEMENTATION_NOT_AUTHORIZED
STL_INPUT_SEMANTICS_STATUS=RESOLVED_BY_CROSS_EVIDENCE_WITH_EXECUTABLE_SCALE_LIMITATION
STL_PARAMETER_POLICY_RESOLVED=False
STL_SPLIT_ORDER_RESOLVED=False
COMPONENT_TARGET_CONSTRUCTION_RESOLVED=False
TEST_LOADED=False
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=HUMAN_REVIEW_OF_STL_ENGINEERING_FALLBACK_OPTIONS_OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 2B engineering fallback and structural period dry-run

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 2B structural STL | Phase 1 remained complete and test-locked; Human authorized a result/performance-blind, train-only period fallback without changing paper architecture or phases. | Froze partition-isolated offline STL semantics, a linear-detrend ACF/FFT agreement rule, and library-default provenance. Ran the rule on eight A/B target training tracks. Every ACF candidate was 2 while spectral candidates were 5–54, exceeding the one-cycle agreement tolerance. No period or smoother contract was frozen; no STL component was decomposed or saved. | The fallback did not become a tuning route and did not expose validation/test capacity. The plan order remains valid, but Phase 2 cannot advance until Human makes a new non-performance-based period decision or primary-source evidence appears. | `PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`; no Phase 3 advancement. | `PHASE2_STL_ENGINEERING_FALLBACK.md`; `PHASE2_STL_PERIOD_DISCOVERY.md`; `output/stl/stl_period_discovery.json`; `scripts/run_phase2_stl_structural_audit.py`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
STL_INPUT=CAPACITY_SEQUENCE
STL_INPUT_PROVENANCE=PAPER_CROSS_EVIDENCE
STL_EXECUTION_MODE=PARTITION_ISOLATED_OFFLINE_STL
STL_CAUSALITY_STATUS=NON_CAUSAL_OFFLINE_DECOMPOSITION
INTER_PARTITION_LEAKAGE=False
INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE=True
PERIOD_DISCOVERY_USES_MODEL_METRICS=False
PERIOD_DISCOVERY_USES_VALIDATION_VALUES=False
PERIOD_DISCOVERY_USES_TEST_VALUES=False
PERIOD_VALUES_FROZEN=False
TRAIN_STL_IMPLEMENTED=False
TEST_LOADED=False
PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=HUMAN_DECISION_ON_FIXED_ENGINEERING_PERIOD_OR_SMALL_BOUNDED_STRUCTURAL_SENSITIVITY_STUDY
```

## Append-only run log — Phase 2C STL period-discovery method audit

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-11 | Phase 2C methodology audit | Phase 2 remained blocked, test-locked, and limited to training-only structural analysis. | Audited the exact Phase 2B ACF/FFT rules, ACF shapes, local peaks, prefix stability, three train-only preprocessing views, spectral top-two peaks, low-frequency risk, and harmonic relations of order at most three. The global ACF argmax rule was found invalid for persistent series because it selects lag 2 rather than a periodic local maximum. No STL, model, metric, validation/test value, or search ran. | This is a Level-1 methodology amendment only. CALCE has diagnostic but insufficient evidence near 50; NASA has no stable periodic family. Phase 2 remains blocked pending Human fixed-period authority or primary-source clarification. | `PHASE_2_GATE=BLOCKED_BY_STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE`; no Phase 3 advancement. | `PERIOD_DISCOVERY_METHOD_AUDIT.md`; `PLAN_MINOR_AMENDMENT_STL-PERIOD-0001.md`; `output/stl/stl_period_method_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
ACF_ORIGINAL_RULE_VALIDITY=INVALID_FOR_PERSISTENT_SERIES
ACF_LAG2_ARTIFACT_STATUS=PERSISTENCE_OR_GLOBAL_ARGMAX_ARTIFACT
STL_PERIOD_IDENTIFIABLE=False
BOUNDED_STRUCTURAL_CANDIDATES_AVAILABLE=False
TRAIN_STL_IMPLEMENTED=False
TEST_LOADED=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
PLAN_MINOR_AMENDMENT=STL-PERIOD-0001
NEXT_ACTION=HUMAN_FIXED_ENGINEERING_PERIOD_OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 2D parameterized STL structural implementation

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-12 | Phase 2D structural implementation | Phase 2C left real-battery period unresolved. Human authorized a result/performance-blind Level-1 separation of software structure from real-data parameter fidelity. | Audited and safely hardened the existing untracked STL draft into an explicit-period module. Ran deterministic synthetic fixture tests only for periods 4, 7, and 12; verified additive reconstruction, length, finite outputs, determinism, metadata, and fail-closed period/partition/input behavior. No NASA/CALCE, validation, or test values were read. | The structural module is valid, but no real-battery period, real STL output, or paper-aligned Phase 2 result exists. A synthetic-only Phase 3 structural side-track may be considered later without advancing real-data fidelity. | `STL_STRUCTURAL_IMPLEMENTATION_GATE=PASS`; `STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD`. | `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0005.md`; `PHASE2D_PARAMETERIZED_STL_STRUCTURAL_IMPLEMENTATION.md`; `output/stl/stl_structural_implementation_audit.json`; `scripts/stl_decomposition.py`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0005
PLAN_CHANGE_STATUS=APPLIED__HUMAN_AUTHORIZED
STL_PERIOD_IDENTIFIABLE=False
NASA_STL_PERIOD=UNRESOLVED
CALCE_STL_PERIOD=UNRESOLVED
REAL_BATTERY_STL_EXECUTED=False
SYNTHETIC_FIXTURE_PERIODS=4,7,12
STL_STRUCTURAL_IMPLEMENTATION_GATE=PASS
STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD
FRAMEWORK_STL_IMPLEMENTATION_STATUS=STRUCTURALLY_REPRODUCED
PAPER_ALIGNED_REAL_STL_OUTPUTS_CREATED=False
NUMERICAL_REPRODUCTION_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING
TEST_LOADED=False
VALIDATION_VALUES_USED=False
TEST_VALUES_USED=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PRIMARY_SOURCE_CLARIFICATION_OR_SEPARATELY_HUMAN_AUTHORIZED_REAL_BATTERY_PERIOD_FALLBACK
```

## Append-only run log — Phase 3A iTransformer specification and dataflow contract

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-12 | Phase 3A specification | Phase 2 structural gate is pass-only on synthetic STL; real period remains blocked and test is locked. Phase 3 was limited to a synthetic-only structural eligibility review. | Re-read the full target paper and Fig.5, audited Eqs.(21)–(23), target-paper iTransformer claims, standard iTransformer boundaries, 10-step-window semantics, residual tensor candidates, graph-attention claim, and missing parameters. No model, real residual, validation/test value, metric, or search was used. | Fig.5 shows residual-only routing but no extra voltage/temperature/HF input; prose claims multivariate channel mixing without mapping it. Forecast/residual target alignment is missing. This is an unresolved tension, not a proven explicit paper contradiction. | Both Phase 3 gates remain blocked; no module is authorized. | `ITRANSFORMER_PAPER_EVIDENCE.md`; `ITRANSFORMER_INPUT_SEMANTICS_AUDIT.md`; `ITRANSFORMER_PARAMETER_EVIDENCE.md`; `PHASE3_ITRANSFORMER_SPEC.md`; `PLAN-CHANGE-0006`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
CURRENT_STRUCTURAL_PHASE=Phase 3
SEQUENCE_LENGTH=10
SEQUENCE_LENGTH_PROVENANCE=PAPER_EXACT
ITRANSFORMER_INPUT_SEMANTICS_STATUS=AMBIGUOUS_BLOCKING_REAL_DATA
FORECAST_TARGET_ALIGNMENT_STATUS=MISSING_BLOCKING
RESIDUAL_TARGET_ALIGNMENT=AMBIGUOUS_BLOCKING__PA-007
ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=BLOCKED_BY_TARGET_ALIGNMENT_MISSING
ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_MISSING_ARCH_PARAMETERS
REAL_RESIDUAL_DATA_USED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
TEST_VALUES_USED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PRIMARY_SOURCE_CLARIFICATION_OR_HUMAN_RESULT_BLIND_TARGET_ALIGNMENT_DECISION
```

## Append-only run log — Phase 3B parameterized synthetic-only iTransformer structural implementation

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-12 | Phase 3B synthetic structural implementation | Phase 3A left real input semantics and target alignment unresolved. Human explicitly authorized a generic synthetic multivariate interface and synthetic one-step projection target for structure-only testing. | Implemented only the common standard iTransformer core: explicit time-history embedding, variable-token inversion, encoder attention, and projection. Ran deterministic synthetic tests for one and three variables, forward/backward shape and gradient checks, finite loss, fixed-seed determinism, pre-registered 64-step tiny-overfit sanity, and fail-closed provenance/config/input checks. No NASA/CALCE, real residual/capacity/HF, validation/test value, metric, graph module, or search was used. | Structural core is valid only as a generic synthetic fixture. It does not select a paper target horizon, input tensor, graph mechanism, or target-paper architecture parameters; real-data fidelity remains blocked. A later synthetic-only framework-scaffold specification is dependency-safe but may not claim paper-aligned framework reproduction. | `ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=PASS`; `ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_TARGET_ALIGNMENT_AND_MISSING_ARCH_PARAMETERS`. | `scripts/itransformer_structural.py`; `scripts/run_itransformer_structural_tests.py`; `PHASE3B_PARAMETERIZED_ITRANSFORMER_STRUCTURAL_IMPLEMENTATION.md`; `output/itransformer/itransformer_structural_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0006
PLAN_CHANGE_STATUS=APPLIED__USER_AUTHORIZED__PHASE_3B_SCOPE_FROZEN
REAL_INPUT_SEMANTICS_INFERENCE=False
REAL_TARGET_ALIGNMENT_INFERENCE=False
PARAMETERIZED_ITRANSFORMER_IMPLEMENTED=True
STANDARD_ITRANSFORMER_CORE_IMPLEMENTED=True
CUSTOM_GRAPH_MODULE_IMPLEMENTED=False
SYNTHETIC_N_VARS_TESTED=1,3
ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=PASS
ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_TARGET_ALIGNMENT_AND_MISSING_ARCH_PARAMETERS
REAL_RESIDUAL_DATA_USED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
TEST_VALUES_USED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_4_SYNTHETIC_ONLY_FRAMEWORK_SCAFFOLD_SPECIFICATION__OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 4A synthetic G-MTC framework scaffold

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-12 | Phase 4A synthetic scaffold | Phase 2/3 structural-only gates passed while all real-data fidelity blockers and test lock remained active. Human authorized interface-only synthetic scaffold work. | Re-read the full target paper and Fig.5, recovered branch/fusion/orchestration evidence, then created only synthetic component, branch-prediction, alignment, and future-fusion payload contracts. Trend/Seasonal use explicit unimplemented TimesNet ports and deterministic mocks only for routing tests; Residual uses the existing synthetic-only iTransformer adapter. No fusion computation, TimesNet, GAN, real data, validation/test value, metric, training, or search was used. | The branch topology is paper-supported and Fig.5 is a consistent schematic omission. The scaffold passes structural routing and fail-closed tests but cannot resolve final output semantics, 9d feature semantics, orchestration, TimesNet, or inherited real-data blockers. | `FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE=PASS`; `FRAMEWORK_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_TIMESNET_AND_ITRANSFORMER_AND_FUSION_AND_TRAINING_ORCHESTRATION`. | `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0007.md`; `PHASE4_GMTC_FRAMEWORK_EVIDENCE.md`; `PHASE4_GMTC_SCAFFOLD_CONTRACT.md`; `PHASE4_GMTC_SCAFFOLD_IMPLEMENTATION.md`; `scripts/gmtc_framework_scaffold.py`; `scripts/run_gmtc_framework_scaffold_tests.py`; `output/framework/gmtc_scaffold_structural_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0007
PLAN_CHANGE_STATUS=APPLIED__USER_AUTHORIZED__PHASE_4A_SCOPE_FROZEN
FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE=PASS
FRAMEWORK_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_TIMESNET_AND_ITRANSFORMER_AND_FUSION_AND_TRAINING_ORCHESTRATION
REAL_BATTERY_DATA_USED=False
REAL_STL_COMPONENTS_USED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_5_TIMESNET_SPECIFICATION_AND_SYNTHETIC_ONLY_STRUCTURAL_WORK__OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 5A TimesNet specification and synthetic standard core

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 5A synthetic standard core | Phase 0/1 are complete; Phase 2–4 real-data gates remain blocked; Phase 4 synthetic scaffold was structurally valid and test locked. Human authorized only a generic TimesNet standard-core structural side-track. | Re-read the target paper, its Eqs.(18)–(20), Fig.5, Phase 2–4 artifacts, and official TimesNet reference architecture. Recovered evidence for FFT, top-k concept, 2D reshape, Inception, and AdaptiveMerge; audited all missing variant semantics. Implemented/tested a parameterized standard reference core only on deterministic synthetic fixtures (known periods, padding, forward/backward, fixed-seed determinism, tiny overfit, fail-closed and scaffold adapter tests). | The standard core software contract is structurally valid. The paper does not specify real branch input semantics, target alignment, k, Trend depth/configuration, Seasonal lightweight mechanism, Inception configuration, or AdaptiveMerge implementation. Thus neither target-paper TimesNet branch nor real-data fidelity is established. | `TIMESNET_STANDARD_CORE_STRUCTURAL_GATE=PASS`; `TIMESNET_BRANCH_VARIANT_FIDELITY_GATE=BLOCKED_BY_INPUT_TARGET_AND_VARIANT_PARAMETERS`. | `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0008.md`; `TIMESNET_PAPER_EVIDENCE.md`; `TIMESNET_INPUT_SEMANTICS_AUDIT.md`; `PHASE5_TIMESNET_STRUCTURAL_IMPLEMENTATION.md`; `scripts/timesnet_structural.py`; `scripts/run_timesnet_structural_tests.py`; `output/timesnet/timesnet_structural_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0008
PLAN_CHANGE_STATUS=APPLIED__USER_AUTHORIZED__PHASE_5A_SCOPE_FROZEN
TIMESNET_STANDARD_CORE_STRUCTURAL_GATE=PASS
TIMESNET_BRANCH_VARIANT_FIDELITY_GATE=BLOCKED_BY_INPUT_TARGET_AND_VARIANT_PARAMETERS
REAL_STL_COMPONENTS_USED=False
REAL_BATTERY_DATA_USED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_6_SYNTHETIC_ONLY_FUSION_SPECIFICATION_AND_INTERFACE_AUDIT__OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 6A adaptive fusion specification and synthetic structural implementation

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 6A synthetic adaptive fusion | Phase 0/1 complete; Phase 2–5 structural side-tracks passed while all real-data fidelity gates and the test lock remained active. Human authorized only a preconstructed synthetic fusion-feature side-track. | Re-read the target PDF (Eqs.(24)–(29), Fig.4, Table 4), recovered the fusion topology and audited feature/dimension/loss omissions. Implemented and tested only a parameterized synthetic B×T×9 core: component attention, Bi-LSTM, temporal attention, 3-way Softmax weights, weighted slots, and final head. No real feature builder, branch prediction, NASA/CALCE, validation/test value, performance metric, SCF-GAN, or search was used. | The structural wiring/normalization/gradient/provenance contract passes. Paper does not provide inference-safe errors/ratios, d semantics, feature selection, loss terms/lambda, or final output domain; real-data fidelity remains blocked and all inherited blockers persist. | `ADAPTIVE_FUSION_STRUCTURAL_GATE=PASS`; `ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_FEATURE_SEMANTICS_LOSS_AND_BRANCH_ALIGNMENT`. | `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0009.md`; `FUSION_PAPER_EVIDENCE.md`; `FUSION_DIMENSION_SEMANTICS_AUDIT.md`; `PHASE6_ADAPTIVE_FUSION_STRUCTURAL_IMPLEMENTATION.md`; `scripts/adaptive_fusion_structural.py`; `scripts/run_adaptive_fusion_structural_tests.py`; `output/fusion/adaptive_fusion_structural_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0009
PLAN_CHANGE_STATUS=APPLIED__USER_AUTHORIZED__PHASE_6A_SCOPE_FROZEN
ADAPTIVE_FUSION_STRUCTURAL_GATE=PASS
ADAPTIVE_FUSION_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_FEATURE_SEMANTICS_LOSS_AND_BRANCH_ALIGNMENT
REAL_BATTERY_DATA_USED=False
REAL_STL_COMPONENTS_USED=False
REAL_BRANCH_PREDICTIONS_USED=False
REAL_FUSION_FEATURE_BUILDER_IMPLEMENTED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=1
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_7_SCF_GAN_SPECIFICATION_AND_SYNTHETIC_ONLY_STRUCTURAL_WORK__OR_PRIMARY_SOURCE_CLARIFICATION
```

## Append-only run log — Phase 7A SCF-GAN specification and cross-phase reconciliation

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 7A SCF-GAN evidence reconciliation | Phase 0/1 complete; Phase 2–6 synthetic structural gates passed while every real-data fidelity gate and the test lock remained active. | Re-read the local target PDF, Phase 0–6 artifacts, §2.2, §3.2, §3.7, §4/§4.2, Tables 2–3, and visually inspected rendered Fig.3/Fig.5. Audited SCF-GAN conditions, output fields, architecture conflicts, source/target roles, augmentation wording, and the top-to-bottom Fig.5 interface. No NASA/CALCE data, validation/test values, GAN training, synthetic rows, model metric, or search was used. | Fig.3 labels a generated 4D HF1/HF2/HF3/SOH vector while §4.2 explicitly names generated HF1/HF2/HF3; Fig.5 contains no generated-features-to-STL arrow; the existing capacity-domain STL cross-evidence has no reported conversion/component-target bridge. This is a core real-data interface defect, not a minor software parameter omission. | `GMTC_CROSS_PHASE_DATAFLOW_GATE=PLAN_CHANGE_REQUIRED`; Level-2 request only; no SCF-GAN implementation authorized. | `SCFGAN_PAPER_EVIDENCE.md`; `SCFGAN_CONDITION_SEMANTICS_AUDIT.md`; `CROSS_PHASE_GMTC_DATAFLOW_RECONCILIATION.md`; `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0010.md`; `output/phase7_figure_visual_evidence/`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0010
PLAN_CHANGE_STATUS=REQUESTED__HUMAN_AUTHORIZATION_REQUIRED
GMTC_CROSS_PHASE_DATAFLOW_GATE=PLAN_CHANGE_REQUIRED
CORE_DATAFLOW_CORRECTION_REQUIRED=True
REAL_SCFGAN_TRAINING=False
REAL_SYNTHETIC_ROWS_CREATED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=FAIL__LEVEL2_CORE_DATAFLOW_RECONCILIATION_REQUIRED
PLAN_DEVIATION_LEVEL=2
PLAN_CHANGE_REQUIRED=True
NEXT_ACTION=HUMAN_REVIEW_AND_AUTHORIZE_OR_REJECT_PLAN_CHANGE_0010
```

## PLAN-CHANGE-0010 — applied cross-phase real-dataflow fallback

`PLAN_CHANGE_ID=PLAN-CHANGE-0010`

`NAME=SCFGAN_AUGMENTED_HF_TO_DECOMPOSED_TARGET_DATAFLOW_RECONCILIATION`

`LEVEL=2`

`CURRENT_STRUCTURAL_PHASE=Phase_7`

`CURRENT_GATE=GMTC_CROSS_PHASE_DATAFLOW_GATE`

`HUMAN_AUTHORIZED=True`

`STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_CROSS_PHASE_FALLBACK`

`CROSS_PHASE_EXECUTION_FALLBACK=HF_INPUT_TO_COMPONENT_TARGET_MODE`

`PROVENANCE=HUMAN_RESULT_BLIND_CROSS_PHASE_FALLBACK`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`TEST_LOADED=False`

`REAL_DATA_USED=False`

`SYNTHETIC_SOH_ASSUMPTION=False`

`STRUCTURAL_MODULES_INVALIDATED=False`

The applied fallback separates the active real-data paths: HF1/HF2/HF3 multivariate windows are future inputs for all three branches; observed training capacity is the STL input and supplies trend/seasonal/residual targets; branch predictions, rather than raw components, supply future fusion inputs. It does not choose SCF-GAN conditions/architecture, synthesize SOH, set an STL period, set prediction alignment, or clear any fidelity blocker.

## Append-only run log — Phase 7B apply PLAN-CHANGE-0010 and reconcile real-dataflow contracts

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 7B governance and contract reconciliation | Phase 7A correctly stopped at a Level-2 cross-phase dataflow defect. Human has now authorized a result/performance-blind Candidate-A execution fallback with no synthetic-SOH assumption. | Applied the fallback solely as an append-only real-data contract: HF1/HF2/HF3 windows are future inputs; observed TRAIN capacity through STL supplies T/S/R supervision targets; future fusion consumes branch predictions. Updated Phase 2–6 active contracts without modifying or invalidating synthetic structural tests. No NASA/CALCE, validation/test value, real GAN, synthetic row, STL decomposition, model, metric, or search was used. | Input and target paths are now explicitly distinct and internally coherent under a Human fallback. Paper contradictions, all real-data fidelity blockers, test lock, and author override remain. | `GMTC_CROSS_PHASE_DATAFLOW_GATE=PASS_WITH_HUMAN_RESULT_BLIND_FALLBACK`; no Phase 7C implementation in this run. | `GMTC_REAL_DATAFLOW_FALLBACK_CONTRACT.md`; `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0010.md`; Phase 2–6 append-only interface records. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PLAN_CHANGE_ID=PLAN-CHANGE-0010
PLAN_CHANGE_STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_CROSS_PHASE_FALLBACK
GMTC_CROSS_PHASE_DATAFLOW_GATE=PASS_WITH_HUMAN_RESULT_BLIND_FALLBACK
FRAMEWORK_REPRODUCTION_DATAFLOW=EXECUTABLE_UNDER_HUMAN_FALLBACK
NUMERICAL_REPRODUCTION_DATAFLOW=NOT_PAPER_EXACT__AUTHOR_OVERRIDE_PENDING
REAL_DATA_USED=False
REAL_SCFGAN_TRAINING=False
REAL_SYNTHETIC_ROWS_CREATED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_7C_AMBIGUITY_PRESERVING_SYNTHETIC_SCFGAN_STRUCTURAL_IMPLEMENTATION
```

## Append-only run log — Phase 7C ambiguity-preserving synthetic SCF-GAN structural implementation

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 7C synthetic SCF-GAN core | PLAN-CHANGE-0010 is applied; the Human fallback permits only a synthetic structural implementation while real SCF-GAN conditions, architecture, output semantics, and augmentation remain blocked. | Implemented a fully explicit, synthetic-only conditional GAN core and a finite coverage matrix over ReLU/LeakyReLU, BN/conditional-affine BN, generator SN off/on, discriminator depth 3/4, and generic feature width 3/4. Tested routing, Tanh/Sigmoid bounds, condition paths, SN wrappers, third-hidden feature matching, stable-equivalent losses with paper lambda 0.1, gradients, determinism, a three-step optimizer smoke, and fail-closed provenance/request guards. | No ambiguity option was selected; no quality metric, real battery data, validation/test value, synthetic SOH, real GAN training, or real synthetic row was used. The structural implementation is valid; the real-data fidelity gate retains all paper contradictions/missing fields. | `SCFGAN_STRUCTURAL_IMPLEMENTATION_GATE=PASS`; `SCFGAN_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_CONDITION_OUTPUT_ARCHITECTURE_AND_AUGMENTATION_AMBIGUITIES`. | `scripts/scfgan_structural.py`; `scripts/run_scfgan_structural_tests.py`; `SCFGAN_PARAMETERIZED_ARCHITECTURE_CONTRACT.md`; `PHASE7C_AMBIGUITY_PRESERVING_SCFGAN_STRUCTURAL_IMPLEMENTATION.md`; `output/scfgan/scfgan_structural_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
PARAMETERIZED_SCFGAN_IMPLEMENTED=True
AMBIGUITY_PRESERVING_IMPLEMENTATION=True
AMBIGUITY_SELECTION_PERFORMED=False
REAL_DATA_USED=False
REAL_SCFGAN_TRAINING=False
REAL_SYNTHETIC_ROWS_CREATED=False
SYNTHETIC_SOH_GENERATION=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=PHASE_8_SYNTHETIC_OR_CONTRACT_LEVEL_INTEGRATION_REVIEW_ONLY
```

## Append-only run log — Phase 8A Full G-MTC integration readiness and semantic seam audit

| Timestamp | Phase | PLAN_AUDIT_BEFORE_RUN | Action | PLAN_AUDIT_AFTER_RUN | Gate result | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-13 | Phase 8A contract integration readiness | The Phase 7B Human result-blind fallback makes the high-level input and target paths coherent, while every real-data fidelity gate remains blocked. | Audited all active Phase 2–7 interfaces without reading NASA/CALCE, validation/test values, or running a numeric pipeline. Added explicit scale, sample-identity/alignment, module-inventory, fusion-seam, and blocker-resolution contracts, plus a metadata-only fail-closed validator. | Contract-level integration is coherent, but its numeric seams are deliberately unresolved: SCF-GAN output scale/HF normalization, synthetic-cycle pairing, branch target alignment/prediction length, real STL targets, fusion feature semantics, final domain, total loss, and orchestration. | `GMTC_CONTRACT_INTEGRATION_GATE=PASS`; `GMTC_SYNTHETIC_NUMERIC_INTEGRATION_GATE=BLOCKED_BY_UNRESOLVED_SCALE_ALIGNMENT_AND_FUSION_SEMANTICS`; `GMTC_REAL_DATA_INTEGRATION_GATE=BLOCKED_BY_RETAINED_FIDELITY_BLOCKERS`. | `PHASE8_GMTC_INTEGRATION_READINESS_AUDIT.md`; `GMTC_SCALE_AND_NORMALIZATION_CONTRACT.md`; `GMTC_SAMPLE_IDENTITY_AND_ALIGNMENT_CONTRACT.md`; `NUMERICAL_REPRODUCTION_BLOCKER_RESOLUTION_MATRIX.md`; `scripts/gmtc_integration_contract.py`; `output/integration/gmtc_contract_integration_audit.json`. |

```text
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS
CURRENT_PHASE=Phase_8
REAL_DATA_USED=False
REAL_SCFGAN_TRAINING=False
REAL_SYNTHETIC_ROWS_CREATED=False
TEST_LOADED=False
VALIDATION_VALUES_USED=False
MODEL_TRAINED=False
MODEL_METRICS_USED=False
GRID_SEARCH_RUN=False
CURRENT_MAX_LEGAL_INTEGRATION_LEVEL=LEVEL_1__SCHEMA_AND_CONTRACT_INTEGRATION
FULL_SYNTHETIC_END_TO_END_EXECUTION=BLOCKED
ADVANCEMENT_TO_PHASE_9=False
STOP_STRUCTURAL_EXPANSION_RECOMMENDED=True
PLAN_HEALTH_AUDIT_AFTER_RUN=PASS__STOP_STRUCTURAL_EXPANSION_RECOMMENDED
PLAN_DEVIATION_LEVEL=0
PLAN_CHANGE_REQUIRED=False
NEXT_ACTION=RESOLVE_PRIMARY_SOURCE_FIDELITY_BLOCKERS_OR_OBTAIN_A_SEPARATE_HUMAN_RESULT_BLIND_FALLBACK
```
