# REPRODUCTION MASTER PLAN

**Project:** faithful reproduction of Hanhua Yang et al., *Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach*, *Energy* 342 (2026) 139463, DOI: 10.1016/j.energy.2025.139463.

**Plan authority:** This is the sole highest-level roadmap for this repository. Every future Codex task must read this complete file before acting and must obey its gates. Its core architecture, phase order, reproduction definitions, and protocol rules may not be silently changed.

**Current plan state:** INITIALIZED — no Phase has passed.

`CURRENT_PLAN_VERSION=1.2.0`

`LAST_PLAN_CHANGE_ID=PLAN-CHANGE-0002`

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
Phase 1  Data Protocol            [..........] 0%
Phase 2  STL                      [..........] 0%
Phase 3  iTransformer             [..........] 0%
Phase 4  Framework Scaffold       [..........] 0%
Phase 5  TimesNet                 [..........] 0%
Phase 6  Adaptive Fusion          [..........] 0%
Phase 7  SCF-GAN                  [..........] 0%
Phase 8  Full G-MTC               [..........] 0%
Phase 9  Numerical Reproduction   [..........] 0%
Phase 10 Acceptance               [..........] 0%

OVERALL_PROGRESS=9%
CURRENT_PHASE=Phase 0
CURRENT_GATE=Phase 0 advancement review: critical protocol ambiguity PA-004
LAST_COMPLETED=CALCE official full-row scan completed; source-safe cross-workbook cycle-boundary and row-null/duplicate reporting remain incomplete; PA-004 remains unresolved
NEXT_ACTION=Persist CALCE source-keyed cycle-boundary and row-null/duplicate audit without splits, while Human obtains PA-004 primary-source clarification; do not enter Phase 1 or implement model code

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

## Phase 2 — STL Reproduction

**OBJECTIVE:** independently reproduce decomposition behavior before neural modeling.

**INPUT:** Phase 1 frozen manifest and exact/approved STL parameters.

**ALLOWED_ACTIONS:** implement and test STL only; inspect raw degradation sequence, Trend, Seasonal, and Residual; create component manifests and reconstruction checks.

**FORBIDDEN_ACTIONS:** GAN, TimesNet, iTransformer, fusion, end-to-end model, training, and numerical-performance claims.

**REQUIRED_ARTIFACTS:** STL configuration record, decomposition outputs with lineage, reconstruction/alignment/boundary tests, and period/window decision linked to paper evidence or ambiguity ID.

**ACCEPTANCE_CRITERIA:** `Trend + Seasonal + Residual ≈ original` under a predefined numerical tolerance; shape, time alignment, boundaries, window, period, and unavailable-cycle behavior are verified without future leakage.

**FAILURE_CONDITIONS:** mismatched indexes, unexplained reconstruction error, full-lifetime fitting that contaminates held-out evaluation, or unlogged parameter substitution.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 2 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 3 — Transformer / iTransformer Branch

**OBJECTIVE:** reproduce and validate the Residual -> iTransformer branch in isolation.

**INPUT:** frozen component manifest and paper-specified/approved iTransformer configuration.

**ALLOWED_ACTIONS:** implement iTransformer branch, tensor contracts, branch-local training checks, loss and prediction-target verification, and non-test development validation.

**FORBIDDEN_ACTIONS:** Ridge or any classical method as replacement; TimesNet substitute; fusion acceptance; SCF-GAN integration; paper-success claims.

**REQUIRED_ARTIFACTS:** iTransformer specification record, input/output tensor contract, branch-local configuration, seed/environment capture, and development-only verification log.

**ACCEPTANCE_CRITERIA:** residual tensor, 10-step sequence handling (unless Phase 0 corrects it with evidence), feature dimensions, target, loss, gradients, and non-test training behavior all match the frozen specification.

**FAILURE_CONDITIONS:** tensor mismatch, target leakage, implicit model substitution, or test-driven architecture selection.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 3 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 4 — G-MTC Framework Scaffold

**OBJECTIVE:** validate the routing graph `STL -> {Trend, Seasonal, Residual} -> temporal outputs -> fusion interface` without claiming complete paper architecture.

**INPUT:** accepted STL and iTransformer artifacts.

**ALLOWED_ACTIONS:** build interfaces and shape/lineage tests. Temporary Trend/Seasonal engineering placeholders are allowed only if explicitly labeled `TEMPORARY_ONLY=True`.

**FORBIDDEN_ACTIONS:** declaring framework or paper reproduction; presenting placeholder outputs as TimesNet; numerical reproduction; end-to-end SCF-GAN tuning.

**REQUIRED_ARTIFACTS:** branch routing contract, component-output schema, fusion-interface contract, and temporary-branch registry when applicable.

**ACCEPTANCE_CRITERIA:** component routing, temporal outputs, fusion input order, shapes, provenance, and gradients are verified. Any placeholder is visibly registered.

**FAILURE_CONDITIONS:** hidden placeholder, component swap, unsourced fusion interface, or any `PAPER_REPRODUCTION_STATUS=ESTABLISHED` claim.

**STATUS RULE:** before Phase 5, `PAPER_FRAMEWORK_REPRODUCTION=PASS` is prohibited and `PAPER_REPRODUCTION_STATUS=NOT_ESTABLISHED`.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 4 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 5 — TimesNet Reproduction

**OBJECTIVE:** formally implement Trend -> TimesNet and Seasonal -> TimesNet under the frozen paper specification.

**INPUT:** Phase 4 interfaces and paper/ambiguity-controlled TimesNet configuration.

**ALLOWED_ACTIONS:** implement TimesNet period detection, 1D-to-2D transformation, Inception-style processing, adaptive merge, Trend and Seasonal branch variants, and branch-local checks.

**FORBIDDEN_ACTIONS:** temporary branch retention; paper-success declaration before later gates; unapproved architecture choices.

**REQUIRED_ARTIFACTS:** two TimesNet branch specifications, tensor/period tests, source evidence map, and removal record for every temporary branch.

**ACCEPTANCE_CRITERIA:** both specified TimesNet branches pass contracts and `NO_TEMPORARY_BRANCHES=True`.

**FAILURE_CONDITIONS:** generic Transformer substitution, unverified hyperparameters represented as paper facts, or residual placeholders.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 5 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

## Phase 6 — Adaptive Fusion

**OBJECTIVE:** reproduce the paper’s attention-based Bi-LSTM adaptive fusion.

**INPUT:** three paper-specified branch outputs and fusion specification.

**ALLOWED_ACTIONS:** implement component attention, time-step attention, Bi-LSTM, softmax weights, weighted fusion, final prediction head, and contract tests.

**FORBIDDEN_ACTIONS:** direct sum represented as the paper fusion; test-selected fusion variants; omission of required 9-feature contract without a deviation record.

**REQUIRED_ARTIFACTS:** fusion configuration, weight/normalization/gradient tests, 9-feature provenance definition, and direct-sum comparator identified as an ablation only.

**ACCEPTANCE_CRITERIA:** weights have expected shape, normalize across the three components, are trainable, receive gradients, and map to the correct component predictions.

**FAILURE_CONDITIONS:** detached weight path, non-normalized weights, data leakage in error/ratio features, or non-paper fusion asserted as paper logic.

**NEXT_PHASE_ENTRY_CONDITION:** Phase 6 gate passes and its `PHASE_COMPLETION_COMMIT` is recorded.

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

## Phase completion ledger

| Phase | Gate status | Completion evidence | PHASE_START_COMMIT | PHASE_COMPLETION_COMMIT | PHASE_COMPLETION_PUSH_STATUS | PHASE_REMOTE_BACKUP_CONFIRMED |
|---|---|---|---|---|---|---|
| 0 | PASS_WITH_REGISTERED_AMBIGUITIES__ADVANCEMENT_BLOCKED | PAPER_REPRODUCTION_SPEC.md; PAPER_AMBIGUITY_REGISTER.md; PAPER_MISSING_PARAMETER_REGISTER.md; JSON specification | f354af6 | f354af6 | PASS | True |
| 1 | NOT_STARTED | — | — | — | — | False |
| 2 | NOT_STARTED | — | — | — | — | False |
| 3 | NOT_STARTED | — | — | — | — | False |
| 4 | NOT_STARTED | — | — | — | — | False |
| 5 | NOT_STARTED | — | — | — | — | False |
| 6 | NOT_STARTED | — | — | — | — | False |
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
