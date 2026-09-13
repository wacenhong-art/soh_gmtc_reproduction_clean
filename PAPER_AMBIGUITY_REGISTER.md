# PAPER AMBIGUITY REGISTER

All records are Phase 0 paper-evidence records. They do not authorize defaults or implementation choices.

## PA-001 — SOH executable denominator

`CATEGORY=target_definition`
`PAPER_LOCATION=PDF p.1; p.3`
`PAPER_TEXT_SUMMARY=SOH is residual capacity relative to nominal capacity; NASA termination is stated relative to initial rated capacity.`
`CONFLICT_OR_MISSING_DETAIL=No executable capacity denominator/formula is supplied for each dataset.`
`POSSIBLE_INTERPRETATION_A=capacity / nominal rated capacity`
`POSSIBLE_INTERPRETATION_B=capacity / first measured or initial capacity`
`IMPLEMENTATION_IMPACT=Label scale and target construction change.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-002 — source/target roster varies by experiment

`CATEGORY=data_protocol`
`PAPER_LOCATION=PDF pp.9–11`
`PAPER_TEXT_SUMMARY=Examples name B0005/B0006 and CS2-35/37 as source; B0007/B0018 and CS2-36/38 as target; separate conditional experiment uses Th6/CS2-37 source and Th5/CS2-35 target.`
`CONFLICT_OR_MISSING_DETAIL=No single experiment-to-roster mapping is fully tabulated.`
`POSSIBLE_INTERPRETATION_A=Each results table has its own source/target roster.`
`POSSIBLE_INTERPRETATION_B=Examples are illustrative, not exhaustive.`
`IMPLEMENTATION_IMPACT=GAN training population and evaluation target differ.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-003 — temporal-model input representation

`CATEGORY=input_and_stl`
`PAPER_LOCATION=PDF pp.3, 6–10`
`PAPER_TEXT_SUMMARY=Paper forms HF=[HF1,HF2,HF3], defines generic STL X_t, and states multivariate 10-step windows.`
`CONFLICT_OR_MISSING_DETAIL=It does not state whether STL decomposes SOH, each HF, an aggregate HF sequence, or another tensor.`
`POSSIBLE_INTERPRETATION_A=STL decomposes a univariate SOH/degradation sequence.`
`POSSIBLE_INTERPRETATION_B=STL decomposes health-feature temporal representation.`
`IMPLEMENTATION_IMPACT=Branch tensor contracts and leakage checks change.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

### Phase 3A audit update — residual branch dataflow

`EVIDENCE_ADDED=PDF p.6 §3.3; p.8 Fig.5 visual audit; p.9 §3.7(2)`

`FINDING=The capacity-domain STL cross-evidence and Fig.5 support residual-only routing, but the prose claims nonlinear interactions between voltage fluctuations and temperature variations. No residual-branch mapping supplies those channels.`

`IMPLEMENTATION_IMPACT=The real iTransformer input tensor cannot be made paper-faithful without an unreported engineering addition.`

`STATUS=RETAINED_AS_BLOCKING_FOR_REAL_DATA_ITRANSFORMER`

### Phase 5A audit update — Trend and Seasonal branch dataflow

`EVIDENCE_ADDED=PDF p.6 §3.3–§3.4 Eqs.(18)–(20); p.8 Fig.5 visual audit; p.9 §3.7(2)`

`FINDING=Fig.5 displays Trend and Seasonal component pathways without an HF/raw-signal side-channel. Section 3.4 defines generic multivariate input X∈R^(T×C), while the paper does not map C to either component branch.`

`IMPLEMENTATION_IMPACT=The real Trend/Seasonal input tensor cannot be chosen as univariate or multivariate without an unreported engineering assumption.`

`STATUS=RETAINED_AS_BLOCKING_FOR_REAL_DATA_TIMESNET`

## PA-004 — split versus small-sample conflict

`CATEGORY=critical_protocol`
`PAPER_LOCATION=PDF p.10 §4 and §4.2`
`PAPER_TEXT_SUMMARY=§4 says target available cycles split chronological 60%/20%/20%. §4.2 says original target training set is first 20% of total life and is augmented with five complete synthetic datasets.`
`CONFLICT_OR_MISSING_DETAIL=The relation between these two constructions is not specified.`
`POSSIBLE_INTERPRETATION_A=60% is target training partition, but only its first 20%-of-life portion is original data before augmentation.`
`POSSIBLE_INTERPRETATION_B=Separate experiments use different small-sample protocols.`
`IMPLEMENTATION_IMPACT=Train/validation/test population and all numerical claims change.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-005 — synthetic amount and insertion point

`CATEGORY=augmentation_protocol`
`PAPER_LOCATION=PDF pp.10–11`
`PAPER_TEXT_SUMMARY=Five complete synthetic HF datasets are generated per original target set; text calls this fivefold expansion.`
`CONFLICT_OR_MISSING_DETAIL=Whether this means five generated sets plus real data, a final 5× total, and how data enter chronological partitions is unstated.`
`POSSIBLE_INTERPRETATION_A=five generated datasets are appended to training data only.`
`POSSIBLE_INTERPRETATION_B=fivefold total augmentation includes a different sampling ratio.`
`IMPLEMENTATION_IMPACT=Augmented sample count and leakage boundary change.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-006 — STL fit scope and timing

`CATEGORY=stl_protocol`
`PAPER_LOCATION=PDF p.6; pp.8–10`
`PAPER_TEXT_SUMMARY=STL additive equation is specified, but fit scope and relation to windowing/augmentation are not.`
`CONFLICT_OR_MISSING_DETAIL=Per-battery/global, train-only/full-life, before/after windowing, and boundary handling are absent.`
`POSSIBLE_INTERPRETATION_A=fit STL independently using training-history-only data.`
`POSSIBLE_INTERPRETATION_B=paper used full-life decomposition before split.`
`IMPLEMENTATION_IMPACT=Future-information risk and strict protocol validity.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-STL-COMPONENT-TARGET — validation/test component-label construction

`CATEGORY=stl_component_target_protocol`
`PAPER_LOCATION=PDF p.6 §3.3; pp.8–9 Fig.5/§3.7; p.13 Fig.9; p.14 §4.4`
`PAPER_TEXT_SUMMARY=The paper shows real/predicted trend, seasonal and residual components, but does not state how real validation/test component labels are derived after chronological splitting.`
`CONFLICT_OR_MISSING_DETAIL=No full-life, partition-isolated, rolling/expanding, or final-SOH-only supervision rule is stated.`
`POSSIBLE_INTERPRETATION_A=Decompose each partition independently.`
`POSSIBLE_INTERPRETATION_B=Decompose the complete lifecycle before splitting.`
`POSSIBLE_INTERPRETATION_C=Use an expanding/rolling decomposition or no component-label supervision.`
`IMPLEMENTATION_IMPACT=Determines branch targets and can create inter-partition or intra-train future-context leakage.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-007 — iTransformer prediction target

`CATEGORY=branch_target`
`PAPER_LOCATION=PDF p.7 Eqs.(21)–(23)`
`PAPER_TEXT_SUMMARY=Generic iTransformer X-to-Y future-variable formulation is supplied.`
`CONFLICT_OR_MISSING_DETAIL=Residual-branch target horizon/output tensor is not specified.`
`POSSIBLE_INTERPRETATION_A=predict residual component then fuse.`
`POSSIBLE_INTERPRETATION_B=predict SOH contribution from residual-input representation.`
`IMPLEMENTATION_IMPACT=Loss and fusion interface differ.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

### Phase 3A audit update — 2026-09-12

`EVIDENCE_ADDED=PDF p.6 §3.3; p.7 Eqs.(21)–(23); p.8 Fig.5 visual audit; p.9 §3.7(2)`

`FINDING=Fig.5 routes only the residual component to its residual representation and does not show voltage, temperature, or HF side inputs. The paper’s channel-mixing/voltage-temperature prose therefore cannot determine the residual input tensor.`

`FORECAST_TARGET_ALIGNMENT_STATUS=MISSING_BLOCKING`

`RESIDUAL_TARGET_ALIGNMENT=AMBIGUOUS_BLOCKING`

`CORE_DATAFLOW_CONTRADICTION_FOUND=False__UNRESOLVED_TENSION_NOT_EXPLICIT_CONTRADICTION`

`IMPLEMENTATION_STATUS=BLOCKED_PENDING_TARGET_ALIGNMENT`

## PA-008 — fusion errors and ratios provenance

`CATEGORY=fusion_protocol`
`PAPER_LOCATION=PDF p.7; p.9 Table 4`
`PAPER_TEXT_SUMMARY=Fusion accepts predictions, errors, and ratios for three components (9d).`
`CONFLICT_OR_MISSING_DETAIL=How errors/ratios are defined at inference without target leakage is not stated.`
`POSSIBLE_INTERPRETATION_A=errors are train/validation-derived auxiliary estimates.`
`POSSIBLE_INTERPRETATION_B=paper diagram uses unavailable-at-inference target errors.`
`IMPLEMENTATION_IMPACT=Potential direct target leakage.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

### Phase 6A audit update — fusion feature semantics and real-data boundary

`EVIDENCE_ADDED=PDF p.7 Eqs.(24)–(29); p.9 Table 4; Fig.4`

`FINDING=The topology, 9 named feature groups, 3-way Softmax, Bi-LSTM widths, weight-network widths, and final-head widths are explicit. The paper does not define component error or ratio formulas, their inference-time availability, d in 9d/3d, the ordering of first 3d slots, L_pred, L_weight, fusion lambda, or a capacity-to-SOH conversion.`

`IMPLEMENTATION_IMPACT=A real fusion-feature builder would require unreported semantics and could introduce direct target leakage. Only synthetic preconstructed BxTx9 feature tensors are authorized for Phase 6A structural testing.`

`STATUS=RETAINED_AS_BLOCKING_FOR_REAL_DATA_FUSION`

## PA-009 — GAN condition semantics

`CATEGORY=gan_condition`
`PAPER_LOCATION=PDF pp.5, 7–8, 11`
`PAPER_TEXT_SUMMARY=Condition c is called cycle period, voltage range, and elsewhere battery characteristics/usage conditions.`
`CONFLICT_OR_MISSING_DETAIL=Exact condition vector and encoding are inconsistent/unstated.`
`POSSIBLE_INTERPRETATION_A=voltage interval plus cycle condition.`
`POSSIBLE_INTERPRETATION_B=voltage interval only.`
`IMPLEMENTATION_IMPACT=Generator/discriminator input dimensions and generation validity change.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## PA-010 — SCF-GAN architecture inconsistency

`CATEGORY=gan_architecture`
`PAPER_LOCATION=PDF p.5 Eq.(7)/(11); pp.8–9 Tables 2–3`
`PAPER_TEXT_SUMMARY=Equation/table activation, hidden-layer count, and spectral-normalization placement differ.`
`CONFLICT_OR_MISSING_DETAIL=Generator Eq.(7) uses LeakyReLU/BN while Table 2 uses ReLU/CBN; discriminator Eq.(11) shows three hidden states while Table 3 lists four; prose says SN both networks while Table 2 does not.`
`POSSIBLE_INTERPRETATION_A=tables are implementation authority.`
`POSSIBLE_INTERPRETATION_B=equations/prose are implementation authority.`
`IMPLEMENTATION_IMPACT=Core SCF-GAN differs.`
`BLOCKS_IMPLEMENTATION=True`
`HUMAN_DECISION_REQUIRED=True`

## Append-only engineering fallback record — PA-006 / PM-002 / PA-STL-COMPONENT-TARGET

`STATUS=HUMAN_FALLBACK_ATTEMPTED__PAPER_AMBIGUITY_RETAINS_PRIORITY`
`FALLBACK=PARTITION_ISOLATED_OFFLINE_STL__TRAIN_ONLY_STRUCTURAL_PERIOD_SELECTION__LIBRARY_DEFAULT_SMOOTHERS_IF_PERIOD_FROZEN`
`RESULT_BLIND=True`
`PERFORMANCE_BLIND=True`
`TEST_LOADED=False`
`AUTHOR_OVERRIDE_PENDING=True`

## Phase 7B authorized execution fallback — cross-phase dataflow only

`PLAN_CHANGE_ID=PLAN-CHANGE-0010`

`STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_CROSS_PHASE_FALLBACK`

`EXECUTION_FALLBACK=HF_INPUT_TO_COMPONENT_TARGET_MODE`

`SCFGAN_REAL_OUTPUT_EXECUTION_POLICY=HF1_HF2_HF3_ONLY__HUMAN_FALLBACK`

`SYNTHETIC_SOH_GENERATION=PROHIBITED_UNDER_CURRENT_FALLBACK`

`PAPER_OUTPUT_SEMANTICS_STATUS=CONTRADICTORY__RETAINED`

`BRANCH_INPUT_FALLBACK=HF1_HF2_HF3_MULTIVARIATE_WINDOW`

`BRANCH_TARGET_FALLBACK=OBSERVED_TRAIN_CAPACITY_STL_COMPONENTS`

`BRANCH_TARGET_ALIGNMENT_STATUS=UNRESOLVED__RETAINED`

`AUTHOR_OVERRIDE_PENDING=True`

This Human fallback is result/performance blind and does not convert any contradictory or missing paper statement into `PAPER_EXACT`.

The deterministic ACF/spectral agreement rule failed for every Protocol-A and Protocol-B target training track. It does not resolve the paper ambiguity and it does not supply a hidden period: `STL_PERIOD_NOT_STRUCTURALLY_IDENTIFIABLE=True`. Any fixed engineering period or bounded structural sensitivity study now requires a separate Human decision.

## Phase 7A cross-phase dataflow update — PA-009 / PA-010 and new interface blocker

`EVIDENCE_ADDED=PDF pp.4–5 §3.2; PDF p.8 Fig.5; PDF pp.8–10 §3.7/§4.2; visual Figure audit retained in output/phase7_figure_visual_evidence/`

`PA009_STATUS=CONTRADICTORY__CYCLE_VOLTAGE_BATTERY_TYPE_CHEMISTRY_AND_USAGE`

`PA010_STATUS=CONTRADICTORY__GENERATOR_ACTIVATION_BN_CBN_SN_SCOPE_AND_DISCRIMINATOR_DEPTH`

`NEW_CROSS_PHASE_BLOCKER=GMTC_CROSS_PHASE_DATAFLOW_UNRESOLVED`

`FINDING=Fig.3 labels SCF-GAN output as four fields HF1/HF2/HF3/SOH, whereas §4.2 explicitly names generated HF1/HF2/HF3 and Fig.5 contains no visible Generated-Features-to-STL arrow. Capacity-domain STL cross-evidence therefore cannot be reconciled with a generated-SOH path without an unreported conversion and component-target rule.`

`STATUS=LEVEL2_PLAN_CHANGE_REQUEST_REQUIRED__DO_NOT_SELECT_A_REAL_DATAFLOW`

`AUTHOR_OVERRIDE_PENDING=True`
