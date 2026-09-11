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