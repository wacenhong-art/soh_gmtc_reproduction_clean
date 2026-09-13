# Phase 8B — Numerical Fidelity Blocker Triage

## Taxonomy

`CLASS_A=PAPER_CROSS_EVIDENCE_RESOLVABLE` — a documented cross-evidence chain can yield one or a near-unique meaning, but no active protocol is changed in this audit.

`CLASS_B=HUMAN_RESULT_BLIND_FALLBACK_FEASIBLE` — a deterministic, leakage-safe, performance-blind fallback can be proposed for later Human authorization.

`CLASS_C=PRIMARY_SOURCE_STRONGLY_PREFERRED` — alternatives materially affect data semantics, training targets, or inference behavior.

`CLASS_D=PAPER_INTERNAL_NONEXECUTABILITY_RISK` — the wording can imply unavailable-at-inference information, label leakage, or an unimplementable method; it needs a methodological audit rather than a default.

## Complete triage

| Blocker | Class | Scientific impact | Execution impact | Fallback risk | Author clarification value | Evidence / decision boundary |
|---|---|---:|---:|---:|---:|---|
| `STL_REAL_BATTERY_PERIOD_MISSING` | B | HIGH | CRITICAL | HIGH | HIGH | The paper gives generic `p` only; structural discovery was non-identifying. A fixed result-blind period needs separate Human authority. |
| `HF_NORMALIZATION_METHOD_MISSING` | B | HIGH | HIGH | MEDIUM | HIGH | Fig.5 names Normalize; method and fit scope absent. Train-only per-feature `[-1,1]` min-max is feasible only as a disclosed joint scale fallback. |
| `SCFGAN_OUTPUT_SCALE_MISSING` | B | HIGH | HIGH | HIGH | HIGH | Tanh is exact, but raw/normalized HF meaning is absent. It may only be decided jointly with HF normalization. |
| `SCFGAN_CONDITION_SEMANTICS` | C | HIGH | CRITICAL | HIGH | CRITICAL | Cycle, voltage, battery type, chemistry and usage statements conflict. |
| `SCFGAN_CONDITION_ENCODING` | C | HIGH | CRITICAL | HIGH | CRITICAL | No real scalar/vector construction or unique synthetic-row-to-cycle mapping is given. |
| `SCFGAN_GENERATOR_ACTIVATION` | B | MEDIUM | MEDIUM | MEDIUM | MEDIUM | Eq.(7) LeakyReLU conflicts with Table 2/Fig.3 ReLU. Configuration-table priority is a possible disclosed fallback, not a resolution. |
| `SCFGAN_BN_CBN` | B | HIGH | HIGH | HIGH | HIGH | Table 2 names CBN, Eq.(8) lacks executable condition-affine mapping. |
| `SCFGAN_GENERATOR_SN_SCOPE` | B | MEDIUM | MEDIUM | MEDIUM | HIGH | Concrete table/figure emphasize discriminator SN; prose says both networks. |
| `SCFGAN_DISCRIMINATOR_DEPTH` | B | MEDIUM | MEDIUM | MEDIUM | HIGH | Eq.(11)/Fig.3 show three hidden transforms; Table 3 lists four. |
| `SCFGAN_NOISE_DIMENSION` | B | LOW | MEDIUM | LOW | MEDIUM | `z~N(0,I)` is exact but latent width is absent; a small explicit non-performance fallback is possible. |
| `SCFGAN_AUGMENTATION_QUANTITY` | B | HIGH | HIGH | MEDIUM | HIGH | Existing Phase 1 Human fallback is retained; it remains non-paper-exact and author-overridable. |
| `TIMESNET_INPUT_SEMANTICS` | C | HIGH | CRITICAL | HIGH | CRITICAL | PLAN-CHANGE-0010 supplies an execution fallback, but paper evidence does not uniquely fix the real tensor. |
| `TIMESNET_TARGET_ALIGNMENT` | C | HIGH | CRITICAL | HIGH | CRITICAL | A 10-step window does not specify endpoint, horizon, or prediction length. |
| `TIMESNET_TOPK` | B | MEDIUM | HIGH | MEDIUM | MEDIUM | The top-k concept is exact; k is missing. A fixed reference-core k can be disclosed, but is not target-paper exact. |
| `TREND_TIMESNET_CONFIGURATION` | C | HIGH | HIGH | HIGH | HIGH | “Deep” and multi-scale are descriptive only; depth, widths, kernels and merge are missing. |
| `SEASONAL_LIGHTWEIGHT_DEFINITION` | C | HIGH | HIGH | HIGH | HIGH | “Lightweight” and adaptive detection are not operationalized. |
| `ITRANSFORMER_INPUT_SEMANTICS` | C | HIGH | CRITICAL | HIGH | CRITICAL | Residual branch and voltage/temperature channel-mixing prose do not identify the real tensor. |
| `ITRANSFORMER_TARGET_ALIGNMENT` | C | HIGH | CRITICAL | HIGH | CRITICAL | Generic forecasting equations do not specify component target endpoint/horizon. |
| `ITRANSFORMER_ARCH_PARAMETERS` | C | HIGH | HIGH | HIGH | HIGH | Heads, layers, width, dropout and claimed graph mechanism are not executable. |
| `FUSION_ERROR_DEFINITION` | D | CRITICAL | CRITICAL | CRITICAL | CRITICAL | Current-target component error needs the true target and would leak a label into fusion input. |
| `FUSION_RATIO_DEFINITION` | D | CRITICAL | CRITICAL | CRITICAL | CRITICAL | No numerator/denominator or inference-time availability is stated. |
| `FUSION_D_SEMANTICS` | B | MEDIUM | HIGH | MEDIUM | MEDIUM | `d=1` is algebraically compatible with 9d/9/3d/Table 4, but not stated. |
| `FUSION_FEATURE_SELECTION` | A | HIGH | HIGH | MEDIUM | HIGH | Eq.(28) identifies three component predictions and Table 4 weights first 3d fields; order remains an explicit cross-evidence chain to document if authorized. |
| `FUSION_FINAL_OUTPUT_DOMAIN` | A | HIGH | HIGH | MEDIUM | HIGH | Fig.5, Eq.(28) context and framework objective repeatedly call the result Final SOH. The component-to-SOH scale bridge remains separately unresolved. |
| `FUSION_LOSS` | C | HIGH | CRITICAL | HIGH | CRITICAL | Eq.(24) names `L_pred` and `L_weight` without executable definitions. |
| `FUSION_LAMBDA` | C | MEDIUM | HIGH | HIGH | HIGH | The fusion coefficient is absent; SCF-GAN lambda=.1 cannot be reused. |
| `GMTC_TRAINING_ORCHESTRATION` | B | HIGH | HIGH | MEDIUM | HIGH | “End-to-end or carefully staged” is ambiguous. A staged contract is a possible declared fallback once component/fusion semantics exist. |
| `SHARED_ADVERSARIAL_TRAINING` | D | HIGH | HIGH | CRITICAL | CRITICAL | No module, loss, schedule, weight, or gradient path is reported; invention would create a different method. |

`TOTAL_ACTIVE_BLOCKERS=28`

`CLASS_A_COUNT=2`

`CLASS_B_COUNT=12`

`CLASS_C_COUNT=11`

`CLASS_D_COUNT=3`

## Target-alignment audit

The paper-exact evidence is only `SEQ_LEN=10`. Fig.5 uses component notation with time subscript `t` in an SOH-estimation diagram, which weakly supports a same-endpoint reading. It does not override the generic forecasting notation or create a target offset.

`SAME_ENDPOINT_ALIGNMENT_EVIDENCE=WEAK_TO_MODERATE_PAPER_CROSS_EVIDENCE__FIG5_TIME_T_AND_SOH_ESTIMATION_CONTEXT`

`NEXT_STEP_ALIGNMENT_EVIDENCE=NOT_FOUND__NO_T_PLUS_1_OR_HORIZON_STATEMENT`

`TARGET_ALIGNMENT_RECOMMENDED_CLASS=CLASS_C`

`TARGET_ALIGNMENT_FALLBACK_FEASIBILITY=HUMAN_RESULT_BLIND_FALLBACK_POSSIBLE_BUT_NOT_SAFE_TO_APPLY_WITHOUT_PRIMARY_SOURCE_OR_EXPLICIT_AUTHORIZATION`

No endpoint convention is chosen here.

## HF-normalization audit

No paper statement identifies `[-1,1]`, `[0,1]`, min-max, z-score, or standardization. Generator Tanh and Fig.5’s Normalize label are insufficient to turn min-max into a paper fact.

`HF_NORMALIZATION_BLOCKER_CLASS=CLASS_B`

`HF_NORMALIZATION_FALLBACK_FEASIBILITY=FEASIBLE_FOR_HUMAN_AUTHORIZATION_ONLY__TRAIN_ONLY_PER_FEATURE_MINMAX_TO_MINUS1_PLUS1__JOINTLY_WITH_SCFGAN_OUTPUT_SCALE`

The proposed fallback would fit each feature only on real TRAIN HFs, use that same normalized domain for GAN and branch inputs, transform validation/test only, and never use downstream performance to modify the scaler. It is not applied in this run.

## Fusion inference-feasibility audit

| Candidate feature definition | Train-time available? | Inference-time available? | Needs true component target? | Label-leakage risk | Decision |
|---|---|---|---|---|---|
| Current-target signed/absolute error, `error_t=f(yhat_t,y_t)` | Yes | No | Yes | Direct / CRITICAL | REJECT |
| Previous-cycle realized error | Potentially | Not established | Requires past component construction | High; noncausal STL and timing unspecified | Not paper-supported; do not adopt |
| Learned error estimate | Potentially | Potentially | No current label after training | Adds unreported model | REJECT |
| Historical aggregate error | Potentially | Potentially | No current label | Split/window/inference update rule absent | Not paper-supported; do not adopt |

`FUSION_ERROR_BLOCKER_CLASS=CLASS_D`

`CURRENT_TARGET_ERROR_CAUSES_LABEL_LEAKAGE=True`

`FUSION_ERROR_EXECUTABLE_CANDIDATE_EXISTS=False`

| Candidate ratio definition | Inference available? | Label dependence | Dimensional consistency | Decision |
|---|---|---|---|---|
| prediction / true component | No | Yes | Possible | REJECT — direct label use |
| component / true total | No | Yes | Unclear | REJECT — target/total meaning missing |
| prediction proportion across branches | Potentially | No | Possible | Not paper-supported; signed/zero/scale rules missing |
| error ratio / historical ratio | Not established | Often yes | Unclear | Not paper-supported |

`FUSION_RATIO_BLOCKER_CLASS=CLASS_D`

`FUSION_RATIO_EXECUTABLE_CANDIDATE_EXISTS=False`

`D_EQUALS_ONE_FALLBACK_FEASIBILITY=FEASIBLE_FOR_HUMAN_AUTHORIZATION_ONLY__MINIMUM_INTERNAL_CONSISTENCY_NOT_PAPER_EXACT`

## Architecture and orchestration priority audit

Tables 2–4 are explicit configuration evidence; equations establish mathematical roles, while Figures are schematic architecture evidence. A future result-blind fallback may privilege an explicit configuration table over generic equation prose for activation, discriminator depth, and SN scope, but must retain the contradiction record and label the choice `HUMAN_FALLBACK`, never `PAPER_EXACT`.

`SCFGAN_CONFIGURATION_TABLE_PRIORITY_FALLBACK=FEASIBLE__CLASS_B__CONTRADICTION_HISTORY_RETAINED`

`TIMESNET_TOPK_FALLBACK=FEASIBLE__CLASS_B__REFERENCE_ARCHITECTURE_VALUE_IS_NOT_TARGET_PAPER_PARAMETER`

`TRAINING_ORCHESTRATION_FALLBACK_FEASIBILITY=CONDITIONALLY_FEASIBLE__STAGED_RESULT_BLIND_PIPELINE_ONLY_AFTER_TARGET_AND_FUSION_SEMANTICS_ARE_RESOLVED`

`SHARED_ADVERSARIAL_POLICY_RECOMMENDATION=OMIT_ONLY_UNDER_EXPLICIT_LIMITATION_OR_PAUSE_STRICT_NUMERICAL_REPRODUCTION__DO_NOT_INVENT`

## Ceiling

`STRICT_REPRODUCTION_CEILING_WITHOUT_AUTHOR=FRAMEWORK_REPRODUCTION_ONLY`

Without an author response or equivalent primary source, a disclosed fallback numerical variant may become possible after separate Human decisions. It cannot be called paper-exact numerical reproduction while Class C/D blockers remain, particularly component target alignment, inference-safe fusion errors/ratios, fusion loss, SCF-GAN condition semantics, and shared adversarial training.

`STRICT_NUMERICAL_REPRODUCTION_CONTINUATION_RECOMMENDATION=PAUSE_FOR_PRIMARY_SOURCE`
