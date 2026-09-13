# Phase 8B — Human Fallback Decision Package

`STATUS=DECISION_PACKAGE_ONLY`

`HUMAN_FALLBACKS_APPLIED=0`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`REAL_DATA_USED=False`

`TEST_LOADED=False`

## Recommend author first

| Item | Why primary source is preferred | Cost of guessing | Reversibility |
|---|---|---|---|
| Branch input tensor and target alignment | Determines what a 10-step HF window predicts for all three branches. | Changes task from estimation to forecasting or sequence reconstruction. | Low once models/results exist. |
| Fusion errors and ratios | Determines whether 9-feature inputs are available without labels at inference. | Direct label leakage or invention of an unreported auxiliary model. | Very low; contaminates full fusion semantics. |
| STL real period/smoothers and component-target construction | Creates all real supervision labels. | Different decompositions yield different branch targets. | Low after training. |
| SCF-GAN condition set and encoding | Determines conditional input and synthetic-row/cycle pairing. | Semantically invalid conditioning or untraceable labels. | Low after augmentation. |
| Fusion loss, lambda, and shared adversarial method | Determines the actual optimization objective. | A new objective, not numerical reproduction. | Low after optimization. |
| Exact TimesNet/iTransformer variants | “Deep/lightweight” and graph claims have no executable configuration. | Major architecture-fidelity loss. | Medium. |

## Safe to request as a Human result-blind fallback

| Item | Proposed fallback | Evidence / cost | Leakage risk | Reversibility |
|---|---|---|---|---|
| HF normalization plus GAN output scale | Fit real TRAIN HF1/HF2/HF3 independently with min-max to `[-1,1]`; retain generated HFs in that domain; transform-only outside TRAIN. | Tanh + Fig.5 Normalize motivate a coherent scale contract, but no paper-exact evidence. | Low if train-only fit is enforced. | High before GAN/model execution. |
| SCF-GAN configuration conflicts | Prefer explicit Tables 2/3 as a declared configuration fallback, while preserving equation/figure conflict records. | Deterministic and result blind; may differ from code. | Low. | High before GAN training. |
| Noise dimension | Small fixed explicit latent width. | Bounded engineering detail, not target-paper parameter. | Low. | High. |
| TimesNet top-k | Fixed standard-core reference value, declared as fallback. | Changes period mix but not the paper’s broad core topology. | Low. | High before branch training. |
| Fusion `d=1` | Use scalar component prediction/error/ratio slots to make 9d=9 and first 3d=3. | Algebraically consistent but unreported. | Low by itself; cannot solve error/ratio semantics. | High. |
| Training orchestration | Staged training only, after all target/fusion semantics are independently frozen. | Compatible with the paper’s “end-to-end or carefully staged” wording, but not a reported experiment choice. | Low if partition isolation remains enforced. | Medium. |
| Augmentation count | Retain the existing Phase 1 `6N` Human fallback with its limitation and author override. | Already frozen result blind; paper’s five-set/fivefold tension remains. | Low if TRAIN-only. | High before materialization. |

## Do not fallback without strong justification

| Item | Why a routine fallback is unsafe | Recommendation |
|---|---|---|
| Current-target component errors in fusion | Requires true current component target inside a future fusion input. | Reject: label leakage. |
| Ratios for fusion | No definition or inference-time source; plausible versions differ materially. | Author first; do not invent. |
| Branch target endpoint/horizon | Same-step, next-step and sequence targets are different learning tasks. | Author first or a separately approved scientific deviation. |
| SCF-GAN condition semantics/encoding | Conditions control augmentation identity and label pairing. | Author first. |
| Shared adversarial training | No loss, module, schedule, or gradient path is reported. | Omit only under an explicit limitation, or pause strict reproduction; never invent. |
| Fusion loss/Lambda | Determines the full objective and trade-off. | Author first. |

## Minimum Executable Fallback Sets

`MEFS_LEVEL2=NO_COMPLETE_SAFE_SET_CURRENTLY`

Numeric synthetic end-to-end integration remains blocked because no paper-supported, inference-safe fusion error/ratio bridge exists. A conditional Level-2 package would require, at minimum: a joint HF normalization/GAN-output-scale fallback; explicit branch target alignment and `pred_len`; a primary-source or separately authorized inference-safe fusion error/ratio policy; `d`/slot ordering; final output scale/domain; and loss/orchestration policy. This is not authorized by this package.

`MEFS_LEVEL3=NO_COMPLETE_SAFE_SET_CURRENTLY`

Real-data fallback integration additionally requires a fixed real STL period/smoother policy, component-label construction, SCF-GAN condition semantics/encoding and output pairing, source/target augmentation execution details, and exact branch architecture configurations. It remains impossible to authorize responsibly while the Level-2 fusion seam is unresolved.

## Three-level reporting recommendation

Future reporting should separate:

1. `FRAMEWORK_REPRODUCTION` — synthetic structural interfaces and contracts only.
2. `FALLBACK_NUMERICAL_REPRODUCTION` — any future result-blind, fully disclosed Human-authorized variant.
3. `PAPER_EXACT_NUMERICAL_REPRODUCTION` — unavailable until primary-source blockers are resolved.
