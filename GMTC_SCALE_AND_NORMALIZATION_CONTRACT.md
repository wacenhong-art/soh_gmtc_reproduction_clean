# G-MTC Scale and Normalization Contract — Phase 8A

## Evidence boundary

Fig.5 visibly contains a `Normalize` stage in the health-feature augmentation panel. The paper and current specification do not state its method, fitted statistics/ranges, scope, inverse transform, or relationship to the Generator Tanh output. This contract records scale seams; it does not invent an adapter.

| Domain | Physical/source meaning | Scale evidence | Active status | Integration consequence |
|---|---|---|---|---|
| Raw HF1 | CC duration from charge start to upper-voltage reach | Time-derived quantity; paper does not state output unit/normalization | `UNIT_PARTIALLY_DEFINED__NORMALIZATION_MISSING` | Cannot select raw versus transformed branch input. |
| Raw HF2 | CV duration from upper-voltage reach to termination current | Time-derived quantity; paper does not state output unit/normalization | `UNIT_PARTIALLY_DEFINED__NORMALIZATION_MISSING` | Same. |
| Raw HF3 | Duration from 3.8 V to 4.0 V | Time-derived quantity; paper does not state output unit/normalization | `UNIT_PARTIALLY_DEFINED__NORMALIZATION_MISSING` | Same. |
| Normalized real HF | Fig.5 visually names `Normalize` before augmentation | No method, parameters, fit partition, or inverse transform reported | `UNRESOLVED` | Cannot construct real scale adapter. |
| SCF-GAN output | Generator Tanh is paper-supported | Tanh alone does not identify raw, min-max, standardized, or latent-HF scale | `SCFGAN_OUTPUT_SCALE_UNRESOLVED` | No min-max-to-[-1,1] assumption is authorized. |
| Branch input HF | Plan-Change-0010 fixes feature roles, not their numeric transform | Raw/normalized/Tanh-native input not reported | `BRANCH_INPUT_SCALE_UNRESOLVED` | Numeric synthetic integration is blocked. |
| Observed capacity / SOH | Capacity-domain STL is cross-evidence; final paper output is called SOH | Label scale, conversion, and denominator provenance remain incomplete | `CAPACITY_SOH_CONVERSION_UNRESOLVED` | Component/fusion output scales cannot be reconciled. |
| STL T/S/R | Additive components inherit the selected STL input scale | Real period missing; input capacity-domain cross-evidence only | `REAL_COMPONENT_SCALE_UNMATERIALIZABLE` | No real component target may be created. |
| Fusion output | Fig.5/Eq.(28) call it Final SOH | Conversion from component/capacity domain not stated | `FINAL_OUTPUT_DOMAIN_UNRESOLVED` | No numeric final-output adapter is authorized. |

`SCFGAN_OUTPUT_SCALE_STATUS=UNRESOLVED`

`HF_NORMALIZATION_METHOD_STATUS=MISSING`

`BRANCH_INPUT_SCALE_STATUS=UNRESOLVED`

`REAL_SYNTHETIC_SCALE_COMPATIBILITY_STATUS=BLOCKED_BY_UNKNOWN_HF_NORMALIZATION_AND_GAN_OUTPUT_SCALE`

`FEATURE_ORDER=[HF1,HF2,HF3]`

`FEATURE_ORDER_PROVENANCE=HUMAN_CROSS_PHASE_FALLBACK_WITH_PAPER_TEXTUAL_SUPPORT`

`MIN_MAX_TO_MINUS1_PLUS1_ASSUMPTION=False`

`TRAIN_ONLY_SCALER_FIT_POLICY=REQUIRED_IF_A_FUTURE_SCALER_IS_AUTHORIZED`

No scaler, normalizer, inverse transform, branch adapter, real feature tensor, or synthetic row is created by this contract.
