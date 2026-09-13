# Numerical Reproduction Blocker Resolution Matrix — Phase 8A

| Blocker ID | Class | Phase | Missing or contradictory item | Paper-evidence status | Result-blind fallback possible? | Primary source preferred? | Author reply may override? | Dependent modules | Blocks Phase 9? |
|---|---|---|---|---|---|---|---|---|---|
| `STL_REAL_BATTERY_PERIOD_MISSING` | A | 2 | Real period and smoother contract | MISSING | Yes, but requires separate Human scientific authority | Yes | Yes | STL, all branch targets, fusion | Yes |
| `HF_NORMALIZATION_METHOD_MISSING` | A | 8 | Method, train-fit scope, inverse transform | MISSING | Yes, with explicit leakage-safe scaler policy | Yes | Yes | SCF-GAN, branch inputs | Yes |
| `SCFGAN_OUTPUT_SCALE_UNRESOLVED` | C | 7/8 | Meaning of Tanh output relative to HF scale | UNRESOLVED | Possible, but high fidelity cost | Strongly | Yes | SCF-GAN → branch seam | Yes |
| `SCFGAN_CONDITION_SEMANTICS_CONTRADICTORY` | B | 7 | Cycle/voltage/battery-type/chemistry/usage condition set | CONTRADICTORY | Possible, high fidelity cost | Strongly | Yes | SCF-GAN | Yes |
| `SCFGAN_CONDITION_ENCODING_MISSING` | C | 7/8 | Scalar/vector construction and cycle identity link | MISSING | Possible only with explicit Human fallback | Strongly | Yes | synthetic-label pairing | Yes |
| `SCFGAN_GENERATOR_ACTIVATION_BN_CBN_SN_DEPTH_CONTRADICTIONS` | B | 7 | Eq./table/figure architecture differences | CONTRADICTORY | Yes, ambiguity-preserving core exists; real variant remains costly | Strongly | Yes | real SCF-GAN | Yes |
| `SCFGAN_NOISE_DIMENSION_MISSING` | D | 7 | Latent width | MISSING | Yes, explicit fallback | Preferred | Yes | real SCF-GAN | Yes |
| `AUGMENTATION_QUANTITY_AMBIGUITY` | B | 1/7 | Five datasets versus fivefold wording | AMBIGUOUS | Existing Human fallback exists; author override retained | Strongly | Yes | augmentation population | Yes |
| `BRANCH_TARGET_ALIGNMENT_MISSING` | C | 3/5/8 | Ten-step endpoint/horizon/pred_len | MISSING | Possible only with explicit Human fallback | Strongly | Yes | all branches, fusion | Yes |
| `TIMESNET_BRANCH_VARIANT_PARAMETERS_MISSING` | A | 5 | Deep/lightweight, top-k, kernel, merge, input axes | MISSING | Yes, with explicit fallback | Preferred | Yes | Trend/Seasonal branches | Yes |
| `ITRANSFORMER_INPUT_AND_ARCH_PARAMETERS_MISSING` | C | 3 | Real input tensor semantics, architecture, graph claim | AMBIGUOUS/MISSING | Possible, high fidelity cost | Strongly | Yes | Residual branch | Yes |
| `FUSION_ERROR_RATIO_D_FEATURE_OUTPUT_LOSS_SEMANTICS_MISSING` | C | 6/8 | Inference-safe builder, d, slots, final domain, losses/lambda | MISSING/AMBIGUOUS | Some details may be fallback-able; error/ratio safety requires strict review | Strongly | Yes | Fusion, final output | Yes |
| `GMTC_TRAINING_ORCHESTRATION_UNSPECIFIED` | B | 4/8 | End-to-end versus staged choice | AMBIGUOUS | Yes, explicit Human fallback | Strongly | Yes | Full framework | Yes |
| `SHARED_ADVERSARIAL_TRAINING_UNSPECIFIED` | C | 4/8 | Loss, schedule, weight, gradient path | NOT_SPECIFIED | Possible only by materially changing method | Strongly | Yes | Full framework | Yes |

Class definitions: `A` = paper omission where a reasonable result-blind fallback may be possible; `B` = internal contradiction where a fallback has higher fidelity cost; `C` = core semantics where primary-source clarification is strongly preferred; `D` = bounded engineering detail. All entries remain active until an explicitly recorded resolution/override occurs.
