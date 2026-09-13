# SCF-GAN Paper Evidence — Phase 7A

## Scope and evidence discipline

This record is a primary-paper audit of the local target PDF.  It does not load a NASA/CALCE row, train a GAN, create a synthetic battery row, access validation/test values, or select an implementation.  `PAPER_EXACT` means the cited text or visual label states the item; `PAPER_CROSS_EVIDENCE` means a bounded inference from multiple cited items; `AMBIGUOUS` and `CONTRADICTORY` preserve uncertainty.

## Evidence ledger

| Item | Paper evidence | Status | Consequence |
|---|---|---|---|
| SCF-GAN role | §2.2 and §3.2 introduce conditional, spectral-normalized GAN augmentation; §3.7 says it synthesizes physically consistent health-feature vectors for limited-voltage target batteries. | `PAPER_EXACT` | Generative augmentation is a framework stage, not a resolved downstream interface. |
| Real feature vector `x` | §3.2 calls `x` a true feature vector; §3.7 calls outputs battery-health feature vectors. | `PAPER_EXACT__DIMENSION_UNSTATED_IN_TEXT` | Text alone does not fix 3 versus 4 dimensions. |
| Generated feature vector | §3.2 Eq.(6) defines `x_tilde=G(z|c)`; §4.2 explicitly says generated `HF1/HF2/HF3`; Fig.3 labels output dimension 4 as `HF1/HF2/HF3/SOH`. | `CONTRADICTORY` | Do not freeze output dimension or claim that SOH is definitely generated. |
| Noise `z` | §3.2: `z ~ N(0,I)`. | `PAPER_EXACT__DISTRIBUTION_ONLY` | Latent dimension is missing. |
| Condition `c` | §3.2 calls it a cycle-period condition; §3.7 calls it a cycle/aging-stage condition and elsewhere battery characteristics/usage; §4.2 calls voltage range conditional; Fig.3 shows Cycle + Battery type; Fig.5 shows Cycle + Voltage. | `CONTRADICTORY` | Neither real condition set nor encoding is executable. |
| Generator | Eq.(7): three hidden states with `LeakyReLU(BN(.))`, Tanh output. Table 2 and Fig.3: three FC hidden layers with ReLU + CBN, Tanh. Fig.3 labels widths 18→256→512→256→4. | `CONTRADICTORY` | Activation, normalization, width/input/output detail cannot be paper-exactly selected. |
| Discriminator | Eq.(11) has three hidden states then output; Table 3 lists four hidden layers plus output; Fig.3 draws three hidden transforms (6→256→512→256→1) with LeakyReLU(0.2), dropout 0.3 and SN. | `CONTRADICTORY` | Hidden-layer count cannot be silently selected. |
| CBN | Paper names Conditional Batch Normalization, while Eq.(8) gives ordinary batch-normalization affine terms without `gamma(c)`/`beta(c)`. | `AMBIGUOUS` | Condition injection inside CBN is not executable. |
| Spectral normalization | §3.2/Table 3/Fig.3 associate SN with discriminator hidden layers; §3.7 says SN applies to both generator and discriminator; Table 2/Fig.3 generator labels CBN but not SN. | `CONTRADICTORY` | Generator SN scope remains unresolved. |
| Feature matching | Eq.(12) uses discriminator intermediate features; nearby prose identifies the third layer in code. | `PAPER_EXACT__LAYER_REFERENCE_RETAINS_DEPTH_CONFLICT` | A third hidden representation is the common named location, but discriminator depth remains contradictory. |
| Losses | Eqs.(13)–(14) define conditional adversarial losses and generator feature matching; `lambda=0.1`, selected by the paper through `{0.01,0.1,1.0}` validation grid search. | `PAPER_EXACT` | Reproduction uses 0.1 without a new search; the original selection history is retained. |
| Optimizer/configuration | Adam is stated. The unified `lr=0.001`, batch 32 and patience 50 statement in §3.7 is not explicitly scoped to every GAN update detail. | `PARTIALLY_SPECIFIED` | LR, betas, epochs, update ratio, initialization and seed remain unreported for exact GAN reproduction. |
| Source/target role | §3.7 gives full-range source examples NASA B0005/B0006 and CALCE CS2-35/CS2-37, and limited-range target examples NASA B0007/B0018 and CALCE CS2-36/CS2-38. | `PAPER_CROSS_EVIDENCE__EXPERIMENT_SPECIFIC` | Consistent with the frozen experiment-level roster; no data are loaded here. |
| Augmentation count | §4.2 says five complete synthetic datasets and separately says training sample size expanded fivefold. | `AMBIGUOUS` | Preserve PA-005 fallback and author override. |

## Figure visual audit

Rendered, inspection-only snapshots are retained at `output/phase7_figure_visual_evidence/page-04.png` (PDF p.4, Fig.3; SHA256 `241930D086FA08EC06321064116E62267681746031923498EA0F26981289B3C8`) and `output/phase7_figure_visual_evidence/page-08.png` (PDF p.8, Fig.5; SHA256 `A7B2EB15E76AF699E927FB85FD54BF76ED5DBF71DCB10DAC7689BE08129ECCAF`).

`FIG3_GENERATOR_LABEL=Generate_HF1_HF2_HF3_SOH_based_on_noise_and_conditions`

`FIG3_GENERATED_FEATURE_DIMENSION=4__HF1_HF2_HF3_SOH`

`FIG3_CONDITION_SET=CYCLE_PLUS_BATTERY_TYPE`

`FIG5_FEATURE_SET=HF1_HF2_HF3_SOH`

`FIG5_CONDITION_SET=CYCLE_PLUS_VOLTAGE`

`FIG5_GENERATED_FEATURE_SET=UNLABELLED_BEYOND_GENERATED_FEATURES`

`FIG5_TOP_BOTTOM_CONNECTION=ABSENT`

Fig.5 visually separates an upper health-feature augmentation panel from a lower STL/hybrid-estimation panel.  No arrow visibly connects `Generated Features` to `STL Decomposition`; therefore no direct SCF-GAN-to-STL interface is paper-exactly established by the figure.

## Frozen Phase 7A statuses

`SCFGAN_ROLE_STATUS=PAPER_EXACT__CONDITIONAL_HEALTH_FEATURE_AUGMENTATION`

`SCFGAN_OUTPUT_SEMANTICS_STATUS=CONTRADICTORY__FIG3_4D_HF1_HF2_HF3_SOH_VS_TEXTUAL_HF123`

`SOH_GENERATED_BY_SCFGAN_STATUS=PAPER_FIGURE_CROSS_EVIDENCE_ONLY__TEXTUAL_CONFIRMATION_ABSENT`

`REAL_FEATURE_DIMENSION_STATUS=CONTRADICTORY__FIG3_4D_VS_TEXT_UNSPECIFIED_OR_HF123`

`GENERATED_FEATURE_DIMENSION_STATUS=CONTRADICTORY__FIG3_4D_VS_TEXT_UNSPECIFIED_OR_HF123`

`GENERATOR_ACTIVATION_CONSISTENCY_STATUS=CONTRADICTORY__EQ7_LEAKYRELU_VS_TABLE2_FIG3_RELU`

`BN_VS_CBN_CONSISTENCY_STATUS=CONTRADICTORY__EQ8_BN_VS_TABLE2_FIG3_CBN`

`CBN_CONDITION_INJECTION_DETAIL_STATUS=MISSING`

`GENERATOR_SN_STATUS=CONTRADICTORY__SECTION37_BOTH_NETWORKS_VS_TABLE2_FIG3_GENERATOR_WITHOUT_SN_LABEL`

`DISCRIMINATOR_SN_STATUS=PAPER_EXACT_FOR_DISCRIMINATOR_HIDDEN_LAYERS`

`SN_SCOPE_CONSISTENCY_STATUS=CONTRADICTORY`

`DISCRIMINATOR_HIDDEN_LAYER_STATUS=CONTRADICTORY__EQ11_3_VS_TABLE3_4_VS_FIG3_3`

`TABLE3_VS_EQ11_CONSISTENCY=CONTRADICTORY`

`NOISE_DISTRIBUTION_STATUS=PAPER_EXACT__STANDARD_NORMAL`

`NOISE_DIMENSION_STATUS=MISSING`

`FEATURE_MATCH_LAYER_STATUS=PAPER_EXACT__THIRD_DISCRIMINATOR_LAYER_IN_CODE__DEPTH_CONTRADICTION_RETAINED`

`FEATURE_MATCH_LAMBDA=0.1`

`FEATURE_MATCH_LAMBDA_PROVENANCE=PAPER_EXACT__PAPER_VALIDATION_GRID_SEARCH_HISTORY`

`CURRENT_REPRODUCTION_GRID_SEARCH=False`
