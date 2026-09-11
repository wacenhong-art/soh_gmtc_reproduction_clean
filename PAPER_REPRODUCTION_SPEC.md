# PAPER REPRODUCTION SPECIFICATION

**Status:** `PHASE_0_SPECIFICATION_FREEZE=COMPLETE_WITH_REGISTERED_AMBIGUITIES`

**Authority:** the local 17-page paper PDF is the sole technical source for this document. This is a specification extraction, not a model implementation, protocol resolution, or performance result.

## Evidence convention

Every field records `VALUE`, `STATUS`, `SOURCE`, `SOURCE_LOCATION`, and `NOTES`. Status is exactly one of `EXACT`, `AMBIGUOUS`, `MISSING`, or `INFERRED`. `AMBIGUOUS`, `MISSING`, and `INFERRED` values are never paper parameters.

**Source key:** `PDF = Data-augmented SOH estimation for lithium-ion batteries under.pdf` (project root), DOI `10.1016/j.energy.2025.139463`.

## 1. Research task and datasets

| Field | VALUE | STATUS | SOURCE / SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| PAPER_TITLE | Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach | EXACT | PDF p.1 | — |
| DOI | 10.1016/j.energy.2025.139463 | EXACT | PDF p.1 | — |
| RESEARCH_OBJECTIVE | Estimate battery SOH under small-sample and incomplete charging-voltage conditions by SCF-GAN augmentation plus STL and component-specific temporal modeling. | EXACT | PDF pp.1, 5, 8–10, 16 | — |
| SOH_DEFINITION | Residual capacity relative to nominal capacity. | EXACT | PDF p.1 | Dataset-specific executable denominator remains ambiguous; see `PA-001`. |
| TARGET_VARIABLE | Capacity-derived SOH | EXACT | PDF pp.1, 3, 10 | Formula/denominator not explicitly operationalized. |
| SOH_UNIT | Percent-form metrics are reported; internal label scale is not stated. | AMBIGUOUS | PDF pp.1, 10, 14–16 | See `PM-001`. |
| DATASETS | NASA Ames PCoE; CALCE; Tsinghua | EXACT | PDF p.3 §2.1 | — |
| DATA_SOURCE | Publicly recognized datasets; detailed download URL/version unavailable. | MISSING | PDF p.3 §2.1; refs.35–37 | — |
| NUMBER_OF_CELLS / CYCLES | Dataset-level counts and per-cell cycle counts are not provided in extractable text/Table 1. | MISSING | PDF p.3 Table 1 | — |

### Dataset-specific conditions

| Dataset | Chemistry / capacity | Charge protocol | Discharge protocol | Temperature / EOL | STATUS | SOURCE_LOCATION |
|---|---|---|---|---|---|---|
| NASA | LiCoO2, 2.0 Ah, commercial 18650 | CC 1.5 A to 4.2 V; CV to 20 mA | CC 2.0 A; cutoff 2.7 V B5, 2.5 V B6, 2.2 V B7 | 24 °C; terminates at 70–80% initial rated capacity | EXACT | PDF p.3 §2.1, Table 1 |
| CALCE | LiCoO2, CS2 pouch, 1.1 Ah | CC 0.5C/0.55 A then CV to 0.05 A | 1C/1.1 A to 2.7 V | 25 °C; EOL 70% (0.77 Ah) | EXACT | PDF p.3 §2.1, Table 1 |
| Tsinghua | Li_x(NiCoMn)_(1/3)O2 pouch, 24 Ah | 4C CC-CV to 4.2 V; cutoff 0.05C | 4C CC to 2.5 V | 25 °C; EOL 80% or 1800 cycles; 10 s sample interval | EXACT | PDF p.3 §2.1, Table 1 |

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| BATTERY_IDS | NASA B0005/B0006/B0007/B0018 (and text references B5/B6/B7/B18); CALCE CS2-35/36/37/38; Tsinghua Th4/5/6/7/8 | EXACT | PDF pp.4, 9–15; Tables 5–8 | Naming normalization must be documented later. |
| SOURCE_BATTERIES | B0005/B0006 and CS2-35/CS2-37 are examples of source batteries; Th6 and CS2-37 are source batteries in conditional-generation experiment. | AMBIGUOUS | PDF pp.9, 11 | Paper supplies examples, not one universal roster; `PA-002`. |
| TARGET_BATTERIES | B0007/B0018 and CS2-36/CS2-38 are examples of targets; Th5/CS2-35 in generation experiment. | AMBIGUOUS | PDF pp.9–11 | Scope differs by experiment; `PA-002`. |

## 2. Raw input and health indicators

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| RAW_SIGNALS | Per-cycle voltage, current, temperature, and capacity are available in NASA description. | EXACT | PDF p.3 §2.1 | Model consumes the three HFs, not a stated raw multichannel tensor. |
| HEALTH_INDICATORS / FEATURE_NAMES | `HF1`, `HF2`, `HF3` | EXACT | PDF p.3 §2.2 | `HF=[HF1,HF2,HF3]`. |
| HF1_EXTRACTION | `t_Vmax - t_0`: CC duration from CC start to first upper-voltage threshold reach/exceedance. | EXACT | PDF p.3 Eq.(1) | Upper threshold is dataset/charging-window dependent. |
| HF2_EXTRACTION | `t_Imin - t_Vmax`: CV duration from upper-voltage reach to termination-current time. | EXACT | PDF p.3 Eq.(2) | — |
| HF3_EXTRACTION | `t_end - t_start`: time from first voltage >=3.8 V to first voltage >=4.0 V. | EXACT | PDF p.3 Eq.(3) | — |
| VOLTAGE_INTERVALS | Full source example 3.6–4.2 V; partial target example 3.8–4.0 V; HF3 3.8–4.0 V. | EXACT | PDF pp.3, 5, 7, 10–11 | Operational selection of all intervals remains ambiguous. |
| TEMPERATURE/CURRENT/CAPACITY_FEATURES | Temperature/current/capacity exist as measurements, but no temperature/current/capacity feature beyond HF durations is specified as temporal-model input. | MISSING | PDF pp.3, 6–10 | — |
| NORMALIZATION / SCALING / INTERPOLATION / MISSING_VALUE_HANDLING / SMOOTHING | Not stated. | MISSING | PDF §§2–4 | Must not be assumed. |
| DIRECT_SOH_OR_HF_INPUT | Paper constructs HF dataset then uses HFs for SOH estimation; exact tensor composition after STL is unclear. | AMBIGUOUS | PDF pp.3, 6–10 | `PA-003`. |

## 3. Sample construction and split protocol

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| SEQUENCE_LENGTH / INPUT_WINDOW | Dedicated 10-step sliding windows for each architecture. | EXACT | PDF p.6 §3.3 | Forecast horizon/stride not stated. |
| FORECAST_HORIZON / STRIDE / FIRST_VALID_SAMPLE / LAST_VALID_SAMPLE | Not stated. | MISSING | PDF §§3–4 | — |
| TEMPORAL_ORDER | Target available cycles are chronologically split. | EXACT | PDF p.10 §4 | — |
| TRAIN_SPLIT / VALIDATION_SPLIT / TEST_SPLIT | First 60% / next 20% / final 20% of target available cycles. | EXACT | PDF p.10 §4 | See conflict with 20% small-sample paragraph. |
| SMALL_SAMPLE_PROTOCOL | Original target training set is described as first 20% of total cycle life, with five complete synthetic datasets generated per target set. | EXACT | PDF p.10 §4.2 | Relationship to 60/20/20 is unresolved; `PA-004`. |
| GAN_TRAINING_POPULATION | Source batteries with full 3.6–4.2 V data train SCF-GAN exclusively; target batteries must not leak into GAN training. | EXACT | PDF pp.7, 9–10 | Exact roster is ambiguous. |
| AUGMENTATION_TIMING | Generated HFs train subsequent temporal model. | EXACT | PDF p.5; pp.8–10 | Exact split-relative insertion and real/synthetic ratio unresolved; `PA-005`. |
| VALIDATION_FOR_MODEL_SELECTION | Validation early stopping, patience 50. | EXACT | PDF pp.9–10 | Any other selection/tuning rule missing. |
| TEST_USAGE | Final 20% target cycles are test partition. | EXACT | PDF p.10 | Test re-use policy missing. |
| PAPER_PROTOCOL_AMBIGUITY | True | EXACT | PDF p.10 §4 vs §4.2 | `PA-004` blocks Phase 1 advancement. |

## 4. STL specification

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| STL_INPUT | `X_t`, an unspecified input time series. | AMBIGUOUS | PDF p.6 Eq.(17) | Could be SOH, HFs, or a multivariate representation; `PA-003`. |
| STL_OUTPUT | `T_t` Trend, `S_t` Seasonal, `R_t` Residual; additive `X_t=T_t+S_t+R_t`. | EXACT | PDF p.6 Eq.(17) | — |
| TREND / SEASONAL / RESIDUAL_DEFINITION | Long-term fade / recurring short-term fluctuations / irregular noise-regeneration-stochastic effects. | EXACT | PDF p.6 §3.3 | — |
| STL_PERIOD | Symbol `p` described with monthly example `p=12`; battery-specific value absent. | MISSING | PDF p.6 §3.3 | `PM-002`. |
| STL_WINDOW / SEASONAL_WINDOW / TREND_WINDOW / ROBUST / CENTERING / BOUNDARY_HANDLING | Not stated. | MISSING | PDF §3.3 | `PM-002`. |
| STL_BEFORE_OR_AFTER_WINDOWING / PER_BATTERY_OR_GLOBAL | Not stated. | MISSING | PDF §§3–4 | `PA-006`. |
| FUTURE_INFORMATION_RISK | Cannot be assessed until fit scope and boundary handling are specified. | AMBIGUOUS | PDF §3.3 | Record only; no method substitution. |

## 5. TimesNet specification

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| TIMESNET_ROLE / INPUT_BRANCHES | Trend -> TimesNet; Seasonal -> TimesNet variant with adaptive cycle detection. | EXACT | PDF pp.6, 8–9, 16 | — |
| METHOD | FFT top-k frequencies; reshape each 1D sequence to 2D by `p_i=ceil(T/f_i)`; parameter-shared Inception blocks; adaptive merge. | EXACT | PDF pp.6–7 Eqs.(18)–(20) | Generic method description, not a complete configured implementation. |
| INPUT_DIM / OUTPUT_DIM / MODEL_DIM / N_LAYERS / TOP_K / KERNEL_SIZE / DROPOUT / ACTIVATION / NORMALIZATION / OUTPUT_HEAD | Not stated. | MISSING | PDF §§3.3–3.4, Table 4 | `PM-003`. |
| SEQUENCE_LENGTH | 10-step branch windows. | EXACT | PDF p.6 §3.3 | — |

## 6. iTransformer specification

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| ITRANSFORMER_ROLE / INPUT_BRANCH | Residual -> iTransformer; channel mixing for nonlinear dependencies. | EXACT | PDF pp.6–7, 9, 16 | — |
| EMBEDDING_METHOD | MLP Embedding of `X_:,n`; MLP Projection after transformer blocks. | EXACT | PDF p.7 Eqs.(21)–(23) | — |
| RESIDUAL_BRANCH_TARGET | Future variables `Y` in generic equation; exact SOH/component target tensor unspecified. | AMBIGUOUS | PDF p.7 Eqs.(21)–(23) | `PA-007`. |
| INPUT_DIM / D_MODEL / N_HEADS / N_LAYERS / D_FF / DROPOUT / ACTIVATION / NORMALIZATION / OUTPUT_HEAD | Not stated. | MISSING | PDF §3.5 | `PM-004`. |
| SEQUENCE_LENGTH | 10-step branch windows. | EXACT | PDF p.6 §3.3 | — |

## 7. Adaptive fusion specification

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| FUSION_INPUTS | Trend/seasonal/residual predictions, errors, and ratios; 9 features. | EXACT | PDF p.7 §3.6; p.9 Table 4 | Provenance/split-safe computation of errors and ratios unresolved; `PA-008`. |
| FUSION_TYPE | Attention-enhanced Bi-LSTM with component attention and time-step attention. | EXACT | PDF p.7 §3.6; Fig.4; p.9 Table 4 | — |
| WEIGHT_NORMALIZATION / SOFTMAX | `w=Softmax(W4 z3+b4)`, `w∈R^3`, sum weights=1. | EXACT | PDF p.7 Eq.(27) | — |
| FINAL_OUTPUT_FORMULA | `ŷ = Wp·ReLU(Wf(Σ_i w_i X_comp,i)+bf)+bp`. | EXACT | PDF p.7 Eq.(28) | — |
| LOSS | `L_total=L_pred+lambda·L_weight`; definitions/coefficients of terms unspecified. | AMBIGUOUS | PDF p.7 Eq.(24) | `PM-005`. |
| CONFIGURATION | Bi-LSTM input 9, hidden 16 (32 bidirectional), 1 layer, dropout .3; time attention 32 Tanh+Softmax; generator 32→128→64→32→3, LeakyReLU α=.1, BatchNorm1d, dropout .3, Softmax; fusion first 3d last-step features; final 3→16→1 ReLU. | EXACT | PDF p.9 Table 4 | — |

## 8. SCF-GAN specification

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| SCF_GAN_FULL_NAME / PURPOSE | Spectral-normalized conditional GAN with feature matching and conditional batch normalization; generates realistic HF vectors under limited/partial charging. | EXACT | PDF pp.2, 5, 8, 16 | — |
| GENERATOR_INPUT / OUTPUT | `concat(z,c)` to synthetic HF vector `x~=G(z|c)`; output Tanh. | EXACT | PDF p.5 Eqs.(6)–(7); p.8 Table 2 | Latent dimension missing. |
| CONDITION_VARIABLE | Called cycle-period condition, voltage range, and (elsewhere) battery characteristics/usage conditions. | AMBIGUOUS | PDF pp.5, 7–8, 11 | `PA-009`. |
| GENERATOR_ARCHITECTURE | Three fully connected hidden layers, ReLU + CBN, then fully connected Tanh. | EXACT | PDF p.8 Table 2 | Eq.(7) says LeakyReLU/BN; activation wording conflicts; `PA-010`. |
| DISCRIMINATOR_INPUT / ARCHITECTURE | `concat(x or G(z|c),c)`; four fully connected LeakyReLU hidden layers with spectral normalization; sigmoid output. | EXACT | PDF p.5 Eq.(11); p.9 Table 3 | Equation displays three hidden states while table lists four; `PA-010`. |
| SPECTRAL_NORMALIZATION | Discriminator layers exact; prose says both generator/discriminator weights, but generator Table 2 does not list it. | AMBIGUOUS | PDF pp.5, 8–9 | `PA-010`. |
| FEATURE_MATCHING | L_FM uses discriminator third-layer feature extractor; generator loss adds `lambda L_FM`; lambda=.1 selected from {.01,.1,1.0}. | EXACT | PDF p.5 Eqs.(12)–(16) | — |
| LOSS_G / LOSS_D | Equations (13)–(14). | EXACT | PDF p.5 | — |
| GAN_EPOCHS / BATCH_SIZE / LR / ADAM_BETAS / LATENT_DIM / HIDDEN_DIMS | Not stated. | MISSING | PDF §§3.1–3.2 | `PM-006`. |
| SYNTHETIC_SAMPLE_COUNT | Five complete synthetic datasets per original target set. | EXACT | PDF pp.10–11 | “fivefold” wording versus five generated sets needs resolution; `PA-005`. |
| REAL_SYNTHETIC_RATIO | Not stated. | MISSING | PDF §4.2 | `PM-006`. |

## 9. Full paper architecture flow

```text
Full-range source HFs ──train──> SCF-GAN
Partial-range target HFs + condition ──generate──> augmented HF data
                                         ↓
                                      STL: X=T+S+R
                         ┌──────────────┼──────────────┐
                 Trend -> TimesNet  Seasonal -> TimesNet  Residual -> iTransformer
                         └──────────────┼──────────────┘
                         predictions/errors/ratios (9d)
                                         ↓
                component attention + Bi-LSTM + time attention
                                         ↓
                          softmax 3 weights + prediction head
                                         ↓
                                        SOH
```

`PAPER_ARCHITECTURE_FLOW=EXTRACTED`

`VERIFIED_FROM_PAPER=True`

**Sources:** PDF pp.5–10, Fig.5, Eqs.(17)–(29), Tables 2–4, conclusion p.16. Exact timing of STL relative to augmentation/window construction remains `PA-006`; the stated high-level order is paper-supported.

## 10. Training and metrics

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| OPTIMIZER | Adam for SCF-GAN; fusion Eq.(29) is Adam-like with weight decay 1e-5. | EXACT | PDF pp.5, 7 | Optimizer for all temporal branches not explicitly named. |
| LEARNING_RATE / BATCH_SIZE | .001 / 32 for all models unified configuration. | EXACT | PDF p.9 | — |
| EARLY_STOPPING / PATIENCE | Validation-loss early stopping, patience 50. | EXACT | PDF pp.9–10 | — |
| EPOCHS / INITIALIZATION / LR_SCHEDULER / GRADIENT_CLIPPING / RANDOM_SEED | Not stated. | MISSING | PDF §§3–4 | `PM-007`. |
| HARDWARE | Intel Core i5-13500, 16 GB RAM. | EXACT | PDF p.10 | GPU not reported. |
| SOFTWARE | Python 3.9; “2.0.1 + cu118” reported, evidently PyTorch but text omits library label. | AMBIGUOUS | PDF p.10 | Preserve wording; `PM-008`. |
| RUNS | Ten parallel experiments. | EXACT | PDF p.10 | Seed values and aggregation method missing. |
| METRICS | RMSE, MAE, MAPE, R². | EXACT | PDF p.10 Eq.(30) | Metrics reported as percentages in Tables 5–8. |
| METRIC_SCALE / AVERAGING | MAE%, MAPE%, RMSE%, R² per battery; aggregation across ten runs not specified. | AMBIGUOUS | PDF pp.10, 14–16 | `PM-009`. |

### PAPER_REPORTED_RESULT_TABLE — developed method only (Tables 6–8)

| Dataset | Battery | Model | MAE% | MAPE% | RMSE% | R² | SOURCE_LOCATION |
|---|---:|---|---:|---:|---:|---:|---|
| NASA | B0005 | Developed method | .5678 | .6742 | .7435 | .9345 | PDF p.14 Table 6 |
| NASA | B0006 | Developed method | .6798 | .7890 | .8901 | .9123 | PDF p.14 Table 6 |
| NASA | B0018 | Developed method | .8946 | .9012 | .9123 | .8765 | PDF p.14 Table 6 |
| CALCE | CS-36 | Developed method | .9123 | .9257 | .9423 | .8569 | PDF p.14 Table 7 |
| CALCE | CS-37 | Developed method | .9568 | .9842 | .9875 | .8457 | PDF p.14 Table 7 |
| CALCE | CS-38 | Developed method | .9358 | .9458 | .9553 | .8345 | PDF p.14 Table 7 |
| Tsinghua | Th4 | Developed method | .8597 | .8597 | .8529 | .8979 | PDF p.15 Table 8 |
| Tsinghua | Th6 | Developed method | .8679 | .8798 | .8921 | .8654 | PDF p.15 Table 8 |
| Tsinghua | Th7 | Developed method | .8978 | .8979 | .9872 | .9124 | PDF p.15 Table 8 |
| Tsinghua | Th8 | Developed method | .8905 | .9048 | .9157 | .8472 | PDF p.15 Table 8 |

## 11. Baselines and paper-reported ablations

| Field | VALUE | STATUS | SOURCE_LOCATION | NOTES |
|---|---|---|---|---|
| BASELINE_MODELS | SVR, GPR, RF, CNN, LSTM, Transformer. | EXACT | PDF pp.14–16 Tables 6–8 | All use identical HFs per paper. |
| COMPARISON_MODELS | SVR, GPR, RF, CNN, LSTM, Transformer. | EXACT | PDF pp.14–16 Tables 6–8 | All use identical HFs per paper. |
| GAN_COMPARISON | GAN, CGAN, SCF-GAN generation-quality comparison. | EXACT | PDF pp.10–12 Fig.6 | — |
| FUSION_ABLATION | Attention-based fusion versus simple direct summation. | EXACT | PDF pp.14–15 Fig.10 | — |
| WITHOUT_GAN / WITHOUT_STL / WITHOUT_TIMESNET / WITHOUT_ITRANSFORMER | No explicit full G-MTC removal ablation table found. | MISSING | PDF §§4.2–4.5 | Do not invent variants. |

## Phase 0 extraction verdict

```text
PAPER_FULLTEXT_STATUS=AVAILABLE
ARCHITECTURE_FLOW_EXTRACTED=True
DATA_PROTOCOL_EXTRACTED=True
TARGET_DEFINITION_EXTRACTED=True
FEATURE_SPEC_EXTRACTED=True
STL_SPEC_EXTRACTED=True
TIMESNET_SPEC_EXTRACTED=True
ITRANSFORMER_SPEC_EXTRACTED=True
FUSION_SPEC_EXTRACTED=True
SCF_GAN_SPEC_EXTRACTED=True
TRAINING_SPEC_EXTRACTED=True
METRICS_EXTRACTED=True
AMBIGUITIES_REGISTERED=True
MISSING_PARAMETERS_REGISTERED=True
NO_UNLABELED_GUESSES=True
PHASE_0_STATUS=PASS_WITH_REGISTERED_AMBIGUITIES__ADVANCEMENT_BLOCKED
ADVANCEMENT_ALLOWED=False
BLOCKER=PA-004 unresolved conflict between chronological 60/20/20 target split and first-20%-of-life small-sample training construction.
```
