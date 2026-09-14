# Full Paper-to-Code Pre-Numerical Consistency Matrix

`AUDIT_MODE=FULL_PAPER_TO_CODE_PRE_NUMERICAL_CONSISTENCY_CLOSURE_AUDIT`  
`PAPER_DIRECT_REVIEW=COMPLETE__17_PAGE_PDF_RENDERED__HF_EQUATIONS_VISUALLY_VERIFIED_ON_PDF_PAGE_3`  
`PLAN_CHANGE_0017=NOT_APPLIED`  
`VALIDATION_DATA_USED=False`  
`TEST_LOADED=False`  
`MODEL_TRAINED=False`  
`GRID_SEARCH_RUN=False`

## Evidence priority

`PAPER_EQUATION` and executable `PAPER_TABLE` take precedence over prose labels
for Tier 2 execution. A priority choice never erases the contradictory paper
label; every such choice below retains `PAPER_INTERNAL_CONTRADICTION=True`.

| Item | Paper evidence | Paper value or semantics | Current protocol/code | Status | Blocker class | Required action | Provenance |
|---|---|---|---|---|---|---|---|
| HF1 prose label | PDF p.3 §2.2 text | “constant voltage charging duration” | Code has no prose label | `PASS_WITH_PAPER_INTERNAL_CONTRADICTION` | Tier 3 only | Preserve label and do not use it to rewrite Eq.(1) | PAPER_EXPLICIT_TEXT |
| HF1 Eq.(1) | PDF p.3 Eq.(1) | `t_Vmax-t_0`: CC-start to upper-voltage time | `prepare_split_independent_data.py:73` implements upper-voltage time minus start | `PASS_WITH_PAPER_INTERNAL_CONTRADICTION` | Field mapping pending | Freeze NASA voltage field and crossing rule | PAPER_EQUATION / UNRESOLVED field mapping |
| HF2 prose label | PDF p.3 §2.2 text | “constant current charging duration” | Code has no prose label | `PASS_WITH_PAPER_INTERNAL_CONTRADICTION` | Tier 3 only | Preserve label and do not rewrite Eq.(2) | PAPER_EXPLICIT_TEXT |
| HF2 Eq.(2) | PDF p.3 Eq.(2) | `t_Imin-t_Vmax`: voltage reach to termination-current time | `prepare_split_independent_data.py:74` has this formula | `BLOCKED` | Feature semantics | Freeze `t_Imin` field/sign/crossing definition | PAPER_EQUATION / UNRESOLVED |
| HF3 Eq.(3) | PDF p.3 Eq.(3) | `t_end-t_start`, first 3.8 V to first 4.0 V | `prepare_split_independent_data.py:75` implements the duration | `TIER2_FALLBACK_READY` | Field mapping pending | Include in canonical terminal-voltage field decision | PAPER_EQUATION / UNRESOLVED field mapping |
| NASA charge field semantics | NASA data documentation | terminal voltage is `Voltage_measured`; charger current is `Current_charge` | NASA code currently selects `Voltage_charge` and `Current_charge` | `BLOCKED` | Feature semantics | Decide field mapping result-blind; do not silently retain charger voltage | OFFICIAL_DATASET_DOCUMENTATION |
| NASA endpoint coverage | TRAIN-only structural audit | all 443 legal TRAIN charges contain 3.8/4.0/4.2 threshold reaches and monotonic time | legacy code has no interpolation specification | `BLOCKED` | Feature semantics | Freeze exact-or-linear endpoint policy for `t_Vmax`, `t_Imin`, `t_start`, `t_end` | RAW_TRAIN_STRUCTURAL_EVIDENCE |
| NASA `t_Imin` availability | TRAIN-only structural audit | legacy positive sampled 20 mA condition succeeds B0005 117, B0006 20, B0007 117, B0018 88 | `0<=Current_charge<=0.02` after `Voltage_charge>=4.2` | `BLOCKED` | Feature semantics | Freeze a physical termination-current crossing rule | RAW_TRAIN_STRUCTURAL_EVIDENCE |
| CALCE HF physical mapping | PDF p.3 / CALCE schema audit | same three time/current/voltage concepts, CALCE cutoff 0.05 A | `Test_Time(s)`, `Voltage(V)`, `Current(A)` are present | `BLOCKED` | Future dataset interface | Freeze charge-segment and cutoff endpoint policy before CALCE execution | PAPER_EQUATION / OFFICIAL_DATASET_DOCUMENTATION |
| CALCE chronological cycle identity | CALCE identity audit | no paper-exact raw-row mapping supplied | global cross-workbook lifecycle order is not frozen | `BLOCKED` | Future dataset interface | Separate CALCE identity/ordering Plan Change if CALCE Tier 2 is enabled | RAW_STRUCTURAL_EVIDENCE |
| NASA pairing | PLAN-CHANGE-0015 / pairing protocol | paper does not publish exact rule | discharge-centered nearest preceding unconsumed charge; 168/168/168/132 | `TIER2_FALLBACK_READY` | None for Phase 9A | Revalidate counts before numeric access | WEB_SUPPORTED_ENGINEERING_FALLBACK |
| Split | Advisor fallback / PLAN-CHANGE-0013 | paper 60/20/20 history remains Tier 3 | pair-first chronological 70/20/10, half-open floor | `TIER2_FALLBACK_READY` | None | Keep pair-first execution | ADVISOR_FALLBACK |
| TEST lock | governance | no test use before authorization | runner semantically skips test indices but `scipy.io.loadmat` loads a whole MAT variable before field filtering | `BLOCKED` | Data alignment | Establish test-safe record-access contract or technical interpretation with automated guard | RESULT_BLIND_ENGINEERING_FALLBACK |
| HF normalization | Advisor fallback | paper method missing | MinMax 0–1, TRAIN-only, clipped VAL | `TIER2_FALLBACK_READY` | None once HF exists | ADVISOR_FALLBACK |
| GAN scale bridge | Fig.5 Normalize / Generator Tanh | paper scale unspecified | no real GAN path enabled | `NOT_REQUIRED_FOR_PHASE9A` | Tier 3/future GAN | Resolve before SCF-GAN phase | UNRESOLVED |
| STL | paper topology / Advisor fallback | paper period not executable | TRAIN-only offline STL, period 7, capacity domain | `TIER2_FALLBACK_READY` | None | Keep partition isolation and reconstruction check | ADVISOR_FALLBACK |
| STL index alignment | Tier 2 target contract | no paper target-index rule | input pairs `j-10:j`, component target `j` | `TIER2_FALLBACK_READY` | None | Retain `HF[t-9:t] -> target[t+1]` | ADVISOR_FALLBACK |
| Trend branch | paper topology / Tier2 config | target-paper deep variant missing | `[B,10,3] -> [B,1,3] -> Linear(3,1) -> [B,1,1]` | `TIER2_FALLBACK_READY` | Tier 3 variant only | Use frozen standard-reference core | ADVISOR_FALLBACK |
| Seasonal branch | paper topology / Tier2 config | lightweight variant missing | separate standard-compatible TimesNet instance and scalar adapter | `TIER2_FALLBACK_READY` | Tier 3 variant only | Preserve separate branch identity | ADVISOR_FALLBACK |
| Residual branch | paper topology / Tier2 config | exact iTransformer details missing | `[B,10,3] -> [B,1,3] -> Linear(3,1) -> [B,1,1]` | `TIER2_FALLBACK_READY` | Tier 3 architecture only | Use frozen iTransformer core | ADVISOR_FALLBACK |
| Scalar readout | web task precedent | paper does not state external scalar adapter | exact shape-checked `Linear(3,1,bias=True)` | `TIER2_FALLBACK_READY` | Tier 3 only | No broadcasting or alternate reducer | WEB_SUPPORTED_ENGINEERING_FALLBACK |
| Fusion | paper fusion ambiguous | paper 9-feature Bi-LSTM cannot be inferred at inference time | Tier 2 component sum in capacity domain | `TIER2_FALLBACK_READY` | Tier 3 fusion only | Never construct error/ratio builder | ADVISOR_FALLBACK |
| Capacity-to-SOH | paper SOH framing / Table 1 | exact denominator missing | PA001 supplies NASA 2.0 Ah, but Tier2 protocol makes capacity the primary output | `BLOCKED` | Scale contract | Explicitly activate or defer optional PA001 conversion and metric policy | HUMAN_FALLBACK |
| Optimizer/run budget | PDF §3.7 / current runner | lr .001, batch 32, patience 50; branch optimizer unspecified | runner hardcodes Adam, 3 epochs, seed 42 without a dedicated frozen run-config artifact | `BLOCKED` | Execution/training contract | Freeze Tier2 dry-run optimizer, epoch, seed, device/determinism and patience policy | PAPER_CROSS_EVIDENCE / UNRESOLVED |
| Tensor/DataLoader path | source-code inspection | multivariate input and component predictions | paired HF `[3] -> [10,3] -> [B,10,3] -> [B,1,3] -> [B,1,1] -> sum` | `TIER2_FALLBACK_READY` | None | Preserve strict scalar MSE shape guard | RESULT_BLIND_ENGINEERING_FALLBACK |
| Output/checkpoint paths | source-code inspection | not specified | metrics path is fixed; no dry-run checkpoint policy | `TIER2_FALLBACK_READY` | None for dry run | Version output directory before rerun; no raw-data writes | ENGINEERING_DETAIL |
| SCF-GAN | paper / Phase 7 audit | condition/output/architecture/scale contradictions remain | synthetic-only structural module; real path disabled | `NOT_REQUIRED_FOR_PHASE9A` | Future GAN | Resolve before a GAN augmentation stage | UNRESOLVED |

## Required shape and temporal route

```text
valid paired charge/discharge identity
  -> canonical HF vector [3]
  -> chronological HF history [10,3]
  -> batch [B,10,3]
  -> each branch core [B,1,3]
  -> branch Linear(3,1) [B,1,1]
  -> strict equal-shape scalar MSE [B,1,1]
  -> Trend + Seasonal + Residual [B,1,1]
  -> Capacity_hat [B,1,1]
  -> optional frozen PA001 SOH conversion only after explicit activation
```

No transpose, output-rank, or broadcast mismatch was found by static source
inspection; all three core/readout contracts were previously structural-tested.
The route remains blocked before model execution by the feature, test-loader,
scale-bridge, and run-configuration rows above.

## PLAN-CHANGE-0017_REVISED closure update

The above pre-fix statuses are historical. For the specifically authorized
NASA-only Tier 2 Phase 9A path, `t_Imin`, canonical voltage field, TEST access
boundary, PA001 SOH bridge, and run configuration are now frozen in
`MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0017-REVISED.md`. The implementation
uses `Voltage_measured` for all voltage thresholds and raw-sign
`Current_charge<=0.02 A` strictly after `t_Vmax`, with a labelled final-timestamp
fallback. `TIER2_PHASE9A_RUN_CONFIG_V1.md` freezes the execution details.

`UNRESOLVED_FEATURE_SEMANTICS=0`
`UNRESOLVED_DATA_ALIGNMENT=0`
`UNRESOLVED_EXECUTION_PARAMETERS=0`
`UNRESOLVED_MODEL_IO_CONTRACTS=0`
`UNRESOLVED_SCALE_CONTRACTS=0`
`UNRESOLVED_TRAINING_CONTRACTS=0`
`PHASE9A_CRITICAL_BLOCKERS_TOTAL=0`

This update does not revise the table's Tier 3 paper-exact evidence status.
