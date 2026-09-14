# Phase 9A Real-Data Numerical Pipeline Dry-Run Report

STATUS=BLOCKED_BEFORE_REAL_DATA_ACCESS  
REAL_DATA_USED=False  
TRAIN_DATA_USED=False  
VALIDATION_DATA_USED=False  
TEST_LOADED=False  
TEST_VALUE_ACCESS_COUNT=0  
GRID_SEARCH_RUN=False  
SCFGAN_TRAINED=False  
SYNTHETIC_HF_USED=False

## Preflight result

PLAN_READ_BEFORE_RUN=True  
PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS__TIER2_PROTOCOL_CONSISTENT

PLAN-CHANGE-0013 establishes chronological 70/20/10 membership, component-sum fusion, 10-to-1 alignment, STL period 7, TRAIN-only min–max 0–1 normalization, hidden width 64, and MSE. The Phase 1 E1 role matrix uniquely identifies the four target batteries; no roster was changed.

## Fail-closed blockers

`PHASE9A_BLOCKED_BY_STL_TEMPORAL_LEAKAGE=True`  
`STL_TEMPORAL_LEAKAGE_AUDIT=FAIL__INTRA_PARTITION_FUTURE_CONTEXT_POSSIBLE`

The existing authorized STL mode is explicitly `PARTITION_ISOLATED_OFFLINE_STL` and `NON_CAUSAL_OFFLINE_DECOMPOSITION`. It prevents TRAIN/VALIDATION/TEST cross-partition contamination, but standard offline STL smoothing can use later observations within the declared partition. This violates Phase 9A's stricter requirement that target construction use no future observations. No alternative STL method was substituted.

`UNRESOLVED_EXECUTION_PARAMETER=TIMESNET_TOP_K,D_MODEL,D_FF,E_LAYERS,NUM_KERNELS,DROPOUT`  
`UNRESOLVED_EXECUTION_PARAMETER=ITRANSFORMER_D_MODEL,N_HEADS,E_LAYERS,D_FF,DROPOUT,ACTIVATION,NORMALIZATION`

The advisor protocol freezes only `PRIMARY_HIDDEN_WIDTH=64`; it does not freeze the remaining architecture parameters required by the existing real-data-incompatible structural modules. The modules are intentionally synthetic-only. No parameter was inferred, defaulted, searched, or copied from another project.

## Consequence

No split materialization, HF/capacity read, scaler fit, STL decomposition, window creation, branch forward/backward pass, checkpoint, metric, or prediction was performed. Therefore the requested split, normalization, temporal-leakage, and dry-run metric artifacts were not created; creating them would falsely imply real-data access or executable target construction.

NEXT_ACTION=HUMAN_OR_ADVISOR_AUTHORIZATION_FOR_CAUSAL_STL_TARGET_POLICY_AND_COMPLETE_BRANCH_EXECUTION_CONFIG

This is a protocol gate, not a scientific failure and not evidence about model performance.

## Retry preflight — 2026-09-14

PLAN-CHANGE-0014 resolved the offline-STL and branch-config blockers. Before any numeric data access, the retry found a distinct label-interface blocker: NASA HF source records are charge operations, while their Phase 1 raw-capacity field is `None`. No authorized pairing from charge HF identity to an observed discharge capacity label exists. The retry is therefore blocked before accessing TRAIN/VALIDATION values; see `PHASE9A_RETRY_PREFLIGHT_BLOCKER_REPORT.md`.

## Real TRAIN/VAL dry-run attempt after all interface freezes — 2026-09-14

`STATUS=BLOCKED__HF2_EXTRACTION_CONTRACT_UNSATISFIED`<br>
`REAL_NUMERICAL_DATA_USED=True__TRAIN_ONLY_FIELD_LEVEL_ACCESS`<br>
`TRAIN_DATA_USED=True`<br>
`VALIDATION_DATA_USED=False__NOT_REACHED`<br>
`TEST_LOADED=False`<br>
`TEST_NUMERICAL_VALUE_ACCESS_COUNT=0`<br>
`MODEL_TRAINED=False`<br>
`GRID_SEARCH_RUN=False`<br>
`SCFGAN_TRAINED=False`<br>
`SYNTHETIC_HF_USED=False`

The frozen discharge-centered pairing and chronological split were reconstructed from metadata for all four NASA batteries before any numerical feature field was accessed: B0005/B0006/B0007 each yielded 168 valid pairs and the 117/34/17 split; B0018 yielded 132 valid pairs and the 92/26/14 split. This satisfied the pairing/split gate.

The first authorized B0007 TRAIN charge record was then evaluated solely with the existing Phase 1 HF definition. Its HF2 rule requires the first post-4.2 V sample with `0 <= Current_charge <= 0.02`. No such sample was present. The runner stopped before accessing that pair's discharge Capacity, before accessing any validation numerical field, and before any scaler, STL, window, model, loss, metric, or checkpoint action.

No alternative current sign convention, absolute-current threshold, interpolation, deletion, or replacement HF definition was applied. Those changes would alter the frozen HF semantics and require a separate, result-blind protocol decision. The earlier `CHARGE_CAPACITY_NONE` issue remains resolved by PLAN-CHANGE-0015 pairing; this is a new, independent HF2-executability blocker.

`NEXT_ACTION=RESULT_BLIND_HF2_SIGN_AND_ENDPOINT_SEMANTICS_AUDIT_OR_EXPLICIT_HUMAN_PROTOCOL_AUTHORIZATION`

## Correction after HF2 semantics audit — 2026-09-14

The preceding attempt record conservatively identified B0007 as the location of
the exception because the original fail-closed error lacked battery/pair context.
The subsequent TRAIN-only audit corrected that attribution without changing any
feature rule: B0007 has zero failures under the existing rule. The actual
failures are B0006 (97 TRAIN charges) and B0018 (4 TRAIN charges); the first is
B0018 pair 14 / raw charge record 40. See `NASA_HF2_SIGN_AND_ENDPOINT_SEMANTICS_AUDIT.md`.

The audit found an identifiable `Voltage_measured` 4.2 V CC-to-CV endpoint in
all 443 legal TRAIN charges. It recommends a separate result-blind Plan Change
before any HF2 implementation changes or Phase 9A retry.

## Executed fixed dry-run after PLAN-CHANGE-0017_REVISED — 2026-09-14

`STATUS=PASS__TRAIN_VAL_DRY_RUN_COMPLETE__NO_TEST_NUMERICAL_ACCESS`
`CURRENT_REPRODUCTION_TIER=TIER2_FALLBACK_NUMERICAL_REPRODUCTION`
`PLAN_READ_BEFORE_RUN=True`
`PLAN_HEALTH_AUDIT_BEFORE_RUN=PASS__FROZEN_TIER2_PROTOCOL_CONSISTENT`

This is a single fixed CPU dry-run, not a model-selection exercise or a
paper-exact result. It used only the Phase 1 E1 NASA target roster B0007 and
B0018. Frozen execution settings were seed 42, three epochs, Adam with learning
rate 0.001, batch size 32, chronological no-shuffle batches, and MSE. No
checkpoint, early stopping, parameter comparison, SCF-GAN, synthetic HF, or
grid search was used.

### Contract and access gates

| Gate | Result |
|---|---|
| Pair-first chronological split | PASS: B0007 `117/34/17`; B0018 `92/26/14` |
| TEST lock | PASS: loaded/access attempts/materializations/transforms/predictions = `0/0/0/0/0` |
| HF extraction | PASS: B0007 `117+34`; B0018 `92+26`; failures `0` |
| TRAIN-only MinMax | PASS; validation only transformed and clipped under the frozen rule |
| TRAIN-only offline STL(7) | PASS; no VAL/TEST input; additive reconstruction within machine precision |
| 10-to-1 alignment | PASS: historical HF windows only; TRAIN targets stay TRAIN; VAL uses past-HF rolling history only |
| Scalar readout / loss shape | PASS: exact `[B,1,1]` prediction-to-target equality; no broadcasting |
| Component sum | PASS: `Trend_hat + Seasonal_hat + Residual_hat` in capacity domain |
| Finite values / gradients | PASS: zero NaN/Inf counts; all six core/readout gradient paths finite and nonzero |
| Fixed-seed repeat | PASS: two full payloads byte-for-byte equivalent |

### Per-epoch losses

Validation has no component/STL labels by frozen policy. Therefore each epoch's
validation loss is the permitted fused capacity-domain `MSE(T_hat+S_hat+R_hat,
y_val)`, recorded only as telemetry and never used for selection.

| Battery | Trend TRAIN MSE (epochs 1/2/3) | Seasonal TRAIN MSE | Residual TRAIN MSE | Fused VAL MSE (epochs 1/2/3) |
|---|---|---|---|---|
| B0007 | 3.7066589494 / 1.2255793737 / 0.2230717139 | 0.0236483291 / 0.0181627694 / 0.0144738411 | 0.0592044276 / 0.0070171639 / 0.0096746829 | 2.0238604546 / 0.0022066499 / 0.6186665893 |
| B0018 | 1.7922203977 / 0.7351082465 / 0.1020117955 | 0.2216096971 / 0.0185198228 / 0.0527355705 | 0.0290509190 / 0.0208108286 / 0.0044814322 | 0.4905633330 / 0.0975133926 / 0.3927336931 |

### Final validation diagnostics

The primary prediction domain is capacity. The SOH values below are the
separately authorized PA001 fallback conversion `SOH=Capacity_Ah/2.0`; neither
domain was used for parameter selection.

| Battery | Capacity MSE / RMSE / MAE / R2 | SOH MSE / RMSE / MAE / R2 |
|---|---|---|
| B0007 | 0.6186665893 / 0.7865536150 / 0.7862246633 / -754.0102539062 | 0.1546666476 / 0.3932768078 / 0.3931123088 / -754.0102208114 |
| B0018 | 0.3927336931 / 0.6266846840 / 0.6256388426 / -594.9241943359 | 0.0981834268 / 0.3133423476 / 0.3128194259 / -594.9242511421 |

Negative validation R2 values do not fail this Phase 9A execution gate and did
not trigger any protocol change. `PHASE9A_PIPELINE_GATE=PASS` and
`SCIENTIFIC_FAILURE=False` mean only that the approved pipeline executed
leakage-safely and reproducibly; they do not claim numerical quality or
paper-exact reproduction.

`NEXT_ACTION=HUMAN_REVIEW_OF_PHASE9A_DIAGNOSTICS__NO_AUTOMATIC_PHASE9B_OR_TUNING`
