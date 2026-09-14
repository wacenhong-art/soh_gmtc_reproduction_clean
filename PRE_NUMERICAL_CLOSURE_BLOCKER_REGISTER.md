# Pre-Numerical Closure Blocker Register

Only blockers capable of stopping a full Tier 2 Phase 9A TRAIN+VALIDATION run
are listed here. Paper-exact-only and future-SCF-GAN issues are deliberately
excluded from this critical register.

| ID | Description | Affected stage | Paper evidence | Code evidence | Recommended resolution | Web? | Advisor? | Author? | Plan Change? |
|---|---|---|---|---|---|---|---|---|---|
| PNC-001 | `t_Imin` is not executable for 101/443 legal TRAIN charges under the legacy sampled positive-current condition. | HF extraction | Eq.(2) defines termination-current time but not field/sign/interpolation | `prepare_split_independent_data.py:71-74`; B0006 97 and B0018 4 legacy failures | Result-blind physical current-crossing rule after `t_Vmax`; preserve Eq.(2) | Yes | Preferable | No | Yes |
| PNC-002 | NASA code uses charger voltage for all voltage endpoints although NASA separately defines terminal voltage; paper gives generic voltage only. | HF1/HF2/HF3 extraction | Eqs.(1)-(3); no field identifier | `prepare_split_independent_data.py:102-106` | Freeze canonical terminal-voltage/current field mapping and crossing interpolation for all endpoints together. | Yes | Yes | No | Yes |
| PNC-003 | The Phase 9A runner invokes `loadmat` on full raw MAT variables before logically filtering TEST pairs. | TEST lock/data access | Governance requires no TEST numerical materialization | `run_phase9a_tier2_paired_train_val_dry_run.py:75-78` | Introduce a test-safe reader contract/guard or explicitly authorize an audited technical definition of field-level access. | No | Yes | No | Yes |
| PNC-004 | Capacity-to-SOH bridge is available under PA001 but is not activated consistently with the Tier2 capacity-primary output policy. | Output scale/metrics | SOH framing and Table 1 provide incomplete denominator support | `REPRODUCTION_PROTOCOLS.md` vs `ADVISOR_AUTHORIZED_NUMERICAL_PROTOCOL_V1.md` | Freeze capacity-primary report and the exact optional PA001 SOH conversion/metric rule. | No | Yes | No | Yes |
| PNC-005 | Branch optimizer and dry-run budget are present in executable code but not frozen in a dedicated governance artifact. | Training execution | LR/batch/patience reported; temporal-branch optimizer unspecified | runner uses Adam, epochs 3, seed 42 | Freeze `optimizer`, epochs, seed, deterministic/device policy, and patience-not-used rule. | No | Yes | No | Yes |

## Noncritical or out-of-scope findings

- `CALCE_GLOBAL_CHRONOLOGICAL_IDENTITY_UNRESOLVED`: blocks future CALCE Tier 2
  work, not NASA-only Phase 9A.
- `SCFGAN_REAL_EXECUTION_UNRESOLVED`: blocks future augmentation, not the
  explicitly GAN-free Phase 9A dry run.
- Paper-exact architecture, fusion, normalization, period, and dataflow gaps
  remain Tier 3 blockers; they do not erase the disclosed Tier 2 fallback path.
- Historical PLAN-CHANGE-0001/0002/0003 records are absent from this worktree;
  the current Plan Change series begins at 0004. This is provenance debt, not a
  runtime Phase 9A blocker.

## Resolution update — PLAN-CHANGE-0017_REVISED

| ID | Resolution status | Frozen result-blind execution rule |
|---|---|---|
| PNC-001 | `RESOLVED_FOR_TIER2__PAPER_EXACT_REMAINS_UNRESOLVED` | raw-sign `Current_charge<=0.02 A` strictly after `t_Vmax`, otherwise final valid charge timestamp with recorded fallback provenance |
| PNC-002 | `RESOLVED_FOR_TIER2__PAPER_EXACT_REMAINS_UNRESOLVED` | `Voltage_measured`, first sampled value at/above 3.8/4.0/4.2 V |
| PNC-003 | `RESOLVED_FOR_TIER2_BY_HUMAN_AUTHORIZED_BOUNDARY` | raw MAT full parse allowed; after split TEST record dereference is mechanically rejected |
| PNC-004 | `RESOLVED_FOR_TIER2_BY_PA001` | `SOH=Capacity_Ah/2.0` for NASA, labelled Human fallback |
| PNC-005 | `RESOLVED_FOR_TIER2` | CPU Adam, seed 42, 3 epochs, lr 0.001, batch 32, no shuffle/drop-last/early stopping/checkpoint |

`PHASE9A_CRITICAL_BLOCKERS_TOTAL=0`
`TIER3_PAPER_EXACT_ONLY_BLOCKERS_RETAINED=True`
