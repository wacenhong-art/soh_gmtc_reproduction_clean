# Unified Pre-Numerical Fix Package

`STATUS=APPLIED_BY_PLAN-CHANGE-0017_REVISED__HUMAN_AUTHORIZED`
`RECOMMENDED_CHANGE_ID=PLAN-CHANGE-0017_REVISED__UNIFIED_PRE_NUMERICAL_CLOSURE`  
`RESULT_BLIND=True`  
`PERFORMANCE_BLIND=True`  
`VALIDATION_OR_TEST_METRIC_USED=False`

## One unified change, not serial patching

1. Retain Eq.(1)–Eq.(3) as the controlling HF definitions despite their prose
   labels: HF1 remains CC duration, HF2 remains CV duration, and HF3 remains
   3.8-to-4.0 V rise time.
2. Freeze one canonical NASA endpoint map: terminal-voltage field for
   `t_Vmax/t_start/t_end`; charger-current field for `t_Imin`; deterministic
   first-crossing interpolation rules for 4.2 V and the termination-current
   crossing. Do not call HF2 a CC duration.
3. Replace duplicated HF extraction implementations with one explicit,
   provenance-aware extractor only after the endpoint map is authorized;
   require 443/443 legal TRAIN endpoint coverage before Phase 9A numerical
   access.
4. Establish an executable TEST-lock reader contract before raw MAT loading:
   either record-level lazy access that never deserializes TEST numeric fields,
   or an explicitly approved, auditable field-access definition. The current
   eager `loadmat` behavior cannot be silently treated as satisfying a literal
   no-materialization policy.
5. Create `TIER2_PHASE9A_RUN_CONFIG_V1`: Adam-or-other explicitly authorized
   optimizer, LR 0.001, batch 32, exactly 3 dry-run epochs, seed 42,
   CPU/device and deterministic policy, and no early stopping/patience use in
   the dry run.
6. Freeze reporting domain: capacity is primary; if SOH metrics are emitted,
   they use only frozen PA001 NASA denominator 2.0 Ah and are labelled Human
   fallback, never paper exact.

## Acceptance bundle before a retry

- endpoint unit tests cover exact, bracketed, absent, nonmonotonic, and
  zero-denominator crossings;
- TRAIN-only endpoint coverage and failure-class manifest has no unresolved
  record;
- TEST reader guard proves zero numeric TEST field access under instrumentation;
- static `[B,10,3] -> [B,1,1] -> [B,1,1]` route and strict MSE shape tests pass;
- no model training, validation tuning, SCF-GAN, or TEST metric is used to
  select any rule.

The proposal was applied only after the explicit Human authorization recorded
in `MASTER_PLAN_CHANGE_REQUEST_PLAN-CHANGE-0017-REVISED.md`. The applied
revision resolves its prior underspecification through exact frozen rules:
terminal `Voltage_measured`; raw-sign `Current_charge<=0.02 A` strictly after
`t_Vmax`, otherwise final valid timestamp; raw full MAT parse but sealed
post-split TEST numerical access; active PA001 NASA 2.0-Ah bridge; and the
versioned CPU Adam run configuration. It remains a Tier 2 fallback, never a
paper-exact implementation claim.
