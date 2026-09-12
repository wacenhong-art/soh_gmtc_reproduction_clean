# PLAN-CHANGE-0008 — synthetic standard TimesNet core

`PLAN_CHANGE_ID=PLAN-CHANGE-0008`

`NAME=ALLOW_SYNTHETIC_STANDARD_TIMESNET_CORE_WHILE_BRANCH_VARIANT_PARAMETERS_REMAIN_UNRESOLVED`

`LEVEL=1`

`STATUS=APPLIED__USER_AUTHORIZED__PHASE_5A_SCOPE_FROZEN`

## Authorization boundary

The Human Researcher authorized a synthetic-only, parameterized implementation of the algorithmic TimesNet core described in target-paper §3.4: FFT period detection, 1D-to-2D reshape, parameter-shared Inception processing, and adaptive multi-period merge. Where implementation detail is absent from the target paper, the standard TimesNet reference architecture is secondary structural evidence only.

`CORE_ARCHITECTURE_CHANGED=False`

`DATA_PROTOCOL_CHANGED=False`

`RESEARCH_GOAL_CHANGED=False`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`REAL_DATA_USED=False`

`TEST_LOADED=False`

## Strict exclusions

This amendment does not choose a real TimesNet input tensor, forecast target, top-k value, Trend depth, Seasonal lightweight mechanism, Inception configuration, or adaptive-merge formula for the target paper. It does not implement the target-paper Trend or Seasonal variants, fusion, SCF-GAN, real STL components, or any real-data model/training workflow.

`TREND_REAL_VARIANT_IMPLEMENTED=False`

`SEASONAL_REAL_VARIANT_IMPLEMENTED=False`

## Gates

`TIMESNET_STANDARD_CORE_STRUCTURAL_GATE` can pass only for deterministic synthetic unit/interface tests. `TIMESNET_BRANCH_VARIANT_FIDELITY_GATE` remains blocked until the missing target-paper branch semantics and parameters are resolved by primary evidence or a separately authorized fallback.
