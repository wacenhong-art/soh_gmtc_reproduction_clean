# Phase 9A Fallback Numerical Pipeline Dry-Run Specification

STATUS=NEXT_ACTION_ONLY__NOT_EXECUTED  
PROVENANCE=ADVISOR_AUTHORIZED_FALLBACK  
TEST_LOCKED=True

The next authorized step is a TRAIN+VALIDATION dry run under `ADVISOR_AUTHORIZED_NUMERICAL_PROTOCOL_V1.md`. It must begin with configuration parsing and contract checks, then use only TRAIN-fitted normalization parameters. The TEST partition remains inaccessible until a separate authorization.

Pre-registered checks: first form `VALID_CHARGE_DISCHARGE_PAIR` samples under `NASA_PAIRING_PROTOCOL_V1.md`, then apply the chronological 70/20/10 split with disjoint half-open ranges to pair order; verify the expected structural pair counts before numerical access; enforce 10-to-1 window alignment; use TRAIN-only offline STL(period=7) with no validation/test context; fit min–max only on TRAIN; preserve the explicit clipping policy; pass each frozen `[B,1,3]` branch-core output through its separate `Linear(3,1,bias=True)` scalar adapter before strict shape-equal component MSE and component-sum arithmetic (`T+S+R`); reject any broadcasting or 9-feature error/ratio builder. Validation has no component/STL labels and compares only the reconstructed final prediction with its observed next-step target. No parameter comparison, model-metric tuning, or author-exact claim is allowed.

This specification does not authorize execution in the current phase.
