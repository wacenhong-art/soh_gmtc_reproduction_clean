# Phase 9A Fallback Numerical Pipeline Dry-Run Specification

STATUS=NEXT_ACTION_ONLY__NOT_EXECUTED  
PROVENANCE=ADVISOR_AUTHORIZED_FALLBACK  
TEST_LOCKED=True

The next authorized step is a TRAIN+VALIDATION dry run under `ADVISOR_AUTHORIZED_NUMERICAL_PROTOCOL_V1.md`. It must begin with configuration parsing and contract checks, then use only TRAIN-fitted normalization parameters. The TEST partition remains inaccessible until a separate authorization.

Pre-registered checks: chronological 70/20/10 split with disjoint half-open ranges, 10-to-1 window alignment, TRAIN-only offline STL(period=7) with no validation/test context, min–max fit scope equal to TRAIN_ONLY, explicit clipping policy, component-sum arithmetic (`T+S+R`), MSE finiteness, and absence of any 9-feature error/ratio builder. Validation has no component/STL labels and compares only the reconstructed final prediction with its observed next-step target. No parameter comparison, model-metric tuning, or author-exact claim is allowed.

This specification does not authorize execution in the current phase.
