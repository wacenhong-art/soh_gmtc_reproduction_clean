# PLAN-CHANGE-0020 — Author-Clarified Small-Sample Protocol

**Authority:** direct Human instruction relaying the paper author's latest
protocol clarification. This is an author-clarified Tier-2 numerical protocol;
it does not retroactively represent earlier fallback work as paper exact.

## Frozen NASA target protocol

```text
AUTHOR_CLARIFIED_PROTOCOL=True
NASA_SOH=CAPACITY_AH/2.0
INTEGER_BOUNDARY_RULE=FLOOR_HALF_OPEN
PAPER_EXPLICIT_INTEGER_ROUNDING=NOT_SPECIFIED

B0007: REAL_TRAIN=[0,33), UNUSED_REAL=[33,100), VAL=[100,134), TEST=[134,168)
B0018: REAL_TRAIN=[0,26), UNUSED_REAL=[26,79), VAL=[79,105), TEST=[105,132)
```

The `UNUSED_REAL` interval is not a training sample source: it is excluded
from scaler/STL fitting, branch and fusion training windows, optimizer updates,
training loss, SCF-GAN training, and training-sample checkpoint selection. It
may appear only as already-observed past inference context required by the
frozen sequential validation path. Expanding-prefix validation component labels
retain the existing rule: each label sees capacity only through its own
endpoint and is used only for real-VAL early stopping/loss.

## SCF-GAN boundary

The existing PC0019 epoch-200 source-only checkpoint and its five persisted
independent trajectories per target may be reused only after provenance
preflight. They remain `SCFGAN_STATUS=PROVISIONAL_TIER2_FALLBACK` and
`PAPER_EXACT_SCFGAN=False`: the author confirmed battery type, cycle index,
and charging-voltage-window condition components, but the voltage-window
encoding remains unresolved. This change neither retrains SCF-GAN nor
regenerates synthetic data.

## Supersession and TEST boundary

Phase9E remains immutable history but is superseded as the final numerical
protocol by this author clarification. Phase9F aborted before TEST numerical
unlock; it is not a current evaluation path. Phase9G keeps TEST numerically
sealed and may not use TEST values for any operation.

```text
PLAN_CHANGE_0020_APPLIED=True
PHASE9E_STATUS=SUPERSEDED_FOR_FINAL_PROTOCOL_BY_AUTHOR_CLARIFICATION
PHASE9F_STATUS=ABORTED_BEFORE_TEST_UNLOCK
TEST_CONTAMINATION=False
CURRENT_REPRODUCTION_TIER=TIER2_AUTHOR_CLARIFIED_PROTOCOL_WITH_PROVISIONAL_SCFGAN_FALLBACK
RESULT_DRIVEN=False
HUMAN_AUTHORIZED=True
```
