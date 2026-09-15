# Phase9E NASA 60/20/20 Pretest Specification

**Status:** Approved execution specification; Tier 2 paper-closest numerical
pretest, not paper-exact full-protocol reproduction.

## Required governance flags

```text
SPLIT_RATIO_PAPER_ALIGNED=True
INTEGER_BOUNDARY_RULE=FLOOR_HALF_OPEN
INTEGER_BOUNDARY_RULE_SOURCE=TIER2_IMPLEMENTATION_CONVENTION
PAPER_EXPLICIT_INTEGER_ROUNDING=NOT_SPECIFIED
PAPER_ALIGNED_SPLIT=True
PAPER_EXACT_FULL_PROTOCOL=False
REPRODUCTION_TIER=TIER2_PAPER_CLOSEST_NUMERICAL
```

## Frozen execution contract

- Pair valid NASA charge-discharge supervised cycles before splitting.
- Split B0007 (168 pairs) as `100/34/34` and B0018 (132 pairs) as `79/26/27`
  using `[0,floor(0.60*N))`, `[floor(0.60*N),floor(0.80*N))`, `[floor(0.80*N),N)`.
- Rebuild all scaler, STL, window, component-label, fusion-feature,
  early-stopping, and checkpoint artifacts from only the new real TRAIN split.
- Preserve the frozen HF definitions, pairing rule, sequence contract,
  period=7, branches, scalar readouts, Phase9B fusion, losses, optimizers,
  sizes, and training budgets. No hyperparameter selection is permitted.
- PC0019 epoch-200 synthetic trajectories may be reused only after their
  source-only, target-condition-only, TEST-free provenance passes; they enter
  TRAIN only as separate trajectories.

## Sealed TEST boundary and gates

No TEST numerical field may enter HF extraction, scaling, STL, windowing,
DataLoader, model execution, checkpoint selection, prediction, or metric.
Any failed provenance, TEST-lock, leakage, shape, reconstruction, or finite
value gate emits `RESEARCH_DIRECTION_REJECTED` and stops. Poor validation
metrics alone never change this frozen protocol.
