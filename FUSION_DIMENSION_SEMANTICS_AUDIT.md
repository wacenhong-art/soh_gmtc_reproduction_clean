# Fusion Dimension and Feature-Semantics Audit — Phase 6A

## 9 / 9d / 3d consistency

| Candidate interpretation | Supporting evidence | Limitation | Status |
|---|---|---|---|
| A. `d=1`, so `9d=9` and first `3d=3` | Algebraically compatible with Table 4 Bi-LSTM input dimension 9. | Paper never defines `d=1`; convenience is not evidence. | Plausible only. |
| B. Each nominal feature is a d-dimensional embedding | Table 4 notation uses `9d` and `3d`. | Same table reports Bi-LSTM input 9 and no embedding map. | Plausible but incomplete. |
| C. Table 4 abbreviates an unreported reshape/projection | Could reconcile notation. | No equation or figure supplies operation. | Unresolved. |

`FUSION_D_STATUS=UNRESOLVED`

`TABLE4_DIMENSION_CONSISTENCY_STATUS=AMBIGUOUS__9D_VS_9_VS_3D`

## Feature semantics

| Group | Paper statement | Executable definition | Status |
|---|---|---|---|
| Prediction | One group for Trend/Seasonal/Residual | Eq.(28) calls `X_comp,i` component-i prediction. | Partially grounded; mapping to first `3d` slots remains ambiguous. |
| Error | One group for each component | Sign, absolute/relative/rolling construction and inference availability absent. | MISSING. |
| Ratio | One group for each component | Numerator, denominator, units, and inference availability absent. | MISSING. |
| First `3d` slots | weighted by component weights at last timestep | No explicit ordering ties slots to prediction fields. | AMBIGUOUS. |

`COMPONENT_ERROR_DEFINITION_STATUS=MISSING`

`COMPONENT_RATIO_DEFINITION_STATUS=MISSING`

`WEIGHTED_FEATURE_SELECTION_STATUS=AMBIGUOUS`

## Structural-only boundary

The Phase 6A fixture reserves its first three scalar slots as **synthetic prediction-like placeholders**, followed by three error-like and three ratio-like placeholders. This is a test interface only.

`REAL_FEATURE_SELECTION_INFERENCE=False`

`REAL_FUSION_FEATURE_BUILDER_IMPLEMENTED=False`

No error or ratio is constructed from actual branch targets under this contract, so no target-leakage pathway is introduced.
