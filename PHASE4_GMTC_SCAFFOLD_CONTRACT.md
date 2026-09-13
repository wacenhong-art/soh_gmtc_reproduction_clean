# Phase 4 G-MTC synthetic framework scaffold contract

`SCOPE=SYNTHETIC_ONLY_STRUCTURAL_ROUTING`

`REAL_BATTERY_DATA_USED=False`

`REAL_STL_COMPONENTS_USED=False`

`TEST_LOADED=False`

## Topology contract

```text
Synthetic STL component fixture
    ├── TrendComponentInput ──> TrendModelPort (Phase 5; NOT_AVAILABLE_YET)
    ├── SeasonalComponentInput -> SeasonalModelPort (Phase 5; NOT_AVAILABLE_YET)
    └── ResidualComponentInput -> StructuralResidualBranchAdapter
                                      └── synthetic-only parameterized iTransformer
all three BranchPrediction payloads
    └── FutureFusionPayload schema only (no fusion execution)
```

`SEQ_LEN=10`

`SEQ_LEN_PROVENANCE=PAPER_EXACT`

`REAL_PRED_LEN_INFERENCE=False`

The test fixture uses `SYNTHETIC_PRED_LEN=1` solely to validate tensor routing. It does not infer the target paper’s real forecast horizon or offset.

## Component and branch ports

Every component input carries: `component_type`, `input_tensor`, `input_shape`, `seq_len`, `batch_ids`, `time_alignment`, `forecast_horizon`, `sample_provenance`, `partition_provenance`, `fidelity_status`, and `synthetic_only`.

Every branch prediction carries the corresponding component identity plus `branch_name`, `output_prediction`, `output_shape`, `model_fidelity_status`, and `synthetic_only`.

`TrendModelPort` and `SeasonalModelPort` are intentionally unimplemented ports:

`IMPLEMENTATION=NOT_AVAILABLE_YET`

`REQUIRED_PHASE=Phase_5`

Their test-time deterministic mocks are explicitly named `MOCK_TREND_BRANCH` and `MOCK_SEASONAL_BRANCH`.

`MOCK_IS_NOT_MODEL_REPRODUCTION=True`

No MLP, LSTM, identity branch, or trained substitute is present.

## Future fusion schema only

The future payload has exactly three component predictions: `TREND`, `SEASONAL`, `RESIDUAL`. Its declared slots are:

```text
predictions: {TREND, SEASONAL, RESIDUAL}
errors:      {TREND, SEASONAL, RESIDUAL}  # schema slot only
ratios:      {TREND, SEASONAL, RESIDUAL}  # schema slot only
```

`FUSION_INPUT_SCHEMA_CREATED=True`

`COMPONENT_RATIO_DEFINITION_STATUS=UNRESOLVED`

`FUSION_FEATURE_D_STATUS=UNRESOLVED`

`FUSION_EXECUTED=False`

No error or ratio value is calculated, and no Bi-LSTM, attention, Softmax, learned weights, final head, or loss path is implemented in this phase.

## Alignment contract

All three branch predictions must agree exactly on `batch_ids`, `time_alignment`, `forecast_horizon`, `sample_provenance`, and `partition_provenance`. Mismatch, missing component, duplicate component, wrong sequence length, real-data provenance, or TEST provenance fails closed. The scaffold never pads, truncates, concatenates, or coerces a mismatched prediction.

`ALIGNMENT_CONTRACT_CREATED=True`

## Real-data fidelity boundary

`FRAMEWORK_SCAFFOLD_STRUCTURAL_GATE` concerns only synthetic topology, ports, routing, alignment, and provenance guards.

`FRAMEWORK_REAL_DATA_FIDELITY_GATE` remains blocked by the inherited unresolved paper/data conditions, including STL period, TimesNet implementation/specification, iTransformer real-data semantics/target alignment/parameters, fusion semantic gaps, training orchestration, and shared adversarial-training detail.

## Phase 7B active real-data route — PLAN-CHANGE-0010

`PHASE4_REAL_INTERFACE_ROLE_REVISED=True`

`SYNTHETIC_SCAFFOLD_INVALIDATED=False`

The synthetic component-to-port fixture remains test-only. The active authorized real-data route is deliberately split:

```text
HF1/HF2/HF3 temporal window → Trend / Seasonal / Residual branch inputs
observed TRAIN capacity → STL → Trend / Seasonal / Residual branch targets
three branch predictions → fusion inputs
```

No arrow in this real-data contract means “component input to branch.” Component predictions, not raw STL components, remain the future fusion inputs. All alignment, provenance, synthetic-only, and TEST guards remain in force.
