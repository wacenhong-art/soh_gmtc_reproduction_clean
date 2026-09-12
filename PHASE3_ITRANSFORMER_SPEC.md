# Phase 3A iTransformer Specification and Dataflow Contract

## Frozen paper facts

`ITRANSFORMER_PAPER_ROLE=Residual component → iTransformer`

`SEQUENCE_LENGTH=10`

`SEQUENCE_LENGTH_PROVENANCE=PAPER_EXACT`

`SEQUENCE_LENGTH_IS_STL_PERIOD=False`

`SEQUENCE_LENGTH_IS_FORECAST_HORIZON=False`

`EMBEDDING_AND_PROJECTION=MLP__PAPER_EXACT`

## Unresolved execution fields

`ITRANSFORMER_INPUT_SEMANTICS_STATUS=UNRESOLVED_BLOCKING_FOR_REAL_DATA`

`FORECAST_TARGET_ALIGNMENT_STATUS=UNRESOLVED_BLOCKING_FOR_REAL_DATA`

`RESIDUAL_TARGET_ALIGNMENT=AMBIGUOUS_BLOCKING__PA-007`

`D_MODEL_STATUS=MISSING`

`N_HEADS_STATUS=MISSING`

`ENCODER_LAYERS_STATUS=MISSING`

`D_FF_STATUS=MISSING`

`DROPOUT_STATUS=MISSING`

`DYNAMIC_GRAPH_ATTENTION_DETAIL_STATUS=NOT_SPECIFIED`

## Dual gates

`ITRANSFORMER_STRUCTURAL_IMPLEMENTATION_GATE=PASS__GENERIC_SYNTHETIC_INTERFACE_ONLY`

`ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_MISSING_ARCH_PARAMETERS`

No real-data iTransformer branch, real residual input, real model training, metric, validation-value access, test-value access, TimesNet, fusion, or search is authorized by this artifact. The narrowly scoped synthetic-only structural exception is recorded below.

## Phase 3B authorized structural exception

Human authorized a pre-registered generic synthetic multivariate interface and a synthetic-only one-step projection target. This permits structural software testing of the standard iTransformer core only; it does not resolve any real target-paper input, output, or architecture field.

`PARAMETERIZED_ITRANSFORMER_IMPLEMENTED=True`

`FRAMEWORK_ITRANSFORMER_STATUS=STRUCTURALLY_IMPLEMENTED_SYNTHETIC_ONLY`

`REAL_RESIDUAL_DATA_USED=False`

`REAL_FORECAST_ALIGNMENT=UNRESOLVED`

`REAL_RESIDUAL_TARGET=UNRESOLVED`

`REAL_COMPONENT_SUPERVISION=UNRESOLVED`

## Numerical-reproduction blockers

`NUMERICAL_REPRODUCTION_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING;ITRANSFORMER_INPUT_SEMANTICS_MISSING;ITRANSFORMER_TARGET_ALIGNMENT_MISSING;ITRANSFORMER_ARCH_PARAMETERS_MISSING`

## Next gate

Before any synthetic-only iTransformer module, Human or primary evidence must freeze an explicit target-alignment contract. Any configuration then used must be a `STRUCTURAL_ENGINEERING_FIXTURE_ONLY`, not a target-paper architecture parameter.
