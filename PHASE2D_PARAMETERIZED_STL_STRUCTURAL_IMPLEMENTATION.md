# Phase 2D: Parameterized STL Structural Implementation Without a Real-Battery Period

## Scope

This is a software-structure check only. `scripts/stl_decomposition.py` accepts a caller-supplied period, a single partition identifier, and a series identifier. It cannot discover a period, infer one from a downstream 10-step window, concatenate partitions, process TEST, read raw battery data, train a model, or calculate prediction metrics.

STL_INPUT=CAPACITY_SEQUENCE

STL_INPUT_PROVENANCE=PAPER_CROSS_EVIDENCE

STL_PERIOD_IDENTIFIABLE=False

NASA_STL_PERIOD=UNRESOLVED

CALCE_STL_PERIOD=UNRESOLVED

REAL_BATTERY_STL_EXECUTED=False

## Synthetic-only validation

`scripts/run_stl_structural_unit_tests.py` uses `SYNTHETIC_STL_UNIT_TEST_FIXTURE` with fixture-only periods 4, 7, and 12. It is not NASA or CALCE data and cannot infer, select, or justify a real-battery period. The frozen reconstruction tolerance is `atol=1e-12`, `rtol=1e-12` before test execution.

SYNTHETIC_FIXTURE_USED_FOR_REAL_PERIOD_INFERENCE=False

P12_MONTHLY_EXAMPLE_USED_AS_REAL_PARAMETER=False

## Dual-gate status

STL_STRUCTURAL_IMPLEMENTATION_GATE=see `output/stl/stl_structural_implementation_audit.json`

STL_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_MISSING_PAPER_PERIOD

PAPER_ALIGNED_REAL_STL_OUTPUTS_CREATED=False

## Component interface contract

| Component | Future consumer | Current status |
| --- | --- | --- |
| trend | TimesNet branch | interface only |
| seasonal | TimesNet branch | interface only |
| residual | iTransformer branch | interface only |

DOWNSTREAM_SEQUENCE_LENGTH=10

DOWNSTREAM_SEQUENCE_LENGTH_IS_STL_PERIOD=False

DOWNSTREAM_SEQUENCE_LENGTH_IS_STL_WINDOW=False

NUMERICAL_REPRODUCTION_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING

AUTHOR_OVERRIDE_PENDING=True

## Executed synthetic audit

`output/stl/stl_structural_implementation_audit.json` records a passing synthetic-only audit.

STL_STRUCTURAL_IMPLEMENTATION_GATE=PASS

FRAMEWORK_STL_IMPLEMENTATION_STATUS=STRUCTURALLY_REPRODUCED

MAX_ABS_RECONSTRUCTION_ERROR=2.220446049250313e-16

MEAN_ABS_RECONSTRUCTION_ERROR=2.7755575615628915e-18

DETERMINISM_TEST=PASS

FAIL_CLOSED_TESTS=PASS
