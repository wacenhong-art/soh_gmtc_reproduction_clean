# Phase 4A synthetic G-MTC framework scaffold implementation

## Implemented scope

`scripts/gmtc_framework_scaffold.py` is a synthetic-only contract module. It provides component inputs, explicit unimplemented TimesNet ports, an adapter to the Phase 3B synthetic iTransformer core, deterministic routing-only mocks, branch prediction payloads, a future-fusion schema, and fail-closed provenance/alignment validation.

`scripts/run_gmtc_framework_scaffold_tests.py` uses only a deterministic synthetic STL-component fixture. It executes no learning, no fusion model, no model metric, no real battery input, and no validation/test value.

## Pre-registered test conditions

`SYNTHETIC_COMPONENTS=TREND,SEASONAL,RESIDUAL`

`SYNTHETIC_SEQ_LEN=10`

`SYNTHETIC_PRED_LEN=1`

`MOCK_TREND_BRANCH=DETERMINISTIC_ROUTING_ONLY`

`MOCK_SEASONAL_BRANCH=DETERMINISTIC_ROUTING_ONLY`

`RESIDUAL_BRANCH=PHASE3B_SYNTHETIC_ONLY_PARAMETERIZED_ITRANSFORMER`

## Audit result

The machine-readable result is `output/framework/gmtc_scaffold_structural_audit.json`.

Its passing status means only that the synthetic scaffold routes exactly three aligned component predictions and rejects forbidden or incompatible payloads. It does not establish TimesNet, Bi-LSTM fusion, SCF-GAN, a real G-MTC model, framework reproduction, or numerical reproduction.
