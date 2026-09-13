# Phase 5A parameterized standard TimesNet structural implementation — synthetic only

## Scope

`PARAMETERIZED_STANDARD_TIMESNET_IMPLEMENTED=True`

`SYNTHETIC_FIXTURE_CREATED=True`

`REAL_STL_COMPONENTS_USED=False`

`TEST_LOADED=False`

The module implements only a generic standard core: explicit FFT top-k detection, deterministic period-to-2D padding/reshape/crop, parameter-shared Inception-style blocks, amplitude-weighted adaptive merging, explicit projection, and strict provenance/configuration validation.

Every synthetic configuration value (`top_k`, `d_model`, `d_ff`, `e_layers`, `num_kernels`, `dropout`, synthetic target alignment) is `SYNTHETIC_STRUCTURAL_FIXTURE_ONLY`.

`TARGET_PAPER_TOPK_INFERENCE=False`

`TREND_REAL_VARIANT_IMPLEMENTED=False`

`SEASONAL_REAL_VARIANT_IMPLEMENTED=False`

## Fixture policy

The deterministic fixture tests single-period and multi-period signals with `n_vars=1,3`, explicit synthetic `top_k=1,2`, and a separate non-divisible reshape/padding test. A synthetic one-step projection target exists only for forward/backward/overfit wiring.

`REAL_TARGET_ALIGNMENT_INFERENCE=False`

`SYNTHETIC_TARGET_ALIGNMENT=ONE_STEP_SYNTHETIC_PROJECTION_ONLY`

No detected synthetic period may be used as an STL period or real-battery TimesNet parameter.

## Gate boundary

`TIMESNET_STANDARD_CORE_STRUCTURAL_GATE=PASS` means generic software contract tests passed.

`TIMESNET_BRANCH_VARIANT_FIDELITY_GATE=BLOCKED_BY_INPUT_TARGET_AND_VARIANT_PARAMETERS`

The standard core does not reproduce the target paper’s unspecified deep Trend or lightweight Seasonal variants.

## Phase 7B real-data interface reconciliation — PLAN-CHANGE-0010

`PHASE5_REAL_INTERFACE_ROLE_REVISED=True`

`TIMESNET_STRUCTURAL_CORE_INVALIDATED=False`

`TREND_REAL_INPUT_FALLBACK=HF1_HF2_HF3_MULTIVARIATE_WINDOW`

`TREND_REAL_SUPERVISION_FALLBACK=STL_TREND_COMPONENT`

`SEASONAL_REAL_INPUT_FALLBACK=HF1_HF2_HF3_MULTIVARIATE_WINDOW`

`SEASONAL_REAL_SUPERVISION_FALLBACK=STL_SEASONAL_COMPONENT`

The former synthetic component fixture is retained for structural testing only. This fallback does not determine real TimesNet input axes, target horizon, deep configuration, lightweight mechanism, top-k, kernels, merge, or any paper parameter.

`TIMESNET_TARGET_ALIGNMENT_MISSING=True`
