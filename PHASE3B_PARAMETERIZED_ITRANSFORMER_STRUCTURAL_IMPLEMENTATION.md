# Phase 3B: Parameterized iTransformer Structural Implementation — Synthetic Only

## Scope

This artifact authorizes a generic synthetic multivariate interface only. It does not select the target paper’s real residual input, channel set, target/horizon alignment, or architecture parameters.

`REAL_INPUT_SEMANTICS_INFERENCE=False`

`REAL_TARGET_ALIGNMENT_INFERENCE=False`

`REAL_RESIDUAL_DATA_USED=False`

`REAL_WINDOW_MANIFEST_CREATED=False`

## Explicit synthetic interface

`ParameterizedITransformerStructural` takes an explicit `ITransformerStructuralConfig` with `seq_len`, `pred_len`, `n_vars`, `d_model`, `n_heads`, `e_layers`, `d_ff`, `dropout`, `activation`, strict-paper mode, and synthetic-only mode.

Its input shape is `[batch, seq_len, n_vars]`; standard iTransformer inversion produces `[batch, n_vars, seq_len]`, embeds time history into variable tokens, applies Transformer self-attention across variables, and projects to `[batch, pred_len, n_vars]`.

Fixture configuration is `seq_len=10`, `pred_len=1`, `d_model=8`, `n_heads=2`, `e_layers=1`, `d_ff=16`, `dropout=0.0`, `activation=gelu`. These values are `SYNTHETIC_STRUCTURAL_FIXTURE_ONLY`, not target-paper parameters.

`SYNTHETIC_N_VARS_ONLY=True`

`SYNTHETIC_TARGET_ALIGNMENT=ONE_STEP_SYNTHETIC_PROJECTION_ONLY`

`SYNTHETIC_TARGET_ALIGNMENT_IS_REAL_INFERENCE=False`

## Safety contract

Strict-paper mode rejects `seq_len != 10`. Invalid/missing config, incompatible `d_model`/heads, wrong rank, NaN/Inf, real-battery provenance, validation provenance, and TEST provenance fail closed.

`CUSTOM_GRAPH_MODULE_IMPLEMENTED=False`

`DYNAMIC_GRAPH_ATTENTION_DETAIL_STATUS=NOT_SPECIFIED`

The module implements only the standard iTransformer core common to target-paper Eqs.(21)–(23): embedding, Transformer encoder blocks, and projection.

## Real-data fidelity remains blocked

`ITRANSFORMER_REAL_DATA_FIDELITY_GATE=BLOCKED_BY_STL_PERIOD_AND_INPUT_SEMANTICS_AND_TARGET_ALIGNMENT_AND_MISSING_ARCH_PARAMETERS`

`NUMERICAL_REPRODUCTION_BLOCKERS=STL_REAL_BATTERY_PERIOD_MISSING;ITRANSFORMER_INPUT_SEMANTICS_MISSING;ITRANSFORMER_TARGET_ALIGNMENT_MISSING;ITRANSFORMER_ARCH_PARAMETERS_MISSING`

## Executed synthetic audit

`output/itransformer/itransformer_structural_audit.json` is a passing synthetic-only audit.

`SYNTHETIC_N_VARS_TESTED=1,3`

`FORWARD_PASS=PASS`

`OUTPUT_SHAPE=PASS`

`BACKWARD_PASS=PASS`

`FINITE_LOSS=PASS`

`NONZERO_GRADIENT=PASS`

`DETERMINISM=PASS`

`TINY_OVERFIT_SANITY=PASS__64_STEPS__2.775003433227539_TO_0.0003344158467371017`

`FAIL_CLOSED_TESTS=PASS`

`REAL_DATA_GUARD=PASS`
