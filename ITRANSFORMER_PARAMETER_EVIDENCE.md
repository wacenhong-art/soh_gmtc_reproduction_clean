# iTransformer Parameter Evidence — Phase 3A

| Parameter | Value | Status | Evidence location | Boundary |
|---|---|---|---|---|
| Sequence length | 10 | PAPER_EXACT | PDF p.6 §3.3 | Input history length only. |
| Prediction length / forecast horizon | Not stated | MISSING | PDF §§3–4 | Do not infer one-step prediction. |
| Input variables / residual tensor channels | Not stated | AMBIGUOUS | PDF pp.6–7, 9; Fig.5 | See `PA-003` / Phase 3A input audit. |
| `d_model` | Not stated | MISSING | PDF §3.5 | Standard defaults are not paper parameters. |
| `n_heads` | Not stated | MISSING | PDF §3.5 | — |
| Encoder layers | Not stated | MISSING | PDF §3.5 | — |
| `d_ff` | Not stated | MISSING | PDF §3.5 | — |
| Dropout | Not stated | MISSING | PDF §3.5 | — |
| Activation | Not stated | MISSING | PDF §3.5 | — |
| Normalization | Not stated | MISSING | PDF §3.5 | — |
| Embedding / projection class | MLP | PAPER_EXACT | PDF p.7 Eqs.(21)–(23) | Dimensions/layer counts absent. |
| Dynamic graph attention | Claimed only | NOT_SPECIFIED | PDF p.7 §3.5 | No executable graph mechanism. |
| Channel mixer beyond iTransformer attention | Not stated | NOT_SPECIFIED | PDF pp.6–7, 9 | Do not add one. |
| Optimizer learning rate | 0.001 | PAPER_EXACT_UNIFIED_TRAINING_CONFIGURATION | PDF p.9 §3.7(2) | Not an iTransformer architecture parameter. |
| Batch size | 32 | PAPER_EXACT_UNIFIED_TRAINING_CONFIGURATION | PDF p.9 §3.7(2) | Not an iTransformer architecture parameter. |
| Early-stopping patience | 50 epochs on validation loss | PAPER_EXACT_UNIFIED_TRAINING_CONFIGURATION | PDF p.9 §3.7(2); p.10 | Not an iTransformer architecture parameter. |

## Standard-versus-target comparison

| Aspect | Standard iTransformer | Target paper | Reproduction consequence |
|---|---|---|---|
| Tokenization | Time points of individual series become variate tokens. | `Embedding(X_:,n)` but `n` is undefined. | Target is compatible with, but does not fully specify, inverted variable tokens. |
| Embedding | Official implementation uses a linear map from lookback length to `d_model`. | States MLP embedding. | Generic agreement; dimensions/layers missing. |
| Attention axis | Native self-attention over variate tokens. | “channel mixing” / generic transformer blocks. | Compatible description; no exact attention configuration. |
| Encoder / normalization / FFN | Encoder-only attention, LayerNorm, FFN. | Transformer blocks only. | Do not import standard defaults as paper facts. |
| Projection | Projects each variate token to forecast length. | MLP projection to `Y_:,n`. | Horizon / output shape missing. |
| Graph attention | No separate dynamic graph module in the official implementation. | Claims dynamic graph attention and iterative update without equations. | Do not silently add a GAT or label standard attention as the claimed graph mechanism. |
| Input dimensionality | Multivariate `[B,L,N]`; variables are tokens. | Residual route is explicit; extra channels are not. | Target real-data tensor remains unresolved. |

Primary comparison sources: the [official iTransformer repository](https://github.com/thuml/iTransformer) and the [iTransformer paper](https://arxiv.org/abs/2310.06625). They are used only to delimit what is standard, not to supply missing target-paper parameters.

## Configuration policy

`MINIMAL_STRUCTURAL_CONFIGURATION_POLICY=NOT_AUTHORIZED_UNTIL_TARGET_ALIGNMENT_IS_FROZEN`

No `d_model`, heads, layers, `d_ff`, dropout, or forecast length is selected in this phase. A later synthetic-only configuration, if Human-authorized, must be explicitly labeled `STRUCTURAL_ENGINEERING_FIXTURE_ONLY` and cannot become a paper parameter.
