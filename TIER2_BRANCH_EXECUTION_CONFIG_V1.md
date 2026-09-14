# Tier 2 Branch Execution Configuration V1

PROVENANCE=ADVISOR_AUTHORIZED_RESULT_BLIND_EXECUTION_FALLBACK  
PAPER_EXACT=False  
RESULT_BLIND=True  
PERFORMANCE_BLIND=True  
REAL_DATA_ACCESSED=False

## Shared contract

| Parameter | Value | Source class | Provenance |
|---|---:|---|---|
| `seq_len` | 10 | A | Advisor protocol / paper window statement |
| `pred_len` | 1 | A | Advisor-authorized fallback |
| `n_vars` | 3 | A | PLAN-CHANGE-0010 HF1/HF2/HF3 input contract |
| `primary_hidden_width` | 64 | A | Advisor-authorized result-blind fallback |
| loss | MSE | A | Advisor-authorized fallback |

## Trend and Seasonal TimesNet branch instances

| Parameter | Value | Source class | Provenance | Result blind |
|---|---:|---|---|---|
| `d_model` | 64 | A | Advisor hidden-width freeze | True |
| `d_ff` | 128 | A | Advisor-authorized 2× representation fallback | True |
| `e_layers` | 1 | A | Advisor-authorized minimal-depth fallback | True |
| `top_k` | 2 | A | Advisor-authorized minimal non-degenerate multi-period fallback | True |
| `dropout` | 0.1 | A | Advisor-authorized execution fallback | True |
| `num_kernels` | 2 | B | Existing structural core's pre-existing Inception fixture configuration; no old-project provenance | True |
| activation | GELU | B | Existing standard-core Inception activation | True |
| `branch_variant` | `STANDARD_REFERENCE_CORE` | B | Existing structural module constraint | True |
| normalization | none beyond module-native operations | B | Existing standard-core implementation has no extra configurable normalization | True |

Trend and Seasonal remain separately instantiated branches with distinct component targets and branch identities. Their exact target-paper deep/lightweight variants remain unresolved; this protocol does not claim to reproduce those variants exactly.

## Residual iTransformer branch instance

| Parameter | Value | Source class | Provenance | Result blind |
|---|---:|---|---|---|
| `d_model` | 64 | A | Advisor hidden-width freeze | True |
| `d_ff` | 128 | A | Advisor-authorized 2× representation fallback | True |
| `e_layers` | 1 | A | Advisor-authorized minimal-depth fallback | True |
| `n_heads` | 4 | A | Advisor-authorized standard divisor of 64 | True |
| `dropout` | 0.1 | A | Advisor-authorized execution fallback | True |
| activation | GELU | B | Existing iTransformer structural-core convention | True |
| encoder normalization | post-norm LayerNorm (`norm_first=False`) | B | Existing PyTorch encoder implementation setting | True |

`64 % 4 = 0`; the configuration is constructor-valid. No dynamic graph module is introduced.

## Execution boundary

`UNRESOLVED_EXECUTION_PARAMETERS=NONE` for the declared Tier 2 standard-core fallback path. Paper-exact architecture blockers remain unresolved and are not erased. `synthetic_only_mode=False` may be used only by a future approved Tier 2 pipeline that independently rejects TEST provenance.
