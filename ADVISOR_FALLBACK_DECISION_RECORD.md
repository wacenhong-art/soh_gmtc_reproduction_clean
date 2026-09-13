# Advisor Fallback Decision Record

ADVISOR_AUTHORIZED_FALLBACK=True  
DATE=2026-09-13 (session record)  
PAPER_EXACT=False  
RESULT_BLIND=True  
PERFORMANCE_BLIND=True  
AUTHOR_REPLY_STATUS=CONTACTED_WAITING

This record preserves the advisor's explicit preparation guidance. It is not an author clarification and must not be represented as a statement of the paper's original implementation.

## Frozen decisions

| Area | Advisor-authorized fallback | Provenance and boundary |
|---|---|---|
| Forecast alignment | 10-step input to one future step: `[x_(t-9), ..., x_t] -> y_(t+1)` | Advisor fallback; no same-step or seq2seq comparison |
| STL period | 7, midpoint of the advisor's 6–8 range | Result-blind midpoint freeze; no period performance comparison |
| Normalization | Per-feature min–max to 0–1, fit on TRAIN only | Validation/test only transform with TRAIN min/max; out-of-range policy is explicit clipping to [0,1] |
| Hidden width | 64 | Smaller endpoint of advisor's 64–128 range; complexity-minimizing, not result-selected |
| Fusion | Direct weighted component fusion, T/S/R = 0.7/0.2/0.1 | Tier 2 execution fallback; does not replace the paper's unresolved 9-feature fusion |
| Loss | MSE, `mean((target-prediction)^2)` | Tier 2 training/evaluation loss; undefined paper error/ratio features remain unresolved |

## Dataflow boundary

Future Tier 2 preparation uses `[HF1, HF2, HF3]` 10-step windows as branch inputs. Observed TRAIN capacity is the source for STL targets (period 7). SCF-GAN may only produce HF1/HF2/HF3; synthetic SOH is prohibited. No real rows or model training are authorized by this record.

Any change to these frozen decisions requires a new Plan Change. Author clarification, if later received, must first pass the primary-source reconciliation audit.
