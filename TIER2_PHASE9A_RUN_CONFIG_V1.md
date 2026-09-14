# Tier 2 Phase 9A run configuration V1

`PLAN_CHANGE=PLAN-CHANGE-0017_REVISED`
`PROVENANCE=HUMAN_AUTHORIZED_RESULT_BLIND_FALLBACK`
`PAPER_EXACT=False`

| Field | Frozen value | Provenance |
|---|---|---|
| seed | 42 | Human authorization |
| dry-run epochs | 3 | Human authorization |
| optimizer | `torch.optim.Adam` | Human authorization |
| Adam beta1/beta2/eps/weight decay | library-native defaults: 0.9/0.999/1e-8/0.0 | execution-native detail, no performance selection |
| learning rate | 0.001 | paper-reported / Human freeze |
| batch size | 32 | paper-reported / Human freeze |
| device | CPU | Human authorization |
| deterministic algorithms | enabled | Human authorization |
| train batch order | chronological constructed windows, no shuffle | Human authorization |
| validation batch order | chronological, no shuffle | Human authorization |
| drop last | false | Human authorization |
| early stopping / patience | disabled / unused for 3-epoch dry run | Human authorization |
| checkpoint policy | disabled for dry run | Human authorization |

The runner currently uses a manual contiguous batch iterator, not a PyTorch
`DataLoader`; its behavior is therefore frozen as the equivalent of
`shuffle=False, drop_last=False`. It does not alter the chronological split or
sample identities.
