# PLAN-CHANGE-0013 — Correct Advisor Split and Fusion Semantics

STATUS=APPLIED__HUMAN_CLARIFICATION__BEFORE_NUMERICAL_EXECUTION  
TYPE=HUMAN_CLARIFICATION_OF_ADVISOR_GUIDANCE  
RESULT_DRIVEN=False  
PAPER_EXACT=False

## Correction

The prior interpretation `FUSION_WEIGHTS=T:0.7,S:0.2,R:0.1` is retracted before numerical execution. Human clarification establishes that 7:2:1 denotes the dataset split, not component weights.

`TRAIN:VALIDATION:TEST = 7:2:1`, implemented chronologically as first 70%, next 20%, final 10%, with half-open zero-based boundaries and no shuffle. This is an advisor-authorized Tier 2 fallback and does not replace the paper's historical split ambiguity.

Tier 2 fusion is now `ADVISOR_AUTHORIZED_COMPONENT_SUM`: `y_hat_(t+1)=T_hat_(t+1)+S_hat_(t+1)+R_hat_(t+1)`, with `FUSION_WEIGHTS=NONE`. The paper's 9-feature Bi-LSTM fusion remains a Tier 1 structural artifact with unresolved error/ratio semantics.

All other frozen parameters remain unchanged: sequence 10, prediction length 1, STL period 7, TRAIN-only min–max 0–1 normalization, hidden width 64, MSE, HF1/HF2/HF3-only SCF-GAN output, and TEST lock.
