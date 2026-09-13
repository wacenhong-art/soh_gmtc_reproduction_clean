# Phase 8B — Author Clarification Shortlist

`EMAIL_STATUS=DO_NOT_SEND__AUTHOR_ALREADY_CONTACTED`

The following six questions are a concise future-reply checklist only. They do not authorize a duplicate message.

1. For each 10-step HF1/HF2/HF3 window, what are the exact trend, seasonal, and residual target cycles/horizon (`t`, `t+1`, or multi-step), and are all three branches fed by the same HF tensor?
2. Which signal is decomposed by STL in the real experiment, and what period/smoother settings and split/component-label procedure were used for train, validation, and test?
3. In the 9 fusion features, what are the exact definitions and inference-time sources of each component error and ratio? Does any feature use the current true component/SOH target?
4. What is `d` in 9d/3d, which three fields are the first 3d values, and is the fused final output capacity, normalized capacity, or SOH (including its normalization formula)?
5. What exactly is the SCF-GAN condition vector and encoding, which fields are generated, and what normalization maps real HF values to/from the Generator Tanh output?
6. Please confirm the actual SCF-GAN and G-MTC training configuration: generator activation/normalization/SN, discriminator depth, latent dimension, end-to-end versus staged orchestration, and the shared-adversarial/fusion-loss terms and weights.

`AUTHOR_CLARIFICATION_VALUE=CRITICAL`

`DUPLICATE_EMAIL_SENT=False`
