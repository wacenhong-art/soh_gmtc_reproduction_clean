# Tier 2 SCF-GAN real-execution protocol V1

**Protocol status:** frozen for Phase 9C smoke execution by PLAN-CHANGE-0018.  
**Provenance:** `HUMAN_AUTHORIZED_RESULT_BLIND_FALLBACK`.  
**Paper-exact status:** unresolved.

## Purpose and boundary

This is a source-only real-data execution smoke protocol, not a numerical
reproduction claim and not downstream augmentation authorization. It preserves
the Phase 7 paper ambiguity record and the Phase 9A/9B frozen protocols.

## Architecture and loss wiring

`z ~ N(0,I), noise_dim=16; c in R^1 is normalized paired-cycle position within
each battery lifecycle.`

Generator: `concat(z,c) -> [64,128,64] -> ReLU + ConditionalBatchNorm ->
HF(3) -> Tanh`; generator SN is false. Discriminator: `concat(HF,c) ->
[64,128,64,32] -> LeakyReLU(0.2) -> Sigmoid`; discriminator SN is true.
Feature matching uses the third discriminator hidden output with lambda 0.1.

The discriminator loss is binary cross entropy for real source HF and
generated HF. The generator loss is non-saturating BCE adversarial loss plus
`0.1 *` mean-squared distance between real and generated third-hidden
features. The source is a real source-HF batch only.

## Data contract

```text
SCFGAN_SOURCE_ONLY_GATE=required
Training HF: B0005 + B0006 only
Training condition: source paired-cycle position only
Target numerical HF / capacity / SOH in GAN training: prohibited
Validation / test values in GAN training: prohibited
```

The source GAN scaler is fitted over all permitted source HF samples only:
`MinMax[-1,1]`. The Tanh generator works in that domain. Generated rows are
inverse-transformed to physical HF after generation. This scaler is distinct
from the Phase 9A Train-only `[0,1]` downstream scaler.

## Target generation contract

For B0007 and B0018, only the first 20% lifecycle paired-cycle identities are
used. Their ordered normalized positions form conditions. For each target,
generate five independent sets of physical `[N,3]` tensors. The dimensions are
strictly HF1, HF2, HF3. No capacity or SOH value is created.

## Frozen runtime configuration

`optimizer=Adam; learning_rate=0.001; betas=(0.9,0.999); batch_size=32;
seed=42; device=CPU; smoke_epochs=20; early_stopping=False; grid_search=False.`

## Retained scientific status

```text
PAPER_GENERATOR_ACTIVATION_CONTRADICTION=True
PAPER_GENERATOR_SN_SCOPE_CONTRADICTION=True
PAPER_DISCRIMINATOR_DEPTH_CONTRADICTION=True
PAPER_SCFGAN_CONDITION_CONTRADICTION=True
PAPER_EXACT_SCFGAN_EXECUTION_STATUS=UNRESOLVED
TIER2_SCFGAN_CONFIG_STATUS=HUMAN_AUTHORIZED_RESULT_BLIND_FALLBACK
```
