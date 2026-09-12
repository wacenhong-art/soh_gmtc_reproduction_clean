# TimesNet paper evidence — Phase 5A

**Primary authority:** project-local target-paper PDF, 17 pages. **Secondary architecture reference:** Wu et al., *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* (ICLR 2023) and its official `thuml/TimesNet` repository, consulted only to make the generic structural test interface explicit. No secondary default is a target-paper parameter.

## Target-paper TimesNet evidence

| Field | Value | Status | Target-paper location | Evidence summary |
|---|---|---|---|---|
| `TIMESNET_SECTION` | §3.4 TimesNet | EXACT | p.6–7 | Defines a three-stage multi-period temporal architecture. |
| `TIMESNET_PURPOSE` | Characterize intra-period and inter-period evolution through 1D-to-2D transformation. | EXACT | p.6 §3.4 | 2D tensors support period-specific and cross-period modelling. |
| `TIMESNET_INPUT_STATEMENT` | `X ∈ R^(T×C)`, where `T` is timesteps and `C` channels. | EXACT | p.6 Eq.(18) | Does not map `C` to the real Trend/Seasonal branch tensor. |
| `TIMESNET_PERIOD_DETECTION` | FFT amplitude analysis selects top-k significant frequencies. | EXACT | p.6 Eq.(18) | `k` is a concept; its integer value is not supplied. |
| `TIMESNET_FFT_EQUATION` | `A=Avg(Amp(FFT(X)))`; `{f1,…,fk}=argTopk(A)`. | EXACT | p.6 Eq.(18) | DC handling and tie handling are not stated. |
| `TIMESNET_2D_RESHAPE_EQUATION` | `pi=ceil(T/fi)` and reshape into a period-dependent 2D tensor. | EXACT | p.7 Eq.(19) | Padding/crop rule is not supplied. |
| `TIMESNET_INCEPTION_STATEMENT` | Parameter-shared Inception blocks process each periodic tensor. | EXACT | p.7 Eq.(20) | Kernel, depth, channels, activation, and normalization are absent. |
| `TIMESNET_ADAPTIVE_MERGE_STATEMENT` | `Xout=AdaptiveMerge({X'i})`. | EXACT | p.7 Eq.(20) | Merge formula is absent. |
| `TREND_TIMESNET_DESCRIPTION` | Deep TimesNet configuration with multi-scale convolutional blocks. | EXACT descriptive claim | p.6 §3.3; p.9 §3.7(2) | No executable depth/configuration is provided. |
| `SEASONAL_TIMESNET_DESCRIPTION` | Lightweight TimesNet variant with adaptive period/cycle detection. | EXACT descriptive claim | p.6 §3.3; p.9 §3.7(2) | No reduction or extra mechanism is specified. |
| `TIMESNET_WINDOW_STATEMENT` | Each architecture processes multivariate inputs using dedicated 10-step sliding windows. | EXACT | p.6 §3.3 | Input history only; not a TimesNet period, top-k, or forecast horizon. |

`STANDARD_TIMESNET_CORE_STATUS=PAPER_SUPPORTED`

`TIMESNET_TOPK_CONCEPT=PAPER_EXACT`

`TIMESNET_TOPK_VALUE=MISSING`

`TIMESNET_DETECTED_PERIODS != STL_PERIOD`

`SEQ_LEN=10`

`SEQ_LEN_IS_STL_PERIOD=False`

`SEQ_LEN_IS_TIMESNET_PERIOD=False`

`SEQ_LEN_IS_TOPK=False`

## Missing target-paper detail

| Field | Status | Why it remains unresolved |
|---|---|---|
| `TREND_DEPTH_VALUE_STATUS` | MISSING | “Deep” is descriptive; block count is absent. |
| `TREND_BLOCK_COUNT_STATUS` | MISSING | No layer/TimesBlock count. |
| `TREND_CHANNEL_WIDTH_STATUS` | MISSING | No channel or model width. |
| `TREND_KERNEL_CONFIGURATION_STATUS` | MISSING | No kernel set/count/padding. |
| `SEASONAL_LIGHTWEIGHT_DEFINITION_STATUS` | DESCRIPTIVE_ONLY__NO_IMPLEMENTATION_DETAIL | No reduced layer/channel/kernel/parameter definition. |
| `ADAPTIVE_PERIOD_DETECTION_STATUS` | AMBIGUOUS__STANDARD_FFT_OR_UNSPECIFIED_EXTRA_MECHANISM | The paper describes standard FFT detection and separately calls Seasonal detection adaptive, without an extra equation/mechanism. |
| `INCEPTION_CONFIGURATION_STATUS` | MISSING | Eq.(20) names Inception only. |
| `ADAPTIVE_MERGE_DETAIL_STATUS` | MISSING | Eq.(20) names AdaptiveMerge only. |
| `TREND_TARGET_ALIGNMENT_STATUS` | UNRESOLVED | No 10-step target offset/horizon/component target is stated. |
| `SEASONAL_TARGET_ALIGNMENT_STATUS` | UNRESOLVED | No 10-step target offset/horizon/component target is stated. |

## Standard-reference boundary

The official TimesNet reference shows a standard realization based on FFT-amplitude weighting, period-dependent padding/reshape, 2D Inception-style convolution, and amplitude-based merging. These details are used only to make a generic synthetic contract testable.

`REFERENCE_ARCHITECTURE_ONLY=True`

`TARGET_PAPER_PARAMETER_INFERENCE=False`
