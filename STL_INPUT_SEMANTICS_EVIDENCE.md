# STL Input Semantics Evidence Audit

**Task:** `PHASE_2A_STL_SPECIFICATION_AND_LEAKAGE_CONTRACT`
**Evidence authority:** project-local paper PDF only.
**Scope:** evidence reconstruction; no STL run, feature/target materialization, model work, or test access.

## Paper evidence recovered

| Evidence ID | Location | Evidence text / visual summary | Implication | Confidence |
|---|---|---|---|---|
| STL-SEM-01 | PDF p.6, §3.3, Eq.(17) | STL decomposes the unspecified input series `X_t` as `X_t=T_t+S_t+R_t`. | The equation itself does not name the battery quantity. | High |
| STL-SEM-02 | PDF p.6, §3.3 | `T_t` is described as long-term monotonic **capacity fade**; `S_t` as recurrent short-term fluctuations; `R_t` as noise, capacity regeneration and other stochastic effects. | Semantic wording favors a capacity/degradation sequence. | High |
| STL-SEM-03 | PDF p.6, §3.3 | The paper says the three dedicated architectures process multivariate inputs in 10-step windows. | This describes downstream branch inputs, not a named STL input tensor. | High |
| STL-SEM-04 | PDF p.8, Fig.5 visual inspection | The upper augmentation panel contains HF1/HF2/HF3/SOH; the lower panel begins separately with an unlabelled “STL Decomposition” block. No arrow connects generated HFs or the top SOH label to that block. | A direct SCF-GAN-HF-to-STL input interface is not explicit. | High |
| STL-SEM-05 | PDF p.9, §3.7(2) | It identifies `T_t`, `S_t`, and `R_t` as decomposed components and assigns the three downstream architectures. | Branch routing is explicit; input quantity remains unstated. | High |
| STL-SEM-06 | PDF p.13, Fig.9 visual inspection | Fig.9(a) has SOH (%) / Error (%); Fig.9(b), (c), and (d) label trend, seasonal, and residual component axes in Ah. | Additive components shown in Ah are cross-evidence for a capacity-domain `X_t`, not a percent-domain SOH sequence. | High |
| STL-SEM-07 | PDF p.14, §4.4 | The final fused SOH curve is compared with the “actual fading capacity”; component figures are described as extracted trend/seasonal/residual predictions. | Paper wording mixes SOH and capacity language but does not provide an executable conversion. | Medium |

## Candidate assessment

| Candidate | Evidence for | Evidence against | Unit / figure consistency | Downstream consistency | Result |
|---|---|---|---|---|---|
| A. SOH percentage sequence | SOH is the stated final task; Fig.9(a) is labelled SOH (%). | Fig.9 component axes are Ah, while additive components must share `X_t` units. | Inconsistent with Fig.9(b–d) unless undocumented scaling/conversion occurred. | Possible but unsupported. | Not selected. |
| B. Capacity sequence | §3.3 calls trend “capacity fade”; Fig.9 components are in Ah. | `X_t` is not explicitly named “capacity.” | Strongly consistent: `T+S+R` all shown in Ah. | Components can be mapped to the stated branch routing. | **Selected by paper cross-evidence.** |
| C. HF1/HF2/HF3 multivariate sequence | Paper constructs HFs and calls branch inputs multivariate. | Fig.9 displays one Ah-valued additive triplet, not three duration-feature decompositions. | Not consistent with shown units. | No per-HF component routing or aggregation is specified. | Not selected. |
| D. HF-derived latent sequence | Could theoretically reconcile HFs with a learned representation. | No latent decomposition, encoder, unit mapping, or formula is stated. | No figure support. | Unsupported. | Not selected. |
| E. Other | None. | No paper evidence. | Unknown. | Unknown. | Unsupported. |

## Fig. 5 top/bottom connection

`FIG5_TOP_BOTTOM_CONNECTION=ABSENT`

The figure visually separates health-feature-based augmentation and the STL-based SOH framework. The lower STL block has no labelled incoming edge from the upper generated-feature output, HF1/HF2/HF3/SOH labels, or an explicit capacity series. Therefore it cannot establish that SCF-GAN HF output directly becomes the STL input.

## Decision boundary

`STL_INPUT_SEMANTICS_STATUS=RESOLVED_BY_CROSS_EVIDENCE_WITH_EXECUTABLE_SCALE_LIMITATION`

`STL_INPUT_SELECTED=CAPACITY_SEQUENCE`

`STL_INPUT_PROVENANCE=PAPER_CROSS_EVIDENCE__PDF_p6_SECTION_3_3__PDF_p13_FIG9`

This is not an assertion that the paper gives a complete capacity-to-SOH or component-label construction. The target denominator remains the Phase-1 Human fallback and the validation/test component-label method remains unresolved.
