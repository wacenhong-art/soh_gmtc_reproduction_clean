# Cross-Phase G-MTC Dataflow Reconciliation — Phase 7A

## Question

The paper calls HF1/HF2/HF3 foundational health indicators, says SCF-GAN generates health-feature vectors for target batteries, and says generated data train the subsequent STL-Transformer-TimesNet model.  Separately, the STL evidence is capacity/SOH-domain cross-evidence and the lower Fig.5 panel routes `Trend`, `Seasonal`, and `Residual` into specialized branches.  This audit asks which explicit interface connects those statements without inventing an unreported dataflow.

## Candidate assessment

| Candidate | Evidence for | Evidence against / missing interface | Result |
|---|---|---|---|
| A. HF windows are branch inputs; capacity/SOH is decomposed only for component supervision targets | HF1/HF2/HF3 are upstream indicators; §3.3 says architectures process multivariate inputs; generated data are used for later model training. | No equation or Figure 5 arrow maps HF windows to any component branch or defines T/S/R supervision targets from a separate capacity/SOH series. | Not executable. |
| B. SCF-GAN emits HF1/HF2/HF3 plus SOH; emitted SOH is decomposed and emitted HFs are branch inputs | Fig.3 explicitly labels a four-dimensional generated vector `HF1/HF2/HF3/SOH`; Fig.5 top feature list includes SOH. | §4.2 only explicitly names generated HF1/HF2/HF3; Fig.5 has no Generated-Features-to-STL arrow; no normalization/capacity-to-SOH or branch-input mapping exists. | **Most supported cross-evidence, but not paper-executable.** |
| C. SCF-GAN-generated HFs themselves enter STL | Generated HFs are upstream data. | STL prose defines trend as monotonic capacity fade; no per-HF STL construction is shown. | Unsupported. |
| D. T/S/R are direct branch inputs only | Fig.5 lower panel and §3.7 route components to specialized models. | Does not specify how SCF-GAN output is used by those branches. | Lower-panel branch routing only; upstream interface unresolved. |

## Visual interface audit

`FIG5_TOP_BOTTOM_CONNECTION=ABSENT`

The top panel visibly produces `Generated Features` and routes them within the GAN panel.  The lower panel begins independently at `STL Decomposition`.  Neither an arrow nor a textual equation defines generated feature fields, conversion to capacity/SOH, decomposition input, branch tensor composition, or component-target construction.

## Reconciliation finding

`CROSS_PHASE_DATAFLOW_STATUS=BLOCKED__TOP_BOTTOM_INTERFACE_ABSENT_AND_GENERATED_OUTPUT_DOMAIN_CONFLICT`

`MOST_SUPPORTED_DATAFLOW_CANDIDATE=B__FIGURE_CROSS_EVIDENCE_ONLY__NOT_EXECUTABLE`

`CORE_DATAFLOW_CORRECTION_REQUIRED=True`

Fig.3's four-dimensional output label makes Candidate B the strongest available *cross-evidence* reading.  It conflicts with the Phase 2 capacity-domain STL cross-evidence unless an unreported SOH/capacity conversion and exact component-construction rule exists.  It also does not resolve the Phase 3/5 branch-tensor ambiguity: existing real-data branch contracts were deliberately left unresolved, not frozen as component-only inputs.  Therefore this is not a basis for silently selecting Candidate B or revising existing structural unit tests.

## Gate and stop rule

`GMTC_CROSS_PHASE_DATAFLOW_GATE=PLAN_CHANGE_REQUIRED`

`PLAN_DEVIATION_LEVEL=2`

`PLAN_CHANGE_REQUIRED=True`

`REAL_DATA_INTERFACE_ASSUMPTIONS_REOPENED=True`

`REAL_SCFGAN_TRAINING=False`

`REAL_SYNTHETIC_ROWS_CREATED=False`

`TEST_LOADED=False`

No SCF-GAN implementation is authorized in this run.  The existing synthetic-only STL, iTransformer, TimesNet and fusion modules remain valid software structural tests; they are not evidence of a resolved real-paper dataflow.
