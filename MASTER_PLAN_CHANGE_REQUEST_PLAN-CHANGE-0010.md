# MASTER PLAN CHANGE REQUEST — PLAN-CHANGE-0010

`NAME=SCFGAN_AUGMENTED_HF_TO_DECOMPOSED_TARGET_DATAFLOW_RECONCILIATION`

`LEVEL=2`

`STATUS=REQUESTED__HUMAN_AUTHORIZATION_REQUIRED`

`HUMAN_AUTHORIZED=False`

`TRIGGER=PHASE_7A_CROSS_PHASE_DATAFLOW_AUDIT`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`TEST_LOADED=False`

`REAL_BATTERY_DATA_USED=False`

## Defect

The target paper gives mutually incomplete evidence for the bridge from SCF-GAN augmentation to STL/component modelling.  Fig.3 labels a generated four-dimensional vector `HF1/HF2/HF3/SOH`; §4.2 explicitly names generated HF1/HF2/HF3; Fig.5 contains no visible Generated-Features-to-STL connection; and Phase 2 capacity-domain STL cross-evidence has no stated SOH/capacity conversion or component-target construction.  The missing bridge changes the real-data domain and tensor meaning, not merely a library parameter.

## Requested amendment scope

Before any real SCF-GAN, real STL, real branch window, or full G-MTC work:

1. reopen the real-data interfaces across Phase 2–5;
2. specify the SCF-GAN output fields and their provenance;
3. specify whether the STL input is capacity, SOH, generated SOH, or another explicitly supported series;
4. specify the branch inputs and T/S/R target alignment;
5. maintain all existing synthetic-only structural modules as tests, without relabelling them as real-paper outputs.

## Non-changes

`CORE_MODEL_ARCHITECTURE_CHANGED=False`

`RESEARCH_GOAL_CHANGED=False`

`RAW_DATA_PROTOCOL_CHANGED=False`

`NO_MODEL_RESULTS_SEEN=True`

`NO_REAL_DATA_WAS_LOADED=True`

## Required Human decision

Choose a documented, result-blind path after reviewing the evidence packet: seek/await primary-source clarification, authorize a pre-registered cross-phase fallback, or determine that strict numerical reproduction remains blocked.  This request does not authorize Candidate A, Candidate B, a real condition encoding, an SCF-GAN architecture variant, a real period, or implementation.

## Audit references

- `SCFGAN_PAPER_EVIDENCE.md`
- `SCFGAN_CONDITION_SEMANTICS_AUDIT.md`
- `CROSS_PHASE_GMTC_DATAFLOW_RECONCILIATION.md`
- target-paper Fig.3 / Fig.5 rendered evidence in `output/phase7_figure_visual_evidence/`

## Application record — 2026-09-13

`HUMAN_AUTHORIZATION=True`

`STATUS=APPLIED__HUMAN_AUTHORIZED__RESULT_BLIND_CROSS_PHASE_FALLBACK`

`APPLIED_POLICY=HF_INPUT_TO_COMPONENT_TARGET_MODE`

`PROVENANCE=HUMAN_RESULT_BLIND_CROSS_PHASE_FALLBACK`

`RESULT_BLIND=True`

`PERFORMANCE_BLIND=True`

`TEXT_EVIDENCE_PRIORITY=True`

`NO_SYNTHETIC_SOH_ASSUMPTION=True`

`AUTHOR_OVERRIDE_PENDING=True`

The authorized fallback uses HF1/HF2/HF3 windows as future branch inputs and observed training capacity through STL as future T/S/R component supervision targets. It does not resolve the paper contradiction, authorize a real SCF-GAN configuration, select a condition encoder, set an STL period, or determine branch target alignment. See `GMTC_REAL_DATAFLOW_FALLBACK_CONTRACT.md`.
