# PLAN-CHANGE-0011 — Formalize Reproduction Fidelity Levels

`STATUS=PROPOSED_NOT_APPLIED`

`PROPOSED_LEVEL=1`

`TRIGGER=PHASE8B_NUMERICAL_FIDELITY_BLOCKER_TRIAGE`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`REAL_DATA_USED=False`

`TEST_LOADED=False`

`CORE_ARCHITECTURE_CHANGED=False`

`DATA_PROTOCOL_CHANGED=False`

`RESEARCH_GOAL_CHANGED=False`

## Proposed governance clarification

The Master Plan should formally distinguish three reporting ceilings:

1. `FRAMEWORK_REPRODUCTION` — structural modules, schemas, and guards validated without real numerical claims.
2. `FALLBACK_NUMERICAL_REPRODUCTION` — a separately Human-authorized, result-blind, fully disclosed numerical variant.
3. `PAPER_EXACT_NUMERICAL_REPRODUCTION` — permitted only when primary-source blockers are resolved.

## Rationale

Phase 8B shows that a numerical integration cannot become paper-exact merely because the framework and synthetic modules execute. Class C/D blockers include branch target semantics, inference-safe fusion errors/ratios, SCF-GAN condition identity, fusion objective, and shared adversarial training. The clarification prevents a fallback result from being reported as a strict numerical reproduction.

## Non-effects

This request does not apply a fallback, alter PLAN-CHANGE-0010, change any active input/target contract, authorize Phase 9, or remove any blocker. Human approval is required before the Master Plan’s active status language changes.

## Application record — 2026-09-13

`HUMAN_AUTHORIZATION=True`

`STATUS=APPLIED__HUMAN_AUTHORIZED__GOVERNANCE_ONLY`

`RESULT_DRIVEN=False`

`PERFORMANCE_DRIVEN=False`

`REAL_DATA_USED=False`

`TEST_LOADED=False`

The change is applied only to make the three reporting tiers and Phase 9 lock explicit. It does not apply any numerical fallback, modify a real-data protocol, alter architecture, clear Class C/D blockers, or authorize experiments.
