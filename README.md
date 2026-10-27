# GMTC State-of-Health Reproduction

This repository records a reproducible research workflow for battery state-of-health (SOH) estimation using a GMTC-oriented architecture. It brings together feature extraction, STL-based decomposition, sequence-model components, adaptive fusion, and SCF-GAN-related experiments, along with the evidence and audit records used to keep the reproduction traceable.

## Scope

The project is an evidence-first reproduction effort rather than a claim of an exact paper replication. It preserves protocol assumptions, unresolved ambiguities, data provenance, implementation contracts, and numerical run audits so that each result can be interpreted within its documented boundary.

## Repository guide

- `scripts/` — pipeline and supporting scripts.
- `references/` — literature and source-material records.
- `output/` — generated experiment artifacts and reports.
- `PAPER_REPRODUCTION_SPEC.md` — paper-to-code reproduction specification.
- `REPRODUCTION_MASTER_PLAN.md` — staged implementation and validation plan.
- `REPRODUCTION_PROTOCOLS.md` — protocol index and execution constraints.
- `DATA_SOURCE_MANIFEST.md` — data-source and provenance record.
- `PHASE*_*.md` / `PHASE*_*.json` — phase-specific designs, metrics, manifests, and audits.

## Reproducibility principles

- Keep dataset provenance, sample identity, splitting, normalization, and pairing decisions explicit.
- Separate pre-test validation from sealed-test evaluation.
- Treat unresolved paper details as documented assumptions, not silent substitutions.
- Preserve weak, failed, and blocked outcomes alongside successful runs.
- Consult phase reports and provenance manifests before comparing metrics across runs.

## Status

This repository contains staged research artifacts. Individual phase reports describe their own readiness, evidence tier, execution status, and any remaining blockers; they should be read before treating a result as final.

## Data and responsible use

Source datasets may have their own licenses and access conditions. Obtain data from the original providers and follow their terms. Do not commit credentials, private datasets, or personally sensitive information.