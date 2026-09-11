# Primary-source Clarification Packet

`PAPER_TITLE=Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach`

`DOI=10.1016/j.energy.2025.139463`

`PRIMARY_CONTACT=Simin Peng, corresponding author, School of Electrical Engineering, Yancheng Institute of Technology`

`CONTACT_SOURCE=Target paper PDF p.1: asterisk corresponding-author footnote and “E-mail address: siminpeng@ycit.edu.cn (S. Peng).”`

`CONTACT_CONFIDENCE=HIGH__TARGET_PAPER_DIRECT_EVIDENCE`

`SECONDARY_CONTACT=Hanhua Yang, first author; use only if Human elects follow-up after the corresponding-author route is exhausted. No secondary email is asserted here.`

## PA-001 evidence and question

`PA001_EVIDENCE_LOCATIONS=PDF p.1 (SOH described as residual capacity relative to nominal capacity); PDF p.3 dataset discussion; PAPER_AMBIGUITY_REGISTER.md PA-001.`

`PA001_EXACT_UNKNOWN=The exact executable denominator for each dataset’s SOH label.`

**Question:** For each experimental dataset, was SOH computed as capacity divided by nominal rated capacity, by the first/initial measured capacity, or by another specified reference capacity?

## PA-004 evidence and question

`PA004_EVIDENCE_LOCATIONS=PDF p.10 §4 (chronological target split: 60% train / 20% validation / 20% test); PDF p.10 §4.2 (original target training set: first 20% of total life; five synthetic datasets); PA004_PROTOCOL_EVIDENCE_TABLE.md E04–E06.`

`PA004_EXACT_UNKNOWN=Whether the 60/20/20 and first-20%-of-life target-data constructions are nested, separate experiments, or another arrangement; consequently, the exact train/validation/test/augmentation target-cycle allocation.`

**Question:** Is the first-20%-of-life original small-sample set nested inside the stated 60% target training partition, a separate experimental setting, or another arrangement? Please identify the target-cycle ranges used for temporal-model training, validation, testing, and SCF-GAN-augmented samples if available.

## Packet boundary

The email draft is in [AUTHOR_CLARIFICATION_REQUEST.md](C:/Users/ROG/Documents/Python/new_soh/AUTHOR_CLARIFICATION_REQUEST.md). No message has been sent. This packet does not select a fallback, create a data split, or change the original ambiguity history.

