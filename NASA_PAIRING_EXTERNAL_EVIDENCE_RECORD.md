# NASA Pairing External Evidence Record

STATUS=EVIDENCE_RECORD__TIER2_ENGINEERING_FALLBACK_ONLY
REAL_HF_VALUES_USED=False
REAL_NUMERICAL_TARGET_VALUES_USED=False
TEST_LOADED=False

## Scope and evidence boundary

This record supports PLAN-CHANGE-0015 only. It does not claim that the target Energy paper's authors used the policy frozen below, and it does not replace the retained paper-exact pairing ambiguity.

`NO_SOURCE_CLAIMS_CANDIDATE_E_AS_TARGET_PAPER_EXACT=True`

## A. Official NASA data-format fact

NASA's [Li-ion Battery Aging Datasets documentation](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) describes top-level `cycle` operations with `charge`, `discharge`, and `impedance` types. Its listed charge fields do not include Capacity; its discharge fields include `Capacity` in Ahr.

`EVIDENCE_CLASS=OFFICIAL_DATA_FORMAT_FACT`
`CHARGE_CAPACITY_NONE_IS_EXPECTED=True`

This supports treating a charge operation as the HF source and a discharge operation as the capacity-label source. It does not by itself specify their exact one-to-one pairing rule.

## B. MathWorks NASA workflow precedent

MathWorks' [Battery Second-Life Application State of Health Estimation of Li-Ion Batteries](https://www.mathworks.com/help/predmaint/ug/battery-second-life-app-soh-estimation.html) workflow notes incomplete measurements and repeated charging cycles without discharge for B0005, B0006, and B0007; it removes related measurements before analysis and extracts charge, discharge, and capacity for normal cycles.

`EVIDENCE_CLASS=EXTERNAL_WORKFLOW_PRECEDENT`

This is a public methodology precedent for excluding incomplete/repeated charging records before constructing aligned charge/discharge/capacity samples. The page does not prescribe this project's exact discharge-centered consumption implementation.

## C. Published NASA preprocessing precedent

The open-access study [State of health estimation of lithium-ion batteries based on convolutional neural network](https://pmc.ncbi.nlm.nih.gov/articles/PMC9736597/) uses B0005, B0006, B0007, and B0018 and reports 168 valid capacity cycles for each of B0005/B0006/B0007 and 132 for B0018. Those counts agree with this project's order-only audit after unmatched charge operations are excluded.

`EVIDENCE_CLASS=PUBLISHED_PREPROCESSING_PRECEDENT`
`EXACT_DISCHARGE_CENTERED_RULE_CONFIRMED_BY_THIS_SOURCE=False`

The literature precedent corroborates usable-cycle counts, not a claim that the publication or the target Energy paper implements Candidate E verbatim.

## D. Local raw-sequence structural evidence

`NASA_DISCHARGE_CENTERED_PAIRING_AUDIT.md` and `NASA_DISCHARGE_CENTERED_PAIRING_MANIFEST.json` establish, without accessing Capacity/HF/SOH values, that Candidate E is deterministic, temporally ordered, one-to-one, and gives the expected valid-pair counts on B0005, B0006, B0007, and B0018. Impedance operations occur between valid members and are therefore permitted by the policy.

`EVIDENCE_CLASS=RAW_SEQUENCE_STRUCTURAL_EVIDENCE`

## Result-blind conclusion

The combined evidence supports a `STRONG_FOR_TIER2_ENGINEERING_REPRODUCTION` pairing fallback: it is data-format-consistent, public-workflow-supported, structurally audited, deterministic, and does not use model results or numerical label/feature values. It remains distinct from `PAPER_EXACT_PAIRING_STATUS=UNRESOLVED`.
