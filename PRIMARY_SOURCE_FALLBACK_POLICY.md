# Primary-source Fallback Policy — PA-001 and PA-004

`AUTHOR_CONTACT_STATUS=CONTACTED`
`AUTHOR_REPLY_STATUS=CONTACTED_WAITING`
`HUMAN_CONFIRMED_EMAIL_SENT=True`
`GMAIL_MESSAGE_ID=1a0906ec8a4ae625`

## Governing rules

1. A clear author reply is a `PRIMARY_SOURCE_RESOLUTION`: preserve the original ambiguity record, record the reply verbatim or as a faithful evidence summary, and update the reproducibility specification with source and date.
2. If there is no reply, only the Human Researcher may authorize a pre-registered engineering decision. Codex must not select an option.
3. Selection is `RESULT_BLIND=True` and `PERFORMANCE_BLIND=True`. Future R², RMSE, MAE, model behavior, or test results are prohibited decision criteria.
4. The waiting interval is a Human decision. Permitted status values are `NOT_CONTACTED`, `CONTACTED_WAITING`, `REPLIED`, and `NO_RESPONSE`.

## PA-001 fallback decision record — Human authorization required

| Field | OPTION_A | OPTION_B |
|---|---|---|
| Interpretation | SOH = capacity / nominal rated capacity | SOH = capacity / first measured or initial capacity |
| Evidence for | The paper’s p.1 definition says residual capacity relative to nominal capacity. | The paper’s dataset discussion refers to initial rated capacity, but does not supply an executable denominator formula. |
| Evidence against | Dataset-specific nominal values and their application are not tabulated as an executable rule. | The paper does not explicitly define SOH using first/initial measured capacity. |
| Impact on reproduction | Changes label scale and all reported-error comparability. | Changes label scale and all reported-error comparability. |
| Strict-reproduction limitation | `NOT_STRICT_UNLESS_PRIMARY_SOURCE_CONFIRMS` | `NOT_STRICT_UNLESS_PRIMARY_SOURCE_CONFIRMS` |

`PA001_CHOSEN_INTERPRETATION=NONE`

## PA-004 fallback decision record — Human authorization required

| Field | OPTION_A | OPTION_B |
|---|---|---|
| Interpretation | Nested: first 20% of life is the original scarce set inside the stated first 60% training partition. | Separate: 60/20/20 and first-20%-of-life describe different experimental constructions. |
| Evidence for | Both statements concern target-battery training and can be read hierarchically. | Statements appear in differently scoped passages; Table 5 uses a roster not identical to §4 examples. |
| Evidence against | No text maps target cycles [0.2N, 0.6N) or says the first 20% is a subset of the 60%. | No text declares separate validation/test populations for the §4.2 construction. |
| Impact on reproduction | Determines original/synthetic training data and preservation of validation/test population. | Requires a separately declared evaluation protocol for the small-sample construction. |
| Strict-reproduction limitation | `NOT_STRICT_UNLESS_PRIMARY_SOURCE_CONFIRMS` | `NOT_STRICT_UNLESS_PRIMARY_SOURCE_CONFIRMS` |

`PA004_CHOSEN_INTERPRETATION=NONE`

## Required record before any fallback execution

Create a Human-authorized `REPRODUCTION_DEVIATION_RECORD` naming the chosen option, evidence, non-performance rationale, protocol impact, and `THIS_IS_NOT_STRICT_REPRODUCTION=True`. Do not load the test set, run training, or advance to protocol freeze merely because a fallback document exists.
