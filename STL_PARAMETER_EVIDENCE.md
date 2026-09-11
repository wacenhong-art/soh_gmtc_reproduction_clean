# STL Parameter Evidence Audit

**Authority:** local paper PDF, full-text search and §3.3 review.
**Result:** the paper gives the additive decomposition and a generic seasonal-period symbol, but no battery-specific STL configuration.

| PARAMETER | PAPER_VALUE | STATUS | EVIDENCE_LOCATION | Notes |
|---|---|---|---|---|
| Additive form | `X_t=T_t+S_t+R_t` | EXACT | PDF p.6 §3.3 Eq.(17) | Mathematical identity, not a software configuration. |
| Seasonal period | `p` | AMBIGUOUS | PDF p.6 §3.3 | Symbol only; no battery value. |
| `p=12` | “e.g., p=12 for monthly data” | EXACT_AS_GENERIC_EXAMPLE | PDF p.6 §3.3 | Not a battery-project parameter. |
| Seasonal window | not reported | MISSING | PDF §§3.3–4.4 | — |
| Trend window | not reported | MISSING | PDF §§3.3–4.4 | — |
| Low-pass window | not reported | MISSING | PDF §§3.3–4.4 | — |
| Robust iterations / robust flag | not reported | MISSING | PDF §§3.3–4.4 | — |
| LOESS degree | not reported | MISSING | PDF §§3.3–4.4 | — |
| Seasonal / trend / low-pass jump | not reported | MISSING | PDF §§3.3–4.4 | — |
| Boundary handling | not reported | MISSING | PDF §§3.3–4.4 | — |
| Library / implementation | not reported | MISSING | PDF §§3.3–4.4 | — |
| Windowing relative to STL | not reported | MISSING | PDF p.6 §3.3; p.9 §3.7 | The 10-step downstream branch windows are not STL windows. |

`P12_MONTHLY_EXAMPLE_IS_PROJECT_PARAMETER=False`

`DOWNSTREAM_SEQUENCE_LENGTH=10`

`DOWNSTREAM_SEQUENCE_LENGTH_IS_STL_WINDOW=False`

`STL_PARAMETER_POLICY_RESOLVED=False`

`STL_PERIOD_STATUS=MISSING_BATTERY_SPECIFIC_VALUE`

`STL_WINDOW_STATUS=MISSING`

The existing missing-parameter entry `PM-002` is reused. No library default, period, or window may be represented as paper-exact. A later leakage-safe engineering choice requires explicit Human authorization and a deviation record.
