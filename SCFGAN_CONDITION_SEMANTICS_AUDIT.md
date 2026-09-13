# SCF-GAN Condition Semantics Audit — Phase 7A

| Candidate condition interpretation | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| Cycle only | §3.2 names `c` a cycle-period condition; §3.7 describes cycle condition/aging stage. | Fig.3 and Fig.5 each display a second condition, and §4.2 names voltage range. | Insufficient. |
| Voltage range only | §4.2 says voltage range is incorporated as a conditional variable. | §3.2, §3.7 and both figures include cycle. | Unsupported. |
| Cycle + voltage | Fig.5 explicitly labels Cycle and Voltage; §4.2 supports voltage range. | Fig.3 instead labels Cycle and Battery type. | Strong figure/text cross-evidence, not unique. |
| Cycle + voltage + battery identity/chemistry | §3.7 prose mentions battery characteristics such as chemistry and usage conditions. | No vector construction, no equation, and Fig.5 shows only two labels. | Unsupported implementation addition. |
| Experiment-specific conditions | Different sections/figures could describe different experiments. | The paper never declares distinct condition vectors or mappings by experiment. | Plausible but unconfirmed. |

`SCFGAN_CONDITION_SEMANTICS_STATUS=CONTRADICTORY__CYCLE_VOLTAGE_BATTERY_TYPE_CHEMISTRY_AND_USAGE`

`FIG5_CONDITION_SET=CYCLE_PLUS_VOLTAGE__PAPER_FIGURE_EVIDENCE`

`CYCLE_CONDITION_ENCODING_STATUS=MISSING`

`VOLTAGE_CONDITION_ENCODING_STATUS=MISSING`

The paper does not state whether cycle is a raw index, normalized index, lifecycle fraction, bin, or another representation.  It does not state whether voltage is a lower/upper range, width, bin, one-hot identifier, or another representation.  No real condition encoder is authorized.

`REAL_CONDITION_ENCODING_INFERENCE=False`
