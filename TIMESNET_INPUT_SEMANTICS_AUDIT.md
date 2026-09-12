# TimesNet input semantics audit — Phase 5A

## Evidence matrix

| Candidate | Evidence for | Evidence against / missing mapping | Verdict |
|---|---|---|---|
| Univariate Trend sequence | Fig.5 lower panel routes a Trend component representation into the branch; §3.7 calls `T_t` the component processed by TimesNet. | §3.3 says each architecture processes multivariate inputs; no tensor axis is defined. | Plausible, not executable as paper fact. |
| Univariate Seasonal sequence | Fig.5 and §3.7 analogously route Seasonal component `S_t`. | Same multivariate-window statement leaves channel count undefined. | Plausible, not executable as paper fact. |
| Multivariate decomposed components | §3.4 generic notation uses `X∈R^(T×C)`. | Paper does not state that each branch receives multiple decomposed channels. | Plausible, unsupported. |
| Component plus HF/raw signals | HFs exist upstream. | No Fig.5 arrow or prose maps HF/voltage/current/temperature side inputs into either TimesNet branch. | Unsupported. |
| Multiple STL-decomposed channels | Compatible with generic `C` notation. | No decomposition-per-channel or branch-routing construction is given. | Unsupported. |

`TIMESNET_INPUT_SEMANTICS_STATUS=UNRESOLVED_BLOCKING_FOR_REAL_DATA`

## Figure 5 audit

The lower panel visibly labels Trend, Seasonal, and Residual component pathways. It does not display an added HF, voltage, current, temperature, or other side-channel arrow for Trend or Seasonal.

`FIG5_TREND_INPUT_STATUS=COMPONENT_ONLY__NO_EXTRA_CHANNELS_SHOWN`

`FIG5_SEASONAL_INPUT_STATUS=COMPONENT_ONLY__NO_EXTRA_CHANNELS_SHOWN`

This graphical evidence cannot resolve whether a component representation is one channel or a multivariate tensor. It must not be used to silently choose either real-data interpretation.
