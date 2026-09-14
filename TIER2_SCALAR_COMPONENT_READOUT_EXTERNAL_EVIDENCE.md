# Tier 2 Scalar Component Readout — External Methodology Evidence

STATUS=WEB_SUPPORTED_TASK_ADAPTATION_EVIDENCE
PAPER_EXACT=False
REAL_HF_VALUES_USED=False
REAL_NUMERICAL_TARGET_VALUES_USED=False
TEST_LOADED=False

## Evidence boundary

This record supports only PLAN-CHANGE-0016's disclosed Tier 2 task adapter. It does not state that the target Energy paper specifies `Linear(3,1)` or that the paper's branch architecture has the same readout.

## Target-paper topology

The project’s retained paper evidence establishes Trend, Seasonal, and Residual as separately predicted component streams before fusion. A Tier 2 branch supervised by one STL component therefore needs one scalar component prediction. The paper does not disclose the adapter from a three-variate intermediate output to that scalar.

`TARGET_PAPER_SCALAR_READOUT_STATUS=UNRESOLVED`

## Time-Series-Library forecasting task precedent

The official [Time-Series-Library data loader](https://raw.githubusercontent.com/thuml/Time-Series-Library/main/data_provider/data_loader.py) explicitly recognizes `M` and `MS` feature modes, retains all data columns for both, and uses a designated target field in its forecasting dataset contract. This establishes multivariate-input forecasting with a task-specific target selection as a normal library task mode; it does not determine this project’s physical component semantics.

`EVIDENCE_CLASS=OFFICIAL_LIBRARY_TASK_PRECEDENT`

## TimesNet output projection precedent

The official [Time-Series-Library TimesNet implementation](https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TimesNet.py) defines a forecasting projection `Linear(d_model, c_out, bias=True)` and returns a forecasting tensor shaped `[B, L, D]`. Thus a linear output projection is a normal output-stage design in the official reference implementation. PLAN-CHANGE-0016 uses a separate, smaller task adapter after the locally frozen core; it does not alter the core or claim to reproduce that reference projection exactly.

`EVIDENCE_CLASS=OFFICIAL_REFERENCE_ARCHITECTURE_ONLY`

## iTransformer output convention

The official [iTransformer implementation](https://raw.githubusercontent.com/thuml/iTransformer/main/model/iTransformer.py) documents input `B L N`, projects each variate token from `B N E` to `B N S`, and returns `B S N`. With three input variates, this naturally yields three forecast channels and does not itself define an external scalar component target head.

`EVIDENCE_CLASS=OFFICIAL_REFERENCE_ARCHITECTURE_ONLY`

## Result-blind conclusion

`TIER2_SCALAR_READOUT=LINEAR_3_TO_1` is a minimal, task-specific mapping from the frozen `[B,1,3]` learned intermediate output to one scalar component target. It is chosen before real numerical access, has no activation, and introduces four trainable parameters per branch. It remains a Tier 2 fallback rather than target-paper evidence.
