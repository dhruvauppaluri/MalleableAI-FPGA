# Policy correction execution, 2026-10-08

## Implemented milestone

The old predictor, DQN checkpoint, and published evidence are preserved.
`tools.audit_policy_features` verifies the original 30 rows and detects ten
cross-model feature collisions (20 distinct input vectors). No benchmark or
quality target was rerun. All previous held-out attempts remain consumed.

`tools.evaluate_relative_policy_baseline` fits personality/balanced median
ratios with each model excluded in turn. Nonbalanced speedup MAE is 0.10194,
0.88399, and 0.14501 percentage points for Qwen3, Qwen3.5, and LFM respectively.
These are exploratory reuse results, not independent test results or calibrated
intervals. Balanced rows are excluded from error averages to avoid trivial zeros.
Only three models and nine workloads exist; personality rows are paired,
not independent observations. This baseline does not estimate absolute cycles.

The new, separate conservative policy compares candidate upper objective cost
including transition overhead with current lower objective cost. It requires
eligibility, residence, platform/context lineage, all seven transition components,
and physical measurement provenance. Strict 5% hysteresis remains. It returns
recommendations only; automatic switching is always false. Its callers must
verify quality/capacity, image identity, interval calibration and measurement
artifacts. It is not wired to production and does not grant approval by itself.

A new compiler feature contract includes architecture, operation/instruction
counts, byte estimates, phase lengths and hardware parameters. Legacy checkpoint
inputs cannot satisfy it. Compiler extraction, training-only scaling, regression
fitting and independent interval calibration remain to be implemented; old
published rows do not contain all those features, so missing fields are not imputed.

## Fresh performance campaign: draft, not dispatched

Use a finite serialized pilot before determining a larger sample budget. Freeze
actual token tapes and hashes, model/variant/configuration/compiler/source IDs,
phase, context occupancy, batch/row count, memory parameters and split before
dispatch. Compare all four approved INT8 personalities on each matched tape.
Use new short/long prefill and decode workloads within the approved context128;
include memory pressure and alternating phase sequences. Compiler-supported
batch/row settings must be checked before including them. Exclude every original
tape identity and every prior consumed suite. Do not relabel old runs as new.

The campaign ledger starts with zero new claims. `policy_campaign.claim` provides
exclusive, fsynced local claims before dispatch; failures remain consumed and
renaming a manifest cannot reclaim an identical case in that registry. This is
a local primitive, not a complete distributed runner. Dispatch requires one
canonical registry and published claims, content-derived canonical cases and
an exclusion check against prior manifests. Those integrations remain open.

Predeclare training/validation/test by workload group and model family, with
personality pairs in the same split. All existing rows are development evidence.
Freeze estimator, calibration and controller before opening a new test split.
There is no quota of positive results. A pilot may establish that switching
does not pay. The previously proposed 100 rows/20 winners are not acceptance
requirements. A 1-point MAE target alone is insufficient when gains are smaller.
Report error, ranking, interval coverage, abstention and realized total cost
against fixed balanced, best fixed, deterministic and unconstrained oracle.

## Remaining execution gates

1. Extract features from exact compiler plans and collect new RTL performance
   groups. This host currently has no Verilator; no new RTL worker was launched.
2. Fit balanced runtime and paired relative-cost estimators. Calibrate intervals
   on separate groups; do not calibrate on test residuals.
3. Obtain physical transition samples keyed by source/destination image,
   platform and context/state compatibility: drain, program, reload, initialize,
   reprefill, warmup and controller overhead, including tail uncertainty.
   There is no physical FPGA attached here. Assumed costs cannot satisfy this gate.
4. Compare deterministic and contextual-bandit candidates when data supports
   profitable decisions. Preserve natural test distributions.
5. If sequential benefit warrants DQN, add cost/uncertainty observations,
   executed-action bootstrap masks, proposed/executed/rejection audit fields,
   and five independently trained seeds in a new checkpoint schema. The legacy
   DQN has not been retrained or corrected by this milestone.
6. Publish frozen independent evaluation exactly once, then request review.
   Quality campaigns are needed only for new configurations. No auto-promotion.

## Verification

Nine focused tests passed for feature completeness/architecture separation,
cost/horizon decisions, strict uncertainty margin,
eligibility, residence, stale context, assumed/invalid costs, immutable consumption
and draft rejection. These synthetic unit tests are not hardware measurements.
