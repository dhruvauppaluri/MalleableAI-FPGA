# Qwen3 precision attribution: first ten cases

The authorized ten distinct, serialized cases completed successfully in 225.07
seconds on the same 16 frozen validation targets. Source was clean and unchanged
at `7118e064e08dae35b58746ef5048f0657284020c`. This is diagnostic emulation,
not an executable ISA/RTL candidate or release approval. No held-out data was used.

## Results

Each bypass retains floating values for the named group and keeps other groups
quantized. Context is 128; baseline quantization uses INT8 and group size 128.
Negative NLL change means lower target loss, not increased top-token agreement.

| Case | Floating bypass | Matches | Agreement | NLL change vs FP32 |
| --- | --- | --- | --- | --- |
| 0 | None (baseline) | 13/16 | 81.25% | +0.579% |
| 1 | All groups (control) | 16/16 | 100% | approximately 0% |
| 2 | Transformer weights | 14/16 | 87.50% | +0.062% |
| 3 | Projection inputs | 13/16 | 81.25% | +1.519% |
| 4 | Key cache | 14/16 | 87.50% | -0.885% |
| 5 | Value cache | 12/16 | 75.00% | +0.097% |
| 6 | Attention queries/probabilities | 12/16 | 75.00% | +0.441% |
| 7 | Output-head weights | 13/16 | 81.25% | +0.689% |
| 8 | Output-head inputs | 13/16 | 81.25% | +0.665% |
| 9 | Key cache + transformer weights | 14/16 | 87.50% | -1.070% |

The final pair was chosen by the predeclared ranking: agreement descending,
candidate NLL ascending, then fixed group order. It was not selected on held-out.

## Interpretation and limits

Both floating controls agree with the original FP32 reference on every target.
The all-INT8 emulation reproduces the original ISA mismatch positions 0, 7 and
10. Its NLL change is +0.579%, versus the earlier ISA diagnostic's +1.01%;
top-token parity is not full-logit equivalence. Keep these measurements separate.

Transformer-weight bypass fixes position 0. Key-cache bypass fixes positions 7
and 10 but introduces a position-4 mismatch, where the FP32 top-two margin is
only 0.0331. Their combined bypass fixes all three original mismatches but
introduces positions 1 and 14. More precision does not improve agreement
monotonically in these interacting quantized paths. The combined case's lower
NLL therefore does not satisfy the agreement gate.

Sixteen targets give 6.25 percentage points per match; at least 15 matches would
exceed 90% on this panel. No targeted case achieved that. The all-floating
control is an upper-bound diagnostic, not a deployment proposal. This small,
adaptively explored panel cannot establish generalization or release acceptance.
The implementation remains unchanged for production arithmetic, and release
acceptance remains blocked.

## Durable evidence and verification

Attempt root, relative to the Linux checkout:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-attribution-01/`.
All ten result hashes were independently recomputed and matched their summaries.
`report.json` SHA-256:
`8b28603e071b3d33593c36baf7f4742c9d86f48e06e71d69eec9b7da9cc0c348`.
Each case retains policy, token-level reference/candidate margins and NLL,
operator differences, source identity, canonical record ID, events and worker
log. The attempt also retains frozen 16/128-target panels and reference records.
There are no failed cases. A NumPy sigmoid exponential-overflow warning occurred
in the pinned reference; recorded token metrics remained finite and the floating
control matched all targets. Do not silently change the pinned arithmetic.

Focused CPU regressions: 31 passed in 10.44 seconds across `test_precision.py`,
`test_diagnostics.py`, and `test_llm.py`. These verify exact INT8 emulation
preservation, floating-reference agreement, isolated bypasses, unchanged input
weights, policy validation and causal panel integrity. The prior full verification
belongs to `b1f2d3f`; it must not be represented as final-source verification of
this newer diagnostic implementation.

## Next action, requiring the next batch selection

Use the already frozen 128-target spread panel before choosing a production
change. Run baseline/all-floating controls and the same seven single bypasses
plus this fixed combined bypass on it, serialized. Preserve full causal prefixes
and record its executed-token count; the 16-target timing alone is not a reliable
estimate for the larger panel. Start with a completed 128-target baseline pilot
to estimate the rest after the user chooses the next batch mode.

Then complete quantize-only and pairwise attribution as needed by the approved
recovery plan; fit compatible channel-rescaling/clipping candidates on calibration
only. Confirm improvements through actual ISA validation before any promotion.
Do not jump directly to higher-precision RTL or held-out testing from this panel.
The ten-case authorization ended with its report. The subsequently authorized
128-target baseline pilot is recorded below; additional cases still need a batch
mode selection.

## Follow-up: 128-target baseline pilot

User authorized one Interactive pilot. The frozen spread panel was reused without
alteration: eight sequences, 128 scored targets, 996 executed context tokens.
Source `c352607e3c9394231491fa1c4a30e9d098f3b04f` stayed clean and unchanged.
Both floating references agreed on all 128 targets. INT8 emulation matched
107/128 (83.59375%); FP32 NLL was 4.79491632, candidate NLL 4.82474171,
degradation +0.622021%. Agreement remains below 90%, requiring at least 116/128
matches on this diagnostic panel. This is not held-out acceptance.

Wall time: 281.44 seconds; candidate time: 256.52 seconds. Estimate 35–45 minutes
for the remaining nine serialized cases, allowing for variation and reference
reuse. Next batch should use the same frozen panel and fixed combined bypass;
do not select a new pair from held-out data or silently change the panel.

Evidence: sibling attempt `precision-128-pilot-01/`, including saved runner,
job/source identity, panel, policy, events, result and summary. Result SHA-256:
`994db61a623d52ce3a3ddf6cf792c38ead207ebf2bc93764bdfe24e2c557d454`.
Canonical record ID:
`213b8e7d0326d9c5d86f8649c4b0c78e2735f4602cf07365f576220f17320260`.
No production arithmetic was changed. No further checkpoint job is active.

## Completed nine-case continuation (inspected 2026-09-29)

The previously dispatched batch continued to completion while the chat was
interrupted. It ran on clean source `6975b1c5e5c2c5ae5b832224297ff44412753ab6`,
using the exact frozen panel and copied reference store. It took 2167.79 seconds
(36 minutes 8 seconds). All nine saved result hashes verified. No worker remains
active, and no cases need rerunning.

| Floating bypass | Matches | Agreement | NLL change vs FP32 |
| --- | --- | --- | --- |
| None (previous baseline) | 107/128 | 83.59% | +0.622% |
| All groups | 128/128 | 100% | approximately 0% |
| Transformer weights | 110/128 | 85.94% | +0.350% |
| Projection inputs | 111/128 | 86.72% | +0.801% |
| Key cache | 117/128 | 91.41% | -0.363% |
| Value cache | 108/128 | 84.38% | +0.389% |
| Attention queries/probabilities | 113/128 | 88.28% | +1.072% |
| Output-head weights | 107/128 | 83.59% | +0.614% |
| Output-head inputs | 108/128 | 84.38% | +0.604% |
| Key cache + transformer weights | 117/128 | 91.41% | -0.149% |

This broader panel changes the priority: key-cache quantization is the strongest
single-group lead. Bypassing transformer-weight quantization in addition adds no
agreement benefit here. A key-cache bypass meets the numerical thresholds on
this diagnostic panel, but it uses floating values in independent emulation.
It has no corresponding production ISA/RTL implementation or held-out approval.
It does not establish a passing release or guarantee full-validation success.
Follow the approved compatible INT8 search first, informed by this finding;
targeted cache precision is the measured fallback for localization and an ADR.

Attempt: `precision-128-remaining-01/` beside the pilot. Report SHA-256:
`31134ec2dbd76c6dee9f747e871b551aa96a0676d3ada0a8ba44a5a8ec21a0c7`.
The earlier lack of a chat completion message was a handoff problem, not a failed
job. These measurements run on CPU; GPU access is required later for CUDA/hybrid.
