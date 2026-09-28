# Zephyrus release implementation and remaining acceptance

The release is blocked. No standalone or full-release pass is claimed, and no
Qwen3-1.7B verifier has been downloaded. The existing numeric contracts and
quality thresholds remain unchanged.

## Qwen3 diagnosis

The chat-authorized interactive pilot used the first 16 validation targets from
the existing `qwen3-frozen-v2.json`, context 128, balanced INT8 matrices and an
INT8 output head. The frozen calibration/validation/held-out tapes were not
regenerated. All three transferred checkpoint revisions, download-manifest
hashes, tokenizer identities and frozen-suite identities matched their pins.

Evidence root:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746`.
Canonical diagnostic ID:
`80857850eb3479cfc0f040ec05e543b873cb9ac658fa2d3a08abf691bfa87348`.
The audit and diagnostic record bind the source commit plus diagnostic source
hashes, checkpoint files, tokenizer files and frozen split hashes.

| Check | Observed result |
| --- | --- |
| Converted weights versus original FP32 state | All 311 tensors equal; zero mismatches |
| Independent FP32 reference versus original model | 16/16 top tokens; max logit error 0.0000763 |
| ISA versus independent quantized emulation | 16/16 top tokens |
| Quantized paths versus original FP32 | 13/16 top tokens, 81.25% |
| Mean FP32 / ISA target NLL | 6.24093 / 6.30383; 1.01% degradation |
| Diagnostic inference time | 75.09 seconds |

Disagreements occur at positions 0, 7 and 10. Their FP32 top-token margins are
0.5761, 0.2844 and 0.3090. Quantized emulation produces the same top-token
disagreements without ISA kernels. The sampled evidence therefore points to
quantization rather than conversion or reference alignment. It does not prove
that all ISA logit differences are negligible: per-token ISA/emulation error
metrics and per-layer attention/MLP residual comparisons remain in the record.
Residual comparisons observe independent references, not physical RTL state.

This bounded validation diagnostic cannot approve a release. Historical Mac
held-out agreement is also below threshold (83.50%, NLL degradation 0.95%);
that historical record is preserved, not relabeled as fresh Zephyrus evidence.
No arithmetic change or full held-out search follows from the pilot.

The next precision investigation needs a separate scope decision. A proposed
scope is validation-only ablation of matrix/input/KV/probability/output-head
quantization, using the recorded layer differences to identify which precision
points move the top-token margins. Changing block size, activation/KV precision,
or operators requires a new numeric-boundary decision and independent RTL
regressions. It is not authorized by the present acceptance plan.

## Independent implementation

- `quality-diagnose --limit-targets 16 --split validation` is bounded to 1–128
  validation targets and always non-selectable. Ordinary quality evaluation and
  the 1,024-target release gates are unchanged.
- `performance-suite --run-index 0` runs one index. Reuse requires every manifest
  setting and microarchitecture identity to match. Attempts use unique folders;
  failures remain recorded. A complete report is emitted only after all ten
  indices have valid results. Whole-suite execution remains supported.
- Workbench indexed jobs freeze their manifests and share a content-addressed
  resume store; trace endpoints resolve recorded, bounded paths inside the new
  job root. Generation evidence explicitly records microarchitecture schema v1.
- Predictor schema v2 defines eleven ordered features, including FIFO depth and
  bandwidth. Fixed token tapes have zero generated-token work. Historical v1
  predictors remain inspectable but cannot silently supply new predictions.
- The predictor/policy release partition is Qwen3 and Qwen3.5 for fitting,
  LFM2.5 for held-out evaluation. Controller requests require four INT8
  personalities and both objectives over horizons 1, 8, 32 and 128. Policy
  reports retain per-episode regressions and label switch costs as assumptions.
- Recommendations and explicit application bind model, tokenizer, variant,
  personality, configuration and frozen suite to recomputed quality measurements.
  Application also requires matched held-out approval and an idle execution
  window. Chat history commits only successful completions; replay restores
  conversation and context settings. Reports expose benchmark and predictor
  evidence alongside quality and policy reports.
- NVIDIA access was restored by cycling G-Helper Eco → Standard. WSL NVIDIA-SMI
  reports the RTX 5070 Ti Laptop GPU and Windows driver 616.92. A fresh CUDA
  matrix computation passed. Keep GPU mode Standard during jobs; no Linux
  NVIDIA driver was installed.
- GPU/hybrid metadata uses NVIDIA-SMI, with explicit unavailable metadata on
  query failure. Versioned timings separate loading/build, prefill, decode and
  inference total, with synchronized CUDA boundaries. CUDA cache tests use a
  tiny synthetic model and require actual CUDA, never a CPU fallback.

## Verification and handoff

The final-source checkpoint-free verification is dispatched in interactive mode
after committing source. Its append-only reports are placed under the evidence
root above in `final-verification/`, with the source commit, unchanged-source
check and log hashes. The CUDA cache report is synthetic protocol evidence,
not official Qwen3-1.7B or full-RTL hybrid acceptance.

Browser testing uses `tools/serve_browser_acceptance.py` on a separate loopback
port with workers disabled. Fixture outputs are explicitly invalid for model
release; recorded Lens data references preserved historical tiny RTL traces.
Browser checks cover keyboard activation, durable replay after reload,
conversation restoration, cancellation, reports and Lens loading. API and reducer
regressions cover reconnect cursors, duplicates, stale events, restart retirement,
lineage rejection and explicit validated application.

Standalone manifest **schema v3** and full-release manifest **schema v2** are
separate interfaces. The full-release template now declares predictor reports,
actual CUDA cache verification, separate browser acceptance, final source
identity and synchronized timings. Browser and checkpoint-free verification
must both match the final source commit.
Checkers reject missing, failed or mismatched evidence; templates are not passes.

Still required after a scope decision resolves Qwen3 quality: all three fresh
validation/held-out approvals and eight-token full-RTL generation runs, fresh
100-token tiny RTL export, 30 indexed fixed-tape benchmark runs, measured
predictor/policy fitting and evaluation, official verifier download, matched
eight-token CUDA/full-RTL hybrid evidence, and both passing release manifests.
Keep PR #3 draft and leave merging to review. Preserve the historical SSM branch
and canceled Mac jobs. Every next heavy batch requires a run-mode choice in chat.
