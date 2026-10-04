# Approved Qwen3 quality recovery plan

Approved by the user on 2026-09-28: staged redesign, compatible INT8 changes
first, then targeted higher precision where measurements justify it. Prefer
the smallest passing change; a measured slowdown is acceptable. This approval
supersedes the earlier exclusion of LLM precision redesign, but does not change
the independent dense numeric contract or permit lowering quality thresholds.

## Starting point and acceptance

Checkout: `/home/dhruv/projects/MalleableAI-FPGA`, branch
`codex/local-llm-platform`, baseline source `b1f2d3fa41ede030681ece3f520e7ca104317ca5`.
That source passed clean checkpoint-free verification, isolated browser checks,
12 actual CUDA cache cases and current-tip CI. Draft PR #3 remains unmerged.
GPU: RTX 5070 Ti Laptop, Windows driver 616.92; use G-Helper Standard mode.

The first 16 Qwen3 validation targets give 81.25% agreement and 1.01% NLL
degradation. All 311 converted tensors match FP32, both floating references
agree on every top token, and independent quantized emulation reproduces the
ISA's three flips. This is a diagnostic, not release approval. Historical
validation/held-out agreements (84.67%/83.50%) also fail. Preserve all records.

Evidence root: `build/zephyrus-jobs/release/20260928T223413Z-b463e746/`.
Diagnostic ID: `80857850eb3479cfc0f040ec05e543b873cb9ac658fa2d3a08abf691bfa87348`.
See `diagnostic.json`, `summary.json`, `input-audit.json`,
`final-verification/verification.json`, `browser-acceptance.json` and
`acceptance-disposition.json`. No official Qwen3-1.7B verifier was downloaded.

Target Qwen3-0.6B first, context 128/balanced. A final candidate requires >=90%
next-token agreement AND <=5% NLL degradation on the unchanged frozen held-out
suite with >=1,024 targets. Then evaluate Qwen3.5/LFM2.5 independently. Original
checkpoints stay intact under `build/models`; Qwen3 suite is
`build/models/quality/qwen3-frozen-v2.json`.

## 1. Attribution before redesign

Add validation-only independent controls for seven groups: transformer weights;
projection input activations (including attention-output/MLP-down); key cache;
value cache; attention queries/probabilities; output-head weights; output-head
inputs. Preserve the 16-target pilot as a regression case. Freeze a deterministic
128-target screening selection spread across validation sequences, with complete
causal prefixes. Record selected sequence/position IDs and token hashes before
candidate evaluation; do not regenerate suites from documentation.

Run baseline and all-floating controls, then bypass-only and quantize-only for
each group. Test pairwise bypasses among the three groups with largest recovered
agreement (NLL improvement breaks ties). Record top tokens/margins, target NLL,
clipping/outlier statistics and operator differences. Residual error alone is
not causal attribution. Floating bypasses are diagnostic-only and never selectable.

## 2. Compatible INT8 improvements

Fit statistics on calibration only; select on validation only. Test compensated
channel rescaling, preserving the unquantized operation, with strengths baseline,
0.25, 0.5, 0.75. Test weight clipping none/99.99th/99.9th percentiles on implicated
groups. Verify unquantized equivalence, shared consumers and tied embedding/head
handling. Preserve source checkpoints; derived tensors/calibration parameters
are separate hashed artifacts. Keep existing INT8 runtime quantization/storage.
Method reference: https://arxiv.org/abs/2211.10438 (SmoothQuant); no model-specific
quality guarantee is assumed.

Screen on 128 targets; send the best three candidates to complete validation.
Aim for >=92% validation agreement as a buffer, retaining the actual 90% gate.

## 3. Targeted precision redesign

If compatible INT8 candidates fail, use attribution to choose the next change.
For weight/input error, emulate 64/32-element quantization groups. For cache,
attention or head error, retain higher precision only in implicated groups,
then localize to operators/layers. Prefer existing FP32 primitives where suitable.
Smaller groups are architectural: D couples matrix depth and quantization, and
the current AXI path assumes D=128. Verify layout/scales/AXI/ISA/RTL together;
never silently use another memory backend to claim equivalent performance.

Rank passing candidates by unchanged ISA/RTL first, fewest changed operator
groups second, then memory footprint and measured cycles. No fixed slowdown cap
was imposed. Write a numeric-contract ADR before implementing changed arithmetic,
rounding or storage. Dense RTL/numerics remain independent and unchanged.

## 4. Prove the actual implementation

Match independent emulation, ISA/compiler and RTL. Test zeros, outliers, rounding,
cache append/read, partial blocks, tied tensors and context boundaries. Require
bit-exact DRAM/TMEM checks on affected operators and tiny-model runs. Run complete
validation through actual ISA, freeze code/weights/calibration/precision/config
identities, then evaluate the candidate once on held-out. On failure preserve
the result and stop promotion; do not repeatedly tune against held-out.

A passing candidate then needs short-prompt + eight greedy tokens entirely in
RTL, checked at every executed step, plus fresh durable 100-token synthetic
evidence. Add a versioned precision-policy artifact to all affected quality,
generation, benchmark and release identities. Preserve legacy baseline identities.
Mixed precision must be named accurately and explicitly supported by release
checks, never mislabeled as the old INT8/head-INT8 candidate.

## Execution and handoff rules

Choose Interactive or Overnight in chat before EACH heavy batch. Interactive
runs one short diagnostic/estimated short job then reports. Overnight stops
dispatching after eight hours and lets the current job finish. Serialize heavy
jobs and estimate using completed pilots (old 16-target pilot: 75 seconds).
Run CPU unit checks freely as implementation verification; no checkpoint job
starts until that batch's mode is selected. Check no other heavy worker is active.

Use fresh append-only attempt folders and preserve failures/canceled Mac queues.
No training/distillation, changed test thresholds, altered held-out suites, GPU
substitution for RTL acceptance, AWS, physical FPGA deployment or automatic merge.
Official CUDA/hybrid remains gated on all three standalone models passing.

Update `docs/STATUS.md` before stopping: source commit/dirty state, current task,
tests, exact commands, running PID/job/attempt path, outcome and exact next step.
If interrupted or credits run out, another agent must inspect active jobs and
artifacts before resuming; never duplicate dispatch. A failed bounded search
ends with a measured precision/quality frontier, not a claimed release pass.
