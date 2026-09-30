# Agent guidance

## Current handoff: Qwen3.5 reference fixed; raw INT8 pilot passes

Codex fixed a confirmed quality-reference defect in ba9a988: constructing the
Qwen3.5 Transformers text tower on the meta device did not restore non-persistent
rotary-frequency buffers from checkpoint weights. Rebuild the rotary embedding
from the unchanged configuration after assigning checkpoint tensors. The position
IDs were correct, but observed sine tables were effectively zero. No ISA/RTL,
matrix/head precision, checkpoint, suite or threshold was changed.
Qwen3.5 reference cache contract is now original-local-safetensors-qwen35-rope-restored-v2;
old Qwen3.5 cache entries must not be reused. Qwen3 and LFM identities are unchanged.
Historical Qwen3.5 78.125% quality and reference-divergence records remain preserved
but used a defective reference and do not describe corrected-reference acceptance.

qwen35-corrected-int8-pilot-01 completed in 329.55 seconds on clean ba9a988:
128 frozen spread-validation targets with complete causal prefixes, raw INT8
matrices/head, balanced emulation: 125/128 = 97.65625% agreement; NLL -0.19908%.
Independent FP32 vs corrected Transformers: 128/128 top tokens, relative L2
1.08e-6 to 1.61e-6. This is diagnostic emulation evidence, NOT actual-ISA or RTL
acceptance. Fourteen focused reference/Qwen3.5 regressions passed before dispatch.
Evidence is under continuation-20260930/qwen35-corrected-int8-pilot-01 (result,
checksummed summary, job/source/PID, runner and fresh reference store). The preceding
qwen35-rope-localization-01 preserves bad-table and matched-input rotation evidence.

No job is running. User was asked to select the next actual-ISA batch mode;
do not revive the revoked Overnight window. Next: short corrected-reference actual
ISA pilot for timing, then unchanged 1024-target full validation of RAW INT8. If
full validation passes, freeze source/config/reference design and perform the
untouched held-out split once via persistent registry, then eight-token full RTL.
Do not launch calibration or precision redesign unless corrected ISA evidence
requires it. Never repeat completed Qwen3/LFM held-out or LFM benchmarks.

## Latest completed Interactive probe — Qwen3.5 attention

User authorized one short attention diagnostic with “go for it”. Codex completed
qwen35-attention-localization-01 in 14.12 seconds on clean source 57edbb3.
16 validation targets from sequence 1 of the existing spread panel were evaluated
with complete causal prefixes. No held-out evaluation or production changes.
Synthetic observation check passed: exact original logits, all capture sites present.

Relative L2 differences before rotary encoding: projection input 9.50e-7,
normalized query 8.88e-7, normalized key 8.73e-7. After rotary encoding:
query 37.75%, key 29.58%; attention probabilities 22.01%.
The first major discrepancy is positional encoding, not projection or normalization.
Loaded rotary parameters agree (theta 1e7, 64 rotary dimensions); the visible
rotation formulas use the same rotate-half convention. Root cause is NOT yet proven.
Next bounded diagnostic: capture actual position IDs and cosine/sine tables and
apply both rotations to identical normalized q/k inputs; check broadcasting,
position layout, mutation/aliasing and table generation before changing arithmetic.
Do not conclude that quantization alone causes the full 78.125% validation failure.

Evidence directory:
build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/qwen35-attention-localization-01/
Contains result.json, checksummed summary.json, job.json, runner.py and the exact
instrumentation.py. Instrumentation is diagnostic-only and not production code.
No worker remains active. Interactive batch complete; choose mode before another
checkpoint job. Overnight authorization remains revoked. Release remains blocked.

## Current mode override — Interactive (2026-09-30)

The user's latest instruction supersedes the eight-hour Overnight authorization.
One bounded Interactive analysis completed; no further checkpoint dispatch is
authorized by the old window. Select a new run mode in chat before the next batch.
The original control/deadline remains historical, with an append-only Interactive
override in continuation-20260930/mode-overrides. Never restart the Overnight loop.

The saved 128-target reference-spread run completed on source 61ac769 in 50.37s.
Independent floating/reference top-token agreement is 119/128 (92.97%); this is
diagnostic evidence, not INT8 acceptance. Interactive analysis verified its hash
and found the first residual error above 0.1% at layer 3's attention output in
all eight sequences. Earlier relative L2 error is about 1e-6; layer 3 is 6.3–9.1%.
This localizes the discrepancy but does not establish its cause.

Evidence: build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/
interactive-localization-audit-20260930T223446Z/result.json.
No new inference or held-out evaluation was performed by this Interactive audit.
Next: instrument the first full-attention layer's q/k normalization, RoPE, scores,
and attention outputs with matched inputs before attributing loss to quantization.
Qwen3.5 full INT8 validation remains 78.125%; release acceptance stays blocked.
No release worker is active. All older run authorizations below are historical.

Cursor, Codex, and Claude Code all read this file. It is the shared project
memory. Chat history is not shared. Leave decisions, status, and code in the
repo.

`docs/STATUS.md` is the handoff. Read it before starting, and update it before
you stop.

## Authoritative continuation (2026-09-30)

Read [the complete approved continuation](docs/release-continuation-20260930.md)
and the current opening of docs/STATUS.md before acting. They supersede earlier
handoff next-actions and expired run windows below. The user approved one NEW
eight-hour Overnight window, starting at the first new checkpoint dispatch after
safeguard implementation/tests. Persist and reuse one deadline across retries and
agent changes. Targeted Qwen3.5 activation/cache/operator redesign is authorized;
INT8 matrices and INT8 output head, dense contract, frozen suites and >=90% / <=5%
gates remain mandatory. Do not repeat completed Qwen3/LFM balanced held-out or
LFM's ten benchmarks. Inspect workers before dispatch; serialize heavy jobs.
While source-bound workers run, record handoffs in their ignored attempt folders,
not tracked files. Do not extend the window or merge PRs. Release is still blocked.

## Project

MalleableAI-FPGA is a model-aware FPGA AI accelerator. The FPGA runs inference.
Training, quantization, hardware selection, and FPGA compilation happen on the
host. The active product is the local pretrained-LLM platform (ADR-0003).
Standalone full-RTL acceptance precedes enabling CUDA and greedy hybrid runs.
AWS provisioning, uploads, AFIs, HBM and physical programming remain excluded.

The verified dense RTL foundation and independently implemented model-aware
software baseline are in this repo. Pinned OpenTPU is a separate execution and
numeric boundary; do not reinterpret the dense contract. ADR-0002 and the SSM
guide are superseded historical documents. See `docs/local-llm-platform.md` and
`docs/STATUS.md` for implemented capabilities and gates; simulation is not
physical-board fit or CUDA performance evidence.

Do not search repository history or external branches for a software
implementation. Design it from `docs/ml-contributor-roadmap.md`,
`docs/numeric-contract.md`, and the observable RTL interfaces.

## Shared rules

- Preserve the signed INT8 and INT32 behavior in `docs/numeric-contract.md`.
- Do not modify verified RTL merely to make a software result pass.
- Treat `rtl/int8_mac.sv` and `rtl/int8_dot_product.sv` as the current hardware
  source of truth, along with the later verified stages:
  `rtl/int8_tiled_accumulator.sv`, `rtl/int8_postprocess.sv`, and
  `rtl/malleable_accelerator_top.sv`.
- Record a numeric or interface change in `docs/numeric-contract.md` or a new
  file under `docs/adr/` before code depends on it.
- Run `make test` before proposing changes that affect the hardware boundary.
  Run `make verify` when the change can affect lint or synthesis.
- Start software work with the smallest complete dense-network pipeline. Keep
  numeric behavior explicit, documented, deterministic, and tested.

## Tasks

Cursor, Codex, and Claude are interchangeable. Any of them may pick up any open
task in `docs/STATUS.md`. Nothing is reserved for one tool.

Claim a task before starting so two agents do not do the same work:

1. Read this file and `docs/STATUS.md`.
2. Choose any row whose status is `open`. Prefer a task whose dependencies are
   already done.
3. Set that row to `in progress` and put your name on it, then start.
4. Put the result in git: code, a doc, or both. Open a PR for anything that
   should be reviewed.
5. Set the row to `done`, add the PR, and note anything newly unblocked. If you
   stop early, set it back to `open` and write what is left.

The next agent starts from `docs/STATUS.md`, not from your chat.

## Commands

```sh
make test      # Icarus Verilog self-checking simulations
make verify    # simulation, Verilator lint, and Yosys synthesis
make verify-llm PYTHON=.venv/bin/python # local + upstream + synthetic RTL + UI
```

Quartus Prime Lite 25.1, when installed:

```sh
quartus_sh --flow compile quartus/int8_mac/int8_mac
quartus_sh --flow compile quartus/int8_dot_product/int8_dot_product
```

The Quartus projects still cover the MAC and dot-product blocks. They do not
yet compile `malleable_accelerator_top`.

Keep user checkpoints/datasets, private media and unrelated `tmp/` files out of
commits. Local model execution must not download files or run remote Python.
The explicit checkpoint-download tool is opt-in only. Full-RTL prefill must not
be replaced with ISA work. Record measured/estimated/unavailable provenance.
Quality search uses validation; held-out promotion groups by base-model identity.
No automatic merge to `main`; deliver reviewed milestones.

## Approved quality recovery and continuation prompt (2026-09-28)

The active task is [Qwen3 quality recovery](docs/qwen3-quality-recovery.md).
The user approved staged LLM precision redesign: compatible INT8 improvements
first, then targeted higher precision and matching ISA/RTL changes if needed.
Prefer the smallest passing change and accept measured slowdown. This supersedes
the previous exclusion of LLM numeric redesign for this task; the independent
dense numeric contract, >=90% agreement and <=5% NLL gates remain unchanged.

Baseline source `b1f2d3fa41ede030681ece3f520e7ca104317ca5` passed checkpoint-free
verification, browser checks, actual CUDA cache tests and CI. Qwen3's 16-target
diagnostic reached 81.25% agreement/1.01% NLL degradation. All 311 tensors match
FP32; the floating references agree; independent quantized emulation reproduces
the ISA flips. Full release is blocked. Evidence root:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/`.
Original checkpoints and frozen suites remain in ignored `build/models`.

### Copy-ready prompt for Claude Code, Cursor or Codex

Continue the Qwen3 quality recovery in `/home/dhruv/projects/MalleableAI-FPGA`
on `codex/local-llm-platform`. Read AGENTS.md, docs/STATUS.md and
docs/qwen3-quality-recovery.md first. Inspect source, active processes and saved
attempts before repeating work. Target >=90% held-out next-token agreement and
<=5% NLL degradation. Start with quantization attribution, then calibration-only
INT8 improvements, then targeted precision redesign justified by measurements.
Preserve frozen suites, original checkpoints, historical failures/canceled jobs
and the independent dense numeric contract. Use calibration for fitting and
validation for selection; freeze the candidate before held-out evaluation.
Choose run mode with the user before each heavy batch and serialize jobs.
Keep exact commands, attempt paths/PIDs, source identity, tests, outcomes and the
next action in docs/STATUS.md so another agent can resume without chat history.
Never duplicate an active job, automatically merge, lower thresholds, or enable
official CUDA/hybrid acceptance before standalone passes. A diagnostic precision
bypass is not executable release evidence until actual ISA/RTL supports it.

## Latest local Cursor continuation (2026-09-29)

Use [the complete Cursor continuation prompt](docs/cursor-quality-continuation.md)
for the current handoff. It supersedes the older copy-ready prompt above when the
user submits it: that prompt explicitly selects Overnight mode for one eight-hour
dispatch window, including its prescribed follow-ups, so do not repeat mode or
candidate approval questions. It does not authorize automatic extension of the
window, threshold/suite changes, or merging PRs.

Latest actual ISA diagnostic: `candidate-isa-pilot-01`, 15/16 = 93.75% agreement,
NLL +0.634761%. This is a small validation diagnostic, not held-out acceptance.
The passing compatible INT8 candidate is
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/quality-recovery-pilot-01/int8/case-00`;
use the original Linux checkpoint, tokenizer and frozen suite. Full validation
was not started while preparing this handoff. Inspect existing jobs first and
never duplicate or overwrite an attempt.

Run the durable `tools/run_candidate_full_validation.py` launcher through the
documented user systemd command. Preserve the first dispatch deadline for all
follow-ups. Keep tracked source/docs unchanged while its source-bound worker
runs; save an early handoff as a new timestamped file inside the ignored attempt
directory, then update `docs/STATUS.md` after completion. Follow the detailed
prompt's pass/fail branches, including the conditional ten distinct precision
cases, production identity integration, frozen held-out evaluation and release
checks. Use LOCAL Cursor with WSL access on Zephyrus; a cloud prompt alone cannot
access the ignored checkpoints and evidence.
