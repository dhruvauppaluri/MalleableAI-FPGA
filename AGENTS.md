# Agent guidance

Cursor, Codex, and Claude Code all read this file. It is the shared project
memory. Chat history is not shared. Leave decisions, status, and code in the
repo.

`docs/STATUS.md` is the handoff. Read it before starting, and update it before
you stop.

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
