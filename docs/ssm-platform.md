# Simulation-first SSM platform

> Superseded by ADR-0003 on 2026-09-27. This is historical documentation;
> commands and SSM RTL below are retired and no longer available. User data and
> checkpoints are preserved. Use `docs/local-llm-platform.md` for the active path.

This implementation is a **research prototype**, not a trained general chatbot
or a board-ready deployment. It adds an end-to-end language-token simulation path
alongside the existing INT8 dense accelerator. The dense numeric contract is
unchanged. Different SSM families share an operator compiler and engine.

## What is implemented

- Trainable diagonal gated SSM and unfused, portable Mamba-1 in PyTorch.
- Local native Safetensors checkpoints; a restricted local Hugging Face Mamba-1
  importer. No remote model code, pickle imports or automatic downloads.
- Byte tokenizer (256 bytes + BOS/EOS/PAD). Imported `tokenizer.json` support.
- Integrity-checked `.mssm` artifact, topology/tensor validation and model hashes.
- Fixed-point compiler for embedding lookup, RMSNorm, projections, causal
  depthwise convolution, recurrent/selective state updates, gating, residuals,
  final normalization and tied vocabulary projection.
- Autonomous operator programs executing on `ssm_operator_engine`. State and
  causal convolution history remain resident between token steps.
- Host greedy/top-k sampling, deterministic seeds, prompt prefill and decode.
- Full-token RTL replay checking **all logits and all recurrent/history words**
  against a separate arbitrary-precision integer reference.
- Compiled 1/2/4/8/16-lane personalities, runtime active-lane control, partial
  tiles, instruction bounds/overlap checks, saturation and execution counters.
- Search among bit-exact RTL-validated candidates with explicit switching
  overhead, remaining horizon, a 5% margin and residence guard.
- SSM Double DQN research training/evaluation with replay, target network,
  legal-action masks, independent held-out model hashes and five evaluation seeds.
- Local training with AdamW, checkpoint/resume, dataset split provenance and
  quantized-versus-floating quality checks. Training uses only the training split.
- SQLite/content-addressed experiment history; an offline loopback web IDE with
  model selection, prompts, jobs, logs, cancel and training resume.
- Automatic Quartus project snapshots/builds, cache keys, report extraction and
  bounded lower-clock retries. No board programming is performed.

## Quick start

Use Python 3.10+ (3.12 tested), Icarus, Verilator and Yosys:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[ssm,ide,test]'
make verify-ssm PYTHON=.venv/bin/python
.venv/bin/python -m malleable.ssm.cli init --width 16 --layers 2
.venv/bin/python -m malleable.ssm.cli export
.venv/bin/python -m malleable.ssm.cli generate --prompt 'Once upon a time' --max-new 8
.venv/bin/python -m malleable.ide --model-root build
```

Open `http://127.0.0.1:8765`. The default model is **random/untrained**; nonsense
output is expected. These commands demonstrate the complete arithmetic path,
not language quality. No Ollama installation is needed for these native SSMs.

The first IDE shell uses offline HTML/JavaScript, not the larger planned
React/TypeScript application. Only files under `--model-root` may be selected.
The server binds to loopback and rejects foreign origins/hosts. It executes an
allowlist of Python commands without a shell. Two workers allow a compilation
or training job to run while another experiment executes. Jobs persist across
restarts; interrupted training can resume from a completed-step checkpoint.

### Train from local licensed text

Separate documents with blank lines. Register provenance before training:

```sh
malleable-ssm dataset --text corpus.txt --source SOURCE --revision PINNED_REVISION --license LICENSE
malleable-ssm train --dataset build/dataset.json --steps 100 --device cuda
malleable-ssm train --dataset build/dataset.json --steps 100 --device cuda --resume
malleable-ssm export
malleable-ssm quality --dataset build/dataset.json --split held-out
```

Document hashes are deduplicated and split deterministically; split overlap
and altered text are rejected. CUDA requires a CUDA-enabled PyTorch installation
on the NVIDIA machine; CPU is supported for smoke tests. Public TinyStories and
Dolly ingestion, a trained 4096-entry BPE, large-scale pretraining, instruction
fine-tuning and the proposed 100-prompt chatbot gate remain later work. A tiny
training smoke test does not satisfy those language-quality requirements.

`quality` checks float/quantized artifact identity, evaluates held-out next-token
NLL and top-1 agreement, and requires ≤5% NLL degradation, ≥90% agreement and
zero observed saturation for its quantization gate. It **does not** establish
usefulness or instruction following. Models and policies are not auto-promoted.

### Local models and GPU comparison

```sh
malleable-ssm import-hf --source /local/hf-mamba1 --model build/imported
malleable-ssm export --model build/imported --artifact build/imported.mssm
malleable-ssm gpu --model build/imported --device cuda --prompt 'Hello' --max-new 32
```

The importer supports local unsharded `model.safetensors`, tied embeddings/head,
SiLU, bias-free input/output projections, convolution bias, standard time-step
rank and RMSNorm epsilon 1e-5. Other architectures/settings are rejected. Current
schema capacity limits are width 512, 32 layers, state 64, expansion 4 and
vocabulary 65536; these are software validation limits, **not a fit claim**.
Models beyond these limits and sharded imports need a future streaming importer.

The GPU baseline uses ordinary unfused FP32 PyTorch operations and synchronized
host timing. It is not a state-of-the-art optimized GPU kernel. FP32 and fixed
point differ; compare quality and arithmetic as well as times. RTL cycles,
assumed-clock estimates and physical GPU wall time are different evidence types.

## Prototype numeric contract

The new engine stores signed **16-bit Q2.14** values in signed 32-bit memory
words. Its range is `[-2, 2 - 2^-14]`. This is deliberately a separately named
`q14-prototype-v1` contract, **not the planned INT4/INT8 production quantizer**.

Matrix rows use signed 64-bit dot sums, then shift 14 with round-to-nearest,
ties away from zero, add Q14 bias and saturate to INT16. A maximum length of
4096 and INT16 operands keep these sums within INT64. Vector multiplication
uses the same rounding; addition and copy saturate/preserve respectively.
Saturation is counted, never silently described as full-quality preservation.

RMS normalization computes:

```
root = max(1, isqrt(sum(x[i]^2) // length + epsilon_q14 * 16384))
y[i] = sat16(round_away(x[i] * 16384 / root))
```

The compiler uses `epsilon_q14=1`. Learned normalization scales are a following
elementwise multiply. The integer square root floors; division rounds nearest
with ties away from zero. Combinational division/square root are synthesizable
but are **not timing-optimized** for Cyclone V.

SiLU, sigmoid and softplus use exported 256-entry LUTs, indexed by
`clamp((x >> 8) + 128, 0, 255)`. Table spacing is 1/64 with represented inputs
`[-2, 127/64]`. Mamba-1 selective decay has a separate exported table for each
channel/state pair, indexed by quantized positive time step. State updates are
lowered to individually rounded multiplies/adds; they need not equal a fused
floating expression. All tables and numeric settings contribute to identity.

The Mamba compiler implements projected input/gate, causal convolution + SiLU,
input-dependent B/C/dt, softplus dt, selective decay and state recurrence, D skip,
SiLU gating, output projection and residual. This follows the Mamba-1 operator
structure; quantization is approximate and must pass quality checks. Sources:
[Mamba-1 reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py),
[selective scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py).

## Program and artifact interfaces

Instruction words are `[opcode, dst, a, b, length, aux, shift, bias_address]`.
Opcodes: 0 DOT, 1 ADD, 2 MUL, 3 LUT, 4 RMS, 5 COPY, 255 END. Destinations must
not overlap sources. Length is 1..4096; shifts 0..62. Compiler-generated programs
are independently validated before loading; the engine rejects invalid commands
when fetched and sets sticky `config_error`. This is not transactional rollback
of earlier valid instructions in a malformed manually supplied program.

`cfg_kind=0` writes data, `1` program words, `2/address=0` active lanes. Writes are
accepted only idle. Busy writes/start commands and invalid result reads flag
errors. Reset aborts execution and clears counters/errors but does not erase RAM;
the loader always reloads a full initialized artifact after reset. Read data is
synchronous with one-cycle `read_valid`. `done` is one cycle.

Each instruction has one fetch cycle. DOT needs `ceil(length/active_lanes)`
cycles, ADD/MUL/COPY need `length`, LUT and RMS need `2*length`. END has one fetch
cycle. Total job cycles freeze idle and reset at each accepted start. Useful MACs,
data-memory reads/writes and saturations are per job; configuration writes are
since reset. LUT reads do not count as MACs. These are logical operations, **not
physical DSP occupancy**. Memory is ideal, resident inferred storage with multiple
read ports; no DDR bandwidth or bank-conflict timing model is implemented yet.

`.mssm` is little endian: 8-byte magic/version, metadata/data lengths, SHA256,
metadata/data CRC32, canonical JSON metadata and signed INT32 tensor data. Tensor
offset/length entries form the versioned section index. No pickle is used.

## Optimization and learning

```sh
malleable-ssm optimize --horizon 1000 --switch-cost '{"drain":0,"program":100000,"warmup":1000}'
malleable-ssm policy-train --artifacts build/train-a.mssm build/train-b.mssm --episodes 100 --switch-cost '{"program":100000}'
malleable-ssm policy-evaluate --checkpoint HASH --artifacts build/unseen.mssm --rtl
```

Candidate search requires bit-exact RTL evidence before selection, compares the
best current-personality plan, and includes model/program reload and explicit
switch scenario costs. Host transfer is an explicitly assumed two-clock/write
or result read programming interface, not UART/DMA bandwidth. Zero programming
cost is labeled idealized. Compilation is a separately recorded development
cost, not asserted to be FPGA programming time.

Latency-first minimizes average completion time of a sequential token stream;
throughput-first minimizes elapsed cycles/token. Energy-first search requires
explicit credible watts per personality and a supplied clock; without it the
objective is disabled. SSM RL currently supports latency/throughput only and
uses estimated instruction-timing rewards. Counter comparison is evidence of
lane scaling, not a definitive causal memory/compute diagnosis. External-memory
and physical power/resource metrics remain unavailable.

SSM RL checkpoints have a separate action/state schema from dense policies. It
is candidate research training, not continual deployment. Evaluation refuses
training model identities, uses at least five independent seeds, and compares
fixed, heuristic, budget-one random and exhaustive baselines. Optional RTL
checks are stored separately from estimated rewards; negative results remain
visible. Elastic model modes, hardware-aware distillation, automatic promotion,
rollback for SSM deployment and a learned SSM performance predictor remain work.

## Automatic Quartus compilation

```sh
malleable-ssm quartus --lanes 8
```

The default characterization target is the **C5G Cyclone V GX
`5CGXFC5C6F27C7`**, not the older dense-project SoC target. Device selection is
independent of simulation and can be changed in the personality record. Do not
interpret this target as evidence that the model or engine fits.

The generator snapshots RTL and produces QPF/QSF/SDC, flow Tcl and timing Tcl.
Cache identity includes source, device, capacities, lanes, clock, constraints,
generator and tool versions. Builds invoke Quartus without a shell. Reports must
affirm successful fit and nonnegative setup **and hold** slack before acceptance;
missing evidence fails closed. Timing extraction uses the official
[TimeQuest Tcl interface](https://resources.altera.com/quartushelp/17.0/tafs/tafs/tcl_pkg_sta_ver_1.0_cmd_get_path_info.htm).
Timing failures may retry in 5 MHz steps to 75 MHz; missing tool, fit failure or
missing timing evidence does not trigger unbounded retries.

Pins are virtual and no external interface is routed. Thus even an accepted
characterization result is **not deployable**. There is no JTAG/flash invocation.
Power numbers remain unavailable without validated switching activity/calibration.
On macOS without Quartus, requests are `pending-tool`; generated files can be
transferred to a Windows/Linux Quartus host and the same command rerun there.
A remote Windows job agent, authenticated transport, actual board pinout,
SRAM/UART/DMA controllers, fitting validation, power-activity flow and physical
programming are deferred. No physical Quartus run was performed on this Mac.

## Validation and remaining milestones

`make verify-ssm PYTHON=.venv/bin/python` requires optional dependencies and runs
existing dense regressions, SSM self-checking controls, 200 randomized mixed
operator cases, all 31 compiled/runtime lane combinations, complete Mamba token
sequences, integer/artifact checks, training/resume, quality smoke tests, policy
leakage guards, IDE API security and Quartus pending/report fixtures. Verilator
lint and Yosys structural checks cover all personalities. Structural synthesis
does not establish FPGA fit, achievable clock or useful chatbot quality.

The implementation therefore completes the **simulation/operator foundation**,
not every milestone of the larger proposal. Still outstanding: production
INT4/INT8 group calibration and elastic modes, training/data/quality milestones,
timing-efficient banked BRAM and nonlinear units, real external-memory timing,
hardware-aware retraining/controller integration, advanced benchmark coverage,
production frontend/remote worker and physical validation. These limitations
are kept explicit so optimization cannot silently rely on unavailable actions.
