# MalleableAI-FPGA

A model-aware FPGA AI accelerator platform: understand a model and workload,
compare supported hardware configurations, and learn whether changing to a
better configuration will repay its overhead.

The active project is the **local pretrained-LLM platform**. It uses pinned
OpenTPU for language-model execution while preserving our independently verified
dense RTL foundation. Simulation comes first, followed by a gated NVIDIA GPU
baseline and greedy speculative-decoding experiment. AWS integration comes last
and is excluded from this release.

Inference runs on the accelerator design. Analysis, quantization, learning,
hardware selection, and compilation run on the host. The custom SSM product
path is retired; historical records and user data are preserved.

## Where we are

The release is **incomplete**. The current controller status is in
[docs/STATUS.md](docs/STATUS.md), with the independent quality campaign in
[its evidence ledger](docs/evidence/controller-quality-continuation-20261006/README.md)
and the [offline predictor/controller audit](docs/evidence/controller-predictor-20261007/README.md).
The table below combines the recorded Mac baseline with the newer cloud evidence:

| Area | Current evidence |
| --- | --- |
| Dense RTL | Simulation, Verilator lint, and Yosys structural checks pass. |
| Checkpoint-free LLM suite | `make verify-llm` passed upstream, personality/format, host, and UI event tests plus the frontend build. |
| Tiny-model full RTL | Durable 100-token sequence: 100/100 steps bit-exact. |
| Qwen3-0.6B INT8 personalities | Published fixed-tape RTL rows and all four held-out quality approvals are reviewable. Compact, compute, and buffered passed a fresh independent exactly-once campaign; balanced retains its prior published approval. |
| Qwen3.5-0.8B / LFM2.5-230M INT8 personalities | All four personalities per model have published held-out quality approval, and their indexed RTL benchmark rows are preserved. |
| Workbench | Service and React UI exist; complete browser/accessibility acceptance remains outstanding. |
| Optimization / learning | All 30 indexed RTL benchmarks were verified without reruns. The Qwen-trained predictor has 109.85% LFM error; an offline learned-policy audit retained balanced and showed no superiority. Automatic switching stays gated. |
| CUDA / hybrid | Commands are gated; Zephyrus acceptance evidence remains outstanding. |

[docs/STATUS.md](docs/STATUS.md) is the authoritative handoff with evidence paths,
caveats, and open tasks. The current controller continuation is in
[draft PR #9](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/9), following
the [PR #8 Zephyrus handoff](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/8).
No automatic merge to `main`.

Newer Zephyrus work passed checkpoint-free verification, browser checks, CUDA
cache tests, and CI; this is not official CUDA/hybrid release acceptance.
The earlier frozen 128-target Qwen3 baseline pilot achieved 83.59% agreement
and 0.62% NLL degradation. It remains historical diagnostic evidence; the
later independent 1,024-target held-out approvals are linked above.
See [the approved recovery plan](docs/qwen3-quality-recovery.md) and
[precision attribution findings](docs/qwen3-precision-attribution-results.md).
The user-approved recovery scope permits targeted LLM precision and matching
ISA/RTL redesign after compatible INT8 experiments, accepting measured slowdown.
It does not change the independent dense contract or quality thresholds.

## What we are building

- Safe local configuration, tokenizer, and Safetensors loading, including indexed
  shards. No remote model Python, pickle imports, or implicit downloads.
- Compatibility reports for architecture, operations, dimensions, numeric
  requirements, memory capacity, and compiler/context limits. Unknown models
  remain inspect-only, not universally executable.
- Full Verilator RTL prompt processing and generation by default, with an
  explicitly selected ISA reference/alternative. No hidden ISA prefill.
- A live workbench for prompts, tokens, model inspection, experiment comparisons,
  quality reports, optimizer decisions, instructions, and bounded traces.
- Durable jobs, ordered/reconnectable SSE events, cancellation, searchable
  experiment storage, and content-addressed evidence with complete lineage.
- Recommendation-first optimization and periodically evaluated learning policies.
  Automatic application requires explicit opt-in and approved configurations.

Initial adapters target **Qwen3-0.6B, Qwen3.5-0.8B, and LFM2.5-230M**. Adapter
availability does not mean release acceptance has passed. GGUF/Ollama imports
and additional architectures are deferred.

OpenTPU is pinned at `15754e971b55591b91048c4c636023fe59b343e7` under
`third_party/opentpu`. Its FP32/vector and quantized-matrix numeric contract is
separate from our dense INT8/INT32 contract. Our orchestration/adapters remain
separate from reviewed upstream source.

## How malleability works now

The initial LLM search space uses separately compiled simulation personalities,
all with one slice and quantization block depth 128:

| Personality | Matrix columns | Vector lanes | FIFO depth |
| --- | ---: | ---: | ---: |
| Compact | 2 | 8 | 128 |
| Balanced | 4 | 8 | 512 |
| Compute | 8 | 16 | 512 |
| Buffered | 4 | 8 | 1,024 |

INT8 is the baseline. Supported INT4/FP4 variants retain an INT8 output head.
Compiler compatibility, capacity, correctness, and quality determine legal
candidates. These are not instantaneous runtime structural changes or physical
bitstream swaps. Incompatible state changes require reset and re-prefill.

Each decision asks:

1. Which validated configuration is best for this workload and objective within
   the experiment budget?
2. Will its improvement repay switching/reload/re-prefill costs over the remaining
   workload?

Switch only with explicit cost scenarios, uncertainty allowance, a default 5%
improvement margin, and at least one decision window of residence. Missing costs
disable cross-personality switching; zero-cost scenarios are labeled idealized.
Keeping the current configuration is a valid outcome.

## Approved completion plan

1. **Standalone correctness and quality.** Finish loading, chat templates, EOS,
   context/preflight, cancellation, timeouts, and build-cache review. Require all
   three official checkpoints to pass short-prompt + eight-token full-RTL
   acceptance against ISA logits/state. Complete frozen quality suites and strict
   standalone evidence manifests.
2. **Workbench and durable service.** Finish report/policy views, instruction and
   bounded trace inspection, conversation history, reconnect/cancellation,
   resource admission, failure provenance, and browser/accessibility acceptance.
   Live execution and recorded Lens replay remain visibly separate. Without trace
   evidence, show “awaiting trace,” not fabricated hardware activity.
3. **Benchmarking, optimization, and learning.** Complete 30 staged real-model
   performance runs, ten per model, using fixed token tapes and AXI scenarios.
   Evaluate exhaustive search, heuristic, measured predictor, and masked Double
   DQN against fixed and budget-matched random baselines. Use base-model-disjoint
   splits, at least five evaluation seeds, no candidate-timing leakage, and
   explicit promotion/rollback. Publish negative results as well as improvements.
4. **CUDA baseline and greedy hybrid.** After standalone acceptance, run local
   Qwen3-1.7B on the Zephyrus GPU, then simulated Qwen3-0.6B drafts with lengths
   1, 2, 4, and 8. Validate tokenizer mappings/formatting, cache rollback,
   rejection, EOS, cancellation, and context limits. Hybrid output must match
   GPU-only greedy output. Measure all end-to-end overhead; retain GPU-only
   execution when hybrid is slower.
5. **Review and publish evidence.** Finish reproducibility, security, upstream
   patch documentation, handoff, CI, and strict full-release checks. Keep the
   release incomplete until standalone, learning, UI, and Zephyrus CUDA/hybrid
   evidence is present. Publish reviewed milestones without automatic merging.

Search uses validation, not held-out data. Selectable variants require held-out
**NLL degradation ≤5% and next-token agreement ≥90%**, with at least 1,024
evaluated target tokens in each validation and held-out split per model.
Failed candidates remain visible but cannot be selected. Do not relax thresholds
or modify verified arithmetic to manufacture a pass.

## Measurements and boundaries

Report compilation/loading, host simulation time, prefill/decode, RTL cycles,
and assumed-clock projections separately. Verilator executes on the CPU; an
NVIDIA GPU does not directly accelerate RTL simulation. CUDA performance must be
measured on the Zephyrus, not substituted with Mac/CPU results.

Counters and controlled comparisons support compute, memory/backpressure,
dependency/controller, and setup diagnoses, including mixed/uncertain results.
Generic two-channel AXI simulation is not AWS HBM. Baseline scenarios use latency
20, stalls 20%, bandwidth 100%; stress scenarios use 100, 50%, and 50%.
Derived queueing and predictions are labeled estimated.

Physical FPGA timing, resource occupancy, power, and AWS bandwidth remain
unavailable. Latency-first and throughput-first objectives are separate;
energy-first stays disabled without a credible supplied energy source.
Simulation and accepted draft tokens alone do not establish acceleration.

## Run locally

Moving to the Zephyrus? Follow the
[Windows + WSL2 Ubuntu setup and continuation guide](docs/zephyrus-handoff.md).
Windows remains installed; Linux tools run inside WSL2. Rebuild Linux tools and
caches instead of copying Mac virtual environments/executables. The ignored
offline package at `build/Zephyrus-Transfer` has its own `START-HERE.md`, source
snapshot, data restoration instructions, and checksums.

Prerequisites: Python 3.10+, Node 22+, Make, Icarus Verilog, Yosys, and the
documented Verilator toolchain (v5.050). See the handoff for installation.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[llm,ide,test]'
make ui
make verify-llm PYTHON=.venv/bin/python
.venv/bin/python -m malleable.ide --model-root build/models
```

Open `http://127.0.0.1:8765`. Place supported checkpoint folders under the model
root. Downloading is a separate, explicit action:

```sh
.venv/bin/python tools/download_llm_models.py --destination build/models
```

Existing Mac checkpoints were downloaded but are not in Git. Weights, traces,
and generated artifacts stay local and ignored. Follow the handoff's fresh
job-directory instructions when migrating stores; do not redispatch historical
Mac-path jobs or automatically resume canceled acceptance jobs.

Defaults: one sequence, greedy decoding, 16 generated tokens, total context
2,048 tokens; maximum 256 generated tokens within that limit. Real-model RTL
tests can take substantial time.

```sh
make test
make verify
make verify-llm PYTHON=.venv/bin/python
make release-check PYTHON=.venv/bin/python RELEASE_MANIFEST=path/to/standalone-release.json
make full-release-check PYTHON=.venv/bin/python FULL_RELEASE_MANIFEST=path/to/full-release.json
```

Replace manifest placeholders with completed evidence manifests. Strict checks
fail when prerequisites/evidence are missing; ordinary CI downloads no models.
See [the local LLM guide](docs/local-llm-platform.md) for interfaces and commands.

## Preserved dense foundation

The independent accelerator implements signed INT8 inputs/weights, INT32
accumulation, tiled reductions, overflow reporting, per-output bias,
requantization, saturation, ReLU, and autonomous execution of up to four dense
layers. It includes inferred memories, ping-pong activations, board-independent
configuration/results, counters, and 1/2/4/8-lane personalities with active-lane
controls. Its arithmetic remains defined by
[the dense numeric contract](docs/numeric-contract.md).

Dense simulation/lint/Yosys regressions remain required. Quartus projects cover
the MAC and dot-product blocks, not the integrated top; physical fitting is
outside this release. The dependency-free dense host baseline remains usable:

```sh
python3 -m malleable analyze --size light
python3 -m malleable benchmark --size light --requests 4
python3 -m malleable optimize --size heavy --lanes 1 --active-lanes 1 --switch-cycles 10000
```

See [the model-aware system guide](docs/model-aware-system.md).

## Scope and contributing

Included: local simulation, validated configuration/quantization selection,
learning evaluation, the workbench, and gated CUDA/greedy hybrid.

Excluded: AWS provisioning/uploads/AFIs/HBM integration, physical programming,
partial reconfiguration, new runtime hardware modes, retraining/distillation,
contextual bandits, phase/operator experience retrieval, and extra architectures.
No universal model compatibility or guaranteed speedup is claimed.

```text
rtl/                 Independent dense SystemVerilog
sim/                 Dense self-checking testbenches
quartus/             Cyclone V block projects
malleable/           Host runtime, adapters, storage, learning and service
frontend/            React/TypeScript workbench
third_party/opentpu/ Pinned upstream engine and Lens assets
tests/               Host/backend/integration regressions
tools/               Downloads, verification and evidence utilities
docs/                Decisions, specifications, status and handoff
build/               Ignored checkpoints, outputs and evidence
```

Read [AGENTS.md](AGENTS.md) and [docs/STATUS.md](docs/STATUS.md), claim an open
task, and leave decisions/results in the repo. Preserve dense schema-v1,
checkpoints/datasets, private media, historical records, and unrelated `tmp/`.
Document numeric/interface changes before code depends on them.

The active release is specified in [the local LLM guide](docs/local-llm-platform.md).
Older SSM/research roadmaps are historical where they conflict with this release.

## License

Project code is [MIT licensed](LICENSE). Vendored OpenTPU retains its upstream
license and notices in `third_party/opentpu`.
