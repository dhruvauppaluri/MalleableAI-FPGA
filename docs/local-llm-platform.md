# Local LLM platform (AWS last)

The independent dense INT8/INT32 backend is unchanged. OpenTPU revision
`15754e971b55591b91048c4c636023fe59b343e7` is vendored under Apache-2.0 with
an upstream manifest and patch ledger. It is not an F2 backend. No AWS SDK,
provisioning, uploads, AFIs, HBM or board programming are included.

## Run

Install `.[llm,ide,test]`, run `make ui`, then
`malleable-ide --model-root build/models`. The service listens only on loopback
and rejects foreign origins/hosts. Place config, tokenizer and Safetensors
(optionally indexed shards) in each checkpoint folder. Symlink escapes,
duplicate/missing shard tensors, nonfinite weights, pickle and remote code are
rejected. GGUF/Ollama and arbitrary architectures are not supported.

Explicit opt-in downloads: `python tools/download_llm_models.py`. This separate
tool records official repository revisions and file hashes, fetches no Python
or pickle files, and is never called by inference. Models remain ignored local
data. Preserve their license files and model cards.

```sh
malleable-llm analyze --model build/models/Qwen3-0.6B
malleable-llm generate --model build/models/Qwen3-0.6B --prompt Hello
malleable-llm generate --model build/models/Qwen3-0.6B --prompt Hello --backend isa
malleable-llm benchmark --model build/models/Qwen3-0.6B --prompt Hello --max-new 8
```

For configuration comparisons, pass the same explicit `--input-tokens` tape to
`benchmark`; ordinary greedy outputs are reported separately and are not treated
as a fixed-workload comparison. The staged per-model performance schedule is
created and run with:

```sh
python tools/create_llm_performance_manifest.py --model build/models/Qwen3-0.6B --output build/models/quality/qwen3-performance.json
malleable-llm performance-suite --manifest build/models/quality/qwen3-performance.json --store build/real-benchmarks/Qwen3-0.6B
```

It schedules eight fixed-tape INT8 comparisons across four personalities and two
short workloads, then balanced INT4/FP4 if both pass frozen validation, or two
INT8 runs under the declared AXI stress scenario. Repeat for each model. Each
entry has a fixed token hash; simulator latency/stall/bandwidth values are
scenario inputs, never physical-memory measurements.

Default: full Verilator RTL, one sequence, greedy decoding, 16 generated tokens,
2,048 total context; at most 256 generated tokens. `--prompt-format raw` permits
short unformatted acceptance prompts; chat formatting is the default. No hidden
ISA prefill is used. Each token's entire program runs through the selected
backend. An independent matching ISA machine checks all persistent image bytes
and TMEM scratch, not merely the selected next token. State persists in DRAM;
per-program TMEM is reinitialized like the simulation harness. Reset re-prefills
from position zero. Temporary binary verification dumps are removed after checks;
complete traces, programs and Lens profiles remain local.

## Personalities and numeric boundary

All personalities use one slice, block depth 128, ACT RAM 128 blocks, TMEM 65,536
words, instruction memory 65,536 words, dispatch window 16, RPB 4, WPB 2, tree
MXU and PAIR disabled. DRAM capacity rounds the compiler image up to a power of
two; it is a simulation allocation, not proof of physical capacity.

| Personality | Matrix columns | Vector lanes | FIFO |
| --- | --- | --- | --- |
| compact | 2 | 8 | 128 |
| balanced | 4 | 8 | 512 |
| compute | 8 | 16 | 512 |
| buffered | 4 | 8 | 1024 |

Other settings are fixed. Generic two-channel 512-bit AXI simulation exposes
latency, stall probability and bandwidth scenarios. There is no F2 HBM claim.
Build caches include sources, tool version and parameters; compilation is
separate from execution. Space-free source staging works around Verilator's
generated make invocation for paths containing spaces.

OpenTPU FP32 round-nearest-even/flush-to-zero and block INT8/INT4/FP4 arithmetic
remain distinct from dense ties-away requantization. The pinned implementation
and its [quantization contract](../third_party/opentpu/docs/quant.md), ISA and
bit-exact FP/operator regressions are authoritative for that backend. INT8 output
heads are retained with 4-bit matrices. Changing this arithmetic requires a new
ADR; it must never be patched to match a convenient software output.

## Jobs, evidence and UI

Jobs and ordered events are durable SQLite records. SSE supports Last-Event-ID,
recovery, duplicate rejection, stale-run isolation and cancellation. Heavy jobs
are serialized; API/inspection remain independent. Historical SSM records stay
readable but cannot resume. Browser activity is coalesced to ten updates/second;
no trace means “awaiting trace.” Full traces stay outside the browser. Lens is
explicitly **recorded replay**, with the actual personality and assumed-clock
labels, not a picture of upstream's physical board.

Header selects model/backend/personality/numeric mode; left is prompt/output;
center is evidence and Lens; right is metric provenance; bottom is job timeline.
Model/log text is escaped. New runs reset/re-prefill and never reuse another
variant's caches. The workbench does not automatically deploy optimizer choices.

Host loading/prefill/decode time, simulated cycles and assumed-clock projections
are separate. Physical timing, power, DSP occupancy, dollars and F2 bandwidth
are unavailable unless later supplied by a real backend. Useful MAC utilization
must not be called DSP resource occupancy. Mac/CPU results are correctness/development
evidence, not the Zephyrus's CUDA performance.

RTL bottleneck labels use conservative, versioned simulation heuristics over the
DRAM-bound cycle fraction, useful-MAC utilization, and dependency-gap cycles.
The numeric DRAM cycle bound is retained as a number, never used as a category;
raw profiles remain on disk while event summaries report only the five largest
dependency gaps. Physical external-memory behavior remains unavailable.

## Quality, optimization and learning

Quality suites contain versioned token-ID sequences in `calibration`,
`validation`, `held-out`; splits must be nonempty, distinct and hashed. Evaluate
original local floating weights versus matching quantized ISA, not against
unrelated/generated reference text. Search uses validation; promotion requires
held-out NLL degradation ≤5% and next-token agreement ≥90%. Failed variants stay
visible and cannot become selectable. A normal prompt result is **not** quality
approval.

```sh
malleable-llm quality --model MODEL --suite SUITE.json --split validation
malleable-llm quality --model MODEL --suite SUITE.json --split held-out
malleable-llm optimize --store build/llm-jobs/research --current RESULT_HASH --window WINDOW.json
malleable-llm train --store STORE --episodes EPISODES.json --passes 20
malleable-llm evaluate --store STORE --episodes HELD_OUT.json --checkpoint HASH --split held-out
malleable-llm promote --store STORE --checkpoint HASH --report EVALUATION_HASH
malleable-llm rollback --store STORE --report DEPLOYMENT_HASH
```

Decision windows define remaining requests, arrival interval, objective and
explicit drain/program/reload/warmup/re-prefill cycle costs for each transition.
Missing costs disable switching; zero is marked idealized. Apply 5% improvement
margin, uncertainty and one-window minimum residence. Latency and throughput
are separate objectives; energy is disabled. Search decisions are recommendations,
not physical reconfiguration. One-time development/build time is not FPGA
programming cost.

Offline episode requests reference stored experiments, not user-invented timing
tables. Double DQN has masked actions, replay, a separate target network, seeded
training and versioned checkpoints. Its rewards are **estimated windows from
measured RTL service cycles**, not measured online rewards. Five-seed evaluation
compares fixed, heuristic, budget-matched random and exhaustive results and
includes regressions. Splits use base-model IDs; optional leave-family-out is
supported. Policies require explicit held-out non-regression promotion and retain
the previous policy for rollback; no per-sample automatic deployment.

## Release evidence

`make verify` retains dense simulation/lint/Yosys structural checks.
`make verify-llm` adds upstream ISA, FP, operators, quantization, compiler and
Verilator regressions, four-personality/three-format state checks, a 100-token
tiny full-RTL sequence and UI event tests/build. Ordinary CI downloads no models.

`malleable-llm release-check --manifest RELEASE.json` is the strict standalone
gate: all three official checkpoint lineages, a prompt plus eight generated
tokens entirely through RTL, the durable tiny 100-token RTL evidence, and a
frozen matching held-out suite with at least 1,024 targets are required. Each
quality result must match its generation's model, tokenizer, variant, personality
and configuration. The quality tool marks only passing held-out evidence as
selectable; validation is search-only. Missing or failed evidence fails, not
skips. The schema template is `examples/standalone-release-manifest.template.json`.

To create the synthetic evidence once (the destination must not already exist):

```sh
python tools/export_tiny_rtl_evidence.py --output build/release-evidence/tiny
```

The resulting `tiny-release.json` includes the exact build ID and every checked
trace hash; its file SHA-256 goes in the release manifest. Frozen quality suites
can be regenerated for a local checkpoint with `tools/create_llm_quality_suite.py`.
Do so before configuration search, and keep the split hashes fixed. Real quality
evaluation uses teacher-forced token tapes and stores only floating-reference
target NLL/top-1 outputs, not full vocabulary-logit arrays.

## Zephyrus CUDA and hybrid gate

This release's final external stage targets the Zephyrus in WSL2 Ubuntu. Install
the NVIDIA driver on Windows, install/update WSL2, then install Linux user-space
packages in Ubuntu as needed. NVIDIA says the Windows driver is the only driver
needed; do not install a Linux display/GPU driver inside WSL. Follow the
[NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

The optional `Qwen3-1.7B` verifier is fetched only by explicit opt-in:

```sh
python tools/download_llm_models.py --include-qwen3-verifier
```

The downloader resolves and records the upstream immutable commit and verifies
downloaded file hashes. Startup and inference never download. Run standalone
acceptance first, then use the passing manifest with `gpu-generate` and
`hybrid-generate`. The hybrid executes the Qwen3-0.6B draft through full RTL,
not ISA; Qwen3-1.7B verifies greedily on CUDA. Tokenizer graphs and probe
encodings must match exactly. The output must match GPU-only greedy output.
CUDA/model-load/decode timing is separate from RTL cycles, and an accepted draft
block is not itself evidence of speedup. Save both result artifacts and the
standalone manifest, then run `malleable-llm full-release-check --manifest
FULL_RELEASE.json`. This final gate also requires the three complete ten-run
fixed-tape reports, a base-model-disjoint five-seed held-out policy evaluation,
and checkpoint-free `make verify-llm` evidence. The template is
`examples/full-release-manifest.template.json`. That final command intentionally
fails unless both actual Zephyrus CUDA and hybrid evidence are present.

## Acceptance and remaining gates

`make verify` retains dense simulation/lint/Yosys structural checks.
`make verify-llm` adds upstream ISA, FP, operators, quantization, compiler and
Verilator regressions, four-personality/three-format state checks, a 100-token
tiny full-RTL sequence and UI event tests/build. Ordinary CI downloads no models.

`malleable-llm release-check --manifest RELEASE.json` validates local standalone
simulation; `malleable-llm full-release-check --manifest FULL_RELEASE.json`
additionally requires benchmark, learning, UI verification, local CUDA and
greedy-hybrid evidence. Neither
gate proves physical FPGA fit, timing, resource occupancy, power, or guaranteed
acceleration. Consult STATUS for actual results.

CUDA/hybrid job APIs are standalone-gated. Draft rejection re-prefills accepted
context, and verifier suffix state is cropped before correction. Unsupported
tokenizer pairs and CPU fallback are rejected. Comparative performance remains
an empirical question; slower hybrid results remain valid evidence. No verifier
checkpoint is downloaded implicitly.
