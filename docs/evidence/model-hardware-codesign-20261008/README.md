# Shared-matrix co-design pilot (2026-10-08)

This branch tests whether a Qwen3-compatible student can reuse its MLP gate and
up matrices across layers and whether the compiler can store each matrix once.
It is an experiment. The approved pretrained models, published quality evidence,
30 RTL benchmarks, and controller policy are unchanged.

## Frozen inputs and scope

- `feasibility.json` reads and verifies the ten published Qwen3 RTL benchmark
  rows. Their row IDs are recorded in the file. No benchmark was dispatched.
- The image projection uses the local Qwen3-0.6B `config.json` (SHA-256 in the
  report), balanced personality, context 128, and INT8. It is a layout
  calculation; the pretrained checkpoint has **not** been tied or retrained.
- `tiny-training.json` records a CPU experiment with a two-layer, 128-wide
  Qwen3-compatible teacher and two students. The ordinary and shared students
  start with identical values and receive identical batches. The shared student
  ties both MLP gate and up weights across its two layers. Synthetic sequence
  seeds, training steps, and the diagnostic pass rule are in the report.
- The synthetic validation seed is **not** an independent release held-out
  partition. An initial training run scored that seed, then failed while
  constructing an incompatible one-slice ISA config. The corrected run reused
  the seed. Neither run opened a published or consumed quality suite.

## Result

| Measure | Ordinary | Shared | Interpretation |
| --- | ---: | ---: | --- |
| Projected Qwen3 image bytes | 623,138,816 | 447,961,088 | 28.11% smaller image if a real tied checkpoint exists |
| Minimum configured DRAM bytes, balanced geometry | 1,073,741,824 | 536,870,912 | Configuration capacity only; no board allocation measured |
| Decode instructions / MM instructions | 444 / 70 | 444 / 70 | No reduction in matrix operations |
| Tiny diagnostic validation NLL | 0.52165 | 0.49537 | Synthetic token arithmetic only |
| Tiny agreement with teacher's top token | 87.75% | 89.24% | Shared misses the predeclared 90% diagnostic gate |
| Tiny image bytes | 428,544 | 365,056 | The tied matrices occupy one region |

For the tiny tied checkpoint, the ordinary and shared ISA layouts produced
bit-exact logits over eight diagnostic tokens. The focused ISA test also checks
KV state and rejects weights that differ after being declared shared. This
validates the storage relocation, not the usefulness of a new language model.

The predeclared diagnostic rule requires shared NLL no more than 105% of the
ordinary control, at least 90% teacher top-token agreement, and bit-exact ISA
layout outputs. The shared student meets the first and third conditions, but
**fails** the second. This candidate must not move to real-model training or
an independent quality campaign on the strength of this result.

The CI tiny RTL test runs a new two-layer random tied workload under the AXI
simulator. It compares ordinary and shared logits, each RTL run with its ISA
state, and instruction counts. CI exports a `tiny-shared-weight-rtl` JSON
artifact containing cycles and AXI reads. This is a new pilot workload, not
one of the 30 published benchmark tapes. RTL performance and physical FPGA
benefit are **pending** until that artifact is reviewed and published here.

## Decision

Keep `shared_matrices` opt-in and require equality of all declared shared
weights before image construction. Do not apply it to the pretrained model or
promote it as a controller personality. Automatic program switching remains
gated. A later candidate needs a frozen architecture and fresh independent
quality and performance evaluations; the consumed suites cannot be reused.
