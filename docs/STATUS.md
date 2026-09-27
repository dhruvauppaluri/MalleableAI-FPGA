# Status

Handoff for Cursor, Codex, and Claude. Update this before ending a session.

Last updated: 2026-09-27. Any of Cursor, Codex, or Claude may claim any `open` row.

## Now

Stage 1 only. The RTL dense-network datapath is simulated, linted, and checked
with coarse Yosys synthesis. The software toolchain does not exist yet.

## Done

- Signed INT8 MAC with signed INT32 accumulation and explicit overflow
- Parameterized INT8 dot product, multi-tile accumulation, bias,
  requantization, INT8 saturation, and optional ReLU
- Descriptor-driven top for up to four dense layers, with ping-pong activation
  buffers
- Self-checking testbenches and `make verify` in GitHub Actions
- Numeric contract, architecture notes, research roadmap, and ML contributor
  roadmap

## Open

| Status | Owner | Item | Notes |
| --- | --- | --- | --- |
| open | | Bit-accurate golden model, then quantization, one dense layer, and a tiny network | Milestone order in `docs/ml-contributor-roadmap.md`. Do not change RTL to force a match. |
| open | | Versioned model descriptor and exported test vectors | ADR-0001 item 4. Depends on the golden model. RTL simulation should be able to consume the vectors. |
| open | | Extend Cyclone V Quartus projects to `malleable_accelerator_top` and record real resource and timing results | Needs Quartus Prime Lite. Current projects are `quartus/int8_mac` and `quartus/int8_dot_product`. |
| open | | Host-to-Cyclone-V transport and benchmark harness | ADR-0001 item 6. Depends on exported artifacts. |
| open | | Review the first software/RTL cross-check against `docs/numeric-contract.md` | Depends on the golden model and vector export. Do this before either side changes the contract. |

## Out of scope until Stage 1 is complete

GPU-only baseline and the FPGA/GPU hybrid. See `docs/research-roadmap.md`.

## Session log

- 2026-09-27: Added shared agent instructions. No implementation change.
