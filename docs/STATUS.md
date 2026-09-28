# Status

Handoff for Cursor, Codex, and Claude. Update this before ending a session.

Last updated: 2026-09-27. Any of Cursor, Codex, or Claude may claim any `open` row.

## Now

Stage 1 remains simulation-first. Dense software/RTL integration now exists.
Codex is completing verification and review handoff for a separate SSM operator
prototype, local training/IDE and automatic Quartus characterization jobs.
No physical-board fit, timing, power or useful-chatbot quality is established.

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
| done | Codex | Bit-accurate dense golden model and tiny-network cross-check | [PR #1](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/1), `malleable/model.py`. Framework calibration remains separate. |
| done | Codex | Versioned dense model descriptor and exported test vectors | [PR #1](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/1), `malleable/records.py` and generated RTL benches. |
| open | | Extend Cyclone V Quartus projects to `malleable_accelerator_top` and record real resource and timing results | Needs Quartus Prime Lite. Current projects are `quartus/int8_mac` and `quartus/int8_dot_product`. |
| open | | Host-to-Cyclone-V transport and benchmark harness | ADR-0001 item 6. Depends on exported artifacts. |
| open | | Review the first software/RTL cross-check against `docs/numeric-contract.md` | Depends on the golden model and vector export. Do this before either side changes the contract. |
| in progress | Codex | SSM simulation/operator foundation, local IDE and automatic Quartus jobs | User-authorized SSM direction; see ADR-0002 and `docs/ssm-platform.md`. Prototype Q14 is separate from dense INT8. |
| open | | Production INT4/INT8 SSM calibration, elastic modes and hardware-aware retraining | Not implemented by the Q14 prototype. Must pass held-out language/quality gates. |
| open | | SSM memory/timing optimization, external-memory model and Windows Quartus worker | Virtual-pin characterization is not deployment. No board programming until explicitly enabled. |

## Out of scope until Stage 1 is complete

GPU-only baseline and the FPGA/GPU hybrid. See `docs/research-roadmap.md`.

## Session log

- 2026-09-27: Added shared agent instructions. No implementation change.
- 2026-09-27 (Codex): Preserved the dense baseline, implemented dense host learning
  and RTL instrumentation, then added a separate SSM simulation foundation.
  New remote guidance was pulled without overwriting local work. Existing and
  new tests pass; full structural sweep and PR handoff are in progress.
