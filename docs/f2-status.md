# F2 platform status (Claude, cloud session)

Handoff for the AWS EC2 F2 (VU47P, HBM, PCIe) platform work. `docs/STATUS.md`
carries only a one-line row; everything else is here so the two files do not
conflict. Update this before stopping.

Branch: `claude/f2-platform`, stacked on `codex/local-llm-platform`.

## Scope and rules

- Design and simulation only. **No AWS provisioning, AFI submission, uploads or
  spend without explicit user approval.** Nothing in this branch calls AWS.
- The accelerator is a frozen block: no changes to `third_party/opentpu/**`,
  `rtl/**`, the ISA, numerics or the dense contract. All new code lives in
  `f2/`, `malleable/f2/`, `tests/test_f2_*.py` and `docs/`.
- Equivalence gate: per-step DRAM and TMEM bytes against the independent ISA
  reference. Trace hashes: raw hashes only where raw cycle stamps are valid;
  timing-normalized projections are new hashes and are labelled as such.
- Every number is labelled measured, simulated, estimated, assumed or
  unavailable. There is no Vivado in the cloud container, so there are no
  synthesis, place-and-route, timing or power results.

## Toolchain used in the cloud container (2026-09-29)

Verilator 5.050 (built from tag v5.050), Icarus Verilog 12.0, Yosys 0.33: the same
versions as the repository's pinned Zephyrus toolchain. Python 3.11, torch 2.14
(CUDA wheel, CPU use only), numpy 2.4. Vivado: not available.

## Progress

| Step | State |
| --- | --- |
| Baseline `make test` on the base branch | passed (29 host tests + RTL benches) |
| Baseline `make verify` | see log entry below |
| PR-A: ADR, RTL wrapper/adapter, testbench, replay | in progress |
| PR-B: host transport and benchmark harness | not started |
| PR-C: Vivado scripts, resource/timing doc, bring-up plan | not started |

## Log

- 2026-09-29: created branch from `origin/codex/local-llm-platform`
  (`4dc0aa6`). Read the F2 shell interface from `aws/aws-fpga` (shallow clone in
  the session scratchpad, commit `b603a81`, release 2.3.4), with the user's
  approval.
