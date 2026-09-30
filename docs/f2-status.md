# F2 platform status (Claude, cloud session)

Handoff for the AWS EC2 F2 (VU47P, HBM, PCIe) platform work. `docs/STATUS.md` carries only a
one-line row; everything else is here so the two files do not conflict. Update this before
stopping.

Design record: [ADR-0008](adr/0008-f2-platform-architecture.md) (proposed; renumber if the
number is taken). Bring-up plan and cost: [f2-bringup.md](f2-bringup.md). Resource/timing
estimate: [f2-resource-timing.md](f2-resource-timing.md). Vivado scripts (untested):
`f2/vivado/README.md`.

Branches (stacked, draft PRs, none merged):

| PR | Branch | Contents |
| --- | --- | --- |
| A | `claude/f2-platform` | ADR, RTL (HBM adapter, OCL front end, CL core), simulation testbenches, replay of the tiny 100-token test, placement map, `make` targets |
| B | `claude/f2-host` | Host transports (real BAR transport, software emulation), DMA-loopback / descriptor / step harness |
| C | `claude/f2-vivado` | Vivado out-of-context and HDK build scripts (untested), HDK-facing top, Yosys resource tool, resource/timing and bring-up documents |

## Scope and rules

- Design and simulation only. **No AWS provisioning, AFI submission, uploads or spend without
  explicit user approval.** Nothing in these branches calls AWS.
- The accelerator is a frozen block: no changes to `third_party/opentpu/**`, `rtl/**`, the ISA,
  numerics or the dense contract. All new code is in `f2/`, `malleable/f2/`,
  `tests/test_f2_*.py` and `docs/`. No shared interface changed, so no numeric ADR was needed.
- Equivalence gate: per-step DRAM and TMEM bytes against the independent ISA machine. Hashes are
  secondary and labelled (see "Hash findings").
- Every number is labelled measured, simulated, estimated, assumed or unavailable. There is no
  Vivado in the cloud container: **no synthesis, place-and-route, timing or power result exists
  for the VU47P.**

## Host harness (PR B)

```sh
python -m malleable.f2 all --mode emulate                # software card; default
python -m malleable.f2 loopback --mode sim               # PCIS write/read through the Verilator RTL
python -m malleable.f2 descriptor --mode sim             # register + program-load + RUN path on the RTL
python -m malleable.f2 steps --mode sim --steps 20       # tiny-model tokens, DRAM equal to the ISA machine each step
# never run, needs an explicit opt-in, a loaded AFI and a PCI address:
python -m malleable.f2 all --mode hardware --enable-hardware --bdf 0000:00:1d.0
```

Results carry their provenance (`emulated`, `simulated`, or `measured on a real F2 card`);
emulated timings are host memcpy/interpreter speed and simulated timings are the simulator's.
`F2BarTransport` (real card through the PCIe BARs) is UNTESTED on hardware; its register and
window plumbing is tested against ordinary files standing in for the BARs.

## Toolchain used in the cloud container (2026-09-29)

Verilator 5.050 (built from tag v5.050), Icarus Verilog 12.0, Yosys 0.33: the same versions as
the repository's pinned Zephyrus toolchain. Python 3.11, torch 2.14 (CUDA wheel, CPU use only),
numpy 2.4. Vivado: not available. HDK facts from a shallow clone of `aws/aws-fpga`
(`b603a81`, release 2.3.4) in the session scratchpad.

## Evidence (simulation; nothing here is hardware or AWS)

| Item | Result |
| --- | --- |
| Baseline `make test` on the base branch | passed (29 host tests + RTL benches) |
| Baseline `make verify` on the base branch | exit 0 |
| Recorded tiny 100-token test regenerated here | 100/100 steps bit-exact against the ISA machine, 406,800 cycles: identical to the recorded Zephyrus total |
| Adapter bench (`f2/sim/tb_f2_adapter.sv`) | passes for 2 and 4 PCs per channel, under several HBM stall/latency settings, three unrelated clocks |
| Shell model (`f2/sim/tb_f2_shell.sv`): registers, OCL decode, DMA loopback both channels, DECERR counting | passes (`tests/test_f2_rtl.py`) |
| **100-token replay through the F2 shell model** (upstream `BoardBackend` unchanged, real OCL/PCIS/HBM-adapter path, 4 steps with images moved through PCIS DMA) | **DRAM 100/100, TMEM 100/100, retired instructions 100/100 equal to the ISA machine; dispatch-projection hash 100/100 equal to the reference run.** Simulated core cycles 478,004 (reference 406,800: the shell path adds the program-load and HBM-model latency); not a hardware number. Evidence: `docs/evidence/f2-tiny-replay-sim.json` |

## Hash findings (important)

- The recorded per-step `trace_sha256` (sha256 of the whole `tb_top` stdout) **cannot be
  reproduced by any re-run**: the file ends with Verilator's report lines, whose wall-clock
  values change on every run of the same binary (`- Verilator: $finish at 40ns; walltime
  0.158 s; speed ...`). Two runs on one machine give two hashes. So a raw-hash cross-check
  against the Zephyrus record is impossible by construction, and any per-step raw hashes saved
  from `tiny-release.json` will not match this or any other run. Only total cycles reproduce
  (406,800, equal to the Zephyrus record).
- New, reproducible hashes defined in `malleable/f2/replay.py` (labelled as new, not "recorded"):
  `ref_trace_normalized_sha256` (the same text without the `- ` report lines; identical across
  independent runs here) and `projection` (dispatch events without cycle stamps; compared
  between the reference and the F2 run). `python -m malleable.f2.tracehash <traces dir>` computes
  the normalized hashes from any saved `step-*/trace.txt`, so a cross-check with Zephyrus needs
  its saved traces, not the raw hashes.
- The originally planned "transparent" (cycle-identical) mode is not possible: the F2 shell
  path loads programs through the boot loader and the HBM model, so its cycle stamps differ from
  `tb_top`'s by construction. The byte gate plus the two labelled hashes replace it; the HBM
  stall/latency/clock-ratio variations are covered in `tests/test_f2_replay.py`.

## Design findings worth knowing

- Upstream already ships a board wrapper (`otpu_board`) and a host driver with pluggable
  transports (`Board`, `BoardBackend`, `SimTransport`). The F2 work wraps the former and adds
  transports for the latter; the LLM path (`Engine` -> `BoardBackend`) is unchanged.
- `otpu_board` wires its slice's `dump` input to 0, so TMEM is not observable through the
  board wrapper; the F2 testbench reads the TMEM shadow hierarchically (`u_slice.u_tmem.shadow`).
- KV cache is interleaved with the weights inside each layer block of the compiler image, so a
  region-aware HBM placement (weights vs KV on separate PCs) is not possible without compiler
  changes. The adapter stripes uniformly at 512 bytes.
- The first bridge implementation selected between sources with variable part-selects: 25,012
  LUT for four PCs under Yosys. Rewritten as explicit multiplexers: 6,402 LUT. Found by the
  Yosys run, not by simulation.
- Vivado OOC synthesis of the frozen core on VU47P (run by the owner, 2026-09-30): 116,080 LUT
  (8.9%), 103,202 FF (4.0%), 496 block RAM tiles (24.6%), 283 DSP (3.1%), post-synthesis setup
  WNS +0.467 ns at 125 MHz. Smaller than the Kintex-7-based estimate; synthesis only, adapter
  and 450 MHz paths not included.
- Not done / not possible here: Vivado place and route or HDK builds, hardware runs, an
  exact model-image size for the three target models (needs the weights, which the cloud
  environment does not have; sizes in the ADR are estimates).

## Open items for the owner

1. The first paid step (out-of-context synthesis, stage 1) was approved and run on 2026-09-30;
   results are in `docs/f2-resource-timing.md` section 5. Next decision: approve (or not) OOC
   place and route (stage 2, `impl=1`; hours of a build instance). Please also report the actual
   AWS charge for stage 1, so the cost estimate can be checked against a real bill.
2. Decide how to use the hash finding: the per-step raw hashes being saved from the Zephyrus
   run are not comparable with any re-run; normalized hashes from the saved traces are.
3. The `verify-llm` Makefile line (`$(MAKE) test-f2`) is a one-line addition and should not be
   merged into the base branch before Zephyrus' final verification run.
4. ADR number: 0008 was free when written; renumber if another ADR took it.

## Log

- 2026-09-29: branch created from `origin/codex/local-llm-platform` (`4dc0aa6`). HDK read
  (with the owner's approval). Toolchain built. Baseline tests run. ADR, RTL, testbenches,
  replay, host transports, harness, Vivado scripts and documents written; validation results
  above.
- 2026-09-30: stage-1 OOC synthesis run by the owner on an AWS FPGA Developer AMI (Vivado
  2025.2); results recorded; one redundant `if` removed from `ooc_clocks.xdc`. No other AWS
  resource was used.
