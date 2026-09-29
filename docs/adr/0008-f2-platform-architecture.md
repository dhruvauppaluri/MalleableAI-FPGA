# ADR-0008: AWS EC2 F2 platform around the frozen OpenTPU accelerator

Status: proposed, 2026-09-29. Scope: design and simulation only. Decider: project owner.

## Context

The owner asked for an AWS EC2 F2 platform (Virtex UltraScale+ VU47P, HBM, PCIe) for the
existing accelerator. ADR-0003 and AGENTS.md exclude "AWS provisioning, uploads, AFIs, HBM
and physical programming". This ADR narrows that exclusion for this task: **F2 design,
RTL and simulation are in scope; provisioning AWS resources, uploading, AFI submission,
board programming and spending money stay excluded until the owner approves each step.**
Nothing in this branch calls AWS.

The accelerator is a frozen block. This ADR changes no arithmetic, ISA, compiler output,
dense contract or OpenTPU source. New code lives in `f2/`, `malleable/f2/`,
`tests/test_f2_*.py` and `docs/`. No shared interface changes, so no numeric ADR is needed.

Facts about the AWS shell below are taken from a shallow clone of `aws/aws-fpga`
(`b603a81`, release 2.3.4, shell `small_shell=0x10212415`) read on 2026-09-29, and are
tagged **[HDK]**. Everything else that depends on the real shell or HBM IP and could not
be confirmed there is tagged **[ASSUMPTION]** and listed in the ledger at the end. Nothing
here has been built with Vivado or run on hardware: there is no Vivado in the working
environment.

## What exists to build on

- `third_party/opentpu/rtl/boards/ypcb-00338/otpu_board.sv` is upstream's board wrapper:
  the slice, a 12-bit AXI-Lite control block (`otpu_ctrl`, register map in
  `opentpu/host/regs.py`), a hardware trace buffer, and two 512-bit AXI4 master channels
  `m0`/`m1` in the core clock domain. Logical DRAM is interleaved over the channels in
  64-byte beats (logical beat b is on channel b % 2 at `BASE[b % 2] + (b // 2) * 64`,
  `BASE = {0, 0x8000_0000}`). Writes are single beats, reads are INCR bursts of up to 8
  beats. IDs are one bit.
- `opentpu/host/board.py` is upstream's host driver: `Board` (registers, DMA split/join,
  program load, run/wait, trace), `BoardBackend` (the `Engine` backend interface:
  `write/read/run`), and a transport interface (`mem_write(ch, off, data)`,
  `mem_read`, `reg_write`, `reg_read`, `reg_read_many`, `poll`). `SimTransport` runs the
  same protocol against a Verilator model of the board.

The F2 platform reuses both unchanged.

## Decision

### 1. Block structure

```
host (Board / BoardBackend, unchanged)
   |  OCL BAR0 (AXI-Lite, 32-bit)          PCIS BAR4 (AXI4, 512-bit)
   v                                        v
  f2_ocl --------- cross to core clk        f2_hbm_adapter.u_pcis (main clk)
   |                                        |
   |  0x0000..0x0FFF -> otpu_ctrl regs      +--- per-PC bridge: async FIFOs + arbitration
   |  0x1000..0x1FFF -> F2 register block   |            ^
   v                                        |            |
 otpu_board (frozen; core clk) --m0/m1--> f2_hbm_adapter.u_core[0/1] (core clk)
                                                         |
                                        N x 256-bit AXI3 PC ports (HBM AXI clk) -> HBM IP
```

`cl_otpu_core` (in `f2/rtl/`) is this structure with simplified port names.
The HDK-facing top (`cl_otpu`, HDK port names from `cl_ports.vh`, `sh_ddr` with
`DDR_PRESENT=0`, the HBM IP wrapper, `AWS_CLK_GEN`, HBM monitor APB wiring) is written in
PR-C and is **untested**; it needs Vivado and the HDK IP to elaborate.

### 2. Shell facts this design relies on [HDK]

| Item | Fact |
| --- | --- |
| Device | `xcvu47p-fsvh2892-2-e`, three SLRs (`build_all.tcl`). |
| Shell | Small Shell. No built-in DMA engine; HDK designs bring their own. XDMA features are marked "currently unsupported on F2". |
| Clocks | `clk_main_a0` 250 MHz fixed (all shell interfaces), `clk_hbm_ref` 100 MHz. Other clocks need the `AWS_CLK_GEN` IP or an MMCM. Recipes: A0 (250/62.5/187.5/250), A1 (250/125/375/500), A2; HBM AXI: H0 250, H1 125, H2 450, H3 300, H4 400 MHz. Build default `A1 B2 C0 H2`. |
| PCIS | 512-bit AXI4, INCR only, ID 16-bit, AppPF **BAR4 = 128 GiB**. Shell times out an unanswered PCIS access after **8 us** and then may need an AFI reload. |
| OCL | 32-bit AXI-Lite, AppPF **BAR0 = 64 MiB**. |
| BAR properties | All BARs prefetchable. |
| HBM | 16 GiB, 32 AXI3 channels (pseudo-channels), 256-bit, up to 450 MHz, 460 GB/s theoretical. The HBM IP's monitor APB interfaces must be connected (AFI creation fails otherwise). `CL_HBM`-style example: 34-bit address, 6-bit ID. |
| DDR | One 64 GiB DIMM through `sh_ddr` (must be instantiated even when unused, `DDR_PRESENT=0`). Not used here. |
| Example memory map | `cl_dram_hbm_dma`: DDR at PCIS 0x00_0000_0000 (64 GiB), HBM at 0x10_0000_0000 (16 GiB). |
| Build | `aws_build_dcp_from_cl.py`, examples take 30-90 min; a DCP with timing failures still produces a tarball but is "for testing only". Developer AMI wants at least 4 vCPU / 32 GiB. |

### 3. Memory map

**Core view (unchanged).** The compiler image is one flat per-slice DRAM image. Its layout
(`opentpu/llm/qwen3.py`): I/O rows (`x`, `cos`, `sin`, final norm, logits) first, then one
block per layer holding norms, quantized projections with their scales, **and that layer's
KV cache**, then the LM head. Weights, KV and activations are therefore interleaved by
layer and are not separable by address range. The image is followed by the program area
(4 x `IMEM_WORDS` bytes, at the next 4 KiB boundary). Logical bytes alternate between the
two core channels every 64 bytes.

**PCIS window (host view).** `HBM_BASE = 1 << 36` (as in the HDK examples), 4 GiB:

```
PCIS address = HBM_BASE + BASE[c] + channel_offset      c = 0, 1;  BASE = {0, 0x8000_0000}
```

This is the same channel/offset addressing the upstream driver uses against the original
board (`BASE[c] + off`), so `Board.write/read` map onto it one to one. Addresses outside the
window (including the DDR range) answer DECERR.

**HBM placement (inside `f2_hbm_adapter` only).** Channel offset `o` of core channel `c`:

```
PC    = c * PCS + ((o >> 9) % PCS)            PCS = PCs per core channel
local = ((o >> 9) / PCS) * 512 + (o & 511)    PC-local byte address
```

512-byte stripes (one full 16-beat, 256-bit burst) rotate over the channel's PCs.
`local` must fit `PC_AW` bits (29 = 512 MiB per real PC); otherwise the piece is answered
DECERR in the adapter and no PC is touched. The core and the PCIS path use the same
function, verified in `f2/sim/tb_f2_adapter.sv` and against a backdoor read of the HBM
model. Because the core still sees the original flat image, results are byte-identical by
construction of the equivalence check (section 8).

No region-aware placement (weights on some PCs, KV on others) is done: KV blocks are
interleaved with weights inside each layer block, so it would need compiler-side changes.
That is future work and would need its own ADR.

**Capacity.** Default `PCS = 2`: 4 PCs, 1 GiB per core channel, **2 GiB total image**,
using PCs 0-3. Rough image sizes (estimates from parameter counts and the image layout;
exact sizes come from `spec.image(...)` once weights are present, which the cloud
environment does not have):

| Model | INT8 weights + scales (est.) | KV at cap 2048 (est.) | Fits 2 GiB |
| --- | --- | --- | --- |
| Qwen3-0.6B | ~0.62 GB | ~0.12 GB | yes |
| Qwen3.5-0.8B | ~0.85 GB | small state | yes (estimate) |
| LFM2.5-230M | ~0.25 GB | small | yes |

`PCS` scales to 16 (all 32 PCs, 8 GiB per channel, 16 GiB total) by parameter.

### 4. Bandwidth, with a margin

The MXU consumes one 128-byte weight chunk per core cycle at full rate
(`third_party/opentpu/docs/wide_dram.md`): **128 B x f_core**.

| Quantity | 125 MHz core | 250 MHz core |
| --- | ---: | ---: |
| Peak core demand (both channels) | 16.0 GB/s | 32.0 GB/s |
| One HBM PC peak (256 bit x 450 MHz) | 14.4 GB/s | 14.4 GB/s |
| One PC at assumed 60% efficiency (range 50-70%) | 8.6 (7.2-10.1) GB/s | same |
| 4 PCs (PCS=2) at 60% | 34.6 (28.8-40.3) GB/s | same |
| Margin, 4 PCs | **2.2x (1.8-2.5x)** | **1.08x (0.9-1.26x)** |
| 8 PCs (PCS=4) at 60% | 69 (58-81) GB/s | margin 2.2x (1.8-2.5x) |

The 60% figure is an **[ASSUMPTION]** for long sequential bursts with rare small writes.
Nothing here measures HBM. It must be replaced by the bring-up measurement in
`docs/f2-bringup.md`.

**Decision:** first bring-up uses **PCS = 2 (4 PCs) at 125 MHz** for the core.
What breaks at 250 MHz:

1. Bandwidth margin collapses to about 1x on four PCs; `PCS = 4` (8 PCs) restores it but
   doubles the adapter, the crossings and the pin-out of the HBM ports.
2. Core timing: upstream reports its own Vivado closure of this block on a Kintex-7
   (xc7k480t): 100 MHz default, 120.755 MHz production (WNS +0.149 ns) and 125.49 MHz for the
   4-bit-weight image (`third_party/opentpu/docs/board.md`; upstream's numbers, not ours).
   UltraScale+ fabric is faster, which makes 125 MHz plausible on VU47P; nothing measured
   supports 250 MHz. 250 MHz would let the core use `clk_main_a0` directly (no core/main
   crossing), which is attractive only if it closes.
3. SLR crossings: HBM sits in the bottom region, PCIS in the SLR1/top region per the
   HDK notes, and the core wherever the tool puts it. Every crossing needs registers on
   both sides ([HDK] implementation tips); the adapter's per-PC bridges are the natural
   crossing point but are not yet floorplanned.
4. The 450 MHz HBM AXI clock closes only with the HDK's pipelining/floorplan recipe
   (`cl_mem_perf` reaches 450 MHz on all 32 channels). The per-PC bridge's request path
   (async-FIFO empty flag -> arbiter -> PC `valid`) is several logic levels and is **not yet
   pipelined for 450 MHz**. Fallback if the first timing report fails there: run the HBM AXI
   clock at 300 MHz (recipe H3, 9.6 GB/s per PC peak) with `PCS = 4` (8 PCs: 46 GB/s at
   60%, 2.9x margin over the 125 MHz core demand) before adding pipeline stages. Both
   options are parameters, not redesigns.

### 5. AXI mapping

| Aspect | Core side (`otpu_board` m0/m1) | Host side (PCIS) | HBM side (per PC) |
| --- | --- | --- | --- |
| Protocol | AXI4 | AXI4 | AXI3 (HBM IP) |
| Data width | 512 | 512 | 256 |
| Clock | core (125 MHz) | main (250 MHz) | HBM AXI (450 MHz) |
| ID | 1 bit | 16 bit | 6 bit, always 0 |
| Burst | INCR; writes 1 beat; reads 1-8 beats | INCR, up to 256 beats | INCR, at most 16 beats (`AxLEN <= 15`) |
| Address | `BASE[c] + off`, 64-byte aligned | window base + `BASE[c] + off` | PC-local |

Conversion (`f2_hbm_router`, one instance per core channel and one for PCIS):
bursts are split at 512-byte stripe boundaries so each piece is one legal AXI3 burst on
one PC; each 512-bit beat becomes two 256-bit beats (low half first) with the strobe
split; an in-order entry FIFO records each piece so responses from different PCs are merged
in issue order and one B (or the final R with `RLAST`) is returned per original burst;
`RRESP`/`BRESP` accumulate by OR. W may arrive before AW and is held until the AW piece is
issued. Only INCR, full 64-byte beats (address bits [5:0] ignored, `AxSIZE` ignored) are
supported; anything else is outside the accelerator's use and the PCIS driver contract.

`f2_hbm_pc_bridge` (one per PC) crosses every request/response channel through an
asynchronous FIFO and arbitrates the two sources (core channel, PCIS) per burst,
round-robin, with W and B following AW grant order and R/B returning to the issuing source
in order. A host access therefore waits behind at most a few bursts, never a whole program,
so the shell's 8 us PCIS timeout is respected under core traffic.

Errors: an out-of-window or out-of-range piece is answered DECERR by the router and pulses
an error counter (F2 register block); a DECERR or SLVERR response from the HBM propagates
upward. The core's own AXI-error bit (`ST_AXI_ERR`) reports it as on the original board.
`hbm_ready` gates the board block's reset and its OCL access (SLVERR/0xDEADBEEF while not
ready).

### 6. Clocks and reset

| Domain | Source | Target |
| --- | --- | --- |
| main | `clk_main_a0` | 250 MHz [HDK] |
| core | `AWS_CLK_GEN` `clk_extra_a1` (recipe A1) or an MMCM from main | **125 MHz** first bring-up |
| HBM AXI | `AWS_CLK_GEN` `clk_hbm_axi` (recipe H2) | 450 MHz [HDK] |
| HBM ref | `clk_hbm_ref` | 100 MHz [HDK] |

`--clock_recipe_a A1 --clock_recipe_hbm H2` (the HDK default). `AWS_CLK_GEN` starts with its
generated resets asserted; after AFI load the host must wait for the MMCMs to lock and release
them (SDK `aws_clkgen_deassert_resets`) before the HBM domain runs [HDK]. One asynchronous
reset `rst_main_n` is synchronized into every domain (`f2_rst_sync`); FIFO pointer resets
overlap because assertion is simultaneous. Core clock 125 MHz is a **plan**, not a timing
result.

### 7. Host protocol

- **Registers:** OCL BAR0, `0x0000-0x0FFF` is the original board register map
  (identical offsets, so `regs.py`, `Board.info()`, snapshot and trace code work unchanged);
  `0x1000-0x1FFF` is the F2 block (ID `0x46324F54`, version, capabilities incl. PCs per
  channel / PC address bits / stripe, status: HBM ready and sticky error bits, error
  counters, core kHz, build id, HBM base, scratch). OCL accesses are serialized through
  an asynchronous crossing to the core clock.
- **Bulk data:** PCIS BAR4. XDMA is unsupported on F2 today [HDK], so the transport maps
  BAR4 and uses CPU loads/stores (or `fpga_pci_lib`); the throughput is unmeasured. A
  CL-initiated PCIM path is future work.
- **Per-token flow** (unchanged from `BoardBackend.run`): write x/cos/sin into the image,
  copy the program to the program area, `PROG_ADDR/PROG_N`, `CTRL_LOAD`, poll
  `ST_LOADING`, `CTRL_RUN`, poll `ST_HALTED`, read counters (and optionally the trace),
  read logits. The host must not touch DRAM regions the running program uses (the
  original board has the same rule).
- **Transports** (`malleable/f2/`): the real BAR transport (PR-B, untested on hardware,
  refuses to run unless explicitly enabled), a software emulation transport, and
  `F2SimTransport` (this PR) that runs the Verilator model.

### 8. Equivalence and evidence rules

The gate is **DRAM and TMEM byte equality after every step** against the independent ISA
machine. `malleable/f2/replay.py` runs the recorded tiny 100-token test (seed 9,
balanced personality) twice per token from identical state: on the recorded path
(`CheckedRtlBackend`, `tb_top`) and through the unmodified upstream `BoardBackend` over the
F2 shell model, and fails on any difference.

Trace hashes:

- `ref_trace_sha256` is the recorded kind: sha256 of the whole `tb_top` stdout
  (`trace.txt`). **It is not reproducible.** The file ends with Verilator's own report lines
  (`- Verilator: $finish at 40ns; walltime 0.158 s; speed ...`), whose wall-clock values
  change on every run of the same binary; two runs on one machine give two hashes, so no
  re-run (ours or the owner's) can match a recorded raw hash. Only the total cycle count
  (406,800 for the 100-token test, identical to the recorded Zephyrus run) reproduces.
- `ref_trace_normalized_sha256` is a **new** hash defined here: the same text without the
  lines that start with `- `. It reproduces exactly between independent runs and builds and
  is comparable only with another `tb_top` run of the same configuration.
  `python -m malleable.f2.tracehash` computes it from saved `step-*/trace.txt` files, so it
  can be cross-checked on any machine that kept its traces.
- `projection` is another **new** hash: sha256 over the dispatch events
  (`pc op w1 w2 w3`) of the trace with cycle stamps removed. It is compared between the
  reference run and the F2 run and must not be described as matching any recorded hash.
  It is a consistency check on instruction issue order, not a numeric check.

Every result is labelled simulated, estimated, assumed, measured or unavailable. The HBM
model (`f2/sim/f2_hbm_model.sv`) is not the AMD HBM IP; simulated core cycles are not
wall-clock time and not F2 performance.

## Alternatives considered

- **DDR4 instead of HBM.** One 64 GiB DIMM behind `sh_ddr` (512-bit AXI4). Simpler
  interface (no width or protocol conversion) and enough capacity, but the HDK files read do
  not state the DIMM's data rate; a single 64-bit DDR4 channel peaks at roughly 19-26 GB/s
  depending on speed grade **[ASSUMPTION, unverified]**, which leaves little or no margin over
  the 16 GB/s core demand once refresh and row misses are counted. HBM is the stated goal.
- **1:1 channel-to-PC mapping.** No striping; one PC per core channel gives 8.6 GB/s per
  channel at the assumed efficiency against an 8 GB/s demand at 125 MHz: no margin. Rejected.
- **Region-aware placement (weights vs KV on separate PCs).** Blocked by the image layout
  (KV inside each layer block); needs compiler changes. Deferred.
- **XDMA/SDE for bulk transfer.** XDMA is unsupported today [HDK]; the Streaming Data
  Engine (`cl_sde`) would add a DMA engine to the CL. Deferred until BAR4 throughput is measured.
- **Re-implementing the core for HBM (wide 256-byte path).** Ruled out: the accelerator is
  frozen. `wide_dram.md` describes the upstream option; it changes the MXU.

## Consequences

- The frozen block runs as is; F2 differences are confined to the wrapper, adapter and host
  transport. If timing on UltraScale+ needs core changes (DSP/RAM primitive mapping), that
  is a separate, explicit decision and will need its own ADR.
- Simulation evidence exists for the wrapper, the adapter and the recorded tiny test; there
  is no synthesis, place-and-route, timing, resource or power evidence, and none on hardware.
- The first real unknown is Vivado closure on UltraScale+; `docs/f2-bringup.md` makes a
  CPU-only out-of-context synthesis on the AWS developer AMI the first paid step, subject
  to the owner's approval.

## Assumption ledger

| # | Assumption | How it will be resolved |
| --- | --- | --- |
| A1 | HBM sustained efficiency of about 60% (50-70%) for the core's burst pattern | Bring-up bandwidth test (`f2-bringup.md`, stage 4) |
| A2 | Each HBM AXI port addresses its own PC with a 29-bit local address inside the IP's 34-bit address (no lateral switching) | Confirm against AMD PG276 and the HBM IP configuration used by the HDK's `cl_hbm_wrapper` before the first build |
| A3 | PCs 0-3 are a good choice for placement (same stack and near the shell's HBM side) | Vivado floorplan review |
| A4 | The core closes at 125 MHz on the VU47P | Out-of-context synthesis, then implementation |
| A5 | BAR4 CPU load/store throughput is adequate for image load (about 1-2 GB/s assumed) | Bring-up stage 3; PCIM or SDE if not |
| A6 | Prefetchable-BAR semantics do not disturb the register reads (counters, trace address auto-increment on `TRACE_HI`) | Bring-up stage 2 register test; the driver already tolerates this on the original board's BAR0 |
| A7 | The CL fits beside the Small Shell (the HDK text says the top SLR is fully available) | Utilization from Vivado |
