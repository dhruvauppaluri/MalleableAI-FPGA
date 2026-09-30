# F2 resource and timing estimate (VU47P)

**The only Vivado result for this design on the VU47P is one out-of-context synthesis of
`cl_otpu_core`** (section 5: Vivado 2025.2, no shell, no place and route, the HBM adapter not
included). There is no place-and-route, post-route timing, power or HDK-build result. Vivado was
not available in the authoring environment; the synthesis was run by the owner on an AWS FPGA
Developer AMI. This document separates kinds of information by how they were obtained.

| Kind | Source | Trust |
| --- | --- | --- |
| Upstream-reported (Kintex-7) | upstream's own Vivado 2026.1 runs of the frozen block, quoted from `third_party/opentpu/docs/board.md` | their measurement of a different device; not ours |
| Yosys generic mapping | our runs of Yosys 0.33 `synth_xilinx -family xcup` on the new blocks (`f2/tools/yosys_resources.py`) | weak: generic mapping, no timing, no DSP/BRAM/URAM decisions |
| Estimate | arithmetic on the two rows above and on device capacity | an estimate |
| Unavailable | Vivado synthesis / place-and-route / timing / power on VU47P | not run |

## 1. The frozen block (upstream-reported, Kintex-7 xc7k480t, 298.6K LUT)

Upstream's numbers for `otpu_board` with `D = 128` and the personalities' relevant options
(their build system, Vivado 2026.1; the design also contains their memory controllers and PCIe
bridge, which our wrapper does not):

| Build | Core clock | LUT | FF | BRAM36 | DSP48 | Timing |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default, MCOLS=2 | 100 MHz | 187,852 (62.9%) | 126,679 | 635 | 267 | met, WNS +0.082 ns |
| MCOLS=4, VPU_CL=4 | 100 MHz | 211,629 (70.9%) | 144,722 | 668 | 443 | met, WNS +0.003 ns |
| Production, MCOLS=2, 4-bit MXU with PAIR | 120.755 MHz | 167.6K (56.1%) | 133.4K | 561 | not quoted here | met, WNS +0.149 ns |
| 4-bit-weight image | 125.49 MHz | - | - | - | - | met, WNS +0.067 ns (with a known RDOT fault; not production) |

`LANES = 16` did not route at 212K placed LUT on the Kintex-7 (`wide_dram.md`, upstream), so the
`compute` personality (MCOLS=8, LANES=16) is not estimated here at all.

## 2. New blocks, Yosys generic mapping (a tool run, but a weak one)

`python f2/tools/yosys_resources.py` (Yosys 0.33, UltraScale+ family cell library, flattened,
no timing, no DSP/BRAM inference). The frozen core cannot be read by this Yosys (SystemVerilog
constructs), so it is not included.

| Block | LUT | FF | RAM32M16 cells | MUXF | CARRY |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f2_async_fifo`, 289 bits x 16 (one crossing) | 18 | 38 | 21 | 0 | 4 |
| `f2_hbm_router`, core channel (2 PCs, 1-bit ID) | 1349 | 393 | 3 | 34 | 90 |
| `f2_hbm_router`, PCIS (4 PCs, 16-bit ID, group select) | 1387 | 425 | 6 | 36 | 90 |
| `f2_hbm_pc_bridge` (one PC, two sources) | 586 | 437 | 97 | 10 | 58 |
| `f2_ocl` (OCL front end, F2 registers, crossings) | 268 | 326 | 7 | 36 | 20 |
| **`f2_hbm_adapter`, PCS_PER_CH=2 (4 PCs)** | **6,402** | **2,959** | **400** | 56 | 496 |
| `f2_hbm_adapter`, PCS_PER_CH=4 (8 PCs) | 9,319 | 4,707 | 788 | 133 | 728 |

A first version of the bridge selected between its two sources with variable part-selects and
mapped to 6,310 LUT and 5,448 MUXF per PC (25,012 LUT for four PCs); rewriting the selection as
explicit two-way multiplexers brought it to 586 LUT per PC. The Yosys run found that; the
change is covered by the same simulation tests. Vivado may map either form differently.

## 3. Estimate for the VU47P (arithmetic, not a result)

Device capacity for the percentages below was first **assumed** from memory (about 1.3 M LUT,
2.6 M FF, 2,160 BRAM36, 960 URAM, 9,024 DSP48E2). Vivado's own report for `xcvu47p-fsvh2892-2-e`
later gave 1,303,680 LUT, 2,607,360 FF, **2,016** block RAM tiles, 960 URAM and 9,024 DSP; the
BRAM figure above was wrong. The estimates in this section were written before the synthesis run
and are kept for comparison; the measured numbers are in section 5. The
Small Shell occupies part of the device, the amount is not stated in the HDK files read; the
HDK text says the top SLR is fully available to the CL.

| Item | LUT | FF | Basis |
| --- | ---: | ---: | --- |
| Frozen core, MCOLS=2 | 170K-190K | 127K-133K | upstream rows above (their build also has memory/PCIe logic, so this is if anything high) |
| Frozen core, MCOLS=4 | about 210K | about 145K | upstream row |
| Adapter, 4 PCs | about 6.4K | about 3.0K | Yosys, section 2 |
| OCL and registers | about 0.3K | about 0.3K | Yosys |
| **Total, MCOLS=2** | **about 180K-200K (about 14-15% of an assumed 1.3 M)** | about 130K-140K | |

BRAM/URAM/DSP: upstream's 560-635 BRAM36 and 267-283 DSP48 (E1) are the starting point; UltraScale+
DSP48E2 and URAM change how the tools map them and no estimate is claimed beyond "the same
order of magnitude". These are far below the device capacity; area is not the expected problem,
timing and routing are.

## 4. Timing outlook (estimate, not a result)

| Path | Clock | Outlook |
| --- | --- | --- |
| Core (frozen block) | 125 MHz plan | Upstream closes 120.8-125.5 MHz on Kintex-7; UltraScale+ is faster, so 125 MHz is plausible. 250 MHz is unsupported by any evidence. |
| Async FIFOs, gray pointers | 125/250/450 MHz pairs | Standard structure; the gray-pointer `set_max_delay` exceptions in `f2/vivado/cl_timing_user.xdc` need checking with `report_cdc`. |
| HBM-side bridge request path | 450 MHz plan | The FIFO empty flag -> arbiter -> PC `valid` path is several logic levels and is not pipelined for 450 MHz. Likely to need pipelining, or the 300 MHz / 8-PC fallback in ADR-0008 section 4. |
| SLR crossings | all | HBM is in the bottom region, PCIS in the SLR1/top region [HDK]; the adapter's bridges and the core are not yet floorplanned. Register stages on both sides of every crossing are required [HDK]. |
| PCIS | 250 MHz | Per-burst arbitration bounds host latency; the shell's 8 us PCIS timeout is not at risk from the core, but is not verified in hardware. |

## 5. Results table (filled only from Vivado reports)

| Run | Tool / version | LUT | FF | Block RAM tiles | URAM | DSP | WNS (core / main / HBM) | Power | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| OOC synthesis, `cl_otpu_core`, PCS=2, MCOLS=2, LANES=8, core 125 MHz | Vivado 2025.2 (build 6299465), `xcvu47p-fsvh2892-2-e` | 116,080 (8.90%) | 103,202 (3.96%) | 496 (24.60%) | 0 | 283 (3.14%) | +0.467 ns at the core clock (see below) | - | **measured, synthesis only** |
| OOC place and route, same configuration, fixed XDC | Vivado 2025.2 (build 6299465), `xcvu47p-fsvh2892-2-e` | 112,545 (8.63%) | 101,669 (3.90%) | 496 (24.60%) | 0 | 283 (3.14%) | +0.221 ns at the core clock (hold +0.014 ns) | 5.157 W (default activity, medium confidence) | **measured, core only, no shell** |
| HDK build (post-route DCP) | - | - | - | - | - | - | - | - | **unavailable** |

Run on 2026-09-30 by the owner on an AWS FPGA Developer AMI 1.19.2 (Ubuntu 24.04), instance
`m7a.4xlarge`, about 12 minutes of synthesis (`f2/vivado/run_ooc.sh ... pcs=2 core_ns=8.0`). Numbers
are copied from Vivado's `utilization_synth.rpt`, `report_utilization`, `timing_synth.rpt` and
`report_clocks` output; the raw report files were on the instance and are not in the repository.

What the synthesis run says:

- Block RAM tiles are 470 RAMB36 + 52 RAMB18; LUT use is 105,328 logic + 8,131 LUTRAM + 2,621
  SRL. Largest blocks by LUT: `otpu_vpu` about 28K, `otpu_axi_dram` about 17K, `otpu_quant` about
  17K, `otpu_mxu` about 17K, `otpu_seq` about 17K.
- Against the Kintex-7 based estimate in section 3: LUT (116K vs 170K-190K) and FF (103K vs
  127K-133K) came out well below it; DSP (283) is at the top of the 267-283 range; Block RAM
  (496 tiles) is below the upstream 560-635 BRAM36.
- Timing: clocks confirmed by `report_clocks` as `clk_core` 8.000 ns (125 MHz), `clk_main`
  4.000 ns, `clk_hbm` 2.222 ns. Setup WNS +0.467 ns, TNS 0, 0 of 341,840 endpoints failing; hold
  WHS +0.014 ns, 0 failing; "All user specified timing constraints are met". **This is a
  post-synthesis result with no placement or routing delay, and is optimistic.** It is the core
  clock domain only: the core is the only logic in the run, so the 250 MHz and 450 MHz domains
  have no timed paths.
- Not covered by this run: the HBM adapter (Yosys, section 2: about 6.4K LUT, 3.0K FF, which
  would bring the total to about 122K LUT), the 450 MHz HBM-side bridge paths, the shell, SLR
  crossings, power.
- One critical warning (reported twice in the log's summary): `ooc_clocks.xdc:5`, an `if` that
  XDC does not support. It was a redundant fallback for an unset `core_ns`; the clock periods
  were verified as above, and the line is removed. The script was otherwise run unchanged on
  first contact. Other tool warnings (412 in synthesis) have not been reviewed here.

### Place and route (same instance, same day)

Run with `f2/vivado/run_ooc.sh build/f2-vivado/ooc-pcs2-impl pcs=2 core_ns=8.0 impl=1` on the
repository at commit `fc7d2ef` (XDC fixed); `route_design` took about 5 minutes elapsed. The log
reports 0 errors and 0 critical warnings in every step.

- Routed setup WNS **+0.221 ns**, TNS 0, 0 of 337,798 endpoints failing; hold WHS +0.014 ns, 0
  failing; pulse-width WPWS +0.579 ns, 0 failing; "All user specified timing constraints are
  met". Setup slack fell from +0.467 ns after synthesis to +0.221 ns after routing, which is
  about 2.8% of the 8 ns period.
- **What this supports:** the frozen core places and routes at 125 MHz on the VU47P when it is
  the only thing in the design. **What it does not support:** 125 MHz as a safe margin in the real
  design (there is no shell, no HBM adapter, no 450 MHz HBM-side paths, and no SLR crossings in
  this run), or any higher clock; 250 MHz remains unsupported by any evidence.
- Routed utilization is slightly below synthesis (112,545 LUT, 101,669 FF; block RAM and DSP
  unchanged), so the LUT/FF comparison with the Kintex-7 estimate in section 3 holds.
- Power: total on-chip 5.157 W = 1.842 W dynamic + 3.315 W device static, confidence "Medium".
  This is Vivado's estimate with default switching activity and no real toggle data; it is not a
  card-power figure, and the static part is the device's.
- The flat utilization report also contains a second table (25.60% LUT, 11.56% FF, 73.81% block
  RAM, 9.83% DSP) whose column headings were not captured. It **appears to be a per-SLR view**
  (the core placed in one SLR); if so, block RAM is the resource to watch when the adapter and
  shell are added. This is an interpretation, not a checked fact.

Procedure: `f2/vivado/README.md`. Still unavailable: a run that includes the HBM adapter and the
450 MHz clock, and the HDK build.
