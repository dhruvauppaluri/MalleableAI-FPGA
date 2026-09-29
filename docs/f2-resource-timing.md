# F2 resource and timing estimate (VU47P)

**There are no Vivado results for this design on the VU47P.** Vivado is not available in the
authoring environment and no tool run on a VU47P has happened. This document separates four
kinds of information; only the first two are results of a tool run, and neither is a
Vivado-on-VU47P result.

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

Device capacity used for percentages (**[ASSUMPTION]**, recalled from the AMD Virtex UltraScale+
product table and not verifiable from the sources available here; check DS923 before relying
on the percentages): about 1.3 M LUT, 2.6 M FF, 2,160 BRAM36, 960 URAM, 9,024 DSP48E2. The
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

## 5. Results table (to be filled only from Vivado reports)

| Run | Tool / version | LUT | FF | BRAM36 | URAM | DSP | WNS (core / main / HBM) | Power | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| OOC synthesis, PCS=2, MCOLS=2 | - | - | - | - | - | - | - | - | **unavailable** |
| OOC place and route | - | - | - | - | - | - | - | - | **unavailable** |
| HDK build (post-route DCP) | - | - | - | - | - | - | - | - | **unavailable** |

Procedure: `f2/vivado/README.md`. The first paid step that can fill these rows is the
out-of-context synthesis in `docs/f2-bringup.md` stage 1, subject to the owner's approval.
