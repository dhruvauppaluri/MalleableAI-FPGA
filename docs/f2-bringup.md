# F2 bring-up plan

Cheapest tests first. **Nothing may be started without the owner's explicit approval of each
paid step.** Status: stage 0 is done, and stages 1 and 2 were run by the owner on 2026-09-30 on one CPU-only
build instance (about 12 minutes of synthesis, about 5 minutes of routing; results in
`docs/f2-resource-timing.md` section 5; the actual AWS charge has not been reported to this
document). Stages 1 and 2 covered the CL wrapper (core, HBM adapter, OCL) with the three clocks, but
no shell and no HBM IP. No AFI has
been built or submitted, no F2 instance has been used, and no other stage has been started.

Everything below that names a price, a duration or a quota is an **assumption to verify**
(AWS pricing page, service quotas, the HDK docs in the release you use). AWS facts marked
[HDK] come from `aws/aws-fpga` release 2.3.4 (`b603a81`) as read on 2026-09-29.

## Stages

Each stage lists what it proves, what it needs, the go/stop criterion, and the paid part.

| # | Stage | Proves | Needs | Go when | Cost (estimate) |
| --- | --- | --- | --- | --- | --- |
| 0 | Local simulation | RTL, adapter, host driver, byte equality on the tiny test | this repo | done in the F2 PRs | none |
| 1 | **DONE 2026-09-30** (`m7a.4xlarge`, AMI 1.19.2, Vivado 2025.2): 116,080 LUT, 103,202 FF, 496 BRAM tiles, 283 DSP, post-synthesis WNS +0.467 ns at 125 MHz; one critical warning (removed). **Out-of-context synthesis of `cl_otpu_core` on the AWS developer AMI** | The frozen core maps to UltraScale+; LUT/FF/BRAM/URAM/DSP use; synthesis timing; unsupported constructs surface | one CPU-only x86 instance (4+ vCPU, 32+ GiB; 8 vCPU / 64 GiB recommended), FPGA Developer AMI, `f2/vivado/run_ooc.sh` | no synthesis errors; utilization leaves room beside the shell; no unexplained critical warnings | 2-4 h x build-instance rate |
| 2 | **DONE 2026-09-30 for the CL wrapper without shell or HBM IP** (routed: 112,545 LUT, 101,669 FF, 496 BRAM tiles, 283 DSP; setup WNS core +0.658 ns at 125 MHz, HBM +0.221 ns at 450 MHz, main +0.457 ns at 250 MHz; hold >= +0.014 ns; 0 critical warnings). Out-of-context place and route (`impl=1`) | Timing at 125 MHz core / 450 MHz HBM side without the shell | same instance | WNS >= 0 at 125 MHz core; the 450 MHz paths (bridge, async FIFO read side) either close or the fallback in ADR-0008 section 4 is chosen | 3-8 h x rate |
| 3 | AWS hello-world AFI from the HDK (`cl_axil_reg_access`) on an F2 instance | The account/AFI flow works end to end (quota, IAM, S3, `create-fpga-image`, load, PCIe access) independent of our RTL | quotas, S3 buckets, IAM role, an F2 instance | AFI `available`; loads; the example's register test passes | build 1-2 h + F2 instance about 1 h |
| 4 | Our CL build (DCP) and AFI | The design fits the Small Shell and closes timing with the HDK's flow | stage 2 passed; `f2/vivado/setup_cl.sh`; HDK build | post-route DCP with no timing violation (a violated DCP is "test only" [HDK]) | 3-8 h x rate per iteration; plan several iterations |
| 5 | Register test on the card | `F2_ID`, board `ID`, `STATUS`, scratch registers, HBM-ready | F2 instance, AFI | `python -m malleable.f2 descriptor --mode hardware ...` first checks pass | short F2 session |
| 6 | DMA loopback both channels (BAR4) | Data integrity through PCIS -> adapter -> HBM -> back; the real throughput of CPU stores/loads through BAR4 (assumption A5) | same | `loopback --mode hardware` all verified; throughput recorded (measured) | short |
| 7 | HBM bandwidth and core-side traffic | The assumed 60% efficiency (A1); `MXU_STARVE`; per-PC balance | same | measured margin over the 16 GB/s core demand at 125 MHz | short |
| 8 | Descriptor path on the card | Program load, RUN, HALT, counters | same | copy program result equals input | short |
| 9 | Tiny 100-token test with per-step DRAM equality vs the ISA machine | Correctness of the whole path on hardware | tiny model only (no weights) | 100/100 steps byte-equal | short |
| 10 | Qwen3-0.6B, short prompt + 8 tokens, per-step check | The acceptance the local release requires, on hardware | model weights on the instance | matches the recorded acceptance behaviour | F2 session hours |

Stage 1 was the first paid step, chosen because it is the cheapest test that can change the plan.
The core was built for a Kintex-7 (`docs/f2-resource-timing.md`); it needed no F2 instance and no
shell. It synthesized on UltraScale+ without errors, used under 9% of the LUTs and about a quarter
of the block RAM tiles, and met 125 MHz after synthesis, so the plan was not changed. That timing
is optimistic (no placement or routing).
Stage 2 then placed and routed the wrapper and closed all three clocks: core 125 MHz with +0.658 ns
of setup slack, the adapter at 450 MHz with +0.221 ns, main 250 MHz with +0.457 ns. (An earlier
version of this page wrongly said the adapter and the 450 MHz paths were not in the run.) The
open questions are the ones this run cannot test: paths to the real HBM IP pins, SLR crossings,
the floorplan and the shell. They need the HDK build (stage 4), which has not been started.

## AFI build steps (stages 3-4) [HDK]

1. Developer machine: launch the FPGA Developer AMI (x86 only, 4+ vCPU / 32+ GiB; compute or
   memory optimized). Clone `aws/aws-fpga`; `source hdk_setup.sh` (about 2 min the first time).
2. `export AWS_FPGA_REPO_DIR=...; f2/vivado/setup_cl.sh` (creates `cl_otpu` from the HDK's
   `CL_TEMPLATE`, overlays our files, copies the HDK's own `cl_mem_hbm_wrapper.sv`).
3. `export CL_DIR=$AWS_FPGA_REPO_DIR/hdk/cl/examples/cl_otpu; cd $CL_DIR/build/scripts;
   ./aws_build_dcp_from_cl.py -c cl_otpu --aws_clk_gen --clock_recipe_a A1 --clock_recipe_hbm H2`
   (`--no-encrypt` eases debugging). The HDK examples take 30-90 minutes; this design is larger.
4. Artifacts: `$CL_DIR/build/checkpoints/*.Developer_CL.tar`, reports in `build/reports/`.
   Read the timing summary first. A `VIOLATED` DCP is for testing only.
5. **Submit only with approval.** Create S3 buckets (DCP and logs), upload the tarball,
   `aws ec2 create-fpga-image --input-storage-location ... --logs-storage-location ...`,
   then `wait_for_afi.py` or `aws ec2 describe-fpga-images` until `available` (can take hours).
6. On the F2 instance: `sudo fpga-load-local-image -S 0 -I <agfi-id>`,
   `sudo fpga-describe-local-image -S 0`.
7. **Clock/reset release.** Because the design instantiates `AWS_CLK_GEN`, the runtime software
   must release the generated resets after the MMCMs lock (`aws_clkgen_deassert_resets(slot)`
   from the SDK; poll `MMCM_LOCK_REG` for `0x151`) before the HBM AXI domain leaves reset
   [HDK, AWS_CLK_GEN spec]. Until then `F2_STATUS.hbm_ready` reads 0 and board registers
   answer SLVERR/`0xDEADBEEF`. Add this step to the host bring-up script before stage 5.
8. Teardown: stop or terminate the instance; delete the buckets/AFIs you no longer need.

## Cost estimate (formulas; assumptions, not quotes)

`cost = hours x hourly rate`. Rates below are **assumptions to check on the AWS pricing page
for your region**, not quotes; none was looked up from AWS.

| Item | Assumed rate | Hours per attempt | Attempts | Range |
| --- | ---: | ---: | ---: | ---: |
| Build/synth instance (8 vCPU / 64 GiB class, developer AMI) | about US$0.8-1.5 / h | stage 1: 2-4; stage 2: 3-8; stage 4: 3-8 | 1 / 1-2 / 2-5 | about US$3-4, 3-12, 5-60 |
| F2 instance (`f2.6xlarge`, 1 FPGA) | about US$1.5-2 / h [ASSUMPTION] | stage 3: 1; stages 5-9: 2-4 | 1 / 1-3 | about US$2, 4-24 |
| F2 instance for stage 10 (model download, several runs) | same | 3-8 | 1-2 | about US$5-30 |
| S3 storage, EBS volumes, data transfer | small | - | - | under US$5 in total |
| AFI creation | no direct charge assumed [ASSUMPTION] | - | - | 0 |

Rough total to reach stage 9: **US$20-130**, dominated by the number of build iterations
(timing closure is the uncertainty). Stage 10 adds US$5-30. Suggested approval structure:
approve stage 1 alone (about US$5); decide the rest on its reports; set a monetary cap per
stage and an instance auto-stop.

Quotas: F2 instances are quota-limited per region (a vCPU quota for F instances); request
before stage 3 (no charge, but it takes time). Region must offer F2 (check the AWS instance
availability list).

## Test assets already in this repository

| Stage | Command | Runs today without hardware |
| --- | --- | --- |
| 5-6 | `python -m malleable.f2 loopback` / `descriptor` (`--mode hardware --enable-hardware --bdf <slot BDF>`) | `--mode emulate` and `--mode sim` |
| 9 | `python -m malleable.f2 steps` | emulate/sim |
| 9 (recorded) | `python -m malleable.f2.replay` | simulation only |

The hardware transport (`malleable/f2/transport.py: F2BarTransport`) has never been run
against a card; expect first-contact fixes (BAR discovery, mapping sizes, write combining).

## Assumption ledger additions

| # | Assumption | Resolved by |
| --- | --- | --- |
| A8 | HBM ECC is not enabled (or is transparent) so `Board.scrub` is unnecessary; a read of never-written HBM may otherwise fault (the original board's DDR3 does). HDK errata: HBM ECC scrubbing / "Initialize Memory Using ECC" are **not supported in the Small Shell** and enabling them makes the HBM monitor time out at AFI load, so the design must not enable them (`cl_otpu.sv` uses the HDK's own wrapper unchanged) | stage 6: read before write |
| A9 | The AFI's PCIe IDs (`cl_id_defines.vh`: template default `0xF010/0x1D0F`, subsystem `0x1D51/0xFEDC`) are acceptable for a private AFI | AFI creation |
| A10 | F2 prices, quotas and region availability as above | AWS pricing/quotas pages |

## HDK checks done locally (2026-09-30, no AWS, against the `aws-fpga` clone at `b603a81`, RC 2.3.4)

These are reading and lint checks against the real HDK files; none of them is a build.

| Check | Result |
| --- | --- |
| A2: HBM address format `{pc[4:0], local[28:0]}` (`f2/hdk/cl_otpu.sv`) | **Consistent with the HDK.** `cl_mem_perf/README.md` (HBM Memory Address Space): channel 0 = `0x0000_0000`-`0x1FFF_FFFF`, channel 1 starts at `0x2000_0000`, "and so on" for 16 GB, i.e. 512 MiB (29 bits) per channel and the channel index in address bits [33:29]. The HDK wrapper passes AXI port *n* straight to HBM port *n* (`HBM_STACK_MAPPING`). Not verified: behaviour of the HBM IP on real hardware. |
| Port names of `cl_mem_hbm_wrapper` and `aws_clk_gen` used by `f2/hdk/cl_otpu.sv` | Scripted check against the real module declarations: every connection names a real port or parameter, and every real port is connected (0 mismatches; the parser's 9 false alarms were interface/parameter names, checked by hand). Port *widths* and directions are covered only by the lint below, which still stubs these two modules. |
| `make lint-f2-hdk` with the real `cl_ports.vh` / `cl_id_defines.vh` | Runs without errors (Verilator 5.050, `-Wno-fatal` with the warning classes listed in the Makefile). The two HDK modules remain stubs, so this is not a check of the real IP. |
| Clock recipes | `Clock_Recipes_User_Guide.md`: **A1** = `clk_main_a0` 250 MHz, `clk_extra_a1` 125 MHz; **H2** = `clk_hbm_axi` 450 MHz. Both match ADR-0008. `aws_build_dcp_from_cl.py` has `--aws_clk_gen`, `--clock_recipe_a`, `--clock_recipe_b`, `--clock_recipe_c`, `--clock_recipe_hbm` (the build command in this file uses `--aws_clk_gen --clock_recipe_a A1 --clock_recipe_hbm H2`). |
| Reset release | `AWS_CLK_GEN_spec.md`: with `AWS_CLK_GEN` instantiated, runtime software must call `aws_clkgen_deassert_resets(slot_id)` (`sdk/userspace/include/fpga_clkgen.h`) after AFI load. Step 7 of the AFI steps above already requires it; it is **not yet in the host bring-up script** (`malleable/f2/transport.py` has no such call). |
| A9: PCIe IDs | `CL_TEMPLATE/design/cl_id_defines.vh`: `CL_SH_ID0 = 32'hF010_1D0F`, `CL_SH_ID1 = 32'h1D51_FEDC`, as assumed. Whether AWS accepts them for a private AFI is decided at AFI creation. |

Open after these checks: the host-side reset release is missing from the bring-up script, and the
first real HDK build (stage 4) is the next thing that can fail.
