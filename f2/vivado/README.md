# Vivado scripts for the F2 platform

**Only the out-of-context flow has been run**: `run_ooc.sh`, `ooc_synth.tcl` and `ooc_clocks.xdc`
ran on Vivado 2025.2 (synthesis and, with `impl=1`, place and route of the core alone, 2026-09-30,
see `docs/f2-resource-timing.md`); the only fix needed was removing an `if` from the XDC.
**Everything else here is UNTESTED**: `setup_cl.sh`, `synth_cl_otpu.tcl`, the HDK-flow constraints
and the HDK-facing top were never run. Their checks are shell/Tcl syntax balance and a Verilator
port-connection lint against stub modules. Expect first-contact fixes.

Nothing here calls AWS, starts a paid resource, or copies AWS source into this repository.
Running any of it on AWS needs the owner's approval (`docs/f2-bringup.md`).

| File | Purpose | Needs |
| --- | --- | --- |
| `ooc_synth.tcl`, `ooc_clocks.xdc`, `run_ooc.sh` | Out-of-context synthesis (and optionally place and route) of `cl_otpu_core` for `xcvu47p-fsvh2892-2-e`, no shell, no HDK. **The recommended first paid step.** | Vivado with xcvu47p support (an FPGA Developer AMI has it) |
| `setup_cl.sh` | Creates `cl_otpu` in an `aws/aws-fpga` clone from `CL_TEMPLATE`, overlays our design files and the scripts below, and copies the HDK's `cl_mem_hbm_wrapper.sv` from your clone | a local HDK clone (`AWS_FPGA_REPO_DIR`) |
| `synth_cl_otpu.tcl` | The CL's synthesis script in the HDK's structure (sources the HDK's common header/footer) | HDK build environment |
| `cl_timing_user.xdc`, `cl_synth_user.xdc`, `small_shell_cl_pnr_user.xdc` | CL timing exceptions (gray-pointer `set_max_delay -datapath_only`), synthesis hints, and a floorplan *template* (not applied) | HDK build environment |
| `lint_stubs.sv` | Stand-ins for HDK modules so `make lint-f2-hdk` can check `f2/hdk/cl_otpu.sv` against `cl_ports.vh` | an HDK clone for `cl_ports.vh` |

The HDK-facing top level is `f2/hdk/cl_otpu.sv`.

## Out-of-context run (no AWS shell)

```sh
f2/vivado/run_ooc.sh build/f2-vivado/ooc-pcs2 pcs=2 core_ns=8.0            # synthesis
f2/vivado/run_ooc.sh build/f2-vivado/ooc-pcs2-impl pcs=2 core_ns=8.0 impl=1 # + place and route
```

It writes Vivado's own reports (`utilization_synth.rpt`, `timing_synth.rpt`,
`clock_interaction.rpt`, `cdc.rpt`, and for `impl=1` the implementation reports and power).
Record the numbers in `docs/f2-resource-timing.md` only from those reports, with the Vivado
version, and mark them "OOC, no shell". OOC timing is an estimate: the shell, the HBM IP,
the floorplan and the real clock network are absent.

## HDK build (needs AWS approval to go beyond a local build)

```sh
export AWS_FPGA_REPO_DIR=/path/to/aws-fpga
f2/vivado/setup_cl.sh
source $AWS_FPGA_REPO_DIR/hdk_setup.sh
export CL_DIR=$AWS_FPGA_REPO_DIR/hdk/cl/examples/cl_otpu
cd $CL_DIR/build/scripts
./aws_build_dcp_from_cl.py -c cl_otpu --aws_clk_gen --clock_recipe_a A1 --clock_recipe_hbm H2
```

Submitting the resulting DCP (S3 upload, `aws ec2 create-fpga-image`) is a separate, paid,
externally visible step: see `docs/f2-bringup.md`.

## Known gaps

- `small_shell_cl_pnr_user.xdc` is a description, not constraints. The Small Shell floorplan
  has to be taken from the HDK examples and the CL assigned to SLRs after a first timing report.
- The gray-pointer `set_max_delay` patterns assume Vivado's flattened register names
  (`*wgray_reg[*]`, `*wgray_r1_reg[*]`, ...); check them with `report_cdc` and
  `report_methodology` at the first build.
- `cl_otpu.sv` connects the HDK's `cl_mem_hbm_wrapper` with a 34-bit HBM address of
  `{pc[4:0], local[28:0]}` (assumption A2 in ADR-0008); confirm against the HBM IP
  configuration before the first build.
- No ILA/VIO/virtual-JTAG debug is wired.
