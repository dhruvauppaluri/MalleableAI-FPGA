#!/usr/bin/env bash
# Create an HDK CL directory for cl_otpu from this repository and a local clone of aws/aws-fpga.
#
# Run once as a dry run against a scratch copy of the HDK (2026-09-30; no Vivado). It only copies files into $AWS_FPGA_REPO_DIR/hdk/cl/examples/cl_otpu; it
# does not call AWS, does not start Vivado, and copies no AWS source into this repository. The
# HDK's own wrapper (cl_mem_hbm_wrapper.sv, ASL-licensed) is copied from your clone into the
# build directory only.
#
#   export AWS_FPGA_REPO_DIR=/path/to/aws-fpga      # a clone of https://github.com/aws/aws-fpga
#   f2/vivado/setup_cl.sh [python]
#   source $AWS_FPGA_REPO_DIR/hdk_setup.sh          # then, in the CL's build/scripts directory:
#   export CL_DIR=$AWS_FPGA_REPO_DIR/hdk/cl/examples/cl_otpu
#   cd $CL_DIR/build/scripts && ./aws_build_dcp_from_cl.py -c cl_otpu --aws_clk_gen \
#        --clock_recipe_a A1 --clock_recipe_hbm H2
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
py="$(command -v "${1:-python3}")" || { echo "python not found: ${1:-python3}" >&2; exit 2; }   # absolute: the script changes directory
: "${AWS_FPGA_REPO_DIR:?set AWS_FPGA_REPO_DIR to a clone of aws/aws-fpga}"
hdk="$AWS_FPGA_REPO_DIR/hdk"
cl="$hdk/cl/examples/cl_otpu"
[ -d "$hdk/cl/examples/CL_TEMPLATE" ] || { echo "not an HDK clone: $AWS_FPGA_REPO_DIR" >&2; exit 2; }
[ ! -e "$cl" ] || { echo "$cl exists; remove it first (this script never overwrites a CL directory)" >&2; exit 2; }

( cd "$hdk/cl/examples" && "$py" create_new_cl.py --new_cl_name cl_otpu )

# design sources, flat (the HDK's encrypt.tcl copies design/*.{v,sv,vh,svh,inc} without recursion)
rm -f "$cl/design/cl_otpu.sv" "$cl/design/cl_otpu_defines.vh"
( cd "$repo" && PYTHONPATH=. "$py" -m malleable.f2.sim --sources --rtl-only ) | tr ' ' '\n' | while read -r f; do
  [ -n "$f" ] && cp "$f" "$cl/design/"
done
cp "$repo/f2/hdk/cl_otpu.sv" "$cl/design/cl_otpu.sv"
# the HDK example's HBM wrapper (needed by cl_otpu.sv) comes from the developer's own clone
cp "$hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv" "$cl/design/"

# build scripts and constraints
cp "$repo/f2/vivado/synth_cl_otpu.tcl" "$cl/build/scripts/synth_cl_otpu.tcl"
rm -f "$cl/build/scripts/synth_cl_otpu.tcl.orig"
cp "$repo/f2/vivado/cl_synth_user.xdc"  "$cl/build/constraints/cl_synth_user.xdc"
cp "$repo/f2/vivado/cl_timing_user.xdc" "$cl/build/constraints/cl_timing_user.xdc"
# small_shell_cl_pnr_user.xdc keeps the template's (empty) content until a floorplan exists

echo "created $cl"
echo "Check that every file the HBM wrapper needs from the HDK library is on the include path at first synthesis."
