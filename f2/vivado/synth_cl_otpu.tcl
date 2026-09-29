# Synthesis script for the F2 build of cl_otpu (AWS HDK, Small Shell).
#
# UNTESTED: never run through Vivado. It follows the structure of the HDK's CL_TEMPLATE script
# (common header and footer are AWS's, sourced from the developer's HDK clone) and reads only
# what cl_otpu needs: the HBM IP, the AWS clock generator's MMCM IP, AXI register slices and the
# SDA crossbar. It is copied to $CL_DIR/build/scripts/ by f2/vivado/setup_cl.sh.

# Common header
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

###############################################################################
print "Reading user source codes"
###############################################################################
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]

###############################################################################
print "Reading CL IP blocks"
###############################################################################

## DDR: the shell's sh_ddr must be instantiated even when the DIMM is unused (DDR_PRESENT = 0);
## with DDR_PRESENT = 0 the controller IP is not needed, but the HDK header reads the shell side.

## HBM (AWS example IP: 16 GiB, 32 AXI3 channels)
read_ip [ list \
  ${HDK_IP_SRC_DIR}/cl_hbm/cl_hbm.xci
]

## Clocking IPs used by AWS_CLK_GEN
read_ip [ list \
  $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/clk_mmcm_a/clk_mmcm_a.xci \
  $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/clk_mmcm_hbm/clk_mmcm_hbm.xci \
  $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_clk_axil_xbar/cl_clk_axil_xbar.xci
]

## AXI register slice IPs referenced by the HDK's HBM wrapper
read_ip [ list \
  ${HDK_IP_SRC_DIR}/axi_register_slice/axi_register_slice.xci \
  ${HDK_IP_SRC_DIR}/cl_axi3_256b_reg_slice/cl_axi3_256b_reg_slice.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter/cl_axi_clock_converter.xci
]

###############################################################################
print "Reading user constraints"
###############################################################################
read_xdc [ list \
  ${constraints_dir}/cl_synth_user.xdc \
  ${constraints_dir}/cl_timing_user.xdc
]
set_property PROCESSING_ORDER LATE [get_files cl_synth_user.xdc]
set_property PROCESSING_ORDER LATE [get_files cl_timing_user.xdc]

###############################################################################
print "Starting synthesizing customer design ${CL}"
###############################################################################
update_compile_order -fileset sources_1

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

# Common footer
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
