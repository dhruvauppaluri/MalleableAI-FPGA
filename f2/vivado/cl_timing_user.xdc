# CL-specific timing exceptions for the F2 build of cl_otpu (f2/hdk/cl_otpu.sv).
# UNTESTED: never read by Vivado. Cell-name patterns follow the RTL names in f2/rtl and must be
# checked against the synthesized netlist at the first build (report_cdc, report_methodology).
#
# Clocks: the shell provides clk_main_a0 (250 MHz); AWS_CLK_GEN provides clk_extra_a1 (core,
# 125 MHz with recipe A1) and clk_hbm_axi (450 MHz with recipe H2). The HDK's common
# constraints already declare them and their relationships; do not redeclare them here.

# Asynchronous FIFO pointers (gray coded): bound the crossing delay, do not time it as synchronous.
# f2_async_fifo: write pointer wgray -> read-side synchronizer wgray_r1; read pointer rgray -> write-side rgray_w1.
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {NAME =~ *wgray_reg[*]}] \
              -to [get_cells -hierarchical -filter {NAME =~ *wgray_r1_reg[*]}] 4.000
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {NAME =~ *rgray_reg[*]}] \
              -to [get_cells -hierarchical -filter {NAME =~ *rgray_w1_reg[*]}] 4.000

# f2_pulse_sync toggle -> first synchronizer flop
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {NAME =~ *u_ces/tog_reg}] \
              -to [get_cells -hierarchical -filter {NAME =~ *u_ces/d1_reg}] 4.000
