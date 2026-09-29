# Out-of-context synthesis (and optional implementation) of cl_otpu_core for the F2 device.
#
# UNTESTED: this script has never been run (no Vivado in the authoring environment). It needs
# a Vivado release with xcvu47p support (an FPGA Developer AMI has one; the part needs a
# licensed edition), and no AWS shell, no HDK and no AWS access: it builds only the design
# verified in simulation, with the frozen OpenTPU sources unchanged.
#
#   vivado -mode batch -source f2/vivado/ooc_synth.tcl -tclargs <out_dir> [key=value ...]
#     keys: pcs=2  core_ns=8.0  mcols=2  lanes=8  impl=0|1  jobs=8
#
# Results (all reports are Vivado's own; nothing is summarized by this script):
#   <out_dir>/utilization_synth.rpt, timing_synth.rpt, clock_interaction.rpt, cdc.rpt,
#   design_analysis_synth.rpt, cl_otpu_core_synth.dcp; with impl=1 also *_impl reports and dcp.
#
# Out-of-context numbers are a de-risking estimate only: the shell, HBM IP, floorplan and the
# real clock network are absent.

set script_dir [file dirname [file normalize [info script]]]
set repo [file normalize $script_dir/../..]
set out_dir [file normalize [lindex $argv 0]]
array set opt {pcs 2 core_ns 8.0 mcols 2 lanes 8 impl 0 jobs 8}
foreach a [lrange $argv 1 end] {
  lassign [split $a =] k v
  if {![info exists opt($k)]} { error "unknown option $k" }
  set opt($k) $v
}
set part xcvu47p-fsvh2892-2-e
file mkdir $out_dir
set_param general.maxThreads $opt(jobs)

# ---- sources: the frozen upstream RTL (never edited) and ours
set up $repo/third_party/opentpu/rtl
set upstream [list \
  $up/vpu/otpu_fp.sv $up/vpu/otpu_fpipe.sv $up/top/otpu_pkg.sv $up/mem/otpu_dram.sv $up/mem/otpu_tmem.sv \
  $up/mem/otpu_axi_dram.sv $up/mem/otpu_actram.sv $up/seq/otpu_seq.sv $up/dma/otpu_dma.sv $up/mxu/otpu_mxu.sv \
  $up/vpu/otpu_quant.sv $up/vpu/otpu_vpu.sv $up/top/otpu_coll.sv $up/top/otpu_slice.sv \
  $up/boards/ypcb-00338/otpu_ctrl.sv $up/boards/ypcb-00338/otpu_trace.sv $up/boards/ypcb-00338/otpu_board.sv]
set ours {}
foreach f {f2_fifo f2_cdc f2_ocl f2_hbm_router f2_hbm_pc_bridge f2_hbm_adapter cl_otpu_core} {
  lappend ours $repo/f2/rtl/$f.sv
}
read_verilog -sv [concat $upstream $ours]

# ---- constraints
set core_ns $opt(core_ns)
read_xdc $script_dir/ooc_clocks.xdc
set_property PROCESSING_ORDER EARLY [get_files ooc_clocks.xdc]

# ---- synthesis
synth_design -mode out_of_context -top cl_otpu_core -part $part \
  -generic PCS_PER_CH=$opt(pcs) -generic MCOLS=$opt(mcols) -generic LANES=$opt(lanes) \
  -generic CORE_KHZ=[expr {int(1e6 / $opt(core_ns))}]
write_checkpoint -force $out_dir/cl_otpu_core_synth.dcp
report_utilization -hierarchical -file $out_dir/utilization_synth.rpt
report_timing_summary -delay_type min_max -max_paths 20 -file $out_dir/timing_synth.rpt
report_clock_interaction -file $out_dir/clock_interaction.rpt
report_cdc -file $out_dir/cdc.rpt
report_design_analysis -logic_level_distribution -file $out_dir/design_analysis_synth.rpt
report_high_fanout_nets -max_nets 30 -file $out_dir/fanout_synth.rpt

if {$opt(impl)} {
  opt_design
  place_design
  phys_opt_design
  route_design
  write_checkpoint -force $out_dir/cl_otpu_core_impl.dcp
  report_utilization -hierarchical -file $out_dir/utilization_impl.rpt
  report_timing_summary -delay_type min_max -max_paths 50 -file $out_dir/timing_impl.rpt
  report_design_analysis -congestion -file $out_dir/congestion_impl.rpt
  report_power -file $out_dir/power_impl.rpt
}
puts "F2 OOC run finished: $out_dir"
