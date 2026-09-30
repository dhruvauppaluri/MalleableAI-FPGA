# Clock constraints for the out-of-context runs (f2/vivado/ooc_synth.tcl).
# Run once through Vivado 2025.2 (OOC synthesis, docs/f2-resource-timing.md). Periods are the ADR-0008
# plan: main 250 MHz, core 125 MHz (first bring-up), HBM AXI 450 MHz. ooc_synth.tcl must set $core_ns
# from its arguments before reading this file (XDC does not support `if`, so there is no default here).
create_clock -name clk_main -period 4.000 [get_ports clk_main]
create_clock -name clk_core -period $core_ns [get_ports clk_core]
create_clock -name clk_hbm  -period 2.222 [get_ports clk_hbm]

# The three domains are unrelated. The crossings are asynchronous FIFOs (gray-coded pointers,
# two-flop synchronizers) and toggle-synchronized pulses; their gray-pointer paths are bounded
# with set_max_delay -datapath_only in cl_timing_user.xdc so they are still timed for skew.
set_clock_groups -asynchronous -group [get_clocks clk_main] -group [get_clocks clk_core] -group [get_clocks clk_hbm]
