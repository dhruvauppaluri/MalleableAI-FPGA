# Small Shell floorplan for the F2 build of cl_otpu.
# UNTESTED and deliberately not applied. The HDK's examples (cl_dram_hbm_dma, cl_mem_perf) put
# the CL in child pblocks of pblock_CL on SLR1/SLR2 (top), leave HBM-facing pipeline stages
# unconstrained across SLRs, and pipeline every SLR crossing. This design needs the same treatment:
#   - the PCIS side of the adapter (f2_hbm_adapter.u_pcis, f2_ocl) next to the shell's PCIS/OCL ports
#   - the accelerator (otpu_board) in one SLR
#   - the per-PC bridges (f2_hbm_pc_bridge) on the HBM side, with register stages on each SLR crossing
# Start from the HDK example's small_shell_cl_pnr_user.xdc for the clock-region ranges, then
# assign these cells. Do not enable it before the first timing report shows what is needed.
