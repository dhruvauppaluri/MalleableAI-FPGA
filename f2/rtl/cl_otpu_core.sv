// F2 platform: custom-logic core for the frozen OpenTPU board block.
//
//   OCL (AXI-Lite, main clk)  --f2_ocl-->  otpu_board control regs (core clk)
//   PCIS (AXI4 512b, main clk) -+
//   otpu_board m0/m1 (core clk) +-> f2_hbm_adapter -> N x HBM pseudo-channel ports (HBM clk)
//
// The accelerator (otpu_board and everything below it) is instantiated unchanged. This
// module adds only clock/reset handling, the host-visible register block, the PCIS window
// and the HBM adapter. The AWS-facing top level (cl_otpu.sv, HDK port names) wraps this
// module; the simulation testbench drives it directly.
//
// UNTESTED on hardware; not run through Vivado.
module cl_otpu_core #(
  parameter int PCS_PER_CH  = 2,
  parameter int PC_AW       = 29,
  parameter int STRIPE      = 9,
  // board block (defaults as in the upstream board testbench)
  parameter int D           = 128,
  parameter int MCOLS       = 2,
  parameter int ACT_BLOCKS  = 128,
  parameter int TMEM_WORDS  = 1 << 16,
  parameter int IMEM_WORDS  = 1 << 15,
  parameter int LANES       = 8,
  parameter int VPU_CL      = (LANES >= 8) ? LANES / 4 : 1,
  parameter int ULANES      = LANES,
  parameter int WIN         = 16,
  parameter int FIFO_DEPTH  = 1024,
  parameter int RPB         = 8 * LANES,
  parameter int WPB         = 1,
  parameter int MXU_IMPL    = 0,
  parameter int CORE_KHZ    = 125000,
  parameter logic [31:0] BUILD_ID = 32'h0F2_0001,
  parameter int TRACE_DEPTH = 16384,
  parameter int TRACE_QD    = 32,
  parameter int PQ_WIN      = 1024,
  parameter int AXI_BL      = 8
) (
  input  logic         clk_main,
  input  logic         rst_main_n,           // asynchronous assert, active low
  input  logic         clk_core,
  input  logic         clk_hbm,
  input  logic         hbm_ready,            // main clock domain (HBM IP initialized)

  // ---- OCL AXI-Lite (main clock)
  input  logic [31:0]  ocl_awaddr,
  input  logic         ocl_awvalid,
  output logic         ocl_awready,
  input  logic [31:0]  ocl_wdata,
  input  logic [3:0]   ocl_wstrb,
  input  logic         ocl_wvalid,
  output logic         ocl_wready,
  output logic [1:0]   ocl_bresp,
  output logic         ocl_bvalid,
  input  logic         ocl_bready,
  input  logic [31:0]  ocl_araddr,
  input  logic         ocl_arvalid,
  output logic         ocl_arready,
  output logic [31:0]  ocl_rdata,
  output logic [1:0]   ocl_rresp,
  output logic         ocl_rvalid,
  input  logic         ocl_rready,

  // ---- PCIS AXI4 (main clock)
  input  logic [15:0]  pcis_awid,
  input  logic [63:0]  pcis_awaddr,
  input  logic [7:0]   pcis_awlen,
  input  logic         pcis_awvalid,
  output logic         pcis_awready,
  input  logic [511:0] pcis_wdata,
  input  logic [63:0]  pcis_wstrb,
  input  logic         pcis_wlast,
  input  logic         pcis_wvalid,
  output logic         pcis_wready,
  output logic [15:0]  pcis_bid,
  output logic [1:0]   pcis_bresp,
  output logic         pcis_bvalid,
  input  logic         pcis_bready,
  input  logic [15:0]  pcis_arid,
  input  logic [63:0]  pcis_araddr,
  input  logic [7:0]   pcis_arlen,
  input  logic         pcis_arvalid,
  output logic         pcis_arready,
  output logic [15:0]  pcis_rid,
  output logic [511:0] pcis_rdata,
  output logic [1:0]   pcis_rresp,
  output logic         pcis_rlast,
  output logic         pcis_rvalid,
  input  logic         pcis_rready,

  // ---- HBM pseudo-channel ports (HBM clock; PC-local addresses)
  output logic [2*PCS_PER_CH*PC_AW-1:0]  hbm_awaddr,
  output logic [2*PCS_PER_CH*4-1:0]      hbm_awlen,
  output logic [2*PCS_PER_CH-1:0]        hbm_awvalid,
  input  logic [2*PCS_PER_CH-1:0]        hbm_awready,
  output logic [2*PCS_PER_CH*256-1:0]    hbm_wdata,
  output logic [2*PCS_PER_CH*32-1:0]     hbm_wstrb,
  output logic [2*PCS_PER_CH-1:0]        hbm_wlast,
  output logic [2*PCS_PER_CH-1:0]        hbm_wvalid,
  input  logic [2*PCS_PER_CH-1:0]        hbm_wready,
  input  logic [2*PCS_PER_CH*2-1:0]      hbm_bresp,
  input  logic [2*PCS_PER_CH-1:0]        hbm_bvalid,
  output logic [2*PCS_PER_CH-1:0]        hbm_bready,
  output logic [2*PCS_PER_CH*PC_AW-1:0]  hbm_araddr,
  output logic [2*PCS_PER_CH*4-1:0]      hbm_arlen,
  output logic [2*PCS_PER_CH-1:0]        hbm_arvalid,
  input  logic [2*PCS_PER_CH-1:0]        hbm_arready,
  input  logic [2*PCS_PER_CH*256-1:0]    hbm_rdata,
  input  logic [2*PCS_PER_CH*2-1:0]      hbm_rresp,
  input  logic [2*PCS_PER_CH-1:0]        hbm_rlast,
  input  logic [2*PCS_PER_CH-1:0]        hbm_rvalid,
  output logic [2*PCS_PER_CH-1:0]        hbm_rready,

  output logic [2:0]   led
);
  // ------------------------------------------------------------ resets
  logic rst_main, rst_core_i, rst_hbm;
  f2_rst_sync u_rm (.clk(clk_main), .arst_n(rst_main_n), .rst(rst_main));
  f2_rst_sync u_rc (.clk(clk_core), .arst_n(rst_main_n), .rst(rst_core_i));
  f2_rst_sync u_rh (.clk(clk_hbm),  .arst_n(rst_main_n), .rst(rst_hbm));

  // The board block waits for the memory: it stays in reset until HBM is initialized.
  logic hbm_ready_core;
  f2_bit_sync u_hr (.clk(clk_core), .d(hbm_ready), .q(hbm_ready_core));
  logic rst_core;
  always_ff @(posedge clk_core) rst_core <= rst_core_i || !hbm_ready_core;

  // ------------------------------------------------------------ OCL and register block
  logic [11:0] b_awaddr, b_araddr;
  logic [31:0] b_wdata, b_rdata;
  logic [3:0]  b_wstrb;
  logic [1:0]  b_bresp, b_rresp;
  logic        b_awvalid, b_awready, b_wvalid, b_wready, b_bvalid, b_bready;
  logic        b_arvalid, b_arready, b_rvalid, b_rready;
  logic        core_err_c, core_err_m, pcis_err_m;

  f2_ocl #(.PCS_PER_CH(PCS_PER_CH), .PC_AW(PC_AW), .STRIPE(STRIPE), .CORE_KHZ(CORE_KHZ),
           .BUILD_ID(BUILD_ID)) u_ocl (
    .main_clk(clk_main), .main_rst(rst_main), .core_clk(clk_core), .core_rst(rst_core_i),
    .ocl_awaddr, .ocl_awvalid, .ocl_awready, .ocl_wdata, .ocl_wstrb, .ocl_wvalid, .ocl_wready,
    .ocl_bresp, .ocl_bvalid, .ocl_bready, .ocl_araddr, .ocl_arvalid, .ocl_arready,
    .ocl_rdata, .ocl_rresp, .ocl_rvalid, .ocl_rready,
    .b_awaddr, .b_awvalid, .b_awready, .b_wdata, .b_wstrb, .b_wvalid, .b_wready,
    .b_bresp, .b_bvalid, .b_bready, .b_araddr, .b_arvalid, .b_arready,
    .b_rdata, .b_rresp, .b_rvalid, .b_rready,
    .hbm_ready, .core_err_pulse(core_err_m), .pcis_err_pulse(pcis_err_m));

  f2_pulse_sync u_ces (.sclk(clk_core), .srst(rst_core_i), .pulse(core_err_c),
                       .dclk(clk_main), .dpulse(core_err_m));

  // ------------------------------------------------------------ the frozen board block
  logic [0:0]   m0_awid, m1_awid, m0_arid, m1_arid;
  logic [31:0]  m0_awaddr, m1_awaddr, m0_araddr, m1_araddr;
  logic [7:0]   m0_awlen_u, m1_awlen_u, m0_arlen, m1_arlen;
  logic         m0_awvalid, m1_awvalid, m0_awready, m1_awready;
  logic [511:0] m0_wdata, m1_wdata, m0_rdata, m1_rdata;
  logic [63:0]  m0_wstrb, m1_wstrb;
  logic         m0_wlast, m1_wlast, m0_wvalid, m1_wvalid, m0_wready, m1_wready;
  logic [0:0]   m0_bid, m1_bid, m0_rid, m1_rid;
  logic [1:0]   m0_bresp, m1_bresp, m0_rresp, m1_rresp;
  logic         m0_bvalid, m1_bvalid, m0_bready, m1_bready;
  logic         m0_rlast, m1_rlast, m0_rvalid, m1_rvalid, m0_rready, m1_rready;
  logic         m0_arvalid, m1_arvalid, m0_arready, m1_arready;
  // unused AXI attribute outputs of the board block (constant by construction)
  logic [2:0]   u_awsize [2], u_arsize [2], u_awprot [2], u_arprot [2];
  logic [1:0]   u_awburst [2], u_arburst [2];
  logic [3:0]   u_awcache [2], u_arcache [2], u_awqos [2], u_arqos [2];
  logic         u_awlock [2], u_arlock [2];
  logic [3:0]   i2c_lo;

  otpu_board #(.D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
               .IMEM_WORDS(IMEM_WORDS), .LANES(LANES), .VPU_CL(VPU_CL), .ULANES(ULANES),
               .WIN(WIN), .FIFO_DEPTH(FIFO_DEPTH), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(0),
               .TRACE_DEPTH(TRACE_DEPTH), .TRACE_QD(TRACE_QD), .PQ_WIN(PQ_WIN),
               .AXI_BL(AXI_BL), .HAS_I2C(1'b0)) u_board (
    .clk(clk_core), .rst(rst_core), .calib({2{hbm_ready_core}}),
    .temp(12'h000),                    // no die-temperature sensor is wired: TEMP never becomes valid
    .led,
    .i2c_lo, .i2c_pin(5'b11111),
    .s_ctl_awaddr(b_awaddr), .s_ctl_awvalid(b_awvalid), .s_ctl_awready(b_awready),
    .s_ctl_wdata(b_wdata), .s_ctl_wstrb(b_wstrb), .s_ctl_wvalid(b_wvalid),
    .s_ctl_wready(b_wready), .s_ctl_bresp(b_bresp), .s_ctl_bvalid(b_bvalid),
    .s_ctl_bready(b_bready), .s_ctl_araddr(b_araddr), .s_ctl_arvalid(b_arvalid),
    .s_ctl_arready(b_arready), .s_ctl_rdata(b_rdata), .s_ctl_rresp(b_rresp),
    .s_ctl_rvalid(b_rvalid), .s_ctl_rready(b_rready),
    .m0_axi_awid(m0_awid), .m0_axi_awaddr(m0_awaddr), .m0_axi_awlen(m0_awlen_u),
    .m0_axi_awsize(u_awsize[0]), .m0_axi_awburst(u_awburst[0]), .m0_axi_awlock(u_awlock[0]),
    .m0_axi_awcache(u_awcache[0]), .m0_axi_awprot(u_awprot[0]), .m0_axi_awqos(u_awqos[0]),
    .m0_axi_awvalid(m0_awvalid), .m0_axi_awready(m0_awready), .m0_axi_wdata(m0_wdata),
    .m0_axi_wstrb(m0_wstrb), .m0_axi_wlast(m0_wlast), .m0_axi_wvalid(m0_wvalid),
    .m0_axi_wready(m0_wready), .m0_axi_bid(m0_bid), .m0_axi_bresp(m0_bresp),
    .m0_axi_bvalid(m0_bvalid), .m0_axi_bready(m0_bready), .m0_axi_arid(m0_arid),
    .m0_axi_araddr(m0_araddr), .m0_axi_arlen(m0_arlen), .m0_axi_arsize(u_arsize[0]),
    .m0_axi_arburst(u_arburst[0]), .m0_axi_arlock(u_arlock[0]), .m0_axi_arcache(u_arcache[0]),
    .m0_axi_arprot(u_arprot[0]), .m0_axi_arqos(u_arqos[0]), .m0_axi_arvalid(m0_arvalid),
    .m0_axi_arready(m0_arready), .m0_axi_rid(m0_rid), .m0_axi_rdata(m0_rdata),
    .m0_axi_rresp(m0_rresp), .m0_axi_rlast(m0_rlast), .m0_axi_rvalid(m0_rvalid),
    .m0_axi_rready(m0_rready),
    .m1_axi_awid(m1_awid), .m1_axi_awaddr(m1_awaddr), .m1_axi_awlen(m1_awlen_u),
    .m1_axi_awsize(u_awsize[1]), .m1_axi_awburst(u_awburst[1]), .m1_axi_awlock(u_awlock[1]),
    .m1_axi_awcache(u_awcache[1]), .m1_axi_awprot(u_awprot[1]), .m1_axi_awqos(u_awqos[1]),
    .m1_axi_awvalid(m1_awvalid), .m1_axi_awready(m1_awready), .m1_axi_wdata(m1_wdata),
    .m1_axi_wstrb(m1_wstrb), .m1_axi_wlast(m1_wlast), .m1_axi_wvalid(m1_wvalid),
    .m1_axi_wready(m1_wready), .m1_axi_bid(m1_bid), .m1_axi_bresp(m1_bresp),
    .m1_axi_bvalid(m1_bvalid), .m1_axi_bready(m1_bready), .m1_axi_arid(m1_arid),
    .m1_axi_araddr(m1_araddr), .m1_axi_arlen(m1_arlen), .m1_axi_arsize(u_arsize[1]),
    .m1_axi_arburst(u_arburst[1]), .m1_axi_arlock(u_arlock[1]), .m1_axi_arcache(u_arcache[1]),
    .m1_axi_arprot(u_arprot[1]), .m1_axi_arqos(u_arqos[1]), .m1_axi_arvalid(m1_arvalid),
    .m1_axi_arready(m1_arready), .m1_axi_rid(m1_rid), .m1_axi_rdata(m1_rdata),
    .m1_axi_rresp(m1_rresp), .m1_axi_rlast(m1_rlast), .m1_axi_rvalid(m1_rvalid),
    .m1_axi_rready(m1_rready));

  // ------------------------------------------------------------ memory adapter
  f2_hbm_adapter #(.PCS_PER_CH(PCS_PER_CH), .PC_AW(PC_AW), .STRIPE(STRIPE)) u_mem (
    .core_clk(clk_core), .core_rst(rst_core), .main_clk(clk_main), .main_rst(rst_main),
    .hbm_clk(clk_hbm), .hbm_rst(rst_hbm),
    .c_awid({m1_awid, m0_awid}), .c_awaddr({m1_awaddr, m0_awaddr}),
    .c_awvalid({m1_awvalid, m0_awvalid}), .c_awready({m1_awready, m0_awready}),
    .c_wdata({m1_wdata, m0_wdata}), .c_wstrb({m1_wstrb, m0_wstrb}),
    .c_wlast({m1_wlast, m0_wlast}), .c_wvalid({m1_wvalid, m0_wvalid}),
    .c_wready({m1_wready, m0_wready}), .c_bid({m1_bid, m0_bid}),
    .c_bresp({m1_bresp, m0_bresp}), .c_bvalid({m1_bvalid, m0_bvalid}),
    .c_bready({m1_bready, m0_bready}), .c_arid({m1_arid, m0_arid}),
    .c_araddr({m1_araddr, m0_araddr}), .c_arlen({m1_arlen, m0_arlen}),
    .c_arvalid({m1_arvalid, m0_arvalid}), .c_arready({m1_arready, m0_arready}),
    .c_rid({m1_rid, m0_rid}), .c_rdata({m1_rdata, m0_rdata}), .c_rresp({m1_rresp, m0_rresp}),
    .c_rlast({m1_rlast, m0_rlast}), .c_rvalid({m1_rvalid, m0_rvalid}),
    .c_rready({m1_rready, m0_rready}),
    .p_awid(pcis_awid), .p_awaddr(pcis_awaddr), .p_awlen(pcis_awlen),
    .p_awvalid(pcis_awvalid), .p_awready(pcis_awready), .p_wdata(pcis_wdata),
    .p_wstrb(pcis_wstrb), .p_wlast(pcis_wlast), .p_wvalid(pcis_wvalid), .p_wready(pcis_wready),
    .p_bid(pcis_bid), .p_bresp(pcis_bresp), .p_bvalid(pcis_bvalid), .p_bready(pcis_bready),
    .p_arid(pcis_arid), .p_araddr(pcis_araddr), .p_arlen(pcis_arlen),
    .p_arvalid(pcis_arvalid), .p_arready(pcis_arready), .p_rid(pcis_rid),
    .p_rdata(pcis_rdata), .p_rresp(pcis_rresp), .p_rlast(pcis_rlast),
    .p_rvalid(pcis_rvalid), .p_rready(pcis_rready),
    .hbm_awaddr, .hbm_awlen, .hbm_awvalid, .hbm_awready, .hbm_wdata, .hbm_wstrb, .hbm_wlast,
    .hbm_wvalid, .hbm_wready, .hbm_bresp, .hbm_bvalid, .hbm_bready, .hbm_araddr, .hbm_arlen,
    .hbm_arvalid, .hbm_arready, .hbm_rdata, .hbm_rresp, .hbm_rlast, .hbm_rvalid, .hbm_rready,
    .core_err_pulse(core_err_c), .pcis_err_pulse(pcis_err_m));
endmodule
