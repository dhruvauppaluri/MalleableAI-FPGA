// AWS F2 HDK top level for the OpenTPU custom logic (Small Shell).
//
// UNTESTED: never elaborated in Vivado, never built, never run on a card. It is written
// against the interface facts in the AWS HDK (release 2.3.4) and the port names of
// `cl_ports.vh`, and is expected to need edits at its first elaboration. It contains no AWS
// source: the HDK's `cl_mem_hbm_wrapper` (HBM IP, monitor APB, reset handling), `aws_clk_gen`
// and `sh_ddr` are referenced, not copied; `f2/vivado/setup_cl.sh` copies the wrapper from
// the developer's local HDK clone at build time.
//
// Structure:
//   SDA (MgmtPF BAR4) -> aws_clk_gen (clock recipe A1: clk_extra_a1 = 125 MHz core clock;
//                        HBM recipe H2: clk_hbm_axi = 450 MHz)
//   OCL, PCIS         -> cl_otpu_core (f2/rtl; the design verified in simulation)
//   cl_otpu_core HBM ports -> cl_mem_hbm_wrapper (AXI3, 256-bit) -> HBM IP
//
// After AFI load the host must release the clock generator's resets and wait for the MMCMs
// to lock (SDK `aws_clkgen_deassert_resets`; AWS_CLK_GEN spec, "Boot and Reset Sequencing"):
// the HBM AXI reset stays asserted until then and the core reports HBM not ready.
module cl_otpu
  #(
    parameter int PCS_PER_CH = 2                // PCs per core channel (docs/adr/0008)
    )
   (
    `include "cl_ports.vh"
    );

`include "cl_id_defines.vh"

  localparam int NPCT  = 2 * PCS_PER_CH;
  localparam int PC_AW = 29;                    // 512 MiB per HBM pseudo-channel

  // ---------------------------------------------------------------- globals
  always_comb begin
    cl_sh_flr_done    = 1'b1;
    cl_sh_status0     = 32'd0;
    cl_sh_status1     = 32'd0;
    cl_sh_status2     = 32'd0;
    cl_sh_id0         = `CL_SH_ID0;
    cl_sh_id1         = `CL_SH_ID1;
    cl_sh_status_vled = 16'd0;
    cl_sh_dma_wr_full = 1'b0;
    cl_sh_dma_rd_full = 1'b0;
  end

  // ---------------------------------------------------------------- PCIM: not used (idle master)
  always_comb begin
    cl_sh_pcim_awid = '0;    cl_sh_pcim_awaddr = '0;  cl_sh_pcim_awlen = '0;   cl_sh_pcim_awsize = '0;
    cl_sh_pcim_awburst = '0; cl_sh_pcim_awcache = '0; cl_sh_pcim_awlock = '0;  cl_sh_pcim_awprot = '0;
    cl_sh_pcim_awqos = '0;   cl_sh_pcim_awuser = '0;  cl_sh_pcim_awvalid = 1'b0;
    cl_sh_pcim_wid = '0;     cl_sh_pcim_wdata = '0;   cl_sh_pcim_wstrb = '0;   cl_sh_pcim_wlast = 1'b0;
    cl_sh_pcim_wuser = '0;   cl_sh_pcim_wvalid = 1'b0;
    cl_sh_pcim_bready = 1'b1;
    cl_sh_pcim_arid = '0;    cl_sh_pcim_araddr = '0;  cl_sh_pcim_arlen = '0;   cl_sh_pcim_arsize = '0;
    cl_sh_pcim_arburst = '0; cl_sh_pcim_arcache = '0; cl_sh_pcim_arlock = '0;  cl_sh_pcim_arprot = '0;
    cl_sh_pcim_arqos = '0;   cl_sh_pcim_aruser = '0;  cl_sh_pcim_arvalid = 1'b0;
    cl_sh_pcim_rready = 1'b1;
  end

  // ---------------------------------------------------------------- DDR: instantiated but not present
  sh_ddr #(.DDR_PRESENT(0)) SH_DDR (
    .clk (clk_main_a0), .rst_n (), .stat_clk (clk_main_a0), .stat_rst_n (),
    .CLK_DIMM_DP (CLK_DIMM_DP), .CLK_DIMM_DN (CLK_DIMM_DN), .M_ACT_N (M_ACT_N), .M_MA (M_MA),
    .M_BA (M_BA), .M_BG (M_BG), .M_CKE (M_CKE), .M_ODT (M_ODT), .M_CS_N (M_CS_N),
    .M_CLK_DN (M_CLK_DN), .M_CLK_DP (M_CLK_DP), .M_PAR (M_PAR), .M_DQ (M_DQ), .M_ECC (M_ECC),
    .M_DQS_DP (M_DQS_DP), .M_DQS_DN (M_DQS_DN), .cl_RST_DIMM_N (RST_DIMM_N),
    .cl_sh_ddr_axi_awid (), .cl_sh_ddr_axi_awaddr (), .cl_sh_ddr_axi_awlen (), .cl_sh_ddr_axi_awsize (),
    .cl_sh_ddr_axi_awvalid (), .cl_sh_ddr_axi_awburst (), .cl_sh_ddr_axi_awuser (), .cl_sh_ddr_axi_awready (),
    .cl_sh_ddr_axi_wdata (), .cl_sh_ddr_axi_wstrb (), .cl_sh_ddr_axi_wlast (), .cl_sh_ddr_axi_wvalid (),
    .cl_sh_ddr_axi_wready (), .cl_sh_ddr_axi_bid (), .cl_sh_ddr_axi_bresp (), .cl_sh_ddr_axi_bvalid (),
    .cl_sh_ddr_axi_bready (), .cl_sh_ddr_axi_arid (), .cl_sh_ddr_axi_araddr (), .cl_sh_ddr_axi_arlen (),
    .cl_sh_ddr_axi_arsize (), .cl_sh_ddr_axi_arvalid (), .cl_sh_ddr_axi_arburst (), .cl_sh_ddr_axi_aruser (),
    .cl_sh_ddr_axi_arready (), .cl_sh_ddr_axi_rid (), .cl_sh_ddr_axi_rdata (), .cl_sh_ddr_axi_rresp (),
    .cl_sh_ddr_axi_rlast (), .cl_sh_ddr_axi_rvalid (), .cl_sh_ddr_axi_rready (),
    .sh_ddr_stat_bus_addr (), .sh_ddr_stat_bus_wdata (), .sh_ddr_stat_bus_wr (), .sh_ddr_stat_bus_rd (),
    .sh_ddr_stat_bus_ack (), .sh_ddr_stat_bus_rdata (), .ddr_sh_stat_int (), .sh_cl_ddr_is_ready ());
  always_comb begin
    cl_sh_ddr_stat_ack   = 1'b0;
    cl_sh_ddr_stat_rdata = 32'd0;
    cl_sh_ddr_stat_int   = 8'd0;
  end

  // ---------------------------------------------------------------- interrupts, JTAG, PCIe pins
  always_comb begin
    cl_sh_apppf_irq_req = 16'd0;
    tdo                 = 1'b0;
    PCIE_EP_TXP = '0;  PCIE_EP_TXN = '0;
    PCIE_RP_PERSTN = 1'b0;  PCIE_RP_TXP = '0;  PCIE_RP_TXN = '0;
  end

  // ---------------------------------------------------------------- clocks: AWS_CLK_GEN on SDA
  // (the instance must be named AWS_CLK_GEN and sit in the CL top module)
  logic gen_clk_main_a0, gen_clk_hbm_ref, gen_clk_extra_a1, gen_clk_hbm_axi;
  logic gen_clk_extra_a2, gen_clk_extra_a3, gen_clk_extra_b0, gen_clk_extra_b1, gen_clk_extra_c0, gen_clk_extra_c1;
  logic gen_rst_main_n, gen_rst_hbm_axi_n, gen_rst_hbm_ref_n, gen_rst_a1_n;
  logic gen_rst_a2_n, gen_rst_a3_n, gen_rst_b0_n, gen_rst_b1_n, gen_rst_c0_n, gen_rst_c1_n;

  aws_clk_gen #(.CLK_GRP_A_EN(1), .CLK_GRP_B_EN(0), .CLK_GRP_C_EN(0), .CLK_HBM_EN(1)) AWS_CLK_GEN (
    .i_clk_main_a0 (clk_main_a0), .i_rst_main_n (rst_main_n), .i_clk_hbm_ref (clk_hbm_ref),
    .s_axil_ctrl_awaddr (sda_cl_awaddr), .s_axil_ctrl_awvalid (sda_cl_awvalid), .s_axil_ctrl_awready (cl_sda_awready),
    .s_axil_ctrl_wdata (sda_cl_wdata), .s_axil_ctrl_wstrb (sda_cl_wstrb), .s_axil_ctrl_wvalid (sda_cl_wvalid),
    .s_axil_ctrl_wready (cl_sda_wready), .s_axil_ctrl_bresp (cl_sda_bresp), .s_axil_ctrl_bvalid (cl_sda_bvalid),
    .s_axil_ctrl_bready (sda_cl_bready), .s_axil_ctrl_araddr (sda_cl_araddr), .s_axil_ctrl_arvalid (sda_cl_arvalid),
    .s_axil_ctrl_arready (cl_sda_arready), .s_axil_ctrl_rdata (cl_sda_rdata), .s_axil_ctrl_rresp (cl_sda_rresp),
    .s_axil_ctrl_rvalid (cl_sda_rvalid), .s_axil_ctrl_rready (sda_cl_rready),
    .o_clk_hbm_ref (gen_clk_hbm_ref), .o_clk_main_a0 (gen_clk_main_a0), .o_clk_extra_a1 (gen_clk_extra_a1),
    .o_clk_extra_a2 (gen_clk_extra_a2), .o_clk_extra_a3 (gen_clk_extra_a3), .o_clk_extra_b0 (gen_clk_extra_b0),
    .o_clk_extra_b1 (gen_clk_extra_b1), .o_clk_extra_c0 (gen_clk_extra_c0), .o_clk_extra_c1 (gen_clk_extra_c1),
    .o_clk_hbm_axi (gen_clk_hbm_axi),
    .o_cl_rst_hbm_axi_n (gen_rst_hbm_axi_n), .o_cl_rst_hbm_ref_n (gen_rst_hbm_ref_n), .o_cl_rst_c1_n (gen_rst_c1_n),
    .o_cl_rst_c0_n (gen_rst_c0_n), .o_cl_rst_b1_n (gen_rst_b1_n), .o_cl_rst_b0_n (gen_rst_b0_n),
    .o_cl_rst_a3_n (gen_rst_a3_n), .o_cl_rst_a2_n (gen_rst_a2_n), .o_cl_rst_a1_n (gen_rst_a1_n),
    .o_cl_rst_main_n (gen_rst_main_n));

  // ---------------------------------------------------------------- HBM (HDK wrapper, AXI3, one port per PC)
  logic [33:0] hbm_araddr [0:NPCT-1], hbm_awaddr [0:NPCT-1];
  logic [1:0]  hbm_arburst [0:NPCT-1], hbm_awburst [0:NPCT-1];
  logic [5:0]  hbm_arid [0:NPCT-1], hbm_awid [0:NPCT-1], hbm_rid [0:NPCT-1], hbm_bid [0:NPCT-1];
  logic [3:0]  hbm_arlen [0:NPCT-1], hbm_awlen [0:NPCT-1];
  logic [2:0]  hbm_arsize [0:NPCT-1], hbm_awsize [0:NPCT-1];
  logic        hbm_arvalid [0:NPCT-1], hbm_awvalid [0:NPCT-1], hbm_rready [0:NPCT-1], hbm_bready [0:NPCT-1];
  logic [255:0] hbm_wdata [0:NPCT-1], hbm_rdata [0:NPCT-1];
  logic        hbm_wlast [0:NPCT-1], hbm_wvalid [0:NPCT-1];
  logic [31:0] hbm_wstrb [0:NPCT-1];
  logic        hbm_arready [0:NPCT-1], hbm_awready [0:NPCT-1], hbm_rlast [0:NPCT-1], hbm_rvalid [0:NPCT-1];
  logic        hbm_wready [0:NPCT-1], hbm_bvalid [0:NPCT-1];
  logic [1:0]  hbm_rresp [0:NPCT-1], hbm_bresp [0:NPCT-1];
  logic        hbm_ready;

  // core-side (flat, PC-local addresses)
  logic [NPCT*PC_AW-1:0] c_awaddr, c_araddr;
  logic [NPCT*4-1:0]     c_awlen, c_arlen;
  logic [NPCT-1:0]       c_awvalid, c_awready, c_wlast, c_wvalid, c_wready, c_bvalid, c_bready;
  logic [NPCT-1:0]       c_arvalid, c_arready, c_rlast, c_rvalid, c_rready;
  logic [NPCT*256-1:0]   c_wdata, c_rdata;
  logic [NPCT*32-1:0]    c_wstrb;
  logic [NPCT*2-1:0]     c_bresp, c_rresp;

  for (genvar p = 0; p < NPCT; p++) begin : g_hbm_port
    // ASSUMPTION A2 (docs/adr/0008): a port addresses its own pseudo-channel: bits [33:29] = PC index
    assign hbm_araddr[p]  = {5'(p), c_araddr[p*PC_AW +: PC_AW]};
    assign hbm_awaddr[p]  = {5'(p), c_awaddr[p*PC_AW +: PC_AW]};
    assign hbm_arburst[p] = 2'b01;   assign hbm_awburst[p] = 2'b01;
    assign hbm_arid[p]    = 6'd0;    assign hbm_awid[p]    = 6'd0;
    assign hbm_arlen[p]   = c_arlen[p*4 +: 4];   assign hbm_awlen[p] = c_awlen[p*4 +: 4];
    assign hbm_arsize[p]  = 3'd5;    assign hbm_awsize[p]  = 3'd5;        // 32 bytes per beat
    assign hbm_arvalid[p] = c_arvalid[p];  assign hbm_awvalid[p] = c_awvalid[p];
    assign c_arready[p]   = hbm_arready[p]; assign c_awready[p]  = hbm_awready[p];
    assign hbm_wdata[p]   = c_wdata[p*256 +: 256];  assign hbm_wstrb[p] = c_wstrb[p*32 +: 32];
    assign hbm_wlast[p]   = c_wlast[p];    assign hbm_wvalid[p]  = c_wvalid[p];
    assign c_wready[p]    = hbm_wready[p];
    assign c_rdata[p*256 +: 256] = hbm_rdata[p];  assign c_rresp[p*2 +: 2] = hbm_rresp[p];
    assign c_rlast[p]     = hbm_rlast[p];  assign c_rvalid[p]    = hbm_rvalid[p];
    assign hbm_rready[p]  = c_rready[p];
    assign c_bresp[p*2 +: 2] = hbm_bresp[p];  assign c_bvalid[p] = hbm_bvalid[p];
    assign hbm_bready[p]  = c_bready[p];
  end

  cfg_bus_t hbm_stat_bus ();
  assign hbm_stat_bus.addr  = '0;
  assign hbm_stat_bus.wdata = '0;
  assign hbm_stat_bus.wr    = 1'b0;
  assign hbm_stat_bus.rd    = 1'b0;
  assign hbm_stat_bus.user  = '0;

  cl_mem_hbm_wrapper #(.NUM_OF_AXI_PORTS(NPCT), .AXI4_INTERFACE(0)) HBM_WRAPPER (
    .apb_clk (gen_clk_hbm_ref),
    .i_clk_250m (gen_clk_main_a0), .i_rst_250m_n (gen_rst_main_n),
    .i_clk_450m (gen_clk_hbm_axi), .i_rst_450m_n (gen_rst_hbm_axi_n),
    .i_axi_araddr (hbm_araddr), .i_axi_arburst (hbm_arburst), .i_axi_arid (hbm_arid), .i_axi_arlen (hbm_arlen),
    .i_axi_arsize (hbm_arsize), .i_axi_arvalid (hbm_arvalid), .i_axi_awaddr (hbm_awaddr),
    .i_axi_awburst (hbm_awburst), .i_axi_awid (hbm_awid), .i_axi_awlen (hbm_awlen), .i_axi_awsize (hbm_awsize),
    .i_axi_awvalid (hbm_awvalid), .i_axi_rready (hbm_rready), .i_axi_bready (hbm_bready),
    .i_axi_wdata (hbm_wdata), .i_axi_wlast (hbm_wlast), .i_axi_wstrb (hbm_wstrb), .i_axi_wvalid (hbm_wvalid),
    .o_axi_arready (hbm_arready), .o_axi_awready (hbm_awready), .o_axi_rdata (hbm_rdata), .o_axi_rid (hbm_rid),
    .o_axi_rlast (hbm_rlast), .o_axi_rresp (hbm_rresp), .o_axi_rvalid (hbm_rvalid), .o_axi_wready (hbm_wready),
    .o_axi_bid (hbm_bid), .o_axi_bresp (hbm_bresp), .o_axi_bvalid (hbm_bvalid),
    .i_hbm_apb_preset_n_1 (hbm_apb_preset_n_1), .o_hbm_apb_paddr_1 (hbm_apb_paddr_1),
    .o_hbm_apb_pprot_1 (hbm_apb_pprot_1), .o_hbm_apb_psel_1 (hbm_apb_psel_1),
    .o_hbm_apb_penable_1 (hbm_apb_penable_1), .o_hbm_apb_pwrite_1 (hbm_apb_pwrite_1),
    .o_hbm_apb_pwdata_1 (hbm_apb_pwdata_1), .o_hbm_apb_pstrb_1 (hbm_apb_pstrb_1),
    .o_hbm_apb_pready_1 (hbm_apb_pready_1), .o_hbm_apb_prdata_1 (hbm_apb_prdata_1),
    .o_hbm_apb_pslverr_1 (hbm_apb_pslverr_1),
    .i_hbm_apb_preset_n_0 (hbm_apb_preset_n_0), .o_hbm_apb_paddr_0 (hbm_apb_paddr_0),
    .o_hbm_apb_pprot_0 (hbm_apb_pprot_0), .o_hbm_apb_psel_0 (hbm_apb_psel_0),
    .o_hbm_apb_penable_0 (hbm_apb_penable_0), .o_hbm_apb_pwrite_0 (hbm_apb_pwrite_0),
    .o_hbm_apb_pwdata_0 (hbm_apb_pwdata_0), .o_hbm_apb_pstrb_0 (hbm_apb_pstrb_0),
    .o_hbm_apb_pready_0 (hbm_apb_pready_0), .o_hbm_apb_prdata_0 (hbm_apb_prdata_0),
    .o_hbm_apb_pslverr_0 (hbm_apb_pslverr_0),
    .hbm_stat_bus (hbm_stat_bus.slave), .o_cl_sh_hbm_stat_int (), .o_hbm_ready (hbm_ready));

  // ---------------------------------------------------------------- the design under test in simulation
  cl_otpu_core #(.PCS_PER_CH(PCS_PER_CH), .PC_AW(PC_AW), .MCOLS(2), .LANES(8), .CORE_KHZ(125000)) CORE (
    .clk_main (gen_clk_main_a0), .rst_main_n (gen_rst_main_n), .clk_core (gen_clk_extra_a1),
    .clk_hbm (gen_clk_hbm_axi), .hbm_ready (hbm_ready),
    .ocl_awaddr (ocl_cl_awaddr), .ocl_awvalid (ocl_cl_awvalid), .ocl_awready (cl_ocl_awready),
    .ocl_wdata (ocl_cl_wdata), .ocl_wstrb (ocl_cl_wstrb), .ocl_wvalid (ocl_cl_wvalid),
    .ocl_wready (cl_ocl_wready), .ocl_bresp (cl_ocl_bresp), .ocl_bvalid (cl_ocl_bvalid),
    .ocl_bready (ocl_cl_bready), .ocl_araddr (ocl_cl_araddr), .ocl_arvalid (ocl_cl_arvalid),
    .ocl_arready (cl_ocl_arready), .ocl_rdata (cl_ocl_rdata), .ocl_rresp (cl_ocl_rresp),
    .ocl_rvalid (cl_ocl_rvalid), .ocl_rready (ocl_cl_rready),
    .pcis_awid (sh_cl_dma_pcis_awid), .pcis_awaddr (sh_cl_dma_pcis_awaddr), .pcis_awlen (sh_cl_dma_pcis_awlen),
    .pcis_awvalid (sh_cl_dma_pcis_awvalid), .pcis_awready (cl_sh_dma_pcis_awready),
    .pcis_wdata (sh_cl_dma_pcis_wdata), .pcis_wstrb (sh_cl_dma_pcis_wstrb), .pcis_wlast (sh_cl_dma_pcis_wlast),
    .pcis_wvalid (sh_cl_dma_pcis_wvalid), .pcis_wready (cl_sh_dma_pcis_wready),
    .pcis_bid (cl_sh_dma_pcis_bid), .pcis_bresp (cl_sh_dma_pcis_bresp), .pcis_bvalid (cl_sh_dma_pcis_bvalid),
    .pcis_bready (sh_cl_dma_pcis_bready), .pcis_arid (sh_cl_dma_pcis_arid), .pcis_araddr (sh_cl_dma_pcis_araddr),
    .pcis_arlen (sh_cl_dma_pcis_arlen), .pcis_arvalid (sh_cl_dma_pcis_arvalid),
    .pcis_arready (cl_sh_dma_pcis_arready), .pcis_rid (cl_sh_dma_pcis_rid), .pcis_rdata (cl_sh_dma_pcis_rdata),
    .pcis_rresp (cl_sh_dma_pcis_rresp), .pcis_rlast (cl_sh_dma_pcis_rlast), .pcis_rvalid (cl_sh_dma_pcis_rvalid),
    .pcis_rready (sh_cl_dma_pcis_rready),
    .hbm_awaddr (c_awaddr), .hbm_awlen (c_awlen), .hbm_awvalid (c_awvalid), .hbm_awready (c_awready),
    .hbm_wdata (c_wdata), .hbm_wstrb (c_wstrb), .hbm_wlast (c_wlast), .hbm_wvalid (c_wvalid),
    .hbm_wready (c_wready), .hbm_bresp (c_bresp), .hbm_bvalid (c_bvalid), .hbm_bready (c_bready),
    .hbm_araddr (c_araddr), .hbm_arlen (c_arlen), .hbm_arvalid (c_arvalid), .hbm_arready (c_arready),
    .hbm_rdata (c_rdata), .hbm_rresp (c_rresp), .hbm_rlast (c_rlast), .hbm_rvalid (c_rvalid),
    .hbm_rready (c_rready), .led ());

  // PCIS user fields and unused inputs are ignored on purpose.
endmodule
