// F2 platform: memory adapter between the frozen accelerator board block
// (otpu_board: two 512-bit AXI4 channels in the core clock domain, 64-byte beats,
// 2 GiB address space per channel at BASE0 / BASE1), the shell's PCIS port (512-bit
// AXI4, main clock) and PCS_PER_CH * 2 HBM pseudo-channel (PC) ports (256-bit, HBM
// AXI clock).
//
// Placement (must match malleable/f2/placement.py):
//   channel c of the core owns PCs [c*PCS .. c*PCS+PCS-1]. Inside a channel,
//   offset o (31 bits) goes to PC c*PCS + o[STRIPE +: log2 PCS] at local address
//   {o[30 : STRIPE+log2 PCS], o[STRIPE-1:0]}.
//   PCIS window (host view): HBM_BASE = 1 << 36 (as in the AWS HDK examples), size
//   4 GiB: address bit 31 = core channel, bits [30:0] = channel offset. A host
//   write of channel offset o therefore lands exactly where the core reads it.
// The core sees the same flat, 2-channel-interleaved image as on the original
// board. Only this module knows about PCs.
//
// UNTESTED on hardware; not run through Vivado.
module f2_hbm_adapter #(
  parameter int PCS_PER_CH = 2,     // PCs per core channel (power of two >= 2)
  parameter int PC_AW      = 29,    // PC-local address bits (29 = 512 MiB in the real HBM)
  parameter int STRIPE     = 9,
  parameter int FAW        = 4
) (
  input  logic                    core_clk,
  input  logic                    core_rst,
  input  logic                    main_clk,
  input  logic                    main_rst,
  input  logic                    hbm_clk,
  input  logic                    hbm_rst,

  // ---- core channels 0 and 1 (from otpu_board m0_* / m1_*; the unused AXI fields are dropped)
  input  logic [1:0]              c_awid,
  input  logic [2*32-1:0]         c_awaddr,
  input  logic [1:0]              c_awvalid,
  output logic [1:0]              c_awready,
  input  logic [2*512-1:0]        c_wdata,
  input  logic [2*64-1:0]         c_wstrb,
  input  logic [1:0]              c_wlast,
  input  logic [1:0]              c_wvalid,
  output logic [1:0]              c_wready,
  output logic [1:0]              c_bid,
  output logic [2*2-1:0]          c_bresp,
  output logic [1:0]              c_bvalid,
  input  logic [1:0]              c_bready,
  input  logic [1:0]              c_arid,
  input  logic [2*32-1:0]         c_araddr,
  input  logic [2*8-1:0]          c_arlen,
  input  logic [1:0]              c_arvalid,
  output logic [1:0]              c_arready,
  output logic [1:0]              c_rid,
  output logic [2*512-1:0]        c_rdata,
  output logic [2*2-1:0]          c_rresp,
  output logic [1:0]              c_rlast,
  output logic [1:0]              c_rvalid,
  input  logic [1:0]              c_rready,

  // ---- PCIS (AXI4, 512-bit, 16-bit ID, 64-bit address)
  input  logic [15:0]             p_awid,
  input  logic [63:0]             p_awaddr,
  input  logic [7:0]              p_awlen,
  input  logic                    p_awvalid,
  output logic                    p_awready,
  input  logic [511:0]            p_wdata,
  input  logic [63:0]             p_wstrb,
  input  logic                    p_wlast,
  input  logic                    p_wvalid,
  output logic                    p_wready,
  output logic [15:0]             p_bid,
  output logic [1:0]              p_bresp,
  output logic                    p_bvalid,
  input  logic                    p_bready,
  input  logic [15:0]             p_arid,
  input  logic [63:0]             p_araddr,
  input  logic [7:0]              p_arlen,
  input  logic                    p_arvalid,
  output logic                    p_arready,
  output logic [15:0]             p_rid,
  output logic [511:0]            p_rdata,
  output logic [1:0]              p_rresp,
  output logic                    p_rlast,
  output logic                    p_rvalid,
  input  logic                    p_rready,

  // ---- HBM pseudo-channel ports (PC p in slice p; local addresses, HBM clock)
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

  output logic                    core_err_pulse,   // core clock: error piece/response on a core channel
  output logic                    pcis_err_pulse    // main clock: error piece/response on PCIS
);
  localparam int NPCT   = 2 * PCS_PER_CH;
  localparam int LOGPCH = $clog2(PCS_PER_CH);

  // ================================================================ core channel routers
  logic [1:0][PCS_PER_CH*PC_AW-1:0]   cr_awaddr, cr_araddr;
  logic [1:0][PCS_PER_CH*4-1:0]       cr_awlen, cr_arlen;
  logic [1:0][PCS_PER_CH-1:0]         cr_awvalid, cr_awready, cr_wlast, cr_wvalid, cr_wready;
  logic [1:0][PCS_PER_CH-1:0]         cr_bvalid, cr_bready, cr_arvalid, cr_arready;
  logic [1:0][PCS_PER_CH-1:0]         cr_rlast, cr_rvalid, cr_rready;
  logic [1:0][PCS_PER_CH*256-1:0]     cr_wdata, cr_rdata;
  logic [1:0][PCS_PER_CH*32-1:0]      cr_wstrb;
  logic [1:0][PCS_PER_CH*2-1:0]       cr_bresp, cr_rresp;
  logic [1:0]                         cr_err;

  for (genvar c = 0; c < 2; c++) begin : g_core
    f2_hbm_router #(.NPC(PCS_PER_CH), .AW(34), .PC_AW(PC_AW), .IDW(1), .STRIPE(STRIPE)) u_r (
      .clk(core_clk), .rst(core_rst),
      .s_awid(c_awid[c]), .s_awaddr({3'b000, c_awaddr[c*32 +: 31]}), .s_awgrp(1'b0), .s_awlen(8'd0),
      .s_awvalid(c_awvalid[c]), .s_awready(c_awready[c]),
      .s_wdata(c_wdata[c*512 +: 512]), .s_wstrb(c_wstrb[c*64 +: 64]), .s_wlast(c_wlast[c]),
      .s_wvalid(c_wvalid[c]), .s_wready(c_wready[c]),
      .s_bid(c_bid[c]), .s_bresp(c_bresp[c*2 +: 2]), .s_bvalid(c_bvalid[c]),
      .s_bready(c_bready[c]),
      .s_arid(c_arid[c]), .s_araddr({3'b000, c_araddr[c*32 +: 31]}), .s_argrp(1'b0),
      .s_arlen(c_arlen[c*8 +: 8]), .s_arvalid(c_arvalid[c]), .s_arready(c_arready[c]),
      .s_rid(c_rid[c]), .s_rdata(c_rdata[c*512 +: 512]), .s_rresp(c_rresp[c*2 +: 2]),
      .s_rlast(c_rlast[c]), .s_rvalid(c_rvalid[c]), .s_rready(c_rready[c]),
      .m_awaddr(cr_awaddr[c]), .m_awlen(cr_awlen[c]), .m_awvalid(cr_awvalid[c]),
      .m_awready(cr_awready[c]), .m_wdata(cr_wdata[c]), .m_wstrb(cr_wstrb[c]),
      .m_wlast(cr_wlast[c]), .m_wvalid(cr_wvalid[c]), .m_wready(cr_wready[c]),
      .m_bresp(cr_bresp[c]), .m_bvalid(cr_bvalid[c]), .m_bready(cr_bready[c]),
      .m_araddr(cr_araddr[c]), .m_arlen(cr_arlen[c]), .m_arvalid(cr_arvalid[c]),
      .m_arready(cr_arready[c]), .m_rdata(cr_rdata[c]), .m_rresp(cr_rresp[c]),
      .m_rlast(cr_rlast[c]), .m_rvalid(cr_rvalid[c]), .m_rready(cr_rready[c]),
      .err_pulse(cr_err[c]));
  end
  assign core_err_pulse = |cr_err;

  // ================================================================ PCIS router
  // window decode: HBM_BASE = 1 << 36; bit 31 = core channel; [30:0] = channel offset
  wire        p_aw_in = p_awaddr[63:37] == '0 && p_awaddr[36] && p_awaddr[35:32] == '0;
  wire        p_ar_in = p_araddr[63:37] == '0 && p_araddr[36] && p_araddr[35:32] == '0;
  // linear router address: channel offset in [30:0]; the two top bits are 11 outside the
  // window, which the router answers with DECERR. The core channel selects the PC group.
  function automatic logic [33:0] win_addr(input logic [63:0] a, input logic in_window);
    win_addr = {in_window ? 2'b00 : 2'b11, 1'b0, a[30:0]};
  endfunction

  logic [NPCT*PC_AW-1:0]  pr_awaddr, pr_araddr;
  logic [NPCT*4-1:0]      pr_awlen, pr_arlen;
  logic [NPCT-1:0]        pr_awvalid, pr_awready, pr_wlast, pr_wvalid, pr_wready;
  logic [NPCT-1:0]        pr_bvalid, pr_bready, pr_arvalid, pr_arready;
  logic [NPCT-1:0]        pr_rlast, pr_rvalid, pr_rready;
  logic [NPCT*256-1:0]    pr_wdata, pr_rdata;
  logic [NPCT*32-1:0]     pr_wstrb;
  logic [NPCT*2-1:0]      pr_bresp, pr_rresp;

  f2_hbm_router #(.NPC(NPCT), .GRPB(1), .AW(34), .PC_AW(PC_AW), .IDW(16), .STRIPE(STRIPE)) u_pcis (
    .clk(main_clk), .rst(main_rst),
    .s_awid(p_awid), .s_awaddr(win_addr(p_awaddr, p_aw_in)), .s_awgrp(p_awaddr[31]), .s_awlen(p_awlen),
    .s_awvalid(p_awvalid), .s_awready(p_awready),
    .s_wdata(p_wdata), .s_wstrb(p_wstrb), .s_wlast(p_wlast), .s_wvalid(p_wvalid),
    .s_wready(p_wready),
    .s_bid(p_bid), .s_bresp(p_bresp), .s_bvalid(p_bvalid), .s_bready(p_bready),
    .s_arid(p_arid), .s_araddr(win_addr(p_araddr, p_ar_in)), .s_argrp(p_araddr[31]), .s_arlen(p_arlen),
    .s_arvalid(p_arvalid), .s_arready(p_arready),
    .s_rid(p_rid), .s_rdata(p_rdata), .s_rresp(p_rresp), .s_rlast(p_rlast),
    .s_rvalid(p_rvalid), .s_rready(p_rready),
    .m_awaddr(pr_awaddr), .m_awlen(pr_awlen), .m_awvalid(pr_awvalid), .m_awready(pr_awready),
    .m_wdata(pr_wdata), .m_wstrb(pr_wstrb), .m_wlast(pr_wlast), .m_wvalid(pr_wvalid),
    .m_wready(pr_wready), .m_bresp(pr_bresp), .m_bvalid(pr_bvalid), .m_bready(pr_bready),
    .m_araddr(pr_araddr), .m_arlen(pr_arlen), .m_arvalid(pr_arvalid), .m_arready(pr_arready),
    .m_rdata(pr_rdata), .m_rresp(pr_rresp), .m_rlast(pr_rlast), .m_rvalid(pr_rvalid),
    .m_rready(pr_rready), .err_pulse(pcis_err_pulse));

  // ================================================================ one bridge per PC
  for (genvar p = 0; p < NPCT; p++) begin : g_pc
    localparam int CH  = p / PCS_PER_CH;
    localparam int SUB = p % PCS_PER_CH;
    f2_hbm_pc_bridge #(.PC_AW(PC_AW), .FAW(FAW)) u_b (
      .hbm_clk, .hbm_rst,
      .s_clk({main_clk, core_clk}), .s_rst({main_rst, core_rst}),
      .s_awaddr({pr_awaddr[p*PC_AW +: PC_AW], cr_awaddr[CH][SUB*PC_AW +: PC_AW]}),
      .s_awlen({pr_awlen[p*4 +: 4], cr_awlen[CH][SUB*4 +: 4]}),
      .s_awvalid({pr_awvalid[p], cr_awvalid[CH][SUB]}),
      .s_awready({pr_awready[p], cr_awready[CH][SUB]}),
      .s_wdata({pr_wdata[p*256 +: 256], cr_wdata[CH][SUB*256 +: 256]}),
      .s_wstrb({pr_wstrb[p*32 +: 32], cr_wstrb[CH][SUB*32 +: 32]}),
      .s_wlast({pr_wlast[p], cr_wlast[CH][SUB]}),
      .s_wvalid({pr_wvalid[p], cr_wvalid[CH][SUB]}),
      .s_wready({pr_wready[p], cr_wready[CH][SUB]}),
      .s_bresp({pr_bresp[p*2 +: 2], cr_bresp[CH][SUB*2 +: 2]}),
      .s_bvalid({pr_bvalid[p], cr_bvalid[CH][SUB]}),
      .s_bready({pr_bready[p], cr_bready[CH][SUB]}),
      .s_araddr({pr_araddr[p*PC_AW +: PC_AW], cr_araddr[CH][SUB*PC_AW +: PC_AW]}),
      .s_arlen({pr_arlen[p*4 +: 4], cr_arlen[CH][SUB*4 +: 4]}),
      .s_arvalid({pr_arvalid[p], cr_arvalid[CH][SUB]}),
      .s_arready({pr_arready[p], cr_arready[CH][SUB]}),
      .s_rdata({pr_rdata[p*256 +: 256], cr_rdata[CH][SUB*256 +: 256]}),
      .s_rresp({pr_rresp[p*2 +: 2], cr_rresp[CH][SUB*2 +: 2]}),
      .s_rlast({pr_rlast[p], cr_rlast[CH][SUB]}),
      .s_rvalid({pr_rvalid[p], cr_rvalid[CH][SUB]}),
      .s_rready({pr_rready[p], cr_rready[CH][SUB]}),
      .m_awaddr(hbm_awaddr[p*PC_AW +: PC_AW]), .m_awlen(hbm_awlen[p*4 +: 4]),
      .m_awvalid(hbm_awvalid[p]), .m_awready(hbm_awready[p]),
      .m_wdata(hbm_wdata[p*256 +: 256]), .m_wstrb(hbm_wstrb[p*32 +: 32]),
      .m_wlast(hbm_wlast[p]), .m_wvalid(hbm_wvalid[p]), .m_wready(hbm_wready[p]),
      .m_bresp(hbm_bresp[p*2 +: 2]), .m_bvalid(hbm_bvalid[p]), .m_bready(hbm_bready[p]),
      .m_araddr(hbm_araddr[p*PC_AW +: PC_AW]), .m_arlen(hbm_arlen[p*4 +: 4]),
      .m_arvalid(hbm_arvalid[p]), .m_arready(hbm_arready[p]),
      .m_rdata(hbm_rdata[p*256 +: 256]), .m_rresp(hbm_rresp[p*2 +: 2]),
      .m_rlast(hbm_rlast[p]), .m_rvalid(hbm_rvalid[p]), .m_rready(hbm_rready[p]));
  end
endmodule
