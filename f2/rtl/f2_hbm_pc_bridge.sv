// F2 platform: one HBM pseudo-channel shared by two request sources that live in
// different clock domains (source 0: the accelerator's channel router, source 1:
// the host PCIS router).
//
// Every request and response channel crosses to or from the HBM AXI clock through
// an asynchronous FIFO. In the HBM domain the sources are arbitrated per burst:
// round-robin on AR and on AW. A W stream follows the order in which AWs were
// granted; responses return to the source that issued the request, in order
// (the PC answers in order because every request uses a single ID). A host
// access therefore waits at most a few bursts behind the core, never for a whole
// core program, which keeps PCIS inside the shell's 8 us timeout.
//
// UNTESTED on hardware; not run through Vivado.
module f2_hbm_pc_bridge #(
  parameter int PC_AW = 29,
  parameter int FAW   = 4        // async FIFO depth 2**FAW entries per channel
) (
  input  logic                   hbm_clk,
  input  logic                   hbm_rst,

  // ---- sources (index 0 and 1), each in its own clock domain
  input  logic [1:0]             s_clk,
  input  logic [1:0]             s_rst,
  input  logic [2*PC_AW-1:0]     s_awaddr,
  input  logic [2*4-1:0]         s_awlen,
  input  logic [1:0]             s_awvalid,
  output logic [1:0]             s_awready,
  input  logic [2*256-1:0]       s_wdata,
  input  logic [2*32-1:0]        s_wstrb,
  input  logic [1:0]             s_wlast,
  input  logic [1:0]             s_wvalid,
  output logic [1:0]             s_wready,
  output logic [2*2-1:0]         s_bresp,
  output logic [1:0]             s_bvalid,
  input  logic [1:0]             s_bready,
  input  logic [2*PC_AW-1:0]     s_araddr,
  input  logic [2*4-1:0]         s_arlen,
  input  logic [1:0]             s_arvalid,
  output logic [1:0]             s_arready,
  output logic [2*256-1:0]       s_rdata,
  output logic [2*2-1:0]         s_rresp,
  output logic [1:0]             s_rlast,
  output logic [1:0]             s_rvalid,
  input  logic [1:0]             s_rready,

  // ---- HBM pseudo-channel port (AXI3 style, 256-bit, HBM clock)
  output logic [PC_AW-1:0]       m_awaddr,
  output logic [3:0]             m_awlen,
  output logic                   m_awvalid,
  input  logic                   m_awready,
  output logic [255:0]           m_wdata,
  output logic [31:0]            m_wstrb,
  output logic                   m_wlast,
  output logic                   m_wvalid,
  input  logic                   m_wready,
  input  logic [1:0]             m_bresp,
  input  logic                   m_bvalid,
  output logic                   m_bready,
  output logic [PC_AW-1:0]       m_araddr,
  output logic [3:0]             m_arlen,
  output logic                   m_arvalid,
  input  logic                   m_arready,
  input  logic [255:0]           m_rdata,
  input  logic [1:0]             m_rresp,
  input  logic                   m_rlast,
  input  logic                   m_rvalid,
  output logic                   m_rready
);
  localparam int AWW = PC_AW + 4;
  localparam int WW  = 256 + 32 + 1;
  localparam int RW  = 256 + 2 + 1;

  // ---- per-source crossings
  logic [1:0]        aw_rv, aw_rr, w_rv, w_rr, ar_rv, ar_rr;     // request FIFOs, HBM side
  logic [1:0][AWW-1:0] aw_rd, ar_rd;
  logic [1:0][WW-1:0]  w_rd;
  logic [1:0]        b_wv, b_wr, r_wv, r_wr;                     // response FIFOs, HBM side
  logic [1:0][1:0]   b_wd;
  logic [1:0][RW-1:0] r_wd;

  for (genvar i = 0; i < 2; i++) begin : g_src
    f2_async_fifo #(.W(AWW), .AW(FAW)) u_aw (
      .wclk(s_clk[i]), .wrst(s_rst[i]),
      .wvalid(s_awvalid[i]), .wready(s_awready[i]),
      .wdata({s_awlen[i*4 +: 4], s_awaddr[i*PC_AW +: PC_AW]}),
      .rclk(hbm_clk), .rrst(hbm_rst),
      .rvalid(aw_rv[i]), .rready(aw_rr[i]), .rdata(aw_rd[i]));
    f2_async_fifo #(.W(WW), .AW(FAW)) u_w (
      .wclk(s_clk[i]), .wrst(s_rst[i]),
      .wvalid(s_wvalid[i]), .wready(s_wready[i]),
      .wdata({s_wlast[i], s_wstrb[i*32 +: 32], s_wdata[i*256 +: 256]}),
      .rclk(hbm_clk), .rrst(hbm_rst),
      .rvalid(w_rv[i]), .rready(w_rr[i]), .rdata(w_rd[i]));
    f2_async_fifo #(.W(AWW), .AW(FAW)) u_ar (
      .wclk(s_clk[i]), .wrst(s_rst[i]),
      .wvalid(s_arvalid[i]), .wready(s_arready[i]),
      .wdata({s_arlen[i*4 +: 4], s_araddr[i*PC_AW +: PC_AW]}),
      .rclk(hbm_clk), .rrst(hbm_rst),
      .rvalid(ar_rv[i]), .rready(ar_rr[i]), .rdata(ar_rd[i]));
    f2_async_fifo #(.W(2), .AW(FAW)) u_b (
      .wclk(hbm_clk), .wrst(hbm_rst),
      .wvalid(b_wv[i]), .wready(b_wr[i]), .wdata(b_wd[i]),
      .rclk(s_clk[i]), .rrst(s_rst[i]),
      .rvalid(s_bvalid[i]), .rready(s_bready[i]), .rdata(s_bresp[i*2 +: 2]));
    f2_async_fifo #(.W(RW), .AW(FAW)) u_r (
      .wclk(hbm_clk), .wrst(hbm_rst),
      .wvalid(r_wv[i]), .wready(r_wr[i]), .wdata(r_wd[i]),
      .rclk(s_clk[i]), .rrst(s_rst[i]),
      .rvalid(s_rvalid[i]), .rready(s_rready[i]),
      .rdata({s_rlast[i], s_rresp[i*2 +: 2], s_rdata[i*256 +: 256]}));
  end

  // ---- AR arbitration and read-source order
  logic ar_busy, ar_sel, ar_pref;
  logic rsrc_wv, rsrc_wr, rsrc_rv, rsrc_rr, rsrc_rd;
  wire  ar_any = ar_rv[0] || ar_rv[1];
  assign m_arvalid = ar_busy && ar_rv[ar_sel] && rsrc_wr;
  assign m_araddr  = ar_rd[ar_sel][PC_AW-1:0];
  assign m_arlen   = ar_rd[ar_sel][PC_AW +: 4];
  wire   ar_fire   = m_arvalid && m_arready;
  assign rsrc_wv   = ar_fire;
  always_comb begin
    ar_rr = '0;
    if (ar_fire) ar_rr[ar_sel] = 1'b1;
  end
  always_ff @(posedge hbm_clk) begin
    if (hbm_rst) begin
      ar_busy <= 1'b0;
      ar_sel  <= 1'b0;
      ar_pref <= 1'b0;
    end else if (!ar_busy) begin
      if (ar_any) begin
        ar_busy <= 1'b1;
        ar_sel  <= (ar_rv[0] && ar_rv[1]) ? ar_pref : ar_rv[1];
      end
    end else if (ar_fire) begin
      ar_busy <= 1'b0;
      ar_pref <= !ar_sel;
    end
  end
  f2_sync_fifo #(.W(1), .AW(5)) u_rsrc (
    .clk(hbm_clk), .rst(hbm_rst),
    .wvalid(rsrc_wv), .wready(rsrc_wr), .wdata(ar_sel),
    .rvalid(rsrc_rv), .rready(rsrc_rr), .rdata(rsrc_rd));
  always_comb begin
    r_wv = '0;
    r_wd = '0;
    m_rready = 1'b0;
    if (rsrc_rv) begin
      m_rready = r_wr[rsrc_rd];
      r_wv[rsrc_rd] = m_rvalid;
      r_wd[rsrc_rd] = {m_rlast, m_rresp, m_rdata};
    end
  end
  assign rsrc_rr = rsrc_rv && m_rvalid && m_rready && m_rlast;

  // ---- AW arbitration; W and B follow the grant order
  logic aw_busy, aw_sel, aw_pref;
  logic wsrc_wr, wsrc_rv, wsrc_rr, wsrc_rd;
  logic bsrc_wr, bsrc_rv, bsrc_rr, bsrc_rd;
  wire  aw_any = aw_rv[0] || aw_rv[1];
  assign m_awvalid = aw_busy && aw_rv[aw_sel] && wsrc_wr && bsrc_wr;
  assign m_awaddr  = aw_rd[aw_sel][PC_AW-1:0];
  assign m_awlen   = aw_rd[aw_sel][PC_AW +: 4];
  wire   aw_fire   = m_awvalid && m_awready;
  always_comb begin
    aw_rr = '0;
    if (aw_fire) aw_rr[aw_sel] = 1'b1;
  end
  always_ff @(posedge hbm_clk) begin
    if (hbm_rst) begin
      aw_busy <= 1'b0;
      aw_sel  <= 1'b0;
      aw_pref <= 1'b0;
    end else if (!aw_busy) begin
      if (aw_any) begin
        aw_busy <= 1'b1;
        aw_sel  <= (aw_rv[0] && aw_rv[1]) ? aw_pref : aw_rv[1];
      end
    end else if (aw_fire) begin
      aw_busy <= 1'b0;
      aw_pref <= !aw_sel;
    end
  end
  f2_sync_fifo #(.W(1), .AW(5)) u_wsrc (
    .clk(hbm_clk), .rst(hbm_rst),
    .wvalid(aw_fire), .wready(wsrc_wr), .wdata(aw_sel),
    .rvalid(wsrc_rv), .rready(wsrc_rr), .rdata(wsrc_rd));
  f2_sync_fifo #(.W(1), .AW(5)) u_bsrc (
    .clk(hbm_clk), .rst(hbm_rst),
    .wvalid(aw_fire), .wready(bsrc_wr), .wdata(aw_sel),
    .rvalid(bsrc_rv), .rready(bsrc_rr), .rdata(bsrc_rd));

  // W: the head of wsrc names the source whose W stream is forwarded
  assign m_wvalid = wsrc_rv && w_rv[wsrc_rd];
  assign m_wdata  = w_rd[wsrc_rd][255:0];
  assign m_wstrb  = w_rd[wsrc_rd][256 +: 32];
  assign m_wlast  = w_rd[wsrc_rd][288];
  always_comb begin
    w_rr = '0;
    if (wsrc_rv && m_wready) w_rr[wsrc_rd] = w_rv[wsrc_rd];
  end
  assign wsrc_rr = m_wvalid && m_wready && m_wlast;

  // B: returned to the source at the head of bsrc
  always_comb begin
    b_wv = '0;
    b_wd = '0;
    m_bready = 1'b0;
    if (bsrc_rv) begin
      m_bready = b_wr[bsrc_rd];
      b_wv[bsrc_rd] = m_bvalid;
      b_wd[bsrc_rd] = m_bresp;
    end
  end
  assign bsrc_rr = bsrc_rv && m_bvalid && m_bready;
endmodule
