// F2 platform: stripe router from one 512-bit AXI4 slave port to NPC HBM
// pseudo-channel (PC) masters (256-bit, AXI3 style: at most 16 beats per burst).
//
// One router, one clock domain. Three instances exist in the adapter: one for
// each of the accelerator's two memory channels (core clock) and one for the
// host PCIS path (shell main clock). All three use the same address function,
// so a byte written by the host lands where the core reads it.
//
//   router address `sa` (bytes, 64-byte beats):
//     pc    = {grp, sa[STRIPE +: LOGPCH]}               stripe -> PC
//     local = {sa[AW-1 : STRIPE+LOGPCH], sa[STRIPE-1:0]} address inside the PC
//   where LOGPCH = log2(NPC) - GRPB. grp is the optional group select (GRPB = 1: the
//   PCIS router selects the core channel with it, the address stays linear so that
//   burst splitting can simply add). local must fit in PC_AW bits, else the piece is
//   answered here with DECERR (no PC is touched).
//
// STRIPE = 9 (512 B) is the largest stripe for which a piece is one legal
// 16-beat AXI3 burst of 256-bit beats, so every burst of a piece stays on one PC.
// Bursts are split at stripe boundaries; the pieces of a burst can go to
// different PCs, so responses are merged strictly in issue order (an entry FIFO
// records the PC and shape of every piece).
//
// Width conversion: each 512-bit beat becomes two 256-bit beats (low half
// first). Read data is recombined the same way.
//
// AXI subset: INCR bursts, full 64-byte beats (address bits [5:0] ignored,
// AxSIZE ignored), any AxLEN up to 255; W may arrive before AW (it is held until
// the AW piece has been issued). One ID per burst, returned unchanged.
//
// UNTESTED on hardware; not run through Vivado.
module f2_hbm_router #(
  parameter int NPC        = 2,     // PCs behind this router, power of two >= 2
  parameter int GRPB       = 0,     // 1: the top PC index bit comes from s_awgrp / s_argrp
  parameter int AW         = 32,    // router address width
  parameter int PC_AW      = 29,    // PC-local address bits (512 MiB in the real HBM)
  parameter int IDW        = 1,
  parameter int STRIPE     = 9,     // log2 bytes per stripe; must be <= 9 and >= 6
  parameter int DEPTH_LOG2 = 5      // outstanding pieces per direction
) (
  input  logic                    clk,
  input  logic                    rst,           // synchronous, active high

  // ---- slave: AXI4, 512-bit
  input  logic [IDW-1:0]          s_awid,
  input  logic [AW-1:0]           s_awaddr,
  input  logic                    s_awgrp,
  input  logic [7:0]              s_awlen,
  input  logic                    s_awvalid,
  output logic                    s_awready,
  input  logic [511:0]            s_wdata,
  input  logic [63:0]             s_wstrb,
  input  logic                    s_wlast,
  input  logic                    s_wvalid,
  output logic                    s_wready,
  output logic [IDW-1:0]          s_bid,
  output logic [1:0]              s_bresp,
  output logic                    s_bvalid,
  input  logic                    s_bready,
  input  logic [IDW-1:0]          s_arid,
  input  logic [AW-1:0]           s_araddr,
  input  logic                    s_argrp,
  input  logic [7:0]              s_arlen,
  input  logic                    s_arvalid,
  output logic                    s_arready,
  output logic [IDW-1:0]          s_rid,
  output logic [511:0]            s_rdata,
  output logic [1:0]              s_rresp,
  output logic                    s_rlast,
  output logic                    s_rvalid,
  input  logic                    s_rready,

  // ---- masters: one AXI3-style 256-bit port per PC (flattened, PC p in slice p)
  output logic [NPC*PC_AW-1:0]    m_awaddr,
  output logic [NPC*4-1:0]        m_awlen,
  output logic [NPC-1:0]          m_awvalid,
  input  logic [NPC-1:0]          m_awready,
  output logic [NPC*256-1:0]      m_wdata,
  output logic [NPC*32-1:0]       m_wstrb,
  output logic [NPC-1:0]          m_wlast,
  output logic [NPC-1:0]          m_wvalid,
  input  logic [NPC-1:0]          m_wready,
  input  logic [NPC*2-1:0]        m_bresp,
  input  logic [NPC-1:0]          m_bvalid,
  output logic [NPC-1:0]          m_bready,
  output logic [NPC*PC_AW-1:0]    m_araddr,
  output logic [NPC*4-1:0]        m_arlen,
  output logic [NPC-1:0]          m_arvalid,
  input  logic [NPC-1:0]          m_arready,
  input  logic [NPC*256-1:0]      m_rdata,
  input  logic [NPC*2-1:0]        m_rresp,
  input  logic [NPC-1:0]          m_rlast,
  input  logic [NPC-1:0]          m_rvalid,
  output logic [NPC-1:0]          m_rready,

  output logic                    err_pulse      // one cycle per error piece or error response
);
  localparam int LOGPC = $clog2(NPC);
  localparam int LOGPCH = LOGPC - GRPB;           // PC index bits taken from the address
  localparam int LAW   = AW - LOGPCH;             // local address width before the range check
  localparam int E     = 2 + 4 + IDW + LOGPC;     // entry: {err, last, pb, id, pc}
  localparam int DL    = DEPTH_LOG2;

  if (NPC < 2 || (1 << LOGPC) != NPC) begin : g_bad_npc
    $error("f2_hbm_router: NPC must be a power of two >= 2");
  end
  if (STRIPE < 6 || STRIPE > 9) begin : g_bad_stripe
    $error("f2_hbm_router: STRIPE must be 6..9 (a piece must fit one 16-beat 256-bit burst)");
  end
  if (GRPB < 0 || GRPB > 1 || LOGPCH < 1) begin : g_bad_grp
    $error("f2_hbm_router: GRPB must be 0 or 1 and leave at least one address-selected PC bit");
  end
  if (LAW <= PC_AW) begin : g_bad_aw
    $error("f2_hbm_router: AW - address-selected PC bits must exceed PC_AW");
  end

  // ------------------------------------------------------------------ shared helpers
  function automatic logic [3:0] avail_beats(input logic [AW-1:0] a);
    // 64-byte beats left in the stripe that contains a: 1 .. 2**(STRIPE-6)
    avail_beats = 4'((1 << (STRIPE - 6)) - int'(a[STRIPE-1:6]));
  endfunction
  function automatic logic [3:0] piece_beats(input logic [8:0] rem, input logic [3:0] avail);
    piece_beats = (rem < {5'b0, avail}) ? rem[3:0] : avail;
  endfunction
  function automatic logic [E-1:0] pack(input logic err, input logic last, input logic [3:0] pb,
                                        input logic [IDW-1:0] id, input logic [LOGPC-1:0] pc);
    pack = {err, last, pb, id, pc};
  endfunction

  logic err_ar, err_aw, err_rsp, err_b;
  assign err_pulse = err_ar || err_aw || err_rsp || err_b;

  // ================================================================== read path
  logic              ar_act;
  logic [AW-1:0]     ar_a;
  logic [8:0]        ar_rem;
  logic [IDW-1:0]    ar_id;
  logic              ar_grp;
  wire  [LOGPC-1:0]  ar_pc    = LOGPC'(ar_a[STRIPE +: LOGPCH]) | (LOGPC'(ar_grp && GRPB != 0) << LOGPCH);
  wire  [LAW-1:0]    ar_loc   = {ar_a[AW-1 : STRIPE+LOGPCH], ar_a[STRIPE-1:0]};
  wire               ar_ok    = (ar_loc >> PC_AW) == '0;
  wire  [3:0]        ar_pb    = piece_beats(ar_rem, avail_beats(ar_a));
  wire               ar_last  = ar_rem == {5'b0, ar_pb};

  logic              rq_wvalid, rq_wready, rq_rvalid, rq_rready;
  logic [E-1:0]      rq_rdata;
  assign s_arready = !ar_act;
  wire ar_fire_pc  = ar_act && ar_ok && rq_wready && m_arready[ar_pc];
  wire ar_fire_err = ar_act && !ar_ok && rq_wready;
  assign rq_wvalid = ar_fire_pc || ar_fire_err;
  assign err_ar    = ar_fire_err;

  always_comb begin
    m_arvalid = '0;
    m_araddr  = {NPC{ar_loc[PC_AW-1:0]}};
    m_arlen   = {NPC{4'(2 * int'(ar_pb) - 1)}};
    if (ar_act && ar_ok && rq_wready) m_arvalid[ar_pc] = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      ar_act <= 1'b0;
    end else if (!ar_act) begin
      if (s_arvalid) begin
        ar_act <= 1'b1;
        ar_a   <= {s_araddr[AW-1:6], 6'b0};
        ar_rem <= {1'b0, s_arlen} + 9'd1;
        ar_id  <= s_arid;
        ar_grp <= s_argrp;
      end
    end else if (rq_wvalid) begin
      ar_a   <= ar_a + AW'({ar_pb, 6'b0});
      ar_rem <= ar_rem - {5'b0, ar_pb};
      if (ar_last) ar_act <= 1'b0;
    end
  end

  f2_sync_fifo #(.W(E), .AW(DL)) u_rq (
    .clk, .rst,
    .wvalid(rq_wvalid), .wready(rq_wready),
    .wdata(pack(!ar_ok, ar_last, ar_pb, ar_id, ar_pc)),
    .rvalid(rq_rvalid), .rready(rq_rready), .rdata(rq_rdata));

  // read data: entries are consumed in issue order; two 256-bit beats -> one 512-bit beat
  wire [LOGPC-1:0] rh_pc   = rq_rdata[LOGPC-1:0];
  wire [IDW-1:0]   rh_id   = rq_rdata[LOGPC +: IDW];
  wire [3:0]       rh_pb   = rq_rdata[LOGPC+IDW +: 4];
  wire             rh_last = rq_rdata[LOGPC+IDW+4];
  wire             rh_err  = rq_rdata[LOGPC+IDW+5];

  logic [3:0]      r_cnt;
  logic            r_half;
  logic [255:0]    r_lo;
  logic [1:0]      r_lo_resp;
  wire  [255:0]    rd_hi   = m_rdata[rh_pc * 256 +: 256];
  wire  [1:0]      rd_resp = m_rresp[rh_pc * 2 +: 2];
  wire             rd_v    = m_rvalid[rh_pc];
  wire             r_beat_last = r_cnt == rh_pb - 4'd1;
  wire             r_out_fire  = s_rvalid && s_rready;

  always_comb begin
    m_rready = '0;
    s_rvalid = 1'b0;
    s_rdata  = '0;
    s_rresp  = 2'b00;
    s_rid    = rh_id;
    s_rlast  = rq_rvalid && rh_last && r_beat_last;
    rq_rready = 1'b0;
    err_rsp   = 1'b0;
    if (rq_rvalid) begin
      if (rh_err) begin
        s_rvalid = 1'b1;
        s_rresp  = 2'b11;
        rq_rready = s_rready && r_beat_last;
      end else if (!r_half) begin
        m_rready[rh_pc] = 1'b1;
      end else begin
        s_rvalid = rd_v;
        s_rdata  = {rd_hi, r_lo};
        s_rresp  = r_lo_resp | rd_resp;
        m_rready[rh_pc] = s_rready;
        rq_rready = rd_v && s_rready && r_beat_last;
      end
    end
    if (r_out_fire && s_rresp != 2'b00 && !rh_err) err_rsp = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      r_cnt  <= '0;
      r_half <= 1'b0;
    end else if (rq_rvalid) begin
      if (rh_err) begin
        if (s_rready) r_cnt <= r_beat_last ? 4'd0 : r_cnt + 4'd1;
      end else if (!r_half) begin
        if (rd_v) begin
          r_lo      <= rd_hi;
          r_lo_resp <= rd_resp;
          r_half    <= 1'b1;
        end
      end else if (rd_v && s_rready) begin
        r_half <= 1'b0;
        r_cnt  <= r_beat_last ? 4'd0 : r_cnt + 4'd1;
      end
    end
  end

  // ================================================================== write path
  logic              aw_act;
  logic [AW-1:0]     aw_a;
  logic [8:0]        aw_rem;
  logic [IDW-1:0]    aw_id;
  logic              aw_grp;
  wire  [LOGPC-1:0]  aw_pc    = LOGPC'(aw_a[STRIPE +: LOGPCH]) | (LOGPC'(aw_grp && GRPB != 0) << LOGPCH);
  wire  [LAW-1:0]    aw_loc   = {aw_a[AW-1 : STRIPE+LOGPCH], aw_a[STRIPE-1:0]};
  wire               aw_ok    = (aw_loc >> PC_AW) == '0;
  wire  [3:0]        aw_pb    = piece_beats(aw_rem, avail_beats(aw_a));
  wire               aw_last  = aw_rem == {5'b0, aw_pb};

  // entry buffer shared by the W engine (pointer wp) and the B engine (pointer bp)
  logic [E-1:0]      wbuf [1 << DL];
  logic [DL:0]       wb_wr, wb_wp, wb_bp;
  wire               wb_full = (wb_wr - wb_bp) == (DL+1)'(1 << DL);

  assign s_awready = !aw_act;
  wire aw_fire_pc  = aw_act && aw_ok && !wb_full && m_awready[aw_pc];
  wire aw_fire_err = aw_act && !aw_ok && !wb_full;
  wire aw_push     = aw_fire_pc || aw_fire_err;
  assign err_aw    = aw_fire_err;

  always_comb begin
    m_awvalid = '0;
    m_awaddr  = {NPC{aw_loc[PC_AW-1:0]}};
    m_awlen   = {NPC{4'(2 * int'(aw_pb) - 1)}};
    if (aw_act && aw_ok && !wb_full) m_awvalid[aw_pc] = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (aw_push) wbuf[wb_wr[DL-1:0]] <= pack(!aw_ok, aw_last, aw_pb, aw_id, aw_pc);
    if (rst) begin
      aw_act <= 1'b0;
      wb_wr  <= '0;
    end else begin
      if (!aw_act) begin
        if (s_awvalid) begin
          aw_act <= 1'b1;
          aw_a   <= {s_awaddr[AW-1:6], 6'b0};
          aw_rem <= {1'b0, s_awlen} + 9'd1;
          aw_id  <= s_awid;
          aw_grp <= s_awgrp;
        end
      end else if (aw_push) begin
        aw_a   <= aw_a + AW'({aw_pb, 6'b0});
        aw_rem <= aw_rem - {5'b0, aw_pb};
        if (aw_last) aw_act <= 1'b0;
      end
      if (aw_push) wb_wr <= wb_wr + 1'b1;
    end
  end

  // ---- W engine: head entry at wp
  wire             wh_v    = wb_wp != wb_wr;
  wire [E-1:0]     wh      = wbuf[wb_wp[DL-1:0]];
  wire [LOGPC-1:0] wh_pc   = wh[LOGPC-1:0];
  wire [3:0]       wh_pb   = wh[LOGPC+IDW +: 4];
  wire             wh_err  = wh[LOGPC+IDW+5];
  logic [3:0]      w_cnt;
  logic            w_half;
  wire             w_beat_last = w_cnt == wh_pb - 4'd1;
  wire [255:0]     w_d = w_half ? s_wdata[511:256] : s_wdata[255:0];
  wire [31:0]      w_s = w_half ? s_wstrb[63:32]   : s_wstrb[31:0];
  wire             w_step = wh_v && (wh_err ? s_wvalid
                                            : (s_wvalid && m_wready[wh_pc]));
  wire             w_pop  = w_step && (wh_err || w_half) && w_beat_last;

  always_comb begin
    m_wvalid = '0;
    m_wdata  = {NPC{w_d}};
    m_wstrb  = {NPC{w_s}};
    m_wlast  = {NPC{w_half && w_beat_last}};
    s_wready = 1'b0;
    if (wh_v) begin
      if (wh_err) begin
        s_wready = s_wvalid;
      end else begin
        m_wvalid[wh_pc] = s_wvalid;
        s_wready        = s_wvalid && m_wready[wh_pc] && w_half;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      w_cnt  <= '0;
      w_half <= 1'b0;
      wb_wp  <= '0;
    end else if (w_step) begin
      if (wh_err) begin
        w_cnt <= w_beat_last ? 4'd0 : w_cnt + 4'd1;
      end else if (!w_half) begin
        w_half <= 1'b1;
      end else begin
        w_half <= 1'b0;
        w_cnt  <= w_beat_last ? 4'd0 : w_cnt + 4'd1;
      end
      if (w_pop) wb_wp <= wb_wp + 1'b1;
    end
  end

  // ---- B engine: head entry at bp; the burst's B is sent after its last piece completes
  wire             bh_v    = wb_bp != wb_wr;
  wire [E-1:0]     bh      = wbuf[wb_bp[DL-1:0]];
  wire [LOGPC-1:0] bh_pc   = bh[LOGPC-1:0];
  wire [IDW-1:0]   bh_id   = bh[LOGPC +: IDW];
  wire             bh_last = bh[LOGPC+IDW+4];
  wire             bh_err  = bh[LOGPC+IDW+5];
  logic [1:0]      b_acc;
  wire  [1:0]      bh_resp = bh_err ? 2'b11 : m_bresp[bh_pc * 2 +: 2];
  wire             bh_ready = bh_err ? (wb_bp != wb_wp) : m_bvalid[bh_pc];  // an error piece waits for its W beats
  wire             b_consume = bh_v && bh_ready && (!bh_last || s_bready);
  assign err_b = b_consume && !bh_err && (bh_resp != 2'b00);

  always_comb begin
    m_bready = '0;
    s_bvalid = 1'b0;
    s_bid    = bh_id;
    s_bresp  = b_acc | bh_resp;
    if (bh_v && !bh_err) m_bready[bh_pc] = !bh_last || s_bready;
    if (bh_v && bh_ready && bh_last) s_bvalid = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      wb_bp <= '0;
      b_acc <= 2'b00;
    end else if (b_consume) begin
      wb_bp <= wb_bp + 1'b1;
      b_acc <= bh_last ? 2'b00 : (b_acc | bh_resp);
    end
  end
endmodule
