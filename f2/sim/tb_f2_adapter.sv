// Self-checking bench for f2_hbm_adapter: random traffic on the two core channels
// (core clock) and the PCIS port (main clock) into the HBM model (HBM clock), three
// unrelated clocks. Checks:
//   1. placement: after the host wrote patterns through PCIS, every byte sits where
//      the documented stripe map says (backdoor read of the HBM model);
//   2. coherence: the core channels read back exactly what PCIS wrote, and vice versa
//      (including partial write strobes), under random ready stalls on every port;
//   3. error responses: PCIS outside the window and core offsets beyond the PC range
//      answer DECERR without a hang and without touching memory;
//   4. concurrent core and PCIS traffic completes (no starvation, no deadlock).
// Prints "F2_ADAPTER PASS" or fails with $fatal.
`timescale 1ns/1ps
module tb_f2_adapter;
  parameter int PCS   = 2;
  parameter int PC_AW = 21;
  localparam int NPCT = 2 * PCS;
  localparam int REG_BEATS = 1024;        // 64 KiB per channel tested
  localparam int TOTAL_BEATS = 2048;

  logic core_clk = 0, main_clk = 0, hbm_clk = 0;
  logic core_rst = 1, main_rst = 1, hbm_rst = 1;
  always #4.0 core_clk = ~core_clk;      // 125 MHz
  always #2.0 main_clk = ~main_clk;      // 250 MHz
  always #1.1 hbm_clk  = ~hbm_clk;       // ~454 MHz

  // core channels
  logic [1:0]        c_awid = '0, c_awvalid = '0, c_wlast = '0, c_wvalid = '0, c_bready = '0;
  logic [1:0]        c_arid = '0, c_arvalid = '0, c_rready = '0;
  logic [2*32-1:0]   c_awaddr = '0, c_araddr = '0;
  logic [2*512-1:0]  c_wdata = '0;
  logic [2*64-1:0]   c_wstrb = '0;
  logic [2*8-1:0]    c_arlen = '0;
  wire  [1:0]        c_awready, c_wready, c_bid, c_bvalid, c_arready, c_rid, c_rlast, c_rvalid;
  wire  [2*2-1:0]    c_bresp, c_rresp;
  wire  [2*512-1:0]  c_rdata;
  // PCIS
  logic [15:0] p_awid = 0, p_arid = 0;
  logic [63:0] p_awaddr = 0, p_araddr = 0;
  logic [7:0]  p_awlen = 0, p_arlen = 0;
  logic        p_awvalid = 0, p_wlast = 0, p_wvalid = 0, p_bready = 0, p_arvalid = 0, p_rready = 0;
  logic [511:0] p_wdata = 0;
  logic [63:0]  p_wstrb = 0;
  wire         p_awready, p_wready, p_bvalid, p_arready, p_rlast, p_rvalid;
  wire [15:0]  p_bid, p_rid;
  wire [1:0]   p_bresp, p_rresp;
  wire [511:0] p_rdata;
  // HBM
  wire [NPCT*PC_AW-1:0]  h_awaddr, h_araddr;
  wire [NPCT*4-1:0]      h_awlen, h_arlen;
  wire [NPCT-1:0]        h_awvalid, h_awready, h_wlast, h_wvalid, h_wready, h_bvalid, h_bready;
  wire [NPCT-1:0]        h_arvalid, h_arready, h_rlast, h_rvalid, h_rready;
  wire [NPCT*256-1:0]    h_wdata, h_rdata;
  wire [NPCT*32-1:0]     h_wstrb;
  wire [NPCT*2-1:0]      h_bresp, h_rresp;
  wire core_err, pcis_err;

  f2_hbm_adapter #(.PCS_PER_CH(PCS), .PC_AW(PC_AW)) dut (
    .core_clk, .core_rst, .main_clk, .main_rst, .hbm_clk, .hbm_rst,
    .c_awid, .c_awaddr, .c_awvalid, .c_awready, .c_wdata, .c_wstrb, .c_wlast, .c_wvalid,
    .c_wready, .c_bid, .c_bresp, .c_bvalid, .c_bready, .c_arid, .c_araddr, .c_arlen,
    .c_arvalid, .c_arready, .c_rid, .c_rdata, .c_rresp, .c_rlast, .c_rvalid, .c_rready,
    .p_awid, .p_awaddr, .p_awlen, .p_awvalid, .p_awready, .p_wdata, .p_wstrb, .p_wlast,
    .p_wvalid, .p_wready, .p_bid, .p_bresp, .p_bvalid, .p_bready, .p_arid, .p_araddr,
    .p_arlen, .p_arvalid, .p_arready, .p_rid, .p_rdata, .p_rresp, .p_rlast, .p_rvalid,
    .p_rready,
    .hbm_awaddr(h_awaddr), .hbm_awlen(h_awlen), .hbm_awvalid(h_awvalid), .hbm_awready(h_awready),
    .hbm_wdata(h_wdata), .hbm_wstrb(h_wstrb), .hbm_wlast(h_wlast), .hbm_wvalid(h_wvalid),
    .hbm_wready(h_wready), .hbm_bresp(h_bresp), .hbm_bvalid(h_bvalid), .hbm_bready(h_bready),
    .hbm_araddr(h_araddr), .hbm_arlen(h_arlen), .hbm_arvalid(h_arvalid),
    .hbm_arready(h_arready), .hbm_rdata(h_rdata), .hbm_rresp(h_rresp), .hbm_rlast(h_rlast),
    .hbm_rvalid(h_rvalid), .hbm_rready(h_rready),
    .core_err_pulse(core_err), .pcis_err_pulse(pcis_err));

  f2_hbm_model #(.NPC(NPCT), .PC_AW(PC_AW)) hbm (
    .clk(hbm_clk), .rst(hbm_rst),
    .awaddr(h_awaddr), .awlen(h_awlen), .awvalid(h_awvalid), .awready(h_awready),
    .wdata(h_wdata), .wstrb(h_wstrb), .wlast(h_wlast), .wvalid(h_wvalid), .wready(h_wready),
    .bresp(h_bresp), .bvalid(h_bvalid), .bready(h_bready),
    .araddr(h_araddr), .arlen(h_arlen), .arvalid(h_arvalid), .arready(h_arready),
    .rdata(h_rdata), .rresp(h_rresp), .rlast(h_rlast), .rvalid(h_rvalid), .rready(h_rready));

  // ---- free-running random ready generators on the response channels
  always @(posedge core_clk) begin
    c_bready <= 2'($urandom);
    c_rready <= 2'($urandom) | 2'($urandom);
  end
  always @(posedge main_clk) begin
    p_bready <= ($urandom % 4) != 0;
    p_rready <= ($urandom % 4) != 0;
  end

  // ---- error counters
  int core_errs = 0, pcis_errs = 0;
  always @(posedge core_clk) if (core_err) core_errs++;
  always @(posedge main_clk) if (pcis_err) pcis_errs++;

  // ---- patterns: beat data as a function of (generation, channel, beat)
  function automatic logic [511:0] pat(input int gen, input int ch, input int beat);
    logic [511:0] v;
    logic [31:0] x;
    x = 32'(gen * 32'h9E3779B1) ^ 32'(ch * 32'h85EBCA6B) ^ 32'(beat * 32'hC2B2AE35) ^ 32'h1234567;
    for (int i = 0; i < 16; i++) begin
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      v[i*32 +: 32] = x;
    end
    return v;
  endfunction

  // expected physical byte location of channel offset `o` of channel `ch`
  function automatic int exp_pc(input int ch, input longint o);
    return ch * PCS + int'((o >> 9) % PCS);
  endfunction
  function automatic longint exp_local(input longint o);
    return ((o >> 9) / PCS) * 512 + (o & 511);
  endfunction

  // =============================================================== bus tasks
  int fails = 0;
  task automatic fail(input string m);
    fails++;
    $display("FAIL: %s", m);
  endtask

  // core write: single 64-byte beat, AW and W presented together (as otpu_axi_dram does)
  task automatic core_write(input int ch, input longint off, input logic [511:0] d, input logic [63:0] st);
    bit aw_d = 0, w_d = 0;
    c_awaddr[ch*32 +: 32] <= 32'(off) | (ch ? 32'h8000_0000 : 0);
    c_awvalid[ch] <= 1; c_wdata[ch*512 +: 512] <= d; c_wstrb[ch*64 +: 64] <= st;
    c_wlast[ch] <= 1; c_wvalid[ch] <= 1;
    while (!(aw_d && w_d)) begin
      @(posedge core_clk);
      if (!aw_d && c_awready[ch]) begin aw_d = 1; c_awvalid[ch] <= 0; end
      if (!w_d && c_wready[ch]) begin w_d = 1; c_wvalid[ch] <= 0; end
    end
    do @(posedge core_clk); while (!(c_bvalid[ch] && c_bready[ch]));
    if (c_bresp[ch*2 +: 2] != 2'b00) fail($sformatf("core write ch%0d off %0d bresp %0d", ch, off, c_bresp[ch*2 +: 2]));
  endtask

  // core read burst of n beats (1..8), data compared with pat(gen, ch, beat) when check
  task automatic core_read(input int ch, input longint off, input int n, input int gen,
                           input bit check, input bit expect_err);
    int got = 0;
    c_araddr[ch*32 +: 32] <= 32'(off) | (ch ? 32'h8000_0000 : 0);
    c_arlen[ch*8 +: 8] <= 8'(n - 1);
    c_arvalid[ch] <= 1;
    do @(posedge core_clk); while (!c_arready[ch]);
    c_arvalid[ch] <= 0;
    while (got < n) begin
      @(posedge core_clk);
      if (c_rvalid[ch] && c_rready[ch]) begin
        if (expect_err) begin
          if (c_rresp[ch*2 +: 2] != 2'b11) fail("core read: expected DECERR");
        end else begin
          if (c_rresp[ch*2 +: 2] != 2'b00) fail($sformatf("core read ch%0d off %0d rresp %0d", ch, off, c_rresp[ch*2 +: 2]));
          if (check && c_rdata[ch*512 +: 512] !== pat(gen, ch, int'(off / 64) + got))
            fail($sformatf("core read ch%0d beat %0d data mismatch", ch, int'(off / 64) + got));
        end
        if ((got == n - 1) != c_rlast[ch]) fail("core read: rlast wrong");
        got++;
      end
    end
  endtask

  // PCIS write of n beats at host address a with data pat(gen, ch, beat) (or expected err)
  task automatic pcis_write(input logic [63:0] a, input int n, input int gen, input int ch,
                            input longint first_beat, input bit expect_err);
    bit aw_d = 0;
    int sent = 0;
    p_awaddr <= a; p_awlen <= 8'(n - 1); p_awvalid <= 1; p_awid <= 16'hA5A5;
    while (!aw_d || sent < n) begin
      if (sent < n && !p_wvalid) begin
        p_wdata <= pat(gen, ch, int'(first_beat) + sent);
        p_wstrb <= '1; p_wlast <= (sent == n - 1); p_wvalid <= (($urandom % 4) != 0);
      end
      @(posedge main_clk);
      if (!aw_d && p_awready) begin aw_d = 1; p_awvalid <= 0; end
      if (p_wvalid && p_wready) begin sent++; p_wvalid <= 0; end
    end
    do @(posedge main_clk); while (!(p_bvalid && p_bready));
    if (p_bid !== 16'hA5A5) fail("pcis write: bid");
    if (expect_err) begin
      if (p_bresp != 2'b11) fail("pcis write: expected DECERR");
    end else if (p_bresp != 2'b00) fail($sformatf("pcis write bresp %0d", p_bresp));
  endtask

  task automatic pcis_read(input logic [63:0] a, input int n, input int gen, input int ch,
                           input longint first_beat, input bit check, input bit expect_err);
    int got = 0;
    p_araddr <= a; p_arlen <= 8'(n - 1); p_arvalid <= 1; p_arid <= 16'h5A5A;
    do @(posedge main_clk); while (!p_arready);
    p_arvalid <= 0;
    while (got < n) begin
      @(posedge main_clk);
      if (p_rvalid && p_rready) begin
        if (p_rid !== 16'h5A5A) fail("pcis read: rid");
        if (expect_err) begin
          if (p_rresp != 2'b11) fail("pcis read: expected DECERR");
        end else begin
          if (p_rresp != 2'b00) fail($sformatf("pcis read rresp %0d", p_rresp));
          if (check && p_rdata !== pat(gen, ch, int'(first_beat) + got))
            fail($sformatf("pcis read ch%0d beat %0d data mismatch", ch, int'(first_beat) + got));
        end
        if ((got == n - 1) != p_rlast) fail("pcis read: rlast wrong");
        got++;
      end
    end
  endtask

  function automatic logic [63:0] haddr(input int ch, input longint beat);
    return (64'd1 << 36) | (ch ? 64'h8000_0000 : 64'd0) | (beat * 64);
  endfunction

  // =============================================================== test sequence
  int phase = 0;
  int done2 = 0, done3 = 0;

  // random burst partition of [0, total) beats into bursts of 1..maxn that stay in a 4 KiB page
  task automatic pcis_fill(input int ch, input int gen, input int first, input int total, input int maxn);
    int b = first;
    while (b < first + total) begin
      automatic int n = 1 + $urandom % maxn;
      automatic int room = 64 - (b % 64);
      if (n > room) n = room;
      if (b + n > first + total) n = first + total - b;
      pcis_write(haddr(ch, b), n, gen, ch, b, 0);
      b += n;
    end
  endtask
  task automatic pcis_verify(input int ch, input int gen, input int first, input int total, input int maxn);
    int b = first;
    while (b < first + total) begin
      automatic int n = 1 + $urandom % maxn;
      automatic int room = 64 - (b % 64);
      if (n > room) n = room;
      if (b + n > first + total) n = first + total - b;
      pcis_read(haddr(ch, b), n, gen, ch, b, 1, 0);
      b += n;
    end
  endtask
  task automatic core_verify(input int ch, input int gen, input int first, input int total);
    int b = first;
    while (b < first + total) begin
      automatic int n = 1 + $urandom % 8;
      automatic int room = 64 - (b % 64);
      if (n > room) n = room;
      if (b + n > first + total) n = first + total - b;
      core_read(ch, longint'(b) * 64, n, gen, 1, 0);
      b += n;
    end
  endtask

  initial begin
    repeat (20) @(posedge hbm_clk);
    core_rst = 0; main_rst = 0; hbm_rst = 0;
    repeat (10) @(posedge main_clk);

    // ---- phase 1: host writes generation-1 patterns everywhere (both channels)
    $display("phase 1: PCIS fill");
    for (int ch = 0; ch < 2; ch++) pcis_fill(ch, 1, 0, TOTAL_BEATS, 16);

    // ---- placement check by backdoor
    $display("phase 1b: placement");
    for (int ch = 0; ch < 2; ch++)
      for (int b = 0; b < TOTAL_BEATS; b++) begin
        automatic logic [511:0] e = pat(1, ch, b);
        automatic int pc = exp_pc(ch, longint'(b) * 64);
        automatic longint loc = exp_local(longint'(b) * 64);
        for (int k = 0; k < 64; k++)
          if (hbm.mem[pc * (1 << PC_AW) + int'(loc) + k] !== e[k*8 +: 8]) begin
            fail($sformatf("placement ch%0d beat %0d byte %0d: pc %0d local %0d", ch, b, k, pc, loc));
            k = 64;
          end
      end

    // ---- phase 2: core reads back both channels while PCIS re-reads (concurrent)
    $display("phase 2: concurrent core + PCIS reads");
    fork
      begin core_verify(0, 1, 0, TOTAL_BEATS); done2++; end
      begin core_verify(1, 1, 0, TOTAL_BEATS); done2++; end
      begin
        for (int ch = 0; ch < 2; ch++) pcis_verify(ch, 1, 0, TOTAL_BEATS, 16);
        done2++;
      end
    join
    if (done2 != 3) fail("phase 2 incomplete");

    // ---- phase 3: core rewrites beats [0, REG_BEATS) (generation 2; the upper half of the beats keep
    //      strobes partial) while PCIS keeps reading the untouched [REG_BEATS, TOTAL_BEATS) range
    $display("phase 3: core writes, PCIS reads other range");
    fork
      begin
        for (int b = 0; b < REG_BEATS; b++) core_write(0, longint'(b) * 64, pat(2, 0, b), (b % 2) ? 64'h0000_0000_FFFF_FFFF : '1);
        done3++;
      end
      begin
        for (int b = 0; b < REG_BEATS; b++) core_write(1, longint'(b) * 64, pat(2, 1, b), '1);
        done3++;
      end
      begin
        for (int ch = 0; ch < 2; ch++) pcis_verify(ch, 1, REG_BEATS, TOTAL_BEATS - REG_BEATS, 16);
        done3++;
      end
    join
    if (done3 != 3) fail("phase 3 incomplete");

    // ---- phase 4: host reads what the core wrote. Odd beats on channel 0 had a partial strobe:
    //      low 32 bytes new (generation 2), high 32 bytes old (generation 1)
    $display("phase 4: PCIS reads core-written data");
    for (int b = 0; b < REG_BEATS; b++) begin
      automatic logic [511:0] e1 = pat(1, 0, b);
      automatic logic [511:0] e2 = pat(2, 0, b);
      automatic logic [511:0] e, got;
      e = (b % 2) ? {e1[511:256], e2[255:0]} : e2;
      pcis_read(haddr(0, b), 1, 0, 0, b, 0, 0);
      got = p_rdata;      // last beat read (single-beat burst)
      if (got !== e) fail($sformatf("phase 4 ch0 beat %0d", b));
    end
    for (int b = 0; b < REG_BEATS; b += 16) pcis_verify(1, 2, b, 16, 16);

    // ---- phase 5: errors
    $display("phase 5: error responses");
    begin
      int ce = core_errs, pe = pcis_errs;
      pcis_read(64'h0000_0000_0000_1000, 3, 0, 0, 0, 0, 1);          // DDR window: not implemented
      pcis_write(64'h0000_0020_0000_0000, 2, 0, 0, 0, 1);            // above the window
      pcis_read(haddr(0, 0) + 64'h1_0000_0000, 1, 0, 0, 0, 0, 1);    // beyond 4 GiB window
      pcis_read(haddr(0, (PCS * (1 << PC_AW)) / 64), 2, 0, 0, 0, 0, 1);  // beyond the PC range
      core_read(0, longint'(PCS) * (1 << PC_AW), 2, 0, 0, 1);          // core: beyond the PC range
      core_read(1, longint'(PCS) * (1 << PC_AW) + 448, 2, 0, 0, 1);    // straddles nothing valid
      repeat (50) @(posedge main_clk);
      if (pcis_errs == pe) fail("no PCIS error pulses");
      if (core_errs == ce) fail("no core error pulses");
    end
    // still alive after errors
    pcis_verify(0, 1, REG_BEATS, 16, 16);

    // ---- traffic reached every PC
    for (int p = 0; p < NPCT; p++)
      if (hbm.rd_beats[p] == 0 || hbm.wr_beats[p] == 0)
        fail($sformatf("PC %0d saw no traffic (rd %0d wr %0d)", p, hbm.rd_beats[p], hbm.wr_beats[p]));

    if (fails == 0) $display("F2_ADAPTER PASS (PCS=%0d, PC_AW=%0d)", PCS, PC_AW);
    else $fatal(1, "F2_ADAPTER FAIL: %0d check(s)", fails);
    $finish;
  end

  initial begin
    #40ms;
    $fatal(1, "F2_ADAPTER TIMEOUT (phase stuck)");
  end
endmodule
