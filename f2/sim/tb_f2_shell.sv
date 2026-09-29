// Simulation testbench for cl_otpu_core (the F2 custom-logic core around the frozen
// OpenTPU board block), driven exactly like the upstream board testbench (tb_board.sv):
//
//   +dir=<d>   <d>/host.txt   the host script (ops below)
//              <d>/ch0.bin, ch1.bin   the two channel images, big-endian 32-bit words
//   output     <d>/ch0_out.bin, ch1_out.bin   the channel memories at the end (raw bytes)
//              <d>/tmem_0.hex                  the slice's TMEM shadow (one hex word per line)
//   stdout     "REG <addr> <value>" per R op, "DONE cycles=<core cycles>"
//
// Script ops (numbers in hex), OCL = the shell's AXI-Lite port, PCIS = the 512-bit AXI4 port:
//   W <addr> <value>          OCL write            (board registers 0x000.., F2 block 0x1000..)
//   P <addr> <mask> <value>   poll an OCL register until (read & mask) == value
//   R <addr>                  OCL read, printed as "REG <addr> <value>"
//   C <cycles>                wait core cycles
//   DW <off> <bytes> <seed>   PCIS write of a pseudo-random pattern; <off> is the offset in the HBM
//                             window (bit 31 = core channel), 64-byte aligned
//   DR <off> <bytes> <seed>   PCIS read and check of that pattern: "DMA_OK" or "DMA_FAIL"
//   DT <off> <bytes>          timed PCIS read: "DMA_TIME bytes=<n> main_cycles=<n>"
//
// The channel images are placed in the HBM model either directly (backdoor, default; fast) or
// through the real PCIS path (+dma_load); the final memories are read out the same way
// (+dma_dump). Simulation of the wrapper and adapter only: nothing here is a physical
// measurement, and the HBM model is not the AMD HBM IP.
`timescale 1ns/1ps
module tb_f2_shell;
  parameter int CH_BYTES   = 1 << 21;      // bytes per core channel
  parameter int PCS        = 2;            // HBM PCs per core channel
  parameter int MCOLS      = 2;
  parameter int LANES      = 8;
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int ULANES     = LANES;
  parameter int WIN        = 16;
  parameter int FIFO_DEPTH = 1024;
  parameter int RPB        = 8 * LANES;
  parameter int WPB        = 1;
  parameter int MXU_IMPL   = 0;
  parameter int IMEM_WORDS = 1 << 15;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int ACT_BLOCKS = 128;
  parameter int CORE_KHZ   = 125000;
  parameter logic [31:0] BUILD_ID = 32'h0F2_0001;
  parameter int TRACE_DEPTH = 16384;
  parameter int HBM_INIT   = 200;          // main-clock cycles until HBM reports ready

  localparam int PC_AW = $clog2(CH_BYTES / PCS);
  localparam int NPCT  = 2 * PCS;

  // ------------------------------------------------------------ clocks, reset
  real core_half = 4.0, main_half = 2.0, hbm_half = 1.1;     // ns; 125 / 250 / ~454 MHz
  logic clk_core = 0, clk_main = 0, clk_hbm = 0;
  initial begin
    real p;
    if ($value$plusargs("core_ns=%f", p)) core_half = p / 2.0;
    if ($value$plusargs("main_ns=%f", p)) main_half = p / 2.0;
    if ($value$plusargs("hbm_ns=%f",  p)) hbm_half  = p / 2.0;
  end
  initial forever #(core_half) clk_core = ~clk_core;
  initial forever #(main_half) clk_main = ~clk_main;
  initial forever #(hbm_half)  clk_hbm  = ~clk_hbm;

  logic rst_n = 0;
  logic hbm_ready = 0;

  // ------------------------------------------------------------ DUT and HBM model
  logic [31:0] ocl_awaddr = 0, ocl_wdata = 0, ocl_araddr = 0;
  logic [3:0]  ocl_wstrb = 4'hF;
  logic        ocl_awvalid = 0, ocl_wvalid = 0, ocl_bready = 1, ocl_arvalid = 0, ocl_rready = 1;
  wire         ocl_awready, ocl_wready, ocl_bvalid, ocl_arready, ocl_rvalid;
  wire  [1:0]  ocl_bresp, ocl_rresp;
  wire  [31:0] ocl_rdata;

  logic [15:0] p_awid = 0, p_arid = 0;
  logic [63:0] p_awaddr = 0, p_araddr = 0;
  logic [7:0]  p_awlen = 0, p_arlen = 0;
  logic        p_awvalid = 0, p_wlast = 0, p_wvalid = 0, p_bready = 0, p_arvalid = 0, p_rready = 0;
  logic [511:0] p_wdata = 0;
  logic [63:0]  p_wstrb = '1;
  wire         p_awready, p_wready, p_bvalid, p_arready, p_rlast, p_rvalid;
  wire  [15:0] p_bid, p_rid;
  wire  [1:0]  p_bresp, p_rresp;
  wire  [511:0] p_rdata;

  wire [NPCT*PC_AW-1:0]  h_awaddr, h_araddr;
  wire [NPCT*4-1:0]      h_awlen, h_arlen;
  wire [NPCT-1:0]        h_awvalid, h_awready, h_wlast, h_wvalid, h_wready, h_bvalid, h_bready;
  wire [NPCT-1:0]        h_arvalid, h_arready, h_rlast, h_rvalid, h_rready;
  wire [NPCT*256-1:0]    h_wdata, h_rdata;
  wire [NPCT*32-1:0]     h_wstrb;
  wire [NPCT*2-1:0]      h_bresp, h_rresp;
  wire [2:0] led;

  cl_otpu_core #(.PCS_PER_CH(PCS), .PC_AW(PC_AW), .MCOLS(MCOLS), .LANES(LANES), .VPU_CL(VPU_CL),
                 .ULANES(ULANES), .WIN(WIN), .FIFO_DEPTH(FIFO_DEPTH), .RPB(RPB), .WPB(WPB),
                 .MXU_IMPL(MXU_IMPL), .IMEM_WORDS(IMEM_WORDS), .TMEM_WORDS(TMEM_WORDS), .ACT_BLOCKS(ACT_BLOCKS),
                 .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .TRACE_DEPTH(TRACE_DEPTH)) dut (
    .clk_main, .rst_main_n(rst_n), .clk_core, .clk_hbm, .hbm_ready,
    .ocl_awaddr, .ocl_awvalid, .ocl_awready, .ocl_wdata, .ocl_wstrb, .ocl_wvalid, .ocl_wready,
    .ocl_bresp, .ocl_bvalid, .ocl_bready, .ocl_araddr, .ocl_arvalid, .ocl_arready,
    .ocl_rdata, .ocl_rresp, .ocl_rvalid, .ocl_rready,
    .pcis_awid(p_awid), .pcis_awaddr(p_awaddr), .pcis_awlen(p_awlen), .pcis_awvalid(p_awvalid),
    .pcis_awready(p_awready), .pcis_wdata(p_wdata), .pcis_wstrb(p_wstrb), .pcis_wlast(p_wlast),
    .pcis_wvalid(p_wvalid), .pcis_wready(p_wready), .pcis_bid(p_bid), .pcis_bresp(p_bresp),
    .pcis_bvalid(p_bvalid), .pcis_bready(p_bready), .pcis_arid(p_arid), .pcis_araddr(p_araddr),
    .pcis_arlen(p_arlen), .pcis_arvalid(p_arvalid), .pcis_arready(p_arready), .pcis_rid(p_rid),
    .pcis_rdata(p_rdata), .pcis_rresp(p_rresp), .pcis_rlast(p_rlast), .pcis_rvalid(p_rvalid),
    .pcis_rready(p_rready),
    .hbm_awaddr(h_awaddr), .hbm_awlen(h_awlen), .hbm_awvalid(h_awvalid), .hbm_awready(h_awready),
    .hbm_wdata(h_wdata), .hbm_wstrb(h_wstrb), .hbm_wlast(h_wlast), .hbm_wvalid(h_wvalid),
    .hbm_wready(h_wready), .hbm_bresp(h_bresp), .hbm_bvalid(h_bvalid), .hbm_bready(h_bready),
    .hbm_araddr(h_araddr), .hbm_arlen(h_arlen), .hbm_arvalid(h_arvalid), .hbm_arready(h_arready),
    .hbm_rdata(h_rdata), .hbm_rresp(h_rresp), .hbm_rlast(h_rlast), .hbm_rvalid(h_rvalid),
    .hbm_rready(h_rready), .led);

  wire hbm_rst = !rst_n;
  f2_hbm_model #(.NPC(NPCT), .PC_AW(PC_AW)) hbm (
    .clk(clk_hbm), .rst(hbm_rst),
    .awaddr(h_awaddr), .awlen(h_awlen), .awvalid(h_awvalid), .awready(h_awready),
    .wdata(h_wdata), .wstrb(h_wstrb), .wlast(h_wlast), .wvalid(h_wvalid), .wready(h_wready),
    .bresp(h_bresp), .bvalid(h_bvalid), .bready(h_bready),
    .araddr(h_araddr), .arlen(h_arlen), .arvalid(h_arvalid), .arready(h_arready),
    .rdata(h_rdata), .rresp(h_rresp), .rlast(h_rlast), .rvalid(h_rvalid), .rready(h_rready));

  // free-running random readiness on the PCIS response channels
  always @(posedge clk_main) begin
    p_bready <= ($urandom % 4) != 0;
    p_rready <= ($urandom % 4) != 0;
  end

  longint core_cyc = 0, main_cyc = 0;
  always @(posedge clk_core) core_cyc <= core_cyc + 1;
  always @(posedge clk_main) main_cyc <= main_cyc + 1;

  // ------------------------------------------------------------ placement (must match placement.py)
  function automatic int place_pc(input int ch, input longint off);
    return ch * PCS + int'((off >> 9) % PCS);
  endfunction
  function automatic longint place_local(input longint off);
    return ((off >> 9) / PCS) * 512 + (off & 511);
  endfunction

  logic [7:0] img [2 * CH_BYTES];
  logic [7:0] out [2 * CH_BYTES];

  task automatic bd_put(input int ch, input longint off, input logic [7:0] b);
    hbm.mem[place_pc(ch, off) * (1 << PC_AW) + int'(place_local(off))] = b;
  endtask
  function automatic logic [7:0] bd_get(input int ch, input longint off);
    return hbm.mem[place_pc(ch, off) * (1 << PC_AW) + int'(place_local(off))];
  endfunction

  // ------------------------------------------------------------ OCL BFM
  task automatic ocl_write(input logic [31:0] a, input logic [31:0] v, output logic [1:0] resp);
    bit aw_d = 0, w_d = 0;
    ocl_awaddr <= a; ocl_wdata <= v; ocl_awvalid <= 1; ocl_wvalid <= 1;
    while (!(aw_d && w_d)) begin
      @(posedge clk_main);
      if (!aw_d && ocl_awready) begin aw_d = 1; ocl_awvalid <= 0; end
      if (!w_d && ocl_wready) begin w_d = 1; ocl_wvalid <= 0; end
    end
    do @(posedge clk_main); while (!ocl_bvalid);
    resp = ocl_bresp;
  endtask
  task automatic ocl_read(input logic [31:0] a, output logic [31:0] v, output logic [1:0] resp);
    ocl_araddr <= a; ocl_arvalid <= 1;
    do @(posedge clk_main); while (!ocl_arready);
    ocl_arvalid <= 0;
    do @(posedge clk_main); while (!ocl_rvalid);
    v = ocl_rdata; resp = ocl_rresp;
  endtask

  // ------------------------------------------------------------ PCIS BFM
  // data source: 0 = channel image `img`, 1 = pseudo-random pattern (seed)
  function automatic logic [7:0] dpat(input longint i, input int seed);
    logic [31:0] x;
    x = 32'(i) * 32'h9E3779B1 ^ 32'(seed) * 32'h85EBCA6B;
    x ^= x >> 15; x *= 32'h2C1B3C6D; x ^= x >> 12;
    return x[7:0];
  endfunction

  function automatic logic [63:0] haddr(input int ch, input longint off);
    return (64'd1 << 36) | (ch ? 64'h8000_0000 : 64'd0) | 64'(off);
  endfunction

  int dma_fails = 0;
  task automatic pcis_wr_burst(input int ch, input longint off, input int nbeats,
                               input int kind, input int seed);
    bit aw_d = 0;
    int sent = 0;
    p_awaddr <= haddr(ch, off); p_awlen <= 8'(nbeats - 1); p_awvalid <= 1; p_awid <= 16'h0F2A;
    while (!aw_d || sent < nbeats) begin
      if (sent < nbeats && !p_wvalid) begin
        for (int k = 0; k < 64; k++)
          p_wdata[k*8 +: 8] <= kind == 0 ? img[ch * CH_BYTES + int'(off) + sent * 64 + k]
                                         : dpat(longint'(ch) * CH_BYTES + off + sent * 64 + k, seed);
        p_wstrb <= '1; p_wlast <= sent == nbeats - 1; p_wvalid <= 1;
      end
      @(posedge clk_main);
      if (!aw_d && p_awready) begin aw_d = 1; p_awvalid <= 0; end
      if (p_wvalid && p_wready) begin sent++; p_wvalid <= 0; end
    end
    do @(posedge clk_main); while (!(p_bvalid && p_bready));
    if (p_bresp != 2'b00) begin dma_fails++; $display("DMA write ch%0d off %0d: bresp %0d", ch, off, p_bresp); end
  endtask

  // kind 0: store into `out`; kind 1: compare with the seed pattern; kind 2: discard
  task automatic pcis_rd_burst(input int ch, input longint off, input int nbeats,
                               input int kind, input int seed);
    int got = 0;
    p_araddr <= haddr(ch, off); p_arlen <= 8'(nbeats - 1); p_arvalid <= 1; p_arid <= 16'h0F2B;
    do @(posedge clk_main); while (!p_arready);
    p_arvalid <= 0;
    while (got < nbeats) begin
      @(posedge clk_main);
      if (p_rvalid && p_rready) begin
        if (p_rresp != 2'b00) begin dma_fails++; $display("DMA read ch%0d off %0d: rresp %0d", ch, off, p_rresp); end
        for (int k = 0; k < 64; k++) begin
          if (kind == 0) out[ch * CH_BYTES + int'(off) + got * 64 + k] = p_rdata[k*8 +: 8];
          else if (kind == 1 && p_rdata[k*8 +: 8] !== dpat(longint'(ch) * CH_BYTES + off + got * 64 + k, seed)) dma_fails++;
        end
        got++;
      end
    end
  endtask

  // whole region [off, off+n) of one channel, in bursts that stay inside 4 KiB pages
  task automatic dma_region(input bit wr, input int ch, input longint off, input longint n,
                            input int kind, input int seed);
    longint a = off;
    while (a < off + n) begin
      automatic int room = 64 - int'((a % 4096) / 64);
      automatic longint left = (off + n - a) / 64;
      automatic int nb = room < 64 ? room : 64;
      if (longint'(nb) > left) nb = int'(left);
      if (wr) pcis_wr_burst(ch, a, nb, kind, seed);
      else pcis_rd_burst(ch, a, nb, kind, seed);
      a += longint'(nb) * 64;
    end
  endtask

  // ------------------------------------------------------------ main sequence
  string dir, line, op;
  integer fd, n;
  longint max_cycles = 64'd1 << 40;
  bit dma_load = 0, dma_dump = 0;
  logic [31:0] words [CH_BYTES / 4];

  task automatic load_channel(input int ch);
    string fn;
    integer f, got;
    fn = $sformatf("%s/ch%0d.bin", dir, ch);
    f = $fopen(fn, "rb");
    if (f == 0) $fatal(1, "cannot open %s", fn);
    got = $fread(words, f);
    $fclose(f);
    if (got != CH_BYTES) $fatal(1, "%s: %0d bytes, expected %0d", fn, got, CH_BYTES);
    for (int i = 0; i < CH_BYTES / 4; i++)
      for (int k = 0; k < 4; k++) img[ch * CH_BYTES + i * 4 + k] = words[i][k*8 +: 8];
    if (!dma_load) begin
      for (longint o = 0; o < CH_BYTES; o++) bd_put(ch, o, img[ch * CH_BYTES + int'(o)]);
    end
  endtask

  task automatic dump_channel(input int ch);
    integer f;
    if (dma_dump) dma_region(0, ch, 0, CH_BYTES, 0, 0);
    else for (longint o = 0; o < CH_BYTES; o++) out[ch * CH_BYTES + int'(o)] = bd_get(ch, o);
    f = $fopen($sformatf("%s/ch%0d_out.bin", dir, ch), "wb");
    for (int i = 0; i < CH_BYTES; i++) $fwrite(f, "%c", out[ch * CH_BYTES + i]);
    $fclose(f);
  endtask

  initial begin
    logic [31:0] a, m, v, r;
    logic [1:0] resp;
    if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= missing");
    void'($value$plusargs("max_cycles=%d", max_cycles));
    dma_load = $test$plusargs("dma_load");
    dma_dump = $test$plusargs("dma_dump");

    repeat (8) @(posedge clk_main);
    rst_n = 1;
    // HBM initialization delay, then the host sees ready
    repeat (HBM_INIT) @(posedge clk_main);
    hbm_ready = 1;
    repeat (10) @(posedge clk_main);

    load_channel(0);
    load_channel(1);
    if (dma_load) begin
      dma_region(1, 0, 0, CH_BYTES, 0, 0);
      dma_region(1, 1, 0, CH_BYTES, 0, 0);
    end

    fd = $fopen($sformatf("%s/host.txt", dir), "r");
    if (fd == 0) $fatal(1, "no host script");
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%s", op);
      if (n != 1) break;
      case (op)
        "W": begin
          n = $fscanf(fd, "%h %h", a, v);
          ocl_write(a, v, resp);
          if (resp != 2'b00) $display("OCL write %h: resp %0d", a, resp);
        end
        "P": begin
          n = $fscanf(fd, "%h %h %h", a, m, v);
          do begin
            ocl_read(a, r, resp);
            if (core_cyc > max_cycles) begin
              $display("TIMEOUT polling %h (%h)", a, r);
              $finish;
            end
          end while ((r & m) != v);
        end
        "R": begin
          n = $fscanf(fd, "%h", a);
          ocl_read(a, r, resp);
          $display("REG %h %h", a, r);
        end
        "C": begin
          n = $fscanf(fd, "%h", v);
          repeat (v) @(posedge clk_core);
        end
        "DW", "DR", "DT": begin
          longint doff, dbytes;
          int dseed, dch;
          longint t0;
          n = $fscanf(fd, "%h %h", doff, dbytes);
          dseed = 0;
          if (op != "DT") n = $fscanf(fd, "%h", dseed);
          dch = int'((doff >> 31) & 1);
          doff = doff & 64'h7FFF_FFFF;
          t0 = main_cyc;
          dma_fails = 0;
          if (op == "DW") dma_region(1, dch, doff, dbytes, 1, dseed);
          else if (op == "DR") dma_region(0, dch, doff, dbytes, 1, dseed);
          else dma_region(0, dch, doff, dbytes, 2, 0);
          if (op == "DT") $display("DMA_TIME bytes=%0d main_cycles=%0d", dbytes, main_cyc - t0);
          else if (dma_fails == 0) $display("DMA_OK %s bytes=%0d", op, dbytes);
          else $display("DMA_FAIL %s bytes=%0d mismatches=%0d", op, dbytes, dma_fails);
        end
        default: $fatal(1, "bad host op %s", op);
      endcase
    end
    $fclose(fd);
    repeat (20) @(posedge clk_main);

    dump_channel(0);
    dump_channel(1);
    begin
      integer f;
      f = $fopen($sformatf("%s/tmem_0.hex", dir), "w");
      for (int i = 0; i < TMEM_WORDS; i++) $fwrite(f, "%08x\n", dut.u_board.u_slice.u_tmem.shadow[i]);
      $fclose(f);
    end
    $display("F2_STATS pc_rd_beats=%0d,%0d,%0d,%0d pc_wr_beats=%0d,%0d,%0d,%0d",
             hbm.rd_beats[0], hbm.rd_beats[1], hbm.rd_beats[NPCT-1 > 1 ? 2 : 0], hbm.rd_beats[NPCT-1],
             hbm.wr_beats[0], hbm.wr_beats[1], hbm.wr_beats[NPCT-1 > 1 ? 2 : 0], hbm.wr_beats[NPCT-1]);
    $display("DONE cycles=%0d", core_cyc);
    $finish;
  end
endmodule
