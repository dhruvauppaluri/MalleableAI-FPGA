// Simulation model of the HBM side of the F2 shell: NPC pseudo-channels, each an
// in-order AXI3-style slave (256-bit data, at most 16 beats per burst, one read
// burst and one write burst in flight per PC, reads and writes independent).
// Every ready is randomly withheld (+hbm_stall=percent, default 20). The first read
// beat and the write response come at least +hbm_lat cycles after the request
// (default 20). +hbm_seed=N reseeds the per-PC xorshift generators.
//
// Addresses at or above 2**PC_AW answer DECERR and touch no memory.
//
// Simulation only: this is not the AMD HBM IP and says nothing about real HBM
// bandwidth, latency, refresh or bank behaviour.
module f2_hbm_model #(
  parameter int NPC   = 4,
  parameter int PC_AW = 21          // bytes per PC = 2**PC_AW (small in simulation)
) (
  input  logic                    clk,
  input  logic                    rst,

  input  logic [NPC*PC_AW-1:0]    awaddr,
  input  logic [NPC*4-1:0]        awlen,
  input  logic [NPC-1:0]          awvalid,
  output logic [NPC-1:0]          awready,
  input  logic [NPC*256-1:0]      wdata,
  input  logic [NPC*32-1:0]       wstrb,
  input  logic [NPC-1:0]          wlast,
  input  logic [NPC-1:0]          wvalid,
  output logic [NPC-1:0]          wready,
  output logic [NPC*2-1:0]        bresp,
  output logic [NPC-1:0]          bvalid,
  input  logic [NPC-1:0]          bready,
  input  logic [NPC*PC_AW-1:0]    araddr,
  input  logic [NPC*4-1:0]        arlen,
  input  logic [NPC-1:0]          arvalid,
  output logic [NPC-1:0]          arready,
  output logic [NPC*256-1:0]      rdata,
  output logic [NPC*2-1:0]        rresp,
  output logic [NPC-1:0]          rlast,
  output logic [NPC-1:0]          rvalid,
  input  logic [NPC-1:0]          rready
);
  localparam int PCB = 1 << PC_AW;
  logic [7:0] mem [NPC * PCB];

  int stall = 20;
  int lat   = 20;
  int seed  = 1;
  longint rd_beats [NPC];
  longint wr_beats [NPC];
  initial begin
    void'($value$plusargs("hbm_stall=%d", stall));
    void'($value$plusargs("hbm_lat=%d", lat));
    void'($value$plusargs("hbm_seed=%d", seed));
    for (int i = 0; i < NPC; i++) begin rd_beats[i] = 0; wr_beats[i] = 0; end
    for (int i = 0; i < NPC * PCB; i++) mem[i] = 8'h00;
  end

  for (genvar p = 0; p < NPC; p++) begin : g_pc
    // per-PC generator: three independent ready decisions per cycle
    logic [31:0] rng;
    logic        ok_ar, ok_aw, ok_w, ok_r;
    always_ff @(posedge clk) begin
      logic [31:0] x;
      if (rst) begin
        rng <= 32'h9E3779B9 ^ 32'(seed * 7919) ^ 32'((p + 1) * 40503);
        if (rng == 0) rng <= 32'd1;
      end else begin
        x = rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; rng <= x;
      end
      ok_ar <= ((rng >> 3)  % 100) >= 32'(stall);
      ok_aw <= ((rng >> 9)  % 100) >= 32'(stall);
      ok_w  <= ((rng >> 15) % 100) >= 32'(stall);
      ok_r  <= ((rng >> 21) % 100) >= 32'(stall);
    end

    // ---- read: AR -> wait -> stream
    logic             rbusy;
    logic [PC_AW-1:0] raddr;
    logic [3:0]       rlen, rcnt;
    int               rwait;
    logic             rbad;
    assign arready[p] = !rbusy && ok_ar;
    always_ff @(posedge clk) begin
      if (rst) begin
        rbusy <= 1'b0;
        rvalid[p] <= 1'b0;
      end else begin
        if (!rbusy) begin
          if (arvalid[p] && arready[p]) begin
            rbusy <= 1'b1; raddr <= araddr[p*PC_AW +: PC_AW]; rlen <= arlen[p*4 +: 4];
            rcnt <= '0; rwait <= lat;
          end
        end else if (!rvalid[p]) begin
          if (rwait > 0) rwait <= rwait - 1;
          else if (ok_r) begin
            rvalid[p] <= 1'b1;
            for (int b = 0; b < 32; b++)
              rdata[p*256 + b*8 +: 8] <= mem[p*PCB + int'(raddr) + int'(rcnt)*32 + b];
            rresp[p*2 +: 2] <= 2'b00;
            rlast[p] <= rcnt == rlen;
            rd_beats[p] <= rd_beats[p] + 1;
          end
        end else if (rready[p]) begin
          rvalid[p] <= 1'b0;
          if (rcnt == rlen) rbusy <= 1'b0;
          else rcnt <= rcnt + 4'd1;
        end
      end
    end

    // ---- write: AW, then W beats, then B after the latency
    logic             awh, bpend;
    logic [PC_AW-1:0] waddr;
    logic [3:0]       wcnt;
    int               bwait;
    assign awready[p] = !awh && !bpend && !bvalid[p] && ok_aw;
    assign wready[p]  = awh && ok_w;
    always_ff @(posedge clk) begin
      if (rst) begin
        awh <= 1'b0; bpend <= 1'b0; bvalid[p] <= 1'b0;
      end else begin
        if (awvalid[p] && awready[p]) begin
          awh <= 1'b1; waddr <= awaddr[p*PC_AW +: PC_AW]; wcnt <= '0;
        end
        if (wvalid[p] && wready[p]) begin
          for (int b = 0; b < 32; b++)
            if (wstrb[p*32 + b])
              mem[p*PCB + int'(waddr) + int'(wcnt)*32 + b] <= wdata[p*256 + b*8 +: 8];
          wr_beats[p] <= wr_beats[p] + 1;
          wcnt <= wcnt + 4'd1;
          if (wlast[p]) begin awh <= 1'b0; bpend <= 1'b1; bwait <= lat; end
        end
        if (bpend) begin
          if (bwait > 0) bwait <= bwait - 1;
          else begin bpend <= 1'b0; bvalid[p] <= 1'b1; bresp[p*2 +: 2] <= 2'b00; end
        end
        if (bvalid[p] && bready[p]) bvalid[p] <= 1'b0;
      end
    end
  end
endmodule
