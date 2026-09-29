// F2 platform: small FIFOs used by the HBM adapter.
//
// Plain SystemVerilog without interfaces or unpacked-array ports, so the same
// files go through Verilator (lint, simulation) and Yosys (structural check).
// UNTESTED on hardware and not run through Vivado (no Vivado in the cloud
// environment).

// Synchronous FIFO, first-word-fall-through, 2**AW entries.
module f2_sync_fifo #(
  parameter int W  = 8,
  parameter int AW = 4
) (
  input  logic          clk,
  input  logic          rst,             // synchronous, active high
  input  logic          wvalid,
  output logic          wready,
  input  logic [W-1:0]  wdata,
  output logic          rvalid,
  input  logic          rready,
  output logic [W-1:0]  rdata
);
  localparam int D = 1 << AW;
  logic [W-1:0] mem [D];
  logic [AW:0]  wp, rp;
  assign wready = (wp - rp) != (AW+1)'(D);
  assign rvalid = wp != rp;
  assign rdata  = mem[rp[AW-1:0]];
  always_ff @(posedge clk) begin
    if (wvalid && wready) mem[wp[AW-1:0]] <= wdata;
    if (rst) begin
      wp <= '0;
      rp <= '0;
    end else begin
      if (wvalid && wready) wp <= wp + 1'b1;
      if (rvalid && rready) rp <= rp + 1'b1;
    end
  end
endmodule

// Asynchronous FIFO (gray-coded pointers, two-flop synchronizers), 2**AW entries.
// Both resets must be asserted together for several cycles of both clocks at
// power-up (the CL derives them from one synchronized reset); afterwards the
// domains are independent.
module f2_async_fifo #(
  parameter int W  = 8,
  parameter int AW = 4
) (
  input  logic          wclk,
  input  logic          wrst,
  input  logic          wvalid,
  output logic          wready,
  input  logic [W-1:0]  wdata,
  input  logic          rclk,
  input  logic          rrst,
  output logic          rvalid,
  input  logic          rready,
  output logic [W-1:0]  rdata
);
  localparam int D = 1 << AW;
  logic [W-1:0] mem [D];

  logic [AW:0] wbin, wgray, rbin, rgray;
  (* ASYNC_REG = "TRUE" *) logic [AW:0] rgray_w1, rgray_w2;   // read pointer in the write domain
  (* ASYNC_REG = "TRUE" *) logic [AW:0] wgray_r1, wgray_r2;   // write pointer in the read domain

  wire wfull  = wgray == {~rgray_w2[AW:AW-1], rgray_w2[AW-2:0]};
  wire rempty = rgray == wgray_r2;
  assign wready = !wfull;
  assign rvalid = !rempty;
  assign rdata  = mem[rbin[AW-1:0]];

  wire          wen   = wvalid && wready;
  wire [AW:0]   wbinn = wbin + 1'b1;
  wire          ren   = rvalid && rready;
  wire [AW:0]   rbinn = rbin + 1'b1;

  always_ff @(posedge wclk) begin
    if (wen) mem[wbin[AW-1:0]] <= wdata;
    if (wrst) begin
      wbin <= '0;
      wgray <= '0;
    end else if (wen) begin
      wbin <= wbinn;
      wgray <= wbinn ^ (wbinn >> 1);
    end
    rgray_w1 <= rgray;
    rgray_w2 <= rgray_w1;
  end

  always_ff @(posedge rclk) begin
    if (rrst) begin
      rbin <= '0;
      rgray <= '0;
    end else if (ren) begin
      rbin <= rbinn;
      rgray <= rbinn ^ (rbinn >> 1);
    end
    wgray_r1 <= wgray;
    wgray_r2 <= wgray_r1;
  end
endmodule
