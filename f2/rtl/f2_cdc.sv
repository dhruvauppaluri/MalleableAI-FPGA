// F2 platform: small clock-domain-crossing helpers for the CL wrapper.
// UNTESTED on hardware; not run through Vivado.

// Reset synchronizer: asserts asynchronously, releases synchronously (active-high output).
module f2_rst_sync (
  input  logic clk,
  input  logic arst_n,        // asynchronous, active low
  output logic rst            // synchronous release, active high
);
  (* ASYNC_REG = "TRUE" *) logic [2:0] q;
  always_ff @(posedge clk or negedge arst_n) begin
    if (!arst_n) q <= 3'b111;
    else         q <= {q[1:0], 1'b0};
  end
  assign rst = q[2];
endmodule

// Level synchronizer (two flops) for quasi-static or slowly changing single bits.
module f2_bit_sync (
  input  logic clk,
  input  logic d,
  output logic q
);
  (* ASYNC_REG = "TRUE" *) logic s1, s2;
  always_ff @(posedge clk) begin
    s1 <= d;
    s2 <= s1;
  end
  assign q = s2;
endmodule

// Pulse crossing by toggle synchronization. Pulses closer together than about
// three destination clocks can merge; the counters built on it are diagnostic.
module f2_pulse_sync (
  input  logic sclk,
  input  logic srst,
  input  logic pulse,
  input  logic dclk,
  output logic dpulse
);
  logic tog;
  always_ff @(posedge sclk) begin
    if (srst) tog <= 1'b0;
    else if (pulse) tog <= ~tog;
  end
  (* ASYNC_REG = "TRUE" *) logic d1, d2;
  logic d3;
  always_ff @(posedge dclk) begin
    d1 <= tog;
    d2 <= d1;
    d3 <= d2;
  end
  assign dpulse = d2 ^ d3;
endmodule
