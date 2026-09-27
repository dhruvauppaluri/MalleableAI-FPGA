`timescale 1ns/1ps

// Dense-layer output processing:
//   1. Add signed INT32 bias using a 33-bit intermediate.
//   2. Multiply by a positive per-output integer multiplier.
//   3. Divide by 2^shift, rounding to nearest with ties away from zero.
//   4. Saturate to signed INT8.
//   5. Optionally apply ReLU.
module int8_postprocess (
    input  wire signed [31:0] accumulator,
    input  wire signed [31:0] bias,
    input  wire        [30:0] multiplier,
    input  wire        [5:0]  shift,
    input  wire               relu_enable,
    output reg  signed [7:0]  result
);

reg signed [32:0] biased;
reg signed [64:0] product;
reg        [64:0] magnitude;
reg        [64:0] rounded_magnitude;
reg signed [65:0] rounded_signed;
reg        [64:0] rounding_offset;

always @(*)
begin
    biased = {accumulator[31], accumulator} + {bias[31], bias};
    product = biased * $signed({1'b0, multiplier});

    if (product < 0)
        magnitude = -product;
    else
        magnitude = product;

    if (shift == 0)
    begin
        rounding_offset  = 65'd0;
        rounded_magnitude = magnitude;
    end
    else
    begin
        rounding_offset  = 65'd1 << (shift - 1'b1);
        rounded_magnitude = (magnitude + rounding_offset) >> shift;
    end

    if (product < 0)
        rounded_signed = -$signed({1'b0, rounded_magnitude});
    else
        rounded_signed = $signed({1'b0, rounded_magnitude});

    if (relu_enable && (rounded_signed < 0))
        result = 8'sd0;
    else if (rounded_signed > 66'sd127)
        result = 8'sd127;
    else if (rounded_signed < -66'sd128)
        result = -8'sd128;
    else
        result = rounded_signed[7:0];
end

endmodule
