`timescale 1ns/1ps

// One-cycle signed multiply-accumulate building block.
//
// Numeric contract:
//   result = accumulator_in + (multiplicand * multiplier)
//
// The multiplication is exact at 2 * INPUT_WIDTH bits. The addition is
// evaluated with one guard bit so signed overflow of ACC_WIDTH can be reported.
// On overflow, result contains the normal two's-complement wrapped value and
// overflow is asserted. Higher-level blocks should normally prevent overflow by
// choosing a sufficiently wide accumulator.
module int8_mac #(
    parameter integer INPUT_WIDTH = 8,
    parameter integer ACC_WIDTH   = 32
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          valid_in,
    input  wire signed [INPUT_WIDTH-1:0] multiplicand,
    input  wire signed [INPUT_WIDTH-1:0] multiplier,
    input  wire signed [ACC_WIDTH-1:0]   accumulator_in,
    output reg  signed [ACC_WIDTH-1:0]   result,
    output reg                           valid_out,
    output reg                           overflow
);

localparam integer PRODUCT_WIDTH = 2 * INPUT_WIDTH;

wire signed [PRODUCT_WIDTH-1:0] product;
wire signed [ACC_WIDTH:0]       accumulator_extended;
wire signed [ACC_WIDTH:0]       product_extended;
wire signed [ACC_WIDTH:0]       sum_extended;
wire                            overflow_comb;

assign product = multiplicand * multiplier;
assign accumulator_extended = {accumulator_in[ACC_WIDTH-1], accumulator_in};
assign product_extended = {
    {(ACC_WIDTH + 1 - PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}},
    product
};
assign sum_extended = accumulator_extended + product_extended;
assign overflow_comb = sum_extended[ACC_WIDTH] != sum_extended[ACC_WIDTH-1];

always @(posedge clk)
begin
    if (reset)
    begin
        result    <= {ACC_WIDTH{1'b0}};
        valid_out <= 1'b0;
        overflow  <= 1'b0;
    end
    else
    begin
        valid_out <= valid_in;
        overflow  <= 1'b0;

        if (valid_in)
        begin
            result   <= sum_extended[ACC_WIDTH-1:0];
            overflow <= overflow_comb;
        end
    end
end

endmodule
