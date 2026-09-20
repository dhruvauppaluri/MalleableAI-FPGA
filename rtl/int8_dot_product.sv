`timescale 1ns/1ps

// Parameterized parallel signed dot-product building block.
//
// Packed lane 0 occupies the least-significant INPUT_WIDTH bits. Each accepted
// input computes:
//   result = accumulator_in + sum(activations[lane] * weights[lane])
//
// The result is registered and valid_out is asserted for one clock when the
// corresponding valid_in sample is accepted.
module int8_dot_product #(
    parameter integer INPUT_WIDTH = 8,
    parameter integer ACC_WIDTH   = 32,
    parameter integer LANES       = 4
) (
    input  wire                               clk,
    input  wire                               reset,
    input  wire                               valid_in,
    input  wire [LANES*INPUT_WIDTH-1:0]       activations,
    input  wire [LANES*INPUT_WIDTH-1:0]       weights,
    input  wire signed [ACC_WIDTH-1:0]        accumulator_in,
    output reg  signed [ACC_WIDTH-1:0]        result,
    output reg                                valid_out,
    output reg                                overflow
);

localparam integer PRODUCT_WIDTH = 2 * INPUT_WIDTH;
localparam integer SUM_WIDTH = ACC_WIDTH + $clog2(LANES + 1) + 1;

integer lane;
reg signed [INPUT_WIDTH-1:0] activation_lane;
reg signed [INPUT_WIDTH-1:0] weight_lane;
reg signed [PRODUCT_WIDTH-1:0] product_lane;
reg signed [SUM_WIDTH-1:0] sum_comb;
reg signed [SUM_WIDTH-1:0] result_sign_extended;
reg overflow_comb;

always @(*)
begin
    sum_comb = {
        {(SUM_WIDTH - ACC_WIDTH){accumulator_in[ACC_WIDTH-1]}},
        accumulator_in
    };

    activation_lane = {INPUT_WIDTH{1'b0}};
    weight_lane     = {INPUT_WIDTH{1'b0}};
    product_lane    = {PRODUCT_WIDTH{1'b0}};

    for (lane = 0; lane < LANES; lane = lane + 1)
    begin
        activation_lane = activations[lane*INPUT_WIDTH +: INPUT_WIDTH];
        weight_lane     = weights[lane*INPUT_WIDTH +: INPUT_WIDTH];
        product_lane    = activation_lane * weight_lane;
        sum_comb = sum_comb + {
            {(SUM_WIDTH - PRODUCT_WIDTH){product_lane[PRODUCT_WIDTH-1]}},
            product_lane
        };
    end

    result_sign_extended = {
        {(SUM_WIDTH - ACC_WIDTH){sum_comb[ACC_WIDTH-1]}},
        sum_comb[ACC_WIDTH-1:0]
    };
    overflow_comb = sum_comb != result_sign_extended;
end

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
            result   <= sum_comb[ACC_WIDTH-1:0];
            overflow <= overflow_comb;
        end
    end
end

endmodule
