`timescale 1ns/1ps

// Accumulates an arbitrary number of LANES-wide dot-product tiles.
//
// start begins a new reduction. Tiles are accepted when tile_valid and
// tile_ready are both high. tile_last marks the final accepted tile. The
// result follows the numeric contract of int8_dot_product: arithmetic wraps to
// signed INT32 and overflow is reported explicitly.
module int8_tiled_accumulator #(
    parameter integer INPUT_WIDTH = 8,
    parameter integer ACC_WIDTH   = 32,
    parameter integer LANES       = 4
) (
    input  wire                               clk,
    input  wire                               reset,
    input  wire                               start,
    input  wire                               tile_valid,
    output wire                               tile_ready,
    input  wire                               tile_last,
    input  wire [LANES*INPUT_WIDTH-1:0]       activations,
    input  wire [LANES*INPUT_WIDTH-1:0]       weights,
    output reg  signed [ACC_WIDTH-1:0]        result,
    output reg                                busy,
    output reg                                done,
    output reg                                overflow
);

reg signed [ACC_WIDTH-1:0] accumulator;
reg                        waiting_for_result;
reg                        pending_last;

wire signed [ACC_WIDTH-1:0] dot_result;
wire                        dot_valid_out;
wire                        dot_overflow;
wire                        accept_tile;

assign tile_ready = busy && !waiting_for_result;
assign accept_tile = tile_valid && tile_ready;

int8_dot_product #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .LANES(LANES)
) dot_product (
    .clk(clk),
    .reset(reset),
    .valid_in(accept_tile),
    .activations(activations),
    .weights(weights),
    .accumulator_in(accumulator),
    .result(dot_result),
    .valid_out(dot_valid_out),
    .overflow(dot_overflow)
);

always @(posedge clk)
begin
    if (reset)
    begin
        accumulator       <= {ACC_WIDTH{1'b0}};
        result            <= {ACC_WIDTH{1'b0}};
        busy              <= 1'b0;
        done              <= 1'b0;
        overflow          <= 1'b0;
        waiting_for_result <= 1'b0;
        pending_last      <= 1'b0;
    end
    else
    begin
        done <= 1'b0;

        if (start && !busy)
        begin
            accumulator        <= {ACC_WIDTH{1'b0}};
            busy               <= 1'b1;
            overflow           <= 1'b0;
            waiting_for_result <= 1'b0;
            pending_last       <= 1'b0;
        end

        if (accept_tile)
        begin
            waiting_for_result <= 1'b1;
            pending_last       <= tile_last;
        end

        if (dot_valid_out && waiting_for_result)
        begin
            accumulator        <= dot_result;
            overflow           <= overflow || dot_overflow;
            waiting_for_result <= 1'b0;

            if (pending_last)
            begin
                result <= dot_result;
                busy   <= 1'b0;
                done   <= 1'b1;
            end
        end
    end
end

endmodule
