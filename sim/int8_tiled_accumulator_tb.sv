`timescale 1ns/1ps

module int8_tiled_accumulator_tb;

localparam integer LANES = 4;
localparam integer ACC_WIDTH = 16;
localparam integer MAX_VALUES = 24;

reg clk;
reg reset;
reg start;
reg tile_valid;
wire tile_ready;
reg tile_last;
reg [LANES*8-1:0] activations;
reg [LANES*8-1:0] weights;
wire signed [ACC_WIDTH-1:0] result;
wire busy;
wire done;
wire overflow;

reg signed [7:0] activation_values [0:MAX_VALUES-1];
reg signed [7:0] weight_values [0:MAX_VALUES-1];

integer errors;
integer seed;
integer test_index;
integer value_index;
integer tile_index;
integer lane;
integer length;
integer timeout;
integer exact_tile;
integer running_value;
integer expected_value;
reg expected_overflow;

int8_tiled_accumulator #(
    .ACC_WIDTH(ACC_WIDTH),
    .LANES(LANES)
) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .tile_valid(tile_valid),
    .tile_ready(tile_ready),
    .tile_last(tile_last),
    .activations(activations),
    .weights(weights),
    .result(result),
    .busy(busy),
    .done(done),
    .overflow(overflow)
);

always #5 clk = ~clk;

task run_reduction;
    input integer test_length;
    begin
        running_value = 0;
        expected_overflow = 1'b0;

        for (tile_index = 0; tile_index < test_length; tile_index = tile_index + LANES)
        begin
            exact_tile = running_value;
            for (lane = 0; lane < LANES; lane = lane + 1)
            begin
                if (tile_index + lane < test_length)
                    exact_tile = exact_tile
                               + activation_values[tile_index + lane]
                               * weight_values[tile_index + lane];
            end
            if ((exact_tile > 32767) || (exact_tile < -32768))
                expected_overflow = 1'b1;
            running_value = $signed(exact_tile[15:0]);
        end
        expected_value = running_value;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        for (tile_index = 0; tile_index < test_length; tile_index = tile_index + LANES)
        begin
            while (!tile_ready)
                @(negedge clk);

            activations = {(LANES*8){1'b0}};
            weights     = {(LANES*8){1'b0}};
            for (lane = 0; lane < LANES; lane = lane + 1)
            begin
                if (tile_index + lane < test_length)
                begin
                    activations[lane*8 +: 8] = activation_values[tile_index + lane];
                    weights[lane*8 +: 8] = weight_values[tile_index + lane];
                end
            end
            tile_last  = (tile_index + LANES >= test_length);
            tile_valid = 1'b1;
            @(negedge clk);
            tile_valid = 1'b0;
        end

        timeout = 0;
        while (!done && timeout < 100)
        begin
            @(negedge clk);
            timeout = timeout + 1;
        end

        if (timeout >= 100)
        begin
            $display("ERROR tiled accumulator timeout");
            errors = errors + 1;
        end
        else
        begin
            if (result !== expected_value[15:0])
            begin
                $display(
                    "ERROR tiled result: length=%0d expected=%0d observed=%0d",
                    test_length,
                    expected_value,
                    result
                );
                errors = errors + 1;
            end
            if (overflow !== expected_overflow)
            begin
                $display(
                    "ERROR tiled overflow: length=%0d expected=%0b observed=%0b",
                    test_length,
                    expected_overflow,
                    overflow
                );
                errors = errors + 1;
            end
        end
    end
endtask

initial
begin
    clk         = 1'b0;
    reset       = 1'b1;
    start       = 1'b0;
    tile_valid  = 1'b0;
    tile_last   = 1'b0;
    activations = {(LANES*8){1'b0}};
    weights     = {(LANES*8){1'b0}};
    errors      = 0;
    seed        = 32'h54494c45;

    repeat (2) @(posedge clk);
    reset = 1'b0;

    for (test_index = 0; test_index < 1000; test_index = test_index + 1)
    begin
        length = (($random(seed) & 32'h7fffffff) % MAX_VALUES) + 1;
        for (value_index = 0; value_index < length; value_index = value_index + 1)
        begin
            activation_values[value_index] = $random(seed);
            weight_values[value_index] = $random(seed);
        end
        run_reduction(length);
    end

    if (errors == 0)
        $display("PASS: tiled accumulation verified with 1000 randomized reductions");
    else
        $display("FAIL: tiled accumulator errors=%0d", errors);

    $finish;
end

endmodule
