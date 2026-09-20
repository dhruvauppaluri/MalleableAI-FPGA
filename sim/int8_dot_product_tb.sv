`timescale 1ns/1ps

module int8_dot_product_tb;

localparam integer LANES = 4;

reg clk;
reg reset;
reg valid_in;
reg [LANES*8-1:0] activations;
reg [LANES*8-1:0] weights;
reg signed [31:0] accumulator_in;

wire signed [31:0] result;
wire valid_out;
wire overflow;

integer errors;
integer seed;
integer test_index;
integer lane;
reg [LANES*8-1:0] random_activations;
reg [LANES*8-1:0] random_weights;
reg signed [31:0] random_accumulator;

int8_dot_product #(
    .LANES(LANES)
) dut (
    .clk(clk),
    .reset(reset),
    .valid_in(valid_in),
    .activations(activations),
    .weights(weights),
    .accumulator_in(accumulator_in),
    .result(result),
    .valid_out(valid_out),
    .overflow(overflow)
);

always #5 clk = ~clk;

task check_dot_product;
    input [LANES*8-1:0] test_activations;
    input [LANES*8-1:0] test_weights;
    input signed [31:0] test_accumulator;
    integer check_lane;
    reg signed [7:0] activation_value;
    reg signed [7:0] weight_value;
    reg signed [63:0] exact_result;
    reg signed [31:0] expected_result;
    reg expected_overflow;
    begin
        exact_result = test_accumulator;
        for (check_lane = 0; check_lane < LANES; check_lane = check_lane + 1)
        begin
            activation_value = test_activations[check_lane*8 +: 8];
            weight_value = test_weights[check_lane*8 +: 8];
            exact_result = exact_result + (activation_value * weight_value);
        end

        expected_result = exact_result[31:0];
        expected_overflow = (exact_result > 64'sh000000007fffffff) ||
                            (exact_result < -64'sh0000000080000000);

        @(negedge clk);
        activations   = test_activations;
        weights       = test_weights;
        accumulator_in = test_accumulator;
        valid_in      = 1'b1;

        @(posedge clk);
        #1;

        if (valid_out !== 1'b1 || result !== expected_result ||
            overflow !== expected_overflow)
        begin
            $display(
                "ERROR: dot product acc=%0d expected=%0d observed=%0d overflow=%0b/%0b",
                test_accumulator,
                expected_result,
                result,
                expected_overflow,
                overflow
            );
            errors = errors + 1;
        end
    end
endtask

initial
begin
    clk               = 1'b0;
    reset             = 1'b1;
    valid_in          = 1'b0;
    activations       = {(LANES*8){1'b0}};
    weights           = {(LANES*8){1'b0}};
    accumulator_in    = 32'sd0;
    errors            = 0;
    seed              = 32'h444f5450;

    repeat (2) @(posedge clk);
    reset = 1'b0;

    // Lane 0 is the least-significant byte.
    check_dot_product(
        {8'sd4, 8'sd3, 8'sd2, 8'sd1},
        {8'sd8, 8'sd7, 8'sd6, 8'sd5},
        32'sd0
    );
    check_dot_product(
        {-8'sd4, 8'sd3, -8'sd2, 8'sd1},
        {8'sd8, -8'sd7, 8'sd6, -8'sd5},
        32'sd100
    );
    check_dot_product(
        {-8'sd128, -8'sd128, -8'sd128, -8'sd128},
        {-8'sd128, -8'sd128, -8'sd128, -8'sd128},
        32'sd0
    );

    // Explicit overflow case.
    check_dot_product(
        {8'sd127, 8'sd127, 8'sd127, 8'sd127},
        {8'sd127, 8'sd127, 8'sd127, 8'sd127},
        32'sh7fffffff
    );

    for (test_index = 0; test_index < 2500; test_index = test_index + 1)
    begin
        for (lane = 0; lane < LANES; lane = lane + 1)
        begin
            random_activations[lane*8 +: 8] = $random(seed);
            random_weights[lane*8 +: 8] = $random(seed);
        end
        random_accumulator = $random(seed) % 1000000;
        check_dot_product(random_activations, random_weights, random_accumulator);
    end

    @(negedge clk);
    valid_in = 1'b0;
    @(posedge clk);
    #1;

    if (valid_out !== 1'b0)
    begin
        $display("ERROR: dot-product valid_out did not follow valid_in low");
        errors = errors + 1;
    end

    if (errors == 0)
        $display("PASS: 4-lane signed INT8 dot product verified");
    else
        $display("FAIL: signed INT8 dot product errors=%0d", errors);

    $finish;
end

endmodule
