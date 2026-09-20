`timescale 1ns/1ps

module int8_mac_tb;

reg clk;
reg reset;
reg valid_in;
reg signed [7:0] multiplicand;
reg signed [7:0] multiplier;
reg signed [31:0] accumulator_in;

wire signed [31:0] result;
wire valid_out;
wire overflow;

integer errors;
integer seed;
integer test_index;
reg signed [7:0] random_a;
reg signed [7:0] random_b;
reg signed [31:0] random_accumulator;

int8_mac dut (
    .clk(clk),
    .reset(reset),
    .valid_in(valid_in),
    .multiplicand(multiplicand),
    .multiplier(multiplier),
    .accumulator_in(accumulator_in),
    .result(result),
    .valid_out(valid_out),
    .overflow(overflow)
);

always #5 clk = ~clk;

task check_mac;
    input signed [7:0] test_a;
    input signed [7:0] test_b;
    input signed [31:0] test_accumulator;
    reg signed [63:0] exact_result;
    reg signed [31:0] expected_result;
    reg expected_overflow;
    begin
        exact_result = test_accumulator + (test_a * test_b);
        expected_result = exact_result[31:0];
        expected_overflow = (exact_result > 64'sh000000007fffffff) ||
                            (exact_result < -64'sh0000000080000000);

        @(negedge clk);
        multiplicand  = test_a;
        multiplier    = test_b;
        accumulator_in = test_accumulator;
        valid_in      = 1'b1;

        @(posedge clk);
        #1;

        if (valid_out !== 1'b1)
        begin
            $display("ERROR: valid_out was not asserted");
            errors = errors + 1;
        end

        if (result !== expected_result)
        begin
            $display(
                "ERROR: a=%0d b=%0d acc=%0d expected=%0d observed=%0d",
                test_a,
                test_b,
                test_accumulator,
                expected_result,
                result
            );
            errors = errors + 1;
        end

        if (overflow !== expected_overflow)
        begin
            $display(
                "ERROR: overflow mismatch a=%0d b=%0d acc=%0d expected=%0b observed=%0b",
                test_a,
                test_b,
                test_accumulator,
                expected_overflow,
                overflow
            );
            errors = errors + 1;
        end
    end
endtask

initial
begin
    clk            = 1'b0;
    reset          = 1'b1;
    valid_in       = 1'b0;
    multiplicand   = 8'sd0;
    multiplier     = 8'sd0;
    accumulator_in = 32'sd0;
    errors         = 0;
    seed           = 32'h4d414343;

    repeat (2) @(posedge clk);
    reset = 1'b0;

    // Directed sign, boundary, and accumulation cases.
    check_mac(8'sd0,    8'sd0,    32'sd0);
    check_mac(8'sd3,    8'sd4,    32'sd5);
    check_mac(-8'sd3,   8'sd4,    32'sd5);
    check_mac(-8'sd128, -8'sd128, 32'sd0);
    check_mac(8'sd127,  8'sd127,  32'sd1000);
    check_mac(-8'sd128, 8'sd127,  -32'sd1000);

    // Explicit positive and negative overflow checks.
    check_mac(8'sd127,  8'sd127,  32'sh7fffffff);
    check_mac(-8'sd128, 8'sd127,  -32'sh7fffffff - 1);

    // Seeded randomized regression. The accumulator range intentionally keeps
    // these cases away from overflow; overflow behavior is covered above.
    for (test_index = 0; test_index < 10000; test_index = test_index + 1)
    begin
        random_a = $random(seed);
        random_b = $random(seed);
        random_accumulator = $random(seed) % 1000000;
        check_mac(random_a, random_b, random_accumulator);
    end

    @(negedge clk);
    valid_in = 1'b0;
    @(posedge clk);
    #1;

    if (valid_out !== 1'b0)
    begin
        $display("ERROR: valid_out did not follow valid_in low");
        errors = errors + 1;
    end

    if (errors == 0)
        $display("PASS: signed INT8 MAC verified with 10000 randomized cases");
    else
        $display("FAIL: signed INT8 MAC errors=%0d", errors);

    $finish;
end

endmodule
