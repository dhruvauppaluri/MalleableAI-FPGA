`timescale 1ns/1ps

module int8_postprocess_tb;

reg signed [31:0] accumulator;
reg signed [31:0] bias;
reg        [30:0] multiplier;
reg        [5:0]  shift;
reg               relu_enable;
wire signed [7:0] result;

integer errors;
integer seed;
integer test_index;
integer random_accumulator;
integer random_bias;
integer random_multiplier;
integer random_shift;
integer random_relu;

int8_postprocess dut (
    .accumulator(accumulator),
    .bias(bias),
    .multiplier(multiplier),
    .shift(shift),
    .relu_enable(relu_enable),
    .result(result)
);

task check_postprocess;
    input signed [31:0] test_accumulator;
    input signed [31:0] test_bias;
    input        [30:0] test_multiplier;
    input        [5:0]  test_shift;
    input               test_relu;
    reg signed [32:0] biased_ref;
    reg signed [64:0] product_ref;
    reg        [64:0] magnitude_ref;
    reg        [64:0] rounded_magnitude_ref;
    reg signed [65:0] rounded_ref;
    reg signed [7:0] expected;
    begin
        biased_ref = {test_accumulator[31], test_accumulator}
                   + {test_bias[31], test_bias};
        product_ref = biased_ref * $signed({1'b0, test_multiplier});

        if (product_ref < 0)
            magnitude_ref = -product_ref;
        else
            magnitude_ref = product_ref;

        if (test_shift == 0)
            rounded_magnitude_ref = magnitude_ref;
        else
            rounded_magnitude_ref =
                (magnitude_ref + (65'd1 << (test_shift - 1'b1))) >> test_shift;

        if (product_ref < 0)
            rounded_ref = -$signed({1'b0, rounded_magnitude_ref});
        else
            rounded_ref = $signed({1'b0, rounded_magnitude_ref});

        if (test_relu && (rounded_ref < 0))
            expected = 8'sd0;
        else if (rounded_ref > 66'sd127)
            expected = 8'sd127;
        else if (rounded_ref < -66'sd128)
            expected = -8'sd128;
        else
            expected = rounded_ref[7:0];

        accumulator = test_accumulator;
        bias         = test_bias;
        multiplier   = test_multiplier;
        shift        = test_shift;
        relu_enable  = test_relu;
        #1;

        if (result !== expected)
        begin
            $display(
                "ERROR postprocess: acc=%0d bias=%0d mult=%0d shift=%0d relu=%0b expected=%0d observed=%0d",
                test_accumulator,
                test_bias,
                test_multiplier,
                test_shift,
                test_relu,
                expected,
                result
            );
            errors = errors + 1;
        end
    end
endtask

initial
begin
    accumulator = 32'sd0;
    bias         = 32'sd0;
    multiplier   = 31'd1;
    shift        = 6'd0;
    relu_enable  = 1'b0;
    errors       = 0;
    seed         = 32'h52515154;

    // Identity, bias, rounding ties, saturation, and activation order.
    check_postprocess(32'sd12, 32'sd0, 31'd1, 6'd0, 1'b0);
    check_postprocess(32'sd10, 32'sd2, 31'd1, 6'd0, 1'b0);
    check_postprocess(32'sd1, 32'sd0, 31'd1, 6'd1, 1'b0);
    check_postprocess(-32'sd1, 32'sd0, 31'd1, 6'd1, 1'b0);
    check_postprocess(32'sd3, 32'sd0, 31'd1, 6'd1, 1'b0);
    check_postprocess(-32'sd3, 32'sd0, 31'd1, 6'd1, 1'b0);
    check_postprocess(32'sd1000, 32'sd0, 31'd1, 6'd0, 1'b0);
    check_postprocess(-32'sd1000, 32'sd0, 31'd1, 6'd0, 1'b0);
    check_postprocess(-32'sd1000, 32'sd0, 31'd1, 6'd0, 1'b1);
    check_postprocess(32'sh7fffffff, 32'sh7fffffff, 31'h7fffffff, 6'd62, 1'b0);
    check_postprocess(-32'sh7fffffff - 1, -32'sh7fffffff - 1,
                      31'h7fffffff, 6'd62, 1'b0);

    for (test_index = 0; test_index < 10000; test_index = test_index + 1)
    begin
        random_accumulator = $random(seed) % 1000000;
        random_bias        = $random(seed) % 100000;
        random_multiplier  = ($random(seed) & 32'h0000ffff) + 1;
        random_shift       = $random(seed) & 6'h1f;
        random_relu        = $random(seed) & 1;
        check_postprocess(
            random_accumulator,
            random_bias,
            random_multiplier,
            random_shift,
            random_relu
        );
    end

    if (errors == 0)
        $display("PASS: bias, requantization, saturation, and ReLU verified");
    else
        $fatal(1, "FAIL: postprocess errors=%0d", errors);

    $finish;
end

endmodule
