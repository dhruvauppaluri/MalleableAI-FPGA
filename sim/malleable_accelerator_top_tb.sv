`timescale 1ns/1ps

module malleable_accelerator_top_tb;

localparam integer LANES = 4;
localparam integer MAX_LAYERS = 4;
localparam integer MAX_DIM = 16;
localparam integer WEIGHT_DEPTH = 1024;
localparam integer PARAM_DEPTH = 64;

localparam [2:0] CFG_DESCRIPTOR = 3'd0;
localparam [2:0] CFG_INPUT      = 3'd1;
localparam [2:0] CFG_WEIGHT     = 3'd2;
localparam [2:0] CFG_BIAS       = 3'd3;
localparam [2:0] CFG_MULTIPLIER = 3'd4;
localparam [2:0] CFG_SHIFT      = 3'd5;

reg clk;
reg reset;
reg cfg_valid;
wire cfg_ready;
reg [2:0] cfg_kind;
reg [15:0] cfg_addr;
reg [31:0] cfg_data;
reg start;
wire busy;
wire done;
wire overflow_error;
wire config_error;
reg result_read_en;
reg [15:0] result_read_addr;
wire signed [7:0] result_read_data;
wire result_read_valid;
wire [15:0] result_count;
wire [63:0] cycles, tiles, useful_macs, compute_cycles, controller_cycles;
wire [63:0] configuration_writes, result_reads;
reg [63:0] saved_cycles;

integer input_values [0:MAX_DIM-1];
integer weight_values [0:WEIGHT_DEPTH-1];
integer bias_values [0:PARAM_DEPTH-1];
integer multiplier_values [0:PARAM_DEPTH-1];
integer shift_values [0:PARAM_DEPTH-1];
integer hidden_values [0:MAX_DIM-1];
integer expected_values [0:MAX_DIM-1];

integer errors;
integer seed;
integer test_index;
integer input_index;
integer output_index;
integer in_count;
integer out_count;
integer accumulator_ref;
integer timeout;

malleable_accelerator_top #(
    .LANES(LANES),
    .MAX_LAYERS(MAX_LAYERS),
    .MAX_DIM(MAX_DIM),
    .WEIGHT_DEPTH(WEIGHT_DEPTH),
    .PARAM_DEPTH(PARAM_DEPTH)
) dut (
    .clk(clk),
    .reset(reset),
    .cfg_valid(cfg_valid),
    .cfg_ready(cfg_ready),
    .cfg_kind(cfg_kind),
    .cfg_addr(cfg_addr),
    .cfg_data(cfg_data),
    .start(start),
    .busy(busy),
    .done(done),
    .overflow_error(overflow_error),
    .config_error(config_error),
    .result_read_en(result_read_en),
    .result_read_addr(result_read_addr),
    .result_read_data(result_read_data),
    .result_read_valid(result_read_valid),
    .result_count(result_count),
    .cycles(cycles), .tiles(tiles), .useful_macs(useful_macs),
    .compute_cycles(compute_cycles), .controller_cycles(controller_cycles),
    .configuration_writes(configuration_writes), .result_reads(result_reads)
);

always #5 clk = ~clk;

function integer requantize_reference;
    input integer accumulator_value;
    input integer bias_value;
    input integer multiplier_value;
    input integer shift_value;
    input integer relu_value;
    reg signed [63:0] product_value;
    reg        [63:0] magnitude_value;
    reg signed [63:0] rounded_value;
    begin
        product_value = (accumulator_value + bias_value) * multiplier_value;
        if (product_value < 0)
            magnitude_value = -product_value;
        else
            magnitude_value = product_value;

        if (shift_value == 0)
            rounded_value = magnitude_value;
        else
            rounded_value =
                (magnitude_value + (64'd1 << (shift_value - 1))) >> shift_value;

        if (product_value < 0)
            rounded_value = -rounded_value;

        if (relu_value && (rounded_value < 0))
            requantize_reference = 0;
        else if (rounded_value > 127)
            requantize_reference = 127;
        else if (rounded_value < -128)
            requantize_reference = -128;
        else
            requantize_reference = rounded_value;
    end
endfunction

task cfg_write;
    input [2:0] write_kind;
    input integer write_addr;
    input signed [31:0] write_data;
    begin
        @(negedge clk);
        if (!cfg_ready)
        begin
            $display("ERROR: configuration interface was not ready");
            errors = errors + 1;
        end
        cfg_kind  = write_kind;
        cfg_addr  = write_addr[15:0];
        cfg_data  = write_data;
        cfg_valid = 1'b1;
        @(negedge clk);
        cfg_valid = 1'b0;
    end
endtask

task write_descriptor;
    input integer descriptor_index;
    input integer descriptor_inputs;
    input integer descriptor_outputs;
    input integer weight_base;
    input integer param_base;
    input integer relu_value;
    integer descriptor_address;
    begin
        descriptor_address = 1 + descriptor_index * 8;
        cfg_write(CFG_DESCRIPTOR, descriptor_address + 0, descriptor_inputs);
        cfg_write(CFG_DESCRIPTOR, descriptor_address + 1, descriptor_outputs);
        cfg_write(CFG_DESCRIPTOR, descriptor_address + 2, weight_base);
        cfg_write(CFG_DESCRIPTOR, descriptor_address + 3, param_base);
        cfg_write(CFG_DESCRIPTOR, descriptor_address + 4, relu_value);
    end
endtask

task pulse_start;
    begin
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
    end
endtask

task wait_for_done;
    begin
        timeout = 0;
        while (!done && timeout < 10000)
        begin
            @(negedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 10000)
        begin
            $display("ERROR: accelerator timed out");
            errors = errors + 1;
        end
    end
endtask

task check_result;
    input integer read_index;
    input integer expected;
    begin
        @(negedge clk);
        result_read_addr = read_index[15:0];
        result_read_en   = 1'b1;
        @(posedge clk);
        #1;
        if (!result_read_valid)
        begin
            $display("ERROR: result %0d was not marked valid", read_index);
            errors = errors + 1;
        end
        else if (result_read_data !== expected[7:0])
        begin
            $display(
                "ERROR: result %0d expected=%0d observed=%0d",
                read_index,
                expected,
                result_read_data
            );
            errors = errors + 1;
        end
        @(negedge clk);
        result_read_en = 1'b0;
    end
endtask

task configure_one_layer;
    input integer layer_inputs;
    input integer layer_outputs;
    integer local_input;
    integer local_output;
    begin
        cfg_write(CFG_DESCRIPTOR, 0, 1);
        write_descriptor(0, layer_inputs, layer_outputs, 0, 0, 0);

        for (local_input = 0; local_input < layer_inputs; local_input = local_input + 1)
            cfg_write(CFG_INPUT, local_input, input_values[local_input]);

        for (local_output = 0; local_output < layer_outputs; local_output = local_output + 1)
        begin
            for (local_input = 0; local_input < layer_inputs; local_input = local_input + 1)
                cfg_write(
                    CFG_WEIGHT,
                    local_output * layer_inputs + local_input,
                    weight_values[local_output * layer_inputs + local_input]
                );
            cfg_write(CFG_BIAS, local_output, bias_values[local_output]);
            cfg_write(CFG_MULTIPLIER, local_output, multiplier_values[local_output]);
            cfg_write(CFG_SHIFT, local_output, shift_values[local_output]);
        end
    end
endtask

task run_random_one_layer;
    integer local_input;
    integer local_output;
    begin
        in_count  = (($random(seed) & 32'h7fffffff) % 9) + 1;
        out_count = (($random(seed) & 32'h7fffffff) % 7) + 1;

        for (local_input = 0; local_input < in_count; local_input = local_input + 1)
            input_values[local_input] = ($random(seed) & 31) - 16;

        for (local_output = 0; local_output < out_count; local_output = local_output + 1)
        begin
            accumulator_ref = 0;
            for (local_input = 0; local_input < in_count; local_input = local_input + 1)
            begin
                weight_values[local_output * in_count + local_input] =
                    ($random(seed) & 31) - 16;
                accumulator_ref = accumulator_ref
                    + input_values[local_input]
                    * weight_values[local_output * in_count + local_input];
            end
            bias_values[local_output] = ($random(seed) & 127) - 64;
            multiplier_values[local_output] = ($random(seed) & 7) + 1;
            shift_values[local_output] = $random(seed) & 7;
            expected_values[local_output] = requantize_reference(
                accumulator_ref,
                bias_values[local_output],
                multiplier_values[local_output],
                shift_values[local_output],
                0
            );
        end

        configure_one_layer(in_count, out_count);
        pulse_start();
        wait_for_done();

        if (result_count !== out_count)
        begin
            $display("ERROR: one-layer result count expected=%0d observed=%0d",
                     out_count, result_count);
            errors = errors + 1;
        end

        for (local_output = 0; local_output < out_count; local_output = local_output + 1)
            check_result(local_output, expected_values[local_output]);
    end
endtask

task run_random_two_layer;
    integer local_input;
    integer hidden_index;
    integer final_index;
    integer weight_address;
    begin
        // Required demonstration topology: 7 inputs -> 5 hidden -> 3 outputs.
        for (local_input = 0; local_input < 7; local_input = local_input + 1)
            input_values[local_input] = ($random(seed) & 31) - 16;

        for (hidden_index = 0; hidden_index < 5; hidden_index = hidden_index + 1)
        begin
            accumulator_ref = 0;
            for (local_input = 0; local_input < 7; local_input = local_input + 1)
            begin
                weight_address = hidden_index * 7 + local_input;
                weight_values[weight_address] = ($random(seed) & 15) - 8;
                accumulator_ref = accumulator_ref
                    + input_values[local_input] * weight_values[weight_address];
            end
            bias_values[hidden_index] = ($random(seed) & 63) - 32;
            multiplier_values[hidden_index] = ($random(seed) & 3) + 1;
            shift_values[hidden_index] = ($random(seed) & 3) + 1;
            hidden_values[hidden_index] = requantize_reference(
                accumulator_ref,
                bias_values[hidden_index],
                multiplier_values[hidden_index],
                shift_values[hidden_index],
                1
            );
        end

        for (final_index = 0; final_index < 3; final_index = final_index + 1)
        begin
            accumulator_ref = 0;
            for (hidden_index = 0; hidden_index < 5; hidden_index = hidden_index + 1)
            begin
                weight_address = 64 + final_index * 5 + hidden_index;
                weight_values[weight_address] = ($random(seed) & 15) - 8;
                accumulator_ref = accumulator_ref
                    + hidden_values[hidden_index] * weight_values[weight_address];
            end
            bias_values[16 + final_index] = ($random(seed) & 63) - 32;
            multiplier_values[16 + final_index] = ($random(seed) & 3) + 1;
            shift_values[16 + final_index] = ($random(seed) & 3) + 1;
            expected_values[final_index] = requantize_reference(
                accumulator_ref,
                bias_values[16 + final_index],
                multiplier_values[16 + final_index],
                shift_values[16 + final_index],
                0
            );
        end

        cfg_write(CFG_DESCRIPTOR, 0, 2);
        write_descriptor(0, 7, 5, 0, 0, 1);
        write_descriptor(1, 5, 3, 64, 16, 0);

        for (local_input = 0; local_input < 7; local_input = local_input + 1)
            cfg_write(CFG_INPUT, local_input, input_values[local_input]);

        for (weight_address = 0; weight_address < 35; weight_address = weight_address + 1)
            cfg_write(CFG_WEIGHT, weight_address, weight_values[weight_address]);
        for (weight_address = 64; weight_address < 79; weight_address = weight_address + 1)
            cfg_write(CFG_WEIGHT, weight_address, weight_values[weight_address]);

        for (hidden_index = 0; hidden_index < 5; hidden_index = hidden_index + 1)
        begin
            cfg_write(CFG_BIAS, hidden_index, bias_values[hidden_index]);
            cfg_write(CFG_MULTIPLIER, hidden_index, multiplier_values[hidden_index]);
            cfg_write(CFG_SHIFT, hidden_index, shift_values[hidden_index]);
        end
        for (final_index = 0; final_index < 3; final_index = final_index + 1)
        begin
            cfg_write(CFG_BIAS, 16 + final_index, bias_values[16 + final_index]);
            cfg_write(CFG_MULTIPLIER, 16 + final_index,
                      multiplier_values[16 + final_index]);
            cfg_write(CFG_SHIFT, 16 + final_index, shift_values[16 + final_index]);
        end

        pulse_start();
        wait_for_done();

        if (result_count !== 3)
        begin
            $display("ERROR: two-layer result count expected=3 observed=%0d", result_count);
            errors = errors + 1;
        end
        for (final_index = 0; final_index < 3; final_index = final_index + 1)
            check_result(final_index, expected_values[final_index]);
    end
endtask

initial
begin
    clk              = 1'b0;
    reset            = 1'b1;
    cfg_valid        = 1'b0;
    cfg_kind         = 3'd0;
    cfg_addr         = 16'd0;
    cfg_data         = 32'd0;
    start            = 1'b0;
    result_read_en   = 1'b0;
    result_read_addr = 16'd0;
    errors           = 0;
    seed             = 32'h4d414c4c;

    repeat (2) @(posedge clk);
    reset = 1'b0;

    // Reject an empty network.
    cfg_write(CFG_DESCRIPTOR, 0, 0);
    pulse_start();
    @(posedge clk);
    #1;
    if (!config_error || busy)
    begin
        $display("ERROR: invalid empty network was not rejected");
        errors = errors + 1;
    end

    // Structurally valid descriptors are still rejected until every required
    // bias, multiplier, and shift has been written.
    cfg_write(CFG_DESCRIPTOR, 0, 1);
    write_descriptor(0, 1, 1, 0, 0, 0);
    cfg_write(CFG_INPUT, 0, 1);
    cfg_write(CFG_WEIGHT, 0, 1);
    pulse_start();
    @(posedge clk);
    #1;
    if (!config_error || busy)
    begin
        $display("ERROR: network with missing parameters was not rejected");
        errors = errors + 1;
    end

    // Randomized dense-layer regression, including dimensions on both sides
    // of LANES and dimensions that require a partial final tile.
    for (test_index = 0; test_index < 1000; test_index = test_index + 1)
        run_random_one_layer();

    // Complete autonomous two-layer networks.
    for (test_index = 0; test_index < 100; test_index = test_index + 1)
        run_random_two_layer();

    // Starting while busy leaves the active job running and raises a sticky
    // configuration error.
    in_count = 9;
    out_count = 7;
    for (input_index = 0; input_index < in_count; input_index = input_index + 1)
        input_values[input_index] = input_index - 4;
    for (output_index = 0; output_index < out_count; output_index = output_index + 1)
    begin
        for (input_index = 0; input_index < in_count; input_index = input_index + 1)
            weight_values[output_index * in_count + input_index] = 1;
        bias_values[output_index] = 0;
        multiplier_values[output_index] = 1;
        shift_values[output_index] = 0;
    end
    configure_one_layer(in_count, out_count);
    cfg_write(3'd6, 0, 3);
    pulse_start();
    pulse_start();
    wait_for_done();
    if (!config_error)
    begin
        $display("ERROR: start while busy did not raise config_error");
        errors = errors + 1;
    end
    if (tiles != 21 || useful_macs != 63 || compute_cycles != 21 ||
        cycles != compute_cycles + controller_cycles)
        $fatal(1,"runtime lane counters incorrect");
    saved_cycles = cycles;
    repeat (5) @(negedge clk);
    if (cycles != saved_cycles) $fatal(1,"idle changed execution cycles");
    cfg_write(3'd6, 0, 0);
    pulse_start();
    if (!config_error || busy) $fatal(1,"zero active lanes accepted");
    cfg_write(3'd6, 0, LANES+1);
    pulse_start();
    if (!config_error || busy) $fatal(1,"excess active lanes accepted");
    cfg_write(3'd6, 0, LANES);

    // The next valid run clears prior sticky errors. Force the accumulator's
    // overflow report to verify top-level propagation independently of the
    // normal 32-bit-safe MAX_DIM limit.
    configure_one_layer(in_count, out_count);
    force dut.tiled_overflow = 1'b1;
    pulse_start();
    wait_for_done();
    release dut.tiled_overflow;
    if (!overflow_error || config_error)
    begin
        $display("ERROR: sticky error clear/overflow propagation failed");
        errors = errors + 1;
    end

    // Reset during an active job returns the controller to idle.
    configure_one_layer(in_count, out_count);
    pulse_start();
    repeat (3) @(posedge clk);
    reset = 1'b1;
    repeat (2) @(posedge clk);
    reset = 1'b0;
    @(posedge clk);
    #1;
    if (busy || done || overflow_error || config_error)
    begin
        $display("ERROR: reset did not return accelerator to idle");
        errors = errors + 1;
    end
    if (cycles || tiles || useful_macs || compute_cycles || controller_cycles ||
        configuration_writes || result_reads) $fatal(1,"reset did not clear counters");

    if (errors == 0)
        $display(
            "PASS: autonomous accelerator verified with 1000 dense layers and 100 two-layer networks"
        );
    else
        $fatal(1, "FAIL: autonomous accelerator errors=%0d", errors);

    $finish;
end

endmodule
