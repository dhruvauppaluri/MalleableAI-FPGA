`timescale 1ns/1ps

// Board-independent autonomous dense-network accelerator.
//
// Configuration kinds:
//   0: descriptor/control
//      address 0                         = layer count
//      address 1 + layer*8 + field       = descriptor field
//      fields: 0 input count, 1 output count, 2 weight base,
//              3 parameter base, 4 ReLU enable
//   1: input activation (signed INT8 in cfg_data[7:0])
//   2: weight (signed INT8 in cfg_data[7:0])
//   3: bias (signed INT32)
//   4: positive multiplier (cfg_data[30:0], cfg_data[31] must be zero)
//   5: right shift (0..62)
// Every input and weight address consumed by the descriptors must be loaded
// before start. Descriptor fields and per-output parameters carry explicit
// validity tracking and are rejected when incomplete.
module malleable_accelerator_top #(
    parameter integer LANES        = 4,
    parameter integer MAX_LAYERS   = 4,
    parameter integer MAX_DIM      = 64,
    parameter integer WEIGHT_DEPTH = MAX_LAYERS * MAX_DIM * MAX_DIM,
    parameter integer PARAM_DEPTH  = MAX_LAYERS * MAX_DIM
) (
    input  wire               clk,
    input  wire               reset,

    input  wire               cfg_valid,
    output wire               cfg_ready,
    input  wire [2:0]         cfg_kind,
    input  wire [15:0]        cfg_addr,
    input  wire [31:0]        cfg_data,

    input  wire               start,
    output reg                busy,
    output reg                done,
    output reg                overflow_error,
    output reg                config_error,

    input  wire               result_read_en,
    input  wire [15:0]        result_read_addr,
    output reg  signed [7:0]  result_read_data,
    output reg                result_read_valid,
    output reg  [15:0]        result_count
);

localparam [2:0] CFG_DESCRIPTOR = 3'd0;
localparam [2:0] CFG_INPUT      = 3'd1;
localparam [2:0] CFG_WEIGHT     = 3'd2;
localparam [2:0] CFG_BIAS       = 3'd3;
localparam [2:0] CFG_MULTIPLIER = 3'd4;
localparam [2:0] CFG_SHIFT      = 3'd5;

localparam [2:0] STATE_IDLE       = 3'd0;
localparam [2:0] STATE_ACC_START  = 3'd1;
localparam [2:0] STATE_SEND_TILE  = 3'd2;
localparam [2:0] STATE_WAIT_ACC   = 3'd3;
localparam [2:0] STATE_STORE      = 3'd4;

reg signed [7:0] activation_bank_a [0:MAX_DIM-1];
reg signed [7:0] activation_bank_b [0:MAX_DIM-1];
reg signed [7:0] weight_memory    [0:WEIGHT_DEPTH-1];
reg signed [31:0] bias_memory     [0:PARAM_DEPTH-1];
reg        [30:0] multiplier_memory [0:PARAM_DEPTH-1];
reg        [5:0]  shift_memory    [0:PARAM_DEPTH-1];
reg                bias_valid      [0:PARAM_DEPTH-1];
reg                multiplier_valid [0:PARAM_DEPTH-1];
reg                shift_valid     [0:PARAM_DEPTH-1];

reg [15:0] descriptor_input_count  [0:MAX_LAYERS-1];
reg [15:0] descriptor_output_count [0:MAX_LAYERS-1];
reg [31:0] descriptor_weight_base  [0:MAX_LAYERS-1];
reg [31:0] descriptor_param_base   [0:MAX_LAYERS-1];
reg        descriptor_relu         [0:MAX_LAYERS-1];
reg [4:0]  descriptor_valid         [0:MAX_LAYERS-1];
reg [15:0] layer_count;
reg        layer_count_valid;

reg [2:0] state;
reg [15:0] layer_index;
reg [15:0] output_index;
reg [15:0] tile_base;
reg        source_bank;
reg        final_bank;

reg                                tiled_start;
reg                                tiled_tile_valid;
wire                               tiled_tile_ready;
reg                                tiled_tile_last;
reg [LANES*8-1:0]                 tiled_activations;
reg [LANES*8-1:0]                 tiled_weights;
wire signed [31:0]                tiled_result;
wire                               tiled_busy;
wire                               tiled_done;
wire                               tiled_overflow;

wire signed [7:0] postprocessed_result;

integer lane;
integer check_layer;
integer check_output;
integer input_offset;
integer weight_offset;
integer reset_index;
reg configuration_valid;
reg [31:0] checked_weight_end;
reg [31:0] checked_param_end;

wire [15:0] descriptor_write_offset;
wire [12:0] descriptor_write_layer;
wire [2:0]  descriptor_write_field;

assign descriptor_write_offset = cfg_addr - 16'd1;
assign descriptor_write_layer = descriptor_write_offset >> 3;
assign descriptor_write_field = descriptor_write_offset[2:0];

assign cfg_ready = !busy;

int8_tiled_accumulator #(
    .LANES(LANES)
) tiled_accumulator (
    .clk(clk),
    .reset(reset),
    .start(tiled_start),
    .tile_valid(tiled_tile_valid),
    .tile_ready(tiled_tile_ready),
    .tile_last(tiled_tile_last),
    .activations(tiled_activations),
    .weights(tiled_weights),
    .result(tiled_result),
    .busy(tiled_busy),
    .done(tiled_done),
    .overflow(tiled_overflow)
);

int8_postprocess postprocess (
    .accumulator(tiled_result),
    .bias(bias_memory[descriptor_param_base[layer_index] + output_index]),
    .multiplier(multiplier_memory[descriptor_param_base[layer_index] + output_index]),
    .shift(shift_memory[descriptor_param_base[layer_index] + output_index]),
    .relu_enable(descriptor_relu[layer_index]),
    .result(postprocessed_result)
);

always @(*)
begin
    tiled_start       = 1'b0;
    tiled_tile_valid  = 1'b0;
    tiled_tile_last   = 1'b0;
    tiled_activations = {(LANES*8){1'b0}};
    tiled_weights     = {(LANES*8){1'b0}};
    input_offset      = 0;
    weight_offset     = 0;
    lane              = 0;

    if (state == STATE_ACC_START)
        tiled_start = 1'b1;

    if (state == STATE_SEND_TILE)
    begin
        tiled_tile_valid = 1'b1;
        tiled_tile_last = (tile_base + LANES >= descriptor_input_count[layer_index]);

        for (lane = 0; lane < LANES; lane = lane + 1)
        begin
            input_offset = tile_base + lane;
            if (input_offset < descriptor_input_count[layer_index])
            begin
                if (source_bank == 1'b0)
                    tiled_activations[lane*8 +: 8] = activation_bank_a[input_offset];
                else
                    tiled_activations[lane*8 +: 8] = activation_bank_b[input_offset];

                weight_offset = descriptor_weight_base[layer_index]
                              + output_index * descriptor_input_count[layer_index]
                              + input_offset;
                tiled_weights[lane*8 +: 8] = weight_memory[weight_offset];
            end
        end
    end
end

always @(*)
begin
    configuration_valid = 1'b1;
    checked_weight_end  = 32'd0;
    checked_param_end   = 32'd0;
    check_layer         = 0;
    check_output        = 0;

    if (!layer_count_valid || (layer_count == 0) || (layer_count > MAX_LAYERS))
        configuration_valid = 1'b0;

    for (check_layer = 0; check_layer < MAX_LAYERS; check_layer = check_layer + 1)
    begin
        if (check_layer < layer_count)
        begin
            if (descriptor_valid[check_layer] != 5'b11111)
                configuration_valid = 1'b0;

            if ((descriptor_input_count[check_layer] == 0) ||
                (descriptor_input_count[check_layer] > MAX_DIM) ||
                (descriptor_output_count[check_layer] == 0) ||
                (descriptor_output_count[check_layer] > MAX_DIM))
                configuration_valid = 1'b0;

            if ((check_layer > 0) &&
                (descriptor_input_count[check_layer] !=
                 descriptor_output_count[check_layer-1]))
                configuration_valid = 1'b0;

            checked_weight_end = descriptor_weight_base[check_layer]
                               + descriptor_input_count[check_layer]
                               * descriptor_output_count[check_layer];
            checked_param_end = descriptor_param_base[check_layer]
                              + descriptor_output_count[check_layer];

            if (checked_weight_end > WEIGHT_DEPTH)
                configuration_valid = 1'b0;
            if (checked_param_end > PARAM_DEPTH)
                configuration_valid = 1'b0;

            for (check_output = 0; check_output < PARAM_DEPTH; check_output = check_output + 1)
            begin
                if ((check_output >= descriptor_param_base[check_layer]) &&
                    (check_output < checked_param_end))
                begin
                    if (!bias_valid[check_output] ||
                        !multiplier_valid[check_output] ||
                        !shift_valid[check_output] ||
                        (multiplier_memory[check_output] == 0) ||
                        (shift_memory[check_output] > 6'd62))
                        configuration_valid = 1'b0;
                end
            end
        end
    end
end

always @(posedge clk)
begin
    if (reset)
    begin
        state             <= STATE_IDLE;
        busy              <= 1'b0;
        done              <= 1'b0;
        overflow_error    <= 1'b0;
        config_error      <= 1'b0;
        result_read_data  <= 8'sd0;
        result_read_valid <= 1'b0;
        result_count      <= 16'd0;
        layer_count       <= 16'd0;
        layer_index       <= 16'd0;
        output_index      <= 16'd0;
        tile_base         <= 16'd0;
        source_bank       <= 1'b0;
        final_bank        <= 1'b0;
        layer_count_valid <= 1'b0;
        for (reset_index = 0; reset_index < MAX_LAYERS; reset_index = reset_index + 1)
            descriptor_valid[reset_index] <= 5'b00000;
        for (reset_index = 0; reset_index < PARAM_DEPTH; reset_index = reset_index + 1)
        begin
            bias_valid[reset_index]       <= 1'b0;
            multiplier_valid[reset_index] <= 1'b0;
            shift_valid[reset_index]      <= 1'b0;
        end
    end
    else
    begin
        done              <= 1'b0;
        result_read_valid <= 1'b0;

        if (start && busy)
            config_error <= 1'b1;

        if (cfg_valid && cfg_ready)
        begin
            case (cfg_kind)
                CFG_DESCRIPTOR:
                begin
                    if (cfg_addr == 0)
                    begin
                        layer_count <= cfg_data[15:0];
                        layer_count_valid <= 1'b1;
                    end
                    else
                    begin
                        if (descriptor_write_layer < MAX_LAYERS)
                        begin
                            case (descriptor_write_field)
                                0:
                                begin
                                    descriptor_input_count[descriptor_write_layer] <= cfg_data[15:0];
                                    descriptor_valid[descriptor_write_layer][0] <= 1'b1;
                                end
                                1:
                                begin
                                    descriptor_output_count[descriptor_write_layer] <= cfg_data[15:0];
                                    descriptor_valid[descriptor_write_layer][1] <= 1'b1;
                                end
                                2:
                                begin
                                    descriptor_weight_base[descriptor_write_layer] <= cfg_data;
                                    descriptor_valid[descriptor_write_layer][2] <= 1'b1;
                                end
                                3:
                                begin
                                    descriptor_param_base[descriptor_write_layer] <= cfg_data;
                                    descriptor_valid[descriptor_write_layer][3] <= 1'b1;
                                end
                                4:
                                begin
                                    descriptor_relu[descriptor_write_layer] <= cfg_data[0];
                                    descriptor_valid[descriptor_write_layer][4] <= 1'b1;
                                end
                                default: config_error <= 1'b1;
                            endcase
                        end
                        else
                            config_error <= 1'b1;
                    end
                end

                CFG_INPUT:
                begin
                    if (cfg_addr < MAX_DIM)
                        activation_bank_a[cfg_addr] <= cfg_data[7:0];
                    else
                        config_error <= 1'b1;
                end

                CFG_WEIGHT:
                begin
                    if (cfg_addr < WEIGHT_DEPTH)
                        weight_memory[cfg_addr] <= cfg_data[7:0];
                    else
                        config_error <= 1'b1;
                end

                CFG_BIAS:
                begin
                    if (cfg_addr < PARAM_DEPTH)
                    begin
                        bias_memory[cfg_addr] <= cfg_data;
                        bias_valid[cfg_addr] <= 1'b1;
                    end
                    else
                        config_error <= 1'b1;
                end

                CFG_MULTIPLIER:
                begin
                    if ((cfg_addr < PARAM_DEPTH) && !cfg_data[31] &&
                        (cfg_data[30:0] != 0))
                    begin
                        multiplier_memory[cfg_addr] <= cfg_data[30:0];
                        multiplier_valid[cfg_addr] <= 1'b1;
                    end
                    else
                    begin
                        config_error <= 1'b1;
                        if (cfg_addr < PARAM_DEPTH)
                            multiplier_valid[cfg_addr] <= 1'b0;
                    end
                end

                CFG_SHIFT:
                begin
                    if ((cfg_addr < PARAM_DEPTH) && (cfg_data <= 32'd62))
                    begin
                        shift_memory[cfg_addr] <= cfg_data[5:0];
                        shift_valid[cfg_addr] <= 1'b1;
                    end
                    else
                    begin
                        config_error <= 1'b1;
                        if (cfg_addr < PARAM_DEPTH)
                            shift_valid[cfg_addr] <= 1'b0;
                    end
                end

                default: config_error <= 1'b1;
            endcase
        end

        if (result_read_en && !busy && (result_read_addr < result_count))
        begin
            if (final_bank == 1'b0)
                result_read_data <= activation_bank_a[result_read_addr];
            else
                result_read_data <= activation_bank_b[result_read_addr];
            result_read_valid <= 1'b1;
        end

        case (state)
            STATE_IDLE:
            begin
                if (start && !busy)
                begin
                    if (configuration_valid)
                    begin
                        busy           <= 1'b1;
                        overflow_error <= 1'b0;
                        config_error   <= 1'b0;
                        result_count   <= 16'd0;
                        layer_index    <= 16'd0;
                        output_index   <= 16'd0;
                        tile_base      <= 16'd0;
                        source_bank    <= 1'b0;
                        state          <= STATE_ACC_START;
                    end
                    else
                        config_error <= 1'b1;
                end
            end

            STATE_ACC_START:
            begin
                tile_base <= 16'd0;
                state     <= STATE_SEND_TILE;
            end

            STATE_SEND_TILE:
            begin
                if (tiled_tile_valid && tiled_tile_ready)
                begin
                    if (tiled_tile_last)
                        state <= STATE_WAIT_ACC;
                    else
                        tile_base <= tile_base + LANES;
                end
            end

            STATE_WAIT_ACC:
            begin
                if (tiled_done)
                begin
                    overflow_error <= overflow_error || tiled_overflow;
                    state          <= STATE_STORE;
                end
            end

            STATE_STORE:
            begin
                if (source_bank == 1'b0)
                    activation_bank_b[output_index] <= postprocessed_result;
                else
                    activation_bank_a[output_index] <= postprocessed_result;

                if (output_index + 1 < descriptor_output_count[layer_index])
                begin
                    output_index <= output_index + 1'b1;
                    state        <= STATE_ACC_START;
                end
                else if (layer_index + 1 < layer_count)
                begin
                    layer_index  <= layer_index + 1'b1;
                    output_index <= 16'd0;
                    source_bank  <= !source_bank;
                    state        <= STATE_ACC_START;
                end
                else
                begin
                    final_bank   <= !source_bank;
                    result_count <= descriptor_output_count[layer_index];
                    busy         <= 1'b0;
                    done         <= 1'b1;
                    state        <= STATE_IDLE;
                end
            end

            default:
            begin
                state        <= STATE_IDLE;
                busy         <= 1'b0;
                config_error <= 1'b1;
            end
        endcase
    end
end

endmodule
