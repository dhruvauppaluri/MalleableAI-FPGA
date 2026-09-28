// Generic fixed-point SSM instruction engine. See docs/ssm-platform.md.
// This is independent of the verified INT8 dense datapath.
module ssm_operator_engine #(
    parameter integer LANES = 4,
    parameter integer MEM_WORDS = 16384,
    parameter integer PROGRAM_WORDS = 8192
) (
    input wire clk, reset,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [1:0] cfg_kind,
    input wire [31:0] cfg_addr,
    input wire signed [31:0] cfg_data,
    input wire start,
    output reg busy, done, config_error,
    output reg [63:0] cycles, useful_macs, memory_reads, memory_writes,
    output reg [63:0] saturation_count, configuration_writes,
    input wire read_en,
    input wire [31:0] read_addr,
    output reg signed [31:0] read_data,
    output reg read_valid
);
    localparam [2:0] IDLE=0, FETCH=1, DOT=2, VECTOR=3, RMS_SUM=4, RMS_WRITE=5, LUT_READ=6;
    reg [2:0] phase;
    reg signed [31:0] mem [0:MEM_WORDS-1];
    reg [31:0] program_mem [0:PROGRAM_WORDS-1];
    reg [31:0] pc, opcode, dst, src_a, src_b, length, aux, shift, bias_addr, index;
    reg [31:0] active_lanes;
    reg [31:0] lookup_addr;
    reg signed [63:0] accumulator, temporary, value;
    reg [63:0] root;
    integer lane, lookup_index;
    assign cfg_ready = !busy && !reset;

    function automatic signed [63:0] rounded(input signed [63:0] v, input [31:0] s);
        reg [63:0] magnitude;
        begin
            magnitude = v < 0 ? $unsigned(-v) : $unsigned(v);
            if (s != 0) magnitude = (magnitude + (64'd1 << (s-1))) >> s;
            rounded = v < 0 ? -$signed(magnitude) : $signed(magnitude);
        end
    endfunction
    function automatic signed [31:0] clamped(input signed [63:0] v);
        begin
            if (v > 32767) clamped = 32767;
            else if (v < -32768) clamped = -32768;
            else clamped = v[31:0];
        end
    endfunction
    function automatic [63:0] isqrt(input [63:0] n);
        reg [63:0] remainder, answer, bit_value;
        integer k;
        begin
            remainder=n; answer=0; bit_value=64'h4000000000000000;
            for (k=0;k<32;k=k+1) begin
                if (remainder >= answer + bit_value) begin
                    remainder = remainder - answer - bit_value;
                    answer = (answer >> 1) + bit_value;
                end else answer = answer >> 1;
                bit_value = bit_value >> 2;
            end
            isqrt=answer;
        end
    endfunction
    function automatic bad_range(input [31:0] base, input [31:0] count);
        begin bad_range = base >= MEM_WORDS || count > MEM_WORDS ||
                          {1'b0,base}+{1'b0,count} > MEM_WORDS; end
    endfunction
    function automatic overlap(input [31:0] d, input [31:0] dc,
                                input [31:0] a, input [31:0] ac);
        begin overlap = {1'b0,d} < {1'b0,a}+{1'b0,ac} &&
                        {1'b0,a} < {1'b0,d}+{1'b0,dc}; end
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            busy<=0; done<=0; config_error<=0; phase<=IDLE; pc<=0;
            cycles<=0; useful_macs<=0; memory_reads<=0; memory_writes<=0;
            saturation_count<=0; configuration_writes<=0;
            read_valid<=0; read_data<=0; active_lanes<=LANES;
            opcode<=0; dst<=0; src_a<=0; src_b<=0; length<=0; aux<=0;
            shift<=0; bias_addr<=0; index<=0; accumulator<=0; root<=1; lookup_addr<=0;
        end else begin
            done<=0; read_valid<=0;
            if (read_en) begin
                if (!busy && read_addr < MEM_WORDS) begin
                    read_data<=mem[read_addr]; read_valid<=1;
                end else config_error<=1;
            end
            if (cfg_valid) begin
                if (!cfg_ready) config_error<=1;
                else if (cfg_kind==0 && cfg_addr < MEM_WORDS && cfg_data >= -32768 && cfg_data <= 32767) begin
                    mem[cfg_addr]<=cfg_data; configuration_writes<=configuration_writes+1;
                end else if (cfg_kind==1 && cfg_addr < PROGRAM_WORDS) begin
                    program_mem[cfg_addr]<=cfg_data; configuration_writes<=configuration_writes+1;
                end else if (cfg_kind==2 && cfg_addr==0 && cfg_data>0 && cfg_data<=LANES) begin
                    active_lanes<=cfg_data; configuration_writes<=configuration_writes+1;
                end else config_error<=1;
            end
            if (start) begin
                if (busy || cfg_valid || config_error) config_error<=1;
                else begin
                    busy<=1; phase<=FETCH; pc<=0; cycles<=0; useful_macs<=0;
                    memory_reads<=0; memory_writes<=0; saturation_count<=0;
                end
            end
            if (busy) begin
                cycles<=cycles+1;
                case (phase)
                    FETCH: begin
                        if (pc > PROGRAM_WORDS-8) begin
                            config_error<=1; busy<=0; phase<=IDLE; done<=1;
                        end else if (program_mem[pc]==255) begin
                            busy<=0; phase<=IDLE; done<=1;
                        end else begin
                            opcode<=program_mem[pc]; dst<=program_mem[pc+1];
                            src_a<=program_mem[pc+2]; src_b<=program_mem[pc+3];
                            length<=program_mem[pc+4]; aux<=program_mem[pc+5];
                            shift<=program_mem[pc+6]; bias_addr<=program_mem[pc+7];
                            index<=0; accumulator<=0;
                            // Full-width, overflow-safe descriptor checks.
                            if (program_mem[pc]>5 || program_mem[pc+4]==0 || program_mem[pc+4]>4096 ||
                                program_mem[pc+6]>62 ||
                                bad_range(program_mem[pc+1],program_mem[pc]==0 ? 1:program_mem[pc+4]) ||
                                bad_range(program_mem[pc+2],program_mem[pc+4]) ||
                                (program_mem[pc]<=2 && bad_range(program_mem[pc+3],program_mem[pc+4])) ||
                                (program_mem[pc]==0 && bad_range(program_mem[pc+7],1)) ||
                                (program_mem[pc]==3 && bad_range(program_mem[pc+5],256)) ||
                                overlap(program_mem[pc+1],program_mem[pc]==0 ? 1:program_mem[pc+4],program_mem[pc+2],program_mem[pc+4]) ||
                                (program_mem[pc]<=2 && overlap(program_mem[pc+1],program_mem[pc]==0 ? 1:program_mem[pc+4],program_mem[pc+3],program_mem[pc+4]))) begin
                                config_error<=1; busy<=0; phase<=IDLE; done<=1;
                            end else if (program_mem[pc]==0) phase<=DOT;
                            else if (program_mem[pc]==4) phase<=RMS_SUM;
                            else phase<=VECTOR;
                        end
                    end
                    DOT: begin
                        temporary=accumulator;
                        for (lane=0;lane<LANES;lane=lane+1)
                            if (lane<active_lanes && index+lane<length)
                                temporary=temporary+$signed(mem[src_a+index+lane])*$signed(mem[src_b+index+lane]);
                        accumulator<=temporary;
                        useful_macs<=useful_macs+((length-index<active_lanes)?length-index:active_lanes);
                        memory_reads<=memory_reads+2*((length-index<active_lanes)?length-index:active_lanes);
                        if (index+active_lanes>=length) begin
                            value=rounded(temporary,shift)+$signed(mem[bias_addr]);
                            mem[dst]<=clamped(value); memory_writes<=memory_writes+1;
                            memory_reads<=memory_reads+2*((length-index<active_lanes)?length-index:active_lanes)+1;
                            if (value>32767 || value< -32768) saturation_count<=saturation_count+1;
                            pc<=pc+8; phase<=FETCH;
                        end else index<=index+active_lanes;
                    end
                    VECTOR, RMS_WRITE: begin
                        value=$signed(mem[src_a+index]);
                        if (phase==RMS_WRITE) begin
                            temporary=value*16384;
                            value=temporary<0 ? -$signed(($unsigned(-temporary)+(root>>1))/root) :
                                                  $signed(($unsigned(temporary)+(root>>1))/root);
                        end else case (opcode)
                            1: value=value+$signed(mem[src_b+index]);
                            2: value=rounded(value*$signed(mem[src_b+index]),shift);
                            3: begin
                                lookup_index=($signed(mem[src_a+index]) >>> shift)+128;
                                if (lookup_index<0) lookup_index=0;
                                if (lookup_index>255) lookup_index=255;
                                lookup_addr<=aux+lookup_index;
                            end
                            default: value=$signed(mem[src_a+index]);
                        endcase
                        if (opcode==3) begin
                            // Register the dependent lookup address: it breaks
                            // the source-read -> LUT-read path and prevents a
                            // resource-sharing feedback loop in synthesis.
                            memory_reads<=memory_reads+1; phase<=LUT_READ;
                        end else begin
                            mem[dst+index]<=clamped(value); memory_writes<=memory_writes+1;
                            memory_reads<=memory_reads+((opcode==1 || opcode==2)?2:1);
                            if (opcode==2) useful_macs<=useful_macs+1;
                            if (value>32767 || value< -32768) saturation_count<=saturation_count+1;
                            if (index+1>=length) begin pc<=pc+8; phase<=FETCH; end
                            else index<=index+1;
                        end
                    end
                    LUT_READ: begin
                        // All memory writes are validated/clamped to INT16.
                        mem[dst+index]<=mem[lookup_addr]; memory_reads<=memory_reads+1;
                        memory_writes<=memory_writes+1;
                        if (index+1>=length) begin pc<=pc+8; phase<=FETCH; end
                        else begin index<=index+1; phase<=VECTOR; end
                    end
                    RMS_SUM: begin
                        temporary=accumulator+$signed(mem[src_a+index])*$signed(mem[src_a+index]);
                        accumulator<=temporary; memory_reads<=memory_reads+1;
                        useful_macs<=useful_macs+1;
                        if (index+1>=length) begin
                            root<=isqrt($unsigned(temporary)/length+({32'd0,aux}<<14));
                            if (isqrt($unsigned(temporary)/length+({32'd0,aux}<<14))==0) root<=1;
                            index<=0; phase<=RMS_WRITE;
                        end else index<=index+1;
                    end
                    default: begin config_error<=1; busy<=0; phase<=IDLE; done<=1; end
                endcase
            end
        end
    end
endmodule
