`timescale 1ns/1ps
module ssm_operator_engine_tb;
reg clk=0; always #5 clk=~clk;
reg reset=1,cfg_valid=0,start=0,read_en=0;
reg [1:0] cfg_kind=0; reg [31:0] cfg_addr=0,read_addr=0; reg signed [31:0] cfg_data=0;
wire cfg_ready,busy,done,config_error,read_valid; wire signed [31:0] read_data;
wire [63:0] cycles,useful_macs,memory_reads,memory_writes,saturation_count,configuration_writes;
ssm_operator_engine #(.MEM_WORDS(512),.PROGRAM_WORDS(512)) dut(.*);
integer i;
task cfg(input [1:0] k,input [31:0] a,input [31:0] d);
begin @(negedge clk);cfg_valid=1;cfg_kind=k;cfg_addr=a;cfg_data=d;
@(negedge clk);cfg_valid=0;end endtask
task rst;
begin @(negedge clk);reset=1;@(negedge clk);reset=0;
if(busy||done||config_error||cycles||useful_macs||configuration_writes||read_valid) $fatal(1,"reset counters");end endtask
task launch;
begin @(negedge clk);start=1;@(negedge clk);start=0;end endtask
task finish_job;
integer watchdog;
begin watchdog=0;while(!done&&watchdog<1000) begin @(negedge clk);watchdog=watchdog+1;end
if(!done) $fatal(1,"watchdog");end endtask
initial begin
rst();
// DOT 7 signed extrema, partial last tile, saturation and immutable idle counters.
cfg(0,0,0);
for(i=0;i<7;i=i+1) begin cfg(0,10+i,32767);cfg(0,20+i,32767);end
cfg(1,0,0);cfg(1,1,30);cfg(1,2,10);cfg(1,3,20);cfg(1,4,7);cfg(1,5,0);cfg(1,6,14);cfg(1,7,0);
cfg(1,8,255);cfg(2,0,3);launch();finish_job();
if(config_error||cycles!=5||useful_macs!=7||saturation_count!=1||memory_reads!=15||memory_writes!=1) $fatal(1,"partial tile counters");
repeat(5) @(negedge clk);
if(cycles!=5) $fatal(1,"idle counter changed");
read_addr=30;read_en=1;@(negedge clk);read_en=0;
if(!read_valid||read_data!=32767) $fatal(1,"saturation read");
@(negedge clk);if(read_valid) $fatal(1,"read valid pulse");
// Repeated execution resets job counters, not accumulated config writes.
launch();finish_job();if(cycles!=5||saturation_count!=1) $fatal(1,"repeat counters");
// Busy commands are rejected but current execution drains normally.
launch();cfg(2,0,1);finish_job();if(!config_error) $fatal(1,"busy command accepted");
rst();cfg(2,0,0);if(!config_error) $fatal(1,"zero lanes accepted");
rst();cfg(2,0,5);if(!config_error) $fatal(1,"unsupported lanes accepted");
rst();cfg(1,0,0);cfg(1,1,30);cfg(1,2,32'hfffffff0);cfg(1,3,20);cfg(1,4,7);cfg(1,5,0);cfg(1,6,14);cfg(1,7,0);
launch();finish_job();if(!config_error) $fatal(1,"invalid descriptor accepted");
rst();cfg(1,2,10);cfg(1,1,10);launch();finish_job();if(!config_error) $fatal(1,"overlap accepted");
// Reset mid execution terminates the job and restores the runtime lane default.
rst();cfg(1,1,30);launch();rst();repeat(10) @(negedge clk);
if(busy||done||cycles||dut.active_lanes!=4) $fatal(1,"reset during execution");
$display("PASS: SSM operator controls, partial tiles, saturation, reset and counters");$finish;
end
endmodule
