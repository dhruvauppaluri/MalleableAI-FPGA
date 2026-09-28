"""Icarus backend. Generated benches are disposable build artifacts."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from .records import identity
from .model import reference

ROOT = Path(__file__).resolve().parents[1]
RTL = ['int8_dot_product.sv', 'int8_tiled_accumulator.sv',
       'int8_postprocess.sv', 'malleable_accelerator_top.sv']


def writes(model):
    output = [(0, 0, len(model.layers))]
    wb = pb = 0
    for i, layer in enumerate(model.layers):
        n, k = len(layer['weights']), len(layer['weights'][0])
        output += [(0, 1+8*i+j, value) for j, value in enumerate([k, n, wb, pb, int(layer['relu'])])]
        output += [(2, wb+j*k+t, value) for j, row in enumerate(layer['weights'])
                   for t, value in enumerate(row)]
        for kind, key in [(3, 'biases'), (4, 'multipliers'), (5, 'shifts')]:
            output += [(kind, pb+j, value) for j, value in enumerate(layer[key])]
        wb += n*k
        pb += n
    return output


class RTLBackend:
    def __init__(self, build_dir=None):
        self.build_dir = Path(build_dir or ROOT / 'build' / 'experiments')
        self.build_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model, inputs, config, models=None):
        """Each input is a request; reuse actually avoids repeated model writes."""
        if not inputs:
            raise ValueError('Empty request stream')
        models = models or [model]*len(inputs)
        if len(models) != len(inputs):
            raise ValueError('Model/request count mismatch')
        for item,values in zip(models,inputs):
            if reference(item, values, config.active_lanes)[2]:
                raise ValueError('Reference arithmetic overflow')
        commands = []
        def emit(entries):
            commands.extend(f"write_cfg(3'd{k},16'd{a},32'h{v & 0xffffffff:08x});"
                            for k, a, v in entries)
        resident = None
        for request_id, (item, values) in enumerate(zip(models,inputs)):
            if item.model_id != resident or not config.reuse:
                emit(writes(item))
            resident = item.model_id
            emit([(6, 0, config.active_lanes)])
            emit([(1, i, v) for i, v in enumerate(values)])
            commands.append('start_job();')
            commands.append(f'$display("C {request_id} %0d %0d %0d %0d %0d %0d %0d", '
                            'cycles, tiles, macs, compute, overhead, writes_count, reads_count);')
            for i in range(len(item.layers[-1]['weights'])):
                commands.append(f'read_result({request_id},{i});')
        body = '\n'.join(commands)
        source = '''`timescale 1ns/1ps
module host_tb;
reg clk=0; always #5 clk=~clk;
reg reset=1, cv=0, start=0, re=0;
reg [2:0] kind=0; reg [15:0] addr=0, ra=0; reg [31:0] data=0;
wire ready,busy,done,overflow,error,rv; wire signed [7:0] rd;
wire [15:0] count; wire [63:0] cycles,tiles,macs,compute,overhead,writes_count,reads_count;
malleable_accelerator_top #(.LANES(LANE_COUNT)) dut(
 .clk(clk),.reset(reset),.cfg_valid(cv),.cfg_ready(ready),.cfg_kind(kind),
 .cfg_addr(addr),.cfg_data(data),.start(start),.busy(busy),.done(done),
 .overflow_error(overflow),.config_error(error),.result_read_en(re),
 .result_read_addr(ra),.result_read_data(rd),.result_read_valid(rv),.result_count(count),
 .cycles(cycles),.tiles(tiles),.useful_macs(macs),.compute_cycles(compute),
 .controller_cycles(overhead),.configuration_writes(writes_count),.result_reads(reads_count));
task write_cfg(input [2:0] k,input [15:0] a,input [31:0] d);
begin @(negedge clk); if(!ready) $fatal(1,"not ready");
cv=1;kind=k;addr=a;data=d; @(negedge clk);cv=0; end endtask
task start_job;
integer timeout;
begin @(negedge clk);start=1;@(negedge clk);start=0;timeout=0;
while(!done && timeout<1000000) begin @(negedge clk);timeout=timeout+1; end
if(!done || overflow || error) $fatal(1,"job failed"); end endtask
task read_result(input integer request_id,input integer index);
begin @(negedge clk);re=1;ra=index;@(posedge clk);#1;
if(!rv) $fatal(1,"invalid read");
$display("O %0d %0d %0d",request_id,index,rd);
@(negedge clk);re=0;end endtask
initial begin repeat(3) @(negedge clk); reset=0;
COMMANDS
$finish;end
initial begin #1000000000; $fatal(1,"watchdog");end
endmodule
'''.replace('LANE_COUNT', str(config.lanes)).replace('COMMANDS', body)
        tool = subprocess.run(['iverilog','-V'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
        build_hash = identity(dict(source=source, tool=tool, rtl=[(ROOT/'rtl'/p).read_text() for p in RTL]))
        directory = self.build_dir / build_hash
        directory.mkdir(exist_ok=True)
        bench, executable = directory/'host_tb.sv', directory/'run.out'
        bench.write_text(source)
        before = time.perf_counter()
        if not executable.exists():
            subprocess.run(['iverilog', '-g2012', '-s', 'host_tb', '-o', str(executable),
                        *[str(ROOT/'rtl'/p) for p in RTL], str(bench)], check=True,
                       capture_output=True, text=True, timeout=120)
        build_seconds = time.perf_counter()-before
        before = time.perf_counter()
        run = subprocess.run(['vvp', str(executable)], check=True, capture_output=True,
                             text=True, timeout=120)
        simulation_seconds = time.perf_counter()-before
        (directory/'trace.txt').write_text(run.stdout)
        observed = [[] for _ in inputs]
        counters = []
        for line in run.stdout.splitlines():
            fields = line.split()
            if fields and fields[0] == 'O':
                observed[int(fields[1])].append(int(fields[3]))
            if fields and fields[0] == 'C':
                counters.append(dict(zip(['cycles', 'tiles', 'macs', 'compute', 'overhead',
                                          'configuration_writes', 'result_reads'], map(int, fields[2:]))))
        expected = [reference(item, x, config.active_lanes)[0] for item,x in zip(models,inputs)]
        if observed != expected or len(counters) != len(inputs):
            raise ValueError(f'RTL/reference mismatch: {observed} != {expected}')
        return dict(counters=counters, outputs=observed, build_hash=build_hash,
                    trace=run.stdout, build_seconds=build_seconds,
                    simulation_seconds=simulation_seconds,
                    tool=tool)
