"""Complete-token RTL replay with independent integer-reference checking."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
from .compiler import compile_token
from .numeric import execute
from .tokenizer import Codec

ROOT=Path(__file__).resolve().parents[2]


def select(logits,rng,top_k=1,temperature=1.0):
    if not 1 <= top_k <= len(logits) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid sampling parameters')
    choices=sorted(range(len(logits)),key=lambda i:(-logits[i],i))[:top_k]
    if top_k==1:
        return choices[0]
    maximum=logits[choices[0]]
    weights=[math.exp((logits[i]-maximum)/16384/temperature) for i in choices]
    return rng.choices(choices,weights=weights,k=1)[0]


def reference_trace(compiled,prompt,max_new=8,seed=0,top_k=1,temperature=1.0):
    codec=Codec(compiled['manifest'].get('tokenizer'))
    tokens=codec.encode(prompt) or [codec.BOS]
    if len(tokens)>2048 or not 0 <= max_new <= 256:
        raise ValueError('prompt/output length outside supported limits')
    if any(t>=compiled['manifest']['config']['vocab_size'] for t in tokens):
        raise ValueError('tokenizer exceeds model vocabulary')
    initial_count=len(tokens)
    memory=list(compiled['memory']); records=[]; generated=[]; rng=random.Random(seed)
    for index in range(initial_count+max(0,max_new-1)):
        token=tokens[index]
        width=compiled['manifest']['config']['width']
        embedding=compiled['embedding_addr']+token*width
        input_values=memory[embedding:embedding+width]
        memory[compiled['input_addr']:compiled['input_addr']+width]=input_values
        memory,saturated=execute(memory,compiled['program'])
        addr=compiled['logits_addr']; vocab=compiled['manifest']['config']['vocab_size']
        logits=memory[addr:addr+vocab]
        states=[memory[a:a+n] for a,n in compiled['state_addresses']]
        records.append(dict(token=token,input=input_values,logits=logits,states=states,saturation_count=saturated))
        if index>=initial_count-1 and len(generated)<max_new:
            next_token=select(logits,rng,top_k,temperature)
            generated.append(next_token); tokens.append(next_token)
            if next_token==codec.EOS:
                break
    # The final generated token need not itself be decoded to obtain its logits.
    return records,generated,codec.decode(generated)


def run_rtl(compiled,records,lanes=4,active_lanes=None,timeout=180):
    active_lanes=lanes if active_lanes is None else active_lanes
    if type(lanes) is not int or lanes not in (1,2,4,8,16) or type(active_lanes) is not int or not 1<=active_lanes<=lanes:
        raise ValueError('unsupported lane configuration')
    if not shutil.which('iverilog') or not shutil.which('vvp'):
        raise RuntimeError('install Icarus Verilog (iverilog and vvp) for RTL simulation')
    memory_words=max(512,1<<(len(compiled['memory'])-1).bit_length())
    program_words=max(512,1<<(len(compiled['program'])*8-1).bit_length())
    if memory_words>1048576 or program_words>262144:
        raise ValueError('artifact exceeds local simulation capacity; use software backend')
    rtl=ROOT/'rtl/ssm_operator_engine.sv'
    key=hashlib.sha256(rtl.read_bytes()+json.dumps([lanes,memory_words,program_words]).encode()).hexdigest()
    directory=ROOT/'build/ssm'/key
    directory.mkdir(parents=True,exist_ok=True)
    # Per-run directory prevents concurrent IDE jobs from overwriting vectors.
    import tempfile
    run_directory=Path(tempfile.mkdtemp(prefix='run-',dir=directory))
    def hexfile(name,values):
        path=run_directory/name
        path.write_text('\n'.join(f'{v & 0xffffffff:08x}' for v in values)+'\n')
        return path
    memfile=hexfile('memory.hex',compiled['memory'])
    progfile=hexfile('program.hex',[word for instruction in compiled['program'] for word in instruction])
    inputfile=hexfile('inputs.hex',[v for record in records for v in record['input']])
    cfg=compiled['manifest']['config']; w=cfg['width']; vocab=cfg['vocab_size']
    reads='\n'.join(f'for(j=0;j<{count};j=j+1) begin rd({addr}+j); $display("S %0d %0d %0d %0d",t,{s},j,$signed(read_data)); end'
                    for s,(addr,count) in enumerate(compiled['state_addresses']))
    bench=f'''module tb;
reg clk=0; always #5 clk=~clk;
reg reset=1,cfg_valid=0,start=0,read_en=0;
reg [1:0] cfg_kind=0; reg [31:0] cfg_addr=0,read_addr=0; reg signed [31:0] cfg_data=0;
wire cfg_ready,busy,done,config_error,read_valid; wire signed [31:0] read_data;
wire [63:0] cycles,useful_macs,memory_reads,memory_writes,saturation_count,configuration_writes;
ssm_operator_engine #(.LANES({lanes}),.MEM_WORDS({memory_words}),.PROGRAM_WORDS({program_words})) dut(.*);
reg [31:0] memory_image[0:{len(compiled['memory'])-1}];
reg [31:0] program_image[0:{len(compiled['program'])*8-1}];
reg [31:0] inputs[0:{len(records)*w-1}];
integer i,j,t,watchdog;
task cfg(input [1:0] k,input [31:0] a,input [31:0] d);
begin @(negedge clk); cfg_valid=1;cfg_kind=k;cfg_addr=a;cfg_data=d;
@(negedge clk);cfg_valid=0;end endtask
task rd(input [31:0] a);
begin @(negedge clk);read_en=1;read_addr=a;@(negedge clk);read_en=0;
if(!read_valid) $fatal(1,"missing read valid");end endtask
initial begin
$readmemh("{memfile.as_posix()}",memory_image);
$readmemh("{progfile.as_posix()}",program_image);
$readmemh("{inputfile.as_posix()}",inputs);
repeat(3) @(negedge clk);reset=0;
for(i=0;i<{len(compiled['memory'])};i=i+1) cfg(0,i,memory_image[i]);
for(i=0;i<{len(compiled['program'])*8};i=i+1) cfg(1,i,program_image[i]);
cfg(2,0,{active_lanes});
for(t=0;t<{len(records)};t=t+1) begin
for(i=0;i<{w};i=i+1) cfg(0,{compiled['input_addr']}+i,inputs[t*{w}+i]);
@(negedge clk);start=1;@(negedge clk);start=0;watchdog=0;
while(!done && watchdog<10000000) begin @(negedge clk);watchdog=watchdog+1;end
if(!done || config_error) $fatal(1,"execution failed token %0d pc %0d",t,dut.pc);
$display("C %0d %0d %0d %0d %0d %0d %0d",t,cycles,useful_macs,memory_reads,memory_writes,saturation_count,configuration_writes);
for(j=0;j<{vocab};j=j+1) begin rd({compiled['logits_addr']}+j);$display("L %0d %0d %0d",t,j,$signed(read_data));end
{reads}
end
$finish;end
endmodule'''
    bench_path=run_directory/'tb.sv'; bench_path.write_text(bench)
    binary=run_directory/'tb.out'; start_time=time.perf_counter()
    version=subprocess.run(['iverilog','-V'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    build=subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),str(rtl),str(bench_path)],capture_output=True,text=True,timeout=timeout)
    if build.returncode:
        raise RuntimeError(build.stderr)
    build_seconds=time.perf_counter()-start_time
    start_time=time.perf_counter()
    result=subprocess.run(['vvp',str(binary)],capture_output=True,text=True,timeout=timeout)
    (run_directory/'trace.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    observed=[dict(logits=[None]*vocab,states=[[None]*count for _,count in compiled['state_addresses']]) for _ in records]
    counters=[]
    for line in result.stdout.splitlines():
        parts=line.split()
        if parts and parts[0]=='L':
            _,t,j,value=parts; observed[int(t)]['logits'][int(j)]=int(value)
        elif parts and parts[0]=='S':
            _,t,s,j,value=parts; observed[int(t)]['states'][int(s)][int(j)]=int(value)
        elif parts and parts[0]=='C':
            numbers=list(map(int,parts[1:])); t=numbers.pop(0)
            row=dict(zip(['cycles','useful_macs','memory_reads','memory_writes','saturation_count','configuration_writes'],numbers),token_index=t)
            counters.append(row)
    if len(counters)!=len(records):
        raise RuntimeError('missing RTL counters')
    for index,(expected,actual) in enumerate(zip(records,observed)):
        if expected['logits']!=actual['logits'] or expected['states']!=actual['states'] or expected['saturation_count']!=counters[index]['saturation_count']:
            raise RuntimeError(f'RTL/reference mismatch at token {index}')
    return dict(bit_exact=True,counters=counters,build_hash=key,tool_version=version,
                build_seconds=build_seconds,host_simulation_seconds=time.perf_counter()-start_time,trace_path=str(run_directory/'trace.txt'))


def generate(path,prompt,max_new=8,seed=0,top_k=1,lanes=4,active_lanes=None,backend='rtl',clock_hz=None,store=None):
    if clock_hz is not None and (not math.isfinite(clock_hz) or clock_hz<=0):
        raise ValueError('assumed clock must be finite and positive')
    compiled=compile_token(path)
    records,tokens,text=reference_trace(compiled,prompt,max_new,seed,top_k)
    evidence=run_rtl(compiled,records,lanes,active_lanes) if backend=='rtl' else {'bit_exact':None,'counters':[]}
    if backend not in ('rtl','integer'):
        raise ValueError('backend must be rtl or integer')
    cycles=sum(c['cycles'] for c in evidence['counters'])
    prompt_count=len(Codec(compiled['manifest'].get('tokenizer')).encode(prompt)) or 1
    prefill_cycles=sum(c['cycles'] for c in evidence['counters'][:prompt_count])
    decode_cycles=sum(c['cycles'] for c in evidence['counters'][prompt_count:])
    decode_steps=max(0,len(tokens)-1)
    metrics={'execution_cycles':{'value':cycles if backend=='rtl' else None,'source':'measured-rtl' if backend=='rtl' else 'unavailable'},
             'prefill_cycles':{'value':prefill_cycles if backend=='rtl' else None,'source':'measured-rtl' if backend=='rtl' else 'unavailable'},
             'decode_cycles':{'value':decode_cycles if backend=='rtl' else None,'source':'measured-rtl' if backend=='rtl' else 'unavailable'},
             'tokens_per_second':{'value':decode_steps*clock_hz/decode_cycles if clock_hz and decode_cycles else None,
                                  'source':'estimated-assumed-clock' if clock_hz else 'unavailable'},
             'physical_power':{'value':None,'source':'unavailable'},'physical_latency':{'value':None,'source':'unavailable'}}
    result=dict(schema_version=1,model_id=compiled['model_id'],backend=backend,seed=seed,prompt=prompt,
                generated_tokens=tokens,text=text,model_trained=compiled['manifest'].get('trained',False),
                config={'lanes':lanes,'active_lanes':active_lanes or lanes},metrics=metrics,evidence=evidence,
                reference_saturations=sum(r['saturation_count'] for r in records),
                quality_gate='not-evaluated',assumed_clock_hz=clock_hz)
    if store:
        result['record_id']=store.save('ssm-experiment',result)
    return result
