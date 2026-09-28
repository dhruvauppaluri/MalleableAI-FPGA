"""Automatic, cached Quartus builds. Compilation never programs a board.

Run on a host with Quartus installed (normally Windows/Linux). A missing tool
leaves a reproducible pending request; simulator runtime is not programming cost.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Personality:
    lanes: int = 4
    memory_words: int = 16384
    program_words: int = 8192
    clock_mhz: int = 100
    device: str = '5CGXFC5C6F27C7'
    family: str = 'Cyclone V'
    schema_version: int = 1

    def __post_init__(self):
        if type(self.lanes) is not int or self.lanes not in (1,2,4,8,16):
            raise ValueError('unsupported personality lanes')
        if type(self.memory_words) is not int or type(self.program_words) is not int or not 512<=self.memory_words<=1048576 or not 512<=self.program_words<=262144:
            raise ValueError('unsupported memory/program capacity')
        if not 25<=self.clock_mhz<=250 or not re.fullmatch(r'[A-Za-z0-9]+',self.device) or self.family!='Cyclone V':
            raise ValueError('invalid device/clock/family')


def tool_version(executable):
    path=shutil.which(executable)
    if not path:
        return None
    result=subprocess.run([path,'--version'],capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError('Quartus version check failed: '+result.stderr)
    return (result.stdout+result.stderr).strip()


def generate(personality,build_root='build/quartus',executable='quartus_sh'):
    version=tool_version(executable)
    source=(ROOT/'rtl/ssm_operator_engine.sv').read_bytes()
    identity=dict(personality=asdict(personality),source_sha256=hashlib.sha256(source).hexdigest(),
                  tool_version=version,ip_dependencies=[],generator_version=2,
                  constraints={'virtual_pins':True,'clock_period_ns':1000/personality.clock_mhz})
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=Path(build_root).resolve()/key
    directory.mkdir(parents=True,exist_ok=True)
    # Snapshot all inputs: subsequent RTL edits cannot mutate a queued build.
    (directory/'ssm_operator_engine.sv').write_bytes(source)
    (directory/'personality.qpf').write_text('PROJECT_REVISION = "personality"\n')
    ports=['reset','cfg_valid','cfg_ready','cfg_kind[*]','cfg_addr[*]','cfg_data[*]','start','busy','done',
           'config_error','cycles[*]','useful_macs[*]','memory_reads[*]','memory_writes[*]',
           'saturation_count[*]','configuration_writes[*]','read_en','read_addr[*]','read_data[*]','read_valid']
    qsf=[f'set_global_assignment -name FAMILY "{personality.family}"',
         f'set_global_assignment -name DEVICE {personality.device}',
         'set_global_assignment -name TOP_LEVEL_ENTITY ssm_operator_engine',
         'set_global_assignment -name SYSTEMVERILOG_FILE ssm_operator_engine.sv',
         'set_global_assignment -name SDC_FILE personality.sdc',
         'set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files',
         f'set_parameter -name LANES {personality.lanes}',
         f'set_parameter -name MEM_WORDS {personality.memory_words}',
         f'set_parameter -name PROGRAM_WORDS {personality.program_words}']
    for port in ['clk']+ports:
        qsf.append(f'set_instance_assignment -name VIRTUAL_PIN ON -to {{{port}}}')
    (directory/'personality.qsf').write_text('\n'.join(qsf)+'\n')
    (directory/'personality.sdc').write_text(f'create_clock -name clk -period {1000/personality.clock_mhz:.6f} [get_ports {{clk}}]\n'
                                          'derive_clock_uncertainty\n')
    (directory/'compile.tcl').write_text('package require ::quartus::flow\nproject_open personality\n'
                                       'execute_flow -compile\nproject_close\n')
    (directory/'timing.tcl').write_text('project_open personality\ncreate_timing_netlist\nread_sdc\nupdate_timing_netlist\n'
        'set output [open output_files/verified-timing.summary w]\n'
        'foreach_in_collection path [get_timing_paths -setup -npaths 1] {\n'
        '  puts $output "Worst-case Setup Slack : [get_path_info -slack $path]"\n}\n'
        'foreach_in_collection path [get_timing_paths -hold -npaths 1] {\n'
        '  puts $output "Worst-case Hold Slack : [get_path_info -slack $path]"\n}\n'
        'close $output\ndelete_timing_netlist\nproject_close\n')
    manifest=dict(schema_version=1,build_hash=key,identity=identity,status='ready' if version else 'pending-tool',
                  board_programming=False,physical_pin_assignment=False,requested_at=time.time())
    manifest_path=directory/'request.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest,indent=2))
    return directory,json.loads(manifest_path.read_text())


def parse_reports(directory,target_mhz):
    directory=Path(directory)
    files=list((directory/'output_files').glob('*.rpt'))+list((directory/'output_files').glob('*.summary'))
    text='\n'.join(f.read_text(errors='replace') for f in files)
    def number(pattern):
        match=re.search(pattern,text,re.I|re.M)
        return float(match.group(1).replace(',','')) if match else None
    resources={name:number(pattern) for name,pattern in {
        'logic_elements':r'Total logic elements\s*[:;]\s*([\d,]+)',
        'alm':r'(?:Logic utilization \(in ALMs\)|Total ALMs)\s*[:;]\s*([\d,]+)',
        'dsp_blocks':r'(?:Total DSP Blocks|DSP block 18-bit elements|Total DSP block 18-bit elements)\s*[:;]\s*([\d,]+)',
        'memory_bits':r'Total (?:block )?memory bits\s*[:;]\s*([\d,]+)',
        'ram_blocks':r'(?:Total RAM Blocks|M10K blocks)\s*[:;]\s*([\d,]+)'}.items()}
    # Timing acceptance must have affirmative setup AND hold slack evidence.
    def worst(pattern):
        values=[float(v) for v in re.findall(pattern,text,re.I|re.M)]
        return min(values) if values else None
    setup=worst(r'(?:Worst-case Setup Slack|Setup slack)\s*[:;]\s*(-?[\d.]+)')
    hold=worst(r'(?:Worst-case Hold Slack|Hold slack)\s*[:;]\s*(-?[\d.]+)')
    fmax=number(r'(?:Restricted Fmax|Fmax)\s*[:;]\s*([\d.]+)\s*MHz')
    fit=bool(re.search(r'Fitter Status\s*[:;]\s*Successful',text,re.I))
    power=number(r'Total Thermal Power Dissipation\s*[:;]\s*([\d.]+)\s*mW')
    return dict(fit_passed=fit,timing_passed=bool(setup is not None and hold is not None and setup>=0 and hold>=0),
                setup_slack_ns=setup,hold_slack_ns=hold,fmax_mhz=fmax,target_mhz=target_mhz,
                timing_scope='reported-corners-register-paths; not physical-interface signoff',
                resources={'source':'quartus-estimated','values':resources},
                power={'source':'unavailable','watts':None,'unvalidated_report_watts':None if power is None else power/1000},
                report_files=[str(f) for f in files],physical_measurements=False)


def compile_personality(personality,build_root='build/quartus',executable='quartus_sh',timeout=3600):
    directory,request=generate(personality,build_root,executable)
    result_path=directory/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result.get('accepted'):
            return dict(result,cache_hit=True)
    if request['identity']['tool_version'] is None:
        return dict(request,reason='Quartus is not installed on this host; transfer the request to a Quartus worker',directory=str(directory),accepted=False)
    log_path=directory/'compile.log'
    start=time.perf_counter()
    with log_path.open('w') as log:
        try:
            process=subprocess.run([executable,'--flow','compile','personality'],cwd=directory,
                                   stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            exit_code=process.returncode
            failure=None
            if exit_code==0:
                shell=Path(shutil.which(executable)).resolve()
                sta=shell.with_name('quartus_sta'+shell.suffix)
                if sta.exists():
                    timing=subprocess.run([str(sta),'-t','timing.tcl'],cwd=directory,
                                          stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                    if timing.returncode: failure='timing extraction failed'
        except subprocess.TimeoutExpired:
            exit_code=None; failure='compile timeout'
    report=parse_reports(directory,personality.clock_mhz)
    # Virtual-pin builds are characterization artifacts, NEVER deployable images.
    result=dict(schema_version=1,build_hash=request['build_hash'],status='completed' if exit_code==0 else 'failed',
                accepted=bool(exit_code==0 and failure is None and report['fit_passed'] and report['timing_passed']),
                returncode=exit_code,failure=failure,elapsed_build_seconds=time.perf_counter()-start,
                directory=str(directory),log=str(log_path),report=report,cache_hit=False,
                deployable=False,board_programming=False)
    result_path.write_text(json.dumps(result,indent=2))
    return result


def automatic_build(personality,build_root='build/quartus',executable='quartus_sh',clock_floor=75):
    """Bounded, deterministic retry only after a completed timing failure."""
    results=[]
    clocks=[personality.clock_mhz]
    if personality.clock_mhz>clock_floor:
        clocks += list(range(personality.clock_mhz-5,clock_floor-1,-5))
    for clock in clocks:
        result=compile_personality(Personality(**dict(asdict(personality),clock_mhz=clock)),build_root,executable)
        results.append(result)
        if result['accepted'] or result['status'] in ('pending-tool','failed'):
            break
        report=result.get('report',{})
        if not report.get('fit_passed') or report.get('setup_slack_ns') is None or report.get('hold_slack_ns') is None:
            break
    return {'accepted':any(r['accepted'] for r in results),'attempts':results,'board_programming':False}
