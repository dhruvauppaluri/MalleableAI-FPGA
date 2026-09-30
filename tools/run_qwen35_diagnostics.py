"""Durable Qwen3.5 alignment/attribution jobs using one immutable dispatch window."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from malleable.llm.models import digest,inspect
from malleable.llm.precision import GROUPS,make_panel
from malleable.records import identity
RELEASE=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'
CONTINUATION=RELEASE/'continuation-20260930'
AUTHORIZATION='2026-09-30 user approved one new eight-hour Overnight window and targeted Qwen3.5 recovery, INT8 matrices/head and unchanged suites/gates'

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())

def source():
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)}

def window():
    CONTINUATION.mkdir(parents=True,exist_ok=True);path=CONTINUATION/'control.json'
    if not path.exists():
        started=time.time()
        write(path,{'schema_version':1,'mode':'overnight','authorization':AUTHORIZATION,
            'started_unix':started,'dispatch_deadline_unix':started+8*3600,
            'started_utc':datetime.fromtimestamp(started,timezone.utc).isoformat()})
    control=json.loads(path.read_text())
    if (control.get('authorization')!=AUTHORIZATION or control.get('mode')!='overnight'
        or control['dispatch_deadline_unix']!=control['started_unix']+8*3600):
        raise ValueError('immutable continuation control mismatch')
    if time.time()>=control['dispatch_deadline_unix']:raise ValueError('dispatch deadline expired; no new heavy job authorized')
    return control

def attempt(name,command,panel,control,frozen,unit):
    # The same source and same persisted deadline are checked before EACH child.
    if time.time()>=control['dispatch_deadline_unix']:return False
    if source()!=frozen:raise ValueError('source changed before dispatch')
    root=CONTINUATION/name
    if root.exists():raise ValueError('attempt exists; inspect it, never overwrite or duplicate')
    root.mkdir();write(root/'panel.json',panel);shutil.copyfile(__file__,root/'runner.py')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith(('MemTotal:','MemAvailable:'))}
    write(root/'job.json',{'schema_version':1,'source':frozen,'command':command,'authorization':AUTHORIZATION,
        'pid':os.getpid(),'unit':unit,'started_unix':time.time(),'panel_id':identity(panel),
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'dispatch_deadline_unix':control['dispatch_deadline_unix'],'memory_preflight':memory,'release_evidence':False})
    started=time.monotonic();result=None
    try:
        with (root/'worker.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            write(root/'child.json',{'pid':process.pid})
            for line in process.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('kind')=='result':result=event['payload']
                elif event.get('kind') in ('phase','quality-progress','precision-progress','reference-progress','error'):
                    print(line.strip(),flush=True)
            code=process.wait()
        if code or result is None:raise RuntimeError(f'diagnostic child exit {code}; inspect worker.log')
        if source()!=frozen:raise ValueError('source changed during diagnostic')
        if digest(ROOT/'build/models/quality/qwen35-frozen-v2.json')!=panel['suite_file_hash']:
            raise ValueError('frozen suite changed during diagnostic')
        write(root/'result.json',result)
        write(root/'summary.json',{'status':'completed','result_sha256':digest(root/'result.json'),
            'record_id':result.get('record_id'),'agreement':result.get('agreement'),
            'float_reference_agreement':result.get('float_reference_agreement'),
            'elapsed_seconds':time.monotonic()-started,'source':frozen,'release_evidence':False})
        print(json.dumps({'attempt':str(root),'status':'completed','elapsed_seconds':time.monotonic()-started}),flush=True)
    except BaseException:
        write(root/'failure.json',{'status':'failed','source':source(),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started});raise
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['alignment','attribution'],required=True)
    parser.add_argument('--unit',required=True)
    parser.add_argument('--alignment-attempt',default='qwen35-alignment-01')
    args=parser.parse_args();os.chdir(ROOT)
    if not re.fullmatch(r'qwen35-alignment-[0-9]{2}',args.alignment_attempt):
        raise ValueError('alignment attempt must be a fresh numbered alignment directory')
    suite=ROOT/'build/models/quality/qwen35-frozen-v2.json'
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=source()
        if frozen['status']:raise ValueError('clean committed source required')
        info=inspect(ROOT/'build/models/Qwen3.5-0.8B',128)
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:'))
        required=(3*info['fp32_tensor_bytes']+3*info['personalities']['balanced']['config']['DRAM_BYTES']
            if args.phase=='alignment' else 4*info['fp32_tensor_bytes'])
        if not info['supported'] or required>min(16*1024**3,available):
            raise ValueError('Qwen3.5 supported-model/memory preflight failed before dispatch')
        common=[sys.executable,'-u','-m','malleable.llm.cli']
        settings=['--model',str(ROOT/'build/models/Qwen3.5-0.8B'),'--suite',str(suite),
            '--context','128','--personality','balanced','--wformat','int8','--split','validation',
            '--max-host-gib','16','--store',str(CONTINUATION/'diagnostic-store')]
        if args.phase=='alignment':
            panel=make_panel(suite,16,'prefix');control=window()
            attempt(args.alignment_attempt,common+['quality-diagnose',*settings,'--limit-targets','16'],
                panel,control,frozen,args.unit)
            return
        alignment=json.loads((CONTINUATION/args.alignment_attempt/'result.json').read_text())
        if alignment.get('float_reference_agreement')!=1 or alignment.get('conversion',{}).get('mismatches'):
            raise ValueError('resolve floating-reference/conversion alignment before quantization attribution')
        panel=make_panel(suite,128,'spread');control=window()
        panel_path=CONTINUATION/'panel-128.json'
        if not panel_path.exists():write(panel_path,panel)
        elif json.loads(panel_path.read_text())!=panel:raise ValueError('frozen panel changed')
        cases=[(),GROUPS,('transformer_weights',),('projection_inputs',),('key_cache',),('value_cache',),
            ('attention',),('head_weights',),('head_inputs',),('transformer_weights','projection_inputs')]
        for index,bypass in enumerate(cases):
            name=f'qwen35-attribution-{index:02d}-01';root=CONTINUATION/name
            if (root/'summary.json').exists():
                summary=json.loads((root/'summary.json').read_text())
                if digest(root/'result.json')!=summary['result_sha256']:raise ValueError('completed diagnostic hash mismatch')
                continue
            # A completed 128-target baseline is the pilot for remaining cases.
            if index==1:
                pilot=json.loads((CONTINUATION/'qwen35-attribution-00-01/summary.json').read_text())
                print(json.dumps({'pilot_seconds':pilot['elapsed_seconds'],
                    'remaining_nine_estimated_seconds':9*pilot['elapsed_seconds'],
                    'provenance':'estimated from completed 128-target baseline; bypass timings may differ'}),flush=True)
            policy={'schema_version':1,'quantized_groups':[g for g in GROUPS if g not in bypass],'group_size':128}
            policy_path=CONTINUATION/f'policy-{index:02d}.json'
            if not policy_path.exists():write(policy_path,policy)
            elif json.loads(policy_path.read_text())!=policy:raise ValueError('frozen policy changed')
            command=common+['quality-precision-diagnose',*settings,'--diagnostic-panel',str(panel_path),
                '--precision-policy',str(policy_path)]
            if not attempt(name,command,panel,control,frozen,args.unit):
                print(json.dumps({'status':'deadline-limited','next_index':index}),flush=True);return

if __name__=='__main__':main()
