"""Durable, append-only actual ISA validation; suitable for a user systemd job."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from malleable.llm.models import digest,inspect
from malleable.llm.quality import gate

RELEASE=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'

def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)

def source():
    files=('malleable/llm/quality.py','malleable/llm/candidates.py','malleable/llm/cli.py',
        'tools/run_candidate_full_validation.py')
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'files':{p:digest(ROOT/p) for p in files}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['overnight'],required=True)
    p.add_argument('--authorization',required=True)
    p.add_argument('--model',type=Path,default=ROOT/'build/models/Qwen3-0.6B')
    p.add_argument('--suite',type=Path,default=ROOT/'build/models/quality/qwen3-frozen-v2.json')
    p.add_argument('--candidate-case',type=Path,default=RELEASE/'quality-recovery-pilot-01/int8/case-00')
    a=p.parse_args(); os.chdir(ROOT); root=a.root.resolve()
    from malleable.llm.continuation import read_control
    control=read_control()
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=source()
        if frozen['status']: raise ValueError('commit clean source before validation')
        if root.exists(): raise ValueError('refusing to overwrite a validation attempt')
        root.mkdir(parents=True); shutil.copyfile(__file__,root/'runner.py')
        started=time.time(); wall=time.monotonic()
        command=[sys.executable,'-u','-m','malleable.llm.cli','quality-candidate-validate',
            '--model',str(a.model.resolve()),'--suite',str(a.suite.resolve()),
            '--candidate-case',str(a.candidate_case.resolve()),'--context','128',
            '--personality','balanced','--wformat','int8','--split','validation',
            '--max-host-gib','16','--store',str(root/'store')]
        write(root/'job.json',{'schema_version':1,'source':frozen,'pid':os.getpid(),
            'started_utc':datetime.now(timezone.utc).isoformat(),'started_unix':started,
            'dispatch_deadline_unix':control['dispatch_deadline_unix'],'mode':a.mode,'authorization':a.authorization,
            'command':command,'suite_sha256':digest(a.suite),'release_evidence':False})
        print(json.dumps({'status':'running','root':str(root),'pid':os.getpid()}),flush=True)
        result=None
        try:
            initial=inspect(a.model,128)
            with (root/'worker.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                write(root/'child.json',{'pid':process.pid})
                for line in process.stdout:
                    log.write(line); log.flush()
                    try: event=json.loads(line)
                    except ValueError: continue
                    if event.get('kind')=='result': result=event['payload']
                    elif event.get('kind') in ('phase','quality-progress','error'): print(line.strip(),flush=True)
                code=process.wait()
            if code or result is None: raise RuntimeError(f'validation worker exit {code}; see worker.log')
            write(root/'result.json',result)
            current=inspect(a.model,128)
            if source()!=frozen or digest(a.suite)!=json.loads((root/'job.json').read_text())['suite_sha256']:
                raise ValueError('source or frozen suite changed during validation')
            if any(initial[k]!=current[k] for k in ('base_model_id','tokenizer_id','weight_files','tokenizer_files')):
                raise ValueError('original checkpoint/tokenizer changed during validation')
            if (result.get('split')!='validation' or not result.get('suite_frozen') or result.get('target_count',0)<1024
                or result.get('samples')!=result['target_count'] or not result.get('derived_candidate')
                or result.get('context')!=128 or result.get('personality')!='balanced'
                or result.get('wformat')!='int8' or result.get('head_format')!='int8'):
                raise ValueError('validation evidence/configuration mismatch')
            checks=gate(result['float_nll'],result['candidate_nll'],result['agreement'])
            summary={'status':'completed','source':frozen,'agreement':result['agreement'],
                'matches':round(result['agreement']*result['target_count']),'target_count':result['target_count'],
                'float_nll':result['float_nll'],'candidate_nll':result['candidate_nll'],
                'nll_degradation_percent':100*checks['nll_degradation'],'validation_passed':checks['passed'],
                'variant_id':result['variant_id'],'configuration_id':result['configuration_id'],
                'derived_candidate':result['derived_candidate'],'record_id':result['record_id'],
                'result_sha256':digest(root/'result.json'),'worker_exit_code':code,
                'elapsed_seconds':time.monotonic()-wall,'selectable':False,'release_evidence':False,
                'next':('integrate generation/evidence identities and freeze before held-out' if checks['passed']
                    else 'preserve failure; execute the authorized ten precision diagnostics')}
            write(root/'summary.json',summary); print(json.dumps(summary),flush=True)
        except BaseException:
            write(root/'failure.json',{'status':'failed','source':source(),'traceback':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-wall}); raise

if __name__=='__main__': main()
