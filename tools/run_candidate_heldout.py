"""Durable, exactly-once frozen held-out ISA evaluation of one frozen candidate."""
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
from malleable.llm import candidates as C
from malleable.llm.models import digest,inspect
from malleable.llm.quality import gate


def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def source():
    files=('malleable/llm/quality.py','malleable/llm/candidates.py','malleable/llm/cli.py',
        'tools/run_candidate_heldout.py')
    return dict(C.source_state(),files={p:digest(ROOT/p) for p in files})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True,help='directory created by freeze_candidate.py')
    p.add_argument('--deadline-unix',type=float,required=True,help='shared dispatch deadline from the first job.json')
    p.add_argument('--model',type=Path,default=ROOT/'build/models/Qwen3-0.6B')
    p.add_argument('--suite',type=Path,default=ROOT/'build/models/quality/qwen3-frozen-v2.json')
    p.add_argument('--candidate-case',type=Path,required=True)
    p.add_argument('--authorization',required=True)
    a=p.parse_args(); os.chdir(ROOT); root=a.root.resolve()
    from malleable.llm.continuation import read_control
    read_control(a.deadline_unix)
    if time.time()>=a.deadline_unix: raise ValueError('dispatch deadline has passed; no new job may start')
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=source()
        if frozen['status']: raise ValueError('commit clean source before held-out evaluation')
        if (root/'job.json').exists() or (root/'heldout-claim.json').exists():
            raise ValueError('refusing to repeat a held-out evaluation')
        shutil.copyfile(__file__,root/'runner.py')
        started=time.time(); wall=time.monotonic()
        command=[sys.executable,'-u','-m','malleable.llm.cli','quality-candidate-validate',
            '--model',str(a.model.resolve()),'--suite',str(a.suite.resolve()),
            '--candidate-case',str(a.candidate_case.resolve()),'--candidate-freeze',str(root/'freeze.json'),
            '--context','128','--personality','balanced','--wformat','int8','--split','held-out',
            '--max-host-gib','16','--store',str(root/'store')]
        write(root/'job.json',{'schema_version':1,'source':frozen,'pid':os.getpid(),
            'started_utc':datetime.now(timezone.utc).isoformat(),'dispatch_deadline_unix':a.deadline_unix,
            'authorization':a.authorization,'command':command,'freeze_sha256':digest(root/'freeze.json'),
            'suite_sha256':digest(a.suite),'release_evidence':False})
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
                    elif event.get('kind') in ('phase','quality-progress','error','failure'): print(line.strip(),flush=True)
                code=process.wait()
            if code or result is None: raise RuntimeError(f'held-out worker exit {code}; see worker.log')
            write(root/'result.json',result)
            current=inspect(a.model,128)
            if source()!=frozen or digest(a.suite)!=json.loads((root/'job.json').read_text())['suite_sha256']:
                raise ValueError('source or frozen suite changed during held-out evaluation')
            if any(initial[k]!=current[k] for k in ('base_model_id','tokenizer_id','weight_files','tokenizer_files')):
                raise ValueError('original checkpoint/tokenizer changed during held-out evaluation')
            freeze=json.loads((root/'freeze.json').read_text())
            if (result.get('split')!='held-out' or not result.get('suite_frozen') or result.get('target_count',0)<1024
                or result.get('samples')!=result['target_count'] or result.get('candidate_freeze_id')!=freeze['freeze_id']
                or result.get('derived_candidate')!=freeze['derived_candidate'] or result.get('variant_id')!=freeze['variant_id']
                or result.get('configuration_id')!=freeze['configuration']['configuration_id']):
                raise ValueError('held-out evidence/configuration mismatch')
            checks=gate(result['float_nll'],result['candidate_nll'],result['agreement'])
            summary={'status':'completed','source':frozen,'freeze_id':freeze['freeze_id'],
                'agreement':result['agreement'],'matches':round(result['agreement']*result['target_count']),
                'target_count':result['target_count'],'float_nll':result['float_nll'],
                'candidate_nll':result['candidate_nll'],'nll_degradation_percent':100*checks['nll_degradation'],
                'held_out_passed':checks['passed'],'selectable':result.get('selectable'),
                'record_id':result['record_id'],'result_sha256':digest(root/'result.json'),
                'worker_exit_code':code,'elapsed_seconds':time.monotonic()-wall,'release_evidence':False,
                'next':('run full-RTL acceptance for this candidate' if checks['passed']
                    else 'record failure and stop promotion; do not tune on held-out')}
            write(root/'summary.json',summary); print(json.dumps(summary),flush=True)
        except BaseException:
            write(root/'failure.json',{'status':'failed','source':source(),'traceback':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-wall}); raise


if __name__=='__main__': main()
