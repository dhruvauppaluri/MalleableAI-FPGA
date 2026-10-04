"""One explicitly authorized, serialized Qwen3.5 acceptance batch, no retries."""
import argparse
from datetime import datetime, timezone
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
from malleable.llm.quality import frozen_suite,gate,reference_contract
from malleable.records import identity

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())

def validate_quality(result,info,suite,split,freeze=None):
    data,hashes=frozen_suite(suite)
    count=sum(len(row)-1 for row in data[split])
    C.validate_configuration(result,info)
    if (count<1024 or result.get('target_count')!=count or result.get('samples')!=count
        or result.get('split')!=split or result.get('suite_frozen') is not True
        or result.get('suite_file_hash')!=digest(suite) or result.get('suite_hashes')!=hashes
        or result.get('derived_candidate') is not None
        or result.get('variant_id')!=C.variant_id(info['base_model_id'],'int8')
        or any(result.get(k)!=info[k] for k in ('base_model_id','tokenizer_id'))
        or result.get('toolchain',{}).get('contract')!=reference_contract(info)):
        raise ValueError('quality result differs from frozen raw INT8 acceptance design')
    checks=gate(result['float_nll'],result['candidate_nll'],result['agreement'])
    if result.get('passed') is not checks['passed']:raise ValueError('inconsistent reported quality gate')
    if split=='held-out' and (not freeze or result.get('candidate_freeze_id')!=freeze['freeze_id']
        or result.get('configuration_id')!=freeze['configuration']['configuration_id']):
        raise ValueError('held-out result differs from its freeze')
    if split=='held-out' and checks['passed'] and result.get('selectable') is not True:
        raise ValueError('passing held-out result lacks approval')
    return checks

def validate_rtl(result,quality):
    workload=result.get('workload',{});steps=result.get('counters',[])
    if (result.get('status')!='completed' or result.get('valid') is not True
        or result.get('backend')!='rtl' or result.get('generation_mode')!='greedy'
        or len(result.get('tokens',[]))!=8 or result.get('prompt_tokens',0)<1
        or len(steps)!=result['prompt_tokens']+7
        or any(s.get('validation')!='bit-exact-dram-and-tmem' or not s.get('trace_sha256') for s in steps)
        or not result.get('toolchain',{}).get('build_id')
        or workload.get('context')!=128 or workload.get('wformat')!='int8'
        or workload.get('max_new')!=8 or result.get('personality')!='balanced'
        or result.get('derived_candidate') is not None
        or any(result.get(k)!=quality[k] for k in ('base_model_id','tokenizer_id','variant_id','configuration_id'))):
        raise ValueError('eight-token full-RTL acceptance failed; preserve early EOS and do not retry automatically')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--unit',required=True);args=parser.parse_args();os.chdir(ROOT)
    root=args.root.resolve();authorization=json.loads((root/'authorization.json').read_text())
    if authorization.get('scope')!=['validation','held-out-once-if-validation-passes','rtl-eight-if-held-out-passes']:
        raise ValueError('explicit full acceptance authorization required')
    if authorization.get('user_instruction')!='go for full acceptance':raise ValueError('authorization mismatch')
    model=ROOT/'build/models/Qwen3.5-0.8B';suite=ROOT/'build/models/quality/qwen35-frozen-v2.json'
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=C.source_state()
        if source['status']:raise ValueError('clean committed source required')
        write(root/'job.json',{'source':source,'pid':os.getpid(),'unit':args.unit,'authorization':authorization,
            'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}})
        shutil.copyfile(__file__,root/'runner.py')
        initial=inspect(model,128);suite_hash=digest(suite);started=time.monotonic();stage='preflight'
        def stable():
            if C.source_state()!=source or digest(suite)!=suite_hash:raise ValueError('source or frozen suite changed')
            current=inspect(model,128)
            if any(current[k]!=initial[k] for k in ('base_model_id','tokenizer_id','weight_files','tokenizer_files')):
                raise ValueError('checkpoint or tokenizer changed')
        def run(name,command):
            stable();directory=root/name;directory.mkdir(exist_ok=True)
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<2*initial['fp32_tensor_bytes']+2*1024**3:raise ValueError('insufficient available memory before dispatch')
            write(directory/'job.json',{'source':source,'command':command,'started_unix':time.time(),
                'memory_available_bytes':available,'authorization_sha256':digest(root/'authorization.json')})
            began=time.monotonic();result=None
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                write(directory/'child.json',{'pid':child.pid})
                for line in child.stdout:
                    log.write(line);log.flush()
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get('kind')=='result':result=event['payload']
                    elif event.get('kind') in ('phase','quality-progress','failure','error'):
                        print(json.dumps({'stage':name,**event}),flush=True)
                code=child.wait()
            if code or result is None:raise RuntimeError(f'{name} child exited {code}; inspect worker.log; no retry')
            write(directory/'result.json',result)
            if result.get('record_id')!=identity({k:v for k,v in result.items() if k!='record_id'}):
                raise ValueError('canonical result identity mismatch')
            stable()
            return result,{'status':'completed','source':source,'elapsed_seconds':time.monotonic()-began,
                'result_sha256':digest(directory/'result.json'),'record_id':result['record_id']}
        common=['--model',str(model),'--suite',str(suite),'--context','128','--personality','balanced',
            '--wformat','int8','--max-host-gib','16']
        cli=[sys.executable,'-u','-m','malleable.llm.cli']
        try:
            if initial['family']!='qwen35' or not initial['supported']:raise ValueError('unsupported checkpoint')
            stage='validation'
            validation,summary=run(stage,cli+['quality',*common,'--split','validation','--store',str(root/stage/'store')])
            checks=validate_quality(validation,initial,suite,stage)
            summary.update(validation_passed=checks['passed'],agreement=validation['agreement'],nll_degradation=checks['nll_degradation'])
            write(root/stage/'summary.json',summary)
            if not checks['passed']:raise ValueError('full validation failed; held-out remains untouched')
            stage='held-out';(root/stage).mkdir()
            stable()
            freeze=C.create_freeze(root/stage/'freeze.json',initial,suite,None,validation,
                digest(root/'validation/result.json'),source)
            # quality.evaluate atomically consumes the persistent held-out registry.
            heldout,summary=run(stage,cli+['quality',*common,'--split','held-out',
                '--candidate-freeze',str(root/stage/'freeze.json'),'--store',str(root/stage/'store')])
            checks=validate_quality(heldout,initial,suite,stage,freeze)
            summary.update(held_out_passed=checks['passed'],agreement=heldout['agreement'],nll_degradation=checks['nll_degradation'])
            write(root/stage/'summary.json',summary)
            if not checks['passed']:raise ValueError('held-out failed; stop promotion, no tuning or repeat')
            stage='rtl-eight'
            result,summary=run(stage,cli+['generate','--model',str(model),'--prompt','Once upon a time there was a small',
                '--prompt-format','raw','--max-new','8','--context','128','--backend','rtl','--personality','balanced',
                '--wformat','int8','--seed','0','--max-host-gib','16','--store',str(root/stage/'store')])
            validate_rtl(result,heldout);summary['rtl_passed']=True;write(root/stage/'summary.json',summary)
            write(root/'terminal.json',{'status':'completed','qwen35_acceptance_passed':True,'source':source,
                'elapsed_seconds':time.monotonic()-started,'scope':'Qwen3.5 standalone evidence; full release still requires manifest checks'})
        except BaseException:
            write(root/'terminal.json',{'status':'failed','stage':stage,'traceback':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-started,'source':C.source_state(),'automatic_retry':False})
            raise

if __name__=='__main__':main()
