"""One approved INT8 pilot, then ten fixed precision cases only on screen failure."""
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
from malleable.llm.precision import GROUPS,diagnose_precision,validate_policy
from malleable.llm.candidates import screen_passes
from malleable.llm.models import digest
from malleable.store import Store

RELEASE=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'
SCHEDULE=(('key-float32',{'key_store':('float32',128)}),
    ('key-float16',{'key_store':('float16',128)}),
    ('key-int16-128',{'key_store':('int16',128)}),
    ('key-int8-64',{'key_store':('int8',64)}),
    ('key-int8-32',{'key_store':('int8',32)}),
    ('key-int8-16',{'key_store':('int8',16)}),
    ('key-query-float32',{'key_store':('float32',128),'query':('float32',128)}),
    ('key-query-float16',{'key_store':('float16',128),'query':('float16',128)}),
    ('key-query-int16-128',{'key_store':('int16',128),'query':('int16',128)}),
    ('key-query-int16-64',{'key_store':('int16',64),'query':('int16',64)}))

def write(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def read(path): return json.loads(path.read_text())

def source():
    files=('malleable/llm/precision.py','malleable/llm/candidates.py','tools/run_quality_recovery_pilot.py')
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'files':{p:digest(ROOT/p) for p in files}}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--authorization',default=''); parser.add_argument('--worker',type=int)
    args=parser.parse_args(); root=args.root.resolve(); os.chdir(ROOT)
    if args.worker is not None:
        frozen=read(root/'plan.json')['source']
        if source()!=frozen: raise ValueError('source changed before precision case')
        case=root/'precision'/f'case-{args.worker:02d}'
        with (case/'events.jsonl').open('x') as log:
            def emit(kind,payload):
                line=json.dumps({'kind':kind,'payload':payload}); log.write(line+'\n'); log.flush()
                print(line,flush=True)
            result=diagnose_precision(ROOT/'build/models/Qwen3-0.6B',ROOT/'build/models/quality/qwen3-frozen-v2.json',
                read(case/'policy.json'),read(root/'panel.json'),root/'precision/floating-references',emit=emit)
        if source()!=frozen: raise ValueError('source changed during precision case')
        result.update(source=frozen,status='completed')
        store=Store(root/'precision/research')
        try: key=store.save('llm-precision-diagnostic',result)
        finally: store.close()
        write(case/'result.json',result)
        summary={k:result[k] for k in ('agreement','candidate_nll','float_nll','nll_degradation_percent',
            'target_count','executed_tokens','candidate_seconds','elapsed_seconds','precision_policy_id')}
        summary.update(case=args.worker,record_id=key,result_sha256=digest(case/'result.json'),
            screen_passes=screen_passes(summary))
        write(case/'summary.json',summary); return
    frozen=source()
    if frozen['status']: raise ValueError('commit clean source before dispatch')
    root.mkdir(parents=True)
    shutil.copyfile(__file__,root/'runner.py')
    shutil.copyfile(RELEASE/'precision-attribution-01/panel-128.json',root/'panel.json')
    cases=[]
    for name,overrides in SCHEDULE:
        policy=validate_policy({'schema_version':2,'quantized_groups':list(GROUPS),'group_size':128,
            'site_overrides':{k:{'format':v[0],'block':v[1]} for k,v in overrides.items()}})
        cases.append({'name':name,'policy':policy})
    write(root/'plan.json',{'source':frozen,'pid':os.getpid(),'authorization':args.authorization,
        'started_utc':datetime.now(timezone.utc).isoformat(),'panel_sha256':digest(root/'panel.json'),
        'pilot_candidates':['a0.5-cnone','a0.5-c99.9'],'calibration_tokens':512,
        'fallback_cases':cases,'fallback_condition':'no pilot candidate reaches agreement >=90% AND NLL degradation <=5%',
        'release_evidence':False,'held_out_used':False})
    started=time.monotonic(); completed=[]
    print(json.dumps({'status':'running','root':str(root),'pid':os.getpid(),'phase':'int8-pilot'}),flush=True)
    try:
        command=[sys.executable,str(ROOT/'tools/run_int8_candidates.py'),'--root',str(root/'int8'),
            '--panel',str(root/'panel.json'),'--reference-store',str(RELEASE/'precision-128-pilot-01/floating-references'),
            '--candidates','a0.5-cnone,a0.5-c99.9','--persist-derived','--authorization',args.authorization]
        with (root/'int8-worker.log').open('x') as log:
            run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode: raise RuntimeError('INT8 implementation/job failure; preserve and fix before numerical fallback')
        pilot=read(root/'int8/report.json'); print(json.dumps({'phase':'int8-pilot','cases':pilot['cases'],'screen_passes':pilot['screen_passes']}),flush=True)
        if source()!=frozen: raise ValueError('source changed during pilot')
        if not pilot['screen_passes']:
            with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                precision=root/'precision'; precision.mkdir()
                shutil.copytree(RELEASE/'precision-128-pilot-01/floating-references',precision/'floating-references')
                for i,config in enumerate(cases):
                    case=precision/f'case-{i:02d}'; case.mkdir(); write(case/'policy.json',config['policy'])
                for i,config in enumerate(cases):
                    case=precision/f'case-{i:02d}'
                    print(json.dumps({'phase':'precision','case':i,'name':config['name'],'status':'running'}),flush=True)
                    with (case/'worker.log').open('x') as log:
                        run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--worker',str(i)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if run.returncode: raise RuntimeError(f'precision case {i} failed')
                    summary=read(case/'summary.json')
                    if digest(case/'result.json')!=summary['result_sha256']: raise ValueError('precision result hash mismatch')
                    summary['name']=config['name']; completed.append(summary)
                    print(json.dumps(summary),flush=True)
        write(root/'report.json',{'status':'completed','source':frozen,'pilot':pilot,'precision_cases':completed,
            'elapsed_seconds':time.monotonic()-started,'release_evidence':False,'held_out_used':False})
        print(json.dumps({'status':'completed','report':str(root/'report.json')}),flush=True)
    except BaseException:
        write(root/'failure.json',{'status':'failed','traceback':traceback.format_exc(),'completed_precision_cases':completed,
            'elapsed_seconds':time.monotonic()-started}); raise

if __name__=='__main__': main()
