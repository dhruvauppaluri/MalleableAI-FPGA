"""One authorized, serialized attribution batch with durable per-case evidence."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from malleable.llm.precision import GROUPS,make_panel,diagnose_precision
from malleable.llm.models import digest
from malleable.records import identity


def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def source():
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'files':{p:digest(ROOT/p) for p in ('malleable/llm/precision.py','tools/run_precision_attribution.py')}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True,type=Path)
    p.add_argument('--model',default='build/models/Qwen3-0.6B',type=Path)
    p.add_argument('--suite',default='build/models/quality/qwen3-frozen-v2.json',type=Path)
    p.add_argument('--worker',type=int); a=p.parse_args(); root=a.root.resolve()
    os.chdir(ROOT)
    if a.worker is not None:
        case=root/f'case-{a.worker:02d}'; config=json.loads((case/'policy.json').read_text())
        panel=json.loads((root/'panel-16.json').read_text())
        with (case/'events.jsonl').open('x') as log:
            def emit(kind,payload):
                line=json.dumps({'kind':kind,'payload':payload},allow_nan=False)
                log.write(line+'\n'); log.flush(); print(line,flush=True)
            result=diagnose_precision(a.model,a.suite,config,panel,root/'floating-references',emit=emit)
        from malleable.store import Store
        result['source']=source(); result['status']='completed'
        store=Store(root/'research')
        try: key=store.save('llm-precision-diagnostic',result)
        finally: store.close()
        write(case/'result.json',result)
        summary={k:result[k] for k in ('agreement','float_nll','candidate_nll','nll_degradation_percent',
            'candidate_seconds','elapsed_seconds','target_count','executed_tokens','precision_policy_id')}
        summary.update(record_id=key,result_sha256=digest(case/'result.json'),case=a.worker)
        write(case/'summary.json',summary); print(json.dumps(summary),flush=True); return
    if root.exists(): raise ValueError('refusing to overwrite previous attribution attempt')
    frozen=source()
    if frozen['status']: raise ValueError('commit a clean source before the diagnostic batch')
    import fcntl
    lockdir=ROOT/'build/zephyrus-jobs'; lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'precision-attribution.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('another precision batch is active')
        root.mkdir(parents=True)
        write(root/'panel-16.json',make_panel(a.suite,16,'prefix'))
        write(root/'panel-128.json',make_panel(a.suite,128,'spread'))
        cases=[('baseline',list(GROUPS)),('all-floating',[])]+[
            ('bypass-'+g,[x for x in GROUPS if x!=g]) for g in GROUPS]
        write(root/'batch.json',{'schema_version':1,'source':frozen,'started_utc':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'authorization':'user selected ten distinct 16-target serialized precision cases, then report',
            'model':str(a.model.resolve()),'suite':str(a.suite.resolve()),'cases':[n for n,_ in cases]+['adaptive-top-two-bypass'],
            'adaptive_rule':'rank bypass cases by agreement descending, then candidate NLL ascending, then GROUPS order',
            'release_evidence':False})
        print(json.dumps({'status':'running','root':str(root),'pid':os.getpid()}),flush=True)
        summaries=[]; wall=time.monotonic()
        for index in range(10):
            if index==9:
                ranked=sorted(range(7),key=lambda i:(-summaries[i+2]['agreement'],summaries[i+2]['candidate_nll'],i))
                disabled=[GROUPS[i] for i in ranked[:2]]
                cases.append(('bypass-'+'-and-'.join(disabled),[g for g in GROUPS if g not in disabled]))
            name,groups=cases[index]; case=root/f'case-{index:02d}'; case.mkdir()
            write(case/'policy.json',{'schema_version':1,'quantized_groups':groups,'group_size':128})
            command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--model',str(a.model.resolve()),
                     '--suite',str(a.suite.resolve()),'--worker',str(index)]
            print(json.dumps({'case':index,'name':name,'status':'running'}),flush=True)
            with (case/'worker.log').open('x') as log:
                run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if run.returncode:
                write(case/'failure.json',{'status':'failed','exit_code':run.returncode,'source':source()})
                write(root/'failure.json',{'status':'failed','case':index,'completed':summaries})
                raise RuntimeError(f'case {index} failed; evidence retained at {case}')
            summary=json.loads((case/'summary.json').read_text()); summary['name']=name; summaries.append(summary)
            estimate=(time.monotonic()-wall)/(index+1)*(9-index)
            print(json.dumps(dict(summary,estimated_remaining_seconds=estimate)),flush=True)
            if source()!=frozen: raise ValueError('source changed during batch; stop dispatch')
        write(root/'report.json',{'schema_version':1,'status':'completed','source':frozen,'cases':summaries,
            'elapsed_seconds':time.monotonic()-wall,'release_evidence':False,
            'conclusion':'diagnostic-only; precision bypasses require executable ISA/RTL implementation before acceptance'})
        print(json.dumps({'status':'completed','report':str(root/'report.json')}),flush=True)


if __name__=='__main__': main()
