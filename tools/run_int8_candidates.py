"""Screen compatible INT8 candidates on the frozen 128-target validation panel.

Serialized, one process per stage, fresh append-only attempt folder. Run only
after the user selects a batch mode; it loads the real checkpoint.
"""
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
from malleable.llm import candidates as C
from malleable.llm.models import digest


def source():
    files=('malleable/llm/candidates.py','malleable/llm/precision.py','tools/run_int8_candidates.py')
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'files':{p:digest(ROOT/p) for p in files}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True,type=Path)
    p.add_argument('--model',default='build/models/Qwen3-0.6B',type=Path)
    p.add_argument('--suite',default='build/models/quality/qwen3-frozen-v2.json',type=Path)
    p.add_argument('--panel',type=Path,help='existing frozen panel-128.json from the attribution attempt')
    p.add_argument('--reference-store',type=Path,help='verified floating-references store to copy')
    p.add_argument('--candidates',default='all-but-baseline',help="'all', 'all-but-baseline' or comma-separated names")
    p.add_argument('--calibration-tokens',type=int,default=512)
    p.add_argument('--persist-derived',action='store_true',help='also write derived safetensors (about 2 GB each)')
    p.add_argument('--authorization',default='')
    p.add_argument('--stage',choices=('calibrate','case'))
    p.add_argument('--index',type=int); a=p.parse_args(); root=a.root.resolve(); os.chdir(ROOT)
    if a.stage=='calibrate': print(json.dumps({'statistics_id':C.run_calibration(root,source())}),flush=True); return
    if a.stage=='case':
        emit=lambda kind,payload: print(json.dumps({'kind':kind,'payload':payload}),flush=True)
        print(json.dumps(C.run_case(root,a.index,source(),emit)),flush=True); return
    if root.exists(): raise ValueError('refusing to overwrite previous candidate attempt')
    if a.panel is None: raise ValueError('--panel is required: reuse the frozen 128-target panel')
    frozen=source()
    if frozen['status']: raise ValueError('commit a clean source before the candidate batch')
    candidates=C.parse_candidates(a.candidates)
    import fcntl
    lockdir=ROOT/'build/zephyrus-jobs'; lockdir.mkdir(parents=True,exist_ok=True)
    lock=(lockdir/'precision-attribution.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise ValueError('another heavy precision batch is active')
    C.prepare_attempt(root,a.model,a.suite,a.panel,candidates,frozen,a.reference_store,a.calibration_tokens,
        a.persist_derived,a.authorization)
    stage=lambda *extra: [sys.executable,str(Path(__file__).resolve()),'--root',str(root),*extra]
    wall=time.monotonic()
    print(json.dumps({'status':'running','root':str(root),'pid':os.getpid(),
        'started':datetime.now(timezone.utc).isoformat()}),flush=True)
    stages=[('calibration','calibration.log',stage('--stage','calibrate'))]+[
        (c.name,f'case-{i:02d}.log',stage('--stage','case','--index',str(i))) for i,c in enumerate(candidates)]
    for name,logname,command in stages:
        print(json.dumps({'stage':name,'status':'running'}),flush=True)
        with (root/logname).open('x') as stream: run=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if run.returncode:
            C._write(root/'failure.json',{'status':'failed','stage':name,'exit_code':run.returncode,'source':source()})
            raise RuntimeError(f'{name} failed; evidence retained at {root}')
        if source()!=frozen: raise ValueError('source changed during batch; stop dispatch')
    report=C.write_report(root,frozen,time.monotonic()-wall)
    print(json.dumps({'status':'completed','ranking':report['ranking'],'top_three':report['top_three']}),flush=True)


if __name__=='__main__': main()
