"""Run checkpoint-free integration gates and save immutable evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--cuda-cache',action='store_true'); args=parser.parse_args()
    root=Path(args.output).resolve()
    if root.exists(): raise ValueError('refusing to overwrite prior verification evidence')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
    if status: raise ValueError('freeze a clean source commit before final verification')
    root.mkdir(parents=True)
    # Pass a short executable name through recursive make. An absolute venv
    # path may contain spaces, and Makefile PYTHON expansions need to remain
    # shell-safe in both the parent and recursive recipes.
    env=os.environ.copy()
    if args.cuda_cache: env['MALLEABLE_CUDA_CACHE_TESTS']='1'
    env['PATH']=str(Path(sys.executable).absolute().parent)+os.pathsep+env.get('PATH','')
    command=['make','verify-llm','PYTHON=python']
    started=datetime.now(timezone.utc).isoformat()
    log=root/'verify-llm.log'
    print(json.dumps({'status':'running','commit':commit,'log':str(log)}),flush=True)
    with log.open('x') as stream:
        result=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,check=False)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    changed=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
    source_unchanged=current==commit and not changed
    report={'schema_version':1,'status':'passed' if result.returncode==0 and source_unchanged else 'failed',
        'commands':['make verify-llm'],'command_argv':command,'started_utc':started,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,
        'commit':commit,'python_executable':sys.executable,'checkpoint_downloads':False,
        'source_unchanged':source_unchanged,'source_dirty':False,
        'log_path':log.name,'log_sha256':sha256(log),
        'scope':'dense RTL + upstream compiler/operator/Verilator + LLM host/UI tests/build; no real checkpoint inference'}
    evidence=root/'verification.json'; evidence.write_text(json.dumps(report,indent=2)+'\n')
    if args.cuda_cache:
        import torch
        from malleable.llm.cuda import driver_metadata
        cuda={'schema_version':1,'status':report['status'],'actual_cuda':torch.cuda.is_available(),'depths':[1,2,4,8],
            'rejection_positions':'every position per depth','eos':True,'context_boundary':128,
            'model_scope':'checkpoint-free tiny synthetic Qwen3; not official verifier acceptance',
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'driver':driver_metadata(),'commit':commit,'log_sha256':report['log_sha256']}
        (root/'cuda-cache-verification.json').write_text(json.dumps(cuda,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'evidence':str(evidence),'sha256':sha256(evidence),'log':str(log)}),flush=True)
    if result.returncode or not source_unchanged:
        print(log.read_text()[-12000:],file=sys.stderr)
        sys.exit(result.returncode or 1)

if __name__=='__main__': main()
