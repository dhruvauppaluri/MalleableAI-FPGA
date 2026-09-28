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
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    root=Path(args.output).resolve()
    if root.exists(): raise ValueError('refusing to overwrite prior verification evidence')
    root.mkdir(parents=True)
    # Pass a short executable name through recursive make. An absolute venv
    # path may contain spaces, and Makefile PYTHON expansions need to remain
    # shell-safe in both the parent and recursive recipes.
    env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).absolute().parent)+os.pathsep+env.get('PATH','')
    command=['make','verify-llm','PYTHON=python']
    started=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
    log=root/'verify-llm.log'; log.write_text(result.stdout)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,text=True,check=False).stdout.strip() or None
    report={'schema_version':1,'status':'passed' if result.returncode==0 else 'failed',
        'commands':['make verify-llm'],'command_argv':command,'started_utc':started,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,
        'commit':commit,'python_executable':sys.executable,'checkpoint_downloads':False,
        'log_path':log.name,'log_sha256':sha256(log),
        'scope':'dense RTL + upstream compiler/operator/Verilator + LLM host/UI tests/build; no real checkpoint inference'}
    evidence=root/'verification.json'; evidence.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'evidence':str(evidence),'sha256':sha256(evidence),'log':str(log)}),flush=True)
    if result.returncode:
        print(result.stdout[-12000:],file=sys.stderr)
        sys.exit(result.returncode)

if __name__=='__main__': main()
