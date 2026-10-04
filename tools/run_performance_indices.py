"""Serialized indexed performance-suite driver: checks the shared deadline before each index."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def source_state():
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)}

def run_indices(manifest,store,deadline,log_path,indices):
    """Each dispatch has fresh evidence; incomplete execution returns nonzero."""
    frozen=source_state()
    if frozen['status']: raise ValueError('benchmark driver requires clean committed source')
    completed=[]; status='complete'; failure=None
    attempts=log_path.parent/(log_path.name+'.attempts'); attempts.mkdir(parents=True,exist_ok=True)
    for index in indices:
        if time.time()>=deadline: status='deadline-limited'; break
        if source_state()!=frozen: status='failed'; failure='source changed before index'; break
        started=time.time(); attempt=attempts/f'{index:02d}-{uuid.uuid4().hex}'
        attempt.mkdir()
        command=[sys.executable,'-u','-m','malleable.llm.cli','performance-suite',
            '--manifest',str(manifest),'--store',str(store),'--run-index',str(index)]
        job={'schema_version':1,'index':index,'source':frozen,'command':command,
            'started_unix':started,'dispatch_deadline_unix':deadline}
        (attempt/'job.json').write_text(json.dumps(job,indent=2))
        with (attempt/'worker.log').open('x') as output:
            code=subprocess.call(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        unchanged=source_state()==frozen
        record={**job,'code':code,'seconds':time.time()-started,'source_unchanged':unchanged,
            'status':'completed' if code==0 and unchanged else 'failed','attempt':str(attempt)}
        (attempt/'result.json').write_text(json.dumps(record,indent=2))
        with log_path.open('a') as log: log.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if code or not unchanged: status='failed'; failure='child failed or source changed'; break
        completed.append(index)
    summary={'schema_version':1,'status':status,'completed_indices':completed,'requested_indices':indices,
        'dispatch_deadline_unix':deadline,'source':frozen,'failure':failure}
    with log_path.open('a') as log: log.write(json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)
    return {'complete':0,'failed':1,'deadline-limited':3}[status]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--store',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True); p.add_argument('--log',type=Path,required=True)
    p.add_argument('--indices',default='0,1,2,3,4,5,6,7,8,9')
    a=p.parse_args()
    from malleable.llm.continuation import read_control
    read_control(a.deadline_unix)
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        indices=list(map(int,a.indices.split(',')))
        if not indices or len(set(indices))!=len(indices) or any(i not in range(10) for i in indices):
            raise ValueError('unique benchmark indices 0..9 required')
        return run_indices(a.manifest,a.store,a.deadline_unix,a.log,indices)


if __name__=='__main__': sys.exit(main())
