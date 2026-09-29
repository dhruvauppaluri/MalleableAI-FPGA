"""Serialized indexed performance-suite driver: checks the shared deadline before each index."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--store',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True); p.add_argument('--log',type=Path,required=True)
    p.add_argument('--indices',default='0,1,2,3,4,5,6,7,8,9')
    a=p.parse_args()
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index in map(int,a.indices.split(',')):
            if time.time()>=a.deadline_unix:
                print(json.dumps({'index':index,'status':'not-dispatched-deadline'}),flush=True); break
            started=time.time()
            with a.log.open('a') as log:
                log.write(json.dumps({'index':index,'status':'start','unix':started})+'\n'); log.flush()
                code=subprocess.call([sys.executable,'-u','-m','malleable.llm.cli','performance-suite','--manifest',str(a.manifest),
                    '--store',str(a.store),'--run-index',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                log.write(json.dumps({'index':index,'status':'exit','code':code,'seconds':time.time()-started})+'\n')
            print(json.dumps({'index':index,'exit':code,'seconds':time.time()-started}),flush=True)


if __name__=='__main__': main()
