"""Loopback-only IDE with durable jobs, no shell execution or board programming."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager


class Jobs:
    def __init__(self,root,model_root,autostart=True):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.model_root=Path(model_root).resolve()
        self.lock=threading.Lock(); self.processes={}; self.stop=threading.Event()
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,kind TEXT,payload TEXT,status TEXT,created REAL,returncode INTEGER)')
            db.execute("UPDATE jobs SET status='interrupted' WHERE status='running'")
        self.threads=[]
        if autostart:
            for _ in range(2):
                thread=threading.Thread(target=self.worker,daemon=True); thread.start(); self.threads.append(thread)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.root/'jobs.sqlite3',timeout=30); db.row_factory=sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def path(self,path,suffix=None):
        candidate=Path(path).resolve()
        if not candidate.is_relative_to(self.model_root) or not candidate.exists():
            raise ValueError('file must exist inside the configured model root')
        if suffix and candidate.suffix!=suffix: raise ValueError('unsupported file type')
        return candidate

    def create(self,kind,payload):
        if kind not in ('generate','analyze','optimize','quartus','train','export','quality'):
            raise ValueError('unsupported job kind')
        allowed={'artifact','model','dataset','prompt','max_new','lanes','active_lanes','seed','backend',
                 'clock_hz','horizon','switch_cost','budget','profile','steps','sequence_length','device',
                 'width','layers','state_size','family','resume','split'}
        if set(payload)-allowed: raise ValueError('unsupported job fields')
        for field,suffix in [('artifact','.mssm'),('dataset','.json'),('model',None)]:
            if field in payload: self.path(payload[field],suffix)
        identifier=uuid.uuid4().hex
        if kind=='train':
            # Model output stays inside the configured local model root.
            payload=dict(payload,model=str(self.model_root/('trained-'+identifier)))
        self.argv(kind,payload)  # Validate before queueing.
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,NULL)',(identifier,kind,json.dumps(payload),'queued',time.time()))
        return identifier

    def argv(self,kind,payload):
        argv=[sys.executable,'-u','-m','malleable.ssm.cli',kind,'--store',str(self.root/'research')]
        for key,value in payload.items():
            if key=='resume':
                if value: argv.append('--resume')
                continue
            if key=='switch_cost': value=json.dumps(value,allow_nan=False)
            if not isinstance(value,(str,int,float)) or isinstance(value,bool): raise ValueError('invalid field value')
            argv.append('--'+key.replace('_','-')+'='+str(value))
        return argv

    def list(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 100')]

    def cancel(self,identifier):
        with self.lock:
            with self.connect() as db:
                row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
                if not row: raise ValueError('unknown job')
                if row['status'] not in ('queued','running'): return
                db.execute("UPDATE jobs SET status='cancelled' WHERE id=?",(identifier,))
            process=self.processes.get(identifier)
            if process and process.poll() is None:
                if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
                else: subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)

    def resume(self,identifier):
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row or row['kind']!='train' or row['status'] not in ('cancelled','interrupted','failed'):
                raise ValueError('only stopped training jobs can resume')
            payload=json.loads(row['payload'])
            if not (Path(payload['model'])/'training.pt').exists(): raise ValueError('no checkpoint to resume')
            payload['resume']=True
            db.execute("UPDATE jobs SET status='queued',payload=? WHERE id=?",(json.dumps(payload),identifier))

    def worker(self):
        while not self.stop.wait(.2):
            with self.lock:
                with self.connect() as db:
                    row=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                    if not row: continue
                    db.execute("UPDATE jobs SET status='running' WHERE id=?",(row['id'],))
                directory=self.root/row['id']; directory.mkdir(exist_ok=True)
                try:
                    log=(directory/'stdout.txt').open('w')
                    process=subprocess.Popen(self.argv(row['kind'],json.loads(row['payload'])),
                                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
                    self.processes[row['id']]=process
                except Exception as error:
                    (directory/'stdout.txt').write_text(str(error))
                    with self.connect() as db: db.execute("UPDATE jobs SET status='failed' WHERE id=?",(row['id'],))
                    continue
            code=process.wait(); log.close()
            with self.lock:
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status=?,returncode=? WHERE id=? AND status='running'",
                               ('completed' if code==0 else 'failed',code,row['id']))
                self.processes.pop(row['id'],None)


def create_app(model_root='build',job_root='build/ide-jobs',start_worker=True):
    from fastapi import FastAPI,Request,HTTPException
    from fastapi.responses import FileResponse
    app=FastAPI(title='Malleable SSM Lab')
    jobs=Jobs(job_root,model_root,start_worker); app.state.jobs=jobs

    @app.middleware('http')
    async def local_only(request:Request,call_next):
        from fastapi.responses import JSONResponse
        host=request.headers.get('host','').split(':')[0]
        origin=request.headers.get('origin')
        if host not in ('localhost','127.0.0.1','testserver') or (origin and origin!=f'http://{request.headers.get("host")}'):
            return JSONResponse({'detail':'loopback same-origin requests only'},status_code=403)
        return await call_next(request)

    @app.get('/api/models')
    def models():
        return [{'path':str(p),'name':p.name} for p in list(jobs.model_root.rglob('*.mssm'))[:100]]

    @app.get('/api/jobs')
    def job_list(): return jobs.list()

    @app.post('/api/jobs/{kind}')
    async def create(kind:str,request:Request):
        try: return {'id':jobs.create(kind,await request.json())}
        except (ValueError,TypeError) as error: raise HTTPException(400,str(error))

    @app.post('/api/job/{identifier}/cancel')
    def cancel(identifier:str):
        try: jobs.cancel(identifier); return {'status':'cancelled'}
        except ValueError as error: raise HTTPException(400,str(error))

    @app.post('/api/job/{identifier}/resume')
    def resume(identifier:str):
        try: jobs.resume(identifier); return {'status':'queued'}
        except ValueError as error: raise HTTPException(400,str(error))

    @app.get('/api/job/{identifier}/log')
    def log(identifier:str):
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
            raise HTTPException(400,'invalid job id')
        path=jobs.root/identifier/'stdout.txt'
        return {'text':path.read_text(errors='replace')[-200000:] if path.exists() else ''}

    @app.get('/')
    def index(): return FileResponse(Path(__file__).parent/'web/index.html')

    @app.get('/app.js')
    def script(): return FileResponse(Path(__file__).parent/'web/app.js',media_type='text/javascript')

    @app.get('/app.css')
    def css(): return FileResponse(Path(__file__).parent/'web/app.css',media_type='text/css')
    return app


def main():
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--model-root',default='build')
    parser.add_argument('--job-root',default='build/ide-jobs')
    args=parser.parse_args()
    uvicorn.run(create_app(args.model_root,args.job_root),host='127.0.0.1',port=args.port)


if __name__=='__main__': main()
