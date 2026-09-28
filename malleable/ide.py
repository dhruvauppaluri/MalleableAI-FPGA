"""Loopback workbench: durable ordered events and isolated allowlisted workers."""
import argparse
import asyncio
from contextlib import contextmanager, asynccontextmanager
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
from .llm.records import GenerationWorkload

KINDS={'generate':'rtl','benchmark':'rtl','analyze':'cpu','quality':'rtl',
    'gpu-generate':'cuda','hybrid-generate':'hybrid','llm-performance-suite':'rtl'}
LEARNING={'llm-optimize':'optimize','llm-train':'train','llm-evaluate':'evaluate',
    'llm-predictor-train':'predictor-train','llm-predictor-evaluate':'predictor-evaluate',
    'llm-promote':'promote','llm-rollback':'rollback'}
KINDS.update(dict.fromkeys(LEARNING,'cpu'))
RETIRED={'train','export','quartus','optimize'}

class Jobs:
    def __init__(self,root,model_root,autostart=True):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.model_root=Path(model_root).resolve(); self.model_root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.processes={}; self.reserved=set(); self.stop=threading.Event()
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,kind TEXT,payload TEXT,status TEXT,created REAL,returncode INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, payload TEXT, created REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS event_run ON events(run_id,id)')
            db.execute("UPDATE jobs SET status='interrupted' WHERE status='running'")
            # Old SSM records stay readable, but are never dispatched/resumed.
            for kind in RETIRED:
                db.execute("UPDATE jobs SET status='retired' WHERE kind=? AND status IN ('queued','interrupted')",(kind,))
            for row in db.execute("SELECT id,payload FROM jobs WHERE status IN ('queued','interrupted')"):
                payload=json.loads(row['payload'])
                if 'artifact' in payload or any(isinstance(v,str) and v.endswith('.mssm') for v in payload.values()):
                    db.execute("UPDATE jobs SET status='retired' WHERE id=?",(row['id'],))
        self.threads=[]
        if autostart:
            for resource in ('rtl','cuda','hybrid','cpu'):
                thread=threading.Thread(target=self.worker,args=(resource,),daemon=True)
                thread.start(); self.threads.append(thread)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.root/'jobs.sqlite3',timeout=30); db.row_factory=sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def path(self,value,suffix=None):
        if not isinstance(value,str): raise ValueError('path must be a string')
        candidate=Path(value).resolve()
        if not candidate.is_relative_to(self.model_root) or not candidate.exists():
            raise ValueError('file must exist within configured model root')
        if suffix and candidate.suffix!=suffix: raise ValueError('unsupported file type')
        return candidate

    def create(self,kind,payload,trusted_internal=False):
        if kind not in KINDS: raise ValueError('unsupported or retired job kind')
        if not isinstance(payload,dict): raise ValueError('object payload required')
        if not trusted_internal and {'previous_config','optimization_session_id'} & set(payload):
            raise ValueError('optimizer lineage fields are reserved for internal validated sessions')
        if kind in LEARNING:
            allowed={'current','window','episodes','checkpoint','report','previous','passes','seed','split'}
            if set(payload)-allowed: raise ValueError('unsupported learning job fields')
            payload=dict(payload)
            for key in ('window','episodes'):
                if key in payload: payload[key]=str(self.path(payload[key],'.json'))
            for key in ('current','checkpoint','report','previous'):
                if key in payload:
                    from .store import Store
                    store=Store(self.root/'research')
                    try: store.load(payload[key])
                    finally: store.close()
            if 'passes' in payload and (type(payload['passes']) is not int or not 1<=payload['passes']<=10000): raise ValueError('invalid passes')
            if 'seed' in payload and (type(payload['seed']) is not int or not 0<=payload['seed']<2**31): raise ValueError('invalid seed')
            if payload.get('split','held-out') not in ('validation','held-out'): raise ValueError('invalid evaluation split')
            return self.enqueue(kind,payload)
        if kind in ('gpu-generate','hybrid-generate'):
            allowed={'model','verifier','standalone_manifest','prompt','prompt_format','messages',
                'max_new','context','dtype','depth','personality','wformat'}
            if set(payload)-allowed: raise ValueError('unsupported GPU/hybrid job fields')
            payload=dict(payload)
            payload['model']=str(self.path(payload.get('model')))
            if not Path(payload['model']).is_dir(): raise ValueError('model checkpoint directory required')
            if kind=='hybrid-generate':
                payload['verifier']=str(self.path(payload.get('verifier')))
                if not Path(payload['verifier']).is_dir(): raise ValueError('verifier checkpoint directory required')
            manifest=payload.get('standalone_manifest')
            if not isinstance(manifest,str): raise ValueError('explicit standalone_manifest is required')
            manifest_path=Path(manifest).resolve()
            if not manifest_path.is_relative_to(self.root) or not manifest_path.is_file():
                raise ValueError('standalone manifest must exist within the job data root')
            from .llm.release import check
            check(manifest_path)
            workload={k:v for k,v in payload.items() if k not in ('model','verifier','standalone_manifest','dtype','depth')}
            GenerationWorkload(**workload)
            if payload.get('dtype','float16') not in ('float16','float32'): raise ValueError('invalid CUDA dtype')
            if kind=='hybrid-generate' and payload.get('depth',4) not in (1,2,4,8): raise ValueError('invalid draft depth')
            return self.enqueue(kind,payload)
        if kind=='llm-performance-suite':
            if set(payload)!={'manifest'}: raise ValueError('manifest path required')
            manifest=self.path(payload['manifest'],'.json')
            data=json.loads(manifest.read_text())
            model=Path(data.get('model_path','')).resolve()
            if not model.is_relative_to(self.model_root) or not model.is_dir():
                raise ValueError('manifest checkpoint must remain inside model root')
            from .llm.models import inspect
            from .llm.experiments import validate_performance_manifest
            validate_performance_manifest(data,inspect(model))
            return self.enqueue(kind,{'manifest':str(manifest)})
        common={'model','context'}
        allowed=common | (set(GenerationWorkload.__dataclass_fields__)-{'schema_version'} |
                          {'previous_config','optimization_session_id'} if kind in ('generate','benchmark')
                          else {'suite','wformat','split','personality','max_host_gib'} if kind=='quality' else set())
        if set(payload)-allowed: raise ValueError('unsupported job fields')
        payload=dict(payload); payload['model']=str(self.path(payload.get('model')))
        if not Path(payload['model']).is_dir(): raise ValueError('model checkpoint directory required')
        if kind in ('generate','benchmark'): GenerationWorkload(**{k:v for k,v in payload.items()
            if k not in ('model','previous_config','optimization_session_id')})
        elif kind=='analyze':
            context=payload.get('context',2048)
            if type(context) is not int or not 2<=context<=2048: raise ValueError('invalid context')
        else:
            payload['suite']=str(self.path(payload.get('suite'),'.json'))
            from .llm.quality import suites
            suites(payload['suite'])
            if payload.get('wformat','int8') not in ('int8','int4','fp4'): raise ValueError('invalid format')
            if payload.get('split','validation') not in ('validation','held-out'): raise ValueError('invalid split')
            if payload.get('personality','balanced') not in ('compact','balanced','compute','buffered'): raise ValueError('invalid personality')
        return self.enqueue(kind,payload)

    def enqueue(self,kind,payload):
        identifier=uuid.uuid4().hex
        self.argv(kind,payload,identifier)
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,NULL)',(identifier,kind,json.dumps(payload),'queued',time.time()))
        self.event(identifier,'status',{'status':'queued'})
        return identifier

    def argv(self,kind,payload,identifier):
        if kind not in KINDS: raise ValueError('retired jobs cannot resume')
        command=LEARNING.get(kind,'performance-suite' if kind=='llm-performance-suite' else kind)
        argv=[sys.executable,'-u','-m','malleable.llm.cli',command,'--store',
            str(self.root/'research' if kind in LEARNING else self.root/identifier)]
        for key,value in payload.items():
            if key in ('previous_config','optimization_session_id'): continue
            if key in ('messages','input_tokens'): value=json.dumps(value,allow_nan=False)
            if type(value) not in (str,int,float): raise ValueError('invalid field type')
            argv.append('--'+key.replace('_','-')+'='+str(value))
        return argv

    def get(self,identifier):
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('invalid run ID')
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        if not row: raise ValueError('unknown run')
        return dict(row)

    def list(self):
        with self.connect() as db: return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 100')]

    def event(self,identifier,kind,payload):
        with self.connect() as db:
            db.execute('INSERT INTO events(run_id,kind,payload,created) VALUES (?,?,?,?)',
                       (identifier,kind,json.dumps(payload,allow_nan=False),time.time()))

    def events(self,identifier,after=0):
        with self.connect() as db:
            return [dict(r,id=r['id'],payload=json.loads(r['payload'])) for r in db.execute(
                'SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 200',(identifier,after))]

    def cancel(self,identifier):
        with self.lock:
            row=self.get(identifier)
            if row['status'] not in ('queued','running'): return
            with self.connect() as db: db.execute("UPDATE jobs SET status='cancelled' WHERE id=?",(identifier,))
            process=self.processes.get(identifier)
            if process and process.poll() is None:
                try:
                    if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
                    else: subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                except ProcessLookupError: pass
            self.event(identifier,'status',{'status':'cancelled'})
            self.terminal(identifier,'cancelled',{'requested_by':'user','previous_status':row['status']})

    def close(self):
        self.stop.set()
        for identifier in list(self.processes): self.cancel(identifier)
        for thread in self.threads: thread.join(timeout=5)

    def worker(self,resource):
        while not self.stop.wait(.1):
            with self.lock:
                with self.connect() as db:
                    queued=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 100").fetchall()
                    row=None
                    for candidate in queued:
                        kind=candidate['kind']; primary=KINDS.get(kind)
                        required={'rtl','cuda'} if primary=='hybrid' else ({primary} if primary in ('rtl','cuda') else set())
                        if primary==resource and required.isdisjoint(self.reserved):
                            row=candidate; self.reserved.update(required); break
                    if not row: continue
                    db.execute("UPDATE jobs SET status='running' WHERE id=?",(row['id'],))
                identifier=row['id']; directory=self.root/identifier; directory.mkdir(exist_ok=True)
                try:
                    process=subprocess.Popen(self.argv(row['kind'],json.loads(row['payload']),identifier),
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,
                        start_new_session=os.name=='posix',env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
                    self.processes[identifier]=process
                except Exception as error:
                    self.event(identifier,'failure',{'message':str(error)})
                    with self.connect() as db: db.execute("UPDATE jobs SET status='failed' WHERE id=?",(identifier,))
                    if resource in ('rtl','cuda'):
                        self.reserved.discard(resource)
                    if resource=='hybrid': self.reserved.difference_update({'rtl','cuda'})
                    self.terminal(identifier,'failed',{'type':type(error).__name__,'message':str(error)})
                    continue
            code=-1
            try:
                self.event(identifier,'status',{'status':'running'})
                with (directory/'stdout.txt').open('w') as log:
                    for line in process.stdout:
                        log.write(line); log.flush()
                        try:
                            value=json.loads(line)
                            if not isinstance(value,dict) or not isinstance(value.get('kind'),str) or 'payload' not in value: raise ValueError()
                        except (ValueError,TypeError): value={'kind':'log','payload':{'text':line[:4000]}}
                        if value['kind'] in ('result','model','candidate'):
                            canonical_id=self.persist(row['kind'],identifier,value)
                            if canonical_id:
                                linked=self.artifact_metadata(identifier)
                                value['payload']=dict(value['payload'],record_id=canonical_id,**linked)
                        self.event(identifier,value['kind'],value['payload'])
                code=process.wait()
            except Exception as error:
                # A storage/event error must not strand a simulator or kill the
                # sole worker. Keep cancellation status authoritative.
                if process.poll() is None:
                    try:
                        if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
                        else: process.terminate()
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
                        else: process.kill()
                        process.wait()
                    except ProcessLookupError: pass
                try: self.event(identifier,'failure',{'type':type(error).__name__,'message':str(error)})
                except (OSError,sqlite3.Error): pass
            finally:
                if process.stdout: process.stdout.close()
                with self.lock:
                    status='completed' if code==0 else 'failed'
                    with self.connect() as db:
                        changed=db.execute("UPDATE jobs SET status=?,returncode=? WHERE id=? AND status='running'",
                                           (status,code,identifier)).rowcount
                    if changed: self.event(identifier,'status',{'status':status,'returncode':code})
                    if changed: self.terminal(identifier,status,{'returncode':code})
                    self.processes.pop(identifier,None)
                    if resource in ('rtl','cuda'):
                        self.reserved.discard(resource)
                    if resource=='hybrid': self.reserved.difference_update({'rtl','cuda'})

    def persist(self,job_kind,identifier,event):
        from .store import Store
        payload=event['payload']; event_kind=event['kind']
        if not isinstance(payload,dict): raise ValueError('record payload must be an object')
        if event_kind=='candidate' and payload.get('valid') is not False: return None
        if event_kind=='model': kind='llm-model'
        elif event_kind=='candidate' or (job_kind in ('benchmark','llm-performance-suite') and 'tokens' in payload): kind='llm-generation'
        else: kind={'generate':'llm-generation','benchmark':'llm-benchmark','quality':'llm-quality','analyze':'llm-model',
            'gpu-generate':'llm-gpu-generation','hybrid-generate':'llm-hybrid-generation',
            'llm-performance-suite':'llm-benchmark-suite',
            'llm-optimize':'llm-decision','llm-train':'llm-policy','llm-evaluate':'llm-policy-evaluation',
            'llm-predictor-train':'llm-predictor','llm-predictor-evaluate':'llm-predictor-evaluation',
            'llm-promote':'llm-policy-deployment','llm-rollback':'llm-policy-deployment'}[job_kind]
        store=Store(self.root/'research')
        try:
            # Transport/job metadata is not part of model or policy identity.
            canonical={k:v for k,v in payload.items() if k not in ('record_id','run_id')}
            canonical.update(self.artifact_metadata(identifier))
            key=store.save(kind,canonical)
            store.save('llm-job-artifact',{'schema_version':1,'run_id':identifier,'artifact_id':key,
                'artifact_kind':kind,'event_kind':event_kind})
            return key
        finally: store.close()

    def artifact_metadata(self,identifier):
        try: payload=json.loads(self.get(identifier)['payload'])
        except ValueError: return {}
        return {k:payload[k] for k in ('previous_config','optimization_session_id') if k in payload}

    def terminal(self,identifier,status,details):
        from .store import Store
        row=self.get(identifier); store=Store(self.root/'research')
        try: store.save('llm-job-terminal',{'schema_version':1,'run_id':identifier,'status':status,
            'job_kind':row['kind'],'request':json.loads(row['payload']),'created':row['created'],
            'finished':time.time(),'details':details})
        finally: store.close()

def create_app(model_root='build/models',job_root='build/ide-jobs',start_worker=True):
    from fastapi import FastAPI,Request,HTTPException
    from fastapi.responses import FileResponse,StreamingResponse,JSONResponse,HTMLResponse
    from fastapi.staticfiles import StaticFiles
    jobs=Jobs(job_root,model_root,start_worker)
    @asynccontextmanager
    async def lifespan(app):
        yield; jobs.close()
    app=FastAPI(title='Malleable LLM Workbench',lifespan=lifespan); app.state.jobs=jobs
    @app.middleware('http')
    async def local_only(request:Request,call_next):
        host=request.headers.get('host',''); hostname=host.split(':')[0]
        origin=request.headers.get('origin')
        site=request.headers.get('sec-fetch-site')
        if hostname not in ('localhost','127.0.0.1','testserver') or (origin and origin!=f'http://{host}') or site not in (None,'same-origin','none'):
            return JSONResponse({'detail':'loopback same-origin access only'},status_code=403)
        return await call_next(request)
    @app.get('/api/models')
    def models():
        paths=[]
        for p in jobs.model_root.rglob('config.json'):
            if p.resolve().is_relative_to(jobs.model_root): paths.append({'path':str(p.parent),'name':p.parent.name})
            if len(paths)>=100: break
        return paths
    @app.get('/api/capabilities')
    def capabilities():
        from .llm.records import PERSONALITIES
        from dataclasses import asdict
        return {'personalities':[asdict(p) for p in PERSONALITIES.values()], 'default_backend':'rtl',
                'models':['Qwen3-0.6B','Qwen3.5-0.8B','LFM2.5-230M'],
                'gpu_hybrid':'explicitly-gated-by-standalone-release',
                'aws':'not-implemented','physical_metrics':'unavailable'}
    @app.get('/api/jobs')
    def listing(): return jobs.list()
    @app.get('/api/records/{kind}')
    def records(kind:str):
        if kind not in ('llm-generation','llm-gpu-generation','llm-hybrid-generation','llm-model','llm-quality','llm-decision','llm-policy','llm-policy-evaluation','llm-policy-deployment',
            'llm-predictor','llm-predictor-evaluation','llm-job-terminal','llm-job-artifact','llm-benchmark','llm-benchmark-suite','llm-optimization-session','llm-optimization-application'):
            raise HTTPException(400,'unknown record kind')
        from .store import Store
        store=Store(jobs.root/'research')
        try:
            return [{k:v for k,v in row.items() if k not in ('w1','w2','target_w1','target_w2','replay','counters','rows','results')}
                for row in store.entries(kind)]
        finally: store.close()
    @app.get('/api/artifact/{key}')
    def artifact(key:str):
        from .store import Store
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise HTTPException(404,'unknown artifact')
        store=Store(jobs.root/'research')
        try:
            path=store.root/'objects'/(key+'.json')
            if not path.is_file(): raise HTTPException(404,'unknown artifact')
            if path.stat().st_size>10_000_000:
                raise HTTPException(413,'artifact exceeds the bounded API response; use the local content-addressed file')
            return store.load(key)
        except (ValueError,FileNotFoundError) as e: raise HTTPException(404,str(e))
        finally: store.close()
    @app.post('/api/jobs/{kind}')
    async def create(kind:str,request:Request):
        try:
            body=await request.body()
            if len(body)>200000: raise HTTPException(413,'payload too large')
            return {'id':jobs.create(kind,json.loads(body))}
        except (ValueError,TypeError,KeyError) as e: raise HTTPException(400,str(e))
    @app.post('/api/optimization-sessions')
    async def optimize(request:Request):
        try:
            body=await request.json()
            if not isinstance(body,dict) or set(body)-{'current','window','auto_apply'}:
                raise ValueError('current, window and optional auto_apply required')
            if type(body.get('auto_apply',False)) is not bool: raise ValueError('auto_apply must be an explicit boolean')
            from .store import Store
            from .records import identity
            from .llm.optimization import decide,DecisionWindow
            store=Store(jobs.root/'research')
            try:
                current=store.load(body['current']); window=DecisionWindow(**body.get('window',{}))
                results=store.records('llm-generation'); qualities=store.records('llm-quality')
                decision=decide(results,qualities,current,window)
                session={'schema_version':1,'current_artifact':body['current'],'previous_configuration':decision['current'],
                    'recommended_configuration':decision['chosen'],'decision':decision,'automatic_application_requested':body.get('auto_apply',False),
                    'application':'recommendation-only','applied_job':None}
                if body.get('auto_apply',False): session['application']='automatic-request-pending'
                session_id=store.save('llm-optimization-session',session)
                application_id=None
                if body.get('auto_apply',False) and decision['chosen']!=decision['current']:
                    chosen=decision['chosen']; generation=next(r for r in reversed(results)
                        if r.get('configuration_id')==current.get('configuration_id')
                        and r.get('workload_identity_v2')==current.get('workload_identity_v2'))
                    candidate=next((r for r in reversed(results) if r.get('base_model_id')==current['base_model_id']
                        and r.get('valid') and r.get('backend')=='rtl'
                        and r.get('personality')+'/'+r.get('workload',{}).get('wformat')==chosen
                        and r.get('workload_identity_v2')==current.get('workload_identity_v2')),None)
                    if not candidate: raise ValueError('automatic application requires a measured, matched RTL candidate')
                    approval=next((q for q in reversed(qualities) if q.get('base_model_id')==current['base_model_id']
                        and q.get('variant_id')==candidate.get('variant_id') and q.get('personality')==candidate.get('personality')
                        and q.get('split')=='held-out' and q.get('selectable') is True),None)
                    if not approval: raise ValueError('automatic application requires held-out-approved candidate quality')
                    with jobs.connect() as db:
                        active=db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running') AND kind IN ('generate','benchmark','quality') LIMIT 1").fetchone()
                    if active: raise ValueError('automatic changes apply only between RTL decision windows')
                    model=store.load(generation['model_record'])
                    model_path=Path(model['path']).resolve()
                    if not model_path.is_relative_to(jobs.model_root): raise ValueError('model lineage path outside configured model root')
                    workload={k:v for k,v in generation['workload'].items() if v is not None}; personality,wformat=chosen.split('/',1)
                    workload.pop('schema_version',None)
                    workload.update(personality=personality,wformat=wformat,backend='rtl')
                    try:
                        job_id=jobs.create('generate',dict(model=str(model_path),**workload,
                            previous_config=decision['current'],optimization_session_id=session_id),trusted_internal=True)
                    except Exception as error:
                        store.save('llm-optimization-application',{'schema_version':1,
                            'optimization_session_id':session_id,'previous_configuration':decision['current'],
                            'applied_configuration':decision['chosen'],'application':'failed-to-queue',
                            'failure':{'type':type(error).__name__,'message':str(error)}})
                        raise
                    session.update(application='automatically-applied-between-windows',applied_job=job_id,
                        held_out_quality_id=identity(approval),previous_configuration=decision['current'])
                    application_id=store.save('llm-optimization-application',{
                        'schema_version':1,'optimization_session_id':session_id,'job_id':job_id,
                        'previous_configuration':decision['current'],'applied_configuration':decision['chosen'],
                        'held_out_quality_id':identity(approval),'application':'queued-between-windows'})
                elif body.get('auto_apply',False): session['application']='retained-current-no-change'
                return {'session_id':session_id,'application_id':application_id,**session}
            finally: store.close()
        except (ValueError,TypeError,KeyError,StopIteration) as e: raise HTTPException(400,str(e))
    @app.post('/api/job/{identifier}/cancel')
    def cancel(identifier:str):
        try: jobs.cancel(identifier); return {'status':'cancelled'}
        except ValueError as e: raise HTTPException(400,str(e))
    @app.post('/api/job/{identifier}/resume')
    def resume(identifier:str):
        raise HTTPException(410,'SSM/training jobs are retired; start a new stateless run')
    @app.get('/api/job/{identifier}/events')
    async def events(identifier:str,request:Request,after:int=0):
        try:
            jobs.get(identifier); cursor=max(after,int(request.headers.get('last-event-id','0')))
            if cursor<0: raise ValueError('invalid event cursor')
        except ValueError as e: raise HTTPException(400,str(e))
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows=jobs.events(identifier,cursor)
                for row in rows:
                    cursor=row['id']
                    yield f'id: {cursor}\ndata: {json.dumps(row)}\n\n'
                if not rows and jobs.get(identifier)['status'] not in ('queued','running'): break
                if not rows: yield ': heartbeat\n\n'
                await asyncio.sleep(.1)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
    @app.get('/api/job/{identifier}/log')
    def log(identifier:str):
        try: jobs.get(identifier)
        except ValueError as e: raise HTTPException(400,str(e))
        path=jobs.root/identifier/'stdout.txt'
        if not path.exists(): return {'text':''}
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-200000)); text=f.read().decode(errors='replace')
        return {'text':text}
    @app.get('/api/job/{identifier}/profile/{step}')
    def profile(identifier:str,step:int):
        try: jobs.get(identifier)
        except ValueError as e: raise HTTPException(400,str(e))
        if not 0<=step<2048: raise HTTPException(400,'invalid step')
        path=jobs.root/identifier/'traces'/f'step-{step:05d}'/'profile.json'
        if not path.is_file(): raise HTTPException(404,'awaiting trace')
        return FileResponse(path,media_type='application/json')
    @app.get('/api/job/{identifier}/trace/{step}')
    def trace(identifier:str,step:int,offset:int=0,limit:int=200):
        try: jobs.get(identifier)
        except ValueError as e: raise HTTPException(400,str(e))
        if not 0<=step<2048 or not 0<=offset<=100000000 or not 1<=limit<=500: raise HTTPException(400,'invalid trace segment')
        path=jobs.root/identifier/'traces'/f'step-{step:05d}'/'trace.txt'
        if not path.is_file(): raise HTTPException(404,'awaiting trace')
        # Byte offsets avoid rereading the whole trace for late segments. Only
        # complete bounded lines enter the browser; full evidence stays on disk.
        with path.open('rb') as stream:
            stream.seek(min(offset,path.stat().st_size))
            lines=[]
            for _ in range(limit):
                line=stream.readline(16384)
                if not line: break
                lines.append(line.decode(errors='replace'))
            return {'mode':'recorded','offset':offset,'next_offset':stream.tell(),'lines':lines}
    @app.get('/api/job/{identifier}/instructions/{step}')
    def instructions(identifier:str,step:int):
        try: jobs.get(identifier)
        except ValueError as e: raise HTTPException(400,str(e))
        if not 0<=step<2048: raise HTTPException(400,'invalid step')
        path=jobs.root/identifier/'traces'/f'step-{step:05d}'/'prog_0.hex'
        if not path.is_file(): raise HTTPException(404,'awaiting compiled instructions')
        from .llm import upstream
        from opentpu import isa
        import numpy as np
        words=np.fromiter((int(t,16) for t in path.read_text().split()),dtype=np.uint32)
        return {'mode':'recorded','step':step,'source':'compiler instruction stream',
            'instructions':[f'{index//8}: {isa.Instr.decode(words[index:index+8])}'
                for index in range(0,min(len(words),2048*8),8)]}
    @app.get('/lens')
    def lens(run:str='',step:int=0):
        url=None
        if run:
            try: jobs.get(run)
            except ValueError as e: raise HTTPException(400,str(e))
            if not 0<=step<2048: raise HTTPException(400,'invalid step')
            url=f'/api/job/{run}/profile/{step}'
        html=(Path(__file__).parents[1]/'third_party/opentpu/opentpu/lens_app.html').read_text()
        html=html.replace('/*__URL__*/null',json.dumps(url))
        html=html.replace('at ${p.clock_mhz || 100} MHz','assumed ${p.clock_mhz || 100} MHz (projection)')
        html=html.replace('at the board clock','at an assumed clock (not board measurement)')
        html=html.replace('MHz core clock.','MHz assumed clock; not physical timing.')
        return HTMLResponse(html)
    dist=Path(__file__).parent/'web/dist'
    if dist.exists(): app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/')
    def index():
        if not (dist/'index.html').exists(): return JSONResponse({'detail':'Build UI: make ui'},status_code=503)
        return FileResponse(dist/'index.html')
    return app

def main():
    import uvicorn
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8765)
    p.add_argument('--model-root',default='build/models'); p.add_argument('--job-root',default='build/ide-jobs')
    a=p.parse_args(); uvicorn.run(create_app(a.model_root,a.job_root),host='127.0.0.1',port=a.port)

if __name__=='__main__': main()
