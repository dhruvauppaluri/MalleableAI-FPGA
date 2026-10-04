"""One immutable eight-hour window for screening and 20 indexed RTL runs."""
import argparse
from datetime import datetime,timezone
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from malleable.llm.models import inspect,digest
from malleable.llm.candidates import source_state
from malleable.llm.experiments import validate_performance_manifest,validate_benchmark_result
from malleable.llm.quality import reference_contract
from malleable.records import identity
from malleable.store import Store
from tools.create_llm_performance_manifest_v2 import create
from tools.run_performance_indices import run_indices
RELEASE=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'
MODELS={'qwen3':'Qwen3-0.6B','qwen35':'Qwen3.5-0.8B'}

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def window(root,now=None):
    now=time.time() if now is None else now;file=root/'control.json'
    if not file.exists():
        write(file,{'schema_version':1,'mode':'overnight','started_unix':now,'dispatch_deadline_unix':now+28800,
            'authorization_sha256':digest(root/'authorization.json'),'started_utc':datetime.fromtimestamp(now,timezone.utc).isoformat()})
    data=json.loads(file.read_text())
    if (data.get('mode')!='overnight' or data.get('dispatch_deadline_unix')!=data.get('started_unix',0)+28800
        or data.get('authorization_sha256')!=digest(root/'authorization.json')):
        raise ValueError('persisted window differs; never reset the deadline')
    return data

def completed(manifest,store_path):
    store=Store(store_path/'research');done={}
    try:
        for attempt in store.records('llm-benchmark-attempt'):
            if attempt.get('manifest_id')==identity(manifest) and attempt.get('status')=='completed':
                index=attempt['run_index'];validate_benchmark_result(manifest,index,attempt['result']);done[index]=attempt['result']
    finally:store.close()
    return done

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--unit',required=True)
    p.add_argument('--resume-from',type=Path,help='previous immutable window; verify/reuse successes, never repeat failed attempts silently')
    args=p.parse_args();os.chdir(ROOT);root=args.root.resolve()
    auth=json.loads((root/'authorization.json').read_text())
    if auth.get('mode')!='overnight' or auth.get('window_hours')!=8 or auth.get('models')!=list(MODELS):
        raise ValueError('explicit eight-hour remaining-benchmarks authorization required')
    with (ROOT/'build/zephyrus-jobs/precision-attribution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=source_state()
        if frozen['status']:raise ValueError('clean source required')
        write(root/'job.json',{'source':frozen,'unit':args.unit,'pid':os.getpid(),'authorization':auth,'command':sys.argv})
        shutil.copyfile(__file__,root/'runner.py');stage='preflight';manifests={};infos={}
        previous=args.resume_from.resolve() if args.resume_from else None
        old_stores=json.loads((previous/'stores.json').read_text()) if previous and (previous/'stores.json').exists() else {}
        stores={family:Path(old_stores.get(family,str(root/family/'store'))) for family in MODELS}
        write(root/'stores.json',{k:str(v) for k,v in stores.items()})
        def stable():
            if source_state()!=frozen:raise ValueError('source changed; stop dispatch')
        def terminal(status,next_action=None):
            data={'status':status,'stage':stage,'next':next_action,'source':frozen,
                'completed_indices':{family:sorted(completed(m,stores[family])) for family,m in manifests.items()},
                'control':json.loads((root/'control.json').read_text()) if (root/'control.json').exists() else None}
            write(root/'terminal.json',data);print(json.dumps(data),flush=True)
        def allowed():
            stable();control=window(root)
            return time.time()<control['dispatch_deadline_unix']
        try:
            # Verify LFM's existing ten results; no new LFM execution.
            lfm_path=RELEASE/'perf/lfm2-performance-manifest.json'
            lfm=json.loads(lfm_path.read_text());validate_performance_manifest(lfm,inspect(lfm['model_path']))
            lfm_done=completed(lfm,RELEASE/'perf/lfm2-01/store')
            if len(lfm_done)!=10:raise ValueError('existing LFM ten-run report is incomplete')
            write(root/'lfm-reuse.json',{'manifest_sha256':digest(lfm_path),'verified_indices':sorted(lfm_done),'rerun':False})
            for family,name in MODELS.items():
                model=ROOT/'build/models'/name;info=inspect(model,128);infos[family]=info
                suite=ROOT/'build/models/quality'/f'{family}-frozen-v2.json'
                screens=[]
                for fmt in ('int4','fp4'):
                    stage=family+'-'+fmt+'-validation';attempt=root/stage
                    if previous and (previous/stage/'summary.json').exists():
                        old=previous/stage;summary=json.loads((old/'summary.json').read_text())
                        if summary.get('status')!='completed' or digest(old/'result.json')!=summary.get('result_sha256'):
                            raise ValueError('cannot reuse invalid prior screen')
                        screens.append(old/'result.json')
                        write(root/('reused-'+stage+'.json'),{'source':str(old),'result_sha256':summary['result_sha256']})
                        continue
                    if previous and (previous/stage).exists():raise ValueError('prior screen failed/incomplete; needs explicit diagnosed retry')
                    if attempt.exists():raise ValueError('fresh screen attempt required; do not overwrite')
                    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                    if available<2*info['fp32_tensor_bytes']+2*1024**3:raise ValueError('memory preflight failed')
                    if not allowed():terminal('deadline-limited','start a separately authorized window; preserve completed evidence');return 3
                    attempt.mkdir();control=window(root);started=time.monotonic()
                    cmd=[sys.executable,'-u','-m','malleable.llm.cli','quality','--model',str(model),'--suite',str(suite),
                        '--split','validation','--context','128','--personality','balanced','--wformat',fmt,
                        '--max-host-gib','16','--store',str(attempt/'store')]
                    write(attempt/'job.json',{'source':frozen,'command':cmd,'control':control,'available_memory':available})
                    result=None
                    with (attempt/'worker.log').open('x') as log:
                        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        write(attempt/'child.json',{'pid':child.pid})
                        for line in child.stdout:
                            log.write(line);log.flush()
                            try:event=json.loads(line)
                            except ValueError:continue
                            if event.get('kind')=='result':result=event['payload']
                            elif event.get('kind') in ('phase','quality-progress','failure'):print(json.dumps({'stage':stage,**event}),flush=True)
                        code=child.wait()
                    stable()
                    if code or result is None:raise RuntimeError('screen execution failed; preserve attempt, no automatic retry')
                    write(attempt/'result.json',result)
                    if result.get('record_id')!=identity({k:v for k,v in result.items() if k!='record_id'}):raise ValueError('noncanonical quality result')
                    if (result.get('toolchain',{}).get('contract')!=reference_contract(info) or result.get('split')!='validation'
                        or result.get('suite_file_hash')!=digest(suite) or result.get('wformat')!=fmt):raise ValueError('screen lineage mismatch')
                    write(attempt/'summary.json',{'status':'completed','source':frozen,'result_sha256':digest(attempt/'result.json'),
                        'elapsed_seconds':time.monotonic()-started,'quality_passed':result['passed'],'agreement':result['agreement']})
                    screens.append(attempt/'result.json')
                case=RELEASE/'quality-recovery-pilot-01/int8/case-00' if family=='qwen3' else None
                manifest_path=root/family/'manifest.json'
                manifests[family]=create(model,manifest_path,screens,case,42)
                print(json.dumps({'stage':family+'-manifest-frozen','formats':[r['wformat'] for r in manifests[family]['runs']]}),flush=True)
            pilots={}
            for index in range(10):
                for family,manifest in manifests.items():
                    stage=f'{family}-index-{index:02d}'
                    if index in completed(manifest,stores[family]):
                        print(json.dumps({'stage':stage,'status':'reused-verified'}),flush=True);continue
                    if not allowed():terminal('deadline-limited','new run window required for remaining indices');return 3
                    info=inspect(manifest['model_path'],128)
                    if any(info[k]!=infos[family][k] for k in ('base_model_id','tokenizer_id','weight_files','tokenizer_files')):
                        raise ValueError('checkpoint/tokenizer changed')
                    control=window(root);began=time.monotonic()
                    code=run_indices(root/family/'manifest.json',stores[family],control['dispatch_deadline_unix'],root/(stage+'.jsonl'),[index])
                    if code:
                        terminal('deadline-limited' if code==3 else 'failed','preserve indexed results and inspect failure');return code
                    elapsed=time.monotonic()-began
                    if index==0:
                        pilots[family]=elapsed
                        write(root/(family+'-pilot-timing.json'),{'measured_first_index_seconds':elapsed,
                            'estimated_remaining_nine_seconds':9*elapsed,'provenance':'estimate only; tape lengths/personality/stress vary'})
                    print(json.dumps({'stage':stage,'completed_seconds':elapsed,'pilot_seconds':pilots}),flush=True)
            terminal('completed','assemble performance evidence with preserved LFM report; controller quality approvals remain separate')
            return 0
        except BaseException:
            write(root/'failure.json',{'stage':stage,'traceback':traceback.format_exc(),'source':source_state()})
            if not (root/'terminal.json').exists():terminal('failed','inspect preserved failure; no automatic retry')
            raise

if __name__=='__main__':sys.exit(main())
