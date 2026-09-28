"""Allowlisted local CLI; JSON-lines output is also the job event protocol."""
import argparse
import json
from pathlib import Path
import sys
from .records import GenerationWorkload

def emit(kind,payload):
    print(json.dumps({'kind':kind,'payload':payload},allow_nan=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['analyze','generate','benchmark','quality','optimize',
        'train','evaluate','promote','rollback','predictor-train','predictor-evaluate',
        'release-check','full-release-check','gpu-generate','hybrid-generate','performance-suite'])
    p.add_argument('--model'); p.add_argument('--verifier'); p.add_argument('--store',default='build/llm-runs')
    p.add_argument('--manifest')
    p.add_argument('--standalone-manifest')
    p.add_argument('--messages'); p.add_argument('--input-tokens')
    p.add_argument('--suite'); p.add_argument('--split',default='validation')
    p.add_argument('--current'); p.add_argument('--window'); p.add_argument('--episodes')
    p.add_argument('--checkpoint'); p.add_argument('--report'); p.add_argument('--previous')
    p.add_argument('--passes',type=int,default=20)
    p.add_argument('--depth',type=int,default=4); p.add_argument('--dtype',choices=['float16','float32'],default='float16')
    for name,default,typ in [('prompt','Hello',str),('prompt_format','chat',str),('max_new',16,int),('context',2048,int),
        ('seed',0,int),('backend','rtl',str),('personality','balanced',str),('wformat','int8',str),
        ('latency',20,int),('stall_percent',20,int),('bandwidth_percent',100,int),
        ('max_host_gib',16.,float),('clock_hz',None,float)]:
        p.add_argument('--'+name.replace('_','-'),default=default,type=typ)
    a=p.parse_args()
    try:
        if a.command=='release-check':
            from .release import check
            emit('result',check(a.manifest)); return
        if a.command=='full-release-check':
            from .full_release import check
            emit('result',check(a.manifest)); return
        if a.command in ('gpu-generate','hybrid-generate'):
            from ..store import Store
            if not a.model: raise ValueError('--model is required')
            messages=json.loads(a.messages) if a.messages else None
            if a.command=='gpu-generate':
                from .cuda import gpu_generate
                result=gpu_generate(a.model,a.prompt,a.max_new,a.context,a.dtype,a.prompt_format,
                    messages,a.standalone_manifest,emit=emit)
                kind='llm-gpu-generation'
            else:
                if not a.verifier: raise ValueError('--verifier is required for hybrid-generate')
                from .cuda import hybrid_generate
                result=hybrid_generate(a.model,a.verifier,a.prompt,a.max_new,a.context,a.depth,
                    a.personality,a.wformat,a.dtype,a.prompt_format,messages,a.standalone_manifest,
                    trace_root=Path(a.store)/'traces',emit=emit)
                kind='llm-hybrid-generation'
            store=Store(Path(a.store)/'research')
            try: result['record_id']=store.save(kind,result)
            finally: store.close()
            emit('result',result); return
        if a.command=='performance-suite':
            from .experiments import run_performance_manifest
            if not a.manifest: raise ValueError('--manifest is required for performance-suite')
            run_performance_manifest(a.manifest,a.store,emit); return
        if a.command in ('optimize','train','evaluate','promote','rollback','predictor-train','predictor-evaluate'):
            from ..store import Store
            from .optimization import decide,DecisionWindow
            from .experiments import episodes_from_store
            from . import learning
            store=Store(a.store)
            try:
                if a.command.startswith('predictor-'):
                    from .optimization import Predictor
                    requests=json.loads(Path(a.episodes).read_text())
                    if not isinstance(requests,list) or not requests: raise ValueError('source experiment hashes required')
                    rows=[store.load(key) for key in requests]
                    predictor=Predictor()
                    if a.command=='predictor-train':
                        predictor.fit(rows)
                        result={'schema_version':1,'algorithm':'ridge-regression','coefficients':predictor.coefficients,
                            'training_models':predictor.training_models,'training_rmse_cycles':predictor.rmse,
                            'evidence_ids':requests,'provenance':'measured-rtl-training'}
                        kind='llm-predictor'
                    else:
                        checkpoint=store.load(a.checkpoint)
                        if checkpoint.get('algorithm')!='ridge-regression' or checkpoint.get('schema_version')!=1:
                            raise ValueError('invalid predictor checkpoint')
                        if set(checkpoint['training_models']) & {r['base_model_id'] for r in rows}:
                            raise ValueError('predictor base-model train/evaluation leakage')
                        predictor.coefficients=checkpoint['coefficients']; predictor.rmse=checkpoint['training_rmse_cycles']
                        predictions=[]
                        for row in rows:
                            if not row.get('valid') or row.get('backend')!='rtl' or not row.get('counters') or any(
                                s.get('validation')!='bit-exact-dram-and-tmem' for s in row['counters']):
                                raise ValueError('predictor evaluation requires validated RTL evidence')
                            measured=sum(s['cycles'] for s in row['counters']); prediction=predictor.predict(row)
                            predictions.append(dict(prediction,measured_cycles=measured,error_cycles=prediction['cycles']-measured))
                        result={'schema_version':1,'checkpoint':a.checkpoint,'evidence_ids':requests,'predictions':predictions,
                            'split':a.split,'provenance':'estimated-predictions-versus-measured-rtl'}
                        kind='llm-predictor-evaluation'
                elif a.command=='optimize':
                    result=decide(store.records('llm-generation'),store.records('llm-quality'),
                        store.load(a.current),DecisionWindow(**json.loads(Path(a.window).read_text()) if a.window else {}))
                    kind='llm-decision'
                elif a.command in ('train','evaluate'):
                    episodes=episodes_from_store(store,json.loads(Path(a.episodes).read_text()))
                    if a.command=='train':
                        result=learning.train(episodes,a.seed,a.passes,store.load(a.checkpoint) if a.checkpoint else None)
                        kind='llm-policy'
                    else:
                        result=learning.evaluate(store.load(a.checkpoint),episodes,range(a.seed,a.seed+5),split=a.split)
                        kind='llm-policy-evaluation'
                elif a.command=='promote':
                    result=learning.promotion(store.load(a.checkpoint),store.load(a.report),a.previous)
                    kind='llm-policy-deployment'
                else:
                    deployment=store.load(a.report); previous=deployment.get('previous_policy')
                    if not previous: raise ValueError('no previous policy for rollback')
                    learning.validate_checkpoint(store.load(previous))
                    result={'schema_version':1,'active_policy':previous,'previous_policy':deployment['active_policy'],
                            'rollback_of':a.report,'promotion':'explicit-rollback'}
                    kind='llm-policy-deployment'
                key=store.save(kind,result); emit('result',dict(result,record_id=key)); return
            finally: store.close()
        if not a.model: raise ValueError('--model is required')
        if a.command=='quality':
            from .quality import evaluate
            from ..store import Store
            result=evaluate(a.model,a.suite,a.wformat,a.split,a.personality,a.max_host_gib,emit=emit,context=a.context)
            store=Store(Path(a.store)/'research')
            try: result['record_id']=store.save('llm-quality',result)
            finally: store.close()
            emit('result',result)
        elif a.command=='analyze':
            from .models import inspect
            emit('result',inspect(a.model,a.context))
        else:
            from .runtime import generate
            keys=GenerationWorkload.__dataclass_fields__
            if a.messages: a.messages=json.loads(a.messages)
            if a.input_tokens: a.input_tokens=json.loads(a.input_tokens)
            w=GenerationWorkload(**{k:getattr(a,k) for k in keys if hasattr(a,k)})
            if a.command=='benchmark':
                if w.input_tokens is None: raise ValueError('configuration benchmarks require a fixed --input-tokens tape')
                from .experiments import benchmark
                emit('result',benchmark(Path(a.model).resolve(),w,Path(a.store),emit))
            else: generate(Path(a.model).resolve(),w,Path(a.store),emit)
    except Exception as error:
        emit('failure',{'type':type(error).__name__,'message':str(error)})
        sys.exit(1)

if __name__=='__main__': main()
