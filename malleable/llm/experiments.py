"""Reproducible local sweeps and evidence-derived sequential learning episodes."""
from dataclasses import replace
from .runtime import generate
from .records import PERSONALITIES
from .optimization import config_key,validated_variants,observations
from ..records import identity
from .candidates import variant_id

def manifest_candidate(data,model_info):
    """A candidate manifest is INT8-only and must re-verify its saved derived lineage."""
    case=data.get('candidate_case')
    if case is None:
        if data.get('derived_candidate') is not None: raise ValueError('derived candidate lineage requires a candidate case')
        return None
    from .candidates import read_derived_candidate,derived_record
    derived=read_derived_candidate(case,model_info)
    if data.get('derived_candidate')!=derived_record(derived):
        raise ValueError('performance manifest derived candidate lineage mismatch')
    return derived

def validate_performance_manifest(data,model_info):
    if data.get('schema_version')!=1 or data.get('base_model_id')!=model_info['base_model_id'] \
        or data.get('tokenizer_id')!=model_info['tokenizer_id']:
        raise ValueError('performance manifest model/tokenizer lineage mismatch')
    derived=manifest_candidate(data,model_info)
    if derived and any(r.get('wformat')!='int8' for r in data.get('runs',[])):
        raise ValueError('derived INT8 candidates cannot be combined with INT4/FP4 comparisons')
    from .quality import gate
    approvals={}
    for evidence in data.get('quality_evidence',[]):
        record=evidence.get('record')
        if not isinstance(record,dict) or identity(record)!=evidence.get('id'):
            raise ValueError('quality evidence hash mismatch in performance manifest')
        if record.get('base_model_id')!=model_info['base_model_id'] or record.get('personality')!='balanced' \
            or record.get('tokenizer_id')!=model_info['tokenizer_id'] or record.get('context')!=128 \
            or record.get('head_format')!='int8' or not record.get('configuration_id') or not record.get('suite_file_hash') \
            or set(record.get('suite_hashes',{}))!={'calibration','validation','held-out'} \
            or record.get('split')!='validation' or record.get('suite_frozen') is not True \
            or record.get('samples',0)<1024 or record.get('target_count',0)<1024 \
            or not gate(record['float_nll'],record['candidate_nll'],record['agreement'])['passed']:
            raise ValueError('performance format lacks frozen validation approval')
        approvals[(record['variant_id'],record['personality'])]=evidence['id']
    evidence_rows=data.get('quality_evidence',[])
    if data.get('quality_evidence_ids',[])!=[e.get('id') for e in evidence_rows]:
        raise ValueError('quality evidence ID list does not match embedded records')
    if type(data.get('seed')) is not int or not 0<=data['seed']<2**31:
        raise ValueError('reproducible benchmark seed required')
    runs=data.get('runs')
    if not isinstance(runs,list) or len(runs)!=10: raise ValueError('staged performance suite requires exactly ten runs per model')
    names=set()
    for run in runs:
        if run.get('workload') not in ('short-a','short-b','axi-stress'): raise ValueError('unknown performance workload')
        if not isinstance(run.get('input_tokens'),list) or not run['input_tokens'] \
            or any(type(t) is not int or t<0 for t in run['input_tokens']): raise ValueError('fixed integer token tape required')
        if type(run.get('context')) is not int or run['context']%128 or not 128<=run['context']<=2048 \
            or len(run['input_tokens'])>run['context']: raise ValueError('invalid performance context/tape size')
        if identity(run['input_tokens'])!=run.get('input_token_hash'): raise ValueError('fixed tape hash mismatch')
        if run.get('personality') not in PERSONALITIES or run.get('wformat') not in ('int8','int4','fp4'):
            raise ValueError('unsupported performance configuration')
        scenario=run.get('memory_scenario')
        if scenario not in ('baseline','stress'): raise ValueError('explicit memory scenario required')
        if scenario=='baseline' and (run.get('latency'),run.get('stall_percent'),run.get('bandwidth_percent'))!=(20,20,100):
            raise ValueError('baseline AXI scenario must be 20/20/100')
        if scenario=='stress' and (run.get('latency'),run.get('stall_percent'),run.get('bandwidth_percent'))!=(100,50,50):
            raise ValueError('stress AXI scenario must be 100/50/50')
        names.add((run['workload'],run['personality'],run['wformat']))
    baseline=[r for r in runs if r['memory_scenario']=='baseline' and r['wformat']=='int8'
              and r['workload'] in ('short-a','short-b')]
    matrix=[(r['workload'],r['personality'],r['wformat']) for r in baseline]
    expected={(workload,p,'int8') for workload in ('short-a','short-b') for p in PERSONALITIES}
    if len(baseline)!=8 or set(matrix)!=expected: raise ValueError('the eight fixed INT8 personality/workload comparisons are required exactly once')
    prompt_hashes=data.get('fixed_prompt_hashes',{})
    for name in ('short-a','short-b'):
        rows=[r for r in runs if r['workload']==name]
        if not rows or prompt_hashes.get(name)!=rows[0]['input_token_hash'] \
            or any(r['input_token_hash']!=rows[0]['input_token_hash'] for r in rows):
            raise ValueError('each workload must reuse its manifest-hashed fixed tape')
    extras=[r for r in runs if (r['workload'],r['personality'],r['wformat']) not in expected]
    if len(extras)!=2: raise ValueError('two quality/stress comparison runs required')
    if all(r['wformat'] in ('int4','fp4') and r['personality']=='balanced' and r['memory_scenario']=='baseline' for r in extras):
        if {r['wformat'] for r in extras}!={'int4','fp4'}: raise ValueError('both balanced INT4 and FP4 variants required')
        for run in extras:
            variant=variant_id(model_info['base_model_id'],run['wformat'])
            if (variant,'balanced') not in approvals: raise ValueError('quantized comparison missing validation approval')
            if not any(e['record'].get('context')==run['context'] and e['record'].get('wformat')==run['wformat']
                and e['record'].get('variant_id')==variant for e in evidence_rows):
                raise ValueError('quantized comparison context/format differs from quality approval')
    elif not all(r['wformat']=='int8' and r['personality'] in ('balanced','compute')
                 and r['memory_scenario']=='stress' and r['workload']=='axi-stress' for r in extras):
        raise ValueError('extras must be balanced INT4/FP4 or balanced/compute INT8 under stress')
    elif {r['personality'] for r in extras}!={'balanced','compute'}:
        raise ValueError('both balanced and compute stress configurations required')
    if len(runs)!=10: raise ValueError('exact 10-run schedule required')
    return True

def validate_benchmark_result(data,index,result):
    """Bind reusable measured evidence to every declared indexed run setting."""
    row=data['runs'][index]; w=result.get('workload',{})
    p=PERSONALITIES[row['personality']]
    cfg=result.get('config',{}); metadata=result.get('microarchitecture',{})
    expected={'personality':row['personality'],'wformat':row['wformat'],'context':row['context'],
        'seed':data['seed'],'backend':'rtl','input_tokens':row['input_tokens'],
        **({'candidate_case':data['candidate_case']} if data.get('candidate_case') else {}),
        'latency':row['latency'],'stall_percent':row['stall_percent'],'bandwidth_percent':row['bandwidth_percent']}
    if (result.get('status')!='completed' or result.get('valid') is not True or result.get('backend')!='rtl'
        or result.get('base_model_id')!=data['base_model_id'] or result.get('tokenizer_id')!=data['tokenizer_id']
        or result.get('benchmark_manifest_id')!=identity(data) or result.get('benchmark_run_index')!=index
        or result.get('benchmark_workload')!=row['workload'] or result.get('memory_scenario')!=row['memory_scenario']
        or result.get('input_token_hash')!=row['input_token_hash'] or result.get('personality')!=row['personality']
        or result.get('generation_mode')!='fixed-token-tape' or result.get('tokens')!=[]
        or result.get('prompt_tokens')!=len(row['input_tokens']) or any(w.get(k)!=v for k,v in expected.items())
        or result.get('variant_id')!=variant_id(data['base_model_id'],row['wformat'],data.get('derived_candidate'))
        or result.get('derived_candidate')!=data.get('derived_candidate')
        or metadata.get('schema_version')!=1 or metadata.get('parameters')!=p.uarch
        or cfg.get('MCOLS')!=p.matrix_columns or cfg.get('LANES')!=p.vector_lanes
        or result.get('configuration_id')!=identity({'config':cfg,'uarch':p.uarch})
        or len(result.get('counters',[]))!=len(row['input_tokens'])
        or any(c.get('validation')!='bit-exact-dram-and-tmem' or c.get('cycles',0)<=0 for c in result['counters'])):
        raise ValueError('invalid or mismatched fixed-tape benchmark result at index '+str(index))
    return True


def run_performance_manifest(path,root,emit=lambda *_:None,run_index=None):
    import json
    from pathlib import Path
    from .models import inspect
    from .records import GenerationWorkload
    from ..store import Store
    import uuid
    data=json.loads(Path(path).read_text()); model=Path(data['model_path']).resolve()
    info=inspect(model)
    validate_performance_manifest(data,info)
    if run_index is not None and (type(run_index) is not int or not 0<=run_index<10):
        raise ValueError('benchmark run index must be 0..9')
    root=Path(root); store=Store(root/'research'); manifest_id=identity(data)
    try:
        completed={}
        for attempt in store.records('llm-benchmark-attempt'):
            if attempt.get('manifest_id')==manifest_id and attempt.get('status')=='completed':
                index=attempt['run_index']; validate_benchmark_result(data,index,attempt['result'])
                completed[index]=attempt['result']
        for index in ([run_index] if run_index is not None else range(10)):
            if index in completed:
                emit('candidate',{'index':index,'status':'reused-verified'}); continue
            row=data['runs'][index]; attempt_id=uuid.uuid4().hex
            attempt={'schema_version':1,'manifest_id':manifest_id,'run_index':index,'attempt_id':attempt_id}
            try:
                result=_run_performance_index(data,model,row,index,root/'attempts'/attempt_id,emit)
                validate_benchmark_result(data,index,result)
                attempt.update(status='completed',result=result); completed[index]=result
            except Exception as error:
                attempt.update(status='failed',failure=str(error),provenance='failed-run')
                store.save('llm-benchmark-attempt',attempt); emit('candidate',attempt)
                if run_index is not None: raise
                continue
            store.save('llm-benchmark-attempt',attempt)
        if len(completed)!=10:
            progress={'schema_version':1,'manifest_id':manifest_id,'status':'incomplete',
                'completed_indices':sorted(completed),'remaining_indices':sorted(set(range(10))-set(completed)),
                'provenance':'indexed-run progress; not a complete performance report'}
            store.save('llm-benchmark-progress',progress); emit('result',progress); return progress
        report={'schema_version':1,'manifest_id':manifest_id,'base_model_id':info['base_model_id'],
            'tokenizer_id':info['tokenizer_id'],'seed':data['seed'],'runs':[completed[i] for i in range(10)],
            'completed':10,'failed':0,
            'provenance':'measured-rtl-fixed-token-tapes; simulator AXI scenarios, not physical memory'}
        key=store.save('llm-benchmark-suite',report)
        emit('result',dict(report,record_id=key)); return report
    finally: store.close()


def _run_performance_index(data,model,row,index,root,emit):
    from .records import GenerationWorkload
    workload=GenerationWorkload(prompt='fixed-token benchmark tape',prompt_format='raw',
        max_new=8,context=row['context'],seed=data['seed'],backend='rtl',personality=row['personality'],
        wformat=row['wformat'],latency=row['latency'],stall_percent=row['stall_percent'],
        bandwidth_percent=row['bandwidth_percent'],input_tokens=row['input_tokens'],
        candidate_case=data.get('candidate_case'))
    emit('candidate',{'index':index,'total':10,'workload':row['workload'],
        'personality':row['personality'],'wformat':row['wformat'],'status':'running'})
    result=generate(model,workload,root,lambda kind,payload:emit(kind,payload) if kind!='result' else None)
    result.update(benchmark_manifest_id=identity(data),benchmark_run_index=index,
        benchmark_workload=row['workload'],memory_scenario=row['memory_scenario'],
        input_token_hash=row['input_token_hash'])
    emit('candidate',result); return result

def benchmark(model,workload,root,emit=lambda *_:None):
    from pathlib import Path
    results=[]
    for personality in PERSONALITIES:
        for wformat in ('int8','int4','fp4'):
            if workload.candidate_case is not None and wformat!='int8': continue
            selected=replace(workload,personality=personality,wformat=wformat)
            emit('candidate',{'personality':personality,'wformat':wformat,'status':'running'})
            try:
                result=generate(model,selected,Path(root)/(personality+'-'+wformat),emit)
                results.append(result)
            except Exception as error:
                failure={'schema_version':1,'personality':personality,'wformat':wformat,
                         'valid':False,'failure':str(error),'workload_id':selected.workload_id}
                emit('candidate',failure); results.append(failure)
    return {'schema_version':1,'results':results,'selection':'quality gate required; failures retained'}

def episodes_from_store(store,requests):
    """Requests group source IDs into episodes; raw timing tables are not trusted."""
    from .optimization import DecisionWindow
    qualities=store.records('llm-quality'); episodes=[]
    for request_episode in requests:
        episode=[]
        for request in request_episode:
            rows=[store.load(key) for key in request['evidence_ids']]
            if not rows: raise ValueError('source experiment evidence required')
            base=rows[0]['base_model_id']
            if any(r['base_model_id']!=base for r in rows): raise ValueError('mixed base-model window')
            # Execution settings may differ; comparison workload/input hashes may not.
            signatures={identity({k:v for k,v in r['workload'].items() if k not in
                ('personality','wformat','backend','clock_hz')}) for r in rows}
            if len(signatures)!=1: raise ValueError('mismatched compared workloads')
            from .quality import quality_matches
            groups=observations(rows,base)
            reference=next((q for q in reversed(qualities) if quality_matches(q,rows[0])),None)
            legal={k:samples for k,samples in groups.items() if k.endswith('/int8') and reference
                and all(any(quality_matches(q,r,frozen_reference=reference) for q in qualities) for _,r in samples)}
            if not legal: raise ValueError('no correctness/quality-validated candidates')
            import statistics
            window=request.get('window',{}); DecisionWindow(**window)
            models=[r for r in store.records('llm-model') if r['base_model_id']==base]
            if not models: raise ValueError('model inspection lineage missing')
            model=models[-1]; cfg=model['config'].get('text_config',model['config']); exemplar=rows[0]
            from .optimization import Predictor
            predictor_record=store.load(request['predictor_id'])
            predictor=Predictor.from_checkpoint(predictor_record)
            episode.append({'base_model_id':base,'family':models[-1]['family'],
                'service_cycles':{k:statistics.median(s[0] for s in v) for k,v in legal.items()},
                'predicted_cycles':{k:predictor.predict(v[-1][1])['cycles'] for k,v in legal.items()},
                'predictor_id':request['predictor_id'],'predictor_training_models':predictor.training_models,
                'evidence_ids':request['evidence_ids'],'window':window,
                'features':[model['fp32_tensor_bytes']/2**32,cfg.get('hidden_size',0)/4096,
                    cfg.get('num_hidden_layers',0)/128,exemplar.get('prompt_tokens',0)/2048,
                    (0 if exemplar.get('generation_mode')=='fixed-token-tape' else exemplar['workload']['max_new'])/256,
                    exemplar['config']['DRAM_BYTES']/2**30],
                'provenance':'estimated-window-from-measured-rtl'})
        episodes.append(episode)
    return episodes
