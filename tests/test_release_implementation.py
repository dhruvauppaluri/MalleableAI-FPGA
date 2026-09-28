"""Fail-closed lineage, resumability and predictor regressions without checkpoints."""
import copy
import json
import subprocess
from unittest.mock import patch
import pytest
from malleable.records import identity
from malleable.store import Store
from malleable.llm.records import PERSONALITIES
from malleable.llm.optimization import (Predictor,PREDICTOR_FEATURES,PREDICTOR_TRAINING_MODELS,
                                      PREDICTOR_HELD_OUT_MODEL)
from malleable.llm.experiments import run_performance_manifest,validate_benchmark_result
from malleable.llm.quality import quality_matches
from malleable.llm.cuda import driver_metadata


def schedule():
    runs=[]
    for name in ('short-a','short-b'):
        for personality in PERSONALITIES:
            runs.append(dict(workload=name,personality=personality,wformat='int8',context=128,
                input_tokens=[1,2],input_token_hash=identity([1,2]),memory_scenario='baseline',
                latency=20,stall_percent=20,bandwidth_percent=100))
    for personality in ('balanced','compute'):
        runs.append(dict(workload='axi-stress',personality=personality,wformat='int8',context=128,
            input_tokens=[1,2],input_token_hash=identity([1,2]),memory_scenario='stress',
            latency=100,stall_percent=50,bandwidth_percent=50))
    return dict(schema_version=1,base_model_id='base',tokenizer_id='tok',seed=42,model_path='/unused',
        runs=runs,fixed_prompt_hashes={'short-a':identity([1,2]),'short-b':identity([1,2])})


def measured(data,index):
    row=data['runs'][index]; p=PERSONALITIES[row['personality']]
    cfg=dict(MCOLS=p.matrix_columns,LANES=p.vector_lanes,DRAM_BYTES=2**30)
    return dict(status='completed',valid=True,backend='rtl',base_model_id=data['base_model_id'],
        tokenizer_id=data['tokenizer_id'],benchmark_manifest_id=identity(data),benchmark_run_index=index,
        benchmark_workload=row['workload'],memory_scenario=row['memory_scenario'],
        input_token_hash=row['input_token_hash'],personality=p.name,generation_mode='fixed-token-tape',tokens=[],
        prompt_tokens=2,variant_id=identity({'base':data['base_model_id'],'format':row['wformat'],'head':'int8'}),
        workload={**{k:row[k] for k in ('personality','wformat','context','input_tokens','latency','stall_percent','bandwidth_percent')},
            'seed':data['seed'],'backend':'rtl','max_new':8},config=cfg,
        configuration_id=identity({'config':cfg,'uarch':p.uarch}),
        microarchitecture={'schema_version':1,'parameters':p.uarch},
        counters=[dict(cycles=100,validation='bit-exact-dram-and-tmem')]*2)


def test_predictor_features_distinguish_fifo_bandwidth_and_fixed_tape():
    data=schedule(); balanced=measured(data,1); buffered=measured(data,3)
    a=Predictor.features(balanced); b=Predictor.features(buffered)
    assert len(a)==len(PREDICTOR_FEATURES)==11
    assert a[2]==b[2]==0 and a[9]!=b[9]
    stressed=copy.deepcopy(balanced); stressed['workload']['bandwidth_percent']=50
    assert Predictor.features(stressed)[10]!=a[10]
    del stressed['microarchitecture']
    with pytest.raises(ValueError,match='microarchitecture'): Predictor.features(stressed)


def test_predictor_frozen_partition_and_v1_cannot_silently_predict():
    data=schedule(); rows=[]
    for model in PREDICTOR_TRAINING_MODELS:
        for index in range(10):
            row=measured(data,index); row['base_model_id']=model; rows.append(row)
    predictor=Predictor().fit(rows); checkpoint=predictor.checkpoint(['fixture-only']*20)
    assert Predictor.from_checkpoint(checkpoint).predict(rows[0])==predictor.predict(rows[0])
    leaked=copy.deepcopy(rows); leaked[0]['base_model_id']=PREDICTOR_HELD_OUT_MODEL
    with pytest.raises(ValueError,match='partition'): Predictor().fit(leaked)
    legacy={**checkpoint,'schema_version':1,'coefficients':[0.]*9}
    assert Predictor.inspect_checkpoint(legacy)['runnable'] is False
    with pytest.raises(ValueError,match='v1 is inspectable only'): Predictor.from_checkpoint(legacy)
    reversed_schema={**checkpoint,'feature_schema':list(reversed(PREDICTOR_FEATURES))}
    with pytest.raises(ValueError,match='ordered'): Predictor.from_checkpoint(reversed_schema)


@pytest.mark.parametrize('field,value', [('context',256),('seed',0),('bandwidth_percent',50),
    ('latency',99),('stall_percent',0),('personality','compute'),('wformat','fp4'),('input_tokens',[2,1])])
def test_benchmark_manifest_binds_every_execution_setting(field,value):
    data=schedule(); result=measured(data,1)
    assert validate_benchmark_result(data,1,result)
    result['workload'][field]=value
    with pytest.raises(ValueError,match='mismatched'): validate_benchmark_result(data,1,result)


def test_index_resume_preserves_failure_reuses_success_and_aggregates_only_when_complete(tmp_path):
    data=schedule(); manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(data))
    calls=[]
    def run(data,model,row,index,root,emit):
        calls.append(index)
        if len(calls)==1: raise RuntimeError('interrupted simulator')
        return measured(data,index)
    with patch('malleable.llm.models.inspect',return_value={'base_model_id':'base','tokenizer_id':'tok'}), \
         patch('malleable.llm.experiments._run_performance_index',side_effect=run):
        with pytest.raises(RuntimeError): run_performance_manifest(manifest,tmp_path/'runs',run_index=1)
        report=run_performance_manifest(manifest,tmp_path/'runs',run_index=1)
        assert report['status']=='incomplete' and report['completed_indices']==[1]
        run_performance_manifest(manifest,tmp_path/'runs',run_index=1)
        assert calls==[1,1]
        report=run_performance_manifest(manifest,tmp_path/'runs')
        assert report['completed']==10 and report['failed']==0
    store=Store(tmp_path/'runs/research')
    try:
        attempts=store.records('llm-benchmark-attempt')
        assert len(attempts)==11 and sum(a['status']=='failed' for a in attempts)==1
        assert len(store.records('llm-benchmark-suite'))==1
    finally: store.close()


def approved(result):
    return dict(**{k:result[k] for k in ('base_model_id','tokenizer_id','variant_id','personality','configuration_id')},
        suite_frozen=True,target_count=1024,samples=1024,split='held-out',selectable=True,
        float_nll=2.,candidate_nll=2.,agreement=1.,suite_file_hash='frozen',
        suite_hashes={'calibration':'c','validation':'v','held-out':'h'})


@pytest.mark.parametrize('field,value', [('tokenizer_id','wrong'),('configuration_id','wrong'),
    ('personality','buffered'),('variant_id','wrong'),('base_model_id','wrong'),('samples',16),
    ('suite_frozen',False),('agreement',.89),('target_count',16),('candidate_nll',3.),('selectable',False)])
def test_failed_or_mismatched_quality_cannot_approve(field,value):
    result=measured(schedule(),1); quality=approved(result)
    assert quality_matches(quality,result,'held-out')
    quality[field]=value
    assert not quality_matches(quality,result,'held-out')


def test_quality_cannot_switch_frozen_suites():
    result=measured(schedule(),1); quality=approved(result); reference=copy.deepcopy(quality)
    quality['suite_file_hash']='changed'
    assert not quality_matches(quality,result,'held-out',reference)


def test_driver_uses_supported_query_and_records_unavailable():
    with patch('malleable.llm.cuda.subprocess.run',return_value=subprocess.CompletedProcess([],0,'616.92\n','')) as run:
        assert driver_metadata()=={'value':'616.92','provenance':'measured-nvidia-smi'}
        assert '--query-gpu=driver_version' in run.call_args.args[0]
    with patch('malleable.llm.cuda.subprocess.run',side_effect=FileNotFoundError('missing')):
        assert driver_metadata()['value'] is None
    with patch('malleable.llm.cuda.subprocess.run',return_value=subprocess.CompletedProcess([],0,'N/A\n','')):
        assert driver_metadata()['provenance']=='unavailable'


def test_restart_emits_terminal_event_and_indexed_trace_paths_are_bounded(tmp_path):
    from malleable.ide import Jobs
    model=tmp_path/'models/fixture'; model.mkdir(parents=True)
    jobs=Jobs(tmp_path/'jobs',tmp_path/'models',False)
    key=jobs.create('generate',{'model':str(model),'prompt':'test'})
    with jobs.connect() as db: db.execute("UPDATE jobs SET status='running' WHERE id=?",(key,))
    jobs.event(key,'status',{'status':'running'})
    reopened=Jobs(tmp_path/'jobs',tmp_path/'models',False)
    assert reopened.get(key)['status']=='interrupted'
    assert reopened.events(key)[-1]['payload']['status']=='interrupted'
    path=tmp_path/'jobs/performance/manifest/attempts/one/traces/step-00000/profile.json'
    jobs.event(key,'counters',{'step':0,'profile_path':str(path)})
    assert jobs.trace_directory(key,0)==path.parent
    jobs.event(key,'counters',{'step':0,'profile_path':str(tmp_path/'outside/profile.json')})
    with pytest.raises(ValueError,match='outside job root'): jobs.trace_directory(key,0)


def test_controller_records_per_case_regressions_and_has_no_oracle_observation():
    from malleable.llm.learning import train,evaluate,state
    window={'base_model_id':'train','family':'qwen3','evidence_ids':['fixture-only'],
        'provenance':'estimated-window-from-measured-rtl','service_cycles':{'balanced/int8':100.},
        'predicted_cycles':{'balanced/int8':90.},'predictor_id':'fixture','features':[0.]*6}
    before=state(window,'balanced/int8',1)
    changed={**window,'service_cycles':{'balanced/int8':100000.}}
    assert state(changed,'balanced/int8',1)==before
    checkpoint=train([[window]],0,1)
    report=evaluate(checkpoint,[[{**window,'base_model_id':'held-out'}]])
    assert len(report['cases'])==5 and 'per_case_regressions' in report


def test_final_verification_requires_matching_browser_coverage_and_clean_commit():
    from malleable.llm.full_release import validate_final_verification
    ui={'status':'passed','commands':['make verify-llm'],'checkpoint_downloads':False,
        'commit':'final','source_unchanged':True,'source_dirty':False}
    browser={'schema_version':1,'status':'passed','commit':'final','passed_checks':[
        'chat-formatting','sse-reconnect','duplicate-stale-events','cancellation',
        'restart','report-lineage','lens-replay','keyboard','validated-application']}
    validate_final_verification(ui,browser,'final')
    with pytest.raises(ValueError): validate_final_verification({**ui,'source_dirty':True},browser,'final')
    with pytest.raises(ValueError): validate_final_verification(ui,{**browser,'commit':'stale'},'final')
    with pytest.raises(ValueError): validate_final_verification(ui,{**browser,'passed_checks':['keyboard']},'final')
