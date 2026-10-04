"""Release approval boundaries tested with local tiny artifacts, never real held-out."""
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import pytest
from malleable.llm import heldout
from malleable.llm import candidates as C
from malleable.llm.release import approved_draft,validate_draft_approval
from malleable.llm.full_release import validate_performance_lineage
from malleable.llm.models import digest
from test_candidate_integration import tiny_candidate,passing_validation,CLEAN
from test_int8_candidates import suite_file
from test_release_implementation import schedule

@pytest.fixture(autouse=True)
def isolated_registry(tmp_path,monkeypatch):
    monkeypatch.setattr(heldout,'registry_path',lambda:tmp_path/'registry.sqlite3')

def test_full_checker_path_verifies_derived_weight_lineage(tmp_path):
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    data=schedule();data.update(model_path=str(model),base_model_id=info['base_model_id'],
        tokenizer_id=info['tokenizer_id'],candidate_case=str(case),derived_candidate=C.derived_record(derived))
    assert validate_performance_lineage(data,info['base_model_id'])['weight_files']==info['weight_files']
    with pytest.raises(ValueError,match='weight lineage'):
        C.read_derived_candidate(case,{k:info[k] for k in ('base_model_id','tokenizer_id')})
    manifest=json.loads((case/'candidate.json').read_text());manifest['derived_id']='0'*64
    (case/'candidate.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):validate_performance_lineage(data,info['base_model_id'])

def test_hybrid_requires_exact_approved_draft_and_checksummed_artifact(tmp_path):
    approved={'base_model_id':'m','tokenizer_id':'t','variant_id':'derived',
        'derived_candidate':{'derived_id':'d','parameters_id':'p'},'configuration_id':'c',
        'personality':'balanced','workload':{'context':128,'wformat':'int8'}}
    selected={'draft_model_id':'m','tokenizer_id':'t','draft_variant_id':'derived',
        'draft_derived_candidate':approved['derived_candidate'],'draft_configuration_id':'c',
        'draft_head_format':'int8','personality':'balanced','context':128,'wformat':'int8'}
    assert validate_draft_approval(approved,selected)
    for field,value in [('draft_variant_id','raw'),('draft_derived_candidate',None),
        ('draft_model_id','other'),('tokenizer_id','other'),('draft_configuration_id','other'),
        ('personality','compact'),('context',256),('wformat','fp4'),('draft_head_format','fp32')]:
        with pytest.raises(ValueError,match='approved standalone'):validate_draft_approval(approved,dict(selected,**{field:value}))
    artifact=tmp_path/'generation.json';artifact.write_text(json.dumps(approved))
    manifest=tmp_path/'standalone.json';manifest.write_text(json.dumps({'models':{'Qwen3-0.6B':
        {'generation':{'path':artifact.name,'sha256':digest(artifact)}}}}))
    assert approved_draft(manifest)==approved
    artifact.write_text('{}')
    with pytest.raises(ValueError,match='hash/path'):approved_draft(manifest)

def test_raw_hybrid_is_rejected_before_verifier_load(tmp_path,monkeypatch):
    import torch
    from malleable.llm import cuda,release
    model,info,case=tiny_candidate(tmp_path);derived=C.read_derived_candidate(case,info)
    approved={'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
        'variant_id':C.variant_id(info['base_model_id'],'int8',derived),'derived_candidate':C.derived_record(derived),
        'configuration_id':'approved','personality':'balanced','workload':{'context':128,'wformat':'int8'}}
    monkeypatch.setattr(cuda,'_standalone_gate',lambda _: {'manifest_sha256':'approved'})
    monkeypatch.setattr(release,'approved_draft',lambda _:approved)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    loaded=[];monkeypatch.setattr(cuda,'CudaVerifier',lambda *a:loaded.append(a))
    with pytest.raises(ValueError,match='approved standalone'):
        cuda.hybrid_generate(model,tmp_path/'verifier','hello',context=128,standalone_manifest='manifest')
    assert loaded==[]

def record():
    return {'base_model_id':'m','tokenizer_id':'t','variant_id':'v','suite_file_hash':'s',
        'configuration':{'configuration_id':'c'},'suite_hashes':{'held-out':'h'},'freeze_id':'f'}

def test_atomic_claim_survives_copy_source_change_and_failed_attempt(tmp_path):
    first=record();heldout.consume(first)
    changed=copy.deepcopy(first);changed.update(freeze_id='copy',source={'commit':'different'},path='/relocated')
    with pytest.raises(ValueError,match='already claimed'):heldout.consume(changed)
    assert heldout.evaluation_key(first)==heldout.evaluation_key(changed)
    with pytest.raises(ValueError,match='complete held-out'):heldout.consume({'freeze_id':'only'})

def test_concurrent_claim_has_exactly_one_winner():
    def claim(_):
        try:heldout.consume(record());return True
        except ValueError:return False
    with ThreadPoolExecutor(max_workers=8) as pool:assert sum(pool.map(claim,range(8)))==1

def test_completed_import_is_idempotent_but_conflicting_evidence_fails(tmp_path):
    data=record();data.update(split='held-out',record_id='r')
    file=tmp_path/'result.json';file.write_text(json.dumps(data));sha=digest(file)
    assert heldout.register_completed(data,file,sha)==heldout.register_completed(data,file,sha)
    changed=dict(data,record_id='other');file.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='already claimed'):heldout.register_completed(changed,file,digest(file))

def test_window_cannot_reset_or_replace_deadline(tmp_path,monkeypatch):
    from malleable.llm import continuation
    path=tmp_path/'control.json';monkeypatch.setattr(continuation,'CONTROL',path)
    monkeypatch.setattr(continuation.time,'time',lambda:100.)
    with pytest.raises(ValueError,match='establish'):continuation.read_control()
    data={'schema_version':1,'mode':'overnight','authorization':'user','started_unix':90.,'dispatch_deadline_unix':90.+28800}
    path.write_text(json.dumps(data))
    assert continuation.read_control(data['dispatch_deadline_unix'])==data
    with pytest.raises(ValueError,match='original'):continuation.read_control(data['dispatch_deadline_unix']+1)
    monkeypatch.setattr(continuation.time,'time',lambda:90.+28800)
    with pytest.raises(ValueError,match='expired'):continuation.read_control()

def test_configuration_fields_are_required_and_recomputed(tmp_path):
    _,info,case=tiny_candidate(tmp_path);derived=C.read_derived_candidate(case,info)
    suite,_=suite_file(tmp_path,info);good=passing_validation(info,derived,suite)
    for field in ('context','personality','wformat','head_format','configuration_id','config','microarchitecture'):
        changed=dict(good);changed.pop(field)
        with pytest.raises(ValueError,match='passing full validation'):
            C.create_freeze(tmp_path/'never.json',info,suite,derived,changed,'h',CLEAN)
    changed=copy.deepcopy(good);changed['config']['LANES']=16
    with pytest.raises(ValueError,match='identity'):
        C.create_freeze(tmp_path/'never.json',info,suite,derived,changed,'h',CLEAN)
    freeze=C.create_freeze(tmp_path/'freeze.json',info,suite,derived,good,'h',CLEAN)
    assert freeze['schema_version']==2
    old=dict(freeze,schema_version=1);old['freeze_id']=C._freeze_id(old)
    (tmp_path/'old.json').write_text(json.dumps(old))
    assert C.verify_freeze(tmp_path/'old.json',info,suite,derived,CLEAN,good['configuration_id'])['schema_version']==1

def driver():
    file=Path(__file__).resolve().parents[1]/'tools/run_performance_indices.py'
    spec=importlib.util.spec_from_file_location('performance_driver',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('child_code,deadline,expected',[(7,100.,1),(0,0.,3),(0,100.,0)])
def test_driver_propagates_failure_and_deadline(tmp_path,monkeypatch,child_code,deadline,expected):
    module=driver();monkeypatch.setattr(module,'source_state',lambda:dict(CLEAN))
    monkeypatch.setattr(module.time,'time',lambda:1.)
    calls=[]
    monkeypatch.setattr(module.subprocess,'call',lambda *a,**k:calls.append(a) or child_code)
    log=tmp_path/'driver.log'
    assert module.run_indices('manifest','store',deadline,log,[0,1])==expected
    assert len(calls)==(0 if expected==3 else 1 if expected==1 else 2)
    summary=json.loads(log.read_text().splitlines()[-1])
    assert summary['status']=={0:'complete',1:'failed',3:'deadline-limited'}[expected]
    assert len(list((tmp_path/'driver.log.attempts').glob('*/result.json')))==len(calls)
