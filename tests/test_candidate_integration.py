"""Derived-candidate identity, freeze, held-out exactly-once and production plumbing."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from llm_fixture import save
from test_int8_candidates import stats, suite_file
from malleable.llm import candidates as C
from malleable.llm.models import digest, inspect, load
from malleable.llm.records import GenerationWorkload
from malleable.records import identity

CLEAN={'commit':'a'*40,'status':''}


def tiny_candidate(tmp_path,model=None,attempt=None):
    model=model or tmp_path/'model'; attempt=attempt or tmp_path/'attempt'; _,_,spec=save(model); info=inspect(model,128); weights=load(model)
    lineage={k:info[k] for k in ('base_model_id','tokenizer_id','weight_files')}
    statistics=stats(spec,weights); statistics['source']=lineage
    candidate=C.Candidate(0.5); parameters=C.fit_parameters(weights,spec,statistics,candidate,identity(statistics))
    derived=C.derive_tensors(weights,spec,parameters)
    artifacts=attempt/'artifacts'; case=attempt/'case-00'; case.mkdir(parents=True)
    sid,_=C.write_artifact(artifacts,'calibration-statistics',statistics)
    pid,_=C.write_artifact(artifacts,'candidate-parameters',parameters)
    did,_=C.write_artifact(artifacts,'derived-tensors',C.tensor_manifest(derived,pid,lineage))
    file=C.write_tensors(artifacts,did,derived)
    meta={'candidate':candidate.record(),'calibration_statistics_id':sid,'parameters_id':pid,
          'derived_id':did,'tensor_file':file.name}
    result=dict(meta,base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],status='completed',
        agreement=0.95,float_nll=4.,candidate_nll=4.,nll_degradation_percent=0.,target_count=128)
    (case/'candidate.json').write_text(json.dumps(meta)); (case/'result.json').write_text(json.dumps(result))
    (case/'summary.json').write_text(json.dumps(dict(result,result_sha256=digest(case/'result.json'))))
    return model,info,case


def passing_validation(info,derived,suite,config_id='cfg'):
    return dict(base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],split='validation',
        suite_frozen=True,target_count=1024,samples=1024,suite_file_hash=digest(suite),
        suite_hashes=json.loads(Path(suite).read_text())['freeze']['split_hashes'],
        context=128,personality='balanced',wformat='int8',head_format='int8',
        derived_candidate=C.derived_record(derived),variant_id=C.variant_id(info['base_model_id'],'int8',derived),
        float_nll=4.,candidate_nll=4.01,agreement=.95,configuration_id=config_id,config={'D':128},
        microarchitecture={'schema_version':1},record_id='r',toolchain={'torch':'x'})


def test_variant_identity_preserves_legacy_and_binds_derived_lineage():
    legacy=identity({'base':'b','format':'int8','head':'int8'})
    assert C.variant_id('b','int8')==legacy
    derived={'derived_id':'d'*64,'parameters_id':'p'*64}
    assert C.variant_id('b','int8',derived)!=legacy
    assert C.variant_id('b','int8',derived)!=C.variant_id('b','int8',dict(derived,derived_id='e'*64))
    assert C.variant_id('b','int8',derived)!=C.variant_id('b','int8',dict(derived,parameters_id='q'*64))


def test_workload_hash_preserved_and_candidate_is_int8_only():
    base=GenerationWorkload('hello')
    assert base.workload_id==GenerationWorkload('hello',candidate_case=None).workload_id
    assert GenerationWorkload('hello',candidate_case='/x').workload_id!=base.workload_id
    with pytest.raises(ValueError,match='INT8'): GenerationWorkload('hello',wformat='int4',candidate_case='/x')


def test_generation_uses_verified_derived_weights_and_records_lineage(tmp_path):
    from malleable.llm.runtime import generate
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    base=generate(model,GenerationWorkload('Hello world',prompt_format='raw',backend='isa',context=128,max_new=2),
        tmp_path/'plain')
    run=generate(model,GenerationWorkload('Hello world',prompt_format='raw',backend='isa',context=128,max_new=2,
        candidate_case=str(case)),tmp_path/'derived')
    assert 'candidate_case' not in base['workload'] and 'derived_candidate' not in base
    assert base['variant_id']==identity({'base':info['base_model_id'],'format':'int8','head':'int8'})
    assert run['variant_id']==C.variant_id(info['base_model_id'],'int8',derived)!=base['variant_id']
    assert run['derived_candidate']==C.derived_record(derived) and run['workload']['candidate_case']==str(case)
    assert run['workload_id']!=base['workload_id']


def test_generation_rejects_tampered_or_mismatched_candidate(tmp_path):
    from malleable.llm.runtime import generate
    model,info,case=tiny_candidate(tmp_path)
    summary=json.loads((case/'summary.json').read_text()); summary['agreement']=0.5
    (case/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        generate(model,GenerationWorkload('Hello',prompt_format='raw',backend='isa',context=128,max_new=1,
            candidate_case=str(case)),tmp_path/'run')


def test_freeze_requires_clean_source_and_exact_passing_validation(tmp_path):
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    suite,_=suite_file(tmp_path,info); good=passing_validation(info,derived,suite)
    with pytest.raises(ValueError,match='clean'):
        C.create_freeze(tmp_path/'a.json',info,suite,derived,good,'h',dict(CLEAN,status=' M x'))
    for update in ({'agreement':.5},{'candidate_nll':5.},{'split':'held-out'},{'target_count':1023},
                   {'variant_id':'0'*64},{'derived_candidate':{}},{'suite_file_hash':'x'},{'personality':'compact'}):
        with pytest.raises(ValueError,match='passing full validation'):
            C.create_freeze(tmp_path/'b.json',info,suite,derived,dict(good,**update),'h',CLEAN)
    freeze=C.create_freeze(tmp_path/'freeze.json',info,suite,derived,good,'h',CLEAN)
    assert freeze['freeze_id']==identity({k:v for k,v in freeze.items() if k!='freeze_id'})
    with pytest.raises(FileExistsError): C.create_freeze(tmp_path/'freeze.json',info,suite,derived,good,'h',CLEAN)


def test_freeze_verification_detects_every_identity_change(tmp_path):
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    suite,data=suite_file(tmp_path,info)
    C.create_freeze(tmp_path/'freeze.json',info,suite,derived,passing_validation(info,derived,suite,'cfg'),'h',CLEAN)
    ok=lambda **k: C.verify_freeze(tmp_path/'freeze.json',k.get('info',info),k.get('suite',suite),
        k.get('derived',derived),k.get('source',CLEAN),k.get('configuration','cfg'))
    assert ok()['kind']==C.FREEZE_KIND
    with pytest.raises(ValueError,match='clean frozen code commit'): ok(source={'commit':'b'*40,'status':''})
    with pytest.raises(ValueError,match='clean frozen code commit'): ok(source=dict(CLEAN,status='?? x'))
    with pytest.raises(ValueError,match='frozen identity'): ok(configuration='other')
    with pytest.raises(ValueError,match='frozen identity'): ok(info=dict(info,tokenizer_id='z'))
    with pytest.raises(ValueError,match='frozen identity'): ok(derived=dict(derived,derived_id='z'*64))
    changed=copy.deepcopy(data); changed['held-out'][0][0]=(changed['held-out'][0][0]+1)%128
    path=tmp_path/'changed.json'; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): ok(suite=path)
    forged=json.loads((tmp_path/'freeze.json').read_text()); forged['source']['commit']='b'*40
    (tmp_path/'forged.json').write_text(json.dumps(forged))
    with pytest.raises(ValueError,match='invalid candidate freeze'):
        C.verify_freeze(tmp_path/'forged.json',info,suite,derived,CLEAN,'cfg')


def test_heldout_claim_is_exactly_once(tmp_path):
    freeze={'freeze_id':'f'*64}; (tmp_path/'freeze.json').write_text('{}')
    claim=C.claim_heldout(tmp_path/'freeze.json',freeze)
    assert json.loads(claim.read_text())['freeze_id']=='f'*64
    with pytest.raises(ValueError,match='already claimed'): C.claim_heldout(tmp_path/'freeze.json',freeze)


def test_heldout_requires_freeze_and_validation_rejects_one(tmp_path):
    from malleable.llm.quality import evaluate
    model,info,case=tiny_candidate(tmp_path); suite,_=suite_file(tmp_path,info)
    with pytest.raises(ValueError,match='requires a frozen candidate'):
        evaluate(model,suite,split='held-out',context=128,candidate_case=case)
    with pytest.raises(ValueError,match='applies only to its single held-out'):
        evaluate(model,suite,split='validation',context=128,candidate_case=case,candidate_freeze=tmp_path/'f.json')


def test_end_to_end_tiny_freeze_and_single_heldout_evaluation(tmp_path,monkeypatch):
    from malleable.llm.quality import evaluate
    model,info,case=tiny_candidate(tmp_path); suite,_=suite_file(tmp_path,info)
    validation=evaluate(model,suite,split='validation',context=128,candidate_case=case,cache_root=tmp_path/'refs')
    derived=C.read_derived_candidate(case,info)
    assert validation['variant_id']==C.variant_id(info['base_model_id'],'int8',derived)
    assert validation['derived_candidate']==C.derived_record(derived) and 'candidate_freeze_id' not in validation
    forced=dict(validation,agreement=.95,candidate_nll=validation['float_nll'],record_id='r')
    freeze_path=tmp_path/'held'/'freeze.json'; freeze_path.parent.mkdir()
    C.create_freeze(freeze_path,info,suite,derived,forced,'h',CLEAN)
    monkeypatch.setattr(C,'source_state',lambda: dict(CLEAN))
    held=evaluate(model,suite,split='held-out',context=128,candidate_case=case,candidate_freeze=freeze_path,
        cache_root=tmp_path/'refs')
    freeze=json.loads(freeze_path.read_text())
    assert held['split']=='held-out' and held['candidate_freeze_id']==freeze['freeze_id']
    assert held['variant_id']==validation['variant_id'] and held['configuration_id']==validation['configuration_id']
    assert held['selectable']==held['passed'] and held['floating_reference_id']!=validation['floating_reference_id']
    with pytest.raises(ValueError,match='already claimed'):
        evaluate(model,suite,split='held-out',context=128,candidate_case=case,candidate_freeze=freeze_path,
            cache_root=tmp_path/'refs')


def lineage(tmp_path):
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    suite,_=suite_file(tmp_path,info)
    freeze=C.create_freeze(tmp_path/'freeze.json',info,suite,derived,passing_validation(info,derived,suite),'h',CLEAN)
    record=C.derived_record(derived); variant=C.variant_id(info['base_model_id'],'int8',derived)
    run={'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],'variant_id':variant,
         'derived_candidate':record,'workload':{'wformat':'int8'},'configuration_id':'cfg'}
    quality={'variant_id':variant,'derived_candidate':record,'candidate_freeze_id':freeze['freeze_id'],
             'suite_file_hash':freeze['suite_file_hash']}
    return run,quality,freeze


def test_release_lineage_requires_one_candidate_identity(tmp_path):
    run,quality,freeze=lineage(tmp_path)
    assert C.check_release_lineage(run,quality,freeze) is True
    assert C.check_release_lineage({'variant_id':'x'},{'variant_id':'x'},None) is False
    legacy_run=dict(run); legacy_run.pop('derived_candidate')
    for r,q,f in [(legacy_run,quality,freeze),
                  (dict(run,variant_id='0'*64),quality,freeze),
                  (run,dict(quality,candidate_freeze_id='0'*64),freeze),
                  (run,dict(quality,derived_candidate=dict(quality['derived_candidate'],derived_id='z')),freeze),
                  (dict(run,configuration_id='other'),quality,freeze),
                  (run,dict(quality,suite_file_hash='x'),freeze),
                  (run,quality,None),(run,quality,dict(freeze,source={'commit':'c','status':''}))]:
        with pytest.raises(ValueError): C.check_release_lineage(r,q,f)
    raw=identity({'base':run['base_model_id'],'format':'int8','head':'int8'})
    with pytest.raises(ValueError): C.check_release_lineage(dict(run,variant_id=raw),dict(quality,variant_id=raw),freeze)


def test_performance_manifest_binds_candidate_lineage_and_variant(tmp_path):
    from malleable.llm import experiments as E
    from test_release_implementation import schedule
    model,info,case=tiny_candidate(tmp_path); derived=C.read_derived_candidate(case,info)
    data=schedule(); data.update(base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],
        candidate_case=str(case),derived_candidate=C.derived_record(derived))
    assert E.manifest_candidate(data,info)['derived_id']==derived['derived_id']
    with pytest.raises(ValueError,match='lineage'): E.manifest_candidate(dict(data,derived_candidate={}),info)
    with pytest.raises(ValueError,match='candidate case'): E.manifest_candidate(dict(data,candidate_case=None),info)
    assert E.manifest_candidate({},info) is None
    plain=copy.deepcopy(data); plain.pop('candidate_case'); plain.pop('derived_candidate')
    assert C.variant_id(data['base_model_id'],'int8',data['derived_candidate'])!=C.variant_id(data['base_model_id'],'int8')
    with pytest.raises(ValueError):
        E.validate_performance_manifest(dict(data,runs=[dict(r,wformat='int4') if i==0 else r
            for i,r in enumerate(data['runs'])]),info)


def test_workbench_accepts_only_verified_candidates_inside_the_job_root(tmp_path):
    from malleable.ide import Jobs
    jobs=Jobs(tmp_path/'jobs',tmp_path/'models',False)
    model,info,case=tiny_candidate(tmp_path,tmp_path/'models'/'m',tmp_path/'jobs'/'release'/'attempt')
    payload=dict(model=str(model),prompt='Hello',prompt_format='raw',backend='isa',context=128,candidate_case=str(case))
    job=jobs.create('generate',dict(payload)); argv=jobs.argv('generate',jobs.get(job) and json.loads(jobs.get(job)['payload']),job)
    assert '--candidate-case='+str(case.resolve()) in argv
    outside_model,_,outside=tiny_candidate(tmp_path/'x',tmp_path/'models'/'n',tmp_path/'outside'/'attempt')
    with pytest.raises(ValueError,match='job data root'): jobs.create('generate',dict(payload,candidate_case=str(outside)))
    with pytest.raises(ValueError,match='unsupported job fields'):
        jobs.create('quality',dict(model=str(model),suite=str(model/'config.json'),candidate_case=str(case)))
    summary=json.loads((case/'summary.json').read_text()); summary['agreement']=.1
    (case/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError): jobs.create('generate',dict(payload))
