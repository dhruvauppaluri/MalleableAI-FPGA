import copy
import pytest
from malleable.llm import candidates as C
from malleable.llm.experiments import validate_performance_manifest,validate_benchmark_result,precision_lineage
from malleable.records import identity
from test_candidate_integration import tiny_candidate,passing_validation
from test_int8_candidates import suite_file
from test_release_implementation import schedule,measured

def fixture(tmp_path,both_pass):
    model,info,case=tiny_candidate(tmp_path);derived=C.read_derived_candidate(case,info)
    suite,_=suite_file(tmp_path,info)
    data=schedule();data.update(schema_version=2,model_path=str(model),base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'])
    screens=[]
    for fmt in ('int4','fp4'):
        q=passing_validation(info,derived,suite);q.pop('record_id')
        q.update(wformat=fmt,derived_candidate=None,variant_id=C.variant_id(info['base_model_id'],fmt),
            agreement=.95 if both_pass or fmt=='int4' else .8,toolchain={'contract':'original-local-safetensors-teacher-forced-v1'})
        screens.append({'id':identity(q),'record':q})
    data['format_screening_evidence']=screens
    data['quality_evidence']=screens if both_pass else screens[:1]
    data['quality_evidence_ids']=[e['id'] for e in data['quality_evidence']]
    if both_pass:
        for i,fmt in enumerate(('int4','fp4'),8):
            data['runs'][i].update(workload='short-a',wformat=fmt,personality='balanced',memory_scenario='baseline',latency=20,stall_percent=20,bandwidth_percent=100)
    for row in data['runs']:
        d=derived if row['wformat']=='int8' else None
        row.update(candidate_case=str(case) if d else None,derived_candidate=C.derived_record(d) if d else None,
            variant_id=C.variant_id(info['base_model_id'],row['wformat'],d),precision_policy=precision_lineage(row['wformat']))
    return data,info

@pytest.mark.parametrize('both_pass',[False,True])
def test_mixed_lineage_and_stress_disposition(tmp_path,both_pass):
    data,info=fixture(tmp_path,both_pass)
    assert validate_performance_manifest(data,info)
    for index in (1,8):
        row=data['runs'][index];result=measured(data,index)
        result.update(variant_id=row['variant_id'],derived_candidate=row['derived_candidate'],benchmark_precision_policy=row['precision_policy'])
        result['workload']['candidate_case']=row['candidate_case']
        assert validate_benchmark_result(data,index,result)
        result['workload']['candidate_case']='wrong'
        with pytest.raises(ValueError):validate_benchmark_result(data,index,result)
    missing=copy.deepcopy(data);missing['runs'][0].pop('precision_policy')
    with pytest.raises(ValueError):validate_performance_manifest(missing,info)
    stale=copy.deepcopy(data);stale['format_screening_evidence'][0]['record']['agreement']=0
    with pytest.raises(ValueError):validate_performance_manifest(stale,info)

def test_v2_rejects_missing_screens_and_changed_candidate(tmp_path):
    data,info=fixture(tmp_path,False)
    missing=copy.deepcopy(data);missing['format_screening_evidence']=[]
    with pytest.raises(ValueError):validate_performance_manifest(missing,info)
    data['runs'][0]['derived_candidate']['parameters_id']='tampered'
    with pytest.raises(ValueError):validate_performance_manifest(data,info)
