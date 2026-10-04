import copy
import pytest
from tools.run_qwen35_acceptance import validate_quality,validate_rtl
from malleable.llm import candidates as C
from malleable.llm.quality import reference_contract
from test_candidate_integration import tiny_candidate,passing_validation
from test_int8_candidates import suite_file

def test_quality_rejects_old_reference_and_does_not_promote_failure(tmp_path):
    _,info,case=tiny_candidate(tmp_path)
    suite,_=suite_file(tmp_path,info)
    result=passing_validation(info,C.read_derived_candidate(case,info),suite)
    info['family']='qwen35'
    result.update(derived_candidate=None,variant_id=C.variant_id(info['base_model_id'],'int8'),
        passed=True,toolchain={'contract':reference_contract(info)})
    assert validate_quality(result,info,suite,'validation')['passed']
    failed=dict(result,agreement=.89,passed=False)
    assert not validate_quality(failed,info,suite,'validation')['passed']
    for change in [dict(toolchain={'contract':'original-local-safetensors-teacher-forced-v1'}),
                   dict(samples=16),dict(head_format='fp32'),dict(variant_id='other'),dict(passed=False)]:
        with pytest.raises(ValueError):validate_quality(dict(result,**change),info,suite,'validation')
    held=dict(result,split='held-out',selectable=True,candidate_freeze_id='f')
    freeze={'freeze_id':'f','configuration':{'configuration_id':result['configuration_id']}}
    assert validate_quality(held,info,suite,'held-out',freeze)['passed']
    with pytest.raises(ValueError):validate_quality(held,info,suite,'held-out',dict(freeze,freeze_id='wrong'))

def test_rtl_rejects_early_eos_and_unchecked_memory():
    quality={'base_model_id':'m','tokenizer_id':'t','variant_id':'v','configuration_id':'c'}
    result=dict(quality,status='completed',valid=True,backend='rtl',generation_mode='greedy',
        tokens=list(range(8)),prompt_tokens=2,personality='balanced',
        workload={'context':128,'wformat':'int8','max_new':8},toolchain={'build_id':'b'},
        counters=[{'validation':'bit-exact-dram-and-tmem','trace_sha256':'h'} for _ in range(9)])
    validate_rtl(result,quality)
    for field,value in [('tokens',[1]),('backend','isa'),('configuration_id','other')]:
        with pytest.raises(ValueError):validate_rtl(dict(result,**{field:value}),quality)
    bad=copy.deepcopy(result);bad['counters'][0]['validation']='dram-only'
    with pytest.raises(ValueError):validate_rtl(bad,quality)
