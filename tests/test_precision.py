import copy
import json
import numpy as np
import pytest

from malleable.llm.precision import GROUPS,OperatorStats,controlled_reference,make_panel,validate_policy
from malleable.records import identity


def policy(groups): return {'schema_version':1,'quantized_groups':list(groups),'group_size':128}


@pytest.fixture(scope='module')
def tiny():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(42)
    cfg=Qwen3Config(hidden_size=128,intermediate_size=128,num_hidden_layers=1,
        num_attention_heads=1,num_key_value_heads=1,head_dim=128,vocab_size=256,
        tie_word_embeddings=True,rope_theta=1e6,rms_norm_eps=1e-6)
    cfg._attn_implementation='eager'
    model=Qwen3ForCausalLM(cfg).float().eval()
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    return Spec(128,1,1,1,128,128,256),weights


def test_all_quantized_control_is_exact_upstream_and_all_float_matches_reference(tiny):
    from opentpu.llm.qwen3 import emulated_logits,reference_logits
    spec,weights=tiny; tokens=[1,4,6]
    baseline,_=controlled_reference(policy(GROUPS))
    assert np.array_equal(baseline(spec,weights,tokens),emulated_logits(spec,weights,tokens))
    floating,_=controlled_reference(policy([]))
    assert np.allclose(floating(spec,weights,tokens),reference_logits(spec,weights,tokens),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('bypassed',GROUPS)
def test_every_group_bypasses_only_its_sites_without_mutating_weights(tiny,bypassed):
    spec,weights=tiny; before={k:v.copy() for k,v in weights.items()}; stats=OperatorStats()
    run,_=controlled_reference(policy([g for g in GROUPS if g!=bypassed]),stats)
    assert np.isfinite(run(spec,weights,[2,7])).all()
    rows=stats.export()
    assert {r['group'] for r in rows}==set(GROUPS)
    for row in rows:
        assert row['quantized']==(row['group']!=bypassed)
        if row['group']==bypassed: assert row['error_squared']==0
    assert all(np.array_equal(before[k],v) for k,v in weights.items())


def frozen(tmp_path):
    data={'schema_version':1,'base_model_id':'model','tokenizer_id':'tokenizer',
        'calibration':[[9999,1,2]],
        'validation':[[100+s]+list(range(128)) for s in range(8)],
        'held-out':[[200+s]+list(range(128)) for s in range(8)]}
    names=('calibration','validation','held-out')
    data['freeze']={'split_hashes':{s:identity(data[s]) for s in names},
                    'target_counts':{s:sum(len(r)-1 for r in data[s]) for s in names}}
    path=tmp_path/'frozen.json'; path.write_text(json.dumps(data)); return path,data


def test_spread_panel_scores_exactly_128_preserves_prefixes_and_excludes_heldout(tmp_path):
    path,data=frozen(tmp_path); panel=make_panel(path,128,'spread')
    assert panel==make_panel(path,128,'spread') and len(panel['rows'])==8
    assert sum(len(r['positions']) for r in panel['rows'])==128
    assert panel['executed_tokens']>128 and panel['split']=='validation'
    for row in panel['rows']:
        assert row['tokens']==data['validation'][row['sequence']][:max(row['positions'])+2]
    prefix=make_panel(path,16)
    assert prefix['executed_tokens']==16 and prefix['rows'][0]['positions']==list(range(16))


@pytest.mark.parametrize('groups',[['unknown'],['head_weights','head_weights'],None])
def test_invalid_policies_rejected(groups):
    with pytest.raises(ValueError): validate_policy({'schema_version':1,'quantized_groups':groups,'group_size':128})


def test_diagnostic_rejects_tampered_panel_before_loading_checkpoint(tmp_path):
    from malleable.llm.precision import diagnose_precision
    path,_=frozen(tmp_path); panel=make_panel(path); panel['rows'][0]['tokens'][0]=500
    with pytest.raises(ValueError,match='panel mismatch'):
        diagnose_precision('unused',path,policy(GROUPS),panel,tmp_path/'ref')


def test_site_precision_rounding_outliers_zeros_and_partial_blocks():
    from malleable.llm.precision import site_quantize
    from opentpu.llm.qwen3 import _fake_q
    x=np.linspace(-0.01,0.01,128); x[0]=100; x[64:]=0
    assert np.array_equal(site_quantize(x,{'format':'int8','block':128}),_fake_q(x,128))
    small=site_quantize(x,{'format':'int8','block':16})
    wide=site_quantize(x,{'format':'int16','block':128})
    assert np.linalg.norm(small-x)<np.linalg.norm(_fake_q(x,128)-x)
    assert np.linalg.norm(wide-x)<np.linalg.norm(_fake_q(x,128)-x)
    assert np.array_equal(wide[64:],np.zeros(64))
    assert np.array_equal(site_quantize(x,{'format':'float16','block':128}),x.astype(np.float16).astype(np.float64))
    with pytest.raises(ValueError,match='divide'): site_quantize(x[:127],{'format':'int16','block':128})


def test_key_only_override_retains_other_quantization_sites(tiny):
    spec,weights=tiny; stats=OperatorStats()
    p=dict(policy(GROUPS),schema_version=2,site_overrides={'key_store':{'format':'int8','block':32}})
    run,_=controlled_reference(p,stats); before={k:v.copy() for k,v in weights.items()}
    assert np.isfinite(run(spec,weights,[2,3,7])).all()
    assert all(row['quantized'] for row in stats.export())
    for row in stats.export():
        assert row['block']==(32 if row['operator']=='key_store' else 128)
    assert all(np.array_equal(weights[k],v) for k,v in before.items())


@pytest.mark.parametrize('overrides',[{'value_store':{'format':'float16','block':128}},
    {'key_store':{'format':'int4','block':128}}, {'query':{'format':'int16','block':17}}])
def test_diagnostic_site_policy_rejects_unapproved_formats(overrides):
    with pytest.raises(ValueError): validate_policy(dict(policy(GROUPS),schema_version=2,site_overrides=overrides))
