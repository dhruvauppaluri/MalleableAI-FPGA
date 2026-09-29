import json
import numpy as np
import pytest

from malleable.llm import candidates as C
from malleable.llm.precision import GROUPS,controlled_reference,make_panel,diagnose_precision
from malleable.records import identity

FLOAT={'schema_version':1,'quantized_groups':[],'group_size':128}


def synthetic(tied=True,seed=3,n_q=2,n_kv=1,layers=2):
    from opentpu.llm.qwen3 import Spec
    rng=np.random.default_rng(seed); d=128; h=128; ffn=256; vocab=64
    spec=Spec(h,layers,n_q,n_kv,d,ffn,vocab,tied=tied)
    norm=lambda n: 1+0.1*rng.standard_normal(n)
    w={'model.embed_tokens.weight':rng.standard_normal((vocab,h))*0.5,'model.norm.weight':norm(h)}
    w['model.norm.weight'][5]*=15
    if not tied: w['lm_head.weight']=rng.standard_normal((vocab,h))*0.08
    for i in range(layers):
        p=f'model.layers.{i}.'
        w[p+'input_layernorm.weight']=norm(h); w[p+'input_layernorm.weight'][9]*=20
        w[p+'post_attention_layernorm.weight']=norm(h); w[p+'post_attention_layernorm.weight'][11]*=12
        w[p+'self_attn.q_norm.weight']=norm(d); w[p+'self_attn.k_norm.weight']=norm(d)
        w[p+'self_attn.k_norm.weight'][7]*=10
        for name,shape in (('self_attn.q_proj',(n_q*d,h)),('self_attn.k_proj',(n_kv*d,h)),('self_attn.v_proj',(n_kv*d,h)),
                           ('self_attn.o_proj',(h,n_q*d)),('mlp.gate_proj',(ffn,h)),('mlp.up_proj',(ffn,h)),('mlp.down_proj',(h,ffn))):
            w[p+name+'.weight']=rng.standard_normal(shape)*0.05
    return spec,{k:v.astype(np.float32) for k,v in w.items()}


def sequences(): return [[3,17,5,40,2,9,11,60],[1,2,3,4,5,6,7,8,9,10]]


def stats(spec,weights):
    return C.collect_statistics(spec,weights,sequences(),{'base_model_id':'synthetic'},'calibration-hash')


def logits(spec,weights,tokens=(4,9,33,7,21)):
    run,_=controlled_reference(FLOAT); return run(spec,weights,list(tokens))


def derive(spec,weights,candidate,dtype=None):
    statistics=stats(spec,weights); parameters=C.fit_parameters(weights,spec,statistics,candidate,identity(statistics))
    return C.derive_tensors(weights,spec,parameters,dtype),parameters,statistics


@pytest.mark.parametrize('tied',[True,False])
@pytest.mark.parametrize('strength',[0.25,0.5,0.75])
def test_rescaling_preserves_unquantized_operation(tied,strength):
    spec,weights=synthetic(tied); sites=C.SITES if not tied else C.DEFAULT_SITES
    candidate=C.Candidate(strength,None,sites)
    derived,parameters,_=derive(spec,weights,candidate,np.float64)
    assert derived and any(not np.allclose(s,1) for s in parameters['scales'].values())
    reference=logits(spec,weights); candidate_logits=logits(spec,C.merged(weights,derived))
    assert np.allclose(candidate_logits,reference,rtol=1e-9,atol=1e-9)
    stored,_,_=derive(spec,weights,candidate)
    assert np.allclose(logits(spec,C.merged(weights,stored)),reference,rtol=2e-4,atol=2e-4)
    assert all(v.dtype==np.float32 for v in stored.values())


def test_shared_consumers_use_joint_statistics_and_one_scale():
    spec,weights=synthetic(); weights=dict(weights); statistics=stats(spec,weights)
    candidate=C.Candidate(0.5,None,('attn_input','attn_output'))
    scales=C.fit_scales(spec,weights,statistics,candidate)
    s=np.asarray(scales['attn_input/0'])
    joint=np.max([np.abs(weights[f'model.layers.0.self_attn.{x}_proj.weight']).astype(np.float64).max(0) for x in 'qkv'],0)
    act=np.asarray(statistics['channels']['attention_input/0'])
    assert np.array_equal(s,C._smooth(act,joint,0.5))
    q_only=C._smooth(act,np.abs(weights['model.layers.0.self_attn.q_proj.weight']).astype(np.float64).max(0),0.5)
    assert not np.array_equal(s,q_only)
    inputs_only={k:v for k,v in scales.items() if k.startswith('attn_input')}
    changed=C.layer_tensors(weights,spec,0,inputs_only)
    assert set(changed)=={f'model.layers.0.self_attn.{x}_proj.weight' for x in 'qkv'}|{'model.layers.0.input_layernorm.weight'}
    for x in 'qkv':
        name=f'model.layers.0.self_attn.{x}_proj.weight'
        assert np.allclose(changed[name]/weights[name].astype(np.float64),s[None,:])
    changed=C.layer_tensors(weights,spec,0,scales)
    assert np.allclose(changed['model.layers.0.input_layernorm.weight']*s,weights['model.layers.0.input_layernorm.weight'])
    d=spec.head_dim; sv=np.asarray(scales['attn_output/0'])
    o=changed['model.layers.0.self_attn.o_proj.weight']/weights['model.layers.0.self_attn.o_proj.weight'].astype(np.float64)
    assert np.allclose(o[:,:d],sv[None,:]) and np.allclose(o[:,d:2*d],sv[None,:])
    v=changed['model.layers.0.self_attn.v_proj.weight']
    assert np.allclose(v*sv[:,None],weights['model.layers.0.self_attn.v_proj.weight']*s[None,:])


def test_qk_scales_are_rope_pair_equal_and_scores_preserved():
    spec,weights=synthetic(); derived,parameters,_=derive(spec,weights,C.Candidate(0.75,None,('qk',)),np.float64)
    s=np.asarray(parameters['scales']['qk/0']); h=spec.head_dim//2
    assert np.array_equal(s[:h],s[h:]) and not np.allclose(s,1)
    assert set(derived)=={f'model.layers.{i}.self_attn.{x}_norm.weight' for i in range(2) for x in 'qk'}
    bad=dict(parameters['scales']); bad['qk/0']=[1.0]*h+[2.0]*h
    with pytest.raises(ValueError,match='RoPE pair'): C.layer_tensors(weights,spec,0,bad)


def test_tied_embedding_is_never_rewritten_and_untying_is_refused():
    spec,weights=synthetic(True); before={k:v.copy() for k,v in weights.items()}
    derived,_,_=derive(spec,weights,C.Candidate(0.75,99.9))
    assert 'model.embed_tokens.weight' not in derived and 'lm_head.weight' not in derived
    assert 'model.norm.weight' not in derived
    assert all(np.array_equal(before[k],v) for k,v in weights.items())
    assert not any(np.shares_memory(v,weights[k]) for k,v in derived.items())
    for kwargs in ({'sites':C.SITES},{'clip_groups':('transformer_weights','head_weights')}):
        with pytest.raises(ValueError,match='tied'):
            C.check_candidate(spec,C.Candidate(0.5,99.9,**kwargs))
    untied,uweights=synthetic(False)
    derived,parameters,_=derive(untied,uweights,C.Candidate(0.5,99.9,C.SITES,('transformer_weights','head_weights')))
    assert {'lm_head.weight','model.norm.weight'}<=set(derived) and 'model.embed_tokens.weight' not in derived
    assert np.abs(derived['lm_head.weight']).max()<=np.float32(parameters['clip_thresholds']['lm_head.weight'])


def test_clipping_is_lossy_percentile_of_weight_only():
    spec,weights=synthetic(); weights['model.layers.0.mlp.up_proj.weight'][3,4]=5.0
    a,pa,_=derive(spec,weights,C.Candidate(0.0,99.99)); b,pb,_=derive(spec,weights,C.Candidate(0.0,99.9))
    assert not C.Candidate(0.0,99.9).preserves_unquantized_operation
    assert set(pa['clip_thresholds'])==set(pb['clip_thresholds'])==set(a)==set(b)
    assert not any('embed' in n or 'norm' in n for n in a) and pa['scales']=={}
    name='model.layers.0.mlp.up_proj.weight'
    assert pb['clip_thresholds'][name]<=pa['clip_thresholds'][name]<5.0 and a[name].max()<5.0
    assert np.isclose(pb['clip_thresholds'][name],np.quantile(np.abs(weights[name].astype(np.float64)),0.999))
    assert np.abs(b[name]).max()<=np.float32(pb['clip_thresholds'][name])
    assert not np.array_equal(b[name],weights[name])


def test_baseline_is_empty_and_hashes_are_deterministic():
    spec,weights=synthetic(); derived,parameters,statistics=derive(spec,weights,C.Candidate())
    assert derived=={} and parameters['scales']=={} and parameters['clip_thresholds']=={}
    assert stats(spec,weights)==statistics and identity(stats(spec,weights))==identity(statistics)
    candidate=C.Candidate(0.5,99.99)
    d1,p1,s1=derive(spec,weights,candidate); d2,p2,s2=derive(spec,weights,candidate)
    assert identity(p1)==identity(p2) and identity(s1)==identity(s2)
    m1=C.tensor_manifest(d1,identity(p1),s1['source']); m2=C.tensor_manifest(d2,identity(p2),s2['source'])
    assert m1==m2 and identity(m1)==identity(m2)
    other,po,_=derive(spec,weights,C.Candidate(0.25,99.99))
    assert identity(po)!=identity(p1) and C.tensor_manifest(other,identity(po),s1['source'])!=m1
    assert len({c.candidate_id for c in C.all_candidates()})==12
    assert [c.name for c in C.all_candidates()][:4]==['a0-cnone','a0-c99.99','a0-c99.9','a0.25-cnone']


def test_artifacts_are_content_hashed_append_only_and_separate(tmp_path):
    spec,weights=synthetic(); derived,parameters,statistics=derive(spec,weights,C.Candidate(0.5,99.99))
    sid,spath=C.write_artifact(tmp_path,'calibration-statistics',statistics)
    pid,ppath=C.write_artifact(tmp_path,'candidate-parameters',parameters)
    did,dpath=C.write_artifact(tmp_path,'derived-tensors',C.tensor_manifest(derived,pid,statistics['source']))
    assert len({sid,pid,did})==3 and spath.name.endswith(sid+'.json') and identity(json.loads(dpath.read_text()))==did
    assert C.write_artifact(tmp_path,'candidate-parameters',parameters)==(pid,ppath)
    ppath.write_text('{}\n')
    with pytest.raises(ValueError,match='corruption'): C.write_artifact(tmp_path,'candidate-parameters',parameters)
    tensors=C.write_tensors(tmp_path,did,derived)
    from safetensors.numpy import load_file
    assert all(np.array_equal(v,derived[k]) for k,v in load_file(str(tensors)).items())
    with pytest.raises(ValueError,match='already exists'): C.write_tensors(tmp_path,did,derived)


def suite_file(tmp_path,info=None,vocab=128):
    rng=np.random.default_rng(11); make=lambda n,length: [rng.integers(0,vocab,length).tolist() for _ in range(n)]
    data={'schema_version':1,'base_model_id':(info or {}).get('base_model_id','m'),
          'tokenizer_id':(info or {}).get('tokenizer_id','t'),
          'calibration':make(3,40),'validation':make(8,129),'held-out':make(8,129)}
    names=('calibration','validation','held-out')
    data['freeze']={'split_hashes':{s:identity(data[s]) for s in names},
                    'target_counts':{s:sum(len(r)-1 for r in data[s]) for s in names}}
    path=tmp_path/'suite.json'; path.write_text(json.dumps(data)); return path,data


def test_calibration_inputs_use_only_the_calibration_split(tmp_path):
    path,data=suite_file(tmp_path)
    rows,split_hash=C.calibration_inputs(path,50,128)
    assert split_hash==identity(data['calibration']) and sum(map(len,rows))==50
    assert rows[0]==data['calibration'][0][:-1][:39] and rows[1]==data['calibration'][1][:-1][:11]
    assert C.calibration_inputs(path,50,128)==(rows,split_hash)
    with pytest.raises(ValueError,match='outside vocabulary'): C.calibration_inputs(path,50,3)
    used={t for row in rows for t in row}; assert used<=set(t for r in data['calibration'] for t in r)


def test_source_checkpoint_files_are_never_modified_and_runner_is_deterministic(tmp_path):
    from llm_fixture import save
    from malleable.llm.models import inspect,digest
    model=tmp_path/'model'; save(model)
    info=inspect(model,128); suite,_=suite_file(tmp_path,info)
    panel=make_panel(suite,128,'spread'); panel_file=tmp_path/'panel-128.json'; panel_file.write_text(json.dumps(panel))
    before={p.name:digest(p) for p in model.iterdir()}
    names=['a0-cnone','a0.5-cnone','a0.5-c99.9']
    candidates=C.parse_candidates(','.join(names)); source={'commit':'test','status':''}
    report=C.run_attempt(tmp_path/'attempt-1',model,suite,panel_file,candidates,source,calibration_tokens=60,persist_derived=True)
    assert {p.name:digest(p) for p in model.iterdir()}==before
    root=tmp_path/'attempt-1'; batch=json.loads((root/'batch.json').read_text())
    assert batch['held_out_used'] is False and batch['selection']=='validation-only' and batch['panel_id']==identity(panel)
    assert report['ranking'][0]==report['top_three'][0] and set(report['ranking'])==set(names)
    cases={c['name']:c for c in report['cases']}
    for n in names:
        assert cases[n]['target_count']==128 and cases[n]['executed_tokens']==panel['executed_tokens']
    ranked=[cases[n] for n in report['ranking']]
    assert [(-c['agreement'],c['candidate_nll']) for c in ranked]==sorted((-c['agreement'],c['candidate_nll']) for c in ranked)
    baseline=json.loads((root/'case-00'/'result.json').read_text())
    direct=diagnose_precision(model,suite,{'schema_version':1,'quantized_groups':list(GROUPS),'group_size':128},
        panel,root/'floating-references')
    assert baseline['agreement']==direct['agreement'] and baseline['candidate_nll']==direct['candidate_nll']
    assert cases['a0-cnone']['derived_tensor_count']==0
    assert cases['a0.5-c99.9']['derived_tensor_count']>=cases['a0.5-cnone']['derived_tensor_count']>0
    assert len({cases[n]['derived_id'] for n in names})==3
    assert cases['a0.5-cnone']['preserves_unquantized_operation'] and not cases['a0.5-c99.9']['preserves_unquantized_operation']
    assert not baseline['selectable'] and baseline['release_evidence'] is False
    for path in (root/'artifacts').glob('*.json'):
        assert path.stem.endswith(identity(json.loads(path.read_text())))
    assert list((root/'artifacts').glob('derived-*.safetensors'))
    with pytest.raises(ValueError,match='refusing to overwrite'):
        C.prepare_attempt(root,model,suite,panel_file,candidates,source)
    again=C.run_attempt(tmp_path/'attempt-2',model,suite,panel_file,candidates[1:2],source,calibration_tokens=60)
    first=cases['a0.5-cnone']; second=again['cases'][0]
    for key in ('calibration_statistics_id','parameters_id','derived_id','agreement','candidate_nll'):
        assert first[key]==second[key]
    tampered=json.loads(panel_file.read_text()); tampered['rows'][0]['tokens'][0]^=1
    panel_file.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='panel mismatch'):
        C.prepare_attempt(tmp_path/'attempt-3',model,suite,panel_file,candidates,source)
    assert not (tmp_path/'attempt-3').exists()


def test_candidate_selection_rejects_unknown_and_duplicates():
    for text in ('a0.6-cnone','a0-cnone,a0-cnone',''):
        with pytest.raises(ValueError): C.parse_candidates(text)
    assert len(C.parse_candidates('all-but-baseline'))==11
