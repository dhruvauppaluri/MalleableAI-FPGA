"""Compatible INT8 candidates: compensated channel rescaling and weight clipping.

Rescaling folds per-channel factors into the tensors around an activation
quantization site so the unquantized network is unchanged (SmoothQuant,
arXiv:2211.10438). Clipping is lossy and is labelled as such. The INT8 runtime
quantizer and storage are untouched: a candidate is only a different set of
float32 tensors. Statistics come from the calibration split, candidates are
ranked on the validation panel, and source checkpoints are never modified.
"""
from dataclasses import dataclass
import hashlib
import gc
import math
import json
from pathlib import Path
import shutil
import time

from ..records import canonical, identity

STRENGTHS=(0.0,0.25,0.5,0.75)
CLIPS=(None,99.99,99.9)
SITES=('attn_input','mlp_input','attn_output','mlp_down','qk','head_input')
DEFAULT_SITES=SITES[:5]
CLIP_GROUPS=('transformer_weights','head_weights')
SCALE_LIMIT=256.0
FLOOR=1e-6
CONTEXT=128
LAYER_WEIGHTS=('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
               'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')
STAT_SITES={'attention_input':'attn_input','mlp_input':'mlp_input','attention_output':'attn_output',
            'value_store':'attn_output','mlp_down':'mlp_down','key_store':'qk','query':'qk','head_input':'head_input'}


@dataclass(frozen=True)
class Candidate:
    strength: float = 0.0
    clip: float | None = None
    sites: tuple = DEFAULT_SITES
    clip_groups: tuple = ('transformer_weights',)

    def __post_init__(self):
        if (self.strength not in STRENGTHS or self.clip not in CLIPS
            or any(s not in SITES for s in self.sites) or len(set(self.sites))!=len(self.sites)
            or any(g not in CLIP_GROUPS for g in self.clip_groups)):
            raise ValueError('unsupported INT8 candidate')

    @property
    def name(self):
        return f"a{self.strength:g}-c{'none' if self.clip is None else format(self.clip,'g')}"

    @property
    def preserves_unquantized_operation(self): return self.clip is None

    def record(self):
        return {'schema_version':1,'name':self.name,'strength':self.strength,'clip_percentile':self.clip,
                'sites':[s for s in SITES if s in self.sites],'clip_groups':list(self.clip_groups),
                'preserves_unquantized_operation':self.preserves_unquantized_operation,
                'scale_limit':SCALE_LIMIT,'floor':FLOOR}

    @property
    def candidate_id(self): return identity(self.record())


def all_candidates(include_baseline=True):
    result=[Candidate(a,c) for a in STRENGTHS for c in CLIPS]
    return result if include_baseline else result[1:]


def parse_candidates(text):
    if text in ('all','all-but-baseline'): return all_candidates(text=='all')
    by_name={c.name:c for c in all_candidates()}
    names=[n for n in text.split(',') if n]
    if not names or len(set(names))!=len(names) or any(n not in by_name for n in names):
        raise ValueError('unknown or duplicate candidate; choose from '+', '.join(by_name))
    return [by_name[n] for n in names]


def _np():
    import numpy as np
    return np


def layer_name(i,name): return f'model.layers.{i}.{name}'


def head_name(spec): return 'model.embed_tokens.weight' if spec.tied else 'lm_head.weight'


def check_candidate(spec,candidate):
    if spec.tied and ('head_input' in candidate.sites or 'head_weights' in candidate.clip_groups):
        raise ValueError('tied embedding/head: rescaling or clipping the shared tensor would change '
            'the input embeddings; untying is a storage change that needs an ADR (see ADR-0005 draft)')
    if spec.head_dim%2 or spec.n_q%spec.n_kv: raise ValueError('unsupported attention shape')


# ------------------------------------------------------------------ calibration statistics
class ChannelMaxima:
    """Duck-typed OperatorStats sink: channelwise max |activation| per rescaling site."""
    def __init__(self): self.rows={}

    def observe(self,key,original,result,quantized,block):
        group,site,layer=key
        if site not in STAT_SITES: return
        np=_np(); value=np.abs(np.asarray(original,np.float64)).reshape(-1)
        name=f"{site}/{'head' if layer is None else layer}"
        previous=self.rows.get(name)
        self.rows[name]=value if previous is None else np.maximum(previous,value)


def calibration_inputs(suite,max_tokens,vocab=None):
    """Causal inputs from the calibration split only, in file order, truncated deterministically."""
    from .quality import frozen_suite
    data,hashes=frozen_suite(suite)
    if type(max_tokens) is not int or max_tokens<1: raise ValueError('calibration token budget must be positive')
    rows=[]; remaining=max_tokens
    for row in data['calibration']:
        tokens=row[:-1][:min(CONTEXT,remaining)]
        if not tokens: continue
        rows.append(tokens); remaining-=len(tokens)
        if remaining<=0: break
    if not rows or (vocab is not None and any(t>=vocab for r in rows for t in r)):
        raise ValueError('calibration split empty or outside vocabulary')
    return rows,hashes['calibration']


def collect_statistics(spec,weights,sequences,source,calibration_hash,group_size=128):
    from .precision import controlled_reference
    sink=ChannelMaxima()
    forward,emulation=controlled_reference({'schema_version':1,'quantized_groups':[],'group_size':group_size},sink)
    for tokens in sequences: forward(spec,weights,tokens,D=group_size,wformat='int8',head_format='int8')
    return {'schema_version':1,'kind':'int8-calibration-statistics','source':source,'split':'calibration',
        'calibration_split_hash':calibration_hash,'sequences':len(sequences),'tokens':sum(map(len,sequences)),
        'calibration_inputs_hash':identity(sequences),'group_size':group_size,
        'statistic':'channelwise maximum absolute value of the unquantized activation at each site',
        'emulation_source_sha256':emulation,
        'channels':{k:[float(x) for x in v] for k,v in sorted(sink.rows.items())}}


# ------------------------------------------------------------------ fitting
def _smooth(act,weight,alpha):
    np=_np(); s=np.ones_like(act); live=(act>FLOOR)&(weight>FLOOR)
    s[live]=act[live]**alpha/weight[live]**(1-alpha)
    return np.clip(s,1/SCALE_LIMIT,SCALE_LIMIT)


def fit_scales(spec,weights,statistics,candidate):
    """SmoothQuant factors s = max|X|^a / max|W|^(1-a); shared consumers use the joint weight maximum."""
    check_candidate(spec,candidate)
    if candidate.strength==0: return {}
    np=_np(); d,G,a=spec.head_dim,spec.n_q//spec.n_kv,candidate.strength; channels=statistics['channels']
    act=lambda site,i: np.asarray(channels[f'{site}/{i}'],np.float64)
    column=lambda n: np.abs(np.asarray(weights[n],np.float64)).max(0)
    def kv_group(v): return v.reshape(spec.n_kv,G,d).max(1)
    out={}
    for i in range(spec.layers):
        n=lambda x: layer_name(i,x)
        if 'attn_input' in candidate.sites:
            out[f'attn_input/{i}']=_smooth(act('attention_input',i),
                np.max([column(n(f'self_attn.{x}_proj.weight')) for x in 'qkv'],0),a)
        if 'mlp_input' in candidate.sites:
            out[f'mlp_input/{i}']=_smooth(act('mlp_input',i),
                np.max([column(n(f'mlp.{x}_proj.weight')) for x in ('gate','up')],0),a)
        if 'attn_output' in candidate.sites:
            o=act('attention_output',i).reshape(spec.n_q,d).reshape(spec.n_kv,G,d).max(1)
            v=act('value_store',i).reshape(spec.n_kv,d)
            out[f'attn_output/{i}']=_smooth(np.maximum(o,v).reshape(-1),
                kv_group(column(n('self_attn.o_proj.weight'))).reshape(-1),a)
        if 'mlp_down' in candidate.sites:
            out[f'mlp_down/{i}']=_smooth(act('mlp_down',i),column(n('mlp.down_proj.weight')),a)
        if 'qk' in candidate.sites:
            h=d//2; k=act('key_store',i).reshape(spec.n_kv,d).max(0); q=act('query',i)
            half=_smooth(np.maximum(k[:h],k[h:]),np.maximum(q[:h],q[h:]),a)
            out[f'qk/{i}']=np.concatenate([half,half])
    if 'head_input' in candidate.sites:
        out['head_input/head']=_smooth(act('head_input','head'),column(head_name(spec)),a)
    return {k:[float(x) for x in v] for k,v in out.items()}


def layer_tensors(weights,spec,i,scales):
    """Float64 rescaled tensors of layer i that the enabled sites touch (compensated pairs)."""
    np=_np(); d,G=spec.head_dim,spec.n_q//spec.n_kv; changed={}
    n=lambda x: layer_name(i,x)
    get=lambda name: changed[name] if name in changed else np.asarray(weights[name],np.float64).copy()
    vec=lambda site: np.asarray(scales[f'{site}/{i}'],np.float64) if f'{site}/{i}' in scales else None
    if (s:=vec('attn_input')) is not None:
        changed[n('input_layernorm.weight')]=get(n('input_layernorm.weight'))/s
        for x in 'qkv': changed[n(f'self_attn.{x}_proj.weight')]=get(n(f'self_attn.{x}_proj.weight'))*s
    if (s:=vec('mlp_input')) is not None:
        changed[n('post_attention_layernorm.weight')]=get(n('post_attention_layernorm.weight'))/s
        for x in ('gate','up'): changed[n(f'mlp.{x}_proj.weight')]=get(n(f'mlp.{x}_proj.weight'))*s
    if (s:=vec('attn_output')) is not None:
        changed[n('self_attn.v_proj.weight')]=get(n('self_attn.v_proj.weight'))/s[:,None]
        full=np.repeat(s.reshape(spec.n_kv,1,d),G,axis=1).reshape(-1)
        changed[n('self_attn.o_proj.weight')]=get(n('self_attn.o_proj.weight'))*full[None,:]
    if (s:=vec('mlp_down')) is not None:
        changed[n('mlp.up_proj.weight')]=get(n('mlp.up_proj.weight'))/s[:,None]
        changed[n('mlp.down_proj.weight')]=get(n('mlp.down_proj.weight'))*s[None,:]
    if (s:=vec('qk')) is not None:
        if not np.array_equal(s[:d//2],s[d//2:]): raise ValueError('query/key scales must be equal across each RoPE pair')
        changed[n('self_attn.q_norm.weight')]=get(n('self_attn.q_norm.weight'))*s
        changed[n('self_attn.k_norm.weight')]=get(n('self_attn.k_norm.weight'))/s
    return changed


def head_tensors(weights,spec,scales):
    np=_np()
    if 'head_input/head' not in scales: return {}
    if spec.tied: raise ValueError('tied embedding/head cannot be rescaled without untying')
    s=np.asarray(scales['head_input/head'],np.float64)
    return {'model.norm.weight':np.asarray(weights['model.norm.weight'],np.float64)/s,
            'lm_head.weight':np.asarray(weights['lm_head.weight'],np.float64)*s[None,:]}


def clip_targets(spec,candidate):
    if candidate.clip is None: return []
    names=[layer_name(i,x) for i in range(spec.layers) for x in LAYER_WEIGHTS] \
        if 'transformer_weights' in candidate.clip_groups else []
    return names+([head_name(spec)] if 'head_weights' in candidate.clip_groups else [])


def _stages(weights,spec,scales):
    """Yield (changed float64 tensors) per layer, then the head; the caller owns memory release."""
    for i in range(spec.layers): yield layer_tensors(weights,spec,i,scales)
    yield head_tensors(weights,spec,scales)


def fit_parameters(weights,spec,statistics,candidate,statistics_id):
    """Calibration parameters: scales plus per-tensor clip thresholds (weight-only percentiles)."""
    np=_np(); scales=fit_scales(spec,weights,statistics,candidate); targets=set(clip_targets(spec,candidate)); clip={}
    if targets:
        for changed in _stages(weights,spec,scales):
            for name in [t for t in targets if t in changed]:
                clip[name]=float(np.quantile(np.abs(changed[name]),candidate.clip/100))
        for name in sorted(targets-set(clip)):
            clip[name]=float(np.quantile(np.abs(np.asarray(weights[name],np.float64)),candidate.clip/100))
    return {'schema_version':1,'kind':'int8-candidate-parameters','candidate':candidate.record(),
        'candidate_id':candidate.candidate_id,'calibration_statistics_id':statistics_id,
        'source':statistics['source'],'scales':scales,'clip_thresholds':dict(sorted(clip.items())),
        'clip_scope':'per-tensor percentile of |weight| after rescaling; data-free'}


def derive_tensors(weights,spec,parameters,dtype=None):
    """New tensors only for names the candidate touches; the input mapping is never mutated."""
    np=_np(); dtype=np.float32 if dtype is None else dtype; scales=parameters['scales']; thresholds=parameters['clip_thresholds']
    derived={}
    for changed in _stages(weights,spec,scales):
        for name in list(changed):
            if name in thresholds: changed[name]=np.clip(changed[name],-thresholds[name],thresholds[name])
        for name,value in changed.items(): derived[name]=value.astype(dtype)
    for name in sorted(set(thresholds)-set(derived)):
        derived[name]=np.clip(np.asarray(weights[name],np.float64),-thresholds[name],thresholds[name]).astype(dtype)
    tiny=np.finfo(np.float32).tiny
    for name,value in derived.items():
        if not np.isfinite(value).all(): raise ValueError('nonfinite derived tensor: '+name)
        if name.endswith('norm.weight') or name.endswith('layernorm.weight'):
            source=np.asarray(weights[name],np.float64)
            if np.any((source!=0)&(np.abs(value)<tiny)): raise ValueError('derived norm weight would flush to zero: '+name)
    return dict(sorted(derived.items()))


def tensor_manifest(tensors,parameters_id,source):
    np=_np(); rows={}
    for name,value in sorted(tensors.items()):
        value=np.ascontiguousarray(value)
        rows[name]={'dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    return {'schema_version':1,'kind':'int8-derived-tensors','source':source,'parameters_id':parameters_id,
        'tensor_count':len(rows),'tensors':rows}


def write_artifact(root,kind,payload):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); key=identity(payload)
    path=root/f'{kind}-{key}.json'
    if path.exists():
        if path.read_text()!=canonical(payload)+'\n': raise ValueError('artifact hash collision or corruption: '+path.name)
    else:
        with path.open('x') as stream: stream.write(canonical(payload)+'\n')
    return key,path


def write_tensors(root,derived_id,tensors):
    from safetensors.numpy import save_file
    path=Path(root)/f'derived-{derived_id}.safetensors'
    if path.exists(): raise ValueError('derived tensor file already exists: '+path.name)
    save_file(tensors,str(path),metadata={'derived_id':derived_id}); return path


def merged(weights,derived):
    """Non-mutating view for a forward pass: derived tensors override the source."""
    return {**weights,**derived}


def read_derived_candidate(case,info):
    """Verify a completed screen's lineage before loading any derived tensors."""
    from safetensors import safe_open
    from .models import local_file
    case=Path(case).resolve(); root=case.parent; artifacts=(root/'artifacts').resolve()
    meta=_read(case/'candidate.json'); summary=_read(case/'summary.json')
    result_file=case/'result.json'
    if _digest(result_file)!=summary['result_sha256']: raise ValueError('candidate screen result hash mismatch')
    result=_read(result_file)
    if (result.get('status')!='completed' or not screen_passes(summary)
        or any(result.get(k)!=summary.get(k) for k in ('agreement','float_nll','candidate_nll','target_count','nll_degradation_percent'))
        or any(result.get(k)!=info.get(k) for k in ('base_model_id','tokenizer_id'))):
        raise ValueError('candidate screen is not eligible for validation')
    def artifact(kind,key):
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid candidate artifact identity')
        obj=_read(local_file(artifacts,f'{kind}-{key}.json'))
        if identity(obj)!=key: raise ValueError('candidate artifact hash mismatch')
        return obj
    statistics=artifact('calibration-statistics',meta['calibration_statistics_id'])
    parameters=artifact('candidate-parameters',meta['parameters_id'])
    manifest=artifact('derived-tensors',meta['derived_id'])
    lineage={k:info[k] for k in ('base_model_id','tokenizer_id','weight_files')}
    if (statistics.get('source')!=lineage or statistics.get('split')!='calibration'
        or parameters.get('source')!=lineage or manifest.get('source')!=lineage
        or parameters.get('calibration_statistics_id')!=meta['calibration_statistics_id']
        or manifest.get('parameters_id')!=meta['parameters_id']
        or parameters.get('candidate')!=meta['candidate']
        or any(result.get(k)!=meta.get(k) for k in ('derived_id','parameters_id','calibration_statistics_id'))):
        raise ValueError('derived candidate lineage mismatch')
    tensor_file=local_file(artifacts,meta['tensor_file'])
    if tensor_file.suffix!='.safetensors': raise ValueError('derived candidate requires safetensors')
    count=0
    with safe_open(str(tensor_file),framework='np') as tensors:
        if (tensors.metadata() or {}).get('derived_id')!=meta['derived_id'] or set(tensors.keys())!=set(manifest['tensors']):
            raise ValueError('derived tensor manifest mismatch')
        for name,row in manifest['tensors'].items():
            view=tensors.get_slice(name)
            if row['dtype']!='float32' or view.get_dtype()!='F32' or view.get_shape()!=row['shape']:
                raise ValueError('derived tensor shape/dtype mismatch')
            size=4
            for dim in row['shape']: size*=dim
            count+=size
    return {'schema_version':1,'derived_id':meta['derived_id'],'parameters_id':meta['parameters_id'],
        'calibration_statistics_id':meta['calibration_statistics_id'],'candidate':meta['candidate'],
        'tensor_file':tensor_file,'manifest':manifest,'bytes':count,'screen_result_sha256':summary['result_sha256']}


def apply_derived_candidate(weights,candidate):
    """Hash/shape checks apply to every tensor before the actual ISA uses it."""
    from safetensors.numpy import load_file
    np=_np(); derived=load_file(str(candidate['tensor_file']))
    for name,value in derived.items():
        row=candidate['manifest']['tensors'][name]
        if (name not in weights or value.shape!=weights[name].shape or value.dtype!=np.float32
            or not np.isfinite(value).all() or hashlib.sha256(value.tobytes()).hexdigest()!=row['sha256']):
            raise ValueError('invalid or corrupted derived tensor: '+name)
    return merged(weights,derived)


# ------------------------------------------------------------------ screening attempts
def _write(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def _read(path): return json.loads(Path(path).read_text())


def _digest(path):
    from .models import digest
    return digest(path)


def prepare_attempt(root,model,suite,panel_file,candidates,source,reference_store=None,calibration_tokens=512,
                    persist_derived=False,authorization=''):
    """Create a fresh append-only attempt folder; refuses any existing path."""
    from .models import inspect
    from .precision import make_panel
    root=Path(root).resolve()
    if root.exists(): raise ValueError('refusing to overwrite previous candidate attempt')
    panel=_read(panel_file); expected=make_panel(suite,128,'spread')
    if panel!=expected or panel['split']!='validation': raise ValueError('frozen 128-target validation panel mismatch')
    info=inspect(model,128)
    if info['family']!='qwen3' or not info['supported']: raise ValueError('INT8 candidates currently support Qwen3')
    if any(info[k]!=panel[k] for k in ('base_model_id','tokenizer_id')):
        raise ValueError('candidate model/tokenizer does not match frozen panel')
    if len({c.name for c in candidates})!=len(candidates) or not candidates: raise ValueError('candidate list must be unique and nonempty')
    root.mkdir(parents=True)
    _write(root/'panel-128.json',panel)
    if reference_store is not None: shutil.copytree(reference_store,root/'floating-references')
    _write(root/'batch.json',{'schema_version':1,'kind':'int8-candidate-screen','source':source,'model':str(Path(model).resolve()),
        'suite':str(Path(suite).resolve()),'suite_file_hash':_digest(suite),'panel_id':identity(panel),
        'panel_token_hash':panel['token_hash'],'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
        'calibration_tokens':calibration_tokens,'persist_derived':bool(persist_derived),
        'candidates':[c.record() for c in candidates],'authorization':authorization,
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'release_evidence':False,'selection':'validation-only',
        'held_out_used':False,'ranking_rule':'agreement descending, candidate NLL ascending, listed order'})
    return root


def _load_batch(root):
    batch=_read(Path(root)/'batch.json'); source={'base_model_id':batch['base_model_id'],'tokenizer_id':batch['tokenizer_id']}
    return batch,source


def run_calibration(root,source=None):
    from .models import inspect,load
    from opentpu.llm import load_spec
    root=Path(root); batch,identity_fields=_load_batch(root)
    info=inspect(batch['model'],128); spec=load_spec(Path(batch['model'])); weights=load(batch['model'])
    sequences,split_hash=calibration_inputs(batch['suite'],batch['calibration_tokens'],spec.vocab)
    started=time.monotonic()
    statistics=collect_statistics(spec,weights,sequences,dict(identity_fields,weight_files=info['weight_files']),split_hash)
    key,path=write_artifact(root/'artifacts','calibration-statistics',statistics)
    _write(root/'calibration.json',{'statistics_id':key,'file':path.name,'tokens':statistics['tokens'],
        'calibration_split_hash':split_hash,'seconds':time.monotonic()-started,'source':source or {}})
    return key


def run_case(root,index,source=None,emit=lambda *_:None):
    from .models import inspect,load
    from .precision import GROUPS,diagnose_precision
    from ..store import Store
    from opentpu.llm import load_spec
    root=Path(root); batch,identity_fields=_load_batch(root); record=batch['candidates'][index]
    candidate=Candidate(record['strength'],record['clip_percentile'],tuple(record['sites']),tuple(record['clip_groups']))
    if candidate.record()!=record: raise ValueError('candidate record tampered')
    case=root/f'case-{index:02d}'; case.mkdir()
    spec=load_spec(Path(batch['model'])); weights=load(batch['model']); info=inspect(batch['model'],128)
    calibration=_read(root/'calibration.json'); statistics=_read(root/'artifacts'/calibration['file'])
    if identity(statistics)!=calibration['statistics_id']: raise ValueError('calibration statistics artifact hash mismatch')
    parameters=fit_parameters(weights,spec,statistics,candidate,calibration['statistics_id'])
    parameters_id,_=write_artifact(root/'artifacts','candidate-parameters',parameters)
    derived=derive_tensors(weights,spec,parameters)
    manifest=tensor_manifest(derived,parameters_id,statistics['source'])
    derived_id,_=write_artifact(root/'artifacts','derived-tensors',manifest)
    tensor_file=None
    if batch['persist_derived'] and derived: tensor_file=write_tensors(root/'artifacts',derived_id,derived).name
    _write(case/'candidate.json',{'candidate':record,'calibration_statistics_id':calibration['statistics_id'],
        'parameters_id':parameters_id,'derived_id':derived_id,'derived_tensor_count':len(derived),'tensor_file':tensor_file})
    # The diagnostic reloads the original checkpoint for its reference. Retain
    # only the derived tensors here, rather than a second full source mapping.
    del weights
    gc.collect()
    panel=_read(root/'panel-128.json')
    policy={'schema_version':1,'quantized_groups':list(GROUPS),'group_size':128}
    with (case/'events.jsonl').open('x') as log:
        def event(kind,payload):
            log.write(json.dumps({'kind':kind,'payload':payload},allow_nan=False)+'\n'); log.flush(); emit(kind,payload)
        result=diagnose_precision(batch['model'],batch['suite'],policy,panel,root/'floating-references',
            emit=event,candidate_weights=derived)
    result.update(kind='int8-candidate-screen',candidate=record,candidate_id=candidate.candidate_id,
        calibration_statistics_id=calibration['statistics_id'],parameters_id=parameters_id,derived_id=derived_id,
        derived_tensor_count=len(derived),source=source or {},status='completed',selectable=False,release_evidence=False,
        scope='independent float64 INT8 emulation of derived tensors against the original FP32 source; validation panel only, no ISA/RTL')
    store=Store(root/'research')
    try: key=store.save('llm-int8-candidate-screen',result)
    finally: store.close()
    _write(case/'result.json',result)
    summary={k:result[k] for k in ('agreement','float_nll','candidate_nll','nll_degradation_percent','candidate_seconds',
        'elapsed_seconds','target_count','executed_tokens')}
    summary.update(name=candidate.name,case=index,record_id=key,result_sha256=_digest(case/'result.json'),
        matches=round(result['agreement']*result['target_count']),calibration_statistics_id=calibration['statistics_id'],
        parameters_id=parameters_id,derived_id=derived_id,derived_tensor_count=len(derived),
        preserves_unquantized_operation=candidate.preserves_unquantized_operation)
    _write(case/'summary.json',summary); return summary


def rank(summaries):
    order=sorted(range(len(summaries)),key=lambda i:(-summaries[i]['agreement'],summaries[i]['candidate_nll'],i))
    return [summaries[i] for i in order]


def screen_passes(summary):
    """Screening eligibility only; this is never ISA/RTL or held-out approval."""
    fields=('agreement','candidate_nll','float_nll','nll_degradation_percent')
    return (all(isinstance(summary.get(k),(int,float)) and math.isfinite(summary[k]) for k in fields)
        and summary.get('target_count')==128 and 0.9<=summary['agreement']<=1
        and summary['candidate_nll']>=0 and summary['float_nll']>0
        and summary['candidate_nll']<=1.05*summary['float_nll']
        and summary['nll_degradation_percent']<=5
        and math.isclose(summary['nll_degradation_percent'],
            100*(summary['candidate_nll']-summary['float_nll'])/summary['float_nll'],abs_tol=1e-8))


def write_report(root,source=None,elapsed=None):
    root=Path(root); batch,_=_load_batch(root)
    summaries=[_read(root/f'case-{i:02d}'/'summary.json') for i in range(len(batch['candidates']))]
    for i,summary in enumerate(summaries):
        if _digest(root/f'case-{i:02d}'/'result.json')!=summary['result_sha256']:
            raise ValueError('candidate result hash mismatch')
        summary['screen_passes']=screen_passes(summary)
    ranked=rank(summaries)
    eligible=[s for s in ranked if s['screen_passes']]
    report={'schema_version':1,'status':'completed','source':source or batch['source'],'panel_id':batch['panel_id'],
        'calibration_statistics_id':_read(root/'calibration.json')['statistics_id'],'cases':summaries,
        'ranking':[s['name'] for s in ranked],'top_three':[s['name'] for s in eligible[:3]],
        'screen_passes':bool(eligible),'screen_gate':{'agreement_min':0.9,'nll_degradation_max_percent':5,'targets':128},
        'elapsed_seconds':elapsed,'release_evidence':False,'held_out_used':False,
        'next':('complete ISA validation of eligible top_three; freeze before held-out' if eligible
            else 'screen failed; preserve results and run the authorized precision diagnostics')}
    _write(root/'report.json',report); return report


def run_attempt(root,model,suite,panel_file,candidates,source,reference_store=None,calibration_tokens=512,
                persist_derived=False,emit=lambda *_:None):
    """Serialized in-process screening; the tool wrapper runs the same stages in subprocesses."""
    started=time.monotonic()
    prepare_attempt(root,model,suite,panel_file,candidates,source,reference_store,calibration_tokens,persist_derived)
    run_calibration(root,source)
    for index in range(len(candidates)): run_case(root,index,source,emit)
    return write_report(root,source,time.monotonic()-started)
