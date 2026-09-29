"""Quality is evidence, not an inferred consequence of bit-exact arithmetic."""
import json
import math
from pathlib import Path
from ..records import identity
from .models import inspect, load, digest
from .records import PERSONALITIES

MIN_RELEASE_TARGETS=1024

def gate(float_nll, candidate_nll, agreement):
    if not all(math.isfinite(v) for v in (float_nll,candidate_nll,agreement)) \
        or float_nll<=0 or candidate_nll<0 or not 0<=agreement<=1:
        raise ValueError('invalid quality measurements')
    degradation=(candidate_nll-float_nll)/float_nll
    return {'nll_degradation':degradation,'next_token_agreement':agreement,
            'passed':candidate_nll<=float_nll*1.05 and .9<=agreement<=1}

def selection_approval(split,is_frozen,target_count,passed):
    """Held-out promotion only; validation measurements stay search-only."""
    return bool(split=='held-out' and is_frozen and type(target_count) is int
                and target_count>=MIN_RELEASE_TARGETS and passed)


def quality_matches(quality,result,split='validation',frozen_reference=None):
    """Recompute approval and bind it to the exact model/tokenizer/configuration."""
    fields=('base_model_id','tokenizer_id','variant_id','personality','configuration_id')
    if any(not quality.get(k) or quality.get(k)!=result.get(k) for k in fields): return False
    if (quality.get('split')!=split or quality.get('suite_frozen') is not True
        or type(quality.get('target_count')) is not int or quality['target_count']<MIN_RELEASE_TARGETS
        or quality.get('samples')!=quality['target_count'] or not quality.get('suite_file_hash')
        or set(quality.get('suite_hashes',{}))!={'calibration','validation','held-out'}): return False
    if frozen_reference and any(quality.get(k)!=frozen_reference.get(k) for k in ('suite_file_hash','suite_hashes')):
        return False
    try: passed=gate(quality['float_nll'],quality['candidate_nll'],quality['agreement'])['passed']
    except (KeyError,ValueError,TypeError): return False
    return passed and (split!='held-out' or quality.get('selectable') is True)

def suites(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema_version')!=1: raise ValueError('unsupported quality suite version')
    sets={}; hashes={}
    for split in ('calibration','validation','held-out'):
        rows=data.get(split)
        if not isinstance(rows,list) or not rows or any(not isinstance(r,list) or len(r)<2
            or len(r)>2048 or any(type(t) is not int or t<0 for t in r) for r in rows):
            raise ValueError('nonempty bounded token sequences required: '+split)
        keys={identity(r) for r in rows}
        if len(keys)!=len(rows): raise ValueError('duplicate quality sequence')
        if any(keys & previous for previous in sets.values()): raise ValueError('quality split leakage')
        sets[split]=keys; hashes[split]=identity(rows)
    return data,hashes

def frozen_suite(path):
    data,hashes=suites(path)
    counts={s:sum(len(row)-1 for row in data[s]) for s in hashes}
    if any(counts[s]<MIN_RELEASE_TARGETS for s in ('validation','held-out')):
        raise ValueError('validation and held-out each require at least 1,024 target tokens')
    if data.get('freeze')!={'split_hashes':hashes,'target_counts':counts}:
        raise ValueError('quality suite must be frozen before configuration search')
    return data,hashes

def floating_reference(model,info,weights):
    """Load only Qwen3.5's text tower; never initialize/download vision code."""
    import torch
    from transformers import AutoModelForCausalLM
    if info['family']!='qwen35':
        return AutoModelForCausalLM.from_pretrained(str(model),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,dtype=torch.float32,
            attn_implementation='eager').eval()
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    config=Qwen3_5TextConfig(**info['config'].get('text_config',info['config']))
    config._attn_implementation='eager'
    with torch.device('meta'): original=Qwen3_5ForCausalLM(config)
    tensors={k:torch.from_numpy(v) for k,v in weights.items()}
    if config.tie_word_embeddings and 'lm_head.weight' not in tensors:
        tensors['lm_head.weight']=tensors['model.embed_tokens.weight']
    original.load_state_dict(tensors,strict=True,assign=True)
    original.tie_weights()
    return original.eval()

def evaluate(model,path,wformat='int8',split='validation',personality='balanced',max_host_gib=12,
             cache_root=None,emit=lambda *_:None,context=2048):
    import numpy as np
    import torch
    import transformers
    import platform
    from ..store import Store
    from . import upstream
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine
    if split not in ('validation','held-out'): raise ValueError('invalid quality split')
    data,hashes=suites(path); info=inspect(model)
    if not info['supported']: raise ValueError('unsupported model')
    if data.get('base_model_id')!=info['base_model_id'] or data.get('tokenizer_id')!=info['tokenizer_id']:
        raise ValueError('quality suite model/tokenizer lineage mismatch')
    if wformat not in ('int8','int4','fp4'): raise ValueError('unsupported quantization')
    spec=load_spec(Path(model))
    if type(context) is not int or context%128 or not 128<=context<=2048: raise ValueError('invalid quality context')
    if max(len(row)-1 for row in data[split])>context: raise ValueError('quality sequence exceeds configured context')
    cfg=PERSONALITIES[personality].config(spec,context,wformat)
    estimate=2*info['fp32_tensor_bytes']+cfg.DRAM_BYTES+max(map(len,data[split]))*spec.vocab*8
    if estimate>max_host_gib*1024**3: raise ValueError('quality reference exceeds configured host memory budget')
    W=load(model)
    if any(t>=spec.vocab for row in data[split] for t in row): raise ValueError('quality token out of range')
    provenance=dict(schema_version=1,base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],
        suite_file_hash=digest(path),split=split,split_hash=hashes[split],dtype='float32',attention='eager',
        torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
        contract='original-local-safetensors-teacher-forced-v1')
    reference_id=identity(provenance)
    cache=Store(cache_root or Path(path).parent/'floating-references')
    try:
        cached=next((r for r in cache.records('llm-float-reference') if r.get('reference_id')==reference_id),None)
        if cached is None:
            emit('phase',{'phase':'floating-reference','split':split,'reference_id':reference_id})
            # This reference never participates in an RTL inference run. Only
            # target losses/top1 are cached, not large vocabulary logit arrays.
            original=floating_reference(model,info,W); rows=[]
            with torch.no_grad():
                for index,row in enumerate(data[split]):
                    logits=original(torch.tensor([row[:-1]])).logits[0].float()
                    if not torch.isfinite(logits).all(): raise ValueError('non-finite floating reference logits')
                    losses=torch.nn.functional.cross_entropy(logits.double(),torch.tensor(row[1:]),reduction='none')
                    rows.append({'nll':losses.tolist(),'top1':logits.argmax(-1).tolist()})
                    emit('quality-progress',{'phase':'floating-reference','completed':index+1,'total':len(data[split])})
            cached={'schema_version':1,'reference_id':reference_id,'provenance':provenance,'rows':rows}
            reference_record=cache.save('llm-float-reference',cached)
            del original,logits
        else: reference_record=identity(cached)
    finally: cache.close()
    engine=Engine(spec,W,cap=context,cfg=cfg,
                  backend='isa',rows=1,pipeline=False,wformat=wformat,head_format='int8')
    fnll=qnll=agree=count=0
    def nll(logits,target):
        x=logits.astype(np.float64); peak=x.max()
        return float(peak+np.log(np.exp(x-peak).sum())-x[target])
    with torch.no_grad():
        for row_index,row in enumerate(data[split]):
            reference=cached['rows'][row_index]
            if len(reference['nll'])!=len(row)-1 or len(reference['top1'])!=len(row)-1:
                raise ValueError('invalid cached reference dimensions')
            engine.reset()
            for i,token in enumerate(row[:-1]):
                q=engine.step(token)[:spec.vocab]
                if not np.isfinite(q).all(): raise ValueError('non-finite candidate logits')
                fnll+=reference['nll'][i]; qnll+=nll(q,row[i+1]); agree+=int(reference['top1'][i]==q.argmax()); count+=1
                if count%16==0: emit('quality-progress',{'phase':'candidate','completed_targets':count,
                    'total_targets':sum(len(r)-1 for r in data[split])})
    checks=gate(fnll/count,qnll/count,agree/count)
    target_count=sum(len(row)-1 for row in data[split])
    is_frozen=data.get('freeze')=={'split_hashes':hashes,'target_counts':{
        s:sum(len(r)-1 for r in data[s]) for s in hashes}}
    # Validation is for search only. A candidate becomes selectable only after
    # the untouched, frozen held-out split independently meets both thresholds.
    selectable=selection_approval(split,is_frozen,target_count,checks['passed'])
    return dict(schema_version=1,base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],
        variant_id=identity({'base':info['base_model_id'],'format':wformat,'head':'int8'}),
        wformat=wformat,head_format='int8',personality=personality,split=split,suite_hashes=hashes,suite_file_hash=digest(path),
        samples=count,float_nll=fnll/count,candidate_nll=qnll/count,agreement=agree/count,
        configuration_id=identity({'config':cfg.__dict__,'uarch':PERSONALITIES[personality].uarch}),
        context=context,config=cfg.__dict__,microarchitecture={'schema_version':1,'parameters':PERSONALITIES[personality].uarch},
        suite_frozen=is_frozen, target_count=target_count,
        floating_reference_id=reference_id,floating_reference_record=reference_record,toolchain=provenance,
        **checks,provenance='measured-isa-versus-original-float',selectable=selectable,
        selection_rule='held-out-only; frozen suite; >=1024 targets; NLL <=5%; next-token agreement >=90%')
