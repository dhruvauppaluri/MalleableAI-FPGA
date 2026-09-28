"""Read-only local checkpoint inspection and strict shard validation."""
import hashlib
import json
from pathlib import Path
from . import upstream
from .records import PERSONALITIES, CONTRACT
from ..records import identity

SUPPORTED = {'qwen3':'qwen3','qwen3_5':'qwen35','qwen3_5_text':'qwen35','lfm2':'lfm2'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def local_file(root,name):
    if not isinstance(name,str) or Path(name).is_absolute(): raise ValueError('invalid checkpoint file')
    p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError('checkpoint path escapes root or missing: '+name)
    return p

def checkpoint(root):
    from safetensors import safe_open
    root=Path(root).resolve()
    c=json.loads(local_file(root,'config.json').read_text())
    if not isinstance(c,dict): raise ValueError('configuration must be an object')
    index=root/'model.safetensors.index.json'
    weight_map=None
    if index.exists():
        index=local_file(root,index.name)
        weight_map=json.loads(index.read_text()).get('weight_map')
        if not isinstance(weight_map,dict) or not weight_map: raise ValueError('invalid shard index')
        if any(not isinstance(k,str) or not isinstance(v,str) for k,v in weight_map.items()):
            raise ValueError('invalid shard index entry')
        files=[local_file(root,n) for n in sorted(set(weight_map.values()))]
    else:
        files=[local_file(root,n.name) for n in sorted(root.glob('*.safetensors'))]
    if not files: raise ValueError('local Safetensors checkpoint required; downloads are disabled')
    tensors={}; owners={}; bytes32=0
    for file in files:
        if file.suffix!='.safetensors': raise ValueError('only Safetensors weights are accepted')
        with safe_open(str(file),framework='pt',device='cpu') as f:
            for name in f.keys():
                if name in tensors: raise ValueError('duplicate tensor: '+name)
                if weight_map is not None and weight_map.get(name)!=file.name:
                    raise ValueError('shard index/tensor ownership mismatch')
                shape=f.get_slice(name).get_shape(); dtype=f.get_slice(name).get_dtype()
                if dtype not in ('F32','F16','BF16'): raise ValueError('unsupported checkpoint tensor dtype')
                count=1
                for d in shape: count*=d
                tensors[name]={'shape':shape,'dtype':dtype}; owners[name]=file.name
                if not name.startswith(('model.visual.','mtp.')): bytes32+=count*4
    if weight_map is not None and set(weight_map)!=set(tensors): raise ValueError('missing indexed tensors')
    weights={f.name:digest(f) for f in files}
    tok={}
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                 'generation_config.json','chat_template.jinja','vocab.json','vocab.txt','merges.txt',
                 'added_tokens.json','tokenizer.model'):
        if (root/name).exists(): tok[name]=digest(local_file(root,name))
    if not (root/'tokenizer.json').exists(): raise ValueError('local tokenizer.json required')
    for file in (root/'chat_templates').glob('*.jinja'):
        name=str(file.relative_to(root)); tok[name]=digest(local_file(root,name))
    return root,c,files,tensors,bytes32,weights,tok

def inspect(root,context=2048):
    if type(context) is not int or not 2<=context<=2048 or context%128:
        raise ValueError('context allocation must be a multiple of 128, up to 2048')
    root,c,files,tensors,bytes32,weights,tok=checkpoint(root)
    family=SUPPORTED.get(c.get('model_type')); errors=[]; configs={}
    if c.get('auto_map'): errors.append('remote model code is disabled')
    if family is None: errors.append('unsupported architecture; add an operator/compiler adapter')
    if family=='qwen3' and (c.get('attention_bias',False) or c.get('use_sliding_window',False)
                          or c.get('rope_scaling') not in (None,{})):
        errors.append('Qwen3 bias/window/scaled-RoPE setting not implemented')
    if family and not errors:
        try:
            from opentpu.llm import load_spec
            spec=load_spec(root)
            for name,p in PERSONALITIES.items():
                try:
                    cfg=p.config(spec,context); spec.check(cfg)
                    image=spec.image(cfg,context,1,1,'int8','int8')
                    from opentpu.isa import assemble
                    programs=image.compile_step(context-1)
                    instruction_words=max(len(assemble(p)) for p in programs)
                    if instruction_words>cfg.IMEM_WORDS: raise ValueError('compiled program exceeds instruction capacity')
                    configs[name]={'legal':True,'config':cfg.__dict__,
                                   'image_bytes':image.nbytes,'simulated_memory_bytes':cfg.DRAM_BYTES,
                                   'memory_requirements':{'weights_scales_norms_padding':image.nbytes-image.kv_bytes-image.layer0,
                                       'kv_or_recurrent_state':image.kv_bytes,'io_activation_logits':image.layer0,
                                       'instruction_bytes':instruction_words*4,'tmem_scratch_bytes':cfg.TMEM_WORDS*4,
                                       'quantized_activation_ram_bytes':cfg.ACT_BLOCKS*cfg.D},
                                   'estimated_peak_host_bytes':bytes32+6*cfg.DRAM_BYTES}
                except (ValueError,AssertionError,KeyError) as error:
                    configs[name]={'legal':False,'reason':str(error)}
        except (ValueError,KeyError,AssertionError) as error: errors.append(str(error))
    return {'schema_version':1,'path':str(root),'family':family,'model_type':c.get('model_type'),
            'base_model_id':identity({'config':c,'weight_files':weights}),
            'tokenizer_id':identity(tok),'weight_files':weights,'tokenizer_files':tok,
            'tensor_count':len(tensors),'fp32_tensor_bytes':bytes32,'config':c,
            'personalities':configs,'supported':bool(configs) and any(v['legal'] for v in configs.values()) and not errors,
            'errors':errors,'numeric_contract':CONTRACT,'upstream_revision':upstream.REVISION,
            'physical_fit':'unavailable','context_limit':context,
            'operations':{'qwen3':['matrix','rmsnorm','rope','softmax','swiglu','kv-cache'],
                          'qwen35':['matrix','rmsnorm','rope','softmax','swiglu','gated-deltanet','convolution','kv-cache'],
                          'lfm2':['matrix','rmsnorm','rope','softmax','swiglu','short-convolution','kv-cache']}.get(family,[]),
            'unsupported_features':['multimodal inputs','arbitrary remote architectures','GGUF','sampling','training']}

def load(root):
    import numpy as np
    import torch
    from safetensors import safe_open
    root,c,files,tensors,_,_,_=checkpoint(root); out={}
    for file in files:
        with safe_open(str(file),framework='pt',device='cpu') as f:
            for name in f.keys():
                if name.startswith(('model.visual.','mtp.')): continue
                key=name.replace('model.language_model.','model.',1)
                if key in out: raise ValueError('ambiguous normalized tensor name')
                value=f.get_tensor(name).to(torch.float32).numpy()
                if not np.isfinite(value).all(): raise ValueError('nonfinite tensor: '+name)
                out[key]=value
    return out
