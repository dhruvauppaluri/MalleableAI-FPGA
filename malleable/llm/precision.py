"""Validation-only quantization attribution; never an executable release policy."""
import ast
import gc
import hashlib
import inspect as source_inspect
import math
from pathlib import Path
import time

from ..records import identity
from .diagnostics import comparison
from .quality import frozen_suite, floating_reference
from .models import inspect, load, digest

GROUPS=('transformer_weights','projection_inputs','key_cache','value_cache',
        'attention','head_weights','head_inputs')
_SITES=(
    ('projection_inputs','attention_input','norm(x, W[p + "input_layernorm.weight"])'),
    ('key_cache','key_store','k'),
    ('value_cache','value_store','v'),
    ('attention','query','q[hq] / math.sqrt(d)'),
    ('attention','probabilities','ppad'),
    ('projection_inputs','attention_output','o.reshape(-1)'),
    ('projection_inputs','mlp_input','norm(x, W[p + "post_attention_layernorm.weight"])'),
    ('projection_inputs','mlp_down','(g / (1 + np.exp(-g))) * u'),
    ('head_inputs','head_input','norm(x, W["model.norm.weight"])'))


def validate_policy(policy):
    if not isinstance(policy,dict): raise ValueError('diagnostic precision policy must be a mapping')
    version=policy.get('schema_version')
    keys={'schema_version','quantized_groups','group_size'}
    if version==2: keys.add('site_overrides')
    if set(policy)!=keys:
        raise ValueError('diagnostic precision policy requires version, quantized_groups and group_size')
    groups=policy['quantized_groups']
    if (type(version) is not int or version not in (1,2)
        or not isinstance(groups,list) or any(not isinstance(g,str) or g not in GROUPS for g in groups)
        or len(set(groups))!=len(groups) or type(policy['group_size']) is not int
        or policy['group_size'] not in (32,64,128)):
        raise ValueError('unsupported diagnostic precision policy')
    result={'schema_version':version,'quantized_groups':[g for g in GROUPS if g in groups],
            'group_size':policy['group_size']}
    if version==2:
        overrides=policy['site_overrides']
        if not isinstance(overrides,dict) or not overrides or set(overrides)-{'key_store','query'}:
            raise ValueError('unsupported diagnostic precision sites')
        for site,config in overrides.items():
            if (not isinstance(config,dict) or set(config)!={'format','block'}
                or config['format'] not in ('float32','float16','int16','int8')
                or type(config['block']) is not int or config['block'] not in (16,32,64,128)
                or ('key_cache' if site=='key_store' else 'attention') not in groups):
                raise ValueError('unsupported site precision override')
        result['site_overrides']={k:dict(overrides[k]) for k in sorted(overrides)}
    return result


def site_quantize(value,config):
    """Independent diagnostic rounding only; this does not implement ISA storage."""
    import numpy as np
    value=np.asarray(value,np.float64); fmt=config['format']; block=config['block']
    if value.shape[-1]%block: raise ValueError('precision block must divide the last axis')
    if fmt in ('float32','float16'):
        result=value.astype(np.float32 if fmt=='float32' else np.float16).astype(np.float64)
    else:
        maximum=127 if fmt=='int8' else 32767
        shaped=value.reshape(*value.shape[:-1],value.shape[-1]//block,block)
        scale=np.maximum(np.abs(shaped).max(-1,keepdims=True)/maximum,np.finfo(np.float64).tiny)
        result=(np.clip(np.rint(shaped/scale),-maximum,maximum)*scale).reshape(value.shape)
    if not np.isfinite(result).all(): raise ValueError('nonfinite site precision conversion')
    return result


def make_panel(suite,limit_targets=16,selection='prefix',context=128):
    if type(limit_targets) is not int or not 1<=limit_targets<=128:
        raise ValueError('diagnostic target count must be 1..128')
    if selection not in ('prefix','spread') or context!=128:
        raise ValueError('diagnostic selection must be prefix/spread at context 128')
    data,hashes=frozen_suite(suite)
    points=[(s,p) for s,row in enumerate(data['validation']) for p in range(len(row)-1)]
    if len(points)<limit_targets: raise ValueError('insufficient validation targets')
    chosen=(points[:limit_targets] if selection=='prefix' else
        [points[i*(len(points)-1)//max(1,limit_targets-1)] for i in range(limit_targets)])
    selected={}
    for sequence,position in chosen: selected.setdefault(sequence,[]).append(position)
    rows=[]
    for sequence,positions in selected.items():
        end=max(positions)+1
        if end>context: raise ValueError('full causal prefix exceeds diagnostic context')
        rows.append({'sequence':sequence,'positions':positions,'tokens':data['validation'][sequence][:end+1]})
    return {'schema_version':1,'split':'validation','selection':selection,'context':context,
        'suite_file_hash':digest(suite),'split_hash':hashes['validation'],
        'base_model_id':data['base_model_id'],'tokenizer_id':data['tokenizer_id'],
        'target_count':limit_targets,'executed_tokens':sum(len(r['tokens'])-1 for r in rows),
        'rows':rows,'token_hash':identity(rows)}


class OperatorStats:
    """Aggregate bounded operator statistics without retaining tensor snapshots."""
    def __init__(self): self.rows={}

    def observe(self,key,original,result,quantized,block):
        import numpy as np
        row=self.rows.setdefault(key,dict(group=key[0],operator=key[1],layer=key[2],
            calls=0,elements=0,reference_squared=0.,error_squared=0.,max_absolute_error=0.,
            max_absolute_value=0.,zeroed_elements=0,quantized=quantized,block=block,
            clipped_elements=0,clipping_rule='none: max-absolute INT8 scaling or floating bypass'))
        row['calls']+=1
        a=np.asarray(original).reshape(-1); b=np.asarray(result).reshape(-1)
        # Vocabulary matrices are large; temporary arrays stay bounded to 1M elements.
        for start in range(0,len(a),1<<20):
            x=a[start:start+(1<<20)].astype(np.float64,copy=False)
            y=b[start:start+(1<<20)]; error=y-x
            row['elements']+=len(x)
            row['reference_squared']+=float(np.dot(x,x)); row['error_squared']+=float(np.dot(error,error))
            row['max_absolute_error']=max(row['max_absolute_error'],float(np.abs(error).max(initial=0)))
            row['max_absolute_value']=max(row['max_absolute_value'],float(np.abs(x).max(initial=0)))
            row['zeroed_elements']+=int(np.count_nonzero((x!=0)&(y==0)))

    def export(self):
        result=[]
        for row in self.rows.values():
            row=dict(row)
            row['relative_l2_error']=math.sqrt(row['error_squared']/max(row['reference_squared'],1e-30))
            rms=math.sqrt(row['reference_squared']/max(1,row['elements']))
            row['max_abs_over_rms']=row['max_absolute_value']/max(rms,1e-30)
            result.append(row)
        return result


def controlled_reference(policy,stats=None):
    """Instrument the pinned independent emulation, asserting every quantization site."""
    import numpy as np
    from opentpu.llm.qwen3 import emulated_logits,_fake_q,_fake_w
    policy=validate_policy(policy); enabled=set(policy['quantized_groups'])
    source=source_inspect.getsource(emulated_logits); tree=ast.parse(source)
    class Rewrite(ast.NodeTransformer):
        def __init__(self): self.q=0; self.w=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if not isinstance(node.func,ast.Name): return node
            if node.func.id=='_fake_q':
                if self.q>=len(_SITES) or len(node.args)!=2: raise ValueError('quantization sites changed upstream')
                group,site,expression=_SITES[self.q]; self.q+=1
                if ast.dump(node.args[0])!=ast.dump(ast.parse(expression,mode='eval').body):
                    raise ValueError('quantization site expression changed upstream: '+site)
                layer=ast.Constant(None) if group=='head_inputs' else ast.Name(id='i',ctx=ast.Load())
                return ast.copy_location(ast.Call(func=ast.Name(id='_precision_q',ctx=ast.Load()),
                    args=[ast.Constant(group),ast.Constant(site),layer,*node.args],keywords=[]),node)
            if node.func.id=='_fake_w':
                self.w+=1
                if self.w!=1 or len(node.args)!=3: raise ValueError('weight quantization changed upstream')
                return ast.copy_location(ast.Call(func=ast.Name(id='_precision_w',ctx=ast.Load()),
                    args=[ast.Name(id='n',ctx=ast.Load()),*node.args,
                          ast.Compare(left=ast.Name(id='n',ctx=ast.Load()),ops=[ast.Eq()],
                                      comparators=[ast.Name(id='head',ctx=ast.Load())])],keywords=[]),node)
            return node
    rewrite=Rewrite(); tree=rewrite.visit(tree)
    if rewrite.q!=len(_SITES) or rewrite.w!=1: raise ValueError('incomplete upstream quantization instrumentation')
    ast.fix_missing_locations(tree)
    def quantize(group,site,layer,value,block):
        config=policy.get('site_overrides',{}).get(site)
        if config is not None:
            output=site_quantize(value,config); block=config['block']
        else:
            output=_fake_q(value,block) if group in enabled else np.asarray(value,np.float64)
        if stats is not None: stats.observe((group,site,layer),value,output,group in enabled,block)
        return output
    def weight(name,value,block,fmt,is_head):
        if fmt!='int8': raise ValueError('precision attribution baseline requires INT8 weights')
        group='head_weights' if is_head else 'transformer_weights'
        output=_fake_w(value,block,fmt) if group in enabled else np.asarray(value,np.float64)
        layer=None if is_head else int(name.split('.')[2])
        if stats is not None: stats.observe((group,name,layer),value,output,group in enabled,block)
        return output
    namespace=dict(emulated_logits.__globals__,_precision_q=quantize,_precision_w=weight)
    exec(compile(tree,'<controlled-independent-emulation>','exec'),namespace)
    return namespace['emulated_logits'],hashlib.sha256(source.encode()).hexdigest()


def token_summary(logits,target):
    import numpy as np
    x=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('nonfinite diagnostic logits')
    first=int(np.argmax(x)); masked=x.copy(); masked[first]=-np.inf
    second=int(np.argmax(masked)); peak=float(x[first])
    return {'top1':first,'runner_up':second,'margin':float(x[first]-x[second]),
        'target_nll':float(peak+np.log(np.exp(x-peak).sum())-x[target])}


def diagnose_precision(model,suite,policy,panel,reference_root,max_host_gib=16,emit=lambda *_:None,
                       candidate_weights=None):
    """Optional derived tensors override the candidate only, never the original reference."""
    import numpy as np
    import torch
    import transformers
    from ..store import Store
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import reference_logits
    policy=validate_policy(policy)
    expected=make_panel(suite,panel.get('target_count'),panel.get('selection'),panel.get('context'))
    if panel!=expected: raise ValueError('frozen validation diagnostic panel mismatch')
    info=inspect(model,128)
    if info['family']!='qwen3' or not info['supported']: raise ValueError('precision attribution currently supports Qwen3')
    if any(info[k]!=panel[k] for k in ('base_model_id','tokenizer_id')): raise ValueError('diagnostic model/tokenizer mismatch')
    derived_bytes=sum(v.nbytes for v in (candidate_weights or {}).values())
    if 4*info['fp32_tensor_bytes']+derived_bytes>max_host_gib*1024**3:
        raise ValueError('precision attribution exceeds configured host memory budget')
    spec=load_spec(Path(model)); weights=load(model)
    if any(t>=spec.vocab for row in panel['rows'] for t in row['tokens']): raise ValueError('diagnostic token outside vocabulary')
    provenance={'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
        'weight_files':info['weight_files'],'panel_id':identity(panel),'torch':torch.__version__,
        'transformers':transformers.__version__,'dtype':'float32','attention':'eager',
        'contract':'original-local-safetensors-teacher-forced-v1'}
    ref_id=identity(provenance); store=Store(reference_root); started=time.monotonic()
    try:
        cached=next((r for r in store.records('llm-precision-reference') if r['reference_id']==ref_id),None)
        if cached is None:
            original=floating_reference(model,info,weights); refs=[]; parity=[]
            with torch.no_grad():
                for row in panel['rows']:
                    tokens=row['tokens']; hf=original(torch.tensor([tokens[:-1]])).logits[0].float().numpy()
                    fp=reference_logits(spec,weights,tokens[:-1])
                    positions=row['positions']; metrics=comparison(fp[positions],hf[positions])
                    matches=sum(int(np.argmax(fp[p]))==int(np.argmax(hf[p])) for p in positions)
                    parity.append(dict(metrics,sequence=row['sequence'],top1_matches=matches,targets=len(positions)))
                    if matches!=len(positions): raise ValueError('independent FP32 top-token mismatch; investigate before attributing quantization')
                    refs.extend({'sequence':row['sequence'],'position':p,'target_token':tokens[p+1],
                                 'floating':token_summary(hf[p],tokens[p+1])} for p in positions)
                    emit('reference-progress',{'sequence':row['sequence'],'targets':len(refs)})
            del original,hf,fp; gc.collect()
            cached={'schema_version':1,'reference_id':ref_id,'provenance':provenance,'tokens':refs,'parity':parity}
            store.save('llm-precision-reference',cached)
        reference_record=identity(cached)
    finally: store.close()
    lookup={(r['sequence'],r['position']):r for r in cached['tokens']}
    if candidate_weights is not None:
        for name,value in candidate_weights.items():
            if (name not in weights or value.shape!=weights[name].shape
                or value.dtype!=np.float32 or not np.isfinite(value).all()):
                raise ValueError('invalid derived candidate tensor: '+name)
        weights={**weights,**candidate_weights}
    stats=OperatorStats(); forward,reference_source=controlled_reference(policy,stats); tokens=[]
    candidate_started=time.monotonic()
    for row in panel['rows']:
        logits=forward(spec,weights,row['tokens'][:-1],D=policy['group_size'],wformat='int8',head_format='int8')
        for position in row['positions']:
            reference=lookup[row['sequence'],position]
            candidate=token_summary(logits[position],row['tokens'][position+1])
            tokens.append(dict(reference,candidate=candidate,agrees=candidate['top1']==reference['floating']['top1']))
        emit('precision-progress',{'completed_targets':len(tokens),'total_targets':panel['target_count'],
            'sequence':row['sequence'],'executed_tokens':len(row['tokens'])-1})
        del logits; gc.collect()
    count=len(tokens); fnll=sum(t['floating']['target_nll'] for t in tokens)/count
    cnll=sum(t['candidate']['target_nll'] for t in tokens)/count
    operators=stats.export()
    for row in operators:
        row['precision_format']=policy.get('site_overrides',{}).get(row['operator'],{}).get(
            'format','int8' if row['quantized'] else 'float64-bypass')
    return {'schema_version':1,'kind':'precision-diagnostic','release_evidence':False,'selectable':False,
        'split':'validation','context':128,'baseline_personality':'balanced','precision_policy':policy,
        'precision_policy_id':identity(policy),'panel_id':identity(panel),'panel':panel,
        'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],'target_count':count,
        'executed_tokens':panel['executed_tokens'],'agreement':sum(t['agrees'] for t in tokens)/count,
        'float_nll':fnll,'candidate_nll':cnll,'nll_degradation_percent':100*(cnll-fnll)/max(fnll,1e-12),
        'tokens':tokens,'operators':operators,'reference_parity':cached['parity'],
        'floating_reference_record':reference_record,'reference_provenance':provenance,
        'upstream_emulation_source_sha256':reference_source,'upstream_revision':info['upstream_revision'],
        'candidate_seconds':time.monotonic()-candidate_started,'elapsed_seconds':time.monotonic()-started,
        'scope':'independent float64 quantization emulation against original FP32; no ISA/RTL implementation or approval'}
