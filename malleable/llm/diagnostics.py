"""Bounded validation diagnostics, distinct from release quality evidence."""
import ast
import gc
import inspect as python_inspect
import time
from pathlib import Path

from .models import inspect, load, digest
from .quality import frozen_suite, floating_reference
from .records import PERSONALITIES
from ..records import identity


def _captured_reference(function, captured):
    """Observe residuals in the independent emulation without changing its math."""
    tree = ast.parse(python_inspect.getsource(function))
    class Capture(ast.NodeTransformer):
        def __init__(self): self.stage = 0
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'x' and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Add)):
                if self.stage>=2: raise ValueError('independent reference residual instrumentation changed')
                stage = ('attention_residual', 'mlp_residual')[self.stage]
                self.stage += 1
                extra = ast.parse(f"_capture('{stage}', i, locals().get('pos'), x)").body[0]
                return [node, ast.copy_location(extra, node)]
            return node
    capture = Capture()
    tree = capture.visit(tree)
    if capture.stage != 2:
        raise ValueError('independent reference residual instrumentation changed')
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    namespace['_capture'] = lambda stage, layer, pos, value: captured.__setitem__((stage, layer, pos), value.copy())
    exec(compile(tree, '<quality-diagnostic-reference>', 'exec'), namespace)
    return namespace[function.__name__]


def comparison(actual, reference):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return {'max_absolute_error': float(np.max(np.abs(a-b))),
            'relative_l2_error': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)),
            'cosine': float(np.dot(a.ravel(), b.ravel())/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))}


def _captured_qwen35(function,captured):
    """Observe bounded samples at existing Qwen3.5 operators without changing math."""
    import numpy as np
    tree=ast.parse(python_inspect.getsource(function)); batched=function.__name__=='reference_logits'
    stages={'y':'convolution','S':'deltanet_state','o':'attention_output','q':'query_normalization','k':'key_normalization',
        'beta':'beta_gate','g':'decay_gate','z':'output_gate','gg':'mlp_gate','u':'mlp_up'}
    class Capture(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='_norm' and 'model.norm.weight' in ast.unparse(node):
                return ast.copy_location(ast.Call(func=ast.Name(id='_capture_head',ctx=ast.Load()),
                    args=[node,ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]),
                        attr='get',ctx=ast.Load()),args=[ast.Constant('pos')],keywords=[])],keywords=[]),node)
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):return node
            name=node.targets[0].id;stage=stages.get(name)
            if name=='h':
                text=ast.unparse(node.value)
                if 'input_layernorm' in text:stage='input_normalization'
                elif 'post_attention_layernorm' in text:stage='mlp_normalization'
            if name=='x' and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Add):
                stage='mlp_residual' if 'mlp.down_proj' in ast.unparse(node.value) else 'attention_residual'
            if name=='x' and 'model.norm.weight' in ast.unparse(node.value):stage='head_normalization'
            if stage is None:return node
            # Initial zeros are not a completed recurrence/attention operator.
            if 'np.zeros' in ast.unparse(node.value):return node
            position="locals().get('t')" if batched and name=='S' else "locals().get('pos')"
            extra=ast.parse(f"_capture('{stage}', locals().get('i'), {position}, {name})").body[0]
            return [node,ast.copy_location(extra,node)]
    tree=Capture().visit(tree);ast.fix_missing_locations(tree)
    def capture(stage,layer,position,value):
        array=np.asarray(value)
        if batched and position is None:
            for index,row in enumerate(array):
                captured[stage,layer,index]=np.asarray(row).reshape(-1)[:4096].copy()
        else:captured[stage,layer,position]=array.reshape(-1)[:4096].copy()
    def capture_head(value,position):
        capture('head_normalization',None,position,value);return value
    namespace=dict(function.__globals__,_capture=capture,_capture_head=capture_head)
    exec(compile(tree,'<qwen35-observed-reference>','exec'),namespace)
    return namespace[function.__name__]


def diagnose_reference_rounding(model,suite,limit_targets=16,max_host_gib=16,emit=lambda *_:None):
    """Validation-only control; ordinary Transformers quality reference stays unchanged."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from unittest.mock import patch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_module
    from .precision import make_panel,token_summary
    from opentpu.llm import load_spec
    from opentpu.llm.qwen35 import reference_logits
    panel=make_panel(suite,limit_targets,'prefix');info=inspect(model,128)
    if info['family']!='qwen35' or not info['supported']:raise ValueError('Qwen3.5 rounding control required')
    if 2*info['fp32_tensor_bytes']>max_host_gib*1024**3:raise ValueError('rounding control exceeds host budget')
    weights=load(model);raw_checked=0;defects=[]
    # Independent file-to-converted-tensor check, not only assigned HF state.
    for file in info['weight_files']:
        with safe_open(str(Path(model)/file),framework='pt',device='cpu') as tensors:
            for name in tensors.keys():
                if name.startswith(('model.visual.','mtp.')):continue
                key=name.replace('model.language_model.','model.',1)
                value=tensors.get_tensor(name).float().numpy();raw_checked+=1
                if key not in weights or value.shape!=weights[key].shape or not np.array_equal(value,weights[key]):defects.append(name)
    if raw_checked!=len(weights):raise ValueError('conversion tensor count mismatch')
    original=floating_reference(model,info,weights);spec=load_spec(Path(model));rows=[];started=time.monotonic()
    with torch.no_grad():
        for row in panel['rows']:
            tokens=row['tokens'];inputs=torch.tensor([tokens[:-1]])
            chunked=original(inputs).logits[0].float().numpy()
            with patch.object(hf_module,'torch_chunk_gated_delta_rule',hf_module.torch_recurrent_gated_delta_rule):
                sequential=original(inputs).logits[0].float().numpy()
            independent=reference_logits(spec,weights,tokens[:-1]);positions=row['positions']
            rows.append({'sequence':row['sequence'],'positions':positions,
                'independent_vs_chunked':comparison(independent[positions],chunked[positions]),
                'independent_vs_sequential':comparison(independent[positions],sequential[positions]),
                'sequential_vs_chunked':comparison(sequential[positions],chunked[positions]),
                'tokens':[{'position':p,'target_token':tokens[p+1],
                    **{name:token_summary(values[p],tokens[p+1]) for name,values in
                        [('chunked',chunked),('sequential',sequential),('independent',independent)]}} for p in positions]})
            emit('reference-progress',{'sequence':row['sequence'],'targets':len(positions)})
    return {'schema_version':1,'kind':'qwen35-reference-rounding-diagnostic','split':'validation',
        'release_evidence':False,'selectable':False,'panel':panel,'panel_id':identity(panel),
        'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
        'conversion':{'raw_tensors_checked':raw_checked,'mismatches':defects},'rows':rows,
        'elapsed_seconds':time.monotonic()-started,'reference_unchanged':True,
        'control':'replace only diagnostic HF chunked DeltaNet with its existing sequential FP32 helper',
        'helper_source_sha256':identity({'chunked':python_inspect.getsource(hf_module.torch_chunk_gated_delta_rule),
            'sequential':python_inspect.getsource(hf_module.torch_recurrent_gated_delta_rule)})}


def diagnose(model, suite, limit_targets=16, personality='balanced', wformat='int8',
             context=128, split='validation', max_host_gib=16, emit=lambda *_: None,candidate_case=None):
    if split != 'validation': raise ValueError('diagnostics are validation-only; held-out is never a search input')
    if type(limit_targets) is not int or not 1 <= limit_targets <= 128:
        raise ValueError('diagnostic limit must be 1..128 targets')
    import numpy as np
    import torch
    import transformers
    import platform
    data, hashes = frozen_suite(suite)
    info = inspect(model, context)
    derived=None
    if candidate_case is not None:
        if context!=128 or personality!='balanced' or wformat!='int8':
            raise ValueError('derived candidate diagnostic requires context128/balanced/INT8')
        from .candidates import read_derived_candidate
        derived=read_derived_candidate(candidate_case,info)
    if info['family'] not in ('qwen3','qwen35') or not info['supported']:
        raise ValueError('unsupported localized independent-reference diagnostic family')
    if data['base_model_id'] != info['base_model_id'] or data['tokenizer_id'] != info['tokenizer_id']:
        raise ValueError('diagnostic suite model/tokenizer lineage mismatch')
    if wformat not in ('int8','int4','fp4'): raise ValueError('unsupported diagnostic format')
    from opentpu.llm.qwen3 import Engine, reference_logits, emulated_logits
    capture=_captured_reference
    if info['family']=='qwen35':
        from opentpu.llm.qwen35 import reference_logits,emulated_logits
        capture=_captured_qwen35
    from opentpu.llm import load_spec
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    spec = load_spec(Path(model))
    cfg = PERSONALITIES[personality].config(spec, context, wformat)
    # FP32 weights, the float64 quantized emulation, the ISA image and scratch.
    estimate=3 * info['fp32_tensor_bytes'] + 3 * cfg.DRAM_BYTES
    if derived: estimate+=derived['bytes']
    if estimate > max_host_gib * 1024**3:
        raise ValueError('diagnostic exceeds configured host memory budget')
    if Path('/proc/meminfo').is_file():
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:'))
        if estimate>available:raise ValueError('diagnostic exceeds currently available host memory')
    weights = load(model)
    original = floating_reference(model, info, weights)
    loaded = original.state_dict()
    defects = []
    for name, value in weights.items():
        tensor = loaded.get(name)
        if tensor is None or tuple(tensor.shape) != value.shape or not np.array_equal(tensor.detach().float().numpy(), value):
            defects.append(name)
    del loaded
    # Freeze a prefix of validation only, including each sequence's causal reset.
    rows, remaining = [], limit_targets
    for row in data['validation']:
        take = min(len(row)-1, remaining)
        if take > context or any(t >= spec.vocab for t in row[:take+1]):
            raise ValueError('diagnostic tokens exceed context or vocabulary')
        rows.append(row[:take+1]); remaining -= take
        if not remaining: break
    hf_rows = []
    started = time.monotonic()
    with torch.no_grad():
        for row in rows:
            hf_rows.append(original(torch.tensor([row[:-1]])).logits[0].float().numpy())
    del original
    gc.collect()
    original_rows=None
    if derived:
        original_rows=[]
        for row in rows:
            floats={}
            fp=capture(reference_logits,floats)(spec,weights,row[:-1])
            original_rows.append((fp,floats))
        from .candidates import apply_derived_candidate
        weights=apply_derived_candidate(weights,derived)
    emit('phase', {'phase':'independent-reference-diagnostics','targets':limit_targets})
    token_results, layer_results, parity = [], [], []
    for row_index, row in enumerate(rows):
        floats, quantized = {}, {}
        if original_rows is None:
            fp = capture(reference_logits, floats)(spec, weights, row[:-1])
        else:
            fp,floats=original_rows[row_index]
        emu = capture(emulated_logits, quantized)(spec, weights, row[:-1], wformat=wformat, head_format='int8')
        hf = hf_rows[row_index]
        parity.append(comparison(fp, hf))
        engine = Engine(spec, weights, cap=context, cfg=cfg, rows=1, pipeline=False,
                        backend='isa', wformat=wformat, head_format='int8')
        for position, token in enumerate(row[:-1]):
            actual = engine.step(token)[:spec.vocab]
            if not all(np.isfinite(v).all() for v in (actual, fp[position], emu[position], hf[position])):
                raise ValueError('nonfinite diagnostic output')
            target = row[position+1]
            def summary(logits):
                x = logits.astype(np.float64); peak = x.max()
                best = np.argpartition(x, -2)[-2:]; best = best[np.argsort(x[best])[::-1]]
                return {'top1':int(best[0]),'margin':float(x[best[0]]-x[best[1]]),
                        'target_nll':float(peak+np.log(np.exp(x-peak).sum())-x[target])}
            values = {'sequence':row_index,'position':position,'input_token':token,'target_token':target,
                      'floating':summary(hf[position]), 'independent_float':summary(fp[position]),
                      'quantized_emulation':summary(emu[position]), 'isa':summary(actual),
                      'isa_vs_emulation':comparison(actual,emu[position]),
                      'isa_vs_floating':comparison(actual,hf[position])}
            token_results.append(values)
            for (stage, layer, pos), value in quantized.items():
                reference_value=(floats.get((stage,layer,pos)) if info['family']=='qwen35'
                    else floats[(stage,layer,None)][position])
                if pos == position and reference_value is not None and value.shape==reference_value.shape:
                    layer_results.append({'sequence':row_index,'position':position,'layer':layer,'stage':stage,
                        **comparison(value,reference_value),
                        'sampling':'first 4096 flattened elements' if info['family']=='qwen35' else 'complete residual',
                        'scope':'independent quantized emulation versus FP32 reference; not RTL state'})
            emit('quality-progress',{'phase':'diagnostic','completed_targets':len(token_results),'total_targets':limit_targets})
        del engine
        gc.collect()
    count = len(token_results)
    def agree(left, right): return sum(r[left]['top1']==r[right]['top1'] for r in token_results)/count
    candidate_record=None
    if derived:
        candidate_record={k:derived[k] for k in ('schema_version','derived_id','parameters_id',
            'calibration_statistics_id','candidate','screen_result_sha256')}
    return {'schema_version':1,'kind':'quality-diagnostic','split':'validation','release_evidence':False,
            'selectable':False,'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
            'suite_file_hash':digest(suite),'split_hash':hashes['validation'],'diagnostic_token_hash':identity(rows),
            'target_count':count,'context':context,'personality':personality,'wformat':wformat,
            'configuration_id':identity({'config':cfg.__dict__,'uarch':PERSONALITIES[personality].uarch}),
            'conversion':{'checked_tensors':len(weights),'mismatches':defects,
                'tokenizer_vocab_size':len(tokenizer),'model_vocab_size':spec.vocab},
            'reference_parity':parity,'agreement':agree('isa','floating'),
            'float_reference_agreement':agree('independent_float','floating'),
            'isa_emulation_agreement':agree('isa','quantized_emulation'),
            'emulation_float_agreement':agree('quantized_emulation','floating'),
            'tokens':token_results,'layer_residuals':layer_results,'elapsed_seconds':time.monotonic()-started,
            'derived_candidate':candidate_record,
            'toolchain':{'torch':torch.__version__,'transformers':transformers.__version__},
            'checkpoint':{'weight_files':info['weight_files'],'tokenizer_files':info['tokenizer_files']},
            'numeric_contract':info['numeric_contract'],'upstream_revision':info['upstream_revision'],
            'python':platform.python_version(),
            'provenance':'bounded validation-only reference/conversion/quantization diagnosis'}
