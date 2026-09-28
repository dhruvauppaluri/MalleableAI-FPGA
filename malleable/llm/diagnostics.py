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


def diagnose(model, suite, limit_targets=16, personality='balanced', wformat='int8',
             context=128, split='validation', max_host_gib=16, emit=lambda *_: None):
    if split != 'validation': raise ValueError('diagnostics are validation-only; held-out is never a search input')
    if type(limit_targets) is not int or not 1 <= limit_targets <= 128:
        raise ValueError('diagnostic limit must be 1..128 targets')
    import numpy as np
    import torch
    import transformers
    import platform
    data, hashes = frozen_suite(suite)
    info = inspect(model, context)
    if info['family'] != 'qwen3' or not info['supported']:
        raise ValueError('localized independent-reference diagnostics currently support Qwen3')
    if data['base_model_id'] != info['base_model_id'] or data['tokenizer_id'] != info['tokenizer_id']:
        raise ValueError('diagnostic suite model/tokenizer lineage mismatch')
    if wformat not in ('int8','int4','fp4'): raise ValueError('unsupported diagnostic format')
    from opentpu.llm.qwen3 import Engine, reference_logits, emulated_logits
    from opentpu.llm import load_spec
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    spec = load_spec(Path(model))
    cfg = PERSONALITIES[personality].config(spec, context, wformat)
    # FP32 weights, the float64 quantized emulation, the ISA image and scratch.
    if 3 * info['fp32_tensor_bytes'] + 3 * cfg.DRAM_BYTES > max_host_gib * 1024**3:
        raise ValueError('diagnostic exceeds configured host memory budget')
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
    emit('phase', {'phase':'independent-reference-diagnostics','targets':limit_targets})
    token_results, layer_results, parity = [], [], []
    for row_index, row in enumerate(rows):
        floats, quantized = {}, {}
        fp = _captured_reference(reference_logits, floats)(spec, weights, row[:-1])
        emu = _captured_reference(emulated_logits, quantized)(spec, weights, row[:-1], wformat=wformat, head_format='int8')
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
                if pos == position:
                    layer_results.append({'sequence':row_index,'position':position,'layer':layer,'stage':stage,
                        **comparison(value, floats[(stage,layer,None)][position]),
                        'scope':'independent quantized emulation versus FP32 reference; not RTL state'})
            emit('quality-progress',{'phase':'diagnostic','completed_targets':len(token_results),'total_targets':limit_targets})
        del engine
        gc.collect()
    count = len(token_results)
    def agree(left, right): return sum(r[left]['top1']==r[right]['top1'] for r in token_results)/count
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
            'toolchain':{'torch':torch.__version__,'transformers':transformers.__version__},
            'checkpoint':{'weight_files':info['weight_files'],'tokenizer_files':info['tokenizer_files']},
            'numeric_contract':info['numeric_contract'],'upstream_revision':info['upstream_revision'],
            'python':platform.python_version(),
            'provenance':'bounded validation-only reference/conversion/quantization diagnosis'}
