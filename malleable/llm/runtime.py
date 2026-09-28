"""Safe local generation; every prompt token uses the explicitly selected backend."""
from dataclasses import asdict
import platform
from pathlib import Path
import time
import shutil
from . import upstream
from .models import inspect, load, local_file
from .records import PERSONALITIES
from ..records import identity
from ..store import Store


def prompt_tokens(tokenizer, workload):
    """Transformers 5 multimodal templates may return a BatchEncoding."""
    if workload.input_tokens is not None: return list(workload.input_tokens)
    if workload.messages is not None and not tokenizer.chat_template: raise ValueError('conversation requires a validated local chat template')
    if workload.prompt_format=='chat' and tokenizer.chat_template:
        tokens=tokenizer.apply_chat_template(workload.messages or [{'role':'user','content':workload.prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(tokens,'keys'): tokens=tokens['input_ids']
    else: tokens=tokenizer.encode(workload.prompt)
    if not isinstance(tokens,(list,tuple)) or any(type(t) is not int for t in tokens):
        raise ValueError('tokenizer must return one integer token sequence')
    return list(tokens)


def generate(model, workload, root, emit=lambda *_: None):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine
    from .backend import CheckedRtlBackend
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    emit('phase',{'phase':'inspection'})
    info=inspect(model,workload.context)
    if not info['supported']: raise ValueError('; '.join(info['errors']) or 'no legal personality')
    spec=load_spec(Path(model)); cfg=PERSONALITIES[workload.personality].config(spec,workload.context,workload.wformat)
    spec.check(cfg)
    estimate=info['fp32_tensor_bytes']+6*cfg.DRAM_BYTES
    if estimate > workload.max_host_gib*1024**3: raise ValueError('estimated host memory exceeds configured budget')
    # All referenced tokenizer files must remain within the checkpoint root.
    for name in info['tokenizer_files']: local_file(Path(model).resolve(),name)
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    tokens=prompt_tokens(tok,workload)
    if not tokens or len(tokens)+(workload.max_new if workload.input_tokens is None else 0) > workload.context: raise ValueError('prompt plus generation exceeds context')
    if any(t<0 or t>=spec.vocab for t in tokens): raise ValueError('tokenizer IDs outside model vocabulary')
    if workload.backend=='rtl':
        # Conservative trace plus temporary-image budget; not physical FPGA fit.
        disk_estimate=2*cfg.DRAM_BYTES+(len(tokens)+workload.max_new)*max(1024**2,cfg.DRAM_BYTES//10)
        if shutil.disk_usage(root).free<disk_estimate+4*1024**3:
            raise ValueError('insufficient free disk for estimated traces, verification dumps and 4 GiB reserve')
    emit('model',info); emit('phase',{'phase':'loading','prompt_tokens':len(tokens)})
    weights=load(model)
    factory='isa' if workload.backend=='isa' else lambda c,i: CheckedRtlBackend(c,i,workload,root/'traces',emit)
    start=time.monotonic()
    engine=Engine(spec,weights,cap=workload.context,cfg=cfg,backend=factory,
                  rows=1,pipeline=False,wformat=workload.wformat,head_format='int8')
    loading=time.monotonic()-start
    start=time.monotonic()
    emit('phase',{'phase':'prefill','backend':workload.backend})
    for index,token in enumerate(tokens):
        logits=engine.step(token)
        emit('prefill',{'completed':index+1,'total':len(tokens)})
    prefill=time.monotonic()-start; start=time.monotonic(); generated=[]
    emit('phase',{'phase':'decode','backend':workload.backend})
    eos_config=info['config'].get('text_config',info['config']).get('eos_token_id',tok.eos_token_id)
    eos=set(eos_config if isinstance(eos_config,list) else [eos_config])
    if tok.eos_token_id is not None: eos.add(tok.eos_token_id)
    for index in range(workload.max_new if workload.input_tokens is None else 0):
        if not np.isfinite(logits[:spec.vocab]).all(): raise ValueError('non-finite generation logits')
        token=int(np.argmax(logits[:spec.vocab])); generated.append(token)
        emit('token',{'index':index,'token_id':token,'text':tok.decode(generated,skip_special_tokens=True),
                      'state':'committed'})
        if token in eos or index+1==workload.max_new: break
        logits=engine.step(token)
    decode=time.monotonic()-start
    cycles=sum(s.get('cycles',0) for s in engine.stats) if workload.backend=='rtl' else None
    bytes_read=sum(s.get('axi_read_bytes',0) for s in engine.stats) if cycles is not None else None
    useful=sum(s.get('useful_macs',0) for s in engine.stats) if cycles is not None else None
    result=dict(schema_version=1,base_model_id=info['base_model_id'],tokenizer_id=info['tokenizer_id'],
        variant_id=identity({'base':info['base_model_id'],'format':workload.wformat,'head':'int8'}),
        workload_id=workload.workload_id,workload=asdict(workload),config=cfg.__dict__,previous_config=None,
        configuration_id=identity({'config':cfg.__dict__,'uarch':PERSONALITIES[workload.personality].uarch}),
        input_token_hash=identity(tokens),status='completed',
        generation_mode='greedy' if workload.input_tokens is None else 'fixed-token-tape',
        workload_identity_v2=identity({'input':tokens,'seed':workload.seed,'context':workload.context,
            'max_new':workload.max_new if workload.input_tokens is None else 0,'mode':'greedy' if workload.input_tokens is None else 'fixed-token-tape',
            'memory':{'latency':workload.latency,'stalls':workload.stall_percent,'bandwidth':workload.bandwidth_percent}}),
        personality=workload.personality,backend=workload.backend,seed=workload.seed,
        tokens=generated,text=tok.decode(generated,skip_special_tokens=True),prompt_tokens=len(tokens),
        counters=engine.stats,valid=True,quality={'status':'not-evaluated','selectable':False},
        metrics={'host_loading_seconds':{'value':loading,'provenance':'measured-host'},
                 'host_build_lookup_seconds':{'value':getattr(engine.backend,'compilation_seconds',None),
                     'provenance':'measured-host' if workload.backend=='rtl' else 'unavailable'},
                 'host_prefill_seconds':{'value':prefill,'provenance':'measured-host'},
                 'host_decode_seconds':{'value':decode,'provenance':'measured-host'},
                 'host_tokens_per_second':{'value':len(generated)/max(prefill+decode,1e-12),'provenance':'measured-host'},
                 'host_executed_steps_per_second':{'value':len(engine.stats)/max(prefill+decode,1e-12),'provenance':'measured-host'},
                 'rtl_cycles':{'value':cycles,'provenance':'measured-simulation' if cycles is not None else 'unavailable'},
                 'useful_mac_utilization':{'value':useful/max(1,cycles*cfg.D*cfg.MCOLS) if useful is not None else None,
                                          'provenance':'rtl-trace-derived' if useful is not None else 'unavailable'},
                 'axi_read_bytes':{'value':bytes_read,'provenance':'measured-simulation' if bytes_read is not None else 'unavailable'},
                 'axi_read_bytes_per_cycle':{'value':bytes_read/max(1,cycles) if bytes_read is not None else None,
                                            'provenance':'measured-simulation' if bytes_read is not None else 'unavailable'},
                 'projected_seconds':{'value':cycles/workload.clock_hz if cycles is not None and workload.clock_hz else None,
                                       'provenance':'assumed-clock' if cycles is not None and workload.clock_hz else 'unavailable'},
                 'physical_timing':{'value':None,'provenance':'unavailable'},
                 'power':{'value':None,'provenance':'unavailable'}},
        toolchain={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
                   'upstream':upstream.REVISION,'build_id':getattr(engine.backend,'build_id',None)})
    store=Store(root/'research')
    try:
        result['model_record']=store.save('llm-model',info)
        key=store.save('llm-generation',result)
    finally: store.close()
    emit('result',dict(result,record_id=key)); return result
