"""Explicit local CUDA baseline and greedy FPGA-simulation/GPU-verifier run."""
from pathlib import Path
import json
import platform
import shutil
import time

from .hybrid import CudaVerifier, EngineDraft, greedy, tokenizer_compatibility
from .models import inspect, load, digest
from .records import GenerationWorkload, PERSONALITIES
from .runtime import prompt_tokens
from ..records import identity


def _standalone_gate(manifest):
    from .release import check
    if not manifest:
        raise ValueError('CUDA/hybrid is disabled until an explicit standalone release manifest passes')
    return check(manifest)


def _prompt(path, prompt, prompt_format, messages, context):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    workload=GenerationWorkload(prompt,prompt_format=prompt_format,messages=messages,
                                backend='isa',context=context)
    tokens=prompt_tokens(tok,workload)
    if not tokens or len(tokens)>=context:
        raise ValueError('prompt must leave room within the configured context')
    return tok,tokens

def _eos_ids(tokenizer,path):
    value=tokenizer.eos_token_id
    config_path=Path(path)/'generation_config.json'
    if config_path.is_file():
        config=json.loads(config_path.read_text())
        value=config.get('eos_token_id',value)
    return set(value if isinstance(value,list) else [value]) - {None}

def _source_manifest(path):
    manifest=Path(path)/'download-manifest.json'
    if not manifest.is_file(): return None
    value=json.loads(manifest.read_text())
    if value.get('safe_formats_only') is not True: raise ValueError('checkpoint source manifest is not Safetensors-only')
    files=value.get('files')
    if not isinstance(files,dict) or not files: raise ValueError('checkpoint source manifest has no file hashes')
    root=Path(path).resolve()
    for name,expected in files.items():
        file=(root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or digest(file)!=expected:
            raise ValueError('checkpoint source-manifest hash/path mismatch')
    return value

def _environment():
    version=Path('/proc/version').read_text().lower() if Path('/proc/version').is_file() else ''
    return {'os':platform.platform(),'machine':platform.machine(),
        'wsl2':'microsoft' in version,'provenance':'measured-host-environment'}

def _rtl_preflight(info,cfg,prompt_tokens,max_new,trace_root,max_host_gib=16):
    required=info['fp32_tensor_bytes']+6*cfg.DRAM_BYTES
    if required>max_host_gib*1024**3:
        raise ValueError(f'RTL draft preflight needs about {required/1024**3:.2f} GiB; configured host limit is {max_host_gib:.2f} GiB')
    if Path('/proc/meminfo').is_file():
        available=next((int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')),0)
        if required+2*1024**3>available:
            raise ValueError(f'RTL draft plus CUDA model needs about {(required+2*1024**3)/1024**3:.2f} GiB; host available {available/1024**3:.2f} GiB')
    disk_needed=2*cfg.DRAM_BYTES+(prompt_tokens+max_new)*max(1024**2,cfg.DRAM_BYTES//10)+4*1024**3
    if shutil.disk_usage(trace_root).free<disk_needed:
        raise ValueError(f'RTL trace/temp preflight needs about {disk_needed/1024**3:.2f} GiB of free disk')


def gpu_generate(model, prompt, max_new=16, context=2048, dtype='float16',
                 prompt_format='chat', messages=None, standalone_manifest=None,emit=lambda *_:None):
    """Run strict, local-only greedy CUDA generation; rejects CPU fallback."""
    import torch
    import transformers
    gate=_standalone_gate(standalone_manifest)
    if not torch.cuda.is_available():
        raise ValueError('CUDA device required; CPU is not GPU performance evidence')
    if not 1<=max_new<=256 or context>2048:
        raise ValueError('invalid generation/context limits')
    path=Path(model).resolve()
    tok,tokens=_prompt(path,prompt,'raw' if prompt_format=='raw' else 'chat',messages,context)
    if len(tokens)+max_new>context: raise ValueError('prompt plus generation exceeds total context')
    emit('phase',{'phase':'cuda-loading','prompt_tokens':len(tokens)})
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    verifier=CudaVerifier(path,dtype)
    target_info=verifier.model_info
    if target_info['family']!='qwen3': raise ValueError('CUDA verifier currently supports Qwen3 only')
    source=_source_manifest(path)
    load_seconds=time.monotonic()-start
    verifier.reset(tokens)
    emit('phase',{'phase':'cuda-decode'})
    eos=_eos_ids(tok,path)
    generated=[]; start=time.monotonic()
    for _ in range(min(max_new,context-len(tokens))):
        token=verifier.next_token(); generated.append(token)
        emit('token',{'index':len(generated)-1,'token_id':token,
            'text':tok.decode(generated,skip_special_tokens=True),'state':'committed'})
        if token in eos: break
        verifier.advance(token)
    torch.cuda.synchronize(); generation_seconds=time.monotonic()-start
    return {'schema_version':1,'status':'completed','backend':'cuda','mode':'greedy',
        'base_model_id':target_info['base_model_id'],
        'source_repository':source.get('repo') if source else None,
        'source_revision':source.get('revision') if source else None,
        'source_manifest_sha256':identity(source) if source else None,
        'tokenizer_id':identity({'vocabulary':tok.get_vocab(),'special':tok.special_tokens_map,
            'graph':tok.backend_tokenizer.to_str()}),
        'input_token_hash':identity(tokens),'tokens':generated,
        'text':tok.decode(generated,skip_special_tokens=True),'prompt_tokens':len(tokens),
        'max_new':max_new,'context':context,'standalone_release_id':gate['manifest_sha256'],
        'hardware':{'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
            'driver':torch._C._cuda_getDriverVersion(),'cuda_runtime':torch.version.cuda,
            'pytorch':torch.__version__,'transformers':transformers.__version__,'dtype':dtype,
            'attention':'eager','tf32':False,'preflight':verifier.preflight,'provenance':'measured-cuda-host'},
        'host_environment':_environment(),
        'metrics':{'model_load_seconds':{'value':load_seconds,'provenance':'measured-host'},
            'generation_seconds':{'value':generation_seconds,'provenance':'measured-cuda-host'},
            'tokens_per_second':{'value':len(generated)/max(generation_seconds,1e-12),'provenance':'measured-cuda-host'},
            'power':{'value':None,'provenance':'unavailable'},'physical_fpga':{'value':None,'provenance':'unavailable'}}}


def hybrid_generate(draft_model, verifier_model, prompt, max_new=16, context=2048,
                    depth=4, personality='balanced', wformat='int8', dtype='float16',
                    prompt_format='chat', messages=None, standalone_manifest=None,
                    trace_root='build/hybrid-traces', cancel=lambda:False, emit=lambda *_:None):
    """Run GPU greedy baseline and compatible speculative draft, then compare."""
    import torch
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine
    gate=_standalone_gate(standalone_manifest)
    if not torch.cuda.is_available():
        raise ValueError('CUDA device required; CPU is not GPU performance evidence')
    if depth not in (1,2,4,8) or personality not in PERSONALITIES:
        raise ValueError('unsupported hybrid configuration')
    draft_path=Path(draft_model).resolve(); target_path=Path(verifier_model).resolve()
    draft_info=inspect(draft_path,context)
    if not draft_info['supported'] or draft_info['family']!='qwen3':
        raise ValueError('initial hybrid draft must be a supported Qwen3 checkpoint')
    draft_tok,prompt_ids=_prompt(draft_path,prompt,prompt_format,messages,context)
    target=CudaVerifier(target_path,dtype)
    target_tok=target.tokenizer
    if len(prompt_ids)+max_new>context: raise ValueError('prompt plus generation exceeds total context')
    tokenizer_id=tokenizer_compatibility(draft_tok,target_tok)
    target_info=target.model_info
    if target_info['family']!='qwen3': raise ValueError('CUDA verifier currently supports Qwen3 only')
    source=_source_manifest(target_path)
    # GPU-only result is the acceptance reference under identical prompt and
    # target settings, not a second model or an inferred prediction.
    target.reset(prompt_ids); eos=_eos_ids(target_tok,target_path); reference=[]
    emit('phase',{'phase':'gpu-baseline'})
    torch.cuda.synchronize(); baseline_started=time.monotonic()
    for _ in range(min(max_new,context-len(prompt_ids))):
        if cancel(): raise InterruptedError('hybrid cancelled during GPU baseline')
        token=target.next_token(); reference.append(token)
        emit('gpu-baseline-token',{'index':len(reference)-1,'token_id':token})
        if token in eos: break
        target.advance(token)
    torch.cuda.synchronize(); baseline_seconds=time.monotonic()-baseline_started
    spec=load_spec(draft_path)
    cfg=PERSONALITIES[personality].config(spec,context,wformat)
    trace_path=Path(trace_root); trace_path.mkdir(parents=True,exist_ok=True)
    _rtl_preflight(draft_info,cfg,len(prompt_ids),max_new,trace_path)
    weights=load(draft_path)
    from .backend import CheckedRtlBackend
    workload=GenerationWorkload(prompt,prompt_format=prompt_format,context=context,max_new=max_new,
        personality=personality,wformat=wformat,backend='rtl',messages=messages)
    draft_engine=Engine(spec,weights,cap=context,cfg=cfg,rows=1,pipeline=False,
        wformat=wformat,head_format='int8',backend=lambda c,i:CheckedRtlBackend(c,i,workload,trace_path,emit))
    start=time.monotonic()
    result=greedy(EngineDraft(draft_engine),target,prompt_ids,
                  max_new=max_new,depth=depth,eos=eos,cancel=cancel,emit=emit)
    result.update({'schema_version':1,'status':'completed','backend':'hybrid-simulated-draft-cuda-verifier',
        'draft_model_id':draft_info['base_model_id'],'verifier_model_id':target_info['base_model_id'],
        'verifier_repository':source.get('repo') if source else None,
        'verifier_revision':source.get('revision') if source else None,
        'verifier_source_manifest_sha256':identity(source) if source else None,
        'tokenizer_pair_id':tokenizer_id,'input_token_hash':identity(prompt_ids),
        'personality':personality,'wformat':wformat,'depth':depth,'context':context,
        'standalone_release_id':gate['manifest_sha256'],'gpu_greedy_tokens':reference,
        'draft_backend':'full-rtl','draft_rtl_cycles':sum(s.get('cycles',0) for s in draft_engine.stats),
        'draft_steps':len(draft_engine.stats),'gpu_baseline_seconds':baseline_seconds,
        'gpu_baseline_tokens_per_second':len(reference)/max(baseline_seconds,1e-12),
        'draft_axi_read_bytes':sum(s.get('axi_read_bytes',0) for s in draft_engine.stats),
        'draft_useful_macs':sum(s.get('useful_macs',0) for s in draft_engine.stats),
        'gpu_device':torch.cuda.get_device_name(0),'gpu_driver':torch._C._cuda_getDriverVersion(),
        'gpu_runtime':torch.version.cuda,'pytorch':torch.__version__,'dtype':dtype,
        'cuda_preflight':target.preflight,'host_environment':_environment(),
        'attention':'eager','tf32':False,'tokenizer_id':draft_info['tokenizer_id'],
        'output_agrees':result['tokens']==reference,'speedup_established':False,
        'hybrid_host_total_seconds':time.monotonic()-start,
        'measurement_scope':'measured local GPU plus host-simulated draft; not FPGA timing'})
    result['text']=target_tok.decode(result['tokens'],skip_special_tokens=True)
    if not result['output_agrees']:
        result['status']='invalid'; result['failure']='hybrid output differs from GPU-only greedy baseline'
        raise ValueError('hybrid output differs from GPU-only greedy baseline')
    return result
