"""Greedy speculative decoding protocol; deployment requires standalone release evidence."""
import time
from pathlib import Path
from ..records import identity

def tokenizer_compatibility(draft,target):
    # Vocabulary size alone does not establish identical token semantics.
    probes=['Hello','  spaces\n','é 中文','<|im_start|>user\nHi<|im_end|>']
    def signature(tok):
        return {'vocabulary':tok.get_vocab(),'special':tok.special_tokens_map,
            'added':tok.get_added_vocab(),'chat_template':tok.chat_template,
            'encoded_probes':[tok.encode(s) for s in probes],
            'tokenizer_graph':tok.backend_tokenizer.to_str()}
    a,b=signature(draft),signature(target)
    if a!=b: raise ValueError('draft/verifier tokenizer mappings, specials, formatting or tokenizer graph differ')
    return identity(a)

def greedy(draft,verifier,prompt,max_new=16,depth=4,eos=None,cancel=lambda:False,emit=lambda *_:None):
    if depth not in (1,2,4,8) or not 1<=max_new<=256: raise ValueError('unsupported draft depth/generation length')
    accepted=list(prompt); output=[]; proposed=accepted_count=0; started=time.monotonic()
    timings={'draft_seconds':0.,'verify_seconds':0.,'rollback_seconds':0.}
    draft.reset(accepted); verifier.reset(accepted)
    prefill_seconds=time.monotonic()-started; decode_started=time.monotonic()
    def is_eos(token):
        return token in eos if isinstance(eos,(set,tuple,list)) else token==eos
    while len(output)<max_new:
        if cancel(): raise InterruptedError('hybrid cancelled')
        start=time.monotonic(); block=[]
        for _ in range(min(depth,max_new-len(output))):
            if cancel(): raise InterruptedError('hybrid cancelled')
            token=draft.next_token(); block.append(token); draft.advance(token)
            emit('token-proposal',{'token_id':token,'position':len(output)+len(block)-1,'state':'proposed'})
            if is_eos(token): break
        timings['draft_seconds']+=time.monotonic()-start; proposed+=len(block)
        start=time.monotonic(); predictions=verifier.verify(block)
        if len(predictions)!=len(block): raise ValueError('invalid verifier block result')
        timings['verify_seconds']+=time.monotonic()-start
        n=0; correction=None; ended=False
        for index,(token,expected) in enumerate(zip(block,predictions)):
            if token!=expected:
                correction=expected
                for j,rejected in enumerate(block[index:],index):
                    emit('token-rejected',{'token_id':rejected,'position':len(output)+j-index,'state':'rejected'})
                output.append(correction); accepted.append(correction)
                emit('token-correction',{'token_id':correction,'position':len(output)-1,'state':'corrected'})
                ended=is_eos(correction); break
            n+=1; accepted_count+=1; output.append(token); accepted.append(token)
            emit('token-accepted',{'token_id':token,'position':len(output)-1,'state':'accepted'})
            if is_eos(token): ended=True; break
        verifier.commit(n,correction)
        if n!=len(block) or correction is not None:
            start=time.monotonic(); draft.reset(accepted)
            timings['rollback_seconds']+=time.monotonic()-start
        if ended: break
    decode_seconds=time.monotonic()-decode_started; elapsed=prefill_seconds+decode_seconds
    return {'tokens':output,'proposed_tokens':proposed,'accepted_tokens':accepted_count,
        'acceptance_rate':accepted_count/max(1,proposed),'host_end_to_end_seconds':elapsed,
        'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds,
        'verified_tokens_per_second':len(output)/max(elapsed,1e-12),'timings':timings,
        'speedup_established':False,'projection':'simulation is not physical FPGA timing'}

class CudaVerifier:
    """Qwen3 cache positions are cropped before correcting a rejected suffix."""
    def __init__(self,path,dtype='float16'):
        import json
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not torch.cuda.is_available(): raise ValueError('CUDA required; CPU is not CUDA performance evidence')
        if dtype not in ('float16','float32'): raise ValueError('unsupported CUDA dtype')
        from .models import inspect
        info=inspect(path,2048)
        if info['family']!='qwen3': raise ValueError('initial CUDA verifier supports Qwen3 only')
        self.model_info=info
        cfg=info['config'].get('text_config',info['config'])
        layers=int(cfg['num_hidden_layers']); hidden=int(cfg['hidden_size'])
        heads=int(cfg.get('num_attention_heads',1)); kv_heads=int(cfg.get('num_key_value_heads',heads))
        head_dim=int(cfg.get('head_dim',hidden//heads)); bytes_per=2 if dtype=='float16' else 4
        weights=info['fp32_tensor_bytes']*bytes_per/4
        kv=layers*2048*kv_heads*head_dim*2*bytes_per
        needed=int(weights+kv+2*1024**3)
        free,total=torch.cuda.mem_get_info()
        if needed>free: raise ValueError(f'CUDA preflight needs about {needed/1024**3:.2f} GiB including safety reserve; {free/1024**3:.2f} GiB is free')
        if Path('/proc/meminfo').is_file():
            mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
            host_needed=int(info['fp32_tensor_bytes']*1.5+kv+2*1024**3)
            if mem.get('MemAvailable',0)<host_needed:
                raise ValueError(f'host-memory preflight needs about {host_needed/1024**3:.2f} GiB; {mem.get("MemAvailable",0)/1024**3:.2f} GiB is available')
        torch.backends.cuda.matmul.allow_tf32=False
        self.torch=torch
        self.preflight={'estimated_vram_bytes':needed,'free_vram_bytes':int(free),
            'total_vram_bytes':int(total),'provenance':'estimated-capacity-before-load'}
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,dtype=getattr(torch,dtype),attn_implementation='eager').to('cuda').eval()
        if self.model.config.model_type!='qwen3': raise ValueError('initial CUDA verifier supports Qwen3 only')
        self.cache=None; self.next=None; self.position=0; self.pending=None

    def forward(self,tokens):
        t=self.torch
        with t.no_grad():
            result=self.model(t.tensor([tokens],device='cuda'),past_key_values=self.cache,use_cache=True)
        t.cuda.synchronize(); self.cache=result.past_key_values
        return result.logits[0]

    def reset(self,tokens):
        self.cache=None; self.position=len(tokens); self.next=self.forward(tokens)[-1]; self.pending=None
    def next_token(self): return int(self.next.argmax().item())
    def advance(self,token): self.next=self.forward([token])[-1]; self.position+=1
    def verify(self,block):
        if self.pending is not None: raise ValueError('uncommitted verifier block')
        first=self.next_token(); logits=self.forward(block)
        self.pending=(self.position,logits)
        return [first]+[int(v.argmax().item()) for v in logits[:-1]]
    def commit(self,n,correction):
        if self.pending is None: raise ValueError('no pending verifier block')
        position,logits=self.pending; self.pending=None
        self.cache.crop(position+n); self.position=position+n
        if n: self.next=logits[n-1]
        if correction is not None: self.advance(correction)

class EngineDraft:
    def __init__(self,engine): self.engine=engine; self.next=None
    def reset(self,tokens):
        self.engine.reset()
        for token in tokens: self.next=self.engine.step(token)
    def next_token(self):
        import numpy as np
        return int(np.argmax(self.next[:self.engine.spec.vocab]))
    def advance(self,token): self.next=self.engine.step(token)
