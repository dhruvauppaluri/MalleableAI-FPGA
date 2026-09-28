"""Portable FP32 GPU baseline; distinct arithmetic and measurement provenance."""
import math
import time
import torch
from .model import LanguageModel
from .tokenizer import Codec


def benchmark(directory,prompt,max_new=8,device='cuda',seed=0):
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    if device not in ('cpu','cuda','mps') or not 1<=max_new<=256: raise ValueError('invalid GPU benchmark settings')
    model,metadata=LanguageModel.load(directory,device); model.eval(); codec=Codec(metadata['tokenizer'])
    tokens=codec.encode(prompt) or [codec.BOS]
    if len(tokens)>2048: raise ValueError('prompt too long')
    def sync():
        if device=='cuda': torch.cuda.synchronize()
        elif device=='mps': torch.mps.synchronize()
    torch.manual_seed(seed)
    with torch.no_grad():
        # Kernel/library warmup is outside measured prompt/decode intervals.
        model.step(torch.tensor([tokens[0]],device=device)); sync()
        states=None; start=time.perf_counter()
        for token in tokens: logits,states=model.step(torch.tensor([token],device=device),states)
        sync(); prefill=time.perf_counter()-start; generated=[]; latencies=[]
        for index in range(max_new):
            start=time.perf_counter(); token=int(logits.argmax(-1).item()); generated.append(token)
            if index+1<max_new: logits,states=model.step(torch.tensor([token],device=device),states)
            sync(); latencies.append(time.perf_counter()-start)
            if token==codec.EOS: break
    ordered=sorted(latencies)
    percentile=lambda q:ordered[min(len(ordered)-1,math.ceil(q*len(ordered))-1)]
    # First generated token was obtained by prefill. Decode service throughput
    # excludes the final sample-only interval and counts actual forward steps.
    decode_steps=max(0,len(generated)-1); decode_seconds=sum(latencies[:decode_steps])
    return dict(schema_version=1,backend='pytorch-portable',device=device,arithmetic='FP32',seed=seed,
                generated_tokens=generated,text=codec.decode(generated),prefill_seconds=prefill,
                decode_tokens_per_second=decode_steps/decode_seconds if decode_seconds else None,
                p50_step_seconds=percentile(.5),p95_step_seconds=percentile(.95),p99_step_seconds=percentile(.99),
                metric_source='measured-host-wall-time-with-device-synchronization',power='unavailable',
                fair_comparison_note='Same floating weights/topology; quantized RTL arithmetic differs. This is not a fused/optimized GPU baseline.')
