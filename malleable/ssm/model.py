"""Small trainable SSMs, using ordinary PyTorch operations on CPU or CUDA.

No fused-kernel dependency; this is a portable correctness/training baseline,
not a GPU performance implementation. Local HF imports never execute remote code.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .artifact import SSMConfig


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        w,n = config.width,config.state_size
        self.norm = nn.Parameter(torch.ones(w))
        if config.family == 'diagonal':
            self.in_proj = nn.Linear(w,w)
            self.gate_proj = nn.Linear(w,w)
            self.decay_logit = nn.Parameter(torch.ones(w,n))
            self.B = nn.Parameter(torch.randn(w,n)*.1)
            self.C = nn.Parameter(torch.randn(w,n)*.1)
            self.out_proj = nn.Linear(w,w)
        else:
            inner = w*config.expansion
            rank = math.ceil(w/16)
            self.in_proj = nn.Linear(w,2*inner,bias=False)
            self.conv_weight = nn.Parameter(torch.randn(inner,config.conv_kernel)*.05)
            self.conv_bias = nn.Parameter(torch.zeros(inner))
            self.x_proj = nn.Linear(inner,rank+2*n,bias=False)
            self.dt_proj = nn.Linear(rank,inner)
            self.A_log = nn.Parameter(torch.log(torch.arange(1,n+1,dtype=torch.float32)).repeat(inner,1))
            self.D = nn.Parameter(torch.ones(inner))
            self.out_proj = nn.Linear(inner,w,bias=False)

    def step(self, x, state=None):
        cfg = self.config
        norm = x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5) * self.norm
        if cfg.family == 'diagonal':
            u = self.in_proj(norm)
            gate = torch.sigmoid(self.gate_proj(norm))
            if state is None:
                state = x.new_zeros(x.shape[0],cfg.width,cfg.state_size)
            new_state = torch.sigmoid(self.decay_logit)*state + u.unsqueeze(-1)*self.B
            y = (new_state*self.C).sum(-1)*gate
            return x+self.out_proj(y),new_state
        inner = cfg.width*cfg.expansion
        if state is None:
            state = (x.new_zeros(x.shape[0],inner,cfg.state_size),
                     x.new_zeros(x.shape[0],inner,cfg.conv_kernel))
        recurrent, history = state
        xz = self.in_proj(norm)
        drive,z = xz.chunk(2,-1)
        history = torch.cat([history[:,:,1:],drive.unsqueeze(-1)],-1)
        drive = F.silu((history*self.conv_weight).sum(-1)+self.conv_bias)
        rank = math.ceil(cfg.width/16)
        dt,B,C = torch.split(self.x_proj(drive),[rank,cfg.state_size,cfg.state_size],-1)
        dt = F.softplus(self.dt_proj(dt))
        decay = torch.exp(-dt.unsqueeze(-1)*self.A_log.exp())
        recurrent = recurrent*decay + drive.unsqueeze(-1)*dt.unsqueeze(-1)*B.unsqueeze(1)
        y = ((recurrent*C.unsqueeze(1)).sum(-1)+drive*self.D)*F.silu(z)
        return x+self.out_proj(y),(recurrent,history)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size,config.width)
        nn.init.normal_(self.embedding.weight,std=.1)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.Parameter(torch.ones(config.width))

    def step(self, tokens, states=None):
        x = self.embedding(tokens)
        states = states or [None]*self.config.layers
        next_states = []
        for block,state in zip(self.blocks,states):
            x,new_state = block.step(x,state)
            next_states.append(new_state)
        x = x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*self.final_norm
        return F.linear(x,self.embedding.weight),next_states

    def forward(self, tokens):
        states = None
        outputs = []
        for t in tokens.unbind(1):
            logits,states = self.step(t,states)
            outputs.append(logits)
        return torch.stack(outputs,1)

    def save(self, directory, tokenizer=None, provenance=None):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(directory/'model.safetensors'))
        (directory/'model.json').write_text(json.dumps({'config':asdict(self.config),'tokenizer':tokenizer or {'kind':'byte-v1'},
                                                     'provenance':provenance or {}},indent=2))

    @classmethod
    def load(cls, directory, device='cpu'):
        from safetensors.torch import load_file
        directory = Path(directory)
        metadata = json.loads((directory/'model.json').read_text())
        model = cls(SSMConfig(**metadata['config']))
        model.load_state_dict(load_file(str(directory/'model.safetensors')),strict=True)
        return model.to(device),metadata


def import_hf(directory, output):
    """Import only local, unsharded Mamba-1 HF Safetensors with tied heads."""
    from safetensors.torch import load_file
    directory = Path(directory)
    cfg = json.loads((directory/'config.json').read_text())
    if cfg.get('model_type') != 'mamba' or cfg.get('hidden_act','silu') != 'silu':
        raise ValueError('only Mamba-1 with SiLU is supported')
    if cfg.get('use_bias',False) or not cfg.get('use_conv_bias',True):
        raise ValueError('unsupported Mamba bias settings')
    config = SSMConfig('mamba1',cfg['hidden_size'],cfg['num_hidden_layers'],cfg.get('state_size',16),
                       cfg.get('expand',2),cfg.get('conv_kernel',4),cfg['vocab_size'])
    if cfg.get('time_step_rank','auto') not in ('auto',math.ceil(config.width/16)):
        raise ValueError('unsupported time-step rank')
    if cfg.get('layer_norm_epsilon',1e-5) != 1e-5:
        raise ValueError('unsupported RMSNorm epsilon')
    weights = load_file(str(directory/'model.safetensors'))
    model = LanguageModel(config)
    mapped = {'embedding.weight':weights['backbone.embeddings.weight'],'final_norm':weights['backbone.norm_f.weight']}
    if 'lm_head.weight' in weights and not torch.equal(weights['lm_head.weight'],mapped['embedding.weight']):
        raise ValueError('untied language head is not supported')
    for i in range(config.layers):
        source=f'backbone.layers.{i}.'
        target=f'blocks.{i}.'
        mapped[target+'norm']=weights[source+'norm.weight']
        for name in ['in_proj.weight','x_proj.weight','dt_proj.weight','dt_proj.bias','A_log','D','out_proj.weight']:
            mapped[target+name]=weights[source+'mixer.'+name]
        mapped[target+'conv_weight']=weights[source+'mixer.conv1d.weight'].squeeze(1)
        mapped[target+'conv_bias']=weights[source+'mixer.conv1d.bias']
    model.load_state_dict(mapped,strict=True)
    tokenizer = json.loads((directory/'tokenizer.json').read_text())
    eos=cfg.get('eos_token_id',0)
    if not isinstance(eos,int): raise ValueError('one EOS token id is required')
    model.save(output,{'kind':'tokenizers-json','json':tokenizer,'eos_id':eos,
                       'bos_id':cfg.get('bos_token_id') if cfg.get('bos_token_id') is not None else eos,
                       'pad_id':cfg.get('pad_token_id') if cfg.get('pad_token_id') is not None else eos},
               {'source':'local-hf-mamba1','path':str(directory.resolve())})
    return model
