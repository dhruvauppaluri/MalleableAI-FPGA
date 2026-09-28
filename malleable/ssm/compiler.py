"""Lower SSM token steps into portable operators; recurrent state stays in RAM."""
import math
from .numeric import ONE, quantize, execute
from .artifact import load, save


class Program:
    def __init__(self):
        self.memory = [0]
        self.instructions = []

    def alloc(self, values):
        address=len(self.memory)
        self.memory.extend(values)
        return address

    def buffer(self, length):
        return self.alloc([0]*length)

    def emit(self, op, a, b, length, aux=0, shift=14, bias=0, dst=None):
        dst=self.buffer(1 if op==0 else length) if dst is None else dst
        self.instructions.append([op,dst,a,b,length,aux,shift,bias])
        return dst


def export(model, metadata, path):
    cfg=model.config
    tensors={k:[quantize(float(v)) for v in tensor.detach().cpu().reshape(-1)]
             for k,tensor in model.state_dict().items()}
    # Explicit nonlinear tables. Per-state selective decay tables preserve
    # different A values; they are not treated as one universal SSM operator.
    tensors['lut.sigmoid']=[quantize(1/(1+math.exp(-(i-128)/64))) for i in range(256)]
    tensors['lut.silu']=[quantize(((i-128)/64)/(1+math.exp(-(i-128)/64))) for i in range(256)]
    tensors['lut.softplus']=[quantize(math.log1p(math.exp((i-128)/64))) for i in range(256)]
    for i,block in enumerate(model.blocks):
        prefix=f'blocks.{i}.'
        if cfg.family=='diagonal':
            tensors[prefix+'decay']=[quantize(float(v)) for v in block.decay_logit.detach().sigmoid().reshape(-1)]
        else:
            for j,rate in enumerate(block.A_log.detach().cpu().exp().reshape(-1)):
                tensors[prefix+f'decay.{j}']=[quantize(math.exp(-float(rate)*max(0,(index-128)/64))) for index in range(256)]
    return save(path,{'config':vars(cfg),'numeric':'q14-prototype-v1','tokenizer':metadata.get('tokenizer',{'kind':'byte-v1'}),
                      'provenance':metadata.get('provenance',{}),'trained':bool(metadata.get('provenance',{}).get('training_steps'))},tensors)


def compile_token(path):
    manifest,tensors,model_id=load(path)
    cfg=manifest['config']; w,n=cfg['width'],cfg['state_size']
    expected={'embedding.weight':cfg['vocab_size']*w,'final_norm':w,
              'lut.sigmoid':256,'lut.silu':256,'lut.softplus':256}
    for layer in range(cfg['layers']):
        base=f'blocks.{layer}.'
        expected[base+'norm']=w
        if cfg['family']=='diagonal':
            for proj in ('in_proj','gate_proj','out_proj'):
                expected[base+proj+'.weight']=w*w; expected[base+proj+'.bias']=w
            for name in ('decay_logit','B','C','decay'): expected[base+name]=w*n
        else:
            inner=w*cfg['expansion']; rank=math.ceil(w/16)
            for name,size in {'in_proj.weight':2*inner*w,'conv_weight':inner*cfg['conv_kernel'],
                              'conv_bias':inner,'x_proj.weight':(rank+2*n)*inner,'dt_proj.weight':inner*rank,
                              'dt_proj.bias':inner,'A_log':inner*n,'D':inner,'out_proj.weight':w*inner}.items():
                expected[base+name]=size
            for j in range(inner*n): expected[base+f'decay.{j}']=256
    if set(tensors)!=set(expected) or any(len(tensors[k])!=size for k,size in expected.items()):
        raise ValueError('artifact tensor names/shapes do not match declared model')
    p=Program()
    params={k:p.alloc(v) for k,v in tensors.items()}
    input_addr=p.buffer(w)
    state_addresses=[]

    def matrix(name,a,k,rows):
        dst=p.buffer(rows)
        biases=params.get(name+'.bias',0)
        for j in range(rows):
            p.emit(0,a,params[name+'.weight']+j*k,k,bias=biases+j if biases else 0,dst=dst+j)
        return dst

    def norm(a,weight):
        return p.emit(2,p.emit(4,a,0,w,aux=1),params[weight],w)

    def nonlinear(name,a,count):
        return p.emit(3,a,0,count,aux=params['lut.'+name],shift=8)

    x=input_addr
    for layer in range(cfg['layers']):
        base=f'blocks.{layer}.'
        normalized=norm(x,base+'norm')
        inner=w if cfg['family']=='diagonal' else w*cfg['expansion']
        state=p.buffer(inner*n)
        state_addresses.append((state,inner*n))
        if cfg['family']=='diagonal':
            u=matrix(base+'in_proj',normalized,w,w)
            gate=nonlinear('sigmoid',matrix(base+'gate_proj',normalized,w,w),w)
            drive=p.buffer(w*n)
            for c in range(w):
                for s in range(n):
                    p.emit(2,u+c,params[base+'B']+c*n+s,1,dst=drive+c*n+s)
            decayed=p.emit(2,state,params[base+'decay'],w*n)
            new=p.emit(1,decayed,drive,w*n)
            weighted=p.emit(2,new,params[base+'C'],w*n)
            y=p.buffer(w)
            ones=p.alloc([ONE]*n)
            for c in range(w):
                p.emit(0,weighted+c*n,ones,n,dst=y+c)
            p.emit(5,new,0,w*n,dst=state)
            y=p.emit(2,y,gate,w)
        else:
            xz=matrix(base+'in_proj',normalized,w,2*inner)
            history=p.buffer(inner*cfg['conv_kernel'])
            state_addresses.append((history,inner*cfg['conv_kernel']))
            conv=p.buffer(inner)
            for c in range(inner):
                updated=p.buffer(cfg['conv_kernel'])
                if cfg['conv_kernel']>1:
                    p.emit(5,history+c*cfg['conv_kernel']+1,0,cfg['conv_kernel']-1,dst=updated)
                p.emit(5,xz+c,0,1,dst=updated+cfg['conv_kernel']-1)
                p.emit(5,updated,0,cfg['conv_kernel'],dst=history+c*cfg['conv_kernel'])
                p.emit(0,updated,params[base+'conv_weight']+c*cfg['conv_kernel'],cfg['conv_kernel'],
                       bias=params[base+'conv_bias']+c,dst=conv+c)
            u=nonlinear('silu',conv,inner)
            rank=math.ceil(w/16)
            projected=matrix(base+'x_proj',u,inner,rank+2*n)
            dt=nonlinear('softplus',matrix(base+'dt_proj',projected,rank,inner),inner)
            drive=p.buffer(inner*n); decay=p.buffer(inner*n); cout=p.buffer(inner*n)
            for c in range(inner):
                du=p.emit(2,dt+c,u+c,1)
                for s in range(n):
                    j=c*n+s
                    p.emit(2,du,projected+rank+s,1,dst=drive+j)
                    p.emit(3,dt+c,0,1,aux=params[base+f'decay.{j}'],shift=8,dst=decay+j)
                    p.emit(5,projected+rank+n+s,0,1,dst=cout+j)
            decayed=p.emit(2,state,decay,inner*n)
            new=p.emit(1,decayed,drive,inner*n)
            weighted=p.emit(2,new,cout,inner*n)
            y=p.buffer(inner); ones=p.alloc([ONE]*n)
            for c in range(inner):
                p.emit(0,weighted+c*n,ones,n,dst=y+c)
            p.emit(5,new,0,inner*n,dst=state)
            y=p.emit(1,y,p.emit(2,u,params[base+'D'],inner),inner)
            y=p.emit(2,y,nonlinear('silu',xz+inner,inner),inner)
        x=p.emit(1,x,matrix(base+'out_proj',y,inner,w),w)
    x=norm(x,'final_norm')
    logits=p.buffer(cfg['vocab_size'])
    for j in range(cfg['vocab_size']):
        p.emit(0,x,params['embedding.weight']+j*w,w,dst=logits+j)
    p.instructions.append([255,0,0,0,0,0,0,0])
    # Check tensor shape and instruction legality without needing an RTL tool.
    execute(p.memory,p.instructions)
    return {'manifest':manifest,'model_id':model_id,'memory':p.memory,'program':p.instructions,
            'input_addr':input_addr,'logits_addr':logits,'state_addresses':state_addresses,
            'embedding_addr':params['embedding.weight']}
