"""Local, resumable SSM training with immutable dataset split provenance."""
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .artifact import SSMConfig
from .model import LanguageModel
from .tokenizer import Codec


def register_dataset(text_path,output,source,revision,license_name):
    if not all([source,revision,license_name]):
        raise ValueError('source, pinned revision and license are required')
    raw=Path(text_path).read_bytes()
    docs=sorted(set(d.strip() for d in raw.decode('utf-8').split('\n\n') if d.strip()))
    if len(docs)<3:
        raise ValueError('at least three distinct blank-line-separated documents required')
    ranked=sorted(docs,key=lambda d:hashlib.sha256(d.encode()).hexdigest())
    validation=max(1,len(docs)//10); heldout=max(1,len(docs)//10)
    splits={'held-out':ranked[:heldout],'validation':ranked[heldout:heldout+validation],
            'train':ranked[heldout+validation:]}
    manifest={'schema_version':1,'source':source,'revision':revision,'license':license_name,
              'content_sha256':hashlib.sha256(raw).hexdigest(),'splits':splits,
              'split_hashes':{k:[hashlib.sha256(d.encode()).hexdigest() for d in v] for k,v in splits.items()}}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(manifest,indent=2))
    return manifest


def load_dataset(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema_version')!=1 or not value.get('license') or not value.get('revision'):
        raise ValueError('invalid dataset manifest')
    seen=set()
    for split in ('train','validation','held-out'):
        docs=value['splits'][split]
        hashes=[hashlib.sha256(d.encode()).hexdigest() for d in docs]
        if not docs or hashes!=value['split_hashes'][split] or any(h in seen for h in hashes):
            raise ValueError('dataset integrity failure or split leakage')
        seen.update(hashes)
    return value


def train(dataset_path,output,config=None,steps=100,sequence_length=32,seed=0,device='cpu',resume=False,cancel=None,progress=None):
    if steps<1 or not 2<=sequence_length<=2048 or device not in ('cpu','cuda','mps'):
        raise ValueError('invalid training settings')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use your NVIDIA/WSL2 worker')
    data=load_dataset(dataset_path)
    digest=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); rng=random.Random(seed)
    torch.use_deterministic_algorithms(True)
    config=config or SSMConfig()
    model=LanguageModel(config).to(device)
    metadata={'tokenizer':{'kind':'byte-v1'}}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    first_step=0; losses=[]
    checkpoint_path=output/'training.pt'
    if resume:
        # Only locally generated tensor checkpoints, never downloaded pickle.
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if checkpoint['dataset_hash']!=digest or checkpoint['config']!=vars(config):
            raise ValueError('resume dataset/config differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng']); rng.setstate(checkpoint['python_rng'])
        first_step=checkpoint['step']; losses=checkpoint['losses']
        if device=='cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    codec=Codec(metadata['tokenizer'])
    documents=[codec.encode(d)+[codec.EOS] for d in data['splits']['train']]
    start=time.perf_counter(); completed=first_step; status='completed'
    for step in range(first_step,first_step+steps):
        if cancel and cancel():
            status='cancelled'; break
        tokens=rng.choice(documents)
        if len(tokens)<=sequence_length:
            tokens=(tokens*((sequence_length+1)//len(tokens)+1))[:sequence_length+1]
        else:
            offset=rng.randrange(len(tokens)-sequence_length)
            tokens=tokens[offset:offset+sequence_length+1]
        batch=torch.tensor([tokens],dtype=torch.long,device=device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,config.vocab_size),batch[:,1:].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        losses.append(float(loss.detach().cpu())); completed=step+1
        if progress:
            progress({'step':completed,'loss':losses[-1]})
        torch.save({'schema_version':1,'config':vars(config),'dataset_hash':digest,'step':completed,
                    'model':model.state_dict(),'optimizer':optimizer.state_dict(),'losses':losses,
                    'torch_rng':torch.get_rng_state(),'python_rng':rng.getstate(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None},checkpoint_path)
    provenance=dict(training_steps=completed,dataset_hash=digest,dataset_source=data['source'],
                    dataset_revision=data['revision'],license=data['license'],seed=seed,device=device,
                    split='train',trained_model_quality='not-promoted')
    model.save(output,metadata['tokenizer'],provenance)
    report=dict(schema_version=1,status=status,training_steps=completed,losses=losses,
                wall_seconds=time.perf_counter()-start,provenance=provenance)
    (output/'training.json').write_text(json.dumps(report,indent=2))
    return report


def quality(model_directory,artifact_path,dataset_path,split='held-out',max_tokens=128):
    from .compiler import compile_token
    from .numeric import execute
    if split not in ('validation','held-out'):
        raise ValueError('quality promotion cannot evaluate training split')
    data=load_dataset(dataset_path)
    model,metadata=LanguageModel.load(model_directory)
    model.eval(); codec=Codec(metadata['tokenizer'])
    compiled=compile_token(artifact_path)
    if vars(model.config)!=compiled['manifest']['config']:
        raise ValueError('floating and quantized model configurations differ')
    # Prevent evaluating an unrelated artifact with a conveniently good model.
    from .compiler import export
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        expected=export(model,metadata,Path(directory)/'expected.mssm')
    if expected!=compiled['model_id']:
        raise ValueError('quantized artifact does not belong to floating model')
    float_loss=0; integer_loss=0; agreement=0; count=0; saturations=0
    with torch.no_grad():
        for doc in data['splits'][split]:
            tokens=(codec.encode(doc)+[codec.EOS])[:max_tokens+1]
            if len(tokens)<2: continue
            floating=model(torch.tensor([tokens[:-1]],dtype=torch.long))[0]
            memory=list(compiled['memory']); outputs=[]
            w=model.config.width
            for token in tokens[:-1]:
                emb=compiled['embedding_addr']+token*w
                memory[compiled['input_addr']:compiled['input_addr']+w]=memory[emb:emb+w]
                memory,sat_count=execute(memory,compiled['program']); saturations+=sat_count
                outputs.append(memory[compiled['logits_addr']:compiled['logits_addr']+model.config.vocab_size])
            fixed=torch.tensor(outputs,dtype=torch.float32)/16384
            targets=torch.tensor(tokens[1:])
            float_loss+=float(F.cross_entropy(floating,targets,reduction='sum'))
            integer_loss+=float(F.cross_entropy(fixed,targets,reduction='sum'))
            agreement+=int((floating.argmax(-1)==fixed.argmax(-1)).sum()); count+=len(targets)
    if not count: raise ValueError('no evaluation tokens')
    degradation=integer_loss/float_loss-1
    result=dict(schema_version=1,split=split,tokens=count,float_nll=float_loss/count,integer_nll=integer_loss/count,
                relative_nll_degradation=degradation,top1_agreement=agreement/count,saturations=saturations,
                quantization_gate_passed=bool(degradation<=.05 and agreement/count>=.9 and saturations==0),
                chatbot_quality_proven=False,model_id=compiled['model_id'],dataset_hash=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest())
    return result
