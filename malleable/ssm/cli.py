import argparse
import json
from pathlib import Path
from .artifact import SSMConfig


def main():
    parser=argparse.ArgumentParser(description='Simulation-first SSM research platform')
    parser.add_argument('command',choices=['init','export','generate','analyze','dataset','train','quality','import-hf','optimize','quartus','gpu','policy-train','policy-evaluate'])
    parser.add_argument('--model',default='build/ssm-model')
    parser.add_argument('--artifact',default='build/model.mssm')
    parser.add_argument('--family',choices=['diagonal','mamba1'],default='diagonal')
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--state-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--prompt',default='Once upon a time')
    parser.add_argument('--max-new',type=int,default=8)
    parser.add_argument('--top-k',type=int,default=1)
    parser.add_argument('--backend',choices=['rtl','integer'],default='rtl')
    parser.add_argument('--lanes',type=int,default=4)
    parser.add_argument('--active-lanes',type=int)
    parser.add_argument('--clock-hz',type=float)
    parser.add_argument('--dataset',default='build/dataset.json')
    parser.add_argument('--text',type=Path)
    parser.add_argument('--source')
    parser.add_argument('--revision')
    parser.add_argument('--license')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--sequence-length',type=int,default=32)
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--split',choices=['validation','held-out'],default='held-out')
    parser.add_argument('--horizon',type=int,default=100)
    parser.add_argument('--switch-cost',help='JSON cycles: drain,program,reload,warmup,compile (explicit scenario)')
    parser.add_argument('--budget',type=int,default=5)
    parser.add_argument('--profile',choices=['latency','throughput','energy'],default='latency')
    parser.add_argument('--power',help='Explicit credible watts JSON by lanes, energy profile only')
    parser.add_argument('--store',default='build/ssm-research')
    parser.add_argument('--quartus-sh',default='quartus_sh')
    parser.add_argument('--artifacts',nargs='+')
    parser.add_argument('--checkpoint')
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--rtl',action='store_true')
    args=parser.parse_args()
    config=SSMConfig(family=args.family,width=args.width,layers=args.layers,state_size=args.state_size)
    if args.command=='init':
        import torch
        from .model import LanguageModel
        torch.manual_seed(args.seed); model=LanguageModel(config)
        model.save(args.model,provenance={'seed':args.seed,'untrained_fixture':True})
        result={'model':args.model,'trained':False,'warning':'Random weights; generated text is not a trained chatbot.'}
    elif args.command=='export':
        from .model import LanguageModel
        from .compiler import export
        model,metadata=LanguageModel.load(args.model)
        result={'model_id':export(model,metadata,args.artifact),'artifact':args.artifact,'numeric':'q14-prototype-v1'}
    elif args.command=='import-hf':
        from .model import import_hf
        model=import_hf(args.source,args.model); result={'config':vars(model.config),'model':args.model}
    elif args.command=='dataset':
        from .training import register_dataset
        if args.text is None: parser.error('dataset requires --text')
        result=register_dataset(args.text,args.dataset,args.source,args.revision,args.license)
    elif args.command=='train':
        from .training import train
        import sys
        result=train(args.dataset,args.model,config,args.steps,args.sequence_length,args.seed,args.device,args.resume,
                     progress=lambda row:print(json.dumps(row),file=sys.stderr,flush=True))
    elif args.command=='quality':
        from .training import quality
        result=quality(args.model,args.artifact,args.dataset,args.split)
    elif args.command=='quartus':
        from ..quartus import Personality,automatic_build
        result=automatic_build(Personality(lanes=args.lanes),executable=args.quartus_sh)
    elif args.command=='gpu':
        from .gpu import benchmark
        result=benchmark(args.model,args.prompt,args.max_new,args.device,args.seed)
    else:
        from ..store import Store
        store=Store(args.store)
        try:
            if args.command.startswith('policy-'):
                from .learning import train,evaluate
                if not args.artifacts: parser.error('policy commands require --artifacts PATH ...')
                if args.command=='policy-train':
                    result={'checkpoint':train(args.artifacts,store,args.episodes,args.seed,
                                               json.loads(args.switch_cost) if args.switch_cost else None,args.profile)}
                else:
                    if not args.checkpoint: parser.error('policy-evaluate requires --checkpoint HASH')
                    result=evaluate(store.load(args.checkpoint),args.artifacts,store,rtl=args.rtl)
            elif args.command=='generate':
                from .backend import generate
                result=generate(args.artifact,args.prompt,args.max_new,args.seed,args.top_k,args.lanes,args.active_lanes,
                                args.backend,args.clock_hz,store)
            elif args.command=='optimize':
                from .optimization import optimize
                result=optimize(args.artifact,args.prompt,{'lanes':args.lanes,'active_lanes':args.active_lanes or args.lanes},
                                args.horizon,json.loads(args.switch_cost) if args.switch_cost else None,args.budget,args.profile,
                                args.clock_hz,json.loads(args.power) if args.power else None,store)
            else:
                from .compiler import compile_token
                from .optimization import service_cycles
                compiled=compile_token(args.artifact)
                result={'model_id':compiled['model_id'],'config':compiled['manifest']['config'],
                        'memory_bytes':len(compiled['memory'])*4,'program_bytes':len(compiled['program'])*32,
                        'instructions':len(compiled['program']),'predicted_cycles_per_token':service_cycles(compiled,args.active_lanes or args.lanes),
                        'cycle_source':'estimated-instruction-model','physical_fit':'unavailable until Quartus compilation'}
        finally: store.close()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
