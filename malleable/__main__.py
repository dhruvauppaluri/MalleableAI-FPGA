import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .records import ModelArtifact, WorkloadSpec, ExecutionConfig, HardwareProfile
from .model import fixture, analyze
from .store import Store
from .experiment import benchmark, suite, optimize_run
from .optimize import Predictor, decide
from .learning import train, evaluate, promote, rollback


def main():
    parser = argparse.ArgumentParser(description='Dense FPGA model-aware experiments')
    parser.add_argument('command',choices=['analyze','benchmark','optimize','train','evaluate','suite','promote','rollback'])
    parser.add_argument('--model',type=Path)
    parser.add_argument('--size',choices=['light','medium','heavy'],default='light')
    parser.add_argument('--store',default='build/research')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--lanes',type=int,default=4)
    parser.add_argument('--active-lanes',type=int,default=4)
    parser.add_argument('--requests',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=8)
    parser.add_argument('--switch-cycles',type=int)
    parser.add_argument('--profile',choices=['latency','throughput','energy'],default='latency')
    parser.add_argument('--power-json',help='JSON mapping compiled lanes to calibrated watts')
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--checkpoint')
    parser.add_argument('--validation')
    parser.add_argument('--heldout')
    parser.add_argument('--split',choices=['validation','held-out'],default='held-out')
    parser.add_argument('--rtl',action='store_true',help='Collect independent RTL evidence during evaluation')
    parser.add_argument('--exhaustive',action='store_true')
    parser.add_argument('--execute',action='store_true',help='Validate optimizer candidates with RTL experiments')
    parser.add_argument('--budget',type=int,default=8)
    parser.add_argument('--arrival',choices=['isolated','sustained','burst'],default='sustained')
    parser.add_argument('--interval-cycles',type=int,default=100)
    parser.add_argument('--deadline-cycles',type=int)
    args = parser.parse_args()
    model = ModelArtifact(**json.loads(args.model.read_text())) if args.model else fixture(args.size,args.seed)
    hardware = HardwareProfile(switch_cycles=args.switch_cycles,
                               power_watts=json.loads(args.power_json) if args.power_json else {})
    workload = WorkloadSpec(request_count=args.requests,horizon=args.horizon,seed=args.seed,
                            arrival_pattern=args.arrival,interval_cycles=args.interval_cycles,
                            deadline_cycles=args.deadline_cycles)
    config = ExecutionConfig(args.lanes,args.active_lanes)
    store = Store(args.store)
    try:
        if args.command == 'analyze':
            result = analyze(model)
        elif args.command == 'benchmark':
            result = asdict(benchmark(model,workload,config,hardware,store))
            if not result['valid']:
                raise RuntimeError(result['failure'])
        elif args.command == 'optimize':
            if args.execute:
                result = dict(optimization=optimize_run(model,workload,hardware,config,store,args.profile,args.budget))
                print(json.dumps(result,indent=2))
                return
            predictor = Predictor()
            for row in store.records('experiment'):
                if row['valid'] and row['model_id'] == model.model_id:
                    trace = store.load(row['provenance']['trace'])
                    predictor.update(model,ExecutionConfig(**row['config']),trace['counters'][0]['cycles'])
            result = asdict(decide(model,workload,hardware,config,args.profile,predictor))
            store.save('decision',result)
        elif args.command == 'train':
            result = dict(checkpoint=train(hardware,store,args.episodes,args.seed,args.profile,
                                          store.load(args.checkpoint) if args.checkpoint else None))
        elif args.command == 'suite':
            result = dict(suite=suite(hardware,store,args.seed,args.exhaustive))
        elif args.command == 'promote':
            if not all([args.checkpoint,args.validation,args.heldout]):
                parser.error('promote requires --checkpoint, --validation and --heldout hashes')
            result = dict(deployment=promote(store,args.checkpoint,args.validation,args.heldout))
        elif args.command == 'rollback':
            result = dict(deployment=rollback(store))
        else:
            if not args.checkpoint:
                parser.error('evaluate requires --checkpoint HASH')
            result = evaluate(store.load(args.checkpoint),hardware,store,
                              range(200000,200005) if args.split == 'validation' else range(100000,100005),
                              args.split,args.rtl)
            result = dict(report=store.save('evaluation',result),**result)
        print(json.dumps(result,indent=2,allow_nan=False))
    finally:
        store.close()


if __name__ == '__main__':
    main()
