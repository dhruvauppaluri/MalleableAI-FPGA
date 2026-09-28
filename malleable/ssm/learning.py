"""SSM Double DQN research policy. Estimated rewards are explicitly labeled.

Uses the same small, dependency-free network implementation as the dense
controller, but separate action/state schemas and independent checkpoints.
No automatic policy/model promotion; held-out evidence is mandatory first.
"""
from collections import deque
from copy import deepcopy
import math
import random
from ..learning import Network,argmax
from .compiler import compile_token
from .optimization import choose,service_cycles

ACTIONS=[{'lanes':l,'active_lanes':a} for l in (1,2,4,8,16) for a in range(1,l+1)]


class SSMNetwork(Network):
    def __init__(self,rng,hidden=16):
        self.w1=[[rng.uniform(-.1,.1) for _ in range(13)] for _ in range(hidden)]
        self.w2=[[rng.uniform(-.1,.1) for _ in range(hidden+1)] for _ in ACTIONS]


def features(compiled,current,horizon,profile):
    cfg=compiled['manifest']['config']
    return [1.,cfg['width']/128,cfg['layers']/6,cfg['state_size']/16,
            len(compiled['memory'])/131072,len(compiled['program'])/8192,
            current['lanes']/16,current['active_lanes']/16,horizon/1024,
            service_cycles(compiled,current['active_lanes'])/100000,
            {'latency':0.,'throughput':.5,'energy':1.}[profile],0.,0.]


def candidates(compiled):
    cfg=compiled['manifest']['config']
    return [dict(a,service_cycles=service_cycles(compiled,a['active_lanes'])+2*(cfg['width']+cfg['vocab_size']),
                 reload_cycles=2*(len(compiled['memory'])+8*len(compiled['program']))) for a in ACTIONS]


def masked(current,switch_cost):
    return [i for i,a in enumerate(ACTIONS) if a['lanes']==current['lanes'] or switch_cost is not None]


def objective(c,current,horizon,switch_cost,profile):
    switch=c['lanes']!=current['lanes']
    overhead=sum((switch_cost or {}).values())+c['reload_cycles'] if switch else 0
    return (c['service_cycles']*(horizon+1)/2+overhead if profile=='latency' else
            c['service_cycles']+overhead/horizon)


def train(paths,store,episodes=100,seed=0,switch_cost=None,profile='latency'):
    if profile not in ('latency','throughput'):
        raise ValueError('SSM policy energy training awaits credible power evidence')
    if episodes<1: raise ValueError('episodes must be positive')
    if any(not math.isfinite(v) or v<0 for v in (switch_cost or {}).values()): raise ValueError('invalid overhead')
    models=[compile_token(path) for path in paths]
    if not models: raise ValueError('training artifacts required')
    rng=random.Random(seed); online=SSMNetwork(rng); target=deepcopy(online)
    replay=deque(maxlen=10000); steps=0
    for episode in range(episodes):
        current={'lanes':4,'active_lanes':4}
        windows=[(rng.randrange(len(models)),rng.choice([1,8,64,1024])) for _ in range(8)]
        for t,(mi,horizon) in enumerate(windows):
            model=models[mi]; rows=candidates(model); state=features(model,current,horizon,profile)
            mask=masked(current,switch_cost)
            action=rng.choice(mask) if rng.random()<max(.05,.7*(1-episode/episodes)) else argmax(online.forward(state)[1],mask)
            chosen=ACTIONS[action]
            baseline=objective(rows[ACTIONS.index(current)],current,horizon,switch_cost,profile)
            reward=-objective(rows[action],current,horizon,switch_cost,profile)/baseline
            ni,nh=windows[min(t+1,len(windows)-1)]
            next_state=features(models[ni],chosen,nh,profile)
            replay.append((state,action,reward,next_state,masked(chosen,switch_cost),t==len(windows)-1))
            if len(replay)>=8:
                for s,a,r,ns,nmask,terminal in rng.sample(list(replay),8):
                    best=argmax(online.forward(ns)[1],nmask)
                    boot=0 if terminal else .95*target.forward(ns)[1][best]
                    online.fit(s,a,r+boot)
            steps+=1
            if steps%25==0: target=deepcopy(online)
            current=chosen
    checkpoint=dict(schema_version=1,algorithm='DoubleDQN',state_schema='ssm-v1',action_schema=ACTIONS,
                    reward_source='estimated-instruction-timing-plus-explicit-overhead',seed=seed,episodes=episodes,
                    switch_cost=switch_cost,profile=profile,training_models=sorted({m['model_id'] for m in models}),
                    w1=online.w1,w2=online.w2,target_w1=target.w1,target_w2=target.w2,
                    replay=list(replay),status='candidate',deployment_approved=False)
    return store.save('ssm-policy',checkpoint)


def evaluate(checkpoint,paths,store,seeds=range(100000,100005),rtl=False):
    from .backend import reference_trace,run_rtl
    if checkpoint.get('state_schema')!='ssm-v1' or checkpoint.get('action_schema')!=ACTIONS:
        raise ValueError('incompatible checkpoint')
    seeds=list(seeds)
    if len(set(seeds))<5: raise ValueError('five independent evaluation seeds required')
    models=[compile_token(path) for path in paths]
    if not models or any(m['model_id'] in checkpoint['training_models'] for m in models):
        raise ValueError('held-out model leakage or empty evaluation')
    net=SSMNetwork(random.Random(0)); net.w1=checkpoint['w1']; net.w2=checkpoint['w2']
    totals={k:[] for k in ('rl','fixed','heuristic','random','exhaustive')}; evidence=[]
    switch=checkpoint['switch_cost']; profile=checkpoint['profile']
    for seed in seeds:
        rng=random.Random(seed); current={k:{'lanes':4,'active_lanes':4} for k in totals}
        score=dict.fromkeys(totals,0.)
        for _ in range(8):
            model=rng.choice(models); rows=candidates(model); horizon=rng.choice([1,8,64,1024])
            for method in totals:
                mask=masked(current[method],switch)
                if method=='rl': action=argmax(net.forward(features(model,current[method],horizon,profile))[1],mask)
                elif method=='fixed': action=ACTIONS.index({'lanes':4,'active_lanes':4})
                elif method=='random': action=rng.choice(mask)
                elif method=='heuristic':
                    decision=choose(rows,current[method],horizon,switch,profile=profile)
                    action=ACTIONS.index(decision['config'])
                else: action=min(mask,key=lambda i:objective(rows[i],current[method],horizon,switch,profile))
                score[method]+=objective(rows[action],current[method],horizon,switch,profile)
                current[method]=ACTIONS[action]
                if rtl and method=='rl':
                    records,_,_=reference_trace(model,'test',0)
                    measured=run_rtl(model,records,**ACTIONS[action])
                    if measured['counters'][0]['cycles']!=service_cycles(model,ACTIONS[action]['active_lanes']):
                        raise RuntimeError('held-out timing prediction mismatch')
                    evidence.append({'model_id':model['model_id'],'seed':seed,'bit_exact':measured['bit_exact'],
                                     'config':ACTIONS[action],'trace_path':measured['trace_path']})
        for k in totals: totals[k].append(score[k])
    report=dict(schema_version=1,split='held-out',seeds=seeds,models=[m['model_id'] for m in models],
                costs=totals,mean_cost={k:sum(v)/len(v) for k,v in totals.items()},evidence=evidence,
                reward_source='estimated-instruction-model; measured RTL checks listed separately',
                policy_promoted=False,quality_gate='not-evaluated',random_search_budget=1,
                warning='Research evaluation only; no automatic continual deployment or chatbot quality claim')
    report['record_id']=store.save('ssm-policy-evaluation',report)
    return report
