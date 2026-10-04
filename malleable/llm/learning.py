"""Offline masked Double DQN over versioned, measured-evidence episode tables.

Window rewards are estimated from measured RTL service cycles and supplied
switch costs. They are never presented as live/physical measurements.
"""
from collections import deque
from copy import deepcopy
import random
import math
from dataclasses import asdict
from ..records import identity
from .optimization import DecisionWindow,objective_cost
from .records import PERSONALITIES

ACTIONS=[p+'/'+fmt for p in PERSONALITIES for fmt in ('int8','int4','fp4')]
SIZE=10+len(ACTIONS)*2

def validate_release_episodes(episodes,split):
    from .optimization import PREDICTOR_TRAINING_MODELS,PREDICTOR_HELD_OUT_MODEL
    validate(episodes)
    models={w['base_model_id'] for ep in episodes for w in ep}
    expected=PREDICTOR_TRAINING_MODELS if split=='training' else {PREDICTOR_HELD_OUT_MODEL}
    if models!=expected: raise ValueError('frozen controller train/evaluation model partition mismatch')
    actions={p+'/int8' for p in PERSONALITIES}
    if any(set(w['service_cycles'])!=actions for ep in episodes for w in ep):
        raise ValueError('primary controller evaluation requires four INT8 personalities')
    coverage={(w.get('window',{}).get('objective','latency'),w.get('window',{}).get('remaining_requests',1))
              for ep in episodes for w in ep}
    if not {(o,h) for o in ('latency','throughput') for h in (1,8,32,128)}<=coverage:
        raise ValueError('latency/throughput horizons 1, 8, 32 and 128 required')

def validate(episodes):
    if not isinstance(episodes,list) or not episodes: raise ValueError('nonempty sequential episodes required')
    for ep in episodes:
        if not isinstance(ep,list) or not ep: raise ValueError('nonempty episode required')
        for w in ep:
            if not w.get('base_model_id') or not w.get('family'): raise ValueError('model identity and family required')
            if w.get('provenance')!='estimated-window-from-measured-rtl': raise ValueError('measured RTL lineage required')
            if not w.get('evidence_ids'): raise ValueError('source experiment hashes required')
            features=w.get('features',[0.]*6)
            if not isinstance(features,list) or len(features)!=6 or any(type(v) not in (int,float)
                or not math.isfinite(v) for v in features): raise ValueError('six finite model/workload features required')
            if not w.get('service_cycles') or any(k not in ACTIONS or type(v) not in (int,float)
                or not math.isfinite(v) or v<=0 for k,v in w['service_cycles'].items()): raise ValueError('invalid legal action table')
            estimates=w.get('predicted_cycles')
            if not w.get('predictor_id') or not isinstance(estimates,dict) or set(estimates)!=set(w['service_cycles']) or any(
                type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in estimates.values()):
                raise ValueError('frozen predictor estimates required; oracle timings are not observations')
            DecisionWindow(**w.get('window',{}))

def mask(w,current,residence):
    window=DecisionWindow(**w.get('window',{})); costs=window.switching_costs or {}
    return [i for i,a in enumerate(ACTIONS) if a in w['service_cycles'] and
        (a==current or (residence>=1 and current+'->'+a in costs))]

def cost(w,current,chosen):
    window=DecisionWindow(**w.get('window',{}))
    costs=(window.switching_costs or {}).get(current+'->'+chosen,{}) if chosen!=current else {}
    return objective_cost(w['service_cycles'][chosen],window,sum(costs.values()))

def state(w,current,residence):
    values=w['predicted_cycles']; norm=max(values.values())
    window=DecisionWindow(**w.get('window',{}))
    return [1.,min(window.remaining_requests/1024,1.),float(window.objective=='throughput'),min(residence,1),
            *w.get('features',[0.]*6),*[float(a==current) for a in ACTIONS],*[values.get(a,norm*2)/norm for a in ACTIONS]]

class Network:
    def __init__(self,rng):
        self.w1=[[rng.uniform(-.1,.1) for _ in range(SIZE)] for _ in range(24)]
        self.w2=[[rng.uniform(-.1,.1) for _ in range(25)] for _ in ACTIONS]
    def forward(self,x):
        h=[math.tanh(sum(a*b for a,b in zip(row,x))) for row in self.w1]+[1.]
        return h,[sum(a*b for a,b in zip(row,h)) for row in self.w2]
    def fit(self,x,action,target,rate=.001):
        h,q=self.forward(x); error=max(-1.,min(1.,q[action]-target)); old=self.w2[action][:]
        for j in range(25): self.w2[action][j]-=rate*error*h[j]
        for j,row in enumerate(self.w1):
            for k in range(SIZE): row[k]-=rate*error*old[j]*(1-h[j]**2)*x[k]

def argmax(q,legal):
    if not legal: raise ValueError('no quality/capacity/switch legal action')
    return max(legal,key=lambda i:q[i])

def safety(w,current,chosen):
    # A policy proposal cannot bypass the 5% margin or uncertainty guard.
    window=DecisionWindow(**w.get('window',{}))
    costs=(window.switching_costs or {}).get(current+'->'+chosen)
    if chosen!=current and (costs is None or
        objective_cost(w['predicted_cycles'][chosen]*(1+window.uncertainty_fraction),window,sum(costs.values()))
        >=.95*objective_cost(w['predicted_cycles'][current],window)):
        return current
    return chosen

def exhaustive_episode(episode):
    """Exact finite-horizon oracle over covered actions and stated switch costs."""
    if not episode: raise ValueError('nonempty episode required')
    initial='balanced/int8' if 'balanced/int8' in episode[0]['service_cycles'] else next(iter(episode[0]['service_cycles']))
    # State is (current action, minimum-residence count capped at one). Oracle
    # timings are used only in the environment score, never in policy state.
    frontier={(initial,1):0.}
    for w in episode:
        updated={}
        for (current,residence),accumulated in frontier.items():
            if current not in w['service_cycles']:
                current=next(iter(w['service_cycles'])); residence=1
            for index in mask(w,current,residence):
                chosen=ACTIONS[index]
                if chosen!=safety(w,current,chosen): continue
                next_residence=1 if chosen==current else 0
                key=(chosen,next_residence)
                total=accumulated+cost(w,current,chosen)
                updated[key]=min(updated.get(key,float('inf')),total)
        if not updated: raise ValueError('no legal exhaustive action for decision window')
        frontier=updated
    return min(frontier.values())

def train(episodes,seed=0,passes=20,parent=None):
    validate(episodes)
    if type(passes) is not int or not 1<=passes<=10000: raise ValueError('invalid pass count')
    rng=random.Random(seed); online=Network(rng); target=deepcopy(online); replay=deque(maxlen=10000); steps=0
    if parent:
        validate_checkpoint(parent)
        online.w1,online.w2=deepcopy(parent['w1']),deepcopy(parent['w2'])
        target.w1,target.w2=deepcopy(parent['target_w1']),deepcopy(parent['target_w2'])
        replay.extend(parent['replay'])
    for repeat in range(passes):
        for ep in episodes:
            current='balanced/int8' if 'balanced/int8' in ep[0]['service_cycles'] else next(iter(ep[0]['service_cycles'])); residence=1
            for index,w in enumerate(ep):
                if current not in w['service_cycles']: current=next(iter(w['service_cycles'])); residence=1
                x=state(w,current,residence); legal=mask(w,current,residence)
                action=rng.choice(legal) if rng.random()<max(.05,.6*(1-repeat/passes)) else argmax(online.forward(x)[1],legal)
                chosen=safety(w,current,ACTIONS[action]); action=ACTIONS.index(chosen)
                reward=-cost(w,current,chosen)/cost(w,current,current)
                terminal=index==len(ep)-1; next_w=ep[min(index+1,len(ep)-1)]
                next_current=chosen if chosen in next_w['service_cycles'] else next(iter(next_w['service_cycles']))
                nr=residence+1 if chosen==current else 0
                nx=state(next_w,next_current,nr); nm=mask(next_w,next_current,nr)
                replay.append([x,action,reward,nx,nm,terminal])
                if len(replay)>=8:
                    for sx,sa,sr,sn,sm,done in rng.sample(list(replay),8):
                        best=argmax(online.forward(sn)[1],sm)
                        online.fit(sx,sa,sr+(0 if done else .95*target.forward(sn)[1][best]))
                steps+=1
                if steps%25==0: target=deepcopy(online)
                current=chosen; residence=nr
    return dict(schema_version=2,observation_schema='frozen-predictor-no-oracle-v2',algorithm='DoubleDQN',status='candidate',seed=seed,passes=passes,
        actions=ACTIONS,state_size=SIZE,w1=online.w1,w2=online.w2,target_w1=target.w1,target_w2=target.w2,
        replay=list(replay),training_models=sorted({w['base_model_id'] for ep in episodes for w in ep}
            | set(parent['training_models'] if parent else [])),
        training_families=sorted({w['family'] for ep in episodes for w in ep}
            | set(parent['training_families'] if parent else [])),episodes_hash=identity(episodes),
        predictor_ids=sorted({w['predictor_id'] for ep in episodes for w in ep}
            | set(parent.get('predictor_ids',[]) if parent else [])),
        reward_provenance='estimated-window-from-measured-rtl',parent=identity(parent) if parent else None)

def validate_checkpoint(c):
    if c.get('schema_version')!=2 or c.get('observation_schema')!='frozen-predictor-no-oracle-v2' or c.get('algorithm')!='DoubleDQN' or c.get('actions')!=ACTIONS:
        raise ValueError('incompatible policy checkpoint')
    if c.get('state_size')!=SIZE or not c.get('training_models') or not c.get('training_families'):
        raise ValueError('policy state/lineage missing')
    for name,rows,cols in [('w1',24,SIZE),('w2',12,25),('target_w1',24,SIZE),('target_w2',12,25)]:
        if len(c.get(name,[]))!=rows or any(len(r)!=cols or any(not math.isfinite(v) for v in r) for r in c[name]):
            raise ValueError('invalid policy weights')
    if not isinstance(c.get('replay'),list) or len(c['replay'])>10000: raise ValueError('invalid replay storage')
    for sample in c['replay']:
        if not isinstance(sample,list) or len(sample)!=6: raise ValueError('invalid replay transition')
        x,action,reward,nx,legal,terminal=sample
        if any(not isinstance(v,list) or len(v)!=SIZE or any(type(n) not in (int,float)
            or not math.isfinite(n) for n in v) for v in (x,nx)):
            raise ValueError('invalid replay state')
        if type(action) is not int or not 0<=action<len(ACTIONS) or type(reward) not in (int,float) or not math.isfinite(reward):
            raise ValueError('invalid replay action/reward')
        if not isinstance(legal,list) or not legal or any(type(i) is not int or not 0<=i<len(ACTIONS) for i in legal) or type(terminal) is not bool:
            raise ValueError('invalid replay mask/terminal')

def evaluate(checkpoint,episodes,seeds=range(100,105),split='held-out',leave_family_out=False,budget=4):
    validate(episodes); validate_checkpoint(checkpoint); seeds=list(seeds)
    if len(set(seeds))<5: raise ValueError('five independent seeds required')
    if split not in ('validation','held-out'): raise ValueError('invalid evaluation split')
    if budget<1: raise ValueError('positive random-search budget required')
    ids={w['base_model_id'] for ep in episodes for w in ep}; families={w['family'] for ep in episodes for w in ep}
    if ids & set(checkpoint['training_models']): raise ValueError('base-model training/evaluation leakage')
    if leave_family_out and families & set(checkpoint['training_families']): raise ValueError('model-family leakage')
    net=Network(random.Random(0)); net.w1,net.w2=checkpoint['w1'],checkpoint['w2']
    if any(set(w.get('predictor_training_models',[])) & ids for ep in episodes for w in ep):
        raise ValueError('predictor base-model training/evaluation leakage')
    totals={k:[] for k in ('rl','fixed','heuristic','predictor','random','exhaustive')}; cases=[]
    for seed in seeds:
        rng=random.Random(seed); scores=dict.fromkeys(totals,0.)
        for episode_index,ep in enumerate(episodes):
            before=dict(scores)
            if 'balanced/int8' not in ep[0]['service_cycles']: raise ValueError('fixed balanced INT8 baseline required')
            scores['exhaustive']+=exhaustive_episode(ep)
            currents=dict.fromkeys(totals,'balanced/int8'); residence=dict.fromkeys(totals,1)
            for w in ep:
                for method in totals:
                    if method=='exhaustive': continue
                    cur=currents[method]
                    if cur not in w['service_cycles']: cur=next(iter(w['service_cycles'])); residence[method]=1
                    legal=mask(w,cur,residence[method]); actions=[ACTIONS[i] for i in legal]
                    if method=='rl': choice=ACTIONS[argmax(net.forward(state(w,cur,residence[method]))[1],legal)]
                    elif method=='fixed': choice=cur
                    elif method=='heuristic':
                        # Deterministic shape/intensity heuristic, not an oracle.
                        desired='compute/int8' if w.get('features',[0.]*6)[1]>=.2 or DecisionWindow(**w.get('window',{})).remaining_requests>=32 else 'compact/int8'
                        choice=desired if desired in actions else cur
                    elif method=='predictor':
                        window=DecisionWindow(**w.get('window',{}))
                        choice=min(actions,key=lambda a:objective_cost(w['predicted_cycles'][a],window,
                            sum((window.switching_costs or {}).get(cur+'->'+a,{}).values()) if a!=cur else 0))
                    else:
                        subset=rng.sample(actions,min(budget,len(actions))) if method=='random' else actions
                        choice=min(subset,key=lambda a:cost(w,cur,a))
                    if method!='random': choice=safety(w,cur,choice)
                    if method=='random':
                        window=DecisionWindow(**w.get('window',{}))
                        overhead=sum((window.switching_costs or {}).get(cur+'->'+choice,{}).values()) if choice!=cur else 0
                        scores[method]+=objective_cost(w['service_cycles'][choice],window,
                            overhead+sum(w['service_cycles'][a] for a in subset))
                    else: scores[method]+=cost(w,cur,choice)
                    currents[method]=choice
                    residence[method]=residence[method]+1 if choice==cur else 0
            case={'seed':seed,'episode_index':episode_index,'episode_hash':identity(ep),
                'base_models':sorted({w['base_model_id'] for w in ep}),
                'evidence_ids':sorted({key for w in ep for key in w['evidence_ids']}),
                'cost':{method:scores[method]-before[method] for method in scores},
                'windows':[w.get('window',{}) for w in ep],
                'switching_cost_provenance':'assumed scenarios; not physical reconfiguration measurements'}
            cases.append(case)
        for method in totals: totals[method].append(scores[method])
    return dict(schema_version=2,observation_schema=checkpoint['observation_schema'],split=split,seeds=seeds,base_models=sorted(ids),families=sorted(families),
        leave_family_out=leave_family_out,policy_id=identity(checkpoint),episode_hash=identity(episodes),
        totals=totals,mean_cost={k:sum(v)/len(v) for k,v in totals.items()},random_budget=budget,
        cases=cases,per_case_regressions=[c for c in cases if c['cost']['rl']>c['cost']['fixed'] or c['cost']['rl']>c['cost']['heuristic']],
        action_coverage=sorted({a for ep in episodes for w in ep for a in w['service_cycles']}),
        horizon_coverage=sorted({w.get('window',{}).get('remaining_requests',1) for ep in episodes for w in ep}),
        objective_coverage=sorted({w.get('window',{}).get('objective','latency') for ep in episodes for w in ep}),
        reward_provenance='estimated-window-from-measured-rtl',negative_results_included=True,
        random_exploration_cost='sum of tried service cycles charged as setup overhead',
        oracle_scope='covered quality-legal measured configurations only')

def promotion(checkpoint,report,previous=None):
    validate_checkpoint(checkpoint)
    if report.get('policy_id')!=identity(checkpoint) or report.get('split')!='held-out':
        raise ValueError('matching held-out evaluation required')
    if len(set(report.get('seeds',[])))<5 or set(report.get('base_models',[])) & set(checkpoint['training_models']):
        raise ValueError('invalid held-out lineage')
    score=report['mean_cost']
    if report.get('observation_schema')!='frozen-predictor-no-oracle-v2' or any(
        type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in score.values()):
        raise ValueError('finite non-leaking evaluation required')
    if score['rl']>score['fixed'] or score['rl']>score['heuristic']:
        raise ValueError('policy fails fixed/heuristic non-regression gate')
    return {'schema_version':1,'active_policy':identity(checkpoint),'previous_policy':previous,
            'evaluation_id':identity(report),'promotion':'explicit','rollback_available':previous is not None}
