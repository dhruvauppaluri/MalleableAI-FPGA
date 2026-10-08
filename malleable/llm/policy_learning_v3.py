"""Offline DQN diagnostic with observable costs and executed-action replay.

This does not accept old checkpoints or authorize an automatic switch.
Episodes contain externally frozen estimates, quality/capacity masks, measured
service outcomes and explicit transition scenarios. Assumed scenarios are
diagnostic only. The policy never observes measured outcomes in its state.
"""
import math
import random
from copy import deepcopy

import numpy as np

from .optimization import DecisionWindow, objective_cost
from ..records import identity

ACTIONS = ('compact', 'balanced', 'compute', 'buffered')
SCHEMA = 'observable-cost-dqn-v3'


def validate(episodes):
    if not episodes or any(not episode for episode in episodes):
        raise ValueError('nonempty episodes required')
    for episode in episodes:
        for step in episode:
            window = DecisionWindow(**step['window'])
            if (set(step['predicted']) != set(ACTIONS) or set(step['measured']) != set(ACTIONS)
                    or set(step['eligible']) - set(ACTIONS)
                    or not step.get('evidence_ids') or not step.get('predictor_id')):
                raise ValueError('complete frozen action/evidence observations required')
            for table in ('predicted','measured'):
                if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0
                       for v in step[table].values()):
                    raise ValueError('finite positive service values required')
            if window.switching_costs is None:
                raise ValueError('explicit transition scenario required')
            if any(a not in step['eligible'] for a in ACTIONS if a == 'balanced'):
                raise ValueError('balanced fallback must be eligible')


def _cost(step, current, chosen, *, measured=False):
    window = DecisionWindow(**step['window'])
    transition = (window.switching_costs or {}).get(current+'->'+chosen, {}) if current != chosen else {}
    return objective_cost(step['measured' if measured else 'predicted'][chosen],window,sum(transition.values()))


def legal(step,current,residence):
    window = DecisionWindow(**step['window'])
    costs = window.switching_costs or {}
    baseline = _cost(step,current,current)
    accepted = []
    reasons = {}
    for action in ACTIONS:
        if action not in step['eligible']:
            reason='quality-or-capacity'
        elif action == current:
            reason=None
        elif residence < 1:
            reason='minimum-residence'
        elif current+'->'+action not in costs:
            reason='missing-transition'
        elif _cost(step,current,action)* (1+window.uncertainty_fraction) >= baseline*.95:
            reason='conservative-gain'
        else:
            reason=None
        if reason is None: accepted.append(ACTIONS.index(action))
        else: reasons[action]=reason
    if ACTIONS.index(current) not in accepted:
        raise ValueError('current action must remain eligible')
    return accepted,reasons


def observation(step,current,residence):
    window=DecisionWindow(**step['window'])
    baseline=step['predicted'][current]
    costs=window.switching_costs or {}
    values=[math.log1p(window.remaining_requests),
            float(window.objective=='throughput'), math.log1p(window.arrival_interval_cycles),
            math.log1p(residence), math.log(baseline),window.uncertainty_fraction]
    values += [float(a==current) for a in ACTIONS]
    for action in ACTIONS:
        values.append(math.log(step['predicted'][action]/baseline))
        values.append(float(action in step['eligible']))
        transition=costs.get(current+'->'+action) if current!=action else {}
        values.append(math.log1p(sum(transition.values())/baseline) if transition is not None else -1.)
    return np.asarray(values,dtype=float)


def _network(rng,dim):
    return [rng.normal(0,.02,(24,dim)),np.zeros(24),rng.normal(0,.02,(len(ACTIONS),24)),np.zeros(len(ACTIONS))]


def _forward(net,x):
    h=np.tanh(net[0]@x+net[1]); return h,net[2]@h+net[3]


def _fit(net,x,action,target,rate):
    h,q=_forward(net,x); error=float(np.clip(q[action]-target,-1,1))
    old=net[2][action].copy()
    net[2][action]-=rate*error*h; net[3][action]-=rate*error
    delta=error*old*(1-h*h)
    net[0]-=rate*np.outer(delta,x); net[1]-=rate*delta


def train(episodes, *, seed, passes=20):
    validate(episodes)
    if type(seed) is not int or type(passes) is not int or not 1<=passes<=1000:
        raise ValueError('bounded independent training seed/passes required')
    rng=np.random.default_rng(seed); action_rng=random.Random(seed)
    dim=len(observation(episodes[0][0],'balanced',1)); online=_network(rng,dim)
    target=deepcopy(online); replay=[]; audit=[]; updates=0
    for repeat in range(passes):
        for episode in episodes:
            current,residence='balanced',1
            for index,step in enumerate(episode):
                available,reasons=legal(step,current,residence)
                state=observation(step,current,residence)
                _,q=_forward(online,state)
                proposed=action_rng.choice(range(len(ACTIONS))) if action_rng.random()<max(.05,.6*(1-repeat/passes)) else int(np.argmax(q))
                executed=proposed if proposed in available else ACTIONS.index(current)
                reason=None if executed==proposed else reasons[ACTIONS[proposed]]
                next_current=ACTIONS[executed]
                next_residence=residence+1 if executed==ACTIONS.index(current) else 0
                done=index==len(episode)-1
                next_step=step if done else episode[index+1]
                next_legal,_=legal(next_step,next_current,next_residence)
                next_state=observation(next_step,next_current,next_residence)
                reward=-_cost(step,current,next_current,measured=True)/_cost(step,current,current,measured=True)
                replay.append((state,executed,reward,next_state,next_legal,done))
                audit.append(dict(proposed=ACTIONS[proposed],executed=next_current,
                                  rejection_reason=reason, evidence_ids=step['evidence_ids']))
                if len(replay)>10000: replay.pop(0)
                if len(replay)>=8:
                    for sx,sa,sr,nx,nmask,terminal in action_rng.sample(replay,8):
                        _,future=_forward(online,nx)
                        best=max(nmask,key=lambda a:future[a])
                        target_value=sr if terminal else sr+.95*_forward(target,nx)[1][best]
                        _fit(online,sx,sa,target_value,.0005)
                    updates+=1
                    if updates%25==0: target=deepcopy(online)
                current,residence=next_current,next_residence
    return dict(schema_version=3,observation_schema=SCHEMA,status='diagnostic-candidate',
                automatic_switching_allowed=False, seed=seed,passes=passes,
                actions=list(ACTIONS),state_size=dim,weights=[v.tolist() for v in online],
                episodes_hash=identity(episodes),predictor_ids=sorted({s['predictor_id'] for ep in episodes for s in ep}),
                training_evidence_ids=sorted({i for ep in episodes for s in ep for i in s['evidence_ids']}),
                proposal_count=len(audit),rejection_count=sum(a['rejection_reason'] is not None for a in audit),
                executed_switch_count=sum(a['executed']!='balanced' for a in audit),
                audit=audit)


def evaluate(checkpoint, episodes, *, method='dqn'):
    validate(episodes)
    if method=='dqn' and (checkpoint.get('observation_schema')!=SCHEMA or checkpoint.get('schema_version')!=3):
        raise ValueError('legacy or incompatible checkpoint')
    network=[np.asarray(v,dtype=float) for v in checkpoint['weights']] if method=='dqn' else None
    results=[]
    for episode in episodes:
        current,residence='balanced',1; total=0; path=[]
        for step in episode:
            available,_=legal(step,current,residence)
            if method=='dqn':
                q=_forward(network,observation(step,current,residence))[1]
                chosen=ACTIONS[max(available,key=lambda a:q[a])]
            elif method=='deterministic':
                chosen=min((ACTIONS[i] for i in available),key=lambda a:_cost(step,current,a))
            elif method=='fixed': chosen='balanced'
            elif method=='predictor': chosen=min(ACTIONS,key=lambda a:step['predicted'][a])
            elif method=='oracle': chosen=min((ACTIONS[i] for i in available),key=lambda a:_cost(step,current,a,measured=True))
            else: raise ValueError('unknown policy')
            if ACTIONS.index(chosen) not in available: chosen=current
            total+=_cost(step,current,chosen,measured=True)
            path.append(chosen)
            residence=residence+1 if chosen==current else 0; current=chosen
        results.append(dict(total_cost=total,actions=path))
    return results
