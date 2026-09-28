"""Small dependency-free Double DQN for reproducible sequential experiments.

Two-layer tanh network, masked online argmax, separate target evaluation,
experience replay and periodic target updates. Rewards are explicitly estimated.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import math
import random
from .model import fixture, analyze
from .records import ExecutionConfig, WorkloadSpec, identity
from .optimize import candidates, Predictor, cost, features

ACTIONS = candidates()
STATE_SIZE = 13


def state(model, workload, current, profile, predictor, recent=None):
    info = analyze(model)
    return [1., info['macs']/16384, info['parameter_bytes']/20000,
            current.lanes/8, current.active_lanes/8, workload.horizon/32,
            workload.interval_cycles/10000, predictor.predict(model,current)/50000,
            {'latency':0.,'throughput':.5,'energy':1.}[profile],
            *((recent or [0.,0.,0.,0.]))]


def counter_state(model, config, cycles):
    tiles = features(model,config)[2]
    return [cycles/50000,tiles/16384,analyze(model)['macs']/max(1,cycles*config.lanes),
            max(0.,1-tiles/max(1,cycles))]


class Network:
    def __init__(self, rng, hidden=16):
        self.w1 = [[rng.uniform(-.1,.1) for _ in range(STATE_SIZE)] for _ in range(hidden)]
        self.w2 = [[rng.uniform(-.1,.1) for _ in range(hidden+1)] for _ in ACTIONS]

    def forward(self, x):
        hidden = [math.tanh(sum(a*b for a,b in zip(row,x))) for row in self.w1]+[1.]
        return hidden, [sum(a*b for a,b in zip(row,hidden)) for row in self.w2]

    def fit(self, x, action, target, rate=.001):
        hidden, q = self.forward(x)
        error = max(-1., min(1., q[action]-target)) # Huber gradient
        old = self.w2[action][:]
        for j in range(len(hidden)):
            self.w2[action][j] -= rate*error*hidden[j]
        for j,row in enumerate(self.w1):
            for k in range(STATE_SIZE):
                row[k] -= rate*error*old[j]*(1-hidden[j]**2)*x[k]


def legal(hardware, current, profile):
    return [i for i,c in enumerate(ACTIONS)
            if (c.lanes == current.lanes or hardware.switch_cycles is not None)
            and (profile != 'energy' or str(c.lanes) in hardware.power_watts)]


def argmax(values, mask):
    if not mask:
        raise ValueError('No legal actions')
    return max(mask,key=lambda i:values[i])


def episode(seed):
    rng = random.Random(seed)
    models = [fixture(size,seed*10+i) for i,size in enumerate(['light','medium','heavy'])]
    for window in range(6):
        # Repeated and alternating models; artifact seeds separate data splits.
        model = models[rng.randrange(3)]
        yield model, WorkloadSpec(horizon=rng.choice([1,8,32]),
                                 arrival_pattern=rng.choice(['isolated','sustained','burst']),
                                 interval_cycles=rng.choice([100,1000,10000]),seed=seed)


def train(hardware, store, episodes=20, seed=0, profile='latency', parent=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if profile == 'energy' and len(hardware.power_watts) < 4:
        raise ValueError('Energy learning requires four calibrated personality powers')
    rng = random.Random(seed)
    online = Network(rng); target = deepcopy(online)
    replay = deque(maxlen=10000)
    predictor = Predictor()
    observations = store.records('experiment')
    steps = 0
    train_ids = []
    if parent is not None:
        validate_checkpoint(parent,hardware)
        if profile != parent['profile']:
            raise ValueError('Cannot resume a different objective')
        online.w1, online.w2 = deepcopy(parent['w1']), deepcopy(parent['w2'])
        target.w1, target.w2 = deepcopy(parent['target_w1']), deepcopy(parent['target_w2'])
        replay.extend(parent['replay'])
        train_ids.extend(parent['training_models'])
        predictor.coefficients = parent['predictor_coefficients']
        predictor.rmse = parent['predictor_rmse']
    for e in range(episodes):
        windows = list(episode(seed+e))
        current = ExecutionConfig()
        resident = None
        recent = None
        for t,(model,workload) in enumerate(windows):
            store.save('model',model)
            store.save('workload',workload)
            train_ids.append(model.model_id)
            for result in observations:
                if result['valid'] and result['model_id'] == model.model_id:
                    trace = store.load(result['provenance']['trace'])
                    predictor.update(model,ExecutionConfig(**result['config']),trace['counters'][0]['cycles'])
            x = state(model,workload,current,profile,predictor,recent)
            mask = legal(hardware,current,profile)
            action = rng.choice(mask) if rng.random()<max(.05,.6*(1-e/max(1,episodes))) else argmax(online.forward(x)[1],mask)
            chosen = ACTIONS[action]
            baseline = cost(model,current,workload,hardware,profile,predictor,current,resident)[0]
            measured = cost(model,chosen,workload,hardware,profile,predictor,current,resident)[0]
            reward = -measured/max(baseline,1e-12)
            terminal = t == len(windows)-1
            next_model,next_workload = windows[min(t+1,len(windows)-1)]
            recent = counter_state(model,chosen,predictor.predict(model,chosen))
            nx = state(next_model,next_workload,chosen,profile,predictor,recent)
            next_mask = legal(hardware,chosen,profile)
            replay.append((x,action,reward,nx,next_mask,terminal))
            if len(replay)>=8:
                for sx,sa,sr,sn,sm,terminal_sample in rng.sample(list(replay),8):
                    best = argmax(online.forward(sn)[1],sm)
                    boot = 0 if terminal_sample else .95*target.forward(sn)[1][best]
                    online.fit(sx,sa,sr+boot)
            steps += 1
            if steps%25 == 0:
                target = deepcopy(online)
            current = chosen
            resident = model.model_id
    checkpoint = dict(schema_version=1,algorithm='DoubleDQN',reward_provenance='estimated:cycle-model',
                      generator_version='dense-fixture-v1',state_counter_provenance='estimated:cycle-model',
                      seed=seed,episodes=episodes,profile=profile,hardware=asdict(hardware),
                      actions=[asdict(c) for c in ACTIONS],w1=online.w1,w2=online.w2,
                      target_w1=target.w1,target_w2=target.w2,replay=list(replay),
                      predictor_coefficients=predictor.coefficients,predictor_rmse=predictor.rmse,
                      training_models=sorted(set(train_ids)),status='candidate',
                      parent=identity(parent) if parent else None)
    return store.save('checkpoint',checkpoint)


def validate_checkpoint(checkpoint, hardware):
    if (checkpoint.get('schema_version') != 1 or checkpoint.get('algorithm') != 'DoubleDQN'
            or identity(checkpoint['hardware']) != identity(hardware)
            or checkpoint['actions'] != [asdict(c) for c in ACTIONS]
            or any(len(row) != STATE_SIZE for row in checkpoint['w1'])):
        raise ValueError('Incompatible checkpoint or hardware scenario')


def evaluate(checkpoint, hardware, store, seeds=range(100000,100005), split='held-out', rtl=False):
    from .optimize import decide
    from .experiment import benchmark
    validate_checkpoint(checkpoint,hardware)
    seeds = list(seeds)
    if len(set(seeds)) < 5:
        raise ValueError('Evaluation requires at least five independent seeds')
    if split not in ('validation','held-out'):
        raise ValueError('Unknown evaluation split')
    network = Network(random.Random(0))
    network.w1,network.w2 = checkpoint['w1'],checkpoint['w2']
    profile = checkpoint['profile']
    totals = {k:[] for k in ['rl','fixed','heuristic','random','exhaustive']}
    models = []
    evidence = []
    decisions = []
    rtl_totals = {k:[] for k in totals}
    for seed in seeds:
        rng = random.Random(seed)
        current = {k:ExecutionConfig() for k in totals}
        scores = dict.fromkeys(totals,0.)
        rtl_scores = dict.fromkeys(totals,0.)
        resident = dict.fromkeys(totals)
        recent = dict.fromkeys(totals)
        for model,workload in episode(seed):
            if model.model_id in checkpoint['training_models']:
                raise ValueError('Training/evaluation model leakage')
            store.save('model',model)
            store.save('workload',workload)
            models.append(model.model_id)
            predictor = Predictor()
            predictor.coefficients = checkpoint['predictor_coefficients']
            predictor.rmse = checkpoint['predictor_rmse']
            for method in totals:
                before = current[method]
                mask = legal(hardware,before,profile)
                if method == 'rl':
                    chosen = ACTIONS[argmax(network.forward(state(model,workload,before,profile,predictor,recent[method]))[1],mask)]
                elif method == 'fixed':
                    chosen = before
                elif method == 'heuristic':
                    chosen = ExecutionConfig(**decide(model,workload,hardware,before,profile,predictor,
                                                      resident_model_id=resident[method]).config)
                elif method == 'random':
                    chosen = ACTIONS[rng.choice(mask)]
                else:
                    chosen = min((ACTIONS[i] for i in mask),key=lambda c:cost(model,c,workload,hardware,profile,predictor,before,resident[method])[0])
                # A deployed learned proposal still obeys the switching safety guard.
                if method == 'rl' and chosen.lanes != before.lanes:
                    same = min((c for c in ACTIONS if c.lanes == before.lanes),
                               key=lambda c:cost(model,c,workload,hardware,profile,predictor,before,resident[method])[0])
                    if cost(model,chosen,workload,hardware,profile,predictor,before,resident[method])[0] >= .95*cost(model,same,workload,hardware,profile,predictor,before,resident[method])[0]:
                        chosen = same
                value = cost(model,chosen,workload,hardware,profile,predictor,before,resident[method])[0]
                scores[method] += value
                decisions.append(dict(seed=seed,method=method,model=model.model_id,
                                      workload=asdict(workload),before=asdict(before),
                                      selected=asdict(chosen),estimated_cost=value))
                recent[method] = counter_state(model,chosen,predictor.predict(model,chosen))
                if rtl and method in ('rl','fixed','heuristic','random','exhaustive'):
                    measured = benchmark(model,WorkloadSpec(request_count=1,seed=seed),chosen,hardware,store)
                    measured.policy_version = identity(checkpoint) if method == 'rl' else method
                    store.save('experiment',measured)
                    evidence.append(dict(method=method,model_id=model.model_id,
                                         experiment=identity(measured),valid=measured.valid))
                    if measured.valid:
                        actual = measured.metrics['rtl_cycles']['value']
                        recent[method] = counter_state(model,chosen,actual)
                        calibrated = Predictor()
                        calibrated.coefficients = [actual,0.,0.]
                        observed_cost = cost(model,chosen,workload,hardware,profile,calibrated,before,resident[method])[0]
                        rtl_scores[method] += observed_cost
                        decisions[-1].update(rtl_calibrated_cost=observed_cost,
                            cycle_prediction_error=predictor.predict(model,chosen)-actual)
                current[method] = chosen
                resident[method] = model.model_id
        for key in totals:
            totals[key].append(scores[key])
            rtl_totals[key].append(rtl_scores[key])
    report = dict(schema_version=1,provenance='estimated:held-out-sequential-simulator',
                  checkpoint=identity(checkpoint),split=split,rtl_evidence=evidence,
                  totals=totals,models=sorted(set(models)),seeds=seeds,decisions=decisions,
                  mean={k:sum(v)/len(v) for k,v in totals.items()},
                  rtl_calibrated_totals=rtl_totals if rtl else None,
                  search_budget=dict(rl=1,random=1,heuristic=len(ACTIONS),exhaustive=len(ACTIONS)),
                  eligible_for_promotion=bool(evidence) and all(x['valid'] for x in evidence)
                      and sum(totals['rl']) <= min(sum(totals['fixed']),sum(totals['heuristic']))
                      and sum(rtl_totals['rl']) <= min(sum(rtl_totals['fixed']),sum(rtl_totals['heuristic'])),
                  reason='Promotion also requires disjoint validation and held-out reports; RTL evidence validates outputs, not physical performance')
    store.save('evaluation',report)
    return report


def promote(store, checkpoint_id, validation_id, heldout_id):
    checkpoint = store.load(checkpoint_id)
    validation, heldout = store.load(validation_id), store.load(heldout_id)
    for report,split in [(validation,'validation'),(heldout,'held-out')]:
        if (report['checkpoint'] != checkpoint_id or report['split'] != split
                or not report['eligible_for_promotion']):
            raise ValueError('Policy failed promotion gates')
        if len(set(report['seeds'])) < 5 or not report['rtl_evidence']:
            raise ValueError('Insufficient independent evaluation evidence')
        for evidence in report['rtl_evidence']:
            experiment = store.load(evidence['experiment'])
            if not evidence['valid'] or not experiment['valid']:
                raise ValueError('Failed RTL experiment in promotion evidence')
    if set(validation['models']) & set(heldout['models']):
        raise ValueError('Validation/held-out model leakage')
    if (set(validation['models']) | set(heldout['models'])) & set(checkpoint['training_models']):
        raise ValueError('Training/evaluation model leakage')
    deployments = store.records('deployment')
    previous = deployments[-1]['checkpoint'] if deployments else None
    return store.save('deployment',dict(checkpoint=checkpoint_id,previous=previous,
        validation=validation_id,heldout=heldout_id,status='promoted-simulation-policy'))


def rollback(store):
    deployments = store.records('deployment')
    if not deployments or not deployments[-1]['previous']:
        raise ValueError('No previous deployed checkpoint')
    last = deployments[-1]
    return store.save('deployment',dict(checkpoint=last['previous'],previous=last['checkpoint'],
                                       status='rollback'))
