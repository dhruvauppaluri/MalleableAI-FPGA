"""Legal actions, measured-data regression and overhead-aware decision policy."""
from dataclasses import asdict
import math
from .records import ExecutionConfig, PolicyDecision
from .model import analyze
from .backend import writes


def candidates():
    return [ExecutionConfig(lanes, active, reuse, dispatch)
            for lanes in (1, 2, 4, 8) for active in range(1, lanes+1)
            for reuse in (False, True) for dispatch in ('fifo', 'group')]


def features(model, config):
    outputs = sum(len(x['weights']) for x in model.layers)
    tiles = sum(len(x['weights'])*math.ceil(len(x['weights'][0])/config.active_lanes)
                for x in model.layers)
    return [1., float(outputs), float(tiles)]


class Predictor:
    """Online least-squares calibration of the RTL cycle formula, ridge regularized."""
    def __init__(self):
        self.samples = []
        # Current FSM: two clocks per tile, three per output. Observations
        # recalibrate this prior rather than assuming future backends match it.
        self.coefficients = [0., 3., 2.]
        self.rmse = 0.

    def update(self, model, config, cycles):
        self.samples.append((features(model, config), float(cycles)))
        # Three-dimensional normal equation, with a prior until enough evidence arrives.
        matrix = [[sum(x[i]*x[j] for x, _ in self.samples)+(1e-6 if i == j else 0)
                   for j in range(3)] + [sum(x[i]*y for x, y in self.samples)] for i in range(3)]
        for i in range(3):
            pivot = max(range(i, 3), key=lambda j: abs(matrix[j][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            divisor = matrix[i][i]
            matrix[i] = [v/divisor for v in matrix[i]]
            for j in range(3):
                if j != i:
                    factor = matrix[j][i]
                    matrix[j] = [a-factor*b for a, b in zip(matrix[j], matrix[i])]
        self.coefficients = [row[-1] for row in matrix]
        self.rmse = math.sqrt(sum((sum(a*b for a,b in zip(x,self.coefficients))-y)**2
                                  for x,y in self.samples)/len(self.samples))

    def predict(self, model, config):
        return max(1., sum(a*b for a,b in zip(features(model,config),self.coefficients)))


def cost(model, config, workload, hardware, profile, predictor, previous=None, resident_model_id=None):
    compute = predictor.predict(model, config)
    requests = workload.horizon
    upload = 2*len(writes(model))
    transfer = 2*(len(model.layers[0]['weights'][0])+len(model.layers[-1]['weights'])+1)
    cold = (previous is None or previous.lanes != config.lanes
            or resident_model_id != model.model_id)
    setup = upload * (requests if not config.reuse else int(cold))
    switch = 0
    if previous is not None and previous.lanes != config.lanes:
        if hardware.switch_cycles is None:
            return float('inf'), 0
        switch = hardware.switch_cycles
    total = requests*(compute+transfer)+setup+switch
    if profile == 'energy':
        watts = hardware.power_watts.get(str(config.lanes))
        if watts is None:
            raise ValueError('Energy optimization requires calibrated power for every candidate')
    if profile not in ('latency','energy','throughput'):
        raise ValueError('Unknown objective')
    # Serial-service queue with explicit arrivals; setup/switch delay the first request.
    finish, latency, raw_latency = float(switch), [], []
    for i in range(requests):
        arrival = (i*workload.interval_cycles if workload.arrival_pattern == 'sustained'
                   else (0 if workload.arrival_pattern == 'burst' or i == 0 else finish))
        loading = upload if not config.reuse or (cold and i == 0) else 0
        finish = max(finish, arrival)+compute+transfer+loading
        elapsed = finish-arrival
        raw_latency.append(elapsed)
        latency.append(elapsed + (max(0, elapsed-workload.deadline_cycles)*10
                                  if workload.deadline_cycles else 0))
    if profile == 'throughput':
        return finish/requests, switch/requests
    if profile == 'energy':
        if workload.deadline_cycles and max(raw_latency) > workload.deadline_cycles:
            return float('inf'), switch/hardware.clock_hz*watts
        return finish/hardware.clock_hz*watts/requests, switch/hardware.clock_hz*watts/requests
    return sum(latency)/requests, switch


def decide(model, workload, hardware, current, profile='latency', predictor=None, residence=1,
           resident_model_id=None):
    predictor = predictor or Predictor()
    available = candidates()
    scored = [(cost(model, c, workload, hardware, profile, predictor, current, resident_model_id), c) for c in available]
    same = min((x for x in scored if x[1].lanes == current.lanes), key=lambda x:x[0][0])
    best = min(scored, key=lambda x:x[0][0])
    if not math.isfinite(best[0][0]):
        raise ValueError('No feasible configuration for objective and constraints')
    uncertainty = max(2*predictor.rmse, .1*predictor.predict(model,current) if not predictor.samples else 0)
    # Convert cycle uncertainty to the objective's units.
    if profile == 'energy':
        uncertainty *= 1/hardware.clock_hz*max(hardware.power_watts.values())
    if best[1].lanes != current.lanes and (residence < 1 or
            same[0][0]-best[0][0] <= .05*same[0][0]+uncertainty):
        best = same
    chosen = best[1]
    action = 'keep' if chosen == current else ('tune-current' if chosen.lanes == current.lanes
                                               else 'switch-personality')
    return PolicyDecision(action, asdict(chosen),
        'Compared finite horizon including reload and explicit switching cost; 5% margin and residence guard',
        best[0][0], best[0][1], uncertainty)


def diagnose(counters, transfer_cycles=0, queue_cycles=0):
    total = max(1, counters['cycles']+transfer_cycles+queue_cycles)
    shares = {'controller/dependency': counters['overhead']/total,
              'compute activity': counters['compute']/total,
              'transfer/setup': transfer_cycles/total, 'queueing': queue_cycles/total}
    return dict(shares=shares, candidates=[k for k,v in shares.items() if v >= .25],
                provenance='estimated:attribution-from-RTL-counters-and-host-scenario',
                confidence='counter attribution; causal diagnosis requires controlled comparisons',
                external_memory='unavailable', resource_occupancy='unavailable')
