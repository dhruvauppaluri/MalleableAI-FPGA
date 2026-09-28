"""Measured RTL experiments with explicit estimated host scheduling metrics."""
from dataclasses import asdict
import random
import math
import time
from .backend import RTLBackend
from .model import analyze
from .records import ExperimentResult, identity
from .optimize import diagnose


def metric(value, unit, source):
    return dict(value=value, unit=unit, provenance=source)


def percentile(values, fraction):
    return sorted(values)[max(0,min(len(values)-1,math.ceil(len(values)*fraction)-1))]


def dispatch(requests, mode='fifo'):
    """Group only already-arrived requests without deadlines; deadlines use EDF.

    Input dictionaries: id, model_id, arrival, service, deadline (absolute or None).
    Service includes upload for the caller's residency decision.
    """
    if mode not in ('fifo','group'):
        raise ValueError('Unknown dispatch mode')
    pending = list(requests)
    if len({x['id'] for x in pending}) != len(pending):
        raise ValueError('Duplicate request IDs')
    now, resident, results = 0, None, []
    while pending:
        now = max(now, min(x['arrival'] for x in pending))
        ready = [x for x in pending if x['arrival'] <= now]
        if any(x.get('deadline') is not None for x in ready):
            job = min(ready, key=lambda x:(x.get('deadline') if x.get('deadline') is not None
                                           else float('inf'), x['arrival']))
        elif mode == 'group':
            job = min(ready, key=lambda x:(x['model_id'] != resident, x['arrival']))
        else:
            job = min(ready, key=lambda x:x['arrival'])
        reload = job.get('reload_cycles',0) if resident != job['model_id'] else 0
        finish = now+job['service']+reload
        results.append(dict(id=job['id'], start=now, finish=finish,
                            deadline_missed=job.get('deadline') is not None and finish > job['deadline']))
        now, resident = finish, job['model_id']
        pending.remove(job)
    return results


def benchmark(model, workload, config, hardware, store, previous=None):
    store.save('model', model); store.save('workload', workload); store.save('hardware', hardware)
    rng = random.Random(workload.seed)
    inputs = [[rng.randint(-128,127) for _ in model.layers[0]['weights'][0]]
              for _ in range(workload.request_count)]
    input_id = store.save('inputs', inputs)
    try:
        run = RTLBackend().run(model, inputs, config)
        trace_id = store.save('trace', run)
        counters = run['counters']
        total_cycles = sum(x['cycles'] for x in counters)
        # Configuration writes and reads each consume two testbench clock slots.
        final_outputs = len(run['outputs'][-1])
        writes_count = counters[-1]['configuration_writes']
        read_count = counters[-1]['result_reads']+final_outputs
        transfer = 2*(writes_count+read_count)
        switch = 0
        if previous is not None and previous.lanes != config.lanes:
            if hardware.switch_cycles is None:
                raise ValueError('Switch scenario requires explicit switching overhead')
            switch = hardware.switch_cycles
        end, latencies, queues = switch, [], []
        previous_writes = 0
        for i, c in enumerate(counters):
            arrival = (i*workload.interval_cycles if workload.arrival_pattern == 'sustained'
                       else (0 if workload.arrival_pattern == 'burst' or i == 0 else end))
            queue = max(0, end-arrival)
            loading = 2*(c['configuration_writes']-previous_writes+len(run['outputs'][i]))
            end = max(end, arrival)+loading+c['cycles']
            previous_writes = c['configuration_writes']
            latencies.append(end-arrival); queues.append(queue)
        seconds = end/hardware.clock_hz
        power = hardware.power_watts.get(str(config.lanes))
        metrics = dict(
            rtl_cycles=metric(total_cycles,'cycles','measured:rtl'),
            throughput=metric(workload.request_count/seconds,'inferences/s','estimated:clock-and-arrivals'),
            latency={f'p{p}': metric(percentile(latencies,p/100)/hardware.clock_hz,'s',
                                     'estimated:clock-and-arrivals') for p in (50,95,99)},
            mac_utilization=metric(sum(c['macs'] for c in counters)/(total_cycles*config.lanes),
                                   'fraction','measured:rtl'),
            interface_bytes=metric(writes_count*4+read_count,'bytes','measured:rtl-interface'),
            effective_bandwidth=metric((writes_count*4+read_count)/seconds,'bytes/s','estimated:clock'),
            energy_per_inference=metric(power*seconds/workload.request_count if power else None,
                                        'J','estimated:calibrated-power' if power else 'unavailable'),
            throughput_per_watt=metric(workload.request_count/seconds/power if power else None,
                                      'inferences/J','estimated:calibrated-power' if power else 'unavailable'),
            monetary_cost=metric(seconds*hardware.dollars_per_second if hardware.dollars_per_second is not None
                                 else None,'USD','estimated:rate' if hardware.dollars_per_second is not None else 'unavailable'),
            resource_occupancy=metric(None,'fraction','unavailable'),
            scaling_efficiency=metric(None,'fraction','unavailable:requires-one-lane-comparison'),
            quality=metric(True,'bit-exact','measured:independent-reference'),
            task_accuracy=metric(None,'fraction','unavailable:no-labels'),
            deadline_misses=metric(sum(x > workload.deadline_cycles for x in latencies)
                                   if workload.deadline_cycles else 0,'requests','estimated:clock-and-arrivals'),
            breakdown=dict(execution_cycles=metric(total_cycles,'cycles','measured:rtl'),
                           transfer_cycles=metric(transfer,'cycles','estimated:host-protocol'),
                           queue_cycles=metric(sum(queues),'cycles','estimated:arrivals'),
                           switch_cycles=metric(switch,'cycles','estimated:explicit-scenario')),
            bottleneck=diagnose(dict(cycles=total_cycles, compute=sum(c['compute'] for c in counters),
                                    overhead=sum(c['overhead'] for c in counters)), transfer,sum(queues)))
        result = ExperimentResult(model.model_id,identity(workload),asdict(config),metrics,
             dict(seed=workload.seed, inputs=input_id, trace=trace_id, build_hash=run['build_hash'],
                  tool=run['tool'], clock_hz_assumption=hardware.clock_hz,
                  build_seconds=run['build_seconds'], host_simulation_seconds=run['simulation_seconds'],
                  switch_scenario=hardware.switch_cycles, idealized_switch=hardware.switch_cycles == 0,
                  initial_residency='cold:each-backend-run',hardware_id=identity(hardware)),
             True, previous_config=asdict(previous) if previous else None)
    except Exception as exc:
        failure_trace = store.save('failure',dict(type=type(exc).__name__,message=str(exc),
            stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None)))
        result = ExperimentResult(model.model_id,identity(workload),asdict(config),{},
                                  dict(seed=workload.seed,inputs=input_id,hardware_id=identity(hardware),
                                       failure_trace=failure_trace),False,str(exc),
                                  previous_config=asdict(previous) if previous else None)
    store.save('experiment',result)
    return result


def suite(hardware, store, seed=0, exhaustive=False):
    """Fixed inputs per scenario, baseline-relative arrival rates, all model sizes."""
    from .model import fixture
    from .records import WorkloadSpec, ExecutionConfig
    from .optimize import candidates
    results = []
    configs = candidates() if exhaustive else [ExecutionConfig(n,n) for n in (1,2,4,8)]
    for size in ('light','medium','heavy'):
        model = fixture(size,seed)
        baseline = benchmark(model,WorkloadSpec(request_count=1,seed=seed),ExecutionConfig(1,1),hardware,store)
        if not baseline.valid:
            raise RuntimeError(baseline.failure)
        cycles = baseline.metrics['rtl_cycles']['value']
        for pattern,ratio in [('isolated',1),('sustained',2),('sustained',1),('sustained',.5),('burst',0)]:
            workload = WorkloadSpec(request_count=4,arrival_pattern=pattern,
                                    interval_cycles=int(cycles*ratio),seed=seed)
            for config in configs:
                result = benchmark(model,workload,config,hardware,store)
                if result.valid:
                    result.metrics['scaling_efficiency'] = metric(
                        cycles/(result.metrics['rtl_cycles']['value']/workload.request_count)/config.lanes,
                        'fraction','measured:rtl-one-lane-baseline')
                    store.save('experiment',result)
                results.append(identity(result))
    return store.save('suite',dict(schema_version=1,seed=seed,results=results,
                                   exhaustive=exhaustive,arrival_reference='one-lane-RTL-service-cycles'))


def benchmark_sequence(models, requests, config, store):
    """Dispatch a mixed-model stream, then verify that exact order on RTL.

    Scheduling time is estimated; the RTL executes dispatched requests serially.
    Each request supplies id, model_id, arrival, input, optional deadline.
    """
    from .optimize import Predictor
    from .backend import writes
    predictor = Predictor()
    jobs = []
    for request in requests:
        model = models[request['model_id']]
        store.save('model',model)
        upload = 2*len(writes(model))
        jobs.append(dict(request,service=predictor.predict(model,config)+2*(
            len(request['input'])+len(model.layers[-1]['weights'])+1)+(0 if config.reuse else upload),
            reload_cycles=upload if config.reuse else 0))
    ordered = dispatch(jobs,config.dispatch)
    by_id = {r['id']:r for r in requests}
    sequence = [by_id[r['id']] for r in ordered]
    run = RTLBackend().run(models[sequence[0]['model_id']],
        [r['input'] for r in sequence],config,[models[r['model_id']] for r in sequence])
    record = dict(schema_version=1,config=asdict(config),requests=requests,
        schedule=ordered,schedule_provenance='estimated:serial-dispatch',
        trace=store.save('trace',run),outputs={str(r['id']):out for r,out in zip(sequence,run['outputs'])},
        valid=True)
    return store.save('sequence',record)


def optimize_run(model, workload, hardware, current, store, profile='latency', budget=8):
    """Bounded experiments followed by a validated, overhead-aware selection."""
    from .optimize import candidates, Predictor, cost
    from .records import integer, PolicyDecision
    integer(budget,1,60)
    started = time.perf_counter()
    predictor = Predictor()
    ranked = sorted(candidates(),key=lambda c:cost(model,c,workload,hardware,profile,predictor,current)[0])
    choices = [current]+[c for c in ranked if c != current][:budget-1]
    evaluated = []
    for candidate in choices:
        if not math.isfinite(cost(model,candidate,workload,hardware,profile,predictor,current)[0]):
            continue
        result = benchmark(model,workload,candidate,hardware,store,previous=current)
        if not result.valid:
            continue
        cycles = result.metrics['rtl_cycles']['value']/workload.request_count
        measured = Predictor()
        measured.coefficients = [cycles,0.,0.]
        objective,switch = cost(model,candidate,workload,hardware,profile,measured,current)
        predictor.update(model,candidate,cycles)
        evaluated.append((objective,candidate,switch,identity(result)))
    if not evaluated:
        raise ValueError('No validated feasible candidate found')
    same = min((x for x in evaluated if x[1].lanes == current.lanes),key=lambda x:x[0],default=None)
    if same is None:
        raise ValueError('Current personality failed validation; refusing automatic switch')
    best = min(evaluated,key=lambda x:x[0])
    uncertainty = 2*predictor.rmse
    if profile == 'energy':
        uncertainty *= max(hardware.power_watts.values())/hardware.clock_hz
    if best[1].lanes != current.lanes and same[0]-best[0] <= .05*same[0]+uncertainty:
        best = same
    action = 'keep' if best[1] == current else ('tune-current' if best[1].lanes == current.lanes else 'switch-personality')
    decision = PolicyDecision(action,asdict(best[1]),
        'Best bit-exact validated candidate within budget; RTL cycles plus explicit host/switch scenario',
        best[0],best[2],uncertainty)
    decision_id = store.save('decision',decision)
    report = dict(schema_version=1,decision=decision_id,budget=budget,model=model.model_id,
                  workload=identity(workload),experiments=[x[3] for x in evaluated],
                  selected_experiment=best[3],predictor_rmse=predictor.rmse,
                  search_seconds=metric(time.perf_counter()-started,'s','measured:host-wall-clock'),
                  globally_optimal=False)
    return store.save('optimization',report)
