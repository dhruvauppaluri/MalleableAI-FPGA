"""Overhead-aware SSM personality search; no unsupported hardware actions."""
import math
from dataclasses import asdict
from .compiler import compile_token
from .backend import reference_trace,run_rtl
from ..quartus import Personality,automatic_build


def service_cycles(compiled,active):
    # Analytic instruction timing, cross-checked against measured RTL below.
    total=0
    for op,_,_,_,length,_,_,_ in compiled['program']:
        total+=1
        if op==255: break
        total+=math.ceil(length/active) if op==0 else length*(2 if op in (3,4) else 1)
    return total


def choose(candidates,current,horizon,switch_cost=None,margin=.05,residence=1,uncertainty=0,profile='latency',power=None,clock_hz=None):
    if type(horizon) is not int or horizon<1 or profile not in ('latency','throughput','energy'):
        raise ValueError('invalid horizon/profile')
    if not 0<=margin<1 or uncertainty<0 or residence<0:
        raise ValueError('invalid decision guard')
    if profile=='energy' and (not power or not clock_hz):
        raise ValueError('energy-first requires an explicit credible power source and clock')
    if profile=='energy' and (not math.isfinite(clock_hz) or clock_hz<=0 or
                             any(not math.isfinite(v) or v<=0 for v in power.values())):
        raise ValueError('clock and credible power values must be finite and positive')
    scored=[]
    for c in candidates:
        switching=c['lanes']!=current['lanes']
        if switching and switch_cost is None: continue
        recurring=0 if not switching else sum(switch_cost.values())+c['reload_cycles']
        elapsed=horizon*c['service_cycles']+recurring
        if profile=='energy':
            watts=power.get(str(c['lanes']))
            if watts is None: continue
            value=elapsed/clock_hz*watts/horizon
        elif profile=='throughput':
            value=elapsed/horizon
        else:
            # A sequential token stream: setup delays every remaining token.
            value=c['service_cycles']*(horizon+1)/2+recurring
        scored.append((value,c,recurring))
    same=[r for r in scored if r[1]['lanes']==current['lanes']]
    if not same: raise ValueError('no validated current-personality plan')
    best_same=min(same,key=lambda r:r[0]); best=min(scored,key=lambda r:r[0])
    if best[1]['lanes']!=current['lanes'] and (residence<1 or best_same[0]-best[0]<=margin*best_same[0]+uncertainty):
        best=best_same
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (switch_cost or {}).values()):
        raise ValueError('switch overheads must be finite nonnegative cycles')
    config={'lanes':best[1]['lanes'],'active_lanes':best[1]['active_lanes']}
    return dict(schema_version=1,action='keep' if config==current else ('tune-current' if config['lanes']==current['lanes'] else 'switch-personality'),
                config=config,previous_config=current,objective=profile,horizon=horizon,objective_cost=best[0],
                current_personality_cost=best_same[0],switch_and_reload_cycles=best[2],margin=margin,uncertainty=uncertainty,
                reason='best validated candidate within budget; finite horizon, reload, uncertainty and residence guard',
                idealized_switch_cost=bool(switch_cost is not None and sum(switch_cost.values())==0),
                overhead_scenario=switch_cost,development_build_cost_included=False)


def optimize(path,prompt='test',current=None,horizon=100,switch_cost=None,budget=5,profile='latency',
             clock_hz=None,power=None,store=None,quartus=True):
    current=current or {'lanes':4,'active_lanes':4}
    if budget<1: raise ValueError('budget must be positive')
    compiled=compile_token(path); records,_,_=reference_trace(compiled,prompt,0)
    # Full compiled lane count first, then runtime tuning candidates. Includes
    # current configuration regardless of budget and cannot select untested RTL.
    actions=[current]+[{'lanes':l,'active_lanes':l} for l in (1,2,4,8,16) if l!=current['lanes']]
    actions += [{'lanes':current['lanes'],'active_lanes':a} for a in range(1,current['lanes']+1) if a!=current['active_lanes']]
    measured=[]
    for action in actions[:budget]:
        evidence=run_rtl(compiled,records,**action)
        service=evidence['counters'][0]['cycles']
        predicted=service_cycles(compiled,action['active_lanes'])
        if service!=predicted:
            raise RuntimeError('timing predictor differs from RTL; candidate invalid')
        transfer=2*(compiled['manifest']['config']['width']+compiled['manifest']['config']['vocab_size'])
        row=dict(action,service_cycles=service+transfer,compute_cycles=service,
                 reload_cycles=2*(len(compiled['memory'])+8*len(compiled['program'])),
                 bit_exact=True,source='measured-rtl-plus-explicit-host-transfer-scenario',evidence=evidence)
        measured.append(row)
        if store: store.save('ssm-candidate',dict(row,model_id=compiled['model_id'],prompt=prompt))
    decision=choose(measured,current,horizon,switch_cost,profile=profile,power=power,clock_hz=clock_hz)
    decision['model_id']=compiled['model_id']; decision['candidate_count']=len(measured)
    decision['bottleneck']={'classification':'compute/controller mixed; use lane comparison evidence',
                            'external_memory':'unavailable','physical_dsp_occupancy':'unavailable',
                            'lane_comparison':[{'lanes':c['lanes'],'cycles':c['compute_cycles']} for c in measured]}
    decision['model_training_action']='none: hardware selection does not change model weights'
    # A different active count is only a register write, not a Quartus change.
    if quartus and decision['action']=='switch-personality':
        personality=Personality(lanes=decision['config']['lanes'],
                                memory_words=max(512,1<<(len(compiled['memory'])-1).bit_length()),
                                program_words=max(512,1<<(len(compiled['program'])*8-1).bit_length()))
        decision['quartus']=automatic_build(personality)
    if store: decision['record_id']=store.save('ssm-decision',decision)
    return decision
