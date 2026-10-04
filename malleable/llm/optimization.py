"""Legal, quality-constrained decisions using measured simulation evidence."""
from dataclasses import dataclass,asdict
import math
import statistics
from ..records import identity
from .quality import gate,quality_matches

PREDICTOR_FEATURES = ('bias', 'input_tokens_div_2048', 'generated_tokens_div_256',
    'matrix_columns_div_8', 'vector_lanes_div_16', 'axi_latency_div_100',
    'axi_stall_percent_div_100', 'dram_gib', 'weight_format', 'fifo_depth_div_1024',
    'axi_bandwidth_percent_div_100')
PREDICTOR_TRAINING_MODELS = {
    '7ab1181d3a2b04ce889880dfc3b94933574441e9c221e950622c39a3ce79a59d',
    '96c8847b78593ff15008955c68e384f5ff40895e425e487aec30b8173542a857'}
PREDICTOR_HELD_OUT_MODEL = '8d6ac525c1b135ad360d0f9f5a822698bbb257b7408f791e2ca695ae473798b3'

@dataclass(frozen=True)
class DecisionWindow:
    remaining_requests:int=1
    arrival_interval_cycles:float=0
    objective:str='latency'
    residence_windows:int=1
    switching_costs:dict|None=None
    uncertainty_fraction:float=.1
    schema_version:int=1
    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1: raise ValueError('unsupported decision schema')
        if self.objective not in ('latency','throughput'): raise ValueError('energy requires a credible calibrated source; unsupported here')
        if type(self.remaining_requests) is not int or self.remaining_requests<1: raise ValueError('positive horizon required')
        if type(self.residence_windows) is not int or self.residence_windows<0: raise ValueError('invalid residence')
        if type(self.uncertainty_fraction) not in (int,float) or not 0<=self.uncertainty_fraction<1: raise ValueError('invalid uncertainty')
        if type(self.arrival_interval_cycles) not in (int,float) or not math.isfinite(self.arrival_interval_cycles) or self.arrival_interval_cycles<0: raise ValueError('invalid arrivals')
        if self.switching_costs is not None:
            if not isinstance(self.switching_costs,dict): raise ValueError('switching scenarios must be an object')
            for costs in self.switching_costs.values():
                if not isinstance(costs,dict) or set(costs)!={'drain','program','reload','warmup','reprefill'}:
                    raise ValueError('explicit drain/program/reload/warmup/reprefill cycle costs required')
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs.values()):
                    raise ValueError('invalid switching cost')

def config_key(result): return result['personality']+'/'+result['workload']['wformat']

def objective_cost(service,window,overhead=0):
    n=window.remaining_requests; interval=window.arrival_interval_cycles
    if not math.isfinite(service) or service<=0 or not math.isfinite(overhead) or overhead<0:
        raise ValueError('finite positive service and nonnegative overhead required')
    # Exact deterministic single-server queue formula; avoids an unbounded
    # Python loop for long deployment horizons.
    if service>=interval:
        latency=overhead+service+(n-1)*(service-interval)/2
        end=overhead+n*service
    else:
        delta=interval-service; k=min(n,math.ceil(overhead/delta))
        latency=service+(k*overhead-delta*k*(k-1)/2)/n
        end=max(overhead+n*service,(n-1)*interval+service)
    return latency if window.objective=='latency' else end/n

def validated_variants(qualities,base_model_id,tokenizer_id=None,context=128):
    valid=set()
    for q in qualities:
        if q.get('base_model_id')!=base_model_id or q.get('split')!='validation' \
            or q.get('samples',0)<1024 or q.get('target_count',0)<1024 or q.get('suite_frozen') is not True: continue
        if q.get('context')!=context or not q.get('configuration_id') or not q.get('tokenizer_id') \
            or not q.get('suite_file_hash') or set(q.get('suite_hashes',{}))!={'calibration','validation','held-out'} \
            or (tokenizer_id is not None and q['tokenizer_id']!=tokenizer_id): continue
        if not gate(q['float_nll'],q['candidate_nll'],q['agreement'])['passed']: continue
        # Validation makes a candidate searchable. Held-out is reserved for promotion.
        valid.add((q['variant_id'],q.get('personality')))
    return valid

def observations(results,base_model_id,workload_id=None):
    grouped={}
    for r in results:
        if r.get('base_model_id')!=base_model_id or not r.get('valid') or r.get('backend')!='rtl': continue
        if workload_id is not None and r['workload_id']!=workload_id: continue
        counters=r.get('counters',[])
        if not counters or any(s.get('validation')!='bit-exact-dram-and-tmem' for s in counters): continue
        service=sum(s['cycles'] for s in counters)
        if service<=0: continue
        grouped.setdefault(config_key(r),[]).append((service,r))
    return grouped

def decide(results,qualities,current,window):
    base=current['base_model_id']
    reference=next((q for q in reversed(qualities) if quality_matches(q,current)),None)
    # Match prompt/context/token count/seed/memory scenario, ignoring only execution settings.
    def comparison(r):
        return identity({k:v for k,v in r['workload'].items() if k not in ('personality','wformat','backend','clock_hz')})
    matches=[r for r in results if r.get('workload') and comparison(r)==comparison(current)
        and r.get('tokenizer_id')==current.get('tokenizer_id') and r.get('input_token_hash')==current.get('input_token_hash')]
    groups=observations(matches,base); cur=config_key(current)
    if cur not in groups: raise ValueError('current configuration requires measured RTL correctness evidence')
    candidates={}; excluded={}
    for key,samples in groups.items():
        if not reference or not all(any(quality_matches(q,r,frozen_reference=reference) for q in qualities) for _,r in samples):
            excluded[key]='missing or failed matched frozen validation quality evidence'; continue
        change=key!=cur
        costs=(window.switching_costs or {}).get(cur+'->'+key)
        if change and window.residence_windows<1: excluded[key]='minimum residence'; continue
        if change and costs is None: excluded[key]='explicit switching/reload scenario missing'; continue
        service=statistics.median(s[0] for s in samples)
        overhead=sum(costs.values()) if change else 0
        spread=max((abs(s[0]-service) for s in samples),default=0)
        uncertainty=max(spread,service*window.uncertainty_fraction)
        candidates[key]={'service_cycles':service,'overhead_cycles':overhead,
            'objective_cost':objective_cost(service,window,overhead),
            'conservative_cost':objective_cost(service+uncertainty,window,overhead),
            'provenance':'estimated-window-from-measured-rtl','sample_count':len(samples),
            'idealized_zero_switch':bool(change and overhead==0)}
    base_cost=objective_cost(statistics.median(s[0] for s in groups[cur]),window)
    chosen=cur; reason='retain current configuration'
    if candidates:
        best=min(candidates,key=lambda k:candidates[k]['conservative_cost'])
        if best!=cur and candidates[best]['conservative_cost']<.95*base_cost:
            chosen=best; reason='improvement repays explicit overhead, uncertainty and 5% margin'
        elif best!=cur: reason='improvement does not repay overhead plus uncertainty and margin'
    return dict(schema_version=1,base_model_id=base,current=cur,chosen=chosen,
        action='keep' if chosen==cur else 'tune-current' if chosen.split('/')[0]==cur.split('/')[0] else 'switch-personality',
        reason=reason,window=asdict(window),candidates=candidates,excluded=excluded,
        baseline_cost=base_cost,quality_constraint='validation-search; held-out promotion required',
        needs_reset_and_reprefill=chosen!=cur,physical_switch=False)

def diagnose(profiles):
    """Use recorded Lens counters; uncertain/mixed diagnoses are deliberate."""
    diagnoses=[]
    for p in profiles:
        bound=p.get('roofline',{}).get('bound','uncertain')
        diagnoses.append({'name':p.get('name'),'classification':bound,
            'evidence':p.get('mxu_gaps',{}),'provenance':'rtl-trace-derived',
            'external_memory':'generic AXI scenario, not measured physical memory',
            'capacity':'simulated allocation only','latency':'outcome, not a hardware bottleneck category'})
    return diagnoses

class Predictor:
    """Ridge regression fitted only from selected training models' measured rows."""
    def __init__(self): self.coefficients=None; self.rmse=None; self.training_models=[]
    @classmethod
    def from_checkpoint(cls,data):
        if data.get('schema_version')!=2 or data.get('algorithm')!='ridge-regression' \
            or data.get('feature_schema')!=list(PREDICTOR_FEATURES):
            raise ValueError('predictor schema v2 with the ordered feature definition required; v1 is inspectable only')
        coefficients=data.get('coefficients')
        if not isinstance(coefficients,list) or len(coefficients)!=len(PREDICTOR_FEATURES) or any(type(v) not in (int,float)
            or not math.isfinite(v) for v in coefficients): raise ValueError('invalid predictor coefficients')
        rmse=data.get('training_rmse_cycles')
        if type(rmse) not in (int,float) or not math.isfinite(rmse) or rmse<0 or not data.get('training_models'):
            raise ValueError('invalid predictor uncertainty/lineage')
        if set(data['training_models'])!=PREDICTOR_TRAINING_MODELS or data.get('held_out_model')!=PREDICTOR_HELD_OUT_MODEL:
            raise ValueError('predictor checkpoint frozen partition mismatch')
        value=cls(); value.coefficients=coefficients; value.rmse=rmse; value.training_models=data['training_models']; return value
    @staticmethod
    def inspect_checkpoint(data):
        return {'schema_version':data.get('schema_version'),'algorithm':data.get('algorithm'),
            'feature_schema':data.get('feature_schema'),'training_models':data.get('training_models'),
            'coefficients':data.get('coefficients'),'runnable':data.get('schema_version')==2}
    def checkpoint(self,evidence_ids):
        if self.coefficients is None: raise ValueError('predictor not fitted')
        return {'schema_version':2,'algorithm':'ridge-regression','feature_schema':list(PREDICTOR_FEATURES),
            'coefficients':self.coefficients,'training_models':self.training_models,
            'training_rmse_cycles':self.rmse,'evidence_ids':evidence_ids,
            'held_out_model':PREDICTOR_HELD_OUT_MODEL,'provenance':'measured-rtl-training'}
    @staticmethod
    def features(row):
        cfg=row['config']; w=row['workload']
        metadata=row.get('microarchitecture',{})
        uarch=metadata.get('parameters')
        if metadata.get('schema_version')!=1 or not isinstance(uarch,dict) or 'FIFO_DEPTH' not in uarch:
            raise ValueError('explicit versioned microarchitecture metadata required for predictor v2')
        generated=0 if row.get('generation_mode')=='fixed-token-tape' else w['max_new']
        return [1.,row.get('prompt_tokens',1)/2048,generated/256,cfg['MCOLS']/8,
                cfg['LANES']/16,w['latency']/100,w['stall_percent']/100,
                cfg['DRAM_BYTES']/2**30,{'int8':1.,'int4':.5,'fp4':.25}[w['wformat']],
                uarch['FIFO_DEPTH']/1024,w['bandwidth_percent']/100]
    def fit(self,rows):
        import numpy as np
        if {r.get('base_model_id') for r in rows}!=PREDICTOR_TRAINING_MODELS:
            raise ValueError('predictor partition is frozen: Qwen3/Qwen3.5 train; LFM2.5 held-out')
        if any(not r.get('valid') or r.get('backend')!='rtl' or not r.get('counters')
              or any(s.get('validation')!='bit-exact-dram-and-tmem' for s in r['counters']) for r in rows):
            raise ValueError('predictor fitting requires validated measured RTL rows')
        if len(rows)<10: raise ValueError('at least ten measured training observations required')
        x=np.array([self.features(r) for r in rows]); y=np.array([sum(s['cycles'] for s in r['counters']) for r in rows])
        beta=np.linalg.solve(x.T@x+np.eye(x.shape[1])*.001,x.T@y)
        self.coefficients=beta.tolist(); self.rmse=float(np.sqrt(np.mean((x@beta-y)**2)))
        self.training_models=sorted({r['base_model_id'] for r in rows}); return self
    def predict(self,row):
        if self.coefficients is None: raise ValueError('predictor not fitted')
        if len(self.coefficients)!=len(PREDICTOR_FEATURES): raise ValueError('predictor coefficient schema mismatch')
        return {'cycles':max(1.,sum(a*b for a,b in zip(self.features(row),self.coefficients))),
                'rmse':self.rmse,'provenance':'estimated-measured-data-regression'}
