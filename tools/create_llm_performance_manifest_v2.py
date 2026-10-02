"""Per-run candidate/precision lineage with measured four-bit disposition."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from malleable.llm.models import inspect
from malleable.llm.candidates import read_derived_candidate,derived_record,variant_id
from malleable.llm.quality import gate
from malleable.llm.experiments import validate_performance_manifest,precision_lineage
from malleable.llm.records import PERSONALITIES
from malleable.records import identity

def create(model,output,quality_paths,candidate_case=None,seed=42):
    model=Path(model).resolve();info=inspect(model,128)
    if not info['supported']:raise ValueError('supported model required')
    derived=read_derived_candidate(candidate_case,info) if candidate_case else None
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    tapes={k:tok.encode(v) for k,v in {'short-a':'A small neural network performs',
        'short-b':'An FPGA can accelerate dense model inference by'}.items()}
    if any(not ids or len(ids)>128 for ids in tapes.values()):raise ValueError('fixed short tapes required')
    screens=[]
    for path in quality_paths:
        record=json.loads(Path(path).read_text());key=record.pop('record_id',None)
        if key!=identity(record):raise ValueError('screen canonical record mismatch')
        screens.append({'id':key,'record':record})
    passed=[e for e in screens if gate(e['record']['float_nll'],e['record']['candidate_nll'],e['record']['agreement'])['passed']]
    runs=[]
    def add(name,personality,wformat,stress=False):
        ids=tapes['short-a' if stress else name];candidate=derived if wformat=='int8' else None
        runs.append({'workload':name,'personality':personality,'wformat':wformat,'context':128,
            'input_tokens':ids,'input_token_hash':identity(ids),'memory_scenario':'stress' if stress else 'baseline',
            'latency':100 if stress else 20,'stall_percent':50 if stress else 20,'bandwidth_percent':50 if stress else 100,
            'candidate_case':str(Path(candidate_case).resolve()) if candidate else None,
            'derived_candidate':derived_record(candidate) if candidate else None,
            'variant_id':variant_id(info['base_model_id'],wformat,candidate),'precision_policy':precision_lineage(wformat)})
    for name in tapes:
        for p in PERSONALITIES:add(name,p,'int8')
    if len(passed)==2:
        for fmt in ('int4','fp4'):add('short-a','balanced',fmt)
    else:
        for p in ('balanced','compute'):add('axi-stress',p,'int8',True)
    data={'schema_version':2,'model_path':str(model),'base_model_id':info['base_model_id'],
        'tokenizer_id':info['tokenizer_id'],'seed':seed,'suite':'10-run-real-model-rtl-v2',
        'quality_evidence_ids':[e['id'] for e in passed],'quality_evidence':passed,
        'format_screening_evidence':screens,'fixed_prompt_hashes':{k:identity(v) for k,v in tapes.items()},
        'runs':runs,'baseline_axi':{'latency':20,'stall_probability_percent':20,'bandwidth_percent':100},
        'stress_axi':{'latency':100,'stall_probability_percent':50,'bandwidth_percent':50},
        'controller_approval':'Performance evidence only; matching personality-specific held-out approval required for automatic application.'}
    validate_performance_manifest(data,info)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    return data
