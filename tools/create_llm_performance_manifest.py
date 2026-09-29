"""Create the fixed 10-run RTL comparison schedule for one local model."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from malleable.llm.models import inspect
from malleable.llm.optimization import validated_variants
from malleable.llm.records import PERSONALITIES
from malleable.llm.experiments import validate_performance_manifest
from malleable.records import identity
from malleable.store import Store


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
    p.add_argument('--store',default='build/llm-runs/research'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); model=Path(a.model).resolve(); info=inspect(model)
    if not info['supported']: raise ValueError('supported local checkpoint required')
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    prompts={'short-a':'A small neural network performs',
             'short-b':'An FPGA can accelerate dense model inference by'}
    tapes={name:tok.encode(text) for name,text in prompts.items()}
    if any(not ids or len(ids)>128 for ids in tapes.values()): raise ValueError('fixed short tape must fit context 128')
    store=Store(a.store)
    try:
        quality=store.records('llm-quality')
        approved=validated_variants(quality,info['base_model_id'],info['tokenizer_id'],128)
        evidence=[{'id':identity(q),'record':q} for q in quality if q.get('base_model_id')==info['base_model_id']
            and q.get('personality')=='balanced' and q.get('split')=='validation'
            and q.get('suite_frozen') is True and q.get('target_count',0)>=1024
            and (q.get('variant_id'),q.get('personality')) in approved]
    finally: store.close()
    runs=[]
    for name,ids in tapes.items():
        for personality in PERSONALITIES:
            runs.append({'workload':name,'personality':personality,'wformat':'int8','context':128,
                'input_tokens':ids,'input_token_hash':identity(ids),'memory_scenario':'baseline',
                'latency':20,'stall_percent':20,'bandwidth_percent':100})
    if all((identity({'base':info['base_model_id'],'format':w,'head':'int8'}),'balanced') in approved
           for w in ('int4','fp4')):
        for wformat in ('int4','fp4'):
            ids=tapes['short-a']; runs.append({'workload':'short-a','personality':'balanced','wformat':wformat,
                'context':128,'input_tokens':ids,'input_token_hash':identity(ids),'memory_scenario':'baseline',
                'latency':20,'stall_percent':20,'bandwidth_percent':100})
    else:
        ids=tapes['short-a']
        for personality in ('balanced','compute'):
            runs.append({'workload':'axi-stress','personality':personality,'wformat':'int8','context':128,
                'input_tokens':ids,'input_token_hash':identity(ids),'memory_scenario':'stress',
                'latency':100,'stall_percent':50,'bandwidth_percent':50})
    data={'schema_version':1,'model_path':str(model),'base_model_id':info['base_model_id'],
        'tokenizer_id':info['tokenizer_id'],'seed':a.seed,'suite':'10-run-real-model-rtl-v1',
        'quality_evidence_ids':[e['id'] for e in evidence],'quality_evidence':evidence,
        'fixed_prompt_hashes':{k:identity(v) for k,v in tapes.items()},
        'runs':runs,'baseline_axi':{'latency':20,'stall_probability_percent':20,'bandwidth_percent':100},
        'stress_axi':{'latency':100,'stall_probability_percent':50,'bandwidth_percent':50}}
    validate_performance_manifest(data,info)
    out=Path(a.output)
    if out.exists(): raise ValueError('refusing to overwrite an existing reproducible manifest')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'manifest':str(out),'manifest_id':identity(data),'runs':len(runs),
        'formats':[r['wformat'] for r in runs],'quality_evidence_ids':evidence}))

if __name__=='__main__': main()
