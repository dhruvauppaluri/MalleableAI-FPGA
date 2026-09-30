"""Register audited historical held-out evidence without evaluating it again."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from malleable.llm.heldout import register_completed
from malleable.llm.models import digest,inspect
from malleable.llm.quality import frozen_suite,gate
from malleable.llm.candidates import read_derived_candidate,derived_record,variant_id,validate_configuration
from malleable.records import identity

def main():
    release=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'
    entries=[('Qwen3-0.6B','qwen3','candidate-heldout-freeze-01',
        'b513e1adb0e25e0b22f23eefadbf8b36441aebcee13e000061cd559e6a2197d9',
        release/'quality-recovery-pilot-01/int8/case-00'),
        ('LFM2.5-230M','lfm2','lfm2-raw-int8-heldout-01',
        '95f6aeedcf2b324207acb596708c17e15f5df43d72da14b28a1800da4ea21371',None)]
    reports=[]
    for name,suite_name,attempt,sha,case in entries:
        path=release/attempt/'result.json';record=json.loads(path.read_text())
        info=inspect(ROOT/'build/models'/name,128)
        suite=ROOT/'build/models/quality'/f'{suite_name}-frozen-v2.json';data,hashes=frozen_suite(suite)
        derived=read_derived_candidate(case,info) if case else None
        validate_configuration(record,info)
        if (record.get('base_model_id')!=info['base_model_id'] or record.get('tokenizer_id')!=info['tokenizer_id']
            or record.get('variant_id')!=variant_id(info['base_model_id'],'int8',derived)
            or record.get('derived_candidate')!=(derived_record(derived) if derived else None)
            or record.get('suite_file_hash')!=digest(suite) or record.get('suite_hashes')!=hashes
            or record.get('target_count')!=sum(len(r)-1 for r in data['held-out'])
            or identity({k:v for k,v in record.items() if k!='record_id'})!=record.get('record_id')
            or not gate(record['float_nll'],record['candidate_nll'],record['agreement'])['passed']):
            raise ValueError('historical held-out identity failed: '+name)
        reports.append({'model':name,**register_completed(record,path,sha)})
    print(json.dumps({'status':'registered','inference_performed':False,'evaluations':reports},indent=2))

if __name__=='__main__':main()
