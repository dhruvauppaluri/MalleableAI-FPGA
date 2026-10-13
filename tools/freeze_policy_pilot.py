"""Create fresh serial prefill pilot, excluding all 30 published input tapes."""
import json
from pathlib import Path
from malleable.llm.policy_campaign import validate_manifest
from malleable.llm.quality import quality_matches
from malleable.records import identity
from tools.evaluate_controller_continuation import published_quality
from tools.evaluate_exported_predictor import load_verified

ROOT=Path(__file__).resolve().parents[1]
TAPES={'qwen3':[4,17,29], 'qwen35':[4,17,29], 'lfm':[4,17,29]}


def manifest(source_id):
    prior={name:load_verified(name) for name in TAPES}
    approvals=published_quality()
    cases=[]
    for name,rows in prior.items():
        controls={r['personality']:r for r in rows if r['benchmark_workload']=='short-a'}
        tape=TAPES[name]
        for personality,row in sorted(controls.items()):
            if not quality_matches(approvals[name][personality],row,split='held-out'):
                raise ValueError('approved same-variant quality required')
            workload=row['workload']
            case=dict(base_model_id=row['base_model_id'],tokenizer_id=row['tokenizer_id'],
                variant_id=row['variant_id'],configuration_id=row['configuration_id'],
                personality=personality,wformat='int8',context=128,seed=82361,
                input_tokens=tape,input_token_hash=identity(tape),
                memory_scenario='fresh-short-prefill',latency=workload['latency'],
                stall_percent=workload['stall_percent'],bandwidth_percent=workload['bandwidth_percent'],
                split='development-pilot',quality_evidence_id=identity(approvals[name][personality]),
                workload_identity_v2=identity({'input':tape,'seed':82361,'context':128,
                    'max_new':0,'mode':'fixed-token-tape','memory':{'latency':workload['latency'],
                    'stalls':workload['stall_percent'],'bandwidth':workload['bandwidth_percent']}}))
            cases.append(case)
    frozen=dict(schema_version=1,status='frozen',campaign_id='policy-short-prefill-pilot-v1',
                source_id=source_id,scope='fresh-development-pilot-no-independent-test',
                cases=cases,automatic_switching_allowed=False)
    validate_manifest(frozen,[row for rows in prior.values() for row in rows])
    return frozen


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-id',required=True)
    a=p.parse_args(); print(json.dumps(manifest(a.source_id),sort_keys=True,indent=2))
