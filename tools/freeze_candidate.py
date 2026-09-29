"""Create the single pre-held-out freeze for a candidate that passed full validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from malleable.llm import candidates as C
from malleable.llm.models import digest,inspect
from malleable.store import Store


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True,help='new held-out attempt directory')
    p.add_argument('--model',type=Path,required=True); p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--candidate-case',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True,help='completed candidate-isa-validation attempt')
    a=p.parse_args()
    summary=json.loads((a.validation/'summary.json').read_text())
    result_file=a.validation/'result.json'; result_sha=digest(result_file)
    if summary.get('status')!='completed' or summary.get('validation_passed') is not True or summary.get('result_sha256')!=result_sha:
        raise ValueError('a completed, passing, hash-verified full validation is required')
    validation=json.loads(result_file.read_text())
    info=inspect(a.model,128); derived=C.read_derived_candidate(a.candidate_case,info)
    a.root.mkdir(parents=True)
    freeze=C.create_freeze(a.root/'freeze.json',info,a.suite,derived,validation,result_sha,C.source_state())
    store=Store(a.root/'store'/'research')
    try: quality_id=store.save('llm-quality',validation)
    finally: store.close()
    print(json.dumps({'freeze_id':freeze['freeze_id'],'validation_quality_record':quality_id,
        'freeze_sha256':digest(a.root/'freeze.json'),'source':freeze['source']}))


if __name__=='__main__': main()
