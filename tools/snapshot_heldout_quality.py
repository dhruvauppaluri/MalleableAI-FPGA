"""Publish verified, exactly-once quality claims from ignored attempt roots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import candidates as C
from malleable.llm.heldout import evaluation_key
from malleable.llm.models import digest, inspect
from malleable.records import identity
from tools.run_personality_quality import check_result
from tools.snapshot_quality_campaign import MODELS, copy_verified


def verify_attempt(attempt, info, suite, derived):
    stage = attempt / 'held-out'
    freeze_path = stage / 'freeze.json'
    freeze = json.loads(freeze_path.read_text())
    personality = freeze['configuration']['personality']
    if personality == 'balanced':
        raise ValueError('balanced held-out must not be repeated')
    validation_path = attempt / 'validation' / 'result.json'
    validation = json.loads(validation_path.read_text())
    check_result(validation, split='validation', info=info, suite=suite,
                 personality=personality, derived=derived)
    if (freeze['validation']['record_id'] != validation['record_id']
        or freeze['validation']['result_sha256'] != digest(validation_path)):
        raise ValueError(f'{attempt}: freeze does not bind its validation')
    C.verify_freeze(freeze_path, info, suite, derived, freeze['source'],
                    validation['configuration_id'])
    row = {'model': Path(info['path']).name, 'attempt': attempt.name,
           'personality': personality, 'freeze_id': freeze['freeze_id'],
           'freeze_sha256': digest(freeze_path),
           'validation_record_id': validation['record_id'],
           'configuration_id': validation['configuration_id']}
    claim_path = stage / 'heldout-claim.json'
    if claim_path.is_file():
        claim = json.loads(claim_path.read_text())
        key = evaluation_key(freeze)
        if (claim.get('key') != key or claim.get('evaluation_id') != identity(key)
            or claim.get('freeze_id') != freeze['freeze_id']
            or claim.get('status') != 'consumed'):
            raise ValueError(f'{attempt}: claim identity mismatch')
        row.update(claimed_utc=claim['claimed_utc'], evaluation_id=claim['evaluation_id'],
                   claim_sha256=digest(claim_path), key=key)
    elif (stage / 'job.json').exists() and not (stage / 'failure.json').exists():
        raise ValueError(f'{attempt}: active preclaim job cannot be snapshotted')
    result_path = stage / 'result.json'
    if result_path.is_file():
        if not claim_path.is_file():
            raise ValueError(f'{attempt}: result without exact-once claim')
        result = json.loads(result_path.read_text())
        summary_path = stage / 'summary.json'
        summary = json.loads(summary_path.read_text())
        checks = check_result(result, split='held-out', info=info, suite=suite,
                              personality=personality, derived=derived, freeze=freeze)
        if (summary.get('status') != 'completed'
            or summary.get('result_sha256') != digest(result_path)
            or summary.get('passed') is not checks['passed']):
            raise ValueError(f'{attempt}: held-out result/summary mismatch')
        row.update(status='passed' if checks['passed'] else 'failed-quality-gate',
                   record_id=result['record_id'], result_sha256=digest(result_path),
                   agreement=result['agreement'], nll_degradation=checks['nll_degradation'],
                   passed=checks['passed'], target_count=result['target_count'])
    elif (stage / 'failure.json').is_file():
        row['status'] = 'consumed-failed' if claim_path.is_file() else 'failed-before-claim'
        row['failure_sha256'] = digest(stage / 'failure.json')
    elif claim_path.is_file():
        raise ValueError(f'{attempt}: live claimed job cannot be snapshotted')
    else:
        row['status'] = 'frozen-pending'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'build/models')
    parser.add_argument('--candidate-case', type=Path)
    args = parser.parse_args()
    snapshot_path = args.output_root / 'snapshots' / f'{args.snapshot}.json'
    if snapshot_path.exists():
        raise ValueError('refusing to overwrite an append-only snapshot')
    ledger_path = args.output_root / 'consumed-ledger.json'
    ledger = json.loads(ledger_path.read_text())
    claims = {entry['evaluation_id']: entry for entry in ledger['exact_evaluations']}
    rows = []
    for model in MODELS:
        info = inspect(args.model_root / model, 128)
        suite = args.suite_root / f'{model}.json'
        candidate = args.candidate_case if model == 'Qwen3-0.6B' else None
        derived = C.read_derived_candidate(candidate, info) if candidate else None
        for attempt in sorted((args.campaign_root / 'runs' / model).glob('*')):
            stage = attempt / 'held-out'
            if not (stage / 'freeze.json').is_file():
                continue
            row = verify_attempt(attempt, info, suite, derived)
            destination = args.output_root / 'published' / 'held-out' / model / attempt.name
            for name in ('freeze.json', 'freeze-summary.json', 'job.json',
                         'heldout-claim.json', 'result.json', 'summary.json', 'failure.json'):
                source = stage / name
                if source.is_file():
                    copy_verified(source, destination / name)
            if 'evaluation_id' in row:
                entry = {'evaluation_id': row['evaluation_id'], 'key': row['key'],
                         'status': 'consumed-published' if row['status'] == 'passed' else row['status'],
                         'freeze_id': row['freeze_id'], 'claimed_utc': row['claimed_utc'],
                         'result_sha256': row.get('result_sha256'),
                         'failure_sha256': row.get('failure_sha256')}
                old = claims.get(entry['evaluation_id'])
                if old is not None and old != entry:
                    raise ValueError(f'{attempt}: published claim differs')
                if old is None:
                    ledger['exact_evaluations'].append(entry)
                    claims[entry['evaluation_id']] = entry
            rows.append(row)
    payload = {'schema_version': 1, 'kind': 'append-only-heldout-quality-snapshot',
               'snapshot': args.snapshot, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'automatic_switching': 'gated', 'attempts': rows}
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    ledger_path.write_text(json.dumps(ledger, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'snapshot': str(snapshot_path), 'claims': sum('evaluation_id' in r for r in rows),
                      'passed': sum(r['status'] == 'passed' for r in rows)}))


if __name__ == '__main__':
    main()
