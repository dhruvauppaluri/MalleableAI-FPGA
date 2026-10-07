"""Publish an append-only, hash-verified snapshot of ignored campaign attempts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import candidates as C
from malleable.llm.models import digest, inspect
from tools.run_personality_quality import check_result

MODELS = ('LFM2.5-230M', 'Qwen3-0.6B', 'Qwen3.5-0.8B')


def copy_verified(source, target):
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'published artifact differs: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'build/models')
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--candidate-case', type=Path)
    args = parser.parse_args()
    snapshot_path = args.output_root / 'snapshots' / f'{args.snapshot}.json'
    if snapshot_path.exists():
        raise ValueError('refusing to overwrite an append-only snapshot')
    rows = []
    for model in MODELS:
        info = inspect(args.model_root / model, 128)
        suite = args.suite_root / f'{model}.json'
        candidate = args.candidate_case if model == 'Qwen3-0.6B' else None
        derived = C.read_derived_candidate(candidate, info) if candidate else None
        model_root = args.campaign_root / 'runs' / model
        for attempt in sorted(model_root.glob('*')) if model_root.exists() else ():
            stage = attempt / 'validation'
            if not (stage / 'job.json').is_file():
                continue
            job = json.loads((stage / 'job.json').read_text())
            row = {'model': model, 'attempt': attempt.name, 'phase': 'validation',
                   'personality': job.get('personality'), 'source': job.get('source'),
                   'job_sha256': digest(stage / 'job.json')}
            destination = args.output_root / 'published' / 'validation' / model / attempt.name
            if (stage / 'result.json').is_file() and (stage / 'summary.json').is_file():
                result = json.loads((stage / 'result.json').read_text())
                summary = json.loads((stage / 'summary.json').read_text())
                if summary.get('result_sha256') != digest(stage / 'result.json'):
                    raise ValueError(f'{attempt}: summary/result hash mismatch')
                checks = check_result(result, split='validation', info=info, suite=suite,
                                      personality=job['personality'], derived=derived)
                if not checks['passed'] or summary.get('passed') is not True:
                    raise ValueError(f'{attempt}: only passing verified validation is publishable')
                for name in ('result.json', 'summary.json'):
                    copy_verified(stage / name, destination / name)
                row.update(status=summary['status'], record_id=result['record_id'],
                           result_sha256=summary['result_sha256'], agreement=result['agreement'],
                           nll_degradation=checks['nll_degradation'], passed=True,
                           configuration_id=result['configuration_id'])
            else:
                row['status'] = 'in-progress'
            if (stage / 'failure.json').is_file():
                copy_verified(stage / 'failure.json', destination / 'failure.json')
                row['failure_sha256'] = digest(stage / 'failure.json')
                if row['status'] == 'in-progress':
                    row['status'] = 'failed-validation-attempt'
            rows.append(row)
    payload = {'schema_version': 1, 'kind': 'append-only-quality-campaign-snapshot',
               'snapshot': args.snapshot, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'automatic_switching': 'gated', 'heldout_opened': False,
               'attempts': rows}
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'snapshot': str(snapshot_path), 'attempts': len(rows),
                      'completed': sum(row.get('passed') is True for row in rows)}))


if __name__ == '__main__':
    main()
