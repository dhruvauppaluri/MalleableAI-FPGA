"""Recover a completed validation rejected only by the historical cache-path defect."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import candidates as C
from malleable.llm.models import digest, inspect
from tools.run_personality_quality import check_result, write_new


def logged_result(path):
    results = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('kind') == 'result':
            results.append(event.get('payload'))
    if len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError('exactly one completed logged result is required')
    return results[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--personality', required=True)
    parser.add_argument('--candidate-case', type=Path)
    args = parser.parse_args()
    stage = args.root.resolve() / 'validation'
    if (not (stage / 'failure.json').is_file() or (stage / 'result.json').exists()
        or (stage / 'summary.json').exists()):
        raise ValueError('one failed wrapper with no finalized result is required')
    failure = json.loads((stage / 'failure.json').read_text())
    if 'source, suite, model, or tokenizer changed during evaluation' not in failure.get('traceback', ''):
        raise ValueError('only the cache-path source-status rejection is recoverable')
    job = json.loads((stage / 'job.json').read_text())
    model = args.model.resolve()
    suite = args.suite.resolve()
    if (job.get('model') != str(model) or job.get('suite') != str(suite)
        or job.get('personality') != args.personality
        or job.get('suite_sha256') != digest(suite)
        or job.get('source', {}).get('status') != ''):
        raise ValueError('job lineage mismatch')
    info = inspect(model, 128)
    candidate = args.candidate_case.resolve() if args.candidate_case else None
    derived = C.read_derived_candidate(candidate, info) if candidate else None
    result = logged_result(stage / 'worker.log')
    checks = check_result(result, split='validation', info=info, suite=suite,
                          personality=args.personality, derived=derived)
    write_new(stage / 'result.json', result)
    summary = {'schema_version': 1, 'status': 'completed-after-cache-path-recovery',
               'recovered_utc': datetime.now(timezone.utc).isoformat(),
               'split': 'validation', 'personality': args.personality,
               'record_id': result['record_id'],
               'result_sha256': digest(stage / 'result.json'),
               'agreement': result['agreement'],
               'nll_degradation': checks['nll_degradation'],
               'passed': checks['passed'], 'selectable': result['selectable'],
               'target_count': result['target_count'],
               'configuration_id': result['configuration_id'],
               'wrapper_failure_sha256': digest(stage / 'failure.json')}
    write_new(stage / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
