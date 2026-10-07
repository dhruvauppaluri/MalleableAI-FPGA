"""Durable ISA validation, freeze, and exactly-once held-out evaluation.

Each root is one model/personality/suite design. Results remain in ignored build/
until the whole campaign has been audited and published.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import candidates as C
from malleable.llm.models import digest, inspect
from malleable.llm.quality import gate
from malleable.llm.records import PERSONALITIES
from malleable.records import identity


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def report(value):
    """Best-effort progress reporting must never determine attempt outcome."""
    try:
        print(json.dumps(value), flush=True)
    except OSError:
        # A detached supervisor may stop reading while the worker is healthy.
        # Durable job, worker, result, and failure records remain authoritative.
        pass


def command_for(split, model, suite, personality, store, candidate_case=None, freeze=None,
                reference_cache=None):
    if split not in ('validation', 'held-out'):
        raise ValueError('unsupported quality split')
    if personality not in ('compact', 'balanced', 'compute', 'buffered'):
        raise ValueError('unsupported personality')
    if split == 'held-out' and personality == 'balanced':
        raise ValueError('balanced held-out is already approved and must not be repeated')
    if split == 'held-out' and freeze is None:
        raise ValueError('held-out dispatch requires its frozen design')
    command = [sys.executable, '-u', '-m', 'malleable.llm.cli',
               'quality-candidate-validate' if candidate_case else 'quality',
               '--model', str(model), '--suite', str(suite), '--split', split,
               '--context', '128', '--personality', personality,
               '--wformat', 'int8', '--max-host-gib', '16', '--store', str(store)]
    if reference_cache:
        command += ['--reference-cache', str(reference_cache)]
    if candidate_case:
        command += ['--candidate-case', str(candidate_case)]
    if freeze:
        command += ['--candidate-freeze', str(freeze)]
    return command


def checked_source():
    source = C.source_state()
    if source['status'] or not source['commit']:
        raise ValueError('commit clean source before quality dispatch or freeze')
    return source


def check_result(result, *, split, info, suite, personality, derived, freeze=None):
    from malleable.llm.quality import frozen_suite
    _, hashes = frozen_suite(suite)
    expected = C.variant_id(info['base_model_id'], 'int8', derived)
    record = C.derived_record(derived) if derived else None
    if (result.get('split') != split or result.get('personality') != personality
        or result.get('base_model_id') != info['base_model_id']
        or result.get('tokenizer_id') != info['tokenizer_id']
        or result.get('variant_id') != expected
        or result.get('derived_candidate') != record
        or result.get('suite_file_hash') != digest(suite)
        or result.get('suite_hashes') != hashes
        or result.get('suite_frozen') is not True
        or result.get('context') != 128
        or result.get('wformat') != 'int8' or result.get('head_format') != 'int8'
        or result.get('target_count', 0) < 1024
        or result.get('samples') != result.get('target_count')):
        raise ValueError('quality result differs from the frozen exact design')
    C.validate_configuration(result, info)
    record_id = result.get('record_id')
    if not record_id or identity({k: v for k, v in result.items() if k != 'record_id'}) != record_id:
        raise ValueError('quality result content address mismatch')
    checks = gate(result['float_nll'], result['candidate_nll'], result['agreement'])
    if result.get('passed') is not checks['passed']:
        raise ValueError('quality gate mismatch')
    if split == 'validation':
        if result.get('selectable') is not False:
            raise ValueError('validation must not be selectable')
    elif (result.get('candidate_freeze_id') != freeze['freeze_id']
          or result.get('selectable') is not checks['passed']):
        raise ValueError('held-out approval/freeze mismatch')
    return checks


def run_quality(args, info, derived, source):
    stage = args.root / args.phase
    if args.phase == 'validation':
        stage.mkdir(parents=True, exist_ok=False)
        freeze = None
        freeze_path = None
    else:
        freeze_path = stage / 'freeze.json'
        if not freeze_path.is_file() or any((stage / name).exists() for name in
                                            ('job.json', 'heldout-claim.json', 'result.json', 'failure.json')):
            raise ValueError('held-out freeze missing or attempt already started')
        expected_configuration = identity({
            'config': info['personalities'][args.personality]['config'],
            'uarch': PERSONALITIES[args.personality].uarch})
        freeze = C.verify_freeze(freeze_path, info, args.suite, derived, source,
                                 expected_configuration)
    initial = inspect(args.model, 128)
    suite_sha = digest(args.suite)
    command = command_for(args.phase, args.model, args.suite, args.personality,
                          stage / 'store', args.candidate_case, freeze_path,
                          args.reference_cache)
    started = time.monotonic()
    write_new(stage / 'job.json', {'schema_version': 1, 'source': source,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': str(args.model), 'model_id': info['base_model_id'],
              'suite': str(args.suite), 'suite_sha256': suite_sha,
              'personality': args.personality, 'command': command,
              'freeze_id': freeze['freeze_id'] if freeze else None,
              'authorization': args.authorization})
    shutil.copyfile(__file__, stage / 'runner.py')
    result = None
    try:
        with (stage / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            write_new(stage / 'child.json', {'pid': process.pid})
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('kind') == 'result':
                    result = event['payload']
                elif event.get('kind') == 'quality-progress':
                    payload = event.get('payload', {})
                    completed = payload.get('completed_targets')
                    if completed and completed % 128 == 0:
                        report({'phase': args.phase, 'personality': args.personality,
                                'completed_targets': completed})
            code = process.wait()
        if code or result is None:
            raise RuntimeError(f'quality worker exit {code}; see {stage / "worker.log"}')
        current = inspect(args.model, 128)
        if (C.source_state() != source or digest(args.suite) != suite_sha
            or any(initial[key] != current[key] for key in
                   ('base_model_id', 'tokenizer_id', 'weight_files', 'tokenizer_files'))):
            raise ValueError('source, suite, model, or tokenizer changed during evaluation')
        checks = check_result(result, split=args.phase, info=info, suite=args.suite,
                              personality=args.personality, derived=derived, freeze=freeze)
        write_new(stage / 'result.json', result)
        summary = {'schema_version': 1, 'status': 'completed', 'split': args.phase,
                   'personality': args.personality, 'record_id': result['record_id'],
                   'result_sha256': digest(stage / 'result.json'),
                   'agreement': result['agreement'],
                   'nll_degradation': checks['nll_degradation'],
                   'passed': checks['passed'], 'selectable': result['selectable'],
                   'target_count': result['target_count'],
                   'configuration_id': result['configuration_id'],
                   'elapsed_seconds': time.monotonic() - started}
        write_new(stage / 'summary.json', summary)
        report(summary)
    except BaseException:
        write_new(stage / 'failure.json', {'status': 'failed',
                  'elapsed_seconds': time.monotonic() - started,
                  'traceback': traceback.format_exc()})
        raise


def freeze_design(args, info, derived, source):
    validation = args.root / 'validation'
    summary = json.loads((validation / 'summary.json').read_text())
    result_file = validation / 'result.json'
    result = json.loads(result_file.read_text())
    if (summary.get('status') != 'completed' or summary.get('passed') is not True
        or summary.get('result_sha256') != digest(result_file)):
        raise ValueError('passing hash-verified validation is required')
    check_result(result, split='validation', info=info, suite=args.suite,
                 personality=args.personality, derived=derived)
    stage = args.root / 'held-out'
    stage.mkdir(parents=True, exist_ok=False)
    freeze = C.create_freeze(stage / 'freeze.json', info, args.suite, derived,
                             result, digest(result_file), source)
    write_new(stage / 'freeze-summary.json', {'freeze_id': freeze['freeze_id'],
              'freeze_sha256': digest(stage / 'freeze.json'),
              'validation_record_id': result['record_id'],
              'configuration_id': result['configuration_id']})
    report({'status': 'frozen', 'personality': args.personality,
            'freeze_id': freeze['freeze_id']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('validation', 'freeze', 'held-out'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--personality', choices=('compact', 'balanced', 'compute', 'buffered'), required=True)
    parser.add_argument('--candidate-case', type=Path)
    parser.add_argument('--reference-cache', type=Path)
    parser.add_argument('--authorization', default='user-requested-controller-quality-campaign')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.root = args.root.resolve()
    args.model = args.model.resolve()
    args.suite = args.suite.resolve()
    args.candidate_case = args.candidate_case.resolve() if args.candidate_case else None
    args.reference_cache = args.reference_cache.resolve() if args.reference_cache else None
    if not args.root.is_relative_to(ROOT / 'build'):
        raise ValueError('attempt root must be in ignored build/')
    if args.reference_cache and not args.reference_cache.is_relative_to(ROOT / 'build'):
        raise ValueError('reference cache must be in ignored build/')
    source = checked_source()
    info = inspect(args.model, 128)
    if not info['supported']:
        raise ValueError('unsupported model')
    derived = C.read_derived_candidate(args.candidate_case, info) if args.candidate_case else None
    if args.phase == 'freeze':
        freeze_design(args, info, derived, source)
    else:
        run_quality(args, info, derived, source)


if __name__ == '__main__':
    main()
