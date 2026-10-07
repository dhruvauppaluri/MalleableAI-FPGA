"""Offline controller audit from the thirty already published RTL benchmark rows.

No RTL simulation or quality target inference is invoked. Request sequences and
switching costs are explicit assumptions; evaluation cannot promote switching.
"""
import json
import hashlib
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_exported_predictor import load_verified, summarize
from malleable.llm.learning import evaluate, train, validate_release_episodes
from malleable.llm.optimization import Predictor
from malleable.llm.quality import quality_matches
from malleable.records import identity


NAMES = ('qwen3', 'qwen35', 'lfm')
PERSONALITIES = ('compact', 'balanced', 'compute', 'buffered')
WORKLOADS = ('short-a', 'short-b')
HORIZONS = (1, 8, 32, 128)
OBJECTIVES = ('latency', 'throughput')
COST_FRACTIONS = (0.0, 0.05, 0.20)


def quality_check():
    ledger = json.loads((ROOT / 'docs/evidence/controller-quality-continuation-20261006/consumed-ledger.json').read_text())
    published = [e for e in ledger['exact_evaluations'] if e['status'] == 'consumed-published']
    if len(published) != 12:
        raise ValueError('all twelve balanced or independent held-out approvals must be published first')
    results = {}
    for name in ('LFM2.5-230M', 'Qwen3-0.6B', 'Qwen3.5-0.8B'):
        rows = [e for e in published if e['key']['base_model_id'] == {
            'LFM2.5-230M': '8d6ac525c1b135ad360d0f9f5a822698bbb257b7408f791e2ca695ae473798b3',
            'Qwen3-0.6B': '7ab1181d3a2b04ce889880dfc3b94933574441e9c221e950622c39a3ce79a59d',
            'Qwen3.5-0.8B': '96c8847b78593ff15008955c68e384f5ff40895e425e487aec30b8173542a857'}[name]]
        if len(rows) != 4:
            raise ValueError(f'{name} requires four published approvals')
        results[name] = [e['evaluation_id'] for e in rows]
    return results


def published_quality():
    ledger = json.loads((ROOT / 'docs/evidence/controller-quality-continuation-20261006/consumed-ledger.json').read_text())
    ledger_by_id = {entry['evaluation_id']: entry for entry in ledger['exact_evaluations']}
    archive_dir = ROOT / 'docs/evidence/cloud-handoff-20261004-archive'
    archive = b''.join(path.read_bytes() for path in sorted(archive_dir.glob('part-*')))
    if hashlib.sha256(archive).hexdigest() != '1a36341516073952ad568bb5075ac683e8e5a15fe2b92ea0f8476677d2993be8':
        raise ValueError('published Zephyrus archive hash mismatch')
    model_names = {'qwen3': 'Qwen3-0.6B', 'qwen35': 'Qwen3.5-0.8B', 'lfm': 'LFM2.5-230M'}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        balanced = {}
        for short, model in model_names.items():
            name = f'cloud-handoff-20261004/standalone-v3/models/{model}/held-out-quality.json'
            balanced[short] = json.load(bundle.extractfile(name))
    attempts = {
        'qwen3': ('Qwen3-0.6B', {'compact': 'compact-retry-01', 'compute': 'compute',
                                'buffered': 'buffered-retry-01'}),
        'qwen35': ('Qwen3.5-0.8B', {'compact': 'compact-retry-03', 'compute': 'compute',
                                  'buffered': 'buffered'}),
        'lfm': ('LFM2.5-230M', {'compact': 'compact', 'compute': 'compute',
                              'buffered': 'buffered'}),
    }
    quality = {}
    for name, (model, personalities) in attempts.items():
        quality[name] = {'balanced': balanced[name]}
        for personality, attempt in personalities.items():
            stage = (ROOT / 'docs/evidence/controller-quality-continuation-20261006/published/held-out'
                     / model / attempt)
            path = stage / 'result.json'
            result = json.loads(path.read_text())
            claim = json.loads((stage / 'heldout-claim.json').read_text())
            freeze = json.loads((stage / 'freeze.json').read_text())
            entry = ledger_by_id.get(claim['evaluation_id'])
            if (entry is None or entry['status'] != 'consumed-published'
                or entry['key'] != claim['key']
                or entry['result_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
                or result['candidate_freeze_id'] != freeze['freeze_id']
                or claim['freeze_id'] != freeze['freeze_id']):
                raise ValueError(f'unmatched exact-once publication: {model}/{attempt}')
            quality[name][personality] = result
    return quality


def episode_set(rows, predictor, predictor_id, model_names, quality):
    episodes = []
    evidence = {}
    for name in model_names:
        model_rows = rows[name]
        for workload in WORKLOADS:
            compared = [r for r in model_rows if r['benchmark_workload'] == workload]
            if len(compared) != 4 or {r['personality'] for r in compared} != set(PERSONALITIES):
                raise ValueError(f'incomplete four-action tape: {name}/{workload}')
            for row in compared:
                if not quality_matches(quality[name][row['personality']], row, split='held-out'):
                    raise ValueError(f'unmatched held-out quality: {name}/{workload}/{row["personality"]}')
            baseline = next(r for r in compared if r['personality'] == 'balanced')
            base_cycles = sum(step['cycles'] for step in baseline['counters'])
            measured = {r['personality'] + '/int8': sum(step['cycles'] for step in r['counters']) for r in compared}
            predicted = {r['personality'] + '/int8': predictor.predict(r)['cycles'] for r in compared}
            row_ids = sorted(identity(r) for r in compared)
            evidence[name + '/' + workload] = row_ids
            for fraction in COST_FRACTIONS:
                overhead = round(base_cycles * fraction)
                components = {
                    'drain': round(overhead * .10), 'program': round(overhead * .40),
                    'reload': round(overhead * .30), 'warmup': round(overhead * .10),
                }
                components['reprefill'] = overhead - sum(components.values())
                costs = {a + '->' + b: components for a in measured for b in measured if a != b}
                episode = []
                for objective in OBJECTIVES:
                    for horizon in HORIZONS:
                        episode.append({
                            'base_model_id': baseline['base_model_id'],
                            'family': 'LFM' if name == 'lfm' else 'Qwen',
                            'provenance': 'estimated-window-from-measured-rtl',
                            'evidence_ids': row_ids,
                            'features': [baseline['prompt_tokens'] / 2048,
                                         baseline['workload']['latency'] / 100,
                                         baseline['workload']['stall_percent'] / 100,
                                         baseline['workload']['bandwidth_percent'] / 100,
                                         baseline['config']['DRAM_BYTES'] / 2**30,
                                         horizon / 128],
                            'service_cycles': measured, 'predicted_cycles': predicted,
                            'predictor_id': predictor_id,
                            'predictor_training_models': predictor.training_models,
                            'window': {'remaining_requests': horizon, 'objective': objective,
                                       'switching_costs': costs, 'uncertainty_fraction': .10},
                        })
                episodes.append(episode)
    return episodes, evidence


def benchmark_comparisons(rows, predictor):
    """Retrospective ranks; measured held-out cycles never enter observations."""
    cases = {}
    for name, model_rows in rows.items():
        cases[name] = {}
        for workload in (*WORKLOADS, 'axi-stress'):
            compared = [r for r in model_rows if r['benchmark_workload'] == workload]
            measured = {r['personality']: sum(s['cycles'] for s in r['counters']) for r in compared}
            predicted = {r['personality']: predictor.predict(r)['cycles'] for r in compared}
            if 'balanced' not in measured:
                raise ValueError(f'missing balanced baseline: {name}/{workload}')
            best = min(measured, key=measured.get)
            predicted_best = min(predicted, key=predicted.get)
            cases[name][workload] = {
                'row_ids': sorted(identity(r) for r in compared),
                'measured_cycles': measured,
                'predicted_cycles': predicted,
                'retrospective_fastest': best,
                'predicted_fastest': predicted_best,
                'retrospective_gain_over_balanced_percent':
                    100 * (1 - measured[best] / measured['balanced']),
                'predicted_gain_over_balanced_percent':
                    100 * (1 - predicted[predicted_best] / predicted['balanced']),
                'eligible_for_four_action_episodes': workload in WORKLOADS,
            }
    return cases


def main():
    quality_ids = quality_check()
    rows = {name: load_verified(name) for name in NAMES}
    quality = published_quality()
    predictor = Predictor().fit(rows['qwen3'] + rows['qwen35'])
    source_ids = sorted(identity(r) for name in ('qwen3', 'qwen35') for r in rows[name])
    checkpoint = predictor.checkpoint(source_ids)
    predictor_id = identity(checkpoint)
    training, training_ids = episode_set(rows, predictor, predictor_id, ('qwen3', 'qwen35'), quality)
    heldout, heldout_ids = episode_set(rows, predictor, predictor_id, ('lfm',), quality)
    validate_release_episodes(training, 'training')
    validate_release_episodes(heldout, 'held-out')
    policy = train(training, seed=42, passes=20)
    # Episode construction and seed reproduce the replay; the deployed candidate
    # needs only the weights. Retain a digest without repeating thousands of
    # state vectors in a review artifact.
    policy['training_replay_id'] = identity(policy['replay'])
    policy['replay'] = []
    evaluation = evaluate(policy, heldout, seeds=range(100, 105), split='held-out',
                          leave_family_out=True, budget=4)
    for case in evaluation['cases']:
        # Every full window is reconstructed from the declared episode design;
        # its hash and each method's cost remain in the case record.
        del case['windows']
    output = {
        'schema_version': 1,
        'kind': 'offline-controller-continuation-from-published-rtl',
        'automatic_switching': 'gated',
        'quality_evaluation_ids': quality_ids,
        'predictor_checkpoint': checkpoint,
        'predictor_id': predictor_id,
        'predictor_scores': {name: summarize(model_rows, predictor) for name, model_rows in rows.items()},
        'benchmark_comparisons': benchmark_comparisons(rows, predictor),
        'benchmark_row_ids': {**training_ids, **heldout_ids},
        'episode_design': {'workloads': WORKLOADS, 'windows_per_episode': 8,
                           'objectives': OBJECTIVES, 'horizons': HORIZONS,
                           'switching_cost_fractions_of_balanced_service': COST_FRACTIONS,
                           'switching_cost_provenance': 'assumed; no physical transition measurements',
                           'request_sequence_provenance': 'synthetic repetitions of published fixed tapes'},
        'training_episode_hash': identity(training),
        'heldout_episode_hash': identity(heldout),
        'policy_id': identity(policy),
        'policy_evaluation': evaluation,
        'limitations': [
            'The LFM benchmark rows were inspected by the earlier published predictor audit; this reuse is exploratory, not untouched held-out policy evidence.',
            'All service cycles are fixed-token RTL tape measurements, not generated-token or physical FPGA timings.',
            'Request sequences and transition costs are assumed; no measured drain, program, reload, warmup or reprefill cost exists.',
            'No automatic switching promotion is allowed from this offline audit.',
        ],
    }
    target = ROOT / 'docs/evidence/controller-predictor-20261007'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'predictor-checkpoint.json').write_text(json.dumps(checkpoint, indent=2, allow_nan=False) + '\n')
    (target / 'policy-candidate.json').write_text(json.dumps(policy, indent=2, allow_nan=False) + '\n')
    (target / 'offline-evaluation.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictor_lfm_mape': output['predictor_scores']['lfm']['mean_absolute_percentage_error'],
                      'evaluation_mean_cost': evaluation['mean_cost'],
                      'per_case_regressions': len(evaluation['per_case_regressions'])}, indent=2))


if __name__ == '__main__':
    main()
