"""Exploratory leave-model-out relative baseline on already inspected rows.

Fits median personality/balanced cycle ratios. No absolute cycle prediction,
confidence calibration, new inference, or promotion is performed.
"""
import json
import statistics
from collections import defaultdict

from tools.evaluate_exported_predictor import load_verified


def evaluate():
    paired = []
    for model in ('qwen3', 'qwen35', 'lfm'):
        workloads = defaultdict(list)
        for row in load_verified(model):
            workloads[row['benchmark_workload']].append(row)
        for workload, rows in workloads.items():
            balanced = next(r for r in rows if r['personality'] == 'balanced')
            baseline = sum(s['cycles'] for s in balanced['counters'])
            for row in rows:
                paired.append(dict(model=model, workload=workload, personality=row['personality'],
                                   ratio=sum(s['cycles'] for s in row['counters']) / baseline))
    folds = []
    for model in ('qwen3', 'qwen35', 'lfm'):
        training = [r for r in paired if r['model'] != model]
        estimates = {p: statistics.median(r['ratio'] for r in training if r['personality'] == p)
                     for p in {r['personality'] for r in training}}
        scored = [dict(r, predicted_ratio=estimates[r['personality']],
                       error_percentage_points=100 * abs(r['ratio'] - estimates[r['personality']]))
                  for r in paired if r['model'] == model and r['personality'] != 'balanced']
        folds.append(dict(excluded_model=model, estimates=estimates, samples=scored,
                          speedup_mae_percentage_points=statistics.mean(r['error_percentage_points'] for r in scored)))
    return dict(schema_version=1, provenance='exploratory-consumed-data-cross-validation',
                independent_evaluation=False, calibrated_intervals=False,
                automatic_switching_allowed=False, folds=folds)


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
