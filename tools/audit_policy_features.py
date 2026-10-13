"""Read-only diagnostic of consumed published evidence; no model evaluation."""
import json
from collections import defaultdict

from malleable.llm.optimization import Predictor
from malleable.records import identity
from tools.evaluate_exported_predictor import load_verified


def audit():
    groups = defaultdict(list)
    rows = []
    for name in ('qwen3', 'qwen35', 'lfm'):
        for row in load_verified(name):
            rows.append(row)
            groups[tuple(Predictor.features(row))].append(dict(
                model=name, evidence_id=identity(row), workload=row['benchmark_workload'],
                personality=row['personality'], cycles=sum(s['cycles'] for s in row['counters'])))
    collisions = [members for members in groups.values()
                  if len({r['model'] for r in members}) > 1]
    return dict(schema_version=1, provenance='exploratory-analysis-of-consumed-evidence',
                row_count=len(rows), evidence_ids=[identity(r) for r in rows],
                unique_feature_vectors=len(groups), cross_model_collisions=collisions,
                independent_holdout=False, new_rtl_runs=0, automatic_switching_allowed=False)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, sort_keys=True))
