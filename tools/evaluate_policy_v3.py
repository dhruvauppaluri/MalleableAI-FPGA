"""Exploratory five-training-seed audit of published, previously inspected rows.

Reuses the published predictor and exactly-once quality evidence. Does not open
a new held-out split or rerun RTL. Transition scenarios are assumptions.
"""
import json
import statistics
from pathlib import Path

from malleable.llm.optimization import Predictor
from malleable.llm.policy_learning_v3 import evaluate, train
from malleable.records import identity
from tools.evaluate_controller_continuation import episode_set, published_quality
from tools.evaluate_exported_predictor import load_verified

ROOT=Path(__file__).resolve().parents[1]
PUBLISHED=ROOT/'docs/evidence/controller-predictor-20261007/predictor-checkpoint.json'


def converted(episodes):
    result=[]
    for episode in episodes:
        result.append([dict(window=w['window'],predicted={k.split('/')[0]:v for k,v in w['predicted_cycles'].items()},
                            measured={k.split('/')[0]:v for k,v in w['service_cycles'].items()},
                            eligible=['compact','balanced','compute','buffered'],
                            evidence_ids=w['evidence_ids'],predictor_id=w['predictor_id'])
                       | {'window': dict(w['window'], switching_costs={
                            a.split('/')[0]+'->'+b.split('/')[0]:c
                            for pair,c in w['window']['switching_costs'].items()
                            for a,b in [pair.split('->')]})} for w in episode])
    return result


def run():
    checkpoint=json.loads(PUBLISHED.read_text())
    predictor=Predictor(); predictor.coefficients=checkpoint['coefficients']
    predictor.rmse=checkpoint['training_rmse_cycles']; predictor.training_models=checkpoint['training_models']
    rows={name:load_verified(name) for name in ('qwen3','qwen35','lfm')}
    quality=published_quality()
    training,_=episode_set(rows,predictor,identity(checkpoint),('qwen3','qwen35'),quality)
    scoring,_=episode_set(rows,predictor,identity(checkpoint),('lfm',),quality)
    train_eps,score_eps=converted(training),converted(scoring)
    summaries=[]
    fixed=evaluate(None,score_eps,method='fixed')
    baselines={method:evaluate(None,score_eps,method=method)
               for method in ('deterministic','predictor','oracle')}
    for seed in range(5):
        model=train(train_eps,seed=seed,passes=10)
        scores=evaluate(model,score_eps)
        summaries.append(dict(seed=seed,checkpoint_id=identity({k:v for k,v in model.items() if k!='audit'}),
            proposal_count=model['proposal_count'],rejection_count=model['rejection_count'],
            executed_switch_count=model['executed_switch_count'],
            mean_relative_cost=statistics.mean(a['total_cost']/b['total_cost'] for a,b in zip(scores,fixed)),
            score_paths=[r['actions'] for r in scores]))
    return dict(schema_version=1,provenance='exploratory-reuse-of-consumed-published-RTL',
                independent_evaluation=False,physical_transition_costs=False,
                training_episode_count=len(train_eps),scoring_episode_count=len(score_eps),
                training_seeds=summaries,baseline_mean_relative_costs={method:statistics.mean(
                    a['total_cost']/b['total_cost'] for a,b in zip(results,fixed))
                    for method,results in baselines.items()},
                automatic_switching_allowed=False,new_rtl_runs=0)


if __name__=='__main__': print(json.dumps(run(),sort_keys=True,indent=2))
