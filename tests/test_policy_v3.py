import unittest

from malleable.llm.policy_learning_v3 import ACTIONS, evaluate, legal, observation, train


def window(program=10, horizon=8):
    pairs = {a+'->'+b: dict(drain=0,program=program,reload=0,warmup=0,reprefill=0)
             for a in ACTIONS for b in ACTIONS if a != b}
    return dict(remaining_requests=horizon, objective='throughput', switching_costs=pairs)


def step(program=10, horizon=8, measured_compute=60):
    return dict(window=window(program,horizon), predicted=dict(compact=105,balanced=100,compute=70,buffered=102),
                measured=dict(compact=105,balanced=100,compute=measured_compute,buffered=102),
                eligible=list(ACTIONS), evidence_ids=['synthetic-unit-test'], predictor_id='synthetic')


class LearningV3Tests(unittest.TestCase):
    def test_cost_observation_and_mask(self):
        a,b=step(program=10),step(program=500)
        self.assertNotEqual(observation(a,'balanced',1).tolist(),observation(b,'balanced',1).tolist())
        self.assertIn(ACTIONS.index('compute'),legal(a,'balanced',1)[0])
        self.assertNotIn(ACTIONS.index('compute'),legal(b,'balanced',1)[0])

    def test_executed_audit_and_five_training_seeds(self):
        episode=[step(program=500)]*2
        models=[train([episode],seed=seed,passes=2) for seed in range(5)]
        self.assertEqual(len({m['seed'] for m in models}),5)
        for model in models:
            self.assertEqual(model['executed_switch_count'],0)
            self.assertTrue(all(a['executed']=='balanced' for a in model['audit']))
            self.assertEqual(evaluate(model,[episode],method='dqn')[0]['actions'],['balanced']*2)

    def test_baselines_only_switch_when_profitable(self):
        episode=[step()]
        results={m:evaluate(None,[episode],method=m)[0] for m in ('fixed','deterministic','oracle')}
        self.assertEqual(results['fixed']['actions'],['balanced'])
        self.assertEqual(results['deterministic']['actions'],['compute'])
        self.assertEqual(results['oracle']['actions'],['compute'])

    def test_legacy_checkpoint_rejected(self):
        with self.assertRaises(ValueError): evaluate(dict(schema_version=2),[[step()]])
