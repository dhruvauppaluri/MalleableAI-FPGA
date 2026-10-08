import unittest

from malleable.llm.conservative_policy import COMPONENTS, recommend
from malleable.llm.optimization import DecisionWindow


class ConservativePolicyTests(unittest.TestCase):
    def decision(self, *, overhead=10, horizon=8, provenance='measured-physical-upper-bound',
                 context='context', residence=1, intervals=None, eligible=None):
        transition = dict(provenance=provenance, platform_id='board', context_id=context,
                          evidence_id='synthetic-test-only', components={k: 0 for k in COMPONENTS})
        transition['components']['program'] = overhead
        return recommend('balanced', intervals or {'balanced': (100, 110), 'compute': (70, 80)},
                         {('balanced', 'compute'): transition}, eligible or {'balanced', 'compute'},
                         DecisionWindow(remaining_requests=horizon, objective='throughput',
                                        residence_windows=residence),
                         platform_id='board', context_id='context')

    def test_profitable_recommendation_never_activates(self):
        result = self.decision()
        self.assertEqual(result['recommended_action'], 'compute')
        self.assertFalse(result['automatic_switching_allowed'])

    def test_horizon_and_transition_cost_affect_decision(self):
        self.assertEqual(self.decision(overhead=100, horizon=1)['recommended_action'], 'balanced')
        self.assertEqual(self.decision(overhead=100, horizon=8)['recommended_action'], 'compute')

    def test_fail_closed(self):
        for kwargs, reason in [({'provenance': 'assumed'}, 'missing-measured-transition'),
                               ({'context': 'stale'}, 'transition-lineage-mismatch'),
                               ({'residence': 0}, 'minimum-residence'),
                               ({'eligible': {'balanced'}}, 'quality-or-capacity')]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.decision(**kwargs)['rejected']['compute'], reason)

    def test_overlapping_uncertainty_and_equal_margin_abstain(self):
        for upper in (105, 95):
            result = self.decision(overhead=0, intervals={'balanced': (100, 110), 'compute': (70, upper)})
            self.assertEqual(result['recommended_action'], 'balanced')

    def test_invalid_costs(self):
        for value in (float('nan'), float('inf'), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.decision(overhead=value)


if __name__ == '__main__':
    unittest.main()
