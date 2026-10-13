import unittest

from malleable.llm.transition_measurement import aggregate
from malleable.llm.conservative_policy import COMPONENTS


def sample(index):
    stamps={}; cursor=0
    for name in sorted(COMPONENTS):
        stamps[name]=[cursor,cursor+index+1]; cursor+=index+1
    return dict(platform_id='board',context_id='empty',source_image_id='a',
                destination_image_id='b',clock_hz=100000000,driver_id='driver',
                trace_id=str(index),provenance='measured-physical-board-clock',stage_boundaries=stamps)


class TransitionTests(unittest.TestCase):
    def test_observed_upper_and_duplicate_rejection(self):
        result=aggregate([sample(i) for i in range(3)])
        self.assertEqual(result['sample_count'],3)
        self.assertFalse(result['tail_coverage_claim'])
        self.assertGreater(sum(result['components'].values()),result['observed_total_max'])
        with self.assertRaises(ValueError): aggregate([sample(0)]*3)

    def test_no_simulated_or_host_provenance(self):
        samples=[sample(i) for i in range(3)]
        samples[1]['provenance']='measured-host'
        with self.assertRaises(ValueError): aggregate(samples)
