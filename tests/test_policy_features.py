import unittest

from malleable.llm.policy_features import FEATURES, features


class FeatureTests(unittest.TestCase):
    def test_architecture_changes_observation(self):
        record = dict(schema_version=1, provenance='compiler-pre-execution',
                      model_id='m', configuration_id='c', compiler_id='compiler',
                      workload_id='w', features={k: 1 for k in FEATURES})
        before = features(record)
        record['features']['hidden_size'] = 1024
        self.assertNotEqual(before, features(record))

    def test_missing_features_and_legacy_schema_rejected(self):
        for record in ({'schema_version': 2}, {'schema_version': 1, 'features': {}}):
            with self.assertRaises(ValueError):
                features(record)
