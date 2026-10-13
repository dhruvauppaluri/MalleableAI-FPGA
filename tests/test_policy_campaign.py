import tempfile
import unittest

from malleable.llm.policy_campaign import claim, validate_manifest
from malleable.records import identity


class CampaignTests(unittest.TestCase):
    def test_failure_or_campaign_rename_cannot_repeat_claim(self):
        case = {'workload_digest': 'new-tape', 'model': 'm', 'personality': 'compute'}
        manifest = dict(schema_version=1, status='frozen', cases=[case])
        with tempfile.TemporaryDirectory() as root:
            claim(root, manifest, identity(case), source_id='source')
            with self.assertRaises(FileExistsError):
                claim(root, dict(manifest, name='retry'), identity(case), source_id='source')

    def test_unfrozen_campaign_rejected(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            claim(root, dict(schema_version=1, status='draft', cases=[]), 'unknown', source_id='source')

    def test_frozen_pilot_excludes_original_tapes(self):
        from tools.freeze_policy_pilot import manifest
        from tools.evaluate_exported_predictor import load_verified
        pilot=manifest('source')
        self.assertEqual(len(pilot['cases']),12)
        consumed=[r for name in ('qwen3','qwen35','lfm') for r in load_verified(name)]
        self.assertTrue(validate_manifest(pilot,consumed))
        pilot['cases'][0]['input_tokens']=consumed[0]['workload']['input_tokens']
        pilot['cases'][0]['input_token_hash']=consumed[0]['input_token_hash']
        with self.assertRaises(ValueError): validate_manifest(pilot,consumed)
