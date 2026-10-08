import tempfile
import unittest

from malleable.llm.policy_campaign import claim
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
