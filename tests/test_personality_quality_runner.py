"""Fail-closed dispatch checks for the new personality held-out campaign."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.run_personality_quality import command_for, freeze_design, report, run_quality
from tools.recover_completed_validation import logged_result


class PersonalityQualityRunnerTest(unittest.TestCase):
    def test_progress_transport_errors_do_not_fail_attempt(self):
        with patch('builtins.print', side_effect=OSError(5, 'detached PTY')):
            report({'completed_targets': 128})

    def test_heldout_command_requires_freeze_and_keeps_raw_candidate_distinct(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, suite, store = root / 'model', root / 'suite.json', root / 'store'
            with self.assertRaisesRegex(ValueError, 'frozen design'):
                command_for('held-out', model, suite, 'compact', store)
            raw = command_for('held-out', model, suite, 'compute', store,
                              freeze=root / 'freeze.json')
            derived = command_for('held-out', model, suite, 'compute', store,
                                  candidate_case=root / 'case', freeze=root / 'freeze.json')
            self.assertIn('quality', raw)
            self.assertNotIn('quality-candidate-validate', raw)
            self.assertNotIn('--candidate-case', raw)
            self.assertIn('--candidate-freeze', raw)
            self.assertIn('quality-candidate-validate', derived)
            self.assertIn('--candidate-case', derived)

    def test_balanced_is_validation_control_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            validation = command_for('validation', root / 'model', root / 'suite.json',
                                     'balanced', root / 'store')
            self.assertIn('balanced', validation)
            with self.assertRaisesRegex(ValueError, 'must not be repeated'):
                command_for('held-out', root / 'model', root / 'suite.json',
                            'balanced', root / 'store', freeze=root / 'freeze.json')

    def test_recovery_requires_exactly_one_completed_result_event(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'worker.log'
            path.write_text('noise\n' + json.dumps({'kind': 'result', 'payload': {'record_id': 'r'}}) + '\n')
            self.assertEqual(logged_result(path), {'record_id': 'r'})
            path.write_text(path.read_text() + json.dumps({'kind': 'result', 'payload': {}}) + '\n')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                logged_result(path)

    def test_heldout_attempt_never_restarts_after_prior_dispatch(self):
        for prior in ('job.json', 'heldout-claim.json', 'result.json', 'failure.json'):
            with self.subTest(prior=prior), TemporaryDirectory() as directory:
                root = Path(directory)
                stage = root / 'held-out'
                stage.mkdir()
                (stage / 'freeze.json').write_text('{}')
                (stage / prior).write_text('{}')
                args = SimpleNamespace(root=root, phase='held-out')
                with self.assertRaisesRegex(ValueError, 'already started'):
                    run_quality(args, {}, None, {})

    def test_failed_validation_cannot_be_frozen(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / 'validation'
            stage.mkdir()
            (stage / 'summary.json').write_text(json.dumps({'status': 'completed', 'passed': False}))
            (stage / 'result.json').write_text('{}')
            args = SimpleNamespace(root=root)
            with self.assertRaisesRegex(ValueError, 'passing hash-verified validation'):
                freeze_design(args, {}, None, {})
            self.assertFalse((root / 'held-out').exists())


if __name__ == '__main__':
    unittest.main()
