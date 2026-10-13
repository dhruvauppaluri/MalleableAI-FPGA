"""The selected pair cannot pass development by matching a damaged control."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_shared_matrix_finetune import development_gate, recipe_for_pair
from screen_shared_matrix_projections import choose_panel


class PairSelectionTests(unittest.TestCase):
    def test_only_original_and_screened_pairs_are_frozen(self):
        self.assertEqual((12, 13), (recipe_for_pair(12, 13)["source_layer"],
                                    recipe_for_pair(12, 13)["target_layer"]))
        with self.assertRaises(ValueError):
            recipe_for_pair(11, 12)

    def test_teacher_anchor_blocks_damaged_matched_students(self):
        recipe = recipe_for_pair(12, 13)
        metrics = {
            "teacher": {"nll": 3.7},
            "ordinary": {"nll": 7.0, "next_token_accuracy": 0.15},
            "shared": {"nll": 7.1, "next_token_accuracy": 0.15,
                       "teacher_agreement": 0.25},
        }
        checks = development_gate(metrics, recipe)
        self.assertTrue(checks["matched_control"])
        self.assertFalse(checks["teacher_anchor"])
        self.assertFalse(checks["passed"])

    def test_both_quality_comparisons_must_pass(self):
        recipe = recipe_for_pair(12, 13)
        metrics = {
            "teacher": {"nll": 3.7},
            "ordinary": {"nll": 3.8, "next_token_accuracy": 0.38},
            "shared": {"nll": 3.85, "next_token_accuracy": 0.375,
                       "teacher_agreement": 0.91},
        }
        self.assertTrue(development_gate(metrics, recipe)["passed"])
        metrics["shared"]["next_token_accuracy"] = 0.36
        self.assertFalse(development_gate(metrics, recipe)["passed"])

    def test_projection_screen_requires_teacher_preservation(self):
        gate = {"panel": "gate_proj", "nll": 3.8,
                "nll_change_percent": 2.0, "teacher_agreement": 0.89}
        up = {"panel": "up_proj", "nll": 3.85,
              "nll_change_percent": 3.0, "teacher_agreement": 0.91}
        self.assertEqual(choose_panel([gate, up])["panel"], "up_proj")
        self.assertIsNone(choose_panel([gate]))


if __name__ == "__main__":
    unittest.main()
