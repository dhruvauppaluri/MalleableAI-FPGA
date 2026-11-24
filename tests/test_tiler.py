import random
import shutil
import unittest
from types import SimpleNamespace

from malleable import tiler
from malleable.backend import RTLBackend, writes
from malleable.model import fixture, reference
from malleable.tiler import FitError, Limits

EXACT = dict(coefficients=dict(tiles=2.0, outputs=3.0, layers=0.0, constant=0.0))


def raw(dims, seed=0):
    rng = random.Random(seed)
    return [dict(weights=[[rng.randint(-128, 127) for _ in range(k)] for _ in range(n)],
                 biases=[0] * n, multipliers=[1] * n, shifts=[8] * n, relu=False)
            for k, n in zip(dims, dims[1:])]


class FitChecks(unittest.TestCase):
    def assertLimit(self, limit, model, lanes=4, limits=Limits()):
        with self.assertRaises(FitError) as ctx:
            tiler.fit_check(model, lanes, limits)
        self.assertEqual(ctx.exception.limit, limit)
        self.assertTrue(str(ctx.exception).startswith(limit + ':'))

    def test_each_limit_is_named(self):
        self.assertLimit('MAX_LAYERS', raw([4, 4, 4, 4, 4, 4]))
        self.assertLimit('MAX_DIM', raw([80, 4]))
        self.assertLimit('MAX_DIM', raw([4, 65]))
        self.assertLimit('LANES', raw([4, 4]), lanes=0)
        self.assertLimit('WEIGHT_DEPTH', raw([64, 64, 64]), limits=Limits(WEIGHT_DEPTH=5000))
        self.assertLimit('PARAM_DEPTH', raw([8, 64, 64]), limits=Limits(PARAM_DEPTH=100))
        bad = raw([4, 4]); bad[0]['multipliers'][1] = 0
        self.assertLimit('MULTIPLIERS', bad)
        bad = raw([4, 4]); bad[0]['shifts'][0] = 63
        self.assertLimit('SHIFTS', bad)
        bad = raw([4, 4]); bad[0]['biases'][0] = 2**31
        self.assertLimit('BIASES', bad)
        bad = raw([4, 4]); bad[0]['weights'][0][0] = 128
        self.assertLimit('WEIGHTS', bad)
        bad = raw([4, 3, 2]); bad[1]['weights'] = [[0] * 5 for _ in range(2)]
        self.assertLimit('ADJACENCY', bad)

    def test_message_names_the_value(self):
        with self.assertRaisesRegex(FitError, r'^MAX_DIM: layer 0 input_count 80 outside 1\.\.64$'):
            tiler.fit_check(raw([80, 4]), 4)

    def test_valid_model_passes(self):
        self.assertEqual(tiler.fit_check(raw([7, 5, 3]), 16).layers[0]['shifts'], [8] * 5)


class Selection(unittest.TestCase):
    def test_smallest_build_meeting_budget_and_monotone(self):
        model = fixture('medium')
        previous = None
        for budget in range(400, 20000, 250):
            try:
                chosen = tiler.plan_build(model, budget, calibration=EXACT, area={})['chosen']['lanes']
            except ValueError:
                continue
            predicted = {L: tiler.predict_cycles(model, L, EXACT) for L in range(1, 17)}
            self.assertLessEqual(predicted[chosen], budget)
            self.assertTrue(all(predicted[L] > budget for L in range(1, chosen)))
            if previous is not None:
                self.assertLessEqual(chosen, previous)  # looser budget never needs a bigger build
            previous = chosen

    def test_budget_equal_to_prediction_is_met(self):
        model = fixture('medium')
        exact = tiler.predict_cycles(model, 4, EXACT)
        self.assertEqual(tiler.plan_build(model, exact, calibration=EXACT, area={})['chosen']['lanes'], 4)

    def test_infeasible_budget_names_best_achievable(self):
        with self.assertRaisesRegex(ValueError, 'best achievable is 40.0 cycles at LANES=7'):
            # K <= 7, so LANES 8..16 only tie LANES=7; the smallest best build is named.
            tiler.plan_build(fixture('light'), 1, calibration=EXACT, area={})

    def test_report_fields_and_area_unavailable(self):
        report = tiler.plan_build(fixture('medium'), 10**6, calibration=EXACT, area={})
        chosen = report['chosen']
        self.assertEqual(chosen['lanes'], 1)
        self.assertEqual(chosen['build']['WEIGHT_DEPTH'], 4 * 64 * 64)
        self.assertTrue(chosen['area']['provenance'].startswith('unavailable'))
        self.assertEqual([r['lanes'] for r in report['runner_ups']], [2, 3, 4])
        self.assertGreaterEqual(chosen['padding_waste'], 0)
        self.assertIn('LANES', tiler.render_report(report))

    def test_padding_waste(self):
        self.assertEqual(tiler.padding_waste(raw([8, 2]), 4), 0)
        self.assertAlmostEqual(tiler.padding_waste(raw([5, 2]), 4), 1 - 5 / 8)


class Emission(unittest.TestCase):
    def test_stream_is_backend_format_plus_lane_write(self):
        model = fixture('light')
        stream = tiler.emit(model, 3)
        self.assertEqual(stream, writes(model) + [(6, 0, 3)])

    @unittest.skipIf(shutil.which('iverilog') is None, 'iverilog not installed')
    def test_emitted_stream_is_bit_exact_in_rtl_and_cycles_predicted(self):
        model = fixture('light')
        inputs = [[random.Random(i).randint(-128, 127) for _ in range(7)] for i in range(3)]
        for lanes in (3, 16):
            run = RTLBackend().run(model, inputs, SimpleNamespace(lanes=lanes, active_lanes=lanes, reuse=True),
                                   stream=tiler.emit(model, lanes))
            self.assertEqual(run['outputs'], [reference(model, x, lanes)[0] for x in inputs])
            self.assertEqual(run['counters'][0]['cycles'], tiler.predict_cycles(model, lanes, EXACT))

    def test_shipped_calibration_has_measured_tile_cost(self):
        k = tiler.load_calibration()['coefficients']
        self.assertAlmostEqual(k['tiles'], 2.0, places=6)
        self.assertAlmostEqual(k['outputs'], 3.0, places=6)


if __name__ == '__main__':
    unittest.main()
