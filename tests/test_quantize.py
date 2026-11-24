import math
import random
import unittest
from fractions import Fraction

from malleable.model import reference, requantize
from malleable.quantize import (
    MULTIPLIER_MAX, SHIFT_MAX, fold_batch_norm, float_forward, quantize_model,
    quantize_multiplier, quantize_values, requantize_output)
from malleable.records import ModelArtifact


def ideal(biased, real_scale, relu):
    """Exact real-valued requantization with ties away from zero (test oracle)."""
    exact = Fraction(biased) * Fraction(real_scale)
    magnitude = math.floor(abs(exact) + Fraction(1, 2))
    value = min(127, max(-128, -magnitude if exact < 0 else magnitude))
    return max(0, value) if relu else value


def distance_to_tie(value):
    """Distance of a rational value from the nearest rounding boundary (k + 1/2)."""
    frac = abs(value) - math.floor(abs(value))
    return abs(frac - Fraction(1, 2))


def random_scale(rng, low=-40, high=34):
    return 2.0 ** rng.uniform(low, high)


class MultiplierProperties(unittest.TestCase):
    def test_multiplier_and_shift_ranges(self):
        rng = random.Random(1)
        for _ in range(5000):
            m, s = quantize_multiplier(random_scale(rng, -80, 40))
            self.assertTrue(1 <= m <= MULTIPLIER_MAX and 0 <= s <= SHIFT_MAX, (m, s))
            self.assertIs(type(m), int)
            self.assertIs(type(s), int)

    def test_relative_error_in_normal_range(self):
        rng = random.Random(2)
        for _ in range(5000):
            real = random_scale(rng, -32, 30.9)
            m, s = quantize_multiplier(real)
            self.assertGreaterEqual(m, 2**30)  # fully normalized
            err = abs(Fraction(m, 2**s) - Fraction(real)) / Fraction(real)
            self.assertLessEqual(err, Fraction(1, 2**31))

    def test_powers_of_two_are_exact(self):
        for e in range(-32, 31):
            m, s = quantize_multiplier(2.0 ** e)
            self.assertEqual(Fraction(m, 2**s), Fraction(2) ** e)

    def test_rejects_invalid_scales(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                quantize_multiplier(bad)

    def test_out_of_range_scales_clip(self):
        self.assertEqual(quantize_multiplier(2.0**40), (MULTIPLIER_MAX, 0))
        m, s = quantize_multiplier(2.0**-70)
        self.assertEqual((m, s), (1, SHIFT_MAX))


class RequantizeProperties(unittest.TestCase):
    """Quantized outputs must equal model.requantize() and the intended real scale."""

    def test_matches_oracle_and_real_scale(self):
        rng = random.Random(3)
        for _ in range(20000):
            acc = rng.randint(-2**31, 2**31 - 1)
            bias = rng.choice([0, rng.randint(-2**31, 2**31 - 1), rng.randint(-1000, 1000)])
            relu = rng.random() < 0.5
            biased = acc + bias  # 33-bit signed intermediate
            # Choose scales that put outputs around the INT8 range, plus extremes.
            target = rng.choice([rng.uniform(0.1, 300), rng.uniform(1e-6, 1e6)])
            real = target / max(1, abs(biased))
            m, s = quantize_multiplier(real)
            got = requantize_output(acc, bias, real, relu)
            self.assertEqual(got, requantize(acc, bias, m, s, relu))
            want = ideal(biased, real, relu)
            # The represented scale m/2^s differs from `real` by a tiny amount;
            # results may only differ when that moves a value across a tie.
            slack = abs(biased) * abs(Fraction(m, 2**s) - Fraction(real))
            if distance_to_tie(Fraction(biased) * Fraction(real)) > slack:
                self.assertEqual(got, want, (acc, bias, real, relu))
            else:
                self.assertLessEqual(abs(got - want), 1)

    def test_exact_ties_round_away_and_saturate_before_relu(self):
        cases = [  # (biased, scale, relu, expected)
            (3, 0.5, False, 2), (-3, 0.5, False, -2), (5, 0.25, False, 1),
            (-5, 0.25, False, -1), (6, 0.25, False, 2), (-6, 0.25, False, -2),
            (1000, 0.5, False, 127), (-1000, 0.5, False, -128),
            (-1000, 0.5, True, 0), (1000, 0.5, True, 127), (-3, 0.5, True, 0),
            (7, 1.0, False, 7), (-2**32, 2.0**-62, False, 0),
        ]
        for biased, scale, relu, expected in cases:
            acc, bias = biased // 2, biased - biased // 2
            self.assertEqual(requantize_output(acc, bias, scale, relu), expected,
                             (biased, scale, relu))
            self.assertEqual(ideal(biased, scale, relu), expected)

    def test_value_quantization_ties_away(self):
        self.assertEqual(quantize_values([0.5, -0.5, 1.5, -1.5, 1000, -1000], 1.0),
                         [1, -1, 2, -2, 127, -128])


def random_float_model(rng, dims, bn_layers=(0,)):
    layers = []
    for i, (k, n) in enumerate(zip(dims, dims[1:])):
        layer = dict(weights=[[rng.gauss(0, 1 / math.sqrt(k)) for _ in range(k)] for _ in range(n)],
                     biases=[rng.gauss(0, 0.1) for _ in range(n)],
                     relu=i < len(dims) - 2)
        if i in bn_layers:
            layer['batch_norm'] = dict(gamma=[rng.uniform(0.5, 2) for _ in range(n)],
                                       beta=[rng.gauss(0, 0.2) for _ in range(n)],
                                       mean=[rng.gauss(0, 0.3) for _ in range(n)],
                                       var=[rng.uniform(0.2, 3) for _ in range(n)],
                                       eps=1e-5)
        layers.append(layer)
    return layers


def unfolded_forward(layers, x):
    for layer in layers:
        y = [sum(w * a for w, a in zip(row, x)) + b
             for row, b in zip(layer['weights'], layer['biases'])]
        bn = layer.get('batch_norm')
        if bn:
            y = [g * (v - m) / math.sqrt(var + bn['eps']) + be
                 for v, g, be, m, var in zip(y, bn['gamma'], bn['beta'], bn['mean'], bn['var'])]
        x = [max(0.0, v) for v in y] if layer['relu'] else y
    return x


class QuantizeModelIntegration(unittest.TestCase):
    def test_batch_norm_fold_matches_unfolded(self):
        rng = random.Random(4)
        layers = random_float_model(rng, [6, 5, 4], bn_layers=(0, 1))
        folded = [(*fold_batch_norm(l), l['relu']) for l in layers]
        for _ in range(20):
            x = [rng.gauss(0, 1) for _ in range(6)]
            for a, b in zip(float_forward(folded, x)[-1], unfolded_forward(layers, x)):
                self.assertAlmostEqual(a, b, places=9)

    def test_small_model_runs_through_reference(self):
        rng = random.Random(5)
        dims = [8, 12, 10, 4]
        layers = random_float_model(rng, dims)
        calibration = [[rng.gauss(0, 1) for _ in range(dims[0])] for _ in range(256)]
        q = quantize_model(layers, calibration)

        # Valid contract v0.2 artifact.
        ModelArtifact(q.artifact.layers)
        self.assertEqual(q.clipped_biases, [])
        for layer, reals in zip(q.artifact.layers, q.real_multipliers):
            for m, s, real in zip(layer['multipliers'], layer['shifts'], reals):
                self.assertEqual((m, s), quantize_multiplier(real))

        lsb = q.output_scales[-1]
        errors = []
        for _ in range(200):
            x = [rng.gauss(0, 1) for _ in range(dims[0])]
            out, intermediates, overflow = reference(q.artifact, q.quantize_input(x))
            self.assertFalse(overflow)
            self.assertEqual(len(intermediates), len(dims) - 1)
            want = unfolded_forward(layers, x)
            # Skip inputs outside the calibration range (they legitimately saturate).
            if any(abs(v) > 127 * lsb for v in want):
                continue
            errors += [abs(a - b) / lsb for a, b in zip(q.dequantize_output(out), want)]
        self.assertGreater(len(errors), 400)
        self.assertLess(sum(errors) / len(errors), 1.5)  # mean error in output LSBs
        self.assertLess(max(errors), 6)

    def test_bit_exact_against_oracle_per_layer(self):
        """Each layer's integer output equals requantize() on its exact accumulator."""
        rng = random.Random(6)
        layers = random_float_model(rng, [5, 7, 3])
        q = quantize_model(layers, [[rng.gauss(0, 1) for _ in range(5)] for _ in range(64)])
        x = q.quantize_input([rng.gauss(0, 1) for _ in range(5)])
        _, intermediates, _ = reference(q.artifact, x, active_lanes=3)
        values = x
        for layer, got in zip(q.artifact.layers, intermediates):
            want = [requantize(sum(a * w for a, w in zip(values, row)), b, m, s, layer['relu'])
                    for row, b, m, s in zip(layer['weights'], layer['biases'],
                                            layer['multipliers'], layer['shifts'])]
            self.assertEqual(got, want)
            values = got

    def test_rejects_bad_inputs(self):
        layers = [dict(weights=[[1.0, 2.0]], biases=[0.0])]
        with self.assertRaises(ValueError):
            quantize_model(layers, [])
        with self.assertRaises(ValueError):
            quantize_model(layers, [[1.0]])
        with self.assertRaises(ValueError):
            quantize_model([dict(weights=[[math.nan, 1.0]])], [[1.0, 1.0]])

    def test_dead_layer_and_zero_row(self):
        layers = [dict(weights=[[0.0, 0.0], [1.0, -1.0]], biases=[0.0, -100.0], relu=True)]
        q = quantize_model(layers, [[1.0, 0.5], [-1.0, 0.25]])
        out, _, _ = reference(q.artifact, q.quantize_input([1.0, 0.0]))
        self.assertEqual(out, [0, 0])


    def test_quantized_bn_model_is_bit_exact_in_rtl(self):
        """A quantized model (with folded BN) runs bit-exactly on the Icarus RTL."""
        import shutil
        if shutil.which('iverilog') is None:
            self.skipTest('iverilog not installed')
        from malleable.backend import RTLBackend
        from malleable.records import ExecutionConfig
        rng = random.Random(7)
        layers = random_float_model(rng, [9, 6, 3], bn_layers=(0, 1))
        q = quantize_model(layers, [[rng.gauss(0, 1) for _ in range(9)] for _ in range(64)])
        inputs = [q.quantize_input([rng.gauss(0, 1) for _ in range(9)]) for _ in range(4)]
        # RTLBackend.run raises on any RTL/golden mismatch.
        run = RTLBackend().run(q.artifact, inputs, ExecutionConfig(4, 4))
        self.assertEqual(run['outputs'], [reference(q.artifact, x, 4)[0] for x in inputs])

if __name__ == '__main__':
    unittest.main()
