"""Post-training quantizer: FP32 dense model -> numeric contract v0.2 artifact.

The output is a :class:`malleable.records.ModelArtifact` that
:func:`malleable.model.reference` (and the RTL) consume: INT8 weights, INT32
biases and a per-output ``multiplier / 2**shift`` pair.  Integer requantization
is never re-derived here; :func:`malleable.model.requantize` is the only oracle
for the contract's 33-bit bias add, 65-bit product, ties-away rounding and
saturate-before-ReLU behavior.  See ``docs/quantization.md`` for the method and
its limits.

Float model format (one dict per layer, applied in order)::

    {'weights': [[float] * inputs] * outputs,   # y = W x + b
     'biases':  [float] * outputs,              # optional, default zeros
     'relu':    bool,                           # optional, default False
     'batch_norm': {'gamma': [...], 'beta': [...], 'mean': [...],
                    'var': [...], 'eps': 1e-5}}  # optional, applied after W x + b

Batch norm is folded into the weights and bias in float before any scale is
computed.  Arithmetic is Python float (IEEE double) on the FP32 values.
"""
from dataclasses import dataclass, field
import math

from .model import requantize
from .records import ModelArtifact

INT8_MAX = 127
INT32_MIN, INT32_MAX = -2**31, 2**31 - 1
MULTIPLIER_MIN, MULTIPLIER_MAX = 1, 2**31 - 1
SHIFT_MIN, SHIFT_MAX = 0, 62
# A layer whose calibration range is exactly zero (e.g. a dead ReLU) gets the
# scale of a +/-1.0 range, so it stays representable instead of dividing by 0.
_EMPTY_RANGE = 1.0


def _round_away(value):
    """Round a finite float to the nearest integer, ties away from zero."""
    magnitude = math.floor(abs(value) + 0.5)
    return -magnitude if value < 0 else magnitude


def _finite(values, what):
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f'{what} contains a non-finite value: {v!r}')


# --------------------------------------------------------------------------
# Batch-norm folding
# --------------------------------------------------------------------------

def fold_batch_norm(layer):
    """Return ``(weights, biases)`` with any batch norm folded in, in float.

    BN(z) = gamma * (z - mean) / sqrt(var + eps) + beta with z = W x + b, so
    W' = s * W (row-wise) and b' = s * (b - mean) + beta, s = gamma / sqrt(var + eps).
    """
    weights = [[float(w) for w in row] for row in layer['weights']]
    n = len(weights)
    biases = [float(b) for b in layer.get('biases') or [0.0] * n]
    if len(biases) != n:
        raise ValueError('Bias count does not match output count')
    bn = layer.get('batch_norm')
    if bn is None:
        return weights, biases
    eps = float(bn.get('eps', 1e-5))
    for key in ('gamma', 'beta', 'mean', 'var'):
        if len(bn[key]) != n:
            raise ValueError(f'batch_norm {key} count does not match output count')
    folded_w, folded_b = [], []
    for j in range(n):
        var = float(bn['var'][j])
        if var + eps <= 0:
            raise ValueError('batch_norm var + eps must be positive')
        s = float(bn['gamma'][j]) / math.sqrt(var + eps)
        folded_w.append([s * w for w in weights[j]])
        folded_b.append(s * (biases[j] - float(bn['mean'][j])) + float(bn['beta'][j]))
    return folded_w, folded_b


def float_forward(folded_layers, x):
    """Float forward pass of already-folded layers; returns every layer output."""
    outputs = []
    for weights, biases, relu in folded_layers:
        x = [sum(w * a for w, a in zip(row, x)) + b for row, b in zip(weights, biases)]
        if relu:
            x = [max(0.0, v) for v in x]
        outputs.append(x)
    return outputs


# --------------------------------------------------------------------------
# Scales
# --------------------------------------------------------------------------

def symmetric_scale(max_abs):
    """Symmetric INT8 scale (zero point 0) covering [-max_abs, max_abs]."""
    if not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError(f'Invalid range {max_abs!r}')
    return (max_abs if max_abs > 0 else _EMPTY_RANGE) / INT8_MAX


def quantize_multiplier(real_scale):
    """Represent ``real_scale`` as ``multiplier / 2**shift`` per contract v0.2.

    The multiplier is normalized into [2**30, 2**31 - 1] whenever the shift
    range allows it, giving a relative error of at most 2**-31.  Outside that:

    * ``real_scale >= 2**31 - 0.5``: clipped to (2**31 - 1, 0).  Any nonzero
      biased accumulator saturates either way, so outputs are unchanged.
    * ``real_scale < 2**-32``: shift is pinned at 62 and the multiplier loses
      leading bits (minimum 1).  |biased| * real_scale stays below 2 here, so
      the output error is bounded by one LSB.
    """
    real_scale = float(real_scale)
    if not math.isfinite(real_scale) or real_scale <= 0:
        raise ValueError(f'Requantization scale must be positive and finite, got {real_scale!r}')
    fraction, exponent = math.frexp(real_scale)  # real = fraction * 2**exponent, fraction in [0.5, 1)
    shift = 31 - exponent
    if shift < SHIFT_MIN:
        return MULTIPLIER_MAX, SHIFT_MIN
    if shift > SHIFT_MAX:
        return max(MULTIPLIER_MIN, _round_away(math.ldexp(real_scale, SHIFT_MAX))), SHIFT_MAX
    multiplier = _round_away(math.ldexp(fraction, 31))
    if multiplier > MULTIPLIER_MAX:  # fraction rounded up to 1.0
        if shift == SHIFT_MIN:
            return MULTIPLIER_MAX, SHIFT_MIN
        multiplier, shift = multiplier >> 1, shift - 1
    return multiplier, shift


def quantize_values(values, scale, low=-128, high=127):
    """Quantize reals with ``round_ties_away(v / scale)`` clamped to [low, high]."""
    return [min(high, max(low, _round_away(v / scale))) for v in values]


def requantize_output(acc, bias, real_scale, relu):
    """Requantize one accumulator for an intended real scale via the oracle."""
    multiplier, shift = quantize_multiplier(real_scale)
    return requantize(acc, bias, multiplier, shift, relu)


# --------------------------------------------------------------------------
# Model quantization
# --------------------------------------------------------------------------

@dataclass
class QuantizedModel:
    """Integer artifact plus the real scales needed to map in and out of it."""
    artifact: ModelArtifact
    input_scale: float
    output_scales: list            # activation scale of each layer's output
    weight_scales: list            # per-layer list of per-output-channel scales
    real_multipliers: list         # per-layer list of s_in * s_w / s_out
    clipped_biases: list = field(default_factory=list)  # (layer, output) pairs

    def quantize_input(self, x):
        _finite(x, 'input')
        return quantize_values([float(v) for v in x], self.input_scale)

    def dequantize_output(self, q):
        return [v * self.output_scales[-1] for v in q]


def _max_abs(rows):
    return max((abs(v) for row in rows for v in row), default=0.0)


def calibrate(folded_layers, calibration_inputs):
    """Min/max calibration: symmetric range of the input and each layer output."""
    if not calibration_inputs:
        raise ValueError('Calibration needs at least one representative input')
    width = len(folded_layers[0][0][0])
    in_max, out_max = 0.0, [0.0] * len(folded_layers)
    for x in calibration_inputs:
        x = [float(v) for v in x]
        if len(x) != width:
            raise ValueError('Calibration input shape mismatch')
        _finite(x, 'calibration input')
        in_max = max(in_max, _max_abs([x]))
        for i, y in enumerate(float_forward(folded_layers, x)):
            _finite(y, f'layer {i} calibration output')
            out_max[i] = max(out_max[i], _max_abs([y]))
    return symmetric_scale(in_max), [symmetric_scale(m) for m in out_max]


def quantize_model(float_layers, calibration_inputs):
    """Quantize an FP32 dense model into a contract v0.2 ``QuantizedModel``.

    Steps: fold batch norm (float) -> min/max calibration of activations ->
    per-output-channel symmetric weight scales -> INT8 weights, INT32 biases at
    scale s_in * s_w -> multiplier/shift for s_in * s_w / s_out.  Biases that
    exceed INT32 are clipped and reported in ``clipped_biases``.
    """
    if not float_layers:
        raise ValueError('Model has no layers')
    folded = []
    for layer in float_layers:
        weights, biases = fold_batch_norm(layer)
        for row in weights:
            _finite(row, 'weights')
        _finite(biases, 'biases')
        folded.append((weights, biases, bool(layer.get('relu', False))))

    input_scale, output_scales = calibrate(folded, calibration_inputs)

    layers, weight_scales, real_multipliers, clipped = [], [], [], []
    s_in = input_scale
    for i, (weights, biases, relu) in enumerate(folded):
        s_out = output_scales[i]
        q_w, q_b, mults, shifts, s_ws, reals = [], [], [], [], [], []
        for j, (row, bias) in enumerate(zip(weights, biases)):
            s_w = symmetric_scale(_max_abs([row]))
            q_w.append(quantize_values(row, s_w, -INT8_MAX, INT8_MAX))
            b = _round_away(bias / (s_in * s_w))
            if not INT32_MIN <= b <= INT32_MAX:
                clipped.append((i, j))
                b = min(INT32_MAX, max(INT32_MIN, b))
            q_b.append(b)
            real = s_in * s_w / s_out
            m, sh = quantize_multiplier(real)
            mults.append(m); shifts.append(sh); s_ws.append(s_w); reals.append(real)
        layers.append(dict(weights=q_w, biases=q_b, multipliers=mults,
                           shifts=shifts, relu=relu))
        weight_scales.append(s_ws)
        real_multipliers.append(reals)
        s_in = s_out
    return QuantizedModel(ModelArtifact(layers), input_scale, output_scales,
                          weight_scales, real_multipliers, clipped)
