# Dense-model quantizer (v1)

`malleable/quantize.py` converts an FP32 dense model into the integer
`ModelArtifact` that `malleable/model.py` and the RTL consume under
[numeric contract v0.2](numeric-contract.md). It does not change the contract;
integer requantization always goes through `model.requantize()`.

## Method

1. **Batch-norm folding (float, first).** For `BN(Wx + b)`,
   `s = gamma / sqrt(var + eps)`, `W' = s * W` per row and
   `b' = s * (b - mean) + beta`. All later scales are computed on `W'` and `b'`.
2. **Activation scales: min/max calibration.** The folded float model runs over
   the representative inputs. The input and every layer output (after ReLU when
   enabled) get a symmetric scale `max|x| / 127`, zero point 0.
3. **Weight scales: per output channel.** `s_w[j] = max|W'[j]| / 127`; weights
   are `round_ties_away(W' / s_w)` clamped to `[-127, 127]` (symmetric; `-128`
   is never produced).
4. **Bias.** `round_ties_away(b' / (s_in * s_w[j]))` in INT32, so it adds
   directly to the accumulator. Out-of-range biases are clipped and listed in
   `QuantizedModel.clipped_biases`.
5. **Multiplier/shift.** The real ratio `M = s_in * s_w[j] / s_out` is written
   as `multiplier / 2^shift` with the multiplier normalized into
   `[2^30, 2^31 - 1]`. The relative error is at most `2^-31` for
   `2^-32 <= M < 2^31`. Outside that window the shift range `[0, 62]` binds:
   large `M` clips to `(2^31 - 1, 0)`, which gives the same outputs because any
   nonzero biased accumulator saturates anyway. Tiny `M` pins `shift = 62` and
   loses multiplier bits, with at most one LSB of output error.

`QuantizedModel` returns the artifact plus `input_scale`, the per-layer
`output_scales`, `weight_scales` and the exact real multipliers.
`quantize_input()` and `dequantize_output()` map between real values and the
integer boundary.

## Known limits

- **Min/max is outlier-sensitive.** A single extreme calibration value
  stretches the scale and wastes resolution. Percentile, MSE or entropy
  calibration is not implemented. Inputs beyond the calibration range saturate.
- **Symmetric only.** Contract v0.2 has no zero point, so post-ReLU activations
  use only `[0, 127]`, which costs one bit of resolution.
- **Calibration uses float activations,** not the quantized pipeline's own
  activations, so error that builds up across layers is not corrected.
- **No accuracy guarantee.** The tests bound error on small random models only:
  mean below 1.5 output LSBs and max below 6. Real models need their own
  validation set.
- **Arithmetic is float64** on FP32 inputs, so results are deterministic but can
  differ in the last ulp from an FP32-only implementation.
- **Bounded by `ModelArtifact`.** The model can have 1 to 4 layers and each
  dimension can be at most 64. Dead layers (zero calibration range) get the
  scale of a ±1.0 range.
