# Numeric contract

This document defines the first bit-accurate boundary between the RTL and ML
contributors. Code must not infer different behavior from a framework default.

## Version 0.1: MAC and dot product

| Quantity | Representation | Range |
| --- | --- | ---: |
| Activation | Signed two's-complement INT8 | -128 to 127 |
| Weight | Signed two's-complement INT8 | -128 to 127 |
| Product | Exact signed INT16 | -16256 to 16384 |
| Accumulator input | Signed two's-complement INT32 | -2147483648 to 2147483647 |
| Result | Signed two's-complement INT32 | -2147483648 to 2147483647 |

The MAC operation is:

```text
exact = accumulator_in + activation * weight
```

The dot-product operation is:

```text
exact = accumulator_in + sum(activation[lane] * weight[lane])
```

Products are sign-extended before addition. No scale, rounding, saturation, or
activation is applied in these blocks.

## Overflow

The model pipeline must normally prove that the accumulator cannot overflow for
the configured reduction length. RTL still reports overflow to make violations
observable.

If the exact result is outside INT32:

- `overflow` is asserted with the result.
- `result` contains the low 32 bits, interpreted as two's-complement.
- The future golden model must report the overflow explicitly.
- The future golden model must be able to reproduce the wrapped RTL result for
  verification, without treating overflow as a valid inference result.

This behavior makes overflow debuggable without silently accepting it as valid
model arithmetic.

## Timing

- Inputs are accepted on a rising clock edge when `valid_in` is high.
- The registered result and `valid_out` become visible immediately after that
  edge and remain associated with the accepted sample.
- When `valid_in` is low, `valid_out` is low and the previous result is retained.
- Synchronous active-high reset clears the result, validity, and overflow flag.

## Planned quantization boundary

The first software implementation should use symmetric signed INT8 values with
`zero_point = 0`. A real value is approximated by:

```text
real_value ~= integer_value * scale
```

Version 0.1 deliberately excludes requantization. Version 0.2 below fixes that
boundary for the dense-network RTL.

## Version 0.2: dense-layer output processing

The autonomous dense-layer engine retains the version 0.1 dot-product behavior
and applies these steps to every output:

```text
biased  = signed_int32_accumulator + signed_int32_bias
product = biased * unsigned_positive_31_bit_multiplier
scaled  = round_ties_away_from_zero(product / 2^right_shift)
output  = saturate_to_signed_int8(scaled)
output  = max(output, 0) when ReLU is enabled
```

- Bias addition uses a signed 33-bit intermediate.
- Multiplication and rounding use a signed 65-bit intermediate.
- The multiplier range is 1 through `2^31 - 1`.
- The right-shift range is 0 through 62.
- A zero shift performs no division or rounding.
- For a nonzero shift, exactly-halfway positive and negative magnitudes round
  away from zero.
- Saturation occurs before ReLU and clamps to `-128` through `127`.
- Bias, multiplier, and shift are independently programmable per output.

The multiplier and shift describe an exact integer operation, not a floating-
point approximation hidden in the RTL. Exporting software must choose values
that reproduce the intended real-valued scale and must use these same rounding
and saturation rules.

## Tiled reductions

Reductions longer than the configured lane count are split into consecutive
tiles. Missing lanes in the final tile are zero-filled. Each tile consumes the
wrapped INT32 result from the preceding tile. Any tile overflow sets a sticky
error for the network run, while execution continues with the documented
wrapped result so mismatches remain reproducible.
