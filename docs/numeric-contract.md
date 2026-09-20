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

Requantization is deliberately not part of version 0.1. Before adding it, the
project will specify the integer multiplier representation, shift direction,
rounding tie rule, saturation, bias scale, and activation order before either
the software or RTL implementation is accepted.
