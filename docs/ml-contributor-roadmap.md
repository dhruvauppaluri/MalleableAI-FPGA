# ML and software contributor roadmap

## Mission

Build the complete software side of MalleableAI-FPGA independently. The first
goal is a bit-accurate golden model for a tiny signed-INT8 dense network. Later,
extend it into a model preparation, export, analysis, deployment, and benchmark
toolchain.

The RTL team provides the arithmetic and interface requirements. The ML
contributor chooses the software structure, dependencies, APIs, and internal
implementation.

## Required development

### 1. Project and test structure

- Choose and document the supported Python version and dependencies.
- Create a maintainable package, command-line entry points, and automated tests.
- Use fixed seeds and deterministic outputs wherever practical.
- Add formatting, linting, type checking, and tests to CI.

### 2. Bit-accurate integer primitives

- Implement signed two's-complement range checks and conversions.
- Implement the signed INT8 multiply and signed INT32 accumulate contract.
- Implement arbitrary-length dot products and tiled partial accumulation.
- Detect overflow explicitly and reproduce RTL output bits for debugging.
- Cross-check boundary values including `-128`, `127`, INT32 minimum, and INT32
  maximum.

Acceptance criterion: software and RTL agree bit-for-bit on directed boundary
cases and at least 10,000 deterministic randomized cases.

### 3. Quantization specification

- Begin with symmetric signed INT8 quantization and `zero_point = 0`.
- Define activation, weight, bias, and output scales.
- Decide whether weight scales are per-tensor or per-channel.
- Specify integer multiplier representation and shifts for requantization.
- Specify the rounding tie rule, saturation behavior, and activation order.
- Record every choice in `docs/numeric-contract.md` before depending on it.

### 4. Golden neural-network inference

- Implement a dense layer using integer arithmetic only.
- Add bias, requantization, signed INT8 saturation, and optional ReLU.
- Compose layers into a tiny end-to-end network.
- Expose intermediate layer results for RTL debugging.
- Keep a floating-point comparison path for measuring quantization error, but
  never use floating point inside the integer golden result.

### 5. Training or model import

- Train a small reproducible model or import one from a documented framework.
- Start with a tiny dense classifier that is fast to test.
- Save training configuration, dataset version, preprocessing, metrics, and
  random seeds.
- Measure floating-point accuracy before quantization and integer accuracy after
  quantization.

### 6. Artifact formats and export

Define versioned, machine-readable formats for:

- Model topology and layer dimensions
- Tensor shapes, layouts, signedness, and byte order
- Weights and biases
- Activation and weight scales
- Quantization parameters
- Input vectors
- Expected outputs and intermediate layer values
- Accelerator requirements

Export both human-readable metadata and compact binary or hexadecimal files
that RTL simulation and the future host runtime can consume directly.

### 7. RTL cross-verification

- Generate deterministic vectors from the golden model.
- Feed the same vectors into SystemVerilog simulation.
- Compare MAC, dot-product, tiled-layer, and full-network results automatically.
- Produce useful mismatch reports showing the layer, index, input values,
  expected bits, actual bits, and relevant scale.
- Make cross-verification a required CI job once stable.

### 8. Model and resource analyzer

- Count operations, parameters, activation sizes, and weight bytes per layer.
- Determine safe accumulator widths from reduction dimensions and value bounds.
- Estimate compute cycles for candidate MAC-lane counts.
- Model DSP, logic, on-chip memory, and external-memory bandwidth constraints.
- Recommend candidate lane counts, tile shapes, buffer sizes, precision, and
  dataflow.
- Keep estimates separate from measured Quartus and board results.

### 9. Experiment and benchmark records

- Define a reproducible record for model, hardware configuration, tool version,
  bitstream, clock frequency, resource use, latency, throughput, and accuracy.
- Import Quartus reports rather than manually copying values where practical.
- Preserve raw measurements alongside summarized results.

### 10. Host integration

- Package model artifacts for the host runtime.
- Transfer configuration, weights, inputs, and tiles to the FPGA.
- Read results and compare them with the golden model.
- Support automated benchmarking across precompiled accelerator personalities.

## Recommended milestone order

1. Bit-accurate MAC and dot-product reference
2. Shared RTL/software vector tests
3. Quantization and requantization specification
4. One integer dense layer with bias and ReLU
5. Tiny end-to-end dense network
6. Model import or training and artifact export
7. Model/resource analyzer
8. Host deployment and benchmark automation

## First handoff to the RTL contributor

The first software handoff should contain:

- A short numeric specification for review
- Deterministic MAC and dot-product vectors
- Boundary and overflow cases
- A documented file format readable from SystemVerilog
- Automated software tests
- Instructions for reproducing every generated artifact

Do not begin the full optimizer until this arithmetic handoff agrees with the
existing RTL bit-for-bit.
