# Stage 2: first real model end to end (digits MLP)

Script: `tools/digits_e2e.py` (deterministic; a rerun reproduces the artifact
byte for byte). Data: `docs/results/digits/` (`digits-int8.json` quantized
artifact, `predictions.json`, `summary.json`).

## Setup

| Item | Value |
|---|---|
| Data | sklearn `load_digits` (bundled, 1797 images of 8×8, 10 classes) |
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=0)`: 1437 train, 360 test |
| FP32 model | `MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=1000, random_state=0)`; converged in 254 iterations |
| Topology | 64 → 32 (ReLU) → 10 (no ReLU) |
| Input preprocessing | none: raw pixels 0..16 in FP32; INT8 inputs come only from `QuantizedModel.quantize_input` (calibrated input scale 16/127 = 0.12598) |
| Quantizer | Stage 1 `quantize_model`, min/max calibration over the training split |
| INT8 prediction | argmax of the 10 INT8 output logits (lowest index wins a tie) |
| RTL | Icarus Verilog 12.0, `malleable_accelerator_top` LANES=4, active_lanes=4, via `RTLBackend` |

## Results

| Metric | Value | Provenance |
|---|---:|---|
| FP32 test accuracy | **96.67 %** (348/360) | measured (host FP32) |
| INT8 test accuracy (golden `model.reference`) | **96.39 %** (347/360) | measured (golden model) |
| INT8 test accuracy (RTL) | **96.39 %** (347/360) | measured (simulated RTL) |
| FP32/INT8 prediction agreement | 99.44 % (358/360) | measured |
| INT8 argmax ties | 1 of 360 | measured |
| Clipped biases / reference overflow | 0 / none | measured |
| **RTL vs golden** | **bit-exact on all 360 test inputs** (every one of the 10 INT8 logits) | measured (simulated RTL) |

Two predictions differ between FP32 and INT8. In one, FP32 was correct and INT8
is wrong. In the other, both are wrong and give different labels.

## RTL counters per inference (LANES=4, simulated)

All 360 requests produced identical counters, which is consistent with the
data-independent FSM timing.

| cycles | compute_cycles | controller_cycles | tiles | useful_macs | util_accept = macs/(compute·L) | util_wall = macs/(cycles·L) |
|---:|---:|---:|---:|---:|---:|---:|
| 1310 | 592 | 718 | 592 | 2368 | 1.000 | 0.452 |

Tiles = 32·⌈64/4⌉ + 10·⌈32/4⌉ = 592. Both K values are multiples of 4, so no
lane is padded and `util_accept` is 1.0. The real cost is serialization:
cycles = 2·592 + 3·42 outputs = 1310 exactly. `compute_cycles` counts only tile
accepts, so `util_accept` hides the accumulator's wait cycle (BACKGROUND FACT).
`util_wall` = 0.452 is the honest figure. Stage 3 and Stage 4 test this
decomposition on more shapes.

These numbers are RTL simulation cycle counts, not hardware runs or a
target-FPGA timing result.
