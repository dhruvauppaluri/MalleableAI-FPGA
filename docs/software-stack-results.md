# Dense accelerator software stack: results (2026-11-24)

This adds the missing software stack for `malleable_accelerator_top`: a
quantizer, a first trained model run end to end, and a tiler that selects
builds. It also gives the first measurements of what lane folding can and
cannot do on this RTL.

**Provenance labels used throughout:**
- **measured (simulated):** Icarus Verilog RTL simulation of the unchanged
  RTL;
- **measured (host):** FP32 and golden-model computation in Python;
- **estimated:** Yosys generic mapping, or a calibrated model.

No FPGA was programmed. No Vivado, Quartus, AWS or cloud-FPGA work was done.
`rtl/`, the numeric contract's arithmetic and `third_party/opentpu/` are
unchanged.

Stage records:
- [Stage 0 baseline](results/stage0-baseline.md)
- [Stage 2 digits](results/digits-e2e.md)
- [Stage 3 lane oracle](results/lane-oracle.md)
- [Stage 4 tiler](results/tiler-v1.md)
- [Stage 5 measurement pack](results/measurement-pack.md)

## What was built

| Piece | Where | Idea from |
|---|---|---|
| FP32 → contract-v0.2 INT8 quantizer: batch-norm folding in float first, per-output-channel weight scales, min/max activation calibration, `multiplier / 2^shift` requantization through `model.requantize()` | `malleable/quantize.py`, `docs/quantization.md`, `tests/test_quantize.py` | Jacob et al. 2018: integer-only requantization (§2.2), BN folding before quantization (§3.2) |
| Digits MLP trained, quantized and run on the golden model and RTL | `tools/digits_e2e.py`, `docs/results/digits/` | — |
| Per-layer oracle lane experiment | `tools/lane_oracle.py` | — |
| Tiler v1: fit checks against RTL limits, a calibrated cycle model, smallest-build-meeting-budget selection, config-stream emission and a report | `malleable/tiler.py`, `malleable/tiler_calibration.json`, `tests/test_tiler.py`, `tools/tiler_verify.py` | FINN / FINN-R (folding as a synthesis-time resource/throughput trade); DNNBuilder (search over parallelism factors against a budget); Gemmini (config emitted from hardware parameters); VTA (software tiling onto a fixed accelerator, which is the contrast here: the tiler picks the accelerator instance) |
| Measurement pack: Yosys area proxies, RTL cycles and a roofline split | `tools/measurement_pack.py` | Roofline (Williams, Waterman, Patterson 2009) |

Supporting edits, none of them arithmetic:
- `model.reference` validates `active_lanes` over 1..64 instead of 1..8.
- `RTLBackend.run(..., stream=)` can be driven by an externally emitted config
  stream.

## Stage 2: first real model

Data: sklearn `load_digits`, stratified 80/20 split, seed 0. Model: MLP
64→32(ReLU)→10.

| | Test accuracy (360 samples) |
|---|---:|
| FP32 (measured, host) | **96.67 %** |
| INT8 golden `model.reference` (measured, host) | **96.39 %** |
| INT8 RTL, LANES=4 (measured, simulated) | **96.39 %** |

The RTL matched the golden model **bit-exactly on all 360 inputs**, across
every one of the 10 logits. Two predictions differ between FP32 and INT8, and
one INT8 argmax is a tie. Each inference takes 1,310 cycles at LANES=4
(simulated).

## Stage 3: folding finding

Four models were measured: digits plus three synthetic shapes fixed before
measurement (aligned 64-wide, odd/prime K, mixed tiny/huge K). Each ran at
builds LANES=4 and 8, as whole models at every `active_lanes` and as chained
single-layer jobs (simulated):

- **Exact cycle law.** Every one of the 216 rows obeys
  `cycles = 2·tiles + 3·outputs`, and per-layer job cycles sum exactly to
  whole-model cycles.
- **No per-layer mix wins.** 0 of 13,136 exhaustively evaluated per-layer lane
  mixes beat all-lanes, and the oracle mix was confirmed by an RTL rerun.
  Ties occur only where ⌈k/a⌉ = ⌈k/LANES⌉.
- **Folding changes the build, not the cycles.** A 4-lane build and an 8-lane
  build running 4 lanes take identical cycles.

This confirms the premise of FINN / FINN-R: folding is a build-time trade. It
also sets up the contrast with MAERI (Kwon et al. 2018), which shows that
flexibility in runtime dataflow costs real interconnect. This design has one
global lane register and no such network, so runtime per-layer folding has
nothing to gain on cycles.

## Stage 4: tiler policy

`plan_build` enumerates builds LANES 1..16 and fit-checks each against
`MAX_LAYERS`, `MAX_DIM`, `WEIGHT_DEPTH`, `PARAM_DEPTH`, layer adjacency and the
contract ranges, raising `FitError` with the limit named. It predicts cycles
as `2·tiles + 3·outputs`, using coefficients fitted by least squares on the
Stage 3 data (residual 2e-12). It then picks the smallest build that meets the
cycle budget and emits the backend-format config stream. Its report gives the
chosen build, predicted cycles, padding waste, runner-ups, and area from
Stage 5 (`unavailable` for builds not synthesized).

Verification ran 64 builds (4 models × LANES 1..16), with the tiler-emitted
stream driving the RTL:
- all 64 were bit-exact;
- predicted − measured cycles was **0** for all 64, including 56 builds outside
  the calibration sample.

## Stage 5: measurement pack

Digits model, MAX_DIM 64:

| LANES | logic gates (est.) | lane datapath gates (est.) | memory bits (est.) | cycles (sim.) | util_accept (sim.) | util_wall (sim.) | % of 2-cycle/tile ceiling | gap share: accumulator serialization / controller |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64,518 | 797 | 150,936 | 4,862 | 1.000 | 0.487 | 97.4 % | 94.9 % / 5.1 % |
| 2 | 65,856 | 1,522 | 150,936 | 2,494 | 1.000 | 0.475 | 94.9 % | 90.4 % / 9.6 % |
| 4 | 68,227 | 2,972 | 150,936 | 1,310 | 1.000 | 0.452 | 90.4 % | 82.5 % / 17.5 % |
| 8 | 72,723 | 5,935 | 150,936 | 718 | 1.000 | 0.412 | 82.5 % | 70.1 % / 29.9 % |
| 16 | 81,230 | 11,862 | 150,936 | 422 | 1.000 | 0.351 | 70.1 % | 54.0 % / 46.0 % |

- **Area barely moves.** Area is estimated with Yosys 0.33 generic mapping
  using `abc -fast`, because the default ABC script exceeded a 30-minute box.
  The top's logic grows only 1.26× across a 16× lane range. Flip-flops and
  memory bits are constant. Most of the area is LANES-independent validity and
  controller logic.
- **Weight-memory caveat.** The weight memory's read ports grow 1 → 16 with
  LANES. Gate counts exclude that cost, and on a real target it would likely
  replicate memory.
- **Utilization is set by serialization, not lane choice.** The accumulator's
  accept/wait handshake caps `util_wall` at 0.5 for every build. `util_accept`
  reads 1.000 throughout and hides this.
- **The bottleneck shifts with width.** It moves from accumulator
  serialization at narrow builds to the 3-cycle per-output controller cost at
  wide builds.
- **What smaller builds save.** The tiler's smallest-sufficient-build policy is
  the correct trade. On this RTL it saves at most ~20 % of estimated logic,
  because the fixed logic dominates.

## What is NOT shown

- **No target-FPGA fit.** The area numbers are generic Yosys gate counts with
  memories left unmapped. There are no LUT, FF, BRAM or DSP counts, no Fmax,
  and no timing closure.
- **No hardware run.** Every cycle number is RTL simulation. No board was
  programmed and nothing ran on hardware.
- **No per-layer runtime folding benefit.** Stage 3 measured 0 of 13,136
  per-layer mixes beating all-lanes on whole-model cycles. Per-layer jobs also
  add host copying between layers, which the hardware counters do not measure.
- **Build-time folding's area benefit is unquantified on a real target.** It
  depends on how the weight memory's read ports map to target memory.
- **Quantizer quality beyond this model is unvalidated.** The quantizer was
  validated on one real model (digits) and small random models. There is no
  claim about larger models or other calibration methods.

## References

- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference," CVPR 2018.
- Y. Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural
  Network Inference," FPGA 2017; M. Blott et al., "FINN-R: An End-to-End
  Deep-Learning Framework for Fast Exploration of Quantized Neural Networks,"
  ACM TRETS 2018.
- H. Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture
  Evaluation via Full-Stack Integration," DAC 2021.
- T. Moreau et al., "A Hardware-Software Blueprint for Flexible Deep Learning
  Specialization" (VTA), 2018.
- H. Kwon, A. Samajdar, T. Krishna, "MAERI: Enabling Flexible Dataflow Mapping
  over DNN Accelerators via Reconfigurable Interconnects," ASPLOS 2018.
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual
  Performance Model for Multicore Architectures," CACM 2009.
- X. Zhang et al., "DNNBuilder: an Automated Tool for Building High-Performance
  DNN Hardware Accelerators for FPGAs," ICCAD 2018. (The task brief cited it as
  FPGA 2018; the paper appeared at ICCAD 2018.)
