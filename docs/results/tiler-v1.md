# Stage 4: tiler v1, build selection

Code: `malleable/tiler.py`. Tests: `tests/test_tiler.py`. Calibration:
`malleable/tiler_calibration.json`. Verification script:
`tools/tiler_verify.py`. Data: `docs/results/data/tiler-verify.json`. Sample
reports: `docs/results/tiler-sample-report.json` (JSON) and
`docs/results/data/tiler-sample-reports.md` (rendered).

## Policy

Stage 3 showed that runtime lane reduction on one build never reduces cycles.
The tiler therefore folds at **build time**: it treats LANES as a synthesis
parameter that trades area for throughput. This follows FINN's PE/SIMD folding
(Umuroglu et al., FPGA 2017; FINN-R, TRETS 2018) and DNNBuilder's
parallelism-factor search (Zhang et al., FPGA 2018). Given a quantized model, a
cycle budget and a LANES range (default 1..16), `plan_build`:

1. **fit-checks** each build against the RTL's `MAX_LAYERS`, `MAX_DIM`,
   `WEIGHT_DEPTH`, `PARAM_DEPTH`, layer adjacency, and the contract ranges of
   every weight, bias, multiplier, shift and ReLU flag. It mirrors the RTL's
   `configuration_valid` logic. `FitError` names the limit, for example
   `MAX_DIM: layer 0 input_count 80 outside 1..64`;
2. **predicts** whole-model cycles from the tile model;
3. **selects the smallest LANES** whose predicted cycles meet the budget. If
   none does, it raises an error naming the best achievable cycles and the
   smallest build that reaches them;
4. **emits** the config-write stream in the backend's `(kind, addr, data)`
   format: `writes(model)` followed by the kind-6 lane write. Config is emitted
   from the hardware parameters, as in Gemmini (Genc et al., DAC 2021). Unlike
   VTA (Moreau et al. 2018), which tiles a model onto one fixed accelerator, the
   tiler picks the accelerator instance;
5. **reports** the chosen build and its parameters, predicted cycles, padding
   waste % (zero-filled lane slots / issued lane slots), runner-up builds that
   also meet the budget with their predicted cycles, and the full candidate
   table. Area per build is read from the Stage 5 sweep when present. Builds the
   sweep did not cover are labeled `unavailable`.

## Calibration

The tiler fits `cycles = a·tiles + b·outputs + c·layers + d` by least squares
(normal equations, as in `optimize.Predictor`). Data: all 216 Stage 3 rows
(simulated RTL, builds LANES 4 and 8, whole-model and single-layer jobs).
Result: **a = 2.000, b = 3.000, c ≈ 0, d ≈ 0**, with a maximum residual of
2e-12 cycles. Two cycles per tile (accept, then wait for the registered
dot-product result) confirms the BACKGROUND FACT. Three cycles per output are
`ACC_START`, `WAIT_ACC` and `STORE`. Predictions are rounded to 1e-6 so that a
budget equal to an exact prediction counts as met. Verification found this
float-residue edge case: a 2730-cycle budget picked LANES=5 instead of 4. It
was fixed before commit and is covered by a regression test.

## Verification (simulated RTL)

For the Stage 2 digits model and the three Stage 3 synthetic shapes, at every
build LANES = 1..16, `tools/tiler_verify.py` instantiated the RTL with that
LANES and drove it with **the tiler-emitted stream** (`RTLBackend.run(...,
stream=...)`). Results:
- **Bit-exact:** all 64 runs matched `model.reference` on every output (4
  requests each).
- **Prediction error:** predicted − measured cycles is **0** for all 64 builds.
  That includes the 56 builds (LANES 1–3, 5–7, 9–16) outside the calibration
  sample, marked * below.

Predicted (estimated: calibrated tile model) / measured (simulated RTL) cycles:

| LANES | digits predicted / measured (err) | syn-aligned predicted / measured (err) | syn-odd predicted / measured (err) | syn-mixed predicted / measured (err) |
|---:|---|---|---|---|
| 1 | 4862 / 4862 (+0) | 33536 / 33536 (+0) | 9626 / 9626 (+0) | 2063 / 2063 (+0) |
| 2 | 2494 / 2494 (+0) | 17152 / 17152 (+0) | 5048 / 5048 (+0) | 1359 / 1359 (+0) |
| 3 | 1754 / 1754 (+0) | 12032 / 12032 (+0) | 3494 / 3494 (+0) | 1003 / 1003 (+0) |
| 4* | 1310 / 1310 (+0) | 8960 / 8960 (+0) | 2730 / 2730 (+0) | 943 / 943 (+0) |
| 5 | 1098 / 1098 (+0) | 7424 / 7424 (+0) | 2270 / 2270 (+0) | 785 / 785 (+0) |
| 6 | 950 / 950 (+0) | 6400 / 6400 (+0) | 1940 / 1940 (+0) | 765 / 765 (+0) |
| 7 | 866 / 866 (+0) | 5888 / 5888 (+0) | 1684 / 1684 (+0) | 755 / 755 (+0) |
| 8* | 718 / 718 (+0) | 4864 / 4864 (+0) | 1506 / 1506 (+0) | 735 / 735 (+0) |
| 9 | 718 / 718 (+0) | 4864 / 4864 (+0) | 1412 / 1412 (+0) | 735 / 735 (+0) |
| 10 | 654 / 654 (+0) | 4352 / 4352 (+0) | 1328 / 1328 (+0) | 725 / 725 (+0) |
| 11 | 570 / 570 (+0) | 3840 / 3840 (+0) | 1234 / 1234 (+0) | 715 / 715 (+0) |
| 12 | 570 / 570 (+0) | 3840 / 3840 (+0) | 1176 / 1176 (+0) | 715 / 715 (+0) |
| 13 | 506 / 506 (+0) | 3328 / 3328 (+0) | 1072 / 1072 (+0) | 705 / 705 (+0) |
| 14 | 506 / 506 (+0) | 3328 / 3328 (+0) | 1072 / 1072 (+0) | 705 / 705 (+0) |
| 15 | 506 / 506 (+0) | 3328 / 3328 (+0) | 1046 / 1046 (+0) | 705 / 705 (+0) |
| 16 | 422 / 422 (+0) | 2816 / 2816 (+0) | 894 / 894 (+0) | 695 / 695 (+0) |

\* LANES 4 and 8 are the calibration builds. All other rows are out of sample.

Utilization per build, `util_accept / util_wall` (simulated RTL):

| model | L=1 | L=2 | L=4 | L=8 | L=16 |
|---|---:|---:|---:|---:|---:|
| digits | 1.000 / 0.487 | 1.000 / 0.475 | 1.000 / 0.452 | 1.000 / 0.412 | 1.000 / 0.351 |
| syn-aligned | 1.000 / 0.489 | 1.000 / 0.478 | 1.000 / 0.457 | 1.000 / 0.421 | 1.000 / 0.364 |
| syn-odd | 1.000 / 0.485 | 0.980 / 0.463 | 0.954 / 0.428 | 0.954 / 0.388 | 0.954 / 0.327 |
| syn-mixed | 1.000 / 0.403 | 0.867 / 0.306 | 0.765 / 0.221 | 0.619 / 0.141 | 0.351 / 0.075 |

`util_accept` stays near 1.0 because it counts only tile-accept cycles.
`util_wall` is capped at 0.5 by the 2-cycle tile and falls further as
per-output controller cycles dominate at wide builds and small K. syn-mixed at
LANES=16 reaches 0.075.

## Sample reports

Budgets were fixed by rule before selection: the measured all-lanes cycles of
the LANES=4 and LANES=8 builds, plus an infeasible budget of 1 cycle. Each
feasible budget selected exactly the build it came from, and nothing smaller
could do as well. Infeasible budgets were rejected with the best achievable
cycles. Example (digits, budget 1310 cycles):

- **Chosen build:** LANES=4 (MAX_LAYERS=4, MAX_DIM=64) for budget 1310 cycles
- **Predicted cycles:** 1310 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 0.00 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 4862 | no | 0.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 2 | 2494 | no | 0.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 3 | 1754 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 (chosen) | 1310 | yes | 0.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 5 (runner-up) | 1098 | yes | 2.55 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 (runner-up) | 950 | yes | 4.21 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 (runner-up) | 866 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 | 718 | yes | 0.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 9 | 718 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 | 654 | yes | 10.30 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 | 570 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 570 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 506 | yes | 4.13 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 506 | yes | 10.98 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 506 | yes | 16.91 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 422 | yes | 0.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |

Area showed as `unavailable` when this example was first rendered, before
Stage 5 had run. The committed `tiler-sample-report.json` and
`data/tiler-sample-reports.md` were regenerated after Stage 5. They carry
ESTIMATED area for LANES ∈ {1, 2, 4, 8, 16} and keep `unavailable` for every
other build. See `measurement-pack.md` for why the area differences between
builds are small on this RTL.

## Changes outside the tiler

- `malleable/backend.py`: `RTLBackend.run` gains an optional `stream=`
  argument, so verification drives the RTL with the tiler's emitted writes
  rather than a stream regenerated by the backend. Default behavior is
  unchanged.
- `malleable/model.py` (Stage 3): `reference` validates `active_lanes` over
  1..64 instead of 1..8. This is validation only.
- No RTL, contract, or `third_party/` file changed.
