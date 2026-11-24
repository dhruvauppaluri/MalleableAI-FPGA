# Stage 3: oracle lane bound (measurement only)

**Every number here is from RTL simulation** (Icarus Verilog 12.0 running
`rtl/malleable_accelerator_top.sv` unchanged). These are simulated cycle
counts, not hardware runs. Script: `tools/lane_oracle.py`. Raw data:
`docs/results/data/lane-oracle.json`. Full table:
`docs/results/data/lane-oracle-table.md`, reproduced below.

## Method

Models. The synthetic shapes were fixed in the script before any measurement
and were not tuned. Weights and parameters are seeded random INT8 in the style
of `model.fixture`, with seed 0.

| model | dims | why |
|---|---|---|
| digits | 64→32→10 | Stage 2 quantized artifact |
| syn-aligned | 64→64→64→64→64 | every K is a multiple of 1, 2, 4 and 8 |
| syn-odd | 61→47→29→13→5 | odd/prime K, padding at most lane counts |
| syn-mixed | 5→64→3→64→2 | K both below and above LANES |

For compiled builds LANES ∈ {4, 8}, each model ran:
- **whole model** at every fixed `active_lanes` 1..LANES (config kind 6);
- **each layer as a single-layer job** at every `active_lanes`, chained
  through the host: layer i's RTL outputs became layer i+1's inputs.

Each job ran 4 seeded requests; counters were identical across requests in
every case. Outputs were checked bit-exactly against `model.reference` (the
backend raises on any mismatch). That makes 216 measured rows (whole-model
plus per-layer) and 13,136 exhaustively evaluated per-layer lane mixes.

`util_accept = useful_macs / (compute_cycles · LANES)` and
`util_wall = useful_macs / (cycles · LANES)`, where LANES is the compiled build
width.

## Results

1. **Per-layer jobs compose exactly.** For every model, build and lane
   setting, whole-model `cycles` equals the sum of the per-layer job `cycles`
   (delta = 0 in all 48 comparisons). A layer boundary costs the FSM nothing
   beyond one more output's `STORE → ACC_START`. Host copying between chained
   layer jobs is **not** counted by the hardware counters and was not measured.
   Any real per-layer reconfiguration would add that cost on top.
2. **Exact cycle law.** All 216 rows satisfy
   `cycles = 2·tiles + 3·outputs`, with `tiles = Σ n·⌈k/active_lanes⌉`. Of the
   cycles, `compute_cycles` = tiles (one accept per tile). The other tile cycle
   is the accumulator waiting for the registered dot-product result. The
   remaining 3 cycles per output are `ACC_START`, `WAIT_ACC` and `STORE`.
3. **No per-layer mix beats all-lanes.** In all 8 model×build cases, the best
   of the 13,136 mixes equals the all-lanes cycles, and 0 mixes are strictly
   faster. The oracle mix was rerun in RTL as chained jobs, and both its
   outputs and its cycles matched.
4. **Ties occur only where padding aligns.** A smaller lane value ties
   all-lanes on a layer only when ⌈k/a⌉ = ⌈k/LANES⌉. Measured cases:
   - syn-odd at LANES=8, K=13: a=7;
   - syn-mixed at LANES=4, K=5 and K=3: a=3;
   - syn-mixed at LANES=8, K=5: a=5..7;
   - syn-mixed at LANES=8, K=3: a=3..7.

   Those are 1 to 24 tying mixes per case (listed in the JSON). The aligned and
   digits shapes have no smaller tying value, because ⌈64/7⌉ = 10 > 8.

## Finding

The data confirms the BACKGROUND FACT and does not refute any part of it. On
a fixed build, lowering `active_lanes`, uniformly or per layer, never
reduces whole-model cycles. At best it ties at shapes where ⌈k/a⌉ = ⌈k/LANES⌉,
and that tie only spares lanes that are already idle. Lane choice also does not
dominate utilization:
- **Accumulator.** The accumulator's accept/wait serialization caps
  `util_wall` at 0.5. Measured values range 0.42–0.46 on 64-wide layers.
- **Controller.** The 3-cycle per-output controller cost dominates small-K
  layers. syn-mixed at LANES=8 reaches only `util_wall` = 0.141: 168 tiles but
  133 outputs, so 399 controller cycles against 168 wait cycles.
- **`util_accept` is misleading.** It reads 1.000 for digits and syn-aligned
  while the engine is idle more than half the time.

What lane choice *does* buy is visible in the table. A 4-lane build and an
8-lane build run at `active_lanes=4` take identical cycles (1310 vs 1310 for
digits, 8960 vs 8960 for syn-aligned). The same cycles are therefore available
from a smaller build. That is the FINN view (Umuroglu et al. 2017; FINN-R
2018): folding is a synthesis-time area/throughput trade. Stage 4's tiler and
Stage 5's area sweep are built on it.

## Table (simulated RTL)

The last column is cycles relative to all-lanes on the same build. "per-layer
oracle" is the best exhaustive mix, with cycles taken from the additive
per-layer data (delta 0) and confirmed by a chained RTL rerun.

| model | build LANES | strategy | cycles | compute_cycles | controller_cycles | tiles | useful_macs | util_accept | util_wall | vs all-lanes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| digits | 4 | fixed 1 | 4862 | 2368 | 2494 | 2368 | 2368 | 0.250 | 0.122 | 3.711x |
| digits | 4 | fixed 2 | 2494 | 1184 | 1310 | 1184 | 2368 | 0.500 | 0.237 | 1.904x |
| digits | 4 | fixed 3 | 1754 | 814 | 940 | 814 | 2368 | 0.727 | 0.338 | 1.339x |
| digits | 4 | fixed 4 | 1310 | 592 | 718 | 592 | 2368 | 1.000 | 0.452 | 1.000x |
| digits | 4 | per-layer oracle [4, 4] | 1310 | 592 | 718 | 592 | 2368 | 1.000 | 0.452 | 1.000x |
| digits | 8 | fixed 1 | 4862 | 2368 | 2494 | 2368 | 2368 | 0.125 | 0.061 | 6.772x |
| digits | 8 | fixed 2 | 2494 | 1184 | 1310 | 1184 | 2368 | 0.250 | 0.119 | 3.474x |
| digits | 8 | fixed 3 | 1754 | 814 | 940 | 814 | 2368 | 0.364 | 0.169 | 2.443x |
| digits | 8 | fixed 4 | 1310 | 592 | 718 | 592 | 2368 | 0.500 | 0.226 | 1.825x |
| digits | 8 | fixed 5 | 1098 | 486 | 612 | 486 | 2368 | 0.609 | 0.270 | 1.529x |
| digits | 8 | fixed 6 | 950 | 412 | 538 | 412 | 2368 | 0.718 | 0.312 | 1.323x |
| digits | 8 | fixed 7 | 866 | 370 | 496 | 370 | 2368 | 0.800 | 0.342 | 1.206x |
| digits | 8 | fixed 8 | 718 | 296 | 422 | 296 | 2368 | 1.000 | 0.412 | 1.000x |
| digits | 8 | per-layer oracle [8, 8] | 718 | 296 | 422 | 296 | 2368 | 1.000 | 0.412 | 1.000x |
| syn-aligned | 4 | fixed 1 | 33536 | 16384 | 17152 | 16384 | 16384 | 0.250 | 0.122 | 3.743x |
| syn-aligned | 4 | fixed 2 | 17152 | 8192 | 8960 | 8192 | 16384 | 0.500 | 0.239 | 1.914x |
| syn-aligned | 4 | fixed 3 | 12032 | 5632 | 6400 | 5632 | 16384 | 0.727 | 0.340 | 1.343x |
| syn-aligned | 4 | fixed 4 | 8960 | 4096 | 4864 | 4096 | 16384 | 1.000 | 0.457 | 1.000x |
| syn-aligned | 4 | per-layer oracle [4, 4, 4, 4] | 8960 | 4096 | 4864 | 4096 | 16384 | 1.000 | 0.457 | 1.000x |
| syn-aligned | 8 | fixed 1 | 33536 | 16384 | 17152 | 16384 | 16384 | 0.125 | 0.061 | 6.895x |
| syn-aligned | 8 | fixed 2 | 17152 | 8192 | 8960 | 8192 | 16384 | 0.250 | 0.119 | 3.526x |
| syn-aligned | 8 | fixed 3 | 12032 | 5632 | 6400 | 5632 | 16384 | 0.364 | 0.170 | 2.474x |
| syn-aligned | 8 | fixed 4 | 8960 | 4096 | 4864 | 4096 | 16384 | 0.500 | 0.229 | 1.842x |
| syn-aligned | 8 | fixed 5 | 7424 | 3328 | 4096 | 3328 | 16384 | 0.615 | 0.276 | 1.526x |
| syn-aligned | 8 | fixed 6 | 6400 | 2816 | 3584 | 2816 | 16384 | 0.727 | 0.320 | 1.316x |
| syn-aligned | 8 | fixed 7 | 5888 | 2560 | 3328 | 2560 | 16384 | 0.800 | 0.348 | 1.211x |
| syn-aligned | 8 | fixed 8 | 4864 | 2048 | 2816 | 2048 | 16384 | 1.000 | 0.421 | 1.000x |
| syn-aligned | 8 | per-layer oracle [8, 8, 8, 8] | 4864 | 2048 | 2816 | 2048 | 16384 | 1.000 | 0.421 | 1.000x |
| syn-odd | 4 | fixed 1 | 9626 | 4672 | 4954 | 4672 | 4672 | 0.250 | 0.121 | 3.526x |
| syn-odd | 4 | fixed 2 | 5048 | 2383 | 2665 | 2383 | 4672 | 0.490 | 0.231 | 1.849x |
| syn-odd | 4 | fixed 3 | 3494 | 1606 | 1888 | 1606 | 4672 | 0.727 | 0.334 | 1.280x |
| syn-odd | 4 | fixed 4 | 2730 | 1224 | 1506 | 1224 | 4672 | 0.954 | 0.428 | 1.000x |
| syn-odd | 4 | per-layer oracle [4, 4, 4, 4] | 2730 | 1224 | 1506 | 1224 | 4672 | 0.954 | 0.428 | 1.000x |
| syn-odd | 8 | fixed 1 | 9626 | 4672 | 4954 | 4672 | 4672 | 0.125 | 0.061 | 6.392x |
| syn-odd | 8 | fixed 2 | 5048 | 2383 | 2665 | 2383 | 4672 | 0.245 | 0.116 | 3.352x |
| syn-odd | 8 | fixed 3 | 3494 | 1606 | 1888 | 1606 | 4672 | 0.364 | 0.167 | 2.320x |
| syn-odd | 8 | fixed 4 | 2730 | 1224 | 1506 | 1224 | 4672 | 0.477 | 0.214 | 1.813x |
| syn-odd | 8 | fixed 5 | 2270 | 994 | 1276 | 994 | 4672 | 0.588 | 0.257 | 1.507x |
| syn-odd | 8 | fixed 6 | 1940 | 829 | 1111 | 829 | 4672 | 0.704 | 0.301 | 1.288x |
| syn-odd | 8 | fixed 7 | 1684 | 701 | 983 | 701 | 4672 | 0.833 | 0.347 | 1.118x |
| syn-odd | 8 | fixed 8 | 1506 | 612 | 894 | 612 | 4672 | 0.954 | 0.388 | 1.000x |
| syn-odd | 8 | per-layer oracle [8, 8, 8, 8] | 1506 | 612 | 894 | 612 | 4672 | 0.954 | 0.388 | 1.000x |
| syn-mixed | 4 | fixed 1 | 2063 | 832 | 1231 | 832 | 832 | 0.250 | 0.101 | 2.188x |
| syn-mixed | 4 | fixed 2 | 1359 | 480 | 879 | 480 | 832 | 0.433 | 0.153 | 1.441x |
| syn-mixed | 4 | fixed 3 | 1003 | 302 | 701 | 302 | 832 | 0.689 | 0.207 | 1.064x |
| syn-mixed | 4 | fixed 4 | 943 | 272 | 671 | 272 | 832 | 0.765 | 0.221 | 1.000x |
| syn-mixed | 4 | per-layer oracle [4, 4, 4, 4] | 943 | 272 | 671 | 272 | 832 | 0.765 | 0.221 | 1.000x |
| syn-mixed | 8 | fixed 1 | 2063 | 832 | 1231 | 832 | 832 | 0.125 | 0.050 | 2.807x |
| syn-mixed | 8 | fixed 2 | 1359 | 480 | 879 | 480 | 832 | 0.217 | 0.077 | 1.849x |
| syn-mixed | 8 | fixed 3 | 1003 | 302 | 701 | 302 | 832 | 0.344 | 0.104 | 1.365x |
| syn-mixed | 8 | fixed 4 | 943 | 272 | 671 | 272 | 832 | 0.382 | 0.110 | 1.283x |
| syn-mixed | 8 | fixed 5 | 785 | 193 | 592 | 193 | 832 | 0.539 | 0.132 | 1.068x |
| syn-mixed | 8 | fixed 6 | 765 | 183 | 582 | 183 | 832 | 0.568 | 0.136 | 1.041x |
| syn-mixed | 8 | fixed 7 | 755 | 178 | 577 | 178 | 832 | 0.584 | 0.138 | 1.027x |
| syn-mixed | 8 | fixed 8 | 735 | 168 | 567 | 168 | 832 | 0.619 | 0.141 | 1.000x |
| syn-mixed | 8 | per-layer oracle [8, 8, 8, 8] | 735 | 168 | 567 | 168 | 832 | 0.619 | 0.141 | 1.000x |
