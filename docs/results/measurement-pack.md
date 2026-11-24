# Stage 5: measurement pack (LANES ∈ {1, 2, 4, 8, 16}, MAX_DIM = 64)

Script: `tools/measurement_pack.py`. Data:
`docs/results/data/measurement-pack.json`. Tables:
`docs/results/data/measurement-pack-tables.md`, reproduced below. Yosys logs
stay under the ignored `build/stage5/`.

**Labels.**
- **ESTIMATED (area):** every area number is Yosys 0.33 generic mapping
  (`proc; flatten; opt; memory -nomap; opt; techmap; opt; abc -fast -g
  AND,NAND,OR,NOR,XOR,XNOR,MUX`). It is not a fit to any FPGA or ASIC target.
  No Vivado or Quartus result exists, and none is implied.
- **MEASURED (simulated):** cycle numbers come from Icarus RTL simulation of the
  unchanged RTL, driven by the tiler-emitted config stream, on all 360 digits
  test inputs. All runs were bit-exact against `model.reference`.
- **DERIVED:** roofline shares split the measured counters using the Stage 4
  calibration.

**Flow deviation (recorded).** The plan specified ABC's default script. On the
flattened LANES=4 top (73,444 gates going into ABC), it was still inside ABC
after the 30-minute time box, so that probe was stopped. Every build instead
uses `abc -fast`: the same generic gate library with a lighter optimization
script, which takes about 3 minutes per build end to end. On LANES=4 it gives
68,844 cells, against 74,061 for techmap with no ABC. Absolute gate counts
depend on the optimization effort. The comparisons between builds use one flow
throughout. A default-ABC rerun could be done on a CPU cluster if the absolute
numbers matter.

### Area proxies per build (ESTIMATED: Yosys generic mapping, not a target fit)

| LANES | top logic gates | top flip-flops | top memory bits | top memory read ports | weight-memory read ports | lane datapath gates (tiled accumulator alone) | top coarse cells | logic vs LANES=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64518 | 602 | 150936 | 1313 | 1 | 797 | 14819 | 1.000x |
| 2 | 65856 | 602 | 150936 | 1316 | 2 | 1522 | 14842 | 1.021x |
| 4 | 68227 | 602 | 150936 | 1322 | 4 | 2972 | 14868 | 1.057x |
| 8 | 72723 | 602 | 150936 | 1334 | 8 | 5935 | 14938 | 1.127x |
| 16 | 81230 | 602 | 150936 | 1358 | 16 | 11862 | 15066 | 1.259x |

### Digits model per build (MEASURED: RTL simulation, 360 test inputs, bit-exact)

| LANES | cycles | compute_cycles | controller_cycles | tiles | useful_macs | util_accept | util_wall | speedup vs LANES=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4862 | 2368 | 2494 | 2368 | 2368 | 1.000 | 0.487 | 1.00x |
| 2 | 2494 | 1184 | 1310 | 1184 | 2368 | 1.000 | 0.475 | 1.95x |
| 4 | 1310 | 592 | 718 | 592 | 2368 | 1.000 | 0.452 | 3.71x |
| 8 | 718 | 296 | 422 | 296 | 2368 | 1.000 | 0.412 | 6.77x |
| 16 | 422 | 148 | 274 | 148 | 2368 | 1.000 | 0.351 | 11.52x |

### Roofline-style summary (digits; achieved from measured counters, split DERIVED with the Stage 4 calibration of 2 cycles/tile)

| LANES | achieved MACs/cycle | 2-cycle/tile ceiling (L/2) | % of tile ceiling | accept ceiling (L) | gap to accept ceiling (cycles) | padding share | accumulator-serialization share | controller-overhead share | gates x cycles (ESTIMATED area-delay, millions) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.487 | 0.5 | 97.4 % | 1 | 2494 | 0.0 % | 94.9 % | 5.1 % | 313.7 |
| 2 | 0.949 | 1 | 94.9 % | 2 | 1310 | 0.0 % | 90.4 % | 9.6 % | 164.2 |
| 4 | 1.808 | 2 | 90.4 % | 4 | 718 | 0.0 % | 82.5 % | 17.5 % | 89.4 |
| 8 | 3.298 | 4 | 82.5 % | 8 | 422 | 0.0 % | 70.1 % | 29.9 % | 52.2 |
| 16 | 5.611 | 8 | 70.1 % | 16 | 274 | 0.0 % | 54.0 % | 46.0 % | 34.3 |

## What the pack shows

1. **Most of the area does not depend on LANES (estimated).** Logic grows only
   1.26× from LANES=1 to LANES=16 (64,518 → 81,230 gates). Flip-flops (602) and
   memory bits (150,936) are identical in every build. The lane datapath
   synthesized alone (`int8_tiled_accumulator` with its dot product) scales
   almost linearly, 797 → 11,862 gates, and accounts for nearly all of the
   difference. The fixed ~63.7k gates are mostly the configuration-validity
   checker: `configuration_valid` reads each 256-entry `bias_valid`,
   `multiplier_valid` and `shift_valid` table combinationally, giving 256 read
   ports each. Range comparators for every descriptor, the postprocess
   multiplier and the FSM make up the rest.
2. **Caveat on memory.** The 16,384×8 weight memory needs one asynchronous
   read port per lane (1 → 16 read ports in the table). Gate counts leave
   memory unmapped (`memory -nomap`). On a real FPGA, extra read ports are
   usually built by replicating the memory, so the LANES-dependent cost of the
   weight memory is **not** in these numbers. The logic-only proxy understates
   how area grows with LANES. Only a target fit can size it, and none was run.
3. **Cycles scale almost linearly with LANES (simulated).** Digits goes from
   4,862 to 422 cycles, an 11.5× speedup for 16× lanes. The shortfall from 16×
   is the fixed 3-cycle per-output controller cost, which does not shrink.
   Digits has no padding at any of these builds (K = 64 and 32).
4. **Roofline.** The accumulator's accept/wait handshake caps every build at
   L/2 MACs per cycle, half the lanes' peak.
   - **Narrow builds:** these sit at 95–97 % of that L/2 ceiling. Their gap to
     the L-MACs/cycle accept ceiling is 90–95 % accumulator serialization.
   - **Wider builds:** tiles shrink, so the per-output controller cycles
     (`ACC_START`, `WAIT_ACC`, `STORE`) take a growing share. At LANES=16 that
     is 46 % of the gap, and the build reaches 70 % of its tile ceiling.
   - **Constant:** `util_accept` reads 1.000 for every build. Only `util_wall`
     shows that the engine is idle 51–65 % of the time.
5. **Consequence for the Stage 4 policy.** In area × cycles on these estimates,
   the widest build is best (34.3 M gate-cycles at LANES=16 vs 313.7 M at
   LANES=1). The tiler's FINN-style choice, the smallest build meeting the
   budget, is still the correct resource trade: it never uses more lanes than
   the budget needs. But on this RTL the logic it saves is small, at most
   ~20 % of the top's estimated logic, because the LANES-independent controller
   and validity logic dominate. Folding at build time pays off when the per-lane
   datapath (and, on a real target, the weight-memory ports) dominates. On this
   design that happens only far beyond LANES=16, or after the fixed checker
   logic is reduced. Changing the checker would be an RTL change, out of scope
   here.
