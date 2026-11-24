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
