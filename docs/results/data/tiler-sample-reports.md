### digits: budget=measured LANES=4

- **Chosen build:** LANES=4 (MAX_LAYERS=4, MAX_DIM=64) for budget 1310 cycles
- **Predicted cycles:** 1310 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 0.00 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 4862 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 2494 | no | 0.00 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 1754 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 (chosen) | 1310 | yes | 0.00 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 (runner-up) | 1098 | yes | 2.55 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 (runner-up) | 950 | yes | 4.21 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 (runner-up) | 866 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 | 718 | yes | 0.00 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 | 718 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 | 654 | yes | 10.30 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 | 570 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 570 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 506 | yes | 4.13 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 506 | yes | 10.98 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 506 | yes | 16.91 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 422 | yes | 0.00 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### digits: budget=measured LANES=8

- **Chosen build:** LANES=8 (MAX_LAYERS=4, MAX_DIM=64) for budget 718 cycles
- **Predicted cycles:** 718 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 0.00 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 4862 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 2494 | no | 0.00 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 1754 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 | 1310 | no | 0.00 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 | 1098 | no | 2.55 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 | 950 | no | 4.21 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 | 866 | no | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 (chosen) | 718 | yes | 0.00 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 (runner-up) | 718 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 (runner-up) | 654 | yes | 10.30 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 (runner-up) | 570 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 570 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 506 | yes | 4.13 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 506 | yes | 10.98 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 506 | yes | 16.91 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 422 | yes | 0.00 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### digits: budget=1 (infeasible)

Rejected: `no build in LANES 1..16 meets 1 cycles; best achievable is 422.0 cycles at LANES=16`

### syn-aligned: budget=measured LANES=4

- **Chosen build:** LANES=4 (MAX_LAYERS=4, MAX_DIM=64) for budget 8960 cycles
- **Predicted cycles:** 8960 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 0.00 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 33536 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 17152 | no | 0.00 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 12032 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 (chosen) | 8960 | yes | 0.00 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 (runner-up) | 7424 | yes | 1.54 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 (runner-up) | 6400 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 (runner-up) | 5888 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 | 4864 | yes | 0.00 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 | 4864 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 | 4352 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 | 3840 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 3840 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 3328 | yes | 1.54 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 3328 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 3328 | yes | 14.67 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 2816 | yes | 0.00 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-aligned: budget=measured LANES=8

- **Chosen build:** LANES=8 (MAX_LAYERS=4, MAX_DIM=64) for budget 4864 cycles
- **Predicted cycles:** 4864 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 0.00 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 33536 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 17152 | no | 0.00 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 12032 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 | 8960 | no | 0.00 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 | 7424 | no | 1.54 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 | 6400 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 | 5888 | no | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 (chosen) | 4864 | yes | 0.00 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 (runner-up) | 4864 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 (runner-up) | 4352 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 (runner-up) | 3840 | yes | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 3840 | yes | 11.11 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 3328 | yes | 1.54 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 3328 | yes | 8.57 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 3328 | yes | 14.67 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 2816 | yes | 0.00 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-aligned: budget=1 (infeasible)

Rejected: `no build in LANES 1..16 meets 1 cycles; best achievable is 2816.0 cycles at LANES=16`

### syn-odd: budget=measured LANES=4

- **Chosen build:** LANES=4 (MAX_LAYERS=4, MAX_DIM=64) for budget 2730 cycles
- **Predicted cycles:** 2730 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 4.58 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 9626 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 5048 | no | 1.97 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 3494 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 (chosen) | 2730 | yes | 4.58 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 (runner-up) | 2270 | yes | 6.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 (runner-up) | 1940 | yes | 6.07 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 (runner-up) | 1684 | yes | 4.79 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 | 1506 | yes | 4.58 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 | 1412 | yes | 8.12 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 | 1328 | yes | 10.67 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 | 1234 | yes | 10.77 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 1176 | yes | 12.90 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 1072 | yes | 9.02 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 1072 | yes | 15.52 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 1046 | yes | 18.46 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 894 | yes | 4.58 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-odd: budget=measured LANES=8

- **Chosen build:** LANES=8 (MAX_LAYERS=4, MAX_DIM=64) for budget 1506 cycles
- **Predicted cycles:** 1506 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 4.58 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 9626 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 5048 | no | 1.97 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 3494 | no | 3.03 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 | 2730 | no | 4.58 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 | 2270 | no | 6.00 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 | 1940 | no | 6.07 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 | 1684 | no | 4.79 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 (chosen) | 1506 | yes | 4.58 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 (runner-up) | 1412 | yes | 8.12 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 (runner-up) | 1328 | yes | 10.67 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 (runner-up) | 1234 | yes | 10.77 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 1176 | yes | 12.90 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 1072 | yes | 9.02 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 1072 | yes | 15.52 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 1046 | yes | 18.46 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 894 | yes | 4.58 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-odd: budget=1 (infeasible)

Rejected: `no build in LANES 1..16 meets 1 cycles; best achievable is 894.0 cycles at LANES=16`

### syn-mixed: budget=measured LANES=4

- **Chosen build:** LANES=4 (MAX_LAYERS=4, MAX_DIM=64) for budget 943 cycles
- **Predicted cycles:** 943 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 23.53 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 2063 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 1359 | no | 13.33 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 1003 | no | 8.17 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 (chosen) | 943 | yes | 23.53 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 (runner-up) | 785 | yes | 13.78 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 (runner-up) | 765 | yes | 24.23 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 (runner-up) | 755 | yes | 33.23 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 | 735 | yes | 38.10 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 | 735 | yes | 44.97 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 | 725 | yes | 48.96 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 | 715 | yes | 52.13 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 715 | yes | 56.12 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 705 | yes | 58.17 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 705 | yes | 61.16 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 705 | yes | 63.75 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 695 | yes | 64.86 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-mixed: budget=measured LANES=8

- **Chosen build:** LANES=8 (MAX_LAYERS=4, MAX_DIM=64) for budget 735 cycles
- **Predicted cycles:** 735 (estimated:calibrated-tile-model(stage3 rtl-simulation))
- **Padding waste:** 38.10 %
- **Cycle model:** 2*tiles + 3*outputs + 0*layers + 0

| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |
|---:|---:|:---:|---:|---:|---|
| 1 | 2063 | no | 0.00 % | 65135 | estimated:yosys-generic-mapping(stage5) |
| 2 | 1359 | no | 13.33 % | 66473 | estimated:yosys-generic-mapping(stage5) |
| 3 | 1003 | no | 8.17 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 4 | 943 | no | 23.53 % | 68844 | estimated:yosys-generic-mapping(stage5) |
| 5 | 785 | no | 13.78 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 6 | 765 | no | 24.23 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 7 | 755 | no | 33.23 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 8 (chosen) | 735 | yes | 38.10 % | 73340 | estimated:yosys-generic-mapping(stage5) |
| 9 (runner-up) | 735 | yes | 44.97 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 10 (runner-up) | 725 | yes | 48.96 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 11 (runner-up) | 715 | yes | 52.13 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 12 | 715 | yes | 56.12 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 13 | 705 | yes | 58.17 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 14 | 705 | yes | 61.16 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 15 | 705 | yes | 63.75 % | unavailable | unavailable:no-stage5-synthesis-for-this-build |
| 16 | 695 | yes | 64.86 % | 81847 | estimated:yosys-generic-mapping(stage5) |

### syn-mixed: budget=1 (infeasible)

Rejected: `no build in LANES 1..16 meets 1 cycles; best achievable is 695.0 cycles at LANES=16`
