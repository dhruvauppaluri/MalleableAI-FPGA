# Stage 0 baseline (2026-11-24)

Checkout: `d5069b3` on `claude/awesome-mendel-fqyyfl`, cloud container (4 cores).

| Command | Result |
|---|---|
| `make test` | PASS: 5 Icarus self-checking benches and 29 host unittests OK |
| `make verify` | PASS: test + Verilator lint (LANES 1/2/4/8) + Yosys coarse synth (LANES 1/2/4/8) |

Tools: Icarus Verilog 12.0, Verilator 5.020, Yosys 0.33, Python 3.13,
numpy 2.5.3, scikit-learn 1.9.1, torch 2.15.0, transformers 5.24.0.

The first `make test` attempt reported 7 import errors because optional
LLM/IDE dependencies (torch, tokenizers, fastapi) were not installed. After
`pip install -e '.[llm,ide,test]'`, the suite passed. No source code was changed.
Verilator lint reported no warnings with the Makefile's suppressions.
