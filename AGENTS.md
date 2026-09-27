# Agent guidance

Cursor, Codex, and Claude Code all read this file. It is the shared project
memory. Chat history is not shared. Leave decisions, status, and code in the
repo.

`docs/STATUS.md` is the handoff. Read it before starting, and update it before
you stop.

## Project

MalleableAI-FPGA is a model-aware FPGA AI accelerator. The FPGA runs inference.
Training, quantization, hardware selection, and FPGA compilation happen on the
host. Stage 1 is a standalone FPGA accelerator. Stage 2 (GPU) and Stage 3
(hybrid) stay out of scope until Stage 1 meets the criteria in
`docs/research-roadmap.md`.

The verified RTL foundation is in this repo. The software and ML toolchain is
greenfield: no golden model, training pipeline, quantizer, model descriptor,
exporter, or hardware analyzer has been implemented.

Do not search repository history or external branches for a software
implementation. Design it from `docs/ml-contributor-roadmap.md`,
`docs/numeric-contract.md`, and the observable RTL interfaces.

## Shared rules

- Preserve the signed INT8 and INT32 behavior in `docs/numeric-contract.md`.
- Do not modify verified RTL merely to make a software result pass.
- Treat `rtl/int8_mac.sv` and `rtl/int8_dot_product.sv` as the current hardware
  source of truth, along with the later verified stages:
  `rtl/int8_tiled_accumulator.sv`, `rtl/int8_postprocess.sv`, and
  `rtl/malleable_accelerator_top.sv`.
- Record a numeric or interface change in `docs/numeric-contract.md` or a new
  file under `docs/adr/` before code depends on it.
- Run `make test` before proposing changes that affect the hardware boundary.
  Run `make verify` when the change can affect lint or synthesis.
- Start software work with the smallest complete dense-network pipeline. Keep
  numeric behavior explicit, documented, deterministic, and tested.

## Tasks

Cursor, Codex, and Claude are interchangeable. Any of them may pick up any open
task in `docs/STATUS.md`. Nothing is reserved for one tool.

Claim a task before starting so two agents do not do the same work:

1. Read this file and `docs/STATUS.md`.
2. Choose any row whose status is `open`. Prefer a task whose dependencies are
   already done.
3. Set that row to `in progress` and put your name on it, then start.
4. Put the result in git: code, a doc, or both. Open a PR for anything that
   should be reviewed.
5. Set the row to `done`, add the PR, and note anything newly unblocked. If you
   stop early, set it back to `open` and write what is left.

The next agent starts from `docs/STATUS.md`, not from your chat.

## Commands

```sh
make test      # Icarus Verilog self-checking simulations
make verify    # simulation, Verilator lint, and Yosys synthesis
```

Quartus Prime Lite 25.1, when installed:

```sh
quartus_sh --flow compile quartus/int8_mac/int8_mac
quartus_sh --flow compile quartus/int8_dot_product/int8_dot_product
```

The Quartus projects still cover the MAC and dot-product blocks. They do not
yet compile `malleable_accelerator_top`.
