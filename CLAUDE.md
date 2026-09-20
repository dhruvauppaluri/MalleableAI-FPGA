# Repository guidance

## Project boundary

This repository contains the verified RTL foundation for a malleable FPGA AI
accelerator. The software and ML toolchain is a greenfield contributor task: no
golden model, training pipeline, quantizer, model descriptor, exporter, or
hardware analyzer has been implemented.

Do not search repository history or external branches for a software
implementation. Design it independently from the requirements in
`docs/ml-contributor-roadmap.md`, `docs/numeric-contract.md`, and the observable
RTL interfaces.

## RTL ownership

- Preserve the signed INT8 and INT32 behavior in `docs/numeric-contract.md`.
- Do not modify verified RTL merely to make a software result pass.
- Treat `rtl/int8_mac.sv` and `rtl/int8_dot_product.sv` as the current hardware
  source of truth.
- Run `make test` before proposing changes that affect the hardware boundary.

## ML contributor ownership

The ML contributor owns the software architecture and implementation for:

- Model training or import
- Quantization and calibration
- Independent bit-accurate golden inference
- Model and accelerator descriptors
- Weights, biases, scales, and test-vector export
- Cross-checking exported results against RTL
- Model analysis and hardware-configuration search

Start with the smallest complete dense-network pipeline. Keep numeric behavior
explicit, documented, deterministic, and tested.
