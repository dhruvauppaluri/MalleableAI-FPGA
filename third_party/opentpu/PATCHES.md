# Local integration patches

Original source: FeSens/openTPU, revision in UPSTREAM.json. Apache-2.0 LICENSE
is retained. Our adapters live in malleable/llm, not in upstream model kernels.

- rtlsim.py: cache includes tool identity; bounded compile concurrency; optional
  streaming callback and per-run trace files; return-code validation.
- sim/verilator/tb_top.sv: optional periodic simulation heartbeat with flushing.
- opentpu/llm/qwen35.py: quantize row-independent model-image matrices in
  bounded row chunks. This preserves packed bytes and addresses while avoiding
  multi-gigabyte temporary arrays for the 0.8B model's vocabulary embedding.

No matrix/vector numeric behavior is changed. The simulation top and behavioral
memory are not a physical FPGA shell. Private media and model weights are absent.
# Space-containing checkout paths

The generated Verilator make invocation fails for this repository's path.
`rtlsim.build` stages source files in a source/tool/parameter-hashed cache under
the OS temporary directory. Original source contents remain the hash inputs.
No RTL arithmetic changes are made. Loopback Lens serving labels upstream clock
calculations as assumed projections, not physical measurements.
# Completion review

`rtlsim.build` now locks each cache key across processes, publishes a completion
marker only after successful linking, keys compiler/OS/flags as well as sources
and Verilator, and bounds compilation to 900 seconds and two parallel processes.
No numerical RTL or ISA semantics are changed.
