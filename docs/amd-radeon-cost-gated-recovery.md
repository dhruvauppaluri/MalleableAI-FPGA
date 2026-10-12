# Cost-gated Radeon Cloud recovery for Qwen3 pilot

## Observed blocker

The original paired real-text attempts 01 and 02 are consumed without quality
scores. An isolated Qwen3 probe loaded the FP32 checkpoint into CPU memory in
about 1.8 seconds, then blocked during `.to("cuda")`. A separate 64 MiB host-
to-GPU tensor copy timed out after 45 seconds. The container's memory events
showed no recorded OOM kill. This is a GPU transfer problem on the observed
Radeon notebook, with the exact node/runtime cause still unknown. No training
recipe adjustment can establish quality while basic transfers fail.

## Before launching another paid instance

Keep the old attempt roots immutable. Have `tools/probe_amd_transfer.py` and
`tools/probe_amd_qwen_load.py` available locally for browser upload or verified
terminal paste, because the notebook's GitHub HTTPS and raw HTTPS routes returned
503/SSL EOF. The health checks need no model download, corpus, or repository
clone. Check the Radeon credit balance and arrange to destroy the instance as
soon as the bounded checks finish. A process timeout does not stop instance
billing.

## On a fresh Radeon instance

Run the default transfer check once, in an exclusive root:

```sh
timeout --signal=INT --kill-after=10s 30s \
  python -u tools/probe_amd_transfer.py \
  --output-dir /workspace/amd-transfer-default-01
```

The check copies four bytes and then 64 MiB from CPU to GPU. It records
fsynced phase events. If both copies complete, keep the default ROCm mode.
If either copy blocks or times out, preserve the claim/events/console output
and try the ROCm SDMA-disabled mode once in a distinct root:

```sh
HSA_ENABLE_SDMA=0 timeout --signal=INT --kill-after=10s 30s \
  python -u tools/probe_amd_transfer.py \
  --output-dir /workspace/amd-transfer-sdma-off-01
```

`HSA_ENABLE_SDMA=0` is a diagnostic transfer-path workaround. It must be set
before importing PyTorch, and its performance effect must be measured. If both
modes fail, stop and destroy the instance; send the recorded logs to Radeon
Cloud support. Do not spend credits downloading a checkpoint or training.

If a transfer mode succeeds, use that same environment mode for the one-model
Qwen3 load/forward probe in `docs/amd-qwen-load-probe.md`, with its two-minute
external cap and a new output root. If CPU load, GPU transfer, and forward all
complete promptly, estimate model-loading and one-step costs before freezing a
new paired development attempt. Freeze and claim in fresh `03` roots; neither
prior attempt can be resumed. Existing train/development text can be reused
for development, while all release held-out suites and the 30 published RTL
benchmarks remain untouched. Keep automatic program switching gated.

These checks are hardware/runtime diagnostics. Passing them does not approve
matrix sharing, a checkpoint, an FPGA personality, or program switching.
