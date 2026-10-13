# Radeon Qwen3 loading and single-tensor transfer diagnostics

The operator ran `tools/probe_amd_qwen_load.py` on the Radeon Cloud notebook,
under `timeout --signal=INT --kill-after=10s 2m`. The script identity in the
claim matches the published source. Import and GPU discovery finished in about
4.2 seconds. FP32 Qwen3-0.6B loaded into CPU memory by 5.991 seconds. The run
then entered `gpu_transfer_started` and never recorded completion. The operator
reported exit status 130. Repeated faulthandler snapshots from the pasted
console showed the main thread inside PyTorch `nn.Module.to`, `_apply`, and
`convert`. No forward pass, training, development score, or held-out review
occurred. An external timeout stopped the process; the exclusive claim and
events are preserved. The console transcript is retained as `console.txt`.

A separate, one-time 64 MiB CPU tensor copy to `cuda` was capped at 45 seconds.
CPU tensor creation completed in 0.15 seconds; the GPU copy never printed
completion and the command exited 130. The exact copy command and output were
provided by the operator in the Codex conversation; `single-copy.txt` records
them. After both processes ended, `rocm-smi` reported 0% GPU use and 0% VRAM
allocation, and the process listing found no matching training or probe job.
These observations isolate the delay to host-to-GPU transfer on this notebook
environment. They do not prove the underlying ROCm/driver, node, or container
cause, nor do they measure Qwen3 training quality. No more paired training
attempts should run on this instance until a basic tensor transfer completes
under a bounded check. Automatic program switching remains gated.
