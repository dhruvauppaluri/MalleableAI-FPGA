# Shared-matrix real-text attempt 01: interrupted

The operator ran the frozen Qwen3-0.6B paired development pilot on an AMD
Radeon Cloud instance with a reported 48 GiB GPU. The repository records the
verbatim claim, freeze, and corpus manifest pasted from the notebook terminal.
The freeze selected 128 training and 32 development sequences from the pinned
WikiText-2 train and validation splits. No release held-out suite was opened.

The process printed no step-10 milestone after at least 27 minutes elapsed.
At the observed snapshot, `ps` reported PID 531 running at about 101% CPU;
`rocm-smi` reported GPU use 100% and VRAM allocation 1%. These snapshots do
not locate the slow operation or establish how many of steps 1–9 completed.
Repeated Ctrl+C did not promptly stop the process. The operator force-stopped
it, and the shell printed `Killed`. The runner did not write `result.json` or
`failure.json`; only `claim.json` remained in the attempt directory. The
operator supplied these logs and manifests in the Codex conversation. This is
an operator-reported interruption, not a scored failure of the matrix-sharing
quality gate. No weights or development metrics were produced.

**Disposition:** attempt 01 is consumed. Never rerun its claim directory or
reinterpret it as a pass. Add phase and per-step progress with elapsed times,
and bound the paid process externally before freezing a distinct follow-up
attempt. Automatic program switching remains gated.
