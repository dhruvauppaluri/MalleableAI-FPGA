# Radeon Qwen3 model-loading probe

The first paired real-text training attempt was force-stopped with no step-10
milestone. A second, five-minute diagnostic reached `loading_teacher` but never
reached `teacher_loaded`; see `docs/evidence/shared-matrix-real-text-20261012-attempt01/`
and `docs/evidence/shared-matrix-real-text-20261012-attempt02/`. Neither attempt
produced a quality result, and both claim directories are consumed.

`tools/probe_amd_qwen_load.py` isolates one unchanged Qwen3-0.6B model from
training and development data. It records library import, CPU checkpoint load,
GPU transfer, and one four-token forward pass as separate phases. It uses the
same deterministic load settings and FP32 model construction as the paired
runner, writes an exclusive claim and fsynced phase events, and emits Python
stack traces every 20 seconds while blocked. It does no training or quality
scoring and opens no held-out suite.

Run only after the model and probe script are present on the Radeon host:

```sh
timeout --signal=INT --kill-after=10s 2m \
  python -u tools/probe_amd_qwen_load.py \
  --model /workspace/shared-models-01/Qwen3-0.6B \
  --output-dir /workspace/amd-qwen-load-probe-01
```

The two-minute timer begins when this command starts; cloud billing continues
until the instance is destroyed. Preserve `claim.json`, `events.jsonl`, any
`result.json` or `failure.json`, terminal output, exit status, and container
memory events. An external SIGKILL can prevent a result/failure file, but the
events file remains. Never rerun the same output directory. Do not initiate
another paired training attempt until this probe identifies the delayed phase
and the cost of a full run can be estimated.
