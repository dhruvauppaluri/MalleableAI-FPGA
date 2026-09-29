# Qwen3 compatible INT8 candidates (recovery plan step 2)

Implements step 2 of [the quality recovery plan](qwen3-quality-recovery.md), CPU-only
and unit-tested on tiny synthetic models. No checkpoint has been run. No numeric
contract, ISA, RTL, frozen suite or threshold changed, so no new numeric ADR is
needed; the one storage-changing option (untying the shared head) is deferred in
[ADR-0005 draft](adr/0005-draft-untied-head-int8-candidates.md).

## Candidate space (12)

Strength `a` in {0 (baseline), 0.25, 0.5, 0.75} by weight clip in {none, 99.99th,
99.9th percentile}. Names are `a<strength>-c<clip>`, for example `a0.5-c99.9`.
`a0-cnone` is the unchanged baseline (empty derived set). Default screening skips it
because the 128-target baseline pilot already exists.

## Method

Rescaling uses SmoothQuant factors `s = max|X|^a / max|W|^(1-a)` per channel,
clamped to [1/256, 256], and folds them into neighbouring tensors so the unquantized
network is unchanged (float64 test tolerance 1e-9):

| Site | Activation quantized | Divide | Multiply |
| --- | --- | --- | --- |
| `attn_input` | shared `q/k/v` input | `input_layernorm.weight` | columns of `q_proj`, `k_proj`, `v_proj` (joint weight max) |
| `mlp_input` | shared `gate/up` input | `post_attention_layernorm.weight` | columns of `gate_proj`, `up_proj` |
| `attn_output` | `o_proj` input and V cache | rows of `v_proj` | `o_proj` columns of every query head in the GQA group |
| `mlp_down` | `down_proj` input | rows of `up_proj` | columns of `down_proj` |
| `qk` | K cache | `k_norm.weight` | `q_norm.weight` (equal across each RoPE pair) |
| `head_input` | head input | `model.norm.weight` | `lm_head` columns (untied only) |

The first five are the default (they cover projection inputs and the key/value
caches, the groups implicated by attribution). Clipping bounds each transformer
weight tensor at a per-tensor percentile of `|W|` taken after rescaling; it is lossy
and reported as `preserves_unquantized_operation: false`.

Tied embedding/head: the shared tensor is never modified. `head_input` rescaling
and head clipping raise an explicit error when `Spec.tied`. The existing INT8
runtime quantizer, group size 128 and storage are untouched; a candidate is only
different float32 tensors (norm weights stay normal float32, checked against FTZ).

## Data discipline and artifacts

Activation maxima come from the frozen suite's calibration split only (first
`--calibration-tokens`, default 512, causal, at most 128 per sequence). Ranking uses
the frozen 128-target validation panel (`--panel` must equal
`make_panel(suite,128,'spread')`); held-out is never read. Weight-clip percentiles
use weights only.

Separate content-hashed artifacts under `<attempt>/artifacts/`, written append-only:
`calibration-statistics-<sha256>.json`, `candidate-parameters-<sha256>.json`
(scales and clip thresholds), `derived-tensors-<sha256>.json` (per-tensor SHA-256
manifest). Derived safetensors (`--persist-derived`, about 2 GB each) are optional
because they regenerate bit-for-bit from source plus parameters. Source checkpoints
are opened read-only and returned tensors never alias them.

## Running (only after a batch mode is selected)

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PYTHONPATH=. .venv/bin/python -u tools/run_int8_candidates.py \
  --root build/zephyrus-jobs/release/20260928T223413Z-b463e746/int8-candidates-01 \
  --panel build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-attribution-01/panel-128.json \
  --reference-store build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-128-pilot-01/floating-references \
  --authorization "<quote the user's selected mode>"
```

The tool requires a clean committed source, takes the shared precision lock, refuses
an existing root, and runs calibration then one candidate per process, serially.
`report.json` ranks by agreement (descending), candidate NLL (ascending), then listed
order. Only candidates meeting >=90% agreement and <=5% NLL degradation on all
128 targets enter `top_three`; failed results stay visible in the ranking.
Derived-only tensor overrides and memory budgeting avoid retaining a duplicate
original checkpoint during screening. Use
`--candidates a0.5-cnone,a0.5-c99.9` for an Interactive subset. Expect calibration
of 512 tokens (a few minutes; float64 emulation, about 0.26 s/token) plus about
4.5 minutes per candidate from the 128-target pilot (256.5 s candidate time), so
11 candidates are roughly 50-60 minutes.

Screening results are independent emulation of derived tensors, not ISA/RTL and not
release evidence. Next after screening: complete validation of the top three through
the actual ISA (goal at least 92% agreement, gate 90%), then a single held-out
evaluation of a frozen winner.
