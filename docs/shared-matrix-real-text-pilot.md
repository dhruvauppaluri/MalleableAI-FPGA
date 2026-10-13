# Shared-matrix real-text development pilot

## Status and purpose

This is a prospective, bounded **development** experiment for Qwen3-0.6B.
The earlier CPU and AMD synthetic diagnostics remain failed under their
original frozen 90% teacher-agreement rule. Their seeds and validation data
are consumed and are not inputs to this experiment.

The pilot asks whether sharing one adjacent layer pair's MLP gate/up weights
degrades real-text next-token quality relative to an ordinary control trained
from exactly the same modified initial weights. This pilot cannot approve a
checkpoint, a hardware personality, or automatic program switching.

The current OpenTPU `shared_matrices` layout shares selected matrix kinds
**across all layers**. It cannot encode this pilot's selective layer pair.
Hardware integration would require a separately reviewed compiler/ISA layout,
numeric checks, fresh RTL performance evidence, and eventually F2 measurements.
The previously projected 28.11% full-layer image reduction does not apply to
this limited pair.

## Prospective frozen recipe

The source of truth is `RECIPE` in `tools/run_shared_matrix_finetune.py`:

| Item | Frozen value |
| --- | --- |
| Model | Pinned public Qwen3-0.6B, local safetensors |
| Shared parameters | Layer 0 and 1 `mlp.gate_proj.weight` and `mlp.up_proj.weight` |
| Initialization | Copy layer 0 gate/up into layer 1 in both students; only the shared student aliases the parameter objects |
| Trainable weights | Those four projections in the ordinary student; the two shared parameters in the tied student |
| Teacher | Original, unchanged pretrained checkpoint |
| Training objective | 50% next-token cross entropy + 50% teacher KL at temperature 2 |
| Train budget | 40 steps, batch 1, sequence length 128, AdamW 1e-4, weight decay 0.01, seed 20261013 |
| Development sample | 128 train and 32 development sequences, each 129 tokens including the next-token target |
| Wall budget | 20 minutes in the runner after its input preflight |
| Development gate | Shared NLL <= 1.05 × ordinary NLL and shared next-token accuracy >= ordinary accuracy minus 0.01 |

The 90% absolute synthetic teacher-agreement gate is **not changed or
retroactively applied** to this new real-text development question. Both
students' teacher agreement is reported, separated by whether the teacher's
prediction was correct. A development pass permits designing a later
independent candidate evaluation; it is not a release pass.

The opt-in corpus helper pins a specific revision of
`Salesforce/wikitext`/`wikitext-2-raw-v1`, writes bounded train and
validation JSONL files, and records file hashes. It does not score the public
test split. WikiText is public and may overlap Qwen's pretraining data, so
this corpus is unsuitable as independent release held-out evidence. A user
may instead supply separate UTF-8 JSONL files whose records contain `text`.
Exact duplicate documents across train/development are rejected.

The `freeze` command records the checkpoint file hashes, input file hashes,
token sequence hashes, recipe, and runner source hash before a run. `run`
rechecks them, writes an exclusive claim directory, trains both students on
the same batches, records conditional quality counts, and exports only the
modified projections as local safetensors. It writes either `result.json` or
`failure.json`; the attempt directory must never be reused. No release
held-out split is accepted by the runner.

The first Radeon attempt was force-stopped after at least 27 minutes with no
step-10 output. Its claim and matching freeze/corpus manifests are preserved in
`docs/evidence/shared-matrix-real-text-20261012-attempt01/`. The exact slow
operation and number of completed steps are unknown; this is not a quality
result. That attempt directory is consumed. The runner now writes phase and
per-step events to `progress.jsonl` and standard output. The internal 20-minute
limit is checked between operations, so it is not a hard process deadline.
Use an external timeout around any new paid attempt and inspect its progress
file before deciding whether another attempt is justified.

Attempt 03 completed on a fresh, healthy Radeon notebook in 16.393 seconds.
It failed the frozen development gate: shared NLL 7.375570 versus ordinary
6.732002 (+9.56%), and shared next-token accuracy 12.2559% versus ordinary
15.4785% (-3.223 percentage points). Both students were far below the
unchanged teacher (NLL 3.724158, accuracy 38.9404%). See
`docs/evidence/shared-matrix-real-text-20261013-attempt03/` for the freeze,
claim, progress, result, and projection hashes. Do not repeat attempts 01–03.
The zero-step screen then scored all 27 adjacent gate/up copies on the same
frozen development set. The 0→1 copy caused +255.18% NLL and -36.743 accuracy
points immediately. Pair 12→13 was least damaging by NLL (+2.684%, -1.733
accuracy points); see `docs/evidence/shared-matrix-pair-screen-20261013-01/`.
This is development selection, not independent quality evidence.

The next prospectively selected recipe uses pair 12→13 with the same 40-step
matched training budget and equal student starting weights. It retains the
ordinary-control comparison and adds an untouched-teacher anchor: shared
development NLL must be at most 105% of teacher NLL and teacher next-token
agreement must be at least 90%. Both checks are necessary to avoid passing
when both students are damaged. The result will still be developmental because
pair 12→13 was selected on this same corpus. It cannot approve a checkpoint
or automatic switching.

Attempt 04 finished in 15.903 seconds. Shared NLL 3.618630 was close to
ordinary 3.591713 and below teacher 3.724158. Its teacher top-token agreement
was 81.8604%, so the predeclared 90% teacher anchor and combined development
gate failed. See `docs/evidence/shared-matrix-real-text-20261013-attempt04/`.
Do not open release held-out data for this candidate or reuse attempt 04.

## Bounded Radeon Cloud procedure

Run these commands in the notebook terminal, one at a time. The model
download is explicit and limited to Qwen3-0.6B; keep it outside git. Check
remaining credits before launch and destroy the Radeon instance after saving
the small JSON result. Downloading and preparation also consume instance time.
Replace `NEXT` below with a fresh attempt identifier after reviewing the
attempt-03 failure; the existing attempt directories cannot be reused.
For the screened 12→13 follow-up, use `04` and add the two layer arguments
shown here. The prior 0→1 run used the default arguments.

```sh
git clone --depth 1 --branch codex/policy-correction https://github.com/dhruvauppaluri/MalleableAI-FPGA.git
cd MalleableAI-FPGA
python -m pip install 'datasets>=3,<5'
python tools/download_llm_models.py --only Qwen3-0.6B --destination /workspace/shared-models-01
python tools/prepare_shared_matrix_text.py --output-dir /workspace/shared-text-01
python tools/run_shared_matrix_finetune.py freeze \
  --model /workspace/shared-models-01/Qwen3-0.6B \
  --train-jsonl /workspace/shared-text-01/train.jsonl \
  --dev-jsonl /workspace/shared-text-01/dev.jsonl \
  --source-layer 12 --target-layer 13 \
  --output /workspace/shared-pair-freeze-NEXT.json
timeout --signal=INT --kill-after=30s 25m \
  python -u tools/run_shared_matrix_finetune.py run \
  --freeze /workspace/shared-pair-freeze-NEXT.json \
  --output-dir /workspace/shared-pair-attempt-NEXT
cat /workspace/shared-pair-attempt-NEXT/result.json
```

If a command fails, preserve its output and any `failure.json`; do not
rerun the same attempt directory. Publish the freeze, corpus manifest,
`progress.jsonl`, result/failure JSON, and hashes of the two projection files.
An external kill may leave only the claim and progress file; record that
interruption without inferring model quality. Keep model
weights and corpus files out of the repository. The only independent
held-out evaluation would be designed and frozen after the development
candidate is selected. Do not reuse any consumed quality suite or the 30
published RTL benchmark tapes.

## Decisions after the development pilot

- If the development gate fails, record the failure. Investigate whether
  different layer pairs, a larger trainable scope, or a different objective
  is justified using new development attempts and distinct attempt roots.
- If it passes, freeze one candidate and a **new** independent real-text
  quality suite before opening it exactly once. Apply the project's real-model
  quality limits, including next-token agreement and NLL degradation against
  the unchanged pretrained checkpoint. Keep this separate from the WikiText
  development score.
- Only after quality approval, implement selective matrix placement and
  compare it with ordinary layout on new matched RTL workloads. Board fit,
  runtime benefit and switching cost remain unmeasured until F2 porting.
