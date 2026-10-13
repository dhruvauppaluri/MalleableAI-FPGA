# Zero-step adjacent gate/up pair screen

`tools/screen_shared_matrix_pairs.py` diagnoses the large common quality drop
in shared-matrix development attempt 03. It reuses that attempt's frozen
model and development token sequences. Before any training, it scores the
untouched Qwen3-0.6B model, then for each of its 27 adjacent layer pairs:

1. Save the target layer's gate/up weights in memory.
2. Copy the source layer's gate/up weights into the target.
3. Score the same 32 development sequences (4,096 next-token targets).
4. Restore the target weights and verify exact restoration.

The copied-weight model has the same forward computation as a freshly tied
student before training. This screen measures the immediate functional damage
of copying. It does not measure whether later optimization can repair that
damage or whether shared parameters help after training. NLL is the primary
ranking statistic; accuracy breaks ties. `pair-results.jsonl` and
`progress.jsonl` are flushed after each pair, so an interrupted attempt still
shows which pairs completed. `result.json` exists only when all 27 finish.

The tool verifies the earlier freeze's source, model, corpus and token hashes.
It creates a new exclusive claim directory, binds its own source hash and the
full candidate list before scoring, and refuses to reuse an attempt root.
Development data may be used for this design choice; it cannot provide
independent held-out or release evidence. The prior three training attempts
remain consumed. No RTL benchmark or automatic switching is involved.

On a healthy Radeon notebook with the existing freeze and model, use a fresh
output root and an external cap:

```sh
timeout --signal=INT --kill-after=10s 6m \
  python -u tools/screen_shared_matrix_pairs.py \
  --freeze /workspace/shared-pair-freeze-03.json \
  --output-dir /workspace/shared-pair-screen-01
```

Preserve `claim.json`, `progress.jsonl`, `pair-results.jsonl`, `result.json`
or `failure.json`, and the console output. Do not repeat a claimed screen
root. The next training candidate must use a separate prospective recipe and
fresh attempt root. Passing a development screen would not approve matrix
sharing or permit automatic program switching.

Screen 01 completed all 27 pairs in 33.844 seconds. The full immutable
result is in `docs/evidence/shared-matrix-pair-screen-20261013-01/`. Pair
0→1 caused +255.18% NLL and -36.743 points accuracy immediately after copy.
The best pair by NLL was 12→13 at +2.684% NLL and -1.733 points accuracy.
This screen is consumed development evidence; do not rerun it as an
independent confirmation.
