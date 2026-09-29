# Cursor continuation prompt — Zephyrus quality recovery

Continue this work directly in Cursor's LOCAL Agent on my Zephyrus laptop.
Execute the work below; do not stop at a plan or ask me to repeat choices already
made here. I choose **Overnight mode** for this continuation: serialize heavy
jobs, stop starting new jobs eight hours after the first new dispatch, and let
an active job finish. Use that same mode for the prescribed follow-up batches
within this window. Record the dispatch deadline and check it before each job.
Do not start another eight-hour window automatically. Routine code fixes,
tests, durable job creation, candidate integration and the conditional ten-case
investigation are authorized. No clarification about run mode, checkpoint
locations, thresholds, candidates or whether to proceed is needed.

## 1. Work on the correct machine and source

Linux checkout: `/home/dhruv/projects/MalleableAI-FPGA`.
WSL distribution/user: `Ubuntu-24.04`, `dhruv`.
Branch: `codex/local-llm-platform`.
Origin: `https://github.com/dhruvauppaluri/MalleableAI-FPGA.git`.
Windows setup workspace: `C:\Users\dhruv\OneDrive\Documents\ChatGPT\malleablefpga`.
Windows access to the Linux checkout:
`\\wsl.localhost\Ubuntu-24.04\home\dhruv\projects\MalleableAI-FPGA`.

If your tools use PowerShell, invoke Linux commands with
`wsl -d Ubuntu-24.04 -u dhruv --exec bash -lc '<Linux commands>'`.
Prefer opening the project through Cursor's WSL environment. The old Cursor
cloud agent cannot inspect these local ignored checkpoints/evidence. A GPU or
cloud worker is not needed for ISA quality or the precision diagnostics: they
run on CPU. If you are actually in a cloud-only environment with no route to
this laptop, report that exact environment limitation and continue read-only
code review/handoff work; do not pretend a prompt creates local access or
download replacement models. The intended executor is local Cursor on Zephyrus.

Read these first:
- `AGENTS.md` and `docs/STATUS.md`.
- `docs/qwen3-quality-recovery.md`.
- `docs/qwen3-int8-candidates.md`.
- `docs/qwen3-precision-attribution-results.md`.
- `docs/adr/0006-diagnostic-key-cache-precision.md`.
- `docs/zephyrus-release-progress.md` and `docs/local-llm-platform.md` for the
  downstream release gates.

Codex's last result commit before this handoff was
`634e2215b53e3eb9dd31a063b677fda484b97233`. Use the latest descendant on the
branch, including this handoff and `tools/run_candidate_full_validation.py`;
do not reset back to that historical commit. Inspect branch, dirty state,
running processes, the shared lock and existing attempt folders first. Only
fast-forward/pull before dispatch when no worker is active and the checkout is
clean. Preserve concurrent changes; never force-push or discard another agent's
work. Do not change source or pull while a source-bound job is active.

Use `.venv/bin/python`, not the symlink-resolved system Python. For execution:
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PYTHONPATH=.`. For RTL verification also add `/home/dhruv/.local/bin` to PATH.

## 2. Trust this completed state, not the older cloud handoff

Release evidence root, relative to the Linux checkout:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/`.
Call it RELEASE below. Checkpoints are under `build/models/` and are ignored by
Git. They exist locally; do not request them again or regenerate frozen suites.

Completed experiments:
- `precision-attribution-01`: all TEN 16-target cases completed.
- `precision-128-pilot-01`: baseline 107/128 = 83.59375%, NLL +0.622021%.
- `precision-128-remaining-01`: all NINE remaining cases completed in 36m08s.
  Key-cache floating bypass reached 117/128 = 91.40625%, NLL -0.363486%.
  All-floating matched 128/128. No cases are missing; do not rerun them.
- `quality-recovery-pilot-01`: compatible INT8 pilot completed in 620.25s.
  Rescaling-only `a0.5-cnone`: 121/128 = 94.53125%, NLL -0.177878%.
  Rescaling + 99.9th-percentile clipping: 5/128 = 3.90625%, NLL +135.676582%;
  rejected. Preserve the failure and do not deploy or repeat that clipping case.
  Calibration used all 256 available calibration tokens; 512 was an upper limit.
  Every one of the 308 persisted tensor hashes was verified for both candidates.
- `candidate-isa-pilot-01`: ACTUAL ISA 15/16 = 93.75%, NLL +0.634761%, 86.54s.
  Original FP32 NLL 6.240925408; ISA NLL 6.280540376. Both floating references
  agree 16/16, all 311 original tensors match, ISA/emulation agree 15/16.
  One mismatch is position 4, with a narrow original margin of 0.0331.
  The independent emulator omits ISA rounding; it is not an exact ISA oracle.
  This is a passing small diagnostic, not held-out or RTL acceptance.

The above jobs are complete. PIDs 2021, 3215, 3837, 22182 and 24214 are
historical; never treat a matching reused PID as proof that their jobs are active.
Inspect the actual command/process and artifacts. Full 1,024-target candidate
validation had NOT started when Codex prepared this handoff.

Exact candidate to validate:
`RELEASE/quality-recovery-pilot-01/int8/case-00` (not case-01).
Name `a0.5-cnone`: strength 0.5, no clipping; INT8 matrices and INT8 output head.
Context 128, balanced personality, original checkpoint/tokenizer.
This is calibrated INT8 rescaling, not a mixed-precision cache implementation.

IDs:
- Derived: `3129232e64ec1532cfb735b48dac385bbeac4a072af98660d0fe85953cd163da`.
- Parameters: `3840e2dbae1647c8b47984167fa6ba40fc5f6de044d7a4d905444bac8fbe2a08`.
- Calibration: `0e413dfe4e2718d421621006377b174169c3bb54cf4a4157b3a4e9f0a9e9dbf1`.
- ISA pilot result SHA-256:
  `03adbf677fe1b67ce0b274f384a8fcded41dc7849e87baaf84f0c4774b6ba7d3`.

Derived files are under `RELEASE/quality-recovery-pilot-01/int8/artifacts/`.
The passing safetensors file is `derived-<derived-ID>.safetensors`, 1,761,900,640
bytes. Windows evidence copies omit these large files; use the original Linux
artifact. Never overwrite it or modify the source checkpoint.

Reviewed PR #4 research code is already incorporated and fixed on this branch.
Do NOT merge/cherry-pick the older PR #4 to obtain it again. Quality filtering now
recomputes the loss gate, rejects mismatched evidence, and excludes failed cases
from eligible top-three results. Derived-only loading avoids a duplicate source
checkpoint. The candidate loader checks calibration/source/tensor hashes.
Actual ISA interfaces `quality-candidate-diagnose` and
`quality-candidate-validate` exist. The latter currently deliberately rejects
held-out evaluation. Latest focused checks: 88 passed in 21.15s. Full verification
of old source `b1f2d3f` is historical and cannot substitute for final-source checks.

## 3. Start durable full validation now, without another approval question

Inspect `candidate-isa-validation-01` first. If absent, use that root and the unit
name below. If an attempt exists, inspect job/child PIDs, service, logs, canonical
records and result hashes. Monitor an active valid worker. Reuse a verified
completed result. Preserve failed/interrupted attempts and choose the next unused
numeric suffix for a retry; never overwrite a folder or duplicate an active job.
Claim the quality-recovery task in `docs/STATUS.md`, record this authorization
and planned command, and commit clean source before dispatch.

Launch from Linux with this command (replace BOTH `01` suffixes only if needed):

```sh
cd /home/dhruv/projects/MalleableAI-FPGA
systemd-run --user --unit=malleable-qwen3-full-validation-01 \
  --property=Type=exec --property=RemainAfterExit=yes \
  --working-directory=/home/dhruv/projects/MalleableAI-FPGA \
  --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=8 --setenv=PYTHONPATH=. \
  /home/dhruv/projects/MalleableAI-FPGA/.venv/bin/python -u \
  /home/dhruv/projects/MalleableAI-FPGA/tools/run_candidate_full_validation.py \
  --root /home/dhruv/projects/MalleableAI-FPGA/build/zephyrus-jobs/release/20260928T223413Z-b463e746/candidate-isa-validation-01 \
  --mode overnight \
  --authorization "User Cursor continuation prompt: Overnight full ISA validation and prescribed follow-up within eight hours"
```

The launcher takes the shared heavy-job lock and records source/PIDs/command,
start/deadline, raw worker output, canonical result, checksummed result/summary
and any failure. `job.json` defines the first dispatch deadline; use that SAME
deadline for all continuation jobs. Do not restart the clock for each attempt.
The process runs independently of agent credits and survives an Agent chat
disconnect through user systemd. Keep the existing WSL attachment alive and the
laptop awake. Do not kill a job because an agent/tool response times out.

Monitor durable files and, as useful:
`systemctl --user show malleable-qwen3-full-validation-01 -p ActiveState -p SubState -p MainPID -p ExecMainStatus`
and `journalctl --user -u malleable-qwen3-full-validation-01 -n 30 --no-pager`.
Use bounded waits and give concise progress updates. On disconnect, the next
agent reads the saved files/service rather than starting another copy.

Expected duration: 90–150 minutes, estimated from the completed 86.54s pilot.
The job evaluates ALL frozen validation targets (>=1,024) through actual ISA,
against ORIGINAL FP32, context 128/balanced/INT8 matrices/INT8 head. Candidate
memory is preflighted; do not bypass it or silently change backend/settings.
The original checkpoint/tokenizer and suite are checked again after execution.
No held-out data is evaluated in this command.

After completion independently recompute the result hash and gates. Pass requires
agreement >=0.90 AND candidate NLL <=1.05 * original FP32 NLL; target_count and
samples must match the entire frozen validation split. Reaching 92% is a useful
buffer, not a replacement threshold. An exit code 0 alone is not a quality pass.
Preserve negative results. A worker/lineage/nonfinite/infrastructure failure is
not evidence that precision is inadequate: fix that defect and retry in a new
attempt before invoking numerical redesign.

## 4. If full validation fails numerically, run these TEN distinct cases

This conditional investigation is already authorized. Do not ask again and do
not repeat the two INT8 pilot candidates or the earlier bypass schedules.
Use a fresh `RELEASE/precision-isa-failure-01/` (next unused suffix if needed).
Freeze the policies, source and panel hash before dispatch. Use the SAME
`RELEASE/precision-attribution-01/panel-128.json`, full causal prefixes (996
executed context tokens), and matching original FP32 reference records from
`RELEASE/precision-128-pilot-01/floating-references`. Copy the closed reference
store into the new attempt; do not modify the historical store.

Run the fixed policy-v2 cases from ADR-0006 and
`quality-recovery-pilot-01/plan.json`, on the ORIGINAL checkpoint, one per process:

1. Key cache float32.
2. Key cache float16.
3. Key cache INT16, block 128.
4. Key cache INT8, block 64.
5. Key cache INT8, block 32.
6. Key cache INT8, block 16.
7. Key cache and query float32.
8. Key cache and query float16.
9. Key cache and query INT16, block 128.
10. Key cache and query INT16, block 64.

Each policy has schema_version 2, all seven quantized_groups in the existing
canonical order, group_size 128, and site_overrides at `key_store` and optionally
`query`, with `format` and `block`. All other groups retain baseline INT8.
Use the existing `quality-precision-diagnose` interface with `--precision-policy`,
`--diagnostic-panel`, `--context 128 --personality balanced --wformat int8
--split validation --max-host-gib 16`, and a new store. Record policy hashes,
source, full command, PIDs, events, per-token predictions/margins/NLL, operator
differences, canonical IDs, result hashes and host timings. Aggregate only
verified completed cases and preserve failed attempts separately. Estimate the
remaining time from the first completed case; check the original deadline before
each dispatch. Typical entire ten-case duration is roughly 40–45 minutes.

Do NOT rerun `tools/run_quality_recovery_pilot.py` as a fallback-only launcher:
it would repeat the completed INT8 pilot and skip fallback because that screening
pilot passed. Make a small serialized fallback-only wrapper using the existing
diagnostic interface and predeclared policy list. Test that wrapper on tiny data
as needed; commit source before real-model execution.

These ten are emulation diagnostics, not ISA/RTL implementation or selectable
release evidence. If they identify a useful change, use it to choose the smallest
compatible improvement. Diagnose any ISA/emulator discrepancy before assuming
all differences are expected rounding. Write/update an ADR BEFORE production
arithmetic/storage/layout changes. Prefer existing FP32 primitives when viable;
smaller cache groups must have explicit scale/layout/compiler/ISA/RTL support.
Global D is coupled to MXU depth, and AXI assumes D=128; never silently change
memory backend or call a local emulation block-size change deployed hardware.
Use calibration for fitting, validation for selection; leave held-out untouched.
Verify the proposed implementation on affected operators/tiny models and actual
ISA full validation. If no bounded candidate passes, save the measured diagnosis,
quality/precision frontier and concrete next design; do not claim a release pass.

## 5. If full validation passes, integrate and freeze before held-out

Keep the passing candidate's derived tensors and calibration parameters fixed.
Current held-out rejection is intentional. Implement the missing candidate
generation/evidence plumbing before exposing held-out approval; do not remove
the restriction and evaluate while identities are incomplete.

Use the checked derived artifact loader for ISA AND RTL generation, quality,
benchmarks, workbench recommendations/application and release checks. Bind
original base model/tokenizer, candidate/derived/parameter/calibration IDs,
quantization/precision policy, personality, configuration/context, frozen suite,
source and toolchain. The original FP32 reference must always remain the original
checkpoint, never the transformed candidate. Preserve legacy baseline identities
and historical records. New calibrated INT8 evidence must not silently match
old raw INT8 evidence. Keep tied embedding/head handling correct; no untying.

Regression coverage: corrupted/mismatched derived artifacts, original-reference
preservation, insufficient targets, split leakage, failed quality, config/context
mismatch, variant identity, generation loading, cache reset/re-prefill and
compiler/ISA/RTL parity on synthetic cases. Current INT8 rescaling changes weights
and norms, not the dense numeric contract. Do not weaken tests to force a pass.

Freeze code commit, original model/tokenizer hashes, derived tensor hashes,
calibration/parameters, precision and exact configuration before held-out.
Then evaluate that candidate ONCE on the unchanged frozen held-out split with
>=1,024 targets, actual ISA, same context128/balanced/INT8/headINT8 and original
FP32 reference. This was authorized in the approved recovery plan; no additional
permission question is needed within the selected window. If held-out fails,
record the failure and stop promotion. Do not tune against held-out or repeatedly
test it. Further search stays on calibration/validation and needs a documented
new evaluation design rather than reuse of held-out feedback.

If held-out passes, run the existing short benchmark prompt plus EIGHT greedy
tokens completely in full RTL, checking DRAM and TMEM at every executed step.
An early EOS with fewer than eight tokens does not satisfy acceptance. Obtain
fresh durable 100-token synthetic RTL evidence. Use pilot timings before larger
jobs. Validate Qwen3.5-0.8B and LFM2.5-230M independently on their own frozen
suites; do not assume Qwen3 rescaling or quality transfers across architectures.
Standalone schema-v3 `release-check` must pass for all three required models.

## 6. Continue the approved release checklist, time permitting

Read the existing implemented interfaces rather than rebuilding them. Preserve
all gates from the approved plan; finish only what can be honestly verified.

- Performance: matching validation/held-out quality for controller personalities;
  balanced INT4/FP4 validation. Three seed-42 manifests with existing fixed token
  tapes and declared memory scenarios; TEN verified indexed RTL runs per model,
  30 total. Eight INT8 comparisons followed by both four-bit formats only if
  both pass validation; otherwise the prescribed two INT8 AXI stress runs.
  Preserve failed attempts; resume only between indices. Bind every setting,
  cycles/traffic/utilization, compile/host time and microarchitecture metadata.
- Predictor/controller: schema v2 with FIFO depth and bandwidth percentage and
  correct fixed-tape work. Train on Qwen3/Qwen3.5 only; LFM is held out from
  fitting/tuning. Freeze predictor before held-out predictions. Primary controller
  covers four INT8 personalities. Reproducible stored-ID episodes for latency
  and throughput, horizons 1/8/32/128 and explicitly assumed switching costs.
  Double DQN seed 0, 20 passes; evaluation seeds 100–104 versus fixed, heuristic,
  predictor, budget-matched random and exhaustive baselines. Promote only on the
  existing held-out non-regression gate; otherwise retain deterministic control.
- Workbench: exact model/tokenizer/variant/config/suite/quality matching. Test
  chat formatting, SSE reconnect/duplicates/stale events, cancellation/restart,
  report lineage/visibility, recorded Lens replay and keyboard access. Default
  recommendations; explicit validated application only between execution windows,
  followed by reset/re-prefill.
- CUDA/hybrid: only AFTER all standalone gates pass, use the existing opt-in
  official Qwen3-1.7B downloader, freeze revision/hashes/tokenizer compatibility.
  Revalidate actual WSL CUDA access; Windows hosts the NVIDIA driver. Use supported
  NVIDIA-SMI metadata. Matched prompt/context128/eight tokens, FP16/eager CUDA,
  TF32 disabled, hybrid draft depth4 through full RTL. Synchronized loading/build,
  prefill/decode/inference-total timing. Test actual CUDA cache correction/cropping
  at depths1/2/4/8, rejection/EOS/context boundaries. Hybrid output must exactly
  match GPU-only greedy output; slower hybrid is still valid measured evidence.
- Final release: assemble full schema-v2 separately from standalone schema-v3;
  reject missing/stale/mismatched/failed artifacts. Fresh checkpoint-free
  `make verify-llm PYTHON=.venv/bin/python` on the final clean source, browser
  acceptance, `release-check` and `full-release-check`. Update docs and independent
  PR #1 review plus PR #3's final review against the dense baseline; check current
  tip CI. Leave all PRs unmerged and preserve SSM history.

AWS, physical FPGA deployment, retraining/distillation, threshold reductions,
changed suites and automatic merges remain excluded. No other person's messages
or data should be sent elsewhere. Keep original checkpoints, Mac records,
canceled jobs and all new failures intact.

## 7. Report and hand off without losing a running job

Keep the source checkout unchanged while a source-bound worker runs: even editing
`docs/STATUS.md` would invalidate its clean-source check. If you must hand off
during a running job, write a NEW timestamped `handoff-<UTC>.json` or `.md` inside
its ignored attempt directory with the command, unit/PIDs, original dispatch
deadline, progress and exact monitoring/resume action. Do not modify `job.json`
or historical results. After the worker finishes, update `docs/STATUS.md` with
source/dirty state, command, attempt/unit/PIDs, shared deadline, exit status,
verified metrics, record IDs/file hashes, tests and exact next action. Update the
task row honestly before stopping.
Commit/push reviewable source/docs using existing authentication; never expose
credentials or force-push. If another agent updated the branch, preserve its work
and resolve the ordinary Git integration before pushing; do not merge PRs.

After eight hours stop NEW dispatches, let the active worker finish and save its
status. If your own credits expire earlier, the durable process should continue;
save a handoff immediately with the precise monitoring/resume instructions.
Report full-validation pass/fail separately from held-out/RTL/full-release status.
An unfinished goal or blocked numeric acceptance must remain visibly unfinished.
Do not ask me questions already answered by this prompt; use judgment for routine
implementation choices and report a concrete limitation if the environment or
unchanged contract makes completion impossible.
