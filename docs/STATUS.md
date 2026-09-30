# Status

## Current handoff: Qwen3.5 reference fixed; raw INT8 pilot passes

Codex fixed a confirmed quality-reference defect in ba9a988: constructing the
Qwen3.5 Transformers text tower on the meta device did not restore non-persistent
rotary-frequency buffers from checkpoint weights. Rebuild the rotary embedding
from the unchanged configuration after assigning checkpoint tensors. The position
IDs were correct, but observed sine tables were effectively zero. No ISA/RTL,
matrix/head precision, checkpoint, suite or threshold was changed.
Qwen3.5 reference cache contract is now original-local-safetensors-qwen35-rope-restored-v2;
old Qwen3.5 cache entries must not be reused. Qwen3 and LFM identities are unchanged.
Historical Qwen3.5 78.125% quality and reference-divergence records remain preserved
but used a defective reference and do not describe corrected-reference acceptance.

qwen35-corrected-int8-pilot-01 completed in 329.55 seconds on clean ba9a988:
128 frozen spread-validation targets with complete causal prefixes, raw INT8
matrices/head, balanced emulation: 125/128 = 97.65625% agreement; NLL -0.19908%.
Independent FP32 vs corrected Transformers: 128/128 top tokens, relative L2
1.08e-6 to 1.61e-6. This is diagnostic emulation evidence, NOT actual-ISA or RTL
acceptance. Fourteen focused reference/Qwen3.5 regressions passed before dispatch.
Evidence is under continuation-20260930/qwen35-corrected-int8-pilot-01 (result,
checksummed summary, job/source/PID, runner and fresh reference store). The preceding
qwen35-rope-localization-01 preserves bad-table and matched-input rotation evidence.

No job is running. User was asked to select the next actual-ISA batch mode;
do not revive the revoked Overnight window. Next: short corrected-reference actual
ISA pilot for timing, then unchanged 1024-target full validation of RAW INT8. If
full validation passes, freeze source/config/reference design and perform the
untouched held-out split once via persistent registry, then eight-token full RTL.
Do not launch calibration or precision redesign unless corrected ISA evidence
requires it. Never repeat completed Qwen3/LFM held-out or LFM benchmarks.

Current repair: Codex restored Qwen3.5 non-persistent rotary buffers after meta
initialization and versioned its reference cache. Fourteen focused tests pass.
Historical Qwen3.5 quality scores used an invalid reference and are preserved,
but cannot establish current acceptance. Qwen3/LFM are unaffected. Next is a
128-target corrected-reference/INT8 validation diagnostic in Interactive mode.
Held-out remains untouched. No thresholds, matrices, head or ISA/RTL changed.


## Latest completed Interactive probe — Qwen3.5 attention

User authorized one short attention diagnostic with “go for it”. Codex completed
qwen35-attention-localization-01 in 14.12 seconds on clean source 57edbb3.
16 validation targets from sequence 1 of the existing spread panel were evaluated
with complete causal prefixes. No held-out evaluation or production changes.
Synthetic observation check passed: exact original logits, all capture sites present.

Relative L2 differences before rotary encoding: projection input 9.50e-7,
normalized query 8.88e-7, normalized key 8.73e-7. After rotary encoding:
query 37.75%, key 29.58%; attention probabilities 22.01%.
The first major discrepancy is positional encoding, not projection or normalization.
Loaded rotary parameters agree (theta 1e7, 64 rotary dimensions); the visible
rotation formulas use the same rotate-half convention. Root cause is NOT yet proven.
Next bounded diagnostic: capture actual position IDs and cosine/sine tables and
apply both rotations to identical normalized q/k inputs; check broadcasting,
position layout, mutation/aliasing and table generation before changing arithmetic.
Do not conclude that quantization alone causes the full 78.125% validation failure.

Evidence directory:
build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/qwen35-attention-localization-01/
Contains result.json, checksummed summary.json, job.json, runner.py and the exact
instrumentation.py. Instrumentation is diagnostic-only and not production code.
No worker remains active. Interactive batch complete; choose mode before another
checkpoint job. Overnight authorization remains revoked. Release remains blocked.

## Current mode override — Interactive (2026-09-30)

The user's latest instruction supersedes the eight-hour Overnight authorization.
One bounded Interactive analysis completed; no further checkpoint dispatch is
authorized by the old window. Select a new run mode in chat before the next batch.
The original control/deadline remains historical, with an append-only Interactive
override in continuation-20260930/mode-overrides. Never restart the Overnight loop.

The saved 128-target reference-spread run completed on source 61ac769 in 50.37s.
Independent floating/reference top-token agreement is 119/128 (92.97%); this is
diagnostic evidence, not INT8 acceptance. Interactive analysis verified its hash
and found the first residual error above 0.1% at layer 3's attention output in
all eight sequences. Earlier relative L2 error is about 1e-6; layer 3 is 6.3–9.1%.
This localizes the discrepancy but does not establish its cause.

Evidence: build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/
interactive-localization-audit-20260930T223446Z/result.json.
No new inference or held-out evaluation was performed by this Interactive audit.
Next: instrument the first full-attention layer's q/k normalization, RoPE, scores,
and attention outputs with matched inputs before attributing loss to quantization.
Qwen3.5 full INT8 validation remains 78.125%; release acceptance stays blocked.
No release worker is active. All older run authorizations below are historical.

Handoff for Cursor, Codex, and Claude. Update this before ending a session.

Last updated: 2026-09-30. Any of Cursor, Codex, or Claude may claim any `open` row.

## Authoritative continuation — 2026-09-30

Read [the complete approved plan](release-continuation-20260930.md) first.
The current verified base is `7693ed9`; Qwen3 calibrated INT8 and LFM raw INT8
pass validation, held-out and eight-token RTL. Tiny 100-token RTL and all ten
LFM benchmark indices pass. Qwen3.5 validation fails at 78.125%; held-out untouched.
Do not repeat completed Qwen3/LFM balanced held-out or LFM benchmark jobs.

Codex owns the current safeguard implementation. No release unit/PID is active.
The authorized eight-hour Overnight window started 2026-09-30T19:14:51Z. Fixed dispatch deadline: 2026-10-01T03:14:51Z (September 30, 8:14:51 PM Pacific).
The five safeguards passed before that first checkpoint dispatch. Reuse the fixed deadline for all subsequent jobs;
the persisted control is at
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/control.json`.
Implemented safeguards (committed and pushed as 0a0004f): derived benchmark lineage, hybrid draft binding, persistent held-out
consumption, complete freeze validation, and benchmark failure/deadline status.
Five safeguard fixes passed 98 focused tests; another 31 diagnostic/deadline tests passed. Existing Qwen3/LFM held-out were verified and registered as consumed, without inference. New freeze v2 supports raw/derived; existing v1 remains checked. Ten Qwen3.5 emulator controls preserve baseline; site map is docs/qwen35-diagnostic-site-map.md. Attempt qwen35-alignment-01 was interrupted by WSL shutdown at 19:15:00Z, with no results; all files and failure-interruption.json preserved. WSL keeper Windows PID 34304/Linux PID 276 now holds a foreground connection; power helper Windows PID 35824 verified active after a monitoring interval. Retry support committed/pushed as 62f35ae. qwen35-alignment-02 COMPLETED in 109.887s on that clean source: ISA/emulation/independent/Transformers top-token agreement 16/16; 320 converted tensors match; result SHA256 4b3a113a4e5bfaa5cc216454eb9e7f023d11fadfa20a97f83672e5071efeefda. Floating logits still differ 2.082% relative L2, max0.51947, despite identical top tokens. Reference-rounding-01 completed on cc9198b in 11.18s; all 320 original-file tensors match conversion. Chunked/sequential HF differ 3.467% relative L2 and 1/16 top tokens; independent/sequential differ 4.457%. This shows floating execution-order sensitivity but does not establish all larger-panel discrepancies. Next: reference-spread-01, 128 frozen spread targets with layer-residual localization, before quantization attribution. No new held-out. Never reset the deadline. INT8 matrices/head retained.
Older entries below are historical and cannot override this authorization/status.

## Historical progress

Latest continuation handoff (2026-09-29): user is moving execution to Cursor
because Codex credits may run out. Use
[the complete local Cursor prompt](cursor-quality-continuation.md), which selects
Overnight mode on receipt, supplies exact source/candidate/commands, a durable
full-validation launcher, eight-hour dispatch deadline, conditional ten-case
schedule and the remaining quality/release gates. No long validation job was
started while preparing this handoff. Local Cursor must use the Zephyrus WSL
checkout; a cloud-only Cursor session cannot access the ignored artifacts.
`tools/run_candidate_full_validation.py` is ready for the first full validation
attempt, protects the shared lock, refuses overwrite and records source/PIDs/
logs/results/failures. Run it via the documented user systemd command.

README refresh completed by Codex on 2026-09-29: the entry point now describes
the approved local LLM release, its five completion milestones, actual
acceptance/quality evidence, compiled simulation personalities, Zephyrus
handoff, and gated CUDA/hybrid scope. No new execution or release evidence was
produced by this documentation-only change. Integrated newer remote quality
recovery work without overwriting its handoff; README links that approved scope.

### Current Zephyrus quality recovery (2026-09-28)

User approved [staged Qwen3 quality recovery](qwen3-quality-recovery.md), including
targeted LLM precision/ISA/RTL redesign after compatible INT8 experiments and
acceptance of a measured slowdown. The dense numeric contract remains unchanged.
This supersedes the earlier numeric-redesign exclusion for this specific scope.
Baseline `b1f2d3f` passed final-source `make verify-llm`, actual CUDA cache tests,
browser acceptance and CI. Evidence is under
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/`.
Qwen3 remains blocked (16-target agreement 81.25%; NLL degradation 1.01%).
Latest outcome: all ten authorized cases completed in 225.07 seconds on clean
source `7118e064e08dae35b58746ef5048f0657284020c` (pushed). Baseline 81.25%;
all-floating 100%; best targeted bypasses 87.50%. The combined key-cache and
transformer-weight bypass fixed the three original mismatches but introduced two
others. No targeted case met 90%; release remains blocked. See
[the complete results and interpretation](qwen3-precision-attribution-results.md).
All ten result hashes verified; focused CPU tests: 31 passed in 10.44 seconds.
The user then authorized the recommended single Interactive 128-target baseline
pilot. It completed on clean source `c352607e3c9394231491fa1c4a30e9d098f3b04f`:
107/128 matches (83.59375%), +0.622021% NLL, 281.44 seconds wall time,
256.52 seconds candidate time, 996 executed context tokens across eight sequences.
Both floating references agree on all 128 targets. Evidence:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-128-pilot-01/`.
Result SHA-256: `994db61a623d52ce3a3ddf6cf792c38ead207ebf2bc93764bdfe24e2c557d454`.
Runner, frozen panel/policy, events, canonical record, summary and source identity
are retained there; Windows evidence copy is in `setup-evidence/precision-128-pilot-01`.
No batch worker remains active. PIDs 2021 and 3215 are historical.
Next: choose the next batch mode for the remaining nine 128-target cases
(all-floating, seven single-group bypasses, fixed key-cache + transformer-weight
bypass), estimated 35–45 minutes serialized using this completed pilot. Reuse the
frozen panel and compatible baseline/reference evidence; preserve each attempt.
The remaining nine cases COMPLETED successfully on clean source
`6975b1c5e5c2c5ae5b832224297ff44412753ab6` in 2167.79 seconds (36m08s).
Attempt: `precision-128-remaining-01/` in the same release root. All nine result
hashes verified; report SHA-256:
`31134ec2dbd76c6dee9f747e871b551aa96a0676d3ada0a8ba44a5a8ec21a0c7`.
No batch worker is active; PID 3837 is historical. Do not rerun this attempt.
Key-cache bypass alone reached 117/128 = 91.40625%, NLL change -0.363486%.
Combined key-cache + transformer-weight bypass also reached 117/128, NLL
-0.149227%. All-floating control reached 128/128. These are independent CPU
emulation diagnostics, not ISA/RTL or held-out approval. See the results document
for all cases. The previously running handoff was stale until this inspection.

Cursor's separate draft PR #4 (`415b762dc1f575b74867396103287efc059d3929`)
implements compatible INT8 rescaling/clipping research code. It has not been
merged or run on the real checkpoint here. Initial review: separating its branch
and calibration/validation is appropriate; checkpoints/evidence access, not a
GPU, is required for these CPU diagnostics. Fix eligibility/ranking to enforce
both quality gates and add actual ISA evaluation of derived tensors before any
promotion. Review memory budgeting for duplicate checkpoint/derived allocations
before a real-model pilot. Cursor must refresh this handoff instead of resuming
missing cases that no longer exist. No new checkpoint batch is dispatched.
Latest completed authorization (2026-09-29): user said "go for it" to review/fix
PR #4 and run the short INT8 pilot, followed on failure by TEN different precision
cases. The pilot COMPLETED on clean source `cb193a318c3bcb0b5a559fc057610bd65c786842`
in 620.25 seconds. `a0.5-cnone` (rescaling only) reached 121/128 = 94.53125%,
NLL change -0.177878%; `a0.5-c99.9` reached 5/128 = 3.90625%, NLL +135.676582%
and was rejected. Because one candidate met both gates, the conditional fallback
was NOT dispatched. All ten fallback policies remain frozen in `plan.json`.
Actual calibration used all 256 available tokens (512 requested maximum).
Attempt: `quality-recovery-pilot-01/` in the release root. Report SHA-256:
`3d82dd224361d50d42131b3dae1cc93e611a93be9830f171f6d2f674951b393a`.
All 308 persisted tensor hashes verified per candidate; see `artifact-integrity.json`.
Best derived ID: `3129232e64ec1532cfb735b48dac385bbeac4a072af98660d0fe85953cd163da`.
No job from this batch remains active; PID 22182 is historical. No held-out used.

Actual derived-weight ISA integration is now implemented and tested: new
`quality-candidate-diagnose` (bounded validation) and `quality-candidate-validate`
(full frozen validation, >=1024 targets). Both bind checked artifact hashes and
original model/tokenizer/weight lineage; derived tensors never become the FP32
reference. Held-out is rejected. The quality variant binds derived/parameter IDs;
production generation/release integration and candidate freeze remain unfinished.
88 focused CPU tests passed in 21.15s, including actual tiny ISA loading parity.
A synthetic outlier fixture showed one top-token difference between float64
emulation and actual ISA (13,13,35 versus 13,13,34). Do not assume emulation is
an exact ISA oracle; reason was not localized by that test. Production arithmetic
and quality thresholds remain unchanged.

The authorized Interactive actual-ISA pilot COMPLETED in 86.54 seconds on clean
source `c4e87cd3532833899b86fbf3d3ceabe2ea6c6988`: 15/16 = 93.75% agreement,
FP32 NLL 6.24092541, ISA NLL 6.28054038, degradation +0.634761%.
Both diagnostic gates passed. All 311 original source tensors match the original
FP32 checkpoint; floating references agree 16/16; ISA/emulation agree 15/16.
Candidate/parameter/calibration identities match the passing rescaling-only case.
Attempt: `candidate-isa-pilot-01/` in the same release root; saved runner,
job/source/command, worker log, canonical record, result and summary.
Result SHA-256: `03adbf677fe1b67ce0b274f384a8fcded41dc7849e87baaf84f0c4774b6ba7d3`.
Canonical record: `f512c4d0365ceccf9089a1a34945458cafc852419e81f676cb3bfc3dcbdb61ed`.
No pilot worker remains active; PID 24214 is historical. This is a 16-target
validation diagnosis, not full validation/held-out or full-RTL acceptance.
Next: actual ISA validation of this same saved candidate on all 1,024 frozen
validation targets. A linear extrapolation of 86.54s/16 is 92 minutes; allow
roughly 90–150 minutes because the longer causal prefixes increase attention
work and startup/reference overhead differs. Select Overnight for the complete
validation job, or Interactive for a bounded 128-target timing check first.
Prepared commands are in docs/qwen3-int8-candidates.md; select `int8/case-00` in
the completed pilot. Memory preflight for the diagnostic is 13.04 GiB, within
the configured 16 GiB budget. No full-validation or held-out job is dispatched.
Reviewed source from PR #4 is incorporated as research code with screening gates,
derived-only memory handling, model/tokenizer checks, and focused regressions.
Diagnostic site-policy v2 and ADR-0006 predeclare ten fallback cases; production
ISA/RTL arithmetic stays unchanged. Run command after a clean commit:
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python -u tools/run_quality_recovery_pilot.py --root build/zephyrus-jobs/release/20260928T223413Z-b463e746/quality-recovery-pilot-01 --authorization "User approved short INT8 pilot then ten different precision cases on failure"`.
Inspect that attempt's `plan.json`, `int8-worker.log`, `int8/report.json`,
`precision/case-*/summary.json`, and `report.json`/`failure.json` before resuming.
Two INT8 cases take roughly 12–15 minutes including calibration. Conditional
fallback is roughly 40–45 minutes based on the completed 128-target baseline.

Batch record: the shared handoff was committed/pushed as `9617f82`. Bounded
precision attribution is implemented with exact baseline-preservation, seven
independent group controls, causal-prefix panels and CPU regressions. The user
selected ten DIFFERENT serialized 16-target cases, then report: baseline,
all-floating, seven single-group bypasses and an adaptive top-two combined bypass.
This is the complete current batch authorization; it does not authorize later
128-target or held-out jobs. Completed attempt folder:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-attribution-01/`.
Historical run command (completed; do not rerun into this folder):
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python -u tools/run_precision_attribution.py --root build/zephyrus-jobs/release/20260928T223413Z-b463e746/precision-attribution-01`.
Before resuming inspect `batch.json` (PID/source), `case-*/summary.json`, worker
logs and `report.json`/`failure.json`; never redispatch an active batch or overwrite
an attempt. The runner freezes both 16-target and 128-target panels but executes
only the authorized 16-target panel. Read the recovery plan before the older status
paragraphs below, which preserve the historical Mac milestones.

The [Zephyrus handoff guide](zephyrus-handoff.md) covers WSL2 installation,
transferring ignored models and complete stores, rebuilding Linux tools,
CUDA preflight, verification, and the continuation prompt for the next agent.
The local transfer folder is `build/Zephyrus-Transfer`; its `START-HERE.md`
describes offline source restoration, data snapshots, checksum verification,
and a fresh Zephyrus job directory. Preserve historical queues and absolute
Mac paths as provenance rather than dispatching them after migration.

The active release is the local pretrained-LLM platform (ADR-0003). The dense
numeric contract and RTL are preserved. The approved completion plan includes
standalone RTL acceptance, workbench/learning evaluation, then gated Zephyrus
CUDA and greedy-hybrid evidence. AWS and physical FPGA deployment are excluded.
Custom SSM product commands/RTL are retired in the working branch; historical
SSM PR #2 and user data remain intact. OpenTPU revision
`15754e971b55591b91048c4c636023fe59b343e7` is vendored as a distinct execution
backend. Full RTL is the default, with an explicitly selected ISA alternative.

Implementation is committed on `codex/local-llm-platform` (`0a4455e` plus the
release-evidence path fix `0242e06`) and pushed. Draft PR #3 targets
`codex/model-aware-baseline`; no merge to `main` has occurred. Local checkpoint
downloads were explicitly authorized and completed for Qwen3-0.6B,
Qwen3.5-0.8B and LFM2.5-230M. Pinned official
revisions/file hashes are recorded; weights stay ignored under `build/models`.
The React/TypeScript workbench is served at loopback `127.0.0.1:8765` with durable
jobs, SSE progress, recorded Lens replay, report views, fixed-tape benchmarks,
and recommendation-first optimization with explicit opt-in automatic application.
CUDA/hybrid commands are wired but require passing standalone evidence. No AWS
integration/resource creation, physical fit/timing/power, or Zephyrus CUDA
performance is established on this Mac.

The latest checkpoint-free `make verify-llm` passed: dense
simulation/lint/Yosys, 108 upstream core tests, 28 model-family tests, 13 LLM
RTL tests (12 personality/format combinations plus 100-token tiny-model
sequence), 29 host tests, four UI event tests, and the frontend production
build. Its immutable report is at ignored
`build/release-evidence/verification-v3/verification.json` (SHA-256
`4d9609253a07a4aa735e92c252523336459abca48c5dfa867c582475c75a7409`, commit
`0242e06518e55db2334d8d95ec801e2a26c9d967`). Durable tiny evidence is at ignored
`build/release-evidence/tiny/tiny-release.json` (100/100 RTL steps bit-exact,
SHA-256 `cdc7194c3c1cac9b15c7433db2e93662bef468e9195ef38484894d6f79df407d`).
Two earlier failed evidence attempts are retained in separate ignored
directories: one exposed path quoting, the next selected system Python by
resolving the venv symlink. Neither ran the suite; verification-v3 passed.
All three original real-model acceptance jobs were canceled in the workbench;
partial bit-exact steps do not count as release acceptance. The user's explicit
instruction is to leave the original Qwen3 job canceled, not requeue it.
One real Qwen3-0.6B INT8/balanced chat completed through the explicitly selected
ISA backend (22 prompt tokens, 16 generated tokens); it is not full-RTL or
quality approval. The CLI research store now has the earlier generation plus two
Qwen3 quality records; the full-RTL acceptance run is in its separate store.
No decision, policy, or policy-evaluation records have been generated.
At the time of the earlier workbench check, no jobs were running or queued.
Earlier prompt jobs failed on chat-template token mapping; its code fix has
host-test coverage but still needs end-to-end real-model retesting.
Fresh Qwen3-0.6B INT8/balanced held-out and validation suites both completed.
Held-out NLL degradation is 0.95% (passes 5%), but next-token agreement is
83.50%; validation agreement is 84.67%. Both fail the 90% threshold and remain
not selectable. A separate fresh Qwen3 full-RTL run completed all 7 raw prompt
tokens and 8 generated tokens in 14 steps, with exact DRAM/TMEM agreement on
every step. It recorded 86,022,637 RTL cycles and 0.00364 host-simulated tokens/s
(not FPGA performance). The run is valid execution evidence but does not pass
quality approval. This completed run predates the bounded bottleneck-summary fix;
its raw counter payload labels the numeric DRAM bound incorrectly, while future
runs use the corrected categorical diagnosis. Neither fresh run resumes a
canceled workbench job. Qwen3.5/LFM quality and full-RTL acceptance, 30 staged
performance runs, predictor/RL held-out evaluations, strict standalone release
manifest, PR review/CI completion, and Zephyrus CUDA/hybrid evidence remain
unfinished.

### Full ISA validation of candidate a0.5-cnone (2026-09-29, Cursor local)

Overnight mode was selected by `cursor-quality-continuation.md`. First dispatch
2026-09-29T17:03:40Z; shared dispatch deadline (job.json `dispatch_deadline_unix`
1790730220) = 2026-09-30T01:03:40Z. Unit `malleable-qwen3-full-validation-01`
(launcher PID 27447, worker PID 27458, exit 0) on clean source
`26279df0fd4abcb0bfbcf136ced3f9cb1724a880`, attempt
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/candidate-isa-validation-01/`.
Result: **PASS**, 963/1024 = 94.04297% agreement, original FP32 NLL 4.70241488,
ISA candidate NLL 4.70262040, degradation +0.004370%, 2733.4s. Gates independently
recomputed (agreement >= 0.90; NLL <= 1.05x). result.json SHA-256
`c8c29777249ac9ac6ba7d47b304e03cb4ce23dfae75ec630a3c2fd892f82806d`; record
`e24ba1b7ad7ddd03bd47dad256deeb746336b56ec913b15c0ecc10cb4224d7b0`; variant
`16e8f30ea0f88024a7dc3e0ece6f61b4569acd605cd755b3cb0c444a44f21b19`; configuration
`224a2edbf7462eae26e3527b8f0e554e5b13b06987bd557a9a4f983370d559dd`; derived ID
`3129232e...cd163da`. Validation evidence only: held-out, full-RTL and release NOT
done. The four ten-case precision fallbacks were correctly NOT run. Local commit
`26279df` (task claim) could not be pushed: this machine has no GitHub credentials.

### Qwen3 candidate a0.5-cnone: integration, freeze and held-out (2026-09-29, Cursor local)

Source commit `0df744eb8f7dd906c434b6f5340dcd39ee000e16` (LOCAL; this machine has
no GitHub credentials, so `26279df`, `9b8215d`, `0df744e` and this update are not
pushed). ADR-0007 and `tests/test_candidate_integration.py` (12 tests, in
`make verify-llm`) bind derived candidates to generation (`candidate_case`),
benchmarks/performance manifests, hybrid draft, workbench jobs (job-root confined),
quality variant identity, the release check (`candidate_freeze` artifact) and a
single frozen held-out evaluation. Focused suites: 91 passed/12 skipped plus 52 passed.
Freeze: `candidate-heldout-freeze-01/freeze.json`, freeze_id
`e4def86bac90fe30ae318776115ee7c48b7c4e3f1098218915970c969a513b84`
(sha256 `664dbd8006271b21058fd9e13330e42bce47c4abba6743e531ceab51af53b957`).
Held-out (unit `malleable-qwen3-heldout-01`, exit 0, 2618.6s, ONE evaluation,
`heldout-claim.json` consumed): **PASS** 964/1024 = 94.14063%, original FP32 NLL
4.43603006, candidate NLL 4.42686625 (-0.2066%), selectable=true. result.json
SHA-256 `b513e1adb0e25e0b22f23eefadbf8b36441aebcee13e000061cd559e6a2197d9`,
record `a51c08844625e716a071d5663b91e5d58276d1c48281fa6891587cb105734a3b`.
Standalone Qwen3 QUALITY gates are now met by the calibrated candidate.
NOT done: Qwen3 eight-token full-RTL run of the candidate, fresh 100-token tiny RTL,
Qwen3.5-0.8B and LFM2.5-230M candidates/quality/RTL, 30 performance runs,
controller/predictor, browser/final verification, CUDA/hybrid. Release remains open.
Do not rerun the held-out evaluation or tune against it.

### Qwen3 candidate full-RTL acceptance (2026-09-29, Cursor local)

Unit `malleable-qwen3-candidate-rtl-02` (exit 0) on source `91a19cee224c9b689bed96726639bfb131ef8def`
(clean): candidate a0.5-cnone, raw prompt "The capital of France is" (5 tokens) + 8 greedy tokens,
context 128, balanced, INT8/INT8 head, full RTL. **PASS**: 8 tokens (12095 13 576 6722 315 9625 374 1083 =
" Paris. The capital of France is also"), 12 executed steps all `bit-exact-dram-and-tmem` with trace
hashes, 73,729,701 simulated cycles, host 0.0034 tokens/s (simulation, not FPGA). variant
`16e8f30e...4f21b19` and configuration `224a2edb...59d` equal the held-out quality record; generation
record `1f47b6df60460ceaaea451828ff7c03aef6d55afdd64f7aac5d17e6a0190b13b`; result.json sha256
`ab0cfc2f77e56d2f0bcb6d7b2634ed00a46e36f91f5463915646e397b19e947a` in
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/candidate-rtl-acceptance-02/`.
Attempt 01 failed only because the systemd unit PATH lacked `verilator` (preserved, noted there);
RTL units need `PATH=/home/dhruv/.local/bin:/usr/local/bin:/usr/bin:/bin`.
Qwen3 now has validation + held-out + eight-token RTL evidence for the candidate.
Still unfinished: fresh 100-token tiny RTL, Qwen3.5/LFM2.5, 30 performance runs, controller, workbench,
CUDA/hybrid, final release checks. Deadline 2026-09-30T01:03:40Z. Commits from `26279df` on are NOT
pushed (no GitHub credentials); continue on the same checkout unless the user pushes. Rolling handoff:
`build/zephyrus-jobs/release/20260928T223413Z-b463e746/handoff-latest.md`.

### Tiny 100-token RTL and Qwen3.5-0.8B raw INT8 validation (2026-09-29, Cursor local)

Fresh durable tiny evidence (unit `malleable-tiny-rtl-100-01`, exit 0): 100/100 steps bit-exact,
406,800 cycles, build_id `eec9b2c397fc7005...`; `release/tiny-rtl-100-01/tiny-release.json` sha256
`bd739609c9acacf0e5d7cae5ead45ab6f628a4271b05ccc2d8537fb44fc52fa7`.
Qwen3.5-0.8B raw INT8 full-ISA validation (unit `malleable-qwen35-validation-01`, exit 0, ~70 min, clean
source `91a19ce`+status commits, `release/qwen35-raw-int8-validation-01/`): **FAIL** agreement 800/1024 =
78.125% (gate 90%), original FP32 NLL 4.34058013, ISA NLL 4.03603121 (-7.02%, NLL gate met but agreement
gate not; NLL below FP32 is unusual and unexplained - do not read as quality), variant
`dcc3d566...169a7b`, record `edbeeaa1...f401`, result.json sha256
`2e7783a59c06cf17aced9bd82b8f6f7da9798b6c247f0510bb7e1a6bb553d424`. Preserved; no held-out run for
raw Qwen3.5. INT8 candidate search (`candidates.py`) and precision attribution are Qwen3-only (they
raise for other families); Qwen3.5 (gated DeltaNet/conv layers) needs new site definitions and
emulation, hence an ADR first. Not started.

### LFM2.5-230M raw INT8 validation (2026-09-29, Cursor local)

Unit `malleable-lfm2-validation-01` (exit 0, ~17 min, `release/lfm2-raw-int8-validation-01/`):
**PASS** 960/1024 = 93.75%, original FP32 NLL 6.78683040, ISA NLL 6.79066570 (+0.0565%), variant
`17c66490b0ddfeb6a9889fed14e18368d8274042abd5fdcfa05ad60dd82d7e63`, configuration
`78a135a7b46ac35b70089341e13593ecf58a67606c959f6f2a14ea1faf4001d3`, record
`fa4dd3b1cf27b2077c237201461bf6f790bcfada9cc15f2dbabeb448f2704e5f`, result.json sha256
`72654c029728190ac27accc55b234d4aab4b01345714765c99ac1c15d1a00483`. Validation only (search evidence,
selectable=false). Raw baseline, no candidate. Commits are now PUSHED (branch head matches origin at
`b0df2fc`; push worked via the Windows Git Credential Manager helper for that one command).
Next: single raw held-out run (`release/lfm2-raw-int8-heldout-01/`), then 8-token full RTL.

LFM2.5-230M raw INT8 single held-out (unit `malleable-lfm2-heldout-01`, exit 0, ~17 min,
`release/lfm2-raw-int8-heldout-01/`): **PASS** 977/1024 = 95.41016%, FP32 NLL 6.50602366, ISA NLL 6.52798781
(+0.3376%), selectable=true, same variant/configuration as validation, record
`711f2f594a4d68ae070502610d0b2670b218f2f8e4091b60d8b3770ed6c6bc80`, result.json sha256
`95f6aeedcf2b324207acb596708c17e15f5df43d72da14b28a1800da4ea21371`. One evaluation; do not rerun.
Next: LFM2.5 eight-token full RTL (`release/lfm2-rtl-acceptance-01/`).

LFM2.5-230M eight-token full RTL: attempt 01 (prompt "The capital of France is") was valid and bit-exact but EOS'd
after 3 tokens (` Paris.`), so it is not acceptance (result sha256 `440cf983...c5eb`, preserved). Attempt 02
(unit `malleable-lfm2-rtl-02`, exit 0, prompt "Once upon a time there was a small", chosen only to avoid early
EOS): **PASS** 8 tokens (7314 14979 1859 1685 27358 23820 810 21281 = " village nestled between rolling hills
and spark"), 9 prompt tokens, 16 steps all `bit-exact-dram-and-tmem`, 37,603,834 cycles, record
`d6ba925e57f6d93ccad17e3aaadbafe6b51242f79f92c6d5010c60cd3981d917`, variant/configuration equal the held-out
record, result.json sha256 `acaba9212635761ebe87ea8f50f44fa2ce8079c388b4e349e4de798073ea6dcb`.
Comparison evidence for regenerated runs (per-step trace hashes/cycles) is tracked in `docs/evidence/`
(`tiny-rtl-100`, `qwen3-candidate-rtl8`, `lfm2-rtl8` `-trace-hashes.json`). Standalone status: Qwen3 (candidate)
and LFM2.5 have validation, held-out and eight-token RTL; Qwen3.5 fails raw validation (78.125%), so the
standalone `release-check` cannot pass until Qwen3.5 has a passing candidate. Next: performance suite.

### LFM2.5-230M indexed performance suite (2026-09-29, Cursor local)

Unit `malleable-perf-lfm2-01` (driver `tools/run_performance_indices.py`, exit 0 after index 9):
**10/10 verified** `llm-benchmark-attempt` records, all `validate_benchmark_result` checks passed, zero failures.
Manifest `release/perf/lfm2-performance-manifest.json` id (sha256)
`3b76747d715b0d10038a1502bb3938dffccbdf23eb5e64912d8bce60f96ba1ba` (seed 42, 8 INT8 personality/workload
comparisons + 2 INT8 AXI stress; INT4/FP4 not validated). Aggregated `llm-benchmark-suite` record id
`8486833b35eeba0ef1ccef4098df5e07360b6ccdb594dcd7eedceae59eae8f52`, store
`release/perf/lfm2-01/store/research/`, logs `release/perf/lfm2-01/driver.log` and `indices.log`.
Per-index driver wall seconds: 404, 404, 949, 434, 557, 661, 1446, 671, 620, 1346 (7,492 seconds, approximately 2.08 h serialized).
Per-index summed RTL cycles: 14214874, 14091076, 14058280, 14074618, 23696663, 23491003, 23430444,
23460308, 26196184, 26177717 (total 202,891,167). Read-only parallelization notes:
`release/perf/lfm2-01/findings-20260929T230500Z.md`. Qwen3/Qwen3.5 performance manifests not started
(Qwen3 needs `candidate_case` in manifest creator; Qwen3.5 needs passing quality/candidate).

## Done

- Signed INT8 MAC with signed INT32 accumulation and explicit overflow
- Parameterized INT8 dot product, multi-tile accumulation, bias,
  requantization, INT8 saturation, and optional ReLU
- Descriptor-driven top for up to four dense layers, with ping-pong activation
  buffers
- Self-checking testbenches and `make verify` in GitHub Actions
- Numeric contract, architecture notes, research roadmap, and ML contributor
  roadmap

## Open

| Status | Owner | Item | Notes |
| --- | --- | --- | --- |
| done | Cursor (local, Zephyrus) | Qwen3 quality recovery and precision attribution | Qwen3 candidate + LFM2.5 standalone gates (validation/held-out/RTL8 + 10-run perf) done; tiny RTL + trace-hash evidence in docs/evidence/. Qwen3.5 raw validation FAIL 78.125%; no candidate path yet. Remaining: Qwen3/Qwen3.5 perf (20 runs), controller/predictor episodes, workbench/browser, final verify-llm/release-check. No new heavy jobs after 2026-09-30T01:03:40Z unless user extends. |
| in progress | Codex | Zephyrus release completion implementation | Authoritative 2026-09-30 plan: fix five safeguards before new Qwen3.5 diagnostics; new eight-hour window not started. |
| done | Codex | Bit-accurate dense golden model and tiny-network cross-check | [PR #1](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/1), `malleable/model.py`. Framework calibration remains separate. |
| done | Codex | Versioned dense model descriptor and exported test vectors | [PR #1](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/1), `malleable/records.py` and generated RTL benches. |
| open | | Extend Cyclone V Quartus projects to `malleable_accelerator_top` and record real resource and timing results | Needs Quartus Prime Lite. Current projects are `quartus/int8_mac` and `quartus/int8_dot_product`. |
| open | | Host-to-Cyclone-V transport and benchmark harness | ADR-0001 item 6. Depends on exported artifacts. |
| open | | Review the first software/RTL cross-check against `docs/numeric-contract.md` | Depends on the golden model and vector export. Do this before either side changes the contract. |
| done | Codex | SSM simulation/operator foundation, local IDE and automatic Quartus jobs | [Draft PR #2](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/2), stacked on #1. Q14 prototype only; see ADR-0002 and `docs/ssm-platform.md`. |
| retired | | Production INT4/INT8 SSM calibration, elastic modes and hardware-aware retraining | User retired the custom SSM direction on 2026-09-27. Preserve prototype history; do not resume this task. |
| retired | | SSM memory/timing optimization, external-memory model and Windows Quartus worker | Superseded by the requested pretrained-LLM/F2 direction. Existing prototype is not a deployable board design. |
| in progress | Codex | Implement the local pretrained-LLM pivot | Approved local plan: retire SSM product, pinned OpenTPU backend, full RTL default, live UI, quantization/learning, then gated greedy hybrid. AWS integration excluded. |
| in progress | Codex | Retirement, pinned integration and safe model import | Implemented and committed; three official checkpoints downloaded and hash-verified. Draft [PR #3](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/3) is open; CI is in progress after fixing the missing Verilator lexer-header dependency. |
| in progress | Codex | Standalone full-RTL acceptance | Qwen3 calibrated and LFM raw INT8 pass validation/held-out/RTL8; tiny 100-token passes. Qwen3.5 fails validation; untouched held-out. |
| in progress | Codex | Live UI and profiling | Durable jobs/SSE, cancellation, history, report views, bounded trace/instruction inspection, and CUDA/hybrid controls are implemented. Full browser-level accessibility/reconnect/cancel test coverage remains. |
| in progress | Codex | Overhead-aware optimization and learning | CLI/controller, predictor and masked Double DQN primitives implemented. Review fixes preserve parent training lineage. Real measured-data evaluations and full UI/service integration remain. |
| in progress | Codex | CUDA baseline and greedy hybrid deployment | Strict standalone-gated CLI/API, WSL2 instructions, opt-in Qwen3-1.7B download and full-RTL draft path are implemented. Mac has no CUDA and verifier is not downloaded; actual Zephyrus evidence remains. |
| proposed | | LLM fast runtime adaptation within a personality | Genuine transcript addition: expose only existing or newly verified compiler/RTL controls; current compiled personalities are not resident runtime modes. Requires explicit scope acceptance. |
| proposed | | Phase/operator-aware experience retrieval | Extend existing persistent storage with reusable shape/context/prefill/decode/scenario signatures and confidence/staleness checks. Do not reuse incomparable evidence. |
| proposed | | Contextual-bandit comparison and complete adaptation overhead | Compare bandit/cost model/heuristic/RL; measure controller, exploration, reload and switching costs. Current RL implementation is not evidence of a performance win. |
| proposed | | Physical static-shell/reconfigurable-region architecture | Later platform-specific extension, not part of the current local release. Validate support, stable interfaces, draining/decoupling, state and real reconfiguration costs; AWS stays last. |

## Explicit release exclusions

AWS provisioning/uploads/HBM/AFIs, board programming, physical reconfiguration,
new model architectures, additional runtime modes, contextual bandits, phase or
operator retrieval, retraining/distillation, and universal compatibility are
excluded. CUDA and greedy hybrid are in scope but hard-gated on standalone
acceptance.

## Session log

- 2026-09-27 (Codex): Audited current build status at the user's request, without
  implementing transcript suggestions or requeueing canceled jobs. Re-ran 24
  host tests, three UI event tests and Python compilation successfully. Confirmed
  one completed real Qwen3 ISA generation, no central quality/policy evidence,
  no queued/running jobs, and uncommitted implementation on the review branch.
  Listed genuine transcript additions as proposed rather than accepted tasks.
  Immediate priority remains standalone real-model correctness/quality gates,
  followed by complete UI/learning integration and review publication; CUDA,
  physical reconfiguration and AWS are later stages.
- 2026-09-27 (Codex): Read the entire user-supplied local conversation export
  and compared its three-timescale adaptive design with current code. The
  model/workload analyzer, persistent experiment database, cost-aware selection
  and offline policy framework align with that vision. Critical distinction:
  the four OpenTPU personalities are separate compiled simulator designs, not
  four register-selectable modes resident in one LLM design. Dense runtime lane
  control does not establish the latter. LLM runtime mapping controls,
  phase/operator-specific experience retrieval, a contextual-bandit comparator,
  automatic validated deployment and physical partial reconfiguration remain
  extensions, not implemented capabilities. Controller learning is separate
  from retraining Qwen. Transcript suggestions did not authorize new scope;
  AWS remains excluded and the original canceled acceptance jobs stay canceled.
- 2026-09-27 (Codex): Implemented the local pivot in a review branch without
  merging to main. Recorded ADR-0003 before integration; kept dense arithmetic
  unchanged. Added isolated OpenTPU harness patches for streaming, bounded build
  parallelism, space-safe caching and timeouts. Downloaded only the three
  authorized official Safetensors checkpoints, retaining licenses/manifests.
  Passed the local verification suite described above. Review found and fixed
  Qwen3.5 BatchEncoding handling, text-only floating reference loading,
  auxiliary tokenizer path validation, worker cleanup, searchable benchmark
  records and continued-policy training lineage. Latest queue-cost/predictor
  edits need another check. User is testing the workbench; do not restart it
  while their jobs run or silently requeue canceled acceptance jobs. All three
  original real-model acceptance runs are canceled, and quality gates have not
  passed. Implementation was interrupted by a request to compare another shared
  chat; that link could not be read, so its contents must be supplied before
  claiming an alignment comparison.
- 2026-09-27: Added shared agent instructions. No implementation change.
- 2026-09-27 (Codex): Preserved the dense baseline, implemented dense host learning
  and RTL instrumentation, then added a separate SSM simulation foundation.
  New remote guidance was pulled without overwriting local work. Existing and
  new tests pass, including 22 host/integration tests and all 31 lane settings.
  Full lint/structural sweep passes locally. Dense PR #1 CI is green; SSM draft
  PR #2 CI is pending. Neither PR is merged; `main` is unchanged. Private media,
  generated artifacts and unrelated `tmp/` content were not published.
- 2026-09-27 (Codex): User supplied a speculative-decoding UI video as an idea.
  It presents an FPGA-simulated draft plus GPU verifier, not standalone SSM
  inference. Asked whether it is UI inspiration or authorization for hybrid work.
  No hybrid backend was added. Recurrent state rollback, tokenizer alignment and
  GPU-only end-to-end comparisons are required before that extension.
- 2026-09-27 (Codex): Reviewed the supplied cross-agent conversation and discussed
  OpenTPU reuse and an AWS EC2 F2 target; neither integration nor cloud resources
  were implemented/provisioned. Source inspection confirms OpenTPU is programmable
  and parameterized, not intrinsically rigid. Reusing it would preserve the
  malleability goal only with validated personality variants, compatible compiler
  mappings, instrumentation and overhead-aware selection. Current runtime lane
  controls do not reallocate physical FPGA resources; physical personality swaps
  and partial reconfiguration remain unimplemented. UI adoption is a requested
  direction, not a completed change. Both SSM PR #2 verification checks passed
  after the earlier publication log recorded them as pending.
- 2026-09-27 (Codex): User explicitly requested scrapping the custom SSM direction.
  Recorded retirement of its open development tasks and the new pretrained-LLM/F2
  direction; no source, artifacts, PRs or history were deleted. Proposed first
  milestone is an OpenTPU-backed simulator/correctness baseline plus extensible
  model compatibility reports and Lens UI integration. F2 needs Vivado/HDK/AFIs,
  new host-transfer and HBM integration, and measured image/state reload costs;
  it is not a Quartus retarget. Current AWS Small Shell has no built-in DMA
  engine, so host-transfer integration must explicitly supply one or use SDE.
  No AWS resources were provisioned or model data uploaded. Cloud execution
  needs confirmed region/quota, budget, data permissions and stop controls.
- 2026-09-28 (Codex): Tightened selectable quality evidence to require a frozen
  held-out split of at least 1,024 targets and exact model/tokenizer/variant/
  personality/configuration lineage; added strict standalone and full-release
  manifest checks. The fresh Qwen3 INT8 result is not selectable: NLL degradation
  is 0.95%, but next-token agreement is 83.5%. Preserved that negative result.
  Added the explicit opt-in verifier downloader, fail-closed CUDA/hybrid runners,
  fixed-tape 30-run schedule validation, durable synthetic 100-token RTL evidence,
  and workbench reports/optimization controls. `make verify-llm` now passes in
  the local venv. The current host is Mac/CPU, so real CUDA/hybrid acceptance
  remains unavailable until Zephyrus WSL2; no verifier was downloaded and no
  canceled acceptance job was resumed. Committed (`0a4455e`, `0242e06`), pushed
  the branch, and opened draft [PR #3](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/3)
  against `codex/model-aware-baseline`. Initial CI runs failed because the
  workflow omitted `libfl-dev` (`FlexLexer.h`); fixed and pushed as `1ddf980`.
  Corrected CI run `36395473732` successfully built Verilator and is running
  full verification; paired PR-triggered run `36395479523` is also queued/running.
  The frozen Qwen3 validation run also completed (0.78% NLL degradation,
  84.67% agreement, not selectable). A fresh Qwen3 full-RTL run completed
  7 prompt + 8 generated tokens in 14 bit-exact DRAM/TMEM steps; 86,022,637
  simulated cycles, host 0.00364 tokens/s. Code review found the prior
  bottleneck field mislabeled OpenTPU's numeric DRAM cycle bound as a category.
  Added a conservative categorized simulation-only diagnosis, preserved raw
  bounds separately, bounded event evidence to five largest gaps, and added
  regression coverage. Checkpoint-free full verify passed afterward; immutable
  report `build/release-evidence/verification-v3/verification.json` records
  commit `0242e06`, exit code 0, and no checkpoint downloads. An earlier
  in-flight Qwen3 run retains the pre-fix oversized raw counter summary; future
  runs use the bounded categorical diagnosis.

- 2026-09-28 (Codex, Zephyrus setup): Restored transfer bundle `a960dd3`
  into `/home/dhruv/projects/MalleableAI-FPGA` in Ubuntu 24.04 WSL2 as
  user `dhruv`; origin points to the project GitHub URL. All 3,555 manifest
  data files passed destination SHA-256 verification. All three transferred
  checkpoints passed local adapter inspection. Installed Verilator 5.050,
  Icarus 12.0, Yosys 0.33, Node 22.23.3, Python 3.12.3 and a project venv.
  PyTorch 2.11.0+cu128 passed actual CUDA matrix multiplication on the RTX
  5070 Ti Laptop GPU (compute capability 12.0). `pip check` passed.
  Full checkpoint-free `make verify-llm` passed on this laptop; immutable
  evidence is `build/release-evidence/zephyrus-setup-verification/verification.json`.
  Environment/freeze/checkpoint reports are in `build/release-evidence/zephyrus-environment`.
  The loopback workbench runs through user systemd service
  `malleable-workbench.service` using fresh `build/zephyrus-jobs`; Windows
  launcher `Start-Workbench.cmd` in the setup workspace keeps a WSL client
  attached and opens http://127.0.0.1:8765. Stop with
  `systemctl --user stop malleable-workbench.service`. Original Mac data and
  queues remain preserved; no historical jobs were resumed. This establishes
  environment readiness, not standalone/full-release model acceptance.
  Existing quality failures and CUDA/hybrid release gates remain applicable.

- 2026-09-28 (Codex, release implementation): Added bounded validation diagnostics, indexed resumable benchmarks, predictor v2, exact quality/configuration matching, controller partition/coverage checks, supported driver metadata, synchronized GPU/hybrid timing, and workbench conversation recovery. The 16-target Qwen3 pilot found 81.25% agreement and 1.01% NLL degradation; all 311 tensors match, both float references agree, and ISA/independent quantized top tokens agree. Acceptance remains blocked for a separate precision investigation. NVIDIA access was restored with G-Helper Eco to Standard and a fresh CUDA computation passed. See docs/zephyrus-release-progress.md for evidence identities and remaining work. Final-source interactive verification records are under build/zephyrus-jobs/release/20260928T223413Z-b463e746/final-verification. PR #3 remains draft; no merge or verifier download is authorized before standalone acceptance.
