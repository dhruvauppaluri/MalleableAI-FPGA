# Status

Handoff for Cursor, Codex, and Claude. Update this before ending a session.

Last updated: 2026-09-28. Any of Cursor, Codex, or Claude may claim any `open` row.

## Now

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
The user has now authorized these remaining nine cases, serialized, then report.
Active attempt will be `precision-128-remaining-01/` in the same release root;
inspect its `batch.json` PID, per-case logs/summaries, and final report/failure
before resuming. Runner is snapshotted as `runner.py` inside the attempt and
reuses a copied floating-reference store from the verified pilot. Cases 1–9 use
the exact earlier policies, including the fixed combined bypass. No held-out or
subsequent calibration batch is authorized by this dispatch.

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
| in progress | Codex | Qwen3 quality recovery and precision attribution | User authorized nine remaining 128-target cases; inspect precision-128-remaining-01 before dispatch. Estimated 35–45 minutes. |
| open | Codex | Zephyrus release completion implementation | Infrastructure verified at b1f2d3f; measured model release remains blocked. Resume after quality recovery. |
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
| in progress | Codex | Standalone full-RTL acceptance | Tiny 100-token and Qwen3 short-prompt + 8-token evidence pass bit-exactness. Qwen3 quality fails the 90% agreement gate; Qwen3.5/LFM evidence remains outstanding. |
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
