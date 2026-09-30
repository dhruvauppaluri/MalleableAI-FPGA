# Finish the Zephyrus release and preserve a complete Cursor/Claude handoff

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

Authoritative continuation approved by the user on 2026-09-30. Older handoffs
are historical. Read this document and AGENTS.md and STATUS.md before work.

## Resume opening

> Continue the Zephyrus release from the verified current checkout. Read AGENTS, the current status, and the complete release-continuation plan. Inspect active jobs and completed evidence before dispatching. The user authorized one new eight-hour Overnight window and targeted Qwen3.5 activation/cache/operator redesign while retaining INT8 matrices/head and unchanged quality thresholds. Preserve the original deadline across agent changes. Fix the five reviewed safeguards first, then follow the plan's measured Qwen3.5 recovery and remaining release sequence. Do not repeat completed Qwen3/LFM held-out evaluations or LFM benchmarks. Keep running jobs durable and record exact resume instructions before stopping. Report blocked or unfinished acceptance honestly; never weaken gates or merge PRs.

## 1. Starting state, authorization, and completion criteria

Continue in `/home/dhruv/projects/MalleableAI-FPGA`, branch
`codex/local-llm-platform`, starting from commit
`7693ed9fcbe1a8e9bec3b25e91b66b56f2e49c1d` or its verified descendant.

| Evidence | Verified status |
| --- | --- |
| Qwen3 calibrated INT8 | Validation 94.04%; held-out 94.14%; eight-token RTL passes |
| LFM2.5 raw INT8 | Validation 93.75%; held-out 95.41%; eight-token RTL passes |
| Synthetic RTL | Fresh 100-token evidence passes |
| LFM benchmarks | All ten indexed runs verified |
| Qwen3.5 raw INT8 | Validation fails at 78.125%; held-out untouched |

The user selected **one new eight-hour Overnight window** and authorized
**targeted Qwen3.5 activation, cache, and operator redesign**. Primary acceptance
must retain **INT8 matrices and an INT8 output head**. The dense arithmetic
contract, frozen suites, >=90% agreement, and <=5% NLL degradation remain unchanged.

Start the eight-hour clock at the first new checkpoint-job dispatch, after
initial implementation and tests. Serialize heavy jobs and check the same
deadline before every subsequent dispatch, including benchmark indices. At the
deadline, stop starting jobs and let the active job finish. Agent changes and
retries do not reset the clock. The entire release will likely require additional
authorized windows.

Completion requires passing standalone schema-v3 and full-release schema-v2
checks, final-source verification, browser acceptance, actual matched
CUDA/hybrid evidence, and prepared PR review evidence. AWS, physical FPGA
deployment, retraining, altered suites, and automatic merges remain excluded.

### Verified identities and artifact locations

Evidence root (R):
`build/zephyrus-jobs/release/20260928T223413Z-b463e746`.
Original checkpoints: `build/models/{Qwen3-0.6B,Qwen3.5-0.8B,LFM2.5-230M}`;
frozen suites: `build/models/quality/{qwen3,qwen35,lfm2}-frozen-v2.json`;
official revisions: `docs/model-revisions.json`.

| Model | Base-model ID |
| --- | --- |
| Qwen3 | `7ab1181d3a2b04ce889880dfc3b94933574441e9c221e950622c39a3ce79a59d` |
| Qwen3.5 | `96c8847b78593ff15008955c68e384f5ff40895e425e487aec30b8173542a857` |
| LFM | `8d6ac525c1b135ad360d0f9f5a822698bbb257b7408f791e2ca695ae473798b3` |

Qwen3 candidate: `R/quality-recovery-pilot-01/int8/case-00`,
`a0.5-cnone`, derived ID
`3129232e64ec1532cfb735b48dac385bbeac4a072af98660d0fe85953cd163da`,
parameters ID `3840e2dbae1647c8b47984167fa6ba40fc5f6de044d7a4d905444bac8fbe2a08`.
Validation: `R/candidate-isa-validation-01` (963/1024), result SHA256
`c8c29777249ac9ac6ba7d47b304e03cb4ce23dfae75ec630a3c2fd892f82806d`.
Held-out: `R/candidate-heldout-freeze-01` (964/1024), result SHA256
`b513e1adb0e25e0b22f23eefadbf8b36441aebcee13e000061cd559e6a2197d9`;
freeze-v1 ID `e4def86bac90fe30ae318776115ee7c48b7c4e3f1098218915970c969a513b84`.
Eight-token RTL: `R/candidate-rtl-acceptance-02`, result SHA256
`ab0cfc2f77e56d2f0bcb6d7b2634ed00a46e36f91f5463915646e397b19e947a`.

LFM: `R/lfm2-raw-int8-validation-01` (960/1024),
`R/lfm2-raw-int8-heldout-01` (977/1024), held-out result SHA256
`95f6aeedcf2b324207acb596708c17e15f5df43d72da14b28a1800da4ea21371`;
`R/lfm2-rtl-acceptance-02`, result SHA256
`acaba9212635761ebe87ea8f50f44fa2ce8079c388b4e349e4de798073ea6dcb`.
LFM manifest: `R/perf/lfm2-performance-manifest.json`, manifest ID
`3b76747d715b0d10038a1502bb3938dffccbdf23eb5e64912d8bce60f96ba1ba`;
ten-run store: `R/perf/lfm2-01/store`; suite ID
`8486833b35eeba0ef1ccef4098df5e07360b6ccdb594dcd7eedceae59eae8f52`.
Measured driver duration: **7,492 seconds, approximately 2.08 hours**.

Qwen3.5 failed validation: `R/qwen35-raw-int8-validation-01`, 800/1024,
result SHA256 `2e7783a59c06cf17aced9bd82b8f6f7da9798b6c247f0510bb7e1a6bb553d424`.
Original NLL 4.3405801321, candidate 4.0360312144 (-7.016%); do not infer a
quantization cause until floating alignment is checked. Held-out is untouched.
Tiny: `R/tiny-rtl-100-01/tiny-release.json`, 100/100 steps,
SHA256 `bd739609c9acacf0e5d7cae5ead45ab6f628a4271b05ccc2d8537fb44fc52fa7`.

Preserve all passing runs, failed/canceled attempts, original hashes, and Mac
history. Do not repeat Qwen3/LFM balanced held-out evaluations or LFM benchmarks.
204 RTL trace hashes and canonical quality/result records were audited before
this continuation. CI at 7693ed9 was green; this is not final-source acceptance.

### Current resume state

At handoff creation: checkout clean at 7693ed9; no active release unit/PID.
New Overnight window **not started**; dispatch deadline **not set**. Old window
expired and does not authorize more jobs. First action: implement/test section B
and family-specific diagnostic controls before dispatching section C. Persist
new control at `R/continuation-20260930/control.json`; use fresh numbered attempts
under that directory. This location is reserved, not a claim that jobs exist.
Record active units/PIDs, exact commands, attempt paths, completion hashes and
deadline in STATUS after jobs finish; while jobs run use ignored attempt handoffs.

## 2. Implementation sequence

### A. Save the handoff before implementation

First save this complete plan as `docs/release-continuation-20260930.md`. Update
AGENTS.md and the top of STATUS.md to identify it as authoritative. Include
verified results/source/candidate/evidence identities, five confirmed review
findings and fixes, Overnight authorization and precision restrictions, work
that must not repeat, exact next action, active units/PIDs, attempts and deadline.
Preserve older handoffs as history. Correct current task rows and LFM benchmark
duration to 7,492 seconds / 2.08 hours. Refresh draft PR description with actual
results when preparing final review.

### B. Fix the five acceptance safeguards

Complete fixes and focused regressions before new checkpoint evaluation:

1. **Derived benchmark verification:** `full_release.py` currently passes only
   base-model/tokenizer IDs to manifest validation; candidate loader requires
   weight lineage and raises KeyError. Pass complete verified model lineage.
   Missing weight lineage must produce a clear validation error. The full-release
   checker must validate a legitimate derived benchmark without bypassing hashes.
2. **Hybrid draft approval:** resolve the approved draft from the standalone
   manifest. Compare model, tokenizer, variant, derived parameters, precision,
   personality, context and configuration before loading the verifier. Reject an
   omitted candidate if the approved draft is calibrated. Enforce the same in the
   full-release checker. Current runtime/final checker fail to bind approval.
3. **Held-out consumption:** replace directory-local claims with an atomic
   persistent registry outside attempts. Key by model, tokenizer, variant,
   configuration, frozen suite/split and declared evaluation design, not path or
   source commit. Register completed Qwen3 and LFM as consumed without rerunning.
   Copying a freeze/new folder cannot permit repetition. Claimed failed runs stay
   consumed; retries need a separately documented evaluation design.
4. **Freeze validation:** require every configuration field and recompute identity
   from configuration and microarchitecture. Missing fields fail. Support existing
   verified freeze-v1 artifacts; use v2 for new evaluations, preserve old hashes.
   Current create_freeze only compares fields if present and trusts config ID.
5. **Benchmark driver status:** preserve success indices and failed attempts,
   propagate child failures, distinguish complete/failed/deadline-limited.
   Never aggregate incomplete suites. Record clean source identity and verify
   unchanged during each index. Current driver exits zero after child failures.

Use the same registry and freeze discipline for later raw and derived release
configurations.

### C. Diagnose and recover Qwen3.5

Reuse existing Qwen3.5 independent reference and quantized emulator. Add
family-specific controls rather than duplicating implementation.

First freeze a 16-target validation panel. Compare original Transformers FP32,
independent floating reference, quantized emulation, and actual ISA. Verify
tokenizer alignment and converted tensors. Record NLL, top tokens, margins and
localized convolution, DeltaNet recurrence, attention, gates, normalization, MLP,
and output-head differences. Resolve/demonstrate rounding-related floating
alignment failures before attributing quantization. Convolution and recurrent
state already use FP32; do not describe them as new precision upgrades.

Freeze a deterministic 128-target spread panel with complete causal prefixes.
Document the exact Qwen3.5 site map before dispatch. Serialize ten distinct cases:

1. Baseline INT8.
2. All quantization bypassed in diagnostic emulation.
3. Transformer-weight quantization bypassed.
4. Projection-input quantization bypassed.
5. Attention key-cache quantization bypassed.
6. Attention value-cache quantization bypassed.
7. Attention query/probability quantization bypassed.
8. Output-head weight quantization bypassed.
9. Output-head input quantization bypassed.
10. Transformer-weight and projection-input quantization bypassed together.

Controls are diagnostic evidence, not deployable configurations.

Next implement compatible INT8 calibration: fit only existing calibration split;
screen strengths 0.25/0.5/0.75 without clipping; correctly handle zero-centered
norm gains and compensate every shared consumer. Initially transform only sites
with demonstrated unquantized equivalence. Do not blindly propagate scaling
through nonlinear gates, convolution or recurrent normalization. Preserve tied
embeddings/head, checkpoints and hashed derived artifacts. Rank by validation
agreement, then NLL, then lower strength. Send at most best three eligible
candidates to complete actual-ISA validation. Aim 92% buffer, actual gate 90%.

If compatible INT8 fails, select targeted activation/cache/operator changes
using attribution. Rank groups by recovered agreement, then NLL improvement,
declared case order breaks ties. Evaluate two best single changes and combination.
Prefer verified FP32 primitives; document storage, rounding, compiler, ISA and
RTL in an ADR before implementation. Matrices/head remain INT8. If passing needs
wider matrices/head or none of bounded candidates passes, preserve frontier and
stop acceptance with concrete design blocker.

### D. Freeze and complete Qwen3.5 standalone acceptance

For passing full-validation candidate, freeze code, original checkpoint/tokenizer,
derived parameters, precision policy, configuration and suites. Evaluate unchanged
>=1,024-target held-out **once**. Failure stops promotion, no tuning/repetition.
On pass, declared short prompt plus exactly eight greedy tokens through full RTL,
checking DRAM/TMEM each step. Preserve EOS attempts; declare alternative prompt
before execution retry, never alter quality suite. Assemble standalone schema-v3
with verified Qwen3/LFM/tiny and new Qwen3.5 evidence. Reuse history only where
model/config/arithmetic/execution dependencies compatible. Record reviewed source
compatibility separately; never rewrite old records or repeat consumed held-out
because safeguards changed.

### E. Complete performance and controller evidence

Add performance manifest v2 with per-run candidate/precision lineage so calibrated
INT8 and independently approved raw four-bit comparisons share a schedule. Read
and verify historical v1 unchanged. Extend creation with candidate support.
Preserve seed42, tapes, memory scenarios and indexed execution. Reuse ten LFM
runs; finish remaining twenty Qwen3/Qwen3.5 runs. Evaluate balanced INT4/FP4
validation; use both extras only if both pass, otherwise two prescribed INT8
stress runs. Complete validation quality for four controller personalities and
matching held-out approval before automatic application; reuse balanced approvals.

Build reproducible episodes from stored experiment IDs. Freeze partition:
Qwen3/Qwen3.5 training, LFM held-out. Fit predictor schema-v2 with FIFO depth,
bandwidth percentage, correct fixed-tape work; freeze before held-out predictions.
Double DQN seed0,20passes; evaluate seeds100-104 against fixed, heuristic,
predictor, budget-matched random and exhaustive. Cover latency/throughput,
horizons1/8/32/128, fourINT8 personalities, declared assumed switching costs.
Preserve per-case regressions. Promote only if existing non-regression gate
passes; otherwise retain validated deterministic controller.

### F. Finish workbench, CUDA/hybrid, and release review

Current-source browser acceptance: chat formatting, reconnect, duplicate/stale
events, cancellation, restart, report lineage, recorded Lens replay, keyboard
access, explicit validated application between windows with reset/re-prefill.

Only after standalone passes: revalidate actual WSL CUDA and supported NVIDIA
driver reporting; opt-in download official Qwen3-1.7B, freeze revision/hashes/
tokenizer compatibility. Matched GPU and full-RTL-draft hybrid: context128,
eighttokens, FP16/eager, TF32off, depth4. Actual CUDA cache correction/cropping
depths1/2/4/8, rejection positions/EOS/context boundaries. Require exact greedy
token agreement and synchronized loading/build,prefill,decode,inference-total.

Assemble full schema-v2; fresh final-source verification and both checkers;
verify current-tip CI; prepare independent PR#1/#3 reviews against dense baseline.
Leave PRs unmerged.

## 3. Tests and acceptance

Required regressions: legitimate derived benchmarks pass complete release path,
missing lineage fails; raw/wrong hybrid cannot borrow another variant approval;
concurrent claims/copied freezes/relocated attempts/changed source cannot reset
held-out; missing/tampered config/candidate/suite/precision rejected; benchmark
child failure propagates, deadline preserves resumability/completed reuse;
Qwen3.5 controls preserve baseline and shared consumers/zero-centered gains/gates/
recurrence reset/partial blocks/context; disjoint calibration/validation/held-out;
predictor/controller reject LFM leakage/oracle observations; required browser and
actual CUDA cases. Focused subsystem tests and hardware verification when
compiler/ISA/RTL changes, complete final-source suite at end. Emulation/mocks are
never actual acceptance evidence.

## 4. Durable execution and handoff contract

Persistent user-systemd jobs, shared heavy lock, fresh append-only directories,
memory preflight, canonical checksums. Each attempt records source, command,
authorization, PID/unit, settings, result hashes and failure disposition. Persist
first dispatch deadline once in continuation-control; all successors reuse it.
Verify sleep prevention after first monitoring interval; laptop powered/WSL up.
Do not edit tracked source/docs during source-bound workers. Interruptions:
new timestamped handoff in ignored attempt; after completion update STATUS and
commit reviewable work. If credits expire save handoff immediately and leave
durable worker running; successor inspects worker/artifacts. After deadline safe
implementation/checkpoint-free tests allowed, heavy dispatch waits for newly
selected window. No automatic extension.
