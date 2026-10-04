# ADR-0007: Derived INT8 candidate production identity and single held-out freeze

Status: accepted for the Qwen3 quality recovery (2026-09-29).

## Context

The compatible INT8 candidate `a0.5-cnone` (calibrated channel rescaling, no
clipping) passed full actual-ISA validation. It changes only float32 weight and
norm tensors; the dense INT8/INT32 contract, ISA, compiler and RTL are
unchanged. Until now only `quality-candidate-validate` could load it, and it
refused held-out data. Generation, benchmarks, release checks and the workbench
had no way to bind a run to a derived candidate, so a raw-INT8 run and a
calibrated-INT8 run could not be told apart by identity.

## Decision

1. **Variant identity.** `variant_id = identity({base, format, head})` for
   legacy raw variants (hash unchanged). A derived candidate adds `derived_id`
   and `parameters_id`, so calibrated INT8 evidence never matches raw INT8
   evidence. One helper (`candidates.variant_id`) is used by quality,
   generation, performance manifests and release checks.
2. **Loading.** Every consumer loads tensors only through
   `read_derived_candidate` + `apply_derived_candidate` (lineage, shape, dtype,
   and per-tensor SHA-256 checks). The FP32 floating reference is always the
   original checkpoint. Tied embedding/head handling is unchanged (no untying).
   Candidates are INT8-only; INT4/FP4 combinations are refused.
3. **Workloads and jobs.** `GenerationWorkload.candidate_case` is optional and
   omitted from the workload hash and stored workload when absent, preserving
   historical workload IDs. Results record `derived_candidate`. Workbench
   generate/benchmark jobs accept a case only inside the job data root and
   re-verify it against the selected model. Performance manifests may carry
   `candidate_case` and `derived_candidate`; their variant is derived from it.
4. **Freeze.** `tools/freeze_candidate.py` creates `freeze.json` from a completed,
   passing, hash-verified full validation. It binds the clean code commit,
   original model/tokenizer/weight hashes, derived and parameter/calibration
   IDs, derived tensor file hash, frozen suite hashes/counts, precision and the
   exact configuration ID. Held-out evaluation recomputes all of it, requires
   the same clean commit, and atomically creates `heldout-claim.json` before
   the first candidate computation. A claim is never overwritten: one
   evaluation per frozen candidate. Infrastructure failure after the claim
   requires a documented new evaluation design, not a silent retry.
5. **Evidence.** Held-out results carry `candidate_freeze_id`. The standalone
   `release-check` requires, for any derived-candidate run or quality record,
   a `candidate_freeze` artifact in the model entry and one consistent
   identity across generation, held-out quality and the freeze.

## Consequences

The numeric contract is unchanged. Search still uses calibration/validation;
held-out is reachable only through a freeze and only once. The workbench cannot
launch derived-candidate quality jobs, hybrid jobs from the UI, or accept
arbitrary paths. Qwen3.5 and LFM2.5 require their own candidates and freezes.
