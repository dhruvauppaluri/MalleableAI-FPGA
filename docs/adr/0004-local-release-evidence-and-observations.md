# ADR-0004: Strict local release evidence and non-oracle policy observations

Status: accepted, 2026-09-27. Scope: the approved local completion plan.

Historical schema-v1 records and their content hashes remain immutable. New
release manifests use schema v2 and bind completed execution, tokenizer,
configuration, held-out quality, suite hashes and correctness evidence.
Validation and held-out quality each require at least 1,024 target tokens. A
quality failure is retained, never converted into approval by changing arithmetic
or excluding a required model. Quality references are cached by base-model,
tokenizer, token-suite and floating toolchain identity, separately from candidates.

New generation records add explicit configuration and input identities without
changing the interpretation of historical workload IDs. Conversation inputs are
validated role/content messages and re-prefilled from reset. Benchmark comparisons
use identical fixed input token tapes, not candidate-dependent greedy outputs.

Job metadata belongs to a separate envelope referring to the canonical artifact
hash. Adding a run ID must not change a policy/checkpoint identity. Terminal
failures and cancellations have their own append-only provenance records.

Policy observation schema v2 excludes untried candidates' measured timings.
Only frozen predictor estimates and current observed counters enter the state
and decision safety guard. Environment rewards and the exhaustive evaluation
oracle may use measured tables. Legacy policies remain readable but cannot be
promoted as non-leaking v2 policies. Fixed, heuristic, predictor, random and
exhaustive baselines are distinct. Runtime automatic application is opt-in and
must enforce quality, costs, residence and reset/re-prefill boundaries.

Simulator personalities remain compiled designs. These changes introduce no new
RTL arithmetic, physical FPGA backend, runtime hardware modes or AWS integration.
CUDA/hybrid execution is blocked until the strict standalone gate passes; a
strict full-release manifest also requires actual WSL2/Zephyrus CUDA evidence.
