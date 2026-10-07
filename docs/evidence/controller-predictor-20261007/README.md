# Offline controller continuation — published RTL evidence

`tools/evaluate_controller_continuation.py` verifies and reads the 30 previously
published indexed RTL benchmark rows. It does not launch RTL simulations or
quality scoring. The prior Zephyrus archive is checked against its published
SHA-256 before its three balanced held-out quality records are read. All nine
new held-out approvals must be present in the consumed ledger, and each of the
four personalities must have a matching approved quality record for every
benchmark row used in the controller episodes.

The frozen ridge predictor trains on 20 Qwen3/Qwen3.5 rows and scores ten LFM
rows without fitting to LFM. Mean absolute percentage error is 25.58% on Qwen3,
19.21% on Qwen3.5, and **109.85% on LFM**. The earlier published predictor audit
already inspected the LFM rows, so this is an exploratory reuse of that family,
not an untouched model-selection holdout.

All three models' short-a and short-b tapes have four INT8 personalities. The
two `axi-stress` rows per model cover only balanced and compute; they are
reported separately and excluded from four-action episodes. Compute is the
retrospective fastest personality on all nine tapes, but its largest measured
gain over balanced is only **2.98%**. The predicted gain over balanced is at
most 1.45%. Neither clears the controller's 5% improvement margin plus 10%
uncertainty allowance before any transition cost.

The offline episode audit uses short-a and short-b fixed tapes, eight declared
latency/throughput windows at horizons 1, 8, 32, and 128, and assumed switch
costs of 0%, 5%, or 20% of balanced service. It trains a masked Double DQN on
Qwen3/Qwen3.5 and evaluates on LFM with five deterministic seeds. Learned,
fixed-balanced, heuristic, frozen-predictor, and safety-constrained exhaustive
costs are equal in every LFM scenario; none finds an approved switch. Random
search costs more because its tried configurations are charged as exploration.
Zero per-case regressions therefore show safe retention, **not policy
superiority**. The exhaustive comparator is constrained by the same predictor
safety rule; it is not an unconstrained physical oracle.

The report's service cycles are measured fixed-token RTL cycles. Request
sequences and switch costs are assumptions, and no physical program, reload,
drain, warmup, or reprefill measurement exists. Automatic program switching and
policy promotion remain gated. The predictor's poor LFM transfer also blocks a
generalization claim.

Artifacts:

- `offline-evaluation.json`: verified row IDs, quality evaluation IDs, predictor
  errors, measured/predicted personality comparisons, episode design, all five
  seeds, all per-case results, and limitations.
- `predictor-checkpoint.json`: frozen 20-row ridge coefficients and lineage.
- `policy-candidate.json`: offline Double DQN candidate and training replay
  digest. The replay can be regenerated from the scripted episodes and seed;
  candidate status only, never active.
