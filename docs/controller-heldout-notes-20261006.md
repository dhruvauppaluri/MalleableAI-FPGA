# Independent review notes for adaptive accelerator control

An adaptive accelerator is valuable only when adaptation repays its disruption.
The final review therefore considers a sequence of requests rather than an
isolated best-case kernel. Every request has an input length, an expected amount
of generated work, a service objective, and a currently resident program. The
decision procedure may retain that resident program or recommend another one.
Either outcome is legitimate when supported by the recorded evidence.

Consider a short interactive request arriving while useful state is already
resident. A wider compute organization might reduce arithmetic cycles slightly,
yet rebuilding state could cost more than the entire request. Retention is then
the rational action. For a long repeated batch, the same organization may have
enough future work to amortize its entry cost. The controller must distinguish
these situations without reading the future measurement it is supposed to
predict.

Service time is decomposed into observable terms. Preparation covers program
availability and compilation artifacts. Admission covers resource checks and
queue placement. Prompt work covers the causal prefix. Incremental work covers
new tokens. A transition may add drain, load, restore, warmup, and replay terms.
The report states which terms are measured, which are calculated from counters,
and which remain explicit assumptions. Combining them into one unexplained
number would hide the experiment's most important uncertainty.

Resource legality precedes optimization. A program that exceeds memory capacity,
uses an unsupported operation, violates a context bound, or lacks approved
numeric behavior is removed before ranking. The mask is part of the decision
record. An optimizer cannot earn a benefit by choosing an action the runtime
would reject. Unknown compatibility produces inspection-only status rather than
an optimistic default.

Quality legality is equally specific. Approval follows the exact model revision,
tokenizer content, quantized variant, compiled configuration, and frozen suite.
Approval for a balanced program does not automatically authorize a compact or
compute-oriented program. Likewise, a validation pass shows that a design is
ready for independent review; it is not itself the final approval. The review
partition is claimed before evaluation and remains consumed after interruption.

The evidence ledger exists outside transient job folders. It records published
claims as well as claims whose raw outputs were lost with an unavailable machine.
This conservative treatment prevents accidental reuse from looking like a clean
experiment. When an old claim lacks enough metadata to reconstruct its exact key,
the replacement study excludes its entire source boundary. The cost is additional
work, but the benefit is an auditable conclusion.

Performance rows also have strict ancestry. Each row points to its manifest,
run index, workload tape, precision policy, configuration, and checked execution
trace. Aggregation verifies those fields before fitting a predictor. Completed
rows are never regenerated merely to simplify a new analysis. Reuse is permitted
because their identities and outputs are already public; mutation is not.

The baseline predictor is a measured hypothesis, not an oracle. Training error
can be small while a held family behaves very differently. The review examines
absolute percentage error, signed bias, ranking stability, and the worst miss.
It also asks whether normalization by work, dimensions, or observed counters
improves transfer without leaking the target. Any replacement predictor is
frozen before the isolated family is scored.

Policy evaluation uses episodes assembled from published rows. Episode order is
declared before scoring so that a method cannot receive a favorable sequence
chosen after its behavior is known. The state contains the resident action,
request descriptors, remaining horizon, frozen predictions, and recent observable
counters when available. Actual service cycles affect the environment's accounting
but are withheld from the policy until they would naturally have been observed.

Several comparators guard against a decorative learning result. Fixed balanced
execution represents the established safe system. A rule-based method applies
the same margin and uncertainty logic used by the product controller. A frozen
predictor method chooses from its estimates. Random search receives the same
action budget as the learned method. Exhaustive analysis supplies a retrospective
lower bound and is never described as an online policy.

The learned candidate is evaluated over multiple deterministic seeds. Reports
retain every regression rather than averaging away a harmful episode. Promotion
requires non-regression against both fixed and heuristic references, compatible
lineage, an isolated evaluation family, and an explicit disposition. If those
conditions are not met, the current deterministic behavior remains selected.
There is no automatic promotion simply because training completed.

Uncertainty changes the break-even calculation. Suppose an alternative appears
faster by a narrow amount while prediction error is broad. The controller reduces
the apparent benefit by an uncertainty allowance before comparing it with entry
cost and the minimum margin. This can turn an attractive point estimate into a
recommendation to stay. That outcome is expected when evidence is weak, not a
failure of the interface.

Residence protects the service from oscillation. After a justified transition,
the new program remains active for a declared decision window unless correctness
or resource safety requires an immediate stop. Without residence, small prediction
changes could cause repeated reloads that dominate useful computation. The episode
accounting includes every such transition, making unstable behavior visible.

Physical claims remain separate from simulation claims. Verified instruction and
state traces establish functional execution in the simulator. They do not reveal
board timing, power, routing closure, configuration-port bandwidth, or thermal
behavior. Target measurements must identify the device, toolchain, bitstream,
clock, interface, and measurement method. Until then, transition scenarios are
labelled offline analyses and automatic application stays disabled.

The independent quality result produced from this document answers one narrow
question: whether an already defined numeric implementation retains next-token
behavior on text that was frozen before scoring. It does not train the model,
select thresholds, or tune a candidate. A passing result can unlock controller
research for that exact program, but it cannot by itself prove acceleration,
predictor generalization, or policy superiority.

Review ends with a complete ledger. For every planned design, the ledger shows
validation status, freeze identity, claim time, terminal state, result digest when
available, and publication commit. Interrupted work remains visible. Missing work
is marked pending rather than inferred from another personality. This makes the
next continuation mechanical: verify the ledger, choose only untouched designs,
and preserve the safety gate until all remaining evidence is reviewable.
