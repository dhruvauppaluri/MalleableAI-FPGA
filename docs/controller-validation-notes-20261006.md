# Validation notes for model-aware program choice

The controller study asks whether a host can choose among several accelerator
programs without weakening inference quality. A program describes a compiled
microarchitecture and its operating constraints. It is not a promise that the
fabric changes instantaneously. The host may compare programs, but a comparison
is useful only when correctness, workload identity, and transition assumptions
are explicit.

The first obligation is lineage. A measurement belongs to one immutable model,
one tokenizer, one quantized variant, one program configuration, and one input
tape. Those identifiers travel together. A row with a matching display name but
a different content identity is a different experiment. The implementation
therefore validates hashes before it considers latency, and it rejects partial
records rather than filling blanks from neighboring runs.

The second obligation is arithmetic retention. Each candidate is compared with
the original floating reference on complete causal prefixes. Agreement counts
the candidate's next-token winner, while loss change observes probability mass
assigned to the authored target. These measurements answer related but distinct
questions. A low loss change does not repair too many changed winners, and a high
winner count does not excuse a badly shifted distribution. Both predeclared
limits must pass. Validation remains nonselectable because it is used to inspect
and freeze a design before the independent review partition is opened.

The third obligation is execution evidence. Cycle counts are accepted only from
records that also preserve model state checks. A short run can expose program
differences, yet it cannot establish general service behavior by itself. The
study keeps compilation, loading, prompt processing, token generation, and host
overhead separate. This prevents an attractive projected clock rate from being
presented as measured end-to-end latency. It also prevents CPU simulation time
from being mislabeled as physical-board performance.

The fourth obligation is comparability. Compact, balanced, compute-oriented,
and buffer-oriented programs are evaluated on the same validation tape for a
given model. The balanced record is a control in this replacement campaign.
Its previously published final approval remains authoritative and is not opened
again. The three alternatives may proceed only when their own validation rows
pass and their exact designs have been frozen. A freeze binds source code,
checkpoint files, tokenizer files, suite content, candidate parameters, context,
numeric format, and microarchitecture configuration.

The fifth obligation is budget discipline. A final-review claim is consumed at
dispatch time, not at successful completion. A crash, disconnection, exception,
or malformed output does not restore the right to examine the same partition.
This rule removes an otherwise subtle incentive to rerun unlucky or inconvenient
outcomes. Durable tombstones from earlier work are checked in addition to the
workspace database, so creating a new checkout cannot erase experimental memory.

Program choice also needs a cost model. Keeping the resident program is always
a legal action. Moving to another one can require draining queued work, loading
configuration data, rebuilding derived buffers, restoring model state, warming
runtime caches, and replaying prompt context. Until those components are measured
on the intended target, the controller may report labelled scenarios but may not
claim a deployable improvement. Zero-cost analysis is an optimistic bound, not a
substitute for transition evidence.

A predictor is evaluated differently from a ranker. Ranking the fastest program
on a few tapes may look successful even when absolute cycle estimates are badly
wrong. The release study therefore reports per-model error, worst-case error,
and held-family behavior. Fitting uses the two designated training families;
the remaining family stays isolated until evaluation. Any feature derived from
the answer being predicted would leak oracle timing and invalidates the study.

The policy comparison has the same boundary. A learned agent receives frozen
predictor outputs and observable request state, never future measured cycles.
Its reward may use measurements inside the evaluation environment, but those
measurements are not policy inputs. Fixed balanced execution, a deterministic
heuristic, predictor-only selection, budget-matched random choice, and exhaustive
analysis provide reference points. Multiple seeds are required because a single
favorable trajectory cannot demonstrate reliable improvement.

Promotion is intentionally conservative. Quality approval alone does not enable
switching. Passing controller evaluation alone does not enable switching either.
The review requires matched quality, complete benchmark lineage, credible costs,
uncertainty allowance, a minimum improvement margin, and a residence interval.
If a candidate fails any part, the deterministic safe behavior remains active.
Negative results are retained because they constrain future design choices.

This validation document is frozen before any score is computed. It describes
the questions the experiment must answer, not the answers. The resulting record
must include target count, agreement, reference loss, candidate loss, program
identity, suite hashes, and a content-derived record identifier. Reviewers should
be able to recompute the gate and verify that no later file silently replaced the
design under test.

Operationally, the batch is serialized within each model family and isolated
across attempt directories. Progress messages are informational; only a terminal
record with a verified digest establishes completion. Cached floating references
may be shared when their lineage matches exactly, while candidate outputs remain
specific to one program. Publication occurs in bounded checkpoints so a later
machine loss cannot erase already reviewed facts. The public ledger is therefore
part of the experiment, not merely a convenience for the operator.
