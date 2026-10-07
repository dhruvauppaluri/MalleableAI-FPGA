# Controller quality continuation — 2026-10-06

The previous cloud workspace is inaccessible. Its ignored results are not treated
as reviewable evidence, but every known held-out dispatch is treated as consumed.
`consumed-ledger.json` carries the three published balanced claims, tombstones all
12 configurations on the frozen 2026-10-05 controller suite, and records the three
ADR-based recovery claims whose exact suite identities were lost. No prior RTL
benchmark or held-out suite may be rerun.

The replacement campaign uses `examples/llm-controller-quality-text-20261006.json`.
Its source documents are disjoint from both earlier published text manifests and
exclude all ADR files. Before scoring, model-specific tokenized suites must be
checked for zero eight-token overlap with both earlier frozen suites and for zero
cross-split overlap. The frozen suite manifest will be published before held-out
dispatch.

Run 12 validation controls so compact, balanced, compute, and buffered are
comparable on the same new suite. Only compact, compute, and buffered may be
frozen and evaluated on held-out data. Balanced already has published approval;
its held-out split must not be repeated. Each of the nine new held-out designs is
claimed exactly once before scoring. Results are published incrementally, including
failures and interrupted claims. Automatic switching remains gated regardless of
quality outcome until transition costs and controller non-regression evidence exist.

## Snapshot 01: LFM validation

All four LFM2.5-230M validation controls passed on the independent suite at
93.6523% agreement and approximately +0.04494% NLL degradation. Compact's
calculation completed on `cafee61` but the wrapper initially rejected its own
untracked cache; the published recovery revalidated the sole completed result,
suite, model, record identity, and gate without inference. The other three ran
after the cache route was corrected on `b8da3c2`. The snapshot also preserves the
two memory-killed Qwen compact validation attempts. No held-out split was opened.
See `snapshots/01-lfm-validation.json` and `published/validation/`.

## Snapshot 02: Qwen3 validation, partial

Compact, balanced, and compute passed the independent validation suite at
92.9688% agreement and approximately -0.04618% NLL degradation. The snapshot
also preserves the original memory-killed compact attempt and a buffered attempt
that reached 768 of 1,024 candidate targets before its detached output pipe
closed. Both are validation-only infrastructure failures: neither opened or
claimed held-out data. Buffered will use a distinct retry root. No held-out split
has been opened. See `snapshots/02-qwen3-validation.json`.

## Snapshot 03: Qwen3 validation complete

The distinct buffered retry completed and passed with the same 92.9688%
agreement and approximately -0.04618% NLL degradation as Qwen3 compact,
balanced, and compute. All four Qwen3 controls now pass the independent suite.
The interrupted buffered attempt remains visible; it was not overwritten or
reclassified. No held-out split has been opened. See
`snapshots/03-qwen3-validation-complete.json`.

## Snapshot 04: Qwen3.5 memory recovery boundary

The first serial Qwen3.5 compact retry completed and durably cached all eight
floating-reference rows, then was OOM-killed while constructing the ISA
candidate. It opened no held-out split. The cache is content-addressed and bound
to the exact model, tokenizer, split, suite, dtype, attention implementation,
framework versions, and reference contract. Subsequent distinct validation
attempts may reuse that verified reference cache so the float model and ISA
image never coexist. The runner now accepts an explicit ignored-build cache
path, and explicitly collects the released reference model before ISA image
construction. See `snapshots/04-qwen35-memory-recovery.json`.

## Snapshot 05: bounded Qwen3.5 image construction

Reusing the verified float reference proved that Qwen3.5 candidate image
construction itself exceeded memory. The second distinct retry is preserved as
a validation-only OOM failure. Its failure occurred before target scoring and
opened no held-out split. The vendored Qwen3.5 image builder now quantizes
row-independent matrices in bounded chunks. A 5,001-row comparison confirmed
the chunked INT8 data and scale arrays are bit-identical to whole-matrix
quantization, and the tiny Qwen3.5 reset/state test passes. See
`snapshots/05-qwen35-candidate-memory-recovery.json`.

## Snapshot 06: Qwen3.5 compact validation

The third distinct compact attempt completed after bounded image construction
and passed at 95.9961% agreement and approximately +0.03861% NLL degradation.
No held-out split was opened. Subsequent ISA scoring uses bounded eight-row
causal programs that return every intermediate logit; a tiny Qwen3.5 comparison
confirmed all eight outputs are bit-identical to eight one-token decode calls.
See `snapshots/06-qwen35-compact-validation.json`.

## Snapshot 07: personality-specific ISA row width

The first Qwen3.5 balanced attempt under multi-row scoring was rejected by the
compiler because eight rows exceed that personality's TMEM. It scored no target
and opened no held-out split. ISA scoring now compiles candidate widths 8, 4, 2,
and 1 for the exact personality before scoring, selects the widest legal width,
and records it in the result. See
`snapshots/07-qwen35-row-width-recovery.json`.

## Snapshot 08: detached PTY recovery

The distinct balanced retry reached 256 targets before a detached PTY made its
best-effort progress print raise `OSError(EIO)`. The attempt is preserved as a
validation-only infrastructure failure and opened no held-out split. Progress
reporting now ignores stdout `OSError`s as well as ordinary broken pipes; a unit
test covers this boundary. Durable files remain authoritative. See
`snapshots/08-qwen35-detached-pty-recovery.json`.

## Snapshot 09: Qwen3.5 balanced validation

The second balanced retry passed the independent validation suite at 95.9961%
agreement and +0.03861% NLL degradation. The exact personality compiled a
four-row ISA width; its completed result records that width. This is a
same-suite control and does not reopen balanced held-out approval. See
`snapshots/09-qwen35-balanced-validation.json`.

## Snapshot 10: Qwen3.5 compute validation

Compute passed the same frozen validation suite at 95.9961% agreement and
+0.03861% NLL degradation, with a compiled four-row ISA width. Three of four
Qwen3.5 personalities now have passing independent validation. See
`snapshots/10-qwen35-compute-validation.json`.

## Snapshot 11: all independent validations complete

Qwen3.5 buffered passed at 95.9961% agreement and +0.03861% NLL degradation.
All 12 model/personality validation controls now pass on the independent suite.
The 19 attempted validation roots include seven preserved infrastructure failures
or interruptions. Only the nine compact, compute, and buffered designs are
eligible for new held-out claims; balanced's published held-out approvals are
never repeated. See `snapshots/11-all-validation-complete.json`.

## Snapshot 12: held-out preflight

The independent suite, complete validation matrix, frozen source, and durable
prior-consumption ledger are published before any new held-out claim. The
held-out snapshot publisher checks each validation, freeze, exact-once claim,
terminal result or failure, and artifact hash before appending a claim to the
durable ledger. The preflight snapshot has zero new claims. Automatic switching
remains gated. See `snapshots/12-heldout-preflight.json`.

## Snapshot 13: LFM compact held-out

LFM2.5 compact was frozen at source `0139d78`, claimed exactly once as evaluation
`acddb5821c8ced9b790d484ffaaf719cc7aaac1887ab9478d689eeb0bf3a04a0`,
and passed 1,024 held-out targets at 92.9688% agreement and +0.09223% NLL
degradation. The freeze, atomic claim, selectable result, and summary hashes are
published under `published/held-out/`. The consumed key is appended to the
durable ledger. See `snapshots/13-lfm-compact-heldout.json`.

## Snapshots 14–15: LFM compute and buffered held-out

Both configurations were frozen and claimed separately, then passed 1,024
held-out targets at 92.9688% agreement and +0.09223% NLL degradation. Their
freeze, claim, result, and summary artifacts are published under
`published/held-out/LFM2.5-230M/`; both consumed keys are in the durable ledger.
With compact, all three missing LFM personalities now have reviewable quality
approval. Transition costs and policy gates still block automatic switching.

## Snapshot 16: Qwen3 compact held-out

Qwen3 compact was frozen and claimed once on the independent suite. It passed
1,024 held-out targets at 94.0430% agreement and +0.03373% NLL degradation.
The freeze, claim, result, and summary are published under
`published/held-out/Qwen3-0.6B/compact-retry-01/`; the exact claim is in the
consumed ledger. Automatic switching remains gated.
