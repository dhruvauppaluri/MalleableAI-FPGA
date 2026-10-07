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
