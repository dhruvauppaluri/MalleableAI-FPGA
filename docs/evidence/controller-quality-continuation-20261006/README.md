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
