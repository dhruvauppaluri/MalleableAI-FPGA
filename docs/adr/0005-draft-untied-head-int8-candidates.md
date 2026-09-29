# ADR-0005 (draft): Untied head tensors for INT8 candidate derivation

Status: draft, 2026-09-29. Not accepted; no code depends on it. Scope: Qwen3
compatible INT8 candidates ([recovery plan step 2](../qwen3-quality-recovery.md)).

## Why this exists

Qwen3-0.6B ties `lm_head` to `model.embed_tokens.weight`. That tensor is read
twice: as the fp32 input embedding row lookup and as the INT8 output head. The
compatible candidates in [`qwen3-int8-candidates.md`](../qwen3-int8-candidates.md)
therefore refuse the two operations that would need different values for the two
uses:

- head-input rescaling (`model.norm.weight` / `s`, head columns `* s`) would
  scale the embedding columns, so every input embedding, the residual stream and
  every RMSNorm statistic would change;
- head-weight clipping would clip embedding rows.

Attribution does not implicate the head (head weights and head inputs each
recovered no agreement on the 16-target panel), so nothing needs this yet.

## Proposed change, if the head is later implicated

Store a separate derived head tensor and stop sharing it with the embedding. This
changes storage, not arithmetic or rounding: the head is still quantized INT8 per
row and D-block and accumulated as today. The cost is one additional
`vocab x hidden` tensor in DRAM (about 155 MB INT8 for Qwen3-0.6B, before
padding) plus a compiler/`Image` change to place `lm_head.weight` separately when
`Spec.tied` is true, and its bit-exact DRAM check.

## Required before implementation

1. Measure that head weights or head inputs are implicated on the 128-target panel.
2. Decide the layout and update `docs/numeric-contract.md` or supersede this ADR.
3. Verify independent emulation, ISA/compiler and RTL agree on the untied layout.
4. Name the resulting variant accurately in policy and release identities.

Dense INT8/INT32 numerics and RTL are unaffected.
