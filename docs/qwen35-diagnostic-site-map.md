# Qwen3.5 validation-only site map

Pinned `opentpu.llm.qwen35.emulated_logits` has ten activation sites, asserted by
AST expression before instrumenting. All controls keep the upstream calculation
and observe or bypass only declared quantizers. This is float64 diagnostic
emulation, not an ISA/RTL precision policy.

| Group | Sites and consumers |
| --- | --- |
| Transformer weights | Every `w(n)` projection except tied embedding/output head; includes linear-attention qkv/z/a/b/out, full-attention q/k/v/o, MLP gate/up/down |
| Projection inputs | Shared input norm for linear qkv/z/a/b or attention q/k/v; linear out projection; attention out projection after sigmoid gate; shared post-attention norm for MLP gate/up; SiLU(gate)*up for down |
| Key cache | Full-attention normalized, rotated K; block128 |
| Value cache | Full-attention V; head-dimension block256, as upstream |
| Attention | Scaled query and padded unnormalized exponential probabilities; normalizing denominator remains upstream |
| Head weights | Tied embedding/output projection weight, or untied lm_head |
| Head input | Final zero-centered normalization output |

Convolution taps/ring and DeltaNet state already use FP32 on the device; emulator
uses float64 without device rounding. Normalization, L2 norm, sigmoid/SiLU gates,
softplus/decay and recurrence have no added quantization controls in these ten
cases. Diagnostics observe existing operators; they do not upgrade production
precision. Zero-centered effective normalization gain is `1 + stored weight`;
DeltaNet output norm is plain gain. Every shared consumer must be compensated in
any later calibration transformation. Never scale recurrence/gates blindly.

Local operator comparisons retain at most the first 4096 flattened elements per
operator/token/layer, explicitly labeled samples. Output losses/top-token margins
and conversion checks use complete tensors/logits. Hooks preserve reference math.

The first real 16-target result had identical top tokens but a 2.082% relative
floating-logit difference. `quality-reference-diagnose` compares the unchanged
original Transformers chunked FP32 reference with a diagnostic-only replacement
using Transformers' existing sequential FP32 DeltaNet helper, and the pinned
independent FP32 reference. It also checks converted tensors directly against
original Safetensors files. This control neither changes ordinary quality
evaluation nor supplies selectable/release evidence. Do not assume that any
remaining difference is quantization error; preserve and localize it first.
