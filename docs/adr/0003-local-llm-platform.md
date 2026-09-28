# ADR-0003: Local LLM platform, AWS last

Status: Accepted. Date: 2026-09-27. Decider: project owner.

## Context

The owner approved replacing the custom SSM product with pretrained LLM
execution using OpenTPU, retaining the model-aware/malleable objective.
The independent dense foundation remains authoritative and unchanged.

## Decision

Vendor OpenTPU revision `15754e971b55591b91048c4c636023fe59b343e7`, preserving
its attribution, and implement our adapters separately. Its ISA contract uses
FP32 round-nearest-even/flush-to-zero vector arithmetic and block-quantized
matrix operations. This is NOT the dense ties-away INT8/INT32 contract or
ADR-0002 Q14. RTL must match the corresponding ISA bits; floating model quality
is a separate gate (NLL degradation <=5%, next-token agreement >=90%).

Retire custom SSM commands, training, RTL and characterization. Keep historical
docs/commits and user files; no history rewrite. Keep dense schema-v1 readers.
Default chat executes prompt AND generation in Verilator. ISA is opt-in.
Profile telemetry distinguishes actual simulation, replay, analytic estimates,
host wall time and assumed-clock projections; physical metrics are unavailable.

Use local Safetensors only, no remote code or automatic downloads. Enable
greedy draft/GPU verification only AFTER standalone release evidence for all
three supported models. Training/distillation and AWS are excluded. The newly
approved staged GPU/hybrid scope supersedes the earlier blanket Stage 2/3
exclusion; it does not remove the standalone acceptance gate.

## Alternatives and consequences

Rewriting a complete transformer core delays a credible baseline. Keeping the
SSM product conflicts with the requested pivot. OpenTPU reuse brings a tested
execution stack but not universal model support or a physical F2 port.
An unchanged core with an RL wrapper is not structural malleability: validated
compiled personalities and explicit switching costs are required.
Real models and CUDA are release prerequisites, not implicit skipped successes.

## Delivery

Reviewable milestones: retirement/vendor boundary; standalone execution;
live workbench; optimization/learning; gated GPU/hybrid. No AWS provisioning,
upload, AFI generation, board programming or automatic merge to main.
