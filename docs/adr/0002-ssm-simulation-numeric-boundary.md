# ADR-0002: separate fixed-point SSM simulation boundary

Status: accepted for the simulation prototype; production quantization deferred.

The user requested generic SSM language inference, simulation first, automatic
Quartus compilation and physical deployment later. Preserve the existing dense
INT8/INT32 arithmetic without reinterpretation.

The new generic operator engine uses a separately versioned Q2.14 INT16
contract and signed INT64 reductions. Integer reference, compiler and RTL share
the documented *contract*, not implementation code. Every accepted configuration
must match all reference logits and recurrent states. Saturation is observable;
language-quality retention is a separate held-out evaluation gate.

This reduces the first end-to-end validation surface but does not fulfill the
production INT4/INT8 group-quantization proposal. The prototype's wide divisions,
square root and multi-read-port memories can fit poorly; only actual fitter and
timing evidence can establish suitability for a board.

SSM operator/interface rules and exact rounding are in `docs/ssm-platform.md`.
Quartus builds use virtual pins, cannot be deployed, and never program hardware.
Missing tool or report evidence fails closed. Physical pinout, transport and
external-memory behavior remain later milestones.

Yosys structural gates disable SAT-based resource sharing for this engine to
bound verification cost at 16 lanes; they still assert no unresolved processes,
latches or logic loops. Lookup addresses are registered so dependent RAM reads
do not introduce the feedback path exposed by the initial synthesis test.
