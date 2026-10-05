# ADR-0009: Proposed fast program selection within one LLM personality

Status: Proposed. This document defines an experiment, not an enabled runtime feature.

## Problem

Compact, balanced, compute, and buffered are separate simulator builds. Their matrix columns, lanes, and FIFO depth are build parameters, so the existing selector cannot make them instantaneous modes. The current OpenTPU sequencer dispatch-window size is also a build parameter. Its current register map provides run and program-load control but no verified register for changing compute structure in a running design.

## Candidate fast action

For one frozen personality and numeric format, prepare two compiler-validated instruction schedules for the same model and inputs. At a quiescent request or token boundary, select a schedule by loading its program through the existing host program-load path. This avoids bitstream replacement. Both schedules must use the same model weights, memory layout, ISA, and numeric contract. The compiler must first demonstrate an actual alternate schedule; merely labeling identical programs as two actions is not an experiment.

The program loader writes a program image to DRAM and copies it into instruction memory while RUN is low. Reset or program loading may disturb transient state. Switching within a conversation is legal only after a test shows that model state is preserved or explicitly reconstructed, with its entire reload and re-prefill cost included. Until then, this candidate is limited to independent request boundaries.

## Validation before policy integration

1. Produce distinct program hashes and verify both against the independent ISA on a small deterministic model, including DRAM and TMEM state checks.
2. Prove the transition at the chosen boundary, including failure and rollback behavior. Measure host load time and simulator cycles separately. Never call host time physical FPGA time.
3. Compare matched fixed-tape workloads using the same personality, model, format, memory scenario, and inputs. Record program identity and transition cost with each result.
4. Admit an action to the predictor and controller only when its quality/correctness lineage and measured cost are complete. Apply the existing uncertainty, residence, and non-regression gates; retain the deterministic fallback.

The current cloud checkout has no model checkpoints or original quality store, and the published manifests do not contain personality-specific quality approvals. This proposal therefore does not assert that program selection is safe or faster yet.
