# ADR-0001: Build the accelerator from bit-accurate verified stages

**Status:** Accepted  
**Date:** 2026-09-19  
**Deciders:** Project contributors

## Context

The project must eventually explore multiple precisions, lane counts, buffer
sizes, tiling strategies, and FPGA platforms. Building an entire neural-network
engine before fixing its numeric behavior would make arithmetic and timing bugs
difficult to isolate. The existing Jane Street RTL work established a useful
pattern of small modules, paired self-checking tests, Quartus projects, and CI.

## Decision

Build the AI accelerator as independently verified stages. Begin with a signed
INT8 MAC, then a parameterized parallel dot product. Connect the independently
developed ML toolchain through a written numeric contract and versioned artifact
formats.

## Options considered

### Option A: Incremental verified stages

| Dimension | Assessment |
| --- | --- |
| Initial complexity | Low |
| Debuggability | High |
| Extensibility | High |
| Time to full network | Moderate |

**Pros:** Arithmetic disagreements are isolated early; blocks can be reused;
tests remain fast; platform-specific logic stays separate.

**Cons:** More interfaces and documentation must be maintained; the first demo
is smaller than an immediately integrated network.

### Option B: Build a complete fixed tiny-network accelerator first

| Dimension | Assessment |
| --- | --- |
| Initial complexity | High |
| Debuggability | Low |
| Extensibility | Moderate |
| Time to first end-to-end demo | Potentially short but risky |

**Pros:** Produces an end-to-end demonstration quickly if every assumption is
correct.

**Cons:** Numeric, scheduling, memory, and control errors become entangled;
fixed choices are harder to generalize into a malleable architecture.

## Consequences

- Every arithmetic stage needs an RTL test and an independently implemented
  software reference.
- Numeric behavior is treated as an API and changed deliberately.
- Board shells can evolve without changing the computation contract.
- Requantization, activation, buffering, and control arrive as later milestones.

## Action items

1. [x] Implement and verify the signed INT8 MAC.
2. [x] Implement and verify a parameterized parallel dot product.
3. [ ] Define and implement the independent golden-model toolchain.
4. [ ] Define the first model descriptor and exported artifact formats.
5. [ ] Implement tiled accumulation, bias, requantization, and ReLU.
6. [ ] Add a host-to-Cyclone-V transport and benchmark harness.
