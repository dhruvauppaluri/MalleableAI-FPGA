# Model-aware dense accelerator experiments

The host keeps network weights unchanged. It compares supported execution
settings, includes switching costs, records evidence, and can train a small
Double DQN controller. Nothing here trains the neural network being accelerated.
The first backend is Icarus RTL simulation, not an FPGA emulator with physical
timing or power accuracy.

## Quick start

Run from the repository root with Python 3.10+, Icarus, Verilator, Yosys and Make.
No Python runtime dependencies are required.

```sh
make verify
python3 -m malleable analyze --size light
python3 -m malleable benchmark --size heavy --lanes 8 --active-lanes 7 --requests 4
python3 -m malleable optimize --size heavy --lanes 1 --active-lanes 1 --horizon 100 --switch-cycles 10000
python3 -m malleable optimize --size light --switch-cycles 10000 --execute --budget 8
python3 -m malleable suite --seed 0
python3 -m malleable suite --seed 0 --exhaustive
```

`suite` covers light 7→5→3, medium 32→32→16, and heavy four 64×64
layers. It uses isolated, burst, and sustained traffic at 0.5×/1×/2× the
one-lane baseline service interval. The exhaustive option evaluates all 60 legal
lane/reuse/dispatch combinations; otherwise it compares four full-lane settings.
The same seed gives the same inputs for all candidates. All results, including
failed candidates, are stored under `build/research` by default.

## Data contract and architecture

Version-1 dataclasses in `malleable/records.py` define the public JSON records:

- `ModelArtifact`: `schema_version: 1`, and one to four `layers`. Each layer
  contains output-major `weights` (INT8 rows), `biases` (INT32), `multipliers`
  (1…2³¹−1), `shifts` (0…62), and boolean `relu`. Parameter arrays have one
  element per output. Adjacent dimensions must match; each dimension is 1…64.
- `WorkloadSpec`: request count, arrival pattern, interval in scenario cycles,
  remaining horizon, optional per-request deadline, and deterministic seed.
- `HardwareProfile`: numeric contract, operators, capacity, personalities,
  runtime controls, assumed clock, explicit switching cost, and optional power
  calibration and monetary rate. Unsupported capability declarations are rejected.
- `ExecutionConfig`: compiled lanes (1/2/4/8), active lanes (1…compiled lanes),
  residency reuse, and FIFO/group dispatch.
- `ExperimentResult`: identities, config, metrics, validity/failure, seed,
  input/trace/build/tool provenance, previous configuration and policy version.
- `PolicyDecision`: keep/tune-current/switch-personality, selected settings,
  objective cost, switching cost, uncertainty and explanation.

Use `--model artifact.json` instead of a generated fixture. Model identity hashes
all topology/weights/numeric settings; separate hashes expose changes to each.
Workload identity is independent. SQLite indexes records and experiments by
model/workload; canonical SHA-256-addressed JSON objects hold models, traces,
inputs, checkpoints and reports. Loads check content integrity.

The flow is artifact validation → legal candidates → cost prediction → decision
→ RTL/reference comparison → stored evidence → predictor/policy update.
Without `--execute`, `optimize` consumes stored RTL observations for that model, using a regularized
least-squares cycle predictor. It proposes a configuration; `benchmark` executes
it. With `--execute`, it validates up to `--budget` candidates including the
current setting and selects the best validated plan under the switching guard.
`suite` performs bounded exhaustive experiments. No command silently programs
a board or alters network weights.

## RTL programming and counters

Existing configuration kinds 0…5 retain their meaning. Kind **6**, address **0**,
sets active lanes. Reset restores the compiled lane count. Zero, excessive lanes,
and nonzero control addresses are rejected. The reduction advances by active
lanes; inactive operands are zero, including the final partial tile.

Model parameters remain resident between requests within one backend invocation
when reuse is enabled. Inputs are always reloaded. Each independent invocation
starts cold. `benchmark_sequence` dispatches repeated/alternating model requests,
loads on model changes, executes the resulting order in RTL, and returns outputs
keyed by original request ID. Deadline-bearing ready jobs use earliest deadline
first; grouping applies only without ready deadlines. It reports missed deadlines,
not a guarantee that an overloaded queue can meet them.

The top-level readable 64-bit outputs are:

- `cycles`: busy clocks for the most recently accepted job.
- `tiles`: accepted reduction tiles for that job.
- `useful_macs`: valid input-weight products, excluding padded lanes.
- `compute_cycles`: clocks accepting a tile (dot-product issue activity).
- `controller_cycles`: busy clocks minus tile-issue clocks, including waits.
- `configuration_writes`: accepted writes since reset, including invalid writes.
- `result_reads`: accepted valid result reads since reset.

Execution counters clear on accepted start and freeze while idle. All counters
clear on synchronous reset. Start during busy or simultaneous configuration is
reported as a configuration error; configuration writes require idle readiness.
These are observable simulation signals, not an invented board register bus.
Issue utilization must not be interpreted as physical DSP occupancy.

## Objectives and switching

Latency minimizes mean request latency with deadline-excess penalties. Throughput
minimizes elapsed cycles per completion. Energy minimizes joules per completion
with a hard latency constraint and is disabled without supplied power values.
For example `--profile energy --power-json '{"1":1,"2":1.2,"4":1.5,"8":2}'`
is only legitimate if those values came from a credible calibration; the tool
does not verify a user's calibration. All seconds depend on `clock_hz` and are
estimates, not measured FPGA timing.

`--switch-cycles` is the aggregate scenario cost of drain/program/warmup and any
required deployment compilation, excluding parameter reload which is counted
separately. Omit it to prohibit personality switching. Zero is explicitly marked
idealized. Build wall time is recorded separately and is never substituted for
FPGA programming time. There is no build-on-board action.

The selector compares the best current-personality plan against alternatives
over the remaining horizon. A switch must clear a 5% margin plus predictor
uncertainty. `decide(..., residence=0)` prevents switching within a residence
window; evaluation permits one decision per window. Same-personality tuning and
retaining settings are always available when legal. Model identity controls
residency, so alternating models incur reloads even without a personality change.

## Learning, evaluation and rollback

```sh
python3 -m malleable train --episodes 20 --seed 0 --switch-cycles 10000
# Use the returned checkpoint hash in the following commands:
python3 -m malleable evaluate --checkpoint HASH --switch-cycles 10000 --split validation --rtl
python3 -m malleable evaluate --checkpoint HASH --switch-cycles 10000 --split held-out --rtl
python3 -m malleable promote --checkpoint HASH --validation VALIDATION_HASH --heldout HELDOUT_HASH
python3 -m malleable rollback
```

Training implements a tanh neural Q-network, Double DQN online-action/target-value
selection, legal masks, bounded replay, Huber gradients and target refreshes.
The checkpoint includes both networks, replay, seed, model lineage, objective,
hardware scenario and predictor. Resume training with `train --checkpoint HASH`
and a new seed range. New checkpoints are candidates, never automatic deployments.

Six-window episodes include repeated and alternating model artifacts and short/
long horizons. Rewards are explicitly **prediction-based**, normalized negative
objective costs. Evaluation uses five independent seeds, rejects model overlap
with training, and compares fixed, heuristic, one-proposal budget-matched random,
and exhaustive-per-window baselines. Exhaustive is not an oracle for the optimal
future sequence. Negative RL results are retained in the report.

Validation and held-out reports must have disjoint model IDs. Promotion requires
both reports, passing bit-exact RTL evidence, and RL estimated cost no worse than
fixed and heuristic baselines in both predicted and RTL-cycle-calibrated costs.
Promotion is explicitly of a **simulation policy**;
it is not proof of physical acceleration. Prior deployment hashes support rollback.
`--rtl` evaluates all methods' selected configurations against RTL; it is slower
than predictor-only evaluation. RTL evidence checks numerical correctness, while
the sequential workload reward remains an estimate. Omit `--rtl` for exploratory
reports which cannot pass promotion. This system does not claim RL is superior.

## Evidence and limits

Metrics label measured RTL cycles/counts/bytes/bit-exactness separately from
estimated clock-based latency percentiles, throughput, scheduling, calibrated
energy, or supplied-rate monetary cost. Host simulation and compilation seconds
are separate provenance fields. Lane scaling uses the one-lane RTL baseline.
Without evidence, power, physical occupancy, external-memory bandwidth, and task
accuracy are unavailable. Counter attribution permits mixed diagnoses and does
not establish memory-bandwidth causality without controlled comparisons.

The software reference uses independent Python arbitrary-precision arithmetic,
explicit INT32 wrapping, 33-bit biased values, 65-bit multiplication semantics,
ties-away rounding, saturation and ReLU. Any overflow or output mismatch rejects
the candidate. Dense dimensions within current limits cannot naturally overflow
the dot accumulator; existing unit tests inject/cover overflow separately.

`make verify` runs legacy randomized tests, all 15 compiled/active lane pairs with
reuse on/off, host integration tests, Verilator lint and coarse structural Yosys
synthesis for all four personalities. Generated artifacts are ignored. Quartus
fitting, actual programming costs, physical power, external-memory timing, DMA
overlap, alternative precision/dataflow, CNNs, transformers and partial
reconfiguration are not supported in this backend.
