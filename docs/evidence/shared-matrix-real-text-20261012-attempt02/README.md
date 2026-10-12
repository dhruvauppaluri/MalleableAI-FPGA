# Shared-matrix real-text attempt 02: loading stopped

The operator used the newly instrumented runner with an exclusive attempt 02
directory and a fresh freeze. The runner reached `loading_teacher` but never
printed `teacher_loaded`. The five-minute external `timeout` command ended with
shell `Killed` and exit status 137; no training step or development scoring
was observed. The attempt is consumed and cannot be resumed or scored.

At collection time, `/sys/fs/cgroup/memory.events` reported `oom 0` and
`oom_kill 0`, with `memory.max` 59055800320 bytes. This excludes a recorded
container memory-limit kill; exit 137 alone does not distinguish the external
timeout's final SIGKILL from other host actions. The observed progress narrows
the expensive operation to loading the first model or transferring it to the
GPU, because the runner combines those operations in `load()`.

The freeze and claim were pasted from the notebook. Their matching SHA-256 is
recorded in `claim.json`; the corpus manifest is identical to attempt 01 and is
not duplicated here. No quality result, trained weights, or release held-out
evidence was produced. Automatic program switching remains gated.
