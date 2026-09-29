"""Replay the recorded tiny 100-token test through the F2 shell model and check it.

Every token's programs run twice, from identical DRAM state:

  reference   the recorded path: CheckedRtlBackend (tb_top, generic two-channel AXI memory)
              checked against the independent ISA machine, as in tools/export_tiny_rtl_evidence.py
  F2 path     the unmodified upstream host driver (Board / BoardBackend) over F2SimTransport:
              register access through the OCL crossing, program load and image traffic through
              the wrapper, HBM adapter and HBM model

The gate is byte equality after every step: the F2 path's DRAM image and TMEM must equal the
ISA machine's. Trace hashes are secondary:

  ref_trace_sha256   sha256 of the reference run's whole stdout, the "recorded" kind
                     (tools/export_tiny_rtl_evidence.py). It is NOT reproducible: the file
                     ends with Verilator's own report lines ("- Verilator: $finish ...
                     walltime 0.158 s; speed ...") whose wall-clock values change on every run
                     of the same binary. Two runs on one machine give different values, so no
                     re-run (ours or Zephyrus') can match a recorded raw hash.
  ref_trace_normalized_sha256
                     NEW hash defined here: the same text without Verilator's report lines
                     (lines starting with "- "). Reproducible run to run; it still contains the
                     tb_top cycle stamps and the RESULT/SLICE/AXI lines, so it can only be
                     compared with another tb_top run of the same configuration, never with an
                     F2 shell run. `python -m malleable.f2.tracehash` computes it from saved
                     trace.txt files (e.g. on Zephyrus, from build/.../step-*/trace.txt).
  projection         NEW hash: sha256 over the dispatch events (pc, op, w1, w2, w3) of the trace,
                     cycle stamps and slot numbers removed. Compared between the reference run
                     and the F2 run; says nothing about any recorded hash.

`recorded_hashes` is a JSON list of per-step hashes, or {"kind": "raw"|"normalized",
"hashes": [...]}. A plain list is treated as raw. Raw values are expected NOT to match (see
above); the summary says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from malleable.llm import upstream
from malleable.llm.backend import CheckedRtlBackend
from malleable.llm.records import PERSONALITIES, GenerationWorkload

from opentpu.host.board import BoardBackend, join
from .sim import F2SimTransport

_D = re.compile(r"^T0 D c=\d+ s=\d+ pc=(\d+) op=([0-9a-f]+) w1=([0-9a-f]+) w2=([0-9a-f]+) w3=([0-9a-f]+)")


def dispatch_events(lines) -> list[tuple]:
    """(pc, op, w1, w2, w3) of every dispatch line, in trace order."""
    out = []
    for line in lines:
        m = _D.match(line)
        if m:
            out.append(m.groups())
    return out


def normalized_trace_text(text: str) -> str:
    """A tb_top trace without Verilator's own report lines (those start with "- " and carry
    wall-clock time, build paths and memory figures)."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("- ")) + "\n"


def normalized_trace_sha256(text: str) -> str:
    return hashlib.sha256(normalized_trace_text(text).encode()).hexdigest()


def projection_hash(lines) -> str:
    """Timing-independent hash of a trace: the dispatch events without cycle stamps."""
    text = "\n".join(" ".join(e) for e in dispatch_events(lines))
    return hashlib.sha256(text.encode()).hexdigest()


def sim_config_for(cfg, image_bytes: int):
    """cfg with the DRAM cut to the model's needs (power of two, at least 4 MiB), as
    opentpu.host.board.sim_config does for the board model."""
    need = -(-image_bytes // 4096) * 4096 + 4 * cfg.IMEM_WORDS
    return replace(cfg, DRAM_BYTES=1 << max(22, (need - 1).bit_length()))


class DualBackend:
    """Engine backend: reference path and F2 path in lock step; the F2 path is checked."""

    def __init__(self, cfg, images, workload, root, personality, dma_steps=(), hbm_stall=20,
                 hbm_lat=20, seed=1, plusargs=None, clocks=None):
        root = Path(root)
        self.ref = CheckedRtlBackend(cfg, images, workload, root / "reference")
        f2cfg = sim_config_for(cfg, len(np.asarray(images[0])))
        self.transport = F2SimTransport(
            f2cfg, uarch=PERSONALITIES[personality].uarch, hbm_stall=hbm_stall, hbm_lat=hbm_lat,
            seed=seed, plusargs=["+trace", *(plusargs or [])], **(clocks or {}))
        self.f2 = BoardBackend(f2cfg, images, transport=self.transport, status=False)
        self.image_bytes = len(np.asarray(images[0]))
        self.dma_steps = set(dma_steps)
        self.step = 0
        self.records: list[dict] = []

    def write(self, s, addr, data):
        self.ref.write(s, addr, data)
        self.f2.write(s, addr, data)

    def read(self, s, addr, nbytes):
        return self.ref.read(s, addr, nbytes)

    def run(self, programs):
        ref = self.ref.run(programs)                 # RTL == ISA, else it raises
        step = self.step
        self.transport.dma = "dma" if step in self.dma_steps else "backdoor"
        started = time.monotonic()
        f2 = self.f2.run(programs)
        wall = time.monotonic() - started
        t = self.transport
        state = self.ref.reference.machine.slices[0]
        n = self.image_bytes
        f2_dram = join([t.ch[0], t.ch[1]])[:n]
        dram_equal = bool(np.array_equal(f2_dram, state.dram[:n]))
        tmem_equal = bool(t.last_run_tmem is not None and np.array_equal(t.last_run_tmem, state.tmem))
        first_bad = None
        if not dram_equal:
            first_bad = int(np.flatnonzero(f2_dram != state.dram[:n])[0])
        ref_text = (Path(self.ref.root) / f"step-{step:05d}" / "trace.txt").read_text()
        ref_proj, f2_proj = projection_hash(ref_text.splitlines()), projection_hash(t.last_run_trace)
        rec = {
            "step": step, "dma_path": step in self.dma_steps,
            "dram_equal": dram_equal, "tmem_equal": tmem_equal, "first_dram_mismatch": first_bad,
            "instructions_equal": f2["instructions"][0] == ref["instructions"][0],
            "ref_instructions": ref["instructions"][0], "f2_instructions": f2["instructions"][0],
            "ref_trace_sha256": ref["trace_sha256"],
            "ref_trace_normalized_sha256": normalized_trace_sha256(ref_text), "ref_cycles": ref["cycles"],
            "ref_projection": ref_proj, "f2_projection": f2_proj,
            "projection_equal": ref_proj == f2_proj,
            "f2_cycles_simulated_core": f2["cycles"], "f2_status": f2["status"],
            "f2_host_wall_seconds": round(wall, 3), "dispatch_events": len(dispatch_events(t.last_run_trace)),
        }
        self.records.append(rec)
        self.step += 1
        if not (dram_equal and tmem_equal):
            raise ValueError(f"F2 path mismatch at step {step}: dram_equal={dram_equal} "
                             f"tmem_equal={tmem_equal} first_dram_mismatch={first_bad}")
        return ref

    def close(self):
        self.f2.close()


def run_replay(out: Path, steps: int = 100, seed: int = 9, personality: str = "balanced",
               dma_steps=(0, 1, 50, 99), hbm_stall: int = 20, hbm_lat: int = 20,
               recorded_hashes: list[str] | None = None, clocks: dict | None = None,
               plusargs=None, progress=print) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from llm_fixture import tiny
    from opentpu.llm.qwen3 import Engine
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new output directory; evidence is immutable")
    out.mkdir(parents=True)
    _, weights, spec = tiny()
    workload = GenerationWorkload("synthetic correctness tape", context=128, seed=seed,
                                  personality=personality)
    cfg = PERSONALITIES[personality].config(spec, 128)
    holder: dict = {}

    def factory(c, images):
        b = DualBackend(c, images, workload, out, personality, dma_steps=dma_steps,
                        hbm_stall=hbm_stall, hbm_lat=hbm_lat, seed=seed, plusargs=plusargs,
                        clocks=clocks)
        holder["backend"] = b
        return b

    engine = Engine(spec, weights, cap=128, cfg=cfg, rows=1, pipeline=False, backend=factory)
    tokens = np.random.default_rng(seed).integers(0, 128, steps).tolist()
    for i, token in enumerate(tokens):
        engine.step(int(token))
        if (i + 1) % 10 == 0:
            progress(json.dumps({"completed_steps": i + 1, "total": steps}))
    b: DualBackend = holder["backend"]
    recs = b.records
    summary = {
        "steps": len(recs),
        "all_dram_equal": all(r["dram_equal"] for r in recs),
        "all_tmem_equal": all(r["tmem_equal"] for r in recs),
        "all_instructions_equal": all(r["instructions_equal"] for r in recs),
        "projection_matches": sum(r["projection_equal"] for r in recs),
        "dma_path_steps": [r["step"] for r in recs if r["dma_path"]],
        "ref_total_cycles": sum(r["ref_cycles"] for r in recs),
        "f2_total_simulated_core_cycles": sum(r["f2_cycles_simulated_core"] for r in recs),
        "f2_flushes": b.transport.flushes, "f2_runs": b.transport.runs,
    }
    if recorded_hashes is not None:
        kind, want = ("raw", recorded_hashes) if isinstance(recorded_hashes, list) else \
            (recorded_hashes.get("kind", "raw"), recorded_hashes["hashes"])
        key = "ref_trace_normalized_sha256" if kind == "normalized" else "ref_trace_sha256"
        got = [r[key] for r in recs]
        summary["recorded_hash_comparison"] = {
            "kind": kind, "compared": min(len(got), len(want)),
            "equal": sum(a == c for a, c in zip(got, want)),
            "count_matches": len(got) == len(want),
            "note": ("raw hashes include Verilator's wall-clock report lines and are not reproducible "
                     "run to run: a mismatch says nothing about the RTL" if kind == "raw" else
                     "normalized hashes exclude Verilator's report lines"),
        }
    evidence = {
        "schema": "f2-replay-v1", "provenance": "simulation: Verilator tb_f2_shell with an HBM model; "
        "reference: tb_top; not a hardware or AWS measurement",
        "personality": personality, "seed": seed, "input_tokens": tokens,
        "memory": {"hbm_stall_percent": hbm_stall, "hbm_latency_cycles": hbm_lat,
                   "clocks_ns": clocks or {"core": 8.0, "main": 4.0, "hbm": 2.2}},
        "shell_params": b.transport.params_full,
        "hash_semantics": {
            "ref_trace_sha256": "sha256 of the tb_top run's whole stdout (recorded style); includes Verilator's "
                                "wall-clock report lines, so it is not reproducible run to run and not comparable to the F2 run",
            "ref_trace_normalized_sha256": "NEW: the same text without Verilator's report lines; reproducible; only "
                                           "comparable with another tb_top run",
            "projection": "NEW hash: dispatch events (pc op w1 w2 w3) without cycle stamps",
        },
        "summary": summary, "steps": recs,
        "toolchain": {"upstream": upstream.REVISION, "python": platform.python_version()},
    }
    (out / "f2-replay.json").write_text(json.dumps(evidence, indent=2) + "\n")
    b.close()
    return evidence


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", required=True, help="new evidence directory")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--personality", default="balanced", choices=list(PERSONALITIES))
    p.add_argument("--dma-steps", default="0,1,50,99",
                   help="steps whose image traffic uses the real PCIS path (others: backdoor)")
    p.add_argument("--hbm-stall", type=int, default=20)
    p.add_argument("--hbm-lat", type=int, default=20)
    p.add_argument("--recorded-hashes", help='JSON list of per-step hashes, or {"kind": "raw"|"normalized", "hashes": [...]}')
    a = p.parse_args(argv)
    dma = [int(x) for x in a.dma_steps.split(",") if x != ""]
    rec = json.loads(Path(a.recorded_hashes).read_text()) if a.recorded_hashes else None
    ev = run_replay(Path(a.output), a.steps, personality=a.personality, dma_steps=dma,
                    hbm_stall=a.hbm_stall, hbm_lat=a.hbm_lat, recorded_hashes=rec)
    print(json.dumps(ev["summary"], indent=2))
    s = ev["summary"]
    ok = s["all_dram_equal"] and s["all_tmem_equal"] and s["all_instructions_equal"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
