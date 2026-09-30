"""Transport and benchmark harness for the F2 card: DMA loopback first, then the command /
descriptor path, then LLM token steps.

Runs over any transport with the upstream board-driver interface: the software card
(`EmulatedTransport`, default), the Verilator model (`F2SimTransport`), or -- only when
explicitly enabled and never run by us -- the real card (`F2BarTransport`).

Every result carries its provenance. Emulated timings are host memcpy / ISA-machine speed;
simulated timings are the simulator's wall clock and simulated cycles; only a run on a real
card is a measurement, and no such run exists.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from malleable.llm import upstream  # noqa: F401

from opentpu import isa as I
from opentpu.host import regs as R
from opentpu.host.board import Board

from . import placement as P
from . import transport as T

PROVENANCE = {
    "emulate": "emulated: software card (ISA machine, no cycle model); timings are host memcpy and "
               "interpreter speed, not a measurement of any hardware",
    "sim": "simulated: Verilator model of the F2 core with an HBM model; wall times are the "
           "simulator's, cycles are simulated core cycles, none is a hardware measurement",
    "hardware": "measured on a real F2 card (this path has never been run)",
}


def _pattern(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8)


def loopback(t, sizes=(64, 4096, 1 << 16, 1 << 20), channels=(0, 1), seed: int = 1,
             unaligned: bool = True, repeat: int = 1) -> dict:
    """Write a pseudo-random pattern into each core channel through the memory window, read it
    back and compare. Reports integrity and throughput per size.

    Over the Verilator model (a transport with `dma_check`) the pattern goes through the
    testbench's real PCIS path (`DW` / `DR`), sizes must be multiples of 64 bytes, and the time
    is simulated main-clock cycles; over every other transport it is `mem_write` / `mem_read`
    with host wall-clock times."""
    rows, ok = [], True
    if hasattr(t, "dma_check"):
        for ch in channels:
            for n in sizes:
                n = max(64, n // 64 * 64)
                lines = t.dma_check(ch, 0x10000, n, seed + 31 * ch + n)
                good = bool(lines) and all(x.startswith("DMA_OK") for x in lines)
                ok &= good
                cycles = t.dma_time(ch, 0x10000, n)
                rows.append({"channel": ch, "bytes": n, "offset": 0x10000, "verified": good,
                             "path": "PCIS write+read through the RTL (simulated)",
                             "read_main_clock_cycles_simulated": cycles,
                             "read_bytes_per_main_cycle_simulated": n / max(cycles, 1)})
        return {"test": "dma_loopback", "passed": ok, "rows": rows, "edge_cases": {}}
    for ch in channels:
        for n in sizes:
            for k in range(repeat):
                off = 0x10000 + (k * (n + 4096) if repeat > 1 else 0)
                data = _pattern(n, seed + 31 * ch + n + k)
                t0 = time.perf_counter()
                t.mem_write(ch, off, data)
                t1 = time.perf_counter()
                back = t.mem_read(ch, off, n)
                t2 = time.perf_counter()
                good = bool(np.array_equal(back, data))
                ok &= good
                rows.append({"channel": ch, "bytes": n, "offset": off, "verified": good,
                             "write_seconds": t1 - t0, "read_seconds": t2 - t1,
                             "write_MBps": n / max(t1 - t0, 1e-9) / 1e6,
                             "read_MBps": n / max(t2 - t1, 1e-9) / 1e6})
    edge = {}
    if unaligned:
        # sub-beat and odd-alignment accesses (the RTL passes byte strobes through)
        for ch in channels:
            data = _pattern(197, seed + 7 + ch)
            t.mem_write(ch, 0x20003, data)
            edge[f"channel{ch}_unaligned_197B_at_+3"] = bool(np.array_equal(t.mem_read(ch, 0x20003, 197), data))
        ok &= all(edge.values())
    return {"test": "dma_loopback", "passed": ok, "rows": rows, "edge_cases": edge}


def descriptor(t, cfg, seed: int = 2) -> dict:
    """The command / descriptor path, exactly the driver's: identity and capability registers,
    scratch registers, a program copied into DRAM and loaded, RUN, poll HALTED, counters,
    result bytes. The program copies 1 KiB DRAM -> TMEM -> DRAM."""
    board = Board(t, lock=False)
    info = board.info()
    checks: dict[str, bool] = {}
    checks["board_id"] = t.reg_read(R.R_ID) == R.ID_OTPU
    checks["configuration_matches"] = (info["D"], info["MCOLS"], info["LANES"]) == (cfg.D, cfg.MCOLS, cfg.LANES)
    checks["memory_ready"] = bool(info["calibrated"])
    f2 = t.reg_read_many([T.F2_ID, T.F2_VERSION, T.F2_CAPS, T.F2_STATUS])
    checks["f2_id"] = f2[0] == T.F2_ID_VALUE
    checks["hbm_ready"] = bool(f2[3] & T.F2_ST_HBM_READY)
    t.reg_write(T.F2_SCRATCH, 0xA5A5_1234)
    checks["f2_scratch"] = t.reg_read(T.F2_SCRATCH) == 0xA5A5_1234
    t.reg_write(R.R_SCRATCH, 0x0F2F_0001)
    checks["board_scratch"] = t.reg_read(R.R_SCRATCH) == 0x0F2F_0001
    checks["unmapped_reads_deadbeef"] = t.reg_read(0x2000) == T.UNMAPPED

    src, dst, words, prog_at = 0x4000, 0x8000, 256, 0x10000
    payload = _pattern(4 * words, seed)
    board.write(src, payload)
    board.write(dst, np.zeros(4 * words, np.uint8))
    prog = [I.ld(src, 0, words), I.st(dst, 0, words), I.halt()]
    t0 = time.perf_counter()
    board.load_program(prog_at, I.assemble(prog))
    st = board.run(timeout=600)
    wall = time.perf_counter() - t0
    result = board.read(dst, 4 * words)
    checks["copy_result"] = bool(np.array_equal(result, payload))
    checks["instructions_retired"] = st["instructions"][0] == len(prog)
    checks["halted_without_error"] = not st["status"] & (R.ST_ERROR | R.ST_AXI_ERR)
    checks["source_untouched"] = bool(np.array_equal(board.read(src, 4 * words), payload))
    err = t.reg_read_many([T.F2_CORE_ERRS, T.F2_PCIS_ERRS])
    checks["no_memory_errors"] = err == [0, 0]
    return {"test": "descriptor_path", "passed": all(checks.values()), "checks": checks,
            "info": {k: info[k] for k in ("D", "MCOLS", "LANES", "regmap", "core_khz", "build_id")},
            "run": {"cycles_reported": st["cycles"], "instructions": st["instructions"][0],
                    "wall_seconds": wall}}


def isa_checked_backend(cfg, images, transport):
    """BoardBackend over `transport` that also runs every program on the independent ISA machine
    and requires the card's DRAM image to match it after every step."""
    from opentpu.host.board import BoardBackend
    from opentpu.llm.qwen3 import IsaBackend

    class Backend(BoardBackend):
        def __init__(self):
            super().__init__(cfg, images, transport=transport, status=False)
            self.ref = IsaBackend(cfg, images)
            self.n = len(np.asarray(images[0]))
            self.step_seconds: list[float] = []
            self.mismatches: list[int] = []

        def write(self, s, addr, data):
            self.ref.write(s, addr, data)
            super().write(s, addr, data)

        # The Engine drives BoardBackend through start() / wait(); hook those two so every
        # step is checked whichever way the caller starts a program.
        def start(self, programs):
            self._t0 = time.perf_counter()
            self._programs = programs
            super().start(programs)

        def wait(self):
            st = super().wait()
            self.step_seconds.append(time.perf_counter() - self._t0)
            if isinstance(self._programs, np.ndarray):
                raise TypeError("the ISA reference needs instruction lists, not assembled words")
            self.ref.run(self._programs)
            step = len(self.step_seconds) - 1
            if not np.array_equal(self.board.read(0, self.n), self.ref.machine.slices[0].dram[:self.n]):
                self.mismatches.append(step)
                raise ValueError(f"card DRAM differs from the ISA machine after step {step}")
            return st

    return Backend()


def steps(make_transport, personality: str = "balanced", n_steps: int = 20, seed: int = 9) -> dict:
    """Run the tiny synthetic model for `n_steps` tokens through Engine -> BoardBackend ->
    transport, requiring the DRAM image to equal the ISA machine's after every step."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from llm_fixture import tiny
    from malleable.llm.records import PERSONALITIES
    from opentpu.llm.qwen3 import Engine
    _, weights, spec = tiny()
    cfg = PERSONALITIES[personality].config(spec, 128)
    holder = {}

    def factory(c, images):
        f2cfg, transport = make_transport(c, images)
        holder["t"] = transport
        holder["b"] = isa_checked_backend(f2cfg, images, transport)
        return holder["b"]

    engine = Engine(spec, weights, cap=128, cfg=cfg, rows=1, pipeline=False, backend=factory)
    tokens = np.random.default_rng(seed).integers(0, 128, n_steps).tolist()
    logits_sha = []
    import hashlib
    for tok in tokens:
        out = engine.step(int(tok))
        logits_sha.append(hashlib.sha256(np.asarray(out, np.float32).tobytes()).hexdigest())
    b = holder["b"]
    checked = len(b.step_seconds)
    return {"test": "llm_steps", "passed": not b.mismatches and checked == n_steps, "steps": n_steps,
            "steps_checked_against_isa": checked, "personality": personality,
            "mean_step_seconds": float(np.mean(b.step_seconds)), "logits_sha256_first": logits_sha[0],
            "logits_sha256_last": logits_sha[-1], "input_tokens_seed": seed}


def report(mode: str, results: list[dict]) -> dict:
    return {"schema": "f2-bench-v1", "mode": mode, "provenance": PROVENANCE[mode],
            "measured_on_hardware": mode == "hardware", "python": platform.python_version(),
            "all_passed": all(r["passed"] for r in results), "results": results}


def write_report(path: str | Path, rep: dict) -> None:
    Path(path).write_text(json.dumps(rep, indent=2, default=str) + "\n")
