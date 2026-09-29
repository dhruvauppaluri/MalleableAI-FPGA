"""Verilator model of the F2 custom-logic core as a transport for the upstream board driver.

`F2SimTransport` speaks the same protocol as opentpu.host.board.SimTransport (the rehearsal
transport for the original board), but runs f2/sim/tb_f2_shell.sv: the frozen board block
behind the F2 wrapper (OCL register crossing, PCIS window, HBM adapter) in front of an HBM
model. `Board` and `BoardBackend` therefore run unchanged.

Everything here is simulation. The HBM model is not the AMD HBM IP; cycle counts are
simulated core cycles and say nothing about the wall-clock speed of a card.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from malleable.llm import upstream  # noqa: F401  (puts third_party/opentpu on sys.path)

from opentpu import rtlsim
from opentpu.host.board import SimTransport
from opentpu.host import regs as R

ROOT = Path(__file__).resolve().parents[2]
F2_RTL = ROOT / "f2" / "rtl"
F2_SIM = ROOT / "f2" / "sim"
RTL_FILES = ["f2_fifo.sv", "f2_cdc.sv", "f2_ocl.sv", "f2_hbm_router.sv", "f2_hbm_pc_bridge.sv",
             "f2_hbm_adapter.sv", "cl_otpu_core.sv"]
SIM_FILES = ["f2_hbm_model.sv", "tb_f2_shell.sv"]


def shell_sources() -> list[Path]:
    """The frozen upstream sources (everything but its simulation top), then ours."""
    srcs = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES if not s.endswith("otpu_top.sv")]
    board = rtlsim.RTL / "boards" / "ypcb-00338"
    srcs += [board / "otpu_ctrl.sv", board / "otpu_trace.sv", board / "otpu_board.sv"]
    srcs += [F2_RTL / f for f in RTL_FILES] + [F2_SIM / f for f in SIM_FILES]
    return srcs


def shell_params(cfg, uarch: dict | None = None, ch_bytes: int | None = None, pcs: int = 2) -> dict:
    """tb_f2_shell parameters that make the wrapped board block match `cfg` (an
    opentpu.isasim.Config) and `uarch` (timing-only knobs: WIN, RPB, WPB, FIFO_DEPTH, ...)."""
    u = dict(uarch or {})
    p = {"CH_BYTES": ch_bytes or cfg.DRAM_BYTES // 2, "PCS": pcs, "MCOLS": cfg.MCOLS,
         "LANES": cfg.LANES, "IMEM_WORDS": cfg.IMEM_WORDS, "TMEM_WORDS": cfg.TMEM_WORDS,
         "ACT_BLOCKS": cfg.ACT_BLOCKS, "VPU_CL": u.get("VPU_CL", cfg.LANES // 4 if cfg.LANES >= 8 else 1),
         "ULANES": u.get("ULANES", cfg.LANES)}
    for k in ("WIN", "RPB", "WPB", "FIFO_DEPTH", "MXU_IMPL"):
        if k in u:
            p[k] = u[k]
    return p


def build(params: dict) -> Path:
    return rtlsim.build("tb_f2_shell", shell_sources(), params)


ADAPTER_FILES = ["f2_fifo.sv", "f2_hbm_router.sv", "f2_hbm_pc_bridge.sv", "f2_hbm_adapter.sv"]


def build_adapter_bench(pcs: int = 2, pc_aw: int = 21) -> Path:
    """Verilator build of f2/sim/tb_f2_adapter.sv (the adapter alone, three unrelated clocks)."""
    srcs = [F2_RTL / f for f in ADAPTER_FILES] + [F2_SIM / "f2_hbm_model.sv", F2_SIM / "tb_f2_adapter.sv"]
    return rtlsim.build("tb_f2_adapter", srcs, {"PCS": pcs, "PC_AW": pc_aw})


def run_adapter_bench(pcs: int = 2, pc_aw: int = 21, plusargs=()) -> str:
    exe = build_adapter_bench(pcs, pc_aw)
    r = subprocess.run([str(exe), *plusargs], capture_output=True, text=True, timeout=1800)
    return r.stdout + r.stderr


class F2SimTransport(SimTransport):
    """SimTransport against tb_f2_shell.

    dma: "backdoor" (default) places the channel images in the HBM model directly and reads
    them back directly; "dma" moves them through the real PCIS path (slower). `dma` can be
    changed between flushes. After a flush that started a program (a write of CTRL_RUN):
    `last_run_tmem` is the slice's TMEM at the end, `last_run_trace` the trace lines."""

    ecc = False

    def __init__(self, cfg, uarch: dict | None = None, ch_bytes: int | None = None, pcs: int = 2,
                 hbm_stall: int = 20, hbm_lat: int = 20, seed: int = 1, plusargs=None,
                 dma: str = "backdoor", core_ns: float | None = None, main_ns: float | None = None,
                 hbm_ns: float | None = None, timeout: float = 3600.0):
        self.params_full = shell_params(cfg, uarch, ch_bytes, pcs)
        super().__init__(ch_bytes=self.params_full["CH_BYTES"], stall=hbm_stall, seed=seed,
                         params={}, plusargs=plusargs)
        self.cfg, self.pcs = cfg, pcs
        self.hbm_lat, self.dma = hbm_lat, dma
        self.clocks = {"core_ns": core_ns, "main_ns": main_ns, "hbm_ns": hbm_ns}
        self.timeout = timeout
        self.tmem_words = cfg.TMEM_WORDS
        self.last_run_tmem: np.ndarray | None = None
        self.last_run_trace: list[str] = []
        self.dma_lines: list[str] = []          # DMA_OK / DMA_FAIL / DMA_TIME lines of the last flush
        self.flushes = 0
        self.runs = 0
        self.stats_line = ""

    def dma_check(self, ch: int, off: int, nbytes: int, seed: int = 1) -> list[str]:
        """Write and read back a pseudo-random pattern through PCIS (`DW` / `DR` script ops).
        Returns the testbench's DMA_OK / DMA_FAIL lines."""
        a = (ch << 31) | off
        self.script += [f"DW {a:x} {nbytes:x} {seed:x}", f"DR {a:x} {nbytes:x} {seed:x}"]
        self.flush()
        return list(self.dma_lines)

    def dma_time(self, ch: int, off: int, nbytes: int) -> int:
        """Simulated main-clock cycles a PCIS read of `nbytes` takes (a model number)."""
        self.script.append(f"DT {(ch << 31) | off:x} {nbytes:x}")
        self.flush()
        for line in self.dma_lines:
            if line.startswith("DMA_TIME"):
                return int(line.split("main_cycles=")[1])
        raise RuntimeError("no DMA_TIME line")

    def _args(self, d: Path) -> list[str]:
        a = [f"+dir={d}", f"+hbm_stall={self.stall}", f"+hbm_seed={self.seed}",
             f"+hbm_lat={self.hbm_lat}", *self.plusargs]
        if self.dma == "dma":
            a += ["+dma_load", "+dma_dump"]
        for k, v in self.clocks.items():
            if v:
                a.append(f"+{k}={v}")
        return a

    def flush(self) -> None:
        if not self.script:
            return
        exe = build(self.params_full)
        ran = any(line == f"W {R.R_CTRL:x} {R.CTRL_RUN:x}" for line in self.script)
        with tempfile.TemporaryDirectory(prefix="f2_shell_") as d:
            d = Path(d)
            for c in (0, 1):
                self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
            (d / "host.txt").write_text("\n".join(self.script) + "\n")
            self.script, self.nreads = [], 0
            r = subprocess.run([str(exe), *self._args(d)], capture_output=True, text=True,
                               timeout=self.timeout)
            out = self.out = r.stdout + r.stderr
            self.flushes += 1
            if "DONE" not in out:
                raise RuntimeError(f"F2 shell simulation failed:\n{out[-3000:]}")
            self.results = []
            self.dma_lines = []
            for line in out.splitlines():
                if line.startswith("REG "):
                    _, a, v = line.split()
                    self.results.append(int(v, 16))
                    self.regs_seen[int(a, 16)] = int(v, 16)
                elif line.startswith("DONE"):
                    self.cycles += int(line.split("=")[1])
                elif line.startswith("DMA_"):
                    self.dma_lines.append(line)
                elif line.startswith("F2_STATS"):
                    self.stats_line = line
            for c in (0, 1):
                self.ch[c] = np.fromfile(d / f"ch{c}_out.bin", np.uint8)
            if ran:
                self.runs += 1
                self.last_run_trace = [ln for ln in out.splitlines() if ln.startswith("T0 ")]
                words = [int(t, 16) for t in (d / "tmem_0.hex").read_text().split()]
                if len(words) != self.tmem_words:
                    raise RuntimeError(f"tmem_0.hex has {len(words)} words, expected {self.tmem_words}")
                self.last_run_tmem = np.asarray(words, np.uint32)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="F2 simulation helpers")
    p.add_argument("--sources", action="store_true", help="print the shell testbench source list")
    p.add_argument("--rtl-only", action="store_true", help="with --sources: the synthesizable sources only")
    a = p.parse_args(argv)
    if a.sources:
        srcs = shell_sources()
        if a.rtl_only:
            srcs = [s for s in srcs if F2_SIM not in s.parents]
        print(" ".join(str(s) for s in srcs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
