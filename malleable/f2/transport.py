"""Host transports for the F2 card, in the interface the upstream board driver expects
(opentpu.host.board: mem_write / mem_read / reg_write / reg_read / reg_read_many / poll).

  F2BarTransport       the real card through the PCIe BARs (UNTESTED on hardware; refuses to
                       run unless explicitly enabled; nothing here calls AWS APIs)
  EmulatedTransport    a software card: register map, F2 register block, memory window and
                       program execution on the bit-exact ISA machine. No timing model.

Both present the original board's view: register offsets 0x000.. are otpu_ctrl's map, offsets
0x1000.. the F2 block (f2/rtl/f2_ocl.sv), and DRAM channel c at BASE[c] + offset. Board and
BoardBackend run over either unchanged.
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np

from malleable.llm import upstream  # noqa: F401  (puts third_party/opentpu on sys.path)

from opentpu import isa as I
from opentpu.host import regs as R
from opentpu.host.board import XdmaTransport, join, split
from opentpu.host.fake import FakeTransport

from . import placement as P

HARDWARE_ENV = "MALLEABLE_F2_ENABLE_HARDWARE"

# F2 register block (byte offsets from the OCL base; f2/rtl/f2_ocl.sv)
F2_BASE = 0x1000
F2_ID, F2_VERSION, F2_CAPS, F2_STATUS, F2_CTRL = (F2_BASE + o for o in (0x00, 0x04, 0x08, 0x0C, 0x10))
F2_CORE_KHZ, F2_BUILD_ID, F2_HBM_BASE_LO, F2_HBM_BASE_HI = (F2_BASE + o for o in (0x14, 0x18, 0x1C, 0x20))
F2_CORE_ERRS, F2_PCIS_ERRS, F2_SCRATCH = (F2_BASE + o for o in (0x24, 0x28, 0x2C))
F2_ID_VALUE = 0x4632_4F54                      # "F2OT"
F2_ST_HBM_READY, F2_ST_CORE_ERR, F2_ST_PCIS_ERR = 1, 2, 4
UNMAPPED = 0xDEAD_BEEF

# AWS_CLK_GEN registers (aws-fpga hdk/docs/AWS_CLK_GEN_spec.md; the host reaches them through the
# BAR the AWS SDK calls PF1 / BAR4). Offsets and the group layout below are from that spec and
# from the register map the SDK documents; none of this has run against a card.
CLKGEN_REG_BASE = 0x58000
CLKGEN_ID, CLKGEN_VER, CLKGEN_BLD, CLKGEN_CLKS_AVAIL = (CLKGEN_REG_BASE + o for o in (0x0, 0x4, 0x8, 0xC))
CLKGEN_GRST, CLKGEN_SYSRST, CLKGEN_LOCK = (CLKGEN_REG_BASE + o for o in (0x10, 0x14, 0x20))
CLKGEN_ID_VALUE = 0x9048_1D0F
CLKGEN_BAR_BYTES = CLKGEN_REG_BASE + 0x1000
# (first CLKS_AVAIL bit, number of clocks, MMCM_LOCK bit) for groups A, B, C and HBM
CLKGEN_GROUPS = ((1, 3, 0), (4, 2, 4), (6, 2, 6), (8, 1, 8))
CLKGEN_TIMEOUT_S = 5.0          # the SDK waits 100 x 50 ms for lock


class HardwareDisabled(RuntimeError):
    """The real-card transport was not explicitly enabled."""


class AxiDecodeError(RuntimeError):
    """An access outside the memory the card's PCs hold (the RTL answers DECERR)."""


def f2_caps(pcs_per_ch: int, pc_aw: int, stripe_log2: int = 9) -> int:
    return (2 << 24) | (stripe_log2 << 16) | (pc_aw << 8) | pcs_per_ch


def clkgen_lock_mask(clks_avail: int) -> int:
    """MMCM_LOCK bits to wait for: one per clock group that has at least one clock enabled.

    This design enables group A and the HBM clock, so the mask is 0x101 (bits 0 and 8);
    a design using all four groups would wait for 0x151."""
    mask = 0
    for first, count, lock_bit in CLKGEN_GROUPS:
        if (clks_avail >> first) & ((1 << count) - 1):
            mask |= 1 << lock_bit
    return mask


def release_clock_resets(read, write, *, timeout: float = CLKGEN_TIMEOUT_S, poll_interval: float = 0.05,
                         sleep=time.sleep, now=time.monotonic) -> dict:
    """Release the CL resets held by AWS_CLK_GEN after AFI load.

    UNTESTED ON HARDWARE. With AWS_CLK_GEN in the design every CL reset except the main one is
    asserted at load and the MMCM outputs are stopped until they lock; releasing a reset before
    lock leaves the HBM IP without a clocked reset (AWS_CLK_GEN_spec.md). So: check the ID, work
    out which MMCMs are enabled, clear the global reset, wait for all their lock bits, and only
    then clear SYS_RST. `read(offset)` and `write(offset, value)` access the AWS_CLK_GEN BAR. If
    lock is not reached the resets are left asserted and TimeoutError is raised.
    """
    ident = read(CLKGEN_ID)
    if ident != CLKGEN_ID_VALUE:
        raise RuntimeError(f"AWS_CLK_GEN ID register reads {ident:#x}, expected {CLKGEN_ID_VALUE:#x}: "
                           "no AWS_CLK_GEN in this AFI, or the wrong BAR")
    avail = read(CLKGEN_CLKS_AVAIL)
    mask = clkgen_lock_mask(avail)
    if not mask:
        raise RuntimeError(f"AWS_CLK_GEN reports no enabled clocks (CLKS_AVAIL={avail:#x})")
    write(CLKGEN_GRST, 0)
    deadline = now() + timeout
    while True:
        lock = read(CLKGEN_LOCK)
        if lock & mask == mask:
            break
        if now() >= deadline:
            raise TimeoutError(f"AWS_CLK_GEN MMCMs not locked after {timeout:g} s "
                               f"(lock={lock:#x}, need {mask:#x}); resets left asserted")
        sleep(poll_interval)
    write(CLKGEN_SYSRST, 0)
    return {"clks_avail": avail, "lock_mask": mask, "lock": lock,
            "version": read(CLKGEN_VER), "build": read(CLKGEN_BLD)}


class F2BarTransport:
    """The real F2 card through the PCIe BARs of one FPGA slot.

    UNTESTED ON HARDWARE. Nothing in this class has run against a card; the tests exercise the
    register and window plumbing against ordinary files standing in for the BARs.

    OCL (AppPF BAR0, 64 MiB) carries the register maps; the HBM window sits in PCIS (AppPF
    BAR4, 128 GiB) at `hbm_offset` (HBM_BASE = 1 << 36, as in the AWS HDK examples). XDMA is
    unsupported on F2 today, so bulk data moves by CPU stores and loads through the mapped
    BAR4 (its throughput is unmeasured; docs/f2-bringup.md).

    Running it requires `enable=True` or MALLEABLE_F2_ENABLE_HARDWARE=1, an AFI loaded in
    the slot, and a BDF (or explicit BAR paths). Only one process should use a slot; the
    upstream Board takes an exclusive lock named after `devname`.
    """

    ecc = False                     # unverified: see the assumption ledger in docs/f2-bringup.md
    threaded = False                # no XDMA pipelining: CPU load/store through the mapping
    batched = False

    def __init__(self, bdf: str | None = None, *, enable: bool = False, bar0: str | Path | None = None,
                 bar4: str | Path | None = None, hbm_offset: int = P.HBM_BASE,
                 ocl_bytes: int = 0x2000, window_bytes: int = P.WINDOW_BYTES, check: bool = True,
                 ch_base: tuple[int, int] = P.CH_BASE, ch_bytes: int = P.CH_BYTES,
                 release_resets: bool = False, clkgen_bar: str | Path | None = None,
                 ready_timeout: float = 30.0):
        if not (enable or os.environ.get(HARDWARE_ENV) == "1"):
            raise HardwareDisabled(
                "F2BarTransport talks to real hardware and is disabled by default; pass enable=True "
                f"or set {HARDWARE_ENV}=1. It has never been run against a card.")
        if bar0 is None or bar4 is None:
            if not bdf:
                raise ValueError("give a PCI address (bdf, e.g. 0000:00:1d.0) or explicit BAR paths")
            base = Path("/sys/bus/pci/devices") / bdf
            bar0, bar4 = base / "resource0", base / "resource4"
        if clkgen_bar is None and bdf:
            # ASSUMPTION (unverified): AWS_CLK_GEN is on PF1, i.e. the same PCI device with function 1
            clkgen_bar = Path("/sys/bus/pci/devices") / (bdf[:-1] + "1") / "resource4"
        self.bdf = bdf
        self.clkgen_bar = clkgen_bar
        self.devname = "f2-" + (bdf or Path(str(bar0)).parent.name).replace(":", "_").replace(".", "_")
        self.dev = str(bar0)
        self.hbm_offset, self.window_bytes = hbm_offset, window_bytes
        self.ch_base, self.ch_bytes = tuple(ch_base), ch_bytes      # window layout; tests shrink it
        self._fds: list[int] = []
        self._ocl = self._map(bar0, 0, ocl_bytes)
        self._hbm = self._map(bar4, hbm_offset, window_bytes)
        self.reads = 0
        self.clkgen_info: dict | None = None
        if release_resets:
            # must precede the identity check: until the resets are released the HBM is not
            # ready and the board registers answer SLVERR
            self.clkgen_info = self.release_clock_resets()
            self.wait_hbm_ready(ready_timeout)
        if check:
            self.check_identity()

    def release_clock_resets(self, timeout: float = CLKGEN_TIMEOUT_S) -> dict:
        """Release the AWS_CLK_GEN resets (see `release_clock_resets`). Safe to repeat."""
        if self.clkgen_bar is None:
            raise ValueError("no AWS_CLK_GEN BAR: give a bdf or clkgen_bar=")
        m = self._map(self.clkgen_bar, 0, CLKGEN_BAR_BYTES)
        try:
            return release_clock_resets(lambda o: struct.unpack_from("<I", m, o)[0],
                                        lambda o, v: struct.pack_into("<I", m, o, v & 0xFFFFFFFF),
                                        timeout=timeout)
        finally:
            m.close()

    def wait_hbm_ready(self, timeout: float = 30.0, poll_interval: float = 0.05) -> None:
        """Wait for F2_STATUS.hbm_ready (HBM IP initialized), which needs the resets released."""
        deadline = time.monotonic() + timeout
        while not self.reg_read(F2_STATUS) & F2_ST_HBM_READY:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"HBM not ready after {timeout:g} s (F2_STATUS={self.reg_read(F2_STATUS):#x})")
            time.sleep(poll_interval)

    def _map(self, path, offset: int, length: int) -> mmap.mmap:
        fd = os.open(str(path), os.O_RDWR | os.O_SYNC)
        self._fds.append(fd)
        return mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)

    def check_identity(self) -> None:
        f2, board = self.reg_read_many([F2_ID, R.R_ID])
        if f2 != F2_ID_VALUE:
            raise RuntimeError(f"F2 ID register reads {f2:#x}, expected {F2_ID_VALUE:#x}: wrong AFI or slot")
        if board != R.ID_OTPU:
            raise RuntimeError(f"board ID register reads {board:#x} (HBM not ready, or a wrong AFI)")

    def close(self) -> None:
        for m in (self._ocl, self._hbm):
            m.close()
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    # ---- registers: 32-bit accesses
    def reg_write(self, off: int, val: int) -> None:
        struct.pack_into("<I", self._ocl, off, val & 0xFFFFFFFF)

    def reg_read(self, off: int) -> int:
        self.reads += 1
        return struct.unpack_from("<I", self._ocl, off)[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(o) for o in offs]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0, expect: float = 0.0) -> int:
        return XdmaTransport.poll(self, off, mask, val, timeout, expect)

    # ---- memory: channel c at BASE[c] inside the HBM window
    def _span(self, ch: int, off: int, n: int) -> int:
        if ch not in (0, 1) or off < 0 or off + n > self.ch_bytes:
            raise AxiDecodeError(f"channel {ch} offset {off:#x} length {n:#x} is outside the window")
        return self.ch_base[ch] + off

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        a = self._span(ch, off, len(data))
        self._hbm[a:a + len(data)] = memoryview(np.ascontiguousarray(data).view(np.uint8).reshape(-1))

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        a = self._span(ch, off, n)
        if out is None:
            return np.frombuffer(self._hbm[a:a + n], np.uint8).copy()
        out[:] = np.frombuffer(self._hbm[a:a + n], np.uint8)
        return out


class EmulatedTransport(FakeTransport):
    """A software card. Not a hardware model: it has the register map, the F2 block, the
    memory window and the address-decode errors, and it *executes* programs (on the bit-exact
    ISA machine, from the program the driver loaded into its DRAM). It has no cycle model:
    CYCLES reads 0 and the counters are not meaningful.

    Every result from it is labelled `emulated`. It exists so the driver, the loopback and
    descriptor paths and the whole LLM flow run without hardware.
    """

    ecc = False

    def __init__(self, cfg, ch_bytes: int | None = None, pcs_per_ch: int = 2, pc_aw: int | None = None,
                 core_khz: int = 125_000, build_id: int = 0x0F2E_0001, hbm_ready: bool = True):
        ch_bytes = ch_bytes or cfg.DRAM_BYTES // 2
        super().__init__(ch_bytes=ch_bytes, regmap=3, devname=None, run_s=0.0, cycles=0,
                         core_khz=core_khz, build_id=build_id, D=cfg.D, MCOLS=cfg.MCOLS,
                         LANES=cfg.LANES, w4=True, pair=bool(getattr(cfg, "PAIR", False)))
        self.cfg = cfg
        self.pcs_per_ch = pcs_per_ch
        self.pc_aw = pc_aw if pc_aw is not None else max(1, (ch_bytes // pcs_per_ch - 1).bit_length())
        self.hbm_ready = hbm_ready
        self.f2_scratch = 0
        self.core_errs = self.pcis_errs = 0
        self.icount = 0
        self.error = False
        self.program_runs = 0

    # ---- memory: the PCIS window
    def _check(self, ch: int, off: int, n: int) -> None:
        if ch not in (0, 1) or off < 0 or off + n > min(len(self.ch[ch]), P.channel_capacity(self.pcs_per_ch, self.pc_aw)):
            self.pcis_errs += 1
            raise AxiDecodeError(f"channel {ch} offset {off:#x} length {n:#x} is beyond the card's memory (DECERR)")

    def mem_write(self, ch, off, data):
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self._check(ch, off, len(data))
        super().mem_write(ch, off, data)

    def mem_read(self, ch, off, n, out=None):
        self._check(ch, off, n)
        return super().mem_read(ch, off, n, out)

    # ---- registers
    def reg_write(self, off, val):
        if off >= F2_BASE:
            if off >= 0x2000:
                return                                 # unmapped: the OCL block answers DECERR
            if off == F2_CTRL and val & 1:
                self.core_errs = self.pcis_errs = 0
            elif off == F2_SCRATCH:
                self.f2_scratch = val & 0xFFFFFFFF
            return
        if not self.hbm_ready:
            return
        was_running = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
        super().reg_write(off, val)
        if off == R.R_CTRL and val & R.CTRL_RUN and not was_running:
            self._execute()

    def reg_read(self, off):
        if off >= F2_BASE:
            self.reads += 1
            status = (F2_ST_HBM_READY if self.hbm_ready else 0) | (F2_ST_CORE_ERR if self.core_errs else 0) \
                | (F2_ST_PCIS_ERR if self.pcis_errs else 0)
            return {F2_ID: F2_ID_VALUE, F2_VERSION: 1,
                    F2_CAPS: f2_caps(self.pcs_per_ch, self.pc_aw), F2_STATUS: status,
                    F2_CTRL: 0, F2_CORE_KHZ: self.core_khz, F2_BUILD_ID: self.build_id,
                    F2_HBM_BASE_LO: 0, F2_HBM_BASE_HI: P.HBM_BASE >> 32,
                    F2_CORE_ERRS: self.core_errs, F2_PCIS_ERRS: self.pcis_errs,
                    F2_SCRATCH: self.f2_scratch}.get(off, UNMAPPED)
        if not self.hbm_ready:
            return UNMAPPED
        if off == R.R_STATUS:
            v = super().reg_read(off)
            return (v | R.ST_ERROR) if self.error else v
        if off == R.R_ICOUNT:
            return self.icount
        if off == R.R_TEMP:
            return 0                                   # no sensor: never valid
        return super().reg_read(off)

    # ---- program execution on the ISA machine
    def _logical(self, addr: int, n: int) -> np.ndarray:
        a0 = addr // 128 * 128
        a1 = -(-(addr + n) // 128) * 128
        parts = [self.ch[c][a0 // 2:a1 // 2] for c in (0, 1)]
        return join(parts)[addr - a0:addr - a0 + n]

    def _execute(self) -> None:
        from opentpu.isasim import Machine, SimError
        addr, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
        words = self._logical(addr, 32 * n).view("<u4")
        program = [I.Instr.decode(words[8 * i:8 * i + 8]) for i in range(n)]
        dram = np.concatenate([join([self.ch[0], self.ch[1]])])
        machine = Machine(self.cfg, [program], [dram])
        self.error = False
        try:
            machine.run(max_steps=1 << 40)
        except SimError:
            self.error = True
        state = machine.slices[0]
        self.icount = state.icount
        self.program_runs += 1
        for c, off, part in split(0, state.dram[:len(dram)]):
            self.ch[c][off:off + len(part)] = part
