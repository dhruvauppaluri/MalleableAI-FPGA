"""Host transports and the benchmark harness, without hardware or a simulator."""
import json

import numpy as np
import pytest

from malleable.f2 import bench, placement as P, transport as T
from malleable.f2.__main__ import main as cli_main
from opentpu.host import regs as R
from opentpu.host.board import Board
from opentpu.isasim import board_config


def cfg(mcols=4):
    return board_config(MCOLS=mcols, LANES=8, IMEM_WORDS=65536, DRAM_BYTES=1 << 22)


def emu():
    return T.EmulatedTransport(cfg())


def test_f2_register_layout_matches_rtl():
    # the value the Verilator shell returns for PCS=2, PC_AW=20, 512-byte stripes (tests/test_f2_rtl.py)
    assert T.f2_caps(2, 20, 9) == 0x02091402
    assert (T.F2_ID, T.F2_SCRATCH, T.F2_PCIS_ERRS) == (0x1000, 0x102C, 0x1028)


def test_emulated_identity_and_registers():
    t = emu()
    board = Board(t, lock=False)
    info = board.info()
    assert (info["D"], info["MCOLS"], info["LANES"]) == (128, 4, 8)
    assert info["calibrated"] and info["core_khz"] == 125_000 and info["temp_c"] is None
    assert t.reg_read(T.F2_ID) == T.F2_ID_VALUE
    caps = t.reg_read(T.F2_CAPS)
    assert (caps & 0xFF, (caps >> 8) & 0xFF, caps >> 24) == (2, 20, 2)
    assert t.reg_read(T.F2_HBM_BASE_HI) == 0x10 and t.reg_read(T.F2_HBM_BASE_LO) == 0
    assert t.reg_read(0x2000) == T.UNMAPPED and t.reg_read(0x1030) == T.UNMAPPED
    t.reg_write(T.F2_SCRATCH, 0xC0FFEE11)
    assert t.reg_read(T.F2_SCRATCH) == 0xC0FFEE11
    t.reg_write(0x2000, 1)                          # ignored, like the RTL's DECERR
    assert t.reg_read(T.F2_STATUS) == T.F2_ST_HBM_READY


def test_board_registers_unavailable_until_hbm_ready():
    t = T.EmulatedTransport(cfg(), hbm_ready=False)
    assert t.reg_read(R.R_ID) == T.UNMAPPED
    assert t.reg_read(T.F2_STATUS) == 0             # F2 block answers, HBM not ready


def test_loopback_all_sizes_channels_and_unaligned():
    r = bench.loopback(emu(), sizes=(1, 63, 64, 65, 4096, 1 << 16, 300_000), repeat=2)
    assert r["passed"] and all(row["verified"] for row in r["rows"])
    assert all(r["edge_cases"].values())


def test_out_of_range_is_decerr_and_counted():
    t = emu()
    cap = P.channel_capacity(t.pcs_per_ch, t.pc_aw)
    with pytest.raises(T.AxiDecodeError):
        t.mem_write(0, cap - 32, np.zeros(64, np.uint8))
    with pytest.raises(T.AxiDecodeError):
        t.mem_read(2, 0, 8)
    assert t.reg_read(T.F2_PCIS_ERRS) == 2 and t.reg_read(T.F2_STATUS) & T.F2_ST_PCIS_ERR
    t.reg_write(T.F2_CTRL, 1)                       # clear
    assert t.reg_read(T.F2_PCIS_ERRS) == 0


def test_descriptor_path_and_failure_detection(monkeypatch):
    c = cfg()
    r = bench.descriptor(T.EmulatedTransport(c), c)
    assert r["passed"], r["checks"]
    assert r["run"]["instructions"] == 3
    # a card that returns wrong data must fail the copy check
    bad = T.EmulatedTransport(c)
    orig = bad.mem_read

    def flipped(ch, off, n, out=None):
        v = np.array(orig(ch, off, n, out))
        if off in (0x10000 // 2, 0x8000 // 2):
            v[0] ^= 1
        return v
    monkeypatch.setattr(bad, "mem_read", flipped)
    r = bench.descriptor(bad, c)
    assert not r["passed"] and not r["checks"]["copy_result"]


def test_program_error_sets_error_status():
    c = cfg()
    t = T.EmulatedTransport(c)
    board = Board(t, lock=False)
    from opentpu import isa as I
    board.load_program(0x10000, I.assemble([I.ld(0, 0, 1 << 20), I.halt()]))   # reads far beyond DRAM
    with pytest.raises(RuntimeError, match="illegal instruction"):
        board.run(timeout=5)


def test_steps_match_isa_every_step():
    def make(c, images):
        from malleable.f2.replay import sim_config_for
        f2cfg = sim_config_for(c, len(np.asarray(images[0])))
        return f2cfg, T.EmulatedTransport(f2cfg)
    r = bench.steps(make, n_steps=8)
    assert r["passed"] and r["steps_checked_against_isa"] == 8


def test_steps_detect_a_corrupting_card(monkeypatch):
    orig = T.EmulatedTransport._execute

    def corrupt(self):
        orig(self)
        if self.program_runs == 3:
            self.ch[0][130] ^= 0x40

    monkeypatch.setattr(T.EmulatedTransport, "_execute", corrupt)

    def make(c, images):
        from malleable.f2.replay import sim_config_for
        f2cfg = sim_config_for(c, len(np.asarray(images[0])))
        return f2cfg, T.EmulatedTransport(f2cfg)
    with pytest.raises(ValueError, match="differs from the ISA machine"):
        bench.steps(make, n_steps=6)


def test_hardware_transport_is_off_by_default(monkeypatch):
    monkeypatch.delenv(T.HARDWARE_ENV, raising=False)
    with pytest.raises(T.HardwareDisabled):
        T.F2BarTransport("0000:00:1d.0")
    # the CLI refuses too, before touching anything
    assert cli_main(["loopback", "--mode", "hardware", "--bdf", "0000:00:1d.0"]) == 2


def _bars(tmp_path, f2id=T.F2_ID_VALUE, board=R.ID_OTPU):
    bar0, bar4 = tmp_path / "bar0", tmp_path / "bar4"
    b0 = bytearray(0x2000)
    b0[0:4] = board.to_bytes(4, "little")
    b0[T.F2_ID:T.F2_ID + 4] = f2id.to_bytes(4, "little")
    bar0.write_bytes(bytes(b0))
    bar4.write_bytes(bytes(1 << 16))
    return bar0, bar4


def test_bar_transport_plumbing_against_files(tmp_path):
    bar0, bar4 = _bars(tmp_path)
    t = T.F2BarTransport(enable=True, bar0=bar0, bar4=bar4, hbm_offset=0, window_bytes=1 << 16,
                         ch_base=(0, 1 << 15), ch_bytes=1 << 15)
    assert t.reg_read(T.F2_ID) == T.F2_ID_VALUE
    t.reg_write(T.F2_SCRATCH, 0x11223344)
    assert t.reg_read(T.F2_SCRATCH) == 0x11223344
    data = np.arange(200, dtype=np.uint8)
    t.mem_write(1, 100, data)
    assert np.array_equal(t.mem_read(1, 100, 200), data)
    with pytest.raises(T.AxiDecodeError):
        t.mem_write(0, (1 << 15) - 10, data)
    t.close()
    raw = bar4.read_bytes()
    assert raw[(1 << 15) + 100:(1 << 15) + 300] == bytes(data)          # channel 1 at its window base
    assert raw[100:300] == bytes(200)                                    # channel 0 untouched
    assert bar0.read_bytes()[T.F2_SCRATCH:T.F2_SCRATCH + 4] == (0x11223344).to_bytes(4, "little")


def test_bar_transport_rejects_wrong_bitstream(tmp_path):
    bar0, bar4 = _bars(tmp_path, f2id=0x1234)
    with pytest.raises(RuntimeError, match="F2 ID"):
        T.F2BarTransport(enable=True, bar0=bar0, bar4=bar4, hbm_offset=0, window_bytes=1 << 16)
    bar0, bar4 = _bars(tmp_path, board=0)
    with pytest.raises(RuntimeError, match="board ID"):
        T.F2BarTransport(enable=True, bar0=bar0, bar4=bar4, hbm_offset=0, window_bytes=1 << 16)


def test_cli_emulate_writes_labelled_report(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli_main(["all", "--steps", "3", "--json", str(out), "--sizes", "64,4096"]) == 0
    rep = json.loads(out.read_text())
    assert rep["mode"] == "emulate" and rep["all_passed"] and not rep["measured_on_hardware"]
    assert rep["provenance"].startswith("emulated")
    assert [r["test"] for r in rep["results"]] == ["dma_loopback", "descriptor_path", "llm_steps"]


def test_loopback_over_the_rtl_model_uses_the_pcis_path():
    from dataclasses import replace
    from malleable.f2.sim import F2SimTransport
    from malleable.llm.records import PERSONALITIES
    c = cfg()
    t = F2SimTransport(c, uarch=PERSONALITIES["balanced"].uarch)
    r = bench.loopback(t, sizes=(64, 4096, 1 << 15))
    assert r["passed"] and all(row["verified"] and "PCIS" in row["path"] for row in r["rows"])
    assert all(row["read_main_clock_cycles_simulated"] > 0 for row in r["rows"])


# ---- AWS_CLK_GEN reset release (register model in place of the real IP; nothing here is hardware)
class FakeClkGen:
    """AWS_CLK_GEN as the spec describes it at load: SYS_RST asserted, MMCMs locking after a
    number of LOCK reads. Records every write so the ordering can be checked."""

    def __init__(self, avail=(1 << 1) | (1 << 8), lock_after=3, ident=T.CLKGEN_ID_VALUE, lock=0x101):
        self.r = {T.CLKGEN_ID: ident, T.CLKGEN_VER: 0x02010000, T.CLKGEN_BLD: 0x09232223,
                  T.CLKGEN_CLKS_AVAIL: avail, T.CLKGEN_GRST: 0, T.CLKGEN_SYSRST: 0xFFFF_FFFE, T.CLKGEN_LOCK: 0}
        self.lock_after, self.lock_value, self.lock_reads, self.writes = lock_after, lock, 0, []

    def read(self, off):
        if off == T.CLKGEN_LOCK:
            self.lock_reads += 1
            return self.lock_value if self.lock_reads > self.lock_after else 0
        return self.r[off]

    def write(self, off, val):
        self.writes.append((off, val))
        self.r[off] = val


def _release(dev, **kw):
    t = [0.0]

    def sleep(dt):
        t[0] += dt
    return T.release_clock_resets(dev.read, dev.write, sleep=sleep, now=lambda: t[0], **kw)


def test_clkgen_lock_mask_follows_the_enabled_groups():
    assert T.clkgen_lock_mask((1 << 1) | (1 << 8)) == 0x101          # this design: group A + HBM
    assert T.clkgen_lock_mask(0x1FF) == 0x151                         # all four groups
    assert T.clkgen_lock_mask(1 << 5) == 0x010                        # only group B's second clock
    assert T.clkgen_lock_mask(0) == 0 and T.clkgen_lock_mask(1) == 0  # bit 0 (main clock) is not a group


def test_clkgen_resets_are_released_only_after_lock():
    dev = FakeClkGen(lock_after=5)
    info = _release(dev)
    assert dev.writes == [(T.CLKGEN_GRST, 0), (T.CLKGEN_SYSRST, 0)]       # global reset first, SYS_RST last
    assert dev.lock_reads == 6 and info["lock_mask"] == 0x101 and info["clks_avail"] == 0x102
    assert dev.r[T.CLKGEN_SYSRST] == 0 and info["version"] == 0x02010000
    _release(dev)                                                         # safe to repeat
    assert dev.r[T.CLKGEN_SYSRST] == 0


def test_clkgen_lock_timeout_leaves_resets_asserted():
    dev = FakeClkGen(lock_after=10**9)
    with pytest.raises(TimeoutError, match="resets left asserted"):
        _release(dev, timeout=1.0)
    assert (T.CLKGEN_SYSRST, 0) not in dev.writes and dev.r[T.CLKGEN_SYSRST] == 0xFFFF_FFFE
    # a partial lock is still a timeout
    dev = FakeClkGen(lock_after=0, lock=0x001)
    with pytest.raises(TimeoutError):
        _release(dev, timeout=1.0)
    assert (T.CLKGEN_SYSRST, 0) not in dev.writes


def test_clkgen_rejects_a_bar_without_the_ip_and_writes_nothing():
    dev = FakeClkGen(ident=0x1234)
    with pytest.raises(RuntimeError, match="ID register"):
        _release(dev)
    assert dev.writes == []
    dev = FakeClkGen(avail=0)
    with pytest.raises(RuntimeError, match="no enabled clocks"):
        _release(dev)
    assert dev.writes == []


def _clkgen_bar(tmp_path, *, lock=0x101, avail=0x102):
    p = tmp_path / "clkgen"
    b = bytearray(T.CLKGEN_BAR_BYTES)
    for off, v in ((T.CLKGEN_ID, T.CLKGEN_ID_VALUE), (T.CLKGEN_CLKS_AVAIL, avail), (T.CLKGEN_LOCK, lock),
                   (T.CLKGEN_SYSRST, 0xFFFF_FFFE)):
        b[off:off + 4] = v.to_bytes(4, "little")
    p.write_bytes(bytes(b))
    return p


def test_bar_transport_releases_resets_before_the_identity_check(tmp_path):
    # board ID register reads 0 (not yet answering) until the HBM is ready: with the release the
    # constructor must do clkgen -> hbm ready -> identity check, in that order
    bar0, bar4 = _bars(tmp_path)
    b0 = bytearray(bar0.read_bytes())
    b0[T.F2_STATUS:T.F2_STATUS + 4] = T.F2_ST_HBM_READY.to_bytes(4, "little")
    bar0.write_bytes(bytes(b0))
    cg = _clkgen_bar(tmp_path)
    t = T.F2BarTransport(enable=True, bar0=bar0, bar4=bar4, hbm_offset=0, window_bytes=1 << 16,
                         release_resets=True, clkgen_bar=cg, ready_timeout=1.0)
    assert t.clkgen_info["lock_mask"] == 0x101
    t.close()
    assert cg.read_bytes()[T.CLKGEN_SYSRST:T.CLKGEN_SYSRST + 4] == bytes(4)   # SYS_RST released
    assert cg.read_bytes()[T.CLKGEN_GRST:T.CLKGEN_GRST + 4] == bytes(4)


def test_bar_transport_reports_an_hbm_that_never_becomes_ready(tmp_path):
    bar0, bar4 = _bars(tmp_path)                       # F2_STATUS stays 0: hbm_ready never set
    with pytest.raises(TimeoutError, match="HBM not ready"):
        T.F2BarTransport(enable=True, bar0=bar0, bar4=bar4, hbm_offset=0, window_bytes=1 << 16,
                         release_resets=True, clkgen_bar=_clkgen_bar(tmp_path), ready_timeout=0.2)


def test_bar_transport_default_clkgen_bar_is_function_1(tmp_path, monkeypatch):
    bar0, bar4 = _bars(tmp_path)
    t = T.F2BarTransport("0000:00:1d.0", enable=True, bar0=bar0, bar4=bar4, hbm_offset=0,
                         window_bytes=1 << 16)
    assert str(t.clkgen_bar) == "/sys/bus/pci/devices/0000:00:1d.1/resource4"   # an unverified assumption
    t.close()


def test_cli_hardware_mode_still_refuses_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(T.HARDWARE_ENV, raising=False)
    assert cli_main(["loopback", "--mode", "hardware", "--bdf", "0000:00:1d.0",
                     "--clkgen-bar", "/nonexistent"]) == 2
