"""F2 wrapper, adapter and shell model in Verilator (no checkpoints, no hardware)."""
from dataclasses import replace

import pytest

from malleable.f2.sim import F2SimTransport, run_adapter_bench
from malleable.llm.records import PERSONALITIES
from llm_fixture import tiny


@pytest.mark.parametrize("pcs", [2, 4])
def test_adapter_bench(pcs):
    out = run_adapter_bench(pcs)
    assert f"F2_ADAPTER PASS (PCS={pcs}" in out, out[-2000:]


@pytest.mark.parametrize("plusargs", [["+hbm_stall=0", "+hbm_lat=1"], ["+hbm_stall=70", "+hbm_lat=60", "+hbm_seed=5"]])
def test_adapter_bench_under_memory_timing(plusargs):
    assert "F2_ADAPTER PASS" in run_adapter_bench(2, plusargs=plusargs)


def _transport(**kw):
    _, _, spec = tiny()
    cfg = replace(PERSONALITIES["balanced"].config(spec, 128), DRAM_BYTES=1 << 22)
    return cfg, F2SimTransport(cfg, uarch=PERSONALITIES["balanced"].uarch, **kw)


def test_shell_registers_and_identity():
    from opentpu.host.board import Board
    cfg, t = _transport()
    board = Board(t)
    info = board.info()
    assert (info["D"], info["MCOLS"], info["LANES"]) == (128, 4, 8)
    assert info["regmap"] >= 2 and info["core_khz"] == 125000 and info["temp_c"] is None
    assert info["calibrated"] and not info["running"]
    f2id, ver, caps, status, khz = t.reg_read_many([0x1000, 0x1004, 0x1008, 0x100C, 0x1014])
    assert f2id == 0x46324F54 and ver == 1 and khz == 125000
    assert (caps & 0xFF, (caps >> 8) & 0xFF, (caps >> 16) & 0xFF, caps >> 24) == (2, 20, 9, 2)
    assert status == 1                              # HBM ready, no error seen
    # unmapped OCL space answers DECERR with 0xDEADBEEF
    assert t.reg_read(0x2000) == 0xDEADBEEF
    # F2 scratch register
    t.reg_write(0x102C, 0xC0FFEE11)
    assert t.reg_read(0x102C) == 0xC0FFEE11
    # the board block's own scratch register (original map) still works through the crossing
    t.reg_write(0x38, 0x12345678)
    assert t.reg_read(0x38) == 0x12345678


def test_dma_loopback_both_channels_and_placement():
    cfg, t = _transport()
    written = [0, 0, 0, 0]
    for ch in (0, 1):
        lines = t.dma_check(ch, 0x1000, 0x8000, seed=3 + ch)
        assert lines and all(x.startswith("DMA_OK") for x in lines), lines
        written = [a + int(b) for a, b in zip(written, t.stats_line.split("pc_wr_beats=")[1].split(","))]
    assert t.dma_time(0, 0, 0x4000) > 0
    # writes reached all four PCs (two per core channel), striped evenly:
    # 32 KiB per channel = 1024 256-bit beats = 512 per PC
    assert written == [512, 512, 512, 512], written


def test_dma_error_paths_do_not_hang_and_are_counted():
    cfg, t = _transport()
    # a window address beyond the channel's PCs answers DECERR: the bench reports failures, not a hang;
    # the F2 block counts it, latches the sticky bit, and the clear register resets both
    beyond = 1 << 21                                 # first byte past the 2 MiB channel model
    t.script += [f"DR {beyond:x} 40 1", "R 1028", "R 100c", "W 1010 1", "R 1028", "R 100c"]
    t.flush()
    assert any(x.startswith("DMA_FAIL") for x in t.dma_lines)
    errs, status, errs_after, status_after = t.results
    assert errs >= 1 and status & 4 and status & 1
    assert errs_after == 0 and not status_after & 4
