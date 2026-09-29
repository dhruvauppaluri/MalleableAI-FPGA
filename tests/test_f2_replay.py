"""The recorded tiny test through the F2 shell model, checked against the ISA machine."""
import pytest

from malleable.f2 import replay
from malleable.f2.sim import F2SimTransport


def _quiet(*_):
    pass


def test_dispatch_projection_ignores_cycle_stamps():
    a = ["T0 D c=6 s=0 pc=1 op=10 w1=00000000 w2=00000000 w3=00000080",
         "T0 S c=7 s=0 u=0 r=7", "T0 D c=7 s=1 pc=2 op=10 w1=00000200 w2=00000080 w3=00000040"]
    b = ["T0 D c=99 s=3 pc=1 op=10 w1=00000000 w2=00000000 w3=00000080",
         "T0 E c=100 s=0", "T0 D c=101 s=5 pc=2 op=10 w1=00000200 w2=00000080 w3=00000040"]
    assert replay.projection_hash(a) == replay.projection_hash(b)
    assert replay.projection_hash(a) != replay.projection_hash(a[:1])
    assert len(replay.dispatch_events(a)) == 2


def test_short_replay_is_byte_exact(tmp_path):
    ev = replay.run_replay(tmp_path / "r", steps=6, dma_steps=(0, 5), progress=_quiet)
    s = ev["summary"]
    assert s["steps"] == 6 and s["all_dram_equal"] and s["all_tmem_equal"] and s["all_instructions_equal"]
    assert s["projection_matches"] == 6 and s["dma_path_steps"] == [0, 5]
    assert ev["provenance"].startswith("simulation")
    assert "not comparable" in ev["hash_semantics"]["ref_trace_sha256"]


@pytest.mark.parametrize("stall,lat,clocks", [
    (0, 1, None),
    (60, 80, {"core_ns": 9.1, "main_ns": 4.0, "hbm_ns": 2.2}),
    (30, 20, {"core_ns": 4.0, "main_ns": 4.0, "hbm_ns": 2.2}),     # core at the main clock (250 MHz)
])
def test_replay_under_memory_and_clock_variations(tmp_path, stall, lat, clocks):
    ev = replay.run_replay(tmp_path / "r", steps=3, dma_steps=(0,), hbm_stall=stall, hbm_lat=lat,
                           clocks=clocks, progress=_quiet)
    s = ev["summary"]
    assert s["all_dram_equal"] and s["all_tmem_equal"] and s["projection_matches"] == 3


def test_normalized_hash_drops_verilator_report_lines():
    a = "T0 D c=1 s=0 pc=1 op=10 w1=0 w2=0 w3=0\nRESULT cycles=5 halted=1 error=0\n- Verilator: $finish at 40ns; walltime 0.158 s\n"
    b = "T0 D c=1 s=0 pc=1 op=10 w1=0 w2=0 w3=0\nRESULT cycles=5 halted=1 error=0\n- Verilator: $finish at 40ns; walltime 0.163 s\n"
    assert replay.normalized_trace_sha256(a) == replay.normalized_trace_sha256(b)
    assert replay.normalized_trace_sha256(a) != replay.normalized_trace_sha256(a.replace("cycles=5", "cycles=6"))


def test_recorded_hash_comparison_is_reported(tmp_path):
    ev = replay.run_replay(tmp_path / "a", steps=2, dma_steps=(), progress=_quiet)
    norm = [r["ref_trace_normalized_sha256"] for r in ev["steps"]]
    # a second run of the same test reproduces the normalized hashes exactly
    ev2 = replay.run_replay(tmp_path / "b", steps=2, dma_steps=(),
                            recorded_hashes={"kind": "normalized", "hashes": norm}, progress=_quiet)
    c = ev2["summary"]["recorded_hash_comparison"]
    assert (c["kind"], c["compared"], c["equal"], c["count_matches"]) == ("normalized", 2, 2, True)
    ev3 = replay.run_replay(tmp_path / "c", steps=2, dma_steps=(),
                            recorded_hashes={"kind": "normalized", "hashes": ["0" * 64, norm[1]]}, progress=_quiet)
    assert ev3["summary"]["recorded_hash_comparison"]["equal"] == 1
    # raw hashes are reported with the warning that they are not reproducible
    ev4 = replay.run_replay(tmp_path / "d", steps=2, dma_steps=(), recorded_hashes=["0" * 64] * 2, progress=_quiet)
    c = ev4["summary"]["recorded_hash_comparison"]
    assert c["kind"] == "raw" and "not reproducible" in c["note"]


def test_tracehash_tool_on_saved_reference_traces(tmp_path):
    from malleable.f2 import tracehash
    ev = replay.run_replay(tmp_path / "a", steps=2, dma_steps=(), progress=_quiet)
    out = tracehash.hashes(tmp_path / "a" / "reference")
    assert out["kind"] == "normalized" and out["steps"] == 2
    assert out["hashes"] == [r["ref_trace_normalized_sha256"] for r in ev["steps"]]
    assert out["total_cycles"] == sum(r["ref_cycles"] for r in ev["steps"])


@pytest.mark.parametrize("what", ["dram", "tmem"])
def test_replay_detects_a_corrupted_f2_path(tmp_path, monkeypatch, what):
    original = F2SimTransport.flush

    def corrupting(self):
        n = self.runs
        original(self)
        if self.runs > n and self.runs == 2:        # the second program run
            if what == "dram":
                self.ch[0][200] ^= 0x01
            else:
                self.last_run_tmem[7] ^= 0x10
    monkeypatch.setattr(F2SimTransport, "flush", corrupting)
    with pytest.raises(ValueError, match="F2 path mismatch"):
        replay.run_replay(tmp_path / "r", steps=4, dma_steps=(), progress=_quiet)
