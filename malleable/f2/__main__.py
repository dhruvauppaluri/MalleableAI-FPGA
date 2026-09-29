"""python -m malleable.f2 {loopback,descriptor,steps,all} [--mode emulate|sim|hardware]

Default mode is the software emulation: it needs no hardware, simulator or AWS access.
`--mode hardware` needs --enable-hardware and a PCI address, refuses otherwise, and has never
been run against a card.
"""
from __future__ import annotations

import argparse
import json
import sys

from malleable.llm import upstream  # noqa: F401

from opentpu.isasim import board_config

from . import bench


def _cfg(personality: str, dram_bytes: int):
    from malleable.llm.records import PERSONALITIES
    p = PERSONALITIES[personality]
    return board_config(MCOLS=p.matrix_columns, LANES=p.vector_lanes, IMEM_WORDS=65536,
                        DRAM_BYTES=dram_bytes), p


def build(mode: str, personality: str, dram_bytes: int, a):
    """(cfg, transport, make_transport) for `mode`."""
    cfg, p = _cfg(personality, dram_bytes)
    if mode == "emulate":
        from .transport import EmulatedTransport
        return cfg, EmulatedTransport(cfg), lambda c, images: _emu(c, images)
    if mode == "sim":
        from .sim import F2SimTransport
        return cfg, F2SimTransport(cfg, uarch=p.uarch, hbm_stall=a.hbm_stall, hbm_lat=a.hbm_lat), \
            lambda c, images: _sim(c, images, p, a)
    from .transport import F2BarTransport
    from opentpu.host.board import Board, device_config
    t = F2BarTransport(a.bdf, enable=a.enable_hardware)
    cfg = device_config(Board(t, lock=False).info())
    return cfg, t, lambda c, images: (cfg, t)


def _emu(c, images):
    from .replay import sim_config_for
    from .transport import EmulatedTransport
    import numpy as np
    f2cfg = sim_config_for(c, len(np.asarray(images[0])))
    return f2cfg, EmulatedTransport(f2cfg)


def _sim(c, images, p, a):
    from .replay import sim_config_for
    from .sim import F2SimTransport
    import numpy as np
    f2cfg = sim_config_for(c, len(np.asarray(images[0])))
    return f2cfg, F2SimTransport(f2cfg, uarch=p.uarch, hbm_stall=a.hbm_stall, hbm_lat=a.hbm_lat)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m malleable.f2", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["loopback", "descriptor", "steps", "all"])
    ap.add_argument("--mode", choices=["emulate", "sim", "hardware"], default="emulate")
    ap.add_argument("--personality", default="balanced")
    ap.add_argument("--dram-bytes", type=int, default=1 << 22, help="card DRAM for loopback/descriptor")
    ap.add_argument("--sizes", default="64,4096,65536,1048576", help="loopback sizes in bytes")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--hbm-stall", type=int, default=20, help="sim mode")
    ap.add_argument("--hbm-lat", type=int, default=20, help="sim mode")
    ap.add_argument("--bdf", help="hardware mode: PCI address of the F2 slot")
    ap.add_argument("--enable-hardware", action="store_true", help="hardware mode: explicit opt-in")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    sizes = tuple(int(x) for x in a.sizes.split(",") if x)
    from .transport import HardwareDisabled
    try:
        cfg, t, make = build(a.mode, a.personality, a.dram_bytes, a)
    except HardwareDisabled as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    results = []
    if a.command in ("loopback", "all"):
        results.append(bench.loopback(t, sizes=sizes, unaligned=a.mode != "sim"))
    if a.command in ("descriptor", "all"):
        results.append(bench.descriptor(t, cfg))
    if a.command in ("steps", "all"):
        results.append(bench.steps(make, a.personality, a.steps))
    rep = bench.report(a.mode, results)
    if a.json:
        bench.write_report(a.json, rep)
    print(json.dumps({"mode": rep["mode"], "provenance": rep["provenance"], "all_passed": rep["all_passed"],
                      "tests": {r["test"]: r["passed"] for r in results}}, indent=2))
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
