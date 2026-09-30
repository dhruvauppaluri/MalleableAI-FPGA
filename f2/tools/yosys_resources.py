"""Yosys `synth_xilinx -family xcup` cell counts for the F2 wrapper blocks.

This is Yosys 0.33's *generic* UltraScale+ mapping, not Vivado: LUT/FF/RAM counts here are a
rough size indication, wide multiplexers map to MUXF7/8/9 chains that Vivado would map
differently, and there is no timing. The frozen OpenTPU core cannot be read by Yosys 0.33
(SystemVerilog constructs), so it is NOT covered; its size comes from upstream's own Vivado
reports (docs/f2-resource-timing.md).

    python f2/tools/yosys_resources.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["f2_fifo.sv", "f2_cdc.sv", "f2_ocl.sv", "f2_hbm_router.sv", "f2_hbm_pc_bridge.sv", "f2_hbm_adapter.sv"]

CASES = [
    # (label, top, {param: value})
    ("f2_async_fifo, 289 bits x 16 (one W channel crossing)", "f2_async_fifo", {"W": 289, "AW": 4}),
    ("f2_hbm_router, core channel (2 PCs, 1-bit ID)", "f2_hbm_router", {"NPC": 2, "AW": 34, "IDW": 1}),
    ("f2_hbm_router, PCIS (4 PCs, 16-bit ID, group select)", "f2_hbm_router", {"NPC": 4, "GRPB": 1, "AW": 34, "IDW": 16}),
    ("f2_hbm_pc_bridge (one PC, two sources)", "f2_hbm_pc_bridge", {}),
    ("f2_ocl (OCL front end + F2 registers + crossings)", "f2_ocl", {}),
    ("f2_hbm_adapter, PCS_PER_CH=2 (4 PCs)", "f2_hbm_adapter", {"PCS_PER_CH": 2}),
    ("f2_hbm_adapter, PCS_PER_CH=4 (8 PCs)", "f2_hbm_adapter", {"PCS_PER_CH": 4}),
]


def run(top: str, params: dict) -> dict:
    files = " ".join(str(ROOT / "f2" / "rtl" / f) for f in FILES)
    chp = "".join(f"chparam -set {k} {v} {top}; " for k, v in params.items())
    script = f"read_verilog -sv {files}; {chp}synth_xilinx -top {top} -family xcup -flatten; stat"
    r = subprocess.run(["yosys", "-q", "-p", script, "-l", "/dev/stdout"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(out[-2000:])
    tail = out[out.rindex("Number of cells"):]
    cells = {m[1]: int(m[0]) for m in re.findall(r"^\s+(\d+)\s+(\S+)\s*$", tail.split("\n\n")[0], re.M)}
    if not cells:
        cells = {m[0]: int(m[1]) for m in re.findall(r"^\s+(\S+)\s+(\d+)\s*$", tail, re.M)}
    return cells


def summarize(cells: dict) -> dict:
    lut = sum(v for k, v in cells.items() if re.fullmatch(r"LUT[1-6]", k))
    ff = sum(v for k, v in cells.items() if k.startswith("FD"))
    ram = sum(v for k, v in cells.items() if k.startswith(("RAM", "RAMB", "URAM")))
    mux = sum(v for k, v in cells.items() if k.startswith("MUXF"))
    return {"LUT": lut, "FF": ff, "LUTRAM_or_BRAM_cells": ram, "MUXF": mux, "CARRY": sum(v for k, v in cells.items() if k.startswith("CARRY"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = []
    for label, top, params in CASES:
        cells = run(top, params)
        rows.append({"block": label, "top": top, "params": params, **summarize(cells)})
    print("| Block | LUT | FF | RAM cells | MUXF | CARRY |\n| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['block']} | {r['LUT']} | {r['FF']} | {r['LUTRAM_or_BRAM_cells']} | {r['MUXF']} | {r['CARRY']} |")
    if a.json:
        Path(a.json).write_text(json.dumps({"tool": "yosys synth_xilinx -family xcup (generic mapping, not Vivado)", "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
