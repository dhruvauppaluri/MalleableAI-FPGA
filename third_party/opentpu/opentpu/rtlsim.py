"""Build and run the SystemVerilog RTL with Verilator."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import platform
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
TB = ROOT / "sim" / "verilator"
# Verilator's generated make invocation does not quote paths containing spaces.
# Cache/stage sources in a space-free temporary-root directory, preserving hashes.
BUILD = Path(tempfile.gettempdir()) / ("malleable-opentpu-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:12])

RTL_SOURCES = [
    "vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv", "mem/otpu_dram.sv", "mem/otpu_tmem.sv",
    "mem/otpu_axi_dram.sv", "mem/otpu_actram.sv", "seq/otpu_seq.sv", "dma/otpu_dma.sv", "mxu/otpu_mxu.sv",
    "vpu/otpu_quant.sv", "vpu/otpu_vpu.sv", "top/otpu_coll.sv", "top/otpu_slice.sv",
    "top/otpu_top.sv",
]


@contextmanager
def _build_lock(path):
    # Local RTL hosts are POSIX (macOS/Linux/WSL2). Never publish a partially
    # linked executable to another worker; OS locks also release on cancellation.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def build(top: str, sources: list[Path], params: dict | None = None) -> Path:
    params = params or {}
    toolchain = {'verilator': subprocess.check_output(['verilator', '--version']).decode(),
                 'compiler': subprocess.check_output([os.environ.get('CXX', 'c++'), '--version']).decode(),
                 'system': platform.platform(), 'machine': platform.machine(),
                 'CXX': os.environ.get('CXX', ''), 'CXXFLAGS': os.environ.get('CXXFLAGS', ''),
                 'LDFLAGS': os.environ.get('LDFLAGS', ''), 'patch': 'stream-v3-locked-j2'}
    h = hashlib.sha256(json.dumps(toolchain, sort_keys=True).encode())
    h.update(top.encode()); h.update(repr(sorted(params.items())).encode())
    for source in sources:
        h.update(Path(source).name.encode()); h.update(Path(source).read_bytes())
    out = BUILD / f'{top}_{h.hexdigest()[:20]}'
    with _build_lock(BUILD / (out.name + '.lock')):
        return _build_locked(top, sources, params, out, toolchain)


def _build_locked(top, sources, params, out, toolchain):
    """Compile `top` with Verilator (--binary); cached on source contents and parameters."""
    exe = out / f"V{top}"
    if exe.exists() and (out / 'complete.json').is_file():
        return exe
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    staged = out / "sources"
    staged.mkdir(exist_ok=True)
    local_sources = []
    for i, source in enumerate(sources):
        target = staged / f"{i}_{Path(source).name}"
        shutil.copyfile(source, target)
        local_sources.append(target)
    cmd = ["verilator", "--binary", "-j", "2", "--top-module", top, "-Wno-fatal",
           "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
           "-O3", "--x-assign", "0", "--x-initial", "0", "-Mdir", str(out)]
    cmd += [f"-G{k}={v}" for k, v in params.items()]
    cmd += [str(s) for s in local_sources]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 and "linker command failed" in r.stdout + r.stderr:
        # the parallel make occasionally archives a stale object: rebuild the archive once
        for f in out.glob("*__ALL.a"):
            f.unlink()
        r = subprocess.run(["make", "-C", str(out), "-f", f"V{top}.mk", "-j", "2"],
                           capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not exe.exists():
        raise RuntimeError(f"verilator failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    (out / 'complete.json').write_text(json.dumps(toolchain, sort_keys=True))
    return exe


def run_fp_vectors(vec_path: Path) -> str:
    exe = build("tb_fp", [RTL / "vpu/otpu_fp.sv", TB / "tb_fp.sv"])
    r = subprocess.run([str(exe), f"+vec={vec_path}"], capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


def _write_hex(path: Path, words: np.ndarray) -> None:
    nz = np.nonzero(words)[0]
    n = int(nz[-1]) + 1 if len(nz) else 1
    path.write_text("\n".join(f"{int(w):08x}" for w in words[:n]) + "\n")


def _read_hex(path: Path, n: int) -> np.ndarray:
    toks = [t for t in path.read_text().split() if not t.startswith(("@", "//"))]
    if len(toks) != n:                  # a short dump (e.g. a full disk) is an error
        raise RuntimeError(f"{path.name}: {len(toks)} words, expected {n}")
    out = np.zeros(n, dtype=np.uint32)
    vals = np.fromiter((int(t, 16) for t in toks), dtype=np.uint64, count=len(toks))
    out[: len(vals)] = vals.astype(np.uint32)
    return out


# Micro-architecture knobs that change timing only (never results). Tests and tools may
# override them through `rtlsim.UARCH` or the `uarch` argument of run().
UARCH = {"WIN": 32, "RPB": 4, "WPB": 2}
# The board build: TMEM is replicated per read port (reads never conflict), one write per bank
# per cycle (simple dual-port block RAM), a 16-entry dispatch window, and a 1024-chunk MXU
# prefetch FIFO (128 KiB of block RAM: the weight stream runs ahead through the serial
# norm -> quantize -> MM dependency chains).
# RPB covers every read lane (8 ports x up to 16 lanes), which selects the slice's shallow
# write-mask arbiter, as on the board.
BOARD_UARCH = {"WIN": 16, "RPB": 128, "WPB": 1, "FIFO_DEPTH": 1024}
if os.environ.get("OTPU_UARCH") == "board":
    UARCH = dict(BOARD_UARCH)
# MXU dot-product implementation (timing only): OTPU_MXU=cascade selects the DSP cascade
# chains (OTPU_MXU_CL products per chain) instead of the adder tree.
if os.environ.get("OTPU_MXU") == "cascade":
    UARCH["MXU_IMPL"] = 1
    UARCH["MXU_CL"] = int(os.environ.get("OTPU_MXU_CL", "16"))
# VPU lanes with the composite functions (exp2, recip, rsqrt; timing only): OTPU_VPU_CL=4
if os.environ.get("OTPU_VPU_CL"):
    UARCH["VPU_CL"] = int(os.environ["OTPU_VPU_CL"])
# TMEM lanes of the MXU and the quantizer when fewer than LANES (timing only): OTPU_ULANES=8
if os.environ.get("OTPU_ULANES"):
    UARCH["ULANES"] = int(os.environ["OTPU_ULANES"])

# The memory path. AXI: the board's AXI adapter in front of a two-channel AXI memory model with
# random stalls (percent) and latency (D = 128 only; other configurations keep the behavioural
# DRAM). BOOT: the program is placed in DRAM and copied into IMEM by the slice's loader, as on
# the board. The environment (OTPU_AXI=1, OTPU_BOOT=1, OTPU_STALL=n) sets the defaults, so the
# whole suite can be run on the board's memory path.
MEMORY = {"AXI": os.environ.get("OTPU_AXI", "0") == "1",
          "BOOT": os.environ.get("OTPU_BOOT", "0") == "1",
          "STALL": int(os.environ.get("OTPU_STALL", "20")),
          "SEED": int(os.environ.get("OTPU_SEED", "1")),
          "BW": int(os.environ.get("OTPU_BW", "100")),        # percent of a beat/cycle/channel
          "LAT": int(os.environ.get("OTPU_LAT", "20")),
          "ARC": int(os.environ.get("OTPU_ARC", "0"))}     # cycles per AXI read transaction


def top_params(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool = False,
               dram_bytes: int | None = None) -> dict:
    p = {"S": cfg.S, "D": cfg.D, "MCOLS": cfg.MCOLS, "ACT_BLOCKS": cfg.ACT_BLOCKS,
         "TMEM_WORDS": cfg.TMEM_WORDS, "IMEM_WORDS": cfg.IMEM_WORDS,
         "DRAM_WORDS": (dram_bytes or cfg.DRAM_BYTES) // 4, "DRAM_LAT": dram_lat,
         "LANES": cfg.LANES,
         "AXI": int(axi)}
    p.update(UARCH)
    p.update(uarch or {})
    return p


def build_top(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool = False,
              dram_bytes: int | None = None) -> Path:
    return build("tb_top", [RTL / s for s in RTL_SOURCES] +
                 [TB / "otpu_axi_mem.sv", TB / "tb_top.sv"],
                 top_params(cfg, dram_lat, uarch, axi, dram_bytes))


def run(cfg, programs: list, images: list, *args, keep: Path | None = None, **kw):
    """Run the RTL; returns (drams as uint8 arrays, tmems as uint32 arrays, stats). The run's
    files (DRAM images, up to the machine's DRAM size) live in a temporary directory that is
    removed afterwards, unless `keep` names a directory to leave them in."""
    if keep:
        return _run(cfg, programs, images, *args, keep=keep, **kw)
    with tempfile.TemporaryDirectory(prefix="otpu_") as d:
        return _run(cfg, programs, images, *args, keep=Path(d), **kw)


def _run(cfg, programs: list, images: list, dram_lat: int = 8, max_cycles: int = 50_000_000,
        keep: Path | None = None, trace: bool = False, uarch: dict | None = None,
        axi: bool | None = None, boot: bool | None = None, stall: int | None = None,
        seed: int | None = None, bw: int | None = None, lat: int | None = None,
        arc: int | None = None, plusargs: list | None = None, on_line=None):
    """Run the RTL; returns (drams as uint8 arrays, tmems as uint32 arrays, stats)."""
    from . import isa as I
    axi = MEMORY["AXI"] if axi is None else axi
    axi = axi and cfg.D == 128
    boot = MEMORY["BOOT"] if boot is None else boot
    stall = MEMORY["STALL"] if stall is None else stall
    seed = MEMORY["SEED"] if seed is None else seed
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="otpu_"))
    tmp.mkdir(parents=True, exist_ok=True)
    imgs, progs = [], []
    for s in range(cfg.S):
        words = I.assemble(programs[s])
        if len(words) > cfg.IMEM_WORDS:
            raise ValueError("program does not fit IMEM")
        _write_hex(tmp / f"prog_{s}.hex", words)
        img = np.asarray(images[s], np.uint8)
        imgs.append(np.concatenate([img, np.zeros(-len(img) % 4, np.uint8)]))
        progs.append(np.asarray(words, "<u4"))
    # boot: every slice's program right after the largest image (one loader address), or, if
    # there is no room, above the machine's DRAM (the simulated DRAM is then doubled)
    at = -(-max(len(i) for i in imgs) // cfg.D) * cfg.D
    grow = boot and at + 4 * max(len(p) for p in progs) > cfg.DRAM_BYTES
    if grow:
        at = cfg.DRAM_BYTES
    exe = build_top(cfg, 20 if axi else dram_lat, uarch, axi,
                    2 * cfg.DRAM_BYTES if grow else None)
    for s in range(cfg.S):
        img = imgs[s]
        if boot:
            if 4 * len(progs[s]) > cfg.DRAM_BYTES:
                raise ValueError("no room in DRAM for the program")
            img = np.concatenate([img, np.zeros(at - len(img), np.uint8), progs[s].view(np.uint8)])
        img.view("<u4").astype(">u4").tofile(tmp / f"dram_{s}.bin")
    args = [str(exe), f"+dir={tmp}", f"+max_cycles={max_cycles}"] + (["+trace"] if trace else [])
    args += list(plusargs or [])
    if axi:
        args += [f"+axi_stall={stall}", f"+axi_seed={seed}",
                 f"+axi_bw={MEMORY['BW'] if bw is None else bw}",
                 f"+axi_lat={MEMORY['LAT'] if lat is None else lat}",
                 f"+axi_arc={MEMORY['ARC'] if arc is None else arc}"]
    if boot:
        args += ["+boot", f"+boot_addr={at}", f"+boot_n={max(len(p) for p in progs) // 8}"]
    if on_line is None:
        r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        out = r.stdout + r.stderr
        code = r.returncode
    else:
        # Complete trace stays on disk; callbacks may coalesce browser updates.
        import time
        import threading
        started = time.monotonic()
        with subprocess.Popen(args + ["+heartbeat=100000"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            timer=threading.Timer(3600,process.kill)
            timer.daemon=True; timer.start()
            try:
                with (tmp / "trace.txt").open("w") as stream:
                    for line in process.stdout:
                        stream.write(line)
                        on_line(line.rstrip())
                        if time.monotonic() - started > 3600:
                            raise TimeoutError("RTL run exceeded one hour")
                code = process.wait()
            except BaseException:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
            finally: timer.cancel()
        out = (tmp / "trace.txt").read_text()
    if code != 0:
        raise RuntimeError(f"RTL exited {code}: {out[-3000:]}")
    import re
    m = re.search(r"RESULT cycles=(\d+) halted=(\d+) error=(\d+)", out)
    if not m:
        raise RuntimeError(f"RTL simulation failed:\n{out[-3000:]}")
    cycles, halted, err = (int(x) for x in m.groups())
    if not halted or err:
        raise RuntimeError(f"RTL did not halt cleanly (halted={halted} error={err}):\n{out[-2000:]}")
    icounts = [int(x) for x in re.findall(r"SLICE \d+ icount=(\d+)", out)]
    drams = [np.fromfile(tmp / f"dram_out_{s}.bin", dtype=np.uint8) for s in range(cfg.S)]
    for s in range(cfg.S):             # a short dump (e.g. a full disk) is an error, not a result
        if len(drams[s]) < len(imgs[s]):
            raise RuntimeError(f"dram_out_{s}.bin has {len(drams[s])} bytes, fewer than the "
                               f"{len(imgs[s])}-byte image: the dump was cut short")
    if boot:                           # the program is not part of the result
        drams = [d[:cfg.DRAM_BYTES] for d in drams]
        for s in range(cfg.S):
            drams[s][at:at + 4 * len(progs[s])] = 0
    tmems = [_read_hex(tmp / f"tmem_{s}.hex", cfg.TMEM_WORDS) for s in range(cfg.S)]
    stats = {"cycles": cycles, "instructions": icounts}
    if axi:                            # per channel: AXI read transactions and beats
        stats["axi_reads"] = [(int(a), int(b)) for a, b in
                              re.findall(r"AXI ch\d ar=(\d+) beats=(\d+)", out)]
        # and the port A reads among them, DDR3 row opens and read-modify-writes (+axi_dram)
        # (of them from port A / QST: rmw_a)
        stats["axi_detail"] = [dict(zip(("ar_a", "row_miss", "rmw", "rmw_a"), map(int, m)))
                               for m in re.findall(r"AXI ch\d .*ar_a=(\d+) row_miss=(\d+) "
                                                   r"rmw=(\d+) rmw_a=(\d+)", out)]
    if trace:
        stats["trace"] = out
    return drams, tmems, stats
