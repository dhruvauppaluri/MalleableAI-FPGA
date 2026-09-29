"""Per-step normalized trace hashes from saved tb_top traces.

    python -m malleable.f2.tracehash build/release-evidence/tiny/traces > hashes.json

Reads step-*/trace.txt (from CheckedRtlBackend / tools/export_tiny_rtl_evidence.py), drops
Verilator's own report lines (they carry wall-clock time and build paths, so the raw hash of
the file changes on every run) and prints {"kind": "normalized", "hashes": [...]} in step
order, plus per-step cycle counts. The output can be given to
`python -m malleable.f2.replay --recorded-hashes`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .replay import normalized_trace_sha256


def hashes(root: Path) -> dict:
    steps = sorted(p for p in Path(root).glob("step-*") if (p / "trace.txt").is_file())
    if not steps:
        raise SystemExit(f"no step-*/trace.txt under {root}")
    out, cycles = [], []
    for p in steps:
        text = (p / "trace.txt").read_text()
        out.append(normalized_trace_sha256(text))
        m = re.search(r"RESULT cycles=(\d+)", text)
        cycles.append(int(m.group(1)) if m else None)
    return {"kind": "normalized", "steps": len(out), "cycles": cycles, "total_cycles": sum(c or 0 for c in cycles),
            "hashes": out}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(hashes(Path(args[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
