#!/usr/bin/env bash
# Wrapper for f2/vivado/ooc_synth.tcl. Run on Vivado 2025.2 (synthesis and impl=1, CL wrapper). Refuses to start without Vivado.
#   f2/vivado/run_ooc.sh <out_dir> [key=value ...]     e.g. pcs=2 core_ns=8.0 impl=0
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if ! command -v vivado >/dev/null 2>&1; then
  echo "vivado is not on PATH. Source the tool settings (an FPGA Developer AMI has them) and retry." >&2
  exit 2
fi
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <out_dir> [pcs=2 core_ns=8.0 mcols=2 lanes=8 impl=0 jobs=8]" >&2
  exit 2
fi
out="$1"; shift
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "refusing to reuse a non-empty output directory: $out" >&2
  exit 2
fi
mkdir -p "$out"
vivado -mode batch -nojournal -log "$out/vivado.log" -source "$here/ooc_synth.tcl" -tclargs "$out" "$@"
