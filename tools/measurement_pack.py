#!/usr/bin/env python3
"""Stage 5: measurement pack across builds LANES in {1, 2, 4, 8, 16} (MAX_DIM 64).

Per build:
  * Yosys 0.33 synthesis of malleable_accelerator_top. Two flows are run:
      - coarse: `synth -run begin:fine` (the Makefile's flow), counting word-level cells;
      - gate: generic gate mapping (proc; flatten; opt; memory -nomap; opt; techmap;
        opt; abc -fast -g AND,NAND,OR,NOR,XOR,XNOR,MUX). Memories stay as $mem_v2 and are
        reported separately as bits and read ports.
    Also run: gate mapping of int8_tiled_accumulator alone (the lane datapath).
    ALL AREA NUMBERS ARE ESTIMATED: generic mapping, not a target-device fit.
    No Vivado/Quartus result exists, and none is implied.
  * RTL simulation (Icarus) of the Stage 2 digits model on all 360 test inputs,
    driven by the tiler-emitted stream: cycles, compute_cycles,
    controller_cycles, tiles, useful_macs and both utilizations (measured, simulated).
  * Roofline-style summary (Williams, Waterman, Patterson, CACM 2009): achieved
    MACs/cycle vs the accept ceiling (LANES) and the 2-cycle/tile ceiling (LANES/2),
    with the gap split into padding, accumulator serialization and controller overhead.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from malleable import tiler  # noqa: E402
from malleable.backend import RTLBackend  # noqa: E402
from malleable.records import ModelArtifact  # noqa: E402

RTL = ['rtl/int8_dot_product.sv', 'rtl/int8_tiled_accumulator.sv', 'rtl/int8_postprocess.sv',
       'rtl/malleable_accelerator_top.sv']
BUILDS = (1, 2, 4, 8, 16)
OUT = ROOT / 'docs' / 'results' / 'data' / 'measurement-pack.json'
LOGS = ROOT / 'build' / 'stage5'
GATE = ('proc; flatten; opt; memory -nomap; opt; techmap; opt; '
        'abc -fast -g AND,NAND,OR,NOR,XOR,XNOR,MUX; opt_clean; stat; write_json {json}')
# `abc -fast`: the default ABC script exceeded the 30-minute time box on the
# flattened LANES=4 top (73k gates, still in ABC after 30 min). -fast is the same
# generic gate mapping with a lighter optimization script (about 10 s in ABC).
EST = 'estimated:yosys-0.33-generic-mapping(abc -fast) (not a target fit; no Vivado/Quartus numbers)'
SIM = 'measured:rtl-simulation(icarus)'


def yosys(script, log, timeout):
    started = time.perf_counter()
    proc = subprocess.run(['yosys', '-q', '-l', str(log), '-p', script], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'yosys failed ({log}): {proc.stderr[-2000:]}')
    return time.perf_counter() - started


def last_stat(log):
    text = Path(log).read_text()
    block = text[text.rindex('Number of cells:'):] if 'Number of cells:' in text else ''
    total = int(re.search(r'Number of cells:\s+(\d+)', block).group(1))
    types = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\s+(\$\S+)\s+(\d+)\s*$', block, re.M)}
    return total, types


def memories(netlist):
    data = json.loads(Path(netlist).read_text())
    rows = []
    for module in data['modules'].values():
        for name, cell in module['cells'].items():
            if cell['type'] == '$mem_v2':
                p = cell['parameters']
                width, size = int(p['WIDTH'], 2), int(p['SIZE'], 2)
                rows.append(dict(name=cell['parameters'].get('MEMID', name), width=width, depth=size,
                                 bits=width * size, read_ports=int(p['RD_PORTS'], 2),
                                 write_ports=int(p['WR_PORTS'], 2)))
    return rows


def gate_summary(types, total):
    ffs = sum(v for k, v in types.items() if 'DFF' in k or 'DLATCH' in k)
    mems = types.get('$mem_v2', 0)
    return dict(cells=total, logic_gates=total - ffs - mems, flip_flops=ffs, memory_cells=mems,
                gate_types={k: v for k, v in sorted(types.items())})


def synthesize(task):
    kind, lanes, timeout = task
    LOGS.mkdir(parents=True, exist_ok=True)
    if kind == 'top-coarse':
        log = LOGS / f'coarse-{lanes}.log'
        seconds = yosys(f"read_verilog -sv {' '.join(RTL)}; chparam -set LANES {lanes} malleable_accelerator_top; "
                        "synth -top malleable_accelerator_top -run begin:fine; stat", log, timeout)
        total, types = last_stat(log)
        return task, dict(cells=total, macc_cells=types.get('$macc', 0), alu_cells=types.get('$alu', 0),
                          mux_cells=types.get('$mux', 0) + types.get('$pmux', 0), seconds=seconds)
    if kind == 'top-gate':
        log, netlist = LOGS / f'gate-{lanes}.log', LOGS / f'gate-{lanes}.json'
        seconds = yosys(f"read_verilog -sv {' '.join(RTL)}; chparam -set LANES {lanes} malleable_accelerator_top; "
                        "hierarchy -top malleable_accelerator_top; " + GATE.format(json=netlist), log, timeout)
        total, types = last_stat(log)
        mems = memories(netlist)
        netlist.unlink()  # large; the stat log is kept under build/
        return task, dict(gate_summary(types, total), memories=mems,
                          memory_bits=sum(m['bits'] for m in mems),
                          memory_read_ports=sum(m['read_ports'] for m in mems), seconds=seconds)
    if kind == 'datapath-gate':
        log, netlist = LOGS / f'datapath-{lanes}.log', LOGS / f'datapath-{lanes}.json'
        seconds = yosys(f"read_verilog -sv rtl/int8_dot_product.sv rtl/int8_tiled_accumulator.sv; "
                        f"chparam -set LANES {lanes} int8_tiled_accumulator; "
                        "hierarchy -top int8_tiled_accumulator; " + GATE.format(json=netlist), log, timeout)
        total, types = last_stat(log)
        netlist.unlink()
        return task, dict(gate_summary(types, total), seconds=seconds)
    raise ValueError(kind)


def digits_inputs():
    from digits_e2e import float_layers, train
    from malleable.quantize import quantize_model
    mlp, x_train, x_test, _, _ = train()
    q = quantize_model(float_layers(mlp), x_train.tolist())
    artifact = json.loads((ROOT / 'docs/results/digits/digits-int8.json').read_text())['artifact']
    if q.artifact.layers != artifact:
        raise RuntimeError('retrained digits model differs from the committed Stage 2 artifact')
    return ModelArtifact(artifact), [q.quantize_input(r.tolist()) for r in x_test]


def simulate(lanes):
    model, inputs = digits_inputs()
    run = RTLBackend().run(model, inputs, SimpleNamespace(lanes=lanes, active_lanes=lanes, reuse=True),
                           stream=tiler.emit(model, lanes))  # raises on any golden mismatch
    c = run['counters'][0]
    if any(r['cycles'] != c['cycles'] for r in run['counters']):
        raise RuntimeError('counters vary across requests')
    outputs = sum(len(l['weights']) for l in model.layers)
    cycles, tiles, macs, compute, controller = c['cycles'], c['tiles'], c['macs'], c['compute'], c['overhead']
    # Roofline decomposition. The derived terms use the Stage 4 calibration
    # (2 cycles/tile, 3 cycles/output); the counters give cycles, compute
    # (= tile accepts) and controller cycles.
    ideal = macs / lanes                 # cycles at the accept-only ceiling (LANES MAC/cycle)
    padding = tiles - ideal              # accept cycles spent on zero-filled lanes
    wait = tiles                         # accumulator registered-result wait, one per tile (derived)
    other = controller - wait            # ACC_START / WAIT_ACC / STORE per output (derived)
    gap = cycles - ideal
    return lanes, dict(
        requests=len(inputs), bit_exact=True, provenance=SIM,
        cycles=cycles, compute_cycles=compute, controller_cycles=controller, tiles=tiles, useful_macs=macs,
        util_accept=macs / (compute * lanes), util_wall=macs / (cycles * lanes),
        roofline=dict(
            achieved_macs_per_cycle=macs / cycles, accept_ceiling=lanes, tile_ceiling=lanes / 2,
            fraction_of_tile_ceiling=(macs / cycles) / (lanes / 2),
            gap_cycles_to_accept_ceiling=gap,
            gap_share=dict(padding=padding / gap if gap else 0.0,
                           accumulator_serialization=wait / gap if gap else 0.0,
                           controller_overhead=other / gap if gap else 0.0),
            controller_other_cycles=other, per_output_controller_cycles=other / outputs,
            provenance='derived:counters+stage4-calibration(2 cycles/tile)'))


def render_markdown(data):
    """Tables for docs/results/measurement-pack.md from the JSON."""
    b = data['builds']
    base = b[0]
    out = ['### Area proxies per build (ESTIMATED: Yosys generic mapping, not a target fit)', '',
           '| LANES | top logic gates | top flip-flops | top memory bits | top memory read ports | '
           'weight-memory read ports | lane datapath gates (tiled accumulator alone) | '
           'top coarse cells | logic vs LANES=1 |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in b:
        g = r['top_gate']
        weight = next((m['read_ports'] for m in g['memories'] if 'weight_memory' in m['name']), None)
        out.append(f"| {r['lanes']} | {g['logic_gates']} | {g['flip_flops']} | {g['memory_bits']} | "
                   f"{g['memory_read_ports']} | {weight} | {r['datapath_gate']['logic_gates']} | "
                   f"{r['top_coarse']['cells']} | {g['logic_gates'] / base['top_gate']['logic_gates']:.3f}x |")
    out += ['', '### Digits model per build (MEASURED: RTL simulation, 360 test inputs, bit-exact)', '',
            '| LANES | cycles | compute_cycles | controller_cycles | tiles | useful_macs | util_accept | '
            'util_wall | speedup vs LANES=1 |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in b:
        d = r['digits']
        out.append(f"| {r['lanes']} | {d['cycles']} | {d['compute_cycles']} | {d['controller_cycles']} | "
                   f"{d['tiles']} | {d['useful_macs']} | {d['util_accept']:.3f} | {d['util_wall']:.3f} | "
                   f"{base['digits']['cycles'] / d['cycles']:.2f}x |")
    out += ['', '### Roofline-style summary (digits; achieved from measured counters, split DERIVED '
            'with the Stage 4 calibration of 2 cycles/tile)', '',
            '| LANES | achieved MACs/cycle | 2-cycle/tile ceiling (L/2) | % of tile ceiling | accept ceiling (L) | '
            'gap to accept ceiling (cycles) | padding share | accumulator-serialization share | '
            'controller-overhead share | gates x cycles (ESTIMATED area-delay, millions) |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in b:
        f = r['digits']['roofline']
        share = f['gap_share']
        out.append(f"| {r['lanes']} | {f['achieved_macs_per_cycle']:.3f} | {f['tile_ceiling']:g} | "
                   f"{100 * f['fraction_of_tile_ceiling']:.1f} % | {f['accept_ceiling']} | "
                   f"{f['gap_cycles_to_accept_ceiling']:g} | {100 * share['padding']:.1f} % | "
                   f"{100 * share['accumulator_serialization']:.1f} % | "
                   f"{100 * share['controller_overhead']:.1f} % | "
                   f"{r['top_gate']['logic_gates'] * r['digits']['cycles'] / 1e6:.1f} |")
    return '\n'.join(out) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800, help='seconds per Yosys build (time box)')
    parser.add_argument('--builds', default=','.join(map(str, BUILDS)))
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(argv)
    table = OUT.with_name('measurement-pack-tables.md')
    if args.render_only:
        table.write_text(render_markdown(json.loads(OUT.read_text())))
        return 0
    builds = [int(b) for b in args.builds.split(',')]
    tasks = [(k, L, args.timeout) for k in ('datapath-gate', 'top-coarse', 'top-gate') for L in builds]
    synth, failures = {}, {}
    with ProcessPoolExecutor(args.jobs) as pool:
        sims = pool.map(simulate, builds)
        futures = {t: pool.submit(synthesize, t) for t in tasks}
        for t, f in futures.items():
            try:
                synth[t[:2]] = f.result()[1]
            except Exception as exc:  # time box or tool failure: recorded, never guessed
                failures[f'{t[0]}:{t[1]}'] = f'{type(exc).__name__}: {str(exc)[-500:]}'
        sim = dict(sims)
    tool = subprocess.run(['yosys', '-V'], capture_output=True, text=True).stdout.strip()
    rows = []
    for L in builds:
        gate = synth.get(('top-gate', L))
        area = (dict(cells=gate['cells'], logic_gates=gate['logic_gates'], flip_flops=gate['flip_flops'],
                     memory_bits=gate['memory_bits'], memory_read_ports=gate['memory_read_ports'],
                     provenance=EST) if gate else None)
        rows.append(dict(lanes=L, area=area, top_gate=gate, top_coarse=synth.get(('top-coarse', L)),
                         datapath_gate=synth.get(('datapath-gate', L)), digits=sim[L]))
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps(dict(schema_version=1, source=source, yosys=tool, max_dim=64, max_layers=4,
                                   area_provenance=EST, simulation_provenance=SIM, failures=failures,
                                   builds=rows), indent=1) + '\n')
    if not failures:
        table.write_text(render_markdown(json.loads(OUT.read_text())))
    for r in rows:
        g, d = r['top_gate'] or {}, r['digits']
        print(f"L={r['lanes']:2d} gates={g.get('logic_gates')} ffs={g.get('flip_flops')} "
              f"mem_bits={g.get('memory_bits')} dp_gates={(r['datapath_gate'] or {}).get('logic_gates')} "
              f"cycles={d['cycles']} util_wall={d['util_wall']:.3f}")
    print('failures:', failures or 'none')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
