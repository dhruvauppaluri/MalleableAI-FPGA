#!/usr/bin/env python3
"""Stage 4: calibrate the tiler's cycle model and verify tiler-emitted builds in RTL.

1. --calibrate: fit cycles = a*tiles + b*outputs + c*layers + d by least squares on
   every measured row of docs/results/data/lane-oracle.json (Stage 3 RTL
   simulation; builds 4/8 only) and write malleable/tiler_calibration.json.
2. For the Stage 2 digits model and the Stage 3 synthetic shapes, at every
   build LANES 1..16, drive Icarus RTL with the tiler-emitted config stream
   (RTLBackend(stream=...)). Check bit-exactness against model.reference, then
   tabulate predicted vs measured cycles. Builds 1-3 and 9-16 are out of sample
   for the calibration.
3. Sample plan_build reports at fixed budgets.

Measured cycles are RTL simulation (measured:rtl-simulation). Predicted cycles
are estimated:calibrated-tile-model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from lane_oracle import inputs_for, load_models  # noqa: E402
from malleable import tiler  # noqa: E402
from malleable.backend import RTLBackend  # noqa: E402

ORACLE = ROOT / 'docs' / 'results' / 'data' / 'lane-oracle.json'
DATA = ROOT / 'docs' / 'results' / 'data' / 'tiler-verify.json'
SAMPLE = ROOT / 'docs' / 'results' / 'tiler-sample-report.json'
SAMPLE_MD = ROOT / 'docs' / 'results' / 'data' / 'tiler-sample-reports.md'


def calibrate():
    data = json.loads(ORACLE.read_text())
    samples = []
    for r in data['results']:
        dims = r['dims']
        for f in r['fixed']:
            feats = dict(tiles=f['tiles'], outputs=sum(dims[1:]), layers=len(dims) - 1, constant=1)
            samples.append((feats, f['cycles']))
        for layer in r['per_layer']:
            for f in layer:
                samples.append((dict(tiles=f['tiles'], outputs=f['n'], layers=1, constant=1), f['cycles']))
    fit = tiler.fit_cycle_model(samples)
    fit.update(source='docs/results/data/lane-oracle.json (Stage 3, Icarus RTL simulation, builds LANES 4 and 8)',
               source_commit=data['source'], provenance='estimated:least-squares-on-rtl-simulation')
    tiler.CALIBRATION.write_text(json.dumps(fit, indent=1) + '\n')
    return fit


def verify(task):
    name, lanes = task
    model = load_models()[name]
    stream = tiler.emit(model, lanes)
    run = RTLBackend().run(model, inputs_for(model),
                           SimpleNamespace(lanes=lanes, active_lanes=lanes, reuse=True),
                           stream=stream)  # raises on any RTL/golden mismatch
    c = run['counters'][0]
    predicted = tiler.predict_cycles(model, lanes)
    return dict(model=name, lanes=lanes, measured_cycles=c['cycles'], compute_cycles=c['compute'],
                controller_cycles=c['overhead'], tiles=c['tiles'], useful_macs=c['macs'],
                util_accept=c['macs'] / (c['compute'] * lanes), util_wall=c['macs'] / (c['cycles'] * lanes),
                predicted_cycles=predicted, error_cycles=predicted - c['cycles'],
                error_pct=100 * (predicted - c['cycles']) / c['cycles'],
                bit_exact=True, stream_writes=len(stream), in_calibration_sample=lanes in (4, 8),
                identical_across_requests=all(r['cycles'] == c['cycles'] for r in run['counters']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--max-lanes', type=int, default=16)
    args = parser.parse_args(argv)
    if args.calibrate or not tiler.CALIBRATION.exists():
        print('calibration:', json.dumps(calibrate()))
    models = load_models()
    tasks = [(n, L) for n in models for L in range(1, args.max_lanes + 1)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(verify, tasks))

    # Sample budgets fixed by rule, not by result: the all-lanes cycles of the
    # LANES=4 and LANES=8 builds measured in Stage 3, and an infeasible budget of 1.
    samples, md = {}, []
    for name, model in models.items():
        measured = {r['lanes']: r['measured_cycles'] for r in rows if r['model'] == name}
        for label, budget in (('budget=measured LANES=4', measured[4]),
                              ('budget=measured LANES=8', measured[8]), ('budget=1 (infeasible)', 1)):
            try:
                report = tiler.plan_build(model, budget)
                samples.setdefault(name, []).append(dict(label=label, report=report))
                md += [f'### {name}: {label}', '', tiler.render_report(report)]
            except ValueError as exc:
                samples.setdefault(name, []).append(dict(label=label, error=str(exc)))
                md += [f'### {name}: {label}', '', f'Rejected: `{exc}`', '']
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    DATA.write_text(json.dumps(dict(schema_version=1, source=source,
                                    measured_provenance='measured:rtl-simulation(icarus)',
                                    predicted_provenance='estimated:calibrated-tile-model',
                                    calibration=tiler.load_calibration(), rows=rows), indent=1) + '\n')
    SAMPLE.write_text(json.dumps(dict(schema_version=1, source=source, samples=samples), indent=1) + '\n')
    SAMPLE_MD.write_text('\n'.join(md))
    worst = max(rows, key=lambda r: abs(r['error_cycles']))
    print(f"rows={len(rows)} all bit-exact={all(r['bit_exact'] for r in rows)} "
          f"max |error|={abs(worst['error_cycles']):.6f} cycles ({worst['model']} L={worst['lanes']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
