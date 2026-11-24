#!/usr/bin/env python3
"""Stage 3: oracle bound on what runtime lane choice can do on this RTL.

All numbers are RTL simulation results (Icarus): provenance measured:rtl-simulation.

For each model and each compiled build LANES in --builds, the script runs:
  * the whole model at each fixed active_lanes 1..LANES (one job per request), and
  * each layer as its own single-layer job at each active_lanes, chained by
    feeding layer i's RTL outputs into layer i+1.
The hardware counters reset per job. Host copying between layer jobs is NOT
measured by the counters, so per-layer mixes understate real cost.

Analysis:
  1. consistency: whole-model cycles minus the per-layer job cycle sum, per
     lane setting (must be constant to compose per-layer data additively);
  2. oracle: exhaustive best per-layer lane mix vs the all-lanes whole model,
     with the oracle mix rerun in RTL as chained jobs and checked bit-exact;
  3. ties: per-layer lane values whose cycles equal all-lanes.

Synthetic shapes are fixed here, before any measurement, and must not be tuned.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.backend import RTLBackend  # noqa: E402
from malleable.model import reference  # noqa: E402
from malleable.records import ModelArtifact  # noqa: E402

PROVENANCE = 'measured:rtl-simulation(icarus)'
REQUESTS = 4
SEED = 0
# Pre-registered synthetic shapes (do not tune; see docs/results/lane-oracle.md).
SYNTHETIC = {
    'syn-aligned': [64, 64, 64, 64, 64],
    'syn-odd': [61, 47, 29, 13, 5],
    'syn-mixed': [5, 64, 3, 64, 2],
}
DIGITS = ROOT / 'docs' / 'results' / 'digits' / 'digits-int8.json'


def synthetic(dims, seed=SEED):
    """Seeded random INT8 model in the style of malleable.model.fixture."""
    rng = random.Random(f'{seed}:{dims}')
    return ModelArtifact([dict(weights=[[rng.randint(-128, 127) for _ in range(k)] for _ in range(n)],
                               biases=[rng.randint(-1000, 1000) for _ in range(n)],
                               multipliers=[rng.randint(1, 16) for _ in range(n)],
                               shifts=[rng.randint(10, 15) for _ in range(n)],
                               relu=i < len(dims) - 2)
                          for i, (k, n) in enumerate(zip(dims, dims[1:]))])


def load_models():
    models = {'digits': ModelArtifact(json.loads(DIGITS.read_text())['artifact'])}
    models.update({name: synthetic(dims) for name, dims in SYNTHETIC.items()})
    return models


def inputs_for(model, seed=SEED):
    rng = random.Random(f'inputs:{seed}:{model.model_id}')
    return [[rng.randint(-128, 127) for _ in model.layers[0]['weights'][0]] for _ in range(REQUESTS)]


def layer_model(model, i):
    return ModelArtifact([model.layers[i]])


def config(lanes, active):
    return SimpleNamespace(lanes=lanes, active_lanes=active, reuse=True)


def counters_of(run):
    rows = run['counters']
    first = {k: rows[0][k] for k in ('cycles', 'tiles', 'macs', 'compute', 'overhead')}
    identical = all({k: r[k] for k in first} == first for r in rows)
    return dict(cycles=first['cycles'], compute_cycles=first['compute'],
                controller_cycles=first['overhead'], tiles=first['tiles'],
                useful_macs=first['macs'], identical_across_requests=identical)


def add_util(row, lanes):
    row['util_accept'] = row['useful_macs'] / (row['compute_cycles'] * lanes)
    row['util_wall'] = row['useful_macs'] / (row['cycles'] * lanes)
    return row


def run_whole(task):
    name, lanes, active = task
    model = load_models()[name]
    run = RTLBackend().run(model, inputs_for(model), config(lanes, active))
    return task, add_util(counters_of(run), lanes), run['outputs']


def run_chain(task):
    """Single-layer jobs chained through the host; lane_mix gives each layer's active lanes."""
    name, lanes, lane_mix = task
    model = load_models()[name]
    values, rows = inputs_for(model), []
    for i, active in enumerate(lane_mix):
        single = layer_model(model, i)
        run = RTLBackend().run(single, values, config(lanes, active))  # raises on mismatch
        rows.append(add_util(counters_of(run), lanes))
        values = run['outputs']
    return task, rows, values


def summarize(models, builds, whole, chains):
    report = []
    for name, model in models.items():
        dims = [len(model.layers[0]['weights'][0])] + [len(l['weights']) for l in model.layers]
        golden = [reference(model, x, 1)[0] for x in inputs_for(model)]
        for lanes in builds:
            fixed = {a: whole[(name, lanes, a)] for a in range(1, lanes + 1)}
            per_layer = {a: chains[(name, lanes, (a,) * len(model.layers))][0]
                         for a in range(1, lanes + 1)}
            delta = {a: fixed[a][0]['cycles'] - sum(r['cycles'] for r in per_layer[a])
                     for a in fixed}
            consistent = len(set(delta.values())) == 1
            d = next(iter(delta.values()))
            layer_cycles = [[per_layer[a][i]['cycles'] for a in range(1, lanes + 1)]
                            for i in range(len(model.layers))]
            mixes = []
            for mix in itertools.product(range(1, lanes + 1), repeat=len(model.layers)):
                mixes.append((sum(layer_cycles[i][a - 1] for i, a in enumerate(mix)) + d, mix))
            best_cycles, best_mix = min(mixes, key=lambda m: (m[0], [-a for a in m[1]]))
            all_lanes = fixed[lanes][0]['cycles']
            best_fixed_a = min(fixed, key=lambda a: (fixed[a][0]['cycles'], -a))
            beating = [m for m in mixes if m[0] < all_lanes]
            tying_mixes = sum(1 for m in mixes if m[0] == all_lanes)
            layer_ties = [[a for a in range(1, lanes) if layer_cycles[i][a - 1] == layer_cycles[i][lanes - 1]]
                          for i in range(len(model.layers))]
            oracle_rows, oracle_out = chains[(name, lanes, best_mix)]
            oracle_measured = sum(r['cycles'] for r in oracle_rows) + d
            report.append(dict(
                model=name, dims=dims, build_lanes=lanes,
                fixed=[dict(active_lanes=a, **fixed[a][0]) for a in range(1, lanes + 1)],
                per_layer=[[dict(layer=i, k=dims[i], n=dims[i + 1], active_lanes=a, **per_layer[a][i])
                            for a in range(1, lanes + 1)] for i in range(len(model.layers))],
                whole_minus_layer_sum=delta, additive_consistent=consistent,
                outputs_bit_exact_fixed=all(fixed[a][1] == golden for a in fixed),
                oracle=dict(mix=list(best_mix), predicted_cycles=best_cycles,
                            measured_chained_cycles_plus_delta=oracle_measured,
                            chained_outputs_bit_exact=oracle_out == golden),
                all_lanes_cycles=all_lanes, best_fixed_active_lanes=best_fixed_a,
                best_fixed_cycles=fixed[best_fixed_a][0]['cycles'],
                mixes_evaluated=len(mixes), mixes_strictly_beating_all_lanes=len(beating),
                mixes_tying_all_lanes=tying_mixes, per_layer_lane_values_tying_all_lanes=layer_ties,
                provenance=PROVENANCE))
    return report


def render_markdown(report):
    """Model x strategy table (all values simulated RTL)."""
    lines = ['| model | build LANES | strategy | cycles | compute_cycles | controller_cycles | tiles | '
             'useful_macs | util_accept | util_wall | vs all-lanes |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report:
        base = r['all_lanes_cycles']
        rows = [(f"fixed {f['active_lanes']}", f) for f in r['fixed']]
        mix = r['oracle']['mix']
        layers = [r['per_layer'][i][a - 1] for i, a in enumerate(mix)]
        agg = {k: sum(x[k] for x in layers) for k in
               ('cycles', 'compute_cycles', 'controller_cycles', 'tiles', 'useful_macs')}
        agg['cycles'] += int(next(iter(r['whole_minus_layer_sum'].values())))
        agg['util_accept'] = agg['useful_macs'] / (agg['compute_cycles'] * r['build_lanes'])
        agg['util_wall'] = agg['useful_macs'] / (agg['cycles'] * r['build_lanes'])
        rows.append((f"per-layer oracle {mix}", agg))
        for label, f in rows:
            lines.append(f"| {r['model']} | {r['build_lanes']} | {label} | {f['cycles']} | "
                         f"{f['compute_cycles']} | {f['controller_cycles']} | {f['tiles']} | "
                         f"{f['useful_macs']} | {f['util_accept']:.3f} | {f['util_wall']:.3f} | "
                         f"{f['cycles'] / base:.3f}x |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--builds', default='4,8', help='compiled LANES values')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'results' / 'data' / 'lane-oracle.json'))
    parser.add_argument('--render-only', action='store_true',
                        help='rewrite the markdown table from an existing --out JSON')
    args = parser.parse_args(argv)
    table = Path(args.out).with_name('lane-oracle-table.md')
    if args.render_only:
        table.write_text(render_markdown(json.loads(Path(args.out).read_text())['results']))
        return 0
    builds = [int(x) for x in args.builds.split(',')]
    models = load_models()

    whole_tasks = [(n, L, a) for n in models for L in builds for a in range(1, L + 1)]
    chain_tasks = [(n, L, (a,) * len(m.layers)) for n, m in models.items()
                   for L in builds for a in range(1, L + 1)]
    with ProcessPoolExecutor(args.jobs) as pool:
        whole = {t: (row, out) for t, row, out in pool.map(run_whole, whole_tasks)}
        chains = {t: (rows, out) for t, rows, out in pool.map(run_chain, chain_tasks)}
        # Oracle mixes need the uniform per-layer data first; compute and run them.
        provisional = {}
        for n, m in models.items():
            for L in builds:
                lc = [[chains[(n, L, (a,) * len(m.layers))][0][i]['cycles'] for a in range(1, L + 1)]
                      for i in range(len(m.layers))]
                best = min(itertools.product(range(1, L + 1), repeat=len(m.layers)),
                           key=lambda mix: (sum(lc[i][a - 1] for i, a in enumerate(mix)), [-a for a in mix]))
                provisional[(n, L, tuple(best))] = None
        missing = [t for t in provisional if t not in chains]
        chains.update({t: (rows, out) for t, rows, out in pool.map(run_chain, missing)})

    report = summarize(models, builds, whole, chains)
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema_version=1, provenance=PROVENANCE, source=source,
                                   requests_per_job=REQUESTS, seed=SEED, synthetic=SYNTHETIC,
                                   results=report), indent=1) + '\n')
    table.write_text(render_markdown(report))
    ok = True
    for r in report:
        print(f"{r['model']:12s} L={r['build_lanes']} all-lanes={r['all_lanes_cycles']:6d} "
              f"oracle={r['oracle']['predicted_cycles']:6d} mix={r['oracle']['mix']} "
              f"delta={sorted(set(r['whole_minus_layer_sum'].values()))} "
              f"beat={r['mixes_strictly_beating_all_lanes']} tie={r['mixes_tying_all_lanes']} "
              f"exact={r['outputs_bit_exact_fixed'] and r['oracle']['chained_outputs_bit_exact']}")
        ok &= (r['additive_consistent'] and r['outputs_bit_exact_fixed']
               and r['oracle']['chained_outputs_bit_exact']
               and all(row['identical_across_requests'] for row in r['fixed']))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
