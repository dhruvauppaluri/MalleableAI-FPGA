"""Tiler v1: build selection for the dense accelerator (folding at build time).

On one build, lane count is a single global runtime setting (config kind 6).
Tiles per output are ceil(input_count / active_lanes), so lowering lanes on a
fixed build can never reduce cycles. Stage 3 measured this: 0 of 13,136
per-layer mixes beat all-lanes (docs/results/lane-oracle.md). The lever is
therefore the *build*. As in FINN's PE/SIMD folding (Umuroglu et al., FPGA
2017; FINN-R, TRETS 2018) and DNNBuilder's parallelism search (Zhang et al.,
FPGA 2018), the tiler enumerates LANES builds, predicts cycles for each, and
picks the smallest build that meets a cycle budget. A smaller build trades
throughput for area. The tiler then emits the configuration-write stream for
that build from its hardware parameters, following the Gemmini pattern (Genc
et al., DAC 2021).

The cycle model is calibrated on Stage 3 RTL-simulation data
(``tiler_calibration.json``):
    cycles = a*tiles + b*outputs + c*layers + d,  tiles = sum_l n_l*ceil(k_l/LANES)
The model is fitted by least squares. Stage 3 data gives a=2 (tile accept plus
the accumulator's registered-result wait), b=3 (ACC_START, WAIT_ACC, STORE per
output), c=0 and d=0.

Fit checks mirror the RTL's ``configuration_valid`` and config-write checks in
``rtl/malleable_accelerator_top.sv``. A model that fails one raises
:class:`FitError` naming the exceeded limit, so no config is emitted that the
RTL would reject.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

from .backend import writes
from .records import ModelArtifact

CALIBRATION = Path(__file__).with_name('tiler_calibration.json')
AREA = Path(__file__).resolve().parents[1] / 'docs' / 'results' / 'data' / 'measurement-pack.json'
FEATURES = ('tiles', 'outputs', 'layers', 'constant')


class FitError(ValueError):
    """The model does not fit the build; ``limit`` names the exceeded RTL limit."""
    def __init__(self, limit, message):
        super().__init__(f'{limit}: {message}')
        self.limit = limit


@dataclass(frozen=True)
class Limits:
    """RTL parameters of ``malleable_accelerator_top`` (defaults = the instantiated build)."""
    MAX_LAYERS: int = 4
    MAX_DIM: int = 64
    WEIGHT_DEPTH: int | None = None   # default MAX_LAYERS * MAX_DIM * MAX_DIM, as in the RTL
    PARAM_DEPTH: int | None = None    # default MAX_LAYERS * MAX_DIM

    def resolved(self):
        return dict(MAX_LAYERS=self.MAX_LAYERS, MAX_DIM=self.MAX_DIM,
                    WEIGHT_DEPTH=self.WEIGHT_DEPTH or self.MAX_LAYERS * self.MAX_DIM * self.MAX_DIM,
                    PARAM_DEPTH=self.PARAM_DEPTH or self.MAX_LAYERS * self.MAX_DIM)


def _layers(model):
    return model.layers if isinstance(model, ModelArtifact) else list(model)


def shape(model):
    """[(k, n)] per layer for an artifact or a raw layer-dict list."""
    return [(len(l['weights'][0]) if l['weights'] else 0, len(l['weights'])) for l in _layers(model)]


def fit_check(model, lanes, limits=Limits()):
    """Raise FitError naming the first RTL limit or contract range the model violates.

    Accepts a ModelArtifact or a raw list of layer dicts (so oversize models can
    be diagnosed rather than rejected by ModelArtifact's own validation).
    """
    lim = limits.resolved()
    layers = _layers(model)
    if type(lanes) is not int or lanes < 1:
        raise FitError('LANES', f'build lane count must be a positive integer, got {lanes!r}')
    if lanes > 0xFFFF:
        raise FitError('LANES', f'{lanes} does not fit the 16-bit active_lanes register')
    if not 1 <= len(layers) <= lim['MAX_LAYERS']:
        raise FitError('MAX_LAYERS', f'model has {len(layers)} layers, build supports 1..{lim["MAX_LAYERS"]}')
    weight_total = param_total = 0
    previous = None
    for i, layer in enumerate(layers):
        n = len(layer['weights'])
        k = len(layer['weights'][0]) if n else 0
        if not 1 <= k <= lim['MAX_DIM']:
            raise FitError('MAX_DIM', f'layer {i} input_count {k} outside 1..{lim["MAX_DIM"]}')
        if not 1 <= n <= lim['MAX_DIM']:
            raise FitError('MAX_DIM', f'layer {i} output_count {n} outside 1..{lim["MAX_DIM"]}')
        if previous is not None and k != previous:
            raise FitError('ADJACENCY', f'layer {i} input_count {k} != layer {i-1} output_count {previous}')
        if any(len(row) != k for row in layer['weights']):
            raise FitError('SHAPE', f'layer {i} has ragged weight rows')
        weight_total += n * k
        param_total += n
        if weight_total > lim['WEIGHT_DEPTH']:
            raise FitError('WEIGHT_DEPTH', f'weights through layer {i} need {weight_total} > {lim["WEIGHT_DEPTH"]} entries')
        if param_total > lim['PARAM_DEPTH']:
            raise FitError('PARAM_DEPTH', f'outputs through layer {i} need {param_total} > {lim["PARAM_DEPTH"]} entries')
        for key, low, high in (('biases', -2**31, 2**31 - 1), ('multipliers', 1, 2**31 - 1),
                               ('shifts', 0, 62)):
            values = layer[key]
            if len(values) != n:
                raise FitError('SHAPE', f'layer {i} has {len(values)} {key} for {n} outputs')
            bad = [v for v in values if type(v) is not int or not low <= v <= high]
            if bad:
                raise FitError(key.upper(), f'layer {i} value {bad[0]!r} outside {low}..{high}')
        if any(type(w) is not int or not -128 <= w <= 127 for row in layer['weights'] for w in row):
            raise FitError('WEIGHTS', f'layer {i} has a weight outside signed INT8')
        if type(layer['relu']) is not bool:
            raise FitError('RELU', f'layer {i} relu must be boolean')
        previous = n
    return ModelArtifact(layers) if not isinstance(model, ModelArtifact) else model


def features(model, lanes):
    dims = shape(model)
    return dict(tiles=sum(n * math.ceil(k / lanes) for k, n in dims),
                outputs=sum(n for _, n in dims), layers=len(dims), constant=1)


def padding_waste(model, lanes):
    """Fraction of issued lane slots that carry zero-filled padding."""
    dims = shape(model)
    issued = sum(n * math.ceil(k / lanes) * lanes for k, n in dims)
    return 1 - sum(n * k for k, n in dims) / issued


def fit_cycle_model(samples):
    """Least squares over [(features_dict, cycles)] by normal equations (as optimize.Predictor)."""
    x = [[float(f[name]) for name in FEATURES] for f, _ in samples]
    y = [float(c) for _, c in samples]
    size = len(FEATURES)
    matrix = [[sum(r[i] * r[j] for r in x) + (1e-9 if i == j else 0) for j in range(size)]
              + [sum(r[i] * t for r, t in zip(x, y))] for i in range(size)]
    for i in range(size):
        pivot = max(range(i, size), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        divisor = matrix[i][i]
        matrix[i] = [v / divisor for v in matrix[i]]
        for j in range(size):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i])]
    coefficients = {name: row[-1] for name, row in zip(FEATURES, matrix)}
    residuals = [sum(coefficients[n] * r[i] for i, n in enumerate(FEATURES)) - t for r, t in zip(x, y)]
    return dict(coefficients=coefficients, samples=len(samples),
                max_abs_residual=max(abs(r) for r in residuals),
                rmse=math.sqrt(sum(r * r for r in residuals) / len(residuals)))


def load_calibration(path=CALIBRATION):
    return json.loads(Path(path).read_text())


def predict_cycles(model, lanes, calibration=None):
    calibration = calibration or load_calibration()
    f = features(model, lanes)
    # Round away least-squares float residue (~1e-12) so a budget equal to an
    # exactly predicted cycle count is met rather than missed by an ulp.
    return round(sum(calibration['coefficients'][name] * f[name] for name in FEATURES), 6)


def load_area(path=AREA):
    """Per-LANES ESTIMATED area proxies from the Stage 5 sweep, if it exists."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {int(b['lanes']): b['area'] for b in data.get('builds', []) if b.get('area')}


def _area_entry(area, lanes):
    if lanes in area:
        return dict(area[lanes], provenance='estimated:yosys-generic-mapping(stage5)')
    return dict(value=None, provenance='unavailable:no-stage5-synthesis-for-this-build')


def emit(model, lanes, limits=Limits()):
    """Configuration-write stream (kind, addr, data) for one build, as the backend formats it."""
    artifact = fit_check(model, lanes, limits)
    return writes(artifact) + [(6, 0, lanes)]


def plan_build(model, cycle_budget, lanes_range=range(1, 17), limits=Limits(), calibration=None,
               area=None, runner_ups=3):
    """Pick the smallest LANES build whose predicted cycles meet ``cycle_budget``.

    Raises FitError if the model fits no build, or ValueError naming the best
    achievable cycles if no build in range meets the budget.
    """
    calibration = calibration or load_calibration()
    area = load_area() if area is None else area
    if not (isinstance(cycle_budget, (int, float)) and cycle_budget > 0):
        raise ValueError(f'cycle budget must be positive, got {cycle_budget!r}')
    candidates, errors = [], {}
    for lanes in sorted(set(lanes_range)):
        try:
            fit_check(model, lanes, limits)
        except FitError as exc:
            errors[lanes] = str(exc)
            continue
        predicted = predict_cycles(model, lanes, calibration)
        candidates.append(dict(lanes=lanes, predicted_cycles=round(predicted, 3),
                               meets_budget=predicted <= cycle_budget,
                               tiles=features(model, lanes)['tiles'],
                               padding_waste=round(padding_waste(model, lanes), 6),
                               area=_area_entry(area, lanes)))
    if not candidates:
        first = next(iter(errors.values()))
        raise FitError(first.split(':')[0], f'model fits no build in range: {first}')
    feasible = [c for c in candidates if c['meets_budget']]
    if not feasible:
        best = min(candidates, key=lambda c: (c['predicted_cycles'], c['lanes']))
        raise ValueError(f'no build in LANES {min(c["lanes"] for c in candidates)}..'
                         f'{max(c["lanes"] for c in candidates)} meets {cycle_budget} cycles; '
                         f'best achievable is {best["predicted_cycles"]} cycles at LANES={best["lanes"]}')
    chosen = feasible[0]
    artifact = fit_check(model, chosen['lanes'], limits)
    return dict(
        schema_version=1, policy='smallest-build-meeting-cycle-budget (build-time folding)',
        model_id=artifact.model_id, shape=shape(artifact), cycle_budget=cycle_budget,
        chosen=dict(chosen, build=dict(LANES=chosen['lanes'], **limits.resolved())),
        runner_ups=[c for c in candidates if c['lanes'] != chosen['lanes'] and c['meets_budget']][:runner_ups],
        candidates=candidates, rejected_builds=errors,
        cycle_model=dict(formula='a*tiles + b*outputs + c*layers + d', **calibration),
        predicted_cycles_provenance='estimated:calibrated-tile-model(stage3 rtl-simulation)',
        config_writes=len(writes(artifact)) + 1)


def render_report(report):
    """Human-readable markdown for one plan_build report."""
    c = report['chosen']
    k = report['cycle_model']['coefficients']
    def term(value, name):
        value = round(value, 6) + 0.0  # drop -0.0
        return f"{value:g}*{name}" if name else f"{value:g}"
    formula = ' + '.join([term(k['tiles'], 'tiles'), term(k['outputs'], 'outputs'),
                          term(k['layers'], 'layers'), term(k['constant'], '')])
    lines = [f"- **Chosen build:** LANES={c['lanes']} (MAX_LAYERS={c['build']['MAX_LAYERS']}, "
             f"MAX_DIM={c['build']['MAX_DIM']}) for budget {report['cycle_budget']} cycles",
             f"- **Predicted cycles:** {c['predicted_cycles']:g} ({report['predicted_cycles_provenance']})",
             f"- **Padding waste:** {100 * c['padding_waste']:.2f} %",
             f"- **Cycle model:** {formula}",
             '', '| LANES | predicted cycles | meets budget | padding waste | area (cells) | area provenance |',
             '|---:|---:|:---:|---:|---:|---|']
    for row in report['candidates']:
        mark = ' (chosen)' if row['lanes'] == c['lanes'] else (
            ' (runner-up)' if any(r['lanes'] == row['lanes'] for r in report['runner_ups']) else '')
        cells = row['area'].get('cells')
        lines.append(f"| {row['lanes']}{mark} | {row['predicted_cycles']:g} | {'yes' if row['meets_budget'] else 'no'} | "
                     f"{100 * row['padding_waste']:.2f} % | {cells if cells is not None else 'unavailable'} | "
                     f"{row['area']['provenance']} |")
    return '\n'.join(lines) + '\n'


def as_dict(limits):
    return asdict(limits)
