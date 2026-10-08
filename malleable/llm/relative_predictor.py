"""Grouped ridge models for log balanced cycles and log relative costs.

Training-only standardization and held-apart group-max residual calibration.
Intervals are experimental: coverage assumes exchangeable workload groups and
does not establish coverage on unseen model families or physical hardware.
"""
import math
import numpy as np

from .policy_features import FEATURES, features
from ..records import identity

SCHEMA = 'architecture-relative-ridge-v1'


def _validate(rows):
    groups = {}
    for row in rows:
        features(row['features'])
        if (not row.get('group_id') or not row.get('evidence_id')
                or row.get('provenance') != 'measured-rtl'
                or type(row['cycles']) not in (int, float)
                or not math.isfinite(row['cycles']) or row['cycles'] <= 0):
            raise ValueError('grouped finite measured RTL evidence required')
        group = groups.setdefault(row['group_id'], {})
        if row['action'] in group:
            raise ValueError('duplicate group/action')
        group[row['action']] = row
    if not groups or any('balanced' not in g for g in groups.values()):
        raise ValueError('each workload needs a paired balanced baseline')
    for group in groups.values():
        if len({r['features']['model_id'] for r in group.values()}) != 1:
            raise ValueError('paired rows must share model identity')
    return groups


def _fit(x, y, penalty):
    x = np.log1p(np.asarray(x, dtype=float))
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    regularizer = np.eye(design.shape[1])*penalty
    regularizer[0, 0] = 0
    coefficients = np.linalg.solve(design.T@design+regularizer, design.T@np.asarray(y))
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist())


def _predict(model, x):
    z = (np.log1p(x)-model['mean'])/model['scale']
    return float(np.dot(np.r_[1, z], model['coefficients']))


def fit(training, calibration, *, alpha=.1, penalty=1.):
    if not 0 < alpha < 1 or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError('valid calibration alpha and ridge penalty required')
    train_groups, cal_groups = _validate(training), _validate(calibration)
    if set(train_groups) & set(cal_groups) or {r['evidence_id'] for r in training} & {r['evidence_id'] for r in calibration}:
        raise ValueError('training/calibration leakage')
    actions = set(next(iter(train_groups.values())))
    if any(set(g) != actions for g in [*train_groups.values(), *cal_groups.values()]):
        raise ValueError('complete matched action coverage required')
    models = {}
    for action in sorted(actions):
        x, y = [], []
        for group in train_groups.values():
            row = group[action]
            x.append(features(row['features']))
            y.append(math.log(row['cycles'] if action == 'balanced'
                              else row['cycles']/group['balanced']['cycles']))
        models[action] = _fit(x, y, penalty)
    candidate = dict(schema_version=1, schema=SCHEMA, feature_names=list(FEATURES),
        models=models, alpha=alpha, penalty=penalty, training_groups=sorted(train_groups),
        calibration_groups=sorted(cal_groups), training_evidence=sorted(r['evidence_id'] for r in training),
        calibration_evidence=sorted(r['evidence_id'] for r in calibration), automatic_switching_allowed=False)
    errors = []
    for group in cal_groups.values():
        predictions = point(candidate, {a:r['features'] for a,r in group.items()})
        errors.append(max(abs(math.log(group[a]['cycles']/predictions[a])) for a in actions))
    rank = math.ceil((len(errors)+1)*(1-alpha))
    # A finite split-conformal quantile cannot be obtained with too few groups.
    candidate['log_radius'] = sorted(errors)[rank-1] if rank <= len(errors) else None
    candidate['calibration_group_count'] = len(errors)
    candidate['status'] = 'candidate' if candidate['log_radius'] is not None else 'insufficient-calibration-groups'
    return candidate


def point(checkpoint, records):
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('feature_names') != list(FEATURES):
        raise ValueError('incompatible predictor checkpoint')
    if set(records) != set(checkpoint['models']) or 'balanced' not in records:
        raise ValueError('matched candidate features required')
    baseline = math.exp(_predict(checkpoint['models']['balanced'], features(records['balanced'])))
    return {a:baseline if a == 'balanced' else baseline*math.exp(_predict(checkpoint['models'][a], features(r)))
            for a,r in records.items()}


def intervals(checkpoint, records):
    radius = checkpoint.get('log_radius')
    if radius is None or not math.isfinite(radius) or radius < 0:
        raise ValueError('insufficient calibrated uncertainty')
    return {a:(v*math.exp(-radius), v*math.exp(radius)) for a,v in point(checkpoint, records).items()}
