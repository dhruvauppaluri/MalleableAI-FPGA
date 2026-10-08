"""Offline policy candidate. Decisions never authorize hardware programming.

Inputs are externally verified quality/capacity eligibility and calibrated
service intervals. Transition upper bounds must include all listed components.
This schema deliberately cannot load legacy DQN checkpoints.
"""
import math

from .optimization import objective_cost

COMPONENTS = frozenset(('drain', 'program', 'reload', 'initialize',
                        'reprefill', 'warmup', 'controller'))


def _number(value, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError('finite nonnegative cost required')
    return value


def recommend(current, intervals, transitions, eligible, window, *,
              platform_id, context_id, margin=.05, minimum_residence=1):
    """Compare candidate upper cost against current lower cost.

    intervals map action to (lower, upper) cycles. Transition records bind
    source/destination image identities via action keys, platform and context.
    Assumed costs are rejected. This is a recommendation, never activation.
    Callers must supply matching image identities and verified eligibility.
    """
    _number(margin)
    if margin >= 1 or type(minimum_residence) is not int or minimum_residence < 1:
        raise ValueError('invalid margin or residence')
    if not platform_id or not context_id or current not in eligible:
        raise ValueError('current eligibility and platform/context required')
    for bounds in intervals.values():
        if len(bounds) != 2 or _number(bounds[0], True) > _number(bounds[1], True):
            raise ValueError('ordered positive service interval required')
    if current not in intervals:
        raise ValueError('current service interval required')
    baseline = objective_cost(intervals[current][0], window)
    chosen, best = current, baseline
    rejected = {}
    for action in sorted(intervals):
        if action == current:
            continue
        reason = None
        record = transitions.get((current, action))
        if action not in eligible:
            reason = 'quality-or-capacity'
        elif window.residence_windows < minimum_residence:
            reason = 'minimum-residence'
        elif not record or record.get('provenance') != 'measured-physical-upper-bound':
            reason = 'missing-measured-transition'
        elif (record.get('platform_id'), record.get('context_id')) != (platform_id, context_id):
            reason = 'transition-lineage-mismatch'
        elif not record.get('evidence_id'):
            reason = 'missing-transition-evidence'
        else:
            components = record.get('components', {})
            if set(components) != COMPONENTS:
                raise ValueError('complete transition components required')
            overhead = sum(_number(v) for v in components.values())
            upper = objective_cost(intervals[action][1], window, overhead)
            if upper >= baseline * (1 - margin):
                reason = 'insufficient-conservative-gain'
            elif upper < best:
                chosen, best = action, upper
        if reason:
            rejected[action] = reason
    return dict(schema_version=1, policy='conservative-interval-v1',
                recommended_action=chosen, current_action=current,
                current_lower_cost=baseline, recommended_upper_cost=best if chosen != current else None,
                rejected=rejected, automatic_switching_allowed=False)
