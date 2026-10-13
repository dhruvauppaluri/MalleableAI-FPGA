"""Validate board-driver stage timestamps and derive observed upper costs.

The driver must export board-clock timestamps for a whole transition. Host
wall-clock build time and RTL estimates cannot enter this record.
"""
import math
from .conservative_policy import COMPONENTS
from ..records import identity


def aggregate(samples, *, margin_fraction=.1):
    if type(margin_fraction) not in (int,float) or not math.isfinite(margin_fraction) or margin_fraction < 0:
        raise ValueError('invalid measurement margin')
    if len(samples) < 3:
        raise ValueError('at least three independently recorded transitions required')
    fields = ('platform_id','context_id','source_image_id','destination_image_id','clock_hz')
    anchor = samples[0]
    if any(not anchor.get(k) for k in fields) or anchor['source_image_id'] == anchor['destination_image_id']:
        raise ValueError('complete matched board/image identity required')
    if type(anchor['clock_hz']) not in (int,float) or not math.isfinite(anchor['clock_hz']) or anchor['clock_hz'] <= 0:
        raise ValueError('positive board clock required')
    ids, totals = set(), []
    per_component = {name:[] for name in COMPONENTS}
    for sample in samples:
        if any(sample.get(k) != anchor[k] for k in fields) or sample.get('provenance') != 'measured-physical-board-clock':
            raise ValueError('mixed/assumed transition evidence')
        if not sample.get('driver_id') or not sample.get('trace_id') or sample['trace_id'] in ids:
            raise ValueError('independent trace and driver identity required')
        ids.add(sample['trace_id'])
        stamps = sample.get('stage_boundaries', {})
        if set(stamps) != COMPONENTS:
            raise ValueError('all transition components required')
        durations = {}
        for name, bounds in stamps.items():
            if (not isinstance(bounds,list) or len(bounds)!=2
                    or any(type(v) is not int or v < 0 for v in bounds) or bounds[1] < bounds[0]):
                raise ValueError('ordered board-clock timestamps required')
            durations[name] = bounds[1]-bounds[0]
            per_component[name].append(durations[name])
        ordered = sorted(stamps.values())
        if any(a[1] != b[0] for a,b in zip(ordered, ordered[1:])):
            raise ValueError('noncontiguous or overlapping stage boundaries')
        totals.append(sum(durations.values()))
    # Sum component maxima is conservative for the observed traces; the margin
    # is a heuristic reserve, not a distribution-free tail guarantee.
    components = {name:math.ceil(max(values)*(1+margin_fraction)) for name,values in per_component.items()}
    return dict(provenance='measured-physical-upper-bound', platform_id=anchor['platform_id'],
        context_id=anchor['context_id'], source_image_id=anchor['source_image_id'],
        destination_image_id=anchor['destination_image_id'], clock_hz=anchor['clock_hz'],
        sample_count=len(samples), evidence_id=identity(samples), components=components,
        upper_method='component-max-plus-margin-heuristic', observed_total_min=min(totals),
        observed_total_max=max(totals), tail_coverage_claim=False)
