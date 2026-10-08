"""Durable exactly-once claims for a frozen offline campaign.

Claim before dispatch. Missing results and failed attempts stay consumed.
The caller must publish claims before remote execution; this module alone
does not coordinate independent clones or authorize dispatch.
"""
import json
import os
from pathlib import Path

from malleable.records import identity


def validate_manifest(manifest, consumed_rows):
    """Reject a relabelled published tape, including on another model."""
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'frozen':
        raise ValueError('frozen versioned campaign required')
    old_tapes={r['input_token_hash'] for r in consumed_rows}
    old_workloads={r['workload_identity_v2'] for r in consumed_rows}
    cases=manifest.get('cases',[])
    if not cases or len({identity(c) for c in cases}) != len(cases):
        raise ValueError('nonempty distinct cases required')
    groups={}
    for case in cases:
        if (case.get('input_token_hash') != identity(case.get('input_tokens'))
                or case['input_token_hash'] in old_tapes
                or case.get('workload_identity_v2') in old_workloads):
            raise ValueError('consumed tape or workload identity')
        if case.get('split') != 'development-pilot' or case.get('wformat') != 'int8':
            raise ValueError('pilot split and approved numeric format required')
        group=(case['base_model_id'],case['input_token_hash'],case['memory_scenario'])
        groups.setdefault(group,set()).add(case['personality'])
    if any(v != {'compact','balanced','compute','buffered'} for v in groups.values()):
        raise ValueError('complete paired personality groups required')
    return True


def claim(root, manifest, case_id, *, source_id):
    if manifest.get('schema_version') != 1 or not source_id:
        raise ValueError('versioned manifest and source identity required')
    if manifest.get('status') != 'frozen':
        raise ValueError('campaign must be frozen before dispatch')
    cases = manifest.get('cases', [])
    ids = [identity(case) for case in cases]
    if len(ids) != len(set(ids)) or case_id not in ids:
        raise ValueError('unknown or duplicate campaign case')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = dict(schema_version=1, manifest_id=identity(manifest),
                  case_id=case_id, source_id=source_id, status='consumed-before-dispatch')
    # Case identity deliberately excludes manifest identity: renaming a
    # campaign must not allow the same case to be dispatched again.
    with (root / (case_id + '.json')).open('x') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return record
