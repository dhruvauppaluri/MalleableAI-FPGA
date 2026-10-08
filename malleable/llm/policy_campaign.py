"""Durable exactly-once claims for a frozen offline campaign.

Claim before dispatch. Missing results and failed attempts stay consumed.
The caller must publish claims before remote execution; this module alone
does not coordinate independent clones or authorize dispatch.
"""
import json
import os
from pathlib import Path

from malleable.records import identity


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
