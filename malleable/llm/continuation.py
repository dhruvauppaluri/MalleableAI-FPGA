"""One persisted Overnight dispatch deadline shared by all continuation runners."""
import json
from pathlib import Path
import time
CONTROL=Path(__file__).resolve().parents[2]/'build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930/control.json'

def read_control(declared_deadline=None):
    if not CONTROL.is_file():raise ValueError('start the authorized first diagnostic to establish the continuation deadline')
    data=json.loads(CONTROL.read_text())
    if (data.get('schema_version')!=1 or data.get('mode')!='overnight'
        or not data.get('authorization') or data.get('dispatch_deadline_unix')!=data.get('started_unix',0)+8*3600):
        raise ValueError('invalid persisted continuation window')
    if declared_deadline is not None and declared_deadline!=data['dispatch_deadline_unix']:
        raise ValueError('dispatch deadline must match the original continuation control')
    if time.time()>=data['dispatch_deadline_unix']:raise ValueError('continuation dispatch deadline expired')
    return data
