import json
import pytest
from tools.run_remaining_benchmarks import window

def test_benchmark_window_cannot_be_reset_or_extended(tmp_path):
    (tmp_path/'authorization.json').write_text(json.dumps({'mode':'overnight','window_hours':8}))
    first=window(tmp_path,now=100)
    assert first['dispatch_deadline_unix']==28900
    assert window(tmp_path,now=500)==first
    bad=dict(first,dispatch_deadline_unix=29000)
    (tmp_path/'control.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='never reset'):window(tmp_path,now=500)

def test_benchmark_window_rejects_changed_authorization(tmp_path):
    (tmp_path/'authorization.json').write_text('{}')
    window(tmp_path,now=100)
    (tmp_path/'authorization.json').write_text('{"window_hours":16}')
    with pytest.raises(ValueError):window(tmp_path,now=500)
