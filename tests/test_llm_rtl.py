"""Release-critical synthetic sequences; intentionally no real checkpoint download."""
import json
import numpy as np
import pytest
from malleable.llm.backend import CheckedRtlBackend
from malleable.llm.records import PERSONALITIES,GenerationWorkload
from malleable.llm import upstream
from opentpu.llm.qwen3 import Engine
from llm_fixture import tiny

@pytest.mark.parametrize('personality',list(PERSONALITIES))
@pytest.mark.parametrize('wformat',['int8','int4','fp4'])
def test_personality_quantization_exact(tmp_path,personality,wformat):
    _,weights,spec=tiny()
    workload=GenerationWorkload('fixture',context=128,personality=personality,wformat=wformat,stall_percent=30)
    cfg=PERSONALITIES[personality].config(spec,128,wformat)
    rtl=Engine(spec,weights,cap=128,cfg=cfg,rows=1,pipeline=False,wformat=wformat,head_format='int8',
        backend=lambda c,i:CheckedRtlBackend(c,i,workload,tmp_path))
    ref=Engine(spec,weights,cap=128,cfg=cfg,backend='isa',rows=1,pipeline=False,wformat=wformat,head_format='int8')
    for token in (0,127,3): assert np.array_equal(rtl.step(token),ref.step(token))
    rtl.reset(); ref.reset()
    assert np.array_equal(rtl.step(0),ref.step(0))

def test_hundred_full_rtl_tokens(tmp_path):
    _,weights,spec=tiny(); workload=GenerationWorkload('fixture',context=128,seed=9)
    cfg=PERSONALITIES['balanced'].config(spec,128)
    engine=Engine(spec,weights,cap=128,cfg=cfg,rows=1,pipeline=False,
        backend=lambda c,i:CheckedRtlBackend(c,i,workload,tmp_path))
    for token in np.random.default_rng(9).integers(0,128,100): engine.step(int(token))
    assert len(engine.stats)==100 and all(s['validation']=='bit-exact-dram-and-tmem' for s in engine.stats)
    (tmp_path/'tiny-release.json').write_text(json.dumps({'backend':'rtl','validated_steps':100,'bit_exact':True}))
