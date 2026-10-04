"""Durable 100-token synthetic release evidence; no pretrained downloads."""
import argparse
import json
import platform
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from llm_fixture import save
from malleable.llm.backend import CheckedRtlBackend
from malleable.llm.records import PERSONALITIES,GenerationWorkload
from malleable.llm.models import inspect,digest
from malleable.llm import upstream
from malleable.records import identity
from malleable.store import Store
from opentpu.llm.qwen3 import Engine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    root=Path(a.output).resolve()
    if root.exists(): raise ValueError('choose a new evidence directory; historical evidence is immutable')
    root.mkdir(parents=True)
    _,weights,spec=save(root/'model'); info=inspect(root/'model',128)
    w=GenerationWorkload('synthetic correctness tape',context=128,seed=9)
    cfg=PERSONALITIES['balanced'].config(spec,128)
    engine=Engine(spec,weights,cap=128,cfg=cfg,rows=1,pipeline=False,
        backend=lambda c,i:CheckedRtlBackend(c,i,w,root/'traces'))
    tokens=np.random.default_rng(9).integers(0,128,100).tolist()
    for i,token in enumerate(tokens):
        engine.step(token)
        if (i+1)%10==0: print(json.dumps({'completed_steps':i+1,'total':100}),flush=True)
    evidence={'schema_version':2,'status':'completed','backend':'rtl','synthetic':True,
        'base_model_id':info['base_model_id'],'tokenizer_id':info['tokenizer_id'],
        'configuration_id':identity({'config':cfg.__dict__,'uarch':PERSONALITIES['balanced'].uarch}),
        'validated_steps':len(engine.stats),'bit_exact':all(s['validation']=='bit-exact-dram-and-tmem' for s in engine.stats),
        'input_tokens':tokens,'input_token_hash':identity(tokens),'seed':9,'counters':engine.stats,
        'toolchain':{'upstream':upstream.REVISION,'python':platform.python_version(),'build_id':engine.backend.build_id}}
    store=Store(root/'research')
    try: key=store.save('llm-tiny-rtl-evidence',evidence)
    finally: store.close()
    file=root/'tiny-release.json'; file.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'record_id':key,'path':str(file),'sha256':digest(file)}),flush=True)

if __name__=='__main__': main()
