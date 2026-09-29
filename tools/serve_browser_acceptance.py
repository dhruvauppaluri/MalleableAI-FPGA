"""Isolated browser fixture server. Workers are disabled; no model inference runs."""
import argparse
import json
from pathlib import Path
import shutil
import uvicorn
from malleable.ide import create_app


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--model-root',required=True)
    p.add_argument('--replay-profile',required=True); p.add_argument('--port',type=int,default=8766)
    args=p.parse_args(); root=Path(args.root).resolve()
    if root.exists(): raise ValueError('browser fixtures require a fresh, isolated root')
    app=create_app(args.model_root,root,False); jobs=app.state.jobs
    models=sorted(Path(args.model_root).glob('*/config.json'))
    model=models[0].parent
    identifier=jobs.create('generate',dict(model=str(model),prompt='Browser acceptance fixture',prompt_format='chat',
        context=128,max_new=8,backend='isa',messages=[{'role':'user','content':'Browser acceptance fixture'}]))
    directory=root/identifier/'traces/step-00000'; directory.mkdir(parents=True)
    source=Path(args.replay_profile).resolve().parent
    for name in ('profile.json','trace.txt','prog_0.hex'): shutil.copyfile(source/name,directory/name)
    jobs.event(identifier,'counters',dict(step=0,cycles=1,profile_path=str(directory/'profile.json'),
        provenance='browser fixture; historical recorded trace replay; not fresh inference'))
    result={'schema_version':1,'status':'completed','backend':'isa','valid':False,
        'personality':'balanced','workload':{'wformat':'int8'},
        'text':'Browser acceptance fixture output.','tokens':[1,2],
        'provenance':'synthetic browser fixture; never release inference evidence'}
    key=jobs.persist('generate',identifier,{'kind':'result','payload':result})
    jobs.event(identifier,'result',dict(result,record_id=key))
    with jobs.connect() as db: db.execute("UPDATE jobs SET status='completed' WHERE id=?",(identifier,))
    jobs.event(identifier,'status',{'status':'completed'})
    (root/'fixture-lineage.json').write_text(json.dumps({'run_id':identifier,'source_profile':str(source/'profile.json'),
        'workers':False,'scope':'browser behavior only; no new RTL or quality acceptance'},indent=2))
    print(json.dumps({'url':f'http://127.0.0.1:{args.port}','fixture_run':identifier}),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port)


if __name__=='__main__': main()
