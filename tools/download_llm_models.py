"""Explicit, opt-in download of the three approved public checkpoints.

Not imported by the inference service. No remote Python/pickle files are fetched.
Revisions and content hashes are recorded beside the local checkpoint.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi,snapshot_download

MODELS=('Qwen/Qwen3-0.6B','Qwen/Qwen3.5-0.8B','LiquidAI/LFM2.5-230M')
VERIFIER='Qwen/Qwen3-1.7B'
PATTERNS=('*.safetensors','*.safetensors.index.json','config.json','generation_config.json',
          'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json',
          'merges.txt','chat_template*.jinja','LICENSE*','README.md')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--destination',default='build/models')
    p.add_argument('--include-qwen3-verifier',action='store_true',
        help='explicitly download the optional Qwen3-1.7B CUDA verifier; never used by inference')
    a=p.parse_args(); root=Path(a.destination).resolve(); root.mkdir(parents=True,exist_ok=True)
    api=HfApi(token=False)
    pins=json.loads((Path(__file__).resolve().parents[1]/'docs/model-revisions.json').read_text())['models']
    repos=MODELS+((VERIFIER,) if a.include_qwen3_verifier else ())
    for repo in repos:
        # The verifier is optional and resolves upstream only after this explicit
        # flag; the resolved immutable commit is written into its local manifest.
        requested=pins.get(repo.split('/')[-1],{}).get('revision','main')
        info=api.model_info(repo,revision=requested,files_metadata=True,token=False)
        files=[s for s in info.siblings if '/' not in s.rfilename
               and any(fnmatch.fnmatch(s.rfilename,pattern) for pattern in PATTERNS)]
        needed=sum(s.size or 0 for s in files)
        if shutil.disk_usage(root).free < needed+4*1024**3:
            raise RuntimeError('insufficient free disk for checkpoint plus 4 GiB reserve')
        local=root/repo.split('/')[-1]
        print(json.dumps({'phase':'download','repo':repo,'revision':info.sha,'bytes':needed}),flush=True)
        snapshot_download(repo,revision=info.sha,local_dir=local,allow_patterns=[s.rfilename for s in files],
                          token=False,max_workers=2)
        hashes={s.rfilename:digest(local/s.rfilename) for s in files}
        for s in files:
            if s.lfs and hashes[s.rfilename]!=s.lfs.sha256: raise RuntimeError('weight content hash mismatch')
        record={'schema_version':1,'repo':repo,'revision':info.sha,'files':hashes,'safe_formats_only':True,
                'download_mode':'explicit-verifier-opt-in' if repo==VERIFIER else 'approved-model-pin'}
        (local/'download-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'phase':'download-complete','repo':repo,'path':str(local),'file_count':len(files)}),flush=True)

if __name__=='__main__': main()
