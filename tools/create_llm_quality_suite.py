"""Tokenize the authored local smoke suite; no downloads or remote code."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from malleable.llm.models import inspect
from malleable.llm.quality import suites
from malleable.llm.models import digest
from malleable.records import identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--text-suite',default='examples/llm-release-quality-text.json'); a=p.parse_args()
    source=json.loads(Path(a.text_suite).read_text()); model=inspect(a.model)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    data={'schema_version':1,'base_model_id':model['base_model_id'],'tokenizer_id':model['tokenizer_id'],
          'description':source['description']}
    if 'source_files' in source:
        repo=Path(__file__).resolve().parents[1]; source_hashes={}
        all_paths=[s for rows in source['source_files'].values() for s in rows]
        if len(all_paths)!=len(set(all_paths)): raise ValueError('source-file split leakage')
        for split in ('calibration','validation','held-out'):
            tape=[]
            for name in source['source_files'][split]:
                file=(repo/name).resolve()
                if not file.is_relative_to(repo/'docs') or not file.is_file(): raise ValueError('suite source must be a repository document')
                source_hashes[name]=digest(file); tape.extend(tokenizer.encode(file.read_text(),add_special_tokens=False))
            targets=256 if split=='calibration' else 1024
            if len(tape)<targets+8: raise ValueError('insufficient distinct source tokens: '+split)
            rows=[]; at=0
            while targets:
                take=min(128,targets); rows.append(tape[at:at+take+1]); at+=take+1; targets-=take
            data[split]=rows
        data['source_document_hashes']=source_hashes
    else:
        for split in ('calibration','validation','held-out'): data[split]=[tokenizer.encode(text) for text in source[split]]
    hashes={s:identity(data[s]) for s in ('calibration','validation','held-out')}
    counts={s:sum(len(r)-1 for r in data[s]) for s in hashes}
    if any(counts[s]<1024 for s in ('validation','held-out')): raise ValueError('source text requires 1,024 target tokens per evaluation split')
    data.update(freeze={'split_hashes':hashes,'target_counts':counts},source_text_sha256=digest(a.text_suite))
    path=Path(a.output)
    if path.exists(): raise ValueError('refusing to overwrite an existing frozen suite; choose a new output path')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2)+'\n')
    suites(path); print(json.dumps({'suite':str(path),'target_counts':counts}))

if __name__=='__main__': main()
