"""Package existing acceptance evidence without inference or historical rewrites."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from malleable.llm.models import digest,inspect
from malleable.llm import candidates as C
from malleable.llm.heldout import registry_path,evaluation_key
from malleable.llm.quality import gate,reference_contract
from malleable.llm.release import check
from malleable.records import identity

RELEASE=ROOT/'build/zephyrus-jobs/release/20260928T223413Z-b463e746'
CONT=RELEASE/'continuation-20260930'

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    out=a.output.resolve();out.mkdir(parents=True)
    source=C.source_state()
    if source['status']:raise ValueError('clean source required for package audit')
    inventory=[];trace_audit=[]
    def copy(src,dest):
        target=out/dest;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        sha=digest(src)
        if digest(target)!=sha:raise ValueError('copy hash mismatch')
        inventory.append({'source':str(src),'path':dest,'sha256':sha})
        return {'path':dest,'sha256':sha}
    def read(path):
        record=json.loads(path.read_text())
        if 'record_id' in record and record['record_id']!=identity({k:v for k,v in record.items() if k!='record_id'}):
            raise ValueError('canonical result mismatch: '+str(path))
        summary=path.parent/'summary.json'
        if summary.exists():
            data=json.loads(summary.read_text())
            if data.get('status')!='completed' or data.get('result_sha256')!=digest(path):raise ValueError('summary mismatch')
        return record
    def traces(record,label):
        for index,row in enumerate(record['counters']):
            profile=Path(row['profile_path']);trace=profile.parent/'trace.txt'
            if row.get('validation')!='bit-exact-dram-and-tmem' or digest(trace)!=row['trace_sha256']:
                raise ValueError('trace integrity mismatch: '+label)
            trace_audit.append({'artifact':label,'index':index,'trace_path':str(trace),
                'trace_sha256':row['trace_sha256'],'profile_path':str(profile),'profile_sha256':digest(profile)})
    entries={
        'Qwen3-0.6B':('qwen3',RELEASE/'candidate-rtl-acceptance-02/result.json',
            RELEASE/'candidate-heldout-freeze-01/result.json',RELEASE/'candidate-isa-validation-01/result.json',
            RELEASE/'candidate-heldout-freeze-01/freeze.json'),
        'Qwen3.5-0.8B':('qwen35',CONT/'qwen35-rtl-eight-02/result.json',
            CONT/'qwen35-full-acceptance-01/held-out/result.json',CONT/'qwen35-full-acceptance-01/validation/result.json',
            CONT/'qwen35-full-acceptance-01/held-out/freeze.json'),
        'LFM2.5-230M':('lfm2',RELEASE/'lfm2-rtl-acceptance-02/result.json',
            RELEASE/'lfm2-raw-int8-heldout-01/result.json',RELEASE/'lfm2-raw-int8-validation-01/result.json',None)}
    manifest={'schema_version':3,'models':{}}
    metrics={}
    for name,(family,generation,quality,validation,freeze) in entries.items():
        run=read(generation);q=read(quality);v=read(validation)
        info=inspect(ROOT/'build/models'/name,128)
        if any(info[k]!=q[k] for k in ('base_model_id','tokenizer_id')):raise ValueError('current checkpoint differs')
        for record,split in ((v,'validation'),(q,'held-out')):
            if record['split']!=split or record['target_count']<1024 or not gate(record['float_nll'],record['candidate_nll'],record['agreement'])['passed']:
                raise ValueError('required full quality split failed')
            if record['toolchain']['contract']!=reference_contract(info):raise ValueError('stale reference contract')
            C.validate_configuration(record,info)
            for key in ('base_model_id','tokenizer_id','variant_id','configuration_id','suite_file_hash','suite_hashes','derived_candidate'):
                if record.get(key)!=q.get(key):raise ValueError('validation/held-out lineage differs: '+key)
        with sqlite3.connect(registry_path()) as db:
            claim=db.execute('SELECT payload FROM evaluations WHERE evaluation_id=?',(identity(evaluation_key(q)),)).fetchone()
        if claim is None:raise ValueError('held-out consumption not registered')
        if name=='Qwen3-0.6B':
            derived=C.read_derived_candidate(RELEASE/'quality-recovery-pilot-01/int8/case-00',info)
            if C.derived_record(derived)!=q['derived_candidate']:raise ValueError('derived candidate differs')
        if len(run['tokens'])!=8:raise ValueError('exactly eight output tokens required')
        traces(run,name)
        base='models/'+name+'/'
        entry={'generation':copy(generation,base+'generation.json'),
            'held_out_quality':copy(quality,base+'held-out-quality.json'),
            'validation_quality':copy(validation,base+'validation-quality.json'),
            'quality_suite':copy(ROOT/'build/models/quality'/f'{family}-frozen-v2.json',base+'quality-suite.json')}
        if freeze:entry['candidate_freeze']=copy(freeze,base+'candidate-freeze.json')
        write(out/base/'heldout-consumption.json',json.loads(claim[0]))
        manifest['models'][name]=entry
        metrics[name]={'validation_agreement':v['agreement'],'held_out_agreement':q['agreement'],
            'held_out_nll_degradation':q['nll_degradation'],'rtl_output_tokens':len(run['tokens']),
            'validated_steps':len(run['counters'])}
    tiny=RELEASE/'tiny-rtl-100-01/tiny-release.json';traces(read(tiny),'tiny-100')
    manifest['tiny_rtl']=copy(tiny,'tiny-rtl.json')
    # Review baseline predates accepted historical generations; no execution math changed.
    baseline='0df744eb8f7dd906c434b6f5340dcd39ee000e16'
    paths=['third_party/opentpu','rtl','malleable/llm/runtime.py','malleable/llm/backend.py','malleable/llm/models.py']
    changes=subprocess.check_output(['git','diff','--name-only',baseline,source['commit'],'--',*paths],cwd=ROOT,text=True)
    if changes.strip():raise ValueError('execution dependency changes need a new compatibility review')
    write(out/'source-compatibility.json',{'review_source':source,'comparison_baseline':baseline,
        'unchanged_execution_paths':paths,'execution_diff':changes,
        'quality_review':'Qwen3/LFM arithmetic and reference v1 unchanged; configuration/freeze/registry safeguards strengthened. Qwen3.5 evidence uses corrected RoPE reference v2 on 608fea7. Historical source records are not rewritten.',
        'scope':'reviewed evidence reuse; not final-source full-platform verification'})
    write(out/'trace-audit.json',{'verified_steps':len(trace_audit),'traces':trace_audit,
        'storage':'Original trace/profile files retained at recorded absolute paths; JSON artifacts copied byte-for-byte.'})
    write(out/'inventory.json',{'source':source,'created_utc':datetime.now(timezone.utc).isoformat(),'artifacts':inventory})
    write(out/'standalone-v3.json',manifest)
    result=check(out/'standalone-v3.json')
    write(out/'release-check.json',result)
    write(out/'summary.json',{'status':'completed','source':source,'release_check':result,'models':metrics,
        'verified_trace_steps':len(trace_audit),'full_release_passed':False})
    print(json.dumps({'output':str(out),'check':result,'metrics':metrics,'verified_trace_steps':len(trace_audit)},indent=2))

if __name__=='__main__':main()
