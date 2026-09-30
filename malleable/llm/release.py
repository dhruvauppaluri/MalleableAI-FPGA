"""Strict standalone release gate: missing evidence is failure, never a skipped pass."""
import json
from pathlib import Path
from .models import digest
from .quality import gate, MIN_RELEASE_TARGETS

REQUIRED=('Qwen3-0.6B','Qwen3.5-0.8B','LFM2.5-230M')

def approved_draft(path):
    """Read the checksummed draft generation from an already verified manifest."""
    manifest=Path(path).resolve(); root=manifest.parent
    data=json.loads(manifest.read_text())
    item=data.get('models',{}).get('Qwen3-0.6B',{}).get('generation')
    if not isinstance(item,dict) or not isinstance(item.get('path'),str):
        raise ValueError('approved standalone Qwen3 draft artifact required')
    file=(root/item['path']).resolve()
    if not file.is_relative_to(root) or digest(file)!=item.get('sha256'):
        raise ValueError('approved draft artifact hash/path mismatch')
    return json.loads(file.read_text())


def validate_draft_approval(approved,selected):
    """Bind execution and final acceptance to the same approved draft identity."""
    workload=approved.get('workload',{})
    expected={'draft_model_id':approved.get('base_model_id'),
        'tokenizer_id':approved.get('tokenizer_id'),
        'draft_variant_id':approved.get('variant_id'),
        'draft_derived_candidate':approved.get('derived_candidate'),
        'draft_configuration_id':approved.get('configuration_id'),
        'draft_head_format':'int8','personality':approved.get('personality'),
        'context':workload.get('context'),'wformat':workload.get('wformat')}
    if any(expected[k] is None for k in expected if k!='draft_derived_candidate'):
        raise ValueError('incomplete approved standalone draft identity')
    if any(selected.get(k)!=v for k,v in expected.items()):
        raise ValueError('selected hybrid draft differs from approved standalone draft identity/configuration')
    return True

def check(path):
    if not path: raise ValueError('release manifest required')
    root=Path(path).resolve().parent; data=json.loads(Path(path).read_text())
    pins=json.loads((Path(__file__).resolve().parents[2]/'docs/model-revisions.json').read_text())['models']
    if data.get('schema_version')!=3: raise ValueError('strict release requires manifest schema v3; legacy evidence remains readable')
    def artifact(item):
        file=(root/item['path']).resolve()
        if not file.is_relative_to(root) or digest(file)!=item['sha256']: raise ValueError('evidence hash/path mismatch')
        return json.loads(file.read_text())
    tiny=artifact(data['tiny_rtl'])
    if (tiny.get('status')!='completed' or tiny.get('backend')!='rtl' or tiny.get('validated_steps',0)<100
        or tiny.get('bit_exact') is not True or not tiny.get('configuration_id') or not tiny.get('input_token_hash')
        or len(tiny.get('counters',[]))!=tiny['validated_steps']
        or any(s.get('validation')!='bit-exact-dram-and-tmem' or not s.get('trace_sha256') for s in tiny['counters'])
        or not tiny.get('toolchain',{}).get('build_id')):
        raise ValueError('100-token tiny full-RTL evidence required')
    for name in REQUIRED:
        entry=data.get('models',{}).get(name)
        if not entry: raise ValueError('missing release evidence: '+name)
        run=artifact(entry['generation']); quality=artifact(entry['held_out_quality'])
        if run.get('base_model_id')!=pins[name]['base_model_id']: raise ValueError('required official model lineage mismatch: '+name)
        if (run.get('status')!='completed' or run.get('generation_mode','greedy')!='greedy'
            or run.get('valid') is not True or run.get('backend')!='rtl' or len(run.get('tokens',[]))<8
            or run.get('prompt_tokens',0)<1 or not run.get('counters')
            or len(run['counters'])!=run['prompt_tokens']+len(run['tokens'])-1
            or not run.get('configuration_id') or not run.get('input_token_hash') or not run.get('tokenizer_id')
            or not run.get('toolchain',{}).get('build_id')
            or any(s.get('validation')!='bit-exact-dram-and-tmem' or not s.get('trace_sha256') for s in run['counters'])):
            raise ValueError('short-prompt + eight-token full-RTL agreement required: '+name)
        if (run.get('workload',{}).get('context')!=128 or run.get('personality')!='balanced'
            or run.get('workload',{}).get('wformat')!='int8' or quality.get('head_format')!='int8'):
            raise ValueError('primary acceptance requires context 128, balanced INT8 matrices and INT8 head: '+name)
        suite=artifact(entry['quality_suite'])
        from .quality import suites
        suite_path=(root/entry['quality_suite']['path']).resolve()
        _,suite_hashes=suites(suite_path)
        expected_counts={s:sum(len(row)-1 for row in suite[s]) for s in suite_hashes}
        expected_freeze={'split_hashes':suite_hashes,'target_counts':expected_counts}
        if (suite.get('base_model_id')!=run['base_model_id'] or suite.get('tokenizer_id')!=run['tokenizer_id']
            or suite.get('freeze')!=expected_freeze or any(expected_counts.get(s,0)<MIN_RELEASE_TARGETS for s in ('validation','held-out'))
            or quality.get('suite_file_hash')!=entry['quality_suite']['sha256']):
            raise ValueError('frozen held-out suite lineage/count mismatch: '+name)
        if (quality.get('base_model_id')!=run['base_model_id'] or quality.get('variant_id')!=run['variant_id']
            or quality.get('personality')!=run.get('personality') or quality.get('samples',0)<1024
            or quality.get('target_count',0)<MIN_RELEASE_TARGETS
            or quality.get('tokenizer_id')!=run['tokenizer_id'] or quality.get('configuration_id')!=run['configuration_id']
            or quality.get('split')!='held-out' or not quality.get('suite_hashes') or quality.get('suite_frozen') is not True
            or quality.get('suite_hashes')!=suite_hashes or quality.get('selectable') is not True
            or not quality.get('floating_reference_record') or not quality.get('floating_reference_id')
            or not gate(quality['float_nll'],quality['candidate_nll'],quality['agreement'])['passed']):
            raise ValueError('held-out quality/lineage gate failed: '+name)
        if run.get('derived_candidate') or quality.get('derived_candidate'):
            from .candidates import check_release_lineage
            if not isinstance(entry.get('candidate_freeze'),dict): raise ValueError('derived candidate freeze artifact required: '+name)
            check_release_lineage(run,quality,artifact(entry['candidate_freeze']))
    return {'schema_version':3,'passed':True,'models':list(REQUIRED),'manifest_sha256':digest(path),
            'scope':'local-standalone-simulation','physical_fpga_evidence':False}
