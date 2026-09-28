"""Strict local release gate, including measured learning and Zephyrus evidence."""
import json
import math
from pathlib import Path
from .models import digest
from .release import check as check_standalone
from ..records import identity


def check(path):
    if not path: raise ValueError('full release manifest required')
    root=Path(path).resolve().parent
    manifest=json.loads(Path(path).read_text())
    if manifest.get('schema_version')!=2:
        raise ValueError('full release manifest schema v2 required')
    standalone_item=manifest.get('standalone_manifest')
    if not isinstance(standalone_item,dict): raise ValueError('standalone evidence manifest required')
    standalone_path=(root/standalone_item['path']).resolve()
    if not standalone_path.is_relative_to(root) or digest(standalone_path)!=standalone_item.get('sha256'):
        raise ValueError('standalone manifest hash/path mismatch')
    standalone=check_standalone(standalone_path)

    def artifact(item):
        if not isinstance(item,dict) or not isinstance(item.get('path'),str):
            raise ValueError('CUDA/hybrid evidence artifact reference required')
        file=(root/item['path']).resolve()
        if not file.is_relative_to(root) or digest(file)!=item.get('sha256'):
            raise ValueError('release evidence hash/path mismatch')
        return json.loads(file.read_text())

    gpu=artifact(manifest.get('gpu_baseline'))
    hybrid=artifact(manifest.get('hybrid'))
    if (gpu.get('backend')!='cuda' or gpu.get('mode')!='greedy'
        or gpu.get('standalone_release_id')!=standalone['manifest_sha256']
        or gpu.get('host_environment',{}).get('wsl2') is not True
        or 'NVIDIA' not in gpu.get('hardware',{}).get('device','')
        or gpu.get('source_repository')!='Qwen/Qwen3-1.7B'
        or not isinstance(gpu.get('source_revision'),str) or len(gpu['source_revision'])!=40
        or gpu.get('hardware',{}).get('provenance')!='measured-cuda-host'
        or gpu.get('metrics',{}).get('tokens_per_second',{}).get('provenance')!='measured-cuda-host'):
        raise ValueError('measured local CUDA greedy baseline required')
    if (hybrid.get('backend')!='hybrid-simulated-draft-cuda-verifier'
        or hybrid.get('standalone_release_id')!=standalone['manifest_sha256']
        or hybrid.get('verifier_repository')!='Qwen/Qwen3-1.7B'
        or hybrid.get('verifier_revision')!=gpu.get('source_revision')
        or hybrid.get('verifier_source_manifest_sha256')!=gpu.get('source_manifest_sha256')
        or hybrid.get('host_environment',{}).get('wsl2') is not True
        or hybrid.get('verifier_model_id')!=gpu.get('base_model_id')
        or hybrid.get('input_token_hash')!=gpu.get('input_token_hash')
        or hybrid.get('gpu_greedy_tokens')!=gpu.get('tokens')
        or hybrid.get('output_agrees') is not True or hybrid.get('speedup_established') is not False
        or hybrid.get('tokenizer_pair_id') is None or hybrid.get('depth') not in (1,2,4,8)):
        raise ValueError('tokenizer-compatible, GPU-agreeing hybrid run required')

    expected={
        'Qwen3-0.6B':'7ab1181d3a2b04ce889880dfc3b94933574441e9c221e950622c39a3ce79a59d',
        'Qwen3.5-0.8B':'96c8847b78593ff15008955c68e384f5ff40895e425e487aec30b8173542a857',
        'LFM2.5-230M':'8d6ac525c1b135ad360d0f9f5a822698bbb257b7408f791e2ca695ae473798b3'}
    performance={}
    entries=manifest.get('performance_suites')
    if not isinstance(entries,list) or len(entries)!=3: raise ValueError('three complete ten-run model performance suites required')
    for entry in entries:
        name=entry.get('model')
        if name not in expected or name in performance: raise ValueError('unknown/duplicate performance model')
        schedule=artifact(entry.get('manifest')); report=artifact(entry.get('report'))
        if (schedule.get('base_model_id')!=expected[name] or report.get('base_model_id')!=expected[name]
            or identity(schedule)!=report.get('manifest_id') or len(schedule.get('runs',[]))!=10
            or len(report.get('runs',[]))!=10 or report.get('completed')!=10 or report.get('failed')!=0):
            raise ValueError('complete ten-run benchmark lineage required: '+name)
        for index,(spec,result) in enumerate(zip(schedule['runs'],report['runs'])):
            if (result.get('status')!='completed' or result.get('valid') is not True or result.get('backend')!='rtl'
                or result.get('generation_mode')!='fixed-token-tape' or result.get('benchmark_run_index')!=index
                or result.get('input_token_hash')!=spec.get('input_token_hash')
                or result.get('memory_scenario')!=spec.get('memory_scenario') or not result.get('counters')
                or any(c.get('validation')!='bit-exact-dram-and-tmem' for c in result['counters'])):
                raise ValueError('invalid or mismatched fixed-tape performance result: '+name)
        performance[name]=digest(root/entry['report']['path'])
    if set(performance)!=set(expected): raise ValueError('missing model performance suite')

    policy=artifact(manifest.get('policy_checkpoint'))
    evaluation=artifact(manifest.get('policy_evaluation'))
    from .learning import validate_checkpoint
    validate_checkpoint(policy)
    if (policy.get('algorithm')!='DoubleDQN' or evaluation.get('policy_id')!=identity(policy)
        or evaluation.get('split')!='held-out' or len(set(evaluation.get('seeds',[])))<5
        or set(evaluation.get('base_models',[])) & set(policy.get('training_models',[]))
        or evaluation.get('observation_schema')!='frozen-predictor-no-oracle-v2'
        or set(evaluation.get('mean_cost',{}))!={'rl','fixed','heuristic','predictor','random','exhaustive'}
        or not evaluation.get('negative_results_included')):
        raise ValueError('base-model-disjoint five-seed held-out policy evaluation required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in evaluation['mean_cost'].values()):
        raise ValueError('held-out comparison costs must be finite and positive')
    disposition=manifest.get('controller_disposition')
    if disposition not in ('double-dqn-promoted','validated-deterministic-retained','previous-policy-retained'):
        raise ValueError('explicit policy promotion/retention outcome required')
    scores=evaluation['mean_cost']
    if disposition=='double-dqn-promoted':
        if scores['rl']>scores['fixed'] or scores['rl']>scores['heuristic']:
            raise ValueError('cannot promote RL after held-out regression against fixed/heuristic baselines')
        deployment=artifact(manifest.get('policy_deployment'))
        if deployment.get('active_policy')!=identity(policy) or deployment.get('promotion')!='explicit':
            raise ValueError('explicit matching policy promotion evidence required')

    ui=artifact(manifest.get('ui_verification'))
    if (ui.get('status')!='passed' or 'make verify-llm' not in ui.get('commands',[])
        or ui.get('checkpoint_downloads') is not False or ui.get('commit') is None):
        raise ValueError('checkpoint-free UI/integration verification evidence required')

    return {'schema_version':2,'passed':True,'standalone':standalone['manifest_sha256'],
        'gpu_baseline':digest(root/manifest['gpu_baseline']['path']),
        'hybrid':digest(root/manifest['hybrid']['path']),
        'performance_suites':performance,'policy_evaluation':digest(root/manifest['policy_evaluation']['path']),
        'ui_verification':digest(root/manifest['ui_verification']['path']),
        'scope':'local-RTL-simulation-plus-measured-CUDA-hybrid','physical_fpga_evidence':False}
