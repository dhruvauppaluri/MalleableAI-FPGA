"""Strict local release gate, including measured learning and Zephyrus evidence."""
import json
import math
from pathlib import Path
from .models import digest
from .release import check as check_standalone
from ..records import identity

def validate_performance_lineage(schedule,expected_model_id):
    """Verify original checkpoint files before derived artifact validation."""
    from .models import inspect
    from .experiments import validate_performance_manifest
    model_path=schedule.get('model_path')
    if not isinstance(model_path,str): raise ValueError('benchmark model path and verified weight lineage required')
    model_info=inspect(model_path)
    if model_info.get('base_model_id')!=expected_model_id:
        raise ValueError('benchmark checkpoint differs from required official model')
    validate_performance_manifest(schedule,model_info)
    return model_info


def validate_final_verification(ui,browser,source_commit):
    if (ui.get('status')!='passed' or 'make verify-llm' not in ui.get('commands',[])
        or ui.get('checkpoint_downloads') is not False or not source_commit
        or ui.get('commit')!=source_commit or ui.get('source_unchanged') is not True
        or ui.get('source_dirty') is not False):
        raise ValueError('fresh verification of the final clean source commit required')
    required={'chat-formatting','sse-reconnect','duplicate-stale-events','cancellation',
        'restart','report-lineage','lens-replay','keyboard','validated-application'}
    if (browser.get('status')!='passed' or browser.get('commit')!=source_commit
        or browser.get('schema_version')!=1 or not required.issubset(set(browser.get('passed_checks',[])))):
        raise ValueError('matching final-source browser acceptance evidence required')


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
    from .release import approved_draft,validate_draft_approval
    approved=approved_draft(standalone_path)

    def artifact(item):
        if not isinstance(item,dict) or not isinstance(item.get('path'),str):
            raise ValueError('CUDA/hybrid evidence artifact reference required')
        file=(root/item['path']).resolve()
        if not file.is_relative_to(root) or digest(file)!=item.get('sha256'):
            raise ValueError('release evidence hash/path mismatch')
        return json.loads(file.read_text())

    gpu=artifact(manifest.get('gpu_baseline'))
    hybrid=artifact(manifest.get('hybrid'))
    validate_draft_approval(approved,hybrid)
    cache=artifact(manifest.get('cuda_cache_verification'))
    if (cache.get('status')!='passed' or cache.get('actual_cuda') is not True
        or cache.get('depths')!=[1,2,4,8] or cache.get('rejection_positions')!='every position per depth'
        or cache.get('eos') is not True or cache.get('context_boundary')!=128):
        raise ValueError('actual CUDA cache crop/correction, EOS and context-boundary evidence required')
    for record in (gpu,hybrid):
        values={name:record.get('metrics',{}).get(name,{}).get('value') for name in
            ('loading_build_seconds','prefill_seconds','decode_seconds','inference_total_seconds')}
        if (record.get('timing_schema',{}).get('schema_version')!=1
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values())
            or not math.isclose(values['inference_total_seconds'],values['prefill_seconds']+values['decode_seconds'],rel_tol=1e-6)):
            raise ValueError('comparable synchronized loading/prefill/decode/inference-total timing required')
    if (gpu.get('status')!='completed' or gpu.get('context')!=128 or gpu.get('max_new')!=8 or len(gpu.get('tokens',[]))!=8
        or gpu.get('hardware',{}).get('dtype')!='float16' or gpu['hardware'].get('attention')!='eager' or gpu['hardware'].get('tf32') is not False
        or hybrid.get('status')!='completed' or hybrid.get('context')!=128 or hybrid.get('depth')!=4 or len(hybrid.get('tokens',[]))!=8
        or hybrid.get('tokens')!=gpu.get('tokens') or hybrid.get('dtype')!='float16' or hybrid.get('attention')!='eager' or hybrid.get('tf32') is not False):
        raise ValueError('canonical context-128 eight-token FP16 eager CUDA/depth-4 hybrid evidence required')
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
        validate_performance_lineage(schedule,expected[name])
        if (schedule.get('base_model_id')!=expected[name] or report.get('base_model_id')!=expected[name]
            or identity(schedule)!=report.get('manifest_id') or schedule.get('seed')!=42 or report.get('seed')!=42 or len(schedule.get('runs',[]))!=10
            or len(report.get('runs',[]))!=10 or report.get('completed')!=10 or report.get('failed')!=0):
            raise ValueError('complete ten-run benchmark lineage required: '+name)
        for index,(spec,result) in enumerate(zip(schedule['runs'],report['runs'])):
            from .experiments import validate_benchmark_result
            validate_benchmark_result(schedule,index,result)
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
    from .optimization import Predictor,PREDICTOR_TRAINING_MODELS,PREDICTOR_HELD_OUT_MODEL
    predictor=artifact(manifest.get('predictor_checkpoint')); predicted=artifact(manifest.get('predictor_evaluation'))
    Predictor.from_checkpoint(predictor)
    if (set(predictor['training_models'])!=PREDICTOR_TRAINING_MODELS or predicted.get('checkpoint')!=identity(predictor)
        or predicted.get('schema_version')!=2 or predicted.get('split')!='held-out' or not predicted.get('predictions')
        or predicted.get('base_models')!=[PREDICTOR_HELD_OUT_MODEL]):
        raise ValueError('frozen measured Qwen training / LFM held-out predictor report required')
    from .learning import validate_checkpoint
    validate_checkpoint(policy)
    if (policy.get('seed')!=0 or policy.get('passes')!=20 or set(policy.get('training_models',[]))!=PREDICTOR_TRAINING_MODELS
        or evaluation.get('seeds')!=list(range(100,105)) or evaluation.get('base_models')!=[PREDICTOR_HELD_OUT_MODEL]
        or evaluation.get('action_coverage')!=sorted(p+'/int8' for p in ('compact','balanced','compute','buffered'))
        or evaluation.get('horizon_coverage')!=[1,8,32,128] or evaluation.get('objective_coverage')!=['latency','throughput']
        or 'per_case_regressions' not in evaluation or not evaluation.get('cases')):
        raise ValueError('primary four-personality, fixed-partition controller evaluation required')
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
    browser=artifact(manifest.get('browser_acceptance'))
    validate_final_verification(ui,browser,manifest.get('source_commit'))

    return {'schema_version':2,'passed':True,'standalone':standalone['manifest_sha256'],
        'gpu_baseline':digest(root/manifest['gpu_baseline']['path']),
        'hybrid':digest(root/manifest['hybrid']['path']),
        'performance_suites':performance,'policy_evaluation':digest(root/manifest['policy_evaluation']['path']),
        'ui_verification':digest(root/manifest['ui_verification']['path']),
        'browser_acceptance':digest(root/manifest['browser_acceptance']['path']),
        'scope':'local-RTL-simulation-plus-measured-CUDA-hybrid','physical_fpga_evidence':False}
