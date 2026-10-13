"""Fresh-seed ROCm/GPU repetition of the tiny shared-matrix diagnostic.

This is generated-data model research only. It does not evaluate Qwen3-0.6B,
consume any held-out suite, establish FPGA performance, or overwrite the earlier
CPU report. The GPU is used for training/scoring; ISA layout equivalence remains
an independent CPU-side check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import transformers

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from malleable.llm import upstream
from opentpu.llm.qwen3 import Engine, Spec, device_config
from tools.train_tiny_shared_student import (BATCH, LENGTH, LR, STUDENT_STEPS,
    TEACHER_STEPS, config, sequences, train, score, weights)

SEEDS={'teacher_init':411,'teacher_train':4101,'student_init':417,
       'student_train':4202,'validation':4303}


def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()+b'\0')
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('refusing to overwrite diagnostic output')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU unavailable; stop the billable notebook')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEEDS['teacher_init'])
    teacher=transformers.Qwen3ForCausalLM(config()).float().to(device)
    teacher_batches=sequences(SEEDS['teacher_train'],TEACHER_STEPS*BATCH).split(BATCH)
    train(teacher,[batch.to(device) for batch in teacher_batches])
    teacher.eval()
    torch.manual_seed(SEEDS['student_init'])
    ordinary=transformers.Qwen3ForCausalLM(config()).float().to(device)
    with torch.no_grad():
        for name in ('gate','up'):
            ordinary.model.layers[1].mlp.__getattr__(name+'_proj').weight.copy_(
                ordinary.model.layers[0].mlp.__getattr__(name+'_proj').weight)
    shared=transformers.Qwen3ForCausalLM(config()).float().to(device)
    shared.load_state_dict(ordinary.state_dict())
    for name in ('gate','up'):
        setattr(getattr(shared.model.layers[1].mlp,name+'_proj'),'weight',
                getattr(shared.model.layers[0].mlp,name+'_proj').weight)
    batches=sequences(SEEDS['student_train'],STUDENT_STEPS*BATCH).split(BATCH)
    batches=[batch.to(device) for batch in batches]
    train(ordinary,batches,teacher)
    train(shared,batches,teacher)
    validation=sequences(SEEDS['validation'],128).to(device)
    teacher_result=score(teacher,teacher,validation)
    ordinary_result=score(ordinary,teacher,validation)
    shared_result=score(shared,teacher,validation)

    # Use an independent host ISA check with fresh seed-derived inputs.
    spec=Spec(128,2,1,1,128,256,128)
    model_weights=weights(shared.cpu())
    cfg=device_config(spec,128,S=1)
    ordinary_engine=Engine(spec,model_weights,cap=128,cfg=cfg,rows=1,pipeline=False)
    shared_engine=Engine(spec,model_weights,cap=128,cfg=cfg,rows=1,pipeline=False,
                         shared_matrices=('wg','wu'))
    isa_match=True
    for token in validation[0,:8].cpu().tolist():
        a=ordinary_engine.step(token); b=shared_engine.step(token)
        isa_match &= bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))
    report=dict(schema_version=1,kind='fresh-seed-amd-gpu-synthetic-diagnostic',
        release_quality_claim=False,automatic_switching_allowed=False,
        device={'pytorch':torch.__version__,'hip':torch.version.hip,
                'name':torch.cuda.get_device_name(0),
                'memory_bytes':torch.cuda.get_device_properties(0).total_memory,
                'transformers':transformers.__version__},
        data={'generator':'arithmetic-token-sequences-v1','seeds':SEEDS,
              'train_steps':{'teacher':TEACHER_STEPS,'student_each':STUDENT_STEPS},
              'validation_sequences':128,'sequence_length':LENGTH,'vocabulary':128},
        training={'batch':BATCH,'learning_rate':LR,
                  'student_start':'ordinary and shared start identical; layer 1 gate/up equal layer 0'},
        teacher={**teacher_result,'weight_sha256':digest(teacher)},
        ordinary_student={**ordinary_result,'weight_sha256':digest(ordinary)},
        shared_student={**shared_result,'weight_sha256':digest(shared)},
        frozen_diagnostic_rule={'shared_nll_at_most_105_percent_of_ordinary':True,
            'shared_teacher_agreement_at_least_90_percent':True,'isa_layout_outputs_bit_exact':True},
        passed_diagnostic_gate=(shared_result['nll']<=1.05*ordinary_result['nll']
            and shared_result['teacher_next_token_agreement']>=.9 and isa_match),
        isa_layout_outputs_bit_exact=isa_match,
        ordinary_image_bytes=ordinary_engine.image.nbytes,
        shared_image_bytes=shared_engine.image.nbytes,
        limitations=['Synthetic task, not the pretrained Qwen3-0.6B checkpoint.',
            'A fresh synthetic seed is not a release held-out evaluation.',
            'GPU training does not establish FPGA timing or switching benefit.',
            'This report is distinct from the earlier CPU diagnostic.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'output':str(output),'gpu':report['device']['name'],
        'ordinary_nll':ordinary_result['nll'],'shared_nll':shared_result['nll'],
        'shared_teacher_agreement':shared_result['teacher_next_token_agreement'],
        'isa_bit_exact':isa_match,'passed_diagnostic_gate':report['passed_diagnostic_gate']}))


if __name__=='__main__': main()
