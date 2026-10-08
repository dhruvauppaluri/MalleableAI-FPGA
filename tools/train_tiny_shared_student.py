"""Deterministic CPU diagnostic of training a Qwen-compatible shared-weight student.

This uses a generated sequence task, not a language quality suite. It neither
opens a release held-out split nor makes any claim about Qwen3-0.6B quality.
The ordinary and tied students start with identical values and see identical
batches. A separate frozen validation seed is used only for this diagnostic.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import upstream  # registers the pinned openTPU package
from opentpu.llm.qwen3 import Engine, Spec, device_config

OUTPUT = ROOT / 'docs/evidence/model-hardware-codesign-20261008/tiny-training.json'
TEACHER_STEPS = 300
STUDENT_STEPS = 300
BATCH = 8
LENGTH = 32
LR = 0.003


def config():
    return transformers.Qwen3Config(hidden_size=128, num_hidden_layers=2,
                                    num_attention_heads=1, num_key_value_heads=1,
                                    head_dim=128, intermediate_size=256,
                                    vocab_size=128, tie_word_embeddings=True,
                                    max_position_embeddings=128)


def sequences(seed, count):
    rng = np.random.default_rng(seed)
    start = rng.integers(0, 64, (count,), dtype=np.int64)
    step = rng.integers(1, 8, (count,), dtype=np.int64)
    tokens = np.empty((count, LENGTH), dtype=np.int64)
    tokens[:, 0], tokens[:, 1] = start, 64 + step
    for t in range(2, LENGTH):
        tokens[:, t] = (start + (t - 1) * step) % 64
    return torch.from_numpy(tokens)


def digest_weights(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode() + b'\0')
        h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def score(model, teacher, data):
    model.eval()
    losses, correct, count, teacher_agree = [], 0, 0, 0
    with torch.no_grad():
        for batch in data.split(BATCH):
            logits = model(input_ids=batch).logits[:, :-1]
            reference = teacher(input_ids=batch).logits[:, :-1]
            target = batch[:, 1:]
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                          target.reshape(-1), reduction='sum').item())
            correct += int((logits.argmax(-1) == target).sum())
            teacher_agree += int((logits.argmax(-1) == reference.argmax(-1)).sum())
            count += target.numel()
    return {'nll': sum(losses) / count, 'next_token_accuracy': correct / count,
            'teacher_next_token_agreement': teacher_agree / count,
            'targets': count}


def train(model, batches, teacher=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=batch).logits[:, :-1]
        target = batch[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        if teacher is None:
            loss = ce
        else:
            with torch.no_grad():
                tlogits = teacher(input_ids=batch).logits[:, :-1]
            temperature = 2.0
            distill = F.kl_div(F.log_softmax(logits / temperature, -1),
                               F.softmax(tlogits / temperature, -1),
                               reduction='sum') / target.numel() * temperature**2
            loss = (ce + distill) / 2
        loss.backward()
        optimizer.step()
    model.eval()


def weights(model):
    return {name: value.detach().float().numpy().copy()
            for name, value in model.state_dict().items()}


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(11)
    teacher = transformers.Qwen3ForCausalLM(config()).float()
    train(teacher, sequences(101, TEACHER_STEPS * BATCH).split(BATCH))
    teacher.eval()
    torch.manual_seed(17)
    ordinary = transformers.Qwen3ForCausalLM(config()).float()
    with torch.no_grad():
        for name in ('gate', 'up'):
            ordinary.model.layers[1].mlp.__getattr__(name + '_proj').weight.copy_(
                ordinary.model.layers[0].mlp.__getattr__(name + '_proj').weight)
    shared = transformers.Qwen3ForCausalLM(config()).float()
    shared.load_state_dict(ordinary.state_dict())
    for name in ('gate', 'up'):
        setattr(getattr(shared.model.layers[1].mlp, name + '_proj'), 'weight',
                getattr(shared.model.layers[0].mlp, name + '_proj').weight)
    batches = sequences(202, STUDENT_STEPS * BATCH).split(BATCH)
    train(ordinary, batches, teacher)
    train(shared, batches, teacher)
    validation = sequences(303, 128)
    teacher_result = score(teacher, teacher, validation)
    control_result = score(ordinary, teacher, validation)
    shared_result = score(shared, teacher, validation)
    spec = Spec(128, 2, 1, 1, 128, 256, 128)
    W = weights(shared)
    cfg = device_config(spec, 128, S=1)
    original_engine = Engine(spec, W, cap=128, cfg=cfg, rows=1, pipeline=False)
    shared_engine = Engine(spec, W, cap=128, cfg=cfg, rows=1, pipeline=False,
                           shared_matrices=('wg', 'wu'))
    isa_match = True
    for token in validation[0, :8].tolist():
        a = original_engine.step(token)
        b = shared_engine.step(token)
        isa_match &= bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    report = {
        'schema_version': 1, 'kind': 'tiny-synthetic-training-diagnostic',
        'release_quality_claim': False,
        'data': {'generator': 'arithmetic-token-sequences-v1', 'train_seeds': [101, 202],
                 'validation_seed': 303, 'validation_sequences': 128,
                 'sequence_length': LENGTH, 'vocabulary': 128},
        'training': {'teacher_steps': TEACHER_STEPS, 'student_steps': STUDENT_STEPS,
                     'batch': BATCH, 'learning_rate': LR, 'distillation_temperature': 2.0,
                     'student_start': 'identical initial values; layer 1 gate/up equal layer 0'},
        'teacher': {**teacher_result, 'weight_sha256': digest_weights(teacher)},
        'ordinary_student': {**control_result, 'weight_sha256': digest_weights(ordinary)},
        'shared_student': {**shared_result, 'weight_sha256': digest_weights(shared)},
        'acceptance_rule_frozen_before_scoring': {
            'shared_nll_at_most_105_percent_of_ordinary': True,
            'shared_teacher_agreement_at_least_90_percent': True,
            'isa_layout_outputs_bit_exact': True},
        'passed_diagnostic_gate': (shared_result['nll'] <= 1.05 * control_result['nll']
                                   and shared_result['teacher_next_token_agreement'] >= .9
                                   and isa_match),
        'isa_layout_outputs_bit_exact': isa_match,
        'ordinary_image_bytes': original_engine.image.nbytes,
        'shared_image_bytes': shared_engine.image.nbytes,
        'limitations': [
            'Synthetic token arithmetic does not measure language usefulness.',
            'This is a tiny Qwen-compatible diagnostic, not the pretrained Qwen3-0.6B checkpoint.',
            'An initial run scored this same synthetic validation seed, then failed when constructing the ISA config. This rerun is diagnostic only and cannot be treated as independent held-out evidence.',
            'ISA equivalence does not establish RTL timing or physical FPGA benefit.',
            'No consumed quality suite or benchmark was opened or rerun.'
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'teacher_nll': teacher_result['nll'],
                      'ordinary_nll': control_result['nll'],
                      'shared_nll': shared_result['nll'],
                      'shared_teacher_agreement': shared_result['teacher_next_token_agreement'],
                      'passed_diagnostic_gate': report['passed_diagnostic_gate']}))


if __name__ == '__main__':
    main()
