"""Read-only feasibility audit for Qwen3 cross-layer projection reuse.

Uses the already published RTL rows. It never dispatches a benchmark or opens
any quality split. DRAM-bound fractions are simulator heuristics, not a board
measurement or a breakdown of weight traffic.
"""
import json
import hashlib
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from malleable.llm import upstream  # registers the pinned local openTPU package
from malleable.llm.records import PERSONALITIES
from malleable.records import identity
from opentpu.isa import MM
from opentpu.llm.qwen3 import Spec, device_config
from tools.evaluate_exported_predictor import load_verified


MODEL = ROOT / 'build/models/Qwen3-0.6B'
OUTPUT = ROOT / 'docs/evidence/model-hardware-codesign-20261008/feasibility.json'


def main():
    spec = Spec.from_hf(MODEL)
    cfg = PERSONALITIES['balanced'].config(spec, 128)
    original = spec.image(cfg, 128)
    shared = spec.image(cfg, 128, shared_matrices=('wg', 'wu'))
    original_program = original.compile_step(0)[0]
    shared_program = shared.compile_step(0)[0]
    dram_options = {key: getattr(cfg, key) for key in
                    ('S', 'D', 'MCOLS', 'ACT_BLOCKS', 'TMEM_WORDS',
                     'IMEM_WORDS', 'LANES', 'PAIR')}
    ordinary_dram = device_config(spec, 128, **dram_options).DRAM_BYTES
    shared_dram = device_config(spec, 128, shared_matrices=('wg', 'wu'),
                                **dram_options).DRAM_BYTES
    rows = load_verified('qwen3')
    by_tape = {}
    for row in rows:
        steps = row['counters']
        key = row['benchmark_workload'] + '/' + row['personality']
        by_tape[key] = {
            'row_id': identity(row),
            'cycles': sum(step['cycles'] for step in steps),
            'axi_read_bytes': sum(step['axi_read_bytes'] for step in steps),
            'rtl_derived_dram_bound_fraction_median': statistics.median(
                step['bottleneck']['evidence']['dram_bound_fraction'] for step in steps),
            'steps': len(steps),
        }
    report = {
        'schema_version': 1,
        'kind': 'read-only-model-hardware-reuse-feasibility',
        'model': 'Qwen3-0.6B',
        'model_config_sha256': hashlib.sha256((MODEL / 'config.json').read_bytes()).hexdigest(),
        'personality': 'balanced', 'context': 128, 'weight_format': 'int8',
        'architecture': {'layers': spec.layers, 'hidden': spec.hidden,
                         'ffn': spec.ffn, 'vocab': spec.vocab,
                         'matrix_shapes_per_slice': original.mats},
        'candidate': {'shared_projections': ['mlp.gate_proj', 'mlp.up_proj'],
                      'sharing_scope': 'all layers; requires an explicitly trained tied checkpoint',
                      'original_image_bytes': original.nbytes,
                      'shared_image_bytes': shared.nbytes,
                      'saved_image_bytes': original.nbytes - shared.nbytes,
                      'saved_image_fraction': (original.nbytes - shared.nbytes) / original.nbytes,
                      'minimum_configured_dram_bytes': [ordinary_dram, shared_dram],
                      'original_layer_stride_bytes': original.LS,
                      'shared_layer_stride_bytes': shared.LS,
                      'decode_program_instructions': [len(original_program), len(shared_program)],
                      'decode_program_mm_instructions': [
                          sum(i.op == MM for i in original_program),
                          sum(i.op == MM for i in shared_program)]},
        'published_benchmark_rows': by_tape,
        'interpretation': [
            'The storage saving is an image-layout projection, not a trained-model result.',
            'The same MM program count means the tied design still performs the matrix operations.',
            'Aggregate AXI bytes do not identify weight fetches or prove runtime savings.',
            'The physical FPGA and its transition costs are unavailable here.',
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'rows': len(rows), 'saved_image_bytes': report['candidate']['saved_image_bytes'],
                      'saved_image_fraction': report['candidate']['saved_image_fraction'],
                      'mm_instructions': report['candidate']['decode_program_mm_instructions']}))


if __name__ == '__main__':
    main()
