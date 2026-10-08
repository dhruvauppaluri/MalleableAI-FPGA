"""Explicit pre-execution feature contract for future predictor fitting.

All counts refer to the exact compiled workload, not observed cycle counters.
Missing compiler instrumentation fails closed instead of guessing zeros.
Scaling must be fitted on training groups only by the future estimator.
"""
import math

FEATURES = ('layers', 'hidden_size', 'intermediate_size', 'attention_heads',
            'kv_heads', 'head_dim', 'vocab_size', 'attention_layers',
            'recurrent_layers', 'convolution_layers', 'matrix_operations',
            'prefill_instructions', 'decode_instructions', 'weight_bytes',
            'activation_bytes', 'kv_bytes', 'lm_head_bytes', 'prompt_tokens',
            'generated_tokens', 'context_tokens', 'batch_size', 'active_rows',
            'matrix_columns', 'vector_lanes', 'fifo_depth', 'memory_latency',
            'bandwidth_percent', 'stall_percent')


def features(record):
    if (record.get('schema_version') != 1
            or record.get('provenance') != 'compiler-pre-execution'
            or not all(record.get(k) for k in ('model_id', 'configuration_id', 'compiler_id', 'workload_id'))):
        raise ValueError('versioned compiler/model/configuration/workload lineage required')
    values = record.get('features', {})
    if set(values) != set(FEATURES):
        raise ValueError('complete architecture/workload features required')
    result = [values[k] for k in FEATURES]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in result):
        raise ValueError('finite nonnegative features required')
    if any(values[k] <= 0 for k in ('layers', 'hidden_size', 'vocab_size', 'batch_size',
                                   'active_rows', 'matrix_columns', 'vector_lanes', 'fifo_depth')):
        raise ValueError('positive model and hardware dimensions required')
    return result


def extract(spec, cfg, personality, *, model_id, workload_id, positions,
            prefill_steps, context=128, latency=20, stall_percent=20, bandwidth_percent=100):
    """Compile exact serial positions without weights or inference.

    Counts are static encoded instructions (loop bodies counted once); matrix
    operations counts encoded MM instructions, not dynamic MACs. Layout byte
    features include padding/scales and apply across slices. The current runtime
    executes one row and one sequence; unsupported batch features are not guessed.
    """
    from . import upstream
    from .records import PERSONALITIES
    from ..records import identity
    from opentpu.isa import MM, assemble
    if (not positions or any(type(p) is not int or not 0 <= p < context for p in positions)
            or type(prefill_steps) is not int or not 0 <= prefill_steps <= len(positions)):
        raise ValueError('explicit in-context positions and prefill boundary required')
    image = spec.image(cfg, context, 1, 1, 'int8', 'int8')
    counts, matrices, hashes = [], 0, []
    for position in positions:
        programs = image.compile_step(position)
        counts.append(sum(len(program) for program in programs))
        matrices += sum(instruction.op == MM for program in programs for instruction in program)
        hashes.append(identity([assemble(p).tolist() for p in programs]))
    kinds = getattr(spec, 'kinds', ('attn',) * spec.layers)
    recurrent = kinds.count('linear')
    convolution = kinds.count('conv') + recurrent
    values = dict(layers=spec.layers, hidden_size=spec.hidden, intermediate_size=spec.ffn,
        attention_heads=spec.n_q, kv_heads=spec.n_kv, head_dim=spec.head_dim, vocab_size=spec.vocab,
        attention_layers=kinds.count('attn'), recurrent_layers=recurrent, convolution_layers=convolution,
        matrix_operations=matrices, prefill_instructions=sum(counts[:prefill_steps]),
        decode_instructions=sum(counts[prefill_steps:]),
        weight_bytes=(image.nbytes-image.kv_bytes-image.layer0)*cfg.S,
        activation_bytes=image.layer0*cfg.S, kv_bytes=image.kv_bytes*cfg.S,
        lm_head_bytes=(image.nbytes-image.head[0])*cfg.S,
        prompt_tokens=prefill_steps, generated_tokens=len(positions)-prefill_steps,
        context_tokens=max(positions)+1, batch_size=1, active_rows=1,
        matrix_columns=cfg.MCOLS, vector_lanes=cfg.LANES,
        fifo_depth=PERSONALITIES[personality].fifo_depth, memory_latency=latency,
        bandwidth_percent=bandwidth_percent, stall_percent=stall_percent)
    record = dict(schema_version=1, provenance='compiler-pre-execution', model_id=model_id,
        configuration_id=identity({'config':cfg.__dict__, 'uarch':PERSONALITIES[personality].uarch}),
        compiler_id=identity({'revision':upstream.REVISION, 'contract':'serial-static-instruction-v1'}),
        workload_id=workload_id, program_hashes=hashes, positions=positions,
        count_semantics='static-encoded-instructions-including-loop-bodies-once', features=values)
    features(record)
    return record
