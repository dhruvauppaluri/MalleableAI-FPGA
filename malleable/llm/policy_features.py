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
