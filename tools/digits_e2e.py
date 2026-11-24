#!/usr/bin/env python3
"""Stage 2: train an FP32 digits MLP, quantize it, and run it on the golden model and RTL.

The data is sklearn's bundled ``load_digits`` (no download). It uses a fixed
stratified split and a recorded seed. The FP32 MLP is 64 -> 32 (ReLU) -> 10, trained on
the raw pixel values (0..16). Its only scaling is the quantizer's input calibration.

Outputs (written to --out, default docs/results/digits):
  digits-int8.json    quantized artifact, scales, seed, split, tool versions
  predictions.json    per-test-sample FP32/INT8 predictions and INT8 logits
  summary.json        accuracies, argmax ties, RTL bit-exact verdict, counters

FP32/INT8 accuracies are measured on the host. The RTL numbers come from Icarus
simulation and are labeled measured:rtl-simulation.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from malleable.backend import RTLBackend  # noqa: E402
from malleable.model import reference  # noqa: E402
from malleable.quantize import quantize_model  # noqa: E402
from malleable.records import ExecutionConfig, identity  # noqa: E402

SEED = 0
TEST_FRACTION = 0.2


def argmax_with_ties(values):
    best = max(values)
    winners = [i for i, v in enumerate(values) if v == best]
    return winners[0], len(winners) > 1


def train(seed=SEED):
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=TEST_FRACTION, stratify=digits.target,
        random_state=seed)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                        max_iter=1000, random_state=seed)
    mlp.fit(x_train, y_train)
    return mlp, x_train, x_test, y_train, y_test


def float_layers(mlp):
    """sklearn coefs_ are (inputs, outputs); the quantizer wants rows per output."""
    return [dict(weights=mlp.coefs_[0].T.tolist(), biases=mlp.intercepts_[0].tolist(), relu=True),
            dict(weights=mlp.coefs_[1].T.tolist(), biases=mlp.intercepts_[1].tolist(), relu=False)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'results' / 'digits'))
    parser.add_argument('--lanes', type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mlp, x_train, x_test, _, y_test = train()
    fp32_pred = mlp.predict(x_test)
    fp32_acc = float((fp32_pred == y_test).mean())

    # Calibration uses the training split only. Inputs are mapped exclusively by
    # QuantizedModel.quantize_input (the quantizer's calibrated input scale).
    q = quantize_model(float_layers(mlp), x_train.tolist())
    inputs = [q.quantize_input(row.tolist()) for row in x_test]

    golden, ties, overflow = [], 0, False
    for x in inputs:
        out_values, _, ovf = reference(q.artifact, x, args.lanes)
        golden.append(out_values)
        overflow |= ovf
    int8_pred = []
    for logits in golden:
        label, tie = argmax_with_ties(logits)
        int8_pred.append(label)
        ties += tie
    int8_acc = float((np.array(int8_pred) == y_test).mean())
    agreement = float((np.array(int8_pred) == fp32_pred).mean())

    started = time.perf_counter()
    rtl_error = None
    try:
        run = RTLBackend().run(q.artifact, inputs, ExecutionConfig(args.lanes, args.lanes))
        rtl_outputs, counters = run['outputs'], run['counters']
    except Exception as exc:  # a mismatch is a bug report, recorded verbatim
        rtl_error = f'{type(exc).__name__}: {exc}'[:2000]
        rtl_outputs, counters, run = None, [], {}
    wall = time.perf_counter() - started
    mismatches = (None if rtl_outputs is None else
                  [i for i, (a, b) in enumerate(zip(rtl_outputs, golden)) if a != b])
    bit_exact = rtl_outputs is not None and not mismatches and len(rtl_outputs) == len(golden)
    rtl_pred = [argmax_with_ties(o)[0] for o in rtl_outputs] if rtl_outputs else None
    rtl_acc = float((np.array(rtl_pred) == y_test).mean()) if rtl_pred else None

    lanes = args.lanes
    per_request = counters[0] if counters else {}
    counters_constant = bool(counters) and all(c['cycles'] == per_request['cycles'] for c in counters)

    artifact = dict(
        schema_version=1, description='sklearn load_digits MLP 64->32(ReLU)->10, contract v0.2 INT8',
        artifact=q.artifact.layers, model_id=q.artifact.model_id,
        input_scale=q.input_scale, output_scales=q.output_scales,
        weight_scales=q.weight_scales, real_multipliers=q.real_multipliers,
        clipped_biases=q.clipped_biases,
        training=dict(seed=SEED, split='train_test_split(test_size=0.2, stratify=y, random_state=0)',
                      train_count=len(x_train), test_count=len(x_test),
                      model="MLPClassifier(hidden_layer_sizes=(32,), activation='relu', "
                            "solver='adam', max_iter=1000, random_state=0)",
                      input_preprocessing='none (raw 0..16 pixels); INT8 via quantizer calibration',
                      calibration='min/max over the training split',
                      n_iter=int(mlp.n_iter_)),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      sklearn=sklearn.__version__))
    (out / 'digits-int8.json').write_text(json.dumps(artifact, indent=1) + '\n')
    (out / 'predictions.json').write_text(json.dumps(dict(
        labels=y_test.tolist(), fp32=fp32_pred.tolist(), int8_golden=int8_pred,
        int8_rtl=rtl_pred, int8_logits=golden), separators=(',', ':')) + '\n')
    summary = dict(
        schema_version=1, model_id=q.artifact.model_id,
        fp32_test_accuracy=dict(value=fp32_acc, provenance='measured:host-fp32'),
        int8_test_accuracy=dict(value=int8_acc, provenance='measured:golden-model(model.reference)'),
        int8_rtl_test_accuracy=dict(value=rtl_acc, provenance='measured:rtl-simulation(icarus)'),
        fp32_int8_prediction_agreement=agreement,
        int8_argmax_ties=ties, reference_overflow=overflow,
        test_count=len(inputs), clipped_biases=len(q.clipped_biases),
        rtl=dict(lanes=lanes, active_lanes=lanes, bit_exact=bit_exact,
                 mismatching_samples=mismatches, error=rtl_error,
                 counters_identical_across_requests=counters_constant,
                 per_request_counters=per_request,
                 util_accept=(per_request['macs'] / (per_request['compute'] * lanes)) if per_request else None,
                 util_wall=(per_request['macs'] / (per_request['cycles'] * lanes)) if per_request else None,
                 provenance='measured:rtl-simulation(icarus)', build_hash=run.get('build_hash'),
                 tool=run.get('tool'), host_wall_seconds=wall),
        source=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip(),
        artifact_identity=identity(q.artifact.layers))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('fp32_test_accuracy', 'int8_test_accuracy',
                                               'int8_argmax_ties')}, indent=1))
    print('RTL bit-exact:', bit_exact, 'counters:', per_request, 'error:', rtl_error)
    return 0 if bit_exact else 1


if __name__ == '__main__':
    sys.exit(main())
