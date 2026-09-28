"""Independent arbitrary-precision reference, features, and benchmark models."""
import random
from .records import ModelArtifact, integer, identity


def analyze(model):
    layers = []
    for layer in model.layers:
        n, k = len(layer['weights']), len(layer['weights'][0])
        layers.append(dict(inputs=k, outputs=n, macs=n*k, parameter_bytes=n*k+9*n,
                           live_activation_bytes=k+n, accumulator_bound=k*16384,
                           biased_bounds=[[-k*16384+b,k*16384+b] for b in layer['biases']]))
    return dict(model_id=model.model_id, family='dense', operators=layers,
                macs=sum(x['macs'] for x in layers),
                parameter_bytes=sum(x['parameter_bytes'] for x in layers),
                live_activation_bytes=max(x['live_activation_bytes'] for x in layers),
                topology_hash=identity([(x['inputs'],x['outputs']) for x in layers]),
                weights_hash=identity([x['weights'] for x in model.layers]),
                numeric_hash=identity([{k:v for k,v in x.items() if k != 'weights'} for x in model.layers]))


def requantize(acc, bias, multiplier, shift, relu):
    product = (acc + bias) * multiplier
    quotient, remainder = divmod(abs(product), 1 << shift)
    if shift and remainder * 2 >= (1 << shift):
        quotient += 1
    signed = -quotient if product < 0 else quotient
    saturated = min(127, max(-128, signed))
    return max(0, saturated) if relu else saturated


def reference(model, inputs, active_lanes=4):
    integer(active_lanes,1,8)
    if len(inputs) != len(model.layers[0]['weights'][0]):
        raise ValueError('Input shape mismatch')
    for x in inputs:
        integer(x, -128, 127)
    values, intermediates, overflow = list(inputs), [], False
    for layer in model.layers:
        outputs = []
        for j, row in enumerate(layer['weights']):
            acc = 0
            for begin in range(0, len(row), active_lanes):
                acc += sum(a*b for a, b in zip(values[begin:begin+active_lanes],
                                              row[begin:begin+active_lanes]))
                overflow |= not -2**31 <= acc < 2**31
                acc = (acc + 2**31) % 2**32 - 2**31
            outputs.append(requantize(acc, layer['biases'][j], layer['multipliers'][j],
                                      layer['shifts'][j], layer['relu']))
        values = outputs
        intermediates.append(outputs)
    return values, intermediates, overflow


def fixture(size='light', seed=0):
    rng = random.Random(seed)
    dims = {'light': [7, 5, 3], 'medium': [32, 32, 16],
            'heavy': [64, 64, 64, 64, 64]}[size]
    return ModelArtifact([dict(weights=[[rng.randint(-128, 127) for _ in range(k)]
                                       for _ in range(n)],
                              biases=[rng.randint(-1000, 1000) for _ in range(n)],
                              multipliers=[rng.randint(1, 16) for _ in range(n)],
                              shifts=[rng.randint(10, 15) for _ in range(n)],
                              relu=i < len(dims)-2)
                          for i, (k, n) in enumerate(zip(dims, dims[1:]))])
