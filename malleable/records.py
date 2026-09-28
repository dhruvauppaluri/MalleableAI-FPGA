"""Versioned, JSON-serializable public records and strict artifact validation."""
from dataclasses import dataclass, asdict, field
import hashlib
import json
import math


def canonical(value):
    return json.dumps(asdict(value) if hasattr(value, '__dataclass_fields__') else value,
                      sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Expected integer in [{low}, {high}], got {value!r}')


@dataclass
class ModelArtifact:
    layers: list
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or not 1 <= len(self.layers) <= 4:
            raise ValueError('Supported artifact: version 1, one to four dense layers')
        previous = None
        for layer in self.layers:
            if set(layer) != {'weights', 'biases', 'multipliers', 'shifts', 'relu'}:
                raise ValueError('Invalid dense layer fields')
            weights = layer['weights']
            if not weights or not weights[0]:
                raise ValueError('Empty weight matrix')
            n, k = len(weights), len(weights[0])
            integer(n, 1, 64); integer(k, 1, 64)
            if previous is not None and k != previous:
                raise ValueError('Adjacent layer dimensions do not match')
            for row in weights:
                if len(row) != k:
                    raise ValueError('Ragged weights')
                for value in row:
                    integer(value, -128, 127)
            for key, low, high in [('biases', -2**31, 2**31-1),
                                   ('multipliers', 1, 2**31-1), ('shifts', 0, 62)]:
                if len(layer[key]) != n:
                    raise ValueError('Parameter count mismatch')
                for value in layer[key]:
                    integer(value, low, high)
            if type(layer['relu']) is not bool:
                raise ValueError('relu must be boolean')
            previous = n

    @property
    def model_id(self):
        return identity(self)


@dataclass
class WorkloadSpec:
    request_count: int = 8
    arrival_pattern: str = 'sustained'
    interval_cycles: int = 100
    horizon: int = 8
    deadline_cycles: int | None = None
    seed: int = 0
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.arrival_pattern not in {'isolated', 'sustained', 'burst'}:
            raise ValueError('Invalid workload version/pattern')
        integer(self.request_count, 1, 10000)
        integer(self.interval_cycles, 0, 2**40)
        integer(self.horizon, 1, 10000)
        if self.deadline_cycles is not None:
            integer(self.deadline_cycles, 1, 2**40)


@dataclass(frozen=True)
class ExecutionConfig:
    lanes: int = 4
    active_lanes: int = 4
    reuse: bool = True
    dispatch: str = 'fifo'
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema_version != 1 or type(self.lanes) is not int or type(self.active_lanes) is not int
                or self.lanes not in (1, 2, 4, 8)
                or not 1 <= self.active_lanes <= self.lanes
                or type(self.reuse) is not bool or self.dispatch not in ('fifo', 'group')):
            raise ValueError('Unsupported execution configuration')


@dataclass
class HardwareProfile:
    clock_hz: float = 100e6
    switch_cycles: int | None = None
    # Optional calibrated watts keyed by compiled lane count. Never inferred.
    power_watts: dict = field(default_factory=dict)
    dollars_per_second: float | None = None
    operators: tuple = ('dense','relu')
    personalities: tuple = (1,2,4,8)
    max_layers: int = 4
    max_dimension: int = 64
    numeric_contract: str = 'int8-int32-bias33-product65-ties-away-v1'
    runtime_controls: tuple = ('active_lanes','reuse','dispatch')
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or not math.isfinite(self.clock_hz) or self.clock_hz <= 0:
            raise ValueError('Invalid clock/profile')
        if (tuple(self.personalities) != (1,2,4,8) or self.max_layers != 4 or self.max_dimension != 64
                or tuple(self.operators) != ('dense','relu')
                or tuple(self.runtime_controls) != ('active_lanes','reuse','dispatch')
                or self.numeric_contract != 'int8-int32-bias33-product65-ties-away-v1'):
            raise ValueError('Backend does not implement this hardware profile')
        if self.switch_cycles is not None:
            integer(self.switch_cycles, 0, 2**50)
        for power in self.power_watts.values():
            if not math.isfinite(power) or power <= 0:
                raise ValueError('Power calibration must be positive')
        if self.dollars_per_second is not None and (not math.isfinite(self.dollars_per_second) or self.dollars_per_second < 0):
            raise ValueError('Negative monetary rate')


@dataclass
class ExperimentResult:
    model_id: str
    workload_id: str
    config: dict
    metrics: dict
    provenance: dict
    valid: bool
    failure: str | None = None
    previous_config: dict | None = None
    policy_version: str = 'baseline-v1'
    schema_version: int = 1


@dataclass
class PolicyDecision:
    action: str
    config: dict
    reason: str
    predicted_cost: float
    switch_cost: float
    uncertainty: float
    schema_version: int = 1
