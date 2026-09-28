"""Versioned, integrity-checked, non-pickle SSM deployment artifacts."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import struct
import zlib

MAGIC = b'MSSM\x00\x01\x00\x00'
HEADER = struct.Struct('<8sQQ32sII')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


@dataclass(frozen=True)
class SSMConfig:
    family: str = 'diagonal'
    width: int = 16
    layers: int = 2
    state_size: int = 8
    expansion: int = 2
    conv_kernel: int = 4
    vocab_size: int = 259
    schema_version: int = 1

    def __post_init__(self):
        if self.family not in ('diagonal','mamba1') or self.schema_version != 1:
            raise ValueError('unsupported SSM family/version')
        for name,limit in [('width',512),('layers',32),('state_size',64),
                           ('expansion',4),('conv_kernel',16),('vocab_size',65536)]:
            v = getattr(self,name)
            if type(v) is not int or not 1 <= v <= limit:
                raise ValueError(f'invalid {name}')
        if self.vocab_size < 259:
            raise ValueError('byte tokenizer requires at least 259 tokens')


def save(path, manifest, tensors):
    """One manifest section + one little-endian signed-int32 tensor section."""
    flat = []
    entries = {}
    for name, values in sorted(tensors.items()):
        values = list(values)
        if not all(type(v) is int and -2147483648 <= v <= 2147483647 for v in values):
            raise ValueError('invalid tensor values')
        entries[name] = {'offset': len(flat), 'length': len(values)}
        flat.extend(values)
    manifest = dict(manifest,tensors=entries,format_version=1)
    meta = canonical(manifest)
    data = struct.pack(f'<{len(flat)}i',*flat)
    digest = hashlib.sha256(meta + data).digest()
    payload = HEADER.pack(MAGIC,len(meta),len(data),digest,zlib.crc32(meta),zlib.crc32(data)) + meta + data
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_bytes(payload)
    return digest.hex()


def load(path):
    payload = Path(path).read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError('truncated artifact')
    magic,mlen,dlen,digest,mcrc,dcrc = HEADER.unpack_from(payload)
    if magic != MAGIC or len(payload) != HEADER.size + mlen + dlen or dlen % 4:
        raise ValueError('invalid artifact header/length')
    meta = payload[HEADER.size:HEADER.size + mlen]
    data = payload[HEADER.size + mlen:]
    if zlib.crc32(meta) != mcrc or zlib.crc32(data) != dcrc or hashlib.sha256(meta+data).digest() != digest:
        raise ValueError('artifact integrity check failed')
    manifest = json.loads(meta)
    if manifest.get('format_version') != 1 or manifest.get('numeric') != 'q14-prototype-v1':
        raise ValueError('unsupported artifact format/numeric contract')
    SSMConfig(**manifest['config'])
    values = list(struct.unpack(f'<{dlen//4}i',data))
    if any(v < -32768 or v > 32767 for v in values):
        raise ValueError('Q14 prototype tensors must fit signed INT16')
    tensors = {}
    for name,entry in manifest['tensors'].items():
        offset,length = entry['offset'],entry['length']
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 1 or offset+length > len(values):
            raise ValueError('invalid tensor section')
        tensors[name] = values[offset:offset+length]
    return manifest,tensors,digest.hex()
