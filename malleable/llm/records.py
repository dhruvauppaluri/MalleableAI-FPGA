from dataclasses import dataclass, asdict
import math

from ..records import identity

CONTRACT = 'opentpu-isa-v1-fp32-rne-ftz-block-quantized'

@dataclass(frozen=True)
class Personality:
    name: str
    matrix_columns: int
    vector_lanes: int
    fifo_depth: int
    schema_version: int = 1

    def config(self, spec, context=2048, wformat='int8'):
        from . import upstream
        from opentpu.llm.qwen3 import device_config
        if self not in PERSONALITIES.values(): raise ValueError('unregistered personality')
        return device_config(spec, context, rows=1, wformat=wformat, head_format='int8',
                             S=1,D=128,MCOLS=self.matrix_columns,LANES=self.vector_lanes,
                             ACT_BLOCKS=128,TMEM_WORDS=65536,IMEM_WORDS=65536,PAIR=False)

    @property
    def uarch(self):
        return dict(WIN=16,RPB=4,WPB=2,FIFO_DEPTH=self.fifo_depth,MXU_IMPL=0,
                    VPU_CL=self.vector_lanes//4,ULANES=self.vector_lanes)

PERSONALITIES = {p.name:p for p in [Personality('compact',2,8,128),
    Personality('balanced',4,8,512),Personality('compute',8,16,512),
    Personality('buffered',4,8,1024)]}

@dataclass(frozen=True)
class GenerationWorkload:
    prompt: str
    prompt_format: str = 'chat'
    max_new: int = 16
    context: int = 2048
    seed: int = 0
    backend: str = 'rtl'
    personality: str = 'balanced'
    wformat: str = 'int8'
    latency: int = 20
    stall_percent: int = 20
    bandwidth_percent: int = 100
    clock_hz: float | None = None
    max_host_gib: float = 16
    schema_version: int = 1
    messages: list | None = None
    input_tokens: list | None = None
    candidate_case: str | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1: raise ValueError('unsupported workload version')
        if self.backend not in ('rtl','isa'): raise ValueError('unsupported backend')
        if self.prompt_format not in ('chat','raw'): raise ValueError('invalid prompt formatting')
        if self.personality not in PERSONALITIES: raise ValueError('unsupported personality')
        if self.wformat not in ('int8','int4','fp4'): raise ValueError('unsupported numeric variant')
        for name,lo,hi in [('max_new',1,256),('context',2,2048),('seed',0,2**31-1),
                           ('latency',1,10000),('stall_percent',0,99),('bandwidth_percent',1,100)]:
            v=getattr(self,name)
            if type(v) is not int or not lo<=v<=hi: raise ValueError('invalid '+name)
        if self.context%128: raise ValueError('context allocation must be a multiple of 128')
        if not isinstance(self.prompt,str) or not self.prompt or len(self.prompt)>100000:
            raise ValueError('nonempty bounded prompt required')
        for v in (self.max_host_gib,self.clock_hz):
            if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<=0): raise ValueError('invalid resource/clock limit')
        if self.messages is not None:
            if not isinstance(self.messages,list) or not self.messages or len(self.messages)>128: raise ValueError('bounded conversation required')
            expected='user'
            for index,message in enumerate(self.messages):
                if not isinstance(message,dict) or set(message)!={'role','content'} or not isinstance(message['content'],str) or not message['content']:
                    raise ValueError('role/content message required')
                role=message['role']
                if role=='system' and index==0: continue
                if role!=expected: raise ValueError('conversation roles must alternate and end with user')
                expected='assistant' if role=='user' else 'user'
            if expected!='assistant' or sum(len(m['content']) for m in self.messages)>100000:
                raise ValueError('bounded conversation must end with user')
            if self.prompt_format!='chat': raise ValueError('conversation requires chat formatting')
        if self.candidate_case is not None and (type(self.candidate_case) is not str or not self.candidate_case
            or self.wformat!='int8'):
            raise ValueError('derived candidates apply only to INT8 workloads with an explicit case path')
        if self.input_tokens is not None and (self.messages is not None or not isinstance(self.input_tokens,list)
            or not self.input_tokens or len(self.input_tokens)>self.context or any(type(t) is not int or t<0 for t in self.input_tokens)):
            raise ValueError('bounded fixed input token tape required')
    @property
    def workload_id(self):
        # Preserve the historical v1 hash for old single-prompt workloads.
        return identity({k:v for k,v in asdict(self).items() if k not in ('messages','input_tokens','candidate_case') or v is not None})
