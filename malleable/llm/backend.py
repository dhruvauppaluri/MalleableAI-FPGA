"""Whole-program RTL execution with a matching independent ISA checker."""
import hashlib
import json
from pathlib import Path
import re
import time
from . import upstream


def diagnose_bottleneck(roofline, useful_mac_utilization, mxu_gaps, cycles):
    """Classify only measured simulation dimensions; keep all raw evidence visible.

    The thresholds are deliberately conservative heuristics, not claims about
    physical FPGA or external-memory behavior.
    """
    dram_bound_cycles = int(roofline.get('bound', 0))
    dram_fraction = float(roofline.get('efficiency', 0.0))
    mac_util = float(useful_mac_utilization)
    gaps = mxu_gaps.get('gaps', [])
    dependency_gap_cycles = sum(max(0, int(end)-int(start)) for start,end,*_ in gaps)
    dependency_fraction = dependency_gap_cycles/max(1,int(cycles))
    gap_summaries = []
    for row in gaps:
        start, end, *rest = row
        gap_summaries.append({
            'start_cycle': int(start), 'end_cycle': int(end),
            'length_cycles': max(0, int(end)-int(start)),
            'dependency_pc': int(rest[0]) if rest else None,
        })
    largest_gaps = sorted(gap_summaries,
        key=lambda gap:gap['length_cycles'], reverse=True)[:5]
    signals = {
        'simulated_axi_memory': dram_fraction >= 0.90,
        'compute': mac_util >= 0.70,
        'dependency_or_controller': dependency_fraction >= 0.20,
    }
    active = [name for name, present in signals.items() if present]
    if len(active) > 1:
        classification = 'mixed'
    elif active:
        classification = active[0].replace('_', '-') + '-limited'
    else:
        classification = 'uncertain'
    return {
        'classification': classification,
        'signals': signals,
        'evidence': {
            'dram_bound_cycles': dram_bound_cycles,
            'dram_bound_fraction': dram_fraction,
            'useful_mac_utilization': mac_util,
            'dependency_gap_cycles': dependency_gap_cycles,
            'dependency_gap_fraction': dependency_fraction,
            'prologue_cycles': int(mxu_gaps.get('prologue', 0)),
            'epilogue_cycles': int(mxu_gaps.get('epilogue', 0)),
            'dependency_gap_count': len(gaps),
            'largest_dependency_gaps': largest_gaps,
            'physical_external_memory': 'unavailable',
        },
        'method': 'conservative-simulation-heuristic-v1',
        'provenance': 'rtl-trace-derived',
    }


class CheckedRtlBackend:
    def __init__(self, cfg, images, workload, root, emit=lambda *_: None):
        import numpy as np
        from opentpu.llm.qwen3 import IsaBackend
        from opentpu import rtlsim
        self.cfg, self.workload = cfg, workload
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.emit = emit; self.step_index = 0
        self.drams = [np.asarray(i, np.uint8).copy() for i in images]
        self.reference = IsaBackend(cfg, images)
        from .records import PERSONALITIES
        self.uarch = PERSONALITIES[workload.personality].uarch
        emit('phase', {'phase':'compiling', 'backend':'rtl'})
        started = time.monotonic()
        self.executable = rtlsim.build_top(cfg, 20, self.uarch, axi=True)
        self.compilation_seconds = time.monotonic()-started
        self.build_id = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        emit('build', {'host_seconds':self.compilation_seconds, 'build_id':self.build_id})

    def write(self, s, addr, data):
        import numpy as np
        value = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if addr < 0 or addr+value.size > len(self.drams[s]): raise ValueError('write outside image')
        self.drams[s][addr:addr+value.size] = value
        self.reference.write(s, addr, data)

    def read(self, s, addr, nbytes):
        if addr < 0 or addr+nbytes > len(self.drams[s]): raise ValueError('read outside image')
        return self.drams[s][addr:addr+nbytes].copy()

    def run(self, programs):
        import numpy as np
        from opentpu import rtlsim, profile, lens
        directory = self.root / f'step-{self.step_index:05d}'
        self.emit('phase', {'phase':'rtl-execution','step':self.step_index})
        last = 0.; summary = {'step':self.step_index, 'cycles':0, 'trace_events':0}
        def line(text):
            nonlocal last
            match = re.search(r'(?:c=|cycles=)(\d+)', text)
            if match: summary['cycles'] = int(match[1])
            if text.startswith('T'): summary['trace_events'] += 1
            now = time.monotonic()
            if now-last >= .1:
                self.emit('activity', dict(summary)); last = now
        started = time.monotonic()
        drams, tmems, stats = rtlsim.run(self.cfg, programs, self.drams,
            keep=directory, max_cycles=1<<40, trace=True, uarch=self.uarch,
            axi=True, boot=False, stall=self.workload.stall_percent,
            seed=self.workload.seed+1, bw=self.workload.bandwidth_percent,
            lat=self.workload.latency, on_line=line)
        stats['host_execution_seconds'] = time.monotonic()-started
        self.emit('phase', {'phase':'isa-validation','step':self.step_index})
        # TMEM is per-program scratch; persistent model state lives in DRAM.
        for s in self.reference.machine.slices: s.tmem.fill(0)
        self.reference.run(programs)
        for i, state in enumerate(self.reference.machine.slices):
            n = len(self.drams[i])
            for label,a,b in [('DRAM',drams[i][:n],state.dram[:n]), ('TMEM',tmems[i],state.tmem)]:
                if not np.array_equal(a,b):
                    first = int(np.flatnonzero(a != b)[0])
                    raise ValueError(f'RTL/ISA {label} mismatch at slice {i}, index {first}')
        self.drams = [d[:len(old)].copy() for d,old in zip(drams,self.drams)]
        trace = stats.pop('trace')
        parsed = profile.parse(trace,self.cfg,programs,name=f'token-step-{self.step_index}')
        data = lens.to_data(parsed,self.uarch,kind='rtl')
        stats['useful_macs']=int(data['macs'])
        stats['useful_mac_utilization']=float(data['macs']/max(1,stats['cycles']*data['peak_macs']))
        stats['axi_read_bytes']=sum(beats*64 for _,beats in stats.get('axi_reads',[]))
        stats['axi_read_bytes_per_cycle']=stats['axi_read_bytes']/max(1,stats['cycles'])
        stats['bottleneck']=diagnose_bottleneck(data['roofline'],
            stats['useful_mac_utilization'],data['mxu_gaps'],stats['cycles'])
        data['clock_mhz']=self.workload.clock_hz/1e6 if self.workload.clock_hz else None
        data['measurement_provenance']='RTL cycles; clock projections are assumed, not physical'
        (directory/'profile.json').write_text(json.dumps({'format':lens.FORMAT,'version':lens.VERSION,'profiles':[data]}))
        # Keep complete trace/assembled programs out of browser event payloads.
        trace_path = directory/'trace.txt'
        stats.update(validation='bit-exact-dram-and-tmem', trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                     profile_path=str(directory/'profile.json'), step=self.step_index,
                     memory_model='generic-two-channel-axi')
        # Binary snapshots are temporary verification intermediates, not model
        # artifacts. Keeping two copies of every real-model image per token
        # would exhaust disk. Persist traces/programs/profiles, not these dumps.
        for i in range(self.cfg.S):
            for filename in (f'dram_{i}.bin',f'dram_out_{i}.bin',f'tmem_{i}.hex'):
                (directory/filename).unlink(missing_ok=True)
        self.emit('counters',stats)
        self.step_index += 1
        return stats
