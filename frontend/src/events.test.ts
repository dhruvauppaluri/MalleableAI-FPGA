import {describe,it,expect} from 'vitest';
import {initial,reduce} from './events';
describe('ordered durable run events',()=>{
 it('rejects duplicate and stale-run events',()=>{let s=initial('a');s=reduce(s,{id:1,run_id:'a',kind:'token',payload:{text:'hello'}});expect(reduce(s,{id:1,run_id:'a',kind:'token',payload:{text:'bad'}})).toBe(s);expect(reduce(s,{id:2,run_id:'b',kind:'status',payload:{status:'completed'}})).toBe(s);});
 it('recovers by replaying ordered events',()=>{let s=initial('a');s=reduce(s,{id:2,run_id:'a',kind:'phase',payload:{phase:'prefill'}});s=reduce(s,{id:3,run_id:'a',kind:'status',payload:{status:'cancelled'}});expect(s.phase).toBe('prefill');expect(s.status).toBe('cancelled');expect(s.last).toBe(3);});
 it('retains profile evidence after the bounded log rolls over',()=>{let s=reduce(initial('a'),{id:1,run_id:'a',kind:'counters',payload:{step:0,cycles:100}});for(let i=2;i<250;i++)s=reduce(s,{id:i,run_id:'a',kind:'activity',payload:{step:1,cycles:i,trace_events:i}});expect(s.profiles).toEqual([0]);expect(s.events.length).toBe(200);expect(s.counters?.cycles).toBe(100);});
 it('shows terminal-only GPU output when no per-token event stream is available',()=>{const s=reduce(initial('gpu'),{id:1,run_id:'gpu',kind:'result',payload:{text:'verified output'}});expect(s.text).toBe('verified output');});
});
