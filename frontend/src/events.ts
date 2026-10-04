export type RunEvent={id:number,run_id:string,kind:string,payload:Record<string,unknown>};
export type RunState={id:string,last:number,status:string,phase:string,text:string,cycles:number,traces:number,step:number,profiles:number[],counters:Record<string,unknown>|null,events:RunEvent[],result:Record<string,unknown>|null};
export const initial=(id:string):RunState=>({id,last:0,status:'queued',phase:'queued',text:'',cycles:0,traces:0,step:0,profiles:[],counters:null,events:[],result:null});
export function reduce(state:RunState,event:RunEvent):RunState {
  if(event.run_id!==state.id || event.id<=state.last) return state;
  const next={...state,last:event.id,events:[...state.events.slice(-199),event]};
  const p=event.payload;
  if(event.kind==='status') next.status=String(p.status);
  if(event.kind==='phase') next.phase=String(p.phase);
  if(event.kind==='token') next.text=String(p.text);
  if(event.kind==='activity'){next.cycles=Number(p.cycles);next.traces=Number(p.trace_events);next.step=Number(p.step);}
  if(event.kind==='counters'){next.cycles=Number(p.cycles);next.step=Number(p.step);next.counters=p;next.profiles=[...new Set([...state.profiles,Number(p.step)])];}
  if(event.kind==='result'){next.result=p;if(typeof p.text==='string')next.text=p.text;}
  return next;
}
