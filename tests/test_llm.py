import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from malleable.llm.records import GenerationWorkload,PERSONALITIES
from malleable.llm.quality import gate,suites,selection_approval,MIN_RELEASE_TARGETS
from malleable.llm.models import inspect,checkpoint
from malleable.ide import Jobs,create_app

class LlmTests(unittest.TestCase):
    def test_bottleneck_uses_categories_and_preserves_raw_bound(self):
        from malleable.llm.backend import diagnose_bottleneck
        gaps={'prologue':1,'epilogue':1,'gaps':[(10,40,3)]}
        memory=diagnose_bottleneck({'bound':950,'efficiency':.95},.2,gaps,1000)
        self.assertEqual(memory['classification'],'simulated-axi-memory-limited')
        self.assertEqual(memory['evidence']['dram_bound_cycles'],950)
        self.assertEqual(memory['evidence']['physical_external_memory'],'unavailable')
        compute=diagnose_bottleneck({'bound':100,'efficiency':.1},.8,{'gaps':[]},1000)
        self.assertEqual(compute['classification'],'compute-limited')
        dependency=diagnose_bottleneck({'bound':100,'efficiency':.1},.1,{'gaps':[(0,300,1)]},1000)
        self.assertEqual(dependency['classification'],'dependency-or-controller-limited')
        mixed=diagnose_bottleneck({'bound':950,'efficiency':.95},.8,{'gaps':[]},1000)
        self.assertEqual(mixed['classification'],'mixed')
        uncertain=diagnose_bottleneck({'bound':100,'efficiency':.1},.1,{'gaps':[]},1000)
        self.assertEqual(uncertain['classification'],'uncertain')
        many=diagnose_bottleneck({'bound':100,'efficiency':.1},.1,
            {'gaps':[(i,i+10,i) for i in range(20)]},1000)
        self.assertEqual(many['evidence']['dependency_gap_count'],20)
        self.assertEqual(len(many['evidence']['largest_dependency_gaps']),5)

    def test_workload_rejects_invalid(self):
        for kw in ({'backend':'fake'},{'personality':'aws'},{'max_new':257},{'context':2049},
                   {'max_new':True},{'max_host_gib':True},{'clock_hz':float('nan')},{'schema_version':2}):
            with self.assertRaises(ValueError): GenerationWorkload('hello',**kw)
        self.assertEqual(GenerationWorkload('hello').backend,'rtl')
        self.assertEqual([(p.matrix_columns,p.vector_lanes,p.fifo_depth) for p in PERSONALITIES.values()],
                         [(2,8,128),(4,8,512),(8,16,512),(4,8,1024)])

    def test_strict_quality_and_splits(self):
        self.assertTrue(gate(2,2.1,.9)['passed'])
        self.assertFalse(gate(2,2.101,.99)['passed'])
        self.assertFalse(gate(2,2,.8999)['passed'])
        with self.assertRaises(ValueError): gate(2,-1,1)
        with self.assertRaises(ValueError): gate(2,2,1.1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'suite.json'
            p.write_text(json.dumps({'schema_version':1,'calibration':[[1,2]],'validation':[[2,3]],'held-out':[[1,2]]}))
            with self.assertRaises(ValueError): suites(p)
        self.assertFalse(selection_approval('validation',True,MIN_RELEASE_TARGETS,True))
        self.assertFalse(selection_approval('held-out',False,MIN_RELEASE_TARGETS,True))
        self.assertFalse(selection_approval('held-out',True,MIN_RELEASE_TARGETS-1,True))
        self.assertFalse(selection_approval('held-out',True,MIN_RELEASE_TARGETS,False))
        self.assertTrue(selection_approval('held-out',True,MIN_RELEASE_TARGETS,True))

    def test_import_and_path_security(self):
        from llm_fixture import save
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model'; save(p)
            info=inspect(p,128)
            self.assertTrue(info['supported']); self.assertEqual(info['physical_fit'],'unavailable')
            old=info['base_model_id']; (p/'tokenizer_config.json').write_text('{}')
            self.assertEqual(inspect(p,128)['base_model_id'],old)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../../secret.safetensors'}}))
            with self.assertRaises(ValueError): checkpoint(p)

    def test_events_retirement_and_same_origin(self):
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'model').mkdir()
            app=create_app(root,root/'jobs',False); jobs=app.state.jobs
            identifier=jobs.create('generate',{'model':str(root/'model'),'prompt':'Hi'})
            jobs.event(identifier,'token',{'text':'<script>unsafe</script>'})
            events=jobs.events(identifier); self.assertEqual(len(events),2)
            self.assertEqual(len(jobs.events(identifier,events[0]['id'])),1)
            jobs.cancel(identifier); self.assertEqual(jobs.get(identifier)['status'],'cancelled')
            with self.assertRaises(ValueError): jobs.create('train',{})
            with self.assertRaisesRegex(ValueError,'reserved'):
                jobs.create('generate',{'model':str(root/'model'),'previous_config':'spoofed'})
            with self.assertRaises(ValueError): jobs.create('generate',{'model':'/tmp','prompt':'Hi'})
            with TestClient(app) as client:
                self.assertEqual(client.get('/api/models',headers={'Origin':'http://evil.com'}).status_code,403)
                self.assertEqual(client.post(f'/api/job/{identifier}/resume').status_code,410)
                replay=client.get(f'/api/job/{identifier}/events',headers={'Last-Event-ID':str(events[0]['id'])}).text
                self.assertNotIn('"status": "queued"',replay)
                self.assertIn('cancelled',replay)
                self.assertEqual(client.get('/api/job/bad/log').status_code,400)
            with jobs.connect() as db:
                db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,NULL)',('a'*32,'train','{}','queued',0))
                db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,NULL)',('b'*32,'generate',json.dumps({'artifact':'old.mssm'}),'queued',0))
            reopened=Jobs(root/'jobs',root,False)
            self.assertEqual(reopened.get('a'*32)['status'],'retired')
            self.assertEqual(reopened.get('b'*32)['status'],'retired')

    def test_chat_template_mapping(self):
        from malleable.llm.runtime import prompt_tokens
        class Tokenizer:
            chat_template='fixture'
            def apply_chat_template(self,*args,**kwargs): return {'input_ids':[1,2,3]}
        self.assertEqual(prompt_tokens(Tokenizer(),GenerationWorkload('hello')),[1,2,3])

    def test_auxiliary_tokenizer_path_escape(self):
        from llm_fixture import save
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'model'; save(p)
            (root/'secret.txt').write_text('not model data')
            (p/'merges.txt').symlink_to(root/'secret.txt')
            with self.assertRaises(ValueError): checkpoint(p)

    def test_benchmark_records_are_searchable(self):
        from malleable.store import Store
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); jobs=Jobs(root/'jobs',root,False)
            jobs.persist('benchmark','a'*32,{'kind':'result','payload':{'tokens':[1],'valid':True}})
            jobs.persist('benchmark','a'*32,{'kind':'candidate','payload':{'valid':False,'failure':'fixture'}})
            store=Store(root/'jobs/research')
            try: self.assertEqual(len(store.records('llm-generation')),2)
            finally: store.close()

    def test_job_result_exposes_canonical_artifact_id(self):
        from malleable.store import Store
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); jobs=Jobs(root/'jobs',root,False)
            key=jobs.persist('generate','a'*32,{'kind':'result','payload':{'status':'completed','record_id':'b'*64}})
            self.assertEqual(len(key),64); self.assertNotEqual(key,'b'*64)
            store=Store(root/'jobs/research')
            try:
                self.assertEqual(store.load(key),{'status':'completed'})
                link=store.records('llm-job-artifact')[-1]
                self.assertEqual(link['artifact_id'],key)
            finally: store.close()

    def test_missing_release_fails(self):
        from malleable.llm.release import check
        with self.assertRaises(ValueError): check(None)
        with tempfile.TemporaryDirectory() as d:
            manifest=Path(d)/'legacy.json'; manifest.write_text(json.dumps({'schema_version':2}))
            with self.assertRaisesRegex(ValueError,'schema v3'): check(manifest)

    def test_full_release_fails_closed(self):
        from malleable.llm.full_release import check
        with self.assertRaises(ValueError): check(None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'models'/'verifier').mkdir(parents=True)
            jobs=Jobs(root/'jobs',root/'models',False)
            with self.assertRaisesRegex(ValueError,'standalone_manifest'):
                jobs.create('gpu-generate',{'model':str(root/'models'/'verifier')})

    def test_fixed_benchmark_manifest_schedule(self):
        from malleable.llm.experiments import validate_performance_manifest
        from malleable.records import identity
        model={'base_model_id':'base','tokenizer_id':'tok'}; runs=[]; tape=[1,2,3]
        for workload in ('short-a','short-b'):
            for personality in PERSONALITIES:
                runs.append({'workload':workload,'personality':personality,'wformat':'int8','context':128,
                    'input_tokens':tape,'input_token_hash':identity(tape),'memory_scenario':'baseline',
                    'latency':20,'stall_percent':20,'bandwidth_percent':100})
        for wformat in ('int4','fp4'):
            runs.append({'workload':'short-a','personality':'balanced','wformat':wformat,'context':128,
                'input_tokens':tape,'input_token_hash':identity(tape),'memory_scenario':'baseline',
                'latency':20,'stall_percent':20,'bandwidth_percent':100})
        quality=[]
        for wformat in ('int4','fp4'):
            row={'base_model_id':'base','variant_id':identity({'base':'base','format':wformat,'head':'int8'}),
                'personality':'balanced','split':'validation','samples':1024,'target_count':1024,
                'suite_frozen':True,'float_nll':2.,'candidate_nll':2.,'agreement':1.}
            quality.append({'id':identity(row),'record':row})
        manifest={'schema_version':1,'base_model_id':'base','tokenizer_id':'tok','runs':runs,
            'quality_evidence':quality,'quality_evidence_ids':[q['id'] for q in quality],
            'seed':42,'fixed_prompt_hashes':{'short-a':identity(tape),'short-b':identity(tape)}}
        self.assertTrue(validate_performance_manifest(manifest,model))
        runs[-1]['bandwidth_percent']=50
        with self.assertRaises(ValueError): validate_performance_manifest(manifest,model)

    def test_optimizer_auto_apply_is_opt_in_and_heldout_gated(self):
        from fastapi.testclient import TestClient
        from malleable.store import Store
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=root/'models'/'fixture'; model.mkdir(parents=True)
            app=create_app(root/'models',root/'jobs',False); jobs=app.state.jobs
            store=Store(root/'jobs'/'research')
            try:
                model_record=store.save('llm-model',{'path':str(model)})
                base={'schema_version':1,'base_model_id':'base','variant_id':'variant',
                    'workload_id':'w','workload_identity_v2':'same-tape','workload':{
                        'prompt':'hello','prompt_format':'raw','max_new':1,'context':128,
                        'backend':'rtl','personality':'compact','wformat':'int8','seed':1,
                        'latency':20,'stall_percent':20,'bandwidth_percent':100,
                        'max_host_gib':4,'schema_version':1},'model_record':model_record,
                    'valid':True,'personality':'compact','input_token_hash':'input','tokenizer_id':'tok','configuration_id':'config',
                    'backend':'rtl','counters':[{'cycles':1000,'validation':'bit-exact-dram-and-tmem'}]}
                current=store.save('llm-generation',base)
                candidate=dict(base,personality='compute',configuration_id='other',
                    workload={**base['workload'],'personality':'compute'},
                    counters=[{'cycles':10,'validation':'bit-exact-dram-and-tmem'}])
                store.save('llm-generation',candidate)
                for personality in ('compact','compute'):
                    q={'base_model_id':'base','variant_id':'variant','personality':personality,
                       'samples':1024,'target_count':1024,'suite_frozen':True,
                       'split':'validation','float_nll':2.,'candidate_nll':2.,'agreement':1.}
                    store.save('llm-quality',q)
                store.save('llm-quality',{'base_model_id':'base','variant_id':'variant','personality':'compute',
                    'samples':1024,'target_count':1024,'split':'held-out','float_nll':2.,'candidate_nll':2.,
                    'agreement':1.,'selectable':True})
            finally: store.close()
            with TestClient(app) as client:
                response=client.post('/api/optimization-sessions',json={'current':current,
                    'window':{'remaining_requests':100,'switching_costs':{
                        'compact/int8->compute/int8':{'drain':0,'program':0,'reload':0,'warmup':0,'reprefill':0}}}})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['application'],'recommendation-only')
                automatic=client.post('/api/optimization-sessions',json={'current':current,'auto_apply':True,
                    'window':{'remaining_requests':100,'switching_costs':{
                        'compact/int8->compute/int8':{'drain':0,'program':0,'reload':0,'warmup':0,'reprefill':0}}}})
                self.assertEqual(automatic.status_code,200,automatic.text)
                result=automatic.json(); self.assertEqual(result['application'],'automatically-applied-between-windows')
                self.assertEqual(jobs.get(result['applied_job'])['status'],'queued')
                application=json.loads(jobs.get(result['applied_job'])['payload'])
                self.assertEqual(application['previous_config'],'compact/int8')
                self.assertEqual(application['optimization_session_id'],result['session_id'])

    def test_switching_break_even(self):
        from malleable.llm.optimization import decide,DecisionWindow
        current={'base_model_id':'base','valid':True,'backend':'rtl','personality':'compact','variant_id':'v',
                 'workload':{'wformat':'int8','prompt':'fixed'},'counters':[{'cycles':1000,'validation':'bit-exact-dram-and-tmem'}]}
        fast=dict(current,personality='compute',counters=[{'cycles':100,'validation':'bit-exact-dram-and-tmem'}])
        q={'base_model_id':'base','variant_id':'v','samples':1024,'target_count':1024,'suite_frozen':True,
           'split':'validation','float_nll':2,'candidate_nll':2,'agreement':1.}
        quality=[dict(q,personality=p) for p in ('compact','compute')]
        costs={'compact/int8->compute/int8':{'drain':0,'program':10000,'reload':0,'warmup':0,'reprefill':0}}
        self.assertEqual(decide([current,fast],quality,current,DecisionWindow())['chosen'],'compact/int8')
        short=decide([current,fast],quality,current,DecisionWindow(switching_costs=costs))
        long=decide([current,fast],quality,current,DecisionWindow(remaining_requests=100,switching_costs=costs))
        self.assertEqual(short['action'],'keep'); self.assertEqual(long['action'],'switch-personality')
        self.assertEqual(decide([current,fast],quality,current,DecisionWindow(remaining_requests=100,switching_costs=costs,residence_windows=0))['action'],'keep')

    def test_learning_determinism_and_group_leakage(self):
        from malleable.llm.learning import train,evaluate,state,exhaustive_episode
        episode=[[{'base_model_id':'train-base','family':'qwen3','evidence_ids':['fixture-only'],
            'provenance':'estimated-window-from-measured-rtl','service_cycles':{'compact/int8':100.,'balanced/int8':50.},
            'predicted_cycles':{'compact/int8':90.,'balanced/int8':60.},'predictor_id':'unit-fixture',
            'window':{}}]*8]
        a=train(episode,7,3); b=train(episode,7,3)
        self.assertEqual(a,b)
        with self.assertRaises(ValueError): evaluate(a,episode)
        unseen=[[dict(w,base_model_id='unseen-base',family='lfm2') for w in episode[0]]]
        self.assertEqual(evaluate(a,unseen)['mean_cost'].keys(),{'rl','fixed','heuristic','predictor','random','exhaustive'})
        legal=unseen[0][0]
        before=state(legal,'balanced/int8',1)
        changed=dict(legal,service_cycles={k:v*1000 for k,v in legal['service_cycles'].items()})
        self.assertEqual(before,state(changed,'balanced/int8',1))
        self.assertGreater(exhaustive_episode(unseen[0]),0)
        with self.assertRaises(ValueError): evaluate(a,unseen,seeds=[1,2])
        continued=train(unseen,7,1,parent=a)
        self.assertEqual(continued['training_models'],['train-base','unseen-base'])
        with self.assertRaises(ValueError): evaluate(continued,episode)
        with self.assertRaises(ValueError): train([[dict(episode[0][0],features=[float('nan')]*6)]])

if __name__=='__main__': unittest.main()
