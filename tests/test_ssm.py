import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from malleable.ssm.artifact import save,load,SSMConfig
from malleable.ssm.numeric import execute,round_shift,quantize
from malleable.ssm.compiler import compile_token
from malleable.ssm.backend import reference_trace,run_rtl,generate
from malleable.ssm.optimization import choose,service_cycles
from malleable.ssm.learning import train as policy_train,evaluate as policy_evaluate
from malleable.quartus import Personality,generate as generate_quartus,parse_reports,automatic_build
from malleable.store import Store


def fixture(path,seed=0,family='diagonal',width=3):
    cfg=SSMConfig(family=family,width=width,layers=1,state_size=2)
    rng=random.Random(seed); w=cfg.width; n=cfg.state_size; tensors={}
    def weights(name,size): tensors[name]=[rng.randrange(-800,801) for _ in range(size)]
    weights('embedding.weight',259*w); tensors['final_norm']=[16384]*w
    b='blocks.0.'; tensors[b+'norm']=[16384]*w
    if family=='diagonal':
        for proj in ('in_proj','gate_proj','out_proj'):
            weights(b+proj+'.weight',w*w); tensors[b+proj+'.bias']=[0]*w
        for name in ('B','C','decay_logit'): weights(b+name,w*n)
        tensors[b+'decay']=[8192]*w*n
    else:
        inner=w*2
        for name,size in {'in_proj.weight':2*inner*w,'conv_weight':inner*4,'conv_bias':inner,
                          'x_proj.weight':(1+2*n)*inner,'dt_proj.weight':inner,
                          'dt_proj.bias':inner,'A_log':inner*n,'D':inner,'out_proj.weight':w*inner}.items():
            weights(b+name,size)
        for j in range(inner*n): tensors[b+f'decay.{j}']=[8192]*256
    import math
    tensors['lut.sigmoid']=[quantize(1/(1+math.exp(-(i-128)/64))) for i in range(256)]
    tensors['lut.silu']=[quantize(((i-128)/64)/(1+math.exp(-(i-128)/64))) for i in range(256)]
    tensors['lut.softplus']=[quantize(math.log1p(math.exp((i-128)/64))) for i in range(256)]
    return save(path,{'config':vars(cfg),'numeric':'q14-prototype-v1','tokenizer':{'kind':'byte-v1'},'trained':False},tensors)


class SSMTests(unittest.TestCase):
    def test_rounding_and_artifact_integrity(self):
        self.assertEqual(round_shift(8192),1); self.assertEqual(round_shift(-8192),-1)
        self.assertEqual(round_shift(8191),0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.mssm'; key=fixture(path)
            self.assertEqual(load(path)[2],key)
            payload=bytearray(path.read_bytes());payload[-1]^=1;path.write_bytes(payload)
            with self.assertRaises(ValueError): load(path)
        with self.assertRaises(ValueError): SSMConfig(family='mamba2')

    def test_independent_operator_randomized(self):
        rng=random.Random(915)
        instructions=[]; memory=[]; expected=[]
        for trial in range(200):
            n=rng.choice([1,3,4,5,7,16]); op=trial%6
            a=len(memory); av=[rng.choice([-32768,-8192,0,8192,32767,rng.randrange(-32768,32768)]) for _ in range(n)];memory+=av
            b=len(memory); bv=[rng.randrange(-32768,32768) for _ in range(n)];memory+=bv
            bias=len(memory);memory.append(rng.randrange(-100,100))
            lut=len(memory);memory+=list(range(-128,128));dst=len(memory);count=1 if op==0 else n;memory += [0]*count
            instructions.append([op,dst,a,b,n,lut if op==3 else 1,14 if op!=3 else 8,bias])
            expected.append((op,dst,av,bv,memory[bias]))
        instructions.append([255,0,0,0,0,0,0,0])
        # RTL oracle is arbitrary-precision reference, with disjoint instructions
        # and signed extremes; verifies all six operators and saturation counts.
        final,saturated=execute(memory,instructions)
        compiled={'memory':memory,'program':instructions,'manifest':{'config':{'width':1,'vocab_size':1}},
                  'input_addr':0,'logits_addr':0,'state_addresses':[(0,len(memory))]}
        record={'input':[memory[0]],'logits':[final[0]],'states':[final],'saturation_count':saturated}
        evidence=run_rtl(compiled,[record],8,3)
        self.assertTrue(evidence['bit_exact'])

    def test_full_tokens_all_runtime_lane_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.mssm';fixture(path)
            compiled=compile_token(path); records,_,_=reference_trace(compiled,'a',1)
            for lanes in (1,2,4,8,16):
                for active in range(1,lanes+1):
                    evidence=run_rtl(compiled,records,lanes,active)
                    self.assertTrue(evidence['bit_exact'])
                    self.assertEqual(evidence['counters'][0]['cycles'],service_cycles(compiled,active))
            result=generate(path,'a',1,backend='integer')
            self.assertIsNone(result['metrics']['tokens_per_second']['value'])
            self.assertFalse(result['model_trained'])

    def test_mamba_causal_state_and_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.mssm';fixture(path,family='mamba1')
            compiled=compile_token(path)
            records,tokens,_=reference_trace(compiled,'ab',2,seed=17,top_k=3)
            self.assertEqual(tokens,reference_trace(compiled,'ab',2,seed=17,top_k=3)[1])
            self.assertNotEqual(records[0]['states'],records[1]['states'])
            for lanes in (1,2,4,8,16):
                self.assertTrue(run_rtl(compiled,records,lanes,lanes)['bit_exact'])

    def test_invalid_instructions(self):
        for program in [[[0,0,0,0,1,0,14,0]],[[5,9,0,0,2,0,14,0]],[[17,2,0,0,1,0,14,0]]]:
            with self.assertRaises(ValueError): execute([0]*10,program)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.mssm';fixture(path)
            manifest,tensors,_=load(path);tensors['embedding.weight'].pop();save(path,manifest,tensors)
            with self.assertRaises(ValueError):compile_token(path)

    def test_known_break_even(self):
        current={'lanes':4,'active_lanes':4}
        rows=[dict(current,service_cycles=100,reload_cycles=1000),{'lanes':8,'active_lanes':8,'service_cycles':60,'reload_cycles':1000}]
        self.assertEqual(choose(rows,current,1,{'program':1000})['action'],'keep')
        self.assertEqual(choose(rows,current,1000,{'program':1000})['action'],'switch-personality')
        self.assertEqual(choose(rows,current,1000,None)['action'],'keep')
        self.assertEqual(choose(rows,current,1000,{'program':0},residence=0)['action'],'keep')
        with self.assertRaises(ValueError):choose(rows,current,10,profile='energy')
        with self.assertRaises(ValueError):choose(rows,current,100,{'program':-1})

    def test_policy_determinism_and_split_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a.mssm';b=root/'b.mssm';fixture(a,1);fixture(b,2)
            store=Store(root/'store')
            try:
                cp=policy_train([a],store,2,17,{'program':1000})
                self.assertEqual(cp,policy_train([a],store,2,17,{'program':1000}))
                with self.assertRaises(ValueError):policy_evaluate(store.load(cp),[a],store)
                report=policy_evaluate(store.load(cp),[b],store)
                self.assertEqual(len(report['seeds']),5);self.assertFalse(report['policy_promoted'])
            finally:store.close()

    def test_quartus_generation_and_fail_closed_reports(self):
        with tempfile.TemporaryDirectory() as d,patch('malleable.quartus.tool_version',return_value=None):
            personality=Personality()
            directory,request=generate_quartus(personality,d)
            self.assertEqual(request['status'],'pending-tool')
            self.assertIn('5CGXFC5C6F27C7',(directory/'personality.qsf').read_text())
            self.assertFalse(automatic_build(personality,d)['accepted'])
            output=directory/'output_files';output.mkdir()
            (output/'fit.summary').write_text('Fitter Status : Successful\nTotal block memory bits : 123,456\nTotal DSP Blocks : 12\n')
            self.assertFalse(parse_reports(directory,100)['timing_passed'])
            (output/'timing.summary').write_text('Worst-case Setup Slack : -0.1\nWorst-case Hold Slack : 0.2\n')
            self.assertFalse(parse_reports(directory,100)['timing_passed'])
            (output/'timing.summary').write_text('Worst-case Setup Slack : 0.1\nWorst-case Hold Slack : 0.2\n')
            report=parse_reports(directory,100)
            self.assertTrue(report['fit_passed'] and report['timing_passed'])
            self.assertEqual(report['resources']['values']['memory_bits'],123456)
            self.assertEqual(report['power']['source'],'unavailable')
        with self.assertRaises(ValueError):Personality(device='bad;exec foo')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),'install .[ssm] for training tests')
class TrainingTests(unittest.TestCase):
    def test_native_mamba_export_and_gpu_cpu_smoke(self):
        import torch
        from malleable.ssm.model import LanguageModel
        from malleable.ssm.compiler import export
        from malleable.ssm.gpu import benchmark
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);torch.manual_seed(29)
            model=LanguageModel(SSMConfig(family='mamba1',width=3,layers=1,state_size=2))
            model.save(root/'model');artifact=root/'model.mssm';export(model,{},artifact)
            self.assertTrue(generate(artifact,'ab',2)['evidence']['bit_exact'])
            report=benchmark(root/'model','ab',2,device='cpu')
            self.assertEqual(report['arithmetic'],'FP32');self.assertEqual(len(report['generated_tokens']),2)

    def test_cpu_training_resume_and_export(self):
        from malleable.ssm.training import register_dataset,train,quality
        from malleable.ssm.model import LanguageModel
        from malleable.ssm.compiler import export
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);text=root/'text.txt'; text.write_text('a small cat.\n\na small dog.\n\na small bird.\n\na small tree.')
            dataset=root/'dataset.json';register_dataset(text,dataset,'local-test','sha-test','CC0-test-fixture')
            config=SSMConfig(width=3,layers=1,state_size=2)
            first=train(dataset,root/'model',config,steps=1,sequence_length=4,seed=3)
            second=train(dataset,root/'model',config,steps=1,sequence_length=4,seed=3,resume=True)
            self.assertEqual(first['training_steps'],1);self.assertEqual(second['training_steps'],2)
            direct=train(dataset,root/'direct',config,steps=2,sequence_length=4,seed=3)
            self.assertEqual(second['losses'],direct['losses'])
            model,metadata=LanguageModel.load(root/'model');artifact=root/'model.mssm';export(model,metadata,artifact)
            report=quality(root/'model',artifact,dataset,max_tokens=8)
            self.assertGreater(report['tokens'],0);self.assertFalse(report['chatbot_quality_proven'])
            result=generate(artifact,'a',1)
            self.assertTrue(result['evidence']['bit_exact'])


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),'install .[ide,test] for IDE tests')
class IDETests(unittest.TestCase):
    def test_api_validation_and_persistent_jobs(self):
        from fastapi.testclient import TestClient
        from malleable.ide import create_app,Jobs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=root/'fixture.mssm';fixture(artifact)
            app=create_app(root,root/'jobs',False);client=TestClient(app)
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(len(client.get('/api/models').json()),1)
            response=client.post('/api/jobs/generate',json={'artifact':str(artifact),'prompt':'hi'})
            self.assertEqual(response.status_code,200)
            identifier=response.json()['id']
            self.assertEqual(client.post(f'/api/job/{identifier}/cancel').status_code,200)
            self.assertEqual(Jobs(root/'jobs',root,False).list()[0]['status'],'cancelled')
            self.assertEqual(client.post('/api/jobs/generate',json={'artifact':'/etc/passwd'}).status_code,400)
            self.assertEqual(client.post('/api/jobs/generate',json={'artifact':str(artifact),'shell':'echo unsafe'}).status_code,400)
            self.assertEqual(client.get('/api/models',headers={'Origin':'https://evil.example'}).status_code,403)
