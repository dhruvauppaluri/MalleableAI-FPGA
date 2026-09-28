import random
import tempfile
import unittest
from malleable.records import *
from malleable.model import fixture, reference, requantize, analyze
from malleable.store import Store
from malleable.backend import RTLBackend
from malleable.optimize import Predictor, decide, cost
from malleable.learning import train, evaluate, promote
from malleable.experiment import dispatch, benchmark_sequence, percentile


class SystemTests(unittest.TestCase):
    def test_reference_rounding(self):
        for v,expected in [(1,1),(-1,-1),(3,2),(-3,-2),(255,127),(-257,-128)]:
            self.assertEqual(requantize(v,0,1,1,False),expected)
        self.assertEqual(requantize(-2**31,-2**31,2**31-1,62,False),-2)

    def test_artifact_validation(self):
        model = fixture()
        self.assertEqual(model.model_id,fixture().model_id)
        self.assertNotEqual(model.model_id,fixture(seed=1).model_id)
        model.layers[0]['weights'][0][0]=128
        with self.assertRaises(ValueError):
            ModelArtifact(model.layers)

    def test_store(self):
        with tempfile.TemporaryDirectory() as path:
            store=Store(path)
            key=store.save('model',fixture())
            self.assertEqual(store.load(key)['schema_version'],1)
            self.assertEqual(len(store.records('model')),1)
            store.close()

    def test_break_even(self):
        model=fixture('heavy')
        current=ExecutionConfig(1,1)
        short=decide(model,WorkloadSpec(horizon=1),HardwareProfile(switch_cycles=10**9),current)
        self.assertNotEqual(short.action,'switch-personality')
        long=decide(model,WorkloadSpec(horizon=1000),HardwareProfile(switch_cycles=1),current)
        self.assertEqual(long.action,'switch-personality')
        unknown=decide(model,WorkloadSpec(),HardwareProfile(),current)
        self.assertNotEqual(unknown.action,'switch-personality')
        with self.assertRaises(ValueError):
            decide(model,WorkloadSpec(),HardwareProfile(),current,'energy')
        residence=decide(model,WorkloadSpec(horizon=1000),HardwareProfile(switch_cycles=1),current,residence=0)
        self.assertNotEqual(residence.action,'switch-personality')

    def test_energy_deadline_and_residency(self):
        model=fixture()
        c=ExecutionConfig()
        hardware=HardwareProfile(switch_cycles=100,power_watts={'1':1,'2':1,'4':1,'8':1})
        with self.assertRaises(ValueError):
            decide(model,WorkloadSpec(deadline_cycles=1),hardware,c,'energy')
        p=Predictor(); w=WorkloadSpec(horizon=1)
        cold=cost(model,c,w,hardware,'latency',p,c)[0]
        hot=cost(model,c,w,hardware,'latency',p,c,model.model_id)[0]
        self.assertGreater(cold,hot)
        with self.assertRaises(ValueError):
            ExecutionConfig(4,1.5)

    def test_rtl_numeric_boundaries(self):
        model=ModelArtifact([dict(weights=[[-128],[127],[1],[-1]],
            biases=[-2**31,2**31-1,0,0],multipliers=[2**31-1,2**31-1,1,1],
            shifts=[62,62,1,1],relu=False)])
        for lanes in (1,8):
            RTLBackend().run(model,[[-128],[127],[1],[-1]],ExecutionConfig(lanes,lanes))

    def test_predictor(self):
        predictor=Predictor()
        for size in ['light','medium','heavy']:
            for lanes in [1,2,4,8]:
                model=fixture(size)
                config=ExecutionConfig(lanes,lanes)
                predictor.update(model,config,100+predictor.predict(model,config))
        self.assertTrue(predictor.rmse>=0)

    def test_dispatch(self):
        jobs=[dict(id=i,model_id=str(i%2),arrival=0,service=10,deadline=100 if i==2 else None)
              for i in range(4)]
        output=dispatch(jobs,'group')
        self.assertEqual(output[0]['id'],2)
        self.assertEqual(sorted(x['id'] for x in output),list(range(4)))
        self.assertEqual(percentile([10,100],.99),100)

    def test_mixed_rtl_dispatch(self):
        models=[fixture(seed=31),fixture(seed=32)]
        requests=[dict(id=i,model_id=models[i%2].model_id,arrival=0,input=[i]*7)
                  for i in range(4)]
        with tempfile.TemporaryDirectory() as path:
            store=Store(path)
            key=benchmark_sequence({m.model_id:m for m in models},requests,ExecutionConfig(dispatch='group'),store)
            result=store.load(key)
            self.assertEqual([x['id'] for x in result['schedule']],[0,2,1,3])
            for r in requests:
                self.assertEqual(result['outputs'][str(r['id'])],reference(models[r['id']%2],r['input'])[0])
            store.close()

    def test_learning_reproducible(self):
        with tempfile.TemporaryDirectory() as path:
            store=Store(path)
            hardware=HardwareProfile(switch_cycles=100)
            key=train(hardware,store,episodes=2)
            self.assertEqual(key,train(hardware,store,episodes=2))
            report=evaluate(store.load(key),hardware,store)
            self.assertEqual(len(report['seeds']),5)
            self.assertFalse(report['eligible_for_promotion'])
            report_id=store.save('evaluation',report)
            with self.assertRaises(ValueError):
                promote(store,key,report_id,report_id)
            with self.assertRaises(ValueError):
                evaluate(store.load(key),hardware,store,seeds=range(5))
            store.close()

    def test_rtl_all_personalities_and_runtime_lanes(self):
        model=fixture()
        values=[[-128,127,0,1,-1,64,-64],[1,2,3,4,5,6,7]]
        backend=RTLBackend()
        for lanes in [1,2,4,8]:
            for active in range(1,lanes+1):
                for reuse in [True,False]:
                    with self.subTest(lanes=lanes,active=active,reuse=reuse):
                        run=backend.run(model,values,ExecutionConfig(lanes,active,reuse))
                        for counters in run['counters']:
                            self.assertEqual(counters['macs'],analyze(model)['macs'])
                            self.assertEqual(counters['tiles'],5*((7+active-1)//active)+3*((5+active-1)//active))
                            self.assertEqual(counters['cycles'],Predictor().predict(model,ExecutionConfig(lanes,active,reuse)))
                            self.assertEqual(counters['cycles'],counters['compute']+counters['overhead'])
                        first,second=run['counters']
                        self.assertEqual(second['result_reads'],3)
                        if reuse:
                            self.assertEqual(second['configuration_writes']-first['configuration_writes'],8)


if __name__=='__main__':
    unittest.main()
