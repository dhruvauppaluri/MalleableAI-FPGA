import unittest

from malleable.llm.relative_predictor import fit, intervals
from malleable.llm.policy_features import FEATURES, extract, features


def rows(start, count):
    result=[]
    for i in range(start,start+count):
        for action,ratio in (('balanced',1),('compute',.9)):
            record=dict(schema_version=1,provenance='compiler-pre-execution',model_id='m',
                        configuration_id=action,compiler_id='compiler',workload_id=str(i),
                        features={k:1 for k in FEATURES})
            record['features']['prompt_tokens']=i+1
            result.append(dict(group_id=str(i),evidence_id=f'{i}-{action}',features=record,
                               action=action,cycles=(100+i)*ratio,provenance='measured-rtl'))
    return result


class PredictorTests(unittest.TestCase):
    def test_compiler_extraction_changes_with_position(self):
        from llm_fixture import tiny
        from malleable.llm.records import PERSONALITIES
        _,_,spec=tiny(); cfg=PERSONALITIES['balanced'].config(spec,128)
        a=extract(spec,cfg,'balanced',model_id='m',workload_id='w',positions=[0],prefill_steps=1)
        b=extract(spec,cfg,'balanced',model_id='m',workload_id='w',positions=[0,1],prefill_steps=1)
        self.assertNotEqual(features(a),features(b))
        self.assertEqual(len(a['program_hashes']),1)

    def test_grouped_training_and_calibration(self):
        model=fit(rows(0,10),rows(10,10))
        self.assertEqual(model['status'],'candidate')
        query={r['action']:r['features'] for r in rows(20,1)}
        bounds=intervals(model,query)
        self.assertLess(bounds['compute'][1],bounds['balanced'][1])
        with self.assertRaises(ValueError): fit(rows(0,10),rows(9,10))

    def test_small_calibration_abstains(self):
        model=fit(rows(0,10),rows(10,3))
        self.assertIsNone(model['log_radius'])
        with self.assertRaises(ValueError): intervals(model,{r['action']:r['features'] for r in rows(20,1)})
