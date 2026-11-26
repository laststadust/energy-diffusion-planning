"""Pure structural tests for immutable full-scale preparation (no torch import)."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from ddv2_energy.scale_data import (assemble_prepared, census_digest, census_frames, census_log,
                         combine_census, publish_immutable, sha256, verify_census_digest)


def frames(n=70, prefix='a'):
    return [{'token': prefix + str(i)} for i in range(n)]


def structural_selector(values, log, split, limit, seed, frame_gap, actor_cap, allowed):
    """Models only complete-window eligibility; scientific selector tested below."""
    assert limit >= len(values) and frame_gap == 1 and actor_cap == 128
    rows = [dict(token=values[i]['token'], log=log, split=split, frame_index=i, actor_count=0)
            for i in range(3, len(values)-10) if values[i]['token'] in allowed]
    return rows, {'not_official_navtrain_token':sum(values[i]['token'] not in allowed for i in range(3,len(values)-10))}


def selection_fixture():
    results = {}
    allowed = set()
    ordered = [('train_log','train'),('val_log','val')]
    for i, (log, split) in enumerate(ordered):
        values = frames(70, str(i)+'_')
        present = {x['token'] for x in values}
        allowed |= present
        r = census_frames(values, log, split, present, 0, 128, selector=structural_selector)
        r['identity'] = {'raw_sha256':str(i)*64}
        results[log] = r
    return combine_census(results, ordered, allowed, {'seed':0}), results, ordered, allowed


class FullCensusTests(unittest.TestCase):
    def test_all_eligible_windows_without_quota_or_spacing(self):
        values = frames()
        allowed = {f['token'] for f in values}
        r = census_frames(values,'log','train',allowed,0,128,selector=structural_selector)
        self.assertEqual(len(r['rows']),57)
        self.assertEqual(r['exclusions']['incomplete_scene_window'],13)
        self.assertEqual([x['frame_index'] for x in r['rows']],list(range(3,60)))
        self.assertEqual(r,census_frames(values,'log','train',allowed,0,128,selector=structural_selector))

    def test_official_membership_edges_and_empty_log(self):
        values=frames(); allowed={'a0','a10','a11','a69','not_in_log'}
        r=census_frames(values,'log','train',allowed,0,128,selector=structural_selector)
        self.assertEqual([x['token'] for x in r['rows']],['a10','a11'])
        self.assertEqual(r['exclusions']['incomplete_scene_window'],2)
        self.assertNotIn('not_in_log',r['official_tokens_present'])
        empty=census_frames([], 'log','train',allowed,0,128,selector=structural_selector)
        self.assertEqual(empty['rows'],[])

    def test_duplicate_token_and_unaccounted_rejections_fail(self):
        values=frames(); values[1]=values[0]
        with self.assertRaises(ValueError):census_frames(values,'log','train',{'a0'},0,128,selector=structural_selector)
        with self.assertRaises(ValueError):
            census_frames(frames(),'log','train',{'a10'},0,128,selector=lambda *a:([],{}))

    def test_combination_disjoint_deterministic_and_accounted(self):
        selected,results,ordered,allowed=selection_fixture()
        self.assertEqual(selected['counts'],{'train':57,'val':57})
        self.assertEqual(selected['eligible_count']+sum(selected['exclusions'].values()),140)
        self.assertEqual(selected,combine_census(dict(reversed(list(results.items()))),ordered,allowed,{'seed':0}))
        with_missing=combine_census(results,ordered,allowed|{'missing'}, {'seed':0})
        self.assertEqual(with_missing['missing_official_tokens'],['missing'])
        self.assertEqual(with_missing['official_token_count'],141)
        self.assertTrue(selected['coverage_complete'])

    def test_partial_wrong_split_nonofficial_duplicates_fail(self):
        selected,results,ordered,allowed=selection_fixture()
        with self.assertRaises(ValueError):combine_census({},ordered,allowed,{})
        bad=copy.deepcopy(results); bad['val_log']['rows'][0]['split']='train'
        with self.assertRaises(ValueError):combine_census(bad,ordered,allowed,{})
        bad=copy.deepcopy(results); bad['val_log']['official_tokens_present'].append('alien')
        with self.assertRaises(ValueError):combine_census(bad,ordered,allowed,{})
        with self.assertRaises(ValueError):combine_census(results,ordered+[ordered[0]],allowed,{})


class ResumeAndPublicationTests(unittest.TestCase):
    def test_immutable_identical_resume_and_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'selection.json'
            publish_immutable(p,{'rows':[1]}); digest=sha256(p)
            publish_immutable(p,{'rows':[1]}); self.assertEqual(digest,sha256(p))
            with self.assertRaises(ValueError):publish_immutable(p,{'rows':[2]})

    def test_census_resume_checks_raw_identity_and_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'log.pkl'; raw.write_bytes(pickle.dumps(frames()))
            marker=Path(folder)/'marker.json'
            job=dict(raw=str(raw),marker=str(marker),log='log',split='train',seed=0,actor_cap=128,
                     spec_sha256='a'*64,allowed_tokens={'a10'})
            result=census_frames(frames(),'log','train',{'a10'},0,128,selector=structural_selector)
            result['identity']={k:job[k] for k in ('log','split','seed','actor_cap','spec_sha256')}
            result['identity']['raw_sha256']=sha256(raw)
            result['payload_sha256']=census_digest(result)
            publish_immutable(marker,result)
            self.assertEqual(census_log(job),result)
            with self.assertRaises(ValueError):census_log(dict(job,actor_cap=32))
            bad=copy.deepcopy(result);bad['rows'][0]['frame_index']=20
            with self.assertRaises(ValueError):verify_census_digest(bad)
            raw.write_bytes(b'changed')
            with self.assertRaises(ValueError):census_log(job)

    def test_complete_only_manifest_and_engineering_subset(self):
        selection,_,_,_=selection_fixture()
        rows=[dict(r,feature_path='f',target_path='t',metric_cache_path='m') for r in selection['rows']]
        contexts=[dict(token=r['token'],log=r['log'],split=r['split'],path='contexts/'+r['token']+'.pt') for r in rows]
        artifacts=[dict(rows=rows,contexts=contexts)]
        full,ctx,engineering=assemble_prepared(selection,artifacts)
        self.assertEqual(full,rows);self.assertEqual(ctx,contexts)
        self.assertEqual(len(engineering),48)
        self.assertEqual(sum(r['split']=='train' for r in engineering),32)
        self.assertEqual(sum(r['split']=='val' for r in engineering),16)
        with self.assertRaises(ValueError):assemble_prepared(selection,[dict(rows=rows[:-1],contexts=contexts)])
        with self.assertRaises(ValueError):assemble_prepared(selection,artifacts+artifacts)


try:
    import torch  # Scientific selection checks require PyTorch; structural checks do not.
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch is required for scientific selection checks')
class ScientificSelectionTests(unittest.TestCase):
    def test_unchanged_scientific_selector_and_missing_future(self):
        from tests.test_prepare_experiment import frames as scientific_frames
        values=scientific_frames(70)
        allowed={x['token'] for x in values}
        r=census_frames(values,'log_a','train',allowed,0,128)
        self.assertEqual(len(r['rows']),57)
        for i,f in enumerate(values):f['anns']['track_tokens']=['different_'+str(i)]
        r=census_frames(values,'log_a','train',allowed,0,128)
        self.assertEqual(r['rows'],[])
        self.assertEqual(r['exclusions']['incomplete_future_actor_labels'],57)

    def test_actor_cap_never_silently_truncates(self):
        import numpy as np
        from tests.test_prepare_experiment import frames as scientific_frames
        values=scientific_frames(20)
        for f in values:
            a=f['anns'];a['gt_boxes']=np.repeat(a['gt_boxes'],129,axis=0)
            a['gt_names']=['vehicle']*129;a['track_tokens']=['actor_'+str(i) for i in range(129)]
            a['gt_velocity_3d']=np.zeros((129,3),dtype=np.float32)
        r=census_frames(values,'log_a','train',{x['token'] for x in values},0,128)
        self.assertEqual(r['rows'],[])
        self.assertEqual(r['exclusions']['current_actor_cap_exceeded'],7)


if __name__=='__main__':
    unittest.main()
