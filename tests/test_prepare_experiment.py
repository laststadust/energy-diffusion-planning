"""CPU-only deterministic selection/resumption checks, without nuPlan datasets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ddv2_energy.contexts import sha256_file
from ddv2_energy.prepare_experiment import (split_logs, select_windows, check_selection,
                                 verify_log_artifact, _publish_context, write_json)
from tests.test_contexts import synthetic_context


def frames(count=100):
    result = []
    for i in range(count):
        result.append(dict(token=f'token_{i}', log_name='log_a', map_location='boston',
                           timestamp=10**15 + i * 500000, roadblock_ids=['route'],
                           ego2global_translation=[0., 0., 0.], ego2global_rotation=[1.,0.,0.,0.],
                           ego_dynamic_state=np.array([2.,0.,0.,0.]),
                           anns=dict(gt_boxes=np.array([[10.,0.,0.,4.,2.,1.7,0.]], dtype=np.float32),
                                     gt_names=['vehicle'], track_tokens=['actor_1'],
                                     gt_velocity_3d=np.zeros((1,3), dtype=np.float32))))
    return result


class SelectionTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_official_only(self):
        train = ['log'+str(i) for i in range(30)]
        first = split_logs(train, ['test'], train+['test','unknown'], 0)
        self.assertEqual(first, split_logs(train[::-1], ['test'], train, 0))
        self.assertEqual(len({n for n,s in first}),30)
        self.assertEqual(sum(s=='val' for n,s in first),6)
        self.assertNotIn('test', [n for n,s in first])
        with self.assertRaises(ValueError):
            split_logs(train, [train[0]], train, 0)
        with self.assertRaises(ValueError):
            split_logs(train, ['test'], ['unknown'], 0)

    def test_exact_window_boundaries_spacing_and_seed(self):
        selected, rejects = select_windows(frames(), 'log_a', 'train', 5, 0, 8, 32)
        repeated, _ = select_windows(frames(), 'log_a', 'train', 5, 0, 8, 32)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected),5)
        indices = sorted(r['frame_index'] for r in selected)
        self.assertTrue(all(3 <= i < 90 for i in indices))
        self.assertTrue(all(b-a >=8 for a,b in zip(indices,indices[1:])))
        self.assertEqual([r['token'] for r in selected], ['token_'+str(r['frame_index']) for r in selected])

    def test_official_token_allowlist_not_only_log_filter(self):
        selected, rejected = select_windows(frames(), 'log_a', 'train', 5, 0, 8, 32, {'token_20','token_50'})
        self.assertEqual({r['token'] for r in selected},{'token_20','token_50'})
        self.assertGreater(rejected['not_official_navtrain_token'],0)

    def test_no_route_partial_future_cap_and_time_are_excluded(self):
        invalid = frames(30)
        for frame in invalid:
            frame['roadblock_ids'] = []
        selected, rejected = select_windows(invalid,'log_a','train',2,0,8,32)
        self.assertEqual(selected,[]); self.assertGreater(rejected['missing_route'],0)
        invalid = frames(30)
        for i, frame in enumerate(invalid):
            frame['anns']['track_tokens'] = ['different_actor_'+str(i)]
        selected, rejected = select_windows(invalid,'log_a','train',2,0,8,32)
        self.assertEqual(selected,[]); self.assertGreater(rejected['incomplete_future_actor_labels'],0)
        invalid = frames(30)
        for frame in invalid:
            ann=frame['anns']; ann['gt_boxes']=np.repeat(ann['gt_boxes'],2,axis=0)
            ann['gt_names']=['vehicle','vehicle']; ann['track_tokens']=['a','b']
            ann['gt_velocity_3d']=np.zeros((2,3),dtype=np.float32)
        selected,rejected=select_windows(invalid,'log_a','train',2,0,8,1)
        self.assertEqual(selected,[]); self.assertGreater(rejected['current_actor_cap_exceeded'],0)
        invalid = frames(30)
        for frame in invalid:
            frame['timestamp'] += int(frame['token'].split('_')[1])*100000
        selected,rejected=select_windows(invalid,'log_a','train',2,0,8,32)
        self.assertEqual(selected,[]); self.assertGreater(rejected['invalid_time_ego_or_reverse'],0)

    def test_missing_or_duplicate_token_and_log_mismatch_fail(self):
        for mode in ('missing','duplicate','log'):
            value=frames()
            if mode=='missing': value[0].pop('token')
            elif mode=='duplicate': value[1]['token']=value[0]['token']
            else:
                for frame in value: frame['log_name']='different'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                select_windows(value,'log_a','train',2,0,8,32)

    def test_counts_overlap_and_temporal_violations_fail(self):
        rows=[dict(token='a',log='log_a',split='train',frame_index=3),
              dict(token='b',log='log_a',split='train',frame_index=11),
              dict(token='c',log='log_b',split='val',frame_index=3)]
        check_selection(rows,2,1,8,8)
        cases=[]
        value=copy.deepcopy(rows); value[-1]['log']='log_a'; cases.append(value)
        value=copy.deepcopy(rows); value[1]['frame_index']=4; cases.append(value)
        value=copy.deepcopy(rows); value[1]['token']='a'; cases.append(value)
        cases.append(rows[:-1])
        for value in cases:
            with self.assertRaises(ValueError): check_selection(value,2,1,8,8)
        with self.assertRaises(ValueError): check_selection(rows,2,1,1,8)


class ResumeTests(unittest.TestCase):
    def test_context_resume_identical_only(self):
        with tempfile.TemporaryDirectory() as folder:
            context=synthetic_context()
            # Canonicalize exactly as production context_from_window does.
            from ddv2_energy.contexts import validate_context
            context=validate_context(context)
            first=_publish_context(folder,context)
            self.assertEqual(first,_publish_context(folder,context))
            changed=copy.deepcopy(context); changed['v0']+=1
            with self.assertRaises(ValueError): _publish_context(folder,changed)

    def test_completed_artifact_hash_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'owned.json'
            write_json(path,dict(state='complete'))
            artifact=dict(file_sha256={str(path):sha256_file(path)})
            self.assertIs(verify_log_artifact(artifact),artifact)
            write_json(path,dict(state='changed'))
            with self.assertRaises(ValueError): verify_log_artifact(artifact)
            path.unlink()
            with self.assertRaises(ValueError): verify_log_artifact(artifact)


if __name__=='__main__':
    unittest.main()
