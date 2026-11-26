"""CPU-only cache/geometry tests; do not load a model, nuPlan map, or dataset."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ddv2_energy.contexts import (
    CURRENT_FIELDS, ContextError, ContextStore, current_only, save_context,
    sha256_file, validate_context, write_manifest,
)
from ddv2_energy.prepare_contexts import context_from_window, validate_requests


def synthetic_context(token="scene_token", log="log_a", split="train", count=2):
    actors = torch.zeros(count, 10)
    if count:
        actors[:, 0] = torch.arange(count, dtype=torch.float32) + 10
        actors[:, 3:5] = torch.tensor([4.5, 1.9])
        actors[:, 7:] = 1
    return dict(schema_version=1, token=token, log=log, split=split, dt=0.5, horizon=8,
                actors=actors, mask=torch.ones(count, dtype=torch.bool),
                sdf=torch.ones(1, 193, 193), origin=[-32, -32], resolution=0.5, v0=2.,
                future=actors.unsqueeze(0).repeat(9, 1, 1),
                future_mask=torch.ones(9, count, dtype=torch.bool),
                label_ok=True, truncated=False, gt_poses=torch.zeros(8, 3))


def raw_window(empty=False):
    """Two tracks, turning ego and permuted future ordering in world coordinates."""
    from ddv2_energy.preprocessing import rot
    result = []
    p0, yaw0 = np.array([100., -24.]), 3.08
    for h in range(9):
        loc = p0 + rot(yaw0) @ np.array([h, .03 * h * h])
        yaw = yaw0 + .03 * h
        centers = [p0 + rot(yaw0) @ np.array([10., 3. + .5*h]),
                   p0 + rot(yaw0) @ np.array([14., -2.])]
        boxes = np.array([[*((center-loc) @ rot(yaw)), 0., 4., 2., 1.7, .2-yaw]
                          for center in centers], dtype=np.float32)
        velocity = np.array([[0., 1., 0.], [np.nan, np.nan, np.nan]], dtype=np.float32)
        names, ids = ["vehicle", "traffic_cone"], ["vehicle_1", "cone_1"]
        if h % 2:
            boxes, velocity = boxes[::-1].copy(), velocity[::-1].copy()
            names, ids = names[::-1], ids[::-1]
        if empty:
            boxes, velocity, names, ids = np.empty((0, 7)), np.empty((0, 3)), [], []
        result.append(dict(token=f"frame_{h}", timestamp=10**15 + h*500000,
                           ego2global_translation=[*loc, 0.],
                           ego2global_rotation=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
                           ego_dynamic_state=np.array([2., 0., 0., 0.]),
                           anns=dict(gt_boxes=boxes, gt_names=names, track_tokens=ids,
                                     gt_velocity_3d=velocity)))
    return result


class ContextTests(unittest.TestCase):
    def test_write_load_roundtrip_and_immutable_current_view(self):
        with tempfile.TemporaryDirectory() as folder:
            source = synthetic_context()
            record = save_context(folder, source)
            write_manifest(folder, [record], {"fixture": True})
            store = ContextStore(folder)
            got = store.load(source["token"])
            self.assertEqual(got["actors"].device.type, "cpu")
            self.assertTrue(torch.equal(got["future"], source["future"]))
            self.assertEqual(store.records[source["token"]]["log"], "log_a")
            view = current_only(got)
            self.assertEqual(set(view), set(CURRENT_FIELDS))
            for key in ("future", "future_mask", "gt_poses", "label_ok"):
                self.assertNotIn(key, view)
            view["actors"][0, 0] += 9
            self.assertTrue(torch.equal(got["actors"], source["actors"]))
            with self.assertRaises(FileExistsError):
                save_context(folder, source)
            with self.assertRaises(FileExistsError):
                write_manifest(folder, [record], {"fixture": True})

    def test_empty_actors_are_valid_and_not_fabricated(self):
        context = validate_context(synthetic_context(count=0))
        self.assertEqual(context["actors"].shape, (0, 10))
        self.assertEqual(context["future"].shape, (9, 0, 10))
        self.assertTrue(context["label_ok"])

    def test_partial_future_is_masked_not_a_safe_training_label(self):
        source = synthetic_context()
        source["future_mask"][3, 0] = False
        source["future"][3, 0] = float("nan")
        with self.assertRaisesRegex(ContextError, "label_ok"):
            validate_context(source)
        source["label_ok"] = False
        got = validate_context(source)
        self.assertTrue(torch.equal(got["actors"], source["actors"]))
        self.assertFalse(got["future_mask"][3, 0])
        self.assertEqual(got["future"][3, 0].abs().sum(), 0)
        self.assertFalse(got["label_ok"])

    def test_scoring_view_is_independent_of_future_labels(self):
        original = synthetic_context()
        changed = copy.deepcopy(original)
        changed["future"][1:, :, :2] += 1000
        changed["gt_poses"] += 500
        changed["label_ok"] = False
        before, after = [current_only(validate_context(x)) for x in (original, changed)]
        for key in before:
            if torch.is_tensor(before[key]):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            else:
                self.assertEqual(before[key], after[key], key)

    def test_invalid_schema_geometry_and_horizon_fail(self):
        changes = [dict(schema_version=2), dict(horizon=6), dict(dt=1.),
                   dict(split="navtest"), dict(resolution=1.), dict(origin=[-30., -32.]),
                   dict(v0=float("nan")), dict(v0=-1.), dict(label_ok=1),
                   dict(mask=torch.ones(2)), dict(sdf=torch.zeros(1, 5, 5)),
                   dict(sdf=torch.full((1, 5, 5), float("nan"))),
                   dict(gt_poses=torch.zeros(6, 3))]
        for change in changes:
            with self.subTest(change=list(change)):
                source = synthetic_context()
                source.update(change)
                with self.assertRaises(ContextError):
                    validate_context(source)
        source = synthetic_context()
        source["actors"][0, 0] = float("nan")
        with self.assertRaisesRegex(ContextError, "nonfinite"):
            validate_context(source)
        source = synthetic_context()
        source["future"][0, 0, 0] += 1
        with self.assertRaisesRegex(ContextError, "must match"):
            validate_context(source)

    def test_safe_paths_missing_context_and_tampering_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            source = synthetic_context()
            record = save_context(folder, source)
            write_manifest(folder, [record], {"fixture": True})
            store = ContextStore(folder)
            for token in ("../scene", "a/b", "a\\b", "C:token", "..", "bad\n", "unknown"):
                with self.subTest(token=token), self.assertRaises(ContextError):
                    store.load(token)
            path = Path(folder) / record["path"]
            source["token"] = "different_token"
            torch.save(source, path)
            with self.assertRaisesRegex(ContextError, "sha256 mismatch"):
                store.load(record["token"])
            # Even a matching checksum does not permit a wrong token inside .pt.
            store.records[record["token"]]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ContextError, "token mismatch"):
                store.load(record["token"])

    def test_manifest_overlap_duplicate_and_escape_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            records = [save_context(folder, synthetic_context()),
                       save_context(folder, synthetic_context(token="other", split="val"))]
            with self.assertRaisesRegex(ContextError, "log overlap"):
                write_manifest(folder, records, {"fixture": True})
            with self.assertRaisesRegex(ContextError, "duplicate"):
                write_manifest(folder, [records[0], records[0]], {"fixture": True})
            records[0]["path"] = "../scene_token.pt"
            with self.assertRaisesRegex(ContextError, "relative path"):
                write_manifest(folder, [records[0]], {"fixture": True})


class PreparationTests(unittest.TestCase):
    request = dict(token="frame_0", log="raw_log", split="train")
    sdf = np.ones((1, 193, 193), dtype=np.float32)

    def test_exact_token_alignment_and_current_actor_identity(self):
        context = context_from_window(self.request, raw_window(), self.sdf)
        self.assertEqual(context["actors"].shape, (2, 10))
        self.assertTrue(context["label_ok"])
        self.assertTrue(torch.allclose(context["gt_poses"][:, 0], torch.arange(1., 9.), atol=1e-5))
        self.assertTrue(torch.allclose(context["gt_poses"][:, 2], .03*torch.arange(1., 9.), atol=1e-5))
        self.assertTrue(torch.allclose(context["future"][:, 0, 0], torch.full((9,), 10.), atol=1e-5))
        self.assertTrue(torch.allclose(context["future"][:, 1, :2], torch.tensor([14., -2.]).repeat(9, 1), atol=1e-5))
        self.assertEqual(context["v0"], 2.)

    def test_empty_raw_actors(self):
        context = context_from_window(self.request, raw_window(empty=True), self.sdf)
        self.assertEqual(len(context["actors"]), 0)
        self.assertTrue(context["label_ok"])

    def test_future_missing_track_does_not_change_current(self):
        window = raw_window()
        original = context_from_window(self.request, window, self.sdf)
        window[4]["anns"]["track_tokens"][0] = "unseen_track"
        partial = context_from_window(self.request, window, self.sdf)
        self.assertFalse(partial["label_ok"])
        self.assertFalse(partial["future_mask"][4, 0])
        self.assertTrue(torch.equal(partial["actors"], original["actors"]))
        self.assertTrue(torch.equal(partial["v0"], original["v0"]))

    def test_truncation_is_explicit_and_uses_only_current_selection(self):
        context = context_from_window(self.request, raw_window(), self.sdf, actor_cap=1)
        self.assertEqual(context["actors"].shape, (1, 10))
        self.assertTrue(context["truncated"])

    def test_invalid_raw_metadata_time_map_and_current_state_fail(self):
        with self.assertRaisesRegex(ContextError, "empty drivable"):
            context_from_window(self.request, raw_window(), None)
        with self.assertRaisesRegex(ContextError, "eight complete"):
            context_from_window(self.request, raw_window()[:8], self.sdf)
        wrong_token = dict(self.request, token="other")
        with self.assertRaisesRegex(ContextError, "does not match"):
            context_from_window(wrong_token, raw_window(), self.sdf)
        window = raw_window()
        window[5]["timestamp"] += 100000
        with self.assertRaisesRegex(ContextError, "timestamps"):
            context_from_window(self.request, window, self.sdf)
        window = raw_window()
        window[0]["anns"]["gt_velocity_3d"][0] = np.nan
        with self.assertRaisesRegex(ContextError, "current actor"):
            context_from_window(self.request, window, self.sdf)
        window = raw_window()
        window[0]["anns"]["gt_names"][0] = "unknown_class"
        with self.assertRaisesRegex(ContextError, "unknown actor classes"):
            context_from_window(self.request, window, self.sdf)

    def test_trainval_requests_only_and_log_disjoint(self):
        self.assertEqual(validate_requests([self.request]), [self.request])
        cases = [[], [dict(self.request, split="navtest")], [self.request, self.request],
                 [self.request, dict(self.request, token="another", split="val")],
                 [dict(self.request, log="../escape")], [dict(self.request, log="file.pkl")]]
        for requests in cases:
            with self.subTest(requests=requests), self.assertRaises(ContextError):
                validate_requests(requests)


if __name__ == "__main__":
    unittest.main()
