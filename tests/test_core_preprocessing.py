"""Synthetic NAVSIM records for coordinate, temporal and supervision checks."""
import copy
import unittest

import numpy as np

from ddv2_energy.preprocessing import rot, scene_labels, wrap


def scene_window():
    frames = []
    location = np.array([100., -24.])
    initial_yaw = 3.08
    for h in range(9):
        yaw = initial_yaw + .03 * h
        ego = location + rot(initial_yaw) @ np.array([h, .03 * h * h])
        vehicle = location + rot(initial_yaw) @ np.array([10., 3. + h * .5])
        cone = location + rot(initial_yaw) @ np.array([14., -2.])
        heading = initial_yaw + .5
        velocity = rot(initial_yaw) @ [0., 1.]
        boxes = [[*(rot(yaw).T @ (point - ego)), 0., 4.5, 1.8, 1.5, wrap(heading - yaw)]
                 for point in (vehicle, cone)]
        # Future array order changes; association must use the track ID.
        order = [1, 0] if h % 2 else [0, 1]
        frames.append(dict(
            timestamp=int(1e15 + h * 500000), ego2global_translation=[*ego, 0.],
            ego2global_rotation=[np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)],
            ego_dynamic_state=[2., 0., 0., 0.],
            anns=dict(gt_boxes=np.array(boxes)[order],
                      track_tokens=np.array(["vehicle", "cone"])[order],
                      gt_names=np.array(["vehicle", "traffic_cone"])[order],
                      gt_velocity_3d=np.array([[*(rot(yaw).T @ velocity), 0.], [np.nan] * 3])[order]),
        ))
    return frames


class ContextTests(unittest.TestCase):
    def test_local_frame_and_track_association(self):
        labels = scene_labels(scene_window(), cap=3)
        h = np.arange(9)
        self.assertTrue(labels["label_ok"])
        np.testing.assert_allclose(labels["xy"], np.stack((h, .03 * h * h), 1), atol=1e-6)
        np.testing.assert_allclose(labels["yaw"], .03 * h, atol=1e-6)
        np.testing.assert_allclose(labels["future"][:, 0, :2],
                                   np.stack((np.full(9, 10.), 3. + h * .5), 1), atol=1e-5)
        np.testing.assert_allclose(labels["future"][:, 0, 5:7],
                                   np.broadcast_to([0., 1.], (9, 2)), atol=1e-6)
        np.testing.assert_allclose(labels["future"][:, 1, 5:7], 0.)
        self.assertEqual(labels["actors"][:, 8].tolist(), [1, 4, 0])
        self.assertEqual(labels["mask"].tolist(), [True, True, False])

    def test_future_labels_do_not_filter_current_inputs(self):
        stationary = scene_window()
        for frame in stationary:
            frame["ego_dynamic_state"][0] = .03
        changed = copy.deepcopy(stationary)
        changed[-1]["ego_dynamic_state"][0] = 2.
        changed[1]["anns"]["track_tokens"][:] = "vehicle"
        a, b = scene_labels(stationary), scene_labels(changed)
        self.assertAlmostEqual(a["v0"], .03)
        self.assertEqual(a["v0"], b["v0"])
        np.testing.assert_array_equal(a["actors"], b["actors"])
        np.testing.assert_array_equal(a["mask"], b["mask"])
        self.assertTrue(a["label_ok"])
        self.assertFalse(b["label_ok"])
        self.assertFalse(b["future_mask"][1].any())

    def test_time_alignment_and_truncation(self):
        frames = scene_window()
        self.assertTrue(scene_labels(frames, cap=1)["truncated"])
        frames[-1]["timestamp"] -= 40000
        self.assertIsNone(scene_labels(frames))
        self.assertIsNone(scene_labels(frames[:-1]))


if __name__ == "__main__":
    unittest.main()
