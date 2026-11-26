"""Synthetic geometry and trajectory contracts; no dataset or GPU is required."""
import unittest

import torch

from ddv2_energy.geometry import road_energy
from ddv2_energy.trajectory import pose_traj


class RoadEnergyTests(unittest.TestCase):
    def test_interior_boundary_and_unknown_region(self):
        trajectory = {
            "xy": torch.tensor([[[0., 3.]], [[0., 0.]], [[40., 3.]]], requires_grad=True),
            "yaw": torch.zeros(3, 1),
        }
        # Signed distance to the half-plane y >= 0.
        sdf = torch.arange(-10, 11.).view(1, 1, 21, 1).expand(3, 1, 21, 21)
        energy = road_energy(trajectory, sdf, (-10., -10.), 1.)
        self.assertEqual(energy[0].item(), 0.)
        self.assertGreater(energy[1].item(), 0.)
        self.assertGreater(energy[2].item(), 0.)
        gradient = torch.autograd.grad(energy.sum(), trajectory["xy"])[0]
        self.assertLess(gradient[1, 0, 1].item(), 0.)
        self.assertGreater(gradient[2, 0, 0].item(), 0.)

    def test_single_map_broadcast(self):
        trajectory = {"xy": torch.tensor([[[0., 0.]], [[1., 0.]]]),
                      "yaw": torch.zeros(2, 1)}
        sdf = torch.full((1, 1, 41, 41), 4.)
        shared = road_energy(trajectory, sdf, (-10., -10.), .5)
        batched = road_energy(trajectory, sdf.expand(2, -1, -1, -1), (-10., -10.), .5)
        torch.testing.assert_close(shared, batched, rtol=0., atol=0.)

    def test_invalid_grid(self):
        trajectory = {"xy": torch.zeros(1, 1, 2), "yaw": torch.zeros(1, 1)}
        with self.assertRaises(ValueError):
            road_energy(trajectory, torch.ones(1, 1, 5, 5), (0., 0.), 0.)


class TrajectoryTests(unittest.TestCase):
    def test_initial_state_and_interval_speed(self):
        poses = torch.tensor([[[3., 4., .2], [3., 5., .4]]], requires_grad=True)
        result = pose_traj(poses, 2., dt=.5)
        torch.testing.assert_close(result["xy"][0, 0], torch.zeros(2))
        torch.testing.assert_close(result["yaw"], torch.tensor([[0., .2, .4]]))
        torch.testing.assert_close(result["speed"], torch.tensor([[2., 10., 2.]]))
        result["speed"].sum().backward()
        self.assertTrue(torch.isfinite(poses.grad).all())

    def test_zero_displacement(self):
        result = pose_traj(torch.zeros(2, 8, 3), .03)
        torch.testing.assert_close(result["speed"][:, 0], torch.full((2,), .03))
        self.assertEqual(result["speed"][:, 1:].sum().item(), 0.)

    def test_invalid_input(self):
        poses = torch.zeros(1, 8, 3)
        for speed in (-1., float("nan"), [1., 2.]):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                pose_traj(poses, speed)
        for dt in (0., -1., float("nan")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                pose_traj(poses, 0., dt)
        with self.assertRaises(ValueError):
            pose_traj(torch.zeros(1, 8, 2), 0.)


if __name__ == "__main__":
    unittest.main()
