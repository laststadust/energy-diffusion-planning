"""Rectangle geometry, risk supervision and graph masking contracts."""
import math
import unittest

import torch

from ddv2_energy.interaction import (
    InteractionConfig, InteractionEnergy, _relations, label_cost,
    rect_corners, rect_signed_distance,
)


def rectangle(x, y, length, width, yaw=0.):
    return rect_corners(torch.tensor([x, y], dtype=torch.float64),
                        torch.tensor(yaw, dtype=torch.float64),
                        torch.tensor(length), torch.tensor(width))


def inputs():
    xy = torch.tensor([[[0., 0.], [2., .1], [4., .25]]], requires_grad=True)
    trajectory = dict(xy=xy, yaw=torch.zeros(1, 3), speed=torch.full((1, 3), 4.))
    actor = torch.tensor([[[10., 0., 0., 4., 2., 0., 0., 1.]]])
    return trajectory, actor


class RectangleTests(unittest.TestCase):
    def test_distance_contact_and_containment(self):
        a = rectangle(0., 0., 4., 2.)
        self.assertAlmostEqual(rect_signed_distance(a, rectangle(5., 4., 2., 2.)).item(), 8. ** .5)
        self.assertEqual(rect_signed_distance(a, rectangle(3., 0., 2., 1.)).item(), 0.)
        self.assertEqual(rect_signed_distance(a, rectangle(0., 0., 1., 1.)).item(), -1.5)
        self.assertEqual(rect_signed_distance(a, a).item(), -2.)

    def test_rotation_invariance_and_derivative(self):
        a = rectangle(0., 0., 4., 2.)
        b = rectangle(4.1, 2.2, 1.4, 1.1, math.pi / 8)
        yaw = torch.tensor(.37, dtype=torch.float64)
        rotation = torch.tensor([[yaw.cos(), -yaw.sin()], [yaw.sin(), yaw.cos()]])
        torch.testing.assert_close(rect_signed_distance(a, b),
                                   rect_signed_distance(a @ rotation, b @ rotation))
        delta = torch.tensor([.41, -.23], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: rect_signed_distance(a + t, b), (delta,)))


class InteractionTests(unittest.TestCase):
    def test_directional_relations_and_future_labels(self):
        trajectory, actor = inputs()
        future = actor[:, None].expand(-1, 3, -1, -1).clone()
        mask = torch.ones(1, 3, 1, dtype=torch.bool)
        relations = _relations(trajectory, future, mask, InteractionConfig())[-1]
        self.assertEqual(relations[0, 0, 0, :2].tolist(), [1., 0.])
        crossing = future.clone()
        crossing[..., 2] = math.pi / 2
        self.assertFalse(_relations(trajectory, crossing, mask, InteractionConfig())[-1][..., :2].any())
        close = future.clone()
        close[..., 0] = 4.
        self.assertGreater(label_cost(trajectory, close, mask).item(), label_cost(trajectory, future, mask).item())
        self.assertEqual(label_cost(trajectory, close, ~mask).item(), 0.)

    def test_empty_scene(self):
        trajectory, actor = inputs()
        model = InteractionEnergy()
        mask = torch.zeros(1, 0, dtype=torch.bool)
        self.assertEqual(model(trajectory, actor[:, :0], mask).item(), 0.)
        self.assertEqual(label_cost(trajectory, actor[:, None, :0].expand(-1, 3, -1, -1),
                                    mask[:, None].expand(-1, 3, -1)).item(), 0.)

    def test_padding_and_finite_gradients(self):
        torch.manual_seed(7)
        model = InteractionEnergy()
        trajectory, actor = inputs()
        valid = torch.ones(1, 1, dtype=torch.bool)
        energy = model(trajectory, actor, valid)
        self.assertGreater(energy.item(), 0.)
        energy.sum().backward()
        self.assertTrue(torch.isfinite(trajectory["xy"].grad).all())
        self.assertGreater(trajectory["xy"].grad.abs().sum().item(), 0.)
        padding = torch.full_like(actor, float("nan"))
        with torch.no_grad():
            torch.testing.assert_close(energy, model(trajectory, torch.cat((actor, padding), 1),
                                                     torch.cat((valid, ~valid), 1)))
            self.assertEqual(model(trajectory, padding, ~valid).item(), 0.)


if __name__ == "__main__":
    unittest.main()
