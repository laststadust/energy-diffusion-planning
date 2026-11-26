"""纯张量奖励回归测试，不依赖 NAVSIM 数据、训练权重或 CUDA。"""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from ddv2_energy.config import EnergyRLConfig
from ddv2_energy.rewards import energy_advantages


class ConfigTests(unittest.TestCase):
    def test_defaults_and_agent_fields(self):
        self.assertFalse(EnergyRLConfig().enabled)
        cfg = EnergyRLConfig.from_agent_config(SimpleNamespace(
            energy_mode="dual_graph", energy_strength=0.2, energy_context_root="contexts",
            energy_graph_checkpoint="graph.pt", energy_chunk_size=8,
        ))
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.strength, 0.2)
        self.assertEqual(cfg.context_root, "contexts")
        self.assertEqual(cfg.graph_checkpoint, "graph.pt")
        self.assertEqual(cfg.chunk_size, 8)
        self.assertEqual(EnergyRLConfig.from_agent_config({"energy_strength": 0}).strength, 0)

    def test_invalid_settings(self):
        invalid = [
            {"mode": "typo"}, {"strength": -1}, {"strength": float("nan")},
            {"road_weight": -0.1, "interaction_weight": 1.1}, {"road_weight": 0.6},
            {"interaction_weight": float("inf")}, {"road_scale": 0},
            {"interaction_scale": -1}, {"std_floor": 0}, {"adv_clip": 0},
            {"motion_weight": 0.01}, {"chunk_size": 0}, {"chunk_size": 2.5},
            {"chunk_size": True}, {"strength": True}, {"strength": 1e100},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                EnergyRLConfig(**kwargs)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EnergyRLConfig(mode="dual_graph", strength=0.1)
        self.pdm = torch.full((2, 4, 3), 0.8)
        self.original = torch.zeros_like(self.pdm)
        self.road = torch.arange(4.0).view(1, 4, 1).expand_as(self.pdm).clone()
        self.interaction = self.road * 0.5
        self.unsafe = torch.zeros_like(self.pdm, dtype=torch.bool)

    def run_reward(self, **kwargs):
        values = dict(pdm=self.pdm, original=self.original, road=self.road,
                      interaction=self.interaction, unsafe=self.unsafe, cfg=self.cfg)
        values.update(kwargs)
        return energy_advantages(**values)

    def test_each_energy_affects_equal_pdm(self):
        for road_weight in (0.0, 1.0):
            with self.subTest(road_weight=road_weight):
                cfg = replace(self.cfg, road_weight=road_weight,
                              interaction_weight=1.0-road_weight)
                result, metrics = self.run_reward(cfg=cfg)
                self.assertTrue((result[:, 0] > result[:, 1]).all())
                self.assertTrue((result[:, -1] == 0).all())
                self.assertGreater(metrics["energy_bounded_cost_mean"].item(), 0)
                self.assertLess(metrics["energy_shaped_reward_mean"].item(), 0.8)

    def test_formula_and_per_anchor_standardization(self):
        pdm = self.pdm + torch.tensor([0.0, 0.02, 0.04]).view(1, 1, 3)
        result, _ = self.run_reward(pdm=pdm)
        cost = 0.5*self.road/(1+self.road) + 0.5*self.interaction/(1+self.interaction)
        shaped = pdm - 0.1*cost
        expected = ((shaped-shaped.mean(1, keepdim=True)) /
                    shaped.std(1, unbiased=False, keepdim=True).clamp_min(0.05)).clamp(0, 5)
        torch.testing.assert_close(result, expected)

    def test_unsafe_never_positive_and_retains_severity(self):
        result, metrics = self.run_reward(unsafe=torch.ones_like(self.unsafe))
        self.assertTrue((result <= -1).all())
        self.assertTrue((result[:, 0] > result[:, -1]).all())
        self.assertEqual(metrics["energy_adv_negative_fraction"].item(), 1.0)

    def test_mixed_safe_and_unsafe(self):
        unsafe = self.unsafe.clone()
        unsafe[:, 0, 0] = True
        result, _ = self.run_reward(unsafe=unsafe)
        self.assertTrue((result[:, 0, 0] < 0).all())
        self.assertTrue((result[:, 0, 1] > 0).all())

    def test_zero_variance_is_finite_zero(self):
        result, metrics = self.run_reward(road=torch.zeros_like(self.road),
                                         interaction=torch.zeros_like(self.interaction))
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        self.assertEqual(metrics["energy_group_std_mean"].item(), 0)

    def test_group_and_anchor_permutations(self):
        result, _ = self.run_reward()
        order_g = torch.tensor([2, 0, 3, 1])
        order_n = torch.tensor([2, 0, 1])
        permute = lambda x: x[:, order_g][:, :, order_n]
        actual, _ = self.run_reward(pdm=permute(self.pdm), original=permute(self.original),
                                   road=permute(self.road), interaction=permute(self.interaction),
                                   unsafe=permute(self.unsafe))
        torch.testing.assert_close(actual, permute(result))
        changed = self.road.clone()
        changed[:, :, 1] *= 10
        actual, _ = self.run_reward(road=changed)
        torch.testing.assert_close(actual[:, :, 0], result[:, :, 0], rtol=0, atol=0)

    def test_no_gradients_no_input_mutation(self):
        inputs = [self.pdm, self.original, self.road, self.interaction]
        for value in inputs:
            value.requires_grad_(True)
        copies = [value.detach().clone() for value in inputs]
        result, metrics = self.run_reward()
        self.assertFalse(result.requires_grad)
        for value in metrics.values():
            self.assertEqual(value.ndim, 0)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))
        for value, snapshot in zip(inputs, copies):
            self.assertIsNone(value.grad)
            torch.testing.assert_close(value.detach(), snapshot, rtol=0, atol=0)

    def test_disabled_exact_original_dtype_and_values(self):
        original = torch.randn_like(self.original, dtype=torch.float64, requires_grad=True)
        for cfg in (EnergyRLConfig(), replace(self.cfg, strength=0)):
            with self.subTest(cfg=cfg):
                result, _ = self.run_reward(original=original, road=None, interaction=None,
                                           unsafe=None, cfg=cfg)
                self.assertEqual(result.dtype, original.dtype)
                self.assertFalse(result.requires_grad)
                self.assertTrue(torch.equal(result, original.detach()))
                self.assertNotEqual(result.data_ptr(), original.data_ptr())

    def test_float32_active_output_and_clip(self):
        result, _ = self.run_reward(pdm=self.pdm.double(), road=self.road.double(),
                                   cfg=replace(self.cfg, adv_clip=0.02))
        self.assertEqual(result.dtype, torch.float32)
        self.assertLessEqual(result.max().item(), 0.020000001)

    def test_invalid_group_count(self):
        with self.assertRaisesRegex(ValueError, "G >= 2"):
            self.run_reward(pdm=self.pdm[:, :1], original=self.original[:, :1])

    def test_disabled_single_group_still_preserves_original(self):
        original = torch.ones_like(self.original[:, :1]) * 0.37
        result, metrics = self.run_reward(
            pdm=self.pdm[:, :1], original=original, cfg=EnergyRLConfig(),
            road=None, interaction=None, unsafe=None,
        )
        self.assertTrue(torch.equal(result, original))
        self.assertEqual(metrics["energy_group_std_mean"].item(), 0)

    def test_invalid_values_fail_loudly(self):
        for field, bad_value in (("pdm", float("nan")), ("original", float("inf")),
                                 ("road", float("nan")), ("road", -1),
                                 ("interaction", float("inf")), ("interaction", -0.1)):
            value = getattr(self, field).clone()
            value[0, 0, 0] = bad_value
            with self.subTest(field=field, bad_value=bad_value), self.assertRaises(ValueError):
                self.run_reward(**{field: value})

    def test_shapes_and_mask_dtype(self):
        with self.assertRaises(ValueError):
            self.run_reward(road=self.road[:, :, :1])
        with self.assertRaises(TypeError):
            self.run_reward(unsafe=self.unsafe.float())
        with self.assertRaises(ValueError):
            self.run_reward(unsafe=self.unsafe[:, :, :1])
        with self.assertRaises(ValueError):
            self.run_reward(pdm=self.pdm[0])


if __name__ == "__main__":
    unittest.main()
