"""Reward-engine isolation tests using explicitly synthetic graph weights."""
import copy
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from ddv2_energy.interaction import InteractionConfig, InteractionEnergy
from ddv2_energy.config import EnergyRLConfig
from ddv2_energy.contexts import save_context, write_manifest
from ddv2_energy.engine import FrozenEnergyEngine, apply_energy_reward


def context(token, log, split="train", empty=False):
    actors = torch.tensor([[12., 0., 0., 4., 2., 0., 0., 1., 1., 1.]])
    if empty:
        actors = actors[:0]
    count = len(actors)
    return dict(schema_version=1, token=token, log=log, split=split, dt=.5, horizon=8,
                actors=actors, mask=torch.ones(count, dtype=torch.bool),
                sdf=torch.ones(1, 193, 193), origin=[-32., -32.], resolution=.5, v0=4.,
                future=actors[None].repeat(9, 1, 1),
                future_mask=torch.ones(9, count, dtype=torch.bool),
                label_ok=True, truncated=False, gt_poses=torch.zeros(8, 3))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        contexts = [context("train_scene", "train_log"),
                    context("empty_scene", "empty_log", empty=True),
                    context("val_scene", "val_log", split="val")]
        records = [save_context(self.root, c) for c in contexts]
        write_manifest(self.root, records, {"synthetic": True})
        graph_config = InteractionConfig(hidden=8, layers=1, heads=2)
        self.checkpoint = dict(format="ddv2_dual_graph_v1", config=asdict(graph_config),
                               state_dict=InteractionEnergy(graph_config).state_dict(),
                               trained_steps=1,
                               provenance=dict(source="DiffusionDriveV2", synthetic=True,
                                               checkpoint_sha256="a" * 64,
                                               train_logs=["train_log"], val_logs=["val_log"]))
        self.weights = self.root / "synthetic_graph.pt"
        torch.save(self.checkpoint, self.weights)
        self.cfg = EnergyRLConfig(mode="dual_graph", context_root=str(self.root),
                                  graph_checkpoint=str(self.weights), chunk_size=2)
        self.poses = torch.zeros(1, 3, 8, 3)
        self.poses[..., 0] = torch.arange(1., 9.) * 2
        self.poses[:, 1, :, 1] = 4.
        self.poses[:, 2, :, 1] = -4.

    def tearDown(self):
        self.temp.cleanup()

    def engine(self, cfg=None):
        return FrozenEnergyEngine(cfg or self.cfg, "cpu", allow_test_checkpoint=True)

    def test_synthetic_checkpoint_is_explicitly_test_only(self):
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            FrozenEnergyEngine(self.cfg, "cpu")
        self.assertIsNotNone(self.engine())

    def test_original_and_zero_strength_never_load_engine(self):
        for config in (SimpleNamespace(),
                       SimpleNamespace(energy_mode="dual_graph", energy_strength=0.)):
            head = SimpleNamespace(_config=config)
            original = torch.tensor([[[0., -1.], [1., 0.]]])
            with mock.patch("ddv2_energy.engine.FrozenEnergyEngine", side_effect=AssertionError("loaded")):
                result, metrics = apply_energy_reward(head, None, None, None, original, None)
            self.assertIs(result, original)
            self.assertEqual(metrics, {})
            self.assertFalse(hasattr(head, "_energy_reward_engine"))
        with self.assertRaisesRegex(ValueError, "Do not instantiate"):
            FrozenEnergyEngine(EnergyRLConfig(), "cpu")

    def test_frozen_graph_is_not_registered_and_init_preserves_rng(self):
        state = torch.random.get_rng_state().clone()
        engine = self.engine()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(engine.graph.training)
        self.assertTrue(all(not p.requires_grad for p in engine.graph.parameters()))
        head = torch.nn.Linear(2, 2)
        keys = list(head.state_dict())
        head._energy_reward_engine = engine
        self.assertEqual(keys, list(head.state_dict()))
        head.train()
        self.assertFalse(engine.graph.training)
        self.assertEqual(len(list(head.parameters())), 2)

    def test_finite_detached_scores_preserve_candidates_and_empty_is_zero(self):
        engine = self.engine()
        poses = self.poses.clone().requires_grad_(True)
        saved = poses.detach().clone()
        road, interaction = engine.score(poses, ["train_scene"])
        self.assertTrue(torch.equal(poses, saved))
        for cost in (road, interaction):
            self.assertEqual(cost.shape, (1, 3))
            self.assertTrue(torch.isfinite(cost).all())
            self.assertTrue((cost >= 0).all())
            self.assertFalse(cost.requires_grad)
        self.assertGreater(interaction.max().item(), 0.)
        _, empty = engine.score(poses, ["empty_scene"])
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
        self.assertTrue(all(p.grad is None for p in engine.graph.parameters()))

    def test_future_is_not_scoring_input_and_chunk_size_is_invariant(self):
        engine = self.engine()
        first = engine.score(self.poses, ["train_scene"])
        changed = engine.store.load("train_scene")
        changed["future"][1:, :, :2] += 1000.
        changed["gt_poses"] += 500.
        changed["label_ok"] = False
        with mock.patch.object(engine.store, "load", return_value=changed):
            second = engine.score(self.poses, ["train_scene"])
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        unchunked = self.engine(replace(self.cfg, chunk_size=3)).score(self.poses, ["train_scene"])
        for a, b in zip(first, unchunked):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))

    def test_refuses_validation_bad_shapes_nan_and_token_count(self):
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, "training contexts"):
            engine.score(self.poses, ["val_scene"])
        bad = self.poses.clone()
        bad[0, 0, 0, 0] = float("nan")
        for poses, tokens in ((bad, ["train_scene"]), (self.poses[:, :, :6], ["train_scene"]),
                              (self.poses, []), (self.poses, "train_scene")):
            with self.subTest(shape=tuple(poses.shape), tokens=tokens), self.assertRaises(ValueError):
                engine.score(poses, tokens)

    def test_rejects_untrained_wrong_provenance_overlapping_logs_and_nan_weights(self):
        variants = []
        for key, value in (("format", "old_graph"), ("trained_steps", 0)):
            variant = copy.deepcopy(self.checkpoint)
            variant[key] = value
            variants.append(variant)
        for key, value in (("source", "DiffusionDriveV1"), ("checkpoint_sha256", "bad"),
                           ("val_logs", ["train_log"])):
            variant = copy.deepcopy(self.checkpoint)
            variant["provenance"][key] = value
            variants.append(variant)
        variant = copy.deepcopy(self.checkpoint)
        next(iter(variant["state_dict"].values())).fill_(float("nan"))
        variants.append(variant)
        for i, checkpoint in enumerate(variants):
            with self.subTest(variant=i):
                torch.save(checkpoint, self.weights)
                with self.assertRaises(ValueError):
                    self.engine()


if __name__ == "__main__":
    unittest.main()
