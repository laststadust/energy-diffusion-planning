"""CPU-only checks for reproducible, bounded and resumable paired RL runs."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from ddv2_energy.run import (EpochSeededSampler, RESUME_FIELDS, atomic_json, audit_weights,
                  capture_rng, finite_scalar_metrics, restore_rng, state_fingerprints,
                  validate_gate_args, verify_resume)


class SamplerTests(unittest.TestCase):
    def test_same_seed_epoch_does_not_consume_global_rng(self):
        first, second = EpochSeededSampler(list(range(20)), 7), EpochSeededSampler(list(range(20)), 7)
        first.set_epoch(4); second.set_epoch(4)
        state = torch.get_rng_state().clone()
        expected = list(first)
        torch.randn(200)
        self.assertEqual(expected, list(second))
        torch.set_rng_state(state)
        list(first)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sorted(expected), list(range(20)))

    def test_resumed_epoch_matches_uninterrupted_without_replaying_loader(self):
        sampler = EpochSeededSampler(range(30), 0)
        sampler.set_epoch(0)
        zero = list(sampler)
        sampler.set_epoch(9)
        nine = list(sampler)
        resumed = EpochSeededSampler(range(30), 0)
        resumed.set_epoch(9)
        self.assertEqual(nine, list(resumed))
        self.assertNotEqual(zero, nine)
        self.assertEqual(len(resumed), 30)


class GateTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(epochs=10, batch_size=1, workers=0, lr=1e-5, gate=False,
                      limit_train_batches=None, limit_val_batches=None, max_steps=-1)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_formal_and_bounded_gate_valid(self):
        validate_gate_args(self.args())
        validate_gate_args(self.args(gate=True, epochs=1, limit_train_batches=2,
                                     limit_val_batches=2, max_steps=2))

    def test_formal_cannot_silently_limit_budget(self):
        for change in (dict(limit_train_batches=2), dict(limit_val_batches=2), dict(max_steps=2)):
            with self.assertRaises(ValueError):
                validate_gate_args(self.args(**change))

    def test_gate_cannot_be_long_training(self):
        base = dict(gate=True, epochs=1, limit_train_batches=2, limit_val_batches=2, max_steps=2)
        for change in (dict(epochs=2), dict(limit_train_batches=3), dict(limit_val_batches=None),
                       dict(max_steps=-1), dict(max_steps=3), dict(workers=-1), dict(lr=float("nan"))):
            with self.assertRaises(ValueError):
                validate_gate_args(self.args(**dict(base, **change)))


class AuditTests(unittest.TestCase):
    def test_exact_fingerprint_and_update_audit(self):
        model = torch.nn.Linear(3, 2)
        first = state_fingerprints(model)
        second = state_fingerprints(model)
        self.assertEqual(first, second)
        prefix = "_transfuser_model._trajectory_head."
        before = {"backbone.weight": "a", prefix + "plan_anchor": "b", prefix + "diff_decoder.weight": "c"}
        after = dict(before, **{prefix + "diff_decoder.weight": "d"})
        result = audit_weights(before, after, require_change=True)
        self.assertEqual(result["changed_denoiser_tensors"], 1)
        self.assertTrue(result["frozen_tensors_unchanged"])
        self.assertTrue(result["anchor_unchanged"])

    def test_frozen_anchor_noop_and_missing_keys_fail(self):
        prefix = "_transfuser_model._trajectory_head."
        before = {"backbone.weight": "a", prefix + "plan_anchor": "b", prefix + "diff_decoder.weight": "c"}
        with self.assertRaises(RuntimeError):
            audit_weights(before, before, require_change=True)
        for key in ("backbone.weight", prefix + "plan_anchor"):
            after = dict(before, **{key: "changed"})
            with self.assertRaises(RuntimeError):
                audit_weights(before, after)
        with self.assertRaises(RuntimeError):
            audit_weights(before, {})

    def test_metrics_are_finite_scalars(self):
        metrics = finite_scalar_metrics({"loss": torch.tensor(.25), "step": 2, "vector": torch.ones(2)})
        self.assertEqual(metrics, {"loss": .25, "step": 2})
        for value in (float("nan"), float("inf"), torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                finite_scalar_metrics({"bad": value})

    def test_rng_roundtrip_without_gpu_access(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            random.seed(7); np.random.seed(7); torch.manual_seed(7)
            state = capture_rng()
            expected = (random.random(), float(np.random.rand()), torch.rand(4))
            restore_rng(state)
            actual = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "checkpoints/last.ckpt"
        self.path.parent.mkdir()
        self.metadata = {key: None for key in RESUME_FIELDS}
        self.metadata.update(source_hashes={"run.py": "source_hash"}, gate=False, seed=0, epochs=10)
        self.control = dict(gate=False, epoch_boundary=True, rng={"torch": torch.get_rng_state()},
                            source_hashes=self.metadata["source_hashes"], last_completed_epoch=2)
        atomic_json(self.root / "run.json", self.metadata)
        torch.save({"ddv2_run_control": self.control}, self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_resume_and_atomic_metadata(self):
        control = verify_resume(self.path, self.metadata)
        self.assertEqual(control["last_completed_epoch"], 2)
        self.assertFalse((self.root / "run.json.tmp").exists())
        self.assertEqual(json.loads((self.root / "run.json").read_text()), self.metadata)

    def test_resume_changed_settings_source_or_data_fail(self):
        for field in ("seed", "epochs", "manifest_sha256", "graph_sha256", "strength", "source_hashes"):
            modified = copy.deepcopy(self.metadata)
            modified[field] = "changed"
            with self.assertRaises(ValueError):
                verify_resume(self.path, modified)

    def test_gate_partial_epoch_missing_rng_or_wrong_source_fail(self):
        for field, value in (("gate", True), ("epoch_boundary", False), ("source_hashes", {})):
            control = dict(self.control, **{field: value})
            torch.save({"ddv2_run_control": control}, self.path)
            with self.assertRaises(ValueError):
                verify_resume(self.path, self.metadata)
        control = dict(self.control)
        del control["rng"]
        torch.save({"ddv2_run_control": control}, self.path)
        with self.assertRaises(ValueError):
            verify_resume(self.path, self.metadata)

    def test_missing_parent_run_metadata_fail(self):
        (self.root / "run.json").unlink()
        with self.assertRaises(ValueError):
            verify_resume(self.path, self.metadata)


if __name__ == "__main__":
    unittest.main()
