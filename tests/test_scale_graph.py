"""Tiny CPU recovery tests; synthetic weights must never become experiment init."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from ddv2_energy.interaction import InteractionConfig, InteractionEnergy
from ddv2_energy.graph_fit import evaluate as real_evaluate, make_batch
from ddv2_energy.scale_graph import accumulate_graph_gradients, fit_graph, read_resume, validate_scale_bank
from tests.test_graph_fit import fixture


class ScaleGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bank, self.contexts = fixture()
        self.path = self.root / "bank.pt"
        torch.save(self.bank, self.path)
        self.context_root = self.root / "contexts"
        self.context_root.mkdir()
        (self.context_root / "manifest.json").write_text('{"synthetic_test":true}', encoding="utf-8")
        self.store = mock.patch("ddv2_energy.graph_fit.ContextStore")
        self.mock_store = self.store.start()
        self.mock_store.return_value.load.side_effect = lambda token: self.contexts[token]
        self.config = InteractionConfig(hidden=8, layers=1, heads=2)
        torch.set_num_threads(1)

    def tearDown(self):
        self.store.stop()
        self.temp.cleanup()

    def run_fit(self, name, **overrides):
        # Effective two-candidate update consists of two one-candidate chunks;
        # recovery must match optimizer/RNG/history with accumulation enabled.
        args = dict(epochs=2, batch_size=2, microbatch_size=1, device="cpu", seed=7, config=self.config)
        args.update(overrides)
        return fit_graph(self.path, self.context_root, self.root / name, **args)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_cpu_epoch_resume_matches_uninterrupted_optimizer_rng_and_best(self):
        full = self.run_fit("full")
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption before second epoch commit")
            return real_evaluate(*args, **kwargs)

        with mock.patch("ddv2_energy.scale_graph.evaluate", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.run_fit("partial")
        self.assertFalse((self.root / "partial/report.json").exists())
        partial = torch.load(self.root / "partial/last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(partial["completed_epoch"], 1)
        before = (self.root / "partial/last.pt").read_bytes()
        resumed = self.run_fit("resumed", resume=self.root / "partial/last.pt")
        a = torch.load(self.root / "full/last.pt", map_location="cpu", weights_only=False)
        b = torch.load(self.root / "resumed/last.pt", map_location="cpu", weights_only=False)
        for key in ("state_dict", "optimizer", "history", "best_snapshot", "trained_steps"):
            self.assert_nested_equal(a[key], b[key])
        self.assertTrue(torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"]))
        self.assertEqual(a["rng"]["python"], b["rng"]["python"])
        self.assertEqual(full["history"], resumed["history"])
        self.assertEqual(before, (self.root / "partial/last.pt").read_bytes())
        self.assertFalse(resumed["future_labels_are_model_inputs"])
        self.assertEqual(resumed["completed_epochs"], 2)
        best = torch.load(self.root / "resumed/best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(best["metrics"], min(resumed["history"], key=lambda m: m["validation"]["huber"]))

    def test_resume_requires_same_settings_sources_data_and_new_output(self):
        self.run_fit("origin", epochs=1)
        checkpoint = self.root / "origin/last.pt"
        original = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for overrides in ({"epochs": 2}, {"batch_size": 3}, {"microbatch_size": 2}, {"seed": 8}, {"lr": .002}):
            with self.subTest(overrides=overrides):
                options = dict(epochs=1, resume=checkpoint)
                options.update(overrides)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.run_fit("bad", **options)
                self.assertFalse((self.root / "bad").exists())
        spec = copy.deepcopy(original["resume_spec"])
        spec["sources"]["ddv2_energy/interaction.py"] = "f" * 64
        with self.assertRaises(ValueError):
            read_resume(checkpoint, spec)
        (self.context_root / "manifest.json").write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_fit("changed-data", epochs=1, resume=checkpoint)
        with self.assertRaises(FileExistsError):
            self.run_fit("origin", epochs=1, resume=checkpoint)

    def test_corrupted_checkpoint_cannot_resume_and_completed_resume_is_safe(self):
        self.run_fit("finished", epochs=1)
        checkpoint = self.root / "finished/last.pt"
        report = self.run_fit("copied-complete", epochs=1, resume=checkpoint)
        self.assertEqual(report["completed_epochs"], 1)
        self.assertEqual(report["trained_steps"], 2)
        with checkpoint.open("ab") as stream:
            stream.write(b"not trusted")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_fit("bad", epochs=1, resume=checkpoint)

    def test_scale_bank_partial_marker_and_future_input_separation(self):
        raw = dict(self.bank, collection_protocol="scale", coverage_complete=False)
        torch.save(raw, self.path)
        with self.assertRaisesRegex(ValueError, "partial"):
            validate_scale_bank(self.path)
        torch.save(self.bank, self.path)
        from ddv2_energy.graph_fit import load_bank
        records, _ = load_bank(self.path, self.context_root)
        first = make_batch([(records[0], 0)], "cpu", self.config)
        changed = copy.deepcopy(records[0])
        changed.context["future"][1:, :, 0] = 1000.
        second = make_batch([(changed, 0)], "cpu", self.config)
        for key in first[0]:
            self.assertTrue(torch.equal(first[0][key], second[0][key]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertFalse(torch.equal(first[3], second[3]))

    def test_full_and_uneven_microbatch_loss_and_gradient_equivalence(self):
        from ddv2_energy.graph_fit import load_bank
        records, _ = load_bank(self.path, self.context_root)
        # Five candidates divided 2+2+1 checks weighting by actual sizes; a
        # naive equal mean over microbatch means would fail this comparison.
        samples = [(records[0], 0), (records[0], 1), (records[1], 2),
                   (records[1], 0), (records[0], 2)]
        torch.manual_seed(17)
        whole = InteractionEnergy(self.config).train()
        micro = copy.deepcopy(whole)
        whole.zero_grad(set_to_none=True)
        a = accumulate_graph_gradients(whole, samples, "cpu", self.config, len(samples))
        micro.zero_grad(set_to_none=True)
        b = accumulate_graph_gradients(micro, samples, "cpu", self.config, 2)
        self.assertAlmostEqual(a, b, places=5)
        for (name, left), (other_name, right) in zip(whole.named_parameters(), micro.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(left.grad)
            self.assertIsNotNone(right.grad)
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=3e-6,
                                       msg=lambda message: name + ": " + message)
        # The partial last effective batch remains one optimizer update.
        report = self.run_fit("partial-group", epochs=1)
        self.assertEqual(report["optimizer_steps_per_epoch"], 2)
        self.assertEqual(report["trained_steps"], 2)
        self.assertEqual(report["microbatch_size"], 1)


if __name__ == "__main__":
    unittest.main()
