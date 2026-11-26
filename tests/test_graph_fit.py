"""Small synthetic-only checks.  Resulting weights are NOT experimental weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from ddv2_energy.interaction import InteractionConfig, InteractionEnergy
from ddv2_energy.graph_fit import CandidateRecord, evaluate, fit_graph, load_bank, make_batch


def fixture():
    actors = torch.tensor([[12., 0., 0., 4., 2., 0., 0., 1., 1., 1.]])
    poses = torch.zeros(3, 8, 3)
    poses[:, :, 0] = torch.arange(1, 9).float()[None] * 2.
    poses[1, :, 1] = 5.
    poses[2, :, 1] = -5.
    rows, contexts = [], {}
    for split in ("train", "val"):
        token = "fixture_" + split
        contexts[token] = {"token": token, "log": "log_" + split, "split": split,
                           "horizon": 8, "dt": .5, "actors": actors.clone(),
                           "mask": torch.ones(1, dtype=torch.bool), "v0": 4.,
                           "future": actors.unsqueeze(0).repeat(9, 1, 1),
                           "future_mask": torch.ones(9, 1, dtype=torch.bool),
                           "label_ok": True, "truncated": False}
        rows.append({"token": token, "poses": poses.clone()})
    bank = {"format": "ddv2_candidate_bank_v1", "source": "DiffusionDriveV2",
            "checkpoint_sha256": "a" * 64, "synthetic": True, "records": rows}
    return bank, contexts


class GraphFitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bank, self.contexts = fixture()
        self.path = self.root / "bank.pt"
        self.store = mock.patch("ddv2_energy.graph_fit.ContextStore")
        self.mock_store = self.store.start()
        self.mock_store.return_value.load.side_effect = lambda token: self.contexts[token]
        self.config = InteractionConfig(hidden=8, layers=1, heads=2)

    def tearDown(self):
        self.store.stop()
        self.temp.cleanup()

    def write_bank(self):
        torch.save(self.bank, self.path)

    def test_reject_bad_source_hash_duplicate_and_missing_split(self):
        for key, value in (("source", "DiffusionDriveV1"), ("checkpoint_sha256", "short"),
                           ("format", "old_native_risk")):
            with self.subTest(key=key):
                bank = copy.deepcopy(self.bank)
                bank[key] = value
                torch.save(bank, self.path)
                with self.assertRaises(ValueError):
                    load_bank(self.path, "unused")
        for records in ([self.bank["records"][0]], self.bank["records"] * 2):
            bank = dict(self.bank, records=records)
            torch.save(bank, self.path)
            with self.assertRaises(ValueError):
                load_bank(self.path, "unused")

    def test_reject_log_overlap_partial_future_truncation_token_mismatch(self):
        cases = [("log", "log_train"), ("label_ok", False),
                 ("truncated", True), ("token", "different")]
        for key, value in cases:
            with self.subTest(key=key):
                self.bank, self.contexts = fixture()
                self.contexts["fixture_val"][key] = value
                self.write_bank()
                with self.assertRaises(ValueError):
                    load_bank(self.path, "unused")
        self.bank, self.contexts = fixture()
        self.contexts["fixture_val"]["future_mask"][2, 0] = False
        self.write_bank()
        with self.assertRaises(ValueError):
            load_bank(self.path, "unused")

    def test_future_changes_target_not_forward_inputs(self):
        self.write_bank()
        records, _ = load_bank(self.path, "unused")
        r = records[0]
        first = make_batch([(r, 0)], "cpu", self.config)
        changed = copy.deepcopy(r)
        changed.context["future"][1:, :, 0] = 1000.
        second = make_batch([(changed, 0)], "cpu", self.config)
        for key in first[0]:
            self.assertTrue(torch.equal(first[0][key], second[0][key]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertFalse(torch.equal(first[3], second[3]))
        self.assertFalse(first[3].requires_grad)
        model = InteractionEnergy(self.config).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*first[:3]), model(*second[:3])))

    def test_training_updates_only_graph_and_marks_synthetic(self):
        self.write_bank()
        torch.manual_seed(7)
        initial = {k: v.clone() for k, v in InteractionEnergy(self.config).state_dict().items()}
        output = self.root / "fit"
        report = fit_graph(self.path, "unused", output, epochs=1, batch_size=2,
                           device="cpu", seed=7, config=self.config)
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["format"], "ddv2_dual_graph_v1")
        self.assertTrue(checkpoint["provenance"]["synthetic"])
        self.assertGreater(checkpoint["trained_steps"], 0)
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in checkpoint["state_dict"].items()))
        self.assertGreater(report["history"][0]["max_preclip_gradient_norm"], 0.)
        self.assertEqual(report["history"][0]["validation"]["candidates"], 3)
        self.assertFalse(report["policy_parameters_updated"])
        with self.assertRaises(FileExistsError):
            fit_graph(self.path, "unused", output, epochs=1, device="cpu", config=self.config)

    def test_candidate_weighted_metrics_and_ranking(self):
        self.write_bank()
        records, _ = load_bank(self.path, "unused")
        model = InteractionEnergy(self.config).eval()
        small = evaluate(model, records, "cpu", 1)
        whole = evaluate(model, records, "cpu", 3)
        self.assertEqual(small["candidates"], 6)
        self.assertGreater(small["ranking_pair_count"], 0)
        self.assertAlmostEqual(small["mae"], whole["mae"], places=5)
        self.assertAlmostEqual(small["huber"], whole["huber"], places=5)


if __name__ == "__main__":
    unittest.main()
