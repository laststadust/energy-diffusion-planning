"""CPU-only runner guard tests; no NAVSIM imports, CUDA or real cache files."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ddv2_energy.runtime import capture_rollout, check_context_alignment, load_manifest, train_head_only


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "requests.json"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, token="train_token", log="train_log", split="train"):
        folder = self.root / log / token
        folder.mkdir(parents=True, exist_ok=True)
        result = dict(token=token, log=log, split=split)
        for key, filename in (("feature_path", "feature.gz"), ("target_path", "target.gz"),
                              ("metric_cache_path", "metric_cache.pkl")):
            path = folder / filename
            path.write_bytes(b"synthetic path-only fixture; not a serialized NAVSIM cache")
            result[key] = str(path)
        return result

    def write(self, rows):
        self.manifest.write_text(json.dumps(rows), encoding="utf-8")
        return self.manifest

    def test_valid_disjoint_train_val_and_optional_metric(self):
        rows = [self.row(), self.row("val_token", "val_log", "val")]
        self.assertEqual(load_manifest(self.write(rows)), rows)
        capture_row = self.row("capture_token", "capture_log")
        del capture_row["metric_cache_path"]
        self.assertEqual(load_manifest(self.write([capture_row]), require_metric=False), [capture_row])
        with self.assertRaises(KeyError):
            load_manifest(self.manifest, require_metric=True)

    def test_empty_invalid_split_duplicate_and_log_overlap_fail(self):
        first = self.row()
        cases = [[], {}, [first, first], [dict(first, split="navtest")],
                 [first, self.row("different_token", "train_log", "val")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_manifest(self.write(rows))

    def test_missing_feature_target_or_metric_fails(self):
        first = self.row()
        for key in ("feature_path", "target_path", "metric_cache_path"):
            row = dict(first, **{key: str(self.root / "does_not_exist")})
            with self.subTest(field=key), self.assertRaises(FileNotFoundError):
                load_manifest(self.write([row]))

    def test_each_cache_path_must_contain_exact_log_and_token_components(self):
        first = self.row()
        wrong_paths = [self.row("different_token", "train_log"),
                       self.row("train_token", "different_log"),
                       self.row("prefix_train_token_suffix", "train_log"),
                       self.row("train_token", "prefix_train_log_suffix")]
        for replacement in wrong_paths:
            for key in ("feature_path", "target_path", "metric_cache_path"):
                row = dict(first, **{key: replacement[key]})
                with self.subTest(field=key, replacement=replacement["token"]), \
                        self.assertRaisesRegex(ValueError, "token AND log"):
                    load_manifest(self.write([row]))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        poses = torch.zeros(8, 3)
        poses[:, 0] = torch.arange(1., 9.)
        poses[:, 2] = .02 * torch.arange(1., 9.)
        self.context = dict(token="alignment_token", gt_poses=poses, v0=5.)
        # Command and acceleration magnitudes deliberately differ from speed.
        self.features = {"status_feature": torch.tensor([[0., 1., 0., 0., 3., 4., 20., -30.]])}
        self.targets = {"trajectory": poses[None].clone()}

    def test_gt_and_observed_speed_match_without_mutation(self):
        before_features, before_targets = copy.deepcopy(self.features), copy.deepcopy(self.targets)
        check_context_alignment(self.context, self.features, self.targets)
        self.assertTrue(torch.equal(self.features["status_feature"], before_features["status_feature"]))
        self.assertTrue(torch.equal(self.targets["trajectory"], before_targets["trajectory"]))

    def test_small_numeric_cache_tolerance_is_accepted(self):
        self.targets["trajectory"][0, 0, 0] += .01
        self.context["v0"] = 5.01
        check_context_alignment(self.context, self.features, self.targets)

    def test_gt_position_heading_shape_and_nonfinite_mismatches_fail(self):
        cases = []
        for dimension in (0, 1, 2):
            targets = copy.deepcopy(self.targets)
            targets["trajectory"][0, 2, dimension] += .1
            cases.append(targets)
        cases.append({"trajectory": torch.zeros(1, 6, 3)})
        cases.append({"trajectory": torch.full((1, 8, 3), float("nan"))})
        for targets in cases:
            with self.subTest(shape=targets["trajectory"].shape), \
                    self.assertRaisesRegex(ValueError, "GT/context mismatch"):
                check_context_alignment(self.context, self.features, targets)

    def test_current_speed_mismatch_and_nonfinite_feature_fail(self):
        for vx, vy in ((6., 8.), (float("nan"), 4.), (float("inf"), 4.)):
            features = copy.deepcopy(self.features)
            features["status_feature"][0, -4:-2] = torch.tensor([vx, vy])
            with self.subTest(vx=vx), self.assertRaisesRegex(ValueError, "current speed/context mismatch"):
                check_context_alignment(self.context, features, self.targets)


class StubHead(nn.Module):
    def __init__(self, capture_flag):
        super().__init__()
        self._config = SimpleNamespace(energy_capture_candidates=capture_flag)
        self.original_calls = 0

    def get_rlloss(self, *args, **kwargs):
        self.original_calls += 1
        return "original_loss_method"


class StubModel(nn.Module):
    def __init__(self, capture_flag=False, failure=None, positional=False):
        super().__init__()
        self._trajectory_head = StubHead(capture_flag)
        self.failure, self.positional = failure, positional
        self.candidates = torch.randn(1, 3, 8, 3, requires_grad=True)
        self.calls = []
        self.grad_enabled = None
        self.captured_flag = None
        self.wrapper_output = None

    def forward(self, features, targets, eta, cal_pdm):
        self.calls.append((features, targets, eta, cal_pdm))
        self.grad_enabled = torch.is_grad_enabled()
        self.captured_flag = self._trajectory_head._config.energy_capture_candidates
        if self.positional:
            self.wrapper_output = self._trajectory_head.get_rlloss("unused", self.candidates)
        else:
            self.wrapper_output = self._trajectory_head.get_rlloss("unused", old_pred=self.candidates)
        if self.failure == "forward":
            raise RuntimeError("synthetic forward failure")
        if self.failure == "missing_output":
            return {}
        return {"energy_candidates": self.wrapper_output}


class RolloutTests(unittest.TestCase):
    def test_single_no_grad_rollout_uses_old_pred_and_restores_wrapper(self):
        for previous in (False, True):
            for positional in (False, True):
                with self.subTest(previous=previous, positional=positional):
                    model = StubModel(previous, positional=positional)
                    agent = SimpleNamespace(_transfuser_model=model)
                    head = model._trajectory_head
                    original = head.get_rlloss
                    features, targets = {"fixture": True}, {"trajectory": True}
                    before_grad = torch.is_grad_enabled()
                    result = capture_rollout(agent, features, targets)
                    self.assertIs(result, model.candidates)
                    self.assertEqual(len(model.calls), 1)
                    self.assertIs(model.calls[0][0], features)
                    self.assertIs(model.calls[0][1], targets)
                    self.assertEqual(model.calls[0][2:], (1., False))
                    self.assertFalse(model.grad_enabled)
                    self.assertTrue(model.captured_flag)
                    self.assertEqual(torch.is_grad_enabled(), before_grad)
                    self.assertEqual(head.get_rlloss, original)
                    self.assertIs(head._config.energy_capture_candidates, previous)
                    self.assertEqual(head.original_calls, 0)
                    self.assertEqual(head.get_rlloss(), "original_loss_method")
                    self.assertEqual(head.original_calls, 1)

    def test_forward_or_output_exception_restores_methods_and_previous_flag(self):
        for previous in (False, True):
            for failure, error in (("forward", RuntimeError), ("missing_output", KeyError)):
                with self.subTest(previous=previous, failure=failure):
                    model = StubModel(previous, failure=failure)
                    head = model._trajectory_head
                    original = head.get_rlloss
                    before_grad = torch.is_grad_enabled()
                    with self.assertRaises(error):
                        capture_rollout(SimpleNamespace(_transfuser_model=model), {}, {})
                    self.assertEqual(head.get_rlloss, original)
                    self.assertIs(head._config.energy_capture_candidates, previous)
                    self.assertEqual(torch.is_grad_enabled(), before_grad)
                    self.assertEqual(head.get_rlloss(), "original_loss_method")


class HeadModeTests(unittest.TestCase):
    def test_non_head_batchnorm_frozen_eval_head_train_and_state_unchanged(self):
        agent = nn.Module()
        model = nn.Module()
        model._backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        model._trajectory_head = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        model._backbone.requires_grad_(False)  # Matches the original frozen backbone.
        agent._transfuser_model = model
        agent.train()
        original_state = {key: value.clone() for key, value in agent.state_dict().items()}
        grad_flags = {key: value.requires_grad for key, value in agent.named_parameters()}

        train_head_only(agent)

        self.assertFalse(agent.training)
        self.assertFalse(model.training)
        self.assertTrue(all(not module.training for module in model._backbone.modules()))
        self.assertTrue(all(module.training for module in model._trajectory_head.modules()))
        self.assertEqual(grad_flags, {key: value.requires_grad for key, value in agent.named_parameters()})
        self.assertEqual(set(original_state), set(agent.state_dict()))
        for key, value in original_state.items():
            self.assertTrue(torch.equal(value, agent.state_dict()[key]), key)

        sample = torch.arange(16., dtype=torch.float32).reshape(4, 4)
        frozen_count = model._backbone[1].num_batches_tracked.clone()
        head_count = model._trajectory_head[1].num_batches_tracked.clone()
        features = model._backbone(sample)
        model._trajectory_head(features).sum().backward()
        self.assertTrue(torch.equal(frozen_count, model._backbone[1].num_batches_tracked))
        self.assertEqual(model._trajectory_head[1].num_batches_tracked.item(), head_count.item()+1)
        self.assertTrue(all(parameter.grad is None for parameter in model._backbone.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model._trajectory_head.parameters()))


if __name__ == "__main__":
    unittest.main()
