"""Small CPU-only tests: export a trained denoiser without resetting SEL weights."""
import unittest
import torch

from ddv2_energy.export_policy import HEAD, merge_state


class ExportPolicyTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "agent._transfuser_model._backbone.weight": torch.tensor([1., 2.]),
            "agent._transfuser_model._backbone.count": torch.tensor(7, dtype=torch.int64),
            "agent." + HEAD + "plan_anchor": torch.zeros(2, 8, 2),
            "agent." + HEAD + "diff_decoder.weight": torch.tensor([3., 4.]),
            "agent." + HEAD + "time_mlp.weight": torch.tensor([1.]),
            "agent." + HEAD + "scorer_decoder.weight": torch.tensor([8., 9.]),
            "agent." + HEAD + "fine_NC_head.weight": torch.tensor([10.]),
        }
        self.trained = {key.removeprefix("agent."): value.clone() for key, value in self.base.items()
                        if "scorer_decoder" not in key and "fine_NC_head" not in key}
        self.trained[HEAD + "diff_decoder.weight"] += 0.2

    def test_preserves_selector_and_original_prefix_without_mutation(self):
        before = {key: value.clone() for key, value in self.base.items()}
        result, report = merge_state(self.base, self.trained)
        self.assertEqual(set(result), set(self.base))
        torch.testing.assert_close(result["agent." + HEAD + "diff_decoder.weight"],
                                   self.trained[HEAD + "diff_decoder.weight"], rtol=0, atol=0)
        for key in ("scorer_decoder.weight", "fine_NC_head.weight", "plan_anchor"):
            self.assertTrue(torch.equal(result["agent." + HEAD + key], self.base["agent." + HEAD + key]))
        for key in self.base:
            self.assertTrue(torch.equal(self.base[key], before[key]))
            self.assertNotEqual(result[key].data_ptr(), self.base[key].data_ptr())
        self.assertEqual(len(report["changed_policy_keys"]), 1)
        self.assertEqual(len(report["preserved_selector_keys"]), 2)

    def test_changed_nonhead_rejected(self):
        self.trained["_transfuser_model._backbone.weight"][0] += 1
        with self.assertRaisesRegex(ValueError, "frozen tensor changed"):
            merge_state(self.base, self.trained)

    def test_changed_anchor_rejected(self):
        self.trained[HEAD + "plan_anchor"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "plan_anchor changed"):
            merge_state(self.base, self.trained)

    def test_unexpected_key_rejected(self):
        self.trained[HEAD + "absent.weight"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "absent from"):
            merge_state(self.base, self.trained)

    def test_shape_and_dtype_rejected(self):
        for value in (torch.ones(3), torch.ones(2, dtype=torch.float64)):
            with self.subTest(value=value):
                self.trained[HEAD + "diff_decoder.weight"] = value
                with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
                    merge_state(self.base, self.trained)

    def test_selector_in_trained_rejected_even_if_unchanged(self):
        self.trained[HEAD + "fine_NC_head.weight"] = self.base["agent." + HEAD + "fine_NC_head.weight"].clone()
        with self.assertRaisesRegex(ValueError, "selector tensors"):
            merge_state(self.base, self.trained)

    def test_noop_export_rejected(self):
        self.trained[HEAD + "diff_decoder.weight"] = self.base["agent." + HEAD + "diff_decoder.weight"].clone()
        with self.assertRaisesRegex(ValueError, "no changed diff_decoder"):
            merge_state(self.base, self.trained)

    def test_nonfinite_input_rejected(self):
        self.trained[HEAD + "diff_decoder.weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            merge_state(self.base, self.trained)

    def test_ambiguous_prefix_rejected(self):
        self.trained["agent." + HEAD + "diff_decoder.weight"] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            merge_state(self.base, self.trained)

    def test_frozen_signed_zero_is_bitwise_checked(self):
        key = "_transfuser_model._backbone.weight"
        self.base["agent." + key][0] = 0.0
        self.trained[key][0] = -0.0
        with self.assertRaisesRegex(ValueError, "frozen tensor changed"):
            merge_state(self.base, self.trained)

    def test_trained_prefixed_and_base_unprefixed_supported(self):
        base = {key.removeprefix("agent."): value for key, value in self.base.items()}
        trained = {"agent." + key: value for key, value in self.trained.items()}
        result, _ = merge_state(base, trained)
        self.assertEqual(set(result), set(base))


if __name__ == "__main__":
    unittest.main()
