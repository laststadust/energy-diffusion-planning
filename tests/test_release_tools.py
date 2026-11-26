"""Release tooling tests using only temporary synthetic files."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class StageTests(unittest.TestCase):
    def setUp(self):
        self.runner = module("train_stage")
        self.recipe = json.loads((ROOT / "configs/training.json").read_text())
        self.args = argparse.Namespace(stage="graph", out="new-run", seed=0,
            bank="bank.pt", contexts="contexts", device="cpu", manifest="manifest.json",
            checkpoint="base.ckpt", mode="dual_graph", graph="graph.pt", resume=None)

    def test_graph_recipe(self):
        cmd = self.runner.build_command(self.args, self.recipe)
        self.assertIn("ddv2_energy.scale_graph", cmd)
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "10")
        self.assertEqual(cmd[cmd.index("--microbatch-size") + 1], "16")

    def test_policy_modes(self):
        self.args.stage = "policy"
        cmd = self.runner.build_command(self.args, self.recipe)
        self.assertIn("--graph", cmd)
        self.assertEqual(cmd[cmd.index("--strength") + 1], "0.1")
        self.args.mode = "original"
        self.assertNotIn("--graph", self.runner.build_command(self.args, self.recipe))

    def test_missing_inputs_and_unknown_settings(self):
        self.args.bank = None
        with self.assertRaises(ValueError):
            self.runner.build_command(self.args, self.recipe)
        self.args.bank = "bank.pt"
        self.recipe["energy"]["motion_weight"] = 1
        with self.assertRaises(ValueError):
            self.runner.build_command(self.args, self.recipe)


@unittest.skipUnless(shutil.which("git"), "Git is required for patch-tool tests")
class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tool = module("apply_upstream")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.file = self.repo / "model.py"
        self.file.write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "model.py"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                        "commit", "-qm", "fixture"], check=True)
        head = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.patch = self.repo / "change.patch"
        text = "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        with self.patch.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
        self.lock = {"commit": head, "patch_sha256": digest(text), "files": [
            {"path": "model.py", "before_sha256": digest("value = 1\n"),
             "after_sha256": digest("value = 2\n")}]}

    def test_check_apply_and_reapply(self):
        self.assertEqual(self.tool.apply_integration(self.repo, self.lock, self.patch, check_only=True), "compatible")
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.assertEqual(self.tool.apply_integration(self.repo, self.lock, self.patch), "applied")
        self.assertEqual(self.tool.apply_integration(self.repo, self.lock, self.patch), "already applied")

    def test_reject_unexpected_source(self):
        self.file.write_text("value = 3\n")
        with self.assertRaises(ValueError):
            self.tool.apply_integration(self.repo, self.lock, self.patch)
        self.assertEqual(self.file.read_text(), "value = 3\n")

    def test_reject_wrong_revision_or_patch(self):
        self.lock["commit"] = "0" * 40
        with self.assertRaises(ValueError):
            self.tool.apply_integration(self.repo, self.lock, self.patch)
        self.lock["commit"] = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.lock["patch_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.tool.apply_integration(self.repo, self.lock, self.patch)


if __name__ == "__main__":
    unittest.main()
