"""Command help must not require datasets, CUDA, or upstream imports."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RuntimeCLITests(unittest.TestCase):
    def test_entrypoint_help_without_navsim(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment.pop("NAVSIM_DEVKIT_ROOT", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        for module in ("collect_bank", "scale_bank", "graph_fit", "scale_graph", "run", "export_policy"):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-m", "ddv2_energy." + module, "--help"],
                    cwd=root, env=environment, text=True, capture_output=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
