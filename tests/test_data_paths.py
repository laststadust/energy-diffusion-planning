"""Path and storage checks that do not require maps, checkpoints or CUDA."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ddv2_energy.prepare_experiment import validate_preparation_paths
from ddv2_energy.scale_data import disk_guard


class PreparationPathTests(unittest.TestCase):
    def roots(self, root):
        root = Path(root)
        repo, data = root / "repo", root / "dataset"
        repo.mkdir()
        (data / "navsim_logs" / "trainval").mkdir(parents=True)
        (data / "maps").mkdir()
        return SimpleNamespace(repo=str(repo), data=str(data),
                               out=str(root / "generated"), feature_cache=None)

    def test_explicit_roots_work_on_any_mount_without_existing_cache(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.roots(root)
            validate_preparation_paths(args)
            self.assertTrue(Path(args.out).is_absolute())
            self.assertFalse(Path(args.out).exists())
            self.assertIsNone(args.feature_cache)

    def test_input_and_output_overlap_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.roots(root)
            for output in (args.data, str(Path(args.data) / "generated"), root):
                args.out = output
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlap"):
                    validate_preparation_paths(args)
            self.assertFalse((Path(args.data) / "generated").exists())

    def test_explicit_missing_cache_fails_instead_of_silently_ignoring_it(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.roots(root)
            args.feature_cache = str(Path(root) / "missing")
            with self.assertRaises(FileNotFoundError):
                validate_preparation_paths(args)

    def test_reused_cache_cannot_be_written_into(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.roots(root)
            cache = Path(root) / "cache"
            cache.mkdir()
            args.feature_cache = str(cache)
            args.out = str(cache / "generated")
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_preparation_paths(args)

    def test_disk_limit_uses_configured_threshold(self):
        with patch("ddv2_energy.scale_data.shutil.disk_usage",
                   return_value=SimpleNamespace(free=10 * 1024 ** 3)):
            disk_guard("unused", minimum_gib=9)
            with self.assertRaisesRegex(RuntimeError, "11 GiB"):
                disk_guard("unused", minimum_gib=11)


if __name__ == "__main__":
    unittest.main()
