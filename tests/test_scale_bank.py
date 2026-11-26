"""CPU-only synthetic tests; outputs are not experiment candidate banks."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ddv2_energy.scale_bank import (PROTOCOL, canonical_hash, committed_shard, merge_shards,
                         publish_shard, seed_token, token_seed, validate_records,
                         verify_published_bank)


class ScaleBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = {"tokens": ["a", "b", "c"], "shard_size": 2,
                     "checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                     "seed": 0, "split_provenance": {}, "protocol": PROTOCOL}
        self.identity = canonical_hash(self.spec)

    def tearDown(self):
        self.tmp.cleanup()

    def records(self, tokens):
        records = []
        for token in tokens:
            seed_token(0, token)
            records.append({"token": token, "poses": torch.randn(80, 8, 3)})
        return records

    def test_token_rng_independent_of_order_and_resume(self):
        first = {r["token"]: r["poses"] for r in self.records(["a", "b", "c"])}
        second = {r["token"]: r["poses"] for r in self.records(["c", "a", "b"])}
        for token in first:
            self.assertTrue(torch.equal(first[token], second[token]))
        self.assertNotEqual(token_seed(0, "a"), token_seed(1, "a"))
        seed_token(0, "a")
        values = (random.random(), float(np.random.random()))
        seed_token(0, "a")
        self.assertEqual(values, (random.random(), float(np.random.random())))
        with self.assertRaises(ValueError):
            token_seed(0, "../unsafe")

    def test_partial_bank_never_published_then_full_merge(self):
        out = self.root / "bank.pt"
        publish_shard(self.root, 0, self.identity, self.records(["a", "b"]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_shards(self.root, self.spec, out)
        self.assertFalse(out.exists())
        publish_shard(self.root, 1, self.identity, self.records(["c"]))
        self.assertEqual(merge_shards(self.root, self.spec, out), 3)
        bank = torch.load(out, map_location="cpu", weights_only=False)
        self.assertTrue(bank["coverage_complete"])
        self.assertEqual(bank["collection_protocol"], PROTOCOL)
        self.assertFalse(bank["pilot_bitwise_equivalent"])
        self.assertEqual([r["token"] for r in bank["records"]], ["a", "b", "c"])
        self.assertEqual(verify_published_bank(self.root, self.spec, out), 3)
        bank["records"][0]["poses"][0, 0, 0] += 1.
        torch.save(bank, out)
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_published_bank(self.root, self.spec, out)
        with self.assertRaises(FileExistsError):
            merge_shards(self.root, self.spec, out)

    def test_corruption_spec_order_duplicate_and_nonfinite_rejected(self):
        records = self.records(["a", "b"])
        publish_shard(self.root, 0, self.identity, records)
        for identity, tokens in (("x" * 64, ["a", "b"]), (self.identity, ["b", "a"])):
            with self.assertRaises(ValueError):
                committed_shard(self.root, 0, identity, tokens)
        for bad, tokens in ((records, ["a", "a"]), (records[:1], ["a", "b"])):
            with self.assertRaises(ValueError):
                validate_records(bad, tokens)
        bad = copy.deepcopy(records)
        bad[0]["poses"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_records(bad, ["a", "b"])
        meta = json.loads((self.root / "shard-000000.json").read_text())
        path = self.root / meta["file"]
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash"):
            committed_shard(self.root, 0, self.identity, ["a", "b"])

    def test_uncommitted_payload_is_not_a_reusable_shard(self):
        torch.save({"records": self.records(["a", "b"])}, self.root / "data-orphan.pt")
        self.assertIsNone(committed_shard(self.root, 0, self.identity, ["a", "b"]))
        publish_shard(self.root, 0, self.identity, self.records(["a", "b"]))
        self.assertEqual(len(committed_shard(self.root, 0, self.identity, ["a", "b"])), 2)
        with self.assertRaises(FileExistsError):
            publish_shard(self.root, 0, self.identity, self.records(["a", "b"]))
        self.assertTrue((self.root / "data-orphan.pt").exists())


if __name__ == "__main__":
    unittest.main()
