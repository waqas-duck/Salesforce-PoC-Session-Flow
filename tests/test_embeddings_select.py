"""F-007: embeddings.select(names) must return the named subset, never the complement.

Before the fix, `select` used `mask = ~np.isin(...)`, returning every row NOT named.
These tests pin the positive-selection contract (and its adversarial edges).
"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from embeddings import embeddings  # noqa: E402


class TestEmbeddingsSelect(unittest.TestCase):
    def setUp(self):
        self.emb = embeddings([[1, 0], [0, 1], [1, 1]], ["a", "c", "b"])

    def test_positive_subset(self):
        # keeps the named rows, in original row order
        self.assertEqual(self.emb.select(["a", "b"]).rownames, ["a", "b"])

    def test_single_name(self):
        self.assertEqual(self.emb.select("c").rownames, ["c"])

    def test_absent_id_returns_empty(self):
        # adversarial: a non-existent id must yield nothing, not everything
        self.assertEqual(self.emb.select(["zzz"]).rownames, [])

    def test_empty_request_returns_empty(self):
        self.assertEqual(self.emb.select([]).rownames, [])

    def test_roundtrip_with_get(self):
        for name in ["a", "b", "c"]:
            np.testing.assert_array_equal(
                self.emb.select([name]).embeddings[0], self.emb.get(name)
            )


if __name__ == "__main__":
    unittest.main()
