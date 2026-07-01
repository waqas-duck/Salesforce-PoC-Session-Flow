"""F-014: recs_params['ret_recs_size'] must be a positive top-N size.

It was `-30`, which turns every `seq[0:ret_recs_size]` top-N slice into
`seq[0:-30]` (drops the last 30 rows) instead of "keep the top 30".
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from sfconfig import sfconfig  # noqa: E402


class TestRecsParams(unittest.TestCase):
    def test_ret_recs_size_is_positive(self):
        params = sfconfig().recs_params
        self.assertGreater(
            params["ret_recs_size"], 0, "ret_recs_size must be a positive top-N size"
        )
        self.assertEqual(params["ret_recs_size"], 30)


if __name__ == "__main__":
    unittest.main()
