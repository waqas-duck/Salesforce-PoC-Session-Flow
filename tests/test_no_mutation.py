"""F-016: calc_rating must not mutate the caller's DataFrame.

The function previously wrote 'rating'/'days' columns onto the caller-owned frame
(aliasing). It must operate on a copy, leaving the caller's DataFrame unchanged.

Dep-guarded (sfutils pulls scipy/sklearn/sentence_transformers); runs in CI, skips locally.
"""
import importlib.util
import pathlib
import sys
import unittest

_HAVE_DEPS = all(
    importlib.util.find_spec(m) for m in ("scipy", "sklearn", "sentence_transformers")
)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


@unittest.skipUnless(_HAVE_DEPS, "sfutils deps (scipy/sklearn/sentence_transformers) not installed")
class TestNoCallerMutation(unittest.TestCase):
    def test_calc_rating_does_not_mutate_caller(self):
        import pandas as pd
        import sfutils

        today = pd.Timestamp("2026-01-10").normalize()
        df = pd.DataFrame({"ATTENDANCE_STATUS": ["attended"], "SESSION_DATETIME": [today]})
        before_cols = list(df.columns)
        recs_params = {
            "time_half_decay": 365,
            "attendance_type_weights": {
                "default": 0, "scheduled": 1, "waitlisted": 4, "attended": 5, "bookmark": 4,
            },
        }
        sfutils.calc_rating(df, today=today, recs_params=recs_params)
        self.assertEqual(list(df.columns), before_cols)  # caller frame untouched


if __name__ == "__main__":
    unittest.main()
