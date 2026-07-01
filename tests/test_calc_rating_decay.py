"""F-017: calc_rating must apply the attendance weight ONCE, not squared.

A tautological guard (`if 'rating' in attendance_df.columns:` — always true, the
column was just created) multiplied the time-decayed rating by the raw weight a
second time, squaring it. With zero time-decay the rating must equal the weight,
not weight**2.

Dep-guarded (sfutils pulls scipy/sklearn/sentence_transformers); runs in CI,
skips locally. Local behavioral proof is via RD verify_fix on the calc_rating diff.
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
class TestCalcRatingWeightAppliedOnce(unittest.TestCase):
    def test_zero_decay_rating_equals_weight_not_squared(self):
        import pandas as pd
        import sfutils

        today = pd.Timestamp("2026-01-10").normalize()
        df = pd.DataFrame(
            {
                "ATTENDANCE_STATUS": ["attended"],
                "SESSION_DATETIME": [today],  # zero days → decay factor == 1
            }
        )
        recs_params = {
            "time_half_decay": 365,
            "attendance_type_weights": {
                "default": 0, "scheduled": 1, "waitlisted": 4, "attended": 5, "bookmark": 4,
            },
        }
        rating = sfutils.calc_rating(df, today=today, recs_params=recs_params)
        # weight applied once at zero decay => 5, not 25 (5**2)
        self.assertAlmostEqual(float(rating.iloc[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
