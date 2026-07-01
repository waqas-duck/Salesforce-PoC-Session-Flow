"""F-010: calc_rating must score 'attended' rows with the 'attended' weight (5),
not the 'waitlisted' weight (4).

Dep-guarded: importing sfutils pulls scipy / scikit-learn / sentence_transformers,
so this runs in CI (where those are installed) and skips in bare environments.
Behavioral proof in the bare env is via RD verify_fix on the calc_rating diff.
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
class TestCalcRatingAttendedWeight(unittest.TestCase):
    def test_attended_outranks_waitlisted(self):
        import pandas as pd
        import sfutils

        today = pd.Timestamp("2026-01-10").normalize()
        df = pd.DataFrame(
            {
                "ATTENDANCE_STATUS": ["attended", "waitlisted"],
                "SESSION_DATETIME": [today, today],  # same-day → identical time decay
            }
        )
        recs_params = {
            "time_half_decay": 365,
            "attendance_type_weights": {
                "default": 0, "scheduled": 1, "waitlisted": 4, "attended": 5, "bookmark": 4,
            },
        }
        rating = sfutils.calc_rating(df, today=today, recs_params=recs_params)
        # attended must score strictly higher than waitlisted (weight 5 > 4);
        # before the fix both used the waitlisted weight and tied.
        self.assertGreater(rating.iloc[0], rating.iloc[1])


if __name__ == "__main__":
    unittest.main()
