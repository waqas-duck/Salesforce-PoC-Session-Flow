"""F-009: calc_user_candidates must forward its `events` argument to
get_user_attendance (the body hardcoded events=None, dropping caller context).

Dep-guarded (sfutils pulls scipy/sklearn/sentence_transformers); runs in CI,
skips locally. Stubs get_user_attendance to capture the forwarded value.
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
class TestEventsForwarded(unittest.TestCase):
    def test_events_is_forwarded(self):
        import pandas as pd
        import sfutils

        captured = {}
        original = sfutils.get_user_attendance

        def stub(user_id, past_attendance, events=None):
            captured["events"] = events
            return pd.DataFrame(columns=["ATTENDEE_ID", "SESSION_ID", "rating"])

        sfutils.get_user_attendance = stub
        try:
            try:
                sfutils.calc_user_candidates(
                    "u1", pd.DataFrame(), pd.DataFrame(), events="SENTINEL"
                )
            except Exception:
                pass  # downstream may fail on empty frames; we only assert forwarding
        finally:
            sfutils.get_user_attendance = original

        self.assertEqual(captured.get("events"), "SENTINEL")


if __name__ == "__main__":
    unittest.main()
