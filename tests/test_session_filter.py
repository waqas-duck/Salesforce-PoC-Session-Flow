"""F-004: sutil.session_filter must resolve and filter sessions (was an undefined symbol).

The symbol had moved onto sfloader._session_filter; sfutils.session_filter now
delegates to it. Dep-guarded (sfutils pulls scipy/sklearn/sentence_transformers).
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
class TestSessionFilter(unittest.TestCase):
    def test_keeps_only_recommendable_session_types(self):
        import pandas as pd
        import sfutils

        df = pd.DataFrame(
            {
                "SESSION_TYPE": ["Breakout", "Meal"],
                "STATUS": ["Accepted", "Accepted"],
                "PUBLISHED": [True, True],
                "SESSIONCODE": ["OK-1", "OK-2"],
            }
        )
        out = sfutils.session_filter(df, True, version=2)
        # 'Breakout' is a recommendable type; 'Meal' is not
        self.assertEqual(list(out["SESSION_TYPE"]), ["Breakout"])


if __name__ == "__main__":
    unittest.main()
