"""Regression guard for F-001: every src/*.py must parse.

The pipeline was non-importable because three modules had syntax errors
(utils.py, sfloader.py, experiments.py). This test fails if any source
module stops parsing again. It only compiles (parses) the files — it does
not import them, so it triggers no module-scope side effects.
"""
import pathlib
import py_compile
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


class TestSourceCompiles(unittest.TestCase):
    def test_all_src_modules_parse(self):
        failures = []
        for path in sorted(SRC.glob("*.py")):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.name}: {exc.msg}")
        self.assertEqual(
            failures, [], "source files failed to parse:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    unittest.main()
