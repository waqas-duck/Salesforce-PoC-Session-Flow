"""F-011: no real @salesforce.com employee emails committed to source.

The config's attendee list must use synthetic fixtures, not real employee PII.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from sfconfig import sfconfig  # noqa: E402


class TestNoEmployeePII(unittest.TestCase):
    def test_no_salesforce_emails_in_config(self):
        for email in sfconfig().attendee_mli_email:
            self.assertNotIn("@salesforce.com", email)


if __name__ == "__main__":
    unittest.main()
