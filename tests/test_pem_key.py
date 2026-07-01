"""F-008: get_private_key_from_secrets_manager must not mangle the PEM body.

The remove_padding path space-joined the base64 body lines, destroying the
newline structure a PEM key needs. It must preserve newlines.
"""
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import utils  # noqa: E402

PEM = "-----BEGIN PRIVATE KEY-----\nAAAABBBB\nCCCCDDDD\n-----END PRIVATE KEY-----"


class TestPemKey(unittest.TestCase):
    def test_body_newlines_preserved(self):
        fake_client = mock.Mock()
        fake_client.get_secret_value.return_value = {"SecretString": PEM}
        with mock.patch.object(utils.boto3, "client", return_value=fake_client):
            out = utils.get_private_key_from_secrets_manager(remove_padding=True)
        # base64 body lines kept on separate lines, not collapsed with spaces
        self.assertIn("AAAABBBB\nCCCCDDDD", out)
        self.assertNotIn("AAAABBBB CCCCDDDD", out)


if __name__ == "__main__":
    unittest.main()
