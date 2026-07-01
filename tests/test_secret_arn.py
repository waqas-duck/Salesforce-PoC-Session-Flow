"""F-013: the Secrets-Manager ARN (with AWS account id) must not be a committed default.

It is now sourced from the SNOWFLAKE_KEY_SECRET_ARN environment variable, so the
function's default no longer embeds the account number or ARN.
"""
import inspect
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import utils  # noqa: E402


class TestSecretArnExternalized(unittest.TestCase):
    def test_default_has_no_committed_arn(self):
        default = inspect.signature(
            utils.get_private_key_from_secrets_manager
        ).parameters["secret_arn"].default
        self.assertNotIn("211125482819", str(default))
        self.assertNotIn("arn:aws", str(default))


if __name__ == "__main__":
    unittest.main()
