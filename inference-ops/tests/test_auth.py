import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference-ops"))

from inference_ops.auth import (
    inference_ops_auth_headers,
    inference_ops_auth_mode,
    inference_ops_token_is_valid,
)


class InferenceOpsAuthTest(unittest.TestCase):
    def test_default_mode_is_trusted_network(self):
        self.assertEqual("trusted_network", inference_ops_auth_mode({}))
        self.assertEqual(
            {"Content-Type": "application/json"},
            inference_ops_auth_headers({}),
        )

    def test_legacy_token_is_ignored_without_explicit_token_mode(self):
        environ = {"INFERENCE_OPS_API_TOKEN": "legacy-secret"}
        self.assertEqual("trusted_network", inference_ops_auth_mode(environ))
        self.assertNotIn("Authorization", inference_ops_auth_headers(environ))

    def test_token_mode_requires_token(self):
        with self.assertRaisesRegex(
            RuntimeError, "INFERENCE_OPS_API_TOKEN is required"
        ):
            inference_ops_auth_headers({"INFERENCE_OPS_AUTH_MODE": "token"})

    def test_token_mode_builds_bearer_header(self):
        headers = inference_ops_auth_headers(
            {
                "INFERENCE_OPS_AUTH_MODE": "token",
                "INFERENCE_OPS_API_TOKEN": "shared-secret",
            }
        )
        self.assertEqual("Bearer shared-secret", headers["Authorization"])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "trusted_network, token"):
            inference_ops_auth_mode({"INFERENCE_OPS_AUTH_MODE": "disabled"})

    def test_bearer_token_validation(self):
        self.assertTrue(
            inference_ops_token_is_valid("Bearer shared-secret", "shared-secret")
        )
        self.assertTrue(
            inference_ops_token_is_valid("bearer shared-secret", "shared-secret")
        )
        self.assertFalse(inference_ops_token_is_valid("", "shared-secret"))
        self.assertFalse(
            inference_ops_token_is_valid("Bearer wrong-secret", "shared-secret")
        )


if __name__ == "__main__":
    unittest.main()
