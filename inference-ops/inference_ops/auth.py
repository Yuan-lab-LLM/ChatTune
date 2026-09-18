import hmac
import os
from collections.abc import Mapping


INFERENCE_OPS_AUTH_MODES = frozenset({"trusted_network", "token"})


def inference_ops_auth_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    mode = str(source.get("INFERENCE_OPS_AUTH_MODE") or "trusted_network").strip().lower()
    if mode not in INFERENCE_OPS_AUTH_MODES:
        raise RuntimeError(
            "INFERENCE_OPS_AUTH_MODE must be one of: trusted_network, token"
        )
    return mode


def inference_ops_api_token(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get("INFERENCE_OPS_API_TOKEN") or "").strip()


def inference_ops_auth_headers(
    environ: Mapping[str, str] | None = None,
    *,
    include_content_type: bool = True,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if include_content_type else {}
    if inference_ops_auth_mode(environ) == "trusted_network":
        return headers
    token = inference_ops_api_token(environ)
    if not token:
        raise RuntimeError(
            "INFERENCE_OPS_API_TOKEN is required when "
            "INFERENCE_OPS_AUTH_MODE=token"
        )
    headers["Authorization"] = f"Bearer {token}"
    return headers


def inference_ops_token_is_valid(authorization: str, expected_token: str) -> bool:
    scheme, _, presented = str(authorization or "").strip().partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        presented.strip(), expected_token
    )
