"""HTTP transport for backend configuration; excluded from LLM/user tools."""
import json
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .auth import inference_ops_token_is_valid


def authorized_backend(authorization, token):
    # Provisioning always requires service authentication, even if other APIs
    # use trusted_network. Local deployments can instead use the Docker channel.
    return bool(token) and inference_ops_token_is_valid(authorization, token)


def forward_configuration(tool_url, service_token, payload=None, *, opener=urlopen):
    parsed = urlsplit(tool_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/worker/tool", "/inference_agent/tool", "/inference_agent"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    endpoint = "/runtime/config" if payload is not None else "/runtime/config/status"
    url = urlunsplit((parsed.scheme, parsed.netloc, path + endpoint, "", ""))
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, method="POST" if body is not None else "GET",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {service_token}"})
    try:
        with opener(request, timeout=10) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError()
        # Never relay arbitrary worker responses containing credentials.
        return {key: result[key] for key in ("configured", "runtime_node_id", "source") if key in result}
    except Exception:
        raise ValueError("无法为目标 worker 初始化运行时配置，请检查 worker 地址、服务鉴权与节点绑定") from None
