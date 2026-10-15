"""Backend-only credential initialization; never starts inference processes."""
import hashlib
import json
import os
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit, urlencode
from urllib.request import Request, urlopen

_cache = {}
_lock = threading.Lock()
REMOTE_INITIALIZE = '''
import importlib, json, os, sys
from pathlib import Path
try:
    payload = json.load(sys.stdin)
    roots = []
    for candidate in payload['candidates']:
        root = Path(candidate)
        if (root / 'inference_ops/experiment_runtime/runtime_credentials.py').is_file():
            roots.append(root.resolve())
    roots = list(dict.fromkeys(roots))
    if len(roots) != 1:
        raise ValueError('无法唯一定位推理代码，请设置 MEDFLOW_INFERENCE_OPS_DIR 为推理容器内目录')
    sys.path.insert(0, str(roots[0]))
    os.chdir(roots[0])
    importlib.import_module('inference_ops.environment').load_runtime_environment()
    credentials = importlib.import_module('inference_ops.experiment_runtime.runtime_credentials')
    print(json.dumps(credentials.initialize(payload['credentials'])))
except Exception as exc:
    print(json.dumps({'error': str(exc)}))
    raise SystemExit(1)
'''


def api_root(url):
    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("INFERENCE_AGENT_URL 格式无效")
    path = parsed.path.rstrip("/")
    for suffix in ("/worker/tool", "/inference_agent/tool", "/inference_agent"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def request_json(url, headers, payload=None, *, opener=urlopen):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, method="POST" if body is not None else "GET", headers=headers)
    try:
        with opener(request, timeout=10) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError("推理 Agent 缺少运行时初始化接口，请更新并手动重启推理 Agent") from None
        raise RuntimeError(f"推理运行时初始化请求失败（HTTP {exc.code}）；请检查服务鉴权和节点绑定") from None
    except (URLError, OSError, ValueError):
        raise RuntimeError("推理运行时初始化失败，请确认手动启动的推理 Agent 地址可达") from None


def provision_via_docker(url, credentials, environ, *, runner=subprocess.run):
    from inference_container import local_inference_settings, container_candidates
    settings = local_inference_settings({**environ, "INFERENCE_AGENT_URL": url, "START_LOCAL_INFERENCE": "auto"})
    if settings is None:
        raise RuntimeError("远端推理节点需要配置 INFERENCE_OPS_API_TOKEN 以鉴权下发，或保留该节点的手动 .env 配置")
    try:
        result = runner(["docker", "inspect", settings["container"]], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise ValueError()
        inspection = json.loads(result.stdout)[0]
        if not inspection.get("State", {}).get("Running"):
            raise ValueError()
        payload = {"candidates": container_candidates(inspection, environ.get("MEDFLOW_INFERENCE_OPS_DIR", "")), "credentials": credentials}
        result = runner(["docker", "exec", "-i", settings["container"], settings["python"], "-c", REMOTE_INITIALIZE],
                        input=json.dumps(payload), capture_output=True, text=True, timeout=20)
        value = json.loads(result.stdout)
        if result.returncode or value.get("configured") is not True or value.get("runtime_node_id") != credentials["runtime_node_id"]:
            raise ValueError()
        return {"configured": True, "runtime_node_id": value["runtime_node_id"]}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        raise RuntimeError("无法通过 Docker 初始化推理节点；请检查容器、推理代码版本及 MEDFLOW_INFERENCE_OPS_DIR（推理容器内路径）") from None


def ensure_inference_runtime(url, environ=None, *, opener=urlopen, runner=subprocess.run, now=time.monotonic):
    environ = os.environ if environ is None else environ
    node = str(environ.get("MEDFLOW_RESOURCE_NODE_ID") or "").strip()
    if not node:
        return  # Legacy callers with an independently configured worker.
    root = api_root(url)
    token = str(environ.get("MEDFLOW_RUNTIME_NODE_TOKEN") or environ.get("MEDFLOW_STUDIO_NODE_TOKEN") or "").strip()
    service_token = str(environ.get("INFERENCE_OPS_API_TOKEN") or "").strip()
    credentials = {"studio_url": str(environ.get("STUDIO_URL") or "").strip(), "runtime_node_id": node, "node_token": token}
    key = hashlib.sha256(json.dumps([root, credentials, service_token, environ.get("MEDFLOW_INFERENCE_CONTAINER"),
        environ.get("MEDFLOW_LOCAL_EVALUATE_CONTAINER"), environ.get("MEDFLOW_INFERENCE_OPS_DIR")], sort_keys=True).encode()).hexdigest()
    headers = {"Content-Type": "application/json"}
    if service_token:
        headers["Authorization"] = f"Bearer {service_token}"
    with _lock:
        if _cache.get(key, 0) > now():
            return
        status_url = root + "/runtime/config/status?" + urlencode({"runtime_node_id": node})
        status = request_json(status_url, headers, opener=opener)
        bound = status.get("runtime_node_id")
        if bound and bound != node:
            raise RuntimeError("推理 Agent 已绑定其他 Runtime 节点，拒绝发送本节点凭证")
        if status.get("configured") and status.get("source") == "environment":
            _cache[key] = now() + 30
            return
        if not credentials["studio_url"] or not token or token.startswith("replace-"):
            raise RuntimeError("主 Agent Runtime 缺少 STUDIO_URL 或节点 Token，无法初始化推理节点")
        if service_token:
            result = request_json(root + "/runtime/config", headers, credentials, opener=opener)
        else:
            result = provision_via_docker(url, credentials, environ, runner=runner)
            if status.get("source") != "controller":
                result = request_json(status_url, headers, opener=opener)
        if result.get("configured") is not True or result.get("runtime_node_id") != node:
            raise RuntimeError("推理节点初始化未确认，请检查推理 Agent 的节点配置")
        import logging
        logging.getLogger(__name__).info("推理节点运行时配置已初始化 node=%s channel=%s", node, "http" if service_token else "docker")
        _cache[key] = now() + 30


def warm_up_inference_runtime():
    import logging
    def initialize():
        try:
            ensure_inference_runtime(os.getenv("INFERENCE_AGENT_URL", "http://127.0.0.1:8899/inference_agent"))
        except Exception as exc:
            logging.getLogger(__name__).warning("推理节点暂未初始化，首次推理调用时会重试：%s", exc)
    threading.Thread(target=initialize, name="inference-runtime-init", daemon=True).start()
