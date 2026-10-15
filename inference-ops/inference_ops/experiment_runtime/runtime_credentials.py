"""Node credentials provisioned by a trusted backend, never by user tools."""
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import uuid


_lock = threading.RLock()


def credential_path(state_dir=None):
    root = Path(state_dir or os.getenv("INFERENCE_OPS_STATE_DIR") or Path(__file__).resolve().parents[2] / "runtime")
    return root / "credentials" / "studio.json"


def validate(payload):
    required = {"studio_url", "runtime_node_id", "node_token"}
    if not isinstance(payload, dict) or set(payload) != required or any(not isinstance(payload[key], str) for key in required):
        raise ValueError("运行时初始化需要 studio_url、runtime_node_id 和 node_token")
    values = {key: value.strip() for key, value in payload.items()}
    values["studio_url"] = values["studio_url"].rstrip("/")
    try:
        url = urlsplit(values["studio_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError()
        url.port
        if any(char.isspace() for char in values["studio_url"]):
            raise ValueError()
    except ValueError:
        raise ValueError("studio_url 必须为有效的 Studio http/https 地址") from None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", values["runtime_node_id"]):
        raise ValueError("runtime_node_id 格式无效")
    token = values["node_token"]
    if not token or len(token) > 8192 or any(char.isspace() for char in token) or token.startswith("replace-"):
        raise ValueError("node_token 未配置或格式无效")
    return values


def saved_credentials(state_dir=None):
    path = credential_path(state_dir)
    if not path.exists():
        return {}
    if path.is_symlink():
        raise ValueError("运行时凭证文件不能为符号链接")
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        raise ValueError("运行时凭证文件不可读取或格式无效，请由后端重新初始化") from None


def explicit_credentials():
    token = os.getenv("MEDFLOW_RUNTIME_NODE_TOKEN", "").strip()
    legacy = os.getenv("MEDFLOW_STUDIO_NODE_TOKEN", "").strip()
    if token and legacy and token != legacy:
        raise ValueError("MEDFLOW_RUNTIME_NODE_TOKEN 与 MEDFLOW_STUDIO_NODE_TOKEN 必须一致")
    return {"studio_url": os.getenv("STUDIO_URL", "").strip().rstrip("/"),
            "runtime_node_id": os.getenv("MEDFLOW_RESOURCE_NODE_ID", "").strip(), "node_token": token or legacy}


def effective_credentials(state_dir=None):
    saved = saved_credentials(state_dir)
    explicit = explicit_credentials()
    if saved and explicit["runtime_node_id"] and explicit["runtime_node_id"] != saved["runtime_node_id"]:
        raise ValueError("运行时凭证的节点身份与部署配置不一致")
    values = {key: explicit.get(key) or saved.get(key, "") for key in explicit}
    if not all(values.values()):
        raise ValueError("实验记录尚未初始化：等待主 Agent 后端下发 Studio/节点配置，或配置 STUDIO_URL、MEDFLOW_RESOURCE_NODE_ID 和 MEDFLOW_RUNTIME_NODE_TOKEN（兼容 MEDFLOW_STUDIO_NODE_TOKEN）")
    return validate(values)


def runtime_status(state_dir=None):
    try:
        values = effective_credentials(state_dir)
        source = "environment" if all(explicit_credentials().values()) else "backend"
        return {"configured": True, "runtime_node_id": values["runtime_node_id"], "source": source}
    except ValueError:
        return {"configured": False, "runtime_node_id": os.getenv("MEDFLOW_RESOURCE_NODE_ID", "").strip()}


def current_node_id(state_dir=None):
    status = runtime_status(state_dir)
    node = status["runtime_node_id"]
    if node:
        os.environ.setdefault("MEDFLOW_RESOURCE_NODE_ID", node)
    return node


def initialize(payload, state_dir=None):
    values = validate(payload)
    with _lock:
        explicit = explicit_credentials()
        saved = saved_credentials(state_dir)
        bound_nodes = {node for node in (explicit["runtime_node_id"], saved.get("runtime_node_id")) if node}
        if any(node != values["runtime_node_id"] for node in bound_nodes):
            raise ValueError("推理节点已经绑定其他 runtime_node_id，拒绝覆盖")
        if any(value and value != values[key] for key, value in explicit.items()):
            raise ValueError("下发配置与推理进程的显式环境配置不一致，请先统一部署配置")
        if not all(explicit.values()) and saved != values:
            path = credential_path(state_dir)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.parent.is_symlink() or path.is_symlink():
                raise ValueError("运行时凭证目录或文件不能为符号链接")
            path.parent.chmod(0o700)
            temporary = path.with_name(f".studio-{uuid.uuid4().hex}.tmp")
            try:
                with os.fdopen(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w", encoding="utf-8") as stream:
                    json.dump(values, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return {"configured": True, "runtime_node_id": values["runtime_node_id"]}


def studio_connection():
    values = effective_credentials()
    return values["studio_url"], values["node_token"]
