"""Bootstrap executed inside the inference container, with settings on stdin.

The caller sends this source through Docker; Agent and inference mounts need
not share paths. Only controller processes created here may be stopped here.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler
from urllib.parse import urlsplit


def locate_ops(candidates):
    for candidate in dict.fromkeys(candidates):
        root = Path(candidate)
        if (root / "inference_ops/__main__.py").is_file():
            return root.resolve()
    raise ValueError("推理容器中未找到 inference-ops；请设置 MEDFLOW_INFERENCE_OPS_DIR 为推理容器内的实际目录")


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (OSError, ValueError, IndexError):
        return None


def owned_process(record):
    pid = record.get("pid")
    if not pid or not record.get("identity") or process_identity(pid) != record["identity"]:
        return False
    try:
        marker = f"MEDFLOW_INFERENCE_LAUNCH_ID={record['launch_id']}".encode()
        return marker in Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0") and os.getpgid(pid) == pid
    except (OSError, KeyError, ValueError):
        return False


def read_record(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def stop_owned(record):
    if not owned_process(record):
        return False
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while owned_process(record) and time.monotonic() < deadline:
        time.sleep(0.1)
    if owned_process(record):
        os.killpg(record["pid"], signal.SIGKILL)
    return True


def ready(port, node_id, launch_id=None):
    try:
        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/readyz", timeout=1) as response:
            value = json.load(response)
        return (value.get("service") == "inference-ops" and value.get("status") == "ready"
                and value.get("runtime_node_id") == node_id and value.get("role") in {"worker", "both"}
                and (launch_id is None or value.get("runtime_launch_id") == launch_id))
    except (OSError, ValueError):
        return False


def missing_setting(value):
    if value is None or isinstance(value, (bool, dict, list)):
        return True
    text = str(value).strip()
    return not text or text.startswith("replace-") or text == "example-model-name" or "<" in text or ">" in text


def validate_agent_configuration(config, environment, config_path):
    problems = []
    for section in ("MEMORY", "ADMIN", "RESOURCE_POOL"):
        if section in config and not isinstance(config[section], dict):
            problems.append(f"{section} 必须是配置对象")
    role = environment["AGENT_ROLE"].strip().lower()
    if role not in {"worker", "both"}:
        problems.append("ROLE/AGENT_ROLE 必须为 worker 或 both；独立 controller 请设置 START_LOCAL_INFERENCE=false")
    for env_key, config_key in (("AGENT_LLM_URL", "LLM_URL"), ("AGENT_LLM_MODEL", "LLM_MODEL")):
        value = environment.get(env_key, config.get(config_key))
        if missing_setting(value):
            problems.append(f"{config_key}/{env_key} 未配置或仍是示例占位符；可在 Runtime 配置 MODEL_BASE_URL、MODEL_NAME 统一复用")
        elif config_key == "LLM_URL":
            try:
                parsed = urlsplit(str(value))
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(char.isspace() for char in str(value)):
                    raise ValueError()
                parsed.port
            except ValueError:
                problems.append("LLM_URL/AGENT_LLM_URL 必须为有效的 http/https 模型服务地址")
    key = environment.get("AGENT_LLM_API_KEY", config.get("LLM_API_KEY", "EMPTY"))
    if missing_setting(key):
        problems.append("LLM_API_KEY/AGENT_LLM_API_KEY 仍是占位符；无需鉴权时可填写 EMPTY")
    memory = config.get("MEMORY") if isinstance(config.get("MEMORY"), dict) else {}
    backend = str(environment.get("INFERENCE_OPS_MEMORY_BACKEND", environment.get("MEDFLOW_AGENT_MEMORY_BACKEND", memory.get("BACKEND", "memory")))).strip().lower()
    if backend not in {"", "memory", "inmemory", "in_memory", "redis"}:
        problems.append("MEMORY.BACKEND 必须为 memory 或 redis")
    if problems:
        raise ValueError(f"推理 Agent 配置检查不通过（推理容器内 {config_path}）：\n- " + "\n- ".join(problems)
                         + "\n修正后重新执行 bash runtime.sh start；尚未启动推理进程，也未改写 agent.yaml。")


def prepare_environment(root, incoming, defaults, port):
    from importlib import import_module
    sys.path.insert(0, str(root))
    environment = os.environ.copy()
    # Local inference .env settings are read inside this container. Runtime
    # credentials then override stale copies, without persisting them to disk.
    import_module("inference_ops.environment").load_runtime_environment(environment)
    environment.update(incoming)
    for key in ("MEDFLOW_STUDIO_NODE_TOKEN", "MEDFLOW_RUNTIME_NODE_TOKEN"):
        environment[key] = incoming["MEDFLOW_RUNTIME_NODE_TOKEN"]
    import yaml
    config_path = Path(environment.get("INFERENCE_OPS_CONFIG") or
                       Path(environment.get("INFERENCE_OPS_CONFIG_DIR") or root / "config") / "agent.yaml")
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ValueError(f"推理容器内缺少 {config_path}；可创建空 agent.yaml 以复用 Runtime 配置，或填写独立推理配置")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f"，第 {mark.line + 1} 行" if mark is not None else ""
        raise ValueError(f"无法读取或解析推理容器内 {config_path}{location}；请检查 YAML 格式") from None
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"推理容器内 {config_path} 顶层必须为配置对象，可留空以复用 Runtime 配置")
    environment["INFERENCE_OPS_CONFIG"] = str(config_path)
    llm_pair = (("AGENT_LLM_URL", "LLM_URL"), ("AGENT_LLM_MODEL", "LLM_MODEL"))
    # An unfinished example is one configuration, not half of two services.
    # Reuse URL, model and key together; explicit inference overrides are kept.
    if (not any(key in environment for key, _ in llm_pair)
            and any(missing_setting(config.get(field)) for _, field in llm_pair)
            and all(not missing_setting(defaults.get(key)) for key, _ in llm_pair)):
        for key, _ in llm_pair:
            environment[key] = defaults[key]
        environment.setdefault("AGENT_LLM_API_KEY", defaults.get("AGENT_LLM_API_KEY") or "EMPTY")
        environment["MEDFLOW_INFERENCE_LLM_SOURCE"] = "runtime"
    else:
        environment["MEDFLOW_INFERENCE_LLM_SOURCE"] = "inference"
    environment.setdefault("AGENT_ROLE", str(config.get("ROLE") or "both"))
    environment.setdefault("INFERENCE_AGENT_HOST", "0.0.0.0")
    environment["INFERENCE_AGENT_PORT"] = str(port)
    environment.setdefault("MEDFLOW_RESOURCE_POOL_ENABLED", "true")
    validate_agent_configuration(config, environment, config_path)
    return environment


def run(payload):
    import fcntl
    operation = payload["operation"]
    root = locate_ops(payload["candidates"])
    state = Path(os.getenv("INFERENCE_OPS_STATE_DIR") or root / "runtime")
    state.mkdir(parents=True, exist_ok=True)
    record_path = state / "runtime-controller.json"
    log = state / "controller.log"
    with (state / "runtime-controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = read_record(record_path)
        owned = owned_process(record)
        base = {"directory": str(root), "log": str(log), "pid": record.get("pid") if owned else None}
        if operation == "stop":
            stopped = stop_owned(record)
            record_path.unlink(missing_ok=True)
            return {**base, "status": "stopped" if stopped else "not_managed"}
        if operation == "logs":
            return {**base, "status": "ok", "text": "\n".join(log.read_text(errors="replace").splitlines()[-100:]) if log.is_file() else "暂无推理控制进程日志"}
        port = int(payload["port"])
        node_id = payload["environment"]["MEDFLOW_RESOURCE_NODE_ID"]
        if operation == "status":
            return {**base, "status": "ready" if ready(port, node_id, record.get("launch_id") if owned else None) else ("starting" if owned else "stopped"), "managed": owned}
        environment = prepare_environment(root, payload["environment"], payload.get("llm_defaults", {}), port)
        base["configuration"] = {"agent_config": environment.get("INFERENCE_OPS_CONFIG"),
                                 "llm_source": environment.get("MEDFLOW_INFERENCE_LLM_SOURCE")}
        missing = [name for name in ("uvicorn", "yaml", "fastapi", "psutil", "langchain", "langgraph") if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError("推理容器缺少预装依赖：" + ", ".join(missing))
        if operation == "check":
            return {**base, "status": "configured"}
        if operation != "start":
            raise ValueError("Unknown inference controller operation")
        if owned:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and owned_process(record):
                if ready(port, node_id, record["launch_id"]):
                    return {**base, "status": "ready", "managed": True}
                time.sleep(0.25)
            raise ValueError(f"已有推理控制进程未就绪，请查看推理容器内日志 {log}；可执行 runtime.sh restart 重试")
        # An existing manually managed service can be reused when it reports
        # the same node identity. Never adopt or signal its PID.
        if ready(port, node_id):
            return {**base, "status": "ready", "managed": False}
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError:
                raise ValueError(f"推理端口 {port} 已被其他进程占用；未终止该进程，请先停止原手工控制进程再重试") from None
        import uuid
        launch_id = uuid.uuid4().hex
        environment["MEDFLOW_INFERENCE_LAUNCH_ID"] = launch_id
        with log.open("ab") as stream:
            child = subprocess.Popen([sys.executable, "-m", "inference_ops"], cwd=root, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": child.pid, "identity": process_identity(child.pid), "launch_id": launch_id}
        temporary = record_path.with_suffix(".tmp")
        try:
            if not record["identity"]:
                raise OSError("无法确认新推理控制进程的启动身份")
            temporary.write_text(json.dumps(record), encoding="utf-8")
            temporary.replace(record_path)
        except OSError:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            temporary.unlink(missing_ok=True)
            raise
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is None and ready(port, node_id, launch_id):
                return {**base, "pid": child.pid, "status": "ready", "managed": True}
            if child.poll() is not None:
                break
            time.sleep(0.25)
        stop_owned(record)
        record_path.unlink(missing_ok=True)
        raise ValueError(f"推理控制进程未就绪，请查看推理容器内日志 {log}")


if __name__ == "__main__":
    try:
        print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
