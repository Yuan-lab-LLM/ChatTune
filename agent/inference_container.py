"""Manage the inference controller through Docker, without sharing mounts."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
from urllib.parse import urlsplit


REMOTE_SOURCE = Path(__file__).with_name("inference_container_runtime.py")
ENV_KEYS = ("STUDIO_URL", "MEDFLOW_RESOURCE_NODE_ID", "INFERENCE_OPS_AUTH_MODE", "INFERENCE_OPS_API_TOKEN",
            "AGENT_LLM_URL", "AGENT_LLM_MODEL", "AGENT_LLM_API_KEY")


def local_inference_settings(environ):
    enabled = environ.get("START_LOCAL_INFERENCE", "false").strip().lower()
    if enabled not in {"auto", "true", "false"}:
        raise ValueError("START_LOCAL_INFERENCE must be auto, true, or false")
    if enabled == "false":
        return None
    url = urlsplit(environ.get("INFERENCE_AGENT_URL") or "http://127.0.0.1:8899/inference_agent")
    container = (environ.get("MEDFLOW_INFERENCE_CONTAINER") or environ.get("MEDFLOW_LOCAL_EVALUATE_CONTAINER")
                 or environ.get("AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER") or "").strip()
    local_hosts = {"localhost", "127.0.0.1", "::1", socket.gethostname().lower(), container.lower()}
    for name in ("IEI_VISIBLE_OS_IP", "MEDFLOW_RESOURCE_NODE_HOST"):
        if environ.get(name):
            local_hosts.add(environ[name].lower())
    try:
        local_hosts.update(address[4][0].lower() for address in socket.getaddrinfo(socket.gethostname(), None))
    except OSError:
        pass
    if enabled == "auto" and (url.scheme != "http" or str(url.hostname).lower() not in local_hosts or not container):
        return None
    if not container:
        raise ValueError("请设置推理容器名 MEDFLOW_LOCAL_EVALUATE_CONTAINER 或 MEDFLOW_INFERENCE_CONTAINER")
    if url.scheme != "http" or not url.hostname:
        raise ValueError("本地自动管理需要 http INFERENCE_AGENT_URL；外部推理服务请设置 START_LOCAL_INFERENCE=false")
    return {"container": container, "python": environ.get("MEDFLOW_INFERENCE_PYTHON") or "python3", "port": url.port or 80}


def container_candidates(inspection, override=""):
    if override:
        return [override]
    roots = [inspection.get("Config", {}).get("WorkingDir") or "/"]
    roots.extend(mount["Destination"] for mount in inspection.get("Mounts", []) if mount.get("Destination"))
    candidates = []
    for root in roots + ["/home/workspace", "/workspace", "/app"]:
        candidates.extend([str(PurePosixPath(root) / "inference-ops"), root])
    return list(dict.fromkeys(candidates))


def deployment_environment(environ):
    values = {key: environ[key] for key in ENV_KEYS if environ.get(key)}
    token = (environ.get("MEDFLOW_RUNTIME_NODE_TOKEN") or environ.get("MEDFLOW_STUDIO_NODE_TOKEN") or "").strip()
    legacy = environ.get("MEDFLOW_STUDIO_NODE_TOKEN", "").strip()
    if legacy and token != legacy:
        raise ValueError("MEDFLOW_RUNTIME_NODE_TOKEN 与 MEDFLOW_STUDIO_NODE_TOKEN 必须一致")
    for key, value in (("STUDIO_URL", values.get("STUDIO_URL")), ("MEDFLOW_RESOURCE_NODE_ID", values.get("MEDFLOW_RESOURCE_NODE_ID")),
                       ("MEDFLOW_RUNTIME_NODE_TOKEN", token)):
        if not value or value.startswith("replace-") or "<" in value:
            raise ValueError(f"Runtime 缺少有效配置 {key}，请检查现有 runtime.env")
    values["MEDFLOW_RUNTIME_NODE_TOKEN"] = token
    return values


def execute(operation, settings, environ, *, runner=subprocess.run):
    inspected = runner(["docker", "inspect", settings["container"]], capture_output=True, text=True, timeout=10)
    if inspected.returncode:
        raise ValueError(f"无法读取推理容器 {settings['container']}；请检查容器是否已创建且 Docker 可访问")
    inspection = json.loads(inspected.stdout)[0]
    candidates = settings.get("candidates") or container_candidates(inspection, environ.get("MEDFLOW_INFERENCE_OPS_DIR", ""))
    if not inspection.get("State", {}).get("Running"):
        if operation in {"stop", "status"}:
            return {"status": "stopped", "directory": candidates[0]}, settings
        raise ValueError(f"推理容器 {settings['container']} 未运行，请先启动该容器")
    environment = deployment_environment(environ) if operation in {"start", "check"} else {"MEDFLOW_RESOURCE_NODE_ID": environ.get("MEDFLOW_RESOURCE_NODE_ID", "")}
    defaults = {key: environ[source] for key, source in (("AGENT_LLM_URL", "MODEL_BASE_URL"),
                ("AGENT_LLM_MODEL", "MODEL_NAME"), ("AGENT_LLM_API_KEY", "MODEL_API_KEY")) if environ.get(source)}
    payload = {"operation": operation, "candidates": candidates, "port": settings["port"], "environment": environment,
               "llm_defaults": defaults if operation in {"start", "check"} else {}}
    result = runner(["docker", "exec", "-i", settings["container"], settings["python"], "-c", REMOTE_SOURCE.read_text(encoding="utf-8")],
                    input=json.dumps(payload), capture_output=True, text=True, timeout=50)
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise ValueError(f"推理容器未能运行启动助手，请检查容器内 Python（当前 {settings['python']}）") from None
    if result.returncode or response.get("status") == "error":
        raise ValueError(f"推理容器 {settings['container']}：" + str(response.get("error") or "推理容器操作失败"))
    return response, {**settings, "candidates": [response["directory"]]}


def manage(operation, state_file, environ=None, *, runner=subprocess.run):
    environ = os.environ if environ is None else environ
    if environ.get("START_LOCAL_INFERENCE", "false").strip().lower() == "false":
        return {"status": "external"}
    state_file = Path(state_file)
    if operation == "stop" and not state_file.is_file():
        return {"status": "not_managed"}
    if operation in {"stop", "status", "logs"} and state_file.is_file():
        settings = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        settings = local_inference_settings(environ)
    if settings is None:
        return {"status": "external"}
    response, settings = execute(operation, settings, environ, runner=runner)
    if operation == "start":
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings), encoding="utf-8")
        temporary.replace(state_file)
    elif operation == "stop":
        state_file.unlink(missing_ok=True)
    return {**response, "container": settings["container"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["start", "stop", "status", "check", "logs"])
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    try:
        response = manage(args.operation, args.state_file)
        if args.operation == "logs" and response.get("text"):
            print(response["text"])
        else:
            details = f" (container={response['container']})" if response.get("container") else ""
            print(f"inference: {response['status']}{details}")
            if (response.get("configuration") or {}).get("llm_source") == "runtime":
                print("inference: LLM 地址和模型已成套复用 Runtime 配置，显式推理凭证保留，agent.yaml 保持不变。")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"inference: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
