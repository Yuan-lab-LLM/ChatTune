# -*- coding: utf-8 -*-
"""Deploy MedFlow Agent as an AgentScope Runtime API service."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent

for path in (BASE_DIR,):
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from agentscope_runtime.engine.deployers import LocalDeployManager  # noqa: E402
from agentscope_runtime.engine.deployers.utils.deployment_modes import (  # noqa: E402
    DeploymentMode,
)
from agentscope_runtime.engine.deployers.utils.service_utils import (  # noqa: E402
    ProcessManager,
)


DEFAULT_REQUIREMENTS = [
    "agentscope==2.0.5",
    "agentscope-runtime==1.1.5",
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "PyYAML",
    "psutil",
    "requests",
    "aioitertools",
    "anthropic",
    "dashscope",
    "docstring_parser",
    "json5",
    "json_repair",
    "mcp>=1.13",
    "numpy",
    "openai",
    "python-datauri",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "python-socketio",
    "shortuuid",
    "tiktoken",
    "sounddevice",
]


class CurrentPythonProcessManager(ProcessManager):
    """Start detached Runtime services with the current Python executable."""

    async def start_detached_process(
        self,
        script_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        env: dict | None = None,
    ) -> int:
        try:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            log_dir = Path(os.getenv("AGENTSCOPE_RUNTIME_LOG_DIR", "runtime/logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            temp_log_file = (
                log_dir / f"process_temp_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
            )

            log_f = open(temp_log_file, "w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    script_path,
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=process_env,
                cwd=os.path.dirname(script_path),
            )

            log_file = log_dir / f"process_{process.pid}.log"
            log_f.close()
            temp_log_file.replace(log_file)
            self._log_file = str(log_file)
            self._log_file_handle = open(log_file, "a", encoding="utf-8")

            await asyncio.sleep(0.5)
            if process.poll() is not None:
                await asyncio.sleep(0.2)
                raise RuntimeError(
                    "Process failed to start. "
                    f"Check logs at {self._log_file}.",
                )
            return process.pid
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start detached process with {sys.executable}: {exc}",
            ) from exc


def _runtime_environment(
    host: str,
    port: int,
    *,
    model_name: str | None = None,
    model_api_key: str | None = None,
    model_base_url: str | None = None,
    default_docker_container: str | None = None,
    keep_think_context: bool | None = None,
    studio_url: str | None = None,
) -> dict[str, str]:
    pythonpath_parts = [str(BASE_DIR)]
    if os.getenv("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])

    env = {
        "HOST": host,
        "PORT": str(port),
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
    }

    inherited_keys = (
        "MODEL_NAME",
        "MODEL_API_KEY",
        "MODEL_BASE_URL",
        "AGENT3_DEFAULT_DOCKER_CONTAINER",
        "AGENT_KEEP_THINK_CONTEXT",
        "STUDIO_URL",
        "MEDFLOW_RESOURCE_API_TOKEN",
        "MEDFLOW_AGENT_API_TOKEN",
        "MEDFLOW_STUDIO_RUNTIME_TOKEN",
        "MEDFLOW_STUDIO_NODE_TOKEN",
        "MEDFLOW_RESOURCE_NODE_ID",
        "MEDFLOW_RESOURCE_NODE_NAME",

        "MEDFLOW_RUNTIME_PROJECT",
        "MEDFLOW_RUNTIME_RUN_NAME",
        "MEDFLOW_STUDIO_REGISTRATION_TIMEOUT_SECONDS",
        "MEDFLOW_RESOURCE_GROUP_ID",
        "MEDFLOW_TRAINING_POOL_ID",
        "MEDFLOW_MULTINODE_NODE_COUNT",
        "MEDFLOW_MULTINODE_GPUS_PER_NODE",
        "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_LORA",
        "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_FULL",
        "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_PRETRAIN_LORA",
        "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_PRETRAIN_FULL",
        "MEDFLOW_DEFAULT_GPU_COUNT_DPO_TRAIN",
        "MEDFLOW_DEFAULT_GPU_COUNT_GRPO_TRAIN",
        "MEDFLOW_DEFAULT_GPU_COUNT_CKPT_EVAL",
        "MEDFLOW_DEFAULT_GPU_COUNT_COMPARE_MODELS",
        "MEDFLOW_DEFAULT_GPU_COUNT_SINGLE_MODEL_EVAL",
        "MEDFLOW_TRAINING_RESERVATION_HEARTBEAT_SECONDS",
        "MEDFLOW_TRAINING_RESERVATION_MAX_HEARTBEAT_FAILURES",
        "MEDFLOW_STUDIO_RESOURCE_REQUEST_TIMEOUT_SECONDS",
        "MEDFLOW_RESOURCE_GPU_QUERY_TIMEOUT_SECONDS",
        "MEDFLOW_GPU_PREFLIGHT_TIMEOUT_SECONDS",
        "MEDFLOW_GPU_PREFLIGHT_BUSY_MEMORY_MB",
        "MEDFLOW_ASSIGNED_GPUS",
        "MEDFLOW_REQUIRE_GPU_ASSIGNMENT",
        "MULTINODE_DOCKER_CONTAINER",
        "NVIDIA_SMI_PATH",
    )
    for key in inherited_keys:
        if os.getenv(key) is not None:
            env[key] = os.environ[key]

    explicit_env = {
        "MODEL_NAME": model_name,
        "MODEL_API_KEY": model_api_key,
        "MODEL_BASE_URL": model_base_url,
        "AGENT3_DEFAULT_DOCKER_CONTAINER": default_docker_container,
        "STUDIO_URL": studio_url,
    }
    for key, value in explicit_env.items():
        if value is not None:
            env[key] = value

    if keep_think_context is not None:
        env["AGENT_KEEP_THINK_CONTEXT"] = "1" if keep_think_context else "0"

    return env


def _check_port_available(host: str, port: int) -> None:
    """Fail fast when the requested service port is already occupied."""
    bind_host = host
    if bind_host in {"0.0.0.0", "::"}:
        bind_host = "" if bind_host == "0.0.0.0" else "::"

    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is already in use on host {host}. "
                "Use another --port, or stop the process currently listening "
                f"on {port} before deploying."
            ) from exc


def _check_studio_available(studio_url: str) -> None:
    """Fail fast when AgentScope Studio is not reachable yet."""
    parsed = urlparse(studio_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(
            f"Invalid studio URL: {studio_url}. "
            "Expected something like http://localhost:3000.",
        )

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=3):
            return
    except OSError as exc:
        raise RuntimeError(
            f"AgentScope Studio is not reachable at {studio_url}. "
            "Start studio first, or pass the correct --studio-url. "
            "runtime_agent.py registers to Studio during import, so Runtime API "
            "cannot start before Studio Server is listening.",
        ) from exc


async def deploy(
    host: str,
    port: int,
    *,
    model_name: str | None = None,
    model_api_key: str | None = None,
    model_base_url: str | None = None,
    default_docker_container: str | None = None,
    keep_think_context: bool | None = None,
    studio_url: str | None = None,
) -> dict[str, str]:
    _check_port_available(host, port)
    effective_studio_url = studio_url or os.getenv("STUDIO_URL") or "http://localhost:3000"
    _check_studio_available(effective_studio_url)
    deployer = LocalDeployManager(host=host, port=port)
    deployer.process_manager = CurrentPythonProcessManager()
    return await deployer.deploy(
        mode=DeploymentMode.DETACHED_PROCESS,
        entrypoint=str(BASE_DIR / "api_app.py") + ":agent_app",
        requirements=DEFAULT_REQUIREMENTS,
        environment=_runtime_environment(
            host,
            port,
            model_name=model_name,
            model_api_key=model_api_key,
            model_base_url=model_base_url,
            default_docker_container=default_docker_container,
            keep_think_context=keep_think_context,
            studio_url=effective_studio_url,
        ),
        use_local_runtime=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy MedFlow Agent API service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--model-name",
        help="Override MODEL_NAME in agent/config/config.yaml",
    )
    parser.add_argument(
        "--model-api-key",
        help="Override MODEL_API_KEY in agent/config/config.yaml",
    )
    parser.add_argument(
        "--model-base-url",
        help="Override MODEL_BASE_URL in agent/config/config.yaml",
    )
    parser.add_argument(
        "--default-docker-container",
        help="Override AGENT3_DEFAULT_DOCKER_CONTAINER",
    )
    parser.add_argument(
        "--studio-url",
        help=(
            "AgentScope Studio server URL used by runtime_agent.py during "
            "startup. Default: STUDIO_URL env or http://localhost:3000"
        ),
    )
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--keep-think-context",
        action="store_true",
        help="Set AGENT_KEEP_THINK_CONTEXT=1",
    )
    think_group.add_argument(
        "--strip-think-context",
        action="store_true",
        help="Set AGENT_KEEP_THINK_CONTEXT=0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_think_context = None
    if args.keep_think_context:
        keep_think_context = True
    elif args.strip_think_context:
        keep_think_context = False

    result = asyncio.run(
        deploy(
            args.host,
            args.port,
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_base_url=args.model_base_url,
            default_docker_container=args.default_docker_container,
            keep_think_context=keep_think_context,
            studio_url=args.studio_url,
        ),
    )
    print(f"Agent3 API deployed: {result['url']}")
    print(f"Deployment id: {result['deploy_id']}")
    print(f"Process endpoint: {result['url']}/process")
    print(f"OpenAI compatible endpoint: {result['url']}/compatible-mode/v1")


if __name__ == "__main__":
    main()




