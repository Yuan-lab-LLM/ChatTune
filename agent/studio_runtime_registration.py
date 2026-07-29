# -*- coding: utf-8 -*-
"""Register the real Agent Runtime process with MedFlow Studio."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


class StudioRuntimeRegistrationError(RuntimeError):
    """Raised when the Runtime cannot publish its Studio run state."""


@dataclass(frozen=True)
class StudioRuntimeRun:
    """The Studio run identity owned by this Agent Runtime process."""

    run_id: str
    project: str
    name: str
    node_id: str


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _host_port() -> tuple[str, str]:
    return _env("HOST") or "127.0.0.1", _env("PORT") or "8099"


def _studio_url() -> str:
    url = _env("STUDIO_URL").rstrip("/")
    if not url:
        raise StudioRuntimeRegistrationError("STUDIO_URL is required for Studio run registration")
    return url


def _runtime_token() -> str:
    token = _env("MEDFLOW_STUDIO_RUNTIME_TOKEN") or _env("AGENTSCOPE_STUDIO_RUNTIME_TOKEN")
    if not token:
        raise StudioRuntimeRegistrationError(
            "MEDFLOW_STUDIO_RUNTIME_TOKEN is required for Studio run registration"
        )
    return token


def studio_runtime_headers() -> dict[str, str]:
    """Headers accepted by Studio runtime-only tRPC endpoints."""
    return {"X-Medflow-Runtime-Token": _runtime_token()}


def _trpc_result_data(payload: Any) -> Any:
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict) and "data" in result:
            return result["data"]
    return payload


def _trpc_error_message(payload: Any) -> str:
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        error_json = error.get("json")
        if isinstance(error_json, dict) and error_json.get("message"):
            return str(error_json["message"])
        if error.get("message"):
            return str(error["message"])
    return str(payload.get("message") or "")


def _post_studio_runtime(procedure: str, data: dict[str, Any]) -> Any:
    url = f"{_studio_url()}/trpc/{procedure}"
    timeout = max(1, int(_env("MEDFLOW_STUDIO_REGISTRATION_TIMEOUT_SECONDS") or "10"))
    try:
        response = requests.post(
            url,
            json=data,
            headers=studio_runtime_headers(),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise StudioRuntimeRegistrationError(
            f"Studio runtime request failed: {url}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not response.ok:
        message = _trpc_error_message(payload) or response.text
        raise StudioRuntimeRegistrationError(
            f"Studio runtime request failed: {url}: HTTP {response.status_code}: {message}"
        )
    result = _trpc_result_data(payload)
    if isinstance(result, dict) and result.get("success") is False:
        raise StudioRuntimeRegistrationError(
            str(result.get("message") or f"Studio {procedure} failed")
        )
    return result


def _generated_runtime_run_id(node_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"runtime-{node_id}-{timestamp}-{uuid.uuid4().hex[:8]}"


def runtime_run_identity(base_dir: Path) -> StudioRuntimeRun:
    """Build the Studio run identity for this concrete Runtime process."""
    node_id = _env("MEDFLOW_RESOURCE_NODE_ID")
    if not node_id:
        raise StudioRuntimeRegistrationError(
            "MEDFLOW_RESOURCE_NODE_ID is required for Studio run registration"
        )
    host, port = _host_port()
    node_name = _env("MEDFLOW_RESOURCE_NODE_NAME") or node_id
    return StudioRuntimeRun(
        run_id=_generated_runtime_run_id(node_id),
        project=_env("MEDFLOW_RUNTIME_PROJECT") or "MedFlow_Runtime",
        name=_env("MEDFLOW_RUNTIME_RUN_NAME") or f"{node_name} ({host}:{port})",
        node_id=node_id,
    )


def register_runtime_run(base_dir: Path) -> StudioRuntimeRun:
    """Register this live Agent Runtime process as a Studio RUNNING run."""
    run = runtime_run_identity(base_dir)
    _post_studio_runtime(
        "registerRun",
        {
            "id": run.run_id,
            "project": run.project,
            "name": run.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid(),
            "status": "running",
            "nodeId": run.node_id,
            "run_dir": str(base_dir),
        },
    )
    return run


def update_runtime_run_status(run_id: str, status: str) -> None:
    """Update the Studio status for this Runtime run."""
    _post_studio_runtime("updateRunStatus", {"runId": run_id, "status": status})
