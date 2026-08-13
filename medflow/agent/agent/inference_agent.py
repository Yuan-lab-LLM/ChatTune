import argparse
import concurrent.futures
import json
import operator
import os
import re
import time
import traceback
from typing import Any, Literal, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from admin_manager import (
    admin_apply_benchmark_stop,
    admin_apply_cleanup,
    admin_apply_service_stop,
    admin_apply_test_stop,
    admin_list_benchmark_jobs,
    admin_list_service_instances,
    admin_list_test_runs,
    admin_preview_benchmark_stop,
    admin_preview_cleanup,
    admin_preview_service_stop,
    admin_preview_test_stop,
)
from tools import (
    benchmark_inspect,
    benchmark_jobs,
    benchmark_list,
    benchmark_report,
    benchmark_run,
    benchmark_stop,
    benchmark_stop_preview,
    config_check,
    config_keys,
    config_show,
    config_update,
    current_request_thread_id,
    current_request_user_aliases,
    current_request_user_id,
    current_request_user_role,
    gpu_recommend_allocation,
    gpu_status,
    model_list,
    node_benchmark_inspect,
    node_benchmark_jobs,
    node_benchmark_list,
    node_benchmark_report,
    node_benchmark_run,
    node_benchmark_stop,
    node_benchmark_stop_preview,
    node_config_check,
    node_config_keys,
    node_config_show,
    node_config_update,
    node_disable,
    node_enable,
    node_gpu_recommend_allocation,
    node_gpu_status,
    node_list,
    node_model_list,
    node_port_status,
    node_recommend_start_target,
    node_service_instance_list,
    node_service_instance_status,
    node_service_instance_stop,
    node_service_instance_stop_preview,
    node_service_instance_tasks,
    node_service_log_context,
    node_service_log_runs,
    node_service_log_search,
    node_service_log_tail,
    node_service_restart,
    node_service_start,
    node_service_start_status,
    node_service_status,
    node_service_stop,
    node_service_test_list,
    node_service_test_run,
    node_service_test_run_all,
    node_service_test_status,
    node_service_test_stop_preview,
    node_service_test_stop,
    port_status,
    reset_current_request_thread_id,
    reset_current_request_resource_context,
    reset_current_request_user_aliases,
    reset_current_request_user_id,
    reset_current_request_user_role,
    running_benchmark_jobs_text,
    service_instance_list,
    service_instance_status,
    service_instance_stop,
    service_instance_stop_preview,
    service_instance_tasks,
    service_log_context,
    service_log_runs,
    service_log_search,
    service_log_tail,
    service_restart,
    service_start,
    service_start_status,
    service_status,
    service_stop,
    service_stop_by_reservation,
    service_stop_by_reservation_result,
    service_test_list,
    service_test_run,
    service_test_run_all,
    service_test_status,
    service_test_stop_preview,
    service_test_stop,
    set_current_request_thread_id,
    set_current_request_resource_context,
    set_current_request_user_aliases,
    set_current_request_user_id,
    set_current_request_user_role,
    sync_worker_service_host_ip,
)

from typing_extensions import Annotated, TypedDict

app = FastAPI()

def patch_openai_reasoning_passthrough() -> None:
    """Keep vLLM/OpenAI-compatible message.reasoning on LangChain AIMessage."""
    try:
        import langchain_openai.chat_models.base as openai_chat_base
    except Exception:
        return

    original = getattr(openai_chat_base, "_convert_dict_to_message", None)
    if original is None or getattr(original, "_medflow_reasoning_patch", False):
        return

    def _convert_dict_to_message_with_reasoning(message_dict):
        message = original(message_dict)
        if isinstance(message, AIMessage):
            reasoning = message_dict.get("reasoning")
            if reasoning is None:
                reasoning = message_dict.get("reasoning_content")
            if reasoning is not None:
                message.additional_kwargs["reasoning"] = reasoning
        return message

    _convert_dict_to_message_with_reasoning._medflow_reasoning_patch = True
    openai_chat_base._convert_dict_to_message = _convert_dict_to_message_with_reasoning

patch_openai_reasoning_passthrough()

class InferenceRequest(BaseModel):
    command: str
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_aliases: Optional[list[str]] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    include_trace: bool = False
    resource_context: Optional[dict[str, Any]] = None

class ToolInvokeRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_aliases: Optional[list[str]] = None
    thread_id: Optional[str] = None
    resource_context: Optional[dict[str, Any]] = None

AGENT_CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../config/agent.yaml")
)

def load_agent_config() -> dict:
    if not os.path.exists(AGENT_CONFIG_FILE):
        return {}
    with open(AGENT_CONFIG_FILE) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data

AGENT_CONFIG = load_agent_config()
AGENT_ROLE = os.getenv("AGENT_ROLE", AGENT_CONFIG.get("ROLE", "worker")).strip().lower()
ADMIN_CONFIG = (
    AGENT_CONFIG.get("ADMIN", {})
    if isinstance(AGENT_CONFIG.get("ADMIN"), dict)
    else {}
)
ADMIN_ENABLED = str(ADMIN_CONFIG.get("ENABLED", False)).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VLLM_URL = os.getenv(
    "AGENT_LLM_URL", AGENT_CONFIG.get("LLM_URL", "http://127.0.0.1:8111/v1")
)
LLM_MODEL = os.getenv(
    "AGENT_LLM_MODEL", AGENT_CONFIG.get("LLM_MODEL", "example-model-name")
)
LLM_API_KEY = os.getenv("AGENT_LLM_API_KEY", AGENT_CONFIG.get("LLM_API_KEY", "EMPTY"))
LLM_TIMEOUT_SECONDS = int(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "300"))
LLM_MAX_RETRIES = int(os.getenv("AGENT_LLM_MAX_RETRIES", "1"))
LLM_MAX_COMPLETION_TOKENS = int(
    os.getenv("AGENT_LLM_MAX_COMPLETION_TOKENS", "4096")
)
INFERENCE_AGENT_HOST = os.getenv(
    "INFERENCE_AGENT_HOST", AGENT_CONFIG.get("HOST", "127.0.0.1")
)
INFERENCE_AGENT_PORT = int(
    os.getenv("INFERENCE_AGENT_PORT", str(AGENT_CONFIG.get("PORT", 8899)))
)
# AGENT_MAX_TOKENS = 8192

def initialize_worker_node_config() -> None:
    if AGENT_ROLE not in {"worker", "both"}:
        return
    try:
        result = sync_worker_service_host_ip()
    except Exception as exc:
        print(
            "[config] failed to synchronize service.yaml HOST_IP: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return
    if result["changed"]:
        print(
            "[config] service.yaml HOST_IP synchronized "
            f"{result['configured_ip'] or 'empty'} -> {result['worker_ip']}",
            flush=True,
        )
    if ADMIN_ENABLED:
        print(
            "[admin] ADMIN.ENABLED=true; /admin/* maintenance endpoints are "
            "enabled and should only be reachable from Studio Server or a "
            "trusted internal network. /inference_agent and /worker/tool are "
            "unaffected.",
            flush=True,
        )

@app.on_event("startup")
def initialize_api_worker_node_config() -> None:
    initialize_worker_node_config()

def require_admin_access() -> None:
    if AGENT_ROLE not in {"worker", "both"} or not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Admin endpoints are disabled")
def require_admin_tool_access() -> None:
    if current_request_user_role() != "admin":
        raise PermissionError("当前请求不是管理员角色，已拒绝执行管理员维护工具。")
    if not ADMIN_ENABLED:
        raise PermissionError("管理员维护能力未启用，已拒绝执行管理员维护工具。")
    if AGENT_ROLE not in {"worker", "both"}:
        raise PermissionError("当前进程不是 worker/both 角色，已拒绝执行本地管理员维护工具。")
@app.get("/admin/cleanup/preview")
def preview_admin_cleanup():
    require_admin_access()
    return admin_preview_cleanup()

@app.post("/admin/cleanup/apply")
def apply_admin_cleanup():
    require_admin_access()
    return admin_apply_cleanup()

@app.get("/admin/services")
def run_admin_service_list(
    limit: int = 20,
    status: str = "",
):
    require_admin_access()
    return admin_list_service_instances(limit=limit, status=status)

@app.get("/admin/services/{instance_id}/stop/preview")
def preview_admin_service_stop(
    instance_id: str,
):
    require_admin_access()
    return admin_preview_service_stop(instance_id)

@app.post("/admin/services/{instance_id}/stop/apply")
def apply_admin_service_stop(
    instance_id: str,
):
    require_admin_access()
    return admin_apply_service_stop(instance_id)

@app.get("/admin/benchmarks")
def run_admin_benchmark_list(
    limit: int = 20,
    status: str = "",
    instance_id: str = "",
):
    require_admin_access()
    return admin_list_benchmark_jobs(
        limit=limit, status=status, instance_id=instance_id
    )

@app.get("/admin/benchmarks/{job_id}/stop/preview")
def preview_admin_benchmark_stop(
    job_id: str,
):
    require_admin_access()
    return admin_preview_benchmark_stop(job_id)

@app.post("/admin/benchmarks/{job_id}/stop/apply")
def apply_admin_benchmark_stop(
    job_id: str,
):
    require_admin_access()
    return admin_apply_benchmark_stop(job_id)

@app.get("/admin/tests")
def run_admin_test_list(
    limit: int = 20,
    status: str = "",
    instance_id: str = "",
):
    require_admin_access()
    return admin_list_test_runs(limit=limit, status=status, instance_id=instance_id)

@app.get("/admin/tests/{test_run_id}/stop/preview")
def preview_admin_test_stop(
    test_run_id: str,
):
    require_admin_access()
    return admin_preview_test_stop(test_run_id)

@app.post("/admin/tests/{test_run_id}/stop/apply")
def apply_admin_test_stop(
    test_run_id: str,
):
    require_admin_access()
    return admin_apply_test_stop(test_run_id)
def admin_tool_parameter_error(name: str) -> dict:
    return {
        "status": "error",
        "operation": "parameter_error",
        "error": f"缺少必填参数: {name}",
    }

@tool
def admin_cleanup_preview() -> dict:
    """Preview local stale inference resources that an admin may clean up."""
    require_admin_tool_access()
    return admin_preview_cleanup()

@tool
def admin_cleanup_apply() -> dict:
    """Apply local stale inference resource cleanup after explicit admin confirmation."""
    require_admin_tool_access()
    return admin_apply_cleanup()

@tool
def admin_service_list(limit: int = 20, status: str = "") -> dict:
    """List local inference service instances visible to an admin."""
    require_admin_tool_access()
    return admin_list_service_instances(limit=limit, status=status)

@tool
def admin_service_stop_preview(instance_id: str = "") -> dict:
    """Preview stopping a local inference service instance as an admin."""
    require_admin_tool_access()
    instance_id = str(instance_id or "").strip()
    if not instance_id:
        return admin_tool_parameter_error("instance_id")
    return admin_preview_service_stop(instance_id)

@tool
def admin_service_stop_apply(instance_id: str = "") -> dict:
    """Stop a local inference service instance after explicit admin confirmation."""
    require_admin_tool_access()
    instance_id = str(instance_id or "").strip()
    if not instance_id:
        return admin_tool_parameter_error("instance_id")
    return admin_apply_service_stop(instance_id)

@tool
def admin_benchmark_list(
    limit: int = 20, status: str = "", instance_id: str = ""
) -> dict:
    """List local benchmark jobs visible to an admin."""
    require_admin_tool_access()
    return admin_list_benchmark_jobs(
        limit=limit, status=status, instance_id=instance_id
    )

@tool
def admin_benchmark_stop_preview(job_id: str = "") -> dict:
    """Preview stopping a local benchmark job as an admin."""
    require_admin_tool_access()
    job_id = str(job_id or "").strip()
    if not job_id:
        return admin_tool_parameter_error("job_id")
    return admin_preview_benchmark_stop(job_id)

@tool
def admin_benchmark_stop_apply(job_id: str = "") -> dict:
    """Stop a local benchmark job after explicit admin confirmation."""
    require_admin_tool_access()
    job_id = str(job_id or "").strip()
    if not job_id:
        return admin_tool_parameter_error("job_id")
    return admin_apply_benchmark_stop(job_id)

@tool
def admin_test_list(limit: int = 20, status: str = "", instance_id: str = "") -> dict:
    """List local test runs visible to an admin."""
    require_admin_tool_access()
    return admin_list_test_runs(limit=limit, status=status, instance_id=instance_id)

@tool
def admin_test_stop_preview(test_run_id: str = "") -> dict:
    """Preview stopping a local test run as an admin."""
    require_admin_tool_access()
    test_run_id = str(test_run_id or "").strip()
    if not test_run_id:
        return admin_tool_parameter_error("test_run_id")
    return admin_preview_test_stop(test_run_id)

@tool
def admin_test_stop_apply(test_run_id: str = "") -> dict:
    """Stop a local test run after explicit admin confirmation."""
    require_admin_tool_access()
    test_run_id = str(test_run_id or "").strip()
    if not test_run_id:
        return admin_tool_parameter_error("test_run_id")
    return admin_apply_test_stop(test_run_id)
os.environ["OPENAI_API_KEY"] = str(LLM_API_KEY)

llm = init_chat_model(
    model=str(LLM_MODEL),
    model_provider="openai",
    api_key=str(LLM_API_KEY),
    base_url=str(VLLM_URL),
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=LLM_MAX_RETRIES,
    max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
)
# max_tokens=AGENT_MAX_TOKENS,

worker_tools = [
    service_status,
    port_status,
    gpu_status,
    service_start,
    service_start_status,
    service_stop,
    service_stop_by_reservation,
    service_instance_list,
    service_instance_status,
    service_instance_tasks,
    service_instance_stop_preview,
    service_instance_stop,
    service_restart,
    config_show,
    config_update,
    config_keys,
    # config_restore,
    model_list,
    config_check,
    gpu_recommend_allocation,
    service_log_runs,
    service_log_tail,
    service_log_search,
    service_log_context,
    service_test_list,
    service_test_run,
    service_test_status,
    service_test_stop_preview,
    service_test_stop,
    service_test_run_all,
    benchmark_list,
    benchmark_inspect,
    benchmark_run,
    benchmark_report,
    benchmark_jobs,
    benchmark_stop_preview,
    benchmark_stop,
]

admin_tools = [
    admin_cleanup_preview,
    admin_cleanup_apply,
    admin_service_list,
    admin_service_stop_preview,
    admin_service_stop_apply,
    admin_benchmark_list,
    admin_benchmark_stop_preview,
    admin_benchmark_stop_apply,
    admin_test_list,
    admin_test_stop_preview,
    admin_test_stop_apply,
]

worker_tools = worker_tools + admin_tools

controller_tools = [
    node_list,
    node_enable,
    node_disable,
    node_service_status,
    node_gpu_status,
    node_config_show,
    node_config_keys,
    node_config_update,
    node_recommend_start_target,
    node_service_start,
    node_service_start_status,
    node_service_stop,
    node_service_instance_list,
    node_service_instance_status,
    node_service_instance_tasks,
    node_service_instance_stop_preview,
    node_service_instance_stop,
    node_service_restart,
    node_port_status,
    node_model_list,
    node_config_check,
    node_gpu_recommend_allocation,
    node_service_log_runs,
    node_service_log_tail,
    node_service_log_search,
    node_service_log_context,
    node_service_test_list,
    node_service_test_run,
    node_service_test_run_all,
    node_service_test_status,
    node_service_test_stop_preview,
    node_service_test_stop,
    node_benchmark_list,
    node_benchmark_inspect,
    node_benchmark_run,
    node_benchmark_report,
    node_benchmark_jobs,
    node_benchmark_stop_preview,
    node_benchmark_stop,
    #node_tool_call,
]

if AGENT_ROLE == "controller":
    tools = controller_tools
elif AGENT_ROLE == "both":
    tools = controller_tools + admin_tools
else:
    tools = worker_tools

ADMIN_TOOL_NAMES = {tool.name for tool in admin_tools}
ADMIN_APPLY_TOOL_NAMES = {name for name in ADMIN_TOOL_NAMES if name.endswith("_apply")}
CONFIRMATION_REQUIRED_APPLY_TOOL_NAMES = {
    "benchmark_stop",
    "node_benchmark_stop",
}
CONFIRMATION_REQUIRED_NODE_TOOL_NAMES = {
    "node_benchmark_stop",
}
SCOPE_ALL_TOOL_NAMES = {
    "service_instance_list",
    "service_test_status",
    "benchmark_jobs",
    "benchmark_report",
    "node_service_instance_list",
    "node_service_test_status",
    "node_benchmark_jobs",
    "node_benchmark_report",
}
worker_tools_by_name = {tool.name: tool for tool in worker_tools}
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

MAX_LLM_CALLS = 30
MAX_TOOL_CALLS = 20
TOOL_RESULT_LOG_CHARS = 2000

AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "900"))
AGENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)

MEMORY_CONFIG = AGENT_CONFIG.get("MEMORY", {}) if isinstance(AGENT_CONFIG.get("MEMORY"), dict) else {}
MEMORY_BACKEND = os.getenv(
    "MEDFLOW_AGENT_MEMORY_BACKEND",
    str(MEMORY_CONFIG.get("BACKEND", "memory")),
).strip().lower()
REDIS_URL = os.getenv(
    "MEDFLOW_AGENT_REDIS_URL",
    str(MEMORY_CONFIG.get("REDIS_URL", "redis://127.0.0.1:6379/0")),
)
REDIS_USERNAME = os.getenv(
    "MEDFLOW_AGENT_REDIS_USERNAME",
    str(MEMORY_CONFIG.get("REDIS_USERNAME", "")),
).strip()
REDIS_PASSWORD = os.getenv(
    "MEDFLOW_AGENT_REDIS_PASSWORD",
    str(MEMORY_CONFIG.get("REDIS_PASSWORD", "")),
)
REDIS_PASSWORD_MIN_LENGTH = int(
    os.getenv(
        "MEDFLOW_AGENT_REDIS_PASSWORD_MIN_LENGTH",
        str(MEMORY_CONFIG.get("REDIS_PASSWORD_MIN_LENGTH", 12)),
    )
)
REDIS_PASSWORD_RESERVED_CHARS = set("@:/?#")
REDIS_PASSWORD_SPECIAL_PATTERN = re.compile(r"[^A-Za-z0-9]")
REDIS_TTL_MINUTES = int(
    os.getenv(
        "MEDFLOW_AGENT_REDIS_TTL_MINUTES",
        str(MEMORY_CONFIG.get("TTL_MINUTES", 60 * 24 * 7)),
    )
)
if REDIS_TTL_MINUTES < 0:
    raise ValueError("MEMORY.TTL_MINUTES must be 0 or a positive integer")
REDIS_CHECKPOINT_PREFIX = os.getenv(
    "MEDFLOW_AGENT_REDIS_CHECKPOINT_PREFIX",
    str(MEMORY_CONFIG.get("CHECKPOINT_PREFIX", "medflow_inference_checkpoint")),
)
REDIS_CHECKPOINT_WRITE_PREFIX = os.getenv(
    "MEDFLOW_AGENT_REDIS_CHECKPOINT_WRITE_PREFIX",
    str(
        MEMORY_CONFIG.get(
            "CHECKPOINT_WRITE_PREFIX", "medflow_inference_checkpoint_write"
        )
    ),
)

def validate_redis_password(password: str) -> None:
    problems = []
    if len(password) < REDIS_PASSWORD_MIN_LENGTH:
        problems.append(f"at least {REDIS_PASSWORD_MIN_LENGTH} characters")
    if re.search(r"\s", password):
        problems.append("no whitespace")
    if not re.search(r"[a-z]", password):
        problems.append("one lowercase letter")
    if not re.search(r"[A-Z]", password):
        problems.append("one uppercase letter")
    if not re.search(r"\d", password):
        problems.append("one digit")
    if not REDIS_PASSWORD_SPECIAL_PATTERN.search(password):
        problems.append("one special character")
    reserved = sorted(char for char in REDIS_PASSWORD_RESERVED_CHARS if char in password)
    if reserved:
        problems.append(
            "no raw Redis URL reserved characters: " + "".join(reserved)
        )
    if problems:
        raise ValueError(
            "MEMORY redis password does not meet policy: " + "; ".join(problems)
        )

def build_redis_url() -> str:
    parsed = urlsplit(REDIS_URL or "redis://127.0.0.1:6379/0")
    if parsed.scheme not in {"redis", "rediss"}:
        raise ValueError("MEMORY.REDIS_URL must use redis:// or rediss://")
    username = REDIS_USERNAME or unquote(parsed.username or "")
    password = REDIS_PASSWORD or unquote(parsed.password or "")
    if not username:
        raise ValueError("MEMORY redis backend requires REDIS_USERNAME")
    if not password:
        raise ValueError("MEMORY redis backend requires REDIS_PASSWORD")
    validate_redis_password(password)

    host = parsed.hostname or "127.0.0.1"
    port = f":{parsed.port}" if parsed.port else ""
    auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return urlunsplit((parsed.scheme, f"{auth}{host}{port}", parsed.path or "/0", parsed.query, parsed.fragment))

def redis_url_for_log(redis_url: str) -> str:
    parsed = urlsplit(redis_url)
    username = quote(unquote(parsed.username or ""), safe="")
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    auth = f"{username}:***@" if username else ""
    return urlunsplit((parsed.scheme, f"{auth}{host}{port}", parsed.path, parsed.query, parsed.fragment))

def build_checkpointer():
    if MEMORY_BACKEND in {"", "memory", "inmemory", "in_memory"}:
        print("[memory] backend=memory checkpointer=InMemorySaver", flush=True)
        return InMemorySaver()

    if MEMORY_BACKEND == "redis":
        redis_url = build_redis_url()
        try:
            from langgraph.checkpoint.redis import RedisSaver
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MEMORY.BACKEND=redis requires the LangGraph Redis checkpointer "
                "package/module (langgraph.checkpoint.redis). Install or bundle the "
                "Redis checkpointer dependency, or set MEMORY.BACKEND=memory."
            ) from exc

        redis_options = {
            "redis_url": redis_url,
            "checkpoint_prefix": REDIS_CHECKPOINT_PREFIX,
            "checkpoint_write_prefix": REDIS_CHECKPOINT_WRITE_PREFIX,
        }
        if REDIS_TTL_MINUTES > 0:
            redis_options["ttl"] = {
                "default_ttl": REDIS_TTL_MINUTES,
                "refresh_on_read": True,
            }

        checkpointer = RedisSaver(
            **redis_options,
        )
        checkpointer.setup()
        ttl_text = str(REDIS_TTL_MINUTES) if REDIS_TTL_MINUTES else "disabled"
        print(
            "[memory] backend=redis "
            f"url={redis_url_for_log(redis_url)} ttl_minutes={ttl_text} "
            f"checkpoint_prefix={REDIS_CHECKPOINT_PREFIX}",
            flush=True,
        )
        return checkpointer

    raise ValueError(
        "Unsupported memory backend: "
        f"{MEMORY_BACKEND}. Use memory or redis."
    )
def run_with_timeout(
    fn,
    timeout_seconds: int,
    timeout_message: str,
    executor=None,
):
    future = (executor or AGENT_EXECUTOR).submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(timeout_message) from exc

def empty_response_data() -> dict:
    return {
        "config_draft": None,
        "service_instances": None,
        "service_instance": None,
        "service_start": None,
        "service_stop": None,
        "test_runs": None,
        "test_run_stop": None,
        "admin_cleanup": None,
        "benchmark_stop": None,
        "benchmark_jobs": None,
        "benchmark_reports": [],
        "nodes": {},
    }

def merge_benchmark_reports(left, right) -> list[dict]:
    """Merge current-turn benchmark reports by job_id, keeping call order."""
    reports = []
    indexes = {}
    for group in (left, right):
        if not isinstance(group, list):
            continue
        for report in group:
            if not isinstance(report, dict):
                continue
            item = dict(report)
            job_id = str(item.get("job_id") or "").strip()
            if job_id and job_id in indexes:
                reports[indexes[job_id]] = item
                continue
            if job_id:
                indexes[job_id] = len(reports)
            reports.append(item)
    return reports

def merge_response_data(left: Optional[dict], right: Optional[dict]) -> dict:
    data = empty_response_data()

    if isinstance(left, dict):
        data["config_draft"] = left.get("config_draft")
        data["service_instances"] = left.get("service_instances")
        data["service_instance"] = left.get("service_instance")
        data["service_start"] = left.get("service_start")
        data["service_stop"] = left.get("service_stop")
        data["test_runs"] = left.get("test_runs")
        data["test_run_stop"] = left.get("test_run_stop")
        data["admin_cleanup"] = left.get("admin_cleanup")
        data["benchmark_stop"] = left.get("benchmark_stop")
        data["benchmark_jobs"] = left.get("benchmark_jobs")
        data["benchmark_reports"] = merge_benchmark_reports(
            [], left.get("benchmark_reports")
        )
        if isinstance(left.get("nodes"), dict):
            data["nodes"] = dict(left["nodes"])

    if not isinstance(right, dict):
        return data

    if right.get("service_instance") is not None:
        data["service_instance"] = right["service_instance"]
    if right.get("service_start") is not None:
        data["service_start"] = right["service_start"]
    if right.get("service_stop") is not None:
        data["service_stop"] = right["service_stop"]
    if right.get("config_draft") is not None:
        data["config_draft"] = right["config_draft"]
    if right.get("test_runs") is not None:
        data["test_runs"] = right["test_runs"]
    if right.get("test_run_stop") is not None:
        data["test_run_stop"] = right["test_run_stop"]
    if right.get("admin_cleanup") is not None:
        data["admin_cleanup"] = right["admin_cleanup"]
    if right.get("benchmark_stop") is not None:
        data["benchmark_stop"] = right["benchmark_stop"]
    if right.get("benchmark_jobs") is not None:
        data["benchmark_jobs"] = right["benchmark_jobs"]
    if right.get("service_instances") is not None:
        data["service_instances"] = right["service_instances"]
    data["benchmark_reports"] = merge_benchmark_reports(
        data["benchmark_reports"], right.get("benchmark_reports")
    )

    nodes = right.get("nodes")
    if isinstance(nodes, dict):
        for node, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue
            merged_node = data["nodes"].get(node)
            if not isinstance(merged_node, dict):
                merged_node = {}
            else:
                merged_node = dict(merged_node)
            if node_data.get("config_draft") is not None:
                merged_node["config_draft"] = node_data["config_draft"]
            if node_data.get("service_instances") is not None:
                merged_node["service_instances"] = node_data["service_instances"]
            if node_data.get("service_instance") is not None:
                merged_node["service_instance"] = node_data["service_instance"]
            if node_data.get("service_start") is not None:
                merged_node["service_start"] = node_data["service_start"]
            if node_data.get("service_stop") is not None:
                merged_node["service_stop"] = node_data["service_stop"]
            if node_data.get("test_runs") is not None:
                merged_node["test_runs"] = node_data["test_runs"]
            if node_data.get("test_run_stop") is not None:
                merged_node["test_run_stop"] = node_data["test_run_stop"]
            if node_data.get("admin_cleanup") is not None:
                merged_node["admin_cleanup"] = node_data["admin_cleanup"]
            if node_data.get("benchmark_stop") is not None:
                merged_node["benchmark_stop"] = node_data["benchmark_stop"]
            if node_data.get("benchmark_jobs") is not None:
                merged_node["benchmark_jobs"] = node_data["benchmark_jobs"]
            merged_node["benchmark_reports"] = merge_benchmark_reports(
                merged_node.get("benchmark_reports"),
                node_data.get("benchmark_reports"),
            )
            if not merged_node["benchmark_reports"]:
                merged_node.pop("benchmark_reports")
            data["nodes"][node] = merged_node

    return data

def compact_response_data(data: Optional[dict]) -> dict:
    """Remove internal empty placeholders from an external API response."""
    merged = merge_response_data(empty_response_data(), data)
    compact = {}
    if merged.get("config_draft") is not None:
        compact["config_draft"] = merged["config_draft"]
    if merged.get("service_instances") is not None:
        compact["service_instances"] = merged["service_instances"]
    if merged.get("service_instance") is not None:
        compact["service_instance"] = merged["service_instance"]
    if merged.get("service_start") is not None:
        compact["service_start"] = merged["service_start"]
    if merged.get("service_stop") is not None:
        compact["service_stop"] = merged["service_stop"]
    if merged.get("test_runs") is not None:
        compact["test_runs"] = merged["test_runs"]
    if merged.get("test_run_stop") is not None:
        compact["test_run_stop"] = merged["test_run_stop"]
    if merged.get("admin_cleanup") is not None:
        compact["admin_cleanup"] = merged["admin_cleanup"]
    if merged.get("benchmark_stop") is not None:
        compact["benchmark_stop"] = merged["benchmark_stop"]
    if merged.get("benchmark_jobs") is not None:
        compact["benchmark_jobs"] = merged["benchmark_jobs"]
    if merged.get("benchmark_reports"):
        compact["benchmark_reports"] = merged["benchmark_reports"]
    if merged.get("nodes"):
        compact["nodes"] = merged["nodes"]
    return compact

def split_tool_observation(observation) -> tuple[str, dict]:
    if isinstance(observation, dict) and "_tool_text" in observation:
        return (
            str(observation.get("_tool_text") or ""),
            merge_response_data(empty_response_data(), observation.get("_response_data")),
        )
    if isinstance(observation, dict):
        response_data = {}
        for key in (
            "config_draft",
            "service_instances",
            "service_instance",
            "service_start",
            "service_stop",
            "test_runs",
            "test_run_stop",
            "admin_cleanup",
            "benchmark_stop",
            "benchmark_jobs",
            "benchmark_reports",
            "nodes",
        ):
            if observation.get(key) is not None:
                response_data[key] = observation[key]
        if observation.get("benchmarks") is not None and "benchmark_reports" not in response_data:
            response_data["benchmark_reports"] = observation["benchmarks"]
        if isinstance(observation.get("tests"), list) and "test_runs" not in response_data:
            response_data["test_runs"] = {
                "operation": "list",
                "items": observation["tests"],
                "summary": {
                    "total": observation.get("total"),
                    "returned": observation.get("returned"),
                },
            }
        if observation.get("test_run_id") is not None and observation.get("operation") is not None:
            response_data.setdefault("test_run_stop", observation)
        if observation.get("job_id") is not None and observation.get("operation") is not None:
            response_data.setdefault("benchmark_stop", observation)
        if isinstance(observation.get("summary"), dict) and all(key in observation for key in ("services", "benchmarks", "tests")):
            response_data.setdefault("admin_cleanup", observation)
        if isinstance(observation.get("data"), dict):
            response_data = merge_response_data(response_data, observation["data"])
        text = observation.get("text") or observation.get("result") or observation.get("message")
        if text is None:
            text = json.dumps(observation, ensure_ascii=False)
        return str(text), merge_response_data(empty_response_data(), response_data)
    return str(observation), empty_response_data()

def get_reasoning_text(message: AnyMessage) -> str:
    reasoning = None
    for attr in ("additional_kwargs", "response_metadata"):
        data = getattr(message, attr, None)
        if isinstance(data, dict) and data.get("reasoning"):
            reasoning = data.get("reasoning")
            break
    if not reasoning:
        return ""
    return str(reasoning).strip()
def pretty_print_cli_message(message: AnyMessage) -> None:
    reasoning = get_reasoning_text(message)
    if not reasoning or not isinstance(message, AIMessage):
        message.pretty_print()
        return

    original_content = message.content
    content = str(original_content or "").strip()
    message.content = f"<think>\n{reasoning}\n</think>"
    if content:
        message.content += f"\n\n{content}"
    try:
        message.pretty_print()
    finally:
        message.content = original_content

def controller_response_data(data: Optional[dict]) -> dict:
    return compact_response_data(data)

def build_tool_api_response(
    status: str,
    result: str,
    *,
    tool: str,
    data: Optional[dict] = None,
    role: Optional[str] = None,
) -> dict:
    response = {
        "status": status,
        "tool": tool,
        "result": result,
        "data": compact_response_data(data),
    }
    if role is not None:
        response["role"] = role
    return response

def build_agent_api_response(
    status: str,
    result: str,
    *,
    thread_id: str,
    data: Optional[dict] = None,
    role: Optional[str] = None,
    usage: Optional[dict] = None,
    trace: Optional[list[dict]] = None,
) -> dict:
    response = {
        "status": status,
        "thread_id": thread_id,
        "result": result,
        "data": controller_response_data(data),
    }
    if role is not None:
        response["role"] = role
    if usage is not None:
        response["usage"] = usage
    if trace is not None:
        response["trace"] = trace
    return response

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_calls: int
    response_data: dict

def graph_request_user_id(config: RunnableConfig) -> str:
    current = current_request_user_id()
    if current:
        return current
    configurable = config.get("configurable", {})
    request_user_id = str(configurable.get("request_user_id") or "").strip()
    if request_user_id:
        return request_user_id
    thread_id = str(configurable.get("thread_id") or "").strip()
    return f"studio:{thread_id}" if thread_id else ""

def graph_request_user_aliases(config: RunnableConfig) -> list[str]:
    current = current_request_user_aliases()
    if current:
        return current
    value = config.get("configurable", {}).get("user_aliases")
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def graph_thread_id(config: RunnableConfig) -> str:
    return str(config.get("configurable", {}).get("thread_id") or "").strip()

def graph_request_user_role(config: RunnableConfig) -> str:
    current = current_request_user_role()
    if current:
        return current
    return str(config.get("configurable", {}).get("user_role") or "").strip().lower()

def graph_resource_context(config: RunnableConfig) -> dict:
    value = config.get("configurable", {}).get("resource_context")
    return dict(value) if isinstance(value, dict) else {}

def reset_turn_state(state: MessagesState) -> dict:
    """Reset counters and response data once at the start of each graph run."""
    return {
        "llm_calls": 0,
        "tool_calls": 0,
        "response_data": empty_response_data(),
    }

def llm_node(state: MessagesState):
    """LLM decides whether to call a tool or not."""

    if state.get("llm_calls", 0) >= MAX_LLM_CALLS:
        return {"messages": [AIMessage(content="LLM 调用次数超过限制，任务已终止。")]}
    system_msg = [
        SystemMessage(
            content=(
                "你是生产级运维智能体。使用中文进行回答。\n"
                "规则：\n"
                "1. 当需要执行系统操作时必须调用工具。\n"
                "2. 不要假设工具执行成功，必须等待 Tool 返回。\n"
                "3. 不允许编造执行结果。\n"
                "4. 工具返回后，必须用中文向用户总结工具结果；不要返回空内容。\n"
                "5. 当请求用户角色为 admin，且用户意图是管理员维护、清理本地残留资源、查看全部实例、跨用户停止任务时，可以调用 admin_* 工具。\n"
                "6. 管理员说预览/查看/列出时只能调用 admin_*_preview 或 admin_*_list；只有明确说确认/执行/应用时才调用 admin_*_apply。\n"
                "7. 普通用户不得调用 admin_* 工具；普通用户预览停止自己的实例、benchmark 或功能测试时，使用 service_instance_stop_preview、benchmark_stop_preview、service_test_stop_preview 或对应 node_*_preview 工具。\n"
                "8. benchmark_stop/node_benchmark_stop 是实际停止操作；用户只说预览/查看时禁止调用，必须等用户明确说确认/执行/应用后才能调用。"
            )
        )
    ]

    try:
        response = run_with_timeout(
           lambda: model_with_tools.invoke(system_msg + state["messages"]),
            LLM_TIMEOUT_SECONDS,
            "模型调用超时",
            executor=LLM_EXECUTOR,
        )
    except Exception as e:
        print("[LLM_ERROR] model invocation failed")
        print(traceback.format_exc())
        error_type = type(e).__name__
        error_msg = str(e).strip() or "模型调用失败"
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"[LLM_ERROR] 模型调用失败: {error_type}: {error_msg}\n"
                        "请检查推理服务状态、模型服务端口和网络连接后重试。"
                    )
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(
    state: MessagesState, config: RunnableConfig
) -> Command[Literal["llm_node", END]]:
    user_token = set_current_request_user_id(graph_request_user_id(config))
    role_token = set_current_request_user_role(graph_request_user_role(config))
    aliases_token = set_current_request_user_aliases(graph_request_user_aliases(config))
    thread_token = set_current_request_thread_id(graph_thread_id(config))
    resource_token = set_current_request_resource_context(
        graph_resource_context(config)
    )
    try:
        return _tool_node(state)
    finally:
        reset_current_request_resource_context(resource_token)
        reset_current_request_thread_id(thread_token)
        reset_current_request_user_aliases(aliases_token)
        reset_current_request_user_role(role_token)
        reset_current_request_user_id(user_token)

def _tool_node(state: MessagesState):
    """Performs the tool call."""

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])
    completed_tool_calls = state.get("tool_calls", 0)
    if completed_tool_calls + len(tool_calls) > MAX_TOOL_CALLS:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content="Tools 调用次数超过限制，任务已终止。")]
            },
        )

    results = []
    response_data = merge_response_data(empty_response_data(), state.get("response_data"))

    for tool_call in tool_calls:
        # tool = tools_by_name[tool_call["name"]]
        # observation = tool.invoke(tool_call["args"])

        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        if (
            tool_name in CONFIRMATION_REQUIRED_NODE_TOOL_NAMES
            and isinstance(tool_args, dict)
            and not str(tool_args.get("confirm_text") or tool_args.get("command") or "").strip()
        ):
            tool_args = dict(tool_args)
            tool_args["confirm_text"] = latest_human_command_text(state["messages"])
        tool = tools_by_name.get(tool_name)
        if tool is None:
            observation = (
                f"[TOOL_ERROR]\n"
                f"name={tool_name}\n"
                f"args={tool_args}\n"
                "error=Unknown tool. Please choose one of the registered tools."
            )
        else:
            try:
                observation = tool.invoke(tool_args)
            except Exception as e:
                observation = (
                    f"[TOOL_ERROR]\n"
                    f"name={tool_name}\n"
                    f"args={tool_args}\n"
                    f"error_type={type(e).__name__}\n"
                    f"error={e}\n"
                    "请向用户说明工具调用失败的原因，并根据错误提示调整参数后重试；"
                    "不要编造工具执行结果。"
        )

        tool_text, tool_response_data = split_tool_observation(observation)
        response_data = merge_response_data(response_data, tool_response_data)

        results.append(
            ToolMessage(
                content=tool_text,
                tool_call_id=tool_call["id"],
            )
        )

    return Command(
        goto="llm_node",
        update={
            "messages": results,
            "tool_calls": completed_tool_calls + len(tool_calls),
            "response_data": response_data,
        },
    )

def route_by_tool(
    state: MessagesState,
) -> Literal["policy_node", END]:
    """Route to policy_node or end."""

    last = state["messages"][-1]

    if last.tool_calls:
        return "policy_node"

    return END

def policy_node(
    state: MessagesState, config: RunnableConfig
) -> Command[Literal["tool_node", END]]:
    user_token = set_current_request_user_id(graph_request_user_id(config))
    role_token = set_current_request_user_role(graph_request_user_role(config))
    aliases_token = set_current_request_user_aliases(graph_request_user_aliases(config))
    thread_token = set_current_request_thread_id(graph_thread_id(config))
    resource_token = set_current_request_resource_context(
        graph_resource_context(config)
    )
    try:
        return _policy_node(state)
    finally:
        reset_current_request_resource_context(resource_token)
        reset_current_request_thread_id(thread_token)
        reset_current_request_user_aliases(aliases_token)
        reset_current_request_user_role(role_token)
        reset_current_request_user_id(user_token)

def latest_human_command_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""

def ordinary_scope_all_requires_admin(command_text: str) -> bool:
    text = re.sub(r"\s+", "", str(command_text or "").lower())
    if not text:
        return True
    current_user_markers = ("当前用户", "当前", "自己", "我的", "mine", "current")
    if any(marker in text for marker in current_user_markers):
        return False
    return any(marker in text for marker in ("所有", "全部", "全量", "all", "跨用户", "管理员"))


def admin_apply_confirmed(command_text: str, tool_args: Optional[dict[str, Any]]) -> bool:
    tool_args = tool_args or {}
    if tool_args.get("confirmed") is True or tool_args.get("confirm") is True:
        return True
    text = str(command_text or tool_args.get("confirm_text") or tool_args.get("command") or "")
    normalized = re.sub(r"\s+", "", text.lower())
    return any(word in normalized for word in ("确认", "执行", "应用", "apply", "confirm", "confirmed"))


def service_instance_status_allows_tasks(
    instance_status: str, response_data: Optional[dict]
) -> bool:
    service_instance = (
        response_data.get("service_instance") if isinstance(response_data, dict) else None
    )
    if isinstance(service_instance, dict) and "ready_for_tasks" in service_instance:
        return service_instance.get("ready_for_tasks") is True

    status_text = str(instance_status or "")
    if "status=running" in status_text:
        return True
    if "status=degraded" not in status_text:
        return False

    for service in ("vllm", "inference", "case2chat"):
        pattern = rf"(?im)^\s*-\s*{re.escape(service)}\s*:\s*\S+\s+RUNNING\b"
        if not re.search(pattern, status_text):
            return False
    return True

def _policy_node(state: MessagesState) -> Command[Literal["tool_node", END]]:
    """Centralized policy enforcement."""

    last = state["messages"][-1]

    tool_call = last.tool_calls[0]
    action = tool_call["name"]

    allowed, message = policy_precheck(
        action,
        tool_args=tool_call.get("args", {}),
        command_text=latest_human_command_text(state["messages"]),
    )

    if not allowed:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=message)]},
        )

    return Command(goto="tool_node")

def policy_precheck(
    action: str,
    tool_map: Optional[dict] = None,
    tool_args: Optional[dict[str, Any]] = None,
    command_text: str = "",
) -> tuple[bool, str]:
    """
    Centralized execution policy layer.
    Returns:
        (True, "")  → allow execution
        (False, msg) → block execution with reason
    """

    tool_map = tool_map or tools_by_name
    tool_args = tool_args or {}

    if action in ADMIN_TOOL_NAMES:
        if current_request_user_role() != "admin":
            return False, "当前请求不是管理员角色，已拒绝调用管理员维护工具。"
        if not ADMIN_ENABLED:
            return False, "管理员维护能力未启用，已拒绝调用管理员维护工具。"
        if AGENT_ROLE not in {"worker", "both"}:
            return False, "当前进程不是 worker/both 角色，已拒绝调用本地管理员维护工具。"
        if action in ADMIN_APPLY_TOOL_NAMES and not admin_apply_confirmed(command_text, tool_args):
            return False, "管理员 apply 工具需要用户明确确认执行；预览或查看请求请使用 preview/list 工具。"
        return True, ""

    if (
        action in SCOPE_ALL_TOOL_NAMES
        and current_request_user_role() != "admin"
        and str(tool_args.get("scope") or "").strip().lower() == "all"
    ):
        if ordinary_scope_all_requires_admin(command_text):
            return False, "当前请求不是管理员角色，已拒绝调用管理员维护工具。"
        tool_args["scope"] = "mine"

    if action in CONFIRMATION_REQUIRED_APPLY_TOOL_NAMES:
        if not admin_apply_confirmed(command_text, tool_args):
            return False, "benchmark 停止需要用户明确确认；预览或查看请求请使用 benchmark_stop_preview/node_benchmark_stop_preview，确认目标无误后再发送确认停止命令。"
        return True, ""
    if current_request_user_role() == "admin" and action in {
        "benchmark_stop",
        "service_test_stop",
        "node_benchmark_stop",
        "node_service_test_stop",
    }:
        if not admin_apply_confirmed(command_text, tool_args):
            return False, "当前请求用户角色为 admin；管理员 apply 工具需要用户明确确认执行。请先使用 preview/list 工具，或在确认目标无误后使用确认命令。"
        return True, ""

    if action in ["service_test_run", "service_test_run_all", "benchmark_run"]:
        instance_id = str(tool_args.get("instance_id") or "latest").strip() or "latest"
        status_tool = tool_map.get("service_instance_status")
        if status_tool is not None:
            instance_status, status_data = split_tool_observation(
                status_tool.invoke({"instance_id": instance_id})
            )
        else:
            instance_status = ""
            status_data = {}
        if not service_instance_status_allows_tasks(instance_status, status_data):
            task_name = "benchmark" if action == "benchmark_run" else "test"
            return False, (
                f"检测到目标推理服务实例未运行，已跳过 {task_name}。\n\n当前实例状态：\n"
                + instance_status
            )
        return True, ""

    if action == "service_stop":
        benchmark_msg = running_benchmark_jobs_text()
        if benchmark_msg:
            return False, benchmark_msg
        return True, ""
    return True, ""

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("reset_turn_state", reset_turn_state)
agent_builder.add_node("llm_node", llm_node)
agent_builder.add_node("policy_node", policy_node)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "reset_turn_state")
agent_builder.add_edge("reset_turn_state", "llm_node")
agent_builder.add_conditional_edges(
    "llm_node",
    route_by_tool,
    ["policy_node", END],
)
LANGGRAPH_MANAGED_CHECKPOINTER = os.getenv(
    "MEDFLOW_LANGGRAPH_MANAGED_CHECKPOINTER", "0"
).strip().lower() in {"1", "true", "yes", "on"}

if LANGGRAPH_MANAGED_CHECKPOINTER:
    agent = agent_builder.compile()
else:
    agent = agent_builder.compile(checkpointer=build_checkpointer())

def normalize_thread_part(value: Optional[str], default: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    text = re.sub(r"[^A-Za-z0-9_.:@-]+", "_", text)
    return text[:120] or default

def resolve_thread_id(req: InferenceRequest) -> str:
    explicit_thread_id = normalize_thread_part(req.thread_id)
    if explicit_thread_id:
        return explicit_thread_id

    user_id = normalize_thread_part(req.user_id)
    session_id = normalize_thread_part(req.session_id)
    if user_id and session_id:
        return f"user:{user_id}:session:{session_id}"
    if user_id:
        return f"user:{user_id}"
    if session_id:
        return f"session:{session_id}"
    return "api"

def request_user_id_from_thread_id(thread_id: str) -> str:
    text = str(thread_id or "").strip()
    match = re.search(r"(?:^|:)user:([^:]+)", text)
    if match:
        return normalize_thread_part(match.group(1))
    if text.startswith("user:"):
        parts = text.split(":")
        if len(parts) >= 2:
            return normalize_thread_part(parts[1])
    return ""


def resolve_request_user_id(user_id: Optional[str], thread_id: str) -> str:
    explicit = normalize_thread_part(user_id)
    if explicit:
        return explicit
    from_thread = request_user_id_from_thread_id(thread_id)
    if from_thread:
        return from_thread
    return normalize_thread_part(thread_id)


def build_trace(messages: list[AnyMessage]) -> list[dict]:
    trace = []
    pending_tool_calls = {}

    for message in messages:
        if isinstance(message, HumanMessage) or isinstance(message, SystemMessage):
            continue

        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", []) or []
            for tool_call in tool_calls:
                pending_tool_calls[tool_call["id"]] = {
                    "type": "tool_call",
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args", {}),
                    "tool_call_id": tool_call.get("id"),
                }
            if message.content:
                trace.append(
                    {
                        "type": "ai",
                        "content_preview": message.content,
                    }
                )
            continue

        if isinstance(message, ToolMessage):
            item = pending_tool_calls.pop(
                message.tool_call_id,
                {
                    "type": "tool_call",
                    "name": None,
                    "args": {},
                    "tool_call_id": message.tool_call_id,
                },
            )
            content = str(message.content)
            item.update(
                {
                    "output_preview": content,
                    "output_truncated": False,
                }
            )
            trace.append(item)

    for item in pending_tool_calls.values():
        trace.append(item)

    return trace

def preview_tool_result(result) -> str:
    text = str(result).replace("\n", "\\n")
    if len(text) > TOOL_RESULT_LOG_CHARS:
        return text[:TOOL_RESULT_LOG_CHARS] + "... truncated ..."
    return text

def latest_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages

def build_usage(messages: list[AnyMessage]) -> dict:
    return {
        "llm_calls": sum(1 for message in messages if isinstance(message, AIMessage)),
        "tool_calls": sum(
            1 for message in messages if isinstance(message, ToolMessage)
        ),
    }

def fallback_answer_from_tool_messages(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "").strip()
        if content:
            return content
    return "操作已执行，但模型未生成最终回复。请稍后重试或查看状态。"

def inference_reservation_stop_request(command: str, resource_context: Optional[dict[str, Any]]) -> bool:
    if not isinstance(resource_context, dict):
        return False
    if not str(resource_context.get("reservation_id") or "").strip():
        return False
    normalized = str(command or "").strip()
    return normalized in {"停止推理服务", "关闭推理服务"}



def normalized_command_key(command: str) -> str:
    return re.sub(r"\s+", "", str(command or "").strip().lower())


def direct_rule_response(command: str, include_trace: bool = False) -> Optional[dict[str, Any]]:
    """Handle documented commands whose tool routing must not depend on the LLM."""
    key = normalized_command_key(command)
    if key == "查看当前推理服务实例列表":
        text, data = split_tool_observation(
            service_instance_list.invoke({"scope": "mine", "limit": 20})
        )
        response = {
            "result": text,
            "data": controller_response_data(data),
        }
        if include_trace:
            response["usage"] = {"llm_calls": 0, "tool_calls": 1}
            response["trace"] = [
                {
                    "type": "tool_call",
                    "name": "service_instance_list",
                    "args": {"scope": "mine", "limit": 20},
                    "output_preview": text,
                    "output_truncated": False,
                }
            ]
        return response
    return None

def run_service_agent(
    command: str,
    thread_id: str = "api",
    include_trace: bool = False,
    user_id: Optional[str] = None,
    resource_context: Optional[dict[str, Any]] = None,
    user_role: Optional[str] = None,
    user_aliases: Optional[list[str]] = None,
):
    resolved_user_id = resolve_request_user_id(user_id, thread_id)
    user_token = set_current_request_user_id(resolved_user_id)
    role_token = set_current_request_user_role(user_role or "")
    aliases_token = set_current_request_user_aliases(user_aliases or [])
    thread_token = set_current_request_thread_id(thread_id)
    resource_token = set_current_request_resource_context(resource_context)
    messages = [HumanMessage(content=command)]

    try:
        direct_response = direct_rule_response(command, include_trace)
        if direct_response is not None:
            return direct_response

        result = agent.invoke(
            {
                "messages": messages,
                "response_data": empty_response_data(),
            },
            {
                "configurable": {
                    "thread_id": thread_id,
                    "request_user_id": resolved_user_id,
                    "user_aliases": user_aliases or [],
                    "user_role": user_role or "",
                    "resource_context": resource_context or {},
                }
            },
        )
    finally:
        reset_current_request_resource_context(resource_token)
        reset_current_request_thread_id(thread_token)
        reset_current_request_user_aliases(aliases_token)
        reset_current_request_user_role(role_token)
        reset_current_request_user_id(user_token)

    new_messages = result["messages"]
    turn_messages = latest_turn_messages(new_messages)

    final_answer = fallback_answer_from_tool_messages(turn_messages)
    for m in reversed(turn_messages):
        if isinstance(m, AIMessage):
            content = str(m.content or "").split("</think>")[-1].strip()
            if not content:
                continue
            final_answer = content
            break

    response = {
        "result": final_answer,
        "data": controller_response_data(result.get("response_data")),
    }
    if include_trace:
        response["usage"] = build_usage(turn_messages)
        response["trace"] = build_trace(turn_messages)

    return response

@app.post("/worker/tool")
def run_inference_agent_tool(req: ToolInvokeRequest):
    request_id = f"tool-{int(time.time() * 1000)}"
    tool_name = str(req.tool or "").strip()
    tool_args = req.args or {}
    resolved_user_id = resolve_request_user_id(req.user_id, req.thread_id or "")
    user_token = set_current_request_user_id(resolved_user_id)
    role_token = set_current_request_user_role(req.user_role or "")
    aliases_token = set_current_request_user_aliases(req.user_aliases or [])
    thread_token = set_current_request_thread_id(req.thread_id or "")
    resource_token = set_current_request_resource_context(req.resource_context)
    start = time.time()
    try:
        try:
            return _run_inference_agent_tool(
                req, request_id, tool_name, tool_args, start
            )
        except Exception as exc:
            duration = time.time() - start
            print(
                f"[worker-tool-api][{request_id}] failed role={AGENT_ROLE} "
                f"thread_id={current_request_thread_id()} "
                f"tool={tool_name} duration={duration:.3f}s "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            print(traceback.format_exc())
            return build_tool_api_response(
                "error",
                (
                    "工具请求处理失败。\n"
                    f"error_type={type(exc).__name__}\n"
                    f"error={exc}"
                ),
                tool=tool_name,
            )
    finally:
        reset_current_request_resource_context(resource_token)
        reset_current_request_thread_id(thread_token)
        reset_current_request_user_aliases(aliases_token)
        reset_current_request_user_role(role_token)
        reset_current_request_user_id(user_token)

def _run_inference_agent_tool(
    req: ToolInvokeRequest,
    request_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    start: float,
):

    print(
        f"\n[worker-tool-api][{request_id}] request role={AGENT_ROLE} user_role={current_request_user_role()} "
        f"thread_id={current_request_thread_id()} tool={tool_name} args={tool_args} "
        f"resource_context={req.resource_context or {}}",
        flush=True,
    )

    if AGENT_ROLE not in {"worker", "both"}:
        return build_tool_api_response(
            "error",
            "当前进程不是 worker/both 角色，不提供 /worker/tool 工具执行接口。",
            tool=tool_name,
            role=AGENT_ROLE,
        )

    tool = worker_tools_by_name.get(tool_name)
    if tool is None:
        response = build_tool_api_response(
            "error",
            (
                f"Unknown worker tool: {tool_name}. "
                f"Available tools: {', '.join(sorted(worker_tools_by_name))}"
            ),
            tool=tool_name,
        )

        duration = time.time() - start
        print(
            f"[worker-tool-api][{request_id}] response role={AGENT_ROLE} "
            f"thread_id={current_request_thread_id()} "
            f"tool={tool_name} status=error duration={duration:.3f}s "
            f"result={preview_tool_result(response['result'])}",
            flush=True,
        )
        return response

    allowed, message = policy_precheck(
        tool_name,
        worker_tools_by_name,
        tool_args=tool_args,
        command_text=str(tool_args.get("confirm_text") or tool_args.get("command") or ""),
    )
    if not allowed:
        response = build_tool_api_response(
            "blocked", message, tool=tool_name
        )

        duration = time.time() - start
        print(
            f"[worker-tool-api][{request_id}] response role={AGENT_ROLE} "
            f"thread_id={current_request_thread_id()} "
            f"tool={tool_name} status=blocked duration={duration:.3f}s "
            f"result={preview_tool_result(response['result'])}",
            flush=True,
        )
        return response

    try:
        result = tool.invoke(tool_args)
        result_text, response_data = split_tool_observation(result)
        status = "ok"
    except Exception as e:
        print(
            f"[worker-tool-api][{request_id}] error role={AGENT_ROLE} "
            f"thread_id={current_request_thread_id()} "
            f"tool={tool_name} invocation failed",
            flush=True,
        )
        print(traceback.format_exc())
        result = (
            f"[TOOL_ERROR]\n"
            f"name={tool_name}\n"
            f"args={tool_args}\n"
            f"error_type={type(e).__name__}\n"
            f"error={e}"
        )
        result_text = result
        response_data = empty_response_data()
        status = "error"

    duration = time.time() - start
    print(
        f"[worker-tool-api][{request_id}] response role={AGENT_ROLE} user_role={current_request_user_role()} "
        f"thread_id={current_request_thread_id()} "
        f"tool={tool_name} status={status} duration={duration:.3f}s "
        f"result={preview_tool_result(result_text)}",
        flush=True,
    )
    return build_tool_api_response(
        status,
        result_text,
        tool=tool_name,
        data=response_data,
    )

@app.post("/inference_agent")
def run_inference_agent(req: InferenceRequest):
    request_id = f"ctrl-{int(time.time() * 1000)}"
    thread_id = resolve_thread_id(req)
    start = time.time()
    print(
        f"\n[controller-api][{request_id}] request role={AGENT_ROLE} user_role={req.user_role or ''} "
        f"thread_id={thread_id} resource_context={req.resource_context or {}} "
        f"command={req.command}",
        flush=True,
    )

    resolved_request_user_id = resolve_request_user_id(req.user_id, thread_id)

    if AGENT_ROLE == "worker":
        return build_agent_api_response(
            "error",
            "当前进程是 worker 角色，只提供 /worker/tool 内部工具接口。",
            thread_id=thread_id,
            role=AGENT_ROLE,
        )

    if inference_reservation_stop_request(req.command, req.resource_context):
        resolved_user_id = resolve_request_user_id(req.user_id, thread_id)
        user_token = set_current_request_user_id(resolved_user_id)
        role_token = set_current_request_user_role(req.user_role or "")
        aliases_token = set_current_request_user_aliases(req.user_aliases or [])
        thread_token = set_current_request_thread_id(thread_id)
        resource_token = set_current_request_resource_context(req.resource_context)
        try:
            stop_result = service_stop_by_reservation_result()
        finally:
            reset_current_request_resource_context(resource_token)
            reset_current_request_thread_id(thread_token)
            reset_current_request_user_aliases(aliases_token)
            reset_current_request_user_role(role_token)
            reset_current_request_user_id(user_token)
        result_text = str(stop_result.get("result") or "")
        status = "ok" if stop_result.get("stopped") else "error"
        duration = time.time() - start
        print(
            f"[controller-api][{request_id}] response role={AGENT_ROLE} "
            f"thread_id={thread_id} duration={duration:.3f}s "
            f"result_len={len(result_text)} result={result_text}",
            flush=True,
        )
        service_stop = stop_result.get("service_stop") if isinstance(stop_result.get("service_stop"), dict) else None
        if service_stop is None:
            service_stop = {
                "stopped": bool(stop_result.get("stopped")),
                "release_ready": bool(stop_result.get("stopped")),
                "instance_id": str(stop_result.get("instance_id") or ""),
                "reservation_id": str(stop_result.get("reservation_id") or (req.resource_context or {}).get("reservation_id") or ""),
                "status": "stopped" if stop_result.get("stopped") else str(stop_result.get("status") or "error"),
                "result": result_text,
            }
        return build_agent_api_response(
            status,
            result_text,
            thread_id=thread_id,
            data={"inference_service_stop": stop_result, "service_stop": service_stop},
            role=AGENT_ROLE,
        )

    try:
        agent_result = run_with_timeout(
            lambda: run_service_agent(
                req.command,
                thread_id,
                req.include_trace,
                resolved_request_user_id,
                req.resource_context,
                req.user_role,
                req.user_aliases,
            ),
            AGENT_TIMEOUT_SECONDS,
            "请求处理超时",
        )
    except TimeoutError as e:
        duration = time.time() - start
        result_text = "请求处理时间较长，本次操作尚未完成，请稍后再试。"
        print(
            f"[controller-api][{request_id}] timeout role={AGENT_ROLE} "
            f"thread_id={thread_id} duration={duration:.3f}s "
            f"timeout={AGENT_TIMEOUT_SECONDS}s error={e}",
            flush=True,
        )
        return build_agent_api_response(
            "timeout", result_text, thread_id=thread_id
        )
    except Exception as e:
        duration = time.time() - start
        print(
            f"[controller-api][{request_id}] failed role={AGENT_ROLE} "
            f"thread_id={thread_id} duration={duration:.3f}s "
            f"error={type(e).__name__}: {e}",
            flush=True,
        )
        print(traceback.format_exc())
        return build_agent_api_response(
            "error",
            "请求处理失败，本次操作未完成，请稍后重试。",
            thread_id=thread_id,
        )

    duration = time.time() - start
    response_data_preview = preview_tool_result(
        json.dumps(agent_result.get("data") or {}, ensure_ascii=False)
    )
    print(
        f"[controller-api][{request_id}] response role={AGENT_ROLE} user_role={req.user_role or ''} "
        f"thread_id={thread_id} duration={duration:.3f}s "
        f"result_len={len(str(agent_result['result']))} result={agent_result['result']} "
        f"data={response_data_preview}",
        flush=True,
    )

    return build_agent_api_response(
        "ok",
        agent_result["result"],
        thread_id=thread_id,
        data=agent_result["data"],
        usage=agent_result.get("usage") if req.include_trace else None,
        trace=agent_result.get("trace") if req.include_trace else None,
    )

def main():
    initialize_worker_node_config()
    print("\n🟢 Inference Service Agent")
    print("Type: start / stop / status / test / logs / exit\n")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bye 👋")
            break

        messages = [HumanMessage(content=str(user_input))]

        cli_thread_id = "cli:1"
        user_token = set_current_request_user_id(cli_thread_id)
        try:
            result = agent.invoke(
                {
                    "messages": messages,
                    "response_data": empty_response_data(),
                },
                {"configurable": {"thread_id": cli_thread_id}},
            )
        finally:
            reset_current_request_user_id(user_token)

        new_messages = result["messages"]

        target_index = -1
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i], HumanMessage):
                target_index = i
                break

        for m in new_messages[target_index:]:
            pretty_print_cli_message(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cli", "api"], default="api")
    args = parser.parse_args()

    if args.mode == "cli":
        main()
    else:
        uvicorn.run(
            app="inference_agent:app",
            host=INFERENCE_AGENT_HOST,
            port=INFERENCE_AGENT_PORT,
        )







