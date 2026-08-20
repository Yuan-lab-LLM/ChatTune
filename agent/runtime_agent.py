import asyncio
import argparse
import inspect
import json
import logging
import logging.handlers
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medflow_agent_tools._dataset_candidates import (
    dataset_find_exclusion_clause,
    dataset_names_from_dataset_info,
    is_dataset_candidate_filename,
)

# 导入配置和内存管理模块
from utils import SessionState, get_current_config, get_memory_manager, init_config
from utils.workflow_manager import (
    STAGES,
    WorkflowDependencies,
    WorkflowManager,
    _benchmark_name,
    _benchmark_enrich_runtime_result,
    _benchmark_result_entry,
    _benchmark_start_command,
    _benchmark_status_command,
    parse_workflow_control_command,
    train_result_reached_completion,
)

# 初始化配置（必须在导入 agentscope 之前）
config = init_config("config/config.yaml")

# 配置日志
log_path = Path(config.logging.file)
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format,
    handlers=[
        logging.handlers.RotatingFileHandler(
            config.logging.file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

AGENT_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
RESET_EPOCHS: Dict[str, int] = {}


def _session_key_candidates(user_id: str) -> Set[str]:
    normalized_user_id = (user_id or "").strip()
    if not normalized_user_id:
        return set()
    bare_user_id = normalized_user_id.strip("[]")
    return {
        normalized_user_id,
        bare_user_id,
        f"[{bare_user_id}]",
    }


def _current_reset_epoch(user_id: str) -> int:
    return max((RESET_EPOCHS.get(candidate, 0) for candidate in _session_key_candidates(user_id)), default=0)


def _mark_context_reset(user_id: str) -> Set[str]:
    candidates = _session_key_candidates(user_id)
    next_epoch = max((RESET_EPOCHS.get(candidate, 0) for candidate in candidates), default=0) + 1
    for candidate in candidates:
        RESET_EPOCHS[candidate] = next_epoch
    return candidates


def _stop_active_workflows_for_candidates(candidates: Set[str]) -> None:
    """Mark active one-click workflows as stopped so the background worker will not resume them."""
    if not candidates:
        return
    try:
        manager = get_workflow_manager()
    except Exception:
        logger.exception("Failed to get workflow manager while resetting context")
        return

    stopped_workflow_ids: Set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            workflow = manager.active_for_user_family(candidate)
            if not workflow:
                continue
            workflow_id = str(workflow.get("workflow_id") or "")
            if not workflow_id or workflow_id in stopped_workflow_ids:
                continue
            if workflow.get("status") != "stopping":
                workflow = manager.begin_stop_by_id(workflow_id, request_user_id=candidate)
            manager.complete_stop(str(workflow["workflow_id"]))
            stopped_workflow_ids.add(workflow_id)
        except Exception:
            logger.exception("Failed to stop active workflow while resetting context: %s", candidate)
    if stopped_workflow_ids:
        logger.info("Stopped active workflows during context reset: %s", sorted(stopped_workflow_ids))


async def reset_user_context(
    user_id: str,
    extra_candidates: Optional[Set[str]] = None,
    cancel_workflows: bool = False,
) -> None:
    """Reset one Studio user's conversational session state."""
    candidates = _mark_context_reset(user_id)
    if extra_candidates:
        candidates = candidates.union(extra_candidates)
    
    if not candidates:
        logger.warning("Skip reset_user_context because user_id is empty")
        return

    memory_manager = await get_memory_manager()
    for candidate in candidates:
        if candidate:
            await memory_manager.remove_session(candidate)
    if cancel_workflows:
        _stop_active_workflows_for_candidates(candidates)
    logger.info("Reset user context for candidates: %s", sorted(candidates))


def handle_studio_reset_context(payload: Dict[str, Any]) -> None:
    """Schedule a Studio-triggered context reset on the agent main loop."""
    username = str(payload.get("username") or "").strip()
    user_id = str(payload.get("userId") or "").strip()
    context_username = str(payload.get("contextUsername") or "").strip()
    run_id = str(payload.get("runId") or "").strip()
    cancel_workflows = bool(payload.get("cancelWorkflows") or payload.get("cancel_workflows"))
    
    # Collect all possible identifiers and generate candidates for each
    all_candidates: Set[str] = set()
    for key in (context_username, username, user_id):
        if key:
            all_candidates.update(_session_key_candidates(key))
    
    # The actual session key in Python is often "userId#runId", so include that too
    if user_id and run_id:
        all_candidates.update(_session_key_candidates(f"{user_id}#{run_id}"))
    
    if not all_candidates:
        logger.warning("Received resetAgentContext without username/userId/contextUsername: %s", payload)
        return
    
    # Use context_username as the primary key for reset_user_context,
    # and pass all other candidates as extras
    primary_key = context_username or username or user_id
    extra_candidates = all_candidates - set(_session_key_candidates(primary_key))
    
    loop = AGENT_MAIN_LOOP
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            reset_user_context(
                primary_key,
                extra_candidates if extra_candidates else None,
                cancel_workflows=cancel_workflows,
            ),
            loop,
        )

        def _log_reset_result(done_future):
            try:
                done_future.result()
            except Exception:
                logger.exception("Studio-triggered context reset failed")

        future.add_done_callback(_log_reset_result)
        return

    # 如果当前线程已有运行中的事件循环（如 FastAPI 请求处理），
    # 不能调用 asyncio.run()，改为创建后台任务
    try:
        current_loop = asyncio.get_running_loop()
        task = current_loop.create_task(
            reset_user_context(
                primary_key,
                extra_candidates if extra_candidates else None,
                cancel_workflows=cancel_workflows,
            )
        )

        def _log_task_result(done_task):
            try:
                done_task.result()
            except Exception:
                logger.exception("Studio-triggered context reset failed")

        task.add_done_callback(_log_task_result)
        return
    except RuntimeError:
        # 没有运行中的事件循环，可以安全使用 asyncio.run()
        pass

    asyncio.run(
        reset_user_context(
            primary_key,
            extra_candidates if extra_candidates else None,
            cancel_workflows=cancel_workflows,
        )
    )

# 导入 agentscope 2.x 相关模块
from agentscope.agent import Agent as AgentScopeAgent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import Base64Source, DataBlock, Msg, TextBlock, URLSource
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit as AgentScopeToolkit, ToolResponse
from medflow_agent_tools import (
    run_script_by_name_data,
    run_script_by_name_data_collect,
    run_script_by_name_assessment,
    run_script_by_name_evaluate,
    run_script_by_name_monitor1,
    run_script_by_name_train,
    validate_training_inputs_preflight,
    run_script_by_name_assessment_monitor,
    run_script_by_name_evaluate_monitor
)
from medflow_agent_tools.runlocal_train import (
    _local_gpu_snapshot_for_runtime_request,
    _release_resource_allocation,
    _runtime_training_resource_request,
)
from resource_api import (
    _read_training_allocation,
    _stop_multinode_allocation_processes,
    _training_allocation_file_from_record,
)

def _model_generate_kwargs() -> Dict[str, Any]:
    generate_kwargs = config.model.generate_kwargs
    if hasattr(generate_kwargs, "model_dump"):
        return generate_kwargs.model_dump()
    return generate_kwargs.dict()


# ========== 全局共享资源 ==========
# 共享模型实例（无状态，可复用）
SHARED_MODEL = OpenAIChatModel(
    credential=OpenAICredential(
        api_key=config.model.api_key,
        base_url=config.model.base_url,
    ),
    model=config.model.name,
    parameters=OpenAIChatModel.Parameters(**_model_generate_kwargs()),
    stream=config.model.stream,
    client_kwargs={"base_url": config.model.base_url} if config.model.base_url else None,
)



class _MedFlowMemory:
    def __init__(self) -> None:
        self.messages: List[Msg] = []

    async def add(self, msg: Msg) -> None:
        self.messages.append(msg)

    async def clear(self) -> None:
        self.messages.clear()


class Toolkit(AgentScopeToolkit):
    """AgentScope 2.x Toolkit with MedFlow's former registration helper."""

    def __init__(self, *args, **kwargs):
        self._schema_log_label = kwargs.pop("schema_log_label", None)
        super().__init__(*args, **kwargs)
        self._pending_add_tool_tasks: List[Any] = []
        self._last_tool_response_msg: Optional[Msg] = None

    def clear_response_msg(self) -> None:
        self._last_tool_response_msg = None

    def pop_response_msg(self) -> Optional[Msg]:
        response_msg = self._last_tool_response_msg
        self._last_tool_response_msg = None
        return response_msg

    async def finalize(self) -> None:
        """Finish async AgentScope 2.x tool registration before Agent construction."""
        pending = self._pending_add_tool_tasks
        self._pending_add_tool_tasks = []
        for task in pending:
            await task
        await self.log_tool_schemas(self._schema_log_label)

    async def log_tool_schemas(self, label: Optional[str] = None) -> None:
        """Log compact schema details so tool exposure is easy to verify."""
        try:
            schemas = await self.get_tool_schemas()
        except Exception as exc:
            logger.warning(
                "AgentScope toolkit schema validation failed%s: %s",
                f" for {label}" if label else "",
                exc,
            )
            return

        schema_parts = []
        for schema in schemas:
            function = schema.get("function") if isinstance(schema, dict) else None
            if not isinstance(function, dict):
                continue
            params = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
            properties = params.get("properties") if isinstance(params, dict) else {}
            required = params.get("required") if isinstance(params, dict) else []
            schema_parts.append(
                "%s(params=%s required=%s)" % (
                    function.get("name"),
                    sorted(properties.keys()) if isinstance(properties, dict) else [],
                    required if isinstance(required, list) else [],
                )
            )

        logger.info(
            "AgentScope toolkit schemas%s: %s",
            f" for {label}" if label else "",
            "; ".join(schema_parts) or "<no tools>",
        )

    @staticmethod
    def _agent_tool_description(func, explicit_description: Optional[str] = None) -> str:
        """Build a tool description that helps the LLM decide when to call it."""
        base = (explicit_description or inspect.getdoc(func) or getattr(func, "__doc__", "") or "").strip()
        name = getattr(func, "__name__", "")
        usage_hints = {
            "_stop_task": "Use when the user asks to stop, cancel, terminate, or end a running MedFlow task.",
            "_call_datacollector": "Use when the user asks for data collection or data preparation.",
            "_call_dataprocessor": "Use when the user asks for data preprocessing or advanced data filtering.",
            "_call_trainer": "Use when the user asks to start or schedule model training, including LoRA SFT, full SFT, enhanced training, GRPO, or multinode training.",
            "_call_evaluator": "Use when the user asks for model evaluation, single-model evaluation, dual-model evaluation, checkpoint evaluation, or benchmark-style assessment.",
            "_call_inference": "Use when the user asks about inference service operations, inference configuration, node operations, function tests, or inference benchmarks.",
            "_call_monitor": "Use when the user asks for training, evaluation, or process monitoring/status.",
            "_call_analysis": "Use when the user asks to analyze training curves or training metrics.",
            "_workflow_control": "Use only for explicit workflow control commands such as continue, skip, retry, or workflow status.",
            "_run_inference_agent_command": "Use for all real inference service, node operation, function-test, and benchmark requests. Pass the user's original request as command.",
        }
        hint = usage_hints.get(name)
        policy = (
            "Call this tool only when its description matches the user's request. "
            "If required parameters are missing, ask the user for the missing values before executing. "
            "Do not invent parameters, statuses, file paths, service states, or execution results. "
            "After the tool returns, summarize only the returned result."
        )
        return "\n\n".join(part for part in [base, hint, policy] if part)

    def register_tool_function(self, func, postprocess_func=None, **kwargs):
        original_signature = inspect.signature(func)
        explicit_description = kwargs.pop("description", None)
        tool_name = kwargs.pop("name", getattr(func, "__name__", None))
        group_name = kwargs.get("group_name", "basic")
        tool_description = self._agent_tool_description(func, explicit_description)

        @wraps(func)
        async def _wrapped(*tool_args, **tool_kwargs):
            if "task_description" in tool_kwargs:
                call_args = (tool_kwargs["task_description"],)
                call_kwargs = {}
            elif "command" in tool_kwargs:
                call_args = (tool_kwargs["command"],)
                call_kwargs = {}
            else:
                call_args = tool_args
                call_kwargs = tool_kwargs
            result = await func(*call_args, **call_kwargs) if asyncio.iscoroutinefunction(func) else func(*call_args, **call_kwargs)
            if postprocess_func and isinstance(result, ToolResponse):
                processed = postprocess_func(
                    {"name": getattr(func, "__name__", ""), "arguments": {"args": tool_args, **tool_kwargs}},
                    result,
                )
                if processed is not None:
                    result = processed
            if isinstance(result, ToolResponse):
                metadata = result.metadata or {}
                response_msg = metadata.get("response_msg")
                if isinstance(response_msg, Msg):
                    self._last_tool_response_msg = response_msg
                return ToolChunk(
                    content=result.content,
                    state=result.state,
                    metadata=metadata,
                    is_last=True,
                )
            return result

        _wrapped.__signature__ = original_signature
        _wrapped.__doc__ = tool_description
        _wrapped.__name__ = tool_name or getattr(func, "__name__", "_wrapped")

        register = getattr(super(), "register_tool_function", None)
        if callable(register):
            register_result = register(_wrapped, name=tool_name, description=tool_description, **kwargs)
            if asyncio.iscoroutine(register_result):
                self._pending_add_tool_tasks.append(register_result)
                return func
            return register_result

        add_result = self.add_tool(
            FunctionTool(
                _wrapped,
                name=tool_name,
                description=tool_description,
            ),
            group_name=group_name,
        )
        if asyncio.iscoroutine(add_result):
            self._pending_add_tool_tasks.append(add_result)
        return func


def _default_agent_state() -> AgentState:
    return AgentState(
        permission_context=PermissionContext(mode=PermissionMode.BYPASS),
    )


def _default_react_config() -> ReActConfig:
    max_iters_raw = os.getenv("MEDFLOW_AGENT_REACT_MAX_ITERS", "8")
    try:
        max_iters = int(max_iters_raw)
    except ValueError:
        max_iters = 8
    return ReActConfig(max_iters=max(1, max_iters))


class Agent(AgentScopeAgent):
    """Thin AgentScope 2.x adapter for MedFlow's existing call sites."""

    def __init__(self, *, name: str, system_prompt: str, model, toolkit=None, **kwargs):
        kwargs.setdefault("state", _default_agent_state())
        kwargs.setdefault("react_config", _default_react_config())
        super().__init__(name=name, system_prompt=system_prompt, model=model, toolkit=toolkit, **kwargs)
        self.memory = _MedFlowMemory()
        self._medflow_hooks: Dict[str, List[Any]] = {}
        self._medflow_toolkit = toolkit

    def register_instance_hook(self, hook_type: str, name: str, hook) -> None:
        self._medflow_hooks.setdefault(hook_type, []).append(hook)

    async def print(self, msg: Msg, *args, **kwargs) -> None:
        hook_kwargs = {"msg": msg, "args": args, "kwargs": kwargs}
        for hook in self._medflow_hooks.get("pre_print", []):
            updated = hook(self, hook_kwargs)
            if isinstance(updated, dict):
                hook_kwargs = updated
                msg = hook_kwargs.get("msg", msg)
        await self.memory.add(msg)

    async def __call__(self, msg: Msg) -> Msg:
        toolkit = self._medflow_toolkit
        if hasattr(toolkit, "clear_response_msg"):
            toolkit.clear_response_msg()
        reply = await self.reply(msg)
        if hasattr(toolkit, "pop_response_msg"):
            response_msg = toolkit.pop_response_msg()
            if isinstance(response_msg, Msg):
                return response_msg
        return reply


def _content_block(block: Any) -> Any:
    if isinstance(block, (TextBlock, DataBlock)):
        return block
    if isinstance(block, str):
        return TextBlock(text=block)
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "text":
            return TextBlock(text=str(block.get("text") or ""))
        if block_type in {"image", "audio", "video", "data"}:
            source = block.get("source") if isinstance(block.get("source"), dict) else {}
            source_type = source.get("type")
            media_type = str(source.get("media_type") or source.get("mediaType") or "application/octet-stream")
            if source_type == "base64" and source.get("data"):
                return DataBlock(
                    source=Base64Source(data=str(source.get("data")), media_type=media_type),
                    name=block.get("name"),
                )
            if source_type == "url" and source.get("url"):
                return DataBlock(
                    source=URLSource(url=str(source.get("url")), media_type=media_type),
                    name=block.get("name"),
                )
    return TextBlock(text=str(block or ""))


def _content_blocks(content: Any) -> List[Any]:
    if isinstance(content, list):
        return [_content_block(block) for block in content]
    return [_content_block(content)]


def _msg(name: str, content: Any, role: str = "assistant", metadata: Optional[Dict[str, Any]] = None) -> Msg:
    return Msg(name=name, content=_content_blocks(content), role=role, metadata=metadata or {})


def _tool_response_text(response: ToolResponse) -> str:
    parts: List[str] = []
    for block in response.content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()

logger.info(f"Shared model initialized: {config.model.name}")

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
TRAIN_PID_REGISTRY = RUNTIME_DIR / "train_pid_registry.jsonl"
EVALUATE_PID_REGISTRY = RUNTIME_DIR / "evaluate_pid_registry.jsonl"
BACKGROUND_TASK_REGISTRY = RUNTIME_DIR / "background_task_registry.jsonl"
WORKFLOW_LOG_DIR = RUNTIME_DIR / "workflow_logs"
DEFAULT_DOCKER_CONTAINER = config.environment.default_docker_container
DEFAULT_EVALUATE_DOCKER_CONTAINER = config.environment.default_evaluate_docker_container
BENCHMARK_LOGS_PATH = "/home/workspace/tests/logs/benchmark"
BENCHMARK_DATASET_ALIASES = {
    "truthfulqa": {"truthfulqa", "truthfulqa-generation"},
}
REQUEST_TRAINING_CONTAINER: ContextVar[str] = ContextVar(
    "request_training_container",
    default=DEFAULT_DOCKER_CONTAINER,
)
REQUEST_EVALUATION_CONTAINER: ContextVar[str] = ContextVar(
    "request_evaluation_container",
    default=DEFAULT_EVALUATE_DOCKER_CONTAINER,
)
REQUEST_GRPO_CONTAINER: ContextVar[str] = ContextVar(
    "request_grpo_container",
    default=DEFAULT_DOCKER_CONTAINER,
)
REQUEST_MULTINODE_TRAINING_CONTAINER: ContextVar[str] = ContextVar(
    "request_multinode_training_container",
    default="",
)
REQUEST_RESOURCE_GROUP_ID: ContextVar[str] = ContextVar(
    "request_resource_group_id",
    default="",
)
REQUEST_TRAINING_POOL_ID: ContextVar[str] = ContextVar(
    "request_training_pool_id",
    default="",
)
REQUEST_USER_ROLE: ContextVar[str] = ContextVar(
    "request_user_role",
    default="",
)
REQUEST_INFERENCE_OWNER_USER_ID: ContextVar[str] = ContextVar(
    "request_inference_owner_user_id",
    default="",
)
REQUEST_INFERENCE_OWNER_ALIASES: ContextVar[tuple[str, ...]] = ContextVar(
    "request_inference_owner_aliases",
    default=(),
)

INFERENCE_RESOURCE_RESERVATIONS: Dict[str, Dict[str, Any]] = {}
INFERENCE_RESOURCE_RESERVATIONS_LOCK = threading.Lock()
INFERENCE_RESOURCE_HEARTBEATS: Dict[str, threading.Event] = {}
INFERENCE_RESOURCE_HEARTBEATS_LOCK = threading.Lock()


def _inference_gpus_per_node() -> int:
    for env_name in ("MEDFLOW_INFERENCE_GPUS_PER_NODE", "AGENT3_INFERENCE_GPUS_PER_NODE"):
        raw_value = str(os.getenv(env_name) or "").strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                logger.warning("Ignore invalid %s=%s", env_name, raw_value)
    service_config = Path(__file__).resolve().parent.parent / "medflow" / "agent" / "config" / "service.yaml"
    try:
        payload = yaml.safe_load(service_config.read_text(encoding="utf-8")) or {}
        runtime = payload.get("RUNTIME") if isinstance(payload.get("RUNTIME"), dict) else {}
        return max(1, int(runtime.get("TENSOR_PARALLEL_SIZE") or 1))
    except Exception:
        return 1


def _inference_command_needs_resource_context(command: str) -> bool:
    normalized = str(command or "").lower()
    return any(
        keyword in normalized
        for keyword in (
            "启动推理服务",
            "重启推理服务",
            "service_start",
            "service_restart",
            "start inference",
            "restart inference",
        )
    )


def _inference_command_may_stop_service(command: str) -> bool:
    normalized = str(command or "").lower()
    return any(
        keyword in normalized
        for keyword in (
            "停止推理服务",
            "关闭推理服务",
            "关闭推理",
            "停止推理",
            "重启推理服务",
            "service_stop",
            "service_instance_stop",
            "service_restart",
            "stop inference",
            "restart inference",
        )
    )

def _inference_protocol_is_service_stop(protocol: Any) -> bool:
    if not isinstance(protocol, dict):
        return False
    return (
        str(protocol.get("type") or "").strip().lower() == "job_stopped"
        and str(protocol.get("jobType") or protocol.get("job_type") or "").strip().lower() == "inference_service"
        and str(protocol.get("action") or "").strip().lower() == "service_stop"
    )


def _inference_payload_is_service_stop(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return _inference_protocol_is_service_stop(payload.get("protocol"))


def _inference_stop_response_is_success(command: str, data: Any) -> bool:
    if not isinstance(data, dict) or not _inference_command_may_stop_service(command):
        return False
    response_data = data.get("data") if isinstance(data.get("data"), dict) else {}
    service_stop = response_data.get("service_stop")
    if isinstance(service_stop, dict):
        return service_stop.get("stopped") is True and service_stop.get("release_ready") is not False
    inference_stop = response_data.get("inference_service_stop")
    if isinstance(inference_stop, dict):
        nested_stop = inference_stop.get("service_stop")
        if isinstance(nested_stop, dict):
            return nested_stop.get("stopped") is True and nested_stop.get("release_ready") is not False
    text = str(data.get("result") or data.get("message") or "")
    status = str(data.get("status") or "").lower()
    if status not in {"error", "failed", "timeout"}:
        return True
    return any(
        keyword in text
        for keyword in (
            "已处于非活动状态",
            "无需重复停止",
            "已停止",
            "无需重复操作",
            "未检测到存活进程",
            "not active",
            "already stopped",
            "already inactive",
            "no running process",
        )
    )
def _inference_resource_env(runtime_node_id: str = "") -> Dict[str, str]:
    return {
        "MEDFLOW_RESOURCE_NODE_ID": runtime_node_id or os.getenv("MEDFLOW_RESOURCE_NODE_ID", ""),
        "MEDFLOW_RESOURCE_GROUP_ID": _current_resource_group_id(),
        "MEDFLOW_TRAINING_POOL_ID": _current_training_pool_id(),
    }


def _default_inference_resource_context(
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
) -> Optional[Dict[str, Any]]:
    runtime_node_id = str(os.getenv("MEDFLOW_RESOURCE_NODE_ID") or "").strip()
    resource_group_id = str(resource_group_id or _current_resource_group_id() or "").strip()
    training_pool_id = str(training_pool_id or _current_training_pool_id() or "").strip()
    if not runtime_node_id:
        return None
    context: Dict[str, Any] = {"runtime_node_id": runtime_node_id}
    if resource_group_id:
        context["resource_group_id"] = resource_group_id
    if training_pool_id:
        context["training_pool_id"] = training_pool_id
    return context


def _inference_request_resource_context(
    command: str,
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
) -> Optional[Dict[str, Any]]:
    prepared = _prepare_inference_resource_context(
        command,
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
    )
    
    if prepared:
        return prepared
    return _default_inference_resource_context(
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
    )

def _prepare_inference_resource_context(
    command: str,
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
) -> Optional[Dict[str, Any]]:
    resource_group_id = str(resource_group_id or _current_resource_group_id() or "").strip()
    training_pool_id = str(training_pool_id or _current_training_pool_id() or "").strip()
    if not resource_group_id or not _inference_command_needs_resource_context(command):
        return None

    runtime_node_id = str(os.getenv("MEDFLOW_RESOURCE_NODE_ID") or "").strip()
    if not runtime_node_id:
        raise RuntimeError("资源池已启用但缺少 MEDFLOW_RESOURCE_NODE_ID，无法为推理服务申请资源")

    gpus_per_node = _inference_gpus_per_node()
    request_data: Dict[str, Any] = {
        "groupId": resource_group_id,
        "runtimeNodeId": runtime_node_id,
        "nodeCount": 1,
        "gpusPerNode": gpus_per_node,
        "taskCategory": "inference",
        "taskType": "inference",
        "taskTypeText": "推理服务",
    }
    if training_pool_id:
        request_data["poolId"] = training_pool_id
    gpu_snapshot = _local_gpu_snapshot_for_runtime_request()
    if gpu_snapshot:
        request_data["runtimeGpuSnapshot"] = gpu_snapshot

    allocation = _runtime_training_resource_request("reserveTrainingResourcesForRuntime", request_data)
    if not isinstance(allocation, dict):
        raise RuntimeError("推理资源申请返回格式无效")

    nodes = allocation.get("nodes") if isinstance(allocation.get("nodes"), list) else []
    node_payload = nodes[0] if nodes and isinstance(nodes[0], dict) else {}
    assigned_gpus = node_payload.get("gpuIndexes") or allocation.get("gpuIndexes") or []
    assigned_gpus = [str(gpu) for gpu in assigned_gpus if str(gpu).strip() != ""]
    reservation_id = str(
        allocation.get("reservationId")
        or allocation.get("reservation_id")
        or node_payload.get("reservationId")
        or ""
    ).strip()
    if not reservation_id or not assigned_gpus:
        raise RuntimeError("推理资源申请失败：未返回 reservation 或 GPU 分配")

    resource_context = {
        "resource_group_id": resource_group_id,
        "training_pool_id": training_pool_id,
        "reservation_id": reservation_id,
        "runtime_node_id": str(node_payload.get("runtimeNodeId") or runtime_node_id),
        "assigned_gpus": assigned_gpus,
        "cuda_visible_devices": ",".join(assigned_gpus),
        "tensor_parallel_size": len(assigned_gpus),
        "gpus_per_node": gpus_per_node,
        "expires_at": allocation.get("expiresAt") or allocation.get("expires_at"),
        "nodes": nodes,
    }
    with INFERENCE_RESOURCE_RESERVATIONS_LOCK:
        INFERENCE_RESOURCE_RESERVATIONS[reservation_id] = resource_context
    return resource_context


def _inference_reservation_heartbeat_seconds() -> int:
    try:
        return max(15, int(os.getenv("MEDFLOW_INFERENCE_RESERVATION_HEARTBEAT_SECONDS", "60")))
    except ValueError:
        return 60


def _inference_reservation_max_heartbeat_failures() -> int:
    try:
        return max(1, int(os.getenv("MEDFLOW_INFERENCE_RESERVATION_MAX_HEARTBEAT_FAILURES", "3")))
    except ValueError:
        return 3


def _stop_inference_resource_heartbeat(reservation_id: str) -> None:
    reservation_id = str(reservation_id or "").strip()
    if not reservation_id:
        return
    with INFERENCE_RESOURCE_HEARTBEATS_LOCK:
        stop_event = INFERENCE_RESOURCE_HEARTBEATS.pop(reservation_id, None)
    if stop_event:
        stop_event.set()


def _start_inference_resource_heartbeat(resource_context: Optional[Dict[str, Any]]) -> None:
    if not isinstance(resource_context, dict):
        return
    reservation_id = str(resource_context.get("reservation_id") or "").strip()
    runtime_node_id = str(resource_context.get("runtime_node_id") or "").strip()
    if not reservation_id or not runtime_node_id:
        return
    with INFERENCE_RESOURCE_HEARTBEATS_LOCK:
        if reservation_id in INFERENCE_RESOURCE_HEARTBEATS:
            return
        stop_event = threading.Event()
        INFERENCE_RESOURCE_HEARTBEATS[reservation_id] = stop_event

    interval = _inference_reservation_heartbeat_seconds()
    max_failures = _inference_reservation_max_heartbeat_failures()

    def heartbeat() -> None:
        failures = 0
        try:
            while not stop_event.wait(interval):
                with INFERENCE_RESOURCE_RESERVATIONS_LOCK:
                    active_context = INFERENCE_RESOURCE_RESERVATIONS.get(reservation_id)
                if active_context is not resource_context and active_context != resource_context:
                    return
                try:
                    renewal = _runtime_training_resource_request(
                        "renewTrainingResourcesForRuntime",
                        {
                            "reservationId": reservation_id,
                            "runtimeNodeId": runtime_node_id,
                        },
                    )
                    if isinstance(renewal, dict):
                        resource_context["expires_at"] = renewal.get("expiresAt") or renewal.get("expires_at") or resource_context.get("expires_at")
                    failures = 0
                except Exception:
                    failures += 1
                    logger.exception("Failed to renew inference GPU reservation %s", reservation_id)
                    if failures >= max_failures:
                        logger.error(
                            "Releasing inference GPU reservation after %s consecutive renewal failures",
                            failures,
                        )
                        _release_inference_resource_context(resource_context)
                        return
        finally:
            with INFERENCE_RESOURCE_HEARTBEATS_LOCK:
                existing = INFERENCE_RESOURCE_HEARTBEATS.get(reservation_id)
                if existing is stop_event:
                    INFERENCE_RESOURCE_HEARTBEATS.pop(reservation_id, None)

    threading.Thread(
        target=heartbeat,
        name=f"medflow-inference-reservation-{reservation_id[:8]}",
        daemon=True,
    ).start()

def _release_inference_resource_context(resource_context: Optional[Dict[str, Any]]) -> None:
    if not isinstance(resource_context, dict):
        return
    reservation_id = str(resource_context.get("reservation_id") or "").strip()
    if not reservation_id:
        return
    _stop_inference_resource_heartbeat(reservation_id)
    with INFERENCE_RESOURCE_RESERVATIONS_LOCK:
        INFERENCE_RESOURCE_RESERVATIONS.pop(reservation_id, None)
    try:
        _release_resource_allocation(reservation_id, _inference_resource_env(str(resource_context.get("runtime_node_id") or "")))
    except Exception:
        logger.exception("Failed to release inference reservation %s", reservation_id)


def _release_known_inference_reservations() -> None:
    with INFERENCE_RESOURCE_RESERVATIONS_LOCK:
        contexts = list(INFERENCE_RESOURCE_RESERVATIONS.values())
        INFERENCE_RESOURCE_RESERVATIONS.clear()
    for resource_context in contexts:
        _release_inference_resource_context(resource_context)

def _current_training_container() -> str:
    return REQUEST_TRAINING_CONTAINER.get() or DEFAULT_DOCKER_CONTAINER


def _current_evaluation_container() -> str:
    return REQUEST_EVALUATION_CONTAINER.get() or DEFAULT_EVALUATE_DOCKER_CONTAINER


def _current_grpo_container() -> str:
    return REQUEST_GRPO_CONTAINER.get() or _current_training_container()


def _current_multinode_training_container() -> str:
    return REQUEST_MULTINODE_TRAINING_CONTAINER.get() or MULTINODE_DOCKER_CONTAINER


def _current_resource_group_id() -> str:
    return REQUEST_RESOURCE_GROUP_ID.get() or os.getenv("MEDFLOW_RESOURCE_GROUP_ID", "")


def _current_training_pool_id() -> str:
    return REQUEST_TRAINING_POOL_ID.get() or os.getenv("MEDFLOW_TRAINING_POOL_ID", "")



def _is_grpo_training_query(script_query: str) -> bool:
    normalized = str(script_query or "").lower()
    return any(keyword in normalized for keyword in ["grpo", "grpo_train"])


def _is_special_training_query(script_query: str) -> bool:
    normalized = str(script_query or "").lower()
    return any(keyword in normalized for keyword in ["grpo", "多机", "multinode"])


def _is_multinode_training_query(script_query: str) -> bool:
    normalized = str(script_query or "").lower()
    return any(
        keyword in normalized
        for keyword in [
            "train_multinode_sft_pipeline",
            "train_multinode_dpo_pipeline",
            "多机",
            "双机",
            "multinode",
            "multi-node",
            "2node",
            "two-node",
        ]
    )


def _owner_aliases_from_any(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").split(",")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _current_user_role() -> str:
    return REQUEST_USER_ROLE.get().strip().lower()


def _current_inference_owner_user_id() -> str:
    return REQUEST_INFERENCE_OWNER_USER_ID.get().strip()


def _current_inference_owner_aliases() -> list[str]:
    return list(REQUEST_INFERENCE_OWNER_ALIASES.get() or ())


def _inference_owner_payload(default_user_id: str = "") -> tuple[str, list[str]]:
    owner_user_id = _current_inference_owner_user_id() or str(default_user_id or "").strip()
    aliases = _owner_aliases_from_any(_current_inference_owner_aliases())
    if owner_user_id.startswith("auth:"):
        aliases.append(owner_user_id.split(":", 1)[1])
    if default_user_id:
        aliases.append(str(default_user_id).split("#", 1)[0].strip())
    return owner_user_id, _owner_aliases_from_any(aliases)


def _workflow_inference_owner_kwargs(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context = context or {}
    owner_user_id = str(context.get("inference_owner_user_id") or "").strip()
    owner_aliases = _owner_aliases_from_any(context.get("inference_owner_aliases"))
    if not owner_user_id:
        owner_user_id, current_aliases = _inference_owner_payload()
        owner_aliases = owner_aliases or current_aliases
    return {
        "owner_user_id": owner_user_id or "workflow-manager",
        "owner_aliases": owner_aliases,
    }
@wraps(run_script_by_name_data)
def run_group_data(*args, **kwargs):
    container = _current_training_container()
    env_vars = dict(kwargs.get("env_vars") or {})
    env_vars.setdefault("container", container)
    kwargs["env_vars"] = env_vars
    response = run_script_by_name_data(*args, **kwargs)
    script_query = str(kwargs.get("script_query") or (args[0] if args else ""))
    if any(keyword in script_query.lower() for keyword in ["preprocess", "preprocessing", "data_preprocessing", "预处理"]):
        response_text = _tool_response_text(response)

        stats = {
            key: int(value)
            for key, value in re.findall(
                r"'(processed_files|input_items|valid_items)'\s*:\s*(\d+)",
                response_text,
            )
        }
        if stats and (stats.get("processed_files", 0) == 0 or stats.get("valid_items", 0) == 0):
            failure_text = (
                "数据预处理未生成可用数据，已停止后续流程，不会进入高级筛选。"
                f"处理文件数：{stats.get('processed_files', 0)}，"
                f"有效数据数：{stats.get('valid_items', 0)}。"
                "请检查输入目录和数据格式后重试。"
            )
            return ToolResponse(content=[TextBlock(type="text", text=failure_text)],
                metadata={
                    "success": False,
                    "protocol_hint": {
                        "type": "job_failed",
                        "agent": "dataprocessor",
                        "message": failure_text,
                        "jobType": "data_preprocess",
                        "script": "data_preprocessing",
                        "errorReason": "no_preprocessed_data",
                        "errorRecoverable": True,
                        **stats,
                    },
                },
            )
    return response


@wraps(run_script_by_name_data_collect)
def run_group_data_collect(*args, **kwargs):
    container = _current_training_container()
    env_vars = dict(kwargs.get("env_vars") or {})
    env_vars.setdefault("container", container)
    kwargs["env_vars"] = env_vars
    return run_script_by_name_data_collect(*args, **kwargs)


@wraps(run_script_by_name_train)
def run_group_train(script_query: str, *args, **kwargs):
    env_vars = dict(kwargs.get("env_vars") or {})
    resource_group_id = _current_resource_group_id().strip()
    if resource_group_id:
        env_vars["MEDFLOW_RESOURCE_GROUP_ID"] = resource_group_id
    training_pool_id = _current_training_pool_id().strip()
    if training_pool_id:
        env_vars["MEDFLOW_TRAINING_POOL_ID"] = training_pool_id
    kwargs["env_vars"] = env_vars
    if _is_grpo_training_query(script_query):
        container = _current_grpo_container()
        env_vars.setdefault("container", container)
        additional_args = dict(kwargs.get("additional_args") or {})
        additional_args.setdefault("container", container)
        kwargs["additional_args"] = additional_args
    elif _is_multinode_training_query(script_query):
        container = _current_multinode_training_container()
        env_vars.setdefault("container", container)
        additional_args = dict(kwargs.get("additional_args") or {})
        additional_args.setdefault("container", container)
        kwargs["additional_args"] = additional_args
    elif not _is_special_training_query(script_query):
        container = _current_training_container()
        env_vars.setdefault("container", container)
        additional_args = dict(kwargs.get("additional_args") or {})
        additional_args.setdefault("container", container)
        kwargs["additional_args"] = additional_args
    return run_script_by_name_train(script_query, *args, **kwargs)


@wraps(run_script_by_name_assessment)
def run_group_assessment(script_query: str, *args, **kwargs):
    env_vars = dict(kwargs.get("env_vars") or {})
    resource_group_id = _current_resource_group_id().strip()
    if resource_group_id:
        env_vars["MEDFLOW_RESOURCE_GROUP_ID"] = resource_group_id
    training_pool_id = _current_training_pool_id().strip()
    if training_pool_id:
        env_vars["MEDFLOW_TRAINING_POOL_ID"] = training_pool_id
    env_vars.setdefault("container", _current_training_container())
    kwargs["env_vars"] = env_vars
    return run_script_by_name_assessment(script_query, *args, **kwargs)


@wraps(run_script_by_name_assessment_monitor)
def run_group_assessment_monitor(script_query: str = "assessment_monitor", *args, **kwargs):
    kwargs.setdefault("container_name", _current_training_container())
    return run_script_by_name_assessment_monitor(script_query, *args, **kwargs)


# Legacy aliases retained for existing callers and persisted workflows.
run_group_evaluate = run_group_assessment
run_group_evaluate_monitor = run_group_assessment_monitor


MULTINODE_DOCKER_CONTAINER = os.getenv("MULTINODE_DOCKER_CONTAINER", "")
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ACKNOWLEDGEMENT_REPLY_DELAY_SECONDS = 4.5
PROTOCOL_VERSION = "1.0"
PROTOCOL_CONFIDENCE_BY_SOURCE = {
    "tool_hint": 0.95,
    "rule": 0.86,
    "llm_classifier": 0.78,
    "restored": 0.72,
    "fallback": 0.5,
}
_WORKFLOW_MANAGER: Optional[WorkflowManager] = None
_WORKFLOW_MANAGER_LOCK = threading.Lock()
_WORKFLOW_EVENT_CONSUMER_TASK: Optional[asyncio.Task] = None
_WORKFLOW_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_WORKFLOW_EVENT_CONSUMER_ID = f"agent3-{os.getpid()}-{int(time.time())}"


def _workflow_inference_has_async_start_signal(text: str) -> bool:
    import re
    #deepseek
    # 匹配“提交”（无论是否带成功） 或 “启动中/后台启动” 或 “执行结果”
    pattern = r"(重启|启动).*提交|正在.*启动|后台启动|执行结果"
    if re.search(pattern, (text or "")):
        return True
    return any(
        keyword in (text or "")
        for keyword in ["任务已启动", "已启动", "启动进度", "当前状态如下"]
    )


def _workflow_training_container(workflow: Dict[str, Any]) -> str:
    context = workflow.get("context") or {}
    explicit = str(context.get("container") or "").strip()
    if explicit:
        return explicit
    if context.get("launch_mode") == "multinode":
        return (
            str(context.get("multinode_training_container") or "").strip()
            or _current_multinode_training_container()
        )
    if str(context.get("train_type") or "").lower() == "grpo":
        return str(context.get("grpo_container") or "").strip() or _current_grpo_container()
    return str(context.get("training_container") or "").strip() or _current_training_container()


def _workflow_evaluation_container(workflow: Dict[str, Any]) -> str:
    context = workflow.get("context") or {}
    return str(context.get("evaluation_container") or "").strip() or _current_evaluation_container()


def _workflow_single_model_evaluation_container(workflow: Dict[str, Any]) -> str:
    """Single-model evaluation consumes the trained artifact in the training container."""
    return _workflow_training_container(workflow)


def _workflow_safe_log_name(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120] or "workflow"


def _workflow_stage_log_path(workflow: Dict[str, Any], stage_name: str) -> Path:
    workflow_id = _workflow_safe_log_name(workflow.get("workflow_id"))
    stage = _workflow_safe_log_name(stage_name)
    return WORKFLOW_LOG_DIR / workflow_id / f"{stage}.log"


def _workflow_tail_file(path: Path, max_chars: int = 6000) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars))
            return handle.read().decode("utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _workflow_stage_log_fields(workflow: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    path = _workflow_stage_log_path(workflow, stage_name)
    fields: Dict[str, Any] = {
        "log_path": str(path),
        "log_command": f"查看一键工作流{stage_name}阶段日志",
    }
    tail = _workflow_tail_file(path)
    if tail:
        fields["log_tail"] = tail
    if path.exists():
        fields["log_updated_at"] = path.stat().st_mtime
    return fields


def _append_workflow_stage_log(
    workflow: Dict[str, Any],
    stage_name: str,
    title: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = _workflow_stage_log_path(workflow, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "workflow_id": workflow.get("workflow_id"),
        "stage": stage_name,
        "title": title,
        "payload": payload or {},
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str))
        handle.write("\n")
    return _workflow_stage_log_fields(workflow, stage_name)


def _workflow_inference_has_failure_signal(text: str) -> bool:
    if _workflow_inference_has_async_start_signal(text):
        return False
    return any(keyword in (text or "") for keyword in ["失败", "超时", "未能正常启动"])

def _workflow_inference_structured_failure(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return True
    status = str(payload.get("status") or payload.get("tool_status") or "").strip().lower()
    if status in {"failed", "failure", "error", "timeout"}:
        return True
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    protocol_type = str(protocol.get("type") or "")
    protocol_status = str(protocol.get("status") or "").strip().lower()
    return protocol_type in {"job_failed", "error"} or protocol_status in {"failed", "failure", "error", "timeout"}


def _workflow_benchmark_has_not_found_signal(text: str) -> bool:
    content = str(text or "")
    lowered = content.lower()
    chinese_patterns = (
        r"没有正在运行或已完成",
        r"没有运行过",
        r"没有.*记录",
        r"不存在",
        r"暂无任务",
    )
    if any(re.search(pattern, content) for pattern in chinese_patterns):
        return True
    english_patterns = (
        r"\bnot\s+found\b",
        r"\bno\b.*\bbenchmark\b",
        r"\bno\b.*\brecords?\b",
        r"\bno\b.*\btasks?\b",
    )
    return any(re.search(pattern, lowered) for pattern in english_patterns)


def _workflow_benchmark_explicit_status(text: str) -> Optional[str]:
    """Prefer explicit benchmark status fields over incidental words."""
    content = str(text or "")
    if _workflow_benchmark_has_not_found_signal(content):
        return "not_found"
    status_lines: List[str] = []
    for pattern in (
        r"(?:当前状态|任务状态|benchmark_status|status)\s*[:：=]\s*([^\n\r]+)",
        r"当前状态\s*[`'\"]?([^`\n\r'\"]+)",
    ):
        status_lines.extend(match.group(1) for match in re.finditer(pattern, content, re.IGNORECASE))

    for line in status_lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in ["failed", "failure"]) or any(keyword in line for keyword in ["失败", "异常", "报错"]):
            return "failed"
        if any(keyword in lowered for keyword in ["running", "processing"]) or any(keyword in line for keyword in ["运行中", "正在运行", "处理中"]):
            return "running"
        if any(keyword in lowered for keyword in ["finished", "completed", "done"]) or any(keyword in line for keyword in ["已完成", "完成", "结束"]):
            return "finished"
        if "stopped" in lowered or any(keyword in line for keyword in ["已停止", "已终止", "停止"]):
            return "stopped"
    return None


def keep_think_in_context() -> bool:
    value = os.getenv("AGENT_KEEP_THINK_CONTEXT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def strip_think_for_context(text: str) -> str:
    if keep_think_in_context():
        return text or ""
    return THINK_TAG_RE.sub("", text or "").strip()


def _inference_tool_payload_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(text or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("__inference_tool_result__") is not True:
        return None
    return payload


def _workflow_response_payload(response: ToolResponse) -> Dict[str, Any]:
    metadata = response.metadata or {}
    hint = metadata.get("protocol_hint") or {}
    text = _tool_response_text(response)

    payload: Dict[str, Any] = {}
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            payload = decoded
    except (TypeError, ValueError):
        pass
    status = payload.get("status") or hint.get("status")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return {
        "status": status,
        "pid": hint.get("pid") or metrics.get("pid"),
        "container": hint.get("container") or metrics.get("container_name"),
        "error": metrics.get("error_detail") or hint.get("errorReason") or metrics.get("error_reason"),
        "metrics": metrics,
        "message": hint.get("message") or text,
    }


def _workflow_evaluate_result_matches_workflow(
    workflow: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Return True only when monitor evidence belongs to this workflow model."""
    context = workflow.get("context") or {}
    target_path = str(context.get("trained_model_path") or "").strip()
    target_name = os.path.basename(target_path.rstrip("/"))
    if not target_name:
        return True

    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    monitor_model_name = str(metrics.get("model_name") or "").strip()
    if monitor_model_name and monitor_model_name != target_name:
        return False

    stages = metrics.get("stages") if isinstance(metrics.get("stages"), list) else []
    target_stage_finished = any(
        isinstance(stage, dict)
        and stage.get("status") == "finished"
        and target_name in str(stage.get("path") or "")
        for stage in stages
    )
    if target_stage_finished:
        return True

    report_text_matches = any(
        isinstance(stage, dict)
        and stage.get("name") == "evaluation_report"
        and target_name in str(stage.get("result") or "")
        for stage in stages
    )
    if report_text_matches:
        return True

    # A generic report path without model-specific artifacts is stale for this
    # workflow; do not let it skip or finish the evaluation stage.
    return False


def _workflow_publish_marker_matches(
    workflow: Dict[str, Any],
    destination: str,
    inference_container: str,
) -> bool:
    """Verify an existing published directory was produced for this workflow source."""
    source = str((workflow.get("context") or {}).get("trained_model_path") or "").strip()
    if not source or not destination or not inference_container:
        return False

    marker_path = f"{destination.rstrip('/')}/.workflow_publish_source.json"
    marker_process = subprocess.run(
        ["docker", "exec", inference_container, "cat", marker_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if marker_process.returncode != 0:
        return False
    try:
        marker = json.loads(marker_process.stdout or "{}")
    except (TypeError, ValueError):
        return False
    return (
        marker.get("workflow_id") == workflow.get("workflow_id")
        and marker.get("source_model_path") == source
        and marker.get("source_model_name") == os.path.basename(source.rstrip("/"))
    )


def _workflow_train_rebound_matches_workflow(
    workflow: Dict[str, Any],
    rebound: Dict[str, Any],
) -> bool:
    """Check that a no-PID monitor rebound is evidence for this workflow."""
    metrics = rebound.get("metrics") if isinstance(rebound.get("metrics"), dict) else {}
    status = rebound.get("status")
    if status not in {"starting", "running", "finished"}:
        return False

    output_dir = str(metrics.get("output_dir") or "").strip()
    if not output_dir:
        return False

    train_type = (workflow.get("context") or {}).get("train_type") or "lora"
    dataset_ref = str(workflow.get("dataset_ref") or "").strip()
    if train_type in {"lora", "full"} and dataset_ref:
        return os.path.basename(output_dir.rstrip("/")) in _workflow_expected_batch_output_model_names(train_type, dataset_ref)

    return True


def _workflow_start_train(workflow: Dict[str, Any]) -> Dict[str, Any]:
    dataset_ref = workflow["dataset_ref"]
    context = workflow["context"]
    train_type = context.get("train_type") or "lora"
    launch_mode = context.get("launch_mode")
    if train_type == "enhanced":
        script_query = "多机增强训练" if launch_mode == "multinode" else "增强训练"
        additional_args = {
            "model_path": context["model_path"],
            "dataset_dir": context["dataset_dir"],
            "dataset_name": context["dataset_name"],
        }
        train_args = dict(context.get("train_args") or {})
        train_args.pop("RESUME", None)
        if launch_mode != "multinode":
            train_args = {
                key: value
                for key, value in train_args.items()
                if key in {"template"}
            }
        additional_args.update(train_args)
    else:
        script_query = "多机lora批量训练" if launch_mode == "multinode" else "lora批量训练"
        additional_args = {
            "data_identifier": dataset_ref,
            "data_dir": f"/home/workspace/dataset_batch_train/{dataset_ref}",
        }
        additional_args.update(context.get("train_args") or {})
    additional_args["container"] = _workflow_training_container(workflow)

    response = run_script_by_name_train(
        script_query,
        additional_args=additional_args,
        env_vars={
            "MEDFLOW_RESOURCE_GROUP_ID": str(context.get("resource_group_id") or "").strip(),
            "MEDFLOW_TRAINING_POOL_ID": str(context.get("training_pool_id") or "").strip(),
        },
        background=True,
    )
    result = _workflow_response_payload(response)
    if not (response.metadata or {}).get("success", False):
        raise RuntimeError(result.get("message") or "LoRA训练启动失败")
    return result


def _workflow_monitor_train(workflow: Dict[str, Any]) -> Dict[str, Any]:
    stage = workflow["stages"]["train"]
    train_type = workflow["context"].get("train_type") or "lora"
    response = run_script_by_name_monitor1(
        "训练状态",
        container_name=stage.get("container"),
        pid=stage.get("pid"),
        train_type=train_type,
        allow_llm=False,
    )
    result = _workflow_response_payload(response)
    if result.get("error") != "pid_ended_no_wandb":
        return result
    rebound = _workflow_response_payload(
        run_script_by_name_monitor1(
            "训练状态",
            container_name=stage.get("container"),
            train_type=train_type,
            allow_llm=False,
        ),
    )
    metrics = rebound.get("metrics") if isinstance(rebound.get("metrics"), dict) else {}
    if (
        (metrics.get("training_process_exists") or rebound.get("status") == "finished")
        and _workflow_train_rebound_matches_workflow(workflow, rebound)
    ):
        rebound["message"] = f"{rebound.get('message') or ''}\n已从启动PID切换为按训练类型监控。".strip()
        return rebound
    result_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    launch_log_tail = str(result_metrics.get("launch_log_tail") or "").strip()
    if launch_log_tail:
        result["error"] = f"训练准备/启动失败：{launch_log_tail[-2000:]}"
    return result


def _workflow_find_trained_model(workflow: Dict[str, Any]) -> str:
    train_type = workflow["context"].get("train_type") or "lora"
    started_at = float(workflow["context"].get("train_started_at") or workflow["created_at"])
    stage_metrics = workflow["stages"]["train"].get("metrics") or {}
    output_dir = str(stage_metrics.get("output_dir") or "").strip()
    if train_type == "enhanced":
        candidates = _workflow_find_dpo_model_candidates(workflow, started_at, output_dir)
    else:
        candidates = _workflow_find_batch_model_candidates(workflow, started_at, train_type, output_dir)
    if len(candidates) != 1:
        raise RuntimeError(
            "无法唯一确定训练产物，"
            f"检测到 {len(candidates)} 个本次训练启动后的模型目录: {candidates}"
        )
    return candidates[0]


def _workflow_find_dpo_model_candidates(
    workflow: Dict[str, Any],
    started_at: float,
    output_dir: Optional[str] = None,
) -> List[str]:
    export_root = "/home/workspace/models/dpo_train/internal/export"
    save_name = os.path.basename(str(output_dir or "").rstrip("/"))
    if save_name:
        expected_export = f"{export_root}/model_medical_{save_name}"
        if _workflow_docker_model_dir_complete(workflow, expected_export):
            return [expected_export]
        return []

    export_script = (
        f"find {shlex.quote(export_root)} -mindepth 1 -maxdepth 1 -type d "
        f"-name 'model_medical_*' -newermt '@{started_at}' -print"
    )
    process = subprocess.run(
        ["docker", "exec", _workflow_training_container(workflow), "sh", "-c", export_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError(f"无法扫描DPO导出产物: {process.stderr.strip()}")
    return [
        path
        for path in list(dict.fromkeys(line.strip() for line in process.stdout.splitlines() if line.strip()))
        if _workflow_docker_model_dir_complete(workflow, path)
    ]

def _workflow_docker_dir_exists(workflow: Dict[str, Any], path: Optional[str]) -> bool:
    if not path:
        return False
    process = subprocess.run(
        ["docker", "exec", _workflow_training_container(workflow), "test", "-d", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return process.returncode == 0

def _workflow_docker_model_dir_complete(workflow: Dict[str, Any], path: Optional[str]) -> bool:
    if not path:
        return False
    quoted = shlex.quote(str(path))
    script = (
        f"test -d {quoted} && "
        f"test -f {quoted}/config.json && "
        f"find {quoted} -maxdepth 1 -type f "
        "\\( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' "
        "-o -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin.index.json' \\) "
        "| grep -q ."
    )
    process = subprocess.run(
        ["docker", "exec", _workflow_training_container(workflow), "sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return process.returncode == 0

def _workflow_find_batch_model_candidates(
    workflow: Dict[str, Any],
    started_at: float,
    train_type: str,
    output_dir: Optional[str] = None,
) -> List[str]:
    root = "/home/workspace/models/batch_train"
    dataset_ref = str(workflow.get("dataset_ref") or (workflow.get("context") or {}).get("dataset_ref") or "").strip()
    expected_paths: List[str] = []
    output_dir = str(output_dir or "").strip().rstrip("/")
    if output_dir:
        output_name = os.path.basename(output_dir)
        expected_paths.append(output_dir if output_name.endswith("_merged") else f"{output_dir}_merged")
    expected_paths.extend(f"{root}/{name}" for name in _workflow_expected_batch_merged_model_names(train_type, dataset_ref))
    expected_paths = list(dict.fromkeys(path for path in expected_paths if path))
    expected_names = [os.path.basename(path.rstrip("/")) for path in expected_paths]
    exact_matches = [path for path in expected_paths if _workflow_docker_dir_exists(workflow, path)]
    if exact_matches:
        return exact_matches

    script = (
        f"find {shlex.quote(root)} -mindepth 1 -maxdepth 2 -type d "
        f"-name '*_merged' -newermt '@{started_at}' -print"
    )
    process = subprocess.run(
        ["docker", "exec", _workflow_training_container(workflow), "sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError(f"无法扫描训练产物: {process.stderr.strip()}")
    candidates = list(dict.fromkeys(line.strip() for line in process.stdout.splitlines() if line.strip()))
    if expected_names:
        candidates = [
            path
            for path in candidates
            if os.path.basename(path.rstrip("/")) in expected_names
        ]
    return candidates

def _workflow_expected_batch_output_model_names(train_type: str, dataset_ref: str) -> List[str]:
    dataset_ref = str(dataset_ref or "").strip()
    if not dataset_ref:
        return []
    prefix = "model_medical_full" if train_type == "full" else "model_medical_lora"
    return [
        f"{prefix}_{dataset_ref}",
        f"{prefix}_{dataset_ref}_merged",
    ]


def _workflow_expected_batch_merged_model_names(train_type: str, dataset_ref: str) -> List[str]:
    return [
        name
        for name in _workflow_expected_batch_output_model_names(train_type, dataset_ref)
        if name.endswith("_merged")
    ]


def _workflow_expected_lora_model_names(dataset_ref: str) -> List[str]:
    return _workflow_expected_batch_output_model_names("lora", dataset_ref)
def _workflow_find_existing_trained_model(workflow: Dict[str, Any]) -> Optional[str]:
    context = workflow.get("context") or {}
    existing = context.get("trained_model_path")
    train_type = context.get("train_type") or "lora"
    if train_type == "enhanced":
        if _workflow_docker_model_dir_complete(workflow, existing):
            return str(existing)
        output_dir = ((workflow.get("stages") or {}).get("train", {}).get("metrics") or {}).get("output_dir")
        save_name = os.path.basename(str(output_dir or "").rstrip("/"))
        if save_name:
            export_path = f"/home/workspace/models/dpo_train/internal/export/model_medical_{save_name}"
            return export_path if _workflow_docker_model_dir_complete(workflow, export_path) else None
        return None
    if _workflow_docker_dir_exists(workflow, existing):
        return str(existing)
    dataset_ref = str(workflow.get("dataset_ref") or context.get("dataset_ref") or "").strip()
    root = "/home/workspace/models/batch_train"
    process = subprocess.run(
        [
            "docker",
            "exec",
            _workflow_training_container(workflow),
            "sh",
            "-c",
            f"find {shlex.quote(root)} -mindepth 1 -maxdepth 2 -type d -name '*_merged' -print",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        return None
    candidates = list(dict.fromkeys(line.strip() for line in process.stdout.splitlines() if line.strip()))
    if dataset_ref:
        expected_names = _workflow_expected_batch_merged_model_names(train_type, dataset_ref)
        candidates = [
            path
            for path in candidates
            if os.path.basename(path.rstrip("/")) in expected_names
        ]
    if len(candidates) > 1:
        raise RuntimeError(
            "检测到多个已存在的训练产物，无法自动跳过训练，请人工确认模型目录: "
            f"{candidates}"
        )
    return candidates[0] if candidates else None


def _workflow_checkpoint_search_roots(workflow: Dict[str, Any]) -> List[str]:
    train_stage = (workflow.get("stages") or {}).get("train", {})
    metrics = train_stage.get("metrics") if isinstance(train_stage.get("metrics"), dict) else {}
    context = workflow.get("context") or {}
    roots = [str(metrics.get("output_dir") or "").strip()]
    if not roots[0]:
        dataset_ref = str(workflow.get("dataset_ref") or context.get("dataset_ref") or "").strip()
        train_type = str(context.get("train_type") or "lora").strip().lower()
        dataset_name = str(context.get("dataset_name") or "").strip()
        search_roots = (
            ["/home/workspace/models/dpo_train/internal/saves"]
            if train_type == "enhanced"
            else ["/home/workspace/models/batch_train"]
        )
        if train_type not in {"enhanced", "lora"}:
            search_roots = [
                "/home/workspace/models/batch_train",
                "/home/workspace/models/dpo_train/internal/saves",
            ]
        for root in search_roots:
            identifiers = [dataset_name, dataset_ref] if train_type == "enhanced" else [dataset_ref]
            identifiers = list(dict.fromkeys(value for value in identifiers if value))
            if not identifiers or train_type == "enhanced":
                identifiers.append("")
            for identifier in identifiers:
                name_filter = f"-name {shlex.quote(f'*{identifier}*')} " if identifier else ""
                script = (
                    f"find {shlex.quote(root)} -mindepth 1 -maxdepth 2 -type d "
                    f"! -name '*_merged' {name_filter}-print"
                )
                try:
                    process = subprocess.run(
                        ["docker", "exec", _workflow_training_container(workflow), "sh", "-c", script],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                except Exception:
                    process = None
                if process and process.returncode == 0:
                    roots.extend(line.strip() for line in process.stdout.splitlines() if line.strip())
                if any(root for root in roots):
                    break
    return list(dict.fromkeys(root for root in roots if root))


def _workflow_find_latest_checkpoint(workflow: Dict[str, Any]) -> Optional[str]:
    roots = _workflow_checkpoint_search_roots(workflow)
    if not roots:
        return None
    script = (
        "for d in "
        + " ".join(shlex.quote(root) for root in roots)
        + "; do "
        "[ -d \"$d\" ] || continue; "
        "for p in \"$d\"/checkpoint-*; do "
        "[ -d \"$p\" ] && stat -c '%Y\t%n' \"$p\"; "
        "done; "
        "done"
    )
    try:
        process = subprocess.run(
            ["docker", "exec", _workflow_training_container(workflow), "sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if process.returncode != 0:
        return None
    candidates: List[tuple[Optional[int], float, str]] = []
    for line in process.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        mtime_text, _, path = line.partition("\t")
        path = path.strip()
        if not path:
            continue
        try:
            mtime = float(mtime_text)
        except ValueError:
            mtime = 0.0
        step_match = re.search(r"(?:^|/)checkpoint-(\d+)$", path)
        step = int(step_match.group(1)) if step_match else None
        candidates.append((step, mtime, path))
    if not candidates:
        return None
    numeric = [item for item in candidates if item[0] is not None]
    if numeric:
        return max(numeric, key=lambda item: (int(item[0] or 0), item[1]))[2]
    return max(candidates, key=lambda item: item[1])[2]


def _workflow_check_existing_stage_output(workflow: Dict[str, Any], stage_name: str) -> Optional[Dict[str, Any]]:
    context = workflow.get("context") or {}
    if stage_name == "train":
        model_path = _workflow_find_existing_trained_model(workflow)
        if model_path:
            return {
                "status": "finished",
                "trained_model_path": model_path,
                "model_path": model_path,
                "message": f"检测到已存在训练产物，跳过训练：{model_path}",
            }
        return None
    if stage_name == "evaluate" and context.get("trained_model_path"):
        result = _workflow_monitor_evaluate(workflow)
        if result.get("status") == "finished" and _workflow_evaluate_result_matches_workflow(workflow, result):
            return {**result, "status": "finished", "message": "检测到已有单模型评估结果，跳过评估"}
        return None
    if stage_name == "publish":
        trained_model_path = context.get("trained_model_path")
        model_name = os.path.basename(trained_model_path) if trained_model_path else workflow["workflow_id"]
        destination = context.get("published_model_path") or f"{config.workflow.publish_dir.rstrip('/')}/{model_name}"
        # 检查推理容器中是否已存在发布目录
        inference_container = _workflow_evaluation_container(workflow)
        check_process = subprocess.run(
            ["docker", "exec", inference_container, "test", "-d", str(destination)],
            capture_output=True, text=True, timeout=30,
        )
        if check_process.returncode == 0:
            if _workflow_publish_marker_matches(workflow, destination, inference_container):
                return {
                    "status": "finished",
                    "published_model_path": destination,
                    "model_path": destination,
                    "message": f"检测到当前工作流已发布目录，跳过发布：{destination}",
                }
            return None
        return None
    if stage_name == "deploy" and context.get("published_model_path"):
        deployed = _workflow_deployed_model_status(
            context["published_model_path"],
            _workflow_evaluation_container(workflow),
            resource_group_id=str(context.get("resource_group_id") or "").strip(),
            training_pool_id=str(context.get("training_pool_id") or "").strip(),
            **_workflow_inference_owner_kwargs(context),
        )
        if deployed:
            return deployed
        return None
    if stage_name == "benchmark":
        result = _workflow_inference_command(
            _benchmark_status_command(workflow),
            _workflow_evaluation_container(workflow),
            resource_group_id=str(context.get("resource_group_id") or "").strip(),
            training_pool_id=str(context.get("training_pool_id") or "").strip(),
            **_workflow_inference_owner_kwargs(context),
        )
        status = result.get("status")
        if status == "finished":
            result = {**result, "result_entry": _benchmark_result_entry(workflow)}
            return {"status": "finished", "result": result, "message": f"检测到{_benchmark_name(workflow)}基准评测已完成，跳过评测"}
        if status == "running":
            result = {**result, "result_entry": _benchmark_result_entry(workflow)}
            return {"status": "running", "result": result, "message": f"检测到{_benchmark_name(workflow)}基准评测正在运行，跳过重复启动"}
    return None


def _workflow_start_evaluate(workflow: Dict[str, Any]) -> Dict[str, Any]:
    response = run_script_by_name_evaluate(
        "单模型评估",
        additional_args={"model_fir": workflow["context"]["trained_model_path"]},
        env_vars={
            "container": _workflow_single_model_evaluation_container(workflow),
            "MEDFLOW_RESOURCE_GROUP_ID": str(
                workflow["context"].get("resource_group_id") or "",
            ).strip(),
            "MEDFLOW_TRAINING_POOL_ID": str(
                workflow["context"].get("training_pool_id") or "",
            ).strip(),
        },
        background=True,
    )
    result = _workflow_response_payload(response)
    if not (response.metadata or {}).get("success", False):
        raise RuntimeError(result.get("message") or "单模型评估启动失败")
    return result


def _workflow_monitor_evaluate(workflow: Dict[str, Any]) -> Dict[str, Any]:
    stage = workflow["stages"]["evaluate"]
    response = run_script_by_name_evaluate_monitor(
        "single_model_evaluation_vpn",
        container_name=stage.get("container") or _workflow_single_model_evaluation_container(workflow),
        pid=stage.get("pid"),
        model_fir=workflow["context"]["trained_model_path"],
    )
    result = _workflow_response_payload(response)
    if result.get("status") == "finished" and not _workflow_evaluate_result_matches_workflow(workflow, result):
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        metrics = {
            **metrics,
            "stale_evaluation_result_ignored": True,
            "expected_model_name": os.path.basename(str(workflow["context"]["trained_model_path"]).rstrip("/")),
        }
        return {
            **result,
            "status": "running" if stage.get("pid") or metrics.get("pid_alive") else "unknown",
            "message": "检测到已有评估报告，但未匹配当前工作流模型，继续等待当前模型评估结果",
            "metrics": metrics,
        }
    return result


def _workflow_start_publish(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """启动后台异步复制，返回进程信息。"""
    source = workflow["context"]["trained_model_path"]
    model_name = os.path.basename(source)
    destination = f"{config.workflow.publish_dir.rstrip('/')}/{model_name}"
    inference_container = _workflow_evaluation_container(workflow)
    workflow_id = workflow["workflow_id"]
    marker_json = json.dumps(
        {
            "workflow_id": workflow_id,
            "source_model_path": source,
            "source_model_name": model_name,
        },
        ensure_ascii=False,
    )

    temp_dir = f"/tmp/workflow_publish_{workflow_id}_{model_name}"
    progress_file = f"/tmp/workflow_publish_{workflow_id}_progress"
    script_path = f"/tmp/workflow_publish_{workflow_id}.sh"

    # 创建后台复制脚本
    script_content = (
        f"#!/bin/bash\n"
        f"set -e\n"
        f'PROGRESS_FILE="{progress_file}"\n'
        f'TEMP_DIR="{temp_dir}"\n'
        f'DESTINATION="{destination}"\n'
        f'INFERENCE_CONTAINER="{inference_container}"\n'
        f'SOURCE="{source}"\n'
        f'TRAIN_CONTAINER="{_workflow_training_container(workflow)}"\n\n'
        f'write_progress() {{\n'
        f'  local percent="$1"\n'
        f'  local step="$2"\n'
        f'  local step_name="$3"\n'
        f'  local message="$4"\n'
        f'  {{\n'
        f'    echo "percent=${{percent}}"\n'
        f'    echo "current_step=${{step}}"\n'
        f'    echo "total_steps=4"\n'
        f'    echo "step_name=${{step_name}}"\n'
        f'    echo "message=${{message}}"\n'
        f'    echo "source=${{SOURCE}}"\n'
        f'    echo "destination=${{DESTINATION}}"\n'
        f'    echo "temp_dir=${{TEMP_DIR}}"\n'
        f'    echo "train_container=${{TRAIN_CONTAINER}}"\n'
        f'    echo "inference_container=${{INFERENCE_CONTAINER}}"\n'
        f'  }} > "$PROGRESS_FILE"\n'
        f'}}\n\n'
        f'rm -rf "$TEMP_DIR"\n'
        f'write_progress 0 1 "copy_to_host" "正在复制模型到宿主机"\n\n'
        f'# 步骤1: 训练容器 → 宿主机\n'
        f'docker cp "$TRAIN_CONTAINER:$SOURCE" "$TEMP_DIR"\n'
        f'printf %s {shlex.quote(marker_json)} > "$TEMP_DIR/.workflow_publish_source.json"\n'
        f'write_progress 45 2 "prepare_destination" "正在准备推理容器目标目录"\n\n'
        f'# 步骤2: 宿主机 → 推理容器\n'
        f'docker exec "$INFERENCE_CONTAINER" mkdir -p "$(dirname "$DESTINATION")"\n'
        f'docker exec "$INFERENCE_CONTAINER" rm -rf "${{DESTINATION}}.tmp"\n'
        f'write_progress 55 3 "copy_to_inference" "正在复制模型到推理容器"\n'
        f'docker cp "$TEMP_DIR" "$INFERENCE_CONTAINER:${{DESTINATION}}.tmp"\n'
        f'write_progress 90 4 "activate_model" "正在完成发布"\n\n'
        f'# 步骤3: 原子性重命名\n'
        f'docker exec "$INFERENCE_CONTAINER" mv "${{DESTINATION}}.tmp" "$DESTINATION"\n'
        f'rm -rf "$TEMP_DIR"\n\n'
        f'write_progress 100 4 "finished" "发布完成"\n'
    )

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    # 启动后台进程
    process = subprocess.Popen(
        ["nohup", "bash", script_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    return {
        "pid": str(process.pid),
        "container": "host",
        "model_path": destination,
        "model_name": model_name,
        "progress_file": progress_file,
        "progress_percent": 0,
        "message": "正在复制模型到宿主机",
        "metrics": {
            "progress_percent": 0,
            "current_step": 1,
            "total_steps": 4,
            "step_name": "copy_to_host",
            "source": source,
            "destination": destination,
            "temp_dir": temp_dir,
            "train_container": _workflow_training_container(workflow),
            "inference_container": inference_container,
            "message": "正在复制模型到宿主机",
        },
    }


def _workflow_parse_publish_progress(content: str) -> Dict[str, Any]:
    """Parse publish progress, keeping compatibility with legacy percent:message."""
    stripped = (content or "").strip()
    if not stripped:
        return {"progress_percent": 0, "message": "等待复制开始"}
    if "\n" not in stripped and ":" in stripped:
        percent, message = stripped.split(":", 1)
        return {"progress_percent": int(percent), "message": message}

    fields: Dict[str, str] = {}
    for line in stripped.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    parsed: Dict[str, Any] = {
        "progress_percent": int(fields.get("percent") or fields.get("progress_percent") or 0),
        "message": fields.get("message") or "等待复制开始",
    }
    for key in (
        "step_name", "source", "destination", "temp_dir",
        "train_container", "inference_container",
    ):
        if fields.get(key):
            parsed[key] = fields[key]
    for key in ("current_step", "total_steps"):
        if fields.get(key):
            parsed[key] = int(fields[key])
    return parsed


def _workflow_monitor_publish(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """检查复制进度。"""
    stage = workflow["stages"]["publish"]
    pid = str(stage.get("pid") or "").strip()
    progress_file = stage.get("progress_file")
    model_path = stage.get("model_path") or ""

    # 检查进程是否还在运行
    pid_alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            pid_alive = True
        except (OSError, ValueError):
            pass

    # 读取进度
    progress: Dict[str, Any] = {"progress_percent": 0, "message": "等待复制开始"}
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            progress = _workflow_parse_publish_progress(content)
        except (ValueError, OSError):
            pass
    progress_percent = int(progress.get("progress_percent") or 0)
    message = str(progress.get("message") or "等待复制开始")
    metrics = dict(progress)

    if not pid_alive:
        if progress_percent >= 100:
            return {
                "status": "finished",
                "model_path": model_path,
                "progress_percent": 100,
                "message": message or "发布完成",
                "metrics": {**metrics, "progress_percent": 100},
            }
        else:
            return {
                "status": "failed",
                "error": f"复制进程异常退出，当前进度: {progress_percent}%",
                "progress_percent": progress_percent,
                "message": message,
                "metrics": metrics,
            }

    return {
        "status": "running",
        "progress_percent": progress_percent,
        "message": message,
        "metrics": metrics,
    }


def _workflow_inference_model_path(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).upper() == "MODEL_PATH" and value:
                return str(value)
            nested = _workflow_inference_model_path(value)
            if nested:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _workflow_inference_model_path(value)
            if nested:
                return nested
    return None


def _workflow_inference_model_name(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).upper() == "MODEL_NAME" and value is not None:
                return str(value)
            nested = _workflow_inference_model_name(value)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _workflow_inference_model_name(value)
            if nested is not None:
                return nested
    return None


def _workflow_model_identifier(value: Optional[str]) -> str:
    return os.path.basename(str(value or "").rstrip("/")).strip()


def _normalize_benchmark_dataset_name(value: Any) -> str:
    text = os.path.basename(str(value or "").strip()).lower()
    if text.endswith(".json"):
        text = text[:-5]
    return text


def _workflow_benchmark_dataset_aliases(benchmark: str) -> List[str]:
    normalized = _normalize_benchmark_dataset_name(benchmark)
    return sorted(BENCHMARK_DATASET_ALIASES.get(normalized, {normalized}))


def _workflow_inference_matches_target(current: Dict[str, Any], target_model_path: str) -> bool:
    target_path = str(target_model_path or "").rstrip("/")
    target_name = _workflow_model_identifier(target_path)
    current_path = str(current.get("model_path") or "").rstrip("/")
    current_name = _workflow_model_identifier(current.get("model_name") or current_path)
    return bool(
        target_name
        and (
            current_name == target_name
            or bool(current_path and target_path and current_path == target_path)
        )
    )


def _workflow_deployed_model_status(
    target_model_path: str,
    evaluation_container: Optional[str] = None,
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
    owner_user_id: str = "",
    owner_aliases: Any = None,
) -> Optional[Dict[str, Any]]:
    inference_kwargs = {}
    if owner_user_id or owner_aliases:
        inference_kwargs.update({
            "owner_user_id": owner_user_id,
            "owner_aliases": owner_aliases,
        })
    if resource_group_id or training_pool_id:
        inference_kwargs.update({
            "resource_group_id": resource_group_id,
            "training_pool_id": training_pool_id,
        })
    current = _workflow_inference_command("查看推理配置", evaluation_container, **inference_kwargs)
    checked = _workflow_inference_command("查看推理服务状态", evaluation_container, **inference_kwargs)
    if not checked.get("all_running"):
        return None
    if not _workflow_inference_matches_target(current, target_model_path):
        return None
    model_name = (
        _workflow_model_identifier(checked.get("model_name") or checked.get("model_path"))
        or _workflow_model_identifier(current.get("model_name") or current.get("model_path"))
        or _workflow_model_identifier(target_model_path)
    )
    return {
        "status": "finished",
        "model_name": model_name,
        "model_path": target_model_path,
        "service_checked": True,
        "all_running": True,
        "service_log_command": "查看推理服务日志",
        "message": "当前推理服务已部署目标模型，跳过部署",
    }


def _workflow_deploy_service_status(workflow: Dict[str, Any]) -> Dict[str, Any]:
    context = workflow.get("context") or {}
    stage = (workflow.get("stages") or {}).get("deploy", {})
    target_model_path = str(context.get("published_model_path") or "").strip()
    next_check_at = float(stage.get("next_service_check_at") or 0)
    if next_check_at and time.time() < next_check_at:
        return {
            "status": "running",
            "model_path": target_model_path,
            "model_name": _workflow_model_identifier(target_model_path),
            "service_checked": False,
            "all_running": False,
            "service_log_command": "查看推理服务日志",
            "deployment_step": "waiting_service",
            "deploy_progress": "4/4",
            "next_service_check_at": next_check_at,
            "message": "推理服务重启命令已提交，等待下一次服务状态检查",
            **_workflow_stage_log_fields(workflow, "deploy"),
        }
    checked = _workflow_inference_command(
        "查看推理服务状态",
        _workflow_evaluation_container(workflow),
        resource_group_id=str(context.get("resource_group_id") or "").strip(),
        training_pool_id=str(context.get("training_pool_id") or "").strip(),
            **_workflow_inference_owner_kwargs(context),
    )
    all_running = checked.get("all_running") is True
    payload = {
        "model_path": target_model_path,
        "model_name": _workflow_model_identifier(target_model_path),
        "service_checked": True,
        "all_running": all_running,
        "service_log_command": "查看推理服务日志",
        "deployment_step": "service_checked" if all_running else "waiting_service",
        "deploy_progress": "4/4",
        "message": "推理服务状态查询完成" if all_running else "服务重启任务已启动，等待推理服务全部运行",
    }
    log_fields = _append_workflow_stage_log(
        workflow,
        "deploy",
        "部署监控查询推理服务状态",
        {**payload, "status": checked.get("status")},
    )
    if all_running:
        return {"status": "finished", **payload, **log_fields}
    return {"status": "running", **payload, **log_fields}


def _workflow_monitor_deploy(workflow: Dict[str, Any]) -> Dict[str, Any]:
    context = workflow.get("context") or {}
    target_model_path = str(context.get("published_model_path") or "").strip()
    if not target_model_path:
        return {
            "status": "running",
            "all_running": False,
            "message": "等待发布模型路径写入后继续检查推理服务",
            **_workflow_stage_log_fields(workflow, "deploy"),
        }
    return _workflow_deploy_service_status(workflow)


def _workflow_benchmark_job_id(workflow: Dict[str, Any]) -> str:
    stage = (workflow.get("stages") or {}).get("benchmark", {})
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    job_id = str(result.get("benchmark_job_id") or "").strip()
    if job_id:
        return job_id
    match = re.search(r'benchmark_report\((?:job_id=)?["\']?([A-Za-z0-9_-]+)["\']?\)', str(result.get("status_command") or ""))
    return match.group(1) if match else ""


def _workflow_structured_benchmark_query(
    workflow: Dict[str, Any],
    *,
    require_result: bool = False,
) -> Dict[str, Any]:
    """Read benchmark status from node-local files instead of an LLM turn."""
    benchmark = _benchmark_name(workflow)
    job_id = _workflow_benchmark_job_id(workflow)
    container = _workflow_evaluation_container(workflow)
    expected_model = os.path.basename(str((workflow.get("context") or {}).get("published_model_path") or "").rstrip("/"))
    benchmark_aliases = _workflow_benchmark_dataset_aliases(benchmark)
    script = r"""
import json
import os
import re
import sys

root, job_id, benchmark, require_result_text, expected_model, aliases_json = sys.argv[1:7]
require_result = require_result_text == "1"
benchmark_aliases = set(json.loads(aliases_json or "[]"))

def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}

def norm_name(value):
    text = os.path.basename(str(value or "").strip()).lower()
    if text.endswith(".json"):
        text = text[:-5]
    return text

def benchmark_matches(dataset_name, target_names):
    if not dataset_name:
        return True
    return dataset_name in target_names

def normalize_status(meta, result):
    status = str(meta.get("status") or result.get("status") or "").strip().lower()
    if status:
        return status
    if result:
        return "finished"
    return "running"

def int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip().rstrip("%")))
    except (TypeError, ValueError):
        return None

def percent_or_none(value):
    try:
        if value is None or value == "":
            return None
        text = str(value).strip()
        percent = float(text.rstrip("%"))
        if "%" not in text and 0 <= percent <= 1:
            percent *= 100
        return round(percent, 1)
    except (TypeError, ValueError):
        return None

def progress_text(processed, total, percent):
    if processed is None or total is None:
        return None
    if percent is None and total:
        percent = round(processed * 100 / total, 1)
    if percent is None:
        return f"{processed}/{total}"
    return f"{processed}/{total} ({percent:.1f}%)"

def progress_from_summary(result):
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    if not summary:
        return {}
    processed = int_or_none(summary.get("processed") or summary.get("done"))
    total = int_or_none(summary.get("total"))
    percent = percent_or_none(summary.get("progress") or summary.get("progress_percent"))
    text = progress_text(processed, total, percent)
    payload = {}
    if processed is not None:
        payload["processed"] = processed
    if total is not None:
        payload["total"] = total
    if percent is not None:
        payload["progress_percent"] = percent
    if text:
        payload["progress"] = text
    elif summary.get("progress") is not None:
        payload["progress"] = str(summary.get("progress"))
    for key in ("correct", "accuracy", "avg_f1", "invalid", "invalid_rate"):
        if key in summary:
            payload[key] = summary[key]
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        for key in ("record_only", "executed", "passed", "failed", "timeout", "executor_error", "pass@1"):
            if key in metrics:
                payload[key] = metrics[key]
    return payload

def read_tail(path, max_bytes=65536):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

def progress_from_run_log(folder):
    text = read_tail(os.path.join(folder, "run.log"))
    matches = re.findall(r"\[(\d+)\s*/\s*(\d+)\]\s*done", text)
    if not matches:
        return {}
    processed = int(matches[-1][0])
    total = int(matches[-1][1])
    percent = round(processed * 100 / total, 1) if total else None
    return {
        "processed": processed,
        "total": total,
        "progress_percent": percent,
        "progress": progress_text(processed, total, percent),
    }

def payload_for(folder):
    meta = load(os.path.join(folder, "meta.json"))
    result = load(os.path.join(folder, "result.json"))
    status = normalize_status(meta, result)
    result_path = os.path.join(folder, "result.json")
    log_file = os.path.join(folder, "run.log")
    payload = {
        "status": status,
        "benchmark_status": status,
        "benchmark_job_id": meta.get("job_id") or os.path.basename(folder),
        "model": meta.get("model"),
        "dataset": meta.get("dataset") or benchmark,
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "folder_path": folder,
        "log_dir": folder,
        "log_file": log_file,
        "result_path": result_path,
        "data": {"meta": meta},
    }
    if result:
        payload["data"]["result"] = result
    summary_progress = progress_from_summary(result)
    has_structured_progress = (
        summary_progress.get("processed") is not None
        and summary_progress.get("total") is not None
    )
    log_progress = {} if has_structured_progress else progress_from_run_log(folder)
    payload.update({**log_progress, **summary_progress})
    return payload

folders = []
skipped = []
target_names = {norm_name(value) for value in benchmark_aliases}
if os.path.isdir(root):
    if job_id and os.path.isdir(os.path.join(root, job_id)):
        folders = [os.path.join(root, job_id)]
    else:
        for name in os.listdir(root):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            meta = load(os.path.join(folder, "meta.json"))
            dataset = str(meta.get("dataset") or meta.get("benchmark") or "").strip()
            dataset_name = norm_name(dataset)
            if benchmark and dataset and dataset_name not in target_names:
                if len(skipped) < 5:
                    skipped.append({
                        "folder": folder,
                        "dataset": dataset,
                        "model": meta.get("model"),
                        "status": meta.get("status"),
                    })
                continue
            folders.append(folder)
        folders.sort(
            key=lambda path: (
                norm_name(load(os.path.join(path, "meta.json")).get("model")) == norm_name(expected_model),
                os.path.getmtime(path),
            ),
            reverse=True,
        )

if not folders:
    print(json.dumps({
        "status": "not_found",
        "benchmark_status": "not_found",
        "debug": {
            "root_exists": os.path.isdir(root),
            "benchmark": benchmark,
            "normalized_benchmark": norm_name(benchmark),
            "benchmark_aliases": sorted(target_names),
            "expected_model": expected_model,
            "skipped_candidates": skipped,
        },
    }, ensure_ascii=False))
    raise SystemExit(0)

payload = payload_for(folders[0])
if require_result and not payload.get("data", {}).get("result") and payload.get("status") not in {"failed", "stopped"}:
    payload["status"] = "running"
    payload["benchmark_status"] = "running"
print(json.dumps(payload, ensure_ascii=False))
"""
    process = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            script,
            BENCHMARK_LOGS_PATH,
            job_id,
            benchmark,
            "1" if require_result else "0",
            expected_model,
            json.dumps(benchmark_aliases),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "结构化 benchmark 状态查询失败")
    try:
        payload = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("结构化 benchmark 状态返回无效 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("结构化 benchmark 状态返回无效 payload")
    payload["result_entry"] = _benchmark_result_entry(workflow)
    if payload.get("benchmark_job_id"):
        payload["status_command"] = f'benchmark_report("{payload["benchmark_job_id"]}")'
        payload["stop_command"] = f'benchmark_stop("{payload["benchmark_job_id"]}")'
        payload["log_command"] = payload["status_command"]
    payload.update(
        _append_workflow_stage_log(
            workflow,
            "benchmark",
            "查询 benchmark 运行状态" if not require_result else "查询 benchmark 结果",
            payload,
        )
    )
    return payload


def _workflow_monitor_benchmark(workflow: Dict[str, Any]) -> Dict[str, Any]:
    return _workflow_structured_benchmark_query(workflow)


def _workflow_benchmark_result(workflow: Dict[str, Any]) -> Dict[str, Any]:
    return _workflow_structured_benchmark_query(workflow, require_result=True)

def _workflow_inference_resource_result_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    resource_context = result.get("resource_context")
    if not isinstance(resource_context, dict):
        return {}
    reservation_id = str(resource_context.get("reservation_id") or "").strip()
    if not reservation_id:
        return {}
    return {
        "inference_resource_context": resource_context,
        "inference_reservation_id": reservation_id,
        "inference_assigned_gpus": resource_context.get("assigned_gpus"),
        "inference_runtime_node_id": resource_context.get("runtime_node_id"),
    }

def _workflow_context_inference_resource_context(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resource_context = context.get("inference_resource_context")
    if isinstance(resource_context, dict):
        return resource_context
    return None


def _normalize_gpu_id_set(value: Any) -> set[str]:
    if value in (None, "", [], {}):
        return set()
    if isinstance(value, str):
        parts = re.split(r"[,;\s]+", value.strip())
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]
    return {str(part).strip() for part in parts if str(part).strip()}


def _workflow_expected_inference_gpus(resource_context: Optional[Dict[str, Any]]) -> set[str]:
    if not isinstance(resource_context, dict):
        return set()
    for key in ("assigned_gpus", "gpu_ids", "gpus", "CUDA_VISIBLE_DEVICES", "cuda_visible_devices"):
        values = _normalize_gpu_id_set(resource_context.get(key))
        if values:
            return values
    nodes = resource_context.get("nodes")
    if isinstance(nodes, list):
        values: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            for key in ("assigned_gpus", "gpu_ids", "gpus", "CUDA_VISIBLE_DEVICES", "cuda_visible_devices"):
                values.update(_normalize_gpu_id_set(node.get(key)))
        if values:
            return values
    return set()


def _workflow_payload_active_service_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[int] = set()

    def collect(source: Any) -> None:
        if not isinstance(source, dict):
            return
        source_id = id(source)
        if source_id in seen:
            return
        seen.add(source_id)
        service_instances = source.get("service_instances")
        if isinstance(service_instances, dict):
            raw_items = service_instances.get("items")
            if isinstance(raw_items, list):
                for item in raw_items:
                    if isinstance(item, dict):
                        status = str(item.get("status") or "").strip().lower()
                        if status in {"running", "degraded"}:
                            items.append(item)
        nodes = source.get("nodes")
        if isinstance(nodes, dict):
            for node_payload in nodes.values():
                collect(node_payload)

    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for source in (protocol, data, payload):
        collect(source)
    return items


def _workflow_payload_item_gpu_ids(item: Dict[str, Any]) -> set[str]:
    values: set[str] = set()
    nested_sources = [item]
    for key in ("resource", "resource_context", "runtime", "config", "config_draft"):
        nested = item.get(key)
        if isinstance(nested, dict):
            nested_sources.append(nested)
            nested_values = nested.get("values")
            if isinstance(nested_values, dict):
                nested_sources.append(nested_values)
    for source in nested_sources:
        for key in (
            "assigned_gpus",
            "assignedGpus",
            "gpu_ids",
            "gpuIds",
            "gpus",
            "CUDA_VISIBLE_DEVICES",
            "cuda_visible_devices",
        ):
            values.update(_normalize_gpu_id_set(source.get(key)))
    return values


def _workflow_payload_item_resource_scope(item: Dict[str, Any]) -> tuple[str, str]:
    sources = [item]
    for key in ("resource", "resource_context"):
        nested = item.get(key)
        if isinstance(nested, dict):
            sources.append(nested)

    resource_group_id = ""
    training_pool_id = ""
    for source in sources:
        if not resource_group_id:
            resource_group_id = str(
                source.get("resource_group_id") or source.get("resourceGroupId") or ""
            ).strip()
        if not training_pool_id:
            training_pool_id = str(
                source.get("training_pool_id") or source.get("trainingPoolId") or ""
            ).strip()
    return resource_group_id, training_pool_id


def _workflow_inference_service_matches_resource(
    status_result: Dict[str, Any],
    resource_context: Optional[Dict[str, Any]],
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
) -> bool:
    if not status_result.get("all_running"):
        return False
    expected_resource_group_id = str(resource_group_id or "").strip()
    expected_training_pool_id = str(training_pool_id or "").strip()
    expected_reservation_id = ""
    if isinstance(resource_context, dict):
        expected_reservation_id = str(
            resource_context.get("reservation_id") or resource_context.get("reservationId") or ""
        ).strip()
    expected_gpus = _workflow_expected_inference_gpus(resource_context)
    resource_scoped = bool(
        expected_resource_group_id
        or expected_training_pool_id
        or expected_reservation_id
        or expected_gpus
    )
    if not resource_scoped:
        return True
    for item in _workflow_payload_active_service_items(status_result):
        if expected_reservation_id and _reservation_id_from_payload_item(item) == expected_reservation_id:
            return True
        if expected_gpus:
            item_gpus = _workflow_payload_item_gpu_ids(item)
            if item_gpus and item_gpus == expected_gpus:
                return True
        if not expected_reservation_id and not expected_gpus and (expected_resource_group_id or expected_training_pool_id):
            item_group_id, item_pool_id = _workflow_payload_item_resource_scope(item)
            if expected_resource_group_id and item_group_id != expected_resource_group_id:
                continue
            if expected_training_pool_id and item_pool_id != expected_training_pool_id:
                continue
            if item_group_id or item_pool_id:
                return True
    return False

def _workflow_ensure_benchmark_inference_ready(workflow: Dict[str, Any]) -> Dict[str, Any]:
    context = workflow.get("context") or {}
    container = _workflow_evaluation_container(workflow)
    resource_group_id = str(context.get("resource_group_id") or "").strip()
    training_pool_id = str(context.get("training_pool_id") or "").strip()
    owner_kwargs = _workflow_inference_owner_kwargs(context)
    resource_context = _workflow_context_inference_resource_context(context)

    status_result = _workflow_inference_command(
        "查看推理服务状态",
        container,
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
        **owner_kwargs,
    )
    if _workflow_inference_service_matches_resource(
        status_result,
        resource_context,
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
    ):
        return status_result

    if not resource_context:
        try:
            resource_context = _inference_request_resource_context(
                "重启推理服务",
                resource_group_id=resource_group_id,
                training_pool_id=training_pool_id,
            )
        except Exception as e:
            raise RuntimeError(f"申请推理服务 GPU 资源失败: {e}")

    start_result = _workflow_inference_command(
        "重启推理服务",
        container,
        resource_context=resource_context,
        **owner_kwargs,
    )
    if not start_result.get("success"):
        error_msg = start_result.get("error") or start_result.get("result") or "推理服务重启命令失败"
        raise RuntimeError(error_msg)
    if isinstance(resource_context, dict):
        context["inference_resource_context"] = resource_context

    for _ in range(12):
        time.sleep(5)
        check = _workflow_inference_command(
            "查看推理服务状态",
            container,
            resource_group_id=resource_group_id,
            training_pool_id=training_pool_id,
            **owner_kwargs,
        )
        if _workflow_inference_service_matches_resource(
            check,
            resource_context,
            resource_group_id=resource_group_id,
            training_pool_id=training_pool_id,
        ):
            return check
    raise RuntimeError("推理服务启动超时，请检查容器日志和资源配置")


def _workflow_start_benchmark(workflow: Dict[str, Any]) -> Dict[str, Any]:
    context = workflow.get("context") or {}
    container = _workflow_evaluation_container(workflow)
    resource_group_id = str(context.get("resource_group_id") or "").strip()
    training_pool_id = str(context.get("training_pool_id") or "").strip()
    owner_kwargs = _workflow_inference_owner_kwargs(context)
    _workflow_ensure_benchmark_inference_ready(workflow)

    result = _workflow_inference_command(
        _benchmark_start_command(workflow),
        container,
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
        **owner_kwargs,
    )

    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("result") or f"{_benchmark_name(workflow)}基准评测启动失败")
    return result


def _workflow_inference_service_stop_command() -> str:
    return "停止推理服务 确认停止"

def _workflow_stop_inference_service(workflow: Dict[str, Any]) -> Dict[str, Any]:
    context = workflow.get("context") or {}
    stop_command = _workflow_inference_service_stop_command()
    result = _workflow_inference_command(
        stop_command,
        _workflow_evaluation_container(workflow),
        resource_group_id=str(context.get("resource_group_id") or "").strip(),
        training_pool_id=str(context.get("training_pool_id") or "").strip(),
            **_workflow_inference_owner_kwargs(context),
    )
    result["stop_command"] = stop_command
    result["service_log_command"] = "查看推理服务日志"
    log_fields = _append_workflow_stage_log(
        workflow,
        "stop_service",
        stop_command,
        result,
    )
    result.update(log_fields)
    result["stop_service_log_path"] = log_fields.get("log_path")
    result["stop_service_log_tail"] = log_fields.get("log_tail")
    result["stop_service_log_updated_at"] = log_fields.get("log_updated_at")
    return result


def _workflow_benchmark_stop_command(workflow: Dict[str, Any]) -> str:
    stage = (workflow.get("stages") or {}).get("benchmark", {})
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    stop_command = str(result.get("stop_command") or "").strip()
    if stop_command:
        #deepseek
        #return stop_command
        return f"{stop_command} 确认停止"
    job_id = str(result.get("benchmark_job_id") or "").strip()
    if job_id:
        #deepseek
        #return f'benchmark_stop("{job_id}")'
        return f'benchmark_stop("{job_id}") 确认停止'
    benchmark = _benchmark_name(workflow)
    benchmark_command = "2024.json" if benchmark in {"2024", "2024.json"} else benchmark
    return f"停止推理基准测试{benchmark_command}"


def _workflow_service_statuses_from_text(result: str) -> List[str]:
    statuses: List[str] = []
    for line in str(result or "").splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        match = re.search(
            r"(?:\b[A-Z_]*PORT\b|\(\s*\d{2,5}\s*\))[^:\n]*:\s*([A-Za-z_]+)",
            normalized,
            re.IGNORECASE,
        )
        if match:
            statuses.append(match.group(1).lower())
    return statuses


def _workflow_service_statuses_from_services(services: Any) -> List[str]:
    if not isinstance(services, list):
        return []
    return [
        str(item.get("status") or "").strip().lower()
        for item in services
        if isinstance(item, dict) and str(item.get("status") or "").strip()
    ]

def _workflow_service_statuses_from_service_instances(service_instances: Any) -> List[str]:
    if not isinstance(service_instances, dict):
        return []
    items = service_instances.get("items")
    if not isinstance(items, list):
        return []
    return [
        str(item.get("status") or "").strip().lower()
        for item in items
        if isinstance(item, dict) and str(item.get("status") or "").strip()
    ]


def _workflow_service_statuses_from_nodes(nodes: Any) -> List[str]:
    if not isinstance(nodes, dict):
        return []
    statuses: List[str] = []
    for node_payload in nodes.values():
        if not isinstance(node_payload, dict):
            continue
        statuses.extend(_workflow_service_statuses_from_services(node_payload.get("services")))
    return statuses
#deepseek
CORE_INFERENCE_PORTS = {"vllm", "inference", "ui", "case2chat"}
#deepseek
def _workflow_are_degraded_ports_healthy(payload: Dict[str, Any]) -> bool:
    """检测所有 degraded 实例的四个核心端口是否都 running。"""
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}

    def _collect_degraded_items(source: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        si = source.get("service_instances")
        if isinstance(si, dict):
            si_items = si.get("items")
            if isinstance(si_items, list):
                for item in si_items:
                    if isinstance(item, dict) and str(item.get("status") or "").lower() == "degraded":
                        items.append(item)
        nodes = source.get("nodes")
        if isinstance(nodes, dict):
            for node_payload in nodes.values():
                if not isinstance(node_payload, dict):
                    continue
                si = node_payload.get("service_instances")
                if isinstance(si, dict):
                    si_items = si.get("items")
                    if isinstance(si_items, list):
                        for item in si_items:
                            if isinstance(item, dict) and str(item.get("status") or "").lower() == "degraded":
                                items.append(item)
        return items

    all_degraded: List[Dict[str, Any]] = []
    for source in (protocol, data, payload):
        all_degraded.extend(_collect_degraded_items(source))

    if not all_degraded:
        return True

    for item in all_degraded:
        port_statuses = item.get("port_statuses")
        if not isinstance(port_statuses, list):
            return False
        running_core = set()
        for ps in port_statuses:
            if not isinstance(ps, dict):
                continue
            name = str(ps.get("name") or "").lower()
            status = str(ps.get("status") or "").lower()
            if name in CORE_INFERENCE_PORTS and status == "running":
                running_core.add(name)
        if not CORE_INFERENCE_PORTS.issubset(running_core):
            return False
    return True


def _workflow_all_running_from_payload(payload: Dict[str, Any], result: str = "") -> Optional[bool]:
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}

    for source in (protocol, data, payload):
        if isinstance(source, dict) and isinstance(source.get("allRunning"), bool):
            return source.get("allRunning") is True
        if isinstance(source, dict) and isinstance(source.get("all_running"), bool):
            return source.get("all_running") is True

    for nodes in (protocol.get("nodes"), data.get("nodes"), payload.get("nodes")):
        if not isinstance(nodes, dict):
            continue
        node_flags = [
            node_payload.get("allRunning")
            for node_payload in nodes.values()
            if isinstance(node_payload, dict) and isinstance(node_payload.get("allRunning"), bool)
        ]
        if node_flags:
            return all(flag is True for flag in node_flags)

    statuses: List[str] = []
    statuses.extend(_workflow_service_statuses_from_nodes(protocol.get("nodes")))
    statuses.extend(_workflow_service_statuses_from_nodes(data.get("nodes")))
    statuses.extend(_workflow_service_statuses_from_nodes(payload.get("nodes")))
    statuses.extend(_workflow_service_statuses_from_services(protocol.get("services")))
    statuses.extend(_workflow_service_statuses_from_services(data.get("services")))
    statuses.extend(_workflow_service_statuses_from_services(payload.get("services")))
    statuses.extend(_workflow_service_statuses_from_service_instances(protocol.get("service_instances")))
    statuses.extend(_workflow_service_statuses_from_service_instances(data.get("service_instances")))
    statuses.extend(_workflow_service_statuses_from_service_instances(payload.get("service_instances")))
    for nodes in (protocol.get("nodes"), data.get("nodes"), payload.get("nodes")):
        if not isinstance(nodes, dict):
            continue
        for node_payload in nodes.values():
            if isinstance(node_payload, dict):
                statuses.extend(_workflow_service_statuses_from_service_instances(node_payload.get("service_instances")))

    
    if statuses:
        #deepseek
        active_statuses = [s for s in statuses if s in {"running", "degraded", "starting"}]
        if not active_statuses:
            return False
        if all(s == "running" for s in active_statuses):
            return True
        if all(s in {"running", "degraded"} for s in active_statuses):
            if _workflow_are_degraded_ports_healthy(payload):
                 return True
        return False

    text_statuses = _workflow_service_statuses_from_text(result)
    #deepseek
    if text_statuses:
        return all(status in {"running", "degraded"} for status in text_statuses)

    lowered = str(result or "").lower()
    if any(keyword in lowered for keyword in ["运行中", "正在运行", "运行状态", "正常运行", "running"]):
        return True
    return None


def _reservation_id_from_payload_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
    resource_context = item.get("resource_context") if isinstance(item.get("resource_context"), dict) else {}
    return str(
        item.get("reservation_id")
        or item.get("reservationId")
        or resource.get("reservation_id")
        or resource.get("reservationId")
        or resource_context.get("reservation_id")
        or resource_context.get("reservationId")
        or ""
    ).strip()


def _service_start_matches_reservation(service_start: Any, reservation_id: str) -> bool:
    if not reservation_id or not isinstance(service_start, dict):
        return False
    if service_start.get("submitted") is not True:
        return False
    return _reservation_id_from_payload_item(service_start) == reservation_id


def _service_instances_hold_reservation(service_instances: Any, reservation_id: str) -> bool:
    if not reservation_id or not isinstance(service_instances, dict):
        return False
    items = service_instances.get("items")
    if not isinstance(items, list):
        return False
    active_statuses = {"starting", "running", "degraded"}
    return any(
        isinstance(item, dict)
        and str(item.get("status") or "").strip().lower() in active_statuses
        and _reservation_id_from_payload_item(item) == reservation_id
        for item in items
    )


def _payload_holds_matching_service_start(payload: Dict[str, Any], reservation_id: str) -> bool:
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    #deepseek
    result = str(payload.get("result") or "")
    for source in (protocol, data, payload):
        if _service_start_matches_reservation(source.get("service_start"), reservation_id):
            return True
    for nodes in (protocol.get("nodes"), data.get("nodes"), payload.get("nodes")):
        if not isinstance(nodes, dict):
            continue
        for node_payload in nodes.values():
            if isinstance(node_payload, dict) and _service_start_matches_reservation(node_payload.get("service_start"), reservation_id):
                return True
    #deepseek
    if not reservation_id:
        return False
    protocol_type = str(protocol.get("type") or "").lower()
    action = str(protocol.get("action") or "").lower()
    if protocol_type == "job_started" and action in {"service_start", "service_restart"}:
        return True
    if reservation_id in result:
        return True
    return False


def _payload_holds_matching_service_instance(payload: Dict[str, Any], reservation_id: str) -> bool:
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for source in (protocol, data, payload):
        if _service_instances_hold_reservation(source.get("service_instances"), reservation_id):
            return True
    for nodes in (protocol.get("nodes"), data.get("nodes"), payload.get("nodes")):
        if not isinstance(nodes, dict):
            continue
        for node_payload in nodes.values():
            if isinstance(node_payload, dict) and _service_instances_hold_reservation(node_payload.get("service_instances"), reservation_id):
                return True
    return False


def _inference_response_holds_resource_reservation(
    command: str,
    payload: Dict[str, Any],
    result: str = "",
    resource_context: Optional[Dict[str, Any]] = None,
) -> bool:
    if not _inference_command_needs_resource_context(command) or not isinstance(payload, dict):
        return False
    reservation_id = str((resource_context or {}).get("reservation_id") or "").strip()
    if not reservation_id:
        return False
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    status_value = str(payload.get("status") or "").lower()
    if status_value in {"error", "failed", "timeout"}:
        return False
    protocol_type = str(protocol.get("type") or "").lower()
    action = str(protocol.get("action") or data.get("action") or payload.get("action") or "").lower()
    if protocol_type in {"job_failed", "error", "need_input", "inference_config"}:
        return False
    if action in {"config_view", "config_update", "service_status", "service_instances", "instance_list", "instance_status", "service_stop_preview", "benchmark_stop_preview", "test_stop_preview", "logs"}:
        return False
    if _payload_holds_matching_service_start(payload, reservation_id):
        return True
    return _payload_holds_matching_service_instance(payload, reservation_id)

def _inference_service_start_submitted(response_data: Any) -> bool:
    if not isinstance(response_data, dict):
        return False
    service_start = response_data.get("service_start")
    if isinstance(service_start, dict) and service_start.get("submitted") is True:
        return True
    service_instances = response_data.get("service_instances")
    if isinstance(service_instances, dict):
        items = service_instances.get("items")
        if isinstance(items, list):
            active_statuses = {"starting", "running", "degraded"}
            if any(
                isinstance(item, dict)
                and str(item.get("status") or "").strip().lower() in active_statuses
                for item in items
            ):
                return True
    nodes = response_data.get("nodes")
    if isinstance(nodes, dict):
        return any(
            isinstance(node_data, dict)
            and _inference_service_start_submitted(node_data)
            for node_data in nodes.values()
        )
    return False


def _inference_service_start_payload_submitted(response_data: Any) -> bool:
    if not isinstance(response_data, dict):
        return False
    service_start = response_data.get("service_start")
    if isinstance(service_start, dict) and service_start.get("submitted") is True:
        return True
    nodes = response_data.get("nodes")
    if isinstance(nodes, dict):
        return any(
            isinstance(node_data, dict)
            and _inference_service_start_payload_submitted(node_data)
            for node_data in nodes.values()
        )
    return False


def _inference_start_result_needs_input(result: str) -> bool:
    text = str(result or "")
    lowered = text.lower()
    return (
        "请提供" in text
        or "请确认" in text
        or "需要" in text and "参数" in text
        or "无法识别模型参数规模" in text
        or "model_param_b" in lowered
    )

def _workflow_inference_command(
    command: str,
    container: Optional[str] = None,
    *,
    resource_group_id: str = "",
    training_pool_id: str = "",
    user_role: str = "",
    owner_user_id: str = "",
    owner_aliases: Any = None,
    resource_context: Optional[Dict[str, Any]] = None,
    #deepseek
    workflow: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    #deepseek
    if workflow is not None:
        ctx = workflow.get("context") or {}
        resource_group_id = str(resource_group_id or ctx.get("resource_group_id") or "").strip()
        training_pool_id = str(training_pool_id or ctx.get("training_pool_id") or "").strip()
        if not user_role:
            user_role = str(ctx.get("user_role") or "").strip()
        if not owner_user_id:
            owner_user_id = str(ctx.get("inference_owner_user_id") or "").strip()
        if not owner_aliases:
            owner_aliases = _owner_aliases_from_any(ctx.get("inference_owner_aliases"))

    
    
    prepare_resource_context = globals().get("_inference_request_resource_context")
    release_resource_context = globals().get("_release_inference_resource_context", lambda _context: None)

    release_known_reservations = globals().get("_release_known_inference_reservations", lambda: None)
    command_may_stop_service = globals().get("_inference_command_may_stop_service", lambda _command: False)
    
    ##deepseek
    # 如果未提供 resource_context，则自动申请
    if resource_context is None:
        try:
            if callable(prepare_resource_context):
                resource_context = prepare_resource_context(
                    command,
                    resource_group_id=resource_group_id,
                    training_pool_id=training_pool_id,
                )
        except Exception:
            release_resource_context(resource_context)
            raise
    # 否则直接使用外部传入的，跳过申请
    else:
        # 确保 resource_context 是 dict
        if not isinstance(resource_context, dict):
            resource_context = {}

    try:
        current_owner = globals().get("_current_inference_owner_user_id")
        current_aliases = globals().get("_current_inference_owner_aliases")
        alias_parser = globals().get("_owner_aliases_from_any", lambda value: list(value or []) if isinstance(value, (list, tuple, set)) else [item.strip() for item in str(value or "").split(",") if item.strip()])
        resolved_owner_user_id = str(owner_user_id or (current_owner() if callable(current_owner) else "") or "workflow-manager").strip()
        resolved_owner_aliases = alias_parser(owner_aliases) or (current_aliases() if callable(current_aliases) else [])
        request_payload: Dict[str, Any] = {
            "command": command,
            "user_id": resolved_owner_user_id,
            "user_aliases": resolved_owner_aliases,
            "thread_id": "workflow-manager",
            "container": container or _current_evaluation_container(),
        }
        if user_role:
            request_payload["user_role"] = user_role.strip().lower()
        if resource_context:
            request_payload["resource_context"] = resource_context
        response = requests.post(
            config.agents.inference.inference_agent_url,
            json=request_payload,
            timeout=360,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        release_resource_context(resource_context)
        raise

    if not isinstance(payload, dict):
        payload = {}
    status_value = str(payload.get("status") or "").lower()
    result = str(payload.get("result") or payload.get("message") or "")
    if isinstance(resource_context, dict) and resource_context.get("reservation_id"):
        if _inference_response_holds_resource_reservation(command, payload, result, resource_context):
            start_resource_heartbeat = globals().get("_start_inference_resource_heartbeat", lambda _context: None)
            start_resource_heartbeat(resource_context)
        else:
            release_resource_context(resource_context)
    ###deepseek
    #_inference_payload_is_service_stop = globals().get("_inference_payload_is_service_stop", lambda _payload: False)
    #if _inference_payload_is_service_stop(payload):
    #    release_known_reservations()
    if command_may_stop_service(command) and _inference_stop_response_is_success(command, payload):
        release_known_reservations()

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    lowered = result.lower()
    all_running = _workflow_all_running_from_payload(payload, result)
    benchmark_status = str(data.get("status") or data.get("benchmark_status") or "").lower()
    explicit_benchmark_status = _workflow_benchmark_explicit_status(result)
    if explicit_benchmark_status:
        benchmark_status = explicit_benchmark_status
    if not benchmark_status:
        for value in ("failed", "finished", "running", "stopped"):
            if value in lowered:
                benchmark_status = value
                break
    if not benchmark_status:
        chinese_statuses = (
            ("failed", ("失败", "异常", "报错")),
            ("finished", ("已完成", "完成", "结束")),
            ("running", ("运行中", "正在运行", "处理中")),
            ("stopped", ("已停止", "已终止", "停止")),
        )
        for status_value, keywords in chinese_statuses:
            if any(keyword in result for keyword in keywords):
                benchmark_status = status_value
                break
    failed = status_value in {"error", "failed", "timeout"} or _workflow_inference_has_failure_signal(result)
    model_name = _workflow_inference_model_name(data)
    if model_name is None:
        model_name = _workflow_inference_model_name(payload)
    return {
        "success": response.ok and not failed,
        "result": result,
        "data": data,
        "protocol": payload.get("protocol") if isinstance(payload.get("protocol"), dict) else None,
        "model_path": _workflow_inference_model_path(data) or _workflow_inference_model_path(payload),
        "model_name": model_name,
        "all_running": all_running is True,
        "status": benchmark_status or None,
        "error": result if failed else None,
        "resource_context": resource_context,
    }

def _workflow_stop_task(workflow: Dict[str, Any]) -> Dict[str, Any]:
    stage_name = workflow["current_stage"]
    stage = workflow["stages"][stage_name]
    if stage_name == "benchmark":
        context = workflow.get("context") or {}
        stop_benchmark_result = _workflow_inference_command(
            _workflow_benchmark_stop_command(workflow),
            _workflow_evaluation_container(workflow),
            resource_group_id=str(context.get("resource_group_id") or "").strip(),
            training_pool_id=str(context.get("training_pool_id") or "").strip(),
            **_workflow_inference_owner_kwargs(context),
        )
        benchmark_log_fields = _append_workflow_stage_log(
            workflow,
            "benchmark",
            "停止 benchmark 任务",
            stop_benchmark_result,
        )
        stop_service_result = _workflow_stop_inference_service(workflow)
        return {
            **benchmark_log_fields,
            "benchmark_stop_result": stop_benchmark_result,
            "inference_service_stop_result": stop_service_result,
            "inference_service_stop_command": _workflow_inference_service_stop_command(),
            "inference_service_log_command": "查看推理服务日志",
            "stop_service_log_path": stop_service_result.get("stop_service_log_path"),
            "stop_service_log_tail": stop_service_result.get("stop_service_log_tail"),
            "stop_service_log_updated_at": stop_service_result.get("stop_service_log_updated_at"),
        }
    if stage_name == "deploy":
        stop_service_result = _workflow_stop_inference_service(workflow)
        return {
            "inference_service_stop_result": stop_service_result,
            "inference_service_stop_command": _workflow_inference_service_stop_command(),
            "inference_service_log_command": "查看推理服务日志",
            "stop_service_log_path": stop_service_result.get("stop_service_log_path"),
            "stop_service_log_tail": stop_service_result.get("stop_service_log_tail"),
            "stop_service_log_updated_at": stop_service_result.get("stop_service_log_updated_at"),
            **_workflow_stage_log_fields(workflow, "deploy"),
        }
    pid = str(stage.get("pid") or "").strip()
    container = str(stage.get("container") or "").strip()
    if not pid:
        return {}
    if container == "host":
        # 宿主机进程（如 publish 后台复制）
        _workflow_stop_host_process(pid)
    else:
        # Docker 容器内进程
        cleanup_patterns = []
        if stage_name == "evaluate":
            model_path = str(workflow.get("context", {}).get("trained_model_path") or "").strip()
            cleanup_patterns = [
                model_path,
                os.path.basename(os.path.normpath(model_path)) if model_path else "",
                "single_model_evaluation_vpn",
                "record_to_diagnose",
                "diagnose_accuracy",
                "thread_medhalt_confidence_eval",
                "--port 8102",
            ]
        pattern_args = " ".join(
            shlex.quote(re.sub(r"([A-Za-z0-9])", r"[\1]", re.escape(pattern), count=1))
            for pattern in cleanup_patterns
            if pattern
        )
        cleanup_command = ""
        if pattern_args:
            cleanup_command = f"""
for pattern in {pattern_args}; do
    pgrep -f "$pattern" 2>/dev/null | while read -r worker_pid; do
        [ "$worker_pid" = "$$" ] || kill -TERM "$worker_pid" 2>/dev/null || true
    done
done
"""
        subprocess.run(
            [
                "docker", "exec", container, "sh", "-c",
                f"kill {shlex.quote(pid)} 2>/dev/null || true\n{cleanup_command}",
            ],
            capture_output=True, text=True, timeout=30,
        )
    return {}


def _workflow_stop_host_process(pid: str) -> str:
    """Stop a host-side workflow helper and its children when possible."""
    pid_value = int(str(pid).strip())
    try:
        os.killpg(pid_value, signal.SIGKILL)
        return f"已结束宿主机进程组: {pid_value}"
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        completed = subprocess.run(
            ["kill", "-9", str(pid_value)],
            capture_output=True, text=True, timeout=30,
        )
        output = "\n".join(
            text for text in (completed.stdout.strip(), completed.stderr.strip()) if text
        )
        return output or f"已结束宿主机进程: {pid_value}"


def _notify_terminal_workflow_update(workflow: Dict[str, Any]) -> None:
    """Bridge terminal workflow state from the manager thread to Studio."""
    loop = _WORKFLOW_EVENT_LOOP
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(
        _push_terminal_workflow_update(workflow),
        loop,
    )


async def _push_terminal_workflow_update(workflow: Dict[str, Any]) -> None:
    try:
        session = await get_or_create_user_session(str(workflow.get("user_id") or ""))
        system = session.system
        if system is None or system.orchestrator is None:
            return
        display = system._workflow_display_text(workflow)
        protocol = system._workflow_protocol(workflow, display, agent="orchestrator")
        system.last_response_protocol = protocol
        await system._safe_agent_print(
            system.orchestrator,
            system._agent_msg(system.orchestrator.name, display, protocol),
        )
    except Exception:
        logger.exception("Failed to push terminal workflow update %s", workflow.get("workflow_id"))


def get_workflow_manager() -> WorkflowManager:
    global _WORKFLOW_MANAGER
    with _WORKFLOW_MANAGER_LOCK:
        if _WORKFLOW_MANAGER is None:
            _WORKFLOW_MANAGER = WorkflowManager(
                db_path=config.workflow.db_path,
                poll_interval=config.workflow.poll_interval,
                auto_start_worker=config.workflow.auto_start_worker,
                train_start_grace_seconds=config.workflow.train_start_grace_seconds,
                event_lease_seconds=config.workflow.event_lease_seconds,
                on_terminal_update=_notify_terminal_workflow_update,
                dependencies=WorkflowDependencies(
                    start_train=_workflow_start_train,
                    monitor_train=_workflow_monitor_train,
                    find_trained_model=_workflow_find_trained_model,
                    start_evaluate=_workflow_start_evaluate,
                    monitor_evaluate=_workflow_monitor_evaluate,
                    start_publish=_workflow_start_publish,
                    monitor_publish=_workflow_monitor_publish,
                    inference_command=_workflow_inference_command,
                    stop_task=_workflow_stop_task,
                    stop_inference_service=_workflow_stop_inference_service,
                    check_existing_stage_output=_workflow_check_existing_stage_output,
                    monitor_deploy=_workflow_monitor_deploy,
                    start_benchmark=_workflow_start_benchmark,
                    monitor_benchmark=_workflow_monitor_benchmark,
                    benchmark_result=_workflow_benchmark_result,
                ),
            )
        return _WORKFLOW_MANAGER


class OrchestratorSystem:
    """智能体编排系统：每个用户一个独立实例"""
    
    def __init__(self, user_id: str, shared_model):
        self.user_id = user_id
        self.training_container = DEFAULT_DOCKER_CONTAINER
        self.evaluation_container = DEFAULT_EVALUATE_DOCKER_CONTAINER
        self.grpo_container = DEFAULT_DOCKER_CONTAINER
        self.multinode_training_container = MULTINODE_DOCKER_CONTAINER
        self.shared_model = shared_model
        self.agents: Dict[str, Agent] = {}
        self.orchestrator: Optional[Agent] = None
        self.user_agent = None
        
        # 状态管理（可序列化）
        self.current_task_state: Dict[str, Any] = {}
        self.pending_parameters: Dict[str, Any] = {}
        self.task_context: Dict[str, Any] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.last_completed_task: Optional[Dict[str, Any]] = None
        self.last_response_protocol: Optional[Dict[str, Any]] = None
        self.current_user_message: str = ""
        self._workflow_child_context: Optional[Dict[str, Any]] = None
        self._last_inference_tool_payload: Optional[Dict[str, Any]] = None
        
        logger.debug(f"OrchestratorSystem created for user {user_id}")
    
    def get_serializable_state(self) -> Dict[str, Any]:
        """获取可序列化的状态"""
        return {
            "current_task_state": self.current_task_state,
            "pending_parameters": self.pending_parameters,
            "task_context": self.task_context,
            "task_history": self.task_history,
            "last_completed_task": self.last_completed_task,
            "last_response_protocol": self.last_response_protocol,
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """从字典恢复状态"""
        self.current_task_state = state.get("current_task_state", {})
        self.pending_parameters = state.get("pending_parameters", {})
        self.task_context = state.get("task_context", {})
        self.task_history = state.get("task_history", [])
        self.last_completed_task = state.get("last_completed_task")
        self.last_response_protocol = self._normalize_protocol(
            state.get("last_response_protocol"),
            source="restored",
        )
        logger.debug(f"State restored for user {self.user_id}")

    def _response_to_text(self, response: Any) -> str:
        if isinstance(response, ToolResponse):
            return _tool_response_text(response)
        if hasattr(response, "get_text_content"):
            text = response.get_text_content()
            if text is not None:
                return text
        return str(response)

    def _tool_response_to_client_text(self, response: Any) -> str:
        text = self._response_to_text(response)
        protocol = None
        if isinstance(response, ToolResponse):
            metadata = response.metadata or {}
            protocol = metadata.get("protocol") or self._protocol_from_hint(
                metadata.get("protocol_hint"),
                text,
            )
        protocol = protocol or self.last_response_protocol
        if isinstance(protocol, dict):
            self.last_response_protocol = protocol
        return self._protocol_json_response(text, protocol)

    def _restore_recent_job_protocol(
        self,
        response_text: str,
        protocol: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if protocol and protocol.get("type") != "message":
            return protocol
        recent = self.last_response_protocol if isinstance(self.last_response_protocol, dict) else None
        if not recent or recent.get("type") not in {"job_started", "job_preparing"}:
            return protocol
        job_type = str(recent.get("jobType") or "").strip()
        text = response_text or ""
        has_job_evidence = bool(
            self._extract_pid(text)
            or re.search(r"后台(?:启动|运行)|已启动|PID|进程\s*ID", text, re.IGNORECASE)
        )
        if not has_job_evidence:
            return protocol

        restored = {**recent, "message": text}
        pid = self._extract_pid(text) or recent.get("pid")
        container = self._extract_container(text) or recent.get("container")
        if pid:
            restored["pid"] = str(pid)
        if container:
            restored["container"] = str(container)
        if job_type == "train" or restored.get("agent") == "trainer":
            train_type = self._normalize_train_type(
                restored.get("trainType") or self._infer_train_type(text)
            )
            launch_mode = restored.get("launchMode") or self._infer_launch_mode(text)
            if train_type:
                restored["trainType"] = train_type
                train_type_text = self._train_type_text(train_type, launch_mode)
                if train_type_text:
                    restored["trainTypeText"] = train_type_text
            if launch_mode:
                restored["launchMode"] = launch_mode
                restored["isMultinode"] = launch_mode == "multinode"
        return self._normalize_protocol(restored, text, source="restored") or restored

    def _protocol_from_hint(self, hint: Any, message: str) -> Optional[Dict[str, Any]]:
        if not isinstance(hint, dict):
            return None
        protocol_type = hint.get("type") or hint.get("protocol_type")
        agent = hint.get("agent") or "orchestrator"
        if not protocol_type:
            return None
        fields = {
            key: value
            for key, value in hint.items()
            if key not in {"type", "protocol_type", "agent", "message", "source", "confidence", "valid"}
        }
        if protocol_type == "need_input" and "title" not in fields and fields.get("kind"):
            fields["title"] = self._need_input_title(str(fields["kind"]))
        if protocol_type == "need_input" and "options" not in fields and fields.get("kind"):
            options = self._schema_options_for_kind(agent, str(fields["kind"]))
            if options:
                fields["options"] = options
        if protocol_type == "monitor_status":
            fields.setdefault("jobType", "train")
        return self._with_protocol(
            str(protocol_type),
            str(agent),
            str(hint.get("message") or message or ""),
            source=str(hint.get("source") or "tool_hint"),
            confidence=hint.get("confidence"),
            **fields,
        )

    def _extract_parenthesized_options(self, text: str) -> List[str]:
        match = re.search(r"（([^）]+)）", text or "")
        if not match:
            return []
        raw_options = re.split(r"[、,，]|或", match.group(1))
        return [item.strip() for item in raw_options if item.strip()]

    def _extract_bullet_items(self, text: str) -> List[str]:
        return [
            item.strip()
            for item in re.findall(r"^\s*-\s*(.+)$", text or "", re.MULTILINE)
            if item.strip()
        ]

    def _extract_dataset_name_choice_options(self, text: str) -> List[str]:
        """Extract concrete dataset_name candidates from a plain clarification message."""
        if not text:
            return []
        candidates: List[str] = []
        for match in re.finditer(
            r"^\s*(?:[-*]|\d+[.、])\s*(?:\*\*)?`?([A-Za-z0-9_.-]+)`?(?:\*\*)?\s*$",
            text,
            re.MULTILINE,
        ):
            candidates.append(match.group(1))
        return self._sanitize_dataset_name_options(candidates)

    def _dataset_name_choice_protocol_from_message(
        self,
        message: str,
        original_task: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert orchestrator prose asking for dataset_name into a real need_input protocol."""
        text = THINK_TAG_RE.sub("", message or "").strip()
        if not text:
            return None
        options = self._extract_dataset_name_choice_options(text)
        if len(options) < 2:
            return None

        known_params = self._extract_named_param_values(original_task or "")
        original_lower = (original_task or "").lower()
        has_dataset_context = bool(
            known_params.get("dataset_dir")
            or "dataset_name" in text
            or "dataset_name" in original_lower
            or "dataset_dir" in original_lower
            or "dpo" in original_lower
            or "enhanced" in original_lower
            or "增强训练" in (original_task or "")
        )
        if not has_dataset_context:
            return None
        dataset_dir = known_params.get("dataset_dir")
        if not dataset_dir:
            dataset_ref = self._extract_dataset_reference(original_task or "")
            if dataset_ref:
                dataset_dir = f"/home/workspace/dataset_daily_train/{dataset_ref}"
                known_params["dataset_dir"] = dataset_dir

        protocol = self._with_protocol(
            "need_input",
            "trainer",
            self._strip_user_waiting_wrapper(text),
            kind="training_params",
            title=self._need_input_title("training_params"),
            requiredParams=["model_path", "dataset_dir", "dataset_name"],
            missingParams=["dataset_name"],
            options=options,
            knownParams=known_params or None,
            action="collect_params",
            status="needs_input",
            jobType="train",
            trainType="enhanced",
            trainTypeText="增强训练",
        )
        return self._normalize_protocol(protocol, text)

    def _build_reply_example(self, items: List[str], prefix: str = "可以直接回复") -> str:
        cleaned_items = [item for item in items if item]
        if not cleaned_items:
            return ""
        if len(cleaned_items) == 1:
            return f"\n\n{prefix}：`{cleaned_items[0]}`"
        example = "，".join(f"{item}=xxx" for item in cleaned_items[:3])
        return f"\n\n{prefix}：`{example}`"

    def _wrap_user_waiting_message(self, category: str, body: str) -> str:
        return (
            f"[等待用户回复|{category}]\n"
            "以下内容是发给用户的补充信息提示，请原样展示给用户，不要代替用户回答。\n\n"
            f"{body.strip()}"
        )

    def _strip_user_waiting_wrapper(self, text: str) -> str:
        content = (text or "").strip()
        if not content.startswith("[等待用户回复|"):
            return content
        lines = content.splitlines()
        if not lines:
            return content
        remaining = lines[1:]
        if remaining and remaining[0].strip() == "以下内容是发给用户的补充信息提示，请原样展示给用户，不要代替用户回答。":
            remaining = remaining[1:]
        return "\n".join(remaining).strip()

    def _protocol_response_enabled(self) -> bool:
        value = os.getenv("AGENT_RESPONSE_PROTOCOL", "").strip().lower()
        return value in {"1", "true", "yes", "on", "json", "structured"}

    def _default_protocol_confidence(self, source: str) -> float:
        return PROTOCOL_CONFIDENCE_BY_SOURCE.get(source, 0.68)

    def _protocol_required_fields(self, protocol_type: str) -> List[str]:
        if protocol_type == "need_input":
            return ["version", "type", "agent", "message", "kind", "title"]
        if protocol_type in {"job_started", "job_preparing", "job_stopped", "job_failed", "monitor_status"}:
            return ["version", "type", "agent", "message", "jobType"]
        if protocol_type.startswith("inference_"):
            return ["version", "type", "agent", "message", "action"]
        return ["version", "type", "agent", "message"]

    def _sanitize_dataset_name_options(self, options: Any) -> List[str]:
        """Keep only concrete dataset_name values in protocol options."""
        raw_options = options if isinstance(options, list) else ([options] if options else [])
        sanitized: List[str] = []
        blocked_values = {"dataset_name", "xxx", "数据集名称", "训练时要使用的数据集名称"}
        for option in raw_options:
            candidate = str(option or "").strip().strip("`'\"").strip()
            if not candidate or candidate in blocked_values:
                continue
            if "=" in candidate or "数据集名称" in candidate or "训练时要使用" in candidate:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate):
                continue
            if candidate not in sanitized:
                sanitized.append(candidate)
        return sanitized

    def _normalize_protocol(
        self,
        protocol: Optional[Dict[str, Any]],
        message: str = "",
        source: Optional[str] = None,
        confidence: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(protocol, dict):
            return None

        normalized = dict(protocol)
        protocol_type = str(normalized.get("type") or "message")
        agent = str(normalized.get("agent") or "orchestrator")
        normalized["version"] = str(normalized.get("version") or PROTOCOL_VERSION)
        normalized["type"] = protocol_type
        normalized["agent"] = agent
        normalized["message"] = self._strip_user_waiting_wrapper(
            str(normalized.get("message") or message or "")
        )

        protocol_source = str(source or normalized.get("source") or "rule")
        normalized["source"] = protocol_source
        try:
            normalized["confidence"] = round(float(confidence if confidence is not None else normalized.get("confidence")), 3)
        except (TypeError, ValueError):
            normalized["confidence"] = self._default_protocol_confidence(protocol_source)

        if protocol_type == "need_input":
            kind = str(normalized.get("kind") or "input")
            normalized["kind"] = kind
            normalized.setdefault("title", self._need_input_title(kind))
            if not normalized.get("options"):
                options = self._schema_options_for_kind(agent, kind)
                if options:
                    normalized["options"] = options
            param_deduper = getattr(self, "_dedupe_required_params", lambda params: [str(param) for param in params if str(param).strip()])
            if "required_params" in normalized and "requiredParams" not in normalized:
                normalized["requiredParams"] = normalized.pop("required_params")
            if normalized.get("requiredParams") is None:
                normalized.pop("requiredParams", None)
            elif not isinstance(normalized.get("requiredParams"), list):
                normalized["requiredParams"] = [str(normalized["requiredParams"])]
            else:
                normalized["requiredParams"] = param_deduper(
                    [str(param) for param in normalized["requiredParams"] if str(param).strip()]
                )
            if normalized.get("missingParams") is None:
                normalized.pop("missingParams", None)
            elif not isinstance(normalized.get("missingParams"), list):
                normalized["missingParams"] = [str(normalized["missingParams"])]
            else:
                normalized["missingParams"] = param_deduper(
                    [str(param) for param in normalized["missingParams"] if str(param).strip()]
                )
            dataset_required = set(str(param) for param in normalized.get("requiredParams") or []) == {"dataset_name"}
            dataset_missing = set(str(param) for param in normalized.get("missingParams") or []) == {"dataset_name"}
            if dataset_required or dataset_missing:
                options = self._sanitize_dataset_name_options(normalized.get("options") or [])
                if options:
                    normalized["options"] = options
                else:
                    normalized.pop("options", None)

        if protocol_type == "monitor_status":
            normalized.setdefault("jobType", "train")

        if protocol_type in {"job_started", "job_preparing", "job_stopped", "job_failed"} and not normalized.get("jobType"):
            inferred_job_type = self._infer_job_type(agent, normalized["message"])
            if inferred_job_type:
                normalized["jobType"] = inferred_job_type

        if protocol_type.startswith("inference_") and not normalized.get("action"):
            normalized["action"] = self._infer_inference_action("", normalized["message"])

        missing = [
            field
            for field in self._protocol_required_fields(protocol_type)
            if normalized.get(field) in (None, "")
        ]
        normalized["valid"] = not missing
        if missing:
            normalized["missingFields"] = missing
            normalized["confidence"] = min(float(normalized.get("confidence") or 0.0), 0.45)
        else:
            normalized.pop("missingFields", None)

        return normalized

    def _with_protocol(
        self,
        protocol_type: str,
        agent: str,
        message: str,
        *,
        source: str = "rule",
        confidence: Optional[Any] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        protocol = {
            "version": PROTOCOL_VERSION,
            "type": protocol_type,
            "agent": agent,
            "message": self._strip_user_waiting_wrapper(message or ""),
        }
        protocol.update({key: value for key, value in fields.items() if value is not None})
        return self._normalize_protocol(protocol, message, source=source, confidence=confidence) or protocol

    def _need_input_title(self, kind: str) -> str:
        titles = {
            "choice": "确认下一步",
            "training_type": "确认训练方式",
            "assessment_type": "确认评估方式",
            "evaluation_type": "确认评估方式",
            "advanced_filter_choice": "确认高级筛选",
            "data_preprocess_params": "补充预处理参数",
            "input_folder": "补充数据路径",
            "schedule_time": "补充定时时间",
            "checkpoint_path": "补充 Checkpoint 路径",
            "training_params": "补充训练参数",
            "assessment_params": "补充评估参数",
            "evaluation_params": "补充评估参数",
            "data_params": "补充数据参数",
        }
        return titles.get(kind, "补充信息")

    def _protocol_json_response(self, message: str, protocol: Optional[Dict[str, Any]] = None) -> str:
        if not self._protocol_response_enabled():
            return message
        protocol = self._normalize_protocol(
            protocol,
            message,
            source=protocol.get("source") if isinstance(protocol, dict) else None,
        )
        payload = {
            "message": message,
            "protocol": protocol,
        }
        if protocol:
            payload.update(protocol)
        return json.dumps(payload, ensure_ascii=False)

    def _agent_msg(
        self,
        name: str,
        content: str,
        protocol: Optional[Dict[str, Any]] = None,
    ) -> Msg:
        metadata = {"protocol": protocol} if protocol else None
        return _msg(name=name, content=content, role="assistant", metadata=metadata)

    async def _safe_agent_print(self, agent: Any, msg: Msg) -> None:
        """Print through AgentScope without letting Studio push failures abort work."""
        try:
            await agent.print(msg, True)
        except Exception as exc:
            logger.warning("Studio pushMessage skipped for agent reply: %s", exc)
            if not getattr(agent, "_disable_console_output", False):
                print(f"{msg.name}: {msg.get_text_content() or ''}")

    def _schema_kind_for_request(
        self,
        agent_name: str,
        request_kind: str,
        required_params: Optional[List[str]] = None,
    ) -> Optional[str]:
        params = set(required_params or [])
        if request_kind == "type":
            return {
                "trainer": "training_type",
                "evaluator": "assessment_type",
            }.get(agent_name)

        if request_kind == "choice":
            if agent_name == "dataprocessor":
                return "advanced_filter_choice"
            return "choice"

        if params:
            if agent_name == "evaluator" and params & {"model_fir", "model_sec"}:
                return "evaluation_params"
            if "schedule_time" in params and agent_name == "trainer":
                return "schedule_time"
            if params == {"input_folder"}:
                return "input_folder"
            if params == {"CKPT_PATH"} and agent_name == "evaluator":
                return "checkpoint_path"
            if params & {"model_path", "train_files", "val_files", "dataset_dir", "dataset_name"}:
                return "training_params"
            if params & {"model_fir", "model_sec"}:
                return "evaluation_params"
            if params & {"data_type", "strategy"}:
                return "data_preprocess_params"

        return None

    def _need_input_kind(self, agent_name: str, request_kind: str, text: str) -> str:
        content = text or ""
        if request_kind == "choice":
            return "choice"
        if agent_name == "trainer":
            if request_kind == "type" or "训练类型" in content:
                return "training_type"
            if "schedule_time" in content:
                return "schedule_time"
            return "training_params"
        if agent_name == "evaluator":
            if request_kind == "type" or "评估类型" in content:
                return "evaluation_type"
            if "CKPT_PATH" in content:
                return "checkpoint_path"
            return "evaluation_params"
        if agent_name == "dataprocessor":
            if "input_folder" in content:
                return "input_folder"
            if request_kind == "choice":
                return "advanced_filter_choice"
            if "data_type" in content or "strategy" in content:
                return "data_preprocess_params"
            if "高级筛选" in content and any(keyword in content for keyword in ["是否", "需要执行", "继续执行"]):
                return "advanced_filter_choice"
            return "data_params"
        if agent_name == "datacollector":
            return "data_collection_params"
        if agent_name == "monitor":
            return "monitor_params"
        return request_kind or "input"

    def _schema_options_for_kind(self, agent_name: str, kind: str) -> List[str]:
        if kind == "training_type":
            return ["LoRA SFT", "全参 SFT", "增强训练", "GRPO", "双机 LoRA SFT", "双机增强训练"]
        if kind in {"assessment_type", "evaluation_type"}:
            return ["双模型", "单模型", "ckpt评估"]
        if kind == "advanced_filter_choice" and agent_name == "dataprocessor":
            return ["是", "否"]
        return []

    def _need_input_options(self, agent_name: str, kind: str, text: str) -> List[str]:
        if kind == "schedule_time":
            return []
        schema_options = self._schema_options_for_kind(agent_name, kind)
        if schema_options:
            return schema_options
        if kind == "training_params":
            return []
        options = self._extract_parenthesized_options(text)
        if options:
            return options
        return []

    def _need_input_protocol(
        self,
        agent_name: str,
        request_kind: str,
        friendly_text: str,
        raw_text: str,
        required_params: Optional[List[str]] = None,
        missing_params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        message = self._strip_user_waiting_wrapper(friendly_text)
        combined_text = "\n".join([raw_text or "", message])
        kind = (
            self._schema_kind_for_request(agent_name, request_kind, required_params)
            or self._need_input_kind(agent_name, request_kind, combined_text)
        )
        options = self._need_input_options(agent_name, kind, combined_text)
        fields: Dict[str, Any] = {
            "kind": kind,
            "title": self._need_input_title(kind),
        }
        if options:
            fields["options"] = options
        if required_params:
            fields["requiredParams"] = required_params
        if missing_params:
            fields["missingParams"] = missing_params
        return self._with_protocol(
            "need_input",
            agent_name,
            message,
            **fields,
        )

    def _pending_state(
        self,
        agent_name: str,
        original_task: str,
        request_kind: str,
        response_text: str,
        friendly_text: str,
        protocol: Dict[str, Any],
        required_params: Optional[List[str]] = None,
        needs_choice: bool = False,
    ) -> Dict[str, Any]:
        message = self._strip_user_waiting_wrapper(friendly_text)
        return {
            "agent": agent_name,
            "kind": protocol.get("kind"),
            "message": message,
            "original_task": original_task,
            "request_kind": request_kind,
            "required_params": required_params or [],
            "needs_choice": needs_choice,
            "needs_params": request_kind == "param",
            "response": response_text,
            "friendly_response": friendly_text,
            "protocol": protocol,
        }

    def _pending_required_and_missing_from_context(
        self,
        agent_name: str,
        task_text: str,
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        pending = self.pending_parameters.get(agent_name) or {}
        protocol = pending.get("protocol") if isinstance(pending, dict) else None
        if not isinstance(protocol, dict):
            return [], [], {}

        required_params = self._dedupe_required_params(
            [str(param) for param in (protocol.get("requiredParams") or pending.get("required_params") or []) if str(param).strip()]
        )
        if not required_params:
            return [], [], {}

        known_params = dict(protocol.get("knownParams") or {})
        provided_params = self._extract_named_param_values(task_text or "")
        merged_known = {
            key: value
            for key, value in {**known_params, **provided_params}.items()
            if key in required_params and value
        }
        missing_params = [param for param in required_params if not merged_known.get(param)]
        return required_params, missing_params, merged_known
    def _infer_job_type(self, agent_name: str, text: str) -> Optional[str]:
        content = (text or "").lower()
        if agent_name == "trainer":
            return "train"
        if agent_name == "dataprocessor":
            return "data_filter" if "score_based_filtering" in content or "高级筛选" in text else "data_preprocess"
        if agent_name == "evaluator":
            return "assessment"
        if agent_name == "datacollector":
            return "data_collect"
        return None

    def _is_multinode_training_request(self, text: str) -> bool:
        content = (text or "").lower()
        return any(
            keyword in content
            for keyword in ["多机", "多节点", "双机", "multinode", "multi-node", "2node", "two-node"]
        )
    def _infer_multinode_train_family(self, text: str) -> Optional[str]:
        content = (text or "").lower()
        if not self._is_multinode_training_request(text):
            return None 
        if (
            "train_multinode_dpo_pipeline" in content
            or "多机增强训练" in text
            or "增强训练" in text
            or "dpo" in content
            or "偏好优化" in text
        ):
            return "enhanced"
        if (
            "train_multinode_sft_pipeline" in content
            or "多机lora批量训练" in content
            or "lora批量训练" in text
            or "lora" in content
            or "sft" in content
            or "批量训练" in text
        ):
            return "lora"
        return None        

    def _extract_container_override(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:container|docker_container|容器|docker)\s*(?:是|为|=|:)?\s*([A-Za-z0-9_.-]+)",
            text or "",
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    def _training_container_for_request(self, text: str) -> str:
        explicit_container = self._extract_container_override(text)
        if explicit_container:
            return explicit_container
        if self._infer_train_type(text) == "grpo":
            return self.grpo_container
        return self.multinode_training_container if self._is_multinode_training_request(text) else self.training_container

    def _extract_multinode_cli_args(self, text: str) -> Dict[str, str]:
        args: Dict[str, str] = {}
        aliases = {
            "model_path": "model-path",
            "MODEL_PATH": "model-path",
            "base_model_path": "model-path",
            "dataset_dir": "dataset-dir",
            "data_dir": "dataset-dir",
            "sft_data_dir": "dataset-dir",
            "sft_dataset_dir": "dataset-dir",
            "dataset_name": "dataset",
            "dataset-name": "dataset",
            "data_identifier": "dataset-date",
            "dataset_date": "dataset-date",
            "data_id": "dataset-date",
            "dataset_id": "dataset-date",
            "MBS": "batch-size",
            "mbs": "batch-size",
            "batch_size": "batch-size",
            "ACC": "acc",
            "gradient_accumulation_steps": "acc",
            "LR": "learning-rate",
            "learning_rate": "learning-rate",
            "TEM": "template",
            "tem": "template",
            "RESUME": "resume-from-checkpoint",
            "resume_from_checkpoint": "resume-from-checkpoint",
            "gpus_per_node": "gpus-per-node",
            "node_count": "node-count",
            "resource_pool_id": "resource-pool-id",
            "resource_group_id": "resource-group-id",
        }

        def canonical_key(raw_key: str) -> str:
            key = (raw_key or "").strip()
            if not key:
                return key
            return aliases.get(key) or aliases.get(key.lower()) or key.replace("_", "-")

        def set_arg(raw_key: str, raw_value: str) -> None:
            key = canonical_key(raw_key)
            value = str(raw_value or "").strip().rstrip("\\")
            if key and value:
                args[key] = value

        boolean_params = {
            "no-python",
            "skip-preflight",
            "skip-data-analysis",
            "skip-merge",
            "replace-tokenizer-before-merge",
            "replace-tokenizer-after-merge",
        }
        for match in re.finditer(
            r"--([A-Za-z0-9][A-Za-z0-9_-]*)\s+(?:\"([^\"]*)\"|'([^']*)'|((?!--)[^\\\s]+))",
            text or "",
        ):
            key = match.group(1).strip()
            if key == "extra-train-args":
                continue
            value = next((group for group in match.groups()[1:] if group is not None), "")
            if value:
                set_arg(key, value)

        for match in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(/[^\s,，;；]+|[A-Za-z0-9_.:+-]+)",
            text or "",
        ):
            key = match.group(1).strip()
            if key == "extra-train-args":
                continue
            set_arg(key, match.group(2))

        for key in boolean_params:
            if re.search(rf"--{re.escape(key)}(?:\s|$)", text or "") and key not in args:
                args[key] = "true"

        named_values = self._extract_named_param_values(text)
        for key, value in named_values.items():
            mapped_key = canonical_key(key)
            args.setdefault(mapped_key, value)

        natural_patterns = {
            "model-path": r"(?:模型路径|模型位置|基础模型路径|模型在)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            "dataset-dir": r"(?:数据集路径|数据路径|数据集目录|数据目录|数据集位置|数据位置)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            "batch-size": r"(?:批量大小|批次大小|批大小|mbs)\s*(?:是|为|=|:)?\s*([0-9]+)",
            "acc": r"(?:梯度累积|梯度累计|累积步数|acc)\s*(?:是|为|=|:)?\s*([0-9]+)",
            "learning-rate": r"(?:学习率|学习速率|lr)\s*(?:是|为|=|:)?\s*([0-9.eE+-]+)",
            "template": r"(?:模型模板|模型类别|tem)\s*(?:是|为|=|:)?\s*([A-Za-z0-9_.-]+)",
        }
        for key, pattern in natural_patterns.items():
            if key in args:
                continue
            match = re.search(pattern, text or "", re.IGNORECASE)
            if match:
                args[key] = match.group(1).strip()

        container = self._extract_container_override(text)
        if container:
            args["container"] = container
        return args



    def _infer_train_type(self, text: str) -> Optional[str]:
        content = (text or "").lower()
        explicit_type_match = re.search(
            r"(?:训练类型|train_type|training_type|trainType)\s*(?:是|为|=|:|：)?\s*([A-Za-z0-9_\-]+|双机增强训练|双机\s*LoRA\s*SFT|增强训练|全参\s*SFT|全参训练|全参批量训练|定时训练|日常训练|LoRA\s*SFT|lora训练|lora批量训练)",
            text or "",
            re.IGNORECASE,
        )
        if explicit_type_match:
            explicit_type = explicit_type_match.group(1).strip().lower()
            explicit_type_key = re.sub(r"\\s+", "", explicit_type)
            explicit_aliases = {
                "lora": "lora",
                "lora_sft": "lora",
                "lorasft": "lora",
                "sft": "lora",
                "lora_train": "lora",
                "lora训练": "lora",
                "lora批量训练": "lora",
                "full": "full",
                "full_sft": "full",
                "fullsft": "full",
                "全参sft": "full",
                "full_train": "full",
                "全参": "full",
                "全参训练": "full",
                "全参批量训练": "full",
                "enhanced": "enhanced",
                "dpo": "enhanced",
                "增强": "enhanced",
                "增强训练": "enhanced",
                "scheduled": "scheduled",
                "schedule": "scheduled",
                "daily": "scheduled",
                "定时训练": "scheduled",
                "日常训练": "scheduled",
                "grpo": "grpo",
                "grpo_train": "grpo",
                "multinode_lora_sft": "lora",
                "multinodelorasft": "lora",
                "双机lorasft": "lora",
                "multinode_enhanced": "enhanced",
                "multinodeenhanced": "enhanced",
                "双机增强训练": "enhanced",
            }
            if explicit_type_key in explicit_aliases:
                return explicit_aliases[explicit_type_key]
            if explicit_type in explicit_aliases:
                return explicit_aliases[explicit_type]
        # Prefer concrete script names / explicit user-facing train modes over broad
        # context words. Tool prompts may mention "增强训练" as an example even when
        # the actual launched script is batch_train_lora.
        if "train_multinode_sft_pipeline" in content or "双机lora" in content or "双机 sft" in content or "多机lora批量训练" in content or "多机sft" in content:
            return "lora"
        if "train_multinode_dpo_pipeline" in content or "双机增强训练" in content or "双机dpo" in content or "多机增强训练" in content or "多机dpo" in content:
            return "enhanced"

        if "batch_train_lora" in content or "lora_sft" in content or "lora sft" in content or "lora批量训练" in text or "lora训练" in text:
            return "lora"
        if "batch_train_full" in content or "full_sft" in content or "full sft" in content or "全参 SFT" in text or "全参SFT" in text or "全参批量训练" in text or "全参训练" in text:
            return "full"
        if "grpo_train" in content or "grpo" in content or "grpo训练" in text:
            return "grpo"
        if "create_command_vpn" in content or "定时训练" in text or "schedule" in content or "日常训练" in text:
            return "scheduled"
        if "dpo_train_launcher" in content or "增强训练" in text or "dpo" in content:
            return "enhanced"
        return None

    def _normalize_train_type(self, train_type: Optional[str]) -> Optional[str]:
        aliases = {
            "dpo": "enhanced",
            "enhance": "enhanced",
            "enhanced": "enhanced",
            "增强训练": "enhanced",
            "schedule": "scheduled",
            "daily": "scheduled",
            "full": "full",
            "full_train": "full",
            "full_sft": "full",
            "fullsft": "full",
            "全参sft": "full",
            "lora": "lora",
            "lora_train": "lora",
            "lora_sft": "lora",
            "lorasft": "lora",
            "sft": "lora",
            "grpo": "grpo",
            "multinode_lora_sft": "lora",
            "multinodelorasft": "lora",
            "双机lorasft": "lora",
            "multinode_enhanced": "enhanced",
            "multinodeenhanced": "enhanced",
            "双机增强训练": "enhanced",
        }
        value = (train_type or "").strip().lower()
        key = re.sub(r"\s+", "", value)
        return aliases.get(key, aliases.get(value, value)) or None
    def _infer_launch_mode(self, text: str) -> Optional[str]:
        return "multinode" if self._is_multinode_training_request(text or "") else None

    def _train_type_text(self, train_type: Optional[str], launch_mode: Optional[str] = None) -> Optional[str]:
        normalized = self._normalize_train_type(train_type)
        if launch_mode == "multinode":
            if normalized == "lora":
                return "双机 LoRA SFT"
            if normalized == "enhanced":
                return "双机增强训练"
        return {
            "lora": "LoRA SFT",
            "full": "全参 SFT",
            "scheduled": "定时训练",
            "enhanced": "增强训练",
            "grpo": "GRPO",
        }.get(normalized or "")

    def _infer_eval_type(self, text: str) -> Optional[str]:
        content = (text or "").lower()
        if "single_model_evaluation_vpn" in content or "单模型" in text:
            return "single_model"
        if "compare_between_models_vpn" in content or "双模型" in text:
            return "compare_between_models"
        if "ckpt_eval" in content or "ckpt" in content or "checkpoint" in content or "检查点" in text:
            return "ckpt"
        return None

    def _eval_type_text(self, eval_type: Optional[str]) -> Optional[str]:
        return {
            "single_model": "单模型评估",
            "compare_between_models": "双模型评估",
            "ckpt": "checkpoint评估",
        }.get(eval_type or "")

    def _load_runtime_registry_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in (TRAIN_PID_REGISTRY, EVALUATE_PID_REGISTRY, BACKGROUND_TASK_REGISTRY):
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, dict):
                            records.append(item)
            except OSError:
                continue
        return records

    def _lookup_runtime_job_record(self, pid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not pid:
            return None
        matches = [
            record
            for record in self._load_runtime_registry_records()
            if str(record.get("pid") or "") == str(pid)
        ]
        if not matches:
            return None
        return matches[-1]

    def _extract_container(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:容器名称|容器名|容器|container_name|container)\s*(?:是|为|:|：|=)?\s*`?([A-Za-z0-9_.-]+)`?",
            text or "",
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _extract_data_filter_fields(self, text: str) -> Dict[str, Any]:
        content = text or ""
        fields: Dict[str, Any] = {}

        def first_path(patterns: List[str]) -> Optional[str]:
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if not match:
                    continue
                value = match.group(1).strip().rstrip("/。")
                if value.startswith("/"):
                    return value
            return None

        input_folder = first_path([
            r"(?:--input_folder|input_folder|inputFolder|输入目录|输入路径|输入数据|处理路径(?:为)?|处理路径为)\s*(?:是|为|:|：|=)?\s*`?([^`\s\n，,。]+)`?",
        ])
        output_folder = first_path([
            r"(?:--output_folder|output_folder|outputFolder|输出目录|输出路径|输出数据)\s*(?:是|为|:|：|=)?\s*`?([^`\s\n，,。]+)`?",
            r"(?:保存(?:在|到)|保存在)\s*`?(/[^`\s\n，,。]+)`?",
        ])
        threshold_match = re.search(
            r"(?:--threshold|threshold|阈值|筛选阈值)[^0-9\n]{0,16}([0-9.]+)",
            content,
            re.IGNORECASE,
        )
        if input_folder:
            fields["inputFolder"] = input_folder
        if output_folder:
            fields["outputFolder"] = output_folder
            fields["outputDatasetName"] = output_folder.rstrip("/").split("/")[-1]
        if threshold_match:
            try:
                fields["threshold"] = float(threshold_match.group(1))
            except ValueError:
                pass
        return fields

    def _extract_pid(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:进程\s*ID\s*(?:\(\s*PID\s*\))?|进程ID|PID|pid|进程号)[^\d\n]{0,40}`?(\d+)`?",
            text or "",
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _job_started_protocol(
        self,
        agent_name: str,
        task_description: str,
        additional_params: str,
        display_text: str,
        raw_text: str,
    ) -> Optional[Dict[str, Any]]:
        combined = "\n".join([task_description or "", additional_params or "", display_text or "", raw_text or ""])
        if not any(keyword in combined for keyword in ["后台启动", "后台运行", "已启动", "进程ID", "PID"]):
            return None
        job_type = self._infer_job_type(agent_name, combined)
        if not job_type:
            return None
        pid = self._extract_pid(combined)
        registry_record = self._lookup_runtime_job_record(pid)
        registry_script = str((registry_record or {}).get("script_name") or "").strip()
        registry_train_type = str((registry_record or {}).get("train_type") or "").strip()
        registry_eval_type = str(
            (registry_record or {}).get("assessmentType")
            or (registry_record or {}).get("evalType")
            or (registry_record or {}).get("eval_type")
            or ""
        ).strip()
        registry_launch_mode = str((registry_record or {}).get("launchMode") or (registry_record or {}).get("launch_mode") or "").strip()
        container = self._extract_container(combined) or (registry_record or {}).get("container")
        protocol = self._with_protocol(
            "job_started",
            agent_name,
            self._strip_user_waiting_wrapper(display_text),
            jobType=job_type,
            container=container,
            pid=pid,
        )
        if agent_name == "trainer":
            train_type = self._normalize_train_type(
                registry_train_type or self._infer_train_type(f"{combined}\n{registry_script}")
            )
            launch_mode = registry_launch_mode or self._infer_launch_mode(f"{combined}\n{registry_script}")
            if train_type:
                protocol["trainType"] = train_type
                train_type_text = self._train_type_text(train_type, launch_mode)
                if train_type_text:
                    protocol["trainTypeText"] = train_type_text
            if launch_mode:
                protocol["launchMode"] = launch_mode
                protocol["isMultinode"] = launch_mode == "multinode"
            if registry_script:
                protocol["script"] = registry_script
        if agent_name == "evaluator":
            eval_type = registry_eval_type or self._infer_eval_type(f"{combined}\n{registry_script}")
            if eval_type:
                protocol["assessmentType"] = eval_type
                protocol["evalType"] = eval_type
                eval_type_text = self._eval_type_text(eval_type)
                if eval_type_text:
                    protocol["assessmentTypeText"] = eval_type_text
                    protocol["evalTypeText"] = eval_type_text
            if registry_script:
                protocol["script"] = registry_script
        if agent_name == "dataprocessor":
            protocol["script"] = registry_script or ("score_based_filtering" if "score_based_filtering" in combined else "data_preprocessing")
            if protocol.get("jobType") == "data_filter":
                protocol.setdefault("container", container or self.training_container)
                protocol.update(self._extract_data_filter_fields(combined))
        return self._mark_workflow_child_protocol(protocol)

    def _mark_workflow_child_protocol(self, protocol: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not protocol or not self._workflow_child_context:
            return protocol
        if protocol.get("type") not in {"job_started", "job_preparing"}:
            return protocol
        workflow = self._workflow_child_context.get("workflow")
        stage_name = self._workflow_child_context.get("stage")
        workflow_fields: Dict[str, Any] = {}
        if isinstance(workflow, dict):
            context = workflow.get("context") or {}
            train_type = self._normalize_train_type(context.get("train_type"))
            launch_mode = context.get("launch_mode")
            stages = self._workflow_public_stages(workflow)
            if stage_name:
                stage = dict(stages.get(stage_name) or {})
                stage["status"] = (
                    "preparing" if protocol.get("type") == "job_preparing"
                    else (stage.get("status") or "running")
                )
                stage["pid"] = stage.get("pid") or protocol.get("pid")
                stage["container"] = stage.get("container") or protocol.get("container")
                stages[stage_name] = {key: value for key, value in stage.items() if value is not None}
            workflow_fields = {
                "workflowStatus": workflow.get("status"),
                "currentStage": workflow.get("current_stage") or stage_name,
                "workflowKey": workflow.get("workflow_key"),
                "datasetRef": workflow.get("dataset_ref"),
                "trainType": train_type,
                "trainTypeText": self._train_type_text(train_type, launch_mode),
                "launchMode": launch_mode,
                "isMultinode": launch_mode == "multinode",
                "trainArgs": context.get("train_args"),
                "stages": stages,
                "benchmark": context.get("benchmark"),
                "evaluationDatasetName": context.get("evaluation_dataset_name"),
                "benchmarkResultEntry": _benchmark_result_entry(workflow),
            }
        elif stage_name:
            workflow_fields = {
                "workflowStatus": "running",
                "currentStage": stage_name,
                "stages": {
                    stage_name: {
                        "status": "running",
                        "pid": protocol.get("pid"),
                        "container": protocol.get("container"),
                    },
                },
            }
        return {
            **protocol,
            "workflowId": self._workflow_child_context.get("workflow_id"),
            "workflowChild": True,
            "workflowStage": self._workflow_child_context.get("stage"),
            **{key: value for key, value in workflow_fields.items() if value is not None},
        }

    def _message_protocol(self, agent_name: str, message: str, protocol_type: str = "message") -> Dict[str, Any]:
        return self._with_protocol(protocol_type, agent_name, message)

    def _normalize_inference_status(self, status_text: str) -> str:
        text = (status_text or "").strip().lower()
        if any(keyword in text for keyword in ["启动中", "starting"]):
            return "starting"
        if any(keyword in text for keyword in ["降级", "degraded"]):
            return "degraded"
        if any(keyword in text for keyword in ["运行", "running", "started", "已启动"]):
            return "running"
        if any(keyword in text for keyword in ["停止", "stopped", "未运行", "not running"]):
            return "stopped"
        if any(keyword in text for keyword in ["失败", "failed", "异常", "error"]):
            return "failed"
        if any(keyword in text for keyword in ["超时", "timeout"]):
            return "timeout"
        return "unknown"

    def _normalize_inference_config_value(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()      
        for _ in range(2):
            text = text.rstrip(",").strip().strip("\"'").strip()
        text = re.sub(r"\s*[（(][^）)]*[）)]\s*$", "", text).strip()
        percent_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", text)
        if percent_match:
            return float(percent_match.group(1)) / 100
        number_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if number_match and re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s*(?:秒|tokens?))?", text, re.IGNORECASE):
            number = number_match.group(0)
            return float(number) if "." in number else int(number)
        return text

    def _canonical_inference_config_key(self, key: str) -> Optional[str]:
        normalized = re.sub(r"\s+", "", str(key or "")).strip()
        aliases = {
            "主机IP": "HOST_IP",
            "CUDA可见设备": "CUDA_VISIBLE_DEVICES",
            "可见GPU设备": "CUDA_VISIBLE_DEVICES",
            "可见GPU": "CUDA_VISIBLE_DEVICES",
            "模型名称": "MODEL_NAME",
            "模型参数量": "MODEL_PARAM_B",
            "主端口": "MASTER_PORT",
            "模型路径": "MODEL_PATH",
            "启动脚本": "START_SCRIPT",
            "启动脚本路径": "START_SCRIPT",
            "日志目录": "LOG_DIR",
            "测试目录": "TEST_DIR",
            "医疗基准目录": "BENCHMARK_DIR",
            "医疗基准测试目录": "BENCHMARK_DIR",
            "通用基准目录": "GENERAL_BENCHMARK_DIR",
            "通用基准测试目录": "GENERAL_BENCHMARK_DIR",
            "代码评估执行器": "HUMANEVAL_EXECUTOR",
            "代码评估镜像": "HUMANEVAL_DOCKER_IMAGE",
            "代码评估超时": "HUMANEVAL_TIMEOUT",
            "代码评估内存限制": "HUMANEVAL_MEMORY",
            "代码评估CPU数": "HUMANEVAL_CPUS",
            "代码评估进程数限制": "HUMANEVAL_PIDS_LIMIT",
            "张量并行规模": "TENSOR_PARALLEL_SIZE",
            "GPU内存利用率": "GPU_MEMORY_UTILIZATION",
            "GPU显存利用率": "GPU_MEMORY_UTILIZATION",
            "GPU利用率阈值": "GPU_UTILIZATION_THRESHOLD",
            "最大上下文长度": "MAX_TOKENS",
        }
        if re.fullmatch(r"[A-Za-z0-9_]+", normalized):
            return normalized
        return aliases.get(normalized)

    def _ingest_inference_config_mapping(self, config_data: Dict[str, Dict[str, Any]], payload: Dict[str, Any]) -> None:
        for raw_section, values in payload.items():
            if not isinstance(values, dict):
                continue
            section_text = str(raw_section)
            if "PORTS" in section_text.upper() or "端口" in section_text:
                section = "ports"
            elif "ENV" in section_text.upper() or "环境" in section_text:
                section = "env"
            elif "RUNTIME" in section_text.upper() or "运行时" in section_text or "参数" in section_text:
                section = "runtime"
            else:
                continue
            for raw_key, raw_value in values.items():
                key = self._canonical_inference_config_key(str(raw_key))
                if not key:
                    continue
                config_data[section][key] = self._normalize_inference_config_value(raw_value)

    def _section_for_inference_json_context(self, context: str, payload: Dict[str, Any]) -> Optional[str]:
        upper_context = context.upper()
        if "PORT" in upper_context or "端口" in context:
            return "ports"
        if "RUNTIME" in upper_context or "运行时" in context or "参数" in context:
            return "runtime"
        if "ENV" in upper_context or "环境" in context:
            return "env"
        port_keys = {"VLLM_OPENAI_PORT", "INFERENCE_PORT", "UI_PORT", "DATA_ANNOTATION_PORT"}
        runtime_keys = {"TENSOR_PARALLEL_SIZE", "GPU_MEMORY_UTILIZATION", "GPU_UTILIZATION_THRESHOLD", "MAX_TOKENS"}
        keys = {str(key) for key in payload.keys()}
        if keys & port_keys:
            return "ports"
        if keys & runtime_keys:
            return "runtime"
        if any(self._canonical_inference_config_key(key) for key in keys):
            return "env"
        return None

    def _extract_inference_config(self, text: str) -> Dict[str, Dict[str, Any]]:
        config_data: Dict[str, Dict[str, Any]] = {"ports": {}, "env": {}, "runtime": {}}
        for match in re.finditer(r"```json\s*\n(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL):
            block = match.group(1)
            try:
                payload = json.loads(block)
            except Exception:
                continue
            if isinstance(payload, dict):
                self._ingest_inference_config_mapping(config_data, payload)
                context = (text or "")[max(0, match.start() - 180):match.start()]
                section = self._section_for_inference_json_context(context, payload)
                if section:
                    for raw_key, raw_value in payload.items():
                        key = self._canonical_inference_config_key(str(raw_key))
                        if key:
                            config_data[section][key] = self._normalize_inference_config_value(raw_value)

        section = None
        section_aliases = {
            "PORTS": "ports",
            "ENV": "env",
            "RUNTIME": "runtime",
        }
        for line in (text or "").splitlines():
            heading = re.search(r"\((PORTS|ENV|RUNTIME)\)", line, re.IGNORECASE)
            if heading:
                section = section_aliases.get(heading.group(1).upper())
                continue
            yaml_heading = re.match(r"\s*(PORTS|ENV|RUNTIME)\s*:\s*$", line, re.IGNORECASE)
            if yaml_heading:
                section = section_aliases.get(yaml_heading.group(1).upper())
                continue
            item = re.match(r"\s*-\s*(?:\*\*)?([^:：\n]+?)(?:\*\*)?\s*[:：]\s*(.+?)\s*$", line)
            if not item:
                item = re.match(r"\s+([^:：\n]+?)\s*[:：]\s*(.+?)\s*$", line)
            if not item or not section:
                continue
            key = self._canonical_inference_config_key(item.group(1)) or item.group(1)
            value = item.group(2).strip().strip("*").strip().strip("\"'")
            config_data[section][key] = self._normalize_inference_config_value(value)

        table_pattern = re.compile(
            r"^\|\s*(?:\*\*)?([^|*`]+?)(?:\*\*)?\s*\|\s*`?([^`|]+)`?\s*\|.*$",
            re.MULTILINE,
        )
        for match in table_pattern.finditer(text or ""):
            key = self._canonical_inference_config_key(match.group(1))
            if not key:
                continue
            value = self._normalize_inference_config_value(match.group(2).strip())
            if key in {"TENSOR_PARALLEL_SIZE", "GPU_MEMORY_UTILIZATION", "GPU_UTILIZATION_THRESHOLD", "MAX_TOKENS"}:
                config_data["runtime"][key] = value
            elif key in {"VLLM_OPENAI_PORT", "INFERENCE_PORT", "UI_PORT", "DATA_ANNOTATION_PORT"}:
                config_data["ports"][key] = value
            else:
                if isinstance(value, str) and "=" in value:
                    maybe_key, maybe_value = value.split("=", 1)
                    canonical = self._canonical_inference_config_key(maybe_key)
                    if canonical:
                        config_data["env"][canonical] = self._normalize_inference_config_value(maybe_value)
                        continue
                config_data["env"][key] = value
        return {key: value for key, value in config_data.items() if value}

    def _extract_inference_services(self, text: str) -> List[Dict[str, Any]]:
        services: List[Dict[str, Any]] = []
        key_pattern = re.compile(
            r"^\s*-\s*([A-Za-z0-9_]+)(?:\s*\((?:端口|port)\s*(\d+)\))?\s*[：:]\s*(.+?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in key_pattern.finditer(text or ""):
            raw_status = match.group(3).strip()
            if not match.group(2) and self._normalize_inference_status(raw_status) == "unknown":
                continue
            services.append(
                {
                    "name": match.group(1),
                    "port": int(match.group(2)) if match.group(2) else None,
                    "status": self._normalize_inference_status(raw_status),
                    "rawStatus": raw_status,
                }
            )
        display_name_map = {
            "VLLM 服务": "VLLM_OPENAI_PORT",
            "vLLM API": "VLLM_OPENAI_PORT",
            "vLLM OpenAI API": "VLLM_OPENAI_PORT",
            "VLLM OpenAI API": "VLLM_OPENAI_PORT",
            "vLLM OpenAI 兼容 API": "VLLM_OPENAI_PORT",
            "VLLM服务": "VLLM_OPENAI_PORT",
            "vLLM 服务": "VLLM_OPENAI_PORT",
            "Inference Server": "INFERENCE_PORT",
            "推理服务": "INFERENCE_PORT",
            "Web UI": "UI_PORT",
            "UI 服务": "UI_PORT",
            "UI服务": "UI_PORT",
            "Case2Chat": "DATA_ANNOTATION_PORT",
            "Data Annotation": "DATA_ANNOTATION_PORT",
            "数据标注服务": "DATA_ANNOTATION_PORT",
        }
        display_pattern = re.compile(
            r"^\s*-?\s*(?:[^\w\s(（:：-]+\s*)?(.+?)\s*[（(]\s*(\d+)\s*(?:端口|port)?\s*[）)]\s*[：:]\s*(.+?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        seen = {(service.get("name"), service.get("port")) for service in services}
        for match in display_pattern.finditer(text or ""):
            label = re.sub(r"\s+", " ", match.group(1).strip())
            port = int(match.group(2))
            raw_status = match.group(3).strip()
            normalized_status = self._normalize_inference_status(raw_status)
            if normalized_status == "unknown":
                continue
            name = display_name_map.get(label, label)
            if (name, port) in seen:
                continue
            services.append(
                {
                    "name": name,
                    "displayName": label,
                    "port": port,
                    "status": normalized_status,
                    "rawStatus": raw_status,
                }
            )
            seen.add((name, port))
        table_pattern = re.compile(r"^\s*\|(.+?)\|\s*$", re.MULTILINE)
        for match in table_pattern.finditer(text or ""):
            cells = [cell.strip().strip("*` ") for cell in match.group(1).split("|")]
            if len(cells) < 3:
                continue
            if all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
                continue
            service_label, port_text, raw_status = cells[0], cells[1], cells[2]
            header_text = "".join(cells).lower()
            if any(keyword in header_text for keyword in ["服务组件端口状态", "servicecomponentportstatus"]):
                continue
            port_match = re.search(r"\d{2,5}", port_text)
            if not service_label or not port_match:
                continue
            port = int(port_match.group(0))
            normalized_status = self._normalize_inference_status(raw_status)
            if normalized_status == "unknown":
                continue
            label = re.sub(r"\s+", " ", service_label)
            name = display_name_map.get(label, label)
            if (name, port) in seen:
                continue
            services.append(
                {
                    "name": name,
                    "displayName": label,
                    "port": port,
                    "status": normalized_status,
                    "rawStatus": raw_status,
                }
            )
            seen.add((name, port))
        return services

    def _normalize_inference_config_payload(
        self, payload: Any
    ) -> Dict[str, Dict[str, Any]]:
        
        if not isinstance(payload, dict):
            return {}
        config_data: Dict[str, Dict[str, Any]] = {"ports": {}, "env": {}, "runtime": {}}
        self._ingest_inference_config_mapping(config_data, payload)
        return {key: value for key, value in config_data.items() if value}


    def _normalize_inference_services_payload(
        self, payload: Any, node: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        
        if not isinstance(payload, list):
            return []
        services: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            service = {
                "name": str(name),
                "port": item.get("port"),
                "status": self._normalize_inference_status(item.get("status")),
                "rawStatus": item.get("rawStatus") or item.get("status"),
            }
            if item.get("displayName"): 
                service["displayName"] = item.get("displayName") 
            if node:
                service["node"] = node
            services.append(service)
        return services     

    def _normalize_inference_port_statuses_payload(
        self,
        payload: Any,
        *,
        fallback_ports: Any = None,
        fallback_status: Any = None,
    ) -> List[Dict[str, Any]]:
        source_items = payload if isinstance(payload, list) else []
        port_statuses: List[Dict[str, Any]] = []
        for item in source_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("key") or item.get("service") or "").strip()
            port = item.get("port")
            if not name or port in (None, ""):
                continue
            status = self._normalize_inference_status(item.get("status"))
            if status == "unknown":
                status = self._normalize_inference_status(fallback_status)
            port_statuses.append(
                {
                    "key": name,
                    "name": name,
                    "displayName": item.get("displayName") or item.get("label") or name,
                    "port": port,
                    "status": status,
                    "rawStatus": item.get("rawStatus") or item.get("status") or status,
                }
            )
        if port_statuses or not isinstance(fallback_ports, dict):
            return port_statuses
        fallback = self._normalize_inference_status(fallback_status)
        for name in ("vllm", "inference", "ui", "case2chat"):
            port = fallback_ports.get(name)
            if port in (None, ""):
                continue
            port_statuses.append(
                {
                    "key": name,
                    "name": name,
                    "displayName": name,
                    "port": port,
                    "status": fallback if fallback != "unknown" else "stopped",
                    "rawStatus": fallback if fallback != "unknown" else "stopped",
                }
            )
        return port_statuses

    def _normalize_inference_service_instances_payload(
        self, payload: Any
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        normalized = dict(payload)
        items = payload.get("items")
        if not isinstance(items, list):
            return normalized
        normalized_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            port_statuses = self._normalize_inference_port_statuses_payload(
                normalized_item.get("port_statuses") or normalized_item.get("portStatuses"),
                fallback_ports=normalized_item.get("ports"),
                fallback_status=normalized_item.get("status"),
            )
            if port_statuses:
                normalized_item["port_statuses"] = port_statuses
            normalized_items.append(normalized_item)
        normalized["items"] = normalized_items
        return normalized

    def _inference_nodes_from_response_data(
        self, response_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(response_data, dict):
            return {}
        payload = response_data.get("nodes")
        if not isinstance(payload, dict):
            return {}

        nodes: Dict[str, Dict[str, Any]] = {}
        for node_key, node_payload in payload.items():
            if not isinstance(node_payload, dict):
                continue
            node_name = str(node_key)
            node_info: Dict[str, Any] = {}

            config_payload = node_payload.get("config")
            if not isinstance(config_payload, dict):
                config_draft = node_payload.get("config_draft")
                if isinstance(config_draft, dict):
                    config_payload = config_draft.get("values")
            config_data = self._normalize_inference_config_payload(config_payload)
            if config_data:
                node_info["config"] = config_data

            services = self._normalize_inference_services_payload(
                node_payload.get("services"),
                node=node_name,
            )
            if services:
                node_info["services"] = services
                node_info["allStopped"] = all(
                    service.get("status") == "stopped" for service in services
                )
                node_info["allRunning"] = all(
                    service.get("status") == "running" for service in services
                )

            service_instances = node_payload.get("service_instances")
            if isinstance(service_instances, dict):
                node_info["service_instances"] = self._normalize_inference_service_instances_payload(service_instances) or service_instances

            service_instance = node_payload.get("service_instance")
            if isinstance(service_instance, dict):
                node_info["service_instance"] = service_instance

            for structured_key in ("test_runs", "test_run_stop", "admin_cleanup", "benchmark_stop"):
                structured_value = node_payload.get(structured_key)
                if isinstance(structured_value, dict):
                    node_info[structured_key] = structured_value

            benchmark_jobs = node_payload.get("benchmark_jobs")
            if isinstance(benchmark_jobs, dict):
                node_info["benchmark_jobs"] = benchmark_jobs

            benchmark_reports = node_payload.get("benchmark_reports")
            if isinstance(benchmark_reports, list):
                node_info["benchmark_reports"] = [
                    report for report in benchmark_reports if isinstance(report, dict)
                ]

            service_start = node_payload.get("service_start")
            if isinstance(service_start, dict):
                node_info["service_start"] = service_start

            if node_info:
                nodes[node_name] = node_info
        return nodes

    def _first_inference_config_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        for node_data in nodes.values():
            config_data = node_data.get("config")
            if isinstance(config_data, dict) and self._has_inference_config_data(config_data):
                return config_data
        return {}

    def _inference_services_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        services: List[Dict[str, Any]] = []
        for node_data in nodes.values():
            node_services = node_data.get("services")
            if isinstance(node_services, list):
                services.extend(
                    service for service in node_services if isinstance(service, dict)
                )
        return services
       
    def _first_inference_service_instances_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for node_data in nodes.values():
            service_instances = node_data.get("service_instances")
            if isinstance(service_instances, dict):
                return service_instances
        return None

    def _first_inference_service_instance_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for node_data in nodes.values():
            service_instance = node_data.get("service_instance")
            if isinstance(service_instance, dict):
                return service_instance
        return None

    def _inference_benchmark_reports_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        for node_data in nodes.values():
            node_reports = node_data.get("benchmark_reports")
            if isinstance(node_reports, list):
                reports.extend(report for report in node_reports if isinstance(report, dict))
        return reports

    def _first_inference_benchmark_jobs_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for node_data in nodes.values():
            benchmark_jobs = node_data.get("benchmark_jobs")
            if isinstance(benchmark_jobs, dict):
                return benchmark_jobs
        return None

    def _first_inference_structured_from_nodes(
        self, nodes: Dict[str, Dict[str, Any]], key: str
    ) -> Optional[Dict[str, Any]]:
        for node_data in nodes.values():
            value = node_data.get(key)
            if isinstance(value, dict):
                return value
        return None


    def _extract_inline_commands(self, text: str) -> List[str]:
        commands: List[str] = []
        for lang, block in re.findall(r"```([A-Za-z0-9_-]*)\s*\n(.*?)```", text or "", flags=re.DOTALL):
            if (lang or "").lower() not in {"bash", "sh", "shell", "console", "terminal", "python", "py"}:
                continue
            for line in block.splitlines():
                command = line.split("#", 1)[0].strip()
                if command:
                    commands.append(command)
        commands.extend(
            command.strip()
            for command in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text or "")
            if command.strip() and re.search(r"[A-Za-z_]+(?:\s|\()", command)
        )
        return list(dict.fromkeys(commands))

    def _has_inference_config_data(self, config_data: Dict[str, Dict[str, Any]]) -> bool:
        return any(bool(values) for values in (config_data or {}).values())

    def _infer_inference_action(
        self,
        task_description: str,
        result: str,
        config_data: Optional[Dict[str, Dict[str, Any]]] = None,
        services: Optional[List[Dict[str, Any]]] = None,
        service_instances: Optional[Dict[str, Any]] = None,
        service_instance: Optional[Dict[str, Any]] = None,
        service_stop: Optional[Dict[str, Any]] = None,
        benchmark_reports: Optional[List[Dict[str, Any]]] = None,
        benchmark_jobs: Optional[Dict[str, Any]] = None,
        benchmark_stop: Optional[Dict[str, Any]] = None,
        test_runs: Optional[Dict[str, Any]] = None,
        test_run_stop: Optional[Dict[str, Any]] = None,
        admin_cleanup: Optional[Dict[str, Any]] = None,
        service_start: Optional[Dict[str, Any]] = None,
    ) -> str:
        task = (task_description or "").lower()
        combined = f"{task_description}\n{result}".lower()
        if isinstance(service_instance, dict):
            return "instance_status"
        if isinstance(service_stop, dict):
            operation = str(service_stop.get("operation") or "").strip().lower()
            if operation == "preview":
                return "service_stop_preview"
        if isinstance(service_start, dict) and service_start.get("submitted") is True:
            return "service_start"
        if any(
            keyword in combined
            for keyword in ["当前请求不是管理员角色", "已拒绝调用管理员维护工具", "not admin", "admin role"]
        ):
            return "message"
        if isinstance(admin_cleanup, dict):
            operation = str(admin_cleanup.get("operation") or "").strip().lower()
            return "admin_cleanup_apply" if operation == "apply" else "admin_cleanup_preview"
        if isinstance(test_run_stop, dict):
            operation = str(test_run_stop.get("operation") or "").strip().lower()
            return "test_stop" if operation == "apply" else "test_stop_preview"
        if isinstance(test_runs, dict):
            operation = str(test_runs.get("operation") or "").strip().lower()
            return "test_running_list" if operation == "list_running" else "test_list"
        if isinstance(benchmark_stop, dict):
            operation = str(benchmark_stop.get("operation") or "").strip().lower()
            return "benchmark_stop_preview" if operation == "preview" else "benchmark_stop"
        if isinstance(benchmark_jobs, dict):
            operation = str(benchmark_jobs.get("operation") or "").strip().lower()
            if operation == "list":
                return "benchmark_list"
        if benchmark_reports:
            if any(
                keyword in task
                for keyword in ["列表", "所有", "全部", "有哪些", "可用", "list", "available"]
            ):
                return "benchmark_list"
            if any(keyword in task for keyword in ["结果", "result", "报告", "report"]):
                return "benchmark_result"
            if any(keyword in task for keyword in ["进度", "progress"]):
                return "benchmark_progress"
            if len(benchmark_reports) > 1:
                return "benchmark_list"
            return "benchmark_status"
        if isinstance(service_instances, dict):
            operation = str(service_instances.get("operation") or "").strip().lower()
            if operation == "list":
                return "instance_list"
            if operation == "status":
                return "service_status"
            if service_instances.get("items") is not None:
                if any(keyword in task for keyword in ["列表", "有哪些", "list", "instances"]):
                    return "instance_list"
                return "service_status"
        if any(keyword in task for keyword in ["清理", "残留", "cleanup", "维护", "管理员", "admin"]):
            return "message"
        if config_data and self._has_inference_config_data(config_data):
            if any(keyword in task for keyword in ["修改", "更新", "设置", "change", "update", "set"]):
                return "config_update"
            return "config_view"
        if services:
            if any(keyword in task for keyword in ["启动", "start"]):
                return "service_start"
            if any(keyword in task for keyword in ["重启", "restart"]):
                return "service_restart"
            if any(keyword in task for keyword in ["停止", "关闭", "stop"]):
                return "service_stop"
            return "service_status"
        if any(keyword in task for keyword in ["日志", "log"]):
            return "logs"
        if any(keyword in task for keyword in ["配置", "config"]):
            if any(keyword in task for keyword in ["修改", "更新", "设置", "change", "update", "set"]):
                return "config_update"
            return "config_view"
        benchmark_task = any(keyword in task for keyword in ["基准", "benchmark", "评测", "测评", "medbench"])
        if benchmark_task or any(keyword in combined for keyword in ["benchmark", "medbench", "step1", "step2", "step3"]):
            if any(keyword in task for keyword in ["可用", "列表", "有哪些", "查看", "list", "available"]) and not any(
                keyword in task for keyword in ["进度", "结果", "状态", "停止", "关闭", "运行", "启动", "执行", "progress", "result", "status", "stop", "run", "start"]
            ):
                return "benchmark_list"
            if any(keyword in task for keyword in ["进度", "progress"]):
                return "benchmark_progress"
            if any(keyword in task for keyword in ["结果", "result"]):
                return "benchmark_result"
            if any(keyword in task for keyword in ["状态", "status"]):
                return "benchmark_status"
            if any(keyword in task for keyword in ["停止", "关闭", "stop"]):
                if any(keyword in task for keyword in ["预览", "preview", "dry-run", "dry run"]):
                    return "benchmark_stop_preview"
                return "benchmark_stop"
            if any(keyword in task for keyword in ["运行", "启动", "执行", "run", "start"]):
                return "benchmark_start"
            if any(keyword in combined for keyword in ["running", "stopped", "finished", "failed"]):
                return "benchmark_status"
            return "benchmark"
        if any(keyword in combined for keyword in ["测试脚本", "功能测试", "basicmedicalrecord", "test script"]):
            if any(keyword in task for keyword in ["运行", "执行", "run"]):
                return "test_run"
            return "test_list"
        if any(keyword in task for keyword in ["状态", "status"]):
            return "service_status"
        if any(keyword in task for keyword in ["重启", "restart"]):
            return "service_restart"
        if any(keyword in task for keyword in ["启动", "start"]):
            return "service_start"
        if any(keyword in task for keyword in ["停止", "关闭", "stop"]):
            return "service_stop"
        if any(keyword in combined for keyword in ["配置", "config"]):
            return "config_view"
        if any(keyword in combined for keyword in ["状态", "status"]):
            return "service_status"
        return "message"

    def _inference_protocol_for_result(
        self,
        task_description: str,
        result: str,
        response_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:    
        nodes = self._inference_nodes_from_response_data(response_data)
        top_level_config: Dict[str, Dict[str, Any]] = {}
        if isinstance(response_data, dict):
            config_draft = response_data.get("config_draft")
            if isinstance(config_draft, dict):
                top_level_config = self._normalize_inference_config_payload(config_draft.get("values"))
            if not top_level_config:
                top_level_config = self._normalize_inference_config_payload(response_data.get("config"))
        config_data = top_level_config or self._first_inference_config_from_nodes(nodes)
        services = self._inference_services_from_nodes(nodes)
        if not services:
            services = self._extract_inference_services(result)
        service_instances = response_data.get("service_instances") if isinstance(response_data, dict) else None
        service_instance = response_data.get("service_instance") if isinstance(response_data, dict) else None
        benchmark_reports = response_data.get("benchmark_reports") if isinstance(response_data, dict) else None
        benchmark_jobs = response_data.get("benchmark_jobs") if isinstance(response_data, dict) else None
        service_stop = response_data.get("service_stop") if isinstance(response_data, dict) else None
        service_start = response_data.get("service_start") if isinstance(response_data, dict) else None
        benchmark_stop = response_data.get("benchmark_stop") if isinstance(response_data, dict) else None
        test_runs = response_data.get("test_runs") if isinstance(response_data, dict) else None
        test_run_stop = response_data.get("test_run_stop") if isinstance(response_data, dict) else None
        admin_cleanup = response_data.get("admin_cleanup") if isinstance(response_data, dict) else None
        if not isinstance(service_instances, dict):
            service_instances = self._first_inference_service_instances_from_nodes(nodes)
        if isinstance(service_instances, dict):
            service_instances = self._normalize_inference_service_instances_payload(service_instances) or service_instances
        if not isinstance(service_instance, dict):
            service_instance = self._first_inference_service_instance_from_nodes(nodes)
        if not isinstance(benchmark_reports, list):
            benchmark_reports = self._inference_benchmark_reports_from_nodes(nodes)
        if not isinstance(service_stop, dict):
            service_stop = self._first_inference_structured_from_nodes(nodes, "service_stop")
        if not isinstance(service_start, dict):
            service_start = self._first_inference_structured_from_nodes(nodes, "service_start")
        if not isinstance(benchmark_jobs, dict):
            benchmark_jobs = self._first_inference_benchmark_jobs_from_nodes(nodes)
        if not isinstance(benchmark_stop, dict):
            benchmark_stop = self._first_inference_structured_from_nodes(nodes, "benchmark_stop")
        if not isinstance(test_runs, dict):
            test_runs = self._first_inference_structured_from_nodes(nodes, "test_runs")
        if not isinstance(test_run_stop, dict):
            test_run_stop = self._first_inference_structured_from_nodes(nodes, "test_run_stop")
        if not isinstance(admin_cleanup, dict):
            admin_cleanup = self._first_inference_structured_from_nodes(nodes, "admin_cleanup")
        action = self._infer_inference_action(
            task_description,
            result,
            config_data=config_data,
            services=services,
            service_instances=service_instances if isinstance(service_instances, dict) else None,
            service_instance=service_instance if isinstance(service_instance, dict) else None,
            service_stop=service_stop if isinstance(service_stop, dict) else None,
            service_start=service_start if isinstance(service_start, dict) else None,
            benchmark_reports=benchmark_reports if isinstance(benchmark_reports, list) else None,
            benchmark_jobs=benchmark_jobs if isinstance(benchmark_jobs, dict) else None,
            benchmark_stop=benchmark_stop if isinstance(benchmark_stop, dict) else None,
            test_runs=test_runs if isinstance(test_runs, dict) else None,
            test_run_stop=test_run_stop if isinstance(test_run_stop, dict) else None,
            admin_cleanup=admin_cleanup if isinstance(admin_cleanup, dict) else None,
        )
        if _inference_service_start_payload_submitted(response_data):
            action = "service_start"
        fields: Dict[str, Any] = {"action": action}
        if isinstance(service_instances, dict):
            fields["service_instances"] = service_instances
        if isinstance(service_instance, dict):
            fields["service_instance"] = service_instance
        if isinstance(service_stop, dict):
            fields["service_stop"] = service_stop
        if isinstance(benchmark_jobs, dict):
            fields["benchmark_jobs"] = benchmark_jobs
        if isinstance(benchmark_stop, dict):
            benchmark_stop = dict(benchmark_stop)
            benchmark_stop_operation = str(benchmark_stop.get("operation") or "").strip().lower()
            if benchmark_stop_operation == "apply" and benchmark_stop.get("stopped") is not False:
                benchmark_stop.setdefault("stopped", True)
                benchmark_stop.setdefault("status", "stopped")
            fields["benchmark_stop"] = benchmark_stop
            if benchmark_stop.get("stopped") is True:
                fields.setdefault("status", "stopped")
                benchmark_stop_job_id = benchmark_stop.get("job_id") or benchmark_stop.get("jobId")
                if benchmark_stop_job_id is not None:
                    fields.setdefault("jobId", benchmark_stop_job_id)
                    fields.setdefault("job_id", benchmark_stop_job_id)
        if isinstance(test_runs, dict):
            fields["test_runs"] = test_runs
        if isinstance(test_run_stop, dict):
            fields["test_run_stop"] = test_run_stop
        if isinstance(admin_cleanup, dict):
            fields["admin_cleanup"] = admin_cleanup
        if isinstance(benchmark_reports, list) and benchmark_reports:
            fields["benchmark_reports"] = benchmark_reports
            first_report = next((item for item in benchmark_reports if isinstance(item, dict)), {})
            if isinstance(first_report, dict):
                for source_key, protocol_key in (
                    ("job_id", "jobId"),
                    ("benchmark_job_id", "benchmark_job_id"),
                    ("status", "status"),
                    ("model", "model"),
                    ("dataset", "dataset"),
                    ("result_path", "resultPath"),
                ):
                    if first_report.get(source_key) is not None:
                        fields.setdefault(protocol_key, first_report.get(source_key))
        if isinstance(service_start, dict):
            fields["service_start"] = service_start
        if services:
            fields["services"] = services
        if config_data and self._has_inference_config_data(config_data):
            fields["config"] = config_data
        if nodes:
            fields["nodes"] = nodes
        commands = self._extract_inline_commands(result)
        if commands:
            fields["commands"] = commands

        if action == "config_view":
            return self._with_protocol("inference_config", "inference", result, jobType="inference_config", **fields)
        if action == "config_update":
            return self._with_protocol("inference_config_updated", "inference", result, jobType="inference_config", **fields)
        if action == "instance_list":
            return self._with_protocol("inference_instance_list", "inference", result, jobType="inference_instance", **fields)
        if action == "instance_status":
            return self._with_protocol("inference_instance_status", "inference", result, jobType="inference_instance", **fields)
        if action in {"service_status", "service_instances"}:
            return self._with_protocol("inference_status", "inference", result, jobType="inference_service", **fields)
        if action in {"service_start", "service_restart"}:
            if _workflow_inference_structured_failure(response_data):
                return self._with_protocol(
                    "job_failed",
                    "inference",
                    result,
                    jobType="inference_service",
                    status="failed",
                    **fields,
                )
            if _inference_service_start_submitted(response_data):
                return self._with_protocol(
                    "job_started",
                    "inference", 
                    result,
                    jobType="inference_service", 
                    **fields)
            if _inference_start_result_needs_input(result):
                return self._with_protocol(
                    "need_input",
                    "inference",
                    result,
                    kind="inference_model_param_b",
                    title="补充模型参数规模",
                    requiredParams=["model_param_b"],
                    jobType="inference_service",
                    status="blocked",
                    **fields,
                )
            #deepseek
            if action == "service_restart":
                import re

                # 匹配“启动”或“重启” + 任意内容 + “提交”，且包含“成功”或“完成”
                pattern = r"(启动|重启).*(?:成功.*提交|已提交|正在.*启动)|提交.*(?:成功|已提交)"
                if re.search(pattern, result):
                    return self._with_protocol(
                        "job_started", 
                        "inference", 
                        result, 
                        jobType="inference_service", 
                        **fields)
            return self._with_protocol(
                "job_failed",
                "inference",
                result,
                jobType="inference_service",
                status="failed",
                **fields,
            )
        if action == "service_stop_preview":
            return self._with_protocol("inference_service_stop_preview", "inference", result, jobType="inference_service", **fields)
        if action == "service_stop":
            return self._with_protocol("job_stopped", "inference", result, jobType="inference_service", **fields)
        if action == "logs":
            return self._with_protocol("inference_logs", "inference", result, jobType="inference_service", **fields)
        if action == "benchmark_start":
            return self._with_protocol("job_started", "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_stop_preview":
            return self._with_protocol("inference_benchmark_stop_preview", "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_stop":
            protocol_type = "job_stopped" if isinstance(benchmark_stop, dict) and benchmark_stop.get("stopped") is True else "inference_benchmark_stop_result"
            return self._with_protocol(protocol_type, "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_status":
            return self._with_protocol("inference_benchmark_status", "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_result":
            return self._with_protocol("inference_benchmark_result", "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_progress":
            return self._with_protocol("inference_benchmark_progress", "inference", result, jobType="inference_benchmark", **fields)
        if action == "benchmark_list":
            return self._with_protocol("inference_benchmark_list", "inference", result, jobType="inference_benchmark", **fields)
        if action == "test_run":
            return self._with_protocol("inference_test_result", "inference", result, jobType="inference_test", **fields)
        if action == "test_stop_preview":
            return self._with_protocol("inference_test_stop_preview", "inference", result, jobType="inference_test", **fields)
        if action == "test_stop":
            protocol_type = "job_stopped" if isinstance(test_run_stop, dict) and test_run_stop.get("stopped") is True else "inference_test_stop_result"
            return self._with_protocol(protocol_type, "inference", result, jobType="inference_test", **fields)
        if action == "test_running_list":
            return self._with_protocol("inference_test_running_list", "inference", result, jobType="inference_test", **fields)
        if action == "test_list":
            return self._with_protocol("inference_test_list", "inference", result, jobType="inference_test", **fields)
        if action == "admin_cleanup_preview":
            return self._with_protocol("inference_admin_cleanup_preview", "inference", result, jobType="inference_admin_cleanup", **fields)
        if action == "admin_cleanup_apply":
            return self._with_protocol("inference_admin_cleanup_result", "inference", result, jobType="inference_admin_cleanup", **fields)
        return self._message_protocol("inference", result)

    def _agent_key_from_print_name(self, name: str) -> str:
        for key in ["dataprocessor", "datacollector", "trainer", "evaluator", "inference", "monitor", "analysis", "Orchestrator"]:
            if (name or "").startswith(key):
                return "orchestrator" if key == "Orchestrator" else key
        return name or "agent"

    def _protocol_for_printed_message(self, agent_name: str, text: str) -> Dict[str, Any]:
        clean_text = strip_think_for_context(text or "")
        if agent_name == "trainer" and (
            "启动前 GPU 快速校验失败" in clean_text
            or "assigned_gpu_preflight_failed" in clean_text
            or ("强制清理显存" in clean_text and "显存占用" in clean_text)
        ):
            display_text = self._gpu_preflight_failure_text(clean_text)
            return self._with_protocol(
                "job_failed",
                "trainer",
                display_text,
                jobType="train",
                errorReason="assigned_gpu_preflight_failed",
                errorRecoverable=True,
            )
        if any(tag in clean_text for tag in ["[参数请求]", "[类型请求]", "[等待用户选择]"]):
            request_kind = "param"
            if "[类型请求]" in clean_text:
                request_kind = "type"
            elif "[等待用户选择]" in clean_text:
                request_kind = "choice"
            friendly_text = self._friendly_request_text(agent_name, clean_text)
            required_params = []
            if request_kind == "param":
                required_params = self._extract_pending_param_names(
                    {
                        "response": clean_text,
                        "friendly_response": friendly_text,
                        "agent_name": agent_name,
                    },
                )
            return self._need_input_protocol(
                agent_name,
                request_kind,
                friendly_text,
                clean_text,
                required_params=required_params,
            )

        job_protocol = self._job_started_protocol(agent_name, "", "", clean_text, clean_text)
        if job_protocol:
            return job_protocol

        if any(keyword in clean_text for keyword in ["确认不执行高级筛选", "当前任务结束", "已完成"]):
            return self._message_protocol(agent_name, clean_text, "task_completed")

        return self._message_protocol(agent_name, clean_text)

    def _gpu_preflight_failure_text(self, text: str) -> str:
        clean_text = re.sub(r"\[[^\]]+\]\s*", "", strip_think_for_context(text or "")).strip()
        match = re.search(r"启动前 GPU 快速校验失败[：:][^\n。]*", clean_text)
        detail = match.group(0).strip() if match else clean_text
        if "分配的 GPU 已有显存占用" in detail:
            return (
                f"训练未启动：{detail}。"
                "系统不支持强制清理显存并继续执行，请先释放占用进程，"
                "或调整资源池 allowedGpuIndexes / GPU 申请数量后重试。"
            )
        return (
            f"训练未启动：{detail or 'GPU 启动前校验失败'}。"
            "请检查 GPU 可见性、显存占用和资源池配置后重试。"
        )

    def _register_protocol_pre_print_hook(self, agent: Agent) -> None:
        def protocol_pre_print_hook(_agent: Agent, hook_kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            msg = hook_kwargs.get("msg")
            if msg is None:
                kwargs = hook_kwargs.get("kwargs", {})
                msg = kwargs.get("msg") if isinstance(kwargs, dict) else None
            if not isinstance(msg, Msg) or msg.role != "assistant":
                return None
            metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
            if metadata.get("protocol"):
                return None
            text = msg.get_text_content() or ""
            agent_key = self._agent_key_from_print_name(msg.name)
            protocol = None
            if agent_key == "inference" and isinstance(self._last_inference_tool_payload, dict):
                tool_protocol = self._last_inference_tool_payload.get("protocol")
                if isinstance(tool_protocol, dict):
                    protocol = self._normalize_protocol(
                        {**tool_protocol, "message": text or tool_protocol.get("message") or ""},
                        text,
                        source=tool_protocol.get("source"),
                        confidence=tool_protocol.get("confidence"),
                    )
            protocol = protocol or self._protocol_for_printed_message(agent_key, text)
            protocol = self._mark_workflow_child_protocol(protocol) or protocol
            msg.metadata = {**metadata, "protocol": protocol}
            return hook_kwargs

        agent.register_instance_hook(
            "pre_print",
            "agent_result_protocol_pre_print_hook",
            protocol_pre_print_hook,
        )

    def _friendly_request_text(self, agent_name: str, response_text: str) -> str:
        text = (response_text or "").strip()
        decision_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        plain_text = re.sub(r"\[[^\]]+\]\s*", "", decision_text).strip()
        bullets = self._extract_bullet_items(decision_text)

        if "[类型请求]" in text:
            if "训练类型" in text:
                return self._wrap_user_waiting_message("训练类型", plain_text)
            if "评估类型" in text:
                return self._wrap_user_waiting_message("评估类型", plain_text)
            return self._wrap_user_waiting_message("类型补充", plain_text)

        if "[参数请求]" in text:
            if "需要参数：" in text:
                params_part = text.split("需要参数：", 1)[1]
                params_text = params_part.split("请用户提供", 1)[0].strip(" 。")
                params = [item.strip() for item in re.split(r"[、,，]", params_text) if item.strip()]
                return self._wrap_user_waiting_message("参数补充", (
                    "我还缺少一些必要信息，补充后就可以继续。\n\n"
                    f"请提供：{', '.join(params)}"
                    f"{self._build_reply_example(params)}"
                ))
            return self._wrap_user_waiting_message("参数补充", plain_text)

        if "[等待用户选择]" in text:
            choice_text = plain_text or "我需要你确认下一步要如何继续。"
            return self._wrap_user_waiting_message("用户选择", choice_text)

        if "需要以下必需参数，但未提供" in text:
            details = "\n".join(f"- {item}" for item in bullets) if bullets else plain_text
            example_keys = []
            for item in bullets[:3]:
                key = item.split("：", 1)[0].strip()
                if key:
                    example_keys.append(key)
            return self._wrap_user_waiting_message("必填参数", (
                "还差几个必填信息，我先不启动任务，避免直接报错。\n\n"
                f"{details}"
                f"{self._build_reply_example(example_keys, prefix='你可以这样回复')}"
            ))

        if "存在无效值或占位值" in text:
            if "model_fir" in text or "model_sec" in text:
                return self._wrap_user_waiting_message("参数补充", (
                    "开始双模型评估前，我还需要先确认两个模型的真实路径。\n\n"
                    "- `model_fir`：第一个参与评估的模型路径。\n"
                    "- `model_sec`：第二个参与评估的模型路径。\n\n"
                    "请把上面的占位内容替换成真实路径后再继续。\n\n"
                    "你可以直接回复，例如：`model_fir=<第一个模型路径>，model_sec=<第二个模型路径>`。"
                ))
            details = "\n".join(f"- {item}" for item in bullets) if bullets else plain_text
            return self._wrap_user_waiting_message("参数修正", (
                "当前提供的参数里，有些值看起来还是占位内容或无效内容，所以我先帮你拦下来了。\n\n"
                f"{details}\n\n"
                "请把这些字段改成真实可用的值后再继续。"
            ))

        if "路径在容器内不存在" in text:
            details = "\n".join(f"- {item}" for item in bullets) if bullets else plain_text
            return self._wrap_user_waiting_message("路径修正", (
                "当前任务暂时还不能启动，因为下面这些路径在容器里没有找到。\n\n"
                f"{details}\n\n"
                "请确认路径是否写对，或者把正确路径直接发给我。"
            ))

        if "不支持以下参数" in text:
            return self._wrap_user_waiting_message("参数修正", (
                "我收到了一些当前脚本不认识的参数，所以先没有继续执行。\n\n"
                f"{plain_text}\n\n"
                "你可以把想传的参数再发我一次，我帮你整理成脚本支持的格式。"
            ))

        return plain_text or text

    def _pending_parameter_prompt(self) -> Optional[str]:
        if not self.pending_parameters:
            return None

        agent_names = list(self.pending_parameters.keys())
        if len(agent_names) == 1:
            task_info = self.pending_parameters[agent_names[0]]
            prompt = task_info.get("message") or task_info.get("friendly_response") or task_info.get("response")
            return self._strip_user_waiting_wrapper(prompt or "")

        lines = ["目前有多个任务都在等你补充信息："]
        for agent_name in agent_names:
            task_info = self.pending_parameters[agent_name]
            prompt = task_info.get("message") or task_info.get("friendly_response") or task_info.get("response") or ""
            lines.append(f"- {agent_name}：{self._strip_user_waiting_wrapper(prompt)}")
        lines.append("")
        lines.append("你可以直接告诉我想先处理哪个任务，以及要补充的信息。")
        return "\n".join(lines)

    def _finalize_waiting_tool_response(
        self,
        _tool_call: Dict[str, Any],
        tool_response: ToolResponse,
    ) -> ToolResponse | None:
        text = self._response_to_text(tool_response)
        if not text.startswith("[等待用户回复|"):
            return None
        return ToolResponse(
            content=tool_response.content,
            metadata={
                "success": True,
                "response_msg": _msg(
                    name=self.orchestrator.name if self.orchestrator else "Orchestrator",
                    content=self._strip_user_waiting_wrapper(text),
                    role="assistant",
                    metadata={
                        "protocol": self._with_protocol(
                            "need_input",
                            "orchestrator",
                            self._strip_user_waiting_wrapper(text),
                            kind="input",
                        ),
                    },
                ),
            },
        )

    def _finalize_tool_protocol_response(
        self,
        _tool_call: Dict[str, Any],
        tool_response: ToolResponse,
    ) -> ToolResponse | None:
        metadata = tool_response.metadata or {}
        hint = metadata.get("protocol_hint")
        text = self._response_to_text(tool_response)
        protocol = self._protocol_from_hint(hint, text)
        if not protocol:
            return None

        protocol_type = str(protocol.get("type") or "")
        if protocol_type not in {"job_failed", "error", "need_input", "job_started", "job_preparing", "monitor_status"}:
            return None

        display_text = str(protocol.get("message") or text or "").strip()
        if protocol_type == "monitor_status":
            display_text, protocol = self._finalize_monitor_status_protocol(protocol, display_text)
        if (
            protocol.get("agent") == "trainer"
            and protocol.get("errorReason") == "assigned_gpu_preflight_failed"
        ):
            display_text = self._gpu_preflight_failure_text(display_text or text)
            protocol = self._with_protocol(
                "job_failed",
                "trainer",
                display_text,
                source=str(protocol.get("source") or "tool_hint"),
                confidence=protocol.get("confidence"),
                jobType="train",
                errorReason="assigned_gpu_preflight_failed",
                errorRecoverable=True,
                container=protocol.get("container"),
                assignedGpus=protocol.get("assignedGpus"),
            )

        return ToolResponse(content=[TextBlock(type="text", text=display_text or text)],
            metadata={
                **metadata,
                "success": metadata.get("success", protocol_type in {"job_started", "job_preparing", "monitor_status"}),
                "protocol": protocol,
                "response_msg": self._agent_msg(
                    str(protocol.get("agent") or "trainer"),
                    display_text or text,
                    protocol,
                ),
            },
        )

    def _friendly_stop_result_text(self, result: str) -> str:
        text = (result or "").strip()
        if not text:
            return "未识别到可终止的任务类型。"

        header_lines = []
        for line in text.splitlines():
            if line.startswith(("已发送", "容器:", "PID:", "目标:", "脚本:")):
                header_lines.append(line)
        header_text = "\n".join(header_lines[:4]).strip()
        main_text = text.split("\n\nworker容器:", 1)[0]
        main_not_found = "未找到匹配进程" in main_text or "未找到可终止进程" in main_text

        if "复查未发现残留进程" in main_text:
            status = "复查未发现残留进程，任务已完成清理。"
            return "\n".join([*header_lines[:4], status]).strip()

        remaining_timeout_match = re.search(r"仍发现残留 PID:\s*([^\n]+)", main_text)
        if remaining_timeout_match:
            status = f"已发送结束指令但尚未确认完成；复查仍发现残留 PID: {remaining_timeout_match.group(1).strip()}。请稍后再次查询任务状态。"
            return "\n".join([*header_lines[:4], status]).strip()

        if "复查状态失败" in main_text:
            status = "已发送结束指令但尚未确认完成；复查状态失败，请稍后查询任务状态或检查容器日志。"
            return "\n".join([*header_lines[:4], status]).strip()
        if (
            "TERM " in main_text
            or "复查未发现训练残留进程" in main_text
            or ("返回码: 0" in main_text and not main_not_found)
        ):
            status = "复查未发现残留进程，任务已完成清理。"
            return "\n".join([*header_lines[:4], status]).strip()

        if main_not_found:
            status = "未找到匹配进程，未执行终止。"
            return "\n".join([*header_lines[:4], status]).strip()

        if "未找到匹配进程" in text or "未找到可终止进程" in text:
            status = "未找到匹配进程，未执行终止。"
            return "\n".join([*header_lines[:4], status]).strip()

        if "复查未发现训练残留进程" in text or "返回码: 0" in text:
            status = "复查未发现残留进程，任务已完成清理。"
            return "\n".join([*header_lines[:4], status]).strip()

        remaining_match = re.search(r"仍有残留PID:\s*([^\n]+)", text)
        if remaining_match:
            status = f"已发送结束指令，但复查仍发现残留 PID: {remaining_match.group(1).strip()}。"
            return "\n".join([*header_lines[:4], status]).strip()

        if "结束指令执行失败" in text or "返回码: 1" in text:
            status = "结束指令执行失败，请检查任务状态或容器日志。"
            return "\n".join([header_text, status]).strip() if header_text else status

        return header_text or text

    def _stop_status_protocol_fields(self, result: str) -> Dict[str, Any]:
        text = result or ""
        fields: Dict[str, Any] = {}
        if "复查未发现残留进程" in text or "复查未发现训练残留进程" in text or "系统已自动重试并完成清理" in text:
            fields["stopStatus"] = "confirmed"
        elif "复查仍发现残留进程" in text or "仍发现残留 PID" in text or "仍有残留PID" in text:
            fields["stopStatus"] = "pending"
        elif "复查状态失败" in text:
            fields["stopStatus"] = "unknown"

        remaining_pids = []
        for match in re.findall(r"(?:残留\s*PID|仍有残留PID)[:：]?\s*([^。\n]+)", text):
            for pid_value in re.findall(r"\b\d+\b", match):
                if pid_value not in remaining_pids:
                    remaining_pids.append(pid_value)
        if remaining_pids:
            fields["remainingPids"] = remaining_pids
        return fields

    def _diagnose_stop_timeout(
        self,
        task_type: Optional[str],
        container: Optional[str],
        pid: Optional[str] = None,
        patterns: Optional[List[str]] = None,
    ) -> str:
        if not container:
            return "结束指令已发送，但复查状态失败；请稍后查询任务状态或检查容器日志。"

        def collect_remaining() -> List[str]:
            remaining: List[str] = []
            if pid:
                remaining.extend(self._check_docker_pid_processes(container, str(pid)))
            elif patterns:
                remaining.extend(self._check_docker_pattern_processes(container, patterns))
            unique_remaining: List[str] = []
            for item in remaining:
                if item not in unique_remaining:
                    unique_remaining.append(item)
            return unique_remaining

        def remaining_pid_summary(remaining: List[str]) -> str:
            remaining_pids = []
            for item in remaining:
                match = re.match(r"^\s*(\d+)\b", item)
                if match and match.group(1) not in remaining_pids:
                    remaining_pids.append(match.group(1))
            return ", ".join(remaining_pids) if remaining_pids else "; ".join(remaining[:3])

        try:
            unique_remaining = collect_remaining()
            if not unique_remaining:
                return "结束指令已发送；虽然等待命令返回超时，但复查未发现残留进程，任务已完成清理。"

            for delay_seconds in (2, 4, 6):
                time.sleep(delay_seconds)
                unique_remaining = collect_remaining()
                if not unique_remaining:
                    return "结束指令已发送；虽然等待命令返回超时，但复查未发现残留进程，任务已完成清理。"
        except (subprocess.SubprocessError, OSError):
            return "结束指令已发送，但复查状态失败；请稍后查询任务状态或检查容器日志。"

        if not unique_remaining:
            return "结束指令已发送；虽然等待命令返回超时，但复查未发现残留进程，任务已完成清理。"

        pid_summary = remaining_pid_summary(unique_remaining)
        return (
            "结束指令已发送，但复查仍发现残留进程；"
            "系统可能仍在清理，请稍后再次查询任务状态。"
            f"\n仍发现残留 PID: {pid_summary}"
        )

    def _last_trainer_launch_target(self) -> Dict[str, Optional[str]]:
        """Extract the most recent training PID/container from trainer's final message."""
        target = {"pid": None, "container": None}
        record = self.last_completed_task or {}
        if record.get("agent") != "trainer":
            return target

        raw_text = "\n".join(
            str(record.get(key) or "")
            for key in ("response", "task", "params")
        )
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)

        pid_match = re.search(
            r"(?:进程\s*ID\s*(?:\(\s*PID\s*\))?|进程ID|PID|pid|进程号)[^\d\n]{0,40}`?(\d+)`?",
            text,
            re.IGNORECASE,
        )
        if pid_match:
            target["pid"] = pid_match.group(1)

        container_match = re.search(
            r"(?:容器名称|容器名|容器|container_name|container)[`'\"*\s]*[:：=]\s*[`'\"]?([A-Za-z0-9_.-]+)",
            text,
            re.IGNORECASE,
        )
        if container_match:
            target["container"] = container_match.group(1).strip("`'\"")

        return target

    def _infer_recent_stop_task_type(self) -> Optional[str]:
        record = self.last_completed_task or {}
        agent = record.get("agent")
        if agent == "trainer":
            return "train"
        if agent == "dataprocessor":
            return "data"
        if agent == "evaluator":
            return "evaluate"
        if agent == "inference":
            return "inference"
        return None
    
    async def initialize_agents(self):
        """初始化所有智能体（每个用户独立）"""
        # 1. 初始化总览agent（编排器）
        orchestrator_toolkit = await self._create_orchestrator_tools()
        self.orchestrator = Agent(
            name=f"Orchestrator_{self.user_id}",
            system_prompt=config.agents.orchestrator.sys_prompt,
            model=self.shared_model,
            toolkit=orchestrator_toolkit,
        )
        self._register_protocol_pre_print_hook(self.orchestrator)
        
        # 2. 初始化各个功能agent
        await self._initialize_function_agents()
        
        logger.info(f"Agents initialized for user {self.user_id}")
    
    async def _create_orchestrator_tools(self) -> Toolkit:
        """为编排器创建工具，用于调用其他agent"""
        toolkit = Toolkit(schema_log_label="orchestrator")
        postprocess = self._finalize_waiting_tool_response
        toolkit.register_tool_function(self._stop_task, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_datacollector, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_dataprocessor, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_trainer, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_evaluator, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_inference, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_monitor, postprocess_func=postprocess)
        toolkit.register_tool_function(self._call_analysis, postprocess_func=postprocess)
        toolkit.register_tool_function(self._workflow_control)
        await toolkit.finalize()
        return toolkit
    
    async def _initialize_function_agents(self):
        """初始化所有功能agent（每个用户独立）"""
        # datacollector
        data_collect_toolkit = Toolkit(schema_log_label="datacollector")
        data_collect_toolkit.register_tool_function(run_group_data_collect)
        await data_collect_toolkit.finalize()
        self.agents["datacollector"] = Agent(
            name=f"datacollector_{self.user_id}",
            system_prompt=config.agents.datacollector.sys_prompt,
            model=self.shared_model,
            toolkit=data_collect_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["datacollector"])
        
        # dataprocessor
        data_toolkit = Toolkit(schema_log_label="dataprocessor")
        data_toolkit.register_tool_function(
            run_group_data,
            postprocess_func=self._finalize_tool_protocol_response,
        )
        await data_toolkit.finalize()
        self.agents["dataprocessor"] = Agent(
            name=f"dataprocessor_{self.user_id}",
            system_prompt=config.agents.dataprocessor.sys_prompt,
            model=self.shared_model,
            toolkit=data_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["dataprocessor"])
        
        # trainer
        train_toolkit = Toolkit(schema_log_label="trainer")
        train_toolkit.register_tool_function(
            run_group_train,
            postprocess_func=self._finalize_tool_protocol_response,
        )
        await train_toolkit.finalize()
        self.agents["trainer"] = Agent(
            name=f"trainer_{self.user_id}",
            system_prompt=config.agents.trainer.sys_prompt,
            model=self.shared_model,
            toolkit=train_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["trainer"])
        
        # evaluator
        evaluate_toolkit = Toolkit(schema_log_label="evaluator")
        evaluate_toolkit.register_tool_function(
            run_group_assessment,
            postprocess_func=self._finalize_tool_protocol_response,
        )
        await evaluate_toolkit.finalize()
        self.agents["evaluator"] = Agent(
            name=f"evaluator_{self.user_id}",
            system_prompt=config.agents.evaluator.sys_prompt,
            model=self.shared_model,
            toolkit=evaluate_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["evaluator"])
        
        # inference
        inference_toolkit = Toolkit(schema_log_label="inference")
        inference_toolkit.register_tool_function(self._run_inference_agent_command)
        await inference_toolkit.finalize()
        self.agents["inference"] = Agent(
            name=f"inference_{self.user_id}",
            system_prompt=config.agents.inference.sys_prompt,
            model=self.shared_model,
            toolkit=inference_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["inference"])
        
        # monitor
        monitor_toolkit = Toolkit(schema_log_label="monitor")
        monitor_toolkit.register_tool_function(
            run_script_by_name_monitor1,
            postprocess_func=self._finalize_tool_protocol_response,
        )
        monitor_toolkit.register_tool_function(
            run_group_assessment_monitor,
            postprocess_func=self._finalize_tool_protocol_response,
        )
        await monitor_toolkit.finalize()
        self.agents["monitor"] = Agent(
            name=f"monitor_{self.user_id}",
            system_prompt=config.agents.monitor.sys_prompt,
            model=self.shared_model,
            toolkit=monitor_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["monitor"])

        #analysis
        self.agents["analysis"] = Agent(
            name=f"analysis_{self.user_id}",
            system_prompt=config.agents.analysis.sys_prompt,
            model=self.shared_model,
            #toolkit=monitor_toolkit,
        )
        self._register_protocol_pre_print_hook(self.agents["analysis"])
    # ========== 编排器工具函数 ==========
    async def _direct_agent_response(
        self,
        agent_name: str,
        content: str,
        task_description: str = "",
        additional_params: str = "",
        protocol: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a deterministic response through the target agent output path."""
        agent = self.agents[agent_name]
        protocol = protocol or self._message_protocol(agent_name, content)
        msg = self._agent_msg(agent.name, content, protocol)
        if agent_name == "dataprocessor" and self._is_acknowledgement_only(content):
            await asyncio.sleep(4.5)
        await agent.print(msg, True)
        await agent.memory.add(msg)

        self.last_completed_task = {
            "agent": agent_name,
            "task": task_description,
            "params": additional_params,
            "response": strip_think_for_context(content),
            "timestamp": asyncio.get_event_loop().time(),
        }
        self.last_response_protocol = protocol
        self.current_task_state = {"status": "idle"}
        return self._protocol_json_response(content, protocol)

    def _format_monitor_status_text(self, protocol: Dict[str, Any], fallback: str = "") -> str:
        status_map = {
            "preparing": "准备中",
            "starting": "启动中",
            "running": "运行中",
            "finished": "已完成",
            "completed": "已完成",
            "failed": "失败",
            "stopped": "已停止",
            "unknown": "未知",
        }
        train_type_map = {
            "lora": "LoRA SFT",
            "lora_sft": "LoRA SFT",
            "full": "全参 SFT",
            "full_sft": "全参 SFT",
            "enhanced": "增强训练",
            "grpo": "GRPO",
            "multinode_lora_sft": "双机 LoRA SFT",
            "multinode_enhanced": "双机增强训练",
        }
        status_raw = str(protocol.get("status") or "unknown").strip().lower()
        status_text = status_map.get(status_raw, str(protocol.get("status") or "未知"))
        container = str(protocol.get("container") or "").strip()
        pid = str(protocol.get("pid") or "").strip()
        job_type = str(protocol.get("jobType") or "").strip().lower()
        agent_name = str(protocol.get("agent") or "").strip().lower()
        is_assessment = job_type in {"assessment", "evaluate"} or agent_name == "assessment_monitor"
        train_type_text_raw = str(protocol.get("trainTypeText") or "").strip()
        train_type_raw = str(protocol.get("trainTypeEn") or protocol.get("trainType") or "").strip()
        train_type = train_type_text_raw or train_type_map.get(train_type_raw.lower(), train_type_raw)
        assessment_type_raw = str(
            protocol.get("assessmentTypeText")
            or protocol.get("evalTypeText")
            or protocol.get("assessmentType")
            or protocol.get("evalType")
            or ""
        ).strip()

        subject_parts = []
        if container:
            subject_parts.append(f"容器：{container}")
        if pid:
            subject_parts.append(f"PID：{pid}")
        subject = f"（{'，'.join(subject_parts)}）" if subject_parts else ""
        task_label = "评估" if is_assessment else "训练"
        lines = [f"当前{task_label}任务{subject}状态为：**{status_text}**。"]

        def _as_non_empty_text(value: Any) -> str:
            return str(value or "").strip()

        def _first_mapping(value: Any) -> Dict[str, Any]:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        return item
            return value if isinstance(value, dict) else {}

        def _mapping_items(value: Any) -> list:
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
            return []

        def _format_percent(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            text = str(value).strip()
            if not text:
                return ""
            return text if text.endswith("%") else f"{text}%"

        def _truncate_log_line(value: Any, limit: int = 180) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return f"{text[:limit]}..."

        def _format_score_value(value: Any) -> str:
            if isinstance(value, float):
                return str(int(value)) if value.is_integer() else str(value)
            return str(value)

        def _assessment_score_lines(scores: Any) -> list:
            if not isinstance(scores, dict):
                return []
            score_lines = []
            for name, result in scores.items():
                stage_name = _as_non_empty_text(name)
                if not stage_name:
                    continue
                if isinstance(result, dict):
                    if result.get("average") is not None:
                        score_lines.append(f"- **{stage_name}**：平均分 {_format_score_value(result.get('average'))}")
                        continue
                    compact_parts = []
                    for key in ("accuracy", "mean_fuzziness"):
                        if result.get(key) is not None:
                            compact_parts.append(f"{key} {_format_score_value(result.get(key))}")
                    if compact_parts:
                        score_lines.append(f"- **{stage_name}**：{'，'.join(compact_parts)}")
                elif result is not None:
                    score_lines.append(f"- **{stage_name}**：{_format_score_value(result)}")
            return score_lines

        def _assessment_running_detail_lines(details: Any) -> list:
            running_lines = []
            for detail in _mapping_items(details):
                if detail.get("status") not in (None, "", "running"):
                    continue
                name = _as_non_empty_text(detail.get("name")) or "当前阶段"
                summary = detail.get("progress_summary") if isinstance(detail.get("progress_summary"), dict) else {}
                log_meta = detail.get("progress_log_meta") if isinstance(detail.get("progress_log_meta"), dict) else {}
                stage_parts = []
                local_parts = []
                step_name = _as_non_empty_text(summary.get("step_name"))
                step_current = summary.get("step_current")
                step_total = summary.get("step_total")
                local_percent = _format_percent(summary.get("local_progress_percent"))
                if step_name:
                    local_parts.append(step_name)
                if step_current is not None and step_total is not None:
                    local_parts.append(f"{step_current}/{step_total}")
                if local_percent:
                    local_parts.append(f"局部进度 {local_percent}")
                if local_parts:
                    stage_parts.append("，".join(local_parts))
                latest_line = _truncate_log_line(summary.get("latest_line"))
                if latest_line:
                    stage_parts.append(f"最新日志行：{latest_line}")
                updated_at = _as_non_empty_text(log_meta.get("updated_at"))
                if updated_at:
                    stage_parts.append(f"日志更新时间：{updated_at}")
                progress_log = _as_non_empty_text(detail.get("progress_log"))
                if progress_log:
                    stage_parts.append(f"日志路径：{progress_log}")
                if not stage_parts:
                    continue
                running_lines.append(f"- **{name}**：{'；'.join(stage_parts)}")
                tail_lines = [
                    _truncate_log_line(item)
                    for item in (summary.get("tail_lines") or [])[-3:]
                    if _truncate_log_line(item)
                ]
                if tail_lines:
                    running_lines.append(f"  局部日志尾部：{' / '.join(tail_lines)}")
            return running_lines

        def _assessment_report_lines(report_path: Any, report_text: Any) -> list:
            report_lines = []
            path = _as_non_empty_text(report_path)
            if path:
                report_lines.append(f"- **报告路径**：{path}")
            content = str(report_text or "").strip()
            if content:
                report_lines.append("- **完整报告内容**：")
                report_lines.append("```text")
                report_lines.extend(content.splitlines())
                report_lines.append("```")
            return report_lines

        running_detail = _first_mapping(protocol.get("currentStageDetails"))
        progress_summary = running_detail.get("progress_summary") if isinstance(running_detail.get("progress_summary"), dict) else {}
        is_assessment_finished = is_assessment and status_raw in {"finished", "completed"}
        assessment_score_lines = _assessment_score_lines(protocol.get("scores")) if is_assessment and not is_assessment_finished else []
        assessment_running_detail_lines = (
            _assessment_running_detail_lines(protocol.get("currentStageDetails"))
            if is_assessment
            else []
        )
        assessment_report_lines = (
            _assessment_report_lines(protocol.get("reportPath"), protocol.get("reportText"))
            if is_assessment_finished
            else []
        )

        detail_lines = []
        if is_assessment and assessment_type_raw:
            detail_lines.append(f"- **评估类型**：{assessment_type_raw}")
        elif train_type:
            detail_lines.append(f"- **训练类型**：{train_type}")
        if is_assessment:
            current_stage = _as_non_empty_text(protocol.get("currentStage") or protocol.get("currentLogicalStage") or running_detail.get("name"))
            if current_stage:
                detail_lines.append(f"- **当前阶段**：{current_stage}")
            completed_stages = protocol.get("completedStages")
            total_stages = protocol.get("totalStages")
            next_stage = _as_non_empty_text(protocol.get("nextStage"))
            stage_progress_parts = []
            if completed_stages is not None and total_stages is not None:
                stage_progress_parts.append(f"已完成 {completed_stages}/{total_stages}")
            if next_stage:
                stage_progress_parts.append(f"下一阶段 {next_stage}")
            if stage_progress_parts:
                detail_lines.append(f"- **阶段进度**：{'，'.join(stage_progress_parts)}")

        latest_loss = protocol.get("latestLoss")
        if latest_loss is not None:
            detail_lines.append(f"- **Loss**：{latest_loss}")
        latest_epoch = protocol.get("latestEpoch")
        if latest_epoch is not None:
            detail_lines.append(f"- **Epoch**：{latest_epoch}")
        latest_step = protocol.get("latestStep")
        if latest_step is not None:
            detail_lines.append(f"- **Step**：{latest_step}")
        progress_percent = protocol.get("progressPercent")
        if progress_percent is not None:
            progress_text = str(progress_percent).strip()
            if progress_text and not progress_text.endswith("%"):
                progress_text = f"{progress_text}%"
            detail_lines.append(f"- **进度**：{progress_text}")
        sub_stage_text = protocol.get("subStageText")
        if sub_stage_text is not None and str(sub_stage_text).strip():
            detail_lines.append(f"- **阶段**：{sub_stage_text}")
        export_dir = protocol.get("exportDir")
        if export_dir is not None and str(export_dir).strip():
            detail_lines.append(f"- **导出目录**：{export_dir}")
        merge_status = protocol.get("mergeStatus")
        if merge_status is not None and str(merge_status).strip():
            detail_lines.append(f"- **Merge 状态**：{merge_status}")
        wandb_text = self._monitor_wandb_display_text(protocol)
        if wandb_text:
            detail_lines.append(f"- **WandB**：{wandb_text}")

        if detail_lines:
            lines.append("\n**详细信息：**")
            lines.extend(detail_lines)

        if assessment_score_lines:
            lines.append("\n**已有评分：**")
            lines.extend(assessment_score_lines)

        if assessment_report_lines:
            lines.append("\n**最终报告：**")
            lines.extend(assessment_report_lines)

        if assessment_running_detail_lines:
            lines.append("\n**运行中阶段详情：**")
            lines.extend(assessment_running_detail_lines)

        if status_raw in {"preparing", "starting"}:
            if is_assessment:
                lines.append("\n评估任务正在初始化中，阶段结果可能仍在生成，请稍后刷新状态。")
            else:
                lines.append("\n训练正在初始化中，指标可能仍在生成，请稍后刷新状态。")
        elif status_raw == "running" and latest_loss is None and latest_step is None:
            if is_assessment:
                step_name = _as_non_empty_text(progress_summary.get("step_name"))
                step_current = progress_summary.get("step_current")
                step_total = progress_summary.get("step_total")
                local_percent = _format_percent(progress_summary.get("local_progress_percent"))
                if step_name and step_current is not None and step_total is not None:
                    percent_suffix = f"（局部进度 {local_percent}）" if local_percent else ""
                    lines.append(f"\n当前阶段正在执行 {step_name}：{step_current}/{step_total}{percent_suffix}。")
                elif assessment_score_lines or assessment_running_detail_lines:
                    lines.append("\n评估任务仍在运行，最终报告尚未生成；以上为当前已产出的阶段结果和运行日志摘要。")
                else:
                    lines.append("\n评估任务正在运行中，但阶段结果/指标数据暂未全部就绪，请稍后刷新状态。")
            else:
                lines.append("\n训练进程正在运行中，但指标数据暂未就绪，请稍后刷新状态。")

        return "\n".join(lines).strip() or str(fallback or "").strip()

    def _finalize_monitor_status_protocol(
        self,
        protocol: Dict[str, Any],
        fallback: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        display_text = self._format_monitor_status_text(protocol, fallback)
        finalized = {**protocol, "message": display_text}
        return display_text, finalized

    def _monitor_wandb_display_text(self, protocol: Dict[str, Any]) -> Optional[str]:
        wandb_url = protocol.get("wandbUrl")
        if wandb_url is not None and str(wandb_url).strip():
            return str(wandb_url).strip()
        wandb_mode = str(protocol.get("wandbMode") or "").strip().lower()
        if wandb_mode == "offline":
            return "离线模式，本次没有在线链接"
        if protocol.get("wandbUrlPending") is True:
            return "生成中，稍后刷新状态"
        return None

    def _append_monitor_wandb_to_text(self, text: str, protocol: Optional[Dict[str, Any]]) -> str:
        content = str(text or "").strip()
        if not isinstance(protocol, dict) or protocol.get("type") != "monitor_status":
            return content
        wandb_text = self._monitor_wandb_display_text(protocol)
        if not wandb_text:
            return content
        if "wandb.ai" in content or "WandB" in content or "离线模式，本次没有在线链接" in content:
            return content
        wandb_line = f"- **WandB**：{wandb_text}"
        return f"{content}\n{wandb_line}" if content else wandb_line

    def _direct_monitor_tool_response(self, task_description: str, tool_response: ToolResponse) -> ToolResponse:
        finalized = self._finalize_tool_protocol_response(
            {"name": "monitor", "arguments": {"script_query": task_description}},
            tool_response,
        ) or tool_response
        display_text = self._response_to_text(finalized)
        protocol = self._normalize_protocol(
            finalized.metadata.get("protocol") if isinstance(finalized.metadata, dict) else None,
            display_text,
        )
        if protocol and protocol.get("type") == "monitor_status":
            display_text, protocol = self._finalize_monitor_status_protocol(
                protocol,
                str(protocol.get("message") or display_text or "").strip(),
            )
            protocol = self._normalize_protocol(
                {**protocol, "message": display_text},
                display_text,
                source=protocol.get("source"),
                confidence=protocol.get("confidence"),
            ) or protocol
            finalized = ToolResponse(
                content=[TextBlock(type="text", text=display_text)],
                metadata={
                    **(finalized.metadata or {}),
                    "success": True,
                    "protocol": protocol,
                    "protocol_hint": protocol,
                    "response_msg": self._agent_msg("monitor", display_text, protocol),
                },
            )
        if "monitor" in self.pending_parameters:
            del self.pending_parameters["monitor"]
        self.last_completed_task = {
            "agent": "monitor",
            "task": task_description,
            "params": "",
            "response": display_text,
            "timestamp": asyncio.get_event_loop().time(),
        }
        self.last_response_protocol = protocol
        self.current_task_state = {"status": "idle"}
        return ToolResponse(
            content=[TextBlock(type="text", text=f"monitor 执行结果：\n{display_text}")],
            metadata={
                **(finalized.metadata or {}),
                "success": (finalized.metadata or {}).get("success", True),
                "protocol": protocol,
                "protocol_hint": protocol,
                "response_msg": self._agent_msg("monitor", display_text, protocol),
            },
        )

    def _is_direct_training_monitor_request(self, task_description: str) -> bool:
        text = str(task_description or "").lower()
        has_monitor_intent = any(keyword in text for keyword in ["监控", "状态", "进度", "monitor", "status"])
        has_training_target = any(keyword in text for keyword in ["训练", "trainer", "train", "training", "pid", "容器", "container", "grpo"])
        return has_monitor_intent and has_training_target

    def _is_direct_assessment_monitor_request(self, task_description: str) -> bool:
        text = str(task_description or "").lower()
        has_monitor_intent = any(keyword in text for keyword in ["监控", "状态", "进度", "monitor", "status"])
        has_assessment_target = any(keyword in text for keyword in ["评估", "评测", "evaluate", "evaluation", "assessment", "eval"])
        return has_monitor_intent and has_assessment_target

    def _agent_request_requires_tool_call(self, agent_name: str, task_description: str) -> bool:
        """Return true when a missing tool call can be verified safely."""
        if agent_name != "inference":
            return False
        text = (task_description or self.current_user_message or "").lower()
        keywords = ["推理", "服务", "节点", "gpu", "benchmark", "基准", "功能测试", "配置", "端口", "日志", "启动", "停止", "重启"]
        return any(keyword in text for keyword in keywords) and not isinstance(self._last_inference_tool_payload, dict)

    def _missing_tool_call_response(self, agent_name: str, task_description: str, agent: Any) -> ToolResponse:
        message = (
            f"{agent_name} 没有产生 AgentScope tool call，因此我没有把这次请求当成已执行。"
            "这通常表示当前模型后端没有按 OpenAI tools/tool_calls 协议返回工具调用，"
            "或工具 schema 没有被模型采纳。请检查 MODEL_BASE_URL 对 tools 的兼容性，"
            "或重试一次；系统不会编造工具执行结果。"
        )
        protocol = self._with_protocol(
            "error",
            agent_name,
            message,
            action="tool_call_missing",
            status="failed",
            task=task_description,
        )
        self.last_response_protocol = protocol
        self.current_task_state = {"status": "idle"}
        return ToolResponse(content=[TextBlock(type="text", text=message)],
            metadata={
                "success": False,
                "protocol": protocol,
                "response_msg": self._agent_msg(agent.name, message, protocol),
            },
        )
    async def _call_agent_with_params(
        self,
        agent_name: str,
        task_description: str,
        additional_params: str = "",
        *,
        pending_original_task: Optional[str] = None,
    ) -> ToolResponse:
        """调用agent并传递参数"""
        logger.info(f"[Orchestrator] 调用 {agent_name}: {task_description[:50]}...")
        
        full_task = task_description
        if additional_params:
            full_task = f"{task_description}\n\n{additional_params}"
        
        agent = self.agents[agent_name]
        msg = _msg(name="Orchestrator", content=full_task, role="user")
        
        # 记录当前任务状态
        self.current_task_state = {
            "agent": agent_name,
            "task": task_description,
            "params_provided": additional_params,
            "status": "executing"
        }
        
        if agent_name == "inference":
            self._last_inference_tool_payload = None
        response = await agent(msg)
        # task_description may contain orchestration-only hints appended for the
        # function agent.  Never persist those hints as user input: doing so makes
        # the next parameter reply parse examples such as "模型位置/数据集位置"
        # and training-mode names from the hints themselves.
        original_task_for_pending = pending_original_task or task_description
        response_text = response.get_text_content() or ""
        logger.info(f"用户{self.user_id}; agent({agent_name})返回{response_text[:200]}...")
        context_response_text = strip_think_for_context(response_text)
        response_metadata = getattr(response, "metadata", None)
        response_protocol = self._normalize_protocol(
            response_metadata.get("protocol") if isinstance(response_metadata, dict) else None,
            context_response_text,
        )
        if agent_name == "monitor" and response_protocol and response_protocol.get("type") == "monitor_status":
            display_text, response_protocol = self._finalize_monitor_status_protocol(
                response_protocol,
                str(response_protocol.get("message") or context_response_text or "").strip(),
            )
            response_protocol = self._normalize_protocol(
                {**response_protocol, "message": display_text},
                display_text,
                source=response_protocol.get("source"),
                confidence=response_protocol.get("confidence"),
            ) or response_protocol
            if agent_name in self.pending_parameters:
                del self.pending_parameters[agent_name]
            self.last_completed_task = {
                "agent": agent_name,
                "task": task_description,
                "params": additional_params,
                "response": display_text,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.last_response_protocol = response_protocol
            self.current_task_state = {"status": "idle"}
            return ToolResponse(content=[TextBlock(type="text", text=f"{agent_name} 执行结果：\n{display_text}")],
                metadata={
                    "success": True,
                    "protocol": response_protocol,
                    "response_msg": self._agent_msg(agent.name, display_text, response_protocol),
                },
            )
        if response_protocol and response_protocol.get("type") in {"job_failed", "error", "job_started", "job_preparing", "need_input"}:
            display_text = str(response_protocol.get("message") or context_response_text or "").strip()
            protocol_type = str(response_protocol.get("type") or "")
            if protocol_type == "need_input":
                response_protocol = self._enrich_dataset_name_need_input_protocol(
                    response_protocol,
                    "\n".join([original_task_for_pending or "", task_description or ""]),
                    display_text or context_response_text,
                    container=self._training_container_for_request(task_description) if agent_name == "trainer" else None,
                )
                request_kind = "param"
                required_params = response_protocol.get("requiredParams") or []
                self.pending_parameters[agent_name] = self._pending_state(
                    agent_name,
                    original_task_for_pending,
                    request_kind,
                    context_response_text,
                    display_text,
                    response_protocol,
                    required_params=required_params,
                    needs_choice=False,
                )
                self.last_response_protocol = response_protocol
                return ToolResponse(content=[TextBlock(type="text", text=display_text)],
                    metadata={
                        "success": True,
                        "protocol": response_protocol,
                        "response_msg": self._agent_msg(agent.name, display_text, response_protocol),
                    },
                )

            if protocol_type == "job_preparing":
                if agent_name in self.pending_parameters:
                    del self.pending_parameters[agent_name]
                self.last_response_protocol = response_protocol
                self.current_task_state = {
                    "status": "preparing",
                    "agent": agent_name,
                    "pid": response_protocol.get("pid"),
                    "container": response_protocol.get("container"),
                }
                return ToolResponse(content=[TextBlock(type="text", text=f"{agent_name} 执行结果：\n{display_text}")],
                    metadata={
                        "success": True,
                        "protocol": response_protocol,
                        "response_msg": self._agent_msg(agent.name, display_text, response_protocol),
                    },
                )

            if agent_name in self.pending_parameters:
                del self.pending_parameters[agent_name]
            self.last_completed_task = {
                "agent": agent_name,
                "task": task_description,
                "params": additional_params,
                "response": display_text,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.last_response_protocol = response_protocol
            self.current_task_state = {"status": "idle"}
            return ToolResponse(content=[TextBlock(type="text", text=f"{agent_name} 执行结果：\n{display_text}")],
                metadata={
                    "success": protocol_type in {"job_started", "job_preparing"},
                    "protocol": response_protocol,
                    "response_msg": self._agent_msg(agent.name, display_text, response_protocol),
                },
            )

        if agent_name == "trainer" and (
            "启动前 GPU 快速校验失败" in response_text
            or "assigned_gpu_preflight_failed" in response_text
            or ("强制清理显存" in response_text and "显存占用" in response_text)
        ):
            display_text = self._gpu_preflight_failure_text(response_text)
            protocol = self._with_protocol(
                "job_failed",
                "trainer",
                display_text,
                jobType="train",
                errorReason="assigned_gpu_preflight_failed",
                errorRecoverable=True,
            )
            self.last_completed_task = {
                "agent": agent_name,
                "task": task_description,
                "params": additional_params,
                "response": display_text,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.last_response_protocol = protocol
            self.current_task_state = {"status": "idle"}
            return ToolResponse(content=[TextBlock(type="text", text=f"{agent_name} 执行结果：\n{display_text}")],
                metadata={
                    "success": False,
                    "protocol": protocol,
                    "response_msg": self._agent_msg(agent.name, display_text, protocol),
                },
            )
        
        # 检查响应类型
        if any(tag in response_text for tag in ["[参数请求]", "[类型请求]", "[等待用户选择]"]):
            if self._is_terminal_waiting_response(agent_name, response_text):
                display_text = re.sub(r"\[[^\]]+\]\s*", "", context_response_text).strip()
                protocol = self._message_protocol(agent_name, display_text, "task_completed")
                self.last_completed_task = {
                    "agent": agent_name,
                    "task": task_description,
                    "params": additional_params,
                    "response": display_text,
                    "timestamp": asyncio.get_event_loop().time()
                }
                self.last_response_protocol = protocol
                self.current_task_state = {"status": "idle"}
                return ToolResponse(content=[TextBlock(type="text", text=display_text)],
                    metadata={
                        "success": True,
                        "protocol": protocol,
                        "response_msg": self._agent_msg(agent.name, display_text, protocol),
                    },
                )

            friendly_text = self._friendly_request_text(agent_name, response_text)
            request_kind = "param"
            if "[类型请求]" in response_text:
                request_kind = "type"
            elif "[等待用户选择]" in response_text:
                request_kind = "choice"
            required_params = []
            missing_params = []
            if request_kind == "param":
                param_text = "\n".join([context_response_text or "", friendly_text or ""])
                missing_params = self._extract_explicit_required_params(param_text)
                if agent_name == "evaluator" and set(missing_params) & {"model_fir", "model_sec"}:
                    missing_params = [param for param in missing_params if param != "model_path"]
                required_params = self._extract_pending_param_names(
                    {
                        "response": context_response_text,
                        "friendly_response": friendly_text,
                        "original_task": original_task_for_pending,
                        "agent_name": agent_name,
                    },
                )
                context_required, context_missing, context_known = self._pending_required_and_missing_from_context(
                    agent_name,
                    full_task,
                )
                if context_required:
                    required_params = context_required
                    missing_params = context_missing
            protocol = self._need_input_protocol(
                agent_name,
                request_kind,
                friendly_text,
                response_text,
                required_params=required_params,
                missing_params=missing_params,
            )
            if request_kind == "param" and context_known:
                protocol = self._normalize_protocol(
                    {**protocol, "knownParams": context_known},
                    str(protocol.get("message") or friendly_text or context_response_text or ""),
                    source=protocol.get("source") if isinstance(protocol, dict) else None,
                    confidence=protocol.get("confidence") if isinstance(protocol, dict) else None,
                ) or protocol
            if request_kind == "param":
                protocol = self._enrich_dataset_name_need_input_protocol(
                    protocol,
                    "\n".join([original_task_for_pending or "", full_task or task_description or ""]),
                    friendly_text or context_response_text,
                    container=self._training_container_for_request(task_description) if agent_name == "trainer" else None,
                )
            self.pending_parameters[agent_name] = self._pending_state(
                agent_name,
                original_task_for_pending,
                request_kind,
                context_response_text,
                friendly_text,
                protocol,
                required_params=required_params,
                needs_choice="[等待用户选择]" in response_text,
            )
            self.last_response_protocol = protocol
            return ToolResponse(content=[TextBlock(type="text", text=friendly_text)],
                metadata={
                    "success": True,
                    "protocol": protocol,
                    "response_msg": self._agent_msg(agent.name, self._strip_user_waiting_wrapper(friendly_text), protocol),
                },
            )
        else:
            if self._agent_request_requires_tool_call(agent_name, task_description):
                return self._missing_tool_call_response(agent_name, task_description, agent)

            # 任务完成或普通响应
            if agent_name in self.pending_parameters:
                del self.pending_parameters[agent_name]
            
            if agent_name == "dataprocessor":
                match = re.search(r'数据类型[为是:：]\s*(sft|dpo)', context_response_text, re.IGNORECASE)
                if match:
                    self.task_context['data_type'] = match.group(1).lower()
                    logger.info(f"[Orchestrator] 记录数据类型: {self.task_context['data_type']}")
            
            # 记录任务历史
            self.last_completed_task = {
                "agent": agent_name,
                "task": task_description,
                "params": additional_params,
                "response": context_response_text,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.current_task_state = {"status": "idle"}

            display_text = self._friendly_request_text(agent_name, response_text)
            protocol = self._job_started_protocol(
                agent_name,
                task_description,
                additional_params,
                display_text,
                context_response_text,
            ) or self._message_protocol(agent_name, display_text)
            self.last_response_protocol = protocol
            tool_text = f"{agent_name} 执行结果：\n{display_text}"
            return ToolResponse(content=[TextBlock(type="text", text=tool_text)],
                metadata={
                    "success": True,
                    "protocol": protocol,
                    "response_msg": self._agent_msg(agent.name, display_text, protocol),
                },
            )
    
    async def _call_datacollector(self, task_description: str, **kwargs) -> ToolResponse:
        """调用datacollector"""
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        additional_params = ""
        if "datacollector" in self.pending_parameters:
            original_task = self.pending_parameters["datacollector"]["original_task"]
            additional_params = task_description.replace(original_task, "").strip()
        return await self._call_agent_with_params("datacollector", task_description, additional_params)
    

    async def _call_dataprocessor(self, task_description: str, **kwargs) -> ToolResponse:
        """调用dataprocessor"""
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        lowered_task = task_description.lower()
        if "medical-example" in lowered_task and not any(
            key in lowered_task for key in ["sft", "dpo", "inspection", "diagnosis", "prescription", "data_type", "strategy"]
        ):
            raw_path = "/home/workspace/dataset/medical-example"
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=self._build_preprocess_guidance("medical-example", raw_path),
                )
            ])

        additional_params = ""
        if "dataprocessor" in self.pending_parameters:
            original_task = self.pending_parameters["dataprocessor"]["original_task"]
            additional_params = task_description.replace(original_task, "").strip()
        return await self._call_agent_with_params("dataprocessor", task_description, additional_params)

    def _docker_path_exists(self, container: str, path: str) -> Optional[bool]:
        try:
            process = subprocess.run(
                ["docker", "exec", container, "sh", "-c", f"test -e {shlex.quote(path)}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return process.returncode == 0
        except Exception:
            return None

    def _list_dataset_candidates_in_container(self, container: str, dataset_dir: str) -> List[str]:
        """列出数据目录内可作为 dataset_name 的 json 文件名（不含 .json）。"""
        try:
            info_command = f"cat {shlex.quote(dataset_dir.rstrip('/') + '/dataset_info.json')} 2>/dev/null"
            info_process = subprocess.run(
                ["docker", "exec", container, "sh", "-c", info_command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if info_process.returncode == 0 and info_process.stdout.strip():
                try:
                    dataset_info = json.loads(info_process.stdout)
                    candidates = dataset_names_from_dataset_info(dataset_info)
                    if candidates:
                        return candidates
                except json.JSONDecodeError:
                    pass

            excluded = dataset_find_exclusion_clause()
            command = (
                f"find {shlex.quote(dataset_dir)} -maxdepth 1 -type f -name '*.json' "
                f"{excluded} -printf '%f\n' | sort"
            )
            process = subprocess.run(
                ["docker", "exec", container, "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if process.returncode != 0 or not process.stdout.strip():
                return []
            candidates = []
            for filename in process.stdout.strip().splitlines():
                filename = filename.strip()
                if not is_dataset_candidate_filename(filename):
                    continue
                candidates.append(filename[:-5])
            return candidates
        except Exception:
            return []

    def _enrich_dataset_name_need_input_protocol(
        self,
        protocol: Dict[str, Any],
        task_description: str,
        response_text: str,
        *,
        container: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach file-level dataset_name options when trainer only needs dataset_name."""
        if not isinstance(protocol, dict) or protocol.get("type") != "need_input":
            return protocol
        if str(protocol.get("agent") or "").lower() != "trainer":
            return protocol

        param_deduper = getattr(self, "_dedupe_required_params", lambda params: [str(param) for param in params if str(param).strip()])
        required_params = param_deduper(
            [str(param) for param in (protocol.get("requiredParams") or []) if str(param).strip()]
        )
        raw_missing_params = protocol.get("missingParams")
        missing_params = param_deduper(
            [str(param) for param in (raw_missing_params or []) if str(param).strip()]
        )

        known_params = dict(protocol.get("knownParams") or {})
        dataset_dir = str(known_params.get("dataset_dir") or "").strip()
        if not dataset_dir:
            machine_text = "\n".join([task_description or "", str(protocol.get("message") or "")])
            dataset_dir_match = re.search(
                r"(?:^|[\s,，;；])dataset_dir\s*(?:=|:)\s*([^\s,，;；`<>]+)",
                machine_text,
                re.IGNORECASE,
            )
            if dataset_dir_match:
                dataset_dir = dataset_dir_match.group(1).strip().strip("'\"")

        legacy_dataset_name_only = raw_missing_params is None and required_params == ["dataset_name"]
        if not missing_params and legacy_dataset_name_only:
            missing_params = ["dataset_name"]
        if missing_params != ["dataset_name"]:
            return protocol
        request_container = container or str(protocol.get("container") or "").strip() or self.training_container
        options = self._list_dataset_candidates_in_container(request_container, dataset_dir) if dataset_dir else []
        if not options:
            options = self._sanitize_dataset_name_options(self._extract_bullet_items(response_text))
        enriched = dict(protocol)
        if dataset_dir and "dataset_name" in required_params:
            enriched["requiredParams"] = ["model_path", "dataset_dir", "dataset_name"]
        elif required_params:
            enriched["requiredParams"] = required_params
        enriched["missingParams"] = ["dataset_name"]
        if dataset_dir:
            enriched["knownParams"] = {**known_params, "dataset_dir": dataset_dir}
        elif known_params:
            enriched["knownParams"] = known_params
        enriched["container"] = request_container
        if options:
            enriched["options"] = options
        else:
            existing_options = [str(option).strip() for option in enriched.get("options") or []]
            if existing_options and all("数据集名称" in option or "dataset_name" in option for option in existing_options):
                enriched.pop("options", None)
        return self._normalize_protocol(enriched, str(enriched.get("message") or response_text or "")) or enriched

    def _find_default_base_model_path(
        self,
        container: str,
        base_root: str = "/home/workspace/models/base",
    ) -> Optional[str]:
        """从容器内基础模型目录中选择一个默认模型路径。"""
        try:
            command = (
                f"find {shlex.quote(base_root)} -mindepth 1 -maxdepth 1 -type d "
                "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-"
            )
            process = subprocess.run(
                ["docker", "exec", container, "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if process.returncode != 0:
                return None
            model_path = (process.stdout or "").strip().splitlines()
            if not model_path:
                return None
            return model_path[0].strip() or None
        except Exception:
            return None

    def _choose_dataset_name(self, candidates: List[str]) -> Optional[str]:
        """在多个候选 dataset_name 中挑选明显更像训练主文件的一个。"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def score(name: str) -> int:
            lower = name.lower()
            value = 0
            if "for_train" in lower:
                value += 100
            if "train" in lower:
                value += 50
            if "altered" in lower:
                value += 20
            if "output" in lower:
                value += 10
            if "qwen" in lower:
                value += 5
            if "test" in lower:
                value -= 100
            return value

        ranked = sorted(((score(name), name) for name in candidates), reverse=True)
        top_score, top_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -10**9
        if top_score >= 50 and top_score > second_score:
            return top_name
        return None

    def _extract_dataset_date(self, text: str) -> Optional[str]:
        match = re.search(r"(?<!\d)(20\d{6})(?!\d)", text or "")
        return match.group(1) if match else None

    def _extract_dataset_reference(self, text: str) -> Optional[str]:
        text = text or ""
        normalized = text.replace("：", ":")
        dataset_patterns = [
            r"(?:data_identifier|dataset_id|dataset_date|data_date|数据标识|数据日期|训练日期)\s*(?:是|为|=|:)?\s*(20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?)",
            r"(?:dataset_dir|data_dir|数据集路径|数据路径|数据集目录|数据目录|数据集位置|数据位置)\s*(?:是|为|=|:)?\s*/[^\s,，;；]*(?:dataset_batch_train|dataset_daily_train)/([A-Za-z0-9_.-]+)",
            r"/[^\s,，;；]*(?:dataset_batch_train|dataset_daily_train)/(20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?)",
            r"(?:用|使用|拿)\s*(20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?)\s*(?:这个|这份)?(?:数据|数据集)",
            r"(20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?)\s*(?:这个|这份)?(?:数据|数据集)",
        ]
        for pattern in dataset_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip("/")

        scrubbed = re.sub(r"/[^\s,，;；]*(?:models|medical_models)/[^\s,，;；]*", " ", normalized)
        scrubbed = re.sub(r"\bmodel_medical_(?:lora|full)_20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?\b", " ", scrubbed)
        ref_match = re.search(r"(?<![A-Za-z0-9_.-])(20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?)(?![A-Za-z0-9_.-])", scrubbed)
        if ref_match:
            return ref_match.group(1)

        dataset_date = self._extract_dataset_date(scrubbed)
        if dataset_date:
            return dataset_date

        lowered = text.lower()
        known_datasets = [
            "medical-example",
        ]
        for dataset_name in known_datasets:
            if dataset_name in lowered:
                return dataset_name
        return None
    def _extract_workflow_benchmark_name(self, text: str) -> str:
        normalized = (text or "").replace("：", ":").strip()
        patterns = [
            r"(?:评测集|评测数据集|基准评测|benchmark)\s*(?:使用|用|为|是|=|:)\s*([A-Za-z0-9_.-]+)",
            r"(?:使用|用)\s*([A-Za-z0-9_.-]+)\s*(?:评测集|评测数据集|基准评测|benchmark)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("`'\"“”‘’")
        return "2024"

    def _has_training_type(self, text: str) -> bool:
        if self._infer_multinode_train_family(text):
            return True 
        keywords = [
            "lora", "LoRA", "sft", "SFT", "全参", "定时", "增强", "dpo", "DPO",
            "grpo", "GRPO", "grpo_train",
            "batch_train_lora", "batch_train_full", "dpo_train_launcher",
        ]
        return any(keyword in (text or "") for keyword in keywords)

    def _build_trainer_system_hints(self, task_description: str) -> str:
        """给 trainer 补充确定性上下文，减少参数幻觉和无谓追问。"""
        hints = [
            "\n[系统补充] 执行训练前必须遵守：",
            "- 禁止编造参数；不得把“模型位置/数据集位置/数据集名称/xxx/待提供”等占位词当成真实参数。",
            "- 所有训练工具都必须后台运行；调用训练工具时 background 必须为 true。",
            "- 如果用户只说想训练模型且未指定训练类型，按新手默认选择 lora批量训练。",
        ]
        inferred_train_type = self._infer_train_type(task_description)
        requested_train_type = self._normalize_train_type(inferred_train_type)
        has_training_type = self._has_training_type(task_description)
        if inferred_train_type == "grpo":
            hints.append(
                "- 本次训练类型明确为 grpo；只使用 model_path、train_files、val_files，"
                "禁止改判为增强训练或多机增强训练。"
            )
        else:
            hints.append(
                "- 增强训练必须使用真实 model_path、dataset_dir、dataset_name；"
                "缺失时返回[参数请求]，不要启动工具。"
            )
        is_multinode = self._is_multinode_training_request(task_description)
        multinode_family = self._infer_multinode_train_family(task_description)
        if is_multinode:
            hints.append("- 用户明确要求多机训练；不要把多机当成新的 train_type，只在现有 SFT/DPO 训练下选择多机脚本。")
            if multinode_family == "lora":
                hints.append("- 本次应调用 run_script_by_name_train，script_query 使用 多机lora批量训练。")
            elif multinode_family == "enhanced":
                hints.append("- 本次应调用 run_script_by_name_train，script_query 使用 多机增强训练。")
                hints.append("- 多机增强训练仍必须优先使用用户真实输入或自动推断的 model_path、dataset_dir、dataset_name。")
            else:
                hints.append("- 用户只说多机训练但未说明 SFT/DPO；如果无法根据数据目录判断，必须先询问训练类型。")
            multinode_args = self._extract_multinode_cli_args(task_description)
            if multinode_args:
                arg_text = ", ".join(f"{key}={value}" for key, value in sorted(multinode_args.items()))
                hints.append(f"- 已从用户请求中识别到多机参数：{arg_text}。调用工具时应放入 additional_args。")

        dataset_ref = self._extract_dataset_reference(task_description)
        if not dataset_ref:
            return "\n".join(hints)

        container = self._training_container_for_request(task_description)
        batch_path = f"/home/workspace/dataset_batch_train/{dataset_ref}"
        daily_path = f"/home/workspace/dataset_daily_train/{dataset_ref}"
        raw_path = f"/home/workspace/dataset/{dataset_ref}"

        batch_exists = self._docker_path_exists(container, batch_path)
        daily_exists = self._docker_path_exists(container, daily_path)
        raw_exists = self._docker_path_exists(container, raw_path)

        hints.append(f"- 识别到数据标识：{dataset_ref}。")
        if batch_exists is True:
            hints.append(f"- SFT候选数据目录存在：{batch_path}。适合 lora批量训练 或 全参批量训练。")
        elif batch_exists is False:
            hints.append(f"- SFT候选数据目录未检测到：{batch_path}。")
        else:
            hints.append(f"- SFT候选数据目录：{batch_path}（当前无法自动确认是否存在）。")

        if daily_exists is True and (not has_training_type or requested_train_type == "enhanced"):
            hints.append(f"- DPO候选数据目录存在：{daily_path}。适合 增强训练。")
        elif daily_exists is False:
            hints.append(f"- DPO候选数据目录未检测到：{daily_path}。")
        else:
            hints.append(f"- DPO候选数据目录：{daily_path}（当前无法自动确认是否存在）。")

        if raw_exists is True:
            hints.append(f"- 原始数据目录存在：{raw_path}；如果用户要直接训练，应先进行数据处理。")

        daily_candidates = self._list_dataset_candidates_in_container(container, daily_path) if daily_exists is True else []
        daily_dataset_name = self._choose_dataset_name(daily_candidates)
        default_base_model_path = self._find_default_base_model_path(container)
        if daily_dataset_name:
            hints.append(f"- 从DPO数据目录推断 dataset_name={daily_dataset_name}。")
        elif len(daily_candidates) > 1:
            hints.append(f"- DPO数据目录下存在多个数据文件候选：{', '.join(daily_candidates[:8])}。不能直接猜测 dataset_name。")

        has_training_type = self._has_training_type(task_description)
        if not has_training_type and daily_exists is True and batch_exists is not True:
            default_mode = "多机增强训练" if is_multinode else "增强训练"
            hints.append(
                f"- 用户未指定训练类型，但仅检测到DPO数据目录；按新手默认执行 {default_mode}，"
                "不要再次询问训练类型。"
            )
            if daily_dataset_name:
                if default_base_model_path:
                    hints.append(
                        "- 增强训练默认补全："
                        f"model_path={default_base_model_path}，"
                        f"dataset_dir={daily_path}，dataset_name={daily_dataset_name}。"
                    )
                else:
                    hints.append(
                        "- 已推断增强训练数据参数，但未能从 /home/workspace/models/base 自动发现基础模型；"
                        "必须先询问 model_path，禁止编造。"
                    )
            else:
                hints.append("- 未能从DPO数据目录内推断唯一 dataset_name；必须先询问用户，禁止编造。")
        elif "增强" in task_description or "dpo" in task_description.lower():
            if daily_exists is True and daily_dataset_name:
                if default_base_model_path:
                    hints.append(
                        "- 若执行增强训练，默认补全："
                        f"model_path={default_base_model_path}，"
                        f"dataset_dir={daily_path}，dataset_name={daily_dataset_name}。"
                    )
                else:
                    hints.append(
                        "- 若执行增强训练，已推断 dataset_dir 和 dataset_name，"
                        "但未能从 /home/workspace/models/base 自动发现基础模型；必须先询问 model_path。"
                    )
            elif daily_exists is True:
                hints.append("- 若执行增强训练，必须先询问 dataset_name；禁止使用日期作为 dataset_name，且不要在多文件场景下自行猜测。")
            else:
                hints.append(
                    f"- 用户想执行增强训练，但未检测到默认DPO数据目录：{daily_path}；"
                    "应先确认真实的 dataset_dir 和 dataset_name"
                )
        elif not has_training_type:
            if batch_exists is True:
                default_mode = "多机lora批量训练" if is_multinode else "lora批量训练"
                hints.append(f"- 用户未指定训练类型，且检测到SFT数据目录；默认执行 {default_mode}，不要再次询问训练类型。")

            elif raw_exists is True:
                hints.append("- 用户未指定训练类型，且只检测到原始数据目录；训练前必须先调用 dataprocessor 做数据预处理，不要直接启动训练。")
            else:
                hints.append("- 未检测到可训练数据目录；必须先向用户确认数据路径或先执行数据准备/预处理。")

        return "\n".join(hints)

    def _strip_trainer_system_hints(self, task_description: str) -> str:
        """Remove orchestration-only suffixes from restored trainer pending state."""
        clean_task = str(task_description or "")
        for marker in ("\n[系统补充]", "\n[系统约束]", "\n[系统路由提示]"):
            clean_task = clean_task.partition(marker)[0]
        return clean_task.strip()

    def _build_training_route_hint(self, message: str) -> Optional[str]:
        text = message or ""
        has_training_keyword = "训练" in text or "train" in text.lower()
        has_training_dataset_path = (
            "/home/workspace/dataset_batch_train/" in text
            or "/home/workspace/dataset_daily_train/" in text
        )
        if not has_training_keyword and not has_training_dataset_path:
            return None

        dataset_ref = self._extract_dataset_reference(text)
        if not dataset_ref:
            return None

        is_multinode = self._is_multinode_training_request(text)
        container = self._training_container_for_request(text)

        batch_path = f"/home/workspace/dataset_batch_train/{dataset_ref}"
        daily_path = f"/home/workspace/dataset_daily_train/{dataset_ref}"
        raw_path = f"/home/workspace/dataset/{dataset_ref}"

        batch_exists = self._docker_path_exists(container, batch_path)
        daily_exists = self._docker_path_exists(container, daily_path)
        raw_exists = self._docker_path_exists(container, raw_path)
        requested_train_type = self._normalize_train_type(self._infer_train_type(text))
        has_training_type = self._has_training_type(text)
        named_values = self._extract_named_param_values(text)
        explicit_paths = [
            value.strip().rstrip("/")
            for key, value in named_values.items()
            if key in {"dataset_dir", "input_folder"} and value
        ]
        explicit_paths.extend(
            path.strip().rstrip("/")
            for path in self._extract_path_tokens(text)
            if path
        )
        explicit_daily_path = any(
            path == daily_path or path.startswith(f"{daily_path}/")
            for path in explicit_paths
        )
        explicit_batch_path = any(
            path == batch_path or path.startswith(f"{batch_path}/")
            for path in explicit_paths
        )

        if (
            batch_exists is True
            and daily_exists is True
            and not has_training_type
            and not explicit_daily_path
            and not explicit_batch_path
        ):
            return (
                "[系统路由提示] 同一数据标识同时存在于 /home/workspace/dataset_batch_train "
                "和 /home/workspace/dataset_daily_train。"
                "目录含义必须按以下规则解释：/home/workspace/dataset_batch_train 是 SFT 数据，适合 lora训练；"
                "/home/workspace/dataset_daily_train 是 DPO 数据，适合增强训练。"
                "无法安全判断训练类型，必须先询问用户选择 lora训练 或 增强训练。"
            )

        if batch_exists is True and requested_train_type in {None, "lora", "full"} and not explicit_daily_path:
            default_mode = "多机lora批量训练" if is_multinode else "lora批量训练"
            return (
                "[系统路由提示] 检测到该日期数据位于 /home/workspace/dataset_batch_train。"
                "/home/workspace/dataset_batch_train 是 SFT 数据目录，适合 lora批量训练 或 全参批量训练，禁止说成增强训练。"
                f"如果用户未明确训练类型，必须调用 _call_trainer，默认执行 {default_mode}。"
            )

        if daily_exists is True and (explicit_daily_path or not has_training_type or requested_train_type == "enhanced"):
            candidates = self._list_dataset_candidates_in_container(container, daily_path)
            dataset_name = self._choose_dataset_name(candidates)
            default_mode = "多机增强训练" if is_multinode else "增强训练"
            if dataset_name:
                return (
                    "[系统路由提示] 检测到该日期数据位于 /home/workspace/dataset_daily_train。"
                    "/home/workspace/dataset_daily_train 是 DPO 数据目录，适合增强训练，禁止说成 LoRA 训练。"
                    f"如果用户未明确训练类型，必须调用 _call_trainer，默认执行{default_mode}；"
                    f"dataset_dir={daily_path}，dataset_name={dataset_name}。"
                )
            return (
                "[系统路由提示] 检测到该日期数据位于 /home/workspace/dataset_daily_train，"
                "/home/workspace/dataset_daily_train 是 DPO 数据目录，适合增强训练，禁止说成 LoRA 训练。"
                f"默认应走{default_mode}，但未能唯一推断 dataset_name；候选包括：{', '.join(candidates[:8]) if candidates else '无'}。必须先询问用户。"
            )

        if raw_exists is True:
            return (
                f"[系统路由提示] 检测到数据 {dataset_ref} 仍在 /home/workspace/dataset 原始数据目录。"
                "训练前不能直接开始训练，必须先调用 dataprocessor 的 data_preprocessing 工具。"
                "调用时带上该数据目录作为 input_folder，让工具先检测格式；"
                "通用格式由工具只追问 data_type，医疗 raw 才追问 data_type 和 strategy。"
            )

        return "[系统路由提示] 未检测到该数据对应的可训练数据目录；先询问用户确认数据位置。"

    def _build_preprocess_guidance(self, dataset_ref: str, raw_path: str) -> str:
        return f"""检测到 `{dataset_ref}` 目前还在原始数据目录 `{raw_path}`，还不能直接开始训练。

先做一步数据预处理就行。我会先调用预处理工具检测数据格式；如果是 OpenAI、ShareGPT、SFT、DPO 或 text 等通用格式，只需要你补充 `data_type=sft` 或 `data_type=dpo`；如果检测为医疗 raw 原始格式，再补充 `strategy=inspection/diagnosis/prescription`。

可以这样理解：
- `sft`：常规监督微调，适合先做 LoRA / 全参训练
- `dpo`：偏好优化数据，适合后续增强训练
- `strategy`：只用于医疗 raw 原始格式，表示检查、诊断或处方方向
"""

    def _is_inference_preview_stop_command(self, message: str) -> bool:
        text = (message or "").lower()
        compact = re.sub(r"\s+", "", text)
        return any(
            marker in compact
            for marker in (
                "预览停止",
                "停止预览",
                "previewstop",
                "stoppreview",
                "dry-runstop",
                "dryrunstop",
                "stopdry-run",
                "stopdryrun",
            )
        )

    def _is_inference_command(self, message: str) -> bool:
        text = (message or "").lower()
        compact = re.sub(r"\s+", "", text)
        inference_keywords = [
            "推理服务",
            "推理配置",
            "推理基准",
            "推理基准测试",
            "基准测试",
            "功能测试",
            "service_test",
            "benchmark",
            "medbench",
            "inference",
            "step1",
            "step2",
            "step3",
            "2021.json",
            "2024.json",
        ]
        if any(keyword in text for keyword in inference_keywords):
            return True
        if "job_id=" in compact or "jobid=" in compact:
            non_inference_targets = (
                "训练",
                "trainer",
                "train",
                "评估",
                "评测",
                "evaluator",
                "evaluate",
                "evaluation",
                "数据处理",
                "预处理",
                "dataprocessor",
            )
            return not any(target in compact for target in non_inference_targets)
        return False

    def _resolve_explicit_agent_route(self, message: str) -> Optional[str]:
        text = (message or "").lower()
        if not text:
            return None
        if self._extract_stop_target_params(message).get("has_stop_intent"):
            if self._is_inference_command(message):
                return "inference"
            return None

        # Evaluation requests must be able to replace a stale training parameter
        # wait.  Without a deterministic route they are treated as another reply
        # to trainer and the old training prompt is returned again.
        evaluation_keywords = [
            "单模型评估", "单模型评测", "双模型评估", "双模型评测",
            "ckpt评估", "ckpt评测", "checkpoint评估", "checkpoint评测",
        ]
        if any(keyword in text for keyword in evaluation_keywords):
            return "evaluator"

        grpo_requested = "grpo" in text or "grpo_train" in text
        grpo_start_intent = any(keyword in text for keyword in ["启动", "开始", "执行", "运行", "训练", "跑", "start", "launch", "run", "train"])
        grpo_monitor_intent = any(keyword in text for keyword in ["监控", "状态", "查询", "查看", "进度", "怎么样", "如何", "monitor", "status"])
        if grpo_requested and grpo_start_intent and not grpo_monitor_intent:
            return "trainer"

        if self._is_multinode_training_request(message) and any(
            keyword in text
            for keyword in ["启动", "开始", "执行", "运行", "训练", "跑", "start", "launch", "run", "train"]
        ):
            return "trainer"

        explicit_training_keywords = ["lora批量训练", "全参批量训练", "定时训练", "增强训练", "多机lora批量训练", "多机增强训练"]

        inference_keywords = [
            "推理服务",
            "推理配置",
            "推理基准",
            "推理基准测试",
            "基准测试",
            "benchmark",
            "medbench",
            "step1",
            "step2",
            "step3",
            "2021.json",
            "2024.json",
        ]
        if any(keyword in text for keyword in inference_keywords):
            return "inference"

        monitor_keywords = ["监控", "状态", "进度", "查看训练", "训练状态", "monitor", "status"]
        monitor_targets = ["训练", "trainer", "pid", "容器", "container", "评估", "evaluation", "推理", "inference"]
        if any(keyword in text for keyword in monitor_keywords) and any(keyword in text for keyword in monitor_targets):
            return "monitor"

        if any(keyword in text for keyword in explicit_training_keywords):
            return "trainer"

        if any(keyword in text for keyword in ["高级筛选", "数据高级筛选", "预处理", "数据预处理", "数据处理"]):
            return "dataprocessor"

        if any(keyword in text for keyword in ["数据准备", "准备数据", "数据收集", "收集数据"]):
            return "datacollector"

        return None

    #def _is_task_related(self, message: str) -> bool:
    #    """判断用户输入是否与业务任务相关。
        
    #    返回 True：包含训练、数据、评估、推理等业务关键词
    #    返回 False：闲聊、问候、个人信息询问等
    #    """
    #    text = (message or "").lower()
    #    if not text:
    #        return False
        
        # 明确闲聊模式
    #    casual_patterns = [
    #        r"^我(?:叫|是|的)(?:什么|谁|名字)",
    #        r"^(?:你好|您好|嗨|hello|hi)",
    #        r"^(?:谢谢|感谢|谢了|thx|thanks)",
    #        r"^(?:再见|拜拜|bye|goodbye)",
    #        r"^(?:今天|现在)(?:天气|几点|星期|日期)",
    #        r"^你(?:是|叫)(?:什么|谁)",
    #        r"^你(?:会|能|可以)(?:做什么|干什么)",
    #    ]
    #    for pattern in casual_patterns:
    #        if re.search(pattern, text):
    #            return False
        
        # 业务关键词
    #    task_keywords = [
            # 训练相关
    #        "训练", "lora", "grpo", "微调", "sft", "dpo", "batch", "定时",
            # 数据相关
    #        "数据", "预处理", "高级筛选", "收集", "dataset", "data",
            # 评估相关
    #        "评估", "评测", "evaluat", "benchmark", "ckpt", "checkpoint",
            # 推理相关
    #        "推理", "inference", "服务", "配置", "启动", "停止", "重启",
    #        "日志", "vllm", "medbench", "2021", "2024", "step1", "step2", "step3",
            # 模型相关
    #        "模型", "model", "checkpoint", "ckpt", "权重", "参数",
            # 监控相关
    #        "监控", "状态", "进度", "monitor", "status", "log",
            # 管理相关
    #        "查看", "list", "目录", "文件", "资源", "gpu", "显存",
    #    ]
    #    return any(keyword in text for keyword in task_keywords)

    def _extract_schedule_time_value(self, text: str) -> Optional[str]:
        raw_text = (text or "").replace("：", ":")
        explicit = re.search(r"schedule_time\s*=\s*([^\n,，;；]+)", raw_text, re.IGNORECASE)
        if explicit:
            return explicit.group(1).strip()

        chinese = re.search(r"(?:定时时间|时间安排|训练时间|时间)\s*(?:是|为|:|：|=)?\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?|[0-9]{1,4})", raw_text)
        if chinese:
            return chinese.group(1).strip()
        return None

    def _normalize_schedule_time_input(self, text: str) -> Optional[str]:
        raw_text = (text or "").replace("：", ":").strip()
        if not raw_text:
            return None

        explicit = self._extract_schedule_time_value(raw_text)
        if explicit:
            raw_text = explicit.strip()
        else:
            bare_time = re.search(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)", raw_text)
            if bare_time:
                raw_text = bare_time.group(1).strip()
            else:
                bare_interval = re.search(r"(?<!\d)(\d{1,4})(?!\d)", raw_text)
                if bare_interval and any(keyword in raw_text.lower() for keyword in ["分钟", "min", "minute", "间隔", "每隔"]):
                    raw_text = bare_interval.group(1).strip()

        time_only = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", raw_text)
        if time_only:
            hour = int(time_only.group(1))
            minute = int(time_only.group(2))
            second = int(time_only.group(3) or 0)
            if hour > 23 or minute > 59 or second > 59:
                return None
            return f"{hour:02d}:{minute:02d}:{second:02d}"

        interval_only = re.fullmatch(r"\d{1,4}", raw_text)
        if interval_only:
            minutes = int(raw_text)
            if minutes <= 0:
                return None
            return str(minutes)

        return None

    def _build_schedule_training_guidance(self) -> str:
        return self._wrap_user_waiting_message(
            "参数补充",
            "开始定时训练前，我还需要先确认训练时间。\n\n"
            "- `schedule_time`：支持两种格式：\n"
            "  `HH:MM:SS` 表示每天这个时间执行一次。\n"
            "  正整数分钟数表示每隔若干分钟执行一次。\n\n"
            "你可以直接回复：`schedule_time=<时间或分钟数>`。",
        )

    def _canonical_required_param(self, token: str) -> Optional[str]:
        value = (token or "").strip().strip("`'\"，,。:：；;()（）[]【】")
        if not value:
            return None
        lower = value.lower()
        aliases = {
            "模型路径": "model_path",
            "模型位置": "model_path",
            "数据集路径": "dataset_dir",
            "数据集目录": "dataset_dir",
            "数据目录": "dataset_dir",
            "数据集名称": "dataset_name",
            "训练数据": "train_files",
            "训练集": "train_files",
            "训练文件": "train_files",
            "训练数据文件": "train_files",
            "验证数据": "val_files",
            "验证集": "val_files",
            "验证文件": "val_files",
            "验证数据文件": "val_files",
            "第一个模型路径": "model_fir",
            "模型一路径": "model_fir",
            "第二个模型路径": "model_sec",
            "模型二路径": "model_sec",
            "checkpoint路径": "CKPT_PATH",
            "ckpt路径": "CKPT_PATH",
            "定时时间": "schedule_time",
            "训练时间": "schedule_time",
            "数据路径": "input_folder",
            "文件夹路径": "input_folder",
        }
        canonical = {
            "model_fir": "model_fir",
            "model_sec": "model_sec",
            "ckpt_path": "CKPT_PATH",
            "checkpoint": "CKPT_PATH",
            "model_path": "model_path",
            "train_files": "train_files",
            "data.train_files": "train_files",
            "val_files": "val_files",
            "data.val_files": "val_files",
            "dataset_dir": "dataset_dir",
            "dataset_name": "dataset_name",
            "input_folder": "input_folder",
            "schedule_time": "schedule_time",
            "data_type": "data_type",
            "strategy": "strategy",
            "container": "container_name",
            "container_name": "container_name",
            "pid": "pid",
        }
        return aliases.get(value) or canonical.get(lower)

    def _dedupe_required_params(self, params: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in params:
            param = self._canonical_required_param(str(item))
            if not param or param in seen:
                continue
            seen.add(param)
            result.append(param)
        return result

    def _extract_explicit_required_params(self, text: str) -> List[str]:
        clean = THINK_TAG_RE.sub("", text or "")
        sections: List[str] = []
        section_patterns = [
            r"(?:需要参数|请提供|缺少必需参数|需要以下必需参数，但未提供|未提供|requiredParams|required_params)\s*[:：]\s*([\s\S]{1,600})",
            r"(?:缺少|需要|补充|确认)(?:以下)?(?:必要|必需)?(?:\d+|[一二三四五六七八九十两]+)?(?:项|个)?(?:参数|信息)[^\n。；;:：]*[:：]?\s*([\s\S]{1,600})",
        ]
        for pattern in section_patterns:
            for match in re.finditer(pattern, clean, re.IGNORECASE):
                section = match.group(1)
                section = re.split(
                    r"(?:\n\s*(?:请用户提供|你可以|例如|示例|如果|当前提供|已阻止)|$)",
                    section,
                    maxsplit=1,
                )[0]
                sections.append(section)

        if not sections:
            if (
                "增强训练" in clean
                and "model_path" in clean
                and "dataset_dir" in clean
                and "dataset_name" in clean
                and any(marker in clean for marker in ("三项", "以下信息", "以下参数"))
            ):
                return ["model_path", "dataset_dir", "dataset_name"]
            return []

        machine_param_re = re.compile(
            r"`?("
            r"model_fir|model_sec|CKPT_PATH|ckpt_path|checkpoint|model_path|dataset_dir|dataset_name|"
            r"train_files|data\.train_files|val_files|data\.val_files|"
            r"input_folder|schedule_time|data_type|strategy|container_name|container|pid"
            r")`?",
            re.IGNORECASE,
        )
        machine_candidates: List[str] = []
        for section in sections:
            for match in machine_param_re.finditer(section):
                machine_candidates.append(match.group(1))
        if machine_candidates:
            return self._dedupe_required_params(machine_candidates)

        candidates: List[str] = []
        param_name_re = re.compile(
            r"`?("
            r"model_fir|model_sec|CKPT_PATH|ckpt_path|checkpoint|model_path|dataset_dir|dataset_name|"
            r"train_files|data\.train_files|val_files|data\.val_files|"
            r"input_folder|schedule_time|data_type|strategy|container_name|container|pid|"
            r"模型路径|模型位置|数据集路径|数据集目录|数据目录|数据集名称|第一个模型路径|模型一路径|"
            r"第二个模型路径|模型二路径|checkpoint路径|ckpt路径|定时时间|训练时间|数据路径|文件夹路径|"
            r"训练数据|训练集|训练文件|训练数据文件|验证数据|验证集|验证文件|验证数据文件"
            r")`?",
            re.IGNORECASE,
        )
        for section in sections:
            for match in param_name_re.finditer(section):
                candidates.append(match.group(1))
        return self._dedupe_required_params(candidates)

    def _full_required_params_for_missing(
        self,
        agent_name: str,
        missing_params: List[str],
        text: str = "",
    ) -> List[str]:
        missing = self._dedupe_required_params(missing_params)
        content = THINK_TAG_RE.sub("", text or "")
        lower = content.lower()
        if (
            agent_name == "trainer"
            and missing == ["dataset_name"]
            and ("增强训练" in content or "dpo_train_launcher" in lower or "dataset_dir" in lower)
        ):
            return ["model_path", "dataset_dir", "dataset_name"]
        return missing

    def _infer_required_params_from_task(self, agent_name: str, text: str) -> List[str]:
        content = THINK_TAG_RE.sub("", text or "")
        lower = content.lower()
        if agent_name == "trainer":
            if "grpo" in lower or "grpo_train" in lower:
                return ["model_path", "train_files", "val_files"]
            if any(keyword in content for keyword in ["增强训练"]) or "dpo_train_launcher" in lower:
                return ["model_path", "dataset_dir", "dataset_name"]
        if agent_name == "evaluator":
            if "ckpt_eval" in lower or "ckpt评估" in content:
                return ["CKPT_PATH"]
            if "single_model_evaluation_vpn" in lower or "单模型" in content:
                return ["model_fir"]
            if "compare_between_models_vpn" in lower or "双模型" in content:
                return ["model_fir", "model_sec"]
        if agent_name == "dataprocessor":
            if "高级筛选" in content or "score_based_filtering" in lower:
                return ["input_folder"]
            if "数据预处理" in content or "data_preprocessing" in lower:
                # Data preprocessing required params depend on the detected input format.
                # Let the tool inspect input_folder or the latest default dataset first.
                return []
        return []

    def _extract_pending_param_names(self, task_info: Dict[str, Any]) -> List[str]:
        structured_params = task_info.get("required_params") or task_info.get("requiredParams")
        if isinstance(structured_params, list):
            return self._dedupe_required_params([str(param) for param in structured_params if str(param).strip()])

        protocol = task_info.get("protocol")
        if isinstance(protocol, dict):
            protocol_params = protocol.get("requiredParams")
            if isinstance(protocol_params, list):
                return self._dedupe_required_params([str(param) for param in protocol_params if str(param).strip()])

        agent_name = str(task_info.get("agent_name") or "")
        response_text = "\n".join(
            str(task_info.get(key, "") or "")
            for key in ("response", "friendly_response", "message")
        )
        explicit_params = self._extract_explicit_required_params(response_text)
        if explicit_params:
            if agent_name == "evaluator" and set(explicit_params) & {"model_fir", "model_sec"}:
                explicit_params = [param for param in explicit_params if param != "model_path"]
            return self._full_required_params_for_missing(agent_name, explicit_params, response_text)

        text = "\n".join(
            str(task_info.get(key, "") or "")
            for key in ("response", "friendly_response", "message", "original_task")
        )
        inferred_params = self._infer_required_params_from_task(agent_name, text)
        if inferred_params:
            return inferred_params

        known_params = [
            "model_fir",
            "model_sec",
            "CKPT_PATH",
            "model_path",
            "train_files",
            "val_files",
            "dataset_dir",
            "dataset_name",
            "input_folder",
            "schedule_time",
            "data_type",
            "strategy",
        ]
        contextual_hits = []
        for param in known_params:
            for match in re.finditer(re.escape(param), text, re.IGNORECASE):
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                window = text[start:end]
                if any(keyword in window for keyword in ["缺少", "需要", "必需", "未提供", "请提供", "确认"]):
                    contextual_hits.append(param)
                    break
        return self._dedupe_required_params(contextual_hits)

    def _extract_path_tokens(self, message: str) -> List[str]:
        text = (message or "").strip().replace("：", ":")
        return re.findall(r"(/[^\s,，;；]+)", text)

    def _looks_like_path_value(self, text: str) -> bool:
        value = (text or "").strip().strip("'\"")
        if not value:
            return False
        if value.startswith("/"):
            return True
        return False

    def _extract_named_param_values(self, message: str) -> Dict[str, str]:
        text = (message or "").replace("：", ":")
        extracted: Dict[str, str] = {}

        patterns = {
            "model_path": [
                r"(?:model_path|模型路径|模型位置|基础模型路径|模型在)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "train_files": [
                r"(?:train_files|data\.train_files|训练数据|训练集|训练文件|训练数据文件)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "val_files": [
                r"(?:val_files|data\.val_files|验证数据|验证集|验证文件|验证数据文件)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "dataset_dir": [
                r"(?:dataset_dir|数据集路径|数据路径|数据集目录|数据目录|数据集位置|数据位置|数据)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "input_folder": [
                r"(?:input_folder|数据路径|数据目录|文件夹路径|目录路径)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "CKPT_PATH": [
                r"(?:CKPT_PATH|ckpt|checkpoint|checkpoint路径|ckpt路径)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "model_fir": [
                r"(?:model_fir|第一个模型路径|模型一路径|模型一)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
            "model_sec": [
                r"(?:model_sec|第二个模型路径|模型二路径|模型二)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            ],
        }

        for param_name, regexes in patterns.items():
            for pattern in regexes:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[param_name] = match.group(1).strip()
                    break

        data_type_match = re.search(
            r"(?:data_type|数据类型)\s*(?:是|为|=|:)?\s*(sft|dpo)",
            text,
            re.IGNORECASE,
        )
        if data_type_match:
            extracted["data_type"] = data_type_match.group(1).lower()

        strategy_match = re.search(
            r"(?:strategy|数据格式|数据策略|任务方向)\s*(?:是|为|=|:)?\s*(inspection|diagnosis|prescription)",
            text,
            re.IGNORECASE,
        )
        if strategy_match:
            extracted["strategy"] = strategy_match.group(1).lower()

        dataset_name_match = re.search(
            r"(?:dataset_name|数据集名称)\s*(?:是|为|=|:)?\s*([A-Za-z0-9_.-]+)",
            text,
            re.IGNORECASE,
        )
        if dataset_name_match:
            extracted["dataset_name"] = dataset_name_match.group(1).strip()

        return extracted

    def _infer_dataset_name_from_dir(self, dataset_dir: str, container: Optional[str] = None) -> Optional[str]:
        dataset_dir = (dataset_dir or "").strip()
        if not dataset_dir.startswith("/"):
            return None
        candidates = self._list_dataset_candidates_in_container(container or self.training_container, dataset_dir)
        return self._choose_dataset_name(candidates)

    def _normalize_dpo_dataset_params(self, values: Dict[str, str], container: Optional[str] = None) -> Dict[str, str]:
        """把增强训练里误填到 dataset_name 的日期归一化为 dataset_dir 子目录。"""
        normalized = dict(values)
        dataset_dir = (normalized.get("dataset_dir") or "").strip().rstrip("/")
        dataset_name = (normalized.get("dataset_name") or "").strip()
        date_match = re.fullmatch(r"20\d{6}", dataset_name)
        if not dataset_dir or not date_match:
            return normalized

        if not dataset_dir.endswith(f"/{dataset_name}"):
            dataset_dir = f"{dataset_dir}/{dataset_name}"
            normalized["dataset_dir"] = dataset_dir

        inferred_dataset_name = self._infer_dataset_name_from_dir(dataset_dir, container=container)
        if inferred_dataset_name:
            normalized["dataset_name"] = inferred_dataset_name
        else:
            normalized.pop("dataset_name", None)
        return normalized

    def _normalize_pending_param_message(self, task_info: Dict[str, Any], message: str) -> str:
        raw_message = (message or "").strip()
        if not raw_message:
            return raw_message

        expected_params = self._extract_pending_param_names(task_info)
        if not expected_params:
            return raw_message

        named_values = self._extract_named_param_values(raw_message)
        merged_values: Dict[str, str] = {
            key: value for key, value in named_values.items() if key in expected_params and value
        }

        if "schedule_time" in expected_params:
            normalized_schedule_time = self._normalize_schedule_time_input(raw_message)
            if normalized_schedule_time:
                merged_values["schedule_time"] = normalized_schedule_time

        path_params = [param for param in expected_params if param in {"model_fir", "model_sec", "CKPT_PATH", "model_path", "train_files", "val_files", "dataset_dir", "input_folder"}]
        path_tokens = self._extract_path_tokens(raw_message)

        unresolved_path_params = [param for param in path_params if param not in merged_values]
        if len(unresolved_path_params) == 1:
            if path_tokens:
                merged_values[unresolved_path_params[0]] = path_tokens[0]
            elif self._looks_like_path_value(raw_message):
                merged_values[unresolved_path_params[0]] = raw_message
        elif len(unresolved_path_params) >= 2 and len(path_tokens) >= len(unresolved_path_params):
            for index, param_name in enumerate(unresolved_path_params):
                merged_values[param_name] = path_tokens[index]

        if "dataset_name" in expected_params and "dataset_name" not in merged_values:
            dataset_dir = merged_values.get("dataset_dir")
            if dataset_dir:
                inferred_dataset_name = self._infer_dataset_name_from_dir(dataset_dir)
                if inferred_dataset_name:
                    merged_values["dataset_name"] = inferred_dataset_name
            elif re.fullmatch(r"[A-Za-z0-9_.-]+", raw_message):
                merged_values["dataset_name"] = raw_message

        if "dataset_dir" in expected_params and "dataset_name" in expected_params:
            merged_values = self._normalize_dpo_dataset_params(merged_values)

        if merged_values:
            ordered_items = [f"{param}={merged_values[param]}" for param in expected_params if param in merged_values]
            if ordered_items:
                return "，".join(ordered_items)

        return raw_message

    def _is_choice_response(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        normalized = text.replace("。", "").replace("！", "").replace("，", "").strip()
        choice_words = {
            "是", "否", "需要", "不需要", "要", "不要", "继续", "不继续",
            "继续执行", "执行", "不执行", "不用", "用", "好", "好的", "确认", "取消",
            "关闭", "完成", "结束",
        }
        return normalized in choice_words


    def _is_terminal_waiting_response(self, agent_name: str, response_text: str) -> bool:
        if "[等待用户选择]" in (response_text or ""):
            return False
        return False

    def _is_close_choice_response(self, message: str) -> bool:
        text = (message or "").strip().lower()
        normalized = text.replace("。", "").replace("！", "").replace("，", "").strip()
        return normalized in {"关闭", "完成", "结束"}

    def _is_acknowledgement_only(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        normalized = re.sub(r"[\s。！!，,\.．～~]+", "", text)
        return normalized in {
            "好",
            "好的",
            "好滴",
            "ok",
            "okay",
            "嗯",
            "嗯嗯",
            "知道了",
            "明白",
            "明白了",
            "收到",
            "了解",
            "了解了",
            "可以",
            "行",
            "谢谢",
            "感谢",
        }

    def _build_acknowledgement_reply(self) -> str:
        record = self.last_completed_task or {}
        response = str(record.get("response") or "").strip()
        if not response:
            return "好的。"

        if record.get("agent") == "dataprocessor" and any(
            keyword in response for keyword in ["后台启动", "后台运行", "请查看日志"]
        ):
            return "好的，数据处理任务已经在后台运行了。后续可以查看日志确认进度。"

        return "好的。"

    def _is_explanatory_followup(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        keywords = [
            "有什么区别",
            "区别",
            "怎么选",
            "推荐",
            "推荐哪个",
            "推荐哪种",
            "哪种更适合",
            "哪个好",
            "什么意思",
            "分别是什么",
        ]
        return any(keyword in text for keyword in keywords)

    def _looks_like_new_task(self, message: str) -> bool:
        text = (message or "").lower()
        task_keywords = [
            "训练", "lora", "全参", "增强训练", "定时训练",
            "评估", "评测", "ckpt",
            "推理", "服务",
            "数据处理", "预处理", "高级筛选", "数据准备", "数据收集",
            "监控", "分析", "停止", "结束",
        ]
        return any(keyword in text for keyword in task_keywords)

    def _trainer_need_input_response(
        self,
        message: str,
        required_params: List[str],
        *,
        kind: str = "training_params",
        options: Optional[List[str]] = None,
        display_message: Optional[str] = None,
        missing_params: Optional[List[str]] = None,
        original_task: Optional[str] = None,
        **fields: Any,
    ) -> ToolResponse:
        missing = missing_params or required_params
        protocol = self._with_protocol(
            "need_input",
            "trainer",
            message,
            kind=kind,
            title=self._need_input_title(kind),
            requiredParams=required_params,
            missingParams=missing,
            options=options or None,
            action="collect_params",
            status="needs_input",
            jobType="train",
            **fields,
        )
        self.last_response_protocol = protocol
        if original_task:
            self.pending_parameters["trainer"] = self._pending_state(
                "trainer",
                original_task,
                "param",
                f"[参数请求] {message}",
                display_message or message,
                protocol,
                required_params=required_params,
            )
        return ToolResponse(content=[TextBlock(type="text", text=display_message or message)],
            metadata={"success": False, "protocol": protocol, "protocol_hint": protocol},
        )


    async def _normalize_training_request(self, task_description: str) -> Optional[ToolResponse]:
        """在 trainer 前确定默认训练类型，避免 trainer 因未显式类型而追问。"""
        is_multinode = self._is_multinode_training_request(task_description)
        multinode_family = self._infer_multinode_train_family(task_description)
        requested_train_type = self._normalize_train_type(self._infer_train_type(task_description))
        has_training_type = self._has_training_type(task_description)
        if has_training_type and requested_train_type not in {None, "lora", "full", "enhanced"}:
            return None

        dataset_ref = self._extract_dataset_reference(task_description)
        if not dataset_ref:
            if is_multinode and not multinode_family:
                return self._trainer_need_input_response(
                    "你要启动多机训练，但我还不能判断是 SFT LoRA 还是 DPO 增强训练。请明确回复：多机lora批量训练 或 多机增强训练。",
                    ["training_type"],
                    kind="training_type",
                    options=["多机lora批量训练", "多机增强训练"],
                )

            return None

        container = self._training_container_for_request(task_description)
        batch_path = f"/home/workspace/dataset_batch_train/{dataset_ref}"
        daily_path = f"/home/workspace/dataset_daily_train/{dataset_ref}"
        raw_path = f"/home/workspace/dataset/{dataset_ref}"

        batch_exists = self._docker_path_exists(container, batch_path)
        daily_exists = self._docker_path_exists(container, daily_path)
        raw_exists = self._docker_path_exists(container, raw_path)
        named_values = self._extract_named_param_values(task_description)
        explicit_paths = [
            value.strip().rstrip("/")
            for key, value in named_values.items()
            if key in {"dataset_dir", "input_folder"} and value
        ]
        explicit_daily_path = any(
            path == daily_path or path.startswith(f"{daily_path}/")
            for path in explicit_paths
        )
        explicit_batch_path = any(
            path == batch_path or path.startswith(f"{batch_path}/")
            for path in explicit_paths
        )

        if (
            batch_exists is True
            and daily_exists is True
            and requested_train_type is None
            and not explicit_daily_path
            and not explicit_batch_path
        ):
            return self._trainer_need_input_response(
                (
                    "同一数据标识同时存在于 SFT 和 DPO 目录，无法安全判断训练类型："
                    f"{batch_path}；{daily_path}。"
                    "请明确回复：lora训练 或 增强训练。"
                ),
                ["training_type"],
                kind="training_type",
                options=["lora训练", "增强训练"],
            )

        if (
            batch_exists is True
            and requested_train_type in {None, "lora", "full"}
            and not explicit_daily_path
        ):
            if requested_train_type == "full":
                mode_text = "全参批量训练"
            else:
                mode_text = "多机lora批量训练" if is_multinode else "lora批量训练"
            normalized_task = (
                f"执行{mode_text}。用户指定数据标识为{dataset_ref}，"
                f"对应SFT数据目录为{batch_path}。不要询问训练类型，立即执行{mode_text}。"
                "调用 run_script_by_name_train 时 additional_args 只能使用 "
                f"data_identifier={dataset_ref}、data_dir={batch_path}"
                f"{'、model_path=' + named_values['model_path'] if named_values.get('model_path') else ''}，禁止使用 data_id。"
            )
            if is_multinode:
                multinode_args = self._extract_multinode_cli_args(task_description)
                for key, value in sorted(multinode_args.items()):
                    normalized_task += f"\n{key}={value}"
            logger.info(f"[Orchestrator] 训练请求已归一化为 LoRA: {dataset_ref}")
            return await self._call_agent_with_params("trainer", normalized_task, "")

        if daily_exists is True:
            candidates = self._list_dataset_candidates_in_container(container, daily_path)
            provided_dataset_name = (named_values.get("dataset_name") or "").strip()
            if provided_dataset_name:
                dataset_name = provided_dataset_name
            else:
                dataset_name = self._choose_dataset_name(candidates)
            dataset_dir = named_values.get("dataset_dir") or daily_path
            known_params = {
                key: value
                for key, value in {
                    "model_path": named_values.get("model_path"),
                    "dataset_dir": dataset_dir,
                    "dataset_name": dataset_name,
                }.items()
                if value
            }
            if not dataset_name:
                message = (
                    f"检测到DPO数据目录 {dataset_dir}，默认应执行增强训练，"
                    "但未能唯一推断 dataset_name。"
                    f"当前候选有：{', '.join(candidates[:8]) if candidates else '无'}。"
                    "请明确告诉我要使用哪个数据集名称。"
                )
                return self._trainer_need_input_response(
                    message,
                    ["model_path", "dataset_dir", "dataset_name"],
                    display_message=message,
                    missing_params=["dataset_name"],
                    options=candidates[:30] or None,
                    original_task=task_description,
                    knownParams=known_params,
                    container=container,
                    trainType="enhanced",
                    trainTypeText="增强训练",
                )
            default_base_model_path = named_values.get("model_path") or self._find_default_base_model_path(container)
            if not default_base_model_path:
                message = (
                    "检测到DPO数据并已确认 dataset_name，默认应执行增强训练，"
                    "但未能从 /home/workspace/models/base 自动发现基础模型。"
                    "请明确提供 model_path，例如 model_path=/home/workspace/models/base/<模型目录名>。"
                )
                return self._trainer_need_input_response(
                    message,
                    ["model_path", "dataset_dir", "dataset_name"],
                    display_message=message,
                    missing_params=["model_path"],
                    original_task=task_description,
                    knownParams=known_params,
                    container=container,
                    trainType="enhanced",
                    trainTypeText="增强训练",
                )
            mode_text = "多机增强训练" if is_multinode else "增强训练"
            normalized_task = (
                f"执行{mode_text}。不要询问训练类型，立即使用以下参数调用增强训练工具：\n"
                f"model_path={default_base_model_path}\n"
                f"dataset_dir={dataset_dir}\n"
                f"dataset_name={dataset_name}"
            )
            if is_multinode:
                multinode_args = self._extract_multinode_cli_args(task_description)
                for key, value in sorted(multinode_args.items()):
                    if key not in {"model_path", "dataset_dir", "dataset_name"}:
                        normalized_task += f"\n{key}={value}"
            logger.info(f"[Orchestrator] 训练请求已归一化为 DPO增强训练: {dataset_ref}/{dataset_name}")
            return await self._call_agent_with_params("trainer", normalized_task, "")

        if raw_exists is True:
            logger.info(f"[Orchestrator] 训练请求转为预处理引导: {dataset_ref}")
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=self._build_preprocess_guidance(dataset_ref, raw_path),
                )
            ])

        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=(
                    f"未检测到数据 {dataset_ref} 对应的可训练数据目录。请确认数据位于 "
                    f"{batch_path}、{daily_path} 或 {raw_path}。"
                ),
            )
        ])
    
    async def _call_trainer(self, task_description: str, **kwargs) -> ToolResponse:
        """调用trainer"""
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        # Keep the user-visible task separate from hints added below.  If trainer
        # asks for more input, only this clean task is allowed into pending state.
        user_task_description = task_description

        lowered_task = task_description.lower()
        if "定时训练" in lowered_task:
            normalized_schedule_time = self._normalize_schedule_time_input(task_description)
            current_schedule_time = self._extract_schedule_time_value(task_description)
            if normalized_schedule_time and not current_schedule_time:
                task_description = f"{task_description}\nschedule_time={normalized_schedule_time}"
                lowered_task = task_description.lower()
            elif current_schedule_time is None:
                task_description = (
                    f"{task_description}\n"
                    "[系统补充] 用户没有提供任何定时时间。定时训练的 schedule_time 是可选参数；"
                    "调用 run_script_by_name_train 时 additional_args 必须为空，禁止自行传入 schedule_time，"
                    "并且 background 必须为 true。"
                )
                lowered_task = task_description.lower()

        is_multinode = self._is_multinode_training_request(task_description)
        multinode_family = self._infer_multinode_train_family(task_description)
        request_container = self._training_container_for_request(task_description)
        if "增强训练" in task_description or "dpo" in lowered_task or multinode_family == "enhanced":
            named_values = self._extract_named_param_values(task_description)
            normalized_values = self._normalize_dpo_dataset_params(named_values, container=request_container)
            for key in ("dataset_dir", "dataset_name"):
                value = normalized_values.get(key)
                if value and named_values.get(key) != value and f"{key}={value}" not in task_description:
                    task_description = f"{task_description}\n{key}={value}"
            named_values = {**named_values, **normalized_values}
            dataset_dir = named_values.get("dataset_dir")
            if not dataset_dir:
                dataset_ref = self._extract_dataset_reference(task_description)
                if dataset_ref:
                    candidate_dir = f"/home/workspace/dataset_daily_train/{dataset_ref}"
                    if self._docker_path_exists(request_container, candidate_dir) is True:
                        dataset_dir = candidate_dir
                        named_values["dataset_dir"] = dataset_dir
                        task_description = f"{task_description}\ndataset_dir={dataset_dir}"

            if dataset_dir:
                inferred_dataset_name = None
                if "dataset_name" not in named_values:
                    inferred_dataset_name = self._infer_dataset_name_from_dir(dataset_dir, container=request_container)
                    if inferred_dataset_name and f"dataset_name={inferred_dataset_name}" not in task_description:
                        task_description = f"{task_description}\ndataset_name={inferred_dataset_name}"
                dataset_name = named_values.get("dataset_name") or inferred_dataset_name
                if dataset_name and "model_path" not in named_values:
                    default_model_path = self._find_default_base_model_path(request_container)
                    if default_model_path:
                        named_values["model_path"] = default_model_path
                        task_description = f"{task_description}\nmodel_path={default_model_path}"
                task_description = (
                    f"{task_description}\n"
                    "[系统补充] 用户当前是在补充增强训练参数，并且已经提供了真实的 dataset_dir。\n"
                    "如果 model_path、dataset_dir、dataset_name 都已提供，必须直接调用增强训练工具，不要再向用户确认。"
                    "如果 dataset_name 仍缺失，只追问 dataset_name。"
                )
            if is_multinode and multinode_family == "enhanced":
                current_values = self._extract_named_param_values(task_description)
                missing = [
                    param
                    for param in ("model_path", "dataset_dir", "dataset_name")
                    if not current_values.get(param)
                ]
                if missing:
                    required = ["model_path", "dataset_dir", "dataset_name"]
                    mode_text = "多机增强训练"
                    dataset_name_options = None
                    if missing == ["dataset_name"] and dataset_dir:
                        dataset_name_options = self._list_dataset_candidates_in_container(
                            request_container,
                            dataset_dir,
                        )[:30]
                    param_descriptions = {
                        "model_path": "本次训练要使用的基础模型路径",
                        "dataset_dir": "训练数据所在的目录路径",
                        "dataset_name": "训练时要使用的数据集名称",
                    }
                    missing_lines = "\n".join(
                        f"- `{param}`：{param_descriptions[param]}。"
                        for param in missing
                    )
                    example = "，".join(f"{param}=<{'模型路径' if param == 'model_path' else '数据目录路径' if param == 'dataset_dir' else '数据集名称'}>" for param in missing)
                    message = (
                        f"开始{mode_text}前，我还需要先确认"
                        f"{'以下信息' if len(missing) > 1 else '这个信息'}。\n"
                        f"{missing_lines}\n"
                        "如果你不确定，也可以直接把相关目录或名称发给我，我再帮你一起确认。\n"
                        f"你可以直接回复，例如：`{example}`。"
                    )
                    response = self._trainer_need_input_response(
                        message,
                        required,
                        display_message=self._wrap_user_waiting_message("参数补充", message),
                        missing_params=missing,
                        options=dataset_name_options,
                        original_task=user_task_description,
                        knownParams=current_values,
                        container=request_container,
                        launchMode="multinode",
                        isMultinode=True,
                        trainType="enhanced",
                        trainTypeText="多机增强训练",
                    )
                    trainer = self.agents["trainer"]
                    trainer_msg = self._agent_msg(
                        trainer.name,
                        f"[参数请求] {message}",
                        response.metadata["protocol"],
                    )
                    await trainer.print(trainer_msg, True)
                    await trainer.memory.add(trainer_msg)
                    response.metadata["response_msg"] = trainer_msg
                    return response

        normalized_response = await self._normalize_training_request(task_description)
        if normalized_response is not None:
            return normalized_response

        data_type = self.task_context.get('data_type')
        if data_type:
            if data_type == 'sft':
                constraint = "\n[系统约束] 当前数据处理类型为 SFT，**仅允许执行 Lora 批量训练 或 全参批量训练**"
            elif data_type == 'dpo':
                constraint = "\n[系统约束] 当前数据处理类型为 DPO，**仅允许执行 定时训练 或 增强训练**"
            else:
                constraint = ""

            if constraint:
                task_description = f"{task_description}\n{constraint}"
                logger.info(f"[Orchestrator] 训练约束已附加: {data_type}")

        task_description = f"{task_description}\n{self._build_trainer_system_hints(task_description)}"

        return await self._call_agent_with_params(
            "trainer",
            task_description,
            "",
            pending_original_task=user_task_description,
        )
    
    async def _call_evaluator(self, task_description: str, **kwargs) -> ToolResponse:
        """调用evaluator"""
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        additional_params = ""
        if "evaluator" in self.pending_parameters:
            original_task = self.pending_parameters["evaluator"]["original_task"]
            additional_params = task_description.replace(original_task, "").strip()
        return await self._call_agent_with_params("evaluator", task_description, additional_params)

    def _inference_failure_protocol(self, command: str, error: str) -> Dict[str, Any]:
        action = self._infer_inference_action(command, error)
        if action.startswith("benchmark"):
            job_type = "inference_benchmark"
        elif action.startswith("test"):
            job_type = "inference_test"
        else:
            job_type = "inference_service"
        return self._with_protocol(
            "job_failed",
            "inference",
            error,
            jobType=job_type,
            action=action,
            status="failed",
            errorReason=error,
        )

    def _request_inference_agent(self, command: str) -> Dict[str, Any]:
        command = str(command or self.current_user_message or "").strip()
        resource_context: Optional[Dict[str, Any]] = None
        try:

            resource_context = _inference_request_resource_context(command)
            owner_user_id, owner_aliases = _inference_owner_payload(self.user_id)
            request_payload: Dict[str, Any] = {
                "command": command,
                "user_id": owner_user_id,
                "user_aliases": owner_aliases,
                "user_role": _current_user_role(),
                "thread_id": f"user:{self.user_id}:inference",
                "container": self.evaluation_container,
            }
            if resource_context:
                request_payload["resource_context"] = resource_context
            response = requests.post(
                config.agents.inference.inference_agent_url,
                json=request_payload,
                timeout=360,
            )
            response.raise_for_status()
            response_data = response.json()
            if not isinstance(response_data, dict):
                raise ValueError("推理服务返回格式不是 JSON 对象")
            result = str(response_data.get("result") or "")
            if not result:
                raise ValueError("推理服务未返回 result")
            data = response_data.get("data") or {}
            status_value = str(response_data.get("status") or "").lower()
            stop_response_success = _inference_stop_response_is_success(command, response_data)
            protocol = self._inference_protocol_for_result(command, result, data)
            logger.info(
                "[inference-protocol] command=%s data_keys=%s protocol=%s",
                command,
                sorted(data.keys()) if isinstance(data, dict) else [],
                json.dumps(protocol, ensure_ascii=False),
            )
            resource_payload = dict(response_data)
            resource_payload["protocol"] = protocol
            if isinstance(resource_context, dict) and resource_context.get("reservation_id"):
                if _inference_response_holds_resource_reservation(command, resource_payload, result, resource_context):
                    _start_inference_resource_heartbeat(resource_context)
                else:
                    _release_inference_resource_context(resource_context)
            if _inference_protocol_is_service_stop(protocol):
                _release_known_inference_reservations()
            success = stop_response_success or status_value not in {"error", "failed", "timeout"}
        except Exception as exc:
            _release_inference_resource_context(resource_context)
            logger.exception("Inference agent command failed: %s", command)
            result = f"推理服务调用失败：{exc}"
            data = {}
            protocol = self._inference_failure_protocol(command, result)
            success = False
        return {
            "success": success,
            "command": command,
            "result": result,
            "data": data,
            "protocol": protocol,
            "resource_context": resource_context,
        }
    def _run_inference_agent_command(self, command: str) -> ToolResponse:
        """调用推理服务。inference agent 必须通过此工具执行推理相关查询/操作。"""
        payload = self._request_inference_agent(command)
        payload = {
            "__inference_tool_result__": True,
            **payload,
        }
        self._last_inference_tool_payload = payload
        return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
            metadata={
                "success": payload["success"],
                "protocol": payload["protocol"],
            },
        )
    
    async def _call_inference(self, task_description: str, **kwargs) -> ToolResponse:
        """调用推理 Agent 处理推理服务、节点运维、功能测试和推理基准评测请求。

适用场景：
- 查看、修改推理配置，例如模型路径、模型名称、端口、GPU、TP、max tokens 等。
- 查看、启动、停止、重启推理服务，并查询启动状态、服务状态和服务日志。
- 查看 GPU 状态、推荐 GPU 分配、根据当前资源推荐在哪个节点启动推理服务。
- 多节点操作：查看节点列表，查看/修改指定节点配置，查看指定节点服务状态、日志、GPU、测试和 benchmark。
- 启用/禁用 worker 节点；controller 节点不能禁用。
- 查看可用的推理功能测试脚本，运行推理功能测试脚本（basicmedicalrecord.sh 等），并查询测试状态、运行列表和测试结果。
- 查看可用的推理 benchmark，运行/停止推理 benchmark，并查询任务列表、进度、状态和结果报告。
- 支持医疗选择题（2021.json、2024.json、step1.json、step2.json、step3.json 等）、MedBench、通用 benchmark 等推理评测相关请求。
"""

        explicit_command = kwargs.pop("command", None)
        command = str(explicit_command or task_description or self.current_user_message).strip()
        #if kwargs:
        #    params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        #    command = f"{command} {params_str}".strip()
        #    logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        #task_for_agent = (
        #    f"{command}\n\n"
        #    "[系统约束] 你必须调用 `_run_inference_agent_command` 工具执行上述推理任务，"
        #    f"command 参数必须精确等于：{command}。等待工具返回后，只根据工具返回结果回复；"
        #    "不要绕过工具直接回答，不要自行编造配置、状态或结果。"
        #)
        #self._last_inference_tool_payload = None
        #agent_response = await self._call_agent_with_params("inference", task_for_agent, "")
        #agent_text = self._response_to_text(agent_response)
        #tool_payload = self._last_inference_tool_payload or #_inference_tool_payload_from_text(agent_text)
        #if not tool_payload:
        #    tool_payload = _inference_tool_payload_from_text(str(self.#last_completed_task.get("response") or "") if self.last_completed_task else "")
        #if not tool_payload:
        #    return agent_response

        payload = self._request_inference_agent(command)
        result = payload["result"]
        protocol = payload["protocol"]
        success = payload["success"]


        self.last_completed_task = {
            "agent": "inference",
            "task": command,
            "params": "",
            "response": result,
            "timestamp": asyncio.get_event_loop().time(),
        }
        self.last_response_protocol = protocol
        self.current_task_state = {"status": "idle"}

        return ToolResponse(content=[TextBlock(type="text", text=f"inference 执行结果：\n{result}")],
            metadata={
                "success": success,
                "protocol": protocol,
                "inference_payload": payload,
                "response_msg": self._agent_msg(
                    self.agents["inference"].name if "inference" in self.agents else "inference",
                    result,
                    protocol,
                ),
            },
        )
    
    async def _call_monitor(self, task_description: str, **kwargs) -> ToolResponse:
        """调用monitor"""
        # Monitoring is a stateless snapshot query. Keeping prior polling
        # tool results makes the prompt grow on every refresh and can exceed
        # the model context window.
        monitor_agent = self.agents.get("monitor")
        if monitor_agent is not None:
            await monitor_agent.memory.clear()
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")
        # 智能路由：推理服务相关查询转发给 inference agent
        inference_keywords = ["推理服务", "service_status", "查看推理", "推理配置", "inference服务", "推理端口"]
        if any(kw in task_description for kw in inference_keywords):
            logger.info(f"[Orchestrator] 检测到推理服务查询，自动转发到 inference agent: {task_description[:50]}...")
            return await self._call_inference(task_description, **kwargs)

        if self._is_direct_assessment_monitor_request(task_description):
            logger.info(f"[Orchestrator] 检测到明确的评估监控意图，直接调用监控工具: {task_description[:50]}...")
            return self._direct_monitor_tool_response(
                task_description,
                run_group_assessment_monitor(task_description),
            )
        if self._is_direct_training_monitor_request(task_description):
            logger.info(f"[Orchestrator] 检测到明确的训练监控意图，直接调用监控工具: {task_description[:50]}...")
            return self._direct_monitor_tool_response(
                task_description,
                run_script_by_name_monitor1(task_description),
            )

        additional_params = ""
        if "monitor" in self.pending_parameters:
            original_task = self.pending_parameters["monitor"]["original_task"]
            additional_params = task_description.replace(original_task, "").strip()
        return await self._call_agent_with_params("monitor", task_description, additional_params)

    async def _call_analysis(self, task_description: str, **kwargs) -> ToolResponse:
        """调用analysis"""
        # 将LLM传入的额外参数追加到task_description
        if kwargs:
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()
            logger.debug(f"[Orchestrator] 接收到额外参数: {kwargs}")

        additional_params = ""
        if "analysis" in self.pending_parameters:
            original_task = self.pending_parameters["analysis"]["original_task"]
            additional_params = task_description.replace(original_task, "").strip()
        return await self._call_agent_with_params("analysis", task_description, additional_params)

    async def _stop_task(
        self,
        task_description: str = "",
        task_type: Optional[str] = None,
        pid: Optional[str] = None,
        container: Optional[str] = None,
        **kwargs,
    ) -> ToolResponse:
        """结束训练、模型评估、数据处理或推理任务。"""
        if kwargs:
            task_type = task_type or kwargs.get("task_type")
            pid = pid or kwargs.get("pid")
            container = container or kwargs.get("container") or kwargs.get("container_name") or kwargs.get("docker")
            params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            task_description = f"{task_description} {params_str}".strip()

        extracted_target = self._extract_stop_target_params(task_description)
        task_type = task_type or extracted_target.get("task_type")
        pid = pid or extracted_target.get("pid")
        container = container or extracted_target.get("container")
        if not task_type and extracted_target.get("has_stop_intent"):
            task_type = self._infer_recent_stop_task_type()
        task_type = self._normalize_stop_task_type(task_type)

        inference_keywords = ["2021", "2024", "step1", "step2", "step3", "benchmark", "medbench", "Medbench", "MedBench", "推理", "基准"]
        if task_type == "inference" or any(kw in task_description for kw in inference_keywords):
            logger.info(f"[Orchestrator] 检测到推理终止任务，自动转发到 inference agent: {task_description[:50]}...")
            inference_stop_description = task_description.strip() or "停止推理基准测试"
            if "停止" not in inference_stop_description and "关闭" not in inference_stop_description and "stop" not in inference_stop_description.lower():
                inference_stop_description = f"停止{inference_stop_description}"
            if pid and str(pid) not in inference_stop_description:
                inference_stop_description = f"{inference_stop_description} PID {pid}"
            return await self._call_inference(inference_stop_description)

        result = await self._handle_stop_command(
            task_description,
            task_type=task_type,
            pid=pid,
            container=container,
        )
        tool_text = result or "未识别到可终止的任务类型。"
        protocol_task_type = self._task_type_from_stop_result(tool_text) or task_type
        user_text = self._friendly_stop_result_text(tool_text)
        stop_status_fields = self._stop_status_protocol_fields(tool_text)
        protocol = self._with_protocol(
            "job_stopped",
            "orchestrator",
            user_text,
            jobType=protocol_task_type,
            container=container or self._extract_container(tool_text),
            pid=pid or self._extract_pid(tool_text),
            **stop_status_fields,
        )
        self.last_response_protocol = protocol
        return ToolResponse(content=[TextBlock(type="text", text=tool_text)],
            metadata={
                "success": True,
                "protocol": protocol,
                "response_msg": self._agent_msg(
                    self.orchestrator.name if self.orchestrator else "Orchestrator",
                    user_text,
                    protocol,
                ),
            },
        )

    def _parse_stop_command(self, message: str) -> Optional[Dict[str, Any]]:
        """识别明确的结束类指令，避免依赖 orchestrator sys_prompt 路由。"""
        extracted = self._extract_stop_target_params(message)
        if not extracted.get("has_stop_intent"):
            return None

        task_type = extracted.get("task_type")
        if not task_type:
            return None

        return {
            "task_type": task_type,
            "pid": extracted.get("pid"),
            "container": extracted.get("container"),
        }

    def _extract_stop_target_params(self, text: str) -> Dict[str, Any]:
        """Extract explicit stop target fields from a user/tool description."""
        normalized_text = (text or "").strip()
        lowered = normalized_text.lower()
        result: Dict[str, Any] = {
            "task_type": None,
            "pid": None,
            "container": None,
            "has_stop_intent": False,
        }
        if not normalized_text:
            return result

        stop_keywords = ["结束", "停止", "终止", "中止", "取消", "关闭", "kill", "stop", "terminate"]
        result["has_stop_intent"] = any(keyword.lower() in lowered for keyword in stop_keywords)

        pid_match = re.search(
            r"(?:pid|PID|进程ID|进程Id|进程id|进程号)\s*(?:是|为|:|：|=)?\s*`?(\d+)`?",
            normalized_text,
            re.IGNORECASE,
        )
        if pid_match:
            result["pid"] = pid_match.group(1)

        container_patterns = [
            r"(?:container_name|container|docker|容器名称|容器名|容器)\s*(?:是|为|:|：|=)\s*`?([A-Za-z0-9_.-]+)`?",
            r"(?:container_name|container|docker|容器名称|容器名|容器)\s+`?([A-Za-z0-9_.-]+)`?\s*(?:里|内|中的|里面|内的)?",
            r"(?:停止|终止|结束|中止|取消|关闭|kill|stop|terminate)\s+([A-Za-z0-9_.-]+)\s*(?:里|内|中的|里面|内的)",
            r"([A-Za-z0-9_.-]+)\s*(?:里|内|中的|里面|内的)\s*(?:训练|评测|评估|数据处理|预处理|高级筛选|train|evaluate|evaluation|eval|data)",
        ]
        container_match = None
        for pattern in container_patterns:
            container_match = re.search(pattern, normalized_text, re.IGNORECASE)
            if container_match:
                break
        if container_match:
            result["container"] = container_match.group(1)

        task_text = normalized_text
        if result.get("container"):
            task_text = re.sub(re.escape(str(result["container"])), " ", task_text, flags=re.IGNORECASE)
        if result.get("pid"):
            task_text = re.sub(re.escape(str(result["pid"])), " ", task_text)
        task_lowered = task_text.lower()

        if any(keyword in task_text for keyword in ["推理", "基准"]) or any(
            keyword in task_lowered for keyword in ["inference", "benchmark", "medbench"]
        ):
            result["task_type"] = "inference"
        elif any(keyword in task_text for keyword in ["训练", "模型训练"]) or any(
            keyword in task_lowered for keyword in ["train", "training", "trainer"]
        ):
            result["task_type"] = "train"
        elif any(keyword in task_text for keyword in ["评测", "评估"]) or any(
            keyword in task_lowered for keyword in ["evaluate", "evaluation", "eval", "evaluator"]
        ):
            result["task_type"] = "assessment"
        elif any(keyword in task_text for keyword in ["数据处理", "预处理", "高级筛选"]) or any(
            keyword in task_lowered for keyword in ["data", "dataprocessor", "data_processor", "data-processing"]
        ):
            result["task_type"] = "data"

        return result

    def _normalize_stop_task_type(self, task_type: Optional[str]) -> Optional[str]:
        if not task_type:
            return None

        normalized = str(task_type).strip().lower()
        aliases = {
            "train": "train",
            "training": "train",
            "trainer": "train",
            "模型训练": "train",
            "训练": "train",
            "evaluate": "assessment",
            "evaluation": "assessment",
            "assessment": "assessment",
            "assessor": "assessment",
            "eval": "assessment",
            "evaluator": "assessment",
            "评测": "assessment",
            "评估": "assessment",
            "data": "data",
            "dataprocessor": "data",
            "data_processor": "data",
            "data-processing": "data",
            "数据处理": "data",
            "预处理": "data",
            "高级筛选": "data",
            "inference": "inference",
            "infer": "inference",
            "推理": "inference",
            "推理服务": "inference",
            "推理基准": "inference",
            "推理基准测试": "inference",
            "benchmark": "inference",
            "medbench": "inference",
            "基准": "inference",
            "基准测试": "inference",
        }
        return aliases.get(normalized, normalized)

    def _all_stop_patterns_for_type(self, task_type: str) -> List[str]:
        if task_type == "train":
            return ["batch_train_lora", "batch_train_full", "dpo_train_launcher", "grpo_train"]
        if task_type == "data":
            return ["data_preprocessing", "score_based_filtering"]
        if task_type in {"assessment", "evaluate"}:
            return ["compare_between_models_vpn", "single_model_evaluation_vpn", "ckpt_eval"]
        return []

    def _task_type_for_script_name(self, script_name: Optional[str]) -> Optional[str]:
        name = os.path.basename(str(script_name or "").strip())
        if name in {"batch_train_lora", "batch_train_full", "dpo_train_launcher", "grpo_train"}:
            return "train"
        if name in {"data_preprocessing", "score_based_filtering"}:
            return "data"
        if name in {"compare_between_models_vpn", "single_model_evaluation_vpn", "ckpt_eval"}:
            return "assessment"
        return None

    def _task_type_from_script_text(self, text: Optional[str]) -> Optional[str]:
        source = str(text or "")
        for task_type in ("assessment", "train", "data"):
            for pattern in self._all_stop_patterns_for_type(task_type):
                if pattern in source:
                    return task_type
        return None

    def _task_type_from_process_lines(self, lines: List[str]) -> Optional[str]:
        for line in lines:
            task_type = self._task_type_from_script_text(line)
            if task_type:
                return task_type
        return None

    def _background_task_record_by_pid(
        self,
        container: Optional[str],
        pid: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        normalized_pid = str(pid or "").strip()
        normalized_container = str(container or "").strip()
        if not normalized_pid:
            return None

        matched = None
        for record in self._load_background_task_records():
            if str(record.get("pid") or "").strip() != normalized_pid:
                continue
            if normalized_container and str(record.get("container") or "").strip() != normalized_container:
                continue
            matched = record
        return matched

    def _task_type_from_background_record(self, record: Optional[Dict[str, Any]]) -> Optional[str]:
        if not record:
            return None
        script_task_type = self._task_type_from_script_text(
            " ".join(
                str(record.get(key) or "")
                for key in ("script_name", "script_path", "command")
            )
        )
        if script_task_type:
            return script_task_type
        return self._normalize_stop_task_type(record.get("task_type"))

    def _infer_stop_task_type_from_pid(
        self,
        container: Optional[str],
        pid: Optional[str],
    ) -> Optional[str]:
        if not container or not pid:
            return None
        process_task_type = self._task_type_from_process_lines(
            self._check_docker_pid_processes(container, pid)
        )
        if process_task_type:
            return process_task_type
        return self._task_type_from_background_record(
            self._background_task_record_by_pid(container, pid)
        )

    def _task_type_from_stop_result(self, result: str) -> Optional[str]:
        text = result or ""
        if "已发送结束模型评估指令" in text:
            return "assessment"
        if "已发送结束训练指令" in text:
            return "train"
        if "已发送结束数据处理指令" in text:
            return "data"
        return self._task_type_from_script_text(text)

    def _load_background_task_records(self) -> List[Dict[str, Any]]:
        if not BACKGROUND_TASK_REGISTRY.exists():
            return []

        records = []
        with BACKGROUND_TASK_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record.get("container"):
                    continue
                if not record.get("task_type"):
                    record["task_type"] = self._task_type_for_script_name(record.get("script_name"))
                record["task_type"] = self._normalize_stop_task_type(record.get("task_type"))
                if record.get("task_type") in {"train", "data", "assessment"}:
                    records.append(record)
        return records

    def _latest_background_task_record(
        self,
        task_type: str,
        preferred_patterns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        preferred = {os.path.basename(pattern) for pattern in (preferred_patterns or [])}
        records = [
            record for record in self._load_background_task_records()
            if record.get("task_type") == task_type
        ]
        def record_started_at(record: Dict[str, Any]) -> float:
            try:
                return float(record.get("started_at") or 0)
            except (TypeError, ValueError):
                return 0

        records.sort(key=record_started_at, reverse=True)
        if not records:
            return None

        if preferred:
            for record in records:
                script_name = os.path.basename(str(record.get("script_name") or record.get("script_path") or ""))
                command = str(record.get("command") or "")
                if script_name in preferred or any(pattern in command for pattern in preferred):
                    return record
        return records[0]

    def _background_resource_records_for_target(
        self,
        task_types: Set[str],
        container: Optional[str],
        pid: Optional[str] = None,
        patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find GPU reservation records belonging to stopped jobs in one container."""
        normalized_container = str(container or "").strip()
        normalized_pid = str(pid or "").strip()
        if not normalized_container:
            return []
        normalized_patterns = [str(pattern).strip() for pattern in (patterns or []) if str(pattern).strip()]
        records = [
            record
            for record in self._load_background_task_records()
            if record.get("task_type") in task_types
            and str(record.get("container") or "").strip() == normalized_container
            and (
                (normalized_pid and str(record.get("pid") or "").strip() == normalized_pid)
                or (
                    not normalized_pid
                    and normalized_patterns
                    and any(
                        pattern in str(record.get("script_name") or record.get("script_path") or record.get("command") or "")
                        for pattern in normalized_patterns
                    )
                )
            )
        ]
        records.sort(
            key=lambda record: float(record.get("started_at") or 0),
            reverse=True,
        )
        unique_records = []
        reservation_ids = set()
        for record in records:
            env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
            reservation_id = str(env_vars.get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip()
            if reservation_id and reservation_id not in reservation_ids:
                reservation_ids.add(reservation_id)
                unique_records.append(record)
        return unique_records

    def _release_stopped_gpu_reservations(
        self,
        task_types: Set[str],
        container: Optional[str],
        pid: Optional[str] = None,
        fallback_patterns: Optional[List[str]] = None,
    ) -> int:
        """Release quotas immediately after stopped GPU jobs have no live processes."""
        if not container:
            return 0
        if pid and self._check_docker_pid_processes(container, pid):
            return 0
        if fallback_patterns and self._check_docker_pattern_processes(container, fallback_patterns):
            return 0
        records = self._background_resource_records_for_target(
            task_types,
            container,
            pid=pid,
            patterns=fallback_patterns,
        )
        released_count = 0
        for record in records:
            record_pid = str(record.get("pid") or "").strip()
            if record_pid and self._check_docker_pid_processes(container, record_pid):
                continue
            env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
            reservation_id = str(env_vars.get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip()
            if not _release_resource_allocation(reservation_id, env_vars):
                continue
            released_count += 1
            logger.info(
                "Released stopped GPU reservation immediately: reservationId=%s container=%s pid=%s",
                reservation_id,
                container,
                record_pid or pid,
            )
        return released_count

    def _background_record_age_seconds(self, record: Dict[str, Any]) -> Optional[float]:
        try:
            started_at = float(record.get("started_at"))
        except (TypeError, ValueError):
            return None
        return max(0.0, time.time() - started_at)

    def _patterns_from_background_record(
        self,
        record: Dict[str, Any],
        fallback_patterns: List[str],
    ) -> List[str]:
        all_patterns = self._all_stop_patterns_for_type(str(record.get("task_type") or ""))
        script_name = os.path.basename(str(record.get("script_name") or record.get("script_path") or ""))
        if script_name in all_patterns:
            merged = [script_name]
            for pattern in fallback_patterns:
                if pattern not in merged:
                    merged.append(pattern)
            return merged
        command = str(record.get("command") or "")
        matched = [pattern for pattern in all_patterns if pattern in command]
        return matched or fallback_patterns

    def _list_running_docker_containers(self) -> List[str]:
        try:
            process = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return []
        if process.returncode != 0:
            return []
        return [line.strip() for line in process.stdout.splitlines() if line.strip()]

    def _infer_container_by_running_patterns(self, patterns: List[str]) -> Optional[Dict[str, Any]]:
        matches = []
        for container in self._list_running_docker_containers():
            matched_patterns = []
            matched_processes = []
            for pattern in patterns:
                processes = self._check_docker_pattern_processes(container, [pattern])
                if processes:
                    matched_patterns.append(pattern)
                    matched_processes.extend(processes)
            if matched_patterns:
                matches.append({
                    "container": container,
                    "patterns": matched_patterns,
                    "processes": matched_processes,
                })
        return matches[0] if len(matches) == 1 else None

    def _resolve_stop_container_and_patterns(
        self,
        task_type: str,
        patterns: List[str],
        explicit_container: Optional[str],
    ) -> Dict[str, Any]:
        if explicit_container:
            return {"container": explicit_container, "patterns": patterns, "source": "explicit"}

        all_patterns = self._all_stop_patterns_for_type(task_type)
        latest_record = self._latest_background_task_record(task_type)
        if latest_record:
            record_patterns = self._patterns_from_background_record(latest_record, patterns)
            running = self._check_docker_pattern_processes(latest_record["container"], record_patterns)
            age = self._background_record_age_seconds(latest_record)
            if running or age is None or age <= 24 * 60 * 60:
                return {
                    "container": latest_record["container"],
                    "patterns": record_patterns,
                    "source": "latest_background_task",
                }

        record = self._latest_background_task_record(task_type, patterns)
        if record:
            record_patterns = self._patterns_from_background_record(record, patterns)
            running = self._check_docker_pattern_processes(record["container"], record_patterns)
            if running:
                return {
                    "container": record["container"],
                    "patterns": record_patterns,
                    "source": "recent_background_task",
                }

        inferred = self._infer_container_by_running_patterns(patterns)
        if inferred:
            return {
                "container": inferred["container"],
                "patterns": inferred["patterns"],
                "source": "running_process",
            }

        if all_patterns and set(patterns) != set(all_patterns):
            inferred = self._infer_container_by_running_patterns(all_patterns)
            if inferred:
                return {
                    "container": inferred["container"],
                    "patterns": inferred["patterns"],
                    "source": "running_process",
                }

        return {"container": self.training_container, "patterns": patterns, "source": "default"}

    def _latest_train_pid_record(self, container: str) -> Optional[Dict[str, Any]]:
        if not TRAIN_PID_REGISTRY.exists():
            return None

        latest = None
        with TRAIN_PID_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("container") != container or not record.get("pid"):
                    continue
                if latest is None or str(record.get("started_at", "")) > str(latest.get("started_at", "")):
                    latest = record
        return latest

    def _train_pid_record_by_pid(self, container: str, pid: str) -> Optional[Dict[str, Any]]:
        if not TRAIN_PID_REGISTRY.exists():
            return None

        matched = None
        with TRAIN_PID_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("container") == container and str(record.get("pid")) == str(pid):
                    matched = record
        return matched

    def _latest_train_pid_record_any_container(self) -> Optional[Dict[str, Any]]:
        if not TRAIN_PID_REGISTRY.exists():
            return None

        latest = None
        with TRAIN_PID_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record.get("container") or not record.get("pid"):
                    continue
                if latest is None or str(record.get("started_at", "")) > str(latest.get("started_at", "")):
                    latest = record
        return latest

    def _train_pid_record_by_pid_any_container(self, pid: str) -> Optional[Dict[str, Any]]:
        if not TRAIN_PID_REGISTRY.exists():
            return None

        matched = None
        with TRAIN_PID_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("container") and str(record.get("pid")) == str(pid):
                    matched = record
        return matched

    def _train_record_fallback_patterns(self, record: Optional[Dict[str, Any]]) -> List[str]:
        """Build narrow fallback patterns for one specific training run."""
        if not record:
            return []

        patterns = []
        env_vars = record.get("env_vars") or {}
        script_args = record.get("script_args") or {}

        for value in (
            env_vars.get("DATASET_DIR"),
            env_vars.get("DATASET_DATE"),
            script_args.get("dataset_dir"),
        ):
            if value:
                patterns.append(str(value))

        dataset_date = env_vars.get("DATASET_DATE")
        if dataset_date:
            patterns.extend([
                f"/home/workspace/dataset_batch_train/{dataset_date}",
                f"/home/workspace/dataset_daily_train/{dataset_date}",
                f"model_medical_lora_{dataset_date}",
                f"model_medical_full_{dataset_date}",
            ])

        dataset_dir = env_vars.get("DATASET_DIR") or script_args.get("dataset_dir")
        if dataset_dir:
            dataset_tail = os.path.basename(os.path.normpath(str(dataset_dir)))
            if dataset_tail:
                patterns.extend([
                    str(dataset_dir),
                    f"model_medical_lora_{dataset_tail}",
                    f"model_medical_full_{dataset_tail}",
                ])

        # Keep only reasonably specific patterns; broad executable names can kill other runs.
        unique = []
        for pattern in patterns:
            pattern = str(pattern).strip()
            if not pattern or len(pattern) < 4:
                continue
            if pattern in {"deepspeed", "torchrun", "accelerate", "train", "batch_train_lora", "batch_train_full"}:
                continue
            if pattern not in unique:
                unique.append(pattern)
        return unique

    def _stop_multinode_training_record_processes(
        self,
        container: Optional[str],
        record: Optional[Dict[str, Any]],
        stop_output: str = "",
    ) -> str:
        if not container:
            return ""
        allocation_file = _training_allocation_file_from_record(record or {})
        if not allocation_file and stop_output:
            match = re.search(r"--allocation-file(?:=|\s+)(\S+)", stop_output)
            if match:
                allocation_file = match.group(1).strip("'\"")
        if not allocation_file:
            return ""
        allocation = _read_training_allocation(container, allocation_file)
        stop_result = _stop_multinode_allocation_processes(container, allocation)
        lines = [f"多机训练节点清理: allocation={allocation_file}"]
        remaining = stop_result.get("remainingGpuPids") or []
        for node in stop_result.get("nodeStopResults") or []:
            before = node.get("before") or []
            node_remaining = node.get("remaining") or []
            lines.append(
                "节点 {alias}({mode}) GPU[{gpus}] 清理前 {before_count} 个训练进程，残留 {remaining_count} 个".format(
                    alias=node.get("sshAlias") or node.get("nodeId") or "<unknown>",
                    mode=node.get("executionMode") or "unknown",
                    gpus=",".join(str(gpu) for gpu in (node.get("gpuIndexes") or [])),
                    before_count=len(before),
                    remaining_count=len(node_remaining),
                )
            )
        if remaining:
            remaining_pids = []
            for item in remaining:
                pid = str(item.get("pid") or "").strip()
                if pid and pid not in remaining_pids:
                    remaining_pids.append(pid)
            pid_summary = ", ".join(remaining_pids) or "; ".join(str(item) for item in remaining[:3])
            lines.append(f"仍发现残留 PID: {pid_summary}")
        else:
            lines.append("多机训练 GPU worker 已完成清理。")
        return "\n".join(lines)

    def _default_train_stop_patterns(self) -> List[str]:
        return [
            "train_multinode_sft_pipeline",
            "train_multinode_dpo_pipeline",
            "batch_train_lora",
            "batch_train_full",
            "dpo_train_launcher",
            "llamafactory-cli train",
            "src/train_bash.py",
            "deepspeed --include",
            "torchrun --nproc_per_node",
            "accelerate launch",
        ]

    def _resolve_train_stop_patterns(self, message: str) -> List[str]:
        text = (message or "").lower()
        if any(keyword in text for keyword in ["train_multinode_dpo_pipeline", "多机增强训练", "多机dpo"]):
            return ["train_multinode_dpo_pipeline"]
        if any(keyword in text for keyword in ["train_multinode_sft_pipeline", "多机lora批量训练", "多机sft"]):
            return ["train_multinode_sft_pipeline"]
        if any(keyword in text for keyword in ["增强训练", "dpo", "dpo_train_launcher"]):
            return ["dpo_train_launcher"]
        if any(keyword in text for keyword in ["lora批量训练", "lora训练", "batch_train_lora"]):
            return ["batch_train_lora"]
        if any(keyword in text for keyword in ["全参批量训练", "全参训练", "batch_train_full"]):
            return ["batch_train_full"]
        if any(keyword in text for keyword in ["grpo训练", "grpo_train", "grpo"]):
            return ["grpo_train"]
        if any(keyword in text for keyword in ["定时训练", "日常训练", "create_command_vpn"]):
            return ["create_command_vpn"]
        recent_patterns = self._recent_trainer_stop_patterns()
        if recent_patterns:
            return recent_patterns
        return self._default_train_stop_patterns()

    def _recent_trainer_stop_patterns(self) -> List[str]:
        record = self.last_completed_task or {}
        if record.get("agent") != "trainer":
            return []

        text = "\n".join(
            str(record.get(key) or "")
            for key in ("task", "params", "response")
        ).lower()
        if "train_multinode_dpo_pipeline" in text or "多机增强训练" in text or "多机dpo" in text:
            return ["train_multinode_dpo_pipeline"]
        if "train_multinode_sft_pipeline" in text or "多机lora批量训练" in text or "多机sft" in text:
            return ["train_multinode_sft_pipeline"]
        if "dpo_train_launcher" in text or "增强训练" in text or "dpo" in text:
            return ["dpo_train_launcher"]
        if "batch_train_lora" in text or "lora批量训练" in text or "lora训练" in text:
            return ["batch_train_lora"]
        if "batch_train_full" in text or "全参批量训练" in text or "全参训练" in text:
            return ["batch_train_full"]
        if "grpo_train" in text or "grpo训练" in text or "grpo" in text:
            return ["grpo_train"]
        if "create_command_vpn" in text or "定时训练" in text or "日常训练" in text:
            return ["create_command_vpn"]
        return []

    def _is_stop_all_request(self, message: str) -> bool:
        text = (message or "").lower()
        all_keywords = ["所有", "全部", "全部的", "all"]
        process_keywords = ["训练进程", "训练任务", "训练", "train"]
        return any(keyword in text for keyword in all_keywords) and any(
            keyword in text for keyword in process_keywords
        )

    def _resolve_data_stop_patterns(self, message: str) -> List[str]:
        text = (message or "").lower()
        if any(keyword in text for keyword in ["高级筛选", "筛选", "score_based_filtering"]):
            return ["score_based_filtering"]
        if any(keyword in text for keyword in ["数据预处理", "预处理", "data_preprocessing"]):
            return ["data_preprocessing"]
        recent_patterns = self._recent_dataprocessor_stop_patterns()
        if recent_patterns:
            return recent_patterns
        return [
            "data_preprocessing",            
            "score_based_filtering",
            
        ]

    def _recent_dataprocessor_stop_patterns(self) -> List[str]:
        record = self.last_completed_task or {}
        if record.get("agent") != "dataprocessor":
            return []

        text = "\n".join(
            str(record.get(key) or "")
            for key in ("task", "params", "response")
        ).lower()
        if "score_based_filtering" in text or "高级筛选" in text or "筛选任务" in text:
            return ["score_based_filtering"]
        if "data_preprocessing" in text or "数据预处理" in text or "预处理任务" in text:
            return ["data_preprocessing"]
        return []

    def _resolve_evaluate_stop_patterns(self, message: str) -> List[str]:
        text = (message or "").lower()
        if any(keyword in text for keyword in ["双模型评估", "双模型", "compare_between_models_vpn"]):
            return self._with_evaluate_child_patterns(["compare_between_models_vpn"], message)
        if any(keyword in text for keyword in ["单模型评估", "单模型", "single_model_evaluation_vpn"]):
            return self._with_evaluate_child_patterns(["single_model_evaluation_vpn"], message)
        if any(keyword in text for keyword in ["ckpt评估", "checkpoint评估", "checkpoint", "ckpt_eval"]):
            return self._with_evaluate_child_patterns(["ckpt_eval"], message)
        recent_patterns = self._recent_evaluator_stop_patterns()
        if recent_patterns:
            return recent_patterns
        return self._with_evaluate_child_patterns([
            "compare_between_models_vpn",
            "single_model_evaluation_vpn",
            "ckpt_eval",
        ], message)

    def _extract_docker_paths_from_text(self, text: str) -> List[str]:
        paths = []
        for match in re.findall(r"/[^\s,，;；`'\")]+", text or ""):
            cleaned = match.rstrip(".。")
            if cleaned.startswith("/home/") and cleaned not in paths:
                paths.append(cleaned)
        return paths

    def _with_evaluate_child_patterns(self, patterns: List[str], message: str = "") -> List[str]:
        combined = list(patterns)
        text = "\n".join([
            message or "",
            *(
                str((self.last_completed_task or {}).get(key) or "")
                for key in ("task", "params", "response")
            ),
        ])
        for path in self._extract_docker_paths_from_text(text):
            if path not in combined:
                combined.append(path)
        return combined

    def _evaluate_worker_stop_targets(
        self,
        patterns: List[str],
        message: str = "",
        container: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = "\n".join([
            message or "",
            " ".join(patterns),
            *(
                str((self.last_completed_task or {}).get(key) or "")
                for key in ("task", "params", "response")
            ),
        ]).lower()
        path_patterns = self._extract_docker_paths_from_text(text)
        model_name_patterns = [
            os.path.basename(os.path.normpath(path))
            for path in path_patterns
            if os.path.basename(os.path.normpath(path))
        ]

        def unique(items: List[str]) -> List[str]:
            result = []
            for item in items:
                item = str(item).strip()
                if item and item not in result:
                    result.append(item)
            return result

        has_model_identifiers = bool(path_patterns or model_name_patterns)

        has_single_model_signal = (
            "single_model_evaluation_vpn" in text
            or "单模型评估" in text
            or "单模型" in text
        )
        has_compare_model_signal = (
            "compare_between_models_vpn" in text
            or "双模型评估" in text
            or "双模型" in text
        )
        explicit_ckpt_signal = any(
            keyword in (message or "").lower()
            for keyword in ["ckpt_eval", "ckpt", "checkpoint"]
        ) or any(keyword in (message or "") for keyword in ["ckpt评估", "checkpoint评估"])
        has_ckpt_signal = explicit_ckpt_signal or (
            not has_single_model_signal
            and not has_compare_model_signal
            and ("ckpt_eval" in text or "ckpt" in text or "checkpoint" in text)
        )

        if has_ckpt_signal:
            worker_patterns = unique([
                "--port 6101",
                "--port 6102",
                "stage_run",
                "excel_thread_stage_consistency_vpn",
                "record_to_diagnose",
                "excel_thread_dignose_consistency_vpn",
                "auto_tagging_stage",
                "auto_tagging_diagnose",
                "/home/workspace/eval/daily_train",
                "/home/workspace/log/daily_train",
                *path_patterns,
                *model_name_patterns,
            ])
            if not has_model_identifiers:
                worker_patterns.append("vllm serve")
            return [{
                "container": container or self.training_container,
                "patterns": worker_patterns,
                "label": "ckpt评估worker",
            }]

        worker_patterns = unique([
            "--port 8101",
            "--port 8102",
            "stage_run",
            "excel_thread_stage_consistency_vpn",
            "record_to_diagnose",
            "diagnose_accuracy",
            "excel_thread_dignose_consistency_vpn",
            "thread_medhalt_confidence_eval",
            "auto_tagging_stage",
            "auto_tagging_diagnose",
            "/home/workspace/log/daily_train",
            *path_patterns,
            *model_name_patterns,
        ])
        if not has_model_identifiers:
            worker_patterns.append("vllm serve")
        return [{
            "container": container or self.training_container,
            "patterns": worker_patterns,
            "label": "单/双模型评估worker",
        }]

    def _stop_evaluate_worker_processes(
        self,
        message: str,
        patterns: List[str],
        container: str,
    ) -> str:
        outputs = []
        for target in self._evaluate_worker_stop_targets(patterns, message, container):
            container = target["container"]
            target_patterns = target["patterns"]
            if not target_patterns:
                continue
            try:
                result = self._run_docker_stop(
                    container,
                    self._build_stop_by_patterns_command(target_patterns),
                    timeout=60,
                )
            except FileNotFoundError:
                raise
            except subprocess.TimeoutExpired:
                result = "结束指令已发送但等待 worker 容器返回超时；请稍后复查任务状态。"
            outputs.append(
                f"worker容器: {container}\n"
                f"目标: {target['label']}\n"
                f"匹配: {', '.join(target_patterns)}\n"
                f"{result}"
            )
        return "\n\n".join(outputs)

    def _recent_evaluator_stop_patterns(self) -> List[str]:
        record = self.last_completed_task or {}
        if record.get("agent") != "evaluator":
            return []

        text = "\n".join(
            str(record.get(key) or "")
            for key in ("task", "params", "response")
        ).lower()
        if "compare_between_models_vpn" in text or "双模型评估" in text or "双模型" in text:
            return self._with_evaluate_child_patterns(["compare_between_models_vpn"])
        if "single_model_evaluation_vpn" in text or "单模型评估" in text or "单模型" in text:
            return self._with_evaluate_child_patterns(["single_model_evaluation_vpn"])
        if "ckpt_eval" in text or "ckpt评估" in text or "checkpoint评估" in text or "checkpoint" in text:
            return self._with_evaluate_child_patterns(["ckpt_eval"])
        return []

    def _agent_self_exclusion_env(self) -> Dict[str, str]:
        return {
            "AGENT_PID": str(os.getpid()),
            "AGENT_CMD_PATTERN": "api_app.py",
        }

    def _agent_self_exclusion_function(self) -> str:
        return r"""
is_agent_self() {
    local p="$1"
    [ -n "$AGENT_PID" ] && [ "$p" = "$AGENT_PID" ] && return 0
    local cmd
    cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
    [ -n "$AGENT_CMD_PATTERN" ] && [ -n "$cmd" ] && printf '%s' "$cmd" | grep -qF "$AGENT_CMD_PATTERN" && return 0
    return 1
}
"""

    def _run_docker_stop(self, container: str, shell_command: str, timeout: int = 45) -> str:
        env_args = []
        for key, value in self._agent_self_exclusion_env().items():
            env_args.extend(["-e", f"{key}={value}"])
        process = subprocess.run(
            ["docker", "exec", *env_args, container, "sh", "-c", shell_command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, returncode = self._hide_transient_remaining_pids(
            container,
            process.stdout,
            process.returncode,
        )
        output = []
        if stdout.strip():
            output.append(f"标准输出:\n{stdout.strip()}")
        if process.stderr.strip():
            output.append(f"标准错误:\n{process.stderr.strip()}")
        output.append(f"返回码: {returncode}")
        return "\n".join(output)

    def _hide_transient_remaining_pids(
        self,
        container: str,
        stdout: str,
        returncode: int,
    ):
        """Suppress one-shot remaining PID reports when a quick recheck is clean."""
        if returncode == 0 or "仍有残留PID" not in (stdout or ""):
            return stdout, returncode

        remaining_pids = []
        for line in stdout.splitlines():
            if "仍有残留PID" not in line:
                continue
            for pid in re.findall(r"\b\d+\b", line):
                if pid not in remaining_pids:
                    remaining_pids.append(pid)
        if not remaining_pids:
            return stdout, returncode

        time.sleep(2)
        pid_list = " ".join(shlex.quote(pid) for pid in remaining_pids)
        check_command = f"""
remaining=""
for pid in {pid_list}; do
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
        stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if [ -n "$cmd" ]; then
            remaining="$remaining $pid"
        fi
    fi
done
printf "%s" "$remaining"
"""
        try:
            recheck = subprocess.run(
                ["docker", "exec", container, "sh", "-c", check_command],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (subprocess.SubprocessError, OSError):
            return stdout, returncode

        if recheck.returncode not in (0, 1) or recheck.stdout.strip():
            return stdout, returncode

        filtered_lines = [
            line for line in stdout.splitlines()
            if "仍有残留PID" not in line
        ]
        filtered_stdout = "\n".join(filtered_lines).strip()
        if filtered_stdout:
            filtered_stdout += "\n"
        filtered_stdout += "复查未发现训练残留进程，已完成清理。"
        return filtered_stdout, 0

    def _docker_process_snapshot(
        self,
        container: str,
        root_pid: Optional[str] = None,
        timeout: int = 10,
    ) -> List[str]:
        """Read the container process table once and resolve descendants in awk."""
        root = str(root_pid or "").strip()
        if root and not root.isdigit():
            return []
        exclusion = self._agent_self_exclusion_function()
        shell_command = f"""{exclusion}
ps -eo pid=,ppid=,pgid=,stat=,args= 2>/dev/null | awk -v root={shlex.quote(root)} \\
    -v agent_pid=\"$AGENT_PID\" -v agent_pattern=\"$AGENT_CMD_PATTERN\" '
function self_process(p, command) {{
    return (agent_pid != "" && p == agent_pid) || (agent_pattern != "" && index(command, agent_pattern) > 0)
}}
function emit(p, command) {{
    if (p != "" && !self_process(p, command) && command != "") print p " " command
}}
function walk(parent, i, child) {{
    for (i = 1; i <= child_count[parent]; i++) {{
        child = children[parent, i]
        if (!seen[child]++) {{
            walk(child)
            emit(child, command_line[child])
        }}
    }}
}}
{{
    line = $0
    pid = $1; parent = $2; group = $3; state = $4
    command = line
    sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]*/, "", command)
    if (state ~ /Z/) next
    command_line[pid] = command
    parent_of[pid] = parent
    group_of[pid] = group
    children[parent, ++child_count[parent]] = pid
}}
END {{
    if (root != "") {{
        walk(root)
        emit(root, command_line[root])
        root_group = group_of[root]
        if (root_group != "") for (pid in group_of) if (group_of[pid] == root_group) emit(pid, command_line[pid])
    }}
}}'
"""
        env_args = []
        for key, value in self._agent_self_exclusion_env().items():
            env_args.extend(["-e", f"{key}={value}"])
        attempts = [timeout, max(timeout, 20)]
        last_error = None
        for attempt_timeout in attempts:
            try:
                process = subprocess.run(
                    ["docker", "exec", *env_args, container, "sh", "-c", shell_command],
                    capture_output=True,
                    text=True,
                    timeout=attempt_timeout,
                )
                if process.returncode not in (0, 1):
                    return []
                return [line.strip() for line in process.stdout.splitlines() if line.strip()]
            except subprocess.TimeoutExpired as exc:
                last_error = exc
        if last_error:
            raise last_error
        return []

    def _check_docker_pattern_processes(self, container: str, patterns: List[str]) -> List[str]:
        if not patterns:
            return []
        regexes = " ".join(
            shlex.quote(self._hide_pattern_from_pgrep_self_match(pattern))
            for pattern in patterns
        )
        exclusion = self._agent_self_exclusion_function()
        shell_command = f"""{exclusion}
snapshot=$(ps -eo pid=,ppid=,pgid=,stat=,args= 2>/dev/null || true)
for pattern in {regexes}; do
    printf "%s\\n" "$snapshot" | awk -v pattern="$pattern" -v agent_pid="$AGENT_PID" -v agent_pattern="$AGENT_CMD_PATTERN" '
    function self_process(p, command) {{ return (agent_pid != "" && p == agent_pid) || (agent_pattern != "" && index(command, agent_pattern) > 0) }}
    {{
        line=$0; pid=$1; state=$4; command=line
        sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]*/, "", command)
        if (!(state ~ /Z/) && !self_process(pid, command) && command ~ pattern) print pid " " command
    }}'
done
"""
        env_args = []
        for key, value in self._agent_self_exclusion_env().items():
            env_args.extend(["-e", f"{key}={value}"])
        attempts = [10, 20]
        last_error = None
        for attempt_timeout in attempts:
            try:
                process = subprocess.run(
                    ["docker", "exec", *env_args, container, "sh", "-c", shell_command],
                    capture_output=True,
                    text=True,
                    timeout=attempt_timeout,
                )
                if process.returncode not in (0, 1):
                    return []
                return [line.strip() for line in process.stdout.splitlines() if line.strip()]
            except subprocess.TimeoutExpired as exc:
                last_error = exc
        if last_error:
            raise last_error
        return []

    def _check_docker_pid_processes(self, container: str, pid: str) -> List[str]:
        pid = str(pid).strip()
        if not pid.isdigit():
            return []
        return self._docker_process_snapshot(container, root_pid=pid)
    def _hide_pattern_from_pgrep_self_match(self, pattern: str) -> str:
        escaped = re.escape(pattern)
        return re.sub(r"([A-Za-z0-9])", r"[\1]", escaped, count=1)

    def _build_stop_by_patterns_command(self, patterns: List[str]) -> str:
        regexes = " ".join(shlex.quote(self._hide_pattern_from_pgrep_self_match(pattern)) for pattern in patterns)
        exclusion = self._agent_self_exclusion_function()
        return f"""{exclusion}
targets=""
collect_descendants() {{
    parent="$1"
    children=$(ps -eo pid=,ppid= | awk -v p="$parent" '$2 == p {{print $1}}')
    for child in $children; do
        collect_descendants "$child"
        echo "$child"
    done
}}

for pattern in {regexes}; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    for pid in $pids; do
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
            cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
            case "$stat" in *Z*) continue ;; esac
            pgid=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
            pgid_pids=""
            if [ -n "$pgid" ]; then
                pgid_pids=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2 == g {{print $1}}')
            fi
            tree_pids=$(collect_descendants "$pid")
            for target in $(printf "%s\\n%s\\n%s\\n" "$tree_pids" "$pgid_pids" "$pid" | awk 'NF && !seen[$1]++ {{print $1}}'); do
                if [ "$target" = "$$" ] || [ "$target" = "$PPID" ]; then
                    continue
                fi
                if is_agent_self "$target"; then
                    echo "SKIP-AGENT $target"
                    continue
                fi
                target_stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
                case "$target_stat" in *Z*) continue ;; esac
                target_cmd=$(ps -p "$target" -o args= 2>/dev/null || true)
                if [ -n "$target_cmd" ]; then
                    echo "TERM $target $target_cmd"
                    kill -TERM "$target" 2>/dev/null || true
                    targets="$targets $target"
                fi
            done
        fi
    done
done

sleep 2
for pid in $targets; do
    if is_agent_self "$pid"; then
        echo "SKIP-AGENT $pid"
        continue
    fi
    if ps -p "$pid" >/dev/null 2>&1; then
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        echo "KILL $pid $cmd"
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

sleep 1
remaining=""
for pattern in {regexes}; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    for pid in $pids; do
            if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
                if is_agent_self "$pid"; then
                    continue
                fi
                stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
                case "$stat" in *Z*) continue ;; esac
                remaining="$remaining $pid"
            fi
    done
done

if [ -n "$remaining" ]; then
    echo "仍有残留PID:$remaining"
    exit 1
fi

if [ -z "$targets" ]; then
    echo "未找到匹配进程"
else
    echo "已终止PID:$targets"
fi
exit 0
"""

    def _build_force_stop_pids_command(self, pids: List[str]) -> str:
        valid_pids = [str(pid).strip() for pid in pids if str(pid).strip().isdigit()]
        if not valid_pids:
            return 'echo "未找到可强制终止进程"; exit 0'
        pid_list = " ".join(shlex.quote(pid) for pid in valid_pids)
        exclusion = self._agent_self_exclusion_function()
        return f"""{exclusion}
targets="{pid_list}"
for pid in $targets; do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && ps -p "$pid" >/dev/null 2>&1; then
        if is_agent_self "$pid"; then
            echo "SKIP-AGENT $pid"
            continue
        fi
        stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        echo "TERM-RETRY $pid $cmd"
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

sleep 2
for pid in $targets; do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && ps -p "$pid" >/dev/null 2>&1; then
        if is_agent_self "$pid"; then
            echo "SKIP-AGENT $pid"
            continue
        fi
        stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        echo "KILL-RETRY $pid $cmd"
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

sleep 1
remaining=""
for pid in $targets; do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && ps -p "$pid" >/dev/null 2>&1; then
        if is_agent_self "$pid"; then
            continue
        fi
        stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        remaining="$remaining $pid"
    fi
done

if [ -n "$remaining" ]; then
    echo "仍有残留PID:$remaining"
    exit 1
fi

echo "已二次清理PID:$targets"
exit 0
"""

    def _build_stop_by_pid_command(self, pid: str, fallback_patterns: Optional[List[str]] = None) -> str:
        quoted_pid = shlex.quote(str(pid))
        regexes = shlex.quote("__codex_no_fallback_match__")
        if fallback_patterns:
            regexes = " ".join(
                shlex.quote(self._hide_pattern_from_pgrep_self_match(pattern))
                for pattern in fallback_patterns
            )
        exclusion = self._agent_self_exclusion_function()
        return f"""{exclusion}
pid={quoted_pid}
cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
if [ -z "$cmd" ]; then
    echo "PID $pid 不存在或已结束"
else
    pgid=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
    pgid_pids=""
    if [ -n "$pgid" ]; then
        pgid_pids=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2 == g {{print $1}}')
    fi

    collect_descendants() {{
        parent="$1"
        children=$(ps -eo pid=,ppid= | awk -v p="$parent" '$2 == p {{print $1}}')
        for child in $children; do
            collect_descendants "$child"
            echo "$child"
        done
    }}

    tree_pids=$(collect_descendants "$pid")
    targets=$(printf "%s\\n%s\\n%s\\n" "$tree_pids" "$pgid_pids" "$pid" | awk 'NF && !seen[$1]++ {{print $1}}')

    for target in $targets; do
        if [ "$target" = "$$" ] || [ "$target" = "$PPID" ]; then
            continue
        fi
        if is_agent_self "$target"; then
            echo "SKIP-AGENT $target"
            continue
        fi
        stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        target_cmd=$(ps -p "$target" -o args= 2>/dev/null || true)
        if [ -n "$target_cmd" ]; then
            echo "TERM $target $target_cmd"
            kill -TERM "$target" 2>/dev/null || true
        fi
    done

    sleep 3
    for target in $targets; do
        if [ "$target" = "$$" ] || [ "$target" = "$PPID" ]; then
            continue
        fi
        if is_agent_self "$target"; then
            echo "SKIP-AGENT $target"
            continue
        fi
        if ps -p "$target" >/dev/null 2>&1; then
            stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
            case "$stat" in *Z*) continue ;; esac
            target_cmd=$(ps -p "$target" -o args= 2>/dev/null || true)
            echo "KILL $target $target_cmd"
            kill -KILL "$target" 2>/dev/null || true
        fi
    done
fi

fallback_targets=""
collect_descendants() {{
    parent="$1"
    children=$(ps -eo pid=,ppid= | awk -v p="$parent" '$2 == p {{print $1}}')
    for child in $children; do
        collect_descendants "$child"
        echo "$child"
    done
}}

for pattern in {regexes}; do
    if [ -n "$pattern" ]; then
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        for matched in $pids; do
            if [ "$matched" != "$$" ] && [ "$matched" != "$PPID" ]; then
                if is_agent_self "$matched"; then
                    echo "SKIP-AGENT $matched"
                    continue
                fi
                matched_stat=$(ps -p "$matched" -o stat= 2>/dev/null || true)
                case "$matched_stat" in *Z*) continue ;; esac
                matched_cmd=$(ps -p "$matched" -o args= 2>/dev/null || true)
                pgid=$(ps -p "$matched" -o pgid= 2>/dev/null | tr -d ' ')
                pgid_pids=""
                if [ -n "$pgid" ]; then
                    pgid_pids=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2 == g {{print $1}}')
                fi
                tree_pids=$(collect_descendants "$matched")
                for target in $(printf "%s\\n%s\\n%s\\n" "$tree_pids" "$pgid_pids" "$matched" | awk 'NF && !seen[$1]++ {{print $1}}'); do
                    if [ "$target" = "$$" ] || [ "$target" = "$PPID" ]; then
                        continue
                    fi
                    if is_agent_self "$target"; then
                        echo "SKIP-AGENT $target"
                        continue
                    fi
                    target_stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
                    case "$target_stat" in *Z*) continue ;; esac
                    target_cmd=$(ps -p "$target" -o args= 2>/dev/null || true)
                    if [ -n "$target_cmd" ]; then
                        echo "TERM-FALLBACK $target $target_cmd"
                        kill -TERM "$target" 2>/dev/null || true
                        fallback_targets="$fallback_targets $target"
                    fi
                done
            fi
        done
    fi
done

sleep 2
for matched in $fallback_targets; do
    if is_agent_self "$matched"; then
        echo "SKIP-AGENT $matched"
        continue
    fi
    if ps -p "$matched" >/dev/null 2>&1; then
        matched_cmd=$(ps -p "$matched" -o args= 2>/dev/null || true)
        echo "KILL-FALLBACK $matched $matched_cmd"
        kill -KILL "$matched" 2>/dev/null || true
    fi
done

remaining=""
for target in $targets $fallback_targets; do
    stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
    if [ -n "$target" ] && [ "$target" != "$$" ] && [ "$target" != "$PPID" ] && ! is_agent_self "$target" && ps -p "$target" >/dev/null 2>&1 && ! printf "%s" "$stat" | grep -q Z; then
        remaining="$remaining $target"
    fi
done
for pattern in {regexes}; do
    if [ -n "$pattern" ]; then
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        for matched in $pids; do
            if [ "$matched" != "$$" ] && [ "$matched" != "$PPID" ] && ! is_agent_self "$matched"; then
                matched_stat=$(ps -p "$matched" -o stat= 2>/dev/null || true)
                case "$matched_stat" in *Z*) continue ;; esac
                remaining="$remaining $matched"
            fi
        done
    fi
done

remaining=$(printf "%s\\n" $remaining | awk 'NF && !seen[$1]++ {{printf " %s", $1}}')
if [ -n "$remaining" ]; then
    echo "仍有残留PID:$remaining"
    exit 1
fi

if [ -z "$targets" ] && [ -z "$fallback_targets" ]; then
    echo "未找到可终止进程"
else
    echo "已终止PID:$targets $fallback_targets"
fi
exit 0
"""

    async def _handle_stop_command(
        self,
        message: str,
        task_type: Optional[str] = None,
        pid: Optional[str] = None,
        container: Optional[str] = None,
    ) -> Optional[str]:
        stop_info = self._parse_stop_command(message)
        if stop_info is None and task_type is None and not (pid and container):
            return None

        if stop_info:
            task_type = task_type or stop_info["task_type"]
            container = container or stop_info["container"]
            pid = pid or stop_info.get("pid")
        task_type = self._normalize_stop_task_type(task_type)
        actual_task_type = self._infer_stop_task_type_from_pid(container, pid)
        if actual_task_type:
            task_type = actual_task_type
        timeout_task_type = task_type
        timeout_container = container
        timeout_pid = pid
        timeout_patterns: Optional[List[str]] = None

        try:
            if task_type is None and pid and container:
                if not str(pid).isdigit():
                    return f"PID 格式不正确：{pid}"
                result = self._run_docker_stop(
                    container,
                    self._build_stop_by_pid_command(pid),
                    timeout=45,
                )
                self.current_task_state = {"status": "idle"}
                self.pending_parameters.clear()
                return f"已发送结束进程指令。\n容器: {container}\nPID: {pid}\n{result}"

            if task_type == "train":
                record = None
                if not pid and not container:
                    launch_target = self._last_trainer_launch_target()
                    pid = pid or launch_target.get("pid")
                    container = container or launch_target.get("container")
                if pid and not container:
                    record = self._train_pid_record_by_pid_any_container(pid)
                    if record:
                        container = record.get("container")
                if not pid and not container and not self._is_stop_all_request(message):
                    record = self._latest_train_pid_record_any_container()
                    if record:
                        container = record.get("container")
                        pid = str(record.get("pid"))
                    else:
                        bg_record = self._latest_background_task_record(
                            "train",
                            self._all_stop_patterns_for_type("train"),
                        )
                        if bg_record:
                            container = bg_record.get("container")
                            if bg_record.get("pid"):
                                pid = str(bg_record.get("pid"))
                if not container and self._is_stop_all_request(message):
                    inferred = self._infer_container_by_running_patterns(self._default_train_stop_patterns())
                    if inferred:
                        container = inferred["container"]
                container = container or self.training_container

                if not pid:
                    patterns = self._resolve_train_stop_patterns(message)
                    timeout_task_type = "train"
                    timeout_container = container
                    timeout_pid = None
                    timeout_patterns = patterns
                    result = self._run_docker_stop(
                        container,
                        self._build_stop_by_patterns_command(patterns),
                    )
                    self._release_stopped_gpu_reservations(
                        {"train"},
                        container,
                        fallback_patterns=patterns,
                    )
                    self.current_task_state = {"status": "idle"}
                    self.pending_parameters.clear()
                    return (
                        f"已发送结束训练指令。\n容器: {container}\n"
                        f"脚本: {', '.join(patterns)}\n{result}"
                    )
                if not str(pid).isdigit():
                    return f"PID 格式不正确：{pid}"
                if record is None:
                    record = self._train_pid_record_by_pid(container, pid)

                train_fallback_patterns = self._train_record_fallback_patterns(record)
                timeout_task_type = "train"
                timeout_container = container
                timeout_pid = pid
                timeout_patterns = train_fallback_patterns
                try:
                    result = self._run_docker_stop(
                        container,
                        self._build_stop_by_pid_command(pid, fallback_patterns=train_fallback_patterns),
                        timeout=45,
                    )
                except subprocess.TimeoutExpired:
                    remaining_by_pid = self._check_docker_pid_processes(container, pid)
                    remaining_by_pattern = self._check_docker_pattern_processes(container, train_fallback_patterns)
                    remaining = []
                    for item in remaining_by_pid + remaining_by_pattern:
                        if item not in remaining:
                            remaining.append(item)
                    if remaining:
                        retry_pids = []
                        for item in remaining:
                            match = re.match(r"^\s*(\d+)\b", item)
                            if match:
                                pid_value = match.group(1)
                                if pid_value not in retry_pids:
                                    retry_pids.append(pid_value)

                        retry_output = self._run_docker_stop(
                            container,
                            self._build_force_stop_pids_command(retry_pids),
                            timeout=20,
                        )
                        final_remaining_by_pid = self._check_docker_pid_processes(container, pid)
                        final_remaining_by_pattern = self._check_docker_pattern_processes(container, train_fallback_patterns)
                        final_remaining = []
                        for item in final_remaining_by_pid + final_remaining_by_pattern:
                            if item not in final_remaining:
                                final_remaining.append(item)

                        if final_remaining:
                            final_remaining_pids = []
                            for item in final_remaining:
                                match = re.match(r"^\s*(\d+)\b", item)
                                if match and match.group(1) not in final_remaining_pids:
                                    final_remaining_pids.append(match.group(1))
                            pid_summary = ", ".join(final_remaining_pids) if final_remaining_pids else "; ".join(final_remaining[:3])
                            result = (
                                "结束指令已发送，系统已自动重试清理训练进程；"
                                "复查仍发现残留进程，请稍后再次查询任务状态。\n"
                                f"仍发现残留 PID: {pid_summary}\n"
                                f"{retry_output}"
                            )
                        else:
                            result = (
                                "终止命令首次执行超时，系统已自动重试并完成清理；"
                                "复查未发现残留进程，任务已完成清理。\n"
                                f"{retry_output}"
                            )
                    else:
                        result = "结束指令已发送；虽然等待命令返回超时，但复查未发现残留进程，任务已完成清理。"
                multinode_stop_result = self._stop_multinode_training_record_processes(container, record, result)
                multinode_remaining = "仍发现残留 PID" in multinode_stop_result
                if multinode_stop_result:
                    result = f"{result}\n{multinode_stop_result}"
                stopped_task_type = self._task_type_from_stop_result(result)
                if stopped_task_type == "assessment":
                    assessment_patterns = self._all_stop_patterns_for_type("assessment")
                    worker_result = self._stop_evaluate_worker_processes(message, assessment_patterns, container)
                    self._release_stopped_gpu_reservations(
                        {"assessment", "evaluate"},
                        container,
                        pid=pid,
                        fallback_patterns=assessment_patterns,
                    )
                    self.current_task_state = {"status": "idle"}
                    self.pending_parameters.clear()
                    extra = f"\n\n{worker_result}" if worker_result else ""
                    return f"已发送结束模型评估指令。\n容器: {container}\nPID: {pid}\n{result}{extra}"
                if not multinode_remaining:
                    self._release_stopped_gpu_reservations(
                        {"train"},
                        container,
                        pid=pid,
                        fallback_patterns=train_fallback_patterns,
                    )
                self.current_task_state = {"status": "idle"}
                self.pending_parameters.clear()
                return f"已发送结束训练指令。\n容器: {container}\nPID: {pid}\n{result}"

            script_patterns = {
                "assessment": self._resolve_evaluate_stop_patterns(message),
                "evaluate": self._resolve_evaluate_stop_patterns(message),
                "data": self._resolve_data_stop_patterns(message),
            }
            patterns = script_patterns[task_type]
            resolved = self._resolve_stop_container_and_patterns(task_type, patterns, container)
            container = resolved["container"]
            patterns = resolved["patterns"]

            if pid:
                if not str(pid).isdigit():
                    return f"PID 格式不正确：{pid}"
                fallback_patterns = patterns if task_type in {"assessment", "evaluate"} else None
                shell_command = self._build_stop_by_pid_command(pid, fallback_patterns=fallback_patterns)
            else:
                shell_command = self._build_stop_by_patterns_command(patterns)

            timeout_task_type = task_type
            timeout_container = container
            timeout_pid = pid
            timeout_patterns = patterns
            result = self._run_docker_stop(container, shell_command)
            worker_result = ""
            if task_type in {"assessment", "evaluate"}:
                worker_result = self._stop_evaluate_worker_processes(message, patterns, container)
            if task_type in {"assessment", "evaluate"}:
                self._release_stopped_gpu_reservations(
                    {"assessment", "evaluate"},
                    container,
                    pid=pid,
                    fallback_patterns=patterns,
                )
            self.current_task_state = {"status": "idle"}
            self.pending_parameters.clear()
            task_name = "模型评估" if task_type in {"assessment", "evaluate"} else "数据处理"
            target = f"PID: {pid}" if pid else f"脚本: {', '.join(patterns)}"
            extra = f"\n\n{worker_result}" if worker_result else ""
            return f"已发送结束{task_name}指令。\n容器: {container}\n{target}\n{result}{extra}"
        except FileNotFoundError:
            return "结束指令执行失败：未找到 docker 命令，请确认运行环境。"
        except subprocess.TimeoutExpired:
            return self._diagnose_stop_timeout(
                timeout_task_type,
                timeout_container,
                pid=timeout_pid,
                patterns=timeout_patterns,
            )
        except Exception as exc:
            logger.exception("Failed to handle stop command")
            return f"结束指令执行失败：{exc}"

    def _is_workflow_start_request(self, message: str) -> bool:
        text = (message or "").lower()
        return (
            "一键" in text
            and "训练" in text
            and ("部署" in text or "发布" in text)
            and ("评测" in text or "测评" in text or "benchmark" in text)
        )

    def _workflow_command(self, message: str) -> Optional[str]:
        if self._is_workflow_start_request(message):
            return "start"
        command = parse_workflow_control_command(message)
        workflow_id = self._extract_workflow_id(message)
        if command and workflow_id:
            return f"{command}:{workflow_id}"
        return command

    def _extract_workflow_id(self, text: str) -> Optional[str]:
        match = re.search(r"(wf-\d{14}-[0-9a-fA-F]{8})(?![0-9a-fA-F])", text or "")
        return match.group(1) if match else None

    def _workflow_protocol(
        self,
        workflow: Dict[str, Any],
        message: str,
        started: bool = False,
        agent: str = "orchestrator",
    ) -> Dict[str, Any]:
        status = workflow.get("status")
        protocol_type = (
            "workflow_started" if started and status == "running" else
            "workflow_finished" if status == "finished" else
            "workflow_failed" if status == "failed" else
            "workflow_status"
        )
        context = workflow.get("context") or {}
        train_type = self._normalize_train_type(context.get("train_type"))
        launch_mode = context.get("launch_mode")
        return self._with_protocol(
            protocol_type,
            agent,
            message,
            workflowId=workflow.get("workflow_id"),
            workflowKey=workflow.get("workflow_key"),
            workflowStatus=status,
            currentStage=workflow.get("current_stage"),
            datasetRef=workflow.get("dataset_ref"),
            trainType=train_type,
            trainTypeText=self._train_type_text(train_type, launch_mode),
            launchMode=launch_mode,
            isMultinode=launch_mode == "multinode",
            trainArgs=context.get("train_args"),
            stages=self._workflow_public_stages(workflow),
            trainedModelPath=context.get("trained_model_path"),
            publishedModelPath=context.get("published_model_path"),
            previousModelName=context.get("old_model_name"),
            benchmark=context.get("benchmark"),
            evaluationDatasetName=context.get("evaluation_dataset_name"),
            benchmarkResultEntry=_benchmark_result_entry(workflow),
            error=workflow.get("error"),
        )

    def _workflow_public_stages(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Keep workflow status responses compact while preserving useful progress."""
        metric_keys = {
            "container_name", "train_type", "output_dir", "latest_loss",
            "latest_epoch", "latest_step", "progress_percent", "overall_progress_percent",
            "current_step",
            "total_steps", "elapsed_time", "remaining_time", "pid_alive",
            "training_process_exists", "error_reason", "wandb_url",
            "wandb_url_pending", "wandb_mode", "wandb_url_source",
            "loss_source", "last_update_time", "stale", "stale_minutes",
            "sub_stage", "sub_stage_text", "train_progress_percent",
            "train_current_step", "train_total_steps", "eval_progress_percent",
            "eval_current_step", "eval_total_steps", "export_dir", "merge_status",
            "iteration_finished", "output_log_success", "error_detail",
            "step_name", "source", "destination", "temp_dir",
            "train_container", "inference_container", "message",
            "report_path", "report_text", "artifacts", "current_stage", "next_stage", "completed_stages",

            "total_stages", "current_stages", "stage_status_counts",
            "deployment_step", "old_model_name", "service_log_command",
        }
        result_keys = {
            "status", "benchmark_status", "processed", "total", "progress", "progress_percent",
            "correct", "accuracy", "invalid", "invalid_rate", "avg_f1",
            "record_only", "executed", "passed", "failed", "timeout", "executor_error", "pass@1",
            "result_entry", "resultEntry", "benchmark_job_id",
            "status_command", "stop_command", "log_command", "message", "error",
            "folder_path", "log_dir", "log_file", "result_path", "inference_service_stopped",
            "inference_service_stop_command", "inference_service_log_command",
            "inference_service_stop_error", "log_path", "log_tail", "log_updated_at",
            "stop_service_log_path", "stop_service_log_tail",
            "stop_service_log_updated_at",
        }
        public_stages: Dict[str, Any] = {}
        for name, stage in (workflow.get("stages") or {}).items():
            public = {
                key: value
                for key, value in stage.items()
                if key not in {"metrics", "debug", "history", "result"}
            }
            result = stage.get("result")
            if isinstance(result, dict):
                public_result = {
                    key: value
                    for key, value in result.items()
                    if key in result_keys and value is not None
                }
                if public_result:
                    public["result"] = public_result
            metrics = stage.get("metrics")
            if isinstance(metrics, dict):
                public_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key in metric_keys and value is not None
                }
                if public_metrics:
                    public["metrics"] = public_metrics
            public_stages[name] = public
        return public_stages

    def _workflow_display_text(self, workflow: Dict[str, Any]) -> str:
        context = workflow.get("context") or {}
        train_type = self._normalize_train_type(context.get("train_type")) or "lora"
        benchmark_name = _benchmark_name(workflow)
        labels = {
            "train": self._train_type_text(train_type) or "模型训练",
            "evaluate": "单模型评估",
            "publish": "版本化发布",
            "deploy": "推理服务部署",
            "benchmark": f"{benchmark_name}基准评测",
        }
        statuses = {
            "pending": "等待中",
            "awaiting_agent": "等待Agent执行",
            "starting_external": "Agent启动中",
            "starting": "启动中",
            "preparing": "准备中",
            "stopping": "停止中",
            "running": "运行中",
            "timeout": "超时",
            "finished": "已完成",
            "failed": "失败",
            "stopped": "已停止",
        }
        lines = [
            f"一键工作流 `{workflow.get('workflow_id')}`：{statuses.get(workflow.get('status'), workflow.get('status'))}",
            f"数据标识：`{workflow.get('dataset_ref')}`",
            f"当前阶段：{labels.get(workflow.get('current_stage'), workflow.get('current_stage'))}",
            "",
        ]
        if workflow.get("workflow_key"):
            lines.insert(2, f"逻辑标识：`{workflow.get('workflow_key')}`")
        for name in ("train", "evaluate", "publish", "deploy", "benchmark"):
            stage = (workflow.get("stages") or {}).get(name, {})
            lines.append(f"- {labels[name]}：{statuses.get(stage.get('status'), stage.get('status'))}")
        details = self._workflow_current_stage_details(workflow)
        if details:
            lines.append("\n当前阶段详情：")
            lines.extend(f"- {detail}" for detail in details)
        summary = self._workflow_output_summary_details(workflow)
        if summary:
            lines.append("\n工作流产出：")
            lines.extend(f"- {detail}" for detail in summary)
        if workflow.get("error"):
            if workflow.get("status") == "stopped":
                lines.append(f"\n停止原因：{workflow['error']}")
                lines.append("可以发送：`继续未完成的一键工作流`")
            else:
                lines.append(f"\n失败原因：{workflow['error']}")
                lines.append("可以发送：`继续上次失败的一键工作流`")
        if workflow.get("status") == "finished":
            lines.append(f"\n可以发送：`{_benchmark_result_entry(workflow)}`")
        return "\n".join(lines)

    def _read_eval_report_text(self, report_path: str, max_lines: int = 100) -> Optional[str]:
        """读取评估报告内容，带行数与长度上限保护。"""
        if not report_path:
            return None
        try:
            path = Path(report_path)
            if not path.is_file():
                return None
            with path.open("r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append("...（报告内容已截断，完整内容请见报告路径）")
                        break
                    lines.append(line.rstrip("\n"))
            text = "\n".join(lines)
            max_chars = 16384
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...（报告内容已截断，完整内容请见报告路径）"
            return text if text.strip() else None
        except Exception:
            return None

    def _workflow_output_summary_details(self, workflow: Dict[str, Any]) -> List[str]:
        """Summarize durable outputs from completed stages."""

        context = workflow.get("context") or {}
        stages = workflow.get("stages") or {}
        details: List[str] = []

        def add(label: str, value: Any) -> None:
            if value is not None and str(value).strip() != "":
                details.append(f"{label}：`{value}`")

        def status_text(value: Any) -> Any:
            return {
                "pending": "等待中",
                "awaiting_agent": "等待Agent执行",
                "starting_external": "Agent启动中",
                "starting": "启动中",
                "preparing": "准备中",
                "running": "运行中",
                "timeout": "超时",
                "finished": "已完成",
                "failed": "失败",
                "stopped": "已停止",
            }.get(value, value)

        train_stage = stages.get("train", {})
        if train_stage.get("status") == "finished":
            trained_model = context.get("trained_model_path") or train_stage.get("model_path")
            add("训练产物", trained_model)

        evaluate_stage = stages.get("evaluate", {})
        evaluate_metrics = evaluate_stage.get("metrics") if isinstance(evaluate_stage.get("metrics"), dict) else {}
        report_path = evaluate_metrics.get("report_path") or evaluate_stage.get("report_path")
        report_text = evaluate_metrics.get("report_text") if isinstance(evaluate_metrics, dict) else None
        if evaluate_stage.get("status") == "finished":
            if report_text:
                if report_path:
                    add("评估报告路径", report_path)
                details.append(f"评估报告内容：\n```\n{report_text}\n```")
            elif report_path:
                report_text = self._read_eval_report_text(report_path)
                if report_text is not None:
                    add("评估报告路径", report_path)
                    details.append(f"评估报告内容：\n```\n{report_text}\n```")
                else:
                    add("评估报告", report_path)
            else:
                add("评估结果", evaluate_stage.get("message") or "单模型评估已完成")



            artifacts = evaluate_metrics.get("artifacts") if isinstance(evaluate_metrics, dict) else []
            for artifact in artifacts if isinstance(artifacts, list) else []:
                if not isinstance(artifact, dict) or not artifact.get("exists"):
                    continue
                artifact_path = artifact.get("path")
                artifact_name = artifact.get("name") or "result"
                if artifact_path:
                    add("评估结果文件", f"{artifact_name}: {artifact_path}")
        publish_stage = stages.get("publish", {})
        published_model = None
        if publish_stage.get("status") == "finished":
            published_model = context.get("published_model_path") or publish_stage.get("model_path")
            add("发布模型", published_model)

        deploy_stage = stages.get("deploy", {})
        published_model_name = os.path.basename(str(published_model)) if published_model else None
        deployed_model = deploy_stage.get("model_name") or published_model_name
        if deploy_stage.get("status") == "finished":
            add("部署模型", deployed_model)
            add("部署日志", deploy_stage.get("service_log_command"))

        benchmark_stage = stages.get("benchmark", {})
        benchmark_result = benchmark_stage.get("result")
        if benchmark_stage.get("status") == "finished" and isinstance(benchmark_result, dict):
            add("基准评测状态", status_text(benchmark_result.get("status") or benchmark_result.get("benchmark_status")))
            add("基准评测结果入口", benchmark_result.get("result_entry") or benchmark_result.get("resultEntry"))
            add("基准评测日志", benchmark_result.get("log_dir") or benchmark_result.get("folder_path"))
            add("关闭服务日志", benchmark_result.get("inference_service_log_command"))
        elif benchmark_stage.get("status") == "finished":
            add("基准评测状态", status_text(benchmark_stage.get("status")))
            add("基准评测结果入口", _benchmark_result_entry(workflow))
        return details

    def _workflow_wandb_display_text(self, metrics: Dict[str, Any]) -> Optional[str]:
        wandb_url = metrics.get("wandb_url")
        if wandb_url is not None and str(wandb_url).strip() != "":
            return str(wandb_url).strip()
        wandb_mode = str(metrics.get("wandb_mode") or "").strip().lower()
        if wandb_mode == "offline":
            return "离线模式，本次没有在线链接"
        if metrics.get("wandb_url_pending") is True:
            return "生成中，稍后刷新状态"
        return None

    def _workflow_step_display_text(self, metrics: Dict[str, Any]) -> Any:
        current_step = metrics.get("current_step")
        total_steps = metrics.get("total_steps")
        if (
            current_step is not None
            and str(current_step).strip() != ""
            and total_steps is not None
            and str(total_steps).strip() != ""
        ):
            return f"{current_step}/{total_steps}"
        return metrics.get("latest_step")

    def _workflow_resume_checkpoint_step(self, context: Dict[str, Any]) -> Optional[int]:
        checkpoint = self._workflow_resume_checkpoint_text(context)
        if not checkpoint:
            return None
        match = re.search(r"(?:^|[/\\])checkpoint-(\d+)(?:$|[/\\])", checkpoint)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _workflow_metrics_behind_resume_checkpoint(
        self,
        metrics: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        checkpoint_step = self._workflow_resume_checkpoint_step(context)
        if checkpoint_step is None or checkpoint_step <= 0:
            return False
        step = metrics.get("current_step")
        if step is None or str(step).strip() == "":
            step = metrics.get("latest_step")
        if step is None or str(step).strip() == "":
            return False
        try:
            return float(step) < float(checkpoint_step)
        except Exception:
            return False

    def _workflow_metric_update_text(self, metrics: Dict[str, Any]) -> Any:
        last_update = metrics.get("last_update_time")
        if last_update is None or str(last_update).strip() == "":
            return None
        if metrics.get("stale") is True:
            stale_minutes = metrics.get("stale_minutes")
            if stale_minutes is not None and str(stale_minutes).strip() != "":
                return f"{last_update}（可能延迟，约{stale_minutes}分钟未更新）"
            return f"{last_update}（可能延迟）"
        return last_update

    def _workflow_train_finalizing_text(
        self,
        workflow: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Optional[str]:
        """Describe the LoRA merge window after training iterations finish."""
        context = workflow.get("context") or {}
        train_type = self._normalize_train_type(context.get("train_type")) or "lora"
        if train_type != "lora" or workflow.get("status") not in {"running", "starting"}:
            return None
        if not (metrics.get("training_process_exists") or metrics.get("pid_alive")):
            return None

        # A success marker may be left in output.log by an earlier iteration or
        # resume.  Never let that stale marker override explicit live progress
        # showing that the current run is still below its final step.
        current_step = None
        total_steps = None
        progress = None
        explicit_incomplete_progress = False
        try:
            current_step = float(metrics.get("current_step"))
            total_steps = float(metrics.get("total_steps"))
            explicit_incomplete_progress = total_steps > 0 and current_step < total_steps
        except (TypeError, ValueError):
            pass
        try:
            progress = float(str(metrics.get("progress_percent")).strip().rstrip("%"))
            explicit_incomplete_progress = explicit_incomplete_progress or progress < 100
        except (TypeError, ValueError):
            pass

        explicit_complete_progress = bool(
            current_step is not None
            and total_steps is not None
            and total_steps > 0
            and current_step >= total_steps
            and progress is not None
            and progress >= 100
        )
        if explicit_complete_progress:
            final_iteration_seen = True
        else:
            has_resume = bool(
                self._workflow_resume_checkpoint_text(context)
                or context.get("auto_resume_checkpoint")
            )
            has_current_run_progress = bool(
                current_step is not None
                and total_steps is not None
                and total_steps > 0
                and progress is not None
            )
            resume_marker_unsafe = bool(
                has_resume
                and (
                    metrics.get("wandb_url_pending") is True
                    or not has_current_run_progress
                    or self._workflow_metrics_behind_resume_checkpoint(metrics, context)
                )
            )
            final_iteration_seen = bool(
                (metrics.get("iteration_finished") or metrics.get("output_log_success"))
                and not explicit_incomplete_progress
                and not resume_marker_unsafe
            )
        if not final_iteration_seen:
            return None
        return "训练迭代已完成，正在合并模型，请稍后刷新状态"

    def _workflow_resume_checkpoint_text(self, context: Dict[str, Any]) -> Optional[str]:
        train_args = context.get("train_args") if isinstance(context.get("train_args"), dict) else {}
        resume = train_args.get("RESUME") if isinstance(train_args, dict) else None
        if resume is not None and str(resume).strip() != "":
            return str(resume).strip()
        return None

    def _workflow_latest_checkpoint_text(self, workflow: Dict[str, Any]) -> Optional[str]:
        train_stage = (workflow.get("stages") or {}).get("train") or {}
        metrics = train_stage.get("metrics")
        if isinstance(metrics, dict):
            for key in ("latest_checkpoint", "latest_checkpoint_path", "checkpoint_path"):
                value = metrics.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        # 训练未完成时不扫描输出目录，避免将同一数据标识的历史
        # checkpoint 显示成本次产物。运行中的 checkpoint 应由监控 metrics 明确提供。
        if train_stage.get("status") != "finished":
            return None
        try:
            checkpoint = _workflow_find_latest_checkpoint(workflow)
        except Exception:
            checkpoint = None
        if checkpoint is not None and str(checkpoint).strip():
            return str(checkpoint).strip()
        return None

    def _workflow_next_stage_display_text(self, workflow: Dict[str, Any]) -> Optional[str]:
        if workflow.get("status") not in {"running", "starting"}:
            return None
        current_stage = workflow.get("current_stage")
        if current_stage not in STAGES:
            return None
        current_index = STAGES.index(current_stage)
        if current_index + 1 >= len(STAGES):
            return None
        next_stage = STAGES[current_index + 1]
        context = workflow.get("context") or {}
        train_type = self._normalize_train_type(context.get("train_type")) or "lora"
        benchmark_name = _benchmark_name(workflow)
        labels = {
            "train": self._train_type_text(train_type) or "模型训练",
            "evaluate": "单模型评估",
            "publish": "版本化发布",
            "deploy": "推理服务部署",
            "benchmark": f"{benchmark_name}基准评测",
        }
        return labels.get(next_stage, next_stage)

    def _workflow_json_display_text(self, value: Any) -> Optional[str]:
        if value in (None, "", [], {}):
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _workflow_first_running_evaluate_detail(self, details: Any) -> Dict[str, Any]:
        if not isinstance(details, list):
            return {}
        for detail in details:
            if isinstance(detail, dict) and detail.get("status") == "running":
                return detail
        for detail in details:
            if isinstance(detail, dict):
                return detail
        return {}

    def _workflow_evaluate_log_step_text(self, metrics: Dict[str, Any]) -> Optional[str]:
        detail = self._workflow_first_running_evaluate_detail(metrics.get("current_stage_details"))
        summary = detail.get("progress_summary") if isinstance(detail, dict) else {}
        if not isinstance(summary, dict):
            return None
        step_name = summary.get("step_name")
        step_current = summary.get("step_current")
        step_total = summary.get("step_total")
        local_progress = summary.get("local_progress_percent")
        parts: List[str] = []
        if step_name is not None and str(step_name).strip() != "":
            parts.append(str(step_name))
        if step_current is not None and step_total is not None:
            parts.append(f"{step_current}/{step_total}")
        if local_progress is not None:
            parts.append(f"约{local_progress}%")
        return "，".join(parts) if parts else None

    def _workflow_evaluate_scores_text(self, scores: Any) -> Optional[str]:
        if not isinstance(scores, dict) or not scores:
            return None
        items: List[str] = []
        for name, result in scores.items():
            stage_name = str(name or "").strip()
            if not stage_name:
                continue
            if isinstance(result, dict):
                if result.get("average") is not None:
                    items.append(f"{stage_name} 平均分 {result.get('average')}")
                    continue
                compact_parts = []
                for key in ("accuracy", "mean_fuzziness"):
                    if result.get(key) is not None:
                        compact_parts.append(f"{key} {result.get(key)}")
                if compact_parts:
                    items.append(f"{stage_name} {'，'.join(compact_parts)}")
            elif result is not None:
                items.append(f"{stage_name} {result}")
        return "；".join(items) if items else None

    def _workflow_evaluate_running_detail_texts(self, metrics: Dict[str, Any]) -> List[str]:
        details = metrics.get("current_stage_details")
        if not isinstance(details, list):
            return []
        lines: List[str] = []
        for detail in details:
            if not isinstance(detail, dict) or detail.get("status") != "running":
                continue
            name = str(detail.get("name") or "当前子阶段").strip()
            summary = detail.get("progress_summary") if isinstance(detail.get("progress_summary"), dict) else {}
            log_meta = detail.get("progress_log_meta") if isinstance(detail.get("progress_log_meta"), dict) else {}
            parts: List[str] = []
            step_name = summary.get("step_name")
            step_current = summary.get("step_current")
            step_total = summary.get("step_total")
            local_progress = summary.get("local_progress_percent")
            step_parts: List[str] = []
            if step_name is not None and str(step_name).strip():
                step_parts.append(str(step_name).strip())
            if step_current is not None and step_total is not None:
                step_parts.append(f"{step_current}/{step_total}")
            if local_progress is not None:
                step_parts.append(f"局部进度 {local_progress}%")
            if step_parts:
                parts.append("，".join(step_parts))
            latest_line = self._workflow_short_display_text(summary.get("latest_line"), limit=120)
            if latest_line:
                parts.append(f"最新日志行：{latest_line}")
            updated_at = log_meta.get("updated_at")
            if updated_at:
                parts.append(f"日志更新时间：{updated_at}")
            progress_log = detail.get("progress_log")
            if progress_log:
                parts.append(f"日志路径：{progress_log}")
            if parts:
                lines.append(f"运行中子阶段 {name}：`{'；'.join(str(part) for part in parts)}`")
            tail_lines = [
                self._workflow_short_display_text(item, limit=120)
                for item in (summary.get("tail_lines") or [])[-2:]
            ]
            tail_lines = [item for item in tail_lines if item]
            if tail_lines:
                lines.append(f"{name} 日志尾部：`{' / '.join(tail_lines)}`")
        return lines
    def _workflow_publish_target_text(self, stage: Dict[str, Any], metrics: Dict[str, Any], context: Dict[str, Any]) -> Any:
        return (
            context.get("published_model_path")
            or stage.get("model_path")
            or metrics.get("destination")
            or stage.get("destination")
            or metrics.get("source")
            or stage.get("source")
        )

    def _workflow_deployment_step_text(self, value: Any) -> Any:
        labels = {
            "awaiting_agent": "等待 Agent 执行",
            "checking_config": "检查配置",
            "updating_config": "修改配置",
            "restarting_service": "重启服务",
            "waiting_service": "等待服务",
            "checking_service": "检查服务",
            "service_checked": "服务检查完成",
        }
        return labels.get(value, value)

    def _workflow_resume_display_snapshot(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Add user-facing deploy details to the immediate resume response."""
        if workflow.get("current_stage") != "deploy":
            return workflow
        stages = workflow.get("stages") if isinstance(workflow.get("stages"), dict) else {}
        stage = stages.get("deploy") if isinstance(stages.get("deploy"), dict) else None
        if not stage or stage.get("status") != "awaiting_agent":
            return workflow
        context = workflow.get("context") if isinstance(workflow.get("context"), dict) else {}
        model_path = stage.get("model_path") or context.get("published_model_path")
        if model_path and not stage.get("model_path"):
            stage["model_path"] = model_path
        if model_path and not stage.get("model_name"):
            stage["model_name"] = os.path.basename(str(model_path).rstrip("/"))
        stage.setdefault("deployment_step", "awaiting_agent")
        stage.setdefault("message", "工作流已恢复，等待 Agent 执行推理服务部署")
        return workflow

    def _workflow_short_display_text(self, value: Any, limit: int = 120) -> Optional[str]:
        if value in (None, "", [], {}):
            return None
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _workflow_current_stage_details(self, workflow: Dict[str, Any]) -> List[str]:
        """Build a compact human-readable summary for the active stage."""
        stage_name = str(workflow.get("current_stage") or "")
        stage = (workflow.get("stages") or {}).get(stage_name, {})
        metrics = stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {}
        context = workflow.get("context") or {}
        details: List[str] = []

        def add(label: str, value: Any) -> None:
            if value is not None and str(value).strip() != "":
                details.append(f"{label}：`{value}`")

        if stage_name != "publish":
            add("容器", stage.get("container") or metrics.get("container_name"))
            add("PID", stage.get("pid") or metrics.get("pid"))
        if stage_name == "train":
            resume_metrics_pending = self._workflow_metrics_behind_resume_checkpoint(metrics, context)
            finalizing_text = self._workflow_train_finalizing_text(workflow, metrics)
            train_type = self._normalize_train_type(context.get("train_type")) or "lora"
            dpo_sub_stage = str(metrics.get("sub_stage") or "").strip().lower()
            dpo_stage_text = None
            if train_type == "enhanced" and dpo_sub_stage == "evaluation":
                dpo_stage_text = "训练已完成，正在执行训练后 Evaluation"
            elif train_type == "enhanced" and dpo_sub_stage == "merge":
                dpo_stage_text = "Evaluation 已完成，正在合并模型"
            add("训练卡号", (context.get("train_args") or {}).get("LOCALHOST_ID"))
            if dpo_stage_text:
                add("训练状态", dpo_stage_text)
                add("训练 step", self._workflow_step_display_text({
                    "current_step": metrics.get("train_current_step"),
                    "total_steps": metrics.get("train_total_steps"),
                    "latest_step": metrics.get("train_current_step"),
                }))
                add("训练进度", metrics.get("train_progress_percent"))
                add("Evaluation step", self._workflow_step_display_text({
                    "current_step": metrics.get("eval_current_step"),
                    "total_steps": metrics.get("eval_total_steps"),
                    "latest_step": metrics.get("eval_current_step"),
                }))
                add("Evaluation 进度", metrics.get("eval_progress_percent"))
                add("导出目录", metrics.get("export_dir"))
                add("Merge 状态", metrics.get("merge_status"))
                add("WandB", self._workflow_wandb_display_text(metrics))
                add("指标来源", metrics.get("loss_source"))
                add("指标更新时间", self._workflow_metric_update_text(metrics))
            elif finalizing_text:
                add("训练状态", finalizing_text)
                add("loss", metrics.get("latest_loss"))
                add("epoch", metrics.get("latest_epoch"))
                add("最终 step", metrics.get("latest_step") or metrics.get("current_step"))
                add("WandB", self._workflow_wandb_display_text(metrics))
                add("指标来源", metrics.get("loss_source"))
                add("指标更新时间", self._workflow_metric_update_text(metrics))
            elif resume_metrics_pending:
                add("续训状态", "等待 checkpoint 恢复后的新指标")
            else:
                add("loss", metrics.get("latest_loss"))
                add("epoch", metrics.get("latest_epoch"))
                add("step", self._workflow_step_display_text(metrics))
                add("进度", metrics.get("progress_percent"))
                add("已运行", metrics.get("elapsed_time"))
                add("预计剩余", metrics.get("remaining_time"))
                add("WandB", self._workflow_wandb_display_text(metrics))
                add("指标来源", metrics.get("loss_source"))
                add("指标更新时间", self._workflow_metric_update_text(metrics))
            resume_checkpoint = self._workflow_resume_checkpoint_text(context)
            latest_checkpoint = self._workflow_latest_checkpoint_text(workflow)
            add("本次续训起点", resume_checkpoint)
            if latest_checkpoint and latest_checkpoint != resume_checkpoint:
                add("最新 checkpoint", latest_checkpoint)
            if context.get("auto_resume_checkpoint"):
                add("续训方式", "自动检测")
            elif context.get("auto_resume_checkpoint_status") == "not_found":
                add("续训方式", "未检测到可用 checkpoint，将按普通启动流程继续")
        elif stage_name == "evaluate":
            evaluate_progress = metrics.get("overall_progress_percent")
            add("进度", evaluate_progress if evaluate_progress is not None else metrics.get("progress_percent"))
            if metrics.get("completed_stages") is not None and metrics.get("total_stages") is not None:
                add("阶段进度", f"{metrics.get('completed_stages')}/{metrics.get('total_stages')}")
            current_stages = metrics.get("current_stages")
            if isinstance(current_stages, list):
                current_stages = "、".join(str(item) for item in current_stages if str(item).strip())
            add("当前子阶段", current_stages)
            add("下一评估阶段", metrics.get("next_stage"))
            running_detail_texts = self._workflow_evaluate_running_detail_texts(metrics)
            if running_detail_texts:
                details.extend(running_detail_texts)
            else:
                add("日志步骤", self._workflow_evaluate_log_step_text(metrics))
            if stage.get("status") == "finished":
                add("报告路径", metrics.get("report_path") or stage.get("report_path"))
            else:
                add("已有评分", self._workflow_evaluate_scores_text(metrics.get("scores")))
        elif stage_name == "publish":
            add("模型名称", stage.get("model_name") or stage.get("model_path") or context.get("published_model_path"))
            publish_progress = metrics.get("progress_percent", stage.get("progress_percent", 0))
            current_step = metrics.get("current_step") or stage.get("current_step")
            total_steps = metrics.get("total_steps") or stage.get("total_steps")
            step_suffix = f" ({current_step}/{total_steps})" if current_step and total_steps else ""
            publish_step_labels = {
                "copy_to_host": "复制到宿主机",
                "prepare_destination": "准备推理容器目标目录",
                "copy_to_inference": "复制到推理容器",
                "activate_model": "原子切换发布目录",
                "finished": "发布完成",
            }
            step_name = metrics.get("step_name") or stage.get("step_name")
            add("进度", f"{publish_progress}%{step_suffix}")
            add("子阶段", publish_step_labels.get(step_name, step_name))
            add("发布目标", self._workflow_publish_target_text(stage, metrics, context))
            add("源模型", metrics.get("source") or stage.get("source"))
            add("目标模型", metrics.get("destination") or stage.get("model_path") or context.get("published_model_path"))
            if stage.get("status") != "finished":
                add("临时目录", metrics.get("temp_dir") or stage.get("temp_dir"))
            add("状态", metrics.get("message") or stage.get("message"))
        elif stage_name == "deploy":
            add("模型名称", stage.get("model_name") or stage.get("model_path") or context.get("published_model_path"))
            add("目标模型路径", stage.get("model_path") or context.get("published_model_path"))
            add("旧模型", stage.get("old_model_name") or context.get("old_model_name"))
            add("部署步骤", self._workflow_deployment_step_text(stage.get("deployment_step") or metrics.get("deployment_step")))
            add("部署进度", stage.get("deploy_progress"))
            if stage.get("config_updated") is not None:
                add("配置修改", "已完成" if stage.get("config_updated") else "未完成")
            if stage.get("service_restarted") is not None:
                add("服务启动", "已完成" if stage.get("service_restarted") else "未完成")
            if stage.get("service_checked") is not None:
                add("服务检查", "已完成" if stage.get("service_checked") else "未完成")
            if stage.get("all_running") is not None:
                add("服务状态", "全部运行" if stage.get("all_running") else "未全部运行")
            add("服务日志", stage.get("service_log_command"))
            add("状态", stage.get("message"))
        elif stage_name == "benchmark":
            result = stage.get("result")
            if isinstance(result, dict):
                stage_status = stage.get("status")
                benchmark_status = (
                    stage_status
                    if stage_status not in {None, "", "finished"}
                    else result.get("status") or result.get("benchmark_status") or stage_status
                )
                add("评测状态", {
                    "pending": "等待中",
                    "awaiting_agent": "等待Agent执行",
                    "starting_external": "Agent启动中",
                    "starting": "启动中",
                    "running": "运行中",
                    "timeout": "超时",
                    "finished": "已完成",
                    "failed": "失败",
                    "stopped": "已停止",
                }.get(benchmark_status, benchmark_status))
                processed = result.get("processed")
                total = result.get("total")
                progress_percent = result.get("progress_percent")
                if processed is not None and total is not None:
                    progress_value = f"{processed}/{total}"
                    if progress_percent is not None and str(progress_percent).strip() != "":
                        progress_value += f" ({progress_percent}%)"
                else:
                    progress_value = result.get("progress") or progress_percent
                add("进度", progress_value)
                def percent_text(value):
                    try:
                        if value is None or str(value).strip() == "":
                            return None
                        percent = float(str(value).strip().rstrip("%"))
                        if "%" not in str(value) and 0 <= percent <= 1:
                            percent *= 100
                        return f"{percent:.1f}%"
                    except (TypeError, ValueError):
                        return str(value)

                correct = result.get("correct")
                accuracy = result.get("accuracy")
                invalid = result.get("invalid")
                invalid_rate = result.get("invalid_rate")
                avg_f1 = result.get("avg_f1")
                if correct is not None:
                    correct_value = str(correct)
                    if processed is not None:
                        correct_value += f"/{processed}"
                    add("正确数", correct_value)
                add("准确率", percent_text(accuracy))
                if invalid is not None:
                    invalid_value = str(invalid)
                    invalid_rate_text = percent_text(invalid_rate)
                    if invalid_rate_text:
                        invalid_value += f" ({invalid_rate_text})"
                    add("无效数", invalid_value)
                add("平均F1", avg_f1)
                executed = result.get("executed")
                passed = result.get("passed")
                failed = result.get("failed")
                timeout_count = result.get("timeout")
                executor_error = result.get("executor_error")
                pass_at_1 = result.get("pass@1")
                add("执行数", executed)
                if passed is not None:
                    passed_value = str(passed)
                    if executed is not None:
                        passed_value += f"/{executed}"
                    add("通过数", passed_value)
                add("失败数", failed)
                add("超时数", timeout_count)
                add("执行器错误", executor_error)
                add("pass@1", percent_text(pass_at_1))
                add("任务ID", result.get("benchmark_job_id"))
                add("查询命令", result.get("status_command"))
                add("停止命令", result.get("stop_command"))
                add("日志命令", result.get("log_command"))
                add("日志目录", result.get("log_dir") or result.get("folder_path"))
                add("运行日志", result.get("log_file"))
                add("结果文件", result.get("result_path"))
                add("结果入口", result.get("result_entry") or result.get("resultEntry"))
                if result.get("inference_service_stopped") is not None:
                    add("关闭服务", "已完成" if result.get("inference_service_stopped") else "未完成")
                add("关闭服务命令", result.get("inference_service_stop_command"))
                add("关闭服务日志", result.get("inference_service_log_command"))
                add("关闭服务错误", self._workflow_short_display_text(result.get("inference_service_stop_error")))
                if stage_status in {"running", "finished"} and (
                    stage_status != "finished" or benchmark_status not in {None, "finished"}
                ):
                    add("状态", self._workflow_short_display_text(result.get("message") or result.get("result")))
            else:
                add("评测状态", stage.get("status"))
                add("结果入口", _benchmark_result_entry(workflow))
                add("状态", self._workflow_short_display_text(result))
        if stage.get("status") == "timeout":
            add("状态", stage.get("message") or "本次状态查询超时，任务可能仍在运行，将继续自动刷新")
            add("最近查询错误", self._workflow_short_display_text(stage.get("last_poll_error")))
        add("下一阶段", self._workflow_next_stage_display_text(workflow))
        return details

    def _workflow_dataset_name_need_input_protocol(
        self,
        message: str,
        dataset_ref: str,
        display: str,
    ) -> Optional[Dict[str, Any]]:
        values = self._extract_named_param_values(message)
        launch_mode = "multinode" if self._is_multinode_training_request(message) else "single"
        container = self._extract_container_override(message) or (
            self.multinode_training_container
            if launch_mode == "multinode"
            else self.training_container
        )
        dataset_dir = values.get("dataset_dir") or f"/home/workspace/dataset_daily_train/{dataset_ref}"
        candidates = self._list_dataset_candidates_in_container(container, dataset_dir)
        options = self._sanitize_dataset_name_options(candidates[:30])
        if not options:
            return None
        known_params = {
            key: value
            for key, value in {
                "model_path": values.get("model_path") or self._find_default_base_model_path(container),
                "dataset_dir": dataset_dir,
            }.items()
            if value
        }
        protocol = self._with_protocol(
            "need_input",
            "trainer",
            display,
            kind="training_params",
            title=self._need_input_title("training_params"),
            requiredParams=["model_path", "dataset_dir", "dataset_name"],
            missingParams=["dataset_name"],
            options=options,
            knownParams=known_params or None,
            action="collect_workflow_params",
            status="needs_input",
            jobType="workflow_train",
            trainType="enhanced",
            trainTypeText="增强训练",
            workflowAction="start",
            datasetRef=dataset_ref,
            container=container,
        )
        return self._normalize_protocol(protocol, display)
    def _workflow_training_context(self, message: str, dataset_ref: str) -> Dict[str, Any]:
        requested_train_type = self._infer_train_type(message)
        benchmark_name = self._extract_workflow_benchmark_name(message)
        launch_mode = "multinode" if self._is_multinode_training_request(message) else "single"
        container = self._extract_container_override(message)
        request_container = container or (
            self.multinode_training_container
            if launch_mode == "multinode"
            else self.training_container
        )
        batch_path = f"/home/workspace/dataset_batch_train/{dataset_ref}"
        daily_path = f"/home/workspace/dataset_daily_train/{dataset_ref}"

        # A bare dataset identifier does not carry its training family. Locate it
        # first so one-click workflows route SFT to LoRA and DPO to enhanced
        # training instead of unconditionally defaulting to LoRA.
        if requested_train_type is None:
            batch_exists = self._docker_path_exists(request_container, batch_path)
            daily_exists = self._docker_path_exists(request_container, daily_path)
            if batch_exists is True and daily_exists is True:
                raise ValueError(
                    "同一数据标识同时存在于 SFT 和 DPO 目录，无法安全判断训练类型："
                    f"{batch_path}；{daily_path}。"
                    "请明确指定 lora训练 或 增强训练。"
                )
            if daily_exists is True:
                train_type = "enhanced"
            elif batch_exists is True:
                train_type = "lora"
            else:
                raise ValueError(
                    "训练数据目录不存在。已检查："
                    f"{batch_path}（SFT）；{daily_path}（DPO）"
                    f"（容器：{request_container}）。"
                )
        else:
            train_type = requested_train_type

        if train_type != "enhanced":
            validation_issue = validate_training_inputs_preflight(
                "batch_train_lora",
                request_container,
                {"DATASET_DIR": batch_path, "DATASET_DATE": dataset_ref},
            )
            if validation_issue:
                raise ValueError(str(validation_issue["message"]))
            context = {
                "train_type": "lora",
                "benchmark": benchmark_name,
                "evaluation_dataset_name": benchmark_name,
                "training_container": self.training_container,
                "evaluation_container": self.evaluation_container,
                "grpo_container": self.grpo_container,
                "multinode_training_container": self.multinode_training_container,
                "resource_group_id": _current_resource_group_id().strip(),
                "training_pool_id": _current_training_pool_id().strip(),
                "train_args": self._workflow_batch_train_args(message),
            }
            if launch_mode == "multinode":
                context["launch_mode"] = "multinode"
                context["train_args"].update(self._extract_multinode_cli_args(message))
            if container:
                context["container"] = container
            return context
        values = self._extract_named_param_values(message)
        dataset_dir = values.get("dataset_dir") or daily_path
        dataset_name = values.get("dataset_name") or self._infer_dataset_name_from_dir(dataset_dir, container=request_container)
        if not dataset_name:
            candidates = self._list_dataset_candidates_in_container(request_container, dataset_dir)
            raise ValueError(
                "无法唯一确定增强训练 dataset_name。"
                f"当前候选：{', '.join(candidates[:8]) if candidates else '无'}。"
                "请在口令中补充 dataset_name=<数据集名称>。"
            )
        model_path = values.get("model_path") or self._find_default_base_model_path(request_container)
        if not model_path:
            raise ValueError("无法自动确定增强训练基础模型。请在口令中补充 model_path=<模型路径>。")
        context =  {
            "train_type": "enhanced",
            "benchmark": benchmark_name,
            "evaluation_dataset_name": benchmark_name,
            "training_container": self.training_container,
            "evaluation_container": self.evaluation_container,
            "grpo_container": self.grpo_container,
            "multinode_training_container": self.multinode_training_container,
            "resource_group_id": _current_resource_group_id().strip(),
            "training_pool_id": _current_training_pool_id().strip(),
            "model_path": model_path,
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
        }
        if launch_mode == "multinode":
            context["launch_mode"] = "multinode"
            context["train_args"] = self._extract_multinode_cli_args(message)
        if container:
            context["container"] = container
        validation_issue = validate_training_inputs_preflight(
            MULTINODE_DPO_SCRIPT if launch_mode == "multinode" else "dpo_train_launcher",
            request_container,
            {},
            {
                "model_path": model_path,
                "dataset_dir": dataset_dir,
                "dataset_name": dataset_name,
            },
        )
        if validation_issue:
            raise ValueError(str(validation_issue["message"]))
        return context

    def _workflow_batch_train_args(self, message: str) -> Dict[str, str]:
        """Extract optional LoRA batch-training overrides from a one-click command."""
        text = (message or "").replace("，", ",").replace("：", ":")
        extracted: Dict[str, str] = {}
        explicit_patterns = {
            "LOCALHOST_ID": r"\bLOCALHOST_ID\s*=\s*([0-9,\s]+)",
            "MBS": r"\bMBS\s*=\s*([0-9]+)",
            "ACC": r"\bACC\s*=\s*([0-9]+)",
            "LR": r"\bLR\s*=\s*([0-9.eE+-]+)",
            "TEM": r"\bTEM\s*=\s*([A-Za-z0-9_.-]+)",
            "RESUME": r"\bRESUME\s*=\s*([^\s,;]+)",
            "model_path": r"\b(?:MODEL_PATH|model_path|base_model_path)\s*=\s*(/[^\s,，;；]+)",
            "container": r"\b(?:container|docker_container)\s*=\s*([A-Za-z0-9_.-]+)",
        }
        for key, pattern in explicit_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()

        if "LOCALHOST_ID" not in extracted:
            gpu_match = re.search(
                r"(?:用|使用|指定)?\s*([0-9]+(?:\s*,\s*[0-9]+)+)\s*(?:这)?(?:几|[一二三四五六七八九十\d]+)?\s*张?(?:卡|显卡|gpu)",
                text,
                re.IGNORECASE,
            ) or re.search(
                r"(?:显卡|卡号|gpu|设备)\s*(?:是|为|=|:)?\s*([0-9]+(?:\s*,\s*[0-9]+)*)",
                text,
                re.IGNORECASE,
            )
            if gpu_match:
                extracted["LOCALHOST_ID"] = gpu_match.group(1)

        chinese_patterns = {
            "MBS": r"(?:批量大小|批次大小|批大小|mbs)\s*(?:是|为|=|:)?\s*([0-9]+)",
            "ACC": r"(?:梯度累积|累积步数|acc)\s*(?:是|为|=|:)?\s*([0-9]+)",
            "LR": r"(?:学习率|学习速率|lr)\s*(?:是|为|=|:)?\s*([0-9.eE+-]+)",
            "TEM": r"(?:模型模板|模型类别|tem)\s*(?:是|为|=|:)?\s*([A-Za-z0-9_.-]+)",
            "model_path": r"(?:模型路径|模型位置|基础模型路径|模型在)\s*(?:是|为|=|:)?\s*(/[^\s,，;；]+)",
            "container": r"(?:容器|docker)\s*(?:是|为|=|:)?\s*([A-Za-z0-9_.-]+)",
        }
        for key, pattern in chinese_patterns.items():
            if key in extracted:
                continue
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()

        if "LOCALHOST_ID" in extracted:
            gpu_ids = [
                value.strip()
                for value in extracted["LOCALHOST_ID"].split(",")
                if value.strip()
            ]
            if not gpu_ids or any(not value.isdigit() for value in gpu_ids):
                raise ValueError("LOCALHOST_ID 格式不正确，请使用逗号分隔的卡号，例如 LOCALHOST_ID=4,5,6,7")
            extracted["LOCALHOST_ID"] = ",".join(gpu_ids)
        return extracted

    def _workflow_trainer_result(self, response: ToolResponse) -> Dict[str, Any]:
        metadata = response.metadata or {}
        protocol = metadata.get("protocol") if isinstance(metadata.get("protocol"), dict) else {}
        response_msg = metadata.get("response_msg")
        display = (
            response_msg.get_text_content()
            if isinstance(response_msg, Msg)
            else self._response_to_text(response)
        ) or ""
        return {
            "display": display.strip(),
            "pid": protocol.get("pid"),
            "container": protocol.get("container"),
            "status": protocol.get("status"),
            "failed": (
                protocol.get("type") in {"job_failed", "error"}
                or protocol.get("status") == "failed"
                or metadata.get("success") is False
            ),
            "protocol": protocol,
            "response_msg": response_msg,
        }

    async def _workflow_inference_event(
        self,
        command: str,
        workflow: Optional[Dict[str, Any]] = None,
        stage_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = workflow.get("context") if isinstance(workflow, dict) else {}
        context = context if isinstance(context, dict) else {}
        owner_kwargs = _workflow_inference_owner_kwargs(context) if workflow else {}
        role_value = str(context.get("user_role") or _current_user_role()).strip().lower()
        resource_group_token = training_pool_token = user_role_token = None
        owner_user_id_token = owner_aliases_token = None
        try:
            if workflow:
                resource_group_token = REQUEST_RESOURCE_GROUP_ID.set(
                    str(context.get("resource_group_id") or "").strip()
                )
                training_pool_token = REQUEST_TRAINING_POOL_ID.set(
                    str(context.get("training_pool_id") or "").strip()
                )
                user_role_token = REQUEST_USER_ROLE.set(role_value)
                owner_user_id_token = REQUEST_INFERENCE_OWNER_USER_ID.set(
                    str(owner_kwargs.get("owner_user_id") or "").strip()
                )
                owner_aliases_token = REQUEST_INFERENCE_OWNER_ALIASES.set(
                    tuple(_owner_aliases_from_any(owner_kwargs.get("owner_aliases")))
                )
            response = await self._call_inference(command, command=command)
        finally:
            if owner_aliases_token is not None:
                REQUEST_INFERENCE_OWNER_ALIASES.reset(owner_aliases_token)
            if owner_user_id_token is not None:
                REQUEST_INFERENCE_OWNER_USER_ID.reset(owner_user_id_token)
            if user_role_token is not None:
                REQUEST_USER_ROLE.reset(user_role_token)
            if training_pool_token is not None:
                REQUEST_TRAINING_POOL_ID.reset(training_pool_token)
            if resource_group_token is not None:
                REQUEST_RESOURCE_GROUP_ID.reset(resource_group_token)
        metadata = response.metadata or {}
        protocol = metadata.get("protocol") if isinstance(metadata.get("protocol"), dict) else {}
        inference_payload = metadata.get("inference_payload") if isinstance(metadata.get("inference_payload"), dict) else {}
        resource_context = (
            inference_payload.get("resource_context")
            if isinstance(inference_payload.get("resource_context"), dict)
            else None
        )
        text = self._response_to_text(response) or ""
        protocol_type = str(protocol.get("type") or "")
        protocol_status = str(protocol.get("status") or "").lower()
        has_structured_protocol = bool(protocol_type or protocol_status)
        failed = (
            metadata.get("success") is False
            or protocol_type in {"job_failed", "error"}
            or protocol_status == "failed"
        )
        if not has_structured_protocol and metadata.get("success") is not False:
            failed = _workflow_inference_has_failure_signal(text)
        commands = protocol.get("commands") if isinstance(protocol.get("commands"), list) else []
        result = {
            "success": not failed,
            "text": text,
            "protocol": protocol,
            "commands": commands,
            "status": protocol.get("status"),
            "model_path": _workflow_inference_model_path(protocol),
            "model_name": _workflow_inference_model_name(protocol),
            "all_running": _workflow_all_running_from_payload({"protocol": protocol}, text) is True,
        }
        if resource_context:
            result["resource_context"] = resource_context
        if workflow and stage_name:
            result.update(
                _append_workflow_stage_log(
                    workflow,
                    stage_name,
                    command,
                    result,
                )
            )
        return result

    async def _workflow_ensure_benchmark_inference_ready_event(
        self,
        workflow: Dict[str, Any],
        stage_name: str = "benchmark",
    ) -> Dict[str, Any]:
        context = workflow.get("context") if isinstance(workflow.get("context"), dict) else {}
        resource_group_id = str(context.get("resource_group_id") or "").strip()
        training_pool_id = str(context.get("training_pool_id") or "").strip()
        resource_context = _workflow_context_inference_resource_context(context)

        status_result = await self._workflow_inference_event("查看推理服务状态", workflow, stage_name)
        if _workflow_inference_service_matches_resource(
            status_result,
            resource_context,
            resource_group_id=resource_group_id,
            training_pool_id=training_pool_id,
        ):
            return status_result

        restarted = await self._workflow_inference_event("重启推理服务", workflow, stage_name)
        if not restarted.get("success"):
            raise RuntimeError(restarted.get("text") or "推理服务重启失败")
        resource_context = restarted.get("resource_context") if isinstance(restarted.get("resource_context"), dict) else resource_context
        if isinstance(resource_context, dict):
            context["inference_resource_context"] = resource_context

        for _ in range(12):
            await asyncio.sleep(5)
            check = await self._workflow_inference_event("查看推理服务状态", workflow, stage_name)
            if _workflow_inference_service_matches_resource(
                check,
                resource_context,
                resource_group_id=resource_group_id,
                training_pool_id=training_pool_id,
            ):
                return check
        raise RuntimeError("推理服务启动超时，请检查容器日志和资源配置")

    async def handle_workflow_event(self, event: Dict[str, Any]) -> None:
        """Execute state-machine events through the existing per-user agents."""
        manager = get_workflow_manager()
        workflow = manager.get(event["workflow_id"])
        event_type = event["event_type"]
        logger.debug(
            "[workflow-debug] handle event workflow_id=%s event_type=%s workflow_status=%s current_stage=%s stage_status=%s",
            event.get("workflow_id"),
            event_type,
            workflow.get("status") if workflow else None,
            workflow.get("current_stage") if workflow else None,
            ((workflow.get("stages") or {}).get(workflow.get("current_stage"), {}) if workflow else {}).get("status"),
        )
        if event_type == "notify_terminal":
            if workflow and workflow.get("status") in {"finished", "failed", "stopped"}:
                display = self._workflow_display_text(workflow)
                protocol = self._workflow_protocol(workflow, display, agent="orchestrator")
                self.last_response_protocol = protocol
                if self.orchestrator:
                    await self._safe_agent_print(
                        self.orchestrator,
                        self._agent_msg(self.orchestrator.name, display, protocol),
                    )
            return
        if not workflow or workflow.get("status") != "running":
            return
        async def push_workflow_update(workflow_snapshot: Dict[str, Any], started: bool = False) -> None:
            display = self._workflow_display_text(workflow_snapshot)
            protocol = self._workflow_protocol(
                workflow_snapshot,
                display,
                started=started,
                agent="orchestrator",
            )
            self.last_response_protocol = protocol
            if self.orchestrator:
                await self._safe_agent_print(
                    self.orchestrator,
                    self._agent_msg(self.orchestrator.name, display, protocol),
                )

        def require_running() -> None:
            if not manager.is_running(event["workflow_id"]):
                raise RuntimeError("一键工作流已停止，取消后续阶段操作")
        def mark_external_starting() -> bool:
            updated = manager.record_external_stage_starting(event["workflow_id"])
            logger.debug(
                "[workflow-debug] event mark external starting workflow_id=%s event_type=%s updated_status=%s current_stage=%s stage_status=%s",
                event.get("workflow_id"),
                event_type,
                updated.get("status") if updated else None,
                updated.get("current_stage") if updated else None,
                ((updated.get("stages") or {}).get(updated.get("current_stage"), {}) if updated else {}).get("status"),
            )
            return bool(updated and updated.get("status") == "running")
        if event_type == "start_evaluate":
            model_fir = workflow["context"]["trained_model_path"]
            manager.activate_worker_for(workflow["workflow_id"])
            if not mark_external_starting():
                return
            task = (
                "执行单模型评估。不要询问评估类型，立即调用单模型评估工具。"
                f"必须使用 model_fir={model_fir}。"
            )
            previous_child_context = self._workflow_child_context
            self._workflow_child_context = {
                "workflow_id": workflow["workflow_id"],
                "stage": "evaluate",
                "workflow": workflow,
            }
            evaluation_container = _workflow_single_model_evaluation_container(workflow)
            training_container_token = REQUEST_TRAINING_CONTAINER.set(evaluation_container)
            evaluation_token = REQUEST_EVALUATION_CONTAINER.set(evaluation_container)
            resource_group_token = REQUEST_RESOURCE_GROUP_ID.set(
                str((workflow.get("context") or {}).get("resource_group_id") or "").strip()
            )
            training_pool_token = REQUEST_TRAINING_POOL_ID.set(
                str((workflow.get("context") or {}).get("training_pool_id") or "").strip()
            )
            try:
                result = self._workflow_trainer_result(await self._call_evaluator(task))
            finally:
                REQUEST_TRAINING_POOL_ID.reset(training_pool_token)
                REQUEST_RESOURCE_GROUP_ID.reset(resource_group_token)
                REQUEST_EVALUATION_CONTAINER.reset(evaluation_token)
                REQUEST_TRAINING_CONTAINER.reset(training_container_token)
                self._workflow_child_context = previous_child_context
            require_running()
            if not result.get("pid") or not result.get("container"):
                raise RuntimeError(result.get("display") or "evaluator 未返回评估 PID 或容器名称")
            workflow = manager.record_running_stage(
                workflow["workflow_id"],
                pid=result["pid"],
                container=result["container"],
                message=result["display"],
            )
            manager.activate_worker_for(workflow["workflow_id"])
            await push_workflow_update(workflow, started=True)
            return
        if event_type == "deploy_model":
            manager.activate_worker_for(workflow["workflow_id"])
            if not mark_external_starting():
                return
            published_path = workflow["context"]["published_model_path"]
            model_name = os.path.basename(published_path)
            current = await self._workflow_inference_event("查看推理配置", workflow, "deploy")
            require_running()
            config_matches_target = _workflow_inference_matches_target(current, published_path)
            old_model_name = current.get("model_name")
            old_model_name_parsed = old_model_name is not None
            if not old_model_name_parsed:
                old_model_name = os.path.basename(current.get("model_path") or "")
                old_model_name_parsed = bool(old_model_name)
            workflow = manager.record_running_stage(
                workflow["workflow_id"],
                model_name=model_name,
                model_path=published_path,
                old_model_name=old_model_name,
                config_update_skipped=config_matches_target,
                deployment_step="checking_config",
                deploy_progress="1/4",
                service_log_command="查看推理服务日志",
                message="当前推理配置已指向目标模型" if config_matches_target else "正在读取当前推理配置",
                **_workflow_stage_log_fields(workflow, "deploy"),
            )
            await push_workflow_update(workflow)
            if config_matches_target:
                checked = await self._workflow_inference_event("查看推理服务状态", workflow, "deploy")
                require_running()
                all_running = checked.get("all_running") is True
                if all_running:
                    manager.record_finished_stage(
                        workflow["workflow_id"],
                        model_name=model_name,
                        model_path=published_path,
                        config_update_skipped=True,
                        service_checked=True,
                        all_running=True,
                        service_log_command="查看推理服务日志",
                        message="当前推理服务已部署目标模型，跳过部署",
                        **_workflow_stage_log_fields(workflow, "deploy"),
                    )
                    workflow = manager.advance(workflow["workflow_id"])
                    await push_workflow_update(workflow)
                    return
                workflow = manager.record_running_stage(
                    workflow["workflow_id"],
                    model_name=model_name,
                    model_path=published_path,
                    old_model_name=old_model_name,
                    config_update_skipped=True,
                    config_updated=True,
                    service_checked=True,
                    all_running=False,
                    deployment_step="restarting_service",
                    deploy_progress="3/4",
                    service_log_command="查看推理服务日志",
                    message="推理配置已指向目标模型，正在重启推理服务",
                    **_workflow_stage_log_fields(workflow, "deploy"),
                )
                await push_workflow_update(workflow)
                restarted = await self._workflow_inference_event("重启推理服务", workflow, "deploy")
                require_running()
                if not restarted["success"]:
                    raise RuntimeError(restarted.get("text") or "推理服务重启失败")
                workflow = manager.record_running_stage(
                    workflow["workflow_id"],
                    model_name=model_name,
                    model_path=published_path,
                    old_model_name=old_model_name,
                    config_update_skipped=True,
                    config_updated=True,
                    service_restarted=True,
                    service_checked=False,
                    all_running=False,
                    deployment_step="waiting_service",
                    deploy_progress="4/4",
                    next_service_check_at=time.time() + config.workflow.poll_interval,
                    service_log_command="查看推理服务日志",
                    message="推理服务重启命令已提交，等待服务全部运行",
                    **_workflow_inference_resource_result_fields(restarted),
                    **_workflow_stage_log_fields(workflow, "deploy"),
                )
                manager.activate_worker_for(workflow["workflow_id"])
                await push_workflow_update(workflow)
                return
            if not old_model_name_parsed:
                raise RuntimeError("无法从推理配置中解析旧 MODEL_NAME，已停止部署以避免无法回滚")
            workflow = manager.update_context(workflow["workflow_id"], old_model_name=old_model_name)
            workflow = manager.record_running_stage(
                workflow["workflow_id"],
                model_name=model_name,
                model_path=published_path,
                old_model_name=old_model_name,
                config_update_skipped=False,
                deployment_step="updating_config",
                deploy_progress="2/4",
                service_log_command="查看推理服务日志",
                message=f"已读取当前配置，正在修改 MODEL_NAME={model_name}",
                **_workflow_stage_log_fields(workflow, "deploy"),
            )
            await push_workflow_update(workflow)
            try:
                updated = await self._workflow_inference_event(
                    f"修改推理配置 MODEL_NAME={model_name}",
                    workflow,
                    "deploy",
                )
                require_running()
                workflow = manager.record_running_stage(
                    workflow["workflow_id"],
                    model_name=model_name,
                    model_path=published_path,
                    old_model_name=old_model_name,
                    config_update_skipped=False,
                    config_updated=updated["success"],
                    deployment_step="restarting_service",
                    deploy_progress="3/4",
                    service_log_command="查看推理服务日志",
                    message="推理配置已修改，正在重启推理服务",
                    **_workflow_stage_log_fields(workflow, "deploy"),
                )
                await push_workflow_update(workflow)
                restarted = await self._workflow_inference_event("重启推理服务", workflow, "deploy")
                require_running()
                if not updated["success"] or not restarted["success"]:
                    raise RuntimeError(restarted.get("text") or updated.get("text") or "推理服务重启失败")
                workflow = manager.record_running_stage(
                    workflow["workflow_id"],
                    model_name=model_name,
                    model_path=published_path,
                    old_model_name=old_model_name,
                    config_update_skipped=False,
                    config_updated=True,
                    service_restarted=True,
                    service_checked=False,
                    all_running=False,
                    deployment_step="waiting_service",
                    deploy_progress="4/4",
                    next_service_check_at=time.time() + config.workflow.poll_interval,
                    service_log_command="查看推理服务日志",
                    message="推理服务重启命令已提交，等待服务全部运行",
                    **_workflow_inference_resource_result_fields(restarted),
                    **_workflow_stage_log_fields(workflow, "deploy"),
                )
                manager.activate_worker_for(workflow["workflow_id"])
                await push_workflow_update(workflow)
                return
            except Exception:
                if old_model_name_parsed:
                    await self._workflow_inference_event(
                        f"修改推理配置 MODEL_NAME={old_model_name}",
                        workflow,
                        "deploy",
                    )
                    await self._workflow_inference_event("重启推理服务", workflow, "deploy")
                raise
        if event_type == "start_benchmark":
            if not mark_external_starting():
                return
            start_command = _benchmark_start_command(workflow)
            logger.debug(
                "[workflow-debug] start benchmark event workflow_id=%s command=%s",
                workflow.get("workflow_id"),
                start_command,
            )
            await self._workflow_ensure_benchmark_inference_ready_event(workflow, "benchmark")
            require_running()
            result = await self._workflow_inference_event(start_command, workflow, "benchmark")
            logger.debug(
                "[workflow-debug] start benchmark result workflow_id=%s success=%s status=%s commands=%s text_head=%s",
                workflow.get("workflow_id"),
                result.get("success"),
                result.get("status"),
                result.get("commands"),
                str(result.get("text") or "")[:300],
            )
            require_running()
            if not result["success"]:
                raise RuntimeError(result.get("text") or f"{_benchmark_name(workflow)}基准评测启动失败")
            benchmark_result = _benchmark_enrich_runtime_result({
                "status": result.get("status") or "running",
                "message": result.get("text"),
                "commands": result.get("commands"),
                "result_entry": _benchmark_result_entry(workflow),
            })
            if benchmark_result.get("status_command") and not benchmark_result.get("log_command"):
                benchmark_result["log_command"] = benchmark_result["status_command"]
            logger.debug(
                "[workflow-debug] parsed benchmark start workflow_id=%s benchmark_status=%s job_id=%s status_command=%s stop_command=%s log_command=%s",
                workflow.get("workflow_id"),
                benchmark_result.get("status"),
                benchmark_result.get("benchmark_job_id"),
                benchmark_result.get("status_command"),
                benchmark_result.get("stop_command"),
                benchmark_result.get("log_command"),
            )
            workflow = manager.record_running_stage(
                workflow["workflow_id"],
                result=benchmark_result,
                **_workflow_stage_log_fields(workflow, "benchmark"),
            )
            manager.activate_worker_for(workflow["workflow_id"])
            workflow = manager.poll_running_stage(workflow["workflow_id"])
            logger.debug(
                "[workflow-debug] post-start benchmark poll workflow_id=%s workflow_status=%s current_stage=%s stage_status=%s",
                workflow.get("workflow_id"),
                workflow.get("status"),
                workflow.get("current_stage"),
                ((workflow.get("stages") or {}).get(workflow.get("current_stage"), {})).get("status"),
            )
            await push_workflow_update(workflow)
            return
        raise ValueError(f"不支持的工作流事件: {event_type}")

    async def _start_workflow_via_trainer(self, manager: WorkflowManager, workflow: Dict[str, Any]) -> str:
        dataset_ref = workflow["dataset_ref"]
        workflow = manager.skip_existing_stage_output(workflow["workflow_id"])
        if workflow.get("current_stage") != "train" or (workflow.get("stages") or {}).get("train", {}).get("status") != "pending":
            workflow = await self._advance_skipped_workflow_start(manager, workflow)
            display = self._workflow_display_text(workflow)
            protocol = self._workflow_protocol(workflow, display, started=False, agent="orchestrator")
            self.last_response_protocol = protocol
            response_msg = self._agent_msg(
                self.orchestrator.name if self.orchestrator else "Orchestrator",
                display,
                protocol,
            )
            if self.orchestrator:
                await self._safe_agent_print(self.orchestrator, response_msg)
            return self._protocol_json_response(display, protocol)
        workflow = manager.record_external_stage_starting(workflow["workflow_id"])
        if not workflow or workflow.get("status") != "running":
            display = "一键工作流已停止，取消启动训练"
            protocol = self._workflow_protocol(workflow, display, agent="orchestrator") if workflow else self._with_protocol(
                "workflow_stopped",
                "orchestrator",
                display,
            )
            self.last_response_protocol = protocol
            return self._protocol_json_response(display, protocol)
        try:
            train_payload = _workflow_start_train(workflow)
        except Exception as exc:
            train_payload = {
                "failed": True,
                "display": str(exc) or "LoRA训练启动失败",
            }
        result = {
            "display": str(train_payload.get("message") or train_payload.get("display") or "").strip(),
            "pid": train_payload.get("pid"),
            "container": train_payload.get("container"),
            "status": train_payload.get("status"),
            "failed": bool(train_payload.get("failed")),
        }
        if not manager.is_running(workflow["workflow_id"]):
            workflow = manager.get(workflow["workflow_id"]) or workflow
            display = "一键工作流已停止，训练启动结果不再写入工作流"
            protocol = self._workflow_protocol(workflow, display, agent="trainer")
            self.last_response_protocol = protocol
            return self._protocol_json_response(display, protocol)
        if result.get("failed") or not result.get("pid") or not result.get("container"):
            failure = result.get("display") or "trainer 未返回训练 PID 或容器名称"
            workflow = manager.fail_current_stage(workflow["workflow_id"], failure)
            protocol = self._workflow_protocol(workflow, failure, agent="trainer")
            self.last_response_protocol = protocol
            response_msg = self._agent_msg(
                self.agents["trainer"].name,
                f"一键工作流启动失败：{failure}",
                protocol,
            )
            await self._safe_agent_print(self.agents["trainer"], response_msg)
            return self._protocol_json_response(failure, protocol)
        record_stage = (
            manager.record_preparing_stage
            if result.get("status") == "preparing"
            else manager.record_running_stage
        )
        workflow = record_stage(
            workflow["workflow_id"],
            pid=result["pid"],
            container=result["container"],
            message=result["display"],
            metrics={},
        )
        manager.activate_worker_for(workflow["workflow_id"])
        # The trainer result only describes the launched Docker process.  For a
        # one-click workflow the user needs the whole pipeline state immediately:
        # the current training stage, every pending stage, and what comes next.
        display = self._workflow_display_text(workflow)
        protocol = self._workflow_protocol(workflow, display, started=True, agent="trainer")
        self.last_response_protocol = protocol
        response_msg = self._agent_msg(
            self.agents["trainer"].name,
            (
                "一键工作流已提交，训练正在准备数据集和模型，尚未确认启动。"
                if result.get("status") == "preparing"
                else "一键工作流已启动，状态栏将自动跟踪训练、评估、发布、部署和基准评测进度。"
            ),
            protocol,
        )
        await self._safe_agent_print(self.agents["trainer"], response_msg)
        return self._protocol_json_response(display, protocol)

    async def _advance_skipped_workflow_start(
        self,
        manager: WorkflowManager,
        workflow: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synchronously start the next real stage after start-time skips."""
        workflow_id = workflow["workflow_id"]
        inline_consumer_id = f"{_WORKFLOW_EVENT_CONSUMER_ID}-inline"
        for _ in range(len(("train", "evaluate", "publish", "deploy", "benchmark")) * 3):
            workflow = manager.advance(workflow_id)
            if not workflow or workflow.get("status") != "running":
                break

            handled_event = False
            events = manager.claim_pending_events(
                inline_consumer_id,
                limit=3,
                workflow_id=workflow_id,
            )
            for event in events:
                handled_event = True
                consumer_id = str(event.get("claimed_by") or inline_consumer_id)
                try:
                    await self.handle_workflow_event(event)
                    manager.complete_event(event["event_id"], consumer_id)
                except Exception as exc:
                    logger.exception("Inline workflow event %s failed", event.get("event_id"))
                    manager.fail_current_stage(workflow_id, str(exc))
                    manager.fail_event(event["event_id"], consumer_id, str(exc))
                    return manager.get(workflow_id) or workflow

            workflow = manager.get(workflow_id) or workflow
            if handled_event:
                continue
            stage_name = workflow.get("current_stage")
            stage = (workflow.get("stages") or {}).get(stage_name, {})
            if stage.get("status") in {"preparing", "running", "starting", "starting_external", "awaiting_agent"}:
                break
        return manager.get(workflow_id) or workflow

    def _workflow_prepare_train_auto_resume(
        self,
        manager: WorkflowManager,
        workflow: Dict[str, Any],
    ) -> Dict[str, Any]:
        if workflow.get("current_stage") != "train":
            return workflow
        train_stage = (workflow.get("stages") or {}).get("train", {})
        if train_stage.get("status") != "pending":
            return workflow
        context = workflow.get("context") or {}
        train_args = dict(context.get("train_args") or {})
        if str(context.get("train_type") or "").lower() == "enhanced":
            train_args.pop("RESUME", None)
            return manager.update_context(
                workflow["workflow_id"],
                train_args=train_args,
                auto_resume_checkpoint="",
                auto_resume_checkpoint_status="unsupported",
            )
        existing_resume = str(train_args.get("RESUME") or "").strip()
        previous_auto_resume = str(context.get("auto_resume_checkpoint") or "").strip()
        if existing_resume and existing_resume != previous_auto_resume:
            return workflow
        checkpoint = _workflow_find_latest_checkpoint(workflow)
        if checkpoint:
            train_args["RESUME"] = checkpoint
            return manager.update_context(
                workflow["workflow_id"],
                train_args=train_args,
                auto_resume_checkpoint=checkpoint,
                auto_resume_checkpoint_status="found",
            )
        return manager.update_context(
            workflow["workflow_id"],
            auto_resume_checkpoint="",
            auto_resume_checkpoint_status="not_found",
        )

    async def _workflow_control(self, action: str) -> ToolResponse:
        """Handle persisted workflow controls when invoked by the orchestrator agent."""
        manager = get_workflow_manager()
        raw_action = (action or "").strip()
        action_parts = raw_action.split(":", 1)
        normalized_action = action_parts[0].strip().lower()
        requested_workflow_id = (
            self._extract_workflow_id(action_parts[1])
            if len(action_parts) > 1
            else self._extract_workflow_id(raw_action)
        )
        if normalized_action not in {
            "resume", "continue", "retry", "继续", "续跑", "恢复",
            "stop", "cancel", "停止", "结束", "取消",
            "status", "query", "view", "状态", "查询", "查看",
        }:
            normalized_action = parse_workflow_control_command(raw_action) or normalized_action
        resumed_in_background = False
        workflow_resource_group_id = _current_resource_group_id().strip()
        is_admin_group_workflow = bool(
            _current_user_role() == "admin" and workflow_resource_group_id
        )
        try:
            if normalized_action in {"resume", "continue", "retry", "继续", "续跑", "恢复"}:
                if requested_workflow_id:
                    workflow = manager.get_for_user_family(
                        requested_workflow_id,
                        self.user_id,
                        workflow_resource_group_id,
                        permission_message="无权继续指定的一键工作流",
                        include_group_users=is_admin_group_workflow,
                    )
                    if not workflow:
                        raise ValueError(f"一键工作流不存在: {requested_workflow_id}")
                    if workflow.get("status") in {"failed", "stopped"}:
                        workflow = manager.resume_by_id(
                            requested_workflow_id,
                            self.user_id,
                            workflow_resource_group_id,
                            include_group_users=is_admin_group_workflow,
                        )
                    elif workflow.get("status") != "running":
                        raise ValueError("指定的一键工作流不是运行、失败或停止状态，无法继续")
                else:
                    active = manager.active_for_user_family(
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                    workflow = active if active else manager.resume(
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                manager.activate_worker_for(workflow["workflow_id"])
                train_stage = (workflow.get("stages") or {}).get("train", {})
                if (
                    workflow.get("current_stage") == "train"
                    and train_stage.get("status") == "pending"
                ):
                    workflow = self._workflow_prepare_train_auto_resume(manager, workflow)
                    await self._start_workflow_via_trainer(manager, workflow)
                    workflow = manager.get(workflow["workflow_id"]) or workflow
                else:
                    workflow = manager.advance(workflow["workflow_id"])
                resumed_in_background = True
            elif normalized_action in {"stop", "cancel", "停止", "结束", "取消"}:
                workflow = (
                    manager.begin_stop_by_id(
                        requested_workflow_id,
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                    if requested_workflow_id
                    else manager.begin_stop(
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                )
                try:
                    stop_result = await self._stop_workflow_runtime(workflow)
                    context_updates: Dict[str, Any] = {}
                    stop_fields: Dict[str, Any] = {}
                    if stop_result:
                        context_updates["stop_result"] = stop_result
                    if workflow.get("current_stage") == "benchmark":
                        stop_fields.update(
                            _append_workflow_stage_log(
                                workflow,
                                "benchmark",
                                "用户停止工作流时停止 benchmark 和推理服务",
                                {"result": stop_result},
                            )
                        )
                        service_log_fields = _append_workflow_stage_log(
                            workflow,
                            "stop_service",
                            "用户停止工作流时关闭推理服务",
                            {"result": stop_result},
                        )
                        stop_fields.update(
                            {
                                "stop_service_log_path": service_log_fields.get("log_path"),
                                "stop_service_log_tail": service_log_fields.get("log_tail"),
                                "stop_service_log_updated_at": service_log_fields.get("log_updated_at"),
                                "inference_service_stop_command": _workflow_inference_service_stop_command(),
                                "inference_service_log_command": "查看推理服务日志",
                            }
                        )
                    elif workflow.get("current_stage") == "deploy":
                        service_log_fields = _append_workflow_stage_log(
                            workflow,
                            "stop_service",
                            "用户停止工作流时关闭推理服务",
                            {"result": stop_result},
                        )
                        stop_fields.update(_workflow_stage_log_fields(workflow, "deploy"))
                        stop_fields.update(
                            {
                                "stop_service_log_path": service_log_fields.get("log_path"),
                                "stop_service_log_tail": service_log_fields.get("log_tail"),
                                "stop_service_log_updated_at": service_log_fields.get("log_updated_at"),
                                "inference_service_stop_command": _workflow_inference_service_stop_command(),
                                "inference_service_log_command": "查看推理服务日志",
                            }
                        )
                    if workflow.get("current_stage") == "benchmark":
                        context_updates["inference_service_stopped_by_benchmark_stop"] = True
                        context_updates["benchmark_service_stopped_at"] = time.time()
                    if context_updates:
                        manager.update_context(workflow["workflow_id"], **context_updates)
                    workflow = manager.complete_stop(workflow["workflow_id"], **stop_fields)
                except Exception as exc:
                    manager.cancel_stop(workflow["workflow_id"], str(exc))
                    raise
            elif normalized_action in {"status", "query", "view", "状态", "查询", "查看"}:
                if requested_workflow_id:
                    workflow = manager.get_for_user_family(
                        requested_workflow_id,
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                    if workflow:
                        workflow = manager.get_status_snapshot(workflow["workflow_id"])
                else:
                    workflow = manager.latest_for_user_family(
                        self.user_id,
                        workflow_resource_group_id,
                        include_group_users=is_admin_group_workflow,
                    )
                if not workflow:
                    raise ValueError(
                        f"未找到一键工作流：{requested_workflow_id}"
                        if requested_workflow_id
                        else "还没有可查询的一键工作流"
                    )
                if workflow.get("status") == "running":
                    workflow = manager.poll_running_stage(workflow["workflow_id"])
            else:
                raise ValueError(f"不支持的一键工作流操作：{action}")
            if resumed_in_background:
                workflow = self._workflow_resume_display_snapshot(workflow)
            display = self._workflow_display_text(workflow)
            if resumed_in_background and workflow.get("status") == "running":
                display = (
                    f"{display}\n\n"
                    "工作流已恢复，后续将在后台继续执行，可通过查看工作流状态刷新。"
                )
            protocol = self._workflow_protocol(workflow, display)
        except ValueError as exc:
            display = str(exc)
            protocol = self._with_protocol("workflow_failed", "orchestrator", display, error=display)
        except Exception as exc:
            logger.exception("Workflow control failed")
            display = f"一键工作流操作失败：{exc}"
            protocol = self._with_protocol("workflow_failed", "orchestrator", display, error=display)
        self.last_response_protocol = protocol
        return ToolResponse(content=[TextBlock(type="text", text=display)],
            metadata={
                "success": True,
                "protocol": protocol,
                "response_msg": self._agent_msg(
                    self.orchestrator.name if self.orchestrator else "Orchestrator",
                    display,
                    protocol,
                ),
            },
        )

    async def _stop_workflow_runtime(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Stop the active runtime task with the same cleanup used by standalone jobs."""
        stage_name = str(workflow.get("current_stage") or "")
        stage = (workflow.get("stages") or {}).get(stage_name, {})
        pid = str(stage.get("pid") or "").strip() or None
        container = str(stage.get("container") or "").strip() or None
        if stage_name == "benchmark":
            command = _workflow_benchmark_stop_command(workflow)
            benchmark_result = await self._workflow_inference_event(command, workflow)
            service_command = _workflow_inference_service_stop_command()
            service_result = await self._workflow_inference_event(service_command, workflow)
            benchmark_text = benchmark_result.get("text") or f"已发送{command}指令"
            service_text = service_result.get("text") or f"已发送{service_command}指令"
            return f"{benchmark_text}\n\n{service_text}"
        if stage_name == "deploy":
            command = _workflow_inference_service_stop_command()
            result = await self._workflow_inference_event(command, workflow)
            return result.get("text") or f"已发送{command}指令"
        if stage_name == "evaluate":
            model_path = str((workflow.get("context") or {}).get("trained_model_path") or "").strip()
            stop_message_parts = ["停止当前单模型评估 single_model_evaluation_vpn"]
            if model_path:
                stop_message_parts.append(model_path)
            result = await self._handle_stop_command(
                " ".join(stop_message_parts),
                task_type="assessment",
                pid=None,
                container=container,
            )
            if not result or "结束指令执行失败" in result or "PID 格式不正确" in result:
                raise ValueError(result or "停止运行中的单模型评估失败")
            return result
        if stage_name == "publish":
            if not pid:
                return "版本化发布尚未记录宿主机 PID，已停止工作流推进"
            if container and container != "host":
                result = subprocess.run(
                    ["docker", "exec", container, "sh", "-c", f"kill {shlex.quote(pid)}"],
                    capture_output=True, text=True, timeout=30,
                )
                output = "\n".join(
                    text for text in (result.stdout.strip(), result.stderr.strip()) if text
                )
                return output or f"已发送停止版本化发布指令。\n容器: {container}\nPID: {pid}"
            result = _workflow_stop_host_process(pid)
            return f"已发送停止版本化发布指令。\n宿主机 PID: {pid}\n{result}"
        task_type = {"train": "train", "evaluate": "evaluate"}.get(stage_name)
        if not task_type:
            return None
        result = await self._handle_stop_command(
            "停止当前一键工作流",
            task_type=task_type,
            pid=pid,
            container=container,
        )
        if not result or "结束指令执行失败" in result or "PID 格式不正确" in result:
            raise ValueError(result or "停止运行中的任务失败")
        return result

    async def _workflow_control_via_orchestrator(self, message: str, command: str) -> str:
        """Run workflow controls through a real orchestrator ReAct turn."""
        requested_workflow_id = self._extract_workflow_id(command) or self._extract_workflow_id(message)
        action = command.split(":", 1)[0]
        if requested_workflow_id:
            control_action = f"{action}:{requested_workflow_id}"
        else:
            control_action = command
        if action in {
            "resume", "continue", "retry", "继续", "续跑", "恢复",
            "stop", "cancel", "停止", "结束", "取消",
        }:
            return self._tool_response_to_client_text(
                await self._workflow_control(control_action)
            )
        if self.orchestrator is None:
            return "系统错误：编排器未初始化"
        prompt = (
            f"当前用户输入：{message}\n"
            "[系统路由提示] 这是持久化一键工作流控制请求。"
            f"必须调用 _workflow_control 工具，并且 action={command}。"
            "工具返回后，请将结果直接回复给用户，不要调用其他工具。"
        )
        response = await self.orchestrator(_msg(name="User", content=prompt, role="user"))
        response_text = response.get_text_content() if response else "系统错误：无法获取响应"
        protocol = response.metadata.get("protocol") if response and isinstance(response.metadata, dict) else None
        if (
            protocol is None
            and isinstance(self.last_response_protocol, dict)
            and str(self.last_response_protocol.get("type") or "").startswith("workflow_")
        ):
            protocol = self._normalize_protocol(
                {**self.last_response_protocol, "message": response_text},
                response_text,
                source="restored",
            )
        protocol = self._normalize_protocol(protocol, response_text) or self._message_protocol("orchestrator", response_text)
        self.last_response_protocol = protocol
        return self._protocol_json_response(response_text, protocol)

    async def _handle_workflow_message(self, message: str) -> Optional[str]:
        command = self._workflow_command(message)
        if not command:
            return None
        manager = get_workflow_manager()
        dataset_ref: Optional[str] = None
        try:
            if command == "start":
                dataset_ref = self._extract_dataset_reference(message)
                if not dataset_ref:
                    raise ValueError("请在一键工作流口令中提供数据标识，例如 20260417")
                workflow_context = self._workflow_training_context(message, dataset_ref)
                owner_user_id, owner_aliases = _inference_owner_payload(self.user_id)
                workflow_context["inference_owner_user_id"] = owner_user_id
                workflow_context["inference_owner_aliases"] = owner_aliases
                #deepseek
                workflow_context["user_role"] = _current_user_role()
                workflow = manager.create(
                    self.user_id,
                    dataset_ref,
                    auto_start=False,
                    train_type=workflow_context["train_type"],
                    context=workflow_context,
                )
                return await self._start_workflow_via_trainer(manager, workflow)
            else:
                return await self._workflow_control_via_orchestrator(message, command)
        except ValueError as exc:
            display = str(exc)
            protocol = None
            if command == "start" and dataset_ref and "dataset_name" in display:
                protocol = self._workflow_dataset_name_need_input_protocol(message, dataset_ref, display)
                if protocol:
                    self.pending_parameters["workflow"] = self._pending_state(
                        "workflow",
                        message,
                        "param",
                        display,
                        str(protocol.get("message") or display),
                        protocol,
                        required_params=protocol.get("requiredParams") or [],
                        needs_choice=False,
                    )
            if protocol is None:
                protocol = self._with_protocol("workflow_failed", "orchestrator", display, error=display)
        self.last_response_protocol = protocol
        return self._protocol_json_response(display, protocol)
    async def process_message(self, message: str, raw_content: Any = None) -> str:
        """处理单条用户消息
        
        Args:
            message: 文本消息
            raw_content: 原始多模态内容（包含图像等）
        """
        self.current_user_message = message or ""
        logger.info(f"用户{self.user_id}; 输入{message}")
        workflow_response = await self._handle_workflow_message(message)
        if workflow_response is not None:
            return workflow_response
        if self.pending_parameters:
            # 有等待参数的agent
            agent_names = list(self.pending_parameters.keys())
            if len(agent_names) == 1:
                agent_name = agent_names[0]
                logger.info(f"[Orchestrator] 将参数传递给 {agent_name}...")
                
                task_info = self.pending_parameters[agent_name]
                original_task = task_info["original_task"]
                if agent_name == "trainer":
                    # Also heal pending state written by older versions, where
                    # trainer-only system hints were persisted with the user task.
                    original_task = self._strip_trainer_system_hints(original_task)
                
                if task_info.get("needs_choice", False):
                    if not self._is_choice_response(message) and self._looks_like_new_task(message):
                        logger.info(f"[Orchestrator] 检测到用户发起新任务，退出 {agent_name} 的等待选择状态")
                        del self.pending_parameters[agent_name]
                        return await self.process_message(message, raw_content=raw_content)
                    # 处理选择型等待
                    del self.pending_parameters[agent_name]
                    if agent_name == "dataprocessor" and self._is_close_choice_response(message):
                        return await self._direct_agent_response(
                            "dataprocessor",
                            "已确认关闭当前数据处理流程，当前任务结束。",
                            original_task,
                            f"用户选择：{message}",
                        )

                    if agent_name == "dataprocessor":
                        full_message = (
                            f"{original_task}\n用户选择：{message}\n"
                            "[系统补充] 用户现在是在回答“是否继续执行高级筛选”。"
                            "请先根据用户选择自行判断是继续还是结束："
                            "如果用户表示否定、拒绝、不需要、不执行或取消，只回复确认不执行高级筛选，"
                            "不要调用任何数据处理工具；如果用户表示肯定、需要或继续，"
                            "只执行数据高级筛选，不要重新执行数据预处理；"
                            "如果缺少高级筛选 input_folder，只追问高级筛选的数据路径。"
                        )
                    else:
                        full_message = f"{original_task}\n用户选择：{message}"
                    
                    if agent_name == "dataprocessor":
                        response = await self._call_dataprocessor(full_message)
                    elif agent_name == "trainer":
                        response = await self._call_trainer(full_message)
                    elif agent_name == "evaluator":
                        response = await self._call_evaluator(full_message)
                    else:
                        response = await self._call_agent_with_params(agent_name, f"{original_task}\n用户回复：{message}")
                else:
                    # 参数型等待
                    request_kind = task_info.get("request_kind", "param")
                    effective_request_kind = request_kind
                    explicit_agent = self._resolve_explicit_agent_route(message)
                    new_task_intent = any(
                        keyword in (message or "").lower()
                        for keyword in ["启动", "开始", "执行", "运行", "训练", "跑", "start", "launch", "run", "train"]
                    )    
                    if (
                        request_kind == "param"
                        and (
                            self._extract_stop_target_params(message).get("has_stop_intent")
                            or (explicit_agent is not None and explicit_agent != agent_name)
                            )
                            or (
                                explicit_agent == agent_name
                                and self._looks_like_new_task(message)
                                and new_task_intent
                            )
                    ):
                        logger.info(f"[Orchestrator] 检测到用户发起新任务，退出 {agent_name} 的参数等待状态")
                        del self.pending_parameters[agent_name]
                        return await self.process_message(message, raw_content=raw_content)
                    if request_kind == "param" and self._is_explanatory_followup(message):
                        pending_kind = task_info.get("kind")
                        pending_text = "\n".join(
                            str(task_info.get(key, "") or "")
                            for key in ("response", "friendly_response", "message", "original_task")
                        )
                        if pending_kind in {"training_type", "evaluation_type", "data_preprocess_params"} or any(
                            keyword in pending_text for keyword in ["data_type", "strategy", "训练类型", "评估类型"]
                        ):
                            effective_request_kind = "type"
                    normalized_message = (
                        self._normalize_pending_param_message(task_info, message)
                        if effective_request_kind == "param"
                        else message
                    )
                    if agent_name == "datacollector":
                        followup = "用户回复" if effective_request_kind == "type" else "参数"
                        response = await self._call_datacollector(f"{original_task}\n{followup}：{normalized_message}")
                    elif agent_name == "dataprocessor":
                        followup = "用户回复" if effective_request_kind == "type" else "参数"
                        response = await self._call_dataprocessor(f"{original_task}\n{followup}：{normalized_message}")
                    elif agent_name == "trainer":
                        followup = "用户回复" if effective_request_kind == "type" else "参数"
                        response = await self._call_trainer(f"{original_task}\n{followup}：{normalized_message}")
                    elif agent_name == "workflow":
                        del self.pending_parameters[agent_name]
                        workflow_message = f"{original_task}，{normalized_message}"
                        workflow_response = await self._handle_workflow_message(workflow_message)
                        if workflow_response is not None:
                            return workflow_response
                        return await self.process_message(workflow_message, raw_content=raw_content)
                    elif agent_name == "evaluator":
                        followup = "用户回复" if effective_request_kind == "type" else "参数"
                        response = await self._call_evaluator(f"{original_task}\n{followup}：{normalized_message}")
                    elif agent_name == "inference":
                        pass
                    elif agent_name == "monitor":
                        followup = "用户回复" if effective_request_kind == "type" else "参数"
                        response = await self._call_monitor(f"{original_task}\n{followup}：{normalized_message}")
                    else:
                        response = f"未知agent: {agent_name}"
                
                return self._tool_response_to_client_text(response)
            else:
                reply = (
                    f"目前有多个任务都在等你补充信息：{', '.join(agent_names)}。\n\n"
                    "你可以直接告诉我想先处理哪个，例如：`给trainer 参数：lora批量训练`"
                )
                protocol = self._with_protocol(
                    "need_input",
                    "orchestrator",
                    reply,
                    kind="multiple_pending_inputs",
                    agents=agent_names,
                )
                self.last_response_protocol = protocol
                return self._protocol_json_response(reply, protocol)
        else:
            # 正常处理用户请求
            logger.info("[Orchestrator] 分析任务中...")

            if self.last_completed_task and self._is_acknowledgement_only(message):
                logger.info("[Orchestrator] 检测到用户仅确认上一轮结果，不触发新任务")
                await asyncio.sleep(ACKNOWLEDGEMENT_REPLY_DELAY_SECONDS)
                record = self.last_completed_task or {}
                agent_name = record.get("agent")
                if agent_name in self.agents:
                    return await self._direct_agent_response(
                        agent_name,
                        self._build_acknowledgement_reply(),
                        str(record.get("task") or ""),
                        f"用户确认：{message}",
                    )
                reply = self._build_acknowledgement_reply()
                protocol = self._message_protocol("orchestrator", reply)
                self.last_response_protocol = protocol
                return self._protocol_json_response(reply, protocol)

            stop_target = self._extract_stop_target_params(message)
            stop_task_type = self._normalize_stop_task_type(stop_target.get("task_type"))
            if self._is_inference_command(message):
                logger.info("[Orchestrator] 检测到推理请求，原样直通 inference agent")
                response = await self._call_inference(message, command=message)
                return self._tool_response_to_client_text(response)
            if stop_target.get("has_stop_intent") and stop_task_type in {"train", "assessment", "data"}:
                logger.info("[Orchestrator] 检测到明确的终止意图，直接执行停止任务")
                response = await self._stop_task(
                    message,
                    task_type=stop_task_type,
                    pid=stop_target.get("pid"),
                    container=stop_target.get("container"),
                )
                return self._tool_response_to_client_text(response)

            explicit_agent = self._resolve_explicit_agent_route(message)
            if explicit_agent == "trainer":
                logger.info("[Orchestrator] 检测到明确的训练类型，直接路由到 trainer")
                response = await self._call_trainer(message)
                return self._tool_response_to_client_text(response)
            if explicit_agent == "dataprocessor":
                logger.info("[Orchestrator] 检测到明确的数据处理意图，直接路由到 dataprocessor")
                response = await self._call_dataprocessor(message)
                return self._tool_response_to_client_text(response)
            if explicit_agent == "datacollector":
                logger.info("[Orchestrator] 检测到明确的数据准备意图，直接路由到 datacollector")
                response = await self._call_datacollector(message)
                return self._tool_response_to_client_text(response)
            if explicit_agent == "evaluator":
                logger.info("[Orchestrator] 检测到明确的评估意图，直接路由到 evaluator")
                response = await self._call_evaluator(message)
                return self._tool_response_to_client_text(response)
            if explicit_agent == "monitor":
                logger.info("[Orchestrator] 检测到明确的监控意图，直接路由到 monitor")
                response = await self._call_monitor(message)
                return self._tool_response_to_client_text(response)
            inference_route_hint = ""
            if explicit_agent == "inference":
                logger.info("[Orchestrator] 检测到明确的推理意图，提示编排器调用 inference 工具")
                inference_route_hint = (
                    "[系统路由提示] 当前用户输入是推理相关请求。必须调用 _call_inference 工具处理，"
                    "不要直接回答，不要调用 trainer/evaluator/monitor；"
                    "将用户原始输入作为 task_description 传入。"
                )
                
            
            # 兜底：如果既不是明确路由，也不是业务相关意图，直接引导用户
            #if explicit_agent is None and not self._is_task_related(message):
            #    logger.info("[Orchestrator] 输入与业务无关，直接回复引导语")
            #    reply = "你好！我可以帮你处理数据准备、模型训练、评估、推理服务等任务。请告诉我具体想做什么？"
            #    protocol = self._message_protocol("orchestrator", reply)
            #    self.last_response_protocol = protocol
            #    return self._protocol_json_response(reply, protocol)
            
            context_parts = []
            agent = None

            if self.last_completed_task:
                agent = self.last_completed_task['agent']
                task_description = self.last_completed_task['task'].split('\n参数：')[0] if '\n参数：' in self.last_completed_task['task'] else self.last_completed_task['task']
                response = self.last_completed_task['response']
                result = f"agent: {agent}, 已完成{task_description}, 当前问题: {response}"
                context_parts.append(result)
            
            context_parts.append(f"当前用户输入：{message}")
            if stop_target.get("has_stop_intent"):
                context_parts.append(
                    "[系统路由提示] 当前用户输入是终止任务请求。必须调用 _stop_task 工具处理，"
                    "不要调用训练、评估或数据处理启动工具；将用户原始输入作为 task_description 传入。"
                )
            training_route_hint = self._build_training_route_hint(message)
            if training_route_hint:
                context_parts.append(training_route_hint)
            if inference_route_hint:
                context_parts.append(inference_route_hint)
            context_message = "\n".join(context_parts) if agent != "inference" or inference_route_hint else message

            if self.orchestrator is None:
                logger.error("Orchestrator is not initialized")
                return "系统错误：编排器未初始化"
            
            # 构建多模态消息内容
            msg_content = context_message
            if raw_content is not None and isinstance(raw_content, list):
                # 如果有原始多模态内容，构建包含文本和图像的消息
                msg_content = [TextBlock(type="text", text=context_message)]
                # 添加图像等非文本内容
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") in ["image", "audio", "video", "data"]:
                        msg_content.append(_content_block(block))
            
            msg = _msg(name="User", content=msg_content, role="user")
            response = await self.orchestrator(msg)
            pending_prompt = self._pending_parameter_prompt()
            if pending_prompt:
                if len(self.pending_parameters) == 1:
                    pending_agent = next(iter(self.pending_parameters.keys()))
                    protocol = self.pending_parameters[pending_agent].get("protocol")
                else:
                    protocol = self._with_protocol(
                        "need_input",
                        "orchestrator",
                        pending_prompt,
                        kind="multiple_pending_inputs",
                        agents=list(self.pending_parameters.keys()),
                    )
                self.last_response_protocol = protocol
                return self._protocol_json_response(pending_prompt, protocol)

            response_text = response.get_text_content() if response else "系统错误：无法获取响应"
            protocol = None
            if response and isinstance(response.metadata, dict):
                protocol = response.metadata.get("protocol")
            if (
                protocol is None
                and self.last_completed_task
                and self.last_completed_task.get("agent") == "inference"
                and isinstance(self.last_response_protocol, dict)
                and self.last_response_protocol.get("agent") == "inference"
            ):
                protocol = self._normalize_protocol(
                    {
                        **self.last_response_protocol,
                        "message": response_text,
                    },
                    response_text,
                    source="restored",
                )
            protocol = self._normalize_protocol(protocol, response_text)
            protocol = self._restore_recent_job_protocol(response_text, protocol)
            dataset_choice_protocol = None
            if protocol is None or protocol.get("type") == "message":
                dataset_choice_protocol = self._dataset_name_choice_protocol_from_message(response_text, message)
            if dataset_choice_protocol:
                protocol = dataset_choice_protocol
                self.pending_parameters["trainer"] = self._pending_state(
                    "trainer",
                    message,
                    "param",
                    response_text,
                    str(protocol.get("message") or response_text),
                    protocol,
                    required_params=protocol.get("requiredParams") or [],
                    needs_choice=False,
                )
            protocol = protocol or self._message_protocol("orchestrator", response_text)
            self.last_response_protocol = protocol
            return self._protocol_json_response(response_text, protocol)

class UserSession:
    """用户会话：包含该用户的所有智能体和相关状态"""
    
    def __init__(self, user_id: str, system: Optional[OrchestratorSystem] = None):
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.access_count = 0
        
        # 如果提供了system，则使用；否则需要稍后重建
        self.system = system
        self._initialized = system is not None
    
    async def initialize(self):
        """初始化会话（异步创建智能体）"""
        if not self._initialized:
            self.system = OrchestratorSystem(self.user_id, SHARED_MODEL)
            await self.system.initialize_agents()
            self._initialized = True
            logger.info(f"UserSession initialized for {self.user_id}")
    
    def update_activity(self):
        """更新最后活跃时间"""
        self.last_active = datetime.now()
        self.access_count += 1
    
    def get_serializable_state(self) -> Dict[str, Any]:
        """获取可序列化的状态"""
        if self.system:
            return self.system.get_serializable_state()
        return {}
    
    async def restore_from_state(self, state: Dict[str, Any]):
        """从状态恢复会话"""
        if self.system:
            self.system.restore_state(state)
            self._initialized = True
            logger.info(f"UserSession restored from state for {self.user_id}")


# ========== 会话管理函数 ==========

async def get_or_create_user_session(user_id: str) -> UserSession:
    """获取或创建用户会话（集成内存管理器）"""
    memory_manager = await get_memory_manager()
    
    # 尝试从内存管理器获取会话
    existing_session = await memory_manager.get_session(user_id)
    
    if existing_session:
        # 活跃态或休眠态，直接返回
        existing_session.update_activity()
        logger.info(f"Session retrieved for user {user_id} (active or hibernate)")
        return existing_session
    
    # 检查是否从磁盘恢复了状态
    restored_state = await memory_manager.get_session_state(user_id)
    
    if restored_state:
        # 从磁盘恢复，需要重建 agents
        logger.info(f"Restoring session from disk for user {user_id}")
        session = UserSession(user_id)
        await session.initialize()
        await session.restore_from_state({
            "current_task_state": restored_state.current_task_state,
            "pending_parameters": restored_state.pending_parameters,
            "task_context": restored_state.task_context,
            "task_history": restored_state.task_history,
            "last_completed_task": restored_state.last_completed_task,
            "last_response_protocol": restored_state.last_response_protocol,
        })
        session.access_count = restored_state.access_count
        
        # 注册到内存管理器
        await memory_manager.register_session(
            user_id, 
            session, 
            SessionState(
                user_id=user_id,
                last_active=datetime.now(),
                created_at=restored_state.created_at,
                access_count=restored_state.access_count,
                task_context=restored_state.task_context,
                task_history=restored_state.task_history,
                pending_parameters=restored_state.pending_parameters,
                current_task_state=restored_state.current_task_state,
                last_completed_task=restored_state.last_completed_task,
                last_response_protocol=restored_state.last_response_protocol,
            )
        )
        return session
    
    # 创建新会话
    logger.info(f"Creating new session for user {user_id}")
    session = UserSession(user_id)
    await session.initialize()
    
    # 注册到内存管理器
    await memory_manager.register_session(
        user_id,
        session,
        SessionState(user_id=user_id, last_active=datetime.now())
    )
    
    return session


async def _workflow_event_consumer_loop() -> None:
    manager = get_workflow_manager()
    while True:
        events = manager.claim_pending_events(_WORKFLOW_EVENT_CONSUMER_ID)
        for event in events:
            try:
                session = await get_or_create_user_session(event["user_id"])
                if session.system is None:
                    raise RuntimeError("工作流会话未正确初始化")
                event_task = asyncio.create_task(session.system.handle_workflow_event(event))
                renew_seconds = max(1, int(config.workflow.event_lease_renew_seconds))
                while True:
                    done, _ = await asyncio.wait({event_task}, timeout=renew_seconds)
                    if done:
                        await event_task
                        break
                    if not manager.renew_event_lease(event["event_id"], _WORKFLOW_EVENT_CONSUMER_ID):
                        event_task.cancel()
                        try:
                            await event_task
                        except asyncio.CancelledError:
                            pass
                        raise RuntimeError("工作流事件租约已失效，停止当前 Agent 操作")
                manager.complete_event(event["event_id"], _WORKFLOW_EVENT_CONSUMER_ID)
                workflow_after_event = manager.get(event["workflow_id"])
                if workflow_after_event and workflow_after_event.get("status") == "running":
                    logger.debug(
                        "[workflow-debug] re-activate after event complete workflow_id=%s event_type=%s current_stage=%s stage_status=%s",
                        event.get("workflow_id"),
                        event.get("event_type"),
                        workflow_after_event.get("current_stage"),
                        ((workflow_after_event.get("stages") or {}).get(workflow_after_event.get("current_stage"), {})).get("status"),
                    )
                    manager.activate_worker_for(event["workflow_id"])
                else:
                    logger.debug(
                        "[workflow-debug] event complete no re-activate workflow_id=%s event_type=%s workflow_status=%s",
                        event.get("workflow_id"),
                        event.get("event_type"),
                        workflow_after_event.get("status") if workflow_after_event else None,
                    )
                try:
                    memory_manager = await get_memory_manager()
                    await memory_manager.update_session_state(
                        event["user_id"],
                        task_context=session.system.task_context,
                        task_history=session.system.task_history,
                        pending_parameters=session.system.pending_parameters,
                        current_task_state=session.system.current_task_state,
                        last_completed_task=session.system.last_completed_task,
                        last_response_protocol=session.system.last_response_protocol,
                    )
                except Exception:
                    logger.exception(
                        "Workflow event %s completed, but session state persistence failed",
                        event["event_id"],
                    )
            except Exception as exc:
                logger.exception("Workflow event %s failed", event["event_id"])
                logger.debug(
                    "[workflow-debug] event failure workflow_id=%s event_type=%s error=%s",
                    event.get("workflow_id"),
                    event.get("event_type"),
                    exc,
                )
                manager.fail_current_stage(event["workflow_id"], str(exc))
                manager.fail_event(event["event_id"], _WORKFLOW_EVENT_CONSUMER_ID, str(exc))
        await asyncio.sleep(1)


def ensure_workflow_event_consumer() -> None:
    """Start the async bridge between persisted workflow events and user agents."""
    global _WORKFLOW_EVENT_CONSUMER_TASK
    global _WORKFLOW_EVENT_LOOP
    _WORKFLOW_EVENT_LOOP = asyncio.get_running_loop()
    if _WORKFLOW_EVENT_CONSUMER_TASK is None or _WORKFLOW_EVENT_CONSUMER_TASK.done():
        _WORKFLOW_EVENT_CONSUMER_TASK = asyncio.create_task(
            _workflow_event_consumer_loop(),
            name="workflow-agent-event-consumer",
        )


async def start_workflow_background_tasks() -> None:
    """Public startup hook for API wrappers importing this module."""
    ensure_workflow_event_consumer()


async def process_user_message(user_id: str, message: str, raw_content: Any = None) -> str:
    """处理用户消息（入口函数）
    
    Args:
        user_id: 用户ID
        message: 文本消息
        raw_content: 原始多模态内容（包含图像等）
    """
    logger.info(f"用户{user_id}; 输入{message}")
    ensure_workflow_event_consumer()
    reset_epoch = _current_reset_epoch(user_id)
    session = await get_or_create_user_session(user_id)
    
    # 更新会话状态
    memory_manager = await get_memory_manager()
    # 检查 system 是否已初始化
    if session.system is None:
        logger.error(f"Session system is not initialized for user {user_id}")
        return "系统错误：会话未正确初始化"
    
    await memory_manager.update_session_state(
        user_id,
        task_context=session.system.task_context,
        task_history=session.system.task_history,
        pending_parameters=session.system.pending_parameters,
        current_task_state=session.system.current_task_state,
        last_completed_task=session.system.last_completed_task,
        last_response_protocol=session.system.last_response_protocol,
    )
    
    # 处理消息，传递原始多模态内容
    response = await session.system.process_message(message, raw_content=raw_content)
    if _current_reset_epoch(user_id) != reset_epoch:
        logger.info("Discard stale response for user %s because context was reset", user_id)
        return "上下文已清空，请重新发送你的消息。"
    await memory_manager.update_session_state(
        user_id,
        task_context=session.system.task_context,
        task_history=session.system.task_history,
        pending_parameters=session.system.pending_parameters,
        current_task_state=session.system.current_task_state,
        last_completed_task=session.system.last_completed_task,
        last_response_protocol=session.system.last_response_protocol,
    )
    logger.info(f"用户{user_id}; 最终返回{response[:200]}...")
    return response


async def process_user_message_structured(
    user_id: str,
    message: str,
    raw_content: Any = None,
    training_container: str = "",
    evaluation_container: str = "",
    grpo_container: str = "",
    multinode_training_container: str = "",
    resource_group_id: str = "",
    training_pool_id: str = "",
    user_role: str = "",
    owner_user_id: str = "",
    owner_aliases: Any = None,
) -> Dict[str, Any]:
    """处理用户消息，并返回前端可直接消费的轻量协议结果。"""
    logger.info(f"用户{user_id}; 输入{message}")
    ensure_workflow_event_consumer()
    reset_epoch = _current_reset_epoch(user_id)
    session = await get_or_create_user_session(user_id)
    resolved_training_container = training_container.strip() or DEFAULT_DOCKER_CONTAINER
    resolved_evaluation_container = evaluation_container.strip() or DEFAULT_EVALUATE_DOCKER_CONTAINER
    resolved_grpo_container = grpo_container.strip() or resolved_training_container
    resolved_multinode_training_container = multinode_training_container.strip() or MULTINODE_DOCKER_CONTAINER
    resolved_resource_group_id = resource_group_id.strip()
    resolved_training_pool_id = training_pool_id.strip()
    resolved_user_role = user_role.strip().lower()
    resolved_owner_user_id = str(owner_user_id or "").strip()
    resolved_owner_aliases = _owner_aliases_from_any(owner_aliases)
    logger.info(
        "用户%s; resolved containers training=%s evaluation=%s grpo=%s multinode=%s resource_group=%s training_pool=%s user_role=%s",
        user_id,
        resolved_training_container,
        resolved_evaluation_container,
        resolved_grpo_container,
        resolved_multinode_training_container,
        resolved_resource_group_id,
        resolved_training_pool_id,
        resolved_user_role,
    )
    memory_manager = await get_memory_manager()
    if session.system is None:
        logger.error(f"Session system is not initialized for user {user_id}")
        return {
            "message": "系统错误：会话未正确初始化",
            "protocol": {
                "version": PROTOCOL_VERSION,
                "type": "error",
                "agent": "orchestrator",
                "message": "系统错误：会话未正确初始化",
                "source": "fallback",
                "confidence": PROTOCOL_CONFIDENCE_BY_SOURCE["fallback"],
                "valid": True,
            },
        }

    training_token = REQUEST_TRAINING_CONTAINER.set(resolved_training_container)
    evaluation_token = REQUEST_EVALUATION_CONTAINER.set(resolved_evaluation_container)
    grpo_token = REQUEST_GRPO_CONTAINER.set(resolved_grpo_container)
    multinode_token = REQUEST_MULTINODE_TRAINING_CONTAINER.set(resolved_multinode_training_container)
    resource_group_token = REQUEST_RESOURCE_GROUP_ID.set(resolved_resource_group_id)
    training_pool_token = REQUEST_TRAINING_POOL_ID.set(resolved_training_pool_id)
    user_role_token = REQUEST_USER_ROLE.set(resolved_user_role)
    owner_user_id_token = REQUEST_INFERENCE_OWNER_USER_ID.set(resolved_owner_user_id)
    owner_aliases_token = REQUEST_INFERENCE_OWNER_ALIASES.set(tuple(resolved_owner_aliases))
    session.system.training_container = resolved_training_container
    session.system.evaluation_container = resolved_evaluation_container
    session.system.grpo_container = resolved_grpo_container
    session.system.multinode_training_container = resolved_multinode_training_container
    try:
        response = await session.system.process_message(message, raw_content=raw_content)
    finally:
        REQUEST_INFERENCE_OWNER_ALIASES.reset(owner_aliases_token)
        REQUEST_INFERENCE_OWNER_USER_ID.reset(owner_user_id_token)
        REQUEST_USER_ROLE.reset(user_role_token)
        REQUEST_TRAINING_CONTAINER.reset(training_token)
        REQUEST_EVALUATION_CONTAINER.reset(evaluation_token)
        REQUEST_GRPO_CONTAINER.reset(grpo_token)
        REQUEST_MULTINODE_TRAINING_CONTAINER.reset(multinode_token)
        REQUEST_RESOURCE_GROUP_ID.reset(resource_group_token)
        REQUEST_TRAINING_POOL_ID.reset(training_pool_token)
    if _current_reset_epoch(user_id) != reset_epoch:
        logger.info("Discard stale structured response for user %s because context was reset", user_id)
        return {
            "message": "上下文已清空，请重新发送你的消息。",
            "protocol": {
                "version": PROTOCOL_VERSION,
                "type": "message",
                "agent": "orchestrator",
                "message": "上下文已清空，请重新发送你的消息。",
                "source": "reset",
                "confidence": 1.0,
                "valid": True,
            },
        }
    protocol = session.system._normalize_protocol(
        session.system.last_response_protocol,
        response,
    ) or session.system._with_protocol(
        "message",
        "orchestrator",
        response,
        source="fallback",
    )
    session.system.last_response_protocol = protocol
    await memory_manager.update_session_state(
        user_id,
        task_context=session.system.task_context,
        task_history=session.system.task_history,
        pending_parameters=session.system.pending_parameters,
        current_task_state=session.system.current_task_state,
        last_completed_task=session.system.last_completed_task,
        last_response_protocol=session.system.last_response_protocol,
    )
    result = {
        "message": response,
        "protocol": protocol,
    }
    logger.info(f"用户{user_id}; 最终返回{result}")
    return result


async def get_memory_stats() -> Dict[str, Any]:
    """获取内存统计信息"""
    memory_manager = await get_memory_manager()
    return memory_manager.get_stats()


def _strip_user_marker(user_id: str) -> str:
    """Normalize a leading chat marker like [test22#session]."""
    normalized = str(user_id or "").strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1].strip()
    return normalized


def _resolve_context_user_id(raw_user_id: str, metadata: Any = None) -> str:
    """Resolve the memory/session key from Studio metadata and text prefix."""
    normalized_user_id = _strip_user_marker(raw_user_id)
    metadata = metadata if isinstance(metadata, dict) else {}

    context_username = str(
        metadata.get("__medflowContextUsername") or "",
    ).strip()
    if context_username:
        return _strip_user_marker(context_username)

    session_id = str(metadata.get("__medflowSessionId") or "").strip()
    base_username = str(
        metadata.get("__medflowUsername") or normalized_user_id,
    ).strip()
    if session_id and session_id != "default" and "#" not in base_username:
        return f"{_strip_user_marker(base_username)}#{session_id}"

    return normalized_user_id


# ========== 命令行交互模式 ==========

async def interactive_mode():
    """命令行交互模式"""
    ensure_workflow_event_consumer()
    print("=" * 60)
    print("多用户Agent系统 - 命令行测试模式")
    print("=" * 60)
    print(f"Think上下文: {'保留' if keep_think_in_context() else '不保留（仅输出展示）'}")
    print("命令格式: [user_id] 消息内容")
    print("例如: alice 你好")
    print("特殊命令:")
    print("  !users - 查看在线用户")
    print("  !stats - 查看内存统计")
    print("  !config - 查看当前配置")
    print("  exit - 退出")
    print("=" * 60)
    
    while True:
        try:
            user_input = (await asyncio.to_thread(input, "\n> ")).strip()
            content = user_input
            
            if user_input.lower() == "exit":
                break
            
            if user_input.lower() == "!users":
                memory_manager = await get_memory_manager()
                stats = memory_manager.get_stats()
                print(f"\n当前在线用户:")
                print(f"  活跃会话: {stats['active_sessions']}")
                print(f"  休眠会话: {stats['hibernate_sessions']}")
                print(f"  总会话: {stats['total_sessions']}")
                continue
             
            if user_input.lower() == "!stats":
                stats = await get_memory_stats()
                print(f"\n内存统计:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == "!config":
                print(f"\n当前配置:")
                print(f"  模型: {config.model.name}")
                print(f"  会话超时: {config.system.session_timeout}秒")
                print(f"  最大活跃会话: {config.system.memory_management.max_active_sessions}")
                print(f"  内存阈值: {config.system.memory_management.memory_threshold}%")
                continue
            
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2:
                print("格式错误：请使用 '[user_id] 消息内容' 格式")
                continue
            
            raw_user_id, message = parts
            user_id = _resolve_context_user_id(
                raw_user_id,
                getattr(user_input_msg, "metadata", None),
            )
            
            print(f"\n[处理用户 {user_id}]")
            start_time = time.time()
            
            # 保留原始多模态内容（包含图像等）
            raw_content = content if isinstance(content, list) else None
            result = await process_user_message_structured(user_id, message, raw_content=raw_content)
            reply = result["message"]
            protocol = result.get("protocol")
            
            elapsed = time.time() - start_time
            print(f"[回复] {reply}")
            if protocol:
                print(f"[协议] {json.dumps(protocol, ensure_ascii=False)}")
            print(f"[耗时] {elapsed:.2f}秒")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"错误: {e}")
    
    print("\n系统退出。")


def parse_startup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多用户Agent系统")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--keep-think-context",
        action="store_true",
        help="保留模型<think>内容到agent上下文和会话状态中。",
    )
    think_group.add_argument(
        "--strip-think-context",
        action="store_true",
        help="不将模型<think>内容写入agent上下文和会话状态，仅保留前端/CLI输出展示。",
    )
    args, _ = parser.parse_known_args()
    if args.keep_think_context:
        os.environ["AGENT_KEEP_THINK_CONTEXT"] = "1"
    elif args.strip_think_context:
        os.environ["AGENT_KEEP_THINK_CONTEXT"] = "0"
    else:
        os.environ.setdefault("AGENT_KEEP_THINK_CONTEXT", "0")
    return args


async def main():
    """主函数"""
    global AGENT_MAIN_LOOP
    AGENT_MAIN_LOOP = asyncio.get_running_loop()
    parse_startup_args()
    try:
        await interactive_mode()
    finally:
        # 清理资源
        memory_manager = await get_memory_manager()
        await memory_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())

