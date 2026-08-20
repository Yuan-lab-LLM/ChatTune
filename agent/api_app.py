# -*- coding: utf-8 -*-
"""MedFlow Runtime API entrypoint.

This module exposes the existing ``runtime_agent`` orchestration logic through
AgentScope Runtime's AgentApp protocol.
"""

from __future__ import annotations

import json
import hmac
import logging
import os
import re
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable


BASE_DIR = Path(__file__).resolve().parent

for path in (BASE_DIR,):
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

# Keep runtime_agent.py's relative config/data paths stable in direct and
# packaged detached-process runs.
os.chdir(BASE_DIR)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agentscope.message import Msg, TextBlock  # noqa: E402
from agentscope_runtime.engine import AgentApp  # noqa: E402
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest  # noqa: E402

from runtime_agent import (  # noqa: E402
    get_memory_manager,
    handle_studio_reset_context,
    process_user_message_structured,
    start_workflow_background_tasks,
)
from studio_runtime_registration import (  # noqa: E402
    StudioRuntimeRegistrationError,
    register_runtime_run,
    update_runtime_run_status,
)
from resource_api import (  # noqa: E402
    router as resource_router,
    start_gpu_background_refresh,
    stop_gpu_background_refresh,
)


logger = logging.getLogger(__name__)
RUNTIME_CONTEXT_MARKER = "__medflow_runtime_context__"
RUNTIME_CONTEXT_RE = re.compile(
    r"\b(training_container|evaluation_container|grpo_container|multinode_training_container|resource_group_id|training_pool_id|user_role|owner_user_id|owner_aliases|context_username)=([^\s]*)"
)
DEFAULT_AGENT_REQUEST_THREADS = 8


def _agent_request_threads_from_env() -> int:
    raw_value = os.getenv("MEDFLOW_AGENT_REQUEST_THREADS", "").strip()
    if not raw_value:
        return DEFAULT_AGENT_REQUEST_THREADS
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_AGENT_REQUEST_THREADS


_AGENT_REQUEST_EXECUTOR = ThreadPoolExecutor(
    max_workers=_agent_request_threads_from_env(),
    thread_name_prefix="agent-request",
)
_WORKER_THREAD_STATE = threading.local()
_SESSION_LOCKS: dict[str, tuple[asyncio.Lock, int]] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


def _session_lock_for_key(session_key: str) -> asyncio.Lock:
    with _SESSION_LOCKS_GUARD:
        entry = _SESSION_LOCKS.get(session_key)
        if entry is None:
            lock = asyncio.Lock()
            _SESSION_LOCKS[session_key] = (lock, 1)
            return lock
        lock, ref_count = entry
        _SESSION_LOCKS[session_key] = (lock, ref_count + 1)
        return lock


def _release_session_lock_for_key(session_key: str) -> None:
    with _SESSION_LOCKS_GUARD:
        entry = _SESSION_LOCKS.get(session_key)
        if entry is None:
            return
        lock, ref_count = entry
        if ref_count <= 1 and not lock.locked():
            _SESSION_LOCKS.pop(session_key, None)
        else:
            _SESSION_LOCKS[session_key] = (lock, max(0, ref_count - 1))


def _worker_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_WORKER_THREAD_STATE, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _WORKER_THREAD_STATE.loop = loop
    asyncio.set_event_loop(loop)
    return loop


def _run_coroutine_in_worker_thread(awaitable_factory: Callable[[], Awaitable[Any]]) -> Any:
    loop = _worker_thread_loop()
    return loop.run_until_complete(awaitable_factory())


def _to_plain_dict(value: Any) -> Any:
    """Convert pydantic/content objects to plain Python values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _request_extra_raw(request: Any, key: str, kwargs: dict[str, Any]) -> Any:
    """Read custom AgentRequest fields across pydantic/framework variants."""
    value = getattr(request, key, None)
    if value:
        return value

    model_extra = getattr(request, "model_extra", None)
    if isinstance(model_extra, dict) and model_extra.get(key):
        return model_extra[key]

    dumped = _to_plain_dict(request)
    if isinstance(dumped, dict) and dumped.get(key):
        return dumped[key]

    original = kwargs.get("request")
    if isinstance(original, dict) and original.get(key):
        return original[key]

    return kwargs.get(key)


def _request_extra_value(request: Any, key: str, kwargs: dict[str, Any]) -> str:
    value = _request_extra_raw(request, key, kwargs)
    return str(value or "")


def _request_extra_list(request: Any, key: str, kwargs: dict[str, Any]) -> list[str]:
    value = _request_extra_raw(request, key, kwargs)
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


def _runtime_context_from_input(request: Any) -> dict[str, str]:
    """Recover bridge context if AgentScope drops custom top-level fields."""
    context: dict[str, str] = {}
    for message in getattr(request, "input", None) or []:
        if _message_role(message) != "system":
            continue
        content = _message_content(message)
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            text = _block_text(block) if not isinstance(block, str) else block
            if RUNTIME_CONTEXT_MARKER not in text:
                continue
            for key, value in RUNTIME_CONTEXT_RE.findall(text):
                if value:
                    context[key] = value
    return context


def _block_type(block: Any) -> str | None:
    block = _to_plain_dict(block)
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_text(block: Any) -> str:
    block = _to_plain_dict(block)
    if isinstance(block, dict):
        return str(block.get("text") or "")
    return str(getattr(block, "text", "") or "")


def _message_role(message: Any) -> str | None:
    message = _to_plain_dict(message)
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_content(message: Any) -> Any:
    message = _to_plain_dict(message)
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _extract_user_payload(request: AgentRequest, msgs: Any) -> tuple[str, Any]:
    """Extract the last user text and any original multimodal blocks."""
    candidates = request.input or []
    if not candidates:
        return "", None

    user_messages = [
        message for message in candidates if _message_role(message) == "user"
    ]
    message = user_messages[-1] if user_messages else candidates[-1]
    content = _message_content(message)

    if isinstance(content, str):
        return content.strip(), None

    if isinstance(content, list):
        text_parts = [
            _block_text(block)
            for block in content
            if _block_type(block) == "text"
        ]
        raw_content = [_to_plain_dict(block) for block in content]
        return " ".join(part for part in text_parts if part).strip(), raw_content

    # Fallback for converted AgentScope messages or simpler ad-hoc requests.
    if hasattr(msgs, "get_text_content"):
        return str(msgs.get_text_content() or "").strip(), None
    if isinstance(msgs, list) and msgs:
        last_msg = msgs[-1]
        if hasattr(last_msg, "get_text_content"):
            return str(last_msg.get_text_content() or "").strip(), None

    return str(content or "").strip(), None


def _safe_identity_part(value: str | None, default: str) -> str:
    """Make user/session ids safe for MemoryManager's local pickle filenames."""
    value = (value or default).strip() or default
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _conversation_key(request: AgentRequest) -> str:
    """Build the state key used by runtime_agent's user-scoped memory manager."""
    user_part = _safe_identity_part(request.user_id, "anonymous")
    session_part = _safe_identity_part(
        request.session_id or request.id,
        "default",
    )
    return f"{user_part}#{session_part}"


def _studio_reply_name(request: AgentRequest) -> str:
    """Build a Studio-compatible reply name for Run page filtering."""
    user_part = _safe_identity_part(request.user_id, "anonymous")
    return f"Orchestrator_[{user_part}]"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage API startup/shutdown resources."""
    try:
        studio_run = register_runtime_run(BASE_DIR)
    except StudioRuntimeRegistrationError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Studio run registration failed: {exc}") from exc

    await start_workflow_background_tasks()
    await get_memory_manager()
    start_gpu_background_refresh()
    try:
        yield
    finally:
        stop_gpu_background_refresh()
        memory_manager = await get_memory_manager()
        await memory_manager.stop()
        try:
            update_runtime_run_status(studio_run.run_id, "done")
        except Exception as exc:
            print(
                f"[AgentAPI] failed to mark Studio run done: {exc}",
                flush=True,
            )


agent_app = AgentApp(
    app_name="Agent3",
    app_description="Agent3 multi-agent orchestration API",
    endpoint_path="/process",
    lifespan=lifespan,
)
agent_app.include_router(resource_router)

@agent_app.middleware("http")
async def authorize_agent_api(request: Request, call_next):
    normalized_path = request.url.path.rstrip("/") or "/"
    protected = normalized_path in {"/process", "/runtime-process", "/reset"} or normalized_path.startswith("/compatible-mode/")
    if not protected:
        return await call_next(request)
    expected = os.getenv("MEDFLOW_AGENT_API_TOKEN", "").strip()
    supplied = request.headers.get("Authorization", "")
    if not expected:
        return JSONResponse({"detail": "MEDFLOW_AGENT_API_TOKEN is not configured"}, status_code=503)
    if not hmac.compare_digest(supplied, f"Bearer {expected}"):
        return JSONResponse({"detail": "Invalid Agent API token"}, status_code=401)
    return await call_next(request)


async def _handle_agent_request(
    request: AgentRequest,
    msgs: Any = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Run the MedFlow workflow for an AgentScope Runtime request."""
    kwargs = kwargs or {}
    if request is None:
        raise ValueError("AgentRequest is required")
    if isinstance(request, dict):
        request = AgentRequest(**request)

    user_id = _conversation_key(request)
    message, raw_content = _extract_user_payload(request, msgs)
    if not message:
        message = "你好"
    runtime_context = _runtime_context_from_input(request)
    training_container = _request_extra_value(request, "training_container", kwargs)
    evaluation_container = _request_extra_value(request, "evaluation_container", kwargs)
    resource_group_id = _request_extra_value(request, "resource_group_id", kwargs)
    training_pool_id = _request_extra_value(request, "training_pool_id", kwargs)
    user_role = _request_extra_value(request, "user_role", kwargs)
    grpo_container = _request_extra_value(request, "grpo_container", kwargs)
    multinode_training_container = _request_extra_value(request, "multinode_training_container", kwargs)
    owner_user_id = _request_extra_value(request, "owner_user_id", kwargs)
    owner_aliases = _request_extra_list(request, "owner_aliases", kwargs)
    context_username = _request_extra_value(request, "context_username", kwargs)
    training_container = training_container or runtime_context.get("training_container", "")
    evaluation_container = (
        evaluation_container or runtime_context.get("evaluation_container", "")
    )
    resource_group_id = resource_group_id or runtime_context.get("resource_group_id", "")
    training_pool_id = training_pool_id or runtime_context.get("training_pool_id", "")
    user_role = user_role or runtime_context.get("user_role", "")
    grpo_container = grpo_container or runtime_context.get("grpo_container", "")
    multinode_training_container = (
        multinode_training_container or runtime_context.get("multinode_training_container", "")
    )
    owner_user_id = owner_user_id or runtime_context.get("owner_user_id", "")
    if not owner_aliases:
        owner_aliases = [
            item.strip()
            for item in runtime_context.get("owner_aliases", "").split(",")
            if item.strip()
        ]
    context_username = context_username or runtime_context.get("context_username", "")
    if context_username:
        context_base = context_username.split("#", 1)[0].strip()
        if context_base and context_base not in owner_aliases:
            owner_aliases.append(context_base)
    print(
        "[AgentAPI] received containers",
        {
            "training_container": training_container,
            "evaluation_container": evaluation_container,
            "grpo_container": grpo_container,
            "multinode_training_container": multinode_training_container,
            "resource_group_id": resource_group_id,
            "training_pool_id": training_pool_id,
            "user_role": user_role,
            "owner_user_id": owner_user_id,
            "owner_aliases": owner_aliases,
            "context_username": context_username,
        },
        flush=True,
    )

    result = await process_user_message_structured(
        user_id=user_id,
        message=message,
        raw_content=raw_content,
        training_container=training_container,
        evaluation_container=evaluation_container,
        grpo_container=grpo_container,
        multinode_training_container=multinode_training_container,
        resource_group_id=resource_group_id,
        training_pool_id=training_pool_id,
        user_role=user_role,
        owner_user_id=owner_user_id,
        owner_aliases=owner_aliases,
    )

    response_text = result.get("message", "")
    protocol = result.get("protocol")
    metadata = {"protocol": protocol} if isinstance(protocol, dict) else None

    text = (
        response_text
        if isinstance(response_text, str)
        else json.dumps(response_text, ensure_ascii=False)
    )
    return text, metadata


async def _run_agent_request(
    request: AgentRequest,
    msgs: Any = None,
    kwargs: dict[str, Any] | None = None,
    handler: Callable[
        [AgentRequest, Any, dict[str, Any] | None],
        Awaitable[tuple[str, dict[str, Any] | None]],
    ] = _handle_agent_request,
) -> tuple[str, dict[str, Any] | None]:
    """Run one Agent request without blocking the FastAPI event loop.

    Requests for the same conversation stay ordered because the Agent session
    context lives in this process and is not safe to mutate concurrently.
    """
    if request is None:
        raise ValueError("AgentRequest is required")
    if isinstance(request, dict):
        request = AgentRequest(**request)

    session_key = _conversation_key(request)
    lock = _session_lock_for_key(session_key)
    logger.info("Agent request queued for session %s", session_key)
    try:
        async with lock:
            logger.info("Agent request entered worker queue for session %s", session_key)
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(
                    _AGENT_REQUEST_EXECUTOR,
                    _run_coroutine_in_worker_thread,
                    lambda: handler(request, msgs, kwargs),
                )
                logger.info("Agent request completed for session %s", session_key)
                return result
            except Exception:
                logger.exception("Agent request failed for session %s", session_key)
                raise
    finally:
        _release_session_lock_for_key(session_key)


@agent_app.query(framework="agentscope")
async def query_func(
    self,
    msgs,
    request: AgentRequest | None = None,
    **kwargs,
):
    """Handle Runtime AgentRequest input with the existing Agent3 workflow."""
    if request is None:
        request = kwargs.get("request")
    if request is None:
        raise ValueError("AgentRequest is required")
    if isinstance(request, dict):
        request = AgentRequest(**request)

    text, metadata = await _run_agent_request(request, msgs, kwargs)

    yield Msg(
        name=_studio_reply_name(request),
        role="assistant",
        content=[TextBlock(text=text)],
        metadata=metadata,
    ), True


def _runtime_sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@agent_app.post("/runtime-process")
async def runtime_process(request: Request):
    """Studio Bridge endpoint that returns the SSE protocol Studio already parses."""
    payload = await request.json()
    agent_request = AgentRequest(**payload)

    async def event_stream():
        try:
            text, metadata = await _run_agent_request(agent_request)
            if text:
                yield _runtime_sse_event({"object": "content", "text": text})
            yield _runtime_sse_event(
                {
                    "object": "message",
                    "status": "completed",
                    "metadata": metadata or {},
                }
            )
            yield _runtime_sse_event({"object": "response", "status": "completed"})
        except Exception as exc:
            print(f"[AgentAPI] runtime-process failed: {exc}", flush=True)
            yield _runtime_sse_event(
                {
                    "object": "response",
                    "status": "failed",
                    "error": {"message": str(exc)},
                }
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""

    user_id: str | None = None
    session_id: str | None = None
    contextUsername: str | None = None
    username: str | None = None
    cancelWorkflows: bool = False


@agent_app.post("/reset")
async def reset_context(request: ResetRequest):
    """Reset agent conversational context for a given user/session."""
    user_part = _safe_identity_part(request.user_id, "anonymous")
    session_part = _safe_identity_part(request.session_id, "default")
    conversation_key = f"{user_part}#{session_part}"

    payload = {
        "userId": request.user_id,
        "username": request.username,
        "contextUsername": request.contextUsername,
        "runId": request.session_id,
        "cancelWorkflows": request.cancelWorkflows,
    }
    handle_studio_reset_context(payload)

    return {
        "success": True,
        "message": f"Context reset requested for {conversation_key}",
    }


if __name__ == "__main__":
    agent_app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8090")),
    )





