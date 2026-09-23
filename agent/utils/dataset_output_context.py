"""Request-scoped state for chaining data preprocessing into filtering."""

from __future__ import annotations

import re
import posixpath
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional


REQUEST_LAST_DATASET_OUTPUT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "request_last_dataset_output",
    default=None,
)


def begin_dataset_output_capture() -> Token:
    """Start one isolated capture using a holder shared by copied contexts."""
    return REQUEST_LAST_DATASET_OUTPUT.set({})


def reset_dataset_output_capture(token: Token) -> None:
    REQUEST_LAST_DATASET_OUTPUT.reset(token)


def record_last_dataset_output(dataset: Dict[str, Any]) -> None:
    """Publish a dataset result so the parent request context can observe it.

    Async tasks receive a copied ContextVar mapping. Mutating the request's
    holder in place makes the result visible to the parent task, while setting a
    fresh value in the child task would keep it child-local.
    """
    snapshot = dict(dataset)
    holder = REQUEST_LAST_DATASET_OUTPUT.get()
    if isinstance(holder, dict):
        holder.clear()
        holder.update(snapshot)
        return
    REQUEST_LAST_DATASET_OUTPUT.set(snapshot)


def current_last_dataset_output() -> Optional[Dict[str, Any]]:
    value = REQUEST_LAST_DATASET_OUTPUT.get()
    return dict(value) if isinstance(value, dict) and value else None


def attach_output_to_filter_choice_protocol(
    protocol: Any,
    output_path: str,
    registered_dataset: Any = None,
) -> Any:
    """Persist a preprocessing output only on the matching choice protocol."""
    path = str(output_path or "").strip()
    if (
        not isinstance(protocol, dict)
        or protocol.get("type") != "need_input"
        or protocol.get("kind") != "advanced_filter_choice"
        or not path
    ):
        return protocol
    known_params = protocol.get("knownParams")
    updated = {
        **protocol,
        "knownParams": {
            **(known_params if isinstance(known_params, dict) else {}),
            "input_folder": path,
        },
        "outputFolder": path,
    }
    if isinstance(registered_dataset, dict):
        updated["registeredDataset"] = {
            "resourceId": registered_dataset.get("resource_id"),
            "datasetType": registered_dataset.get("dataset_type"),
            "datasetName": registered_dataset.get("dataset_name"),
            "sourcePath": registered_dataset.get("source_path"),
            "containerName": registered_dataset.get("container"),
            "storageState": registered_dataset.get("storage_state"),
        }
    return updated


def resolve_pending_dataset_output_path(task_info: Any) -> str:
    """Read the preprocessing output persisted with a pending filter choice."""
    if not isinstance(task_info, dict):
        return ""
    protocol = task_info.get("protocol")
    if isinstance(protocol, dict):
        known_params = protocol.get("knownParams")
        if isinstance(known_params, dict):
            path = str(known_params.get("input_folder") or "").strip()
            if path:
                return path
        path = str(protocol.get("outputFolder") or "").strip()
        if path:
            return path

    # Compatibility for waiting states saved before outputFolder was added to
    # the protocol. Only accept the exact system marker written by the runtime.
    original_task = str(task_info.get("original_task") or "")
    match = re.search(
        r"\[系统上下文\]\s*本次数据处理产物路径：input_folder=([^\s]+)",
        original_task,
    )
    return match.group(1).strip() if match else ""


def pending_output_manifest_entry(
    task_info: Any,
    actor_user_id: str,
    container: str,
) -> Optional[Dict[str, Any]]:
    """Build a narrowly scoped authorization entry for a just-created output.

    The resource manifest received on the next request may lag behind dataset
    registration. Only the server-authored pending protocol can bridge that
    window, and only for the current user's canonical private output path.
    """
    if not isinstance(task_info, dict):
        return None
    protocol = task_info.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("type") != "need_input"
        or protocol.get("kind") != "advanced_filter_choice"
    ):
        return None

    known_params = protocol.get("knownParams")
    known_path = (
        str(known_params.get("input_folder") or "").strip()
        if isinstance(known_params, dict)
        else ""
    )
    output_path = str(protocol.get("outputFolder") or "").strip()
    if not known_path or known_path != output_path:
        return None

    actor = str(actor_user_id or "").strip()
    target_container = str(container or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", actor) or not target_container:
        return None
    normalized_path = posixpath.normpath(output_path)
    private_root = f"/home/workspace/medflow/users/{actor}/datasets/"
    if output_path != normalized_path or not output_path.startswith(private_root):
        return None
    relative_parts = output_path[len(private_root):].split("/")
    if len(relative_parts) != 2:
        return None
    dataset_type, dataset_name = relative_parts
    if dataset_type not in {"raw", "sft", "dpo", "pt"} or not dataset_name:
        return None

    registered = protocol.get("registeredDataset")
    resource_id = None
    if isinstance(registered, dict):
        registered_path = str(registered.get("sourcePath") or "").strip()
        registered_type = str(registered.get("datasetType") or "").strip().lower()
        registered_name = str(registered.get("datasetName") or "").strip()
        registered_container = str(registered.get("containerName") or "").strip()
        registered_state = str(registered.get("storageState") or "").strip().lower()
        if (
            registered_path != output_path
            or registered_type != dataset_type
            or registered_name != dataset_name
            or registered_container != target_container
            or registered_state != "ready"
        ):
            return None
        resource_id = registered.get("resourceId")

    return {
        "resourceId": resource_id,
        "datasetType": dataset_type,
        "datasetName": dataset_name,
        "sourcePath": output_path,
        "ownerUserId": actor,
        "visibility": "private",
        "storageState": "ready",
        "containerName": target_container,
    }


def classify_advanced_filter_choice(message: str) -> str:
    normalized = re.sub(r"[\s。！!，,.]+", "", str(message or "").lower())
    if normalized in {
        "是", "需要", "要", "继续", "继续执行", "执行", "用", "好", "好的", "确认",
    }:
        return "accept"
    if normalized in {
        "否", "不需要", "不要", "不继续", "不执行", "不用", "取消", "关闭", "完成", "结束",
    }:
        return "decline"
    return "unknown"


def build_advanced_filter_request(output_path: str, user_choice: str) -> str:
    """Build an unambiguous follow-up that cannot be routed to preprocessing."""
    path = str(output_path or "").strip()
    if not path:
        raise ValueError("output_path is required")
    return (
        "执行数据高级筛选处理（score_based_filtering）\n"
        f"参数：input_folder={path}\n"
        f"用户选择：{user_choice}\n"
        "[系统补充] 用户已确认继续高级筛选；input_folder "
        "是刚完成的本次预处理产物路径。只执行高级筛选，不要重新执行数据预处理，"
        "也不要再追问 input_folder。"
    )
