# -*- coding: utf-8 -*-
"""
监控智能体工具（新）
- 从旧项目 training_monitor.py 迁移核心逻辑
- 供 agent2 monitor 智能体调用
"""

import asyncio
import concurrent.futures
import json
import math
import os
import re
import shlex
import subprocess
import threading
import time
import traceback
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from agentscope.credential import OpenAICredential
from agentscope.tool import ToolResponse
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from ._train_monitor_helpers import (
    extract_wandb_history,
    parse_output_log_history,
    parse_wandb_run_timestamp,
    select_wandb_run,
    select_wandb_run_by_start_time,
    validate_pid_binding,
    should_return_status_record,
    is_integer_step_value,
    infer_train_type_from_name,
    normalize_train_type,
    public_train_type,
    public_train_type_text,
)

def _load_agent_config():
    try:
        from utils.config import get_current_config, init_config

        try:
            return get_current_config()
        except Exception:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            return init_config(os.path.join(project_root, "config", "config.yaml"))
    except Exception:
        return None


_AGENT_CONFIG = _load_agent_config()
DEFAULT_BASE_URL = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "base_url", None)
    or os.getenv("AGENT3_MODEL_BASE_URL")
    or os.getenv("MODEL_BASE_URL", "")
)
DEFAULT_MODEL_NAME = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "name", None)
    or os.getenv("AGENT3_MODEL_NAME")
    or os.getenv("MODEL_NAME", "")
)
DEFAULT_MODEL_API_KEY = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "api_key", None)
    or os.getenv("AGENT3_MODEL_API_KEY")
    or os.getenv("MODEL_API_KEY", "")
)
DEFAULT_SAVES_ROOT = "/home/workspace/models/dpo_train/internal/saves"
DEFAULT_BATCH_TRAIN_ROOT = "/home/workspace/models/batch_train"
DEFAULT_PRETRAIN_ROOT = "/home/workspace/models/pretrain"
DEFAULT_WANDB_ROOT = "/home/workspace/llamafactory/wandb"
DEFAULT_LOG_WANDB_ROOT = "/home/workspace/log/wandb/wandb"
DEFAULT_VERL_WANDB_ROOT = "/home/workspace/verl/wandb"
DEFAULT_INSINFER_WANDB_ROOT = "/usr/local/insinfersystem/wandb"
DEFAULT_PRETRAIN_LOG_ROOT = "/home/workspace/log/pretrain"
DEFAULT_BATCH_TRAIN_LOG_ROOT = "/home/workspace/log/batch_train"
DEFAULT_HISTORY_LIMIT = 200
DEFAULT_TIME_WINDOW_MINUTES = 180
DEFAULT_STALE_MINUTES = 10
DEFAULT_MIN_HISTORY_FOR_LLM = 2
STARTING_TEXT = "训练正在启动中，请稍后查看实际训练loss和step/epoch"
DEFAULT_LOG_ROOT = "/home/workspace/log"
TRAINING_METRICS_WANDB_CACHE_TTL_SECONDS = 60

_RUNTIME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime"),
)
_WANDB_SNAPSHOT_DIR = os.path.join(_RUNTIME_DIR, "wandb_snapshots")
_TRAIN_PID_REGISTRY = os.path.join(_RUNTIME_DIR, "train_pid_registry.jsonl")
_TRAIN_STATUS_REGISTRY = os.path.join(_RUNTIME_DIR, "train_status_registry.jsonl")
_WANDB_URL_DIR = os.path.join(_RUNTIME_DIR, "wandb_urls")

_LLM_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LLM_LOOP_THREAD: Optional[threading.Thread] = None
_LLM_LOOP_LOCK = threading.Lock()
_LLM_LOOP_READY = threading.Event()
_LAST_OK_BY_CONTAINER: Dict[str, Dict[str, Any]] = {}
_WANDB_RUN_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _wandb_display_text(metrics: Dict[str, Any]) -> Optional[str]:
    wandb_url = metrics.get("wandb_url")
    if wandb_url is not None and str(wandb_url).strip():
        return str(wandb_url).strip()
    wandb_mode = str(metrics.get("wandb_mode") or "").strip().lower()
    if wandb_mode == "offline":
        return "离线模式，本次没有在线链接"
    if metrics.get("wandb_url_pending") is True:
        return "生成中，稍后刷新状态"
    return None


def _append_wandb_to_analysis_text(text: Any, metrics: Dict[str, Any]) -> str:
    content = str(text or "").strip()
    wandb_text = _wandb_display_text(metrics)
    if not wandb_text:
        return content
    if "wandb.ai" in content or "WandB" in content or "离线模式，本次没有在线链接" in content:
        return content
    wandb_line = f"- **WandB**：{wandb_text}"
    return f"{content}\n{wandb_line}" if content else wandb_line


def _payload_with_user_visible_wandb(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    enriched = dict(payload)
    enriched["analysis_text"] = _append_wandb_to_analysis_text(
        enriched.get("analysis_text") or "",
        metrics,
    )
    return enriched


def _payload_for_monitor_display(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        return payload
    status = str(payload.get("status") or metrics.get("status") or "").strip().lower()
    progress_realtime_confirmed = bool(metrics.get("progress_realtime_confirmed"))
    has_metric_anchor = (
        metrics.get("latest_loss") is not None
        or metrics.get("latest_step") is not None
        or metrics.get("latest_epoch") is not None
    )
    train_type = str(
        metrics.get("train_type_public")
        or metrics.get("train_type")
        or ""
    ).strip().lower()
    startup_error_reason = str(metrics.get("error_reason") or "")
    startup_error_reasons = {
        "wandb_run_not_found",
        "wandb_no_metrics",
    }
    unconfirmed_startup_train_types = {
        "lora",
        "lora_sft",
        "pretrain_lora",
        "pretrain_full",
        "enhanced",
        "multinode_lora_sft",
        "multinode_enhanced",
    }
    if (
        status in {"starting", "running"}
        and train_type in unconfirmed_startup_train_types
        and metrics.get("training_process_exists") is True
        and not has_metric_anchor
        and not progress_realtime_confirmed
        and startup_error_reason in startup_error_reasons
    ):
        sanitized_metrics = dict(metrics)
        sanitized_metrics["progress_percent"] = None
        sanitized_metrics["sub_stage"] = None
        sanitized_metrics["sub_stage_text"] = None
        debug = sanitized_metrics.get("debug")
        if isinstance(debug, dict):
            sanitized_debug = dict(debug)
            sanitized_debug["progress_sanitized_reason"] = "unconfirmed_starting_metrics"
            sanitized_metrics["debug"] = sanitized_debug
        sanitized_payload = dict(payload)
        sanitized_payload["metrics"] = sanitized_metrics
        return sanitized_payload
    return payload

def _monitor_protocol_hint(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    status = str(payload.get("status") or "unknown")
    progress_percent = metrics.get("progress_percent")
    latest_loss = metrics.get("latest_loss")
    latest_step = metrics.get("latest_step")
    latest_epoch = None if latest_step is not None else metrics.get("latest_epoch")
    required_params: List[str] = []
    error_reason = str(metrics.get("error_reason") or "")
    if error_reason in {"multiple_containers", "container_unknown", "container_required"}:
        required_params.append("container_name")
    if error_reason in {"pid_invalid", "pid_wandb_not_found", "pid_not_found"}:
        required_params.append("pid")

    if required_params:
        return {
            "type": "need_input",
            "agent": "monitor",
            "kind": "monitor_params",
            "message": payload.get("analysis_text") or "请补充监控所需参数。",
            "requiredParams": required_params,
            "missingParams": required_params,
            "action": "collect_params",
            "status": status,
            "jobType": "train",
            "container": metrics.get("container_name"),
            "pid": metrics.get("pid"),
            "errorReason": error_reason or None,
            "errorRecoverable": True,
        }

    return {
        "type": "monitor_status",
        "agent": "monitor",
        "message": payload.get("analysis_text") or "",
        "jobType": "train",
        "action": "monitor_status",
        "status": status,
        "container": metrics.get("container_name"),
        "pid": metrics.get("pid"),
        "pidAlive": metrics.get("pid_alive"),
        "trainType": metrics.get("train_type"),
        "trainTypeEn": metrics.get("train_type_public"),
        "trainTypeText": metrics.get("train_type_text"),
        "launchMode": metrics.get("launchMode") or metrics.get("launch_mode"),
        "isMultinode": (
            metrics.get("isMultinode")
            if metrics.get("isMultinode") is not None
            else metrics.get("is_multinode")
        ),
        "scriptName": metrics.get("scriptName") or metrics.get("script_name"),
        "latestLoss": latest_loss,
        "latestStep": latest_step,
        "latestEpoch": latest_epoch,
        "progressPercent": progress_percent,
        "subStage": metrics.get("sub_stage"),
        "subStageText": metrics.get("sub_stage_text"),
        "exportDir": metrics.get("export_dir"),
        "mergeStatus": metrics.get("merge_status"),
        "progress": {
            "percent": progress_percent,
            "step": latest_step,
            "epoch": latest_epoch,
            "loss": latest_loss,
        },
        "wandbUrl": metrics.get("wandb_url"),
        "wandbMode": metrics.get("wandb_mode"),
        "wandbUrlPending": metrics.get("wandb_url_pending"),
        "succeeded": metrics.get("succeeded"),
        "successState": metrics.get("success_state"),
        "completionReason": metrics.get("completion_reason"),
        "errorReason": error_reason or None,
        "errorSummary": metrics.get("error_summary") or None,
        "errorDetail": metrics.get("error_detail") or None,
        "errorRecoverable": bool(error_reason) and error_reason != "pid_ended_no_wandb",
    }


def _monitor_response(payload: Dict[str, Any]) -> ToolResponse:
    payload = _payload_with_user_visible_wandb(payload)
    display_payload = _payload_for_monitor_display(payload)
    text = json.dumps(payload, ensure_ascii=False)
    return ToolResponse(content=[TextBlock(type="text", text=text)],
        metadata={
            "success": payload.get("status") not in {"failed", "unknown"},
            "protocol_hint": _monitor_protocol_hint(display_payload),
        },
    )
_PROGRESS_STATE_BY_LOG: Dict[str, Dict[str, Any]] = {}

# CHANGE_REASON(2026-02-11):
# 复用 LLM client，避免频繁 aclose() 导致 event loop is closed；
# 同时减少建连开销，降低空响应概率。
_LLM_MODEL: Optional[OpenAIChatModel] = None
_LLM_MODEL_KEY: Optional[Tuple[str, str, str]] = None  # (base_url, model_name, api_key)


def _run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return process.returncode, (process.stdout or "").strip(), (process.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "")
        stderr = (exc.stderr or "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="ignore")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="ignore")
        timeout_msg = f"command_timeout_after_{timeout}s"
        err_text = f"{timeout_msg}: {' '.join(cmd)}"
        merged_err = "\n".join([part for part in [stderr.strip(), err_text] if part])
        return 124, str(stdout).strip(), merged_err
    except Exception as exc:
        return 1, "", f"command_exec_error: {exc}"


def _docker_exec(container: str, args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    return _run_cmd(["docker", "exec", container] + args, timeout=timeout)


def _docker_exec_python(
    container: str,
    script: str,
    args: Optional[List[str]] = None,
    timeout: int = 15,
) -> Tuple[int, str, str]:
    candidates = ["python", "python3", "/opt/conda/bin/python"]
    last: Tuple[int, str, str] = (1, "", "")
    for py in candidates:
        cmd = ["docker", "exec", container, py, "-c", script]
        if args:
            cmd.extend(args)
        code, out, err = _run_cmd(cmd, timeout=timeout)
        last = (code, out, err)
        if code == 0:
            return last
        err_lower = (err or "").lower()
        if (
            "executable file not found" in err_lower
            or "no such file or directory" in err_lower
            or "not found" in err_lower
        ) and not (out or "").strip():
            continue
        return last
    return last


def _should_use_pid_file_url(
    training_process_exists: bool,
    pid_started_at: Optional[datetime],
    file_mtime: Optional[float],
) -> bool:
    if not training_process_exists:
        return True
    if pid_started_at is None or file_mtime is None:
        return False
    return file_mtime >= pid_started_at.timestamp()


def _wandb_url_file(pid: Optional[str]) -> Optional[str]:
    if not pid:
        return None
    safe_pid = re.sub(r"[^\w.-]", "_", str(pid))
    return os.path.join(_WANDB_URL_DIR, f"{safe_pid}.txt")


def _read_wandb_url_file(pid: Optional[str]) -> Optional[str]:
    path = _wandb_url_file(pid)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        return None
    if content and "wandb.ai" in content:
        return content.splitlines()[0].strip()
    return None


def _list_pid_url_run_ids(current_pid: Optional[str]) -> List[str]:
    if not os.path.isdir(_WANDB_URL_DIR):
        return []
    run_ids: List[str] = []
    for name in os.listdir(_WANDB_URL_DIR):
        if not name.endswith(".txt"):
            continue
        pid_name = name[:-4]
        if current_pid and str(current_pid) == pid_name:
            continue
        path = os.path.join(_WANDB_URL_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            continue
        run_id = _extract_run_id_from_wandb_url(content)
        if run_id:
            run_ids.append(run_id)
    return run_ids


def _write_wandb_url_file(pid: Optional[str], wandb_url: Optional[str]) -> Optional[str]:
    if not pid or not wandb_url:
        return None
    os.makedirs(_WANDB_URL_DIR, exist_ok=True)
    path = _wandb_url_file(pid)
    if not path:
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(wandb_url).strip() + "\n")
    return path


def _extract_wandb_url_from_log_text(text: str) -> Optional[str]:
    if not text or "wandb" not in text.lower():
        return None
    run_match = re.search(
        r"https?://wandb\.ai/[^\s\"')>,;]+/runs/[A-Za-z0-9]+",
        text,
        re.IGNORECASE,
    )
    if run_match:
        return run_match.group(0).rstrip(").,;\"'")

    project_match = re.search(
        r"View project at\s+(https?://wandb\.ai/[^\s\"')>,;]+)",
        text,
        re.IGNORECASE,
    )
    local_run_match = re.search(
        r"(?:offline-)?run-\d{8}_\d{6}-([A-Za-z0-9]+)",
        text,
        re.IGNORECASE,
    )
    if project_match and local_run_match:
        project_url = project_match.group(1).rstrip(").,;\"'")
        run_id = local_run_match.group(1)
        return f"{project_url.rstrip('/')}/runs/{run_id}"
    return None


def _read_wandb_url_from_log_file(container: str, path: str) -> Optional[str]:
    script = r"""
import re, sys
path = sys.argv[1]
project_url = None
local_run_id = None
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "wandb" not in line.lower():
                continue
            run_match = re.search(r"https?://wandb\.ai/[^\s\"')>,;]+/runs/[A-Za-z0-9]+", line, re.I)
            if run_match:
                print(run_match.group(0).rstrip(").,;\"'"))
                raise SystemExit(0)
            if project_url is None:
                project_match = re.search(r"View project at\s+(https?://wandb\.ai/[^\s\"')>,;]+)", line, re.I)
                if project_match:
                    project_url = project_match.group(1).rstrip(").,;\"'")
            if local_run_id is None:
                local_run_match = re.search(r"(?:offline-)?run-\d{8}_\d{6}-([A-Za-z0-9]+)", line, re.I)
                if local_run_match:
                    local_run_id = local_run_match.group(1)
    if project_url and local_run_id:
        print(project_url.rstrip("/") + "/runs/" + local_run_id)
except Exception:
    pass
"""
    g_code, g_out, _ = _docker_exec_python(container, script, [path], timeout=10)
    if g_code == 0 and g_out:
        candidate = _extract_wandb_url_from_log_text(g_out.splitlines()[0].strip()) or g_out.splitlines()[0].strip()
        if candidate and "wandb.ai" in candidate:
            return candidate
    return None

def _find_log_file_by_pid(container: str, pid: str) -> Optional[str]:
    if not container or not pid:
        return None
    script = r"""
import os, sys
pid = sys.argv[1]
fd_dir = f"/proc/{pid}/fd"
items = []
if os.path.isdir(fd_dir):
    for name in os.listdir(fd_dir):
        path = os.path.join(fd_dir, name)
        try:
            target = os.readlink(path)
        except Exception:
            continue
        if not target or not os.path.isabs(target):
            continue
        if target.startswith(("socket:", "pipe:", "anon_inode:")):
            continue
        try:
            if not os.path.isfile(target):
                continue
            mtime = os.path.getmtime(target)
            size = os.path.getsize(target)
        except Exception:
            continue
        items.append((mtime, size, target))
if not items:
    print("")
    raise SystemExit(0)
def score(item):
    mtime, size, path = item
    s = 0
    if path.endswith(".log"):
        s += 2
    lower = path.lower()
    if "train" in lower:
        s += 1
    if "dpo" in lower:
        s += 1
    return (s, mtime, size)
items.sort(key=score, reverse=True)
print(items[0][2])
"""
    code, out, _ = _docker_exec_python(container, script, [str(pid)])
    if code == 0 and out:
        return out.strip()
    return None


def _extract_wandb_run_id(run_name: Optional[str], run_dir: Optional[str]) -> Optional[str]:
    for value in (run_name, os.path.basename(run_dir) if run_dir else None):
        if not value:
            continue
        m = re.search(r"(?:offline-)?run-\d{8}_\d{6}-([A-Za-z0-9]+)", value)
        if m:
            return m.group(1)
        if "-" in value:
            tail = value.split("-")[-1]
            if re.fullmatch(r"[A-Za-z0-9]+", tail):
                return tail
    return None


def _find_log_files_by_run_id(container: str, root: str, run_id: str, limit: int = 12) -> List[str]:
    if not container or not root or not run_id:
        return []
    script = r"""
import os, sys
root, run_id, limit_text = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    limit = max(1, int(limit_text))
except Exception:
    limit = 12
matches = []
for current_root, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
    for name in files:
        if not name.endswith(".log"):
            continue
        path = os.path.join(current_root, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                found = any(run_id in line for line in f)
        except Exception:
            continue
        if found:
            lower = path.lower()
            score = 0
            if "/wandb/" in lower:
                score += 20
            if lower.endswith("/logs/debug.log") or lower.endswith("/logs/debug-internal.log"):
                score += 20
            if "train" in lower:
                score -= 5
            if "dpo" in lower:
                score -= 2
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0
            matches.append((score, -mtime, path))
matches.sort()
for _score, _mtime, path in matches[:limit]:
    print(path)
"""
    code, out, _ = _docker_exec_python(container, script, [root, run_id, str(limit)], timeout=20)
    if code == 0 and out:
        return [line.strip() for line in out.splitlines() if line.strip()]
    return []


def _find_log_file_by_run_id(container: str, root: str, run_id: str) -> Optional[str]:
    matches = _find_log_files_by_run_id(container, root, run_id, limit=1)
    return matches[0] if matches else None

def _extract_run_id_from_wandb_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/runs/([A-Za-z0-9]+)", url)
    if m:
        return m.group(1)
    return None


def _find_wandb_run_dir_by_id(container: str, root: str, run_id: str) -> Optional[str]:
    if not container or not root or not run_id:
        return None
    cmd = (
        f"find {shlex.quote(root)} -maxdepth 1 -type d "
        f"\\( -name 'run-*-{run_id}' -o -name 'offline-run-*-{run_id}' \\) "
        "-print | head -n 1"
    )
    code, out, _ = _docker_exec(container, ["sh", "-c", cmd], timeout=10)
    if code == 0 and out:
        return out.splitlines()[0].strip()
    return None



def _training_metrics_wandb_cache_key(
    container: Optional[str],
    pid: Optional[str],
    train_type: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    if not container or not pid:
        return None
    normalized_train_type = normalize_train_type(train_type) or "unknown"
    return (str(container), str(pid), normalized_train_type)


def _wandb_output_log_exists(container: str, run_dir: Optional[str]) -> bool:
    if not container or not run_dir:
        return False
    output_log_path = f"{str(run_dir).rstrip('/')}/files/output.log"
    code, _, _ = _docker_exec(container, ["test", "-f", output_log_path], timeout=5)
    return code == 0


def _get_cached_wandb_run(
    container: str,
    pid: Optional[str],
    train_type: Optional[str],
    current_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    key = _training_metrics_wandb_cache_key(container, pid, train_type)
    if not key:
        return None
    entry = _WANDB_RUN_CACHE.get(key)
    if not entry:
        return None
    if float(entry.get("expires_at") or 0) < time.time():
        _WANDB_RUN_CACHE.pop(key, None)
        return None
    run_dir = str(entry.get("wandb_run_dir") or "").rstrip("/")
    if not run_dir:
        _WANDB_RUN_CACHE.pop(key, None)
        return None
    cached_run_id = str(entry.get("wandb_url_run_id") or "").strip()
    if current_run_id and cached_run_id and cached_run_id != str(current_run_id):
        _WANDB_RUN_CACHE.pop(key, None)
        return None
    if not _wandb_output_log_exists(container, run_dir):
        _WANDB_RUN_CACHE.pop(key, None)
        return None
    run = _run_record_from_dir(run_dir)
    if not run:
        _WANDB_RUN_CACHE.pop(key, None)
        return None
    run["_wandb_root"] = str(entry.get("wandb_root") or "").strip()
    run["_wandb_root_source"] = str(entry.get("wandb_root_source") or "cache").strip() or "cache"
    run["_cache_hit"] = True
    return run


def _remember_wandb_run(
    container: str,
    pid: Optional[str],
    train_type: Optional[str],
    wandb_run_dir: Optional[str],
    wandb_root: Optional[str],
    wandb_root_source: Optional[str],
    wandb_url_run_id: Optional[str],
) -> None:
    key = _training_metrics_wandb_cache_key(container, pid, train_type)
    if not key or not wandb_run_dir:
        return
    run_dir = str(wandb_run_dir).rstrip("/")
    _WANDB_RUN_CACHE[key] = {
        "wandb_run_dir": run_dir,
        "wandb_root": str(wandb_root or "").strip(),
        "wandb_root_source": str(wandb_root_source or "").strip(),
        "wandb_url_run_id": str(wandb_url_run_id or _extract_wandb_run_id(None, run_dir) or "").strip(),
        "expires_at": time.time() + TRAINING_METRICS_WANDB_CACHE_TTL_SECONDS,
        "cached_at": datetime.now().isoformat(),
    }


def _find_wandb_run_from_url(
    container: str,
    wandb_url: Optional[str],
    root_candidates: List[Tuple[str, str]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    run_id = _extract_run_id_from_wandb_url(wandb_url)
    if not run_id:
        return None, "wandb_url_run_id_missing"
    for candidate_root, candidate_source in root_candidates:
        matched_run_dir = _find_wandb_run_dir_by_id(container, candidate_root, run_id)
        if not matched_run_dir:
            continue
        run = _run_record_from_dir(matched_run_dir)
        if not run:
            continue
        run["_wandb_root"] = candidate_root
        run["_wandb_root_source"] = candidate_source
        run["_wandb_url_run_id"] = run_id
        return run, "wandb_url_match"
    return None, "wandb_url_run_not_found"


def _extract_param(cmd: str, name: str) -> Optional[str]:
    pattern = rf"{re.escape(name)}\s+([^\s]+)"
    match = re.search(pattern, cmd)
    if match:
        return match.group(1)
    pattern_eq = rf"{re.escape(name)}=([^\s]+)"
    match_eq = re.search(pattern_eq, cmd)
    if match_eq:
        return match_eq.group(1)
    return None


def _infer_wandb_root_from_cmd(cmd: str) -> Optional[str]:
    if not cmd:
        return None
    try:
        parts = shlex.split(cmd)
    except Exception:
        parts = cmd.split()
    for part in parts:
        token = str(part).strip("'\"()")
        if token.startswith("/usr/local/insinfersystem/") and not token.endswith(".py"):
            base_dir = os.path.dirname(token)
            return os.path.join(base_dir, "wandb")
    if "run_grpo_qwen3_8b_260417" in cmd:
        return DEFAULT_INSINFER_WANDB_ROOT
    if (
        "verl.trainer.main_ppo" in cmd
        or "grpo_trainer" in cmd
        or "run_qwen3-8b_260417.sh" in cmd
        or "/home/workspace/verl/" in cmd
    ):
        return DEFAULT_VERL_WANDB_ROOT
    if not parts:
        return None
    first = parts[0]
    first_base = os.path.basename(first)
    if first_base in {"python", "python3"} or first_base.startswith("python"):
        return None
    for part in parts:
        token = str(part).strip("'\"()")
        if token.startswith("/home/workspace/verl/"):
            return DEFAULT_VERL_WANDB_ROOT
    return None


def _infer_wandb_root_from_pid_record(pid_record: Optional[Dict[str, Any]]) -> Optional[str]:
    if not pid_record:
        return None
    wandb_root = str(pid_record.get("wandb_root") or "").strip()
    if wandb_root:
        return wandb_root
    script_name = str(pid_record.get("script_name") or "").lower()
    script_path = str(pid_record.get("script_path") or "")
    working_dir = str(pid_record.get("docker_working_dir") or "")
    command = str(pid_record.get("command") or "")
    if (
        "/usr/local/insinfersystem" in working_dir
        or "/usr/local/insinfersystem" in script_path
        or "/usr/local/insinfersystem" in command
        or "run_grpo_qwen3_8b_260417" in command
    ):
        return DEFAULT_INSINFER_WANDB_ROOT
    if "grpo" in script_name:
        return DEFAULT_VERL_WANDB_ROOT
    if "/home/workspace/verl" in working_dir or "/home/workspace/verl" in command:
        return DEFAULT_VERL_WANDB_ROOT
    if "/home/workspace/llamafactory" in working_dir:
        return DEFAULT_WANDB_ROOT
    return None


def _infer_wandb_mode(
    pid_record: Optional[Dict[str, Any]],
    wandb_run_name: Optional[str],
    wandb_run_dir: Optional[str],
) -> str:
    if pid_record:
        mode = str(pid_record.get("wandb_mode") or "").strip().lower()
        if mode in {"online", "offline"}:
            return mode
        env_vars = pid_record.get("env_vars")
        if isinstance(env_vars, dict):
            env_mode = str(env_vars.get("WANDB_MODE") or "").strip().lower()
            if env_mode == "offline":
                return "offline"
    for value in (wandb_run_name, os.path.basename(wandb_run_dir) if wandb_run_dir else None):
        if value and str(value).startswith("offline-run-"):
            return "offline"
    return "online"


def _is_log_viewer_cmd(cmd: str) -> bool:
    parts = shlex.split(cmd or "")
    if not parts:
        return False
    executable = os.path.basename(parts[0])
    return executable in {"tail", "grep", "cat", "less", "more", "watch"}

def _parse_ps_aux(ps_output: str) -> List[Dict[str, Any]]:
    processes: List[Dict[str, Any]] = []
    lines = ps_output.splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        if " ps aux" in line or "grep " in line:
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        cmd = " ".join(parts[10:])
        if _is_log_viewer_cmd(cmd):
            continue
        if not _is_training_cmd(cmd):
            continue
        pid = parts[1]
        output_dir = _extract_param(cmd, "--output_dir") or _extract_param(cmd, "--output-dir") or ""
        train_type = normalize_train_type(_infer_train_type(cmd))
        processes.append(
            {
                "pid": pid,
                "cmd": cmd,
                "output_dir": output_dir,
                "train_type": train_type,
            }
        )
    return processes


def _parse_ps_start_times(ps_output: str) -> Dict[str, datetime]:
    start_times: Dict[str, datetime] = {}
    for line in ps_output.splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        pid = parts[0]
        time_parts = parts[1:6]
        try:
            start_times[pid] = datetime.strptime(" ".join(time_parts), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            continue
    return start_times


def _is_training_cmd(cmd: str) -> bool:
    if "train_multinode_sft_pipeline" in cmd:
        return True
    if "train_multinode_dpo_pipeline" in cmd:
        return True
    if "deepspeed" in cmd and "train" in cmd:
        return True
    if "/usr/local/insinfersystem/train" in cmd:
        return True
    if "batch_train" in cmd:
        return True
    if "dpo_train_launcher" in cmd:
        return True
    if "llamafactory" in cmd and "train" in cmd:
        return True
    if "verl.trainer.main_ppo" in cmd:
        return True
    if "grpo_trainer" in cmd:
        return True
    if "run_qwen3-8b_260417.sh" in cmd:
        return True
    if "run_grpo_qwen3_8b_260417" in cmd:
        return True
    return False


def _infer_train_type(cmd: str) -> Optional[str]:
    if not cmd:
        return None
    named_type = normalize_train_type(infer_train_type_from_name(cmd))
    if named_type:
        return named_type
    stage = (_extract_param(cmd, "--stage") or "").strip().lower()
    finetuning_type = (
        _extract_param(cmd, "--finetuning_type")
        or _extract_param(cmd, "--finetuning-type")
        or ""
    ).strip().lower()
    if stage == "pt":
        return "pretrain_full" if finetuning_type == "full" else "pretrain_lora"
    stage_type = normalize_train_type(_extract_param(cmd, "--stage"))
    if stage_type:
        return stage_type
    return None


def _read_proc_cmdline(container: str, pid: Optional[str]) -> str:
    if not container or not pid:
        return ""
    script = r"""
import sys
pid = sys.argv[1]
try:
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        data = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    print(data.strip())
except Exception:
    print("")
"""
    code, out, _ = _docker_exec_python(container, script, [str(pid)])
    if code == 0 and out:
        return out.strip()
    return ""


def _pretrain_stdout_log_path(output_dir: Optional[str]) -> Optional[str]:
    return _trainer_stdout_log_path(output_dir, "pretrain_lora")


def _trainer_stdout_log_path(
    output_dir: Optional[str],
    train_type_public: Optional[str],
) -> Optional[str]:
    if not output_dir:
        return None
    name = os.path.basename(str(output_dir).rstrip("/"))
    if not name:
        return None
    if train_type_public in {"pretrain_lora", "pretrain_full"}:
        if not (name.startswith("model_pretrain_lora_") or name.startswith("model_pretrain_full_")):
            return None
        return f"{DEFAULT_PRETRAIN_LOG_ROOT.rstrip('/')}/{name}-train-log.log"
    if train_type_public in {"lora_sft", "full_sft"}:
        if not (name.startswith("model_lora_") or name.startswith("model_full_")):
            return None
        return f"{DEFAULT_BATCH_TRAIN_LOG_ROOT.rstrip('/')}/{name}-train-log.log"
    return None


def _find_pretrain_log_by_output_dir(container: str, output_dir: Optional[str]) -> Optional[str]:
    path = _trainer_stdout_log_path(output_dir, "pretrain_lora")
    if path and _docker_exec(container, ["test", "-f", path])[0] == 0:
        return path
    return None


def _find_trainer_stdout_log_by_output_dir(
    container: str,
    output_dir: Optional[str],
    train_type_public: Optional[str],
) -> Optional[str]:
    path = _trainer_stdout_log_path(output_dir, train_type_public)
    if path and _docker_exec(container, ["test", "-f", path])[0] == 0:
        return path
    return None


def _resolve_output_dir_from_pid_context(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(pid_record, dict):
        for key in ("output_dir", "outputDir"):
            value = str(pid_record.get(key) or "").strip()
            if value:
                return value, f"pid_registry:{key}"
        command = str(pid_record.get("command") or "")
        output_dir = _extract_param(command, "--output_dir") or _extract_param(command, "--output-dir")
        if output_dir:
            return output_dir, "pid_registry:command"
        script_args = pid_record.get("script_args")
        if isinstance(script_args, dict):
            for key in ("output_dir", "output-dir"):
                value = str(script_args.get(key) or "").strip()
                if value:
                    return value, f"pid_registry:script_args:{key}"

    cmdline = _read_proc_cmdline(container, pid)
    output_dir = _extract_param(cmdline, "--output_dir") or _extract_param(cmdline, "--output-dir")
    if output_dir:
        return output_dir, "proc_cmdline"
    return None, None


def _select_process(
    processes: List[Dict[str, Any]],
    start_times: Dict[str, datetime],
    train_type: Optional[str],
    session_output_dir: Optional[str],
    time_window_minutes: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not processes:
        return None, "no_process"

    candidates = [p for p in processes if p.get("output_dir")]
    if not candidates:
        candidates = processes[:]

    if train_type:
        typed = [p for p in candidates if (p.get("train_type") or "").lower() == train_type.lower()]
        if typed:
            candidates = typed

    # CHANGE_REASON(2026-02-06): 禁用 session_output_dir 匹配来源（上层不再提供）
    if session_output_dir:
        matched = [p for p in candidates if p.get("output_dir") == session_output_dir]
        if matched:
            candidates = matched

    if time_window_minutes > 0:
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        recent = []
        for p in candidates:
            start_time = start_times.get(p.get("pid", ""))
            if start_time and start_time >= cutoff:
                recent.append(p)
        if recent:
            candidates = recent

    candidates.sort(key=lambda p: start_times.get(p.get("pid", ""), datetime.min), reverse=True)
    return candidates[0], "ps_aux"


def _pick_container_by_training_process(names: List[str], timeout: int = 5) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    在多个运行容器中，自动挑选“唯一一个”包含训练进程的容器。
    - 训练进程识别规则复用 _parse_ps_aux() / _is_training_cmd()
    - 若唯一命中，则返回该容器名
    - 否则返回 None，并附带 debug 信息
    """
    debug: Dict[str, Any] = {"checked": [], "candidates": []}

    for name in names:
        try:
            code, out, err = _docker_exec(name, ["ps", "aux"], timeout=timeout)
        except Exception as exc:
            debug["checked"].append(
                {"container": name, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        if code != 0:
            debug["checked"].append(
                {"container": name, "ok": False, "reason": err or out or "ps_aux_failed"}
            )
            continue

        procs = _parse_ps_aux(out)
        debug["checked"].append({"container": name, "ok": True, "proc_count": len(procs)})

        if procs:
            debug["candidates"].append(
                {
                    "container": name,
                    "proc_count": len(procs),
                    "selected_cmd": procs[0].get("cmd"),
                }
            )

    if len(debug["candidates"]) == 1:
        return debug["candidates"][0]["container"], debug

    return None, debug


def _get_container_name(container_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    获取要监控的容器名。
    修复点：
    - 当 docker ps 返回多个容器时，不再直接报 multiple_containers
    - 尝试自动选择“唯一一个包含训练进程”的容器
    """
    if container_name:
        return container_name, None

    code, out, err = _run_cmd(["docker", "ps", "--format", "{{.Names}}"])
    if code != 0:
        return None, f"docker ps failed: {err or out}"

    names = [n.strip() for n in out.splitlines() if n.strip()]
    if not names:
        return None, "no_running_container"
    if len(names) == 1:
        return names[0], None

    # CHANGE_REASON(2026-02-06): 多容器时自动挑选“唯一包含训练进程”的容器，避免误报 multiple_containers
    picked, _debug = _pick_container_by_training_process(names)

    if picked:
        return picked, None

    return None, "multiple_containers"


def _extract_log_file(cmd: str) -> Optional[str]:
    log_file = _extract_param(cmd, "--log_file") or _extract_param(cmd, "--log-file")
    if log_file:
        if log_file.startswith("/"):
            return log_file
        log_dir = _extract_param(cmd, "--log_dir") or _extract_param(cmd, "--log-dir")
        if log_dir:
            return os.path.join(log_dir, log_file)
        return log_file
    parts = cmd.split()
    for i, part in enumerate(parts):
        if part == ">" and i + 1 < len(parts):
            next_part = parts[i + 1]
            if next_part not in ["2>&1", "&>", ">>"]:
                if "2>&1" in next_part:
                    if i > 0:
                        return parts[i - 1]
                else:
                    return next_part
    redirect_patterns = [
        r">\s+([^\s&>]+)",
        r">>\s+([^\s&>]+)",
    ]
    for pattern in redirect_patterns:
        match = re.search(pattern, cmd)
        if match:
            return match.group(1)
    return None


def _resume_checkpoint_from_pid_record(pid_record: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(pid_record, dict):
        return None
    env_vars = pid_record.get("env_vars")
    if isinstance(env_vars, dict):
        resume = env_vars.get("RESUME") or env_vars.get("resume_from_checkpoint")
        if resume is not None and str(resume).strip():
            return str(resume).strip()
    script_args = pid_record.get("script_args")
    if isinstance(script_args, dict):
        resume = script_args.get("RESUME") or script_args.get("resume_from_checkpoint")
        if resume is not None and str(resume).strip():
            return str(resume).strip()
    command = str(pid_record.get("command") or "")
    for pattern in (
        r"(?:^|\s)-e\s+RESUME=([^\s]+)",
        r"(?:^|\s)RESUME=([^\s]+)",
        r"(?:^|\s)--resume_from_checkpoint\s+([^\s]+)",
        r"(?:^|\s)--resume-from-checkpoint\s+([^\s]+)",
    ):
        match = re.search(pattern, command)
        if match:
            return match.group(1).strip("'\"")
    return None


def _checkpoint_step_from_path(checkpoint: Optional[str]) -> Optional[int]:
    if not checkpoint:
        return None
    match = re.search(r"(?:^|[/\\])checkpoint-(\d+)(?:$|[/\\])", str(checkpoint))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _latest_integer_step_from_history(history: Any) -> Optional[Any]:
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if is_integer_step_value(step):
            return step
    return None


def _apply_finished_wandb_status(
    status: str,
    iteration_finished: bool,
    training_process_exists: bool,
) -> str:
    """Do not let a terminal W&B state overwrite a detected training error."""
    if iteration_finished and not training_process_exists and status != "failed":
        return "finished"
    return status


def _format_step_like(original: Any, value: float) -> Any:
    if isinstance(original, int):
        return int(value)
    if isinstance(original, float):
        return float(value)
    if float(value).is_integer():
        return int(value)
    return value


def _apply_resume_step_offset(
    latest_step: Any,
    current_step: Any,
    pid_record: Optional[Dict[str, Any]],
    force_local_step: bool = False,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    checkpoint = _resume_checkpoint_from_pid_record(pid_record)
    checkpoint_step = _checkpoint_step_from_path(checkpoint)
    numeric_latest = _to_float_or_none(latest_step)
    if checkpoint_step is None or checkpoint_step <= 0 or numeric_latest is None:
        return latest_step, None
    if numeric_latest >= checkpoint_step and not force_local_step:
        return latest_step, None

    adjusted = checkpoint_step + numeric_latest
    numeric_current = _to_float_or_none(current_step)
    if numeric_current is not None and numeric_current >= checkpoint_step:
        adjusted = max(adjusted, numeric_current)
    return _format_step_like(latest_step, adjusted), {
        "resume_checkpoint": checkpoint,
        "resume_checkpoint_step": checkpoint_step,
        "raw_latest_step": latest_step,
        "adjusted_latest_step": _format_step_like(latest_step, adjusted),
        "step_adjustment": "checkpoint_offset",
    }


def _apply_resume_history_offset(
    history: Any,
    pid_record: Optional[Dict[str, Any]],
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    if not isinstance(history, list) or not history:
        return history, None
    checkpoint = _resume_checkpoint_from_pid_record(pid_record)
    checkpoint_step = _checkpoint_step_from_path(checkpoint)
    if checkpoint_step is None or checkpoint_step <= 0:
        return history, None

    numeric_steps: List[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        step = _to_float_or_none(item.get("step"))
        if step is not None:
            numeric_steps.append(step)
    if not numeric_steps or min(numeric_steps) >= checkpoint_step:
        return history, None

    adjusted_history: List[Any] = []
    for item in history:
        if not isinstance(item, dict):
            adjusted_history.append(item)
            continue
        adjusted_item = dict(item)
        step = _to_float_or_none(item.get("step"))
        if step is not None:
            adjusted_item["raw_step"] = item.get("step")
            adjusted_item["step"] = _format_step_like(item.get("step"), checkpoint_step + step)
        adjusted_history.append(adjusted_item)

    return adjusted_history, {
        "resume_checkpoint": checkpoint,
        "resume_checkpoint_step": checkpoint_step,
        "raw_history_step_min": min(numeric_steps),
        "raw_history_step_max": max(numeric_steps),
        "history_step_adjustment": "checkpoint_offset",
    }


def _parse_output_dir_from_log(container: str, log_file: str) -> Optional[str]:
    if not log_file:
        return None
    script = r"""
import os, re, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("")
    raise SystemExit(0)
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()[-200:]
pattern = re.compile(r"output_dir\s*(?:is|=|:)\s*([/\w\-\.]+)")
for line in reversed(lines):
    m = pattern.search(line)
    if m:
        print(m.group(1))
        raise SystemExit(0)
print("")
"""
    code, out, _ = _docker_exec_python(container, script, [log_file])
    if code == 0 and out:
        return out.strip()
    return None


def _scan_latest_dir(
    container: str,
    root: str,
    cutoff_ts: float,
) -> Optional[str]:
    script = r"""
import json, os, sys
root = sys.argv[1]
cutoff = float(sys.argv[2])
if not os.path.isdir(root):
    print("")
    raise SystemExit(0)
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
if not dirs:
    print("")
    raise SystemExit(0)
items = []
for d in dirs:
    path = os.path.join(root, d)
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        continue
    if mtime >= cutoff:
        items.append((mtime, path))
if not items:
    print("")
    raise SystemExit(0)
items.sort(key=lambda x: x[0], reverse=True)
print(items[0][1])
"""
    code, out, _ = _docker_exec_python(container, script, [root, str(cutoff_ts)])
    if code == 0 and out:
        return out.strip()
    return None


def _file_mtime(container: str, path: str) -> Optional[float]:
    script = r"""
import os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("")
    raise SystemExit(0)
print(os.path.getmtime(path))
"""
    code, out, _ = _docker_exec_python(container, script, [path])
    if code == 0 and out:
        try:
            return float(out.strip())
        except Exception:
            return None
    return None


def _load_pid_registry() -> List[Dict[str, Any]]:
    if not os.path.exists(_TRAIN_PID_REGISTRY):
        return []
    records: List[Dict[str, Any]] = []
    with open(_TRAIN_PID_REGISTRY, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _get_pid_record_by_pid(container: str, pid: str) -> Optional[Dict[str, Any]]:
    if not container or not pid:
        return None
    records = [
        r for r in _load_pid_registry() if r.get("container") == container and str(r.get("pid")) == str(pid)
    ]
    if not records:
        return None
    records.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return records[0]


def _get_latest_pid_record(
    container: str,
    max_age_minutes: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not container:
        return None
    records = [r for r in _load_pid_registry() if r.get("container") == container]
    if not records:
        return None

    def _parse_time(v: Any) -> Optional[datetime]:
        if not v:
            return None
        try:
            return datetime.fromisoformat(str(v))
        except Exception:
            return None

    records.sort(
        key=lambda r: _parse_time(r.get("started_at")) or datetime.min,
        reverse=True,
    )
    latest = records[0]
    if max_age_minutes is not None:
        started_at = _parse_time(latest.get("started_at"))
        if started_at:
            age_minutes = (datetime.now() - started_at).total_seconds() / 60
            if age_minutes > max_age_minutes:
                return None
    return latest


def _read_launch_log_tail(
    container: str,
    pid_record: Optional[Dict[str, Any]],
    line_count: int = 120,
) -> str:
    if not container or not isinstance(pid_record, dict):
        return ""
    path = str(pid_record.get("launch_log_file") or "").strip()
    if not path:
        return ""
    safe_count = max(1, min(int(line_count), 500))
    code, out, _ = _docker_exec(
        container,
        ["sh", "-c", f"test -f {shlex.quote(path)} && tail -n {safe_count} {shlex.quote(path)}"],
    )
    return out.strip() if code == 0 else ""


def _truncate_log_line_for_user(line: str, max_chars: int = 500) -> str:
    text = str(line or "")
    if len(text) <= max_chars:
        return text
    head_chars = max(80, max_chars // 2)
    tail_chars = max(80, max_chars - head_chars - 20)
    return f"{text[:head_chars]} ... [已截断] ... {text[-tail_chars:]}"


def _format_launch_log_tail_for_user(
    log_tail: str,
    *,
    log_path: Optional[str] = None,
    max_lines: int = 30,
    max_chars: int = 2000,
) -> str:
    if not log_tail:
        return ""
    lines = [line.rstrip() for line in str(log_tail).splitlines() if line.strip()]
    if max_lines > 0:
        lines = lines[-max_lines:]
    text = "\n".join(_truncate_log_line_for_user(line) for line in lines).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = f"...（已截断，仅显示最后 {max_chars} 字）\n{text[-max_chars:]}"
    if log_path:
        return f"日志路径：{log_path}\n{text}".strip()
    return text


def _extract_ray_disk_warning(log_text: str) -> Optional[Dict[str, str]]:
    if not log_text:
        return None
    pattern = re.compile(
        r"(?P<path>/[^\s]+)\s+is\s+over\s+(?P<threshold>\d+)%\s+full,\s+"
        r"available\s+space:\s+(?P<available>[^;]+);\s+capacity:\s+(?P<capacity>[^.]+(?:\.\d+)?\s*[A-Za-z]+)",
        re.IGNORECASE,
    )
    match = pattern.search(log_text)
    if not match:
        return None
    return {
        "path": match.group("path").strip(),
        "threshold_percent": match.group("threshold").strip(),
        "available": match.group("available").strip(),
        "capacity": match.group("capacity").strip(),
    }


def _format_disk_warning_summary(warning: Optional[Dict[str, str]]) -> str:
    if not warning:
        return ""
    path = warning.get("path") or "磁盘目录"
    threshold = warning.get("threshold_percent") or "95"
    available = warning.get("available") or "未知"
    return f"磁盘提示：{path} 使用率超过 {threshold}%，剩余空间 {available}。"


def _summarize_training_error_from_logs(*texts: Any) -> str:
    joined = "\n".join(str(text or "") for text in texts if str(text or "").strip())
    if not joined:
        return ""
    lower = joined.lower()
    lines = [line.strip() for line in joined.splitlines() if line.strip()]

    if "cannot find sufficient samples" in lower or (
        "sufficient samples" in lower and "dataset size" in lower
    ):
        return "可用训练样本不足，数据集太小或预处理后没有足够样本，请增加数据量或检查数据格式。"
    if "cuda out of memory" in lower or re.search(r"\boom\b", lower):
        return "显存不足，训练进程被 CUDA OOM 中断。"
    if (
        "no space left on device" in lower
        or "disk quota exceeded" in lower
        or " is over " in lower and "available space" in lower
        or "磁盘" in joined and ("空间不足" in joined or "剩余空间" in joined)
    ):
        return "磁盘空间不足或剩余空间过低，训练无法继续写入日志或模型文件。"
    for line in lines:
        line_lower = line.lower()
        if not re.search(r"(?:no checkpoint found|checkpoint .* not found|model .* not found|no such file or directory|file not found|not found|cannot find)", line_lower):
            continue
        if re.search(r"(?:checkpoint|ckpt|model|adapter|safetensors|\.bin\b)", line_lower):
            return "找不到模型或 checkpoint 文件，训练无法继续。"
    for line in lines:
        line_lower = line.lower()
        if re.search(r"(?:no such file or directory|file not found|not found|cannot find)", line_lower):
            if line_lower.lstrip().startswith(("[", "+", "/usr/local/insinfersystem/train")):
                continue
            return "找不到训练所需的文件或路径，请检查启动参数中的目录配置。"
    if re.search(r"(?:no checkpoint found|checkpoint .* not found|model .* not found)", lower):
        return "找不到模型或 checkpoint 文件，训练无法继续。"
    if re.search(r"(?:no such file or directory|file not found|not found|cannot find)", lower):
        return "找不到训练所需的文件或路径，请检查启动参数中的目录配置。"
    if re.search(r"(?:returned non-zero|non-zero exit|exit status|exit code|calledprocesserror)", lower):
        return "训练脚本返回非零退出码，可能是启动参数、环境或依赖配置错误。"
    if re.search(r"(?:sigkill|killed|137\b|exitcode_137)", lower):
        return "训练进程被系统结束，可能是内存/显存不足、资源限制或手动终止。"
    if "segmentation fault" in lower:
        return "训练进程发生段错误，可能与底层依赖、驱动或运行环境有关。"
    if "permission denied" in lower:
        return "训练进程没有访问所需文件或目录的权限。"
    if "runtimeerror" in lower or "traceback" in lower or "exception" in lower:
        return "训练脚本抛出运行时异常，请查看日志中的 traceback 定位具体报错。"
    if "loss_not_finite" in lower or re.search(r"\b(?:nan|inf)\b", lower):
        return "训练指标出现 NaN 或 Inf，训练未能稳定继续。"
    return ""


def _load_status_registry() -> List[Dict[str, Any]]:
    if not os.path.exists(_TRAIN_STATUS_REGISTRY):
        return []
    records: List[Dict[str, Any]] = []
    with open(_TRAIN_STATUS_REGISTRY, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _get_status_record_by_pid(container: str, pid: str) -> Optional[Dict[str, Any]]:
    if not container or not pid:
        return None
    records = [
        r
        for r in _load_status_registry()
        if r.get("container") == container and str(r.get("pid")) == str(pid)
    ]
    if not records:
        return None
    records.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return records[0]


def _record_train_status(
    container: Optional[str],
    pid: Optional[str],
    status: str,
    error_reason: Optional[str],
    latest_loss: Optional[Any],
    latest_epoch: Optional[Any],
    latest_step: Optional[Any],
    wandb_run_name: Optional[str],
    wandb_run_dir: Optional[str],
    wandb_mode: str,
    output_log_error: Optional[str],
    output_log_success: bool,
    last_update_time: Optional[str],
    completion_reason: Optional[str] = None,
    completion_source: Optional[str] = None,
) -> None:
    if not container or not pid:
        return
    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    record = {
        "container": container,
        "pid": str(pid),
        "status": status,
        "error_reason": error_reason,
        "latest_loss": latest_loss,
        "latest_epoch": latest_epoch,
        "latest_step": latest_step,
        "wandb_run_name": wandb_run_name,
        "wandb_run_dir": wandb_run_dir,
        "wandb_mode": wandb_mode,
        "output_log_error": output_log_error,
        "output_log_success": output_log_success,
        "completion_reason": completion_reason,
        "completion_source": completion_source,
        "last_update_time": last_update_time,
        "updated_at": datetime.now().isoformat(),
    }
    with open(_TRAIN_STATUS_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_payload_from_status_record(
    container: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    status = record.get("status") or "unknown"
    latest_loss = record.get("latest_loss")
    latest_epoch = record.get("latest_epoch")
    latest_step = record.get("latest_step")
    error_reason = record.get("error_reason")
    last_update_time = record.get("last_update_time")
    wandb_mode = record.get("wandb_mode") or _infer_wandb_mode(record, record.get("wandb_run_name"), record.get("wandb_run_dir"))
    succeeded = bool(status == "finished" and not error_reason)
    completion_reason = record.get("completion_reason") or ("status_registry_finished" if succeeded else None)
    completion_source = record.get("completion_source") or ("status_registry" if succeeded else None)
    error_summary = _summarize_training_error_from_logs(
        record.get("output_log_error"),
        record.get("launch_log_tail"),
        error_reason,
    )
    analysis_text = "训练状态已记录。"
    if status == "finished":
        analysis_text = f"训练已结束，最后loss为{latest_loss}，请根据需要检查最终指标与模型导出。"
    elif status == "failed":
        if error_summary:
            analysis_text = f"当前训练状态异常。原因：{error_summary}"
        else:
            analysis_text = f"当前训练状态异常，错误原因为“{error_reason}”，请检查日志与容器进程以进一步排查问题。"
    elif status == "interrupted":
        reason_text = error_summary or f"未检测到正常完成标记（原因：{error_reason}），请检查日志或确认是否手动终止。"
        analysis_text = (
            f"训练已中断或提前停止，最后loss为{latest_loss}，"
            f"原因：{reason_text}"
        )
    record_train_type = normalize_train_type(record.get("train_type") or record.get("trainType"))
    record_launch_mode = record.get("launch_mode") or record.get("launchMode")
    record_script = record.get("script_name") or record.get("scriptName")
    payload = {
        "status": status,
        "metrics": {
            "container_name": container,
            "pid": record.get("pid"),
            "pid_alive": False,
            "train_type": record_train_type or "unknown",
            "train_type_public": public_train_type(record_train_type, record_launch_mode, record_script),
            "train_type_text": public_train_type_text(record_train_type, record_launch_mode, record_script),
            "launch_mode": record_launch_mode,
            "launchMode": record_launch_mode,
            "is_multinode": record_launch_mode == "multinode",
            "isMultinode": record_launch_mode == "multinode",
            "script_name": record_script,
            "scriptName": record_script,
            "training_process_exists": False,
            "latest_loss": latest_loss,
            "latest_epoch": latest_epoch,
            "latest_step": latest_step,
            "last_update_time": last_update_time,
            "wandb_run_name": record.get("wandb_run_name"),
            "wandb_run_dir": record.get("wandb_run_dir"),
            "wandb_mode": wandb_mode,
            "finished": status == "finished",
            "succeeded": succeeded,
            "success_state": "succeeded" if succeeded else None,
            "completion_reason": completion_reason,
            "completion_source": completion_source,
            "output_log_success": record.get("output_log_success"),
            "error_reason": error_reason,
            "error_summary": error_summary or None,
            "error_detail": record.get("output_log_error"),
            "history": [],
            "history_count": 0,
            "loss_source": "status_registry",
            "debug": {
                "status_registry": True,
                "output_log_error": record.get("output_log_error"),
                "output_log_success": record.get("output_log_success"),
                "updated_at": record.get("updated_at"),
            },
        },
        "analysis_text": analysis_text,
    }
    return payload

def _check_pid_alive(container: str, pid: str, timeout: int = 5) -> bool:
    if not container or not pid:
        return False
    code, out, err = _docker_exec(
        container,
        ["sh", "-c", f"ps -p {pid} -o pid="],
        timeout=timeout,
    )
    if code == 124 or ("command_timeout" in (err or "").lower()):
        code, out, _ = _docker_exec(
            container,
            ["sh", "-c", f"ps -p {pid} -o pid="],
            timeout=max(timeout + 7, 12),
        )
    return code == 0 and bool(out.strip())


def _list_wandb_runs(container: str, root: str) -> List[Dict[str, Any]]:
    script = r"""
import json, os, sys
root = sys.argv[1]
runs = []
if os.path.isdir(root):
    for name in os.listdir(root):
        if not (name.startswith("run-") or name.startswith("offline-run-")):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            runs.append({
                "name": name,
                "path": path,
                "mtime": os.path.getmtime(path),
            })
print(json.dumps({"runs": runs}))
"""
    code, out, _ = _docker_exec_python(container, script, [root])
    if code != 0 or not out:
        shell_cmd = (
            f"for d in {shlex.quote(root)}/run-* {shlex.quote(root)}/offline-run-*; "
            "do [ -d \"$d\" ] && printf '%s\\n' \"$d\"; done"
        )
        ls_code, ls_out, _ = _docker_exec(container, ["sh", "-c", shell_cmd])
        if ls_code != 0 or not ls_out:
            return []
        runs: List[Dict[str, Any]] = []
        for line in ls_out.splitlines():
            path = line.strip()
            if not path:
                continue
            name = os.path.basename(path)
            if not (name.startswith("run-") or name.startswith("offline-run-")):
                continue
            mtime = 0.0
            ts = parse_wandb_run_timestamp(name)
            if ts:
                mtime = ts.timestamp()
            runs.append({"name": name, "path": path, "mtime": mtime})
        return runs
    try:
        payload = json.loads(out)
        return payload.get("runs", []) or []
    except Exception:
        return []


def _wandb_root_candidates(
    primary_root: str,
    primary_source: str,
    user_supplied_wandb_root: bool,
) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    def add(root: Optional[str], source: str) -> None:
        value = str(root or "").strip()
        if not value:
            return
        if any(existing == value for existing, _ in candidates):
            return
        candidates.append((value, source))

    add(primary_root, primary_source)
    if not user_supplied_wandb_root:
        for fallback_root in (
            DEFAULT_LOG_WANDB_ROOT,
            DEFAULT_WANDB_ROOT,
            DEFAULT_INSINFER_WANDB_ROOT,
        ):
            add(fallback_root, f"fallback:{fallback_root}")
    return candidates


def _read_wandb_json_file(container: str, path: str) -> Optional[Dict[str, Any]]:
    code, out, _ = _docker_exec(container, ["sh", "-c", f"cat {shlex.quote(path)}"])
    if code != 0 or not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def _find_wandb_run_by_pid(
    container: str,
    runs: List[Dict[str, Any]],
    pid: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not pid:
        return None, "pid_not_provided"
    for run in runs:
        run_dir = run.get("path")
        if not run_dir:
            continue
        meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
        meta = _read_wandb_json_file(container, meta_path)
        if not meta:
            continue
        meta_pid = meta.get("pid") or meta.get("process_id")
        if meta_pid is not None and str(meta_pid) == str(pid):
            return run, "metadata_pid_match"
    return None, "metadata_pid_not_found"


def _read_wandb_history_tail(
    container: str,
    path: str,
    history_limit: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    cmd = f"tail -n {int(history_limit)} {shlex.quote(path)}"
    code, out, _ = _docker_exec(container, ["sh", "-c", cmd], timeout=15)
    records: List[Dict[str, Any]] = []
    if code == 0 and out:
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    count = None
    wc_code, wc_out, _ = _docker_exec(container, ["sh", "-c", f"wc -l {shlex.quote(path)}"])
    if wc_code == 0 and wc_out:
        try:
            count = int(wc_out.strip().split()[0])
        except Exception:
            count = None
    return records, count


def _parse_output_log_progress(lines: List[str]) -> Dict[str, Any]:
    progress_percent = None
    current_step = None
    total_steps = None
    elapsed_time = None
    remaining_time = None
    progress_line = None

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def _strip_ansi(text: str) -> str:
        return ansi_re.sub("", text)

    def _parse_time(value: str) -> Optional[str]:
        if not value:
            return None
        if re.fullmatch(r"\d+:\d{2}(?::\d{2}(?:\.\d+)?)?", value):
            return value
        return None

    def _is_non_training_tqdm(line: str) -> bool:
        lower = line.lower()
        return any(
            marker in lower
            for marker in (
                "converting format of dataset",
                "running tokenizer on dataset",
                "loading checkpoint shards",
                "generating train split",
                "generating validation split",
                "map (num_proc",
                "filter (num_proc",
            )
        )

    candidates: List[Dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        cleaned = _strip_ansi(line).strip()
        if not cleaned or "%" not in cleaned or "/" not in cleaned or "[" not in cleaned:
            continue
        if _is_non_training_tqdm(cleaned):
            continue
        pct_match = re.search(r"(\d+(?:\.\d+)?)%", cleaned)
        step_match = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
        bracket_match = re.search(r"\[(.*?)\]", cleaned)
        if not (pct_match and step_match and bracket_match):
            continue

        item_current_step: Optional[int] = None
        item_total_steps: Optional[int] = None
        try:
            item_current_step = int(step_match.group(1))
            item_total_steps = int(step_match.group(2))
        except Exception:
            item_current_step = None
            item_total_steps = None

        item_elapsed_time = None
        item_remaining_time = None
        time_part = bracket_match.group(1).split(",", 1)[0].strip()
        if "<" in time_part:
            left, right = time_part.split("<", 1)
            item_elapsed_time = _parse_time(left.strip())
            item_remaining_time = _parse_time(right.strip())
        else:
            item_elapsed_time = _parse_time(time_part)

        candidates.append(
            {
                "progress_percent": f"{pct_match.group(1)}%",
                "current_step": item_current_step,
                "total_steps": item_total_steps,
                "elapsed_time": item_elapsed_time,
                "remaining_time": item_remaining_time,
                "progress_line": cleaned,
                "progress_index": line_index,
            }
        )

    if candidates:
        latest = candidates[-1]
        progress_percent = latest.get("progress_percent")
        current_step = latest.get("current_step")
        total_steps = latest.get("total_steps")
        elapsed_time = latest.get("elapsed_time")
        remaining_time = latest.get("remaining_time")
        progress_line = latest.get("progress_line")

    observed_steps = []
    for item in candidates:
        step_val = item.get("current_step")
        if isinstance(step_val, int):
            observed_steps.append(step_val)
    unique_steps = sorted(set(observed_steps))

    return {
        "progress_percent": progress_percent,
        "current_step": current_step,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "remaining_time": remaining_time,
        "progress_line": progress_line,
        "progress_index": latest.get("progress_index") if candidates else None,
        "progress_candidate_count": len(candidates),
        "progress_unique_step_count": len(unique_steps),
    }


def _lines_have_training_start_signal(lines: List[str]) -> bool:
    for line in lines or []:
        lower = str(line or "").lower()
        if any(
            marker in lower
            for marker in (
                "***** running training *****",
                "training started.",
                "start training",
                "process rank:",
                "distributed training: true",
                "initializing torchbackend",
                "world info dict:",
                "multinode_train_entry",
                "running on the following workers",
                "loading weights file",
                "detected deepspeed zero-3",
            )
        ):
            return True
    return False

def _dpo_sub_stage_text(sub_stage: Optional[str]) -> Optional[str]:
    return {
        "training": "训练中",
        "evaluation": "训练后 Evaluation 中",
        "merge": "Merge/Export 中",
        "finished": "已完成",
    }.get(str(sub_stage or "").strip().lower())


def _extract_export_dir_from_text(text: str) -> Optional[str]:
    for pattern in (
        r"export_dir\s+is\s*:\s*(\S+)",
        r"--export_dir\s+(\S+)",
        r"Configuration saved in\s+(\S+)/config\.json",
    ):
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return match.group(1).rstrip("/")
    return None


def _parse_dpo_launch_log_state(log_text: str) -> Dict[str, Any]:
    """Extract launcher stage from the outer launch log.

    The historical name is kept because tests and callers use it, but the
    parser covers DPO, batch LoRA/full, and multinode launcher markers.
    """
    text = str(log_text or "")
    lower = text.lower()
    export_dir = _extract_export_dir_from_text(text)
    training_started = (
        "training started." in lower
        or "start training" in lower
        or "***** running training *****" in lower
    )
    merge_started = (
        "merging lora weights" in lower
        or "merge started." in lower
    )
    merge_succeeded = (
        "merge weights operation completed successfully" in lower
        or "merge command executed successfully" in lower
    )
    merge_failed = (
        "merge weights operation failed" in lower
        or "merge command failed" in lower
        or "export failed" in lower
    )
    launcher_finished = (
        "train launcher successfully" in lower
        or "dpo multi-node train launcher completed successfully" in lower
        or "training finished. duration" in lower
    )

    sub_stage = None
    merge_status = None
    if launcher_finished and merge_succeeded and not merge_failed:
        sub_stage = "finished"
        merge_status = "finished"
    elif merge_failed:
        sub_stage = "merge"
        merge_status = "failed"
    elif merge_started:
        sub_stage = "merge"
        merge_status = "running"
    elif training_started:
        sub_stage = "training"

    return {
        "sub_stage": sub_stage,
        "sub_stage_text": _dpo_sub_stage_text(sub_stage),
        "export_dir": export_dir,
        "merge_status": merge_status,
        "training_started": training_started,
        "merge_started": merge_started,
        "merge_succeeded": merge_succeeded,
        "merge_failed": merge_failed,
        "launcher_finished": launcher_finished,
    }


def _parse_dpo_output_log_stage(lines: List[str]) -> Dict[str, Any]:
    """Split DPO output.log into training and post-training evaluation progress."""
    phase = None
    train_lines: List[str] = []
    eval_lines: List[str] = []
    train_started = False
    train_completed = False
    train_metrics_found = False
    eval_started = False
    eval_metrics_found = False

    for raw_line in lines or []:
        line = str(raw_line or "")
        lower = line.lower()
        if "***** running training *****" in lower:
            phase = "training"
            train_started = True
        elif "training completed" in lower:
            train_completed = True
        elif "***** train metrics *****" in lower:
            train_metrics_found = True
            train_completed = True
        elif "***** running evaluation *****" in lower:
            phase = "evaluation"
            eval_started = True
        elif "***** eval metrics *****" in lower:
            eval_metrics_found = True

        if phase == "evaluation":
            eval_lines.append(line)
        elif phase == "training":
            train_lines.append(line)

    train_progress = _parse_output_log_progress(train_lines)
    eval_progress = _parse_output_log_progress(eval_lines)
    sub_stage = None
    if eval_started:
        sub_stage = "evaluation"
    elif train_started or train_progress.get("progress_percent"):
        sub_stage = "training"

    return {
        "sub_stage": sub_stage,
        "sub_stage_text": _dpo_sub_stage_text(sub_stage),
        "train_completed": bool(train_completed or train_metrics_found),
        "train_metrics_found": train_metrics_found,
        "eval_started": eval_started,
        "eval_metrics_found": eval_metrics_found,
        "train_progress_percent": train_progress.get("progress_percent"),
        "train_current_step": train_progress.get("current_step"),
        "train_total_steps": train_progress.get("total_steps"),
        "train_elapsed_time": train_progress.get("elapsed_time"),
        "train_remaining_time": train_progress.get("remaining_time"),
        "eval_progress_percent": eval_progress.get("progress_percent"),
        "eval_current_step": eval_progress.get("current_step"),
        "eval_total_steps": eval_progress.get("total_steps"),
        "eval_elapsed_time": eval_progress.get("elapsed_time"),
        "eval_remaining_time": eval_progress.get("remaining_time"),
    }


def _has_dpo_training_context(lines: List[str]) -> bool:
    for line in lines or []:
        lower = str(line or "").lower()
        if (
            "dpo_train_launcher" in lower
            or "/dpo_train/" in lower
            or " dpo_train/" in lower
            or "'--stage', 'dpo'" in lower
            or '"--stage", "dpo"' in lower
            or "--stage dpo" in lower
        ):
            return True
    return False


def _is_deepspeed_completion_cleanup_line(lower: str) -> bool:
    text = lower or ""
    return bool(
        "sigkill_handler" in text
        or "killing subprocess" in text
        or (
            "exits with return code = -9" in text
            and ("launch.py" in text or "/usr/local/insinfersystem/train" in text)
        )
    )


def _is_ray_completion_cleanup_line(lower: str) -> bool:
    text = lower or ""
    return bool(
        "worker exits unexpectedly by a signal" in text
        or "worker died or was killed while executing a task" in text
        or "systemexit is raised" in text
        or "the process receives a sigterm" in text
        or "worker exit type: system_error" in text
    )


def _is_completion_cleanup_line(lower: str) -> bool:
    return _is_deepspeed_completion_cleanup_line(lower) or _is_ray_completion_cleanup_line(lower)


def _parse_training_completion_from_output_log(lines: List[str]) -> Dict[str, Any]:
    progress_info = _parse_output_log_progress(lines)
    dpo_stage_info = _parse_dpo_output_log_stage(lines)
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(text: str) -> str:
        return ansi_re.sub("", text or "").strip()

    completion_line = None
    completion_reason = None
    completion_index: Optional[int] = None
    latest_error_index: Optional[int] = None
    latest_error_line = None
    latest_runtime_error_line = None
    train_metrics_found = False
    train_loss_found = False
    train_runtime_found = False
    eval_metrics_found = False
    grpo_final_validation_found = False
    grpo_model_saved_found = False
    grpo_wandb_summary_found = False
    grpo_wandb_synced_found = False
    grpo_wandb_logs_found = False
    first_error_before_completion = False
    dpo_context_seen = _has_dpo_training_context(lines)

    completed_markers = [
        "training completed",
        "training finished",
        "finished training",
        "completed training",
        "training complete",
        "train complete",
        "train finished",
    ]
    err_keywords = [
        "traceback",
        "exception",
        "runtimeerror",
        "calledprocesserror",
        "cuda out of memory",
        "out of memory",
        "sigkill",
        "segmentation fault",
        "training failed",
        "merge weights operation failed",
        "merge command failed",
        "export failed",
        "returned non-zero",
        "exit status",
        "no checkpoint found",
    ]

    for idx, line in enumerate(lines or []):
        cleaned = _clean(line)
        lower = cleaned.lower()
        if not lower:
            continue

        is_shell_trace = lower.startswith("+") and (
            "python" in lower
            or "accelerate launch" in lower
            or "verl.trainer.main_ppo" in lower
            or "truncation=error" in lower
        )
        is_completion_cleanup_noise = (
            completion_index is not None
            and idx > completion_index
            and _is_completion_cleanup_line(lower)
        )
        if is_completion_cleanup_noise:
            continue
        if not is_shell_trace and any(k in lower for k in err_keywords):
            latest_error_index = idx
            latest_error_line = cleaned
            if "runtimeerror:" in lower:
                latest_runtime_error_line = cleaned
        if not is_shell_trace and "error" in lower:
            if "nccl_async_error_handling" not in lower and "torch_nccl_async_error_handling" not in lower:
                latest_error_index = idx
                latest_error_line = cleaned

        if "***** train metrics *****" in lower:
            train_metrics_found = True
            if completion_index is None:
                completion_index = idx
                completion_line = cleaned
                completion_reason = "train_metrics_marker"
        if "***** eval metrics *****" in lower:
            eval_metrics_found = True
        if re.search(r"(?:^|[^\w])train_loss\s*[:=]", lower):
            train_loss_found = True
        if re.search(r"(?:^|[^\w])train_runtime\s*[:=]", lower):
            train_runtime_found = True
        if any(marker in lower for marker in completed_markers):
            completion_index = idx
            completion_line = cleaned
            completion_reason = "training_completed_marker"
        if "final validation metrics" in lower:
            grpo_final_validation_found = True
            if completion_index is None:
                completion_index = idx
                completion_line = cleaned
                completion_reason = "grpo_final_validation_metrics"
        if "saved model" in lower and "/global_step_" in lower:
            grpo_model_saved_found = True
        if "wandb:" in lower and "run summary" in lower:
            grpo_wandb_summary_found = True
        if "wandb:" in lower and "synced" in lower:
            grpo_wandb_synced_found = True
        if "wandb:" in lower and "find logs at" in lower:
            grpo_wandb_logs_found = True

    completed = False
    if completion_reason == "training_completed_marker":
        completed = True
    elif train_metrics_found and (train_loss_found or train_runtime_found):
        completed = True
        completion_reason = completion_reason or "train_metrics_marker"
    elif train_loss_found and train_runtime_found:
        completed = True
        completion_reason = "train_metric_values"
    elif grpo_final_validation_found and (
        grpo_model_saved_found
        or grpo_wandb_summary_found
        or grpo_wandb_synced_found
        or grpo_wandb_logs_found
    ):
        completed = True
        completion_reason = "grpo_launch_log_completion"

    if completed and latest_error_index is not None and completion_index is not None:
        first_error_before_completion = latest_error_index < completion_index

    error_after_completion = (
        completed
        and latest_error_index is not None
        and completion_index is not None
        and latest_error_index > completion_index
    )
    if error_after_completion or first_error_before_completion:
        completed = False
        completion_reason = "error_before_completion" if first_error_before_completion else "error_after_completion"

    active_error_line = None
    if latest_error_index is not None and (
        first_error_before_completion or completion_index is None or latest_error_index > completion_index
    ):
        active_error_line = latest_runtime_error_line or latest_error_line

    # output.log can be appended across resumed/repeated runs.  A completion
    # marker from an older run must not complete a newer run whose unfinished
    # progress appears later in the same file.
    progress_after_completion = (
        completed
        and completion_index is not None
        and progress_info.get("progress_index") is not None
        and progress_info["progress_index"] > completion_index
    )
    if progress_after_completion:
        current_step = _to_float_or_none(progress_info.get("current_step"))
        total_steps = _to_float_or_none(progress_info.get("total_steps"))
        progress_percent = _to_float_or_none(
            str(progress_info.get("progress_percent") or "").rstrip("%")
        )
        explicitly_incomplete = (
            current_step is not None
            and total_steps is not None
            and total_steps > 0
            and current_step < total_steps
        ) or (progress_percent is not None and progress_percent < 100)
        if explicitly_incomplete:
            completed = False
            completion_reason = "new_progress_after_completion_marker"

    if dpo_context_seen and dpo_stage_info.get("train_progress_percent"):
        progress_info = {
            **progress_info,
            "progress_percent": dpo_stage_info.get("train_progress_percent"),
            "current_step": dpo_stage_info.get("train_current_step"),
            "total_steps": dpo_stage_info.get("train_total_steps"),
        }
    elif dpo_context_seen and not dpo_stage_info.get("eval_progress_percent"):
        progress_info = {
            **progress_info,
            "progress_percent": None,
            "current_step": None,
            "total_steps": None,
            "progress_line": None,
        }

    return {
        "completed": completed,
        "succeeded": completed,
        "completion_reason": completion_reason,
        "completion_line": completion_line,
        "completion_index": completion_index,
        "train_metrics_found": train_metrics_found,
        "train_loss_found": train_loss_found,
        "train_runtime_found": train_runtime_found,
        "eval_metrics_found": eval_metrics_found,
        "grpo_final_validation_found": grpo_final_validation_found,
        "grpo_model_saved_found": grpo_model_saved_found,
        "grpo_wandb_summary_found": grpo_wandb_summary_found,
        "grpo_wandb_synced_found": grpo_wandb_synced_found,
        "grpo_wandb_logs_found": grpo_wandb_logs_found,
        "error_after_completion": error_after_completion,
        "latest_error_line": latest_error_line,
        "active_error_line": active_error_line,
        "progress_percent": progress_info.get("progress_percent"),
        "current_step": progress_info.get("current_step"),
        "total_steps": progress_info.get("total_steps"),
        "progress_line": progress_info.get("progress_line"),
        **dpo_stage_info,
    }


def _confirm_realtime_tqdm_progress(
    log_key: str,
    parsed_progress: Dict[str, Any],
    log_mtime: Optional[float],
) -> Tuple[bool, str]:
    current_step = parsed_progress.get("current_step")
    total_steps = parsed_progress.get("total_steps")
    candidate_count = int(parsed_progress.get("progress_candidate_count") or 0)
    unique_step_count = int(parsed_progress.get("progress_unique_step_count") or 0)

    state = _PROGRESS_STATE_BY_LOG.get(log_key) or {}
    prev_step = state.get("last_step")
    prev_confirmed = bool(state.get("confirmed"))

    confirmed = False
    reason = "tqdm_missing"

    if current_step is None or total_steps is None:
        reason = "tqdm_parse_incomplete"
    elif current_step <= 0:
        reason = "tqdm_zero_only"
    elif unique_step_count >= 2:
        confirmed = True
        reason = "tqdm_multi_step_snapshot"
    elif isinstance(prev_step, int) and current_step > prev_step:
        confirmed = True
        reason = "tqdm_step_increased"
    elif prev_confirmed and isinstance(prev_step, int) and current_step >= prev_step:
        confirmed = True
        reason = "tqdm_already_confirmed_stream"
    elif candidate_count >= 2:
        confirmed = True
        reason = "tqdm_multi_line_snapshot"
    else:
        reason = "tqdm_wait_next_poll"

    _PROGRESS_STATE_BY_LOG[log_key] = {
        "last_step": current_step,
        "last_total_steps": total_steps,
        "last_mtime": log_mtime,
        "confirmed": bool(confirmed or prev_confirmed),
        "updated_at": datetime.now().isoformat(),
    }
    return confirmed, reason


def _read_output_log(container: str, path: str, history_limit: int) -> Dict[str, Any]:
    cmd = f"tail -n {int(history_limit)} {shlex.quote(path)}"
    code, out, _ = _docker_exec(container, ["sh", "-c", cmd], timeout=15)
    history: List[Dict[str, Any]] = []
    error_hint = None
    success_hint = None
    progress_percent = None
    current_step = None
    total_steps = None
    elapsed_time = None
    remaining_time = None
    progress_line = None
    progress_candidate_count = 0
    progress_unique_step_count = 0
    training_start_signal = False
    completion_info: Dict[str, Any] = {"completed": False, "succeeded": False}
    if code == 0 and out:
        lines = out.splitlines()
        training_start_signal = _lines_have_training_start_signal(lines)
        history = parse_output_log_history(lines)
        progress_info = _parse_output_log_progress(lines)
        completion_info = _parse_training_completion_from_output_log(lines)
        error_hint = completion_info.get("active_error_line")
        if _has_dpo_training_context(lines) and completion_info.get("train_progress_percent"):
            progress_info = {
                **progress_info,
                "progress_percent": completion_info.get("train_progress_percent"),
                "current_step": completion_info.get("train_current_step"),
                "total_steps": completion_info.get("train_total_steps"),
                "elapsed_time": completion_info.get("train_elapsed_time"),
                "remaining_time": completion_info.get("train_remaining_time"),
            }
        progress_percent = progress_info.get("progress_percent")
        current_step = progress_info.get("current_step")
        total_steps = progress_info.get("total_steps")
        elapsed_time = progress_info.get("elapsed_time")
        remaining_time = progress_info.get("remaining_time")
        progress_line = progress_info.get("progress_line")
        progress_candidate_count = int(progress_info.get("progress_candidate_count") or 0)
        progress_unique_step_count = int(progress_info.get("progress_unique_step_count") or 0)
        err_keywords = [
            "traceback",
            "error",
            "exception",
            "runtimeerror",
            "calledprocesserror",
            "cuda out of memory",
            "out of memory",
            "killed",
            "sigkill",
            "segmentation fault",
            "training failed",
            "merge weights operation failed",
            "merge command failed",
            "export failed",
            "returned non-zero",
            "exit status",
            "no checkpoint found",
        ]
        success_keywords = [
            "training completed",
            "training finished",
            "train finished",
            "finished training",
            "train complete",
            "completed training",
            "saving model",
            "model saved",
            "training complete",
            "训练完成",
            "训练结束",
            "训练已完成",
            "保存模型",
            "模型保存完成",
            "完成训练",
        ]
        completion_index = completion_info.get("completion_index")
        dpo_context_seen = _has_dpo_training_context(lines)
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            if isinstance(completion_index, int) and idx < completion_index:
                break
            lower = line.lower()
            if (
                isinstance(completion_index, int)
                and idx > completion_index
                and _is_completion_cleanup_line(lower)
            ):
                continue
            if lower.startswith("+") and (
                "python" in lower
                or "accelerate launch" in lower
                or "verl.trainer.main_ppo" in lower
                or "truncation=error" in lower
            ):
                continue
            if "nccl_async_error_handling" in lower or "torch_nccl_async_error_handling" in lower:
                continue

            if any(k in lower for k in err_keywords):
                error_hint = line.strip()
                success_hint = None
                break
            if not error_hint and any(k in lower for k in success_keywords):
                success_hint = line.strip()
                break
    count = None
    wc_code, wc_out, _ = _docker_exec(container, ["sh", "-c", f"wc -l {shlex.quote(path)}"])
    if wc_code == 0 and wc_out:
        try:
            count = int(wc_out.strip().split()[0])
        except Exception:
            count = None

    latest = history[-1] if history else None
    last_update_time = None
    mtime = _file_mtime(container, path)
    if mtime:
        last_update_time = datetime.fromtimestamp(mtime).isoformat()

    path_lower = (path or "").lower()
    require_realtime_gate = any(
        marker in path_lower
        for marker in (
            "grpo",
            "verl",
            "multinode",
            "train_multi",
            "batch_train",
            "-train-log",
        )
    )
    if require_realtime_gate:
        progress_realtime_confirmed, progress_realtime_reason = _confirm_realtime_tqdm_progress(
            log_key=f"{container}:{path}",
            parsed_progress={
                "current_step": current_step,
                "total_steps": total_steps,
                "progress_candidate_count": progress_candidate_count,
                "progress_unique_step_count": progress_unique_step_count,
            },
            log_mtime=mtime,
        )
        if not progress_realtime_confirmed:
            progress_percent = None
            current_step = None
            total_steps = None
            elapsed_time = None
            remaining_time = None
    else:
        progress_realtime_confirmed = bool(progress_percent is not None and current_step is not None)
        progress_realtime_reason = (
            "realtime_gate_not_required" if progress_realtime_confirmed else "tqdm_missing"
        )

    return {
        "history": history,
        "latest": latest,
        "count": len(history) if count is None else count,
        "last_update_time": last_update_time,
        "path": path,
        "error_hint": error_hint,
        "success_hint": success_hint,
        "success_found": bool(success_hint) or bool(completion_info.get("completed")),
        "succeeded": bool(completion_info.get("completed")),
        "completion_info": completion_info,
        "progress_percent": progress_percent,
        "current_step": current_step,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "remaining_time": remaining_time,
        "progress_line": progress_line,
        "progress_realtime_confirmed": progress_realtime_confirmed,
        "progress_realtime_reason": progress_realtime_reason,
        "progress_candidate_count": progress_candidate_count,
        "progress_unique_step_count": progress_unique_step_count,
        "training_start_signal": training_start_signal,
        "sub_stage": completion_info.get("sub_stage"),
        "sub_stage_text": completion_info.get("sub_stage_text"),
        "train_progress_percent": completion_info.get("train_progress_percent"),
        "train_current_step": completion_info.get("train_current_step"),
        "train_total_steps": completion_info.get("train_total_steps"),
        "train_elapsed_time": completion_info.get("train_elapsed_time"),
        "train_remaining_time": completion_info.get("train_remaining_time"),
        "eval_progress_percent": completion_info.get("eval_progress_percent"),
        "eval_current_step": completion_info.get("eval_current_step"),
        "eval_total_steps": completion_info.get("eval_total_steps"),
        "eval_elapsed_time": completion_info.get("eval_elapsed_time"),
        "eval_remaining_time": completion_info.get("eval_remaining_time"),
        "eval_started": completion_info.get("eval_started"),
        "eval_metrics_found": completion_info.get("eval_metrics_found"),
    }


def _log_data_has_any_metrics(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    return bool(
        data.get("latest")
        or data.get("history")
        or data.get("current_step") is not None
        or data.get("progress_percent") is not None
    )


def _log_data_has_chart_metrics(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    latest = data.get("latest")
    if isinstance(latest, dict) and latest.get("step") is not None and latest.get("loss") is not None:
        return True
    history = data.get("history")
    if isinstance(history, list):
        return any(
            isinstance(item, dict)
            and item.get("loss") is not None
            and (item.get("step") is not None or item.get("_step") is not None)
            for item in history
        )
    return False


def _select_effective_log_data(
    primary: Optional[Dict[str, Any]],
    fallback: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [item for item in (primary, fallback) if isinstance(item, dict) and item]
    if not candidates:
        return {}
    for item in candidates:
        if _log_data_has_chart_metrics(item):
            return item
    for item in candidates:
        if item.get("latest") or item.get("history"):
            return item
    for item in candidates:
        if _log_data_has_any_metrics(item):
            return item
    return candidates[0]

def _run_record_from_dir(run_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not run_dir:
        return None
    clean_dir = str(run_dir).rstrip("/")
    if not clean_dir:
        return None
    name = os.path.basename(clean_dir)
    ts = parse_wandb_run_timestamp(name)
    return {
        "name": name,
        "path": clean_dir,
        "mtime": ts.timestamp() if ts else 0.0,
    }


def _candidate_wandb_runs_for_ended_pid(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    wandb_root: str,
    wandb_runs: List[Dict[str, Any]],
    run_start_time: Optional[datetime],
    time_window_minutes: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(run: Optional[Dict[str, Any]], reason: str, score: float = 0.0) -> None:
        if not run or not run.get("path"):
            return
        key = str(run.get("path"))
        if key in seen:
            return
        seen.add(key)
        item = dict(run)
        item["select_reason"] = reason
        item["select_score"] = score
        candidates.append(item)

    roots = []
    for root in (wandb_root, DEFAULT_INSINFER_WANDB_ROOT, DEFAULT_VERL_WANDB_ROOT, DEFAULT_WANDB_ROOT):
        if root and root not in roots:
           roots.append(root) 

    wandb_url = _read_wandb_url_file(pid)
    run_id = _extract_run_id_from_wandb_url(wandb_url)
    if run_id:
        for root in roots:
            _add(
                _run_record_from_dir(_find_wandb_run_dir_by_id(container, root, run_id)),
                "pid_wandb_url",
                0.0,
            )

    training_log_path = None
    if pid_record:
        training_log_path = _extract_log_file(str(pid_record.get("command") or ""))
    if training_log_path:
        log_wandb_url = _read_wandb_url_from_log_file(container, training_log_path)
        log_run_id = _extract_run_id_from_wandb_url(log_wandb_url)
        if log_run_id:
            _write_wandb_url_file(pid, log_wandb_url)
            for root in roots:
                _add(
                    _run_record_from_dir(_find_wandb_run_dir_by_id(container, root, log_run_id)),
                    "training_log_wandb_url",
                    1.0,
                )

    selected_by_time, time_reason = select_wandb_run_by_start_time(
        wandb_runs,
        run_start_time,
        time_window_minutes,
        allow_early_seconds=120,
    )
    _add(selected_by_time, time_reason, 2.0)

    if run_start_time:
        earliest = run_start_time - timedelta(seconds=120)
        latest = run_start_time + timedelta(minutes=max(1, int(time_window_minutes)))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for run in wandb_runs:
            ts = parse_wandb_run_timestamp(run.get("name") or "")
            if not ts and isinstance(run.get("mtime"), (int, float)) and run.get("mtime"):
                ts = datetime.fromtimestamp(float(run.get("mtime")))
            if not ts or ts < earliest or ts > latest:
                continue
            diff = abs((ts - run_start_time).total_seconds())
            scored.append((diff, run))
        scored.sort(key=lambda item: item[0])
        for diff, run in scored[:12]:
            _add(run, "start_time_output_log_scan", 10.0 + diff)

    candidates.sort(key=lambda item: float(item.get("select_score") or 0.0))
    return candidates


def _build_finished_payload_from_output_log_candidate(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    run: Dict[str, Any],
    history_limit: int,
) -> Optional[Dict[str, Any]]:
    run_dir = run.get("path")
    if not run_dir:
        return None
    output_log_path = os.path.join(run_dir, "files", "output.log")
    output_log_data = _read_output_log(container, output_log_path, history_limit)
    count = output_log_data.get("count")
    output_log_payload = output_log_data
    if isinstance(count, int) and count > history_limit:
        output_log_payload = _read_output_log(container, output_log_path, count)

    completion_info = output_log_payload.get("completion_info") or {}
    if not completion_info.get("completed"):
        return None

    history = _normalize_history_loss(output_log_payload.get("history") or [])
    latest_metrics = output_log_payload.get("latest") or {}
    raw_latest_loss = latest_metrics.get("loss")
    latest_loss, _ = _normalize_loss_value(raw_latest_loss)
    latest_epoch = latest_metrics.get("epoch")
    latest_step = latest_metrics.get("step")
    progress_percent = output_log_payload.get("progress_percent")
    current_step = output_log_payload.get("current_step")
    total_steps = output_log_payload.get("total_steps")
    history, resume_history_info = _apply_resume_history_offset(history, pid_record)
    latest_step, resume_step_info = _apply_resume_step_offset(
        latest_step,
        current_step,
        pid_record,
        force_local_step=bool(resume_history_info),
    )
    resume_adjustment_info = resume_step_info or resume_history_info
    wandb_run_name = run.get("name") or os.path.basename(str(run_dir).rstrip("/"))
    wandb_mode = _infer_wandb_mode(pid_record, wandb_run_name, run_dir)
    last_update_time = output_log_payload.get("last_update_time")
    completion_reason = completion_info.get("completion_reason") or "output_log_completion_marker"
    completion_source = "wandb_output_log"

    _record_train_status(
        container,
        pid,
        "finished",
        None,
        latest_loss,
        latest_epoch,
        latest_step,
        wandb_run_name,
        run_dir,
        wandb_mode,
        output_log_payload.get("error_hint"),
        True,
        last_update_time,
        completion_reason,
        completion_source,
    )

    analysis_text = (
        "训练已完成，已根据 PID 定位到 W&B output.log，并解析到训练完成标记。"
    )
    if latest_loss is not None:
        analysis_text += f" 最后loss为{latest_loss}。"

    return {
        "status": "finished",
        "metrics": {
            "container_name": container,
            "pid": str(pid) if pid else None,
            "pid_alive": False,
            "training_process_exists": False,
            "latest_loss": latest_loss,
            "latest_epoch": latest_epoch,
            "latest_step": latest_step,
            "resume_checkpoint": (
                resume_adjustment_info.get("resume_checkpoint") if resume_adjustment_info else None
            ),
            "resume_checkpoint_step": (
                resume_adjustment_info.get("resume_checkpoint_step") if resume_adjustment_info else None
            ),
            "raw_latest_step": (
                resume_step_info.get("raw_latest_step") if resume_step_info else None
            ),
            "step_adjustment": (
                resume_step_info.get("step_adjustment") if resume_step_info else None
            ),
            "history_step_adjustment": (
                resume_history_info.get("history_step_adjustment") if resume_history_info else None
            ),
            "progress_percent": progress_percent,
            "current_step": current_step,
            "total_steps": total_steps,
            "latest_step_or_epoch": latest_epoch if latest_epoch is not None else latest_step,
            "history": history,
            "history_count": output_log_payload.get("count", len(history)) or len(history),
            "last_update_time": last_update_time,
            "wandb_run_name": wandb_run_name,
            "wandb_run_dir": run_dir,
            "wandb_mode": wandb_mode,
            "wandb_select_reason": run.get("select_reason"),
            "finished": True,
            "succeeded": True,
            "success_state": "succeeded",
            "completion_reason": completion_reason,
            "completion_source": completion_source,
            "completion_info": completion_info,
            "output_log_success": True,
            "output_log_success_hint": output_log_payload.get("success_hint"),
            "output_log_path": output_log_path,
            "loss_source": "output.log",
            "error_reason": None,
            "debug": {
                "ended_pid_output_log_fallback": True,
                "output_log_error": output_log_payload.get("error_hint"),
                "output_log_progress_line": output_log_payload.get("progress_line"),
                "candidate_reason": run.get("select_reason"),
            },
        },
        "analysis_text": analysis_text,
    }


def _build_finished_payload_from_grpo_launch_log(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    launch_log_tail: str,
) -> Optional[Dict[str, Any]]:
    if not launch_log_tail:
        return None
    lower = launch_log_tail.lower()
    if "grpo" not in lower and "verl" not in lower:
        return None

    lines = launch_log_tail.splitlines()
    completion_info = _parse_training_completion_from_output_log(lines)
    if not completion_info.get("completed") or not completion_info.get("grpo_final_validation_found"):
        return None

    history = _normalize_history_loss(parse_output_log_history(lines))
    latest_metrics = history[-1] if history else {}
    raw_latest_loss = latest_metrics.get("loss") if isinstance(latest_metrics, dict) else None
    latest_loss, _ = _normalize_loss_value(raw_latest_loss)
    latest_epoch = latest_metrics.get("epoch") if isinstance(latest_metrics, dict) else None
    latest_step = latest_metrics.get("step") if isinstance(latest_metrics, dict) else None
    progress_percent = completion_info.get("progress_percent")
    current_step = completion_info.get("current_step")
    total_steps = completion_info.get("total_steps")
    history, resume_history_info = _apply_resume_history_offset(history, pid_record)
    latest_step, resume_step_info = _apply_resume_step_offset(
        latest_step,
        current_step,
        pid_record,
        force_local_step=bool(resume_history_info),
    )
    resume_adjustment_info = resume_step_info or resume_history_info
    launch_log_path = (pid_record or {}).get("launch_log_file") if isinstance(pid_record, dict) else None
    completion_reason = "grpo_launch_log_completion"
    completion_source = "launch_log"
    cleanup_lines = [
        line.strip()
        for line in lines
        if _is_completion_cleanup_line(line.lower())
    ][-5:]

    _record_train_status(
        container,
        pid,
        "finished",
        None,
        latest_loss,
        latest_epoch,
        latest_step,
        None,
        None,
        _infer_wandb_mode(pid_record, None, None),
        None,
        True,
        None,
        completion_reason,
        completion_source,
    )

    analysis_text = "训练已完成，已解析到最终验证指标、模型保存或 W&B 同步信息。"
    if latest_loss is not None:
        analysis_text += f" 最后loss为{latest_loss}。"

    return {
        "status": "finished",
        "metrics": {
            "container_name": container,
            "pid": str(pid) if pid else None,
            "pid_alive": False,
            "training_process_exists": False,
            "train_type": "grpo",
            "train_type_public": "grpo",
            "train_type_text": "GRPO",
            "latest_loss": latest_loss,
            "latest_epoch": latest_epoch,
            "latest_step": latest_step,
            "resume_checkpoint": (
                resume_adjustment_info.get("resume_checkpoint") if resume_adjustment_info else None
            ),
            "resume_checkpoint_step": (
                resume_adjustment_info.get("resume_checkpoint_step") if resume_adjustment_info else None
            ),
            "raw_latest_step": (
                resume_step_info.get("raw_latest_step") if resume_step_info else None
            ),
            "step_adjustment": (
                resume_step_info.get("step_adjustment") if resume_step_info else None
            ),
            "history_step_adjustment": (
                resume_history_info.get("history_step_adjustment") if resume_history_info else None
            ),
            "progress_percent": progress_percent,
            "current_step": current_step,
            "total_steps": total_steps,
            "latest_step_or_epoch": latest_epoch if latest_epoch is not None else latest_step,
            "history": history,
            "history_count": len(history),
            "launch_log_path": launch_log_path,
            "metrics_log_path": launch_log_path,
            "metrics_log_source": "launch_log",
            "finished": True,
            "succeeded": True,
            "success_state": "succeeded",
            "completion_reason": completion_reason,
            "completion_source": completion_source,
            "completion_info": completion_info,
            "output_log_success": True,
            "output_log_success_hint": completion_info.get("completion_line"),
            "loss_source": "launch_log",
            "error_reason": None,
            "debug": {
                "ended_pid_grpo_launch_log_fallback": True,
                "launch_log_cleanup_lines": cleanup_lines,
            },
        },
        "analysis_text": analysis_text,
    }


def _launch_log_has_dpo_success(log_text: str) -> bool:
    lower = str(log_text or "").lower()
    state = _parse_dpo_launch_log_state(log_text)
    return bool(
        state.get("merge_status") == "finished"
        and state.get("launcher_finished")
        and (
            "train launcher successfully" in lower
            or "dpo multi-node train launcher completed successfully" in lower
            or "training finished. duration" in lower
        )
    )


def _launch_log_has_error_after_dpo_success(log_text: str) -> bool:
    lines = str(log_text or "").splitlines()
    success_index = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if (
            "train launcher successfully" in lower
            or "dpo multi-node train launcher completed successfully" in lower
            or "training finished. duration" in lower
        ):
            success_index = idx
    if success_index is None:
        return False
    error_keywords = (
        "traceback",
        "runtimeerror",
        "exception",
        "merge weights operation failed",
        "merge command failed",
        "export failed",
        "returned non-zero",
        "exit status",
        "no checkpoint found",
    )
    for line in lines[success_index + 1:]:
        lower = line.lower()
        if any(keyword in lower for keyword in error_keywords):
            return True
    return False


def _build_finished_payload_from_dpo_launch_log(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    launch_log_tail: str,
) -> Optional[Dict[str, Any]]:
    dpo_launch_state = _parse_dpo_launch_log_state(launch_log_tail)
    if not _launch_log_has_dpo_success(launch_log_tail):
        return None
    if _launch_log_has_error_after_dpo_success(launch_log_tail):
        return None
    output_dir = None
    if isinstance(pid_record, dict):
        output_dir = str(pid_record.get("output_dir") or pid_record.get("outputDir") or "").strip() or None
    if not output_dir:
        match = re.search(r"/home/workspace/models/dpo_train/internal/saves/(\d{14})", launch_log_tail)
        if match:
            output_dir = f"/home/workspace/models/dpo_train/internal/saves/{match.group(1)}"

    export_dir = dpo_launch_state.get("export_dir")
    if not export_dir and output_dir:
        save_name = os.path.basename(output_dir.rstrip("/"))
        if save_name:
            export_candidates = [
                f"/home/workspace/models/dpo_train/internal/export/model_dpo_{save_name}",
                f"/home/workspace/models/dpo_train/internal/export/model_medical_{save_name}",
            ]
            export_dir = export_candidates[0]
            for candidate in export_candidates:
                if _docker_exec(container, ["test", "-d", candidate], timeout=5)[0] == 0:
                    export_dir = candidate
                    break

    record_train_type = normalize_train_type(
        (pid_record or {}).get("train_type") or (pid_record or {}).get("trainType")
        if isinstance(pid_record, dict)
        else None
    )
    record_launch_mode = (
        (pid_record or {}).get("launch_mode") or (pid_record or {}).get("launchMode")
        if isinstance(pid_record, dict)
        else None
    )
    record_script = (
        (pid_record or {}).get("script_name") or (pid_record or {}).get("scriptName")
        if isinstance(pid_record, dict)
        else None
    )
    train_type_value = record_train_type or "unknown"
    train_type_text = public_train_type_text(record_train_type, record_launch_mode, record_script)
    display_name = train_type_text or "训练"
    analysis_text = f"{display_name}已完成，已从启动日志确认 merge/export 成功。"
    if export_dir:
        analysis_text += f"\n导出模型：{export_dir}"

    return {
        "status": "finished",
        "metrics": {
            "container_name": container,
            "pid": str(pid) if pid else None,
            "pid_alive": False,
            "training_process_exists": False,
            "train_type": train_type_value,
            "train_type_public": public_train_type(record_train_type, record_launch_mode, record_script),
            "train_type_text": train_type_text,
            "output_dir": output_dir,
            "export_dir": export_dir,
            "sub_stage": "finished",
            "sub_stage_text": _dpo_sub_stage_text("finished"),
            "merge_status": dpo_launch_state.get("merge_status") or "finished",
            "finished": True,
            "succeeded": True,
            "success_state": "succeeded",
            "completion_reason": "dpo_launcher_merge_success",
            "completion_source": "launch_log",
            "output_log_success": True,
            "output_log_success_hint": "launcher merge/export successfully.",
            "loss_source": "launch_log",
            "error_reason": None,
            "launch_log_path": (pid_record or {}).get("launch_log_file") if isinstance(pid_record, dict) else None,
        },
        "analysis_text": analysis_text,
    }

def _try_build_finished_payload_from_ended_pid_output_log(
    container: str,
    pid: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    wandb_root: str,
    wandb_runs: List[Dict[str, Any]],
    run_start_time: Optional[datetime],
    time_window_minutes: int,
    history_limit: int,
) -> Optional[Dict[str, Any]]:
    candidates = _candidate_wandb_runs_for_ended_pid(
        container,
        pid,
        pid_record,
        wandb_root,
        wandb_runs,
        run_start_time,
        time_window_minutes,
    )
    for run in candidates:
        payload = _build_finished_payload_from_output_log_candidate(
            container,
            pid,
            pid_record,
            run,
            history_limit,
        )
        if payload:
            return payload
    return None

def _read_wandb_run(
    container: str,
    run_dir: str,
    history_limit: int,
) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
    metadata_path = os.path.join(run_dir, "files", "wandb-metadata.json")
    history_path = os.path.join(run_dir, "files", "wandb-history.jsonl")

    summary = _read_wandb_json_file(container, summary_path) if run_dir else None
    metadata = _read_wandb_json_file(container, metadata_path) if run_dir else None

    history_records: List[Dict[str, Any]] = []
    raw_history_count = None
    if run_dir:
        history_records, raw_history_count = _read_wandb_history_tail(container, history_path, history_limit)

    last_update_time = None
    if run_dir:
        history_mtime = _file_mtime(container, history_path)
        if history_mtime:
            last_update_time = datetime.fromtimestamp(history_mtime).isoformat()

    extracted = extract_wandb_history(
        history_records,
        history_limit=history_limit,
        summary=summary,
        last_update_time=last_update_time,
    )

    exitcode = None
    state = None
    if metadata:
        exitcode = metadata.get("exitcode") or metadata.get("exit_code")
        state = metadata.get("state") or metadata.get("status")
    if summary and exitcode is None:
        exitcode = summary.get("exitcode") or summary.get("exit_code")
    if summary and state is None:
        state = summary.get("state") or summary.get("status")

    finished = exitcode == 0 or (isinstance(state, str) and state.lower() in {"finished", "success", "completed"})
    failed = (
        (exitcode is not None and exitcode != 0)
        or (isinstance(state, str) and state.lower() in {"failed", "crashed", "error"})
    )

    valid_history = extracted.get("history") or []
    history_truncated = (
        raw_history_count is not None and raw_history_count > len(history_records)
    )

    return {
        "summary": summary,
        "metadata": metadata,
        "history": valid_history,
        "history_raw": history_records,
        "latest": extracted.get("latest"),
        "history_count": len(valid_history),
        "raw_history_count": raw_history_count,
        "history_truncated": history_truncated,
        "last_update_time": extracted.get("last_update_time"),
        "loss_key": extracted.get("loss_key"),
        "step_key": extracted.get("step_key"),
        "epoch_key": extracted.get("epoch_key"),
        "exitcode": exitcode,
        "state": state,
        "finished": finished,
        "failed": failed,
        "history_path": history_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
    }


def _persist_wandb_snapshot(
    container: Optional[str],
    run_name: Optional[str],
    run_dir: Optional[str],
    wandb_data: Dict[str, Any],
    output_log_data: Optional[Dict[str, Any]] = None,
    output_log_full_data: Optional[Dict[str, Any]] = None,
    metrics_history: Optional[List[Dict[str, Any]]] = None,
    metrics_source: Optional[str] = None,
) -> Optional[str]:
    if not container or not run_name:
        return None
    os.makedirs(_WANDB_SNAPSHOT_DIR, exist_ok=True)
    snapshot_dir = os.path.join(_WANDB_SNAPSHOT_DIR, container, run_name)
    os.makedirs(snapshot_dir, exist_ok=True)

    def _write_json(path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = wandb_data.get("summary")
    metadata = wandb_data.get("metadata")
    history = wandb_data.get("history")
    history_raw = wandb_data.get("history_raw")
    output_payload = output_log_full_data or output_log_data
    output_history = output_payload.get("history") if isinstance(output_payload, dict) else None
    output_meta = {}
    if isinstance(output_payload, dict):
        output_meta = {
            "path": output_payload.get("path"),
            "count": output_payload.get("count"),
            "last_update_time": output_payload.get("last_update_time"),
            "latest": output_payload.get("latest"),
            "error_hint": output_payload.get("error_hint"),
            "success_hint": output_payload.get("success_hint"),
            "success_found": output_payload.get("success_found"),
            "progress_percent": output_payload.get("progress_percent"),
            "current_step": output_payload.get("current_step"),
            "total_steps": output_payload.get("total_steps"),
            "elapsed_time": output_payload.get("elapsed_time"),
            "remaining_time": output_payload.get("remaining_time"),
            "source": "full" if output_log_full_data else "tail",
        }
    effective_metrics_history = metrics_history or output_history or history or []
    effective_metrics_source = metrics_source or ("output.log" if output_history else "wandb")
    wandb_history_source = "wandb"
    if (not history) and output_history:
        history = output_history
        history_raw = output_history
        wandb_history_source = "output.log"
    if summary is not None:
        _write_json(os.path.join(snapshot_dir, "wandb-summary.json"), summary)
    if metadata is not None:
        _write_json(os.path.join(snapshot_dir, "wandb-metadata.json"), metadata)
    if history is not None:
        history_path = os.path.join(snapshot_dir, "wandb-history.jsonl")
        with open(history_path, "w", encoding="utf-8") as f:
            for item in history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if history_raw is not None:
        raw_path = os.path.join(snapshot_dir, "wandb-history-raw.jsonl")
        with open(raw_path, "w", encoding="utf-8") as f:
            for item in history_raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if output_history is not None:
        out_hist_path = os.path.join(snapshot_dir, "output-log-history.jsonl")
        with open(out_hist_path, "w", encoding="utf-8") as f:
            for item in output_history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if output_meta:
        _write_json(os.path.join(snapshot_dir, "output-log-meta.json"), output_meta)
    if effective_metrics_history:
        metrics_path = os.path.join(snapshot_dir, "metrics-history.jsonl")
        with open(metrics_path, "w", encoding="utf-8") as f:
            for item in effective_metrics_history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_json(
            os.path.join(snapshot_dir, "metrics-latest.json"),
            effective_metrics_history[-1],
        )

    snapshot = {
        "container": container,
        "run_name": run_name,
        "run_dir": run_dir,
        "captured_at": datetime.now().isoformat(),
        "history_count": wandb_data.get("history_count"),
        "raw_history_count": wandb_data.get("raw_history_count"),
        "history_truncated": wandb_data.get("history_truncated"),
        "exitcode": wandb_data.get("exitcode"),
        "state": wandb_data.get("state"),
        "loss_key": wandb_data.get("loss_key"),
        "step_key": wandb_data.get("step_key"),
        "epoch_key": wandb_data.get("epoch_key"),
        "latest": wandb_data.get("latest"),
        "last_update_time": wandb_data.get("last_update_time"),
        "metrics_source": effective_metrics_source,
        "metrics_count": len(effective_metrics_history) if effective_metrics_history else 0,
        "wandb_history_source": wandb_history_source,
        "output_log_success_hint": output_meta.get("success_hint") if output_meta else None,
        "output_log_error_hint": output_meta.get("error_hint") if output_meta else None,
    }
    _write_json(os.path.join(snapshot_dir, "snapshot.json"), snapshot)
    return snapshot_dir


# CHANGE_REASON(2026-02-06): loss 误报修复（支持字符串/可解析 float）
def _normalize_loss_value(value: Any) -> Tuple[Optional[float], str]:
    if value is None:
        return None, "loss_none"
    if isinstance(value, (int, float)):
        return float(value), "loss_numeric"
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("nan", "+nan", "-nan"):
            return float("nan"), "loss_string_nan"
        if s in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf"), "loss_string_inf"
        if s in ("-inf", "-infinity"):
            return float("-inf"), "loss_string_ninf"
        try:
            return float(s), "loss_string_float"
        except Exception:
            return None, "loss_string_unparseable"
    try:
        return float(value), f"loss_coerced_{type(value).__name__}"
    except Exception:
        return None, f"loss_unparseable_{type(value).__name__}"


def _normalize_history_loss(history: Any) -> Any:
    if not isinstance(history, list):
        return history
    new_hist = []
    for item in history:
        if isinstance(item, dict) and "loss" in item:
            v, _ = _normalize_loss_value(item.get("loss"))
            ni = dict(item)
            ni["loss"] = v
            new_hist.append(ni)
        else:
            new_hist.append(item)
    return new_hist


def _find_last_finite_loss(history: Any, lookback: int = 20) -> Tuple[Optional[float], Optional[Any]]:
    if not isinstance(history, list) or not history:
        return None, None
    cnt = 0
    for item in reversed(history):
        if cnt >= lookback:
            break
        cnt += 1
        if not isinstance(item, dict):
            continue
        v = item.get("loss")
        fv, _ = _normalize_loss_value(v)
        if isinstance(fv, (int, float)) and math.isfinite(fv):
            axis = item.get("step") if item.get("step") is not None else item.get("epoch")
            return float(fv), axis
    return None, None


def _latest_events_file(container: str, output_dir: str) -> Tuple[Optional[str], Optional[float]]:
    script = r"""
import json, os, sys
root = sys.argv[1]
events = []
for dirpath, _, filenames in os.walk(os.path.join(root, "runs")):
    for name in filenames:
        if name.startswith("events.out.tfevents"):
            path = os.path.join(dirpath, name)
            events.append((os.path.getmtime(path), path))
if not events:
    print(json.dumps({"path": "", "mtime": None}))
    raise SystemExit(0)
events.sort(key=lambda x: x[0], reverse=True)
print(json.dumps({"path": events[0][1], "mtime": events[0][0]}))
"""
    code, out, _ = _docker_exec_python(container, script, [output_dir])
    if code != 0 or not out:
        return None, None
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return None, None
    path = payload.get("path") or None
    mtime = payload.get("mtime")
    return path, mtime


def _latest_tensorboard_dir(container: str, output_dir: str) -> Optional[str]:
    script = r"""
import os, sys
root = os.path.join(sys.argv[1], "runs")
if not os.path.isdir(root):
    print("")
    raise SystemExit(0)
subdirs = []
for d in os.listdir(root):
    path = os.path.join(root, d)
    if os.path.isdir(path):
        subdirs.append((os.path.getmtime(path), path))
if not subdirs:
    print("")
    raise SystemExit(0)
subdirs.sort(key=lambda x: x[0], reverse=True)
print(subdirs[0][1])
"""
    code, out, _ = _docker_exec_python(container, script, [output_dir])
    if code == 0 and out:
        return out.strip()
    return None


def _read_trainer_log(container: str, log_path: str, history_limit: int) -> Dict[str, Any]:
    script = r"""
import json, os, sys, datetime
path = sys.argv[1]
limit = int(sys.argv[2])
history = []

def to_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return None

def to_int_if_possible(v):
    num = to_number(v)
    if num is None:
        return None
    if abs(num - int(num)) < 1e-9:
        return int(num)
    return num

def pick_loss(d):
    for k in ("loss", "train_loss", "training_loss"):
        if k in d:
            return to_number(d[k])
    return None

def pick_lr(d):
    for k in ("lr", "learning_rate", "train/lr", "train_learning_rate"):
        if k in d:
            return to_number(d[k])
    return None

def pick_step(d):
    for k in ("current_steps", "step", "global_step", "iteration"):
        if k in d:
            return to_int_if_possible(d[k])
    return None

def pick_epoch(d):
    for k in ("epoch", "train_epoch"):
        if k in d:
            return to_number(d[k])
    return None

def pick_percent(d):
    for k in ("percent", "percentage", "progress"):
        if k in d:
            return to_number(d[k])
    return None

def pick_completed(d):
    for k in ("completed_steps", "completed_step"):
        if k in d:
            return to_int_if_possible(d[k])
    return None

def pick_max(d):
    for k in ("max_steps", "total_steps"):
        if k in d:
            return to_int_if_possible(d[k])
    return None

def pick_elapsed(d):
    for k in ("elapsed_time", "elapsed"):
        if k in d and d[k] is not None:
            return str(d[k])
    return None

def pick_remaining(d):
    for k in ("remaining_time", "remaining"):
        if k in d and d[k] is not None:
            return str(d[k])
    return None

count = 0
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        count += 1
        item = {
            "step": pick_step(data),
            "epoch": pick_epoch(data),
            "loss": pick_loss(data),
            "learning_rate": pick_lr(data),
            "percent": pick_percent(data),
            "completed_steps": pick_completed(data),
            "max_steps": pick_max(data),
            "elapsed_time": pick_elapsed(data),
            "remaining_time": pick_remaining(data),
        }
        history.append(item)

if limit > 0 and len(history) > limit:
    history = history[-limit:]
latest = None
for item in reversed(history):
    if item.get("loss") is not None:
        latest = item
        break
if latest is None and history:
    latest = history[-1]
mtime = os.path.getmtime(path)
finished = False
progress_percent = None
current_step = None
total_steps = None
elapsed_time = None
remaining_time = None
progress_item = history[-1] if history else latest
if progress_item:
    percent = progress_item.get("percent")
    if isinstance(percent, (int, float)) and percent >= 100:
        finished = True
    if isinstance(percent, (int, float)):
        progress_percent = f"{percent:g}%"
    current_step = progress_item.get("step")
    completed = progress_item.get("completed_steps")
    max_steps = progress_item.get("max_steps")
    total_steps = max_steps
    elapsed_time = progress_item.get("elapsed_time")
    remaining_time = progress_item.get("remaining_time")
    if isinstance(completed, (int, float)) and isinstance(max_steps, (int, float)) and max_steps > 0 and completed >= max_steps:
        finished = True

print(json.dumps({
    "history": history,
    "latest": latest,
    "count": count,
    "last_update_time": datetime.datetime.fromtimestamp(mtime).isoformat(),
    "finished": finished,
    "path": path,
    "progress_percent": progress_percent,
    "current_step": current_step,
    "total_steps": total_steps,
    "elapsed_time": elapsed_time,
    "remaining_time": remaining_time,
    "progress_realtime_confirmed": current_step is not None,
    "progress_realtime_reason": "trainer_log",
    "training_start_signal": bool(history),
    "completion_info": {"completed": finished, "succeeded": finished}
}))
"""
    code, out, _ = _docker_exec_python(container, script, [log_path, str(history_limit)])
    if code != 0 or not out:
        return {"error": "trainer_log_read_failed"}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"error": "trainer_log_invalid_json"}


def _read_tensorboard_events(container: str, events_path: str, history_limit: int) -> Dict[str, Any]:
    script = r"""
import json, os, sys, datetime
path = sys.argv[1]
limit = int(sys.argv[2])
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print(json.dumps({"error": "tensorboard_not_available", "message": str(e)}))
    raise SystemExit(0)

ea = EventAccumulator(path)
ea.Reload()
tags = ea.Tags().get("scalars", [])

def pick_tag(candidates):
    for t in candidates:
        if t in tags:
            return t
    return None

loss_tag = pick_tag(["train/loss", "loss", "train_loss", "training_loss", "loss/train"])
epoch_tag = pick_tag(["train/epoch", "epoch", "epochs"])
if not loss_tag:
    print(json.dumps({"error": "loss_tag_not_found", "available_tags": tags}))
    raise SystemExit(0)

loss_events = ea.Scalars(loss_tag)
history = [{"step": e.step, "loss": e.value} for e in loss_events]
if epoch_tag:
    epoch_events = ea.Scalars(epoch_tag)
    epoch_map = {e.step: e.value for e in epoch_events}
    for item in history:
        item["epoch"] = epoch_map.get(item["step"])

if limit > 0 and len(history) > limit:
    history = history[-limit:]
latest = history[-1] if history else None
mtime = os.path.getmtime(path)

print(json.dumps({
    "history": history,
    "latest": latest,
    "count": len(loss_events),
    "last_update_time": datetime.datetime.fromtimestamp(mtime).isoformat(),
    "loss_tag": loss_tag,
    "epoch_tag": epoch_tag
}))
"""
    code, out, _ = _docker_exec_python(container, script, [events_path, str(history_limit)])
    if code != 0 or not out:
        return {"error": "events_read_failed"}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"error": "events_invalid_json"}


# CHANGE_REASON(2026-02-11):
# 兼容 OpenAIChatModel 可能返回的多种形态，避免误判 EmptyResponse
def _extract_text_from_response(response: Any) -> str:
    try:
        if response is None:
            return ""

        # 1) 直接就是字符串
        if isinstance(response, str):
            return response.strip()

        # 2) 常见 OpenAI-style dict
        if isinstance(response, dict):
            # {"choices":[{"message":{"content":"..."}}]}
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            return content.strip()
                    # 兼容 {"text": "..."} 这种
                    text = c0.get("text")
                    if isinstance(text, str):
                        return text.strip()
            # 兼容 {"content": "..."} 或 {"output_text": "..."}
            for k in ("content", "output_text", "text"):
                v = response.get(k)
                if isinstance(v, str):
                    return v.strip()
            # 兼容 {"content":[{"type":"text","text":"..."}]}
            v = response.get("content")
            if isinstance(v, list):
                texts = []
                for item in v:
                    if isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str):
                            texts.append(t)
                return "".join(texts).strip()

        # 3) 有 choices 属性（对象）
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            # message.content
            msg = getattr(c0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content.strip()
            # text
            text = getattr(c0, "text", None)
            if isinstance(text, str):
                return text.strip()

        # 4) content 属性：可能是 str / list[block] / list[dict]
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for block in content:
                if hasattr(block, "text"):
                    t = getattr(block, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                        continue
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        texts.append(t)
            return "".join(texts).strip()

        return ""
    except Exception:
        return ""


def _run_async_with_error(coro) -> Tuple[Optional[Any], Optional[str]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(coro), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    result: Dict[str, Any] = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as exc:
            result["error"] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        err = result["error"]
        if isinstance(err, BaseException):
            return None, f"{type(err).__name__}: {err}"
        return None, str(err)
    return result.get("value"), None


def _ensure_llm_loop(timeout: float = 5.0) -> Optional[asyncio.AbstractEventLoop]:
    global _LLM_LOOP, _LLM_LOOP_THREAD
    with _LLM_LOOP_LOCK:
        if _LLM_LOOP and _LLM_LOOP.is_running():
            return _LLM_LOOP

        _LLM_LOOP_READY.clear()

        def _loop_runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # 关键：先把 loop 放进全局，再立刻 set ready（不要依赖 call_soon）
            with _LLM_LOOP_LOCK:
                globals()["_LLM_LOOP"] = loop  # 等价于 global _LLM_LOOP = loop，但更防嵌套误判
            _LLM_LOOP_READY.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        _LLM_LOOP_THREAD = threading.Thread(target=_loop_runner, daemon=True)
        _LLM_LOOP_THREAD.start()

    if _LLM_LOOP_READY.wait(timeout=timeout):
        with _LLM_LOOP_LOCK:
            if _LLM_LOOP and _LLM_LOOP.is_running():
                return _LLM_LOOP
            # 有些情况下 set 了但 loop 还没进入 run_forever，这里兜底也返回
            if _LLM_LOOP:
                return _LLM_LOOP
    return None


# CHANGE_REASON(2026-02-11):
# 1) 复用 OpenAIChatModel（常驻 client），避免每次 aclose() 引起 event loop is closed
# 2) 更强的响应解析器，避免误判 EmptyResponse
# 3) 当空响应时，把 response 类型/预览放到错误信息里，便于定位
def _call_monitor_llm(model_name: str, base_url: str, analysis_input: Dict[str, Any], api_key: str = DEFAULT_MODEL_API_KEY) -> Tuple[Optional[str], Optional[str]]:
    global _LLM_MODEL, _LLM_MODEL_KEY

    prompt = json.dumps(analysis_input, ensure_ascii=False, sort_keys=True)
    system_prompt = (
        "你是训练监控助手。根据给定的结构化指标，输出中文评价，包含："
        "1)训练是否正常进行；2)loss总体趋势（允许波动但要判断趋势）；"
        "3)若异常（NaN、爆炸、长时间无更新、文件缺失）给出原因猜测与建议；"
        "若仅有step没有epoch，请明确说明用step替代。不要复述原始数据，200字内。"
    )

    # 关键修复：把模型缓存/创建放在同步函数里，并显式 global，避免 UnboundLocalError
    def _get_or_create_model_sync() -> OpenAIChatModel:
        global _LLM_MODEL, _LLM_MODEL_KEY
        key = (base_url, model_name, api_key)
        if _LLM_MODEL is not None and _LLM_MODEL_KEY == key:
            return _LLM_MODEL

        _LLM_MODEL = OpenAIChatModel(
            credential=OpenAICredential(
                api_key=api_key,
                base_url=base_url,
            ),
            model=model_name,
            parameters=OpenAIChatModel.Parameters(
                temperature=0,
                top_p=1,
                max_tokens=256,
            ),
            stream=False,
            client_kwargs={"base_url": base_url} if base_url else None,
        )
        _LLM_MODEL_KEY = key
        return _LLM_MODEL

    async def _invoke():
        model = _get_or_create_model_sync()
        return await model(
            [
                Msg(
                    name="system",
                    content=[TextBlock(text=system_prompt)],
                    role="system",
                ),
                Msg(
                    name="user",
                    content=[TextBlock(text=prompt)],
                    role="user",
                ),
            ],
        )

    # 先尝试使用常驻 loop（不成功也不强行包装成 loop timeout，直接走 fallback 并返回真实错误）
    loop = _ensure_llm_loop(timeout=3.0)

    def _empty_err_with_preview(response: Any) -> str:
        preview = ""
        try:
            preview = (repr(response)[:500] + "...") if response is not None else ""
        except Exception:
            preview = ""
        return f"EmptyResponse: model returned empty content; response_type={type(response).__name__}; preview={preview}"

    # 1) 有 loop 就用 run_coroutine_threadsafe
    if loop is not None and loop.is_running():
        last_err: Optional[str] = None
        for _ in range(2):
            future = None
            try:
                future = asyncio.run_coroutine_threadsafe(_invoke(), loop)
                response = future.result(timeout=30)
                text = _extract_text_from_response(response)
                if text:
                    return text, None
                last_err = _empty_err_with_preview(response)
            except concurrent.futures.TimeoutError:
                try:
                    if future:
                        future.cancel()
                except Exception:
                    pass
                last_err = "TimeoutError: model response timed out"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2)
        return None, last_err or "UnknownError: llm_call_failed"

    # 2) fallback：当前线程直接跑协程（并返回真实 err，不要伪装成 loop timeout）
    response, err = _run_async_with_error(_invoke())
    if err:
        return None, err
    text = _extract_text_from_response(response)
    if text:
        return text, None
    return None, _empty_err_with_preview(response)


def monitor_training(
    container_name: Optional[str] = None,
    train_type: Optional[str] = None,
    session_id: Optional[str] = None,
    pid: Optional[str] = None,
    launch_mode: Optional[str] = None,
    is_multinode: Optional[bool] = None,
    script_name: Optional[str] = None,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    time_window_minutes: int = DEFAULT_TIME_WINDOW_MINUTES,
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_MODEL_API_KEY,
    wandb_root: Optional[str] = None,
    allow_llm: bool = True,
) -> ToolResponse:
    try:
        _ = session_id  # session_id 当前未参与定位（保留参数兼容）
        user_supplied_wandb_root = bool(wandb_root)
        wandb_root = (wandb_root or DEFAULT_WANDB_ROOT).strip()
        wandb_root_source = "user_input" if user_supplied_wandb_root else "default"
        request_launch_mode = str(launch_mode or "").strip() or ("multinode" if is_multinode else None)
        request_script_name = str(script_name or "").strip() or None

        container_name, container_err = _get_container_name(container_name)
        if container_err:
            if container_err == "multiple_containers":
                analysis_text = "检测到多个运行容器，请提供 container_name（如“容器名称是 xxx”）。"
            elif container_err == "no_running_container":
                analysis_text = "未检测到运行中的容器，请先启动训练容器。"
            else:
                analysis_text = "无法确定训练容器，请提供 container_name。"
            payload = {
                "status": "failed",
                "metrics": {"error_reason": container_err},
                "analysis_text": analysis_text,
            }
            return _monitor_response(payload)

        code, out, err = _run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
        if code != 0 or out.strip().lower() != "true":
            payload = {
                "status": "failed",
                "metrics": {"container_name": container_name, "error_reason": err or out or "container_not_running"},
                "analysis_text": "训练容器未运行，请先启动容器。",
            }
            return _monitor_response(payload)

        ps_code, ps_out, _ = _docker_exec(container_name, ["ps", "aux"])
        processes = _parse_ps_aux(ps_out) if ps_code == 0 else []
        ps_start_code, ps_start_out, _ = _docker_exec(container_name, ["ps", "-eo", "pid,lstart,cmd"])
        start_times = _parse_ps_start_times(ps_start_out) if ps_start_code == 0 else {}

        pid_source = None
        pid_record = None
        if pid:
            pid = str(pid)
            if not pid.isdigit():
                payload = {
                    "status": "failed",
                    "metrics": {"container_name": container_name, "error_reason": "pid_invalid"},
                    "analysis_text": "PID 格式不正确，请重新输入正确的进程号。",
                }
                return _monitor_response(payload)
            pid_source = "input"
            pid_record = _get_pid_record_by_pid(container_name, pid)
        else:
            pid_record = _get_latest_pid_record(container_name, time_window_minutes)
            if pid_record and pid_record.get("pid"):
                pid = str(pid_record.get("pid"))
                pid_source = "registry"
        pid_alive = _check_pid_alive(container_name, pid) if pid else False

        if pid_alive and isinstance(pid_record, dict) and pid_record.get("launch_status") == "preparing":
            ready_marker = str(pid_record.get("startup_ready_marker") or "").strip()
            launch_log_tail = _read_launch_log_tail(container_name, pid_record)
            if ready_marker and ready_marker not in launch_log_tail:
                elapsed_seconds = None
                try:
                    started_at = datetime.fromisoformat(str(pid_record.get("started_at") or ""))
                    elapsed_seconds = max(0, int((datetime.now() - started_at).total_seconds()))
                except Exception:
                    pass
                progress_lines = [line.strip() for line in launch_log_tail.splitlines() if line.strip()]
                progress_detail = progress_lines[-1][-300:] if progress_lines else None
                elapsed_text = (
                    f"已等待约 {elapsed_seconds // 60} 分 {elapsed_seconds % 60} 秒，"
                    if elapsed_seconds is not None
                    else ""
                )
                analysis_text = (
                    f"训练任务仍在准备数据集和模型，{elapsed_text}"
                    "尚未进入训练阶段，请稍后再查询。"
                )
                if progress_detail:
                    analysis_text += f"\n最新准备日志：{progress_detail}"
                preparing_train_type = normalize_train_type(pid_record.get("train_type") or pid_record.get("trainType"))
                preparing_launch_mode = (
                    pid_record.get("launch_mode")
                    or pid_record.get("launchMode")
                    or request_launch_mode
                )
                preparing_script = (
                    pid_record.get("script_name")
                    or pid_record.get("scriptName")
                    or request_script_name
                )
                payload = {
                    "status": "preparing",
                    "metrics": {
                        "container_name": container_name,
                        "pid": pid,
                        "pid_alive": True,
                        "training_process_exists": True,
                        "train_type": preparing_train_type or "unknown",
                        "train_type_public": public_train_type(preparing_train_type, preparing_launch_mode, preparing_script),
                        "train_type_text": public_train_type_text(preparing_train_type, preparing_launch_mode, preparing_script),
                        "launch_mode": preparing_launch_mode,
                        "launchMode": preparing_launch_mode,
                        "is_multinode": preparing_launch_mode == "multinode",
                        "isMultinode": preparing_launch_mode == "multinode",
                        "script_name": preparing_script,
                        "scriptName": preparing_script,
                        "launch_log_path": pid_record.get("launch_log_file"),
                        "preparing_elapsed_seconds": elapsed_seconds,
                        "preparation_detail": progress_detail,
                    },
                    "analysis_text": analysis_text,
                }
                return _monitor_response(payload)

        pid_started_at = None
        if pid and pid in start_times:
            pid_started_at = start_times.get(pid)
        elif pid_record and pid_record.get("started_at"):
            try:
                pid_started_at = datetime.fromisoformat(str(pid_record.get("started_at")))
            except Exception:
                pid_started_at = None

        selected_proc = None
        output_dir_source = "no_process"
        pid_matches_proc = False
        if pid and processes:
            for proc in processes:
                if proc.get("pid") == str(pid):
                    selected_proc = proc
                    pid_matches_proc = True
                    output_dir_source = "pid_match"
                    break
        if not selected_proc and not (pid_source == "input" and pid):
            selected_proc, output_dir_source = _select_process(
                processes,
                start_times,
                train_type,
                None,
                time_window_minutes,
            )

        output_dir = selected_proc.get("output_dir") if selected_proc else None
        if not train_type and selected_proc:
            inferred = _infer_train_type(selected_proc.get("cmd", ""))
            if inferred:
                train_type = inferred

        if not output_dir and selected_proc:
            log_file = _extract_log_file(selected_proc.get("cmd", ""))
            output_dir = _parse_output_dir_from_log(container_name, log_file) if log_file else None
            if output_dir:
                output_dir_source = "log_parse"

        if pid_record:
            record_train_type = normalize_train_type(pid_record.get("train_type") or pid_record.get("trainType"))
            record_script = str(pid_record.get("script_name") or pid_record.get("scriptName") or "")
            if record_train_type and (not train_type or record_script in {"dpo_train_launcher", "train_multinode_dpo_pipeline", "train_multinode_sft_pipeline"}):
                train_type = record_train_type
        train_type = normalize_train_type(train_type)
        launch_mode = None
        script_name_for_type = None
        if pid_record:
            launch_mode = pid_record.get("launch_mode") or pid_record.get("launchMode")
            script_name_for_type = pid_record.get("script_name") or pid_record.get("scriptName")
        if not launch_mode:
            launch_mode = request_launch_mode
        if not script_name_for_type:
            script_name_for_type = request_script_name
        if not script_name_for_type and selected_proc:
            script_name_for_type = selected_proc.get("cmd")
        train_type_value = train_type or "unknown"
        train_type_public = public_train_type(train_type, launch_mode, script_name_for_type)
        train_type_text = public_train_type_text(train_type, launch_mode, script_name_for_type)
        launch_log_tail_for_state = (
            _read_launch_log_tail(container_name, pid_record)
            if isinstance(pid_record, dict)
            else ""
        )
        dpo_launch_state = _parse_dpo_launch_log_state(launch_log_tail_for_state)
        uses_trainer_log_metrics = train_type_public in {
            "pretrain_lora",
            "pretrain_full",
            "lora_sft",
            "full_sft",
        }

        run_start_time: Optional[datetime] = None
        if selected_proc:
            run_start_time = start_times.get(selected_proc.get("pid", ""))
        if not run_start_time and pid_started_at:
            run_start_time = pid_started_at

        cutoff_ts = (datetime.now() - timedelta(minutes=time_window_minutes)).timestamp()
        if run_start_time:
            cutoff_ts = max(cutoff_ts, run_start_time.timestamp())

        if not output_dir:
            context_output_dir, context_output_dir_source = _resolve_output_dir_from_pid_context(
                container_name,
                pid,
                pid_record,
            )
            if context_output_dir:
                output_dir = context_output_dir
                output_dir_source = context_output_dir_source or "pid_context"

        allow_output_dir_scan = pid_source != "input"
        if not output_dir and allow_output_dir_scan:
            scan_root = (
                DEFAULT_PRETRAIN_ROOT
                if train_type_public in {"pretrain_lora", "pretrain_full"}
                else DEFAULT_BATCH_TRAIN_ROOT
            )
            output_dir = _scan_latest_dir(container_name, scan_root, cutoff_ts)
            if output_dir:
                output_dir_source = "scan:pretrain" if scan_root == DEFAULT_PRETRAIN_ROOT else "scan:batch_train"
        if not output_dir and allow_output_dir_scan:
            output_dir = _scan_latest_dir(container_name, DEFAULT_SAVES_ROOT, cutoff_ts)
            if output_dir:
                output_dir_source = "scan:saves"
        if not output_dir and train_type_public in {"pretrain_lora", "pretrain_full"}:
            output_dir = _scan_latest_dir(container_name, DEFAULT_PRETRAIN_ROOT, cutoff_ts)
            if output_dir:
                output_dir_source = "scan:pretrain"

        output_dir_exists = None
        if output_dir:
            output_dir_exists = _docker_exec(container_name, ["test", "-d", output_dir])[0] == 0

        training_process_exists = bool(processes) or pid_alive
        if pid_source == "input":
            training_process_exists = pid_alive or pid_matches_proc

        status_record = _get_status_record_by_pid(container_name, pid) if pid else None
        logger.info(
            "[train-monitor] pid_binding "
            "container=%s pid=%s pid_source=%s pid_alive=%s "
            "pid_matches_proc=%s training_process_exists=%s "
            "process_count=%s status_record_exists=%s",
            container_name,
            pid,
            pid_source,
            pid_alive,
            pid_matches_proc,
            training_process_exists,
            len(processes),
            bool(status_record),
        )
        if (not user_supplied_wandb_root) and wandb_root == DEFAULT_WANDB_ROOT:
            inferred_from_pid_record = _infer_wandb_root_from_pid_record(pid_record)
            if inferred_from_pid_record:
                wandb_root = inferred_from_pid_record
                wandb_root_source = "pid_registry"

        if (not user_supplied_wandb_root) and selected_proc and wandb_root == DEFAULT_WANDB_ROOT:
            cmd_wandb_root = _infer_wandb_root_from_cmd(selected_proc.get("cmd", ""))
            if cmd_wandb_root:
                wandb_root = cmd_wandb_root
                wandb_root_source = "process_cmd"

        training_log_path = (
            _find_trainer_stdout_log_by_output_dir(container_name, output_dir, train_type_public)
            if uses_trainer_log_metrics
            else None
        )
        wandb_url = _read_wandb_url_file(pid) if pid else None
        wandb_url_source = "pid_file" if wandb_url else None
        wandb_url_file = _wandb_url_file(pid) if pid else None
        wandb_url_file_exists = bool(wandb_url_file and os.path.exists(wandb_url_file))
        wandb_url_pid_file_hit = bool(wandb_url)
        if wandb_url and wandb_url_file and os.path.exists(wandb_url_file):
            try:
                file_mtime = os.path.getmtime(wandb_url_file)
                if not _should_use_pid_file_url(training_process_exists, pid_started_at, file_mtime):
                    wandb_url = None
                    wandb_url_source = None
            except Exception:
                pass
        if wandb_url:
            current_run_id = _extract_run_id_from_wandb_url(wandb_url)
            if current_run_id:
                existing_run_ids = _list_pid_url_run_ids(pid)
                if current_run_id in existing_run_ids:
                    wandb_url = None
                    wandb_url_source = None
        wandb_url_run_id = _extract_run_id_from_wandb_url(wandb_url)

        wandb_runs: List[Dict[str, Any]] = []
        wandb_root_candidates = _wandb_root_candidates(
            wandb_root,
            wandb_root_source,
            user_supplied_wandb_root,
        )
        wandb_run = _get_cached_wandb_run(
            container_name,
            pid,
            train_type_value,
            wandb_url_run_id,
        )
        wandb_select_reason = "cache_hit" if wandb_run else "no_runs"
        if not wandb_run and wandb_url_run_id:
            wandb_run, wandb_select_reason = _find_wandb_run_from_url(
                container_name,
                wandb_url,
                wandb_root_candidates,
            )
        if not wandb_run:
            seen_wandb_run_paths: set[str] = set()
            for candidate_root, candidate_source in wandb_root_candidates:
                candidate_runs = _list_wandb_runs(container_name, candidate_root)
                logger.info(
                    "[train-monitor] wandb_runs_discovered "
                    "container=%s pid=%s wandb_root=%s wandb_root_source=%s "
                    "wandb_runs_count=%s candidate_runs=%s",
                    container_name,
                    pid,
                    candidate_root,
                    candidate_source,
                    len(candidate_runs),
                    [run.get("name") for run in candidate_runs[:5]],
                )
                for candidate_run in candidate_runs:
                    run_path = str(candidate_run.get("path") or "").strip()
                    if not run_path or run_path in seen_wandb_run_paths:
                        continue
                    seen_wandb_run_paths.add(run_path)
                    annotated_run = dict(candidate_run)
                    annotated_run["_wandb_root"] = candidate_root
                    annotated_run["_wandb_root_source"] = candidate_source
                    wandb_runs.append(annotated_run)
            if pid:
                wandb_run, wandb_select_reason = _find_wandb_run_by_pid(
                    container_name,
                    wandb_runs,
                    pid,
                )
            if not wandb_run and pid_source == "input" and (pid_record or pid_matches_proc or run_start_time):
                wandb_run, wandb_select_reason = select_wandb_run_by_start_time(
                    wandb_runs,
                    run_start_time,
                    time_window_minutes,
                    allow_early_seconds=0 if pid_alive else 60,
                )
        pid_ok, pid_reason = validate_pid_binding(
            pid_source,
            pid_record,
            pid_matches_proc,
            bool(wandb_run or status_record),
            pid_alive,
        )
        if not pid_ok:
            msg = "未找到对应的 PID，请确认后重新输入。"
            if pid_reason == "pid_wandb_not_found":
                msg = "未找到该 PID 对应的训练记录，请确认 PID 是否正确。"
            payload = {
                "status": "failed",
                "metrics": {
                    "container_name": container_name,
                    "pid": pid,
                    "pid_alive": pid_alive,
                    "error_reason": pid_reason,
                },
                "analysis_text": msg,
            }
            return _monitor_response(payload)
        if pid_reason == "pid_ended" and not wandb_run and not status_record:
            completed_payload = _try_build_finished_payload_from_ended_pid_output_log(
                container_name,
                pid,
                pid_record,
                wandb_root,
                wandb_runs,
                run_start_time,
                time_window_minutes,
                history_limit,
            )
            if completed_payload:
                if container_name:
                    _LAST_OK_BY_CONTAINER[container_name] = {
                        "payload": json.loads(json.dumps(completed_payload, ensure_ascii=False)),
                        "cached_at": datetime.now().isoformat(),
                    }
                return _monitor_response(completed_payload)

            launch_log_tail = _read_launch_log_tail(container_name, pid_record, line_count=500)
            completed_payload = _build_finished_payload_from_grpo_launch_log(
                container_name,
                pid,
                pid_record,
                launch_log_tail,
            )
            if completed_payload:
                if container_name:
                    _LAST_OK_BY_CONTAINER[container_name] = {
                        "payload": json.loads(json.dumps(completed_payload, ensure_ascii=False)),
                        "cached_at": datetime.now().isoformat(),
                    }
                return _monitor_response(completed_payload)

            completed_payload = _build_finished_payload_from_dpo_launch_log(
                container_name,
                pid,
                pid_record,
                launch_log_tail,
            )
            if completed_payload:
                if container_name:
                    _LAST_OK_BY_CONTAINER[container_name] = {
                        "payload": json.loads(json.dumps(completed_payload, ensure_ascii=False)),
                        "cached_at": datetime.now().isoformat(),
                    }
                return _monitor_response(completed_payload)

            disk_warning = _extract_ray_disk_warning(launch_log_tail)
            disk_warning_summary = _format_disk_warning_summary(disk_warning)
            error_summary = (
                _summarize_training_error_from_logs(disk_warning_summary, launch_log_tail)
                or "训练进程已结束，但没有找到本次训练的指标或完成记录，可能是启动失败、手动停止或进程提前退出。"
            )
            analysis_text = f"训练已中断或异常结束。原因：{error_summary}"
            if launch_log_tail and not disk_warning_summary:
                launch_log_summary = _format_launch_log_tail_for_user(
                    launch_log_tail,
                    log_path=(pid_record or {}).get("launch_log_file"),
                )
                if launch_log_summary:
                    analysis_text += f"\n启动日志摘要：\n{launch_log_summary}"
            payload = {
                "status": "interrupted",
                "metrics": {
                    "container_name": container_name,
                    "pid": pid,
                    "pid_alive": pid_alive,
                    "training_process_exists": False,
                    "error_reason": "pid_ended_no_wandb",
                    "launch_log_path": (pid_record or {}).get("launch_log_file"),
                    "launch_log_tail": launch_log_tail,
                    "disk_warning": disk_warning,
                    "disk_warning_summary": disk_warning_summary or None,
                    "error_summary": error_summary,
                },
                "analysis_text": analysis_text,
            }
            return _monitor_response(payload)
        if should_return_status_record(
            pid_source,
            pid_alive,
            pid_matches_proc,
            bool(wandb_run),
            bool(status_record),
        ):
            payload = _build_payload_from_status_record(container_name, status_record or {})
            return _monitor_response(payload)
        if not wandb_run:
            if pid_source == "input":
                wandb_select_reason = "pid_no_wandb_match"
            else:
                wandb_run, wandb_select_reason = select_wandb_run(
                    wandb_runs,
                    run_start_time,
                    time_window_minutes,
                    fallback_latest=not bool(pid or run_start_time),
                )
        if wandb_run:
            selected_wandb_root = str(wandb_run.get("_wandb_root") or "").strip()
            selected_wandb_root_source = str(wandb_run.get("_wandb_root_source") or "").strip()
            if selected_wandb_root:
                wandb_root = selected_wandb_root
            if selected_wandb_root_source:
                wandb_root_source = selected_wandb_root_source
        logger.info(
            "[train-monitor] wandb_run_selected "
            "container=%s pid=%s wandb_select_reason=%s wandb_run_dir=%s "
            "wandb_run_name=%s run_start_time=%s time_window_minutes=%s",
            container_name,
            pid,
            wandb_select_reason,
            wandb_run.get("path") if wandb_run else None,
            wandb_run.get("name") if wandb_run else None,
            run_start_time.isoformat() if run_start_time else None,
            time_window_minutes,
        )
        wandb_run_dir = wandb_run.get("path") if wandb_run else None
        wandb_run_name = wandb_run.get("name") if wandb_run else None
        wandb_mode = _infer_wandb_mode(pid_record, wandb_run_name, wandb_run_dir)
        wandb_data: Dict[str, Any] = {}
        if wandb_run_dir:
            wandb_data = _read_wandb_run(container_name, wandb_run_dir, history_limit)
        output_log_data: Dict[str, Any] = {}
        output_log_full_data: Optional[Dict[str, Any]] = None
        output_log_path = (
            os.path.join(wandb_run_dir, "files", "output.log")
            if wandb_run_dir
            else None
        )
        if output_log_path:
            output_log_data = _read_output_log(container_name, output_log_path, history_limit)
            count = output_log_data.get("count")
            if isinstance(count, int) and count > history_limit:
                output_log_full_data = _read_output_log(container_name, output_log_path, count)

        if not wandb_url and pid and pid_alive:
            pid_log_path = _find_log_file_by_pid(container_name, pid)
            if not training_log_path:
                training_log_path = pid_log_path
            for candidate_log_path in [training_log_path, pid_log_path]:
                if not candidate_log_path:
                    continue
                wandb_url = _read_wandb_url_from_log_file(container_name, candidate_log_path)
                if wandb_url:
                    dup_run_id = _extract_run_id_from_wandb_url(wandb_url)
                    if dup_run_id and dup_run_id in _list_pid_url_run_ids(pid):
                        wandb_url = None
                        wandb_url_source = None
                    else:
                        _write_wandb_url_file(pid, wandb_url)
                        wandb_url_source = "training_log"
                        training_log_path = candidate_log_path
                        break
        wandb_url_run_id = _extract_run_id_from_wandb_url(wandb_url)
        run_id = wandb_url_run_id or _extract_wandb_run_id(wandb_run_name, wandb_run_dir)
        if run_id and (training_log_path is None or wandb_url is None):
            matched_logs = _find_log_files_by_run_id(container_name, DEFAULT_LOG_ROOT, run_id)
            for matched_log in matched_logs:
                if not training_log_path:
                    training_log_path = matched_log
                if wandb_url:
                    break
                candidate_url = _read_wandb_url_from_log_file(container_name, matched_log)
                if not candidate_url:
                    continue
                dup_run_id = _extract_run_id_from_wandb_url(candidate_url)
                if dup_run_id and dup_run_id in _list_pid_url_run_ids(pid):
                    continue
                wandb_url = candidate_url
                training_log_path = matched_log
                _write_wandb_url_file(pid, wandb_url)
                wandb_url_source = "training_log"
                break
        pid_record_log_path = _extract_log_file(str(pid_record.get("command") or "")) if pid_record else None
        if (not training_log_path) and pid_record_log_path:
            training_log_path = pid_record_log_path
        trainer_stdout_log_path = (
            _find_trainer_stdout_log_by_output_dir(container_name, output_dir, train_type_public)
            if uses_trainer_log_metrics
            else None
        )
        if trainer_stdout_log_path and (
            not training_log_path
            or str(training_log_path).startswith("/usr/local/insinfersystem/")
        ):
            training_log_path = trainer_stdout_log_path
        if (not wandb_url) and pid_record_log_path and pid_record_log_path != training_log_path and wandb_mode != "offline":
            log_wandb_url = _read_wandb_url_from_log_file(container_name, pid_record_log_path)
            if log_wandb_url:
                dup_run_id = _extract_run_id_from_wandb_url(log_wandb_url)
                if not (dup_run_id and dup_run_id in _list_pid_url_run_ids(pid)):
                    wandb_url = log_wandb_url
                    training_log_path = pid_record_log_path
                    _write_wandb_url_file(pid, wandb_url)
                    wandb_url_source = "training_log"
        if not wandb_url and training_log_path and wandb_mode != "offline":
            log_wandb_url = _read_wandb_url_from_log_file(container_name, training_log_path)
            if log_wandb_url:
                dup_run_id = _extract_run_id_from_wandb_url(log_wandb_url)
                if dup_run_id and dup_run_id in _list_pid_url_run_ids(pid):
                    wandb_url = None
                    wandb_url_source = None
                else:
                    wandb_url = log_wandb_url
                    _write_wandb_url_file(pid, wandb_url)
                    wandb_url_source = "training_log"
        wandb_url_run_id = _extract_run_id_from_wandb_url(wandb_url)
        training_log_data: Dict[str, Any] = {}
        if training_log_path and (
            not output_log_data or not output_log_data.get("latest")
        ):
            training_log_data = _read_output_log(container_name, training_log_path, history_limit)
        trainer_log_path = f"{str(output_dir).rstrip('/')}/trainer_log.jsonl" if output_dir else None
        trainer_log_data: Dict[str, Any] = {}
        if uses_trainer_log_metrics and trainer_log_path:
            trainer_log_data = _read_trainer_log(container_name, trainer_log_path, history_limit)
        logger.info(
            "[train-monitor] wandb_url_discovery "
            "container=%s pid=%s wandb_url_file=%s wandb_url_file_exists=%s "
            "pid_file_hit=%s training_log_path=%s training_log_url_hit=%s "
            "wandb_url_source=%s wandb_url_run_id=%s",
            container_name,
            pid,
            wandb_url_file,
            wandb_url_file_exists,
            wandb_url_pid_file_hit,
            training_log_path,
            bool(wandb_url and wandb_url_source == "training_log"),
            wandb_url_source,
            _extract_run_id_from_wandb_url(wandb_url),
        )
        output_or_training_log_data = _select_effective_log_data(output_log_data, training_log_data)
        effective_output_log_data = (
            _select_effective_log_data(trainer_log_data, output_or_training_log_data)
            if uses_trainer_log_metrics
            else output_or_training_log_data
        )
        wandb_run_dir_corrected = None
        if wandb_url_run_id and not (wandb_run_dir and wandb_url_run_id in wandb_run_dir):
            matched_run_dir = _find_wandb_run_dir_by_id(container_name, wandb_root, wandb_url_run_id)
            if not matched_run_dir:
                for candidate_root in (DEFAULT_LOG_WANDB_ROOT, DEFAULT_WANDB_ROOT, DEFAULT_INSINFER_WANDB_ROOT):
                    if candidate_root == wandb_root:
                        continue
                    matched_run_dir = _find_wandb_run_dir_by_id(
                        container_name, candidate_root, wandb_url_run_id
                    )
                    if matched_run_dir:
                        break
            if matched_run_dir and matched_run_dir != wandb_run_dir:
                wandb_run_dir = matched_run_dir
                wandb_run_name = os.path.basename(matched_run_dir)
                wandb_mode = _infer_wandb_mode(pid_record, wandb_run_name, wandb_run_dir)
                wandb_select_reason = "wandb_url_match"
                wandb_run_dir_corrected = matched_run_dir
                wandb_data = _read_wandb_run(container_name, wandb_run_dir, history_limit)
                output_log_data = {}
                output_log_full_data = None
                output_log_path = os.path.join(wandb_run_dir, "files", "output.log")
                output_log_data = _read_output_log(container_name, output_log_path, history_limit)
                count = output_log_data.get("count")
                if isinstance(count, int) and count > history_limit:
                    output_log_full_data = _read_output_log(container_name, output_log_path, count)
                output_or_training_log_data = _select_effective_log_data(output_log_data, training_log_data)
                effective_output_log_data = (
                    _select_effective_log_data(trainer_log_data, output_or_training_log_data)
                    if uses_trainer_log_metrics
                    else output_or_training_log_data
                )

                history = wandb_data.get("history") or []
                history = _normalize_history_loss(history)
                history_count = wandb_data.get("history_count", len(history))
                if history_count is None:
                    history_count = len(history)
                last_update_time = wandb_data.get("last_update_time")
                loss_source = "wandb"
                raw_latest_loss = (wandb_data.get("latest") or {}).get("loss")
                latest_loss, loss_parse_note = _normalize_loss_value(raw_latest_loss)
                latest_epoch = (wandb_data.get("latest") or {}).get("epoch")
                latest_step = (wandb_data.get("latest") or {}).get("step")
                if latest_step is None:
                    latest_step = _latest_integer_step_from_history(history)
                output_latest = effective_output_log_data.get("latest") if effective_output_log_data else None
                should_use_output_log = False
                if output_latest:
                    output_latest_step = output_latest.get("step")
                    output_last_update_time = effective_output_log_data.get("last_update_time")
                    output_has_step = isinstance(output_latest_step, (int, float)) or (
                        isinstance(output_latest_step, str) and output_latest_step.strip() != ""
                    )
                    if latest_loss is None and output_has_step:
                        should_use_output_log = True
                    elif isinstance(output_latest_step, (int, float)) and isinstance(latest_step, (int, float)):
                        should_use_output_log = output_latest_step >= latest_step
                    elif output_has_step and output_last_update_time and last_update_time:
                        try:
                            should_use_output_log = datetime.fromisoformat(output_last_update_time) >= datetime.fromisoformat(last_update_time)
                        except ValueError:
                            should_use_output_log = False
                    elif output_has_step and output_last_update_time and not last_update_time:
                        should_use_output_log = True

                if should_use_output_log:
                    history = effective_output_log_data.get("history") or []
                    history = _normalize_history_loss(history)
                    history_count = len(history)
                    last_update_time = effective_output_log_data.get("last_update_time") or last_update_time
                    if effective_output_log_data is trainer_log_data:
                        loss_source = "trainer_log"
                    elif effective_output_log_data is training_log_data:
                        loss_source = "training_log"
                    else:
                        loss_source = "output.log"
                    raw_latest_loss = (effective_output_log_data.get("latest") or {}).get("loss")
                    latest_loss, loss_parse_note = _normalize_loss_value(raw_latest_loss)
                    latest_epoch = (effective_output_log_data.get("latest") or {}).get("epoch")
                    latest_step = (effective_output_log_data.get("latest") or {}).get("step")
                    if latest_step is None:
                        latest_step = _latest_integer_step_from_history(history)
                snapshot_dir = _persist_wandb_snapshot(
                    container_name,
                    wandb_run_name,
                    wandb_run_dir,
                    wandb_data,
                    output_log_data=effective_output_log_data,
                    output_log_full_data=output_log_full_data,
                    metrics_history=history,
                    metrics_source=loss_source,
                )
            elif wandb_run_dir and wandb_url_run_id not in wandb_run_dir:
                wandb_run_dir = None
                wandb_run_name = None
                wandb_select_reason = "wandb_url_mismatch"
                wandb_run_dir_corrected = None

        pending_reset_has_metrics = bool(
            _log_data_has_any_metrics(wandb_data)
            or _log_data_has_any_metrics(output_log_data)
            or _log_data_has_any_metrics(training_log_data)
            or _log_data_has_any_metrics(trainer_log_data)
        )
        if (
            wandb_mode != "offline"
            and training_process_exists
            and not wandb_url
            and wandb_run_dir
            and not pending_reset_has_metrics
            and wandb_select_reason in {
                "by_start_time",
                "by_mtime",
                "fallback_latest",
            }
        ):
            logger.info(
                "[train-monitor] pending_wandb_url_reset_before "
                "container=%s pid=%s wandb_run_dir=%s wandb_select_reason=%s "
                "training_log_path=%s latest_loss_before_reset=%s "
                "history_count_before_reset=%s output_log_latest_exists=%s "
                "training_log_latest_exists=%s",
                container_name,
                pid,
                wandb_run_dir,
                wandb_select_reason,
                training_log_path,
                ((wandb_data.get("latest") or {}).get("loss") if wandb_data else None),
                wandb_data.get("history_count", len(wandb_data.get("history") or [])) if wandb_data else 0,
                bool(output_log_data.get("latest")) if output_log_data else False,
                bool(training_log_data.get("latest")) if training_log_data else False,
            )
            wandb_run_dir = None
            wandb_run_name = None
            wandb_mode = _infer_wandb_mode(pid_record, wandb_run_name, wandb_run_dir)
            wandb_select_reason = "pending_wandb_url"
            wandb_data = {}
            output_log_data = {}
            output_log_full_data = None
            output_or_training_log_data = _select_effective_log_data(training_log_data, output_log_data)
            effective_output_log_data = (
                _select_effective_log_data(trainer_log_data, output_or_training_log_data)
                if uses_trainer_log_metrics
                else output_or_training_log_data
            )

            logger.info(
                "[train-monitor] pending_wandb_url_reset_after "
                "container=%s pid=%s wandb_select_reason=%s wandb_run_dir=%s "
                "effective_output_log_latest_exists=%s",
                container_name,
                pid,
                wandb_select_reason,
                wandb_run_dir,
                bool(effective_output_log_data.get("latest")) if effective_output_log_data else False,
            )
        history = wandb_data.get("history") or []
        history = _normalize_history_loss(history)
        history_count = wandb_data.get("history_count", len(history))
        if history_count is None:
            history_count = len(history)
        last_update_time = wandb_data.get("last_update_time")
        loss_source = "wandb"

        raw_latest_loss = (wandb_data.get("latest") or {}).get("loss")
        latest_loss, loss_parse_note = _normalize_loss_value(raw_latest_loss)
        latest_epoch = (wandb_data.get("latest") or {}).get("epoch")
        latest_step = (wandb_data.get("latest") or {}).get("step")
        if latest_step is None:
            latest_step = _latest_integer_step_from_history(history)

        output_latest = effective_output_log_data.get("latest") if effective_output_log_data else None
        should_use_output_log = False
        if output_latest:
            output_latest_step = output_latest.get("step")
            output_last_update_time = effective_output_log_data.get("last_update_time")
            output_has_step = isinstance(output_latest_step, (int, float)) or (
                isinstance(output_latest_step, str) and output_latest_step.strip() != ""
            )
            if latest_loss is None and output_has_step:
                should_use_output_log = True
            elif isinstance(output_latest_step, (int, float)) and isinstance(latest_step, (int, float)):
                should_use_output_log = output_latest_step >= latest_step
            elif output_has_step and output_last_update_time and last_update_time:
                try:
                    should_use_output_log = datetime.fromisoformat(output_last_update_time) >= datetime.fromisoformat(last_update_time)
                except ValueError:
                    should_use_output_log = False
            elif output_has_step and output_last_update_time and not last_update_time:
                should_use_output_log = True

        if should_use_output_log:
            history = effective_output_log_data.get("history") or []
            history = _normalize_history_loss(history)
            history_count = len(history)
            last_update_time = effective_output_log_data.get("last_update_time") or last_update_time
            if effective_output_log_data is trainer_log_data:
                loss_source = "trainer_log"
            elif effective_output_log_data is training_log_data:
                loss_source = "training_log"
            else:
                loss_source = "output.log"
            raw_latest_loss = (effective_output_log_data.get("latest") or {}).get("loss")
            latest_loss, loss_parse_note = _normalize_loss_value(raw_latest_loss)
            latest_epoch = (effective_output_log_data.get("latest") or {}).get("epoch")
            latest_step = (effective_output_log_data.get("latest") or {}).get("step")
            if latest_step is None:
                latest_step = _latest_integer_step_from_history(history)

        logger.info(
            "[train-monitor] metrics_read "
            "container=%s pid=%s wandb_history_count=%s "
            "wandb_latest_loss_present=%s output_log_latest_present=%s "
            "training_log_latest_present=%s trainer_log_latest_present=%s "
            "loss_source=%s latest_loss=%s "
            "latest_step=%s wandb_select_reason=%s wandb_run_dir=%s",
            container_name,
            pid,
            history_count,
            ((wandb_data.get("latest") or {}).get("loss") is not None) if wandb_data else False,
            bool(output_log_data.get("latest")) if output_log_data else False,
            bool(training_log_data.get("latest")) if training_log_data else False,
            bool(trainer_log_data.get("latest")) if trainer_log_data else False,
            loss_source,
            latest_loss,
            latest_step,
            wandb_select_reason,
            wandb_run_dir,
        )
        snapshot_dir = _persist_wandb_snapshot(
            container_name,
            wandb_run_name,
            wandb_run_dir,
            wandb_data,
            output_log_data=effective_output_log_data,
            output_log_full_data=output_log_full_data,
            metrics_history=history,
            metrics_source=loss_source,
        )

        if effective_output_log_data is output_log_data:
            output_log_payload = output_log_full_data or effective_output_log_data
        else:
            output_log_payload = effective_output_log_data
        output_log_error = output_log_payload.get("error_hint") if output_log_payload else None
        output_log_success_hint = output_log_payload.get("success_hint") if output_log_payload else None
        output_log_finished = bool(output_log_payload.get("success_found")) if output_log_payload else False
        completion_info = output_log_payload.get("completion_info") if output_log_payload else {}
        if not isinstance(completion_info, dict):
            completion_info = {}
        latest_learning_rate = None
        latest_kl_coef = None
        if output_log_payload:
            latest_metrics = output_log_payload.get("latest") or {}
            latest_learning_rate = (
                latest_metrics.get("learning_rate")
                if latest_metrics.get("learning_rate") is not None
                else latest_metrics.get("actor_lr")
            )
            latest_kl_coef = latest_metrics.get("actor_kl_coef")
        progress_percent = output_log_payload.get("progress_percent") if output_log_payload else None
        current_step = output_log_payload.get("current_step") if output_log_payload else None
        total_steps = output_log_payload.get("total_steps") if output_log_payload else None
        elapsed_time = output_log_payload.get("elapsed_time") if output_log_payload else None
        remaining_time = output_log_payload.get("remaining_time") if output_log_payload else None
        progress_realtime_confirmed = (
            bool(output_log_payload.get("progress_realtime_confirmed"))
            if output_log_payload
            else False
        )
        progress_realtime_reason = output_log_payload.get("progress_realtime_reason") if output_log_payload else None
        dpo_sub_stage = output_log_payload.get("sub_stage") if output_log_payload else None
        dpo_sub_stage_text = output_log_payload.get("sub_stage_text") if output_log_payload else None
        export_dir = dpo_launch_state.get("export_dir")
        merge_status = dpo_launch_state.get("merge_status")
        if dpo_launch_state.get("sub_stage"):
            dpo_sub_stage = dpo_launch_state.get("sub_stage")
            dpo_sub_stage_text = dpo_launch_state.get("sub_stage_text")
        if dpo_sub_stage == "merge" and merge_status == "running":
            progress_percent = None
            current_step = None
            total_steps = None
            elapsed_time = None
            remaining_time = None
            progress_realtime_confirmed = False
            progress_realtime_reason = "launcher_merge_in_progress"
        history, resume_history_info = _apply_resume_history_offset(history, pid_record)
        latest_step, resume_step_info = _apply_resume_step_offset(
            latest_step,
            current_step,
            pid_record,
            force_local_step=bool(resume_history_info),
        )
        resume_adjustment_info = resume_step_info or resume_history_info
        axis = "step" if latest_step is not None else ("epoch" if latest_epoch is not None else "step")
        axis_note = "step_available" if latest_step is not None else ("epoch_available" if latest_epoch is not None else "epoch_missing_use_step")

        iteration_finished = bool(wandb_data.get("finished"))
        wandb_failed = bool(wandb_data.get("failed"))
        completion_reason = completion_info.get("completion_reason") if output_log_finished else None
        completion_source = None
        if output_log_finished:
            completion_source = (
                "wandb_output_log"
                if output_log_payload and str(output_log_payload.get("path") or "").endswith("/files/output.log")
                else "training_log"
            )
            if not completion_reason:
                completion_reason = "output_log_success_hint"
        elif iteration_finished:
            completion_reason = "wandb_finished_state"
            completion_source = "wandb_state"

        status = "running"
        error_reason = None
        if not training_process_exists:
            if wandb_failed:
                status = "failed"
                exitcode = wandb_data.get("exitcode")
                error_reason = f"wandb_exitcode_{exitcode}" if exitcode is not None else "wandb_exitcode_nonzero"
            elif output_log_error:
                status = "failed"
                error_reason = "output_log_error"
            elif iteration_finished or output_log_finished:
                status = "finished"
            elif history_count or latest_loss is not None:
                status = "interrupted"
                error_reason = error_reason or "process_ended_without_success_marker"
            else:
                status = "failed"
                error_reason = error_reason or "training_process_not_found"
        else:
            if not wandb_run_dir:
                if output_log_payload and output_log_payload.get("training_start_signal"):
                    status = "running"
                else:
                    status = "starting"
                error_reason = error_reason or "wandb_run_not_found"
            elif not history and latest_loss is None:
                status = "starting"
                error_reason = error_reason or "wandb_no_metrics"

        if output_dir and output_dir_exists is False and training_process_exists and status == "running":
            status = "starting"
            error_reason = error_reason or "output_dir_missing_but_process_running"
        elif output_dir and output_dir_exists is False and not training_process_exists and status == "running":
            status = "failed"
            error_reason = error_reason or "output_dir_missing"

        launcher_finished_success = bool(
            merge_status == "finished"
            and dpo_launch_state.get("launcher_finished")
            and not dpo_launch_state.get("merge_failed")
        )
        if merge_status == "failed":
            status = "failed"
            error_reason = "merge_failed"
            completion_reason = None
            completion_source = None
        elif merge_status == "running" and training_process_exists:
            status = "running"
            if error_reason in {
                "wandb_run_not_found",
                "wandb_no_metrics",
                "output_dir_missing_but_process_running",
            }:
                error_reason = None
        elif launcher_finished_success:
            status = "finished"
            error_reason = None
            completion_reason = completion_reason or "launcher_merge_success"
            completion_source = completion_source or "launch_log"

        if history_count == 0 and latest_loss is not None:
            history_count = 1

        effective_history_count = history_count
        if isinstance(history, list):
            effective_history_count = max(effective_history_count, len(history))

        llm_ready = False
        if effective_history_count >= DEFAULT_MIN_HISTORY_FOR_LLM:
            llm_ready = True
        elif isinstance(latest_step, (int, float)) and latest_step >= DEFAULT_MIN_HISTORY_FOR_LLM:
            llm_ready = True
        elif isinstance(latest_epoch, (int, float)) and latest_epoch >= DEFAULT_MIN_HISTORY_FOR_LLM:
            llm_ready = True

        stale = False
        stale_minutes = None
        if last_update_time:
            try:
                last_dt = datetime.fromisoformat(last_update_time)
                stale_minutes = round((datetime.now() - last_dt).total_seconds() / 60, 2)
                if stale_minutes >= DEFAULT_STALE_MINUTES:
                    stale = True
            except ValueError:
                pass

        loss_warning = None
        loss_available = latest_loss is not None
        loss_is_finite = True
        if loss_available:
            loss_is_finite = isinstance(latest_loss, (int, float)) and math.isfinite(latest_loss)
            if not loss_is_finite:
                last_finite, last_axis = _find_last_finite_loss(history, lookback=20)
                if last_finite is not None and training_process_exists and not stale:
                    loss_warning = f"latest_loss_not_finite_use_last_finite(axis={last_axis})"
                    latest_loss = last_finite
                    loss_is_finite = True

        if stale and merge_status != "running" and status != "finished":
            status = "failed"
            error_reason = error_reason or "metrics_stale"
        status = _apply_finished_wandb_status(
            status,
            iteration_finished,
            training_process_exists,
        )
        if merge_status == "failed":
            status = "failed"
            error_reason = "merge_failed"
        elif merge_status == "running" and training_process_exists:
            status = "running"
            error_reason = None
        elif launcher_finished_success:
            status = "finished"
            error_reason = None
            completion_reason = completion_reason or "launcher_merge_success"
            completion_source = completion_source or "launch_log"
        if loss_available and not loss_is_finite:
            status = "failed"
            error_reason = error_reason or "loss_not_finite"
        waiting_for_metrics = bool(
            training_process_exists
            and latest_loss is None
            and effective_history_count == 0
        )
        if waiting_for_metrics and status not in {"finished", "failed", "interrupted"}:
            if status not in {"running", "starting"}:
                status = "starting"
        if status == "failed" and not error_reason:
            error_reason = "unknown"

        error_summary = ""
        if status in {"failed", "interrupted", "unknown", "stopped"}:
            error_summary = _summarize_training_error_from_logs(
                output_log_error,
                trainer_log_data.get("error") if trainer_log_data else None,
                launch_log_tail_for_state,
                error_reason,
            )

        if latest_step is None and current_step is not None:
            latest_step = current_step
        if latest_step is None:
            latest_step = _latest_integer_step_from_history(history)
        if status == "finished":
            progress_percent = "100%"
            if current_step is None and latest_step is not None:
                current_step = latest_step
        else:
            progress_value = _to_float_or_none(str(progress_percent).rstrip("%") if progress_percent is not None else None)
            if progress_value is not None:
                progress_percent = f"{max(0.0, min(100.0, progress_value)):g}%"

        if pid and status in {"finished", "failed", "interrupted"}:
            _record_train_status(
                container_name,
                pid,
                status,
                error_reason,
                latest_loss,
                latest_epoch,
                latest_step,
                wandb_run_name,
                wandb_run_dir,
                wandb_mode,
                output_log_error,
                output_log_finished,
                last_update_time,
                completion_reason if status == "finished" else None,
                completion_source if status == "finished" else None,
            )

        # ===== 新增：统一“先报数”摘要（与是否调用 LLM 无关）=====
        def _fmt_float(v: Any, nd: int = 4) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, (int, float)):
                try:
                    if math.isfinite(float(v)):
                        return f"{float(v):.{nd}f}"
                    return str(v)
                except Exception:
                    return str(v)
            return str(v)

        def _status_cn(s: str) -> str:
            return {
                "running": "运行中",
                "finished": "已结束",
                "failed": "异常",
                "starting": "启动中",
                "interrupted": "中断",
                "unknown": "未知",
            }.get(s, s)

        indicator_text = (
            f"指标：状态={_status_cn(status)}，"
            f"loss={_fmt_float(latest_loss, 4)}，"
            f"step={latest_step if latest_step is not None else 'N/A'}，"
            f"来源={loss_source}，"
            f"last_update={last_update_time or 'N/A'}"
        )
        if latest_step is None:
            indicator_text = indicator_text.replace(
                "step=N/A，",
                f"epoch={_fmt_float(latest_epoch, 4)}，step=N/A，",
            )
        if stale_minutes is not None:
            indicator_text += f"（stale={stale}，{stale_minutes}min）"

        # ===== LLM 评价输出规则（按你的要求） =====
        llm_called = False
        llm_err = None
        if merge_status == "running" and dpo_sub_stage == "merge":
            analysis_text = "训练迭代已结束，正在进行 Merge/Export。"
        elif iteration_finished and training_process_exists:
            analysis_text = "迭代结束，在保存模型。"
        elif allow_llm and llm_ready:
            analysis_input = {
                "status": status,
                "train_type": train_type_value,
                "train_type_public": train_type_public,
                "train_type_text": train_type_text,
                "container_name": container_name,
                "output_dir": output_dir,
                "loss_source": loss_source,
                "axis": axis,
                "axis_note": axis_note,
                "latest": {
                    "loss": latest_loss,
                    "epoch": latest_epoch,
                    "step": latest_step,
                    "finished": iteration_finished,
                    "last_update_time": last_update_time,
                },
                "stale": stale,
                "stale_minutes": stale_minutes,
                "history": history,
            }
            llm_text, llm_err = _call_monitor_llm(model_name, base_url, analysis_input, api_key)
            llm_called = True
            if llm_err:
                analysis_text = "训练评价生成失败，请检查相关配置或联系技术支持。"
            else:
                analysis_text = llm_text or "训练评价生成失败，请稍后重试。"

        # 未触发 LLM：保持原 JSON payload（尽量不改动无关输出协议）
        if not (iteration_finished and training_process_exists) and not llm_called:
            analysis_text = "历史数据不足，暂无法评估训练趋势。"
            if status == "finished":
                analysis_text = f"训练已结束，最后loss为{latest_loss}，请根据需要检查最终指标与模型导出。"
            elif status == "failed":
                if error_summary:
                    analysis_text = f"当前训练状态异常。原因：{error_summary}"
                else:
                    analysis_text = f"当前训练状态异常，错误原因为“{error_reason}”，请检查日志与容器进程以进一步排查问题。"
            elif status == "interrupted":
                reason_text = error_summary or "未检测到正常完成标记，可能是进程提前退出或被手动终止。"
                analysis_text = f"训练已中断或提前停止，最后loss为{latest_loss}。原因：{reason_text}"
            elif status == "starting":
                analysis_text = f"{STARTING_TEXT}（指标数据尚未写入）"
            elif merge_status == "running" and dpo_sub_stage == "merge":
                analysis_text = "训练迭代已结束，正在进行 Merge/Export。"
            elif latest_loss is None and training_process_exists:
                analysis_text = "已检测到训练进程，正在等待 loss、lr 等指标写入。"

        # ===== 新增：无论是否调用 LLM，都把“指标”放在最前面 =====
        analysis_text = f"{indicator_text}\n{analysis_text}"

        metrics_log_path = (output_log_payload.get("path") if output_log_payload else None) or training_log_path
        if output_log_payload is trainer_log_data:
            metrics_log_source = "trainer_log"
        elif output_log_payload is training_log_data:
            metrics_log_source = "training_log"
        elif output_log_payload:
            metrics_log_source = (
                "wandb_output_log"
                if str(output_log_payload.get("path") or "").endswith("/files/output.log")
                else loss_source
            )
        else:
            metrics_log_source = None

        payload = {
            "status": status,
            "metrics": {
                "container_name": container_name,
                "train_type": train_type_value,
                "train_type_public": train_type_public,
                "train_type_text": train_type_text,
                "launch_mode": launch_mode,
                "launchMode": launch_mode,
                "is_multinode": launch_mode == "multinode",
                "isMultinode": launch_mode == "multinode",
                "script_name": script_name_for_type,
                "scriptName": script_name_for_type,
                "output_dir": output_dir,
                "output_dir_source": output_dir_source,
                "output_dir_exists": output_dir_exists,
                "wandb_root": wandb_root,
                "wandb_root_source": wandb_root_source,
                "wandb_run_dir": wandb_run_dir,
                "wandb_run_name": wandb_run_name,
                "wandb_mode": wandb_mode,
                "wandb_select_reason": wandb_select_reason,
                "wandb_snapshot_dir": snapshot_dir,
                "wandb_url": wandb_url,
                "wandb_url_source": wandb_url_source,
                "wandb_url_file": wandb_url_file,
                "wandb_url_run_id": wandb_url_run_id,
                "wandb_run_dir_corrected": wandb_run_dir_corrected,
                "wandb_url_pending": bool(training_process_exists and not wandb_url and wandb_mode != "offline"),
                "training_log_path": training_log_path,
                "metrics_log_path": metrics_log_path,
                "metrics_log_source": metrics_log_source,
                "pid": pid,
                "pid_alive": pid_alive,
                "pid_source": pid_source,
                "pid_started_at": pid_started_at.isoformat() if pid_started_at else None,
                "training_process_exists": training_process_exists,
                "loss_source": loss_source,
                "latest_loss": latest_loss,
                "latest_learning_rate": latest_learning_rate,
                "latest_kl_coef": latest_kl_coef,
                "latest_epoch": latest_epoch if latest_step is None else None,
                "latest_step": latest_step,
                "resume_checkpoint": (
                    resume_adjustment_info.get("resume_checkpoint") if resume_adjustment_info else None
                ),
                "resume_checkpoint_step": (
                    resume_adjustment_info.get("resume_checkpoint_step") if resume_adjustment_info else None
                ),
                "raw_latest_step": (
                    resume_step_info.get("raw_latest_step") if resume_step_info else None
                ),
                "step_adjustment": (
                    resume_step_info.get("step_adjustment") if resume_step_info else None
                ),
                "history_step_adjustment": (
                    resume_history_info.get("history_step_adjustment") if resume_history_info else None
                ),
                "raw_history_step_min": (
                    resume_history_info.get("raw_history_step_min") if resume_history_info else None
                ),
                "raw_history_step_max": (
                    resume_history_info.get("raw_history_step_max") if resume_history_info else None
                ),
                "progress_percent": progress_percent,
                "current_step": current_step,
                "total_steps": total_steps,
                "elapsed_time": elapsed_time,
                "remaining_time": remaining_time,
                "sub_stage": dpo_sub_stage,
                "sub_stage_text": dpo_sub_stage_text,
                "train_progress_percent": output_log_payload.get("train_progress_percent") if output_log_payload else None,
                "train_current_step": output_log_payload.get("train_current_step") if output_log_payload else None,
                "train_total_steps": output_log_payload.get("train_total_steps") if output_log_payload else None,
                "eval_progress_percent": output_log_payload.get("eval_progress_percent") if output_log_payload else None,
                "eval_current_step": output_log_payload.get("eval_current_step") if output_log_payload else None,
                "eval_total_steps": output_log_payload.get("eval_total_steps") if output_log_payload else None,
                "export_dir": export_dir,
                "merge_status": merge_status,
                "progress_realtime_confirmed": progress_realtime_confirmed,
                "progress_realtime_reason": progress_realtime_reason,
                "training_start_signal": output_log_payload.get("training_start_signal") if output_log_payload else False,
                "latest_step_or_epoch": latest_step if latest_step is not None else latest_epoch,
                "axis": axis,
                "axis_note": axis_note,
                "history_count": history_count,
                "history_limit": history_limit,
                "history": history,
                "metrics_waiting": waiting_for_metrics,
                "last_update_time": last_update_time,
                "stale": stale,
                "stale_minutes": stale_minutes,
                "finished": status == "finished",
                "succeeded": bool(status == "finished" and not error_reason),
                "success_state": "succeeded" if status == "finished" and not error_reason else None,
                "completion_reason": completion_reason if status == "finished" and not error_reason else None,
                "completion_source": completion_source if status == "finished" and not error_reason else None,
                "completion_info": completion_info,
                "output_log_success": output_log_finished,
                "output_log_success_hint": output_log_success_hint,
                "iteration_finished": iteration_finished,
                "error_reason": error_reason,
                "error_summary": error_summary or None,
                "error_detail": output_log_error if status == "failed" else None,
                "debug": {
                    "selected_cmd": selected_proc.get("cmd") if selected_proc else None,
                    "process_count": len(processes),
                    "wandb_history_path": wandb_data.get("history_path"),
                    "wandb_summary_path": wandb_data.get("summary_path"),
                    "wandb_metadata_path": wandb_data.get("metadata_path"),
                    "wandb_raw_history_count": wandb_data.get("raw_history_count"),
                    "wandb_exitcode": wandb_data.get("exitcode"),
                    "wandb_state": wandb_data.get("state"),
                    "wandb_loss_key": wandb_data.get("loss_key"),
                    "wandb_step_key": wandb_data.get("step_key"),
                    "wandb_epoch_key": wandb_data.get("epoch_key"),
                    "wandb_history_truncated": wandb_data.get("history_truncated"),
                    "trainer_log_path": trainer_log_path,
                    "trainer_log_latest": trainer_log_data.get("latest") if trainer_log_data else None,
                    "trainer_log_error": trainer_log_data.get("error") if trainer_log_data else None,
                    "output_log_path": effective_output_log_data.get("path") if effective_output_log_data else None,
                    "output_log_latest": effective_output_log_data.get("latest") if effective_output_log_data else None,
                    "output_log_error": output_log_error,
                    "output_log_success": output_log_finished,
                    "output_log_success_hint": output_log_success_hint,
                    "output_log_progress": {
                        "progress_percent": progress_percent,
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "elapsed_time": elapsed_time,
                        "remaining_time": remaining_time,
                    },
                    "dpo_launch_state": dpo_launch_state,
                    "output_log_progress_line": output_log_payload.get("progress_line")
                    if output_log_payload
                    else None,
                    "output_log_progress_realtime_confirmed": (
                        output_log_payload.get("progress_realtime_confirmed")
                        if output_log_payload
                        else None
                    ),
                    "output_log_progress_realtime_reason": (
                        output_log_payload.get("progress_realtime_reason")
                        if output_log_payload
                        else None
                    ),
                    "output_log_progress_candidate_count": (
                        output_log_payload.get("progress_candidate_count")
                        if output_log_payload
                        else None
                    ),
                    "output_log_progress_unique_step_count": (
                        output_log_payload.get("progress_unique_step_count")
                        if output_log_payload
                        else None
                    ),
                    "wandb_url_repr": repr(wandb_url) if wandb_url else None,
                    "wandb_url_run_id_debug": _extract_run_id_from_wandb_url(wandb_url)
                    if wandb_url
                    else None,
                    "monitor_code_marker": "progress_debug_v2",
                    "llm_called": llm_called,
                    "llm_allowed": allow_llm,
                    "llm_error": llm_err,
                    "min_history_for_llm": DEFAULT_MIN_HISTORY_FOR_LLM,
                    "raw_latest_loss": raw_latest_loss,
                    "loss_parse_note": loss_parse_note,
                    "loss_warning": loss_warning,
                },
            },
            "analysis_text": analysis_text,
        }
        _remember_wandb_run(
            container_name,
            pid,
            train_type_value,
            wandb_run_dir,
            wandb_root,
            wandb_root_source,
            wandb_url_run_id,
        )

        # 缓存最近一次成功指标，避免偶发异常导致用户无结果可查
        if container_name:
            _LAST_OK_BY_CONTAINER[container_name] = {
                "payload": json.loads(json.dumps(payload, ensure_ascii=False)),
                "cached_at": datetime.now().isoformat(),
            }
        return _monitor_response(payload)

    except Exception as exc:
        err_text = str(exc)
        tb = traceback.format_exc()
        # 若存在上次成功结果，优先回退，避免用户卡在错误状态
        if container_name and container_name in _LAST_OK_BY_CONTAINER:
            cached = _LAST_OK_BY_CONTAINER[container_name]
            payload = cached.get("payload", {}) or {}
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            debug = metrics.get("debug", {}) if isinstance(metrics, dict) else {}
            debug.update(
                {
                    "internal_error": err_text,
                    "traceback": tb,
                    "cached_at": cached.get("cached_at"),
                    "fallback_used": True,
                }
            )
            if isinstance(metrics, dict):
                metrics["debug"] = debug
                metrics["error_reason"] = "internal_error_suppressed"
            payload["metrics"] = metrics
            payload["analysis_text"] = (
                f"监控内部出现异常，已返回上次成功指标（{cached.get('cached_at')}）。"
            )
            return _monitor_response(payload)

        payload = {
            "status": "unknown",
            "metrics": {
                "container_name": container_name,
                "error_reason": "internal_error_suppressed"
                if "exceptions must derive from BaseException" in err_text
                else err_text,
                "debug": {
                    "internal_error": err_text,
                    "traceback": tb,
                    "fallback_used": False,
                },
            },
            "analysis_text": "训练监控发生异常，请稍后重试。",
        }
        return _monitor_response(payload)
'''
def run_script_by_name_monitor1(
    script_query: str,
    additional_args: Dict[str, Any] = None,
    container_name: Optional[str] = None,
    train_type: Optional[str] = None,
    session_id: Optional[str] = None,
    pid: Optional[str] = None,
    list_only: bool = False,
    background: bool = None,
    env_vars: Dict[str, str] = None,
    use_docker: bool = None,
    **kwargs
) -> ToolResponse:
    """
    监控智能体入口：支持“训练状态/监控训练/训练怎么样了”等指令
    """
    _ = list_only
    _ = background
    _ = use_docker

    def _clean_name(v: Optional[str]) -> Optional[str]:
        if not v or not isinstance(v, str):
            return None
        s = v.strip()
        s = s.rstrip(",，;；。.")
        return s or None

    if not container_name and kwargs:
        container_name = (
            kwargs.get("container_name")
            or kwargs.get("container")
            or kwargs.get("容器")
            or kwargs.get("容器名称")
            or kwargs.get("容器名")
            or kwargs.get("容器ID")
        )

    if not train_type and kwargs:
        train_type = kwargs.get("train_type") or kwargs.get("训练类型")
    if not session_id and kwargs:
        session_id = kwargs.get("session_id")
    if not pid and kwargs:
        pid = kwargs.get("pid") or kwargs.get("process_id") or kwargs.get("进程号") or kwargs.get("进程ID")
    wandb_root = None
    if kwargs:
        wandb_root = kwargs.get("wandb_root") or kwargs.get("wandb路径") or kwargs.get("wandb目录")

    history_limit = None
    time_window_minutes = None
    if kwargs:
        history_limit = kwargs.get("history_limit")
        time_window_minutes = kwargs.get("time_window_minutes")

    if additional_args:
        container_name = _clean_name(
            additional_args.get("container_name")
            or additional_args.get("container")
            or additional_args.get("容器")
            or additional_args.get("容器名称")
            or additional_args.get("容器名")
            or additional_args.get("容器ID")
            or container_name
        )
        train_type = additional_args.get("train_type") or additional_args.get("训练类型") or train_type
        session_id = additional_args.get("session_id") or session_id
        pid = additional_args.get("pid") or additional_args.get("process_id") or additional_args.get("进程号") or pid
        wandb_root = (
            additional_args.get("wandb_root")
            or additional_args.get("wandb路径")
            or additional_args.get("wandb目录")
            or wandb_root
        )
        history_limit = additional_args.get("history_limit", history_limit)
        time_window_minutes = additional_args.get("time_window_minutes", time_window_minutes)

    if not container_name and env_vars:
        container_name = env_vars.get("container") or env_vars.get("container_name")

    # Collect potential launch params early. This is used to detect
    # accidental "start GRPO training" requests routed to monitor.
    launch_model_path = (
        _pick_value(additional_args, ["model_path", "actor_rollout_ref.model.path"])
        or _pick_value(kwargs, ["model_path", "actor_rollout_ref.model.path"])
    )
    launch_train_files = (
        _pick_value(additional_args, ["train_files", "data.train_files"])
        or _pick_value(kwargs, ["train_files", "data.train_files"])
    )
    launch_val_files = (
        _pick_value(additional_args, ["val_files", "data.val_files"])
        or _pick_value(kwargs, ["val_files", "data.val_files"])
    )
    launch_gpu = (
        _pick_value(additional_args, ["gpu", "CUDA_VISIBLE_DEVICES", "gpus"])
        or _pick_value(kwargs, ["gpu", "CUDA_VISIBLE_DEVICES", "gpus"])
    )

    text_sources: List[str] = []
    if isinstance(script_query, str) and script_query.strip():
        text_sources.append(script_query)
    td = kwargs.get("task_description")
    if isinstance(td, str) and td.strip():
        text_sources.append(td)

    def _extract_first(pattern: str) -> Optional[str]:
        for text in text_sources:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return (m.group(1) or "").strip()
        return None

    def _looks_like_start_grpo_request() -> bool:
        merged = "\n".join(text_sources)
        if not merged:
            return False
        lower = merged.lower()
        has_grpo = "grpo" in lower
        has_start = (
            ("开始训练" in merged)
            or ("启动训练" in merged)
            or ("进行grpo训练" in merged)
            or ("开始grpo训练" in merged)
            or ("拉起训练" in merged)
            or ("start training" in lower)
        )
        ask_monitor_only = (
            "监控" in merged
            or "状态" in merged
            or "进度" in merged
            or "loss" in lower
            or "学习率" in merged
        )
        return has_grpo and has_start and not ask_monitor_only

    # If a "start GRPO training" request mistakenly routes to monitor tool,
    # hand it over to training tool directly to avoid startup failures.
    if _looks_like_start_grpo_request():
        start_args: Dict[str, Any] = dict(additional_args or {})

        parsed_container = _extract_first(
            r"(?:容器名称|容器名|容器|container(?:_name)?)\s*(?:是|为|=|:)?\s*([\w.-]+)"
        )
        parsed_model_path = _extract_first(
            r"(?:模型路径|模型位置|model_path|actor_rollout_ref\.model\.path)\s*(?:是|为|=|:)?\s*([^\s,，;；]+)"
        )
        parsed_train_files = _extract_first(
            r"(?:train_files|data\.train_files|训练文件|训练数据|训练集)\s*(?:是|为|=|:)?\s*([^\s,，;；]+)"
        )
        parsed_val_files = _extract_first(
            r"(?:val_files|data\.val_files|验证文件|验证数据|验证集)\s*(?:是|为|=|:)?\s*([^\s,，;；]+)"
        )
        parsed_gpu = _extract_first(
            r"(?:显卡|gpu|gpus|cuda_visible_devices)\s*(?:是|为|=|:)?\s*([0-9,\s]+)"
        )

        if parsed_container and "container" not in start_args and "container_name" not in start_args:
            start_args["container"] = parsed_container
        if parsed_model_path and "model_path" not in start_args:
            start_args["model_path"] = parsed_model_path
        if parsed_train_files and "train_files" not in start_args:
            start_args["train_files"] = parsed_train_files
        if parsed_val_files and "val_files" not in start_args:
            start_args["val_files"] = parsed_val_files
        if parsed_gpu and "gpu" not in start_args and "CUDA_VISIBLE_DEVICES" not in start_args:
            start_args["gpu"] = parsed_gpu.replace(" ", "")

        from .runlocal_train import run_script_by_name_train

        return run_script_by_name_train(
            script_query="grpo训练",
            additional_args=start_args,
            env_vars=env_vars,
            use_docker=use_docker,
        )

    if not container_name:
        for text in text_sources:
            m = re.search(r"(?:容器名称|容器名|容器|container)\s*(?:是|为|:|=)?\s*([\w.-]+)", text)
            if m:
                container_name = m.group(1)
                break
    if not pid:
        for text in text_sources:
            m = re.search(r"(?:pid|进程号|进程id)\s*(?:是|为|:|=)?\s*(\d+)", text, re.IGNORECASE)
            if m:
                pid = m.group(1)
                break
    if not wandb_root:
        for text in text_sources:
            m = re.search(r"(?:wandb_root|wandb路径|wandb目录)\s*(?:是|为|:|=)?\s*([/\w\.-]+)", text, re.IGNORECASE)
            if m:
                wandb_root = m.group(1)
                break

    container_name = _clean_name(container_name)
    wandb_root = _clean_name(wandb_root)

    return monitor_training(
        container_name=container_name,
        train_type=train_type,
        session_id=session_id,
        pid=pid,
        wandb_root=wandb_root,
        history_limit=(history_limit or DEFAULT_HISTORY_LIMIT),
        time_window_minutes=(time_window_minutes or DEFAULT_TIME_WINDOW_MINUTES),
    )
'''


def run_script_by_name_monitor1(
    script_query: str,
    additional_args: Dict[str, Any] = None,
    container_name: Optional[str] = None,
    train_type: Optional[str] = None,
    session_id: Optional[str] = None,
    pid: Optional[str] = None,
    list_only: bool = False,
    background: bool = None,
    env_vars: Dict[str, str] = None,
    use_docker: bool = None,
    **kwargs
) -> ToolResponse:
    """Monitor tool entry. Also guards accidental start-training intents."""
    _ = list_only
    _ = background
    _ = use_docker

    def _clean_name(v: Optional[str]) -> Optional[str]:
        if not v or not isinstance(v, str):
            return None
        s = v.strip()
        s = s.rstrip(",;")
        return s or None

    def _pick_value(src: Optional[Dict[str, Any]], keys: List[str]) -> Optional[Any]:
        if not src:
            return None
        for key in keys:
            if key in src and src.get(key) not in (None, ""):
                return src.get(key)
        return None

    def _as_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "y"}:
                return True
            if normalized in {"0", "false", "no", "off", "n"}:
                return False
        return default

    if not container_name:
        container_name = _pick_value(kwargs, ["container_name", "container"])
    if not train_type:
        train_type = _pick_value(kwargs, ["train_type"])
    if not session_id:
        session_id = _pick_value(kwargs, ["session_id"])
    if not pid:
        pid = _pick_value(kwargs, ["pid", "process_id"])

    wandb_root = _pick_value(kwargs, ["wandb_root"])
    history_limit = _pick_value(kwargs, ["history_limit"])
    time_window_minutes = _pick_value(kwargs, ["time_window_minutes"])
    allow_llm = _as_bool(_pick_value(kwargs, ["allow_llm", "use_llm"]), True)

    if additional_args:
        container_name = _clean_name(
            _pick_value(additional_args, ["container_name", "container"]) or container_name
        )
        train_type = _pick_value(additional_args, ["train_type"]) or train_type
        session_id = _pick_value(additional_args, ["session_id"]) or session_id
        pid = _pick_value(additional_args, ["pid", "process_id"]) or pid
        wandb_root = _pick_value(additional_args, ["wandb_root"]) or wandb_root
        history_limit = _pick_value(additional_args, ["history_limit"]) or history_limit
        time_window_minutes = (
            _pick_value(additional_args, ["time_window_minutes"]) or time_window_minutes
        )
        allow_llm = _as_bool(
            _pick_value(additional_args, ["allow_llm", "use_llm"]),
            allow_llm,
        )

    if not container_name and env_vars:
        container_name = env_vars.get("container") or env_vars.get("container_name")

    # Collect potential launch params early. This is used to detect
    # accidental "start GRPO training" requests routed to monitor.
    launch_model_path = (
        _pick_value(additional_args, ["model_path", "actor_rollout_ref.model.path"])
        or _pick_value(kwargs, ["model_path", "actor_rollout_ref.model.path"])
    )
    launch_train_files = (
        _pick_value(additional_args, ["train_files", "data.train_files"])
        or _pick_value(kwargs, ["train_files", "data.train_files"])
    )
    launch_val_files = (
        _pick_value(additional_args, ["val_files", "data.val_files"])
        or _pick_value(kwargs, ["val_files", "data.val_files"])
    )
    launch_gpu = (
        _pick_value(additional_args, ["gpu", "CUDA_VISIBLE_DEVICES", "gpus"])
        or _pick_value(kwargs, ["gpu", "CUDA_VISIBLE_DEVICES", "gpus"])
    )

    text_sources: List[str] = []
    if isinstance(script_query, str) and script_query.strip():
        text_sources.append(script_query)
    td = kwargs.get("task_description")
    if isinstance(td, str) and td.strip():
        text_sources.append(td)

    def _extract_first(pattern: str) -> Optional[str]:
        for text in text_sources:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return (m.group(1) or "").strip()
        return None

    def _looks_like_start_grpo_request() -> bool:
        merged = "\n".join(text_sources)
        if not merged:
            merged = ""
        lower = merged.lower()
        has_grpo = ("grpo" in lower) or (str(train_type or "").lower() == "grpo")
        has_start = (
            ("\u5f00\u59cb\u8bad\u7ec3" in merged)
            or ("\u542f\u52a8\u8bad\u7ec3" in merged)
            or ("\u8fdb\u884cgrpo\u8bad\u7ec3" in lower)
            or ("\u5f00\u59cbgrpo\u8bad\u7ec3" in lower)
            or ("\u62c9\u8d77\u8bad\u7ec3" in merged)
            or ("start training" in lower)
        )
        has_launch_params = bool(
            launch_model_path
            or launch_train_files
            or launch_val_files
            or launch_gpu
            or _extract_first(
                r"(?:model_path|actor_rollout_ref\.model\.path|data\.train_files|data\.val_files)\s*(?:=|:|\u662f|\u4e3a)\s*([^\s,;\uFF0C\uFF1B]+)"
            )
        )
        ask_monitor_only = (
            ("\u76d1\u63a7" in merged)
            or ("\u72b6\u6001" in merged)
            or ("\u8fdb\u5ea6" in merged)
            or ("loss" in lower)
            or ("\u5b66\u4e60\u7387" in merged)
        )
        # Start intent can be explicit ("start training") or implicit
        # (GRPO + launch params provided in same request).
        return has_grpo and ((has_start and not ask_monitor_only) or has_launch_params)

    if _looks_like_start_grpo_request():
        start_args: Dict[str, Any] = dict(additional_args or {})
        # Also absorb top-level kwargs when monitor agent passes tool args directly.
        for key in [
            "container",
            "container_name",
            "model_path",
            "train_files",
            "val_files",
            "gpu",
            "CUDA_VISIBLE_DEVICES",
            "train_type",
        ]:
            val = kwargs.get(key)
            if val not in (None, "") and key not in start_args:
                start_args[key] = val
        parsed_container = _extract_first(
            r"(?:\u5bb9\u5668\u540d\u79f0|\u5bb9\u5668\u540d|\u5bb9\u5668|container(?:_name)?)\s*(?:\u662f|\u4e3a|=|:)?\s*([\w.-]+)"
        )
        parsed_model_path = _extract_first(
            r"(?:\u6a21\u578b\u8def\u5f84|\u6a21\u578b\u4f4d\u7f6e|model_path|actor_rollout_ref\.model\.path)\s*(?:\u662f|\u4e3a|=|:)?\s*([^\s,;\uFF0C\uFF1B]+)"
        )
        parsed_train_files = _extract_first(
            r"(?:train_files|data\.train_files|\u8bad\u7ec3\u6587\u4ef6|\u8bad\u7ec3\u6570\u636e|\u8bad\u7ec3\u96c6)\s*(?:\u662f|\u4e3a|=|:)?\s*([^\s,;\uFF0C\uFF1B]+)"
        )
        parsed_val_files = _extract_first(
            r"(?:val_files|data\.val_files|\u9a8c\u8bc1\u6587\u4ef6|\u9a8c\u8bc1\u6570\u636e|\u9a8c\u8bc1\u96c6)\s*(?:\u662f|\u4e3a|=|:)?\s*([^\s,;\uFF0C\uFF1B]+)"
        )
        parsed_gpu = _extract_first(
            r"(?:\u663e\u5361|gpu|gpus|cuda_visible_devices)\s*(?:\u662f|\u4e3a|=|:)?\s*([0-9,\s]+)"
        )

        if parsed_container and "container" not in start_args and "container_name" not in start_args:
            start_args["container"] = parsed_container
        if parsed_model_path and "model_path" not in start_args:
            start_args["model_path"] = parsed_model_path
        if parsed_train_files and "train_files" not in start_args:
            start_args["train_files"] = parsed_train_files
        if parsed_val_files and "val_files" not in start_args:
            start_args["val_files"] = parsed_val_files
        if parsed_gpu and "gpu" not in start_args and "CUDA_VISIBLE_DEVICES" not in start_args:
            start_args["gpu"] = parsed_gpu.replace(" ", "")

        from .runlocal_train import run_script_by_name_train

        return run_script_by_name_train(
            script_query="grpo",
            additional_args=start_args,
            env_vars=env_vars,
            use_docker=use_docker,
        )

    if not container_name:
        parsed_container = _extract_first(
            r"(?:\u5bb9\u5668\u540d\u79f0|\u5bb9\u5668\u540d|\u5bb9\u5668|container)\s*(?:\u662f|\u4e3a|=|:)?\s*([\w.-]+)"
        )
        if parsed_container:
            container_name = parsed_container
    if not pid:
        parsed_pid = _extract_first(
            r"(?:pid|process_id|\u8fdb\u7a0b\u53f7|\u8fdb\u7a0bid)\s*(?:\u662f|\u4e3a|=|:)?\s*(\d+)"
        )
        if parsed_pid:
            pid = parsed_pid
    if not wandb_root:
        parsed_wandb_root = _extract_first(
            r"(?:wandb_root|wandb_path|wandb_dir)\s*(?:\u662f|\u4e3a|=|:)?\s*([/\w\.-]+)"
        )
        if parsed_wandb_root:
            wandb_root = parsed_wandb_root

    container_name = _clean_name(container_name)
    wandb_root = _clean_name(wandb_root)

    return monitor_training(
        container_name=container_name,
        train_type=train_type,
        session_id=session_id,
        pid=pid,
        wandb_root=wandb_root,
        history_limit=(history_limit or DEFAULT_HISTORY_LIMIT),
        time_window_minutes=(time_window_minutes or DEFAULT_TIME_WINDOW_MINUTES),
        allow_llm=allow_llm,
    )


__all__ = [
    "run_script_by_name_monitor1",
    "monitor_training",
]



