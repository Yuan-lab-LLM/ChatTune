# -*- coding: utf-8 -*-
"""
模型评估监控工具。

用于监控 compare_between_models_vpn/single_model_evaluation_vpn 等评估任务的
容器内 PID、阶段进度和阶段输出结果。
"""

import json
import os
import re
import shlex
import subprocess
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ._config_defaults import get_default_docker_container
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

DEFAULT_EVAL_ROOT = "/home/workspace/eval/daily_train"
DEFAULT_LOG_ROOT = "/home/workspace/log/daily_train"
DEFAULT_CONTAINER = get_default_docker_container()

_RUNTIME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime")
)
_EVALUATE_PID_REGISTRY = os.path.join(_RUNTIME_DIR, "evaluate_pid_registry.jsonl")
_BACKGROUND_TASK_REGISTRY = os.path.join(_RUNTIME_DIR, "background_task_registry.jsonl")
_AUTO_TAGGING_RECOVERY_REGISTRY = os.path.join(_RUNTIME_DIR, "evaluate_auto_tagging_recovery.jsonl")
MAX_LOG_TAIL_CHARS = 1200
MAX_LOG_LINE_CHARS = 500


def _truncate_text(text: Optional[str], limit: int = MAX_LOG_TAIL_CHARS) -> Optional[str]:
    if not text:
        return None
    value = str(text)
    if len(value) <= limit:
        return value
    return f"...[truncated {len(value) - limit} chars]...\n{value[-limit:]}"


def _monitor_protocol_hint(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    status = str(payload.get("status") or "unknown")
    error_reason = str(metrics.get("error_reason") or "")
    required_params: List[str] = []
    if error_reason in {"pid_required", "pid_not_found"}:
        required_params.append("pid")
    if error_reason in {"container_required", "container_unknown"}:
        required_params.append("container_name")
    if error_reason in {"model_names_required"}:
        required_params.extend(metrics.get("required_params") or ["model_fir", "model_sec"])

    if required_params:
        return {
            "type": "need_input",
            "agent": "assessment_monitor",
            "kind": "assessment_monitor_params",
            "message": payload.get("analysis_text") or "请补充评估监控所需参数。",
            "requiredParams": required_params,
            "status": status,
            "container": metrics.get("container_name"),
            "pid": metrics.get("pid"),
            "errorReason": error_reason or None,
        }

    return {
        "type": "monitor_status",
        "agent": "assessment_monitor",
        "message": payload.get("analysis_text") or "",
        "jobType": "assessment",
        "status": status,
        "container": metrics.get("container_name"),
        "pid": metrics.get("pid"),
        "pidAlive": metrics.get("pid_alive"),
        "currentStage": metrics.get("current_stage"),
        "currentLogicalStage": metrics.get("current_stage"),
        "currentStages": metrics.get("current_stages"),
        "currentStageDetails": metrics.get("current_stage_details"),
        "nextStage": metrics.get("next_stage"),
        "runningStages": metrics.get("running_stages"),
        "completedStages": metrics.get("completed_stages"),
        "totalStages": metrics.get("total_stages"),
        "overallProgressPercent": metrics.get("overall_progress_percent"),
        "progress": {
            "percent": metrics.get("overall_progress_percent"),
            "completedStages": metrics.get("completed_stages"),
            "totalStages": metrics.get("total_stages"),
        },
        "stageStatusCounts": metrics.get("stage_status_counts"),
        "stages": metrics.get("stages"),
        "artifacts": metrics.get("artifacts"),
        "script": metrics.get("script_name"),
        "scores": metrics.get("scores"),
        "reportPath": metrics.get("report_path"),
        "reportText": metrics.get("report_text"),
        "errorReason": error_reason or None,
    }


def _monitor_response(payload: Dict[str, Any]) -> ToolResponse:
    display_text = _monitor_display_text(payload)
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
        metadata={
            "success": payload.get("status") not in {"failed", "unknown"},
            "protocol_hint": _monitor_protocol_hint(payload),
            "response_text": display_text,
        },
    )


def _display_value(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(str(item) for item in value) if value else "无"
    return "无" if value in (None, "", [], {}) else str(value)


def _monitor_display_text(payload: Dict[str, Any]) -> str:
    """Build a deterministic user-facing assessment status summary."""
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    status_map = {
        "starting": "启动中",
        "running": "运行中",
        "finished": "已完成",
        "interrupted": "已中断",
        "failed": "失败",
        "unknown": "未知",
    }
    status = str(payload.get("status") or "unknown")
    counts = metrics.get("stage_status_counts") or {}
    current_details = metrics.get("current_stage_details") or []
    artifacts = [
        item
        for item in (metrics.get("artifacts") or [])
        if isinstance(item, dict) and item.get("exists")
    ]
    lines = [
        "当前评估任务状态如下：",
        "",
        f"- **任务状态**：{status_map.get(status, status)}",
        f"- **评估脚本**：{_display_value(metrics.get('script_name'))}",
        f"- **容器名称**：{_display_value(metrics.get('container_name'))}",
        f"- **PID**：{_display_value(metrics.get('pid'))}，PID是否存活：{'是' if metrics.get('pid_alive') else '否'}",
        (
            f"- **整体进度**：已完成阶段数 {metrics.get('completed_stages', 0)}/"
            f"{metrics.get('total_stages', 0)}，整体百分比 {metrics.get('overall_progress_percent', 0.0)}%"
        ),
        (
            f"  - finished: {counts.get('finished', 0)}，"
            f"running: {counts.get('running', 0)}，pending: {counts.get('pending', 0)}"
        ),
        f"- **当前逻辑阶段**：{_display_value(metrics.get('current_stage'))}",
        f"- **当前实际运行子阶段**：{_display_value(metrics.get('current_stages'))}",
    ]
    parallel_group = metrics.get("parallel_stage_group") or {}
    if isinstance(parallel_group, dict) and parallel_group.get("summary"):
        lines.append(f"- **并行阶段状态**：{parallel_group['summary']}")
    lines.append(f"- **下一阶段**：{_display_value(metrics.get('next_stage'))}")
    for detail in current_details:
        if not isinstance(detail, dict) or detail.get("status") != "running":
            continue
        summary = detail.get("progress_summary") or {}
        lines.extend(
            [
                f"- **运行子阶段 `{detail.get('name')}` 日志路径**：{_display_value(detail.get('progress_log'))}",
                f"  - 日志更新时间：{_display_value((detail.get('progress_log_meta') or {}).get('updated_at'))}",
            ],
        )
        if summary.get("step_name"):
            lines.append(f"  - 当前日志步骤：{summary['step_name']}")
        if summary.get("step_current") is not None and summary.get("step_total") is not None:
            lines.append(
                f"  - 步骤进度：{summary['step_current']}/{summary['step_total']}"
                f"（约 {summary.get('local_progress_percent')}%）",
            )
        elif summary.get("local_progress_percent") is not None:
            lines.append(f"  - 当前日志步骤进度：约 {summary['local_progress_percent']}%")
        if summary.get("latest_line"):
            lines.append(f"  - 最新日志行：{summary['latest_line']}")
        if summary.get("error_detected"):
            lines.append(f"  - **检测到错误线索**：{summary.get('latest_error_line')}")
        else:
            lines.append("  - 错误检测：未检测到错误")
        if summary.get("local_progress_percent") is not None:
            lines.append("  - 说明：上述进度仅表示当前日志步骤，不代表当前评估阶段或整体评估进度。")
    if metrics.get("scores"):
        lines.append(f"- **已有评分**：{json.dumps(metrics['scores'], ensure_ascii=False)}")
    if metrics.get("report_path"):
        lines.append(f"- **最终报告路径**：{metrics['report_path']}")
    if metrics.get("report_text"):
        lines.append("- **最终报告内容**：")
        lines.append("```")
        lines.extend(str(metrics["report_text"]).splitlines())
        lines.append("```")
    if metrics.get("error_log"):
        lines.append(f"- **错误日志**：{metrics['error_log']}")
    if metrics.get("error_tail"):
        lines.append(f"- **错误日志尾部**：{metrics['error_tail']}")
    recovery = metrics.get("auto_tagging_recovery") or {}
    if isinstance(recovery, dict) and recovery.get("message"):
        lines.append(f"- **自动恢复**：{recovery['message']}")
        review_rows = _auto_tagging_review_rows(recovery)
        if review_rows:
            lines.append(f"  - 需人工复核行：{review_rows}")
        if recovery.get("new_pid"):
            lines.append(f"  - 续跑 PID：{recovery['new_pid']}")
    if artifacts:
        lines.append("- **已生成的结果文件**：")
        lines.extend(f"  - {item.get('name')}: {item.get('path')}" for item in artifacts)
    missing = []
    if not metrics.get("scores"):
        missing.append("评分")
    if not metrics.get("report_path"):
        missing.append("最终报告")
    if not artifacts:
        missing.append("结果文件")
    if missing:
        lines.append(f"- **暂未生成**：{'、'.join(missing)}")
    return "\n".join(lines)


def _run_cmd(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return process.returncode, (process.stdout or "").strip(), (process.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, stdout.strip(), (stderr.strip() or f"command_timeout_after_{timeout}s")
    except Exception as exc:
        return 1, "", f"command_exec_error: {exc}"


def _docker_sh(container: str, script: str, timeout: int = 15) -> Tuple[int, str, str]:
    return _run_cmd(["docker", "exec", container, "sh", "-lc", script], timeout=timeout)


def _path_exists(container: str, path: str) -> bool:
    code, _, _ = _docker_sh(container, f"test -e {shlex.quote(path)}", timeout=8)
    return code == 0


def _path_stat(container: str, path: str) -> Optional[Dict[str, Any]]:
    code, out, _ = _docker_sh(
        container,
        f"stat -c '%s|%Y' {shlex.quote(path)}",
        timeout=8,
    )
    if code != 0 or not out:
        return None
    try:
        size_text, mtime_text = out.split("|", 1)
        mtime = float(mtime_text)
        return {
            "size_bytes": int(size_text),
            "updated_at": datetime.fromtimestamp(mtime).isoformat(),
            "updated_at_ts": mtime,
        }
    except Exception:
        return None


def _read_tail(container: str, path: str, lines: int = 120) -> str:
    code, out, _ = _docker_sh(container, f"tail -n {int(lines)} {shlex.quote(path)}", timeout=15)
    return out if code == 0 else ""


_LOG_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)


def _record_start_ts(record: Optional[Dict[str, Any]]) -> Optional[float]:
    if not record:
        return None
    started_at = _record_sort_ts(record)
    return started_at if started_at > 0 else None


def _log_line_timestamp(line: str) -> Optional[float]:
    match = _LOG_TIMESTAMP_RE.match(line or "")
    if not match:
        return None
    try:
        return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}").timestamp()
    except Exception:
        return None


def _filter_log_tail_since(text: str, started_at_ts: Optional[float]) -> str:
    if not text or started_at_ts is None:
        return text or ""
    selected: List[str] = []
    include_current_block = False
    for line in text.splitlines():
        line_ts = _log_line_timestamp(line)
        if line_ts is not None:
            include_current_block = line_ts >= started_at_ts
        if include_current_block:
            selected.append(line)
    return "\n".join(selected).strip()


def _read_error_tail_since(
    container: str,
    path: str,
    started_at_ts: Optional[float],
    lines: int = 80,
) -> str:
    tail = _read_tail(container, path, lines=max(int(lines) * 5, int(lines)))
    filtered = _filter_log_tail_since(tail, started_at_ts)
    if not filtered:
        return ""
    return "\n".join(filtered.splitlines()[-int(lines):])

def _read_full_text(
    container: str,
    path: Optional[str],
    max_lines: int = 200,
    max_chars: int = 16384,
) -> Optional[str]:
    """从容器内读取完整文本内容，带行数与长度上限保护。"""
    if not container or not path:
        return None
    code, out, _ = _docker_sh(
        container,
        f"head -n {int(max_lines)} {shlex.quote(path)}",
        timeout=15,
    )
    if code != 0:
        return None
    text = out.rstrip("\n")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...（报告内容已截断，完整内容请见报告路径）"
    return text if text.strip() else None


def _single_eval_dates_for_model(container: str, model_name: str) -> List[str]:
    """Return candidate single-model evaluation date keys for a model name."""
    if not model_name:
        return []
    patterns = [
        f"*_{model_name}_common_eval_out",
        f"*_{model_name}_medical_eval_out",
    ]
    find_expr = " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in patterns)
    code, out, _ = _docker_sh(
        container,
        (
            f"find {shlex.quote(DEFAULT_EVAL_ROOT)} -maxdepth 1 -type d "
            f"\\( {find_expr} \\) -printf '%T@|%f\\n'"
        ),
        timeout=20,
    )
    if code != 0 or not out:
        return []

    candidates: Dict[str, float] = {}
    pattern = re.compile(
        rf"^(?P<date>.+)_{re.escape(model_name)}_(?:common|medical)_eval_out$",
    )
    for line in out.splitlines():
        try:
            mtime_text, basename = line.split("|", 1)
            mtime = float(mtime_text)
        except ValueError:
            continue
        match = pattern.match(basename.strip())
        if match:
            date_key = match.group("date")
            candidates[date_key] = max(candidates.get(date_key, 0.0), mtime)

    def sort_key(date_key: str) -> Tuple[int, float]:
        report_path = f"{DEFAULT_LOG_ROOT}/{date_key}_single_model_valuation_evaluation_report.txt"
        report_meta = _path_stat(container, report_path)
        if report_meta:
            return (1, float(report_meta.get("updated_at_ts") or 0.0))
        return (0, candidates.get(date_key, 0.0))

    return sorted(candidates, key=sort_key, reverse=True)


def _resolve_single_eval_date(container: str, model_name: str, date_str: Optional[str]) -> str:
    requested = str(date_str or "").strip()
    if requested:
        return requested
    candidates = _single_eval_dates_for_model(container, model_name)
    return candidates[0] if candidates else datetime.now().strftime("%Y%m%d")


def _check_pid_alive(container: str, pid: str) -> bool:
    if not container or not pid:
        return False
    code, out, _ = _docker_sh(container, f"ps -p {shlex.quote(str(pid))} -o pid=", timeout=8)
    return code == 0 and bool(out.strip())


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _load_eval_records() -> List[Dict[str, Any]]:
    records = _load_jsonl(_EVALUATE_PID_REGISTRY)
    for item in _load_jsonl(_BACKGROUND_TASK_REGISTRY):
        if item.get("task_type") in {"assessment", "evaluate"}:
            records.append(item)
    records.sort(key=_record_sort_ts)
    return records


def _record_sort_ts(record: Dict[str, Any]) -> float:
    raw = record.get("started_at_ts") or record.get("started_at") or 0
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(raw)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(str(raw)).timestamp()
    except Exception:
        return 0.0


def _latest_record(pid: Optional[str], container: Optional[str]) -> Optional[Dict[str, Any]]:
    records = _load_eval_records()
    if pid:
        matched = [r for r in records if str(r.get("pid") or "") == str(pid)]
        if container:
            matched = [r for r in matched if r.get("container") == container]
        return matched[-1] if matched else None
    if container:
        records = [r for r in records if r.get("container") == container]
    return records[-1] if records else None


def _basename(path_or_name: Optional[str]) -> Optional[str]:
    if not path_or_name:
        return None
    return os.path.basename(str(path_or_name).rstrip("/")) or str(path_or_name)


def _get_arg_from_command(command: str, name: str) -> Optional[str]:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except Exception:
        parts = command.split()
    flag = f"--{name}"
    for idx, part in enumerate(parts):
        if part == flag and idx + 1 < len(parts):
            return parts[idx + 1]
        if part.startswith(f"{flag}="):
            return part.split("=", 1)[1]
        if part.startswith(f"-e") and "=" in part:
            env_part = part[2:] if part != "-e" else ""
            if not env_part and idx + 1 < len(parts):
                env_part = parts[idx + 1]
            if env_part.startswith(f"{name}="):
                return env_part.split("=", 1)[1]
    return None


def _script_from_record(record: Optional[Dict[str, Any]], fallback: str) -> str:
    if not record:
        return fallback
    raw = str(record.get("script_name") or record.get("script_path") or record.get("command") or fallback)
    lower = raw.lower()
    if "single_model_evaluation_vpn" in lower:
        return "single_model_evaluation_vpn"
    if "ckpt_eval" in lower:
        return "ckpt_eval"
    if "compare_between_models_vpn" in lower:
        return "compare_between_models_vpn"
    return fallback


def _record_value(record: Optional[Dict[str, Any]], name: str) -> Optional[str]:
    if not record:
        return None
    script_args = record.get("script_args") if isinstance(record.get("script_args"), dict) else {}
    env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
    value = script_args.get(name) or env_vars.get(name)
    if value:
        return str(value)
    return _get_arg_from_command(str(record.get("command") or ""), name)


def _fallback_latest_record(pid: Optional[str], container: Optional[str]) -> Optional[Dict[str, Any]]:
    exact = _latest_record(pid, container)
    if exact:
        return exact
    if not container:
        return None
    records = [r for r in _load_eval_records() if r.get("container") == container]
    return records[-1] if records else None


def _date_from_record(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return datetime.now().strftime("%Y%m%d")
    started_at_raw = record.get("started_at") or record.get("started_at_ts") or ""
    if isinstance(started_at_raw, (int, float)):
        return datetime.fromtimestamp(float(started_at_raw)).strftime("%Y%m%d")
    started_at = str(started_at_raw)
    try:
        return datetime.fromisoformat(started_at).strftime("%Y%m%d")
    except Exception:
        return datetime.now().strftime("%Y%m%d")


def _extract_average(text: str) -> Optional[float]:
    match = re.search(r"Average\s*[:：]\s*([+-]?\d+(?:\.\d+)?)", text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _json_value(text: str, path: List[str]) -> Optional[Any]:
    try:
        data = json.loads(text or "{}")
    except Exception:
        return None
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_subject_scores(text: str) -> Dict[str, float]:
    block_match = re.search(r"subject scores:(.*?)(summary scores:|$)", text or "", re.I | re.S)
    scores_text = block_match.group(1) if block_match else (text or "")
    scores: Dict[str, float] = {}
    for subject, value in re.findall(r"([^\n:：]{2,30})\s*[:：]\s*([+-]?\d+(?:\.\d+)?)", scores_text):
        name = subject.strip(" -\t")
        if not name or re.search(r"(Average|summary|score)", name, re.I):
            continue
        try:
            scores[name] = float(value)
        except Exception:
            continue
    return scores


def _progress_log_summary(text: str) -> Dict[str, Any]:
    lines = [
        _truncate_text(line.strip(), MAX_LOG_LINE_CHARS)
        for line in (text or "").splitlines()
        if line.strip()
    ]
    percent_matches = re.findall(
        r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%",
        text or "",
    )
    local_progress_percent: Optional[float] = None
    if percent_matches:
        try:
            local_progress_percent = min(100.0, float(percent_matches[-1]))
        except Exception:
            local_progress_percent = None
    latest_line = lines[-1] if lines else None
    step_name: Optional[str] = None
    step_current: Optional[int] = None
    step_total: Optional[int] = None
    tqdm_match = re.search(
        r"^\s*([^:]+):\s*\d{1,3}(?:\.\d+)?%\|.*?\|\s*(\d+)\s*/\s*(\d+)",
        latest_line or "",
    )
    if tqdm_match:
        step_name = tqdm_match.group(1).strip()
        step_current = int(tqdm_match.group(2))
        step_total = int(tqdm_match.group(3))
    error_lines = [
        line
        for line in lines
        if re.search(r"(traceback|exception|error|failed|fatal)", line, re.I)
    ]
    return {
        "has_progress_log": bool(text),
        "local_progress_percent": local_progress_percent,
        "step_name": step_name,
        "step_current": step_current,
        "step_total": step_total,
        "latest_line": latest_line,
        "tail_lines": lines[-5:],
        "error_detected": bool(error_lines),
        "latest_error_line": error_lines[-1] if error_lines else None,
        "note": "已检测到阶段运行日志，但日志中的百分比可能只是模型加载、格式化或批次预测等局部进度，不能代表当前阶段或整体评估进度。",
    }


def _stage(
    path: str,
    name: str,
    kind: str,
    container: str,
    progress_logs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    exists = _path_exists(container, path)
    progress_log = next((item for item in (progress_logs or []) if _path_exists(container, item)), None)
    output_meta = _path_stat(container, path) if exists else None
    progress_log_meta = _path_stat(container, progress_log) if progress_log else None
    result: Any = None
    progress_summary: Optional[Dict[str, Any]] = None
    if exists:
        if kind == "binary":
            result = {"exists": True}
        else:
            content = _read_tail(container, path, lines=160)
            if kind == "eval_results":
                result = {
                    "average": _extract_average(content),
                    "subject_scores": _extract_subject_scores(content),
                }
            elif kind == "json":
                result = {
                    "content_tail": _truncate_text(content),
                }
                if "diagnose_accuracy" in name:
                    result["accuracy"] = _json_value(content, ["summary", "accuracy"])
                if "medhalt" in name:
                    result["mean_fuzziness"] = _json_value(content, ["statistics", "mean value of evaluation fuzziness"])
            else:
                result = _truncate_text(content)
    elif progress_log:
        progress_tail = _read_tail(container, progress_log, lines=80)
        progress_summary = _progress_log_summary(progress_tail)
        result = {
            "progress_log": progress_log,
            "progress_summary": progress_summary,
        }
    return {
        "name": name,
        "status": "finished" if exists else ("running" if progress_log else "pending"),
        "path": path,
        "output_meta": output_meta,
        "progress_log": progress_log,
        "progress_log_meta": progress_log_meta,
        "progress_summary": progress_summary,
        "result": result,
    }


def _artifact(path: str, name: str, kind: str, container: str) -> Dict[str, Any]:
    exists = _path_exists(container, path)
    result: Any = None
    if exists:
        if kind == "binary":
            result = {"exists": True}
        else:
            content = _read_tail(container, path, lines=80)
            result = _truncate_text(content)
    return {
        "name": name,
        "path": path,
        "exists": exists,
        "kind": kind,
        "output_meta": _path_stat(container, path) if exists else None,
        "result": result,
    }


def _build_compare_stages(
    container: str,
    date_str: str,
    origin_model_name: str,
    update_model_name: str,
) -> List[Dict[str, Any]]:
    return [
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{update_model_name}_common_eval_out/results.log", "update_common_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{update_model_name}-common-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{update_model_name}-common-eval-log.log"]),
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{origin_model_name}_common_eval_out/results.log", "origin_common_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{origin_model_name}-common-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{origin_model_name}-common-eval-log.log"]),
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{update_model_name}_medical_eval_out/results.log", "update_medical_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{update_model_name}-medical-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{update_model_name}-medical-eval-log.log"]),
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{origin_model_name}_medical_eval_out/results.log", "origin_medical_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{origin_model_name}-medical-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{origin_model_name}-medical-eval-log.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{origin_model_name}-stage_run_origin.log", "origin_stage_run", "text", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-origin_stage_run.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-stage_run_update_model.log", "update_stage_run", "text", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-update_stage_run.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-stage_inconsistent_responses.xlsx", "stage_consistency", "binary", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-stage_consistency.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-diagnosis_run_update_model.json", "update_diagnosis_run", "json", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-update_record_to_diagnose.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{origin_model_name}-diagnosis_run_origin.json", "origin_diagnosis_run", "json", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-origin_record_to_diagnose.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-diagnose_inconsistent_responses.xlsx", "diagnosis_consistency", "binary", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{update_model_name}-diagnosis_consistency.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-tagged-stage_inconsistent_responses.xlsx", "stage_auto_tagging", "binary", container, [f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-stage.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-tagged-diagnose_inconsistent_responses.xlsx", "diagnose_auto_tagging", "binary", container, [f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-diagnose.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/{date_str}_compare_between_models_evaluation_report.txt", "evaluation_report", "text", container),
    ]


def _build_compare_artifacts(container: str, date_str: str) -> List[Dict[str, Any]]:
    return [
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-stage_inconsistent_responses.xlsx", "stage_inconsistent_responses", "binary", container),
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-diagnose_inconsistent_responses.xlsx", "diagnose_inconsistent_responses", "binary", container),
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-tagged-stage_inconsistent_responses.xlsx", "tagged_stage_inconsistent_responses", "binary", container),
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-tagged-diagnose_inconsistent_responses.xlsx", "tagged_diagnose_inconsistent_responses", "binary", container),
        _artifact("/usr/local/insinfersystem/compare_eval.log", "compare_eval_log", "text", container),
    ]


def _build_single_stages(container: str, date_str: str, model_name: str) -> List[Dict[str, Any]]:
    return [
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{model_name}_common_eval_out/results.log", "common_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{model_name}-common-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{model_name}-common-eval-log.log"]),
        _stage(f"{DEFAULT_EVAL_ROOT}/{date_str}_{model_name}_medical_eval_out/results.log", "medical_eval", "eval_results", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}_{model_name}-medical-eval-log.log", f"{DEFAULT_EVAL_ROOT}/{date_str}_{model_name}-medical-eval-log.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{model_name}-diagnosis_run_origin.json", "record_to_diagnose", "json", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{model_name}-record_to_diagnose.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/internal_consistent/{date_str}-{model_name}-medhalt_confidence_eval.json", "thread_medhalt_confidence_eval", "json", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{model_name}-thread_medhalt_confidence_eval.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/{date_str}-{model_name}-diagnose_accuracy_result.json", "diagnose_accuracy", "json", container, [f"{DEFAULT_LOG_ROOT}/internal_log/{date_str}-{model_name}-diagnose_accuracy.log"]),
        _stage(f"{DEFAULT_LOG_ROOT}/{date_str}_single_model_valuation_evaluation_report.txt", "evaluation_report", "text", container),
    ]


def _build_single_artifacts(container: str, date_str: str) -> List[Dict[str, Any]]:
    return [
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-diagnose_inconsistent_responses.xlsx", "diagnose_inconsistent_responses", "binary", container),
        _artifact(f"{DEFAULT_LOG_ROOT}/save_csv/{date_str}-stage_inconsistent_responses.xlsx", "stage_inconsistent_responses", "binary", container),
        _artifact("/usr/local/insinfersystem/single_eval.log", "single_eval_log", "text", container),
    ]



def _latest_base_model_name(container: str) -> Optional[str]:
    script = (
        "base=/home/workspace/models/base; "
        "[ -d \"$base\" ] || exit 0; "
        "find \"$base\" -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort | tail -n 1"
    )
    code, out, _ = _docker_sh(container, script, timeout=8)
    if code != 0:
        return None
    value = (out or "").strip().splitlines()
    return value[-1].strip() if value else None

def _ckpt_output_path(ckpt_path: Optional[str]) -> Optional[str]:
    if not ckpt_path:
        return None
    path = str(ckpt_path).rstrip("/")
    idx = path.rfind("/check")
    if idx < 0:
        return None
    model_path_new = path[:idx]
    end = path[path.rfind("check"):]
    return f"{model_path_new}-{end}_merged"


def _ckpt_output_name(ckpt_path: Optional[str]) -> Optional[str]:
    output_path = _ckpt_output_path(ckpt_path)
    return os.path.basename(output_path) if output_path else None


def _build_ckpt_stages(container: str, date_str: str, model_name: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    eval_root = DEFAULT_EVAL_ROOT
    log_root = DEFAULT_LOG_ROOT
    internal_log_root = f"{log_root}/internal_log"
    save_csv_root = f"{log_root}/save_csv"
    origin_model_name = _latest_base_model_name(container) or "origin_model"
    update_model_name = model_name
    return [
        _stage(output_path or model_name, "merge_checkpoint", "binary", container, [f"{log_root}/{model_name.removesuffix('_merged')}-merge-log.log"]),
        _stage(f"{eval_root}/{date_str}_{origin_model_name}_common_eval_out/results.log", "origin_common_eval", "eval_results", container, [f"{internal_log_root}/{date_str}_{origin_model_name}-common-eval-log.log"]),
        _stage(f"{eval_root}/{date_str}_{update_model_name}_common_eval_out/results.log", "update_common_eval", "eval_results", container, [f"{internal_log_root}/{date_str}_{update_model_name}-common-eval-log.log"]),
        _stage(f"{eval_root}/{date_str}_{origin_model_name}_medical_eval_out/results.log", "origin_medical_eval", "eval_results", container, [f"{internal_log_root}/{date_str}_{origin_model_name}-medical-eval-log.log"]),
        _stage(f"{eval_root}/{date_str}_{update_model_name}_medical_eval_out/results.log", "update_medical_eval", "eval_results", container, [f"{internal_log_root}/{date_str}_{update_model_name}-medical-eval-log.log"]),
        _stage(f"{internal_log_root}/{date_str}-{origin_model_name}-stage_run_origin.log", "origin_stage_run", "text", container, [f"{internal_log_root}/{date_str}-{update_model_name}-origin_stage_run.log"]),
        _stage(f"{internal_log_root}/{date_str}-{update_model_name}-stage_run_update_model.log", "update_stage_run", "text", container, [f"{internal_log_root}/{date_str}-{update_model_name}-update_stage_run.log"]),
        _stage(f"{save_csv_root}/{date_str}-stage_inconsistent_responses.xlsx", "stage_consistency", "binary", container, [f"{internal_log_root}/{date_str}-{update_model_name}-stage_consistency.log"]),
        _stage(f"{internal_log_root}/{date_str}-{update_model_name}-diagnosis_run_update_model.json", "update_diagnosis_run", "json", container, [f"{internal_log_root}/{date_str}-{update_model_name}-update_record_to_diagnose.log"]),
        _stage(f"{internal_log_root}/{date_str}-{origin_model_name}-diagnosis_run_origin.json", "origin_diagnosis_run", "json", container, [f"{internal_log_root}/{date_str}-{update_model_name}-origin_record_to_diagnose.log"]),
        _stage(f"{save_csv_root}/{date_str}-diagnose_inconsistent_responses.xlsx", "diagnosis_consistency", "binary", container, [f"{internal_log_root}/{date_str}-{update_model_name}-diagnosis_consistency.log"]),
        _stage(f"{save_csv_root}/{date_str}-tagged-stage_inconsistent_responses.xlsx", "stage_auto_tagging", "binary", container, [f"{save_csv_root}/{date_str}-stage.log"]),
        _stage(f"{save_csv_root}/{date_str}-tagged-diagnose_inconsistent_responses.xlsx", "diagnose_auto_tagging", "binary", container, [f"{save_csv_root}/{date_str}-diagnose.log"]),
        _stage(f"{log_root}/{date_str}_ckpt_eval_report.txt", "evaluation_report", "text", container),
    ]


def _build_ckpt_artifacts(container: str, date_str: str) -> List[Dict[str, Any]]:
    log_root = DEFAULT_LOG_ROOT
    save_csv_root = f"{log_root}/save_csv"
    return [
        _artifact(f"{save_csv_root}/{date_str}-stage_inconsistent_responses.xlsx", "stage_inconsistent_responses", "binary", container),
        _artifact(f"{save_csv_root}/{date_str}-diagnose_inconsistent_responses.xlsx", "diagnose_inconsistent_responses", "binary", container),
        _artifact(f"{save_csv_root}/{date_str}-tagged-stage_inconsistent_responses.xlsx", "tagged_stage_inconsistent_responses", "binary", container),
        _artifact(f"{save_csv_root}/{date_str}-tagged-diagnose_inconsistent_responses.xlsx", "tagged_diagnose_inconsistent_responses", "binary", container),
        _artifact("/usr/local/insinfersystem/ckpt_eval.log", "ckpt_eval_log", "text", container),
    ]

def _score_summary(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    for item in stages:
        result = item.get("result")
        if isinstance(result, dict) and "average" in result:
            scores[item["name"]] = result
        elif isinstance(result, dict):
            compact = {
                key: value
                for key, value in result.items()
                if key in {"accuracy", "mean_fuzziness"} and value is not None
            }
            if compact:
                scores[item["name"]] = compact
    return scores


def _logical_stage_name(stage_name: str) -> str:
    name = stage_name or ""
    for prefix in ("update_", "origin_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    suffix_map = {
        "stage_run_update_model": "stage_run",
        "stage_run_origin": "stage_run",
        "origin_stage_run": "stage_run",
        "update_stage_run": "stage_run",
        "record_to_diagnose_update_model": "record_to_diagnose",
        "record_to_diagnose_origin": "record_to_diagnose",
        "origin_diagnosis_run": "record_to_diagnose",
        "update_diagnosis_run": "record_to_diagnose",
    }
    return suffix_map.get(name, name)


_PARALLEL_STAGE_GROUPS = [
    ("single_inference", ["record_to_diagnose", "thread_medhalt_confidence_eval"]),
    ("stage_run", ["origin_stage_run", "update_stage_run", "stage_run_origin", "stage_run_update_model"]),
    ("auto_tagging", ["stage_auto_tagging", "diagnose_auto_tagging"]),
]

_STAGE_STATUS_LABELS = {
    "finished": "已完成",
    "running": "运行中",
    "pending": "待运行",
}


def _parallel_stage_group_status(
    stages: List[Dict[str, Any]],
    running_stage_names: List[str],
) -> Optional[Dict[str, Any]]:
    if not running_stage_names:
        return None
    running_set = set(running_stage_names)
    stage_by_name = {
        str(stage.get("name") or ""): stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name")
    }
    for group_name, group_members in _PARALLEL_STAGE_GROUPS:
        present_members = [name for name in group_members if name in stage_by_name]
        if len(present_members) < 2 or not running_set.intersection(present_members):
            continue
        members = []
        for name in present_members:
            status = str(stage_by_name[name].get("status") or "unknown")
            members.append({
                "name": name,
                "status": status,
                "status_label": _STAGE_STATUS_LABELS.get(status, status),
            })
        return {
            "name": group_name,
            "members": members,
            "summary": "，".join(
                f"{item['name']}: {item['status_label']}"
                for item in members
            ),
        }
    return None

def _report_text_matches_terms(report_text: Optional[str], terms: Optional[List[str]]) -> bool:
    required_terms = [str(term).strip() for term in (terms or []) if str(term or "").strip()]
    if not required_terms:
        return True
    text = str(report_text or "").lower()
    return all(term.lower() in text for term in required_terms)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _path_nonempty(container: str, path: str) -> bool:
    meta = _path_stat(container, path)
    return bool(meta and int(meta.get("size_bytes") or 0) > 0)


def _copy_container_file_if_missing(container: str, source: str, target: str) -> bool:
    if not _path_nonempty(container, source) or _path_nonempty(container, target):
        return False
    script = (
        f"mkdir -p {shlex.quote(os.path.dirname(target))} && "
        f"cp -f {shlex.quote(source)} {shlex.quote(target)} && "
        f"test -s {shlex.quote(target)}"
    )
    code, _, _ = _docker_sh(container, script, timeout=30)
    return code == 0


def _load_auto_tagging_recovery_records() -> List[Dict[str, Any]]:
    return _load_jsonl(_AUTO_TAGGING_RECOVERY_REGISTRY)


def _recovery_record_matches(
    record: Dict[str, Any],
    container: str,
    script_name: str,
    date_str: str,
    pid: Optional[str] = None,
) -> bool:
    if record.get("container") != container:
        return False
    if record.get("script_name") != script_name:
        return False
    if str(record.get("date_str") or "") != str(date_str):
        return False
    if pid and str(record.get("original_pid") or "") != str(pid) and str(record.get("new_pid") or "") != str(pid):
        return False
    return True


def _latest_auto_tagging_recovery(
    container: str,
    script_name: str,
    date_str: str,
    pid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    matches = [
        record
        for record in _load_auto_tagging_recovery_records()
        if _recovery_record_matches(record, container, script_name, date_str, pid)
    ]
    return matches[-1] if matches else None


def _auto_tagging_invalid_rows(error_tail: str) -> Optional[str]:
    match = re.search(r"Invalid results for rows:\s*(\[[0-9,\s]+\])", error_tail or "")
    return match.group(1).strip() if match else None


def _auto_tagging_review_rows(record: Dict[str, Any]) -> Optional[str]:
    value = str(record.get("review_rows") or "").strip()
    return value if re.fullmatch(r"\[[0-9,\s]+\]", value) else None


def _failed_auto_tagging_stages(error_tail: str) -> List[str]:
    text = error_tail or ""
    failure_markers = (
        "returned non-zero exit status",
        "Failed to execute script",
        "CalledProcessError",
        "Invalid results for rows",
    )
    if not any(marker in text for marker in failure_markers):
        return []
    stages: List[str] = []
    if "auto_tagging_stage" in text:
        stages.append("stage_auto_tagging")
    if "auto_tagging_diagnose" in text:
        stages.append("diagnose_auto_tagging")
    return stages


def _auto_tagging_log_paths(date_str: str, stage_name: str, model_name: Optional[str]) -> List[str]:
    paths: List[str] = []
    save_csv_root = f"{DEFAULT_LOG_ROOT}/save_csv"
    internal_log_root = f"{DEFAULT_LOG_ROOT}/internal_log"
    if stage_name == "stage_auto_tagging":
        paths.append(f"{save_csv_root}/{date_str}-stage.log")
        if model_name:
            paths.append(f"{internal_log_root}/{date_str}-{model_name}-stage_auto_tagging.log")
    elif stage_name == "diagnose_auto_tagging":
        paths.append(f"{save_csv_root}/{date_str}-diagnose.log")
        if model_name:
            paths.append(f"{internal_log_root}/{date_str}-{model_name}-diagnose_auto_tagging.log")
    return paths


def _auto_tagging_recovery_text(
    container: str,
    date_str: str,
    metrics: Dict[str, Any],
    model_name: Optional[str],
) -> str:
    parts = [str(metrics.get("error_tail") or "")]
    for detail in metrics.get("current_stage_details") or []:
        if not isinstance(detail, dict):
            continue
        summary = detail.get("progress_summary") or {}
        if isinstance(summary, dict):
            parts.extend(str(line) for line in (summary.get("tail_lines") or []) if line)
            latest_line = summary.get("latest_line")
            if latest_line:
                parts.append(str(latest_line))
        progress_log = detail.get("progress_log")
        if progress_log and _path_exists(container, str(progress_log)):
            parts.append(_read_tail(container, str(progress_log), lines=80))

    for stage_name in ("stage_auto_tagging", "diagnose_auto_tagging"):
        for log_path in _auto_tagging_log_paths(date_str, stage_name, model_name):
            if _path_exists(container, log_path):
                parts.append(_read_tail(container, log_path, lines=80))
    return "\n".join(part for part in parts if part)


def _auto_tagging_file_pair(date_str: str, stage_name: str) -> Optional[Tuple[str, str]]:
    save_csv_root = f"{DEFAULT_LOG_ROOT}/save_csv"
    if stage_name == "stage_auto_tagging":
        return (
            f"{save_csv_root}/{date_str}-stage_inconsistent_responses.xlsx",
            f"{save_csv_root}/{date_str}-tagged-stage_inconsistent_responses.xlsx",
        )
    if stage_name == "diagnose_auto_tagging":
        return (
            f"{save_csv_root}/{date_str}-diagnose_inconsistent_responses.xlsx",
            f"{save_csv_root}/{date_str}-tagged-diagnose_inconsistent_responses.xlsx",
        )
    return None


def _auto_tagging_recovery_message(record: Dict[str, Any], report_done: bool) -> str:
    stage_text = "、".join(record.get("failed_stages") or []) or "auto-tagging"
    review_rows = _auto_tagging_review_rows(record)
    row_text = f"（行号：{review_rows}）" if review_rows else "（具体行号请查看 auto-tagging 日志）"
    if report_done:
        return f"{stage_text} 曾解析失败{row_text}，已用原始 inconsistent Excel 生成占位 tagged 文件并完成续跑；请人工复核相关行。"
    if record.get("new_pid"):
        return f"{stage_text} 解析失败后已自动补齐 tagged 文件并启动续跑，等待最终报告生成。"
    if record.get("start_error"):
        return f"{stage_text} 解析失败后已补齐 tagged 文件，但自动续跑启动失败：{record.get('start_error')}"
    if record.get("copied_files"):
        return f"{stage_text} 解析失败后已补齐 tagged 文件，正在准备自动续跑。"
    return f"{stage_text} 解析失败，已记录自动恢复状态。"


def _auto_tagging_rerun_env(record: Optional[Dict[str, Any]], container: str) -> Dict[str, Any]:
    env_vars = dict(record.get("env_vars") or {}) if isinstance(record, dict) else {}
    for key in ("MEDFLOW_ASSIGNED_GPUS", "CUDA_VISIBLE_DEVICES", "LOCALHOST_ID"):
        env_vars.pop(key, None)
    env_vars["container"] = container
    return env_vars


def _auto_tagging_recovery_can_retry(record: Dict[str, Any]) -> bool:
    if record.get("new_pid"):
        return False
    if record.get("status") == "copied" and not record.get("start_error"):
        return True
    start_error = str(record.get("start_error") or "")
    return record.get("status") == "rerun_start_failed" and "不能手工指定物理 GPU 卡号" in start_error


def _start_auto_tagging_recovery_rerun(
    script_name: str,
    record: Optional[Dict[str, Any]],
    container: str,
    model_fir: Optional[str],
    model_sec: Optional[str],
    ckpt_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    env_vars = _auto_tagging_rerun_env(record, container)
    additional_args: Dict[str, Any]
    if script_name == "ckpt_eval":
        additional_args = {"CKPT_PATH": ckpt_path or _record_value(record, "CKPT_PATH")}
        tem = _record_value(record, "TEM")
        if tem:
            additional_args["TEM"] = tem
    else:
        additional_args = {
            "model_fir": model_fir or _record_value(record, "model_fir"),
            "model_sec": model_sec or _record_value(record, "model_sec"),
        }
    additional_args = {key: value for key, value in additional_args.items() if value}
    try:
        from .runlocal_evaluate import run_script_by_name_evaluate

        response = run_script_by_name_evaluate(
            script_name,
            additional_args=additional_args,
            background=True,
            env_vars=env_vars,
            use_docker=True,
            skip_prompt=True,
        )
    except Exception as exc:
        logger.exception("Failed to auto rerun evaluation after auto-tagging recovery")
        return None, f"auto_rerun_exception: {exc}"

    hint = (getattr(response, "metadata", {}) or {}).get("protocol_hint") or {}
    new_pid = hint.get("pid")
    if new_pid:
        return str(new_pid), None
    return None, str(hint.get("message") or _truncate_text(str(response), 500) or "auto rerun did not return pid")


def _maybe_recover_auto_tagging_failure(
    payload: Dict[str, Any],
    record: Optional[Dict[str, Any]],
    *,
    container: str,
    pid: Optional[str],
    pid_alive: bool,
    script_name: str,
    date_str: str,
    model_fir: Optional[str] = None,
    model_sec: Optional[str] = None,
    ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    existing_recovery = _latest_auto_tagging_recovery(container, script_name, date_str, pid)
    report_done = bool(metrics.get("report_path"))
    if existing_recovery:
        can_resume_copied_recovery = (
            payload.get("status") == "failed"
            and not pid_alive
            and not report_done
            and pid
            and existing_recovery.get("copied_files")
            and _auto_tagging_recovery_can_retry(existing_recovery)
        )
        if not can_resume_copied_recovery:
            metrics["auto_tagging_recovery"] = {
                **existing_recovery,
                "review_rows": _auto_tagging_review_rows(existing_recovery),
                "message": _auto_tagging_recovery_message(existing_recovery, report_done),
            }
            if report_done:
                payload["analysis_text"] = (
                    f"{payload.get('analysis_text') or ''} auto-tagging 曾失败，已自动占位续跑；请人工复核相关行。"
                ).strip()
            return payload

        new_pid, start_error = _start_auto_tagging_recovery_rerun(
            script_name,
            record,
            container,
            model_fir,
            model_sec,
            ckpt_path,
        )
        resumed_record = {
            **existing_recovery,
            "status": "rerun_started" if new_pid else "rerun_start_failed",
            "new_pid": new_pid,
            "start_error": start_error,
            "updated_at": datetime.now().isoformat(),
            "updated_at_ts": time.time(),
        }
        _append_jsonl(_AUTO_TAGGING_RECOVERY_REGISTRY, resumed_record)
        metrics["auto_tagging_recovery"] = {
            **resumed_record,
            "review_rows": _auto_tagging_review_rows(resumed_record),
            "message": _auto_tagging_recovery_message(resumed_record, report_done=False),
        }
        if new_pid:
            payload["status"] = "running"
            metrics["pid"] = str(new_pid)
            metrics["pid_alive"] = True
            metrics["error_reason"] = None
            payload["analysis_text"] = "检测到 auto-tagging 已补齐但续跑未启动，现已自动启动同参数续跑，等待最终报告生成。"
        else:
            payload["analysis_text"] = f"检测到 auto-tagging 已补齐 tagged 文件，但自动续跑启动失败：{start_error}"
        return payload

    if payload.get("status") != "failed" or pid_alive or report_done or not pid:
        return payload

    model_name = (
        metrics.get("update_model_name")
        or metrics.get("model_name")
        or _basename(model_sec)
        or _ckpt_output_name(ckpt_path)
    )
    recovery_text = _auto_tagging_recovery_text(container, date_str, metrics, str(model_name) if model_name else None)
    failed_stages = _failed_auto_tagging_stages(recovery_text)
    if not failed_stages:
        return payload

    copied_files = []
    for stage_name in failed_stages:
        pair = _auto_tagging_file_pair(date_str, stage_name)
        if not pair:
            continue
        source, target = pair
        if _copy_container_file_if_missing(container, source, target):
            copied_files.append({"stage": stage_name, "source": source, "target": target})

    if not copied_files:
        return payload

    recovery_record: Dict[str, Any] = {
        "container": container,
        "script_name": script_name,
        "date_str": str(date_str),
        "original_pid": str(pid),
        "failed_stages": failed_stages,
        "copied_files": copied_files,
        "review_rows": _auto_tagging_invalid_rows(recovery_text),
        "status": "copied",
        "created_at": datetime.now().isoformat(),
        "created_at_ts": time.time(),
    }
    _append_jsonl(_AUTO_TAGGING_RECOVERY_REGISTRY, recovery_record)

    new_pid, start_error = _start_auto_tagging_recovery_rerun(
        script_name,
        record,
        container,
        model_fir,
        model_sec,
        ckpt_path,
    )
    recovery_record.update({
        "status": "rerun_started" if new_pid else "rerun_start_failed",
        "new_pid": new_pid,
        "start_error": start_error,
        "updated_at": datetime.now().isoformat(),
        "updated_at_ts": time.time(),
    })
    _append_jsonl(_AUTO_TAGGING_RECOVERY_REGISTRY, recovery_record)

    metrics["auto_tagging_recovery"] = {
        **recovery_record,
        "message": _auto_tagging_recovery_message(recovery_record, report_done=False),
    }
    if new_pid:
        payload["status"] = "running"
        metrics["pid"] = str(new_pid)
        metrics["pid_alive"] = True
        metrics["error_reason"] = None
        payload["analysis_text"] = "检测到 auto-tagging 可恢复失败，已补齐 tagged 文件并自动启动同参数续跑，等待最终报告生成。"
    else:
        payload["analysis_text"] = f"检测到 auto-tagging 可恢复失败并已补齐 tagged 文件，但自动续跑启动失败：{start_error}"
    return payload


def _build_payload(
    container: str,
    pid: Optional[str],
    pid_alive: bool,
    script_name: str,
    date_str: str,
    stages: List[Dict[str, Any]],
    report_stage_name: Optional[str],
    error_log: Optional[str],
    report_match_terms: Optional[List[str]] = None,
    started_at_ts: Optional[float] = None,
) -> Dict[str, Any]:
    for index, stage in enumerate(stages, start=1):
        stage["index"] = index
        stage["total"] = len(stages)
    report_stage = next((s for s in stages if s.get("name") == report_stage_name), None) if report_stage_name else None
    report_text: Optional[str] = None
    report_done = False
    if report_stage_name:
        non_report_stages = [s for s in stages if s.get("name") != report_stage_name]
        pre_report_done = bool(non_report_stages) and all(
            stage.get("status") == "finished" for stage in non_report_stages
        )
        report_file_done = bool(report_stage and report_stage.get("status") == "finished")
        if pre_report_done and report_file_done and report_stage:
            report_text = _read_full_text(container, report_stage.get("path"))
            report_done = _report_text_matches_terms(report_text, report_match_terms)
        if report_stage and not report_done:
            report_text = None
            report_stage["status"] = "pending"
            report_stage["output_meta"] = None
            report_stage["result"] = None
    else:
        report_done = bool(stages) and all(stage.get("status") == "finished" for stage in stages)
    completed = [s for s in stages if s.get("status") == "finished"]
    running = [s for s in stages if s.get("status") == "running"]
    next_stage = next((s["name"] for s in stages if s.get("status") == "pending"), None)
    running_stage_names = [s["name"] for s in running]
    current_stage_names = list(running_stage_names)
    running_logical_names = []
    for stage_name in running_stage_names:
        logical_name = _logical_stage_name(stage_name)
        if logical_name not in running_logical_names:
            running_logical_names.append(logical_name)
    if len(running_logical_names) == 1:
        current = running_logical_names[0]
    elif len(running_logical_names) > 1:
        current = "、".join(running_logical_names)
    else:
        current = next_stage if completed else None
    if report_done:
        current = None
        next_stage = None
    error_exists = _path_exists(container, error_log) if error_log else False
    error_tail = (
        _read_error_tail_since(container, error_log, started_at_ts, lines=80)
        if error_exists and error_log
        else ""
    )
    overall_progress_percent = 100.0 if report_done else (
        round(len(completed) * 100 / len(stages), 2) if stages else 0.0
    )
    stage_status_counts = {
        "finished": len(completed),
        "running": len(running),
        "pending": len(stages) - len(completed) - len(running),
    }
    current_stage_details = running if running else (
        [next(s for s in stages if s.get("name") == next_stage)]
        if next_stage
        else []
    )
    parallel_group_status = _parallel_stage_group_status(stages, running_stage_names)
    if pid_alive and not running_stage_names and next_stage and completed:
        current_stage_names = [next_stage]

    if report_done:
        status = "finished"
        analysis_text = "模型评估已结束，已读取最终报告与各阶段输出结果。"
    elif pid_alive:
        status = "running" if completed or running else "starting"
        if running:
            analysis_text = f"模型评估正在运行，当前阶段为 {current}，已检测到阶段运行日志。"
        elif completed:
            analysis_text = f"模型评估正在运行，已完成 {len(completed)}/{len(stages)} 个阶段，下一阶段约为 {next_stage or '收尾'}。"
        else:
            analysis_text = "模型评估已启动，暂未检测到阶段输出文件，正在等待首个阶段产出。"
    elif completed:
        if error_tail:
            status = "failed"
            analysis_text = "评估进程已结束且检测到本次运行错误日志；已返回已完成阶段的结果，请优先查看错误日志尾部。"
        else:
            status = "interrupted"
            analysis_text = "评估进程已结束，但最终报告尚未生成；已返回已完成阶段的结果，请检查错误日志或末尾输出。"
    else:
        status = "failed"
        analysis_text = "未检测到运行中的评估进程，也未找到评估输出文件。"

    return {
        "status": status,
        "analysis_text": analysis_text,
        "metrics": {
            "container_name": container,
            "pid": str(pid) if pid else None,
            "pid_alive": pid_alive,
            "script_name": script_name,
            "date_str": str(date_str),
            "current_stage": current,
            "current_stages": current_stage_names,
            "current_stage_details": current_stage_details,
            "parallel_stage_group": parallel_group_status,
            "next_stage": next_stage,
            "running_stages": running_stage_names,
            "completed_stages": len(completed),
            "total_stages": len(stages),
            "overall_progress_percent": overall_progress_percent,
            "stage_status_counts": stage_status_counts,
            "scores": _score_summary(stages),
            "stages": stages,
            "report_path": report_stage.get("path") if report_done and report_stage else None,
            "report_text": report_text,
            "error_log": error_log if error_tail else None,
            "error_tail": _truncate_text(error_tail),
            "error_detail": _truncate_text(error_tail) if status == "failed" and error_tail else None,
            "error_reason": None if status in {"running", "starting", "finished"} else status,
        },
    }


_CKPT_ACTIVE_LOG_FRESH_SECONDS = 300

_CKPT_ACTIVE_LOG_STAGES = {
    "origin_stage_run",
    "update_stage_run",
    "stage_consistency",
    "origin_diagnosis_run",
    "update_diagnosis_run",
    "diagnosis_consistency",
    "stage_auto_tagging",
    "diagnose_auto_tagging",
}


def _stage_updated_at_ts(stage: Dict[str, Any]) -> float:
    for meta_key in ("output_meta", "progress_log_meta"):
        meta = stage.get(meta_key) or {}
        value = meta.get("updated_at_ts")
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0



def _ckpt_stage_process_alive(container: str, stage: Dict[str, Any]) -> bool:
    targets = [
        str(stage.get(key) or "").strip()
        for key in ("path", "progress_log")
    ]
    targets = [target for target in targets if target]
    if not container or not targets:
        return False
    code, out, _ = _docker_sh(container, "ps -eo args=", timeout=8)
    if code != 0 or not out:
        return False
    return any(target in out for target in targets)


def _adjust_ckpt_active_stage(
    payload: Dict[str, Any],
    started_at_ts: Optional[float],
) -> Dict[str, Any]:
    """Treat ckpt log-like outputs as active while the ckpt process is alive."""
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        return payload
    if payload.get("status") == "finished":
        return payload
    if metrics.get("running_stages"):
        return payload

    stages = metrics.get("stages") or []
    active_candidates = []
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("name") not in _CKPT_ACTIVE_LOG_STAGES:
            continue
        updated_at_ts = _stage_updated_at_ts(stage)
        if updated_at_ts <= 0:
            continue
        if started_at_ts is not None and updated_at_ts < started_at_ts:
            continue
        active_candidates.append((updated_at_ts, stage))
    if not active_candidates:
        return payload

    active_updated_at_ts, active_stage = max(active_candidates, key=lambda item: item[0])
    active_stage_name = str(active_stage.get("name") or "")
    pid_alive = bool(metrics.get("pid_alive"))
    child_process_alive = False
    if not pid_alive:
        child_process_alive = _ckpt_stage_process_alive(
            str(metrics.get("container_name") or ""),
            active_stage,
        )
    if (
        not pid_alive
        and not child_process_alive
        and time.time() - active_updated_at_ts > _CKPT_ACTIVE_LOG_FRESH_SECONDS
    ):
        return payload

    previous_status = str(active_stage.get("status") or "")
    active_stage["status"] = "running"
    active_detail = dict(active_stage)
    if not active_detail.get("progress_log"):
        active_detail["progress_log"] = active_detail.get("path")
    if not active_detail.get("progress_log_meta"):
        active_detail["progress_log_meta"] = active_detail.get("output_meta")

    metrics["current_stage"] = _logical_stage_name(active_stage_name)
    metrics["current_stages"] = [active_stage_name]
    metrics["current_stage_details"] = [active_detail]
    metrics["running_stages"] = [active_stage_name]
    counts = dict(metrics.get("stage_status_counts") or {})
    if previous_status in counts:
        counts[previous_status] = max(int(counts.get(previous_status) or 0) - 1, 0)
    counts["running"] = max(int(counts.get("running") or 0), 0) + 1
    metrics["stage_status_counts"] = counts
    if previous_status == "finished":
        completed_stages = max(int(metrics.get("completed_stages") or 0) - 1, 0)
        metrics["completed_stages"] = completed_stages
        total_stages = int(metrics.get("total_stages") or 0)
        if total_stages > 0:
            metrics["overall_progress_percent"] = round(completed_stages * 100 / total_stages, 2)
    payload["status"] = "running"
    metrics["error_reason"] = None
    if pid_alive:
        payload["analysis_text"] = f"模型评估正在运行，当前 ckpt 子阶段约为 {active_stage_name}，已检测到该阶段日志更新。"
    elif child_process_alive:
        payload["analysis_text"] = f"模型评估仍在运行，启动 PID 已结束，但当前 ckpt 子进程 {active_stage_name} 仍在运行。"
    else:
        payload["analysis_text"] = f"模型评估仍在运行，启动 PID 已结束，但当前 ckpt 子阶段 {active_stage_name} 的日志仍在更新。"
    return payload

def run_script_by_name_evaluate_monitor(
    script_query: str = "assessment_monitor",
    additional_args: Dict[str, Any] = None,
    container_name: Optional[str] = None,
    pid: Optional[str] = None,
    model_fir: Optional[str] = None,
    model_sec: Optional[str] = None,
    date_str: Optional[str] = None,
) -> ToolResponse:
    """监控模型评估任务的阶段状态和阶段结果。"""
    args = additional_args or {}
    container = container_name or args.get("container") or args.get("container_name")
    pid = pid or args.get("pid")
    record = _fallback_latest_record(pid, container)

    if record:
        container = container or record.get("container")
        pid = pid or record.get("pid")
        model_fir = model_fir or args.get("model_fir") or _record_value(record, "model_fir")
        model_sec = model_sec or args.get("model_sec") or _record_value(record, "model_sec")
        date_str = date_str or args.get("date_str") or _date_from_record(record)
    else:
        model_fir = model_fir or args.get("model_fir")
        model_sec = model_sec or args.get("model_sec")
        date_str = date_str or args.get("date_str")

    container = container or DEFAULT_CONTAINER
    if not container:
        return _monitor_response({
            "status": "unknown",
            "analysis_text": "未指定容器名称，且未找到评估启动记录。",
            "metrics": {"error_reason": "container_required", "pid": pid},
        })

    if pid and not record:
        return _monitor_response({
            "status": "failed",
            "analysis_text": "未找到该 PID 对应的评估记录，请确认 PID 是否来自评估启动返回。",
            "metrics": {"container_name": container, "pid": pid, "error_reason": "pid_not_found"},
        })

    script_name = _script_from_record(record, script_query)
    if script_name in {"assessment_monitor", "evaluate_monitor", "monitor", "assessment监控", "评测监控", "评估监控"}:
        script_name = "compare_between_models_vpn" if model_sec else "single_model_evaluation_vpn"

    pid_alive = _check_pid_alive(container, str(pid)) if pid else False
    started_at_ts = _record_start_ts(record)

    if script_name == "single_model_evaluation_vpn":
        model_name = _basename(model_fir or _record_value(record, "model_path"))
        if not model_name:
            return _monitor_response({
                "status": "unknown",
                "analysis_text": "缺少模型名/模型路径，无法定位 single_model_evaluation_vpn 的输出文件。",
                "metrics": {
                    "container_name": container,
                    "pid": pid,
                    "script_name": script_name,
                    "required_params": ["model_fir"],
                    "error_reason": "model_names_required",
                },
            })
        date_str = _resolve_single_eval_date(container, model_name, date_str)
        stages = _build_single_stages(container, str(date_str), model_name)
        payload = _build_payload(
            container=container,
            pid=pid,
            pid_alive=pid_alive,
            script_name=script_name,
            date_str=str(date_str),
            stages=stages,
            report_stage_name="evaluation_report",
            error_log=f"{DEFAULT_LOG_ROOT}/{date_str}_single_model_valuation_error.log",
            report_match_terms=[model_name],
            started_at_ts=started_at_ts,
        )
        payload["metrics"]["model_name"] = model_name
        payload["metrics"]["model_fir"] = model_fir
        payload["metrics"]["artifacts"] = _build_single_artifacts(container, str(date_str))
        return _monitor_response(payload)

    if script_name == "ckpt_eval":
        ckpt_path = args.get("CKPT_PATH") or args.get("ckpt") or _record_value(record, "CKPT_PATH")
        output_path = _ckpt_output_path(ckpt_path)
        model_name = _ckpt_output_name(ckpt_path) or args.get("model_name")
        if not model_name:
            return _monitor_response({
                "status": "unknown",
                "analysis_text": "缺少 checkpoint 路径，无法定位 ckpt_eval 的输出文件。",
                "metrics": {
                    "container_name": container,
                    "pid": pid,
                    "script_name": script_name,
                    "required_params": ["CKPT_PATH"],
                    "error_reason": "model_names_required",
                },
            })
        date_str = date_str or args.get("date_str") or _date_from_record(record)
        stages = _build_ckpt_stages(container, str(date_str), str(model_name), output_path=output_path)
        payload = _build_payload(
            container=container,
            pid=pid,
            pid_alive=pid_alive,
            script_name=script_name,
            date_str=str(date_str),
            stages=stages,
            report_stage_name="evaluation_report",
            error_log=f"{DEFAULT_LOG_ROOT}/{date_str}_ckpt_eval_error.log",
            report_match_terms=[str(model_name)],
        )
        payload = _adjust_ckpt_active_stage(payload, started_at_ts)
        payload["metrics"]["ckpt_path"] = ckpt_path
        payload["metrics"]["merge_output_path"] = output_path
        payload["metrics"]["model_name"] = model_name
        payload["metrics"]["artifacts"] = _build_ckpt_artifacts(container, str(date_str))
        payload = _maybe_recover_auto_tagging_failure(
            payload,
            record,
            container=container,
            pid=pid,
            pid_alive=pid_alive,
            script_name=script_name,
            date_str=str(date_str),
            ckpt_path=ckpt_path,
        )
        return _monitor_response(payload)

    origin_model_name = _basename(model_fir)
    update_model_name = _basename(model_sec)
    date_str = date_str or datetime.now().strftime("%Y%m%d")
    if not origin_model_name or not update_model_name:
        return _monitor_response({
            "status": "unknown",
            "analysis_text": "缺少模型名/模型路径，无法定位 compare_between_models_vpn 的输出文件。",
            "metrics": {
                "container_name": container,
                "pid": pid,
                "script_name": script_name,
                "required_params": ["model_fir", "model_sec"],
                "error_reason": "model_names_required",
            },
        })

    stages = _build_compare_stages(container, str(date_str), origin_model_name, update_model_name)
    payload = _build_payload(
        container=container,
        pid=pid,
        pid_alive=pid_alive,
        script_name="compare_between_models_vpn",
        date_str=str(date_str),
        stages=stages,
        report_stage_name="evaluation_report",
        error_log=f"{DEFAULT_LOG_ROOT}/{date_str}_compare_between_models_error.log",
        report_match_terms=[origin_model_name, update_model_name],
        started_at_ts=started_at_ts,
    )
    payload["metrics"]["origin_model_name"] = origin_model_name
    payload["metrics"]["update_model_name"] = update_model_name
    payload["metrics"]["artifacts"] = _build_compare_artifacts(container, str(date_str))
    payload = _maybe_recover_auto_tagging_failure(
        payload,
        record,
        container=container,
        pid=pid,
        pid_alive=pid_alive,
        script_name="compare_between_models_vpn",
        date_str=str(date_str),
        model_fir=model_fir,
        model_sec=model_sec,
    )
    return _monitor_response(payload)


run_script_by_name_assessment_monitor = run_script_by_name_evaluate_monitor

__all__ = [
    "run_script_by_name_assessment_monitor",
    "run_script_by_name_evaluate_monitor",
]





