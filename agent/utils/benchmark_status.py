"""Helpers shared by workflow benchmark startup and status readers."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
REPORT_PATTERN = re.compile(r"benchmark_report\((?:job_id=)?[\"']?([A-Za-z0-9_-]+)[\"']?\)")
START_PATTERNS = (
    re.compile(r"(?:\*\*)?Job\s*ID(?:\*\*)?\s*[:：]\s*`?([A-Za-z0-9_-]+)", re.I),
    re.compile(r"任务\s*ID(?:为|是)?\s*[:：]?\s*`?([A-Za-z0-9_-]+)", re.I),
)


def valid_benchmark_job_id(value: Any) -> str:
    job_id = str(value or "").strip()
    return job_id if JOB_ID_PATTERN.fullmatch(job_id) and job_id != "submit_locks" else ""


def extract_benchmark_job_id(payload: Any) -> str:
    """Prefer explicit fields, then parse the inference startup response text."""
    if not isinstance(payload, dict):
        return ""
    for source in (payload, payload.get("data"), payload.get("protocol")):
        if not isinstance(source, dict):
            continue
        for key in ("benchmark_job_id", "job_id", "jobId"):
            job_id = valid_benchmark_job_id(source.get(key))
            if job_id:
                return job_id
    for key in ("status_command", "result", "message", "text"):
        text = str(payload.get(key) or "")
        match = REPORT_PATTERN.search(text)
        if match and valid_benchmark_job_id(match.group(1)):
            return match.group(1)
        for pattern in START_PATTERNS:
            match = pattern.search(text)
            if match and valid_benchmark_job_id(match.group(1)):
                return match.group(1)
    return ""


def recover_benchmark_job_id(stage: dict[str, Any]) -> str:
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    job_id = extract_benchmark_job_id(result)
    if job_id:
        return job_id
    log_path = stage.get("log_path") or result.get("log_path")
    if not log_path:
        return ""
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or not str(entry.get("title") or "").startswith("运行推理基准测试"):
            continue
        job_id = extract_benchmark_job_id(entry.get("payload"))
        if job_id:
            return job_id
    return ""


def recover_benchmark_resource_context(stage: dict[str, Any]) -> dict[str, Any]:
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    context = result.get("resource_context")
    if isinstance(context, dict) and context.get("runtime_node_id"):
        return context
    log_path = stage.get("log_path") or result.get("log_path")
    if not log_path:
        return {}
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or not str(entry.get("title") or "").startswith("运行推理基准测试"):
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        context = payload.get("resource_context")
        if isinstance(context, dict) and context.get("runtime_node_id"):
            return context
    return {}


def inference_benchmark_status_url(inference_agent_url: str) -> str:
    base = str(inference_agent_url or "").strip().rstrip("/")
    if base.endswith("/inference_agent"):
        base = base[: -len("/inference_agent")]
    return f"{base}/benchmark/status"


def query_inference_benchmark_status(
    inference_agent_url: str,
    *,
    job_id: str,
    node: str,
    user_id: str,
    user_aliases: list[str],
    user_role: str,
    resource_context: dict[str, Any] | None,
    headers: dict[str, str],
) -> dict[str, Any]:
    request = urllib.request.Request(
        inference_benchmark_status_url(inference_agent_url),
        data=json.dumps({
            "job_id": job_id,
            "node": node,
            "user_id": user_id,
            "user_aliases": user_aliases,
            "user_role": user_role,
            "resource_context": resource_context or {},
        }, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(str(payload.get("error") if isinstance(payload, dict) else "invalid benchmark status response"))
    report = payload.get("report")
    if not isinstance(report, dict) or str(report.get("job_id") or "") != job_id:
        raise RuntimeError("Benchmark status response does not match the requested job")
    return report


def workflow_benchmark_result(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    sample_progress = report.get("sample_progress") if isinstance(report.get("sample_progress"), dict) else {}
    job_id = valid_benchmark_job_id(report.get("job_id"))
    status = str(report.get("status") or "").lower()
    payload = {
        "status": status,
        "benchmark_status": status,
        "benchmark_job_id": job_id,
        "model": report.get("model"),
        "dataset": report.get("dataset"),
        "start_time": report.get("start_time"),
        "end_time": report.get("end_time"),
        "processed": report.get("processed", sample_progress.get("done_samples")),
        "total": report.get("total", sample_progress.get("total_samples")),
        "progress_percent": report.get("progress_percent", sample_progress.get("percent")),
        "eta_status": report.get("eta_status"),
        "estimated_remaining_seconds": report.get("estimated_remaining_seconds"),
        "eta_calculated_at": report.get("eta_calculated_at"),
        "log_file": artifacts.get("log"),
        "result_path": artifacts.get("output"),
        "folder_path": str(Path(artifacts["meta"]).parent) if artifacts.get("meta") else None,
        "error": report.get("error"),
    }
    for key in ("correct", "accuracy", "invalid", "invalid_rate", "avg_f1", "record_only", "executed", "passed", "failed", "timeout", "executor_error", "pass@1"):
        value = metrics.get(key, summary.get(key))
        if value is not None:
            payload[key] = value
    if job_id:
        payload.update(
            status_command=f'benchmark_report("{job_id}")',
            stop_command=f'benchmark_stop("{job_id}")',
        )
    return payload


def format_benchmark_remaining(seconds: Any) -> str:
    try:
        remaining = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "暂不可用"
    if remaining < 60:
        return "不足1分钟"
    minutes = remaining // 60
    if minutes < 60:
        return f"约{minutes}分钟"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"约{hours}小时{minutes}分钟" if minutes else f"约{hours}小时"
    days, hours = divmod(hours, 24)
    return f"约{days}天{hours}小时" if hours else f"约{days}天"
