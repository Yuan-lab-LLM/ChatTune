"""Benchmark ETA calculation and display helpers."""

from datetime import datetime
import os
import re
import time
from typing import Optional


def medbench_sample_counts(progress: object) -> tuple[int, int]:
    """Aggregate completed and total samples across MedBench files."""

    if not isinstance(progress, dict):
        return 0, 0
    files = progress.get("files")
    if not isinstance(files, dict):
        return 0, 0
    done_samples = 0
    total_samples = 0
    for stat in files.values():
        if not isinstance(stat, dict):
            continue
        try:
            done_samples += int(stat.get("done") or 0)
        except (TypeError, ValueError):
            pass
        try:
            total_samples += int(stat.get("total") or 0)
        except (TypeError, ValueError):
            pass
    return done_samples, total_samples


def benchmark_log_progress(log_path: object) -> tuple[int | None, int | None]:
    """Read the most recent completed-sample counter from a runner log."""
    if not log_path:
        return None, None
    try:
        with open(os.fspath(log_path), "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 65536))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, TypeError):
        return None, None
    matches = re.findall(r"\[(\d+)\s*/\s*(\d+)\]\s*done", tail)
    if not matches:
        return None, None
    return tuple(map(int, matches[-1]))


def benchmark_eta_fields(
    status: object,
    start_time: object,
    processed: object,
    total: object,
    *,
    now: Optional[float] = None,
) -> dict:
    """Estimate remaining benchmark time from cumulative sample throughput."""

    calculated_at = float(time.time() if now is None else now)
    normalized_status = str(status or "").strip().lower()
    terminal_success = normalized_status in {
        "complete",
        "completed",
        "done",
        "finished",
        "success",
        "succeeded",
    }
    terminal_failure = normalized_status in {
        "cancelled",
        "canceled",
        "error",
        "failed",
        "stopped",
    }

    try:
        processed_value = int(processed) if processed is not None else None
        total_value = int(total) if total is not None else None
    except (TypeError, ValueError):
        processed_value = None
        total_value = None

    payload = {"eta_calculated_at": calculated_at}
    if terminal_success or (
        processed_value is not None
        and total_value is not None
        and total_value > 0
        and processed_value >= total_value
    ):
        return {
            **payload,
            "eta_status": "complete",
            "estimated_remaining_seconds": 0,
        }
    if terminal_failure:
        return {**payload, "eta_status": "unavailable"}

    if (
        processed_value is None
        or total_value is None
        or processed_value <= 0
        or total_value <= processed_value
    ):
        return {**payload, "eta_status": "calculating"}

    try:
        start_text = str(start_time or "").strip()
        if not start_text:
            raise ValueError("missing start time")
        if start_text.replace(".", "", 1).isdigit():
            started_at = float(start_text)
        else:
            parsed = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            started_at = parsed.timestamp()
        elapsed_seconds = max(0.0, calculated_at - started_at)
    except (TypeError, ValueError, OverflowError):
        return {**payload, "eta_status": "unavailable"}

    if elapsed_seconds <= 0:
        return {**payload, "eta_status": "calculating"}
    remaining_seconds = round(
        elapsed_seconds * (total_value - processed_value) / processed_value
    )
    return {
        **payload,
        "eta_status": "available",
        "estimated_remaining_seconds": max(0, int(remaining_seconds)),
    }


def format_benchmark_eta(seconds: object) -> str:
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


def benchmark_eta_text(payload: dict) -> str:
    eta_status = str(payload.get("eta_status") or "")
    if eta_status == "available":
        return format_benchmark_eta(payload.get("estimated_remaining_seconds"))
    if eta_status == "complete":
        return "已完成"
    if eta_status == "calculating":
        return "计算中"
    return "暂不可用"
