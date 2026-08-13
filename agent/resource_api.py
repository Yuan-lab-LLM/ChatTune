# -*- coding: utf-8 -*-
"""Protected node-local resource management API."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from time import monotonic, sleep
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)


DATASET_PATHS = {
    "raw": "/home/workspace/dataset",
    "sft": "/home/workspace/dataset_batch_train",
    "dpo": "/home/workspace/dataset_daily_train",
}
MODEL_PATHS = {
    "base_train": "/home/workspace/models/base",
    "batch_trained": "/home/workspace/models/batch_train",
    "daily_trained": "/home/workspace/models/dpo_train/internal/saves",
    "inference": "/home/workspace/medical_models",
}
DPO_MODEL_PATHS = {
    "saves": "/home/workspace/models/dpo_train/internal/saves",
    "export": "/home/workspace/models/dpo_train/internal/export",
}
TEST_PATHS = {
    "medical": "/home/workspace/tests/medical/choice",
    "general": "/home/workspace/tests/general",
}
EVALUATION_LOGS_PATH = "/home/workspace/tests/logs/benchmark"
BENCHMARK_DATASET_ALIASES = {
    "truthfulqa": {"truthfulqa", "truthfulqa-generation"},
}
GRPO_MODEL_PATH = "/home/workspace/models/grpo_train"
GRPO_DATA_ROOT = "/home/workspace/verl/examples/data_preprocess/data"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
GPU_CACHE_TTL_SECONDS = float(os.getenv("MEDFLOW_RESOURCE_GPU_CACHE_TTL_SECONDS", "30"))
try:
    GPU_BUSY_MEMORY_THRESHOLD_MB = max(
        0, int(os.getenv("MEDFLOW_GPU_PREFLIGHT_BUSY_MEMORY_MB", "200"))
    )
except ValueError:
    GPU_BUSY_MEMORY_THRESHOLD_MB = 200
GPU_QUERY_TIMEOUT_SECONDS = max(
    1,
    int(os.getenv("MEDFLOW_RESOURCE_GPU_QUERY_TIMEOUT_SECONDS", "10")),
)
GPU_REFRESH_INTERVAL_SECONDS = max(
    5,
    int(os.getenv("MEDFLOW_RESOURCE_GPU_REFRESH_INTERVAL_SECONDS", "10")),
)
GPU_SNAPSHOT_MAX_AGE_SECONDS = max(
    5,
    int(os.getenv("MEDFLOW_RESOURCE_GPU_SNAPSHOT_MAX_AGE_SECONDS", "30")),
)
GPU_RESERVATION_MAX_TTL_SECONDS = max(
    60,
    int(os.getenv("MEDFLOW_GPU_RESERVATION_MAX_TTL_SECONDS", "900")),
)
_GPU_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_GPU_DETECTED_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_GPU_LAST_SUCCESS: tuple[float, str, list[dict[str, Any]]] | None = None
_GPU_QUERY_LOCK = threading.Lock()
_GPU_REFRESH_STOP = threading.Event()
_GPU_REFRESH_THREAD: threading.Thread | None = None
_RESERVATION_LOCK = threading.Lock()
_RESERVATION_FILE = Path(
    os.getenv("MEDFLOW_GPU_RESERVATION_FILE", "/tmp/medflow_gpu_reservations.json")
)
_RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
_AGENT_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
_OUTER_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"


def _unique_registry_paths(*paths: Path) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


_TRAIN_PID_REGISTRIES = _unique_registry_paths(
    Path(os.getenv("MEDFLOW_TRAIN_PID_REGISTRY", str(_RUNTIME_DIR / "train_pid_registry.jsonl"))),
    _AGENT_RUNTIME_DIR / "train_pid_registry.jsonl",
    _OUTER_RUNTIME_DIR / "train_pid_registry.jsonl",
)
_EVALUATE_PID_REGISTRIES = _unique_registry_paths(
    Path(os.getenv("MEDFLOW_EVALUATE_PID_REGISTRY", str(_RUNTIME_DIR / "evaluate_pid_registry.jsonl"))),
    _AGENT_RUNTIME_DIR / "evaluate_pid_registry.jsonl",
    _OUTER_RUNTIME_DIR / "evaluate_pid_registry.jsonl",
)
_INFERENCE_PID_REGISTRIES = _unique_registry_paths(
    Path(os.getenv("MEDFLOW_INFERENCE_PID_REGISTRY", str(_RUNTIME_DIR / "inference_pid_registry.jsonl"))),
    _AGENT_RUNTIME_DIR / "inference_pid_registry.jsonl",
    _OUTER_RUNTIME_DIR / "inference_pid_registry.jsonl",
)
_BACKGROUND_TASK_REGISTRIES = _unique_registry_paths(
    Path(os.getenv("MEDFLOW_BACKGROUND_TASK_REGISTRY", str(_RUNTIME_DIR / "background_task_registry.jsonl"))),
    _AGENT_RUNTIME_DIR / "background_task_registry.jsonl",
    _OUTER_RUNTIME_DIR / "background_task_registry.jsonl",
)
INTERNAL_DATASET_FILES = frozenset(
    {
        "dataset_info.json",
        "preprocessing_audit.json",
        "preprocessing_summary.json",
        "score_audit.json",
        "score_filter_process.log",
        "score_progress.json",
        "score_summary.json",
    }
)

router = APIRouter(prefix="/internal/resources", tags=["internal-resources"])


class ContainerRequest(BaseModel):
    container: str | None = None


class DatasetRequest(ContainerRequest):
    datasetType: str
    datasetName: str


class DatasetFilterStatusRequest(ContainerRequest):
    outputFolder: str | None = None
    inputFolder: str | None = None
    threshold: float | None = None


class ModelRequest(ContainerRequest):
    modelType: str
    modelName: str
    modelPath: str | None = None


class UploadDatasetRequest(DatasetRequest):
    filename: str
    fileBase64: str


class MedicalTestRequest(ContainerRequest):
    filename: str


class UploadMedicalTestRequest(MedicalTestRequest):
    testType: str = "medical"
    fileBase64: str


class EvaluationResultDownloadRequest(ContainerRequest):
    folderPath: str
    filename: str


class GpuReservationPrepareRequest(BaseModel):
    reservationId: str
    gpuIndexes: list[int]
    expiresAt: str


class GpuReservationRequest(BaseModel):
    reservationId: str


class GpuReservationRenewRequest(BaseModel):
    reservationId: str
    expiresAt: str


class TrainingAllocationWriteRequest(BaseModel):
    reservationId: str
    allocation: dict[str, Any]
    container: str | None = None



class TrainingReservationStopProcessRequest(BaseModel):
    reservationId: str


class InferenceReservationStopServiceRequest(BaseModel):
    reservationId: str
    container: str | None = None
    resourceContext: dict[str, Any] | None = None


class EvaluationResultRequest(ContainerRequest):
    folderPath: str


class TrainingMetricsRequest(ContainerRequest):
    pid: str | None = None
    trainType: str | None = None
    historyLimit: int = 200
    timeWindowMinutes: int = 120


def _training_metrics(request: TrainingMetricsRequest) -> dict[str, Any]:
    from medflow_agent_tools.runlocal_monitor import monitor_training

    response = monitor_training(
        container_name=request.container,
        pid=request.pid,
        train_type=request.trainType,
        history_limit=request.historyLimit,
        time_window_minutes=request.timeWindowMinutes,
        allow_llm=False,
    )
    content = getattr(response, "content", None) or []
    if not content:
        raise HTTPException(502, "Training monitor returned empty output")
    first = content[0]
    text = (
        first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")
    )
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(502, "Training monitor returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(502, "Training monitor returned invalid payload")
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    logger.info(
        "[train-monitor] resource_api_training_metrics "
        "request_container=%s request_pid=%s request_train_type=%s "
        "status=%s latest_loss=%s history_count=%s "
        "wandb_select_reason=%s wandb_url_pending=%s",
        request.container,
        request.pid,
        request.trainType,
        data.get("status"),
        metrics.get("latest_loss"),
        metrics.get("history_count"),
        metrics.get("wandb_select_reason"),
        metrics.get("wandb_url_pending"),
    )
    return data


def _workflow_benchmark_name(context: dict[str, Any]) -> str:
    value = str(
        context.get("benchmark") or context.get("evaluation_dataset_name") or "2024"
    ).strip()
    return value or "2024"


def _workflow_benchmark_result_entry(context: dict[str, Any]) -> str:
    benchmark = _workflow_benchmark_name(context)
    command_name = "2024.json" if benchmark in {"2024", "2024.json"} else benchmark
    return f"查看推理基准测试{command_name}结果"


def _benchmark_dataset_aliases(benchmark: str) -> list[str]:
    normalized = _normalize_benchmark_name(benchmark)
    return sorted(BENCHMARK_DATASET_ALIASES.get(normalized, {normalized}))


def _normalize_benchmark_name(value: Any) -> str:
    text = os.path.basename(str(value or "").strip()).lower()
    if text.endswith(".json"):
        text = text[:-5]
    return text


def _workflow_benchmark_job_id(stage: dict[str, Any]) -> str:
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    job_id = str(result.get("benchmark_job_id") or "").strip()
    if job_id:
        return job_id
    for key in ("status_command", "message", "result"):
        match = re.search(
            r'benchmark_report\((?:job_id=)?["\']?([A-Za-z0-9_-]+)["\']?\)',
            str(result.get(key) or ""),
        )
        if match:
            return match.group(1)
    return ""


def _workflow_evaluation_container(context: dict[str, Any]) -> str:
    return _container(
        str(
            context.get("evaluation_container")
            or context.get("container")
            or os.getenv("AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER", "")
            or ""
        ).strip()
        or None
    )


def _query_structured_benchmark_status(
    context: dict[str, Any],
    stage: dict[str, Any],
) -> dict[str, Any]:
    benchmark = _workflow_benchmark_name(context)
    job_id = _workflow_benchmark_job_id(stage)
    container = _workflow_evaluation_container(context)
    expected_model = os.path.basename(
        str(context.get("published_model_path") or "").rstrip("/")
    )
    print(
        "[resource_api] workflow benchmark structured query: "
        f"benchmark={benchmark} job_id={job_id or '<none>'} "
        f"model={expected_model or '<none>'} container={container}",
        flush=True,
    )
    benchmark_aliases = _benchmark_dataset_aliases(benchmark)
    script = r"""
import json
import os
import re
import sys

root, job_id, benchmark, expected_model, aliases_json = sys.argv[1:6]
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
        "dataset": meta.get("dataset") or meta.get("benchmark") or benchmark,
        "model": meta.get("model"),
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
if os.path.isdir(root):
    if job_id and os.path.isdir(os.path.join(root, job_id)):
        folders = [os.path.join(root, job_id)]
    else:
        target_names = {norm_name(value) for value in benchmark_aliases}
        for name in os.listdir(root):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            meta = load(os.path.join(folder, "meta.json"))
            dataset = str(meta.get("dataset") or meta.get("benchmark") or "").strip()
            dataset_name = norm_name(dataset)
            if benchmark and dataset and not benchmark_matches(dataset_name, target_names):
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
else:
    print(json.dumps(payload_for(folders[0]), ensure_ascii=False))
"""
    output = _docker(
        container,
        [
            "python3",
            "-c",
            script,
            EVALUATION_LOGS_PATH,
            job_id,
            benchmark,
            expected_model,
            json.dumps(benchmark_aliases),
        ],
        timeout=30,
    ).stdout
    payload = json.loads(output or "{}")
    if not isinstance(payload, dict):
        raise HTTPException(502, "Invalid benchmark status payload")
    payload["result_entry"] = _workflow_benchmark_result_entry(context)
    if payload.get("benchmark_job_id"):
        payload["status_command"] = f'benchmark_report("{payload["benchmark_job_id"]}")'
        payload["stop_command"] = f'benchmark_stop("{payload["benchmark_job_id"]}")'
        payload["log_command"] = payload["status_command"]
    print(
        "[resource_api] workflow benchmark structured query result: "
        f"benchmark={benchmark} job_id={payload.get('benchmark_job_id') or '<none>'} "
        f"status={payload.get('status') or payload.get('benchmark_status') or '<none>'} "
        f"folder={payload.get('folder_path') or '<none>'} "
        f"debug={json.dumps(payload.get('debug') or {}, ensure_ascii=False)[:1000]}",
        flush=True,
    )
    return payload


def _maybe_refresh_workflow_benchmark_status(
    db_path: Path,
    workflow: dict[str, Any],
    stages: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow_id = str(workflow.get("workflow_id") or "")
    if (
        workflow.get("status") != "running"
        or workflow.get("current_stage") != "benchmark"
    ):
        print(
            "[resource_api] workflow benchmark reconcile skipped: "
            f"workflow_id={workflow_id or '<unknown>'} "
            f"workflow_status={workflow.get('status')} current_stage={workflow.get('current_stage')}",
            flush=True,
        )
        return stages, workflow
    stage = stages.get("benchmark")
    if not isinstance(stage, dict) or stage.get("status") not in {"running", "timeout"}:
        print(
            "[resource_api] workflow benchmark reconcile skipped: "
            f"workflow_id={workflow_id or '<unknown>'} benchmark_stage_status="
            f"{stage.get('status') if isinstance(stage, dict) else '<invalid>'}",
            flush=True,
        )
        return stages, workflow
    print(
        "[resource_api] workflow benchmark reconcile started: "
        f"workflow_id={workflow_id} benchmark={_workflow_benchmark_name(context)} "
        f"stage_status={stage.get('status')} updated_at={stage.get('updated_at')}",
        flush=True,
    )
    try:
        result = _query_structured_benchmark_status(context, stage)
    except Exception as exc:
        print(
            "[resource_api] workflow benchmark reconcile query failed: "
            f"workflow_id={workflow_id} error={exc}",
            flush=True,
        )
        stage["status"] = "timeout"
        stage["message"] = "本次状态查询超时，任务可能仍在运行，将继续自动刷新"
        stage["last_poll_error"] = str(exc)
        stage["updated_at"] = datetime.now(timezone.utc).timestamp()
        return stages, workflow

    status = str(result.get("status") or result.get("benchmark_status") or "").lower()
    if status == "not_found":
        print(
            "[resource_api] workflow benchmark reconcile no matching benchmark record: "
            f"workflow_id={workflow_id} benchmark={_workflow_benchmark_name(context)}",
            flush=True,
        )
        return stages, workflow
    if status in {"finished", "completed", "done"}:
        stage["status"] = "finished"
        stage["result"] = result
        stage["updated_at"] = datetime.now(timezone.utc).timestamp()
        stage.pop("last_poll_error", None)
        stage.pop("error", None)
        workflow["status"] = "finished"
        workflow["error"] = None
    elif status in {"running", "processing"}:
        stage["status"] = "running"
        stage["result"] = result
        stage["updated_at"] = datetime.now(timezone.utc).timestamp()
        stage.pop("last_poll_error", None)
        stage.pop("error", None)
        workflow["status"] = "running"
        workflow["error"] = None
    else:
        print(
            "[resource_api] workflow benchmark reconcile ignored unsupported status: "
            f"workflow_id={workflow_id} status={status or '<empty>'}",
            flush=True,
        )
        return stages, workflow

    workflow["updated_at"] = datetime.now(timezone.utc).timestamp()
    with sqlite3.connect(db_path, timeout=30) as connection:
        connection.execute(
            """
            UPDATE workflows
            SET status=?, stages_json=?, error=?, updated_at=?
            WHERE workflow_id=?
            """,
            (
                workflow["status"],
                json.dumps(stages, ensure_ascii=False),
                workflow.get("error"),
                workflow["updated_at"],
                workflow["workflow_id"],
            ),
        )
    if workflow["status"] == "finished":
        # resource_api writes terminal benchmark state directly to SQLite.
        # Enqueue a one-shot event so the Agent session can push the final
        # workflow_finished protocol to Studio.
        with sqlite3.connect(db_path, timeout=30) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO workflow_events(
                    event_id, workflow_id, user_id, event_type, payload_json,
                    status, claimed_by, claimed_until, error, created_at, updated_at
                ) VALUES (?, ?, ?, 'notify_terminal', '{}', 'pending', NULL, NULL, NULL, ?, ?)
                """,
                (
                    f"{workflow_id}:notify_terminal",
                    workflow["workflow_id"],
                    workflow["user_id"],
                    workflow["updated_at"],
                    workflow["updated_at"],
                ),
            )

    print(
        "[resource_api] workflow benchmark reconcile saved: "
        f"workflow_id={workflow_id} workflow_status={workflow['status']} "
        f"benchmark_status={stage.get('status')} job_id="
        f"{(stage.get('result') if isinstance(stage.get('result'), dict) else {}).get('benchmark_job_id') or '<none>'}",
        flush=True,
    )
    return stages, workflow


def _workflow_status(workflow_id: str) -> dict[str, Any]:
    db_path = Path(
        os.getenv("AGENT3_WORKFLOW_DB_PATH", "./data/workflows/workflows.db")
    ).resolve()
    if not db_path.exists():
        raise HTTPException(404, "Workflow database not found")

    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM workflows WHERE workflow_id=?",
            (workflow_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise HTTPException(404, "Workflow not found")

    workflow = dict(row)
    stages = json.loads(workflow["stages_json"])
    context = json.loads(workflow["context_json"])
    stages, workflow = _maybe_refresh_workflow_benchmark_status(
        db_path, workflow, stages, context
    )
    metric_keys = {
        "container",
        "container_name",
        "pid",
        "train_type",
        "train_type_public",
        "train_type_text",
        "output_dir",
        "latest_loss",
        "latest_learning_rate",
        "latest_epoch",
        "latest_step",
        "progress_percent",
        "current_step",
        "total_steps",
        "elapsed_time",
        "remaining_time",
        "pid_alive",
        "training_process_exists",
        "history_count",
        "history_limit",
        "loss_source",
        "last_update_time",
        "stale",
        "stale_minutes",
        "wandb_url",
        "wandb_url_pending",
        "wandb_mode",
        "wandb_url_source",
        "metrics_log_path",
        "metrics_log_source",
        "output_log_success",
        "error_reason",
        "error_detail",
    }
    public_stages = {}
    for name, stage in stages.items():
        public = {
            key: value
            for key, value in stage.items()
            if key not in {"metrics", "debug", "history"}
        }
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

    train_type = context.get("train_type")
    train_type_text = {
        "lora": "lora批量训练",
        "full": "全参批量训练",
        "enhanced": "增强训练",
        "grpo": "grpo训练",
    }.get(train_type, train_type)
    benchmark_entry = context.get("benchmark") or "2024"
    return {
        "version": "1.0",
        "type": "workflow_status",
        "agent": "workflow_monitor",
        "workflowId": workflow["workflow_id"],
        "workflowStatus": workflow["status"],
        "currentStage": workflow["current_stage"],
        "datasetRef": workflow["dataset_ref"],
        "trainType": train_type,
        "trainTypeText": train_type_text,
        "stages": public_stages,
        "benchmark": context.get("benchmark"),
        "evaluationDatasetName": context.get("evaluation_dataset_name"),
        "benchmarkResultEntry": (
            "查看2024基准评测结果"
            if benchmark_entry in ("2024", "2024.json")
            else f"查看推理基准测试{benchmark_entry}结果"
        ),
        "error": workflow.get("error"),
        "source": "workflow_db",
        "confidence": 1.0,
        "valid": True,
    }


def _node_meta(status: str = "online") -> dict[str, str]:
    return {
        "nodeId": os.getenv("MEDFLOW_RESOURCE_NODE_ID", os.getenv("HOSTNAME", "local")),
        "nodeName": os.getenv(
            "MEDFLOW_RESOURCE_NODE_NAME", os.getenv("HOSTNAME", "Local Node")
        ),
        "status": status,
        "collectedAt": datetime.now(timezone.utc).isoformat(),
    }


def _response(data: Any, status: str = "online") -> dict[str, Any]:
    return {**_node_meta(status), "data": data}


def _authorize(authorization: str | None = Header(default=None)) -> None:
    token = os.getenv("MEDFLOW_RESOURCE_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(503, "MEDFLOW_RESOURCE_API_TOKEN is not configured")
    if authorization != f"Bearer {token}":
        raise HTTPException(401, "Invalid resource API token")


def _safe_name(value: str, label: str) -> str:
    value = value.strip()
    if not SAFE_NAME.fullmatch(value):
        raise HTTPException(400, f"Invalid {label}")
    return value


def _container(value: str | None) -> str:
    return _safe_name(
        value or os.getenv("AGENT3_DEFAULT_DOCKER_CONTAINER", "agent3"),
        "container",
    )


def _dataset_path(dataset_type: str, dataset_name: str) -> str:
    if dataset_type not in DATASET_PATHS:
        raise HTTPException(400, "Invalid dataset type")
    return f"{DATASET_PATHS[dataset_type]}/{_safe_name(dataset_name, 'dataset name')}"


def _evaluation_result_path(folder_path: str, filename: str) -> tuple[str, str]:
    folder_name = Path(folder_path).name
    if not re.fullmatch(r"[0-9]+_[a-f0-9]+", folder_name):
        raise HTTPException(400, "Invalid evaluation result folder")
    safe_filename = _safe_name(filename, "filename")
    return f"{EVALUATION_LOGS_PATH}/{folder_name}/{safe_filename}", safe_filename


def _filter_status_folder(request: DatasetFilterStatusRequest) -> str:
    if request.outputFolder:
        folder = request.outputFolder
    elif request.inputFolder and request.threshold is not None:
        base = str(request.inputFolder).rstrip("/")
        folder = f"{base}_{request.threshold:g}_score_filter"
    else:
        raise HTTPException(400, "必须提供 outputFolder 或 inputFolder+threshold")

    normalized = Path(folder).as_posix()
    if not normalized.startswith("/") or ".." in normalized.split("/"):
        raise HTTPException(400, "Invalid filter status folder")
    if not any(normalized.startswith(prefix) for prefix in DATASET_PATHS.values()):
        raise HTTPException(400, "Filter status folder must be under a dataset root")
    return normalized


def _run(
    args: list[str],
    timeout: int = 30,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=text,
            timeout=timeout,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise HTTPException(
            502, stderr.strip() or f"Command failed: {args[0]}"
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(502, str(exc)) from exc


def _docker(container: str, args: list[str], timeout: int = 30, text: bool = True):
    return _run(["docker", "exec", container, *args], timeout=timeout, text=text)


def _docker_bytes(container: str, args: list[str], timeout: int = 60) -> bytes:
    return _docker(container, args, timeout=timeout, text=False).stdout


def _file_base64_response(filename: str, payload: bytes) -> dict[str, str]:
    return {
        "filename": filename,
        "fileBase64": base64.b64encode(payload).decode("ascii"),
    }


def _container_status(container: str) -> dict[str, Any]:
    target = _container(container)
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", target],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(502, str(exc)) from exc
    if inspect.returncode == 0:
        return {
            "container": target,
            "exists": True,
            "running": inspect.stdout.strip() == "true",
        }
    return {"container": target, "exists": False, "running": False}


def _list_dirs(container: str, base_path: str) -> list[dict[str, Any]]:
    script = (
        'for dir in "$1"/*; do '
        '[ -d "$dir" ] || continue; '
        'name=$(basename "$dir"); '
        'mtime=$(date -r "$dir" -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo ""); '
        'size=$(du -sh "$dir" 2>/dev/null | awk "{print \\$1}"); '
        'printf "%s\\t%s\\t%s\\n" "$name" "$mtime" "$size"; '
        "done"
    )
    output = _docker(container, ["sh", "-c", script, "resource-list", base_path]).stdout
    result = []
    for line in output.splitlines():
        name, _, remainder = line.partition("\t")
        created_at, _, size = remainder.partition("\t")
        if name:
            result.append(
                {
                    "name": name,
                    "path": base_path,
                    "createdAt": created_at,
                    "size": size,
                }
            )
    return result


def _list_model_checkpoints(container: str, model_dir: str) -> list[str]:
    script = (
        'if [ ! -d "$1" ]; then exit 0; fi; '
        'find "$1" -maxdepth 1 -type d -name "checkpoint-*" '
        '-printf "%f\\n" 2>/dev/null | sort -V'
    )
    output = _docker(
        container,
        ["sh", "-c", script, "checkpoint-list", model_dir],
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def _datasets(container: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for kind, path in DATASET_PATHS.items():
        entries = _list_dirs(container, path)
        visible_entries = []
        for entry in entries:
            entry["type"] = kind
            files = _docker(
                container,
                [
                    "sh",
                    "-c",
                    'find "$1" -maxdepth 1 -type f -printf "%f\\n" 2>/dev/null',
                    "files",
                    f"{path}/{entry['name']}",
                ],
            ).stdout.splitlines()
            entry["files"] = [name for name in files if _is_dataset_data_file(name)]
            if not entry["files"]:
                continue
            entry["filePreviews"] = []
            visible_entries.append(entry)
        result[kind] = visible_entries
    return result


def _is_dataset_data_file(filename: str) -> bool:
    return filename not in INTERNAL_DATASET_FILES and filename.lower().endswith((".json", ".jsonl"))


def _extract_dataset_archive(
    archive: Path,
    extract_dir: Path,
) -> Path:
    extract_root = extract_dir.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            resolved = (extract_dir / member.name).resolve()
            if not str(resolved).startswith(f"{extract_root}{os.sep}"):
                raise HTTPException(400, "Unsafe archive path")
            if member.issym() or member.islnk():
                raise HTTPException(400, "Dataset archive must not contain links")
        tar.extractall(extract_dir)

    children = [
        child
        for child in extract_dir.iterdir()
        if child.name not in {"__MACOSX", ".DS_Store"}
    ]
    content_root = (
        children[0] if len(children) == 1 and children[0].is_dir() else extract_dir
    )
    if not any(
        child.is_file() and _is_dataset_data_file(child.name)
        for child in content_root.iterdir()
    ):
        raise HTTPException(
            400,
            "Dataset archive must contain a .json or .jsonl file at its top level",
        )
    return content_root


def _models(container: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for kind, path in MODEL_PATHS.items():
        if kind == "daily_trained":
            daily_entries: list[dict[str, Any]] = []
            for dpo_path, merged in (
                (DPO_MODEL_PATHS["saves"], False),
                (DPO_MODEL_PATHS["export"], True),
            ):
                for entry in _list_dirs(container, dpo_path):
                    entry["type"] = kind
                    entry["merged"] = merged
                    entry["checkpoints"] = _list_model_checkpoints(
                        container,
                        f"{dpo_path}/{entry['name']}",
                    )
                    daily_entries.append(entry)
            result[kind] = daily_entries
            continue

        entries = _list_dirs(container, path)
        for entry in entries:
            entry["type"] = kind
            entry["merged"] = entry["name"].endswith("_merged")
            if kind == "batch_trained":
                entry["checkpoints"] = _list_model_checkpoints(
                    container,
                    f"{path}/{entry['name']}",
                )
        result[kind] = entries
    return result


def _clone_gpus(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(gpu) for gpu in gpus]


def _apply_current_reservations(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = _clone_gpus(gpus)
    reserved = _reserved_gpu_indexes()
    for gpu in result:
        base_available = bool(gpu.get("available", False))
        gpu["reserved"] = gpu["index"] in reserved
        gpu["available"] = bool(base_available and not gpu["reserved"])
    return result


def _annotate_gpu_snapshot(
    gpus: list[dict[str, Any]],
    *,
    collected_at: str,
    age_seconds: float,
    stale: bool,
    error: str | None = None,
) -> list[dict[str, Any]]:
    result = _apply_current_reservations(gpus)
    for gpu in result:
        gpu["collectedAt"] = collected_at
        gpu["ageSeconds"] = round(age_seconds, 3)
        gpu["maxAgeSeconds"] = GPU_SNAPSHOT_MAX_AGE_SECONDS
        if stale:
            gpu["stale"] = True
        if error:
            gpu["error"] = error
    return result


def _gpu_unavailable_status(error: str) -> list[dict[str, Any]]:
    collected_at = datetime.now(timezone.utc).isoformat()
    return [
        {
            "available": False,
            "name": "GPU status unavailable",
            "index": -1,
            "memoryTotal": 0,
            "memoryUsed": 0,
            "memoryFree": 0,
            "utilization": 0,
            "temperature": 0,
            "processes": [],
            "stale": True,
            "collectedAt": collected_at,
            "ageSeconds": None,
            "maxAgeSeconds": GPU_SNAPSHOT_MAX_AGE_SECONDS,
            "error": error,
        }
    ]

def _expire_gpu_cache() -> None:
    global _GPU_CACHE
    _GPU_CACHE = None


def _refresh_gpus_once() -> bool:
    if not _GPU_QUERY_LOCK.acquire(blocking=False):
        print(
            "[resource_api] GPU refresh skipped: previous refresh still running",
            flush=True,
        )
        return False
    try:
        print("[resource_api] GPU refresh started", flush=True)
        _gpus_locked(force=True)
        snapshot = _GPU_LAST_SUCCESS
        gpu_count = len(snapshot[2]) if snapshot else 0
        print(
            f"[resource_api] GPU refresh succeeded: gpu_count={gpu_count}", flush=True
        )
        return True
    except Exception as exc:
        print(f"[resource_api] Background GPU refresh failed: {exc}", flush=True)
        return False
    finally:
        _GPU_QUERY_LOCK.release()


def _refresh_gpus_in_background() -> None:
    threading.Thread(
        target=_refresh_gpus_once, name="medflow-gpu-refresh-once", daemon=True
    ).start()


def start_gpu_background_refresh() -> None:
    global _GPU_REFRESH_THREAD
    if _GPU_REFRESH_THREAD and _GPU_REFRESH_THREAD.is_alive():
        return
    _GPU_REFRESH_STOP.clear()

    def loop() -> None:
        print(
            f"[resource_api] GPU background refresh started; interval={GPU_REFRESH_INTERVAL_SECONDS}s",
            flush=True,
        )
        while not _GPU_REFRESH_STOP.is_set():
            _refresh_gpus_once()
            _GPU_REFRESH_STOP.wait(GPU_REFRESH_INTERVAL_SECONDS)

    _GPU_REFRESH_THREAD = threading.Thread(
        target=loop, name="medflow-gpu-refresh-loop", daemon=True
    )
    _GPU_REFRESH_THREAD.start()


def stop_gpu_background_refresh() -> None:
    _GPU_REFRESH_STOP.set()


def _gpus() -> list[dict[str, Any]]:
    global _GPU_CACHE
    now = monotonic()
    cached = _GPU_CACHE
    if cached and cached[0] > now:
        print(f"[resource_api] /gpus cache hit: gpu_count={len(cached[1])}", flush=True)
        return cached[1]

    snapshot = _GPU_LAST_SUCCESS
    if not snapshot:
        _refresh_gpus_in_background()
        print(
            "[resource_api] /gpus no snapshot; scheduled background refresh", flush=True
        )
        result = _gpu_unavailable_status(
            "资源节点 GPU 状态尚未生成，正在后台刷新，请稍后重试或检查节点探测服务"
        )
        _GPU_CACHE = (monotonic() + min(GPU_CACHE_TTL_SECONDS, 5), result)
        return result

    collected_monotonic, collected_at, gpus = snapshot
    age_seconds = monotonic() - collected_monotonic
    if age_seconds > GPU_SNAPSHOT_MAX_AGE_SECONDS:
        _refresh_gpus_in_background()
        print(
            f"[resource_api] /gpus stale snapshot: age={age_seconds:.3f}s "
            f"max={GPU_SNAPSHOT_MAX_AGE_SECONDS}s collectedAt={collected_at}",
            flush=True,
        )
        result = _annotate_gpu_snapshot(
            gpus,
            collected_at=collected_at,
            age_seconds=age_seconds,
            stale=True,
            error="资源节点 GPU 状态过期，正在后台刷新",
        )
        _GPU_CACHE = (monotonic() + min(GPU_CACHE_TTL_SECONDS, 5), result)
        return result

    result = _annotate_gpu_snapshot(
        gpus,
        collected_at=collected_at,
        age_seconds=age_seconds,
        stale=False,
    )
    _GPU_CACHE = (monotonic() + GPU_CACHE_TTL_SECONDS, result)
    available_count = sum(1 for gpu in result if gpu.get("available", False))
    print(
        f"[resource_api] /gpus snapshot served: gpu_count={len(result)} "
        f"available={available_count} age={age_seconds:.3f}s collectedAt={collected_at}",
        flush=True,
    )
    return result


def _gpus_snapshot(
    max_age_seconds: int = GPU_SNAPSHOT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    snapshot = _GPU_LAST_SUCCESS
    if not snapshot:
        _refresh_gpus_in_background()
        print(
            "[resource_api] /gpus/snapshot no snapshot; scheduled background refresh",
            flush=True,
        )
        raise HTTPException(
            503, "资源节点 GPU 状态尚未生成，请稍后重试或检查节点探测服务"
        )
    collected_monotonic, collected_at, gpus = snapshot
    age_seconds = monotonic() - collected_monotonic
    if age_seconds > max_age_seconds:
        _refresh_gpus_in_background()
        print(
            f"[resource_api] /gpus/snapshot stale: age={age_seconds:.3f}s "
            f"max={max_age_seconds}s collectedAt={collected_at}",
            flush=True,
        )
        raise HTTPException(
            503,
            "资源节点 GPU 状态过期，请稍后重试或检查节点探测服务",
        )
    print(
        f"[resource_api] /gpus/snapshot served: gpu_count={len(gpus)} "
        f"age={age_seconds:.3f}s collectedAt={collected_at}",
        flush=True,
    )
    return {
        "gpus": _apply_current_reservations(gpus),
        "collectedAt": collected_at,
        "ageSeconds": round(age_seconds, 3),
        "maxAgeSeconds": max_age_seconds,
    }


def _gpus_locked(force: bool = False) -> list[dict[str, Any]]:
    global _GPU_CACHE, _GPU_DETECTED_CACHE, _GPU_LAST_SUCCESS
    started_at = monotonic()
    if not force and _GPU_CACHE and _GPU_CACHE[0] > started_at:
        return _GPU_CACHE[1]

    query_args = [
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    configured_path = os.getenv("NVIDIA_SMI_PATH", "").strip()
    host_candidates = [
        configured_path,
        shutil.which("nvidia-smi") or "",
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
    ]
    output = ""
    errors: list[str] = []

    host_path = next(
        (
            path
            for path in dict.fromkeys(path for path in host_candidates if path)
            if Path(path).is_file()
        ),
        "",
    )
    if host_path:
        try:
            output = _run(
                [host_path, *query_args],
                timeout=GPU_QUERY_TIMEOUT_SECONDS,
            ).stdout
        except HTTPException as exc:
            errors.append(f"{host_path}: {exc.detail}")

    if not output:
        try:
            output = _docker(
                _container(None),
                ["nvidia-smi", *query_args],
                timeout=GPU_QUERY_TIMEOUT_SECONDS,
            ).stdout
        except HTTPException as exc:
            errors.append(f"docker exec {_container(None)} nvidia-smi: {exc.detail}")

    if not output:
        # A node without a visible GPU is still a healthy resource node.
        detail = "; ".join(errors)
        print(f"[resource_api] GPU detection unavailable: {detail}")
        if _GPU_LAST_SUCCESS is not None and errors:
            raise HTTPException(503, f"GPU detection unavailable: {detail}")
        collected_at = datetime.now(timezone.utc).isoformat()
        _GPU_LAST_SUCCESS = (monotonic(), collected_at, [])
        _GPU_DETECTED_CACHE = (monotonic() + GPU_CACHE_TTL_SECONDS, [])
        _GPU_CACHE = (monotonic() + GPU_CACHE_TTL_SECONDS, [])
        return []

    detected_result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        total, used = int(parts[2]), int(parts[3])
        detected_result.append(
            {
                "available": used < GPU_BUSY_MEMORY_THRESHOLD_MB,
                "name": parts[1],
                "index": int(parts[0]),
                "memoryTotal": total,
                "memoryUsed": used,
                "memoryFree": total - used,
                "utilization": int(parts[4]),
                "temperature": int(parts[5]),
                "processes": [],
            }
        )
    collected_at = datetime.now(timezone.utc).isoformat()
    _GPU_LAST_SUCCESS = (monotonic(), collected_at, detected_result)
    _GPU_DETECTED_CACHE = (monotonic() + GPU_CACHE_TTL_SECONDS, detected_result)
    result = _apply_current_reservations(detected_result)
    elapsed = monotonic() - started_at
    if elapsed >= 1:
        print(
            f"[resource_api] GPU detection completed in {elapsed:.2f}s "
            f"via {'host' if host_path and not errors else 'docker'}",
            flush=True,
        )
    _GPU_CACHE = (monotonic() + GPU_CACHE_TTL_SECONDS, result)
    return result


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validate_reservation_expiry(value: str) -> str:
    try:
        expires_at = _parse_timestamp(value)
    except ValueError as exc:
        raise HTTPException(400, "expiresAt must be an ISO-8601 timestamp") from exc
    seconds = (expires_at - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        raise HTTPException(400, "expiresAt must be in the future")
    if seconds > GPU_RESERVATION_MAX_TTL_SECONDS:
        raise HTTPException(
            400,
            f"expiresAt exceeds maximum TTL of {GPU_RESERVATION_MAX_TTL_SECONDS} seconds",
        )
    return expires_at.isoformat()


def _load_reservations() -> dict[str, dict[str, Any]]:
    reservations = _load_raw_reservations()
    now = datetime.now(timezone.utc)
    active: dict[str, dict[str, Any]] = {}
    for key, value in reservations.items():
        if not isinstance(value, dict) or not value.get("expiresAt"):
            continue
        try:
            if _parse_timestamp(str(value["expiresAt"])) > now:
                active[key] = value
        except ValueError:
            continue
    return active


def _load_raw_reservations() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_RESERVATION_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _reservation_status() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    active: dict[str, dict[str, Any]] = {}
    expired: dict[str, dict[str, Any]] = {}
    raw = _load_raw_reservations()
    for key, value in raw.items():
        if not isinstance(value, dict) or not value.get("expiresAt"):
            continue
        try:
            if _parse_timestamp(str(value["expiresAt"])) > now:
                active[key] = value
            else:
                expired[key] = value
        except ValueError:
            expired[key] = value
    if expired:
        _save_reservations(active)
    gpu_owners = {
        str(gpu): reservation_id
        for reservation_id, reservation in active.items()
        for gpu in reservation.get("gpuIndexes", [])
    }
    return {
        "reservationFile": str(_RESERVATION_FILE),
        "active": active,
        "expiredCleaned": expired,
        "gpuOwners": gpu_owners,
        "activeCount": len(active),
        "expiredCleanedCount": len(expired),
    }


def _save_reservations(reservations: dict[str, dict[str, Any]]) -> None:
    _RESERVATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = _RESERVATION_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(reservations, ensure_ascii=False), encoding="utf-8")
    temp.replace(_RESERVATION_FILE)


def _reserved_gpu_indexes(except_reservation_id: str | None = None) -> set[int]:
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        _save_reservations(reservations)
        return {
            int(gpu)
            for reservation_id, reservation in reservations.items()
            if reservation_id != except_reservation_id
            for gpu in reservation.get("gpuIndexes", [])
        }


def _remove_reservation(reservation_id: str) -> bool:
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        existed = reservations.pop(reservation_id, None) is not None
        _save_reservations(reservations)
        _expire_gpu_cache()
        return existed


@router.get("/health", dependencies=[Depends(_authorize)])
def health():
    return _response({"ok": True})


@router.get("/workflow-status", dependencies=[Depends(_authorize)])
def workflow_status(workflowId: str):
    return _response(_workflow_status(workflowId))


@router.post("/training-metrics", dependencies=[Depends(_authorize)])
def training_metrics(request: TrainingMetricsRequest):
    return _response(_training_metrics(request))


@router.get("/gpus", dependencies=[Depends(_authorize)])
def gpus():
    return _response(_gpus())


@router.get("/gpus/snapshot", dependencies=[Depends(_authorize)])
def gpus_snapshot():
    return _response(_gpus_snapshot())


@router.get("/gpu-reservations/status", dependencies=[Depends(_authorize)])
def gpu_reservation_status():
    with _RESERVATION_LOCK:
        status = _reservation_status()
    return _response(status)


@router.post("/gpu-reservations/prepare", dependencies=[Depends(_authorize)])
def prepare_gpu_reservation(request: GpuReservationPrepareRequest):
    gpu_indexes = sorted(set(request.gpuIndexes))
    if not request.reservationId.strip() or not gpu_indexes:
        raise HTTPException(400, "reservationId and gpuIndexes are required")
    expires_at = _validate_reservation_expiry(request.expiresAt)
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        existing = reservations.get(request.reservationId)
        if existing and sorted(existing.get("gpuIndexes", [])) == gpu_indexes:
            existing["expiresAt"] = expires_at
            existing["updatedAt"] = datetime.now(timezone.utc).isoformat()
            _save_reservations(reservations)
            return _response(
                {
                    "reservationId": request.reservationId,
                    "gpuIndexes": gpu_indexes,
                    "status": existing.get("status", "prepared"),
                }
            )
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        reserved = {
            int(gpu)
            for reservation_id, reservation in reservations.items()
            if reservation_id != request.reservationId
            for gpu in reservation.get("gpuIndexes", [])
        }
        conflicts = sorted(reserved.intersection(gpu_indexes))
        if conflicts:
            raise HTTPException(409, f"GPU(s) already reserved: {conflicts}")
        reservations[request.reservationId] = {
            "gpuIndexes": gpu_indexes,
            "expiresAt": expires_at,
            "status": "prepared",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        _save_reservations(reservations)
        _expire_gpu_cache()
    return _response(
        {
            "reservationId": request.reservationId,
            "gpuIndexes": gpu_indexes,
            "status": "prepared",
        }
    )


@router.post("/gpu-reservations/commit", dependencies=[Depends(_authorize)])
def commit_gpu_reservation(request: GpuReservationRequest):
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        reservation = reservations.get(request.reservationId)
        if not reservation:
            raise HTTPException(404, "GPU reservation not found or expired")
        reservation["status"] = "committed"
        reservation["updatedAt"] = datetime.now(timezone.utc).isoformat()
        _save_reservations(reservations)
    return _response({"reservationId": request.reservationId, "status": "committed"})


@router.post("/gpu-reservations/renew", dependencies=[Depends(_authorize)])
def renew_gpu_reservation(request: GpuReservationRenewRequest):
    expires_at = _validate_reservation_expiry(request.expiresAt)
    with _RESERVATION_LOCK:
        reservations = _load_reservations()
        reservation = reservations.get(request.reservationId)
        if not reservation:
            raise HTTPException(404, "GPU reservation not found or expired")
        reservation["expiresAt"] = expires_at
        reservation["updatedAt"] = datetime.now(timezone.utc).isoformat()
        _save_reservations(reservations)
    return _response({"reservationId": request.reservationId, "expiresAt": expires_at})


@router.post("/gpu-reservations/rollback", dependencies=[Depends(_authorize)])
def rollback_gpu_reservation(request: GpuReservationRequest):
    return _response(
        {
            "reservationId": request.reservationId,
            "released": _remove_reservation(request.reservationId),
        }
    )



def _runtime_record_started_at(record: dict[str, Any]) -> float:
    for key in ("started_at_ts", "started_at"):
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return 0.0


def _runtime_registry_specs() -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for label, paths in (
        ("train_pid", _TRAIN_PID_REGISTRIES),
        ("evaluate_pid", _EVALUATE_PID_REGISTRIES),
        ("inference_pid", _INFERENCE_PID_REGISTRIES),
        ("background_task", _BACKGROUND_TASK_REGISTRIES),
    ):
        specs.extend((label, path) for path in paths)
    return specs


def _record_reservation_id(record: dict[str, Any]) -> str:
    direct = str(record.get("reservationId") or record.get("reservation_id") or "").strip()
    if direct:
        return direct
    env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
    return str(env_vars.get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip()


def _find_runtime_process_by_reservation(
    reservation_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    normalized = reservation_id.strip()
    if not normalized:
        raise HTTPException(400, "reservationId is required")
    checked: list[dict[str, str]] = []
    matches: list[dict[str, Any]] = []
    for registry_type, path in _runtime_registry_specs():
        checked.append({"type": registry_type, "path": str(path), "exists": str(path.exists()).lower()})
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if _record_reservation_id(record) != normalized:
                        continue
                    enriched = dict(record)
                    enriched["registryType"] = registry_type
                    enriched["registryPath"] = str(path)
                    matches.append(enriched)
        except OSError as exc:
            raise HTTPException(502, f"Failed to read Runtime process registry {path}: {exc}") from exc
    if not matches:
        return None, checked
    matches.sort(key=_runtime_record_started_at, reverse=True)
    return matches[0], checked


def _assigned_gpu_indexes_from_record(record: dict[str, Any]) -> set[int]:
    env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
    raw = str(
        record.get("assignedGpus")
        or record.get("assigned_gpus")
        or env_vars.get("MEDFLOW_ASSIGNED_GPUS")
        or env_vars.get("CUDA_VISIBLE_DEVICES")
        or ""
    )
    indexes: set[int] = set()
    for item in re.split(r"[,\s]+", raw):
        item = item.strip()
        if not item:
            continue
        try:
            indexes.add(int(item))
        except ValueError:
            continue
    return indexes


def _reservation_gpus_idle(record: dict[str, Any]) -> bool:
    assigned = _assigned_gpu_indexes_from_record(record)
    if not assigned:
        return False
    gpus = _gpus_locked(force=True)
    matched = [gpu for gpu in gpus if int(gpu.get("index", -1)) in assigned]
    if not matched:
        return False
    return all(int(gpu.get("memoryUsed") or 0) < GPU_BUSY_MEMORY_THRESHOLD_MB for gpu in matched)


def _docker_pid_snapshot(container: str, pid: str) -> list[str]:
    script = r'''
root="$1"
if ! printf "%s" "$root" | grep -Eq '^[0-9]+$'; then exit 0; fi
ps -eo pid=,ppid=,pgid=,stat=,args= 2>/dev/null | awk -v root="$root" '
    {
        line=$0; pid=$1; ppid=$2; pgid=$3; stat=$4; command=line
        sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]*/, "", command)
        parent[pid]=ppid; group[pid]=pgid; state[pid]=stat; cmd[pid]=command
        if (pid == root) root_group=pgid
    }
    END {
        changed=1
        while (changed) {
            changed=0
            for (pid in parent) {
                if (pid == root || live[parent[pid]]) {
                    if (!live[pid]) { live[pid]=1; changed=1 }
                }
            }
        }
        if (root_group != "") {
            for (pid in group) {
                if (group[pid] == root_group) live[pid]=1
            }
        }
        for (pid in live) {
            if (state[pid] !~ /Z/) print pid " " cmd[pid]
        }
    }
'
'''
    process = subprocess.run(
        ["docker", "exec", container, "sh", "-c", script, "sh", pid],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if process.returncode not in (0, 1):
        raise HTTPException(502, process.stderr.strip() or "Failed to inspect process")
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def _stop_docker_pid(container: str, pid: str) -> dict[str, Any]:
    before = _docker_pid_snapshot(container, pid)
    if not before:
        return {
            "container": container,
            "pid": pid,
            "alreadyExited": True,
            "stopped": False,
            "remainingPids": [],
            "message": f"PID {pid} 不存在或已退出",
        }

    stop_script = r'''
pid="$1"
collect_descendants() {
    parent="$1"
    children=$(ps -eo pid=,ppid= | awk -v p="$parent" '$2 == p {print $1}')
    for child in $children; do
        collect_descendants "$child"
        echo "$child"
    done
}
pgid=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
pgid_pids=""
if [ -n "$pgid" ]; then
    pgid_pids=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2 == g {print $1}')
fi
targets=$(printf "%s\n%s\n%s\n" "$(collect_descendants "$pid")" "$pgid_pids" "$pid" | awk 'NF && !seen[$1]++ {print $1}')
for target in $targets; do
    if [ "$target" != "$$" ] && [ "$target" != "$PPID" ] && ps -p "$target" >/dev/null 2>&1; then
        stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        kill -TERM "$target" 2>/dev/null || true
    fi
done
sleep 3
for target in $targets; do
    if [ "$target" != "$$" ] && [ "$target" != "$PPID" ] && ps -p "$target" >/dev/null 2>&1; then
        stat=$(ps -p "$target" -o stat= 2>/dev/null || true)
        case "$stat" in *Z*) continue ;; esac
        kill -KILL "$target" 2>/dev/null || true
    fi
done
'''
    process = subprocess.run(
        ["docker", "exec", container, "sh", "-c", stop_script, "sh", pid],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode not in (0, 1):
        raise HTTPException(502, process.stderr.strip() or "Failed to stop process")
    sleep(1)
    remaining = _docker_pid_snapshot(container, pid)
    remaining_pids: list[str] = []
    for line in remaining:
        match = re.match(r"^\s*(\d+)\b", line)
        if match and match.group(1) not in remaining_pids:
            remaining_pids.append(match.group(1))
    return {
        "container": container,
        "pid": pid,
        "alreadyExited": False,
        "stopped": not remaining,
        "remainingPids": remaining_pids,
        "message": "进程已停止" if not remaining else "停止后仍发现残留进程",
        "before": before,
        "remaining": remaining,
    }


@router.post("/training-reservations/stop-process", dependencies=[Depends(_authorize)])
def stop_training_reservation_process(request: TrainingReservationStopProcessRequest):
    reservation_id = request.reservationId.strip()
    if not reservation_id:
        raise HTTPException(400, "reservationId is required")
    record, checked_registries = _find_runtime_process_by_reservation(reservation_id)
    if not record:
        raise HTTPException(
            404,
            {
                "message": "No Runtime process record found for reservation",
                "reservationId": reservation_id,
                "checkedRegistries": checked_registries,
            },
        )
    container = str(record.get("container") or "").strip()
    pid = str(record.get("pid") or "").strip()
    if not container:
        raise HTTPException(409, "Runtime process record has no container")
    if not pid or not pid.isdigit():
        raise HTTPException(409, "Runtime process record has no valid pid")
    result = _stop_docker_pid(_container(container), pid)
    if result.get("remainingPids"):
        if _reservation_gpus_idle(record):
            result["remainingNonGpuPids"] = result.get("remainingPids") or []
            result["remainingPids"] = []
            result["stopped"] = True
            result["gpuIdle"] = True
            result["message"] = "训练进程 GPU 占用已释放；仅剩非 GPU 残留进程"
        else:
            raise HTTPException(409, result.get("message") or "Process still running")
    return _response(
        {
            "reservationId": reservation_id,
            "registryType": record.get("registryType"),
            "registryPath": record.get("registryPath"),
            **result,
        }
    )



def _inference_agent_url() -> str:
    return os.getenv("INFERENCE_AGENT_URL", "http://127.0.0.1:8899/inference_agent").strip()


def _inference_stop_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    response_data = data.get("data") if isinstance(data.get("data"), dict) else {}
    service_stop = response_data.get("service_stop")
    if isinstance(service_stop, dict):
        return service_stop
    inference_stop = response_data.get("inference_service_stop")
    if isinstance(inference_stop, dict):
        nested_stop = inference_stop.get("service_stop")
        if isinstance(nested_stop, dict):
            return nested_stop
    return {}


def _inference_stop_response_is_success(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    stop_payload = _inference_stop_payload(data)
    if stop_payload:
        return stop_payload.get("stopped") is True and stop_payload.get("release_ready") is not False
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

def _stop_inference_service(request: InferenceReservationStopServiceRequest) -> dict[str, Any]:
    url = _inference_agent_url()
    if not url:
        raise HTTPException(400, "INFERENCE_AGENT_URL is not configured")
    payload = {
        "command": "停止推理服务",
        "user_id": "resource-reservation-admin",
        "user_role": "admin",
        "thread_id": f"reservation:{request.reservationId}:inference-stop",
    }
    if isinstance(request.resourceContext, dict) and request.resourceContext:
        payload["resource_context"] = request.resourceContext
    container = (request.container or "").strip()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=360) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(exc.code, f"Inference service stop failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(502, f"Inference service stop failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Inference service stop timed out") from exc
    try:
        data = json.loads(response_body or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "Inference service stop returned invalid JSON") from exc
    if not _inference_stop_response_is_success(data):
        message = str(data.get("result") or data.get("message") or "Inference service stop failed")
        raise HTTPException(502, message)
    stop_payload = _inference_stop_payload(data)
    payload_reservation_id = str(
        stop_payload.get("reservation_id") or stop_payload.get("reservationId") or ""
    ).strip()
    if payload_reservation_id and payload_reservation_id != request.reservationId:
        raise HTTPException(409, "Inference stop response reservation_id does not match request")
    stopped = bool(stop_payload.get("stopped", True)) if stop_payload else True
    release_ready = bool(stop_payload.get("release_ready", stopped)) if stop_payload else True
    return {
        "reservationId": request.reservationId,
        "container": container or None,
        "stopped": stopped,
        "releaseReady": release_ready,
        "serviceStop": stop_payload or None,
        "message": str(data.get("result") or data.get("message") or "推理服务已停止"),
        "inferenceResponse": data,
    }


@router.post("/inference-reservations/stop-service", dependencies=[Depends(_authorize)])
def stop_inference_reservation_service(request: InferenceReservationStopServiceRequest):
    reservation_id = request.reservationId.strip()
    if not reservation_id:
        raise HTTPException(400, "reservationId is required")
    return _response(_stop_inference_service(request))

@router.post("/gpu-reservations/release", dependencies=[Depends(_authorize)])
def release_gpu_reservation(request: GpuReservationRequest):
    return _response(
        {
            "reservationId": request.reservationId,
            "released": _remove_reservation(request.reservationId),
        }
    )


@router.post("/training-allocations/write", dependencies=[Depends(_authorize)])
def write_training_allocation(request: TrainingAllocationWriteRequest):
    reservation_id = _safe_name(request.reservationId, "reservation id")
    container = request.container or os.getenv("MULTINODE_DOCKER_CONTAINER", "")
    if not container:
        raise HTTPException(400, "Missing MULTINODE_DOCKER_CONTAINER")
    target = _container(container)
    container_path = f"/tmp/medflow_training_allocation_{reservation_id}.json"
    payload = json.dumps(request.allocation, ensure_ascii=False).encode("utf-8")
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / f"{reservation_id}.json"
        local_path.write_bytes(payload)
        _run(
            ["docker", "cp", str(local_path), f"{target}:{container_path}"], timeout=60
        )
    return _response(
        {
            "reservationId": reservation_id,
            "container": target,
            "allocationFile": container_path,
        }
    )


@router.post("/containers/status", dependencies=[Depends(_authorize)])
def container_status(request: ContainerRequest):
    return _response(_container_status(_container(request.container)))


@router.get("/datasets", dependencies=[Depends(_authorize)])
def datasets(container: str | None = None):
    return _response(_datasets(_container(container)))


@router.get("/models", dependencies=[Depends(_authorize)])
def models(container: str | None = None):
    return _response(_models(_container(container)))


@router.get("/medical-tests", dependencies=[Depends(_authorize)])
def medical_tests(container: str | None = None):
    target = _container(container)
    result = []
    for category, path in TEST_PATHS.items():
        if category == "general":
            script = (
                'if [ ! -d "$1" ]; then exit 0; fi; '
                'for d in "$1"/*; do '
                '[ -d "$d" ] || continue; '
                'name=$(basename "$d"); '
                'size=$(du -sb "$d" 2>/dev/null | cut -f1); '
                'printf "%s\\t%s\\n" "$name" "$size"; '
                "done"
            )
        else:
            script = (
                'find "$1" -maxdepth 1 -type f -printf "%f\\t%s\\n" 2>/dev/null || true'
            )
        for line in _docker(
            target, ["sh", "-c", script, "tests", path]
        ).stdout.splitlines():
            filename, _, size = line.partition("\t")
            if filename:
                result.append(
                    {
                        "filename": filename,
                        "type": "general" if category == "general" else "json",
                        "size": size,
                        "description": "",
                        "category": category,
                    }
                )
    return _response(result)


@router.get("/evaluation-results", dependencies=[Depends(_authorize)])
def evaluation_results(container: str | None = None):
    target = _container(container)
    script = """
import json
import os
import re
import sys

root = sys.argv[1]
items = []
if os.path.isdir(root):
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder) or not re.fullmatch(r"[0-9]+_[a-f0-9]+", name):
            continue
        def load(filename):
            path = os.path.join(folder, filename)
            try:
                with open(path, encoding="utf-8") as handle:
                    value = json.load(handle)
                    return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}
        meta = load("meta.json")
        result = load("result.json")
        summary = result.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        total_score = result.get("total_score", summary.get("total_score"))
        item = {
            "jobId": meta.get("job_id") or name,
            "model": meta.get("model") or "unknown",
            "dataset": meta.get("dataset") or "unknown",
            "status": meta.get("status") or "unknown",
            "accuracy": summary.get("accuracy") or 0,
            "avgF1": summary.get("avg_f1") or 0,
            "startTime": meta.get("start_time") or "",
            "endTime": meta.get("end_time"),
            "folderPath": folder,
        }
        if isinstance(total_score, (int, float)):
            item["totalScore"] = total_score
        items.append(item)
print(json.dumps(items, ensure_ascii=False))
"""
    output = _docker(
        target,
        ["python3", "-c", script, EVALUATION_LOGS_PATH],
        timeout=120,
    ).stdout
    try:
        items = json.loads(output or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "Invalid evaluation result response") from exc
    return _response(items)


@router.get("/grpo", dependencies=[Depends(_authorize)])
def grpo(container: str | None = None):
    target = _container(container or os.getenv("MEDFLOW_GRPO_DOCKER_CONTAINER", ""))
    models_data = [
        {**item, "type": "grpo_base"}
        for item in _list_dirs(target, GRPO_MODEL_PATH)
        if str(item.get("size") or "").strip().lower().replace(" ", "") != "0b"
    ]

    def list_parquet_files(directory: str) -> list[dict[str, str]]:
        output = _docker(
            target,
            [
                "sh",
                "-c",
                'if [ -d "$1" ]; then find "$1" -type f -name "*.parquet" -printf "%f\\t%p\\n" 2>/dev/null | sort; fi',
                "grpo",
                directory,
            ],
        ).stdout
        files: list[dict[str, str]] = []
        for line in output.splitlines():
            name, _, path = line.partition("\t")
            if name and path:
                files.append(
                    {
                        "name": name,
                        "path": path,
                        "directory": str(Path(path).parent),
                        "type": "parquet",
                    }
                )
        return files

    train_files = list_parquet_files(f"{GRPO_DATA_ROOT}/train")
    val_files = list_parquet_files(f"{GRPO_DATA_ROOT}/val")

    if not train_files and not val_files:
        files = list_parquet_files(GRPO_DATA_ROOT)
        train_files = [item for item in files if "train" in item["name"].lower()]
        val_files = [
            item
            for item in files
            if any(
                keyword in item["name"].lower() for keyword in ("val", "valid", "dev")
            )
        ]
        train_files = train_files or files
        val_files = val_files or files

    return _response(
        {
            "containerName": target,
            "models": models_data,
            "trainFiles": train_files,
            "valFiles": val_files,
        }
    )


@router.get("/environment-check", dependencies=[Depends(_authorize)])
def environment_check(container: str | None = None):
    target = _container(container)
    status = (
        _run(["docker", "inspect", "-f", "{{.State.Running}}", target]).stdout.strip()
        == "true"
    )
    datasets_data, models_data, gpu_data = _datasets(target), _models(target), _gpus()
    dataset_count = sum(map(len, datasets_data.values()))
    model_count = sum(map(len, models_data.values()))
    available_gpus = sum(1 for gpu in gpu_data if gpu["available"])
    return _response(
        {
            "containerName": target,
            "checkedAt": _node_meta()["collectedAt"],
            "overallStatus": "ok" if status else "error",
            "counts": {
                "datasets": dataset_count,
                "models": model_count,
                "medicalTests": 0,
                "evaluationResults": 0,
                "gpus": len(gpu_data),
                "availableGpus": available_gpus,
            },
            "gpuInfo": gpu_data,
            "items": [
                {
                    "key": "container",
                    "title": "Docker 容器",
                    "status": "ok" if status else "error",
                    "summary": f"容器 {target} 正在运行"
                    if status
                    else f"容器 {target} 未运行",
                },
                {
                    "key": "gpu",
                    "title": "GPU 状态",
                    "status": "ok" if available_gpus > 0 else "warning",
                    "summary": f"检测到 {len(gpu_data)} 张 GPU，{available_gpus} 张空闲",
                },
                {
                    "key": "datasets",
                    "title": "数据集",
                    "status": "ok" if dataset_count > 0 else "warning",
                    "summary": f"发现 {dataset_count} 个可用数据集",
                    "count": dataset_count,
                },
                {
                    "key": "models",
                    "title": "模型",
                    "status": "ok" if model_count > 0 else "warning",
                    "summary": f"发现 {model_count} 个模型目录",
                    "count": model_count,
                },
            ],
        }
    )


@router.post("/datasets/previews", dependencies=[Depends(_authorize)])
def dataset_previews(request: DatasetRequest):
    target = _container(request.container)
    dataset_path = _dataset_path(request.datasetType, request.datasetName)
    script = r"""
import json
import os
import sys

dataset_path = sys.argv[1]
internal_files = set(sys.argv[2:])
items = []
if not os.path.isdir(dataset_path):
    print(json.dumps({"error": "not_found"}))
    sys.exit(0)
for filename in sorted(os.listdir(dataset_path)):
    if filename in internal_files:
        continue
    if not filename.endswith(".json"):
        continue
    full_path = os.path.join(dataset_path, filename)
    if not os.path.isfile(full_path):
        continue
    lines = []
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as handle:
            for _, line in zip(range(3), handle):
                lines.append(line.rstrip("\n"))
    except OSError:
        continue
    items.append({"filename": filename, "preview": "\n".join(lines)})
print(json.dumps({"items": items}, ensure_ascii=False))
"""
    output = _docker(
        target,
        ["python3", "-c", script, dataset_path, *sorted(INTERNAL_DATASET_FILES)],
        timeout=60,
    ).stdout
    try:
        parsed = json.loads(output or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "Invalid dataset preview response") from exc
    if parsed.get("error") == "not_found":
        raise HTTPException(404, f'Dataset "{request.datasetName}" not found')
    return _response(parsed.get("items", []))


@router.post("/datasets/filter-status", dependencies=[Depends(_authorize)])
def filter_status(request: DatasetFilterStatusRequest):
    target = _container(request.container)
    folder = _filter_status_folder(request)
    script = r"""
import json
import os
import sys

folder = sys.argv[1]
progress_path = os.path.join(folder, "score_progress.json")
result = {"status": "not_started", "folder": folder, "data": None}

if not os.path.isdir(folder):
    result["status"] = "not_started"
elif not os.path.isfile(progress_path):
    result["status"] = "not_started"
else:
    try:
        with open(progress_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            result = {"status": "ok", "folder": folder, "data": data}
        else:
            result = {"status": "invalid_progress", "folder": folder, "data": None}
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "invalid_progress", "folder": folder, "error": str(exc), "data": None}

print(json.dumps(result, ensure_ascii=False))
"""
    output = _docker(target, ["python3", "-c", script, folder], timeout=30).stdout
    try:
        parsed = json.loads(output or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "Invalid filter status response") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(502, "Invalid filter status payload")
    status = parsed.get("status")
    if status == "invalid_progress":
        raise HTTPException(502, "进度文件损坏或无法解析")
    return _response(parsed)


@router.post("/datasets/download", dependencies=[Depends(_authorize)])
def download_dataset(request: DatasetRequest):
    target = _container(request.container)
    dataset_name = _safe_name(request.datasetName, "dataset name")
    dataset_path = _dataset_path(request.datasetType, dataset_name)
    exists = _docker(
        target,
        ["sh", "-c", 'test -d "$1" && echo yes || echo no', "exists", dataset_path],
    ).stdout.strip()
    if exists != "yes":
        raise HTTPException(404, f'Dataset "{dataset_name}" not found')
    payload = _docker_bytes(
        target,
        ["tar", "-C", DATASET_PATHS[request.datasetType], "-czf", "-", dataset_name],
        timeout=180,
    )
    return _response(_file_base64_response(f"{dataset_name}.tar.gz", payload))


@router.delete("/datasets", dependencies=[Depends(_authorize)])
def delete_dataset(request: DatasetRequest):
    kind = request.datasetType
    if kind not in DATASET_PATHS:
        raise HTTPException(400, "Invalid dataset type")
    name = _safe_name(request.datasetName, "dataset name")
    _docker(
        _container(request.container),
        ["rm", "-rf", "--", f"{DATASET_PATHS[kind]}/{name}"],
    )
    return _response({"success": True})


@router.delete("/models", dependencies=[Depends(_authorize)])
def delete_model(request: ModelRequest):
    kind = request.modelType
    if kind not in MODEL_PATHS:
        raise HTTPException(400, "Invalid model type")
    name = _safe_name(request.modelName, "model name")
    allowed_paths = (
        set(DPO_MODEL_PATHS.values())
        if kind == "daily_trained"
        else {MODEL_PATHS[kind]}
    )
    base_path = request.modelPath or MODEL_PATHS[kind]
    if base_path not in allowed_paths:
        raise HTTPException(400, "Invalid model path")
    _docker(
        _container(request.container),
        ["rm", "-rf", "--", f"{base_path}/{name}"],
    )
    return _response({"success": True})


@router.delete("/medical-tests", dependencies=[Depends(_authorize)])
def delete_medical_test(request: MedicalTestRequest):
    filename = _safe_name(request.filename, "filename")
    target = _container(request.container)
    for path in TEST_PATHS.values():
        _docker(target, ["rm", "-f", "--", f"{path}/{filename}"])
    return _response({"success": True})


@router.post("/medical-tests/download", dependencies=[Depends(_authorize)])
def download_medical_test(request: MedicalTestRequest):
    filename = _safe_name(request.filename, "filename")
    target = _container(request.container)
    for category, path in TEST_PATHS.items():
        test_path = f"{path}/{filename}"
        if category == "general":
            exists = _docker(
                target,
                [
                    "sh",
                    "-c",
                    'test -e "$1" && echo yes || echo no',
                    "exists",
                    test_path,
                ],
            ).stdout.strip()
            if exists == "yes":
                payload = _docker_bytes(
                    target,
                    ["tar", "-C", path, "-czf", "-", filename],
                    timeout=120,
                )
                return _response(_file_base64_response(f"{filename}.tar.gz", payload))
            continue
        exists = _docker(
            target,
            ["sh", "-c", 'test -f "$1" && echo yes || echo no', "exists", test_path],
        ).stdout.strip()
        if exists == "yes":
            return _response(
                _file_base64_response(
                    filename, _docker_bytes(target, ["cat", test_path])
                )
            )
    raise HTTPException(404, f'Medical test "{filename}" not found')


@router.post("/evaluation-results/download", dependencies=[Depends(_authorize)])
def download_evaluation_result(request: EvaluationResultDownloadRequest):
    target = _container(request.container)
    source_path, filename = _evaluation_result_path(
        request.folderPath, request.filename
    )
    exists = _docker(
        target,
        ["sh", "-c", 'test -f "$1" && echo yes || echo no', "exists", source_path],
    ).stdout.strip()
    if exists != "yes":
        raise HTTPException(404, f'Evaluation result "{filename}" not found')
    return _response(
        _file_base64_response(filename, _docker_bytes(target, ["cat", source_path]))
    )


@router.delete("/evaluation-results", dependencies=[Depends(_authorize)])
def delete_evaluation_result(request: EvaluationResultRequest):
    target = _container(request.container)
    folder_name = Path(request.folderPath).name
    if not re.fullmatch(r"[0-9]+_[a-f0-9]+", folder_name):
        raise HTTPException(400, "Invalid evaluation result folder")
    _docker(
        target,
        [
            "sh",
            "-c",
            'test -d "$1" && rm -rf -- "$1"',
            "delete",
            f"{EVALUATION_LOGS_PATH}/{folder_name}",
        ],
    )
    return _response({"success": True, "message": "Evaluation result deleted"})


@router.post("/datasets/upload", dependencies=[Depends(_authorize)])
def upload_dataset(request: UploadDatasetRequest):
    if request.datasetType not in DATASET_PATHS:
        raise HTTPException(400, "Invalid dataset type")
    name = _safe_name(request.datasetName, "dataset name")
    payload = base64.b64decode(request.fileBase64, validate=True)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds 20MB")
    target = _container(request.container)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / _safe_name(request.filename, "filename")
        archive.write_bytes(payload)
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()
        content_root = _extract_dataset_archive(archive, extract_dir)
        _docker(target, ["mkdir", "-p", f"{DATASET_PATHS[request.datasetType]}/{name}"])
        _run(
            [
                "docker",
                "cp",
                f"{content_root}/.",
                f"{target}:{DATASET_PATHS[request.datasetType]}/{name}",
            ],
            timeout=60,
        )
    return _response({"success": True})


@router.post("/medical-tests/upload", dependencies=[Depends(_authorize)])
def upload_medical_test(request: UploadMedicalTestRequest):
    category = request.testType if request.testType in TEST_PATHS else "medical"
    filename = _safe_name(request.filename, "filename")
    payload = base64.b64decode(request.fileBase64, validate=True)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds 20MB")
    target = _container(request.container)
    destination = TEST_PATHS[category]
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_path = Path(temp_dir) / filename
        upload_path.write_bytes(payload)
        _docker(target, ["mkdir", "-p", destination])
        _run(
            ["docker", "cp", str(upload_path), f"{target}:{destination}/{filename}"],
            timeout=60,
        )
    return _response({"success": True})
