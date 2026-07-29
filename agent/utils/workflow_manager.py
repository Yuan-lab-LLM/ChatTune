# -*- coding: utf-8 -*-
"""Persistent train-evaluate-deploy-benchmark workflow orchestration."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

STAGES = ("train", "evaluate", "publish", "deploy", "benchmark")
TERMINAL_WORKFLOW_STATUSES = {"finished", "failed", "stopped"}
TRANSIENT_TIMEOUT_STATUS = "timeout"
TRANSIENT_TIMEOUT_MESSAGE = "本次状态查询超时，任务可能仍在运行，将继续自动刷新"
AGENT_EVENT_TYPES = {"start_evaluate", "deploy_model", "start_benchmark", "notify_terminal"}
WORKFLOW_KEY_VERSION = "workflow:v1"


def parse_workflow_control_command(message: str) -> Optional[str]:
    """Recognize persisted workflow controls without catching standalone jobs."""
    text = (message or "").lower()
    if not any(keyword in text for keyword in ["工作流", "训练部署评测"]):
        return None
    if any(keyword in text for keyword in ["停止", "结束", "取消", "stop"]):
        return "stop"
    if any(keyword in text for keyword in ["继续", "续跑", "恢复", "resume", "retry"]):
        return "resume"
    if any(keyword in text for keyword in ["状态", "进度", "查看", "查询", "status"]):
        return "status"
    return None


def build_workflow_key(
    user_id: str,
    dataset_ref: str,
    train_type: str = "lora",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a stable logical identity for equivalent one-click workflow inputs."""
    context = context or {}
    payload = {
        "version": WORKFLOW_KEY_VERSION,
        "user_id": str(user_id or "").strip(),
        "dataset_ref": str(dataset_ref or "").strip(),
        "train_type": str(train_type or "lora").strip().lower(),
    }
    for key in ("dataset_name", "dataset_dir", "evaluation_dataset_name"):
        value = context.get(key)
        if value is not None and str(value).strip():
            payload[key] = str(value).strip()
    resource_group_id = str(context.get("resource_group_id") or "").strip()
    if resource_group_id:
        payload["resource_group_id"] = resource_group_id
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "wk-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _benchmark_name(workflow: Dict[str, Any]) -> str:
    context = workflow.get("context") or {}
    benchmark = str(
        context.get("benchmark")
        or context.get("evaluation_dataset_name")
        or "2024"
    ).strip()
    return benchmark or "2024"


def _benchmark_command_name(workflow: Dict[str, Any]) -> str:
    benchmark = _benchmark_name(workflow)
    return "2024.json" if benchmark in {"2024", "2024.json"} else benchmark


def _benchmark_start_command(workflow: Dict[str, Any]) -> str:
    benchmark = _benchmark_command_name(workflow)
    return f"运行推理基准测试{benchmark}"


def _benchmark_status_command(workflow: Dict[str, Any]) -> str:
    benchmark = _benchmark_command_name(workflow)
    return f"查看推理基准测试{benchmark}状态"


def _benchmark_result_entry(workflow: Dict[str, Any]) -> str:
    benchmark = _benchmark_command_name(workflow)
    return f"查看推理基准测试{benchmark}结果"


def _benchmark_text(result: Dict[str, Any]) -> str:
    return "\n".join(
        str(result.get(key) or "")
        for key in ("result", "message", "text")
        if result.get(key) is not None
    )


def _has_timeout_signal(value: Any) -> bool:
    text = str(value or "")
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("timeout", "timed out")) or "超时" in text


def _result_has_timeout_signal(result: Dict[str, Any]) -> bool:
    status = str(result.get("status") or "").strip().lower()
    if status in {"failed", "stopped", "interrupted"}:
        return False
    if status == TRANSIENT_TIMEOUT_STATUS:
        return True
    for key in ("error", "message", "result", "text"):
        if _has_timeout_signal(result.get(key)):
            return True
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return any(_has_timeout_signal(data.get(key)) for key in ("error", "message", "status"))


def _result_has_transient_poll_error(result: Dict[str, Any]) -> bool:
    """Detect status-query infrastructure failures without treating the task as failed."""
    if _result_has_timeout_signal(result):
        return True
    text = "\n".join(
        str(result.get(key) or "")
        for key in ("error", "message", "result", "text")
        if result.get(key) is not None
    )
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "llm_error",
            "badrequesterror",
            "maximum context length",
            "context length",
            "input_tokens",
        )
    ) or "模型调用失败" in text


def _exception_has_timeout_signal(exc: Exception) -> bool:
    class_name = exc.__class__.__name__.lower()
    return "timeout" in class_name or _has_timeout_signal(exc)


def _benchmark_has_not_found_signal(text: str) -> bool:
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


def _benchmark_explicit_status(result: Dict[str, Any]) -> Optional[str]:
    """Extract explicit benchmark status fields from report text."""
    text = _benchmark_text(result)
    if _benchmark_has_not_found_signal(text):
        return "not_found"
    lowered_text = text.lower()
    if re.search(r"(?:测试状态|基准测试任务).*?(?:已完成|finished|completed|done)", text, re.IGNORECASE):
        return "finished"
    if (
        any(keyword in lowered_text for keyword in ("finished", "completed", "done"))
        and any(keyword in text for keyword in ("已处理数", "完成进度", "结果文件", "运行日志", "Task ID", "任务 ID", "任务ID"))
    ):
        return "finished"
    status_lines: List[str] = []
    for pattern in (
        r"(?:当前状态|任务状态|benchmark_status|status)\s*[:：=]\s*([^\n\r]+)",
        r"当前状态\s*[`'\"]?([^`\n\r'\"]+)",
    ):
        status_lines.extend(match.group(1) for match in re.finditer(pattern, text, re.IGNORECASE))

    for line in status_lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in ("timeout", "timed out")) or "超时" in line:
            return TRANSIENT_TIMEOUT_STATUS
        if any(keyword in lowered for keyword in ("failed", "failure")) or any(keyword in line for keyword in ("失败", "异常", "报错")):
            return "failed"
        if any(keyword in lowered for keyword in ("running", "processing")) or any(keyword in line for keyword in ("运行中", "正在运行", "处理中")):
            return "running"
        if any(keyword in lowered for keyword in ("finished", "completed", "done")) or any(keyword in line for keyword in ("已完成", "完成", "结束")):
            return "finished"
        if "stopped" in lowered or any(keyword in line for keyword in ("已停止", "已终止", "停止")):
            return "stopped"
    return None


def _benchmark_report_command(result: Dict[str, Any]) -> Optional[str]:
    commands = result.get("commands") if isinstance(result.get("commands"), list) else []
    for command in commands:
        command_text = str(command or "").strip()
        if command_text.startswith("benchmark_report"):
            return command_text
    match = re.search(r'benchmark_report\((?:job_id=)?["\']?([A-Za-z0-9_-]+)["\']?\)', _benchmark_text(result))
    if match:
        return f'benchmark_report("{match.group(1)}")'
    job_match = re.search(r"任务ID(?:为|是)?[:：]?\s*`?([A-Za-z0-9_-]+)`?", _benchmark_text(result))
    if job_match:
        return f'benchmark_report("{job_match.group(1)}")'
    return None


def _benchmark_stop_command(result: Dict[str, Any]) -> Optional[str]:
    commands = result.get("commands") if isinstance(result.get("commands"), list) else []
    for command in commands:
        command_text = str(command or "").strip()
        if command_text.startswith("benchmark_stop"):
            return command_text
    match = re.search(r'benchmark_stop\((?:job_id=)?["\']?([A-Za-z0-9_-]+)["\']?\)', _benchmark_text(result))
    if match:
        return f'benchmark_stop("{match.group(1)}")'
    return None


def _benchmark_job_id_from_report_command(command: Optional[str]) -> Optional[str]:
    match = re.search(r'benchmark_report\((?:job_id=)?["\']?([A-Za-z0-9_-]+)["\']?\)', str(command or ""))
    return match.group(1) if match else None


def _benchmark_enrich_runtime_result(result: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(result)
    explicit_status = _benchmark_explicit_status(enriched)
    if explicit_status:
        enriched["status"] = explicit_status
    report_command = _benchmark_report_command(enriched)
    if report_command:
        enriched["status_command"] = report_command
        enriched["benchmark_job_id"] = _benchmark_job_id_from_report_command(report_command)
    stop_command = _benchmark_stop_command(enriched)
    if stop_command:
        enriched["stop_command"] = stop_command
    return enriched


def _benchmark_runtime_status_command(workflow: Dict[str, Any]) -> str:
    stage = (workflow.get("stages") or {}).get("benchmark", {})
    result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
    status_command = str(result.get("status_command") or "").strip()
    if status_command:
        return status_command
    report_command = _benchmark_report_command(result)
    return report_command or _benchmark_status_command(workflow)


def _int_value(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip().rstrip("%")))
    except (TypeError, ValueError):
        return None


def _progress_value(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def train_result_reached_completion(result: Dict[str, Any]) -> bool:
    """Return True when training reached its final step and stopped saving."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    if metrics.get("training_process_exists") or metrics.get("pid_alive"):
        return False
    current_step = _int_value(metrics.get("current_step") or metrics.get("latest_step"))
    total_steps = _int_value(metrics.get("total_steps"))
    progress = _progress_value(metrics.get("progress_percent"))
    if current_step is not None and total_steps is not None and total_steps > 0:
        return current_step >= total_steps
    return bool(progress is not None and progress >= 100)


def train_result_has_explicit_incomplete_progress(result: Dict[str, Any]) -> bool:
    """Return True when available counters prove training is not complete."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    current_step = _int_value(metrics.get("current_step") or metrics.get("latest_step"))
    total_steps = _int_value(metrics.get("total_steps"))
    progress = _progress_value(metrics.get("progress_percent"))
    return bool(
        (
            current_step is not None
            and total_steps is not None
            and total_steps > 0
            and current_step < total_steps
        )
        or (progress is not None and progress < 100)
    )


def train_result_has_startup_evidence(result: Dict[str, Any]) -> bool:
    """Return True when an ended launch PID may still have a live training run."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    if metrics.get("training_process_exists") or metrics.get("pid_alive"):
        return True
    return any(
        metrics.get(key)
        for key in (
            "wandb_run_dir",
            "wandb_run_name",
            "wandb_url",
            "output_dir",
            "latest_loss",
            "latest_step",
            "latest_epoch",
            "history_count",
        )
    )


@dataclass
class WorkflowDependencies:
    start_train: Callable[[Dict[str, Any]], Dict[str, Any]]
    monitor_train: Callable[[Dict[str, Any]], Dict[str, Any]]
    find_trained_model: Callable[[Dict[str, Any]], str]
    start_evaluate: Callable[[Dict[str, Any]], Dict[str, Any]]
    monitor_evaluate: Callable[[Dict[str, Any]], Dict[str, Any]]
    start_publish: Callable[[Dict[str, Any]], Dict[str, Any]]
    monitor_publish: Callable[[Dict[str, Any]], Dict[str, Any]]
    inference_command: Callable[[str], Dict[str, Any]]
    stop_task: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None
    check_existing_stage_output: Optional[Callable[[Dict[str, Any], str], Optional[Dict[str, Any]]]] = None
    start_benchmark: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    monitor_benchmark: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    benchmark_result: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    monitor_deploy: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None
    stop_inference_service: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class WorkflowManager:
    """SQLite-backed workflow state machine with a lightweight polling worker."""

    def __init__(
        self,
        db_path: str,
        dependencies: WorkflowDependencies,
        poll_interval: int = 30,
        auto_start_worker: bool = True,
        agent_events: bool = True,
        train_start_grace_seconds: int = 180,
        event_lease_seconds: int = 300,
        on_terminal_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.dependencies = dependencies
        self.poll_interval = max(1, int(poll_interval))
        self.auto_start_worker = bool(auto_start_worker)
        self.agent_events = agent_events
        self.train_start_grace_seconds = max(0, int(train_start_grace_seconds))
        self.event_lease_seconds = max(1, int(event_lease_seconds))
        self.on_terminal_update = on_terminal_update
        self._terminal_updates_sent: set[str] = set()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_workflow_ids: set[str] = set()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        if auto_start_worker:
            self.start_worker()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    workflow_key TEXT,
                    user_id TEXT NOT NULL,
                    dataset_ref TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    stages_json TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(workflows)").fetchall()
            }
            if "workflow_key" not in columns:
                connection.execute("ALTER TABLE workflows ADD COLUMN workflow_key TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflows_user_status
                ON workflows(user_id, status, updated_at)
                """,
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflows_key_status
                ON workflows(workflow_key, status, updated_at)
                """,
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_events (
                    event_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claimed_by TEXT,
                    claimed_until REAL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_events_status
                ON workflow_events(status, claimed_until, created_at)
                """,
            )

    def start_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="workflow-manager",
                daemon=True,
            )
            self._worker.start()
            logger.debug(
                "[workflow-debug] worker thread started poll_interval=%s auto_start_worker=%s",
                self.poll_interval,
                self.auto_start_worker,
            )

    def activate_worker_for(self, workflow_id: str) -> None:
        """Poll one explicitly started/resumed workflow without recovering all old rows."""
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            return
        with self._lock:
            self._worker_workflow_ids.add(workflow_id)
            activated_count = len(self._worker_workflow_ids)
        logger.debug(
            "[workflow-debug] activate worker workflow_id=%s auto_start_worker=%s activated_count=%s",
            workflow_id,
            self.auto_start_worker,
            activated_count,
        )
        self.start_worker()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=max(2, self.poll_interval + 1))

    def _worker_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self.run_pending_once(only_activated=not self.auto_start_worker)
            except Exception:
                logger.exception("Workflow worker poll failed")

    def _decode(self, row: sqlite3.Row) -> Dict[str, Any]:
        workflow = dict(row)
        workflow.setdefault("workflow_key", None)
        workflow["stages"] = json.loads(workflow.pop("stages_json"))
        workflow["context"] = json.loads(workflow.pop("context_json"))
        return workflow

    def _save(
        self,
        workflow: Dict[str, Any],
        allow_terminal_override: bool = False,
    ) -> Dict[str, Any]:
        if not allow_terminal_override:
            workflow = self._finalize_finished_benchmark_shutdown(workflow)
        workflow["updated_at"] = time.time()
        with self._connect() as connection:
            current_row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id=?",
                (workflow["workflow_id"],),
            ).fetchone()
            if (
                current_row
                and (
                    current_row["status"] in TERMINAL_WORKFLOW_STATUSES
                    and workflow["status"] == "running"
                    or current_row["status"] == "stopping"
                    and workflow["status"] != "stopping"
                )
                and not allow_terminal_override
            ):
                return self._decode(current_row)
            connection.execute(
                """
                UPDATE workflows
                SET workflow_key=?, status=?, current_stage=?, stages_json=?, context_json=?,
                    error=?, updated_at=?
                WHERE workflow_id=?
                """,
                (
                    workflow.get("workflow_key"),
                    workflow["status"],
                    workflow["current_stage"],
                    json.dumps(workflow["stages"], ensure_ascii=False),
                    json.dumps(workflow["context"], ensure_ascii=False),
                    workflow.get("error"),
                    workflow["updated_at"],
                    workflow["workflow_id"],
                ),
            )
        if workflow.get("status") in TERMINAL_WORKFLOW_STATUSES:
            workflow_id = str(workflow["workflow_id"])
            should_notify = False
            with self._lock:
                self._worker_workflow_ids.discard(workflow_id)
                if workflow_id not in self._terminal_updates_sent:
                    self._terminal_updates_sent.add(workflow_id)
                    should_notify = True
            if should_notify and self.on_terminal_update:
                try:
                    self.on_terminal_update(workflow)
                except Exception:
                    logger.exception("Failed to notify terminal workflow update %s", workflow_id)
        return workflow

    def _get(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id=?",
                (workflow_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def get(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self._get(workflow_id)

    def _needs_finished_benchmark_shutdown(self, workflow: Optional[Dict[str, Any]]) -> bool:
        if not workflow:
            return False
        if workflow.get("status") != "finished" or workflow.get("current_stage") != "benchmark":
            return False
        stage = (workflow.get("stages") or {}).get("benchmark", {})
        if stage.get("status") != "finished":
            return False
        result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
        return (
            result.get("inference_service_stopped") is None
            and not result.get("inference_service_stop_error")
            and self.dependencies.stop_inference_service is not None
        )

    def _finalize_finished_benchmark_shutdown(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        if self._needs_finished_benchmark_shutdown(workflow):
            stage = workflow["stages"]["benchmark"]
            result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
            self._finish_benchmark_stage(workflow, result)
        return workflow

    def _repair_finished_benchmark_shutdown(self, workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not workflow:
            return {}
        if not self._needs_finished_benchmark_shutdown(workflow):
            return workflow
        self._finalize_finished_benchmark_shutdown(workflow)
        return self._save(workflow, allow_terminal_override=True)

    def get_status_snapshot(self, workflow_id: str) -> Dict[str, Any]:
        """Return the persisted workflow row and repair missing terminal benchmark cleanup."""
        return self._repair_finished_benchmark_shutdown(self._get(workflow_id))

    def is_running(self, workflow_id: str) -> bool:
        workflow = self._get(workflow_id)
        return bool(workflow and workflow["status"] == "running")

    def _user_family_conditions(self, user_id: str) -> tuple[str, tuple[Any, ...]]:
        """Match a user across per-session ids like username#session."""
        normalized = str(user_id or "").strip().strip("[]")
        if not normalized:
            return "user_id=?", ("",)
        base = normalized.split("#", 1)[0].strip()
        keys = list(dict.fromkeys(value for value in (normalized, base) if value))
        clauses = [f"user_id IN ({','.join('?' for _ in keys)})"]
        params: list[Any] = [*keys]
        if base:
            clauses.append("user_id LIKE ?")
            params.append(f"{base}#%")
        return f"({' OR '.join(clauses)})", tuple(params)

    def _resource_group_conditions(
        self,
        resource_group_id: Optional[str],
    ) -> tuple[str, tuple[Any, ...]]:
        normalized = str(resource_group_id or "").strip()
        if not normalized:
            return "", ()
        return " AND json_extract(context_json, '$.resource_group_id')=?", (normalized,)

    def _assert_workflow_resource_group(
        self,
        workflow: Dict[str, Any],
        resource_group_id: Optional[str],
    ) -> None:
        expected = str(resource_group_id or "").strip()
        if not expected:
            return
        actual = str((workflow.get("context") or {}).get("resource_group_id") or "").strip()
        if actual == expected:
            return
        if actual:
            raise ValueError(
                f"指定的一键工作流属于资源组 `{actual}`，当前资源组为 `{expected}`，无法操作"
            )
        raise ValueError(
            f"指定的一键工作流缺少资源组信息，当前资源组为 `{expected}`，无法操作"
        )

    def get_for_user_family(
        self,
        workflow_id: str,
        user_id: str,
        resource_group_id: Optional[str] = None,
        permission_message: str = "无权访问指定的一键工作流",
    ) -> Optional[Dict[str, Any]]:
        where_sql, params = self._user_family_conditions(user_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM workflows
                WHERE workflow_id=? AND {where_sql}
                """,
                (workflow_id, *params),
            ).fetchone()
        workflow = self._decode(row) if row else None
        if not workflow:
            if self._get(workflow_id):
                raise ValueError(permission_message)
            return None
        self._assert_workflow_resource_group(workflow, resource_group_id)
        return workflow

    def latest_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM workflows
                WHERE user_id=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        workflow = self._decode(row) if row else None
        return self._repair_finished_benchmark_shutdown(workflow) if workflow else None

    def latest_for_user_family(
        self,
        user_id: str,
        resource_group_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        where_sql, params = self._user_family_conditions(user_id)
        group_sql, group_params = self._resource_group_conditions(resource_group_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM workflows
                WHERE {where_sql}{group_sql}
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (*params, *group_params),
            ).fetchone()
        workflow = self._decode(row) if row else None
        return self._repair_finished_benchmark_shutdown(workflow) if workflow else None

    def active_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM workflows
                WHERE user_id=? AND status IN ('running', 'stopping')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def active_for_user_family(
        self,
        user_id: str,
        resource_group_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        where_sql, params = self._user_family_conditions(user_id)
        group_sql, group_params = self._resource_group_conditions(resource_group_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM workflows
                WHERE {where_sql}{group_sql} AND status IN ('running', 'stopping')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (*params, *group_params),
            ).fetchone()
        return self._decode(row) if row else None

    def resumable_for_user_family(
        self,
        user_id: str,
        resource_group_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        where_sql, params = self._user_family_conditions(user_id)
        group_sql, group_params = self._resource_group_conditions(resource_group_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM workflows
                WHERE {where_sql}{group_sql} AND status IN ('failed', 'stopped')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (*params, *group_params),
            ).fetchone()
        return self._decode(row) if row else None

    def _resume_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        if (
            workflow.get("current_stage") == "benchmark"
            and (workflow.get("context") or {}).get("inference_service_stopped_by_benchmark_stop")
        ):
            workflow["current_stage"] = "deploy"
            deploy_stage = workflow["stages"]["deploy"]
            deploy_stage["status"] = "pending"
            deploy_stage.pop("error", None)
            deploy_stage["updated_at"] = time.time()
            benchmark_stage = workflow["stages"]["benchmark"]
            benchmark_stage.clear()
            benchmark_stage.update({"status": "pending", "updated_at": time.time()})
            workflow["context"].pop("inference_service_stopped_by_benchmark_stop", None)
            workflow["context"].pop("benchmark_service_stopped_at", None)
            workflow["status"] = "running"
            workflow["error"] = None
            return self._save(workflow, allow_terminal_override=True)
        stage = workflow["stages"][workflow["current_stage"]]
        # A final step is not a durable training result: checkpoint/model export
        # may have failed after the progress bar reached 100%.  Resume training
        # through the pending path; its artifact check can still skip a rerun
        # when a valid merged model actually exists.
        stage["status"] = "pending"
        stage.pop("error", None)
        stage["updated_at"] = time.time()
        if workflow["current_stage"] == "train":
            workflow["context"].pop("trained_model_path", None)
            workflow["context"].pop("train_started_at", None)
        workflow["status"] = "running"
        workflow["error"] = None
        return self._save(workflow, allow_terminal_override=True)

    def create(
        self,
        user_id: str,
        dataset_ref: str,
        auto_start: bool = True,
        train_type: str = "lora",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            workflow_context = {
                "train_type": train_type,
                "dataset_ref": dataset_ref,
                "benchmark": "2024",
            }
            workflow_context.update(context or {})
            workflow_key = build_workflow_key(user_id, dataset_ref, train_type, workflow_context)
            workflow_id = f"wf-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            stages = {
                name: {"status": "pending", "updated_at": now}
                for name in STAGES
            }
            workflow = {
                "workflow_id": workflow_id,
                "workflow_key": workflow_key,
                "user_id": user_id,
                "dataset_ref": dataset_ref,
                "status": "running",
                "current_stage": "train",
                "stages": stages,
                "context": workflow_context,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            reusable_workflow: Optional[Dict[str, Any]] = None
            group_sql, group_params = self._resource_group_conditions(
                workflow_context.get("resource_group_id"),
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                active = connection.execute(
                    f"""
                    SELECT workflow_id FROM workflows
                    WHERE user_id=?{group_sql} AND status IN ('running', 'stopping')
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (user_id, *group_params),
                ).fetchone()
                if active:
                    raise ValueError(f"用户已有运行中的工作流: {active['workflow_id']}")
                reusable = connection.execute(
                    """
                    SELECT * FROM workflows
                    WHERE user_id=? AND workflow_key=? AND status IN ('failed', 'stopped')
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (user_id, workflow_key),
                ).fetchone()
                if reusable:
                    reusable_workflow = self._decode(reusable)
                    reusable_workflow["workflow_key"] = reusable_workflow.get("workflow_key") or workflow_key
                else:
                    connection.execute(
                        """
                        INSERT INTO workflows(
                            workflow_id, workflow_key, user_id, dataset_ref, status, current_stage,
                            stages_json, context_json, error, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            workflow_id,
                            workflow_key,
                            user_id,
                            dataset_ref,
                            "running",
                            "train",
                            json.dumps(stages, ensure_ascii=False),
                            json.dumps(workflow_context, ensure_ascii=False),
                            None,
                            now,
                            now,
                        ),
                    )
            if reusable_workflow:
                workflow = self._resume_workflow(reusable_workflow)
                return self.advance(workflow["workflow_id"]) if auto_start else workflow
            return self.advance(workflow_id) if auto_start else workflow

    def record_running_stage(self, workflow_id: str, **fields: Any) -> Dict[str, Any]:
        """Record a stage started outside the worker, such as through a ReAct agent."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow or workflow["status"] != "running":
                return workflow or {}
            if workflow["current_stage"] == "train":
                workflow["context"]["train_started_at"] = time.time()
            self._stage_result(workflow, "running", **fields)
            return self._save(workflow)

    def record_preparing_stage(self, workflow_id: str, **fields: Any) -> Dict[str, Any]:
        """Record an accepted training task whose data/model preparation is still running."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow or workflow["status"] != "running":
                return workflow or {}
            if workflow["current_stage"] == "train":
                workflow["context"]["train_started_at"] = time.time()
            self._stage_result(workflow, "preparing", **fields)
            return self._save(workflow)

    def record_external_stage_starting(self, workflow_id: str) -> Dict[str, Any]:
        """Prevent the worker from starting a stage while an agent is launching it."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow or workflow["status"] != "running":
                return workflow or {}
            self._stage_result(workflow, "starting_external")
            return self._save(workflow)

    def record_finished_stage(self, workflow_id: str, **fields: Any) -> Dict[str, Any]:
        """Record an agent-owned stage completion and move to the next stage."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow or workflow["status"] != "running":
                return workflow or {}
            if workflow.get("current_stage") == "benchmark":
                result = dict(fields.get("result")) if isinstance(fields.get("result"), dict) else {}
                for key, value in fields.items():
                    if key != "result" and value is not None and result.get(key) is None:
                        result[key] = value
                self._finish_benchmark_stage(workflow, result)
                return self._save(workflow)
            self._stage_result(workflow, "finished", **fields)
            self._move_next(workflow)
            return self._save(workflow)

    def update_context(self, workflow_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                return {}
            workflow["context"].update({key: value for key, value in fields.items() if value is not None})
            return self._save(workflow)

    def fail_current_stage(self, workflow_id: str, error: str) -> Dict[str, Any]:
        """Persist a failure reported by an external stage starter."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                return {}
            if workflow["status"] != "running":
                return workflow
            return self._fail(workflow, error)

    def begin_stop(
        self,
        user_id: str,
        resource_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reserve an active workflow for stopping so workers stop advancing it."""
        with self._lock:
            workflow = self.active_for_user_family(user_id, resource_group_id)
            if not workflow:
                raise ValueError("当前没有运行中的一键工作流")
            if workflow["status"] == "stopping":
                return workflow
            workflow["status"] = "stopping"
            workflow["context"]["stop_requested_at"] = time.time()
            return self._save(workflow)

    def begin_stop_by_id(
        self,
        workflow_id: str,
        request_user_id: Optional[str] = None,
        resource_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reserve a specific active workflow for stopping."""
        with self._lock:
            workflow: Optional[Dict[str, Any]]
            if request_user_id:
                workflow = self.get_for_user_family(
                    workflow_id,
                    request_user_id,
                    resource_group_id,
                    permission_message="无权停止指定的一键工作流",
                )
            else:
                workflow = self._get(workflow_id)
                if workflow:
                    self._assert_workflow_resource_group(workflow, resource_group_id)
            if not workflow:
                raise ValueError(f"一键工作流不存在: {workflow_id}")
            if workflow["status"] not in {"running", "stopping"}:
                raise ValueError("指定的一键工作流不是运行中状态")
            if workflow["status"] == "stopping":
                return workflow
            workflow["status"] = "stopping"
            workflow["context"]["stop_requested_at"] = time.time()
            return self._save(workflow)

    def complete_stop(self, workflow_id: str, **fields: Any) -> Dict[str, Any]:
        """Persist a completed runtime cleanup as stopped."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                raise ValueError("一键工作流不存在")
            stage = workflow["stages"][workflow["current_stage"]]
            stage["status"] = "stopped"
            stage["updated_at"] = time.time()
            stage.update({key: value for key, value in fields.items() if value is not None})
            workflow["status"] = "stopped"
            workflow["error"] = "用户已停止工作流"
            workflow = self._save(workflow, allow_terminal_override=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE workflow_events
                    SET status='cancelled', claimed_until=NULL, updated_at=?
                    WHERE workflow_id=? AND status IN ('pending', 'claimed')
                    """,
                    (time.time(), workflow["workflow_id"]),
                )
            return workflow

    def cancel_stop(self, workflow_id: str, error: str) -> Dict[str, Any]:
        """Restore polling when runtime cleanup could not be completed."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                return {}
            workflow["status"] = "running"
            workflow["context"]["stop_error"] = error
            return self._save(workflow, allow_terminal_override=True)

    def stop(
        self,
        user_id: str,
        invoke_stop_task: bool = True,
        resource_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        workflow = self.begin_stop(user_id, resource_group_id)
        stop_fields: Dict[str, Any] = {}
        try:
            if invoke_stop_task and self.dependencies.stop_task:
                cleanup_result = self.dependencies.stop_task(workflow)
                if isinstance(cleanup_result, dict):
                    stop_fields = cleanup_result
        except Exception as exc:
            self.cancel_stop(workflow["workflow_id"], str(exc))
            raise
        return self.complete_stop(workflow["workflow_id"], **stop_fields)

    def resume(
        self,
        user_id: str,
        resource_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            workflow = self.resumable_for_user_family(user_id, resource_group_id)
            if not workflow:
                active = self.active_for_user_family(user_id, resource_group_id)
                if active:
                    status_text = "正在停止" if active.get("status") == "stopping" else "仍在运行"
                    raise ValueError(
                        f"一键工作流 `{active['workflow_id']}`{status_text}，"
                        f"当前阶段：{active.get('current_stage') or '未知'}；"
                        "请先查看状态，确需重启当前阶段时先停止该工作流"
                    )
                raise ValueError("没有可续跑的一键工作流")
            return self._resume_workflow(workflow)

    def resume_by_id(
        self,
        workflow_id: str,
        request_user_id: Optional[str] = None,
        resource_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            workflow = (
                self.get_for_user_family(
                    workflow_id,
                    request_user_id,
                    resource_group_id,
                    permission_message="无权继续指定的一键工作流",
                )
                if request_user_id
                else self._get(workflow_id)
            )
            if not workflow:
                raise ValueError(f"一键工作流不存在: {workflow_id}")
            if not request_user_id:
                self._assert_workflow_resource_group(workflow, resource_group_id)
            if workflow["status"] not in {"failed", "stopped"}:
                raise ValueError("指定的一键工作流不是失败或停止状态，无需续跑")
            return self._resume_workflow(workflow)

    def enqueue_event(
        self,
        workflow: Dict[str, Any],
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        if event_type not in AGENT_EVENT_TYPES:
            raise ValueError(f"不支持的工作流事件: {event_type}")
        now = time.time()
        event_id = f"{workflow['workflow_id']}:{event_type}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_events(
                    event_id, workflow_id, user_id, event_type, payload_json,
                    status, claimed_by, claimed_until, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status='pending',
                    claimed_by=NULL,
                    claimed_until=NULL,
                    error=NULL,
                    updated_at=excluded.updated_at
                WHERE workflow_events.status IN ('failed', 'cancelled', 'done')
                """,
                (
                    event_id,
                    workflow["workflow_id"],
                    workflow["user_id"],
                    event_type,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return event_id

    def claim_pending_events(
        self,
        consumer_id: str,
        limit: int = 10,
        lease_seconds: Optional[int] = None,
        workflow_id: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Claim pending events atomically so only one process invokes each agent."""
        now = time.time()
        claimed_until = now + max(1, int(lease_seconds or self.event_lease_seconds))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE workflow_events
                SET status='pending', claimed_by=NULL, claimed_until=NULL, updated_at=?
                WHERE status='claimed' AND claimed_until < ?
                """,
                (now, now),
            )
            workflow_filter = "AND e.workflow_id=?" if workflow_id else ""
            params: tuple[Any, ...] = (
                (workflow_id, max(1, limit)) if workflow_id else (max(1, limit),)
            )
            rows = connection.execute(
                f"""
                SELECT e.* FROM workflow_events e
                JOIN workflows w ON w.workflow_id=e.workflow_id
                WHERE e.status='pending' AND w.status='running'
                {workflow_filter}
                ORDER BY e.created_at
                LIMIT ?
                """,
                params,
            ).fetchall()
            event_ids = [str(row["event_id"]) for row in rows]
            if event_ids:
                placeholders = ",".join("?" for _ in event_ids)
                connection.execute(
                    f"""
                    UPDATE workflow_events
                    SET status='claimed', claimed_by=?, claimed_until=?, updated_at=?
                    WHERE event_id IN ({placeholders}) AND status='pending'
                    """,
                    (consumer_id, claimed_until, now, *event_ids),
                )
            claimed_rows = connection.execute(
                """
                SELECT * FROM workflow_events
                WHERE claimed_by=? AND status='claimed' AND updated_at=?
                ORDER BY created_at
                """,
                (consumer_id, now),
            ).fetchall()
        return [self._decode_event(row) for row in claimed_rows]

    def renew_event_lease(
        self,
        event_id: str,
        consumer_id: str,
        lease_seconds: Optional[int] = None,
    ) -> bool:
        claimed_until = time.time() + max(1, int(lease_seconds or self.event_lease_seconds))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_events
                SET claimed_until=?, updated_at=?
                WHERE event_id=? AND claimed_by=? AND status='claimed'
                """,
                (claimed_until, time.time(), event_id, consumer_id),
            )
        return cursor.rowcount == 1

    def complete_event(self, event_id: str, consumer_id: str) -> None:
        self._finish_event(event_id, consumer_id, "done")

    def fail_event(self, event_id: str, consumer_id: str, error: str) -> None:
        self._finish_event(event_id, consumer_id, "failed", error)

    def _finish_event(
        self,
        event_id: str,
        consumer_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE workflow_events
                SET status=?, claimed_until=NULL, error=?, updated_at=?
                WHERE event_id=? AND claimed_by=? AND status='claimed'
                """,
                (status, error, time.time(), event_id, consumer_id),
            )

    def _decode_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        event = dict(row)
        event["payload"] = json.loads(event.pop("payload_json"))
        return event

    def run_pending_once(self, only_activated: bool = False) -> None:
        with self._connect() as connection:
            if only_activated:
                with self._lock:
                    workflow_ids = sorted(self._worker_workflow_ids)
                if not workflow_ids:
                    return
                placeholders = ",".join("?" for _ in workflow_ids)
                rows = connection.execute(
                    f"SELECT workflow_id, status, current_stage, stages_json FROM workflows WHERE workflow_id IN ({placeholders})",
                    workflow_ids,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT workflow_id, status, current_stage, stages_json FROM workflows WHERE status='running'",
                ).fetchall()
        scan_details = []
        for row in rows:
            stage_status = None
            try:
                stages = json.loads(row["stages_json"]) if "stages_json" in row.keys() else {}
                stage_status = (stages.get(row["current_stage"]) or {}).get("status")
            except Exception:
                stage_status = "<decode_error>"
            scan_details.append(
                {
                    "workflow_id": str(row["workflow_id"]),
                    "status": row["status"] if "status" in row.keys() else None,
                    "current_stage": row["current_stage"] if "current_stage" in row.keys() else None,
                    "stage_status": stage_status,
                }
            )
        logger.debug(
            "[workflow-debug] worker scan only_activated=%s workflow_count=%s workflows=%s",
            only_activated,
            len(rows),
            scan_details,
        )
        for row in rows:
            try:
                self.advance(str(row["workflow_id"]))
            except Exception:
                logger.exception("Failed to advance workflow %s", row["workflow_id"])

    def refresh_status(self, workflow_id: str) -> Dict[str, Any]:
        """Compatibility alias for explicit status polling."""
        return self.poll_running_stage(workflow_id)

    def poll_running_stage(self, workflow_id: str) -> Dict[str, Any]:
        """Refresh a running stage without starting a pending next stage."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                return {}
            if workflow["status"] != "running":
                if (
                    workflow["status"] == "failed"
                    and workflow.get("current_stage") == "benchmark"
                ):
                    return self._reconcile_terminal_benchmark(workflow)
                return self._repair_finished_benchmark_shutdown(workflow) or {}
            stage = workflow["stages"][workflow["current_stage"]]
            logger.debug(
                "[workflow-debug] poll workflow_id=%s status=%s current_stage=%s stage_status=%s",
                workflow_id,
                workflow.get("status"),
                workflow.get("current_stage"),
                stage.get("status"),
            )
            if stage.get("status") == "pending" and workflow.get("current_stage") in {"deploy", "benchmark"}:
                logger.debug(
                    "[workflow-debug] poll advancing pending external stage workflow_id=%s stage=%s",
                    workflow_id,
                    workflow.get("current_stage"),
                )
                return self.advance(workflow_id)
            if (
                stage.get("status") in {"awaiting_agent", "starting_external"}
                and workflow.get("current_stage") in {"deploy", "benchmark"}
            ):
                logger.debug(
                    "[workflow-debug] poll advancing awaitable external stage workflow_id=%s stage=%s stage_status=%s",
                    workflow_id,
                    workflow.get("current_stage"),
                    stage.get("status"),
                )
                return self.advance(workflow_id)
            if stage.get("status") not in {"preparing", "running", "starting", TRANSIENT_TIMEOUT_STATUS}:
                return workflow
        return self.advance(workflow_id)

    def skip_existing_stage_output(self, workflow_id: str) -> Dict[str, Any]:
        """Apply only the stage artifact skip check, without starting work."""
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow or workflow["status"] != "running":
                return workflow or {}
            stage_name = workflow["current_stage"]
            stage = workflow["stages"][stage_name]
            if self._skip_existing_stage_output(workflow, stage_name, stage):
                return self._save(workflow)
            return workflow

    def _stage_result(
        self,
        workflow: Dict[str, Any],
        status: str,
        **fields: Any,
    ) -> None:
        stage = workflow["stages"][workflow["current_stage"]]
        stage["status"] = status
        stage["updated_at"] = time.time()
        stage.update({key: value for key, value in fields.items() if value is not None})

    def _details(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in result.items() if key != "status"}

    def _skip_existing_stage_output(
        self,
        workflow: Dict[str, Any],
        stage_name: str,
        stage: Dict[str, Any],
    ) -> bool:
        if stage.get("status") != "pending" or not self.dependencies.check_existing_stage_output:
            return False
        if self.agent_events and stage_name == "benchmark":
            return False
        result = self.dependencies.check_existing_stage_output(workflow, stage_name)
        if not result:
            return False
        context_updates = result.get("context") if isinstance(result.get("context"), dict) else {}
        workflow["context"].update({key: value for key, value in context_updates.items() if value is not None})
        for key in ("trained_model_path", "published_model_path", "old_model_path", "old_model_name"):
            if result.get(key) is not None:
                workflow["context"][key] = result[key]
        status = result.get("status") or "finished"
        details = self._details(result)
        details["skipped"] = True
        if status == "finished":
            self._stage_result(workflow, "finished", **details)
            self._move_next(workflow)
            return True
        if status == "running":
            self._stage_result(workflow, "running", **details)
            return True
        return False

    def _move_next(self, workflow: Dict[str, Any]) -> None:
        current_index = STAGES.index(workflow["current_stage"])
        if current_index + 1 == len(STAGES):
            workflow["status"] = "finished"
            return
        workflow["current_stage"] = STAGES[current_index + 1]
        self.activate_worker_for(str(workflow["workflow_id"]))

    def _fail(self, workflow: Dict[str, Any], exc: Exception | str) -> Dict[str, Any]:
        # A monitor call can be in flight while another worker/process handles
        # an explicit user stop. Never turn that stop into a stage failure.
        current = self._get(workflow["workflow_id"])
        if current and current.get("status") in {"stopping", "stopped"}:
            return current
        error = str(exc)
        self._stage_result(workflow, "failed", error=error)
        workflow["status"] = "failed"
        workflow["error"] = error
        return self._save(workflow)

    def _record_transient_timeout(
        self,
        workflow: Dict[str, Any],
        stage: Dict[str, Any],
        error: Any,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = str(error or TRANSIENT_TIMEOUT_MESSAGE).strip() or TRANSIENT_TIMEOUT_MESSAGE
        stage["status"] = TRANSIENT_TIMEOUT_STATUS
        stage["updated_at"] = time.time()
        stage["last_poll_error"] = message
        stage["message"] = TRANSIENT_TIMEOUT_MESSAGE
        if result:
            stage.update({key: value for key, value in result.items() if value is not None})
            stage["status"] = TRANSIENT_TIMEOUT_STATUS
            stage["last_poll_error"] = message
            stage["message"] = TRANSIENT_TIMEOUT_MESSAGE
        workflow["status"] = "running"
        workflow["error"] = None

    def _benchmark_result_fallback(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if self.dependencies.benchmark_result:
                result = self.dependencies.benchmark_result(workflow)
            elif self.agent_events:
                return None
            else:
                result = self.dependencies.inference_command(_benchmark_result_entry(workflow))
        except Exception:
            return None
        enriched = _benchmark_enrich_runtime_result({
            **result,
            "result_entry": _benchmark_result_entry(workflow),
        })
        return enriched if _benchmark_explicit_status(enriched) == "finished" else None

    def _monitor_with_timeout(
        self,
        workflow: Dict[str, Any],
        stage: Dict[str, Any],
        monitor: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        try:
            result = monitor(workflow)
        except Exception as exc:
            if _exception_has_timeout_signal(exc):
                self._record_transient_timeout(workflow, stage, exc)
                return None
            raise
        if _result_has_timeout_signal(result):
            self._record_transient_timeout(
                workflow,
                stage,
                result.get("error") or result.get("message") or result.get("result") or TRANSIENT_TIMEOUT_MESSAGE,
                result,
            )
            return None
        if stage.get("status") == TRANSIENT_TIMEOUT_STATUS and result.get("status") not in {None, TRANSIENT_TIMEOUT_STATUS}:
            stage.pop("last_poll_error", None)
            stage.pop("error", None)
        return result

    def _finish_benchmark_stage(
        self,
        workflow: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        final_result = dict(result)
        logger.debug(
            "[workflow-debug] finish benchmark workflow_id=%s incoming_status=%s job_id=%s has_stop_service=%s",
            workflow.get("workflow_id"),
            result.get("status") or result.get("benchmark_status"),
            result.get("benchmark_job_id"),
            self.dependencies.stop_inference_service is not None,
        )
        if self.dependencies.stop_inference_service:
            try:
                logger.debug(
                    "[workflow-debug] stopping inference service after benchmark workflow_id=%s",
                    workflow.get("workflow_id"),
                )
                stop_result = self.dependencies.stop_inference_service(workflow)
                logger.debug(
                    "[workflow-debug] stop inference service returned workflow_id=%s success=%s status=%s error=%s",
                    workflow.get("workflow_id"),
                    stop_result.get("success") if isinstance(stop_result, dict) else None,
                    stop_result.get("status") if isinstance(stop_result, dict) else None,
                    stop_result.get("error") if isinstance(stop_result, dict) else None,
                )
                final_result["inference_service_stopped"] = bool(
                    stop_result.get("success", True) if isinstance(stop_result, dict) else True
                )
                final_result["inference_service_stop_result"] = stop_result
                final_result["inference_service_stop_command"] = "停止推理服务"
                final_result["inference_service_log_command"] = "查看推理服务日志"
                if isinstance(stop_result, dict):
                    for source_key, target_key in (
                        ("stop_service_log_path", "stop_service_log_path"),
                        ("stop_service_log_tail", "stop_service_log_tail"),
                        ("stop_service_log_updated_at", "stop_service_log_updated_at"),
                        ("log_path", "stop_service_log_path"),
                        ("log_tail", "stop_service_log_tail"),
                        ("log_updated_at", "stop_service_log_updated_at"),
                    ):
                        if stop_result.get(source_key) is not None and final_result.get(target_key) is None:
                            final_result[target_key] = stop_result[source_key]
            except Exception as exc:
                logger.exception(
                    "Workflow %s benchmark finished but inference service shutdown failed",
                    workflow.get("workflow_id"),
                )
                final_result["inference_service_stopped"] = False
                final_result["inference_service_stop_error"] = str(exc)
                final_result["inference_service_stop_command"] = "停止推理服务"
                final_result["inference_service_log_command"] = "查看推理服务日志"
        self._stage_result(workflow, "finished", result=final_result)
        stage = workflow["stages"][workflow["current_stage"]]
        for key in (
            "log_path",
            "log_tail",
            "log_updated_at",
            "log_command",
            "stop_service_log_path",
            "stop_service_log_tail",
            "stop_service_log_updated_at",
        ):
            if final_result.get(key) is not None:
                stage[key] = final_result[key]
        self._move_next(workflow)

    def _reconcile_terminal_benchmark(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        stage = (workflow.get("stages") or {}).get("benchmark", {})
        previous_result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
        has_timeout_failure = _has_timeout_signal(workflow.get("error")) or _has_timeout_signal(stage.get("error"))
        if not (
            previous_result.get("status_command")
            or previous_result.get("benchmark_job_id")
            or _benchmark_report_command(previous_result)
            or has_timeout_failure
        ):
            return workflow
        try:
            if self.dependencies.monitor_benchmark:
                result = self.dependencies.monitor_benchmark(workflow)
            elif self.agent_events:
                return workflow
            else:
                result = self.dependencies.inference_command(_benchmark_runtime_status_command(workflow))
        except Exception as exc:
            if not _exception_has_timeout_signal(exc):
                return workflow
            workflow["status"] = "running"
            workflow["error"] = None
            self._record_transient_timeout(workflow, stage, exc)
            return self._save(workflow, allow_terminal_override=True)
        result = _benchmark_enrich_runtime_result({
            **{key: previous_result.get(key) for key in ("status_command", "stop_command", "benchmark_job_id") if previous_result.get(key)},
            **result,
            "result_entry": _benchmark_result_entry(workflow),
        })
        if _result_has_transient_poll_error(result):
            workflow["status"] = "running"
            workflow["error"] = None
            self._record_transient_timeout(
                workflow,
                stage,
                result.get("error") or result.get("message") or result.get("result"),
                {
                    "result": {
                        **{
                            key: previous_result.get(key)
                            for key in ("status_command", "stop_command", "benchmark_job_id", "result_entry")
                            if previous_result.get(key)
                        },
                        **result,
                    },
                },
            )
            return self._save(workflow, allow_terminal_override=True)
        status = _benchmark_explicit_status(result) or result.get("status") or "running"
        if status == "finished":
            workflow["status"] = "running"
            workflow["error"] = None
            self._finish_benchmark_stage(workflow, result)
            return self._save(workflow, allow_terminal_override=True)
        if status in {"failed", "stopped", "interrupted"}:
            return workflow
        workflow["status"] = "running"
        workflow["error"] = None
        self._stage_result(workflow, "running", result=result)
        saved = self._save(workflow, allow_terminal_override=True)
        with self._lock:
            self._worker_workflow_ids.add(str(saved["workflow_id"]))
        return saved

    def advance(self, workflow_id: str) -> Dict[str, Any]:
        with self._lock:
            workflow = self._get(workflow_id)
            if not workflow:
                return {}
            if workflow["status"] != "running":
                logger.debug(
                    "[workflow-debug] advance terminal/non-running workflow_id=%s workflow_status=%s current_stage=%s",
                    workflow_id,
                    workflow.get("status"),
                    workflow.get("current_stage"),
                )
                return self._repair_finished_benchmark_shutdown(workflow) or workflow
            try:
                stage_name = workflow["current_stage"]
                stage = workflow["stages"][stage_name]
                logger.debug(
                    "[workflow-debug] advance workflow_id=%s workflow_status=%s stage=%s stage_status=%s",
                    workflow_id,
                    workflow.get("status"),
                    stage_name,
                    stage.get("status"),
                )
                if self._skip_existing_stage_output(workflow, stage_name, stage):
                    return self._save(workflow)
                if stage_name == "train":
                    self._advance_train(workflow, stage)
                elif stage_name == "evaluate":
                    self._advance_evaluate(workflow, stage)
                elif stage_name == "publish":
                    self._advance_publish(workflow, stage)
                elif stage_name == "deploy":
                    self._advance_deploy(workflow, stage)
                elif stage_name == "benchmark":
                    self._advance_benchmark(workflow, stage)
                logger.debug(
                    "[workflow-debug] advance result workflow_id=%s workflow_status=%s stage=%s stage_status=%s",
                    workflow_id,
                    workflow.get("status"),
                    workflow.get("current_stage"),
                    (workflow.get("stages") or {}).get(workflow.get("current_stage"), {}).get("status"),
                )
                return self._save(workflow)
            except Exception as exc:
                current = self._get(workflow_id)
                if current and current.get("status") in {"stopping", "stopped"}:
                    return current
                logger.exception("Workflow %s failed", workflow_id)
                return self._fail(workflow, exc)

    def _advance_train(self, workflow: Dict[str, Any], stage: Dict[str, Any]) -> None:
        if stage["status"] == "starting_external":
            return
        if stage["status"] == "pending":
            workflow["context"]["train_started_at"] = time.time()
            result = self.dependencies.start_train(workflow)
            self._stage_result(workflow, "running", **self._details(result))
            return
        result = self._monitor_with_timeout(workflow, stage, self.dependencies.monitor_train)
        if result is None:
            return
        if (
            result.get("error") == "pid_ended_no_wandb"
            and train_result_has_startup_evidence(result)
            and time.time() - float(stage.get("updated_at") or workflow["created_at"])
            < self.train_start_grace_seconds
        ):
            stage.update(result)
            stage["status"] = "starting"
            stage["message"] = "训练启动进程已结束，正在等待 wandb 或训练状态记录就绪"
            return
        if train_result_reached_completion(result):
            stage.update(result)
            try:
                workflow["context"]["trained_model_path"] = self.dependencies.find_trained_model(workflow)
            except Exception:
                if result.get("status") in {"failed", "interrupted", "unknown"}:
                    raise RuntimeError(result.get("error") or "训练保存或模型导出失败")
                raise
            self._stage_result(workflow, "finished", **self._details(result))
            self._move_next(workflow)
            return
        if result.get("status") in {"failed", "interrupted", "unknown"}:
            raise RuntimeError(result.get("error") or "模型训练失败")
        if result.get("status") != "finished":
            stage.update(result)
            return
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        if metrics.get("training_process_exists") or metrics.get("pid_alive"):
            stage.update(result)
            stage["status"] = "running"
            stage["error"] = None
            return
        if train_result_has_explicit_incomplete_progress(result):
            current_step = _int_value(metrics.get("current_step") or metrics.get("latest_step"))
            total_steps = _int_value(metrics.get("total_steps"))
            progress = _progress_value(metrics.get("progress_percent"))
            raise RuntimeError(
                "训练进程已结束，但尚未达到最终步"
                f"（step={current_step if current_step is not None else 'N/A'}/"
                f"{total_steps if total_steps is not None else 'N/A'}，"
                f"进度={progress if progress is not None else 'N/A'}%）"
            )
        workflow["context"]["trained_model_path"] = self.dependencies.find_trained_model(workflow)
        self._stage_result(workflow, "finished", **self._details(result))
        self._move_next(workflow)

    def _advance_evaluate(self, workflow: Dict[str, Any], stage: Dict[str, Any]) -> None:
        if stage["status"] in {"awaiting_agent", "starting_external"}:
            return
        if stage["status"] == "pending":
            if self.agent_events:
                self.enqueue_event(
                    workflow,
                    "start_evaluate",
                    {"model_fir": workflow["context"]["trained_model_path"]},
                )
                self._stage_result(workflow, "awaiting_agent")
                return
            result = self.dependencies.start_evaluate(workflow)
            self._stage_result(workflow, "running", **self._details(result))
            return
        result = self._monitor_with_timeout(workflow, stage, self.dependencies.monitor_evaluate)
        if result is None:
            return
        if result.get("status") in {"failed", "interrupted", "unknown"}:
            raise RuntimeError(result.get("error") or "单模型评估失败")
        if result.get("status") != "finished":
            stage.update(result)
            return
        self._stage_result(workflow, "finished", **self._details(result))
        self._move_next(workflow)

    def _advance_publish(self, workflow: Dict[str, Any], stage: Dict[str, Any]) -> None:
        if not self.is_running(workflow["workflow_id"]):
            return
        if stage["status"] == "pending":
            result = self.dependencies.start_publish(workflow)
            self._stage_result(workflow, "running", **self._details(result))
            return
        result = self._monitor_with_timeout(workflow, stage, self.dependencies.monitor_publish)
        if result is None:
            return
        if result.get("status") == "finished":
            published_path = result.get("model_path", "")
            workflow["context"]["published_model_path"] = published_path
            details = self._details(result)
            details["model_path"] = published_path
            self._stage_result(workflow, "finished", **details)
            self._move_next(workflow)
        elif result.get("status") == "failed":
            raise RuntimeError(result.get("error") or "模型发布失败")
        else:
            stage.update(result)

    def _advance_deploy(self, workflow: Dict[str, Any], stage: Dict[str, Any]) -> None:
        if self.agent_events:
            if (
                stage["status"] in {"awaiting_agent", "running", "starting_external"}
                and self.dependencies.monitor_deploy
            ):
                result = self.dependencies.monitor_deploy(workflow)
                logger.debug(
                    "[workflow-debug] deploy monitor workflow_id=%s stage_status=%s result_status=%s all_running=%s message=%s",
                    workflow.get("workflow_id"),
                    stage.get("status"),
                    result.get("status") if isinstance(result, dict) else None,
                    result.get("all_running") if isinstance(result, dict) else None,
                    result.get("message") if isinstance(result, dict) else None,
                )
                if result and result.get("status") == "finished":
                    details = self._details(result)
                    self._stage_result(workflow, "finished", **details)
                    self._move_next(workflow)
                elif result and stage["status"] != "starting_external":
                    stage.update(result)
                    if result.get("status") == "running":
                        stage["status"] = "running"
                return
            if stage["status"] == "pending":
                published_path = workflow["context"]["published_model_path"]
                model_name = Path(published_path).name
                self.enqueue_event(
                    workflow,
                    "deploy_model",
                    {"model_name": model_name},
                )
                self._stage_result(workflow, "awaiting_agent")
            return
        published_path = workflow["context"]["published_model_path"]
        model_name = Path(published_path).name
        current = self.dependencies.inference_command("查看推理配置")
        old_model_name = current.get("model_name")
        old_model_name_parsed = old_model_name is not None
        if not old_model_name_parsed:
            old_model_name = Path(current.get("model_path") or "").name
            old_model_name_parsed = bool(old_model_name)
        workflow["context"]["old_model_name"] = old_model_name
        try:
            self.dependencies.inference_command(f"修改推理配置 MODEL_NAME={model_name}")
            restarted = self.dependencies.inference_command("重启推理服务")
            checked = self.dependencies.inference_command("查看推理服务状态")
            if not restarted.get("success", True) or not checked.get("all_running", False):
                raise RuntimeError(restarted.get("error") or checked.get("error") or "推理服务重启后未全部运行")
        except Exception:
            if old_model_name_parsed:
                self.dependencies.inference_command(f"修改推理配置 MODEL_NAME={old_model_name}")
                self.dependencies.inference_command("重启推理服务")
            raise
        self._stage_result(
            workflow,
            "finished",
            model_name=model_name,
            service_log_command="查看推理服务日志",
        )
        self._move_next(workflow)

    def _advance_benchmark(self, workflow: Dict[str, Any], stage: Dict[str, Any]) -> None:
        previous_result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
        if stage["status"] in {"awaiting_agent", "starting_external"}:
            has_runtime_handle = bool(
                previous_result.get("status_command")
                or previous_result.get("benchmark_job_id")
                or _benchmark_report_command(previous_result)
            )
            logger.debug(
                "[workflow-debug] benchmark awaitable workflow_id=%s stage_status=%s has_runtime_handle=%s job_id=%s status_command=%s",
                workflow.get("workflow_id"),
                stage.get("status"),
                has_runtime_handle,
                previous_result.get("benchmark_job_id"),
                previous_result.get("status_command"),
            )
            if not has_runtime_handle:
                return
        if stage["status"] == "pending":
            if self.agent_events:
                self.enqueue_event(
                    workflow,
                    "start_benchmark",
                    {"benchmark": _benchmark_name(workflow)},
                )
                self._stage_result(workflow, "awaiting_agent")
                return
            if self.dependencies.start_benchmark:
                result = self.dependencies.start_benchmark(workflow)
            else:
                result = self.dependencies.inference_command(_benchmark_start_command(workflow))
            if not result.get("success", True):
                raise RuntimeError(result.get("error") or f"{_benchmark_name(workflow)}基准评测启动失败")
            result = _benchmark_enrich_runtime_result({**result, "result_entry": _benchmark_result_entry(workflow)})
            self._stage_result(workflow, "running", result=result)
            return
        try:
            if self.dependencies.monitor_benchmark:
                result = self.dependencies.monitor_benchmark(workflow)
            elif self.agent_events:
                stage["updated_at"] = time.time()
                stage["message"] = stage.get("message") or "等待结构化 benchmark 状态服务返回"
                return
            else:
                result = self.dependencies.inference_command(_benchmark_runtime_status_command(workflow))
        except Exception as exc:
            if _exception_has_timeout_signal(exc):
                self._record_transient_timeout(workflow, stage, exc)
                return
            raise
        result = _benchmark_enrich_runtime_result({
            **{key: previous_result.get(key) for key in ("status_command", "stop_command", "benchmark_job_id") if previous_result.get(key)},
            **result,
            "result_entry": _benchmark_result_entry(workflow),
        })
        if _result_has_transient_poll_error(result):
            fallback = self._benchmark_result_fallback(workflow)
            if fallback:
                self._finish_benchmark_stage(workflow, fallback)
                return
            self._record_transient_timeout(
                workflow,
                stage,
                result.get("error") or result.get("message") or result.get("result"),
                {
                    "result": {
                        **{
                            key: previous_result.get(key)
                            for key in ("status_command", "stop_command", "benchmark_job_id", "result_entry")
                            if previous_result.get(key)
                        },
                        **result,
                    },
                },
            )
            return
        status = _benchmark_explicit_status(result) or result.get("status") or "running"
        logger.debug(
            "[workflow-debug] benchmark monitor workflow_id=%s resolved_status=%s raw_status=%s job_id=%s status_command=%s result_entry=%s",
            workflow.get("workflow_id"),
            status,
            result.get("status"),
            result.get("benchmark_job_id"),
            result.get("status_command"),
            result.get("result_entry"),
        )
        if status in {"failed", "stopped", "interrupted"}:
            raise RuntimeError(result.get("error") or f"{_benchmark_name(workflow)}基准评测失败")
        if status != "finished":
            stage["result"] = result
            stage["status"] = "running"
            stage["updated_at"] = time.time()
            for key in ("log_path", "log_tail", "log_updated_at", "log_command"):
                if result.get(key) is not None:
                    stage[key] = result[key]
            stage.pop("last_poll_error", None)
            stage.pop("error", None)
            return
        self._finish_benchmark_stage(workflow, result)
