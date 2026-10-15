"""Durable execution intents and an ordered outbox for Studio experiment records.

Only the Python standard library is required. No network call is made at import.
The local journal is authoritative for launch claims; losing a response never
authorizes launching another process for the same request key.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.request import Request, urlopen
import uuid

from .experiment_process import redact

logger = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled"}
ENV_ALLOWLIST = {
    "MODEL_PATH", "DATASET_DIR", "DATASET_DATE", "TEM", "MBS", "ACC", "LR", "RESUME", "CKPT_PATH",
    "CUDA_VISIBLE_DEVICES", "MEDFLOW_ASSIGNED_GPUS", "MEDFLOW_TRAINING_RESERVATION_ID",
    "WANDB_PROJECT", "WANDB_RUN_ID", "WANDB_MODE", "WANDB_DIR", "WORLD_SIZE", "NNODES",
    "MASTER_ADDR", "MASTER_PORT", "NCCL_SOCKET_IFNAME", "USE_LIGER_KERNEL", "USE_ADAM_MINI",
    'MEDFLOW_EVAL_TASK_DIR', 'MEDFLOW_EVAL_BASE_MODEL',
    'MEDFLOW_EVAL_IMAGE_ID',
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)



def safe_environment(env: dict[str, Any]) -> dict[str, Any]:
    return redact({key: value for key, value in env.items() if key in ENV_ALLOWLIST})


def studio_connection() -> tuple[str, str]:
    from .runtime_credentials import studio_connection as connection
    return connection()


def studio_request(procedure: str, payload: dict[str, Any]) -> dict[str, Any]:
    url, token = studio_connection()
    request = Request(f"{url}/trpc/{procedure}", data=canonical(payload).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "X-MedFlow-Runtime-Token": token})
    with urlopen(request, timeout=20) as response:
        result = json.load(response)
    for key in ("result", "data", "json"):
        if isinstance(result, dict) and key in result:
            result = result[key]
    if not isinstance(result, dict) or result.get("success") is False:
        raise ValueError("Studio experiment request failed")
    return result.get("data", result)


class ExperimentTracker:
    def __init__(self, db_path: str | Path, node_id: str,
                 send: Callable[[str, dict[str, Any]], dict[str, Any]] = studio_request, *, collector_fn=None):
        self.db_path = str(db_path)
        self.node_id = node_id
        self.send = send
        self.collector_fn = collector_fn
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.collector_worker: threading.Thread | None = None
        self.log_worker: threading.Thread | None = None
        self.flush_lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiment_intent (
                    request_key TEXT PRIMARY KEY, request_json TEXT NOT NULL,
                    handle_json TEXT, claimed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_execution (
                    attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, experiment_id TEXT NOT NULL,
                    container TEXT NOT NULL, actor_user_id TEXT NOT NULL,
                    state TEXT NOT NULL, last_sequence INTEGER NOT NULL DEFAULT 0,
                    runtime_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_outbox (
                    event_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT, UNIQUE(attempt_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_outbox_pending
                    ON experiment_outbox(acknowledged, attempt_id, sequence);
            """)
            from .experiment_logs import initialize
            initialize(db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def prepare(self, *, request_key: str, actor_user_id: str, name: str, kind: str,
                container: str, spec: dict[str, Any], context: dict[str, Any] | None = None,
                inputs: list[dict[str, str]] | None = None) -> dict[str, Any]:
        if not self.node_id or not actor_user_id:
            raise ValueError("实验记录缺少 Runtime 节点或已认证用户")
        context = context or {}
        request = {"runtimeNodeId": self.node_id, "actorUserId": actor_user_id,
                   "requestKey": request_key, "name": name, "kind": kind, "container": container,
                   "spec": redact(spec), "inputs": inputs or [], "project": context.get("project", "default")}
        for key in ("experimentId", "taskId", "conversationRunId", "workflowId", "resumeFromVersionId", "resourceReservationId"):
            if context.get(key):
                request[key] = context[key]
        encoded = canonical(request)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM experiment_intent WHERE request_key=?", (request_key,)).fetchone()
            if existing:
                if existing["request_json"] != encoded:
                    raise ValueError("Experiment request key was already used for different inputs")
                if existing["handle_json"]:
                    return json.loads(existing["handle_json"])
            else:
                db.execute("INSERT INTO experiment_intent(request_key,request_json,created_at) VALUES (?,?,?)",
                           (request_key, encoded, utc_now()))
        # Durable intent precedes remote creation; the remote request key handles lost responses.
        handle = self.send("prepareExperimentTask", request)
        for key in ("experimentId", "taskId", "attemptId"):
            if not handle.get(key):
                raise ValueError("Incomplete experiment execution handle")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE experiment_intent SET handle_json=? WHERE request_key=?", (canonical(handle), request_key))
            db.execute("""INSERT OR IGNORE INTO experiment_execution
                (attempt_id,task_id,experiment_id,container,actor_user_id,state,updated_at)
                VALUES (?,?,?,?,?,?,?)""", (handle["attemptId"], handle["taskId"], handle["experimentId"],
                                          container, actor_user_id, handle.get("status", "prepared"), utc_now()))
            if handle.get('logCaptureEnabled'):
                db.execute('INSERT OR IGNORE INTO experiment_log_enabled(attempt_id) VALUES (?)', (handle['attemptId'],))
        return handle

    def claim_launch(self, request_key: str) -> bool:
        with self.connect() as db:
            changed = db.execute("UPDATE experiment_intent SET claimed=1 WHERE request_key=? AND claimed=0 AND handle_json IS NOT NULL",
                                 (request_key,)).rowcount
        return changed == 1

    def emit(self, attempt_id: str, event_type: str, payload: dict[str, Any], *, event_key: str | None = None,
             preserve_terminal: bool = False) -> dict[str, Any]:
        if event_type not in {"status", "configuration", "artifact", "evaluation", "service_operation"}:
            raise ValueError("Unsupported experiment event type")
        payload = redact(payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event_id = self.event_id(attempt_id, event_key) if event_key is not None else f"evt-{uuid.uuid4()}"
            prior = db.execute('SELECT event_json FROM experiment_outbox WHERE event_id=?', (event_id,)).fetchone()
            if prior:
                event = json.loads(prior['event_json'])
                if event['type'] != event_type or event['payload'] != payload:
                    raise ValueError('Immutable collector event conflicts with its previous payload')
                return event
            row = db.execute("SELECT * FROM experiment_execution WHERE attempt_id=?", (attempt_id,)).fetchone()
            if not row:
                raise ValueError("Unknown experiment execution")
            state, runtime = row["state"], json.loads(row["runtime_json"])
            if event_type == "status":
                new_state = payload.get("status")
                if new_state not in {"prepared", "starting", "running", "unknown", *TERMINAL}:
                    raise ValueError("Invalid execution state")
                if state in TERMINAL and state != new_state:
                    if preserve_terminal:
                        new_state = state
                        payload['status'] = state
                        payload.pop('error', None)
                    else:
                        raise ValueError("Terminal execution state is immutable")
                order = {"prepared": 0, "starting": 1, "running": 2}
                if new_state in order and state in order and order[new_state] < order[state]:
                    raise ValueError("Execution state cannot move backwards")
                state = new_state
                runtime.update(payload.get("runtime") or {})
            event = {"eventId": event_id, "attemptId": attempt_id,
                     "sequence": row["last_sequence"] + 1, "occurredAt": utc_now(), "type": event_type, "payload": payload}
            db.execute("INSERT INTO experiment_outbox(event_id,attempt_id,sequence,event_json) VALUES (?,?,?,?)",
                       (event["eventId"], attempt_id, event["sequence"], canonical(event)))
            db.execute("UPDATE experiment_execution SET state=?,runtime_json=?,last_sequence=?,updated_at=? WHERE attempt_id=?",
                       (state, canonical(runtime), event["sequence"], utc_now(), attempt_id))
        return event

    @staticmethod
    def event_id(attempt_id: str, event_key: str) -> str:
        return 'evt-' + hashlib.sha256(canonical([attempt_id, event_key]).encode()).hexdigest()

    def emit_once(self, attempt_id, event_type, payload, event_key):
        return self.emit(attempt_id, event_type, payload, event_key=event_key)

    def event_payload(self, attempt_id, event_key):
        with self.connect() as db:
            row = db.execute('SELECT event_json FROM experiment_outbox WHERE event_id=?',
                             (self.event_id(attempt_id, event_key),)).fetchone()
        return json.loads(row['event_json'])['payload'] if row else None

    def latest_payload(self, attempt_id, event_type):
        with self.connect() as db:
            rows = db.execute('SELECT event_json FROM experiment_outbox WHERE attempt_id=? ORDER BY sequence DESC',
                              (attempt_id,)).fetchall()
        return next((event['payload'] for row in rows if (event := json.loads(row['event_json']))['type'] == event_type), None)

    def collectable_executions(self):
        with self.connect() as db:
            rows = db.execute('SELECT attempt_id, runtime_json FROM experiment_execution').fetchall()
        return [self.execution(row['attempt_id']) for row in rows
                if ((runtime := json.loads(row['runtime_json'])).get('experimentDirectory') or runtime.get('operation') == 'dataset_upload')
                and runtime.get('archiveStatus') != 'complete']

    def execution(self, attempt_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM experiment_execution WHERE attempt_id=?", (attempt_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["runtime"] = json.loads(result.pop("runtime_json"))
        return result

    def find_execution(self, container: str, pid: str) -> dict[str, Any] | None:
        # PID by itself is not a durable identity; ignore ambiguous/reused bindings.
        with self.connect() as db:
            rows = db.execute("SELECT * FROM experiment_execution WHERE container=?", (container,)).fetchall()
        matches = [dict(row) for row in rows if str(json.loads(row["runtime_json"]).get("pid", "")) == str(pid)]
        return self.execution(matches[0]["attempt_id"]) if len(matches) == 1 else None

    def flush_once(self) -> int:
        if not self.flush_lock.acquire(blocking=False):
            return 0
        acknowledged_count = 0
        try:
            with self.connect() as db:
                attempts = db.execute("SELECT DISTINCT attempt_id FROM experiment_outbox WHERE acknowledged=0").fetchall()
            for attempt in attempts:
                with self.connect() as db:
                    rows = db.execute("""SELECT * FROM experiment_outbox WHERE acknowledged=0 AND attempt_id=?
                        ORDER BY sequence LIMIT 100""", (attempt["attempt_id"],)).fetchall()
                if not rows:
                    continue
                try:
                    result = self.send("appendExperimentEvents", {"runtimeNodeId": self.node_id,
                        "events": [json.loads(row["event_json"]) for row in rows]})
                    acknowledged = set(result.get("acknowledged") or [])
                    sent = {row["event_id"] for row in rows}
                    if not acknowledged.issubset(sent):
                        raise ValueError("Studio acknowledged an event outside the submitted batch")
                    with self.connect() as db:
                        for event_id in acknowledged:
                            db.execute("UPDATE experiment_outbox SET acknowledged=1,last_error=NULL WHERE event_id=?", (event_id,))
                    acknowledged_count += len(acknowledged)
                except Exception as exc:
                    with self.connect() as db:
                        db.execute("UPDATE experiment_outbox SET last_error=? WHERE acknowledged=0 AND attempt_id=?",
                                   (str(redact(str(exc)))[:2000], attempt["attempt_id"]))
                    logger.warning("Experiment events remain queued for %s: %s", attempt["attempt_id"], type(exc).__name__)
            return acknowledged_count
        finally:
            self.flush_lock.release()

    def pending_count(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM experiment_outbox WHERE acknowledged=0").fetchone()[0]

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        collector = self.collector_fn
        if collector is None:
            from .experiment_collector import reconcile_all
            collector = reconcile_all
        self.stop_event.clear()
        def loop():
            while not self.stop_event.is_set():
                try:
                    self.flush_once()
                except Exception:
                    logger.exception("Experiment outbox poll failed")
                self.stop_event.wait(10)
        self.worker = threading.Thread(target=loop, name="experiment-outbox", daemon=True)
        self.worker.start()
        def collect_loop():
            while not self.stop_event.wait(15):
                try:
                    collector(self)
                except Exception:
                    logger.exception('Experiment collection failed; will retry')
        self.collector_worker = threading.Thread(target=collect_loop, name='experiment-collector', daemon=True)
        self.collector_worker.start()
        def log_loop():
            from .experiment_logs import sync_all
            while not self.stop_event.wait(2.5):
                try:
                    sync_all(self)
                except Exception:
                    logger.exception('Experiment log collection failed; will retry')
        self.log_worker = threading.Thread(target=log_loop, name='experiment-logs', daemon=True)
        self.log_worker.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=1)
        if self.collector_worker:
            self.collector_worker.join(timeout=1)
        if self.log_worker:
            self.log_worker.join(timeout=1)


_tracker: ExperimentTracker | None = None
_tracker_lock = threading.Lock()


def get_experiment_tracker() -> ExperimentTracker:
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            default = Path(__file__).resolve().parents[3] / "runtime" / "experiments.sqlite3"
            _tracker = ExperimentTracker(os.getenv("MEDFLOW_EXPERIMENT_DB", str(default)),
                                         os.getenv("MEDFLOW_RESOURCE_NODE_ID", "").strip())
            _tracker.start()
    return _tracker
