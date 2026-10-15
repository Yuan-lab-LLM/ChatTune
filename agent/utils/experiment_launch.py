"""Adapter between existing tool launchers/monitors and experiment records."""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Callable
import uuid
import threading
import time

from .experiment_tracker import get_experiment_tracker, redact, safe_environment, TERMINAL
from .experiment_capture import pin_launch_inputs
from .experiment_collector import prepare_container_execution

logger = logging.getLogger(__name__)


@contextmanager
def _timed_stage(script_name: str, stage: str):
    started = time.monotonic()
    try:
        yield
    except Exception:
        logger.info("Experiment launch timing: script=%s stage=%s result=failed elapsedMs=%.1f",
                    script_name, stage, (time.monotonic() - started) * 1000)
        raise
    else:
        logger.info("Experiment launch timing: script=%s stage=%s result=ok elapsedMs=%.1f",
                    script_name, stage, (time.monotonic() - started) * 1000)


class ExperimentLaunchUncertain(RuntimeError):
    experiment_launch_uncertain = True


def launch_with_experiment(launcher: Callable[..., Any], run_kwargs: dict[str, Any], *,
                           actor_user_id: str, script_name: str, kind: str,
                           requested: dict[str, Any] | None = None,
                           context: dict[str, Any] | None = None) -> Any:
    tracker = get_experiment_tracker()
    reservation = (run_kwargs.get('env_vars') or {}).get('MEDFLOW_TRAINING_RESERVATION_ID')
    stop, failures = threading.Event(), []
    def renew():
        tracker.send('renewTrainingResourcesForRuntime', {'reservationId': reservation, 'runtimeNodeId': tracker.node_id})
    def keep_alive():
        while not stop.wait(30):
            try:
                renew()
            except Exception as exc:
                failures.append(exc)
                return
    def check():
        if failures:
            raise ValueError('GPU reservation renewal failed while freezing experiment inputs') from failures[0]
        if reservation:
            with _timed_stage(script_name, "reservation_recheck"):
                renew()  # Revalidate immediately before launch, after potentially long snapshot copies.
    if reservation:
        with _timed_stage(script_name, "reservation_renew"):
            renew()
        threading.Thread(target=keep_alive, name='experiment-preparation-lease', daemon=True).start()
    try:
        with _timed_stage(script_name, "total"):
            return _launch_with_experiment(launcher, run_kwargs, actor_user_id=actor_user_id, script_name=script_name,
                                          kind=kind, requested=requested, context=context, preparation_check=check)
    finally:
        stop.set()


def _launch_with_experiment(launcher, run_kwargs, *, actor_user_id, script_name, kind,
                            requested=None, context=None, preparation_check=lambda: None):
    tracker = get_experiment_tracker()
    context = context if context is not None else {}
    request_key = context.setdefault("requestKey", f"request-{uuid.uuid4()}")
    # Authorize creation before accessing any paths in future remote requests;
    # current launchers have already performed their dataset/model access preflight.
    with _timed_stage(script_name, "pin_inputs"):
        assets = pin_launch_inputs(run_kwargs, node_id=tracker.node_id, owner_id=actor_user_id, kind=kind,
                                   resume_checkpoint_name=context.get('resumeCheckpointName'),
                                   replay_assets=context.get('replayInputAssets'))
    if 'expectedInputVersions' in context:
        expected = sorted((asset['assetVersionId'], asset['role'], asset['fingerprint']) for asset in context['expectedInputVersions'])
        actual = sorted((asset['id'], asset['role'], asset['fingerprint']) for asset in assets)
        if expected != actual:
            raise ValueError('Draft input versions are unavailable or changed; select and record new versions explicitly')
    env = dict(run_kwargs.get("env_vars") or {})
    training_env = {key: value for key, value in safe_environment(env).items()
                    if key not in {"CUDA_VISIBLE_DEVICES", "MEDFLOW_ASSIGNED_GPUS", "MEDFLOW_TRAINING_RESERVATION_ID",
                                   "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "NNODES", "WANDB_RUN_ID", "WANDB_DIR"}}
    spec = {"schemaVersion": 1, "script": script_name, 'kind': kind,
            "requested": redact(requested or {}),
            "inputVersions": [{"assetVersionId": asset["id"], "role": asset["role"], "fingerprint": asset["fingerprint"]} for asset in assets],
            "resolved": {"scriptPath": run_kwargs.get("script_path"),
                         "arguments": redact({key: value for key, value in (run_kwargs.get("script_args") or {}).items()
                             if key not in {"allocation-file", "master-addr", "master-port", "hostfile", "run_id", "run-id"}}),
                         "positionalArguments": redact(run_kwargs.get("positional_args") or []),
                         "environment": training_env}}
    resume_names = [asset['metadata']['resumeDirectoryName'] for asset in assets
                    if asset.get('metadata', {}).get('resumeDirectoryName')]
    if resume_names:
        spec['resolved']['resumeCheckpointName'] = resume_names[0]
    with _timed_stage(script_name, "record_prepare"):
        handle = tracker.prepare(request_key=request_key, actor_user_id=actor_user_id,
                                 name=context.get("name") or script_name, kind=kind,
                                 container=str(run_kwargs.get("docker_container") or "local"),
                                 spec=spec, context=context, inputs=context.get("inputs"))
        # Do not mutate the supplied request identity: a retry of this call must replay exactly the same request.
        context["handle"] = handle
        if not tracker.claim_launch(request_key):
            state = tracker.execution(handle["attemptId"])
            raise ValueError(f"实验执行 {handle['attemptId']} 已登记启动，当前状态 {state['state'] if state else 'unknown'}；请查询原任务，不能重复启动")
    attempt_id = handle["attemptId"]
    try:
        with _timed_stage(script_name, "container_prepare"):
            archive_runtime = prepare_container_execution(run_kwargs, handle=handle, actor_user_id=actor_user_id,
                                                         script_name=script_name, assets=assets)
            preparation_check()
    except Exception as exc:
        # Only local preparation has happened: no training process has been requested.
        tracker.emit(attempt_id, 'status', {'status': 'failed', 'error': f'Execution preparation failed: {exc}'})
        raise
    with _timed_stage(script_name, "record_events"):
        env = dict(run_kwargs.get('env_vars') or {})
        for asset in assets:
            tracker.emit(attempt_id, "artifact", asset)
        tracker.emit(attempt_id, "configuration", {"configuration": {
            "schemaVersion": 1, "requested": spec["requested"], "resolved": spec["resolved"],
            "execution": {"arguments": redact(run_kwargs.get('script_args') or {}),
                          "positionalArguments": redact(run_kwargs.get('positional_args') or []),
                          "environment": safe_environment(env), "inputTransforms": archive_runtime.get('inputTransforms', []),
                          "runtimeCompatibility": archive_runtime.get('runtimeCompatibility', [])},
            "actual": None, "completeness": "incomplete", "missing": ["trainer_resolved_configuration"]},
            "environment": safe_environment(env)})
        env.update({"MEDFLOW_EXPERIMENT_ID": handle["experimentId"], "MEDFLOW_EXPERIMENT_TASK_ID": handle["taskId"],
                    "MEDFLOW_EXPERIMENT_ATTEMPT_ID": attempt_id})
        run_kwargs["env_vars"] = env
        tracker.emit(attempt_id, "status", {"status": "starting", "runtime": {**archive_runtime,
            'reservationId': env.get('MEDFLOW_TRAINING_RESERVATION_ID'), 'kind': kind, 'scriptName': script_name}})
    try:
        with _timed_stage(script_name, "process_start"):
            response = launcher(**run_kwargs)
    except Exception as exc:
        # Launcher exceptions may occur after a process was created. Keep the launch claim and reconcile.
        tracker.emit(attempt_id, "status", {"status": "unknown", "error": str(exc)}, preserve_terminal=True)
        raise ExperimentLaunchUncertain(f"实验 {attempt_id} 启动结果待核对：{exc}") from exc
    metadata = dict(response.metadata or {})
    hint = dict(metadata.get("protocol_hint") or {})
    runtime = {key: hint[key] for key in ("pid", "container", "runId", "launchLogPath", "logPath", "outputDir") if hint.get(key) is not None}
    runtime.update({"scriptName": script_name, "kind": kind,
                    "reservationId": env.get("MEDFLOW_TRAINING_RESERVATION_ID")})
    if metadata.get("success"):
        status = "running" if runtime.get("pid") else "unknown"
    else:
        # A tool error alone is not evidence that no child process exists.
        status = "unknown"
    recorded = tracker.emit(attempt_id, "status", {"status": status, "runtime": runtime,
                 **({"error": str(hint.get("errorReason") or hint.get("message") or "启动结果待核对")} if status == "unknown" else {})},
                 preserve_terminal=True)
    status = recorded['payload']['status']
    if status in TERMINAL:
        metadata['success'] = status == 'succeeded'
        hint['status'] = 'finished' if status == 'succeeded' else status
    hint.update({"experimentId": handle["experimentId"], "experimentTaskId": handle["taskId"], "experimentAttemptId": attempt_id})
    if kind == 'preprocess' and (run_kwargs.get('script_args') or {}).get('output_folder'):
        hint['outputFolder'] = run_kwargs['script_args']['output_folder']
    evaluated = next((asset for asset in assets if asset["role"] == "evaluated_model"), None)
    if evaluated:
        hint["evaluatedModelPath"] = (env['CKPT_PATH'] if script_name == 'ckpt_eval'
                                       else evaluated["location"]["path"])
    metadata.update({"experiment": handle, "protocol_hint": hint})
    if status == "unknown":
        metadata["experiment_launch_uncertain"] = True
    response.metadata = metadata
    return response


def request_workflow_cancellation(workflow: dict[str, Any]) -> None:
    """Persist local stage cancellation before sending any process stop signal."""
    stage_name = workflow.get('current_stage')
    if stage_name not in {'train', 'evaluate', 'publish'}:
        return  # Native inference stages record stop intent in their own worker.
    stage = (workflow.get('stages') or {}).get(stage_name) or {}
    recorded = stage.get('result') if isinstance(stage.get('result'), dict) else {}
    attempt = stage.get('experiment_attempt_id') or recorded.get('experiment_attempt_id')
    if not attempt:
        return  # Historical workflows may predate experiment tracking.
    tracker = get_experiment_tracker()
    execution = tracker.execution(attempt)
    context = workflow.get('context') or {}
    experiment = stage.get('experiment_id') or recorded.get('experiment_id') or context.get('experiment_id')
    task = stage.get('experiment_task_id') or recorded.get('experiment_task_id')
    if (not execution or execution['actor_user_id'] != str(context.get('actor_user_id') or '')
            or (experiment and execution['experiment_id'] != experiment)
            or (task and execution['task_id'] != task)):
        raise ValueError('Execution record is unavailable for this workflow owner and stage')
    if execution['state'] in TERMINAL or execution['runtime'].get('cancelRequested'):
        return
    tracker.emit(attempt, 'status', {'status': execution['state'], 'runtime': {
        'cancelRequested': True, 'cancelWorkflowId': workflow['workflow_id']}}, preserve_terminal=True)


def _medical_evaluation_observation(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Keep bounded monitor observations separate from verified asset versions.

    Legacy files can have shared names. Existence/scores are useful telemetry,
    but cannot establish ownership or archive completion for this attempt.
    Never copy patient responses, report bodies or log tails into the record.
    """
    stages = metrics.get('stages')
    if not isinstance(stages, list):
        return None

    def text(value, limit=4096):
        return value[:limit] if isinstance(value, str) else None

    rows = []
    for item in stages[:100]:
        if not isinstance(item, dict) or not text(item.get('name')):
            continue
        result = item.get('result') if isinstance(item.get('result'), dict) else {}
        scores = {key: value for key, value in result.items()
                  if key in {'average', 'accuracy', 'mean_fuzziness'}
                  and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
        progress = item.get('progress_summary') or result.get('progress_summary') or {}
        rows.append({'name': text(item['name'], 200), 'path': text(item.get('path')),
                     'status': item.get('status') if item.get('status') in {'finished', 'running', 'pending'} else 'unknown',
                     'exists': bool(item.get('output_meta')) or item.get('status') == 'finished',
                     'scores': scores, 'logPath': text(item.get('progress_log')),
                     'logError': isinstance(progress, dict) and progress.get('error_detected') is True})
    artifacts = metrics.get('artifacts')
    for item in artifacts[:100] if isinstance(artifacts, list) else []:
        if not isinstance(item, dict) or not text(item.get('name')):
            continue
        rows.append({'name': text(item['name'], 200), 'path': text(item.get('path')),
                     'status': 'finished' if item.get('exists') is True else 'pending',
                     'exists': item.get('exists') is True, 'scores': {}})
    return {'source': 'medical_monitor', 'currentStage': text(metrics.get('current_stage'), 200),
            'items': rows}


def record_monitor_result(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    container = str(metrics.get("container_name") or payload.get("container") or "")
    pid = str(metrics.get("pid") or payload.get("pid") or "")
    if not container or not pid:
        return
    tracker = get_experiment_tracker()
    execution = tracker.find_execution(container, pid)
    if not execution:
        return
    raw = payload.get("status")
    status = {"finished": "succeeded", "failed": "failed", "stopped": "cancelled", "interrupted": "failed",
              "running": "running", "starting": "starting", "unknown": "unknown"}.get(raw)
    if not status:
        return  # A polling timeout does not change execution state.
    if execution['runtime'].get('experimentDirectory') and (status in TERMINAL or execution['state'] in TERMINAL):
        if status in TERMINAL and execution['state'] not in TERMINAL:
            # The monitor can finish before the background archive reaches this attempt.
            # Only the matching supervisor journal may confirm a terminal state.
            from .experiment_collector import observe_execution_status
            observe_execution_status(tracker, execution)
            execution = tracker.execution(execution['attempt_id'])
        status = execution['state']  # Preserve telemetry; supervisor evidence decides terminal status.
    if status == "succeeded" and (metrics.get("training_process_exists") or metrics.get("pid_alive")):
        status = "running"
    if status == "starting" and execution["state"] == "running":
        return
    runtime = {key: metrics[key] for key in ("pid", "output_dir", "wandb_url", "wandb_run_dir", "log_file",
        "current_step", "total_steps", "latest_loss", "run_id", "report_path") if metrics.get(key) is not None}
    if execution['runtime'].get('scriptName') in {'single_model_evaluation_vpn', 'compare_between_models_vpn', 'ckpt_eval'}:
        observation = _medical_evaluation_observation(metrics)
        if observation is not None:
            runtime['evaluationObservation'] = observation
    merged = {**execution["runtime"], **runtime}
    if execution["state"] == status and execution["runtime"] == merged:
        return
    tracker.emit(execution["attempt_id"], "status", {"status": status, "runtime": runtime,
        **({"error": str(metrics.get("error_detail") or metrics.get("error") or payload.get("error") or "任务执行失败")}
           if status == "failed" else {})}, preserve_terminal=True)


def try_record_monitor_result(payload: dict[str, Any]) -> None:
    try:
        record_monitor_result(payload)
    except Exception:
        logger.exception("Failed to persist experiment monitor result")
