"""Collect and pin task inputs inside the existing Docker container."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from .experiment_tracker import canonical

logger = logging.getLogger(__name__)


def container_python(container: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', container):
        raise ValueError('Invalid task container')
    executable = None
    for candidate in ("python3", "python", "/opt/conda/bin/python"):
        result = subprocess.run(["docker", "exec", container, candidate, "--version"], capture_output=True, timeout=10)
        if result.returncode == 0:
            executable = candidate
            break
    if not executable:
        raise ValueError("Container requires Python to record immutable inputs")
    return executable


def container_json(container: str, module: str, request: dict[str, Any], *arguments: str,
                   timeout=60) -> dict[str, Any]:
    executable = container_python(container)
    code = Path(__file__).with_name(module).read_text(encoding='utf-8')
    result = subprocess.run(['docker', 'exec', '-i', container, executable, '-c', code, *arguments],
        input=json.dumps(request), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError(f'Container experiment operation failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout)


def container_asset(container: str, source: str, owner_id: str, kind: str, operation="snapshot") -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner_id):
        raise ValueError("Invalid experiment asset owner")
    if not source.startswith("/") or ".." in PurePosixPath(source).parts:
        raise ValueError("Experiment input must be an absolute container path")
    request = {"operation": operation, "path": source,
               'assetKind': kind,
               "versionRoot": f"/home/workspace/medflow/users/{owner_id}/experiment_assets/{kind}"}
    return container_json(container, 'experiment_assets.py', request, timeout=3600)


def asset_registration(capture: dict[str, Any], *, node_id: str, container: str, owner_id: str,
                       source: str, kind: str, role: str, direction="input") -> dict[str, Any]:
    fingerprint = capture["fingerprint"]
    asset_id = "av-" + hashlib.sha256(canonical([owner_id, node_id, container, kind, fingerprint]).encode()).hexdigest()
    return {"id": asset_id, "kind": kind, "role": role, "direction": direction,
            "logicalName": PurePosixPath(source).name, "fingerprint": fingerprint,
            "integrity": "verified", "location": {"container": container, "path": capture.get("path", source)},
            "manifest": {key: capture[key] for key in ("schemaVersion", "files", "byteCount", "fileCount")},
            "metadata": {"sourcePath": source, "storage": "snapshot" if capture.get("path") else "external"}}


def recorded_input(replay_assets, container, asset_kind, role, dataset_name=None):
    """Resolve one explicit draft binding; never substitute a current source path."""
    if not isinstance(replay_assets, list):
        raise ValueError('Draft input bindings must be a list')
    matches = [asset for asset in replay_assets if isinstance(asset, dict) and asset.get('kind') == asset_kind and
               asset.get('role') == role and (dataset_name is None or
               asset.get('metadata', {}).get('datasetName') == dataset_name)]
    if len(matches) != 1:
        raise ValueError(f'Draft requires one recorded {dataset_name or role} input version')
    asset = matches[0]
    location = asset.get('location') or {}
    if location.get('container') != container:
        raise ValueError('Draft input version belongs to another container')
    path = str(location.get('path') or '')
    if not path.startswith('/') or '..' in PurePosixPath(path).parts:
        raise ValueError('Draft input version must have an absolute container path')
    return asset


def pin_launch_inputs(run_kwargs: dict[str, Any], *, node_id: str, owner_id: str,
                      kind: str, capture_fn=container_asset, resume_checkpoint_name=None,
                      replay_assets=None) -> list[dict[str, Any]]:
    container = str(run_kwargs.get("docker_container") or "")
    if not container:
        raise ValueError("Experiment input capture requires a task container")
    env = dict(run_kwargs.get("env_vars") or {})
    args = dict(run_kwargs.get("script_args") or {})
    registrations: dict[str, dict[str, Any]] = {}
    captures: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    def replay_source(asset_kind, role, dataset_name=None):
        return recorded_input(replay_assets, container, asset_kind, role, dataset_name)

    def pin(path: str, asset_kind: str, role: str) -> str:
        if not path or not path.startswith("/"):
            raise ValueError(f"{role} input must be resolved to an absolute path before execution")
        key = (path, asset_kind)
        if key not in captures:
            captured = capture_fn(container, path, owner_id, asset_kind)
            if isinstance(captured.get("copyStats"), dict):
                logger.info("Experiment input snapshot copy: kind=%s role=%s copyStats=%s",
                            asset_kind, role, captured["copyStats"])
            registration = asset_registration(captured, node_id=node_id, container=container, owner_id=owner_id,
                                              source=path, kind=asset_kind, role=role)
            # File inputs are stored under a version directory but remain file arguments.
            root = captured["path"]
            if captured.get("sourceIsFile"):
                root = f"{root}/{PurePosixPath(path).name}"
            captures[key] = (root, registration)
        replacement, registration = captures[key]
        registrations[f"{registration['id']}:{role}"] = {**registration, "role": role}
        return replacement

    mappings = {
        "MODEL_PATH": ("model", "base_model" if kind == "train" else "processing_model" if kind == "preprocess" else "evaluated_model"),
        "DATASET_DIR": ("dataset", "source_dataset" if kind == "preprocess" else "training_dataset"), "RESUME": ("checkpoint", "resume_checkpoint"),
        "CKPT_PATH": ("checkpoint", "evaluated_model"),
    }
    for key, (asset_kind, role) in mappings.items():
        if env.get(key):
            env[key] = pin(str(env[key]), asset_kind, role)
    arg_mappings = {
        "model_path": mappings["MODEL_PATH"], "model-path": mappings["MODEL_PATH"],
        "model_fir": ("model", "evaluated_model"), "model_sec": ("model", "baseline_model"),
        "model-fir": ("model", "evaluated_model"), "model-sec": ("model", "baseline_model"),
        "dataset_dir": ("dataset", "training_dataset"), "dataset-dir": ("dataset", "training_dataset"),
        "resume-from-checkpoint": ("checkpoint", "resume_checkpoint"),
        "resume_from_checkpoint": ("checkpoint", "resume_checkpoint"), "CKPT_PATH": ("checkpoint", "evaluated_model"),
        "ckpt-path": ("checkpoint", "evaluated_model"),
        'input_folder': ('dataset', 'source_dataset'), 'input-folder': ('dataset', 'source_dataset'),
        'input_file': ('dataset', 'source_dataset'), 'input-file': ('dataset', 'source_dataset'),
    }
    for key, (asset_kind, role) in arg_mappings.items():
        if args.get(key):
            args[key] = pin(str(args[key]), asset_kind, role)
    if kind == 'preprocess' and run_kwargs.get('experiment_resume_output'):
        pin(str(run_kwargs['experiment_resume_output']), 'dataset', 'preprocessing_resume')
    script_name = str(run_kwargs.get('script_name') or '')
    if script_name.startswith('generic_') and kind == 'evaluate':
        from .experiment_evaluation import DATASETS
        selected = args.get('eval_datasets') or args.get('eval-datasets') or ''
        names = selected if isinstance(selected, list) else str(selected).split(',')
        for name in dict.fromkeys(str(item).strip().lower() for item in names):
            if name not in DATASETS:
                raise ValueError(f'Unknown evaluation dataset: {name}')
            recorded = replay_source('dataset', 'evaluation_dataset', name) if replay_assets is not None else None
            source = recorded['location']['path'] if recorded else f'/home/workspace/evaluation/{name}'
            pin(source, 'dataset', 'evaluation_dataset')
            registration = captures[(source, 'dataset')][1]
            if recorded and (registration['id'] != recorded['id'] or registration['fingerprint'] != recorded['fingerprint']):
                raise ValueError('Draft evaluation dataset snapshot is unavailable or changed')
            registrations[f"{registration['id']}:evaluation_dataset"]['metadata']['datasetName'] = name
    if kind == 'evaluate' and script_name in {'generic_ckpt_eval', 'ckpt_eval'}:
        if replay_assets is not None:
            recorded = replay_source('model', 'base_model')
            source = recorded['location']['path']
            env['MEDFLOW_EVAL_BASE_MODEL'] = pin(source, 'model', 'base_model')
            registration = captures[(source, 'model')][1]
            if registration['id'] != recorded['id'] or registration['fingerprint'] != recorded['fingerprint']:
                raise ValueError('Draft base model snapshot is unavailable or changed')
        else:
            checkpoint = args.get('CKPT_PATH') or args.get('ckpt-path') or env.get('CKPT_PATH')
            config = container_json(container, 'experiment_process.py', {'path': f'{checkpoint}/adapter_config.json'}, 'read-json')
            env['MEDFLOW_EVAL_BASE_MODEL'] = pin(str(config.get('base_model_name_or_path') or ''), 'model', 'base_model')
    overrides = {}
    for item in run_kwargs.get('positional_args') or []:
        key, separator, value = str(item).partition('=')
        if separator:
            if key in overrides and key in {'trainer.resume_from_path', 'trainer.resume_mode'}:
                raise ValueError('Duplicate GRPO resume override')
            overrides[key] = value
    resume_path = overrides.get('trainer.resume_from_path')
    resume_mode = overrides.get('trainer.resume_mode')
    if resume_mode == 'resume_path' and (not resume_path or resume_path == 'null'):
        raise ValueError('GRPO resume_path requires an explicit checkpoint directory')
    if resume_path and resume_path != 'null' and resume_mode != 'resume_path':
        raise ValueError('GRPO checkpoint is only used with trainer.resume_mode=resume_path')
    positional = []
    for argument in run_kwargs.get("positional_args") or []:
        key, separator, value = str(argument).partition("=")
        if separator and key == "actor_rollout_ref.model.path":
            argument = f"{key}={pin(value, 'model', 'base_model')}"
        elif separator and key == 'trainer.resume_from_path' and value != 'null':
            # VERL derives the resumed step from the directory name. Preserve it
            # as an execution alias while the underlying content stays pinned.
            checkpoint_name = resume_checkpoint_name or PurePosixPath(value).name
            if not re.fullmatch(r'global_step_[0-9]+', str(checkpoint_name)):
                raise ValueError('GRPO resume requires a global_step_<number> checkpoint directory')
            argument = f"{key}={pin(value, 'checkpoint', 'resume_checkpoint')}"
            registration = captures[(value, 'checkpoint')][1]
            registrations[f"{registration['id']}:resume_checkpoint"]['metadata']['resumeDirectoryName'] = checkpoint_name
        elif separator and key in {"data.train_files", "data.val_files"}:
            role = "training_dataset" if key == "data.train_files" else "validation_dataset"
            if value.startswith("["):
                values = ast.literal_eval(value)
                if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                    raise ValueError("Unsupported GRPO file list")
                argument = f"{key}={json.dumps([pin(item, 'dataset', role) for item in values])}"
            else:
                argument = f"{key}={pin(value, 'dataset', role)}"
        positional.append(argument)
    run_kwargs.update(env_vars=env, script_args=args, positional_args=positional)
    return list(registrations.values())
