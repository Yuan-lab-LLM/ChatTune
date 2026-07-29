# -*- coding: utf-8 -*-
"""
调用本地脚本的工具函数，支持通过名称、别名查找并运行脚本。
相对于另一版本，补充修改参数的能力
加上改docker
加上推理
加上监控
"""

import subprocess
import sys
import json
import os
import threading
import time
import shlex
import re
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from ._dataset_candidates import (
    dataset_find_exclusion_clause,
    dataset_names_from_dataset_info,
    is_dataset_candidate_filename,
)
from agentscope.tool import ToolResponse

logger = logging.getLogger(__name__)
from ._config_defaults import get_default_docker_container
from agentscope.message import TextBlock
from ._train_monitor_helpers import (
    build_background_shell_command,
    infer_train_type_from_name,
    public_train_type,
    public_train_type_text,
    parse_pid_from_output,
    require_assigned_gpu_env,
)
from ._template_policy import (
    DEFAULT_TEMPLATE,
    MODEL_HINT_KEYS,
    TEMPLATE_PARAM_KEYS,
    first_nonempty,
    non_qwen3_template_required_issue,
    normalize_template,
    template_validation_issue,
    text_mentions_non_qwen3_model,
)


_RUNTIME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime"),
)
_TRAIN_PID_REGISTRY = os.path.join(_RUNTIME_DIR, "train_pid_registry.jsonl")
_BACKGROUND_TASK_REGISTRY = os.path.join(_RUNTIME_DIR, "background_task_registry.jsonl")
DEFAULT_DOCKER_CONTAINER = get_default_docker_container()
MULTINODE_DOCKER_CONTAINER = os.getenv("MULTINODE_DOCKER_CONTAINER", "")
GRPO_DOCKER_CONTAINER = os.getenv("MEDFLOW_GRPO_DOCKER_CONTAINER", "")
MULTINODE_SFT_SCRIPT = "train_multinode_sft_pipeline"
MULTINODE_DPO_SCRIPT = "train_multinode_dpo_pipeline"
LLAMAFACTORY_BINARY_WORKDIR = "/usr/local/insinfersystem"
LLAMAFACTORY_SOURCE_WORKDIR = "/home/workspace/llamafactory"
MULTINODE_BINARY_FIXED_CLI_ARGS = {
    "llamafactory-dir": LLAMAFACTORY_BINARY_WORKDIR,
    "deepspeed-bin": "deepspeed",
    "deepspeed-config": f"{LLAMAFACTORY_BINARY_WORKDIR}/examples/deepspeed/ds_z3_config.json",
    "train-entry": f"{LLAMAFACTORY_BINARY_WORKDIR}/train",
    "no-python": True,
}
MULTINODE_BINARY_PATHS = {
    MULTINODE_SFT_SCRIPT: f"{LLAMAFACTORY_BINARY_WORKDIR}/{MULTINODE_SFT_SCRIPT}",
    MULTINODE_DPO_SCRIPT: f"{LLAMAFACTORY_BINARY_WORKDIR}/{MULTINODE_DPO_SCRIPT}",
}
MULTINODE_SOURCE_PATHS = {
    MULTINODE_SFT_SCRIPT: f"{LLAMAFACTORY_SOURCE_WORKDIR}/{MULTINODE_SFT_SCRIPT}.py",
    MULTINODE_DPO_SCRIPT: f"{LLAMAFACTORY_SOURCE_WORKDIR}/{MULTINODE_DPO_SCRIPT}.py",
}
GRPO_BINARY_WORKDIR = "/usr/local/insinfersystem"
GRPO_BINARY_WRAPPER = f"{GRPO_BINARY_WORKDIR}/run_grpo_qwen3_8b_260417"
GRPO_SOURCE_WORKDIR = "/home/workspace/verl"
GRPO_SOURCE_SCRIPT = "examples/grpo_trainer/run_qwen3-8b_260417.sh"
GRPO_SOURCE_SCRIPT_ABS = f"{GRPO_SOURCE_WORKDIR}/{GRPO_SOURCE_SCRIPT}"
GRPO_LOG_DIR = "/home/workspace/log/grpo_train"
RESOURCE_LAUNCHER_ONLY_PARAMS = {
    "gpu-count", "node-count", "gpus-per-node", "resource-pool-id", "resource-group-id",
}
MANUAL_GPU_ENV_KEYS = {"MEDFLOW_ASSIGNED_GPUS", "CUDA_VISIBLE_DEVICES", "LOCALHOST_ID"}
_ACTIVE_RESERVATION_HEARTBEATS: set[str] = set()
_ACTIVE_RESERVATION_HEARTBEATS_LOCK = threading.Lock()

MULTINODE_SFT_CLI_PARAMS = [
    "base-dataset-path", "base-model-path", "batch-train-path",
    "dataset-dir", "dataset-date", "dataset", "dataset-name",
    "model-path", "output-dir", "data-log-file",
    "batch-size", "acc", "lr", "learning-rate", "template", "resume-from-checkpoint",
    "llamafactory-dir", "deepspeed-bin", "deepspeed-config", "train-entry", "no-python",
    "master-addr", "master-port", "hostfile", "allocation-file", "ds35-alias", "ds36-alias",
    "ds35-gpu", "ds36-gpu", "include-resources", "nccl-ifname",
    "pdsh-rcmd-type", "nccl-debug", "skip-preflight",
    "log-dir", "log-file",
    "cutoff-len", "preprocessing-num-workers", "max-grad-norm",
    "lr-scheduler-type", "logging-steps", "save-steps", "save-strategy",
    "num-train-epochs", "warmup-ratio", "bf16", "val-size",
    "per-device-eval-batch-size", "eval-strategy", "eval-steps",
    "use-fast-tokenizer", "disable-gradient-checkpointing", "print-param-status",
    "flash-attn", "torch-empty-cache-steps", "use-liger-kernel", "use-adam-mini",
    "dataloader-num-workers", "group-by-length", "report-to", "use-unsloth-gc",
    "max-steps", "skip-data-analysis", "skip-merge",
    "replace-tokenizer-before-merge", "export-bin", "merge-cuda-visible-devices",
    "node-count", "gpus-per-node", "resource-pool-id", "resource-group-id",
]

MULTINODE_DPO_CLI_PARAMS = [
    "llamafactory-dir", "deepspeed-bin", "deepspeed-config", "train-entry", "no-python",
    "master-addr", "master-port", "hostfile", "allocation-file", "ds35-alias", "ds36-alias",
    "ds35-gpu", "ds36-gpu", "include-resources",
    "pdsh-rcmd-type", "nccl-ifname", "nccl-debug",
    "model-path", "dataset-dir", "dataset", "dataset-name", "template", "output-dir",
    "batch-size", "acc", "lr", "learning-rate", "max-steps",
    "run-id", "log-dir", "log-file",
    "skip-merge", "export-dir", "llamafactory-cli", "export-size",
    "export-device", "export-legacy-format",
    "replace-tokenizer-before-merge", "replace-tokenizer-after-merge",
    "skip-preflight", "node-count", "gpus-per-node", "resource-pool-id", "resource-group-id",
]

MULTINODE_BOOLEAN_CLI_PARAMS = {
    "no-python",
    "skip-preflight",
    "skip-data-analysis",
    "skip-merge",
    "replace-tokenizer-before-merge",
    "replace-tokenizer-after-merge",
}

MULTINODE_PARAM_MAPPING = {
    "container": "container",
    "docker": "container",
    "容器": "container",
    "docker_container": "container",
    "container_name": "container",
    "model_path": "model-path",
    "模型路径": "model-path",
    "模型位置": "model-path",
    "dataset_dir": "dataset-dir",
    "data_dir": "dataset-dir",
    "sft_data_dir": "dataset-dir",
    "sft_dataset_dir": "dataset-dir",
    "数据集路径": "dataset-dir",
    "数据集目录": "dataset-dir",
    "数据目录": "dataset-dir",
    "dataset_date": "dataset-date",
    "data_identifier": "dataset-date",
    "data_id": "dataset-date",
    "dataset_id": "dataset-date",
    "data_date": "dataset-date",
    "数据集日期": "dataset-date",
    "数据日期": "dataset-date",
    "dataset": "dataset",
    "dataset_name": "dataset",
    "dataset-name": "dataset",
    "数据集": "dataset",
    "数据集名称": "dataset",
    "数据集名": "dataset",
    "mbs": "batch-size",
    "MBS": "batch-size",
    "batch_size": "batch-size",
    "批量大小": "batch-size",
    "批次大小": "batch-size",
    "批大小": "batch-size",
    "acc": "acc",
    "ACC": "acc",
    "lr": "lr",
    "LR": "lr",
    "learning_rate": "learning-rate",
    "学习率": "learning-rate",
    "学习速率": "learning-rate",
    "tem": "template",
    "TEM": "template",
    "template": "template",
    "output_dir": "output-dir",
    "输出目录": "output-dir",
    "输出路径": "output-dir",
    "保存模型路径": "output-dir",
    "保存路径": "output-dir",
    "log_dir": "log-dir",
    "日志目录": "log-dir",
    "log_file": "log-file",
    "日志文件": "log-file",
    "master_addr": "master-addr",
    "主节点地址": "master-addr",
    "master_port": "master-port",
    "主节点端口": "master-port",
    "allocation_file": "allocation-file",
    "资源分配文件": "allocation-file",
    "多机配置文件": "hostfile",
    "配置文件": "hostfile",
    "多机文件": "hostfile",
    "include_resources": "include-resources",
    "include资源": "include-resources",
    "nccl_ifname": "nccl-ifname",
    "pdsh_rcmd_type": "pdsh-rcmd-type",
    "nccl_debug": "nccl-debug",
    "ds35_alias": "ds35-alias",
    "ds36_alias": "ds36-alias",
    "ds35_gpu": "ds35-gpu",
    "ds36_gpu": "ds36-gpu",
    "主节点ip": "master-addr",
    "主服务器ip": "master-addr",
    "主节点IP": "master-addr",
    "主服务器IP": "master-addr",
    "主服务器地址": "master-addr",
    "主端口": "master-port",
    "主节点用户名": "ds35-alias",
    "主服务器用户名称": "ds35-alias",
    "主服务器用户名": "ds35-alias",
    "主节点用户名称": "ds35-alias",
    "副节点用户名": "ds36-alias",
    "副服务器用户名称": "ds36-alias",
    "副服务器用户名": "ds36-alias",
    "副节点用户名称": "ds36-alias",
    "工作节点用户名": "ds36-alias",
    "工作服务器用户名称": "ds36-alias",
    "工作服务器用户名": "ds36-alias",
    "工作节点用户名称": "ds36-alias",
    "主节点卡号": "ds35-gpu",
    "主服务器卡号": "ds35-gpu",
    "副节点卡号": "ds36-gpu",
    "副服务器卡号": "ds36-gpu",
    "工作节点卡号": "ds36-gpu",
    "工作服务器卡号": "ds36-gpu",
    "NCCL网络接口": "nccl-ifname",
    "nccl网络接口": "nccl-ifname",
    "NCCL网卡": "nccl-ifname",
    "nccl网卡": "nccl-ifname",
    "通信网卡": "nccl-ifname",
    "网卡名称": "nccl-ifname",
    "最大步数": "max-steps",
    "不合并权重": "skip-merge",
    "不合并": "skip-merge",
    "不merge": "skip-merge",
    "max_steps": "max-steps",
    "run_id": "run-id",
    "export_dir": "export-dir",
    "export_size": "export-size",
    "export_device": "export-device",
    "export_legacy_format": "export-legacy-format",
    "llamafactory_dir": "llamafactory-dir",
    "deepspeed_bin": "deepspeed-bin",
    "deepspeed_config": "deepspeed-config",
    "train_entry": "train-entry",
    "no_python": "no-python",
    "skip_preflight": "skip-preflight",
    "skip_merge": "skip-merge",
    "skip_data_analysis": "skip-data-analysis",
    "replace_tokenizer_before_merge": "replace-tokenizer-before-merge",
    "replace_tokenizer_after_merge": "replace-tokenizer-after-merge",
    "data_log_file": "data-log-file",
    "base_dataset_path": "base-dataset-path",
    "base_model_path": "base-model-path",
    "batch_train_path": "batch-train-path",
    "resume_from_checkpoint": "resume-from-checkpoint",
    "cutoff_len": "cutoff-len",
    "preprocessing_num_workers": "preprocessing-num-workers",
    "max_grad_norm": "max-grad-norm",
    "lr_scheduler_type": "lr-scheduler-type",
    "logging_steps": "logging-steps",
    "save_steps": "save-steps",
    "save_strategy": "save-strategy",
    "num_train_epochs": "num-train-epochs",
    "warmup_ratio": "warmup-ratio",
    "val_size": "val-size",
    "per_device_eval_batch_size": "per-device-eval-batch-size",
    "eval_strategy": "eval-strategy",
    "eval_steps": "eval-steps",
    "use_fast_tokenizer": "use-fast-tokenizer",
    "disable_gradient_checkpointing": "disable-gradient-checkpointing",
    "print_param_status": "print-param-status",
    "flash_attn": "flash-attn",
    "torch_empty_cache_steps": "torch-empty-cache-steps",
    "use_liger_kernel": "use-liger-kernel",
    "use_adam_mini": "use-adam-mini",
    "dataloader_num_workers": "dataloader-num-workers",
    "group_by_length": "group-by-length",
    "report_to": "report-to",
    "use_unsloth_gc": "use-unsloth-gc",
    "export_bin": "export-bin",
    "merge_cuda_visible_devices": "merge-cuda-visible-devices",
    "llamafactory_cli": "llamafactory-cli",
}
def _docker_test_path(container: str, path: str, test_flag: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "exec", container, "sh", "-c", f"test {test_flag} {shlex.quote(path)}"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return False, str(exc)
    detail = str(result.stderr or result.stdout or "").strip()
    return result.returncode == 0, detail


def _select_python_executable_in_container(container: str) -> Tuple[Optional[str], Optional[str]]:
    py_candidates = ["/opt/conda/bin/python", "python3", "python"]
    details = []
    for py in py_candidates:
        try:
            check_py = ["docker", "exec", container, "sh", "-c", f"command -v {py} >/dev/null 2>&1"]
            result = subprocess.run(check_py, capture_output=True, text=True)
            if result.returncode == 0:
                return py, None
            detail = str(result.stderr or result.stdout or "").strip()
            if detail:
                details.append(f"{py}: {detail}")
        except Exception as exc:
            details.append(f"{py}: {exc}")
    return None, "; ".join(details) if details else None


def _merge_fixed_cli_args(script_args: Dict[str, Any], fixed_args: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(fixed_args)
    for key, value in (script_args or {}).items():
        if key not in merged:
            merged[key] = value
    return merged


def _select_multinode_docker_entrypoint(
    container: str,
    script_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    binary_path = MULTINODE_BINARY_PATHS.get(script_name)
    source_abs = MULTINODE_SOURCE_PATHS.get(script_name)

    binary_detail = ""
    if binary_path:
        binary_exists, binary_detail = _docker_test_path(container, binary_path, "-x")
        if binary_exists:
            return (
                {
                    "mode": "binary",
                    "script_path": binary_path,
                    "docker_working_dir": LLAMAFACTORY_BINARY_WORKDIR,
                    "docker_executable": binary_path,
                    "fixed_cli_args": MULTINODE_BINARY_FIXED_CLI_ARGS,
                },
                None,
            )

    source_detail = ""
    if source_abs:
        source_exists, source_detail = _docker_test_path(container, source_abs, "-f")
        if source_exists:
            python_executable, python_detail = _select_python_executable_in_container(container)
            if not python_executable:
                detail = f"\npython check: {python_detail}" if python_detail else ""
                return (
                    None,
                    (
                        "ERROR: multi-node training source fallback exists, but no Python executable "
                        "was found in the container.\n"
                        f"source path: {source_abs}"
                        f"{detail}"
                    ),
                )
            return (
                {
                    "mode": "source",
                    "script_path": os.path.basename(source_abs),
                    "docker_working_dir": LLAMAFACTORY_SOURCE_WORKDIR,
                    "docker_executable": python_executable,
                    "fixed_cli_args": {},
                },
                None,
            )

    details = []
    if binary_detail:
        details.append(f"binary check: {binary_detail}")
    if source_detail:
        details.append(f"source check: {source_detail}")
    suffix = "\n" + "\n".join(details) if details else ""
    return (
        None,
        (
            "ERROR: multi-node training entrypoint was not found in the container.\n"
            f"binary path: {binary_path}\n"
            f"source fallback: {source_abs}"
            f"{suffix}"
        ),
    )

def _select_grpo_docker_entrypoint(container: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    binary_exists, binary_detail = _docker_test_path(container, GRPO_BINARY_WRAPPER, "-x")
    if binary_exists:
        return (
            {
                "mode": "binary",
                "script_path": GRPO_BINARY_WRAPPER,
                "docker_working_dir": GRPO_BINARY_WORKDIR,
                "docker_executable": GRPO_BINARY_WRAPPER,
                "wandb_dir": GRPO_BINARY_WORKDIR,
            },
            None,
        )

    source_exists, source_detail = _docker_test_path(container, GRPO_SOURCE_SCRIPT_ABS, "-f")
    if source_exists:
        return (
            {
                "mode": "source",
                "script_path": GRPO_SOURCE_SCRIPT,
                "docker_working_dir": GRPO_SOURCE_WORKDIR,
                "docker_executable": "bash",
                "wandb_dir": GRPO_SOURCE_WORKDIR,
            },
            None,
        )

    details = []
    if binary_detail:
        details.append(f"binary check: {binary_detail}")
    if source_detail:
        details.append(f"source check: {source_detail}")
    suffix = "\n" + "\n".join(details) if details else ""
    return (
        None,
        (
            "ERROR: GRPO training entrypoint was not found in the container.\n"
            f"binary path: {GRPO_BINARY_WRAPPER}\n"
            f"source fallback: {GRPO_SOURCE_SCRIPT_ABS}"
            f"{suffix}"
        ),
    )

def _grpo_shell_prefix(log_file: str, wandb_dir: str) -> str:
    wandb_root = f"{wandb_dir.rstrip('/')}/wandb"
    return (
        f"mkdir -p {shlex.quote(GRPO_LOG_DIR)} {shlex.quote(wandb_root)}"
        f" && echo MEDFLOW_TRAINING_COMMAND_READY >> {shlex.quote(log_file)}"
    )


def _normalize_train_type(train_type: Optional[str]) -> Optional[str]:
    aliases = {
        "dpo": "enhanced",
        "enhance": "enhanced",
        "enhanced": "enhanced",
        "增强训练": "enhanced",
        "schedule": "scheduled",
        "daily": "scheduled",
        "full": "full",
        "full_train": "full",
        "full_sft": "full",
        "fullsft": "full",
        "全参sft": "full",
        "lora": "lora",
        "lora_train": "lora",
        "lora_sft": "lora",
        "lorasft": "lora",
        "sft": "lora",
        "grpo": "grpo",
        "multinode_lora_sft": "lora",
        "multinodelorasft": "lora",
        "双机lorasft": "lora",
        "multinode_enhanced": "enhanced",
        "multinodeenhanced": "enhanced",
        "双机增强训练": "enhanced",
    }
    value = (train_type or "").strip().lower()
    key = re.sub(r"\s+", "", value)
    return aliases.get(key, aliases.get(value, value)) or None

def _train_type_text(train_type: Optional[str]) -> Optional[str]:
    return {
        "lora": "LoRA SFT",
        "full": "全参 SFT",
        "enhanced": "增强训练",
        "scheduled": "定时训练",
        "grpo": "GRPO",
    }.get(_normalize_train_type(train_type) or "")


def _is_multinode_train_script(script_name: Optional[str]) -> bool:
    return str(script_name or "").strip() in {MULTINODE_SFT_SCRIPT, MULTINODE_DPO_SCRIPT}


def _resource_request_config(script_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config = script_info.get("resource_request")
    return config if isinstance(config, dict) and config.get("enabled") else None


def _resource_task_metadata(script_info: Dict[str, Any]) -> Dict[str, str]:
    script_name = str(script_info.get("name") or "").strip()
    train_type = infer_train_type_from_name(script_name)
    config = _resource_request_config(script_info) or {}
    metadata = {"taskCategory": str(config.get("task_category") or "training")}
    if config.get("task_type"):
        metadata["taskType"] = str(config["task_type"])
    if config.get("task_type_text"):
        metadata["taskTypeText"] = str(config["task_type_text"])
    if metadata.get("taskType") and metadata.get("taskTypeText"):
        return metadata
    if train_type:
        metadata["taskType"] = str(public_train_type(train_type, None, script_name) or train_type)
        metadata["taskTypeText"] = str(public_train_type_text(train_type, None, script_name) or train_type)
    elif script_name:
        metadata["taskType"] = script_name
        metadata["taskTypeText"] = script_name
    return metadata

def _default_resource_request(
    script_info: Dict[str, Any],
    launcher_options: Dict[str, Any],
    env_vars: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    config = _resource_request_config(script_info)
    if not config:
        raise ValueError("当前脚本未启用资源池自动预约")
    env_key = str(config.get("gpu_count_env") or "").strip()
    default_gpu_count = (
        str((env_vars or {}).get(env_key) or "").strip()
        or os.getenv(env_key, "").strip()
        if env_key
        else ""
    ) or config.get("gpus_per_node", 1)
    is_multinode = bool(config.get("multinode"))
    try:
        node_count = int(
            (launcher_options.get("node-count") if is_multinode else "")
            or ((env_vars or {}).get("MEDFLOW_MULTINODE_NODE_COUNT", "") if is_multinode else "")
            or (os.getenv("MEDFLOW_MULTINODE_NODE_COUNT", "") if is_multinode else "")
            or config.get("node_count", 1)
        )
        gpus_per_node = int(
            launcher_options.get("gpu-count")
            or (launcher_options.get("gpus-per-node") if is_multinode else "")
            or ((env_vars or {}).get("MEDFLOW_MULTINODE_GPUS_PER_NODE", "") if is_multinode else "")
            or (os.getenv("MEDFLOW_MULTINODE_GPUS_PER_NODE", "") if is_multinode else "")
            or default_gpu_count
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("资源预约节点数和 GPU 数必须是整数") from exc
    if node_count < 1 or gpus_per_node < 1:
        raise ValueError("资源预约节点数和 GPU 数必须是正整数")
    return node_count, gpus_per_node


def _studio_runtime_headers() -> Dict[str, str]:
    token = (
        os.getenv("MEDFLOW_STUDIO_NODE_TOKEN")
        or os.getenv("MEDFLOW_STUDIO_RUNTIME_TOKEN")
        or os.getenv("AGENTSCOPE_STUDIO_RUNTIME_TOKEN")
        or ""
    ).strip()
    return {"X-MedFlow-Runtime-Token": token} if token else {}


def _trpc_result_data(payload: Dict[str, Any]) -> Any:
    current: Any = payload
    for key in ("result", "data", "json"):
        if isinstance(current, dict) and key in current:
            current = current[key]
    return current


def _trpc_error_message(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        error_json = error.get("json")
        if isinstance(error_json, dict) and error_json.get("message"):
            return str(error_json["message"])
        if error.get("message"):
            return str(error["message"])
    return str(payload.get("message") or "")


def _configured_resource_id(value: Any) -> str:
    resource_id = str(value or "").strip()
    if resource_id.lower().startswith("replace-with-"):
        return ""
    return resource_id


def _runtime_training_resource_request(procedure: str, data: Dict[str, Any]) -> Any:
    studio_url = os.getenv("STUDIO_URL", "").strip().rstrip("/")
    if not studio_url:
        raise ValueError("GPU 资源池自动分配需要配置 STUDIO_URL")
    timeout_seconds = max(
        1,
        int(os.getenv("MEDFLOW_STUDIO_RESOURCE_REQUEST_TIMEOUT_SECONDS", "90")),
    )
    response = requests.post(
        f"{studio_url}/trpc/{procedure}",
        json=data,
        headers=_studio_runtime_headers(),
        timeout=timeout_seconds,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not response.ok:
        error_message = _trpc_error_message(payload)
        if error_message:
            raise ValueError(error_message)
    response.raise_for_status()
    result = _trpc_result_data(payload)
    if isinstance(result, dict) and result.get("success") is False:
        raise ValueError(str(result.get("message") or "Studio 资源分配失败"))
    return result.get("data") if isinstance(result, dict) and "data" in result else result


def _local_gpu_snapshot_for_runtime_request() -> Optional[Dict[str, Any]]:
    try:
        from resource_api import _gpus_snapshot  # type: ignore
    except Exception:
        logger.debug("Runtime GPU snapshot helper is unavailable", exc_info=True)
        return None
    try:
        snapshot = _gpus_snapshot()
    except Exception:
        logger.info("Runtime GPU snapshot is not ready; Studio allocation will fail fast if it needs it", exc_info=True)
        return None
    gpus = snapshot.get("gpus") if isinstance(snapshot, dict) else None
    if not isinstance(gpus, list):
        return None
    return {
        "gpus": gpus,
        "collectedAt": snapshot.get("collectedAt"),
        "ageSeconds": snapshot.get("ageSeconds"),
        "maxAgeSeconds": snapshot.get("maxAgeSeconds"),
    }


def _build_resource_request(
    script_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    env_vars: Dict[str, str],
    launcher_options: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, int, bool]:
    is_multinode = _is_multinode_train_script(script_info.get("name"))
    request_group_id = _configured_resource_id(
        env_vars.get("MEDFLOW_RESOURCE_GROUP_ID")
        or launcher_options.get("resource-group-id")
        or ""
    )
    process_group_id = _configured_resource_id(os.getenv("MEDFLOW_RESOURCE_GROUP_ID", ""))
    group_id = request_group_id or process_group_id
    pool_id = _configured_resource_id(
        launcher_options.get("resource-pool-id")
        or env_vars.get("MEDFLOW_TRAINING_POOL_ID")
        or (os.getenv("MEDFLOW_TRAINING_POOL_ID", "") if not request_group_id else "")
    )
    runtime_node_id = _configured_resource_id(
        env_vars.get("MEDFLOW_RESOURCE_NODE_ID")
        or os.getenv("MEDFLOW_RESOURCE_NODE_ID", "")
    )
    if not group_id or not runtime_node_id:
        raise ValueError(
            "GPU 资源池自动分配需要配置 MEDFLOW_RESOURCE_GROUP_ID 和 MEDFLOW_RESOURCE_NODE_ID；"
            "用户组配置多个资源池时还需要 MEDFLOW_TRAINING_POOL_ID"
        )
    if not request_group_id and process_group_id:
        groups = _runtime_training_resource_request(
            "listRuntimeGroupsForRuntime",
            {"runtimeNodeId": runtime_node_id},
        )
        if isinstance(groups, list) and len(groups) > 1:
            group_names = ", ".join(
                str(item.get("groupName") or item.get("groupId") or "")
                for item in groups
                if isinstance(item, dict)
            )
            raise ValueError(
                "当前 Runtime 绑定了多个用户组，不能回退使用进程级 MEDFLOW_RESOURCE_GROUP_ID；"
                f"请通过请求上下文传入 resource_group_id。绑定用户组：{group_names}"
            )
    node_count, gpus_per_node = _default_resource_request(script_info, launcher_options, env_vars)
    request_data = {
        "groupId": group_id,
        "runtimeNodeId": runtime_node_id,
        "nodeCount": node_count,
        "gpusPerNode": gpus_per_node,
    }
    request_data.update(_resource_task_metadata(script_info))
    runtime_gpu_snapshot = _local_gpu_snapshot_for_runtime_request()
    if runtime_gpu_snapshot:
        request_data["runtimeGpuSnapshot"] = runtime_gpu_snapshot
    if pool_id:
        request_data["poolId"] = pool_id
    return request_data, node_count, gpus_per_node, is_multinode


def _preflight_resource_allocation(
    script_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    env_vars: Dict[str, str],
    launcher_options: Dict[str, Any],
) -> None:
    _resume_resource_reservation_heartbeats()
    is_multinode = _is_multinode_train_script(script_info.get("name"))
    if is_multinode and str(cli_args.get("allocation-file") or "").strip():
        return
    request_data, node_count, gpus_per_node, _ = _build_resource_request(
        script_info,
        cli_args,
        env_vars,
        launcher_options,
    )
    logger.info(
        "Preflighting Studio training resources: script=%s groupId=%s poolId=%s "
        "runtimeNodeId=%s nodeCount=%s gpusPerNode=%s runtimeGpuSnapshot=%s",
        script_info.get("name"),
        request_data.get("groupId"),
        request_data.get("poolId") or "",
        request_data.get("runtimeNodeId"),
        node_count,
        gpus_per_node,
        "yes" if request_data.get("runtimeGpuSnapshot") else "no",
    )
    _runtime_training_resource_request(
        "preflightTrainingResourcesForRuntime",
        request_data,
    )


def _prepare_resource_allocation(
    script_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    env_vars: Dict[str, str],
    launcher_options: Dict[str, Any],
) -> Optional[str]:
    _resume_resource_reservation_heartbeats()
    is_multinode = _is_multinode_train_script(script_info.get("name"))
    if is_multinode and str(cli_args.get("allocation-file") or "").strip():
        return None
    request_data, node_count, gpus_per_node, _ = _build_resource_request(
        script_info,
        cli_args,
        env_vars,
        launcher_options,
    )
    logger.info(
        "Requesting Studio training resources: script=%s groupId=%s poolId=%s "
        "runtimeNodeId=%s nodeCount=%s gpusPerNode=%s runtimeGpuSnapshot=%s",
        script_info.get("name"),
        request_data.get("groupId"),
        request_data.get("poolId") or "",
        request_data.get("runtimeNodeId"),
        node_count,
        gpus_per_node,
        "yes" if request_data.get("runtimeGpuSnapshot") else "no",
    )
    allocation = _runtime_training_resource_request(
        "reserveTrainingResourcesForRuntime",
        request_data,
    )
    if not isinstance(allocation, dict):
        raise ValueError("Studio 返回了无效的 GPU 资源分配结果")
    reservation_id = str(allocation.get("reservationId") or "").strip() or None
    master = next(
        (node for node in allocation.get("nodes", []) if isinstance(node, dict) and node.get("isMaster")),
        None,
    )
    if not master or not allocation.get("reservationId"):
        _release_resource_allocation(reservation_id, env_vars)
        raise ValueError("Studio 返回的资源分配缺少主节点或 reservationId")
    env_vars["MEDFLOW_ASSIGNED_GPUS"] = ",".join(str(index) for index in master.get("gpuIndexes", []))
    env_vars["CUDA_VISIBLE_DEVICES"] = env_vars["MEDFLOW_ASSIGNED_GPUS"]
    env_vars["LOCALHOST_ID"] = env_vars["MEDFLOW_ASSIGNED_GPUS"]
    env_vars["MEDFLOW_TRAINING_RESERVATION_ID"] = str(allocation["reservationId"])
    logger.info(
        "Studio training resources reserved: reservationId=%s assignedGpus=%s "
        "nodeCount=%s gpusPerNode=%s",
        allocation["reservationId"],
        env_vars["MEDFLOW_ASSIGNED_GPUS"],
        node_count,
        gpus_per_node,
    )
    if is_multinode:
        if not allocation.get("allocationFile"):
            _release_resource_allocation(reservation_id, env_vars)
            raise ValueError("Studio 返回的多机资源分配缺少 allocationFile")
        cli_args["allocation-file"] = str(allocation["allocationFile"])
    return reservation_id


def _release_resource_allocation(reservation_id: Optional[str], env_vars: Dict[str, str]) -> bool:
    if not reservation_id:
        return False
    try:
        _runtime_training_resource_request(
            "releaseTrainingResourcesForRuntime",
            {
                "reservationId": reservation_id,
                "runtimeNodeId": str(
                    env_vars.get("MEDFLOW_RESOURCE_NODE_ID")
                    or os.getenv("MEDFLOW_RESOURCE_NODE_ID", "")
                ).strip(),
            },
        )
        return True
    except Exception:
        logger.exception("Failed to release GPU resource reservation %s", reservation_id)
        return False


def _assigned_gpu_indexes(env_vars: Optional[Dict[str, str]]) -> List[int]:
    raw = str(
        (env_vars or {}).get("MEDFLOW_ASSIGNED_GPUS")
        or (env_vars or {}).get("CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()
    result: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError as exc:
            raise ValueError(f"分配的 GPU 卡号无效：{raw}") from exc
    return sorted(set(result))


def _validate_assigned_gpus_available(docker_container: Optional[str], env_vars: Optional[Dict[str, str]]) -> None:
    assigned = _assigned_gpu_indexes(env_vars)
    if not assigned or not docker_container:
        return
    try:
        timeout_seconds = max(1, int(os.getenv("MEDFLOW_GPU_PREFLIGHT_TIMEOUT_SECONDS", "5")))
    except ValueError:
        timeout_seconds = 5
    try:
        busy_threshold_mb = max(0, int(os.getenv("MEDFLOW_GPU_PREFLIGHT_BUSY_MEMORY_MB", "200")))
    except ValueError:
        busy_threshold_mb = 200
    query = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    max_attempts = 3
    retry_interval_seconds = 1
    result = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                ["docker", "exec", docker_container, "sh", "-c", query],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            break
        except subprocess.TimeoutExpired:
            logger.warning(
                "Assigned GPU preflight timed out: container=%s attempt=%s/%s timeoutSeconds=%s",
                docker_container,
                attempt,
                max_attempts,
                timeout_seconds,
            )
            if attempt == max_attempts:
                raise ValueError(
                    "启动前 GPU 快速校验失败："
                    f"容器 {docker_container} 内 nvidia-smi 连续 {max_attempts} 次超时"
                    f"（单次 {timeout_seconds} 秒）"
                ) from None
            time.sleep(retry_interval_seconds)
    assert result is not None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ValueError(f"启动前 GPU 快速校验失败：{detail or 'nvidia-smi 执行失败'}")
    usage: Dict[int, int] = {}
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            usage[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    missing = [index for index in assigned if index not in usage]
    if missing:
        raise ValueError(f"启动前 GPU 快速校验失败：分配的 GPU 不存在或容器不可见：{missing}")
    busy = [f"{index}({usage[index]}MiB)" for index in assigned if usage[index] >= busy_threshold_mb]
    if busy:
        raise ValueError(
            "启动前 GPU 快速校验失败：分配的 GPU 已有显存占用："
            + ", ".join(busy)
        )
    logger.info(
        "Assigned GPU preflight passed: container=%s assigned=%s usage=%s thresholdMb=%s",
        docker_container,
        ",".join(str(index) for index in assigned),
        ",".join(f"{index}:{usage[index]}MiB" for index in assigned),
        busy_threshold_mb,
    )


def _start_resource_reservation_heartbeat(
    reservation_id: str,
    env_vars: Dict[str, str],
    response: ToolResponse,
) -> None:
    hint = (response.metadata or {}).get("protocol_hint") or {}
    container = str(hint.get("container") or "").strip()
    pid = str(hint.get("pid") or "").strip()
    if not container or not pid:
        logger.warning("Cannot monitor reservation %s without container and pid", reservation_id)
        return
    with _ACTIVE_RESERVATION_HEARTBEATS_LOCK:
        if reservation_id in _ACTIVE_RESERVATION_HEARTBEATS:
            return
        _ACTIVE_RESERVATION_HEARTBEATS.add(reservation_id)
    try:
        interval = max(15, int(os.getenv("MEDFLOW_TRAINING_RESERVATION_HEARTBEAT_SECONDS", "60")))
    except ValueError:
        interval = 60
    try:
        max_failures = max(1, int(os.getenv("MEDFLOW_TRAINING_RESERVATION_MAX_HEARTBEAT_FAILURES", "3")))
    except ValueError:
        max_failures = 3

    def heartbeat() -> None:
        failures = 0
        try:
            while True:
                if not _check_pid_in_container(container, pid):
                    _release_resource_allocation(reservation_id, env_vars)
                    return
                try:
                    _runtime_training_resource_request(
                        "renewTrainingResourcesForRuntime",
                        {
                            "reservationId": reservation_id,
                            "runtimeNodeId": str(
                                env_vars.get("MEDFLOW_RESOURCE_NODE_ID")
                                or os.getenv("MEDFLOW_RESOURCE_NODE_ID", "")
                            ).strip(),
                        },
                    )
                    failures = 0
                except Exception:
                    failures += 1
                    logger.exception("Failed to renew GPU resource reservation %s", reservation_id)
                    if failures >= max_failures:
                        logger.error(
                            "Stopping GPU task after %s consecutive reservation renewal failures",
                            failures,
                        )
                        try:
                            subprocess.run(
                                ["docker", "exec", container, "sh", "-c", f"kill -TERM {shlex.quote(pid)}"],
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                            for _ in range(10):
                                if not _check_pid_in_container(container, pid):
                                    break
                                time.sleep(1)
                            if _check_pid_in_container(container, pid):
                                subprocess.run(
                                    ["docker", "exec", container, "sh", "-c", f"kill -KILL {shlex.quote(pid)}"],
                                    capture_output=True,
                                    text=True,
                                    timeout=15,
                                )
                        except Exception:
                            logger.exception("Failed to stop GPU task pid=%s container=%s", pid, container)
                        _release_resource_allocation(reservation_id, env_vars)
                        return
                time.sleep(interval)
        finally:
            with _ACTIVE_RESERVATION_HEARTBEATS_LOCK:
                _ACTIVE_RESERVATION_HEARTBEATS.discard(reservation_id)

    threading.Thread(
        target=heartbeat,
        name=f"medflow-reservation-{reservation_id[:8]}",
        daemon=True,
    ).start()


def _finish_resource_launch(
    response: ToolResponse,
    reservation_id: Optional[str],
    env_vars: Dict[str, str],
) -> ToolResponse:
    metadata = response.metadata or {}
    response.metadata = metadata
    if reservation_id:
        metadata["trainingReservationId"] = reservation_id
    if metadata.get("success") is False:
        _release_resource_allocation(reservation_id, env_vars)
    elif reservation_id:
        _start_resource_reservation_heartbeat(reservation_id, env_vars, response)
    return response


def _resume_resource_reservation_heartbeats() -> None:
    if not os.path.exists(_BACKGROUND_TASK_REGISTRY):
        return
    latest: Dict[str, Dict[str, Any]] = {}
    try:
        with open(_BACKGROUND_TASK_REGISTRY, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                env = record.get("env_vars") if isinstance(record, dict) else None
                reservation_id = str((env or {}).get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip()
                if reservation_id:
                    latest[reservation_id] = record
    except OSError:
        logger.exception("Failed to read background task registry for reservation recovery")
        return
    for reservation_id, record in latest.items():
        container = str(record.get("container") or "").strip()
        pid = str(record.get("pid") or "").strip()
        env_vars = record.get("env_vars") if isinstance(record.get("env_vars"), dict) else {}
        if not container or not pid:
            continue
        if not _check_pid_in_container(container, pid):
            if _release_resource_allocation(reservation_id, env_vars):
                logger.info(
                    "Released stopped GPU resource reservation during recovery: "
                    "reservationId=%s container=%s pid=%s",
                    reservation_id,
                    container,
                    pid,
                )
            continue
        response = ToolResponse(content=[], metadata={
            "success": True,
            "protocol_hint": {"container": container, "pid": pid},
        })
        _start_resource_reservation_heartbeat(reservation_id, env_vars, response)
        logger.info("Recovered multi-node reservation heartbeat %s", reservation_id)


def _launch_mode_for_script(script_name: Optional[str]) -> Optional[str]:
    return "multinode" if _is_multinode_train_script(script_name) else None


def _display_train_type_text(train_type: Optional[str], script_name: Optional[str]) -> Optional[str]:
    if _launch_mode_for_script(script_name) == "multinode":
        if _normalize_train_type(train_type) == "lora":
            return "双机 LoRA SFT"
        if _normalize_train_type(train_type) == "enhanced":
            return "双机增强训练"
    return _train_type_text(train_type)


def _coerce_cli_value(script_info: Dict[str, Any], cli_param: str, value: Any) -> Any:
    if cli_param not in set(script_info.get("boolean_cli_params", [])):
        return str(value)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "1", "true", "yes", "y", "on", "是", "启用", "开启"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "不", "关闭"}:
        return False
    return True


def _cli_arg_value(cli_args: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = cli_args.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""



def _normalize_multinode_cli_args(script_name: str, cli_args: Dict[str, Any]) -> None:
    if not _is_multinode_train_script(script_name):
        return
    for legacy_key in ("dataset-name", "dataset_name"):
        if legacy_key in cli_args and "dataset" not in cli_args:
            cli_args["dataset"] = cli_args.get(legacy_key)
        cli_args.pop(legacy_key, None)

def _record_background_task(
    task_type: str,
    container: Optional[str],
    script_name: Optional[str],
    script_path: str,
    command: List[str],
    pid: Optional[str] = None,
    train_type: Optional[str] = None,
    script_args: Optional[Dict[str, Any]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    positional_args: Optional[List[str]] = None,
    status: str = "started",
) -> None:
    if not container:
        return

    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    train_type = _normalize_train_type(train_type)
    launch_mode = _launch_mode_for_script(script_name)
    train_type_public = public_train_type(train_type, launch_mode, script_name)
    train_type_text = public_train_type_text(train_type, launch_mode, script_name)
    reservation_id = str((env_vars or {}).get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip() or None
    record = {
        "task_type": task_type,
        "reservationId": reservation_id,
        "container": container,
        "pid": pid,
        "script_name": script_name or os.path.basename(script_path),
        "script_path": script_path,
        "train_type": train_type,
        "trainType": train_type,
        "trainTypeEn": train_type_public,
        "trainTypeText": train_type_text,
        "launch_mode": launch_mode,
        "launchMode": launch_mode,
        "isMultinode": launch_mode == "multinode",
        "command": " ".join(command),
        "script_args": script_args or {},
        "env_vars": env_vars or {},
        "positional_args": positional_args or [],
        "status": status,
        "started_at": time.time(),
        "started_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if script_name == "grpo_train":
        wandb_dir = (env_vars or {}).get("WANDB_DIR") or "/home/workspace/verl"
        wandb_mode = "offline" if (env_vars or {}).get("WANDB_MODE") == "offline" else "online"
        record.update(
            {
                "wandb_mode": wandb_mode,
                "wandb_dir": wandb_dir,
                "wandb_root": f"{wandb_dir.rstrip('/')}/wandb",
            }
        )
    with open(_BACKGROUND_TASK_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_train_pid(
    container: Optional[str],
    pid: str,
    script_name: Optional[str],
    script_path: str,
    train_type: Optional[str],
    docker_working_dir: Optional[str],
    command: str,
    env_vars: Optional[Dict[str, str]],
    script_args: Optional[Dict[str, Any]],
    positional_args: Optional[List[str]],
    launch_status: str = "started",
    launch_log_file: Optional[str] = None,
    startup_ready_marker: Optional[str] = None,
) -> None:
    if not container:
        return
    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    train_type = _normalize_train_type(train_type)
    launch_mode = _launch_mode_for_script(script_name)
    train_type_public = public_train_type(train_type, launch_mode, script_name)
    train_type_text = public_train_type_text(train_type, launch_mode, script_name)
    reservation_id = str((env_vars or {}).get("MEDFLOW_TRAINING_RESERVATION_ID") or "").strip() or None
    record = {
        "reservationId": reservation_id,
        "container": container,
        "pid": pid,
        "started_at": datetime.now().isoformat(),
        "script_name": script_name,
        "script_path": script_path,
        "train_type": train_type,
        "trainType": train_type,
        "trainTypeEn": train_type_public,
        "trainTypeText": train_type_text,
        "launch_mode": launch_mode,
        "launchMode": launch_mode,
        "isMultinode": launch_mode == "multinode",
        "docker_working_dir": docker_working_dir,
        "command": command,
        "env_vars": env_vars or {},
        "script_args": script_args or {},
        "positional_args": positional_args or [],
        "launch_status": launch_status,
        "launch_log_file": launch_log_file,
        "startup_ready_marker": startup_ready_marker,
    }
    if script_name == "grpo_train":
        wandb_dir = (env_vars or {}).get("WANDB_DIR") or "/home/workspace/verl"
        wandb_mode = "offline" if (env_vars or {}).get("WANDB_MODE") == "offline" else "online"
        record.update(
            {
                "wandb_mode": wandb_mode,
                "wandb_dir": wandb_dir,
                "wandb_root": f"{wandb_dir.rstrip('/')}/wandb",
            }
        )
    with open(_TRAIN_PID_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _check_pid_in_container(container: str, pid: str, timeout: int = 5) -> bool:
    try:
        process = subprocess.run(
            ["docker", "exec", container, "sh", "-c", f"ps -p {pid} -o pid="],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return process.returncode == 0 and bool(process.stdout.strip())
    except Exception:
        return False


def _read_container_log_tail(
    container: str,
    log_file: Optional[str],
    line_count: int = 120,
    timeout: int = 5,
) -> str:
    """读取容器内后台启动日志的有界尾部，用于返回 PID 早退的真实原因。"""
    if not container or not log_file:
        return ""
    try:
        safe_line_count = max(1, min(int(line_count), 500))
        process = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                f"test -f {shlex.quote(log_file)} && tail -n {safe_line_count} {shlex.quote(log_file)}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if process.returncode != 0:
            return ""
        return (process.stdout or "").strip()
    except Exception:
        logger.exception(
            "Failed to read startup log tail: container=%s log=%s",
            container,
            log_file,
        )
        return ""


class ScriptManager:
    def __init__(self):
        # 参数映射表：中文 -> 环境变量名
        self.param_mapping = {
            "批量大小": "MBS",
            "批次大小": "MBS", 
            "批大小": "MBS", 
            "mbs":"MBS",
            "MBS":"MBS",
            "梯度累积": "ACC",
            "累积步数": "ACC",
            "acc":"ACC",
            "ACC":"ACC",
            "gradient_accumulation_steps":"ACC",
            "学习率": "LR",
            "学习速率": "LR",
            "lr":"LR",
            "LR":"LR",
            "learning_rate":"LR",
            "模型": "TEM",
            "基础模型": "TEM",
            "tem":"TEM",
            "tem":"TEM",
            "恢复训练": "RESUME",
            "继续训练": "RESUME",
            "检查点": "RESUME",
    
            "设备": "LOCALHOST_ID",
            "gpu": "LOCALHOST_ID",
            "显卡": "LOCALHOST_ID",
            
            "容器": "container",  # 添加容器参数映射
            "docker": "container",
            "ckpt":"CKPT_PATH",
            "checkpoint":"CKPT_PATH",
            "ckpt位置":"CKPT_PATH",
            "checkpoint位置":"CKPT_PATH",
            "第一个模型": "model_fir",
            "模型一": "model_fir",
            "第一个模型路径": "model_fir",
            "第二个模型": "model_sec", 
            "模型二": "model_sec",
            "第二个模型路径": "model_sec",
            "模型位置": "model_path",
            "模型路径": "model_path",
            "数据集": "DATASET_DIR",
            "数据集路径":"DATASET_DIR",
            "dataset_dir":"DATASET_DIR",
            "sft_data_dir":"DATASET_DIR",
            "sft_dataset_dir":"DATASET_DIR",
            "data_dir":"DATASET_DIR",
            "训练数据路径":"DATASET_DIR",
            "数据集名称":"dataset_name",
            "数据日期":"DATASET_DATE",
            "数据标识":"DATASET_DATE",
            "日期":"DATASET_DATE",
            "dataset_date":"DATASET_DATE",
            "data_date":"DATASET_DATE",
            "data_identifier":"DATASET_DATE",
            "data_id":"DATASET_DATE",
            "dataset_id":"DATASET_DATE",
            "训练日期":"DATASET_DATE",
            "模型类别": "template",
            #"输出路径": "output_dir",
            "时间": "schedule_time",
            "定时时间": "schedule_time",
            "schedule_time": "schedule_time",  # 添加这一行：允许直接使用参数名
            "time":"schedule_time",
            "操作类型":"action",
            "action":"action"
        }
        
        # 可用参数的中文描述
        self.param_descriptions = {
            "MBS": "批量大小/批次大小 (Micro Batch Size)",
            "ACC": "梯度累积步数 (Gradient Accumulation Steps)",
            "LR": "学习率 (Learning Rate)",
            "TEM": "模型/基础模型 (Model Template)",
            "RESUME": "恢复训练/继续训练/检查点 (Resume from checkpoint)",
            "LOCALHOST_ID": "设备/GPU/显卡 (GPU IDs)",
            "container": "Docker容器名称 (Docker container name)" , # 添加容器描述
            
            "schedule_time": "定时时间/时间 (Schedule Time)",
            "model_fir": "第一个模型路径 (First Model Path)",
            "model_sec": "第二个模型路径 (Second Model Path)",
            "model_path": "模型路径 (Model Path)",
            "dataset_dir": "数据集路径 (Dataset Directory)",
            "dataset_name": "数据集名称 (Dataset Name)",
            "DATASET_DIR": "批量训练数据集路径 (Batch Training Dataset Directory)",
            "DATASET_DATE": "批量训练数据日期 (Batch Training Dataset Date)",
            "template": "模型模板/类别 (Model Template)",
            #"output_dir": "输出路径 (Output Directory)",
            "CKPT_PATH": "检查点路径 (Checkpoint Path)",

            "action":"操作类型"
            
        }
        
        self.scripts = {
            


            "batch_train_lora": {
                "path": "batch_train_lora",
                "description": "lora批量训练工具",
                "aliases": ["LoRA SFT", "lora_sft", "lora sft", "lora-train", "lora_train", "lora批量训练", "lora训练","训练"],
                "notes": "支持修改训练参数",
                "supports_background": True,
                "long_running": True,
                "default_env": {  # 默认环境变量配置
                    "MBS": "1",
                    "ACC": "8", 
                    "LR": "1e-7",
                    "TEM": "qwen3",
                "RESUME": ""
                },
                "supported_params": ["MBS", "ACC", "LR", "TEM", "RESUME", "LOCALHOST_ID", "container", "DATASET_DIR", "DATASET_DATE"],  # 支持修改的参数
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "log_file": "/usr/local/insinfersystem/batch_train_lora-launch.log",
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/batch_train_lora",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "node_count": 1, "gpus_per_node": 8,
                    "gpu_count_env": "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_LORA",
                },
            },
           
            "batch_train_full": {
                "path": "batch_train_full",
                "description": "全参批量训练工具",
                "aliases": ["全参 SFT", "全参SFT", "full_sft", "full sft", "full-train", "full_train", "全参批量训练", "全参训练"],
                "supports_background": True,
                "long_running": True,
                "default_env": {
                    "MBS": "1",
                    "ACC": "8",
                    "LR": "1e-6",
                    "TEM": "qwen3",
                "RESUME": ""
                },
                "supported_params": ["MBS", "ACC", "LR", "TEM", "RESUME", "LOCALHOST_ID", "container", "DATASET_DIR", "DATASET_DATE"],
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/batch_train_full",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "node_count": 1, "gpus_per_node": 8,
                    "gpu_count_env": "MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_FULL",
                },
            },
            "create_command_vpn": {
                "path": "create_command_vpn",
                "description": "定时训练工具",
                "aliases": ["定时训练", "日常训练"],
                "supports_background": True,
                "long_running": True,
                "default_env": {
                    "TEM": "qwen3",
                },
                "supported_params": [ "TEM"],
                "uses_positional_args": True,  # 新增：标记使用位置参数
                "positional_param_name": "schedule_time",  # 位置参数的名称
                "cli_args_only": True,  # 新增：标记此脚本只使用命令行参数
                "default_cli_args": {},
                "supported_cli_params": [],  # 新增：支持的命令行参数
                "param_mapping": {  # 新增：命令行参数的中文映射
                    "时间": "schedule_time",
                    "定时时间": "schedule_time",
                    "time":"schedule_time"
                      },
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_SCHEDULER_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/create_command_vpn",
                "requires_docker": True      
            },
            "dpo_train_launcher": {
                "path": "dpo_train_launcher",
                "description": "增强训练工具",
                "aliases": ["增强训练", "dpo", "dpo_train", "dpo-train", "dpo_train_launcher", "enhance", "enhanced"],
                "supports_background": True,
                "long_running": True,
                "default_env": {
    
                },
                "supported_params": [],
                "cli_args_only": True,  # 新增：标记此脚本只使用命令行参数
                "default_cli_args": {    # 新增：默认命令行参数
                    "model_path": "",
                    "dataset_dir": "",
                    "dataset_name":"",
                    "template":"qwen3",
                    #"output_dir":""
                    },
                "supported_cli_params": ["model_path", "dataset_dir","dataset_name","template"],  # 新增：支持的命令行参数
                "required_cli_params": ["model_path","dataset_dir","dataset_name"],
                "param_mapping": {  # 新增：命令行参数的中文映射
                    "模型位置": "model_path",
                    "模型路径": "model_path",
                    "model_path": "model_path",
                    "数据集": "dataset_dir", 
                    "数据集路径":"dataset_dir",
                    "dataset_dir":"dataset_dir",
                    "数据集名称":"dataset_name",
                    "dataset_name":"dataset_name",
                    "模型类别": "template",
                    "template": "template",
                    #"输出路径": "output_dir"
                    },
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/dpo_train_launcher",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "node_count": 1, "gpus_per_node": 8,
                    "gpu_count_env": "MEDFLOW_DEFAULT_GPU_COUNT_DPO_TRAIN",
                },
            },
            MULTINODE_SFT_SCRIPT: {
                "path": f"{MULTINODE_SFT_SCRIPT}.py",
                "description": "多机lora批量训练工具",
                "aliases": ["双机 LoRA SFT", "双机lora sft", "multinode_lora_sft", "multinode lora sft", "multi-node lora sft", "multinode_sft", "multinode sft", "双机lora训练", "双机sft", "多机lora批量训练", "多机lora训练", "多机sft", "多机SFT", "多机SFT LoRA", "多机sft lora训练"],
                "supports_background": True,
                "long_running": True,
                "default_env": {},
                "supported_params": [],
                "cli_args_only": True,
                "default_cli_args": {},
                "supported_cli_params": MULTINODE_SFT_CLI_PARAMS,
                "required_cli_params": [],
                "boolean_cli_params": sorted(MULTINODE_BOOLEAN_CLI_PARAMS),
                "param_mapping": MULTINODE_PARAM_MAPPING,
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": MULTINODE_DOCKER_CONTAINER,
                "docker_working_dir": "/home/workspace/llamafactory",
                "docker_executable": "python",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "multinode": True, "node_count": 2, "gpus_per_node": 1,
                },
            },
            MULTINODE_DPO_SCRIPT: {
                "path": f"{MULTINODE_DPO_SCRIPT}.py",
                "description": "多机增强训练工具",
                "aliases": [
                    "双机增强训练", "双机dpo", "双机DPO",
                    "多机增强训练", "多机dpo", "多机DPO", "多机DPO LoRA", "多机dpo训练",
                    "multinode_enhanced", "multinode enhanced", "multi-node enhanced", "multinode_dpo", "multinode dpo", "multi-node dpo",
                ],
                "supports_background": True,
                "long_running": True,
                "default_env": {},
                "supported_params": [],
                "cli_args_only": True,
                "default_cli_args": {},
                "supported_cli_params": MULTINODE_DPO_CLI_PARAMS,
                "required_cli_params": ["model-path", "dataset-dir", "dataset-name"],
                "boolean_cli_params": sorted(MULTINODE_BOOLEAN_CLI_PARAMS),
                "param_mapping": MULTINODE_PARAM_MAPPING,
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": MULTINODE_DOCKER_CONTAINER,
                "docker_working_dir": "/home/workspace/llamafactory",
                "docker_executable": "python",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "multinode": True, "node_count": 2, "gpus_per_node": 1,
                },
            },
            "grpo_train": {
                "path": GRPO_SOURCE_SCRIPT,
                "description": "GRPO训练工具",
                "aliases": ["grpo训练", "GRPO训练", "grpo", "GRPO"],
                "supports_background": True,
                "long_running": True,
                "default_env": {
                    "PYTHONUNBUFFERED": "1",
                    "TQDM_POSITION": "-1",
                    "TQDM_MININTERVAL": "0",
                    "TQDM_MINITERS": "1",
                    "WANDB_CONSOLE": "redirect",
                    "WANDB_DIR": GRPO_SOURCE_WORKDIR,
                },
                "supported_params": [],
                "cli_args_only": True,
                "default_cli_args": {
                    "model_path": "",
                    "train_files": "",
                    "val_files": "",
                    "wandb_mode": "online",
                },
                "supported_cli_params": ["model_path", "train_files", "val_files", "wandb_mode"],
                "required_cli_params": ["model_path", "train_files", "val_files"],
                "param_mapping": {
                    "模型路径": "model_path",
                    "model_path": "model_path",
                    "模型位置": "model_path",
                    "actor_rollout_ref.model.path": "model_path",
                    "训练数据": "train_files",
                    "训练集": "train_files",
                    "训练文件": "train_files",
                    "train_files": "train_files",
                    "data.train_files": "train_files",
                    "验证数据": "val_files",
                    "验证集": "val_files",
                    "验证文件": "val_files",
                    "val_files": "val_files",
                    "data.val_files": "val_files",
                    "wandb_mode": "wandb_mode",
                    "wandb模式": "wandb_mode",
                    "W&B模式": "wandb_mode",
                    "WandB模式": "wandb_mode",
                    "离线wandb": "wandb_mode",
                    "离线W&B": "wandb_mode",
                    "wandb离线": "wandb_mode",
                    "W&B离线": "wandb_mode",
                    "在线wandb": "wandb_mode",
                    "在线W&B": "wandb_mode",
                    "wandb在线": "wandb_mode",
                    "W&B在线": "wandb_mode",
                },
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    "startup_ready_marker": "MEDFLOW_TRAINING_COMMAND_READY",
                    "startup_ready_timeout": 15,
                    "startup_poll_interval": 1,
                },
                "docker_container": GRPO_DOCKER_CONTAINER,
                "docker_working_dir": GRPO_SOURCE_WORKDIR,
                "docker_executable": "bash",
                "requires_docker": True,
                "resource_request": {
                    "enabled": True, "node_count": 1, "gpus_per_node": 4,
                    "gpu_count_env": "MEDFLOW_DEFAULT_GPU_COUNT_GRPO_TRAIN",
                },
            },
        }
        
        # 添加脚本配置验证
        self._validate_scripts()
    
    def _validate_scripts(self):
        """验证脚本配置"""
        for name, info in self.scripts.items():
            script_path = info.get("path")
            if script_path and not os.path.exists(script_path):
                # 尝试从当前目录查找
                current_dir = os.path.join(os.getcwd(), script_path)
                #if not os.path.exists(current_dir):
                    #print(f"警告: 脚本 {name} 的路径在本容器内不存在: {script_path}")
    
    def map_chinese_to_env_var(self, chinese_param: str) -> Optional[str]:
        """将中文参数名映射到环境变量名"""
        return self.param_mapping.get(chinese_param)
    
    def get_available_params(self, script_name: str) -> List[str]:
        """获取脚本支持的参数列表（中文）"""
        script_info = self.scripts.get(script_name)
        if not script_info:
            return []
        
        supported_env_vars = script_info.get("supported_params", [])
        # 将环境变量名映射回中文（可能有多个中文对应一个环境变量）
        chinese_params = []
        for env_var in supported_env_vars:
            # 找到所有映射到这个环境变量的中文参数名
            chinese_list = [ch for ch, ev in self.param_mapping.items() if ev == env_var]
            chinese_params.extend(chinese_list)
        
        return list(set(chinese_params))  # 去重
    
    def get_param_description(self, env_var: str) -> str:
        """获取参数描述"""
        return self.param_descriptions.get(env_var, env_var)
    
    def find_script(self, query: str) -> Optional[Dict[str, Any]]:
        """根据名称、别名或描述查找脚本"""
        query = query.lower()
        normalized_query = re.sub(r"[\s_-]+", "", query)
        if "多机" in query or "multinode" in query or "multi-node" in query or "2node" in query:
            if "dpo" in query or "增强" in query:
                info = self.scripts.get(MULTINODE_DPO_SCRIPT)
                return {**info, "name": MULTINODE_DPO_SCRIPT} if info else None
            if "sft" in query or "lora" in query or "批量" in query:
                info = self.scripts.get(MULTINODE_SFT_SCRIPT)
                return {**info, "name": MULTINODE_SFT_SCRIPT} if info else None
        
        for name, info in self.scripts.items():
            # 检查名称
            if query == name.lower():
                return {**info, "name": name}
            
            # 检查别名
            for alias in info.get("aliases", []):
                normalized_alias = re.sub(r"[\s_-]+", "", alias.lower())
                if query == alias.lower() or normalized_query == normalized_alias:
                    return {**info, "name": name}
            
            # 检查描述
            if query in info.get("description", "").lower():
                return {**info, "name": name}
        
        return None
    
    def get_script_info_with_cli_params(self, script_name: str) -> str:
        """获取脚本的详细信息，包括命令行参数"""
        script_info = self.scripts.get(script_name)
        if not script_info:
            return ""
    
        result = []
        result.append(f"脚本: {script_name}")
        result.append(f"描述: {script_info.get('description', '')}")
    
        # 显示命令行参数
        if script_info.get("cli_args_only", False):
            result.append("参数传递方式: 命令行参数")
            cli_params = script_info.get("supported_cli_params", [])
            if cli_params:
                result.append("支持的命令行参数:")
                for cli_param in cli_params:
                    # 找到对应的中文名
                    chinese_names = [ch for ch, ev in script_info.get("param_mapping", {}).items() if ev == cli_param]
                    # 标记必需参数
                    is_required = cli_param in script_info.get("required_cli_params", [])
                    if chinese_names:
                        result.append(f"  - {cli_param} ({'/'.join(chinese_names)}) {'(必需)' if is_required else ''}")
                    else:
                        result.append(f"  - {cli_param} {'(必需)' if is_required else ''}")
        else:
            # 显示环境变量参数
            result.append("参数传递方式: 环境变量")
            supported_params = script_info.get("supported_params", [])
            if supported_params:
                result.append("支持的环境变量参数:")
                for param in supported_params:
                    # 找到对应的中文名
                    chinese_names = [ch for ch, ev in self.param_mapping.items() if ev == param]
                    desc = self.get_param_description(param)
                    if chinese_names:
                        result.append(f"  - {param} ({'/'.join(chinese_names)}) - {desc}")
                    else:
                        result.append(f"  - {param} - {desc}")
    
        return "\n".join(result)
    
    
    
    
    
    def list_scripts(self) -> str:
        """列出所有可用脚本"""
        result = ["可用脚本："]
        for name, info in self.scripts.items():
            result.append(f"- {name}: {info['description']}")
            result.append(f"  别名: {', '.join(info.get('aliases', []))}")
            result.append(f"  路径: {info['path']}")
            
            if info.get("cli_args_only", False):
                # 显示命令行参数
                cli_params = info.get('supported_cli_params', [])
                if cli_params:
                    param_descs = []
                    for param in cli_params:
                        # 找到对应的中文名
                        chinese_names = [ch for ch, ev in info.get('param_mapping', {}).items() if ev == param]
                        is_required = param in info.get('required_cli_params', [])
                        if chinese_names:
                            param_descs.append(f"{param} ({'/'.join(chinese_names)}) {'(必需)' if is_required else ''}")
                        else:
                            param_descs.append(f"{param} {'(必需)' if is_required else ''}")
                    result.append(f"  命令行参数: {', '.join(param_descs)}")
            else:
                # 显示环境变量参数
                supported_params = info.get('supported_params', [])
                if supported_params:
                    param_descs = []
                    for param in supported_params:
                        desc = self.get_param_description(param)
                        # 找到对应的中文名
                        chinese_names = [ch for ch, ev in self.param_mapping.items() if ev == param]
                        if chinese_names:
                            param_descs.append(f"{param} ({'/'.join(chinese_names)}) - {desc}")
                        else:
                            param_descs.append(f"{param} - {desc}")
                    result.append(f"  可修改参数: {', '.join(param_descs)}")
            
            # 显示默认配置
            if info.get('default_env'):
                result.append(f"  默认环境变量: {info['default_env']}")
            if info.get('default_cli_args'):
                result.append(f"  默认命令行参数: {info['default_cli_args']}")
            
            # 显示特殊属性
            if info.get('supports_background'):
                result.append("  支持后台运行")
            if info.get('long_running'): 
                result.append("  长时间运行任务")
            result.append("")  # 空行分隔
        return "\n".join(result)


def run_local_script(
    script_path: str = "test.py",
    script_args: Dict[str, Any] = None,
    env_vars: Dict[str, str] = None,
    timeout: int = None,
    capture_output: bool = True,
    working_dir: str = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    positional_args: List[str] = None,  # 新增：位置参数
    **kwargs
) -> ToolResponse:
    """
    运行本地Python脚本并返回执行结果。
    """
    
    try:
        if docker_container:
            return run_script_in_docker(
                script_path=script_path,
                script_args=script_args,
                positional_args=positional_args,  # 传递给Docker函数
                env_vars=env_vars,
                timeout=timeout,
                capture_output=capture_output,
                working_dir=working_dir,
                docker_container=docker_container,
                docker_image=docker_image,
                docker_volume_mounts=docker_volume_mounts,
                docker_working_dir=docker_working_dir,
                docker_executable=docker_executable,
                **kwargs
            )
        
        # 检查脚本文件是否存在
        abs_script_path = os.path.abspath(script_path)
        if not os.path.exists(abs_script_path):
            # 尝试在当前目录下查找
            current_path = os.path.join(os.getcwd(), script_path)
            if os.path.exists(current_path):
                abs_script_path = os.path.abspath(current_path)
            else:
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"错误: 找不到脚本文件: {script_path}\n尝试的路径: {abs_script_path}\n当前目录: {os.getcwd()}"
                    )
                ])
        
        # 构建Python执行命令
        python_executable = sys.executable
        command = [python_executable, abs_script_path]
        if positional_args:
            for arg in positional_args:
                command.append(str(arg))
        # 添加命令行参数
        if script_args:
            for key, value in script_args.items():
                # 处理不同类型的参数
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    # 对于字符串参数，确保特殊字符不被shell解释
                    # 特别是对于时间参数如 "{16:08:55}"，需要直接传递
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                    continue
                else:
                    command.extend([f"--{key}", str(value)])
        
        # 设置环境变量
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # 设置工作目录
        cwd = working_dir if working_dir else os.path.dirname(abs_script_path)
        if not cwd or not os.path.exists(cwd):
            cwd = os.getcwd()
        
        # 打印调试信息
        print(f"执行脚本: {abs_script_path}")
        print(f"工作目录: {cwd}")
        print(f"命令: {' '.join(command)}")
        print(f"环境变量: {env_vars}")
        
        # 执行脚本
        if timeout:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd
            )
        else:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                env=env,
                cwd=cwd
            )
        
        # 构建结果消息
        result_text = []
        
        if process.stdout and capture_output:
            result_text.append(f"标准输出:\n{process.stdout}")
        
        if process.stderr and capture_output:
            result_text.append(f"标准错误:\n{process.stderr}")
        
        # 添加返回码信息
        result_text.append(f"返回码: {process.returncode}")
        result_text.append(f"######请检查对应的log文件######")
        # 如果没有捕获输出但有返回码，只显示返回码
        if not capture_output:
            result_text = [f"脚本执行完成，返回码: {process.returncode}"]
        
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="\n\n".join(result_text)
            )
        ])
        
    except subprocess.TimeoutExpired:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 脚本执行超时 (超过 {timeout} 秒)"
            )
        ])
    
    except FileNotFoundError as e:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 找不到Python解释器或脚本文件: {str(e)}"
            )
        ])
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 执行脚本时发生异常: {str(e)}\n\n详细错误:\n{error_trace}"
            )
        ])


def check_required_params(script_info: Dict[str, Any], cli_args: Dict[str, str]) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    检查必需参数是否已提供
    
    Args:
        script_info: 脚本信息
        cli_args: 当前已提供的命令行参数
        
    Returns:
        Tuple[是否通过检查, 缺失参数列表, 需要询问的参数信息]
    """
    required_params = script_info.get("required_cli_params", [])
    missing_params = []
    params_to_ask = {}
    
    for param in required_params:
        # 检查参数是否已提供
        if param == "dataset-name" and _is_multinode_train_script(script_info.get("name")) and cli_args.get("dataset"):
            continue
        if param not in cli_args or not cli_args[param]:
            missing_params.append(param)
            
            # 获取参数描述
            param_desc = param
            param_mapping = script_info.get("param_mapping", {})
            chinese_names = [ch for ch, ev in param_mapping.items() if ev == param]
            if chinese_names:
                param_desc = f"{param} ({'/'.join(chinese_names)})"
            
            params_to_ask[param] = {
                "name": param,
                "description": param_desc,
                "chinese_names": chinese_names
            }
    
    return len(missing_params) == 0, missing_params, params_to_ask


def _is_placeholder_value(param_name: str, value: Any) -> bool:
    """识别模型编造的占位参数，避免误启动后台任务。"""
    normalized_param_name = (param_name or "").replace("-", "_")
    if value is None:
        return True

    text = str(value).strip().strip("'\"")
    if not text:
        return True

    lower = text.lower()
    placeholders = {
        "模型位置",
        "模型路径",
        "数据集位置",
        "数据集路径",
        "数据集名称",
        "待提供",
        "未提供",
        "未知",
        "默认",
        "xxx",
        "none",
        "null",
        "unknown",
        "model_path",
        "dataset_dir",
        "dataset_name",
    }
    if text in placeholders or lower in placeholders:
        return True

    if normalized_param_name in {"model_path", "dataset_dir"}:
        # 增强训练需要真实路径；单纯日期或自然语言短语不能作为路径执行。
        if re.fullmatch(r"20\d{6}", text):
            return True
        if not text.startswith("/") and not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
            return True

    if normalized_param_name == "dataset_name" and re.fullmatch(r"20\d{6}(数据)?", text):
        return True

    return False


def validate_cli_param_values(script_info: Dict[str, Any], cli_args: Dict[str, str]) -> List[str]:
    invalid = []
    for param in script_info.get("required_cli_params", []):
        value = cli_args.get(param)
        if param == "dataset-name" and _is_multinode_train_script(script_info.get("name")):
            value = value or cli_args.get("dataset")
        if _is_placeholder_value(param, value):
            invalid.append(f"{param}={value!r}")
    return invalid


def _list_dataset_candidates_in_container(container: str, dataset_dir: str) -> List[str]:
    """列出增强训练数据目录内可作为 dataset_name 的 json 文件名（不含 .json）。"""
    try:
        info_command = f"cat {shlex.quote(dataset_dir.rstrip('/') + '/dataset_info.json')} 2>/dev/null"
        info_process = subprocess.run(
            ["docker", "exec", container, "sh", "-c", info_command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if info_process.returncode == 0 and info_process.stdout.strip():
            try:
                dataset_info = json.loads(info_process.stdout)
                candidates = dataset_names_from_dataset_info(dataset_info)
                if candidates:
                    return candidates
            except json.JSONDecodeError:
                pass

        excluded = dataset_find_exclusion_clause()
        command = (
            f"find {shlex.quote(dataset_dir)} -maxdepth 1 -type f -name '*.json' "
            f"{excluded} -printf '%f\n' | sort"
        )
        process = subprocess.run(
            ["docker", "exec", container, "sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if process.returncode != 0 or not process.stdout.strip():
            return []
        candidates = []
        for filename in process.stdout.strip().splitlines():
            filename = filename.strip()
            if not is_dataset_candidate_filename(filename):
                continue
            candidates.append(filename[:-5])
        return candidates
    except Exception:
        return []

def _choose_dataset_name(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def score(name: str) -> int:
        lower = name.lower()
        value = 0
        if "for_train" in lower:
            value += 100
        if "train" in lower:
            value += 50
        if "altered" in lower:
            value += 20
        if "output" in lower:
            value += 10
        if "qwen" in lower:
            value += 5
        if "test" in lower:
            value -= 100
        return value

    ranked = sorted(((score(name), name) for name in candidates), reverse=True)
    top_score, top_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -10**9
    if top_score >= 50 and top_score > second_score:
        return top_name
    return None


def _normalize_dpo_cli_args(script_name: str, cli_args: Dict[str, str], container: str) -> None:
    """把 dpo_train_launcher 中误作为 dataset_name 的日期改成 dataset_dir 子目录。"""
    if script_name not in {"dpo_train_launcher", MULTINODE_DPO_SCRIPT}:
        return
    dataset_dir_key = "dataset-dir" if "dataset-dir" in cli_args else "dataset_dir"
    dataset_name_key = "dataset-name" if "dataset-name" in cli_args else ("dataset" if "dataset" in cli_args else "dataset_name")
    dataset_dir = str(cli_args.get(dataset_dir_key, "")).strip().rstrip("/")
    dataset_name = str(cli_args.get(dataset_name_key, "")).strip()
    if not dataset_dir or not re.fullmatch(r"20\d{6}", dataset_name):
        return

    if not dataset_dir.endswith(f"/{dataset_name}"):
        dataset_dir = f"{dataset_dir}/{dataset_name}"
        cli_args[dataset_dir_key] = dataset_dir

    inferred_dataset_name = _choose_dataset_name(
        _list_dataset_candidates_in_container(container, dataset_dir),
    )
    if inferred_dataset_name:
        cli_args[dataset_name_key] = inferred_dataset_name


def _training_template_value(
    script_name: str,
    env_vars: Optional[Dict[str, Any]],
    cli_args: Optional[Dict[str, Any]],
) -> str:
    env_vars = env_vars or {}
    cli_args = cli_args or {}
    if script_name in {"batch_train_lora", "batch_train_full"}:
        return str(env_vars.get("TEM") or DEFAULT_TEMPLATE).strip()
    if script_name in {MULTINODE_SFT_SCRIPT, "dpo_train_launcher", MULTINODE_DPO_SCRIPT}:
        return first_nonempty(
            [
                cli_args.get("template"),
                cli_args.get("TEM"),
                env_vars.get("template"),
                env_vars.get("TEM"),
                DEFAULT_TEMPLATE,
            ]
        )
    return ""


def _training_model_hint(
    script_name: str,
    env_vars: Optional[Dict[str, Any]],
    cli_args: Optional[Dict[str, Any]],
) -> str:
    env_vars = env_vars or {}
    cli_args = cli_args or {}
    values: List[Any] = []
    if script_name in {MULTINODE_SFT_SCRIPT, "dpo_train_launcher", MULTINODE_DPO_SCRIPT}:
        values.extend(
            [
                cli_args.get("model_path"),
                cli_args.get("model-path"),
                env_vars.get("model_path"),
                env_vars.get("model-path"),
            ]
        )
    return first_nonempty(values)


def _training_template_policy_issue(
    script_name: str,
    container: str,
    env_vars: Optional[Dict[str, Any]],
    cli_args: Optional[Dict[str, Any]],
    *,
    explicit_template: bool,
    model_hint: str = "",
) -> Optional[Dict[str, Any]]:
    if script_name not in {
        "batch_train_lora",
        "batch_train_full",
        "dpo_train_launcher",
        MULTINODE_SFT_SCRIPT,
        MULTINODE_DPO_SCRIPT,
    }:
        return None
    template = _training_template_value(script_name, env_vars, cli_args)
    param = "TEM" if script_name in {"batch_train_lora", "batch_train_full"} else "template"
    template_issue = template_validation_issue(template, param=param, container=container)
    if template_issue:
        return template_issue

    detected_model_hint = model_hint or _training_model_hint(script_name, env_vars, cli_args)
    if (
        not explicit_template
        and normalize_template(template) == DEFAULT_TEMPLATE
        and text_mentions_non_qwen3_model(detected_model_hint)
    ):
        return non_qwen3_template_required_issue(
            param=param,
            container=container,
            model_hint=detected_model_hint,
        )
    return None


def validate_training_inputs_preflight(
    script_name: str,
    container: str,
    env_vars: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate local training inputs inside Docker before reserving GPUs."""
    env_vars = env_vars or {}
    cli_args = cli_args or {}
    script_name = str(script_name or "").strip()
    container = str(container or "").strip()
    supported_scripts = {
        "batch_train_lora",
        "batch_train_full",
        "dpo_train_launcher",
        MULTINODE_SFT_SCRIPT,
        MULTINODE_DPO_SCRIPT,
        "grpo_train",
    }
    if script_name not in supported_scripts:
        return None

    template_issue = _training_template_policy_issue(
        script_name,
        container,
        env_vars,
        cli_args,
        explicit_template=False,
    )
    if template_issue:
        return template_issue
    def issue(reason: str, message: str, **fields: Any) -> Dict[str, Any]:
        return {
            "error_reason": reason,
            "message": message,
            "container": container,
            **fields,
        }

    def run_shell(command: str) -> Any:
        try:
            result = subprocess.run(
                ["docker", "exec", container, "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return issue(
                "dataset_validation_unavailable",
                f"无法在容器 {container} 中校验训练输入：{exc}",
            )
        stderr = str(result.stderr or "").strip()
        unavailable_markers = (
            "no such container",
            "is not running",
            "cannot connect to the docker daemon",
            "permission denied",
            "error response from daemon",
        )
        if result.returncode != 0 and any(
            marker in stderr.lower() for marker in unavailable_markers
        ):
            return issue(
                "dataset_validation_unavailable",
                f"无法访问训练容器 {container}，已阻止启动：{stderr or 'Docker校验失败'}",
            )
        return result

    def check_path(path: str, flag: str, reason: str, label: str, **fields: Any) -> Optional[Dict[str, Any]]:
        clean_path = str(path or "").strip()
        if not clean_path:
            return issue(reason, f"{label}未提供。", **fields)
        check = run_shell(f"test {flag} {shlex.quote(clean_path)}")
        if isinstance(check, dict):
            return check
        if check.returncode != 0:
            return issue(reason, f"{label}不存在或类型不正确：{clean_path}（容器：{container}）。", **fields)
        return None

    def validate_resume_checkpoint() -> Optional[Dict[str, Any]]:
        resume = (
            str(env_vars.get("RESUME") or "").strip()
            or _cli_arg_value(cli_args, "resume-from-checkpoint", "resume_from_checkpoint").strip()
        )
        if not resume:
            return None
        return check_path(
            resume,
            "-d",
            "checkpoint_missing",
            "续训 checkpoint 路径",
            checkpoint_path=resume,
        )

    def validate_dataset_info_dir(
        dataset_dir: str,
        *,
        dataset_names: Optional[List[str]] = None,
        missing_reason: str = "dataset_info_missing",
    ) -> Optional[Dict[str, Any]]:
        dataset_dir = str(dataset_dir or "").strip().rstrip("/")
        directory_issue = check_path(dataset_dir, "-d", "dataset_dir_missing", "训练数据目录", dataset_dir=dataset_dir)
        if directory_issue:
            return directory_issue

        info_result = run_shell(f"cat {shlex.quote(dataset_dir + '/dataset_info.json')} 2>/dev/null")
        if isinstance(info_result, dict):
            return info_result
        if info_result.returncode != 0 or not str(info_result.stdout or "").strip():
            return issue(
                missing_reason,
                (
                    f"训练数据目录缺少 dataset_info.json：{dataset_dir}。"
                    "当前数据集目录并非有效的训练数据目录；高级筛选未产出可训练数据或数据筛选流程未产出最新的可训练数据。建议您前往“数据管理”页面，选择可用的数据后重新进行训练。"
                ),
                dataset_dir=dataset_dir,
            )
        try:
            dataset_info = json.loads(info_result.stdout)
        except (TypeError, ValueError):
            return issue(
                "dataset_info_invalid",
                f"dataset_info.json 格式无效：{dataset_dir}/dataset_info.json",
                dataset_dir=dataset_dir,
            )
        if not isinstance(dataset_info, dict) or not dataset_info:
            return issue(
                "dataset_info_empty",
                f"dataset_info.json 没有注册任何可训练数据集：{dataset_dir}/dataset_info.json",
                dataset_dir=dataset_dir,
            )

        selected_names = dataset_names or [str(name).strip() for name in dataset_info.keys() if str(name).strip()]
        if not selected_names:
            return issue(
                "dataset_name_missing",
                f"未提供可校验的 dataset_name（数据目录：{dataset_dir}）。",
                dataset_dir=dataset_dir,
            )

        missing_dataset_names: List[str] = []
        missing_file_names: List[str] = []
        unsafe_file_names: List[str] = []
        missing_dataset_files: List[str] = []
        for raw_name in selected_names:
            key = str(raw_name).strip()
            if not key:
                continue
            entry = dataset_info.get(key)
            if not isinstance(entry, dict):
                missing_dataset_names.append(key)
                continue
            file_name = entry.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                missing_file_names.append(key)
                continue
            clean_file_name = file_name.strip()
            parts = [part for part in clean_file_name.split("/") if part]
            if clean_file_name.startswith("/") or "\\" in clean_file_name or ".." in parts:
                unsafe_file_names.append(f"{key}: {clean_file_name}")
                continue
            file_check = run_shell(f"test -f {shlex.quote(dataset_dir + '/' + clean_file_name)}")
            if isinstance(file_check, dict):
                return file_check
            if file_check.returncode != 0:
                missing_dataset_files.append(f"{key}: {clean_file_name}")

        if missing_dataset_names:
            return issue(
                "dataset_name_missing",
                f"dataset_info.json 中找不到 dataset_name：{', '.join(missing_dataset_names)}。",
                dataset_dir=dataset_dir,
                missing_dataset_names=missing_dataset_names,
            )
        if missing_file_names:
            return issue(
                "dataset_info_file_name_missing",
                f"dataset_info.json 中以下数据集缺少 file_name：{', '.join(missing_file_names)}。",
                dataset_dir=dataset_dir,
                missing_dataset_names=missing_file_names,
            )
        if unsafe_file_names:
            return issue(
                "dataset_info_file_name_invalid",
                f"dataset_info.json 中以下 file_name 非法：{', '.join(unsafe_file_names)}。",
                dataset_dir=dataset_dir,
                invalid_file_names=unsafe_file_names,
            )
        if missing_dataset_files:
            return issue(
                "dataset_files_missing",
                f"dataset_info.json 引用的数据文件不存在：{', '.join(missing_dataset_files)}。",
                dataset_dir=dataset_dir,
                missing_dataset_files=missing_dataset_files,
            )
        return None

    def split_local_files(value: str) -> List[str]:
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    resume_issue = validate_resume_checkpoint()
    if resume_issue and script_name in {"batch_train_lora", "batch_train_full", MULTINODE_SFT_SCRIPT, MULTINODE_DPO_SCRIPT}:
        return resume_issue

    if script_name in {"batch_train_lora", "batch_train_full"}:
        dataset_dir = str(env_vars.get("DATASET_DIR") or "").strip().rstrip("/")
        dataset_date = str(env_vars.get("DATASET_DATE") or "").strip()
        if not dataset_dir and dataset_date:
            dataset_dir = f"/home/workspace/dataset_batch_train/{dataset_date}"
        if not dataset_dir:
            latest = run_shell(
                "find /home/workspace/dataset_batch_train -mindepth 1 -maxdepth 1 "
                "-type d -printf '%f\\n' 2>/dev/null | sort | tail -n 1"
            )
            if isinstance(latest, dict):
                return latest
            latest_name = str(latest.stdout or "").strip()
            if latest.returncode != 0 or not latest_name:
                return issue(
                    "dataset_dir_missing",
                    "未找到可用的批量训练数据目录，请提供 DATASET_DIR 或 DATASET_DATE。",
                )
            dataset_dir = f"/home/workspace/dataset_batch_train/{latest_name}"
        return validate_dataset_info_dir(dataset_dir)

    if script_name == MULTINODE_SFT_SCRIPT:
        dataset_dir = _cli_arg_value(cli_args, "dataset-dir", "dataset_dir").strip().rstrip("/")
        dataset_date = _cli_arg_value(cli_args, "dataset-date", "dataset_date").strip()
        if not dataset_dir and dataset_date:
            base_dataset_path = _cli_arg_value(cli_args, "base-dataset-path", "base_dataset_path").strip() or "/home/workspace/dataset_batch_train"
            dataset_dir = f"{base_dataset_path.rstrip('/')}/{dataset_date}"
        if not dataset_dir:
            return issue(
                "dataset_dir_missing",
                "多机 SFT 未提供 dataset-dir 或 dataset-date，已阻止启动。",
            )
        return validate_dataset_info_dir(dataset_dir)

    if script_name == "grpo_train":
        grpo_values = cli_args.get("_grpo_preflight") if isinstance(cli_args.get("_grpo_preflight"), dict) else {}
        model_path = str(grpo_values.get("model_path") or cli_args.get("model_path") or "").strip()
        train_files = str(grpo_values.get("train_files") or cli_args.get("train_files") or "").strip()
        val_files = str(grpo_values.get("val_files") or cli_args.get("val_files") or "").strip()
        model_issue = check_path(model_path, "-e", "model_path_missing", "GRPO 模型路径", model_path=model_path)
        if model_issue:
            return model_issue
        for param_name, raw_value, label in (
            ("train_files", train_files, "GRPO 训练文件"),
            ("val_files", val_files, "GRPO 验证文件"),
        ):
            files = split_local_files(raw_value)
            if not files:
                return issue(
                    "training_files_missing",
                    f"{label}未提供。",
                    param=param_name,
                )
            invalid_files = [item for item in files if not item.startswith("/")]
            if invalid_files:
                return issue(
                    "training_files_invalid",
                    f"{label}必须是容器内绝对路径：{', '.join(invalid_files)}。",
                    param=param_name,
                    invalid_files=invalid_files,
                )
            for file_path in files:
                file_issue = check_path(
                    file_path,
                    "-f",
                    "training_files_missing",
                    label,
                    param=param_name,
                    file_path=file_path,
                )
                if file_issue:
                    return file_issue
        return None

    model_path = _cli_arg_value(cli_args, "model_path", "model-path")
    dataset_dir = _cli_arg_value(cli_args, "dataset_dir", "dataset-dir").rstrip("/")
    dataset_name = _cli_arg_value(cli_args, "dataset_name", "dataset-name", "dataset")
    for path, reason, label in (
        (model_path, "model_path_missing", "基础模型路径"),
        (dataset_dir, "dataset_dir_missing", "训练数据目录"),
    ):
        check = run_shell(f"test -e {shlex.quote(path)}")
        if isinstance(check, dict):
            return check
        if not path or check.returncode != 0:
            return issue(
                reason,
                f"{label}不存在：{path or '未提供'}（容器：{container}）。",
                model_path=model_path,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
            )

    names = [name.strip() for name in dataset_name.split(",") if name.strip()]
    if not names:
        return issue(
            "dataset_name_missing",
            f"增强训练未提供 dataset_name（数据目录：{dataset_dir}）。",
            dataset_dir=dataset_dir,
        )

    info: Dict[str, Any] = {}
    info_result = run_shell(f"cat {shlex.quote(dataset_dir + '/dataset_info.json')} 2>/dev/null")
    if isinstance(info_result, dict):
        return info_result
    if info_result.returncode == 0 and str(info_result.stdout or "").strip():
        try:
            decoded = json.loads(info_result.stdout)
            if isinstance(decoded, dict):
                info = decoded
        except (TypeError, ValueError):
            return issue(
                "dataset_validation_unavailable",
                f"dataset_info.json 格式无效：{dataset_dir}/dataset_info.json",
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
            )

    missing_names: List[str] = []
    for name in names:
        candidates = [name] if name.endswith((".json", ".jsonl")) else [f"{name}.json", f"{name}.jsonl"]
        entry = info.get(name)
        if isinstance(entry, dict):
            file_name = entry.get("file_name")
            if isinstance(file_name, str) and file_name.strip():
                candidates.insert(0, file_name.strip())
            elif isinstance(file_name, list):
                candidates = [str(value).strip() for value in file_name if str(value).strip()] + candidates
        command = " || ".join(
            f"test -f {shlex.quote(dataset_dir + '/' + candidate)}"
            for candidate in dict.fromkeys(candidates)
        )
        check = run_shell(command)
        if isinstance(check, dict):
            return check
        if check.returncode != 0:
            missing_names.append(name)
    if missing_names:
        return issue(
            "dataset_name_missing",
            f"数据目录 {dataset_dir} 中找不到 dataset_name 对应文件：{', '.join(missing_names)}。",
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            missing_dataset_names=missing_names,
        )
    return None

def _training_preflight_issue(reason: str, message: str, **fields: Any) -> Dict[str, Any]:
    return {
        "error_reason": reason,
        "message": message,
        **fields,
    }


def _validate_training_container_preflight(container: str) -> Optional[Dict[str, Any]]:
    container = str(container or "").strip()
    if not container:
        return _training_preflight_issue(
            "container_required",
            "错误: 需要指定Docker容器才能运行脚本",
            required_params=["container"],
        )
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return _training_preflight_issue(
            "container_preflight_failed",
            f"训练容器 {container} 预检失败，无法访问 Docker：{exc}",
            container=container,
        )
    if result.returncode != 0:
        detail = str(result.stderr or result.stdout or "Docker校验失败").strip()
        return _training_preflight_issue(
            "container_preflight_failed",
            f"训练容器 {container} 不存在或未运行：{detail}",
            container=container,
        )
    if str(result.stdout or "").strip().lower() != "true":
        return _training_preflight_issue(
            "container_preflight_failed",
            f"训练容器 {container} 存在但未运行，请先启动容器。",
            container=container,
        )
    return None


def run_training_preflight(
    script_name: str,
    container: str,
    env_vars: Dict[str, str],
    cli_args: Dict[str, Any],
    script_info: Dict[str, Any],
    launcher_options: Dict[str, Any],
    resource_request: Optional[Dict[str, Any]],
    container_prechecked: bool = False,
) -> Optional[Dict[str, Any]]:
    if not container_prechecked:
        container_issue = _validate_training_container_preflight(container)
        if container_issue:
            return container_issue

    dataset_issue = validate_training_inputs_preflight(
        script_name,
        container,
        env_vars,
        cli_args,
    )
    if dataset_issue:
        reason = str(dataset_issue.get("error_reason") or "dataset_preflight_failed")
        if reason == "dataset_validation_unavailable":
            dataset_issue["error_reason"] = "dataset_preflight_failed"
        return dataset_issue

    if resource_request:
        try:
            _preflight_resource_allocation(
                script_info,
                cli_args,
                env_vars,
                launcher_options,
            )
        except Exception as exc:
            return _training_preflight_issue(
                "resource_preflight_failed",
                f"错误: GPU 资源预检失败: {exc}",
                container=container,
            )
    return None


def _train_protocol_hint(
    protocol_type: str,
    script_name: str,
    message: str = "",
    **fields: Any,
) -> Dict[str, Any]:
    script_name = str(script_name or "").strip()
    script_for_inference = str(
        fields.get("script") or fields.get("scriptName") or script_name
    ).strip()
    hint = {
        "type": protocol_type,
        "agent": "trainer",
        "message": message,
        "script": script_for_inference or script_name,
        "scriptName": script_for_inference or script_name,
        "content": message,
    }
    if protocol_type == "need_input":
        required_params = fields.get("requiredParams") or fields.get("required_params") or []
        fields.setdefault("status", "needs_input")
        fields.setdefault("action", "collect_params")
        fields.setdefault("requiredParams", required_params)
        if required_params:
            fields.setdefault("missingParams", required_params)
        if script_name in {"dpo_train_launcher", MULTINODE_DPO_SCRIPT} or any(
            param in required_params for param in ["model_path", "dataset_dir", "dataset_name", "model-path", "dataset-dir", "dataset-name"]
        ):
            fields.setdefault("kind", "training_params")
        elif script_name == "create_command_vpn" or "schedule_time" in required_params:
            fields.setdefault("kind", "schedule_time")
        else:
            fields.setdefault("kind", "training_params")
    elif protocol_type in {"job_started", "job_preparing"}:
        fields.setdefault("status", "running" if protocol_type == "job_started" else "preparing")
        fields.setdefault("action", "start")
        fields.setdefault("background", True)
        fields.setdefault("jobType", "train")
        train_type = (
            fields.get("trainType")
            or fields.get("train_type")
            or infer_train_type_from_name(script_for_inference or script_name)
        )
        train_type = _normalize_train_type(train_type)
        launch_mode = fields.get("launchMode") or fields.get("launch_mode")
        launch_mode = launch_mode or _launch_mode_for_script(script_for_inference or script_name)
        if train_type:
            fields.setdefault("trainType", train_type)
            fields.setdefault("trainTypeEn", public_train_type(train_type, launch_mode, script_for_inference or script_name))
            fields.setdefault("trainTypeText", public_train_type_text(train_type, launch_mode, script_for_inference or script_name))
        if launch_mode:
            fields.setdefault("launchMode", launch_mode)
            fields.setdefault("isMultinode", True)
    elif protocol_type in {"job_failed", "error"}:
        fields.setdefault("status", "failed")
        fields.setdefault("action", "start")
        fields.setdefault("jobType", "train")
    hint.update({key: value for key, value in fields.items() if value is not None})
    return hint


def _train_error_response(
    script_name: str,
    message: str,
    *,
    error_reason: str,
    required_params: Optional[List[str]] = None,
    missing_params: Optional[List[str]] = None,
    recoverable: bool = False,
    **fields: Any,
) -> ToolResponse:
    protocol_type = "need_input" if required_params else "job_failed"
    return ToolResponse(content=[TextBlock(type="text", text=message)],
        metadata={
            "success": False,
            "protocol_hint": _train_protocol_hint(
                protocol_type,
                script_name,
                message,
                requiredParams=required_params or [],
                missingParams=(missing_params if missing_params is not None else required_params or []),
                errorReason=error_reason,
                errorRecoverable=recoverable,
                **fields,
            ),
        },
    )


def _train_preparing_response(script_name: str, message: str, **fields: Any) -> ToolResponse:
    """任务已提交但训练尚未确认启动。"""
    return ToolResponse(content=[TextBlock(type="text", text=message)],
        metadata={
            "success": True,
            "protocol_hint": _train_protocol_hint(
                "job_preparing",
                script_name,
                message,
                **fields,
            ),
        },
    )


def run_script_by_name_train(
    script_query: str,
    additional_args: Dict[str, Any] = None,
    list_only: bool = False,
    background: bool = None,
    env_vars: Dict[str, str] = None,
    use_docker: bool = None,
    skip_prompt: bool = False,  # 新增：是否跳过参数询问
    **kwargs
) -> ToolResponse:
    """
    根据名称、别名或描述运行脚本，支持修改参数
    
    Args:
        script_query (`str`): 脚本名称、别名或描述
        additional_args (`Dict[str, Any]`): 额外的脚本参数
        list_only (`bool`): 如果为True，只列出脚本不运行
        background (`bool`): 是否在后台运行
        env_vars (`Dict[str, str]`): 环境变量，可以包含container参数
        skip_prompt (`bool`): 是否跳过必需参数询问（用于内部递归调用）
        **kwargs: 传递给run_local_script的额外参数
        
    Returns:
        `ToolResponse`: 包含脚本执行结果的ToolResponse对象
    """
    manager = ScriptManager()
    
    if list_only or script_query.lower() in ["列表", "list", "ls", "all"]:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=manager.list_scripts()
            )
        ])
    
    script_info = manager.find_script(script_query)
    
    if not script_info:
        response_text = f"未找到脚本: {script_query}\n\n{manager.list_scripts()}"
        return _train_error_response(
            str(script_query),
            response_text,
            error_reason="script_not_found",
            recoverable=True,
        )
    
    script_name = script_info["name"]
    resource_request = _resource_request_config(script_info)
    # 获取脚本是否使用位置参数
    uses_positional_args = script_info.get("uses_positional_args", False)
    positional_param_name = script_info.get("positional_param_name", "")
    # 初始化所有变量
    params_to_update = {}
    cli_params_to_update = {}  # 添加这一行
    positional_args = []  # 新增：位置参数列表
    unknown_params = []
    launcher_options: Dict[str, Any] = {}
    manual_gpu_overrides = [
        key for key in MANUAL_GPU_ENV_KEYS
        if resource_request and key in (env_vars or {}) and str((env_vars or {}).get(key) or "").strip()
    ]
    explicit_template_provided = False
    non_qwen3_model_hints: List[str] = []
    
     # 修改：为特定脚本添加特殊处理
    if script_name == "create_command_vpn":
        # create_command_vpn 使用位置参数
        script_info["uses_positional_args"] = True
        script_info["positional_param_name"] = "schedule_time"
    if script_name == "start-server":
        # create_command_vpn 使用位置参数
        script_info["uses_positional_args"] = True
        script_info["positional_param_name"] = "action"
    # 解析参数修改请求
    if additional_args:    
        
        for param_name, param_value in additional_args.items():
            # 批量训练里“数据集/数据集名称=20260506”常被模型写成 dataset_name。
            # LoRA/全参批量训练不使用 dataset_name；日期型值应作为 DATASET_DATE 处理。
            if script_name in {"batch_train_lora", "batch_train_full", MULTINODE_SFT_SCRIPT}:
                pname = str(param_name).strip()
                pname_lower = pname.lower()
                pvalue = str(param_value).strip()
                if (
                    pname in {"数据集名称", "数据集名"}
                    or pname_lower in {"dataset_name", "data_name"}
                ) and re.fullmatch(r"20\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?", pvalue):
                    if script_name == MULTINODE_SFT_SCRIPT:
                        cli_params_to_update["dataset-date"] = pvalue
                    else:
                        params_to_update["DATASET_DATE"] = pvalue
                    continue

            pname = str(param_name).strip()
            pname_lower = pname.lower()
            normalized_launcher_param = pname_lower.replace("_", "-")
            template_managed_script = script_name in {"batch_train_lora", "batch_train_full", MULTINODE_SFT_SCRIPT, "dpo_train_launcher", MULTINODE_DPO_SCRIPT}
            if template_managed_script and (pname in TEMPLATE_PARAM_KEYS or pname_lower in {"tem", "template"}):
                explicit_template_provided = True
                if script_info.get("cli_args_only", False):
                    cli_params_to_update["template"] = str(param_value).strip()
                else:
                    params_to_update["TEM"] = str(param_value).strip()
                continue
            if template_managed_script and (pname in MODEL_HINT_KEYS or pname_lower in MODEL_HINT_KEYS):
                hint = str(param_value).strip()
                if text_mentions_non_qwen3_model(hint):
                    non_qwen3_model_hints.append(hint)
                    continue
                if "qwen3" in hint.lower() or "qwen-3" in hint.lower():
                    if script_info.get("cli_args_only", False):
                        cli_params_to_update["template"] = DEFAULT_TEMPLATE
                    else:
                        params_to_update["TEM"] = DEFAULT_TEMPLATE
                    continue
            if normalized_launcher_param in RESOURCE_LAUNCHER_ONLY_PARAMS:
                launcher_options[normalized_launcher_param] = param_value
                continue
            if pname in {"GPU数量", "GPU数", "显卡数量", "显卡数", "卡数"}:
                launcher_options["gpu-count"] = param_value
                continue
            if resource_request and (
                pname in {"显卡", "gpu", "卡号", "设备", "显卡号"}
                or pname_lower in {"gpu", "cuda_visible_devices", "gpu_ids", "gpus", "localhost_id"}
            ):
                manual_gpu_overrides.append(pname)
                continue
            if (
                "container" in pname_lower
                or "容器" in pname
                or pname_lower in {"docker", "docker_container"}
            ):
                params_to_update["container"] = str(param_value)
                continue

            # GRPO 特殊参数：容器和 GPU 作为环境变量覆盖，不进入命令行参数
            if script_name == "grpo_train":
                if (
                    "wandb" in pname_lower
                    or "w&b" in pname_lower
                    or "离线" in pname
                    or "在线" in pname
                ):
                    value_lower = str(param_value).strip().lower()
                    if "offline" in value_lower or "离线" in pname or "离线" in str(param_value):
                        cli_params_to_update["wandb_mode"] = "offline"
                    elif "online" in value_lower or "在线" in pname or "在线" in str(param_value):
                        cli_params_to_update["wandb_mode"] = "online"
                    else:
                        cli_params_to_update["wandb_mode"] = str(param_value).strip()
                    continue
                if (
                    pname in {"显卡", "gpu", "卡号", "设备", "显卡号"}
                    or pname_lower in {"gpu", "cuda_visible_devices", "gpu_ids", "gpus"}
                ):
                    params_to_update["CUDA_VISIBLE_DEVICES"] = str(param_value)
                    continue

            # 检查是否是命令行参数（针对特定脚本）
            if script_info.get("cli_args_only", False):
                # 尝试将中文参数名映射到命令行参数名
                cli_param = None
                
                # 首先检查脚本特定的映射
                if script_info.get("param_mapping"):
                    cli_param = script_info.get("param_mapping", {}).get(param_name)
                
                # 如果没有找到，检查通用映射
                if not cli_param:
                    cli_param = manager.map_chinese_to_env_var(param_name)
                
                # 如果还没有找到，但参数名直接就是支持的参数名，则直接使用
                if not cli_param and param_name in script_info.get("supported_cli_params", []):
                    cli_param = param_name
                #cli_param = script_info.get("param_mapping", {}).get(param_name)
                if cli_param:
                    # 检查该参数是否被脚本支持
                    if uses_positional_args and cli_param == positional_param_name:
                        # 如果是使用位置参数的脚本，并且是位置参数名，添加到位置参数列表
                        positional_args.append(str(param_value))
                    elif cli_param in script_info.get("supported_cli_params", []):
                        cli_params_to_update[cli_param] = _coerce_cli_value(script_info, cli_param, param_value)
                    else:
                        unknown_params.append(f"{param_name} ({cli_param})")
                else:
                    unknown_params.append(param_name)
            else:
                # 尝试将中文参数名映射到环境变量名
                env_var = manager.map_chinese_to_env_var(param_name)
                if not env_var and param_name in script_info.get("supported_params", []):
                    env_var = param_name
                if env_var:
                    # 检查该参数是否被脚本支持
                    if env_var in script_info.get("supported_params", []):
                        params_to_update[env_var] = str(param_value)
                    else:
                        unknown_params.append(f"{param_name} ({env_var})")
                else:
                    unknown_params.append(param_name)
        
        # 如果有未知参数，则提示可用参数而不运行脚本
        if unknown_params:
            available_params_text = []
            
            if script_info.get("cli_args_only", False):
                supported_cli_params = script_info.get("supported_cli_params", [])
                if supported_cli_params:
                    available_params_text.append("该脚本通过命令行参数接收输入:")
                    for cli_param in script_info.get("supported_cli_params", []):
                        # 找到对应的中文名
                        param_mapping = script_info.get("param_mapping", {})
                        chinese_names = [ch for ch, ev in param_mapping.items() if ev == cli_param]
                        if chinese_names:
                            available_params_text.append(f"  - {cli_param} ({'/'.join(chinese_names)})")
                        else:
                            available_params_text.append(f"  - {cli_param}")
                        # 如果是位置参数，特别说明
                        if uses_positional_args and cli_param == positional_param_name:
                            available_params_text[-1] += " (位置参数)"
            else:
            
                available_params = manager.get_available_params(script_name)
                if available_params:
                    available_params_text.append("该脚本通过环境变量接收输入:")
                    for param in available_params:
                        env_var = manager.map_chinese_to_env_var(param)
                        desc = manager.get_param_description(env_var)
                        available_params_text.append(f"  - {param}: {desc}")
                        #param_descriptions.append(f"  - {param}: {desc}")
                else:
                    available_params_text.append("该脚本没有可配置的参数")
            response_text = (
                f"错误！脚本 '{script_name}' 不支持以下参数: {', '.join(unknown_params)}\n\n"
                f"该脚本支持的参数:\n" + "\n".join(available_params_text) + "\n\n"
                f"请使用支持的参数重新指定，然后确认运行。"
            )
            return _train_error_response(
                script_name,
                response_text,
                error_reason="unsupported_params",
                recoverable=True,
                invalidParams=unknown_params,
            )

        if script_name in {"batch_train_lora", "batch_train_full"}:
            dataset_date = str(params_to_update.get("DATASET_DATE") or "").strip()
            dataset_dir = str(params_to_update.get("DATASET_DIR") or "").strip().rstrip("/")
            if dataset_date and dataset_dir.endswith("/dataset_batch_train"):
                params_to_update["DATASET_DIR"] = f"{dataset_dir}/{dataset_date}"
        elif script_name == MULTINODE_SFT_SCRIPT:
            dataset_date = str(cli_params_to_update.get("dataset-date") or "").strip()
            dataset_dir = str(cli_params_to_update.get("dataset-dir") or "").strip().rstrip("/")
            if dataset_date and dataset_dir.endswith("/dataset_batch_train"):
                cli_params_to_update["dataset-dir"] = f"{dataset_dir}/{dataset_date}"
    
    # 准备命令行参数
    cli_args = {}
    if script_info.get("default_cli_args"):
        cli_args = script_info.get("default_cli_args", {}).copy()
    
    # 更新命令行参数
    for cli_param, value in cli_params_to_update.items():
        cli_args[cli_param] = value
    launcher_options.update({
        key: cli_args.pop(key)
        for key in RESOURCE_LAUNCHER_ONLY_PARAMS
        if key in cli_args
    })
    if manual_gpu_overrides:
        return _train_error_response(
            script_name,
            "错误: 资源池任务不能手工指定物理 GPU 卡号，请使用 gpu-count 指定需要的 GPU 数量。",
            error_reason="manual_gpu_assignment_forbidden",
            recoverable=True,
            invalidParams=sorted(set(manual_gpu_overrides)),
        )

    normalize_container = str(
        params_to_update.get("container")
        or (env_vars or {}).get("container")
        or script_info.get("docker_container")
        or DEFAULT_DOCKER_CONTAINER
    )
    _normalize_dpo_cli_args(script_name, cli_args, normalize_container)
    _normalize_multinode_cli_args(script_name, cli_args)
    template_issue = _training_template_policy_issue(
        script_name,
        normalize_container,
        params_to_update,
        cli_args,
        explicit_template=explicit_template_provided,
        model_hint=first_nonempty(non_qwen3_model_hints),
    )
    if template_issue:
        return _train_error_response(
            script_name,
            str(template_issue["message"]),
            error_reason=str(template_issue["error_reason"]),
            recoverable=True,
            **{key: value for key, value in template_issue.items() if key not in {"message", "error_reason"}},
        )

    prefix_args = None
    shell_prefix = None
    shell_suffix = None
    log_file_override = None
    launch_stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    if script_name == "grpo_train":
        log_file_override = f"{GRPO_LOG_DIR}/{datetime.now().strftime('%Y%m%d%H%M%S')}-grpo-train.log"
    elif script_info.get("default_args", {}).get("startup_ready_marker"):
        # 每次启动必须使用独立日志，否则上一次的就绪标记会造成假成功。
        log_file_override = f"/tmp/medflow-{script_name}-launch-{launch_stamp}.log"
    
    # 检查必需参数和值合法性。skip_prompt 只跳过追问流程，不能跳过启动前校验。
    if script_info.get("cli_args_only", False):
        all_params_provided, missing_params, params_to_ask = check_required_params(script_info, cli_args)
        
        if not all_params_provided:
            if skip_prompt:
                response_text = (
                    f"错误！脚本 '{script_name}' 缺少必需参数，已阻止启动：\n"
                    + "\n".join(f"- {param}" for param in missing_params)
                    + "\n请提供真实可用的参数；例如增强训练需要真实 model_path、dataset_dir、dataset_name。"
                )
                return _train_error_response(
                    script_name,
                    response_text,
                    error_reason="missing_required_params",
                    required_params=script_info.get("required_cli_params", []),
                    missing_params=missing_params,
                    recoverable=True,
                )

            # 构建询问消息
            prompt_text = f"错误！脚本 '{script_name}' 需要以下必需参数，但未提供：\n\n"
            
            for param_name, param_info in params_to_ask.items():
                prompt_text += f"- {param_info['description']}\n"
            
            
            
            # 序列化当前已收集的参数，以便后续使用
            current_args = {
                "additional_args": additional_args or {},
                "cli_params_to_update": cli_params_to_update,
                "params_to_update": params_to_update,
                "positional_args": positional_args
            }
            prompt_text += f"当前参数: {json.dumps(current_args, ensure_ascii=False)}\n"
            
            return _train_error_response(
                script_name,
                prompt_text,
                error_reason="missing_required_params",
                required_params=script_info.get("required_cli_params", []),
                missing_params=missing_params,
                recoverable=True,
                currentArgs=current_args,
            )

        invalid_params = validate_cli_param_values(script_info, cli_args)
        if invalid_params:
            response_text = (
                f"错误！脚本 '{script_name}' 的必需参数存在无效值或占位值，已阻止启动：\n"
                + "\n".join(f"- {item}" for item in invalid_params)
                + "\n请提供真实可用的参数；例如增强训练需要真实 model_path、dataset_dir、dataset_name。"
            )
            return _train_error_response(
                script_name,
                response_text,
                error_reason="invalid_required_params",
                required_params=script_info.get("required_cli_params", []),
                missing_params=[],
                recoverable=True,
                invalidParams=invalid_params,
            )
    
    # 获取脚本路径
    script_path = script_info.get("path")
    
    # 检查是否需要Docker执行
    requires_docker = script_info.get("requires_docker", False)
    use_docker_flag = use_docker or (use_docker is None and requires_docker)
    
    # 如果不使用Docker，检查主机上的脚本路径
    if not use_docker_flag:
        use_docker_flag=True
        #if not os.path.exists(script_path):
            # 尝试从当前目录查找
        #    current_path = os.path.join(os.getcwd(), script_path)
        #    if not os.path.exists(current_path):
        #        return ToolResponse(content=[
        #            TextBlock(
        #                type="text",
        #                text=f"错误: 脚本文件不存在: {script_path}\n请检查路径是否正确。"
        #            )
        #        ])
        #    script_path = current_path
    
    # 检查是否是长时间运行任务
    long_running = script_info.get("long_running", False)
    if not script_info.get("supports_background"):
        return _train_error_response(
            script_name,
            f"错误: 训练脚本 '{script_name}' 未配置后台运行能力，已阻止前台启动。",
            error_reason="background_not_supported",
        )
    background = True
    
    # 准备环境变量
    final_env_vars = {}
    
    # 添加脚本默认环境变量
    if script_info.get("default_env"):
        final_env_vars.update(script_info["default_env"])
    
    # 添加用户提供的环境变量
    if env_vars:
        final_env_vars.update(env_vars)
    # 更新环境变量参数
    for env_var, value in params_to_update.items():
        final_env_vars[env_var] = value
    if not resource_request:
        try:
            final_env_vars = require_assigned_gpu_env(final_env_vars)
        except ValueError as exc:
            return _train_error_response(
                script_name,
                f"错误: {exc}",
                error_reason="gpu_assignment_required",
                recoverable=True,
            )
        
    # 提取用户指定的容器名称（如果有）
    user_container = None
    if "container" in final_env_vars:
        user_container = final_env_vars.pop("container")  # 从环境变量中移除，因为它不是真正的环境变量
    for container_key in ("container_name", "docker_container"):
        container_value = final_env_vars.pop(container_key, None)
        if container_value and not user_container:
            user_container = str(container_value).strip()
    # 如果有参数需要更新，则更新环境变量
    #if additional_args:
        # 更新参数
       # for param_name, param_value in additional_args.items():
      #      env_var = manager.map_chinese_to_env_var(param_name)
       #     if env_var and env_var in script_info.get("supported_params", []):
        #        final_env_vars[env_var] = str(param_value)
    
    # 如果有位置参数，需要特殊处理
    #if uses_positional_args and positional_args:
        # 对于使用位置参数的脚本，我们需要确保正确的参数传递方式
        # 我们将在后面的执行中处理
        #pass
    # 合并默认参数和额外参数
    args = script_info.get("default_args", {}).copy()
    
    # 移除后台运行器参数，避免它们被当成 --xxx 传给训练程序。
    args.pop("background", None)
    args.pop("log_file", None)
    startup_ready_marker = args.pop("startup_ready_marker", None)
    startup_ready_timeout = args.pop("startup_ready_timeout", None)
    startup_poll_interval = args.pop("startup_poll_interval", None)
    
    # 将命令行参数合并到args中
    if script_name == "grpo_train":
        # VERL 的参数通过 shell 脚本 $@ 透传，采用 key=value 的位置参数，而非 --key value
        model_path = str(cli_args.pop("model_path", "")).strip()
        train_files = str(cli_args.pop("train_files", "")).strip()
        val_files = str(cli_args.pop("val_files", "")).strip()
        wandb_mode = str(cli_args.pop("wandb_mode", "online")).strip().lower()
        if wandb_mode not in {"online", "offline"}:
            return _train_error_response(
                script_name,
                f"错误：grpo_train 的 wandb_mode 只支持 online 或 offline，当前值为 {wandb_mode!r}。",
                error_reason="invalid_wandb_mode",
                required_params=["wandb_mode"],
                recoverable=True,
                invalidParams=[wandb_mode],
            )
        final_env_vars["WANDB_DIR"] = GRPO_SOURCE_WORKDIR
        final_env_vars["WANDB_CONSOLE"] = "redirect"
        if wandb_mode == "offline":
            final_env_vars["WANDB_MODE"] = "offline"
        else:
            final_env_vars.pop("WANDB_MODE", None)
        launcher_options["_grpo_preflight"] = {
            "model_path": model_path,
            "train_files": train_files,
            "val_files": val_files,
        }
        positional_args.extend(
            [
                f"actor_rollout_ref.model.path={model_path}",
                f"data.train_files={train_files}",
                f"data.val_files={val_files}",
            ]
        )

    _normalize_multinode_cli_args(script_name, cli_args)
    args.update(cli_args)
    
    # 准备传递给run_local_script的kwargs
    run_kwargs = {
        "script_path": script_path,
        "script_args": args,
        "positional_args": positional_args,  # 添加位置参数
        "env_vars": final_env_vars,
        **kwargs
    }
    run_kwargs["script_name"] = script_name
    run_kwargs["train_type"] = infer_train_type_from_name(script_name)
    if prefix_args:
        run_kwargs["prefix_args"] = prefix_args
    if shell_prefix:
        run_kwargs["shell_prefix"] = shell_prefix
    if shell_suffix:
        run_kwargs["shell_suffix"] = shell_suffix
    if log_file_override:
        run_kwargs["log_file"] = log_file_override
    if startup_ready_marker:
        run_kwargs["startup_ready_marker"] = startup_ready_marker
    if startup_ready_timeout is not None:
        run_kwargs["startup_ready_timeout"] = startup_ready_timeout
    if startup_poll_interval is not None:
        run_kwargs["startup_poll_interval"] = startup_poll_interval
    
    # 对于长时间运行的任务，设置较大的超时时间或不设超时
    if long_running and "timeout" not in kwargs:
        run_kwargs["timeout"] = None  # 不设超时
    
    # 如果脚本要求Docker或用户指定使用Docker
    if use_docker_flag:
        # 准备Docker相关参数 - 使用用户指定的容器或默认容器
        docker_kwargs = {
            "docker_container": user_container or script_info.get("docker_container"),
            "docker_image": script_info.get("docker_image"),
            "docker_volume_mounts": script_info.get("docker_volume_mounts"),
            "docker_working_dir": script_info.get("docker_working_dir", 
                                               script_info.get("default_args", {}).get("working_dir")),
            "docker_executable": script_info.get("docker_executable", "python"),
        }

        if not docker_kwargs["docker_container"]:
            response_text = (
                f"错误: 需要指定Docker容器才能运行脚本\n"
                f"脚本: {script_info['description']}\n"
                f"请在 runtime.env 设置 AGENT3_DEFAULT_DOCKER_CONTAINER/MEDFLOW_GRPO_DOCKER_CONTAINER，或在调用参数中传入 container"
            )
            return _train_error_response(
                script_name,
                response_text,
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )
        container_issue = _validate_training_container_preflight(str(docker_kwargs["docker_container"]))
        if container_issue:
            return _train_error_response(
                script_name,
                str(container_issue["message"]),
                error_reason=str(container_issue["error_reason"]),
                required_params=container_issue.get("required_params"),
                recoverable=True,
                **{
                    key: value
                    for key, value in container_issue.items()
                    if key not in {"message", "error_reason", "required_params"}
                },
            )

        if script_name in {MULTINODE_SFT_SCRIPT, MULTINODE_DPO_SCRIPT} and docker_kwargs.get("docker_container"):
            selected_entry, entry_error = _select_multinode_docker_entrypoint(
                str(docker_kwargs["docker_container"]),
                script_name,
            )
            if not selected_entry:
                return _train_error_response(
                    script_name,
                    entry_error or "ERROR: multi-node training entrypoint was not found in the container.",
                    error_reason="multinode_entrypoint_not_found",
                    recoverable=True,
                    container=str(docker_kwargs["docker_container"]),
                    binaryPath=MULTINODE_BINARY_PATHS.get(script_name),
                    sourcePath=MULTINODE_SOURCE_PATHS.get(script_name),
                )
            docker_kwargs["docker_working_dir"] = selected_entry["docker_working_dir"]
            docker_kwargs["docker_executable"] = selected_entry["docker_executable"]
            script_path = selected_entry["script_path"]
            args = _merge_fixed_cli_args(args, selected_entry.get("fixed_cli_args") or {})
            run_kwargs["script_path"] = script_path
            run_kwargs["script_args"] = args

        if script_name == "dpo_train_launcher" and docker_kwargs.get("docker_container"):
            container = docker_kwargs["docker_container"]
            try:
                check_cmd = [
                    "docker",
                    "exec",
                    container,
                    "sh",
                    "-c",
                    "test -f /usr/local/insinfersystem/dpo_train_launcher",
                ]
                exists = subprocess.run(check_cmd, capture_output=True, text=True).returncode == 0
            except Exception:
                exists = True
            if not exists:
                fallback_path = "/home/workspace/llamafactory/dpo_train_launcher.py"
                # 优先使用已确认的 fallback 路径，不再因检测失败而中止
                try:
                    test_cmd = [
                        "docker",
                        "exec",
                        container,
                        "sh",
                        "-c",
                        f"test -f {shlex.quote(fallback_path)}",
                    ]
                    subprocess.run(test_cmd, capture_output=True, text=True)
                except Exception:
                    pass

                docker_kwargs["docker_working_dir"] = "/home/workspace/llamafactory"
                # 选择容器内可用的 python 可执行文件
                py_candidates = ["/opt/conda/bin/python", "python3", "python"]
                chosen_py = None
                for py in py_candidates:
                    try:
                        check_py = ["docker", "exec", container, "sh", "-c", f"command -v {py} >/dev/null 2>&1"]
                        if subprocess.run(check_py, capture_output=True, text=True).returncode == 0:
                            chosen_py = py
                            break
                    except Exception:
                        continue
                if not chosen_py:
                    return _train_error_response(
                        script_name,
                        "错误: 容器内未找到可用的 Python 可执行文件（/opt/conda/bin/python、python3、python）。",
                        error_reason="python_not_found_in_container",
                        container=container,
                    )
                docker_kwargs["docker_executable"] = chosen_py
                script_path = fallback_path
                run_kwargs["script_path"] = script_path

        
        if script_name in ("batch_train_lora", "batch_train_full") and docker_kwargs.get("docker_container"):
            container = docker_kwargs["docker_container"]
            ins_path = f"/usr/local/insinfersystem/{script_name}"
            try:
                check_cmd = [
                    "docker",
                    "exec",
                    container,
                    "sh",
                    "-c",
                    f"test -f {shlex.quote(ins_path)}",
                ]
                exists = subprocess.run(check_cmd, capture_output=True, text=True).returncode == 0
            except Exception:
                exists = True
            if exists:
                docker_kwargs["docker_working_dir"] = "/usr/local/insinfersystem"
                docker_kwargs["docker_executable"] = ins_path
                script_path = script_name
                run_kwargs["script_path"] = script_path
            else:
                fallback_path = f"/home/workspace/llamafactory/{script_name}.py"
                # 优先使用已确认的 fallback 路径，不再因检测失败而中止
                try:
                    test_cmd = [
                        "docker",
                        "exec",
                        container,
                        "sh",
                        "-c",
                        f"test -f {shlex.quote(fallback_path)}",
                    ]
                    subprocess.run(test_cmd, capture_output=True, text=True)
                except Exception:
                    pass

                docker_kwargs["docker_working_dir"] = "/home/workspace/llamafactory"
                # 选择容器内可用的 python 可执行文件
                py_candidates = ["/opt/conda/bin/python", "python3", "python"]
                chosen_py = None
                for py in py_candidates:
                    try:
                        check_py = ["docker", "exec", container, "sh", "-c", f"command -v {py} >/dev/null 2>&1"]
                        if subprocess.run(check_py, capture_output=True, text=True).returncode == 0:
                            chosen_py = py
                            break
                    except Exception:
                        continue
                if not chosen_py:
                    return _train_error_response(
                        script_name,
                        "错误: 容器内未找到可用的 Python 可执行文件（/opt/conda/bin/python、python3、python）。",
                        error_reason="python_not_found_in_container",
                        container=container,
                    )
                docker_kwargs["docker_executable"] = chosen_py
                script_path = fallback_path
                run_kwargs["script_path"] = script_path
        
        # 合并到run_kwargs中
        if script_name == "grpo_train":
            selected_entry, entry_error = _select_grpo_docker_entrypoint(str(docker_kwargs["docker_container"]))
            if not selected_entry:
                return _train_error_response(
                    script_name,
                    entry_error or "ERROR: GRPO entrypoint was not found in the container.",
                    error_reason="grpo_entrypoint_not_found",
                    recoverable=True,
                    container=str(docker_kwargs["docker_container"]),
                    binaryPath=GRPO_BINARY_WRAPPER,
                    sourcePath=GRPO_SOURCE_SCRIPT_ABS,
                )
            docker_kwargs["docker_working_dir"] = selected_entry["docker_working_dir"]
            docker_kwargs["docker_executable"] = selected_entry["docker_executable"]
            script_path = selected_entry["script_path"]
            final_env_vars["WANDB_DIR"] = selected_entry["wandb_dir"]
            final_env_vars["WANDB_CONSOLE"] = "redirect"
            run_kwargs["script_path"] = script_path
            run_kwargs["env_vars"] = final_env_vars
            if log_file_override:
                run_kwargs["shell_prefix"] = _grpo_shell_prefix(log_file_override, selected_entry["wandb_dir"])

        for k, v in docker_kwargs.items():
            if v is not None:
                run_kwargs[k] = v

        preflight_args = dict(args)
        if script_name == "grpo_train" and "_grpo_preflight" in launcher_options:
            preflight_args["_grpo_preflight"] = launcher_options["_grpo_preflight"]
        validation_issue = run_training_preflight(
            script_name,
            str(docker_kwargs["docker_container"]),
            final_env_vars,
            preflight_args,
            script_info,
            launcher_options,
            resource_request,
            container_prechecked=True,
        )
        if validation_issue:
            return _train_error_response(
                script_name,
                str(validation_issue["message"]),
                error_reason=str(validation_issue["error_reason"]),
                required_params=validation_issue.get("required_params"),
                recoverable=True,
                **{
                    key: value
                    for key, value in validation_issue.items()
                    if key not in {"message", "error_reason", "required_params"}
                },
            )
        log_file = run_kwargs.get("log_file") or script_info.get("default_args", {}).get("log_file")
        if log_file:
            run_kwargs["log_file"] = log_file
        reservation_id = None
        if resource_request:
            try:
                reservation_id = _prepare_resource_allocation(script_info, args, final_env_vars, launcher_options)
                final_env_vars = require_assigned_gpu_env(final_env_vars)
                run_kwargs["env_vars"] = final_env_vars
            except Exception as exc:
                _release_resource_allocation(reservation_id, final_env_vars)
                return _train_error_response(
                    script_name,
                    f"错误: GPU 资源分配失败: {exc}",
                    error_reason="resource_allocation_failed",
                    recoverable=True,
                )
        try:
            return _finish_resource_launch(
                run_docker_in_background(**run_kwargs),
                reservation_id,
                final_env_vars,
            )
        except Exception:
            _release_resource_allocation(reservation_id, final_env_vars)
            raise
    
    # 对于本地脚本的后台运行，需要移除Docker相关参数
    local_kwargs = {k: v for k, v in run_kwargs.items() 
                   if not k.startswith('docker_') and k not in ['docker_container', 'docker_image']}
    reservation_id = None
    if resource_request:
        try:
            reservation_id = _prepare_resource_allocation(script_info, args, final_env_vars, launcher_options)
            final_env_vars = require_assigned_gpu_env(final_env_vars)
            local_kwargs["env_vars"] = final_env_vars
        except Exception as exc:
            _release_resource_allocation(reservation_id, final_env_vars)
            return _train_error_response(
                script_name,
                f"错误: GPU 资源分配失败: {exc}",
                error_reason="resource_allocation_failed",
                recoverable=True,
            )
    try:
        return _finish_resource_launch(
            run_script_in_background(**local_kwargs),
            reservation_id,
            final_env_vars,
        )
    except Exception:
        _release_resource_allocation(reservation_id, final_env_vars)
        raise



def process_parameter_response(script_query: str, user_response: str, previous_args: Dict[str, Any] = None) -> ToolResponse:
    """
    处理用户对必需参数的响应
    
    Args:
        script_query: 脚本查询名称
        user_response: 用户提供的参数响应
        previous_args: 之前已经收集的参数
        
    Returns:
        ToolResponse: 执行结果
    """
    # 解析用户提供的参数
    parsed_args = {}
    lines = user_response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or '###' in line:
            continue
            
        # 尝试解析 "参数名: 参数值" 格式
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                # 去除可能的注释
                if '#' in value:
                    value = value.split('#')[0].strip()
                parsed_args[key] = value
    
    # 如果没有解析到参数，尝试其他格式
    if not parsed_args and '=' in user_response:
        for line in lines:
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    parsed_args[key] = value
    
    # 合并之前的参数
    final_args = previous_args or {}
    if parsed_args:
        # 合并参数
        for key, value in parsed_args.items():
            final_args[key] = value
    
    # 重新运行脚本，这次跳过询问
    return run_script_by_name_train(
        script_query=script_query,
        additional_args=final_args,
        skip_prompt=True  # 跳过询问，直接执行
    )


def validate_docker_container(
    container_name: str,
    check_script_path: str = None,
    working_dir: str = None
) -> Dict[str, Any]:
    """
    验证Docker容器状态和路径
    """
    result = {
        "container_exists": False,
        "container_running": False,
        "script_exists": False,
        "errors": []
    }
    
    try:
        # 检查容器是否存在
        check_cmd = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        process = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            result["errors"].append(f"Docker命令失败: {process.stderr}")
            return result
        
        containers = [c.strip() for c in process.stdout.split('\n') if c.strip()]
        
        if container_name not in containers:
            result["errors"].append(f"容器 {container_name} 不存在")
            return result
        
        result["container_exists"] = True
        
        # 检查容器是否正在运行
        check_cmd = ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        process = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            running_containers = [c.strip() for c in process.stdout.split('\n') if c.strip()]
            result["container_running"] = container_name in running_containers
        
        # 检查脚本路径
        if check_script_path:
            # 构建容器内的绝对路径
            if working_dir and not os.path.isabs(check_script_path):
                abs_path = os.path.join(working_dir, check_script_path)
            else:
                abs_path = check_script_path
            
            # 检查文件是否存在
            check_cmd = ["docker", "exec", container_name, "test", "-f", abs_path]
            process = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                result["script_exists"] = True
                result["script_path"] = abs_path
            else:
                # 尝试查找文件
                find_cmd = ["docker", "exec", container_name, "find", working_dir or "/", 
                          "-name", os.path.basename(check_script_path), "-type", "f"]
                process = subprocess.run(find_cmd, capture_output=True, text=True, timeout=5)
                
                if process.returncode == 0 and process.stdout.strip():
                    found_path = process.stdout.strip().split('\n')[0]
                    result["script_exists"] = True
                    result["script_path"] = found_path
                    result["warnings"] = [f"使用找到的路径: {found_path}"]
                else:
                    result["errors"].append(f"在容器中未找到脚本: {check_script_path}")
        
    except subprocess.TimeoutExpired:
        result["errors"].append("Docker命令执行超时")
    except Exception as e:
        result["errors"].append(f"Docker验证异常: {str(e)}")
    
    return result


def run_script_in_docker(
    script_path: str,
    script_args: Dict[str, Any] = None,
    positional_args: List[str] = None,  # 新增：位置参数
    env_vars: Dict[str, str] = None,
    timeout: int = None,
    capture_output: bool = True,
    working_dir: str = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    validate_container: bool = True,
    **kwargs
) -> ToolResponse:
    """
    在Docker容器中运行Python脚本，支持通过环境变量传递参数
    """
    
    try:
        # 验证容器状态
        if validate_container and docker_container:
            validation = validate_docker_container(
                container_name=docker_container,
                check_script_path=script_path,
                working_dir=docker_working_dir
            )
            
            if validation.get("errors"):
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"Docker容器验证失败:\n" + "\n".join(validation["errors"])
                    )
                ])
            
            # 更新脚本路径为找到的路径
            if validation.get("script_path"):
                script_path = validation["script_path"]
            
            if not validation.get("container_running"):
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"容器 {docker_container} 存在但未运行，请先启动容器"
                    )
                ])
        
        # 构建Docker命令
        if docker_container:
            # 在运行中的容器内执行
            command = ["docker", "exec"]
        elif docker_image:
            # 启动新的容器执行
            command = ["docker", "run", "--rm"]
        else:
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text="错误: 需要指定docker_container或docker_image"
                )
            ])
        
        # 添加环境变量
        if env_vars:
            for key, value in env_vars.items():
                if value:  # 只添加非空值
                    command.extend(["-e", f"{key}={value}"])

        if docker_container and env_vars and env_vars.get("MEDFLOW_TRAINING_RESERVATION_ID"):
            try:
                _validate_assigned_gpus_available(docker_container, env_vars)
            except Exception as exc:
                return _train_error_response(
                    script_name or script_path,
                    f"错误: {exc}。训练未启动；系统不支持强制清理显存并继续执行，请先释放占用进程或调整资源池 GPU 配置后重试。",
                    error_reason="assigned_gpu_preflight_failed",
                    recoverable=True,
                    container=docker_container,
                    assignedGpus=env_vars.get("MEDFLOW_ASSIGNED_GPUS") or env_vars.get("CUDA_VISIBLE_DEVICES"),
                )

        # 添加卷挂载
        if docker_volume_mounts:
            for host_path, container_path in docker_volume_mounts.items():
                if os.path.exists(host_path):
                    command.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}"])
        
        # 设置工作目录
        if docker_working_dir:
            command.extend(["-w", docker_working_dir])
        
        # 指定容器/镜像
        if docker_container:
            command.append(docker_container)
        elif docker_image:
            command.append(docker_image)
        #if docker_working_dir:
            #shell_command_parts.append(f"cd {docker_working_dir}")
        # 添加执行命令
        #command.append(docker_executable)
        #command.append(script_path)
        if docker_executable and script_path and docker_executable == script_path:
            # 如果两者相同，只保留一个
            command.append(script_path)  # 这里使用脚本路径作为可执行文件
        else:
            # 添加Python命令和脚本
            if docker_executable:
                command.append(docker_executable)
            command.append(script_path)  # 这里使用容器内的路径
        
        
        if positional_args:
            for arg in positional_args:
                command.append(str(arg))
        
        # 添加脚本参数
        if script_args:
            for key, value in script_args.items():
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                    continue
                else:
                    command.extend([f"--{key}", str(value)])
        
        # 打印调试信息
        print(f"Docker命令: {' '.join(command)}")
        
        # 执行Docker命令
        if timeout:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
        else:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True
            )
        
        # 构建结果消息
        result_text = []
        
        if process.stdout and capture_output:
            result_text.append(f"标准输出:\n{process.stdout}")
        
        if process.stderr and capture_output:
            result_text.append(f"标准错误:\n{process.stderr}")
        
        result_text.append(f"返回码: {process.returncode}")
        result_text.append(f"######请检查对应的log文件######")
        
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="\n\n".join(result_text)
            )
        ])
        
    except subprocess.TimeoutExpired:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: Docker脚本执行超时 (超过 {timeout} 秒)"
            )
        ])
    
    except FileNotFoundError:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="错误: Docker命令未找到，请确保Docker已安装并可用"
            )
        ])
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: Docker执行脚本时发生异常: {str(e)}\n\n详细错误:\n{error_trace}"
            )
        ])


def run_docker_in_background(
    script_path: str,
    script_args: Dict[str, Any] = None,
    positional_args: List[str] = None,  # 新增：位置参数
    env_vars: Dict[str, str] = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    prefix_args: List[str] = None,
    script_name: str = None,
    train_type: str = None,
    log_file: str = None,
    pid_file: str = None,
    shell_prefix: str = None,
    shell_suffix: str = None,
    check_startup: bool = True,  # 新增：是否检查启动状态
    startup_check_delay: int = 5,  # 新增：启动检查延迟时间（秒）
    startup_ready_marker: str = None,
    startup_ready_timeout: int = 15,
    startup_poll_interval: float = 1.0,
    startup_stability_seconds: float = 2.0,
    **kwargs
) -> ToolResponse:
    """
    在Docker容器中后台运行脚本，支持通过环境变量传递参数
    """
    try:
        # 如果没有指定容器，尝试从环境变量中获取
        if env_vars and any(key in env_vars for key in ("container", "container_name", "docker_container")):
            env_vars_copy = env_vars.copy()
            for container_key in ("container", "container_name", "docker_container"):
                container_value = env_vars_copy.pop(container_key, None)
                if container_value and not docker_container:
                    docker_container = str(container_value).strip()
            env_vars = env_vars_copy
        
        if not docker_container and not docker_image:
            return _train_error_response(
                script_name or script_path,
                "错误: 需要指定docker_container或docker_image",
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )
        
        if not train_type:
            train_type = infer_train_type_from_name(script_name or script_path or docker_executable)

        # 构建Docker后台运行命令
        if docker_container:
            # 在运行中的容器内执行
            command = ["docker", "exec"]
        elif docker_image:
            # 启动新的容器并后台运行
            command = ["docker", "run", "-d", "--rm"]
        else:
            return _train_error_response(
                script_name or script_path,
                "错误: 需要指定docker_container或docker_image",
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )
        
        # 添加环境变量
        if env_vars:
            for key, value in env_vars.items():
                if value:  # 只添加非空值
                    command.extend(["-e", f"{key}={value}"])

        if docker_container and env_vars and env_vars.get("MEDFLOW_TRAINING_RESERVATION_ID"):
            try:
                _validate_assigned_gpus_available(docker_container, env_vars)
            except Exception as exc:
                return _train_error_response(
                    script_name or script_path,
                    f"错误: {exc}。训练未启动；系统不支持强制清理显存并继续执行，请先释放占用进程或调整资源池 GPU 配置后重试。",
                    error_reason="assigned_gpu_preflight_failed",
                    recoverable=True,
                    container=docker_container,
                    assignedGpus=env_vars.get("MEDFLOW_ASSIGNED_GPUS") or env_vars.get("CUDA_VISIBLE_DEVICES"),
                )

        # 添加卷挂载
        if docker_volume_mounts:
            for host_path, container_path in docker_volume_mounts.items():
                if os.path.exists(host_path):
                    command.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}"])
        
        # 指定容器/镜像
        if docker_container:
            command.append(docker_container)
            shell_command = build_background_shell_command(
                script_path=script_path,
                docker_executable=docker_executable,
                prefix_args=prefix_args,
                positional_args=positional_args,
                script_args=script_args,
                docker_working_dir=docker_working_dir,
                log_file=log_file,
                shell_prefix=shell_prefix,
                shell_suffix=shell_suffix,
            )
            command.extend(["sh", "-c", shell_command])
        
        elif docker_image:
            command.append(docker_image)
            # 对于镜像，直接添加可执行文件和参数
            #if docker_executable == script_path:
            if  script_path in docker_executable:
                # 如果可执行文件和脚本路径相同，只添加脚本路径
                command.append(script_path)
            else:
                command.append(docker_executable)
                command.append(script_path)
            # 添加位置参数
            if positional_args:
                for arg in positional_args:
                    # 格式化时间参数
                    formatted_arg = arg
                    if ":" in arg and arg.count(":") == 2:
                        if not arg.startswith("{") and not arg.endswith("}"):
                            formatted_arg = f"{{{arg}}}"
                    command.append(formatted_arg)
            # 添加脚本参数
            if script_args and not positional_args:
                for key, value in script_args.items():
                    if isinstance(value, bool):
                        if value:
                            command.extend([f"--{key}"])
                    elif isinstance(value, (int, float, str)):
                        # 格式化schedule_time参数
                        if key == "schedule_time" and ":" in str(value) and str(value).count(":") == 2:
                            if not str(value).startswith("{") and not str(value).endswith("}"):
                                formatted_value = f"{{{value}}}"
                                command.extend([f"--{key}", formatted_value])
                            else:
                                command.extend([f"--{key}", str(value)])
                        else:
                            command.extend([f"--{key}", str(value)])
                    elif isinstance(value, list):
                        for item in value:
                            command.extend([f"--{key}", str(item)])
                    elif value is None:
                        continue
                    else:
                        command.extend([f"--{key}", str(value)])
        
        # 打印调试信息
        logger.info(f"Docker后台命令: {' '.join(command)}")
        _record_background_task(
            task_type="train",
            container=docker_container,
            script_name=script_name,
            script_path=script_path,
            train_type=train_type,
            command=command,
            env_vars=env_vars,
            script_args=script_args,
            positional_args=positional_args,
            status="starting",
        )

        if docker_container:
            logger.info(f"执行命令: {' '.join(command)}")
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            stdout = process.stdout or ""
            stderr = process.stderr or ""
            if process.returncode != 0:
                error_msg = f"脚本在Docker容器中启动失败，返回码: {process.returncode}\n"
                if stdout:
                    error_msg += f"标准输出:\n{stdout}\n"
                if stderr:
                    error_msg += f"错误信息:\n{stderr}\n"
                error_msg += "######请检查对应的log文件######"
                return _train_error_response(
                    script_name or script_path,
                    error_msg,
                    error_reason="docker_start_failed",
                    container=docker_container,
                    returnCode=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            pid = parse_pid_from_output(stdout) or parse_pid_from_output(stderr)
            if not pid:
                response_text = f"脚本已启动，但未能解析PID。\n标准输出:\n{stdout}\n标准错误:\n{stderr}"
                return _train_error_response(
                    script_name or script_path,
                    response_text,
                    error_reason="pid_parse_failed",
                    recoverable=True,
                    container=docker_container,
                    stdout=stdout,
                    stderr=stderr,
                )

            launch_preparing = False
            if check_startup:
                startup_log = ""
                if startup_ready_marker and log_file:
                    deadline = time.monotonic() + max(1, int(startup_ready_timeout))
                    poll_interval = max(0.1, float(startup_poll_interval))
                    while True:
                        if not _check_pid_in_container(docker_container, pid):
                            startup_log = _read_container_log_tail(docker_container, log_file)
                            response_text = (
                                f"训练未启动成功，准备阶段进程已退出，PID：{pid}\n"
                                f"标准输出:\n{stdout}\n标准错误:\n{stderr}"
                            )
                            if startup_log:
                                response_text += f"\n启动日志({log_file})尾部:\n{startup_log}"
                            return _train_error_response(
                                script_name or script_path,
                                response_text,
                                error_reason="startup_process_exited",
                                container=docker_container,
                                pid=str(pid),
                                stdout=stdout,
                                stderr=stderr,
                                launchLogPath=log_file,
                                launchLogTail=startup_log,
                            )

                        startup_log = _read_container_log_tail(docker_container, log_file)
                        if startup_ready_marker in startup_log:
                            time.sleep(max(0.0, float(startup_stability_seconds)))
                            if _check_pid_in_container(docker_container, pid):
                                break
                            continue
                        if time.monotonic() >= deadline:
                            launch_preparing = True
                            break
                        time.sleep(poll_interval)
                elif startup_check_delay > 0:
                    time.sleep(startup_check_delay)
                    if not _check_pid_in_container(docker_container, pid):
                        startup_log = _read_container_log_tail(docker_container, log_file)
                        response_text = (
                            f"脚本在Docker容器中启动失败或已退出，PID未存活：{pid}\n"
                            f"标准输出:\n{stdout}\n标准错误:\n{stderr}"
                        )
                        if startup_log:
                            response_text += f"\n启动日志({log_file})尾部:\n{startup_log}"
                        return _train_error_response(
                            script_name or script_path,
                            response_text,
                            error_reason="pid_not_alive",
                            container=docker_container,
                            pid=str(pid),
                            stdout=stdout,
                            stderr=stderr,
                            launchLogPath=log_file,
                            launchLogTail=startup_log,
                        )

            if pid_file:
                pid_dir = os.path.dirname(pid_file)
                if pid_dir and not os.path.exists(pid_dir):
                    os.makedirs(pid_dir, exist_ok=True)
                with open(pid_file, "w", encoding="utf-8") as f:
                    f.write(str(pid))

            _record_train_pid(
                container=docker_container,
                pid=str(pid),
                script_name=script_name,
                script_path=script_path,
                train_type=train_type,
                docker_working_dir=docker_working_dir,
                command=" ".join(command),
                env_vars=env_vars,
                script_args=script_args,
                positional_args=positional_args,
                launch_status="preparing" if launch_preparing else "started",
                launch_log_file=log_file,
                startup_ready_marker=startup_ready_marker,
            )
            _record_background_task(
                task_type="train",
                container=docker_container,
                pid=str(pid),
                script_name=script_name,
                script_path=script_path,
                train_type=train_type,
                command=command,
                env_vars=env_vars,
                script_args=script_args,
                positional_args=positional_args,
                status="preparing" if launch_preparing else "started",
            )

            if launch_preparing:
                response_text = (
                    f"训练任务已提交，正在准备数据集和模型，尚未确认训练启动。\n"
                    f"脚本: {os.path.basename(script_path)}\n"
                    f"容器: {docker_container}\n"
                    f"PID: {pid}\n"
                    f"准备时间可能较长，请稍后使用该 PID 查询状态。"
                )
                return _train_preparing_response(
                    script_name or script_path,
                    response_text,
                    container=docker_container,
                    pid=str(pid),
                    script=os.path.basename(script_path),
                    command=" ".join(command),
                    scriptArgs=script_args or {},
                    envKeys=sorted((env_vars or {}).keys()),
                    launchLogPath=log_file,
                )

            response_text = (
                f"脚本已在Docker容器中后台启动运行\n"
                f"脚本: {os.path.basename(script_path)}\n"
                f"容器: {docker_container}\n"
                f"PID: {pid}\n"
                f"请稍后使用该PID或容器名称查询训练状态。"
            )
            return ToolResponse(content=[
                TextBlock(type="text", text=response_text)
            ], metadata={
                "success": True,
                "protocol_hint": _train_protocol_hint(
                    "job_started",
                    script_name,
                    response_text,
                    container=docker_container,
                    pid=str(pid),
                    script=os.path.basename(script_path),
                    command=" ".join(command),
                    scriptArgs=script_args or {},
                    envKeys=sorted((env_vars or {}).keys()),
                ),
            })

        # docker_image 分支保持原有行为
        def run_docker_process():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                time.sleep(20)

                returncode = process.poll()
                stdout, stderr = "", ""
                if returncode is not None:
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except Exception:
                        pass
                else:
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        stdout = ""
                        stderr = ""

                container_id = stdout.strip() if docker_image else docker_container

                process_info = {
                    "container_id": container_id,
                    "pid": process.pid,
                    "command": " ".join(command),
                    "log_file": log_file,
                    "pid_file": pid_file,
                    "start_time": time.time(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode or 0,
                }

                if pid_file:
                    pid_dir = os.path.dirname(pid_file)
                    if pid_dir and not os.path.exists(pid_dir):
                        os.makedirs(pid_dir, exist_ok=True)
                    with open(pid_file, "w", encoding="utf-8") as f:
                        f.write(str(container_id))

                if returncode in (None, 0):
                    _record_background_task(
                        task_type="train",
                        container=container_id,
                        script_name=script_name,
                        script_path=script_path,
                        train_type=train_type,
                        command=command,
                        env_vars=env_vars,
                        script_args=script_args,
                        positional_args=positional_args,
                    )

                return process_info
            except Exception as e:
                import traceback
                return {"error": str(e), "traceback": traceback.format_exc()}

        process_info = run_docker_process()

        if check_startup and startup_check_delay > 0:
            if "error" in process_info:
                response_text = f"启动Docker进程失败: {process_info['error']}\n\n详细错误:\n{process_info.get('traceback', '无')}"
                return _train_error_response(
                    script_name or script_path,
                    response_text,
                    error_reason="docker_process_failed",
                    container=docker_container or docker_image,
                    traceback=process_info.get("traceback"),
                )

            returncode = process_info.get("returncode", 0)
            stdout = process_info.get("stdout", "")
            stderr = process_info.get("stderr", "")

            if returncode != 0 and returncode is not None:
                error_msg = f"脚本在Docker容器中启动失败，返回码: {returncode}\n"
                if stdout:
                    error_msg += f"标准输出:\n{stdout}\n"
                if stderr:
                    error_msg += f"错误信息:\n{stderr}\n"
                error_msg += "######请检查对应的log文件######"

                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            logs = f.readlines()[-20:]
                            if logs:
                                error_msg += f"日志文件最后20行:\n{''.join(logs)}\n"
                    except Exception:
                        error_msg += f"无法读取日志文件: {log_file}\n"

                return _train_error_response(
                    script_name or script_path,
                    error_msg,
                    error_reason="docker_start_failed",
                    container=docker_container or docker_image,
                    returnCode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
        response_text = (
            f"脚本已在Docker容器中后台启动运行\n"
            f"脚本: {os.path.basename(script_path)}\n"
            f"容器: {process_info.get('container_id', docker_container or docker_image)}\n"
            
        )
        return ToolResponse(content=[
            TextBlock(type="text", text=response_text)
        ], metadata={
            "success": True,
            "protocol_hint": _train_protocol_hint(
                "job_started",
                script_name,
                response_text,
                container=process_info.get("container_id", docker_container or docker_image),
                script=os.path.basename(script_path),
                pid=str(process_info.get("pid") or ""),
                command=str(process_info.get("command") or " ".join(command)),
                scriptArgs=script_args or {},
                envKeys=sorted((env_vars or {}).keys()),
            ),
        })
    
    except Exception as e:
        import traceback
        response_text = f"错误: 无法在Docker容器中后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return _train_error_response(
            script_name or script_path,
            response_text,
            error_reason="docker_background_exception",
            container=docker_container or docker_image,
        )

def run_script_in_background(
    script_path: str,
    script_args: Dict[str, Any] = None,
    env_vars: Dict[str, str] = None,
    log_file: str = None,
    pid_file: str = None,
    check_startup: bool = True,  # 新增：是否检查启动状态
    startup_check_delay: int = 3,  # 新增：启动检查延迟时间（秒）
    **kwargs
) -> ToolResponse:
    """
    在后台运行本地脚本
    """
    try:
        # 检查脚本文件是否存在
        abs_script_path = os.path.abspath(script_path)
        if not os.path.exists(abs_script_path):
            return _train_error_response(
                os.path.basename(script_path),
                f"错误: 找不到脚本文件: {script_path}",
                error_reason="script_file_not_found",
                recoverable=True,
                scriptPath=script_path,
            )
        
        # 构建Python执行命令
        python_executable = sys.executable
        command = [python_executable, abs_script_path]
        
        # 添加命令行参数
        if script_args:
            for key, value in script_args.items():
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                        continue
                else:
                    command.extend([f"--{key}", str(value)])
        
        # 设置环境变量
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # 设置日志文件
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 重定向输出到日志文件
            with open(log_file, 'a') as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=log_handle,
                    env=env,
                    cwd=os.path.dirname(abs_script_path)
                )
        else:
            # 使用空设备，丢弃输出
            with open(os.devnull, 'w') as devnull:
                process = subprocess.Popen(
                    command,
                    stdout=devnull,
                    stderr=devnull,
                    env=env,
                    cwd=os.path.dirname(abs_script_path)
                )
        
        # 保存PID到文件
        if pid_file:
            pid_dir = os.path.dirname(pid_file)
            if pid_dir and not os.path.exists(pid_dir):
                os.makedirs(pid_dir, exist_ok=True)
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
        # 检查进程是否启动失败
        if check_startup and startup_check_delay > 0:
            # 等待指定的延迟时间
            time.sleep(startup_check_delay)
            
            # 检查进程是否还在运行
            returncode = process.poll()  # 如果进程已结束，返回退出码，否则返回None
            
            if returncode is not None and returncode != 0:
                error_msg = f"脚本后台启动失败，返回码: {returncode}\n"
                
                # 尝试从日志文件读取错误信息
                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            logs = f.readlines()[-20:]  # 读取最后20行
                            if logs:
                                error_msg += f"日志文件最后20行:\n{''.join(logs)}\n"
                    except:
                        error_msg += f"无法读取日志文件: {log_file}\n"
                else:
                    error_msg += "未指定日志文件，无法获取详细错误信息\n"
                
                # 删除PID文件（因为进程已结束）
                if pid_file and os.path.exists(pid_file):
                    try:
                        os.remove(pid_file)
                    except:
                        pass
                
                return _train_error_response(
                    os.path.basename(script_path),
                    error_msg,
                    error_reason="local_start_failed",
                    returnCode=returncode,
                    logFile=log_file,
                    pidFile=pid_file,
                )
        response_text = (
            f"脚本已在后台启动运行\n"
            f"PID: {process.pid}\n"
            f"脚本: {os.path.basename(script_path)}\n"
        )
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=response_text
            )
        ], metadata={
            "success": True,
            "protocol_hint": _train_protocol_hint(
                "job_started",
                os.path.basename(script_path),
                response_text,
                pid=str(process.pid),
                script=os.path.basename(script_path),
                command=" ".join(command),
                scriptArgs=script_args or {},
                logFile=log_file,
                pidFile=pid_file,
            ),
        })
    
    except Exception as e:
        import traceback
        response_text = f"错误: 无法在后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return _train_error_response(
            os.path.basename(script_path),
            response_text,
            error_reason="local_background_exception",
            scriptPath=script_path,
        )


# 为了兼容性，添加一个别名
test_script = run_local_script

_resume_resource_reservation_heartbeats()











