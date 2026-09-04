# -*- coding: utf-8 -*-
"""Lightweight helpers shared by training launch and monitor tools."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re
import ast
import json
import shlex
import os

PID_MARKER = "__PID__:"
_WANDB_URL_RE = re.compile(r"https?://wandb\.ai/\S+", re.IGNORECASE)
_GPU_LIST_RE = re.compile(r"^\d+(?:,\d+)*$")


def require_assigned_gpu_env(env_vars: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Normalize the platform GPU assignment for every train/evaluate launch."""
    result = {str(key): str(value) for key, value in (env_vars or {}).items()}
    assigned = (
        result.get("MEDFLOW_ASSIGNED_GPUS")
        or result.get("CUDA_VISIBLE_DEVICES")
        or result.get("LOCALHOST_ID")
        or os.environ.get("MEDFLOW_ASSIGNED_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("LOCALHOST_ID")
        or ""
    )
    assigned = re.sub(r"\s+", "", str(assigned))
    if not _GPU_LIST_RE.fullmatch(assigned):
        raise ValueError(
            "启动训练或评估前必须传入 MEDFLOW_ASSIGNED_GPUS，"
            "值应为逗号分隔的物理 GPU 编号。"
        )
    if len(set(assigned.split(","))) != len(assigned.split(",")):
        raise ValueError("MEDFLOW_ASSIGNED_GPUS 不能包含重复 GPU 编号。")
    result["MEDFLOW_ASSIGNED_GPUS"] = assigned
    result["MEDFLOW_REQUIRE_GPU_ASSIGNMENT"] = "1"
    return result


def infer_train_type_from_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lower = name.lower()
    if "create_command_vpn" in lower or "schedule" in lower or "定时" in name:
        return "scheduled"
    if "grpo" in lower:
        return "grpo"
    if "batch_train_pretrain_full" in lower:
        return "pretrain_full"
    if "batch_train_pretrain_lora" in lower:
        return "pretrain_lora"
    if "train_multinode_sft_pipeline" in lower:
        return "lora"
    if "train_multinode_dpo_pipeline" in lower:
        return "enhanced"
    if "dpo" in lower:
        return "enhanced"
    if "lora" in lower:
        return "lora"
    if "full" in lower:
        return "full"
    return None


def normalize_train_type(train_type: Optional[str]) -> Optional[str]:
    if train_type is None:
        return None
    value = str(train_type).strip().lower()
    if not value:
        return None
    aliases = {
        "dpo": "enhanced",
        "enhance": "enhanced",
        "enhanced": "enhanced",
        "增强": "enhanced",
        "增强训练": "enhanced",
        "lora": "lora",
        "sft": "lora",
        "lora训练": "lora",
        "lora批量训练": "lora",
        "full": "full",
        "full_train": "full",
        "pretrain_full": "pretrain_full",
        "pretrainfull": "pretrain_full",
        "full_pt": "pretrain_full",
        "fullpt": "pretrain_full",
        "全参pt": "pretrain_full",
        "全参预训练": "pretrain_full",
        "pretrain_lora": "pretrain_lora",
        "pretrainlora": "pretrain_lora",
        "lora_pt": "pretrain_lora",
        "lorapt": "pretrain_lora",
        "lora预训练": "pretrain_lora",
        "预训练": "pretrain_lora",
        "pt": "pretrain_lora",
        "pt训练": "pretrain_lora",
        "继续预训练": "pretrain_lora",
        "全参": "full",
        "全参训练": "full",
        "全参批量训练": "full",
        "scheduled": "scheduled",
        "schedule": "scheduled",
        "定时": "scheduled",
        "定时训练": "scheduled",
        "grpo": "grpo",
    }
    return aliases.get(value, value)


def public_train_type(
    train_type: Optional[str],
    launch_mode: Optional[str] = None,
    script_name: Optional[str] = None,
) -> Optional[str]:
    normalized = normalize_train_type(train_type)
    script = str(script_name or "").strip().lower()
    is_multinode = str(launch_mode or "").strip().lower() in {"multinode", "dual_node", "dual"}
    is_multinode = is_multinode or "train_multinode_sft_pipeline" in script or "train_multinode_dpo_pipeline" in script
    if normalized == "lora":
        return "multinode_lora_sft" if is_multinode else "lora_sft"
    if normalized == "full":
        return "full_sft"
    if normalized == "pretrain_lora":
        return "pretrain_lora"
    if normalized == "pretrain_full":
        return "pretrain_full"
    if normalized == "enhanced":
        return "multinode_enhanced" if is_multinode else "enhanced"
    if normalized == "grpo":
        return "grpo"
    if normalized == "scheduled":
        return "scheduled"
    return normalized


def public_train_type_text(
    train_type: Optional[str],
    launch_mode: Optional[str] = None,
    script_name: Optional[str] = None,
) -> Optional[str]:
    public_type = public_train_type(train_type, launch_mode, script_name)
    return {
        "lora_sft": "LoRA SFT",
        "full_sft": "全参 SFT",
        "pretrain_lora": "LoRA PT",
        "pretrain_full": "全参 PT",
        "enhanced": "增强训练",
        "grpo": "GRPO",
        "multinode_lora_sft": "双机 LoRA SFT",
        "multinode_enhanced": "双机增强训练",
        "scheduled": "定时训练",
    }.get(public_type or "")

def parse_pid_from_output(text: str) -> Optional[str]:
    if not text:
        return None
    marker_match = re.search(rf"{re.escape(PID_MARKER)}(\d+)", text)
    if marker_match:
        return marker_match.group(1)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.isdigit():
            return line
    return None


def parse_wandb_url_from_lines(lines: List[str]) -> Optional[str]:
    if not lines:
        return None
    for line in reversed(lines):
        if "wandb.ai" not in line:
            continue
        match = _WANDB_URL_RE.search(line)
        if not match:
            continue
        url = match.group(0).rstrip(").,;\"'")
        return url
    return None


def validate_pid_binding(
    pid_source: Optional[str],
    pid_record: Optional[Dict[str, Any]],
    pid_matches_proc: bool,
    wandb_pid_match: bool,
    pid_alive: bool,
) -> Tuple[bool, str]:
    if pid_source != "input":
        return True, "pid_not_required"
    if not pid_record and not pid_matches_proc and not wandb_pid_match:
        return False, "pid_not_found"
    if not pid_alive and not wandb_pid_match:
        if pid_record:
            return True, "pid_ended"
        return False, "pid_wandb_not_found"
    return True, "ok"


def should_return_status_record(
    pid_source: Optional[str],
    pid_alive: bool,
    pid_matches_proc: bool,
    wandb_pid_match: bool,
    status_record_exists: bool,
) -> bool:
    if pid_source != "input":
        return False
    if not status_record_exists:
        return False
    return not pid_alive and not pid_matches_proc and not wandb_pid_match


def build_background_shell_command(
    script_path: str,
    docker_executable: str,
    prefix_args: Optional[List[str]],
    positional_args: Optional[List[str]],
    script_args: Optional[Dict[str, Any]],
    docker_working_dir: Optional[str],
    log_file: Optional[str],
    shell_prefix: Optional[str] = None,
    shell_suffix: Optional[str] = None,
) -> str:
    inner_parts: List[str] = []
    if docker_executable and script_path and (
        docker_executable == script_path or script_path in docker_executable
    ):
        inner_parts.append(docker_executable)
    else:
        if docker_executable:
            inner_parts.append(docker_executable)
        if prefix_args:
            inner_parts.extend(str(arg) for arg in prefix_args)
        inner_parts.append(script_path)

    if positional_args:
        inner_parts.extend(str(arg) for arg in positional_args)

    if script_args:
        for key, value in script_args.items():
            if isinstance(value, bool):
                if value:
                    inner_parts.append(f"--{key}")
            elif isinstance(value, (int, float, str)):
                inner_parts.append(f"--{key}")
                inner_parts.append(str(value))
            elif isinstance(value, list):
                for item in value:
                    inner_parts.append(f"--{key}")
                    inner_parts.append(str(item))
            elif value is None:
                continue
            else:
                inner_parts.append(f"--{key}")
                inner_parts.append(str(value))

    inner_cmd_str = " ".join(shlex.quote(part) for part in inner_parts)
    if shell_suffix:
        inner_cmd_str = f"{inner_cmd_str} {shell_suffix}"
    redirect = ""
    if log_file:
        redirect = f" >> {shlex.quote(log_file)} 2>&1"
    full_cmd = f"{inner_cmd_str}{redirect} & echo {PID_MARKER}$!"
    if shell_prefix:
        full_cmd = f"{shell_prefix} && {full_cmd}"
    if docker_working_dir:
        return f"cd {shlex.quote(docker_working_dir)} && ( {full_cmd} )"
    return full_cmd


def parse_wandb_run_timestamp(name: str) -> Optional[datetime]:
    if not name:
        return None
    match = re.match(r"^(?:offline-)?run-(\d{8}_\d{6})-", name)
    if not match:
        return None
    stamp = match.group(1)
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def select_wandb_run_by_start_time(
    runs: List[Dict[str, Any]],
    run_start_time: Optional[datetime],
    max_minutes: int,
    allow_early_seconds: int = 0,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not runs:
        return None, "no_runs"
    if not run_start_time:
        return None, "no_start_time"
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    tolerance_seconds = max(0, int(allow_early_seconds))
    latest_allowed = run_start_time + timedelta(minutes=max_minutes) if max_minutes > 0 else None
    earliest_allowed = run_start_time - timedelta(seconds=tolerance_seconds)
    for r in runs:
        ts = parse_wandb_run_timestamp(r.get("name") or "")
        if ts:
            if ts < earliest_allowed:
                continue
            if latest_allowed and ts > latest_allowed:
                continue
            diff = abs((ts - run_start_time).total_seconds())
            candidates.append((diff, r))
            continue
        mtime = r.get("mtime")
        if isinstance(mtime, (int, float)):
            mtime_dt = datetime.fromtimestamp(mtime)
            if mtime_dt < earliest_allowed:
                continue
            if latest_allowed and mtime_dt > latest_allowed:
                continue
            diff = abs(mtime - run_start_time.timestamp())
            candidates.append((diff, r))
    if not candidates:
        return None, "start_time_out_of_range"
    candidates.sort(key=lambda item: item[0])
    best_diff, best_run = candidates[0]
    if max_minutes > 0 and best_diff > max_minutes * 60:
        return None, "start_time_too_far"
    return best_run, "by_start_time"


def select_log_file_by_start_time(
    items: List[Dict[str, Any]],
    pid_started_at: Optional[datetime],
    allow_early_seconds: int = 120,
) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    if not pid_started_at:
        return max(items, key=lambda it: it.get("mtime", 0))
    earliest = pid_started_at.timestamp() - max(0, int(allow_early_seconds))
    candidates = [it for it in items if (it.get("mtime") or 0) >= earliest]
    if not candidates:
        return None
    return max(candidates, key=lambda it: it.get("mtime", 0))


def select_wandb_run(
    runs: List[Dict[str, Any]],
    run_start_time: Optional[datetime],
    time_window_minutes: int,
    fallback_latest: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not runs:
        return None, "no_runs"
    now = datetime.now()
    cutoff = now - timedelta(minutes=time_window_minutes)
    if run_start_time:
        cutoff = max(cutoff, run_start_time - timedelta(minutes=5))

    candidates: List[Dict[str, Any]] = []
    for r in runs:
        mtime = r.get("mtime")
        if isinstance(mtime, (int, float)) and mtime >= cutoff.timestamp():
            candidates.append(r)

    if not candidates:
        if fallback_latest:
            runs.sort(key=lambda r: r.get("mtime", 0), reverse=True)
            return runs[0], "fallback_latest"
        return None, "no_recent_runs"

    def _score(r: Dict[str, Any]) -> Tuple[float, float]:
        ts = parse_wandb_run_timestamp(r.get("name") or "")
        ts_score = ts.timestamp() if ts else 0.0
        return (r.get("mtime", 0.0), ts_score)

    candidates.sort(key=_score, reverse=True)
    return candidates[0], "by_mtime"


def pick_metric_key(records: List[Dict[str, Any]], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        for rec in records:
            if key in rec:
                return key
    return None


def fallback_key_by_substring(records: List[Dict[str, Any]], keyword: str) -> Optional[str]:
    freq: Dict[str, int] = {}
    for rec in records:
        for k, v in rec.items():
            if keyword not in k.lower():
                continue
            if isinstance(v, (int, float, str)):
                freq[k] = freq.get(k, 0) + 1
    if not freq:
        return None
    return sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def is_integer_step_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return value >= 0 and value.is_integer()
    text = str(value).strip()
    if not text:
        return False
    return bool(re.fullmatch(r"\d+", text))


def extract_wandb_history(
    records: List[Dict[str, Any]],
    history_limit: int,
    summary: Optional[Dict[str, Any]] = None,
    last_update_time: Optional[str] = None,
) -> Dict[str, Any]:
    key_records = records if records else ([summary] if summary else [])
    loss_key = pick_metric_key(
        key_records,
        [
            "train/loss",
            "loss",
            "train_loss",
            "training_loss",
            "loss/train",
            "lm_loss",
            "total_loss",
        ],
    ) or fallback_key_by_substring(key_records, "loss")
    step_key = pick_metric_key(
        key_records,
        ["train/global_step", "step", "global_step", "train/step", "train/steps"],
    ) or fallback_key_by_substring(key_records, "step")
    if step_key == "_step":
        step_key = None
    epoch_key = pick_metric_key(
        key_records,
        ["train/epoch", "epoch", "epochs", "trainer/epoch"],
    ) or fallback_key_by_substring(key_records, "epoch")

    def _is_present(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    history: List[Dict[str, Any]] = []
    for rec in records[-history_limit:] if history_limit > 0 else records:
        loss_val = rec.get(loss_key) if loss_key else None
        step_val = rec.get(step_key) if step_key else None
        epoch_val = rec.get(epoch_key) if epoch_key else None
        if not _is_present(loss_val) or not (
            _is_present(step_val) or is_integer_step_value(epoch_val)
        ):
            continue
        if not _is_present(step_val) and is_integer_step_value(epoch_val):
            step_val = epoch_val
        history.append(
            {
                "step": step_val,
                "epoch": epoch_val,
                "loss": loss_val,
                "timestamp": rec.get("_timestamp"),
            }
        )

    latest = history[-1] if history else None
    if not latest and summary:
        loss_val = summary.get(loss_key) if loss_key else None
        step_val = summary.get(step_key) if step_key else None
        epoch_val = summary.get(epoch_key) if epoch_key else None
        if _is_present(loss_val) and (_is_present(step_val) or is_integer_step_value(epoch_val)):
            if not _is_present(step_val) and is_integer_step_value(epoch_val):
                step_val = epoch_val
            latest = {"step": step_val, "epoch": epoch_val, "loss": loss_val, "timestamp": None}

    if not last_update_time and records:
        ts = records[-1].get("_timestamp")
        if isinstance(ts, (int, float)):
            last_update_time = datetime.fromtimestamp(ts).isoformat()

    return {
        "history": history,
        "latest": latest,
        "loss_key": loss_key,
        "step_key": step_key,
        "epoch_key": epoch_key,
        "last_update_time": last_update_time,
    }


def parse_output_log_history(lines: List[str]) -> List[Dict[str, Any]]:
    loss_patterns = [
        r"(?:^|[^\w])(?:loss|train_loss|training_loss)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
        r"(?:^|[^\w])(?:actor/kl_loss|actor_kl_loss)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    ]
    lr_patterns = [
        r"(?:^|[^\w])(?:learning_rate|lr)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
        r"(?:^|[^\w])(?:actor/lr|actor_lr)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    ]
    step_patterns = [
        r"(?:^|[^\w])(?:step|global_step|current_steps|iteration|iter)\s*[:=]\s*(\d+)",
        r"(?:^|[^\w])(?:training/global_step|training_global_step)\s*[:=]\s*(\d+)",
        r"(?:^|[^\w])step\s+(\d+)\s*/\s*\d+",
    ]
    epoch_patterns = [
        r"(?:^|[^\w])(?:epoch|train_epoch)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
        r"(?:^|[^\w])(?:training/epoch|training_epoch)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    ]
    pg_loss_patterns = [
        r"(?:^|[^\w])(?:actor/pg_loss|actor_pg_loss)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    ]
    kl_coef_patterns = [
        r"(?:^|[^\w])(?:actor/kl_coef|actor_kl_coef)\s*[:=]\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    ]

    def _parse_line(line: str) -> Optional[Dict[str, Any]]:
        if line and "{" in line and "}" in line:
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                payload = line[start : end + 1]
                try:
                    data = ast.literal_eval(payload)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    return None
        return None

    def _pick_first(d: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _to_number(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"[+-]?\d+", text):
            try:
                return int(text)
            except Exception:
                return None
        if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", text):
            try:
                return float(text)
            except Exception:
                return None
        return None

    def _is_present(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _parse_step_metric_line(line: str) -> Optional[Dict[str, Any]]:
        # Example:
        # step:1 - actor/lr:1e-06 - actor/kl_loss:0.12 - training/global_step:1 - training/epoch:0
        if " - " not in line or ":" not in line:
            return None
        if "step:" not in line and "training/global_step" not in line:
            return None

        parsed: Dict[str, Any] = {}
        for part in line.split(" - "):
            seg = part.strip()
            if ":" not in seg:
                continue
            key, value = seg.split(":", 1)
            key = key.strip()
            value_num = _to_number(value)
            if value_num is None:
                continue
            parsed[key] = value_num

        if not parsed:
            return None

        actor_kl_loss = _pick_first(parsed, ["actor/kl_loss", "actor_kl_loss"])
        actor_pg_loss = _pick_first(parsed, ["actor/pg_loss", "actor_pg_loss"])
        actor_kl_coef = _pick_first(parsed, ["actor/kl_coef", "actor_kl_coef"])

        item: Dict[str, Any] = {
            "loss": actor_kl_loss,
            "learning_rate": _pick_first(parsed, ["actor/lr", "actor_lr", "learning_rate", "lr"]),
            "step": _pick_first(parsed, ["training/global_step", "training_global_step", "step"]),
            "epoch": _pick_first(parsed, ["training/epoch", "training_epoch", "epoch", "train_epoch"]),
            "actor_kl_loss": actor_kl_loss,
            "actor_pg_loss": actor_pg_loss,
            "actor_kl_coef": actor_kl_coef,
        }
        if any(v is not None for v in item.values()):
            return item
        return None

    def _extract_number(patterns: List[str], line: str) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    return None
        return None

    def _extract_int(patterns: List[str], line: str) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    return None
        return None

    history: List[Dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Ignore shell trace command echo lines (e.g. set -x output) that may contain lr values.
        if stripped.startswith("+") and (
            "python" in stripped
            or "accelerate launch" in stripped
            or "verl.trainer.main_ppo" in stripped
            or "run_qwen3-8b_260417.sh" in stripped
        ):
            continue

        step_item = _parse_step_metric_line(stripped)
        if step_item is not None:
            history.append(step_item)
            continue

        data = _parse_line(stripped)
        if isinstance(data, dict):
            actor_kl_loss = _to_number(_pick_first(data, ["actor/kl_loss", "actor_kl_loss"]))
            actor_pg_loss = _to_number(_pick_first(data, ["actor/pg_loss", "actor_pg_loss"]))
            actor_kl_coef = _to_number(_pick_first(data, ["actor/kl_coef", "actor_kl_coef"]))
            item = {
                "loss": actor_kl_loss if actor_kl_loss is not None else _to_number(_pick_first(data, ["loss", "train_loss", "training_loss"])),
                "learning_rate": _to_number(_pick_first(data, ["actor/lr", "actor_lr", "learning_rate", "lr"])),
                "step": _to_number(_pick_first(data, ["training/global_step", "training_global_step", "step", "global_step", "current_steps", "iteration"])),
                "epoch": _to_number(_pick_first(data, ["training/epoch", "training_epoch", "epoch", "train_epoch"])),
                "actor_kl_loss": actor_kl_loss,
                "actor_pg_loss": actor_pg_loss,
                "actor_kl_coef": actor_kl_coef,
            }
        else:
            parsed_kl_loss = _extract_number(loss_patterns, stripped)
            item = {
                "loss": parsed_kl_loss,
                "learning_rate": _extract_number(lr_patterns, stripped),
                "step": _extract_int(step_patterns, stripped),
                "epoch": _extract_number(epoch_patterns, stripped),
                "actor_kl_loss": parsed_kl_loss,
                "actor_pg_loss": _extract_number(pg_loss_patterns, stripped),
                "actor_kl_coef": _extract_number(kl_coef_patterns, stripped),
            }
        if any(v is not None for v in item.values()):
            history.append(item)

    deduped_by_step: Dict[str, Dict[str, Any]] = {}
    for item in history:
        if not _is_present(item.get("loss")):
            continue
        if not _is_present(item.get("step")) and is_integer_step_value(item.get("epoch")):
            item = dict(item)
            item["step"] = item.get("epoch")
        if not _is_present(item.get("step")):
            continue
        step_key_text = str(item.get("step"))
        previous = deduped_by_step.get(step_key_text)
        if previous:
            merged = dict(previous)
            for key, value in item.items():
                if _is_present(value):
                    merged[key] = value
            deduped_by_step[step_key_text] = merged
        else:
            deduped_by_step[step_key_text] = item

    return list(deduped_by_step.values())



