"""Model-aware GPU counts for the existing packaged medical launchers.

Two-model launchers split the whitelist into ceil(N/2), floor(N/2), with
the updated model first. No changes to those packaged programs are needed.
"""
from __future__ import annotations

import json
import math
import subprocess
from typing import Any, Dict, List

MEDICAL_SCRIPTS = {"single_model_evaluation_vpn", "compare_between_models_vpn", "ckpt_eval"}
PLAN_ENV = "MEDFLOW_EVALUATION_TP_PLAN"


class MedicalTPError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _read_json(container: str, path: str, *, optional: bool = False) -> Dict[str, Any]:
    try:
        result = subprocess.run(["docker", "exec", container, "cat", path],
                                capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MedicalTPError("medical_model_config_unavailable", f"无法读取模型配置 {path}：{exc}") from exc
    if result.returncode != 0:
        if optional:
            return {}
        raise MedicalTPError("medical_model_config_unavailable", f"无法读取模型配置：{path}。")
    try:
        if len(result.stdout.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("JSON exceeds 4 MiB")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (ValueError, TypeError) as exc:
        raise MedicalTPError("medical_model_config_invalid", f"模型配置格式无效：{path}（{exc}）。") from exc


def _positive(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
        return number if number > 0 and str(value).strip() == str(number) else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def _weight_bytes(container: str, model_path: str) -> int:
    # Inspect only metadata/file sizes; prefer safetensors over duplicate .bin files.
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _read_json(container, f"{model_path}/{filename}", optional=True)
        metadata = index.get("metadata")
        size = _positive(metadata.get("total_size")) if isinstance(metadata, dict) else 0
        if size:
            return size
    command = '''for pattern in 'model*.safetensors' 'pytorch_model*.bin'; do
total=0
for file in "$1"/$pattern; do
    if test -f "$file"; then
        size=$(stat -c '%s' "$file") || exit 1
        total=$((total + size))
    fi
done
if test "$total" -gt 0; then printf '%s' "$total"; exit 0; fi
done'''
    try:
        result = subprocess.run(["docker", "exec", container, "sh", "-c", command, "sh", model_path],
                                capture_output=True, text=True, timeout=8)
        return _positive(result.stdout.strip()) if result.returncode == 0 else 0
    except (OSError, subprocess.SubprocessError):
        return 0


def model_profile(container: str, model_path: str) -> Dict[str, Any]:
    config = _read_json(container, f"{model_path.rstrip('/')}/config.json")
    text = config.get("text_config") or config.get("llm_config") or config
    if not isinstance(text, dict):
        raise MedicalTPError("medical_model_config_invalid", f"模型 text_config 格式无效：{model_path}。")
    heads = _positive(text.get("num_attention_heads") or text.get("n_head") or text.get("num_heads"))
    if not heads:
        raise MedicalTPError("medical_attention_heads_unknown", f"模型配置缺少有效的注意力头数，无法自动确定 TP：{model_path}。")
    kv_heads = _positive(text.get("num_key_value_heads") or text.get("n_head_kv") or text.get("num_kv_heads") or text.get("multi_query_group_num"))
    if not kv_heads:
        kv_heads = 1 if text.get("multi_query") else heads
    # MLA replicates a latent cache; ordinary GQA's KV-head rule does not apply.
    mla = bool(text.get("kv_lora_rank"))
    hidden = _positive(text.get("hidden_size") or text.get("n_embd"))
    intermediate = _positive(text.get("intermediate_size") or text.get("ffn_hidden_size"))
    head_dim = _positive(text.get("head_dim")) or (hidden // heads if hidden and hidden % heads == 0 else 0)
    quantized = bool(config.get("quantization_config") or text.get("quantization_config"))
    weights = _weight_bytes(container, model_path)
    # vLLM auto dtype converts float32 weights to float16.
    dtype = str(text.get("dtype") or text.get("torch_dtype") or config.get("torch_dtype") or "").lower()
    if dtype in {"float32", "torch.float32"} and not quantized:
        weights //= 2
    return {"modelPath": model_path, "heads": heads, "kvHeads": kv_heads, "mla": mla,
            "hidden": hidden, "intermediate": intermediate, "headDim": head_dim,
            "layers": _positive(text.get("num_hidden_layers") or text.get("n_layer")),
            "weightBytes": weights, "quantized": quantized}


def tp_issue(profile: Dict[str, Any], tp: int) -> str:
    if tp < 1:
        return "TP 必须是正整数"
    if profile["heads"] % tp:
        return f"注意力头数 {profile['heads']} 不能被 TP={tp} 整除"
    kv = profile["kvHeads"]
    if not profile["mla"] and (kv % tp if tp <= kv else tp % kv):
        return f"KV 头数 {kv} 与 TP={tp} 不兼容（须支持均分或完整复制）"
    for key in ("hidden", "intermediate"):
        if profile[key] and profile[key] % tp:
            return f"模型维度 {key}={profile[key]} 不能被 TP={tp} 整除"
    return ""


def service_tps(gpu_count: int, service_count: int) -> List[int]:
    base, remainder = divmod(gpu_count, service_count)
    return [base + (index < remainder) for index in range(service_count)]


def plan_gpu_count(profiles: List[Dict[str, Any]], requested: Any, *, explicit: bool) -> Dict[str, Any]:
    count = _positive(requested)
    services = len(profiles)
    if not count or not services or count < services:
        raise MedicalTPError("medical_gpu_count_invalid", f"医疗评测 GPU 数必须是正整数且至少为 {services}。")
    common = [tp for tp in range(1, count // services + 1) if all(not tp_issue(p, tp) for p in profiles)]
    if explicit:
        tps = service_tps(count, services)
        problems = [f"{p['modelPath']}：{issue}" for p, tp in zip(profiles, tps) if (issue := tp_issue(p, tp))]
        if problems:
            suggestions = ", ".join(str(tp * services) for tp in common) or "无"
            raise MedicalTPError("medical_tp_incompatible", "指定 gpu-count 不兼容：" + "；".join(problems) + f"。当前范围内建议 gpu-count：{suggestions}。")
    else:
        if not common:
            raise MedicalTPError("medical_tp_incompatible", "当前卡数范围内没有共同合法 TP。")
        count = common[-1] * services
        tps = [common[-1]] * services
    return {"requestedGpuCount": int(requested), "gpuCount": count, "tensorParallelSizes": tps, "models": profiles}


def plan_medical_resources(script: str, container: str, cli: Dict[str, Any], env: Dict[str, str],
                           requested: int, *, explicit: bool,
                           base_model_path: str | None = None) -> Dict[str, Any]:
    if script == "single_model_evaluation_vpn":
        paths = [str(cli.get("model_fir") or env.get("model_fir") or "").strip()]
    elif script == "compare_between_models_vpn":
        # The first GPU group serves model_sec in the existing launcher.
        paths = [str(cli.get(key) or env.get(key) or "").strip() for key in ("model_sec", "model_fir")]
    else:
        checkpoint = str(cli.get("CKPT_PATH") or env.get("CKPT_PATH") or "").strip().rstrip("/")
        adapter = {} if base_model_path else _read_json(container, f"{checkpoint}/adapter_config.json")
        base = str(base_model_path or adapter.get("base_model_name_or_path") or "").strip()
        if not base.startswith("/"):
            raise MedicalTPError("base_model_unavailable", f"checkpoint 基础模型必须是容器内绝对路径：{base or '<empty>'}。")
        paths = [base, base]  # Merged checkpoint retains the base architecture.
    profiles_by_path = {path: model_profile(container, path) for path in dict.fromkeys(paths)}
    return plan_gpu_count([profiles_by_path[path] for path in paths], requested, explicit=explicit)


def validate_allocation(plan: Dict[str, Any], assigned: str, snapshot: Dict[str, Any]) -> List[str]:
    ids = [item.strip() for item in assigned.split(",") if item.strip()]
    if len(ids) != plan["gpuCount"] or len(set(ids)) != len(ids):
        raise MedicalTPError("medical_gpu_allocation_mismatch", f"GPU 分配与 TP 计划不一致：计划 {plan['gpuCount']} 张，实际 {assigned}。")
    gpus = {str(gpu.get("index")): gpu for gpu in snapshot.get("gpus", [])}
    warnings = []
    cursor = 0
    for profile, tp in zip(plan["models"], plan["tensorParallelSizes"]):
        group = ids[cursor:cursor + tp]
        cursor += tp
        weights = profile["weightBytes"]
        if not weights:
            warnings.append(f"{profile['modelPath']} 权重大小未知，跳过显存估算")
            continue
        # Cache for one 24k-token request; max-num-seqs=10 does not require ten full requests.
        kv_bytes = 0
        if not profile["mla"] and profile["layers"] and profile["headDim"]:
            kv_bytes = 2 * 2 * 24000 * profile["layers"] * max(1, profile["kvHeads"] // tp) * profile["headDim"]
        required_mib = math.ceil((weights / tp + kv_bytes) / (1024 ** 2)) + 1024
        for gpu_id in group:
            gpu = gpus.get(gpu_id, {})
            total = float(gpu.get("memoryTotal") or 0)
            free = float(gpu.get("memoryFree") or 0)
            if not total or gpu.get("memoryFree") is None:
                warnings.append(f"GPU {gpu_id} 显存快照不完整，跳过显存估算")
                continue
            budget = min(total * 0.8, free)
            if budget < required_mib:
                raise MedicalTPError("medical_gpu_memory_insufficient",
                    f"合法 TP={tp} 下显存不足：{profile['modelPath']} 在 GPU {gpu_id} "
                    f"预计至少需要 {required_mib} MiB，预算 {int(budget)} MiB。请申请更多兼容卡数或更换大显存 GPU。")
        if profile["quantized"]:
            warnings.append(f"{profile['modelPath']} 已校验基础 TP 维度，量化后端额外约束以 vLLM 启动为准")
    return warnings
