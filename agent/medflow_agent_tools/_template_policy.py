# -*- coding: utf-8 -*-
"""Shared LLaMAFactory template policy for train/evaluate tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TEMPLATE_AMBIGUOUS_MESSAGE = (
    "无法唯一识别模型模板，请从候选模板中选择或显式提供 TEM/template。"
)
UNSUPPORTED_TEMPLATE_MESSAGE = "当前版本暂不支持该模型/template。"
TEMPLATE_REGISTRY_UNAVAILABLE_MESSAGE = "无法读取训练容器中的 LLaMA-Factory 模板注册表，已阻止启动。"

_TEMPLATE_REGISTRY_CACHE: Dict[str, Dict[str, Any]] = {}
_TEMPLATE_REGISTRY_CACHE_LOCK = threading.Lock()
_MAX_ADAPTER_DEPTH = 4
_TEMPLATE_REGISTRY_CLI = "/usr/local/insinfersystem/llamafactory-cli"
_TEMPLATE_REGISTRY_SCHEMA = "llamafactory.templates/v1"


def _rule(
    rule_id: str,
    template: str,
    model_family: str,
    name_pattern: str,
    *,
    model_types: Sequence[str],
    architectures: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "id": rule_id,
        "template": template,
        "modelFamily": model_family,
        "namePattern": re.compile(name_pattern, re.IGNORECASE),
        "modelTypes": {item.lower() for item in model_types},
        "architectures": {item.lower() for item in architectures},
    }


# Rules in this table identify concrete public model generations.  In particular,
# Qwen3.6 deliberately reuses Qwen3.5's Transformers model_type and architecture.
# A rule only resolves when the basename and at least one compatible config signal
# agree, so a misleading directory name cannot silently select a template.
EXACT_MODEL_RULES: Tuple[Dict[str, Any], ...] = (
    _rule(
        "qwen3_6",
        "qwen3_6",
        "qwen3.6",
        r"^qwen(?:[-_.]?3[._]6)(?:$|[-_.])",
        model_types=("qwen3_5",),
        architectures=("Qwen3_5ForConditionalGeneration",),
    ),
    _rule(
        "qwen3_5",
        "qwen3_5",
        "qwen3.5",
        r"^qwen(?:[-_.]?3[._]5)(?:$|[-_.])",
        model_types=("qwen3_5", "qwen3_5_moe"),
        architectures=("Qwen3_5ForConditionalGeneration",),
    ),
    _rule(
        "llama3",
        "llama3",
        "llama3",
        r"^(?:meta[-_.]?)?llama(?:[-_.]?3)(?:$|[-_.])",
        model_types=("llama",),
    ),
    _rule(
        "llama2",
        "llama2",
        "llama2",
        r"^(?:meta[-_.]?)?llama(?:[-_.]?2)(?:$|[-_.])",
        model_types=("llama",),
    ),
    _rule(
        "deepseekr1",
        "deepseekr1",
        "deepseek-r1",
        r"^deepseek(?:[-_.]?r1)(?:$|[-_.])",
        model_types=("deepseek_v3", "deepseek_v2", "llama", "qwen2"),
    ),
    _rule(
        "deepseek3",
        "deepseek3",
        "deepseek-v3",
        r"^deepseek(?:[-_.]?v3)(?:$|[-_.])",
        model_types=("deepseek_v3",),
    ),
    _rule(
        "mistral_small",
        "mistral_small",
        "mistral-small",
        r"^mistral(?:[-_.]?small)(?:$|[-_.])",
        model_types=("mistral", "mistral3"),
    ),
    _rule(
        "ministral",
        "ministral",
        "ministral",
        r"^ministral(?:$|[-_.])",
        model_types=("mistral",),
    ),
)


# Only model types with one stable text template belong here.  Ambiguous model
# types are handled separately and never auto-selected.
UNIQUE_MODEL_TYPE_TEMPLATES: Dict[str, str] = {
    "baichuan2": "baichuan2",
    "chatglm": "chatglm3",
    "cohere": "cohere",
    "falcon": "falcon",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "glm4": "glm4",
    "glm4_moe": "glm4_moe",
    "granite": "granite3",
    "granitemoe": "granite3",
    "internlm2": "intern2",
    "mllama": "mllama",
    "phi3": "phi",
    "phi4": "phi4",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen2_audio": "qwen2_audio",
    "qwen2_vl": "qwen2_vl",
    "qwen2_5_vl": "qwen2_vl",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_next": "qwen3",
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_moe": "qwen3_vl",
    "telechat2": "telechat2",
    "xverse": "xverse",
}

AMBIGUOUS_MODEL_TYPE_TEMPLATES: Dict[str, List[str]] = {
    "deepseek_v2": ["deepseek", "deepseekr1"],
    "deepseek_v3": ["deepseek3", "deepseekr1"],
    "llama": ["llama2", "llama3"],
    "mistral": ["mistral", "mistral_small", "ministral"],
    "qwen3_5": ["qwen3_5", "qwen3_6"],
    "qwen3_5_moe": ["qwen3_5", "qwen3_6"],
}

# Templates listed as supported by the bundled LLaMAFactory README table.
# Entries with Template='-' are intentionally excluded.
SUPPORTED_TEMPLATES = {
    "baichuan2",
    "chatglm3",
    "cohere",
    "deepseek",
    "deepseek3",
    "deepseekr1",
    "falcon",
    "gemma",
    "gemma3",
    "glm4",
    "glmz1",
    "granite3",
    "hunyuan",
    "index",
    "intern2",
    "intern_vl",
    "kimi_vl",
    "llama2",
    "llama3",
    "llama4",
    "mllama",
    "llava",
    "llava_next",
    "llava_next_video",
    "mimo",
    "cpm",
    "cpm3",
    "minicpm_o",
    "minicpm_v",
    "ministral",
    "mistral",
    "mistral_small",
    "paligemma",
    "phi",
    "phi_small",
    "phi4",
    "pixtral",
    "qwen",
    "qwen3",
    "qwen2_audio",
    "qwen2_omni",
    "qwen2_vl",
    "skywork_o1",
    "telechat2",
    "xverse",
    "yi",
    "yi_vl",
    "yuan",
    "ernie_nothink",
    "falcon_h1",
    "gemma2",
    "gemma3n",
    "glm4_moe",
    "glm4_5v",
    "gpt_oss",
    "granite4",
    "hunyuan_small",
    "intern_s1",
    "bailing_v2",
    "lfm2",
    "lfm2_vl",
    "mimo_v2",
    "cpm4",
    "minimax1",
    "minimax2",
    "ministral3",
    "phi4_mini",
    "qwen3_nothink",
    "qwen3_5",
    "qwen3_5_nothink",
    "qwen3_6",
    "qwen3_omni",
    "qwen3_vl",
    "seed_oss",
    "seed_coder",

}

TEMPLATE_PARAM_KEYS = {
    "TEM",
    "template",
    "模型模板",
    "模型类别",
}

MODEL_HINT_KEYS = {
    "模型",
    "基础模型",
    "模型路径",
    "模型位置",
    "基础模型路径",
    "model",
    "model_name",
    "model_path",
    "model-path",
    "MODEL_PATH",
    "base_model_path",
    "模型名",
    "模型名称",
}

def normalize_template(value: Any) -> str:
    return str(value or "").strip().lower()


def is_supported_template(value: Any) -> bool:
    return normalize_template(value) in SUPPORTED_TEMPLATES


def template_validation_issue(
    value: Any,
    *,
    param: str = "TEM/template",
    container: str = "",
) -> Optional[Dict[str, Any]]:
    template = normalize_template(value)
    if not template:
        return None
    if template in SUPPORTED_TEMPLATES:
        return None
    return {
        "error_reason": "unsupported_template",
        "message": UNSUPPORTED_TEMPLATE_MESSAGE,
        "container": container,
        "param": param,
        "template": str(value).strip(),
    }


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _docker_run(container: str, args: Sequence[str], *, timeout: int = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", str(container), *[str(arg) for arg in args]],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _managed_model_roots() -> Tuple[str, ...]:
    try:
        from resource_catalog import load_resource_catalog_config

        catalog = load_resource_catalog_config()
        roots = [
            str(item.get("path") or "").rstrip("/")
            for item in catalog.get("models", [])
            if isinstance(item, dict) and str(item.get("path") or "").startswith("/")
        ]
    except (ImportError, ValueError, TypeError):
        roots = ["/home/workspace/models", "/home/workspace/medical_models"]
    return tuple(dict.fromkeys(root for root in roots if root and root != "/"))


def _is_within_managed_model_root(path: str) -> bool:
    clean = str(path or "").rstrip("/")
    return any(clean == root or clean.startswith(root + "/") for root in _managed_model_roots())


def _runtime_template_registry(
    container: str,
) -> Tuple[Optional[Set[str]], Optional[str], str, Optional[Dict[str, Any]]]:
    """Read templates from the packaged CLI, with an Agent fallback for legacy images."""
    container = str(container or "").strip()
    if not container:
        return None, "container is empty", "runtime_cli", None
    try:
        identity = subprocess.run(
            ["docker", "inspect", "--format={{.Id}}:{{.Image}}", container],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc), "runtime_cli", None

    if identity.returncode != 0:
        detail = str(identity.stderr or identity.stdout or "Docker inspect failed").strip()
        return None, detail, "runtime_cli", None

    container_identity = str(identity.stdout or "").strip() or f"container:{container}"
    try:
        fingerprint_result = _docker_run(
            container,
            ["stat", "-Lc", "%i:%s:%Y", _TEMPLATE_REGISTRY_CLI],
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc), "runtime_cli", None
    if fingerprint_result.returncode != 0:
        detail = str(fingerprint_result.stderr or fingerprint_result.stdout or "packaged CLI not found").strip()
        return None, detail, "runtime_cli", None

    fingerprint = str(fingerprint_result.stdout or "").strip()
    if not fingerprint:
        return None, "packaged CLI fingerprint is empty", "runtime_cli", None
    cache_key = f"{container_identity}:{fingerprint}"

    with _TEMPLATE_REGISTRY_CACHE_LOCK:
        cached = _TEMPLATE_REGISTRY_CACHE.get(cache_key)
        if cached:
            warning = dict(cached["warning"]) if isinstance(cached.get("warning"), dict) else None
            return set(cached["templates"]), None, str(cached["source"]), warning

    try:
        result = _docker_run(
            container,
            [_TEMPLATE_REGISTRY_CLI, "templates", "--json"],
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc), "runtime_cli", None

    output_parts = (
        str(result.stdout or "").strip(),
        str(result.stderr or "").strip(),
    )
    combined_output = "\n".join(part for part in output_parts if part)
    if "Unknown command: templates" in combined_output:
        templates = set(SUPPORTED_TEMPLATES)
        warning = {
            "code": "template_registry_legacy_fallback",
            "message": "训练镜像不支持模板注册表查询，已使用 Agent 兼容模板清单。",
        }
        with _TEMPLATE_REGISTRY_CACHE_LOCK:
            _TEMPLATE_REGISTRY_CACHE[cache_key] = {
                "templates": set(templates),
                "source": "agent_legacy_fallback",
                "warning": dict(warning),
            }
        return templates, None, "agent_legacy_fallback", warning

    if result.returncode != 0:
        detail = combined_output or f"exit {result.returncode}"
        return None, detail, "runtime_cli", None

    payload: Optional[Dict[str, Any]] = None
    for line in reversed([line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]):
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            payload = decoded
            break

    if payload is None:
        return None, "packaged CLI returned invalid registry JSON", "runtime_cli", None
    if payload.get("schemaVersion") != _TEMPLATE_REGISTRY_SCHEMA:
        return None, "packaged CLI returned an unsupported registry schema", "runtime_cli", None
    values = payload.get("templates")
    invalid_values = (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
    )
    if invalid_values:
        return None, "packaged CLI returned an invalid template list", "runtime_cli", None

    runtime_templates = {normalize_template(item) for item in values}
    templates = runtime_templates & SUPPORTED_TEMPLATES
    if not templates:
        return None, "packaged CLI registry has no Agent-supported templates", "runtime_cli", None
    with _TEMPLATE_REGISTRY_CACHE_LOCK:
        _TEMPLATE_REGISTRY_CACHE[cache_key] = {
            "templates": set(templates),
            "source": "runtime_cli",
            "warning": None,
        }
    return templates, None, "runtime_cli", None


def _read_model_metadata(container: str, model_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Read model JSON files without importing model-provided Python code."""
    path = str(model_path or "").strip().rstrip("/")
    if not path.startswith("/"):
        return None, {
            "error_reason": "model_path_invalid",
            "message": f"模型路径必须是训练容器内的绝对路径：{path or '<empty>'}。",
            "modelPath": path,
        }
    try:
        canonical_result = _docker_run(container, ["readlink", "-f", path])
    except (OSError, subprocess.SubprocessError) as exc:
        return None, {
            "error_reason": "model_metadata_unavailable",
            "message": f"无法读取模型元数据：{exc}",
            "modelPath": path,
        }
    canonical = str(canonical_result.stdout or "").strip().rstrip("/")
    if canonical_result.returncode != 0 or not canonical:
        return None, {
            "error_reason": "model_path_missing",
            "message": f"模型目录不存在或无法解析：{path}。",
            "modelPath": path,
        }
    if not _is_within_managed_model_root(canonical):
        return None, {
            "error_reason": "model_path_unmanaged",
            "message": f"模型路径不在平台受管理目录中：{canonical}。",
            "modelPath": canonical,
        }
    try:
        directory = _docker_run(container, ["test", "-d", canonical])
    except (OSError, subprocess.SubprocessError) as exc:
        return None, {
            "error_reason": "model_metadata_unavailable",
            "message": f"无法校验模型目录：{exc}",
            "modelPath": canonical,
        }
    if directory.returncode != 0:
        return None, {
            "error_reason": "model_path_missing",
            "message": f"模型路径不是有效目录：{canonical}。",
            "modelPath": canonical,
        }

    def read_json(filename: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            result = _docker_run(container, ["cat", f"{canonical}/{filename}"])
        except (OSError, subprocess.SubprocessError) as exc:
            return None, str(exc)
        if result.returncode != 0:
            return None, None
        raw = str(result.stdout or "")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            return None, f"{filename} exceeds 1 MiB"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"invalid {filename}: {exc}"
        if not isinstance(value, dict):
            return None, f"{filename} must be a JSON object"
        return value, None

    config, config_error = read_json("config.json")
    tokenizer, tokenizer_error = read_json("tokenizer_config.json")
    adapter, adapter_error = read_json("adapter_config.json")
    errors = [item for item in (config_error, tokenizer_error, adapter_error) if item]
    if errors:
        return None, {
            "error_reason": "model_metadata_invalid",
            "message": "模型元数据格式无效：" + "; ".join(errors),
            "modelPath": canonical,
        }
    return {
        "canonicalPath": canonical,
        "modelName": os.path.basename(canonical),
        "config": config or {},
        "tokenizerConfig": tokenizer or {},
        "adapterConfig": adapter or {},
    }, None


def _metadata_evidence(metadata: Dict[str, Any]) -> Dict[str, Any]:
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    tokenizer = metadata.get("tokenizerConfig") if isinstance(metadata.get("tokenizerConfig"), dict) else {}
    architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
    digest_payload = {
        "config": config,
        "tokenizerConfig": tokenizer,
        "adapterConfig": metadata.get("adapterConfig") if isinstance(metadata.get("adapterConfig"), dict) else {},
    }
    metadata_digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    chat_template = tokenizer.get("chat_template")
    chat_text = chat_template if isinstance(chat_template, str) else json.dumps(chat_template, ensure_ascii=False) if isinstance(chat_template, (list, dict)) else ""
    chat_marker = (
        "qwen" if "<|im_start|>" in chat_text
        else "llama3" if "<|start_header_id|>" in chat_text
        else "mistral" if "[INST]" in chat_text and "[/INST]" in chat_text
        else ""
    )
    return {
        "modelPath": str(metadata.get("canonicalPath") or ""),
        "modelName": str(metadata.get("modelName") or ""),
        "modelType": str(config.get("model_type") or ""),
        "architectures": [str(item) for item in architectures if str(item).strip()],
        "hasChatTemplate": isinstance(tokenizer.get("chat_template"), (str, list, dict)),
        "chatTemplateMarker": chat_marker,
        "metadataDigest": metadata_digest,
    }


def _registered_candidates(candidates: Iterable[str], registered: Set[str]) -> List[str]:
    return [item for item in dict.fromkeys(candidates) if item in registered]


def _nothink_template(template: str, metadata: Dict[str, Any], registered: Set[str]) -> str:
    model_name = str(metadata.get("modelName") or os.path.basename(str(metadata.get("canonicalPath") or "")))
    name = str(model_name or "").lower()
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    tokenizer = metadata.get("tokenizerConfig") if isinstance(metadata.get("tokenizerConfig"), dict) else {}
    explicitly_nothink = bool(
        re.search(r"(?:^|[-_.])(nothink|non[-_.]?thinking)(?:$|[-_.])", name)
        or (template == "qwen3" and re.search(r"(?:^|[-_.])instruct[-_.]?2507(?:$|[-_.])", name))
        or config.get("enable_thinking") is False
        or tokenizer.get("enable_thinking") is False
    )
    candidate = f"{template}_nothink"
    return candidate if explicitly_nothink and candidate in registered else template


def _infer_named_template(metadata: Dict[str, Any], registered: Set[str]) -> Dict[str, Any]:
    evidence = _metadata_evidence(metadata)
    model_name = evidence["modelName"]
    model_type = str(evidence["modelType"] or "").lower()
    architectures = {str(item).lower() for item in evidence["architectures"]}

    def chat_template_conflicts(template: str) -> bool:
        marker = str(evidence.get("chatTemplateMarker") or "")
        expected = (
            "qwen" if template == "qwen" or template.startswith("qwen3")
            else "llama3" if template == "llama3"
            else ""
        )
        return bool(marker and expected and marker != expected)

    name_rules = [rule for rule in EXACT_MODEL_RULES if rule["namePattern"].search(model_name)]
    def rule_is_compatible(rule: Dict[str, Any]) -> bool:
        type_matches = bool(model_type and model_type in rule["modelTypes"])
        architecture_matches = bool(architectures & rule["architectures"])
        if model_type and architectures and rule["architectures"]:
            return type_matches and architecture_matches
        return type_matches or architecture_matches

    compatible_rules = [rule for rule in name_rules if rule_is_compatible(rule)]
    if name_rules and not compatible_rules and (model_type or architectures):
        return {
            "status": "error",
            "error_reason": "model_identity_conflict",
            "message": "模型目录名称与 config.json 中的模型类型/架构不一致，已阻止自动选择模板。",
            "evidence": evidence,
            "options": _registered_candidates((rule["template"] for rule in name_rules), registered),
        }
    if len(compatible_rules) == 1:
        rule = compatible_rules[0]
        template = _nothink_template(rule["template"], metadata, registered)
        if chat_template_conflicts(template):
            return {
                "status": "error",
                "error_reason": "model_identity_conflict",
                "message": "tokenizer_config.json 的 chat_template 格式标记与候选模型模板不一致。",
                "evidence": evidence,
                "options": _registered_candidates([template], registered),
            }
        if template not in registered:
            return {
                "status": "error",
                "error_reason": "template_not_available_in_image",
                "message": f"训练容器未注册模型所需模板：{template}。",
                "template": template,
                "evidence": evidence,
            }
        return {
            "status": "resolved",
            "template": template,
            "modelFamily": rule["modelFamily"],
            "source": "exact_model_rule",
            "evidence": evidence,
        }
    if len(compatible_rules) > 1:
        candidates = list(dict.fromkeys(rule["template"] for rule in compatible_rules))
        return {
            "status": "error",
            "error_reason": "template_resolution_ambiguous",
            "message": TEMPLATE_AMBIGUOUS_MESSAGE,
            "options": candidates,
            "availableOptions": _registered_candidates(candidates, registered),
            "evidence": evidence,
        }

    if model_type in UNIQUE_MODEL_TYPE_TEMPLATES:
        template = _nothink_template(UNIQUE_MODEL_TYPE_TEMPLATES[model_type], metadata, registered)
        if chat_template_conflicts(template):
            return {
                "status": "error",
                "error_reason": "model_identity_conflict",
                "message": "tokenizer_config.json 的 chat_template 格式标记与候选模型模板不一致。",
                "evidence": evidence,
                "options": _registered_candidates([template], registered),
            }
        if template not in registered:
            return {
                "status": "error",
                "error_reason": "template_not_available_in_image",
                "message": f"训练容器未注册模型所需模板：{template}。",
                "template": template,
                "evidence": evidence,
            }
        return {
            "status": "resolved",
            "template": template,
            "modelFamily": model_type,
            "source": "unique_model_type",
            "evidence": evidence,
        }

    candidates = list(AMBIGUOUS_MODEL_TYPE_TEMPLATES.get(model_type, []))
    return {
        "status": "error",
        "error_reason": "template_resolution_ambiguous" if candidates else "template_resolution_unknown",
        "message": TEMPLATE_AMBIGUOUS_MESSAGE,
        "options": candidates,
        "availableOptions": _registered_candidates(candidates, registered),
        "evidence": evidence,
    }


def resolve_model_template(
    container: str,
    model_path: str,
    stage: str,
    explicit_template: Any = None,
    *,
    registered_templates: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    _visited: Optional[Set[str]] = None,
    _depth: int = 0,
) -> Dict[str, Any]:
    """Resolve a safe named LLaMAFactory template without loading model code."""
    explicit = normalize_template(explicit_template)
    if explicit == "auto":
        explicit = ""
    if explicit:
        explicit_issue = template_validation_issue(explicit, container=container)
        if explicit_issue:
            return {"status": "error", **explicit_issue}

    registry_source = "provided"
    registry_warning: Optional[Dict[str, Any]] = None

    def annotate_registry(result: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(result)
        annotated["registrySource"] = registry_source
        if registry_warning:
            warnings = [dict(item) for item in annotated.get("warnings", []) if isinstance(item, dict)]
            if not any(item.get("code") == registry_warning.get("code") for item in warnings):
                warnings.append(dict(registry_warning))
            annotated["warnings"] = warnings
        return annotated

    if registered_templates is None:
        registered, registry_error, registry_source, registry_warning = _runtime_template_registry(container)
        if not registered:
            return annotate_registry({
                "status": "error",
                "error_reason": "template_registry_unavailable",
                "message": TEMPLATE_REGISTRY_UNAVAILABLE_MESSAGE,
                "container": container,
                "detail": registry_error or "",
            })
    else:
        registered = {
            normalize_template(item)
            for item in registered_templates
            if normalize_template(item) in SUPPORTED_TEMPLATES
        }

    if explicit and explicit not in registered:
        return annotate_registry({
            "status": "error",
            "error_reason": "template_not_available_in_image",
            "message": f"所选容器未注册显式模板：{explicit}。",
            "template": explicit,
            "container": container,
        })

    if metadata is None:
        metadata, metadata_issue = _read_model_metadata(container, model_path)
        if metadata_issue:
            return annotate_registry({"status": "error", "container": container, **metadata_issue})
    else:
        metadata = dict(metadata)
        canonical = str(metadata.get("canonicalPath") or model_path or "").rstrip("/")
        metadata.setdefault("canonicalPath", canonical)
        metadata.setdefault("modelName", os.path.basename(canonical))
        metadata.setdefault("config", {})
        metadata.setdefault("tokenizerConfig", {})
        metadata.setdefault("adapterConfig", {})

    inferred = _infer_named_template(metadata, registered)
    if explicit:
        result = {
            "status": "resolved",
            "template": explicit,
            "modelFamily": str(inferred.get("modelFamily") or "explicit"),
            "source": "explicit",
            "evidence": _metadata_evidence(metadata),
        }
        inferred_template = str(inferred.get("template") or "") if inferred.get("status") == "resolved" else ""
        if inferred_template and inferred_template != explicit:
            result["warnings"] = [{
                "code": "template_override_conflict",
                "message": f"显式模板 {explicit} 覆盖了自动识别结果 {inferred_template}。",
                "detectedTemplate": inferred_template,
            }]
        return annotate_registry(result)

    adapter = metadata.get("adapterConfig") if isinstance(metadata.get("adapterConfig"), dict) else {}
    base_model = str(adapter.get("base_model_name_or_path") or "").strip()
    if base_model:
        if not base_model.startswith("/"):
            return annotate_registry({
                "status": "error",
                "error_reason": "base_model_unavailable",
                "message": f"adapter 记录的基础模型不是容器内绝对路径：{base_model}。",
                "baseModel": base_model,
                "evidence": _metadata_evidence(metadata),
            })
        visited = set(_visited or set())
        current = str(metadata.get("canonicalPath") or model_path).rstrip("/")
        if current in visited or _depth >= _MAX_ADAPTER_DEPTH:
            return annotate_registry({
                "status": "error",
                "error_reason": "adapter_base_cycle",
                "message": "adapter 基础模型引用存在循环或超过最大递归深度。",
                "modelPath": current,
            })
        visited.add(current)
        base_result = resolve_model_template(
            container,
            base_model,
            stage,
            registered_templates=registered,
            _visited=visited,
            _depth=_depth + 1,
        )
        if base_result.get("status") != "resolved" and base_result.get("error_reason") in {
            "model_path_missing", "model_metadata_unavailable", "model_path_invalid",
        }:
            base_result = dict(base_result)
            base_result["error_reason"] = "base_model_unavailable"
            base_result["message"] = f"adapter 的基础模型不可访问：{base_model}。"
            base_result["baseModel"] = base_model
        if base_result.get("status") == "resolved":
            base_result = dict(base_result)
            base_result["source"] = "adapter_base"
            evidence = dict(base_result.get("evidence") or {})
            evidence.update({"adapterPath": current, "baseModel": base_model})
            base_result["evidence"] = evidence
        return annotate_registry(base_result)

    inferred["container"] = container
    inferred["stage"] = str(stage or "")
    return annotate_registry(inferred)


def verify_model_template_resolution(
    container: str,
    model_path: str,
    stage: str,
    original: Dict[str, Any],
    explicit_template: Any = None,
) -> Optional[Dict[str, Any]]:
    """Re-read model identity immediately before resource allocation."""
    current = resolve_model_template(
        container,
        model_path,
        stage,
        explicit_template=explicit_template,
    )
    if current.get("status") != "resolved":
        return {
            "error_reason": "model_changed_during_preflight",
            "message": "模型路径或元数据在启动预检期间发生变化，已阻止启动。",
            "previous": original,
            "current": current,
        }
    previous_evidence = original.get("evidence") if isinstance(original.get("evidence"), dict) else {}
    current_evidence = current.get("evidence") if isinstance(current.get("evidence"), dict) else {}
    identity_keys = (
        "modelPath", "modelName", "modelType", "architectures", "metadataDigest",
        "adapterPath", "baseModel",
    )
    previous_identity = {key: previous_evidence.get(key) for key in identity_keys}
    current_identity = {key: current_evidence.get(key) for key in identity_keys}
    if original.get("template") != current.get("template") or previous_identity != current_identity:
        return {
            "error_reason": "model_changed_during_preflight",
            "message": "模型路径或元数据在启动预检期间发生变化，已阻止启动。",
            "previous": {
                "template": original.get("template"),
                "source": original.get("source"),
                "evidence": previous_identity,
            },
            "current": {
                "template": current.get("template"),
                "source": current.get("source"),
                "evidence": current_identity,
            },
        }
    return None
