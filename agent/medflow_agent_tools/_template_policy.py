# -*- coding: utf-8 -*-
"""Shared LLaMAFactory template policy for train/evaluate tools."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


DEFAULT_TEMPLATE = "qwen3"
NON_QWEN3_TEMPLATE_REQUIRED_MESSAGE = (
    "当前版本默认支持 Qwen3 训练与评估；如果使用其他模型，请显式提供 "
    "TEM/template，例如 TEM=llama3。"
)
UNSUPPORTED_TEMPLATE_MESSAGE = "当前版本暂不支持该模型/template。"

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
    "model",
    "model_name",
    "模型名",
    "模型名称",
}

_NON_QWEN3_MARKERS = (
    "baichuan",
    "bloom",
    "chatglm",
    "command-r",
    "cohere",
    "deepseek",
    "falcon",
    "gemma",
    "glm-4",
    "glm4",
    "glm-z1",
    "glmz1",
    "gpt-2",
    "granite",
    "hunyuan",
    "index",
    "internlm",
    "intern-vl",
    "internvl",
    "kimi-vl",
    "llama",
    "llava",
    "mimo",
    "minicpm",
    "ministral",
    "mistral",
    "mixtral",
    "olmo",
    "paligemma",
    "phi",
    "pixtral",
    "qwen1",
    "qwen2",
    "qwen-1",
    "qwen-2",
    "qwq",
    "qvq",
    "skywork",
    "starcoder",
    "telechat",
    "xverse",
    "yi-",
    "yi/",
    "yuan",
)


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


def text_mentions_non_qwen3_model(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values if value is not None).lower()
    if not text:
        return False
    compact = re.sub(r"[\s_]+", "-", text)
    if "qwen3" in compact or "qwen-3" in compact:
        return False
    return any(marker in compact for marker in _NON_QWEN3_MARKERS)


def non_qwen3_template_required_issue(
    *,
    param: str = "TEM/template",
    container: str = "",
    model_hint: str = "",
) -> Dict[str, Any]:
    issue: Dict[str, Any] = {
        "error_reason": "template_required_for_non_qwen3_model",
        "message": NON_QWEN3_TEMPLATE_REQUIRED_MESSAGE,
        "container": container,
        "param": param,
    }
    if model_hint:
        issue["modelHint"] = model_hint
    return issue


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
