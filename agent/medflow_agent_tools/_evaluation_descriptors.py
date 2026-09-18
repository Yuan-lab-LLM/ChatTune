# -*- coding: utf-8 -*-
"""Read-only descriptors for the existing evaluation entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


EVALUATION_DESCRIPTORS: List[Dict[str, Any]] = [
    {
        "id": "single_model_evaluation_vpn",
        "displayName": "内置医疗单模型评估",
        "aliases": ["single_model_evaluation_vpn", "单模型评估"],
        "entry": {"sourceScript": "single_model_evaluation_vpn.py", "containerCommand": "/usr/local/insinfersystem/single_model_evaluation_vpn", "invocationStyle": "argv"},
        "params": [{"name": "model_fir", "label": "模型路径", "type": "model", "required": True}],
        "requiredParams": ["model_fir"],
        "resourceSelectors": ["evaluation"],
        "paths": {"logRoot": "/home/workspace/log/daily_train", "reportRoot": "/home/workspace/eval/daily_train"},
        "evaluationScope": "legacyMedical",
        "supportsCustomDataset": False,
        "builtInSuites": ["medical_eval", "common_medical_subjects", "diagnosis_accuracy"],
    },
    {
        "id": "compare_between_models_vpn",
        "displayName": "内置医疗双模型评估",
        "aliases": ["compare_between_models_vpn", "双模型评估", "双模型对比评估"],
        "entry": {"sourceScript": "compare_between_models_vpn.py", "containerCommand": "/usr/local/insinfersystem/compare_between_models_vpn", "invocationStyle": "argv"},
        "params": [
            {"name": "model_fir", "label": "第一个模型路径", "type": "model", "required": True},
            {"name": "model_sec", "label": "第二个模型路径", "type": "model", "required": True},
        ],
        "requiredParams": ["model_fir", "model_sec"],
        "resourceSelectors": ["evaluation"],
        "paths": {"logRoot": "/home/workspace/log/daily_train", "reportRoot": "/home/workspace/eval/daily_train"},
        "evaluationScope": "legacyMedical",
        "supportsCustomDataset": False,
        "builtInSuites": ["medical_eval", "common_medical_subjects", "medical_inference_compare"],
    },
    {
        "id": "ckpt_eval",
        "displayName": "内置医疗 Checkpoint 评估",
        "aliases": ["ckpt_eval", "checkpoint assessment", "ckpt assessment", "checkpoint评估", "ckpt评估"],
        "entry": {"sourceScript": "ckpt_eval.py", "containerCommand": "/usr/local/insinfersystem/ckpt_eval", "invocationStyle": "env"},
        "params": [
            {"name": "CKPT_PATH", "label": "Checkpoint 路径", "type": "checkpoint", "required": True},
            {"name": "TEM", "label": "模型模板（自动识别）", "type": "template", "required": False, "advanced": True},
        ],
        "requiredParams": ["CKPT_PATH"],
        "resourceSelectors": ["evaluation", "checkpoint"],
        "paths": {"logRoot": "/home/workspace/log/daily_train", "reportRoot": "/home/workspace/eval/daily_train"},
        "evaluationScope": "legacyMedical",
        "supportsCustomDataset": False,
        "builtInSuites": ["medical_eval", "common_medical_subjects", "medical_inference_compare"],
    },
    {
        "id": "generic_single_model_eval",
        "displayName": "LLaMA-Factory 通用单模型评测",
        "aliases": ["generic_single_model_eval", "通用单模型评测", "通用单模型评估"],
        "entry": {
            "sourceScript": "generic_single_model_eval.py",
            "containerCommand": "/usr/local/insinfersystem/generic_single_model_eval",
            "invocationStyle": "argv",
        },
        "params": [
            {"name": "model_fir", "label": "模型路径", "type": "model", "required": True},
            {
                "name": "eval_datasets",
                "label": "内置评测集",
                "type": "enum",
                "required": True,
                "multiple": True,
                "options": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
            },
        ],
        "requiredParams": ["model_fir", "eval_datasets"],
        "resourceSelectors": ["model"],
        "paths": {
            "logRoot": "/home/workspace/log/llamafactory_eval",
            "reportRoot": "/home/workspace/eval/llamafactory",
        },
        "evaluationScope": "llamafactoryBuiltin",
        "supportsCustomDataset": False,
        "builtInSuites": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
    },
    {
        "id": "generic_compare_models_eval",
        "displayName": "LLaMA-Factory 通用双模型评测",
        "aliases": ["generic_compare_models_eval", "通用双模型评测", "通用双模型评估"],
        "entry": {
            "sourceScript": "generic_compare_models_eval.py",
            "containerCommand": "/usr/local/insinfersystem/generic_compare_models_eval",
            "invocationStyle": "argv",
        },
        "params": [
            {"name": "model_fir", "label": "第一个模型路径", "type": "model", "required": True},
            {"name": "model_sec", "label": "第二个模型路径", "type": "model", "required": True},
            {
                "name": "eval_datasets",
                "label": "内置评测集",
                "type": "enum",
                "required": True,
                "multiple": True,
                "options": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
            },
        ],
        "requiredParams": ["model_fir", "model_sec", "eval_datasets"],
        "resourceSelectors": ["model"],
        "paths": {
            "logRoot": "/home/workspace/log/llamafactory_eval",
            "reportRoot": "/home/workspace/eval/llamafactory",
        },
        "evaluationScope": "llamafactoryBuiltin",
        "supportsCustomDataset": False,
        "builtInSuites": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
    },
    {
        "id": "generic_ckpt_eval",
        "displayName": "LLaMA-Factory 通用 Checkpoint 评测",
        "aliases": ["generic_ckpt_eval", "通用checkpoint评测", "通用ckpt评测", "通用checkpoint评估", "通用ckpt评估"],
        "entry": {
            "sourceScript": "generic_ckpt_eval.py",
            "containerCommand": "/usr/local/insinfersystem/generic_ckpt_eval",
            "invocationStyle": "argv",
        },
        "params": [
            {"name": "CKPT_PATH", "label": "Checkpoint 路径", "type": "checkpoint", "required": True},
            {"name": "TEM", "label": "模型模板（自动识别）", "type": "template", "required": False, "advanced": True},
            {
                "name": "eval_datasets",
                "label": "内置评测集",
                "type": "enum",
                "required": True,
                "multiple": True,
                "options": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
            },
        ],
        "requiredParams": ["CKPT_PATH", "eval_datasets"],
        "resourceSelectors": ["checkpoint"],
        "paths": {
            "logRoot": "/home/workspace/log/llamafactory_eval",
            "reportRoot": "/home/workspace/eval/llamafactory",
        },
        "evaluationScope": "llamafactoryBuiltin",
        "supportsCustomDataset": False,
        "builtInSuites": ["ceval", "cmbsingle", "cmmlu", "medmcqa", "medqa", "mmlu"],
    },
]

for _descriptor in EVALUATION_DESCRIPTORS:
    _default_scope = "general" if _descriptor["evaluationScope"] == "llamafactoryBuiltin" else "medical"
    _descriptor["params"].insert(
        0,
        {
            "name": "evaluation_scope",
            "label": "评测范围",
            "type": "enum",
            "required": False,
            "default": _default_scope,
            "options": ["medical", "general"],
        },
    )


def list_evaluation_descriptors() -> List[Dict[str, Any]]:
    return deepcopy(EVALUATION_DESCRIPTORS)


def find_evaluation_descriptor(value: str) -> Optional[Dict[str, Any]]:
    normalized = "".join(str(value or "").strip().lower().split())
    if not normalized:
        return None
    for descriptor in EVALUATION_DESCRIPTORS:
        entry = descriptor["entry"]
        candidates = [
            descriptor["id"], descriptor["displayName"], entry["sourceScript"],
            entry["sourceScript"].removesuffix(".py"), entry["containerCommand"],
            entry["containerCommand"].rstrip("/").split("/")[-1], *descriptor["aliases"],
        ]
        if any("".join(str(candidate).strip().lower().split()) == normalized for candidate in candidates):
            return deepcopy(descriptor)
    return None
