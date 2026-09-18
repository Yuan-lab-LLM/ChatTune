# -*- coding: utf-8 -*-
"""Shared, read-only resource directory catalog for the node resource API.

The optional ``MEDFLOW_RESOURCE_CATALOG_CONFIG`` value uses the same JSON shape
returned by Studio's ``queryResourceCatalogConfig`` endpoint.  Invalid entries
fail fast during process startup so delete/upload operations can never fall back
to an unexpected directory.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import PurePosixPath
from typing import Any, Dict


DEFAULT_RESOURCE_CATALOG: Dict[str, Any] = {
    "datasets": [
        {"id": "raw", "path": "/home/workspace/dataset", "aliases": ["raw"], "dataTypes": ["raw", "custom"]},
        {"id": "sft", "path": "/home/workspace/dataset_batch_train", "aliases": ["batch_train", "sft"], "dataTypes": ["sft"]},
        {"id": "dpo", "path": "/home/workspace/dataset_daily_train", "aliases": ["daily_train", "dpo"], "dataTypes": ["dpo"]},
        {"id": "pt", "path": "/home/workspace/dataset_pretrain", "aliases": ["text", "pretrain", "pt"], "dataTypes": ["pt"]},
        {"id": "rl", "path": "/home/workspace/verl/examples/data_preprocess/data", "aliases": ["grpo", "verl"], "dataTypes": ["rl"]},
        {"id": "eval", "path": "/home/workspace/eval", "aliases": ["evaluation", "medicalTest"], "dataTypes": ["eval"]},
    ],
    "models": [
        {"id": "base", "path": "/home/workspace/models/base", "modelTypes": ["base"], "legacyTypes": ["base_train"]},
        {"id": "batchTrain", "path": "/home/workspace/models/batch_train", "modelTypes": ["adapter", "merged", "checkpoint"], "legacyTypes": ["batch_trained"]},
        {"id": "pretrain", "path": "/home/workspace/models/pretrain", "modelTypes": ["adapter", "merged", "checkpoint"], "legacyTypes": ["pretrain"]},
        {"id": "dpoSaves", "path": "/home/workspace/models/dpo_train/internal/saves", "modelTypes": ["adapter", "checkpoint"], "legacyTypes": ["daily_trained"]},
        {"id": "dpoExport", "path": "/home/workspace/models/dpo_train/internal/export", "modelTypes": ["merged"], "legacyTypes": ["daily_trained"]},
        {"id": "inference", "path": "/home/workspace/medical_models", "modelTypes": ["inference", "merged"], "legacyTypes": ["inference"]},
        {"id": "grpo", "path": "/home/workspace/models/grpo_train", "modelTypes": ["checkpoint", "reward"], "legacyTypes": ["grpo"]},
    ],
    "evaluation": [
        {"id": "evaluationDataset", "legacyTypes": ["medicalTest"], "categories": ["medical", "general", "custom"], "path": "/home/workspace/tests"}
    ],
}


def _managed_path(value: Any, label: str) -> str:
    path = str(value or "").strip().replace("\\", "/")
    normalized = str(PurePosixPath(path))
    if not path.startswith("/") or normalized == "/" or ".." in PurePosixPath(path).parts:
        raise ValueError(f"{label} must be an absolute managed directory")
    return normalized.rstrip("/")


def _merge_catalog(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(defaults)
    for section in ("datasets", "models", "evaluation"):
        supplied = override.get(section)
        if supplied is None:
            continue
        if not isinstance(supplied, list):
            raise ValueError(f"resource catalog {section} must be a list")
        by_id = {str(item.get("id")): deepcopy(item) for item in result[section]}
        order = [str(item.get("id")) for item in result[section]]
        for item in supplied:
            if not isinstance(item, dict) or not str(item.get("id") or "").strip():
                raise ValueError(f"resource catalog {section} entry requires id")
            item_id = str(item["id"]).strip()
            merged = {**by_id.get(item_id, {}), **item, "id": item_id}
            merged["path"] = _managed_path(merged.get("path"), f"{section}.{item_id}.path")
            by_id[item_id] = merged
            if item_id not in order:
                order.append(item_id)
        result[section] = [by_id[item_id] for item_id in order]
    return result


def load_resource_catalog_config(raw: str | None = None) -> Dict[str, Any]:
    encoded = os.getenv("MEDFLOW_RESOURCE_CATALOG_CONFIG", "") if raw is None else raw
    if not str(encoded or "").strip():
        return deepcopy(DEFAULT_RESOURCE_CATALOG)
    try:
        override = json.loads(str(encoded))
    except json.JSONDecodeError as exc:
        raise ValueError("MEDFLOW_RESOURCE_CATALOG_CONFIG must be valid JSON") from exc
    if not isinstance(override, dict):
        raise ValueError("MEDFLOW_RESOURCE_CATALOG_CONFIG must be a JSON object")
    return _merge_catalog(DEFAULT_RESOURCE_CATALOG, override)


def resource_path_maps(catalog: Dict[str, Any]) -> Dict[str, Any]:
    dataset_paths = {
        str(item["id"]): str(item["path"])
        for item in catalog["datasets"]
        if str(item.get("id")) not in {"rl", "eval"}
    }
    model_paths: Dict[str, str] = {}
    model_type_hints: Dict[str, list[str]] = {}
    model_by_id = {str(item["id"]): item for item in catalog["models"]}
    for item in catalog["models"]:
        for legacy_type in item.get("legacyTypes") or [item["id"]]:
            legacy_name = str(legacy_type)
            model_paths.setdefault(legacy_name, str(item["path"]))
            model_type_hints.setdefault(
                legacy_name,
                [str(value) for value in item.get("modelTypes") or ["custom"]],
            )
    dpo_paths = {
        "saves": str(model_by_id["dpoSaves"]["path"]),
        "export": str(model_by_id["dpoExport"]["path"]),
    }
    dataset_by_id = {str(item["id"]): item for item in catalog["datasets"]}
    evaluation = catalog["evaluation"][0]
    return {
        "datasets": dataset_paths,
        "models": model_paths,
        "modelTypes": model_type_hints,
        "dpo": dpo_paths,
        "grpoModel": str(model_by_id["grpo"]["path"]),
        "grpoData": str(dataset_by_id["rl"]["path"]),
        "evaluation": str(evaluation["path"]),
    }
