# -*- coding: utf-8 -*-
"""Load and validate the shared training descriptor manifest."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1
MANIFEST_ENV = "MEDFLOW_TRAINING_DESCRIPTOR_PATH"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "training_descriptors.v1.json"
)

_ENUMS = {
    "backend": {"llamafactory", "verl"},
    "stage": {"sft", "pt", "dpo", "grpo"},
    "finetuningType": {"lora", "full", "rl"},
    "invocationStyle": {"env", "argv", "keyValue"},
    "paramType": {
        "text", "number", "boolean", "enum", "model", "dataset", "file",
        "template", "checkpoint", "argvRemainder",
    },
    "containerRole": {"training", "multinodeTraining", "grpoTraining"},
    "artifact": {"adapter", "checkpoint", "merged"},
    "deployable": {"merged", "checkpoint", "none"},
}

_DESCRIPTOR_FIELDS = {
    "id", "displayName", "aliases", "backend", "stage", "finetuningType",
    "entry", "params", "resourceRequirements", "paths", "monitoring",
    "artifactPolicy",
}


def _reject_unknown(value: Dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {', '.join(unknown)}")


def _normalize(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _strings(value: Any, label: str, *, nonempty: bool = False) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = [_text(item, f"{label} item") for item in value]
    if nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _identity_values(descriptor: Dict[str, Any]) -> List[str]:
    entry = descriptor["entry"]
    source = entry["sourceScript"]
    command = entry["containerCommand"]
    return [
        descriptor["id"], descriptor["displayName"], source,
        source.rsplit("/", 1)[-1].removesuffix(".py"), command,
        command.rstrip("/").rsplit("/", 1)[-1], *descriptor["aliases"],
    ]


def _validate_params(descriptor: Dict[str, Any]) -> None:
    descriptor_id = descriptor["id"]
    params = descriptor.get("params")
    if not isinstance(params, list):
        raise ValueError(f"training descriptor {descriptor_id} params must be a list")
    seen: set[str] = set()
    for param in params:
        if not isinstance(param, dict):
            raise ValueError(f"training descriptor {descriptor_id} contains a non-object param")
        _reject_unknown(
            param,
            {
                "name", "label", "type", "required", "aliases", "default",
                "options", "multiple", "advanced", "resourceOnly",
            },
            f"training descriptor {descriptor_id} param",
        )
        name = _text(param.get("name"), f"training descriptor {descriptor_id} param name")
        if name in seen:
            raise ValueError(f"training descriptor {descriptor_id} has duplicate param name: {name}")
        seen.add(name)
        param_type = _text(param.get("type"), f"training descriptor {descriptor_id} param {name} type")
        if param_type not in _ENUMS["paramType"]:
            raise ValueError(f"training descriptor {descriptor_id} param {name} has unsupported type: {param_type}")
        if not isinstance(param.get("required"), bool):
            raise ValueError(f"training descriptor {descriptor_id} param {name} required must be boolean")
        for field in ("advanced", "resourceOnly", "multiple"):
            if field in param and not isinstance(param[field], bool):
                raise ValueError(f"training descriptor {descriptor_id} param {name} {field} must be boolean")
        if "label" in param:
            _text(param["label"], f"training descriptor {descriptor_id} param {name} label")
        if "aliases" in param:
            _strings(param["aliases"], f"training descriptor {descriptor_id} param {name} aliases")
        if "options" in param:
            options = _strings(param["options"], f"training descriptor {descriptor_id} param {name} options", nonempty=True)
            if param_type != "enum":
                raise ValueError(f"training descriptor {descriptor_id} param {name} options require enum type")
            if "default" in param and param["default"] not in options:
                raise ValueError(f"training descriptor {descriptor_id} param {name} default must be one of options")
        if "default" in param:
            default = param["default"]
            if not isinstance(default, (str, int, float, bool, list)):
                raise ValueError(f"training descriptor {descriptor_id} param {name} has unsupported default value")
            if isinstance(default, list) and any(not isinstance(item, str) for item in default):
                raise ValueError(f"training descriptor {descriptor_id} param {name} default list must contain strings")


def validate_training_descriptors(
    descriptors: Optional[List[Dict[str, Any]]] = None,
    *,
    allow_derived: bool = True,
) -> None:
    values = _TRAINING_DESCRIPTORS if descriptors is None else descriptors
    if not isinstance(values, list) or not values:
        raise ValueError("training descriptor manifest must contain at least one descriptor")
    seen_ids: set[str] = set()
    identity_owners: Dict[str, str] = {}

    for descriptor in values:
        if not isinstance(descriptor, dict):
            raise ValueError("training descriptor must be an object")
        allowed_fields = (
            _DESCRIPTOR_FIELDS | {"requiredParams", "resourceSelectors"}
            if allow_derived
            else _DESCRIPTOR_FIELDS
        )
        _reject_unknown(descriptor, allowed_fields, "training descriptor")
        descriptor_id = _text(descriptor.get("id"), "training descriptor id")
        if descriptor_id in seen_ids:
            raise ValueError(f"duplicate training descriptor id: {descriptor_id}")
        seen_ids.add(descriptor_id)
        _text(descriptor.get("displayName"), f"training descriptor {descriptor_id} displayName")
        descriptor["aliases"] = _strings(descriptor.get("aliases"), f"training descriptor {descriptor_id} aliases")

        for field in ("backend", "stage", "finetuningType"):
            value = _text(descriptor.get(field), f"training descriptor {descriptor_id} {field}")
            if value not in _ENUMS[field]:
                raise ValueError(f"training descriptor {descriptor_id} has unsupported {field}: {value}")

        entry = descriptor.get("entry")
        if not isinstance(entry, dict):
            raise ValueError(f"training descriptor {descriptor_id} is missing entry configuration")
        _reject_unknown(
            entry,
            {"sourceScript", "containerCommand", "invocationStyle"},
            f"training descriptor {descriptor_id} entry",
        )
        _text(entry.get("sourceScript"), f"training descriptor {descriptor_id} sourceScript")
        _text(entry.get("containerCommand"), f"training descriptor {descriptor_id} containerCommand")
        invocation = _text(entry.get("invocationStyle"), f"training descriptor {descriptor_id} invocationStyle")
        if invocation not in _ENUMS["invocationStyle"]:
            raise ValueError(f"training descriptor {descriptor_id} has unsupported invocationStyle: {invocation}")
        _validate_params(descriptor)
        if "requiredParams" in descriptor:
            required_params = _strings(
                descriptor["requiredParams"],
                f"training descriptor {descriptor_id} requiredParams",
            )
            expected = [param["name"] for param in descriptor["params"] if param["required"]]
            if required_params != expected:
                raise ValueError(f"training descriptor {descriptor_id} requiredParams are inconsistent")

        resources = descriptor.get("resourceRequirements")
        if not isinstance(resources, dict):
            raise ValueError(f"training descriptor {descriptor_id} is missing resource configuration")
        _reject_unknown(
            resources,
            {"gpuCount", "nodeCount", "gpusPerNode", "containerRole", "resourcePool"},
            f"training descriptor {descriptor_id} resourceRequirements",
        )
        role = _text(resources.get("containerRole"), f"training descriptor {descriptor_id} containerRole")
        if role not in _ENUMS["containerRole"]:
            raise ValueError(f"training descriptor {descriptor_id} has unsupported containerRole: {role}")
        for field in ("gpuCount", "nodeCount", "gpusPerNode"):
            if field in resources:
                _positive_int(resources[field], f"training descriptor {descriptor_id} {field}")
        if "resourcePool" in resources and not isinstance(resources["resourcePool"], bool):
            raise ValueError(f"training descriptor {descriptor_id} resourcePool must be boolean")
        if "resourceSelectors" in descriptor:
            resource_selectors = _strings(
                descriptor["resourceSelectors"],
                f"training descriptor {descriptor_id} resourceSelectors",
            )
            expected = [role, *(["trainingPool"] if resources.get("resourcePool") else [])]
            if resource_selectors != expected:
                raise ValueError(f"training descriptor {descriptor_id} resourceSelectors are inconsistent")

        paths = descriptor.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"training descriptor {descriptor_id} paths must be an object")
        _reject_unknown(
            paths,
            {"defaultDatasetRoot", "outputRoot", "logRoot", "exportRoot"},
            f"training descriptor {descriptor_id} paths",
        )
        for field, value in paths.items():
            _text(value, f"training descriptor {descriptor_id} paths.{field}")

        monitoring = descriptor.get("monitoring")
        if not isinstance(monitoring, dict):
            raise ValueError(f"training descriptor {descriptor_id} monitoring must be an object")
        _reject_unknown(
            monitoring,
            {"primaryMetric", "fallbackMetrics", "logFiles", "wandbDir", "completionStates"},
            f"training descriptor {descriptor_id} monitoring",
        )
        _text(monitoring.get("primaryMetric"), f"training descriptor {descriptor_id} primaryMetric")
        for field in ("fallbackMetrics", "logFiles", "completionStates"):
            if field in monitoring:
                _strings(monitoring[field], f"training descriptor {descriptor_id} {field}")
        if "wandbDir" in monitoring:
            _text(monitoring["wandbDir"], f"training descriptor {descriptor_id} wandbDir")

        policy = descriptor.get("artifactPolicy")
        if not isinstance(policy, dict):
            raise ValueError(f"training descriptor {descriptor_id} artifactPolicy must be an object")
        _reject_unknown(
            policy,
            {"produces", "deployable"},
            f"training descriptor {descriptor_id} artifactPolicy",
        )
        produces = _strings(policy.get("produces"), f"training descriptor {descriptor_id} artifactPolicy.produces", nonempty=True)
        invalid = [item for item in produces if item not in _ENUMS["artifact"]]
        if invalid:
            raise ValueError(f"training descriptor {descriptor_id} has unsupported artifacts: {', '.join(invalid)}")
        deployable = _text(policy.get("deployable"), f"training descriptor {descriptor_id} artifactPolicy.deployable")
        if deployable not in _ENUMS["deployable"]:
            raise ValueError(f"training descriptor {descriptor_id} has unsupported deployable type: {deployable}")

        for identity in _identity_values(descriptor):
            normalized = _normalize(identity)
            owner = identity_owners.get(normalized)
            if owner is not None and owner != descriptor_id:
                raise ValueError(f"training descriptor alias conflict: {identity!r} belongs to {owner} and {descriptor_id}")
            identity_owners[normalized] = descriptor_id


def training_descriptor_manifest_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv(MANIFEST_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MANIFEST_PATH


def load_training_descriptors(path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    manifest_path = training_descriptor_manifest_path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"training descriptor manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"training descriptor manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("training descriptor manifest must be a JSON object")
    _reject_unknown(payload, {"schemaVersion", "descriptors"}, "training descriptor manifest")
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"unsupported training descriptor schemaVersion: {payload.get('schemaVersion')!r}")
    raw = payload.get("descriptors")
    if not isinstance(raw, list):
        raise ValueError("training descriptor manifest descriptors must be a list")

    descriptors = deepcopy(raw)
    validate_training_descriptors(descriptors, allow_derived=False)
    for descriptor in descriptors:
        descriptor["requiredParams"] = [param["name"] for param in descriptor["params"] if param["required"]]
        resources = descriptor["resourceRequirements"]
        descriptor["resourceSelectors"] = [
            resources["containerRole"],
            *(["trainingPool"] if resources.get("resourcePool") else []),
        ]
    return descriptors


_TRAINING_DESCRIPTORS = load_training_descriptors()


def list_training_descriptors() -> List[Dict[str, Any]]:
    return deepcopy(_TRAINING_DESCRIPTORS)


def find_training_descriptor(value: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize(value)
    if not normalized:
        return None
    for descriptor in _TRAINING_DESCRIPTORS:
        if any(_normalize(alias) == normalized for alias in _identity_values(descriptor)):
            return deepcopy(descriptor)
    return None
