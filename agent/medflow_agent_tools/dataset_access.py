"""Runtime-side enforcement for Studio-issued dataset access manifests."""

from __future__ import annotations

import posixpath
import logging
import subprocess
from typing import Any, Iterable


class DatasetAccessError(ValueError):
    pass


logger = logging.getLogger(__name__)


def canonical_dataset_path(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw.startswith("/") or ".." in raw.split("/"):
        raise DatasetAccessError("数据集路径必须是无目录穿越的绝对路径")
    normalized = posixpath.normpath(raw)
    if normalized != raw:
        raise DatasetAccessError("数据集路径未规范化")
    return normalized


def allowed_dataset_entries(
    manifest: Any,
    container: str,
    dataset_types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(manifest, (list, tuple)):
        raise DatasetAccessError("缺少数据集授权清单")
    wanted = {str(value).strip().lower() for value in (dataset_types or [])}
    result: list[dict[str, Any]] = []
    for raw in manifest:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("storageState") or "").strip().lower() != "ready" or raw.get("containerName") != container:
            continue
        dataset_type = str(raw.get("datasetType") or "").strip().lower()
        if wanted and dataset_type not in wanted:
            continue
        try:
            source_path = canonical_dataset_path(raw.get("sourcePath"))
        except DatasetAccessError:
            continue
        result.append({**raw, "sourcePath": source_path, "datasetType": dataset_type})
    return result


def require_allowed_dataset_path(
    path: Any,
    manifest: Any,
    container: str,
    dataset_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    requested = canonical_dataset_path(path)
    for entry in allowed_dataset_entries(manifest, container, dataset_types):
        if entry["sourcePath"] == requested:
            assert_dataset_realpath(container, requested)
            return entry
    raise DatasetAccessError(f"数据集不在当前用户的授权清单中：{requested}")


def assert_dataset_realpath(container: str, path: str) -> None:
    process = subprocess.run(
        ["docker", "exec", container, "realpath", "-e", path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    resolved = (process.stdout or "").strip().rstrip("/")
    if process.returncode != 0 or resolved != path:
        raise DatasetAccessError(f"数据集不存在或包含符号链接逃逸：{path}")


def latest_dataset_by_file_mtime(
    manifest: Any,
    container: str,
    dataset_types: Iterable[str],
) -> str:
    entries = allowed_dataset_entries(manifest, container, dataset_types)
    manifest_count = len(manifest) if isinstance(manifest, (list, tuple)) else 0
    logger.info(
        "Dataset manifest selection: container=%s types=%s manifest_count=%d allowed_count=%d",
        container,
        sorted({str(value).strip().lower() for value in dataset_types}),
        manifest_count,
        len(entries),
    )
    if not entries:
        raise DatasetAccessError(
            f"授权清单中没有可用于容器 {container} 的 raw 数据集（清单共 {manifest_count} 项）"
        )
    best: tuple[int, str] | None = None
    script = (
        'find "$1" -maxdepth 1 -type f '
        '\\( -name "*.json" -o -name "*.jsonl" -o -name "*.parquet" -o -name "*.txt" \\) '
        '-printf "%T@\\n" 2>/dev/null | sort -nr | head -n 1'
    )
    for entry in entries:
        path = entry["sourcePath"]
        assert_dataset_realpath(container, path)
        process = subprocess.run(
            ["docker", "exec", container, "sh", "-c", script, "dataset-mtime", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if process.returncode != 0 or not process.stdout.strip():
            continue
        try:
            mtime = int(float(process.stdout.strip().splitlines()[0]))
        except ValueError:
            continue
        candidate = (mtime, path)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise DatasetAccessError(
            f"授权清单中的 {len(entries)} 个 raw 数据集均未找到有效数据文件"
        )
    return best[1]


def latest_dataset_by_name(
    manifest: Any,
    container: str,
    dataset_types: Iterable[str],
) -> str:
    entries = allowed_dataset_entries(manifest, container, dataset_types)
    if not entries:
        raise DatasetAccessError("授权清单中没有可用于此训练类型的数据集")
    selected = max(entries, key=lambda entry: (str(entry.get("datasetName") or ""), entry["sourcePath"]))
    assert_dataset_realpath(container, selected["sourcePath"])
    return selected["sourcePath"]
