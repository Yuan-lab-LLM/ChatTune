#!/usr/bin/env python3
"""Migrate stopped legacy inference-agent state into Inference Ops state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {"starting", "running", "degraded", "submitting", "pending"}


def rewrite_value(value: Any, rewrites: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_value(item, rewrites) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_value(item, rewrites) for item in value]
    if isinstance(value, str):
        for source, target in rewrites:
            value = value.replace(source, target)
        return value
    return value


def json_records(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            yield path, payload


def jsonl_records(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield path, line_number, payload


def state_records(root: Path):
    for path, payload in json_records(root):
        yield str(path), payload
    for path, line_number, payload in jsonl_records(root):
        yield f"{path}:{line_number}", payload


def active_records(root: Path) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    for path, payload in state_records(root):
        status = str(
            payload.get("status") or payload.get("launch_status") or ""
        ).strip().lower()
        if status in ACTIVE_STATUSES:
            active.append(
                {
                    "path": path,
                    "status": status,
                    "id": str(
                        payload.get("instance_id")
                        or payload.get("job_id")
                        or payload.get("test_run_id")
                        or payload.get("run_id")
                        or ""
                    ),
                }
            )
    return active


def migrate(
    source: Path,
    target: Path,
    apply: bool,
    extra_rewrites: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    # Keep the spelling supplied by the deployment environment. On Windows,
    # resolve() may expand an 8.3 path and prevent exact metadata rewriting.
    source = source.absolute()
    target = target.absolute()
    if source == target:
        raise ValueError("legacy and target state directories must differ")
    if not source.is_dir():
        raise ValueError(f"legacy state directory does not exist: {source}")
    rewrites = [(str(source), str(target)), *(extra_rewrites or [])]

    active = active_records(source)
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "legacy_state": str(source),
        "target_state": str(target),
        "active_records": active,
        "json_records": sum(1 for _ in state_records(source)),
        "files": sum(1 for path in source.rglob("*") if path.is_file()),
    }
    if active:
        summary["status"] = "blocked"
        summary["reason"] = "legacy state still contains active records"
        return summary
    if not apply:
        summary["status"] = "ready"
        return summary

    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.migrate-", dir=target.parent))
    staged = temp_root / target.name
    backup = None
    try:
        shutil.copytree(source, staged)
        for path, payload in list(json_records(staged)):
            rewritten = rewrite_value(payload, rewrites)
            if isinstance(rewritten, dict):
                rewritten.setdefault("schema_version", 1)
                rewritten.setdefault("provider", "medflow")
            path.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for path in staged.rglob("*.jsonl"):
            source_text = path.read_text(encoding="utf-8")
            rewritten_lines = []
            for line in source_text.splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    rewritten_lines.append(line)
                    continue
                rewritten = rewrite_value(payload, rewrites)
                if isinstance(rewritten, dict):
                    rewritten.setdefault("schema_version", 1)
                    rewritten.setdefault("provider", "medflow")
                rewritten_lines.append(json.dumps(rewritten, ensure_ascii=False))
            path.write_text(
                "\n".join(rewritten_lines) + ("\n" if source_text.endswith("\n") else ""),
                encoding="utf-8",
            )
        for suffix in ("*.yaml", "*.yml"):
            for path in staged.rglob(suffix):
                text = path.read_text(encoding="utf-8")
                for old, new in rewrites:
                    text = text.replace(old, new)
                path.write_text(text, encoding="utf-8")
        if target.exists():
            backup = target.with_name(f"{target.name}.backup-{timestamp}")
            if backup.exists():
                raise ValueError(f"backup path already exists: {backup}")
            target.replace(backup)
        staged.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    summary["status"] = "migrated"
    summary["backup"] = str(backup) if backup else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-state", required=True, type=Path)
    parser.add_argument("--target-state", required=True, type=Path)
    parser.add_argument(
        "--rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="additional absolute path prefix to rewrite; may be repeated",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        rewrites = []
        for item in args.rewrite:
            old, separator, new = item.partition("=")
            if not separator or not old or not new:
                raise ValueError(f"invalid --rewrite value: {item}")
            rewrites.append((old, new))
        result = migrate(
            args.legacy_state,
            args.target_state,
            args.apply,
            extra_rewrites=rewrites,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 3 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
