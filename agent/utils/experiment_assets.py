"""Content manifests and managed immutable copies; usable inside task containers.

CLI input is JSON on stdin. This file can be copied into existing containers;
it does not install packages or import the Agent runtime.
"""
from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from typing import Any

def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_references(source: str | Path) -> None:
    root = Path(source)
    info = root / 'dataset_info.json'
    if not root.is_dir() or not info.is_file():
        return
    entries = json.loads(info.read_text(encoding='utf-8'))
    if not isinstance(entries, dict):
        raise ValueError('dataset_info.json must contain a dataset mapping')
    for entry in entries.values():
        filename = entry.get('file_name') if isinstance(entry, dict) else None
        if not isinstance(filename, str) or not filename:
            raise ValueError('Resolve remote dataset references to local files before creating a version')
        relative = PurePosixPath(filename)
        if relative.is_absolute() or '..' in relative.parts or '\\' in filename or ':' in filename:
            raise ValueError('Dataset references must remain inside the version directory')
        if not (root / filename).is_file():
            raise ValueError(f'Dataset references a missing local file: {filename}')


def _stat_signature(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _content_manifest(files: list[dict[str, Any]]) -> dict[str, Any]:
    # Hash declared bytes and relative names, never machine-specific paths.
    return {"schemaVersion": 1, "fingerprint": hashlib.sha256(encode(files)).hexdigest(), "files": files,
            "byteCount": sum(item["size"] for item in files), "fileCount": len(files)}


def manifest(source: str | Path, *, signatures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    source = Path(source).absolute()
    if source.is_symlink() or not source.exists():
        raise ValueError("Asset path must exist and cannot be a symbolic link")
    files: list[dict[str, Any]] = []
    observed_signatures = []
    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not accepted in immutable assets: {path.name}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Asset contains a non-regular file")
        relative = path.name if source.is_file() else path.relative_to(source).as_posix()
        before = path.stat()
        checksum = file_digest(path)
        after = path.stat()
        if _stat_signature(before) != _stat_signature(after):
            raise ValueError("Asset changed while calculating its fingerprint")
        files.append({"path": relative, "size": after.st_size, "sha256": checksum})
        observed_signatures.append({"path": relative, "stat": _stat_signature(after)})
    if not files:
        raise ValueError("Cannot register an empty asset")
    if signatures is not None:
        signatures.extend(observed_signatures)
    return _content_manifest(files)


def file_signatures(source: Path) -> list[dict[str, Any]]:
    """List every file without reading its bytes; reject changed tree structure."""
    if source.is_symlink() or not source.exists():
        raise ValueError("Asset path must exist and cannot be a symbolic link")
    signatures = []
    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not accepted in immutable assets: {path.name}")
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Asset contains a non-regular file")
        signatures.append({
            "path": path.name if source.is_file() else path.relative_to(source).as_posix(),
            "stat": _stat_signature(info),
        })
    if not signatures:
        raise ValueError("Cannot register an empty asset")
    return signatures


def _try_reflink(reader, writer) -> bool:
    """Clone independent CoW extents when Linux and both filesystems support it."""
    if not sys.platform.startswith("linux"):
        return False
    import fcntl
    try:
        fcntl.ioctl(writer.fileno(), 0x40049409, reader.fileno())  # Linux FICLONE
        return True
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.ENOSYS}:
            raise
        # An unsupported clone must leave an empty target for streaming fallback.
        writer.seek(0)
        writer.truncate(0)
        return False


def _copy_manifest(source: Path, temporary: Path, source_is_file: bool,
                   signatures: list[dict[str, Any]], *,
                   copied_signatures: list[dict[str, Any]] | None = None,
                   copy_stats: dict[str, int] | None = None) -> dict[str, Any]:
    """Clone or stream each file, verifying it while its bytes are still hot."""
    files = []
    observed_signatures = []
    stats = {"reflinkFiles": 0, "reflinkBytes": 0, "streamedFiles": 0, "streamedBytes": 0}
    for item in signatures:
        original = source if source_is_file else source / item["path"]
        if original.is_symlink() or (not source_is_file and source.resolve() not in original.resolve().parents):
            raise ValueError("Source changed while creating snapshot")
        target = temporary / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with original.open("rb") as reader:
            info = os.fstat(reader.fileno())
            if not stat.S_ISREG(info.st_mode) or _stat_signature(info) != item["stat"]:
                raise ValueError("Source changed while creating snapshot")
            with target.open("xb") as writer:
                cloned = _try_reflink(reader, writer)
                if cloned:
                    size = os.fstat(writer.fileno()).st_size
                else:
                    for chunk in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size != item["stat"][2] or _stat_signature(os.fstat(reader.fileno())) != item["stat"]:
                raise ValueError("Source changed while creating snapshot")
            before = target.stat()
            checksum = file_digest(target)
            after = target.stat()
            if (before.st_size != size or _stat_signature(before) != _stat_signature(after)
                    or (not cloned and checksum != digest.hexdigest())):
                raise ValueError("Snapshot copy failed content verification")
            if _stat_signature(os.fstat(reader.fileno())) != item["stat"]:
                raise ValueError("Source changed while creating snapshot")
        if original.is_symlink() or _stat_signature(original.stat()) != item["stat"]:
            raise ValueError("Source changed while creating snapshot")
        files.append({"path": item["path"], "size": size, "sha256": checksum})
        observed_signatures.append({"path": item["path"], "stat": _stat_signature(after)})
        method = "reflink" if cloned else "streamed"
        stats[method + "Files"] += 1
        stats[method + "Bytes"] += size
    if copied_signatures is not None:
        copied_signatures.extend(observed_signatures)
    if copy_stats is not None:
        copy_stats.update(stats)
    return _content_manifest(files)


def _cache_file(root: Path, source: Path) -> Path:
    key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return root / f".snapshot-cache-{key}.json"


def _read_cache(root: Path, source: Path) -> dict[str, Any] | None:
    try:
        cache = json.loads(_cache_file(root, source).read_text(encoding="utf-8"))
        if (cache.get("cacheSchema") != 1 or cache.get("sourcePath") != str(source)
                or cache.get("sourceIsFile") != source.is_file()):
            return None
        content = cache["manifest"]
        files = content["files"]
        fingerprint = content["fingerprint"]
        if (content.get("schemaVersion") != 1 or not isinstance(files, list) or not files
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)
                or hashlib.sha256(encode(files)).hexdigest() != fingerprint
                or content["fileCount"] != len(files)
                or content["byteCount"] != sum(item["size"] for item in files)):
            return None
        expected = [(item["path"], item["size"]) for item in files]
        for key in ("sourceSignatures", "destinationSignatures"):
            signatures = cache[key]
            if not isinstance(signatures, list) or any(
                    not isinstance(item.get("stat"), list) or len(item["stat"]) != 5
                    or not all(isinstance(value, int) for value in item["stat"])
                    for item in signatures):
                return None
            if [(item["path"], item["stat"][2]) for item in signatures] != expected:
                return None
        return cache
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None


def _write_cache(root: Path, source: Path, content: dict[str, Any],
                 source_signatures: list[dict[str, Any]], destination_signatures: list[dict[str, Any]]) -> bool:
    payload = {"cacheSchema": 1, "sourcePath": str(source), "sourceIsFile": source.is_file(),
               "manifest": content,
               "sourceSignatures": source_signatures, "destinationSignatures": destination_signatures}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".snapshot-cache-writing-",
                                         suffix=".json", dir=root, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(_cache_file(root, source))
        return True
    except OSError:
        # The cache is optional; the next request can always use full SHA-256.
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def snapshot(source: str | Path, version_root: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    timings: dict[str, float] = {}

    def measured(name: str, phase_started: float) -> None:
        timings[name] = round((time.monotonic() - phase_started) * 1000, 1)

    def with_timings(result: dict[str, Any], reused: bool) -> dict[str, Any]:
        return {**result, "timingMs": {**timings, "total": round((time.monotonic() - started) * 1000, 1)},
                "reused": reused}

    source = Path(source).absolute()
    if source.is_symlink():
        raise ValueError("Snapshot source cannot be a symbolic link")
    root = Path(version_root).resolve()
    if root == source.resolve() or source.resolve() in root.parents:
        raise ValueError("Version storage cannot be inside its source")
    root.mkdir(parents=True, exist_ok=True)
    cache = _read_cache(root, source)
    phase_started = time.monotonic()
    source_is_file = source.is_file()
    source_signatures = file_signatures(source)
    source_cache_hit = bool(cache and source_signatures == cache["sourceSignatures"])
    before = cache["manifest"] if source_cache_hit else None
    measured("sourceSignatures", phase_started)
    destination = root / before["fingerprint"] if before else None
    if destination is not None and destination.exists():
        phase_started = time.monotonic()
        destination_signatures = file_signatures(destination) if cache and source_cache_hit else None
        destination_cache_hit = bool(source_cache_hit and destination_signatures == cache["destinationSignatures"])
        destination_hashed_signatures: list[dict[str, Any]] = []
        existing = (cache["manifest"] if destination_cache_hit else
                    manifest(destination, signatures=destination_hashed_signatures))
        measured("existingManifest", phase_started)
        if existing["fingerprint"] != before["fingerprint"]:
            raise ValueError("Managed version contents were modified")
        cache_status = "hit" if destination_cache_hit else "unstable"
        if not destination_cache_hit or not source_cache_hit:
            current_source = file_signatures(source)
            current_destination = file_signatures(destination)
            if (current_source == source_signatures
                    and current_destination == (destination_signatures if destination_cache_hit
                                                else destination_hashed_signatures)):
                cache_status = ("stored" if _write_cache(root, source, before, current_source, current_destination)
                                else "write_failed")
        return with_timings({**existing, "path": str(destination), "sourcePath": str(source),
                             "sourceIsFile": source_is_file, "integrity": "verified",
                             "cacheHit": destination_cache_hit, "cacheStatus": cache_status}, True)
    temporary = Path(tempfile.mkdtemp(prefix=".creating-", dir=root))
    try:
        # Without a trusted cache, discover the fingerprint during the copy.
        # This also handles a cached version whose destination has disappeared.
        phase_started = time.monotonic()
        copied_signatures: list[dict[str, Any]] = []
        copy_stats: dict[str, int] = {}
        streamed = _copy_manifest(source, temporary, source_is_file, source_signatures,
                                  copied_signatures=copied_signatures, copy_stats=copy_stats)
        measured("copyAndHash", phase_started)
        if before is not None and streamed["fingerprint"] != before["fingerprint"]:
            raise ValueError("Source changed while creating snapshot")
        before = streamed
        destination = root / before["fingerprint"]
        phase_started = time.monotonic()
        # Each file was already hashed/verified just after copying. Metadata must
        # still match that exact verified tree; do not reread all model weights.
        current_copied_signatures = file_signatures(temporary)
        measured("copiedManifest", phase_started)
        if current_copied_signatures != copied_signatures:
            raise ValueError("Snapshot copy failed content verification")
        phase_started = time.monotonic()
        source_after_signatures = file_signatures(source)
        measured("sourceRecheck", phase_started)
        if source_after_signatures != source_signatures or source.is_file() != source_is_file:
            raise ValueError("Source changed while creating snapshot")
        # Keep metadata in the record store, never alongside source data: legacy
        # preprocessors enumerate every JSON file, including hidden JSON files.
        phase_started = time.monotonic()
        reused = False
        try:
            if destination.exists():
                raise FileExistsError(str(destination))
            # Windows may briefly deny directory renames while another process
            # holds a file handle. Retry the same atomic operation, never overwrite
            # a target or relax filesystem permissions.
            for attempt in range(5):
                try:
                    temporary.rename(destination)
                    break
                except PermissionError as exc:
                    if os.name != 'nt' or getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except OSError:
            if not destination.exists():
                raise
            existing_signatures: list[dict[str, Any]] = []
            if manifest(destination, signatures=existing_signatures)["fingerprint"] != before["fingerprint"]:
                raise ValueError("Managed version contents were modified")
            copied_signatures = existing_signatures
            reused = True
        for item in before["files"]:
            target = destination / item["path"]
            target.chmod(target.stat().st_mode & ~0o222)
        measured("finalize", phase_started)
        current_source = file_signatures(source)
        current_destination = file_signatures(destination)
        # Rename/chmod can change ctime; the other identity fields must still
        # match the bytes hashed in the temporary directory.
        cache_status = "unstable"
        if (current_source == source_after_signatures and
                [(item["path"], item["stat"][:4]) for item in current_destination] ==
                [(item["path"], item["stat"][:4]) for item in copied_signatures]):
            cache_status = ("stored" if _write_cache(root, source, before, current_source, current_destination)
                            else "write_failed")
        return with_timings({**before, "path": str(destination), "sourcePath": str(source),
                             "sourceIsFile": source_is_file, "integrity": "verified",
                             "cacheHit": False, "cacheStatus": cache_status, "copyStats": copy_stats}, reused)
    finally:
        # This is our own newly allocated staging directory, never the source/version directory.
        if temporary.exists() and temporary.parent.resolve() == root and temporary.name.startswith(".creating-"):
            shutil.rmtree(temporary)


if __name__ == "__main__":
    import sys
    request = json.load(sys.stdin)
    if request.get('assetKind') == 'dataset':
        validate_dataset_references(request['path'])
    result = snapshot(request["path"], request["versionRoot"]) if request.get("operation") == "snapshot" else manifest(request["path"])
    print(json.dumps(result, ensure_ascii=False))
