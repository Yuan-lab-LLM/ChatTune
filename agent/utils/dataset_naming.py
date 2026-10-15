"""Compact names shared by dataset tools and experiment execution."""

from __future__ import annotations

from datetime import datetime
import hashlib
import re


_GENERATED_DATASET_SUFFIX = re.compile(
    r"_(?P<purpose>raw|sft|dpo|pt|filtered|processed)_"
    r"(?:20\d{6}|\d{6})-\d{6}-"
    r"(?:\d{3}(?:-\d{20}-[A-Za-z0-9_-]+)?|[a-f0-9]{10})$",
    re.IGNORECASE,
)


def _safe_dataset_name(value: object, fallback: str = "dataset") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return re.sub(r"_+", "_", normalized).strip("._-") or fallback


def _dataset_lineage_name(value: object) -> str:
    # Strip complete legacy execution suffixes before truncating the source.
    normalized = _safe_dataset_name(value)
    while True:
        compacted = _GENERATED_DATASET_SUFFIX.sub("", normalized).strip("._-")
        if not compacted or compacted == normalized:
            return normalized
        normalized = compacted


def build_dataset_output_name(
    source_name: object,
    output_type: object,
    *,
    filtered: bool = False,
    created_at: datetime | None = None,
) -> str:
    """Build ``source_purpose_YYMMDD-HHMMSS-mmm`` names with a short source."""
    moment = created_at or datetime.now()
    source = _dataset_lineage_name(source_name)[:24].rstrip("._-")
    purpose = "filtered" if filtered else _safe_dataset_name(output_type, "processed").lower()[:12]
    return f"{source}_{purpose}_{moment:%y%m%d-%H%M%S}-{moment.microsecond // 1000:03d}"


def build_dataset_execution_name(
    requested_name: str,
    attempt_id: str,
    *,
    created_at: datetime | None = None,
) -> str:
    """Replace a provisional timestamp with one timestamp and a short attempt hash."""
    moment = created_at or datetime.now()
    name = _safe_dataset_name(requested_name)
    match = _GENERATED_DATASET_SUFFIX.search(name)
    if match:
        source = _dataset_lineage_name(name)[:24].rstrip("._-")
        label = f"{source}_{match['purpose'].lower()}"
    else:
        label = name[:32].rstrip("._-")
    short_id = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:10]
    # Shared date directories retain their date prefix for legacy latest ordering.
    if re.fullmatch(r"20\d{6}", label):
        return f"{label}-{moment:%H%M%S}-{short_id}"
    return f"{label}_{moment:%y%m%d-%H%M%S}-{short_id}"
