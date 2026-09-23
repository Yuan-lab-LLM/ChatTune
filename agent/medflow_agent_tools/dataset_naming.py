"""Human-readable names for Runtime-generated dataset artifacts."""

from __future__ import annotations

from datetime import datetime
import re


_GENERATED_DATASET_SUFFIX = re.compile(
    r"_(?:raw|sft|dpo|pt|filtered)_20\d{6}-\d{6}-\d{3}$",
    re.IGNORECASE,
)


def _safe_dataset_name(value: object, fallback: str = "dataset") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized[:64] or fallback


def _dataset_lineage_name(value: object) -> str:
    """Keep the original source label when processing a generated dataset again."""

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
    """Build ``source_purpose_YYYYMMDD-HHMMSS-mmm`` artifact names."""

    moment = created_at or datetime.now()
    safe_source = _dataset_lineage_name(source_name)
    normalized_type = _safe_dataset_name(output_type, "processed").lower()
    purpose = "filtered" if filtered else normalized_type
    readable_time = moment.strftime("%Y%m%d-%H%M%S")
    milliseconds = moment.microsecond // 1000
    return f"{safe_source}_{purpose}_{readable_time}-{milliseconds:03d}"
