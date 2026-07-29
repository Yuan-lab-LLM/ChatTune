# -*- coding: utf-8 -*-
"""Shared helpers for dataset_name candidate discovery."""

import shlex
from typing import Any, List


INTERNAL_DATASET_FILENAMES = frozenset(
    {
        "dataset_info.json",
        "preprocessing_audit.json",
        "preprocessing_summary.json",
        "score_audit.json",
        "score_filter_process.log",
        "score_progress.json",
        "score_summary.json",
    }
)


def dataset_names_from_dataset_info(dataset_info: Any) -> List[str]:
    if not isinstance(dataset_info, dict):
        return []
    return [str(name).strip() for name in dataset_info.keys() if str(name).strip()]


def dataset_find_exclusion_clause() -> str:
    return " ".join(
        f"! -name {shlex.quote(filename)}"
        for filename in sorted(INTERNAL_DATASET_FILENAMES)
    )


def is_dataset_candidate_filename(filename: str) -> bool:
    clean_name = (filename or "").strip()
    if not clean_name.endswith(".json"):
        return False
    if clean_name.lower() in INTERNAL_DATASET_FILENAMES:
        return False
    stem = clean_name[:-5]
    return stem.lower() != "test"
