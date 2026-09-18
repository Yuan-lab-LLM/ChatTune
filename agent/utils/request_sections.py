"""Utilities for normalizing tagged user-input requests from function agents."""

from __future__ import annotations

import re
from typing import Optional


REQUEST_TAG_BY_KIND = {
    "type": "[类型请求]",
    "scope": "[范围请求]",
    "choice": "[等待用户选择]",
    "param": "[参数请求]",
}

REQUEST_KIND_PRIORITY = ("type", "scope", "choice", "param")


def request_kind_from_text(text: str) -> Optional[str]:
    """Return the highest-priority request kind present in an agent response."""
    content = str(text or "")
    for kind in REQUEST_KIND_PRIORITY:
        if REQUEST_TAG_BY_KIND[kind] in content:
            return kind
    return None


def extract_request_section(text: str, request_kind: str) -> str:
    """Extract one tagged request section and discard any later request sections."""
    content = re.sub(
        r"<think>.*?</think>",
        "",
        str(text or ""),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    target_tag = REQUEST_TAG_BY_KIND.get(str(request_kind or ""))
    if target_tag and target_tag in content:
        section_start = content.find(target_tag) + len(target_tag)
        section_end = len(content)
        for tag in REQUEST_TAG_BY_KIND.values():
            next_start = content.find(tag, section_start)
            if next_start >= 0:
                section_end = min(section_end, next_start)
        content = content[section_start:section_end]

    # Preserve the previous friendly-text behavior for any non-request wrappers
    # that may be included inside the selected section.
    return re.sub(r"\[[^\]]+\]\s*", "", content).strip()


def request_kind_from_protocol_kind(protocol_kind: str) -> str:
    """Restore the pending-state request kind from a normalized protocol kind."""
    kind = str(protocol_kind or "").strip()
    if kind in {"training_type", "assessment_type", "evaluation_type"}:
        return "type"
    if kind == "evaluation_scope":
        return "scope"
    if kind in {"choice", "advanced_filter_choice"}:
        return "choice"
    return "param"
