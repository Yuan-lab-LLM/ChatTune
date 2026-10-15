"""sitecustomize for medical vLLM processes using Transformers 4.x.

The packaged training runtime may save Transformers 5's list-format
extra_special_tokens. Normalize constructor kwargs in memory for old serving
runtimes; never edit the model or its content-addressed snapshot.
"""
from __future__ import annotations

from functools import wraps
from importlib.metadata import version
import os
import re
import sys


def normalize_special_tokens(kwargs):
    extra = kwargs.get("extra_special_tokens")
    if not isinstance(extra, (list, tuple)):
        return kwargs
    additional = kwargs.get("additional_special_tokens") or []
    if not isinstance(additional, (list, tuple)):
        raise TypeError("additional_special_tokens must be a list or tuple")
    merged, seen = [], set()
    for token in [*additional, *extra]:
        # AddedToken instances expose content and must retain all their flags.
        content = token if isinstance(token, str) else getattr(token, "content", None)
        if not isinstance(content, str):
            raise TypeError("Special tokens must be strings or AddedToken objects")
        if content not in seen:
            merged.append(token)
            seen.add(content)
    normalized = dict(kwargs)
    normalized["extra_special_tokens"] = {}
    normalized["additional_special_tokens"] = merged
    return normalized


def patch_tokenizer_base(base, installed_version):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed_version)
    if not match or not (4, 51, 0) <= tuple(map(int, match.groups())) < (5, 0, 0):
        return False
    original = base.__init__
    if getattr(original, "_medflow_tokenizer_compat", False):
        return False

    @wraps(original)
    def compatible_init(self, *args, **kwargs):
        normalized = normalize_special_tokens(kwargs)
        if normalized is not kwargs:
            print(
                "[medflow] Normalized list-format extra_special_tokens for "
                f"Transformers {installed_version}; preserved special tokens",
                file=sys.stderr, flush=True,
            )
        original(self, *args, **normalized)

    compatible_init._medflow_tokenizer_compat = True
    base.__init__ = compatible_init
    return True


def install_patch():
    if os.environ.get("MEDFLOW_VLLM_TOKENIZER_COMPAT") != "1":
        return False
    installed_version = version("transformers")
    # Avoid importing Transformers 5.x or unrelated Python processes.
    if not installed_version.startswith("4."):
        return False
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    return patch_tokenizer_base(PreTrainedTokenizerBase, installed_version)


if __name__ == "sitecustomize":
    try:
        install_patch()
    except Exception as exc:
        print(f"[medflow] vLLM tokenizer compatibility setup failed: {exc}", file=sys.stderr, flush=True)
