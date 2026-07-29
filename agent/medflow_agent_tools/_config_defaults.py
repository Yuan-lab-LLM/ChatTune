"""Shared defaults loaded from the agent configuration when available."""

import os
from typing import Any, Optional


def _load_agent_config() -> Optional[Any]:
    try:
        from utils.config import get_current_config, init_config

        try:
            return get_current_config()
        except Exception:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ".."),
            )
            return init_config(os.path.join(project_root, "config", "config.yaml"))
    except Exception:
        return None


def get_default_docker_container() -> str:
    """Return the configured default execution container."""
    config = _load_agent_config()
    return (
        os.getenv("AGENT3_DEFAULT_DOCKER_CONTAINER")
        or getattr(getattr(config, "environment", None), "default_docker_container", None)
        or "agent3"
    )
