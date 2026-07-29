# -*- coding: utf-8 -*-
"""Deprecated compatibility wrapper for :mod:`runtime_agent`.

Use ``runtime_agent.py`` as the Agent API orchestration entrypoint. This module
is kept for one release cycle so existing scripts and deployments that import
``agent3_new`` continue to work.
"""

from runtime_agent import *  # noqa: F401,F403
