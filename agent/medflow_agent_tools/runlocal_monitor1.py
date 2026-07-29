# -*- coding: utf-8 -*-
"""Deprecated compatibility wrapper for :mod:`medflow_agent_tools.runlocal_monitor`.

Use ``medflow_agent_tools.runlocal_monitor`` for new imports. This module is kept
for one release cycle so existing deployments and user code continue to work.
"""

from .runlocal_monitor import *  # noqa: F401,F403
