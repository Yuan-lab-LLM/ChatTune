"""Scoped vLLM workaround for FastAPI lazy routers and old metrics middleware.

Only the vLLM process tree receives this directory on PYTHONPATH. Python loads
sitecustomize before the vLLM CLI and before its spawned API server imports the
metrics middleware.
"""

import os


def _version(distribution):
    from importlib.metadata import version
    import re

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version(distribution))
    return tuple(map(int, match.groups())) if match else (0, 0, 0)


def _patch_old_instrumentator():
    if _version("fastapi") < (0, 137, 0):
        return
    if _version("prometheus-fastapi-instrumentator") >= (8, 0, 1):
        return

    from prometheus_fastapi_instrumentator import routing
    from starlette.routing import Match, Mount

    def _get_route_name(scope, routes, route_name=None):
        for route in routes:
            path = getattr(route, "path", None)
            if path is None:
                continue
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                route_name = path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child = _get_route_name(child_scope, route.routes, route_name)
                    route_name = None if child is None else route_name + child
                return route_name
            if match == Match.PARTIAL and route_name is None:
                route_name = path
        return None

    routing._get_route_name = _get_route_name
    print("[inference-ops] Applied vLLM route compatibility patch", flush=True)


if os.environ.get("INFERENCE_OPS_VLLM_ROUTE_COMPAT") == "1":
    try:
        _patch_old_instrumentator()
    except Exception as exc:
        import sys

        print(f"[inference-ops] vLLM route compatibility patch failed: {exc}", file=sys.stderr)
