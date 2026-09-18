import importlib.util
import os
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "scripts" / "vllm_compat" / "sitecustomize.py"


class VllmRouteCompatTests(unittest.TestCase):
    def test_old_instrumentator_skips_fastapi_router_without_path(self):
        spec = importlib.util.spec_from_file_location("vllm_route_compat_test", SHIM)
        shim = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"INFERENCE_OPS_VLLM_ROUTE_COMPAT": "0"}):
            spec.loader.exec_module(shim)

        class Match(Enum):
            NONE = 0
            PARTIAL = 1
            FULL = 2

        class Mount:
            pass

        class PathlessRouter:
            def matches(self, _scope):
                return Match.FULL, {}

        class Route:
            path = "/v1/models"

            def matches(self, _scope):
                return Match.FULL, {}

        routing = types.ModuleType("prometheus_fastapi_instrumentator.routing")
        original = lambda *_args: "old"
        routing._get_route_name = original
        instrumentator = types.ModuleType("prometheus_fastapi_instrumentator")
        instrumentator.routing = routing
        starlette = types.ModuleType("starlette")
        starlette_routing = types.ModuleType("starlette.routing")
        starlette_routing.Match = Match
        starlette_routing.Mount = Mount

        fake_modules = {
            "prometheus_fastapi_instrumentator": instrumentator,
            "prometheus_fastapi_instrumentator.routing": routing,
            "starlette": starlette,
            "starlette.routing": starlette_routing,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            with mock.patch.object(
                shim, "_version",
                side_effect=lambda name: (0, 141, 1) if name == "fastapi" else (7, 0, 0),
            ):
                shim._patch_old_instrumentator()
            self.assertEqual(
                "/v1/models",
                routing._get_route_name({"path": "/v1/models"}, [PathlessRouter(), Route()]),
            )

            routing._get_route_name = original
            with mock.patch.object(
                shim, "_version",
                side_effect=lambda name: (0, 141, 1) if name == "fastapi" else (8, 1, 0),
            ):
                shim._patch_old_instrumentator()
            self.assertIs(original, routing._get_route_name)


if __name__ == "__main__":
    unittest.main()
