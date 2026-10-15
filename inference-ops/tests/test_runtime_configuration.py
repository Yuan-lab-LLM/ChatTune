"""Backend provisioning, restart recovery and secret-boundary regressions."""
import ast
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[2]
OS_ENV = {key: value for key, value in os.environ.items() if key in {"SystemRoot", "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}}
sys.path.insert(0, str(REPO / "inference-ops"))
sys.path.insert(0, str(REPO / "agent"))
from inference_ops.experiment_runtime import runtime_credentials as credentials
from inference_ops.experiment_runtime import experiment_tracker
from inference_ops.runtime_configuration import authorized_backend, forward_configuration
import inference_runtime_config as backend


class CredentialStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        environment = patch.dict(os.environ, {**OS_ENV, "INFERENCE_OPS_STATE_DIR": str(self.state)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.values = {"studio_url": "http://studio:3000", "runtime_node_id": "node-1", "node_token": "private-node-secret"}

    def test_initialization_is_idempotent_and_restores_credentials_after_reload(self):
        result = credentials.initialize(self.values)
        path = credentials.credential_path()
        before = path.stat().st_mtime_ns
        self.assertEqual(credentials.initialize(self.values), result)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(credentials.runtime_status(), {"configured": True, "runtime_node_id": "node-1", "source": "backend"})
        self.assertEqual(credentials.current_node_id(), "node-1")
        self.assertNotIn("private-node-secret", json.dumps(result))
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        process_env = {**OS_ENV, "INFERENCE_OPS_STATE_DIR": str(self.state)}
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);from inference_ops.experiment_runtime.runtime_credentials import studio_connection;print(json.dumps({'restored':studio_connection()==('http://studio:3000','private-node-secret')}))"
        restored = subprocess.run([sys.executable, "-B", "-c", code, str(REPO / "inference-ops")],
                                  env=process_env, capture_output=True, text=True, timeout=10)
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual(json.loads(restored.stdout), {"restored": True})

    def test_other_node_cannot_overwrite_persisted_credentials(self):
        credentials.initialize(self.values)
        before = credentials.credential_path().read_bytes()
        with self.assertRaisesRegex(ValueError, "其他 runtime_node_id"):
            credentials.initialize({**self.values, "runtime_node_id": "node-2"})
        self.assertEqual(credentials.credential_path().read_bytes(), before)

    def test_token_rotation_is_atomic_and_background_sender_uses_new_token(self):
        credentials.initialize(self.values)
        credentials.initialize({**self.values, "node_token": "rotated-secret"})
        captured = []
        def respond(request, timeout):
            captured.append(request.get_header("X-medflow-runtime-token"))
            return io.BytesIO(b'{"result":{"data":{"json":{"success":true}}}}')
        with patch.object(experiment_tracker, "urlopen", side_effect=respond):
            experiment_tracker.studio_request("renewTrainingResourcesForRuntime", {"runtimeNodeId": "node-1"})
        self.assertEqual(captured, ["rotated-secret"])
        self.assertEqual(list(credentials.credential_path().parent.glob("*.tmp")), [])

    def test_explicit_manual_credentials_remain_authoritative(self):
        with patch.dict(os.environ, {"STUDIO_URL": self.values["studio_url"], "MEDFLOW_RESOURCE_NODE_ID": "node-1", "MEDFLOW_STUDIO_NODE_TOKEN": "private-node-secret"}):
            self.assertEqual(credentials.runtime_status()["source"], "environment")
            credentials.initialize(self.values)
            self.assertFalse(credentials.credential_path().exists())
            with self.assertRaises(ValueError) as error:
                credentials.initialize({**self.values, "node_token": "different-secret"})
            self.assertNotIn("different-secret", str(error.exception))

    def test_incomplete_or_invalid_input_is_not_persisted(self):
        for changes in ({"node_token": "replace-token"}, {"runtime_node_id": "../other"}, {"studio_url": "file:///secret"},
                        {"studio_url": "http://private-user:private-password@studio"}, {"node_token": {}}, {"extra": "not-allowed"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as error:
                credentials.initialize({**self.values, **changes})
            self.assertNotIn("private-password", str(error.exception))
        self.assertFalse(credentials.credential_path().exists())


class RuntimeHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "worker-state"
        environment = patch.dict(os.environ, {**OS_ENV, "INFERENCE_OPS_STATE_DIR": str(self.state)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        backend._cache.clear()
        self.posts = []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def reply(self, status, value):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                self.reply(200, credentials.runtime_status(owner.state))
            def do_POST(self):
                if not authorized_backend(self.headers.get("Authorization"), "service-secret"):
                    self.reply(403, {"detail": "backend only"})
                    return
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.posts.append(value)
                try:
                    self.reply(200, credentials.initialize(value, owner.state))
                except ValueError:
                    self.reply(409, {"detail": "node binding conflict"})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.env = {"STUDIO_URL": "http://studio:3000", "MEDFLOW_RESOURCE_NODE_ID": "node-1", "MEDFLOW_RUNTIME_NODE_TOKEN": "node-secret",
                    "INFERENCE_OPS_API_TOKEN": "service-secret", "START_LOCAL_INFERENCE": "false"}

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_real_backend_http_initializes_manual_worker_and_caches_requests(self):
        backend.ensure_inference_runtime(self.url + "/inference_agent", self.env)
        backend.ensure_inference_runtime(self.url + "/inference_agent", self.env)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(credentials.studio_connection(), ("http://studio:3000", "node-secret"))
        with urlopen(self.url + "/runtime/config/status") as response:
            status = response.read().decode()
        self.assertNotIn("node-secret", status)
        self.assertNotIn("service-secret", status)
        backend.ensure_inference_runtime(self.url + "/inference_agent", {**self.env, "MEDFLOW_RUNTIME_NODE_TOKEN": "rotated-token"})
        self.assertEqual(credentials.studio_connection()[1], "rotated-token")
        self.assertEqual(len(self.posts), 2)

    def test_browser_without_service_token_cannot_initialize(self):
        request = Request(self.url + "/runtime/config", data=b'{}', method="POST", headers={"Origin": "http://studio-ui"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)
        self.assertFalse(credentials.credential_path(self.state).exists())

    def test_backend_refuses_to_send_secrets_to_another_bound_node(self):
        credentials.initialize({"studio_url": "http://other-studio", "runtime_node_id": "node-2", "node_token": "other-node-token"}, self.state)
        with self.assertRaisesRegex(RuntimeError, "其他 Runtime 节点"):
            backend.ensure_inference_runtime(self.url + "/inference_agent", self.env)
        self.assertEqual(self.posts, [])

    def test_local_docker_provisioning_works_while_process_management_is_disabled(self):
        root = Path(self.temp.name) / "other-mount/inference-ops"
        (root / "inference_ops/experiment_runtime").mkdir(parents=True)
        for directory in (root / "inference_ops", root / "inference_ops/experiment_runtime"):
            (directory / "__init__.py").write_text("", encoding="utf-8")
        (root / "inference_ops/environment.py").write_text("def load_runtime_environment(): pass\n", encoding="utf-8")
        (root / "inference_ops/experiment_runtime/runtime_credentials.py").write_text(Path(credentials.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        calls = []
        def docker(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["docker", "inspect"]:
                return types.SimpleNamespace(returncode=0, stdout=json.dumps([{"State": {"Running": True},
                    "Config": {"WorkingDir": str(root)}, "Mounts": []}]))
            self.assertEqual(args[:6], ["docker", "exec", "-i", "infer-container", "python3", "-c"])
            self.assertNotIn("node-secret", " ".join(args))
            self.assertNotIn("Popen", args[-1])
            return subprocess.run([sys.executable, "-B", "-c", args[-1]], input=kwargs["input"], capture_output=True, text=True,
                                  timeout=10, env={**os.environ, "INFERENCE_OPS_STATE_DIR": str(self.state)})
        env = {**self.env, "MEDFLOW_LOCAL_EVALUATE_CONTAINER": "infer-container", "MEDFLOW_INFERENCE_OPS_DIR": str(root)}
        env.pop("INFERENCE_OPS_API_TOKEN")
        backend.ensure_inference_runtime(self.url + "/inference_agent", env, runner=docker)
        self.assertEqual(len(calls), 2)
        self.assertEqual(credentials.studio_connection()[1], "node-secret")
        self.assertEqual(self.posts, [])


class ConfigurationEndpointTests(unittest.TestCase):
    def test_initialization_requires_service_auth_even_in_trusted_network(self):
        self.assertFalse(authorized_backend("", ""))
        self.assertFalse(authorized_backend("Bearer anything", ""))
        self.assertFalse(authorized_backend("Bearer wrong", "service-secret"))
        self.assertTrue(authorized_backend("Bearer service-secret", "service-secret"))

    def test_api_authenticates_before_reading_body_and_hides_validation_input(self):
        tree = ast.parse((REPO / "inference-ops/inference_ops/inference_agent.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "initialize_runtime_configuration")
        function.decorator_list = []
        class HttpError(Exception):
            def __init__(self, status_code, detail): self.status_code, self.detail = status_code, detail
        namespace = {"Request": object, "HTTPException": HttpError, "authorized_backend": authorized_backend,
            "INFERENCE_OPS_API_TOKEN": "service-secret", "AGENT_ROLE": "both", "runtime_credentials": credentials,
            "ops_tools": types.SimpleNamespace(STATE_DIR="unused"), "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "endpoint", "exec"), namespace)
        class Incoming:
            def __init__(self, header): self.headers = header; self.read = False
            async def body(self): self.read = True; return b'{"node_token":"private-input"}'
        request = Incoming({})
        with self.assertRaises(HttpError) as error:
            asyncio.run(namespace[function.name](request))
        self.assertEqual(error.exception.status_code, 403)
        self.assertFalse(request.read)
        request = Incoming({"authorization": "Bearer service-secret"})
        with self.assertRaises(HttpError) as error:
            asyncio.run(namespace[function.name](request))
        self.assertEqual(error.exception.status_code, 409)
        self.assertNotIn("private-input", error.exception.detail)

    def test_controller_forwarding_preserves_auth_and_filters_response(self):
        def respond(request, timeout):
            self.assertEqual(request.full_url, "http://worker:8899/runtime/config")
            self.assertEqual(request.get_header("Authorization"), "Bearer service-secret")
            return io.BytesIO(b'{"configured":true,"runtime_node_id":"node-1","node_token":"private-secret"}')
        result = forward_configuration("http://worker:8899/worker/tool", "service-secret", {"node_token": "private-secret"}, opener=respond)
        self.assertNotIn("private-secret", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
