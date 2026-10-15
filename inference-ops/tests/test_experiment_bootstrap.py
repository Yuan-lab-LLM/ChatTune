"""Exercise worker bootstrap and real collector threads without service dependencies."""
import ast
import importlib
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
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inference-ops"))
from inference_ops.environment import load_runtime_environment
from inference_ops import experiment_records
from inference_ops.experiment_runtime import experiment_tracker as worker_tracker

AGENT_PACKAGE = "_agent_experiment_bootstrap_tests"
package = types.ModuleType(AGENT_PACKAGE)
package.__path__ = [str(REPO / "agent/utils")]
sys.modules[AGENT_PACKAGE] = package
agent_tracker = importlib.import_module(f"{AGENT_PACKAGE}.experiment_tracker")


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class EnvironmentFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "worker.env"

    def test_file_loads_quotes_crlf_and_preserves_process_values(self):
        self.path.write_bytes(("\ufeff# worker\r\nexport STUDIO_URL='http://studio:3000'\r\n"
            'MEDFLOW_RUNTIME_NODE_TOKEN="token # with spaces" # comment\r\n'
            "MEDFLOW_RESOURCE_NODE_ID=node-1\r\nAGENT_ROLE=worker # role\r\nEMPTY=''\r\n").encode("utf-8"))
        env = {"INFERENCE_OPS_ENV_FILE": str(self.path), "STUDIO_URL": "http://override:3000"}
        load_runtime_environment(env)
        self.assertEqual(env["STUDIO_URL"], "http://override:3000")
        self.assertEqual(env["MEDFLOW_RUNTIME_NODE_TOKEN"], "token # with spaces")
        self.assertEqual(env["MEDFLOW_RESOURCE_NODE_ID"], "node-1")
        self.assertEqual(env["AGENT_ROLE"], "worker")
        self.assertEqual(env["EMPTY"], "")

    def test_values_are_literal_and_do_not_execute_shell_commands(self):
        self.path.write_text("TOKEN='$(echo private)'\nPATH_VALUE=C:\\models\\model\n", encoding="utf-8")
        env = {"INFERENCE_OPS_ENV_FILE": str(self.path)}
        load_runtime_environment(env)
        self.assertEqual(env["TOKEN"], "$(echo private)")
        self.assertEqual(env["PATH_VALUE"], r"C:\models\model")

    def test_bad_file_reports_location_without_secrets_or_partial_load(self):
        for content in ("STUDIO_URL=http://studio\nTOKEN='private-secret\n", "private-secret\n"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                env = {"INFERENCE_OPS_ENV_FILE": str(self.path)}
                with self.assertRaises(ValueError) as caught:
                    load_runtime_environment(env)
                self.assertNotIn("private-secret", str(caught.exception))
                self.assertIn(str(self.path), str(caught.exception))
                self.assertNotIn("STUDIO_URL", env)

    def test_explicit_missing_file_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "INFERENCE_OPS_ENV_FILE does not exist"):
            load_runtime_environment({"INFERENCE_OPS_ENV_FILE": str(self.path)})

    def test_default_dotenv_is_local_to_package_and_optional(self):
        with patch("inference_ops.environment.__file__", str(Path(self.temp.name) / "inference_ops/environment.py")):
            env = {}
            load_runtime_environment(env)
            self.assertEqual(env, {})
            (Path(self.temp.name) / ".env").write_text("AGENT_ROLE=both\n", encoding="utf-8")
            load_runtime_environment(env)
            self.assertEqual(env, {"AGENT_ROLE": "both"})

    def test_cli_loads_env_before_importing_app(self):
        self.path.write_text("STUDIO_URL=http://studio:3000\nMEDFLOW_RUNTIME_NODE_TOKEN=node-secret\n", encoding="utf-8")
        # A tiny fake app reads settings at import, as the real app does. This
        # executes the actual CLI without requiring uvicorn/langchain locally.
        source = '''
import importlib.abc, importlib.util, sys, types
sys.path.insert(0, sys.argv[1])
class FakeApp(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'inference_ops.inference_agent':
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec): return None
    def exec_module(self, module):
        from inference_ops.environment import load_runtime_environment
        load_runtime_environment()
        import os
        assert os.environ['STUDIO_URL'] == 'http://studio:3000'
        assert os.environ['MEDFLOW_RUNTIME_NODE_TOKEN'] == 'node-secret'
        module.INFERENCE_AGENT_HOST = '127.0.0.1'
        module.INFERENCE_AGENT_PORT = 8899
        module.app = object()
        module.main = lambda: None
sys.meta_path.insert(0, FakeApp())
sys.modules['uvicorn'] = types.SimpleNamespace(run=lambda *args, **kwargs: print('configured'))
from inference_ops.__main__ import main
sys.argv = ['inference_ops', '--env-file', sys.argv[2]]
main()
'''
        env = {key: value for key, value in os.environ.items()
               if key not in {"STUDIO_URL", "MEDFLOW_RUNTIME_NODE_TOKEN", "INFERENCE_OPS_ENV_FILE"}}
        result = subprocess.run([sys.executable, "-B", "-c", source, str(REPO / "inference-ops"), str(self.path)],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("configured", result.stdout)


class StudioCredentialTests(unittest.TestCase):
    def test_both_trackers_accept_new_and_legacy_tokens_in_real_requests(self):
        for module in (worker_tracker, agent_tracker):
            for token_env in ({"MEDFLOW_RUNTIME_NODE_TOKEN": "node-secret"},
                              {"MEDFLOW_STUDIO_NODE_TOKEN": "node-secret"},
                              {"MEDFLOW_RUNTIME_NODE_TOKEN": "node-secret", "MEDFLOW_STUDIO_NODE_TOKEN": "node-secret"}):
                with self.subTest(module=module.__name__, keys=list(token_env)):
                    def respond(request, timeout):
                        self.assertEqual(request.full_url, "http://studio:3000/trpc/renewTrainingResourcesForRuntime")
                        self.assertEqual(request.get_header("X-medflow-runtime-token"), "node-secret")
                        self.assertEqual(json.loads(request.data), {"runtimeNodeId": "node-1"})
                        return io.BytesIO(b'{"result":{"data":{"json":{"success":true}}}}')
                    with patch.dict(os.environ, {"STUDIO_URL": " http://studio:3000/ ", "MEDFLOW_RESOURCE_NODE_ID": "node-1", **token_env}, clear=True), \
                         patch.object(module, "urlopen", side_effect=respond):
                        self.assertEqual(module.studio_request("renewTrainingResourcesForRuntime", {"runtimeNodeId": "node-1"}), {"success": True})

    def test_missing_or_conflicting_credentials_never_send_request(self):
        for module in (worker_tracker, agent_tracker):
            for env in ({}, {"STUDIO_URL": "http://studio"}, {"MEDFLOW_RUNTIME_NODE_TOKEN": "private-a"},
                        {"STUDIO_URL": "http://studio", "MEDFLOW_RUNTIME_NODE_TOKEN": "private-a", "MEDFLOW_STUDIO_NODE_TOKEN": "private-b"}):
                with self.subTest(module=module.__name__, keys=list(env)), patch.dict(os.environ, env, clear=True), \
                     patch.object(module, "urlopen") as send:
                    with self.assertRaises(ValueError) as caught:
                        module.studio_request("test", {})
                    self.assertNotIn("private-a", str(caught.exception))
                    self.assertNotIn("private-b", str(caught.exception))
                    send.assert_not_called()


class WorkerCollectorTests(unittest.TestCase):
    def test_bundled_worker_runs_custom_collector_without_generic_module(self):
        self.assertFalse((REPO / "inference-ops/inference_ops/experiment_runtime/experiment_collector.py").exists())
        with tempfile.TemporaryDirectory() as root:
            called = threading.Event()
            errors = []
            tracker = worker_tracker.ExperimentTracker(Path(root) / "records.sqlite3", "node-1", collector_fn=lambda current: called.set())
            original_wait = tracker.stop_event.wait
            with patch.object(tracker.stop_event, "wait", side_effect=lambda delay: original_wait(min(delay, 0.01))), \
                 patch.object(threading, "excepthook", side_effect=errors.append):
                try:
                    tracker.start()
                    self.assertTrue(called.wait(2), "custom collector was never invoked")
                    self.assertTrue(tracker.collector_worker.is_alive())
                    tracker.start()  # Already running: does not create another thread.
                finally:
                    tracker.close()
            self.assertEqual(errors, [])
            self.assertFalse(tracker.collector_worker.is_alive())

    def test_missing_default_collector_fails_before_starting_any_threads(self):
        with tempfile.TemporaryDirectory() as root:
            tracker = worker_tracker.ExperimentTracker(Path(root) / "records.sqlite3", "node-1")
            with self.assertRaises(ModuleNotFoundError):
                tracker.start()
            self.assertIsNone(tracker.worker)
            self.assertIsNone(tracker.collector_worker)


class WorkerReadinessTests(unittest.TestCase):
    def test_configuration_preflight_does_not_create_tracker(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(experiment_records, "get_tracker") as create:
            self.assertIn("STUDIO_URL", experiment_records.experiment_configuration_error())
            create.assert_not_called()
        with patch.dict(os.environ, {"STUDIO_URL": "http://studio", "MEDFLOW_RUNTIME_NODE_TOKEN": "secret"}, clear=True):
            self.assertIn("MEDFLOW_RESOURCE_NODE_ID", experiment_records.experiment_configuration_error())
            with patch.dict(os.environ, {"MEDFLOW_RESOURCE_NODE_ID": "node-1"}):
                self.assertEqual(experiment_records.experiment_configuration_error(), "")

    def test_worker_readiness_reports_credentials_but_controller_does_not_require_them(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.yaml"
            config.write_text("", encoding="utf-8")
            namespace = {"os": os, "INFERENCE_OPS_AUTH_MODE": "trusted_network", "INFERENCE_OPS_API_TOKEN": "",
                "AGENT_CONFIG_FILE": str(config), "AGENT_ROLE": "worker",
                "ops_tools": types.SimpleNamespace(CONFIG_FILE=str(config), STATE_DIR=root),
                "experiment_configuration_error": experiment_records.experiment_configuration_error,
                "JSONResponse": lambda **kwargs: kwargs}
            ready = load_function(REPO / "inference-ops/inference_ops/inference_agent.py", "readyz", namespace)
            with patch.dict(os.environ, {}, clear=True):
                result = ready()
                self.assertEqual(result["status_code"], 503)
                self.assertIn("STUDIO_URL", result["content"]["missing"][0])
                namespace["AGENT_ROLE"] = "controller"
                self.assertEqual(ready()["status"], "ready")

    def test_missing_credentials_stop_start_before_reserving_ports_or_snapshotting(self):
        package_name = "_worker_preflight_test"
        dependency = types.SimpleNamespace(experiment_configuration_error=lambda: "missing STUDIO_URL")
        namespace = {"__package__": package_name, "copy": __import__("copy"),
            "current_request_user_id": lambda: "user-1", "managed_resource_error": lambda **kwargs: "",
            "ensure_user_draft_config": lambda: "draft.yaml", "show_config": lambda: {},
            "apply_runtime_host_ip": lambda cfg: ("127.0.0.1", "127.0.0.1"),
            "check_service_start_static_config": lambda cfg: {"ok": True}}
        start = load_function(REPO / "inference-ops/inference_ops/tools.py", "start_service", namespace)
        with patch.dict(sys.modules, {f"{package_name}.experiment_records": dependency}):
            result = start()
        self.assertIn("missing STUDIO_URL", result)
        self.assertIn("--env-file", result)


if __name__ == "__main__":
    unittest.main()
