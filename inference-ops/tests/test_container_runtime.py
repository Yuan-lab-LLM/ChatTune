"""Cross-container configuration and ownership tests; no Docker/GPU required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "agent" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("inference_container")
remote = load("inference_container_runtime")


class ContainerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state/inference.json"
        self.env = {"START_LOCAL_INFERENCE": "auto", "MEDFLOW_LOCAL_EVALUATE_CONTAINER": "infer-container", "INFERENCE_AGENT_URL": "http://127.0.0.1:8899/inference_agent",
            "STUDIO_URL": "http://studio:3000", "MEDFLOW_RESOURCE_NODE_ID": "node-1", "MEDFLOW_RUNTIME_NODE_TOKEN": "private-node-token",
            "MEDFLOW_AGENT_API_TOKEN": "private-agent-token", "MEDFLOW_STUDIO_RUNTIME_TOKEN": "private-center-token",
            "MODEL_BASE_URL": "http://llm:7000/v1", "MODEL_NAME": "agent-model", "MODEL_API_KEY": "private-model-token"}
        self.inspection = {"State": {"Running": True}, "Config": {"WorkingDir": "/srv/inference-root"},
                           "Mounts": [{"Source": "/host/other-path", "Destination": "/custom/models-and-code"}]}
        self.calls = []

    def docker(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["docker", "inspect"]:
            return types.SimpleNamespace(returncode=0, stdout=json.dumps([self.inspection]))
        payload = json.loads(kwargs["input"])
        status = {"start": "ready", "status": "ready", "stop": "stopped", "check": "configured", "logs": "ok"}[payload["operation"]]
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"status": status, "directory": "/custom/models-and-code/inference-ops", "managed": True}))

    def test_remote_endpoint_is_left_external_by_default(self):
        result = launcher.manage("start", self.state, {**self.env, "INFERENCE_AGENT_URL": "http://other-node:8899/inference_agent"}, runner=self.docker)
        self.assertEqual(result["status"], "external")
        self.assertEqual(self.calls, [])
        self.assertFalse(self.state.exists())

    def test_explicit_disable_does_not_call_docker(self):
        self.assertEqual(launcher.manage("start", self.state, {**self.env, "START_LOCAL_INFERENCE": "false"}, runner=self.docker)["status"], "external")
        self.assertEqual(self.calls, [])

    def test_manual_deployment_is_default_even_with_local_container_configured(self):
        env = {key: value for key, value in self.env.items() if key != "START_LOCAL_INFERENCE"}
        for operation in ("start", "check", "status", "stop", "logs"):
            with self.subTest(operation=operation):
                self.assertEqual(launcher.manage(operation, self.state, env, runner=self.docker)["status"], "external")
        self.assertEqual(self.calls, [])
        self.assertIsNone(launcher.local_inference_settings(env))

    def test_disabled_management_does_not_touch_existing_automatic_controller(self):
        launcher.manage("start", self.state, self.env, runner=self.docker)
        saved = self.state.read_text()
        self.calls.clear()
        for operation in ("start", "check", "status", "stop", "logs"):
            with self.subTest(operation=operation):
                self.assertEqual(launcher.manage(operation, self.state, {**self.env, "START_LOCAL_INFERENCE": "false"}, runner=self.docker)["status"], "external")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state.read_text(), saved)

    def test_start_uses_container_mounts_and_stdin_without_copying_other_credentials(self):
        result = launcher.manage("start", self.state, self.env, runner=self.docker)
        self.assertEqual(result["status"], "ready")
        args, options = self.calls[-1]
        self.assertEqual(args[:6], ["docker", "exec", "-i", "infer-container", "python3", "-c"])
        self.assertNotIn("private-node-token", " ".join(args))
        payload = json.loads(options["input"])
        self.assertIn("/custom/models-and-code/inference-ops", payload["candidates"])
        self.assertNotIn("/host/other-path", payload["candidates"])
        self.assertNotIn(str(REPO), payload["candidates"])
        self.assertEqual(payload["environment"]["MEDFLOW_RUNTIME_NODE_TOKEN"], "private-node-token")
        self.assertNotIn("MEDFLOW_AGENT_API_TOKEN", payload["environment"])
        self.assertNotIn("MEDFLOW_STUDIO_RUNTIME_TOKEN", payload["environment"])
        self.assertEqual(payload["llm_defaults"]["AGENT_LLM_MODEL"], "agent-model")
        self.assertNotIn("private-", self.state.read_text())

    def test_stop_uses_recorded_container_and_path_after_settings_change(self):
        launcher.manage("start", self.state, self.env, runner=self.docker)
        self.calls.clear()
        launcher.manage("stop", self.state, {**self.env, "MEDFLOW_LOCAL_EVALUATE_CONTAINER": "other-container",
            "MEDFLOW_INFERENCE_OPS_DIR": "/other-root", "INFERENCE_AGENT_URL": "http://remote:8899/inference_agent"}, runner=self.docker)
        args, kwargs = self.calls[-1]
        self.assertEqual(args[3], "infer-container")
        self.assertEqual(json.loads(kwargs["input"])["candidates"], ["/custom/models-and-code/inference-ops"])
        self.assertNotIn("private-node-token", kwargs["input"])
        self.assertFalse(self.state.exists())

    def test_unmanaged_stop_does_not_contact_any_container(self):
        self.assertEqual(launcher.manage("stop", self.state, self.env, runner=self.docker)["status"], "not_managed")
        self.assertEqual(self.calls, [])

    def test_custom_container_path_python_and_port_are_supported(self):
        env = {**self.env, "MEDFLOW_INFERENCE_CONTAINER": "custom-inference", "MEDFLOW_INFERENCE_PYTHON": "/venv/bin/python",
               "MEDFLOW_INFERENCE_OPS_DIR": "/independent/inference code", "INFERENCE_AGENT_URL": "http://127.0.0.1:9900/inference_agent"}
        launcher.manage("start", self.state, env, runner=self.docker)
        args, options = self.calls[-1]
        self.assertEqual(args[3:5], ["custom-inference", "/venv/bin/python"])
        payload = json.loads(options["input"])
        self.assertEqual(payload["candidates"], ["/independent/inference code"])
        self.assertEqual(payload["port"], 9900)

    def test_failed_launch_does_not_record_managed_state(self):
        def fail(args, **kwargs):
            result = self.docker(args, **kwargs)
            if args[:2] == ["docker", "exec"]:
                return types.SimpleNamespace(returncode=1, stdout=json.dumps({"status": "error", "error": "startup failed"}))
            return result
        with self.assertRaisesRegex(ValueError, "startup failed"):
            launcher.manage("start", self.state, self.env, runner=fail)
        self.assertFalse(self.state.exists())

    def test_stopped_container_can_be_stopped_again_without_error(self):
        launcher.manage("start", self.state, self.env, runner=self.docker)
        self.inspection["State"]["Running"] = False
        self.calls.clear()
        self.assertEqual(launcher.manage("stop", self.state, self.env, runner=self.docker)["status"], "stopped")
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.state.exists())

    def test_bad_node_configuration_fails_without_sending_secrets(self):
        for updates in ({"MEDFLOW_RUNTIME_NODE_TOKEN": ""}, {"MEDFLOW_STUDIO_NODE_TOKEN": "other-secret"}):
            with self.subTest(updates=updates):
                self.calls.clear()
                with self.assertRaises(ValueError) as error:
                    launcher.manage("start", self.state, {**self.env, **updates}, runner=self.docker)
                self.assertNotIn("private-", str(error.exception))
                self.assertFalse(any(args[:2] == ["docker", "exec"] for args, _ in self.calls))


class RemoteBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "different mount/inference-ops"
        (self.root / "inference_ops").mkdir(parents=True)
        (self.root / "inference_ops/__main__.py").write_text("", encoding="utf-8")
        self.payload = {"operation": "start", "candidates": [str(self.root)], "port": 8899,
            "environment": {"STUDIO_URL": "http://studio", "MEDFLOW_RUNTIME_NODE_TOKEN": "secret", "MEDFLOW_RESOURCE_NODE_ID": "node-1"},
            "llm_defaults": {"AGENT_LLM_MODEL": "agent-model"}}
        self.record = self.root / "runtime/runtime-controller.json"
        self.fcntl = patch.dict(sys.modules, {"fcntl": types.SimpleNamespace(LOCK_EX=1, flock=lambda *args: None)})
        self.fcntl.start()
        self.addCleanup(self.fcntl.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        for name, value in (("prepare_environment", lambda *args: dict(self.payload["environment"])),
                            ("owned_process", lambda record: bool(record.get("pid"))),
                            ("process_identity", lambda pid: "identity-1")):
            patched = patch.object(remote, name, side_effect=value)
            patched.start()
            self.addCleanup(patched.stop)
        dependencies = patch.object(remote.importlib.util, "find_spec", return_value=object())
        dependencies.start()
        self.addCleanup(dependencies.stop)

    def test_discovery_uses_paths_inside_container_and_requires_real_package(self):
        self.assertEqual(remote.locate_ops([str(self.root.parent), str(self.root)]), self.root.resolve())
        with self.assertRaisesRegex(ValueError, "MEDFLOW_INFERENCE_OPS_DIR"):
            remote.locate_ops(["/not-mounted"])

    def test_start_records_ownership_and_waits_for_its_own_ready_response(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        with patch.object(remote, "ready", side_effect=[False, True]) as ready, patch.object(remote.subprocess, "Popen", return_value=child) as spawn:
            result = remote.run(self.payload)
        self.assertEqual(result["status"], "ready")
        record = json.loads(self.record.read_text())
        self.assertEqual(record["pid"], 123)
        self.assertEqual(record["identity"], "identity-1")
        self.assertNotIn("secret", self.record.read_text())
        args, kwargs = spawn.call_args
        self.assertEqual(args[0], [sys.executable, "-m", "inference_ops"])
        self.assertEqual(kwargs["cwd"], self.root.resolve())
        self.assertEqual(kwargs["env"]["MEDFLOW_RUNTIME_NODE_TOKEN"], "secret")
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(ready.call_args.args[2], record["launch_id"])

    def test_existing_managed_process_is_reused_without_duplicate_spawn(self):
        self.record.parent.mkdir()
        self.record.write_text(json.dumps({"pid": 123, "identity": "identity-1", "launch_id": "launch-1"}))
        with patch.object(remote, "ready", return_value=True), patch.object(remote.subprocess, "Popen") as spawn:
            self.assertEqual(remote.run(self.payload)["pid"], 123)
            spawn.assert_not_called()

    def test_external_ready_process_is_reused_without_adopting_or_signalling(self):
        with patch.object(remote, "ready", return_value=True), patch.object(remote.subprocess, "Popen") as spawn:
            result = remote.run(self.payload)
        self.assertFalse(result["managed"])
        self.assertFalse(self.record.exists())
        spawn.assert_not_called()

    def test_port_conflict_does_not_kill_or_spawn_processes(self):
        probe = Mock()
        probe.__enter__ = Mock(return_value=probe)
        probe.__exit__ = Mock(return_value=False)
        probe.bind.side_effect = OSError("busy")
        with patch.object(remote, "ready", return_value=False), patch.object(remote.socket, "socket", return_value=probe), \
             patch.object(remote, "stop_owned") as stop, patch.object(remote.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "已被其他进程占用"):
                remote.run(self.payload)
            stop.assert_not_called()
            spawn.assert_not_called()

    def test_failed_child_is_rolled_back_and_record_is_removed(self):
        child = Mock(pid=123)
        child.poll.return_value = 1
        with patch.object(remote, "ready", return_value=False), patch.object(remote.subprocess, "Popen", return_value=child), \
             patch.object(remote, "stop_owned") as stop:
            with self.assertRaisesRegex(ValueError, "未就绪"):
                remote.run(self.payload)
            self.assertEqual(stop.call_args.args[0]["pid"], 123)
        self.assertFalse(self.record.exists())

    def test_stop_only_delegates_to_owned_process_guard(self):
        self.record.parent.mkdir()
        self.record.write_text(json.dumps({"pid": 123, "identity": "identity-1", "launch_id": "launch-1"}))
        with patch.object(remote, "stop_owned", return_value=True) as stop:
            result = remote.run({**self.payload, "operation": "stop"})
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(stop.call_args.args[0]["pid"], 123)
        self.assertFalse(self.record.exists())

    def test_invalid_agent_configuration_never_spawns_controller_or_records_pid(self):
        with patch.object(remote, "prepare_environment", side_effect=ValueError("agent.yaml: LLM_MODEL 未配置")), \
             patch.object(remote.subprocess, "Popen") as spawn:
            for operation in ("check", "start"):
                with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "LLM_MODEL"):
                    remote.run({**self.payload, "operation": operation})
            spawn.assert_not_called()
        self.assertFalse(self.record.exists())


class RemoteIdentityTests(unittest.TestCase):
    def test_pid_reuse_never_authorizes_stop(self):
        with patch.object(remote, "process_identity", return_value="other-start"), patch.object(remote.os, "killpg", create=True) as kill:
            self.assertFalse(remote.stop_owned({"pid": 42, "identity": "old-start", "launch_id": "launch"}))
            kill.assert_not_called()

    def test_matching_pid_requires_environment_marker_and_process_group(self):
        record = {"pid": 42, "identity": "start", "launch_id": "launch"}
        for environment, group, expected in ((b"MEDFLOW_INFERENCE_LAUNCH_ID=launch\0", 42, True),
                                              (b"MEDFLOW_INFERENCE_LAUNCH_ID=other\0", 42, False),
                                              (b"MEDFLOW_INFERENCE_LAUNCH_ID=launch\0", 99, False)):
            with self.subTest(group=group, expected=expected), patch.object(remote, "process_identity", return_value="start"), \
                 patch.object(Path, "read_bytes", return_value=environment), patch.object(remote.os, "getpgid", return_value=group, create=True):
                self.assertEqual(remote.owned_process(record), expected)

    def test_readiness_rejects_different_node_or_launch(self):
        valid = {"status": "ready", "service": "inference-ops", "role": "both", "runtime_node_id": "node-1", "runtime_launch_id": "launch-1"}
        for updates, expected in (({}, True), ({"runtime_node_id": "node-2"}, False), ({"runtime_launch_id": "old"}, False), ({"role": "controller"}, False)):
            with self.subTest(updates=updates):
                response = io.BytesIO(json.dumps({**valid, **updates}).encode())
                opener = types.SimpleNamespace(open=lambda *args, **kwargs: response)
                with patch.object(remote, "build_opener", return_value=opener):
                    self.assertEqual(remote.ready(8899, "node-1", "launch-1"), expected)


class InferenceDefaultsTests(unittest.TestCase):
    def prepare(self, config, local_env, defaults=None):
        with tempfile.TemporaryDirectory() as temp, patch.object(sys, "path", list(sys.path)):
            root = Path(temp)
            (root / "config").mkdir()
            (root / "config/agent.yaml").write_text("config", encoding="utf-8")
            loader = types.SimpleNamespace(load_runtime_environment=lambda environ: environ.update(local_env))
            incoming = {"STUDIO_URL": "http://studio", "MEDFLOW_RESOURCE_NODE_ID": "node-1", "MEDFLOW_RUNTIME_NODE_TOKEN": "current-token"}
            if defaults is None:
                defaults = {"AGENT_LLM_URL": "http://agent-llm/v1", "AGENT_LLM_MODEL": "agent-model", "AGENT_LLM_API_KEY": "model-token"}
            with patch.dict(os.environ, {}, clear=True), patch("importlib.import_module", return_value=loader), \
                 patch.dict(sys.modules, {"yaml": types.SimpleNamespace(safe_load=lambda text: config)}):
                return remote.prepare_environment(root, incoming, defaults, 9900)

    def test_empty_config_reuses_agent_model_and_overrides_stale_node_credentials(self):
        env = self.prepare({}, {"MEDFLOW_STUDIO_NODE_TOKEN": "stale-token"})
        self.assertEqual(env["AGENT_LLM_MODEL"], "agent-model")
        self.assertEqual(env["AGENT_LLM_URL"], "http://agent-llm/v1")
        self.assertEqual(env["MEDFLOW_STUDIO_NODE_TOKEN"], "current-token")
        self.assertEqual(env["MEDFLOW_RUNTIME_NODE_TOKEN"], "current-token")
        self.assertEqual(env["AGENT_ROLE"], "both")
        self.assertEqual(env["INFERENCE_AGENT_PORT"], "9900")
        self.assertEqual(env["MEDFLOW_RESOURCE_POOL_ENABLED"], "true")

    def test_explicit_inference_model_settings_are_preserved(self):
        env = self.prepare({"LLM_MODEL": "inference-model", "LLM_URL": "http://inference-llm/v1", "ROLE": "worker"}, {"AGENT_LLM_API_KEY": "inference-key"})
        self.assertNotIn("AGENT_LLM_MODEL", env)
        self.assertNotIn("AGENT_LLM_URL", env)
        self.assertEqual(env["AGENT_LLM_API_KEY"], "inference-key")
        self.assertEqual(env["AGENT_ROLE"], "worker")

    def test_unedited_example_reuses_runtime_url_and_model_as_one_configuration(self):
        config = {"LLM_URL": "http://127.0.0.1:6000/v1", "LLM_MODEL": "example-model-name",
                  "LLM_API_KEY": "example-key", "ROLE": "both"}
        before = dict(config)
        env = self.prepare(config, {})
        self.assertEqual(env["AGENT_LLM_URL"], "http://agent-llm/v1")
        self.assertEqual(env["AGENT_LLM_MODEL"], "agent-model")
        self.assertEqual(env["AGENT_LLM_API_KEY"], "model-token")
        self.assertEqual(env["MEDFLOW_INFERENCE_LLM_SOURCE"], "runtime")
        self.assertEqual(config, before)

    def test_empty_config_and_missing_runtime_settings_report_actionable_fields(self):
        with self.assertRaises(ValueError) as error:
            self.prepare({}, {}, defaults={})
        for field in ("agent.yaml", "LLM_URL", "LLM_MODEL", "MODEL_BASE_URL", "MODEL_NAME", "runtime.sh start"):
            self.assertIn(field, str(error.exception))

    def test_runtime_placeholders_are_not_accepted_as_fallback(self):
        with self.assertRaisesRegex(ValueError, "LLM_MODEL"):
            self.prepare({}, {}, defaults={"AGENT_LLM_URL": "http://agent-llm/v1", "AGENT_LLM_MODEL": "replace-model-name"})

    def test_explicit_environment_overrides_yaml_without_requiring_runtime_defaults(self):
        env = self.prepare({"LLM_URL": "<model-host>", "LLM_MODEL": "example-model-name"},
            {"AGENT_LLM_URL": "https://custom.example/v1", "AGENT_LLM_MODEL": "custom-model"}, defaults={})
        self.assertEqual(env["AGENT_LLM_MODEL"], "custom-model")
        self.assertEqual(env["AGENT_LLM_URL"], "https://custom.example/v1")

    def test_partial_explicit_environment_does_not_mix_with_runtime_model(self):
        with self.assertRaisesRegex(ValueError, "LLM_MODEL"):
            self.prepare({}, {"AGENT_LLM_URL": "http://independent-service/v1"})

    def test_invalid_config_shape_role_url_and_backend_are_rejected(self):
        invalid = [(["ROLE", "both"], "顶层"),
                   ({"ROLE": "controller"}, "ROLE"),
                   ({"MEMORY": "redis"}, "MEMORY"),
                   ({"MEMORY": {"BACKEND": "unknown"}}, "MEMORY.BACKEND"),
                   ({"LLM_URL": "not-a-url", "LLM_MODEL": "valid-model"}, "LLM_URL"),
                   ({"LLM_URL": "https://llm:wrong/v1", "LLM_MODEL": "valid-model"}, "LLM_URL"),
                   ({"LLM_URL": "http://llm/v1", "LLM_MODEL": "valid-model", "LLM_API_KEY": "replace-key"}, "LLM_API_KEY")]
        for config, field in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, field):
                self.prepare(config, {})

    def test_parser_error_reports_location_without_secret_source_line(self):
        class InvalidYaml(Exception):
            problem_mark = types.SimpleNamespace(line=3)
        with tempfile.TemporaryDirectory() as temp, patch.object(sys, "path", list(sys.path)):
            root = Path(temp)
            (root / "config").mkdir()
            config_path = root / "config/agent.yaml"
            config_path.write_text("PRIVATE_CONFIG_LINE", encoding="utf-8")
            loader = types.SimpleNamespace(load_runtime_environment=lambda environ: None)
            with patch.dict(os.environ, {}, clear=True), patch("importlib.import_module", return_value=loader), \
                 patch.dict(sys.modules, {"yaml": types.SimpleNamespace(safe_load=Mock(side_effect=InvalidYaml("PRIVATE_CONFIG_LINE")))}):
                with self.assertRaises(ValueError) as error:
                    remote.prepare_environment(root, {"MEDFLOW_RUNTIME_NODE_TOKEN": "secret"}, {}, 8899)
            self.assertIn("agent.yaml", str(error.exception))
            self.assertIn("第 4 行", str(error.exception))
            self.assertNotIn("PRIVATE_CONFIG_LINE", str(error.exception))
            self.assertEqual(config_path.read_text(), "PRIVATE_CONFIG_LINE")


if __name__ == "__main__":
    unittest.main()
