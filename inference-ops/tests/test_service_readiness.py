import ast
import importlib.util
import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "inference-ops" / "inference_ops" / "tools.py"
RUNTIME = REPO / "agent" / "runtime_agent.py"
MANAGER = REPO / "agent" / "utils" / "workflow_manager.py"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ServiceReadinessTests(unittest.TestCase):
    def test_listening_port_does_not_finish_startup(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run"
            run_dir.mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({
                    "run_id": "run",
                    "status": "starting",
                    "script_pid": 123,
                    "ports": {"vllm": 7111},
                }), encoding="utf-8",
            )
            namespace = {
                "os": os,
                "json": json,
                "time": time,
                "get_service_run_log_root": lambda: root,
                "get_service_log_dir": lambda _run_id: str(run_dir),
                "is_process_running": lambda _pid: True,
                "check_port_status": lambda _port: True,
                "service_instance_is_legacy": lambda _meta: False,
                "SERVICE_TASK_REQUIRED_PORTS": ("vllm",),
                "LEGACY_SERVICE_TASK_REQUIRED_PORTS": ("vllm",),
            }
            status = load_function(TOOLS, "service_start_status_text", namespace)("run")
            self.assertIn("status=starting", status)
            self.assertNotIn("status=finished", status)

    def test_api_probe_bypasses_proxy_and_reports_http_500(self):
        namespace = {"urllib": urllib}
        probe = load_function(TOOLS, "openai_api_ready_error", namespace)
        cfg = {"ENV": {"HOST_IP": "10.0.0.1"}, "PORTS": {"VLLM_OPENAI_PORT": 7111}}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        opener = mock.Mock()
        opener.open.return_value = Response()
        with mock.patch.object(urllib.request, "build_opener", return_value=opener) as build:
            self.assertEqual("", probe(cfg, "instance-1", {"status": "running"}))
            self.assertEqual({}, build.call_args.args[0].proxies)

            opener.open.side_effect = urllib.error.HTTPError(
                "http://10.0.0.1:7111/v1/models", 500, "Internal Server Error", {}, None
            )
            error = probe(cfg, "instance-1", {"status": "running"})
            self.assertIn("HTTP Error 500", error)
            self.assertIn("instance_status=running", error)

    def test_unready_api_does_not_create_benchmark_job(self):
        namespace = {
            "resolve_task_service_instance_id": lambda _id, _task: "instance-1",
            "require_running_service_instance_config": lambda _id: (
                "instance-1", {"ENV": {}, "PORTS": {}}, {"status": "running"}
            ),
            "openai_api_ready_error": lambda *_args: "HTTP Error 500",
            "acquire_benchmark_submit_lock": lambda *_args: self.fail("job lock acquired"),
        }
        run = load_function(TOOLS, "benchmark_run_unified", namespace)
        self.assertEqual("HTTP Error 500", run("math"))

    def test_manual_benchmark_reports_starting_instance(self):
        namespace = {
            "current_request_user_id": lambda: "user-1",
            "service_instance_access_error": lambda *_args: "",
            "get_running_service_instance_config": lambda _id: (
                None, None, {"instance_id": "instance-1", "status": "starting"}
            ),
        }
        require = load_function(TOOLS, "require_running_service_instance_config", namespace)
        result = require("instance-1")
        self.assertIn("仍在启动中", result)
        self.assertIn("instance_id=instance-1", result)

    def test_workflow_waits_for_target_then_advances(self):
        state = {"status": "starting", "all_running": False, "instance_id": "instance-1"}

        def query(*_args, **_kwargs):
            return {
                "all_running": state["all_running"],
                "protocol": {"nodes": {"local": {"service_instances": {"items": [{
                    "instance_id": state["instance_id"], "model": "model-a",
                    "status": state["status"], "error": state.get("error"),
                    "started_at": "2026-09-17 18:54:44",
                }]}}}},
            }

        namespace = {
            "os": os,
            "time": time,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "config": SimpleNamespace(workflow=SimpleNamespace(poll_interval=1)),
            "_workflow_model_identifier": lambda path: os.path.basename(path),
            "_workflow_stage_log_fields": lambda *_args: {},
            "_workflow_evaluation_container": lambda _workflow: "container",
            "_workflow_inference_owner_kwargs": lambda _context: {},
            "_workflow_inference_command": query,
            "_workflow_context_inference_resource_context": lambda _context: None,
            "_workflow_inference_service_matches_resource": lambda checked, *_args, **_kwargs: checked["all_running"],
            "_append_workflow_stage_log": lambda *_args: {},
        }
        namespace["_workflow_target_service_instance"] = load_function(
            RUNTIME, "_workflow_target_service_instance", namespace
        )
        monitor = load_function(RUNTIME, "_workflow_deploy_service_status", namespace)
        workflow = {
            "context": {"published_model_path": "/models/model-a"},
            "stages": {"deploy": {"status": "running"}},
        }
        waiting = monitor(workflow)
        self.assertEqual("running", waiting["status"])
        self.assertFalse(waiting["all_running"])
        state.update(status="running", all_running=True)
        self.assertEqual("finished", monitor(workflow)["status"])
        state.update(status="failed", all_running=False)
        state["error"] = "prometheus-fastapi-instrumentator needs an update"
        with self.assertRaisesRegex(RuntimeError, "prometheus-fastapi-instrumentator needs an update"):
            monitor(workflow)
        state.update(status="stopped", all_running=False)
        workflow["stages"]["deploy"]["service_wait_started_at"] = time.time() - 1901
        workflow["stages"]["deploy"]["service_wait_instance_id"] = "instance-1"
        self.assertTrue(monitor(workflow)["restart_required"])
        state.update(status="starting", all_running=False)
        with self.assertRaisesRegex(RuntimeError, "超时"):
            monitor(workflow)
        state["instance_id"] = "instance-2"
        renewed = monitor(workflow)
        self.assertEqual("running", renewed["status"])
        self.assertEqual("instance-2", renewed["service_wait_instance_id"])
        self.assertGreater(renewed["service_wait_started_at"], time.time() - 10)
        workflow["stages"]["deploy"].update(renewed)
        workflow["stages"]["deploy"]["next_service_check_at"] = 0
        state.update(status="running", all_running=True)
        workflow["stages"]["deploy"]["service_wait_started_at"] = time.time() - 1901
        self.assertEqual("finished", monitor(workflow)["status"])
        state.update(status="starting", all_running=False)
        workflow["stages"]["deploy"]["service_wait_started_at"] = time.time()
        workflow["stages"]["deploy"]["next_service_check_at"] = 0
        previous_query = namespace["_workflow_inference_command"]
        namespace["_workflow_inference_command"] = lambda *_args, **_kwargs: {
            **previous_query(),
            "result": (
                "instance-2: /v1/models HTTP 500; "
                "AttributeError: '_IncludedRouter' object has no attribute 'path'"
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "prometheus_fastapi_instrumentator"):
            monitor(workflow)

    def test_workflow_does_not_restart_loading_instance(self):
        status = {
            "all_running": False,
            "protocol": {"service_instances": {"items": [{
                "model": "model-a", "status": "starting", "started_at": "2026-09-17 18:54:44"
            }]}},
        }
        namespace = {
            "os": os,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "_workflow_evaluation_container": lambda _workflow: "container",
            "_workflow_inference_owner_kwargs": lambda _context: {},
            "_workflow_context_inference_resource_context": lambda _context: None,
            "_workflow_inference_command": lambda command, *_args, **_kwargs: (
                status if command == "查看推理服务状态" else self.fail("unexpected restart")
            ),
            "_workflow_inference_service_matches_resource": lambda *_args, **_kwargs: False,
        }
        namespace["_workflow_target_service_instance"] = load_function(
            RUNTIME, "_workflow_target_service_instance", namespace
        )
        ensure = load_function(RUNTIME, "_workflow_ensure_benchmark_inference_ready", namespace)
        result = ensure({"context": {"published_model_path": "/models/model-a"}})
        self.assertTrue(result["waiting_service"])
        status["protocol"]["service_instances"]["items"][0]["status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "启动失败"):
            ensure({"context": {"published_model_path": "/models/model-a"}})

    def test_workflow_matches_target_despite_unrelated_startup(self):
        namespace = {
            "os": os,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "_workflow_expected_inference_gpus": lambda _context: set(),
        }
        namespace["_workflow_target_service_instance"] = load_function(
            RUNTIME, "_workflow_target_service_instance", namespace
        )
        matches = load_function(RUNTIME, "_workflow_inference_service_matches_resource", namespace)
        status = {
            "all_running": False,
            "protocol": {"service_instances": {"items": [
                {"model": "model-a", "status": "running", "started_at": "2026-09-17 19:00:00"},
                {"model": "other-model", "status": "starting", "started_at": "2026-09-17 19:01:00"},
            ]}},
        }
        self.assertTrue(matches(status, None, target_model_path="/models/model-a"))
        self.assertFalse(matches(status, None, target_model_path="/models/other-model"))

    def test_workflow_defers_then_submits_once(self):
        spec = importlib.util.spec_from_file_location("workflow_manager_readiness_test", MANAGER)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        starts = []

        def start_benchmark(_workflow):
            starts.append(1)
            if len(starts) == 1:
                return {"waiting_service": True, "success": True}
            return {"success": True, "status": "running", "benchmark_job_id": "job-1"}

        noop = lambda *_args, **_kwargs: {}
        deps = module.WorkflowDependencies(
            start_train=noop, monitor_train=noop, find_trained_model=lambda _wf: "",
            start_evaluate=noop, monitor_evaluate=noop, start_publish=noop,
            monitor_publish=noop, inference_command=lambda *_args, **_kwargs: self.fail("unexpected restart"),
            start_benchmark=start_benchmark,
            monitor_deploy=lambda _wf: {"status": "finished", "all_running": True},
        )
        with tempfile.TemporaryDirectory() as root:
            manager = module.WorkflowManager(
                str(Path(root) / "workflows.sqlite"), deps,
                poll_interval=1, auto_start_worker=False, agent_events=False,
            )
            manager.start_worker = lambda: None
            try:
                workflow = manager.create("user-1", "dataset-1", auto_start=False)
                workflow["current_stage"] = "benchmark"
                workflow["context"]["benchmark"] = "math"
                workflow["stages"]["deploy"]["status"] = "finished"
                manager._save(workflow)

                first = manager.advance(workflow["workflow_id"])
                self.assertEqual("deploy", first["current_stage"])
                self.assertEqual("waiting_service", first["stages"]["deploy"]["deployment_step"])
                second = manager.advance(workflow["workflow_id"])
                self.assertEqual("benchmark", second["current_stage"])
                third = manager.advance(workflow["workflow_id"])
                self.assertEqual("running", third["stages"]["benchmark"]["status"])
                self.assertEqual(2, len(starts))
                self.assertEqual("job-1", third["stages"]["benchmark"]["result"]["benchmark_job_id"])
            finally:
                manager.shutdown()

    def test_stopped_service_requeues_deploy_once(self):
        spec = importlib.util.spec_from_file_location("workflow_manager_restart_test", MANAGER)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        noop = lambda *_args, **_kwargs: {}
        polls = []

        def monitor_deploy(_workflow):
            polls.append(1)
            return {
                "status": "running", "restart_required": True,
                "message": "目标推理服务已停止，准备重新启动后继续工作流",
            }

        deps = module.WorkflowDependencies(
            start_train=noop, monitor_train=noop, find_trained_model=lambda _wf: "",
            start_evaluate=noop, monitor_evaluate=noop, start_publish=noop,
            monitor_publish=noop,
            inference_command=lambda *_args, **_kwargs: self.fail("unexpected direct restart"),
            monitor_deploy=monitor_deploy,
        )
        with tempfile.TemporaryDirectory() as root:
            manager = module.WorkflowManager(
                str(Path(root) / "workflows.sqlite"), deps,
                poll_interval=1, auto_start_worker=False, agent_events=True,
            )
            manager.start_worker = lambda: None
            try:
                workflow = manager.create("user-1", "dataset-1", auto_start=False)
                workflow["current_stage"] = "deploy"
                workflow["context"]["published_model_path"] = "/models/model-a"
                workflow["stages"]["deploy"].update(
                    status="running", deployment_step="waiting_service",
                    service_wait_started_at=time.time() - 1901,
                    service_wait_instance_id="old-instance",
                )
                manager._save(workflow)

                resumed = manager.advance(workflow["workflow_id"])
                self.assertEqual("running", resumed["status"])
                self.assertEqual("awaiting_agent", resumed["stages"]["deploy"]["status"])
                self.assertNotIn("service_wait_started_at", resumed["stages"]["deploy"])
                self.assertNotIn("service_wait_instance_id", resumed["stages"]["deploy"])
                again = manager.advance(workflow["workflow_id"])
                self.assertEqual("awaiting_agent", again["stages"]["deploy"]["status"])
                self.assertEqual(1, len(polls))
                connection = manager._connect()
                try:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM workflow_events WHERE workflow_id=? AND event_type='deploy_model'",
                        (workflow["workflow_id"],),
                    ).fetchone()[0]
                finally:
                    connection.close()
                self.assertEqual(1, count)
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
