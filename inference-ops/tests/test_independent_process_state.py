import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
TOOLS_PATH = REPO / "inference-ops" / "inference_ops" / "tools.py"
AGENT_PATH = REPO / "inference-ops" / "inference_ops" / "inference_agent.py"


def load_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class IndependentProcessStateTests(unittest.TestCase):
    def test_verified_launch_identity_survives_missing_or_changed_run_environment(self):
        dependency = SimpleNamespace(deployment_owned_identity=lambda meta, pid: {'startTicks': '1'} if pid == 101 else None)
        namespace = {'__package__': 'inference_test_package'}
        classify = load_function(TOOLS_PATH, 'process_instance_run_id', namespace)
        meta = {'run_id': 'run-1', 'instance_id': 'instance-1', 'experimentDirectory': '/deployment'}
        with patch.dict(sys.modules, {'inference_test_package.experiment_records': dependency}):
            for environment in ('', 'changed-value'):
                namespace['process_service_run_id'] = lambda pid: environment
                self.assertEqual(classify(meta, 101), 'run-1')
                self.assertEqual(classify(meta, 102), environment)

    def test_shared_port_does_not_authorize_unidentified_peer_process(self):
        namespace = {'Optional': Optional, 'SERVICE_INSTANCE_PORT_NAMES': ('vllm',),
            'process_instance_run_id': lambda meta, pid: {101: 'run-1', 102: '', 103: 'other-run'}[pid],
            'service_instance_recorded_pids': lambda meta: [101], 'pid_is_alive': lambda pid: True}
        port_ownership = load_function(TOOLS_PATH, 'service_instance_port_ownership', namespace)
        process_ownership = load_function(TOOLS_PATH, 'service_instance_process_ownership', namespace)
        meta = {'instance_id': 'instance-1', 'run_id': 'run-1', 'ports': {'vllm': 7111}}
        ports = port_ownership(meta, ({7111: {101, 102, 103}}, set()))
        self.assertEqual(len(ports['owned']), 1)
        self.assertEqual(process_ownership(meta, ports), {'owned': [101], 'unverified': [102], 'reused': [103]})

    def refresh_instance_status(
        self,
        *,
        script_status="starting",
        launcher_running=True,
        vllm_pid=200,
        vllm_running=False,
        vllm_port_running=False,
    ):
        with tempfile.TemporaryDirectory() as root:
            status_file = Path(root) / "status.json"
            status_file.write_text(
                json.dumps(
                    {
                        "status": script_status,
                        "finished_at": None,
                        "error": "startup failed" if script_status == "failed" else None,
                    }
                ),
                encoding="utf-8",
            )
            live_pids = []
            if launcher_running:
                live_pids.append(100)
            if vllm_running:
                live_pids.append(vllm_pid)
            port_ownership = {
                "owned": ([{"name": "vllm"}] if vllm_port_running else []),
                "unverified": [],
                "reused": [],
                "closed": ([] if vllm_port_running else [{"name": "vllm"}]),
            }
            namespace = {
                "os": os,
                "json": json,
                "LEGACY_SERVICE_TASK_REQUIRED_PORTS": ["vllm"],
                "SERVICE_TASK_REQUIRED_PORTS": ["vllm"],
                "service_instance_recorded_pid": (
                    lambda _meta, service: {
                        "start-service": 100,
                        "vllm": vllm_pid,
                    }.get(service, 0)
                ),
                "service_instance_port_ownership": lambda _meta: port_ownership,
                "service_instance_process_ownership": (
                    lambda _meta, _ports: {
                        "owned": live_pids,
                        "unverified": [],
                        "reused": [],
                    }
                ),
                "service_instance_is_legacy": lambda _meta: False,
                "mark_meta_stale_fixed": lambda meta, reason, **extra: meta.update(
                    {"stale_status_fixed": {"reason": reason, **extra}}
                ),
                "finish_meta_if_missing": lambda meta: meta.setdefault(
                    "finished_at", "2026-09-16 14:00:00"
                ),
                "current_time_text": lambda: "2026-09-16 14:00:00",
            }
            refresh = load_function(
                TOOLS_PATH,
                "refresh_service_instance_status",
                namespace,
            )
            return refresh(
                {
                    "instance_id": "instance-1",
                    "status": "starting",
                    "script_pid": 100,
                    "status_file": str(status_file),
                }
            )

    def test_dead_vllm_is_failed_even_while_launcher_is_alive(self):
        status = self.refresh_instance_status(
            launcher_running=True,
            vllm_pid=200,
            vllm_running=False,
            vllm_port_running=False,
        )
        self.assertEqual("failed", status["status"])
        self.assertEqual("starting", status["script_status"])

    def test_explicit_script_failure_is_not_overridden_by_live_launcher(self):
        status = self.refresh_instance_status(
            script_status="failed",
            launcher_running=True,
            vllm_running=False,
            vllm_port_running=False,
        )
        self.assertEqual("failed", status["status"])
        self.assertEqual("startup failed", status["error"])

    def test_launcher_before_vllm_pid_remains_starting(self):
        status = self.refresh_instance_status(
            launcher_running=True,
            vllm_pid=0,
            vllm_running=False,
            vllm_port_running=False,
        )
        self.assertEqual("starting", status["status"])

    def test_live_vllm_and_port_remain_starting_until_api_ready(self):
        status = self.refresh_instance_status(
            launcher_running=True,
            vllm_pid=200,
            vllm_running=True,
            vllm_port_running=True,
        )
        self.assertEqual("starting", status["status"])

    def test_finished_startup_with_live_vllm_and_port_becomes_running(self):
        status = self.refresh_instance_status(
            script_status="finished",
            launcher_running=False,
            vllm_pid=200,
            vllm_running=True,
            vllm_port_running=True,
        )
        self.assertEqual("running", status["status"])

    def test_failed_startup_stays_failed_even_if_port_is_listening(self):
        status = self.refresh_instance_status(
            script_status="failed",
            launcher_running=True,
            vllm_pid=200,
            vllm_running=True,
            vllm_port_running=True,
        )
        self.assertEqual("failed", status["status"])

    def test_failed_instance_does_not_reserve_gpu_memory(self):
        namespace = {
            "Dict": Dict,
            "list_service_instance_ids": lambda: ["failed-instance"],
            "visible_service_instance_meta": lambda _instance_id: {
                "status": "failed",
                "actual_gpus": "0,1,2,3",
            },
            "parse_visible_gpus": lambda value: value.split(","),
        }
        reserve = load_function(
            TOOLS_PATH,
            "reserve_instance_gpu_memory",
            namespace,
        )
        memory = {str(index): (0, 40960) for index in range(4)}
        adjusted, reserved = reserve(memory)
        self.assertEqual(memory, adjusted)
        self.assertEqual([], reserved)

    def test_port_allocation_refreshes_stale_instance_before_leasing(self):
        namespace = {
            "list_service_instance_ids": lambda: ["stale-instance"],
            "visible_service_instance_meta": lambda _instance_id: {
                "status": "failed",
                "ports": {"vllm": 7111, "master": 50121},
            },
            "listening_port_pids": lambda _ports: ({}, set()),
            "service_instance_holds_port_lease": lambda meta, _snapshot: (
                meta.get("status") != "failed"
            ),
        }
        used_ports = load_function(TOOLS_PATH, "used_instance_ports", namespace)
        self.assertEqual(set(), used_ports())

    def test_failed_launcher_without_listener_does_not_hold_port_lease(self):
        namespace = {
            "Optional": Optional,
            "service_instance_port_ownership": lambda _meta, _snapshot: {
                "owned": [],
                "unverified": [],
                "reused": [],
                "closed": [{"name": "vllm", "port": 7111}],
            },
        }
        holds_lease = load_function(
            TOOLS_PATH,
            "service_instance_holds_port_lease",
            namespace,
        )
        self.assertFalse(
            holds_lease(
                {"status": "failed", "script_pid": 100},
                ({}, set()),
            )
        )

    def test_record_inference_pid_writes_only_local_registry(self):
        with tempfile.TemporaryDirectory() as root:
            registry = Path(root) / "inference_pid_registry.jsonl"
            namespace = {
                "os": os,
                "json": json,
                "current_time_text": lambda: "2026-09-16 12:00:00",
                "get_inference_pid_registry_path": lambda: str(registry),
            }
            record = load_function(TOOLS_PATH, "record_inference_pid", namespace)
            record(
                {
                    "instance_id": "instance-1",
                    "run_id": "run-1",
                    "script_pid": 123,
                    "status": "starting",
                    "actual_gpus": "4,5",
                    "resource": {
                        "reservation_id": "reservation-1",
                        "runtime_node_id": "node-a",
                    },
                }
            )

            payload = json.loads(registry.read_text(encoding="utf-8").strip())
            self.assertEqual("reservation-1", payload["reservationId"])
            self.assertEqual("instance-1", payload["instance_id"])
            self.assertEqual("starting", payload["launch_status"])

    def test_readyz_does_not_require_resource_api_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            state_dir = Path(root) / "state"
            state_dir.mkdir()
            agent_config = Path(root) / "agent.yaml"
            service_config = Path(root) / "service.yaml"
            agent_config.write_text("ROLE: both\n", encoding="utf-8")
            service_config.write_text("PROVIDER: vllm\n", encoding="utf-8")
            namespace = {
                "os": os,
                "INFERENCE_OPS_AUTH_MODE": "trusted_network",
                "INFERENCE_OPS_API_TOKEN": "",
                "AGENT_CONFIG_FILE": str(agent_config),
                "AGENT_ROLE": "both",
                "ops_tools": SimpleNamespace(
                    CONFIG_FILE=str(service_config),
                    STATE_DIR=str(state_dir),
                ),
                "JSONResponse": lambda **kwargs: kwargs,
                "experiment_configuration_error": lambda: "",
            }
            readyz = load_function(AGENT_PATH, "readyz", namespace)

            self.assertEqual("ready", readyz()["status"])


if __name__ == "__main__":
    unittest.main()
