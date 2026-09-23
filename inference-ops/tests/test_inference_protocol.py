import ast
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "agent" / "runtime_agent.py"
INFERENCE_AGENT = REPO / "inference-ops" / "inference_ops" / "inference_agent.py"


def load_functions(path: Path, names: List[str], namespace: Dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    namespace.update({"__builtins__": __builtins__})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class InferenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "re": re,
        }
        load_functions(
            RUNTIME,
            [
                "_inference_response_sources",
                "_inference_api_response_data",
                "_inference_start_protocol_decision",
            ],
            namespace,
        )
        load_functions(
            INFERENCE_AGENT,
            [
                "empty_response_data",
                "merge_benchmark_reports",
                "merge_response_data",
                "compact_response_data",
                "controller_response_data",
                "response_data_for_api",
            ],
            namespace,
        )
        cls.decide = staticmethod(namespace["_inference_start_protocol_decision"])
        cls.normalize_data = staticmethod(namespace["_inference_api_response_data"])
        cls.response_data_for_api = staticmethod(namespace["response_data_for_api"])

    def test_multiline_restart_starting_is_submitted(self):
        result = (
            "推理服务已触发重启，当前状态如下：\n"
            "操作结果：已成功停止旧实例，并启动了新的推理服务实例。\n"
            "当前状态：正在启动中 (Starting)"
        )
        decision = self.decide("service_restart", result, {})
        self.assertEqual("submitted", decision["kind"])
        self.assertEqual("starting", decision["status"])
        self.assertNotEqual("failed", decision["kind"])

    def test_structured_start_submission_wins(self):
        decision = self.decide(
            "service_restart",
            "重启结果暂未生成完整文本",
            {
                "service_start": {
                    "submitted": True,
                    "instance_id": "instance-1",
                    "status": "starting",
                    "reservation_id": "reservation-1",
                    "gpus": "0,1,2,3",
                    "ports": {"vllm": 7111},
                }
            },
        )
        self.assertEqual("submitted", decision["kind"])
        self.assertEqual("structured_service_start_submitted", decision["reason"])

    def test_structured_failure_is_failed(self):
        decision = self.decide(
            "service_restart",
            "重启失败",
            {"service_start": {"submitted": False, "status": "failed"}},
        )
        self.assertEqual("failed", decision["kind"])

    def test_starting_and_running_instances_are_not_failed(self):
        for status in ("starting", "running", "degraded"):
            with self.subTest(status=status):
                decision = self.decide(
                    "service_start",
                    "服务状态已返回",
                    {"service_instances": {"items": [{"status": status}]}},
                )
                self.assertEqual("submitted", decision["kind"])
                self.assertEqual(status, decision["status"])

    def test_old_stop_failure_does_not_override_new_start(self):
        result = (
            "停止旧实例失败：旧实例已不存在。\n"
            "重新启动结果：启动了新的推理服务实例。\n"
            "当前状态：Starting"
        )
        decision = self.decide("service_restart", result, {})
        self.assertEqual("submitted", decision["kind"])

    def test_explicit_start_failure_text_is_failed(self):
        decision = self.decide(
            "service_restart",
            "重启推理服务失败：GPU 资源不足。",
            {},
        )
        self.assertEqual("failed", decision["kind"])

    def test_unknown_start_result_is_not_reported_as_failed(self):
        decision = self.decide("service_restart", "已收到请求。", {})
        self.assertEqual("unknown", decision["kind"])
        self.assertEqual("unknown", decision["status"])

    def test_api_response_data_keeps_structured_start_fields(self):
        data = self.normalize_data(
            {
                "status": "ok",
                "data": {
                    "service_start": {
                        "submitted": True,
                        "instance_id": "instance-1",
                        "gpus": "0,1,2,3",
                        "ports": {"vllm": 7111},
                    }
                },
            }
        )
        self.assertTrue(data["service_start"]["submitted"])
        self.assertEqual("0,1,2,3", data["service_start"]["gpus"])
        self.assertEqual(7111, data["service_start"]["ports"]["vllm"])

    def test_top_level_worker_data_is_normalized(self):
        data = self.normalize_data(
            {
                "status": "ok",
                "service_start": {"submitted": True, "status": "starting"},
            }
        )
        self.assertTrue(data["service_start"]["submitted"])

    def test_api_response_data_falls_back_to_legacy_graph_state(self):
        data = self.response_data_for_api(
            {
                "service_start": {
                    "submitted": True,
                    "instance_id": "instance-legacy",
                }
            }
        )
        self.assertEqual("instance-legacy", data["service_start"]["instance_id"])


if __name__ == "__main__":
    unittest.main()
