"""GPU planning regressions; stdlib only, no CUDA or service dependencies."""
import ast
import copy
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "inference-ops/inference_ops/tools.py"
RUNTIME = REPO / "agent/runtime_agent.py"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class GpuAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "Qwen2.5-0.5B-Instruct"
        self.model.mkdir()
        self.arch = {"num_attention_heads": 14, "num_key_value_heads": 2,
                     "hidden_size": 896, "intermediate_size": 4864,
                     "num_hidden_layers": 24, "vocab_size": 151936,
                     "max_position_embeddings": 32768, "torch_dtype": "bfloat16"}
        (self.model / "config.json").write_text(json.dumps(self.arch), encoding="utf-8")
        self.cfg = {"ENV": {"MODEL_PATH": self.temp.name, "MODEL_NAME": self.model.name,
                            "CUDA_VISIBLE_DEVICES": "4,5,6,7"},
                    "RUNTIME": {"TENSOR_PARALLEL_SIZE": 4, "GPU_MEMORY_UTILIZATION": 0.8}}
        self.ns = {"Optional": Optional, "Dict": Dict, "os": os, "json": json,
                   "ast": ast, "copy": copy, "resource_pool_managed": lambda: True,
                   "resource_context_gpu_ids": lambda: ["0", "1", "2", "3"],
                   "show_config": lambda: copy.deepcopy(self.cfg),
                   "get_gpu_memory_map": lambda: ({str(i): (0, 40960) for i in range(4)}, None),
                   "get_gpu_utilization_map": lambda: ({str(i): 0 for i in range(4)}, None),
                   "reserve_instance_gpu_memory": lambda memory: (memory, []),
                   "managed_resource_error": lambda **kwargs: ""}
        names = {
            "parse_float", "clamp_int", "load_model_config", "get_text_model_config",
            "model_architecture_notes", "normalize_precision", "detect_quantization",
            "bytes_per_param_for_model", "bytes_per_kv_cache", "model_config_precision",
            "estimate_params_from_config", "estimate_params_from_safetensors_index",
            "get_model_max_len", "estimate_kv_cache_mib", "get_model_memory_profile",
            "unknown_model_size_analysis", "memory_margin_mib", "estimate_required_memory_mib",
            "generate_tp_candidates", "model_tensor_parallel_issue", "parse_visible_gpus",
            "describe_model_profile", "gpu_eval_status_text", "format_gpu_budget_table",
            "gpu_vllm_budget_mib", "gpu_vllm_planned_limit_mib", "evaluate_gpu_selection",
            "missing_model_name_analysis", "recommend_gpu_for_config", "recommend_gpu",
            "apply_auto_gpu_allocation", "apply_requested_gpu_allocation", "gpu_recommend_allocation",
        }
        load_functions(TOOLS, names, self.ns)

    def test_small_model_uses_one_assigned_gpu_despite_stale_draft_tp4(self):
        original = copy.deepcopy(self.cfg)
        runtime = copy.deepcopy(self.cfg)
        ok, error, allocation = self.ns["apply_requested_gpu_allocation"](runtime)
        self.assertTrue(ok, error)
        self.assertEqual(runtime["RUNTIME"]["TENSOR_PARALLEL_SIZE"], 1)
        self.assertEqual(runtime["ENV"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(allocation["mode"], "auto")
        self.assertEqual(self.cfg, original)

    def test_manual_incompatible_tp_fails_before_launch_with_reason(self):
        ok, error, _ = self.ns["apply_requested_gpu_allocation"](self.cfg, "0,1,2,3", 4)
        self.assertFalse(ok)
        self.assertIn("14 不能被 TP=4 整除", error)
        self.assertEqual(self.cfg["ENV"]["CUDA_VISIBLE_DEVICES"], "4,5,6,7")

    def test_manual_fallback_uses_compatible_allocation(self):
        ok, error, result = self.ns["apply_requested_gpu_allocation"](self.cfg, "0,1,2,3", 4, True)
        self.assertTrue(ok, error)
        self.assertEqual(result["recommended_tp"], 1)

    def test_manual_compatible_selection_is_preserved(self):
        ok, error, result = self.ns["apply_requested_gpu_allocation"](self.cfg, "2,3", 2)
        self.assertTrue(ok, error)
        self.assertEqual(result["mode"], "manual")
        self.assertEqual(result["recommended_gpus"], "2,3")

    def test_manual_selection_cannot_escape_reservation(self):
        ok, error, _ = self.ns["apply_requested_gpu_allocation"](self.cfg, "4", 1, True)
        self.assertFalse(ok)
        self.assertIn("超出资源池边界", error)

    def test_auto_ignores_busy_and_unassigned_gpus(self):
        self.ns["resource_context_gpu_ids"] = lambda: ["1", "3"]
        self.ns["get_gpu_utilization_map"] = lambda: ({"0": 0, "1": 90, "2": 0, "3": 0}, None)
        result = self.ns["recommend_gpu_for_config"](self.cfg)
        self.assertTrue(result["ok"])
        self.assertEqual(result["recommended_gpus"], "3")

    def test_larger_model_selects_smallest_compatible_tp_that_fits(self):
        self.cfg["ENV"]["MODEL_PARAM_B"] = 20
        self.cfg["RUNTIME"]["MAX_MODEL_LEN"] = 4096
        result = self.ns["recommend_gpu_for_config"](self.cfg)
        self.assertTrue(result["ok"], result["analysis"])
        self.assertEqual(result["recommended_tp"], 2)

    def test_no_compatible_tp_with_enough_memory_is_failure(self):
        self.cfg["ENV"]["MODEL_PARAM_B"] = 40
        self.cfg["RUNTIME"]["MAX_MODEL_LEN"] = 4096
        result = self.ns["recommend_gpu_for_config"](self.cfg)
        self.assertFalse(result["ok"])
        self.assertFalse(result["current_ok"])
        self.assertIn("排除 TP=4", result["analysis"])

    def test_non_power_of_two_tp_is_considered(self):
        self.arch.update(num_attention_heads=12, num_key_value_heads=3,
                         hidden_size=1536, intermediate_size=6144)
        (self.model / "config.json").write_text(json.dumps({"text_config": self.arch}), encoding="utf-8")
        self.cfg["ENV"]["MODEL_PARAM_B"] = 30
        self.cfg["RUNTIME"]["MAX_MODEL_LEN"] = 4096
        result = self.ns["recommend_gpu_for_config"](self.cfg)
        self.assertTrue(result["ok"], result["analysis"])
        self.assertEqual(result["recommended_tp"], 3)

    def test_kv_head_partitioning_replication_and_mla(self):
        issue = self.ns["model_tensor_parallel_issue"]
        self.assertEqual(issue({"model_cfg": {"num_attention_heads": 16, "num_key_value_heads": 2}}, 4), "")
        self.assertIn("KV 头数", issue({"model_cfg": {"num_attention_heads": 12, "num_key_value_heads": 3}}, 4))
        self.assertEqual(issue({"model_cfg": {"num_attention_heads": 12, "num_key_value_heads": 3, "kv_lora_rank": 512}}, 4), "")
        self.assertIn("intermediate_size", issue({"model_cfg": {"num_attention_heads": 12, "intermediate_size": 10}}, 3))

    def test_missing_heads_does_not_claim_multi_gpu_is_valid(self):
        issue = self.ns["model_tensor_parallel_issue"]
        self.assertEqual(issue({"model_cfg": {}}, 1), "")
        self.assertIn("缺少有效注意力头数", issue({"model_cfg": {}}, 2))

    def test_reservation_plan_is_read_only_and_launch_stays_bounded(self):
        self.ns["resource_context_gpu_ids"] = lambda: []
        before = copy.deepcopy(self.cfg)
        plan = self.ns["gpu_recommend_allocation"](for_reservation=True)
        self.assertTrue(plan["ok"], plan["analysis"])
        self.assertEqual(plan["recommended_tp"], 1)
        self.assertEqual(self.cfg, before)
        ok, _, _ = self.ns["apply_auto_gpu_allocation"](self.cfg)
        self.assertFalse(ok)


class InferenceReservationCountTests(unittest.TestCase):
    def setUp(self):
        self.post = Mock()
        self.ns = {"Any": Any, "os": os, "json": json, "re": re,
                   "requests": SimpleNamespace(post=self.post),
                   "_inference_worker_tool_url": lambda: "http://worker/worker/tool",
                   "_inference_ops_headers": lambda: {"Authorization": "Bearer test"}}
        load_functions(RUNTIME, {"_positive_inference_gpu_count", "_inference_gpus_per_node"}, self.ns)
        self.environment = patch.dict(os.environ, {"MEDFLOW_INFERENCE_GPUS_PER_NODE": "", "AGENT3_INFERENCE_GPUS_PER_NODE": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def response(self, result, status="ok"):
        self.post.return_value = SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"status": status, "result": result})

    def test_reservation_uses_worker_model_plan_instead_of_draft(self):
        self.response(json.dumps({"ok": True, "recommended_tp": 1}))
        count = self.ns["_inference_gpus_per_node"](owner_user_id="owner", owner_aliases=["alias"], user_role="admin")
        self.assertEqual(count, 1)
        payload = self.post.call_args.kwargs["json"]
        self.assertEqual(payload["tool"], "gpu_recommend_allocation")
        self.assertEqual(payload["args"], {"for_reservation": True})
        self.assertEqual(payload["user_id"], "owner")
        self.assertEqual(payload["user_aliases"], ["alias"])

    def test_failed_or_invalid_plan_does_not_fall_back_to_draft(self):
        for result in ({"ok": False, "analysis": "显存不足"}, {"ok": True, "recommended_tp": 0},
                       {"ok": True, "recommended_tp": True}, "invalid json"):
            with self.subTest(result=result):
                self.response(result)
                with self.assertRaises(RuntimeError):
                    self.ns["_inference_gpus_per_node"](owner_user_id="owner")

    def test_operator_gpu_count_override_is_preserved(self):
        with patch.dict(os.environ, {"MEDFLOW_INFERENCE_GPUS_PER_NODE": "2"}):
            self.assertEqual(self.ns["_inference_gpus_per_node"](owner_user_id="owner"), 2)
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
