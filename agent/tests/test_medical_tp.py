"""Dependency-free tests; temporary inputs are removed and no GPU is required."""
import ast
import importlib.util
import json
import logging
from pathlib import Path
import shlex
import subprocess
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("medical_tp_test", ROOT / "medflow_agent_tools/_medical_tp.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def profile(heads=30, kv=6, **kwargs):
    value = dict(modelPath="/models/test", heads=heads, kvHeads=kv, mla=False,
                 hidden=heads * 128, intermediate=15360, headDim=128,
                 layers=24, weightBytes=0, quantized=False)
    value.update(kwargs)
    return value


class MedicalTPTests(unittest.TestCase):
    def test_single_default_selects_three_instead_of_four(self):
        plan = tp.plan_gpu_count([profile()], 4, explicit=False)
        self.assertEqual((plan["gpuCount"], plan["tensorParallelSizes"]), (3, [3]))

    def test_two_models_select_common_tp(self):
        plan = tp.plan_gpu_count([profile(), profile(heads=32, kv=8)], 8, explicit=False)
        self.assertEqual((plan["gpuCount"], plan["tensorParallelSizes"]), (4, [2, 2]))

    def test_explicit_invalid_count_returns_alternatives(self):
        with self.assertRaisesRegex(tp.MedicalTPError, "建议 gpu-count：1, 2, 3") as error:
            tp.plan_gpu_count([profile()], 4, explicit=True)
        self.assertEqual(error.exception.reason, "medical_tp_incompatible")

    def test_explicit_odd_count_preserves_actual_split(self):
        plan = tp.plan_gpu_count([profile(), profile(heads=32, kv=8)], 5, explicit=True)
        self.assertEqual(plan["tensorParallelSizes"], [3, 2])
        with self.assertRaises(tp.MedicalTPError):
            tp.plan_gpu_count([profile(heads=32, kv=8), profile()], 5, explicit=True)

    def test_kv_head_replication_and_invalid_split(self):
        self.assertEqual(tp.tp_issue(profile(heads=32, kv=2), 8), "")
        self.assertIn("KV 头数", tp.tp_issue(profile(heads=30, kv=4), 3))

    def test_hidden_and_intermediate_dimensions_are_checked(self):
        self.assertIn("hidden", tp.tp_issue(profile(hidden=4096), 3))
        self.assertIn("intermediate", tp.tp_issue(profile(intermediate=4096), 3))

    def test_mla_does_not_apply_ordinary_kv_rule(self):
        self.assertEqual(tp.tp_issue(profile(kv=4, mla=True), 3), "")

    def test_invalid_gpu_counts(self):
        for count in (0, -1, "bad", True, 2.5):
            with self.subTest(count=count), self.assertRaises(tp.MedicalTPError):
                tp.plan_gpu_count([profile()], count, explicit=True)
        with self.assertRaises(tp.MedicalTPError):
            tp.plan_gpu_count([profile(), profile()], 1, explicit=True)

    def test_nested_config_aliases_and_fp32_memory(self):
        config = {"text_config": {"n_head": 30, "n_head_kv": 6, "n_embd": 3840,
                                  "n_layer": 24, "torch_dtype": "float32"}}
        with patch.object(tp, "_read_json", return_value=config), patch.object(tp, "_weight_bytes", return_value=100):
            model = tp.model_profile("trainer", "/models/test")
        self.assertEqual((model["heads"], model["kvHeads"], model["weightBytes"]), (30, 6, 50))

    def test_unknown_heads_and_invalid_nested_config_fail_early(self):
        for config in ({}, {"text_config": "bad"}, {"num_attention_heads": True}):
            with self.subTest(config=config), patch.object(tp, "_read_json", return_value=config):
                with self.assertRaises(tp.MedicalTPError):
                    tp.model_profile("trainer", "/model")

    def test_compare_paths_follow_updated_first_order(self):
        with patch.object(tp, "model_profile", side_effect=lambda _, path: profile(modelPath=path)) as reader:
            plan = tp.plan_medical_resources("compare_between_models_vpn", "trainer",
                {"model_fir": "/origin", "model_sec": "/updated"}, {}, 8, explicit=False)
        self.assertEqual([p["modelPath"] for p in plan["models"]], ["/updated", "/origin"])
        self.assertEqual(reader.call_count, 2)

    def test_checkpoint_reads_base_once_for_both_services(self):
        with patch.object(tp, "_read_json", return_value={"base_model_name_or_path": "/base"}), \
                patch.object(tp, "model_profile", return_value=profile(modelPath="/base")) as reader:
            plan = tp.plan_medical_resources("ckpt_eval", "trainer", {}, {"CKPT_PATH": "/checkpoint"}, 8, explicit=False)
        self.assertEqual(plan["tensorParallelSizes"], [3, 3])
        reader.assert_called_once_with("trainer", "/base")

    def test_checkpoint_rejects_hub_id(self):
        with patch.object(tp, "_read_json", return_value={"base_model_name_or_path": "org/model"}):
            with self.assertRaisesRegex(tp.MedicalTPError, "绝对路径"):
                tp.plan_medical_resources("ckpt_eval", "trainer", {}, {"CKPT_PATH": "/checkpoint"}, 8, explicit=False)

    def test_checkpoint_replay_uses_recorded_base(self):
        with patch.object(tp, "_read_json", side_effect=AssertionError("original adapter must not be read")), \
                patch.object(tp, "model_profile", return_value=profile(modelPath="/versions/base")) as reader:
            plan = tp.plan_medical_resources("ckpt_eval", "trainer", {}, {"CKPT_PATH": "/checkpoint"},
                8, explicit=False, base_model_path="/versions/base")
        reader.assert_called_once_with("trainer", "/versions/base")
        self.assertEqual(plan["gpuCount"], 6)

    def test_json_reads_pass_paths_as_arguments(self):
        path = "/models/name with spaces; touch BAD/config.json"
        with patch.object(tp.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout='{"a": 1}')) as run:
            self.assertEqual(tp._read_json("trainer", path), {"a": 1})
        self.assertEqual(run.call_args.args[0], ["docker", "exec", "trainer", "cat", path])

    def test_bad_json_and_unreadable_config(self):
        for result in (types.SimpleNamespace(returncode=1, stdout=""),
                       types.SimpleNamespace(returncode=0, stdout="[]"),
                       types.SimpleNamespace(returncode=0, stdout="broken")):
            with patch.object(tp.subprocess, "run", return_value=result), self.assertRaises(tp.MedicalTPError):
                tp._read_json("trainer", "/config.json")

    def test_weight_size_from_index_and_stat_fallback(self):
        with patch.object(tp, "_read_json", return_value={"metadata": {"total_size": 123}}):
            self.assertEqual(tp._weight_bytes("trainer", "/model"), 123)
        with patch.object(tp, "_read_json", return_value={}), \
                patch.object(tp.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="456")) as run:
            self.assertEqual(tp._weight_bytes("trainer", "/model with spaces"), 456)
        self.assertEqual(run.call_args.args[0][-1], "/model with spaces")

    def test_allocation_count_duplicates_and_memory(self):
        plan = tp.plan_gpu_count([profile(weightBytes=60 * 1024 ** 3)], 4, explicit=False)
        for ids in ("0,1", "0,0,1"):
            with self.assertRaisesRegex(tp.MedicalTPError, "分配与 TP 计划不一致"):
                tp.validate_allocation(plan, ids, {})
        snapshot = {"gpus": [dict(index=i, memoryTotal=24000, memoryFree=24000) for i in (4, 5, 6)]}
        with self.assertRaisesRegex(tp.MedicalTPError, "显存不足") as error:
            tp.validate_allocation(plan, "4,5,6", snapshot)
        self.assertEqual(error.exception.reason, "medical_gpu_memory_insufficient")
        for gpu in snapshot["gpus"]:
            gpu.update(memoryTotal=80000, memoryFree=78000)
        self.assertEqual(tp.validate_allocation(plan, "4,5,6", snapshot), [])

    def test_unknown_weights_are_reported(self):
        plan = tp.plan_gpu_count([profile()], 4, explicit=False)
        self.assertIn("权重大小未知", tp.validate_allocation(plan, "0,1,2", {})[0])

    def test_zero_free_memory_is_not_treated_as_unknown(self):
        plan = tp.plan_gpu_count([profile(weightBytes=1024)], 4, explicit=False)
        with self.assertRaises(tp.MedicalTPError):
            tp.validate_allocation(plan, "0,1,2", {"gpus": [dict(index=0, memoryTotal=80000, memoryFree=0)]})


class MedicalPreflightIntegrationTests(unittest.TestCase):
    """Exercise the real preflight function without importing optional agent SDKs."""
    def setUp(self):
        path = ROOT / "medflow_agent_tools/runlocal_evaluate.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"run_evaluate_preflight", "_evaluate_preflight_issue"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.resource_calls = []
        self.namespace = dict(Any=object, Dict=dict, List=list, Optional=__import__('typing').Optional,
            subprocess=types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""),
                                            TimeoutExpired=subprocess.TimeoutExpired),
            shlex=shlex, json=json, logger=logging.getLogger("test"), MEDICAL_SCRIPTS=tp.MEDICAL_SCRIPTS,
            PLAN_ENV=tp.PLAN_ENV, MedicalTPError=tp.MedicalTPError,
            GENERIC_EVAL_SCRIPTS=set(),
            _validate_evaluate_container_preflight=lambda _: None,
            _validate_evaluate_gpu_snapshot_preflight=lambda _: None,
            _default_resource_request=lambda *a: (1, 4),
            plan_medical_resources=lambda *a, **k: tp.plan_gpu_count([profile()], a[4], explicit=k["explicit"]),
            _preflight_resource_allocation=lambda *a: self.resource_calls.append(dict(a[3])))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), self.namespace)

    def check(self, options, env):
        return self.namespace["run_evaluate_preflight"]("single_model_evaluation_vpn", "trainer", env,
            {"model_fir": "/model"}, {}, options, {"enabled": True})

    def test_default_plan_reaches_resource_preflight_and_environment(self):
        options, env = {}, {}
        self.assertIsNone(self.check(options, env))
        self.assertEqual(self.resource_calls, [{"gpu-count": 3}])
        self.assertEqual(json.loads(env[tp.PLAN_ENV])["tensorParallelSizes"], [3])

    def test_invalid_explicit_count_does_not_request_resources(self):
        issue = self.check({"gpu-count": 4}, {})
        self.assertEqual(issue["error_reason"], "medical_tp_incompatible")
        self.assertEqual(self.resource_calls, [])

    def test_generic_route_does_not_plan_tp(self):
        self.namespace["plan_medical_resources"] = lambda *a, **k: self.fail("unexpected TP planning")
        self.assertIsNone(self.namespace["run_evaluate_preflight"]("other_script", "trainer", {}, {}, {}, {}, {"enabled": True}))
        self.assertEqual(self.resource_calls, [{}])


class MedicalAllocationIntegrationTests(unittest.TestCase):
    """Run the existing reservation/validation/error-release block verbatim."""
    def setUp(self):
        path = ROOT / "medflow_agent_tools/runlocal_evaluate.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_script_by_name_evaluate")
        block = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == "resource_request"
                     and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                             and call.func.id == "_prepare_resource_allocation" for call in ast.walk(node)))
        wrapper = ast.parse("def check():\n    global final_env_vars\n    reservation_id = None\n").body[0]
        wrapper.body.append(block)
        self.released = []
        self.assigned = "0,1,2"
        plan = tp.plan_gpu_count([profile(weightBytes=60 * 1024 ** 3)], 4, explicit=False)
        env = {tp.PLAN_ENV: json.dumps(plan)}
        def reserve(*args):
            env["MEDFLOW_ASSIGNED_GPUS"] = self.assigned
            return "reservation-1"
        self.namespace = dict(resource_request={"enabled": True}, script_name="single_model_evaluation_vpn",
            script_info={}, args={}, final_env_vars=env, launcher_options={}, run_kwargs={}, json=json,
            MEDICAL_SCRIPTS=tp.MEDICAL_SCRIPTS, PLAN_ENV=tp.PLAN_ENV, MedicalTPError=tp.MedicalTPError,
            logger=logging.getLogger("test"), validate_allocation=tp.validate_allocation,
            _prepare_resource_allocation=reserve,
            _local_gpu_snapshot_for_runtime_request=lambda: {"gpus": [
                dict(index=i, memoryTotal=80000, memoryFree=78000) for i in range(3)]},
            require_assigned_gpu_env=lambda env: env,
            _release_resource_allocation=lambda rid, env: self.released.append(rid),
            _evaluate_error_response=lambda script, message, **kwargs: kwargs)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), self.namespace)

    def test_mismatched_allocation_is_released(self):
        self.assigned = "0,1"
        issue = self.namespace["check"]()
        self.assertEqual(issue["error_reason"], "medical_gpu_allocation_mismatch")
        self.assertEqual(self.released, ["reservation-1"])

    def test_insufficient_memory_is_released(self):
        self.namespace["_local_gpu_snapshot_for_runtime_request"] = lambda: {"gpus": [
            dict(index=i, memoryTotal=24000, memoryFree=24000) for i in range(3)]}
        issue = self.namespace["check"]()
        self.assertEqual(issue["error_reason"], "medical_gpu_memory_insufficient")
        self.assertEqual(self.released, ["reservation-1"])

    def test_success_preserves_whitelist_and_reservation(self):
        self.assertIsNone(self.namespace["check"]())
        self.assertEqual(self.released, [])
        self.assertEqual(self.namespace["run_kwargs"]["env_vars"]["MEDFLOW_ASSIGNED_GPUS"], "0,1,2")


if __name__ == "__main__":
    unittest.main()
