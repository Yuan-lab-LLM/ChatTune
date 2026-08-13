import importlib
import sys
import types
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2] / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

if "agentscope" not in sys.modules:
    agentscope = types.ModuleType("agentscope")
    tool = types.ModuleType("agentscope.tool")
    message = types.ModuleType("agentscope.message")

    class ToolResponse:
        def __init__(self, content=None, metadata=None):
            self.content = content or []
            self.metadata = metadata or {}

    class TextBlock:
        def __init__(self, type="text", text=""):
            self.type = type
            self.text = text

    tool.ToolResponse = ToolResponse
    message.TextBlock = TextBlock
    sys.modules["agentscope"] = agentscope
    sys.modules["agentscope.tool"] = tool
    sys.modules["agentscope.message"] = message

pkg = types.ModuleType("medflow_agent_tools")
pkg.__path__ = [str(AGENT_ROOT / "medflow_agent_tools")]
sys.modules.setdefault("medflow_agent_tools", pkg)

runlocal_data = importlib.import_module("medflow_agent_tools.runlocal_data")
runlocal_train = importlib.import_module("medflow_agent_tools.runlocal_train")
is_supported_template = importlib.import_module("medflow_agent_tools._template_policy").is_supported_template


class DataPreprocessingFormatTests(unittest.TestCase):
    def test_detects_general_formats(self):
        cases = [
            ({"messages": [{"role": "user", "content": "hi"}]}, "openai"),
            ({"conversations": [{"from": "human", "value": "hi"}]}, "sharegpt"),
            ({"instruction": "i", "input": "x", "output": "y"}, "sft"),
            ({"instruction": "i", "input": "x", "chosen": "a", "rejected": "b"}, "dpo"),
            ({"text": "plain"}, "text"),
        ]
        for item, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(runlocal_data.detect_data_preprocessing_item_format(item), expected)

    def test_general_format_requires_only_data_type_until_data_type_is_known(self):
        script_info = {"required_cli_params": ["data_type", "strategy"]}
        self.assertEqual(
            runlocal_data.effective_data_preprocessing_required_params(
                "data_preprocessing",
                script_info,
                {"data_type": "", "strategy": ""},
                "openai",
            ),
            ["data_type"],
        )
        self.assertEqual(
            runlocal_data.effective_data_preprocessing_required_params(
                "data_preprocessing",
                script_info,
                {"data_type": "sft", "strategy": ""},
                "sharegpt",
            ),
            [],
        )
        self.assertEqual(
            runlocal_data.effective_data_preprocessing_required_params(
                "data_preprocessing",
                script_info,
                {"data_type": "dpo", "strategy": ""},
                "text",
            ),
            [],
        )

    def test_raw_keeps_legacy_required_params(self):
        script_info = {"required_cli_params": ["data_type", "strategy"]}
        self.assertEqual(
            runlocal_data.effective_data_preprocessing_required_params(
                "data_preprocessing",
                script_info,
                {"data_type": "", "strategy": ""},
                "raw",
            ),
            ["data_type", "strategy"],
        )

    def test_unknown_format_does_not_request_strategy(self):
        script_info = {"required_cli_params": ["data_type", "strategy"]}
        self.assertEqual(
            runlocal_data.effective_data_preprocessing_required_params(
                "data_preprocessing",
                script_info,
                {"data_type": "", "strategy": ""},
                "unknown",
            ),
            [],
        )

    def test_invalid_data_type_is_rejected(self):
        invalid = runlocal_data.validate_cli_param_values(
            {"required_cli_params": []},
            {"data_type": "ppo"},
            [],
        )
        self.assertEqual(invalid, ["data_type='ppo'"])


class TrainingAndTemplatePolicyTests(unittest.TestCase):
    def test_batch_train_accepts_model_path_env(self):
        manager = runlocal_train.ScriptManager()
        for script_name in ("batch_train_lora", "batch_train_full"):
            with self.subTest(script_name=script_name):
                script_info = manager.scripts[script_name]
                self.assertIn("MODEL_PATH", script_info["supported_params"])
                self.assertEqual(manager.map_chinese_to_env_var("模型路径"), "MODEL_PATH")
                self.assertEqual(manager.map_chinese_to_env_var("model_path"), "MODEL_PATH")

    def test_batch_train_model_path_is_template_hint_source(self):
        self.assertEqual(
            runlocal_train._training_model_hint(
                "batch_train_lora",
                {"MODEL_PATH": "/home/workspace/models/llama3"},
                {},
            ),
            "/home/workspace/models/llama3",
        )

    def test_new_templates_are_supported(self):
        for template in (
            "ernie_nothink",
            "falcon_h1",
            "gemma2",
            "glm4_moe",
            "gpt_oss",
            "qwen3_nothink",
            "qwen3_5_nothink",
            "seed_coder",
        ):
            with self.subTest(template=template):
                self.assertTrue(is_supported_template(template))


class DataPreprocessingRuntimeFlowTests(unittest.TestCase):
    def setUp(self):
        self._orig_latest = runlocal_data.resolve_latest_dataset_subdir
        self._orig_detect = runlocal_data.detect_data_preprocessing_dir_format_in_container
        self._orig_validate = runlocal_data.validate_docker_dir_params
        self._orig_run_background = runlocal_data.run_docker_in_background
        self._orig_run_local = runlocal_data.run_local_script
        self.captured_background_kwargs = None

    def tearDown(self):
        runlocal_data.resolve_latest_dataset_subdir = self._orig_latest
        runlocal_data.detect_data_preprocessing_dir_format_in_container = self._orig_detect
        runlocal_data.validate_docker_dir_params = self._orig_validate
        runlocal_data.run_docker_in_background = self._orig_run_background
        runlocal_data.run_local_script = self._orig_run_local

    def _stub_flow(self, latest=None, detected="openai"):
        runlocal_data.resolve_latest_dataset_subdir = lambda container, root: latest
        runlocal_data.detect_data_preprocessing_dir_format_in_container = lambda container, path: detected
        runlocal_data.validate_docker_dir_params = lambda **kwargs: None
        def fake_run_docker_in_background(**kwargs):
            self.captured_background_kwargs = kwargs
            return runlocal_data.ToolResponse(
                content=[runlocal_data.TextBlock(type='text', text='started')],
                metadata={'success': True, 'protocol_hint': {'scriptArgs': kwargs.get('script_args', {})}},
            )
        runlocal_data.run_docker_in_background = fake_run_docker_in_background
        runlocal_data.run_local_script = fake_run_docker_in_background

    def _protocol_for(self, additional_args=None):
        response = runlocal_data.run_script_by_name_data(
            "data_preprocessing",
            additional_args=additional_args or {},
            env_vars={"container": "qingnang_train"},
        )
        return response.metadata["protocol_hint"]

    def test_explicit_openai_folder_requires_only_data_type(self):
        self._stub_flow(detected="openai")
        protocol = self._protocol_for({"input_folder": "/home/workspace/dataset/openai"})
        self.assertEqual(protocol["requiredParams"], ["data_type"])
        self.assertEqual(protocol["missingParams"], ["data_type"])
        self.assertEqual(protocol["detectedFormat"], "openai")
        self.assertEqual(protocol["inputFolder"], "/home/workspace/dataset/openai")

    def test_default_latest_openai_requires_only_data_type(self):
        self._stub_flow(latest="/home/workspace/dataset/openai", detected="openai")
        protocol = self._protocol_for()
        self.assertEqual(protocol["requiredParams"], ["data_type"])
        self.assertEqual(protocol["detectedFormat"], "openai")
        self.assertEqual(protocol["selectedInputFolder"], "/home/workspace/dataset/openai")
        self.assertEqual(protocol["sourceInputFolder"], "default_latest")

    def test_default_latest_raw_requires_data_type_and_strategy(self):
        self._stub_flow(latest="/home/workspace/dataset/raw", detected="raw")
        response = runlocal_data.run_script_by_name_data(
            "data_preprocessing",
            env_vars={"container": "qingnang_train"},
        )
        protocol = response.metadata["protocol_hint"]
        message = response.content[0].text
        self.assertEqual(protocol["requiredParams"], ["data_type", "strategy"])
        self.assertEqual(protocol["missingParams"], ["data_type", "strategy"])
        self.assertEqual(protocol["detectedFormat"], "raw")
        self.assertEqual(protocol["sourceInputFolder"], "default_latest")
        self.assertNotIn("错误！", message)
        self.assertIn("还需要补充以下参数", message)

    def test_default_latest_unknown_asks_for_input_folder_not_strategy(self):
        self._stub_flow(latest="/home/workspace/dataset/unknown", detected="unknown")
        protocol = self._protocol_for()
        self.assertEqual(protocol["errorReason"], "unknown_data_format")
        self.assertEqual(protocol["requiredParams"], ["input_folder"])
        self.assertNotIn("strategy", protocol.get("requiredParams", []))
        self.assertEqual(protocol["detectedFormat"], "unknown")
        self.assertEqual(protocol["sourceInputFolder"], "default_latest")

    def test_explicit_unknown_asks_for_input_folder_not_strategy(self):
        self._stub_flow(detected="unknown")
        protocol = self._protocol_for({"input_folder": "/home/workspace/dataset/unknown"})
        self.assertEqual(protocol["errorReason"], "unknown_data_format")
        self.assertEqual(protocol["requiredParams"], ["input_folder"])
        self.assertNotIn("strategy", protocol.get("requiredParams", []))
        self.assertEqual(protocol["detectedFormat"], "unknown")
        self.assertEqual(protocol["sourceInputFolder"], "explicit")


    def _run_started_args(self, additional_args, detected="openai"):
        self._stub_flow(detected=detected)
        response = runlocal_data.run_script_by_name_data(
            "data_preprocessing",
            additional_args=additional_args,
            env_vars={"container": "qingnang_train"},
            background=True,
        )
        self.assertTrue(response.metadata.get("success"), response.metadata)
        self.assertIsNotNone(self.captured_background_kwargs)
        return self.captured_background_kwargs["script_args"]

    def test_general_formats_do_not_pass_empty_strategy(self):
        for detected in ("openai", "sharegpt", "text", "sft", "dpo"):
            with self.subTest(detected=detected):
                script_args = self._run_started_args(
                    {
                        "input_folder": f"/home/workspace/dataset/{detected}",
                        "data_type": "dpo",
                    },
                    detected=detected,
                )
                self.assertEqual(script_args["data_type"], "dpo")
                self.assertNotIn("strategy", script_args)

    def test_general_format_preserves_non_empty_strategy(self):
        script_args = self._run_started_args(
            {
                "input_folder": "/home/workspace/dataset/openai",
                "data_type": "dpo",
                "strategy": "diagnosis",
            },
            detected="openai",
        )
        self.assertEqual(script_args["strategy"], "diagnosis")

    def test_raw_missing_strategy_still_requests_strategy(self):
        self._stub_flow(detected="raw")
        response = runlocal_data.run_script_by_name_data(
            "data_preprocessing",
            additional_args={
                "input_folder": "/home/workspace/dataset/raw",
                "data_type": "dpo",
            },
            env_vars={"container": "qingnang_train"},
            background=True,
        )
        protocol = response.metadata["protocol_hint"]
        message = response.content[0].text
        self.assertEqual(protocol["requiredParams"], ["data_type", "strategy"])
        self.assertEqual(protocol["missingParams"], ["strategy"])
        self.assertNotIn("错误！", message)
        self.assertIsNone(self.captured_background_kwargs)

    def test_raw_with_strategy_passes_strategy(self):
        script_args = self._run_started_args(
            {
                "input_folder": "/home/workspace/dataset/raw",
                "data_type": "dpo",
                "strategy": "inspection",
            },
            detected="raw",
        )
        self.assertEqual(script_args["strategy"], "inspection")

class RuntimePreprocessRuleTests(unittest.TestCase):
    def test_runtime_defers_data_preprocessing_required_params_to_tool(self):
        source = (AGENT_ROOT / "runtime_agent.py").read_text(encoding="utf-8")
        self.assertIn('if "数据预处理" in content or "data_preprocessing" in lower:', source)
        self.assertIn("Let the tool inspect input_folder or the latest default dataset first.", source)
        self.assertIn("通用格式由工具只追问 data_type，医疗 raw 才追问 data_type 和 strategy", source)
        self.assertNotIn("让它报 data_type/strategy", source)

    def test_config_tells_dataprocessor_to_detect_format_before_asking(self):
        config = (AGENT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("必须先调用数据预处理工具，让工具检测输入目录格式后再决定缺参", config)
        self.assertIn("通用格式只问 `data_type`", config)
        self.assertIn("raw 才问 `data_type` 和 `strategy`", config)
        self.assertNotIn("只有在用户已经明确提供 `data_type` 和 `strategy` 后，才能开始执行数据预处理", config)
if __name__ == "__main__":
    unittest.main()
