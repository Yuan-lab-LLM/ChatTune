import ast
import copy
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TOOLS_PATH = REPO / "inference-ops" / "inference_ops" / "tools.py"


def load_tools_symbols(*names):
    tree = ast.parse(TOOLS_PATH.read_text(encoding="utf-8"), filename=str(TOOLS_PATH))
    wanted = set(names)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {"copy": copy, "check_port": lambda _port: True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TOOLS_PATH), "exec"), namespace)
    return namespace


class SeparationContractTests(unittest.TestCase):
    def test_service_instance_list_text_uses_markdown_blocks(self):
        symbols = load_tools_symbols(
            "DEFAULT_LIST_LIMIT",
            "list_service_instances_text",
        )
        symbols.update(
            {
                "list_service_instances_payload": lambda _scope, _limit: {
                    "scope": "mine",
                    "limit": 20,
                    "returned": 2,
                    "items": [
                        {
                            "instance_id": "20260916_134937_353be1ed",
                            "owner": "self",
                            "status": "starting",
                            "gpus": "0,1,2,3",
                            "model": "Qwen3.6-27B",
                            "ports": {"vllm": 7111},
                            "started_at": "2026-09-16 13:49:39",
                        },
                        {
                            "instance_id": "20260916_100417_3ffb3044",
                            "owner": "self",
                            "status": "failed",
                            "gpus": "4,5,6,7",
                            "model": "Qwen3.6-27B",
                            "ports": {"vllm": 7111},
                            "started_at": "2026-09-16 10:04:19",
                        },
                    ],
                    "summary": {
                        "total": 2,
                        "returned": 2,
                        "hidden": 0,
                        "owners": {"self": 2, "other": 0, "unknown": 0},
                    },
                },
            }
        )

        result = symbols["list_service_instances_text"]("mine", 20)

        self.assertIn("### 推理服务实例", result)
        self.assertIn("- **范围**：当前用户", result)
        self.assertIn("#### 1. `20260916_134937_353be1ed`", result)
        self.assertIn("- **状态**：启动中（`starting`）", result)
        self.assertIn("- **GPU**：`0,1,2,3`", result)
        self.assertIn("#### 2. `20260916_100417_3ffb3044`", result)
        self.assertIn("- **状态**：失败（`failed`）", result)
        self.assertNotIn(" | ", result)

    def test_inference_ops_is_standalone(self):
        self.assertTrue((REPO / "inference-ops" / "inference_ops").is_dir())
        active_files = [
            REPO / "inference-ops" / "scripts" / "start-service.sh",
            REPO / "inference-ops" / "config" / "service.yaml",
            REPO / "inference-ops" / "config" / "service.default.yaml",
            REPO / "inference-ops" / ".env.example",
            REPO / "runtime.sh",
        ]
        forbidden = ("MEDFLOW_ROOT", "../medflow", "src/inference.py", "case2chat.py")
        for path in active_files:
            source = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{marker} found in {path}")

    def test_vllm_only_configuration_and_schema_v2(self):
        for filename in ("service.yaml", "service.default.yaml"):
            source = (REPO / "inference-ops" / "config" / filename).read_text(
                encoding="utf-8"
            )
            self.assertIn("PROVIDER: vllm", source)
            self.assertIn("VLLM_OPENAI_PORT:", source)
            self.assertIn("MASTER_PORT:", source)
            for removed in (
                "INFERENCE_PORT:",
                "UI_PORT:",
                "DATA_ANNOTATION_PORT:",
                "TEST_DIR:",
                "MAX_TOKENS:",
            ):
                self.assertNotIn(removed, source)

        tools_source = (
            REPO / "inference-ops" / "inference_ops" / "tools.py"
        ).read_text(encoding="utf-8")
        self.assertIn("STATE_SCHEMA_VERSION = 2", tools_source)
        self.assertIn('"provider": "vllm"', tools_source)
        self.assertIn('"vllm": allocated_ports["vllm"]', tools_source)
        self.assertIn('"master": allocated_ports["master"]', tools_source)

    def test_legacy_config_and_schema_v1_status_are_read_compatibly(self):
        symbols = load_tools_symbols(
            "STATE_SCHEMA_VERSION",
            "ACTIVE_SERVICE_INSTANCE_PORT_NAMES",
            "LEGACY_SERVICE_INSTANCE_PORT_NAMES",
            "SERVICE_TASK_REQUIRED_PORTS",
            "LEGACY_SERVICE_TASK_REQUIRED_PORTS",
            "normalize_config_paths",
            "service_instance_is_legacy",
            "service_instance_port_services",
        )
        normalized = symbols["normalize_config_paths"](
            {
                "PROVIDER": "medflow",
                "PORTS": {
                    "VLLM_OPENAI_PORT": 7111,
                    "INFERENCE_PORT": 7013,
                    "UI_PORT": 7860,
                    "DATA_ANNOTATION_PORT": 7016,
                },
                "ENV": {"TEST_DIR": "/legacy/tests"},
                "RUNTIME": {"MAX_TOKENS": 8192},
            }
        )
        self.assertEqual("vllm", normalized["PROVIDER"])
        self.assertEqual({"VLLM_OPENAI_PORT": 7111}, normalized["PORTS"])
        self.assertNotIn("TEST_DIR", normalized["ENV"])
        self.assertNotIn("MAX_TOKENS", normalized["RUNTIME"])

        legacy = {
            "schema_version": 1,
            "provider": "medflow",
            "ports": {
                "vllm": 7111,
                "inference": 7013,
                "ui": 7860,
                "case2chat": 7016,
                "master": 50121,
            },
        }
        current = {
            "schema_version": 2,
            "provider": "vllm",
            "ports": {**legacy["ports"]},
        }
        self.assertTrue(symbols["service_instance_is_legacy"](legacy))
        self.assertFalse(symbols["service_instance_is_legacy"](current))
        self.assertEqual(
            ["vllm", "inference", "ui", "case2chat", "master"],
            [item["name"] for item in symbols["service_instance_port_services"](legacy)],
        )
        self.assertEqual(
            ["vllm", "master"],
            [item["name"] for item in symbols["service_instance_port_services"](current)],
        )
        self.assertEqual(
            ["vllm"],
            [
                item["name"]
                for item in symbols["service_instance_port_services"](
                    current,
                    include_master=False,
                )
            ],
        )

    def test_service_status_only_exposes_user_facing_service_ports(self):
        symbols = load_tools_symbols("service_status_data")
        symbols.update(
            {
                "show_config": lambda: {
                    "PORTS": {"VLLM_OPENAI_PORT": 7111},
                    "ENV": {"MASTER_PORT": 50121},
                },
                "check_port": lambda _port: False,
                "current_request_user_id": lambda: "user-1",
                "list_service_instance_ids": lambda: [],
                "current_time_text": lambda: "2026-09-15 16:50:29",
            }
        )

        status = symbols["service_status_data"]()

        self.assertEqual(
            ["VLLM_OPENAI_PORT"],
            [item["name"] for item in status["services"]],
        )
        self.assertEqual(
            ["VLLM_OPENAI_PORT"],
            [item["name"] for item in status["template_services"]],
        )
        self.assertNotIn("master", status["service_instances"]["ports"])
        self.assertNotIn("master", status["text"].lower())

    def test_existing_http_contract_and_health_endpoints_are_present(self):
        source = (
            REPO / "inference-ops" / "inference_ops" / "inference_agent.py"
        ).read_text(encoding="utf-8")
        for route in (
            '"/inference_agent"',
            '"/worker/tool"',
            '"/admin/services"',
            '"/healthz"',
            '"/readyz"',
        ):
            self.assertIn(route, source)
        self.assertIn("INFERENCE_OPS_AUTH_MODE", source)
        self.assertIn("INFERENCE_OPS_API_TOKEN", source)

    def test_startup_never_installs_dependencies_or_kills_by_port(self):
        source = (REPO / "inference-ops" / "scripts" / "start-service.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("npm install", source)
        self.assertNotIn("kill -9 $(lsof", source)
        self.assertIn("vllm serve", source)
        self.assertIn("/v1/models", source)
        self.assertIn('"schema_version": 2', source)
        self.assertNotIn("MEDFLOW_ROOT", source)
        self.assertIn("PID_DIR", source)

    def test_functional_test_interfaces_are_removed(self):
        agent_source = (
            REPO / "inference-ops" / "inference_ops" / "inference_agent.py"
        ).read_text(encoding="utf-8")
        tools_source = (
            REPO / "inference-ops" / "inference_ops" / "tools.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"/admin/tests', agent_source)
        for tool_name in ("service_test_", "node_service_test_", "admin_test_"):
            self.assertNotIn(tool_name, agent_source)
            self.assertNotIn(tool_name, tools_source)

    def test_callers_apply_the_configured_auth_mode(self):
        runtime_agent = (REPO / "agent" / "runtime_agent.py").read_text(encoding="utf-8")
        resource_api = (REPO / "agent" / "resource_api.py").read_text(encoding="utf-8")
        studio_router = (
            REPO / "studio" / "packages" / "server" / "src" / "trpc" / "router.ts"
        ).read_text(encoding="utf-8")
        self.assertIn("_inference_ops_headers()", runtime_agent)
        self.assertIn("INFERENCE_OPS_AUTH_MODE", runtime_agent)
        self.assertIn("_inference_ops_request_headers()", resource_api)
        self.assertIn("INFERENCE_OPS_AUTH_MODE", resource_api)
        self.assertIn("getInferenceOpsHeaders()", studio_router)

    def test_runtime_defaults_to_trusted_network_auth(self):
        runtime = (REPO / "runtime.sh").read_text(encoding="utf-8")
        inference_agent = (
            REPO / "inference-ops" / "inference_ops" / "inference_agent.py"
        ).read_text(encoding="utf-8")
        tools = (
            REPO / "inference-ops" / "inference_ops" / "tools.py"
        ).read_text(encoding="utf-8")
        resource_api = (REPO / "agent" / "resource_api.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("INFERENCE_OPS_AUTH_MODE=trusted_network", runtime)
        self.assertIn('case "$INFERENCE_OPS_AUTH_MODE" in', runtime)
        self.assertNotIn("INFERENCE_OPS_API_TOKEN=$inference_token", runtime)
        for source in (runtime, inference_agent, tools):
            self.assertNotIn("INFERENCE_OPS_RESOURCE_API_URL", source)
            self.assertNotIn("INFERENCE_OPS_RESOURCE_API_TOKEN", source)
        self.assertNotIn("report_inference_process", tools)
        self.assertNotIn("/internal/resources/inference-processes/", tools)
        self.assertNotIn('"/inference-processes/{instance_id}"', resource_api)


if __name__ == "__main__":
    unittest.main()
