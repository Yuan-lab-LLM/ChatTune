# -*- coding: utf-8 -*-
"""
调用本地脚本的工具函数，支持通过名称、别名查找并运行脚本。
相对于另一版本，补充修改参数的能力
加上改docker
加上推理
加上监控
"""

import subprocess
import json
import os
import time
import logging
import posixpath
import re
import shlex
import sys
import threading
from typing import Dict, Any, List, Optional,Tuple
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from ._config_defaults import get_default_docker_container

logger = logging.getLogger(__name__)
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

DEFAULT_DOCKER_CONTAINER = get_default_docker_container()
try:
    from utils.config import get_current_config

    _AGENT_CONFIG = get_current_config()
except Exception:
    _AGENT_CONFIG = None
DEFAULT_MODEL_NAME = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "name", None)
    or os.getenv("AGENT3_MODEL_NAME")
    or os.getenv("MODEL_NAME", "")
)
DEFAULT_MODEL_API_KEY = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "api_key", None)
    or os.getenv("AGENT3_MODEL_API_KEY")
    or os.getenv("MODEL_API_KEY", "")
)
DEFAULT_MODEL_BASE_URL = (
    getattr(getattr(_AGENT_CONFIG, "model", None), "base_url", None)
    or os.getenv("AGENT3_MODEL_BASE_URL")
    or os.getenv("MODEL_BASE_URL", "")
)

_RUNTIME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime")
)
_BACKGROUND_TASK_REGISTRY = os.path.join(_RUNTIME_DIR, "background_task_registry.jsonl")


def _data_protocol_hint(
    protocol_type: str,
    script_name: str,
    message: str = "",
    **fields: Any,
) -> Dict[str, Any]:
    script_name = str(script_name or "").strip()
    hint = {
        "type": protocol_type,
        "agent": "dataprocessor",
        "message": message,
        "script": script_name,
        "scriptName": script_name,
    }
    if protocol_type == "need_input":
        required_params = fields.get("requiredParams") or []
        fields.setdefault("status", "needs_input")
        fields.setdefault("action", "collect_params")
        fields.setdefault("requiredParams", required_params)
        if required_params:
            fields.setdefault("missingParams", required_params)
        if script_name == "score_based_filtering" or required_params == ["input_folder"]:
            fields.setdefault("kind", "input_folder")
        elif script_name == "data_preprocessing" or set(required_params) & {"data_type", "strategy"}:
            fields.setdefault("kind", "data_preprocess_params")
        else:
            fields.setdefault("kind", "data_params")
    elif protocol_type == "job_started":
        fields.setdefault("status", "started")
        fields.setdefault("action", "start")
        fields.setdefault("background", True)
        fields.setdefault("jobType", "data_filter" if script_name == "score_based_filtering" else "data_preprocess")
        fields.setdefault("script", script_name)
    elif protocol_type in {"job_failed", "error"}:
        fields.setdefault("status", "failed")
        fields.setdefault("action", "start")
        fields.setdefault("jobType", "data_filter" if script_name == "score_based_filtering" else "data_preprocess")
    hint.update({key: value for key, value in fields.items() if value is not None})
    return hint


def _data_error_response(
    script_name: str,
    message: str,
    *,
    error_reason: str,
    required_params: Optional[List[str]] = None,
    missing_params: Optional[List[str]] = None,
    recoverable: bool = False,
    **fields: Any,
) -> ToolResponse:
    protocol_type = "need_input" if required_params else "job_failed"
    return ToolResponse(content=[TextBlock(type="text", text=message)],
        metadata={
            "success": False,
            "protocol_hint": _data_protocol_hint(
                protocol_type,
                script_name,
                message,
                requiredParams=required_params or [],
                missingParams=(missing_params if missing_params is not None else required_params or []),
                errorReason=error_reason,
                errorRecoverable=recoverable,
                **fields,
            ),
        },
    )


def _record_background_task(
    task_type: str,
    container: Optional[str],
    script_name: Optional[str],
    script_path: str,
    command: List[str],
    script_args: Optional[Dict[str, Any]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    status: str = "started",
) -> None:
    if not container:
        return

    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    record = {
        "task_type": task_type,
        "container": container,
        "script_name": script_name or os.path.basename(script_path),
        "script_path": script_path,
        "command": " ".join(command),
        "script_args": script_args or {},
        "env_vars": env_vars or {},
        "status": status,
        "started_at": time.time(),
        "started_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_BACKGROUND_TASK_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ScriptManager:
    def __init__(self):
        # 参数映射表：中文 -> 环境变量名
        self.param_mapping = {
            "批量大小": "MBS",
            "批次大小": "MBS",
            "mbs":"MBS",
            "MBS":"MBS",
            "梯度累积": "ACC",
            "累积步数": "ACC",
            "acc":"ACC",
            "ACC":"ACC",
            "学习率": "LR",
            "学习速率": "LR",
            "lr":"LR",
            "LR":"LR",
            "模型": "TEM",
            "基础模型": "TEM",
            "tem":"TEM",
            "tem":"TEM",
            "恢复训练": "RESUME",
            "继续训练": "RESUME",
            "检查点": "RESUME",

            "设备": "LOCALHOST_ID",
            "gpu": "LOCALHOST_ID",
            "显卡": "LOCALHOST_ID",

            "容器": "container",  # 添加容器参数映射
            "docker": "container",
            "ckpt":"CKPT_PATH",
            "checkpoint":"CKPT_PATH",
            "ckpt位置":"CKPT_PATH",
            "checkpoint位置":"CKPT_PATH",
            "第一个模型": "model_fir",
            "模型一": "model_fir",
            "第一个模型路径": "model_fir",
            "第二个模型": "model_sec",
            "模型二": "model_sec",
            "第二个模型路径": "model_sec",
            "模型位置": "model_path",
            "模型路径": "model_path",
            "数据集": "dataset_dir",
            "数据集路径":"dataset_dir",
            "数据集名称":"dataset_name",
            "模型类别": "template",
            #"输出路径": "output_dir",
            "时间": "schedule_time",
            "定时时间": "schedule_time",
            "schedule_time": "schedule_time",  # 添加这一行：允许直接使用参数名
            "操作类型":"action",
            "action":"action"
        }

        # 可用参数的中文描述
        self.param_descriptions = {
            "MBS": "批量大小/批次大小 (Micro Batch Size)",
            "ACC": "梯度累积步数 (Gradient Accumulation Steps)",
            "LR": "学习率 (Learning Rate)",
            "TEM": "模型/基础模型 (Model Template)",
            "RESUME": "恢复训练/继续训练/检查点 (Resume from checkpoint)",
            "LOCALHOST_ID": "设备/GPU/显卡 (GPU IDs)",
            "container": "Docker容器名称 (Docker container name)" , # 添加容器描述

            "schedule_time": "定时时间/时间 (Schedule Time)",
            "model_fir": "第一个模型路径 (First Model Path)",
            "model_sec": "第二个模型路径 (Second Model Path)",
            "model_path": "模型路径 (Model Path)",
            "dataset_dir": "数据集路径 (Dataset Directory)",
            "dataset_name": "数据集名称 (Dataset Name)",
            "template": "模型模板/类别 (Model Template)",
            #"output_dir": "输出路径 (Output Directory)",
            "CKPT_PATH": "检查点路径 (Checkpoint Path)",

            "action":"操作类型"

        }

        self.scripts = {

            "data_preprocessing": {
                "path": "data_preprocessing",
                "description": "数据预处理工具，用于对数据进行清洗、标注和格式化，以便于模型训练。",
                "aliases": ["数据预处理","预处理", "preprocess", "preprocessing","preprocess_sft_diagnosis"," data_transform","preprocess_data","data_preprocess"],
                "supports_background": False,
                "long_running": False,
                "default_env": {

                },
                "cli_args_only": True,  # 新增：标记此脚本只使用命令行参数
                "default_cli_args": {    # 新增：默认命令行参数
                    "input_folder": "/home/workspace/dataset",
                    "data_type":"",
                    "strategy":"",
                    "model_name": DEFAULT_MODEL_NAME,
                    "api_key": DEFAULT_MODEL_API_KEY,
                    "base_url": DEFAULT_MODEL_BASE_URL
                    },
                "required_cli_params": ["data_type", "strategy"],  # 新增：必需的命令行参数
                "supported_cli_params": ["input_folder","data_type","strategy","model_name","api_key","base_url"],  # 新增：支持的命令行参数
                "validate_existing_dirs": ["input_folder"],
                "container_relative_path_base": "/home/workspace",
                "param_mapping": {  # 新增：命令行参数的中文映射
                    "数据预处理输入路径": "input_folder",
                    "预处理数据类型": "data_type",
                    "预处理数据格式":"strategy",
                    "--input_folder": "input_folder",
                    "--data_type": "data_type",
                    "--strategy":"strategy",
                    "dpo处理模型名称":"model_name",
                    "dpo处理模型apikey":"api_key",
                    "dpo处理模型地址":"base_url",
                    "--model_name":"model_name",
                    "--api_key":"api_key",
                    "--base_url":"base_url"
                    },
                "supported_params": [],
                "default_args": {
                    "background": False,
                    "capture_output": False,
                    #"log_file": "/home/workspace/agent.log"
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/data_preprocessing",
                "requires_docker": True
            },
             "score_based_filtering": {
                "path": "score_based_filtering",
                "description": "数据高级筛选工具，用于对预处理完成的数据进行进一步的打分筛选，从而获得更高质量的模型训练数据",
                "aliases": ["数据高级筛选"],
                "supports_background": True,
                "long_running": True,
                "default_env": {

                },
                "cli_args_only": True,  # 新增：标记此脚本只使用命令行参数
                "default_cli_args": {    # 新增：默认命令行参数
                    "input_folder": "",
                    "threshold":"90",
                    "model_name": DEFAULT_MODEL_NAME,
                    "api_key": DEFAULT_MODEL_API_KEY,
                    "base_url": DEFAULT_MODEL_BASE_URL
                    },
                "required_cli_params": ["input_folder"],  # 新增：必需的命令行参数
                "supported_cli_params": ["input_folder","threshold","model_name","api_key","base_url"],  # 新增：支持的命令行参数
                "validate_existing_dirs": ["input_folder"],
                "container_relative_path_base": "/home/workspace",
                "param_mapping": {  # 新增：命令行参数的中文映射
                    "数据清洗输入路径": "input_folder",
                    "数据清洗阈值":"threshold",
                    "数据清洗模型名称":"model_name",
                    "数据清洗模型apikey":"api_key",
                    "数据清洗模型地址":"base_url",
                    "--input_folder": "input_folder",
                    "--threshold":"threshold",
                    "--model_name":"model_name",
                    "--api_key":"api_key",
                    "--base_url":"base_url"
                    },
                "supported_params": [],
                "default_args": {
                    "background": True,
                    "capture_output": False,
                    #"log_file": "/home/workspace/agent.log"
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/usr/local/insinfersystem",
                "docker_executable": "/usr/local/insinfersystem/score_based_filtering",
                "requires_docker": True
            },



        }

        # 添加脚本配置验证
        self._validate_scripts()

    def _validate_scripts(self):
        """验证脚本配置"""
        for name, info in self.scripts.items():
            script_path = info.get("path")
            if script_path and not os.path.exists(script_path):
                # 尝试从当前目录查找
                current_dir = os.path.join(os.getcwd(), script_path)
                #if not os.path.exists(current_dir):
                    #print(f"警告: 脚本 {name} 的路径在本容器内不存在: {script_path}")

    def map_chinese_to_env_var(self, chinese_param: str) -> Optional[str]:
        """将中文参数名映射到环境变量名"""
        return self.param_mapping.get(chinese_param)

    def get_available_params(self, script_name: str) -> List[str]:
        """获取脚本支持的参数列表（中文）"""
        script_info = self.scripts.get(script_name)
        if not script_info:
            return []

        supported_env_vars = script_info.get("supported_params", [])
        # 将环境变量名映射回中文（可能有多个中文对应一个环境变量）
        chinese_params = []
        for env_var in supported_env_vars:
            # 找到所有映射到这个环境变量的中文参数名
            chinese_list = [ch for ch, ev in self.param_mapping.items() if ev == env_var]
            chinese_params.extend(chinese_list)

        return list(set(chinese_params))  # 去重

    def get_param_description(self, env_var: str) -> str:
        """获取参数描述"""
        return self.param_descriptions.get(env_var, env_var)

    def find_script(self, query: str) -> Optional[Dict[str, Any]]:
        """根据名称、别名或描述查找脚本"""
        query = query.lower()

        for name, info in self.scripts.items():
            # 检查名称
            if query == name.lower():
                return {**info, "name": name}

            # 检查别名
            for alias in info.get("aliases", []):
                if query == alias.lower():
                    return {**info, "name": name}

            # 检查描述
            if query in info.get("description", "").lower():
                return {**info, "name": name}

        return None

    def get_script_info_with_cli_params(self, script_name: str) -> str:
        """获取脚本的详细信息，包括命令行参数"""
        script_info = self.scripts.get(script_name)
        if not script_info:
            return ""

        result = []
        result.append(f"脚本: {script_name}")
        result.append(f"描述: {script_info.get('description', '')}")

        # 显示命令行参数
        if script_info.get("cli_args_only", False):
            result.append("参数传递方式: 命令行参数")
            cli_params = script_info.get("supported_cli_params", [])
            if cli_params:
                result.append("支持的命令行参数:")
                for cli_param in cli_params:
                    # 找到对应的中文名
                    chinese_names = [ch for ch, ev in script_info.get("param_mapping", {}).items() if ev == cli_param]
                    # 标记必需参数
                    is_required = cli_param in script_info.get("required_cli_params", [])
                    if chinese_names:
                        result.append(f"  - {cli_param} ({'/'.join(chinese_names)}) {'(必需)' if is_required else ''}")
                    else:
                        result.append(f"  - {cli_param} {'(必需)' if is_required else ''}")
        else:
            # 显示环境变量参数
            result.append("参数传递方式: 环境变量")
            supported_params = script_info.get("supported_params", [])
            if supported_params:
                result.append("支持的环境变量参数:")
                for param in supported_params:
                    # 找到对应的中文名
                    chinese_names = [ch for ch, ev in self.param_mapping.items() if ev == param]
                    desc = self.get_param_description(param)
                    if chinese_names:
                        result.append(f"  - {param} ({'/'.join(chinese_names)}) - {desc}")
                    else:
                        result.append(f"  - {param} - {desc}")

        return "\n".join(result)





    def list_scripts(self) -> str:
        """列出所有可用脚本"""
        result = ["可用脚本："]
        for name, info in self.scripts.items():
            result.append(f"- {name}: {info['description']}")
            result.append(f"  别名: {', '.join(info.get('aliases', []))}")
            result.append(f"  路径: {info['path']}")

            if info.get("cli_args_only", False):
                # 显示命令行参数
                cli_params = info.get('supported_cli_params', [])
                if cli_params:
                    param_descs = []
                    for param in cli_params:
                        # 找到对应的中文名
                        chinese_names = [ch for ch, ev in info.get('param_mapping', {}).items() if ev == param]
                        is_required = param in info.get('required_cli_params', [])
                        if chinese_names:
                            param_descs.append(f"{param} ({'/'.join(chinese_names)}) {'(必需)' if is_required else ''}")
                        else:
                            param_descs.append(f"{param} {'(必需)' if is_required else ''}")
                    result.append(f"  命令行参数: {', '.join(param_descs)}")
            else:
                # 显示环境变量参数
                supported_params = info.get('supported_params', [])
                if supported_params:
                    param_descs = []
                    for param in supported_params:
                        desc = self.get_param_description(param)
                        # 找到对应的中文名
                        chinese_names = [ch for ch, ev in self.param_mapping.items() if ev == param]
                        if chinese_names:
                            param_descs.append(f"{param} ({'/'.join(chinese_names)}) - {desc}")
                        else:
                            param_descs.append(f"{param} - {desc}")
                    result.append(f"  可修改参数: {', '.join(param_descs)}")

            # 显示默认配置
            if info.get('default_env'):
                result.append(f"  默认环境变量: {info['default_env']}")
            if info.get('default_cli_args'):
                result.append(f"  默认命令行参数: {info['default_cli_args']}")

            # 显示特殊属性
            if info.get('supports_background'):
                result.append("  支持后台运行")
            if info.get('long_running'):
                result.append("  长时间运行任务")
            result.append("")  # 空行分隔
        return "\n".join(result)


def run_local_script(
    script_path: str = "test.py",
    script_args: Dict[str, Any] = None,
    env_vars: Dict[str, str] = None,
    timeout: int = None,
    capture_output: bool = True,
    working_dir: str = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    positional_args: List[str] = None,  # 新增：位置参数
    **kwargs
) -> ToolResponse:
    """
    运行本地Python脚本并返回执行结果。
    """

    try:
        if docker_container:
            return run_script_in_docker(
                script_path=script_path,
                script_args=script_args,
                positional_args=positional_args,  # 传递给Docker函数
                env_vars=env_vars,
                timeout=timeout,
                capture_output=capture_output,
                working_dir=working_dir,
                docker_container=docker_container,
                docker_image=docker_image,
                docker_volume_mounts=docker_volume_mounts,
                docker_working_dir=docker_working_dir,
                docker_executable=docker_executable,
                **kwargs
            )

        # 检查脚本文件是否存在
        abs_script_path = os.path.abspath(script_path)
        if not os.path.exists(abs_script_path):
            # 尝试在当前目录下查找
            current_path = os.path.join(os.getcwd(), script_path)
            if os.path.exists(current_path):
                abs_script_path = os.path.abspath(current_path)
            else:
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"错误: 找不到脚本文件: {script_path}\n尝试的路径: {abs_script_path}\n当前目录: {os.getcwd()}"
                    )
                ])

        # 构建Python执行命令
        python_executable = sys.executable
        command = [python_executable, abs_script_path]
        if positional_args:
            for arg in positional_args:
                command.append(str(arg))
        # 添加命令行参数
        if script_args:
            for key, value in script_args.items():
                # 处理不同类型的参数
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    # 对于字符串参数，确保特殊字符不被shell解释
                    # 特别是对于时间参数如 "{16:08:55}"，需要直接传递
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                    continue
                else:
                    command.extend([f"--{key}", str(value)])

        # 设置环境变量
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # 设置工作目录
        cwd = working_dir if working_dir else os.path.dirname(abs_script_path)
        if not cwd or not os.path.exists(cwd):
            cwd = os.getcwd()

        # 打印调试信息
        print(f"执行脚本: {abs_script_path}")
        print(f"工作目录: {cwd}")
        print(f"命令: {' '.join(command)}")
        print(f"环境变量: {env_vars}")

        # 执行脚本
        if timeout:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd
            )
        else:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                env=env,
                cwd=cwd
            )

        # 构建结果消息
        result_text = []

        if process.stdout and capture_output:
            result_text.append(f"标准输出:\n{process.stdout}")

        if process.stderr and capture_output:
            result_text.append(f"标准错误:\n{process.stderr}")

        # 添加返回码信息
        result_text.append(f"返回码: {process.returncode}")
        result_text.append(f"######请检查对应的log文件######")
        # 如果没有捕获输出但有返回码，只显示返回码
        if not capture_output:
            result_text = [f"脚本执行完成，返回码: {process.returncode}"]

        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="\n\n".join(result_text)
            )
        ])

    except subprocess.TimeoutExpired:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 脚本执行超时 (超过 {timeout} 秒)"
            )
        ])

    except FileNotFoundError as e:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 找不到Python解释器或脚本文件: {str(e)}"
            )
        ])

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 执行脚本时发生异常: {str(e)}\n\n详细错误:\n{error_trace}"
            )
        ])


GENERAL_DATA_PREPROCESSING_FORMATS = {"openai", "sharegpt", "sft", "dpo", "text"}
RAW_DATA_PREPROCESSING_FORMATS = {"raw", "unknown", ""}


def _is_messages_format(item: Dict[str, Any], messages_key: str, role_key: str, content_key: str) -> bool:
    messages = item.get(messages_key)
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    return all(
        isinstance(message, dict)
        and isinstance(message.get(role_key), str)
        and isinstance(message.get(content_key), str)
        for message in messages
    )


def detect_data_preprocessing_item_format(item: Any) -> str:
    """Mirror llamafactory/data_pipeline_utils.py::detect_item_format for agent-side gating."""
    if not isinstance(item, dict):
        return "unknown"
    keys = set(item)
    if _is_messages_format(item, "messages", "role", "content"):
        return "openai"
    if _is_messages_format(item, "conversations", "from", "value"):
        return "sharegpt"
    if {"input", "chosen", "rejected", "instruction"} <= keys:
        return "dpo"
    if {"input", "output", "instruction"} <= keys:
        return "sft"
    if "text" in keys:
        return "text"
    if keys & {"主诉", "现病史", "诊断"}:
        return "raw"
    return "unknown"


def _iter_sample_json_items(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload[:5]
    return [payload]


def detect_data_preprocessing_text_format(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown"
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except (TypeError, ValueError):
                continue
            detected = detect_data_preprocessing_item_format(payload)
            if detected != "unknown":
                return detected
        return "unknown"

    for item in _iter_sample_json_items(payload):
        detected = detect_data_preprocessing_item_format(item)
        if detected != "unknown":
            return detected
    return "unknown"


def detect_data_preprocessing_dir_format_in_container(
    docker_container: str,
    container_path: str,
    *,
    sample_limit: int = 5,
) -> str:
    if not docker_container or not container_path:
        return "unknown"
    find_command = (
        f"find {shlex.quote(container_path)} -maxdepth 1 -type f "
        "\\( -name '*.json' -o -name '*.jsonl' \\) "
        "! -name 'dataset_info.json' "
        "! -name 'preprocessing_audit.json' "
        "! -name 'preprocessing_summary.json' "
        "! -name 'score_audit.json' "
        "! -name 'score_progress.json' "
        "! -name 'score_summary.json' "
        "! -name 'test.json' "
        f"-print | sort | head -n {max(1, int(sample_limit))}"
    )
    try:
        files_result = subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c", find_command],
            capture_output=True,
            text=True,
        )
    except Exception:
        logger.exception("Failed to list preprocessing data samples in container.")
        return "unknown"

    if files_result.returncode != 0:
        return "unknown"

    for file_path in [line.strip() for line in str(files_result.stdout or "").splitlines() if line.strip()]:
        try:
            content_result = subprocess.run(
                ["docker", "exec", docker_container, "head", "-c", "65536", file_path],
                capture_output=True,
                text=True,
            )
        except Exception:
            logger.exception("Failed to read preprocessing data sample: %s", file_path)
            continue
        if content_result.returncode != 0:
            continue
        detected = detect_data_preprocessing_text_format(content_result.stdout)
        if detected != "unknown":
            return detected
    return "unknown"


def effective_data_preprocessing_required_params(
    script_name: str,
    script_info: Dict[str, Any],
    cli_args: Dict[str, str],
    detected_format: str = "",
) -> List[str]:
    required_params = list(script_info.get("required_cli_params", []))
    if script_name != "data_preprocessing":
        return required_params

    data_type = str(cli_args.get("data_type") or "").strip().lower()
    detected_format = str(detected_format or "").strip().lower()
    if detected_format in GENERAL_DATA_PREPROCESSING_FORMATS:
        if data_type in {"sft", "dpo"}:
            return []
        return ["data_type"]
    if detected_format == "unknown":
        return []
    return required_params



def cleanup_data_preprocessing_cli_args(
    script_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    args: Dict[str, Any],
    detected_format: str,
) -> None:
    detected_format = str(detected_format or "").strip().lower()
    data_type = str(cli_args.get("data_type") or "").strip().lower()
    supported_cli_params = set(script_info.get("supported_cli_params", []))
    default_cli_args = script_info.get("default_cli_args", {})

    if (
        detected_format in GENERAL_DATA_PREPROCESSING_FORMATS
        and data_type in {"sft", "dpo"}
        and not str(cli_args.get("strategy") or "").strip()
    ):
        cli_args.pop("strategy", None)
        args.pop("strategy", None)

    for key in list(cli_args.keys()):
        if key not in supported_cli_params:
            continue
        value = cli_args.get(key)
        if str(value or "").strip():
            continue
        if key in default_cli_args:
            cli_args.pop(key, None)
            args.pop(key, None)

def check_required_params(
    script_info: Dict[str, Any],
    cli_args: Dict[str, str],
    required_params: Optional[List[str]] = None,
) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    检查必需参数是否已提供

    Args:
        script_info: 脚本信息
        cli_args: 当前已提供的命令行参数

    Returns:
        Tuple[是否通过检查, 缺失参数列表, 需要询问的参数信息]
    """
    required_params = list(required_params if required_params is not None else script_info.get("required_cli_params", []))
    missing_params = []
    params_to_ask = {}

    for param in required_params:
        # 检查参数是否已提供
        if param not in cli_args or not cli_args[param]:
            missing_params.append(param)

            # 获取参数描述
            param_desc = param
            param_mapping = script_info.get("param_mapping", {})
            chinese_names = [ch for ch, ev in param_mapping.items() if ev == param]
            if chinese_names:
                param_desc = f"{param} ({'/'.join(chinese_names)})"

            params_to_ask[param] = {
                "name": param,
                "description": param_desc,
                "chinese_names": chinese_names
            }

    return len(missing_params) == 0, missing_params, params_to_ask


def _is_placeholder_value(param_name: str, value: Any) -> bool:
    """识别模型编造的占位参数，避免误启动数据处理任务。"""
    if value is None:
        return True

    text = str(value).strip().strip(chr(39) + chr(34))
    if not text:
        return True

    lower = text.lower()
    placeholders = {
        "待提供",
        "未提供",
        "未指定",
        "未知",
        "默认",
        "xxx",
        "none",
        "null",
        "unknown",
        "input_folder",
        "data_type",
        "strategy",
        "需高级筛选的数据路径",
        "高级筛选的数据路径",
        "数据路径",
        "数据目录",
        "文件夹路径",
    }
    if text in placeholders or lower in placeholders:
        return True

    if param_name == "input_folder":
        normalized_path = lower.replace("\\", "/").rstrip("/")
        if normalized_path in {
            "/path/to/your/data",
            "<你的数据目录路径>",
            "data/processed_data",
            "processed_data",
            "/home/workspace/data/processed_data",
        }:
            return True
        if re.fullmatch(r"20\d{6}", text):
            return True
        if any(token in text for token in ["数据路径", "数据目录", "文件夹路径", "高级筛选"]) and not text.startswith("/"):
            return True

    return False


def validate_cli_param_values(
    script_info: Dict[str, Any],
    cli_args: Dict[str, str],
    required_params: Optional[List[str]] = None,
) -> List[str]:
    invalid = []
    for param in list(required_params if required_params is not None else script_info.get("required_cli_params", [])):
        value = cli_args.get(param)
        if _is_placeholder_value(param, value):
            invalid.append(f"{param}={value!r}")
    if "data_type" in cli_args and str(cli_args.get("data_type") or "").strip():
        data_type = str(cli_args.get("data_type") or "").strip().lower()
        if data_type not in {"sft", "dpo"}:
            invalid.append(f"data_type={cli_args.get('data_type')!r}")
    return invalid


def _resolve_container_path(path_value: str, base_dir: Optional[str] = None) -> str:
    """Resolve a CLI path the same way the container process will see it."""
    path_text = str(path_value or "").strip()
    if path_text.startswith("/"):
        return posixpath.normpath(path_text)
    return posixpath.normpath(posixpath.join(base_dir or "/", path_text))


def validate_docker_dir_params(
    script_name: str,
    script_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    docker_container: Optional[str],
    docker_working_dir: Optional[str] = None,
) -> Optional[ToolResponse]:
    """Validate configured directory CLI args inside the target Docker container."""
    dir_params = script_info.get("validate_existing_dirs", [])
    if not dir_params or not docker_container:
        return None

    for param_name in dir_params:
        raw_path = cli_args.get(param_name)
        if _is_placeholder_value(param_name, raw_path):
            continue

        relative_path_base = script_info.get("container_relative_path_base") or docker_working_dir
        container_path = _resolve_container_path(str(raw_path), relative_path_base)
        exists_cmd = ["docker", "exec", docker_container, "test", "-e", container_path]
        exists_result = subprocess.run(exists_cmd, capture_output=True, text=True)

        if exists_result.returncode != 0:
            response_text = (
                f"错误！脚本 '{script_name}' 启动前检查未通过，已阻止执行。\n"
                f"- 参数 {param_name} 指向的目录不存在: {raw_path}\n"
                f"- 容器内解析路径: {container_path}\n"
                f"请确认该目录已在容器 {docker_container} 中创建，或重新提供正确的 {param_name}。"
            )
            return _data_error_response(
                script_name,
                response_text,
                error_reason="container_path_missing",
                required_params=[param_name],
                recoverable=True,
                container=docker_container,
                containerPath=container_path,
            )

        dir_cmd = ["docker", "exec", docker_container, "test", "-d", container_path]
        dir_result = subprocess.run(dir_cmd, capture_output=True, text=True)
        if dir_result.returncode != 0:
            response_text = (
                f"错误！脚本 '{script_name}' 启动前检查未通过，已阻止执行。\n"
                f"- 参数 {param_name} 不是目录: {raw_path}\n"
                f"- 容器内解析路径: {container_path}\n"
                f"请提供一个可访问的数据文件夹路径。"
            )
            return _data_error_response(
                script_name,
                response_text,
                error_reason="container_path_not_directory",
                required_params=[param_name],
                recoverable=True,
                container=docker_container,
                containerPath=container_path,
            )


        if script_name in {"data_preprocessing", "score_based_filtering"} and param_name == "input_folder":
            def candidate_file_check() -> Optional[ToolResponse]:
                find_command = (
                    f"find {shlex.quote(container_path)} -maxdepth 1 -type f "
                    "\\( -name '*.json' -o -name '*.jsonl' \\) "
                    "! -name 'dataset_info.json' "
                    "! -name 'preprocessing_audit.json' "
                    "! -name 'preprocessing_summary.json' "
                    "! -name 'score_audit.json' "
                    "! -name 'score_progress.json' "
                    "! -name 'score_summary.json' "
                    "! -name 'test.json' -print -quit"
                )
                candidate_result = subprocess.run(
                    ["docker", "exec", docker_container, "sh", "-c", find_command],
                    capture_output=True,
                    text=True,
                )
                if candidate_result.returncode != 0 or not str(candidate_result.stdout or "").strip():
                    response_text = (
                        f"错误！脚本 '{script_name}' 启动前检查未通过，已阻止执行。\n"
                        f"- 参数 {param_name} 指向的目录没有可处理的数据文件: {raw_path}\n"
                        f"- 容器内解析路径: {container_path}\n"
                        "请提供包含真实 .json/.jsonl 数据文件的输入目录。"
                    )
                    return _data_error_response(
                        script_name,
                        response_text,
                        error_reason="dataset_files_missing",
                        required_params=[param_name],
                        recoverable=True,
                        container=docker_container,
                        containerPath=container_path,
                    )
                return None

            if script_name == "score_based_filtering":
                info_command = f"cat {shlex.quote(posixpath.join(container_path, 'dataset_info.json'))} 2>/dev/null"
                info_result = subprocess.run(
                    ["docker", "exec", docker_container, "sh", "-c", info_command],
                    capture_output=True,
                    text=True,
                )
                if info_result.returncode == 0 and str(info_result.stdout or "").strip():
                    try:
                        dataset_info = json.loads(info_result.stdout)
                    except (TypeError, ValueError):
                        return _data_error_response(
                            script_name,
                            f"dataset_info.json 格式无效：{posixpath.join(container_path, 'dataset_info.json')}",
                            error_reason="dataset_info_invalid",
                            required_params=[param_name],
                            recoverable=True,
                            container=docker_container,
                            containerPath=container_path,
                        )
                    if not isinstance(dataset_info, dict) or not dataset_info:
                        return _data_error_response(
                            script_name,
                            f"dataset_info.json 没有注册任何可处理数据集：{posixpath.join(container_path, 'dataset_info.json')}",
                            error_reason="dataset_info_empty",
                            required_params=[param_name],
                            recoverable=True,
                            container=docker_container,
                            containerPath=container_path,
                        )
                    missing_files = []
                    invalid_files = []
                    for dataset_key, entry in dataset_info.items():
                        key = str(dataset_key).strip() or "<空数据集名>"
                        file_name = entry.get("file_name") if isinstance(entry, dict) else None
                        if not isinstance(file_name, str) or not file_name.strip():
                            invalid_files.append(key)
                            continue
                        clean_file_name = file_name.strip()
                        parts = [part for part in clean_file_name.split("/") if part]
                        if clean_file_name.startswith("/") or "\\" in clean_file_name or ".." in parts:
                            invalid_files.append(f"{key}: {clean_file_name}")
                            continue
                        file_path = posixpath.join(container_path, clean_file_name)
                        file_result = subprocess.run(
                            ["docker", "exec", docker_container, "test", "-f", file_path],
                            capture_output=True,
                            text=True,
                        )
                        if file_result.returncode != 0:
                            missing_files.append(f"{key}: {clean_file_name}")
                    if invalid_files:
                        return _data_error_response(
                            script_name,
                            f"dataset_info.json 中以下 file_name 无效：{', '.join(invalid_files)}。",
                            error_reason="dataset_info_file_name_invalid",
                            required_params=[param_name],
                            recoverable=True,
                            container=docker_container,
                            containerPath=container_path,
                        )
                    if missing_files:
                        return _data_error_response(
                            script_name,
                            f"dataset_info.json 引用的数据文件不存在：{', '.join(missing_files)}。",
                            error_reason="dataset_files_missing",
                            required_params=[param_name],
                            recoverable=True,
                            container=docker_container,
                            containerPath=container_path,
                        )
                else:
                    candidate_error = candidate_file_check()
                    if candidate_error:
                        return candidate_error
            else:
                candidate_error = candidate_file_check()
                if candidate_error:
                    return candidate_error
    return None


def resolve_latest_dataset_subdir(docker_container: str, root_path: str) -> Optional[str]:
    """Return the first-level dataset directory with the newest JSON/JSONL file."""
    script = (
        'root="$1"; [ -d "$root" ] || exit 2; '
        'for d in "$root"/*; do '
        '[ -d "$d" ] || continue; '
        'latest=$(find "$d" -maxdepth 1 -type f '
        '\\( -name "*.json" -o -name "*.jsonl" \\) '
        '! -name "preprocessing_audit.json" '
        '! -name "preprocessing_summary.json" '
        '! -name "score_audit.json" '
        '! -name "score_progress.json" '
        '! -name "score_summary.json" '
        '! -name "test.json" '
        '-printf "%T@\\n" 2>/dev/null | sort -nr | head -n 1); '
        '[ -n "$latest" ] && printf "%s\\t%s\\n" "$latest" "$d"; '
        'done | sort -nr | head -n 1 | cut -f2-'
    )
    result = subprocess.run(
        ["docker", "exec", docker_container, "sh", "-c", script, "dataset-root", root_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def infer_data_preprocessing_cli_args(script_query: str, script_name: str) -> Dict[str, str]:
    """Infer stable preprocessing CLI args from script aliases such as preprocess_sft_diagnosis."""
    if script_name != "data_preprocessing":
        return {}

    query = (script_query or "").lower()
    inferred = {}
    if "sft" in query:
        inferred["data_type"] = "sft"
    elif "dpo" in query:
        inferred["data_type"] = "dpo"

    for strategy in ("inspection", "diagnosis", "prescription"):
        if strategy in query:
            inferred["strategy"] = strategy
            break

    return inferred


def run_script_by_name_data(
    script_query: str,
    additional_args: Dict[str, Any] = None,
    list_only: bool = False,
    background: bool = None,
    env_vars: Dict[str, str] = None,
    use_docker: bool = None,
    skip_prompt: bool = False,  # 新增：是否跳过参数询问
    **kwargs
) -> ToolResponse:
    """
    根据名称、别名或描述运行脚本，支持修改参数

    Args:
        script_query (`str`): 脚本名称、别名或描述
        additional_args (`Dict[str, Any]`): 额外的脚本参数
        list_only (`bool`): 如果为True，只列出脚本不运行
        background (`bool`): 是否在后台运行
        env_vars (`Dict[str, str]`): 环境变量，可以包含container参数
        skip_prompt (`bool`): 是否跳过必需参数询问（用于内部递归调用）
        **kwargs: 传递给run_local_script的额外参数

    Returns:
        `ToolResponse`: 包含脚本执行结果的ToolResponse对象
    """
    manager = ScriptManager()

    if list_only or script_query.lower() in ["列表", "list", "ls", "all"]:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=manager.list_scripts()
            )
        ])

    script_info = manager.find_script(script_query)

    if not script_info:
        response_text = f"未找到脚本: {script_query}\n\n{manager.list_scripts()}"
        return _data_error_response(
            str(script_query),
            response_text,
            error_reason="script_not_found",
            recoverable=True,
        )

    script_name = script_info["name"]
    # 获取脚本是否使用位置参数
    uses_positional_args = script_info.get("uses_positional_args", False)
    positional_param_name = script_info.get("positional_param_name", "")
    # 初始化所有变量
    params_to_update = {}
    cli_params_to_update = {}  # 添加这一行
    positional_args = []  # 新增：位置参数列表
    unknown_params = []

     # 修改：为特定脚本添加特殊处理
    if script_name == "create_command_vpn":
        # create_command_vpn 使用位置参数
        script_info["uses_positional_args"] = True
        script_info["positional_param_name"] = "schedule_time"
    if script_name == "start-server":
        # create_command_vpn 使用位置参数
        script_info["uses_positional_args"] = True
        script_info["positional_param_name"] = "action"
    # 解析参数修改请求
    if additional_args:

        for param_name, param_value in additional_args.items():
            # 检查是否是命令行参数（针对特定脚本）
            if script_info.get("cli_args_only", False):
                # 尝试将中文参数名映射到命令行参数名
                cli_param = None

                # 首先检查脚本特定的映射
                if script_info.get("param_mapping"):
                    cli_param = script_info.get("param_mapping", {}).get(param_name)

                # 如果没有找到，检查通用映射
                if not cli_param:
                    cli_param = manager.map_chinese_to_env_var(param_name)

                # 如果还没有找到，但参数名直接就是支持的参数名，则直接使用
                if not cli_param and param_name in script_info.get("supported_cli_params", []):
                    cli_param = param_name
                print(cli_param)
                print(positional_param_name)
                #cli_param = script_info.get("param_mapping", {}).get(param_name)
                if cli_param:
                    # 检查该参数是否被脚本支持
                    if uses_positional_args and cli_param == positional_param_name:
                        # 如果是使用位置参数的脚本，并且是位置参数名，添加到位置参数列表
                        positional_args.append(str(param_value))
                    elif cli_param in script_info.get("supported_cli_params", []):
                        cli_params_to_update[cli_param] = str(param_value)
                    else:
                        unknown_params.append(f"{param_name} ({cli_param})")
                else:
                    unknown_params.append(param_name)
            else:
                # 尝试将中文参数名映射到环境变量名
                env_var = manager.map_chinese_to_env_var(param_name)
                if env_var:
                    # 检查该参数是否被脚本支持
                    if env_var in script_info.get("supported_params", []):
                        params_to_update[env_var] = str(param_value)
                    else:
                        unknown_params.append(f"{param_name} ({env_var})")
                else:
                    unknown_params.append(param_name)

        # 如果有未知参数，则提示可用参数而不运行脚本
        if unknown_params:
            available_params_text = []

            if script_info.get("cli_args_only", False):
                supported_cli_params = script_info.get("supported_cli_params", [])
                if supported_cli_params:
                    available_params_text.append("该脚本通过命令行参数接收输入:")
                    for cli_param in script_info.get("supported_cli_params", []):
                        # 找到对应的中文名
                        param_mapping = script_info.get("param_mapping", {})
                        chinese_names = [ch for ch, ev in param_mapping.items() if ev == cli_param]
                        if chinese_names:
                            available_params_text.append(f"  - {cli_param} ({'/'.join(chinese_names)})")
                        else:
                            available_params_text.append(f"  - {cli_param}")
                        # 如果是位置参数，特别说明
                        if uses_positional_args and cli_param == positional_param_name:
                            available_params_text[-1] += " (位置参数)"
            else:

                available_params = manager.get_available_params(script_name)
                if available_params:
                    available_params_text.append("该脚本通过环境变量接收输入:")
                    for param in available_params:
                        env_var = manager.map_chinese_to_env_var(param)
                        desc = manager.get_param_description(env_var)
                        available_params_text.append(f"  - {param}: {desc}")
                        #param_descriptions.append(f"  - {param}: {desc}")
                else:
                    available_params_text.append("该脚本没有可配置的参数")
            response_text = (
                f"错误！脚本 '{script_name}' 不支持以下参数: {', '.join(unknown_params)}\n\n"
                f"该脚本支持的参数:\n" + "\n".join(available_params_text) + "\n\n"
                f"请使用支持的参数重新指定，然后确认运行。"
            )
            return _data_error_response(
                script_name,
                response_text,
                error_reason="unsupported_params",
                recoverable=True,
                invalidParams=unknown_params,
            )

    # 准备命令行参数
    cli_args = {}
    if script_info.get("default_cli_args"):
        cli_args = script_info.get("default_cli_args", {}).copy()

    # 别名如 preprocess_sft_diagnosis 应稳定补齐必填参数，避免依赖模型每次显式传参。
    for cli_param, value in infer_data_preprocessing_cli_args(script_query, script_name).items():
        if not str(cli_args.get(cli_param) or "").strip():
            cli_args[cli_param] = value

    # 更新命令行参数
    for cli_param, value in cli_params_to_update.items():
        cli_args[cli_param] = value

    def missing_required_response(
        required_params: List[str],
        missing_params: List[str],
        params_to_ask: Dict[str, Any],
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        nl = chr(10)
        prompt_text = f"脚本 '{script_name}' 还需要补充以下参数，补充后将继续处理：{nl}{nl}"
        for param_name, param_info in params_to_ask.items():
            prompt_text += f"- {param_info['description']}{nl}"
        current_args = {
            "additional_args": additional_args or {},
            "cli_params_to_update": cli_params_to_update,
            "params_to_update": params_to_update,
            "positional_args": positional_args,
        }
        extra_fields = dict(extra_fields or {})
        for key in ("inputFolder", "selectedInputFolder", "detectedFormat", "sourceInputFolder"):
            if key in extra_fields:
                current_args[key] = extra_fields[key]
        prompt_text += f"当前参数: {json.dumps(current_args, ensure_ascii=False)}{nl}"
        return _data_error_response(
            script_name,
            prompt_text,
            error_reason="missing_required_params",
            required_params=required_params,
            missing_params=missing_params,
            recoverable=True,
            currentArgs=current_args,
            **extra_fields,
        )

    def invalid_required_response(
        required_params: List[str],
        missing_params: List[str],
        invalid_params: List[str],
        all_params_provided: bool,
    ) -> ToolResponse:
        missing_items = [f"{param}={cli_args.get(param)!r}" for param in missing_params]
        blocked_items = invalid_params or missing_items
        response_text = (
            f"错误！脚本 '{script_name}' 的必需参数存在无效值或占位值，已阻止启动：\n"
            + "\n".join(f"- {item}" for item in blocked_items)
            + "\n请提供真实可用的参数后再继续。"
        )
        return _data_error_response(
            script_name,
            response_text,
            error_reason="invalid_required_params",
            required_params=required_params,
            missing_params=(missing_params if not all_params_provided and skip_prompt else []),
            recoverable=True,
            invalidParams=blocked_items,
        )

    # data_preprocessing 的 data_type/strategy 必填规则依赖输入目录格式，目录校验后再判定。
    if script_info.get("cli_args_only", False):
        initial_required_params = [] if script_name == "data_preprocessing" else effective_data_preprocessing_required_params(script_name, script_info, cli_args)
        all_params_provided, missing_params, params_to_ask = check_required_params(
            script_info,
            cli_args,
            initial_required_params,
        )

        if not all_params_provided and not skip_prompt:
            return missing_required_response(initial_required_params, missing_params, params_to_ask)

        invalid_params = validate_cli_param_values(script_info, cli_args, initial_required_params)
        if invalid_params or (not all_params_provided and skip_prompt):
            return invalid_required_response(initial_required_params, missing_params, invalid_params, all_params_provided)

    # 为 score_based_filtering 推导并固定输出目录，便于前端轮询进度
    filter_output_folder = None
    filter_output_dataset_name = None
    if script_name == "score_based_filtering":
        input_folder = str(cli_args.get("input_folder") or "").rstrip("/")
        try:
            threshold = float(cli_args.get("threshold") or 90)
        except (TypeError, ValueError):
            threshold = 90.0
        if input_folder and not str(cli_args.get("output_folder") or "").strip():
            filter_output_folder = f"{input_folder}_{threshold:g}_score_filter"
            cli_args["output_folder"] = filter_output_folder
            filter_output_dataset_name = posixpath.basename(filter_output_folder)
        elif str(cli_args.get("output_folder") or "").strip():
            filter_output_folder = str(cli_args["output_folder"]).rstrip("/")
            filter_output_dataset_name = posixpath.basename(filter_output_folder)

    # 获取脚本路径
    script_path = script_info.get("path")

    # 检查是否需要Docker执行
    requires_docker = script_info.get("requires_docker", False)
    use_docker_flag = use_docker or (use_docker is None and requires_docker)

    # 如果不使用Docker，检查主机上的脚本路径
    if not use_docker_flag:
        use_docker_flag=True
        #if not os.path.exists(script_path):
            # 尝试从当前目录查找
        #    current_path = os.path.join(os.getcwd(), script_path)
        #    if not os.path.exists(current_path):
        #        return ToolResponse(content=[
        #            TextBlock(
        #                type="text",
        #                text=f"错误: 脚本文件不存在: {script_path}\n请检查路径是否正确。"
        #            )
        #        ])
        #    script_path = current_path

    # 检查是否需要后台运行
    if background is None:
        background = script_info.get("default_args", {}).get("background", False)

    # 检查是否是长时间运行任务
    long_running = script_info.get("long_running", False)

    # 准备环境变量
    final_env_vars = {}

    # 添加脚本默认环境变量
    if script_info.get("default_env"):
        final_env_vars.update(script_info["default_env"])

    # 添加用户提供的环境变量
    if env_vars:
        final_env_vars.update(env_vars)
    # 更新环境变量参数
    for env_var, value in params_to_update.items():
        final_env_vars[env_var] = value

    # 容错：模型有时会把 input_folder 误塞进 env_vars.container。
    # container 必须是 Docker 容器名；以 / 开头的值应视为数据路径而不是容器。
    misplaced_container_path = None
    if "container" in final_env_vars:
        container_value = str(final_env_vars.get("container") or "").strip()
        if container_value.startswith("/"):
            misplaced_container_path = container_value
            final_env_vars.pop("container", None)
            if (
                script_info.get("cli_args_only", False)
                and "input_folder" in script_info.get("supported_cli_params", [])
                and not str(cli_args.get("input_folder") or "").strip()
            ):
                cli_args["input_folder"] = misplaced_container_path

    # 提取用户指定的容器名称（如果有）
    user_container = None
    if "container" in final_env_vars:
        user_container = final_env_vars.pop("container")  # 从环境变量中移除，因为它不是真正的环境变量
    logger.info(
        "数据处理目标Docker容器: %s",
        user_container or script_info.get("docker_container") or "<未指定>",
    )

    # 如果有参数需要更新，则更新环境变量
    #if additional_args:
        # 更新参数
       # for param_name, param_value in additional_args.items():
      #      env_var = manager.map_chinese_to_env_var(param_name)
       #     if env_var and env_var in script_info.get("supported_params", []):
        #        final_env_vars[env_var] = str(param_value)

    # 如果有位置参数，需要特殊处理
    #if uses_positional_args and positional_args:
        # 对于使用位置参数的脚本，我们需要确保正确的参数传递方式
        # 我们将在后面的执行中处理
        #pass
    # 合并默认参数和额外参数
    args = script_info.get("default_args", {}).copy()

    # 移除后台运行相关的参数
    args.pop("background", None)

    # 将命令行参数合并到args中
    args.update(cli_args)

    # 准备传递给run_local_script的kwargs
    run_kwargs = {
        "script_path": script_path,
        "script_args": args,
        "positional_args": positional_args,  # 添加位置参数
        "env_vars": final_env_vars,
        **kwargs
    }

    # 对于长时间运行的任务，设置较大的超时时间或不设超时
    if long_running and "timeout" not in kwargs:
        run_kwargs["timeout"] = None  # 不设超时

    # 如果脚本要求Docker或用户指定使用Docker
    if use_docker_flag:
        # 准备Docker相关参数 - 使用用户指定的容器或默认容器
        docker_kwargs = {
            "docker_container": user_container or script_info.get("docker_container"),
            "docker_image": script_info.get("docker_image"),
            "docker_volume_mounts": script_info.get("docker_volume_mounts"),
            "docker_working_dir": script_info.get("docker_working_dir",
                                               script_info.get("default_args", {}).get("working_dir")),
            "docker_executable": script_info.get("docker_executable", "python"),
        }

        # 检查是否指定了容器
        if not docker_kwargs["docker_container"]:
            response_text = (
                f"错误: 需要指定Docker容器才能运行脚本\n"
                f"脚本: {script_info['description']}\n"
                f"请通过环境变量指定容器，例如: env_vars={{'container': 'qwen3_zh'}}"
            )
            return _data_error_response(
                script_name,
                response_text,
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )

        preprocessing_diagnostics: Dict[str, Any] = {}
        if script_name == "data_preprocessing":
            relative_path_base = script_info.get("container_relative_path_base") or docker_kwargs.get("docker_working_dir")
            source_input_folder = _resolve_container_path(str(cli_args.get("input_folder") or ""), relative_path_base)
            user_provided_input_folder = bool(
                str(cli_params_to_update.get("input_folder") or "").strip()
                or str((additional_args or {}).get("input_folder") or "").strip()
            )
            selected_input_folder = ""
            input_source_kind = "explicit" if user_provided_input_folder else "default"
            if source_input_folder == "/home/workspace/dataset":
                selected_input_folder = resolve_latest_dataset_subdir(
                    docker_kwargs["docker_container"],
                    "/home/workspace/dataset",
                ) or ""
                if not selected_input_folder:
                    return _data_error_response(
                        script_name,
                        "未在 /home/workspace/dataset 的一级子目录中找到可处理的 JSON/JSONL 数据文件，已停止预处理。",
                        error_reason="dataset_input_not_found",
                        recoverable=True,
                        container=docker_kwargs["docker_container"],
                        containerPath="/home/workspace/dataset",
                        inputFolder="/home/workspace/dataset",
                        sourceInputFolder=input_source_kind,
                    )
                cli_args["input_folder"] = selected_input_folder
                args["input_folder"] = selected_input_folder
                input_source_kind = "explicit_latest" if user_provided_input_folder else "default_latest"

            preprocessing_diagnostics = {
                "inputFolder": _resolve_container_path(str(cli_args.get("input_folder") or ""), relative_path_base),
                "selectedInputFolder": selected_input_folder,
                "sourceInputFolder": input_source_kind,
            }

        dir_validation_error = validate_docker_dir_params(
            script_name=script_name,
            script_info=script_info,
            cli_args=cli_args,
            docker_container=docker_kwargs["docker_container"],
            docker_working_dir=docker_kwargs.get("docker_working_dir"),
        )
        if dir_validation_error:
            return dir_validation_error

        if script_info.get("cli_args_only", False) and script_name == "data_preprocessing":
            relative_path_base = script_info.get("container_relative_path_base") or docker_kwargs.get("docker_working_dir")
            input_container_path = _resolve_container_path(str(cli_args.get("input_folder") or ""), relative_path_base)
            detected_format = detect_data_preprocessing_dir_format_in_container(
                str(docker_kwargs["docker_container"]),
                input_container_path,
            )
            preprocessing_diagnostics.update(
                {
                    "inputFolder": input_container_path,
                    "detectedFormat": detected_format,
                }
            )
            if detected_format == "unknown":
                if preprocessing_diagnostics.get("sourceInputFolder") == "default_latest":
                    message = (
                        "已检查默认输入目录 /home/workspace/dataset 下最新数据集，"
                        f"但无法识别其数据格式：{input_container_path}。"
                        "请指定一个包含 OpenAI、ShareGPT、SFT、DPO、text 或医疗 raw 格式数据的 input_folder。"
                    )
                else:
                    message = (
                        f"无法识别输入目录的数据格式：{input_container_path}。"
                        "请改用包含 OpenAI、ShareGPT、SFT、DPO、text 或医疗 raw 格式数据的 input_folder。"
                    )
                current_args = {
                    "additional_args": additional_args or {},
                    "cli_params_to_update": cli_params_to_update,
                    "params_to_update": params_to_update,
                    "positional_args": positional_args,
                    **preprocessing_diagnostics,
                }
                return _data_error_response(
                    script_name,
                    message,
                    error_reason="unknown_data_format",
                    required_params=["input_folder"],
                    missing_params=["input_folder"],
                    recoverable=True,
                    currentArgs=current_args,
                    container=docker_kwargs["docker_container"],
                    containerPath=input_container_path,
                    **preprocessing_diagnostics,
                )
            required_params = effective_data_preprocessing_required_params(
                script_name,
                script_info,
                cli_args,
                detected_format,
            )
            all_params_provided, missing_params, params_to_ask = check_required_params(
                script_info,
                cli_args,
                required_params,
            )
            if not all_params_provided and not skip_prompt:
                return missing_required_response(required_params, missing_params, params_to_ask, preprocessing_diagnostics)
            invalid_params = validate_cli_param_values(script_info, cli_args, required_params)
            if invalid_params or (not all_params_provided and skip_prompt):
                return invalid_required_response(required_params, missing_params, invalid_params, all_params_provided)
            cleanup_data_preprocessing_cli_args(script_info, cli_args, args, detected_format)
            run_kwargs["script_args"] = args

        # 合并到run_kwargs中
        for k, v in docker_kwargs.items():
            if v is not None:
                run_kwargs[k] = v

        # 对于Docker执行，需要特殊处理后台运行
        if background and script_info.get("supports_background"):
            # 对于Docker后台运行，使用专门的Docker后台执行函数
            log_file = script_info.get("default_args", {}).get("log_file")
            if log_file:
                run_kwargs["log_file"] = log_file
            run_kwargs["script_name"] = script_name
            if filter_output_folder:
                run_kwargs["protocol_hint_extra"] = {
                    "outputFolder": filter_output_folder,
                    "inputFolder": str(cli_args.get("input_folder") or "").rstrip("/"),
                    "threshold": float(cli_args.get("threshold") or 90),
                    "outputDatasetName": filter_output_dataset_name,
                }
            return run_docker_in_background(**run_kwargs)
        else:
            # 普通Docker执行
            return run_local_script(**run_kwargs)

    # 根据是否后台运行选择不同的执行方式
    if background:
        # 对于本地脚本的后台运行，需要移除Docker相关参数
        local_kwargs = {k: v for k, v in run_kwargs.items()
                       if not k.startswith('docker_') and k not in ['docker_container', 'docker_image']}
        return run_script_in_background(**local_kwargs)
    else:
        return run_local_script(**run_kwargs)


def process_parameter_response(script_query: str, user_response: str, previous_args: Dict[str, Any] = None) -> ToolResponse:
    """
    处理用户对必需参数的响应

    Args:
        script_query: 脚本查询名称
        user_response: 用户提供的参数响应
        previous_args: 之前已经收集的参数

    Returns:
        ToolResponse: 执行结果
    """
    # 解析用户提供的参数
    parsed_args = {}
    lines = user_response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or '###' in line:
            continue

        # 尝试解析 "参数名: 参数值" 格式
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                # 去除可能的注释
                if '#' in value:
                    value = value.split('#')[0].strip()
                parsed_args[key] = value

    # 如果没有解析到参数，尝试其他格式
    if not parsed_args and '=' in user_response:
        for line in lines:
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    parsed_args[key] = value

    # 合并之前的参数
    final_args = previous_args or {}
    if parsed_args:
        # 合并参数
        for key, value in parsed_args.items():
            final_args[key] = value

    # 重新运行脚本，这次跳过询问
    return run_script_by_name_data(
        script_query=script_query,
        additional_args=final_args,
        skip_prompt=True  # 跳过询问，直接执行
    )


def validate_docker_container(
    container_name: str,
    check_script_path: str = None,
    working_dir: str = None
) -> Dict[str, Any]:
    """
    验证Docker容器状态和路径
    """
    result = {
        "container_exists": False,
        "container_running": False,
        "script_exists": False,
        "errors": []
    }

    try:
        # 检查容器是否存在
        check_cmd = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        process = subprocess.run(check_cmd, capture_output=True, text=True)

        if process.returncode != 0:
            result["errors"].append(f"Docker命令失败: {process.stderr}")
            return result

        containers = [c.strip() for c in process.stdout.split('\n') if c.strip()]

        if container_name not in containers:
            result["errors"].append(f"容器 {container_name} 不存在")
            return result

        result["container_exists"] = True

        # 检查容器是否正在运行
        check_cmd = ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        process = subprocess.run(check_cmd, capture_output=True, text=True)

        if process.returncode == 0:
            running_containers = [c.strip() for c in process.stdout.split('\n') if c.strip()]
            result["container_running"] = container_name in running_containers

        # 检查脚本路径
        if check_script_path:
            # 构建容器内的绝对路径
            if working_dir and not os.path.isabs(check_script_path):
                abs_path = os.path.join(working_dir, check_script_path)
            else:
                abs_path = check_script_path

            # 检查文件是否存在
            check_cmd = ["docker", "exec", container_name, "test", "-f", abs_path]
            process = subprocess.run(check_cmd, capture_output=True, text=True)

            if process.returncode == 0:
                result["script_exists"] = True
                result["script_path"] = abs_path
            else:
                # 尝试查找文件
                find_cmd = ["docker", "exec", container_name, "find", working_dir or "/",
                          "-name", os.path.basename(check_script_path), "-type", "f"]
                process = subprocess.run(find_cmd, capture_output=True, text=True, timeout=5)

                if process.returncode == 0 and process.stdout.strip():
                    found_path = process.stdout.strip().split('\n')[0]
                    result["script_exists"] = True
                    result["script_path"] = found_path
                    result["warnings"] = [f"使用找到的路径: {found_path}"]
                else:
                    result["errors"].append(f"在容器中未找到脚本: {check_script_path}")

    except subprocess.TimeoutExpired:
        result["errors"].append("Docker命令执行超时")
    except Exception as e:
        result["errors"].append(f"Docker验证异常: {str(e)}")

    return result


def run_script_in_docker(
    script_path: str,
    script_args: Dict[str, Any] = None,
    positional_args: List[str] = None,  # 新增：位置参数
    env_vars: Dict[str, str] = None,
    timeout: int = None,
    capture_output: bool = True,
    working_dir: str = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    validate_container: bool = True,
    **kwargs
) -> ToolResponse:
    """
    在Docker容器中运行Python脚本，支持通过环境变量传递参数
    """

    try:
        # 验证容器状态
        if validate_container and docker_container:
            validation = validate_docker_container(
                container_name=docker_container,
                check_script_path=script_path,
                working_dir=docker_working_dir
            )

            if validation.get("errors"):
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"Docker容器验证失败:\n" + "\n".join(validation["errors"])
                    )
                ])

            # 更新脚本路径为找到的路径
            if validation.get("script_path"):
                script_path = validation["script_path"]

            if not validation.get("container_running"):
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"容器 {docker_container} 存在但未运行，请先启动容器"
                    )
                ])

        # 构建Docker命令
        if docker_container:
            # 在运行中的容器内执行
            command = ["docker", "exec"]
        elif docker_image:
            # 启动新的容器执行
            command = ["docker", "run", "--rm"]
        else:
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text="错误: 需要指定docker_container或docker_image"
                )
            ])

        # 添加环境变量
        if env_vars:
            for key, value in env_vars.items():
                if value:  # 只添加非空值
                    command.extend(["-e", f"{key}={value}"])

        # 添加卷挂载
        if docker_volume_mounts:
            for host_path, container_path in docker_volume_mounts.items():
                if os.path.exists(host_path):
                    command.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}"])

        # 设置工作目录
        if docker_working_dir:
            command.extend(["-w", docker_working_dir])

        # 指定容器/镜像
        if docker_container:
            command.append(docker_container)
        elif docker_image:
            command.append(docker_image)
        #if docker_working_dir:
            #shell_command_parts.append(f"cd {docker_working_dir}")
        # 添加执行命令
        #command.append(docker_executable)
        #command.append(script_path)
        if docker_executable and script_path and (
            docker_executable == script_path or script_path in docker_executable
        ):
            # 如果两者相同，只保留一个
            command.append(docker_executable)
        else:
            # 添加Python命令和脚本
            if docker_executable:
                command.append(docker_executable)
            command.append(script_path)  # 这里使用容器内的路径


        if positional_args:
            for arg in positional_args:
                command.append(str(arg))

        # 添加脚本参数
        if script_args:
            for key, value in script_args.items():
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                    continue
                else:
                    command.extend([f"--{key}", str(value)])

        # 打印调试信息
        logger.info(f"Docker命令: {' '.join(command)}")

        # 执行Docker命令
        if timeout:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
        else:
            process = subprocess.run(
                command,
                capture_output=capture_output,
                text=True
            )

        # 构建结果消息
        result_text = []

        if process.stdout and capture_output:
            result_text.append(f"标准输出:\n{process.stdout}")

        if process.stderr and capture_output:
            result_text.append(f"标准错误:\n{process.stderr}")

        result_text.append(f"返回码: {process.returncode}")
        result_text.append(f"######请检查对应的log文件######")

        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="\n\n".join(result_text)
            )
        ])

    except subprocess.TimeoutExpired:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: Docker脚本执行超时 (超过 {timeout} 秒)"
            )
        ])

    except FileNotFoundError:
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text="错误: Docker命令未找到，请确保Docker已安装并可用"
            )
        ])

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: Docker执行脚本时发生异常: {str(e)}\n\n详细错误:\n{error_trace}"
            )
        ])


def run_docker_in_background(
    script_path: str,
    script_args: Dict[str, Any] = None,
    positional_args: List[str] = None,  # 新增：位置参数
    env_vars: Dict[str, str] = None,
    docker_container: str = None,
    docker_image: str = None,
    docker_volume_mounts: Dict[str, str] = None,
    docker_working_dir: str = None,
    docker_executable: str = "python",
    log_file: str = None,
    pid_file: str = None,
    check_startup: bool = True,  # 新增：是否检查启动状态
    startup_check_delay: int = 5,  # 新增：启动检查延迟时间（秒）
    script_name: str = None,
    protocol_hint_extra: Dict[str, Any] = None,
    **kwargs
) -> ToolResponse:
    """
    在Docker容器中后台运行脚本，支持通过环境变量传递参数
    """
    try:
        # 如果没有指定容器，尝试从环境变量中获取
        if not docker_container and env_vars and "container" in env_vars:
            docker_container = env_vars.get("container")
            # 从环境变量中移除container，避免传递给脚本
            env_vars_copy = env_vars.copy()
            env_vars_copy.pop("container", None)
            env_vars = env_vars_copy

        if not docker_container and not docker_image:
            return _data_error_response(
                script_name or script_path,
                "错误: 需要指定docker_container或docker_image",
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )

        # 构建Docker后台运行命令
        if docker_container:
            # 在运行中的容器内执行
            command = ["docker", "exec"]
        elif docker_image:
            # 启动新的容器并后台运行
            command = ["docker", "run", "-d", "--rm"]
        else:
            return _data_error_response(
                script_name or script_path,
                "错误: 需要指定docker_container或docker_image",
                error_reason="container_required",
                required_params=["container"],
                recoverable=True,
            )

        # 添加环境变量
        if env_vars:
            for key, value in env_vars.items():
                if value:  # 只添加非空值
                    command.extend(["-e", f"{key}={value}"])

        # 添加卷挂载
        if docker_volume_mounts:
            for host_path, container_path in docker_volume_mounts.items():
                if os.path.exists(host_path):
                    command.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}"])

        # 指定容器/镜像
        if docker_container:
            command.append(docker_container)

            # 构建要在容器内执行的shell命令
            shell_command_parts = []

            # 如果有工作目录，先cd到该目录
            if docker_working_dir:
                shell_command_parts.append(f"cd {docker_working_dir}")

            # 获取脚本的基本命令
            #if docker_executable == script_path:
            if docker_executable and script_path and script_path in docker_executable:
                # 如果可执行文件和脚本路径相同，说明脚本本身是可执行的
                base_cmd = docker_executable
            else:
                # 否则使用可执行文件运行脚本
                base_cmd = f"{docker_executable} {script_path}"
            ##假设全是ins相关可执行代码
            #base_cmd = docker_executable

            # 初始化命令部分列表
            cmd_parts = [base_cmd]
            # 添加位置参数（格式化为 {时间} 格式）
            #final_cmd = base_cmd
            if positional_args:
                for arg in positional_args:
                    cmd_parts.append(str(arg))
                    ## 格式化时间参数为 {HH:MM:SS} 格式
                    #formatted_arg = arg
                    #if ":" in arg and arg.count(":") == 2:
                        # 如果参数是时间格式，确保用花括号包裹
                     #   if not arg.startswith("{") and not arg.endswith("}"):
                      #      formatted_arg = f"{{{arg}}}"
                    #cmd_parts.append(formatted_arg)

            # 添加脚本参数
            if script_args:
                for key, value in script_args.items():
                    if isinstance(value, bool):
                        if value:
                            cmd_parts.append(f"--{key}")
                    elif isinstance(value, (int, float, str)):
                        cmd_parts.append(f"--{key}")
                        cmd_parts.append(str(value))
                        # 对于schedule_time参数，格式化为花括号
                        #if key == "schedule_time" and ":" in str(value) and str(value).count(":") == 2:
                        #    if not str(value).startswith("{") and not str(value).endswith("}"):
                        #        formatted_value = f"{{{value}}}"
                        #        cmd_parts.append(f"--{key}")
                         #       cmd_parts.append(formatted_value)
                         #   else:
                         #       cmd_parts.append(f"--{key}")
                         #       cmd_parts.append(str(value))
                        #else:
                           # cmd_parts.append(f"--{key}")
                          #  cmd_parts.append(str(value))
                    elif isinstance(value, list):
                        for item in value:
                            cmd_parts.append(f"--{key}")
                            cmd_parts.append(str(item))
                    elif value is None:
                        continue
                    else:
                        cmd_parts.append(f"--{key}")
                        cmd_parts.append(str(value))

            # 将命令部分合并为完整的命令字符串
            python_cmd = " ".join(cmd_parts)
            shell_command_parts.append(python_cmd)

            # 将所有命令用 && 连接
            shell_command = " && ".join(shell_command_parts)

            # 通过 /bin/bash -c 执行shell命令
            command.extend(["/bin/bash", "-c", shell_command])

        elif docker_image:
            command.append(docker_image)
            # 对于镜像，直接添加可执行文件和参数
            #if docker_executable == script_path:
            if docker_executable and script_path and script_path in docker_executable:
                # 如果可执行文件和脚本路径相同，只添加脚本路径
                command.append(docker_executable)
            else:
                command.append(docker_executable)
                command.append(script_path)
            # 添加位置参数
            if positional_args:
                for arg in positional_args:
                    # 格式化时间参数
                    formatted_arg = arg
                    if ":" in arg and arg.count(":") == 2:
                        if not arg.startswith("{") and not arg.endswith("}"):
                            formatted_arg = f"{{{arg}}}"
                    command.append(formatted_arg)
            # 添加脚本参数
            if script_args and not positional_args:
                for key, value in script_args.items():
                    if isinstance(value, bool):
                        if value:
                            command.extend([f"--{key}"])
                    elif isinstance(value, (int, float, str)):
                        # 格式化schedule_time参数
                        if key == "schedule_time" and ":" in str(value) and str(value).count(":") == 2:
                            if not str(value).startswith("{") and not str(value).endswith("}"):
                                formatted_value = f"{{{value}}}"
                                command.extend([f"--{key}", formatted_value])
                            else:
                                command.extend([f"--{key}", str(value)])
                        else:
                            command.extend([f"--{key}", str(value)])
                    elif isinstance(value, list):
                        for item in value:
                            command.extend([f"--{key}", str(item)])
                    elif value is None:
                        continue
                    else:
                        command.extend([f"--{key}", str(value)])

        # 打印调试信息
        logger.info(f"Docker后台命令: {' '.join(command)}")
        _record_background_task(
            task_type="data",
            container=docker_container,
            script_name=script_name,
            script_path=script_path,
            command=command,
            script_args=script_args,
            env_vars=env_vars,
            status="starting",
        )

        def run_docker_process():
            try:
                # 执行Docker命令
                logger.info(f"执行命令: {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # 等待一段时间以获取初始输出
                time.sleep(20)

                # 检查进程是否仍在运行
                returncode = process.poll()
                # 获取输出
                stdout, stderr = "", ""
                if returncode is not None:
                    # 如果进程已结束，获取所有输出
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except:
                        pass
                else:
                    # 如果进程仍在运行，只获取部分输出
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        # 进程仍在运行，这是正常的
                        stdout = ""
                        stderr = ""

                # 获取容器ID或进程信息
                container_id = stdout.strip() if docker_image else docker_container

                process_info = {
                    "container_id": container_id,
                    "pid": process.pid,
                    "command": " ".join(command),
                    "log_file": log_file,
                    "pid_file": pid_file,
                    "start_time": time.time(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode or 0  # 如果进程仍在运行，returncode为None，设为0
                }


                # 保存PID到文件
                if pid_file:
                    pid_dir = os.path.dirname(pid_file)
                    if pid_dir and not os.path.exists(pid_dir):
                        os.makedirs(pid_dir, exist_ok=True)
                    with open(pid_file, 'w') as f:
                        f.write(str(container_id))

                if returncode in (None, 0):
                    _record_background_task(
                        task_type="data",
                        container=container_id,
                        script_name=script_name,
                        script_path=script_path,
                        command=command,
                        script_args=script_args,
                        env_vars=env_vars,
                    )

                return process_info
            except Exception as e:
                import traceback
                return {"error": str(e), "traceback": traceback.format_exc()}

        # 直接运行Docker进程，不使用线程
        process_info = run_docker_process()

        # 检查进程是否启动失败
        if check_startup and startup_check_delay > 0:
            # 如果process_info包含错误，立即返回
            if "error" in process_info:
                response_text = f"启动Docker进程失败: {process_info['error']}\n\n详细错误:\n{process_info.get('traceback', '无')}"
                return _data_error_response(
                    script_name or script_path,
                    response_text,
                    error_reason="docker_process_failed",
                    container=docker_container or docker_image,
                    traceback=process_info.get("traceback"),
                )

            returncode = process_info.get("returncode", 0)
            stdout = process_info.get("stdout", "")
            stderr = process_info.get("stderr", "")

            # 检查进程是否立即退出（返回码非0）
            if returncode != 0 and returncode is not None:
                error_msg = f"脚本在Docker容器中启动失败，返回码: {returncode}\n"
                # 如果有标准输出，也显示一部分
                if stdout:
                    error_msg += f"标准输出:\n{stdout}\n"
                # 如果有错误输出，添加到消息中
                if stderr:
                    error_msg += f"错误信息:\n{stderr}\n"
                error_msg += f"######请检查对应的log文件######"


                # 建议检查日志
                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            logs = f.readlines()[-20:]  # 读取最后20行
                            if logs:
                                error_msg += f"日志文件最后20行:\n{''.join(logs)}\n"
                    except:
                        error_msg += f"无法读取日志文件: {log_file}\n"

                return _data_error_response(
                    script_name or script_path,
                    error_msg,
                    error_reason="docker_start_failed",
                    container=docker_container or docker_image,
                    returnCode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
        response_text = (
            f"脚本已在Docker容器中后台启动运行\n"
            f"脚本: {os.path.basename(script_path)}\n"
            f"容器: {docker_container or docker_image}\n"
            f"请稍后在左侧数据集管理列表中查看"
            #f"日志文件: {log_file or '无'}\n"
            #f"PID文件: {pid_file or '无'}\n"

        )
        return ToolResponse(content=[TextBlock(type="text", text=response_text)],
            metadata={
                "success": True,
                "response_text": response_text,
                "protocol_hint": _data_protocol_hint(
                    "job_started",
                    script_name,
                    response_text,
                    container=docker_container or docker_image,
                    command=" ".join(command),
                    scriptArgs=script_args or {},
                    envKeys=sorted((env_vars or {}).keys()),
                    **(protocol_hint_extra or {}),
                ),
            },
        )

    except Exception as e:
        import traceback
        response_text = f"错误: 无法在Docker容器中后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return _data_error_response(
            script_name or script_path,
            response_text,
            error_reason="docker_background_exception",
            container=docker_container or docker_image,
        )

def run_script_in_background(
    script_path: str,
    script_args: Dict[str, Any] = None,
    env_vars: Dict[str, str] = None,
    log_file: str = None,
    pid_file: str = None,
    check_startup: bool = True,  # 新增：是否检查启动状态
    startup_check_delay: int = 3,  # 新增：启动检查延迟时间（秒）
    **kwargs
) -> ToolResponse:
    """
    在后台运行本地脚本
    """
    try:
        # 检查脚本文件是否存在
        abs_script_path = os.path.abspath(script_path)
        if not os.path.exists(abs_script_path):
            return _data_error_response(
                os.path.basename(script_path),
                f"错误: 找不到脚本文件: {script_path}",
                error_reason="script_file_not_found",
                recoverable=True,
                scriptPath=script_path,
            )

        # 构建Python执行命令
        python_executable = sys.executable
        command = [python_executable, abs_script_path]

        # 添加命令行参数
        if script_args:
            for key, value in script_args.items():
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{key}"])
                elif isinstance(value, (int, float, str)):
                    command.extend([f"--{key}", str(value)])
                elif isinstance(value, list):
                    for item in value:
                        command.extend([f"--{key}", str(item)])
                elif value is None:
                        continue
                else:
                    command.extend([f"--{key}", str(value)])

        # 设置环境变量
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # 设置日志文件
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # 重定向输出到日志文件
            with open(log_file, 'a') as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=log_handle,
                    env=env,
                    cwd=os.path.dirname(abs_script_path)
                )
        else:
            # 使用空设备，丢弃输出
            with open(os.devnull, 'w') as devnull:
                process = subprocess.Popen(
                    command,
                    stdout=devnull,
                    stderr=devnull,
                    env=env,
                    cwd=os.path.dirname(abs_script_path)
                )

        # 保存PID到文件
        if pid_file:
            pid_dir = os.path.dirname(pid_file)
            if pid_dir and not os.path.exists(pid_dir):
                os.makedirs(pid_dir, exist_ok=True)
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
        # 检查进程是否启动失败
        if check_startup and startup_check_delay > 0:
            # 等待指定的延迟时间
            time.sleep(startup_check_delay)

            # 检查进程是否还在运行
            returncode = process.poll()  # 如果进程已结束，返回退出码，否则返回None

            if returncode is not None and returncode != 0:
                error_msg = f"脚本后台启动失败，返回码: {returncode}\n"

                # 尝试从日志文件读取错误信息
                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            logs = f.readlines()[-20:]  # 读取最后20行
                            if logs:
                                error_msg += f"日志文件最后20行:\n{''.join(logs)}\n"
                    except:
                        error_msg += f"无法读取日志文件: {log_file}\n"
                else:
                    error_msg += "未指定日志文件，无法获取详细错误信息\n"

                # 删除PID文件（因为进程已结束）
                if pid_file and os.path.exists(pid_file):
                    try:
                        os.remove(pid_file)
                    except:
                        pass

                return _data_error_response(
                    os.path.basename(script_path),
                    error_msg,
                    error_reason="local_start_failed",
                    returnCode=returncode,
                    logFile=log_file,
                    pidFile=pid_file,
                )
        response_text = (
            f"脚本已在后台启动运行\n"
            f"PID: {process.pid}\n"
            f"脚本: {os.path.basename(script_path)}\n"
        )
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=response_text
            )
        ], metadata={
            "success": True,
            "protocol_hint": _data_protocol_hint(
                "job_started",
                os.path.basename(script_path),
                response_text,
                pid=str(process.pid),
                script=os.path.basename(script_path),
                command=" ".join(command),
                scriptArgs=script_args or {},
                logFile=log_file,
                pidFile=pid_file,
            ),
        })

    except Exception as e:
        import traceback
        response_text = f"错误: 无法在后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return _data_error_response(
            os.path.basename(script_path),
            response_text,
            error_reason="local_background_exception",
            scriptPath=script_path,
        )


# 为了兼容性，添加一个别名
test_script = run_local_script
