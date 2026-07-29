# -*- coding: utf-8 -*-
"""
调用本地脚本的工具函数，支持通过名称、别名查找并运行脚本。
相对于另一版本，补充修改参数的能力
加上改docker
加上推理
加上监控
"""

import subprocess
import sys
import json
import os
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable,Tuple
from agentscope.tool import ToolResponse
from ._config_defaults import get_default_docker_container
from agentscope.message import TextBlock

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_CONTAINER = get_default_docker_container()


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
            "data_prepare": {
                "path": "data_prepare.py",
                "description": "数据准备工具，用于收集各类医疗数据",
                "aliases": ["数据准备","准备数据","数据收集"],
                "supports_background": False,
                "long_running": False,
                "default_env": {
                    
                },
                "cli_args_only": True,  # 新增：标记此脚本只使用命令行参数
                "default_cli_args": {    # 新增：默认命令行参数
                    "data_type": "",
                    #"data_dir":"/home/workspace/dataset",
                    #"example_data_dir":"/home/workspace/dataset-example",
                    },
                
                "supported_cli_params": ["data_type"],  # 新增：支持的命令行参数
                "param_mapping": {  # 新增：命令行参数的中文映射
                    "数据类型": "data_type",
                    #"输入路径":"data_dir",
                    #"示例路径":"example_data_dir"
                    },
                "supported_params": [],
                "default_args": {
                    "background": False,
                    "capture_output": False,
                    #"log_file": "/home/workspace/agent.log"
                },
                "docker_container": DEFAULT_DOCKER_CONTAINER,
                "docker_working_dir": "/home/workspace",
                "docker_executable": "/usr/bin/python",
                "requires_docker": True      
            }

            
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


def check_required_params(script_info: Dict[str, Any], cli_args: Dict[str, str]) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    检查必需参数是否已提供
    
    Args:
        script_info: 脚本信息
        cli_args: 当前已提供的命令行参数
        
    Returns:
        Tuple[是否通过检查, 缺失参数列表, 需要询问的参数信息]
    """
    required_params = script_info.get("required_cli_params", [])
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


def run_script_by_name_data_collect(
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
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"未找到脚本: {script_query}\n\n{manager.list_scripts()}"
            )
        ])
    
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
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=f"错误！脚本 '{script_name}' 不支持以下参数: {', '.join(unknown_params)}\n\n"
                         f"该脚本支持的参数:\n" + "\n".join(available_params_text) + "\n\n"
                         f"请使用支持的参数重新指定，然后确认运行。"
                )
            ])
    
    # 准备命令行参数
    cli_args = {}
    if script_info.get("default_cli_args"):
        cli_args = script_info.get("default_cli_args", {}).copy()
    
    # 更新命令行参数
    for cli_param, value in cli_params_to_update.items():
        cli_args[cli_param] = value
    
    # 检查必需参数
    if script_info.get("cli_args_only", False) and not skip_prompt:
        all_params_provided, missing_params, params_to_ask = check_required_params(script_info, cli_args)
        
        if not all_params_provided:
            # 构建询问消息
            prompt_text = f"错误！脚本 '{script_name}' 需要以下必需参数，但未提供：\n\n"
            
            for param_name, param_info in params_to_ask.items():
                prompt_text += f"- {param_info['description']}\n"
            
            
            
            # 序列化当前已收集的参数，以便后续使用
            current_args = {
                "additional_args": additional_args or {},
                "cli_params_to_update": cli_params_to_update,
                "params_to_update": params_to_update,
                "positional_args": positional_args
            }
            prompt_text += f"当前参数: {json.dumps(current_args, ensure_ascii=False)}\n"
            
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=prompt_text
                )
            ])
    
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
        
    # 提取用户指定的容器名称（如果有）
    user_container = None
    if "container" in final_env_vars:
        user_container = final_env_vars.pop("container")  # 从环境变量中移除，因为它不是真正的环境变量
    logger.info(
        "数据采集目标Docker容器: %s",
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
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=f"错误: 需要指定Docker容器才能运行脚本\n"
                         f"脚本: {script_info['description']}\n"
                         f"请通过环境变量指定容器，例如: env_vars={{'container': 'qwen3_zh'}}"
                )
            ])
        
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
    return run_script_by_name_data_collect(
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
        if docker_executable and script_path and docker_executable == script_path:
            # 如果两者相同，只保留一个
            command.append(script_path)  # 这里使用脚本路径作为可执行文件
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
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text="错误: 需要指定docker_container或docker_image"
                )
            ])
        
        # 构建Docker后台运行命令
        if docker_container:
            # 在运行中的容器内执行
            command = ["docker", "exec"]
        elif docker_image:
            # 启动新的容器并后台运行
            command = ["docker", "run", "-d", "--rm"]
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
            if  script_path in docker_executable:
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
            if  script_path in docker_executable:
                # 如果可执行文件和脚本路径相同，只添加脚本路径
                command.append(script_path)
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
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=f"启动Docker进程失败: {process_info['error']}\n\n详细错误:\n{process_info.get('traceback', '无')}"
                    )
                ])
            
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
                
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=error_msg
                    )
                ])
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"脚本已在Docker容器中后台启动运行\n"
                     f"脚本: {os.path.basename(script_path)}\n"
                     #f"容器: {docker_container or docker_image}\n"
                     #f"日志文件: {log_file or '无'}\n"
                     #f"PID文件: {pid_file or '无'}\n"
                     
            )
        ])
    
    except Exception as e:
        import traceback
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 无法在Docker容器中后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            )
        ])

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
            return ToolResponse(content=[
                TextBlock(
                    type="text",
                    text=f"错误: 找不到脚本文件: {script_path}"
                )
            ])
        
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
                
                return ToolResponse(content=[
                    TextBlock(
                        type="text",
                        text=error_msg
                    )
                ])
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"脚本已在后台启动运行\n"
                     f"PID: {process.pid}\n"
                     f"脚本: {os.path.basename(script_path)}\n"
                     #f"日志文件: {log_file or '无'}\n"
                     #f"PID文件: {pid_file or '无'}"
            )
        ])
    
    except Exception as e:
        import traceback
        return ToolResponse(content=[
            TextBlock(
                type="text",
                text=f"错误: 无法在后台运行脚本: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            )
        ])


# 为了兼容性，添加一个别名
test_script = run_local_script
