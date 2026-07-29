"""
配置管理模块
使用 Pydantic 进行配置验证和加载
支持 YAML 配置文件和环境变量覆盖
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from functools import lru_cache


class MemoryManagementConfig(BaseModel):
    """内存管理配置"""
    enabled: bool = True
    strategy: Literal["lru_hybrid", "lru", "pool"] = "lru_hybrid"
    active_timeout: int = 1800  # 活跃超时时间（秒）
    max_active_sessions: int = 50  # 最大活跃会话数
    max_hibernate_sessions: int = 200  # 最大休眠会话数
    memory_threshold: int = 80  # 内存告警阈值（百分比）
    check_interval: int = 300  # 内存检查间隔（秒）


class PersistenceConfig(BaseModel):
    """持久化配置"""
    enabled: bool = True
    storage_type: Literal["local"] = "local"
    serialization_path: str = "./data/sessions"
    auto_save_interval: int = 300  # 自动保存间隔（秒）


class SystemConfig(BaseModel):
    """系统配置"""
    session_timeout: int = 3600
    max_concurrent_sessions: int = 100
    memory_management: MemoryManagementConfig = Field(default_factory=MemoryManagementConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)


class EnvironmentConfig(BaseModel):
    """运行环境配置"""
    default_docker_container: str = ""
    default_evaluate_docker_container: str = ""


class WorkflowConfig(BaseModel):
    """One-click train, evaluate, deploy and benchmark workflow."""
    db_path: str = "./data/workflows/workflows.db"
    poll_interval: int = 30
    auto_start_worker: bool = False
    train_start_grace_seconds: int = 180
    event_lease_seconds: int = 300
    event_lease_renew_seconds: int = 60
    publish_dir: str = "/home/workspace/medical_models"


class ConnectionPoolConfig(BaseModel):
    """连接池配置"""
    max_connections: int = 20
    max_keepalive: int = 10
    timeout: int = 30


class GenerateKwargsConfig(BaseModel):
    """生成参数配置"""
    temperature: float = 0
    top_p: float = 1


class ModelConfig(BaseModel):
    """模型配置"""
    name: str = ""
    api_key: str = ""
    base_url: str = ""
    stream: bool = False
    generate_kwargs: GenerateKwargsConfig = Field(default_factory=GenerateKwargsConfig)
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)


class AgentConfig(BaseModel):
    """单个Agent配置"""
    sys_prompt: str = ""

class InferenceAgent(AgentConfig):
    inference_agent_url: str

class AgentsConfig(BaseModel):
    """所有Agents配置"""
    orchestrator: AgentConfig = Field(default_factory=AgentConfig)
    datacollector: AgentConfig = Field(default_factory=AgentConfig)
    dataprocessor: AgentConfig = Field(default_factory=AgentConfig)
    trainer: AgentConfig = Field(default_factory=AgentConfig)
    evaluator: AgentConfig = Field(default_factory=AgentConfig)
    inference: InferenceAgent = Field(default_factory=InferenceAgent)
    monitor: AgentConfig = Field(default_factory=AgentConfig)
    analysis: AgentConfig = Field(default_factory=AgentConfig)
    manager: AgentConfig = Field(default_factory=AgentConfig)


class StudioConfig(BaseModel):
    """Studio配置"""
    url: str = "http://localhost:3000"


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./data/logs/agent3.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class AppConfig(BaseModel):
    """应用主配置"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    studio: StudioConfig = Field(default_factory=StudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _expand_env_vars(value: Any) -> Any:
    """递归展开环境变量"""
    if isinstance(value, str):
        # 匹配 ${VAR} 或 ${VAR:default} 模式
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env(match):
            env_expr = match.group(1)
            if ':' in env_expr:
                var_name, default_val = env_expr.split(':', 1)
                return os.getenv(var_name, default_val)
            else:
                return os.getenv(env_expr, '')
        
        return re.sub(pattern, replace_env, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_yaml_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 展开环境变量
    config = _expand_env_vars(config)
    
    return config


@lru_cache()
def get_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    获取配置（单例模式）
    支持环境变量覆盖:
    - AGENT3_SESSION_TIMEOUT=3600
    - AGENT3_MODEL_NAME=xxx
    - AGENT3_MODEL_API_KEY=xxx
    """
    # 加载 YAML 配置
    yaml_config = load_yaml_config(config_path)
    
    # 环境变量覆盖
    env_overrides = {}
    
    # 系统配置覆盖
    session_timeout = os.getenv("AGENT3_SESSION_TIMEOUT")
    if session_timeout:
        env_overrides.setdefault("system", {})["session_timeout"] = int(session_timeout)
    
    max_sessions = os.getenv("AGENT3_MAX_CONCURRENT_SESSIONS")
    if max_sessions:
        env_overrides.setdefault("system", {})["max_concurrent_sessions"] = int(max_sessions)

    if os.getenv("AGENT3_DEFAULT_DOCKER_CONTAINER"):
        env_overrides.setdefault("environment", {})["default_docker_container"] = os.getenv("AGENT3_DEFAULT_DOCKER_CONTAINER")

    if os.getenv("AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER"):
        env_overrides.setdefault("environment", {})["default_evaluate_docker_container"] = os.getenv("AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER")

    if os.getenv("AGENT3_WORKFLOW_DB_PATH"):
        env_overrides.setdefault("workflow", {})["db_path"] = os.getenv("AGENT3_WORKFLOW_DB_PATH")

    if os.getenv("AGENT3_WORKFLOW_POLL_INTERVAL"):
        env_overrides.setdefault("workflow", {})["poll_interval"] = int(os.getenv("AGENT3_WORKFLOW_POLL_INTERVAL"))

    if os.getenv("AGENT3_WORKFLOW_AUTO_START_WORKER"):
        env_overrides.setdefault("workflow", {})["auto_start_worker"] = (
            os.getenv("AGENT3_WORKFLOW_AUTO_START_WORKER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    if os.getenv("AGENT3_WORKFLOW_TRAIN_START_GRACE_SECONDS"):
        env_overrides.setdefault("workflow", {})["train_start_grace_seconds"] = int(
            os.getenv("AGENT3_WORKFLOW_TRAIN_START_GRACE_SECONDS")
        )

    if os.getenv("AGENT3_WORKFLOW_EVENT_LEASE_SECONDS"):
        env_overrides.setdefault("workflow", {})["event_lease_seconds"] = int(
            os.getenv("AGENT3_WORKFLOW_EVENT_LEASE_SECONDS")
        )

    if os.getenv("AGENT3_WORKFLOW_EVENT_LEASE_RENEW_SECONDS"):
        env_overrides.setdefault("workflow", {})["event_lease_renew_seconds"] = int(
            os.getenv("AGENT3_WORKFLOW_EVENT_LEASE_RENEW_SECONDS")
        )

    if os.getenv("AGENT3_WORKFLOW_PUBLISH_DIR"):
        env_overrides.setdefault("workflow", {})["publish_dir"] = os.getenv("AGENT3_WORKFLOW_PUBLISH_DIR")
    
    # 模型配置覆盖
    if os.getenv("AGENT3_MODEL_NAME"):
        env_overrides.setdefault("model", {})["name"] = os.getenv("AGENT3_MODEL_NAME")
    
    if os.getenv("AGENT3_MODEL_API_KEY"):
        env_overrides.setdefault("model", {})["api_key"] = os.getenv("AGENT3_MODEL_API_KEY")
    
    if os.getenv("AGENT3_MODEL_BASE_URL"):
        env_overrides.setdefault("model", {})["base_url"] = os.getenv("AGENT3_MODEL_BASE_URL")
    
    # 合并配置
    merged_config = _deep_merge(yaml_config, env_overrides)
    
    # 验证并返回
    return AppConfig(**merged_config)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def reload_config(config_path: str = "config/config.yaml") -> AppConfig:
    """重新加载配置（用于热更新）"""
    get_config.cache_clear()
    return get_config(config_path)


# 全局配置实例
_config: Optional[AppConfig] = None


def init_config(config_path: str = "config/config.yaml") -> AppConfig:
    """初始化配置"""
    global _config
    _config = get_config(config_path)
    return _config


def get_current_config() -> AppConfig:
    """获取当前配置"""
    global _config
    if _config is None:
        _config = init_config()
    return _config
