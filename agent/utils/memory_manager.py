"""
内存管理模块
实现 LRU + 两级缓存策略：
- 活跃态（Active）：完整会话在内存，包含所有 agents
- 休眠态（Hibernate）：保留上下文，释放 agents，可快速恢复
- 持久态（Persistent）：序列化到磁盘，下次访问时恢复
"""

import asyncio
import pickle
import psutil
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set, Any, Union, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field

from utils.config import get_current_config

logger = logging.getLogger(__name__)


class _AsyncThreadLock:
    """Async context wrapper for state shared across request worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self):
        await asyncio.to_thread(self._lock.acquire)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()
        return False


def _safe_path_part(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._") or "default"


def _split_session_key(user_id: str) -> Tuple[str, str]:
    raw_key = (user_id or "").strip().strip("[]")
    if "#" in raw_key:
        base_user_id, session_id = raw_key.split("#", 1)
    else:
        base_user_id, session_id = raw_key, "default"
    return _safe_path_part(base_user_id), _safe_path_part(session_id)


@dataclass
class SessionState:
    """会话状态"""
    user_id: str
    last_active: datetime
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    # 会话上下文（可序列化）
    task_context: Dict[str, Any] = field(default_factory=dict)
    task_history: list = field(default_factory=list)
    pending_parameters: Dict[str, Any] = field(default_factory=dict)
    current_task_state: Dict[str, Any] = field(default_factory=dict)
    last_completed_task: Optional[Dict[str, Any]] = None
    last_response_protocol: Optional[Dict[str, Any]] = None


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        self.config = get_current_config().system.memory_management
        self.persistence_config = get_current_config().system.persistence
        
        # 活跃会话：完整会话对象
        self.active_sessions: OrderedDict[str, Any] = OrderedDict()
        
        # 休眠会话：仅保留状态，无 agents
        self.hibernate_sessions: OrderedDict[str, SessionState] = OrderedDict()
        
        # 会话状态（所有会话的状态信息）
        self.session_states: Dict[str, SessionState] = {}
        
        # 锁：请求现在会在多个 worker 线程的独立事件循环中执行，
        # 这里使用线程级锁，避免 asyncio.Lock 跨 event loop 复用。
        self._lock = _AsyncThreadLock()
        self._lifecycle_lock = threading.Lock()
        
        # 序列化路径
        self.persistence_path = Path(self.persistence_config.serialization_path)
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # 启动后台任务
        self._cleanup_task = None
        self._monitor_task = None
        self._running = False
        
        logger.info(f"MemoryManager initialized with strategy: {self.config.strategy}")
    
    async def start(self):
        """启动内存管理后台任务"""
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("MemoryManager background tasks started")
    
    async def stop(self):
        """停止内存管理后台任务"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # 持久化所有会话
        await self._persist_all_sessions()
        logger.info("MemoryManager stopped")
    
    async def register_session(self, user_id: str, session: Any, state: SessionState):
        """注册新会话"""
        async with self._lock:
            # 检查是否需要回收内存
            await self._ensure_memory_available()
            
            # 添加到活跃会话
            self.active_sessions[user_id] = session
            self.active_sessions.move_to_end(user_id)  # 更新LRU
            
            # 保存状态
            self.session_states[user_id] = state
            
            logger.debug(f"Session {user_id} registered as active")
    
    async def get_session(self, user_id: str) -> Optional[Any]:
        """获取会话"""
        async with self._lock:
            # 1. 检查活跃会话
            if user_id in self.active_sessions:
                self.active_sessions.move_to_end(user_id)  # 更新LRU
                state = self.session_states[user_id]
                state.last_active = datetime.now()
                state.access_count += 1
                return self.active_sessions[user_id]
            
            # 2. 检查休眠会话
            if user_id in self.hibernate_sessions:
                logger.info(f"Restoring session {user_id} from hibernate")
                return await self._restore_from_hibernate(user_id)
            
            # 3. 检查持久化存储
            persisted_state = await self._load_from_disk(user_id)
            if persisted_state:
                logger.info(f"Restoring session {user_id} from disk")
                self.session_states[user_id] = persisted_state
                return None  # 返回None，让上层重建agents
            
            return None
    
    async def update_session_state(self, user_id: str, **kwargs):
        """更新会话状态"""
        async with self._lock:
            if user_id in self.session_states:
                state = self.session_states[user_id]
                for key, value in kwargs.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
    
    async def get_session_state(self, user_id: str) -> Optional[SessionState]:
        """获取会话状态"""
        async with self._lock:
            return self.session_states.get(user_id)
    
    async def remove_session(self, user_id: str):
        """移除会话"""
        async with self._lock:
            # 从活跃会话移除
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            
            # 从休眠会话移除
            if user_id in self.hibernate_sessions:
                del self.hibernate_sessions[user_id]
            
            # 从状态移除
            if user_id in self.session_states:
                del self.session_states[user_id]
            
            # 删除持久化文件
            await self._delete_persisted_file(user_id)
            
            logger.debug(f"Session {user_id} removed")
    
    async def _ensure_memory_available(self):
        """确保有足够的内存空间"""
        # 检查活跃会话数
        while len(self.active_sessions) >= self.config.max_active_sessions:
            # 将最久未使用的会话移入休眠
            await self._hibernate_oldest_session()
        
        # 检查休眠会话数
        while len(self.hibernate_sessions) >= self.config.max_hibernate_sessions:
            # 将最久未使用的休眠会话移入持久态
            await self._persist_oldest_hibernate_session()
    
    async def _hibernate_oldest_session(self):
        """将最老的活跃会话移入休眠"""
        if not self.active_sessions:
            return
        
        # 获取最老的会话
        user_id, session = self.active_sessions.popitem(last=False)
        state = self.session_states[user_id]
        
        # 保存状态
        state.last_active = datetime.now()
        
        # 提取可序列化的状态（需要由会话对象提供）
        if hasattr(session, 'get_serializable_state'):
            serializable_state = session.get_serializable_state()
            for key, value in serializable_state.items():
                if hasattr(state, key):
                    setattr(state, key, value)
        
        # 添加到休眠会话
        self.hibernate_sessions[user_id] = state
        self.hibernate_sessions.move_to_end(user_id)
        
        logger.info(f"Session {user_id} moved to hibernate state")
    
    async def _restore_from_hibernate(self, user_id: str) -> Optional[Any]:
        """从休眠状态恢复会话"""
        # 确保内存空间
        await self._ensure_memory_available()
        
        # 获取状态
        state = self.hibernate_sessions.pop(user_id)
        
        # 更新状态
        state.last_active = datetime.now()
        state.access_count += 1
        
        # 重新添加到活跃会话（会话对象需要由上层重建）
        self.session_states[user_id] = state
        
        logger.info(f"Session {user_id} restored from hibernate")
        return None  # 返回None，让上层重建agents
    
    async def _persist_oldest_hibernate_session(self):
        """将最老的休眠会话持久化到磁盘"""
        if not self.hibernate_sessions:
            return
        
        # 获取最老的休眠会话
        user_id, state = self.hibernate_sessions.popitem(last=False)
        
        # 保存到磁盘
        await self._save_to_disk(user_id, state)
        
        # 从状态字典移除
        if user_id in self.session_states:
            del self.session_states[user_id]
        
        logger.info(f"Session {user_id} persisted to disk")
    
    async def _save_to_disk(self, user_id: str, state: SessionState):
        """保存会话状态到磁盘"""
        try:
            base_user_id, session_id = _split_session_key(user_id)
            user_dir = self.persistence_path / base_user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / f"{session_id}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to persist session {user_id}: {e}")
    
    async def _load_from_disk(self, user_id: str) -> Optional[SessionState]:
        """从磁盘加载会话状态"""
        try:
            base_user_id, session_id = _split_session_key(user_id)
            file_path = self.persistence_path / base_user_id / f"{session_id}.pkl"
            legacy_file_path = self.persistence_path / f"{user_id}.pkl"
            if not file_path.exists() and legacy_file_path.exists():
                file_path = legacy_file_path
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            # 加载后删除文件
            file_path.unlink()
            
            return state
        except Exception as e:
            logger.error(f"Failed to load session {user_id} from disk: {e}")
            return None
    
    async def _delete_persisted_file(self, user_id: str):
        """删除持久化文件"""
        try:
            base_user_id, session_id = _split_session_key(user_id)
            for file_path in (
                self.persistence_path / base_user_id / f"{session_id}.pkl",
                self.persistence_path / f"{user_id}.pkl",
            ):
                if file_path.exists():
                    file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete persisted file for {user_id}: {e}")
    
    async def _persist_all_sessions(self):
        """持久化所有会话（用于系统关闭）"""
        if not self.persistence_config.enabled:
            return
        
        # 持久化所有休眠会话
        for user_id, state in list(self.hibernate_sessions.items()):
            await self._save_to_disk(user_id, state)
        
        # 持久化所有活跃会话的状态
        for user_id, session in self.active_sessions.items():
            state = self.session_states[user_id]
            if hasattr(session, 'get_serializable_state'):
                serializable_state = session.get_serializable_state()
                for key, value in serializable_state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
            await self._save_to_disk(user_id, state)
        
        logger.info("All sessions persisted")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.active_timeout)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        async with self._lock:
            now = datetime.now()
            timeout = timedelta(seconds=self.config.active_timeout)
            
            # 检查活跃会话
            to_hibernate = []
            for user_id, state in self.session_states.items():
                if user_id in self.active_sessions and (now - state.last_active) > timeout:
                    to_hibernate.append(user_id)
            
            for user_id in to_hibernate:
                await self._hibernate_oldest_session()
                logger.debug(f"Session {user_id} auto-hibernated due to inactivity")
    
    async def _monitor_loop(self):
        """内存监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval)
                await self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    async def _check_memory_usage(self):
        """检查内存使用情况"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > self.config.memory_threshold:
            logger.warning(f"Memory usage {usage_percent}% exceeds threshold {self.config.memory_threshold}%")
            
            # 强制回收
            async with self._lock:
                # 将一半活跃会话移入休眠
                sessions_to_hibernate = len(self.active_sessions) // 2
                for _ in range(sessions_to_hibernate):
                    await self._hibernate_oldest_session()
                
                # 将所有休眠会话持久化
                while self.hibernate_sessions:
                    await self._persist_oldest_hibernate_session()
            
            logger.info(f"Emergency memory cleanup completed")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取内存统计信息"""
        return {
            "active_sessions": len(self.active_sessions),
            "hibernate_sessions": len(self.hibernate_sessions),
            "total_sessions": len(self.session_states),
            "memory_usage_percent": psutil.virtual_memory().percent
        }


# 全局内存管理器实例
_memory_manager: Optional[MemoryManager] = None
_memory_manager_init_lock = threading.Lock()


async def get_memory_manager() -> MemoryManager:
    """获取内存管理器（单例）"""
    global _memory_manager
    with _memory_manager_init_lock:
        if _memory_manager is None:
            _memory_manager = MemoryManager()
        manager = _memory_manager
    await manager.start()
    return manager
