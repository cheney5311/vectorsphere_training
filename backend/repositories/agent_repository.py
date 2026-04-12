"""智能体仓库层

提供智能体相关数据的持久化存储和访问接口，使用 SQLAlchemy ORM 操作数据库。
支持智能体、会话、消息、记忆、执行记录和工具的 CRUD 操作。
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import or_, desc, func
from sqlalchemy.orm import Session

from backend.modules.database.manager import get_database_manager
from backend.schemas.agent_models import (
    AgentEntity,
    AgentSessionEntity,
    AgentMessageEntity,
    AgentMemoryEntity,
    AgentExecutionEntity,
    AgentToolEntity
)

logger = logging.getLogger(__name__)


def get_agent_repository() -> 'AgentRepository':
    """获取智能体仓库实例

    Returns:
        AgentRepository: 智能体仓库实例
    """
    return AgentRepository()


class AgentRepository:
    """智能体仓库

    提供智能体数据的 CRUD 操作，使用 SQLAlchemy ORM 与数据库交互。
    """

    def __init__(self):
        """初始化智能体仓库"""
        self._db_manager = None

    @property
    def db_manager(self):
        """懒加载数据库管理器"""
        if self._db_manager is None:
            self._db_manager = get_database_manager()
        return self._db_manager

    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self.db_manager.get_session()

    # ============================================================================
    # 智能体 CRUD 操作
    # ============================================================================

    def create_agent(self, agent_data: Dict[str, Any]) -> AgentEntity:
        """创建智能体

        Args:
            agent_data: 智能体数据字典，包含:
                - user_id: 用户ID (必需)
                - name: 智能体名称 (必需)
                - description: 描述 (可选)
                - agent_type: 类型 (可选, 默认 'chat')
                - version: 版本号 (可选, 默认 '1.0.0')
                - config: 配置 (可选)
                - capabilities: 能力列表 (可选)
                - system_prompt: 系统提示词 (可选)
                - model_config: 模型配置 (可选)
                - is_public: 是否公开 (可选)

        Returns:
            AgentEntity: 创建的智能体实体

        Raises:
            Exception: 创建失败时抛出异常
        """
        session = self._get_session()
        try:
            agent = AgentEntity(
                id=uuid.uuid4(),
                user_id=agent_data.get('user_id'),
                name=agent_data.get('name'),
                description=agent_data.get('description'),
                agent_type=agent_data.get('agent_type', 'chat'),
                version=agent_data.get('version', '1.0.0'),
                status='active',
                config=agent_data.get('config', {}),
                capabilities=agent_data.get('capabilities', []),
                system_prompt=agent_data.get('system_prompt'),
                model_config=agent_data.get('model_config', {}),
                is_public=agent_data.get('is_public', False),
                tenant_id=agent_data.get('tenant_id')
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)
            logger.info("Created agent: %s - %s", agent.id, agent.name)
            return agent
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create agent: %s", str(e))
            raise
        finally:
            session.close()

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentEntity]:
        """根据ID获取智能体

        Args:
            agent_id: 智能体ID (UUID字符串)

        Returns:
            AgentEntity: 智能体实体，不存在则返回None
        """
        session = self._get_session()
        try:
            agent = session.query(AgentEntity).filter(
                AgentEntity.id == agent_id
            ).first()
            return agent
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get agent %s: %s", agent_id, str(e))
            return None
        finally:
            session.close()

    def get_agents_by_user(self,
                          user_id: str,
                          status: Optional[str] = None,
                          agent_type: Optional[str] = None,
                          is_public: Optional[bool] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[AgentEntity]:
        """获取用户的智能体列表

        Args:
            user_id: 用户ID
            status: 状态过滤 (active, inactive, deleted)
            agent_type: 类型过滤
            is_public: 是否公开过滤
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[AgentEntity]: 智能体列表
        """
        session = self._get_session()
        try:
            query = session.query(AgentEntity).filter(
                AgentEntity.user_id == user_id
            )

            if status:
                query = query.filter(AgentEntity.status == status)
            if agent_type:
                query = query.filter(AgentEntity.agent_type == agent_type)
            if is_public is not None:
                query = query.filter(AgentEntity.is_public == is_public)

            agents = query.order_by(desc(AgentEntity.updated_at))\
                         .offset(offset)\
                         .limit(limit)\
                         .all()
            return agents
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get agents for user %s: %s", user_id, str(e))
            return []
        finally:
            session.close()

    def get_public_agents(self,
                         agent_type: Optional[str] = None,
                         limit: int = 100,
                         offset: int = 0) -> List[AgentEntity]:
        """获取公开智能体列表

        Args:
            agent_type: 类型过滤
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[AgentEntity]: 公开智能体列表
        """
        session = self._get_session()
        try:
            # pylint: disable=singleton-comparison
            query = session.query(AgentEntity).filter(
                AgentEntity.is_public == True,
                AgentEntity.status == 'active'
            )

            if agent_type:
                query = query.filter(AgentEntity.agent_type == agent_type)

            agents = query.order_by(desc(AgentEntity.execution_count))\
                         .offset(offset)\
                         .limit(limit)\
                         .all()
            return agents
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get public agents: %s", str(e))
            return []
        finally:
            session.close()

    def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[AgentEntity]:
        """更新智能体

        Args:
            agent_id: 智能体ID
            update_data: 要更新的字段字典

        Returns:
            AgentEntity: 更新后的智能体实体
        """
        session = self._get_session()
        try:
            agent = session.query(AgentEntity).filter(
                AgentEntity.id == agent_id
            ).first()

            if not agent:
                return None

            # 更新允许的字段
            allowed_fields = [
                'name', 'description', 'agent_type', 'version', 'status',
                'config', 'capabilities', 'system_prompt', 'model_config',
                'is_public'
            ]

            for field in allowed_fields:
                if field in update_data:
                    setattr(agent, field, update_data[field])

            agent.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(agent)
            logger.info("Updated agent: %s", agent_id)
            return agent
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update agent %s: %s", agent_id, str(e))
            raise
        finally:
            session.close()

    def delete_agent(self, agent_id: str, soft_delete: bool = True) -> bool:
        """删除智能体

        Args:
            agent_id: 智能体ID
            soft_delete: 是否软删除 (默认True)

        Returns:
            bool: 删除是否成功
        """
        session = self._get_session()
        try:
            agent = session.query(AgentEntity).filter(
                AgentEntity.id == agent_id
            ).first()

            if not agent:
                return False

            if soft_delete:
                agent.status = 'deleted'
                agent.updated_at = datetime.utcnow()
            else:
                session.delete(agent)

            session.commit()
            logger.info("Deleted agent: %s (soft=%s)", agent_id, soft_delete)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to delete agent %s: %s", agent_id, str(e))
            return False
        finally:
            session.close()

    def update_agent_statistics(self,
                               agent_id: str,
                               execution_success: bool,
                               response_time_ms: float) -> bool:
        """更新智能体统计信息

        Args:
            agent_id: 智能体ID
            execution_success: 执行是否成功
            response_time_ms: 响应时间(毫秒)

        Returns:
            bool: 更新是否成功
        """
        session = self._get_session()
        try:
            agent = session.query(AgentEntity).filter(
                AgentEntity.id == agent_id
            ).first()

            if not agent:
                return False

            # 更新执行次数
            agent.execution_count = (agent.execution_count or 0) + 1

            # 更新成功率
            if execution_success:
                successful_count = (agent.success_rate or 0) * (agent.execution_count - 1) + 1
                agent.success_rate = successful_count / agent.execution_count
            else:
                successful_count = (agent.success_rate or 0) * (agent.execution_count - 1)
                agent.success_rate = successful_count / agent.execution_count

            # 更新平均响应时间
            prev_avg = agent.avg_response_time or 0
            agent.avg_response_time = (
                prev_avg * (agent.execution_count - 1) + response_time_ms
            ) / agent.execution_count

            session.commit()
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update agent statistics: %s", str(e))
            return False
        finally:
            session.close()

    # ============================================================================
    # 会话管理
    # ============================================================================

    def create_session(self, session_data: Dict[str, Any]) -> AgentSessionEntity:
        """创建会话

        Args:
            session_data: 会话数据字典，包含:
                - agent_id: 智能体ID (必需)
                - user_id: 用户ID (必需)
                - title: 会话标题 (可选)
                - context: 会话上下文 (可选)
                - metadata: 元数据 (可选)

        Returns:
            AgentSessionEntity: 创建的会话实体
        """
        session = self._get_session()
        try:
            agent_session = AgentSessionEntity(
                id=uuid.uuid4(),
                agent_id=session_data.get('agent_id'),
                user_id=session_data.get('user_id'),
                title=session_data.get('title', '新会话'),
                status='active',
                context=session_data.get('context', {}),
                metadata=session_data.get('metadata', {}),
                message_count=0,
                tenant_id=session_data.get('tenant_id')
            )
            session.add(agent_session)
            session.commit()
            session.refresh(agent_session)
            logger.info("Created session: %s", agent_session.id)
            return agent_session
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create session: %s", str(e))
            raise
        finally:
            session.close()

    def get_session_by_id(self, session_id: str) -> Optional[AgentSessionEntity]:
        """根据ID获取会话

        Args:
            session_id: 会话ID

        Returns:
            AgentSessionEntity: 会话实体
        """
        session = self._get_session()
        try:
            agent_session = session.query(AgentSessionEntity).filter(
                AgentSessionEntity.id == session_id
            ).first()
            return agent_session
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get session %s: %s", session_id, str(e))
            return None
        finally:
            session.close()

    def get_user_sessions(self,
                         user_id: str,
                         agent_id: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: int = 50,
                         offset: int = 0) -> List[AgentSessionEntity]:
        """获取用户的会话列表

        Args:
            user_id: 用户ID
            agent_id: 智能体ID过滤 (可选)
            status: 状态过滤 (可选)
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[AgentSessionEntity]: 会话列表
        """
        session = self._get_session()
        try:
            query = session.query(AgentSessionEntity).filter(
                AgentSessionEntity.user_id == user_id
            )

            if agent_id:
                query = query.filter(AgentSessionEntity.agent_id == agent_id)
            if status:
                query = query.filter(AgentSessionEntity.status == status)

            sessions = query.order_by(desc(AgentSessionEntity.last_message_at))\
                           .offset(offset)\
                           .limit(limit)\
                           .all()
            return sessions
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get sessions for user %s: %s", user_id, str(e))
            return []
        finally:
            session.close()

    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> Optional[AgentSessionEntity]:
        """更新会话

        Args:
            session_id: 会话ID
            update_data: 要更新的字段

        Returns:
            AgentSessionEntity: 更新后的会话实体
        """
        session = self._get_session()
        try:
            agent_session = session.query(AgentSessionEntity).filter(
                AgentSessionEntity.id == session_id
            ).first()

            if not agent_session:
                return None

            allowed_fields = ['title', 'status', 'context', 'metadata', 'summary']
            for field in allowed_fields:
                if field in update_data:
                    setattr(agent_session, field, update_data[field])

            agent_session.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(agent_session)
            return agent_session
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update session %s: %s", session_id, str(e))
            raise
        finally:
            session.close()

    def delete_session(self, session_id: str) -> bool:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 删除是否成功
        """
        session = self._get_session()
        try:
            agent_session = session.query(AgentSessionEntity).filter(
                AgentSessionEntity.id == session_id
            ).first()

            if not agent_session:
                return False

            session.delete(agent_session)
            session.commit()
            logger.info("Deleted session: %s", session_id)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to delete session %s: %s", session_id, str(e))
            return False
        finally:
            session.close()

    # ============================================================================
    # 消息管理
    # ============================================================================

    def add_message(self, message_data: Dict[str, Any]) -> AgentMessageEntity:
        """添加消息

        Args:
            message_data: 消息数据字典，包含:
                - session_id: 会话ID (必需)
                - role: 角色 (user, assistant, system, function)
                - content: 内容 (必需)
                - content_type: 内容类型 (可选, 默认 'text')
                - tokens_used: 使用的token数 (可选)
                - latency_ms: 响应延迟 (可选)
                - metadata: 元数据 (可选)
                - tool_calls: 工具调用 (可选)
                - function_call: 函数调用 (可选)
                - parent_message_id: 父消息ID (可选)

        Returns:
            AgentMessageEntity: 创建的消息实体
        """
        session = self._get_session()
        try:
            message = AgentMessageEntity(
                id=uuid.uuid4(),
                session_id=message_data.get('session_id'),
                role=message_data.get('role'),
                content=message_data.get('content'),
                content_type=message_data.get('content_type', 'text'),
                tokens_used=message_data.get('tokens_used', 0),
                latency_ms=message_data.get('latency_ms', 0),
                metadata=message_data.get('metadata', {}),
                tool_calls=message_data.get('tool_calls'),
                function_call=message_data.get('function_call'),
                parent_message_id=message_data.get('parent_message_id'),
                tenant_id=message_data.get('tenant_id')
            )
            session.add(message)

            # 更新会话的消息计数和最后消息时间
            agent_session = session.query(AgentSessionEntity).filter(
                AgentSessionEntity.id == message_data.get('session_id')
            ).first()
            if agent_session:
                agent_session.message_count = (agent_session.message_count or 0) + 1
                agent_session.last_message_at = datetime.utcnow()

            session.commit()
            session.refresh(message)
            return message
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to add message: %s", str(e))
            raise
        finally:
            session.close()

    def get_session_messages(self,
                            session_id: str,
                            limit: int = 100,
                            offset: int = 0,
                            order: str = 'asc') -> List[AgentMessageEntity]:
        """获取会话的消息列表

        Args:
            session_id: 会话ID
            limit: 返回数量限制
            offset: 偏移量
            order: 排序方式 ('asc' 或 'desc')

        Returns:
            List[AgentMessageEntity]: 消息列表
        """
        session = self._get_session()
        try:
            query = session.query(AgentMessageEntity).filter(
                AgentMessageEntity.session_id == session_id
            )

            if order == 'desc':
                query = query.order_by(desc(AgentMessageEntity.created_at))
            else:
                query = query.order_by(AgentMessageEntity.created_at)

            messages = query.offset(offset).limit(limit).all()
            return messages
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get messages for session %s: %s", session_id, str(e))
            return []
        finally:
            session.close()

    def get_recent_context_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取会话的最近消息作为上下文

        Args:
            session_id: 会话ID
            limit: 消息数量限制

        Returns:
            List[Dict]: 消息列表 (格式化为LLM上下文)
        """
        messages = self.get_session_messages(session_id, limit=limit, order='desc')
        messages.reverse()  # 按时间正序排列

        return [
            {
                'role': msg.role,
                'content': msg.content,
                'tool_calls': msg.tool_calls
            }
            for msg in messages
        ]

    # ============================================================================
    # 长期记忆管理
    # ============================================================================

    def create_memory(self, memory_data: Dict[str, Any]) -> AgentMemoryEntity:
        """创建长期记忆

        Args:
            memory_data: 记忆数据字典，包含:
                - agent_id: 智能体ID (必需)
                - user_id: 用户ID (必需)
                - memory_type: 记忆类型 (fact, preference, event, skill)
                - content: 记忆内容 (必需)
                - embedding: 向量嵌入 (可选)
                - importance: 重要性分数 (可选, 0-1)
                - source_session_id: 来源会话ID (可选)
                - metadata: 元数据 (可选)
                - expires_at: 过期时间 (可选)

        Returns:
            AgentMemoryEntity: 创建的记忆实体
        """
        session = self._get_session()
        try:
            memory = AgentMemoryEntity(
                id=uuid.uuid4(),
                agent_id=memory_data.get('agent_id'),
                user_id=memory_data.get('user_id'),
                memory_type=memory_data.get('memory_type', 'fact'),
                content=memory_data.get('content'),
                embedding=memory_data.get('embedding'),
                importance=memory_data.get('importance', 0.5),
                access_count=0,
                source_session_id=memory_data.get('source_session_id'),
                metadata=memory_data.get('metadata', {}),
                expires_at=memory_data.get('expires_at'),
                is_active=True,
                tenant_id=memory_data.get('tenant_id')
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.info("Created memory: %s", memory.id)
            return memory
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create memory: %s", str(e))
            raise
        finally:
            session.close()

    def get_agent_memories(self,
                          agent_id: str,
                          user_id: str,
                          memory_type: Optional[str] = None,
                          min_importance: float = 0.0,
                          limit: int = 50) -> List[AgentMemoryEntity]:
        """获取智能体的长期记忆

        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            memory_type: 记忆类型过滤 (可选)
            min_importance: 最小重要性阈值
            limit: 返回数量限制

        Returns:
            List[AgentMemoryEntity]: 记忆列表
        """
        session = self._get_session()
        try:
            # pylint: disable=singleton-comparison
            query = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.agent_id == agent_id,
                AgentMemoryEntity.user_id == user_id,
                AgentMemoryEntity.is_active == True,
                AgentMemoryEntity.importance >= min_importance
            )

            # 过滤过期记忆
            query = query.filter(
                or_(
                    AgentMemoryEntity.expires_at == None,
                    AgentMemoryEntity.expires_at > datetime.utcnow()
                )
            )

            if memory_type:
                query = query.filter(AgentMemoryEntity.memory_type == memory_type)

            memories = query.order_by(desc(AgentMemoryEntity.importance))\
                           .limit(limit)\
                           .all()
            return memories
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get memories: %s", str(e))
            return []
        finally:
            session.close()

    def search_memories_by_relevance(self,
                                    agent_id: str,
                                    user_id: str,
                                    # pylint: disable=unused-argument
                                    query_embedding: List[float],
                                    limit: int = 10) -> List[AgentMemoryEntity]:
        """根据相似度搜索记忆

        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            query_embedding: 查询向量
            limit: 返回数量限制

        Returns:
            List[AgentMemoryEntity]: 相关记忆列表

        Note:
            当前使用简单的获取所有记忆方式，
            生产环境建议使用向量数据库 (如 pgvector, Milvus) 进行相似度搜索
        """
        # 简化实现: 获取所有活跃记忆，后续可以集成向量数据库
        memories = self.get_agent_memories(agent_id, user_id, limit=limit * 2)

        # 如果有embedding，可以在这里计算相似度排序
        # 目前直接按重要性返回
        return memories[:limit]

    def update_memory_access(self, memory_id: str) -> bool:
        """更新记忆访问统计

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 更新是否成功
        """
        session = self._get_session()
        try:
            memory = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.id == memory_id
            ).first()

            if not memory:
                return False

            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.utcnow()

            session.commit()
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update memory access: %s", str(e))
            return False
        finally:
            session.close()

    def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """更新记忆重要性

        Args:
            memory_id: 记忆ID
            importance: 新的重要性分数 (0-1)

        Returns:
            bool: 更新是否成功
        """
        session = self._get_session()
        try:
            memory = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.id == memory_id
            ).first()

            if not memory:
                return False

            memory.importance = max(0.0, min(1.0, importance))
            memory.updated_at = datetime.utcnow()

            session.commit()
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update memory importance: %s", str(e))
            return False
        finally:
            session.close()

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆 (软删除)

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 删除是否成功
        """
        session = self._get_session()
        try:
            memory = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.id == memory_id
            ).first()

            if not memory:
                return False

            memory.is_active = False
            memory.updated_at = datetime.utcnow()

            session.commit()
            logger.info("Deleted memory: %s", memory_id)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to delete memory: %s", str(e))
            return False
        finally:
            session.close()

    def consolidate_memories(self, agent_id: str, user_id: str) -> int:
        """整合记忆 - 清理过期和低重要性记忆

        Args:
            agent_id: 智能体ID
            user_id: 用户ID

        Returns:
            int: 清理的记忆数量
        """
        session = self._get_session()
        try:
            # pylint: disable=singleton-comparison
            # 清理过期记忆
            expired_count = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.agent_id == agent_id,
                AgentMemoryEntity.user_id == user_id,
                AgentMemoryEntity.is_active == True,
                AgentMemoryEntity.expires_at != None,
                AgentMemoryEntity.expires_at < datetime.utcnow()
            ).update({'is_active': False})

            # 清理长时间未访问的低重要性记忆
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            unused_count = session.query(AgentMemoryEntity).filter(
                AgentMemoryEntity.agent_id == agent_id,
                AgentMemoryEntity.user_id == user_id,
                AgentMemoryEntity.is_active == True,
                AgentMemoryEntity.importance < 0.3,
                or_(
                    AgentMemoryEntity.last_accessed_at == None,
                    AgentMemoryEntity.last_accessed_at < cutoff_date
                )
            ).update({'is_active': False})

            session.commit()
            total = expired_count + unused_count
            logger.info("Consolidated %d memories for agent %s", total, agent_id)
            return total
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to consolidate memories: %s", str(e))
            return 0
        finally:
            session.close()

    # ============================================================================
    # 执行记录管理
    # ============================================================================

    def create_execution(self, execution_data: Dict[str, Any]) -> AgentExecutionEntity:
        """创建执行记录

        Args:
            execution_data: 执行数据字典，包含:
                - agent_id: 智能体ID (必需)
                - user_id: 用户ID (必需)
                - session_id: 会话ID (可选)
                - execution_type: 执行类型 (可选)
                - input_data: 输入数据 (可选)

        Returns:
            AgentExecutionEntity: 创建的执行记录实体
        """
        session = self._get_session()
        try:
            execution = AgentExecutionEntity(
                id=uuid.uuid4(),
                agent_id=execution_data.get('agent_id'),
                session_id=execution_data.get('session_id'),
                user_id=execution_data.get('user_id'),
                execution_type=execution_data.get('execution_type', 'chat'),
                input_data=execution_data.get('input_data'),
                status='pending',
                started_at=datetime.utcnow(),
                tenant_id=execution_data.get('tenant_id')
            )
            session.add(execution)
            session.commit()
            session.refresh(execution)
            return execution
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create execution: %s", str(e))
            raise
        finally:
            session.close()

    def update_execution(self, execution_id: str, update_data: Dict[str, Any]) -> Optional[AgentExecutionEntity]:
        """更新执行记录

        Args:
            execution_id: 执行ID
            update_data: 要更新的字段，可包含:
                - status: 状态 (running, completed, failed)
                - output_data: 输出数据
                - error_message: 错误信息
                - tokens_input: 输入token数
                - tokens_output: 输出token数
                - cost: 成本
                - graph_state: LangGraph状态
                - checkpoints: 检查点

        Returns:
            AgentExecutionEntity: 更新后的执行记录
        """
        session = self._get_session()
        try:
            execution = session.query(AgentExecutionEntity).filter(
                AgentExecutionEntity.id == execution_id
            ).first()

            if not execution:
                return None

            allowed_fields = [
                'status', 'output_data', 'error_message', 'tokens_input',
                'tokens_output', 'cost', 'graph_state', 'checkpoints'
            ]

            for field in allowed_fields:
                if field in update_data:
                    setattr(execution, field, update_data[field])

            # 如果状态变为完成或失败，记录完成时间
            if update_data.get('status') in ['completed', 'failed']:
                execution.completed_at = datetime.utcnow()
                if execution.started_at:
                    execution.duration_ms = int(
                        (execution.completed_at - execution.started_at).total_seconds() * 1000
                    )

            execution.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(execution)
            return execution
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update execution %s: %s", execution_id, str(e))
            raise
        finally:
            session.close()

    def get_execution_by_id(self, execution_id: str) -> Optional[AgentExecutionEntity]:
        """根据ID获取执行记录

        Args:
            execution_id: 执行ID

        Returns:
            AgentExecutionEntity: 执行记录实体
        """
        session = self._get_session()
        try:
            execution = session.query(AgentExecutionEntity).filter(
                AgentExecutionEntity.id == execution_id
            ).first()
            return execution
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get execution %s: %s", execution_id, str(e))
            return None
        finally:
            session.close()

    def get_executions(self,
                      agent_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      status: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[AgentExecutionEntity]:
        """获取执行记录列表

        Args:
            agent_id: 智能体ID过滤 (可选)
            user_id: 用户ID过滤 (可选)
            session_id: 会话ID过滤 (可选)
            status: 状态过滤 (可选)
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[AgentExecutionEntity]: 执行记录列表
        """
        session = self._get_session()
        try:
            query = session.query(AgentExecutionEntity)

            if agent_id:
                query = query.filter(AgentExecutionEntity.agent_id == agent_id)
            if user_id:
                query = query.filter(AgentExecutionEntity.user_id == user_id)
            if session_id:
                query = query.filter(AgentExecutionEntity.session_id == session_id)
            if status:
                query = query.filter(AgentExecutionEntity.status == status)

            executions = query.order_by(desc(AgentExecutionEntity.started_at))\
                             .offset(offset)\
                             .limit(limit)\
                             .all()
            return executions
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get executions: %s", str(e))
            return []
        finally:
            session.close()

    def get_running_executions_count(self, agent_id: str) -> int:
        """获取正在运行的执行数量

        Args:
            agent_id: 智能体ID

        Returns:
            int: 正在运行的执行数量
        """
        session = self._get_session()
        try:
            # pylint: disable=not-callable
            count = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.agent_id == agent_id,
                AgentExecutionEntity.status.in_(['pending', 'running'])
            ).scalar()
            return count or 0
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get running executions count: %s", str(e))
            return 0
        finally:
            session.close()

    # ============================================================================
    # 工具管理
    # ============================================================================

    def create_tool(self, tool_data: Dict[str, Any]) -> AgentToolEntity:
        """创建工具

        Args:
            tool_data: 工具数据字典，包含:
                - agent_id: 智能体ID (可选, NULL表示全局工具)
                - name: 工具名称 (必需)
                - description: 描述 (可选)
                - tool_type: 类型 (builtin, custom, api)
                - schema: 工具schema (可选)
                - config: 配置 (可选)

        Returns:
            AgentToolEntity: 创建的工具实体
        """
        session = self._get_session()
        try:
            tool = AgentToolEntity(
                id=uuid.uuid4(),
                agent_id=tool_data.get('agent_id'),
                name=tool_data.get('name'),
                description=tool_data.get('description'),
                tool_type=tool_data.get('tool_type', 'custom'),
                schema=tool_data.get('schema'),
                config=tool_data.get('config', {}),
                is_enabled=True,
                tenant_id=tool_data.get('tenant_id')
            )
            session.add(tool)
            session.commit()
            session.refresh(tool)
            logger.info("Created tool: %s - %s", tool.id, tool.name)
            return tool
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create tool: %s", str(e))
            raise
        finally:
            session.close()

    def get_agent_tools(self,
                       agent_id: str,
                       include_global: bool = True,
                       only_enabled: bool = True) -> List[AgentToolEntity]:
        """获取智能体可用的工具

        Args:
            agent_id: 智能体ID
            include_global: 是否包含全局工具
            only_enabled: 是否只返回启用的工具

        Returns:
            List[AgentToolEntity]: 工具列表
        """
        session = self._get_session()
        try:
            if include_global:
                # pylint: disable=singleton-comparison
                query = session.query(AgentToolEntity).filter(
                    or_(
                        AgentToolEntity.agent_id == agent_id,
                        AgentToolEntity.agent_id == None
                    )
                )
            else:
                query = session.query(AgentToolEntity).filter(
                    AgentToolEntity.agent_id == agent_id
                )

            if only_enabled:
                # pylint: disable=singleton-comparison
                query = query.filter(AgentToolEntity.is_enabled == True)

            tools = query.all()
            return tools
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get tools for agent %s: %s", agent_id, str(e))
            return []
        finally:
            session.close()

    def update_tool(self, tool_id: str, update_data: Dict[str, Any]) -> Optional[AgentToolEntity]:
        """更新工具

        Args:
            tool_id: 工具ID
            update_data: 要更新的字段

        Returns:
            AgentToolEntity: 更新后的工具实体
        """
        session = self._get_session()
        try:
            tool = session.query(AgentToolEntity).filter(
                AgentToolEntity.id == tool_id
            ).first()

            if not tool:
                return None

            allowed_fields = ['name', 'description', 'schema', 'config', 'is_enabled']
            for field in allowed_fields:
                if field in update_data:
                    setattr(tool, field, update_data[field])

            tool.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(tool)
            return tool
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update tool %s: %s", tool_id, str(e))
            raise
        finally:
            session.close()

    def update_tool_statistics(self, tool_id: str, latency_ms: float) -> bool:
        """更新工具统计信息

        Args:
            tool_id: 工具ID
            latency_ms: 执行延迟(毫秒)

        Returns:
            bool: 更新是否成功
        """
        session = self._get_session()
        try:
            tool = session.query(AgentToolEntity).filter(
                AgentToolEntity.id == tool_id
            ).first()

            if not tool:
                return False

            tool.execution_count = (tool.execution_count or 0) + 1
            prev_avg = tool.avg_latency_ms or 0
            tool.avg_latency_ms = (
                prev_avg * (tool.execution_count - 1) + latency_ms
            ) / tool.execution_count

            session.commit()
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update tool statistics: %s", str(e))
            return False
        finally:
            session.close()

    # ============================================================================
    # 搜索和统计
    # ============================================================================

    def search_agents(self,
                     query: str,
                     user_id: Optional[str] = None,
                     include_public: bool = True,
                     limit: int = 50) -> List[AgentEntity]:
        """搜索智能体

        Args:
            query: 搜索关键词
            user_id: 用户ID (搜索用户自己的智能体)
            include_public: 是否包含公开智能体
            limit: 返回数量限制

        Returns:
            List[AgentEntity]: 匹配的智能体列表
        """
        session = self._get_session()
        try:
            search_pattern = f"%{query}%"

            base_query = session.query(AgentEntity).filter(
                AgentEntity.status == 'active',
                or_(
                    AgentEntity.name.ilike(search_pattern),
                    AgentEntity.description.ilike(search_pattern)
                )
            )

            if user_id and include_public:
                # pylint: disable=singleton-comparison
                base_query = base_query.filter(
                    or_(
                        AgentEntity.user_id == user_id,
                        AgentEntity.is_public == True
                    )
                )
            elif user_id:
                base_query = base_query.filter(AgentEntity.user_id == user_id)
            elif include_public:
                # pylint: disable=singleton-comparison
                base_query = base_query.filter(AgentEntity.is_public == True)

            agents = base_query.order_by(desc(AgentEntity.execution_count))\
                              .limit(limit)\
                              .all()
            return agents
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to search agents: %s", str(e))
            return []
        finally:
            session.close()

    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体统计信息

        Args:
            agent_id: 智能体ID

        Returns:
            Dict: 统计信息字典
        """
        session = self._get_session()
        try:
            agent = session.query(AgentEntity).filter(
                AgentEntity.id == agent_id
            ).first()

            if not agent:
                return {}

            # pylint: disable=not-callable, singleton-comparison
            # 获取执行统计
            total_executions = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.agent_id == agent_id
            ).scalar() or 0

            successful_executions = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.agent_id == agent_id,
                AgentExecutionEntity.status == 'completed'
            ).scalar() or 0

            failed_executions = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.agent_id == agent_id,
                AgentExecutionEntity.status == 'failed'
            ).scalar() or 0

            avg_duration = session.query(func.avg(AgentExecutionEntity.duration_ms)).filter(
                AgentExecutionEntity.agent_id == agent_id,
                AgentExecutionEntity.duration_ms != None
            ).scalar() or 0

            # 获取会话统计
            total_sessions = session.query(func.count(AgentSessionEntity.id)).filter(
                AgentSessionEntity.agent_id == agent_id
            ).scalar() or 0

            active_sessions = session.query(func.count(AgentSessionEntity.id)).filter(
                AgentSessionEntity.agent_id == agent_id,
                AgentSessionEntity.status == 'active'
            ).scalar() or 0

            # 获取记忆统计
            total_memories = session.query(func.count(AgentMemoryEntity.id)).filter(
                AgentMemoryEntity.agent_id == agent_id,
                AgentMemoryEntity.is_active == True
            ).scalar() or 0

            return {
                'agent_id': str(agent_id),
                'name': agent.name,
                'execution_count': agent.execution_count or 0,
                'success_rate': agent.success_rate or 0,
                'avg_response_time': agent.avg_response_time or 0,
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'avg_duration_ms': float(avg_duration),
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'total_memories': total_memories
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get agent statistics: %s", str(e))
            return {}
        finally:
            session.close()

    def get_user_agent_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户的智能体使用摘要

        Args:
            user_id: 用户ID

        Returns:
            Dict: 使用摘要
        """
        session = self._get_session()
        try:
            # pylint: disable=not-callable
            # 智能体统计
            total_agents = session.query(func.count(AgentEntity.id)).filter(
                AgentEntity.user_id == user_id,
                AgentEntity.status != 'deleted'
            ).scalar() or 0

            active_agents = session.query(func.count(AgentEntity.id)).filter(
                AgentEntity.user_id == user_id,
                AgentEntity.status == 'active'
            ).scalar() or 0

            # 会话统计
            total_sessions = session.query(func.count(AgentSessionEntity.id)).filter(
                AgentSessionEntity.user_id == user_id
            ).scalar() or 0

            # 执行统计
            total_executions = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.user_id == user_id
            ).scalar() or 0

            # 本周执行数
            week_ago = datetime.utcnow() - timedelta(days=7)
            weekly_executions = session.query(func.count(AgentExecutionEntity.id)).filter(
                AgentExecutionEntity.user_id == user_id,
                AgentExecutionEntity.started_at >= week_ago
            ).scalar() or 0

            return {
                'user_id': user_id,
                'total_agents': total_agents,
                'active_agents': active_agents,
                'total_sessions': total_sessions,
                'total_executions': total_executions,
                'weekly_executions': weekly_executions
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get user agent summary: %s", str(e))
            return {}
        finally:
            session.close()
