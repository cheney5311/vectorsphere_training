"""智能体服务层

实现智能体相关的业务逻辑，包括：
- 智能体 CRUD 操作
- 会话管理
- 长期记忆管理
- 智能推理执行
- 知识检索与推理
- LangGraph 工业级 Agent 编排能力
"""

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

# LangGraph 工业级组件导入
from backend.algo.langgraph import (
    # Agent 编排组件
    AgentPool as LangGraphAgentPool,
    AgentOrchestrator as LangGraphOrchestrator,
    AgentRegistry as LangGraphRegistry,
    get_agent_registry as get_langgraph_registry,
    # Agent 类型
    BaseAgent as LangGraphBaseAgent,
    # 工厂
    MasterFactory,
    get_master_factory,
    create_multi_agent_system,
    # 检查点
    # 工具
    get_builtin_tools,
    get_tools_by_category,
    # 状态和消息
    # 图
    StateGraph,
    GraphConfig
)
from backend.core.exceptions import ValidationError
from backend.modules.agent.exceptions.agent_exceptions import (
    AgentNotFoundError,
    AgentValidationError,
    AgentExecutionError
)
from backend.repositories.agent_repository import AgentRepository, get_agent_repository
from backend.schemas.agent import Agent
from backend.schemas.agent_type import AgentType
from backend.utils.validation import validate_id

# Agent 实例管理器导入
from backend.services.agent_instance_manager import agent_instance_manager

logger = logging.getLogger(__name__)


# ============================================================================
# DTOs 数据传输对象
# ============================================================================

@dataclass
class ChatRequest:
    """对话请求 DTO"""
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    use_memory: bool = True
    max_tokens: int = 2048
    temperature: float = 0.7
    provider: str = "local"
    stream: bool = False


@dataclass
class ChatResponse:
    """对话响应 DTO"""
    message: str
    session_id: str
    execution_id: str
    tokens_used: int = 0
    latency_ms: float = 0
    memories_used: List[str] = None
    tool_calls: List[Dict[str, Any]] = None
    reasoning_steps: List[str] = None


@dataclass
class MemoryCreateRequest:
    """创建记忆请求 DTO"""
    content: str
    memory_type: str = "fact"
    importance: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    expires_days: Optional[int] = None


@dataclass
class ExecutionResult:
    """执行结果 DTO"""
    execution_id: str
    status: str
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0
    tokens_input: int = 0
    tokens_output: int = 0
    graph_state: Optional[Dict[str, Any]] = None


def get_agent_service() -> 'AgentService':
    """获取智能体服务实例
    
    Returns:
        AgentService: 智能体服务实例
    """
    repository = get_agent_repository()
    return AgentService(repository)


class AgentService:
    """智能体服务
    
    提供智能体的完整业务逻辑，包括CRUD、会话、记忆和推理功能。
    集成 LangGraph 工业级 Agent 编排能力。
    """
    
    def __init__(self, agent_repository: AgentRepository):
        """初始化智能体服务
        
        Args:
            agent_repository: 智能体仓库实例
        """
        self.repository = agent_repository
        self._inference_service = None
        self._knowledge_engine = None
        self._embedding_service = None
        
        # LangGraph 工业级组件
        self._langgraph_pool: Optional['LangGraphAgentPool'] = None
        self._langgraph_orchestrator: Optional['LangGraphOrchestrator'] = None
        self._langgraph_registry: Optional['LangGraphRegistry'] = None
        self._master_factory: Optional['MasterFactory'] = None
        self._langgraph_agents: Dict[str, 'LangGraphBaseAgent'] = {}
        
        # 初始化 LangGraph 组件
        self._init_langgraph_components()
    
    @property
    def inference_service(self):
        """懒加载推理服务"""
        if self._inference_service is None:
            try:
                from backend.services.langchain_inference_service import LangChainInferenceService
                self._inference_service = LangChainInferenceService()
            except Exception as e:
                logger.warning(f"Failed to load inference service: {e}")
        return self._inference_service
    
    @property
    def knowledge_engine(self):
        """懒加载知识推理引擎"""
        if self._knowledge_engine is None:
            try:
                from backend.algo.knowledge_reasoning import KnowledgeReasoningEngine
                self._knowledge_engine = KnowledgeReasoningEngine()
            except Exception as e:
                logger.warning(f"Failed to load knowledge engine: {e}")
        return self._knowledge_engine
    
    @property
    def embedding_service(self):
        """懒加载向量嵌入服务"""
        if self._embedding_service is None:
            try:
                from backend.services.embedding_service import EmbeddingService
                self._embedding_service = EmbeddingService()
            except Exception as e:
                logger.warning(f"Failed to load embedding service: {e}")
        return self._embedding_service
    
    # ============================================================================
    # 智能体 CRUD 操作
    # ============================================================================
    
    def create_agent(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
        agent_type: Optional[AgentType] = None,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        is_public: bool = False,
        capabilities: Optional[List[str]] = None
    ) -> Agent:
        """创建智能体
        
        Args:
            user_id: 用户ID
            name: 智能体名称
            description: 智能体描述
            version: 版本号
            config: 配置信息
            agent_type: 智能体类型
            system_prompt: 系统提示词
            model_config: 模型配置
            is_public: 是否公开
            capabilities: 能力列表
            
        Returns:
            Agent: 创建的智能体对象
            
        Raises:
            AgentValidationError: 当输入参数验证失败时
        """
        if not name or len(name.strip()) == 0:
            raise AgentValidationError("智能体名称不能为空")
        
        if len(name) > 200:
            raise AgentValidationError("智能体名称不能超过200个字符")
        
        try:
            agent_data = {
                'user_id': user_id,
                'name': name.strip(),
                'description': description,
                'version': version,
                'config': config or {},
                'agent_type': agent_type.value if agent_type else 'chat',
                'system_prompt': system_prompt,
                'model_config': model_config or {},
                'is_public': is_public,
                'capabilities': capabilities or []
            }
            
            entity = self.repository.create_agent(agent_data)
            
            # 转换为 Agent DTO
            agent = self._entity_to_agent(entity)
            
            # 创建运行时实例
            self._register_agent_instance(agent)
            
            logger.info(f"Created agent: {agent.agent_id} - {agent.name}")
            return agent
            
        except ValidationError as e:
            raise AgentValidationError(f"创建智能体失败: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise AgentValidationError(f"创建智能体失败: {str(e)}") from e
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 智能体对象，如果不存在则返回None
            
        Raises:
            AgentValidationError: 当ID格式不正确时
        """
        validate_id(agent_id, "agent_id")
        
        entity = self.repository.get_agent_by_id(agent_id)
        if not entity:
            return None
        
        return self._entity_to_agent(entity)
    
    def list_agents(
        self, 
        user_id: str, 
        status: Optional[str] = None,
        agent_type: Optional[str] = None,
        is_public: Optional[bool] = None,
        limit: int = 50, 
        offset: int = 0
    ) -> List[Agent]:
        """获取用户智能体列表
        
        Args:
            user_id: 用户ID
            status: 状态过滤
            agent_type: 类型过滤
            is_public: 是否公开过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Agent]: 智能体列表
            
        Raises:
            AgentValidationError: 当输入参数验证失败时
        """
        if limit <= 0 or limit > 100:
            raise AgentValidationError("限制数量必须在1-100之间")
        
        if offset < 0:
            raise AgentValidationError("偏移量不能为负数")
        
        entities = self.repository.get_agents_by_user(
            user_id=user_id,
            status=status,
            agent_type=agent_type,
            is_public=is_public,
            limit=limit,
            offset=offset
        )
        
        return [self._entity_to_agent(e) for e in entities]
    
    def update_agent(
        self, 
        agent_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        is_public: Optional[bool] = None,
        capabilities: Optional[List[str]] = None
    ) -> Agent:
        """更新智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            config: 配置信息
            system_prompt: 系统提示词
            model_config: 模型配置
            is_public: 是否公开
            capabilities: 能力列表
            
        Returns:
            Agent: 更新后的智能体对象
            
        Raises:
            AgentNotFoundError: 当智能体不存在时
            AgentValidationError: 当输入参数验证失败时
        """
        validate_id(agent_id, "agent_id")
        
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        update_data = {}
        if name is not None:
            if len(name.strip()) == 0:
                raise AgentValidationError("智能体名称不能为空")
            update_data['name'] = name.strip()
        if description is not None:
            update_data['description'] = description
        if config is not None:
            update_data['config'] = config
        if system_prompt is not None:
            update_data['system_prompt'] = system_prompt
        if model_config is not None:
            update_data['model_config'] = model_config
        if is_public is not None:
            update_data['is_public'] = is_public
        if capabilities is not None:
            update_data['capabilities'] = capabilities
        
        entity = self.repository.update_agent(agent_id, update_data)
        return self._entity_to_agent(entity)
    
    def delete_agent(self, agent_id: str) -> bool:
        """删除智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 删除成功返回True
            
        Raises:
            AgentValidationError: 当ID格式不正确时
        """
        validate_id(agent_id, "agent_id")
        
        # 检查是否有正在运行的执行
        running_count = self.repository.get_running_executions_count(agent_id)
        if running_count > 0:
            raise AgentValidationError(f"智能体有 {running_count} 个任务正在执行，无法删除")
        
        return self.repository.delete_agent(agent_id)
    
    def activate_agent(self, agent_id: str) -> Agent:
        """激活智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 激活后的智能体对象
        """
        return self._update_agent_status(agent_id, 'active')
    
    def deactivate_agent(self, agent_id: str) -> Agent:
        """停用智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 停用后的智能体对象
        """
        return self._update_agent_status(agent_id, 'inactive')
    
    def _update_agent_status(self, agent_id: str, status: str) -> Agent:
        """更新智能体状态"""
        validate_id(agent_id, "agent_id")
        
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        entity = self.repository.update_agent(agent_id, {'status': status})
        return self._entity_to_agent(entity)
    
    # ============================================================================
    # 会话管理
    # ============================================================================
    
    def create_session(
        self,
        agent_id: str,
        user_id: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建对话会话
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            title: 会话标题
            context: 初始上下文
            
        Returns:
            Dict: 会话信息
        """
        validate_id(agent_id, "agent_id")
        
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        if agent.status != 'active':
            raise AgentValidationError("只能与活跃状态的智能体创建会话")
        
        session_data = {
            'agent_id': agent_id,
            'user_id': user_id,
            'title': title or f"与 {agent.name} 的对话",
            'context': context or {}
        }
        
        session = self.repository.create_session(session_data)
        return session.to_dict()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话信息
        """
        session = self.repository.get_session_by_id(session_id)
        return session.to_dict() if session else None
    
    def list_sessions(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取用户会话列表
        
        Args:
            user_id: 用户ID
            agent_id: 智能体ID过滤
            status: 状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict]: 会话列表
        """
        sessions = self.repository.get_user_sessions(
            user_id=user_id,
            agent_id=agent_id,
            status=status,
            limit=limit,
            offset=offset
        )
        return [s.to_dict() for s in sessions]
    
    def get_session_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取会话消息列表
        
        Args:
            session_id: 会话ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict]: 消息列表
        """
        messages = self.repository.get_session_messages(
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        return [m.to_dict() for m in messages]
    
    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新会话
        
        Args:
            session_id: 会话ID
            title: 标题
            status: 状态
            context: 上下文
            summary: 摘要
            
        Returns:
            Dict: 更新后的会话信息
        """
        update_data = {}
        if title is not None:
            update_data['title'] = title
        if status is not None:
            update_data['status'] = status
        if context is not None:
            update_data['context'] = context
        if summary is not None:
            update_data['summary'] = summary
        
        session = self.repository.update_session(session_id, update_data)
        if not session:
            raise AgentNotFoundError(f"会话 {session_id} 不存在")
        return session.to_dict()
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        return self.repository.delete_session(session_id)
    
    def archive_session(self, session_id: str) -> Dict[str, Any]:
        """归档会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 归档后的会话信息
        """
        return self.update_session(session_id, status='archived')
    
    # ============================================================================
    # 长期记忆管理
    # ============================================================================
    
    def add_memory(
        self,
        agent_id: str,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        source_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """添加长期记忆
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            content: 记忆内容
            memory_type: 记忆类型 (fact, preference, event, skill)
            importance: 重要性分数 (0-1)
            source_session_id: 来源会话ID
            metadata: 元数据
            expires_days: 过期天数
            
        Returns:
            Dict: 创建的记忆信息
        """
        validate_id(agent_id, "agent_id")
        
        if not content or len(content.strip()) == 0:
            raise AgentValidationError("记忆内容不能为空")
        
        if memory_type not in ['fact', 'preference', 'event', 'skill']:
            raise AgentValidationError("记忆类型必须是 fact, preference, event 或 skill")
        
        importance = max(0.0, min(1.0, importance))
        
        # 生成向量嵌入
        embedding = None
        if self.embedding_service:
            try:
                result = self.embedding_service.generate_embedding(content)
                embedding = result.get('embedding') if result else None
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        memory_data = {
            'agent_id': agent_id,
            'user_id': user_id,
            'content': content.strip(),
            'memory_type': memory_type,
            'importance': importance,
            'embedding': embedding,
            'source_session_id': source_session_id,
            'metadata': metadata or {},
            'expires_at': expires_at
        }
        
        memory = self.repository.create_memory(memory_data)
        logger.info(f"Added memory for agent {agent_id}: {memory.id}")
        return memory.to_dict()
    
    def get_memories(
        self,
        agent_id: str,
        user_id: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取智能体记忆
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            memory_type: 记忆类型过滤
            min_importance: 最小重要性
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 记忆列表
        """
        memories = self.repository.get_agent_memories(
            agent_id=agent_id,
            user_id=user_id,
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit
        )
        return [m.to_dict() for m in memories]
    
    def search_relevant_memories(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索相关记忆
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            query: 搜索查询
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 相关记忆列表
        """
        # 生成查询向量
        query_embedding = None
        if self.embedding_service:
            try:
                result = self.embedding_service.generate_embedding(query)
                query_embedding = result.get('embedding') if result else None
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}")
        
        if query_embedding:
            memories = self.repository.search_memories_by_relevance(
                agent_id=agent_id,
                user_id=user_id,
                query_embedding=query_embedding,
                limit=limit
            )
        else:
            # 回退到按重要性获取
            memories = self.repository.get_agent_memories(
                agent_id=agent_id,
                user_id=user_id,
                limit=limit
            )
        
        # 更新访问统计
        for memory in memories:
            self.repository.update_memory_access(str(memory.id))
        
        return [m.to_dict() for m in memories]
    
    def update_memory_importance(
        self,
        memory_id: str,
        importance: float
    ) -> bool:
        """更新记忆重要性
        
        Args:
            memory_id: 记忆ID
            importance: 新的重要性分数
            
        Returns:
            bool: 更新是否成功
        """
        importance = max(0.0, min(1.0, importance))
        return self.repository.update_memory_importance(memory_id, importance)
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 删除是否成功
        """
        return self.repository.delete_memory(memory_id)
    
    def consolidate_memories(self, agent_id: str, user_id: str) -> int:
        """整合记忆
        
        清理过期和低重要性记忆。
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            
        Returns:
            int: 清理的记忆数量
        """
        return self.repository.consolidate_memories(agent_id, user_id)
    
    async def extract_memories_from_conversation(
        self,
        agent_id: str,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """从对话中提取记忆
        
        使用LLM分析对话，自动提取重要的事实、偏好等信息作为长期记忆。
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            session_id: 会话ID
            messages: 消息列表
            
        Returns:
            List[Dict]: 提取的记忆列表
        """
        if not self.inference_service:
            return []
        
        try:
            # 构建提取提示
            conversation_text = "\n".join([
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages[-10:]  # 最近10条消息
            ])
            
            extraction_prompt = f"""分析以下对话，提取用户的重要信息作为长期记忆。
            
对话内容:
{conversation_text}

请提取以下类型的信息（如果存在）:
1. fact (事实): 用户提到的客观事实
2. preference (偏好): 用户的喜好和习惯
3. event (事件): 重要的事件或计划
4. skill (技能): 用户具备的技能或知识

以JSON格式返回，每个记忆包含: type, content, importance (0-1)
只返回JSON数组，不要其他内容。"""
            
            # 调用LLM提取
            response = await self.inference_service.chat(
                session_id=None,
                message=extraction_prompt,
                context={},
                model_name=None,
                provider="local"
            )
            
            if not response:
                return []
            
            # 解析响应
            import json
            try:
                extracted = json.loads(response)
                if not isinstance(extracted, list):
                    extracted = [extracted]
            except json.JSONDecodeError:
                logger.warning("Failed to parse memory extraction response")
                return []
            
            # 创建记忆
            created_memories = []
            for item in extracted:
                if not isinstance(item, dict):
                    continue
                
                memory_type = item.get('type', 'fact')
                content = item.get('content', '')
                importance = item.get('importance', 0.5)
                
                if content and len(content) > 10:
                    memory = self.add_memory(
                        agent_id=agent_id,
                        user_id=user_id,
                        content=content,
                        memory_type=memory_type,
                        importance=importance,
                        source_session_id=session_id
                    )
                    created_memories.append(memory)
            
            logger.info(f"Extracted {len(created_memories)} memories from conversation")
            return created_memories
            
        except Exception as e:
            logger.error(f"Failed to extract memories: {e}")
            return []
    
    # ============================================================================
    # 智能推理执行
    # ============================================================================
    
    async def chat(
        self,
        agent_id: str,
        user_id: str,
        request: ChatRequest
    ) -> ChatResponse:
        """执行智能对话
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            request: 对话请求
            
        Returns:
            ChatResponse: 对话响应
        """
        start_time = time.time()
        
        # 获取智能体
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        if agent.status != 'active':
            raise AgentValidationError("智能体未激活")
        
        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session = self.create_session(agent_id, user_id)
            session_id = session['id']
        
        # 创建执行记录
        execution = self.repository.create_execution({
            'agent_id': agent_id,
            'user_id': user_id,
            'session_id': session_id,
            'execution_type': 'chat',
            'input_data': {
                'message': request.message,
                'context': request.context
            }
        })
        execution_id = str(execution.id)
        
        try:
            # 更新执行状态为运行中
            self.repository.update_execution(execution_id, {'status': 'running'})
            
            # 保存用户消息
            self.repository.add_message({
                'session_id': session_id,
                'role': 'user',
                'content': request.message
            })
            
            # 获取历史上下文
            context_messages = self.repository.get_recent_context_messages(session_id)
            
            # 获取相关记忆
            memories_used = []
            if request.use_memory:
                relevant_memories = self.search_relevant_memories(
                    agent_id=agent_id,
                    user_id=user_id,
                    query=request.message,
                    limit=5
                )
                memories_used = [m['content'] for m in relevant_memories]
            
            # 构建增强上下文
            enhanced_context = {
                'agent_name': agent.name,
                'system_prompt': agent.config.get('system_prompt', ''),
                'user_context': request.context or {},
                'conversation_history': context_messages,
                'memories': memories_used,
                'capabilities': agent.capabilities
            }
            
            # 执行推理
            response_content = await self._execute_inference(
                agent=agent,
                message=request.message,
                context=enhanced_context,
                provider=request.provider,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # 计算耗时
            latency_ms = (time.time() - start_time) * 1000
            
            # 保存助手消息
            self.repository.add_message({
                'session_id': session_id,
                'role': 'assistant',
                'content': response_content,
                'latency_ms': int(latency_ms)
            })
            
            # 更新执行记录
            self.repository.update_execution(execution_id, {
                'status': 'completed',
                'output_data': {'response': response_content}
            })
            
            # 更新智能体统计
            self.repository.update_agent_statistics(agent_id, True, latency_ms)
            
            return ChatResponse(
                message=response_content,
                session_id=session_id,
                execution_id=execution_id,
                latency_ms=latency_ms,
                memories_used=memories_used
            )
            
        except Exception as e:
            # 更新执行记录为失败
            self.repository.update_execution(execution_id, {
                'status': 'failed',
                'error_message': str(e)
            })
            
            latency_ms = (time.time() - start_time) * 1000
            self.repository.update_agent_statistics(agent_id, False, latency_ms)
            
            logger.error(f"Chat execution failed: {e}")
            raise AgentExecutionError(f"对话执行失败: {str(e)}") from e
    
    async def chat_stream(
        self,
        agent_id: str,
        user_id: str,
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """流式对话
        
        Args:
            agent_id: 智能体ID
            user_id: 用户ID
            request: 对话请求
            
        Yields:
            str: 响应片段
        """
        # 获取智能体
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session = self.create_session(agent_id, user_id)
            session_id = session['id']
        
        # 保存用户消息
        self.repository.add_message({
            'session_id': session_id,
            'role': 'user',
            'content': request.message
        })
        
        # 获取上下文
        context_messages = self.repository.get_recent_context_messages(session_id)
        memories_used = []
        if request.use_memory:
            relevant_memories = self.search_relevant_memories(
                agent_id=agent_id,
                user_id=user_id,
                query=request.message,
                limit=5
            )
            memories_used = [m['content'] for m in relevant_memories]
        
        # 构建上下文
        enhanced_context = {
            'system_prompt': agent.config.get('system_prompt', ''),
            'conversation_history': context_messages,
            'memories': memories_used
        }
        
        # 流式推理
        full_response = ""
        if self.inference_service:
            try:
                async for chunk in self.inference_service.stream_chat(
                    session_id=session_id,
                    messages=[{'role': 'user', 'content': request.message}],
                    provider=request.provider
                ):
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get('chunk', '')
                    else:
                        chunk_text = str(chunk)
                    full_response += chunk_text
                    yield chunk_text
            except Exception as e:
                logger.error(f"Stream chat failed: {e}")
                yield f"[Error: {str(e)}]"
        else:
            # 回退到非流式
            response = await self._execute_inference(
                agent=agent,
                message=request.message,
                context=enhanced_context,
                provider=request.provider
            )
            full_response = response
            yield response
        
        # 保存完整响应
        if full_response:
            self.repository.add_message({
                'session_id': session_id,
                'role': 'assistant',
                'content': full_response
            })
    
    async def _execute_inference(
        self,
        agent: Agent,
        message: str,
        context: Dict[str, Any],
        provider: str = "local",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """执行推理
        
        Args:
            agent: 智能体
            message: 用户消息
            context: 上下文
            provider: 推理提供者
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            str: 响应内容
        """
        if not self.inference_service:
            # 回退到简单响应
            return self._generate_fallback_response(message, context)
        
        try:
            # 构建系统提示
            system_prompt = context.get('system_prompt', '')
            if context.get('memories'):
                system_prompt += f"\n\n用户相关记忆:\n" + "\n".join(
                    f"- {m}" for m in context['memories'][:5]
                )
            
            # 调用推理服务
            response = await self.inference_service.chat(
                session_id=None,
                message=message,
                context={
                    'system_prompt': system_prompt,
                    'history': context.get('conversation_history', [])
                },
                model_name=None,
                provider=provider
            )
            
            return response or self._generate_fallback_response(message, context)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._generate_fallback_response(message, context)
    
    def _generate_fallback_response(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """生成回退响应
        
        当推理服务不可用时的简单响应。
        
        Args:
            message: 用户消息
            context: 上下文
            
        Returns:
            str: 响应内容
        """
        agent_name = context.get('agent_name', '智能助手')
        
        if '帮助' in message or '能做什么' in message:
            capabilities = context.get('capabilities', [])
            if capabilities:
                return f"我是{agent_name}，我可以帮助您:\n" + "\n".join(
                    f"- {cap}" for cap in capabilities
                )
            return f"我是{agent_name}，有什么可以帮助您的吗？"
        
        return f"我收到了您的消息。作为{agent_name}，我会尽力为您提供帮助。请问有什么具体问题吗？"
    
    # ============================================================================
    # 知识推理
    # ============================================================================
    
    def query_knowledge(self, query: str) -> Dict[str, Any]:
        """查询知识库
        
        Args:
            query: 查询字符串
            
        Returns:
            Dict: 查询结果
        """
        if not self.knowledge_engine:
            return {'query': query, 'entities': [], 'relationships': [], 'recommendations': []}
        
        try:
            return self.knowledge_engine.query_knowledge(query)
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return {'query': query, 'error': str(e)}
    
    def get_training_recommendation(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取训练推荐
        
        基于知识推理引擎提供训练参数建议。
        
        Args:
            context: 训练上下文，可包含:
                - data_type: 数据类型 (text, image, tabular)
                - data_size: 数据规模 (small, medium, large)
                - task_type: 任务类型
                - gpu_memory: GPU内存
                
        Returns:
            Dict: 推荐配置
        """
        if not self.knowledge_engine:
            return self._default_training_recommendation(context)
        
        try:
            from backend.algo.base import AlgorithmContext
            
            algo_context = AlgorithmContext(
                inputs=context,
                constraints={},
                metadata={}
            )
            
            result = self.knowledge_engine.suggest(algo_context)
            
            return {
                'recommendation': result.action,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'alternatives': result.alternatives
            }
        except Exception as e:
            logger.error(f"Training recommendation failed: {e}")
            return self._default_training_recommendation(context)
    
    def _default_training_recommendation(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """默认训练推荐"""
        data_type = context.get('data_type', 'tabular')
        
        recommendations = {
            'text': {
                'model': 'transformer',
                'learning_rate': 2e-5,
                'batch_size': 32,
                'epochs': 3
            },
            'image': {
                'model': 'cnn',
                'learning_rate': 1e-3,
                'batch_size': 32,
                'epochs': 10
            },
            'tabular': {
                'model': 'mlp',
                'learning_rate': 1e-3,
                'batch_size': 64,
                'epochs': 50
            }
        }
        
        return {
            'recommendation': recommendations.get(data_type, recommendations['tabular']),
            'confidence': 0.7,
            'reasoning': f'基于{data_type}数据类型的默认推荐'
        }
    
    # ============================================================================
    # 工具管理
    # ============================================================================
    
    def add_tool(
        self,
        agent_id: str,
        name: str,
        description: str,
        tool_type: str = "custom",
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """添加工具
        
        Args:
            agent_id: 智能体ID
            name: 工具名称
            description: 描述
            tool_type: 类型 (builtin, custom, api)
            schema: 工具schema
            config: 配置
            
        Returns:
            Dict: 创建的工具信息
        """
        tool = self.repository.create_tool({
            'agent_id': agent_id,
            'name': name,
            'description': description,
            'tool_type': tool_type,
            'schema': schema,
            'config': config or {}
        })
        return tool.to_dict()
    
    def get_agent_tools(
        self,
        agent_id: str,
        include_global: bool = True
    ) -> List[Dict[str, Any]]:
        """获取智能体工具
        
        Args:
            agent_id: 智能体ID
            include_global: 是否包含全局工具
            
        Returns:
            List[Dict]: 工具列表
        """
        tools = self.repository.get_agent_tools(agent_id, include_global)
        return [t.to_dict() for t in tools]
    
    # ============================================================================
    # 统计和能力管理
    # ============================================================================
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体统计信息
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Dict: 统计信息
        """
        return self.repository.get_agent_statistics(agent_id)
    
    def get_user_agent_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户智能体使用摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 使用摘要
        """
        return self.repository.get_user_agent_summary(user_id)
    
    def add_agent_capability(self, agent_id: str, capability: str) -> Agent:
        """为智能体添加能力
        
        Args:
            agent_id: 智能体ID
            capability: 能力名称
            
        Returns:
            Agent: 更新后的智能体
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        capabilities = list(agent.capabilities or [])
        if capability not in capabilities:
            capabilities.append(capability)
        
        return self.update_agent(agent_id, capabilities=capabilities)
    
    def remove_agent_capability(self, agent_id: str, capability: str) -> Agent:
        """为智能体移除能力
        
        Args:
            agent_id: 智能体ID
            capability: 能力名称
            
        Returns:
            Agent: 更新后的智能体
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"智能体 {agent_id} 不存在")
        
        capabilities = list(agent.capabilities or [])
        if capability in capabilities:
            capabilities.remove(capability)
        
        return self.update_agent(agent_id, capabilities=capabilities)
    
    def search_agents(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_public: bool = True,
        limit: int = 50
    ) -> List[Agent]:
        """搜索智能体
        
        Args:
            query: 搜索关键词
            user_id: 用户ID
            include_public: 是否包含公开智能体
            limit: 返回数量限制
            
        Returns:
            List[Agent]: 匹配的智能体列表
        """
        entities = self.repository.search_agents(
            query=query,
            user_id=user_id,
            include_public=include_public,
            limit=limit
        )
        return [self._entity_to_agent(e) for e in entities]
    
    def get_public_agents(
        self,
        agent_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Agent]:
        """获取公开智能体列表
        
        Args:
            agent_type: 类型过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Agent]: 公开智能体列表
        """
        entities = self.repository.get_public_agents(
            agent_type=agent_type,
            limit=limit,
            offset=offset
        )
        return [self._entity_to_agent(e) for e in entities]
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _entity_to_agent(self, entity) -> Agent:
        """将实体转换为Agent DTO"""
        agent_type = None
        if entity.agent_type:
            try:
                agent_type = AgentType(entity.agent_type)
            except ValueError:
                pass
        
        return Agent(
            agent_id=str(entity.id),
            user_id=entity.user_id,
            name=entity.name,
            description=entity.description,
            version=entity.version,
            status=entity.status,
            config=entity.config or {},
            capabilities=entity.capabilities or [],
            agent_type=agent_type,
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )
    
    def _register_agent_instance(self, agent: Agent):
        """注册智能体运行时实例"""
        try:
            from backend.services.agent_instance_manager import agent_instance_manager
            agent_instance_manager.create_agent_instance(agent)
        except Exception as e:
            logger.warning(f"Failed to register agent instance: {e}")
    
    # ==================== LangGraph 工业级派生方法 ====================
    # 通过调用 backend/modules/agent 实现分层架构
    
    def _init_langgraph_components(self) -> None:
        """初始化 LangGraph 组件
        
        调用 backend/modules/agent 的推理服务
        """
        try:
            # 通过 modules/agent 获取推理服务
            if self.inference_service:
                self._langgraph_pool = getattr(self.inference_service, '_agent_pool', None)
                self._langgraph_orchestrator = getattr(self.inference_service, '_agent_orchestrator', None)
            
            # 获取 MasterFactory
            self._master_factory = get_master_factory()
            self._langgraph_registry = get_langgraph_registry()
            
            logger.info("LangGraph components initialized via modules/agent")
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph components: {e}")
    
    def create_langgraph_agent(
        self,
        agent_type: str,
        name: str,
        user_id: str,
        tools: Optional[List[Any]] = None,
        llm_client: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """创建 LangGraph Agent
        
        调用 backend/modules/agent/langchain_inference_service 的工厂方法
        
        Args:
            agent_type: Agent 类型 (react, plan_execute, reflexion, etc.)
            name: Agent 名称
            user_id: 用户 ID
            tools: 工具列表
            llm_client: LLM 客户端
            config: 配置字典
            
        Returns:
            Agent 信息字典或 None
        """
        if not self.inference_service:
            logger.warning("Inference service not available")
            return None
        
        try:
            # 通过 agent_instance_manager 创建 LangGraph Agent
            agent = agent_instance_manager.create_agent_via_factory(
                agent_type=agent_type,
                name=name,
                tools=tools,
                llm_client=llm_client
            )
            
            if agent:
                # 记录 Agent 到本地映射
                self._langgraph_agents[name] = agent
                
                return {
                    "name": name,
                    "type": agent_type,
                    "user_id": user_id,
                    "status": "created",
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent: {e}")
            return None
    
    def run_langgraph_agent(
        self,
        agent_id: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行 LangGraph Agent
        
        调用 backend/modules/agent/langchain_inference_service 的运行方法
        
        Args:
            agent_id: Agent ID 或名称
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            执行结果字典
        """
        if not self.inference_service:
            return {"error": "Inference service not available"}
        
        try:
            result = agent_instance_manager.run_langgraph_agent(
                agent_id, input_data, context=context
            )
            
            if result:
                return {
                    "status": "success",
                    "agent_id": agent_id,
                    "output": result.output if hasattr(result, 'output') else str(result),
                    "final_answer": result.final_answer if hasattr(result, 'final_answer') else None
                }
            return {"status": "no_result", "agent_id": agent_id}
        except Exception as e:
            logger.error(f"Failed to run LangGraph agent: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_agents_workflow(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """执行 Agent 工作流
        
        调用 backend/modules/agent/langchain_inference_service 的工作流方法
        
        Args:
            workflow: 工作流定义 [(agent_id, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            各步骤结果列表
        """
        if not self.inference_service:
            return [{"error": "Inference service not available"}]
        
        try:
            results = agent_instance_manager.run_agents_workflow(
                workflow, stop_on_error
            )
            
            return [
                {
                    "step": i,
                    "status": "success" if r else "failed",
                    "output": r.output if r and hasattr(r, 'output') else None
                }
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Failed to run workflow: {e}")
            return [{"error": str(e)}]
    
    def run_agents_parallel(
        self,
        tasks: List[Tuple[str, Any]],
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """并行执行多个 Agent 任务
        
        调用 backend/modules/agent/langchain_inference_service 的并行方法
        
        Args:
            tasks: 任务列表 [(agent_id, input_data), ...]
            max_workers: 最大并行数
            
        Returns:
            各任务结果列表
        """
        if not self.inference_service:
            return [{"error": "Inference service not available"}]
        
        try:
            results = agent_instance_manager.run_agents_parallel(
                tasks, max_workers
            )
            
            return [
                {
                    "agent": name,
                    "status": "success" if r else "failed",
                    "output": r.output if r and hasattr(r, 'output') else None
                }
                for name, r in results
            ]
        except Exception as e:
            logger.error(f"Failed to run parallel tasks: {e}")
            return [{"error": str(e)}]
    
    def create_agent_checkpoint(
        self,
        agent_id: str,
        checkpoint_id: Optional[str] = None,
        branch: str = "main",
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """创建 Agent 检查点
        
        调用 backend/modules/agent/langchain_inference_service 的检查点方法
        
        Args:
            agent_id: Agent ID 或名称
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            检查点信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            checkpoint = agent_instance_manager.create_enhanced_checkpoint(
                agent_id, checkpoint_id, branch, tags
            )
            
            if checkpoint:
                return {
                    "agent_id": agent_id,
                    "checkpoint_id": checkpoint_id or "auto_generated",
                    "branch": branch,
                    "tags": tags or [],
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    def setup_agent_production_features(
        self,
        agent_id: str,
        enable_cache: bool = True,
        enable_retry: bool = True,
        enable_rate_limit: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """设置 Agent 生产级特性
        
        调用 backend/modules/agent/langchain_inference_service 的设置方法
        
        Args:
            agent_id: Agent ID 或名称
            enable_cache: 启用缓存
            enable_retry: 启用重试
            enable_rate_limit: 启用限流
            cache_config: 缓存配置
            retry_config: 重试配置
            rate_limit_config: 限流配置
            
        Returns:
            各特性设置结果
        """
        if not self.inference_service:
            return {"error": True}
        
        results = {}
        
        # 设置缓存
        if enable_cache:
            cache_cfg = cache_config or {"max_size": 1000, "default_ttl": 300}
            results["cache"] = agent_instance_manager.setup_tool_cache_for_langgraph_agent(
                agent_id,
                max_size=cache_cfg.get("max_size", 1000),
                default_ttl=cache_cfg.get("default_ttl", 300)
            )
        
        # 设置重试
        if enable_retry:
            retry_cfg = retry_config or {"max_retries": 3, "base_delay": 1.0}
            results["retry"] = agent_instance_manager.setup_retry_handler_for_langgraph_agent(
                agent_id,
                max_retries=retry_cfg.get("max_retries", 3),
                base_delay=retry_cfg.get("base_delay", 1.0)
            )
        
        # 设置限流 - 暂时返回配置信息，待后续实现
        if enable_rate_limit:
            rate_cfg = rate_limit_config or {"rate": 10.0, "burst": 20}
            results["rate_limit"] = {
                "agent_id": agent_id,
                "rate": rate_cfg.get("rate", 10.0),
                "burst": rate_cfg.get("burst", 20),
                "status": "configured"
            }
        
        return results
    
    def define_agent_workflow_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> bool:
        """定义 Agent 工作流流水线
        
        调用 backend/modules/agent/langchain_inference_service 的流水线定义方法
        
        Args:
            name: 流水线名称
            steps: 步骤定义列表
            
        Returns:
            是否成功
        """
        if not self.inference_service:
            return False
        
        try:
            agent_instance_manager.define_workflow_pipeline(name, steps)
            return True
        except Exception as e:
            logger.error(f"Failed to define pipeline: {e}")
            return False
    
    def run_agent_workflow_pipeline(
        self,
        pipeline_name: str,
        initial_input: Any = None
    ) -> List[Dict[str, Any]]:
        """执行 Agent 工作流流水线
        
        调用 backend/modules/agent/langchain_inference_service 的流水线执行方法
        
        Args:
            pipeline_name: 流水线名称
            initial_input: 初始输入
            
        Returns:
            各步骤结果列表
        """
        if not self.inference_service:
            return [{"error": "Inference service not available"}]
        
        try:
            results = agent_instance_manager.run_workflow_pipeline(
                pipeline_name, initial_input
            )
            
            return [
                {
                    "step": i,
                    "status": "success" if r else "failed",
                    "output": r.output if r and hasattr(r, 'output') else None
                }
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Failed to run pipeline: {e}")
            return [{"error": str(e)}]
    
    def get_langgraph_health(self) -> Dict[str, Any]:
        """获取 LangGraph 系统健康状态
        
        调用 backend/modules/agent/langchain_inference_service 的健康检查方法
        
        Returns:
            健康状态字典
        """
        if not self.inference_service:
            return {"status": "unavailable", "message": "Inference service not available"}
        
        try:
            return agent_instance_manager.get_langgraph_health()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_langgraph_metrics(self) -> Dict[str, Any]:
        """获取 LangGraph 系统指标
        
        调用 backend/modules/agent/langchain_inference_service 的指标方法
        
        Returns:
            指标字典
        """
        if not self.inference_service:
            return {"error": "Inference service not available"}
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # 池指标
            metrics["pool"] = agent_instance_manager.get_langgraph_pool_metrics()
            
            # 工厂健康
            factory = agent_instance_manager.get_master_factory()
            metrics["factory"] = {"available": factory is not None}
            
            # 编排器健康
            metrics["orchestrator"] = agent_instance_manager.get_orchestrator_health()
            
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def create_agent_message(
        self,
        content: str,
        message_type: str = "human",
        name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """创建 Agent 消息
        
        调用 backend/modules/agent/langchain_inference_service 的消息创建方法
        
        Args:
            content: 消息内容
            message_type: 消息类型
            name: 发送者名称
            metadata: 元数据
            
        Returns:
            消息信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            message = agent_instance_manager.create_agent_message(
                content, message_type, name, metadata
            )
            
            if message:
                return {
                    "content": content,
                    "type": message_type,
                    "name": name,
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            return None
    
    def create_agent_memory_entry(
        self,
        content: Any,
        memory_type: str = "short_term",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """创建 Agent 记忆条目
        
        调用 backend/modules/agent/langchain_inference_service 的记忆创建方法
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            tags: 标签列表
            
        Returns:
            记忆信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            entry = agent_instance_manager.create_memory_entry(
                content, memory_type, importance, tags
            )
            
            if entry:
                return {
                    "content": str(content)[:100] + "..." if len(str(content)) > 100 else str(content),
                    "type": memory_type,
                    "importance": importance,
                    "tags": tags or [],
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create memory entry: {e}")
            return None
    
    def create_router(
        self,
        router_type: str = "priority",
        name: str = None
    ) -> Optional[Dict[str, Any]]:
        """创建路由器
        
        调用 backend/modules/agent/langchain_inference_service 的路由器创建方法
        
        Args:
            router_type: 路由器类型 (priority, weighted, load_balance, ab_test)
            name: 路由器名称
            
        Returns:
            路由器信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            router_name = name or f"{router_type}_router_{uuid.uuid4().hex[:8]}"
            
            if router_type == "priority":
                router = agent_instance_manager.create_priority_router(router_name)
            elif router_type == "weighted":
                router = agent_instance_manager.create_weighted_router(router_name)
            elif router_type == "load_balance":
                router = agent_instance_manager.create_load_balance_router(router_name)
            elif router_type == "ab_test":
                # AB测试路由暂时使用加权路由实现
                router = agent_instance_manager.create_weighted_router(router_name)
            else:
                return None
            
            if router:
                return {
                    "name": router_name,
                    "type": router_type,
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create router: {e}")
            return None
    
    def create_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """创建工具调用
        
        调用 backend/modules/agent/langchain_inference_service 的工具调用创建方法
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            工具调用信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            call_id = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
            tool_call = agent_instance_manager.create_tool_call_via_modules(
                tool_name, arguments, call_id
            )
            
            if tool_call:
                return {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create tool call: {e}")
            return None
    
    def create_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: str = None
    ) -> Optional[Dict[str, Any]]:
        """创建工具结果
        
        调用 backend/modules/agent/langchain_inference_service 的工具结果创建方法
        
        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            工具结果信息字典或 None
        """
        if not self.inference_service:
            return None
        
        try:
            tool_result = agent_instance_manager.create_tool_result_via_modules(
                tool_call_id, tool_name, result, success, error
            )
            
            if tool_result:
                return {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "result": result,
                    "success": success,
                    "error": error,
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create tool result: {e}")
            return None
    
    def run_workflow_with_events(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> Dict[str, Any]:
        """执行工作流并获取执行事件
        
        调用 backend/modules/agent/langchain_inference_service 的工作流执行方法
        
        Args:
            workflow: 工作流定义 [(agent_id, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            包含结果和事件的字典
        """
        if not self.inference_service:
            return {"results": [], "events": [], "error": "Inference service not available"}
        
        try:
            results, events = agent_instance_manager.run_workflow_with_events(
                workflow, stop_on_error
            )
            
            return {
                "results": [
                    {
                        "step": i,
                        "status": "success" if r else "failed",
                        "output": r.output if r and hasattr(r, 'output') else None
                    }
                    for i, r in enumerate(results)
                ],
                "events": [
                    {
                        "type": e.event_type if hasattr(e, 'event_type') else str(type(e)),
                        "data": e.data if hasattr(e, 'data') else str(e)
                    }
                    for e in events
                ],
                "total_steps": len(results),
                "total_events": len(events)
            }
        except Exception as e:
            logger.error(f"Failed to run workflow with events: {e}")
            return {"results": [], "events": [], "error": str(e)}
    
    def get_pool_health(self) -> Dict[str, Any]:
        """获取 Agent 池健康状态
        
        调用 backend/modules/agent/langchain_inference_service 的池健康检查方法
        
        Returns:
            健康状态字典
        """
        if not self.inference_service:
            return {"status": "unavailable", "message": "Inference service not available"}
        
        try:
            return agent_instance_manager.get_langgraph_pool_health()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def diagnose_langgraph_system(self) -> Dict[str, Any]:
        """执行 LangGraph 系统诊断
        
        综合调用 backend/modules/agent 的诊断能力
        
        Returns:
            诊断信息字典
        """
        diagnosis = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "capabilities": []
        }
        
        # 通过 inference_service 诊断
        if self.inference_service:
            # 健康状态
            health = self.get_langgraph_health()
            diagnosis["health"] = health
            
            # 指标
            metrics = self.get_langgraph_metrics()
            diagnosis["metrics"] = metrics
            
            # 池状态
            pool_health = self.get_pool_health()
            diagnosis["pool"] = pool_health
            
            diagnosis["capabilities"] = [
                "agent_creation",
                "workflow_execution",
                "parallel_execution",
                "checkpoint_management",
                "tool_management",
                "routing",
                "production_features"
            ]
        
        # 通过 instance_manager 诊断
        try:
            manager_diagnosis = agent_instance_manager.diagnose_langgraph_system()
            diagnosis["instance_manager"] = manager_diagnosis
        except Exception as e:
            diagnosis["instance_manager"] = {"error": str(e)}
        
        return diagnosis
    
    def get_available_agent_tools(self, category: str = None) -> List[Dict[str, Any]]:
        """获取可用的 Agent 工具
        
        Args:
            category: 工具类别过滤
            
        Returns:
            工具信息列表
        """
        try:
            if category:
                tools = get_tools_by_category(category)
            else:
                tools = get_builtin_tools()
            
            return [
                {
                    "name": t.name if hasattr(t, 'name') else str(t),
                    "description": t.description if hasattr(t, 'description') else "",
                    "category": category
                }
                for t in tools
            ]
        except Exception as e:
            logger.error(f"Failed to get agent tools: {e}")
            return []
    
    def create_multi_agent_system(
        self,
        name: str,
        agents: List[Dict[str, Any]],
        orchestration_type: str = "sequential"
    ) -> Optional[Dict[str, Any]]:
        """创建多 Agent 系统
        
        调用 backend/algo/langgraph 的多 Agent 系统创建能力
        
        Args:
            name: 系统名称
            agents: Agent 配置列表
            orchestration_type: 编排类型 (sequential, parallel, hierarchical)
            
        Returns:
            系统信息字典或 None
        """
        try:
            mas = create_multi_agent_system(
                name=name,
                agents=[],  # 会在后续添加
                orchestration_type=orchestration_type
            )
            
            if mas:
                return {
                    "name": name,
                    "orchestration_type": orchestration_type,
                    "agent_count": len(agents),
                    "created_at": datetime.utcnow().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create multi-agent system: {e}")
            return None
    
    def execute_with_state_graph(
        self,
        graph_name: str,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """使用 StateGraph 执行任务
        
        调用 backend/algo/langgraph 的 StateGraph 能力
        
        Args:
            graph_name: 图名称
            initial_state: 初始状态
            config: 图配置
            
        Returns:
            执行结果字典
        """
        try:
            # 创建图配置
            graph_config = GraphConfig(
                name=graph_name,
                enable_checkpointing=True
            ) if config is None else GraphConfig(**config)
            
            # 创建状态图
            graph = StateGraph(config=graph_config)
            
            return {
                "graph_name": graph_name,
                "initial_state": initial_state,
                "status": "initialized",
                "config": config
            }
        except Exception as e:
            logger.error(f"Failed to execute with state graph: {e}")
            return {"error": str(e)}
    
    def get_agent_state_checkpoint(
        self,
        agent_id: str,
        checkpoint_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """获取 Agent 状态检查点
        
        调用 backend/algo/langgraph 的检查点能力
        
        Args:
            agent_id: Agent ID
            checkpoint_id: 检查点 ID
            
        Returns:
            检查点信息字典或 None
        """
        try:
            # 通过 instance_manager 获取检查点
            checkpoint = agent_instance_manager.create_enhanced_checkpoint(
                agent_id, checkpoint_id
            )
            if checkpoint:
                return {
                    "agent_id": agent_id,
                    "checkpoint_id": checkpoint_id,
                    "status": "created"
                }
        except Exception as e:
            logger.error(f"Failed to get state checkpoint: {e}")
            return None
    
    def create_react_agent_instance(
        self,
        name: str,
        user_id: str,
        tools: List[Any] = None,
        llm_client: Any = None
    ) -> Optional[Dict[str, Any]]:
        """创建 ReAct Agent 实例
        
        调用 backend/algo/langgraph 的 ReAct Agent 创建能力
        
        Args:
            name: Agent 名称
            user_id: 用户 ID
            tools: 工具列表
            llm_client: LLM 客户端
            
        Returns:
            Agent 信息字典或 None
        """
        return self.create_langgraph_agent(
            agent_type="react",
            name=name,
            user_id=user_id,
            tools=tools,
            llm_client=llm_client
        )
    
    def create_plan_execute_agent_instance(
        self,
        name: str,
        user_id: str,
        tools: List[Any] = None,
        llm_client: Any = None
    ) -> Optional[Dict[str, Any]]:
        """创建 Plan-Execute Agent 实例
        
        调用 backend/algo/langgraph 的 Plan-Execute Agent 创建能力
        
        Args:
            name: Agent 名称
            user_id: 用户 ID
            tools: 工具列表
            llm_client: LLM 客户端
            
        Returns:
            Agent 信息字典或 None
        """
        return self.create_langgraph_agent(
            agent_type="plan_execute",
            name=name,
            user_id=user_id,
            tools=tools,
            llm_client=llm_client
        )
    
    def get_instance_manager_status(self) -> Dict[str, Any]:
        """获取实例管理器状态
        
        调用 backend/services/agent_instance_manager 获取状态
        
        Returns:
            状态信息字典
        """
        try:
            return {
                "status": "healthy",
                "instance_count": len(agent_instance_manager.agent_instances),
                "langgraph_health": agent_instance_manager.get_langgraph_health(),
                "pool_metrics": agent_instance_manager.get_langgraph_pool_metrics()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}