"""LangChain 推理服务

提供基于 LangChain 的智能推理和会话管理功能。
支持多种 LLM 提供者：本地模型、ChatGPT、DeepSeek 等。
"""

import asyncio
import json
import logging
import os
import uuid
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult, Generation



from .local_model_service import get_local_model_service
from .api_config_manager import APIConfigManager, APIProvider
from .chatgpt_api_client import ChatGPTAPIClient, ChatMessage
from .deepseek_api_client import DeepSeekAPIClient, DeepSeekMessage

logger = logging.getLogger(__name__)


class MultiProviderLLM(BaseLanguageModel):
    """多提供者 LLM 包装器，支持本地模型、ChatGPT、DeepSeek 等"""

    # 配置允许任意类型
    class Config:
        """Pydantic 配置"""
        arbitrary_types_allowed = True

    provider: str = "local"
    model_name: Optional[str] = None
    api_config_manager: Optional[Any] = None  # 使用 Any 避免 Pydantic 验证问题
    model_service: Optional[Any] = None
    chatgpt_client: Optional[Any] = None
    deepseek_client: Optional[Any] = None

    def __init__(self,
                 provider: str = "local",
                 model_name: str = None,
                 api_config_manager: APIConfigManager = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.provider = provider
        self.model_name = model_name
        self.api_config_manager = api_config_manager or APIConfigManager()

        # 初始化相应的客户端
        self.model_service = None
        self.chatgpt_client = None
        self.deepseek_client = None

        self._init_clients()

    def _init_clients(self):
        """初始化客户端"""
        if self.provider == "local":
            self.model_service = get_local_model_service()
        elif self.provider == "chatgpt":
            config = self.api_config_manager.get_config(APIProvider.OPENAI)
            if config:
                self.chatgpt_client = ChatGPTAPIClient(config)
        elif self.provider == "deepseek":
            config = self.api_config_manager.get_config(APIProvider.DEEPSEEK)
            if config:
                self.deepseek_client = DeepSeekAPIClient(config)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """同步调用"""
        # 在异步环境中运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._acall(prompt, stop, **kwargs)
            )
            return result or ""
        finally:
            loop.close()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """异步调用"""
        try:
            if self.provider == "local" and self.model_service:
                result = await self.model_service.generate_response(prompt, self.model_name, **kwargs)
                return result or ""

            elif self.provider == "chatgpt" and self.chatgpt_client:
                # 将 prompt 转换为 ChatMessage
                message = ChatMessage(role="user", content=prompt)
                response = await self.chatgpt_client.simple_chat(message, **kwargs)
                return response.content if response else ""

            elif self.provider == "deepseek" and self.deepseek_client:
                # 将 prompt 转换为 DeepSeekMessage
                message = DeepSeekMessage(role="user", content=prompt)
                response = await self.deepseek_client.simple_chat(message, **kwargs)
                return response.content if response else ""

            else:
                logger.error(f"Provider {self.provider} not available or not configured")
                return ""

        except Exception as e:
            logger.error(f"Failed to generate response with {self.provider}: {str(e)}")
            return ""

    @property
    def _llm_type(self) -> str:
        return f"multi_provider_{self.provider}_llm"

    def generate_prompt(self, prompts, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """生成提示"""
        generations = []
        for prompt in prompts:
            # 将 PromptValue 转换为字符串
            prompt_str = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
            result = self._call(prompt_str, stop, **kwargs)
            generations.append([Generation(text=result)])

        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """异步生成提示"""
        generations = []
        for prompt in prompts:
            # 将 PromptValue 转换为字符串
            prompt_str = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
            result = await self._acall(prompt_str, stop, **kwargs)
            generations.append([Generation(text=result)])

        return LLMResult(generations=generations)

    def predict(self, text: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """预测"""
        return self._call(text, stop, **kwargs)

    async def apredict(self, text: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """异步预测"""
        return await self._acall(text, stop, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        """预测消息"""
        # 将消息转换为文本
        text = "\n".join([msg.content for msg in messages])
        result = self._call(text, stop, **kwargs)
        return AIMessage(content=result)

    async def apredict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                                **kwargs) -> BaseMessage:
        """异步预测消息"""
        # 将消息转换为文本
        text = "\n".join([msg.content for msg in messages])
        result = await self._acall(text, stop, **kwargs)
        return AIMessage(content=result)

    def invoke(self, input_data, config=None, **kwargs):
        """调用模型"""
        if isinstance(input_data, str):
            return self._call(input_data, **kwargs)
        elif isinstance(input_data, list):
            # 处理消息列表
            text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in input_data])
            return self._call(text, **kwargs)
        else:
            return self._call(str(input_data), **kwargs)


class CustomLLM(BaseLanguageModel):
    """自定义 LLM 包装器，用于集成本地模型服务（保持向后兼容）"""

    model_name: Optional[str] = None
    model_service: Optional[Any] = None

    def __init__(self, model_service, model_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_service = model_service
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """同步调用"""
        # 在异步环境中运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.model_service.generate_response(prompt, self.model_name, **kwargs)
            )
            return result or ""
        finally:
            loop.close()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """异步调用"""
        result = await self.model_service.generate_response(prompt, self.model_name, **kwargs)
        return result or ""

    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"

    def generate_prompt(self, prompts, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """生成提示"""
        generations = []
        for prompt in prompts:
            # 将 PromptValue 转换为字符串
            prompt_str = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
            result = self._call(prompt_str, stop, **kwargs)
            generations.append([Generation(text=result)])

        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """异步生成提示"""
        generations = []
        for prompt in prompts:
            # 将 PromptValue 转换为字符串
            prompt_str = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
            result = await self._acall(prompt_str, stop, **kwargs)
            generations.append([Generation(text=result)])

        return LLMResult(generations=generations)

    def predict(self, text: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """预测"""
        return self._call(text, stop, **kwargs)

    async def apredict(self, text: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """异步预测"""
        return await self._acall(text, stop, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        """预测消息"""
        # 将消息转换为文本
        text = "\n".join([msg.content for msg in messages])
        result = self._call(text, stop, **kwargs)
        return AIMessage(content=result)

    async def apredict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                                **kwargs) -> BaseMessage:
        """异步预测消息"""
        # 将消息转换为文本
        text = "\n".join([msg.content for msg in messages])
        result = await self._acall(text, stop, **kwargs)
        return AIMessage(content=result)

    def invoke(self, input_data, config=None, **kwargs):
        """调用模型"""
        if isinstance(input_data, str):
            return self._call(input_data, **kwargs)
        elif isinstance(input_data, list):
            # 处理消息列表
            text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in input_data])
            return self._call(text, **kwargs)
        else:
            return self._call(str(input_data), **kwargs)


class ConversationSession:
    """会话管理类"""

    def __init__(self,
                 session_id: str,
                 user_id: str,
                 agent_id: str,
                 memory_type: str = "buffer",
                 max_token_limit: int = 2000):
        self.session_id = session_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit

        # 初始化记忆
        self.memory = self._create_memory()
        self.messages: List[BaseMessage] = []
        self.context: Dict[str, Any] = {}

    def _create_memory(self):
        """创建记忆对象"""
        if self.memory_type == "summary":
            return ConversationSummaryBufferMemory(
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
        else:
            return ConversationBufferMemory(
                return_messages=True
            )

    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        if self.memory:
            if isinstance(message, HumanMessage):
                self.memory.chat_memory.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                self.memory.chat_memory.add_ai_message(message.content)
        self.last_activity = datetime.now()

    def get_messages(self) -> List[BaseMessage]:
        """获取消息历史"""
        return self.messages

    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return self.context

    def update_context(self, context: Dict[str, Any]):
        """更新上下文"""
        self.context.update(context)
        self.last_activity = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'memory_type': self.memory_type,
            'max_token_limit': self.max_token_limit,
            'messages': [
                {
                    'type': type(msg).__name__,
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                }
                for msg in self.messages
            ],
            'context': self.context
        }


class LangChainInferenceService:
    """LangChain 推理服务"""

    def __init__(self):
        self.model_service = get_local_model_service()
        self.api_config_manager = APIConfigManager()
        self.sessions: Dict[str, ConversationSession] = {}
        self.redis_client = None
        self.session_timeout = timedelta(hours=24)  # 会话超时时间

        # 支持的提供者
        self.supported_providers = ["local", "chatgpt", "deepseek"]

        # 初始化 Redis 连接
        self._init_redis()

        # 默认提示模板
        self.default_system_prompt = """你是一个智能训练助手，专门帮助用户管理和优化机器学习模型训练任务。

你的主要能力包括：
1. 创建和管理训练会话
2. 监控训练进度和性能指标
3. 提供训练优化建议
4. 解答训练相关问题
5. 协助模型部署和评估

请根据用户的问题和上下文信息，提供准确、有用的回答。如果需要执行具体的训练操作，请明确说明需要的参数和步骤。

当前会话上下文：
{context}

对话历史：
{history}

用户问题：{input}

请提供详细、专业的回答："""

    def _init_redis(self):
        """初始化 Redis 连接"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis connection established for session persistence")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    async def create_session(self,
                             user_id: str,
                             agent_id: str,
                             memory_type: str = "buffer",
                             max_token_limit: int = 2000) -> str:
        """创建新的会话"""
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            max_token_limit=max_token_limit
        )

        self.sessions[session_id] = session

        # 持久化到 Redis
        if self.redis_client:
            try:
                await self._save_session_to_redis(session)
            except Exception as e:
                logger.warning(f"Failed to save session to Redis: {str(e)}")

        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        # 先从内存中查找
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # 检查是否超时
            if datetime.now() - session.last_activity > self.session_timeout:
                await self.delete_session(session_id)
                return None
            return session

        # 从 Redis 中恢复
        if self.redis_client:
            try:
                session = await self._load_session_from_redis(session_id)
                if session:
                    self.sessions[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to load session from Redis: {str(e)}")

        return None

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        # 从内存中删除
        if session_id in self.sessions:
            del self.sessions[session_id]

        # 从 Redis 中删除
        if self.redis_client:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logger.warning(f"Failed to delete session from Redis: {str(e)}")

        logger.info(f"Deleted session: {session_id}")
        return True

    async def chat(self,
                   session_id: str,
                   message: str,
                   context: Optional[Dict[str, Any]] = None,
                   model_name: Optional[str] = None,
                   provider: str = "local") -> Optional[str]:
        """进行对话"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None

        try:
            # 验证提供者
            if provider not in self.supported_providers:
                logger.error(f"Unsupported provider: {provider}")
                return None

            # 更新上下文
            if context:
                session.update_context(context)

            # 添加用户消息
            user_message = HumanMessage(content=message)
            session.add_message(user_message)

            # 创建多提供者 LLM
            llm = MultiProviderLLM(
                provider=provider,
                model_name=model_name,
                api_config_manager=self.api_config_manager
            )

            # 简化提示模板，只使用 history 和 input
            simplified_prompt = """你是一个智能训练助手，专门帮助用户管理和优化机器学习模型训练任务。

你的主要能力包括：
1. 创建和管理训练会话
2. 监控训练进度和性能指标
3. 提供训练优化建议
4. 解答训练相关问题
5. 协助模型部署和评估

对话历史：
{history}

用户问题：{input}

请提供详细、专业的回答："""

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=simplified_prompt
            )

            conversation = ConversationChain(
                llm=llm,
                memory=session.memory,
                prompt=prompt,
                verbose=True
            )

            # 生成响应
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conversation.predict(input=message)
            )

            # 添加 AI 响应
            ai_message = AIMessage(content=response)
            session.add_message(ai_message)

            # 保存会话
            if self.redis_client:
                try:
                    await self._save_session_to_redis(session)
                except Exception as e:
                    logger.warning(f"Failed to save session to Redis: {str(e)}")

            return response

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return None

    async def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取会话历史"""
        session = await self.get_session(session_id)
        if not session:
            return None

        history = []
        for msg in session.get_messages():
            history.append({
                'type': type(msg).__name__,
                'content': msg.content,
                'timestamp': datetime.now().isoformat()
            })

        return history

    async def clear_session_history(self, session_id: str) -> bool:
        """清除会话历史"""
        session = await self.get_session(session_id)
        if not session:
            return False

        session.messages.clear()
        if session.memory:
            session.memory.clear()

        # 保存会话
        if self.redis_client:
            try:
                await self._save_session_to_redis(session)
            except Exception as e:
                logger.warning(f"Failed to save session to Redis: {str(e)}")

        return True

    async def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """列出用户的所有会话"""
        user_sessions = []

        # 从内存中查找
        for session in self.sessions.values():
            if session.user_id == user_id:
                user_sessions.append(session.to_dict())

        # 从 Redis 中查找
        if self.redis_client:
            try:
                pattern = f"session:*"
                keys = self.redis_client.keys(pattern)
                for key in keys:
                    session_data = self.redis_client.hgetall(key)
                    if session_data.get('user_id') == user_id:
                        session_id = key.split(':')[1]
                        if session_id not in self.sessions:
                            user_sessions.append(json.loads(session_data.get('data', '{}')))
            except Exception as e:
                logger.warning(f"Failed to list sessions from Redis: {str(e)}")

        return user_sessions

    async def _save_session_to_redis(self, session: ConversationSession):
        """保存会话到 Redis"""
        if not self.redis_client:
            return

        key = f"session:{session.session_id}"
        data = {
            'user_id': session.user_id,
            'agent_id': session.agent_id,
            'data': json.dumps(session.to_dict())
        }

        self.redis_client.hset(key, mapping=data)
        # 设置过期时间
        self.redis_client.expire(key, int(self.session_timeout.total_seconds()))

    async def _load_session_from_redis(self, session_id: str) -> Optional[ConversationSession]:
        """从 Redis 加载会话"""
        if not self.redis_client:
            return None

        key = f"session:{session_id}"
        session_data = self.redis_client.hgetall(key)

        if not session_data:
            return None

        try:
            data = json.loads(session_data.get('data', '{}'))
            session = ConversationSession(
                session_id=data['session_id'],
                user_id=data['user_id'],
                agent_id=data['agent_id'],
                memory_type=data.get('memory_type', 'buffer'),
                max_token_limit=data.get('max_token_limit', 2000)
            )

            # 恢复消息历史
            for msg_data in data.get('messages', []):
                if msg_data['type'] == 'HumanMessage':
                    msg = HumanMessage(content=msg_data['content'])
                elif msg_data['type'] == 'AIMessage':
                    msg = AIMessage(content=msg_data['content'])
                else:
                    continue
                session.add_message(msg)

            # 恢复上下文
            session.context = data.get('context', {})

            return session

        except Exception as e:
            logger.error(f"Failed to deserialize session data: {str(e)}")
            return None

    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        current_time = datetime.now()

        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.delete_session(session_id)

        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def multi_turn_chat(self,
                              session_id: str,
                              messages: List[Dict[str, str]],
                              provider: str = "local",
                              model_name: Optional[str] = None,
                              **kwargs) -> Optional[str]:
        """多轮对话"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None

        if provider not in self.supported_providers:
            logger.error(f"Unsupported provider: {provider}")
            return None

        try:
            # 使用本地模型服务的多轮对话功能
            if provider == "local":
                return await self.model_service.multi_turn_chat(
                    conversation_history=messages[:-1],
                    new_message=messages[-1]["content"],
                    model_name=model_name,
                    **kwargs
                )

            # 对于云端 API，直接调用相应的客户端
            elif provider == "chatgpt":
                config = self.api_config_manager.get_config(APIProvider.OPENAI)
                if not config:
                    logger.error("ChatGPT configuration not found")
                    return None

                client = ChatGPTAPIClient(config)
                chat_messages = [ChatMessage(role=msg["role"], content=msg["content"])
                                 for msg in messages[:-1]]
                response = await client.multi_turn_chat(
                    chat_messages,
                    messages[-1]["content"],
                    **kwargs
                )
                return response.content if response else None

            elif provider == "deepseek":
                config = self.api_config_manager.get_config(APIProvider.DEEPSEEK)
                if not config:
                    logger.error("DeepSeek configuration not found")
                    return None

                client = DeepSeekAPIClient(config)
                deepseek_messages = [DeepSeekMessage(role=msg["role"], content=msg["content"])
                                     for msg in messages[:-1]]
                response = await client.multi_turn_chat(
                    deepseek_messages,
                    messages[-1]["content"],
                    **kwargs
                )
                return response.content if response else None

        except Exception as e:
            logger.error(f"Multi-turn chat failed with {provider}: {str(e)}")
            return None

    async def stream_chat(self,
                          session_id: str,
                          messages: List[Dict[str, str]],
                          provider: str = "local",
                          model_name: Optional[str] = None,
                          **kwargs):
        """流式聊天"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return

        if provider not in self.supported_providers:
            logger.error(f"Unsupported provider: {provider}")
            return

        try:
            # 使用本地模型服务的流式聊天功能
            if provider == "local":
                async for chunk in self.model_service.stream_chat(
                        messages=messages,
                        model_name=model_name,
                        **kwargs
                ):
                    yield chunk

            # 对于云端 API，直接调用相应的客户端
            elif provider == "chatgpt":
                config = self.api_config_manager.get_config(APIProvider.OPENAI)
                if not config:
                    logger.error("ChatGPT configuration not found")
                    return

                client = ChatGPTAPIClient(config)
                chat_messages = [ChatMessage(role=msg["role"], content=msg["content"])
                                 for msg in messages]
                async for chunk in client.stream_chat(chat_messages, **kwargs):
                    yield chunk

            elif provider == "deepseek":
                config = self.api_config_manager.get_config(APIProvider.DEEPSEEK)
                if not config:
                    logger.error("DeepSeek configuration not found")
                    return

                client = DeepSeekAPIClient(config)
                deepseek_messages = [DeepSeekMessage(role=msg["role"], content=msg["content"])
                                     for msg in messages]
                async for chunk in client.stream_chat(deepseek_messages, **kwargs):
                    yield chunk

        except Exception as e:
            logger.error(f"Stream chat failed with {provider}: {str(e)}")

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """获取可用的提供者列表"""
        providers = []

        # 本地模型
        providers.append({
            "name": "local",
            "display_name": "本地模型",
            "available": True,
            "models": list(self.model_service.model_configs.keys()) if self.model_service else []
        })

        # ChatGPT
        chatgpt_config = self.api_config_manager.get_config(APIProvider.OPENAI)
        providers.append({
            "name": "chatgpt",
            "display_name": "ChatGPT",
            "available": chatgpt_config is not None,
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] if chatgpt_config else []
        })

        # DeepSeek
        deepseek_config = self.api_config_manager.get_config(APIProvider.DEEPSEEK)
        providers.append({
            "name": "deepseek",
            "display_name": "DeepSeek",
            "available": deepseek_config is not None,
            "models": ["deepseek-chat", "deepseek-coder"] if deepseek_config else []
        })

        return providers

    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """获取提供者状态"""
        if provider == "local":
            return {
                "name": provider,
                "available": True,
                "models": list(self.model_service.model_configs.keys()) if self.model_service else [],
                "default_model": self.model_service.default_model if self.model_service else None
            }

        elif provider == "chatgpt":
            config = self.api_config_manager.get_config(APIProvider.OPENAI)
            return {
                "name": provider,
                "available": config is not None,
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] if config else [],
                "default_model": "gpt-3.5-turbo" if config else None
            }

        elif provider == "deepseek":
            config = self.api_config_manager.get_config(APIProvider.DEEPSEEK)
            return {
                "name": provider,
                "available": config is not None,
                "models": ["deepseek-chat", "deepseek-coder"] if config else [],
                "default_model": "deepseek-chat" if config else None
            }

        else:
            return {
                "name": provider,
                "available": False,
                "models": [],
                "default_model": None
            }

    async def configure_provider(self, provider: str, config: Dict[str, Any]) -> bool:
        """配置提供者"""
        try:
            if provider == "chatgpt":
                api_config = self.api_config_manager.create_config(
                    provider=APIProvider.OPENAI,
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                    **config
                )
                self.api_config_manager.add_config(api_config)
                return True

            elif provider == "deepseek":
                api_config = self.api_config_manager.create_config(
                    provider=APIProvider.DEEPSEEK,
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                    **config
                )
                self.api_config_manager.add_config(api_config)
                return True

            else:
                logger.error(f"Unsupported provider for configuration: {provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to configure provider {provider}: {str(e)}")
            return False


# 全局实例
_langchain_inference_service = None


def get_langchain_inference_service() -> LangChainInferenceService:
    """获取 LangChain 推理服务实例"""
    global _langchain_inference_service
    if _langchain_inference_service is None:
        _langchain_inference_service = LangChainInferenceService()
    return _langchain_inference_service
