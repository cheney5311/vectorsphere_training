"""
ChatGPT API 客户端
使用 OpenAI SDK 实现与 ChatGPT 的集成
支持异步调用、错误处理、重试机制和使用统计
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, AsyncGenerator

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .api_config_manager import api_config_manager, APIProvider

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """聊天消息数据类"""
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ChatResponse:
    """聊天响应数据类"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    timestamp: float


@dataclass
class UsageStats:
    """使用统计数据类"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0


class ChatGPTAPIClient:
    """ChatGPT API 客户端"""
    
    def __init__(self):
        self.config = api_config_manager.get_config(APIProvider.OPENAI.value)
        self.client: Optional[AsyncOpenAI] = None
        self.usage_stats = UsageStats()
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化 OpenAI 客户端"""
        if not self.config or not self.config.enabled:
            logger.warning("ChatGPT 配置未启用或不存在")
            return
        
        if not self.config.api_key:
            logger.warning("ChatGPT API 密钥未配置")
            return
        
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            logger.info("ChatGPT API 客户端初始化成功")
        except Exception as e:
            logger.error(f"ChatGPT API 客户端初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.client is not None and self.config and self.config.enabled
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse:
        """
        发送聊天完成请求
        
        Args:
            messages: 聊天消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式响应
            **kwargs: 其他参数
        
        Returns:
            ChatResponse: 聊天响应
        """
        if not self.is_available():
            raise RuntimeError("ChatGPT API 客户端不可用")
        
        start_time = time.time()
        
        try:
            # 准备请求参数
            request_params = {
                "model": model or self.config.model_name,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": stream,
                **kwargs
            }
            
            # 发送请求
            response = await self.client.chat.completions.create(**request_params)
            
            # 处理响应
            if stream:
                return self._handle_stream_response(response, start_time)
            else:
                return self._handle_response(response, start_time)
        
        except openai.RateLimitError as e:
            logger.warning(f"ChatGPT API 速率限制: {e}")
            self.usage_stats.failed_requests += 1
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"ChatGPT API 超时: {e}")
            self.usage_stats.failed_requests += 1
            raise
        except Exception as e:
            logger.error(f"ChatGPT API 请求失败: {e}")
            self.usage_stats.failed_requests += 1
            raise
        finally:
            self.usage_stats.total_requests += 1
    
    def _handle_response(self, response, start_time: float) -> ChatResponse:
        """处理普通响应"""
        response_time = time.time() - start_time
        
        choice = response.choices[0]
        usage = response.usage
        
        # 更新统计信息
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens += usage.total_tokens
        self.usage_stats.prompt_tokens += usage.prompt_tokens
        self.usage_stats.completion_tokens += usage.completion_tokens
        
        # 计算平均响应时间
        total_successful = self.usage_stats.successful_requests
        self.usage_stats.average_response_time = (
            (self.usage_stats.average_response_time * (total_successful - 1) + response_time) 
            / total_successful
        )
        
        # 估算成本（基于 GPT-3.5-turbo 定价）
        cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
        self.usage_stats.total_cost += cost
        
        return ChatResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "total_tokens": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens
            },
            finish_reason=choice.finish_reason,
            response_time=response_time,
            timestamp=time.time()
        )
    
    async def _handle_stream_response(self, response, start_time: float) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        try:
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            raise
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        计算 API 调用成本
        基于 GPT-3.5-turbo 的定价：$0.0015 / 1K prompt tokens, $0.002 / 1K completion tokens
        """
        prompt_cost = (prompt_tokens / 1000) * 0.0015
        completion_cost = (completion_tokens / 1000) * 0.002
        return prompt_cost + completion_cost
    
    async def simple_chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        简单聊天接口
        
        Args:
            message: 用户消息
            system_prompt: 系统提示
        
        Returns:
            str: AI 回复
        """
        messages = []
        
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        messages.append(ChatMessage(role="user", content=message))
        
        response = await self.chat_completion(messages)
        return response.content
    
    async def multi_turn_chat(
        self,
        conversation_history: List[ChatMessage],
        new_message: str
    ) -> ChatResponse:
        """
        多轮对话
        
        Args:
            conversation_history: 对话历史
            new_message: 新消息
        
        Returns:
            ChatResponse: 聊天响应
        """
        messages = conversation_history.copy()
        messages.append(ChatMessage(role="user", content=new_message))
        
        return await self.chat_completion(messages)
    
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天
        
        Args:
            messages: 聊天消息列表
            **kwargs: 其他参数
        
        Yields:
            str: 流式响应内容
        """
        if not self.is_available():
            raise RuntimeError("ChatGPT API 客户端不可用")
        
        start_time = time.time()
        
        try:
            request_params = {
                "model": self.config.model_name,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            response = await self.client.chat.completions.create(**request_params)
            
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"流式聊天失败: {e}")
            self.usage_stats.failed_requests += 1
            raise
        finally:
            self.usage_stats.total_requests += 1
    
    def get_usage_stats(self) -> UsageStats:
        """获取使用统计"""
        return self.usage_stats
    
    def reset_usage_stats(self):
        """重置使用统计"""
        self.usage_stats = UsageStats()
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_message = ChatMessage(role="user", content="Hello")
            response = await self.chat_completion([test_message], max_tokens=10)
            logger.info("ChatGPT API 连接测试成功")
            return True
        except Exception as e:
            logger.error(f"ChatGPT API 连接测试失败: {e}")
            return False
    
    def update_config(self):
        """更新配置"""
        self.config = api_config_manager.get_config(APIProvider.OPENAI.value)
        self._initialize_client()


# 全局 ChatGPT 客户端实例
chatgpt_client = ChatGPTAPIClient()