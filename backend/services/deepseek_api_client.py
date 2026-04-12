"""
DeepSeek API 客户端
使用 HTTP 请求实现与 DeepSeek API 的集成
支持异步调用、错误处理、重试机制和使用统计
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, AsyncGenerator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .api_config_manager import api_config_manager, APIProvider

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekMessage:
    """DeepSeek 消息数据类"""
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class DeepSeekResponse:
    """DeepSeek 响应数据类"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    timestamp: float


@dataclass
class DeepSeekUsageStats:
    """DeepSeek 使用统计数据类"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0


class DeepSeekAPIError(Exception):
    """DeepSeek API 错误"""
    pass


class DeepSeekRateLimitError(DeepSeekAPIError):
    """DeepSeek 速率限制错误"""
    pass


class DeepSeekTimeoutError(DeepSeekAPIError):
    """DeepSeek 超时错误"""
    pass


class DeepSeekAPIClient:
    """DeepSeek API 客户端"""
    
    def __init__(self):
        self.config = api_config_manager.get_config(APIProvider.DEEPSEEK.value)
        self.usage_stats = DeepSeekUsageStats()
        self.session: Optional[httpx.AsyncClient] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化 HTTP 客户端"""
        if not self.config or not self.config.enabled:
            logger.warning("DeepSeek 配置未启用或不存在")
            return
        
        if not self.config.api_key:
            logger.warning("DeepSeek API 密钥未配置")
            return
        
        try:
            self.session = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
            logger.info("DeepSeek API 客户端初始化成功")
        except Exception as e:
            logger.error(f"DeepSeek API 客户端初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.session is not None and self.config and self.config.enabled
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((DeepSeekRateLimitError, DeepSeekTimeoutError))
    )
    async def chat_completion(
        self,
        messages: List[DeepSeekMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> DeepSeekResponse:
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
            DeepSeekResponse: 聊天响应
        """
        if not self.is_available():
            raise RuntimeError("DeepSeek API 客户端不可用")
        
        start_time = time.time()
        
        try:
            # 准备请求数据
            request_data = {
                "model": model or self.config.model_name,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": stream,
                **kwargs
            }
            
            # 发送请求
            response = await self.session.post(
                "/chat/completions",
                json=request_data
            )
            
            # 检查响应状态
            await self._check_response_status(response)
            
            # 处理响应
            if stream:
                return await self._handle_stream_response(response, start_time)
            else:
                return await self._handle_response(response, start_time)
        
        except DeepSeekRateLimitError as e:
            logger.warning(f"DeepSeek API 速率限制: {e}")
            self.usage_stats.failed_requests += 1
            raise
        except DeepSeekTimeoutError as e:
            logger.warning(f"DeepSeek API 超时: {e}")
            self.usage_stats.failed_requests += 1
            raise
        except Exception as e:
            logger.error(f"DeepSeek API 请求失败: {e}")
            self.usage_stats.failed_requests += 1
            raise
        finally:
            self.usage_stats.total_requests += 1
    
    async def _check_response_status(self, response: httpx.Response):
        """检查响应状态"""
        if response.status_code == 200:
            return
        elif response.status_code == 429:
            raise DeepSeekRateLimitError("API 速率限制")
        elif response.status_code == 408:
            raise DeepSeekTimeoutError("API 请求超时")
        else:
            error_text = await response.aread()
            raise DeepSeekAPIError(f"API 请求失败: {response.status_code} - {error_text}")
    
    async def _handle_response(self, response: httpx.Response, start_time: float) -> DeepSeekResponse:
        """处理普通响应"""
        response_time = time.time() - start_time
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise DeepSeekAPIError(f"响应解析失败: {e}")
        
        if "error" in data:
            raise DeepSeekAPIError(f"API 错误: {data['error']}")
        
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        # 更新统计信息
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens += usage.get("total_tokens", 0)
        self.usage_stats.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage_stats.completion_tokens += usage.get("completion_tokens", 0)
        
        # 计算平均响应时间
        total_successful = self.usage_stats.successful_requests
        self.usage_stats.average_response_time = (
            (self.usage_stats.average_response_time * (total_successful - 1) + response_time) 
            / total_successful
        )
        
        # 估算成本（基于 DeepSeek 定价）
        cost = self._calculate_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        self.usage_stats.total_cost += cost
        
        return DeepSeekResponse(
            content=choice["message"]["content"],
            model=data.get("model", "deepseek-chat"),
            usage={
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)
            },
            finish_reason=choice.get("finish_reason", "stop"),
            response_time=response_time,
            timestamp=time.time()
        )
    
    async def _handle_stream_response(self, response: httpx.Response, start_time: float) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        try:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # 移除 "data: " 前缀
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            raise
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        计算 API 调用成本
        基于 DeepSeek 的定价（假设价格，需要根据实际定价调整）
        """
        # DeepSeek 的实际定价需要根据官方文档调整
        prompt_cost = (prompt_tokens / 1000) * 0.001  # 假设价格
        completion_cost = (completion_tokens / 1000) * 0.002  # 假设价格
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
            messages.append(DeepSeekMessage(role="system", content=system_prompt))
        
        messages.append(DeepSeekMessage(role="user", content=message))
        
        response = await self.chat_completion(messages)
        return response.content
    
    async def multi_turn_chat(
        self,
        conversation_history: List[DeepSeekMessage],
        new_message: str
    ) -> DeepSeekResponse:
        """
        多轮对话
        
        Args:
            conversation_history: 对话历史
            new_message: 新消息
        
        Returns:
            DeepSeekResponse: 聊天响应
        """
        messages = conversation_history.copy()
        messages.append(DeepSeekMessage(role="user", content=new_message))
        
        return await self.chat_completion(messages)
    
    async def stream_chat(
        self,
        messages: List[DeepSeekMessage],
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
            raise RuntimeError("DeepSeek API 客户端不可用")
        
        start_time = time.time()
        
        try:
            request_data = {
                "model": self.config.model_name,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            response = await self.session.post(
                "/chat/completions",
                json=request_data
            )
            
            await self._check_response_status(response)
            
            async for chunk in self._handle_stream_response(response, start_time):
                yield chunk
        
        except Exception as e:
            logger.error(f"流式聊天失败: {e}")
            self.usage_stats.failed_requests += 1
            raise
        finally:
            self.usage_stats.total_requests += 1
    
    def get_usage_stats(self) -> DeepSeekUsageStats:
        """获取使用统计"""
        return self.usage_stats
    
    def reset_usage_stats(self):
        """重置使用统计"""
        self.usage_stats = DeepSeekUsageStats()
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_message = DeepSeekMessage(role="user", content="Hello")
            response = await self.chat_completion([test_message], max_tokens=10)
            logger.info("DeepSeek API 连接测试成功")
            return True
        except Exception as e:
            logger.error(f"DeepSeek API 连接测试失败: {e}")
            return False
    
    def update_config(self):
        """更新配置"""
        self.config = api_config_manager.get_config(APIProvider.DEEPSEEK.value)
        if self.session:
            asyncio.create_task(self.session.aclose())
        self._initialize_client()
    
    async def close(self):
        """关闭客户端"""
        if self.session:
            await self.session.aclose()


# 全局 DeepSeek 客户端实例
deepseek_client = DeepSeekAPIClient()