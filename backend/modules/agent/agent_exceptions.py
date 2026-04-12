"""智能体模块异常定义

定义智能体模块相关的自定义异常，包括：
- 基础 Agent 异常
- LangGraph 执行异常
- 状态管理异常
- 工具调用异常
- 检查点异常
- 策略异常
"""

import sys
import os
from typing import Optional, Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, ResourceNotFoundError, BusinessLogicError


# ==================== 基础异常 ====================

class AgentError(BusinessLogicError):
    """智能体模块基础异常"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "agent_id": self.agent_id,
            "details": self.details
        }


class AgentNotFoundError(AgentError, ResourceNotFoundError):
    """智能体不存在异常"""
    pass


class AgentValidationError(AgentError, ValidationError):
    """智能体验证异常"""
    pass


class AgentBusinessLogicError(AgentError, BusinessLogicError):
    """智能体业务逻辑异常"""
    pass


# ==================== 执行异常 ====================

class AgentExecutionError(AgentError):
    """Agent 执行异常
    
    在 Agent 执行过程中发生的错误。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 iteration: int = 0, node_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.iteration = iteration
        self.node_name = node_name
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "iteration": self.iteration,
            "node_name": self.node_name
        })
        return result


class AgentTimeoutError(AgentExecutionError):
    """Agent 超时异常
    
    当 Agent 执行超过指定时间限制时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 timeout_seconds: float = 0.0, elapsed_seconds: float = 0.0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details=details)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": self.elapsed_seconds
        })
        return result


class MaxIterationsExceededError(AgentExecutionError):
    """最大迭代次数超限异常
    
    当 Agent 执行迭代次数超过最大限制时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 max_iterations: int = 0, current_iteration: int = 0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, iteration=current_iteration, details=details)
        self.max_iterations = max_iterations
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "max_iterations": self.max_iterations
        })
        return result


class AgentTerminatedError(AgentExecutionError):
    """Agent 被终止异常
    
    当 Agent 被外部强制终止时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 reason: str = "unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details=details)
        self.reason = reason


class AgentCancelledError(AgentExecutionError):
    """Agent 被取消异常
    
    当 Agent 执行被用户或系统取消时抛出。
    """
    pass


# ==================== 状态异常 ====================

class StateError(AgentError):
    """状态相关异常基类"""
    pass


class StateValidationError(StateError, ValidationError):
    """状态验证异常
    
    当状态数据不符合验证规则时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 validation_errors: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.validation_errors = validation_errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["validation_errors"] = self.validation_errors
        return result


class StateTransitionError(StateError):
    """状态转换异常
    
    当状态转换无效或不允许时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 from_status: Optional[str] = None, to_status: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.from_status = from_status
        self.to_status = to_status
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "from_status": self.from_status,
            "to_status": self.to_status
        })
        return result


class StateNotFoundError(StateError, ResourceNotFoundError):
    """状态不存在异常
    
    当尝试访问不存在的状态时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 thread_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.thread_id = thread_id


class StateSerializationError(StateError):
    """状态序列化异常
    
    当状态序列化或反序列化失败时抛出。
    """


# ==================== 工具异常 ====================

class ToolError(AgentError):
    """工具相关异常基类"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError, ResourceNotFoundError):
    """工具不存在异常"""


class ToolExecutionError(ToolError):
    """工具执行异常
    
    当工具执行失败时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 tool_input: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, tool_name, details)
        self.tool_input = tool_input
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "original_error": str(self.original_error) if self.original_error else None
        })
        return result


class ToolValidationError(ToolError, ValidationError):
    """工具参数验证异常
    
    当工具参数不符合要求时抛出。
    """
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 parameter_errors: Optional[Dict[str, str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, tool_name, details)
        self.parameter_errors = parameter_errors or {}


class ToolTimeoutError(ToolError):
    """工具超时异常"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 timeout_seconds: float = 0.0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, tool_name, details)
        self.timeout_seconds = timeout_seconds


class ToolRateLimitError(ToolError):
    """工具频率限制异常"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 retry_after: float = 0.0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, tool_name, details)
        self.retry_after = retry_after


# ==================== 检查点异常 ====================

class CheckpointError(AgentError):
    """检查点相关异常基类"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 checkpoint_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.checkpoint_id = checkpoint_id


class CheckpointNotFoundError(CheckpointError, ResourceNotFoundError):
    """检查点不存在异常"""


class CheckpointSaveError(CheckpointError):
    """检查点保存异常"""


class CheckpointLoadError(CheckpointError):
    """检查点加载异常"""


class CheckpointCorruptedError(CheckpointError):
    """检查点损坏异常"""


# ==================== 策略异常 ====================

class StrategyError(AgentError):
    """策略相关异常基类"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 strategy_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.strategy_type = strategy_type


class StrategyNotFoundError(StrategyError, ResourceNotFoundError):
    """策略不存在异常"""


class StrategyExecutionError(StrategyError):
    """策略执行异常"""


class StrategyConfigurationError(StrategyError, ValidationError):
    """策略配置异常"""


# ==================== 图执行异常 ====================

class GraphError(AgentError):
    """图相关异常基类"""


class GraphBuildError(GraphError):
    """图构建异常"""


class GraphExecutionError(GraphError):
    """图执行异常"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 node_name: Optional[str] = None,
                 edge_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.node_name = node_name
        self.edge_name = edge_name


class GraphCycleDetectedError(GraphError):
    """图循环检测异常
    
    当检测到无限循环时抛出。
    """


class InvalidNodeError(GraphError):
    """无效节点异常"""


class InvalidEdgeError(GraphError):
    """无效边异常"""


# ==================== 回调异常 ====================

class CallbackError(AgentError):
    """回调相关异常"""
 
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 callback_name: Optional[str] = None,
                 event_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.callback_name = callback_name
        self.event_type = event_type


# ==================== 多 Agent 异常 ====================

class MultiAgentError(AgentError):
    """多 Agent 系统异常基类"""


class AgentCoordinationError(MultiAgentError):
    """Agent 协调异常"""
 
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 involved_agents: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.involved_agents = involved_agents or []


class AgentCommunicationError(MultiAgentError):
    """Agent 通信异常"""


# ==================== LLM 异常 ====================

class LLMError(AgentError):
    """LLM 相关异常基类"""
 
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 model: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.model = model


class LLMConnectionError(LLMError):
    """LLM 连接异常"""


class LLMResponseError(LLMError):
    """LLM 响应异常"""


class LLMRateLimitError(LLMError):
    """LLM 频率限制异常"""
 
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 model: Optional[str] = None,
                 retry_after: float = 0.0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, model, details)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """LLM 上下文长度超限异常"""
  
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 model: Optional[str] = None,
                 max_tokens: int = 0, requested_tokens: int = 0,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, model, details)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


# ==================== 会话异常 ====================

class SessionError(AgentError):
    """会话相关异常基类"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, agent_id, details)
        self.session_id = session_id


class SessionNotFoundError(SessionError, ResourceNotFoundError):
    """会话不存在异常"""


class SessionExpiredError(SessionError):
    """会话过期异常"""


class SessionLimitExceededError(SessionError):
    """会话数量超限异常"""


# ==================== 工具函数 ====================

def handle_agent_error(error: Exception, agent_id: Optional[str] = None) -> AgentError:
    """将通用异常转换为 AgentError
    
    Args:
        error: 原始异常
        agent_id: Agent ID
        
    Returns:
        AgentError 实例
    """
    if isinstance(error, AgentError):
        return error
   
    # 根据异常类型转换
    error_message = str(error)
    error_type = type(error).__name__
 
    if "timeout" in error_message.lower():
        return AgentTimeoutError(error_message, agent_id)
    elif "not found" in error_message.lower():
        return AgentNotFoundError(error_message, agent_id)
    elif "validation" in error_message.lower():
        return AgentValidationError(error_message, agent_id)
    else:
        return AgentExecutionError(
            f"Execution error ({error_type}): {error_message}",
            agent_id,
            details={"original_error_type": error_type}
        )
