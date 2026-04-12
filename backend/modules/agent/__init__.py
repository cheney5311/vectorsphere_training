"""
Agent 模块

该模块提供智能体相关的功能，包括：
- 训练助手智能体
- 推理服务
- 会话管理
- API 配置管理
- LangGraph 集成

与 LangGraph 的集成：
- 支持多种 Agent 类型（ReAct、Plan&Execute、Reflexion 等）
- 支持策略模式和回调系统
- 支持工具调用和状态管理
- 支持检查点和恢复
"""

# 向后兼容：旧的导入路径
from ...services.chatgpt_api_client import (
    ChatGPTAPIClient, ChatMessage, ChatResponse, UsageStats, chatgpt_client
)
from ...services.deepseek_api_client import (
    DeepSeekAPIClient, DeepSeekMessage, deepseek_client
)

# Agent 类型
from .agent_type import (
    AgentType, LangGraphAgentType, CollaborationMode,
    ExecutionMode, AgentCapability, AgentStatus,
    AGENT_TYPE_MAPPING, AGENT_CAPABILITIES,
    get_langgraph_type, get_agent_capabilities, get_recommended_tools,
    is_multi_agent_type, supports_streaming, supports_memory, get_default_config
)

# 异常
from .agent_exceptions import (
    AgentError, AgentNotFoundError, AgentValidationError, AgentBusinessLogicError,
    AgentExecutionError, AgentTimeoutError, MaxIterationsExceededError,
    AgentTerminatedError, AgentCancelledError,
    StateError, StateValidationError, StateTransitionError, StateNotFoundError,
    StateSerializationError,
    ToolError, ToolNotFoundError, ToolExecutionError, ToolValidationError,
    ToolTimeoutError, ToolRateLimitError,
    CheckpointError, CheckpointNotFoundError, CheckpointSaveError,
    CheckpointLoadError, CheckpointCorruptedError,
    StrategyError, StrategyNotFoundError, StrategyExecutionError,
    StrategyConfigurationError,
    GraphError, GraphBuildError, GraphExecutionError, GraphCycleDetectedError,
    InvalidNodeError, InvalidEdgeError,
    CallbackError,
    MultiAgentError, AgentCoordinationError, AgentCommunicationError,
    LLMError, LLMConnectionError, LLMResponseError, LLMRateLimitError,
    LLMContextLengthError,
    SessionError, SessionNotFoundError, SessionExpiredError, SessionLimitExceededError,
    handle_agent_error
)

# API 配置管理
from .api_config_manager import (
    APIProvider, CheckpointerType, StrategyType as ConfigStrategyType,
    APIConfig, CheckpointerConfig, StrategyConfig, CallbackConfig,
    AgentInstanceConfig,
    APIConfigManager, get_api_config_manager, set_api_config_manager,
    api_config_manager,
    # 配置构建器
    build_api_config, build_agent_config
)

# 本地模型服务
from .local_model_service import (
    ModelType, ModelConfig, GenerationMetrics, LocalModelService
)

# 会话管理
from .session_history_manager import (
    MessageType, ChatMessage as SessionChatMessage, SessionInfo, SessionCheckpoint,
    SessionHistoryManager
)

# 推理服务（含 LangGraph 工业级集成）
from .langchain_inference_service import (
    CustomLLM, ConversationSession, LangChainInferenceService,
    get_langchain_inference_service, set_langchain_inference_service,
    reset_langchain_inference_service
)

# 训练助手
from .training_assistant_agent import (
    TrainingAssistantConfig, TrainingAssistantAgent
)

# 导出列表
__all__ = [
    # 向后兼容
    "ChatGPTAPIClient", "ChatMessage", "ChatResponse", "UsageStats", "chatgpt_client",
    "DeepSeekAPIClient", "DeepSeekMessage", "deepseek_client",
    
    # Agent 类型
    "AgentType", "LangGraphAgentType", "CollaborationMode",
    "ExecutionMode", "AgentCapability", "AgentStatus",
    "AGENT_TYPE_MAPPING", "AGENT_CAPABILITIES",
    "get_langgraph_type", "get_agent_capabilities", "get_recommended_tools",
    "is_multi_agent_type", "supports_streaming", "supports_memory", "get_default_config",
    
    # 异常
    "AgentError", "AgentNotFoundError", "AgentValidationError", "AgentBusinessLogicError",
    "AgentExecutionError", "AgentTimeoutError", "MaxIterationsExceededError",
    "AgentTerminatedError", "AgentCancelledError",
    "StateError", "StateValidationError", "StateTransitionError", "StateNotFoundError",
    "StateSerializationError",
    "ToolError", "ToolNotFoundError", "ToolExecutionError", "ToolValidationError",
    "ToolTimeoutError", "ToolRateLimitError",
    "CheckpointError", "CheckpointNotFoundError", "CheckpointSaveError",
    "CheckpointLoadError", "CheckpointCorruptedError",
    "StrategyError", "StrategyNotFoundError", "StrategyExecutionError",
    "StrategyConfigurationError",
    "GraphError", "GraphBuildError", "GraphExecutionError", "GraphCycleDetectedError",
    "InvalidNodeError", "InvalidEdgeError",
    "CallbackError",
    "MultiAgentError", "AgentCoordinationError", "AgentCommunicationError",
    "LLMError", "LLMConnectionError", "LLMResponseError", "LLMRateLimitError",
    "LLMContextLengthError",
    "SessionError", "SessionNotFoundError", "SessionExpiredError", "SessionLimitExceededError",
    "handle_agent_error",
    
    # API 配置
    "APIProvider", "CheckpointerType", "ConfigStrategyType",
    "APIConfig", "CheckpointerConfig", "StrategyConfig", "CallbackConfig",
    "AgentInstanceConfig",
    "APIConfigManager", "get_api_config_manager", "set_api_config_manager",
    "api_config_manager",
    "build_api_config", "build_agent_config",
    
    # 本地模型
    "ModelType", "ModelConfig", "GenerationMetrics", "LocalModelService",
    
    # 会话管理
    "MessageType", "SessionChatMessage", "SessionInfo", "SessionCheckpoint",
    "SessionHistoryManager",
    
    # 推理服务（含 LangGraph 工业级集成）
    "CustomLLM", "ConversationSession", "LangChainInferenceService",
    "get_langchain_inference_service", "set_langchain_inference_service",
    "reset_langchain_inference_service",
    
    # 训练助手
    "TrainingAssistantConfig", "TrainingAssistantAgent",
]
