"""Agent 类型定义模块

定义智能体类型枚举，包括：
- 基础业务 Agent 类型
- LangGraph Agent 类型
- 协作模式类型
"""

from enum import Enum
from typing import Dict, Any, List


class AgentType(Enum):
    """Agent 类型枚举 - 业务层面"""
    # 兼容现有定义
    BASIC = "basic"
    TRAINING_ASSISTANT = "training_assistant"
    DATA_PROCESSOR = "data_processor"
    MODEL_MANAGER = "model_manager"

    # 新增以匹配 AgentInstanceManager 使用
    DIALOGUE_MANAGER = "dialogue_manager"
    INFERENCE_EXECUTOR = "inference_executor"
    TRAINING_COORDINATOR = "training_coordinator"
    DECISION_SUPPORT = "decision_support"
    
    # 高级 Agent 类型
    RESEARCH_ASSISTANT = "research_assistant"
    CODE_ASSISTANT = "code_assistant"
    DATA_ANALYST = "data_analyst"
    CUSTOMER_SERVICE = "customer_service"
    CREATIVE_WRITER = "creative_writer"
    QA_SPECIALIST = "qa_specialist"


class LangGraphAgentType(Enum):
    """LangGraph Agent 类型枚举 - 框架层面
    
    对应 langgraph.agents 中的 Agent 实现。
    """
    # ReAct 类型 - 思考-行动循环
    REACT = "react"
    
    # 计划执行类型 - 先计划后执行
    PLAN_EXECUTE = "plan_execute"
    
    # 反思类型 - 带自我反思
    REFLEXION = "reflexion"
    
    # 多 Agent 系统
    MULTI_AGENT = "multi_agent"
    
    # 工具调用类型
    TOOL_CALLING = "tool_calling"
    
    # 对话类型 - 带记忆管理
    CONVERSATIONAL = "conversational"
    
    # 思维链类型
    CHAIN_OF_THOUGHT = "chain_of_thought"
    
    # 自问自答类型
    SELF_ASK = "self_ask"
    
    # 层级类型 - 管理者/工作者架构
    HIERARCHICAL = "hierarchical"
    
    # 工作流类型 - 复杂流程编排
    WORKFLOW = "workflow"


class CollaborationMode(Enum):
    """Agent 协作模式"""
    # 监督者模式 - 一个监督者 Agent 协调其他 Agent
    SUPERVISOR = "supervisor"
    
    # 轮询模式 - Agent 轮流执行
    ROUND_ROBIN = "round_robin"
    
    # 并行模式 - Agent 并行执行
    PARALLEL = "parallel"
    
    # 投票模式 - Agent 投票决策
    VOTING = "voting"
    
    # 竞争模式 - Agent 竞争最优解
    COMPETITIVE = "competitive"
    
    # 分层模式 - 分层协作
    HIERARCHICAL = "hierarchical"


class ExecutionMode(Enum):
    """执行模式"""
    # 同步执行
    SYNC = "sync"
    
    # 异步执行
    ASYNC = "async"
    
    # 流式执行
    STREAMING = "streaming"
    
    # 批量执行
    BATCH = "batch"


class AgentCapability(Enum):
    """Agent 能力枚举"""
    # 基础能力
    TEXT_GENERATION = "text_generation"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    
    # 高级能力
    PLANNING = "planning"
    REFLECTION = "reflection"
    MEMORY = "memory"
    LEARNING = "learning"
    
    # 专业能力
    CODE_EXECUTION = "code_execution"
    DATA_ANALYSIS = "data_analysis"
    WEB_SEARCH = "web_search"
    FILE_OPERATIONS = "file_operations"
    
    # 协作能力
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    HUMAN_IN_LOOP = "human_in_loop"


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"


# ==================== 类型映射 ====================

# 业务类型到 LangGraph 类型的映射
AGENT_TYPE_MAPPING: Dict[AgentType, LangGraphAgentType] = {
    AgentType.BASIC: LangGraphAgentType.REACT,
    AgentType.TRAINING_ASSISTANT: LangGraphAgentType.PLAN_EXECUTE,
    AgentType.DATA_PROCESSOR: LangGraphAgentType.TOOL_CALLING,
    AgentType.MODEL_MANAGER: LangGraphAgentType.WORKFLOW,
    AgentType.DIALOGUE_MANAGER: LangGraphAgentType.CONVERSATIONAL,
    AgentType.INFERENCE_EXECUTOR: LangGraphAgentType.TOOL_CALLING,
    AgentType.TRAINING_COORDINATOR: LangGraphAgentType.HIERARCHICAL,
    AgentType.DECISION_SUPPORT: LangGraphAgentType.CHAIN_OF_THOUGHT,
    AgentType.RESEARCH_ASSISTANT: LangGraphAgentType.REACT,
    AgentType.CODE_ASSISTANT: LangGraphAgentType.CHAIN_OF_THOUGHT,
    AgentType.DATA_ANALYST: LangGraphAgentType.PLAN_EXECUTE,
    AgentType.CUSTOMER_SERVICE: LangGraphAgentType.CONVERSATIONAL,
    AgentType.CREATIVE_WRITER: LangGraphAgentType.REFLEXION,
    AgentType.QA_SPECIALIST: LangGraphAgentType.SELF_ASK,
}


# Agent 类型默认能力
AGENT_CAPABILITIES: Dict[AgentType, List[AgentCapability]] = {
    AgentType.BASIC: [
        AgentCapability.TEXT_GENERATION,
        AgentCapability.CONVERSATION,
    ],
    AgentType.TRAINING_ASSISTANT: [
        AgentCapability.TEXT_GENERATION,
        AgentCapability.CONVERSATION,
        AgentCapability.PLANNING,
        AgentCapability.TOOL_USE,
        AgentCapability.MEMORY,
    ],
    AgentType.DATA_PROCESSOR: [
        AgentCapability.TOOL_USE,
        AgentCapability.DATA_ANALYSIS,
        AgentCapability.FILE_OPERATIONS,
    ],
    AgentType.MODEL_MANAGER: [
        AgentCapability.TOOL_USE,
        AgentCapability.PLANNING,
        AgentCapability.FILE_OPERATIONS,
    ],
    AgentType.DIALOGUE_MANAGER: [
        AgentCapability.CONVERSATION,
        AgentCapability.MEMORY,
        AgentCapability.HUMAN_IN_LOOP,
    ],
    AgentType.RESEARCH_ASSISTANT: [
        AgentCapability.TEXT_GENERATION,
        AgentCapability.WEB_SEARCH,
        AgentCapability.TOOL_USE,
        AgentCapability.MEMORY,
    ],
    AgentType.CODE_ASSISTANT: [
        AgentCapability.CODE_EXECUTION,
        AgentCapability.PLANNING,
        AgentCapability.TOOL_USE,
    ],
    AgentType.DATA_ANALYST: [
        AgentCapability.DATA_ANALYSIS,
        AgentCapability.PLANNING,
        AgentCapability.TOOL_USE,
    ],
}


# ==================== 工具函数 ====================

def get_langgraph_type(agent_type: AgentType) -> LangGraphAgentType:
    """获取业务 Agent 类型对应的 LangGraph 类型
    
    Args:
        agent_type: 业务 Agent 类型
        
    Returns:
        对应的 LangGraph Agent 类型
    """
    return AGENT_TYPE_MAPPING.get(agent_type, LangGraphAgentType.REACT)


def get_agent_capabilities(agent_type: AgentType) -> List[AgentCapability]:
    """获取 Agent 类型的默认能力列表
    
    Args:
        agent_type: Agent 类型
        
    Returns:
        能力列表
    """
    return AGENT_CAPABILITIES.get(agent_type, [AgentCapability.TEXT_GENERATION])


def get_recommended_tools(agent_type: AgentType) -> List[str]:
    """获取 Agent 类型推荐的工具列表
    
    Args:
        agent_type: Agent 类型
        
    Returns:
        工具名称列表
    """
    tool_mapping = {
        AgentType.TRAINING_ASSISTANT: [
            "training_create", "training_monitor", "model_download",
            "training_history", "training_statistics"
        ],
        AgentType.DATA_PROCESSOR: [
            "data_query", "data_transform", "data_validate", "file_read", "file_write"
        ],
        AgentType.MODEL_MANAGER: [
            "model_list", "model_load", "model_save", "model_evaluate"
        ],
        AgentType.RESEARCH_ASSISTANT: [
            "web_search", "web_scrape", "summarize", "knowledge_search"
        ],
        AgentType.CODE_ASSISTANT: [
            "code_execute", "file_read", "file_write", "shell_command"
        ],
        AgentType.DATA_ANALYST: [
            "data_query", "data_visualize", "calculate", "report_generate"
        ],
        AgentType.CUSTOMER_SERVICE: [
            "knowledge_search", "ticket_create", "order_lookup"
        ],
    }
    return tool_mapping.get(agent_type, [])


def is_multi_agent_type(agent_type: LangGraphAgentType) -> bool:
    """判断是否是多 Agent 类型
    
    Args:
        agent_type: LangGraph Agent 类型
        
    Returns:
        是否支持多 Agent
    """
    return agent_type in [
        LangGraphAgentType.MULTI_AGENT,
        LangGraphAgentType.HIERARCHICAL,
        LangGraphAgentType.WORKFLOW,
    ]


def supports_streaming(agent_type: LangGraphAgentType) -> bool:
    """判断 Agent 类型是否支持流式输出
    
    Args:
        agent_type: LangGraph Agent 类型
        
    Returns:
        是否支持流式
    """
    # 所有 Agent 类型都支持流式输出
    return True


def supports_memory(agent_type: LangGraphAgentType) -> bool:
    """判断 Agent 类型是否支持记忆
    
    Args:
        agent_type: LangGraph Agent 类型
        
    Returns:
        是否支持记忆
    """
    return agent_type in [
        LangGraphAgentType.CONVERSATIONAL,
        LangGraphAgentType.REFLEXION,
        LangGraphAgentType.PLAN_EXECUTE,
        LangGraphAgentType.WORKFLOW,
    ]


def get_default_config(agent_type: LangGraphAgentType) -> Dict[str, Any]:
    """获取 Agent 类型的默认配置
    
    Args:
        agent_type: LangGraph Agent 类型
        
    Returns:
        默认配置字典
    """
    base_config = {
        "max_iterations": 10,
        "temperature": 0.7,
        "model": "gpt-4",
        "timeout": 300.0,
    }
    
    type_configs = {
        LangGraphAgentType.REACT: {
            "max_iterations": 10,
        },
        LangGraphAgentType.PLAN_EXECUTE: {
            "max_iterations": 15,
            "enable_replanning": True,
        },
        LangGraphAgentType.REFLEXION: {
            "max_iterations": 12,
            "max_reflections": 3,
        },
        LangGraphAgentType.MULTI_AGENT: {
            "max_iterations": 20,
            "collaboration_mode": "supervisor",
        },
        LangGraphAgentType.CONVERSATIONAL: {
            "max_iterations": 10,
            "conversation_memory_size": 50,
        },
        LangGraphAgentType.CHAIN_OF_THOUGHT: {
            "max_iterations": 10,
            "temperature": 0.3,
        },
        LangGraphAgentType.SELF_ASK: {
            "max_iterations": 10,
            "max_depth": 3,
        },
        LangGraphAgentType.HIERARCHICAL: {
            "max_iterations": 25,
            "max_workers": 3,
        },
        LangGraphAgentType.WORKFLOW: {
            "max_iterations": 30,
        },
    }
    
    config = base_config.copy()
    if agent_type in type_configs:
        config.update(type_configs[agent_type])
    
    return config
