from enum import Enum


class AgentType(Enum):
    """智能体类型枚举
    
    定义系统支持的所有智能体类型。
    """
    # 基础类型
    CHAT = "chat"  # 通用对话智能体
    TRAINING = "training"
    INFERENCE = "inference"
    DATA = "data"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"
    
    # 专业类型
    DIALOGUE_MANAGER = "dialogue_manager"  # 对话管理
    INFERENCE_EXECUTOR = "inference_executor"  # 推理执行
    TRAINING_COORDINATOR = "training_coordinator"  # 训练协调
    DECISION_SUPPORT = "decision_support"  # 决策支持
    TRAINING_ASSISTANT = "training_assistant"  # 训练助手
    
    # 高级类型
    REASONING = "reasoning"  # 推理智能体
    PLANNER = "planner"  # 计划智能体
    MULTI_AGENT = "multi_agent"  # 多智能体协作