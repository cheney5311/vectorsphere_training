"""智能体服务接口

定义智能体服务的接口规范，包含工业级 LangGraph 集成能力。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

# 修复导入错误，使用正确的模块路径
from backend.schemas.agent import Agent
from backend.schemas.agent_type import AgentType


class AgentServiceInterface(ABC):
    """智能体服务接口
    
    定义智能体服务的完整接口，包括：
    - 基础 CRUD 操作
    - 会话管理
    - 记忆管理
    - LangGraph 工业级编排能力
    """
    
    @abstractmethod
    def create_agent(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
        agent_type: Optional[AgentType] = None
    ) -> Agent:
        """创建智能体
        
        Args:
            user_id: 用户ID
            name: 智能体名称
            description: 智能体描述
            version: 版本号
            config: 配置信息
            
        Returns:
            Agent: 创建的智能体对象
        """
        pass
        
    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 智能体对象，如果不存在则返回None
        """
        pass
        
    @abstractmethod
    def list_agents(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Agent]:
        """获取用户智能体列表
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Agent]: 智能体列表
        """
        pass
        
    @abstractmethod
    def update_agent(
        self, 
        agent_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """更新智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            config: 配置信息
            
        Returns:
            Agent: 更新后的智能体对象
        """
        pass
        
    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """删除智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        pass
        
    @abstractmethod
    def activate_agent(self, agent_id: str) -> Agent:
        """激活智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 激活后的智能体对象
        """
        pass
        
    @abstractmethod
    def deactivate_agent(self, agent_id: str) -> Agent:
        """停用智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Agent: 停用后的智能体对象
        """
        pass
        
    @abstractmethod
    def add_agent_capability(self, agent_id: str, capability: str) -> Agent:
        """为智能体添加能力
        
        Args:
            agent_id: 智能体ID
            capability: 能力名称
            
        Returns:
            Agent: 更新后的智能体对象
        """
        pass
        
    @abstractmethod
    def remove_agent_capability(self, agent_id: str, capability: str) -> Agent:
        """为智能体移除能力
        
        Args:
            agent_id: 智能体ID
            capability: 能力名称
            
        Returns:
            Agent: 更新后的智能体对象
        """
        pass
    
    # ==================== LangGraph 工业级接口 ====================
    
    @abstractmethod
    def create_langgraph_agent(
        self,
        agent_type: str,
        name: str,
        user_id: str,
        tools: Optional[List[Any]] = None,
        llm_client: Any = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """创建 LangGraph Agent
        
        Args:
            agent_type: Agent 类型 (react, plan_execute, reflexion, etc.)
            name: Agent 名称
            user_id: 用户 ID
            tools: 工具列表
            llm_client: LLM 客户端
            **kwargs: 其他参数
            
        Returns:
            Agent 信息字典或 None
        """
        pass
    
    @abstractmethod
    def run_langgraph_agent(
        self,
        agent_id: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行 LangGraph Agent
        
        Args:
            agent_id: Agent ID
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            执行结果字典
        """
        pass
    
    @abstractmethod
    def run_agents_workflow(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """执行 Agent 工作流
        
        Args:
            workflow: 工作流定义 [(agent_id, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            各步骤结果列表
        """
        pass
    
    @abstractmethod
    def run_agents_parallel(
        self,
        tasks: List[Tuple[str, Any]],
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """并行执行多个 Agent 任务
        
        Args:
            tasks: 任务列表 [(agent_id, input_data), ...]
            max_workers: 最大并行数
            
        Returns:
            各任务结果列表
        """
        pass
    
    @abstractmethod
    def create_agent_checkpoint(
        self,
        agent_id: str,
        checkpoint_id: Optional[str] = None,
        branch: str = "main",
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """创建 Agent 检查点
        
        Args:
            agent_id: Agent ID
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            检查点信息字典或 None
        """
        pass
    
    @abstractmethod
    def get_langgraph_health(self) -> Dict[str, Any]:
        """获取 LangGraph 系统健康状态
        
        Returns:
            健康状态字典
        """
        pass
    
    @abstractmethod
    def get_langgraph_metrics(self) -> Dict[str, Any]:
        """获取 LangGraph 系统指标
        
        Returns:
            指标字典
        """
        pass
    
    @abstractmethod
    def create_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """创建工具调用
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            工具调用信息字典或 None
        """
        pass
    
    @abstractmethod
    def create_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """创建工具结果
        
        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            工具结果信息字典或 None
        """
        pass
    
    @abstractmethod
    def run_workflow_with_events(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> Dict[str, Any]:
        """执行工作流并获取执行事件
        
        Args:
            workflow: 工作流定义 [(agent_id, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            包含结果和事件的字典
        """
        pass
    
    @abstractmethod
    def diagnose_langgraph_system(self) -> Dict[str, Any]:
        """执行 LangGraph 系统诊断
        
        Returns:
            诊断信息字典
        """
        pass
    
    @abstractmethod
    def get_available_agent_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可用的 Agent 工具
        
        Args:
            category: 工具类别过滤
            
        Returns:
            工具信息列表
        """
        pass
    
    @abstractmethod
    def create_multi_agent_system(
        self,
        name: str,
        agents: List[Dict[str, Any]],
        orchestration_type: str = "sequential"
    ) -> Optional[Dict[str, Any]]:
        """创建多 Agent 系统
        
        Args:
            name: 系统名称
            agents: Agent 配置列表
            orchestration_type: 编排类型
            
        Returns:
            系统信息字典或 None
        """
        pass