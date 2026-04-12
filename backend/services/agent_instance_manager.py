"""智能体实例管理器

负责智能体实例的创建、注册、管理和协作调度。
集成 LangGraph 工业级 Agent 编排能力。
"""

from typing import Dict, Any, List, Optional, Type, Tuple, Callable
import logging
import json
import uuid
from datetime import datetime
from abc import ABC, abstractmethod

from backend.schemas.agent import Agent
from backend.schemas.agent_type import AgentType
from backend.modules.agent.training_assistant_agent import TrainingAssistantAgent

# LangGraph 工业级组件导入
from backend.algo.langgraph import (
    # Agent 编排组件
    AgentPool as LangGraphAgentPool,
    AgentOrchestrator as LangGraphOrchestrator,
    AgentRegistry as LangGraphRegistry,
    get_agent_registry as get_langgraph_registry,
    # Agent 类型
    BaseAgent as LangGraphBaseAgent,
    AgentConfig as LangGraphAgentConfig,
    ReActAgent,
    PlanAndExecuteAgent,
    ReflexionAgent,
    MultiAgentSystem,
    # 工厂
    MasterFactory,
    get_master_factory,
    AgentFactory as LangGraphAgentFactory,
    create_react_agent,
    create_plan_execute_agent,
    # 检查点
    Checkpointer,
    MemoryCheckpointer,
    create_memory_checkpointer,
    # 工具
    Tool as LangGraphTool,
    ToolRegistry as LangGraphToolRegistry,
    get_builtin_tools,
    get_tools_by_category,
    # 状态
    AgentState,
    AgentStatus as LangGraphAgentStatus,
    StateCheckpoint,
    # 图
    StateGraph,
    GraphConfig,
    GraphStatus
)

# backend/modules/agent 层导入 (分层架构)
from backend.modules.agent import (
    LangChainInferenceService,
    get_langchain_inference_service,
)

# 智能体实现类
class BaseAgent(ABC):
    """智能体基础抽象类"""
    
    def __init__(self, agent_model: Agent):
        self.agent_model = agent_model
        self.agent_id = agent_model.agent_id
        self.name = agent_model.name
        self.description = agent_model.description
        self.capabilities = agent_model.capabilities
        self.status = agent_model.status
        self.prompt_template = ""
        self.context_memory = {}
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理输入数据并返回结果
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            处理结果字典
        """
    
    @abstractmethod
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        获取智能体的提示词
        
        Args:
            context: 上下文信息
            
        Returns:
            格式化的提示词
        """
    
    def update_status(self, status: str):
        """更新智能体状态"""
        self.status = status
        self.agent_model.status = status
        logging.info("Agent %s status updated to: %s", self.name, status)
    
    def add_capability(self, capability: str):
        """添加能力"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.agent_model.add_capability(capability)
    
    def remove_capability(self, capability: str):
        """移除能力"""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self.agent_model.remove_capability(capability)
    
    def update_context_memory(self, key: str, value: Any):
        """更新上下文记忆"""
        self.context_memory[key] = value
    
    def get_context_memory(self, key: str) -> Any:
        """获取上下文记忆"""
        return self.context_memory.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.agent_model.to_dict()


class DialogueManagerAgent(BaseAgent):
    """对话管理智能体"""
    
    def __init__(self, agent_model: Agent):
        super().__init__(agent_model)
        self.role = "dialogue_manager"
        self.prompt_template = """
你是一个智能对话管理系统。请根据以下上下文和用户输入生成合适的响应。

上下文信息:
{context}

对话历史:
{history}

用户输入:
{user_input}

请生成一个合适的响应:
"""
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理对话输入"""
        self.update_status("processing")
        
        # 获取用户输入
        user_input = input_data.get("user_input", "")
        if not user_input:
            return {
                "status": "error",
                "error": "用户输入不能为空",
                "agent_id": self.agent_id
            }
        
        # 生成响应（模拟实现）
        response = await self._generate_response(user_input, context or {})
        
        self.update_status("idle")
        
        return {
            "status": "success",
            "response": response,
            "agent_id": self.agent_id,
            "confidence": 0.9,
            "context_used": context or {}
        }
    
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """获取对话管理提示词"""
        context = context or {}
        return self.prompt_template.format(
            context=json.dumps(context, ensure_ascii=False, indent=2),
            history="",  # 简化实现
            user_input=context.get("user_input", "")
        )
    
    async def _generate_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """生成响应"""
        # 根据输入内容生成不同类型的响应
        if "训练" in user_input or "模型" in user_input:
            return "我理解您想了解模型训练相关的内容。我们可以讨论训练配置、数据集准备、模型优化等方面的问题。您具体想了解哪个方面呢？"
        elif "数据" in user_input or "数据集" in user_input:
            return "关于数据集管理，我们可以帮助您上传、处理和管理训练数据。您需要处理什么类型的数据？"
        elif "帮助" in user_input or "功能" in user_input:
            return "我是对话管理智能体，可以帮助您进行自然语言交互。我可以协助您处理模型训练、数据管理、系统配置等任务。请告诉我您需要什么帮助？"
        else:
            return f"我收到了您的消息：'{user_input}'。我是一个智能对话助手，可以帮您处理各种任务。请问有什么我可以帮您的吗？"


class InferenceExecutorAgent(BaseAgent):
    """推理执行智能体"""
    
    def __init__(self, agent_model: Agent):
        super().__init__(agent_model)
        self.role = "inference_executor"
        self.prompt_template = """
你是一个模型推理执行系统。请根据以下输入执行推理任务。

任务描述:
{task_description}

输入数据:
{input_data}

请执行推理并返回结果:
"""
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行推理任务"""
        self.update_status("processing")
        
        # 执行推理（模拟实现）
        result = await self._execute_inference(input_data, context or {})
        
        self.update_status("idle")
        
        return {
            "status": "success",
            "result": result,
            "agent_id": self.agent_id,
            "confidence": 0.85,
            "context_used": context or {}
        }
    
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """获取推理执行提示词"""
        context = context or {}
        return self.prompt_template.format(
            task_description=context.get("task_description", "执行推理任务"),
            input_data=json.dumps(context.get("input_data", {}), ensure_ascii=False, indent=2)
        )
    
    async def _execute_inference(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行推理"""
        # 模拟推理执行
        return {
            "output": "推理结果",
            "details": "这是模拟的推理结果",
            "confidence": 0.85
        }


class TrainingCoordinatorAgent(BaseAgent):
    """训练协调智能体"""
    
    def __init__(self, agent_model: Agent):
        super().__init__(agent_model)
        self.role = "training_coordinator"
        self.prompt_template = """
你是一个模型训练协调系统。请根据以下配置协调训练任务。

训练配置:
{training_config}

资源信息:
{resource_info}

请协调训练任务并返回状态:
"""
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """协调训练任务"""
        self.update_status("processing")
        
        # 协调训练（模拟实现）
        result = await self._coordinate_training(input_data, context or {})
        
        self.update_status("idle")
        
        return {
            "status": "success",
            "result": result,
            "agent_id": self.agent_id,
            "confidence": 0.9,
            "context_used": context or {}
        }
    
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """获取训练协调提示词"""
        context = context or {}
        return self.prompt_template.format(
            training_config=json.dumps(context.get("training_config", {}), ensure_ascii=False, indent=2),
            resource_info=json.dumps(context.get("resource_info", {}), ensure_ascii=False, indent=2)
        )
    
    async def _coordinate_training(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """协调训练"""
        # 模拟训练协调
        return {
            "status": "training_started",
            "job_id": "job_12345",
            "message": "训练任务已启动"
        }


class DecisionSupportAgent(BaseAgent):
    """决策支持智能体"""
    
    def __init__(self, agent_model: Agent):
        super().__init__(agent_model)
        self.role = "decision_support"
        self.prompt_template = """
你是一个决策支持系统。请根据以下数据提供决策建议。

分析数据:
{analysis_data}

决策上下文:
{decision_context}

请提供决策建议:
"""
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """提供决策支持"""
        self.update_status("processing")
        
        # 提供决策支持（模拟实现）
        result = await self._provide_decision_support(input_data, context or {})
        
        self.update_status("idle")
        
        return {
            "status": "success",
            "result": result,
            "agent_id": self.agent_id,
            "confidence": 0.8,
            "context_used": context or {}
        }
    
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """获取决策支持提示词"""
        context = context or {}
        return self.prompt_template.format(
            analysis_data=json.dumps(context.get("analysis_data", {}), ensure_ascii=False, indent=2),
            decision_context=json.dumps(context.get("decision_context", {}), ensure_ascii=False, indent=2)
        )
    
    async def _provide_decision_support(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """提供决策支持"""
        # 模拟决策支持
        return {
            "recommendation": "建议采取行动A",
            "confidence": 0.8,
            "alternatives": ["行动A", "行动B", "行动C"],
            "reasoning": "基于数据分析，行动A的成功率最高"
        }


class AgentInstanceManager:
    """智能体实例管理器
    
    集成 LangGraph 工业级 Agent 编排能力：
    - Agent 池管理：负载均衡、健康检查、自动扩缩容
    - Agent 编排：顺序执行、并行执行、工作流编排
    - 检查点管理：状态保存、恢复、分支
    - 工厂模式：统一创建各类 Agent
    """
    
    def __init__(self):
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.DIALOGUE_MANAGER: DialogueManagerAgent,
            AgentType.INFERENCE_EXECUTOR: InferenceExecutorAgent,
            AgentType.TRAINING_COORDINATOR: TrainingCoordinatorAgent,
            AgentType.DECISION_SUPPORT: DecisionSupportAgent,
            AgentType.TRAINING_ASSISTANT: TrainingAssistantAgent
        }
        self.collaboration_rules: Dict[str, List[str]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # LangGraph 工业级组件
        self._langgraph_pool: Optional['LangGraphAgentPool'] = None
        self._langgraph_orchestrator: Optional['LangGraphOrchestrator'] = None
        self._langgraph_registry: Optional['LangGraphRegistry'] = None
        self._master_factory: Optional['MasterFactory'] = None
        self._checkpointers: Dict[str, 'Checkpointer'] = {}
        
        # backend/modules/agent 层服务引用
        self._inference_service: Optional['LangChainInferenceService'] = None
        
        # 初始化 LangGraph 组件
        self._init_langgraph_components()
        
        # 初始化 modules/agent 层连接
        self._init_modules_agent_layer()
        
    def create_agent_instance(self, agent_model: Agent) -> Optional[BaseAgent]:
        """
        创建智能体实例
        
        Args:
            agent_model: 智能体模型对象
            
        Returns:
            创建的智能体实例或None
        """
        try:
            if not agent_model.agent_type:
                logging.error("Agent model has no type specified")
                return None
            
            agent_class = self.agent_types.get(agent_model.agent_type)
            if not agent_class:
                logging.error("Unknown agent type: %s", agent_model.agent_type)
                return None
            
            agent_instance = agent_class(agent_model)
            if agent_model.agent_id is not None:
                self.agent_instances[agent_model.agent_id] = agent_instance
                logging.info(
                    "Agent instance %s (%s) created successfully",
                    agent_model.name, agent_model.agent_id
                )
                return agent_instance
            else:
                logging.error("Agent model has no ID specified")
                return None
            
        except Exception as e:
            logging.error("Failed to create agent instance: %s", e)
            return None
    
    def get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """
        获取智能体实例
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体实例或None
        """
        return self.agent_instances.get(agent_id)
    
    def destroy_agent_instance(self, agent_id: str) -> bool:
        """
        销毁智能体实例
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            销毁是否成功
        """
        try:
            if agent_id in self.agent_instances:
                del self.agent_instances[agent_id]
                logging.info("Agent instance %s destroyed successfully", agent_id)
                return True
            else:
                logging.warning("Agent instance %s not found for destruction", agent_id)
                return False
                
        except Exception as e:
            logging.error("Failed to destroy agent instance %s: %s", agent_id, e)
            return False
    
    def list_agent_instances(self) -> List[Dict[str, Any]]:
        """
        列出所有智能体实例
        
        Returns:
            智能体实例信息列表
        """
        return [agent.to_dict() for agent in self.agent_instances.values()]
    
    def get_agent_instance_status(self, agent_id: str) -> Optional[str]:
        """
        获取智能体实例状态
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体实例状态或None
        """
        agent = self.get_agent_instance(agent_id)
        return agent.status if agent else None
    
    async def execute_agent_task(self, agent_id: str, input_data: Dict[str, Any], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行智能体任务
        
        Args:
            agent_id: 智能体ID
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            执行结果
        """
        try:
            agent = self.get_agent_instance(agent_id)
            if not agent:
                return {"error": f"Agent {agent_id} not found"}
            
            return await agent.process(input_data, context)
            
        except Exception as e:
            logging.error("Agent task execution failed: %s", e)
            return {
                "agent_id": agent_id,
                "error": str(e),
                "status": "failed"
            }
    
    def get_agent_prompt(self, agent_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        获取智能体提示词
        
        Args:
            agent_id: 智能体ID
            context: 上下文信息
            
        Returns:
            格式化的提示词或None
        """
        agent = self.get_agent_instance(agent_id)
        if not agent:
            return None
        
        return agent.get_prompt(context)
    
    # ==================== LangGraph 工业级派生方法 ====================
    
    def _init_langgraph_components(self) -> None:
        """初始化 LangGraph 工业级组件"""
        try:
            self._langgraph_pool = LangGraphAgentPool(max_agents=50)
            self._langgraph_orchestrator = LangGraphOrchestrator(self._langgraph_pool)
            self._langgraph_registry = get_langgraph_registry()
            self._master_factory = get_master_factory()
            logging.info("LangGraph components initialized successfully")
        except Exception as e:
            logging.error("Failed to initialize LangGraph components: %s", e)
    
    def _init_modules_agent_layer(self) -> None:
        """初始化 backend/modules/agent 层连接
        
        建立与业务层的连接，实现 services -> modules/agent -> algo 分层架构
        """
        try:
            self._inference_service = get_langchain_inference_service(use_langgraph=True)
            logging.info("modules/agent layer connection established")
        except Exception as e:
            logging.error("Failed to initialize modules/agent layer: %s", e)
    
    def get_inference_service(self) -> Optional['LangChainInferenceService']:
        """获取推理服务实例
        
        Returns:
            LangChainInferenceService 实例或 None
        """
        if self._inference_service is None:
            self._init_modules_agent_layer()
        return self._inference_service
    
    def get_master_factory(self) -> Optional['MasterFactory']:
        """获取 MasterFactory 实例
        
        调用 langgraph.get_master_factory
        
        Returns:
            MasterFactory 实例或 None
        """
        if self._master_factory is None:
            self._master_factory = get_master_factory()
        return self._master_factory
    
    def get_langgraph_pool(self) -> Optional['LangGraphAgentPool']:
        """获取 LangGraph Agent 池
        
        Returns:
            AgentPool 实例或 None
        """
        return self._langgraph_pool
    
    def get_langgraph_orchestrator(self) -> Optional['LangGraphOrchestrator']:
        """获取 LangGraph 编排器
        
        Returns:
            AgentOrchestrator 实例或 None
        """
        return self._langgraph_orchestrator
    
    # ==================== Agent 池管理 ====================
    
    def create_langgraph_agent(self, agent_type: str = "react",
                               name: str = None,
                               tools: List[Any] = None,
                               llm_client: Any = None,
                               **kwargs) -> Optional['LangGraphBaseAgent']:
        """使用 LangGraph 工厂创建 Agent
        
        调用 MasterFactory.agents.create
        
        Args:
            agent_type: Agent 类型 (react, plan_execute, reflexion, etc.)
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            **kwargs: 其他参数
            
        Returns:
            LangGraph Agent 实例或 None
        """
        if not self._master_factory:
            logging.warning("LangGraph not available for agent creation")
            return None
        
        try:
            agent = self._master_factory.agents.create(
                agent_type,
                name=name or f"agent_{uuid.uuid4().hex[:8]}",
                tools=tools,
                llm_client=llm_client,
                **kwargs
            )
            
            # 注册到池
            if self._langgraph_pool and agent:
                self._langgraph_pool.register(agent.name, agent)
                logging.info("LangGraph agent '%s' created and registered", agent.name)
            
            return agent
        except Exception as e:
            logging.error("Failed to create LangGraph agent: %s", e)
            return None
    
    def register_to_langgraph_pool(self, name: str, agent: 'LangGraphBaseAgent') -> bool:
        """注册 Agent 到 LangGraph 池
        
        调用 AgentPool.register
        
        Args:
            name: Agent 名称
            agent: LangGraph Agent 实例
            
        Returns:
            是否成功
        """
        if not self._langgraph_pool:
            return False
        
        return self._langgraph_pool.register(name, agent)
    
    def unregister_from_langgraph_pool(self, name: str) -> Optional['LangGraphBaseAgent']:
        """从 LangGraph 池注销 Agent
        
        调用 AgentPool.unregister
        
        Args:
            name: Agent 名称
            
        Returns:
            被注销的 Agent 或 None
        """
        if not self._langgraph_pool:
            return None
        
        return self._langgraph_pool.unregister(name)
    
    def run_langgraph_agent(self, name: str, 
                           input_data: Any,
                           **kwargs) -> Optional['AgentState']:
        """运行 LangGraph 池中的 Agent
        
        调用 AgentPool.run
        
        Args:
            name: Agent 名称
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            AgentState 结果或 None
        """
        if not self._langgraph_pool:
            return None
        
        try:
            return self._langgraph_pool.run(name, input_data, **kwargs)
        except Exception as e:
            logging.error("Failed to run LangGraph agent '%s': %s", name, e)
            return None
    
    def get_langgraph_pool_metrics(self) -> Dict[str, Any]:
        """获取 LangGraph 池指标
        
        调用 AgentPool.get_pool_metrics
        
        Returns:
            池指标字典
        """
        if not self._langgraph_pool:
            return {"error": "LangGraph pool not available"}
        
        return self._langgraph_pool.get_pool_metrics()
    
    def get_langgraph_pool_health(self) -> Dict[str, Any]:
        """获取 LangGraph 池健康状态
        
        调用 AgentPool.get_pool_health_with_factory
        
        Returns:
            健康状态字典
        """
        if not self._langgraph_pool:
            return {"error": "LangGraph pool not available"}
        
        try:
            return self._langgraph_pool.get_pool_health_with_factory()
        except Exception as e:
            return {"error": str(e)}
    
    def setup_tool_cache_for_langgraph_agent(self, agent_name: str,
                                             max_size: int = 1000,
                                             default_ttl: int = 300) -> bool:
        """为 LangGraph Agent 设置工具缓存
        
        调用 AgentPool.setup_tool_cache_for_agent
        
        Args:
            agent_name: Agent 名称
            max_size: 缓存最大大小
            default_ttl: 默认 TTL
            
        Returns:
            是否成功
        """
        if not self._langgraph_pool:
            return False
        
        return self._langgraph_pool.setup_tool_cache_for_agent(
            agent_name, max_size, default_ttl
        )
    
    def setup_retry_handler_for_langgraph_agent(self, agent_name: str,
                                                max_retries: int = 3,
                                                base_delay: float = 1.0) -> bool:
        """为 LangGraph Agent 设置重试处理器
        
        调用 AgentPool.setup_retry_handler_for_agent
        
        Args:
            agent_name: Agent 名称
            max_retries: 最大重试次数
            base_delay: 基础延迟
            
        Returns:
            是否成功
        """
        if not self._langgraph_pool:
            return False
        
        return self._langgraph_pool.setup_retry_handler_for_agent(
            agent_name, max_retries, base_delay
        )
    
    # ==================== 编排管理 ====================
    
    def add_agent_to_orchestrator(self, agent: 'LangGraphBaseAgent') -> None:
        """添加 Agent 到编排器
        
        调用 AgentOrchestrator.add_agent
        
        Args:
            agent: LangGraph Agent 实例
        """
        if self._langgraph_orchestrator:
            self._langgraph_orchestrator.add_agent(agent)
    
    def define_workflow_pipeline(self, name: str, steps: List[Dict[str, Any]]) -> None:
        """定义工作流流水线
        
        调用 AgentOrchestrator.define_pipeline
        
        Args:
            name: 流水线名称
            steps: 步骤定义列表
        """
        if self._langgraph_orchestrator:
            self._langgraph_orchestrator.define_pipeline(name, steps)
    
    def run_workflow_pipeline(self, pipeline_name: str,
                             initial_input: Any = None) -> List['AgentState']:
        """执行工作流流水线
        
        调用 AgentOrchestrator.run_pipeline
        
        Args:
            pipeline_name: 流水线名称
            initial_input: 初始输入
            
        Returns:
            各步骤结果列表
        """
        if not self._langgraph_orchestrator:
            return []
        
        try:
            return self._langgraph_orchestrator.run_pipeline(pipeline_name, initial_input)
        except Exception as e:
            logging.error("Failed to run pipeline '%s': %s", pipeline_name, e)
            return []
    
    def run_agents_parallel(self, tasks: List[Tuple[str, Any]],
                           max_workers: int = 4) -> List[Tuple[str, 'AgentState']]:
        """并行运行多个 Agent
        
        调用 AgentOrchestrator.run_parallel
        
        Args:
            tasks: 任务列表 [(agent_name, input_data), ...]
            max_workers: 最大并行数
            
        Returns:
            结果列表 [(agent_name, result), ...]
        """
        if not self._langgraph_orchestrator:
            return []
        
        try:
            return self._langgraph_orchestrator.run_parallel(tasks, max_workers)
        except Exception as e:
            logging.error("Failed to run parallel tasks: %s", e)
            return []
    
    def run_agents_workflow(self, workflow: List[Tuple[str, Any]],
                           stop_on_error: bool = False) -> List['AgentState']:
        """执行 Agent 工作流
        
        调用 AgentOrchestrator.run_workflow
        
        Args:
            workflow: 工作流定义 [(agent_name, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            结果列表
        """
        if not self._langgraph_orchestrator:
            return []
        
        try:
            return self._langgraph_orchestrator.run_workflow(workflow, stop_on_error)
        except Exception as e:
            logging.error("Failed to run workflow: %s", e)
            return []
    
    def run_workflow_with_events(self, workflow: List[Tuple[str, Any]],
                                stop_on_error: bool = False) -> Tuple[List['AgentState'], List[Any]]:
        """执行工作流并生成执行事件
        
        调用 AgentOrchestrator.run_workflow_with_events
        
        Args:
            workflow: 工作流定义
            stop_on_error: 遇错是否停止
            
        Returns:
            (结果列表, 事件列表)
        """
        if not self._langgraph_orchestrator:
            return [], []
        
        try:
            return self._langgraph_orchestrator.run_workflow_with_events(workflow, stop_on_error)
        except Exception as e:
            logging.error("Failed to run workflow with events: %s", e)
            return [], []
    
    def get_orchestrator_health(self) -> Dict[str, Any]:
        """获取编排器健康状态
        
        调用 AgentOrchestrator.get_orchestrator_health_with_factory
        
        Returns:
            健康状态字典
        """
        if not self._langgraph_orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            return self._langgraph_orchestrator.get_orchestrator_health_with_factory()
        except Exception as e:
            return {"error": str(e)}
    
    def create_execution_event(self, event_type: str,
                              node_name: str,
                              data: Dict[str, Any] = None) -> Optional[Any]:
        """创建执行事件
        
        调用 AgentOrchestrator.create_execution_event
        
        Args:
            event_type: 事件类型
            node_name: 节点名称
            data: 事件数据
            
        Returns:
            ExecutionEvent 实例或 None
        """
        if not self._langgraph_orchestrator:
            return None
        
        return self._langgraph_orchestrator.create_execution_event(event_type, node_name, data)
    
    # ==================== 检查点管理 ====================
    
    def create_checkpointer(self, checkpointer_type: str = "memory",
                           **kwargs) -> Optional['Checkpointer']:
        """创建检查点器
        
        调用 MasterFactory.get_checkpointer_instance
        
        Args:
            checkpointer_type: 检查点器类型 (memory, redis, sqlite, file)
            **kwargs: 检查点器参数
            
        Returns:
            Checkpointer 实例或 None
        """
        if not self._master_factory:
            return None
        
        try:
            checkpointer = self._master_factory.get_checkpointer_instance(
                checkpointer_type, **kwargs
            )
            cp_id = f"cp_{uuid.uuid4().hex[:8]}"
            self._checkpointers[cp_id] = checkpointer
            return checkpointer
        except Exception as e:
            logging.error("Failed to create checkpointer: %s", e)
            return None
    
    def create_enhanced_checkpoint(self, agent_name: str,
                                   checkpoint_id: str = None,
                                   branch: str = "main",
                                   tags: List[str] = None) -> Optional[Any]:
        """为 Agent 创建增强检查点
        
        调用 AgentPool.create_enhanced_checkpoint_for_agent
        
        Args:
            agent_name: Agent 名称
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            EnhancedCheckpoint 实例或 None
        """
        if not self._langgraph_pool:
            return None
        
        return self._langgraph_pool.create_enhanced_checkpoint_for_agent(
            agent_name, checkpoint_id, branch, tags
        )
    
    # ==================== 工具管理 ====================
    
    def get_available_builtin_tools(self) -> List[Any]:
        """获取可用的内置工具
        
        调用 langgraph.get_builtin_tools
        
        Returns:
            内置工具列表
        """
        try:
            return get_builtin_tools()
        except Exception as e:
            logging.error("Failed to get builtin tools: %s", e)
            return []
    
    def get_tools_by_category(self, category: str) -> List[Any]:
        """按类别获取工具
        
        调用 langgraph.get_tools_by_category
        
        Args:
            category: 工具类别
            
        Returns:
            工具列表
        """
        try:
            return get_tools_by_category(category)
        except Exception as e:
            logging.error("Failed to get tools by category: %s", e)
            return []
    
    def create_tool_metrics(self, tool_name: str) -> Optional[Any]:
        """创建工具指标
        
        调用 MasterFactory.create_tool_metrics
        
        Args:
            tool_name: 工具名称
            
        Returns:
            ToolMetrics 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_tool_metrics(tool_name)
    
    def create_tool_cache(self, max_size: int = 1000,
                         default_ttl: int = 300) -> Optional[Any]:
        """创建工具缓存
        
        调用 MasterFactory.create_tool_cache
        
        Args:
            max_size: 缓存最大大小
            default_ttl: 默认 TTL
            
        Returns:
            ToolCache 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_tool_cache(max_size, default_ttl)
    
    # ==================== 状态和消息管理 ====================
    
    def create_agent_message(self, content: str,
                            message_type: str = "human",
                            name: str = None,
                            metadata: Dict[str, Any] = None) -> Optional[Any]:
        """创建 Agent 消息
        
        调用 MasterFactory.create_agent_message
        
        Args:
            content: 消息内容
            message_type: 消息类型
            name: 发送者名称
            metadata: 元数据
            
        Returns:
            AgentMessage 实例或 None
        """
        if not self._master_factory:
            return None
        
        try:
            # 获取消息类型枚举
            from backend.algo.langgraph import MessageType
            type_map = {
                "human": MessageType.HUMAN,
                "ai": MessageType.AI,
                "system": MessageType.SYSTEM,
                "tool": MessageType.TOOL,
                "function": MessageType.FUNCTION
            }
            msg_type = type_map.get(message_type.lower(), MessageType.HUMAN)
            
            return self._master_factory.create_agent_message(
                content=content,
                message_type=msg_type,
                name=name,
                metadata=metadata
            )
        except Exception as e:
            logging.error("Failed to create agent message: %s", e)
            return None
    
    def create_memory_entry(self, content: Any,
                           memory_type: str = "short_term",
                           importance: float = 0.5,
                           tags: List[str] = None) -> Optional[Any]:
        """创建记忆条目
        
        调用 MasterFactory.create_memory_entry
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            tags: 标签列表
            
        Returns:
            MemoryEntry 实例或 None
        """
        if not self._master_factory:
            return None
        
        try:
            mem_type = self._master_factory.get_memory_type(memory_type)
            return self._master_factory.create_memory_entry(
                content=content,
                memory_type=mem_type,
                importance=importance,
                tags=tags
            )
        except Exception as e:
            logging.error("Failed to create memory entry: %s", e)
            return None
    
    def create_state_manager(self) -> Optional[Any]:
        """创建状态管理器
        
        调用 MasterFactory.create_state_manager
        
        Returns:
            StateManager 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_state_manager()
    
    # ==================== 图和节点管理 ====================
    
    def create_graph_metrics(self) -> Optional[Any]:
        """创建图指标
        
        调用 MasterFactory.create_graph_metrics
        
        Returns:
            GraphMetrics 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_graph_metrics()
    
    def create_node_config(self, name: str,
                          node_type: str = "transform",
                          description: str = "",
                          timeout: float = 60.0,
                          retry_count: int = 3) -> Optional[Any]:
        """创建节点配置
        
        调用 MasterFactory.create_node_config
        
        Args:
            name: 节点名称
            node_type: 节点类型
            description: 节点描述
            timeout: 超时时间
            retry_count: 重试次数
            
        Returns:
            NodeConfig 实例或 None
        """
        if not self._master_factory:
            return None
        
        try:
            nt = self._master_factory.get_node_type(node_type)
            return self._master_factory.create_node_config(
                name=name,
                node_type=nt,
                description=description,
                timeout=timeout,
                retry_count=retry_count
            )
        except Exception as e:
            logging.error("Failed to create node config: %s", e)
            return None
    
    # ==================== 路由器管理 ====================
    
    def create_priority_router(self, name: str = "priority_router") -> Optional[Any]:
        """创建优先级路由器
        
        调用 MasterFactory.create_priority_router
        
        Args:
            name: 路由器名称
            
        Returns:
            PriorityRouter 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_priority_router(name)
    
    def create_weighted_router(self, name: str = "weighted_router") -> Optional[Any]:
        """创建权重路由器
        
        调用 MasterFactory.create_weighted_router
        
        Args:
            name: 路由器名称
            
        Returns:
            WeightedRouter 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_weighted_router(name)
    
    def create_load_balance_router(self, name: str = "lb_router") -> Optional[Any]:
        """创建负载均衡路由器
        
        调用 MasterFactory.create_load_balance_router
        
        Args:
            name: 路由器名称
            
        Returns:
            LoadBalanceRouter 实例或 None
        """
        if not self._master_factory:
            return None
        
        return self._master_factory.create_load_balance_router(name)
    
    # ==================== Agent 类型创建 ====================
    
    def create_react_agent_instance(
        self,
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        config: Dict[str, Any] = None
    ) -> Optional['ReActAgent']:
        """创建 ReAct Agent 实例
        
        调用 langgraph.create_react_agent 和 ReActAgent
        
        Args:
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            config: Agent 配置
            
        Returns:
            ReActAgent 实例或 None
        """
        try:
            agent_name = name or f"react_agent_{uuid.uuid4().hex[:8]}"
            
            # 使用工厂函数创建
            agent = create_react_agent(
                name=agent_name,
                tools=tools or [],
                llm_client=llm_client
            )
            
            # 注册到池
            if self._langgraph_pool and agent:
                self._langgraph_pool.register(agent_name, agent)
                logging.info("ReAct agent '%s' created and registered", agent_name)
            
            return agent
        except Exception as e:
            logging.error("Failed to create ReAct agent: %s", e)
            return None
    
    def create_plan_execute_agent_instance(
        self,
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        planner_llm: Any = None,
        config: Dict[str, Any] = None
    ) -> Optional['PlanAndExecuteAgent']:
        """创建 Plan-and-Execute Agent 实例
        
        调用 langgraph.create_plan_execute_agent 和 PlanAndExecuteAgent
        
        Args:
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            planner_llm: 规划器 LLM
            config: Agent 配置
            
        Returns:
            PlanAndExecuteAgent 实例或 None
        """
        try:
            agent_name = name or f"plan_exec_agent_{uuid.uuid4().hex[:8]}"
            
            # 使用工厂函数创建
            agent = create_plan_execute_agent(
                name=agent_name,
                tools=tools or [],
                llm_client=llm_client,
                planner_llm=planner_llm
            )
            
            # 注册到池
            if self._langgraph_pool and agent:
                self._langgraph_pool.register(agent_name, agent)
                logging.info("Plan-Execute agent '%s' created and registered", agent_name)
            
            return agent
        except Exception as e:
            logging.error("Failed to create Plan-Execute agent: %s", e)
            return None
    
    def create_reflexion_agent_instance(
        self,
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        max_reflections: int = 3,
        quality_threshold: float = 0.8,
        config: Dict[str, Any] = None
    ) -> Optional['ReflexionAgent']:
        """创建 Reflexion Agent 实例
        
        调用 ReflexionAgent 类
        
        Args:
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            max_reflections: 最大反思次数
            quality_threshold: 质量阈值 (0-1)
            config: Agent 配置
            
        Returns:
            ReflexionAgent 实例或 None
        """
        try:
            agent_name = name or f"reflexion_agent_{uuid.uuid4().hex[:8]}"
            
            # 创建 Agent 配置
            agent_config = LangGraphAgentConfig(
                name=agent_name,
                **(config or {})
            )
            
            # 创建 Reflexion Agent (使用正确的参数签名)
            agent = ReflexionAgent(
                config=agent_config,
                tools=tools or [],
                llm_client=llm_client,
                max_reflections=max_reflections,
                quality_threshold=quality_threshold
            )
            
            # 注册到池
            if self._langgraph_pool and agent:
                self._langgraph_pool.register(agent_name, agent)
                logging.info("Reflexion agent '%s' created and registered", agent_name)
            
            return agent
        except Exception as e:
            logging.error("Failed to create Reflexion agent: %s", e)
            return None
    
    def create_multi_agent_system_instance(
        self,
        name: str = None,
        roles: List[Any] = None,
        llm_client: Any = None,
        collaboration_mode: str = "supervisor",
        config: Dict[str, Any] = None
    ) -> Optional['MultiAgentSystem']:
        """创建多 Agent 系统实例
        
        调用 MultiAgentSystem 类
        
        Args:
            name: 系统名称
            roles: AgentRole 列表 (定义每个 Agent 的角色、名称、提示词和工具)
            llm_client: LLM 客户端
            collaboration_mode: 协作模式 (supervisor, round_robin, hierarchical)
            config: 系统配置
            
        Returns:
            MultiAgentSystem 实例或 None
        """
        try:
            system_name = name or f"mas_{uuid.uuid4().hex[:8]}"
            
            # 创建 Agent 配置
            agent_config = LangGraphAgentConfig(
                name=system_name,
                **(config or {})
            )
            
            # 创建多 Agent 系统 (使用正确的参数签名)
            mas = MultiAgentSystem(
                config=agent_config,
                roles=roles or [],
                llm_client=llm_client,
                collaboration_mode=collaboration_mode
            )
            
            logging.info("Multi-agent system '%s' created with %d roles", system_name, len(roles or []))
            return mas
        except Exception as e:
            logging.error("Failed to create multi-agent system: %s", e)
            return None
    
    def create_langgraph_agent_config(
        self,
        name: str,
        system_prompt: str = "",
        max_iterations: int = 10,
        timeout: float = 300.0,
        model: str = "gpt-4",
        temperature: float = 0.7,
        **kwargs
    ) -> Optional['LangGraphAgentConfig']:
        """创建 LangGraph Agent 配置
        
        调用 LangGraphAgentConfig 类
        
        Args:
            name: Agent 名称
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            timeout: 超时时间
            model: LLM 模型名称
            temperature: 温度参数
            **kwargs: 其他配置参数
            
        Returns:
            LangGraphAgentConfig 实例或 None
        
        Note:
            tools 和 llm_client 应直接传递给 Agent 构造函数,
            而非 AgentConfig
        """
        try:
            # AgentConfig 只包含配置参数, 不包含 tools 和 llm_client
            config = LangGraphAgentConfig(
                name=name,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                timeout=timeout,
                model=model,
                temperature=temperature,
                **kwargs
            )
            logging.info("Agent config '%s' created", name)
            return config
        except Exception as e:
            logging.error("Failed to create agent config: %s", e)
            return None
    
    # ==================== Agent 工厂管理 ====================
    
    def get_agent_factory(self) -> Optional['LangGraphAgentFactory']:
        """获取 LangGraph Agent 工厂
        
        调用 LangGraphAgentFactory
        
        Returns:
            AgentFactory 实例或 None
        """
        if self._master_factory:
            return self._master_factory.agents
        
        return LangGraphAgentFactory()
    
    def create_agent_via_factory(
        self,
        agent_type: str,
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        **kwargs
    ) -> Optional['LangGraphBaseAgent']:
        """通过工厂创建 Agent
        
        调用 LangGraphAgentFactory.create
        
        Args:
            agent_type: Agent 类型
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            **kwargs: 其他参数
            
        Returns:
            LangGraph Agent 实例或 None
        """
        factory = self.get_agent_factory()
        if not factory:
            return None
        
        try:
            agent_name = name or f"{agent_type}_agent_{uuid.uuid4().hex[:8]}"
            agent = factory.create(
                agent_type,
                name=agent_name,
                tools=tools,
                llm_client=llm_client,
                **kwargs
            )
            
            # 注册到池
            if self._langgraph_pool and agent:
                self._langgraph_pool.register(agent_name, agent)
                logging.info(f"Agent '{agent_name}' created via factory and registered")
            
            return agent
        except Exception as e:
            logging.error(f"Failed to create agent via factory: {e}")
            return None
    
    # ==================== 检查点增强管理 ====================
    
    def create_memory_checkpointer_instance(
        self,
        max_checkpoints: int = 100
    ) -> Optional['MemoryCheckpointer']:
        """创建内存检查点器实例
        
        调用 langgraph.create_memory_checkpointer 和 MemoryCheckpointer
        
        Args:
            max_checkpoints: 最大检查点数
            
        Returns:
            MemoryCheckpointer 实例或 None
        """
        try:
            # 使用正确的参数签名 (不包含 auto_cleanup)
            checkpointer = create_memory_checkpointer(
                max_checkpoints=max_checkpoints
            )
            
            # 存储引用
            cp_id = f"mem_cp_{uuid.uuid4().hex[:8]}"
            self._checkpointers[cp_id] = checkpointer
            
            logging.info("Memory checkpointer '%s' created", cp_id)
            return checkpointer
        except Exception as e:
            logging.error("Failed to create memory checkpointer: %s", e)
            return None
    
    def create_state_checkpoint(
        self,
        state: Any,
        checkpoint_id: str = None,
        thread_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional['StateCheckpoint']:
        """创建状态检查点
        
        调用 StateCheckpoint 类
        
        Args:
            state: 状态数据 (字典形式)
            checkpoint_id: 检查点 ID
            thread_id: 线程 ID (用于标识会话)
            metadata: 元数据
            
        Returns:
            StateCheckpoint 实例或 None
        """
        try:
            cp_id = checkpoint_id or f"state_cp_{uuid.uuid4().hex[:8]}"
            t_id = thread_id or f"thread_{uuid.uuid4().hex[:8]}"
            
            # 确保 state 是字典格式
            state_dict = state if isinstance(state, dict) else {"data": state}
            
            # 使用正确的参数签名 (包含必需的 thread_id)
            checkpoint = StateCheckpoint(
                checkpoint_id=cp_id,
                thread_id=t_id,
                state=state_dict,
                metadata=metadata or {}
            )
            
            logging.info("State checkpoint '%s' created", cp_id)
            return checkpoint
        except Exception as e:
            logging.error("Failed to create state checkpoint: %s", e)
            return None
    
    # ==================== 工具注册表管理 ====================
    
    def get_tool_registry(self) -> Optional['LangGraphToolRegistry']:
        """获取工具注册表
        
        调用 LangGraphToolRegistry
        
        Returns:
            ToolRegistry 实例或 None
        """
        # MasterFactory 通过 tools 属性访问 ToolFactory
        # ToolFactory 内部管理工具注册表
        if self._master_factory:
            return self._master_factory.tools._registry
        
        return LangGraphToolRegistry()
    
    def create_langgraph_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: List[Any] = None,
        return_direct: bool = False
    ) -> Optional['LangGraphTool']:
        """创建 LangGraph 工具
        
        调用 LangGraphTool 类
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            parameters: 工具参数列表 (ToolParameter 对象)
            return_direct: 是否直接返回
            
        Returns:
            LangGraphTool 实例或 None
        """
        try:
            # 使用正确的参数签名 (parameters 而非 args_schema)
            tool = LangGraphTool(
                name=name,
                func=func,
                description=description or f"Tool: {name}",
                parameters=parameters or [],
                return_direct=return_direct
            )
            
            # 注册到工具注册表
            registry = self.get_tool_registry()
            if registry:
                registry.register(tool)
                logging.info("Tool '%s' created and registered", name)
            
            return tool
        except Exception as e:
            logging.error("Failed to create tool: %s", e)
            return None
    
    def register_tool_to_registry(
        self,
        tool: 'LangGraphTool',
        alias: str = None
    ) -> bool:
        """注册工具到注册表
        
        调用 LangGraphToolRegistry.register
        
        Args:
            tool: 工具实例
            alias: 工具别名
            
        Returns:
            是否成功
        """
        registry = self.get_tool_registry()
        if not registry:
            return False
        
        try:
            registry.register(tool, alias=alias)
            logging.info("Tool '%s' registered with alias '%s'", tool.name, alias)
            return True
        except Exception as e:
            logging.error("Failed to register tool: %s", e)
            return False
    
    # ==================== 状态图管理 ====================
    
    def create_state_graph(
        self,
        name: str = None,
        state_class: Any = None,
        config: Dict[str, Any] = None
    ) -> Optional['StateGraph']:
        """创建状态图
        
        调用 StateGraph 类
        
        Args:
            name: 图名称
            state_class: 状态类 (默认为 AgentState)
            config: 图配置
            
        Returns:
            StateGraph 实例或 None
        """
        try:
            graph_name = name or f"graph_{uuid.uuid4().hex[:8]}"
            
            # 创建图配置
            graph_config = GraphConfig(
                name=graph_name,
                **(config or {})
            ) if config else GraphConfig(name=graph_name)
            
            # 创建状态图 (使用正确的参数: state_class 而非 state_schema)
            graph = StateGraph(
                state_class=state_class or AgentState,
                config=graph_config
            )
            
            logging.info("State graph '%s' created", graph_name)
            return graph
        except Exception as e:
            logging.error("Failed to create state graph: %s", e)
            return None
    
    def create_graph_config(
        self,
        name: str,
        max_iterations: int = 10,
        timeout: float = 300.0,
        enable_checkpointing: bool = True,
        enable_streaming: bool = False,
        **kwargs
    ) -> Optional['GraphConfig']:
        """创建图配置
        
        调用 GraphConfig 类
        
        Args:
            name: 配置名称
            max_iterations: 最大迭代次数
            timeout: 总超时时间 (秒)
            enable_checkpointing: 是否启用检查点
            enable_streaming: 是否启用流式输出
            **kwargs: 其他配置
            
        Returns:
            GraphConfig 实例或 None
        """
        try:
            # 使用正确的 GraphConfig 参数签名
            config = GraphConfig(
                name=name,
                max_iterations=max_iterations,
                timeout=timeout,
                enable_checkpointing=enable_checkpointing,
                enable_streaming=enable_streaming,
                **kwargs
            )
            
            logging.info("Graph config '%s' created", name)
            return config
        except Exception as e:
            logging.error("Failed to create graph config: %s", e)
            return None
    
    def get_graph_status(self, graph: 'StateGraph') -> Optional['GraphStatus']:
        """获取图状态
        
        调用 GraphStatus 枚举
        
        Args:
            graph: 状态图实例
            
        Returns:
            GraphStatus 枚举值或 None
        """
        try:
            # GraphStatus 是枚举，直接从 graph.status 获取
            if hasattr(graph, 'status'):
                return graph.status
            return GraphStatus.BUILDING
        except Exception as e:
            logging.error("Failed to get graph status: %s", e)
            return None
    
    # ==================== Agent 状态管理 ====================
    
    def get_langgraph_agent_status(self, agent_name: str) -> Optional['LangGraphAgentStatus']:
        """获取 LangGraph Agent 状态
        
        调用 LangGraphAgentStatus 枚举
        
        Args:
            agent_name: Agent 名称
            
        Returns:
            AgentStatus 枚举值或 None
        """
        if not self._langgraph_pool:
            return None
        
        try:
            agent = self._langgraph_pool.get(agent_name)
            if agent:
                # AgentStatus 是枚举，从 agent 的状态属性获取
                if hasattr(agent, 'state') and hasattr(agent.state, 'status'):
                    return agent.state.status
                elif hasattr(agent, 'status'):
                    return agent.status
                return LangGraphAgentStatus.IDLE
            return None
        except Exception as e:
            logging.error("Failed to get agent status: %s", e)
            return None
    
    def set_langgraph_agent_status(
        self,
        agent_name: str,
        status: str
    ) -> bool:
        """设置 LangGraph Agent 状态
        
        调用 LangGraphAgentStatus 枚举
        
        Args:
            agent_name: Agent 名称
            status: 状态值 (idle, running, paused, failed, completed, etc.)
            
        Returns:
            是否成功
        """
        if not self._langgraph_pool:
            return False
        
        try:
            agent = self._langgraph_pool.get(agent_name)
            if agent:
                # AgentStatus 是枚举，从字符串映射到枚举值
                status_map = {
                    "idle": LangGraphAgentStatus.IDLE,
                    "running": LangGraphAgentStatus.RUNNING,
                    "paused": LangGraphAgentStatus.PAUSED,
                    "completed": LangGraphAgentStatus.COMPLETED,
                    "failed": LangGraphAgentStatus.FAILED,
                    "cancelled": LangGraphAgentStatus.CANCELLED,
                    "timeout": LangGraphAgentStatus.TIMEOUT
                }
                new_status = status_map.get(status.lower(), LangGraphAgentStatus.IDLE)
                
                # 设置状态
                if hasattr(agent, 'state') and hasattr(agent.state, 'status'):
                    agent.state.status = new_status
                elif hasattr(agent, 'status'):
                    agent.status = new_status
                    
                logging.info("Agent '%s' status set to '%s'", agent_name, status)
                return True
            return False
        except Exception as e:
            logging.error("Failed to set agent status: %s", e)
            return False
    
    # ==================== 综合诊断 ====================
    
    def get_langgraph_health(self) -> Dict[str, Any]:
        """获取 LangGraph 组件综合健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # 检查工厂
        if self._master_factory:
            try:
                health["components"]["factory"] = self._master_factory.health_check()
            except Exception as e:
                health["components"]["factory"] = {"status": "error", "error": str(e)}
        else:
            health["components"]["factory"] = {"status": "not_initialized"}
        
        # 检查池
        if self._langgraph_pool:
            try:
                health["components"]["pool"] = self._langgraph_pool.get_pool_metrics()
            except Exception as e:
                health["components"]["pool"] = {"status": "error", "error": str(e)}
        else:
            health["components"]["pool"] = {"status": "not_initialized"}
        
        # 检查编排器
        if self._langgraph_orchestrator:
            try:
                health["components"]["orchestrator"] = {
                    "status": "healthy",
                    "execution_history_count": len(self._langgraph_orchestrator.get_execution_history(10))
                }
            except Exception as e:
                health["components"]["orchestrator"] = {"status": "error", "error": str(e)}
        else:
            health["components"]["orchestrator"] = {"status": "not_initialized"}
        
        return health
    
    def diagnose_langgraph_system(self) -> Dict[str, Any]:
        """执行 LangGraph 系统诊断
        
        Returns:
            诊断信息字典
        """
        diagnosis = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "capabilities": []
        }

        # 诊断工厂
        if self._master_factory:
            diagnosis["components"]["master_factory"] = "initialized"
            diagnosis["capabilities"].extend([
                "agent_creation",
                "message_creation",
                "memory_management",
                "tool_management",
                "checkpoint_management"
            ])
        
        # 诊断池
        if self._langgraph_pool:
            diagnosis["components"]["agent_pool"] = "initialized"
            diagnosis["capabilities"].extend([
                "agent_pooling",
                "load_balancing",
                "health_checking"
            ])
        
        # 诊断编排器
        if self._langgraph_orchestrator:
            diagnosis["components"]["orchestrator"] = "initialized"
            diagnosis["capabilities"].extend([
                "workflow_execution",
                "parallel_execution",
                "pipeline_management"
            ])
        
        # 诊断 modules/agent 层
        if self._inference_service:
            diagnosis["components"]["inference_service"] = "initialized"
            diagnosis["capabilities"].append("inference_via_modules_layer")
            
            # 获取 modules/agent 层的综合健康状态
            try:
                modules_health = self._inference_service.get_langgraph_comprehensive_health()
                diagnosis["modules_agent_health"] = modules_health
            except Exception as e:
                diagnosis["modules_agent_health"] = {"error": str(e)}
        
        return diagnosis
    
    # ==================== 通过 modules/agent 层调用的方法 ====================
    # 实现 services -> modules/agent -> algo 分层架构
    
    def create_langgraph_agent_via_modules(
        self,
        agent_type: str = "react",
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        **kwargs
    ) -> Optional['LangGraphBaseAgent']:
        """通过 modules/agent 层创建 LangGraph Agent
        
        调用 backend/modules/agent/langchain_inference_service 的工厂方法
        
        Args:
            agent_type: Agent 类型
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            **kwargs: 其他参数
            
        Returns:
            LangGraph Agent 实例或 None
        """
        if not self._inference_service:
            logging.warning("Inference service not available, falling back to direct LangGraph")
            return self.create_langgraph_agent(agent_type, name, tools, llm_client, **kwargs)
        
        try:
            agent = self._inference_service.create_langgraph_agent_via_factory(
                agent_type=agent_type,
                name=name or f"agent_{uuid.uuid4().hex[:8]}",
                tools=tools,
                llm_client=llm_client,
                **kwargs
            )
            
            if agent:
                logging.info(f"Created LangGraph agent via modules/agent layer: {agent.name}")
            
            return agent
        except Exception as e:
            logging.error(f"Failed to create agent via modules layer: {e}")
            return None
    
    def run_langgraph_agent_via_modules(
        self,
        agent_name: str,
        input_data: Any,
        **kwargs
    ) -> Optional['AgentState']:
        """通过 modules/agent 层运行 LangGraph Agent
        
        调用 backend/modules/agent/langchain_inference_service 的运行方法
        
        Args:
            agent_name: Agent 名称
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            AgentState 结果或 None
        """
        if not self._inference_service:
            logging.warning("Inference service not available, falling back to direct LangGraph")
            return self.run_langgraph_agent(agent_name, input_data, **kwargs)
        
        try:
            return self._inference_service.run_langgraph_agent(agent_name, input_data, **kwargs)
        except Exception as e:
            logging.error(f"Failed to run agent via modules layer: {e}")
            return None
    
    def run_workflow_via_modules(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> List['AgentState']:
        """通过 modules/agent 层执行工作流
        
        调用 backend/modules/agent/langchain_inference_service 的工作流方法
        
        Args:
            workflow: 工作流定义
            stop_on_error: 遇错是否停止
            
        Returns:
            结果列表
        """
        if not self._inference_service:
            logging.warning("Inference service not available, falling back to direct LangGraph")
            return self.run_agents_workflow(workflow, stop_on_error)
        
        try:
            return self._inference_service.run_agents_workflow(workflow, stop_on_error)
        except Exception as e:
            logging.error(f"Failed to run workflow via modules layer: {e}")
            return []
    
    def run_parallel_via_modules(
        self,
        tasks: List[Tuple[str, Any]],
        max_workers: int = 4
    ) -> List[Tuple[str, 'AgentState']]:
        """通过 modules/agent 层并行执行任务
        
        调用 backend/modules/agent/langchain_inference_service 的并行方法
        
        Args:
            tasks: 任务列表
            max_workers: 最大并行数
            
        Returns:
            结果列表
        """
        if not self._inference_service:
            logging.warning("Inference service not available, falling back to direct LangGraph")
            return self.run_agents_parallel(tasks, max_workers)
        
        try:
            return self._inference_service.run_agents_parallel(tasks, max_workers)
        except Exception as e:
            logging.error(f"Failed to run parallel via modules layer: {e}")
            return []
    
    def create_message_via_modules(
        self,
        content: str,
        message_type: str = "human",
        name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[Any]:
        """通过 modules/agent 层创建消息
        
        调用 backend/modules/agent/langchain_inference_service 的消息创建方法
        
        Args:
            content: 消息内容
            message_type: 消息类型
            name: 发送者名称
            metadata: 元数据
            
        Returns:
            AgentMessage 实例或 None
        """
        if not self._inference_service:
            return self.create_agent_message(content, message_type, name, metadata)
        
        try:
            return self._inference_service.create_agent_message_via_factory(
                content, message_type, name, metadata
            )
        except Exception as e:
            logging.error(f"Failed to create message via modules layer: {e}")
            return None
    
    def create_memory_entry_via_modules(
        self,
        content: Any,
        memory_type: str = "short_term",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> Optional[Any]:
        """通过 modules/agent 层创建记忆条目
        
        调用 backend/modules/agent/langchain_inference_service 的记忆创建方法
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            tags: 标签列表
            
        Returns:
            MemoryEntry 实例或 None
        """
        if not self._inference_service:
            return self.create_memory_entry(content, memory_type, importance, tags)
        
        try:
            return self._inference_service.create_memory_entry_via_factory(
                content, memory_type, importance, tags
            )
        except Exception as e:
            logging.error(f"Failed to create memory entry via modules layer: {e}")
            return None
    
    def create_tool_call_via_modules(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = None
    ) -> Optional[Any]:
        """通过 modules/agent 层创建工具调用
        
        调用 backend/modules/agent/langchain_inference_service 的工具调用创建方法
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            ToolCall 实例或 None
        """
        if not self._inference_service:
            return None
        
        try:
            return self._inference_service.create_tool_call_via_factory(
                tool_name, arguments, tool_call_id
            )
        except Exception as e:
            logging.error(f"Failed to create tool call via modules layer: {e}")
            return None
    
    def create_tool_result_via_modules(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: str = None
    ) -> Optional[Any]:
        """通过 modules/agent 层创建工具结果
        
        调用 backend/modules/agent/langchain_inference_service 的工具结果创建方法
        
        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            ToolResult 实例或 None
        """
        if not self._inference_service:
            return None
        
        try:
            return self._inference_service.create_tool_result_via_factory(
                tool_call_id, tool_name, result, success, error
            )
        except Exception as e:
            logging.error(f"Failed to create tool result via modules layer: {e}")
            return None
    
    def setup_production_features_via_modules(
        self,
        agent_name: str,
        enable_cache: bool = True,
        enable_retry: bool = True,
        enable_rate_limit: bool = True,
        cache_config: Dict[str, Any] = None,
        retry_config: Dict[str, Any] = None,
        rate_limit_config: Dict[str, Any] = None
    ) -> Dict[str, bool]:
        """通过 modules/agent 层设置生产级特性
        
        调用 backend/modules/agent/langchain_inference_service 的生产级特性设置方法
        
        Args:
            agent_name: Agent 名称
            enable_cache: 启用缓存
            enable_retry: 启用重试
            enable_rate_limit: 启用限流
            cache_config: 缓存配置
            retry_config: 重试配置
            rate_limit_config: 限流配置
            
        Returns:
            各特性设置结果
        """
        if not self._inference_service:
            # 回退到直接调用
            results = {}
            if enable_cache:
                results["cache"] = self.setup_tool_cache_for_langgraph_agent(
                    agent_name,
                    **(cache_config or {"max_size": 1000, "default_ttl": 300})
                )
            if enable_retry:
                results["retry"] = self.setup_retry_handler_for_langgraph_agent(
                    agent_name,
                    **(retry_config or {"max_retries": 3, "base_delay": 1.0})
                )
            return results
        
        results = {}
        
        if enable_cache:
            cache_cfg = cache_config or {"max_size": 1000, "default_ttl": 300}
            results["cache"] = self._inference_service.setup_agent_tool_cache(
                agent_name,
                max_size=cache_cfg.get("max_size", 1000),
                default_ttl=cache_cfg.get("default_ttl", 300)
            )
        
        if enable_retry:
            retry_cfg = retry_config or {"max_retries": 3, "base_delay": 1.0}
            results["retry"] = self._inference_service.setup_agent_retry_handler(
                agent_name,
                max_retries=retry_cfg.get("max_retries", 3),
                base_delay=retry_cfg.get("base_delay", 1.0),
                max_delay=retry_cfg.get("max_delay", 60.0)
            )
        
        if enable_rate_limit:
            rate_cfg = rate_limit_config or {"rate": 10.0, "burst": 20}
            results["rate_limit"] = self._inference_service.setup_agent_rate_limiter(
                agent_name,
                rate=rate_cfg.get("rate", 10.0),
                burst=rate_cfg.get("burst", 20)
            )
        
        return results
    
    def get_comprehensive_health_via_modules(self) -> Dict[str, Any]:
        """通过 modules/agent 层获取综合健康状态
        
        调用 backend/modules/agent/langchain_inference_service 的健康检查方法
        
        Returns:
            综合健康状态字典
        """
        health = {"timestamp": datetime.utcnow().isoformat(), "direct_langgraph": self.get_langgraph_health()}
        
        # 直接 LangGraph 健康

        # 通过 modules/agent 层的健康
        if self._inference_service:
            try:
                health["modules_agent"] = self._inference_service.get_langgraph_comprehensive_health()
            except Exception as e:
                health["modules_agent"] = {"error": str(e)}
        else:
            health["modules_agent"] = {"status": "not_initialized"}
        
        # 综合状态
        if health.get("direct_langgraph", {}).get("status") == "healthy" and \
           health.get("modules_agent", {}).get("status") == "healthy":
            health["overall_status"] = "healthy"
        elif "error" in health.get("direct_langgraph", {}) or "error" in health.get("modules_agent", {}):
            health["overall_status"] = "error"
        else:
            health["overall_status"] = "degraded"
        
        return health
    
    # ==================== 高层组合方法 - 调用新增派生方法 ====================
    
    def create_complete_agent_system(
        self,
        system_name: str,
        agent_configs: List[Dict[str, Any]],
        use_multi_agent: bool = True,
        checkpointer_type: str = "memory",
        enable_production_features: bool = True
    ) -> Dict[str, Any]:
        """创建完整的 Agent 系统
        
        组合调用新增派生方法：
        - create_langgraph_agent_config
        - create_react_agent_instance / create_plan_execute_agent_instance / create_reflexion_agent_instance
        - create_multi_agent_system_instance
        - create_memory_checkpointer_instance
        - setup_production_features_via_modules
        
        Args:
            system_name: 系统名称
            agent_configs: Agent 配置列表
            use_multi_agent: 是否使用多 Agent 系统
            checkpointer_type: 检查点器类型
            enable_production_features: 启用生产级特性
            
        Returns:
            系统创建结果字典
        """
        result = {
            "system_name": system_name,
            "created_at": datetime.utcnow().isoformat(),
            "agents": [],
            "checkpointer": None,
            "multi_agent_system": None,
            "production_features": {}
        }

        created_agents = []
        
        # 1. 为每个配置创建 Agent
        for config in agent_configs:
            agent_type = config.get("type", "react")
            agent_name = config.get("name", f"{agent_type}_{uuid.uuid4().hex[:8]}")
            tools = config.get("tools", [])
            llm_client = config.get("llm_client")
            
            # 创建 Agent 配置
            agent_config = self.create_langgraph_agent_config(
                name=agent_name,
                tools=tools,
                llm_client=llm_client,
                system_prompt=config.get("system_prompt", ""),
                max_iterations=config.get("max_iterations", 10),
                timeout=config.get("timeout", 300.0)
            )
            
            # 根据类型创建 Agent
            agent = None
            if agent_type == "react":
                agent = self.create_react_agent_instance(
                    name=agent_name,
                    tools=tools,
                    llm_client=llm_client,
                    config=config
                )
            elif agent_type == "plan_execute":
                agent = self.create_plan_execute_agent_instance(
                    name=agent_name,
                    tools=tools,
                    llm_client=llm_client,
                    planner_llm=config.get("planner_llm"),
                    config=config
                )
            elif agent_type == "reflexion":
                agent = self.create_reflexion_agent_instance(
                    name=agent_name,
                    tools=tools,
                    llm_client=llm_client,
                    max_reflections=config.get("max_reflections", 3),
                    quality_threshold=config.get("quality_threshold", 0.8),
                    config=config
                )
            else:
                # 使用工厂创建
                agent = self.create_agent_via_factory(
                    agent_type=agent_type,
                    name=agent_name,
                    tools=tools,
                    llm_client=llm_client
                )
            
            if agent:
                created_agents.append(agent)
                result["agents"].append({
                    "name": agent_name,
                    "type": agent_type,
                    "status": "created",
                    "config": agent_config is not None
                })
                
                # 3. 设置生产级特性
                if enable_production_features:
                    features = self.setup_production_features_via_modules(
                        agent_name=agent_name,
                        enable_cache=config.get("enable_cache", True),
                        enable_retry=config.get("enable_retry", True),
                        enable_rate_limit=config.get("enable_rate_limit", True)
                    )
                    result["production_features"][agent_name] = features
        
        # 2. 创建检查点器
        if checkpointer_type == "memory":
            checkpointer = self.create_memory_checkpointer_instance(
                max_checkpoints=100
            )
            if checkpointer:
                result["checkpointer"] = {
                    "type": "memory",
                    "status": "created"
                }
        
        # 4. 创建多 Agent 系统 (注意: MultiAgentSystem 需要 roles 而非 agents)
        if use_multi_agent and len(created_agents) > 1:
            # created_agents 是 Agent 实例列表,需要创建一个简化的多 Agent 编排
            # 直接使用编排器运行工作流，而非创建 MultiAgentSystem
            logging.info(
                "Created %d agents for system '%s', ready for workflow execution",
                len(created_agents), system_name
            )
            result["multi_agent_system"] = {
                "name": system_name,
                "agent_count": len(created_agents),
                "status": "ready_for_workflow"
            }
        
        result["status"] = "success" if created_agents else "failed"
        return result
    
    def create_agent_with_graph(
        self,
        agent_name: str,
        agent_type: str = "react",
        tools: List[Any] = None,
        llm_client: Any = None,
        graph_config: Dict[str, Any] = None,
        state_class: Any = None,
        enable_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """创建带状态图的 Agent
        
        组合调用新增派生方法：
        - create_graph_config
        - create_state_graph
        - create_react_agent_instance / create_agent_via_factory
        - create_state_checkpoint
        - get_graph_status
        
        Args:
            agent_name: Agent 名称
            agent_type: Agent 类型
            tools: 工具列表
            llm_client: LLM 客户端
            graph_config: 图配置字典
            state_class: 状态类 (默认为 AgentState)
            enable_checkpoint: 启用检查点
            
        Returns:
            创建结果字典
        """
        result = {
            "agent_name": agent_name,
            "created_at": datetime.utcnow().isoformat(),
            "agent": None,
            "graph": None,
            "config": None,
            "checkpoint": None
        }

        # 1. 创建图配置 (使用正确的参数名)
        config = self.create_graph_config(
            name=f"{agent_name}_graph_config",
            max_iterations=graph_config.get("max_iterations", 10) if graph_config else 10,
            timeout=graph_config.get("timeout", 300.0) if graph_config else 300.0,
            enable_checkpointing=enable_checkpoint
        )
        if config:
            result["config"] = {"name": f"{agent_name}_graph_config", "status": "created"}
        
        # 2. 创建状态图 (使用 state_class 而非 state_schema)
        graph = self.create_state_graph(
            name=f"{agent_name}_graph",
            state_class=state_class,
            config=graph_config
        )
        if graph:
            graph_info: Dict[str, Any] = {"name": f"{agent_name}_graph", "status": "created"}
            # 获取图状态
            graph_status = self.get_graph_status(graph)
            if graph_status:
                graph_info["graph_status"] = str(graph_status)
            result["graph"] = graph_info
        
        # 3. 创建 Agent
        if agent_type == "react":
            agent = self.create_react_agent_instance(
                name=agent_name,
                tools=tools,
                llm_client=llm_client
            )
        else:
            agent = self.create_agent_via_factory(
                agent_type=agent_type,
                name=agent_name,
                tools=tools,
                llm_client=llm_client
            )
        
        if agent:
            result["agent"] = {"name": agent_name, "type": agent_type, "status": "created"}
            
            # 4. 创建初始检查点
            if enable_checkpoint:
                checkpoint = self.create_state_checkpoint(
                    state={"agent_name": agent_name, "initialized": True},
                    checkpoint_id=f"{agent_name}_init",
                    metadata={"agent_type": agent_type, "created_at": datetime.utcnow().isoformat()}
                )
                if checkpoint:
                    result["checkpoint"] = {"id": f"{agent_name}_init", "status": "created"}
        
        result["status"] = "success" if agent else "failed"
        return result
    
    def register_custom_tools(
        self,
        tools: List[Dict[str, Any]],
        category: str = "custom"
    ) -> Dict[str, Any]:
        """批量注册自定义工具
        
        组合调用新增派生方法：
        - get_tool_registry
        - create_langgraph_tool
        - register_tool_to_registry
        
        Args:
            tools: 工具定义列表，每个包含 name, func, description
            category: 工具类别
            
        Returns:
            注册结果字典
        """
        result = {
            "category": category,
            "registered": [],
            "failed": [],
            "registry_status": None
        }

        # 获取工具注册表
        registry = self.get_tool_registry()
        if registry:
            result["registry_status"] = "available"
        else:
            result["registry_status"] = "not_available"
        
        for tool_def in tools:
            tool_name = tool_def.get("name")
            tool_func = tool_def.get("func")
            tool_desc = tool_def.get("description", "")
            tool_params = tool_def.get("parameters")
            return_direct = tool_def.get("return_direct", False)
            
            if not tool_name or not tool_func:
                result["failed"].append({
                    "name": tool_name or "unknown",
                    "error": "Missing name or func"
                })
                continue
            
            # 创建工具 (使用 parameters 而非 args_schema)
            tool = self.create_langgraph_tool(
                name=tool_name,
                func=tool_func,
                description=tool_desc,
                parameters=tool_params,
                return_direct=return_direct
            )
            
            if tool:
                # 注册到注册表
                success = self.register_tool_to_registry(tool, category)
                if success:
                    result["registered"].append({
                        "name": tool_name,
                        "category": category,
                        "status": "registered"
                    })
                else:
                    result["failed"].append({
                        "name": tool_name,
                        "error": "Failed to register"
                    })
            else:
                result["failed"].append({
                    "name": tool_name,
                    "error": "Failed to create tool"
                })
        
        result["total_registered"] = len(result["registered"])
        result["total_failed"] = len(result["failed"])
        return result
    
    def manage_agent_lifecycle(
        self,
        agent_name: str,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """管理 Agent 生命周期
        
        组合调用新增派生方法：
        - get_langgraph_agent_status
        - set_langgraph_agent_status
        - create_state_checkpoint
        - get_graph_status
        
        Args:
            agent_name: Agent 名称
            action: 动作 (start, pause, resume, stop, checkpoint, status)
            **kwargs: 其他参数
            
        Returns:
            操作结果字典
        """
        result = {
            "agent_name": agent_name,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }

        # 获取当前状态
        current_status = self.get_langgraph_agent_status(agent_name)
        if current_status:
            result["previous_status"] = str(current_status)
        
        if action == "start":
            success = self.set_langgraph_agent_status(agent_name, "running")
            result["success"] = success
            result["new_status"] = "running" if success else None
            
        elif action == "pause":
            # 创建暂停检查点
            checkpoint = self.create_state_checkpoint(
                state={"agent_name": agent_name, "paused": True},
                checkpoint_id=f"{agent_name}_pause_{uuid.uuid4().hex[:8]}",
                metadata={"action": "pause"}
            )
            success = self.set_langgraph_agent_status(agent_name, "paused")
            result["success"] = success
            result["new_status"] = "paused" if success else None
            result["checkpoint_created"] = checkpoint is not None
            
        elif action == "resume":
            success = self.set_langgraph_agent_status(agent_name, "running")
            result["success"] = success
            result["new_status"] = "running" if success else None
            
        elif action == "stop":
            # 创建停止检查点
            checkpoint = self.create_state_checkpoint(
                state={"agent_name": agent_name, "stopped": True},
                checkpoint_id=f"{agent_name}_stop_{uuid.uuid4().hex[:8]}",
                metadata={"action": "stop"}
            )
            success = self.set_langgraph_agent_status(agent_name, "completed")
            result["success"] = success
            result["new_status"] = "completed" if success else None
            result["checkpoint_created"] = checkpoint is not None
            
        elif action == "checkpoint":
            checkpoint = self.create_state_checkpoint(
                state=kwargs.get("state", {"agent_name": agent_name}),
                checkpoint_id=kwargs.get("checkpoint_id", f"{agent_name}_cp_{uuid.uuid4().hex[:8]}"),
                metadata=kwargs.get("metadata", {})
            )
            result["success"] = checkpoint is not None
            result["checkpoint"] = {
                "id": checkpoint.checkpoint_id if checkpoint else None,
                "created": checkpoint is not None
            }
            
        elif action == "status":
            status = self.get_langgraph_agent_status(agent_name)
            result["success"] = status is not None
            result["status"] = str(status) if status else None
            
        else:
            result["error"] = f"Unknown action: {action}"
            result["success"] = False
        
        return result
    
    def execute_multi_agent_task(
        self,
        system_name: str,
        task_input: Any,
        agents: List[Dict[str, Any]] = None,
        orchestration_type: str = "sequential",
        enable_events: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """执行多 Agent 任务
        
        组合调用新增派生方法：
        - create_multi_agent_system_instance
        - create_react_agent_instance / create_plan_execute_agent_instance
        - run_workflow_with_events
        - run_agents_parallel
        - get_langgraph_agent_status
        
        Args:
            system_name: 系统名称
            task_input: 任务输入
            agents: Agent 配置列表
            orchestration_type: 编排类型 (sequential, parallel)
            enable_events: 启用事件
            max_workers: 最大并行数
            
        Returns:
            执行结果字典
        """
        result = {
            "system_name": system_name,
            "started_at": datetime.utcnow().isoformat(),
            "agents_created": [],
            "execution_results": [],
            "events": []
        }

        created_agents = []
        
        # 1. 创建 Agents
        if agents:
            for agent_config in agents:
                agent_type = agent_config.get("type", "react")
                agent_name = agent_config.get("name", f"{agent_type}_{uuid.uuid4().hex[:8]}")
                
                if agent_type == "react":
                    agent = self.create_react_agent_instance(
                        name=agent_name,
                        tools=agent_config.get("tools"),
                        llm_client=agent_config.get("llm_client")
                    )
                elif agent_type == "plan_execute":
                    agent = self.create_plan_execute_agent_instance(
                        name=agent_name,
                        tools=agent_config.get("tools"),
                        llm_client=agent_config.get("llm_client")
                    )
                else:
                    agent = self.create_agent_via_factory(
                        agent_type=agent_type,
                        name=agent_name,
                        tools=agent_config.get("tools"),
                        llm_client=agent_config.get("llm_client")
                    )
                
                if agent:
                    created_agents.append((agent_name, agent))
                    result["agents_created"].append(agent_name)
        
        # 2. 创建多 Agent 系统
        if len(created_agents) > 1:
            mas = self.create_multi_agent_system_instance(
                name=system_name,
                roles=[a[1] for a in created_agents],
                collaboration_mode=orchestration_type
            )
            result["multi_agent_system"] = mas is not None
        
        # 3. 执行任务
        if created_agents:
            workflow = [(name, task_input) for name, _ in created_agents]
            
            if orchestration_type == "parallel":
                # 并行执行
                parallel_results = self.run_agents_parallel(workflow, max_workers)
                for name, state in parallel_results:
                    agent_status = self.get_langgraph_agent_status(name)
                    result["execution_results"].append({
                        "agent": name,
                        "status": str(agent_status) if agent_status else "unknown",
                        "output": state.output if state and hasattr(state, 'output') else None
                    })
            else:
                # 顺序执行（带事件）
                if enable_events:
                    states, events = self.run_workflow_with_events(workflow, stop_on_error=False)
                    for i, state in enumerate(states):
                        agent_name = created_agents[i][0] if i < len(created_agents) else f"agent_{i}"
                        agent_status = self.get_langgraph_agent_status(agent_name)
                        result["execution_results"].append({
                            "agent": agent_name,
                            "status": str(agent_status) if agent_status else "unknown",
                            "output": state.output if state and hasattr(state, 'output') else None
                        })
                    result["events"] = [str(e) for e in events]
                else:
                    states = self.run_agents_workflow(workflow, stop_on_error=False)
                    for i, state in enumerate(states):
                        agent_name = created_agents[i][0] if i < len(created_agents) else f"agent_{i}"
                        result["execution_results"].append({
                            "agent": agent_name,
                            "output": state.output if state and hasattr(state, 'output') else None
                        })
        
        result["completed_at"] = datetime.utcnow().isoformat()
        result["status"] = "success" if result["execution_results"] else "no_results"
        return result
    
    def get_full_system_diagnostics(self) -> Dict[str, Any]:
        """获取完整系统诊断
        
        Returns:
            完整诊断信息字典
        """
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "components": {},
            "capabilities": {},
            "health": {}
        }
        
        # 1. 基础系统诊断
        system_diag = self.diagnose_langgraph_system()
        diagnostics["system"] = system_diag
        
        # 2. 健康检查
        health = self.get_langgraph_health()
        diagnostics["health"]["langgraph"] = health
        
        # 3. modules/agent 层健康
        modules_health = self.get_comprehensive_health_via_modules()
        diagnostics["health"]["modules_agent"] = modules_health
        
        # 4. 组件状态
        # 工具注册表
        registry = self.get_tool_registry()
        diagnostics["components"]["tool_registry"] = {
            "available": registry is not None,
            "builtin_tools": len(self.get_available_builtin_tools())
        }
        
        # Agent 工厂
        factory = self.get_agent_factory()
        diagnostics["components"]["agent_factory"] = {
            "available": factory is not None
        }
        
        # 5. 能力清单
        diagnostics["capabilities"] = {
            "agent_types": ["react", "plan_execute", "reflexion", "multi_agent"],
            "checkpointer_types": ["memory"],
            "graph_features": ["state_graph", "graph_config", "graph_status"],
            "tool_features": ["tool_registry", "custom_tools", "builtin_tools"],
            "lifecycle_management": ["start", "pause", "resume", "stop", "checkpoint", "status"],
            "execution_modes": ["sequential", "parallel", "workflow", "pipeline"]
        }
        
        # 6. 实例统计
        diagnostics["instances"] = {
            "agent_instances": len(self.agent_instances),
            "checkpointers": len(self._checkpointers),
            "active_workflows": len(self.active_workflows)
        }
        
        return diagnostics


# 全局智能体实例管理器实例
agent_instance_manager = AgentInstanceManager()