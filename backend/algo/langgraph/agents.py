"""
Agent 实现 - 生产级

提供各种类型的智能 Agent，支持复杂任务处理和多 Agent 协作：

核心 Agent 类型：
- BaseAgent: Agent 基类，定义通用接口和功能
- ReActAgent: ReAct 推理-行动循环，支持工具调用
- PlanAndExecuteAgent: 计划执行 Agent，两阶段工作流
- ReflexionAgent: 反思 Agent，通过自我反思改进输出
- MultiAgentSystem: 多 Agent 系统，支持协作和分工

扩展 Agent 类型：
- ToolCallingAgent: 专注于工具调用的轻量级 Agent
- ConversationalAgent: 多轮对话型 Agent，支持上下文管理
- ChainOfThoughtAgent: 思维链 Agent，显式推理步骤
- SelfAskAgent: 自问自答 Agent，递归分解问题
- HierarchicalAgent: 层级 Agent，任务分解与委派
- AdaptiveAgent: 自适应 Agent，动态策略调整
- WorkflowAgent: 工作流 Agent，复杂流程编排

高级特性：
- AgentPool: Agent 池，管理多个 Agent 实例
- AgentOrchestrator: Agent 编排器，协调多 Agent 执行
- AgentRegistry: Agent 注册表，统一管理

生产级特性：
- 回调和钩子系统：事件通知、日志记录、自定义处理、Webhook 集成
- 执行指标和监控：延迟、Token 使用、成功率追踪、性能分析
- 错误处理：重试机制、超时控制、断路器、优雅降级
- 状态管理：检查点、恢复、持久化、分支、回滚
- 人机协作：中断点、审批流程、反馈集成
- 内存管理：短期/长期记忆、记忆压缩、实体提取

使用示例：
    # 创建 ReAct Agent
    agent = ReActAgent(
        config=AgentConfig(name="assistant", max_iterations=10),
        tools=[search_tool, calculator_tool],
        llm_client=openai_client
    )
    
    # 运行 Agent
    result = agent.invoke("帮我计算 123 + 456")
    print(result.final_answer)
    
    # 流式输出
    for chunk in agent.stream("搜索最新的 AI 新闻"):
        print(chunk, end="", flush=True)
    
    # 使用 Agent 池
    pool = AgentPool()
    pool.register("assistant", agent)
    result = pool.run("assistant", "帮我搜索天气")
    
    # 使用编排器
    orchestrator = AgentOrchestrator()
    orchestrator.add_agent(agent)
    result = orchestrator.run_workflow([("assistant", "任务1"), ("assistant", "任务2")])
"""

import logging
import asyncio
import time
import uuid
import json
import traceback
import copy
import hashlib
import queue
import signal
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    AsyncIterator, Iterator, TypeVar, Generic, Set, Type
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, Future
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import weakref

# 状态相关
from .state import (
    AgentState, AgentMessage, MessageType, ToolCall, ToolResult,
    AgentStatus, ExecutionMode as StateExecutionMode, InterruptType,
    ErrorType, ContextType, MemoryType, PriorityLevel,
    StateCheckpoint, StateManager, StateValidator, StateSerializer,
    StateEventEmitter, MessageBuffer, AgentMemory, MemoryEntry,
    PlanStep, AgentPlan, Reflection, ExecutionContext as StateExecutionContext
)

# 节点相关
from .nodes import (
    BaseNode, LLMNode, ToolNode, HumanNode, NodeFactory, LLMProvider,
    NodeConfig, NodeType, NodeMetrics, NodeRegistry, get_node_registry,
    TransformNode, BranchNode, ParallelNode, RetryNode, CacheNode,
    MemoryNode, ReflectionNode, PlanningNode, LoggingNode, MetricsNode,
    RateLimitNode, ValidationNode, NodeChain, NodeParallelGroup
)

# 边相关
from .edges import (
    Edge, ConditionalEdge, LoopEdge, ParallelEdge, RetryEdge,
    FallbackEdge, TimeoutEdge, CircuitBreakerEdge, WeightedEdge,
    EdgeType, EdgeCondition, EdgeConditions, EdgeManager,
    EdgeBuilder, edge_from, route_after_agent, route_after_tools,
    should_continue_condition, PriorityRouter, WeightedRouter,
    LoadBalanceRouter, ABTestRouter
)

# 
from .graph import (
    StateGraph, GraphBuilder, CompiledGraph, GraphConfig,
    GraphStatus, ExecutionMode as GraphExecutionMode,
    GraphMetrics, ExecutionEvent, GraphRunner, GraphAnalyzer,
    Subgraph, create_simple_graph, create_react_graph
)

# 
from .tools import (
    Tool, ToolParameter, ToolRegistry, ToolExecutor, ToolManager,
    ToolCategory, ToolStatus, ToolMetrics, ToolCache, ToolRateLimiter,
    RetryHandler, ToolHooks, get_global_registry, get_tool_manager,
    tool, async_tool
)

# 检查点相关
from .checkpointer import (
    Checkpointer, MemoryCheckpointer, RedisCheckpointer,
    SQLiteCheckpointer, FileCheckpointer, EnhancedCheckpoint,
    CheckpointTag, CheckpointDiff, CheckpointMetrics,
    CompressionType, get_checkpointer, create_memory_checkpointer
)

# 内置工具
from .builtin_tools import (
    get_builtin_tools, get_tools_by_category, get_tools_for_agent,
    get_tool_by_name, get_tool_info, ToolCategory as BuiltinToolCategory
)

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
AgentT = TypeVar('AgentT', bound='BaseAgent')


# ==================== 枚举和常量 ====================

class AgentEventType(Enum):
    """Agent 事件类型"""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    HUMAN_INPUT_REQUEST = "human_input_request"
    HUMAN_INPUT_RECEIVED = "human_input_received"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    STATE_UPDATE = "state_update"


class AgentMode(Enum):
    """Agent 运行模式"""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"


class StrategyType(Enum):
    """策略类型"""
    STATE = "state"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"
    TOOL = "tool"
    CHECKPOINT = "checkpoint"
    MEMORY = "memory"
    ROUTING = "routing"


# ==================== 策略模式 ====================

class BaseStrategy(ABC):
    """策略基类
    
    所有策略必须继承此类并实现 execute 方法。
    策略用于在运行时动态决定调用哪些方法。
    """
    
    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self._enabled = True
        self._metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_time_ms": 0.0
        }
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """执行策略
        
        Args:
            context: 执行上下文，包含必要的参数和状态
            
        Returns:
            策略执行结果
        """
        pass
    
    @abstractmethod
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """判断是否应该执行此策略
        
        Args:
            context: 执行上下文
            
        Returns:
            是否应该执行
        """
        pass
    
    def enable(self) -> None:
        """启用策略"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用策略"""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.copy()
    
    def _record_execution(self, success: bool, duration_ms: float) -> None:
        self._metrics["executions"] += 1
        if success:
            self._metrics["successes"] += 1
        else:
            self._metrics["failures"] += 1
        self._metrics["total_time_ms"] += duration_ms


class StateStrategy(BaseStrategy):
    """状态管理策略基类
    
    用于管理 Agent 状态的创建、更新、验证和持久化。
    """
    
    @abstractmethod
    def create_state(self, input_data: Any, **kwargs) -> AgentState:
        """创建初始状态"""
        pass
    
    @abstractmethod
    def update_state(self, state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """更新状态"""
        pass
    
    @abstractmethod
    def validate_state(self, state: AgentState) -> Tuple[bool, List[str]]:
        """验证状态"""
        pass


class DefaultStateStrategy(StateStrategy):
    """默认状态管理策略"""
    
    def __init__(self, 
                 name: str = "default_state",
                 config: Dict[str, Any] = None,
                 validator: StateValidator = None,
                 serializer: StateSerializer = None,
                 memory: AgentMemory = None):
        super().__init__(name, config)
        self._validator = validator or StateValidator()
        self._serializer = serializer or StateSerializer()
        self._memory = memory or AgentMemory()
        self._message_buffer = MessageBuffer(
            max_size=config.get("max_buffer_size", 100) if config else 100
        )
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and "input_data" in context
    
    def execute(self, context: Dict[str, Any]) -> AgentState:
        start_time = time.time()
        try:
            input_data = context.get("input_data")
            thread_id = context.get("thread_id")
            
            state = self.create_state(input_data, thread_id=thread_id)
            
            # 加载历史记忆
            if thread_id and self._memory:
                memories = self._memory.retrieve(thread_id, limit=10)
                for mem in memories:
                    state.data.setdefault("memories", []).append(mem.content)
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return state
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def create_state(self, input_data: Any, **kwargs) -> AgentState:
        """创建初始状态"""
        if isinstance(input_data, AgentState):
            return input_data
        
        thread_id = kwargs.get("thread_id") or str(uuid.uuid4())
        
        state = AgentState(
            input=input_data if isinstance(input_data, str) else str(input_data),
            thread_id=thread_id
        )
        
        if isinstance(input_data, str):
            state.add_message(AgentMessage.human(input_data))
        elif isinstance(input_data, dict):
            if "messages" in input_data:
                for msg in input_data["messages"]:
                    if isinstance(msg, AgentMessage):
                        state.add_message(msg)
                    elif isinstance(msg, dict):
                        state.add_message(AgentMessage.from_dict(msg))
            if "input" in input_data:
                state.input = input_data["input"]
                state.add_message(AgentMessage.human(input_data["input"]))
        
        return state
    
    def update_state(self, state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """更新状态"""
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.data[key] = value
        
        state.updated_at = datetime.utcnow()
        return state
    
    def validate_state(self, state: AgentState) -> Tuple[bool, List[str]]:
        """验证状态"""
        return self._validator.validate(state)
    
    def serialize_state(self, state: AgentState) -> str:
        """序列化状态"""
        return self._serializer.serialize(state)
    
    def deserialize_state(self, data: str) -> AgentState:
        """反序列化状态"""
        return self._serializer.deserialize(data)


class NodeStrategy(BaseStrategy):
    """节点策略基类
    
    用于选择和创建执行节点。
    """
    
    @abstractmethod
    def select_node(self, state: AgentState, available_nodes: List[str]) -> str:
        """选择下一个节点"""
        pass
    
    @abstractmethod
    def create_node(self, node_type: str, config: Dict[str, Any]) -> BaseNode:
        """创建节点"""
        pass


class DefaultNodeStrategy(NodeStrategy):
    """默认节点策略"""
    
    def __init__(self, 
                 name: str = "default_node",
                 config: Dict[str, Any] = None,
                 node_factory: NodeFactory = None,
                 node_registry: NodeRegistry = None):
        super().__init__(name, config)
        self._factory = node_factory or NodeFactory()
        self._registry = node_registry or get_node_registry()
        self._node_cache: Dict[str, BaseNode] = {}
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and ("node_type" in context or "state" in context)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        start_time = time.time()
        try:
            if "node_type" in context:
                result = self.create_node(context["node_type"], context.get("node_config", {}))
            else:
                state = context.get("state")
                available = context.get("available_nodes", [])
                result = self.select_node(state, available)
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def select_node(self, state: AgentState, available_nodes: List[str]) -> str:
        """基于状态选择下一个节点"""
        if not available_nodes:
            return "__end__"
        
        # 检查是否有待处理的工具调用
        if state.pending_tool_calls:
            if "tools" in available_nodes:
                return "tools"
        
        # 检查是否需要人工介入
        if state.requires_human_input:
            if "human" in available_nodes:
                return "human"
        
        # 检查是否完成
        if state.final_answer:
            return "__end__"
        
        # 检查迭代限制
        if state.current_iteration >= state.max_iterations:
            return "__end__"
        
        # 默认返回第一个可用节点
        return available_nodes[0]
    
    def create_node(self, node_type: str, config: Dict[str, Any]) -> BaseNode:
        """创建节点"""
        # 尝试从缓存获取
        cache_key = f"{node_type}_{hash(json.dumps(config, sort_keys=True, default=str))}"
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]
        
        # 创建节点
        node_config = NodeConfig(**config) if config else NodeConfig()
        
        node_creators = {
            "llm": lambda: LLMNode(name=config.get("name", "llm"), **config),
            "tool": lambda: ToolNode(name=config.get("name", "tools"), **config),
            "human": lambda: HumanNode(name=config.get("name", "human"), **config),
            "transform": lambda: TransformNode(name=config.get("name", "transform"), **config),
            "branch": lambda: BranchNode(name=config.get("name", "branch"), **config),
            "parallel": lambda: ParallelNode(name=config.get("name", "parallel"), **config),
            "retry": lambda: RetryNode(name=config.get("name", "retry"), **config),
            "cache": lambda: CacheNode(name=config.get("name", "cache"), **config),
            "memory": lambda: MemoryNode(name=config.get("name", "memory"), **config),
            "reflection": lambda: ReflectionNode(name=config.get("name", "reflection"), **config),
            "planning": lambda: PlanningNode(name=config.get("name", "planning"), **config),
            "logging": lambda: LoggingNode(name=config.get("name", "logging"), **config),
            "metrics": lambda: MetricsNode(name=config.get("name", "metrics"), **config),
            "rate_limit": lambda: RateLimitNode(name=config.get("name", "rate_limit"), **config),
            "validation": lambda: ValidationNode(name=config.get("name", "validation"), **config),
        }
        
        creator = node_creators.get(node_type.lower())
        if creator:
            node = creator()
            self._node_cache[cache_key] = node
            return node
        
        raise ValueError(f"Unknown node type: {node_type}")
    
    def create_node_chain(self, node_configs: List[Dict[str, Any]]) -> NodeChain:
        """创建节点链"""
        nodes = [self.create_node(cfg.get("type", "transform"), cfg) for cfg in node_configs]
        return NodeChain(nodes=nodes)
    
    def create_parallel_group(self, node_configs: List[Dict[str, Any]], 
                             merge_strategy: str = "merge") -> NodeParallelGroup:
        """创建并行节点组"""
        nodes = [self.create_node(cfg.get("type", "transform"), cfg) for cfg in node_configs]
        return NodeParallelGroup(nodes=nodes, merge_strategy=merge_strategy)


class EdgeStrategy(BaseStrategy):
    """边策略基类
    
    用于决定状态转移和路由。
    """
    
    @abstractmethod
    def route(self, state: AgentState, edges: List[Edge]) -> str:
        """路由到下一个节点"""
        pass
    
    @abstractmethod
    def create_edge(self, edge_type: str, **kwargs) -> Edge:
        """创建边"""
        pass


class DefaultEdgeStrategy(EdgeStrategy):
    """默认边策略"""
    
    def __init__(self, 
                 name: str = "default_edge",
                 config: Dict[str, Any] = None,
                 edge_manager: EdgeManager = None):
        super().__init__(name, config)
        self._manager = edge_manager or EdgeManager()
        self._routers = {
            "priority": PriorityRouter(),
            "weighted": WeightedRouter(),
            "load_balance": LoadBalanceRouter(),
            "ab_test": ABTestRouter()
        }
        self._edge_builder = EdgeBuilder()
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and "state" in context
    
    def execute(self, context: Dict[str, Any]) -> str:
        start_time = time.time()
        try:
            state = context.get("state")
            edges = context.get("edges", [])
            router_type = context.get("router_type", "priority")
            
            result = self.route(state, edges, router_type)
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def route(self, state: AgentState, edges: List[Edge], 
             router_type: str = "priority") -> str:
        """路由到下一个节点"""
        if not edges:
            return "__end__"
        
        # 使用指定的路由器
        router = self._routers.get(router_type)
        if router:
            return router.route(state, edges)
        
        # 默认：评估条件边
        for edge in edges:
            if isinstance(edge, ConditionalEdge):
                if edge.evaluate(state):
                    return edge.target
            else:
                return edge.target
        
        return "__end__"
    
    def create_edge(self, edge_type: str, **kwargs) -> Edge:
        """创建边"""
        edge_creators = {
            "simple": lambda: Edge(**kwargs),
            "conditional": lambda: ConditionalEdge(**kwargs),
            "loop": lambda: LoopEdge(**kwargs),
            "parallel": lambda: ParallelEdge(**kwargs),
            "retry": lambda: RetryEdge(**kwargs),
            "fallback": lambda: FallbackEdge(**kwargs),
            "timeout": lambda: TimeoutEdge(**kwargs),
            "circuit_breaker": lambda: CircuitBreakerEdge(**kwargs),
            "weighted": lambda: WeightedEdge(**kwargs),
        }
        
        creator = edge_creators.get(edge_type.lower())
        if creator:
            return creator()
        
        raise ValueError(f"Unknown edge type: {edge_type}")
    
    def create_conditional_edge(self, source: str, condition: Callable, 
                               branches: Dict[str, str]) -> ConditionalEdge:
        """创建条件边"""
        return ConditionalEdge(
            source=source,
            condition=condition,
            branches=branches
        )
    
    def should_continue(self, state: AgentState) -> str:
        """判断是否继续执行"""
        return should_continue_condition(state)


class GraphStrategy(BaseStrategy):
    """图策略基类
    
    用于构建和管理执行图。
    """
    
    @abstractmethod
    def build_graph(self, nodes: List[BaseNode], edges: List[Edge]) -> StateGraph:
        """构建图"""
        pass
    
    @abstractmethod
    def execute_graph(self, graph: CompiledGraph, state: AgentState) -> AgentState:
        """执行图"""
        pass


class DefaultGraphStrategy(GraphStrategy):
    """默认图策略"""
    
    def __init__(self, 
                 name: str = "default_graph",
                 config: Dict[str, Any] = None,
                 graph_runner: GraphRunner = None):
        super().__init__(name, config)
        self._runner = graph_runner or GraphRunner()
        self._analyzer = GraphAnalyzer()
        self._builder = GraphBuilder()
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and ("nodes" in context or "graph" in context)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        start_time = time.time()
        try:
            if "graph" in context and "state" in context:
                result = self.execute_graph(context["graph"], context["state"])
            else:
                nodes = context.get("nodes", [])
                edges = context.get("edges", [])
                result = self.build_graph(nodes, edges)
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def build_graph(self, nodes: List[BaseNode], edges: List[Edge]) -> StateGraph:
        """构建图"""
        graph_config = GraphConfig(
            name=self.config.get("graph_name", "agent_graph"),
            max_iterations=self.config.get("max_iterations", 20),
            execution_mode=GraphExecutionMode(self.config.get("execution_mode", "sequential"))
        )
        
        graph = StateGraph(config=graph_config)
        
        # 添加节点
        for node in nodes:
            graph.add_node(node.name, node)
        
        # 添加边
        for edge in edges:
            if isinstance(edge, ConditionalEdge):
                graph.add_conditional_edges(
                    edge.source,
                    edge.condition,
                    edge.branches
                )
            else:
                graph.add_edge(edge.source, edge.target)
        
        return graph
    
    def execute_graph(self, graph: CompiledGraph, state: AgentState) -> AgentState:
        """执行图"""
        return self._runner.run(graph, state)
    
    async def execute_graph_async(self, graph: CompiledGraph, 
                                  state: AgentState) -> AgentState:
        """异步执行图"""
        return await self._runner.run_async(graph, state)
    
    def analyze_graph(self, graph: StateGraph) -> Dict[str, Any]:
        """分析图结构"""
        return self._analyzer.analyze(graph)
    
    def create_subgraph(self, name: str, parent_graph: StateGraph, 
                       node_names: List[str]) -> Subgraph:
        """创建子图"""
        return Subgraph(name=name, parent=parent_graph, nodes=node_names)


class ToolStrategy(BaseStrategy):
    """工具策略基类
    
    用于选择和执行工具。
    """
    
    @abstractmethod
    def select_tools(self, state: AgentState, available_tools: List[Tool]) -> List[Tool]:
        """选择工具"""
        pass
    
    @abstractmethod
    def execute_tool(self, tool: Tool, **kwargs) -> Any:
        """执行工具"""
        pass


class DefaultToolStrategy(ToolStrategy):
    """默认工具策略"""
    
    def __init__(self, 
                 name: str = "default_tool",
                 config: Dict[str, Any] = None,
                 tool_manager: ToolManager = None,
                 tool_registry: ToolRegistry = None):
        super().__init__(name, config)
        self._manager = tool_manager or get_tool_manager()
        self._registry = tool_registry or get_global_registry()
        self._executor = ToolExecutor()
        self._cache = ToolCache()
        self._rate_limiter = ToolRateLimiter()
        self._retry_handler = RetryHandler()
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and ("tool_calls" in context or "state" in context)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        start_time = time.time()
        try:
            if "tool" in context:
                result = self.execute_tool(context["tool"], **context.get("tool_args", {}))
            elif "tool_calls" in context:
                result = self.execute_tool_calls(context["tool_calls"])
            else:
                state = context.get("state")
                available = context.get("available_tools", [])
                result = self.select_tools(state, available)
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def select_tools(self, state: AgentState, available_tools: List[Tool]) -> List[Tool]:
        """基于状态选择工具"""
        if not state.pending_tool_calls:
            return []
        
        selected = []
        for tool_call in state.pending_tool_calls:
            for tool in available_tools:
                if tool.name == tool_call.name:
                    selected.append(tool)
                    break
        
        return selected
    
    def execute_tool(self, tool: Tool, **kwargs) -> Any:
        """执行单个工具"""
        # 检查速率限制
        if not self._rate_limiter.check(tool.name):
            raise RuntimeError(f"Rate limit exceeded for tool: {tool.name}")
        
        # 检查缓存
        cache_key = f"{tool.name}_{hash(json.dumps(kwargs, sort_keys=True, default=str))}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # 执行工具
        result = self._retry_handler.execute(
            lambda: self._executor.execute(tool, **kwargs),
            max_retries=self.config.get("max_retries", 3)
        )
        
        # 缓存结果
        if tool.cacheable:
            self._cache.set(cache_key, result, ttl=tool.cache_ttl)
        
        return result
    
    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """执行多个工具调用"""
        results = []
        for call in tool_calls:
            tool = self._registry.get(call.name)
            if tool:
                try:
                    output = self.execute_tool(tool, **call.arguments)
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        result=str(output),
                        success=True
                    ))
                except Exception as e:
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        result=str(e),
                        success=False,
                        error=str(e)
                    ))
            else:
                results.append(ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    result=f"Tool not found: {call.name}",
                    success=False,
                    error=f"Tool not found: {call.name}"
                ))
        return results


    def get_builtin_tools(self, category: str = None) -> List[Tool]:
        """获取内置工具"""
        if category:
            return get_tools_by_category(BuiltinToolCategory(category))
        return get_builtin_tools()

    
    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """获取适合特定 Agent 类型的工具"""
        return get_tools_for_agent(agent_type)


class CheckpointStrategy(BaseStrategy):
    """检查点策略基类
    
    用于管理状态的持久化和恢复。
    """
    
    @abstractmethod
    def save(self, state: AgentState, checkpoint_id: str = None) -> str:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """加载检查点"""
        pass
    
    @abstractmethod
    def list_checkpoints(self, thread_id: str = None) -> List[str]:
        """列出检查点"""
        pass


class DefaultCheckpointStrategy(CheckpointStrategy):
    """默认检查点策略"""
    
    def __init__(self, 
                 name: str = "default_checkpoint",
                 config: Dict[str, Any] = None,
                 checkpointer: Checkpointer = None):
        super().__init__(name, config)
        self._checkpointer = checkpointer or create_memory_checkpointer()
        self._compression = CompressionType(config.get("compression", "none")) if config else CompressionType.NONE
        self._tags: Dict[str, CheckpointTag] = {}
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and ("state" in context or "checkpoint_id" in context)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        start_time = time.time()
        try:
            action = context.get("action", "save")
            
            if action == "save":
                state = context.get("state")
                checkpoint_id = context.get("checkpoint_id")
                result = self.save(state, checkpoint_id)
            elif action == "load":
                checkpoint_id = context.get("checkpoint_id")
                result = self.load(checkpoint_id)
            elif action == "list":
                thread_id = context.get("thread_id")
                result = self.list_checkpoints(thread_id)
            elif action == "diff":
                id1 = context.get("checkpoint_id_1")
                id2 = context.get("checkpoint_id_2")
                result = self.diff(id1, id2)
            elif action == "rollback":
                checkpoint_id = context.get("checkpoint_id")
                result = self.rollback(checkpoint_id)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def save(self, state: AgentState, checkpoint_id: str = None) -> str:
        """保存检查点"""
        checkpoint_id = checkpoint_id or str(uuid.uuid4())
        
        checkpoint = EnhancedCheckpoint(
            checkpoint_id=checkpoint_id,
            thread_id=state.thread_id,
            state=state,
            timestamp=datetime.utcnow(),
            metadata={
                "iteration": state.current_iteration,
                "status": state.status.value if state.status else "unknown"
            }
        )
        
        self._checkpointer.save(checkpoint)
        return checkpoint_id
    
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """加载检查点"""
        checkpoint = self._checkpointer.load(checkpoint_id)
        if checkpoint:
            return checkpoint.state
        return None
    
    def list_checkpoints(self, thread_id: str = None) -> List[str]:
        """列出检查点"""
        return self._checkpointer.list(thread_id)
    
    def delete(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        return self._checkpointer.delete(checkpoint_id)
    
    def tag(self, checkpoint_id: str, tag_name: str, 
           description: str = None) -> CheckpointTag:
        """为检查点添加标签"""
        tag = CheckpointTag(
            name=tag_name,
            checkpoint_id=checkpoint_id,
            description=description,
            created_at=datetime.utcnow()
        )
        self._tags[tag_name] = tag
        return tag
    
    def get_by_tag(self, tag_name: str) -> Optional[AgentState]:
        """通过标签获取检查点"""
        tag = self._tags.get(tag_name)
        if tag:
            return self.load(tag.checkpoint_id)
        return None
    
    def diff(self, checkpoint_id_1: str, checkpoint_id_2: str) -> Optional[CheckpointDiff]:
        """比较两个检查点"""
        cp1 = self._checkpointer.load(checkpoint_id_1)
        cp2 = self._checkpointer.load(checkpoint_id_2)
        
        if not cp1 or not cp2:
            return None
        
        return CheckpointDiff(
            checkpoint_id_1=checkpoint_id_1,
            checkpoint_id_2=checkpoint_id_2,
            changes=self._compute_changes(cp1.state, cp2.state)
        )
    
    def _compute_changes(self, state1: AgentState, state2: AgentState) -> Dict[str, Any]:
        """计算状态变化"""
        changes = {}
        
        # 比较消息
        msgs1 = len(state1.messages) if state1.messages else 0
        msgs2 = len(state2.messages) if state2.messages else 0
        if msgs1 != msgs2:
            changes["messages_added"] = msgs2 - msgs1
        
        # 比较迭代
        if state1.current_iteration != state2.current_iteration:
            changes["iteration_diff"] = state2.current_iteration - state1.current_iteration
        
        # 比较状态
        if state1.status != state2.status:
            changes["status_change"] = {
                "from": state1.status.value if state1.status else None,
                "to": state2.status.value if state2.status else None
            }
        
        return changes
    
    def rollback(self, checkpoint_id: str) -> Optional[AgentState]:
        """回滚到指定检查点"""
        return self.load(checkpoint_id)
    
    def get_metrics(self) -> CheckpointMetrics:
        """获取检查点指标"""
        return self._checkpointer.get_metrics()


class MemoryStrategy(BaseStrategy):
    """记忆策略基类
    
    用于管理 Agent 的短期和长期记忆。
    """
    
    @abstractmethod
    def store(self, key: str, value: Any, memory_type: MemoryType = MemoryType.SHORT_TERM) -> None:
        """存储记忆"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str, memory_type: MemoryType = None) -> Optional[Any]:
        """检索记忆"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """搜索相关记忆"""
        pass


class DefaultMemoryStrategy(MemoryStrategy):
    """默认记忆策略"""
    
    def __init__(self, 
                 name: str = "default_memory",
                 config: Dict[str, Any] = None,
                 memory: AgentMemory = None):
        super().__init__(name, config)
        self._memory = memory or AgentMemory()
        self._short_term: Dict[str, Any] = {}
        self._long_term: Dict[str, MemoryEntry] = {}
        self._working: Dict[str, Any] = {}
        self._buffer = MessageBuffer(max_size=config.get("buffer_size", 50) if config else 50)
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and ("key" in context or "query" in context)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        start_time = time.time()
        try:
            action = context.get("action", "retrieve")
            
            if action == "store":
                self.store(
                    context["key"], 
                    context["value"],
                    MemoryType(context.get("memory_type", "short_term"))
                )
                result = True
            elif action == "retrieve":
                memory_type = MemoryType(context["memory_type"]) if context.get("memory_type") else None
                result = self.retrieve(context["key"], memory_type)
            elif action == "search":
                result = self.search(context["query"], context.get("limit", 10))
            elif action == "consolidate":
                result = self.consolidate()
            else:
                raise ValueError(f"Unknown action: {action}")
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def store(self, key: str, value: Any, memory_type: MemoryType = MemoryType.SHORT_TERM) -> None:
        """存储记忆"""
        if memory_type == MemoryType.SHORT_TERM:
            self._short_term[key] = value
        elif memory_type == MemoryType.LONG_TERM:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=value,
                memory_type=memory_type,
                created_at=datetime.utcnow(),
                importance=0.5
            )
            self._long_term[key] = entry
        elif memory_type == MemoryType.WORKING:
            self._working[key] = value
    
    def retrieve(self, key: str, memory_type: MemoryType = None) -> Optional[Any]:
        """检索记忆"""
        if memory_type == MemoryType.SHORT_TERM or memory_type is None:
            if key in self._short_term:
                return self._short_term[key]
        
        if memory_type == MemoryType.LONG_TERM or memory_type is None:
            if key in self._long_term:
                return self._long_term[key].content
        
        if memory_type == MemoryType.WORKING or memory_type is None:
            if key in self._working:
                return self._working[key]
        
        return None
    
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """搜索相关记忆"""
        return self._memory.search(query, limit)
    
    def consolidate(self) -> int:
        """整合短期记忆到长期记忆"""
        consolidated = 0
        
        # 将重要的短期记忆移到长期
        for key, value in list(self._short_term.items()):
            if self._is_important(key, value):
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=value,
                    memory_type=MemoryType.LONG_TERM,
                    created_at=datetime.utcnow(),
                    importance=0.7
                )
                self._long_term[key] = entry
                del self._short_term[key]
                consolidated += 1
        
        return consolidated
    
    def _is_important(self, key: str, value: Any) -> bool:
        """判断记忆是否重要"""
        # 简单启发式：包含特定关键词
        important_keywords = ["error", "result", "answer", "conclusion", "decision"]
        content = str(value).lower()
        return any(kw in content for kw in important_keywords)
    
    def clear_short_term(self) -> None:
        """清除短期记忆"""
        self._short_term.clear()
    
    def clear_working(self) -> None:
        """清除工作记忆"""
        self._working.clear()


class RoutingStrategy(BaseStrategy):
    """路由策略基类
    
    用于决定消息和任务的路由。
    """
    
    @abstractmethod
    def route(self, state: AgentState, destinations: List[str]) -> str:
        """路由到目标"""
        pass


class DefaultRoutingStrategy(RoutingStrategy):
    """默认路由策略"""
    
    def __init__(self, 
                 name: str = "default_routing",
                 config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._routers = {
            "priority": PriorityRouter(),
            "weighted": WeightedRouter(),
            "load_balance": LoadBalanceRouter(),
            "ab_test": ABTestRouter()
        }
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        return self._enabled and "state" in context and "destinations" in context
    
    def execute(self, context: Dict[str, Any]) -> str:
        start_time = time.time()
        try:
            state = context.get("state")
            destinations = context.get("destinations", [])
            result = self.route(state, destinations)
            
            self._record_execution(True, (time.time() - start_time) * 1000)
            return result
        except Exception as e:
            self._record_execution(False, (time.time() - start_time) * 1000)
            raise
    
    def route(self, state: AgentState, destinations: List[str]) -> str:
        """路由到目标"""
        if not destinations:
            return "__end__"
        
        # 基于状态的路由逻辑
        if state.final_answer:
            return "__end__"
        
        if state.pending_tool_calls and "tools" in destinations:
            return "tools"
        
        if state.requires_human_input and "human" in destinations:
            return "human"
        
        if state.error and "error_handler" in destinations:
            return "error_handler"
        
        # 使用预定义条件
        return route_after_agent(state)
    
    def use_router(self, router_type: str, state: AgentState, 
                  destinations: List[str]) -> str:
        """使用特定路由器"""
        router = self._routers.get(router_type)
        if router:
            return router.route(state, destinations)
        return destinations[0] if destinations else "__end__"


# ==================== 策略管理器 ====================

class StrategyManager:
    """策略管理器
    
    统一管理和协调各种策略的执行。
    """
    
    def __init__(self):
        self._strategies: Dict[StrategyType, BaseStrategy] = {}
        self._default_strategies: Dict[StrategyType, Type[BaseStrategy]] = {
            StrategyType.STATE: DefaultStateStrategy,
            StrategyType.NODE: DefaultNodeStrategy,
            StrategyType.EDGE: DefaultEdgeStrategy,
            StrategyType.GRAPH: DefaultGraphStrategy,
            StrategyType.TOOL: DefaultToolStrategy,
            StrategyType.CHECKPOINT: DefaultCheckpointStrategy,
            StrategyType.MEMORY: DefaultMemoryStrategy,
            StrategyType.ROUTING: DefaultRoutingStrategy,
        }
        self._execution_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def register(self, strategy_type: StrategyType, strategy: BaseStrategy) -> None:
        """注册策略"""
        with self._lock:
            self._strategies[strategy_type] = strategy
    
    def unregister(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """注销策略"""
        with self._lock:
            return self._strategies.pop(strategy_type, None)
    
    def get(self, strategy_type: StrategyType) -> BaseStrategy:
        """获取策略，如果不存在则创建默认策略"""
        with self._lock:
            if strategy_type not in self._strategies:
                default_class = self._default_strategies.get(strategy_type)
                if default_class:
                    self._strategies[strategy_type] = default_class()
            return self._strategies.get(strategy_type)
    
    def execute(self, strategy_type: StrategyType, 
               context: Dict[str, Any]) -> Any:
        """执行策略"""
        strategy = self.get(strategy_type)
        if not strategy:
            raise ValueError(f"No strategy registered for type: {strategy_type}")
        
        if not strategy.is_enabled:
            raise RuntimeError(f"Strategy {strategy.name} is disabled")
        
        if not strategy.should_execute(context):
            return None
        
        start_time = time.time()
        try:
            result = strategy.execute(context)
            
            with self._lock:
                self._execution_history.append({
                    "strategy_type": strategy_type.value,
                    "strategy_name": strategy.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_ms": (time.time() - start_time) * 1000,
                    "success": True
                })
            
            return result
        except Exception as e:
            with self._lock:
                self._execution_history.append({
                    "strategy_type": strategy_type.value,
                    "strategy_name": strategy.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "error": str(e)
                })
            raise
    
    def execute_chain(self, 
                     chain: List[Tuple[StrategyType, Dict[str, Any]]]) -> List[Any]:
        """执行策略链"""
        results = []
        for strategy_type, context in chain:
            result = self.execute(strategy_type, context)
            results.append(result)
            # 将结果传递给下一个策略
            if isinstance(result, AgentState):
                context["state"] = result
        return results
    
    def execute_parallel(self, 
                        tasks: List[Tuple[StrategyType, Dict[str, Any]]],
                        max_workers: int = 4) -> List[Any]:
        """并行执行多个策略"""
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.execute, st, ctx): i
                for i, (st, ctx) in enumerate(tasks)
            }
            
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
        
        return results
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略的指标"""
        metrics = {}
        for strategy_type, strategy in self._strategies.items():
            metrics[strategy_type.value] = strategy.get_metrics()
        return metrics
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取执行历史"""
        with self._lock:
            return self._execution_history[-limit:]
    
    def reset_all(self) -> None:
        """重置所有策略"""
        with self._lock:
            for strategy in self._strategies.values():
                strategy._metrics = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_time_ms": 0.0
                }
            self._execution_history.clear()


# 全局策略管理器
_global_strategy_manager: Optional[StrategyManager] = None
_strategy_manager_lock = threading.Lock()


def get_strategy_manager() -> StrategyManager:
    """获取全局策略管理器"""
    global _global_strategy_manager
    
    if _global_strategy_manager is None:
        with _strategy_manager_lock:
            if _global_strategy_manager is None:
                _global_strategy_manager = StrategyManager()
    
    return _global_strategy_manager


def set_strategy_manager(manager: StrategyManager) -> None:
    """设置全局策略管理器"""
    global _global_strategy_manager
    with _strategy_manager_lock:
        _global_strategy_manager = manager


# ==================== 策略装饰器 ====================

def with_strategy(strategy_type: StrategyType):
    """策略装饰器 - 为方法添加策略执行"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            manager = getattr(self, '_strategy_manager', None) or get_strategy_manager()
            strategy = manager.get(strategy_type)
            
            if strategy and strategy.is_enabled:
                # 构建上下文
                context = {
                    "args": args,
                    "kwargs": kwargs,
                    "self": self
                }
                
                if strategy.should_execute(context):
                    # 让策略预处理
                    pre_result = strategy.execute(context)
                    if pre_result is not None:
                        kwargs['_strategy_result'] = pre_result
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def strategy_based(strategy_type: StrategyType, 
                  fallback: Callable = None):
    """策略基础装饰器 - 完全由策略控制执行"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            manager = getattr(self, '_strategy_manager', None) or get_strategy_manager()
            strategy = manager.get(strategy_type)
            
            context = {
                "args": args,
                "kwargs": kwargs,
                "self": self,
                "original_func": func
            }
            
            if strategy and strategy.is_enabled and strategy.should_execute(context):
                return strategy.execute(context)
            elif fallback:
                return fallback(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ==================== 回调系统 ====================

@dataclass
class AgentEvent:
    """Agent 事件"""
    event_type: AgentEventType
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    state: Optional[AgentState] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "duration_ms": self.duration_ms,
            "error": str(self.error) if self.error else None
        }


class AgentCallback(ABC):
    """Agent 回调基类"""
    
    @abstractmethod
    def on_event(self, event: AgentEvent) -> None:
        """处理事件"""
        pass


class LoggingCallback(AgentCallback):
    """日志回调"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(f"{__name__}.callback")
        self.log_level = log_level
    
    def on_event(self, event: AgentEvent) -> None:
        msg = f"[{event.agent_name}] {event.event_type.value}"
        if event.duration_ms > 0:
            msg += f" ({event.duration_ms:.2f}ms)"
        if event.error:
            msg += f" ERROR: {event.error}"
        self.logger.log(self.log_level, msg)


class MetricsCallback(AgentCallback):
    """指标收集回调"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_iterations": 0,
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "events": []
        }
        self._lock = threading.Lock()
    
    def on_event(self, event: AgentEvent) -> None:
        with self._lock:
            self.metrics["events"].append(event.to_dict())
            
            if event.event_type == AgentEventType.AGENT_START:
                self.metrics["total_runs"] += 1
            elif event.event_type == AgentEventType.AGENT_END:
                self.metrics["successful_runs"] += 1
                self.metrics["total_latency_ms"] += event.duration_ms
            elif event.event_type == AgentEventType.AGENT_ERROR:
                self.metrics["failed_runs"] += 1
            elif event.event_type == AgentEventType.ITERATION_END:
                self.metrics["total_iterations"] += 1
            elif event.event_type == AgentEventType.LLM_END:
                self.metrics["total_llm_calls"] += 1
                if "tokens" in event.data:
                    self.metrics["total_tokens"] += event.data["tokens"]
            elif event.event_type == AgentEventType.TOOL_END:
                self.metrics["total_tool_calls"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return self.metrics.copy()
    
    def reset(self) -> None:
        with self._lock:
            self.metrics = {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_iterations": 0,
                "total_llm_calls": 0,
                "total_tool_calls": 0,
                "total_latency_ms": 0.0,
                "total_tokens": 0,
                "events": []
            }


class StreamingCallback(AgentCallback):
    """流式回调 - 实时事件推送"""
    
    def __init__(self, on_token: Callable[[str], None] = None,
                 on_chunk: Callable[[Dict[str, Any]], None] = None):
        self.on_token = on_token
        self.on_chunk = on_chunk
        self._buffer = []
        self._lock = threading.Lock()
    
    def on_event(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event.to_dict())
        
        # 处理流式内容
        if event.event_type == AgentEventType.LLM_END:
            content = event.data.get('content', '')
            if self.on_token and content:
                for char in content:
                    self.on_token(char)
        
        if self.on_chunk:
            self.on_chunk(event.to_dict())
    
    def get_buffer(self) -> List[Dict[str, Any]]:
        """获取事件缓冲区"""
        with self._lock:
            return list(self._buffer)
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()


class WebhookCallback(AgentCallback):
    """Webhook 回调 - 发送事件到外部 URL"""
    
    def __init__(self, webhook_url: str, 
                 event_filter: List[AgentEventType] = None,
                 headers: Dict[str, str] = None,
                 timeout: float = 5.0,
                 async_send: bool = True):
        self.webhook_url = webhook_url
        self.event_filter = event_filter or list(AgentEventType)
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.async_send = async_send
        self._executor = ThreadPoolExecutor(max_workers=2) if async_send else None
        self._failed_events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def on_event(self, event: AgentEvent) -> None:
        if event.event_type not in self.event_filter:
            return
        
        if self.async_send and self._executor:
            self._executor.submit(self._send_event, event)
        else:
            self._send_event(event)
    
    def _send_event(self, event: AgentEvent) -> None:
        """发送事件到 Webhook"""
        try:
            import httpx
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.webhook_url,
                    json=event.to_dict(),
                    headers=self.headers
                )
                if response.status_code >= 400:
                    logger.warning(f"Webhook failed: {response.status_code}")
                    with self._lock:
                        self._failed_events.append(event.to_dict())
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            with self._lock:
                self._failed_events.append(event.to_dict())
    
    def get_failed_events(self) -> List[Dict[str, Any]]:
        """获取发送失败的事件"""
        with self._lock:
            return list(self._failed_events)
    
    def retry_failed(self) -> int:
        """重试发送失败的事件"""
        with self._lock:
            events = list(self._failed_events)
            self._failed_events.clear()
        
        success_count = 0
        for event_dict in events:
            try:
                import httpx
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.webhook_url,
                        json=event_dict,
                        headers=self.headers
                    )
                    if response.status_code < 400:
                        success_count += 1
                    else:
                        with self._lock:
                            self._failed_events.append(event_dict)
            except Exception:
                with self._lock:
                    self._failed_events.append(event_dict)
        
        return success_count
    
    def close(self) -> None:
        """关闭资源"""
        if self._executor:
            self._executor.shutdown(wait=False)


class BufferedCallback(AgentCallback):
    """缓冲回调 - 批量处理事件"""
    
    def __init__(self, batch_size: int = 10, 
                 flush_interval: float = 5.0,
                 on_flush: Callable[[List[AgentEvent]], None] = None):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.on_flush = on_flush
        self._buffer: List[AgentEvent] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._start_timer()
    
    def _start_timer(self) -> None:
        """启动定时刷新"""
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(self.flush_interval, self._auto_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def _auto_flush(self) -> None:
        """自动刷新"""
        self.flush()
        self._start_timer()
    
    def on_event(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.batch_size:
                self._do_flush()
    
    def _do_flush(self) -> None:
        """执行刷新（内部，需要持有锁）"""
        if not self._buffer:
            return
        
        events = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()
        
        if self.on_flush:
            try:
                self.on_flush(events)
            except Exception as e:
                logger.error(f"Buffered callback flush error: {e}")
    
    def flush(self) -> None:
        """手动刷新"""
        with self._lock:
            self._do_flush()
    
    def close(self) -> None:
        """关闭并刷新剩余事件"""
        if self._flush_timer:
            self._flush_timer.cancel()
        self.flush()


class ProfilingCallback(AgentCallback):
    """性能分析回调"""
    
    def __init__(self):
        self._profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._current_spans: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def on_event(self, event: AgentEvent) -> None:
        with self._lock:
            span_key = f"{event.agent_name}:{event.event_type.value}"
            
            # 记录开始事件
            if event.event_type in [AgentEventType.AGENT_START, 
                                    AgentEventType.ITERATION_START,
                                    AgentEventType.LLM_START,
                                    AgentEventType.TOOL_START]:
                self._current_spans[span_key] = {
                    "start_time": event.timestamp,
                    "event_type": event.event_type.value,
                    "data": event.data
                }
            
            # 记录结束事件
            elif event.event_type in [AgentEventType.AGENT_END,
                                      AgentEventType.AGENT_ERROR,
                                      AgentEventType.ITERATION_END,
                                      AgentEventType.LLM_END,
                                      AgentEventType.LLM_ERROR,
                                      AgentEventType.TOOL_END,
                                      AgentEventType.TOOL_ERROR]:
                start_key = span_key.replace("_end", "_start").replace("_error", "_start")
                start_span = self._current_spans.pop(start_key, None)
                
                profile_entry = {
                    "event_type": event.event_type.value,
                    "duration_ms": event.duration_ms,
                    "timestamp": event.timestamp.isoformat(),
                    "success": "error" not in event.event_type.value.lower(),
                    "data": event.data
                }
                
                if start_span:
                    actual_duration = (event.timestamp - start_span["start_time"]).total_seconds() * 1000
                    profile_entry["actual_duration_ms"] = actual_duration
                
                self._profiles[event.agent_name].append(profile_entry)
    
    def get_profile(self, agent_name: str = None) -> Dict[str, Any]:
        """获取性能分析报告"""
        with self._lock:
            if agent_name:
                entries = self._profiles.get(agent_name, [])
            else:
                entries = [e for entries in self._profiles.values() for e in entries]
            
            if not entries:
                return {"message": "No profiling data"}
            
            # 计算统计
            durations = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
            success_count = sum(1 for e in entries if e.get("success", False))
            
            return {
                "total_events": len(entries),
                "success_rate": success_count / len(entries) if entries else 0,
                "total_duration_ms": sum(durations),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "by_event_type": self._group_by_event_type(entries),
                "entries": entries[-50:]  # 最近50条
            }
    
    def _group_by_event_type(self, entries: List[Dict]) -> Dict[str, Dict]:
        """按事件类型分组统计"""
        groups = defaultdict(list)
        for e in entries:
            groups[e["event_type"]].append(e.get("duration_ms", 0))
        
        return {
            event_type: {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations) if durations else 0
            }
            for event_type, durations in groups.items()
        }
    
    def reset(self) -> None:
        """重置分析数据"""
        with self._lock:
            self._profiles.clear()
            self._current_spans.clear()


class CompositeCallback(AgentCallback):
    """组合回调 - 聚合多个回调"""
    
    def __init__(self, callbacks: List[AgentCallback] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: AgentCallback) -> 'CompositeCallback':
        self.callbacks.append(callback)
        return self
    
    def remove(self, callback: AgentCallback) -> bool:
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            return True
        return False
    
    def on_event(self, event: AgentEvent) -> None:
        for callback in self.callbacks:
            try:
                callback.on_event(event)
            except Exception as e:
                logger.error(f"Composite callback error: {e}")


class FilteredCallback(AgentCallback):
    """过滤回调 - 只处理特定事件"""
    
    def __init__(self, inner: AgentCallback, 
                 include_types: List[AgentEventType] = None,
                 exclude_types: List[AgentEventType] = None,
                 filter_func: Callable[[AgentEvent], bool] = None):
        self.inner = inner
        self.include_types = set(include_types) if include_types else None
        self.exclude_types = set(exclude_types) if exclude_types else set()
        self.filter_func = filter_func
    
    def on_event(self, event: AgentEvent) -> None:
        # 检查包含类型
        if self.include_types and event.event_type not in self.include_types:
            return
        
        # 检查排除类型
        if event.event_type in self.exclude_types:
            return
        
        # 自定义过滤
        if self.filter_func and not self.filter_func(event):
            return
        
        self.inner.on_event(event)


class CallbackManager:
    """回调管理器 - 增强版"""
    
    def __init__(self, callbacks: List[AgentCallback] = None):
        self.callbacks = callbacks or []
        self._lock = threading.Lock()
        self._async_executor = ThreadPoolExecutor(max_workers=4)
        self._event_history: deque = deque(maxlen=1000)
        self._enabled = True
    
    def add_callback(self, callback: AgentCallback) -> None:
        with self._lock:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: AgentCallback) -> bool:
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
                return True
            return False
    
    def clear_callbacks(self) -> None:
        """清除所有回调"""
        with self._lock:
            self.callbacks.clear()
    
    def enable(self) -> None:
        """启用回调"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用回调"""
        self._enabled = False
    
    def emit(self, event: AgentEvent) -> None:
        """发送事件到所有回调"""
        if not self._enabled:
            return
        
        # 记录到历史
        self._event_history.append(event.to_dict())
        
        for callback in self.callbacks:
            try:
                callback.on_event(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def emit_async(self, event: AgentEvent) -> Future:
        """异步发送事件"""
        return self._async_executor.submit(self.emit, event)
    
    async def aemit(self, event: AgentEvent) -> None:
        """异步发送事件（协程版）"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._async_executor, self.emit, event)
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件历史"""
        return list(self._event_history)[-limit:]
    
    def get_callbacks_by_type(self, callback_type: Type[AgentCallback]) -> List[AgentCallback]:
        """按类型获取回调"""
        return [cb for cb in self.callbacks if isinstance(cb, callback_type)]
    
    def close(self) -> None:
        """关闭资源"""
        self._async_executor.shutdown(wait=False)
        
        # 关闭支持关闭的回调
        for callback in self.callbacks:
            if hasattr(callback, 'close'):
                try:
                    callback.close()
                except Exception:
                    pass


# ==================== 执行追踪 ====================

@dataclass
class ExecutionTrace:
    """执行追踪记录"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    error: Optional[str] = None
    input: str = ""
    output: Optional[str] = None
    
    def add_iteration(self, iteration_data: Dict[str, Any]) -> None:
        self.iterations.append({
            **iteration_data,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def finish(self, output: str = None, error: str = None) -> None:
        self.end_time = datetime.utcnow()
        self.output = output
        self.error = error
        self.status = "error" if error else "completed"
        if self.start_time:
            self.total_latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
            "input": self.input,
            "output": self.output
        }


class ExecutionTracer:
    """执行追踪器"""
    
    def __init__(self, max_traces: int = 100):
        self.max_traces = max_traces
        self.traces: Dict[str, ExecutionTrace] = {}
        self._lock = threading.Lock()
    
    def start_trace(self, agent_name: str, input_text: str) -> ExecutionTrace:
        trace = ExecutionTrace(agent_name=agent_name, input=input_text)
        with self._lock:
            self.traces[trace.trace_id] = trace
            # 清理旧追踪
            if len(self.traces) > self.max_traces:
                oldest_keys = sorted(
                    self.traces.keys(),
                    key=lambda k: self.traces[k].start_time
                )[:len(self.traces) - self.max_traces]
                for key in oldest_keys:
                    del self.traces[key]
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        return self.traces.get(trace_id)
    
    def get_all_traces(self) -> List[ExecutionTrace]:
        with self._lock:
            return list(self.traces.values())


# ==================== 执行上下文 ====================

@dataclass
class ExecutionContext:
    """执行上下文"""
    trace: Optional[ExecutionTrace] = None
    callback_manager: Optional[CallbackManager] = None
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    
    def request_cancel(self) -> None:
        """请求取消执行"""
        self.cancel_requested = True
    
    def is_cancelled(self) -> bool:
        return self.cancel_requested


# ==================== 执行控制 ====================

class CircuitBreaker:
    """断路器 - 防止级联失败"""
    
    class State(Enum):
        CLOSED = "closed"      # 正常
        OPEN = "open"          # 断开（拒绝请求）
        HALF_OPEN = "half_open"  # 半开（尝试恢复）
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_requests: int = 3,
                 name: str = ""):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.name = name or f"circuit_breaker_{uuid.uuid4().hex[:8]}"
        
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_count = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> State:
        """获取当前状态"""
        with self._lock:
            return self._check_state()
    
    def _check_state(self) -> State:
        """检查并更新状态"""
        now = datetime.utcnow()
        
        if self._state == self.State.OPEN:
            if self._last_failure_time:
                elapsed = (now - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state = self.State.HALF_OPEN
                    self._half_open_count = 0
                    logger.info(f"CircuitBreaker '{self.name}' entering half-open state")
        
        return self._state
    
    def is_available(self) -> bool:
        """是否可用"""
        return self.state != self.State.OPEN
    
    def record_success(self) -> None:
        """记录成功"""
        with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._half_open_count += 1
                if self._half_open_count >= self.half_open_requests:
                    self._state = self.State.CLOSED
                    self._failure_count = 0
                    logger.info(f"CircuitBreaker '{self.name}' closed (recovered)")
            else:
                self._failure_count = 0
    
    def record_failure(self, error: Exception = None) -> None:
        """记录失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            if self._state == self.State.HALF_OPEN:
                self._state = self.State.OPEN
                logger.warning(f"CircuitBreaker '{self.name}' re-opened (failure in half-open)")
            elif self._failure_count >= self.failure_threshold:
                self._state = self.State.OPEN
                logger.warning(f"CircuitBreaker '{self.name}' opened (threshold reached)")
    
    def reset(self) -> None:
        """重置"""
        with self._lock:
            self._state = self.State.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "half_open_count": self._half_open_count
            }
    
    @contextmanager
    def protect(self):
        """保护上下文管理器"""
        if not self.is_available():
            raise RuntimeError(f"CircuitBreaker '{self.name}' is open")
        
        try:
            yield
            self.record_success()
        except Exception as e:
            self.record_failure(e)
            raise


class RateLimiter:
    """速率限制器 - 令牌桶算法"""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 burst_size: int = 10,
                 name: str = ""):
        self.rate = requests_per_minute / 60.0  # 每秒令牌数
        self.burst_size = burst_size
        self.name = name or f"rate_limiter_{uuid.uuid4().hex[:8]}"
        
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()
        self._waiting_count = 0
    
    def _refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst_size, self._tokens + elapsed * self.rate)
        self._last_update = now
    
    def acquire(self, timeout: float = 0) -> bool:
        """获取令牌
        
        Args:
            timeout: 等待超时（0表示不等待）
            
        Returns:
            是否成功获取
        """
        start = time.time()
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True
                
                if timeout <= 0:
                    return False
                
                # 计算等待时间
                wait_time = (1 - self._tokens) / self.rate
                self._waiting_count += 1
            
            # 等待
            elapsed = time.time() - start
            if elapsed >= timeout:
                with self._lock:
                    self._waiting_count -= 1
                return False
            
            time.sleep(min(wait_time, timeout - elapsed))
            
            with self._lock:
                self._waiting_count -= 1
    
    async def aacquire(self, timeout: float = 0) -> bool:
        """异步获取令牌"""
        start = time.time()
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True
                
                if timeout <= 0:
                    return False
                
                wait_time = (1 - self._tokens) / self.rate
            
            elapsed = time.time() - start
            if elapsed >= timeout:
                return False
            
            await asyncio.sleep(min(wait_time, timeout - elapsed))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "name": self.name,
                "tokens_available": self._tokens,
                "rate_per_second": self.rate,
                "burst_size": self.burst_size,
                "waiting_count": self._waiting_count
            }


class ExecutionController:
    """执行控制器 - 整合断路器和限流器"""
    
    def __init__(self,
                 enable_circuit_breaker: bool = True,
                 enable_rate_limiter: bool = True,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 requests_per_minute: int = 60,
                 burst_size: int = 10):
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        ) if enable_circuit_breaker else None
        
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size
        ) if enable_rate_limiter else None
        
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.circuit_breaker and not self.circuit_breaker.is_available():
            return False
        
        if self.rate_limiter and not self.rate_limiter.acquire():
            return False
        
        return True
    
    def record_execution(self, success: bool, error: Exception = None) -> None:
        """记录执行结果"""
        with self._lock:
            self._execution_count += 1
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
        
        if self.circuit_breaker:
            if success:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            stats = {
                "total_executions": self._execution_count,
                "successful_executions": self._success_count,
                "failed_executions": self._failure_count,
                "success_rate": self._success_count / self._execution_count if self._execution_count > 0 else 0
            }
        
        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_stats()
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        return stats
    
    @contextmanager
    def controlled_execution(self):
        """受控执行的上下文管理器"""
        if not self.can_execute():
            raise RuntimeError("Execution blocked by controller")
        
        success = True
        error = None
        try:
            yield
        except Exception as e:
            success = False
            error = e
            raise
        finally:
            self.record_execution(success, error)


# ==================== 内存管理 ====================

class AgentMemoryManager:
    """Agent 内存管理器"""
    
    def __init__(self, 
                 max_short_term: int = 100,
                 max_long_term: int = 1000,
                 compression_threshold: int = 50):
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.compression_threshold = compression_threshold
        
        self._short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_short_term))
        self._long_term: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._entities: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._summaries: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def store_short_term(self, thread_id: str, content: str, 
                        memory_type: str = "conversation") -> None:
        """存储短期记忆"""
        with self._lock:
            self._short_term[thread_id].append({
                "content": content,
                "type": memory_type,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # 检查是否需要压缩
            if len(self._short_term[thread_id]) >= self.compression_threshold:
                self._compress_short_term(thread_id)
    
    def _compress_short_term(self, thread_id: str) -> None:
        """压缩短期记忆到长期记忆"""
        memories = list(self._short_term[thread_id])
        
        if len(memories) < self.compression_threshold // 2:
            return
        
        # 移动到长期记忆
        older_memories = memories[:len(memories) // 2]
        self._short_term[thread_id] = deque(
            memories[len(memories) // 2:],
            maxlen=self.max_short_term
        )
        
        # 存储到长期记忆
        self._long_term[thread_id].extend(older_memories)
        
        # 限制长期记忆大小
        if len(self._long_term[thread_id]) > self.max_long_term:
            self._long_term[thread_id] = self._long_term[thread_id][-self.max_long_term:]
    
    def get_short_term(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取短期记忆"""
        with self._lock:
            return list(self._short_term[thread_id])[-limit:]
    
    def get_long_term(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取长期记忆"""
        with self._lock:
            return self._long_term[thread_id][-limit:]
    
    def store_entity(self, thread_id: str, entity_name: str, 
                    entity_data: Dict[str, Any]) -> None:
        """存储实体信息"""
        with self._lock:
            self._entities[thread_id][entity_name] = {
                **entity_data,
                "updated_at": datetime.utcnow().isoformat()
            }
    
    def get_entity(self, thread_id: str, entity_name: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        with self._lock:
            return self._entities[thread_id].get(entity_name)
    
    def get_all_entities(self, thread_id: str) -> Dict[str, Any]:
        """获取所有实体"""
        with self._lock:
            return dict(self._entities[thread_id])
    
    def set_summary(self, thread_id: str, summary: str) -> None:
        """设置对话摘要"""
        with self._lock:
            self._summaries[thread_id] = summary
    
    def get_summary(self, thread_id: str) -> Optional[str]:
        """获取对话摘要"""
        with self._lock:
            return self._summaries.get(thread_id)
    
    def clear(self, thread_id: str) -> None:
        """清除指定线程的所有记忆"""
        with self._lock:
            self._short_term.pop(thread_id, None)
            self._long_term.pop(thread_id, None)
            self._entities.pop(thread_id, None)
            self._summaries.pop(thread_id, None)
    
    def get_context(self, thread_id: str) -> Dict[str, Any]:
        """获取完整上下文"""
        with self._lock:
            return {
                "short_term": list(self._short_term[thread_id]),
                "long_term": self._long_term[thread_id][-20:],
                "entities": dict(self._entities[thread_id]),
                "summary": self._summaries.get(thread_id)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """导出所有数据"""
        with self._lock:
            return {
                "short_term": {k: list(v) for k, v in self._short_term.items()},
                "long_term": dict(self._long_term),
                "entities": dict(self._entities),
                "summaries": dict(self._summaries)
            }


@dataclass
class AgentConfig:
    """Agent 配置 - 生产级增强版"""
    # 基本配置
    name: str = "agent"
    description: str = ""
    version: str = "1.0.0"
    
    # 执行控制
    max_iterations: int = 10
    timeout: float = 300.0
    iteration_timeout: float = 60.0
    
    # LLM 配置
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    provider: str = "openai"
    system_prompt: Optional[str] = None
    fallback_models: List[str] = field(default_factory=list)
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # 功能开关
    enable_checkpointing: bool = True
    enable_human_in_loop: bool = False
    enable_streaming: bool = False
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_memory: bool = True
    enable_circuit_breaker: bool = False
    enable_rate_limiter: bool = False
    verbose: bool = False
    
    # 回调配置
    callbacks: List[AgentCallback] = field(default_factory=list)
    
    # 工具配置
    tool_timeout: float = 30.0
    parallel_tool_execution: bool = True
    tool_selection_strategy: str = "auto"  # auto, all, specific
    max_tool_calls_per_iteration: int = 5
    
    # 断路器配置
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 30.0
    
    # 限流配置
    rate_limit_rpm: int = 60  # requests per minute
    rate_limit_burst: int = 10
    
    # 记忆配置
    memory_max_short_term: int = 100
    memory_max_long_term: int = 1000
    memory_compression_threshold: int = 50
    
    # 检查点配置
    checkpoint_interval: int = 1  # 每 N 步保存
    checkpoint_compression: bool = True
    checkpoint_max_history: int = 50
    
    # 输出配置
    return_intermediate_steps: bool = False
    return_full_trace: bool = False
    output_format: str = "text"  # text, json, structured
    
    # 人机协作配置
    human_approval_timeout: float = 300.0
    human_feedback_required: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "model": self.model,
            "temperature": self.temperature,
            "provider": self.provider,
            "enable_checkpointing": self.enable_checkpointing,
            "enable_human_in_loop": self.enable_human_in_loop,
            "enable_streaming": self.enable_streaming,
            "enable_memory": self.enable_memory,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "enable_rate_limiter": self.enable_rate_limiter,
            "tool_selection_strategy": self.tool_selection_strategy,
            "checkpoint_interval": self.checkpoint_interval,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def merge(self, override: Dict[str, Any]) -> 'AgentConfig':
        """合并配置"""
        config_dict = self.to_dict()
        config_dict.update(override)
        return AgentConfig.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        return errors
    
    @classmethod
    def for_agent_type(cls, agent_type: str, **kwargs) -> 'AgentConfig':
        """为特定 Agent 类型创建预设配置"""
        presets = {
            "react": {
                "max_iterations": 10,
                "temperature": 0.7,
                "tool_selection_strategy": "auto"
            },
            "plan_execute": {
                "max_iterations": 20,
                "temperature": 0.3,
                "enable_memory": True
            },
            "reflexion": {
                "max_iterations": 15,
                "temperature": 0.5
            },
            "conversational": {
                "max_iterations": 5,
                "temperature": 0.8,
                "enable_memory": True,
                "memory_compression_threshold": 30
            },
            "tool_calling": {
                "max_iterations": 3,
                "temperature": 0.1,
                "parallel_tool_execution": True
            },
            "chain_of_thought": {
                "max_iterations": 5,
                "temperature": 0.3
            }
        }
        
        preset = presets.get(agent_type.lower(), {})
        preset.update(kwargs)
        return cls(**preset)


class BaseAgent(ABC):
    """Agent 基类 - 生产级增强实现
    
    定义 Agent 的基本接口和公共功能，包括：
    - 执行控制：超时、重试、取消、断路器、限流
    - 回调系统：事件通知、日志记录、Webhook、流式
    - 指标追踪：延迟、Token、成功率、性能分析
    - 状态管理：检查点、恢复、分支、回滚
    - 内存管理：短期/长期记忆、实体提取、压缩
    
    所有具体 Agent 类型必须继承此类并实现 _build_graph 方法。
    """
    
    # 类级别注册表
    _agent_registry: Dict[str, Type['BaseAgent']] = {}
    
    def __init__(self, 
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 memory_manager: AgentMemoryManager = None,
                 execution_controller: ExecutionController = None):
        self.config = config or AgentConfig()
        self.tools = tools or []
        self.llm_client = llm_client
        self.checkpointer = checkpointer or MemoryCheckpointer()
        
        # 验证配置
        config_errors = self.config.validate()
        if config_errors:
            logger.warning(f"Agent config validation warnings: {config_errors}")
        
        self._graph: Optional[CompiledGraph] = None
        self._tool_registry = ToolRegistry()
        self._tool_executor: Optional[ToolExecutor] = None
        
        # 注册工具
        for tool_item in self.tools:
            self._tool_registry.register(tool_item)
        
        # 创建工具执行器
        self._tool_executor = ToolExecutor(self._tool_registry)
        
        # 回调管理
        all_callbacks = list(self.config.callbacks) + (callbacks or [])
        if self.config.enable_metrics:
            all_callbacks.append(MetricsCallback())
        if self.config.verbose:
            all_callbacks.append(LoggingCallback(logging.DEBUG))
        self._callback_manager = CallbackManager(all_callbacks)
        
        # 执行追踪
        self._tracer = ExecutionTracer() if self.config.enable_tracing else None
        self._current_trace: Optional[ExecutionTrace] = None
        
        # 指标收集
        self._metrics_callback = next(
            (cb for cb in self._callback_manager.callbacks if isinstance(cb, MetricsCallback)),
            None
        )
        
        # 性能分析回调
        self._profiling_callback: Optional[ProfilingCallback] = None
        if self.config.enable_metrics:
            self._profiling_callback = ProfilingCallback()
            self._callback_manager.add_callback(self._profiling_callback)
        
        # 内存管理
        self._memory_manager = memory_manager
        if self.config.enable_memory and not self._memory_manager:
            self._memory_manager = AgentMemoryManager(
                max_short_term=self.config.memory_max_short_term,
                max_long_term=self.config.memory_max_long_term,
                compression_threshold=self.config.memory_compression_threshold
            )
        
        # 执行控制
        self._execution_controller = execution_controller
        if (self.config.enable_circuit_breaker or self.config.enable_rate_limiter) and not self._execution_controller:
            self._execution_controller = ExecutionController(
                enable_circuit_breaker=self.config.enable_circuit_breaker,
                enable_rate_limiter=self.config.enable_rate_limiter,
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery,
                requests_per_minute=self.config.rate_limit_rpm,
                burst_size=self.config.rate_limit_burst
            )
        
        # 状态管理
        self._state_manager: Optional[StateManager] = None
        self._state_validator: Optional[StateValidator] = None
        self._state_event_emitter: Optional[StateEventEmitter] = None
        
        # 执行状态
        self._is_running = False
        self._is_paused = False
        self._cancel_requested = False
        self._current_state: Optional[AgentState] = None
        self._execution_count = 0
        self._lock = threading.Lock()
        
        # 检查点追踪
        self._checkpoint_history: List[str] = []
        self._checkpoint_tags: Dict[str, str] = {}
        
        # 人机协作
        self._human_input_queue: queue.Queue = queue.Queue()
        self._pending_approval: Optional[Dict[str, Any]] = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        
        # ==================== 策略管理 ====================
        # 初始化策略管理器
        self._strategy_manager = get_strategy_manager()
        
        # 注册自定义策略（如果有）
        self._init_strategies()
    
    @classmethod
    def register_type(cls, name: str, agent_class: Type['BaseAgent']) -> None:
        """注册 Agent 类型"""
        cls._agent_registry[name.lower()] = agent_class
    
    @classmethod
    def get_registered_type(cls, name: str) -> Optional[Type['BaseAgent']]:
        """获取注册的 Agent 类型"""
        return cls._agent_registry.get(name.lower())
    
    @classmethod
    def list_registered_types(cls) -> List[str]:
        """列出所有注册的 Agent 类型"""
        return list(cls._agent_registry.keys())
    
    @property
    def name(self) -> str:
        """获取 Agent 名称"""
        return self.config.name
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._is_running
    
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """构建 Agent 图"""
        pass
    
    # ==================== 策略相关方法 ====================
    
    def _init_strategies(self) -> None:
        """初始化策略
        
        子类可以重写此方法来注册自定义策略。
        """
        # 状态策略
        if not self._strategy_manager.get(StrategyType.STATE):
            state_strategy = DefaultStateStrategy(
                name=f"{self.name}_state_strategy",
                config={
                    "max_buffer_size": self.config.memory_max_short_term
                },
                memory=self._memory_manager.memory if self._memory_manager else None
            )
            self._strategy_manager.register(StrategyType.STATE, state_strategy)
        
        # 节点策略
        if not self._strategy_manager.get(StrategyType.NODE):
            node_strategy = DefaultNodeStrategy(
                name=f"{self.name}_node_strategy",
                config={
                    "default_model": self.config.model
                }
            )
            self._strategy_manager.register(StrategyType.NODE, node_strategy)
        
        # 边策略
        if not self._strategy_manager.get(StrategyType.EDGE):
            edge_strategy = DefaultEdgeStrategy(
                name=f"{self.name}_edge_strategy"
            )
            self._strategy_manager.register(StrategyType.EDGE, edge_strategy)
        
        # 图策略
        if not self._strategy_manager.get(StrategyType.GRAPH):
            graph_strategy = DefaultGraphStrategy(
                name=f"{self.name}_graph_strategy",
                config={
                    "graph_name": f"{self.name}_graph",
                    "max_iterations": self.config.max_iterations
                }
            )
            self._strategy_manager.register(StrategyType.GRAPH, graph_strategy)
        
        # 工具策略
        if not self._strategy_manager.get(StrategyType.TOOL):
            tool_strategy = DefaultToolStrategy(
                name=f"{self.name}_tool_strategy",
                config={
                    "max_retries": self.config.max_tool_retries,
                    "timeout": self.config.tool_timeout
                },
                tool_registry=self._tool_registry
            )
            self._strategy_manager.register(StrategyType.TOOL, tool_strategy)
        
        # 检查点策略
        if not self._strategy_manager.get(StrategyType.CHECKPOINT):
            checkpoint_strategy = DefaultCheckpointStrategy(
                name=f"{self.name}_checkpoint_strategy",
                config={
                    "compression": self.config.checkpoint_compression
                },
                checkpointer=self.checkpointer
            )
            self._strategy_manager.register(StrategyType.CHECKPOINT, checkpoint_strategy)
        
        # 记忆策略
        if not self._strategy_manager.get(StrategyType.MEMORY):
            memory_strategy = DefaultMemoryStrategy(
                name=f"{self.name}_memory_strategy",
                config={
                    "buffer_size": self.config.memory_max_short_term
                }
            )
            self._strategy_manager.register(StrategyType.MEMORY, memory_strategy)
        
        # 路由策略
        if not self._strategy_manager.get(StrategyType.ROUTING):
            routing_strategy = DefaultRoutingStrategy(
                name=f"{self.name}_routing_strategy"
            )
            self._strategy_manager.register(StrategyType.ROUTING, routing_strategy)
    
    def set_strategy(self, strategy_type: StrategyType, strategy: BaseStrategy) -> None:
        """设置策略
        
        Args:
            strategy_type: 策略类型
            strategy: 策略实例
        """
        self._strategy_manager.register(strategy_type, strategy)
        self.logger.info(f"Registered strategy: {strategy_type.value} -> {strategy.name}")
    
    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """获取策略
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            策略实例
        """
        return self._strategy_manager.get(strategy_type)
    
    def execute_strategy(self, strategy_type: StrategyType, 
                        context: Dict[str, Any]) -> Any:
        """执行策略
        
        Args:
            strategy_type: 策略类型
            context: 执行上下文
            
        Returns:
            策略执行结果
        """
        return self._strategy_manager.execute(strategy_type, context)
    
    def execute_strategy_chain(self, 
                              chain: List[Tuple[StrategyType, Dict[str, Any]]]) -> List[Any]:
        """执行策略链
        
        Args:
            chain: [(策略类型, 上下文), ...]
            
        Returns:
            执行结果列表
        """
        return self._strategy_manager.execute_chain(chain)
    
    def get_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略的指标
        
        Returns:
            策略指标字典
        """
        return self._strategy_manager.get_all_metrics()
    
    # ==================== 基于策略的核心操作 ====================
    
    def create_state_via_strategy(self, input_data: Any, **kwargs) -> AgentState:
        """通过状态策略创建状态
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            AgentState
        """
        context = {
            "input_data": input_data,
            **kwargs
        }
        return self.execute_strategy(StrategyType.STATE, context)
    
    def select_node_via_strategy(self, state: AgentState, 
                                available_nodes: List[str]) -> str:
        """通过节点策略选择下一个节点
        
        Args:
            state: 当前状态
            available_nodes: 可用节点列表
            
        Returns:
            选中的节点名称
        """
        context = {
            "state": state,
            "available_nodes": available_nodes
        }
        return self.execute_strategy(StrategyType.NODE, context)
    
    def create_node_via_strategy(self, node_type: str, 
                                config: Dict[str, Any] = None) -> BaseNode:
        """通过节点策略创建节点
        
        Args:
            node_type: 节点类型
            config: 节点配置
            
        Returns:
            节点实例
        """
        context = {
            "node_type": node_type,
            "node_config": config or {}
        }
        return self.execute_strategy(StrategyType.NODE, context)
    
    def route_via_strategy(self, state: AgentState, 
                          edges: List[Edge] = None,
                          router_type: str = "priority") -> str:
        """通过边策略进行路由
        
        Args:
            state: 当前状态
            edges: 可用边列表
            router_type: 路由器类型
            
        Returns:
            目标节点名称
        """
        context = {
            "state": state,
            "edges": edges or [],
            "router_type": router_type
        }
        return self.execute_strategy(StrategyType.EDGE, context)
    
    def execute_tools_via_strategy(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """通过工具策略执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            工具结果列表
        """
        context = {
            "tool_calls": tool_calls
        }
        return self.execute_strategy(StrategyType.TOOL, context)
    
    def save_checkpoint_via_strategy(self, state: AgentState, 
                                    checkpoint_id: str = None) -> str:
        """通过检查点策略保存状态
        
        Args:
            state: 要保存的状态
            checkpoint_id: 检查点ID
            
        Returns:
            检查点ID
        """
        context = {
            "action": "save",
            "state": state,
            "checkpoint_id": checkpoint_id
        }
        return self.execute_strategy(StrategyType.CHECKPOINT, context)
    
    def load_checkpoint_via_strategy(self, checkpoint_id: str) -> Optional[AgentState]:
        """通过检查点策略加载状态
        
        Args:
            checkpoint_id: 检查点ID
            
        Returns:
            状态或 None
        """
        context = {
            "action": "load",
            "checkpoint_id": checkpoint_id
        }
        return self.execute_strategy(StrategyType.CHECKPOINT, context)
    
    def store_memory_via_strategy(self, key: str, value: Any, 
                                 memory_type: str = "short_term") -> None:
        """通过记忆策略存储记忆
        
        Args:
            key: 记忆键
            value: 记忆值
            memory_type: 记忆类型
        """
        context = {
            "action": "store",
            "key": key,
            "value": value,
            "memory_type": memory_type
        }
        self.execute_strategy(StrategyType.MEMORY, context)
    
    def retrieve_memory_via_strategy(self, key: str, 
                                    memory_type: str = None) -> Optional[Any]:
        """通过记忆策略检索记忆
        
        Args:
            key: 记忆键
            memory_type: 记忆类型
            
        Returns:
            记忆值或 None
        """
        context = {
            "action": "retrieve",
            "key": key,
            "memory_type": memory_type
        }
        return self.execute_strategy(StrategyType.MEMORY, context)
    
    def search_memory_via_strategy(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """通过记忆策略搜索记忆
        
        Args:
            query: 搜索查询
            limit: 返回数量限制
            
        Returns:
            记忆条目列表
        """
        context = {
            "action": "search",
            "query": query,
            "limit": limit
        }
        return self.execute_strategy(StrategyType.MEMORY, context)
    
    def route_to_destination_via_strategy(self, state: AgentState, 
                                         destinations: List[str]) -> str:
        """通过路由策略选择目标
        
        Args:
            state: 当前状态
            destinations: 可用目标列表
            
        Returns:
            选中的目标
        """
        context = {
            "state": state,
            "destinations": destinations
        }
        return self.execute_strategy(StrategyType.ROUTING, context)
    
    def compile(self) -> CompiledGraph:
        """编译 Agent"""
        graph = self._build_graph()
        self._graph = graph.compile(self.checkpointer)
        return self._graph
    
    def _prepare_state(self, input_data: Union[str, Dict[str, Any], AgentState]) -> AgentState:
        """准备初始状态"""
        if isinstance(input_data, AgentState):
            state = input_data.copy()
        elif isinstance(input_data, dict):
            state = AgentState(
                input=input_data.get('input', str(input_data)),
                metadata=input_data.get('metadata', {}),
                data=input_data.get('data', {})
            )
            if 'messages' in input_data:
                for msg in input_data['messages']:
                    if isinstance(msg, AgentMessage):
                        state.add_message(msg)
                    elif isinstance(msg, dict):
                        state.add_message(AgentMessage.from_dict(msg))
        else:
            state = AgentState(input=str(input_data))
            state.add_message(AgentMessage.human(str(input_data)))
        
        state.agent_id = self.config.name
        state.max_iterations = self.config.max_iterations
        return state
    
    def _emit_event(self, event_type: AgentEventType, 
                    data: Dict[str, Any] = None,
                    state: AgentState = None,
                    error: Exception = None,
                    duration_ms: float = 0.0) -> None:
        """发送事件"""
        event = AgentEvent(
            event_type=event_type,
            agent_name=self.config.name,
            data=data or {},
            state=state,
            error=error,
            duration_ms=duration_ms
        )
        self._callback_manager.emit(event)
    
    def invoke(self, 
              input_data: Union[str, Dict[str, Any], AgentState],
              config: Dict[str, Any] = None,
              timeout: float = None,
              thread_id: str = None,
              checkpoint_tags: List[str] = None) -> AgentState:
        """同步调用 Agent
        
        Args:
            input_data: 输入数据（字符串、字典或 AgentState）
            config: 运行配置
            timeout: 超时时间（秒）
            thread_id: 线程 ID（用于状态持久化）
            checkpoint_tags: 检查点标签
            
        Returns:
            最终的 AgentState
        """
        if self._graph is None:
            self.compile()
        
        # 检查执行控制
        if self._execution_controller and not self._execution_controller.can_execute():
            raise RuntimeError(f"Agent {self.name} execution blocked by controller")
        
        with self._lock:
            if self._is_running:
                raise RuntimeError(f"Agent {self.name} is already running")
            self._is_running = True
            self._is_paused = False
            self._cancel_requested = False
            self._execution_count += 1
        
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        # 使用状态策略准备状态
        try:
            state = self.create_state_via_strategy(input_data, thread_id=thread_id)
        except Exception:
            # 回退到默认方法
            state = self._prepare_state(input_data)
        
        # 设置线程 ID
        if thread_id:
            state.thread_id = thread_id
        
        self._current_state = state
        execution_success = False
        
        # 开始追踪
        if self._tracer:
            self._current_trace = self._tracer.start_trace(self.config.name, state.input)
        
        self._emit_event(AgentEventType.AGENT_START, {
            "input": state.input,
            "thread_id": state.thread_id,
            "execution_count": self._execution_count
        }, state)
        
        try:
            # 带超时的执行
            result = self._execute_with_timeout(state, config, timeout)
            execution_success = True
            
            duration_ms = (time.time() - start_time) * 1000
            
            # 使用记忆策略存储结果
            if self._memory_manager and result.final_answer:
                try:
                    self.store_memory_via_strategy(
                        f"response_{result.thread_id}",
                        result.final_answer,
                        "short_term"
                    )
                except Exception:
                    # 回退到直接存储
                    self._memory_manager.store_short_term(
                        result.thread_id, 
                        result.final_answer,
                        "response"
                    )
            
            # 使用检查点策略保存
            if self.config.enable_checkpointing:
                try:
                    cp_id = self.save_checkpoint_via_strategy(result)
                    self._checkpoint_history.append(cp_id)
                    
                    # 添加标签
                    if checkpoint_tags:
                        checkpoint_strategy = self.get_strategy(StrategyType.CHECKPOINT)
                        if checkpoint_strategy and hasattr(checkpoint_strategy, 'tag'):
                            for tag in checkpoint_tags:
                                checkpoint_strategy.tag(cp_id, tag)
                except Exception as cp_error:
                    self.logger.warning(f"Strategy checkpoint failed, using fallback: {cp_error}")
                    try:
                        cp_id = self.save_checkpoint(result, checkpoint_tags)
                        self._checkpoint_history.append(cp_id)
                    except Exception as cp_error2:
                        self.logger.warning(f"Failed to save checkpoint: {cp_error2}")
            
            self._emit_event(
                AgentEventType.AGENT_END, 
                {
                    "output": result.final_answer,
                    "iterations": result.iteration,
                    "duration_ms": duration_ms
                }, 
                result,
                duration_ms=duration_ms
            )
            
            # 完成追踪
            if self._current_trace:
                self._current_trace.finish(output=result.final_answer)
                self._current_trace.total_tokens = result.total_tokens
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Agent execution failed: {e}\n{traceback.format_exc()}")
            self._emit_event(AgentEventType.AGENT_ERROR, {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, error=e, duration_ms=duration_ms)
            
            if self._current_trace:
                self._current_trace.finish(error=str(e))
            
            state.set_error(str(e))
            return state
            
        finally:
            # 记录到执行控制器
            if self._execution_controller:
                self._execution_controller.record_execution(
                    execution_success,
                    None if execution_success else Exception("Execution failed")
                )
            
            with self._lock:
                self._is_running = False
                self._current_trace = None
                self._current_state = None
    
    def invoke_with_memory(self,
                          input_data: Union[str, Dict[str, Any]],
                          thread_id: str,
                          include_history: bool = True,
                          max_history: int = 10,
                          **kwargs) -> AgentState:
        """带内存上下文的调用
        
        Args:
            input_data: 输入数据
            thread_id: 线程 ID
            include_history: 是否包含历史
            max_history: 最大历史条数
            **kwargs: 其他参数
            
        Returns:
            AgentState
        """
        state = self._prepare_state(input_data)
        state.thread_id = thread_id
        
        # 添加内存上下文
        if include_history and self._memory_manager:
            context = self._memory_manager.get_context(thread_id)
            
            # 添加摘要
            if context.get("summary"):
                state.add_message(AgentMessage.system(f"对话摘要: {context['summary']}"))
            
            # 添加短期记忆
            for memory in context.get("short_term", [])[-max_history:]:
                content = memory.get("content", "")
                memory_type = memory.get("type", "conversation")
                if memory_type == "response":
                    state.add_message(AgentMessage.ai(content))
                else:
                    state.add_message(AgentMessage.human(content))
        
        # 存储输入
        if self._memory_manager:
            self._memory_manager.store_short_term(thread_id, str(input_data), "input")
        
        return self.invoke(state, thread_id=thread_id, **kwargs)
    
    def invoke_batch(self, 
                    inputs: List[Union[str, Dict[str, Any], AgentState]],
                    config: Dict[str, Any] = None,
                    max_workers: int = 4,
                    fail_fast: bool = False) -> List[AgentState]:
        """批量调用 Agent
        
        Args:
            inputs: 输入列表
            config: 运行配置
            max_workers: 最大并行数
            fail_fast: 遇错立即停止
            
        Returns:
            结果列表
        """
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.invoke, inp, config): i 
                for i, inp in enumerate(inputs)
            }
            
            for future in futures:
                try:
                    result = future.result()
                    results.append((futures[future], result))
                except Exception as e:
                    if fail_fast:
                        raise
                    errors.append((futures[future], e))
                    results.append((futures[future], None))
        
        # 按原顺序排序
        results.sort(key=lambda x: x[0])
        
        if errors:
            self.logger.warning(f"Batch execution had {len(errors)} failures")
        
        return [r[1] for r in results]
    
    def _execute_with_timeout(self, state: AgentState, config: Dict[str, Any], 
                              timeout: float) -> AgentState:
        """带超时的执行"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_loop, state, config)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                self._cancel_requested = True
                state.set_error(f"Execution timeout after {timeout}s")
                return state
    
    def _execute_loop(self, state: AgentState, config: Dict[str, Any]) -> AgentState:
        """执行主循环"""
        while state.should_continue() and not self._cancel_requested:
            iteration_start = time.time()
            
            self._emit_event(
                AgentEventType.ITERATION_START,
                {"iteration": state.iteration},
                state
            )
            
            try:
                state = self._graph.invoke(state, config)
            except Exception as e:
                self.logger.error(f"Iteration {state.iteration} failed: {e}")
                # 重试逻辑
                if state.iteration < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (self.config.retry_backoff ** state.iteration))
                    continue
                else:
                    raise
            
            iteration_duration = (time.time() - iteration_start) * 1000
            self._emit_event(
                AgentEventType.ITERATION_END,
                {"iteration": state.iteration, "duration_ms": iteration_duration},
                state,
                duration_ms=iteration_duration
            )
            
            # 追踪迭代
            if self._current_trace:
                self._current_trace.add_iteration({
                    "iteration": state.iteration,
                    "duration_ms": iteration_duration,
                    "has_tool_calls": bool(state.pending_tool_calls),
                    "message_count": len(state.messages)
                })
            
            if not state.increment_iteration():
                self.logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
                break
        
        return state
    
    async def ainvoke(self,
                     input_data: Union[str, Dict[str, Any], AgentState],
                     config: Dict[str, Any] = None) -> AgentState:
        """异步调用 Agent"""
        if self._graph is None:
            self.compile()
        
        state = self._prepare_state(input_data)
        
        if self._tracer:
            self._current_trace = self._tracer.start_trace(self.config.name, state.input)
        
        await self._callback_manager.aemit(AgentEvent(
            event_type=AgentEventType.AGENT_START,
            agent_name=self.config.name,
            data={"input": state.input}
        ))
        
        try:
            result = await self._graph.ainvoke(state, config)
            
            if self._current_trace:
                self._current_trace.finish(output=result.final_answer)
            
            return result
        except Exception as e:
            self.logger.error(f"Async execution failed: {e}")
            if self._current_trace:
                self._current_trace.finish(error=str(e))
            state.set_error(str(e))
            return state
    
    def stream(self,
              input_data: Union[str, Dict[str, Any], AgentState],
              config: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        """流式调用 Agent
        
        Yields:
            包含状态更新的字典
        """
        if self._graph is None:
            self.compile()
        
        state = self._prepare_state(input_data)
        
        self._emit_event(AgentEventType.AGENT_START, {"input": state.input})
        
        try:
            for chunk in self._graph.stream(state, config):
                yield chunk
                
                # 检查是否需要取消
                if self._cancel_requested:
                    break
        except Exception as e:
            self.logger.error(f"Stream execution failed: {e}")
            yield {"error": str(e)}
    
    async def astream(self,
                     input_data: Union[str, Dict[str, Any], AgentState],
                     config: Dict[str, Any] = None) -> AsyncIterator[Dict[str, Any]]:
        """异步流式调用"""
        if self._graph is None:
            self.compile()
        
        state = self._prepare_state(input_data)
        
        try:
            async for chunk in self._graph.astream(state, config):
                yield chunk
        except Exception as e:
            self.logger.error(f"Async stream failed: {e}")
            yield {"error": str(e)}
    
    def cancel(self) -> bool:
        """取消正在执行的任务"""
        if self._is_running:
            self._cancel_requested = True
            self.logger.info(f"Cancel requested for agent {self.name}")
            return True
        return False
    
    def add_tool(self, tool: Tool) -> None:
        """添加工具"""
        self.tools.append(tool)
        self._tool_registry.register(tool)
        # 需要重新编译
        self._graph = None
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(i)
                self._tool_registry.unregister(tool_name)
                self._graph = None
                return True
        return False
    
    def get_tool_schemas(self, format: str = "openai") -> List[Dict[str, Any]]:
        """获取工具 schema"""
        return self._tool_registry.get_tool_schemas(format)
    
    def add_callback(self, callback: AgentCallback) -> None:
        """添加回调"""
        self._callback_manager.add_callback(callback)
    
    def remove_callback(self, callback: AgentCallback) -> None:
        """移除回调"""
        self._callback_manager.remove_callback(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        if self._metrics_callback:
            return self._metrics_callback.get_metrics()
        return {}
    
    def get_traces(self) -> List[ExecutionTrace]:
        """获取执行追踪"""
        if self._tracer:
            return self._tracer.get_all_traces()
        return []
    
    def get_current_trace(self) -> Optional[ExecutionTrace]:
        """获取当前追踪"""
        return self._current_trace
    
    def reset_metrics(self) -> None:
        """重置指标"""
        if self._metrics_callback:
            self._metrics_callback.reset()
    
    def save_checkpoint(self, state: AgentState, tags: List[str] = None,
                       branch: str = "main") -> str:
        """保存检查点
        
        Args:
            state: 状态
            tags: 标签列表
            branch: 分支名称
            
        Returns:
            检查点 ID
        """
        thread_id = state.thread_id
        
        # 使用增强版检查点保存
        if hasattr(self.checkpointer, 'save') and callable(self.checkpointer.save):
            # 尝试使用增强版参数
            try:
                checkpoint_id = self.checkpointer.save(
                    state, 
                    tags=tags,
                    branch=branch
                )
            except TypeError:
                # 回退到基本版本
                checkpoint_id = self.checkpointer.save(state)
        else:
            checkpoint_id = str(uuid.uuid4())
        
        # 记录标签
        if tags:
            for tag in tags:
                self._checkpoint_tags[tag] = checkpoint_id
        
        self._emit_event(
            AgentEventType.CHECKPOINT_SAVED,
            {
                "checkpoint_id": checkpoint_id, 
                "thread_id": thread_id,
                "tags": tags,
                "branch": branch
            }
        )
        return checkpoint_id
    
    def restore_checkpoint(self, thread_id: str = None, 
                          checkpoint_id: str = None,
                          tag: str = None) -> Optional[AgentState]:
        """恢复检查点
        
        Args:
            thread_id: 线程 ID
            checkpoint_id: 检查点 ID
            tag: 标签名称
            
        Returns:
            恢复的状态
        """
        # 通过标签查找
        if tag and tag in self._checkpoint_tags:
            checkpoint_id = self._checkpoint_tags[tag]
        
        # 加载检查点
        state = None
        if hasattr(self.checkpointer, 'load'):
            state = self.checkpointer.load(checkpoint_id)
        elif hasattr(self.checkpointer, 'get'):
            state = self.checkpointer.get(thread_id, checkpoint_id)
        
        if state:
            self._emit_event(
                AgentEventType.CHECKPOINT_RESTORED,
                {
                    "checkpoint_id": checkpoint_id, 
                    "thread_id": thread_id,
                    "tag": tag
                }
            )
        return state
    
    def rollback(self, thread_id: str, steps: int = 1) -> Optional[AgentState]:
        """回滚到之前的状态
        
        Args:
            thread_id: 线程 ID
            steps: 回滚步数
            
        Returns:
            回滚后的状态
        """
        if hasattr(self.checkpointer, 'rollback'):
            return self.checkpointer.rollback(thread_id, steps=steps)
        
        # 手动回滚
        history = self.get_state_history(thread_id)
        if len(history) > steps:
            return history[-steps - 1]
        return None
    
    def create_branch(self, checkpoint_id: str, branch_name: str) -> Optional[str]:
        """从检查点创建分支
        
        Args:
            checkpoint_id: 检查点 ID
            branch_name: 分支名称
            
        Returns:
            新分支的检查点 ID
        """
        if hasattr(self.checkpointer, 'create_branch'):
            return self.checkpointer.create_branch(checkpoint_id, branch_name)
        return None
    
    def get_checkpoint_diff(self, checkpoint_id1: str, 
                           checkpoint_id2: str) -> Optional[Dict[str, Any]]:
        """比较两个检查点的差异
        
        Args:
            checkpoint_id1: 第一个检查点 ID
            checkpoint_id2: 第二个检查点 ID
            
        Returns:
            差异信息
        """
        if hasattr(self.checkpointer, 'diff'):
            diff = self.checkpointer.diff(checkpoint_id1, checkpoint_id2)
            return diff.to_dict() if diff else None
        return None
    
    def get_state_history(self, thread_id: str, 
                         branch: str = None,
                         limit: int = 10) -> List[AgentState]:
        """获取状态历史
        
        Args:
            thread_id: 线程 ID
            branch: 分支名称
            limit: 限制数量
            
        Returns:
            状态列表
        """
        if hasattr(self.checkpointer, 'list_checkpoints'):
            checkpoints = self.checkpointer.list_checkpoints(
                thread_id=thread_id,
                branch=branch,
                limit=limit
            )
            # 从检查点恢复状态
            states = []
            for cp in checkpoints:
                if hasattr(cp, 'state'):
                    states.append(cp.state)
            return states
        elif hasattr(self.checkpointer, 'list'):
            return self.checkpointer.list(thread_id)[:limit]
        return []
    
    def pause(self) -> bool:
        """暂停执行"""
        with self._lock:
            if self._is_running and not self._is_paused:
                self._is_paused = True
                self.logger.info(f"Agent {self.name} paused")
                return True
            return False
    
    def resume(self) -> bool:
        """恢复执行"""
        with self._lock:
            if self._is_paused:
                self._is_paused = False
                self.logger.info(f"Agent {self.name} resumed")
                return True
            return False
    
    def provide_human_input(self, feedback: str, approval: bool = True) -> None:
        """提供人工输入
        
        Args:
            feedback: 反馈内容
            approval: 是否批准
        """
        self._human_input_queue.put({
            "feedback": feedback,
            "approval": approval,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self._emit_event(
            AgentEventType.HUMAN_INPUT_RECEIVED,
            {"feedback": feedback[:100], "approval": approval}
        )
    
    def request_human_approval(self, request: Dict[str, Any], 
                              timeout: float = None) -> Optional[Dict[str, Any]]:
        """请求人工批准
        
        Args:
            request: 请求内容
            timeout: 超时时间
            
        Returns:
            人工响应
        """
        timeout = timeout or self.config.human_approval_timeout
        
        self._pending_approval = request
        self._emit_event(
            AgentEventType.HUMAN_INPUT_REQUEST,
            {"request": request}
        )
        
        try:
            response = self._human_input_queue.get(timeout=timeout)
            self._pending_approval = None
            return response
        except queue.Empty:
            self._pending_approval = None
            return None
    
    def get_current_state(self) -> Optional[AgentState]:
        """获取当前执行状态"""
        return self._current_state
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = {
            "name": self.config.name,
            "type": self.__class__.__name__,
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "execution_count": self._execution_count,
            "tool_count": len(self.tools),
            "callbacks_count": len(self._callback_manager.callbacks)
        }
        
        # 添加性能分析数据
        if self._profiling_callback:
            stats["profiling"] = self._profiling_callback.get_profile(self.config.name)
        
        # 添加执行控制器统计
        if self._execution_controller:
            stats["execution_controller"] = self._execution_controller.get_stats()
        
        # 添加工具统计
        if self._tool_executor:
            stats["tool_executor"] = self._tool_executor.get_stats()
        
        # 添加检查点统计
        if hasattr(self.checkpointer, 'get_metrics'):
            stats["checkpointer"] = self.checkpointer.get_metrics().to_dict()
        
        return stats
    
    def get_memory_context(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取内存上下文"""
        if self._memory_manager:
            return self._memory_manager.get_context(thread_id)
        return None
    
    def clear_memory(self, thread_id: str) -> None:
        """清除内存"""
        if self._memory_manager:
            self._memory_manager.clear(thread_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.config.name,
            "type": self.__class__.__name__,
            "version": self.config.version,
            "description": self.config.description,
            "config": self.config.to_dict(),
            "tools": [t.name for t in self.tools],
            "metrics": self.get_metrics(),
            "execution_stats": self.get_execution_stats()
        }
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置（可用于重建 Agent）"""
        return {
            "agent_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "tools": [t.name for t in self.tools],
            "checkpointer_type": type(self.checkpointer).__name__
        }
    
    # ==================== 增强方法：状态模块集成 ====================
    
    def create_execution_context(self,
                                priority: PriorityLevel = PriorityLevel.NORMAL,
                                execution_mode: StateExecutionMode = StateExecutionMode.ASYNC,
                                interrupt_type: InterruptType = InterruptType.NONE,
                                timeout: float = None,
                                metadata: Dict[str, Any] = None) -> StateExecutionContext:
        """创建执行上下文
        
        Args:
            priority: 优先级
            execution_mode: 执行模式
            interrupt_type: 中断类型
            timeout: 超时时间
            metadata: 元数据
        
        Returns:
            StateExecutionContext 实例
        """
        return StateExecutionContext(
            priority=priority,
            execution_mode=execution_mode,
            interrupt_type=interrupt_type,
            timeout=timeout or self.config.timeout,
            metadata=metadata or {}
        )
    
    def create_plan(self, goal: str, steps: List[str],
                   context: Dict[str, Any] = None) -> AgentPlan:
        """创建执行计划
        
        Args:
            goal: 目标
            steps: 步骤列表
            context: 上下文
        
        Returns:
            AgentPlan 实例
        """
        plan_steps = [
            PlanStep(
                step_id=str(uuid.uuid4()),
                description=step,
                order=i
            )
            for i, step in enumerate(steps)
        ]
        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            steps=plan_steps,
            context=context or {}
        )
    
    def create_reflection(self, content: str, quality_score: float = 0.0,
                         suggestions: List[str] = None) -> Reflection:
        """创建反思记录
        
        Args:
            content: 反思内容
            quality_score: 质量评分
            suggestions: 改进建议
        
        Returns:
            Reflection 实例
        """
        return Reflection(
            reflection_id=str(uuid.uuid4()),
            content=content,
            quality_score=quality_score,
            suggestions=suggestions or [],
            created_at=datetime.utcnow()
        )
    
    def create_state_checkpoint(self, state: AgentState = None,
                               description: str = "") -> StateCheckpoint:
        """创建状态检查点
        
        Args:
            state: 要保存的状态（默认使用当前状态）
            description: 描述
        
        Returns:
            StateCheckpoint 实例
        """
        target_state = state or self._current_state
        if not target_state:
            raise ValueError("No state available for checkpoint")
        
        return StateCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            state=target_state.copy(),
            created_at=datetime.utcnow(),
            description=description
        )
    
    def get_error_type(self, exception: Exception) -> ErrorType:
        """获取异常对应的错误类型
        
        Args:
            exception: 异常
        
        Returns:
            ErrorType 枚举值
        """
        if isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT
        elif isinstance(exception, ValueError):
            return ErrorType.VALIDATION
        elif isinstance(exception, MemoryError):
            return ErrorType.RESOURCE
        else:
            return ErrorType.EXECUTION
    
    # ==================== 增强方法：节点模块集成 ====================
    
    def create_node(self, node_type: NodeType, name: str,
                   config: Dict[str, Any] = None, **kwargs) -> BaseNode:
        """创建节点
        
        Args:
            node_type: 节点类型
            name: 节点名称
            config: 节点配置
            **kwargs: 额外参数
        
        Returns:
            BaseNode 实例
        """
        registry = get_node_registry()
        return registry.create(node_type.value, name=name, config=config, **kwargs)
    
    def get_node_metrics(self, node_name: str = None) -> Dict[str, NodeMetrics]:
        """获取节点指标
        
        Args:
            node_name: 节点名称（可选，不指定则返回所有）
        
        Returns:
            节点指标字典
        """
        metrics = {}
        if self._graph:
            for name, node in self._graph.nodes.items():
                if node_name and name != node_name:
                    continue
                if hasattr(node, 'get_metrics'):
                    metrics[name] = node.get_metrics()
                else:
                    metrics[name] = NodeMetrics(
                        node_name=name,
                        total_calls=0,
                        successful_calls=0,
                        failed_calls=0,
                        total_latency_ms=0.0
                    )
        return metrics
    
    def create_node_chain(self, nodes: List[BaseNode]) -> NodeChain:
        """创建节点链
        
        Args:
            nodes: 节点列表
        
        Returns:
            NodeChain 实例
        """
        return NodeChain(nodes=nodes)
    
    def create_parallel_node_group(self, nodes: List[BaseNode],
                                  merge_strategy: str = "merge") -> NodeParallelGroup:
        """创建并行节点组
        
        Args:
            nodes: 节点列表
            merge_strategy: 合并策略
        
        Returns:
            NodeParallelGroup 实例
        """
        return NodeParallelGroup(nodes=nodes, merge_strategy=merge_strategy)
    
    # ==================== 增强方法：边模块集成 ====================
    
    def create_conditional_edge(self, source: str,
                               condition: Callable[[AgentState], str],
                               branches: Dict[str, str],
                               edge_type: EdgeType = EdgeType.CONDITIONAL) -> ConditionalEdge:
        """创建条件边
        
        Args:
            source: 源节点
            condition: 条件函数
            branches: 分支映射
            edge_type: 边类型
        
        Returns:
            ConditionalEdge 实例
        """
        return ConditionalEdge(
            source=source,
            condition=condition,
            branches=branches,
            edge_type=edge_type
        )
    
    def create_edge_conditions(self, *conditions: EdgeCondition) -> EdgeConditions:
        """创建边条件集合
        
        Args:
            *conditions: 条件列表
        
        Returns:
            EdgeConditions 实例
        """
        return EdgeConditions(conditions=list(conditions))
    
    def route_tools(self, state: AgentState) -> str:
        """工具路由函数
        
        Args:
            state: 当前状态
        
        Returns:
            下一个节点名称
        """
        return route_after_tools(state)
    
    def create_priority_router(self, priorities: Dict[str, int]) -> PriorityRouter:
        """创建优先级路由器
        
        Args:
            priorities: 节点优先级映射
        
        Returns:
            PriorityRouter 实例
        """
        return PriorityRouter(priorities=priorities)
    
    def create_load_balance_router(self, nodes: List[str],
                                  strategy: str = "round_robin") -> LoadBalanceRouter:
        """创建负载均衡路由器
        
        Args:
            nodes: 节点列表
            strategy: 负载均衡策略
        
        Returns:
            LoadBalanceRouter 实例
        """
        return LoadBalanceRouter(nodes=nodes, strategy=strategy)
    
    # ==================== 增强方法：图模块集成 ====================
    
    def create_simple_graph(self, name: str, nodes: List[BaseNode],
                           edges: List[Edge] = None) -> StateGraph:
        """创建简单的 Agent 图
        
        Args:
            name: 图名称
            nodes: 节点列表
            edges: 边列表
        
        Returns:
            StateGraph 实例
        """
        return create_simple_graph(name, nodes, edges)
    
    def create_react_graph(self, name: str, llm_node: LLMNode,
                          tool_node: ToolNode,
                          config: GraphConfig = None) -> StateGraph:
        """创建 ReAct Agent 图
        
        Args:
            name: 图名称
            llm_node: LLM 节点
            tool_node: 工具节点
            config: 图配置
        
        Returns:
            StateGraph 实例
        """
        return create_react_graph(name, llm_node, tool_node, config)
    
    def get_graph_status(self) -> GraphStatus:
        """获取图状态
        
        Returns:
            GraphStatus 枚举值
        """
        if self._graph and hasattr(self._graph, 'status'):
            return self._graph.status
        return GraphStatus.IDLE
    
    def get_graph_metrics(self) -> GraphMetrics:
        """获取图执行指标
        
        Returns:
            GraphMetrics 实例
        """
        if self._graph and hasattr(self._graph, 'get_metrics'):
            return self._graph.get_metrics()
        return GraphMetrics(
            graph_name=self.config.name,
            total_runs=self._execution_count,
            successful_runs=0,
            failed_runs=0
        )
    
    def create_execution_event(self, event_type: str, node_name: str,
                              data: Dict[str, Any] = None) -> ExecutionEvent:
        """创建执行事件
        
        Args:
            event_type: 事件类型
            node_name: 节点名称
            data: 事件数据
        
        Returns:
            ExecutionEvent 实例
        """
        return ExecutionEvent(
            event_type=event_type,
            node_name=node_name,
            timestamp=datetime.utcnow(),
            data=data or {}
        )
    
    # ==================== 增强方法：工具模块集成 ====================
    
    def create_tool_parameter(self, name: str, param_type: str,
                             description: str, required: bool = True,
                             default: Any = None) -> ToolParameter:
        """创建工具参数定义
        
        Args:
            name: 参数名
            param_type: 参数类型
            description: 描述
            required: 是否必需
            default: 默认值
        
        Returns:
            ToolParameter 实例
        """
        return ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default
        )
    
    def get_tool_status(self, tool_name: str) -> ToolStatus:
        """获取工具状态
        
        Args:
            tool_name: 工具名称
        
        Returns:
            ToolStatus 枚举值
        """
        tool_item = self._tool_registry.get(tool_name)
        if tool_item and hasattr(tool_item, 'status'):
            return tool_item.status
        return ToolStatus.READY
    
    def get_tool_metrics_by_name(self, tool_name: str) -> ToolMetrics:
        """获取工具指标
        
        Args:
            tool_name: 工具名称
        
        Returns:
            ToolMetrics 实例
        """
        tool_item = self._tool_registry.get(tool_name)
        if tool_item and hasattr(tool_item, 'get_metrics'):
            return tool_item.get_metrics()
        return ToolMetrics(
            tool_name=tool_name,
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            total_latency_ms=0.0
        )
    
    def create_tool_hooks(self, before_call: Callable = None,
                         after_call: Callable = None,
                         on_error: Callable = None) -> ToolHooks:
        """创建工具钩子
        
        Args:
            before_call: 调用前钩子
            after_call: 调用后钩子
            on_error: 错误钩子
        
        Returns:
            ToolHooks 实例
        """
        return ToolHooks(
            before_call=before_call,
            after_call=after_call,
            on_error=on_error
        )
    
    def add_tool_with_decorator(self, name: str, description: str,
                               func: Callable) -> Tool:
        """使用装饰器方式添加工具
        
        Args:
            name: 工具名称
            description: 描述
            func: 工具函数
        
        Returns:
            创建的 Tool 实例
        """
        @tool(name=name, description=description)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        
        self.add_tool(wrapped)
        return wrapped
    
    def add_async_tool_with_decorator(self, name: str, description: str,
                                     func: Callable) -> Tool:
        """使用装饰器方式添加异步工具
        
        Args:
            name: 工具名称
            description: 描述
            func: 异步工具函数
        
        Returns:
            创建的 Tool 实例
        """
        @async_tool(name=name, description=description)
        async def wrapped(*args, **kwargs):
            return await func(*args, **kwargs)
        
        self.add_tool(wrapped)
        return wrapped
    
    # ==================== 增强方法：检查点模块集成 ====================
    
    def set_redis_checkpointer(self, host: str = "localhost",
                              port: int = 6379,
                              db: int = 0,
                              password: str = None,
                              prefix: str = "agent_checkpoint") -> None:
        """设置 Redis 检查点器
        
        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库索引
            password: 密码
            prefix: 键前缀
        """
        self.checkpointer = RedisCheckpointer(
            host=host,
            port=port,
            db=db,
            password=password,
            prefix=prefix
        )
    
    def set_sqlite_checkpointer(self, db_path: str = "checkpoints.db",
                               table_name: str = "checkpoints") -> None:
        """设置 SQLite 检查点器
        
        Args:
            db_path: 数据库路径
            table_name: 表名
        """
        self.checkpointer = SQLiteCheckpointer(
            db_path=db_path,
            table_name=table_name
        )
    
    def set_file_checkpointer(self, directory: str = "./checkpoints",
                             compression: CompressionType = CompressionType.GZIP) -> None:
        """设置文件检查点器
        
        Args:
            directory: 存储目录
            compression: 压缩类型
        """
        self.checkpointer = FileCheckpointer(
            directory=directory,
            compression=compression
        )
    
    def get_checkpointer_type(self) -> str:
        """获取检查点器类型"""
        return type(self.checkpointer).__name__
    
    # ==================== 增强方法：内置工具模块集成 ====================
    
    def get_builtin_tool(self, name: str) -> Optional[Tool]:
        """按名称获取内置工具
        
        Args:
            name: 工具名称
        
        Returns:
            Tool 实例或 None
        """
        return get_tool_by_name(name)
    
    def get_builtin_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取内置工具详细信息
        
        Args:
            name: 工具名称
        
        Returns:
            工具信息字典或 None
        """
        return get_tool_info(name)
    
    def add_builtin_tools(self, tool_names: List[str] = None) -> List[str]:
        """添加内置工具
        
        Args:
            tool_names: 要添加的工具名称列表（可选，不指定则添加所有）
        
        Returns:
            添加的工具名称列表
        """
        added = []
        if tool_names:
            for name in tool_names:
                t = get_tool_by_name(name)
                if t:
                    self.add_tool(t)
                    added.append(name)
        else:
            for t in get_builtin_tools():
                self.add_tool(t)
                added.append(t.name)
        return added
    
    def add_builtin_tools_by_category(self, 
                                     category: BuiltinToolCategory) -> List[str]:
        """按类别添加内置工具
        
        Args:
            category: 工具类别
        
        Returns:
            添加的工具名称列表
        """
        added = []
        for t in get_tools_by_category(category.value):
            self.add_tool(t)
            added.append(t.name)
        return added
    
    def add_recommended_builtin_tools(self) -> List[str]:
        """添加当前 Agent 类型推荐的内置工具
        
        Returns:
            添加的工具名称列表
        """
        added = []
        agent_type = self.__class__.__name__.lower().replace("agent", "")
        for t in get_tools_for_agent(agent_type):
            self.add_tool(t)
            added.append(t.name)
        return added
    
    def close(self) -> None:
        """关闭并清理资源"""
        # 取消执行
        if self._is_running:
            self.cancel()
        
        # 关闭回调管理器
        self._callback_manager.close()
        
        # 关闭检查点
        if hasattr(self.checkpointer, 'close'):
            self.checkpointer.close()
        
        self.logger.info(f"Agent {self.name} closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "idle"
        return f"{self.__class__.__name__}(name='{self.config.name}', tools={len(self.tools)}, status={status})"
    
    # ==================== 工业级派生方法 ====================
    
    def _get_master_factory(self):
        """延迟加载 MasterFactory 避免循环导入"""
        if not hasattr(self, '_master_factory') or self._master_factory is None:
            from .factory import get_master_factory
            self._master_factory = get_master_factory()
        return self._master_factory
    
    def create_agent_message_via_factory(self, content: str,
                                        message_type: MessageType = MessageType.HUMAN,
                                        name: str = None,
                                        metadata: Dict[str, Any] = None) -> AgentMessage:
        """通过工厂创建 Agent 消息
        
        调用 MasterFactory.create_agent_message
        
        Args:
            content: 消息内容
            message_type: 消息类型
            name: 发送者名称
            metadata: 元数据
            
        Returns:
            AgentMessage 实例
        """
        factory = self._get_master_factory()
        return factory.create_agent_message(
            content=content,
            message_type=message_type,
            name=name or self.name,
            metadata=metadata
        )
    
    def create_tool_call_via_factory(self, tool_name: str,
                                    arguments: Dict[str, Any],
                                    tool_call_id: str = None) -> ToolCall:
        """通过工厂创建工具调用
        
        调用 MasterFactory.create_tool_call
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            ToolCall 实例
        """
        factory = self._get_master_factory()
        return factory.create_tool_call(
            name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
        )
    
    def create_tool_result_via_factory(self, tool_call_id: str,
                                       tool_name: str,
                                       result: Any,
                                       success: bool = True,
                                       error: str = None) -> ToolResult:
        """通过工厂创建工具结果
        
        调用 MasterFactory.create_tool_result
        
        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            ToolResult 实例
        """
        factory = self._get_master_factory()
        return factory.create_tool_result(
            tool_call_id=tool_call_id,
            name=tool_name,
            result=result,
            success=success,
            error=error
        )
    
    def create_memory_entry_via_factory(self, content: Any,
                                       memory_type_name: str = "short_term",
                                       importance: float = 0.5,
                                       tags: List[str] = None) -> MemoryEntry:
        """通过工厂创建记忆条目
        
        调用 MasterFactory.create_memory_entry
        
        Args:
            content: 记忆内容
            memory_type_name: 记忆类型名称
            importance: 重要性分数
            tags: 标签列表
            
        Returns:
            MemoryEntry 实例
        """
        factory = self._get_master_factory()
        mem_type = factory.get_memory_type(memory_type_name)
        return factory.create_memory_entry(
            content=content,
            memory_type=mem_type,
            importance=importance,
            tags=tags or [self.name]
        )
    
    def create_plan_step_via_factory(self, description: str,
                                    action: str = "",
                                    dependencies: List[str] = None) -> PlanStep:
        """通过工厂创建计划步骤
        
        调用 MasterFactory.create_plan_step
        
        Args:
            description: 步骤描述
            action: 步骤动作
            dependencies: 依赖步骤
            
        Returns:
            PlanStep 实例
        """
        factory = self._get_master_factory()
        return factory.create_plan_step(
            description=description,
            action=action,
            dependencies=dependencies
        )
    
    def create_enhanced_checkpoint_via_factory(self, checkpoint_id: str = None,
                                              branch: str = "main",
                                              tags: List[str] = None) -> 'EnhancedCheckpoint':
        """通过工厂创建增强检查点
        
        调用 MasterFactory.create_enhanced_checkpoint
        
        Args:
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            EnhancedCheckpoint 实例
        """
        factory = self._get_master_factory()
        cp_id = checkpoint_id or f"cp_{self.name}_{uuid.uuid4().hex[:8]}"
        
        # 获取当前状态
        state_data = {}
        if hasattr(self, '_current_state') and self._current_state:
            state_data = self._current_state.__dict__ if hasattr(self._current_state, '__dict__') else {}
        
        return factory.create_enhanced_checkpoint(
            checkpoint_id=cp_id,
            thread_id=f"thread_{self.name}",
            state=state_data,
            tags=tags or [self.name],
            branch=branch
        )
    
    def create_checkpoint_tag_via_factory(self, tag_name: str,
                                         checkpoint_id: str,
                                         description: str = "") -> 'CheckpointTag':
        """通过工厂创建检查点标签
        
        调用 MasterFactory.create_checkpoint_tag
        
        Args:
            tag_name: 标签名称
            checkpoint_id: 检查点 ID
            description: 标签描述
            
        Returns:
            CheckpointTag 实例
        """
        factory = self._get_master_factory()
        return factory.create_checkpoint_tag(
            name=tag_name,
            checkpoint_id=checkpoint_id,
            description=description
        )
    
    def setup_tool_cache_via_factory(self, max_size: int = 1000,
                                    default_ttl: int = 300) -> 'ToolCache':
        """通过工厂设置工具缓存
        
        调用 MasterFactory.create_tool_cache
        
        Args:
            max_size: 缓存最大大小
            default_ttl: 默认 TTL
            
        Returns:
            ToolCache 实例
        """
        factory = self._get_master_factory()
        tool_cache = factory.create_tool_cache(max_size, default_ttl)
        
        if self._tool_executor:
            self._tool_executor._cache = tool_cache
        
        return tool_cache
    
    def setup_retry_handler_via_factory(self, max_retries: int = 3,
                                       base_delay: float = 1.0,
                                       max_delay: float = 60.0) -> 'RetryHandler':
        """通过工厂设置重试处理器
        
        调用 MasterFactory.create_retry_handler
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟
            max_delay: 最大延迟
            
        Returns:
            RetryHandler 实例
        """
        factory = self._get_master_factory()
        retry_handler = factory.create_retry_handler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay
        )
        
        if self._tool_executor:
            self._tool_executor._retry_handler = retry_handler
        
        return retry_handler
    
    def setup_rate_limiter_via_factory(self, rate: float = 10.0,
                                      burst: int = 20) -> 'ToolRateLimiter':
        """通过工厂设置限流器
        
        调用 MasterFactory.create_tool_rate_limiter
        
        Args:
            rate: 速率
            burst: 突发数
            
        Returns:
            ToolRateLimiter 实例
        """
        factory = self._get_master_factory()
        rate_limiter = factory.create_tool_rate_limiter(rate, burst)
        
        if self._tool_executor:
            self._tool_executor._rate_limiter = rate_limiter
        
        return rate_limiter
    
    def setup_tool_hooks_via_factory(self, before_execute: List[Callable] = None,
                                    after_execute: List[Callable] = None,
                                    on_error: List[Callable] = None) -> 'ToolHooks':
        """通过工厂设置工具钩子
        
        调用 MasterFactory.create_tool_hooks
        
        Args:
            before_execute: 执行前钩子列表
            after_execute: 执行后钩子列表
            on_error: 错误钩子列表
            
        Returns:
            ToolHooks 实例
        """
        factory = self._get_master_factory()
        return factory.create_tool_hooks(
            before_execute=before_execute,
            after_execute=after_execute,
            on_error=on_error
        )
    
    def get_node_metrics_via_factory(self) -> 'NodeMetrics':
        """通过工厂获取节点指标
        
        调用 MasterFactory.create_node_metrics
        
        Returns:
            NodeMetrics 实例
        """
        factory = self._get_master_factory()
        return factory.create_node_metrics(self.name)
    
    def get_tool_metrics_via_factory(self, tool_name: str) -> 'ToolMetrics':
        """通过工厂获取工具指标
        
        调用 MasterFactory.create_tool_metrics
        
        Args:
            tool_name: 工具名称
            
        Returns:
            ToolMetrics 实例
        """
        factory = self._get_master_factory()
        return factory.create_tool_metrics(tool_name)
    
    def get_graph_metrics_via_factory(self) -> 'GraphMetrics':
        """通过工厂获取图指标
        
        调用 MasterFactory.create_graph_metrics
        
        Returns:
            GraphMetrics 实例
        """
        factory = self._get_master_factory()
        return factory.create_graph_metrics()
    
    def get_checkpoint_metrics_via_factory(self) -> 'CheckpointMetrics':
        """通过工厂获取检查点指标
        
        调用 MasterFactory.create_checkpoint_metrics
        
        Returns:
            CheckpointMetrics 实例
        """
        factory = self._get_master_factory()
        return factory.create_checkpoint_metrics()
    
    def get_agent_health_with_factory(self) -> Dict[str, Any]:
        """使用工厂获取 Agent 健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        factory = self._get_master_factory()
        return {
            "agent_name": self.name,
            "agent_type": self.__class__.__name__,
            "is_running": self._is_running,
            "tools_count": len(self.tools),
            "factory_health": factory.health_check(),
            "factory_metrics": factory.get_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }


class ReActAgent(BaseAgent):
    """ReAct Agent - 生产级实现
    
    实现 Reasoning + Acting 循环，是最常用的 Agent 模式：
    
    工作流程：
    1. **思考 (Thought)**：LLM 分析问题，思考需要什么信息或操作
    2. **行动 (Action)**：如果需要，调用合适的工具
    3. **观察 (Observation)**：获取工具执行结果
    4. **重复**：继续思考和行动，直到得出最终答案
    
    特性：
    - 支持多种 LLM 后端
    - 自动工具调用解析和执行
    - 可配置的终止条件
    - 支持中间步骤导出
    - 支持人机协作中断
    
    使用示例：
        agent = ReActAgent(
            config=AgentConfig(name="assistant"),
            tools=[search_tool, calculator_tool],
            llm_client=openai_client
        )
        result = agent.invoke("帮我计算 123 + 456 等于多少")
    """
    
    def __init__(self, 
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 stop_sequences: List[str] = None,
                 early_stopping_method: str = "force"):
        """初始化 ReAct Agent
        
        Args:
            config: Agent 配置
            tools: 可用工具列表
            llm_client: LLM 客户端
            checkpointer: 检查点管理器
            callbacks: 回调列表
            stop_sequences: 停止序列
            early_stopping_method: 早停方法 ("force" | "generate")
        """
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        
        self.stop_sequences = stop_sequences or ["最终答案", "Final Answer", "任务完成"]
        self.early_stopping_method = early_stopping_method
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """默认 ReAct 系统提示"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ]) if self.tools else "当前没有可用的工具。"
        
        return f"""你是一个智能助手，使用 ReAct（Reasoning + Acting）框架来解决问题。

## 可用工具
{tool_descriptions}

## 工作流程

对于每个问题，请按以下步骤处理：

1. **思考 (Thought)**
   - 分析问题的核心需求
   - 确定需要什么信息或操作
   - 规划解决步骤

2. **行动 (Action)**
   - 如果需要外部信息，调用合适的工具
   - 确保提供正确的参数
   - 一次只调用必要的工具

3. **观察 (Observation)**
- 分析工具返回的结果
   - 判断是否需要进一步操作
   - 提取有用的信息

4. **重复或总结**
   - 如果信息不足，继续思考和行动
   - 如果已有足够信息，给出最终答案

## 重要规则

- 每次只思考一步，避免过度规划
- 工具调用要精确，参数要完整
- 遇到错误时尝试其他方法
- 当你准备好回答时，以"**最终答案**"或"**总结**"开头

## 格式要求

- 思考过程要清晰，便于理解
- 最终答案要直接、准确
- 如果无法完成任务，诚实说明原因"""
    
    def _build_graph(self) -> StateGraph:
        """构建 ReAct 图 - 使用增强方法"""
        # 使用 create_react_graph 辅助方法（如果简单场景）或手动构建
        graph = StateGraph(config=GraphConfig(
            name=f"react_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 获取 LLM 提供者
        provider = LLMProvider.OPENAI
        if self.config.provider:
            try:
                provider = LLMProvider(self.config.provider)
            except ValueError:
                pass
        
        # 使用 create_node 方法创建 LLM 节点
        try:
            agent_node = self.create_node(
                NodeType.LLM, "agent",
                config={
                    "llm_client": self.llm_client,
                    "provider": provider,
                    "system_prompt": self.config.system_prompt,
                    "tools": self.tools,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "model": self.config.model,
                    "retry_count": self.config.max_retries,
                    "retry_delay": self.config.retry_delay,
                    "enable_streaming": self.config.enable_streaming
                }
            )
        except Exception:
            # 回退到直接创建
            agent_node = LLMNode(
                name="agent",
                llm_client=self.llm_client,
                provider=provider,
                system_prompt=self.config.system_prompt,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.model,
                retry_count=self.config.max_retries,
                retry_delay=self.config.retry_delay,
                enable_streaming=self.config.enable_streaming
            )
        graph.add_node("agent", agent_node)
        
        # 使用 create_node 方法创建工具节点
        try:
            tool_node = self.create_node(
                NodeType.TOOL, "tools",
                config={
                    "registry": self._tool_registry,
                    "parallel": self.config.parallel_tool_execution
                }
            )
        except Exception:
            # 回退到直接创建
            tool_node = ToolNode(
                name="tools",
                registry=self._tool_registry,
                parallel=self.config.parallel_tool_execution
            )
        graph.add_node("tools", tool_node)
        
        # 记录节点指标
        self._emit_event(AgentEventType.STATE_UPDATE, {
            "action": "nodes_created",
            "nodes": ["agent", "tools"]
        })
        
        # 添加人机交互节点（如果启用）
        if self.config.enable_human_in_loop:
            try:
                human_node = self.create_node(NodeType.HUMAN, "human")
            except Exception:
                human_node = HumanNode(name="human")
            graph.add_node("human", human_node)
        
        # 设置入口点
        graph.set_entry_point("agent")
        
        # 创建执行上下文用于路由决策
        exec_ctx = self.create_execution_context(
            priority=PriorityLevel.NORMAL,
            execution_mode=StateExecutionMode.ASYNC,
            interrupt_type=InterruptType.ON_TOOL_CALL if self.config.enable_human_in_loop else InterruptType.NONE,
            timeout=self.config.timeout
        )
        
        # 自定义路由函数 - 使用增强的路由方法
        def custom_route_after_agent(state: AgentState) -> str:
            """自定义 Agent 后路由"""
            # 检查是否有待执行的工具调用
            if state.pending_tool_calls:
                # 创建执行事件
                self._emit_event(
                    AgentEventType.TOOL_START,
                    {"tools": [tc.name for tc in state.pending_tool_calls]}
                )
                # 记录工具状态
                for tc in state.pending_tool_calls:
                    status = self.get_tool_status(tc.name)
                    if status != ToolStatus.READY:
                        self.logger.warning(f"Tool {tc.name} status: {status}")
                return "tools"
            
            # 检查最后消息
            last_message = state.get_last_message()
            if last_message and last_message.type == MessageType.AI:
                content = last_message.content or ""
                
                # 检查是否包含停止序列
                if any(stop in content for stop in self.stop_sequences):
                    return "__end__"
                
                # 检查是否请求人工输入
                if self.config.enable_human_in_loop and state.waiting_for_human:
                    return "human"
            
            # 检查是否已有最终答案
            if state.final_answer:
                return "__end__"
            
            # 检查迭代次数
            if state.iteration >= state.max_iterations:
                self.logger.warning(f"Max iterations reached ({state.iteration})")
                # 创建状态检查点用于调试
                try:
                    checkpoint = self.create_state_checkpoint(state, "max_iterations_reached")
                    self.logger.info(f"Created checkpoint: {checkpoint.checkpoint_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create checkpoint: {e}")
                return "__end__"
            
            # 继续 Agent 循环
            return "agent"
        
        # 添加条件边 - 使用 create_conditional_edge
        if self.config.enable_human_in_loop:
            cond_edge = self.create_conditional_edge(
                "agent",
                custom_route_after_agent,
                {
                    "tools": "tools",
                    "human": "human",
                    "agent": "agent",
                    "__end__": "__end__"
                },
                EdgeType.CONDITIONAL
            )
            graph.add_conditional_edges(
                "agent",
                custom_route_after_agent,
                cond_edge.branches
            )
            graph.add_edge("human", "agent")
        else:
            cond_edge = self.create_conditional_edge(
                "agent",
                custom_route_after_agent,
                {
                    "tools": "tools",
                    "agent": "agent",
                    "__end__": "__end__"
                },
                EdgeType.CONDITIONAL
            )
            graph.add_conditional_edges(
                "agent",
                custom_route_after_agent,
                cond_edge.branches
            )
        
        # 工具执行后的路由 - 使用 route_tools 方法
        def enhanced_route_after_tools(state: AgentState) -> str:
            """工具执行后路由 - 增强版"""
            # 发送工具完成事件
            self._emit_event(
                AgentEventType.TOOL_END,
                {"results": [tr.name for tr in state.tool_results]}
            )
            
            # 收集工具指标
            for tr in state.tool_results:
                try:
                    metrics = self.get_tool_metrics_by_name(tr.name)
                    self.logger.debug(f"Tool {tr.name} metrics: calls={metrics.total_calls}")
                except Exception:
                    pass
            
            # 使用 route_tools 方法进行路由
            return self.route_tools(state) if hasattr(self, 'route_tools') else "agent"
        
        graph.add_conditional_edges(
            "tools",
            enhanced_route_after_tools,
            {"agent": "agent"}
        )
        
        return graph
    
    def invoke_with_scratchpad(self, 
                               input_data: Union[str, Dict[str, Any]],
                               scratchpad: str = None) -> AgentState:
        """带草稿本的调用
        
        允许传入之前的思考过程，支持续写。
        
        Args:
            input_data: 输入数据
            scratchpad: 之前的思考草稿
            
        Returns:
            最终状态
        """
        state = self._prepare_state(input_data)
        
        if scratchpad:
            state.add_message(AgentMessage.ai(scratchpad))
        
        return self.invoke(state)
    
    def get_intermediate_steps(self, state: AgentState) -> List[Tuple[Dict, str]]:
        """获取中间步骤
        
        Args:
            state: Agent 状态
            
        Returns:
            (action, observation) 元组列表
        """
        steps = []
        for step in state.intermediate_steps:
            action = step.get('action', {})
            observation = step.get('observation', '')
            steps.append((action, str(observation)))
        return steps


class PlanAndExecuteAgent(BaseAgent):
    """计划执行 Agent - 生产级实现
    
    两阶段工作流，适合复杂任务处理：
    
    工作流程：
    1. **计划阶段**：分析任务，制定详细的执行计划
    2. **执行阶段**：逐步执行计划中的每个步骤
    3. **评估阶段**：评估执行结果
    4. **重规划**（可选）：根据执行结果调整计划
    
    特性：
    - 任务自动分解
    - 步骤依赖管理
    - 失败重试和回滚
    - 执行进度追踪
    - 支持部分执行和恢复
    
    使用示例：
        agent = PlanAndExecuteAgent(
            config=AgentConfig(name="planner"),
            tools=[research_tool, write_tool],
            enable_replanning=True
        )
        result = agent.invoke("撰写一篇关于 AI 的研究报告")
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 enable_replanning: bool = True,
                 max_replans: int = 3,
                 step_timeout: float = 60.0):
        """初始化计划执行 Agent
        
        Args:
            config: Agent 配置
            tools: 可用工具列表
            llm_client: LLM 客户端
            checkpointer: 检查点管理器
            callbacks: 回调列表
            enable_replanning: 是否启用重规划
            max_replans: 最大重规划次数
            step_timeout: 单步执行超时
        """
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.enable_replanning = enable_replanning
        self.max_replans = max_replans
        self.step_timeout = step_timeout
        self._replan_count = 0
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        tool_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools]) if self.tools else "无"
        
        return f"""你是一个专业的任务规划和执行助手。

## 可用工具
{tool_list}

## 工作方式

### 1. 计划阶段
分析任务需求，制定详细的执行计划：
- 将复杂任务分解为简单的子步骤
- 每个步骤应该明确、具体、可执行
- 考虑步骤之间的依赖关系和先后顺序
- 估计每个步骤的难度和所需资源

### 2. 执行阶段
按计划逐步执行：
- 严格按照计划顺序执行
- 使用合适的工具完成每个步骤
- 记录每个步骤的输出结果
- 遇到问题时及时记录

### 3. 评估阶段
评估执行结果：
- 检查每个步骤是否成功完成
- 验证输出是否符合预期
- 如果失败，分析原因

### 4. 重规划（如需要）
根据执行情况调整计划：
- 修正失败的步骤
- 添加遗漏的步骤
- 优化执行顺序

## 输出格式

制定计划时，请使用以下格式：
```
计划：
1. [步骤1描述]
2. [步骤2描述]
...
```

执行完成后，请总结：
```
执行结果：
- 步骤1: [结果]
- 步骤2: [结果]
...

最终答案：[综合结论]
```"""
    
    def _build_graph(self) -> StateGraph:
        """构建计划执行图 - 使用增强方法"""
        graph = StateGraph(config=GraphConfig(
            name=f"plan_execute_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 获取 LLM 提供者
        provider = LLMProvider.OPENAI
        if self.config.provider:
            try:
                provider = LLMProvider(self.config.provider)
            except ValueError:
                pass
        
        # 创建执行上下文
        exec_context = self.create_execution_context(
            priority=PriorityLevel.HIGH,
            execution_mode=StateExecutionMode.ASYNC,
            timeout=self.config.timeout,
            metadata={"agent_type": "plan_execute", "enable_replanning": self.enable_replanning}
        )
        
        # 使用 create_node 创建 LLM 节点用于计划生成
        try:
            planner_llm = self.create_node(
                NodeType.LLM, "planner_llm",
                config={
                    "llm_client": self.llm_client,
                    "provider": provider,
                    "system_prompt": self.config.system_prompt,
                    "tools": [],
                    "temperature": 0.3,
                    "model": self.config.model
                }
            )
        except Exception:
            planner_llm = LLMNode(
                name="planner_llm",
                llm_client=self.llm_client,
                provider=provider,
                system_prompt=self.config.system_prompt,
                tools=[],
                temperature=0.3,
                model=self.config.model
            )
        
        # 使用 create_node 创建 LLM 节点用于步骤执行
        try:
            executor_llm = self.create_node(
                NodeType.LLM, "executor_llm",
                config={
                    "llm_client": self.llm_client,
                    "provider": provider,
                    "system_prompt": "你是一个任务执行助手，请根据计划执行当前步骤。",
                    "tools": self.tools,
                    "temperature": self.config.temperature,
                    "model": self.config.model
                }
            )
        except Exception:
            executor_llm = LLMNode(
                name="executor_llm",
                llm_client=self.llm_client,
                provider=provider,
                system_prompt="你是一个任务执行助手，请根据计划执行当前步骤。",
                tools=self.tools,
                temperature=self.config.temperature,
                model=self.config.model
            )
        
        # 计划节点 - 使用 create_plan 方法
        def planner(state: AgentState) -> AgentState:
            """制定或更新计划 - 使用增强方法"""
            self._emit_event(
                AgentEventType.STATE_UPDATE,
                {"phase": "planning", "replan_count": self._replan_count}
            )
            
            # 创建执行事件记录计划阶段开始
            plan_event = self.create_execution_event(
                event_type="plan_start",
                node_name="planner",
                data={"replan_count": self._replan_count, "goal": state.input}
            )
            self.logger.debug(f"Planning event: {plan_event.event_type}")
            
            if not state.plan or (self.enable_replanning and state.error):
                # 需要制定新计划
                if state.error and self._replan_count < self.max_replans:
                    # 重规划 - 使用 get_error_type 分析错误
                    error_type = self.get_error_type(Exception(state.error))
                    prompt = f"""之前的计划执行遇到问题（错误类型: {error_type.value}），需要重新规划。

原任务：{state.input}

已完成的步骤：
{self._format_completed_steps(state)}

遇到的问题：{state.error}

请制定新的执行计划，考虑已完成的工作："""
                    self._replan_count += 1
                    state.error = None
                else:
                    # 首次规划
                    prompt = f"""请为以下任务制定详细的执行计划：

任务：{state.input}

请列出具体的执行步骤，每个步骤应该：
1. 明确具体，可以独立执行
2. 有清晰的完成标准
3. 考虑与其他步骤的依赖关系

输出格式：
1. [步骤描述]
2. [步骤描述]
...
"""
                
                # 调用 LLM 生成计划
                plan_state = AgentState(input=prompt)
                plan_state.add_message(AgentMessage.human(prompt))
                
                plan_result = planner_llm(plan_state)
                
                # 解析计划
                last_msg = plan_result.get_last_message()
                if last_msg:
                    plan_text = last_msg.content
                    state.plan = self._parse_plan(plan_text)
                
                if not state.plan:
                    # 使用 create_plan 创建默认计划
                    default_plan = self.create_plan(
                        goal=state.input,
                        steps=[
                            "分析任务要求",
                            "收集必要信息",
                            "执行核心操作",
                            "验证结果",
                            "生成最终答案"
                        ],
                        context={"auto_generated": True}
                    )
                    state.plan = [step.description for step in default_plan.steps]
                    state.data["agent_plan"] = default_plan
                
                state.current_step = 0
                state.add_message(AgentMessage.ai(
                    f"**计划制定完成**\n\n" + 
                    "\n".join([f"{i+1}. {step}" for i, step in enumerate(state.plan)])
                ))
                
                # 创建状态检查点
                try:
                    checkpoint = self.create_state_checkpoint(state, "plan_created")
                    self.logger.info(f"Plan checkpoint created: {checkpoint.checkpoint_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create plan checkpoint: {e}")
            
            return state
        
        # 执行节点 - 使用增强的工具方法
        def executor(state: AgentState) -> AgentState:
            """执行当前步骤 - 使用增强方法"""
            if state.current_step >= len(state.plan):
                return state
            
            step = state.plan[state.current_step]
            step_num = state.current_step + 1
            
            # 创建执行事件
            exec_event = self.create_execution_event(
                event_type="step_execute",
                node_name="executor",
                data={"step_num": step_num, "total": len(state.plan), "step": step}
            )
            
            self._emit_event(
                AgentEventType.STATE_UPDATE,
                {"phase": "executing", "step": step_num, "total": len(state.plan)}
            )
            
            state.add_message(AgentMessage.ai(f"**执行步骤 {step_num}/{len(state.plan)}**：{step}"))
            
            try:
                # 构建执行提示
                exec_prompt = f"""当前任务：{state.input}

执行计划中的步骤 {step_num}：{step}

已完成的步骤：
{self._format_completed_steps(state)}

请执行这个步骤，使用可用的工具完成任务。完成后说明结果。"""
                
                # 创建执行状态
                exec_state = AgentState(input=exec_prompt)
                exec_state.add_message(AgentMessage.human(exec_prompt))
                
                # 执行步骤（可能涉及工具调用）
                exec_result = executor_llm(exec_state)
                
                # 处理工具调用 - 使用增强的工具方法
                while exec_result.pending_tool_calls:
                    # 获取工具状态
                    for tc in exec_result.pending_tool_calls:
                        tool_status = self.get_tool_status(tc.name)
                        if tool_status != ToolStatus.READY:
                            self.logger.warning(f"Tool {tc.name} not ready: {tool_status}")
                    
                    # 使用 create_node 创建工具节点
                    try:
                        tool_node = self.create_node(
                            NodeType.TOOL, "step_tools",
                            config={"registry": self._tool_registry}
                        )
                    except Exception:
                        tool_node = ToolNode("tools", registry=self._tool_registry)
                    
                    exec_result = tool_node(exec_result)
                    
                    # 收集工具指标
                    for tr in exec_result.tool_results:
                        try:
                            metrics = self.get_tool_metrics_by_name(tr.name)
                            self.logger.debug(f"Tool {tr.name} executed, total calls: {metrics.total_calls}")
                        except Exception:
                            pass
                    
                    # 继续 LLM 处理
                    exec_result = executor_llm(exec_result)
                
                # 获取执行结果
                result_msg = exec_result.get_last_message()
                result_text = result_msg.content if result_msg else "步骤执行完成"
                
                state.step_results[state.current_step] = result_text
                state.current_step += 1
                
                state.add_message(AgentMessage.ai(f"**步骤 {step_num} 结果**：{result_text}"))
                
            except Exception as e:
                self.logger.error(f"Step {step_num} execution failed: {e}")
                state.step_results[state.current_step] = f"执行失败: {e}"
                state.error = str(e)
            
            return state
        
        # 评估节点
        def evaluator(state: AgentState) -> AgentState:
            """评估执行结果"""
            self._emit_event(
                AgentEventType.STATE_UPDATE,
                {"phase": "evaluating"}
            )
            
            if state.current_step >= len(state.plan) and not state.error:
                # 所有步骤完成，生成总结
                summary_parts = ["**计划执行完成**\n"]
                
                for i, step in enumerate(state.plan):
                    result = state.step_results.get(i, "未执行")
                    summary_parts.append(f"**步骤 {i+1}**: {step}")
                    summary_parts.append(f"  结果: {result}\n")
                
                summary_parts.append("\n**最终答案**：")
                summary_parts.append(f"根据以上执行结果，任务「{state.input}」已完成。")
                
                final_summary = "\n".join(summary_parts)
                state.set_final_answer(final_summary)
            
            return state
        
        # 路由函数
        def route_after_planner(state: AgentState) -> str:
            """规划后路由"""
            if state.plan and state.current_step < len(state.plan):
                return "executor"
            return "evaluator"
        
        def route_after_executor(state: AgentState) -> str:
            """执行后路由"""
            if state.error and self.enable_replanning and self._replan_count < self.max_replans:
                return "planner"
            if state.current_step < len(state.plan):
                return "executor"
            return "evaluator"
        
        def route_after_evaluator(state: AgentState) -> str:
            """评估后路由"""
            if state.final_answer:
                return "__end__"
            if state.error and self.enable_replanning and self._replan_count < self.max_replans:
                return "planner"
            return "__end__"
        
        # 添加节点
        graph.add_node("planner", planner)
        graph.add_node("executor", executor)
        graph.add_node("evaluator", evaluator)
        
        # 添加工具节点
        tool_node = ToolNode("tools", registry=self._tool_registry)
        graph.add_node("tools", tool_node)
        
        # 设置入口点
        graph.set_entry_point("planner")
        
        # 添加边
        graph.add_conditional_edges("planner", route_after_planner, {
            "executor": "executor",
            "evaluator": "evaluator"
        })
        
        graph.add_conditional_edges("executor", route_after_executor, {
            "planner": "planner",
            "executor": "executor",
            "evaluator": "evaluator"
        })
        
        graph.add_conditional_edges("evaluator", route_after_evaluator, {
            "planner": "planner",
            "__end__": "__end__"
        })
        
        return graph
    
    def _parse_plan(self, plan_text: str) -> List[str]:
        """解析计划文本"""
        lines = plan_text.strip().split('\n')
        plan = []
        
        for line in lines:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 移除数字前缀
            import re
            match = re.match(r'^[\d]+[.、)]\s*(.+)$', line)
            if match:
                step = match.group(1).strip()
                if step and len(step) > 2:  # 过滤太短的内容
                    plan.append(step)
        
        return plan
    
    def _format_completed_steps(self, state: AgentState) -> str:
        """格式化已完成的步骤"""
        if not state.step_results:
            return "无"
        
        lines = []
        for i in sorted(state.step_results.keys()):
            step_name = state.plan[i] if i < len(state.plan) else f"步骤{i+1}"
            result = state.step_results[i]
            lines.append(f"- 步骤{i+1} ({step_name}): {result}")
        
        return "\n".join(lines) if lines else "无"
    
    def get_progress(self, state: AgentState) -> Dict[str, Any]:
        """获取执行进度"""
        total_steps = len(state.plan) if state.plan else 0
        completed_steps = len(state.step_results)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "current_step": state.current_step,
            "progress_percent": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "replan_count": self._replan_count,
            "has_error": bool(state.error)
        }


class ReflexionAgent(BaseAgent):
    """反思 Agent - 生产级实现
    
    通过自我反思和迭代改进来提高输出质量：
    
    工作流程：
    1. **生成**：产生初始响应
    2. **评估**：评价响应质量（多维度）
    3. **反思**：分析不足之处
    4. **改进**：根据反思重新生成
    5. **重复**：直到满足质量标准或达到最大次数
    
    特性：
    - 多维度质量评估
    - 结构化反思生成
    - 渐进式改进
    - 评估历史追踪
    
    使用示例：
        agent = ReflexionAgent(
            config=AgentConfig(name="writer"),
            quality_threshold=0.85,
            max_reflections=3
        )
        result = agent.invoke("写一篇关于人工智能的文章")
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 max_reflections: int = 3,
                 quality_threshold: float = 0.8,
                 evaluation_criteria: List[str] = None):
        """初始化反思 Agent
        
        Args:
            config: Agent 配置
            tools: 可用工具列表
            llm_client: LLM 客户端
            checkpointer: 检查点管理器
            callbacks: 回调列表
            max_reflections: 最大反思次数
            quality_threshold: 质量阈值（0-1）
            evaluation_criteria: 评估标准列表
        """
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.max_reflections = max_reflections
        self.quality_threshold = quality_threshold
        self.evaluation_criteria = evaluation_criteria or [
            "准确性", "完整性", "清晰度", "相关性", "逻辑性"
        ]
    
    def _build_graph(self) -> StateGraph:
        """构建反思图"""
        graph = StateGraph(config=GraphConfig(
            name=f"reflexion_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 获取 LLM 提供者
        provider = LLMProvider.OPENAI
        if self.config.provider:
            try:
                provider = LLMProvider(self.config.provider)
            except ValueError:
                pass
        
        # 生成器 LLM
        generator_llm = LLMNode(
            name="generator_llm",
            llm_client=self.llm_client,
            provider=provider,
            system_prompt=self._generator_prompt(),
            tools=self.tools,
            temperature=self.config.temperature,
            model=self.config.model
        )
        
        # 评估器 LLM
        evaluator_llm = LLMNode(
            name="evaluator_llm",
            llm_client=self.llm_client,
            provider=provider,
            system_prompt=self._evaluator_prompt(),
            tools=[],
            temperature=0.2,
            model=self.config.model
        )
        
        # 反思器 LLM
        reflector_llm = LLMNode(
            name="reflector_llm",
            llm_client=self.llm_client,
            provider=provider,
            system_prompt=self._reflector_prompt(),
            tools=[],
            temperature=0.3,
            model=self.config.model
        )
        
        # 生成节点
        def generator(state: AgentState) -> AgentState:
            """生成响应"""
            self._emit_event(
                AgentEventType.STATE_UPDATE,
                {"phase": "generating", "attempt": len(state.reflections) + 1}
            )
            
            # 构建提示
            if state.reflections:
                # 有反思历史，需要改进
                prompt = f"""原始任务：{state.input}

之前的响应存在以下问题：
{state.reflections[-1]}

请根据反思意见改进你的回答："""
            else:
                prompt = state.input
            
            # 调用 LLM 生成
            gen_state = AgentState(input=prompt)
            gen_state.add_message(AgentMessage.human(prompt))
            
            gen_result = generator_llm(gen_state)
            
            # 获取生成结果
            last_msg = gen_result.get_last_message()
            response = last_msg.content if last_msg else f"回答第 {len(state.reflections) + 1} 次尝试"
            
            state.add_message(AgentMessage.ai(f"**第 {len(state.reflections) + 1} 次生成**\n\n{response}"))
            state.output = response
            state.data['current_response'] = response
            
            return state
        
        # 评估节点
        def evaluator(state: AgentState) -> AgentState:
            """评估响应质量"""
            self._emit_event(AgentEventType.STATE_UPDATE, {"phase": "evaluating"})
            
            current_response = state.data.get('current_response', state.output)
            
            # 构建评估提示
            eval_prompt = f"""请评估以下回答的质量：

原始问题：{state.input}

回答内容：
{current_response}

请从以下维度评分（0-1）：
{chr(10).join([f'- {c}' for c in self.evaluation_criteria])}

输出格式：
维度: 分数
...
总分: X.XX"""
            
            # 调用 LLM 评估
            eval_state = AgentState(input=eval_prompt)
            eval_state.add_message(AgentMessage.human(eval_prompt))
            
            eval_result = evaluator_llm(eval_state)
            
            # 解析评估结果
            last_msg = eval_result.get_last_message()
            eval_text = last_msg.content if last_msg else ""
            
            score = self._parse_evaluation_score(eval_text)
            
            state.data['evaluation_score'] = score
            state.data['evaluation_details'] = eval_text
            state.data.setdefault('evaluation_history', []).append({
                'attempt': len(state.reflections) + 1,
                'score': score,
                'details': eval_text
            })
            
            state.add_message(AgentMessage.system(f"**评估结果** - 总分: {score:.2f}"))
            
            return state
        
        # 反思节点 - 使用 create_reflection 方法
        def reflector(state: AgentState) -> AgentState:
            """生成反思 - 使用增强方法"""
            self._emit_event(AgentEventType.STATE_UPDATE, {"phase": "reflecting"})
            
            current_response = state.data.get('current_response', state.output)
            eval_details = state.data.get('evaluation_details', '')
            score = state.data.get('evaluation_score', 0)
            
            # 创建执行事件
            reflect_event = self.create_execution_event(
                event_type="reflection_start",
                node_name="reflector",
                data={"score": score, "attempt": len(state.reflections) + 1}
            )
            self.logger.debug(f"Reflection event: {reflect_event.event_type}")
            
            # 构建反思提示
            reflect_prompt = f"""请分析以下回答的不足之处并提供改进建议：

原始问题：{state.input}

当前回答：
{current_response}

评估结果（总分 {score:.2f}）：
{eval_details}

请提供具体的改进建议，指出：
1. 主要问题是什么
2. 应该如何改进
3. 需要补充什么内容"""
            
            # 调用 LLM 反思
            reflect_state = AgentState(input=reflect_prompt)
            reflect_state.add_message(AgentMessage.human(reflect_prompt))
            
            reflect_result = reflector_llm(reflect_state)
            
            # 获取反思内容
            last_msg = reflect_result.get_last_message()
            reflection_content = last_msg.content if last_msg else f"需要改进（分数 {score:.2f}）"
            
            # 使用 create_reflection 创建反思对象
            reflection_obj = self.create_reflection(
                content=reflection_content,
                quality_score=score,
                suggestions=self._extract_suggestions(reflection_content)
            )
            
            # 存储反思对象到状态数据
            state.data.setdefault('reflection_objects', []).append({
                'id': reflection_obj.reflection_id,
                'content': reflection_obj.content,
                'score': reflection_obj.quality_score,
                'suggestions': reflection_obj.suggestions,
                'created_at': reflection_obj.created_at.isoformat()
            })
            
            state.reflections.append(reflection_content)
            state.add_message(AgentMessage.system(f"**反思 #{len(state.reflections)}**\n\n{reflection_content}"))
            
            # 创建检查点用于调试
            try:
                checkpoint = self.create_state_checkpoint(state, f"reflection_{len(state.reflections)}")
                self.logger.debug(f"Reflection checkpoint: {checkpoint.checkpoint_id}")
            except Exception as e:
                self.logger.warning(f"Failed to create reflection checkpoint: {e}")
            
            return state
        
        def _extract_suggestions(self, content: str) -> List[str]:
            """从反思内容中提取建议"""
            suggestions = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('1.', '2.', '3.', '-', '•', '*')):
                    suggestion = line.lstrip('0123456789.-•* ')
                    if suggestion:
                        suggestions.append(suggestion)
            return suggestions[:5]  # 最多5条建议
        
        self._extract_suggestions = _extract_suggestions
        
        # 路由函数
        def route_after_evaluation(state: AgentState) -> str:
            """评估后路由"""
            score = state.data.get('evaluation_score', 0)
            reflection_count = len(state.reflections)
            
            # 达到质量标准
            if score >= self.quality_threshold:
                state.set_final_answer(state.output)
                return "__end__"
            
            # 达到最大反思次数
            if reflection_count >= self.max_reflections:
                self.logger.warning(
                    f"Max reflections ({self.max_reflections}) reached with score {score:.2f}"
                )
                state.set_final_answer(state.output)
                return "__end__"
            
            return "reflector"
        
        # 添加节点
        graph.add_node("generator", generator)
        graph.add_node("evaluator", evaluator)
        graph.add_node("reflector", reflector)
        
        # 设置入口点
        graph.set_entry_point("generator")
        
        # 添加边
        graph.add_edge("generator", "evaluator")
        
        graph.add_conditional_edges("evaluator", route_after_evaluation, {
            "reflector": "reflector",
            "__end__": "__end__"
        })
        
        graph.add_edge("reflector", "generator")
        
        return graph
    
    def _generator_prompt(self) -> str:
        """生成器系统提示"""
        return """你是一个高质量内容生成助手。
请根据用户的要求生成准确、完整、清晰的回答。
如果收到改进建议，请认真考虑并改进你的回答。"""
    
    def _evaluator_prompt(self) -> str:
        """评估器系统提示"""
        return f"""你是一个严格的质量评估专家。
请客观评估回答的质量，评估维度包括：
{', '.join(self.evaluation_criteria)}

每个维度评分 0-1，并计算总分（平均值）。
请提供具体的评估理由。"""
    
    def _reflector_prompt(self) -> str:
        """反思器系统提示"""
        return """你是一个批判性思维专家。
请分析回答的不足之处，提供具体、可操作的改进建议。
要指出问题所在，并说明如何改进。"""
    
    def _parse_evaluation_score(self, eval_text: str) -> float:
        """解析评估分数"""
        import re
        
        # 尝试匹配 "总分: X.XX" 格式
        match = re.search(r'总分[：:]\s*(\d+\.?\d*)', eval_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 尝试匹配所有分数并计算平均
        scores = re.findall(r'[：:]\s*(\d+\.?\d*)', eval_text)
        if scores:
            try:
                valid_scores = [float(s) for s in scores if 0 <= float(s) <= 1]
                if valid_scores:
                    return sum(valid_scores) / len(valid_scores)
            except ValueError:
                pass
        
        # 默认分数基于反思次数
        return 0.5
    
    def get_evaluation_history(self, state: AgentState) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return state.data.get('evaluation_history', [])


@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str
    description: str
    system_prompt: str
    tools: List[Tool] = field(default_factory=list)


class MultiAgentSystem(BaseAgent):
    """多 Agent 系统
    
    多个 Agent 协作完成任务：
    - Supervisor: 协调和分配任务
    - Workers: 执行具体任务的专业 Agent
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 roles: List[AgentRole] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 collaboration_mode: str = "supervisor"):  # supervisor, round_robin, hierarchical
        
        tools = []  # 多 Agent 系统本身不直接使用工具
        super().__init__(config, tools, llm_client, checkpointer)
        
        self.roles = roles or []
        self.collaboration_mode = collaboration_mode
        self._workers: Dict[str, BaseAgent] = {}
        
        # 为每个角色创建 Agent
        for role in self.roles:
            worker_config = AgentConfig(
                name=role.name,
                system_prompt=role.system_prompt
            )
            worker = ReActAgent(
                config=worker_config,
                tools=role.tools,
                llm_client=llm_client
            )
            self._workers[role.name] = worker
    
    def _build_graph(self) -> StateGraph:
        """构建多 Agent 图"""
        graph = StateGraph(config=GraphConfig(
            name=f"multi_agent_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        if self.collaboration_mode == "supervisor":
            return self._build_supervisor_graph(graph)
        elif self.collaboration_mode == "round_robin":
            return self._build_round_robin_graph(graph)
        else:
            return self._build_supervisor_graph(graph)
    
    def _build_supervisor_graph(self, graph: StateGraph) -> StateGraph:
        """构建 Supervisor 模式图"""
        
        # Supervisor 节点
        def supervisor(state: AgentState) -> AgentState:
            """Supervisor 决定下一个执行的 Worker"""
            # 分析任务，选择合适的 Worker
            if not state.data.get('selected_worker'):
                # 简单选择：按顺序
                workers = list(self._workers.keys())
                if workers:
                    state.data['selected_worker'] = workers[0]
                    state.data['worker_index'] = 0
            
            return state
        
        # Worker 执行节点
        def worker_executor(state: AgentState) -> AgentState:
            """执行选定的 Worker"""
            worker_name = state.data.get('selected_worker')
            
            if worker_name and worker_name in self._workers:
                worker = self._workers[worker_name]
                
                # 执行 Worker
                worker_state = worker.invoke(state.input)
                
                # 合并结果
                state.add_messages(worker_state.messages[-2:])  # 添加 Worker 的最后两条消息
                state.data[f'{worker_name}_result'] = worker_state.output
                
                # 移到下一个 Worker 或完成
                worker_index = state.data.get('worker_index', 0)
                workers = list(self._workers.keys())
                
                if worker_index + 1 < len(workers):
                    state.data['worker_index'] = worker_index + 1
                    state.data['selected_worker'] = workers[worker_index + 1]
                else:
                    state.data['all_workers_done'] = True
            
            return state
        
        # 聚合节点
        def aggregator(state: AgentState) -> AgentState:
            """聚合所有 Worker 的结果"""
            results = []
            for worker_name in self._workers:
                result = state.data.get(f'{worker_name}_result')
                if result:
                    results.append(f"{worker_name}: {result}")
            
            final_answer = "各 Agent 执行结果：\n" + "\n".join(results)
            state.set_final_answer(final_answer)
            
            return state
        
        # 路由函数
        def route_supervisor(state: AgentState) -> str:
            if state.data.get('all_workers_done'):
                return "aggregator"
            if state.data.get('selected_worker'):
                return "worker"
            return "__end__"
        
        def route_worker(state: AgentState) -> str:
            if state.data.get('all_workers_done'):
                return "aggregator"
            return "supervisor"
        
        # 添加节点
        graph.add_node("supervisor", supervisor)
        graph.add_node("worker", worker_executor)
        graph.add_node("aggregator", aggregator)
        
        # 设置入口点
        graph.set_entry_point("supervisor")
        
        # 添加边
        graph.add_conditional_edges("supervisor", route_supervisor, {
            "worker": "worker",
            "aggregator": "aggregator",
            "__end__": "__end__"
        })
        
        graph.add_conditional_edges("worker", route_worker, {
            "supervisor": "supervisor",
            "aggregator": "aggregator"
        })
        
        graph.set_finish_point("aggregator")
        
        return graph
    
    def _build_round_robin_graph(self, graph: StateGraph) -> StateGraph:
        """构建轮询模式图"""
        # 简化版：依次执行每个 Worker
        
        prev_node = None
        for i, (name, worker) in enumerate(self._workers.items()):
            
            def create_worker_node(w, n):
                def worker_node(state: AgentState) -> AgentState:
                    worker_state = w.invoke(state.input)
                    state.add_messages(worker_state.messages[-2:])
                    state.data[f'{n}_result'] = worker_state.output
                    return state
                return worker_node
            
            node_name = f"worker_{name}"
            graph.add_node(node_name, create_worker_node(worker, name))
            
            if prev_node:
                graph.add_edge(prev_node, node_name)
            else:
                graph.set_entry_point(node_name)
            
            prev_node = node_name
        
        # 添加聚合节点
        def final_aggregator(state: AgentState) -> AgentState:
            results = [
                f"{name}: {state.data.get(f'{name}_result', 'N/A')}"
                for name in self._workers
            ]
            state.set_final_answer("\n".join(results))
            return state
        
        graph.add_node("aggregator", final_aggregator)
        if prev_node:
            graph.add_edge(prev_node, "aggregator")
        graph.set_finish_point("aggregator")
        
        return graph
    
    def add_worker(self, role: AgentRole) -> None:
        """添加 Worker"""
        self.roles.append(role)
        worker_config = AgentConfig(
            name=role.name,
            system_prompt=role.system_prompt
        )
        worker = ReActAgent(
            config=worker_config,
            tools=role.tools,
            llm_client=self.llm_client
        )
        self._workers[role.name] = worker
        self._graph = None  # 需要重新编译
    
    def get_worker(self, name: str) -> Optional[BaseAgent]:
        """获取 Worker"""
        return self._workers.get(name)

    def get_all_workers(self) -> Dict[str, BaseAgent]:
        """获取所有 Worker"""
        return self._workers.copy()


# ==================== 扩展 Agent 类型 ====================

class ToolCallingAgent(BaseAgent):
    """工具调用 Agent - 轻量级专用版本
    
    专注于高效的工具调用，适用于：
    - 单次工具调用任务
    - API 集成
    - 自动化脚本
    
    特性：
    - 快速工具选择
    - 最小化 LLM 调用
    - 支持批量工具执行
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 auto_execute: bool = True):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.auto_execute = auto_execute
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        tool_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools]) if self.tools else "无可用工具"
        return f"""你是一个高效的工具调用助手。

可用工具：
{tool_list}

任务：分析用户请求，选择合适的工具执行。
- 直接调用工具，避免多余的解释
- 如果需要多个工具，按顺序调用
- 工具执行完成后，简洁地总结结果"""
    
    def _build_graph(self) -> StateGraph:
        """构建工具调用图 - 使用增强方法"""
        graph = StateGraph(config=GraphConfig(
            name=f"tool_calling_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 创建执行上下文
        exec_context = self.create_execution_context(
            priority=PriorityLevel.NORMAL,
            execution_mode=StateExecutionMode.SYNC,
            timeout=self.config.tool_timeout,
            metadata={"agent_type": "tool_calling", "auto_execute": self.auto_execute}
        )
        
        # 使用 create_node 创建 LLM 节点
        try:
            llm_node = self.create_node(
                NodeType.LLM, "llm",
                config={
                    "llm_client": self.llm_client,
                    "system_prompt": self.config.system_prompt,
                    "tools": self.tools,
                    "temperature": 0.1,
                    "model": self.config.model
                }
            )
        except Exception:
            llm_node = LLMNode(
                name="llm",
                llm_client=self.llm_client,
                system_prompt=self.config.system_prompt,
                tools=self.tools,
                temperature=0.1,
                model=self.config.model
            )
        graph.add_node("llm", llm_node)
        
        # 使用 create_node 创建工具节点
        try:
            tool_node = self.create_node(
                NodeType.TOOL, "tools",
                config={
                    "registry": self._tool_registry,
                    "parallel": self.config.parallel_tool_execution
                }
            )
        except Exception:
            tool_node = ToolNode(
                name="tools",
                registry=self._tool_registry,
                parallel=self.config.parallel_tool_execution
            )
        graph.add_node("tools", tool_node)
        
        # 设置入口点
        graph.set_entry_point("llm")
        
        # 路由 - 使用增强方法
        def route(state: AgentState) -> str:
            if state.pending_tool_calls:
                # 检查所有待调用工具的状态
                for tc in state.pending_tool_calls:
                    tool_status = self.get_tool_status(tc.name)
                    if tool_status != ToolStatus.READY:
                        self.logger.warning(f"Tool {tc.name} status: {tool_status}")
                    # 获取工具信息
                    tool_info = self.get_builtin_tool_info(tc.name)
                    if tool_info:
                        self.logger.debug(f"Tool info for {tc.name}: {tool_info.get('category', 'unknown')}")
                return "tools"
            if state.final_answer:
                return "__end__"
            last_msg = state.get_last_message()
            if last_msg and last_msg.type == MessageType.AI:
                state.set_final_answer(last_msg.content)
            return "__end__"
        
        # 使用 create_conditional_edge 创建条件边
        cond_edge = self.create_conditional_edge(
            "llm", route,
            {"tools": "tools", "__end__": "__end__"},
            EdgeType.CONDITIONAL
        )
        graph.add_conditional_edges("llm", route, cond_edge.branches)
        
        graph.add_edge("tools", "llm")
        
        return graph
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """直接调用工具 - 使用增强方法"""
        # 检查工具状态
        tool_status = self.get_tool_status(tool_name)
        if tool_status != ToolStatus.READY:
            self.logger.warning(f"Tool {tool_name} status is {tool_status}, may not be ready")
        
        tool = self._tool_registry.get_tool(tool_name)
        if not tool:
            # 尝试从内置工具获取
            tool = self.get_builtin_tool(tool_name)
            if tool:
                self.add_tool(tool)
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # 执行工具并收集指标
        result = tool.invoke(kwargs)
        
        # 获取工具指标
        try:
            metrics = self.get_tool_metrics_by_name(tool_name)
            self.logger.debug(f"Tool {tool_name} metrics after call: {metrics.total_calls}")
        except Exception:
            pass
        
        return result
    
    def call_tools_batch(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """批量调用工具
        
        Args:
            tool_calls: 工具调用列表，格式为 [{"name": "tool_name", "args": {...}}, ...]
        
        Returns:
            结果列表
        """
        results = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})
            try:
                result = self.call_tool(name, **args)
                results.append({"name": name, "success": True, "result": result})
            except Exception as e:
                results.append({"name": name, "success": False, "error": str(e)})
        return results


class ConversationalAgent(BaseAgent):
    """对话型 Agent - 多轮对话专用
    
    适用于：
    - 客服对话
    - 交互式助手
    - 多轮问答
    
    特性：
    - 上下文管理
    - 对话历史压缩
    - 情感分析（可选）
    - 对话状态追踪
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 max_history_length: int = 20,
                 enable_summarization: bool = True):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.max_history_length = max_history_length
        self.enable_summarization = enable_summarization
        self._conversation_summary = ""
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """你是一个友好的对话助手。

对话原则：
- 保持对话连贯性，记住之前的上下文
- 回答要简洁、有帮助
- 适时确认理解是否正确
- 在不确定时主动询问

如果对话变长，请注意参考历史摘要。"""
    
    def _build_graph(self) -> StateGraph:
        """构建对话图"""
        graph = StateGraph(config=GraphConfig(
            name=f"conversational_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 上下文管理节点
        def context_manager(state: AgentState) -> AgentState:
            """管理对话上下文"""
            # 检查是否需要压缩历史
            if len(state.messages) > self.max_history_length and self.enable_summarization:
                # 生成摘要（简化版本）
                old_messages = state.messages[:-self.max_history_length//2]
                summary_text = f"[对话摘要] 之前讨论了: "
                topics = set()
                for msg in old_messages:
                    if msg.type == MessageType.HUMAN:
                        words = msg.content[:50]
                        topics.add(words)
                summary_text += ", ".join(list(topics)[:3])
                
                self._conversation_summary = summary_text
                state.data['conversation_summary'] = summary_text
                
                # 保留最近的消息
                state.messages = state.messages[-self.max_history_length//2:]
            
            return state
        
        # LLM 节点
        llm_node = LLMNode(
            name="llm",
            llm_client=self.llm_client,
            system_prompt=self.config.system_prompt,
            tools=self.tools,
            temperature=self.config.temperature,
            model=self.config.model
        )
        
        # 响应节点
        def responder(state: AgentState) -> AgentState:
            """生成对话响应"""
            # 添加摘要到上下文
            if self._conversation_summary:
                state.add_message(AgentMessage.system(self._conversation_summary))
            
            # 调用 LLM
            state = llm_node(state)
            
            # 标记为已响应（对话模式不设置 final_answer，保持对话继续）
            state.data['responded'] = True
            
            return state
        
        # 工具节点
        tool_node = ToolNode("tools", registry=self._tool_registry)
        
        # 添加节点
        graph.add_node("context_manager", context_manager)
        graph.add_node("responder", responder)
        graph.add_node("tools", tool_node)
        
        # 设置流程
        graph.set_entry_point("context_manager")
        graph.add_edge("context_manager", "responder")
        
        def route_after_response(state: AgentState) -> str:
            if state.pending_tool_calls:
                return "tools"
            return "__end__"
        
        graph.add_conditional_edges("responder", route_after_response, {
            "tools": "tools",
            "__end__": "__end__"
        })
        
        graph.add_edge("tools", "responder")
        
        return graph
    
    def chat(self, message: str, thread_id: str = None) -> str:
        """简化的对话接口 - 使用增强方法"""
        state = AgentState(
            input=message,
            thread_id=thread_id or str(uuid.uuid4())
        )
        state.add_message(AgentMessage.human(message))
        
        # 恢复历史 - 使用增强的检查点方法
        if thread_id:
            try:
                # 尝试从检查点恢复
                history = self.checkpointer.get(thread_id)
                if history:
                    state = history
                    state.add_message(AgentMessage.human(message))
            except Exception as e:
                self.logger.warning(f"Failed to restore history: {e}")
        
        # 创建执行事件
        chat_event = self.create_execution_event(
            event_type="chat_start",
            node_name="conversational",
            data={"thread_id": state.thread_id, "message_length": len(message)}
        )
        self.logger.debug(f"Chat event: {chat_event.event_type}")
        
        # 执行
        result = self.invoke(state)
        
        # 保存状态 - 使用增强的检查点方法
        try:
            checkpoint = self.create_state_checkpoint(result, f"chat_{state.thread_id}")
            self.checkpointer.save(result, checkpoint.checkpoint_id)
            self.logger.debug(f"Saved chat checkpoint: {checkpoint.checkpoint_id}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
            try:
                self.save_checkpoint(result, result.thread_id)
            except Exception:
                pass
        
        # 返回响应
        last_ai = None
        for msg in reversed(result.messages):
            if msg.type == MessageType.AI:
                last_ai = msg.content
                break
        
        return last_ai or ""
    
    def chat_with_context(self, message: str, thread_id: str,
                         context: Dict[str, Any] = None) -> str:
        """带上下文的对话
        
        Args:
            message: 用户消息
            thread_id: 会话 ID
            context: 额外上下文信息
        
        Returns:
            AI 响应
        """
        # 创建执行上下文
        exec_context = self.create_execution_context(
            priority=PriorityLevel.NORMAL,
            execution_mode=StateExecutionMode.ASYNC,
            metadata={"context": context or {}}
        )
        
        state = AgentState(input=message, thread_id=thread_id)
        state.add_message(AgentMessage.human(message))
        
        if context:
            state.data["user_context"] = context
            state.add_message(AgentMessage.system(f"上下文信息: {json.dumps(context, ensure_ascii=False)}"))
        
        result = self.invoke(state)
        
        last_ai = None
        for msg in reversed(result.messages):
            if msg.type == MessageType.AI:
                last_ai = msg.content
                break
        
        return last_ai or ""
    
    def get_conversation_summary(self, thread_id: str = None) -> str:
        """获取对话摘要"""
        return self._conversation_summary
    
    def clear_history(self, thread_id: str) -> None:
        """清除对话历史"""
        self.checkpointer.clear(thread_id)
        self._conversation_summary = ""
        self.logger.info(f"Cleared history for thread: {thread_id}")


class ChainOfThoughtAgent(BaseAgent):
    """思维链 Agent - 显式推理
    
    通过显式的思维链推理解决复杂问题：
    
    特性：
    - 步骤化推理
    - 推理过程可视化
    - 支持推理验证
    - 自动推理分解
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 show_reasoning: bool = True,
                 verify_steps: bool = False):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.show_reasoning = show_reasoning
        self.verify_steps = verify_steps
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """你是一个逻辑推理专家，使用思维链方法解决问题。

对于每个问题，请：

1. **理解问题**
   - 明确问题的核心
   - 识别已知信息
   - 确定需要找出什么

2. **分步推理**
   - 将问题分解为小步骤
   - 每一步都要有明确的推理
   - 标注每步的依据

3. **验证结果**
   - 检查推理的逻辑性
   - 验证答案的合理性

格式：
思考步骤 1: [推理内容]
思考步骤 2: [推理内容]
...
因此，答案是: [最终答案]"""
    
    def _build_graph(self) -> StateGraph:
        """构建思维链图"""
        graph = StateGraph(config=GraphConfig(
            name=f"cot_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # LLM 节点
        llm_node = LLMNode(
            name="reasoner",
            llm_client=self.llm_client,
            system_prompt=self.config.system_prompt,
            tools=self.tools,
            temperature=0.3,
            model=self.config.model
        )
        graph.add_node("reasoner", llm_node)
        
        # 工具节点
        tool_node = ToolNode("tools", registry=self._tool_registry)
        graph.add_node("tools", tool_node)
        
        # 验证节点（可选）
        if self.verify_steps:
            def verifier(state: AgentState) -> AgentState:
                """验证推理步骤"""
                last_msg = state.get_last_message()
                if last_msg and last_msg.type == MessageType.AI:
                    # 简单验证：检查是否有结论
                    content = last_msg.content
                    if "答案是" in content or "结论" in content or "因此" in content:
                        state.data['verified'] = True
                    else:
                        state.data['verified'] = False
                return state
            
            graph.add_node("verifier", verifier)
        
        # 设置入口点
        graph.set_entry_point("reasoner")
        
        # 路由
        def route(state: AgentState) -> str:
            if state.pending_tool_calls:
                return "tools"
            if self.verify_steps:
                return "verifier"
            last_msg = state.get_last_message()
            if last_msg and last_msg.type == MessageType.AI:
                state.set_final_answer(last_msg.content)
            return "__end__"
        
        def route_after_verify(state: AgentState) -> str:
            if state.data.get('verified', True):
                last_msg = state.get_last_message()
                if last_msg:
                    state.set_final_answer(last_msg.content)
                return "__end__"
            return "reasoner"
        
        if self.verify_steps:
            graph.add_conditional_edges("reasoner", route, {
                "tools": "tools",
                "verifier": "verifier"
            })
            graph.add_conditional_edges("verifier", route_after_verify, {
                "reasoner": "reasoner",
                "__end__": "__end__"
            })
        else:
            graph.add_conditional_edges("reasoner", route, {
                "tools": "tools",
                "__end__": "__end__"
            })
        
        graph.add_edge("tools", "reasoner")
        
        return graph
    
    def extract_reasoning_steps(self, state: AgentState) -> List[str]:
        """提取推理步骤"""
        steps = []
        for msg in state.messages:
            if msg.type == MessageType.AI:
                content = msg.content
                import re
                # 匹配 "步骤 N:" 或 "思考步骤 N:" 格式
                matches = re.findall(r'(?:思考)?步骤\s*\d+[：:]\s*(.+?)(?=(?:思考)?步骤|\n\n|$)', content, re.DOTALL)
                steps.extend([m.strip() for m in matches if m.strip()])
        return steps


class SelfAskAgent(BaseAgent):
    """自问自答 Agent - 递归分解问题
    
    通过自我提问和回答来分解和解决复杂问题：
    
    工作流程：
    1. 分析问题，确定是否需要中间问题
    2. 提出中间问题并尝试回答
    3. 使用工具获取答案（如果需要）
    4. 综合所有答案得出最终结论
    
    特性：
    - 自动问题分解
    - 递归求解
    - 答案综合
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 max_sub_questions: int = 5,
                 search_tool_name: str = None):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.max_sub_questions = max_sub_questions
        self.search_tool_name = search_tool_name
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """你是一个善于分解问题的助手，使用自问自答方法解决复杂问题。

工作方式：
1. 先判断是否可以直接回答
2. 如果不能，提出一个中间问题来帮助回答
3. 使用搜索工具查找中间问题的答案
4. 根据中间答案继续推理

输出格式：
- 如果需要中间问题：
  Follow up: [中间问题]
  
- 如果可以回答：
  So the final answer is: [最终答案]

中间答案格式：
Intermediate answer: [答案内容]"""
    
    def _build_graph(self) -> StateGraph:
        """构建自问自答图"""
        graph = StateGraph(config=GraphConfig(
            name=f"self_ask_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # LLM 节点
        llm_node = LLMNode(
            name="thinker",
            llm_client=self.llm_client,
            system_prompt=self.config.system_prompt,
            tools=self.tools,
            temperature=0.3,
            model=self.config.model
        )
        graph.add_node("thinker", llm_node)
        
        # 工具节点
        tool_node = ToolNode("tools", registry=self._tool_registry)
        graph.add_node("tools", tool_node)
        
        # 解析节点
        def parser(state: AgentState) -> AgentState:
            """解析 LLM 输出"""
            last_msg = state.get_last_message()
            if not last_msg or last_msg.type != MessageType.AI:
                return state
            
            content = last_msg.content or ""
            
            # 检查是否有最终答案
            if "final answer is:" in content.lower():
                import re
                match = re.search(r'final answer is[：:]\s*(.+?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
                if match:
                    state.set_final_answer(match.group(1).strip())
            
            # 检查是否有中间问题
            elif "follow up:" in content.lower():
                import re
                match = re.search(r'follow up[：:]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
                if match:
                    sub_question = match.group(1).strip()
                    state.data.setdefault('sub_questions', []).append(sub_question)
            
            return state
        
        graph.add_node("parser", parser)
        
        # 设置入口点
        graph.set_entry_point("thinker")
        
        # 路由
        def route(state: AgentState) -> str:
            if state.pending_tool_calls:
                return "tools"
            return "parser"
        
        def route_after_parser(state: AgentState) -> str:
            if state.final_answer:
                return "__end__"
            
            sub_questions = state.data.get('sub_questions', [])
            if len(sub_questions) >= self.max_sub_questions:
                # 强制生成答案
                state.set_final_answer("基于已有信息的综合答案")
                return "__end__"
            
            return "thinker"
        
        graph.add_conditional_edges("thinker", route, {
            "tools": "tools",
            "parser": "parser"
        })
        
        graph.add_conditional_edges("parser", route_after_parser, {
            "thinker": "thinker",
            "__end__": "__end__"
        })
        
        graph.add_edge("tools", "thinker")
        
        return graph


class HierarchicalAgent(BaseAgent):
    """层级 Agent - 任务分解与委派
    
    多层级的 Agent 架构，支持任务的分解和委派：
    
    架构：
    - Manager: 任务分解和调度
    - Workers: 执行具体子任务
    - Aggregator: 结果汇总
    
    特性：
    - 自动任务分解
    - 动态 Worker 分配
    - 结果聚合
    - 失败处理
    """
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 worker_configs: List[AgentConfig] = None,
                 max_workers: int = 5,
                 parallel_execution: bool = True):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.worker_configs = worker_configs or []
        self.max_workers = max_workers
        self.parallel_execution = parallel_execution
        self._workers: Dict[str, BaseAgent] = {}
        
        # 初始化 Workers
        self._init_workers()
        
        if self.config.system_prompt is None:
            self.config.system_prompt = self._default_system_prompt()
    
    def _init_workers(self) -> None:
        """初始化 Workers"""
        for i, worker_config in enumerate(self.worker_configs):
            worker = ReActAgent(
                config=worker_config,
                tools=self.tools,
                llm_client=self.llm_client
            )
            self._workers[worker_config.name] = worker
    
    def _default_system_prompt(self) -> str:
        workers_desc = "\n".join([
            f"- {name}: {w.config.description or '通用 Worker'}"
            for name, w in self._workers.items()
        ]) if self._workers else "暂无专门的 Worker"
        
        return f"""你是一个任务管理 Agent，负责将复杂任务分解并分配给合适的 Worker。

可用 Workers：
{workers_desc}

工作流程：
1. 分析任务，确定需要哪些子任务
2. 将子任务分配给合适的 Worker
3. 收集各 Worker 的结果
4. 综合结果给出最终答案

输出任务分解时使用格式：
TASK_DECOMPOSITION:
- task_1: [任务描述] -> worker: [worker名称]
- task_2: [任务描述] -> worker: [worker名称]
END_DECOMPOSITION"""
    
    def _build_graph(self) -> StateGraph:
        """构建层级 Agent 图"""
        graph = StateGraph(config=GraphConfig(
            name=f"hierarchical_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # Manager 节点
        manager_llm = LLMNode(
            name="manager",
            llm_client=self.llm_client,
            system_prompt=self.config.system_prompt,
            tools=[],
            temperature=0.3,
            model=self.config.model
        )
        
        def manager(state: AgentState) -> AgentState:
            """Manager: 分析和分解任务"""
            if not state.data.get('tasks_decomposed'):
                # 分解任务
                result = manager_llm(state)
                
                last_msg = result.get_last_message()
                if last_msg:
                    tasks = self._parse_task_decomposition(last_msg.content)
                    state.data['subtasks'] = tasks
                    state.data['tasks_decomposed'] = True
                    state.data['completed_tasks'] = []
            
            return state
        
        graph.add_node("manager", manager)
        
        # Worker 执行节点
        def worker_executor(state: AgentState) -> AgentState:
            """执行子任务"""
            subtasks = state.data.get('subtasks', [])
            completed = state.data.get('completed_tasks', [])
            
            # 找到下一个未完成的任务
            for task in subtasks:
                if task['id'] not in completed:
                    worker_name = task.get('worker', list(self._workers.keys())[0] if self._workers else None)
                    worker = self._workers.get(worker_name)
                    
                    if worker:
                        try:
                            result = worker.invoke(task['description'])
                            task['result'] = result.final_answer or result.output
                            task['success'] = True
                        except Exception as e:
                            task['result'] = f"执行失败: {e}"
                            task['success'] = False
                    else:
                        task['result'] = "未找到对应的 Worker"
                        task['success'] = False
                    
                    completed.append(task['id'])
                    state.data['completed_tasks'] = completed
                    break
            
            return state
        
        graph.add_node("worker_executor", worker_executor)
        
        # 聚合节点
        def aggregator(state: AgentState) -> AgentState:
            """聚合结果"""
            subtasks = state.data.get('subtasks', [])
            
            results = []
            for task in subtasks:
                results.append(f"- {task.get('description', '未知任务')}: {task.get('result', '无结果')}")
            
            summary = "任务执行结果：\n" + "\n".join(results)
            state.set_final_answer(summary)
            
            return state
        
        graph.add_node("aggregator", aggregator)
        
        # 设置入口点
        graph.set_entry_point("manager")
        
        # 路由
        def route_after_manager(state: AgentState) -> str:
            if state.data.get('subtasks'):
                return "worker_executor"
            return "aggregator"
        
        def route_after_worker(state: AgentState) -> str:
            subtasks = state.data.get('subtasks', [])
            completed = state.data.get('completed_tasks', [])
            
            if len(completed) >= len(subtasks):
                return "aggregator"
            return "worker_executor"
        
        graph.add_conditional_edges("manager", route_after_manager, {
            "worker_executor": "worker_executor",
            "aggregator": "aggregator"
        })
        
        graph.add_conditional_edges("worker_executor", route_after_worker, {
            "worker_executor": "worker_executor",
            "aggregator": "aggregator"
        })
        
        return graph
    
    def _parse_task_decomposition(self, content: str) -> List[Dict[str, Any]]:
        """解析任务分解"""
        tasks = []
        import re
        
        # 查找 TASK_DECOMPOSITION 块
        match = re.search(r'TASK_DECOMPOSITION:(.*?)(?:END_DECOMPOSITION|$)', content, re.DOTALL)
        if match:
            block = match.group(1)
            # 解析每个任务
            task_matches = re.findall(r'-\s*task_(\d+):\s*(.+?)\s*->\s*worker:\s*(\w+)', block, re.IGNORECASE)
            for task_id, desc, worker in task_matches:
                tasks.append({
                    'id': f"task_{task_id}",
                    'description': desc.strip(),
                    'worker': worker.strip()
                })
        
        # 如果没有找到格式化的分解，创建默认任务
        if not tasks:
            tasks.append({
                'id': 'task_1',
                'description': content[:200],
                'worker': list(self._workers.keys())[0] if self._workers else 'default'
            })
        
        return tasks
    
    def add_worker(self, name: str, config: AgentConfig = None) -> None:
        """添加 Worker - 使用增强方法"""
        config = config or AgentConfig(name=name)
        
        # 使用推荐的内置工具
        worker_tools = self.tools.copy()
        try:
            recommended = get_tools_for_agent("react")
            for t in recommended:
                if t.name not in [tool.name for tool in worker_tools]:
                    worker_tools.append(t)
        except Exception:
            pass
        
        worker = ReActAgent(
            config=config,
            tools=worker_tools,
            llm_client=self.llm_client
        )
        self._workers[name] = worker
        self._graph = None
        
        self.logger.info(f"Added worker '{name}' with {len(worker_tools)} tools")
    
    def get_worker_metrics(self, name: str) -> Dict[str, Any]:
        """获取 Worker 指标"""
        worker = self._workers.get(name)
        if worker:
            return worker.get_metrics()
        return {}
    
    def get_all_workers_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 Workers 的指标"""
        metrics = {}
        for name, worker in self._workers.items():
            metrics[name] = worker.get_metrics()
        return metrics


class WorkflowAgent(BaseAgent):
    """工作流 Agent - 复杂流程编排
    
    支持定义和执行复杂的工作流：
    
    特性：
    - 可视化工作流定义
    - 条件分支
    - 并行执行
    - 循环处理
    - 错误处理
    """
    
    @dataclass
    class WorkflowStep:
        """工作流步骤"""
        name: str
        action: Callable[[AgentState], AgentState]
        next_step: Optional[str] = None
        condition: Optional[Callable[[AgentState], str]] = None
        retry_count: int = 0
        timeout: float = 60.0
        on_error: Optional[str] = None
    
    def __init__(self,
                 config: AgentConfig = None,
                 tools: List[Tool] = None,
                 llm_client: Any = None,
                 checkpointer: Checkpointer = None,
                 callbacks: List[AgentCallback] = None,
                 workflow_steps: List['WorkflowAgent.WorkflowStep'] = None):
        super().__init__(config, tools, llm_client, checkpointer, callbacks)
        self.workflow_steps = workflow_steps or []
        self._steps_map: Dict[str, 'WorkflowAgent.WorkflowStep'] = {}
        
        for step in self.workflow_steps:
            self._steps_map[step.name] = step
    
    def add_step(self, step: 'WorkflowStep') -> 'WorkflowAgent':
        """添加工作流步骤"""
        self.workflow_steps.append(step)
        self._steps_map[step.name] = step
        self._graph = None
        return self
    
    def _build_graph(self) -> StateGraph:
        """构建工作流图"""
        graph = StateGraph(config=GraphConfig(
            name=f"workflow_{self.config.name}",
            max_iterations=self.config.max_iterations
        ))
        
        # 为每个步骤创建节点
        for step in self.workflow_steps:
            graph.add_node(step.name, step.action)
        
        # 设置入口点
        if self.workflow_steps:
            graph.set_entry_point(self.workflow_steps[0].name)
        
        # 添加边
        for step in self.workflow_steps:
            if step.condition:
                # 条件边
                branches = {}
                for next_step in self.workflow_steps:
                    branches[next_step.name] = next_step.name
                branches["__end__"] = "__end__"
                
                graph.add_conditional_edges(step.name, step.condition, branches)
            elif step.next_step:
                # 普通边
                graph.add_edge(step.name, step.next_step)
            else:
                # 没有下一步，设置为结束
                graph.set_finish_point(step.name)
        
        return graph
    
    @classmethod
    def from_definition(cls, definition: Dict[str, Any], 
                       llm_client: Any = None,
                       tools: List[Tool] = None) -> 'WorkflowAgent':
        """从定义创建工作流 Agent"""
        config = AgentConfig.from_dict(definition.get('config', {}))
        
        steps = []
        for step_def in definition.get('steps', []):
            step = cls.WorkflowStep(
                name=step_def['name'],
                action=step_def.get('action', lambda s: s),
                next_step=step_def.get('next_step'),
                timeout=step_def.get('timeout', 60.0),
                on_error=step_def.get('on_error')
            )
            steps.append(step)
        
        return cls(
            config=config,
            tools=tools,
            llm_client=llm_client,
            workflow_steps=steps
        )


# ==================== Agent 编排 ====================

class AgentPool:
    """Agent 池 - 管理多个 Agent 实例
    
    特性：
    - Agent 注册和生命周期管理
    - 负载均衡
    - 健康检查
    - 自动扩缩容
    """
    
    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._health_check_interval = 30.0
    
    def register(self, name: str, agent: BaseAgent) -> bool:
        """注册 Agent"""
        with self._lock:
            if len(self._agents) >= self.max_agents:
                logger.warning(f"Agent pool is full, cannot register {name}")
                return False
            
            self._agents[name] = agent
            self._agent_stats[name] = {
                "registered_at": datetime.utcnow().isoformat(),
                "executions": 0,
                "errors": 0,
                "last_used": None
            }
            return True
    
    def unregister(self, name: str) -> Optional[BaseAgent]:
        """注销 Agent"""
        with self._lock:
            agent = self._agents.pop(name, None)
            self._agent_stats.pop(name, None)
            return agent
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """获取 Agent"""
        return self._agents.get(name)
    
    def run(self, name: str, input_data: Union[str, Dict[str, Any]], 
           **kwargs) -> AgentState:
        """运行指定 Agent"""
        agent = self.get(name)
        if not agent:
            raise ValueError(f"Agent '{name}' not found in pool")
        
        # 更新统计
        with self._lock:
            self._agent_stats[name]["executions"] += 1
            self._agent_stats[name]["last_used"] = datetime.utcnow().isoformat()
        
        try:
            return agent.invoke(input_data, **kwargs)
        except Exception as e:
            with self._lock:
                self._agent_stats[name]["errors"] += 1
            raise
    
    def run_best(self, input_data: Union[str, Dict[str, Any]], 
                agent_type: str = None, **kwargs) -> Tuple[str, AgentState]:
        """选择最佳 Agent 运行
        
        基于负载和历史性能选择。
        """
        candidates = []
        
        with self._lock:
            for name, agent in self._agents.items():
                if agent_type and not isinstance(agent, agent_type):
                    continue
                if not agent.is_running:
                    stats = self._agent_stats[name]
                    score = stats["executions"] - stats["errors"] * 2
                    candidates.append((name, score))
        
        if not candidates:
            raise RuntimeError("No available agents")
        
        # 选择得分最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name = candidates[0][0]
        
        result = self.run(best_name, input_data, **kwargs)
        return best_name, result
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有 Agent"""
        with self._lock:
            return [
                {
                    "name": name,
                    "type": type(agent).__name__,
                    "is_running": agent.is_running,
                    "stats": self._agent_stats.get(name, {})
                }
                for name, agent in self._agents.items()
            ]
    
    def health_check(self) -> Dict[str, str]:
        """健康检查"""
        results = {}
        for name, agent in self._agents.items():
            try:
                if agent.is_running:
                    results[name] = "busy"
                else:
                    results[name] = "healthy"
            except Exception:
                results[name] = "unhealthy"
        return results
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """获取 Agent 池的汇总指标
        
        Returns:
            池指标字典
        """
        total_executions = 0
        total_errors = 0
        agents_status = {}
        
        with self._lock:
            for name, stats in self._agent_stats.items():
                total_executions += stats.get("executions", 0)
                total_errors += stats.get("errors", 0)
                agents_status[name] = {
                    "executions": stats.get("executions", 0),
                    "errors": stats.get("errors", 0),
                    "error_rate": stats["errors"] / stats["executions"] if stats.get("executions", 0) > 0 else 0
                }
        
        return {
            "total_agents": len(self._agents),
            "max_agents": self.max_agents,
            "total_executions": total_executions,
            "total_errors": total_errors,
            "success_rate": (total_executions - total_errors) / total_executions if total_executions > 0 else 1.0,
            "agents": agents_status
        }
    
    def get_agent_graph_status(self, name: str) -> Dict[str, Any]:
        """获取指定 Agent 的图状态
        
        Args:
            name: Agent 名称
        
        Returns:
            图状态信息
        """
        agent = self.get(name)
        if not agent:
            return {"error": f"Agent '{name}' not found"}
        
        return {
            "name": name,
            "graph_status": agent.get_graph_status().value if hasattr(agent, 'get_graph_status') else "unknown",
            "graph_metrics": agent.get_graph_metrics().__dict__ if hasattr(agent, 'get_graph_metrics') else {}
        }
    
    def create_agent_from_builder(self, agent_type: str, **kwargs) -> str:
        """使用增强构建器创建并注册 Agent
        
        Args:
            agent_type: Agent 类型
            **kwargs: 构建器参数
        
        Returns:
            注册的 Agent 名称
        """
        from . import build_enhanced_agent
        
        builder = build_enhanced_agent(agent_type)
        
        # 应用配置
        if "name" in kwargs:
            builder.with_name(kwargs["name"])
        if "model" in kwargs:
            builder.with_model(kwargs["model"])
        if "tools" in kwargs:
            builder.with_tools(kwargs["tools"])
        if "llm_client" in kwargs:
            builder.with_llm_client(kwargs["llm_client"])
        
        agent = builder.build()
        self.register(agent.name, agent)
        return agent.name
    
    def close_all(self) -> None:
        """关闭所有 Agent"""
        for agent in self._agents.values():
            try:
                agent.close()
            except Exception as e:
                logger.error(f"Error closing agent: {e}")
        self._agents.clear()
    
    # ==================== 工业级派生方法 ====================
    
    def _get_master_factory(self):
        """延迟加载 MasterFactory 避免循环导入"""
        if not hasattr(self, '_master_factory') or self._master_factory is None:
            from .factory import get_master_factory
            self._master_factory = get_master_factory()
        return self._master_factory
    
    def create_enhanced_checkpoint_for_agent(self, agent_name: str,
                                            checkpoint_id: str = None,
                                            branch: str = "main",
                                            tags: List[str] = None) -> Optional['EnhancedCheckpoint']:
        """为指定 Agent 创建增强检查点
        
        调用 MasterFactory.create_enhanced_checkpoint
        
        Args:
            agent_name: Agent 名称
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            EnhancedCheckpoint 实例或 None
        """
        agent = self.get(agent_name)
        if not agent:
            logger.warning(f"Agent '{agent_name}' not found for checkpoint")
            return None
        
        factory = self._get_master_factory()
        cp_id = checkpoint_id or f"cp_{agent_name}_{uuid.uuid4().hex[:8]}"
        thread_id = f"thread_{agent_name}"
        
        # 获取当前状态
        state_data = {}
        if hasattr(agent, '_current_state') and agent._current_state:
            state_data = agent._current_state.__dict__ if hasattr(agent._current_state, '__dict__') else {}
        
        return factory.create_enhanced_checkpoint(
            checkpoint_id=cp_id,
            thread_id=thread_id,
            state=state_data,
            tags=tags or [agent_name],
            branch=branch
        )
    
    def create_tool_metrics_for_agent(self, agent_name: str) -> Dict[str, 'ToolMetrics']:
        """为指定 Agent 创建工具指标
        
        调用 MasterFactory.create_tool_metrics
        
        Args:
            agent_name: Agent 名称
            
        Returns:
            工具指标字典 {tool_name: ToolMetrics}
        """
        agent = self.get(agent_name)
        if not agent:
            return {}
        
        factory = self._get_master_factory()
        metrics = {}
        
        for tool in agent.tools:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            metrics[tool_name] = factory.create_tool_metrics(tool_name)
        
        return metrics
    
    def setup_tool_cache_for_agent(self, agent_name: str,
                                   max_size: int = 1000,
                                   default_ttl: int = 300) -> bool:
        """为指定 Agent 设置工具缓存
        
        调用 MasterFactory.create_tool_cache
        
        Args:
            agent_name: Agent 名称
            max_size: 缓存最大大小
            default_ttl: 默认 TTL
            
        Returns:
            是否成功
        """
        agent = self.get(agent_name)
        if not agent:
            return False
        
        factory = self._get_master_factory()
        tool_cache = factory.create_tool_cache(max_size, default_ttl)
        
        # 设置到 Agent
        if hasattr(agent, '_tool_executor') and agent._tool_executor:
            agent._tool_executor._cache = tool_cache
            return True
        return False
    
    def setup_retry_handler_for_agent(self, agent_name: str,
                                      max_retries: int = 3,
                                      base_delay: float = 1.0,
                                      max_delay: float = 60.0) -> bool:
        """为指定 Agent 设置重试处理器
        
        调用 MasterFactory.create_retry_handler
        
        Args:
            agent_name: Agent 名称
            max_retries: 最大重试次数
            base_delay: 基础延迟
            max_delay: 最大延迟
            
        Returns:
            是否成功
        """
        agent = self.get(agent_name)
        if not agent:
            return False
        
        factory = self._get_master_factory()
        retry_handler = factory.create_retry_handler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay
        )
        
        if hasattr(agent, '_tool_executor') and agent._tool_executor:
            agent._tool_executor._retry_handler = retry_handler
            return True
        return False
    
    def setup_rate_limiter_for_agent(self, agent_name: str,
                                     rate: float = 10.0,
                                     burst: int = 20) -> bool:
        """为指定 Agent 设置限流器
        
        调用 MasterFactory.create_tool_rate_limiter
        
        Args:
            agent_name: Agent 名称
            rate: 速率
            burst: 突发数
            
        Returns:
            是否成功
        """
        agent = self.get(agent_name)
        if not agent:
            return False
        
        factory = self._get_master_factory()
        rate_limiter = factory.create_tool_rate_limiter(rate, burst)
        
        if hasattr(agent, '_tool_executor') and agent._tool_executor:
            agent._tool_executor._rate_limiter = rate_limiter
            return True
        return False
    
    def get_node_metrics_for_agent(self, agent_name: str) -> Optional['NodeMetrics']:
        """获取指定 Agent 的节点指标
        
        调用 MasterFactory.create_node_metrics
        
        Args:
            agent_name: Agent 名称
            
        Returns:
            NodeMetrics 实例或 None
        """
        agent = self.get(agent_name)
        if not agent:
            return None
        
        factory = self._get_master_factory()
        return factory.create_node_metrics(agent_name)
    
    def create_graph_metrics_for_pool(self) -> 'GraphMetrics':
        """为整个池创建图指标
        
        调用 MasterFactory.create_graph_metrics
        
        Returns:
            GraphMetrics 实例
        """
        factory = self._get_master_factory()
        return factory.create_graph_metrics()
    
    def create_state_manager_for_pool(self) -> 'StateManager':
        """为整个池创建状态管理器
        
        调用 MasterFactory.create_state_manager
        
        Returns:
            StateManager 实例
        """
        factory = self._get_master_factory()
        return factory.create_state_manager()
    
    def get_pool_health_with_factory(self) -> Dict[str, Any]:
        """使用工厂获取池健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        factory = self._get_master_factory()
        factory_health = factory.health_check()
        pool_health = self.health_check()
        pool_metrics = self.get_pool_metrics()
        
        return {
            "pool_health": pool_health,
            "pool_metrics": pool_metrics,
            "factory_health": factory_health,
            "timestamp": datetime.utcnow().isoformat()
        }


class AgentOrchestrator:
    """Agent 编排器 - 协调多 Agent 执行
    
    支持：
    - 顺序执行
    - 并行执行
    - 条件执行
    - 流水线
    """
    
    def __init__(self, pool: AgentPool = None):
        self.pool = pool or AgentPool()
        self._pipelines: Dict[str, List[Dict[str, Any]]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_agent(self, agent: BaseAgent) -> None:
        """添加 Agent 到池"""
        self.pool.register(agent.name, agent)
    
    def remove_agent(self, name: str) -> None:
        """移除 Agent"""
        self.pool.unregister(name)
    
    def add_agent_with_builtin_tools(self, agent: BaseAgent, 
                                     tool_names: List[str] = None) -> None:
        """添加 Agent 并配置内置工具
        
        Args:
            agent: Agent 实例
            tool_names: 内置工具名称列表
        """
        # 添加内置工具
        if tool_names:
            added = agent.add_builtin_tools(tool_names)
            logger.info(f"Added builtin tools to {agent.name}: {added}")
        
        self.pool.register(agent.name, agent)
    
    def add_agent_with_checkpointer(self, agent: BaseAgent,
                                    checkpointer_type: str = "memory",
                                    **checkpointer_kwargs) -> None:
        """添加 Agent 并配置检查点器
        
        Args:
            agent: Agent 实例
            checkpointer_type: 检查点器类型 (memory/redis/sqlite/file)
            **checkpointer_kwargs: 检查点器参数
        """
        if checkpointer_type == "redis":
            agent.set_redis_checkpointer(**checkpointer_kwargs)
        elif checkpointer_type == "sqlite":
            agent.set_sqlite_checkpointer(**checkpointer_kwargs)
        elif checkpointer_type == "file":
            agent.set_file_checkpointer(**checkpointer_kwargs)
        # memory 是默认的
        
        self.pool.register(agent.name, agent)
    
    def define_pipeline(self, name: str, 
                       steps: List[Dict[str, Any]]) -> None:
        """定义执行流水线
        
        步骤格式：
        {
            "agent": "agent_name",
            "input": "static input" or callable,
            "condition": optional callable,
            "parallel_with": optional list of step indices
        }
        """
        self._pipelines[name] = steps
    
    def run_pipeline(self, pipeline_name: str, 
                    initial_input: Any = None) -> List[AgentState]:
        """执行流水线"""
        steps = self._pipelines.get(pipeline_name)
        if not steps:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        results = []
        current_input = initial_input
        
        for step in steps:
            agent_name = step["agent"]
            
            # 检查条件
            condition = step.get("condition")
            if condition and not condition(current_input, results):
                continue
            
            # 准备输入
            step_input = step.get("input", current_input)
            if callable(step_input):
                step_input = step_input(current_input, results)
            
            # 执行
            result = self.pool.run(agent_name, step_input)
            results.append(result)
            current_input = result.final_answer or result.output
        
        # 记录历史
        with self._lock:
            self._execution_history.append({
                "pipeline": pipeline_name,
                "timestamp": datetime.utcnow().isoformat(),
                "steps_executed": len(results)
            })
        
        return results
    
    def run_parallel(self, tasks: List[Tuple[str, Any]], 
                    max_workers: int = 4) -> List[Tuple[str, AgentState]]:
        """并行执行多个任务
        
        Args:
            tasks: [(agent_name, input_data), ...]
            max_workers: 最大并行数
            
        Returns:
            [(agent_name, result), ...]
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.pool.run, name, inp): (name, i)
                for i, (name, inp) in enumerate(tasks)
            }
            
            for future in futures:
                name, idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, name, result))
                except Exception as e:
                    logger.error(f"Parallel execution error for {name}: {e}")
                    results.append((idx, name, None))
        
        # 按原顺序排序
        results.sort(key=lambda x: x[0])
        return [(r[1], r[2]) for r in results]
    
    def run_workflow(self, workflow: List[Tuple[str, Any]], 
                    stop_on_error: bool = False) -> List[AgentState]:
        """执行工作流
        
        Args:
            workflow: [(agent_name, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            结果列表
        """
        results = []
        previous_output = None
        
        for agent_name, input_data in workflow:
            # 如果输入是 None，使用上一步的输出
            if input_data is None and previous_output:
                input_data = previous_output
            
            try:
                result = self.pool.run(agent_name, input_data)
                results.append(result)
                previous_output = result.final_answer or result.output
            except Exception as e:
                logger.error(f"Workflow step failed: {agent_name} - {e}")
                if stop_on_error:
                    raise
                results.append(None)
        
        return results
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取执行历史"""
        with self._lock:
            return self._execution_history[-limit:]
    
    # ==================== 工业级派生方法 ====================
    
    def _get_master_factory(self):
        """延迟加载 MasterFactory 避免循环导入"""
        if not hasattr(self, '_master_factory') or self._master_factory is None:
            from .factory import get_master_factory
            self._master_factory = get_master_factory()
        return self._master_factory
    
    def create_execution_event(self, event_type: str,
                              node_name: str,
                              data: Dict[str, Any] = None) -> 'ExecutionEvent':
        """创建执行事件
        
        调用 MasterFactory.create_execution_event
        
        Args:
            event_type: 事件类型
            node_name: 节点名称
            data: 事件数据
            
        Returns:
            ExecutionEvent 实例
        """
        factory = self._get_master_factory()
        return factory.create_execution_event(
            event_type=event_type,
            node_name=node_name,
            data=data
        )
    
    def create_edge_condition_for_workflow(self, 
                                          condition_func: Callable[[AgentState], bool],
                                          name: str = "workflow_condition") -> 'EdgeCondition':
        """为工作流创建边条件
        
        调用 MasterFactory.create_edge_condition
        
        Args:
            condition_func: 条件函数
            name: 条件名称
            
        Returns:
            EdgeCondition 实例
        """
        factory = self._get_master_factory()
        return factory.create_edge_condition(
            condition_func=condition_func,
            name=name
        )
    
    def create_priority_router_for_orchestration(self, name: str = "orchestrator_priority") -> 'PriorityRouter':
        """为编排创建优先级路由器
        
        调用 MasterFactory.create_priority_router
        
        Args:
            name: 路由器名称
            
        Returns:
            PriorityRouter 实例
        """
        factory = self._get_master_factory()
        return factory.create_priority_router(name)
    
    def create_weighted_router_for_orchestration(self, name: str = "orchestrator_weighted") -> 'WeightedRouter':
        """为编排创建权重路由器
        
        调用 MasterFactory.create_weighted_router
        
        Args:
            name: 路由器名称
            
        Returns:
            WeightedRouter 实例
        """
        factory = self._get_master_factory()
        return factory.create_weighted_router(name)
    
    def create_load_balance_router_for_orchestration(self, name: str = "orchestrator_lb") -> 'LoadBalanceRouter':
        """为编排创建负载均衡路由器
        
        调用 MasterFactory.create_load_balance_router
        
        Args:
            name: 路由器名称
            
        Returns:
            LoadBalanceRouter 实例
        """
        factory = self._get_master_factory()
        return factory.create_load_balance_router(name)
    
    def create_ab_test_router_for_orchestration(self, name: str = "orchestrator_ab",
                                               test_name: str = "") -> 'ABTestRouter':
        """为编排创建 A/B 测试路由器
        
        调用 MasterFactory.create_ab_test_router
        
        Args:
            name: 路由器名称
            test_name: 测试名称
            
        Returns:
            ABTestRouter 实例
        """
        factory = self._get_master_factory()
        return factory.create_ab_test_router(name, test_name)
    
    def setup_checkpoint_for_pipeline(self, pipeline_name: str) -> Optional['EnhancedCheckpoint']:
        """为流水线设置检查点
        
        调用 MasterFactory.create_enhanced_checkpoint
        
        Args:
            pipeline_name: 流水线名称
            
        Returns:
            EnhancedCheckpoint 实例或 None
        """
        if pipeline_name not in self._pipelines:
            return None
        
        factory = self._get_master_factory()
        return factory.create_enhanced_checkpoint(
            checkpoint_id=f"pipeline_{pipeline_name}_{uuid.uuid4().hex[:8]}",
            thread_id=f"pipeline_thread_{pipeline_name}",
            state={"pipeline_name": pipeline_name, "steps": len(self._pipelines[pipeline_name])},
            tags=[pipeline_name, "pipeline"],
            branch="main"
        )
    
    def create_memory_entry_for_execution(self, content: Any,
                                         importance: float = 0.5) -> 'MemoryEntry':
        """为执行创建记忆条目
        
        调用 MasterFactory.create_memory_entry
        
        Args:
            content: 记忆内容
            importance: 重要性分数
            
        Returns:
            MemoryEntry 实例
        """
        factory = self._get_master_factory()
        mem_type = factory.get_memory_type("episodic")
        return factory.create_memory_entry(
            content=content,
            memory_type=mem_type,
            importance=importance,
            tags=["orchestrator", "execution"]
        )
    
    def create_plan_step_for_workflow(self, description: str,
                                     action: str = "",
                                     dependencies: List[str] = None) -> 'PlanStep':
        """为工作流创建计划步骤
        
        调用 MasterFactory.create_plan_step
        
        Args:
            description: 步骤描述
            action: 步骤动作
            dependencies: 依赖步骤
            
        Returns:
            PlanStep 实例
        """
        factory = self._get_master_factory()
        return factory.create_plan_step(
            description=description,
            action=action,
            dependencies=dependencies
        )
    
    def get_orchestrator_health_with_factory(self) -> Dict[str, Any]:
        """使用工厂获取编排器健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        factory = self._get_master_factory()
        factory_health = factory.health_check()
        pool_health = self.pool.health_check() if self.pool else {}
        
        return {
            "pipelines_count": len(self._pipelines),
            "pipelines": list(self._pipelines.keys()),
            "execution_history_count": len(self._execution_history),
            "pool_health": pool_health,
            "factory_health": factory_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run_workflow_with_events(self, workflow: List[Tuple[str, Any]],
                                stop_on_error: bool = False) -> Tuple[List[AgentState], List['ExecutionEvent']]:
        """执行工作流并生成执行事件
        
        调用 MasterFactory.create_execution_event
        
        Args:
            workflow: 工作流定义
            stop_on_error: 遇错是否停止
            
        Returns:
            (结果列表, 事件列表)
        """
        results = []
        events = []
        previous_output = None
        
        # 开始事件
        events.append(self.create_execution_event(
            "workflow_start",
            "orchestrator",
            {"workflow_length": len(workflow)}
        ))
        
        for i, (agent_name, input_data) in enumerate(workflow):
            # 步骤开始事件
            events.append(self.create_execution_event(
                "step_start",
                agent_name,
                {"step_index": i, "input_type": type(input_data).__name__}
            ))
            
            if input_data is None and previous_output:
                input_data = previous_output
            
            try:
                result = self.pool.run(agent_name, input_data)
                results.append(result)
                previous_output = result.final_answer or result.output
                
                # 步骤完成事件
                events.append(self.create_execution_event(
                    "step_complete",
                    agent_name,
                    {"step_index": i, "success": True}
                ))
            except Exception as e:
                logger.error(f"Workflow step failed: {agent_name} - {e}")
                
                # 步骤失败事件
                events.append(self.create_execution_event(
                    "step_error",
                    agent_name,
                    {"step_index": i, "error": str(e)}
                ))
                
                if stop_on_error:
                    raise
                results.append(None)
        
        # 完成事件
        events.append(self.create_execution_event(
            "workflow_complete",
            "orchestrator",
            {"total_steps": len(workflow), "successful_steps": sum(1 for r in results if r is not None)}
        ))
        
        return results, events
    
    def run_parallel_with_load_balance(self, tasks: List[Tuple[str, Any]],
                                       max_workers: int = 4) -> List[Tuple[str, AgentState]]:
        """使用负载均衡并行执行任务
        
        调用 MasterFactory.create_load_balance_router
        
        Args:
            tasks: 任务列表
            max_workers: 最大并行数
            
        Returns:
            结果列表
        """
        # 创建负载均衡路由器
        lb_router = self.create_load_balance_router_for_orchestration()
        
        # 根据 Agent 负载重新排序任务
        sorted_tasks = []
        for agent_name, inp in tasks:
            agent = self.pool.get(agent_name)
            if agent and not agent.is_running:
                sorted_tasks.insert(0, (agent_name, inp))  # 空闲 Agent 优先
            else:
                sorted_tasks.append((agent_name, inp))
        
        return self.run_parallel(sorted_tasks, max_workers)


class AgentRegistry:
    """Agent 注册表 - 全局 Agent 管理
    
    单例模式，提供全局 Agent 注册和发现。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._agents = {}
                    cls._instance._agent_classes = {}
                    cls._instance._factories = {}
        return cls._instance
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """注册 Agent 实例"""
        self._agents[name] = agent
    
    def register_class(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """注册 Agent 类"""
        self._agent_classes[name] = agent_class
    
    def register_factory(self, name: str, 
                        factory: Callable[..., BaseAgent]) -> None:
        """注册 Agent 工厂函数"""
        self._factories[name] = factory
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取 Agent 实例"""
        return self._agents.get(name)
    
    def create_agent(self, class_name: str, **kwargs) -> Optional[BaseAgent]:
        """创建 Agent"""
        agent_class = self._agent_classes.get(class_name)
        if agent_class:
            return agent_class(**kwargs)
        
        factory = self._factories.get(class_name)
        if factory:
            return factory(**kwargs)
        
        return None
    
    def list_agents(self) -> List[str]:
        """列出所有注册的 Agent"""
        return list(self._agents.keys())
    
    def list_classes(self) -> List[str]:
        """列出所有注册的 Agent 类"""
        return list(self._agent_classes.keys())
    
    def unregister(self, name: str) -> None:
        """注销"""
        self._agents.pop(name, None)
        self._agent_classes.pop(name, None)
        self._factories.pop(name, None)
    
    # ==================== 工业级派生方法 ====================
    
    def _get_master_factory(self):
        """延迟加载 MasterFactory 避免循环导入"""
        if not hasattr(self, '_master_factory') or self._master_factory is None:
            from .factory import get_master_factory
            self._master_factory = get_master_factory()
        return self._master_factory
    
    def create_agent_with_factory(self, agent_type: str, **kwargs) -> Optional[BaseAgent]:
        """使用 MasterFactory 创建 Agent
        
        调用 MasterFactory.agents.create
        
        Args:
            agent_type: Agent 类型
            **kwargs: Agent 参数
            
        Returns:
            Agent 实例或 None
        """
        try:
            factory = self._get_master_factory()
            return factory.agents.create(agent_type, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create agent with factory: {e}")
            return None
    
    def create_and_register_agent(self, agent_type: str, name: str, **kwargs) -> Optional[BaseAgent]:
        """使用工厂创建并注册 Agent
        
        调用 MasterFactory.agents.create 并注册
        
        Args:
            agent_type: Agent 类型
            name: Agent 名称
            **kwargs: Agent 参数
            
        Returns:
            Agent 实例或 None
        """
        agent = self.create_agent_with_factory(agent_type, name=name, **kwargs)
        if agent:
            self.register_agent(name, agent)
        return agent
    
    def get_checkpointer_for_agent(self, agent_name: str,
                                   checkpointer_type: str = "memory",
                                   **kwargs) -> Optional['Checkpointer']:
        """为 Agent 获取检查点器
        
        调用 MasterFactory.get_checkpointer_instance
        
        Args:
            agent_name: Agent 名称
            checkpointer_type: 检查点器类型
            **kwargs: 检查点器参数
            
        Returns:
            Checkpointer 实例或 None
        """
        factory = self._get_master_factory()
        return factory.get_checkpointer_instance(checkpointer_type, **kwargs)
    
    def create_state_serializer(self) -> 'StateSerializer':
        """创建状态序列化器
        
        调用 MasterFactory.create_state_serializer
        
        Returns:
            StateSerializer 实例
        """
        factory = self._get_master_factory()
        return factory.create_state_serializer()
    
    def create_state_validator(self) -> 'StateValidator':
        """创建状态验证器
        
        调用 MasterFactory.create_state_validator
        
        Returns:
            StateValidator 实例
        """
        factory = self._get_master_factory()
        return factory.create_state_validator()
    
    def create_message_buffer(self, max_size: int = 100) -> 'MessageBuffer':
        """创建消息缓冲区
        
        调用 MasterFactory.create_message_buffer
        
        Args:
            max_size: 缓冲区最大大小
            
        Returns:
            MessageBuffer 实例
        """
        factory = self._get_master_factory()
        return factory.create_message_buffer(max_size)
    
    def get_node_registry_from_factory(self) -> 'NodeRegistry':
        """从工厂获取节点注册表
        
        调用 MasterFactory.get_node_registry_instance
        
        Returns:
            NodeRegistry 实例
        """
        factory = self._get_master_factory()
        return factory.get_node_registry_instance()
    
    def get_registry_health(self) -> Dict[str, Any]:
        """获取注册表健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        factory = self._get_master_factory()
        return {
            "registered_agents": len(self._agents),
            "registered_classes": len(self._agent_classes),
            "registered_factories": len(self._factories),
            "factory_health": factory.health_check(),
            "factory_metrics": factory.get_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }


def get_agent_registry() -> AgentRegistry:
    """获取全局 Agent 注册表"""
    return AgentRegistry()


# ==================== Agent 工厂和便捷函数 ====================

class AgentFactory:
    """Agent 工厂"""
    
    _agent_types = {
        "react": ReActAgent,
        "plan_execute": PlanAndExecuteAgent,
        "reflexion": ReflexionAgent,
        "multi_agent": MultiAgentSystem,
        "tool_calling": ToolCallingAgent,
        "conversational": ConversationalAgent,
        "chain_of_thought": ChainOfThoughtAgent,
        "self_ask": SelfAskAgent,
        "hierarchical": HierarchicalAgent,
        "workflow": WorkflowAgent,
    }
    
    @classmethod
    def create(cls, 
               agent_type: str,
               config: AgentConfig = None,
               tools: List[Tool] = None,
               llm_client: Any = None,
               **kwargs) -> BaseAgent:
        """创建 Agent
        
        Args:
            agent_type: Agent 类型
            config: Agent 配置
            tools: 工具列表
            llm_client: LLM 客户端
            **kwargs: 额外参数
            
        Returns:
            Agent 实例
        """
        agent_class = cls._agent_types.get(agent_type.lower())
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(cls._agent_types.keys())}")
        
        return agent_class(
            config=config,
            tools=tools,
            llm_client=llm_client,
            **kwargs
        )
    
    @classmethod
    def register(cls, name: str, agent_class: type) -> None:
        """注册新的 Agent 类型"""
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class} must be a subclass of BaseAgent")
        cls._agent_types[name.lower()] = agent_class
    
    @classmethod
    def list_types(cls) -> List[str]:
        """列出所有可用的 Agent 类型"""
        return list(cls._agent_types.keys())


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """创建 Agent 的便捷函数"""
    return AgentFactory.create(agent_type, **kwargs)


def create_react_agent(tools: List[Tool] = None, **kwargs) -> ReActAgent:
    """创建 ReAct Agent 的便捷函数"""
    return ReActAgent(tools=tools, **kwargs)


def create_conversational_agent(tools: List[Tool] = None, **kwargs) -> ConversationalAgent:
    """创建对话 Agent 的便捷函数"""
    return ConversationalAgent(tools=tools, **kwargs)


def create_plan_execute_agent(tools: List[Tool] = None, **kwargs) -> PlanAndExecuteAgent:
    """创建 Plan-and-Execute Agent 的便捷函数"""
    return PlanAndExecuteAgent(tools=tools, **kwargs)


def create_reflexion_agent(tools: List[Tool] = None, **kwargs) -> ReflexionAgent:
    """创建 Reflexion Agent 的便捷函数"""
    return ReflexionAgent(tools=tools, **kwargs)


def create_tool_calling_agent(tools: List[Tool] = None, **kwargs) -> ToolCallingAgent:
    """创建 Tool-Calling Agent 的便捷函数"""
    return ToolCallingAgent(tools=tools, **kwargs)


def create_chain_of_thought_agent(tools: List[Tool] = None, **kwargs) -> ChainOfThoughtAgent:
    """创建 Chain-of-Thought Agent 的便捷函数"""
    return ChainOfThoughtAgent(tools=tools, **kwargs)


def create_self_ask_agent(tools: List[Tool] = None, **kwargs) -> SelfAskAgent:
    """创建 Self-Ask Agent 的便捷函数"""
    return SelfAskAgent(tools=tools, **kwargs)


def create_hierarchical_agent(tools: List[Tool] = None, 
                             worker_configs: List[AgentConfig] = None,
                             **kwargs) -> HierarchicalAgent:
    """创建层级 Agent 的便捷函数"""
    return HierarchicalAgent(tools=tools, worker_configs=worker_configs, **kwargs)


def create_workflow_agent(workflow_steps: List['WorkflowAgent.WorkflowStep'] = None,
                         **kwargs) -> WorkflowAgent:
    """创建工作流 Agent 的便捷函数"""
    return WorkflowAgent(workflow_steps=workflow_steps, **kwargs)


def create_multi_agent_system(agents: List[BaseAgent] = None,
                             mode: str = "supervisor",
                             **kwargs) -> MultiAgentSystem:
    """创建多 Agent 系统的便捷函数"""
    return MultiAgentSystem(agents=agents, mode=mode, **kwargs)


# ==================== Agent 快捷操作 ====================

async def quick_ask(question: str, 
                   agent_type: str = "react",
                   tools: List[Tool] = None,
                   llm_client: Any = None,
                   **kwargs) -> str:
    """快速提问
    
    Args:
        question: 问题
        agent_type: Agent 类型
        tools: 工具列表
        llm_client: LLM 客户端
        
    Returns:
        答案字符串
    """
    agent = AgentFactory.create(
        agent_type=agent_type,
        tools=tools,
        llm_client=llm_client,
        **kwargs
    )
    
    result = await agent.ainvoke(question)
    return result.final_answer or result.output or ""


def quick_ask_sync(question: str, 
                  agent_type: str = "react",
                  tools: List[Tool] = None,
                  llm_client: Any = None,
                  **kwargs) -> str:
    """快速提问 (同步版本)"""
    agent = AgentFactory.create(
        agent_type=agent_type,
        tools=tools,
        llm_client=llm_client,
        **kwargs
    )
    
    result = agent.invoke(question)
    return result.final_answer or result.output or ""


async def quick_chat(messages: List[str],
                    system_prompt: str = None,
                    llm_client: Any = None,
                    **kwargs) -> List[str]:
    """快速对话
    
    Args:
        messages: 消息列表（用户消息）
        system_prompt: 系统提示
        llm_client: LLM 客户端
        
    Returns:
        回复列表
    """
    config = AgentConfig(system_prompt=system_prompt) if system_prompt else None
    agent = ConversationalAgent(config=config, llm_client=llm_client, **kwargs)
    
    responses = []
    thread_id = str(uuid.uuid4())
    
    for msg in messages:
        result = await agent.ainvoke(msg, thread_id=thread_id)
        response = result.final_answer or result.output or ""
        responses.append(response)
    
    return responses


async def quick_plan_execute(task: str,
                            tools: List[Tool] = None,
                            llm_client: Any = None,
                            **kwargs) -> Dict[str, Any]:
    """快速计划并执行
    
    Args:
        task: 任务描述
        tools: 工具列表
        llm_client: LLM 客户端
        
    Returns:
        包含计划和结果的字典
    """
    agent = PlanAndExecuteAgent(tools=tools, llm_client=llm_client, **kwargs)
    result = await agent.ainvoke(task)
    
    return {
        "task": task,
        "plan": result.data.get("plan", []),
        "executed_steps": result.data.get("executed_steps", []),
        "result": result.final_answer or result.output
    }


# ==================== 增强辅助类 ====================

class AgentStateHelper:
    """Agent 状态辅助类 - 封装对 state 模块的高级操作"""
    
    @staticmethod
    def create_execution_context(
        priority: PriorityLevel = PriorityLevel.NORMAL,
        execution_mode: StateExecutionMode = StateExecutionMode.ASYNC,
        interrupt_type: InterruptType = InterruptType.NONE,
        timeout: float = 300.0,
        metadata: Dict[str, Any] = None
    ) -> StateExecutionContext:
        """创建执行上下文
        
        Args:
            priority: 优先级
            execution_mode: 执行模式
            interrupt_type: 中断类型
            timeout: 超时时间
            metadata: 元数据
        
        Returns:
            StateExecutionContext 实例
        """
        return StateExecutionContext(
            priority=priority,
            execution_mode=execution_mode,
            interrupt_type=interrupt_type,
            timeout=timeout,
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_checkpoint(
        state: AgentState,
        checkpoint_id: str = None,
        description: str = ""
    ) -> StateCheckpoint:
        """创建状态检查点
        
        Args:
            state: 要保存的状态
            checkpoint_id: 检查点 ID
            description: 描述
        
        Returns:
            StateCheckpoint 实例
        """
        return StateCheckpoint(
            checkpoint_id=checkpoint_id or str(uuid.uuid4()),
            state=state.copy(),
            created_at=datetime.utcnow(),
            description=description
        )
    
    @staticmethod
    def create_plan(
        goal: str,
        steps: List[str],
        context: Dict[str, Any] = None
    ) -> AgentPlan:
        """创建执行计划
        
        Args:
            goal: 目标
            steps: 步骤列表
            context: 上下文
        
        Returns:
            AgentPlan 实例
        """
        plan_steps = [
            PlanStep(
                step_id=str(uuid.uuid4()),
                description=step,
                order=i
            )
            for i, step in enumerate(steps)
        ]
        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            steps=plan_steps,
            context=context or {}
        )
    
    @staticmethod
    def create_reflection(
        content: str,
        quality_score: float = 0.0,
        suggestions: List[str] = None
    ) -> Reflection:
        """创建反思记录
        
        Args:
            content: 反思内容
            quality_score: 质量评分
            suggestions: 改进建议
        
        Returns:
            Reflection 实例
        """
        return Reflection(
            reflection_id=str(uuid.uuid4()),
            content=content,
            quality_score=quality_score,
            suggestions=suggestions or [],
            created_at=datetime.utcnow()
        )
    
    @staticmethod
    def get_error_type_for_exception(e: Exception) -> ErrorType:
        """根据异常类型获取 ErrorType"""
        if isinstance(e, TimeoutError):
            return ErrorType.TIMEOUT
        elif isinstance(e, ValueError):
            return ErrorType.VALIDATION
        elif isinstance(e, MemoryError):
            return ErrorType.RESOURCE
        else:
            return ErrorType.EXECUTION


class AgentNodeHelper:
    """Agent 节点辅助类 - 封装对 nodes 模块的高级操作"""
    
    @staticmethod
    def create_node_by_type(
        node_type: NodeType,
        name: str,
        config: Dict[str, Any] = None,
        **kwargs
    ) -> BaseNode:
        """根据类型创建节点
        
        Args:
            node_type: 节点类型
            name: 节点名称
            config: 节点配置
            **kwargs: 额外参数
        
        Returns:
            BaseNode 实例
        """
        registry = get_node_registry()
        return registry.create(node_type.value, name=name, config=config, **kwargs)
    
    @staticmethod
    def get_node_metrics(node: BaseNode) -> NodeMetrics:
        """获取节点指标
        
        Args:
            node: 节点实例
        
        Returns:
            NodeMetrics 实例
        """
        if hasattr(node, 'get_metrics'):
            return node.get_metrics()
        return NodeMetrics(
            node_name=node.name,
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            total_latency_ms=0.0
        )
    
    @staticmethod
    def create_node_chain(nodes: List[BaseNode]) -> NodeChain:
        """创建节点链
        
        Args:
            nodes: 节点列表
        
        Returns:
            NodeChain 实例
        """
        return NodeChain(nodes=nodes)
    
    @staticmethod
    def create_parallel_group(
        nodes: List[BaseNode],
        merge_strategy: str = "merge"
    ) -> NodeParallelGroup:
        """创建并行节点组
        
        Args:
            nodes: 节点列表
            merge_strategy: 合并策略
        
        Returns:
            NodeParallelGroup 实例
        """
        return NodeParallelGroup(nodes=nodes, merge_strategy=merge_strategy)


class AgentEdgeHelper:
    """Agent 边辅助类 - 封装对 edges 模块的高级操作"""
    
    @staticmethod
    def create_conditional_edge(
        source: str,
        condition: Callable[[AgentState], str],
        branches: Dict[str, str],
        edge_type: EdgeType = EdgeType.CONDITIONAL
    ) -> ConditionalEdge:
        """创建条件边
        
        Args:
            source: 源节点
            condition: 条件函数
            branches: 分支映射
            edge_type: 边类型
        
        Returns:
            ConditionalEdge 实例
        """
        return ConditionalEdge(
            source=source,
            condition=condition,
            branches=branches,
            edge_type=edge_type
        )
    
    @staticmethod
    def create_edge_conditions(
        *conditions: EdgeCondition
    ) -> EdgeConditions:
        """创建边条件集合
        
        Args:
            *conditions: 条件列表
        
        Returns:
            EdgeConditions 实例
        """
        return EdgeConditions(conditions=list(conditions))
    
    @staticmethod
    def route_after_tools_call(state: AgentState) -> str:
        """工具调用后的路由"""
        return route_after_tools(state)
    
    @staticmethod
    def create_priority_router(
        priorities: Dict[str, int]
    ) -> PriorityRouter:
        """创建优先级路由器
        
        Args:
            priorities: 节点优先级映射
        
        Returns:
            PriorityRouter 实例
        """
        return PriorityRouter(priorities=priorities)


class AgentGraphHelper:
    """Agent 图辅助类 - 封装对 graph 模块的高级操作"""
    
    @staticmethod
    def create_simple_agent_graph(
        name: str,
        nodes: List[BaseNode],
        edges: List[Edge] = None
    ) -> StateGraph:
        """创建简单的 Agent 图
        
        Args:
            name: 图名称
            nodes: 节点列表
            edges: 边列表
        
        Returns:
            StateGraph 实例
        """
        return create_simple_graph(name, nodes, edges)
    
    @staticmethod
    def create_react_agent_graph(
        name: str,
        llm_node: LLMNode,
        tool_node: ToolNode,
        config: GraphConfig = None
    ) -> StateGraph:
        """创建 ReAct Agent 图
        
        Args:
            name: 图名称
            llm_node: LLM 节点
            tool_node: 工具节点
            config: 图配置
        
        Returns:
            StateGraph 实例
        """
        return create_react_graph(name, llm_node, tool_node, config)
    
    @staticmethod
    def get_graph_metrics(graph: CompiledGraph) -> GraphMetrics:
        """获取图执行指标
        
        Args:
            graph: 编译后的图
        
        Returns:
            GraphMetrics 实例
        """
        if hasattr(graph, 'get_metrics'):
            return graph.get_metrics()
        return GraphMetrics(
            graph_name=graph.config.name if graph.config else "unknown",
            total_runs=0,
            successful_runs=0,
            failed_runs=0
        )
    
    @staticmethod
    def create_execution_event(
        event_type: str,
        node_name: str,
        data: Dict[str, Any] = None
    ) -> ExecutionEvent:
        """创建执行事件
        
        Args:
            event_type: 事件类型
            node_name: 节点名称
            data: 事件数据
        
        Returns:
            ExecutionEvent 实例
        """
        return ExecutionEvent(
            event_type=event_type,
            node_name=node_name,
            timestamp=datetime.utcnow(),
            data=data or {}
        )
    
    @staticmethod
    def get_graph_status(graph: StateGraph) -> GraphStatus:
        """获取图状态
        
        Args:
            graph: 状态图
        
        Returns:
            GraphStatus 枚举值
        """
        if hasattr(graph, 'status'):
            return graph.status
        return GraphStatus.IDLE


class AgentToolHelper:
    """Agent 工具辅助类 - 封装对 tools 模块的高级操作"""
    
    @staticmethod
    def create_tool_parameter(
        name: str,
        param_type: str,
        description: str,
        required: bool = True,
        default: Any = None
    ) -> ToolParameter:
        """创建工具参数定义
        
        Args:
            name: 参数名
            param_type: 参数类型
            description: 描述
            required: 是否必需
            default: 默认值
        
        Returns:
            ToolParameter 实例
        """
        return ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default
        )
    
    @staticmethod
    def get_tool_status(tool: Tool) -> ToolStatus:
        """获取工具状态
        
        Args:
            tool: 工具实例
        
        Returns:
            ToolStatus 枚举值
        """
        if hasattr(tool, 'status'):
            return tool.status
        return ToolStatus.READY
    
    @staticmethod
    def get_tool_metrics(tool: Tool) -> ToolMetrics:
        """获取工具指标
        
        Args:
            tool: 工具实例
        
        Returns:
            ToolMetrics 实例
        """
        if hasattr(tool, 'get_metrics'):
            return tool.get_metrics()
        return ToolMetrics(
            tool_name=tool.name,
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            total_latency_ms=0.0
        )
    
    @staticmethod
    def create_tool_hooks(
        before_call: Callable = None,
        after_call: Callable = None,
        on_error: Callable = None
    ) -> ToolHooks:
        """创建工具钩子
        
        Args:
            before_call: 调用前钩子
            after_call: 调用后钩子
            on_error: 错误钩子
        
        Returns:
            ToolHooks 实例
        """
        return ToolHooks(
            before_call=before_call,
            after_call=after_call,
            on_error=on_error
        )
    
    @staticmethod
    def create_sync_tool(
        name: str,
        description: str,
        func: Callable,
        parameters: List[ToolParameter] = None
    ) -> Tool:
        """创建同步工具
        
        Args:
            name: 工具名称
            description: 描述
            func: 工具函数
            parameters: 参数列表
        
        Returns:
            Tool 实例
        """
        @tool(name=name, description=description)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    
    @staticmethod
    def create_async_tool(
        name: str,
        description: str,
        func: Callable,
        parameters: List[ToolParameter] = None
    ) -> Tool:
        """创建异步工具
        
        Args:
            name: 工具名称
            description: 描述
            func: 异步工具函数
            parameters: 参数列表
        
        Returns:
            Tool 实例
        """
        @async_tool(name=name, description=description)
        async def wrapped(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapped


class AgentCheckpointerHelper:
    """Agent 检查点辅助类 - 封装对 checkpointer 模块的高级操作"""
    
    @staticmethod
    def create_memory_checkpointer() -> MemoryCheckpointer:
        """创建内存检查点器"""
        return create_memory_checkpointer()
    
    @staticmethod
    def create_redis_checkpointer(
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = None,
        prefix: str = "agent_checkpoint"
    ) -> RedisCheckpointer:
        """创建 Redis 检查点器
        
        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库索引
            password: 密码
            prefix: 键前缀
        
        Returns:
            RedisCheckpointer 实例
        """
        return RedisCheckpointer(
            host=host,
            port=port,
            db=db,
            password=password,
            prefix=prefix
        )
    
    @staticmethod
    def create_sqlite_checkpointer(
        db_path: str = "checkpoints.db",
        table_name: str = "checkpoints"
    ) -> SQLiteCheckpointer:
        """创建 SQLite 检查点器
        
        Args:
            db_path: 数据库路径
            table_name: 表名
        
        Returns:
            SQLiteCheckpointer 实例
        """
        return SQLiteCheckpointer(
            db_path=db_path,
            table_name=table_name
        )
    
    @staticmethod
    def create_file_checkpointer(
        directory: str = "./checkpoints",
        compression: CompressionType = CompressionType.GZIP
    ) -> FileCheckpointer:
        """创建文件检查点器
        
        Args:
            directory: 存储目录
            compression: 压缩类型
        
        Returns:
            FileCheckpointer 实例
        """
        return FileCheckpointer(
            directory=directory,
            compression=compression
        )
    
    @staticmethod
    def get_checkpointer_by_type(
        checkpointer_type: str,
        **kwargs
    ) -> Checkpointer:
        """根据类型获取检查点器
        
        Args:
            checkpointer_type: 检查点器类型
            **kwargs: 额外参数
        
        Returns:
            Checkpointer 实例
        """
        return get_checkpointer(checkpointer_type, **kwargs)


class AgentBuiltinToolHelper:
    """Agent 内置工具辅助类 - 封装对 builtin_tools 模块的高级操作"""
    
    @staticmethod
    def get_tool(name: str) -> Optional[Tool]:
        """按名称获取内置工具
        
        Args:
            name: 工具名称
        
        Returns:
            Tool 实例或 None
        """
        return get_tool_by_name(name)
    
    @staticmethod
    def get_tool_information(name: str) -> Optional[Dict[str, Any]]:
        """获取工具详细信息
        
        Args:
            name: 工具名称
        
        Returns:
            工具信息字典或 None
        """
        return get_tool_info(name)
    
    @staticmethod
    def get_tools_by_cat(category: BuiltinToolCategory) -> List[Tool]:
        """按类别获取工具
        
        Args:
            category: 工具类别
        
        Returns:
            工具列表
        """
        return get_tools_by_category(category.value)
    
    @staticmethod
    def get_all_builtin_tools() -> List[Tool]:
        """获取所有内置工具"""
        return get_builtin_tools()
    
    @staticmethod
    def get_agent_recommended_tools(agent_type: str) -> List[Tool]:
        """获取 Agent 推荐的工具
        
        Args:
            agent_type: Agent 类型
        
        Returns:
            推荐工具列表
        """
        return get_tools_for_agent(agent_type)


# ==================== 增强的 Agent 构建器 ====================

class EnhancedAgentBuilder:
    """增强的 Agent 构建器 - 支持完整的模块集成"""
    
    def __init__(self, agent_type: str = "react"):
        """初始化构建器
        
        Args:
            agent_type: Agent 类型
        """
        self._agent_type = agent_type
        self._config = AgentConfig()
        self._tools: List[Tool] = []
        self._llm_client = None
        self._checkpointer: Optional[Checkpointer] = None
        self._callbacks: List[AgentCallback] = []
        self._custom_nodes: Dict[str, BaseNode] = {}
        self._custom_edges: List[Edge] = []
        self._execution_context: Optional[StateExecutionContext] = None
        
        # 辅助类
        self._state_helper = AgentStateHelper()
        self._node_helper = AgentNodeHelper()
        self._edge_helper = AgentEdgeHelper()
        self._graph_helper = AgentGraphHelper()
        self._tool_helper = AgentToolHelper()
        self._checkpoint_helper = AgentCheckpointerHelper()
        self._builtin_helper = AgentBuiltinToolHelper()
    
    def with_config(self, config: AgentConfig) -> 'EnhancedAgentBuilder':
        """设置配置"""
        self._config = config
        return self
    
    def with_name(self, name: str) -> 'EnhancedAgentBuilder':
        """设置名称"""
        self._config.name = name
        return self
    
    def with_model(self, model: str) -> 'EnhancedAgentBuilder':
        """设置模型"""
        self._config.model = model
        return self
    
    def with_temperature(self, temperature: float) -> 'EnhancedAgentBuilder':
        """设置温度"""
        self._config.temperature = temperature
        return self
    
    def with_max_iterations(self, max_iterations: int) -> 'EnhancedAgentBuilder':
        """设置最大迭代次数"""
        self._config.max_iterations = max_iterations
        return self
    
    def with_timeout(self, timeout: float) -> 'EnhancedAgentBuilder':
        """设置超时"""
        self._config.timeout = timeout
        return self
    
    def with_system_prompt(self, prompt: str) -> 'EnhancedAgentBuilder':
        """设置系统提示"""
        self._config.system_prompt = prompt
        return self
    
    def with_tools(self, tools: List[Tool]) -> 'EnhancedAgentBuilder':
        """添加工具"""
        self._tools.extend(tools)
        return self
    
    def with_builtin_tools(self, tool_names: List[str] = None) -> 'EnhancedAgentBuilder':
        """添加内置工具"""
        if tool_names:
            for name in tool_names:
                t = self._builtin_helper.get_tool(name)
                if t:
                    self._tools.append(t)
        else:
            self._tools.extend(self._builtin_helper.get_all_builtin_tools())
        return self
    
    def with_tools_for_agent_type(self) -> 'EnhancedAgentBuilder':
        """添加 Agent 类型推荐的工具"""
        recommended = self._builtin_helper.get_agent_recommended_tools(self._agent_type)
        self._tools.extend(recommended)
        return self
    
    def with_llm_client(self, llm_client: Any) -> 'EnhancedAgentBuilder':
        """设置 LLM 客户端"""
        self._llm_client = llm_client
        return self
    
    def with_memory_checkpointer(self) -> 'EnhancedAgentBuilder':
        """使用内存检查点器"""
        self._checkpointer = self._checkpoint_helper.create_memory_checkpointer()
        return self
    
    def with_redis_checkpointer(self, **kwargs) -> 'EnhancedAgentBuilder':
        """使用 Redis 检查点器"""
        self._checkpointer = self._checkpoint_helper.create_redis_checkpointer(**kwargs)
        return self
    
    def with_sqlite_checkpointer(self, **kwargs) -> 'EnhancedAgentBuilder':
        """使用 SQLite 检查点器"""
        self._checkpointer = self._checkpoint_helper.create_sqlite_checkpointer(**kwargs)
        return self
    
    def with_file_checkpointer(self, **kwargs) -> 'EnhancedAgentBuilder':
        """使用文件检查点器"""
        self._checkpointer = self._checkpoint_helper.create_file_checkpointer(**kwargs)
        return self
    
    def with_callback(self, callback: AgentCallback) -> 'EnhancedAgentBuilder':
        """添加回调"""
        self._callbacks.append(callback)
        return self
    
    def with_logging(self, level: int = logging.INFO) -> 'EnhancedAgentBuilder':
        """添加日志回调"""
        self._callbacks.append(LoggingCallback(level))
        return self
    
    def with_metrics(self) -> 'EnhancedAgentBuilder':
        """添加指标回调"""
        self._callbacks.append(MetricsCallback())
        self._config.enable_metrics = True
        return self
    
    def with_streaming(self, 
                      on_token: Callable = None,
                      on_chunk: Callable = None) -> 'EnhancedAgentBuilder':
        """添加流式回调"""
        self._callbacks.append(StreamingCallback(on_token, on_chunk))
        self._config.enable_streaming = True
        return self
    
    def with_execution_context(self,
                              priority: PriorityLevel = PriorityLevel.NORMAL,
                              execution_mode: StateExecutionMode = StateExecutionMode.ASYNC,
                              **kwargs) -> 'EnhancedAgentBuilder':
        """设置执行上下文"""
        self._execution_context = self._state_helper.create_execution_context(
            priority=priority,
            execution_mode=execution_mode,
            **kwargs
        )
        return self
    
    def with_circuit_breaker(self,
                            threshold: int = 5,
                            recovery_timeout: float = 30.0) -> 'EnhancedAgentBuilder':
        """启用熔断器"""
        self._config.enable_circuit_breaker = True
        self._config.circuit_breaker_threshold = threshold
        self._config.circuit_breaker_recovery = recovery_timeout
        return self
    
    def with_rate_limiter(self,
                         rpm: int = 60,
                         burst: int = 10) -> 'EnhancedAgentBuilder':
        """启用限流器"""
        self._config.enable_rate_limiter = True
        self._config.rate_limit_rpm = rpm
        self._config.rate_limit_burst = burst
        return self
    
    def with_human_in_loop(self) -> 'EnhancedAgentBuilder':
        """启用人机协作"""
        self._config.enable_human_in_loop = True
        return self
    
    def with_memory(self,
                   max_short_term: int = 100,
                   max_long_term: int = 1000) -> 'EnhancedAgentBuilder':
        """启用记忆"""
        self._config.enable_memory = True
        self._config.memory_max_short_term = max_short_term
        self._config.memory_max_long_term = max_long_term
        return self
    
    def with_checkpointing(self,
                          interval: int = 1,
                          compression: bool = True) -> 'EnhancedAgentBuilder':
        """启用检查点"""
        self._config.enable_checkpointing = True
        self._config.checkpoint_interval = interval
        self._config.checkpoint_compression = compression
        return self
    
    def add_custom_node(self, name: str, node: BaseNode) -> 'EnhancedAgentBuilder':
        """添加自定义节点"""
        self._custom_nodes[name] = node
        return self
    
    def add_custom_edge(self, edge: Edge) -> 'EnhancedAgentBuilder':
        """添加自定义边"""
        self._custom_edges.append(edge)
        return self
    
    def build(self) -> BaseAgent:
        """构建 Agent
        
        Returns:
            构建的 Agent 实例
        """
        agent_class = AgentFactory._agent_types.get(self._agent_type.lower())
        if not agent_class:
            raise ValueError(f"Unknown agent type: {self._agent_type}")
        
        agent = agent_class(
            config=self._config,
            tools=self._tools,
            llm_client=self._llm_client,
            checkpointer=self._checkpointer,
            callbacks=self._callbacks
        )
        
        return agent


def build_enhanced_agent(agent_type: str = "react") -> EnhancedAgentBuilder:
    """创建增强 Agent 构建器
    
    Args:
        agent_type: Agent 类型
    
    Returns:
        EnhancedAgentBuilder 实例
    """
    return EnhancedAgentBuilder(agent_type)


# ==================== 模块初始化 ====================

def _register_builtin_agents():
    """注册内置 Agent 类型到全局注册表"""
    registry = get_agent_registry()
    
    # 注册 Agent 类
    registry.register_class("react", ReActAgent)
    registry.register_class("plan_execute", PlanAndExecuteAgent)
    registry.register_class("reflexion", ReflexionAgent)
    registry.register_class("multi_agent", MultiAgentSystem)
    registry.register_class("tool_calling", ToolCallingAgent)
    registry.register_class("conversational", ConversationalAgent)
    registry.register_class("chain_of_thought", ChainOfThoughtAgent)
    registry.register_class("self_ask", SelfAskAgent)
    registry.register_class("hierarchical", HierarchicalAgent)
    registry.register_class("workflow", WorkflowAgent)


# 模块加载时注册内置 Agent
_register_builtin_agents()


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "AgentEventType",
    "AgentMode",
    "AgentStatus",
    "StrategyType",
    
    # 配置
    "AgentConfig",
    
    # 回调
    "AgentEvent",
    "AgentCallback",
    "LoggingCallback",
    "MetricsCallback",
    "StreamingCallback",
    "WebhookCallback",
    "BufferedCallback",
    "ProfilingCallback",
    "CallbackManager",
    
    # 追踪
    "ExecutionTrace",
    "ExecutionTracer",
    
    # 上下文
    "ExecutionContext",
    
    # 策略基类
    "BaseStrategy",
    "StateStrategy",
    "NodeStrategy",
    "EdgeStrategy",
    "GraphStrategy",
    "ToolStrategy",
    "CheckpointStrategy",
    "MemoryStrategy",
    "RoutingStrategy",
    
    # 默认策略实现
    "DefaultStateStrategy",
    "DefaultNodeStrategy",
    "DefaultEdgeStrategy",
    "DefaultGraphStrategy",
    "DefaultToolStrategy",
    "DefaultCheckpointStrategy",
    "DefaultMemoryStrategy",
    "DefaultRoutingStrategy",
    
    # 策略管理
    "StrategyManager",
    "get_strategy_manager",
    "set_strategy_manager",
    
    # 策略装饰器
    "with_strategy",
    "strategy_based",
    
    # 基类
    "BaseAgent",
    
    # Agent 类型
    "ReActAgent",
    "PlanAndExecuteAgent",
    "ReflexionAgent",
    "MultiAgentSystem",
    "ToolCallingAgent",
    "ConversationalAgent",
    "ChainOfThoughtAgent",
    "SelfAskAgent",
    "HierarchicalAgent",
    "WorkflowAgent",
    
    # 编排
    "AgentPool",
    "AgentOrchestrator",
    "AgentRegistry",
    "get_agent_registry",
    
    # 工厂
    "AgentFactory",
    
    # 便捷函数
    "create_agent",
    "create_react_agent",
    "create_conversational_agent",
    "create_plan_execute_agent",
    "create_reflexion_agent",
    "create_tool_calling_agent",
    "create_chain_of_thought_agent",
    "create_self_ask_agent",
    "create_hierarchical_agent",
    "create_workflow_agent",
    "create_multi_agent_system",
    
    # 快捷操作
    "quick_ask",
    "quick_ask_sync",
    "quick_chat",
    "quick_plan_execute",
    
    # 增强辅助类
    "AgentStateHelper",
    "AgentNodeHelper",
    "AgentEdgeHelper",
    "AgentGraphHelper",
    "AgentToolHelper",
    "AgentCheckpointerHelper",
    "AgentBuiltinToolHelper",
    
    # 增强构建器
    "EnhancedAgentBuilder",
    "build_enhanced_agent",
]
