"""
Agent 工厂 - 生产级实现

提供完整的 Agent 及相关组件的工厂模式实现：

核心工厂：
- AgentFactory: Agent 工厂，创建各种类型的 Agent
- NodeFactory: 节点工厂，创建图节点
- EdgeFactory: 边工厂，创建图边
- GraphFactory: 图工厂，创建执行图
- ToolFactory: 工具工厂，创建和管理工具
- CheckpointerFactory: 检查点工厂，创建检查点器
- StrategyFactory: 策略工厂，创建执行策略

构建器：
- PipelineBuilder: 流水线构建器
- WorkflowBuilder: 工作流构建器
- AgentBuilder: Agent 构建器（流式API）

模板：
- AgentTemplate: Agent 模板
- 预置模板：研究、编码、数据分析、ML训练等

特性：
- 统一的工厂注册机制
- 配置驱动的组件创建
- 策略模式集成
- 完整的类型支持
- 验证和诊断
"""

import logging
import uuid
import threading
import copy
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Callable, 
    Type, Tuple, TypeVar, Generic, Set
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps

# 状态相关
from .state import (
    AgentState, AgentMessage, MessageType, ToolCall, ToolResult,
    AgentStatus, StateCheckpoint, StateManager, StateValidator,
    StateSerializer, StateEventEmitter, MessageBuffer, AgentMemory,
    MemoryEntry, PlanStep, AgentPlan, Reflection, MemoryType
)

# 节点相关
from .nodes import (
    BaseNode, LLMNode, ToolNode, HumanNode, NodeFactory as BaseNodeFactory,
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
    LoadBalanceRouter, ABTestRouter, RoutingStrategy,
    create_conditional_edge, create_retry_edge, create_fallback_edge
)

# 图相关
from .graph import (
    StateGraph, GraphBuilder, CompiledGraph, GraphConfig,
    GraphStatus, ExecutionMode, GraphMetrics, ExecutionEvent,
    GraphRunner, GraphAnalyzer, Subgraph,
    create_simple_graph, create_react_graph
)

# 工具相关
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
    CompressionType, get_checkpointer, create_memory_checkpointer,
    create_redis_checkpointer, create_sqlite_checkpointer,
    create_file_checkpointer
)

# 内置工具
from .builtin_tools import (
    get_builtin_tools, get_tools_by_category, get_tools_for_agent,
    get_tool_by_name, get_tool_info, get_search_tools, get_code_tools,
    get_data_tools, get_system_tools, get_http_tools, get_memory_tools,
    get_knowledge_tools, get_text_tools, get_file_tools,
    ToolCategory as BuiltinToolCategory
)

# Agent 相关
from .agents import (
    BaseAgent, ReActAgent, PlanAndExecuteAgent, 
    ReflexionAgent, MultiAgentSystem, ToolCallingAgent,
    ConversationalAgent, ChainOfThoughtAgent, SelfAskAgent,
    HierarchicalAgent, WorkflowAgent,
    AgentConfig, AgentRole, AgentEventType, AgentMode,
    AgentCallback, LoggingCallback, MetricsCallback,
    StreamingCallback, WebhookCallback, CallbackManager,
    ExecutionTrace, ExecutionTracer, AgentPool, AgentOrchestrator,
    AgentRegistry, get_agent_registry,
    # 策略相关
    StrategyType, BaseStrategy, StateStrategy, NodeStrategy,
    EdgeStrategy, GraphStrategy, ToolStrategy, CheckpointStrategy,
    MemoryStrategy, RoutingStrategy as AgentRoutingStrategy,
    DefaultStateStrategy, DefaultNodeStrategy, DefaultEdgeStrategy,
    DefaultGraphStrategy, DefaultToolStrategy, DefaultCheckpointStrategy,
    DefaultMemoryStrategy, DefaultRoutingStrategy,
    StrategyManager, get_strategy_manager, set_strategy_manager,
    # 辅助类（新增）
    AgentStateHelper, AgentNodeHelper, AgentEdgeHelper,
    AgentGraphHelper, AgentToolHelper, AgentCheckpointerHelper,
    AgentBuiltinToolHelper, EnhancedAgentBuilder
)

# 状态模块增强
from .state import (
    InterruptType, ErrorType, ContextType,
    PriorityLevel, ExecutionContext
)
# 为了向后兼容，创建别名
StateExecutionMode = ExecutionMode  # 使用 graph 模块中的 ExecutionMode
StateExecutionContext = ExecutionContext

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
AgentT = TypeVar('AgentT', bound=BaseAgent)


# ==================== 枚举和常量 ====================

class FactoryType(Enum):
    """工厂类型"""
    AGENT = "agent"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"
    TOOL = "tool"
    CHECKPOINTER = "checkpointer"
    STRATEGY = "strategy"
    CALLBACK = "callback"


class ComponentType(Enum):
    """组件类型"""
    # Agent 类型
    REACT_AGENT = "react_agent"
    PLAN_EXECUTE_AGENT = "plan_execute_agent"
    REFLEXION_AGENT = "reflexion_agent"
    MULTI_AGENT = "multi_agent"
    TOOL_CALLING_AGENT = "tool_calling_agent"
    CONVERSATIONAL_AGENT = "conversational_agent"
    CHAIN_OF_THOUGHT_AGENT = "chain_of_thought_agent"
    SELF_ASK_AGENT = "self_ask_agent"
    HIERARCHICAL_AGENT = "hierarchical_agent"
    WORKFLOW_AGENT = "workflow_agent"
    
    # 节点类型
    LLM_NODE = "llm_node"
    TOOL_NODE = "tool_node"
    HUMAN_NODE = "human_node"
    TRANSFORM_NODE = "transform_node"
    BRANCH_NODE = "branch_node"
    PARALLEL_NODE = "parallel_node"
    RETRY_NODE = "retry_node"
    CACHE_NODE = "cache_node"
    MEMORY_NODE = "memory_node"
    REFLECTION_NODE = "reflection_node"
    PLANNING_NODE = "planning_node"
    
    # 边类型
    SIMPLE_EDGE = "simple_edge"
    CONDITIONAL_EDGE = "conditional_edge"
    LOOP_EDGE = "loop_edge"
    PARALLEL_EDGE = "parallel_edge"
    RETRY_EDGE = "retry_edge"
    FALLBACK_EDGE = "fallback_edge"
    TIMEOUT_EDGE = "timeout_edge"
    CIRCUIT_BREAKER_EDGE = "circuit_breaker_edge"
    
    # 检查点类型
    MEMORY_CHECKPOINTER = "memory_checkpointer"
    REDIS_CHECKPOINTER = "redis_checkpointer"
    SQLITE_CHECKPOINTER = "sqlite_checkpointer"
    FILE_CHECKPOINTER = "file_checkpointer"


# ==================== 工厂配置 ====================

@dataclass
class FactoryConfig:
    """工厂配置
    
    统一的工厂配置，控制组件创建行为。
    """
    # 基本配置
    name: str = "default_factory"
    description: str = ""
    version: str = "1.0.0"
    
    # 默认值
    default_model: str = "gpt-4"
    default_temperature: float = 0.7
    default_max_iterations: int = 10
    default_timeout: float = 300.0
    
    # 功能开关
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True
    enable_logging: bool = True
    
    # 检查点配置
    default_checkpointer_type: str = "memory"
    checkpointer_config: Dict[str, Any] = field(default_factory=dict)
    
    # 策略配置
    enable_strategies: bool = True
    default_strategies: Dict[str, str] = field(default_factory=dict)
    
    # 工具配置
    auto_load_builtin_tools: bool = True
    tool_categories: List[str] = field(default_factory=list)
    
    # 回调配置
    default_callbacks: List[str] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "default_model": self.default_model,
            "default_temperature": self.default_temperature,
            "default_max_iterations": self.default_max_iterations,
            "default_timeout": self.default_timeout,
            "enable_caching": self.enable_caching,
            "enable_metrics": self.enable_metrics,
            "enable_validation": self.enable_validation,
            "default_checkpointer_type": self.default_checkpointer_type,
            "enable_strategies": self.enable_strategies,
            "auto_load_builtin_tools": self.auto_load_builtin_tools,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactoryConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def merge(self, override: Dict[str, Any]) -> 'FactoryConfig':
        """合并配置"""
        config_dict = self.to_dict()
        config_dict.update(override)
        return FactoryConfig.from_dict(config_dict)


# ==================== 工厂注册表 ====================

class FactoryRegistry:
    """工厂注册表
    
    管理所有工厂和组件的注册。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        """初始化注册表"""
        self._factories: Dict[FactoryType, Dict[str, Any]] = {
            ft: {} for ft in FactoryType
        }
        self._creators: Dict[str, Callable] = {}
        self._validators: Dict[str, Callable] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._config = FactoryConfig()
    
    def register_factory(self, factory_type: FactoryType, 
                        name: str, factory: Any) -> None:
        """注册工厂"""
        self._factories[factory_type][name] = factory
    
    def get_factory(self, factory_type: FactoryType, 
                   name: str) -> Optional[Any]:
        """获取工厂"""
        return self._factories.get(factory_type, {}).get(name)
    
    def register_creator(self, component_type: str, 
                        creator: Callable) -> None:
        """注册组件创建器"""
        self._creators[component_type] = creator
    
    def get_creator(self, component_type: str) -> Optional[Callable]:
        """获取创建器"""
        return self._creators.get(component_type)
    
    def register_validator(self, component_type: str, 
                          validator: Callable) -> None:
        """注册验证器"""
        self._validators[component_type] = validator
    
    def validate(self, component_type: str, component: Any) -> Tuple[bool, List[str]]:
        """验证组件"""
        validator = self._validators.get(component_type)
        if validator:
            return validator(component)
        return True, []
    
    def register_template(self, name: str, template: Dict[str, Any]) -> None:
        """注册模板"""
        self._templates[name] = template
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模板"""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self._templates.keys())
    
    def set_config(self, config: FactoryConfig) -> None:
        """设置全局配置"""
        self._config = config
    
    def get_config(self) -> FactoryConfig:
        """获取全局配置"""
        return self._config
    
    def list_factories(self, factory_type: FactoryType = None) -> Dict[str, List[str]]:
        """列出所有工厂"""
        if factory_type:
            return {factory_type.value: list(self._factories.get(factory_type, {}).keys())}
        return {ft.value: list(factories.keys()) for ft, factories in self._factories.items()}


def get_factory_registry() -> FactoryRegistry:
    """获取全局工厂注册表"""
    return FactoryRegistry()


# ==================== 基础工厂类 ====================

class BaseFactory(ABC):
    """工厂基类
    
    所有工厂的基类，定义通用接口。
    """
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or get_factory_registry().get_config()
        self._cache: Dict[str, Any] = {}
        self._metrics = {
            "created": 0,
            "cache_hits": 0,
            "errors": 0
        }
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def create(self, component_type: str, **kwargs) -> Any:
        """创建组件"""
        pass
    
    def _cache_key(self, component_type: str, **kwargs) -> str:
        """生成缓存键"""
        import hashlib
        import json
        key_data = f"{component_type}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.config.enable_caching and key in self._cache:
            with self._lock:
                self._metrics["cache_hits"] += 1
            return self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """设置缓存"""
        if self.config.enable_caching:
            self._cache[key] = value
    
    def _record_creation(self) -> None:
        """记录创建"""
        with self._lock:
            self._metrics["created"] += 1
    
    def _record_error(self) -> None:
        """记录错误"""
        with self._lock:
            self._metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, int]:
        """获取指标"""
        return self._metrics.copy()
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()


# ==================== 节点工厂 ====================

class NodeFactory(BaseFactory):
    """节点工厂
    
    创建各种类型的图节点。
    """
    
    # 节点类型映射
    _node_types: Dict[str, Type[BaseNode]] = {
        "llm": LLMNode,
        "tool": ToolNode,
        "human": HumanNode,
        "transform": TransformNode,
        "branch": BranchNode,
        "parallel": ParallelNode,
        "retry": RetryNode,
        "cache": CacheNode,
        "memory": MemoryNode,
        "reflection": ReflectionNode,
        "planning": PlanningNode,
        "logging": LoggingNode,
        "metrics": MetricsNode,
        "rate_limit": RateLimitNode,
        "validation": ValidationNode,
    }
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        self._node_registry = get_node_registry()
    
    def create(self, component_type: str, **kwargs) -> BaseNode:
        """创建节点
        
        Args:
            component_type: 节点类型
            **kwargs: 节点参数
        
        Returns:
            节点实例
        """
        node_type = component_type.lower().replace("_node", "")
        
        # 检查缓存
        if kwargs.get("use_cache", False):
            cache_key = self._cache_key(node_type, **kwargs)
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        try:
            node_class = self._node_types.get(node_type)
            if not node_class:
                raise ValueError(f"Unknown node type: {node_type}")
            
            # 准备参数
            name = kwargs.pop("name", f"{node_type}_node")
            node_config = kwargs.pop("node_config", None)
            
            # 创建节点
            if node_type == "llm":
                node = self._create_llm_node(name, **kwargs)
            elif node_type == "tool":
                node = self._create_tool_node(name, **kwargs)
            elif node_type == "human":
                node = self._create_human_node(name, **kwargs)
            else:
                node = node_class(name=name, **kwargs)
            
            self._record_creation()
            
            # 缓存
            if kwargs.get("use_cache", False):
                self._set_cached(cache_key, node)
            
            return node
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create node {component_type}: {e}")
            raise
    
    def _create_llm_node(self, name: str, **kwargs) -> LLMNode:
        """创建 LLM 节点"""
        return LLMNode(
            name=name,
            llm_client=kwargs.get("llm_client"),
            system_prompt=kwargs.get("system_prompt"),
            tools=kwargs.get("tools", []),
            temperature=kwargs.get("temperature", self.config.default_temperature),
            model=kwargs.get("model", self.config.default_model),
            max_tokens=kwargs.get("max_tokens", 2048),
            stop_sequences=kwargs.get("stop_sequences"),
            response_format=kwargs.get("response_format")
        )
    
    def _create_tool_node(self, name: str, **kwargs) -> ToolNode:
        """创建工具节点"""
        registry = kwargs.get("registry") or get_global_registry()
        return ToolNode(
            name=name,
            registry=registry,
            parallel_execution=kwargs.get("parallel_execution", True),
            max_parallel=kwargs.get("max_parallel", 4),
            timeout=kwargs.get("timeout", 30.0)
        )
    
    def _create_human_node(self, name: str, **kwargs) -> HumanNode:
        """创建人工节点"""
        return HumanNode(
            name=name,
            prompt_template=kwargs.get("prompt_template"),
            timeout=kwargs.get("timeout", 300.0),
            default_response=kwargs.get("default_response")
        )
    
    def create_llm_node(self, name: str = "llm", **kwargs) -> LLMNode:
        """创建 LLM 节点的便捷方法"""
        return self.create("llm", name=name, **kwargs)
    
    def create_tool_node(self, name: str = "tools", **kwargs) -> ToolNode:
        """创建工具节点的便捷方法"""
        return self.create("tool", name=name, **kwargs)
    
    def create_human_node(self, name: str = "human", **kwargs) -> HumanNode:
        """创建人工节点的便捷方法"""
        return self.create("human", name=name, **kwargs)
    
    def create_transform_node(self, name: str, 
                             transform_func: Callable[[AgentState], AgentState],
                             **kwargs) -> TransformNode:
        """创建转换节点"""
        return TransformNode(name=name, transform_func=transform_func, **kwargs)
    
    def create_branch_node(self, name: str,
                          condition_func: Callable[[AgentState], str],
                          branches: Dict[str, str],
                          **kwargs) -> BranchNode:
        """创建分支节点"""
        return BranchNode(
            name=name, 
            condition_func=condition_func,
            branches=branches,
            **kwargs
        )
    
    def create_parallel_node(self, name: str,
                            nodes: List[BaseNode],
                            merge_strategy: str = "merge",
                            **kwargs) -> ParallelNode:
        """创建并行节点"""
        return ParallelNode(
            name=name,
            nodes=nodes,
            merge_strategy=merge_strategy,
            **kwargs
        )
    
    def create_retry_node(self, name: str,
                         inner_node: BaseNode,
                         max_retries: int = 3,
                         **kwargs) -> RetryNode:
        """创建重试节点"""
        return RetryNode(
            name=name,
            inner_node=inner_node,
            max_retries=max_retries,
            **kwargs
        )
    
    def create_cache_node(self, name: str,
                         inner_node: BaseNode,
                         ttl: int = 3600,
                         **kwargs) -> CacheNode:
        """创建缓存节点"""
        return CacheNode(
            name=name,
            inner_node=inner_node,
            ttl=ttl,
            **kwargs
        )
    
    def create_node_chain(self, nodes: List[BaseNode]) -> NodeChain:
        """创建节点链"""
        return NodeChain(nodes=nodes)
    
    def create_parallel_group(self, nodes: List[BaseNode],
                             merge_strategy: str = "merge") -> NodeParallelGroup:
        """创建并行节点组"""
        return NodeParallelGroup(nodes=nodes, merge_strategy=merge_strategy)
    
    def register_node_type(self, name: str, node_class: Type[BaseNode]) -> None:
        """注册新的节点类型"""
        self._node_types[name.lower()] = node_class
    
    def list_node_types(self) -> List[str]:
        """列出所有节点类型"""
        return list(self._node_types.keys())


# ==================== 边工厂 ====================

class EdgeFactory(BaseFactory):
    """边工厂
    
    创建各种类型的图边。
    """
    
    # 边类型映射
    _edge_types: Dict[str, Type[Edge]] = {
        "simple": Edge,
        "conditional": ConditionalEdge,
        "loop": LoopEdge,
        "parallel": ParallelEdge,
        "retry": RetryEdge,
        "fallback": FallbackEdge,
        "timeout": TimeoutEdge,
        "circuit_breaker": CircuitBreakerEdge,
        "weighted": WeightedEdge,
    }
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        self._edge_manager = EdgeManager()
    
    def create(self, component_type: str, **kwargs) -> Edge:
        """创建边
        
        Args:
            component_type: 边类型
            **kwargs: 边参数
            
        Returns:
            边实例
        """
        edge_type = component_type.lower().replace("_edge", "")
        
        try:
            edge_class = self._edge_types.get(edge_type)
            if not edge_class:
                raise ValueError(f"Unknown edge type: {edge_type}")
            
            # 创建边
            source = kwargs.pop("source")
            target = kwargs.pop("target", "__end__")
            
            if edge_type == "conditional":
                edge = self._create_conditional_edge(source, target, **kwargs)
            elif edge_type == "retry":
                edge = self._create_retry_edge(source, target, **kwargs)
            elif edge_type == "fallback":
                edge = self._create_fallback_edge(source, target, **kwargs)
            elif edge_type == "circuit_breaker":
                edge = self._create_circuit_breaker_edge(source, target, **kwargs)
            elif edge_type == "weighted":
                edge = self._create_weighted_edge(source, **kwargs)
            else:
                edge = edge_class(source=source, target=target, **kwargs)
            
            self._record_creation()
            return edge
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create edge {component_type}: {e}")
            raise
    
    def _create_conditional_edge(self, source: str, target: str, **kwargs) -> ConditionalEdge:
        """创建条件边"""
        return ConditionalEdge(
            source=source,
            target=target,
            condition_func=kwargs.get("condition_func"),
            branches=kwargs.get("branches", {}),
            condition=kwargs.get("condition")
        )
    
    def _create_retry_edge(self, source: str, target: str, **kwargs) -> RetryEdge:
        """创建重试边"""
        return RetryEdge(
            source=source,
            target=target,
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 1.0),
            backoff_multiplier=kwargs.get("backoff_multiplier", 2.0),
            fallback_target=kwargs.get("fallback_target", "")
        )
    
    def _create_fallback_edge(self, source: str, target: str, **kwargs) -> FallbackEdge:
        """创建降级边"""
        return FallbackEdge(
            source=source,
            target=target,
            fallback_targets=kwargs.get("fallback_targets", []),
            fallback_conditions=kwargs.get("fallback_conditions", {})
        )
    
    def _create_circuit_breaker_edge(self, source: str, target: str, 
                                    **kwargs) -> CircuitBreakerEdge:
        """创建断路器边"""
        return CircuitBreakerEdge(
            source=source,
            target=target,
            failure_threshold=kwargs.get("failure_threshold", 5),
            recovery_timeout=kwargs.get("recovery_timeout", 30.0),
            fallback_target=kwargs.get("fallback_target", "")
        )
    
    def _create_weighted_edge(self, source: str, **kwargs) -> WeightedEdge:
        """创建权重边"""
        return WeightedEdge(
            source=source,
            target=kwargs.get("target", "__end__"),
            weighted_targets=kwargs.get("weighted_targets", {}),
            routing_strategy=kwargs.get("routing_strategy", RoutingStrategy.WEIGHTED_RANDOM)
        )
    
    def create_simple_edge(self, source: str, target: str, **kwargs) -> Edge:
        """创建简单边"""
        return self.create("simple", source=source, target=target, **kwargs)
    
    def create_conditional_edge(self, source: str,
                               condition_func: Callable[[AgentState], str],
                               branches: Dict[str, str],
                               **kwargs) -> ConditionalEdge:
        """创建条件边"""
        return self.create(
            "conditional",
            source=source,
            condition_func=condition_func,
            branches=branches,
            **kwargs
        )
    
    def create_loop_edge(self, source: str, target: str,
                        max_iterations: int = 10,
                        continue_condition: Callable[[AgentState], bool] = None,
                        **kwargs) -> LoopEdge:
        """创建循环边"""
        return LoopEdge(
            source=source,
            target=target,
            max_iterations=max_iterations,
            continue_condition=continue_condition,
            **kwargs
        )
    
    def create_parallel_edge(self, source: str, targets: List[str],
                            **kwargs) -> ParallelEdge:
        """创建并行边"""
        return ParallelEdge(
            source=source,
            targets=targets,
            **kwargs
        )
    
    def create_retry_edge(self, source: str, target: str,
                         max_retries: int = 3,
                         **kwargs) -> RetryEdge:
        """创建重试边"""
        return self.create(
            "retry",
            source=source,
            target=target,
            max_retries=max_retries,
            **kwargs
        )
    
    def create_timeout_edge(self, source: str, target: str,
                           timeout_seconds: float = 30.0,
                           timeout_target: str = "__timeout__",
                           **kwargs) -> TimeoutEdge:
        """创建超时边"""
        return TimeoutEdge(
            source=source,
            target=target,
            timeout_seconds=timeout_seconds,
            timeout_target=timeout_target,
            **kwargs
        )
    
    def create_fallback_edge(self, source: str, target: str,
                            fallback_targets: List[str],
                            **kwargs) -> FallbackEdge:
        """创建降级边"""
        return self.create(
            "fallback",
            source=source,
            target=target,
            fallback_targets=fallback_targets,
            **kwargs
        )
    
    def create_circuit_breaker_edge(self, source: str, target: str,
                                   failure_threshold: int = 5,
                                   **kwargs) -> CircuitBreakerEdge:
        """创建断路器边"""
        return self.create(
            "circuit_breaker",
            source=source,
            target=target,
            failure_threshold=failure_threshold,
            **kwargs
        )
    
    def get_builder(self, source: str) -> EdgeBuilder:
        """获取边构建器"""
        return edge_from(source)
    
    def get_edge_manager(self) -> EdgeManager:
        """获取边管理器"""
        return self._edge_manager
    
    def register_edge_type(self, name: str, edge_class: Type[Edge]) -> None:
        """注册新的边类型"""
        self._edge_types[name.lower()] = edge_class
    
    def list_edge_types(self) -> List[str]:
        """列出所有边类型"""
        return list(self._edge_types.keys())
    
    # 预定义条件
    @staticmethod
    def condition_has_tool_calls() -> Callable[[AgentState], bool]:
        """工具调用条件"""
        return EdgeConditions.has_tool_calls
    
    @staticmethod
    def condition_has_final_answer() -> Callable[[AgentState], bool]:
        """最终答案条件"""
        return EdgeConditions.has_final_answer
    
    @staticmethod
    def condition_should_continue() -> Callable[[AgentState], bool]:
        """继续执行条件"""
        return EdgeConditions.should_continue
    
    @staticmethod
    def condition_max_iterations(n: int) -> Callable[[AgentState], bool]:
        """最大迭代条件"""
        return EdgeConditions.iteration_less_than(n)


# ==================== 图工厂 ====================

class GraphFactory(BaseFactory):
    """图工厂
    
    创建各种类型的执行图。
    """
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        self._node_factory = NodeFactory(config)
        self._edge_factory = EdgeFactory(config)
        self._graph_runner = GraphRunner()
    
    def create(self, component_type: str, **kwargs) -> Union[StateGraph, CompiledGraph]:
        """创建图
        
        Args:
            component_type: 图类型 (react, branching, simple, custom)
            **kwargs: 图参数
            
        Returns:
            图实例
        """
        graph_type = component_type.lower()
        
        try:
            if graph_type == "react":
                graph = self._create_react_graph(**kwargs)
            elif graph_type == "branching":
                graph = self._create_branching_graph(**kwargs)
            elif graph_type == "simple":
                graph = self._create_simple_graph(**kwargs)
            elif graph_type == "parallel":
                graph = self._create_parallel_graph(**kwargs)
            elif graph_type == "loop":
                graph = self._create_loop_graph(**kwargs)
            else:
                graph = self._create_custom_graph(**kwargs)
            
            self._record_creation()
            return graph
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create graph {component_type}: {e}")
            raise
    
    def _create_react_graph(self, **kwargs) -> CompiledGraph:
        """创建 ReAct 图"""
        llm_node = kwargs.get("llm_node") or self._node_factory.create_llm_node(
            name="agent",
            llm_client=kwargs.get("llm_client"),
            system_prompt=kwargs.get("system_prompt"),
            tools=kwargs.get("tools", []),
            model=kwargs.get("model", self.config.default_model),
            temperature=kwargs.get("temperature", self.config.default_temperature)
        )
        
        tool_node = kwargs.get("tool_node") or self._node_factory.create_tool_node(
            name="tools",
            registry=kwargs.get("tool_registry")
        )
        
        should_continue = kwargs.get("should_continue") or route_after_agent
        
        return create_react_graph(
            llm_node=llm_node,
            tool_node=tool_node,
            should_continue=should_continue,
            config=kwargs.get("graph_config")
        )
    
    def _create_branching_graph(self, **kwargs) -> CompiledGraph:
        """创建分支图"""
        entry_node = kwargs.get("entry_node")
        condition = kwargs.get("condition")
        branches = kwargs.get("branches", {})
        merge_node = kwargs.get("merge_node")
        
        config = kwargs.get("graph_config") or GraphConfig(
            name=kwargs.get("name", "branching_graph"),
            max_iterations=kwargs.get("max_iterations", self.config.default_max_iterations)
        )
        
        builder = GraphBuilder(name=config.name, config=config)
        
        # 添加入口节点
        if entry_node:
            entry_name, entry = entry_node
            builder.add_node(entry_name, entry)
            builder.set_entry_point(entry_name)
        
        # 添加分支
        branch_targets = {}
        for branch_name, branch_nodes in branches.items():
            for node_name, node in branch_nodes:
                builder.add_node(node_name, node)
            if branch_nodes:
                branch_targets[branch_name] = branch_nodes[0][0]
        
        # 添加条件边
        if entry_node and condition:
            builder.add_conditional_edges(entry_node[0], condition, branch_targets)
        
        # 添加合并节点
        if merge_node:
            merge_name, merge = merge_node
            builder.add_node(merge_name, merge)
            builder.set_finish_point(merge_name)
            
            # 连接分支到合并节点
            for branch_nodes in branches.values():
                if branch_nodes:
                    last_node = branch_nodes[-1][0]
                    builder.add_edge(last_node, merge_name)
        
        return builder.compile(kwargs.get("checkpointer"))
    
    def _create_simple_graph(self, **kwargs) -> CompiledGraph:
        """创建简单顺序图"""
        nodes = kwargs.get("nodes", [])
        config = kwargs.get("graph_config")
        
        return create_simple_graph(nodes=nodes, config=config)
    
    def _create_parallel_graph(self, **kwargs) -> CompiledGraph:
        """创建并行图"""
        parallel_nodes = kwargs.get("parallel_nodes", [])
        merge_func = kwargs.get("merge_func")
        
        config = kwargs.get("graph_config") or GraphConfig(
            name=kwargs.get("name", "parallel_graph")
        )
        
        builder = GraphBuilder(name=config.name, config=config)
        
        # 添加入口节点
        entry_name = kwargs.get("entry_name", "entry")
        entry_func = kwargs.get("entry_func", lambda s: s)
        builder.add_node(entry_name, entry_func)
        builder.set_entry_point(entry_name)
        
        # 添加并行节点组
        parallel_group = [(name, node) for name, node in parallel_nodes]
        builder.parallel("parallel_group", parallel_group, merge_func)
        
        # 连接
        builder.add_edge(entry_name, "parallel_group_start")
        
        return builder.compile(kwargs.get("checkpointer"))
    
    def _create_loop_graph(self, **kwargs) -> CompiledGraph:
        """创建循环图"""
        body_nodes = kwargs.get("body_nodes", [])
        condition = kwargs.get("condition")
        max_iterations = kwargs.get("max_iterations", 10)
        
        config = kwargs.get("graph_config") or GraphConfig(
            name=kwargs.get("name", "loop_graph"),
            max_iterations=max_iterations
        )
        
        builder = GraphBuilder(name=config.name, config=config)
        builder.loop_while(condition, body_nodes, max_iterations)
        
        return builder.compile(kwargs.get("checkpointer"))
    
    def _create_custom_graph(self, **kwargs) -> CompiledGraph:
        """创建自定义图"""
        nodes = kwargs.get("nodes", [])
        edges = kwargs.get("edges", [])
        conditional_edges = kwargs.get("conditional_edges", {})
        entry_point = kwargs.get("entry_point")
        finish_points = kwargs.get("finish_points", [])
        
        config = kwargs.get("graph_config") or GraphConfig(
            name=kwargs.get("name", "custom_graph"),
            max_iterations=kwargs.get("max_iterations", self.config.default_max_iterations)
        )
        
        builder = GraphBuilder(name=config.name, config=config)
        
        # 添加节点
        for node_name, node in nodes:
            builder.add_node(node_name, node)
        
        # 添加边
        for source, target in edges:
            builder.add_edge(source, target)
        
        # 添加条件边
        for source, (condition_func, branches) in conditional_edges.items():
            builder.add_conditional_edges(source, condition_func, branches)
        
        # 设置入口点
        if entry_point:
            builder.set_entry_point(entry_point)
        elif nodes:
            builder.set_entry_point(nodes[0][0])
        
        # 设置结束点
        for fp in finish_points:
            builder.set_finish_point(fp)
        
        return builder.compile(kwargs.get("checkpointer"))
    
    def create_react_graph(self, **kwargs) -> CompiledGraph:
        """创建 ReAct 图的便捷方法"""
        return self.create("react", **kwargs)
    
    def create_simple_graph(self, nodes: List[Tuple[str, Union[BaseNode, Callable]]],
                           **kwargs) -> CompiledGraph:
        """创建简单图的便捷方法"""
        return self.create("simple", nodes=nodes, **kwargs)
    
    def get_builder(self, name: str = "default", 
                   config: GraphConfig = None) -> GraphBuilder:
        """获取图构建器"""
        return GraphBuilder(name=name, config=config)
    
    def get_graph_runner(self) -> GraphRunner:
        """获取图运行器"""
        return self._graph_runner
    
    def get_analyzer(self, graph: Union[StateGraph, CompiledGraph]) -> GraphAnalyzer:
        """获取图分析器"""
        return GraphAnalyzer(graph)
    
    def create_subgraph(self, name: str, 
                       graph: Union[StateGraph, CompiledGraph],
                       **kwargs) -> Subgraph:
        """创建子图"""
        return Subgraph(
            name=name,
            graph=graph,
            input_mapper=kwargs.get("input_mapper"),
            output_mapper=kwargs.get("output_mapper"),
            condition=kwargs.get("condition")
        )


# ==================== 工具工厂 ====================

class ToolFactory(BaseFactory):
    """工具工厂
    
    创建和管理工具。
    """
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        self._registry = get_global_registry()
        self._manager = get_tool_manager()
        
        # 自动加载内置工具
        if self.config.auto_load_builtin_tools:
            self._load_builtin_tools()
    
    def _load_builtin_tools(self) -> None:
        """加载内置工具"""
        categories = self.config.tool_categories
        
        if not categories:
            # 加载所有内置工具
            tools = get_builtin_tools()
        else:
            # 加载指定类别的工具
            tools = []
            for cat in categories:
                try:
                    cat_enum = BuiltinToolCategory(cat)
                    tools.extend(get_tools_by_category(cat_enum))
                except ValueError:
                    self.logger.warning(f"Unknown tool category: {cat}")
        
        for t in tools:
            if not self._registry.get(t.name):
                self._registry.register(t)
        
        self.logger.info(f"Loaded {len(tools)} builtin tools")
    
    def create(self, component_type: str, **kwargs) -> Tool:
        """创建工具
        
        Args:
            component_type: 工具类型或名称
            **kwargs: 工具参数
            
        Returns:
            工具实例
        """
        try:
            # 尝试获取内置工具
            builtin = get_tool_by_name(component_type)
            if builtin:
                self._record_creation()
                return builtin
            
            # 创建自定义工具
            return self._create_custom_tool(component_type, **kwargs)
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create tool {component_type}: {e}")
            raise
    
    def _create_custom_tool(self, name: str, **kwargs) -> Tool:
        """创建自定义工具"""
        func = kwargs.get("func")
        if not func:
            raise ValueError("Tool function is required")
        
        description = kwargs.get("description", f"Custom tool: {name}")
        parameters = kwargs.get("parameters", [])
        category = kwargs.get("category", ToolCategory.CUSTOM)
        timeout = kwargs.get("timeout", 30.0)
        
        # 使用装饰器创建工具
        tool_decorator = tool(
            name=name,
            description=description,
            category=category,
            timeout=timeout
        )
        
        tool_instance = tool_decorator(func)
        
        # 注册
        if kwargs.get("register", True):
            self._registry.register(tool_instance)
        
        self._record_creation()
        return tool_instance
    
    def create_tool(self, name: str, func: Callable,
                   description: str = None,
                   **kwargs) -> Tool:
        """创建工具的便捷方法"""
        return self.create(
            name,
            func=func,
            description=description or f"Tool: {name}",
            **kwargs
        )
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._registry.get(name) or get_tool_by_name(name)
    
    def get_tools(self, names: List[str] = None) -> List[Tool]:
        """获取多个工具"""
        if names:
            return [self.get_tool(name) for name in names if self.get_tool(name)]
        return list(self._registry.get_all().values())
    
    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """获取适合特定 Agent 的工具"""
        return get_tools_for_agent(agent_type)
    
    def get_search_tools(self) -> List[Tool]:
        """获取搜索工具"""
        return get_search_tools()
    
    def get_code_tools(self) -> List[Tool]:
        """获取代码工具"""
        return get_code_tools()
    
    def get_data_tools(self) -> List[Tool]:
        """获取数据工具"""
        return get_data_tools()
    
    def get_system_tools(self) -> List[Tool]:
        """获取系统工具"""
        return get_system_tools()
    
    def get_http_tools(self) -> List[Tool]:
        """获取 HTTP 工具"""
        return get_http_tools()
    
    def get_memory_tools(self) -> List[Tool]:
        """获取记忆工具"""
        return get_memory_tools()
    
    def get_knowledge_tools(self) -> List[Tool]:
        """获取知识库工具"""
        return get_knowledge_tools()
    
    def get_text_tools(self) -> List[Tool]:
        """获取文本处理工具"""
        return get_text_tools()
    
    def get_file_tools(self) -> List[Tool]:
        """获取文件工具"""
        return get_file_tools()
    
    def get_registry(self) -> ToolRegistry:
        """获取工具注册表"""
        return self._registry
    
    def get_manager(self) -> ToolManager:
        """获取工具管理器"""
        return self._manager
    
    def get_executor(self, registry: ToolRegistry = None) -> ToolExecutor:
        """获取工具执行器"""
        return ToolExecutor(registry or self._registry)
    
    def register_tool(self, tool_instance: Tool) -> None:
        """注册工具"""
        self._registry.register(tool_instance)
    
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        return self._registry.unregister(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._registry.get_all().keys())
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return get_tool_info()
    
    def get_tool_schemas(self, format: str = "openai") -> List[Dict[str, Any]]:
        """获取工具 Schema"""
        return self._registry.get_tool_schemas(format)


# ==================== 检查点工厂 ====================

class CheckpointerFactory(BaseFactory):
    """检查点工厂
    
    创建各种类型的检查点器。
    """
    
    # 检查点类型映射
    _checkpointer_types: Dict[str, Callable[..., Checkpointer]] = {
        "memory": create_memory_checkpointer,
        "redis": create_redis_checkpointer,
        "sqlite": create_sqlite_checkpointer,
        "file": create_file_checkpointer,
    }
    
    def create(self, component_type: str, **kwargs) -> Checkpointer:
        """创建检查点器
        
        Args:
            component_type: 检查点类型 (memory, redis, sqlite, file)
            **kwargs: 检查点器参数
            
        Returns:
            检查点器实例
        """
        cp_type = component_type.lower().replace("_checkpointer", "")
        
        try:
            creator = self._checkpointer_types.get(cp_type)
            if not creator:
                raise ValueError(f"Unknown checkpointer type: {cp_type}")
            
            # 应用默认压缩
            if "compression" not in kwargs:
                kwargs["compression"] = CompressionType.GZIP
            
            checkpointer = creator(**kwargs)
            self._record_creation()
            return checkpointer
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create checkpointer {component_type}: {e}")
            raise
    
    def create_memory_checkpointer(self, max_checkpoints: int = 100,
                                  **kwargs) -> MemoryCheckpointer:
        """创建内存检查点器"""
        return self.create("memory", max_checkpoints=max_checkpoints, **kwargs)
    
    def create_redis_checkpointer(self, host: str = "localhost",
                                 port: int = 6379,
                                 **kwargs) -> RedisCheckpointer:
        """创建 Redis 检查点器"""
        return self.create("redis", host=host, port=port, **kwargs)
    
    def create_sqlite_checkpointer(self, db_path: str = "checkpoints.db",
                                  **kwargs) -> SQLiteCheckpointer:
        """创建 SQLite 检查点器"""
        return self.create("sqlite", db_path=db_path, **kwargs)
    
    def create_file_checkpointer(self, base_dir: str = "./checkpoints",
                                **kwargs) -> FileCheckpointer:
        """创建文件检查点器"""
        return self.create("file", base_dir=base_dir, **kwargs)
    
    def get_default_checkpointer(self) -> Checkpointer:
        """获取默认检查点器"""
        cp_type = self.config.default_checkpointer_type
        cp_config = self.config.checkpointer_config
        return self.create(cp_type, **cp_config)
    
    def register_checkpointer_type(self, name: str, 
                                  creator: Callable[..., Checkpointer]) -> None:
        """注册新的检查点类型"""
        self._checkpointer_types[name.lower()] = creator
    
    def list_checkpointer_types(self) -> List[str]:
        """列出所有检查点类型"""
        return list(self._checkpointer_types.keys())


# ==================== 策略工厂 ====================

class StrategyFactory(BaseFactory):
    """策略工厂
    
    创建各种执行策略。
    """
    
    # 策略类型映射
    _strategy_types: Dict[StrategyType, Type[BaseStrategy]] = {
        StrategyType.STATE: DefaultStateStrategy,
        StrategyType.NODE: DefaultNodeStrategy,
        StrategyType.EDGE: DefaultEdgeStrategy,
        StrategyType.GRAPH: DefaultGraphStrategy,
        StrategyType.TOOL: DefaultToolStrategy,
        StrategyType.CHECKPOINT: DefaultCheckpointStrategy,
        StrategyType.MEMORY: DefaultMemoryStrategy,
        StrategyType.ROUTING: DefaultRoutingStrategy,
    }
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        self._strategy_manager = get_strategy_manager()
    
    def create(self, component_type: str, **kwargs) -> BaseStrategy:
        """创建策略
        
        Args:
            component_type: 策略类型
            **kwargs: 策略参数
            
        Returns:
            策略实例
        """
        try:
            # 解析策略类型
            if isinstance(component_type, str):
                strategy_type = StrategyType(component_type.lower())
            else:
                strategy_type = component_type
            
            strategy_class = self._strategy_types.get(strategy_type)
            if not strategy_class:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            name = kwargs.pop("name", f"{strategy_type.value}_strategy")
            strategy_config = kwargs.pop("config", {})
            
            strategy = strategy_class(name=name, config=strategy_config, **kwargs)
            self._record_creation()
            return strategy
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create strategy {component_type}: {e}")
            raise
    
    def create_state_strategy(self, **kwargs) -> StateStrategy:
        """创建状态策略"""
        return self.create(StrategyType.STATE, **kwargs)
    
    def create_node_strategy(self, **kwargs) -> NodeStrategy:
        """创建节点策略"""
        return self.create(StrategyType.NODE, **kwargs)
    
    def create_edge_strategy(self, **kwargs) -> EdgeStrategy:
        """创建边策略"""
        return self.create(StrategyType.EDGE, **kwargs)
    
    def create_graph_strategy(self, **kwargs) -> GraphStrategy:
        """创建图策略"""
        return self.create(StrategyType.GRAPH, **kwargs)
    
    def create_tool_strategy(self, **kwargs) -> ToolStrategy:
        """创建工具策略"""
        return self.create(StrategyType.TOOL, **kwargs)
    
    def create_checkpoint_strategy(self, **kwargs) -> CheckpointStrategy:
        """创建检查点策略"""
        return self.create(StrategyType.CHECKPOINT, **kwargs)
    
    def create_memory_strategy(self, **kwargs) -> MemoryStrategy:
        """创建记忆策略"""
        return self.create(StrategyType.MEMORY, **kwargs)
    
    def create_routing_strategy(self, **kwargs) -> AgentRoutingStrategy:
        """创建路由策略"""
        return self.create(StrategyType.ROUTING, **kwargs)
    
    def create_strategy_set(self, **kwargs) -> Dict[StrategyType, BaseStrategy]:
        """创建一套完整的策略
        
        Args:
            **kwargs: 各策略类型的配置，如 state={...}, node={...}
            
        Returns:
            策略字典
        """
        strategies = {}
        
        for strategy_type in StrategyType:
            type_config = kwargs.get(strategy_type.value, {})
            strategies[strategy_type] = self.create(strategy_type, **type_config)
        
        return strategies
    
    def get_strategy_manager(self) -> StrategyManager:
        """获取策略管理器"""
        return self._strategy_manager
    
    def register_strategy_type(self, strategy_type: StrategyType,
                              strategy_class: Type[BaseStrategy]) -> None:
        """注册新的策略类型"""
        self._strategy_types[strategy_type] = strategy_class
    
    def list_strategy_types(self) -> List[str]:
        """列出所有策略类型"""
        return [st.value for st in self._strategy_types.keys()]
    
    def configure_strategy_manager(self, strategies: Dict[StrategyType, BaseStrategy]) -> None:
        """配置策略管理器"""
        for strategy_type, strategy in strategies.items():
            self._strategy_manager.register(strategy_type, strategy)


# ==================== 回调工厂 ====================

class CallbackFactory(BaseFactory):
    """回调工厂
    
    创建各种类型的回调。
    """
    
    # 回调类型映射
    _callback_types: Dict[str, Type[AgentCallback]] = {
        "logging": LoggingCallback,
        "metrics": MetricsCallback,
        "streaming": StreamingCallback,
        "webhook": WebhookCallback,
    }
    
    def create(self, component_type: str, **kwargs) -> AgentCallback:
        """创建回调
        
        Args:
            component_type: 回调类型
            **kwargs: 回调参数
            
        Returns:
            回调实例
        """
        cb_type = component_type.lower().replace("_callback", "")
        
        try:
            callback_class = self._callback_types.get(cb_type)
            if not callback_class:
                raise ValueError(f"Unknown callback type: {cb_type}")
            
            callback = callback_class(**kwargs)
            self._record_creation()
            return callback
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create callback {component_type}: {e}")
            raise
    
    def create_logging_callback(self, log_level: int = logging.INFO) -> LoggingCallback:
        """创建日志回调"""
        return self.create("logging", log_level=log_level)
    
    def create_metrics_callback(self) -> MetricsCallback:
        """创建指标回调"""
        return self.create("metrics")
    
    def create_streaming_callback(self, 
                                  on_token: Callable[[str], None] = None,
                                  on_chunk: Callable[[Dict[str, Any]], None] = None
                                  ) -> StreamingCallback:
        """创建流式回调"""
        return self.create("streaming", on_token=on_token, on_chunk=on_chunk)
    
    def create_webhook_callback(self, webhook_url: str, **kwargs) -> WebhookCallback:
        """创建 Webhook 回调"""
        return self.create("webhook", webhook_url=webhook_url, **kwargs)
    
    def create_callback_manager(self, 
                               callbacks: List[AgentCallback] = None) -> CallbackManager:
        """创建回调管理器"""
        return CallbackManager(callbacks or [])
    
    def create_default_callbacks(self) -> List[AgentCallback]:
        """创建默认回调集合"""
        callbacks = []
        
        if self.config.enable_logging:
            callbacks.append(self.create_logging_callback())
        if self.config.enable_metrics:
            callbacks.append(self.create_metrics_callback())
        
        return callbacks
    
    def register_callback_type(self, name: str, 
                              callback_class: Type[AgentCallback]) -> None:
        """注册新的回调类型"""
        self._callback_types[name.lower()] = callback_class
    
    def list_callback_types(self) -> List[str]:
        """列出所有回调类型"""
        return list(self._callback_types.keys())


class AgentFactory(BaseFactory):
    """Agent 工厂 - 生产级实现
    
    提供完整的 Agent 创建、配置和管理功能。
    
    特性：
    - 支持所有内置 Agent 类型
    - 策略模式集成
    - 配置驱动的创建
    - 自动工具加载
    - 回调管理
    - 检查点集成
    """
    
    # Agent 类型映射
    _agent_types: Dict[str, Type[BaseAgent]] = {
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
    
    def __init__(self, config: FactoryConfig = None):
        super().__init__(config)
        
        # 子工厂
        self._node_factory = NodeFactory(config)
        self._edge_factory = EdgeFactory(config)
        self._graph_factory = GraphFactory(config)
        self._tool_factory = ToolFactory(config)
        self._checkpointer_factory = CheckpointerFactory(config)
        self._strategy_factory = StrategyFactory(config)
        self._callback_factory = CallbackFactory(config)
        
        # 全局注册表和池
        self._agent_registry = get_agent_registry()
        self._agent_pool = AgentPool()
        
        # 默认检查点器
        self._default_checkpointer = None
        
    def create(self, component_type: str, **kwargs) -> BaseAgent:
        """创建 Agent
        
        Args:
            component_type: Agent 类型
            **kwargs: Agent 参数
            
        Returns:
            Agent 实例
        """
        agent_type = component_type.lower().replace("_agent", "").replace("agent", "")
        
        try:
            agent_class = self._agent_types.get(agent_type)
            if not agent_class:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # 创建 Agent
            agent = self._create_agent(agent_class, agent_type, **kwargs)
            
            # 配置策略
            if self.config.enable_strategies:
                self._configure_strategies(agent, kwargs.get("strategies", {}))
            
            # 添加回调
            callbacks = kwargs.get("callbacks")
            if callbacks is None and self.config.default_callbacks:
                callbacks = self._callback_factory.create_default_callbacks()
            if callbacks:
                for cb in callbacks:
                    agent.add_callback(cb)
            
            # 注册到池
            if kwargs.get("register_to_pool", True):
                self._agent_pool.register(agent)
            
            # 注册到全局注册表
            if kwargs.get("register_globally", False):
                self._agent_registry.register(agent)
            
            self._record_creation()
            return agent
            
        except Exception as e:
            self._record_error()
            self.logger.error(f"Failed to create agent {component_type}: {e}")
            raise
    
    def _create_agent(self, agent_class: Type[BaseAgent], 
                     agent_type: str, **kwargs) -> BaseAgent:
        """创建 Agent 实例"""
        # 准备配置
        config = kwargs.get("config")
        if not config:
            config = AgentConfig(
                name=kwargs.get("name", f"{agent_type}_agent"),
                max_iterations=kwargs.get("max_iterations", self.config.default_max_iterations),
                model=kwargs.get("model", self.config.default_model),
                temperature=kwargs.get("temperature", self.config.default_temperature),
                timeout=kwargs.get("timeout", self.config.default_timeout),
                system_prompt=kwargs.get("system_prompt")
            )
        
        # 准备工具
        tools = kwargs.get("tools")
        if tools is None and self.config.auto_load_builtin_tools:
            tools = self._tool_factory.get_tools_for_agent(agent_type)
        tools = tools or []
        
        # 准备检查点器
        checkpointer = kwargs.get("checkpointer")
        if checkpointer is None and self.config.enable_caching:
            checkpointer = self._get_default_checkpointer()
        
        # LLM 客户端
        llm_client = kwargs.get("llm_client")
        
        # 根据 Agent 类型创建
        if agent_type == "react":
            return agent_class(
            config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer
            )
        elif agent_type == "plan_execute":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                enable_replanning=kwargs.get("enable_replanning", True)
            )
        elif agent_type == "reflexion":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                max_reflections=kwargs.get("max_reflections", 3)
            )
        elif agent_type == "multi_agent":
            return agent_class(
                config=config,
                roles=kwargs.get("roles", []),
                llm_client=llm_client,
                checkpointer=checkpointer,
                collaboration_mode=kwargs.get("collaboration_mode", "supervisor")
            )
        elif agent_type == "tool_calling":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer
            )
        elif agent_type == "conversational":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                conversation_memory_size=kwargs.get("conversation_memory_size", 50)
            )
        elif agent_type == "chain_of_thought":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer
            )
        elif agent_type == "self_ask":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                max_depth=kwargs.get("max_depth", 3)
            )
        elif agent_type == "hierarchical":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                max_workers=kwargs.get("max_workers", 3)
            )
        elif agent_type == "workflow":
            return agent_class(
                config=config,
                tools=tools,
                llm_client=llm_client,
                checkpointer=checkpointer,
                workflow_definition=kwargs.get("workflow_definition", {})
            )
        else:
            # 通用创建
            return agent_class(
                config=config,
                tools=tools,
            llm_client=llm_client,
            checkpointer=checkpointer
        )
    
    def _configure_strategies(self, agent: BaseAgent, 
                            strategies_config: Dict[str, Any]) -> None:
        """配置 Agent 策略"""
        # 使用默认策略或自定义策略
        for strategy_type in StrategyType:
            type_config = strategies_config.get(strategy_type.value, {})
            strategy = self._strategy_factory.create(strategy_type, **type_config)
            agent.set_strategy(strategy_type, strategy)
    
    def _get_default_checkpointer(self) -> Checkpointer:
        """获取默认检查点器"""
        if self._default_checkpointer is None:
            self._default_checkpointer = self._checkpointer_factory.get_default_checkpointer()
        return self._default_checkpointer
    
    # ==================== 增强配置方法 ====================
    
    def configure_agent_with_builtin_tools(self, agent: BaseAgent,
                                          tool_names: List[str] = None,
                                          categories: List[str] = None) -> BaseAgent:
        """为 Agent 配置内置工具
        
        Args:
            agent: Agent 实例
            tool_names: 工具名称列表
            categories: 工具类别列表
        
        Returns:
            配置后的 Agent
        """
        added_tools = []
        
        # 按名称添加
        if tool_names:
            added = agent.add_builtin_tools(tool_names)
            added_tools.extend(added)
        
        # 按类别添加
        if categories:
            for cat in categories:
                try:
                    cat_enum = BuiltinToolCategory(cat)
                    added = agent.add_builtin_tools_by_category(cat_enum)
                    added_tools.extend(added)
                except ValueError:
                    self.logger.warning(f"Unknown tool category: {cat}")
        
        # 如果没有指定，添加推荐工具
        if not tool_names and not categories:
            added = agent.add_recommended_builtin_tools()
            added_tools.extend(added)
        
        self.logger.info(f"Configured agent {agent.name} with builtin tools: {added_tools}")
        return agent
    
    def configure_agent_checkpointer(self, agent: BaseAgent,
                                    checkpointer_type: str = "memory",
                                    **kwargs) -> BaseAgent:
        """配置 Agent 检查点器
        
        Args:
            agent: Agent 实例
            checkpointer_type: 检查点器类型 (memory/redis/sqlite/file)
            **kwargs: 检查点器参数
        
        Returns:
            配置后的 Agent
        """
        if checkpointer_type == "redis":
            agent.set_redis_checkpointer(**kwargs)
        elif checkpointer_type == "sqlite":
            agent.set_sqlite_checkpointer(**kwargs)
        elif checkpointer_type == "file":
            agent.set_file_checkpointer(**kwargs)
        else:
            # memory 是默认的
            pass
        
        self.logger.info(f"Configured agent {agent.name} with {checkpointer_type} checkpointer")
        return agent
    
    def configure_agent_execution_context(self, agent: BaseAgent,
                                         priority: PriorityLevel = PriorityLevel.NORMAL,
                                         execution_mode: StateExecutionMode = StateExecutionMode.ASYNC,
                                         timeout: float = None,
                                         **metadata) -> StateExecutionContext:
        """为 Agent 创建执行上下文
        
        Args:
            agent: Agent 实例
            priority: 优先级
            execution_mode: 执行模式
            timeout: 超时时间
            **metadata: 额外元数据
        
        Returns:
            执行上下文
        """
        return agent.create_execution_context(
            priority=priority,
            execution_mode=execution_mode,
            timeout=timeout or agent.config.timeout,
            metadata=metadata
        )
    
    def configure_agent_with_graph(self, agent: BaseAgent,
                                  graph_type: str = "react",
                                  **graph_kwargs) -> BaseAgent:
        """配置 Agent 的执行图
        
        Args:
            agent: Agent 实例
            graph_type: 图类型
            **graph_kwargs: 图参数
        
        Returns:
            配置后的 Agent
        """
        if graph_type == "simple":
            graph = agent.create_simple_graph(
                nodes=graph_kwargs.get("nodes", []),
                config=graph_kwargs.get("config")
            )
        elif graph_type == "react":
            graph = agent.create_react_graph(
                llm_node=graph_kwargs.get("llm_node"),
                tool_node=graph_kwargs.get("tool_node"),
                config=graph_kwargs.get("config")
            )
        else:
            self.logger.warning(f"Unknown graph type: {graph_type}")
            return agent
        
        self.logger.info(f"Configured agent {agent.name} with {graph_type} graph")
        return agent
    
    def get_agent_metrics(self, agent: BaseAgent) -> Dict[str, Any]:
        """获取 Agent 的详细指标
        
        Args:
            agent: Agent 实例
        
        Returns:
            指标字典
        """
        metrics = agent.get_metrics()
        
        # 获取图指标
        try:
            graph_metrics = agent.get_graph_metrics()
            metrics["graph"] = {
                "total_executions": graph_metrics.total_executions,
                "successful_executions": graph_metrics.successful_executions,
                "failed_executions": graph_metrics.failed_executions,
                "avg_execution_time": graph_metrics.avg_execution_time
            }
        except Exception:
            pass
        
        # 获取图状态
        try:
            graph_status = agent.get_graph_status()
            metrics["graph_status"] = graph_status.value
        except Exception:
            pass
        
        return metrics
    
    def get_agent_tool_metrics(self, agent: BaseAgent, 
                              tool_name: str = None) -> Dict[str, Any]:
        """获取 Agent 的工具指标
        
        Args:
            agent: Agent 实例
            tool_name: 工具名称（可选，不指定则返回所有）
        
        Returns:
            工具指标字典
        """
        if tool_name:
            try:
                metrics = agent.get_tool_metrics_by_name(tool_name)
                return {
                    "name": tool_name,
                    "total_calls": metrics.total_calls,
                    "success_calls": metrics.success_calls,
                    "failed_calls": metrics.failed_calls,
                    "avg_duration": metrics.avg_duration
                }
            except Exception as e:
                return {"error": str(e)}
        else:
            # 获取所有工具指标
            all_metrics = {}
            for t in agent.tools:
                try:
                    m = agent.get_tool_metrics_by_name(t.name)
                    all_metrics[t.name] = {
                        "total_calls": m.total_calls,
                        "success_calls": m.success_calls,
                        "failed_calls": m.failed_calls
                    }
                except Exception:
                    all_metrics[t.name] = {"error": "No metrics available"}
            return all_metrics
    
    def create_agent_plan(self, agent: BaseAgent,
                         goal: str,
                         steps: List[str],
                         context: Dict[str, Any] = None) -> AgentPlan:
        """使用 Agent 创建执行计划
        
        Args:
            agent: Agent 实例
            goal: 目标描述
            steps: 步骤列表
            context: 上下文信息
        
        Returns:
            AgentPlan 实例
        """
        return agent.create_plan(goal=goal, steps=steps, context=context)
    
    def create_agent_reflection(self, agent: BaseAgent,
                               content: str,
                               quality_score: float = 0.0,
                               suggestions: List[str] = None) -> Reflection:
        """使用 Agent 创建反思
        
        Args:
            agent: Agent 实例
            content: 反思内容
            quality_score: 质量分数
            suggestions: 建议列表
        
        Returns:
            Reflection 实例
        """
        return agent.create_reflection(
            content=content,
            quality_score=quality_score,
            suggestions=suggestions
        )
    
    def create_agent_state_checkpoint(self, agent: BaseAgent,
                                     state: AgentState,
                                     description: str = "") -> StateCheckpoint:
        """使用 Agent 创建状态检查点
        
        Args:
            agent: Agent 实例
            state: 状态
            description: 描述
        
        Returns:
            StateCheckpoint 实例
        """
        return agent.create_state_checkpoint(state, description)
    
    # ==================== 增强构建器方法 ====================
    
    def get_enhanced_builder(self, agent_type: str = "react") -> 'EnhancedAgentBuilder':
        """获取增强的 Agent 构建器
        
        Args:
            agent_type: Agent 类型
        
        Returns:
            EnhancedAgentBuilder 实例
        """
        return EnhancedAgentBuilder(agent_type)
    
    def create_with_enhanced_features(self, agent_type: str,
                                     enable_builtin_tools: bool = True,
                                     enable_metrics: bool = True,
                                     enable_logging: bool = True,
                                     checkpointer_type: str = "memory",
                                     **kwargs) -> BaseAgent:
        """创建具有增强功能的 Agent
        
        Args:
            agent_type: Agent 类型
            enable_builtin_tools: 是否启用内置工具
            enable_metrics: 是否启用指标收集
            enable_logging: 是否启用日志
            checkpointer_type: 检查点器类型
            **kwargs: 其他参数
        
        Returns:
            增强的 Agent 实例
        """
        # 创建基础 Agent
        agent = self.create(agent_type, **kwargs)
        
        # 配置内置工具
        if enable_builtin_tools:
            self.configure_agent_with_builtin_tools(agent)
        
        # 配置检查点器
        self.configure_agent_checkpointer(agent, checkpointer_type)
        
        # 添加回调
        if enable_metrics:
            agent.add_callback(MetricsCallback())
        if enable_logging:
            agent.add_callback(LoggingCallback())
        
        return agent
    
    # ==================== 便捷创建方法 ====================
    
    def create_react_agent(self, name: str = "react_agent",
        tools: List[Tool] = None,
        llm_client: Any = None,
        system_prompt: str = None,
                          max_iterations: int = 10,
                          model: str = "gpt-4",
                          temperature: float = 0.7,
                          checkpointer: Checkpointer = None,
                          **kwargs) -> ReActAgent:
        """创建 ReAct Agent"""
        return self.create(
            "react",
            name=name,
            tools=tools,
            llm_client=llm_client,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model=model,
            temperature=temperature,
            checkpointer=checkpointer,
            **kwargs
        )
    
    def create_plan_execute_agent(self, name: str = "plan_execute_agent",
                                 tools: List[Tool] = None,
                                 llm_client: Any = None,
                                 system_prompt: str = None,
                                 max_iterations: int = 15,
                                 enable_replanning: bool = True,
                                 checkpointer: Checkpointer = None,
                                 **kwargs) -> PlanAndExecuteAgent:
        """创建计划执行 Agent"""
        return self.create(
            "plan_execute",
            name=name,
            tools=tools,
            llm_client=llm_client,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            enable_replanning=enable_replanning,
            checkpointer=checkpointer,
            **kwargs
        )
    
    def create_reflexion_agent(self, name: str = "reflexion_agent",
        tools: List[Tool] = None,
        llm_client: Any = None,
        max_reflections: int = 3,
                              checkpointer: Checkpointer = None,
                              **kwargs) -> ReflexionAgent:
        """创建反思 Agent"""
        return self.create(
            "reflexion",
            name=name,
            tools=tools,
            llm_client=llm_client,
            max_reflections=max_reflections,
            checkpointer=checkpointer,
            **kwargs
        )
    
    def create_multi_agent_system(self, name: str = "multi_agent_system",
        roles: List[AgentRole] = None,
        llm_client: Any = None,
        collaboration_mode: str = "supervisor",
                                 checkpointer: Checkpointer = None,
                                 **kwargs) -> MultiAgentSystem:
        """创建多 Agent 系统"""
        return self.create(
            "multi_agent",
            name=name,
            roles=roles,
            llm_client=llm_client,
            collaboration_mode=collaboration_mode,
            checkpointer=checkpointer,
            **kwargs
        )
    
    def create_tool_calling_agent(self, name: str = "tool_calling_agent",
                                 tools: List[Tool] = None,
                                 llm_client: Any = None,
                                 **kwargs) -> ToolCallingAgent:
        """创建工具调用 Agent"""
        return self.create("tool_calling", name=name, tools=tools, 
                          llm_client=llm_client, **kwargs)
    
    def create_conversational_agent(self, name: str = "conversational_agent",
                                   tools: List[Tool] = None,
                                   llm_client: Any = None,
                                   conversation_memory_size: int = 50,
                                   **kwargs) -> ConversationalAgent:
        """创建对话 Agent"""
        return self.create(
            "conversational",
            name=name,
            tools=tools,
            llm_client=llm_client,
            conversation_memory_size=conversation_memory_size,
            **kwargs
        )
    
    def create_chain_of_thought_agent(self, name: str = "cot_agent",
                                     tools: List[Tool] = None,
                                     llm_client: Any = None,
                                     **kwargs) -> ChainOfThoughtAgent:
        """创建思维链 Agent"""
        return self.create("chain_of_thought", name=name, tools=tools,
                          llm_client=llm_client, **kwargs)
    
    def create_self_ask_agent(self, name: str = "self_ask_agent",
                             tools: List[Tool] = None,
                             llm_client: Any = None,
                             max_depth: int = 3,
                             **kwargs) -> SelfAskAgent:
        """创建自问 Agent"""
        return self.create(
            "self_ask",
            name=name,
            tools=tools,
            llm_client=llm_client,
            max_depth=max_depth,
            **kwargs
        )
    
    def create_hierarchical_agent(self, name: str = "hierarchical_agent",
                                 tools: List[Tool] = None,
                                 llm_client: Any = None,
                                 max_workers: int = 3,
                                 **kwargs) -> HierarchicalAgent:
        """创建层级 Agent"""
        return self.create(
            "hierarchical",
            name=name,
            tools=tools,
            llm_client=llm_client,
            max_workers=max_workers,
            **kwargs
        )
    
    def create_workflow_agent(self, name: str = "workflow_agent",
                             tools: List[Tool] = None,
                             llm_client: Any = None,
                             workflow_definition: Dict[str, Any] = None,
                             **kwargs) -> WorkflowAgent:
        """创建工作流 Agent"""
        return self.create(
            "workflow",
            name=name,
            tools=tools,
            llm_client=llm_client,
            workflow_definition=workflow_definition or {},
            **kwargs
        )
    
    def create_custom_agent(self, name: str,
                           nodes: List[Tuple[str, Any]],
                           edges: List[Tuple[str, str]],
                           conditional_edges: Dict[str, Tuple[Callable, Dict]] = None,
        entry_point: str = None,
        finish_points: List[str] = None,
                           checkpointer: Checkpointer = None) -> CompiledGraph:
        """创建自定义 Agent 图
        
        Args:
            name: Agent 名称
            nodes: 节点列表 [(name, node_or_func), ...]
            edges: 边列表 [(source, target), ...]
            conditional_edges: 条件边 {source: (condition_func, branches)}
            entry_point: 入口点
            finish_points: 结束点列表
            checkpointer: 检查点器
        
        Returns:
            编译后的图
        """
        return self._graph_factory.create(
            "custom",
            name=name,
            nodes=nodes,
            edges=edges,
            conditional_edges=conditional_edges,
            entry_point=entry_point,
            finish_points=finish_points,
            checkpointer=checkpointer or self._get_default_checkpointer()
        )
    
    # ==================== 从模板创建 ====================
    
    def create_from_template(self, template_name: str, **kwargs) -> BaseAgent:
        """从模板创建 Agent
        
        Args:
            template_name: 模板名称
            **kwargs: 覆盖参数
            
        Returns:
            Agent 实例
        """
        template = get_factory_registry().get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        # 合并模板配置和覆盖参数
        merged_config = {**template, **kwargs}
        agent_type = merged_config.pop("agent_type", "react")
        
        return self.create(agent_type, **merged_config)
    
    def create_from_config(self, config: Dict[str, Any]) -> BaseAgent:
        """从配置创建 Agent
        
        Args:
            config: Agent 配置字典
            
        Returns:
            Agent 实例
        """
        agent_type = config.pop("agent_type", "react")
        return self.create(agent_type, **config)
    
    # ==================== 批量创建 ====================
    
    def create_batch(self, configs: List[Dict[str, Any]]) -> List[BaseAgent]:
        """批量创建 Agent
        
        Args:
            configs: 配置列表
            
        Returns:
            Agent 列表
        """
        return [self.create_from_config(config.copy()) for config in configs]
    
    # ==================== Agent 管理 ====================
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取 Agent"""
        return self._agent_pool.get(name) or self._agent_registry.get(name)
    
    def list_agents(self) -> List[str]:
        """列出所有 Agent"""
        pool_agents = set(self._agent_pool.list_agents())
        registry_agents = set(self._agent_registry.list_agents())
        return list(pool_agents | registry_agents)
    
    def get_agent_pool(self) -> AgentPool:
        """获取 Agent 池"""
        return self._agent_pool
    
    def get_agent_registry(self) -> AgentRegistry:
        """获取 Agent 注册表"""
        return self._agent_registry
    
    # ==================== 子工厂访问 ====================
    
    def get_node_factory(self) -> NodeFactory:
        """获取节点工厂"""
        return self._node_factory
    
    def get_edge_factory(self) -> EdgeFactory:
        """获取边工厂"""
        return self._edge_factory
    
    def get_graph_factory(self) -> GraphFactory:
        """获取图工厂"""
        return self._graph_factory
    
    def get_tool_factory(self) -> ToolFactory:
        """获取工具工厂"""
        return self._tool_factory
    
    def get_checkpointer_factory(self) -> CheckpointerFactory:
        """获取检查点工厂"""
        return self._checkpointer_factory
    
    def get_strategy_factory(self) -> StrategyFactory:
        """获取策略工厂"""
        return self._strategy_factory
    
    def get_callback_factory(self) -> CallbackFactory:
        """获取回调工厂"""
        return self._callback_factory
    
    # ==================== Agent 类型注册 ====================
    
    def register_agent_type(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """注册新的 Agent 类型"""
        self._agent_types[name.lower()] = agent_class
    
    def list_agent_types(self) -> List[str]:
        """列出所有 Agent 类型"""
        return list(self._agent_types.keys())
    
    # ==================== 静态工厂方法（向后兼容）====================
    
    @staticmethod
    def quick_create_react(tools: List[Tool] = None,
                          llm_client: Any = None,
                          **kwargs) -> ReActAgent:
        """快速创建 ReAct Agent（静态方法）"""
        factory = AgentFactory()
        return factory.create_react_agent(tools=tools, llm_client=llm_client, **kwargs)
    
    @staticmethod
    def quick_create_plan_execute(tools: List[Tool] = None,
                                 llm_client: Any = None,
                                 **kwargs) -> PlanAndExecuteAgent:
        """快速创建计划执行 Agent（静态方法）"""
        factory = AgentFactory()
        return factory.create_plan_execute_agent(tools=tools, llm_client=llm_client, **kwargs)
    
    @staticmethod
    def quick_create_multi_agent(roles: List[AgentRole] = None,
                                llm_client: Any = None,
                                **kwargs) -> MultiAgentSystem:
        """快速创建多 Agent 系统（静态方法）"""
        factory = AgentFactory()
        return factory.create_multi_agent_system(roles=roles, llm_client=llm_client, **kwargs)


# ==================== Agent 构建器 ====================

class AgentBuilder:
    """Agent 构建器
    
    提供流式 API 构建 Agent。
    
    使用示例:
        agent = (AgentBuilder("my_agent")
            .with_tools([tool1, tool2])
            .with_llm(llm_client)
            .with_system_prompt("You are a helpful assistant.")
            .with_max_iterations(15)
            .with_checkpointer(checkpointer)
            .with_callback(LoggingCallback())
            .build())
    """
    
    def __init__(self, name: str = "agent", agent_type: str = "react"):
        self.name = name
        self.agent_type = agent_type
        self._config: Dict[str, Any] = {"name": name}
        self._tools: List[Tool] = []
        self._callbacks: List[AgentCallback] = []
        self._strategies: Dict[str, Any] = {}
        self._factory = AgentFactory()
    
    def with_type(self, agent_type: str) -> 'AgentBuilder':
        """设置 Agent 类型"""
        self.agent_type = agent_type
        return self
    
    def with_tools(self, tools: List[Tool]) -> 'AgentBuilder':
        """添加工具"""
        self._tools.extend(tools)
        return self
    
    def with_tool(self, tool: Tool) -> 'AgentBuilder':
        """添加单个工具"""
        self._tools.append(tool)
        return self
    
    def with_builtin_tools(self, *categories: str) -> 'AgentBuilder':
        """添加内置工具"""
        for cat in categories:
            tools = get_tools_by_category(BuiltinToolCategory(cat))
            self._tools.extend(tools)
        return self
    
    def with_llm(self, llm_client: Any) -> 'AgentBuilder':
        """设置 LLM 客户端"""
        self._config["llm_client"] = llm_client
        return self
    
    def with_model(self, model: str) -> 'AgentBuilder':
        """设置模型"""
        self._config["model"] = model
        return self
    
    def with_temperature(self, temperature: float) -> 'AgentBuilder':
        """设置温度"""
        self._config["temperature"] = temperature
        return self
    
    def with_system_prompt(self, prompt: str) -> 'AgentBuilder':
        """设置系统提示"""
        self._config["system_prompt"] = prompt
        return self
    
    def with_max_iterations(self, max_iterations: int) -> 'AgentBuilder':
        """设置最大迭代次数"""
        self._config["max_iterations"] = max_iterations
        return self
    
    def with_timeout(self, timeout: float) -> 'AgentBuilder':
        """设置超时"""
        self._config["timeout"] = timeout
        return self
    
    def with_checkpointer(self, checkpointer: Checkpointer) -> 'AgentBuilder':
        """设置检查点器"""
        self._config["checkpointer"] = checkpointer
        return self
    
    def with_memory_checkpointer(self, **kwargs) -> 'AgentBuilder':
        """使用内存检查点器"""
        self._config["checkpointer"] = create_memory_checkpointer(**kwargs)
        return self
    
    def with_redis_checkpointer(self, **kwargs) -> 'AgentBuilder':
        """使用 Redis 检查点器"""
        self._config["checkpointer"] = create_redis_checkpointer(**kwargs)
        return self
    
    def with_sqlite_checkpointer(self, db_path: str = "checkpoints.db", 
                                **kwargs) -> 'AgentBuilder':
        """使用 SQLite 检查点器"""
        self._config["checkpointer"] = create_sqlite_checkpointer(db_path, **kwargs)
        return self
    
    def with_callback(self, callback: AgentCallback) -> 'AgentBuilder':
        """添加回调"""
        self._callbacks.append(callback)
        return self
    
    def with_logging(self, level: int = logging.INFO) -> 'AgentBuilder':
        """添加日志回调"""
        self._callbacks.append(LoggingCallback(log_level=level))
        return self
    
    def with_metrics(self) -> 'AgentBuilder':
        """添加指标回调"""
        self._callbacks.append(MetricsCallback())
        return self
    
    def with_streaming(self, 
                      on_token: Callable[[str], None] = None,
                      on_chunk: Callable[[Dict[str, Any]], None] = None
                      ) -> 'AgentBuilder':
        """添加流式回调"""
        self._callbacks.append(StreamingCallback(on_token=on_token, on_chunk=on_chunk))
        return self
    
    def with_strategy(self, strategy_type: StrategyType, 
                     strategy: BaseStrategy) -> 'AgentBuilder':
        """设置策略"""
        self._strategies[strategy_type] = strategy
        return self
    
    def with_config(self, **kwargs) -> 'AgentBuilder':
        """设置额外配置"""
        self._config.update(kwargs)
        return self
    
    # ==================== 增强方法 ====================
    
    def with_builtin_tools_by_name(self, *names: str) -> 'AgentBuilder':
        """按名称添加内置工具"""
        for name in names:
            tool = get_tool_by_name(name)
            if tool:
                self._tools.append(tool)
        return self
    
    def with_recommended_tools(self) -> 'AgentBuilder':
        """添加推荐的内置工具"""
        recommended = get_tools_for_agent(self.agent_type)
        self._tools.extend(recommended)
        return self
    
    def with_execution_context(self, priority: PriorityLevel = PriorityLevel.NORMAL,
                              execution_mode: StateExecutionMode = StateExecutionMode.ASYNC,
                              **metadata) -> 'AgentBuilder':
        """设置执行上下文"""
        self._config["execution_context"] = {
            "priority": priority,
            "execution_mode": execution_mode,
            "metadata": metadata
        }
        return self
    
    def with_circuit_breaker(self, failure_threshold: int = 5,
                            recovery_timeout: float = 30.0) -> 'AgentBuilder':
        """启用熔断器"""
        self._config["circuit_breaker"] = {
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout
        }
        return self
    
    def with_rate_limiter(self, max_requests: int = 100,
                         time_window: float = 60.0) -> 'AgentBuilder':
        """启用限流器"""
        self._config["rate_limiter"] = {
            "max_requests": max_requests,
            "time_window": time_window
        }
        return self
    
    def with_file_checkpointer(self, base_dir: str = "./checkpoints") -> 'AgentBuilder':
        """使用文件检查点器"""
        self._config["checkpointer"] = create_file_checkpointer(base_dir=base_dir)
        return self
    
    def with_tool_hooks(self, before_call: Callable = None,
                       after_call: Callable = None,
                       on_error: Callable = None) -> 'AgentBuilder':
        """设置工具钩子"""
        self._config["tool_hooks"] = {
            "before_call": before_call,
            "after_call": after_call,
            "on_error": on_error
        }
        return self
    
    def with_human_in_loop(self, enabled: bool = True,
                          approval_timeout: float = 300.0) -> 'AgentBuilder':
        """启用人机协作"""
        self._config["enable_human_in_loop"] = enabled
        self._config["human_approval_timeout"] = approval_timeout
        return self
    
    def with_parallel_tool_execution(self, enabled: bool = True,
                                    max_parallel: int = 4) -> 'AgentBuilder':
        """启用并行工具执行"""
        self._config["parallel_tool_execution"] = enabled
        self._config["max_parallel_tools"] = max_parallel
        return self
    
    def with_plan_config(self, enable_replanning: bool = True,
                        max_replans: int = 3) -> 'AgentBuilder':
        """配置计划执行参数（仅对 plan_execute Agent 有效）"""
        self._config["enable_replanning"] = enable_replanning
        self._config["max_replans"] = max_replans
        return self
    
    def with_reflection_config(self, max_reflections: int = 3,
                              quality_threshold: float = 0.8) -> 'AgentBuilder':
        """配置反思参数（仅对 reflexion Agent 有效）"""
        self._config["max_reflections"] = max_reflections
        self._config["quality_threshold"] = quality_threshold
        return self
    
    def build(self) -> BaseAgent:
        """构建 Agent"""
        self._config["tools"] = self._tools
        self._config["callbacks"] = self._callbacks
        self._config["strategies"] = self._strategies
        
        return self._factory.create(self.agent_type, **self._config)
    
    def build_enhanced(self) -> BaseAgent:
        """构建增强的 Agent
        
        自动配置内置工具、指标和日志
        """
        agent = self.build()
        
        # 配置内置工具
        if not self._tools:
            agent.add_recommended_builtin_tools()
        
        return agent
    
    def build_and_invoke(self, input_data: Union[str, Dict[str, Any]]) -> AgentState:
        """构建并执行 Agent"""
        agent = self.build()
        return agent.invoke(input_data)


# ==================== 流水线构建器 ====================

class PipelineBuilder:
    """流水线构建器
    
    构建多 Agent 流水线。
    
    使用示例:
        pipeline = (PipelineBuilder("research_pipeline")
            .add_stage("research", research_agent)
            .add_stage("analyze", analyze_agent)
            .add_stage("summarize", summary_agent)
            .with_error_handler(error_handler)
            .build())
        
        result = pipeline.run("What is AI?")
    """
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._stages: List[Tuple[str, BaseAgent, Dict[str, Any]]] = []
        self._error_handler: Optional[Callable] = None
        self._input_mapper: Optional[Callable] = None
        self._output_mapper: Optional[Callable] = None
        self._timeout: float = 300.0
        self._checkpointer: Optional[Checkpointer] = None
    
    def add_stage(self, name: str, agent: BaseAgent, 
                 **kwargs) -> 'PipelineBuilder':
        """添加流水线阶段"""
        self._stages.append((name, agent, kwargs))
        return self
    
    def with_error_handler(self, handler: Callable[[Exception, str, AgentState], AgentState]
                          ) -> 'PipelineBuilder':
        """设置错误处理器"""
        self._error_handler = handler
        return self
    
    def with_input_mapper(self, mapper: Callable[[Any], Dict[str, Any]]
                         ) -> 'PipelineBuilder':
        """设置输入映射器"""
        self._input_mapper = mapper
        return self
    
    def with_output_mapper(self, mapper: Callable[[AgentState], Any]
                          ) -> 'PipelineBuilder':
        """设置输出映射器"""
        self._output_mapper = mapper
        return self
    
    def with_timeout(self, timeout: float) -> 'PipelineBuilder':
        """设置超时"""
        self._timeout = timeout
        return self
    
    def with_checkpointer(self, checkpointer: Checkpointer) -> 'PipelineBuilder':
        """设置检查点器"""
        self._checkpointer = checkpointer
        return self
    
    def build(self) -> 'Pipeline':
        """构建流水线"""
        return Pipeline(
            name=self.name,
            stages=self._stages,
            error_handler=self._error_handler,
            input_mapper=self._input_mapper,
            output_mapper=self._output_mapper,
            timeout=self._timeout,
            checkpointer=self._checkpointer
        )


class Pipeline:
    """Agent 流水线
    
    顺序执行多个 Agent 的流水线。
    """
    
    def __init__(self, name: str,
                stages: List[Tuple[str, BaseAgent, Dict[str, Any]]],
                error_handler: Callable = None,
                input_mapper: Callable = None,
                output_mapper: Callable = None,
                timeout: float = 300.0,
                checkpointer: Checkpointer = None):
        self.name = name
        self._stages = stages
        self._error_handler = error_handler
        self._input_mapper = input_mapper
        self._output_mapper = output_mapper
        self._timeout = timeout
        self._checkpointer = checkpointer or MemoryCheckpointer()
        self._execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.Pipeline.{name}")
    
    def run(self, input_data: Any, thread_id: str = None) -> Any:
        """运行流水线
        
        Args:
            input_data: 输入数据
            thread_id: 线程 ID
            
        Returns:
            输出结果
        """
        thread_id = thread_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        # 输入映射
        if self._input_mapper:
            state_data = self._input_mapper(input_data)
        else:
            state_data = input_data if isinstance(input_data, dict) else {"input": input_data}
        
        current_state = AgentState(**state_data, thread_id=thread_id)
        
        self.logger.info(f"Starting pipeline '{self.name}' with thread_id: {thread_id}")
        
        # 执行各阶段
        for stage_name, agent, kwargs in self._stages:
            try:
                self.logger.info(f"Executing stage: {stage_name}")
                stage_start = datetime.now()
                
                # 执行 Agent
                result = agent.invoke(current_state)
                
                # 记录执行历史
                self._execution_history.append({
                    "stage": stage_name,
                    "success": True,
                    "duration": (datetime.now() - stage_start).total_seconds(),
                    "thread_id": thread_id
                })
                
                # 更新状态
                current_state = result
                
                # 保存检查点
                if self._checkpointer:
                    self._checkpointer.save(
                        thread_id=thread_id,
                        state=current_state.to_dict() if hasattr(current_state, 'to_dict') else vars(current_state)
                    )
                
            except Exception as e:
                self.logger.error(f"Stage '{stage_name}' failed: {e}")
                
                # 记录失败
                self._execution_history.append({
                    "stage": stage_name,
                    "success": False,
                    "error": str(e),
                    "thread_id": thread_id
                })
                
                # 错误处理
                if self._error_handler:
                    current_state = self._error_handler(e, stage_name, current_state)
                else:
                    raise
        
        total_duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Pipeline '{self.name}' completed in {total_duration:.2f}s")
        
        # 输出映射
        if self._output_mapper:
            return self._output_mapper(current_state)
        
        return current_state
    
    async def arun(self, input_data: Any, thread_id: str = None) -> Any:
        """异步运行流水线"""
        # 简单实现：在线程池中运行同步版本
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, input_data, thread_id)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self._execution_history.copy()
    
    def clear_history(self) -> None:
        """清除执行历史"""
        self._execution_history.clear()


# ==================== 工作流构建器 ====================

class WorkflowBuilder:
    """工作流构建器
    
    构建复杂的 Agent 工作流，支持条件分支、并行执行等。
    
    使用示例:
        workflow = (WorkflowBuilder("decision_workflow")
            .add_node("classify", classifier_agent)
            .add_node("handle_a", agent_a)
            .add_node("handle_b", agent_b)
            .add_conditional_edge("classify", route_func, {
                "type_a": "handle_a",
                "type_b": "handle_b"
            })
            .set_entry("classify")
            .build())
    """
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self._nodes: Dict[str, Union[BaseAgent, Callable]] = {}
        self._edges: List[Tuple[str, str]] = []
        self._conditional_edges: Dict[str, Tuple[Callable, Dict[str, str]]] = {}
        self._entry_point: Optional[str] = None
        self._finish_points: List[str] = []
        self._checkpointer: Optional[Checkpointer] = None
        self._graph_config: Optional[GraphConfig] = None
    
    def add_node(self, name: str, 
                agent_or_func: Union[BaseAgent, Callable]) -> 'WorkflowBuilder':
        """添加节点"""
        self._nodes[name] = agent_or_func
        return self
    
    def add_edge(self, source: str, target: str) -> 'WorkflowBuilder':
        """添加边"""
        self._edges.append((source, target))
        return self
    
    def add_conditional_edge(self, source: str,
                            condition_func: Callable[[AgentState], str],
                            branches: Dict[str, str]) -> 'WorkflowBuilder':
        """添加条件边"""
        self._conditional_edges[source] = (condition_func, branches)
        return self
    
    def add_parallel_nodes(self, names: List[str],
                          agents_or_funcs: List[Union[BaseAgent, Callable]],
                          merge_node: str = None) -> 'WorkflowBuilder':
        """添加并行节点"""
        for name, agent in zip(names, agents_or_funcs):
            self._nodes[name] = agent
        
        # 如果有合并节点，添加边
        if merge_node:
            for name in names:
                self._edges.append((name, merge_node))
        
        return self
    
    def set_entry(self, node_name: str) -> 'WorkflowBuilder':
        """设置入口点"""
        self._entry_point = node_name
        return self
    
    def set_finish(self, *node_names: str) -> 'WorkflowBuilder':
        """设置结束点"""
        self._finish_points.extend(node_names)
        return self
    
    def with_checkpointer(self, checkpointer: Checkpointer) -> 'WorkflowBuilder':
        """设置检查点器"""
        self._checkpointer = checkpointer
        return self
    
    def with_config(self, config: GraphConfig) -> 'WorkflowBuilder':
        """设置图配置"""
        self._graph_config = config
        return self
    
    def build(self) -> CompiledGraph:
        """构建工作流"""
        config = self._graph_config or GraphConfig(
            name=self.name,
            max_iterations=20
        )
        
        builder = GraphBuilder(name=self.name, config=config)
        
        # 添加节点
        for name, agent_or_func in self._nodes.items():
            # 如果是 Agent，包装为节点函数
            if isinstance(agent_or_func, BaseAgent):
                node_func = self._wrap_agent_as_node(agent_or_func)
                builder.add_node(name, node_func)
            else:
                builder.add_node(name, agent_or_func)
        
        # 添加边
        for source, target in self._edges:
            builder.add_edge(source, target)
        
        # 添加条件边
        for source, (condition_func, branches) in self._conditional_edges.items():
                builder.add_conditional_edges(source, condition_func, branches)
        
        # 设置入口点
        if self._entry_point:
            builder.set_entry_point(self._entry_point)
        
        # 设置结束点
        for fp in self._finish_points:
                builder.set_finish_point(fp)
        
        return builder.compile(self._checkpointer)
    
    def _wrap_agent_as_node(self, agent: BaseAgent) -> Callable:
        """将 Agent 包装为节点函数"""
        def node_func(state: AgentState) -> AgentState:
            return agent.invoke(state)
        return node_func


# ==================== Agent 模板 ====================

@dataclass
class AgentTemplate:
    """Agent 模板
    
    定义 Agent 的预配置模板。
    """
    name: str
    description: str
    agent_type: str
    config: Dict[str, Any]
    tools: List[str] = field(default_factory=list)  # 工具名称列表
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "config": self.config,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTemplate':
        return cls(**data)


# 预定义模板
RESEARCH_AGENT_TEMPLATE = AgentTemplate(
    name="research_agent",
    description="Research Agent for information gathering and analysis",
    agent_type="react",
    config={
        "max_iterations": 15,
        "model": "gpt-4",
        "temperature": 0.3
    },
    tools=["web_search", "web_scrape", "summarize"],
    system_prompt="""You are a research assistant specialized in gathering and analyzing information.
Your goal is to find accurate, relevant information and present it clearly.
Always cite your sources and be thorough in your research."""
)

CODING_AGENT_TEMPLATE = AgentTemplate(
    name="coding_agent",
    description="Coding Agent for software development tasks",
    agent_type="chain_of_thought",
    config={
        "max_iterations": 20,
        "model": "gpt-4",
        "temperature": 0.2
    },
    tools=["code_execute", "file_read", "file_write", "shell_command"],
    system_prompt="""You are an expert software developer.
Think step by step when solving coding problems.
Write clean, well-documented, and tested code.
Follow best practices and design patterns."""
)

DATA_ANALYSIS_AGENT_TEMPLATE = AgentTemplate(
    name="data_analysis_agent",
    description="Data Analysis Agent for data processing and insights",
    agent_type="plan_execute",
    config={
        "max_iterations": 25,
        "model": "gpt-4",
        "temperature": 0.1,
        "enable_replanning": True
    },
    tools=["data_query", "data_transform", "data_visualize", "calculate"],
    system_prompt="""You are a data analyst specialized in extracting insights from data.
Create a plan before analyzing data.
Use appropriate statistical methods and visualizations.
Present findings clearly with supporting evidence."""
)

ML_TRAINING_AGENT_TEMPLATE = AgentTemplate(
    name="ml_training_agent",
    description="Machine Learning Agent for model training and evaluation",
    agent_type="hierarchical",
    config={
        "max_iterations": 30,
        "model": "gpt-4",
        "temperature": 0.2,
        "max_workers": 4
    },
    tools=["data_preprocess", "model_train", "model_evaluate", "hyperparameter_tune"],
    system_prompt="""You are a machine learning engineer.
Follow the ML lifecycle: data preparation, model selection, training, evaluation.
Track experiments and document results.
Optimize for both performance and efficiency."""
)

CUSTOMER_SERVICE_AGENT_TEMPLATE = AgentTemplate(
    name="customer_service_agent",
    description="Customer Service Agent for handling inquiries",
    agent_type="conversational",
    config={
        "max_iterations": 10,
        "model": "gpt-4",
        "temperature": 0.7,
        "conversation_memory_size": 100
    },
    tools=["knowledge_search", "ticket_create", "order_lookup"],
    system_prompt="""You are a friendly and helpful customer service representative.
Listen carefully to customer needs and provide accurate information.
Be empathetic and solution-oriented.
Escalate complex issues when necessary."""
)

QA_AGENT_TEMPLATE = AgentTemplate(
    name="qa_agent",
    description="Question Answering Agent for knowledge-based responses",
    agent_type="self_ask",
    config={
        "max_iterations": 10,
        "model": "gpt-4",
        "temperature": 0.3,
        "max_depth": 3
    },
    tools=["knowledge_search", "web_search", "calculate"],
    system_prompt="""You are a knowledgeable assistant that answers questions accurately.
Break down complex questions into simpler sub-questions.
Verify your answers when possible.
Admit when you don't know something."""
)

CREATIVE_WRITING_AGENT_TEMPLATE = AgentTemplate(
    name="creative_writing_agent",
    description="Creative Writing Agent for content generation",
    agent_type="reflexion",
    config={
        "max_iterations": 15,
        "model": "gpt-4",
        "temperature": 0.8,
        "max_reflections": 3
    },
    tools=["web_search", "text_generate", "grammar_check"],
    system_prompt="""You are a creative writer with expertise in various genres.
Create engaging, original content.
Reflect on your writing and improve it iteratively.
Pay attention to style, tone, and audience."""
)


def _register_default_templates() -> None:
    """注册默认模板"""
    registry = get_factory_registry()
    
    templates = [
        RESEARCH_AGENT_TEMPLATE,
        CODING_AGENT_TEMPLATE,
        DATA_ANALYSIS_AGENT_TEMPLATE,
        ML_TRAINING_AGENT_TEMPLATE,
        CUSTOMER_SERVICE_AGENT_TEMPLATE,
        QA_AGENT_TEMPLATE,
        CREATIVE_WRITING_AGENT_TEMPLATE,
    ]
    
    for template in templates:
        registry.register_template(template.name, template.to_dict())


# 在模块加载时注册默认模板
_register_default_templates()


# ==================== 组合工厂 ====================

class MasterFactory:
    """主工厂
    
    提供对所有子工厂的统一访问。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: FactoryConfig = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(config)
        return cls._instance
    
    def _init(self, config: FactoryConfig = None) -> None:
        """初始化主工厂"""
        self.config = config or FactoryConfig()
        
        # 创建所有子工厂
        self._agent_factory = AgentFactory(self.config)
        self._node_factory = NodeFactory(self.config)
        self._edge_factory = EdgeFactory(self.config)
        self._graph_factory = GraphFactory(self.config)
        self._tool_factory = ToolFactory(self.config)
        self._checkpointer_factory = CheckpointerFactory(self.config)
        self._strategy_factory = StrategyFactory(self.config)
        self._callback_factory = CallbackFactory(self.config)
        
        self.logger = logging.getLogger(f"{__name__}.MasterFactory")
    
    @property
    def agents(self) -> AgentFactory:
        """Agent 工厂"""
        return self._agent_factory
    
    @property
    def nodes(self) -> NodeFactory:
        """节点工厂"""
        return self._node_factory
    
    @property
    def edges(self) -> EdgeFactory:
        """边工厂"""
        return self._edge_factory
    
    @property
    def graphs(self) -> GraphFactory:
        """图工厂"""
        return self._graph_factory
    
    @property
    def tools(self) -> ToolFactory:
        """工具工厂"""
        return self._tool_factory
    
    @property
    def checkpointers(self) -> CheckpointerFactory:
        """检查点工厂"""
        return self._checkpointer_factory
    
    @property
    def strategies(self) -> StrategyFactory:
        """策略工厂"""
        return self._strategy_factory
    
    @property
    def callbacks(self) -> CallbackFactory:
        """回调工厂"""
        return self._callback_factory
    
    def create_agent(self, agent_type: str, **kwargs) -> BaseAgent:
        """创建 Agent 的便捷方法"""
        return self._agent_factory.create(agent_type, **kwargs)
    
    def agent_builder(self, name: str = "agent", 
                     agent_type: str = "react") -> AgentBuilder:
        """获取 Agent 构建器"""
        return AgentBuilder(name, agent_type)
    
    def pipeline_builder(self, name: str = "pipeline") -> PipelineBuilder:
        """获取流水线构建器"""
        return PipelineBuilder(name)
    
    def workflow_builder(self, name: str = "workflow") -> WorkflowBuilder:
        """获取工作流构建器"""
        return WorkflowBuilder(name)
    
    def get_metrics(self) -> Dict[str, Dict[str, int]]:
        """获取所有工厂的指标"""
        return {
            "agents": self._agent_factory.get_metrics(),
            "nodes": self._node_factory.get_metrics(),
            "edges": self._edge_factory.get_metrics(),
            "graphs": self._graph_factory.get_metrics(),
            "tools": self._tool_factory.get_metrics(),
            "checkpointers": self._checkpointer_factory.get_metrics(),
            "strategies": self._strategy_factory.get_metrics(),
            "callbacks": self._callback_factory.get_metrics()
        }
    
    def clear_all_caches(self) -> None:
        """清除所有工厂的缓存"""
        self._agent_factory.clear_cache()
        self._node_factory.clear_cache()
        self._edge_factory.clear_cache()
        self._graph_factory.clear_cache()
        self._tool_factory.clear_cache()
        self._checkpointer_factory.clear_cache()
        self._strategy_factory.clear_cache()
        self._callback_factory.clear_cache()
    
    # ==================== 增强方法 ====================
    
    def enhanced_agent_builder(self, agent_type: str = "react") -> 'EnhancedAgentBuilder':
        """获取增强的 Agent 构建器
        
        Args:
            agent_type: Agent 类型
        
        Returns:
            EnhancedAgentBuilder 实例
        """
        return EnhancedAgentBuilder(agent_type)
    
    def create_enhanced_agent(self, agent_type: str,
                             enable_builtin_tools: bool = True,
                             enable_metrics: bool = True,
                             enable_logging: bool = True,
                             checkpointer_type: str = "memory",
                             **kwargs) -> BaseAgent:
        """创建具有增强功能的 Agent
        
        Args:
            agent_type: Agent 类型
            enable_builtin_tools: 是否启用内置工具
            enable_metrics: 是否启用指标收集
            enable_logging: 是否启用日志
            checkpointer_type: 检查点器类型
            **kwargs: 其他参数
        
        Returns:
            增强的 Agent 实例
        """
        return self._agent_factory.create_with_enhanced_features(
            agent_type,
            enable_builtin_tools=enable_builtin_tools,
            enable_metrics=enable_metrics,
            enable_logging=enable_logging,
            checkpointer_type=checkpointer_type,
            **kwargs
        )
    
    def create_production_agent(self, agent_type: str,
                               llm_client: Any = None,
                               **kwargs) -> BaseAgent:
        """创建生产级 Agent
        
        自动配置所有生产级特性：
        - 内置工具
        - 指标收集
        - 日志
        - 检查点
        - 熔断器
        - 限流器
        
        Args:
            agent_type: Agent 类型
            llm_client: LLM 客户端
            **kwargs: 其他参数
        
        Returns:
            生产级 Agent 实例
        """
        # 使用增强构建器
        builder = self.enhanced_agent_builder(agent_type)
        
        # 配置 LLM
        if llm_client:
            builder.with_llm_client(llm_client)
        
        # 配置名称
        name = kwargs.pop("name", f"production_{agent_type}_agent")
        builder.with_name(name)
        
        # 配置模型
        model = kwargs.pop("model", self.config.default_model)
        builder.with_model(model)
        
        # 配置工具
        tools = kwargs.pop("tools", None)
        if tools:
            builder.with_tools(tools)
        
        # 启用所有生产级特性
        builder.with_builtin_tools()
        builder.with_metrics()
        builder.with_logging()
        builder.with_circuit_breaker()
        builder.with_rate_limiter()
        
        # 配置检查点器
        checkpointer_type = kwargs.pop("checkpointer_type", "memory")
        if checkpointer_type == "redis":
            builder.with_redis_checkpointer(**kwargs.pop("redis_config", {}))
        elif checkpointer_type == "sqlite":
            builder.with_sqlite_checkpointer(**kwargs.pop("sqlite_config", {}))
        elif checkpointer_type == "file":
            builder.with_file_checkpointer(**kwargs.pop("file_config", {}))
        else:
            builder.with_memory_checkpointer()
        
        # 应用其他配置
        for key, value in kwargs.items():
            if hasattr(builder, f"with_{key}"):
                getattr(builder, f"with_{key}")(value)
        
        return builder.build()
    
    def configure_agent(self, agent: BaseAgent,
                       builtin_tools: List[str] = None,
                       checkpointer_type: str = None,
                       add_metrics: bool = False,
                       add_logging: bool = False) -> BaseAgent:
        """配置现有 Agent
        
        Args:
            agent: Agent 实例
            builtin_tools: 内置工具列表
            checkpointer_type: 检查点器类型
            add_metrics: 是否添加指标回调
            add_logging: 是否添加日志回调
        
        Returns:
            配置后的 Agent
        """
        # 配置内置工具
        if builtin_tools:
            self._agent_factory.configure_agent_with_builtin_tools(
                agent, tool_names=builtin_tools
            )
        
        # 配置检查点器
        if checkpointer_type:
            self._agent_factory.configure_agent_checkpointer(
                agent, checkpointer_type
            )
        
        # 添加回调
        if add_metrics:
            agent.add_callback(MetricsCallback())
        if add_logging:
            agent.add_callback(LoggingCallback())
        
        return agent
    
    def get_agent_diagnostics(self, agent: BaseAgent) -> Dict[str, Any]:
        """获取 Agent 诊断信息
        
        Args:
            agent: Agent 实例
        
        Returns:
            诊断信息字典
        """
        diagnostics = {
            "name": agent.name,
            "type": type(agent).__name__,
            "is_running": agent.is_running,
            "tools_count": len(agent.tools),
            "tools": [t.name for t in agent.tools],
        }
        
        # 获取指标
        try:
            metrics = self._agent_factory.get_agent_metrics(agent)
            diagnostics["metrics"] = metrics
        except Exception as e:
            diagnostics["metrics_error"] = str(e)
        
        # 获取工具指标
        try:
            tool_metrics = self._agent_factory.get_agent_tool_metrics(agent)
            diagnostics["tool_metrics"] = tool_metrics
        except Exception as e:
            diagnostics["tool_metrics_error"] = str(e)
        
        # 获取图状态
        try:
            graph_status = agent.get_graph_status()
            diagnostics["graph_status"] = graph_status.value
        except Exception:
            pass
        
        return diagnostics
    
    def health_check(self) -> Dict[str, Any]:
        """执行健康检查
        
        Returns:
            健康状态字典
        """
        health = {
            "status": "healthy",
            "factories": {},
            "errors": []
        }
        
        # 检查各工厂
        factories = [
            ("agents", self._agent_factory),
            ("nodes", self._node_factory),
            ("edges", self._edge_factory),
            ("graphs", self._graph_factory),
            ("tools", self._tool_factory),
            ("checkpointers", self._checkpointer_factory),
            ("strategies", self._strategy_factory),
            ("callbacks", self._callback_factory),
        ]
        
        for name, factory in factories:
            try:
                metrics = factory.get_metrics()
                health["factories"][name] = {
                    "status": "healthy",
                    "created": metrics.get("created", 0),
                    "errors": metrics.get("errors", 0)
                }
                if metrics.get("errors", 0) > 0:
                    health["factories"][name]["status"] = "degraded"
            except Exception as e:
                health["factories"][name] = {"status": "error", "error": str(e)}
                health["errors"].append(f"{name}: {e}")
        
        # 设置总体状态
        if health["errors"]:
            health["status"] = "unhealthy"
        elif any(f.get("status") == "degraded" for f in health["factories"].values()):
            health["status"] = "degraded"
        
        return health
    
    # =========================================================================
    # 派生功能：调用 state 模块 (AgentMessage, MessageType, ToolCall, ToolResult, 
    #           AgentStatus, StateManager, StateValidator, StateSerializer, 
    #           StateEventEmitter, MessageBuffer, AgentMemory, MemoryEntry, 
    #           PlanStep, MemoryType)
    # =========================================================================
    
    def create_agent_message(
        self,
        content: str,
        message_type: MessageType = MessageType.HUMAN,
        name: str = None,
        tool_calls: List[ToolCall] = None,
        tool_call_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> AgentMessage:
        """创建 Agent 消息
        
        调用 state 模块的 AgentMessage 和 MessageType
        
        Args:
            content: 消息内容
            message_type: 消息类型
            name: 发送者名称
            tool_calls: 工具调用列表
            tool_call_id: 工具调用 ID
            metadata: 元数据
            
        Returns:
            AgentMessage 实例
        """
        return AgentMessage(
            content=content,
            type=message_type,
            name=name,
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id,
            metadata=metadata or {}
        )
    
    def create_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = None
    ) -> ToolCall:
        """创建工具调用
        
        调用 state 模块的 ToolCall
        
        Args:
            name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            ToolCall 实例
        """
        return ToolCall(
            id=tool_call_id or str(uuid.uuid4()),
            name=name,
            arguments=arguments
        )
    
    def create_tool_result(
        self,
        tool_call_id: str,
        name: str,
        result: Any,
        success: bool = True,
        error: str = None
    ) -> ToolResult:
        """创建工具结果
        
        调用 state 模块的 ToolResult
        
        Args:
            tool_call_id: 工具调用 ID
            name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            ToolResult 实例
        """
        return ToolResult(
            tool_call_id=tool_call_id,
            name=name,
            result=result,
            success=success,
            error=error
        )
    
    def get_agent_status(self, status_name: str) -> AgentStatus:
        """获取 Agent 状态枚举
        
        调用 state 模块的 AgentStatus
        
        Args:
            status_name: 状态名称 (idle, initializing, running, waiting_tool, 
                        waiting_human, waiting_approval, completed, failed, 
                        paused, cancelled, timeout)
            
        Returns:
            AgentStatus 枚举值
        """
        status_map = {
            "idle": AgentStatus.IDLE,
            "initializing": AgentStatus.INITIALIZING,
            "running": AgentStatus.RUNNING,
            "waiting_tool": AgentStatus.WAITING_TOOL,
            "waiting_human": AgentStatus.WAITING_HUMAN,
            "waiting_approval": AgentStatus.WAITING_APPROVAL,
            "completed": AgentStatus.COMPLETED,
            "failed": AgentStatus.FAILED,
            "paused": AgentStatus.PAUSED,
            "cancelled": AgentStatus.CANCELLED,
            "timeout": AgentStatus.TIMEOUT
        }
        return status_map.get(status_name.lower(), AgentStatus.IDLE)
    
    def create_state_manager(self) -> StateManager:
        """创建状态管理器
        
        调用 state 模块的 StateManager
        
        Returns:
            StateManager 实例
        """
        return StateManager()
    
    def create_state_validator(self) -> StateValidator:
        """创建状态验证器
        
        调用 state 模块的 StateValidator
        
        Returns:
            StateValidator 实例
        """
        return StateValidator()
    
    def create_state_serializer(self) -> StateSerializer:
        """创建状态序列化器
        
        调用 state 模块的 StateSerializer
        
        Returns:
            StateSerializer 实例
        """
        return StateSerializer()
    
    def create_state_event_emitter(self) -> StateEventEmitter:
        """创建状态事件发射器
        
        调用 state 模块的 StateEventEmitter
        
        Returns:
            StateEventEmitter 实例
        """
        return StateEventEmitter()
    
    def create_message_buffer(self, max_size: int = 100) -> MessageBuffer:
        """创建消息缓冲区
        
        调用 state 模块的 MessageBuffer
        
        Args:
            max_size: 最大缓冲区大小
            
        Returns:
            MessageBuffer 实例
        """
        return MessageBuffer(max_size=max_size)
    
    def create_agent_memory(self, memory_type: MemoryType = None) -> AgentMemory:
        """创建 Agent 记忆
        
        调用 state 模块的 AgentMemory 和 MemoryType
        
        Args:
            memory_type: 记忆类型
            
        Returns:
            AgentMemory 实例
        """
        memory = AgentMemory()
        if memory_type:
            memory.default_type = memory_type
        return memory
    
    def create_memory_entry(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> MemoryEntry:
        """创建记忆条目
        
        调用 state 模块的 MemoryEntry 和 MemoryType
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            metadata: 元数据
            tags: 标签列表
            
        Returns:
            MemoryEntry 实例
        """
        return MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            tags=set(tags) if tags else set()
        )
    
    def create_plan_step(
        self,
        step_id: str = None,
        description: str = "",
        action: str = "",
        dependencies: List[str] = None,
        status: str = "pending"
    ) -> PlanStep:
        """创建计划步骤
        
        调用 state 模块的 PlanStep
        
        Args:
            step_id: 步骤 ID（可选，自动生成）
            description: 步骤描述
            action: 步骤动作
            dependencies: 依赖步骤列表
            status: 步骤状态
            
        Returns:
            PlanStep 实例
        """
        kwargs = {
            "description": description,
            "action": action,
            "dependencies": dependencies or [],
            "status": status
        }
        if step_id:
            kwargs["step_id"] = step_id
        return PlanStep(**kwargs)
    
    def get_memory_type(self, type_name: str) -> MemoryType:
        """获取记忆类型枚举
        
        调用 state 模块的 MemoryType
        
        Args:
            type_name: 类型名称 (short_term, long_term, episodic, semantic, working)
            
        Returns:
            MemoryType 枚举值
        """
        type_map = {
            "short_term": MemoryType.SHORT_TERM,
            "long_term": MemoryType.LONG_TERM,
            "episodic": MemoryType.EPISODIC,
            "semantic": MemoryType.SEMANTIC,
            "working": MemoryType.WORKING
        }
        return type_map.get(type_name.lower(), MemoryType.SHORT_TERM)
    
    # =========================================================================
    # 派生功能：调用 nodes 模块 (BaseNodeFactory, NodeConfig, NodeType, 
    #           NodeMetrics, NodeRegistry)
    # =========================================================================
    
    def get_base_node_factory(self) -> BaseNodeFactory:
        """获取基础节点工厂
        
        调用 nodes 模块的 BaseNodeFactory
        
        Returns:
            BaseNodeFactory 实例
        """
        return BaseNodeFactory()
    
    def create_node_config(
        self,
        name: str,
        node_type: NodeType = NodeType.TRANSFORM,
        description: str = "",
        timeout: float = 60.0,
        retry_count: int = 3
    ) -> NodeConfig:
        """创建节点配置
        
        调用 nodes 模块的 NodeConfig 和 NodeType
        
        Args:
            name: 节点名称
            node_type: 节点类型
            description: 节点描述
            timeout: 超时时间
            retry_count: 重试次数
            
        Returns:
            NodeConfig 实例
        """
        return NodeConfig(
            name=name,
            node_type=node_type,
            description=description,
            timeout=timeout,
            retry_count=retry_count
        )
    
    def get_node_type(self, type_name: str) -> NodeType:
        """获取节点类型枚举
        
        调用 nodes 模块的 NodeType
        
        Args:
            type_name: 类型名称
            
        Returns:
            NodeType 枚举值
        """
        type_map = {
            "llm": NodeType.LLM,
            "tool": NodeType.TOOL,
            "human": NodeType.HUMAN,
            "transform": NodeType.TRANSFORM,
            "branch": NodeType.BRANCH,
            "parallel": NodeType.PARALLEL,
            "start": NodeType.START,
            "end": NodeType.END
        }
        return type_map.get(type_name.lower(), NodeType.TRANSFORM)
    
    def create_node_metrics(self, node_name: str = "default_node") -> NodeMetrics:
        """创建节点指标
        
        调用 nodes 模块的 NodeMetrics
        
        Args:
            node_name: 节点名称
        
        Returns:
            NodeMetrics 实例
        """
        return NodeMetrics(node_name=node_name)
    
    def get_node_registry_instance(self) -> NodeRegistry:
        """获取节点注册表实例
        
        调用 nodes 模块的 NodeRegistry 和 get_node_registry
        
        Returns:
            NodeRegistry 实例
        """
        return get_node_registry()
    
    
    def get_edge_type(self, type_name: str) -> EdgeType:
        """获取边类型枚举
        
        调用 edges 模块的 EdgeType
        
        Args:
            type_name: 类型名称 (normal, conditional, loop, parallel, 
                      timeout, retry, fallback, weighted, delayed, event, transform)
            
        Returns:
            EdgeType 枚举值
        """
        type_map = {
            "normal": EdgeType.NORMAL,
            "conditional": EdgeType.CONDITIONAL,
            "loop": EdgeType.LOOP,
            "parallel": EdgeType.PARALLEL,
            "timeout": EdgeType.TIMEOUT,
            "retry": EdgeType.RETRY,
            "fallback": EdgeType.FALLBACK,
            "weighted": EdgeType.WEIGHTED,
            "delayed": EdgeType.DELAYED,
            "event": EdgeType.EVENT,
            "transform": EdgeType.TRANSFORM
        }
        return type_map.get(type_name.lower(), EdgeType.NORMAL)
    
    def create_edge_condition(
        self,
        condition_func: Callable[[AgentState], bool],
        name: str = "custom_condition",
        priority: int = 0,
        description: str = ""
    ) -> EdgeCondition:
        """创建边条件
        
        调用 edges 模块的 EdgeCondition
        
        Args:
            condition_func: 条件函数
            name: 条件名称
            priority: 优先级
            description: 描述
            
        Returns:
            EdgeCondition 实例
        """
        return EdgeCondition(
            name=name,
            condition_func=condition_func,
            priority=priority,
            description=description
        )
    
    def get_route_after_tools_condition(self) -> Callable:
        """获取工具执行后的路由条件
        
        调用 edges 模块的 route_after_tools
        
        Returns:
            路由函数
        """
        return route_after_tools
    
    def get_should_continue_condition(self) -> Callable:
        """获取继续执行条件
        
        调用 edges 模块的 should_continue_condition
        
        Returns:
            条件函数
        """
        return should_continue_condition
    
    def create_priority_router(
        self,
        name: str = "priority_router"
    ) -> PriorityRouter:
        """创建优先级路由器
        
        调用 edges 模块的 PriorityRouter
        根据边的优先级选择路由
        
        Args:
            name: 路由器名称
            
        Returns:
            PriorityRouter 实例
        """
        return PriorityRouter(name=name)
    
    def create_weighted_router(
        self,
        name: str = "weighted_router"
    ) -> WeightedRouter:
        """创建权重路由器
        
        调用 edges 模块的 WeightedRouter
        根据边的权重随机选择路由
        
        Args:
            name: 路由器名称
            
        Returns:
            WeightedRouter 实例
        """
        return WeightedRouter(name=name)
    
    def create_load_balance_router(
        self,
        name: str = "load_balance_router"
    ) -> LoadBalanceRouter:
        """创建负载均衡路由器
        
        调用 edges 模块的 LoadBalanceRouter
        选择负载最小的边
        
        Args:
            name: 路由器名称
            
        Returns:
            LoadBalanceRouter 实例
        """
        return LoadBalanceRouter(name=name)
    
    def create_ab_test_router(
        self,
        name: str = "ab_test_router",
        test_name: str = "",
        user_field: str = "thread_id"
    ) -> ABTestRouter:
        """创建 A/B 测试路由器
        
        调用 edges 模块的 ABTestRouter
        根据用户标识进行稳定分流
        
        Args:
            name: 路由器名称
            test_name: 测试名称
            user_field: 用户标识字段
            
        Returns:
            ABTestRouter 实例
        """
        return ABTestRouter(
            name=name,
            test_name=test_name,
            user_field=user_field
        )
    
    def create_conditional_edge_from_func(
        self,
        source: str,
        condition_func: Callable[[AgentState], str],
        branches: Dict[str, str]
    ) -> ConditionalEdge:
        """从条件函数创建条件边
        
        调用 edges 模块的 create_conditional_edge
        
        Args:
            source: 源节点
            condition_func: 条件函数
            branches: 分支映射
            
        Returns:
            ConditionalEdge 实例
        """
        return create_conditional_edge(
            source=source,
            condition_func=condition_func,
            branches=branches
        )
    
    def create_retry_edge_from_config(
        self,
        source: str,
        target: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> RetryEdge:
        """从配置创建重试边
        
        调用 edges 模块的 create_retry_edge
        
        Args:
            source: 源节点
            target: 目标节点
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            
        Returns:
            RetryEdge 实例
        """
        return create_retry_edge(
            source=source,
            target=target,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def create_fallback_edge_from_config(
        self,
        source: str,
        target: str,
        fallback_targets: List[str]
    ) -> FallbackEdge:
        """从配置创建降级边
        
        调用 edges 模块的 create_fallback_edge
        
        Args:
            source: 源节点
            target: 目标节点
            fallback_targets: 降级目标列表
            
        Returns:
            FallbackEdge 实例
        """
        return create_fallback_edge(
            source=source,
            target=target,
            fallback_targets=fallback_targets
        )
    
    # =========================================================================
    # 派生功能：调用 graph 模块 (GraphStatus, ExecutionMode, GraphMetrics,
    #           ExecutionEvent)
    # =========================================================================
    
    def get_graph_status(self, status_name: str) -> GraphStatus:
        """获取图状态枚举
        
        调用 graph 模块的 GraphStatus
        
        Args:
            status_name: 状态名称 (building, compiled, running, paused, 
                        completed, failed, cancelled)
            
        Returns:
            GraphStatus 枚举值
        """
        status_map = {
            "building": GraphStatus.BUILDING,
            "compiled": GraphStatus.COMPILED,
            "running": GraphStatus.RUNNING,
            "paused": GraphStatus.PAUSED,
            "completed": GraphStatus.COMPLETED,
            "failed": GraphStatus.FAILED,
            "cancelled": GraphStatus.CANCELLED
        }
        return status_map.get(status_name.lower(), GraphStatus.BUILDING)
    
    def get_execution_mode(self, mode_name: str) -> ExecutionMode:
        """获取执行模式枚举
        
        调用 graph 模块的 ExecutionMode
        
        Args:
            mode_name: 模式名称 (sequential, parallel, async)
            
        Returns:
            ExecutionMode 枚举值
        """
        mode_map = {
            "sequential": ExecutionMode.SEQUENTIAL,
            "parallel": ExecutionMode.PARALLEL,
            "async": ExecutionMode.ASYNC
        }
        return mode_map.get(mode_name.lower(), ExecutionMode.SEQUENTIAL)
    
    def create_graph_metrics(self) -> GraphMetrics:
        """创建图指标
        
        调用 graph 模块的 GraphMetrics
        
        Returns:
            GraphMetrics 实例
        """
        return GraphMetrics()
    
    def create_execution_event(
        self,
        event_type: str,
        node_name: str,
        data: Dict[str, Any] = None
    ) -> ExecutionEvent:
        """创建执行事件
        
        调用 graph 模块的 ExecutionEvent
        
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
            data=data or {}
        )
    
    # =========================================================================
    # 派生功能：调用 tools 模块 (ToolParameter, ToolStatus, ToolMetrics,
    #           ToolCache, ToolRateLimiter, RetryHandler, ToolHooks, async_tool)
    # =========================================================================
    
    def create_tool_parameter(
        self,
        name: str,
        param_type: str,
        description: str = "",
        required: bool = True,
        default: Any = None
    ) -> ToolParameter:
        """创建工具参数
        
        调用 tools 模块的 ToolParameter
        
        Args:
            name: 参数名称
            param_type: 参数类型
            description: 参数描述
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
    
    def get_tool_status(self, status_name: str) -> ToolStatus:
        """获取工具状态枚举
        
        调用 tools 模块的 ToolStatus
        
        Args:
            status_name: 状态名称 (active, disabled, maintenance, 
                        deprecated, experimental)
            
        Returns:
            ToolStatus 枚举值
        """
        status_map = {
            "active": ToolStatus.ACTIVE,
            "disabled": ToolStatus.DISABLED,
            "maintenance": ToolStatus.MAINTENANCE,
            "deprecated": ToolStatus.DEPRECATED,
            "experimental": ToolStatus.EXPERIMENTAL
        }
        return status_map.get(status_name.lower(), ToolStatus.ACTIVE)
    
    def create_tool_metrics(self, tool_name: str = "default_tool") -> ToolMetrics:
        """创建工具指标
        
        调用 tools 模块的 ToolMetrics
        
        Args:
            tool_name: 工具名称
        
        Returns:
            ToolMetrics 实例
        """
        return ToolMetrics(tool_name=tool_name)
    
    def create_tool_cache(
        self,
        max_size: int = 1000,
        default_ttl: int = 300
    ) -> ToolCache:
        """创建工具缓存
        
        调用 tools 模块的 ToolCache
        
        Args:
            max_size: 最大缓存大小
            default_ttl: 默认过期时间（秒）
            
        Returns:
            ToolCache 实例
        """
        return ToolCache(max_size=max_size, default_ttl=default_ttl)
    
    def create_tool_rate_limiter(
        self,
        rate: float = 10.0,
        burst: int = 20,
        per_tool: bool = True
    ) -> ToolRateLimiter:
        """创建工具限流器
        
        调用 tools 模块的 ToolRateLimiter（令牌桶算法）
        
        Args:
            rate: 每秒请求数
            burst: 突发容量
            per_tool: 是否按工具限流
            
        Returns:
            ToolRateLimiter 实例
        """
        return ToolRateLimiter(
            rate=rate,
            burst=burst,
            per_tool=per_tool
        )
    
    def create_retry_handler(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> RetryHandler:
        """创建重试处理器
        
        调用 tools 模块的 RetryHandler
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟
            max_delay: 最大延迟
            
        Returns:
            RetryHandler 实例
        """
        return RetryHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay
        )
    
    def create_tool_hooks(
        self,
        before_execute: List[Callable] = None,
        after_execute: List[Callable] = None,
        on_error: List[Callable] = None
    ) -> ToolHooks:
        """创建工具钩子
        
        调用 tools 模块的 ToolHooks
        
        Args:
            before_execute: 执行前钩子列表
            after_execute: 执行后钩子列表
            on_error: 错误钩子列表
            
        Returns:
            ToolHooks 实例
        """
        hooks = ToolHooks()
        if before_execute:
            for callback in before_execute:
                hooks.add_before(callback)
        if after_execute:
            for callback in after_execute:
                hooks.add_after(callback)
        if on_error:
            for callback in on_error:
                hooks.add_on_error(callback)
        return hooks
    
    def create_async_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        category: ToolCategory = ToolCategory.CUSTOM,
        timeout: float = 30.0
    ) -> Tool:
        """创建异步工具
        
        调用 tools 模块的 async_tool 装饰器
        
        Args:
            name: 工具名称
            func: 异步函数
            description: 工具描述
            category: 工具类别
            timeout: 超时时间
            
        Returns:
            Tool 实例
        """
        decorated = async_tool(
            name=name,
            description=description,
            category=category,
            timeout=timeout
        )
        return decorated(func)
    
    # =========================================================================
    # 派生功能：调用 checkpointer 模块 (EnhancedCheckpoint, CheckpointTag,
    #           CheckpointDiff, CheckpointMetrics, get_checkpointer)
    # =========================================================================
    
    def create_enhanced_checkpoint(
        self,
        checkpoint_id: str,
        thread_id: str,
        state: Dict[str, Any],
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        branch: str = "main"
    ) -> EnhancedCheckpoint:
        """创建增强检查点
        
        调用 checkpointer 模块的 EnhancedCheckpoint
        
        Args:
            checkpoint_id: 检查点 ID
            thread_id: 线程 ID
            state: 状态数据
            tags: 标签列表
            metadata: 元数据
            branch: 分支名称
            
        Returns:
            EnhancedCheckpoint 实例
        """
        return EnhancedCheckpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            state=state,
            tags=tags or [],
            metadata=metadata or {},
            branch=branch
        )
    
    def create_checkpoint_tag(
        self,
        name: str,
        checkpoint_id: str,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> CheckpointTag:
        """创建检查点标签
        
        调用 checkpointer 模块的 CheckpointTag
        
        Args:
            name: 标签名称
            checkpoint_id: 检查点 ID
            description: 标签描述
            metadata: 元数据
            
        Returns:
            CheckpointTag 实例
        """
        return CheckpointTag(
            name=name,
            checkpoint_id=checkpoint_id,
            description=description,
            metadata=metadata or {}
        )
    
    def create_checkpoint_diff(
        self,
        from_checkpoint_id: str,
        to_checkpoint_id: str,
        added_keys: List[str] = None,
        removed_keys: List[str] = None,
        modified_keys: List[str] = None
    ) -> CheckpointDiff:
        """创建检查点差异
        
        调用 checkpointer 模块的 CheckpointDiff
        
        Args:
            from_checkpoint_id: 源检查点 ID
            to_checkpoint_id: 目标检查点 ID
            added_keys: 新增的键列表
            removed_keys: 删除的键列表
            modified_keys: 修改的键列表
            
        Returns:
            CheckpointDiff 实例
        """
        return CheckpointDiff(
            from_checkpoint_id=from_checkpoint_id,
            to_checkpoint_id=to_checkpoint_id,
            added_keys=added_keys or [],
            removed_keys=removed_keys or [],
            modified_keys=modified_keys or []
        )
    
    def create_checkpoint_metrics(self) -> CheckpointMetrics:
        """创建检查点指标
        
        调用 checkpointer 模块的 CheckpointMetrics
        
        Returns:
            CheckpointMetrics 实例
        """
        return CheckpointMetrics()
    
    def get_checkpointer_instance(
        self,
        checkpointer_type: str = "memory",
        **kwargs
    ) -> Checkpointer:
        """获取检查点器实例
        
        调用 checkpointer 模块的 get_checkpointer
        
        Args:
            checkpointer_type: 检查点器类型 (memory, redis, sqlite, file)
            **kwargs: 检查点器参数
            
        Returns:
            Checkpointer 实例
        """
        return get_checkpointer(checkpointer_type, **kwargs)


def get_master_factory(config: FactoryConfig = None) -> MasterFactory:
    """获取主工厂单例"""
    return MasterFactory(config)


# ==================== 便捷函数 ====================

# 获取默认工厂
_default_factory: Optional[AgentFactory] = None


def _get_default_factory() -> AgentFactory:
    """获取默认 Agent 工厂"""
    global _default_factory
    if _default_factory is None:
        _default_factory = AgentFactory()
    return _default_factory


def create_react_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    system_prompt: str = None,
    **kwargs
) -> ReActAgent:
    """创建 ReAct Agent 的便捷函数"""
    return _get_default_factory().create_react_agent(
        tools=tools,
        llm_client=llm_client,
        system_prompt=system_prompt,
        **kwargs
    )


def create_plan_execute_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    **kwargs
) -> PlanAndExecuteAgent:
    """创建计划执行 Agent 的便捷函数"""
    return _get_default_factory().create_plan_execute_agent(
        tools=tools,
        llm_client=llm_client,
        **kwargs
    )


def create_reflexion_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    max_reflections: int = 3,
    **kwargs
) -> ReflexionAgent:
    """创建反思 Agent 的便捷函数"""
    return _get_default_factory().create_reflexion_agent(
        tools=tools,
        llm_client=llm_client,
        max_reflections=max_reflections,
        **kwargs
    )


def create_multi_agent_system(
    roles: List[AgentRole] = None,
    llm_client: Any = None,
    **kwargs
) -> MultiAgentSystem:
    """创建多 Agent 系统的便捷函数"""
    return _get_default_factory().create_multi_agent_system(
        roles=roles,
        llm_client=llm_client,
        **kwargs
    )


def create_tool_calling_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    **kwargs
) -> ToolCallingAgent:
    """创建工具调用 Agent 的便捷函数"""
    return _get_default_factory().create_tool_calling_agent(
        tools=tools,
        llm_client=llm_client,
        **kwargs
    )


def create_conversational_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    conversation_memory_size: int = 50,
    **kwargs
) -> ConversationalAgent:
    """创建对话 Agent 的便捷函数"""
    return _get_default_factory().create_conversational_agent(
        tools=tools,
        llm_client=llm_client,
        conversation_memory_size=conversation_memory_size,
        **kwargs
    )


def create_chain_of_thought_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    **kwargs
) -> ChainOfThoughtAgent:
    """创建思维链 Agent 的便捷函数"""
    return _get_default_factory().create_chain_of_thought_agent(
        tools=tools,
        llm_client=llm_client,
        **kwargs
    )


def create_self_ask_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    max_depth: int = 3,
    **kwargs
) -> SelfAskAgent:
    """创建自问 Agent 的便捷函数"""
    return _get_default_factory().create_self_ask_agent(
        tools=tools,
        llm_client=llm_client,
        max_depth=max_depth,
        **kwargs
    )


def create_hierarchical_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    max_workers: int = 3,
    **kwargs
) -> HierarchicalAgent:
    """创建层级 Agent 的便捷函数"""
    return _get_default_factory().create_hierarchical_agent(
        tools=tools,
        llm_client=llm_client,
        max_workers=max_workers,
        **kwargs
    )


def create_workflow_agent(
    tools: List[Tool] = None,
    llm_client: Any = None,
    workflow_definition: Dict[str, Any] = None,
    **kwargs
) -> WorkflowAgent:
    """创建工作流 Agent 的便捷函数"""
    return _get_default_factory().create_workflow_agent(
        tools=tools,
        llm_client=llm_client,
        workflow_definition=workflow_definition,
        **kwargs
    )


def create_agent_from_template(
    template_name: str,
    llm_client: Any = None,
    **kwargs
) -> BaseAgent:
    """从模板创建 Agent 的便捷函数"""
    return _get_default_factory().create_from_template(
        template_name,
        llm_client=llm_client,
        **kwargs
    )


# ==================== 预置 Agent 模板函数 ====================

def create_research_agent(llm_client: Any = None, **kwargs) -> ReActAgent:
    """创建研究型 Agent
    
    专注于信息搜索和分析。
    """
    return create_agent_from_template("research_agent", llm_client=llm_client, **kwargs)


def create_coding_agent(llm_client: Any = None, **kwargs) -> ChainOfThoughtAgent:
    """创建编程型 Agent
    
    专注于代码生成和调试。
    """
    return create_agent_from_template("coding_agent", llm_client=llm_client, **kwargs)


def create_data_analyst_agent(llm_client: Any = None, **kwargs) -> PlanAndExecuteAgent:
    """创建数据分析 Agent
    
    专注于数据分析和可视化。
    """
    return create_agent_from_template("data_analysis_agent", llm_client=llm_client, **kwargs)


def create_ml_training_agent(llm_client: Any = None, **kwargs) -> HierarchicalAgent:
    """创建机器学习训练 Agent
    
    专注于 ML 模型训练流程。
    """
    return create_agent_from_template("ml_training_agent", llm_client=llm_client, **kwargs)


def create_customer_service_agent(llm_client: Any = None, **kwargs) -> ConversationalAgent:
    """创建客服 Agent
    
    专注于客户服务和支持。
    """
    return create_agent_from_template("customer_service_agent", llm_client=llm_client, **kwargs)


def create_qa_agent(llm_client: Any = None, **kwargs) -> SelfAskAgent:
    """创建问答 Agent
    
    专注于知识问答。
    """
    return create_agent_from_template("qa_agent", llm_client=llm_client, **kwargs)


def create_creative_writing_agent(llm_client: Any = None, **kwargs) -> ReflexionAgent:
    """创建创意写作 Agent
    
    专注于创意内容生成。
    """
    return create_agent_from_template("creative_writing_agent", llm_client=llm_client, **kwargs)


# ==================== 快捷函数 ====================

def quick_agent(agent_type: str = "react", **kwargs) -> BaseAgent:
    """快速创建 Agent
    
    Args:
        agent_type: Agent 类型
        **kwargs: Agent 参数
        
    Returns:
        Agent 实例
    """
    return _get_default_factory().create(agent_type, **kwargs)


def quick_pipeline(*agents: Tuple[str, BaseAgent]) -> Pipeline:
    """快速创建流水线
    
    Args:
        *agents: (名称, Agent) 元组
        
    Returns:
        Pipeline 实例
    """
    builder = PipelineBuilder("quick_pipeline")
    for name, agent in agents:
        builder.add_stage(name, agent)
    return builder.build()


def quick_workflow(nodes: Dict[str, BaseAgent],
                  edges: List[Tuple[str, str]],
                  entry: str) -> CompiledGraph:
    """快速创建工作流
    
    Args:
        nodes: 节点字典
        edges: 边列表
        entry: 入口点
        
    Returns:
        编译后的图
    """
    builder = WorkflowBuilder("quick_workflow")
    for name, agent in nodes.items():
        builder.add_node(name, agent)
    for source, target in edges:
        builder.add_edge(source, target)
    builder.set_entry(entry)
    return builder.build()


def build_agent(name: str = "agent", agent_type: str = "react") -> AgentBuilder:
    """获取 Agent 构建器
    
    Args:
        name: Agent 名称
        agent_type: Agent 类型
        
    Returns:
        AgentBuilder 实例
        
    使用示例:
        agent = (build_agent("my_agent")
            .with_tools([tool1, tool2])
            .with_llm(llm_client)
            .build())
    """
    return AgentBuilder(name, agent_type)


def build_pipeline(name: str = "pipeline") -> PipelineBuilder:
    """获取流水线构建器
    
    Args:
        name: 流水线名称
        
    Returns:
        PipelineBuilder 实例
    """
    return PipelineBuilder(name)


def build_workflow(name: str = "workflow") -> WorkflowBuilder:
    """获取工作流构建器
    
    Args:
        name: 工作流名称
        
    Returns:
        WorkflowBuilder 实例
    """
    return WorkflowBuilder(name)


# ==================== 增强便捷函数 ====================

def build_enhanced_agent(agent_type: str = "react") -> 'EnhancedAgentBuilder':
    """获取增强的 Agent 构建器
    
    Args:
        agent_type: Agent 类型
        
    Returns:
        EnhancedAgentBuilder 实例
        
    使用示例:
        agent = (build_enhanced_agent("react")
            .with_name("my_agent")
            .with_builtin_tools()
            .with_metrics()
            .with_logging()
            .with_circuit_breaker()
            .build())
    """
    return EnhancedAgentBuilder(agent_type)


def create_production_agent(
    agent_type: str = "react",
    llm_client: Any = None,
    **kwargs
) -> BaseAgent:
    """创建生产级 Agent
    
    自动配置所有生产级特性。
    
    Args:
        agent_type: Agent 类型
        llm_client: LLM 客户端
        **kwargs: 其他参数
        
    Returns:
        生产级 Agent 实例
    """
    return get_master_factory().create_production_agent(
        agent_type,
        llm_client=llm_client,
        **kwargs
    )


def create_enhanced_agent(
    agent_type: str = "react",
    enable_builtin_tools: bool = True,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    checkpointer_type: str = "memory",
    **kwargs
) -> BaseAgent:
    """创建增强的 Agent
    
    Args:
        agent_type: Agent 类型
        enable_builtin_tools: 是否启用内置工具
        enable_metrics: 是否启用指标
        enable_logging: 是否启用日志
        checkpointer_type: 检查点器类型
        **kwargs: 其他参数
        
    Returns:
        增强的 Agent 实例
    """
    return _get_default_factory().create_with_enhanced_features(
        agent_type,
        enable_builtin_tools=enable_builtin_tools,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging,
        checkpointer_type=checkpointer_type,
        **kwargs
    )


def configure_agent_tools(
    agent: BaseAgent,
    tool_names: List[str] = None,
    categories: List[str] = None
) -> BaseAgent:
    """为 Agent 配置工具
    
    Args:
        agent: Agent 实例
        tool_names: 工具名称列表
        categories: 工具类别列表
        
    Returns:
        配置后的 Agent
    """
    return _get_default_factory().configure_agent_with_builtin_tools(
        agent,
        tool_names=tool_names,
        categories=categories
    )


def get_agent_diagnostics(agent: BaseAgent) -> Dict[str, Any]:
    """获取 Agent 诊断信息
    
    Args:
        agent: Agent 实例
        
    Returns:
        诊断信息字典
    """
    return get_master_factory().get_agent_diagnostics(agent)


def factory_health_check() -> Dict[str, Any]:
    """执行工厂健康检查
    
    Returns:
        健康状态字典
    """
    return get_master_factory().health_check()


# ==================== 导出列表 ====================

__all__ = [
    # 枚举
    "FactoryType",
    "ComponentType",
    
    # 配置
    "FactoryConfig",
    
    # 注册表
    "FactoryRegistry",
    "get_factory_registry",
    
    # 基础工厂
    "BaseFactory",
    
    # 核心工厂
    "AgentFactory",
    "NodeFactory",
    "EdgeFactory",
    "GraphFactory",
    "ToolFactory",
    "CheckpointerFactory",
    "StrategyFactory",
    "CallbackFactory",
    
    # 主工厂
    "MasterFactory",
    "get_master_factory",
    
    # 构建器
    "AgentBuilder",
    "PipelineBuilder",
    "WorkflowBuilder",
    "Pipeline",
    
    # 模板
    "AgentTemplate",
    "RESEARCH_AGENT_TEMPLATE",
    "CODING_AGENT_TEMPLATE",
    "DATA_ANALYSIS_AGENT_TEMPLATE",
    "ML_TRAINING_AGENT_TEMPLATE",
    "CUSTOMER_SERVICE_AGENT_TEMPLATE",
    "QA_AGENT_TEMPLATE",
    "CREATIVE_WRITING_AGENT_TEMPLATE",
    
    # 便捷函数 - Agent 创建
    "create_react_agent",
    "create_plan_execute_agent",
    "create_reflexion_agent",
    "create_multi_agent_system",
    "create_tool_calling_agent",
    "create_conversational_agent",
    "create_chain_of_thought_agent",
    "create_self_ask_agent",
    "create_hierarchical_agent",
    "create_workflow_agent",
    "create_agent_from_template",
    
    # 便捷函数 - 预置模板
    "create_research_agent",
    "create_coding_agent",
    "create_data_analyst_agent",
    "create_ml_training_agent",
    "create_customer_service_agent",
    "create_qa_agent",
    "create_creative_writing_agent",
    
    # 增强便捷函数
    "build_enhanced_agent",
    "create_production_agent",
    "create_enhanced_agent",
    "configure_agent_tools",
    "get_agent_diagnostics",
    "factory_health_check",
    
    # 辅助类（从 agents 模块导入）
    "AgentStateHelper",
    "AgentNodeHelper",
    "AgentEdgeHelper",
    "AgentGraphHelper",
    "AgentToolHelper",
    "AgentCheckpointerHelper",
    "AgentBuiltinToolHelper",
    "EnhancedAgentBuilder",
    
    # 状态模块增强类型
    "StateExecutionMode",
    "InterruptType",
    "ErrorType",
    "ContextType",
    "PriorityLevel",
    "StateExecutionContext",
    
    # 快捷函数
    "quick_agent",
    "quick_pipeline",
    "quick_workflow",
    "build_agent",
    "build_pipeline",
    "build_workflow",
]

