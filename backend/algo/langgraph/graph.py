"""
状态图定义（生产级）

定义 LangGraph 核心图结构：
- StateGraph: 状态图
- GraphBuilder: 图构建器
- CompiledGraph: 编译后的图
- Subgraph: 子图
- GraphRunner: 图执行器
- GraphAnalyzer: 图分析器

生产级特性：
- 子图嵌套与组合
- 并行节点执行
- 循环检测与死锁预防
- 中断与恢复
- 执行历史与回放
- 性能指标收集
- 图可视化（DOT, Mermaid, JSON）
- 事件与回调系统
- 配置热更新
- 图序列化/反序列化
"""

import asyncio
import json
import logging
import threading
import time
import uuid
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Any, Dict, List, Optional, Union, Callable,
    TypeVar, Type, Set, Tuple, Iterator, AsyncIterator, Generator,
    Awaitable
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps

from .state import AgentState, StateCheckpoint, AgentStatus, AgentMessage
from .nodes import BaseNode, StartNode, EndNode, NodeType, LLMNode, ToolNode
from .edges import (
    Edge, ConditionalEdge, LoopEdge, ParallelEdge, EdgeType,
    EdgeManager, EdgeMetrics, TimeoutEdge, RetryEdge, FallbackEdge,
    WeightedEdge, CircuitBreakerEdge, RoutingStrategy
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=AgentState)


class GraphStatus(Enum):
    """图状态"""
    BUILDING = "building"
    COMPILED = "compiled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"      # 并行执行
    ASYNC = "async"           # 异步执行


class InterruptType(Enum):
    """中断类型"""
    BEFORE_NODE = "before_node"
    AFTER_NODE = "after_node"
    ON_CONDITION = "on_condition"
    ON_ERROR = "on_error"


@dataclass
class GraphConfig:
    """图配置（增强版）"""
    name: str = "default"
    description: str = ""
    version: str = "1.0.0"
    
    # 执行控制
    max_iterations: int = 10
    timeout: float = 300.0  # 总超时（秒）
    node_timeout: float = 60.0  # 单节点超时
    max_parallel_nodes: int = 4  # 最大并行节点数
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # 检查点
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1  # 每N步保存检查点
    auto_resume: bool = True  # 自动从检查点恢复
    
    # 流式与调试
    enable_streaming: bool = False
    debug: bool = False
    trace_execution: bool = False  # 跟踪执行路径
    
    # 错误处理
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    fail_fast: bool = False  # 遇错立即失败
    
    # 并行执行
    enable_parallel: bool = False
    thread_pool_size: int = 4
    
    # 事件与回调
    enable_events: bool = True
    enable_callbacks: bool = True
    
    # 指标
    enable_metrics: bool = True
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "node_timeout": self.node_timeout,
            "max_parallel_nodes": self.max_parallel_nodes,
            "execution_mode": self.execution_mode.value,
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_interval": self.checkpoint_interval,
            "enable_streaming": self.enable_streaming,
            "debug": self.debug,
            "trace_execution": self.trace_execution,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "enable_parallel": self.enable_parallel,
            "enable_events": self.enable_events,
            "enable_metrics": self.enable_metrics,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphConfig':
        if 'execution_mode' in data and isinstance(data['execution_mode'], str):
            data['execution_mode'] = ExecutionMode(data['execution_mode'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def merge(self, override: Dict[str, Any]) -> 'GraphConfig':
        """合并配置覆盖"""
        config_dict = self.to_dict()
        config_dict.update(override)
        return GraphConfig.from_dict(config_dict)


@dataclass
class ExecutionContext:
    """执行上下文"""
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thread_id: str = ""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_run_id: Optional[str] = None
    
    # 执行状态
    current_node: str = ""
    visited_nodes: List[str] = field(default_factory=list)
    execution_path: List[Tuple[str, str]] = field(default_factory=list)  # (from, to)
    
    # 时间
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 状态
    is_interrupted: bool = False
    interrupt_reason: str = ""
    error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "current_node": self.current_node,
            "visited_nodes": self.visited_nodes,
            "execution_path": self.execution_path,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "config": self.config,
            "is_interrupted": self.is_interrupted,
            "interrupt_reason": self.interrupt_reason,
            "error": str(self.error) if self.error else None
        }


@dataclass
class GraphMetrics:
    """图执行指标"""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_execution_time_ms: float = 0.0
    total_nodes_executed: int = 0
    node_execution_times: Dict[str, float] = field(default_factory=dict)
    node_execution_counts: Dict[str, int] = field(default_factory=dict)
    edge_traversal_counts: Dict[str, int] = field(default_factory=dict)
    last_execution_time: Optional[datetime] = None
    
    def record_run(self, success: bool, execution_time_ms: float) -> None:
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        self.total_execution_time_ms += execution_time_ms
        self.last_execution_time = datetime.utcnow()
    
    def record_node_execution(self, node_name: str, execution_time_ms: float) -> None:
        self.total_nodes_executed += 1
        if node_name not in self.node_execution_times:
            self.node_execution_times[node_name] = 0.0
            self.node_execution_counts[node_name] = 0
        self.node_execution_times[node_name] += execution_time_ms
        self.node_execution_counts[node_name] += 1
    
    def record_edge_traversal(self, edge_key: str) -> None:
        if edge_key not in self.edge_traversal_counts:
            self.edge_traversal_counts[edge_key] = 0
        self.edge_traversal_counts[edge_key] += 1
    
    @property
    def success_rate(self) -> float:
        return self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        return self.total_execution_time_ms / self.total_runs if self.total_runs > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "total_nodes_executed": self.total_nodes_executed,
            "node_execution_times": {k: round(v, 2) for k, v in self.node_execution_times.items()},
            "node_execution_counts": self.node_execution_counts,
            "edge_traversal_counts": self.edge_traversal_counts,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None
        }
    
    def reset(self) -> None:
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.total_execution_time_ms = 0.0
        self.total_nodes_executed = 0
        self.node_execution_times.clear()
        self.node_execution_counts.clear()
        self.edge_traversal_counts.clear()
        self.last_execution_time = None


@dataclass
class ExecutionEvent:
    """执行事件"""
    event_type: str
    node_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "node_name": self.node_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


# 回调类型
GraphCallback = Callable[[ExecutionEvent], None]
AsyncGraphCallback = Callable[[ExecutionEvent], Awaitable[None]]


class StateGraph:
    """状态图（增强版）
    
    LangGraph 的核心组件，管理节点和边的关系。
    
    生产级特性：
    - 子图嵌套
    - 并行节点
    - 循环检测
    - 图合并
    - 序列化/反序列化
    - 可视化导出
    """
    
    def __init__(self, state_class: Type[T] = AgentState, config: GraphConfig = None):
        self.state_class = state_class
        self.config = config or GraphConfig()
        self.status = GraphStatus.BUILDING
        self.graph_id = str(uuid.uuid4())[:8]
        
        # 节点和边
        self._nodes: Dict[str, BaseNode] = {}
        self._edges: List[Edge] = []
        self._conditional_edges: Dict[str, ConditionalEdge] = {}
        self._parallel_groups: Dict[str, List[str]] = {}  # 并行节点组
        
        # 边管理器
        self._edge_manager = EdgeManager()
        
        # 特殊节点
        self._entry_point: str = None
        self._finish_points: Set[str] = set()
        
        # 子图
        self._subgraphs: Dict[str, 'StateGraph'] = {}
        
        # 中断点
        self._interrupt_before: Set[str] = set()
        self._interrupt_after: Set[str] = set()
        
        # 节点元数据
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 添加起始和结束节点
        self._nodes["__start__"] = StartNode()
        self._nodes["__end__"] = EndNode()
        self._finish_points.add("__end__")
        
        # 指标
        self._metrics = GraphMetrics()
        
        # 回调
        self._callbacks: List[GraphCallback] = []
        self._async_callbacks: List[AsyncGraphCallback] = []
        
        # 锁
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
    
    def add_node(self, name: str, node: Union[BaseNode, Callable, 'StateGraph'],
                metadata: Dict[str, Any] = None) -> 'StateGraph':
        """添加节点
        
        Args:
            name: 节点名称
            node: 节点实例、可调用对象或子图
            metadata: 节点元数据
        """
        with self._lock:
            if name in self._nodes:
                raise ValueError(f"Node '{name}' already exists")
        
            # 处理子图
            if isinstance(node, StateGraph):
                self._subgraphs[name] = node
                node = self._wrap_subgraph_as_node(name, node)
            elif callable(node) and not isinstance(node, BaseNode):
                # 将函数包装为节点
                node = self._wrap_function_as_node(name, node)
        
            self._nodes[name] = node
            
            # 保存元数据
            if metadata:
                self._node_metadata[name] = metadata
            
        self.logger.debug(f"Added node: {name}")
        return self
    
    def _wrap_function_as_node(self, name: str, func: Callable) -> BaseNode:
        """将函数包装为节点"""
        from .nodes import NodeConfig, NodeType
        
        class FunctionNode(BaseNode):
            def __init__(self, node_name: str, func: Callable):
                config = NodeConfig(
                    name=node_name,
                    node_type=NodeType.CUSTOM,
                    description=func.__doc__ or f"Custom node: {node_name}"
                )
                super().__init__(config)
                self.func = func
                self._is_async = asyncio.iscoroutinefunction(func)
            
            def __call__(self, state: AgentState) -> AgentState:
                result = self.func(state)
                if result is None:
                    return state
                return result
            
            async def ainvoke(self, state: AgentState) -> AgentState:
                if self._is_async:
                    result = await self.func(state)
                else:
                    result = self.func(state)
                if result is None:
                    return state
                return result
        
        return FunctionNode(name, func)
    
    def _wrap_subgraph_as_node(self, name: str, subgraph: 'StateGraph') -> BaseNode:
        """将子图包装为节点"""
        from .nodes import NodeConfig, NodeType
        
        class SubgraphNode(BaseNode):
            def __init__(self, node_name: str, graph: 'StateGraph'):
                config = NodeConfig(
                    name=node_name,
                    node_type=NodeType.CUSTOM,
                    description=f"Subgraph: {graph.config.name}"
                )
                super().__init__(config)
                self.subgraph = graph
                self._compiled = None
            
            def _get_compiled(self) -> 'CompiledGraph':
                if self._compiled is None:
                    self._compiled = self.subgraph.compile()
                return self._compiled
            
            def __call__(self, state: AgentState) -> AgentState:
                return self._get_compiled().invoke(state)
            
            async def ainvoke(self, state: AgentState) -> AgentState:
                return await self._get_compiled().ainvoke(state)
        
        return SubgraphNode(name, subgraph)
    
    def add_edge(self, source: str, target: str, **kwargs) -> 'StateGraph':
        """添加普通边"""
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        edge = Edge(source=source, target=target, **kwargs)
        self._edges.append(edge)
        self._edge_manager.add_edge(edge)
            
        self.logger.debug(f"Added edge: {source} -> {target}")
        return self
    
    def add_conditional_edges(self, 
                             source: str,
                             condition: Callable[[T], str],
                             branches: Dict[str, str] = None,
                             **kwargs) -> 'StateGraph':
        """添加条件边
        
        Args:
            source: 源节点
            condition: 条件函数，返回目标节点名
            branches: 条件结果映射
        """
        with self._lock:
            self._validate_node(source)
        
        if branches:
            for target in branches.values():
                self._validate_node(target)
        
        cond_edge = ConditionalEdge(
            source=source,
            target="",
            condition_func=condition,
                branches=branches or {},
                **kwargs
        )
        self._conditional_edges[source] = cond_edge
        self._edge_manager.add_edge(cond_edge)
            
        self.logger.debug(f"Added conditional edge from: {source}")
        return self
    
    def add_parallel_nodes(self, group_name: str, node_names: List[str],
                          merge_func: Callable[[List[AgentState]], AgentState] = None) -> 'StateGraph':
        """添加并行节点组
        
        Args:
            group_name: 组名称
            node_names: 节点名称列表
            merge_func: 结果合并函数
        """
        with self._lock:
            for name in node_names:
                self._validate_node(name)
            
            self._parallel_groups[group_name] = {
                "nodes": node_names,
                "merge_func": merge_func or self._default_merge
            }
            
            self.logger.debug(f"Added parallel group: {group_name} with {len(node_names)} nodes")
            return self
    
    def _default_merge(self, states: List[AgentState]) -> AgentState:
        """默认的状态合并函数"""
        if not states:
            return self.state_class()
        
        # 使用第一个状态作为基础，合并消息
        merged = states[0].copy()
        for state in states[1:]:
            for msg in state.messages:
                if msg.id not in [m.id for m in merged.messages]:
                    merged.messages.append(msg)
        
        return merged
    
    def set_entry_point(self, name: str) -> 'StateGraph':
        """设置入口点"""
        with self._lock:
            self._validate_node(name)
            self._entry_point = name
            # 自动添加从 __start__ 到入口点的边
            self.add_edge("__start__", name)
        return self
    
    def set_finish_point(self, name: str) -> 'StateGraph':
        """设置结束点"""
        with self._lock:
            self._validate_node(name)
            self._finish_points.add(name)
            # 自动添加到 __end__ 的边
            self.add_edge(name, "__end__")
        
        return self
    
    def add_interrupt_before(self, node_name: str) -> 'StateGraph':
        """在节点执行前添加中断点"""
        self._interrupt_before.add(node_name)
        return self
    
    def add_interrupt_after(self, node_name: str) -> 'StateGraph':
        """在节点执行后添加中断点"""
        self._interrupt_after.add(node_name)
        return self
    
    def remove_interrupt(self, node_name: str) -> 'StateGraph':
        """移除中断点"""
        self._interrupt_before.discard(node_name)
        self._interrupt_after.discard(node_name)
        return self
    
    def _validate_node(self, name: str) -> None:
        """验证节点存在"""
        if name not in self._nodes and name not in ["__start__", "__end__"]:
            # 允许引用尚未添加的节点（后续会添加）
            pass
    
    def get_node(self, name: str) -> Optional[BaseNode]:
        """获取节点"""
        return self._nodes.get(name)
    
    def get_nodes(self) -> Dict[str, BaseNode]:
        """获取所有节点"""
        return self._nodes.copy()
    
    def get_edges(self) -> List[Edge]:
        """获取所有边"""
        return self._edges.copy()
    
    def get_subgraph(self, name: str) -> Optional['StateGraph']:
        """获取子图"""
        return self._subgraphs.get(name)
    
    def has_node(self, name: str) -> bool:
        """检查节点是否存在"""
        return name in self._nodes
    
    def remove_node(self, name: str) -> bool:
        """移除节点"""
        with self._lock:
            if name in ["__start__", "__end__"]:
                raise ValueError("Cannot remove special nodes")
            
            if name in self._nodes:
                del self._nodes[name]
                # 移除相关边
                self._edges = [e for e in self._edges if e.source != name and e.target != name]
                if name in self._conditional_edges:
                    del self._conditional_edges[name]
                return True
            return False
    
    def remove_edge(self, source: str, target: str) -> bool:
        """移除边"""
        with self._lock:
            original_len = len(self._edges)
            self._edges = [e for e in self._edges if not (e.source == source and e.target == target)]
            return len(self._edges) < original_len
    
    # ==================== 回调管理 ====================
    
    def add_callback(self, callback: GraphCallback) -> None:
        """添加同步回调"""
        self._callbacks.append(callback)
    
    def add_async_callback(self, callback: AsyncGraphCallback) -> None:
        """添加异步回调"""
        self._async_callbacks.append(callback)
    
    def remove_callback(self, callback: GraphCallback) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # ==================== 编译 ====================
    
    def compile(self, checkpointer=None, **kwargs) -> 'CompiledGraph':
        """编译图
        
        验证图结构并生成可执行的编译图。
        """
        # 验证图结构
        errors = self.validate()
        if errors:
            raise ValueError(f"Graph validation failed: {errors}")
        
        # 创建编译图
        compiled = CompiledGraph(
            graph=self,
            checkpointer=checkpointer,
            **kwargs
        )
        
        self.status = GraphStatus.COMPILED
        self.logger.info(f"Graph '{self.config.name}' compiled successfully")
        
        return compiled
    
    def validate(self) -> List[str]:
        """验证图结构
        
        Returns:
            错误消息列表（空列表表示验证通过）
        """
        errors = []
        
        # 检查入口点
        if not self._entry_point:
            errors.append("Entry point not set. Use set_entry_point()")
        
        # 检查所有边的节点是否存在
        for edge in self._edges:
            if edge.source not in self._nodes:
                errors.append(f"Edge source node '{edge.source}' not found")
            if edge.target not in self._nodes:
                errors.append(f"Edge target node '{edge.target}' not found")
        
        # 检查条件边
        for source, cond_edge in self._conditional_edges.items():
            if source not in self._nodes:
                errors.append(f"Conditional edge source '{source}' not found")
            for target in cond_edge.branches.values():
                if target not in self._nodes:
                    errors.append(f"Conditional edge target '{target}' not found")
        
        # 检查并行组
        for group_name, group_info in self._parallel_groups.items():
            for node_name in group_info["nodes"]:
                if node_name not in self._nodes:
                    errors.append(f"Parallel group '{group_name}' node '{node_name}' not found")
        
        # 检查循环（可选，允许循环但需要有终止条件）
        cycles = self._detect_cycles()
        if cycles and self.config.debug:
            self.logger.warning(f"Cycles detected in graph: {cycles}")
        
        if not errors:
            self.logger.debug("Graph validation passed")
        
        return errors
    
    def _validate_graph(self) -> None:
        """验证图结构（兼容旧版本）"""
        errors = self.validate()
        if errors:
            raise ValueError(f"Graph validation failed: {errors}")
    
    def _detect_cycles(self) -> List[List[str]]:
        """检测图中的循环"""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # 获取所有邻居
            neighbors = []
            for edge in self._edges:
                if edge.source == node:
                    neighbors.append(edge.target)
            if node in self._conditional_edges:
                neighbors.extend(self._conditional_edges[node].branches.values())
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # 发现循环
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            path.pop()
            rec_stack.remove(node)
        
        for node in self._nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    # ==================== 合并与组合 ====================
    
    def merge(self, other: 'StateGraph', prefix: str = "") -> 'StateGraph':
        """合并另一个图
        
        Args:
            other: 要合并的图
            prefix: 节点名前缀（避免冲突）
        
        Returns:
            合并后的新图
        """
        merged = StateGraph(
            state_class=self.state_class,
            config=GraphConfig(
                name=f"{self.config.name}+{other.config.name}",
                **{k: v for k, v in self.config.to_dict().items() if k != 'name'}
            )
        )
        
        # 复制当前图的节点和边
        for name, node in self._nodes.items():
            if name not in ["__start__", "__end__"]:
                merged.add_node(name, node)
        
        for edge in self._edges:
            if edge.source not in ["__start__", "__end__"] or edge.target not in ["__start__", "__end__"]:
                merged.add_edge(edge.source, edge.target)
        
        # 添加另一个图的节点和边（带前缀）
        for name, node in other._nodes.items():
            if name not in ["__start__", "__end__"]:
                new_name = f"{prefix}{name}" if prefix else name
                if new_name not in merged._nodes:
                    merged.add_node(new_name, node)
        
        for edge in other._edges:
            source = f"{prefix}{edge.source}" if prefix and edge.source not in ["__start__", "__end__"] else edge.source
            target = f"{prefix}{edge.target}" if prefix and edge.target not in ["__start__", "__end__"] else edge.target
            if source not in ["__start__", "__end__"] or target not in ["__start__", "__end__"]:
                merged.add_edge(source, target)
        
        return merged
    
    def chain(self, other: 'StateGraph', connector_edge: bool = True) -> 'StateGraph':
        """链接两个图（串联）
        
        当前图的结束连接到另一个图的开始。
        """
        chained = self.merge(other, prefix="")
        
        if connector_edge:
            # 连接当前图的结束点到另一个图的入口点
            for finish_point in self._finish_points:
                if finish_point != "__end__" and other._entry_point:
                    chained.add_edge(finish_point, other._entry_point)
        
        return chained
    
    # ==================== 序列化 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "graph_id": self.graph_id,
            "config": self.config.to_dict(),
            "nodes": list(self._nodes.keys()),
            "edges": [
                {"source": e.source, "target": e.target, "type": e.edge_type.value}
                for e in self._edges
            ],
            "conditional_edges": {
                source: {
                    "branches": cond.branches,
                    "target": cond.target
                }
                for source, cond in self._conditional_edges.items()
            },
            "parallel_groups": {
                name: {"nodes": info["nodes"]}
                for name, info in self._parallel_groups.items()
            },
            "entry_point": self._entry_point,
            "finish_points": list(self._finish_points),
            "interrupt_before": list(self._interrupt_before),
            "interrupt_after": list(self._interrupt_after),
            "subgraphs": {name: sg.to_dict() for name, sg in self._subgraphs.items()},
            "node_metadata": self._node_metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    # ==================== 可视化 ====================
    
    def to_mermaid(self) -> str:
        """导出为 Mermaid 图表格式"""
        lines = ["graph TD"]
        
        # 节点样式
        for name in self._nodes:
            if name == "__start__":
                lines.append(f"    {name}((Start))")
            elif name == "__end__":
                lines.append(f"    {name}((End))")
            else:
                lines.append(f"    {name}[{name}]")
        
        # 普通边
        for edge in self._edges:
            lines.append(f"    {edge.source} --> {edge.target}")
        
        # 条件边
        for source, cond in self._conditional_edges.items():
            for branch, target in cond.branches.items():
                lines.append(f"    {source} -->|{branch}| {target}")
        
        return "\n".join(lines)
    
    def to_dot(self) -> str:
        """导出为 DOT 格式（Graphviz）"""
        lines = [f'digraph "{self.config.name}" {{']
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box];')
        
        # 特殊节点样式
        lines.append('    __start__ [shape=circle, label="Start"];')
        lines.append('    __end__ [shape=doublecircle, label="End"];')
        
        # 普通节点
        for name in self._nodes:
            if name not in ["__start__", "__end__"]:
                metadata = self._node_metadata.get(name, {})
                label = metadata.get('label', name)
                lines.append(f'    {name} [label="{label}"];')
        
        # 边
        for edge in self._edges:
            lines.append(f'    {edge.source} -> {edge.target};')
        
        # 条件边
        for source, cond in self._conditional_edges.items():
            for branch, target in cond.branches.items():
                lines.append(f'    {source} -> {target} [label="{branch}"];')
        
        lines.append('}')
        return "\n".join(lines)
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        nodes = []
        edges_data = []
        
        for name, node in self._nodes.items():
            node_data = {
                "id": name,
                "label": name,
                "type": node.node_type.value if hasattr(node, 'node_type') else "custom",
                "metadata": self._node_metadata.get(name, {}),
                "is_entry": name == self._entry_point,
                "is_finish": name in self._finish_points,
                "has_interrupt_before": name in self._interrupt_before,
                "has_interrupt_after": name in self._interrupt_after
            }
            nodes.append(node_data)
        
        for edge in self._edges:
            edge_data = {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "is_conditional": False
            }
            edges_data.append(edge_data)
        
        for source, cond in self._conditional_edges.items():
            for branch, target in cond.branches.items():
                edge_data = {
                    "source": source,
                    "target": target,
                    "type": "conditional",
                    "label": branch,
                    "is_conditional": True
                }
                edges_data.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges_data,
            "config": self.config.to_dict(),
            "parallel_groups": list(self._parallel_groups.keys()),
            "subgraphs": list(self._subgraphs.keys())
        }
    
    # ==================== 指标 ====================
    
    def get_metrics(self) -> GraphMetrics:
        """获取图指标"""
        return self._metrics
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self._metrics.reset()
    
    # =========================================================================
    # 派生功能：调用 state 模块 (StateCheckpoint, AgentStatus)
    # =========================================================================
    
    def create_checkpoint(self, state: T, description: str = "") -> StateCheckpoint:
        """创建状态检查点
        
        调用 state 模块的 StateCheckpoint 类
        
        Args:
            state: 当前状态
            description: 检查点描述
        
        Returns:
            StateCheckpoint 实例
        """
        checkpoint = StateCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            thread_id=state.thread_id,
            state=state.to_dict() if hasattr(state, 'to_dict') else {},
            parent_id=None,
            description=description,
            version=state.iteration,
            branch="main",
            tags=list(self.config.tags),
            metadata={
                "graph_name": self.config.name,
                "graph_id": self.graph_id,
                "node_count": len(self._nodes),
                "created_at": datetime.utcnow().isoformat()
            }
        )
        self.logger.debug(f"Created checkpoint: {checkpoint.checkpoint_id}")
        return checkpoint
    
    def validate_checkpoint(self, checkpoint: StateCheckpoint) -> bool:
        """验证检查点完整性
        
        调用 StateCheckpoint 的 verify_integrity 方法
        
        Args:
            checkpoint: 状态检查点
        
        Returns:
            是否通过验证
        """
        is_valid = checkpoint.verify_integrity()
        if not is_valid:
            self.logger.warning(f"Checkpoint {checkpoint.checkpoint_id} integrity check failed")
        return is_valid
    
    def get_status_for_state(self, state: T) -> AgentStatus:
        """根据状态获取 Agent 状态枚举
        
        调用 state 模块的 AgentStatus 枚举
        
        Args:
            state: 当前状态
        
        Returns:
            AgentStatus 枚举值
        """
        # 检查状态中是否有 status 属性
        if hasattr(state, 'status'):
            status_val = state.status
            if isinstance(status_val, AgentStatus):
                return status_val
            elif isinstance(status_val, str):
                try:
                    return AgentStatus(status_val)
                except ValueError:
                    pass
        
        # 根据迭代次数判断状态
        if state.iteration == 0:
            return AgentStatus.IDLE
        elif state.should_continue():
            return AgentStatus.RUNNING
        else:
            return AgentStatus.COMPLETED
    
    def set_state_status(self, state: T, status: AgentStatus) -> T:
        """设置状态的 Agent 状态
        
        调用 state 模块的 AgentStatus 枚举
        
        Args:
            state: 当前状态
            status: 要设置的状态
        
        Returns:
            更新后的状态
        """
        if hasattr(state, 'status'):
            state.status = status
        state.metadata['agent_status'] = status.value
        self.logger.debug(f"Set state status to: {status.value}")
        return state
    
    # =========================================================================
    # 派生功能：调用 nodes 模块 (NodeType)
    # =========================================================================
    
    def get_nodes_by_type(self, node_type: NodeType) -> Dict[str, BaseNode]:
        """按类型获取节点
        
        调用 nodes 模块的 NodeType 枚举
        
        Args:
            node_type: 节点类型
        
        Returns:
            匹配类型的节点字典
        """
        result = {}
        for name, node in self._nodes.items():
            if hasattr(node, 'node_type') and node.node_type == node_type:
                result[name] = node
            elif hasattr(node, 'config') and hasattr(node.config, 'node_type'):
                if node.config.node_type == node_type:
                    result[name] = node
        return result
    
    def count_nodes_by_type(self) -> Dict[NodeType, int]:
        """统计各类型节点数量
        
        调用 nodes 模块的 NodeType 枚举
        
        Returns:
            类型到数量的映射
        """
        counts = {nt: 0 for nt in NodeType}
        for name, node in self._nodes.items():
            if hasattr(node, 'node_type'):
                counts[node.node_type] = counts.get(node.node_type, 0) + 1
            elif hasattr(node, 'config') and hasattr(node.config, 'node_type'):
                counts[node.config.node_type] = counts.get(node.config.node_type, 0) + 1
        return {k: v for k, v in counts.items() if v > 0}
    
    def has_node_type(self, node_type: NodeType) -> bool:
        """检查是否存在指定类型的节点
        
        调用 nodes 模块的 NodeType 枚举
        
        Args:
            node_type: 节点类型
        
        Returns:
            是否存在
        """
        return len(self.get_nodes_by_type(node_type)) > 0
    
    # =========================================================================
    # 派生功能：调用 edges 模块 (各种边类型)
    # =========================================================================
    
    def add_loop_edge(self, source: str, target: str, 
                     max_iterations: int = 10,
                     continue_condition: Callable[[T], bool] = None) -> 'StateGraph':
        """添加循环边
        
        调用 edges 模块的 LoopEdge 类
        
        Args:
            source: 源节点
            target: 目标节点
            max_iterations: 最大迭代次数
            continue_condition: 继续条件函数
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        loop_edge = LoopEdge(
            source=source,
            target=target,
            max_iterations=max_iterations,
            continue_condition=continue_condition
        )
        self._edges.append(loop_edge)
        self._edge_manager.add_edge(loop_edge)
        
        self.logger.debug(f"Added loop edge: {source} -> {target} (max_iter={max_iterations})")
        return self
    
    def add_parallel_edge(self, source: str, targets: List[str],
                         merge_strategy: str = "all",
                         timeout: float = 30.0) -> 'StateGraph':
        """添加并行边
        
        调用 edges 模块的 ParallelEdge 类
        
        Args:
            source: 源节点
            targets: 目标节点列表
            merge_strategy: 合并策略（all, first, majority）
            timeout: 并行执行超时
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            for target in targets:
                self._validate_node(target)
        
        parallel_edge = ParallelEdge(
            source=source,
            target=targets[0] if targets else "",
            targets=targets,
            merge_strategy=merge_strategy,
            timeout=timeout
        )
        self._edges.append(parallel_edge)
        self._edge_manager.add_edge(parallel_edge)
        
        self.logger.debug(f"Added parallel edge: {source} -> [{', '.join(targets)}]")
        return self
    
    def add_timeout_edge(self, source: str, target: str,
                        timeout_seconds: float = 30.0,
                        timeout_target: str = "__timeout__",
                        on_timeout: Callable[[T], T] = None) -> 'StateGraph':
        """添加超时边
        
        调用 edges 模块的 TimeoutEdge 类
        
        Args:
            source: 源节点
            target: 目标节点
            timeout_seconds: 超时时间（秒）
            timeout_target: 超时后跳转的目标节点
            on_timeout: 超时回调函数
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        timeout_edge = TimeoutEdge(
            source=source,
            target=target,
            timeout_seconds=timeout_seconds,
            timeout_target=timeout_target,
            on_timeout=on_timeout
        )
        self._edges.append(timeout_edge)
        self._edge_manager.add_edge(timeout_edge)
        
        self.logger.debug(f"Added timeout edge: {source} -> {target} (timeout={timeout_seconds}s)")
        return self
    
    def add_retry_edge(self, source: str, target: str,
                      max_retries: int = 3,
                      retry_delay: float = 1.0,
                      backoff_multiplier: float = 2.0,
                      fallback_target: str = "") -> 'StateGraph':
        """添加重试边
        
        调用 edges 模块的 RetryEdge 类
        
        Args:
            source: 源节点
            target: 目标节点
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟
            backoff_multiplier: 退避乘数
            fallback_target: 重试失败后的降级目标
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        retry_edge = RetryEdge(
            source=source,
            target=target,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backoff_multiplier=backoff_multiplier,
            fallback_target=fallback_target
        )
        self._edges.append(retry_edge)
        self._edge_manager.add_edge(retry_edge)
        
        self.logger.debug(f"Added retry edge: {source} -> {target} (max_retries={max_retries})")
        return self
    
    def add_fallback_edge(self, source: str, target: str,
                         fallback_targets: List[str] = None,
                         fallback_conditions: Dict[str, Callable[[T], bool]] = None) -> 'StateGraph':
        """添加降级边
        
        调用 edges 模块的 FallbackEdge 类
        
        Args:
            source: 源节点
            target: 主目标节点
            fallback_targets: 降级目标节点列表
            fallback_conditions: 降级条件映射
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        fallback_edge = FallbackEdge(
            source=source,
            target=target,
            fallback_targets=fallback_targets or [],
            fallback_conditions=fallback_conditions or {}
        )
        self._edges.append(fallback_edge)
        self._edge_manager.add_edge(fallback_edge)
        
        self.logger.debug(f"Added fallback edge: {source} -> {target} (fallbacks={fallback_targets})")
        return self
    
    def add_weighted_edge(self, source: str,
                         weighted_targets: Dict[str, float],
                         routing_strategy: str = "weighted_random") -> 'StateGraph':
        """添加带权重边
        
        调用 edges 模块的 WeightedEdge 类
        
        Args:
            source: 源节点
            weighted_targets: 目标节点及其权重 {target: weight}
            routing_strategy: 路由策略
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            for target in weighted_targets:
                self._validate_node(target)
        
        # 找到第一个目标作为主目标
        first_target = list(weighted_targets.keys())[0] if weighted_targets else ""
        
        weighted_edge = WeightedEdge(
            source=source,
            target=first_target,
            weighted_targets=weighted_targets,
            routing_strategy=RoutingStrategy(routing_strategy) if isinstance(routing_strategy, str) else routing_strategy
        )
        self._edges.append(weighted_edge)
        self._edge_manager.add_edge(weighted_edge)
        
        self.logger.debug(f"Added weighted edge: {source} -> {weighted_targets}")
        return self
    
    def add_circuit_breaker_edge(self, source: str, target: str,
                                failure_threshold: int = 5,
                                recovery_timeout: float = 30.0,
                                fallback_target: str = "") -> 'StateGraph':
        """添加断路器边
        
        调用 edges 模块的 CircuitBreakerEdge 类
        
        Args:
            source: 源节点
            target: 目标节点
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时
            fallback_target: 断路时的降级目标
        
        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self._validate_node(source)
            self._validate_node(target)
        
        circuit_breaker_edge = CircuitBreakerEdge(
            source=source,
            target=target,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            fallback_target=fallback_target
        )
        self._edges.append(circuit_breaker_edge)
        self._edge_manager.add_edge(circuit_breaker_edge)
        
        self.logger.debug(f"Added circuit breaker edge: {source} -> {target} (threshold={failure_threshold})")
        return self
    
    def get_edge_metrics(self) -> Dict[str, EdgeMetrics]:
        """获取所有边的指标
        
        调用 edges 模块的 EdgeMetrics 类
        
        Returns:
            边名称到指标的映射
        """
        metrics = {}
        for edge in self._edges:
            edge_key = f"{edge.source}->{edge.target}"
            if hasattr(edge, 'metrics'):
                metrics[edge_key] = edge.metrics
            else:
                # 创建空指标
                metrics[edge_key] = EdgeMetrics()
        return metrics
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """按类型获取边
        
        调用 edges 模块的 EdgeType 枚举
        
        Args:
            edge_type: 边类型
        
        Returns:
            匹配类型的边列表
        """
        return [edge for edge in self._edges if edge.edge_type == edge_type]
    
    def count_edges_by_type(self) -> Dict[EdgeType, int]:
        """统计各类型边数量
        
        调用 edges 模块的 EdgeType 枚举
        
        Returns:
            类型到数量的映射
        """
        counts = {}
        for edge in self._edges:
            edge_type = edge.edge_type
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    # =========================================================================
    # 派生功能：调用 contextlib (contextmanager)
    # =========================================================================
    
    @contextmanager
    def execution_context(self, state: T = None):
        """执行上下文管理器
        
        调用 contextlib 模块的 contextmanager 装饰器
        
        使用方式：
            with graph.execution_context(state) as ctx:
                # 执行操作
                pass
        
        Args:
            state: 初始状态（可选）
        
        Yields:
            执行上下文字典
        """
        context = {
            'graph_id': self.graph_id,
            'graph_name': self.config.name,
            'start_time': datetime.utcnow(),
            'state': state,
            'checkpoint': None,
            'error': None,
            'success': False
        }
        
        try:
            # 创建检查点
            if state:
                context['checkpoint'] = self.create_checkpoint(state, "execution_start")
            
            self.logger.debug(f"Entering execution context for graph: {self.config.name}")
            yield context
            
            context['success'] = True
            
        except Exception as e:
            context['error'] = e
            self.logger.error(f"Error in execution context: {e}")
            raise
        finally:
            context['end_time'] = datetime.utcnow()
            context['duration_ms'] = (context['end_time'] - context['start_time']).total_seconds() * 1000
            self.logger.debug(f"Exiting execution context, success={context['success']}, duration={context['duration_ms']:.2f}ms")
    
    @contextmanager
    def node_execution_scope(self, node_name: str):
        """节点执行作用域
        
        调用 contextlib 模块的 contextmanager 装饰器
        
        使用方式：
            with graph.node_execution_scope("my_node") as scope:
                # 执行节点逻辑
                pass
        
        Args:
            node_name: 节点名称
        
        Yields:
            节点执行作用域字典
        """
        scope = {
            'node_name': node_name,
            'start_time': time.time(),
            'metrics': EdgeMetrics(),
            'error': None,
            'success': False
        }
        
        try:
            self.logger.debug(f"Entering node scope: {node_name}")
            yield scope
            scope['success'] = True
            
        except Exception as e:
            scope['error'] = e
            scope['metrics'].record_traversal(
                (time.time() - scope['start_time']) * 1000,
                success=False
            )
            raise
        finally:
            elapsed_ms = (time.time() - scope['start_time']) * 1000
            if scope['success']:
                scope['metrics'].record_traversal(elapsed_ms, success=True)
            self._metrics.record_node_execution(node_name, elapsed_ms)
            self.logger.debug(f"Exiting node scope: {node_name}, elapsed={elapsed_ms:.2f}ms")
    
    @contextmanager
    def interrupt_guard(self, node_name: str, interrupt_type: InterruptType = InterruptType.BEFORE_NODE):
        """中断保护作用域
        
        调用 contextlib 模块的 contextmanager 装饰器
        在作用域内临时添加中断点，退出后自动移除
        
        Args:
            node_name: 节点名称
            interrupt_type: 中断类型
        
        Yields:
            None
        """
        # 添加中断点
        if interrupt_type == InterruptType.BEFORE_NODE:
            self._interrupt_before.add(node_name)
        elif interrupt_type == InterruptType.AFTER_NODE:
            self._interrupt_after.add(node_name)
        
        try:
            self.logger.debug(f"Interrupt guard active for: {node_name} ({interrupt_type.value})")
            yield
        finally:
            # 移除中断点
            if interrupt_type == InterruptType.BEFORE_NODE:
                self._interrupt_before.discard(node_name)
            elif interrupt_type == InterruptType.AFTER_NODE:
                self._interrupt_after.discard(node_name)
            self.logger.debug(f"Interrupt guard released for: {node_name}")


class CompiledGraph:
    """编译后的图（增强版）
    
    可执行的图，支持同步和异步执行。
    
    生产级特性：
    - 中断与恢复
    - 并行节点执行
    - 执行历史
    - 事件回调
    - 性能指标
    - 超时控制
    - 错误恢复
    """
    
    def __init__(self, graph: StateGraph, checkpointer=None, **kwargs):
        self.graph = graph
        self.checkpointer = checkpointer
        self.config = graph.config
        self.logger = logging.getLogger(f"{__name__}.CompiledGraph.{graph.config.name}")
        
        # 构建邻接表
        self._adjacency: Dict[str, List[str]] = {}
        self._build_adjacency()
        
        # 执行状态
        self._is_running = False
        self._is_interrupted = False
        self._interrupt_node: Optional[str] = None
        self._current_context: Optional[ExecutionContext] = None
        
        # 执行历史
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history_size = kwargs.get('max_history_size', 100)
        
        # 线程池（并行执行）
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        if self.config.enable_parallel:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # 指标
        self._metrics = graph._metrics
        
        # 回调
        self._callbacks = graph._callbacks
        self._async_callbacks = graph._async_callbacks
        
        # 事件队列
        self._event_queue: List[ExecutionEvent] = []
        
        # 锁
        self._lock = threading.RLock()
    
    def _build_adjacency(self) -> None:
        """构建邻接表"""
        for node_name in self.graph._nodes:
            self._adjacency[node_name] = []
        
        for edge in self.graph._edges:
            if edge.target not in self._adjacency[edge.source]:
                self._adjacency[edge.source].append(edge.target)
    
    # ==================== 主要执行方法 ====================
    
    def invoke(self, 
              input_data: Union[str, Dict[str, Any], T],
              config: Dict[str, Any] = None,
              interrupt_before: List[str] = None,
              interrupt_after: List[str] = None) -> T:
        """同步执行图
        
        Args:
            input_data: 输入数据（字符串、字典或状态对象）
            config: 运行时配置
            interrupt_before: 在这些节点前中断
            interrupt_after: 在这些节点后中断
        
        Returns:
            最终状态
        """
        start_time = time.time()
        success = False
        
        try:
            # 初始化状态和上下文
            state = self._initialize_state(input_data, config)
            context = self._create_context(state, config)
            
            # 设置中断点
            if interrupt_before:
                for node in interrupt_before:
                    self.graph._interrupt_before.add(node)
            if interrupt_after:
                for node in interrupt_after:
                    self.graph._interrupt_after.add(node)
            
            # 执行图
            state = self._execute(state, context)
            success = True
            
            return state
            
        except InterruptedError as e:
            # 中断不算失败
            self.logger.info(f"Execution interrupted: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_run(success, elapsed_ms)
            
            # 清理临时中断点
            if interrupt_before:
                for node in interrupt_before:
                    self.graph._interrupt_before.discard(node)
            if interrupt_after:
                for node in interrupt_after:
                    self.graph._interrupt_after.discard(node)
    
    async def ainvoke(self,
                     input_data: Union[str, Dict[str, Any], T],
                     config: Dict[str, Any] = None,
                     interrupt_before: List[str] = None,
                     interrupt_after: List[str] = None) -> T:
        """异步执行图"""
        start_time = time.time()
        success = False
        
        try:
            state = self._initialize_state(input_data, config)
            context = self._create_context(state, config)
            
            if interrupt_before:
                for node in interrupt_before:
                    self.graph._interrupt_before.add(node)
            if interrupt_after:
                for node in interrupt_after:
                    self.graph._interrupt_after.add(node)
            
            state = await self._aexecute(state, context)
            success = True
            
            return state
            
        except InterruptedError as e:
            self.logger.info(f"Async execution interrupted: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Async execution failed: {e}")
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_run(success, elapsed_ms)
            
            if interrupt_before:
                for node in interrupt_before:
                    self.graph._interrupt_before.discard(node)
            if interrupt_after:
                for node in interrupt_after:
                    self.graph._interrupt_after.discard(node)
    
    def stream(self,
              input_data: Union[str, Dict[str, Any], T],
              config: Dict[str, Any] = None) -> Generator[Dict[str, Any], None, None]:
        """流式执行图，逐步返回状态"""
        state = self._initialize_state(input_data, config)
        context = self._create_context(state, config)
        
        current_node = "__start__"
        
        while current_node != "__end__":
            # 检查中断
            if self._check_interrupt_before(current_node):
                yield {"event": "interrupt", "node": current_node, "state": state, "type": "before"}
                return
            
            # 执行当前节点
            node = self.graph.get_node(current_node)
            if node:
                node_start = time.time()
                state = node(state)
                node_time = (time.time() - node_start) * 1000
                
                self._metrics.record_node_execution(current_node, node_time)
                self._emit_event("node_executed", current_node, {"execution_time_ms": node_time})
                
                yield {"event": "node_complete", "node": current_node, "state": state}
            
            # 检查中断
            if self._check_interrupt_after(current_node):
                yield {"event": "interrupt", "node": current_node, "state": state, "type": "after"}
                return
            
            # 获取下一个节点
            prev_node = current_node
            current_node = self._get_next_node(current_node, state)
            context.execution_path.append((prev_node, current_node))
            
            # 增加迭代计数
            if current_node not in ["__start__", "__end__"]:
                state.increment_iteration()
            
            # 检查是否应该停止
            if not state.should_continue() and current_node != "__end__":
                yield {"event": "max_iterations", "node": current_node, "state": state}
                current_node = "__end__"
        
        # 执行结束节点
        end_node = self.graph.get_node("__end__")
        if end_node:
            state = end_node(state)
            yield {"event": "complete", "node": "__end__", "state": state}
    
    async def astream(self,
                     input_data: Union[str, Dict[str, Any], T],
                     config: Dict[str, Any] = None) -> AsyncIterator[Dict[str, Any]]:
        """异步流式执行"""
        state = self._initialize_state(input_data, config)
        context = self._create_context(state, config)
        
        current_node = "__start__"
        
        while current_node != "__end__":
            if self._check_interrupt_before(current_node):
                yield {"event": "interrupt", "node": current_node, "state": state, "type": "before"}
                return
            
            node = self.graph.get_node(current_node)
            if node:
                node_start = time.time()
                state = await node.ainvoke(state)
                node_time = (time.time() - node_start) * 1000
                
                self._metrics.record_node_execution(current_node, node_time)
                await self._aemit_event("node_executed", current_node, {"execution_time_ms": node_time})
                
                yield {"event": "node_complete", "node": current_node, "state": state}
            
            if self._check_interrupt_after(current_node):
                yield {"event": "interrupt", "node": current_node, "state": state, "type": "after"}
                return
            
            prev_node = current_node
            current_node = self._get_next_node(current_node, state)
            context.execution_path.append((prev_node, current_node))
            
            if current_node not in ["__start__", "__end__"]:
                state.increment_iteration()
            
            if not state.should_continue() and current_node != "__end__":
                yield {"event": "max_iterations", "node": current_node, "state": state}
                current_node = "__end__"
        
        end_node = self.graph.get_node("__end__")
        if end_node:
            state = await end_node.ainvoke(state)
            yield {"event": "complete", "node": "__end__", "state": state}
    
    # ==================== 中断与恢复 ====================
    
    def interrupt(self, reason: str = "") -> None:
        """中断执行"""
        with self._lock:
            self._is_interrupted = True
            if self._current_context:
                self._current_context.is_interrupted = True
                self._current_context.interrupt_reason = reason
    
    def resume(self, state: T = None, config: Dict[str, Any] = None) -> T:
        """恢复执行
        
        Args:
            state: 要恢复的状态（可选，默认从检查点加载）
            config: 运行时配置
        
        Returns:
            最终状态
        """
        with self._lock:
            self._is_interrupted = False
        
        # 如果没有提供状态，从检查点加载
        if state is None and self.checkpointer:
            thread_id = config.get('thread_id') if config else None
            state = self.checkpointer.get_latest(thread_id)
        
        if state is None:
            raise ValueError("No state to resume from")
        
        # 继续执行
        return self.invoke(state, config)
    
    async def aresume(self, state: T = None, config: Dict[str, Any] = None) -> T:
        """异步恢复执行"""
        with self._lock:
            self._is_interrupted = False
        
        if state is None and self.checkpointer:
            thread_id = config.get('thread_id') if config else None
            state = await asyncio.to_thread(self.checkpointer.get_latest, thread_id)
        
        if state is None:
            raise ValueError("No state to resume from")
        
        return await self.ainvoke(state, config)
    
    def _check_interrupt_before(self, node_name: str) -> bool:
        """检查是否应该在节点前中断"""
        return (self._is_interrupted or 
                node_name in self.graph._interrupt_before)
    
    def _check_interrupt_after(self, node_name: str) -> bool:
        """检查是否应该在节点后中断"""
        return (self._is_interrupted or 
                node_name in self.graph._interrupt_after)
    
    # ==================== 内部执行方法 ====================
    
    def _create_context(self, state: T, config: Dict[str, Any] = None) -> ExecutionContext:
        """创建执行上下文"""
        context = ExecutionContext(
            graph_id=self.graph.graph_id,
            thread_id=state.thread_id,
            start_time=datetime.utcnow(),
            config=config or {}
        )
        self._current_context = context
        return context
    
    def _initialize_state(self,
                         input_data: Union[str, Dict[str, Any], T],
                         config: Dict[str, Any] = None) -> T:
        """初始化状态"""
        if isinstance(input_data, self.graph.state_class):
            state = input_data
        elif isinstance(input_data, dict):
            # 从字典创建状态
            if 'input' not in input_data and 'messages' not in input_data:
                input_data = {'input': str(input_data)}
            state = self.graph.state_class(**input_data)
        else:
            # 字符串输入
            state = self.graph.state_class(input=str(input_data))
            state.add_message(AgentMessage.human(str(input_data)))
        
        # 应用配置
        if config:
            state.max_iterations = config.get('max_iterations', state.max_iterations)
            state.metadata.update(config.get('metadata', {}))
            if 'thread_id' in config:
                state.thread_id = config['thread_id']
        
        return state
    
    def _execute(self, state: T, context: ExecutionContext) -> T:
        """执行图"""
        with self._lock:
            self._is_running = True
        
        current_node = "__start__"
        visited_count: Dict[str, int] = {}
        checkpoint_counter = 0
        
        try:
            while current_node != "__end__":
                # 记录访问次数
                visited_count[current_node] = visited_count.get(current_node, 0) + 1
                context.visited_nodes.append(current_node)
                context.current_node = current_node
                
                # 检查中断
                if self._check_interrupt_before(current_node):
                    raise InterruptedError(f"Interrupted before node: {current_node}")
                
                # 执行节点
                node = self.graph.get_node(current_node)
                if node:
                    node_start = time.time()
                    self.logger.debug(f"Executing node: {current_node}")
                    
                    # 检查是否是并行组
                    if current_node in self.graph._parallel_groups:
                        state = self._execute_parallel_group(current_node, state)
                    else:
                        state = self._execute_node_with_retry(node, state, current_node)
                    
                    node_time = (time.time() - node_start) * 1000
                    self._metrics.record_node_execution(current_node, node_time)
                    self._emit_event("node_executed", current_node, {"execution_time_ms": node_time})
                    
                    # 保存检查点
                    checkpoint_counter += 1
                    if (self.checkpointer and self.config.enable_checkpointing and
                        checkpoint_counter >= self.config.checkpoint_interval):
                        self.checkpointer.save(state)
                        checkpoint_counter = 0
                    
                    # 检查中断
                    if self._check_interrupt_after(current_node):
                        if self.checkpointer:
                            self.checkpointer.save(state)
                        raise InterruptedError(f"Interrupted after node: {current_node}")
                
                # 获取下一个节点
                prev_node = current_node
                next_node = self._get_next_node(current_node, state)
                context.execution_path.append((prev_node, next_node))
                self._metrics.record_edge_traversal(f"{prev_node}->{next_node}")
                
                # 增加迭代计数（只在非特殊节点时）
                if next_node not in ["__start__", "__end__"]:
                    state.increment_iteration()
                
                # 检查是否应该停止
                if not state.should_continue() and next_node != "__end__":
                    self.logger.info("Execution stopped: should_continue() returned False")
                    next_node = "__end__"
                
                current_node = next_node
            
            # 执行结束节点
            end_node = self.graph.get_node("__end__")
            if end_node:
                state = end_node(state)
            
            context.end_time = datetime.utcnow()
            self._record_execution(context, state, success=True)
            
            return state
        
        except Exception as e:
            context.error = e
            context.end_time = datetime.utcnow()
            self._record_execution(context, state, success=False)
            raise
        finally:
            with self._lock:
                self._is_running = False
    
    async def _aexecute(self, state: T, context: ExecutionContext) -> T:
        """异步执行图"""
        with self._lock:
            self._is_running = True
        
        current_node = "__start__"
        checkpoint_counter = 0
        
        try:
            while current_node != "__end__":
                context.visited_nodes.append(current_node)
                context.current_node = current_node
                
                if self._check_interrupt_before(current_node):
                    raise InterruptedError(f"Interrupted before node: {current_node}")
                
                node = self.graph.get_node(current_node)
                if node:
                    node_start = time.time()
                    self.logger.debug(f"Executing node: {current_node}")
                    
                    if current_node in self.graph._parallel_groups:
                        state = await self._aexecute_parallel_group(current_node, state)
                    else:
                        state = await self._aexecute_node_with_retry(node, state, current_node)
                    
                    node_time = (time.time() - node_start) * 1000
                    self._metrics.record_node_execution(current_node, node_time)
                    await self._aemit_event("node_executed", current_node, {"execution_time_ms": node_time})
                    
                    checkpoint_counter += 1
                    if (self.checkpointer and self.config.enable_checkpointing and
                        checkpoint_counter >= self.config.checkpoint_interval):
                        await asyncio.to_thread(self.checkpointer.save, state)
                        checkpoint_counter = 0
                    
                    if self._check_interrupt_after(current_node):
                        if self.checkpointer:
                            await asyncio.to_thread(self.checkpointer.save, state)
                        raise InterruptedError(f"Interrupted after node: {current_node}")
                
                prev_node = current_node
                next_node = self._get_next_node(current_node, state)
                context.execution_path.append((prev_node, next_node))
                
                if next_node not in ["__start__", "__end__"]:
                    state.increment_iteration()
                
                if not state.should_continue() and next_node != "__end__":
                    next_node = "__end__"
                
                current_node = next_node
            
            end_node = self.graph.get_node("__end__")
            if end_node:
                state = await end_node.ainvoke(state)
            
            context.end_time = datetime.utcnow()
            self._record_execution(context, state, success=True)
            
            return state
        
        except Exception as e:
            context.error = e
            context.end_time = datetime.utcnow()
            self._record_execution(context, state, success=False)
            raise
        finally:
            with self._lock:
                self._is_running = False
    
    def _execute_node_with_retry(self, node: BaseNode, state: T, node_name: str) -> T:
        """带重试的节点执行"""
        if not self.config.retry_on_error:
            return node(state)
        
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return node(state)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Node '{node_name}' failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    if self.config.fail_fast:
                        raise
        
        raise last_error
    
    async def _aexecute_node_with_retry(self, node: BaseNode, state: T, node_name: str) -> T:
        """带重试的异步节点执行"""
        if not self.config.retry_on_error:
            return await node.ainvoke(state)
        
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await node.ainvoke(state)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Node '{node_name}' failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    if self.config.fail_fast:
                        raise
        
        raise last_error
    
    def _execute_parallel_group(self, group_name: str, state: T) -> T:
        """执行并行节点组"""
        group_info = self.graph._parallel_groups[group_name]
        node_names = group_info["nodes"]
        merge_func = group_info["merge_func"]
        
        if not self._thread_pool:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        futures = []
        for node_name in node_names:
            node = self.graph.get_node(node_name)
            if node:
                # 每个并行节点使用状态的副本
                state_copy = state.copy()
                futures.append(self._thread_pool.submit(node, state_copy))
        
        # 收集结果
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=self.config.node_timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Parallel node execution failed: {e}")
                if self.config.fail_fast:
                    raise
        
        # 合并结果
        return merge_func(results)
    
    async def _aexecute_parallel_group(self, group_name: str, state: T) -> T:
        """异步执行并行节点组"""
        group_info = self.graph._parallel_groups[group_name]
        node_names = group_info["nodes"]
        merge_func = group_info["merge_func"]
        
        tasks = []
        for node_name in node_names:
            node = self.graph.get_node(node_name)
            if node:
                state_copy = state.copy()
                tasks.append(node.ainvoke(state_copy))
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"Parallel node execution failed: {r}")
                if self.config.fail_fast:
                    raise r
            else:
                valid_results.append(r)
        
        return merge_func(valid_results)
    
    def _get_next_node(self, current: str, state: T) -> str:
        """获取下一个节点"""
        # 检查条件边
        if current in self.graph._conditional_edges:
            cond_edge = self.graph._conditional_edges[current]
            target = cond_edge.get_target(state)
            self.logger.debug(f"Conditional routing: {current} -> {target}")
            return target
        
        # 检查普通边
        if current in self._adjacency and self._adjacency[current]:
            return self._adjacency[current][0]
        
        # 默认到结束
        return "__end__"
    
    # ==================== 事件与回调 ====================
    
    def _emit_event(self, event_type: str, node_name: str, data: Dict[str, Any] = None) -> None:
        """触发事件"""
        if not self.config.enable_events:
            return
        
        event = ExecutionEvent(
            event_type=event_type,
            node_name=node_name,
            data=data or {}
        )
        self._event_queue.append(event)
        
        # 调用回调
        if self.config.enable_callbacks:
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.warning(f"Callback error: {e}")
    
    async def _aemit_event(self, event_type: str, node_name: str, data: Dict[str, Any] = None) -> None:
        """异步触发事件"""
        if not self.config.enable_events:
            return
        
        event = ExecutionEvent(
            event_type=event_type,
            node_name=node_name,
            data=data or {}
        )
        self._event_queue.append(event)
        
        if self.config.enable_callbacks:
            for callback in self._async_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.warning(f"Async callback error: {e}")
    
    def get_events(self, clear: bool = False) -> List[ExecutionEvent]:
        """获取事件队列"""
        events = list(self._event_queue)
        if clear:
            self._event_queue.clear()
        return events
    
    # ==================== 执行历史 ====================
    
    def _record_execution(self, context: ExecutionContext, state: T, success: bool) -> None:
        """记录执行历史"""
        record = {
            "run_id": context.run_id,
            "start_time": context.start_time.isoformat() if context.start_time else None,
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "success": success,
            "visited_nodes": context.visited_nodes,
            "execution_path": context.execution_path,
            "final_iteration": state.iteration,
            "error": str(context.error) if context.error else None
        }
        
        with self._lock:
            self._execution_history.append(record)
            if len(self._execution_history) > self._max_history_size:
                self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_execution_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """获取执行历史"""
        with self._lock:
            history = list(self._execution_history)
        
        if limit:
            history = history[-limit:]
        return history
    
    def clear_history(self) -> None:
        """清空执行历史"""
        with self._lock:
            self._execution_history.clear()
    
    # ==================== 状态管理 ====================
    
    def get_state(self, config: Dict[str, Any] = None) -> Optional[T]:
        """获取当前状态（从检查点）"""
        if self.checkpointer:
            thread_id = config.get('thread_id') if config else None
            return self.checkpointer.get_latest(thread_id)
        return None
    
    def update_state(self, state: T, as_node: str = None) -> None:
        """更新状态
        
        Args:
            state: 新状态
            as_node: 模拟从哪个节点更新
        """
        if self.checkpointer:
            if as_node:
                state.metadata["updated_from_node"] = as_node
            self.checkpointer.save(state)
    
    def get_state_history(self, thread_id: str = None, limit: int = 10) -> List[T]:
        """获取状态历史"""
        if self.checkpointer and hasattr(self.checkpointer, 'get_history'):
            return self.checkpointer.get_history(thread_id)[:limit]
        return []
    
    # ==================== 指标 ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        return self._metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self._metrics.reset()
    
    # ==================== 可视化数据 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        viz = self.graph.get_visualization_data()
        viz["metrics"] = self.get_metrics()
        viz["execution_history"] = self.get_execution_history(10)
        return viz
    
    # ==================== 清理 ====================
    
    def close(self) -> None:
        """关闭资源"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class InterruptedError(Exception):
    """执行中断异常"""
    pass


class GraphBuilder:
    """图构建器（增强版）
    
    提供流式 API 来构建状态图。
    
    支持：
    - 链式调用
    - 子图嵌套
    - 并行节点
    - 条件分支
    - 循环
    """
    
    def __init__(self, name: str = "default", state_class: Type[T] = AgentState,
                config: GraphConfig = None):
        config = config or GraphConfig(name=name)
        if name != "default":
            config.name = name
        self.graph = StateGraph(state_class=state_class, config=config)
        self._current_node: str = None
        self._branch_stack: List[str] = []  # 分支栈
    
    def add_node(self, name: str, node: Union[BaseNode, Callable, StateGraph],
                metadata: Dict[str, Any] = None) -> 'GraphBuilder':
        """添加节点"""
        self.graph.add_node(name, node, metadata)
        self._current_node = name
        return self
    
    def add_llm_node(self, name: str, **kwargs) -> 'GraphBuilder':
        """添加 LLM 节点"""
        node = LLMNode(name=name, **kwargs)
        return self.add_node(name, node)
    
    def add_tool_node(self, name: str = "tools", **kwargs) -> 'GraphBuilder':
        """添加工具节点"""
        node = ToolNode(name=name, **kwargs)
        return self.add_node(name, node)
    
    def add_subgraph(self, name: str, subgraph: StateGraph,
                    metadata: Dict[str, Any] = None) -> 'GraphBuilder':
        """添加子图"""
        return self.add_node(name, subgraph, metadata)
    
    def add_edge(self, source: str, target: str, **kwargs) -> 'GraphBuilder':
        """添加边"""
        self.graph.add_edge(source, target, **kwargs)
        return self
    
    def edge_to(self, target: str) -> 'GraphBuilder':
        """从当前节点添加边"""
        if self._current_node:
            self.graph.add_edge(self._current_node, target)
        return self
    
    def then(self, name: str, node: Union[BaseNode, Callable] = None) -> 'GraphBuilder':
        """添加下一个节点并连接边
        
        简化写法：builder.then("node_a", func_a).then("node_b", func_b)
        """
        if node is not None:
            self.add_node(name, node)
        if self._current_node:
            self.add_edge(self._current_node, name)
        self._current_node = name
        return self
    
    def branch(self, condition: Callable[[T], str], branches: Dict[str, str]) -> 'GraphBuilder':
        """从当前节点添加分支
        
        Args:
            condition: 条件函数
            branches: 分支映射 {条件结果: 目标节点}
        """
        if self._current_node:
            self.graph.add_conditional_edges(self._current_node, condition, branches)
            self._branch_stack.append(self._current_node)
        return self
    
    def add_conditional_edges(self, 
                             source: str,
                             condition: Callable,
                             branches: Dict[str, str] = None) -> 'GraphBuilder':
        """添加条件边"""
        self.graph.add_conditional_edges(source, condition, branches)
        return self
    
    def conditional_to(self,
                      condition: Callable,
                      branches: Dict[str, str] = None) -> 'GraphBuilder':
        """从当前节点添加条件边"""
        if self._current_node:
            self.graph.add_conditional_edges(self._current_node, condition, branches)
        return self
    
    def loop_while(self, condition: Callable[[T], bool], 
                  body_nodes: List[Tuple[str, Union[BaseNode, Callable]]],
                  max_iterations: int = 10) -> 'GraphBuilder':
        """添加循环结构
        
        Args:
            condition: 循环条件（返回 True 继续循环）
            body_nodes: 循环体节点
            max_iterations: 最大迭代次数
        """
        if not body_nodes:
            return self
        
        loop_start = f"__loop_start_{len(self._branch_stack)}"
        loop_end = f"__loop_end_{len(self._branch_stack)}"
        
        # 添加循环体节点
        prev_node = loop_start
        for name, node in body_nodes:
            self.add_node(name, node)
            if prev_node != loop_start:
                self.add_edge(prev_node, name)
            prev_node = name
        
        # 条件判断函数
        iteration_counter = {"count": 0}
        
        def loop_condition(state: T) -> str:
            iteration_counter["count"] += 1
            if iteration_counter["count"] >= max_iterations:
                return loop_end
            return body_nodes[0][0] if condition(state) else loop_end
        
        # 连接当前节点到循环开始
        if self._current_node:
            self.add_edge(self._current_node, body_nodes[0][0])
        
        # 循环体最后一个节点的条件边
        self.graph.add_conditional_edges(
            body_nodes[-1][0],
            loop_condition,
            {body_nodes[0][0]: body_nodes[0][0], loop_end: loop_end}
        )
        
        self._current_node = body_nodes[-1][0]
        return self
    
    def parallel(self, group_name: str, node_specs: List[Tuple[str, Union[BaseNode, Callable]]],
                merge_func: Callable[[List[AgentState]], AgentState] = None) -> 'GraphBuilder':
        """添加并行节点组
        
        Args:
            group_name: 组名称
            node_specs: [(节点名, 节点)] 列表
            merge_func: 结果合并函数
        """
        node_names = []
        for name, node in node_specs:
            self.add_node(name, node)
            node_names.append(name)
        
        self.graph.add_parallel_nodes(group_name, node_names, merge_func)
        
        # 从当前节点连接到并行组
        if self._current_node:
            for name in node_names:
                self.add_edge(self._current_node, name)
        
        return self
    
    def set_entry_point(self, name: str) -> 'GraphBuilder':
        """设置入口点"""
        self.graph.set_entry_point(name)
        return self
    
    def set_finish_point(self, name: str) -> 'GraphBuilder':
        """设置结束点"""
        self.graph.set_finish_point(name)
        return self
    
    def with_interrupt_before(self, *node_names: str) -> 'GraphBuilder':
        """在节点前添加中断点"""
        for name in node_names:
            self.graph.add_interrupt_before(name)
        return self
    
    def with_interrupt_after(self, *node_names: str) -> 'GraphBuilder':
        """在节点后添加中断点"""
        for name in node_names:
            self.graph.add_interrupt_after(name)
        return self
    
    def with_callback(self, callback: GraphCallback) -> 'GraphBuilder':
        """添加回调"""
        self.graph.add_callback(callback)
        return self
    
    def with_async_callback(self, callback: AsyncGraphCallback) -> 'GraphBuilder':
        """添加异步回调"""
        self.graph.add_async_callback(callback)
        return self
    
    def compile(self, checkpointer=None, **kwargs) -> CompiledGraph:
        """编译图"""
        return self.graph.compile(checkpointer, **kwargs)
    
    def build(self) -> StateGraph:
        """返回构建的图（不编译）"""
        return self.graph
    
    def validate(self) -> List[str]:
        """验证图"""
        return self.graph.validate()
    
    def to_mermaid(self) -> str:
        """导出 Mermaid"""
        return self.graph.to_mermaid()
    
    def to_dot(self) -> str:
        """导出 DOT"""
        return self.graph.to_dot()


class Subgraph:
    """子图（增强版）
    
    可组合的图单元，可以嵌入到其他图中。
    
    支持：
    - 状态映射
    - 输入/输出转换
    - 条件执行
    """
    
    def __init__(self, 
                 name: str,
                 graph: Union[StateGraph, CompiledGraph],
                 input_mapper: Callable[[AgentState], AgentState] = None,
                 output_mapper: Callable[[AgentState], AgentState] = None,
                 condition: Callable[[AgentState], bool] = None):
        """
        Args:
            name: 子图名称
            graph: 状态图或编译图
            input_mapper: 输入状态映射函数
            output_mapper: 输出状态映射函数
            condition: 执行条件函数
        """
        self.name = name
        self._graph = graph
        self._compiled = None
        self.input_mapper = input_mapper
        self.output_mapper = output_mapper
        self.condition = condition
    
    @property
    def graph(self) -> CompiledGraph:
        if self._compiled is None:
            if isinstance(self._graph, CompiledGraph):
                self._compiled = self._graph
            else:
                self._compiled = self._graph.compile()
        return self._compiled
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行子图"""
        # 检查条件
        if self.condition and not self.condition(state):
            return state
        
        # 输入映射
        input_state = self.input_mapper(state) if self.input_mapper else state
        
        # 执行子图
        output_state = self.graph.invoke(input_state)
        
        # 输出映射
        if self.output_mapper:
            output_state = self.output_mapper(output_state)
        
        return output_state
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        """异步执行子图"""
        if self.condition and not self.condition(state):
            return state
        
        input_state = self.input_mapper(state) if self.input_mapper else state
        output_state = await self.graph.ainvoke(input_state)
        
        if self.output_mapper:
            output_state = self.output_mapper(output_state)
        
        return output_state


class GraphRunner:
    """图运行器
    
    管理多个图的执行，支持：
    - 批量执行
    - 并行执行
    - 调度策略
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._graphs: Dict[str, CompiledGraph] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(f"{__name__}.GraphRunner")
    
    def register(self, name: str, graph: Union[StateGraph, CompiledGraph]) -> None:
        """注册图"""
        if isinstance(graph, StateGraph):
            graph = graph.compile()
        self._graphs[name] = graph
    
    def unregister(self, name: str) -> bool:
        """注销图"""
        if name in self._graphs:
            del self._graphs[name]
            return True
        return False
    
    def run(self, name: str, input_data: Union[str, Dict[str, Any], T],
           config: Dict[str, Any] = None) -> T:
        """运行指定图"""
        if name not in self._graphs:
            raise ValueError(f"Graph '{name}' not found")
        return self._graphs[name].invoke(input_data, config)
    
    async def arun(self, name: str, input_data: Union[str, Dict[str, Any], T],
                  config: Dict[str, Any] = None) -> T:
        """异步运行指定图"""
        if name not in self._graphs:
            raise ValueError(f"Graph '{name}' not found")
        return await self._graphs[name].ainvoke(input_data, config)
    
    def run_batch(self, name: str, inputs: List[Union[str, Dict[str, Any], T]],
                 config: Dict[str, Any] = None) -> List[T]:
        """批量运行"""
        if name not in self._graphs:
            raise ValueError(f"Graph '{name}' not found")
        
        graph = self._graphs[name]
        results = []
        
        for input_data in inputs:
            try:
                result = graph.invoke(input_data, config)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch run failed for input: {e}")
                results.append(None)
        
        return results
    
    def run_parallel(self, tasks: List[Tuple[str, Union[str, Dict[str, Any], T], Dict[str, Any]]]) -> List[T]:
        """并行运行多个图
        
        Args:
            tasks: [(图名称, 输入, 配置)] 列表
        
        Returns:
            结果列表
        """
        futures = []
        for name, input_data, config in tasks:
            if name in self._graphs:
                future = self._thread_pool.submit(
                    self._graphs[name].invoke, input_data, config
                )
                futures.append(future)
            else:
                futures.append(None)
        
        results = []
        for future in futures:
            if future:
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel run failed: {e}")
                    results.append(None)
            else:
                results.append(None)
        
        return results
    
    async def arun_parallel(self, tasks: List[Tuple[str, Union[str, Dict[str, Any], T], Dict[str, Any]]]) -> List[T]:
        """异步并行运行"""
        async_tasks = []
        for name, input_data, config in tasks:
            if name in self._graphs:
                async_tasks.append(self._graphs[name].ainvoke(input_data, config))
            else:
                async_tasks.append(asyncio.sleep(0))  # 占位
        
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        processed_results = []
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"Async parallel run failed: {r}")
                processed_results.append(None)
            else:
                processed_results.append(r)
        
        return processed_results
    
    def get_graph(self, name: str) -> Optional[CompiledGraph]:
        """获取图"""
        return self._graphs.get(name)
    
    def list_graphs(self) -> List[str]:
        """列出所有注册的图"""
        return list(self._graphs.keys())
    
    def close(self) -> None:
        """关闭运行器"""
        self._thread_pool.shutdown(wait=False)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class GraphAnalyzer:
    """图分析器
    
    分析图结构和执行性能。
    """
    
    def __init__(self, graph: Union[StateGraph, CompiledGraph]):
        self.graph = graph if isinstance(graph, StateGraph) else graph.graph
    
    def get_node_count(self) -> int:
        """获取节点数量"""
        return len(self.graph._nodes)
    
    def get_edge_count(self) -> int:
        """获取边数量"""
        return len(self.graph._edges) + len(self.graph._conditional_edges)
    
    def get_depth(self) -> int:
        """获取图深度（从入口到最远结束点的最长路径）"""
        if not self.graph._entry_point:
            return 0
        
        visited = set()
        max_depth = 0
        
        def dfs(node: str, depth: int):
            nonlocal max_depth
            if node in visited or node == "__end__":
                max_depth = max(max_depth, depth)
                return
            
            visited.add(node)
            
            # 获取后继节点
            successors = []
            for edge in self.graph._edges:
                if edge.source == node:
                    successors.append(edge.target)
            if node in self.graph._conditional_edges:
                successors.extend(self.graph._conditional_edges[node].branches.values())
            
            for successor in successors:
                dfs(successor, depth + 1)
            
            visited.remove(node)
        
        dfs(self.graph._entry_point, 0)
        return max_depth
    
    def get_all_paths(self, max_depth: int = 20) -> List[List[str]]:
        """获取所有可能路径"""
        if not self.graph._entry_point:
            return []
        
        paths = []
        
        def dfs(node: str, current_path: List[str], depth: int):
            if depth > max_depth:
                return
            
            current_path = current_path + [node]
            
            if node == "__end__" or node in self.graph._finish_points:
                paths.append(current_path)
                return
            
            # 获取后继节点
            successors = []
            for edge in self.graph._edges:
                if edge.source == node:
                    successors.append(edge.target)
            if node in self.graph._conditional_edges:
                successors.extend(self.graph._conditional_edges[node].branches.values())
            
            if not successors:
                paths.append(current_path)
                return
            
            for successor in successors:
                dfs(successor, current_path, depth + 1)
        
        dfs(self.graph._entry_point, [], 0)
        return paths
    
    def find_bottlenecks(self) -> List[str]:
        """找出瓶颈节点（所有路径必经的节点）"""
        all_paths = self.get_all_paths()
        if not all_paths:
            return []
        
        # 统计每个节点在多少路径中出现
        from collections import Counter
        node_counts = Counter()
        for path in all_paths:
            for node in set(path):  # 每条路径中只计一次
                node_counts[node] += 1
        
        total_paths = len(all_paths)
        bottlenecks = [node for node, count in node_counts.items() 
                      if count == total_paths and node not in ["__start__", "__end__"]]
        
        return bottlenecks
    
    def detect_unreachable_nodes(self) -> List[str]:
        """检测不可达节点"""
        if not self.graph._entry_point:
            return list(self.graph._nodes.keys())
        
        reachable = set()
        
        def dfs(node: str):
            if node in reachable:
                return
            reachable.add(node)
            
            for edge in self.graph._edges:
                if edge.source == node:
                    dfs(edge.target)
            if node in self.graph._conditional_edges:
                for target in self.graph._conditional_edges[node].branches.values():
                    dfs(target)
        
        dfs("__start__")
        
        unreachable = [node for node in self.graph._nodes if node not in reachable]
        return unreachable
    
    def get_cyclic_nodes(self) -> Set[str]:
        """获取参与循环的节点"""
        cycles = self.graph._detect_cycles()
        cyclic_nodes = set()
        for cycle in cycles:
            cyclic_nodes.update(cycle)
        return cyclic_nodes
    
    def get_summary(self) -> Dict[str, Any]:
        """获取图分析摘要"""
        return {
            "name": self.graph.config.name,
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "depth": self.get_depth(),
            "entry_point": self.graph._entry_point,
            "finish_points": list(self.graph._finish_points),
            "has_cycles": len(self.graph._detect_cycles()) > 0,
            "parallel_groups": list(self.graph._parallel_groups.keys()),
            "subgraphs": list(self.graph._subgraphs.keys()),
            "bottlenecks": self.find_bottlenecks(),
            "unreachable_nodes": self.detect_unreachable_nodes(),
            "cyclic_nodes": list(self.get_cyclic_nodes()),
            "interrupt_before": list(self.graph._interrupt_before),
            "interrupt_after": list(self.graph._interrupt_after)
        }


# ==================== 便捷函数 ====================

def create_simple_graph(nodes: List[Tuple[str, Union[BaseNode, Callable]]],
                       entry: str = None,
                       config: GraphConfig = None) -> CompiledGraph:
    """创建简单的线性图
    
    Args:
        nodes: (名称, 节点) 元组列表
        entry: 入口点（默认为第一个节点）
        config: 图配置
    """
    builder = GraphBuilder(config=config)
    
    prev_name = None
    for name, node in nodes:
        builder.add_node(name, node)
        if prev_name:
            builder.add_edge(prev_name, name)
        prev_name = name
    
    entry_point = entry or nodes[0][0]
    builder.set_entry_point(entry_point)
    builder.set_finish_point(nodes[-1][0])
    
    return builder.compile()


def create_branching_graph(
    entry_node: Tuple[str, Union[BaseNode, Callable]],
    condition: Callable[[T], str],
    branches: Dict[str, List[Tuple[str, Union[BaseNode, Callable]]]],
    merge_node: Tuple[str, Union[BaseNode, Callable]] = None,
    config: GraphConfig = None
) -> CompiledGraph:
    """创建分支图
    
    Args:
        entry_node: 入口节点 (名称, 节点)
        condition: 条件函数
        branches: 分支字典 {条件值: [(名称, 节点)...]}
        merge_node: 合并节点 (名称, 节点)
        config: 图配置
    """
    builder = GraphBuilder(config=config)
    
    # 添加入口节点
    entry_name, entry = entry_node
    builder.add_node(entry_name, entry)
    builder.set_entry_point(entry_name)
    
    # 添加分支
    branch_mapping = {}
    branch_end_nodes = []
    
    for branch_key, branch_nodes in branches.items():
        if not branch_nodes:
            continue
        
        # 添加分支节点
        prev_name = None
        for name, node in branch_nodes:
            builder.add_node(name, node)
            if prev_name:
                builder.add_edge(prev_name, name)
            prev_name = name
        
        # 记录分支映射和结束节点
        branch_mapping[branch_key] = branch_nodes[0][0]
        branch_end_nodes.append(branch_nodes[-1][0])
    
    # 添加条件边
    builder.add_conditional_edges(entry_name, condition, branch_mapping)
    
    # 添加合并节点
    if merge_node:
        merge_name, merge = merge_node
        builder.add_node(merge_name, merge)
        for end_node in branch_end_nodes:
            builder.add_edge(end_node, merge_name)
        builder.set_finish_point(merge_name)
    else:
        for end_node in branch_end_nodes:
            builder.set_finish_point(end_node)
    
    return builder.compile()


def create_react_graph(
    llm_node: Union[BaseNode, Callable],
    tool_node: Union[BaseNode, Callable],
    should_continue: Callable[[T], str] = None,
    config: GraphConfig = None
) -> CompiledGraph:
    """创建 ReAct 风格的图
    
    交替执行 LLM 和工具，直到完成。
    
    Args:
        llm_node: LLM 节点
        tool_node: 工具节点
        should_continue: 继续条件函数（返回 "tools" 或 "end"）
        config: 图配置
    """
    if should_continue is None:
        def should_continue(state: T) -> str:
            # 默认逻辑：如果最后一条消息有工具调用，则继续
            if state.messages and hasattr(state.messages[-1], 'tool_calls'):
                if state.messages[-1].tool_calls:
                    return "tools"
            return "end"
    
    builder = GraphBuilder(name="react", config=config)
    
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    
    builder.set_entry_point("llm")
    
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", "end": "__end__"}
    )
    
    builder.add_edge("tools", "llm")
    
    return builder.compile()


def merge_graphs(*graphs: StateGraph, name: str = "merged") -> StateGraph:
    """合并多个图
    
    Args:
        graphs: 要合并的图
        name: 新图名称
    
    Returns:
        合并后的新图
    """
    if not graphs:
        return StateGraph(config=GraphConfig(name=name))
    
    merged = graphs[0]
    for i, graph in enumerate(graphs[1:], 1):
        merged = merged.merge(graph, prefix=f"g{i}_")
    
    merged.config.name = name
    return merged


def chain_graphs(*graphs: StateGraph, name: str = "chained") -> StateGraph:
    """串联多个图
    
    Args:
        graphs: 要串联的图
        name: 新图名称
    
    Returns:
        串联后的新图
    """
    if not graphs:
        return StateGraph(config=GraphConfig(name=name))
    
    chained = graphs[0]
    for graph in graphs[1:]:
        chained = chained.chain(graph)
    
    chained.config.name = name
    return chained

