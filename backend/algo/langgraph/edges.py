"""
图边定义（生产级）

定义 LangGraph 中使用的各种边类型：
- Edge: 普通边
- ConditionalEdge: 条件边
- LoopEdge: 循环边
- ParallelEdge: 并行边
- TimeoutEdge: 超时边
- RetryEdge: 重试边
- FallbackEdge: 降级边
- WeightedEdge: 带权重边
- DelayedEdge: 延迟边
- EventEdge: 事件驱动边
- TransformEdge: 转换边

生产级特性：
- 复合条件（AND/OR/NOT）
- 多种路由策略（优先级、权重、负载均衡、A/B测试）
- 执行前后钩子
- 遍历统计和性能指标
- 边验证和类型检查
- 可视化支持
"""

import logging
import re
import time
import uuid
import random
import asyncio
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from functools import wraps
import copy

from .state import AgentState, AgentStatus, MessageType

logger = logging.getLogger(__name__)


# ==================== 类型定义 ====================

class EdgeType(Enum):
    """边类型"""
    NORMAL = "normal"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    TIMEOUT = "timeout"
    RETRY = "retry"
    FALLBACK = "fallback"
    WEIGHTED = "weighted"
    DELAYED = "delayed"
    EVENT = "event"
    TRANSFORM = "transform"


class RoutingStrategy(Enum):
    """路由策略"""
    FIRST_MATCH = "first_match"       # 第一个匹配
    PRIORITY = "priority"              # 优先级路由
    WEIGHTED_RANDOM = "weighted_random"  # 权重随机
    ROUND_ROBIN = "round_robin"        # 轮询
    LOAD_BALANCE = "load_balance"      # 负载均衡
    AB_TEST = "ab_test"                # A/B 测试
    RANDOM = "random"                  # 随机选择


class ConditionOperator(Enum):
    """条件运算符"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


class ComparisonOperator(Enum):
    """比较运算符"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # 正则匹配
    IN = "in"
    NOT_IN = "not_in"
    IS_NONE = "is_none"
    IS_NOT_NONE = "is_not_none"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


@dataclass
class EdgeMetrics:
    """边性能指标"""
    traversal_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0.0
    last_traversal_time: Optional[datetime] = None
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    
    def record_traversal(self, elapsed_ms: float, success: bool = True) -> None:
        """记录遍历"""
        self.traversal_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_time_ms += elapsed_ms
        self.last_traversal_time = datetime.utcnow()
        self.min_time_ms = min(self.min_time_ms, elapsed_ms)
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.traversal_count if self.traversal_count > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.traversal_count if self.traversal_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "traversal_count": self.traversal_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float('inf') else 0,
            "max_time_ms": round(self.max_time_ms, 2),
            "last_traversal_time": self.last_traversal_time.isoformat() if self.last_traversal_time else None
        }
    
    def reset(self) -> None:
        """重置指标"""
        self.traversal_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time_ms = 0.0
        self.last_traversal_time = None
        self.min_time_ms = float('inf')
        self.max_time_ms = 0.0


@dataclass
class EdgeHooks:
    """边钩子"""
    before_traverse: List[Callable[[AgentState], AgentState]] = field(default_factory=list)
    after_traverse: List[Callable[[AgentState], AgentState]] = field(default_factory=list)
    on_error: List[Callable[[AgentState, Exception], AgentState]] = field(default_factory=list)
    
    def add_before(self, hook: Callable[[AgentState], AgentState]) -> None:
        self.before_traverse.append(hook)
    
    def add_after(self, hook: Callable[[AgentState], AgentState]) -> None:
        self.after_traverse.append(hook)
    
    def add_on_error(self, hook: Callable[[AgentState, Exception], AgentState]) -> None:
        self.on_error.append(hook)
    
    def run_before(self, state: AgentState) -> AgentState:
        for hook in self.before_traverse:
            try:
                state = hook(state) or state
            except Exception as e:
                logger.warning(f"Before hook error: {e}")
        return state
    
    def run_after(self, state: AgentState) -> AgentState:
        for hook in self.after_traverse:
            try:
                state = hook(state) or state
            except Exception as e:
                logger.warning(f"After hook error: {e}")
        return state
    
    def run_on_error(self, state: AgentState, error: Exception) -> AgentState:
        for hook in self.on_error:
            try:
                state = hook(state, error) or state
            except Exception as e:
                logger.warning(f"Error hook error: {e}")
        return state


# ==================== 条件系统 ====================

@dataclass
class EdgeCondition:
    """边条件
    
    定义边的触发条件。支持简单条件和复合条件。
    """
    name: str
    condition_func: Callable[[AgentState], bool]
    priority: int = 0  # 优先级，数字越大优先级越高
    description: str = ""
    enabled: bool = True
    cache_result: bool = False  # 是否缓存结果
    _cached_result: Optional[bool] = field(default=None, repr=False)
    _cache_state_id: Optional[str] = field(default=None, repr=False)
    
    def evaluate(self, state: AgentState) -> bool:
        """评估条件"""
        if not self.enabled:
            return False
        
        try:
            # 检查缓存
            if self.cache_result and self._cache_state_id == state.thread_id:
                return self._cached_result
            
            result = self.condition_func(state)
            
            # 更新缓存
            if self.cache_result:
                self._cached_result = result
                self._cache_state_id = state.thread_id
            
            return result
        except Exception as e:
            logger.error(f"Edge condition '{self.name}' evaluation error: {e}")
            return False
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cached_result = None
        self._cache_state_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "description": self.description,
            "enabled": self.enabled
        }


@dataclass
class StateCondition(EdgeCondition):
    """状态属性条件
    
    基于状态属性的条件判断。
    """
    field_path: str = ""
    operator: ComparisonOperator = ComparisonOperator.EQUALS
    expected_value: Any = None
    
    def __post_init__(self):
        if not self.condition_func:
            self.condition_func = self._evaluate_field
    
    def _get_field_value(self, state: AgentState) -> Any:
        """获取状态字段值（支持嵌套路径）"""
        obj = state
        for part in self.field_path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj
    
    def _evaluate_field(self, state: AgentState) -> bool:
        """评估字段条件"""
        value = self._get_field_value(state)
        op = self.operator
        expected = self.expected_value
        
        if op == ComparisonOperator.EQUALS:
            return value == expected
        elif op == ComparisonOperator.NOT_EQUALS:
            return value != expected
        elif op == ComparisonOperator.GREATER_THAN:
            return value > expected
        elif op == ComparisonOperator.LESS_THAN:
            return value < expected
        elif op == ComparisonOperator.GREATER_EQUAL:
            return value >= expected
        elif op == ComparisonOperator.LESS_EQUAL:
            return value <= expected
        elif op == ComparisonOperator.CONTAINS:
            return expected in value if value else False
        elif op == ComparisonOperator.NOT_CONTAINS:
            return expected not in value if value else True
        elif op == ComparisonOperator.MATCHES:
            return bool(re.match(expected, str(value))) if value else False
        elif op == ComparisonOperator.IN:
            return value in expected if expected else False
        elif op == ComparisonOperator.NOT_IN:
            return value not in expected if expected else True
        elif op == ComparisonOperator.IS_NONE:
            return value is None
        elif op == ComparisonOperator.IS_NOT_NONE:
            return value is not None
        elif op == ComparisonOperator.IS_EMPTY:
            return not value if value is not None else True
        elif op == ComparisonOperator.IS_NOT_EMPTY:
            return bool(value) if value is not None else False
        
        return False


@dataclass
class CompositeCondition(EdgeCondition):
    """复合条件
    
    支持 AND/OR/NOT 组合多个条件。
    """
    conditions: List[EdgeCondition] = field(default_factory=list)
    operator: ConditionOperator = ConditionOperator.AND
    
    def __post_init__(self):
        self.condition_func = self._evaluate_composite
    
    def _evaluate_composite(self, state: AgentState) -> bool:
        """评估复合条件"""
        if not self.conditions:
            return True
        
        results = [c.evaluate(state) for c in self.conditions if c.enabled]
        
        if self.operator == ConditionOperator.AND:
            return all(results)
        elif self.operator == ConditionOperator.OR:
            return any(results)
        elif self.operator == ConditionOperator.NOT:
            return not results[0] if results else True
        elif self.operator == ConditionOperator.XOR:
            return sum(results) == 1
        
        return False
    
    def add_condition(self, condition: EdgeCondition) -> 'CompositeCondition':
        """添加条件"""
        self.conditions.append(condition)
        return self


@dataclass
class ThresholdCondition(EdgeCondition):
    """阈值条件
    
    基于数值阈值的条件。
    """
    field_path: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def __post_init__(self):
        self.condition_func = self._evaluate_threshold
    
    def _get_field_value(self, state: AgentState) -> Any:
        obj = state
        for part in self.field_path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj
    
    def _evaluate_threshold(self, state: AgentState) -> bool:
        """评估阈值"""
        value = self._get_field_value(state)
        if value is None:
            return False
        
        try:
            num_value = float(value)
            if self.min_value is not None and num_value < self.min_value:
                return False
            if self.max_value is not None and num_value > self.max_value:
                return False
            return True
        except (TypeError, ValueError):
            return False


@dataclass
class TimeCondition(EdgeCondition):
    """时间条件
    
    基于时间的条件判断。
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_age_seconds: Optional[float] = None  # 状态最大年龄
    time_of_day_start: Optional[str] = None  # "HH:MM" 格式
    time_of_day_end: Optional[str] = None
    weekdays: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    
    def __post_init__(self):
        self.condition_func = self._evaluate_time
    
    def _evaluate_time(self, state: AgentState) -> bool:
        """评估时间条件"""
        now = datetime.utcnow()
        
        # 检查时间范围
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        
        # 检查状态年龄
        if self.max_age_seconds:
            age = (now - state.updated_at).total_seconds()
            if age > self.max_age_seconds:
                return False
        
        # 检查时间段
        if self.time_of_day_start and self.time_of_day_end:
            current_time = now.strftime("%H:%M")
            if not (self.time_of_day_start <= current_time <= self.time_of_day_end):
                return False
        
        # 检查星期
        if self.weekdays is not None:
            if now.weekday() not in self.weekdays:
                return False
        
        return True


@dataclass
class MessageCondition(EdgeCondition):
    """消息条件
    
    基于消息内容的条件。
    """
    message_type: Optional[MessageType] = None
    content_pattern: Optional[str] = None  # 正则表达式
    has_tool_calls: Optional[bool] = None
    message_count_min: Optional[int] = None
    message_count_max: Optional[int] = None
    
    def __post_init__(self):
        self.condition_func = self._evaluate_message
    
    def _evaluate_message(self, state: AgentState) -> bool:
        """评估消息条件"""
        # 检查消息数量
        msg_count = len(state.messages)
        if self.message_count_min is not None and msg_count < self.message_count_min:
            return False
        if self.message_count_max is not None and msg_count > self.message_count_max:
            return False
        
        # 检查最后一条消息
        last_msg = state.get_last_message()
        if not last_msg:
            return self.message_type is None and self.content_pattern is None
        
        # 检查消息类型
        if self.message_type and last_msg.type != self.message_type:
            return False
        
        # 检查内容模式
        if self.content_pattern:
            if not re.search(self.content_pattern, last_msg.content or ""):
                return False
        
        # 检查工具调用
        if self.has_tool_calls is not None:
            has_calls = bool(last_msg.tool_calls)
            if has_calls != self.has_tool_calls:
                return False
        
        return True


# ==================== 边基类 ====================

@dataclass
class Edge:
    """普通边（增强版）
    
    连接两个节点的基本边，支持：
    - 条件判断
    - 执行钩子
    - 性能指标
    - 元数据
    """
    source: str
    target: str
    edge_type: EdgeType = EdgeType.NORMAL
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    enabled: bool = True
    priority: int = 0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 钩子
    hooks: EdgeHooks = field(default_factory=EdgeHooks)
    
    # 指标
    metrics: EdgeMetrics = field(default_factory=EdgeMetrics)
    
    # 条件
    condition: Optional[EdgeCondition] = None
    
    # 转换函数
    transform_func: Optional[Callable[[AgentState], AgentState]] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.source}->{self.target}"
    
    def can_traverse(self, state: AgentState) -> bool:
        """判断是否可以通过此边"""
        if not self.enabled:
            return False
        if self.condition:
            return self.condition.evaluate(state)
        return True
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """遍历边
        
        Returns:
            (处理后的状态, 目标节点名)
        """
        start_time = time.time()
        success = True
        
        try:
            # 执行前钩子
            state = self.hooks.run_before(state)
            
            # 执行转换
            if self.transform_func:
                state = self.transform_func(state) or state
            
            # 执行后钩子
            state = self.hooks.run_after(state)
            
            return state, self.get_target(state)
            
        except Exception as e:
            success = False
            logger.error(f"Edge '{self.name}' traverse error: {e}")
            state = self.hooks.run_on_error(state, e)
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, success)
    
    def get_target(self, state: AgentState) -> str:
        """获取目标节点"""
        return self.target
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self.metrics.reset()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "edge_id": self.edge_id,
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "weight": self.weight,
            "description": self.description,
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict(),
            "condition": self.condition.to_dict() if self.condition else None
        }
    
    def copy(self) -> 'Edge':
        """复制边"""
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        status = "✓" if self.enabled else "✗"
        return f"Edge[{status}]({self.source} -> {self.target})"


@dataclass
class ConditionalEdge(Edge):
    """条件边
    
    根据条件决定是否通过的边。
    """
    condition: EdgeCondition = None
    condition_func: Callable[[AgentState], str] = None  # 返回目标节点名
    branches: Dict[str, str] = field(default_factory=dict)  # 条件结果 -> 目标节点
    
    def __post_init__(self):
        self.edge_type = EdgeType.CONDITIONAL
    
    def can_traverse(self, state: AgentState) -> bool:
        """判断是否可以通过此边"""
        if self.condition:
            return self.condition.evaluate(state)
        return True
    
    def get_target(self, state: AgentState) -> str:
        """获取实际目标节点"""
        if self.condition_func:
            result = self.condition_func(state)
            return self.branches.get(result, result)
        return self.target
    
    def __repr__(self) -> str:
        return f"ConditionalEdge({self.source} -> [conditional])"


@dataclass
class LoopEdge(Edge):
    """循环边
    
    支持循环执行的边。
    """
    max_iterations: int = 10
    continue_condition: Callable[[AgentState], bool] = None
    
    def __post_init__(self):
        self.edge_type = EdgeType.LOOP
    
    def can_traverse(self, state: AgentState) -> bool:
        """判断是否可以继续循环"""
        # 检查迭代次数
        if state.iteration >= self.max_iterations:
            logger.debug(f"Loop edge: max iterations ({self.max_iterations}) reached")
            return False
        
        # 检查继续条件
        if self.continue_condition:
            return self.continue_condition(state)
        
        return True
    
    def __repr__(self) -> str:
        return f"LoopEdge({self.source} -> {self.target}, max={self.max_iterations})"


@dataclass
class ParallelEdge(Edge):
    """并行边
    
    支持并行执行多个目标节点。
    """
    targets: List[str] = field(default_factory=list)
    merge_strategy: str = "all"  # all, first, majority
    timeout: float = 30.0  # 并行执行超时
    
    def __post_init__(self):
        self.edge_type = EdgeType.PARALLEL
    
    def get_targets(self) -> List[str]:
        """获取所有目标节点"""
        return self.targets
    
    def __repr__(self) -> str:
        return f"ParallelEdge({self.source} -> [{', '.join(self.targets)}])"


@dataclass
class TimeoutEdge(Edge):
    """超时边
    
    带超时控制的边，如果执行超时则转向超时处理节点。
    """
    timeout_seconds: float = 30.0
    timeout_target: str = ""  # 超时后跳转的目标节点
    on_timeout: Optional[Callable[[AgentState], AgentState]] = None
    
    def __post_init__(self):
        self.edge_type = EdgeType.TIMEOUT
        if not self.timeout_target:
            self.timeout_target = "__timeout__"
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """带超时的遍历"""
        start_time = time.time()
        success = True
        timed_out = False
        
        try:
            # 执行前钩子
            state = self.hooks.run_before(state)
            
            # 使用线程执行转换（以支持超时）
            if self.transform_func:
                result_container = {"state": state, "error": None}
                
                def run_transform():
                    try:
                        result_container["state"] = self.transform_func(state) or state
                    except Exception as e:
                        result_container["error"] = e
                
                thread = threading.Thread(target=run_transform)
                thread.start()
                thread.join(timeout=self.timeout_seconds)
                
                if thread.is_alive():
                    # 超时
                    timed_out = True
                    logger.warning(f"Edge '{self.name}' timed out after {self.timeout_seconds}s")
                    
                    if self.on_timeout:
                        state = self.on_timeout(state)
                    else:
                        state.metadata["timeout_edge"] = self.name
                        state.metadata["timeout_at"] = datetime.utcnow().isoformat()
                    
                    return state, self.timeout_target
                
                if result_container["error"]:
                    raise result_container["error"]
                
                state = result_container["state"]
            
            # 执行后钩子
            state = self.hooks.run_after(state)
            
            return state, self.get_target(state)
            
        except Exception as e:
            success = False
            logger.error(f"TimeoutEdge '{self.name}' error: {e}")
            state = self.hooks.run_on_error(state, e)
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, success and not timed_out)
    
    def __repr__(self) -> str:
        return f"TimeoutEdge({self.source} -> {self.target}, timeout={self.timeout_seconds}s)"


@dataclass
class RetryEdge(Edge):
    """重试边
    
    支持自动重试的边，可配置重试次数、退避策略等。
    """
    max_retries: int = 3
    retry_delay: float = 1.0  # 初始重试延迟（秒）
    backoff_multiplier: float = 2.0  # 退避乘数
    max_delay: float = 60.0  # 最大延迟
    retry_on_exceptions: Tuple[type, ...] = (Exception,)  # 可重试的异常类型
    retry_condition: Optional[Callable[[AgentState, Exception], bool]] = None
    on_retry: Optional[Callable[[AgentState, int, Exception], AgentState]] = None
    fallback_target: str = ""  # 重试失败后的降级目标
    
    # 运行时状态
    _current_retry: int = field(default=0, repr=False)
    _last_error: Optional[Exception] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.edge_type = EdgeType.RETRY
    
    def _should_retry(self, state: AgentState, error: Exception) -> bool:
        """判断是否应该重试"""
        # 检查重试次数
        if self._current_retry >= self.max_retries:
            return False
        
        # 检查异常类型
        if not isinstance(error, self.retry_on_exceptions):
            return False
        
        # 检查自定义条件
        if self.retry_condition:
            return self.retry_condition(state, error)
        
        return True
    
    def _calculate_delay(self) -> float:
        """计算重试延迟"""
        delay = self.retry_delay * (self.backoff_multiplier ** self._current_retry)
        return min(delay, self.max_delay)
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """带重试的遍历"""
        self._current_retry = 0
        self._last_error = None
        
        while True:
            start_time = time.time()
            success = True
            
            try:
                # 执行前钩子
                state = self.hooks.run_before(state)
                
                # 执行转换
                if self.transform_func:
                    state = self.transform_func(state) or state
                
                # 执行后钩子
                state = self.hooks.run_after(state)
                
                # 成功，重置重试计数
                self._current_retry = 0
                self._last_error = None
                
                elapsed_ms = (time.time() - start_time) * 1000
                self.metrics.record_traversal(elapsed_ms, True)
                
                return state, self.get_target(state)
                
            except Exception as e:
                success = False
                self._last_error = e
                
                elapsed_ms = (time.time() - start_time) * 1000
                self.metrics.record_traversal(elapsed_ms, False)
                
                if self._should_retry(state, e):
                    self._current_retry += 1
                    delay = self._calculate_delay()
                    
                    logger.warning(f"RetryEdge '{self.name}' retry {self._current_retry}/{self.max_retries} "
                                 f"after {delay:.1f}s: {e}")
                    
                    # 执行重试回调
                    if self.on_retry:
                        state = self.on_retry(state, self._current_retry, e)
                    
                    # 记录重试信息
                    state.metadata["retry_count"] = self._current_retry
                    state.metadata["retry_delay"] = delay
                    state.metadata["last_error"] = str(e)
                    
                    # 等待后重试
                    time.sleep(delay)
                    continue
                else:
                    # 不再重试
                    logger.error(f"RetryEdge '{self.name}' failed after {self._current_retry} retries: {e}")
                    state = self.hooks.run_on_error(state, e)
                    
                    if self.fallback_target:
                        return state, self.fallback_target
                    raise
    
    def reset(self) -> None:
        """重置重试状态"""
        self._current_retry = 0
        self._last_error = None
    
    def __repr__(self) -> str:
        return f"RetryEdge({self.source} -> {self.target}, max_retries={self.max_retries})"


@dataclass
class FallbackEdge(Edge):
    """降级边
    
    支持多级降级的边，当主目标失败时依次尝试备选目标。
    """
    fallback_targets: List[str] = field(default_factory=list)
    fallback_conditions: Dict[str, Callable[[AgentState], bool]] = field(default_factory=dict)
    on_fallback: Optional[Callable[[AgentState, str, str], AgentState]] = None
    
    def __post_init__(self):
        self.edge_type = EdgeType.FALLBACK
    
    def get_effective_target(self, state: AgentState) -> str:
        """获取有效目标
        
        根据条件判断使用主目标还是降级目标。
        """
        # 先尝试主目标
        if self.condition is None or self.condition.evaluate(state):
            return self.target
        
        # 依次尝试降级目标
        for fallback in self.fallback_targets:
            condition = self.fallback_conditions.get(fallback)
            if condition is None or condition(state):
                if self.on_fallback:
                    state = self.on_fallback(state, self.target, fallback)
                logger.info(f"FallbackEdge '{self.name}' using fallback: {fallback}")
                return fallback
        
        # 返回第一个降级目标作为默认
        if self.fallback_targets:
            return self.fallback_targets[0]
        
        return self.target
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """带降级的遍历"""
        start_time = time.time()
        
        try:
            state = self.hooks.run_before(state)
            
            if self.transform_func:
                state = self.transform_func(state) or state
            
            state = self.hooks.run_after(state)
            
            target = self.get_effective_target(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, True)
            
            return state, target
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            
            logger.warning(f"FallbackEdge '{self.name}' error, trying fallback: {e}")
            state = self.hooks.run_on_error(state, e)
            
            # 返回第一个降级目标
            if self.fallback_targets:
                if self.on_fallback:
                    state = self.on_fallback(state, self.target, self.fallback_targets[0])
                return state, self.fallback_targets[0]
            
            raise
    
    def add_fallback(self, target: str, condition: Callable[[AgentState], bool] = None) -> 'FallbackEdge':
        """添加降级目标"""
        self.fallback_targets.append(target)
        if condition:
            self.fallback_conditions[target] = condition
        return self
    
    def __repr__(self) -> str:
        fallbacks = ", ".join(self.fallback_targets)
        return f"FallbackEdge({self.source} -> {self.target} | [{fallbacks}])"


@dataclass
class WeightedEdge(Edge):
    """带权重边
    
    用于实现权重路由、A/B测试等场景。
    """
    # 目标及其权重 {target: weight}
    weighted_targets: Dict[str, float] = field(default_factory=dict)
    routing_strategy: RoutingStrategy = RoutingStrategy.WEIGHTED_RANDOM
    
    # 轮询状态
    _round_robin_index: int = field(default=0, repr=False)
    _target_list: List[str] = field(default_factory=list, repr=False)
    
    # A/B 测试配置
    ab_test_name: str = ""
    ab_test_user_field: str = "thread_id"  # 用于分流的字段
    
    # 负载均衡状态
    _target_loads: Dict[str, int] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        self.edge_type = EdgeType.WEIGHTED
        self._target_list = list(self.weighted_targets.keys())
        self._target_loads = {t: 0 for t in self._target_list}
    
    def _select_weighted_random(self) -> str:
        """权重随机选择"""
        targets = list(self.weighted_targets.keys())
        weights = list(self.weighted_targets.values())
        total = sum(weights)
        normalized = [w / total for w in weights]
        return random.choices(targets, weights=normalized, k=1)[0]
    
    def _select_round_robin(self) -> str:
        """轮询选择"""
        if not self._target_list:
            return self.target
        target = self._target_list[self._round_robin_index % len(self._target_list)]
        self._round_robin_index += 1
        return target
    
    def _select_load_balance(self) -> str:
        """负载均衡选择"""
        if not self._target_loads:
            return self.target
        # 选择负载最小的目标
        min_load = min(self._target_loads.values())
        candidates = [t for t, l in self._target_loads.items() if l == min_load]
        selected = random.choice(candidates)
        self._target_loads[selected] += 1
        return selected
    
    def _select_ab_test(self, state: AgentState) -> str:
        """A/B 测试分流"""
        # 获取用户标识
        user_id = getattr(state, self.ab_test_user_field, state.thread_id)
        
        # 使用哈希进行稳定分流
        hash_value = hash(f"{self.ab_test_name}:{user_id}")
        
        targets = list(self.weighted_targets.keys())
        weights = list(self.weighted_targets.values())
        total = sum(weights)
        
        position = abs(hash_value) % int(total * 100)
        cumulative = 0
        
        for target, weight in zip(targets, weights):
            cumulative += weight * 100
            if position < cumulative:
                return target
        
        return targets[-1] if targets else self.target
    
    def select_target(self, state: AgentState) -> str:
        """根据策略选择目标"""
        if not self.weighted_targets:
            return self.target
        
        if self.routing_strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return self._select_weighted_random()
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin()
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCE:
            return self._select_load_balance()
        elif self.routing_strategy == RoutingStrategy.AB_TEST:
            return self._select_ab_test(state)
        elif self.routing_strategy == RoutingStrategy.RANDOM:
            return random.choice(list(self.weighted_targets.keys()))
        elif self.routing_strategy == RoutingStrategy.PRIORITY:
            # 按权重排序，选择最高权重
            return max(self.weighted_targets.items(), key=lambda x: x[1])[0]
        else:
            return self.target
    
    def get_target(self, state: AgentState) -> str:
        """获取目标节点"""
        return self.select_target(state)
    
    def release_load(self, target: str) -> None:
        """释放负载（用于负载均衡）"""
        if target in self._target_loads:
            self._target_loads[target] = max(0, self._target_loads[target] - 1)
    
    def get_load_stats(self) -> Dict[str, int]:
        """获取负载统计"""
        return dict(self._target_loads)
    
    def __repr__(self) -> str:
        targets_str = ", ".join([f"{t}:{w}" for t, w in self.weighted_targets.items()])
        return f"WeightedEdge({self.source} -> [{targets_str}], strategy={self.routing_strategy.value})"


@dataclass
class DelayedEdge(Edge):
    """延迟边
    
    在执行前添加延迟，用于限流、节奏控制等场景。
    """
    delay_seconds: float = 1.0
    delay_func: Optional[Callable[[AgentState], float]] = None  # 动态延迟函数
    jitter: float = 0.0  # 随机抖动（0-1）
    skip_condition: Optional[Callable[[AgentState], bool]] = None  # 跳过延迟的条件
    
    def __post_init__(self):
        self.edge_type = EdgeType.DELAYED
    
    def _calculate_delay(self, state: AgentState) -> float:
        """计算实际延迟"""
        # 动态延迟
        if self.delay_func:
            base_delay = self.delay_func(state)
        else:
            base_delay = self.delay_seconds
        
        # 添加抖动
        if self.jitter > 0:
            jitter_range = base_delay * self.jitter
            base_delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, base_delay)
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """带延迟的遍历"""
        start_time = time.time()
        
        try:
            state = self.hooks.run_before(state)
            
            # 检查是否跳过延迟
            should_delay = True
            if self.skip_condition and self.skip_condition(state):
                should_delay = False
            
            # 执行延迟
            if should_delay:
                delay = self._calculate_delay(state)
                if delay > 0:
                    logger.debug(f"DelayedEdge '{self.name}' waiting {delay:.2f}s")
                    time.sleep(delay)
            
            # 执行转换
            if self.transform_func:
                state = self.transform_func(state) or state
            
            state = self.hooks.run_after(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, True)
            
            return state, self.get_target(state)
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            state = self.hooks.run_on_error(state, e)
            raise
    
    def __repr__(self) -> str:
        return f"DelayedEdge({self.source} -> {self.target}, delay={self.delay_seconds}s)"


@dataclass
class EventEdge(Edge):
    """事件驱动边
    
    响应特定事件的边，用于实现事件驱动的工作流。
    """
    event_types: List[str] = field(default_factory=list)  # 响应的事件类型
    event_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    event_handler: Optional[Callable[[AgentState, str, Dict[str, Any]], AgentState]] = None
    timeout: float = 0  # 等待事件的超时时间（0表示不等待）
    
    # 事件队列
    _event_queue: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self.edge_type = EdgeType.EVENT
    
    def push_event(self, event_type: str, event_data: Dict[str, Any] = None) -> bool:
        """推送事件"""
        with self._lock:
            # 检查事件类型
            if self.event_types and event_type not in self.event_types:
                return False
            
            # 检查事件过滤
            if self.event_filter and not self.event_filter(event_type, event_data or {}):
                return False
            
            self._event_queue.append((event_type, event_data or {}))
            return True
    
    def pop_event(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """弹出事件"""
        with self._lock:
            if self._event_queue:
                return self._event_queue.pop(0)
            return None
    
    def has_event(self) -> bool:
        """是否有待处理事件"""
        with self._lock:
            return len(self._event_queue) > 0
    
    def clear_events(self) -> None:
        """清空事件队列"""
        with self._lock:
            self._event_queue.clear()
    
    def can_traverse(self, state: AgentState) -> bool:
        """只有在有事件时才能遍历"""
        if not self.enabled:
            return False
        if self.condition and not self.condition.evaluate(state):
            return False
        return self.has_event()
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """处理事件并遍历"""
        start_time = time.time()
        
        try:
            state = self.hooks.run_before(state)
            
            # 获取事件
            event = self.pop_event()
            if event:
                event_type, event_data = event
                
                # 记录事件信息
                state.metadata["last_event_type"] = event_type
                state.metadata["last_event_data"] = event_data
                
                # 处理事件
                if self.event_handler:
                    state = self.event_handler(state, event_type, event_data) or state
                
                logger.debug(f"EventEdge '{self.name}' processed event: {event_type}")
            
            # 执行转换
            if self.transform_func:
                state = self.transform_func(state) or state
            
            state = self.hooks.run_after(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, True)
            
            return state, self.get_target(state)
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            state = self.hooks.run_on_error(state, e)
            raise
    
    def __repr__(self) -> str:
        events = ", ".join(self.event_types) if self.event_types else "*"
        return f"EventEdge({self.source} -> {self.target}, events=[{events}])"


@dataclass
class TransformEdge(Edge):
    """转换边
    
    专门用于状态转换的边，支持多个转换器链式执行。
    """
    transformers: List[Callable[[AgentState], AgentState]] = field(default_factory=list)
    validate_before: Optional[Callable[[AgentState], bool]] = None
    validate_after: Optional[Callable[[AgentState], bool]] = None
    rollback_on_error: bool = False
    
    def __post_init__(self):
        self.edge_type = EdgeType.TRANSFORM
    
    def add_transformer(self, transformer: Callable[[AgentState], AgentState]) -> 'TransformEdge':
        """添加转换器"""
        self.transformers.append(transformer)
        return self
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """执行转换链"""
        start_time = time.time()
        original_state = state.copy() if self.rollback_on_error else None
        
        try:
            state = self.hooks.run_before(state)
            
            # 前置验证
            if self.validate_before and not self.validate_before(state):
                raise ValueError(f"TransformEdge '{self.name}' pre-validation failed")
            
            # 执行转换链
            for i, transformer in enumerate(self.transformers):
                try:
                    state = transformer(state) or state
                except Exception as e:
                    logger.error(f"TransformEdge '{self.name}' transformer {i} failed: {e}")
                    raise
            
            # 执行主转换函数
            if self.transform_func:
                state = self.transform_func(state) or state
            
            # 后置验证
            if self.validate_after and not self.validate_after(state):
                raise ValueError(f"TransformEdge '{self.name}' post-validation failed")
            
            state = self.hooks.run_after(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, True)
            
            return state, self.get_target(state)
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            
            # 回滚
            if self.rollback_on_error and original_state:
                logger.warning(f"TransformEdge '{self.name}' rolling back due to error: {e}")
                state = original_state
            
            state = self.hooks.run_on_error(state, e)
            raise
    
    def __repr__(self) -> str:
        return f"TransformEdge({self.source} -> {self.target}, transformers={len(self.transformers)})"


@dataclass
class AsyncEdge(Edge):
    """异步边
    
    支持异步执行的边，用于非阻塞操作。
    """
    async_transform: Optional[Callable[[AgentState], Any]] = None  # 异步转换函数
    gather_timeout: float = 30.0
    
    def __post_init__(self):
        self.edge_type = EdgeType.NORMAL  # 复用 NORMAL 类型
    
    async def async_traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """异步遍历"""
        start_time = time.time()
        
        try:
            state = self.hooks.run_before(state)
            
            # 执行异步转换
            if self.async_transform:
                state = await asyncio.wait_for(
                    self._run_async_transform(state),
                    timeout=self.gather_timeout
                )
            
            # 执行同步转换
            if self.transform_func:
                state = self.transform_func(state) or state
            
            state = self.hooks.run_after(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, True)
            
            return state, self.get_target(state)
            
        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            logger.error(f"AsyncEdge '{self.name}' timed out")
            raise
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_traversal(elapsed_ms, False)
            state = self.hooks.run_on_error(state, e)
            raise
    
    async def _run_async_transform(self, state: AgentState) -> AgentState:
        """执行异步转换"""
        if asyncio.iscoroutinefunction(self.async_transform):
            return await self.async_transform(state) or state
        else:
            # 在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.async_transform, state) or state
    
    def __repr__(self) -> str:
        return f"AsyncEdge({self.source} -> {self.target})"


@dataclass 
class CircuitBreakerEdge(Edge):
    """断路器边
    
    实现断路器模式，防止级联失败。
    """
    failure_threshold: int = 5  # 失败阈值
    recovery_timeout: float = 30.0  # 恢复超时（秒）
    half_open_requests: int = 3  # 半开状态允许的请求数
    fallback_target: str = ""  # 断路时的降级目标
    
    # 状态
    _failure_count: int = field(default=0, repr=False)
    _last_failure_time: Optional[datetime] = field(default=None, repr=False)
    _state: str = field(default="closed", repr=False)  # closed, open, half_open
    _half_open_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self.edge_type = EdgeType.FALLBACK
    
    def _check_state(self) -> str:
        """检查并更新断路器状态"""
        with self._lock:
            now = datetime.utcnow()
            
            if self._state == "open":
                # 检查是否应该进入半开状态
                if self._last_failure_time:
                    elapsed = (now - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = "half_open"
                        self._half_open_count = 0
                        logger.info(f"CircuitBreakerEdge '{self.name}' entering half-open state")
            
            return self._state
    
    def _record_success(self) -> None:
        """记录成功"""
        with self._lock:
            if self._state == "half_open":
                self._half_open_count += 1
                if self._half_open_count >= self.half_open_requests:
                    self._state = "closed"
                    self._failure_count = 0
                    logger.info(f"CircuitBreakerEdge '{self.name}' closed (recovered)")
            else:
                self._failure_count = 0
    
    def _record_failure(self) -> None:
        """记录失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            if self._state == "half_open":
                # 半开状态下失败，重新打开
                self._state = "open"
                logger.warning(f"CircuitBreakerEdge '{self.name}' re-opened (failure in half-open)")
            elif self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(f"CircuitBreakerEdge '{self.name}' opened (threshold reached)")
    
    def can_traverse(self, state: AgentState) -> bool:
        """检查是否可以遍历"""
        current_state = self._check_state()
        
        if current_state == "open":
            logger.debug(f"CircuitBreakerEdge '{self.name}' is open, blocking")
            return False
        
        return super().can_traverse(state)
    
    def traverse(self, state: AgentState) -> Tuple[AgentState, str]:
        """带断路器的遍历"""
        current_state = self._check_state()
        
        # 断路器打开时使用降级目标
        if current_state == "open" and self.fallback_target:
            state.metadata["circuit_breaker_state"] = "open"
            return state, self.fallback_target
        
        try:
            result = super().traverse(state)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            
            if self.fallback_target and self._check_state() == "open":
                state.metadata["circuit_breaker_state"] = "open"
                state.metadata["circuit_breaker_error"] = str(e)
                return state, self.fallback_target
            
            raise
    
    def get_circuit_state(self) -> Dict[str, Any]:
        """获取断路器状态"""
        with self._lock:
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "half_open_count": self._half_open_count
            }
    
    def reset(self) -> None:
        """重置断路器"""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_count = 0
    
    def __repr__(self) -> str:
        return f"CircuitBreakerEdge({self.source} -> {self.target}, state={self._state})"


# ==================== 预定义条件 ====================

class EdgeConditions:
    """预定义的边条件"""
    
    @staticmethod
    def has_tool_calls(state: AgentState) -> bool:
        """是否有待执行的工具调用"""
        return len(state.pending_tool_calls) > 0
    
    @staticmethod
    def no_tool_calls(state: AgentState) -> bool:
        """是否没有工具调用"""
        return len(state.pending_tool_calls) == 0
    
    @staticmethod
    def has_final_answer(state: AgentState) -> bool:
        """是否有最终答案"""
        return state.final_answer is not None
    
    @staticmethod
    def should_continue(state: AgentState) -> bool:
        """是否应该继续执行"""
        return state.should_continue()
    
    @staticmethod
    def is_completed(state: AgentState) -> bool:
        """是否已完成"""
        return state.status == AgentStatus.COMPLETED
    
    @staticmethod
    def is_failed(state: AgentState) -> bool:
        """是否失败"""
        return state.status == AgentStatus.FAILED
    
    @staticmethod
    def waiting_for_human(state: AgentState) -> bool:
        """是否等待人工输入"""
        return state.waiting_for_human
    
    @staticmethod
    def max_iterations_reached(state: AgentState) -> bool:
        """是否达到最大迭代次数"""
        return state.iteration >= state.max_iterations
    
    @staticmethod
    def has_error(state: AgentState) -> bool:
        """是否有错误"""
        return state.error is not None
    
    @staticmethod
    def is_running(state: AgentState) -> bool:
        """是否正在运行"""
        return state.status == AgentStatus.RUNNING
    
    @staticmethod
    def is_idle(state: AgentState) -> bool:
        """是否空闲"""
        return state.status == AgentStatus.IDLE
    
    @staticmethod
    def is_paused(state: AgentState) -> bool:
        """是否暂停"""
        return state.status == AgentStatus.PAUSED
    
    @staticmethod
    def has_messages(state: AgentState) -> bool:
        """是否有消息"""
        return len(state.messages) > 0
    
    @staticmethod
    def has_human_message(state: AgentState) -> bool:
        """最后一条是否是人类消息"""
        last = state.get_last_message()
        return last is not None and last.type == MessageType.HUMAN
    
    @staticmethod
    def has_ai_message(state: AgentState) -> bool:
        """最后一条是否是 AI 消息"""
        last = state.get_last_message()
        return last is not None and last.type == MessageType.AI
    
    @staticmethod
    def has_tool_message(state: AgentState) -> bool:
        """最后一条是否是工具消息"""
        last = state.get_last_message()
        return last is not None and last.type == MessageType.TOOL
    
    @staticmethod
    def has_plan(state: AgentState) -> bool:
        """是否有计划"""
        return len(state.plan) > 0
    
    @staticmethod
    def plan_completed(state: AgentState) -> bool:
        """计划是否完成"""
        return state.current_step >= len(state.plan)
    
    @staticmethod
    def has_reflections(state: AgentState) -> bool:
        """是否有反思"""
        return len(state.reflections) > 0
    
    @staticmethod
    def iteration_less_than(n: int) -> Callable[[AgentState], bool]:
        """迭代次数小于 n"""
        def check(state: AgentState) -> bool:
            return state.iteration < n
        return check
    
    @staticmethod
    def message_count_greater_than(n: int) -> Callable[[AgentState], bool]:
        """消息数大于 n"""
        def check(state: AgentState) -> bool:
            return len(state.messages) > n
        return check
    
    @staticmethod
    def metadata_equals(key: str, value: Any) -> Callable[[AgentState], bool]:
        """元数据等于指定值"""
        def check(state: AgentState) -> bool:
            return state.metadata.get(key) == value
        return check
    
    @staticmethod
    def metadata_contains(key: str) -> Callable[[AgentState], bool]:
        """元数据包含指定键"""
        def check(state: AgentState) -> bool:
            return key in state.metadata
        return check
    
    @staticmethod
    def data_equals(key: str, value: Any) -> Callable[[AgentState], bool]:
        """自定义数据等于指定值"""
        def check(state: AgentState) -> bool:
            return state.data.get(key) == value
        return check
    
    @staticmethod
    def content_contains(pattern: str) -> Callable[[AgentState], bool]:
        """最后消息内容包含模式"""
        def check(state: AgentState) -> bool:
            last = state.get_last_message()
            if last and last.content:
                return pattern in last.content
            return False
        return check
    
    @staticmethod
    def content_matches(regex: str) -> Callable[[AgentState], bool]:
        """最后消息内容匹配正则"""
        def check(state: AgentState) -> bool:
            last = state.get_last_message()
            if last and last.content:
                return bool(re.search(regex, last.content))
            return False
        return check
    
    @staticmethod
    def always_true(state: AgentState) -> bool:
        """始终为真"""
        return True
    
    @staticmethod
    def always_false(state: AgentState) -> bool:
        """始终为假"""
        return False


# ==================== 路由器 ====================

class BaseRouter(ABC):
    """路由器基类"""
    
    def __init__(self, name: str = ""):
        self.name = name or f"router_{uuid.uuid4().hex[:8]}"
        self._metrics = EdgeMetrics()
    
    @abstractmethod
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        """选择一条边"""
        pass
    
    def get_metrics(self) -> EdgeMetrics:
        return self._metrics


class PriorityRouter(BaseRouter):
    """优先级路由器
    
    根据边的优先级选择。
    """
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        # 按优先级排序
        sorted_edges = sorted(
            [e for e in edges if e.enabled],
            key=lambda e: e.priority,
            reverse=True
        )
        
        # 选择第一个可以遍历的边
        for edge in sorted_edges:
            if edge.can_traverse(state):
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.record_traversal(elapsed_ms, True)
                return edge
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, False)
        return None


class WeightedRouter(BaseRouter):
    """权重路由器
    
    根据边的权重随机选择。
    """
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        # 过滤可遍历的边
        traversable = [e for e in edges if e.enabled and e.can_traverse(state)]
        
        if not traversable:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_traversal(elapsed_ms, False)
            return None
        
        # 按权重选择
        weights = [e.weight for e in traversable]
        total = sum(weights)
        normalized = [w / total for w in weights]
        
        selected = random.choices(traversable, weights=normalized, k=1)[0]
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, True)
        
        return selected


class RoundRobinRouter(BaseRouter):
    """轮询路由器"""
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._index = 0
        self._lock = threading.Lock()
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        traversable = [e for e in edges if e.enabled and e.can_traverse(state)]
        
        if not traversable:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_traversal(elapsed_ms, False)
            return None
        
        with self._lock:
            selected = traversable[self._index % len(traversable)]
            self._index += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, True)
        
        return selected


class LoadBalanceRouter(BaseRouter):
    """负载均衡路由器
    
    选择负载最小的边。
    """
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._loads: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        traversable = [e for e in edges if e.enabled and e.can_traverse(state)]
        
        if not traversable:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_traversal(elapsed_ms, False)
            return None
        
        with self._lock:
            # 初始化负载
            for edge in traversable:
                if edge.edge_id not in self._loads:
                    self._loads[edge.edge_id] = 0
            
            # 选择负载最小的
            min_load = min(self._loads.get(e.edge_id, 0) for e in traversable)
            candidates = [e for e in traversable if self._loads.get(e.edge_id, 0) == min_load]
            selected = random.choice(candidates)
            
            # 增加负载
            self._loads[selected.edge_id] = self._loads.get(selected.edge_id, 0) + 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, True)
        
        return selected
    
    def release_load(self, edge_id: str) -> None:
        """释放负载"""
        with self._lock:
            if edge_id in self._loads:
                self._loads[edge_id] = max(0, self._loads[edge_id] - 1)


class ABTestRouter(BaseRouter):
    """A/B 测试路由器
    
    根据用户标识进行稳定分流。
    """
    
    def __init__(self, name: str = "", test_name: str = "", 
                 user_field: str = "thread_id"):
        super().__init__(name)
        self.test_name = test_name or name
        self.user_field = user_field
        self._variants: Dict[str, float] = {}  # edge_id -> weight
    
    def set_variant_weight(self, edge_id: str, weight: float) -> None:
        """设置变体权重"""
        self._variants[edge_id] = weight
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        traversable = [e for e in edges if e.enabled and e.can_traverse(state)]
        
        if not traversable:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_traversal(elapsed_ms, False)
            return None
        
        # 获取用户标识
        user_id = getattr(state, self.user_field, state.thread_id)
        
        # 计算哈希
        hash_value = hash(f"{self.test_name}:{user_id}")
        
        # 准备权重
        weights = []
        for edge in traversable:
            w = self._variants.get(edge.edge_id, edge.weight)
            weights.append(w)
        
        total = sum(weights)
        position = abs(hash_value) % int(total * 1000)
        
        cumulative = 0
        selected = traversable[0]
        
        for edge, weight in zip(traversable, weights):
            cumulative += weight * 1000
            if position < cumulative:
                selected = edge
                break
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, True)
        
        return selected


class ConditionalRouter(BaseRouter):
    """条件路由器
    
    根据条件函数决定路由。
    """
    
    def __init__(self, name: str = "", 
                 route_func: Callable[[AgentState], str] = None):
        super().__init__(name)
        self.route_func = route_func
    
    def route(self, state: AgentState, edges: List[Edge]) -> Optional[Edge]:
        start_time = time.time()
        
        if not self.route_func:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.record_traversal(elapsed_ms, False)
            return None
        
        # 获取目标
        target = self.route_func(state)
        
        # 查找匹配的边
        for edge in edges:
            if edge.enabled and edge.target == target and edge.can_traverse(state):
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.record_traversal(elapsed_ms, True)
                return edge
        
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.record_traversal(elapsed_ms, False)
        return None


# ==================== 边管理器 ====================

class EdgeManager:
    """边管理器
    
    管理图中的所有边，提供查询、验证、统计等功能。
    """
    
    def __init__(self):
        self._edges: Dict[str, Edge] = {}
        self._source_index: Dict[str, List[str]] = {}  # source -> [edge_ids]
        self._target_index: Dict[str, List[str]] = {}  # target -> [edge_ids]
        self._type_index: Dict[EdgeType, List[str]] = {}  # type -> [edge_ids]
        self._router: Optional[BaseRouter] = None
        self._lock = threading.RLock()
    
    def add_edge(self, edge: Edge) -> None:
        """添加边"""
        with self._lock:
            self._edges[edge.edge_id] = edge
            
            # 更新源索引
            if edge.source not in self._source_index:
                self._source_index[edge.source] = []
            self._source_index[edge.source].append(edge.edge_id)
            
            # 更新目标索引
            if edge.target not in self._target_index:
                self._target_index[edge.target] = []
            self._target_index[edge.target].append(edge.edge_id)
            
            # 更新类型索引
            if edge.edge_type not in self._type_index:
                self._type_index[edge.edge_type] = []
            self._type_index[edge.edge_type].append(edge.edge_id)
            
            logger.debug(f"Added edge: {edge}")
    
    def remove_edge(self, edge_id: str) -> Optional[Edge]:
        """移除边"""
        with self._lock:
            edge = self._edges.pop(edge_id, None)
            if edge:
                # 更新索引
                if edge.source in self._source_index:
                    self._source_index[edge.source] = [
                        eid for eid in self._source_index[edge.source] if eid != edge_id
                    ]
                if edge.target in self._target_index:
                    self._target_index[edge.target] = [
                        eid for eid in self._target_index[edge.target] if eid != edge_id
                    ]
                if edge.edge_type in self._type_index:
                    self._type_index[edge.edge_type] = [
                        eid for eid in self._type_index[edge.edge_type] if eid != edge_id
                    ]
            return edge
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """获取边"""
        return self._edges.get(edge_id)
    
    def get_edges_from(self, source: str) -> List[Edge]:
        """获取从指定节点出发的所有边"""
        edge_ids = self._source_index.get(source, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def get_edges_to(self, target: str) -> List[Edge]:
        """获取指向指定节点的所有边"""
        edge_ids = self._target_index.get(target, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """获取指定类型的所有边"""
        edge_ids = self._type_index.get(edge_type, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def get_all_edges(self) -> List[Edge]:
        """获取所有边"""
        return list(self._edges.values())
    
    def get_nodes(self) -> Set[str]:
        """获取所有节点"""
        nodes = set()
        for edge in self._edges.values():
            nodes.add(edge.source)
            nodes.add(edge.target)
            if isinstance(edge, ParallelEdge):
                nodes.update(edge.targets)
        return nodes
    
    def set_router(self, router: BaseRouter) -> None:
        """设置路由器"""
        self._router = router
    
    def route(self, state: AgentState, source: str) -> Optional[Edge]:
        """从指定源路由"""
        edges = self.get_edges_from(source)
        
        if not edges:
            return None
        
        if self._router:
            return self._router.route(state, edges)
        
        # 默认：优先级路由
        sorted_edges = sorted(
            [e for e in edges if e.enabled],
            key=lambda e: e.priority,
            reverse=True
        )
        
        for edge in sorted_edges:
            if edge.can_traverse(state):
                return edge
        
        return None
    
    def validate(self) -> List[str]:
        """验证所有边
        
        Returns:
            错误消息列表
        """
        errors = []
        nodes = self.get_nodes()
        
        for edge in self._edges.values():
            # 检查源节点
            if edge.source not in nodes and edge.source != "__start__":
                errors.append(f"Edge '{edge.name}': source '{edge.source}' not found")
            
            # 检查目标节点
            if edge.target not in nodes and edge.target != "__end__":
                errors.append(f"Edge '{edge.name}': target '{edge.target}' not found")
            
            # 检查并行边的目标
            if isinstance(edge, ParallelEdge):
                for target in edge.targets:
                    if target not in nodes and target != "__end__":
                        errors.append(f"ParallelEdge '{edge.name}': target '{target}' not found")
            
            # 检查降级边的目标
            if isinstance(edge, FallbackEdge):
                for target in edge.fallback_targets:
                    if target not in nodes and target != "__end__":
                        errors.append(f"FallbackEdge '{edge.name}': fallback target '{target}' not found")
        
        return errors
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_traversals = 0
        total_time = 0.0
        type_counts = {}
        
        for edge in self._edges.values():
            total_traversals += edge.metrics.traversal_count
            total_time += edge.metrics.total_time_ms
            
            type_name = edge.edge_type.value
            if type_name not in type_counts:
                type_counts[type_name] = 0
            type_counts[type_name] += 1
        
        return {
            "total_edges": len(self._edges),
            "total_nodes": len(self.get_nodes()),
            "total_traversals": total_traversals,
            "total_time_ms": round(total_time, 2),
            "type_counts": type_counts,
            "sources": list(self._source_index.keys()),
            "targets": list(self._target_index.keys())
        }
    
    def get_edge_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有边的指标"""
        return {
            edge.edge_id: {
                "name": edge.name,
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "metrics": edge.metrics.to_dict()
            }
            for edge in self._edges.values()
        }
    
    def reset_metrics(self) -> None:
        """重置所有边的指标"""
        for edge in self._edges.values():
            edge.reset_metrics()
    
    def enable_all(self) -> None:
        """启用所有边"""
        for edge in self._edges.values():
            edge.enabled = True
    
    def disable_all(self) -> None:
        """禁用所有边"""
        for edge in self._edges.values():
            edge.enabled = False
    
    def enable_edges(self, edge_ids: List[str]) -> None:
        """启用指定边"""
        for eid in edge_ids:
            if eid in self._edges:
                self._edges[eid].enabled = True
    
    def disable_edges(self, edge_ids: List[str]) -> None:
        """禁用指定边"""
        for eid in edge_ids:
            if eid in self._edges:
                self._edges[eid].enabled = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "edges": [e.to_dict() for e in self._edges.values()],
            "stats": self.get_stats()
        }
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据
        
        返回适合可视化的图结构数据。
        """
        nodes = []
        edges_data = []
        
        # 收集节点
        node_set = self.get_nodes()
        for node in node_set:
            incoming = len(self.get_edges_to(node))
            outgoing = len(self.get_edges_from(node))
            nodes.append({
                "id": node,
                "label": node,
                "incoming_edges": incoming,
                "outgoing_edges": outgoing
            })
        
        # 收集边
        for edge in self._edges.values():
            edge_data = {
                "id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "label": edge.name,
                "enabled": edge.enabled,
                "weight": edge.weight,
                "priority": edge.priority,
                "traversal_count": edge.metrics.traversal_count
            }
            
            # 特殊边类型的额外信息
            if isinstance(edge, ParallelEdge):
                edge_data["parallel_targets"] = edge.targets
            elif isinstance(edge, FallbackEdge):
                edge_data["fallback_targets"] = edge.fallback_targets
            elif isinstance(edge, WeightedEdge):
                edge_data["weighted_targets"] = edge.weighted_targets
            elif isinstance(edge, RetryEdge):
                edge_data["max_retries"] = edge.max_retries
            elif isinstance(edge, CircuitBreakerEdge):
                edge_data["circuit_state"] = edge.get_circuit_state()
            
            edges_data.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges_data,
            "stats": self.get_stats()
        }
    
    def clear(self) -> None:
        """清空所有边"""
        with self._lock:
            self._edges.clear()
            self._source_index.clear()
            self._target_index.clear()
            self._type_index.clear()


def should_continue_condition(state: AgentState) -> str:
    """标准的继续/结束条件函数
    
    用于 ReAct 循环的条件判断。
    返回 "continue", "end", 或 "tools"。
    """
    # 检查是否完成
    if state.final_answer is not None:
        return "end"
    
    # 检查是否有错误
    if state.error is not None:
        return "end"
    
    # 检查是否达到最大迭代
    if state.iteration >= state.max_iterations:
        return "end"
    
    # 检查是否有工具调用
    if state.pending_tool_calls:
        return "tools"
    
    # 检查最后一条消息
    last_message = state.get_last_message()
    if last_message and last_message.tool_calls:
        return "tools"
    
    # 继续执行
    return "continue"


def route_by_tool_calls(state: AgentState) -> str:
    """根据工具调用情况路由
    
    返回 "tools" 或 "agent"。
    """
    if state.pending_tool_calls:
        return "tools"
    
    last_message = state.get_last_message()
    if last_message and last_message.tool_calls:
        return "tools"
    
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """Agent 节点后的路由
    
    决定是执行工具、继续推理还是结束。
    """
    # 有最终答案
    if state.final_answer:
        return "__end__"
    
    # 有工具调用
    if state.pending_tool_calls:
        return "tools"
    
    # 检查最后消息
    last_message = state.get_last_message()
    if last_message:
        if last_message.tool_calls:
            return "tools"
        if last_message.content and "最终答案" in last_message.content:
            return "__end__"
    
    # 达到最大迭代
    if not state.should_continue():
        return "__end__"
    
    # 继续到 Agent
    return "agent"


def route_after_tools(state: AgentState) -> str:
    """工具节点后的路由
    
    工具执行完成后返回 Agent。
    """
    return "agent"


# ==================== 边构建器 ====================

class EdgeBuilder:
    """边构建器
    
    提供流式 API 来构建边。
    """
    
    def __init__(self, source: str):
        self.source = source
        self._target: str = None
        self._condition: EdgeCondition = None
        self._condition_func: Callable = None
        self._branches: Dict[str, str] = {}
        self._max_iterations: int = 10
        self._is_loop: bool = False
        self._is_parallel: bool = False
        self._parallel_targets: List[str] = []
        self._name: str = ""
        self._priority: int = 0
        self._weight: float = 1.0
        self._transform_func: Callable = None
        self._hooks: EdgeHooks = EdgeHooks()
        self._metadata: Dict[str, Any] = {}
        
        # 高级边类型
        self._timeout: float = 0
        self._timeout_target: str = ""
        self._max_retries: int = 0
        self._retry_delay: float = 1.0
        self._fallback_targets: List[str] = []
        self._weighted_targets: Dict[str, float] = {}
        self._routing_strategy: RoutingStrategy = RoutingStrategy.FIRST_MATCH
        self._delay: float = 0
        self._event_types: List[str] = []
        self._transformers: List[Callable] = []
        self._circuit_breaker: bool = False
        self._failure_threshold: int = 5
    
    def to(self, target: str) -> 'EdgeBuilder':
        """设置目标节点"""
        self._target = target
        return self
    
    def named(self, name: str) -> 'EdgeBuilder':
        """设置边名称"""
        self._name = name
        return self
    
    def with_priority(self, priority: int) -> 'EdgeBuilder':
        """设置优先级"""
        self._priority = priority
        return self
    
    def with_weight(self, weight: float) -> 'EdgeBuilder':
        """设置权重"""
        self._weight = weight
        return self
    
    def with_condition(self, condition: EdgeCondition) -> 'EdgeBuilder':
        """设置条件"""
        self._condition = condition
        return self
    
    def when(self, condition_func: Callable[[AgentState], str]) -> 'EdgeBuilder':
        """设置条件函数"""
        self._condition_func = condition_func
        return self
    
    def branch(self, condition_result: str, target: str) -> 'EdgeBuilder':
        """添加分支"""
        self._branches[condition_result] = target
        return self
    
    def loop(self, max_iterations: int = 10) -> 'EdgeBuilder':
        """设置为循环边"""
        self._is_loop = True
        self._max_iterations = max_iterations
        return self
    
    def parallel(self, targets: List[str]) -> 'EdgeBuilder':
        """设置为并行边"""
        self._is_parallel = True
        self._parallel_targets = targets
        return self
    
    def with_timeout(self, seconds: float, timeout_target: str = "__timeout__") -> 'EdgeBuilder':
        """设置超时"""
        self._timeout = seconds
        self._timeout_target = timeout_target
        return self
    
    def with_retry(self, max_retries: int = 3, delay: float = 1.0) -> 'EdgeBuilder':
        """设置重试"""
        self._max_retries = max_retries
        self._retry_delay = delay
        return self
    
    def with_fallback(self, targets: List[str]) -> 'EdgeBuilder':
        """设置降级目标"""
        self._fallback_targets = targets
        return self
    
    def with_weighted_targets(self, targets: Dict[str, float]) -> 'EdgeBuilder':
        """设置权重目标"""
        self._weighted_targets = targets
        return self
    
    def with_routing_strategy(self, strategy: RoutingStrategy) -> 'EdgeBuilder':
        """设置路由策略"""
        self._routing_strategy = strategy
        return self
    
    def with_delay(self, seconds: float) -> 'EdgeBuilder':
        """设置延迟"""
        self._delay = seconds
        return self
    
    def on_events(self, event_types: List[str]) -> 'EdgeBuilder':
        """设置事件类型"""
        self._event_types = event_types
        return self
    
    def add_transformer(self, transformer: Callable[[AgentState], AgentState]) -> 'EdgeBuilder':
        """添加转换器"""
        self._transformers.append(transformer)
        return self
    
    def with_circuit_breaker(self, failure_threshold: int = 5) -> 'EdgeBuilder':
        """设置断路器"""
        self._circuit_breaker = True
        self._failure_threshold = failure_threshold
        return self
    
    def transform(self, func: Callable[[AgentState], AgentState]) -> 'EdgeBuilder':
        """设置转换函数"""
        self._transform_func = func
        return self
    
    def before(self, hook: Callable[[AgentState], AgentState]) -> 'EdgeBuilder':
        """添加前置钩子"""
        self._hooks.add_before(hook)
        return self
    
    def after(self, hook: Callable[[AgentState], AgentState]) -> 'EdgeBuilder':
        """添加后置钩子"""
        self._hooks.add_after(hook)
        return self
    
    def on_error(self, hook: Callable[[AgentState, Exception], AgentState]) -> 'EdgeBuilder':
        """添加错误钩子"""
        self._hooks.add_on_error(hook)
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'EdgeBuilder':
        """添加元数据"""
        self._metadata[key] = value
        return self
    
    def build(self) -> Edge:
        """构建边"""
        # 断路器边
        if self._circuit_breaker:
            return CircuitBreakerEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                failure_threshold=self._failure_threshold,
                fallback_target=self._fallback_targets[0] if self._fallback_targets else ""
            )
        
        # 事件边
        if self._event_types:
            return EventEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                event_types=self._event_types
            )
        
        # 转换边
        if self._transformers:
            return TransformEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                transformers=self._transformers
            )
        
        # 延迟边
        if self._delay > 0:
            return DelayedEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                delay_seconds=self._delay
            )
        
        # 权重边
        if self._weighted_targets:
            return WeightedEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                weighted_targets=self._weighted_targets,
                routing_strategy=self._routing_strategy
            )
        
        # 重试边
        if self._max_retries > 0:
            return RetryEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                max_retries=self._max_retries,
                retry_delay=self._retry_delay,
                fallback_target=self._fallback_targets[0] if self._fallback_targets else ""
            )
        
        # 超时边
        if self._timeout > 0:
            return TimeoutEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                timeout_seconds=self._timeout,
                timeout_target=self._timeout_target
            )
        
        # 降级边
        if self._fallback_targets:
            return FallbackEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                fallback_targets=self._fallback_targets
            )
        
        # 并行边
        if self._is_parallel:
            return ParallelEdge(
                source=self.source,
                target=self._parallel_targets[0] if self._parallel_targets else "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                targets=self._parallel_targets
            )
        
        # 循环边
        if self._is_loop:
            return LoopEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                max_iterations=self._max_iterations,
                continue_condition=self._condition.condition_func if self._condition else None
            )
        
        # 条件边
        if self._condition or self._condition_func or self._branches:
            return ConditionalEdge(
                source=self.source,
                target=self._target or "",
                name=self._name,
                priority=self._priority,
                weight=self._weight,
                condition=self._condition,
                transform_func=self._transform_func,
                hooks=self._hooks,
                metadata=self._metadata,
                condition_func=self._condition_func,
                branches=self._branches
            )
        
        # 普通边
        return Edge(
            source=self.source,
            target=self._target or "",
            name=self._name,
            priority=self._priority,
            weight=self._weight,
            condition=self._condition,
            transform_func=self._transform_func,
            hooks=self._hooks,
            metadata=self._metadata
        )


def edge_from(source: str) -> EdgeBuilder:
    """创建边构建器"""
    return EdgeBuilder(source)


# ==================== 便捷函数 ====================

def create_edge(source: str, target: str, **kwargs) -> Edge:
    """创建普通边"""
    return Edge(source=source, target=target, **kwargs)


def create_conditional_edge(source: str, condition_func: Callable[[AgentState], str],
                           branches: Dict[str, str] = None, **kwargs) -> ConditionalEdge:
    """创建条件边"""
    return ConditionalEdge(
        source=source,
        target="",
        condition_func=condition_func,
        branches=branches or {},
        **kwargs
    )


def create_loop_edge(source: str, target: str, max_iterations: int = 10,
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


def create_parallel_edge(source: str, targets: List[str], **kwargs) -> ParallelEdge:
    """创建并行边"""
    return ParallelEdge(
        source=source,
        target=targets[0] if targets else "",
        targets=targets,
        **kwargs
    )


def create_timeout_edge(source: str, target: str, timeout: float,
                       timeout_target: str = "__timeout__", **kwargs) -> TimeoutEdge:
    """创建超时边"""
    return TimeoutEdge(
        source=source,
        target=target,
        timeout_seconds=timeout,
        timeout_target=timeout_target,
        **kwargs
    )


def create_retry_edge(source: str, target: str, max_retries: int = 3,
                     retry_delay: float = 1.0, **kwargs) -> RetryEdge:
    """创建重试边"""
    return RetryEdge(
        source=source,
        target=target,
        max_retries=max_retries,
        retry_delay=retry_delay,
        **kwargs
    )


def create_fallback_edge(source: str, target: str,
                        fallback_targets: List[str], **kwargs) -> FallbackEdge:
    """创建降级边"""
    return FallbackEdge(
        source=source,
        target=target,
        fallback_targets=fallback_targets,
        **kwargs
    )


def create_weighted_edge(source: str, weighted_targets: Dict[str, float],
                        strategy: RoutingStrategy = RoutingStrategy.WEIGHTED_RANDOM,
                        **kwargs) -> WeightedEdge:
    """创建权重边"""
    return WeightedEdge(
        source=source,
        target=list(weighted_targets.keys())[0] if weighted_targets else "",
        weighted_targets=weighted_targets,
        routing_strategy=strategy,
        **kwargs
    )


def create_delayed_edge(source: str, target: str, delay: float, **kwargs) -> DelayedEdge:
    """创建延迟边"""
    return DelayedEdge(
        source=source,
        target=target,
        delay_seconds=delay,
        **kwargs
    )


def create_event_edge(source: str, target: str,
                     event_types: List[str], **kwargs) -> EventEdge:
    """创建事件边"""
    return EventEdge(
        source=source,
        target=target,
        event_types=event_types,
        **kwargs
    )


def create_transform_edge(source: str, target: str,
                         transformers: List[Callable[[AgentState], AgentState]] = None,
                         **kwargs) -> TransformEdge:
    """创建转换边"""
    return TransformEdge(
        source=source,
        target=target,
        transformers=transformers or [],
        **kwargs
    )


def create_circuit_breaker_edge(source: str, target: str,
                               failure_threshold: int = 5,
                               fallback_target: str = "",
                               **kwargs) -> CircuitBreakerEdge:
    """创建断路器边"""
    return CircuitBreakerEdge(
        source=source,
        target=target,
        failure_threshold=failure_threshold,
        fallback_target=fallback_target,
        **kwargs
    )


def create_async_edge(source: str, target: str,
                     async_transform: Callable = None, **kwargs) -> AsyncEdge:
    """创建异步边"""
    return AsyncEdge(
        source=source,
        target=target,
        async_transform=async_transform,
        **kwargs
    )


# ==================== 条件构建器 ====================

def condition(name: str, func: Callable[[AgentState], bool],
              priority: int = 0, description: str = "") -> EdgeCondition:
    """创建条件"""
    return EdgeCondition(
        name=name,
        condition_func=func,
        priority=priority,
        description=description
    )


def state_condition(name: str, field_path: str, operator: ComparisonOperator,
                   expected_value: Any, **kwargs) -> StateCondition:
    """创建状态条件"""
    return StateCondition(
        name=name,
        condition_func=None,
        field_path=field_path,
        operator=operator,
        expected_value=expected_value,
        **kwargs
    )


def composite_condition(name: str, conditions: List[EdgeCondition],
                       operator: ConditionOperator = ConditionOperator.AND,
                       **kwargs) -> CompositeCondition:
    """创建复合条件"""
    return CompositeCondition(
        name=name,
        condition_func=None,
        conditions=conditions,
        operator=operator,
        **kwargs
    )


def threshold_condition(name: str, field_path: str, min_value: float = None,
                       max_value: float = None, **kwargs) -> ThresholdCondition:
    """创建阈值条件"""
    return ThresholdCondition(
        name=name,
        condition_func=None,
        field_path=field_path,
        min_value=min_value,
        max_value=max_value,
        **kwargs
    )


def time_condition(name: str, **kwargs) -> TimeCondition:
    """创建时间条件"""
    return TimeCondition(
        name=name,
        condition_func=None,
        **kwargs
    )


def message_condition(name: str, **kwargs) -> MessageCondition:
    """创建消息条件"""
    return MessageCondition(
        name=name,
        condition_func=None,
        **kwargs
    )


# ==================== 路由器构建器 ====================

def priority_router(name: str = "") -> PriorityRouter:
    """创建优先级路由器"""
    return PriorityRouter(name)


def weighted_router(name: str = "") -> WeightedRouter:
    """创建权重路由器"""
    return WeightedRouter(name)


def round_robin_router(name: str = "") -> RoundRobinRouter:
    """创建轮询路由器"""
    return RoundRobinRouter(name)


def load_balance_router(name: str = "") -> LoadBalanceRouter:
    """创建负载均衡路由器"""
    return LoadBalanceRouter(name)


def ab_test_router(name: str = "", test_name: str = "",
                  user_field: str = "thread_id") -> ABTestRouter:
    """创建 A/B 测试路由器"""
    return ABTestRouter(name, test_name, user_field)


def conditional_router(name: str = "",
                      route_func: Callable[[AgentState], str] = None) -> ConditionalRouter:
    """创建条件路由器"""
    return ConditionalRouter(name, route_func)

