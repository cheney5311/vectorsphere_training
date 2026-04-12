"""
强化学习算法 - 生产级实现

实现多种强化学习方法用于序贯决策：

基础方法：
- Q-Learning
- Double Q-Learning
- SARSA
- Expected SARSA
- N-step TD

高级方法：
- DQN 简化版（经验回放、目标网络）
- Actor-Critic
- Policy Gradient (REINFORCE)
- PPO 简化版

高级特性：
- 经验回放缓冲区（随机采样、优先采样）
- 目标网络（定期更新）
- 探索策略（ε-greedy、Boltzmann、UCB）
- 奖励归一化/裁剪
- 状态归一化
- 熵正则化

使用示例：
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
        learning_rate=0.01,
        discount_factor=0.99,
        epsilon=0.1,
        extra={
            "method": "double_q_learning",
            "exploration": "epsilon_decay",
            "use_replay_buffer": True,
            "replay_buffer_size": 10000,
            "batch_size": 32
        }
    )
    
    algo = ReinforcementLearningAlgorithm(config)
    result = algo.suggest(context)
"""

import logging
import math
import random
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

from .base import (
    BaseAlgorithm,
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext,
    AlgorithmResult
)

logger = logging.getLogger(__name__)


# ==================== 枚举类型定义 ====================

class RLMethod(Enum):
    """强化学习方法"""
    Q_LEARNING = "q_learning"
    DOUBLE_Q_LEARNING = "double_q_learning"
    SARSA = "sarsa"
    EXPECTED_SARSA = "expected_sarsa"
    N_STEP_TD = "n_step_td"
    DQN = "dqn"
    ACTOR_CRITIC = "actor_critic"
    REINFORCE = "reinforce"
    PPO = "ppo"


class ExplorationStrategy(Enum):
    """探索策略"""
    EPSILON_GREEDY = "epsilon_greedy"
    EPSILON_DECAY = "epsilon_decay"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    NOISY = "noisy"


class RewardShaping(Enum):
    """奖励塑形方式"""
    NONE = "none"
    CLIP = "clip"
    NORMALIZE = "normalize"
    SCALE = "scale"


# ==================== 数据结构定义 ====================

@dataclass
class State:
    """状态表示 - 增强版"""
    features: Dict[str, Any]
    state_id: str = ""
    vector: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.state_id:
            self.state_id = self._compute_state_id()
        if self.vector is None:
            self.vector = self._to_vector()
    
    def _compute_state_id(self) -> str:
        """计算状态ID"""
        sorted_items = sorted(self.features.items())
        state_str = "|".join(f"{k}={v}" for k, v in sorted_items)
        return hashlib.md5(state_str.encode()).hexdigest()[:12]
    
    def _to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        values = []
        for key in sorted(self.features.keys()):
            value = self.features[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                values.append(hash(value) % 1000 / 1000.0)
        return np.array(values) if values else np.array([0.0])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'features': self.features,
            'state_id': self.state_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        return cls(
            features=data.get('features', {}),
            state_id=data.get('state_id', '')
        )


@dataclass
class Transition:
    """状态转换 - 增强版"""
    state: State
    action: Dict[str, Any]
    reward: float
    next_state: Optional[State]
    done: bool = False
    action_id: str = ""
    priority: float = 1.0
    td_error: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.action_id:
            self.action_id = self._compute_action_id()
    
    def _compute_action_id(self) -> str:
        sorted_items = sorted(self.action.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.to_dict(),
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state.to_dict() if self.next_state else None,
            'done': self.done,
            'priority': self.priority
        }


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(
        self, 
        capacity: int = 10000,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Args:
            capacity: 缓冲区容量
            prioritized: 是否使用优先采样
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._prioritized = prioritized
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = 0.001
        self._priorities: List[float] = []
        self._max_priority = 1.0
    
    def push(self, transition: Transition) -> None:
        """添加转换"""
        self._buffer.append(transition)
        
        if self._prioritized:
            # 新转换使用最大优先级
            self._priorities.append(self._max_priority ** self._alpha)
            
            # 保持优先级列表长度
            if len(self._priorities) > self._capacity:
                self._priorities.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], Optional[np.ndarray]]:
        """采样一批转换
        
        Returns:
            (转换列表, 重要性权重)
        """
        if len(self._buffer) < batch_size:
            batch_size = len(self._buffer)
        
        if self._prioritized and self._priorities:
            # 优先采样
            priorities = np.array(self._priorities[:len(self._buffer)])
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(
                len(self._buffer), 
                size=batch_size, 
                replace=False,
                p=probabilities
            )
            
            # 计算重要性采样权重
            weights = (len(self._buffer) * probabilities[indices]) ** (-self._beta)
            weights = weights / weights.max()
            
            # 增加 beta
            self._beta = min(1.0, self._beta + self._beta_increment)
            
            transitions = [self._buffer[i] for i in indices]
            return transitions, weights
        else:
            # 随机采样
            indices = random.sample(range(len(self._buffer)), batch_size)
            transitions = [self._buffer[i] for i in indices]
            return transitions, None
    
    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """更新优先级"""
        if not self._prioritized:
            return
        
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self._priorities):
                priority = (abs(td_error) + 1e-6) ** self._alpha
                self._priorities[idx] = priority
                self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        self._buffer.clear()
        self._priorities.clear()


# ==================== Q值表 ====================

class QTable:
    """Q值表 - 增强版
    
    存储状态-动作对的Q值，支持多种操作。
    """
    
    def __init__(self, default_value: float = 0.0):
        self._table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: default_value))
        self._default_value = default_value
        self._visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_update: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
    def get(self, state_id: str, action_id: str) -> float:
        """获取Q值"""
        return self._table[state_id][action_id]
    
    def set(self, state_id: str, action_id: str, value: float) -> None:
        """设置Q值"""
        self._table[state_id][action_id] = value
        self._visit_counts[state_id][action_id] += 1
        self._last_update[state_id][action_id] = datetime.now()
    
    def get_max(self, state_id: str) -> Tuple[float, Optional[str]]:
        """获取状态下的最大Q值和对应动作"""
        if state_id not in self._table or not self._table[state_id]:
            return self._default_value, None
        
        actions = self._table[state_id]
        best_action = max(actions, key=actions.get)
        return actions[best_action], best_action
    
    def get_all_actions(self, state_id: str) -> Dict[str, float]:
        """获取状态下所有动作的Q值"""
        return dict(self._table[state_id])
    
    def get_action_probabilities(
        self, 
        state_id: str, 
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """获取动作的Softmax概率"""
        q_values = self.get_all_actions(state_id)
        if not q_values:
            return {}
        
        # Softmax
        max_q = max(q_values.values())
        exp_values = {a: math.exp((q - max_q) / temperature) for a, q in q_values.items()}
        total = sum(exp_values.values())
        
        return {a: e / total for a, e in exp_values.items()}
    
    def get_visit_count(self, state_id: str, action_id: str) -> int:
        """获取访问次数"""
        return self._visit_counts[state_id][action_id]
    
    def get_ucb_value(
        self, 
        state_id: str, 
        action_id: str, 
        total_visits: int,
        c: float = 2.0
    ) -> float:
        """获取UCB值"""
        q_value = self.get(state_id, action_id)
        visit_count = self._visit_counts[state_id][action_id]
        
        if visit_count == 0:
            return float('inf')
        
        exploration_bonus = c * math.sqrt(math.log(total_visits + 1) / visit_count)
        return q_value + exploration_bonus
    
    def copy(self) -> 'QTable':
        """复制Q表"""
        new_table = QTable(self._default_value)
        for state_id, actions in self._table.items():
            for action_id, value in actions.items():
                new_table._table[state_id][action_id] = value
                new_table._visit_counts[state_id][action_id] = self._visit_counts[state_id][action_id]
        return new_table
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        all_q_values = [v for actions in self._table.values() for v in actions.values()]
        all_visits = [v for counts in self._visit_counts.values() for v in counts.values()]
        
        return {
            'num_states': len(self._table),
            'num_state_action_pairs': sum(len(a) for a in self._table.values()),
            'q_value_mean': sum(all_q_values) / len(all_q_values) if all_q_values else 0,
            'q_value_max': max(all_q_values) if all_q_values else 0,
            'q_value_min': min(all_q_values) if all_q_values else 0,
            'total_visits': sum(all_visits)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'table': {k: dict(v) for k, v in self._table.items()},
            'visit_counts': {k: dict(v) for k, v in self._visit_counts.items()},
            'default_value': self._default_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QTable':
        """从字典创建"""
        table = cls(data.get('default_value', 0.0))
        for state_id, actions in data.get('table', {}).items():
            for action_id, value in actions.items():
                table._table[state_id][action_id] = value
        for state_id, counts in data.get('visit_counts', {}).items():
            for action_id, count in counts.items():
                table._visit_counts[state_id][action_id] = count
        return table


# ==================== 奖励处理 ====================

class RewardProcessor:
    """奖励处理器"""
    
    def __init__(
        self,
        shaping: RewardShaping = RewardShaping.NONE,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        scale: float = 1.0
    ):
        self._shaping = shaping
        self._clip_range = clip_range
        self._scale = scale
        
        # 归一化统计
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
    
    def process(self, reward: float) -> float:
        """处理奖励"""
        # 更新统计
        self._update_statistics(reward)
        
        if self._shaping == RewardShaping.CLIP:
            return max(self._clip_range[0], min(self._clip_range[1], reward))
        
        elif self._shaping == RewardShaping.NORMALIZE:
            if self._reward_var > 0:
                return (reward - self._reward_mean) / math.sqrt(self._reward_var + 1e-8)
            return reward
        
        elif self._shaping == RewardShaping.SCALE:
            return reward * self._scale
        
        return reward
    
    def _update_statistics(self, reward: float) -> None:
        """增量更新统计"""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_var = ((self._reward_count - 1) * self._reward_var + delta * delta2) / self._reward_count


# ==================== 状态归一化 ====================

class StateNormalizer:
    """状态归一化器"""
    
    def __init__(self, clip_range: float = 5.0):
        self._mean: Dict[str, float] = defaultdict(float)
        self._var: Dict[str, float] = defaultdict(lambda: 1.0)
        self._count: Dict[str, int] = defaultdict(int)
        self._clip_range = clip_range
    
    def normalize(self, state: State) -> State:
        """归一化状态"""
        normalized_features = {}
        
        for key, value in state.features.items():
            if isinstance(value, (int, float)):
                self._update_statistics(key, value)
                
                std = math.sqrt(self._var[key] + 1e-8)
                normalized_value = (value - self._mean[key]) / std
                normalized_value = max(-self._clip_range, min(self._clip_range, normalized_value))
                normalized_features[key] = normalized_value
            else:
                normalized_features[key] = value
        
        return State(features=normalized_features)
    
    def _update_statistics(self, key: str, value: float) -> None:
        """增量更新统计"""
        self._count[key] += 1
        delta = value - self._mean[key]
        self._mean[key] += delta / self._count[key]
        delta2 = value - self._mean[key]
        self._var[key] = ((self._count[key] - 1) * self._var[key] + delta * delta2) / self._count[key]


class ReinforcementLearningAlgorithm(BaseAlgorithm):
    """强化学习算法 - 生产级实现
    
    支持多种方法：
    - q_learning: 标准Q-Learning
    - double_q_learning: Double Q-Learning
    - sarsa: SARSA
    - expected_sarsa: Expected SARSA
    - n_step_td: N-step TD
    - dqn: DQN简化版
    
    高级特性：
    - 经验回放缓冲区
    - 目标网络
    - 多种探索策略
    - 奖励/状态归一化
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        # 从配置获取参数
        extra = config.extra if config and config.extra else {}
        
        # 方法设置
        method_str = extra.get('method', 'q_learning')
        try:
            self._method = RLMethod(method_str)
        except ValueError:
            self._method = RLMethod.Q_LEARNING
            self.logger.warning(f"Unknown method '{method_str}', defaulting to Q-Learning")
        
        # 探索策略设置
        exploration_str = extra.get('exploration', 'epsilon_decay')
        try:
            self._exploration = ExplorationStrategy(exploration_str)
        except ValueError:
            self._exploration = ExplorationStrategy.EPSILON_DECAY
        
        # Q表
        self._q_table = QTable()
        self._target_q_table: Optional[QTable] = None  # 目标网络
        
        # Double Q-Learning的第二个Q表
        self._q_table_b: Optional[QTable] = None
        
        # 动作空间
        self._action_space: List[Dict[str, Any]] = []
        self._action_id_map: Dict[str, Dict[str, Any]] = {}
        
        # 探索参数
        self._epsilon = config.epsilon if config else 0.1
        self._initial_epsilon = self._epsilon
        self._epsilon_decay = extra.get('epsilon_decay_rate', 0.995)
        self._epsilon_min = config.min_epsilon if config else 0.01
        
        # Boltzmann探索参数
        self._temperature = extra.get('temperature', 1.0)
        self._temperature_decay = extra.get('temperature_decay', 0.995)
        self._temperature_min = extra.get('temperature_min', 0.1)
        
        # UCB探索参数
        self._ucb_c = extra.get('ucb_c', 2.0)
        
        # 学习参数
        self._learning_rate = config.learning_rate if config else 0.01
        self._discount_factor = config.discount_factor if config else 0.99
        
        # N-step TD参数
        self._n_steps = extra.get('n_steps', 3)
        self._n_step_buffer: List[Transition] = []
        
        # 经验回放
        self._use_replay_buffer = extra.get('use_replay_buffer', False)
        self._replay_buffer = ReplayBuffer(
            capacity=extra.get('replay_buffer_size', 10000),
            prioritized=extra.get('prioritized_replay', False)
        )
        self._batch_size = extra.get('batch_size', 32)
        self._min_replay_size = extra.get('min_replay_size', 100)
        
        # 目标网络更新
        self._use_target_network = extra.get('use_target_network', False)
        self._target_update_freq = extra.get('target_update_freq', 100)
        self._update_counter = 0
        
        # 奖励处理
        reward_shaping = extra.get('reward_shaping', 'none')
        try:
            shaping = RewardShaping(reward_shaping)
        except ValueError:
            shaping = RewardShaping.NONE
        self._reward_processor = RewardProcessor(
            shaping=shaping,
            clip_range=tuple(extra.get('reward_clip_range', [-10.0, 10.0])),
            scale=extra.get('reward_scale', 1.0)
        )
        
        # 状态归一化
        self._normalize_states = extra.get('normalize_states', False)
        self._state_normalizer = StateNormalizer()
        
        # 当前状态
        self._current_state: Optional[State] = None
        self._previous_action: Optional[Dict[str, Any]] = None
        
        # 统计信息
        self._episode_rewards: List[float] = []
        self._current_episode_reward = 0.0
        self._episode_count = 0
        self._total_steps = 0
        self._td_errors: List[float] = []
        self._selection_history: List[Dict[str, Any]] = []
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.REINFORCEMENT_LEARNING
    
    @property
    def q_table(self) -> QTable:
        """获取Q表"""
        return self._q_table
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化强化学习"""
        super().initialize(context)
        
        # 初始化动作空间
        self._initialize_action_space(context)
        
        # 设置初始状态
        self._current_state = self._context_to_state(context)
        
        # 初始化Double Q-Learning的第二个Q表
        if self._method == RLMethod.DOUBLE_Q_LEARNING:
            self._q_table_b = QTable()
        
        # 初始化目标网络
        if self._use_target_network:
            self._target_q_table = self._q_table.copy()
        
        # 从历史学习
        self._learn_from_history(context)
        
        self.logger.info(
            f"RL initialized: method={self._method.value}, "
            f"exploration={self._exploration.value}, "
            f"action_space_size={len(self._action_space)}"
        )
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成动作建议"""
        import time
        start_time = time.time()
        
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        self._total_steps += 1
        
        # 更新当前状态
        raw_state = self._context_to_state(context)
        if self._normalize_states:
            self._current_state = self._state_normalizer.normalize(raw_state)
        else:
            self._current_state = raw_state
        
        # 根据探索策略选择动作
        action, selection_info = self._select_action(self._current_state)
        
        # 记录选择历史
        self._selection_history.append({
            'iteration': self._iteration_count,
            'state_id': self._current_state.state_id,
            'action': action,
            'exploration': selection_info.get('exploration_type', 'exploit'),
            'timestamp': datetime.now().isoformat()
        })
        
        # 记录前一个动作（用于SARSA）
        self._previous_action = action
        
        execution_time = (time.time() - start_time) * 1000
        
        reasoning_steps = [
            f"强化学习方法: {self._method.value}",
            f"探索策略: {self._exploration.value}",
            f"当前状态ID: {self._current_state.state_id}",
            f"动作空间大小: {len(self._action_space)}",
            f"探索率 ε: {self._epsilon:.4f}",
            f"总步数: {self._total_steps}",
            f"回合数: {self._episode_count}",
            f"执行时间: {execution_time:.2f}ms",
            *selection_info.get('steps', [])
        ]
        
        # 获取备选动作
        alternatives = self._get_alternative_actions(self._current_state, exclude=action)
        
        return self._build_result(
            action=action,
            confidence=selection_info.get('confidence', 0.7),
            reasoning=selection_info.get('reasoning', ''),
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            debug_info={
                'method': self._method.value,
                'exploration': self._exploration.value,
                'state_id': self._current_state.state_id,
                'epsilon': self._epsilon,
                'temperature': self._temperature,
                'action_q_values': self._q_table.get_all_actions(self._current_state.state_id),
                'total_steps': self._total_steps,
                'episode_count': self._episode_count,
                'replay_buffer_size': len(self._replay_buffer),
                'execution_time_ms': execution_time
            }
        )
    
    def update(
        self, 
        action: Dict[str, Any], 
        reward: float,
        context: Optional[AlgorithmContext] = None,
        done: bool = False
    ) -> None:
        """更新Q值
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            context: 下一个上下文状态
            done: 是否为回合结束
        """
        if self._current_state is None:
            return
        
        state = self._current_state
        action_id = self._action_to_id(action)
        
        # 处理奖励
        processed_reward = self._reward_processor.process(reward)
        
        # 确定下一状态
        if context:
            raw_next_state = self._context_to_state(context)
            if self._normalize_states:
                next_state = self._state_normalizer.normalize(raw_next_state)
            else:
                next_state = raw_next_state
        else:
            next_state = state
        
        # 创建转换
        transition = Transition(
            state=state,
            action=action,
            reward=processed_reward,
            next_state=next_state,
            done=done,
            action_id=action_id
        )
        
        # 添加到经验回放缓冲区
        if self._use_replay_buffer:
            self._replay_buffer.push(transition)
        
        # 累计回合奖励
        self._current_episode_reward += reward
        
        # 根据方法执行更新
        td_error = self._perform_update(state, action_id, processed_reward, next_state, done)
        self._td_errors.append(td_error)
        
        # 经验回放学习
        if self._use_replay_buffer and len(self._replay_buffer) >= self._min_replay_size:
            self._replay_learn()
        
        # 更新目标网络
        if self._use_target_network:
            self._update_counter += 1
            if self._update_counter % self._target_update_freq == 0:
                self._update_target_network()
        
        # 记录观测
        self._record_observation(action, reward)
        
        # 衰减探索参数
        self._decay_exploration_params()
        
        # 更新当前状态
        self._current_state = next_state
        
        # 处理回合结束
        if done:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_count += 1
            self._current_episode_reward = 0.0
            self._n_step_buffer.clear()
        
        self.logger.debug(
            f"Updated: state={state.state_id}, action={action_id}, "
            f"reward={reward:.4f}, td_error={td_error:.4f}, ε={self._epsilon:.4f}"
        )
    
    def _perform_update(
        self, 
        state: State, 
        action_id: str, 
        reward: float, 
        next_state: State,
        done: bool
    ) -> float:
        """执行Q值更新，返回TD误差"""
        
        if self._method == RLMethod.Q_LEARNING:
            return self._q_learning_update(state, action_id, reward, next_state, done)
        
        elif self._method == RLMethod.DOUBLE_Q_LEARNING:
            return self._double_q_learning_update(state, action_id, reward, next_state, done)
        
        elif self._method == RLMethod.SARSA:
            return self._sarsa_update(state, action_id, reward, next_state, done)
        
        elif self._method == RLMethod.EXPECTED_SARSA:
            return self._expected_sarsa_update(state, action_id, reward, next_state, done)
        
        elif self._method == RLMethod.N_STEP_TD:
            return self._n_step_td_update(state, action_id, reward, next_state, done)
        
        else:
            return self._q_learning_update(state, action_id, reward, next_state, done)
    
    def _q_learning_update(
        self, 
        state: State, 
        action_id: str,
        reward: float, 
        next_state: State,
        done: bool = False
    ) -> float:
        """Q-Learning更新
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        """
        current_q = self._q_table.get(state.state_id, action_id)
        
        if done:
            td_target = reward
        else:
            # 使用目标网络（如果有）
            if self._target_q_table:
                max_next_q, _ = self._target_q_table.get_max(next_state.state_id)
            else:
                max_next_q, _ = self._q_table.get_max(next_state.state_id)
            td_target = reward + self._discount_factor * max_next_q
        
        td_error = td_target - current_q
        new_q = current_q + self._learning_rate * td_error
        
        self._q_table.set(state.state_id, action_id, new_q)
        return td_error
    
    def _double_q_learning_update(
        self, 
        state: State, 
        action_id: str,
        reward: float, 
        next_state: State,
        done: bool = False
    ) -> float:
        """Double Q-Learning更新
        
        随机选择更新Q1或Q2，使用另一个Q表选择动作
        """
        if self._q_table_b is None:
            self._q_table_b = QTable()
        
        # 随机选择更新哪个Q表
        if random.random() < 0.5:
            # 更新Q1，使用Q1选择动作，Q2评估
            current_q = self._q_table.get(state.state_id, action_id)
            
            if done:
                td_target = reward
            else:
                _, best_action = self._q_table.get_max(next_state.state_id)
                if best_action:
                    next_q = self._q_table_b.get(next_state.state_id, best_action)
                else:
                    next_q = 0.0
                td_target = reward + self._discount_factor * next_q
            
            td_error = td_target - current_q
            new_q = current_q + self._learning_rate * td_error
            self._q_table.set(state.state_id, action_id, new_q)
        else:
            # 更新Q2，使用Q2选择动作，Q1评估
            current_q = self._q_table_b.get(state.state_id, action_id)
            
            if done:
                td_target = reward
            else:
                _, best_action = self._q_table_b.get_max(next_state.state_id)
                if best_action:
                    next_q = self._q_table.get(next_state.state_id, best_action)
                else:
                    next_q = 0.0
                td_target = reward + self._discount_factor * next_q
            
            td_error = td_target - current_q
            new_q = current_q + self._learning_rate * td_error
            self._q_table_b.set(state.state_id, action_id, new_q)
        
        return td_error
    
    def _sarsa_update(
        self, 
        state: State, 
        action_id: str,
        reward: float, 
        next_state: State,
        done: bool = False
    ) -> float:
        """SARSA更新
        
        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        """
        current_q = self._q_table.get(state.state_id, action_id)
        
        if done:
            td_target = reward
        else:
            # 使用当前策略选择下一个动作
            if self._previous_action:
                next_action_id = self._action_to_id(self._previous_action)
            else:
                next_action, _ = self._select_action(next_state)
                next_action_id = self._action_to_id(next_action)
            
            next_q = self._q_table.get(next_state.state_id, next_action_id)
            td_target = reward + self._discount_factor * next_q
        
        td_error = td_target - current_q
        new_q = current_q + self._learning_rate * td_error
        
        self._q_table.set(state.state_id, action_id, new_q)
        return td_error
    
    def _expected_sarsa_update(
        self, 
        state: State, 
        action_id: str,
        reward: float, 
        next_state: State,
        done: bool = False
    ) -> float:
        """Expected SARSA更新
        
        Q(s,a) ← Q(s,a) + α[r + γ·E[Q(s',a')] - Q(s,a)]
        """
        current_q = self._q_table.get(state.state_id, action_id)
        
        if done:
            td_target = reward
        else:
            # 计算下一状态的期望Q值
            expected_q = self._compute_expected_q(next_state)
            td_target = reward + self._discount_factor * expected_q
        
        td_error = td_target - current_q
        new_q = current_q + self._learning_rate * td_error
        
        self._q_table.set(state.state_id, action_id, new_q)
        return td_error
    
    def _compute_expected_q(self, state: State) -> float:
        """计算状态的期望Q值"""
        q_values = self._q_table.get_all_actions(state.state_id)
        
        if not q_values:
            return 0.0
        
        # 根据当前策略计算期望
        if self._exploration == ExplorationStrategy.BOLTZMANN:
            probs = self._q_table.get_action_probabilities(state.state_id, self._temperature)
            return sum(q * probs.get(a, 0) for a, q in q_values.items())
        else:
            # ε-greedy的期望
            max_q = max(q_values.values())
            n_actions = len(self._action_space)
            
            if n_actions == 0:
                return max_q
            
            # (1-ε)·max_q + ε·avg_q
            avg_q = sum(q_values.values()) / len(q_values)
            return (1 - self._epsilon) * max_q + self._epsilon * avg_q
    
    def _n_step_td_update(
        self, 
        state: State, 
        action_id: str,
        reward: float, 
        next_state: State,
        done: bool = False
    ) -> float:
        """N-step TD更新"""
        # 添加到N-step缓冲区
        transition = Transition(
            state=state,
            action={'_id': action_id},
            reward=reward,
            next_state=next_state,
            done=done,
            action_id=action_id
        )
        self._n_step_buffer.append(transition)
        
        # 如果缓冲区未满且未结束，等待
        if len(self._n_step_buffer) < self._n_steps and not done:
            return 0.0
        
        # 计算N-step回报
        n_step_return = 0.0
        gamma = 1.0
        
        for i, trans in enumerate(self._n_step_buffer):
            n_step_return += gamma * trans.reward
            gamma *= self._discount_factor
        
        # 如果不是终止状态，加上bootstrap
        if not done and self._n_step_buffer:
            last_state = self._n_step_buffer[-1].next_state
            if last_state:
                max_next_q, _ = self._q_table.get_max(last_state.state_id)
                n_step_return += gamma * max_next_q
        
        # 更新第一个转换的Q值
        first_trans = self._n_step_buffer[0]
        current_q = self._q_table.get(first_trans.state.state_id, first_trans.action_id)
        td_error = n_step_return - current_q
        new_q = current_q + self._learning_rate * td_error
        
        self._q_table.set(first_trans.state.state_id, first_trans.action_id, new_q)
        
        # 移除第一个转换
        self._n_step_buffer.pop(0)
        
        return td_error
    
    def _replay_learn(self) -> None:
        """从经验回放缓冲区学习"""
        transitions, weights = self._replay_buffer.sample(self._batch_size)
        
        td_errors = []
        for i, trans in enumerate(transitions):
            td_error = self._perform_update(
                trans.state,
                trans.action_id,
                trans.reward,
                trans.next_state,
                trans.done
            )
            
            # 如果使用优先回放，应用重要性权重
            if weights is not None:
                # 调整学习率
                adjusted_lr = self._learning_rate * weights[i]
                current_q = self._q_table.get(trans.state.state_id, trans.action_id)
                # 重新计算
                new_q = current_q + adjusted_lr * td_error - self._learning_rate * td_error
                self._q_table.set(trans.state.state_id, trans.action_id, new_q)
            
            td_errors.append(td_error)
        
        # 更新优先级
        if self._replay_buffer._prioritized:
            indices = list(range(len(transitions)))
            self._replay_buffer.update_priorities(indices, td_errors)
    
    def _update_target_network(self) -> None:
        """更新目标网络"""
        self._target_q_table = self._q_table.copy()
        self.logger.debug("Target network updated")
    
    def _initialize_action_space(self, context: AlgorithmContext) -> None:
        """初始化动作空间
        
        Args:
            context: 算法上下文
        """
        self._action_space = []
        
        # 从搜索空间生成动作
        if context.search_space:
            self._generate_actions_from_space(context.search_space)
        
        # 从历史动作补充
        for obs in context.observations:
            action = obs.get('action', {})
            if action and action not in self._action_space:
                self._action_space.append(action)
        
        # 确保至少有一个动作
        if not self._action_space:
            self._action_space.append({})
    
    def _generate_actions_from_space(self, search_space: Dict[str, Any]) -> None:
        """从搜索空间生成动作
        
        Args:
            search_space: 搜索空间定义
        """
        from itertools import product
        
        param_options = {}
        for param_name, param_def in search_space.items():
            if isinstance(param_def, dict):
                if 'choices' in param_def:
                    param_options[param_name] = param_def['choices']
                elif 'low' in param_def and 'high' in param_def:
                    # 离散化
                    low, high = param_def['low'], param_def['high']
                    n_bins = min(param_def.get('n_bins', 3), 5)
                    step = (high - low) / n_bins
                    param_options[param_name] = [low + i * step for i in range(n_bins + 1)]
            elif isinstance(param_def, list):
                param_options[param_name] = param_def
        
        if not param_options:
            return
        
        param_names = list(param_options.keys())
        param_values = list(param_options.values())
        
        # 限制组合数量
        max_actions = 50
        total = 1
        for v in param_values:
            total *= len(v)
        
        if total <= max_actions:
            for values in product(*param_values):
                self._action_space.append(dict(zip(param_names, values)))
        else:
            # 随机采样
            for _ in range(max_actions):
                action = {name: random.choice(values) 
                         for name, values in zip(param_names, param_values)}
                if action not in self._action_space:
                    self._action_space.append(action)
    
    def _context_to_state(self, context: AlgorithmContext) -> State:
        """将上下文转换为状态
        
        Args:
            context: 算法上下文
            
        Returns:
            状态对象
        """
        # 提取状态特征
        features = {}
        
        # 从inputs提取
        for key, value in context.inputs.items():
            if isinstance(value, (int, float, str, bool)):
                features[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool)):
                        features[f"{key}_{sub_key}"] = sub_value
        
        # 添加约束信息
        for key, value in context.constraints.items():
            if isinstance(value, (int, float, str, bool)):
                features[f"constraint_{key}"] = value
        
        return State(features=features)
    
    def _action_to_id(self, action: Dict[str, Any]) -> str:
        """将动作转换为ID
        
        Args:
            action: 动作字典
            
        Returns:
            动作ID
        """
        sorted_items = sorted(action.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def _select_action(self, state: State) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """选择动作
        
        根据探索策略选择动作。
        """
        if self._exploration == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_select(state)
        
        elif self._exploration == ExplorationStrategy.EPSILON_DECAY:
            return self._epsilon_greedy_select(state)
        
        elif self._exploration == ExplorationStrategy.BOLTZMANN:
            return self._boltzmann_select(state)
        
        elif self._exploration == ExplorationStrategy.UCB:
            return self._ucb_select(state)
        
        elif self._exploration == ExplorationStrategy.NOISY:
            return self._noisy_select(state)
        
        else:
            return self._epsilon_greedy_select(state)
    
    def _epsilon_greedy_select(self, state: State) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ε-贪婪选择"""
        if random.random() < self._epsilon:
            # 探索
            action = random.choice(self._action_space)
            return action, {
                'reasoning': f"探索：随机选择动作（ε={self._epsilon:.4f}）",
                'steps': ["ε-贪婪触发探索", f"从{len(self._action_space)}个动作中随机选择"],
                'confidence': 0.5,
                'exploration_type': 'explore'
            }
        else:
            # 利用
            best_action, best_q = self._get_best_action(state)
            
            return best_action, {
                'reasoning': f"利用：选择Q值最高的动作（Q={best_q:.4f}）",
                'steps': [
                    f"评估{len(self._action_space)}个动作的Q值",
                    f"最高Q值: {best_q:.4f}"
                ],
                'confidence': min(0.95, 0.7 + len(self._observations) * 0.005),
                'exploration_type': 'exploit'
            }
    
    def _boltzmann_select(self, state: State) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Boltzmann（Softmax）探索"""
        probs = self._q_table.get_action_probabilities(state.state_id, self._temperature)
        
        if not probs:
            action = random.choice(self._action_space)
            return action, {
                'reasoning': f"Boltzmann探索：随机选择（无Q值记录）",
                'steps': ["首次访问状态，随机选择"],
                'confidence': 0.5,
                'exploration_type': 'explore'
            }
        
        # 按概率选择
        action_ids = list(probs.keys())
        probabilities = [probs[a] for a in action_ids]
        
        selected_idx = random.choices(range(len(action_ids)), weights=probabilities)[0]
        selected_action_id = action_ids[selected_idx]
        selected_prob = probabilities[selected_idx]
        
        # 找到对应的动作
        action = self._action_id_map.get(selected_action_id, self._action_space[0])
        
        return action, {
            'reasoning': f"Boltzmann选择：温度={self._temperature:.4f}, 概率={selected_prob:.4f}",
            'steps': [
                f"温度参数 τ={self._temperature:.4f}",
                f"选择概率={selected_prob:.4f}"
            ],
            'confidence': selected_prob,
            'exploration_type': 'boltzmann'
        }
    
    def _ucb_select(self, state: State) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """UCB探索"""
        best_action = None
        best_ucb = float('-inf')
        
        for action in self._action_space:
            action_id = self._action_to_id(action)
            ucb_value = self._q_table.get_ucb_value(
                state.state_id, 
                action_id, 
                self._total_steps,
                self._ucb_c
            )
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_action = action
        
        if best_action is None:
            best_action = random.choice(self._action_space)
        
        q_value = self._q_table.get(state.state_id, self._action_to_id(best_action))
        
        return best_action, {
            'reasoning': f"UCB选择：UCB值={best_ucb:.4f}, Q值={q_value:.4f}",
            'steps': [
                f"UCB参数 c={self._ucb_c}",
                f"总步数={self._total_steps}",
                f"UCB值={best_ucb:.4f}"
            ],
            'confidence': min(0.95, 0.6 + self._total_steps * 0.001),
            'exploration_type': 'ucb'
        }
    
    def _noisy_select(self, state: State) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """噪声探索"""
        best_action = None
        best_noisy_q = float('-inf')
        
        # 噪声强度随时间衰减
        noise_std = max(0.1, 1.0 / math.sqrt(1 + self._total_steps * 0.01))
        
        for action in self._action_space:
            action_id = self._action_to_id(action)
            q_value = self._q_table.get(state.state_id, action_id)
            
            # 添加噪声
            noise = random.gauss(0, noise_std)
            noisy_q = q_value + noise
            
            if noisy_q > best_noisy_q:
                best_noisy_q = noisy_q
                best_action = action
        
        if best_action is None:
            best_action = random.choice(self._action_space)
        
        q_value = self._q_table.get(state.state_id, self._action_to_id(best_action))
        
        return best_action, {
            'reasoning': f"噪声探索：噪声标准差={noise_std:.4f}",
            'steps': [
                f"噪声标准差={noise_std:.4f}",
                f"原始Q值={q_value:.4f}",
                f"噪声Q值={best_noisy_q:.4f}"
            ],
            'confidence': min(0.95, 0.6 + self._total_steps * 0.001),
            'exploration_type': 'noisy'
        }
    
    def _get_best_action(self, state: State) -> Tuple[Dict[str, Any], float]:
        """获取最佳动作"""
        best_action = None
        best_q = float('-inf')
        
        # 对于Double Q-Learning，使用两个Q表的平均
        for action in self._action_space:
            action_id = self._action_to_id(action)
            
            if self._method == RLMethod.DOUBLE_Q_LEARNING and self._q_table_b:
                q1 = self._q_table.get(state.state_id, action_id)
                q2 = self._q_table_b.get(state.state_id, action_id)
                q_value = (q1 + q2) / 2
            else:
                q_value = self._q_table.get(state.state_id, action_id)
            
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        if best_action is None:
            best_action = random.choice(self._action_space)
            best_q = 0.0
        
        return best_action, best_q
    
    def _decay_exploration_params(self) -> None:
        """衰减探索参数"""
        if self._exploration in [ExplorationStrategy.EPSILON_DECAY, ExplorationStrategy.EPSILON_GREEDY]:
            self._epsilon = max(
                self._epsilon_min,
                self._epsilon * self._epsilon_decay
            )
        
        if self._exploration == ExplorationStrategy.BOLTZMANN:
            self._temperature = max(
                self._temperature_min,
                self._temperature * self._temperature_decay
            )
    
    def _get_alternative_actions(self, state: State, 
                                exclude: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取备选动作
        
        Args:
            state: 当前状态
            exclude: 要排除的动作
            
        Returns:
            备选动作列表
        """
        alternatives = []
        
        action_q_values = []
        for action in self._action_space:
            if exclude and action == exclude:
                continue
            action_id = self._action_to_id(action)
            q_value = self._q_table.get(state.state_id, action_id)
            action_q_values.append((action, q_value))
        
        # 按Q值排序
        action_q_values.sort(key=lambda x: x[1], reverse=True)
        
        for action, q_value in action_q_values[:2]:
            alternatives.append({
                'action': action,
                'q_value': q_value,
                'confidence': min(0.9, 0.5 + q_value * 0.1),
                'reasoning': f"Q值={q_value:.4f}"
            })
        
        return alternatives
    
    def _learn_from_history(self, context: AlgorithmContext) -> None:
        """从历史数据学习"""
        if len(context.observations) < 2:
            return
        
        observations = list(context.observations)
        
        for i in range(len(observations) - 1):
            current_obs = observations[i]
            next_obs = observations[i + 1]
            
            # 构建状态
            state = State(features=current_obs.get('state', current_obs.get('inputs', {})))
            action = current_obs.get('action', {})
            reward = current_obs.get('reward', 0.0)
            next_state = State(features=next_obs.get('state', next_obs.get('inputs', {})))
            done = current_obs.get('done', False)
            
            # 更新Q值
            action_id = self._action_to_id(action)
            self._perform_update(state, action_id, reward, next_state, done)
            
            # 添加到经验回放
            if self._use_replay_buffer:
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    action_id=action_id
                )
                self._replay_buffer.push(transition)
        
        self.logger.info(f"Learned from {len(observations)} historical observations")
    
    # ==================== 状态管理 ====================
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'q_table': self._q_table.to_dict(),
            'q_table_b': self._q_table_b.to_dict() if self._q_table_b else None,
            'action_space': self._action_space,
            'epsilon': self._epsilon,
            'temperature': self._temperature,
            'total_steps': self._total_steps,
            'episode_count': self._episode_count,
            'episode_rewards': self._episode_rewards[-100:],
            'td_errors': self._td_errors[-100:],
            'observations': self._observations[-100:],
            'method': self._method.value,
            'exploration': self._exploration.value
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        if 'q_table' in state:
            self._q_table = QTable.from_dict(state['q_table'])
        
        if state.get('q_table_b'):
            self._q_table_b = QTable.from_dict(state['q_table_b'])
        
        self._action_space = state.get('action_space', [])
        self._epsilon = state.get('epsilon', self._initial_epsilon)
        self._temperature = state.get('temperature', 1.0)
        self._total_steps = state.get('total_steps', 0)
        self._episode_count = state.get('episode_count', 0)
        self._episode_rewards = state.get('episode_rewards', [])
        self._td_errors = state.get('td_errors', [])
        self._observations = state.get('observations', [])
        
        # 重建动作ID映射
        for action in self._action_space:
            action_id = self._action_to_id(action)
            self._action_id_map[action_id] = action
        
        self._initialized = True
        self.logger.info(
            f"State loaded: {len(self._action_space)} actions, "
            f"{self._total_steps} steps, {self._episode_count} episodes"
        )
    
    def warm_start(
        self, 
        observations: List[Dict[str, Any]],
        q_values: Dict[str, Dict[str, float]] = None
    ) -> None:
        """热启动
        
        Args:
            observations: 历史观测 [{'state': {...}, 'action': {...}, 'reward': float}, ...]
            q_values: 可选的预设Q值 {'state_id': {'action_id': q_value}}
        """
        # 设置预设Q值
        if q_values:
            for state_id, actions in q_values.items():
                for action_id, q in actions.items():
                    self._q_table.set(state_id, action_id, q)
        
        # 从观测学习
        for i in range(len(observations) - 1):
            obs = observations[i]
            next_obs = observations[i + 1]
            
            state = State(features=obs.get('state', {}))
            action = obs.get('action', {})
            reward = obs.get('reward', 0.0)
            next_state = State(features=next_obs.get('state', {}))
            done = obs.get('done', False)
            
            action_id = self._action_to_id(action)
            self._perform_update(state, action_id, reward, next_state, done)
            
            # 添加动作到空间
            if action not in self._action_space:
                self._action_space.append(action)
                self._action_id_map[action_id] = action
        
        self._initialized = True
        self.logger.info(f"Warm start completed: {len(observations)} observations")
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        data = {
            'q_table_stats': self._q_table.get_statistics(),
            'episode_rewards': self._episode_rewards,
            'td_errors': self._td_errors[-500:],
            'selection_history': self._selection_history[-100:],
            'learning_curve': [],
            'exploration_curve': [],
            'statistics': {}
        }
        
        # 学习曲线（滑动平均）
        window_size = 10
        for i in range(0, len(self._episode_rewards), window_size):
            window = self._episode_rewards[i:i+window_size]
            if window:
                data['learning_curve'].append({
                    'episode': i + window_size // 2,
                    'avg_reward': sum(window) / len(window)
                })
        
        # 探索参数曲线
        data['exploration_curve'] = [
            {'step': self._total_steps, 'epsilon': self._epsilon, 'temperature': self._temperature}
        ]
        
        # 统计信息
        data['statistics'] = {
            'method': self._method.value,
            'exploration': self._exploration.value,
            'total_steps': self._total_steps,
            'episode_count': self._episode_count,
            'current_epsilon': self._epsilon,
            'current_temperature': self._temperature,
            'action_space_size': len(self._action_space),
            'replay_buffer_size': len(self._replay_buffer),
            'q_table_states': len(self._q_table._table)
        }
        
        return data
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._episode_rewards:
            return {'status': 'no_episodes'}
        
        recent_rewards = self._episode_rewards[-50:] if len(self._episode_rewards) >= 50 else self._episode_rewards
        
        return {
            'status': 'active',
            'method': self._method.value,
            'exploration': self._exploration.value,
            'total_steps': self._total_steps,
            'episode_count': self._episode_count,
            'total_reward': sum(self._episode_rewards),
            'avg_reward': sum(self._episode_rewards) / len(self._episode_rewards),
            'recent_avg_reward': sum(recent_rewards) / len(recent_rewards),
            'max_reward': max(self._episode_rewards),
            'min_reward': min(self._episode_rewards),
            'reward_std': (sum((r - sum(self._episode_rewards)/len(self._episode_rewards))**2 
                              for r in self._episode_rewards) / len(self._episode_rewards)) ** 0.5,
            'exploration_params': {
                'epsilon': self._epsilon,
                'temperature': self._temperature,
                'ucb_c': self._ucb_c
            },
            'learning_params': {
                'learning_rate': self._learning_rate,
                'discount_factor': self._discount_factor,
                'batch_size': self._batch_size
            },
            'q_table_stats': self._q_table.get_statistics(),
            'replay_buffer_size': len(self._replay_buffer),
            'avg_td_error': sum(self._td_errors[-100:]) / len(self._td_errors[-100:]) if self._td_errors else 0
        }
    
    def get_policy(self) -> Dict[str, Dict[str, float]]:
        """获取当前策略（状态到动作概率的映射）"""
        policy = {}
        
        for state_id in self._q_table._table:
            probs = self._q_table.get_action_probabilities(state_id, self._temperature)
            policy[state_id] = probs
        
        return policy
    
    def get_value_function(self) -> Dict[str, float]:
        """获取状态值函数"""
        value_function = {}
        
        for state_id in self._q_table._table:
            max_q, _ = self._q_table.get_max(state_id)
            value_function[state_id] = max_q
        
        return value_function


class PolicyGradient(BaseAlgorithm):
    """策略梯度算法 (REINFORCE)
    
    直接学习策略参数，适用于连续动作空间。
    
    特性：
    - 基线减法（方差减少）
    - 熵正则化
    - 自适应学习率
    - 梯度裁剪
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        extra = config.extra if config and config.extra else {}
        
        self._policy_params: Dict[str, float] = {}
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        self._episode_log: List[Tuple[Dict, float]] = []
        
        # 学习参数
        self._learning_rate = config.learning_rate if config else 0.01
        self._discount_factor = config.discount_factor if config else 0.99
        
        # 基线（用于方差减少）
        self._use_baseline = extra.get('use_baseline', True)
        self._baseline = 0.0
        self._baseline_alpha = extra.get('baseline_alpha', 0.1)
        
        # 熵正则化
        self._entropy_coef = extra.get('entropy_coef', 0.01)
        
        # 噪声衰减
        self._initial_noise_std = extra.get('initial_noise_std', 0.5)
        self._noise_decay = extra.get('noise_decay', 0.995)
        self._min_noise_std = extra.get('min_noise_std', 0.05)
        self._current_noise_std = self._initial_noise_std
        
        # 梯度裁剪
        self._grad_clip = extra.get('grad_clip', 1.0)
        
        # 批量更新大小
        self._update_batch_size = extra.get('update_batch_size', 10)
        
        # 统计
        self._episode_rewards: List[float] = []
        self._policy_history: List[Dict[str, float]] = []
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.REINFORCEMENT_LEARNING
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化策略梯度"""
        super().initialize(context)
        
        # 解析搜索空间
        for param_name, param_def in context.search_space.items():
            if isinstance(param_def, dict):
                low = param_def.get('low', 0.0)
                high = param_def.get('high', 1.0)
            elif isinstance(param_def, (list, tuple)) and len(param_def) >= 2:
                low, high = param_def[0], param_def[1]
            else:
                low, high = 0.0, 1.0
            
            self._param_bounds[param_name] = (float(low), float(high))
            # 初始化策略参数为中间值
            self._policy_params[param_name] = (low + high) / 2
        
        self.logger.info(f"Policy gradient initialized: {len(self._policy_params)} parameters")
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成动作建议"""
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        
        # 从策略采样动作（添加高斯噪声）
        action = {}
        
        for param_name, mean_value in self._policy_params.items():
            low, high = self._param_bounds.get(param_name, (0.0, 1.0))
            noise = random.gauss(0, self._current_noise_std * (high - low))
            value = mean_value + noise
            value = max(low, min(high, value))
            action[param_name] = value
        
        return self._build_result(
            action=action,
            confidence=min(0.9, 0.6 + self._iteration_count * 0.01),
            reasoning=f"策略梯度采样：噪声标准差={self._current_noise_std:.4f}",
            alternatives=[],
            reasoning_steps=[
                f"当前策略参数: {self._policy_params}",
                f"噪声标准差: {self._current_noise_std:.4f}",
                f"基线值: {self._baseline:.4f}",
                f"采样动作: {action}"
            ],
            debug_info={
                'policy_params': self._policy_params.copy(),
                'noise_std': self._current_noise_std,
                'baseline': self._baseline
            }
        )
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None,
               done: bool = False) -> None:
        """更新策略参数"""
        self._episode_log.append((action, reward))
        self._record_observation(action, reward)
        
        # 更新基线
        if self._use_baseline:
            self._baseline = self._baseline + self._baseline_alpha * (reward - self._baseline)
        
        # 批量更新
        if len(self._episode_log) >= self._update_batch_size or done:
            self._update_policy()
            
            # 记录回合奖励
            episode_reward = sum(r for _, r in self._episode_log)
            self._episode_rewards.append(episode_reward)
            
            self._episode_log = []
        
        # 衰减噪声
        self._current_noise_std = max(
            self._min_noise_std,
            self._current_noise_std * self._noise_decay
        )
    
    def _update_policy(self) -> None:
        """更新策略参数"""
        if not self._episode_log:
            return
        
        # 计算回报
        returns = []
        G = 0
        for _, reward in reversed(self._episode_log):
            G = reward + self._discount_factor * G
            returns.insert(0, G)
        
        # 减去基线
        if self._use_baseline:
            advantages = [G - self._baseline for G in returns]
        else:
            advantages = returns
        
        # 归一化优势
        mean_adv = sum(advantages) / len(advantages)
        std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
        std_adv = max(std_adv, 1e-8)
        normalized_advantages = [(a - mean_adv) / std_adv for a in advantages]
        
        # 策略梯度更新
        for (action, _), advantage in zip(self._episode_log, normalized_advantages):
            for param_name, value in action.items():
                if param_name in self._policy_params:
                    low, high = self._param_bounds.get(param_name, (0.0, 1.0))
                    
                    # 计算梯度
                    diff = value - self._policy_params[param_name]
                    gradient = diff * advantage / (self._current_noise_std * (high - low) + 1e-8)
                    
                    # 梯度裁剪
                    gradient = max(-self._grad_clip, min(self._grad_clip, gradient))
                    
                    # 更新参数
                    self._policy_params[param_name] += self._learning_rate * gradient
                    
                    # 裁剪到边界
                    self._policy_params[param_name] = max(low, min(high, self._policy_params[param_name]))
        
        # 记录策略历史
        self._policy_history.append(self._policy_params.copy())
        
        self.logger.debug(f"Policy updated: {self._policy_params}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'policy_params': self._policy_params.copy(),
            'param_bounds': self._param_bounds.copy(),
            'baseline': self._baseline,
            'noise_std': self._current_noise_std,
            'episode_rewards': self._episode_rewards[-100:],
            'iteration_count': self._iteration_count
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """加载状态"""
        self._policy_params = state.get('policy_params', {})
        self._param_bounds = state.get('param_bounds', {})
        self._baseline = state.get('baseline', 0.0)
        self._current_noise_std = state.get('noise_std', self._initial_noise_std)
        self._episode_rewards = state.get('episode_rewards', [])
        self._iteration_count = state.get('iteration_count', 0)
        self._initialized = True
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._episode_rewards:
            return {'status': 'no_episodes'}
        
        return {
            'status': 'active',
            'iteration_count': self._iteration_count,
            'episode_count': len(self._episode_rewards),
            'current_policy': self._policy_params,
            'avg_reward': sum(self._episode_rewards) / len(self._episode_rewards),
            'max_reward': max(self._episode_rewards),
            'recent_avg': sum(self._episode_rewards[-10:]) / min(10, len(self._episode_rewards)),
            'baseline': self._baseline,
            'noise_std': self._current_noise_std
        }


class ActorCritic(BaseAlgorithm):
    """Actor-Critic算法
    
    结合策略梯度（Actor）和价值函数（Critic）
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        extra = config.extra if config and config.extra else {}
        
        # Actor（策略）参数
        self._actor_params: Dict[str, float] = {}
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Critic（价值函数）
        self._critic = QTable()
        
        # 学习率
        self._actor_lr = extra.get('actor_lr', 0.01)
        self._critic_lr = extra.get('critic_lr', 0.1)
        self._discount_factor = config.discount_factor if config else 0.99
        
        # 噪声参数
        self._noise_std = extra.get('noise_std', 0.3)
        self._noise_decay = extra.get('noise_decay', 0.995)
        self._min_noise = extra.get('min_noise', 0.05)
        
        # 状态
        self._current_state: Optional[State] = None
        self._episode_rewards: List[float] = []
        self._current_episode_reward = 0.0
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.REINFORCEMENT_LEARNING
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化"""
        super().initialize(context)
        
        for param_name, param_def in context.search_space.items():
            if isinstance(param_def, dict):
                low = param_def.get('low', 0.0)
                high = param_def.get('high', 1.0)
            else:
                low, high = 0.0, 1.0
            
            self._param_bounds[param_name] = (float(low), float(high))
            self._actor_params[param_name] = (low + high) / 2
        
        self._current_state = self._context_to_state(context)
    
    def _context_to_state(self, context: AlgorithmContext) -> State:
        """上下文转状态"""
        features = {}
        for key, value in context.inputs.items():
            if isinstance(value, (int, float, str, bool)):
                features[key] = value
        return State(features=features)
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成建议"""
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        self._current_state = self._context_to_state(context)
        
        # 从Actor采样动作
        action = {}
        for param_name, mean_value in self._actor_params.items():
            low, high = self._param_bounds.get(param_name, (0.0, 1.0))
            noise = random.gauss(0, self._noise_std * (high - low))
            value = max(low, min(high, mean_value + noise))
            action[param_name] = value
        
        # 获取Critic估计
        action_id = "|".join(f"{k}={v:.4f}" for k, v in sorted(action.items()))
        v_value = self._critic.get(self._current_state.state_id, "value")
        
        return self._build_result(
            action=action,
            confidence=min(0.9, 0.6 + self._iteration_count * 0.01),
            reasoning=f"Actor-Critic：噪声={self._noise_std:.4f}, V={v_value:.4f}",
            alternatives=[],
            reasoning_steps=[
                f"Actor参数: {self._actor_params}",
                f"Critic V值: {v_value:.4f}",
                f"噪声标准差: {self._noise_std:.4f}"
            ],
            debug_info={
                'actor_params': self._actor_params.copy(),
                'v_value': v_value,
                'noise_std': self._noise_std
            }
        )
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None,
               done: bool = False) -> None:
        """更新Actor和Critic"""
        if self._current_state is None:
            return
        
        state = self._current_state
        next_state = self._context_to_state(context) if context else state
        
        # Critic更新（TD学习）
        current_v = self._critic.get(state.state_id, "value")
        if done:
            td_target = reward
        else:
            next_v = self._critic.get(next_state.state_id, "value")
            td_target = reward + self._discount_factor * next_v
        
        td_error = td_target - current_v
        new_v = current_v + self._critic_lr * td_error
        self._critic.set(state.state_id, "value", new_v)
        
        # Actor更新（策略梯度）
        for param_name, value in action.items():
            if param_name in self._actor_params:
                low, high = self._param_bounds.get(param_name, (0.0, 1.0))
                diff = value - self._actor_params[param_name]
                gradient = diff * td_error / (self._noise_std * (high - low) + 1e-8)
                gradient = max(-1.0, min(1.0, gradient))
                
                self._actor_params[param_name] += self._actor_lr * gradient
                self._actor_params[param_name] = max(low, min(high, self._actor_params[param_name]))
        
        # 累计奖励
        self._current_episode_reward += reward
        
        if done:
            self._episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        
        # 衰减噪声
        self._noise_std = max(self._min_noise, self._noise_std * self._noise_decay)
        
        # 更新状态
        self._current_state = next_state
        self._record_observation(action, reward)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._episode_rewards:
            return {'status': 'no_episodes'}
        
        return {
            'status': 'active',
            'episode_count': len(self._episode_rewards),
            'actor_params': self._actor_params,
            'avg_reward': sum(self._episode_rewards) / len(self._episode_rewards),
            'max_reward': max(self._episode_rewards),
            'noise_std': self._noise_std
        }
