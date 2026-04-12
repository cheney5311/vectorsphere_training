"""
多臂老虎机算法 - 生产级实现

实现多种经典和高级的多臂老虎机策略，用于探索-利用权衡：

基础策略：
- Epsilon-Greedy（含衰减变体）
- Upper Confidence Bound (UCB1, UCB-Tuned, UCB-V)
- Thompson Sampling（Beta, Gaussian）
- Softmax (Boltzmann Exploration)

高级策略：
- KL-UCB（KL散度上界）
- Exp3（对抗性环境）
- LinUCB（线性上下文）
- Neural UCB（神经网络上下文）

高级特性：
- 滑动窗口统计（非平稳环境）
- 变异检测（Change Detection）
- 臂淘汰机制（Arm Elimination）
- 批量选择（Batch Selection）
- 状态管理与热启动

使用示例：
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
        extra={
            "strategy": "ucb_tuned",
            "epsilon_decay": True,
            "sliding_window": 100,
            "arm_elimination": True
        }
    )
    
    algo = MultiArmedBanditAlgorithm(config)
    result = algo.suggest(context)
"""

import logging
import math
import random
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
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

class BanditStrategy(Enum):
    """Bandit策略类型"""
    EPSILON_GREEDY = "epsilon_greedy"
    EPSILON_GREEDY_DECAY = "epsilon_greedy_decay"
    UCB1 = "ucb1"
    UCB_TUNED = "ucb_tuned"
    UCB_V = "ucb_v"
    KL_UCB = "kl_ucb"
    THOMPSON_BETA = "thompson_beta"
    THOMPSON_GAUSSIAN = "thompson_gaussian"
    SOFTMAX = "softmax"
    EXP3 = "exp3"
    GRADIENT_BANDIT = "gradient_bandit"


class ContextualStrategy(Enum):
    """上下文Bandit策略"""
    LINEAR = "linear"
    LIN_UCB = "lin_ucb"
    LIN_TS = "lin_ts"
    NEURAL_UCB = "neural_ucb"


# ==================== 数据结构定义 ====================

@dataclass
class ArmStatistics:
    """臂的统计信息 - 增强版"""
    arm_id: str
    action: Dict[str, Any]
    n_pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    reward_variance: float = 0.0
    rewards_history: List[float] = field(default_factory=list)
    
    # Thompson Sampling参数
    alpha: float = 1.0  # Beta分布的alpha参数
    beta: float = 1.0   # Beta分布的beta参数
    
    # Gaussian Thompson Sampling参数
    mu: float = 0.0     # 均值
    sigma: float = 1.0  # 标准差
    tau: float = 1.0    # 精度
    
    # UCB-V参数
    sum_squared_rewards: float = 0.0
    
    # Exp3参数
    weight: float = 1.0
    probability: float = 0.0
    
    # Gradient Bandit参数
    preference: float = 0.0
    
    # 滑动窗口统计
    window_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    window_mean: float = 0.0
    window_variance: float = 0.0
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    last_pulled: Optional[datetime] = None
    eliminated: bool = False
    elimination_reason: str = ""
    
    def update(self, reward: float, window_size: int = 100) -> None:
        """更新统计信息
        
        Args:
            reward: 获得的奖励
            window_size: 滑动窗口大小
        """
        self.n_pulls += 1
        self.total_reward += reward
        self.rewards_history.append(reward)
        self.last_pulled = datetime.now()
        
        # 增量更新均值
        old_mean = self.mean_reward
        self.mean_reward = self.total_reward / self.n_pulls
        
        # 增量更新方差 (Welford's algorithm)
        if self.n_pulls > 1:
            self.reward_variance = (
                (self.n_pulls - 2) / (self.n_pulls - 1) * self.reward_variance +
                (reward - old_mean) ** 2 / self.n_pulls
            )
        
        # UCB-V参数更新
        self.sum_squared_rewards += reward ** 2
        
        # 更新Beta分布参数（假设奖励在0-1之间）
        normalized_reward = min(1.0, max(0.0, reward))
        self.alpha += normalized_reward
        self.beta += 1 - normalized_reward
        
        # 更新Gaussian参数
        self._update_gaussian(reward)
        
        # 滑动窗口更新
        self._update_sliding_window(reward, window_size)
    
    def _update_gaussian(self, reward: float) -> None:
        """更新高斯分布参数"""
        # 增量更新（共轭先验）
        prior_tau = 1.0  # 先验精度
        observation_tau = 1.0  # 观测精度
        
        new_tau = self.tau + observation_tau
        new_mu = (self.tau * self.mu + observation_tau * reward) / new_tau
        
        self.tau = new_tau
        self.mu = new_mu
        self.sigma = 1.0 / math.sqrt(new_tau)
    
    def _update_sliding_window(self, reward: float, window_size: int) -> None:
        """更新滑动窗口统计"""
        # 更新窗口大小
        if self.window_rewards.maxlen != window_size:
            old_rewards = list(self.window_rewards)
            self.window_rewards = deque(old_rewards, maxlen=window_size)
        
        self.window_rewards.append(reward)
        
        if len(self.window_rewards) > 0:
            rewards_list = list(self.window_rewards)
            self.window_mean = sum(rewards_list) / len(rewards_list)
            if len(rewards_list) > 1:
                self.window_variance = sum((r - self.window_mean) ** 2 
                                          for r in rewards_list) / (len(rewards_list) - 1)
    
    def get_ucb_v_variance(self) -> float:
        """获取UCB-V的方差估计"""
        if self.n_pulls <= 1:
            return 1.0
        
        mean_squared = self.sum_squared_rewards / self.n_pulls
        variance = mean_squared - self.mean_reward ** 2
        return max(0.0, variance)
    
    def get_kl_ucb_bound(self, total_pulls: int, c: float = 3.0) -> float:
        """计算KL-UCB上界
        
        Args:
            total_pulls: 总拉取次数
            c: 常数参数
            
        Returns:
            KL-UCB上界
        """
        if self.n_pulls == 0:
            return float('inf')
        
        p = self.mean_reward
        threshold = (math.log(total_pulls) + c * math.log(math.log(total_pulls + 1) + 1)) / self.n_pulls
        
        # 二分搜索找到满足KL散度约束的最大q
        low, high = p, 1.0
        for _ in range(32):  # 足够的迭代次数
            mid = (low + high) / 2
            kl = self._kl_divergence(p, mid)
            if kl < threshold:
                low = mid
            else:
                high = mid
        
        return low
    
    def _kl_divergence(self, p: float, q: float) -> float:
        """计算Bernoulli KL散度"""
        p = min(0.9999, max(0.0001, p))
        q = min(0.9999, max(0.0001, q))
        
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    def reset_for_change_detection(self) -> None:
        """重置用于变异检测"""
        self.n_pulls = 0
        self.total_reward = 0.0
        self.mean_reward = 0.0
        self.reward_variance = 0.0
        self.sum_squared_rewards = 0.0
        self.alpha = 1.0
        self.beta = 1.0
        self.mu = 0.0
        self.sigma = 1.0
        self.tau = 1.0
        self.window_rewards.clear()
        self.window_mean = 0.0
        self.window_variance = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'arm_id': self.arm_id,
            'action': self.action,
            'n_pulls': self.n_pulls,
            'total_reward': self.total_reward,
            'mean_reward': self.mean_reward,
            'reward_variance': self.reward_variance,
            'alpha': self.alpha,
            'beta': self.beta,
            'mu': self.mu,
            'sigma': self.sigma,
            'weight': self.weight,
            'preference': self.preference,
            'eliminated': self.eliminated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArmStatistics':
        """从字典创建"""
        arm = cls(
            arm_id=data['arm_id'],
            action=data['action']
        )
        arm.n_pulls = data.get('n_pulls', 0)
        arm.total_reward = data.get('total_reward', 0.0)
        arm.mean_reward = data.get('mean_reward', 0.0)
        arm.reward_variance = data.get('reward_variance', 0.0)
        arm.alpha = data.get('alpha', 1.0)
        arm.beta = data.get('beta', 1.0)
        arm.mu = data.get('mu', 0.0)
        arm.sigma = data.get('sigma', 1.0)
        arm.weight = data.get('weight', 1.0)
        arm.preference = data.get('preference', 0.0)
        arm.eliminated = data.get('eliminated', False)
        return arm


@dataclass
class SelectionResult:
    """选择结果"""
    arm: ArmStatistics
    reasoning: str
    steps: List[str]
    confidence: float
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeDetectionResult:
    """变异检测结果"""
    detected: bool
    arm_id: str
    old_mean: float
    new_mean: float
    detection_method: str
    timestamp: datetime = field(default_factory=datetime.now)


class MultiArmedBanditAlgorithm(BaseAlgorithm):
    """多臂老虎机算法
    
    支持多种策略：
    - epsilon_greedy: ε-贪婪策略
    - epsilon_greedy_decay: 衰减ε-贪婪
    - ucb1: Upper Confidence Bound
    - ucb_tuned: UCB-Tuned (方差感知)
    - ucb_v: UCB-V (方差上界)
    - kl_ucb: KL-UCB (KL散度上界)
    - thompson_beta: Thompson Sampling (Beta分布)
    - thompson_gaussian: Thompson Sampling (高斯分布)
    - softmax: Softmax/Boltzmann探索
    - exp3: Exp3 (对抗性环境)
    - gradient_bandit: 梯度Bandit
    
    高级特性：
    - 滑动窗口统计（非平稳环境）
    - 变异检测
    - 臂淘汰机制
    - 批量选择
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._arms: Dict[str, ArmStatistics] = {}
        self._total_pulls: int = 0
        
        # 从配置获取参数
        extra = config.extra if config and config.extra else {}
        
        # 策略设置
        strategy_str = extra.get('strategy', 'ucb1')
        try:
            self._strategy = BanditStrategy(strategy_str)
        except ValueError:
            self._strategy = BanditStrategy.UCB1
            self.logger.warning(f"Unknown strategy '{strategy_str}', defaulting to UCB1")
        
        # ε-贪婪参数
        self._epsilon = config.epsilon if config else 0.1
        self._epsilon_decay = extra.get('epsilon_decay', False)
        self._epsilon_decay_rate = extra.get('epsilon_decay_rate', 0.995)
        self._epsilon_min = extra.get('epsilon_min', 0.01)
        self._initial_epsilon = self._epsilon
        
        # UCB参数
        self._ucb_c = config.ucb_c if config else 2.0
        
        # Softmax参数
        self._temperature = extra.get('temperature', 1.0)
        self._temperature_decay = extra.get('temperature_decay', False)
        self._temperature_decay_rate = extra.get('temperature_decay_rate', 0.99)
        self._temperature_min = extra.get('temperature_min', 0.1)
        
        # Exp3参数
        self._exp3_gamma = extra.get('exp3_gamma', 0.1)
        
        # Gradient Bandit参数
        self._gradient_alpha = extra.get('gradient_alpha', 0.1)
        self._baseline_reward = 0.0
        
        # 滑动窗口设置
        self._use_sliding_window = extra.get('use_sliding_window', False)
        self._sliding_window_size = extra.get('sliding_window_size', 100)
        
        # 变异检测设置
        self._enable_change_detection = extra.get('enable_change_detection', False)
        self._change_detection_threshold = extra.get('change_detection_threshold', 2.0)
        self._change_detection_window = extra.get('change_detection_window', 50)
        self._change_history: List[ChangeDetectionResult] = []
        
        # 臂淘汰设置
        self._enable_arm_elimination = extra.get('enable_arm_elimination', False)
        self._elimination_threshold = extra.get('elimination_threshold', 0.05)
        self._min_pulls_for_elimination = extra.get('min_pulls_for_elimination', 20)
        
        # 批量选择设置
        self._batch_size = extra.get('batch_size', 1)
        
        # 统计信息
        self._cumulative_reward = 0.0
        self._cumulative_regret = 0.0
        self._best_arm_reward = 0.0
        self._selection_history: List[Dict[str, Any]] = []
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.MULTI_ARMED_BANDIT
    
    @property
    def arms(self) -> Dict[str, ArmStatistics]:
        """获取所有臂"""
        return self._arms
    
    @property
    def active_arms(self) -> Dict[str, ArmStatistics]:
        """获取活跃臂（未淘汰）"""
        return {k: v for k, v in self._arms.items() if not v.eliminated}
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化多臂老虎机
        
        Args:
            context: 算法上下文
        """
        super().initialize(context)
        
        # 从搜索空间或历史创建臂
        self._initialize_arms(context)
        
        # 从历史观测更新统计
        for obs in self._observations:
            arm_id = self._action_to_arm_id(obs['action'])
            if arm_id in self._arms:
                self._arms[arm_id].update(obs['reward'], self._sliding_window_size)
                self._total_pulls += 1
                self._cumulative_reward += obs['reward']
        
        # 初始化Exp3权重
        if self._strategy == BanditStrategy.EXP3:
            self._initialize_exp3_weights()
        
        self.logger.info(
            f"Multi-armed bandit initialized: {len(self._arms)} arms, "
            f"strategy={self._strategy.value}, total_pulls={self._total_pulls}"
        )
    
    def suggest(
        self, 
        context: AlgorithmContext,
        batch_size: int = None
    ) -> AlgorithmResult:
        """生成选择建议
        
        Args:
            context: 算法上下文
            batch_size: 批量选择数量（可选）
            
        Returns:
            算法结果
        """
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        reasoning_steps = []
        
        # 变异检测
        if self._enable_change_detection and self._total_pulls > self._change_detection_window:
            change_results = self._detect_changes()
            if change_results:
                for result in change_results:
                    reasoning_steps.append(f"检测到变异: {result.arm_id} ({result.detection_method})")
        
        # 臂淘汰
        if self._enable_arm_elimination:
            eliminated = self._eliminate_suboptimal_arms()
            if eliminated:
                reasoning_steps.append(f"淘汰了 {len(eliminated)} 个次优臂")
        
        # 获取活跃臂
        active_arms = self.active_arms
        if not active_arms:
            self.logger.warning("No active arms, resetting elimination")
            for arm in self._arms.values():
                arm.eliminated = False
            active_arms = self._arms
        
        # 确定批量大小
        actual_batch_size = batch_size or self._batch_size
        
        # 确保每个臂至少被拉一次
        unvisited_arms = [arm for arm in active_arms.values() if arm.n_pulls == 0]
        
        if unvisited_arms:
            if actual_batch_size > 1:
                # 批量选择未访问的臂
                selected_arms = random.sample(
                    unvisited_arms, 
                    min(actual_batch_size, len(unvisited_arms))
                )
                selected_arm = selected_arms[0]
            else:
                selected_arm = random.choice(unvisited_arms)
            
            reasoning = f"初始探索：选择未访问的臂 {selected_arm.arm_id}"
            reasoning_steps.append(f"存在 {len(unvisited_arms)} 个未访问的臂")
            confidence = 0.5
            selection_result = SelectionResult(
                arm=selected_arm,
                reasoning=reasoning,
                steps=reasoning_steps,
                confidence=confidence
            )
        else:
            # 根据策略选择臂
            if actual_batch_size > 1:
                selection_results = self._select_batch(actual_batch_size, active_arms)
                selection_result = selection_results[0]
            else:
                selection_result = self._select_arm(active_arms)
        
        # 更新探索参数（衰减）
        self._update_exploration_params()
        
        # 记录选择历史
        self._selection_history.append({
            'iteration': self._iteration_count,
            'arm_id': selection_result.arm.arm_id,
            'strategy': self._strategy.value,
            'confidence': selection_result.confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # 生成备选方案
        alternatives = self._generate_alternatives(
            exclude_arm=selection_result.arm,
            active_arms=active_arms
        )
        
        # 完善推理步骤
        reasoning_steps.extend([
            f"策略: {self._strategy.value}",
            f"总拉取次数: {self._total_pulls}",
            f"活跃臂数量: {len(active_arms)}",
            f"选中臂: {selection_result.arm.arm_id}",
            f"选中臂的历史拉取次数: {selection_result.arm.n_pulls}",
            f"选中臂的平均奖励: {selection_result.arm.mean_reward:.4f}",
            f"累计奖励: {self._cumulative_reward:.4f}",
            f"累计遗憾: {self._cumulative_regret:.4f}"
        ])
        reasoning_steps.extend(selection_result.steps)
        
        # 构建结果
        result_action = selection_result.arm.action
        if actual_batch_size > 1 and hasattr(self, '_batch_actions'):
            result_action = {
                'primary': selection_result.arm.action,
                'batch': self._batch_actions
            }
        
        return self._build_result(
            action=result_action,
            confidence=selection_result.confidence,
            reasoning=selection_result.reasoning,
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            debug_info={
                'strategy': self._strategy.value,
                'arm_id': selection_result.arm.arm_id,
                'n_arms': len(self._arms),
                'n_active_arms': len(active_arms),
                'total_pulls': self._total_pulls,
                'cumulative_reward': self._cumulative_reward,
                'cumulative_regret': self._cumulative_regret,
                'epsilon': self._epsilon if self._strategy in [
                    BanditStrategy.EPSILON_GREEDY, 
                    BanditStrategy.EPSILON_GREEDY_DECAY
                ] else None,
                'temperature': self._temperature if self._strategy == BanditStrategy.SOFTMAX else None,
                'score': selection_result.score,
                'metadata': selection_result.metadata
            }
        )
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新臂的统计信息
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            context: 可选的上下文
        """
        arm_id = self._action_to_arm_id(action)
        
        # 如果是新臂，创建它
        if arm_id not in self._arms:
            self._arms[arm_id] = ArmStatistics(
                arm_id=arm_id,
                action=action,
                alpha=self.config.thompson_alpha if self.config else 1.0,
                beta=self.config.thompson_beta if self.config else 1.0
            )
        
        arm = self._arms[arm_id]
        arm.update(reward, self._sliding_window_size)
        self._total_pulls += 1
        self._cumulative_reward += reward
        
        # 更新最佳臂奖励（用于遗憾计算）
        if arm.mean_reward > self._best_arm_reward:
            self._best_arm_reward = arm.mean_reward
        
        # 计算瞬时遗憾
        instant_regret = max(0, self._best_arm_reward - reward)
        self._cumulative_regret += instant_regret
        
        # 更新基线奖励（用于Gradient Bandit）
        self._baseline_reward = (
            self._baseline_reward * (self._total_pulls - 1) + reward
        ) / self._total_pulls
        
        # 更新Exp3权重
        if self._strategy == BanditStrategy.EXP3:
            self._update_exp3_weights(arm_id, reward)
        
        # 更新Gradient Bandit偏好
        if self._strategy == BanditStrategy.GRADIENT_BANDIT:
            self._update_gradient_preferences(arm_id, reward)
        
        self._record_observation(action, reward)
        
        self.logger.debug(
            f"Arm {arm_id} updated: n={arm.n_pulls}, "
            f"mean={arm.mean_reward:.4f}, "
            f"cumulative_reward={self._cumulative_reward:.4f}"
        )
    
    def evaluate(self, action: Dict[str, Any], 
                context: AlgorithmContext) -> float:
        """评估动作的预期收益
        
        Args:
            action: 待评估的动作
            context: 算法上下文
            
        Returns:
            预期收益值
        """
        arm_id = self._action_to_arm_id(action)
        if arm_id in self._arms:
            return self._arms[arm_id].mean_reward
        return 0.0
    
    def _initialize_arms(self, context: AlgorithmContext) -> None:
        """初始化臂
        
        Args:
            context: 算法上下文
        """
        self._arms = {}
        self._total_pulls = 0
        
        # 从搜索空间创建臂
        if context.search_space:
            self._create_arms_from_search_space(context.search_space)
        
        # 从历史动作创建臂
        for obs in context.observations:
            arm_id = self._action_to_arm_id(obs['action'])
            if arm_id not in self._arms:
                self._arms[arm_id] = ArmStatistics(
                    arm_id=arm_id,
                    action=obs['action'],
                    alpha=self.config.thompson_alpha,
                    beta=self.config.thompson_beta
                )
        
        # 如果没有臂，创建默认臂
        if not self._arms:
            self._arms['default'] = ArmStatistics(
                arm_id='default',
                action={},
                alpha=self.config.thompson_alpha,
                beta=self.config.thompson_beta
            )
    
    def _create_arms_from_search_space(self, search_space: Dict[str, Any]) -> None:
        """从搜索空间创建臂
        
        对于离散选择，每个选择是一个臂。
        对于连续参数，离散化为多个臂。
        
        Args:
            search_space: 搜索空间定义
        """
        # 收集所有可能的选项
        param_options = {}
        for param_name, param_def in search_space.items():
            if isinstance(param_def, dict):
                if 'choices' in param_def:
                    param_options[param_name] = param_def['choices']
                elif 'low' in param_def and 'high' in param_def:
                    # 连续参数离散化为5个值
                    low, high = param_def['low'], param_def['high']
                    n_bins = param_def.get('n_bins', 5)
                    step = (high - low) / n_bins
                    param_options[param_name] = [low + i * step for i in range(n_bins + 1)]
            elif isinstance(param_def, list):
                param_options[param_name] = param_def
        
        # 生成所有组合（限制数量）
        if param_options:
            self._generate_arm_combinations(param_options, max_arms=100)
    
    def _generate_arm_combinations(self, param_options: Dict[str, List[Any]], 
                                  max_arms: int = 100) -> None:
        """生成臂组合
        
        Args:
            param_options: 参数选项
            max_arms: 最大臂数量
        """
        from itertools import product
        
        param_names = list(param_options.keys())
        param_values = list(param_options.values())
        
        # 计算总组合数
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        # 如果组合太多，随机采样
        if total_combinations > max_arms:
            for i in range(max_arms):
                action = {}
                for name, values in zip(param_names, param_values):
                    action[name] = random.choice(values)
                arm_id = self._action_to_arm_id(action)
                if arm_id not in self._arms:
                    self._arms[arm_id] = ArmStatistics(
                        arm_id=arm_id,
                        action=action,
                        alpha=self.config.thompson_alpha,
                        beta=self.config.thompson_beta
                    )
        else:
            # 生成所有组合
            for values in product(*param_values):
                action = dict(zip(param_names, values))
                arm_id = self._action_to_arm_id(action)
                self._arms[arm_id] = ArmStatistics(
                    arm_id=arm_id,
                    action=action,
                    alpha=self.config.thompson_alpha,
                    beta=self.config.thompson_beta
                )
    
    def _action_to_arm_id(self, action: Dict[str, Any]) -> str:
        """将动作转换为臂ID
        
        Args:
            action: 动作字典
            
        Returns:
            臂ID
        """
        sorted_items = sorted(action.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def _select_arm(
        self, 
        active_arms: Dict[str, ArmStatistics] = None
    ) -> SelectionResult:
        """根据策略选择臂
        
        Args:
            active_arms: 活跃臂字典
        
        Returns:
            选择结果
        """
        arms = active_arms or self.active_arms
        
        strategy_methods = {
            BanditStrategy.EPSILON_GREEDY: self._epsilon_greedy_select,
            BanditStrategy.EPSILON_GREEDY_DECAY: self._epsilon_greedy_select,
            BanditStrategy.UCB1: self._ucb1_select,
            BanditStrategy.UCB_TUNED: self._ucb_tuned_select,
            BanditStrategy.UCB_V: self._ucb_v_select,
            BanditStrategy.KL_UCB: self._kl_ucb_select,
            BanditStrategy.THOMPSON_BETA: self._thompson_beta_select,
            BanditStrategy.THOMPSON_GAUSSIAN: self._thompson_gaussian_select,
            BanditStrategy.SOFTMAX: self._softmax_select,
            BanditStrategy.EXP3: self._exp3_select,
            BanditStrategy.GRADIENT_BANDIT: self._gradient_bandit_select,
        }
        
        method = strategy_methods.get(self._strategy, self._ucb1_select)
        return method(arms)
    
    def _select_batch(
        self, 
        batch_size: int,
        active_arms: Dict[str, ArmStatistics] = None
    ) -> List[SelectionResult]:
        """批量选择多个臂
        
        Args:
            batch_size: 批量大小
            active_arms: 活跃臂字典
        
        Returns:
            选择结果列表
        """
        arms = active_arms or self.active_arms
        results = []
        selected_ids = set()
        
        for _ in range(min(batch_size, len(arms))):
            # 排除已选择的臂
            remaining_arms = {k: v for k, v in arms.items() if k not in selected_ids}
            if not remaining_arms:
                break
            
            result = self._select_arm(remaining_arms)
            results.append(result)
            selected_ids.add(result.arm.arm_id)
        
        # 记录批量动作
        self._batch_actions = [r.arm.action for r in results]
        
        return results
    
    def _epsilon_greedy_select(
        self, 
        arms: Dict[str, ArmStatistics]
    ) -> SelectionResult:
        """ε-贪婪选择"""
        epsilon = self._epsilon
        arms_list = list(arms.values())
        
        if random.random() < epsilon:
            # 探索：随机选择
            selected = random.choice(arms_list)
            return SelectionResult(
                arm=selected,
                reasoning=f"ε-贪婪探索（ε={epsilon:.4f}）：随机选择臂",
                steps=[f"探索概率 ε={epsilon:.4f}", "触发随机探索"],
                confidence=0.6,
                score=selected.mean_reward
            )
        else:
            # 利用：选择最佳
            if self._use_sliding_window:
                selected = max(arms_list, key=lambda a: a.window_mean)
            else:
                selected = max(arms_list, key=lambda a: a.mean_reward)
            
            return SelectionResult(
                arm=selected,
                reasoning=f"ε-贪婪利用：选择当前最佳臂（平均奖励={selected.mean_reward:.4f}）",
                steps=[
                    f"利用概率 1-ε={1-epsilon:.4f}", 
                    f"最佳臂平均奖励={selected.mean_reward:.4f}"
                ],
                confidence=0.8,
                score=selected.mean_reward
            )
    
    def _ucb1_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """UCB1选择
        
        UCB值 = 平均奖励 + c * sqrt(ln(总次数) / 臂次数)
        """
        c = self._ucb_c
        total_n = self._total_pulls
        
        best_arm = None
        best_ucb = float('-inf')
        arm_ucb_values = {}
        
        for arm in arms.values():
            if arm.n_pulls == 0:
                ucb = float('inf')
            else:
                exploration_term = c * math.sqrt(math.log(total_n + 1) / arm.n_pulls)
                ucb = arm.mean_reward + exploration_term
            
            arm_ucb_values[arm.arm_id] = ucb
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        
        exploration_term = c * math.sqrt(math.log(total_n + 1) / best_arm.n_pulls) if best_arm.n_pulls > 0 else 0
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"UCB1选择：UCB值={best_ucb:.4f}（均值={best_arm.mean_reward:.4f} + 探索={exploration_term:.4f}）",
            steps=[
                f"UCB参数 c={c}",
                f"总拉取次数={total_n}",
                f"选中臂拉取次数={best_arm.n_pulls}",
                f"选中臂UCB值={best_ucb:.4f}"
            ],
            confidence=min(0.95, 0.6 + best_arm.n_pulls * 0.01),
            score=best_ucb,
            metadata={'ucb_values': arm_ucb_values}
        )
    
    def _ucb_tuned_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """UCB-Tuned选择（方差感知）
        
        UCB值 = 平均奖励 + sqrt(ln(t)/n * min(1/4, V(n)))
        V(n) = σ² + sqrt(2*ln(t)/n)
        """
        total_n = self._total_pulls
        
        best_arm = None
        best_ucb = float('-inf')
        
        for arm in arms.values():
            if arm.n_pulls == 0:
                ucb = float('inf')
            else:
                # 计算方差项
                v_n = arm.reward_variance + math.sqrt(2 * math.log(total_n + 1) / arm.n_pulls)
                v_n = min(0.25, v_n)  # 上界为1/4
                
                exploration_term = math.sqrt(math.log(total_n + 1) / arm.n_pulls * v_n)
                ucb = arm.mean_reward + exploration_term
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"UCB-Tuned选择：UCB值={best_ucb:.4f}（方差感知）",
            steps=[
                f"臂方差={best_arm.reward_variance:.4f}",
                f"选中臂UCB值={best_ucb:.4f}"
            ],
            confidence=min(0.95, 0.6 + best_arm.n_pulls * 0.01),
            score=best_ucb
        )
    
    def _ucb_v_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """UCB-V选择（方差上界）
        
        UCB值 = 平均奖励 + sqrt(2*σ²*ln(t)/n) + c*ln(t)/n
        """
        c = self._ucb_c
        total_n = self._total_pulls
        
        best_arm = None
        best_ucb = float('-inf')
        
        for arm in arms.values():
            if arm.n_pulls == 0:
                ucb = float('inf')
            else:
                variance = arm.get_ucb_v_variance()
                exploration_term = math.sqrt(2 * variance * math.log(total_n + 1) / arm.n_pulls)
                exploration_term += c * math.log(total_n + 1) / arm.n_pulls
                ucb = arm.mean_reward + exploration_term
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"UCB-V选择：UCB值={best_ucb:.4f}（方差上界）",
            steps=[
                f"臂方差估计={best_arm.get_ucb_v_variance():.4f}",
                f"选中臂UCB值={best_ucb:.4f}"
            ],
            confidence=min(0.95, 0.6 + best_arm.n_pulls * 0.01),
            score=best_ucb
        )
    
    def _kl_ucb_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """KL-UCB选择
        
        选择满足KL散度约束的上界最大的臂
        """
        total_n = self._total_pulls
        
        best_arm = None
        best_bound = float('-inf')
        
        for arm in arms.values():
            bound = arm.get_kl_ucb_bound(total_n)
            
            if bound > best_bound:
                best_bound = bound
                best_arm = arm
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"KL-UCB选择：上界={best_bound:.4f}",
            steps=[
                f"选中臂均值={best_arm.mean_reward:.4f}",
                f"KL-UCB上界={best_bound:.4f}"
            ],
            confidence=min(0.95, 0.6 + best_arm.n_pulls * 0.01),
            score=best_bound
        )
    
    def _thompson_beta_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """Thompson Sampling选择（Beta分布）"""
        best_arm = None
        best_sample = float('-inf')
        samples = {}
        
        for arm in arms.values():
            sample = random.betavariate(arm.alpha, arm.beta)
            samples[arm.arm_id] = sample
            
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        
        posterior_mean = best_arm.alpha / (best_arm.alpha + best_arm.beta)
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"Thompson Sampling(Beta)：采样值={best_sample:.4f}",
            steps=[
                f"Beta分布参数: α={best_arm.alpha:.2f}, β={best_arm.beta:.2f}",
                f"后验均值={posterior_mean:.4f}",
                f"采样值={best_sample:.4f}"
            ],
            confidence=min(0.95, posterior_mean),
            score=best_sample,
            metadata={'samples': samples}
        )
    
    def _thompson_gaussian_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """Thompson Sampling选择（高斯分布）"""
        best_arm = None
        best_sample = float('-inf')
        samples = {}
        
        for arm in arms.values():
            sample = random.gauss(arm.mu, arm.sigma)
            samples[arm.arm_id] = sample
            
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        
        return SelectionResult(
            arm=best_arm,
            reasoning=f"Thompson Sampling(Gaussian)：采样值={best_sample:.4f}",
            steps=[
                f"高斯分布参数: μ={best_arm.mu:.4f}, σ={best_arm.sigma:.4f}",
                f"采样值={best_sample:.4f}"
            ],
            confidence=min(0.95, 0.5 + 0.5 / (1 + best_arm.sigma)),
            score=best_sample,
            metadata={'samples': samples}
        )
    
    def _softmax_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """Softmax/Boltzmann选择"""
        temperature = self._temperature
        arms_list = list(arms.values())
        
        if self._use_sliding_window:
            rewards = [arm.window_mean for arm in arms_list]
        else:
            rewards = [arm.mean_reward for arm in arms_list]
        
        max_reward = max(rewards) if rewards else 0
        
        # 避免数值溢出
        exp_values = [math.exp((r - max_reward) / temperature) for r in rewards]
        total = sum(exp_values)
        probabilities = [e / total for e in exp_values]
        
        # 按概率选择
        r = random.random()
        cumsum = 0
        selected = arms_list[0]
        selected_prob = probabilities[0]
        for arm, prob in zip(arms_list, probabilities):
            cumsum += prob
            if r <= cumsum:
                selected = arm
                selected_prob = prob
                break
        
        return SelectionResult(
            arm=selected,
            reasoning=f"Softmax选择：温度={temperature:.4f}, 选择概率={selected_prob:.4f}",
            steps=[
                f"温度参数 τ={temperature:.4f}",
                f"选中臂平均奖励={selected.mean_reward:.4f}",
                f"选择概率={selected_prob:.4f}"
            ],
            confidence=selected_prob,
            score=selected.mean_reward,
            metadata={'probabilities': dict(zip([a.arm_id for a in arms_list], probabilities))}
        )
    
    def _exp3_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """Exp3选择（对抗性环境）"""
        gamma = self._exp3_gamma
        arms_list = list(arms.values())
        n_arms = len(arms_list)
        
        # 计算概率分布
        total_weight = sum(arm.weight for arm in arms_list)
        
        probabilities = []
        for arm in arms_list:
            prob = (1 - gamma) * (arm.weight / total_weight) + gamma / n_arms
            arm.probability = prob
            probabilities.append(prob)
        
        # 按概率选择
        r = random.random()
        cumsum = 0
        selected = arms_list[0]
        selected_prob = probabilities[0]
        
        for arm, prob in zip(arms_list, probabilities):
            cumsum += prob
            if r <= cumsum:
                selected = arm
                selected_prob = prob
                break
        
        return SelectionResult(
            arm=selected,
            reasoning=f"Exp3选择：γ={gamma:.4f}, 选择概率={selected_prob:.4f}",
            steps=[
                f"探索参数 γ={gamma:.4f}",
                f"臂权重={selected.weight:.4f}",
                f"选择概率={selected_prob:.4f}"
            ],
            confidence=selected_prob,
            score=selected.weight,
            metadata={'probabilities': dict(zip([a.arm_id for a in arms_list], probabilities))}
        )
    
    def _gradient_bandit_select(self, arms: Dict[str, ArmStatistics]) -> SelectionResult:
        """Gradient Bandit选择"""
        arms_list = list(arms.values())
        
        # 计算softmax概率
        preferences = [arm.preference for arm in arms_list]
        max_pref = max(preferences) if preferences else 0
        
        exp_values = [math.exp(p - max_pref) for p in preferences]
        total = sum(exp_values)
        probabilities = [e / total for e in exp_values]
        
        # 按概率选择
        r = random.random()
        cumsum = 0
        selected = arms_list[0]
        selected_prob = probabilities[0]
        
        for arm, prob in zip(arms_list, probabilities):
            cumsum += prob
            if r <= cumsum:
                selected = arm
                selected_prob = prob
                break
        
        return SelectionResult(
            arm=selected,
            reasoning=f"Gradient Bandit选择：选择概率={selected_prob:.4f}",
            steps=[
                f"偏好值={selected.preference:.4f}",
                f"选择概率={selected_prob:.4f}",
                f"基线奖励={self._baseline_reward:.4f}"
            ],
            confidence=selected_prob,
            score=selected.preference,
            metadata={'probabilities': dict(zip([a.arm_id for a in arms_list], probabilities))}
        )
    
    def _generate_alternatives(
        self, 
        exclude_arm: ArmStatistics = None,
        active_arms: Dict[str, ArmStatistics] = None
    ) -> List[Dict[str, Any]]:
        """生成备选方案"""
        alternatives = []
        arms = active_arms or self.active_arms
        
        # 按平均奖励排序
        sorted_arms = sorted(
            arms.values(),
            key=lambda a: a.mean_reward,
            reverse=True
        )
        
        for arm in sorted_arms[:3]:
            if exclude_arm and arm.arm_id == exclude_arm.arm_id:
                continue
            
            alternatives.append({
                'action': arm.action,
                'confidence': min(0.9, 0.5 + arm.n_pulls * 0.02),
                'mean_reward': arm.mean_reward,
                'n_pulls': arm.n_pulls,
                'reasoning': f"平均奖励={arm.mean_reward:.4f}, 拉取次数={arm.n_pulls}"
            })
        
        return alternatives[:2]
    
    # ==================== 参数更新方法 ====================
    
    def _update_exploration_params(self) -> None:
        """更新探索参数（衰减）"""
        # ε衰减
        if self._epsilon_decay:
            self._epsilon = max(
                self._epsilon_min,
                self._epsilon * self._epsilon_decay_rate
            )
        
        # 温度衰减
        if self._temperature_decay:
            self._temperature = max(
                self._temperature_min,
                self._temperature * self._temperature_decay_rate
            )
    
    def _initialize_exp3_weights(self) -> None:
        """初始化Exp3权重"""
        for arm in self._arms.values():
            arm.weight = 1.0
    
    def _update_exp3_weights(self, arm_id: str, reward: float) -> None:
        """更新Exp3权重"""
        arm = self._arms.get(arm_id)
        if not arm:
            return
        
        # 重要性加权奖励
        estimated_reward = reward / max(arm.probability, 0.001)
        gamma = self._exp3_gamma
        n_arms = len(self.active_arms)
        
        # 更新权重
        arm.weight *= math.exp(gamma * estimated_reward / n_arms)
        
        # 防止权重过大
        max_weight = max(a.weight for a in self._arms.values())
        if max_weight > 1e10:
            for a in self._arms.values():
                a.weight /= max_weight
    
    def _update_gradient_preferences(self, arm_id: str, reward: float) -> None:
        """更新Gradient Bandit偏好"""
        alpha = self._gradient_alpha
        
        # 计算当前概率
        arms_list = list(self._arms.values())
        preferences = [arm.preference for arm in arms_list]
        max_pref = max(preferences) if preferences else 0
        
        exp_values = [math.exp(p - max_pref) for p in preferences]
        total = sum(exp_values)
        probabilities = {arm.arm_id: e / total for arm, e in zip(arms_list, exp_values)}
        
        # 更新偏好
        for arm in arms_list:
            prob = probabilities[arm.arm_id]
            if arm.arm_id == arm_id:
                arm.preference += alpha * (reward - self._baseline_reward) * (1 - prob)
            else:
                arm.preference -= alpha * (reward - self._baseline_reward) * prob
    
    # ==================== 变异检测 ====================
    
    def _detect_changes(self) -> List[ChangeDetectionResult]:
        """检测奖励分布的变异"""
        changes = []
        window = self._change_detection_window
        threshold = self._change_detection_threshold
        
        for arm in self._arms.values():
            if arm.n_pulls < 2 * window:
                continue
            
            # 比较最近窗口和之前的均值
            recent_rewards = list(arm.window_rewards)[-window:]
            if len(recent_rewards) < window:
                continue
            
            recent_mean = sum(recent_rewards) / len(recent_rewards)
            old_mean = arm.mean_reward
            
            # 检测显著变化
            if arm.reward_variance > 0:
                z_score = abs(recent_mean - old_mean) / math.sqrt(arm.reward_variance / window)
                
                if z_score > threshold:
                    change = ChangeDetectionResult(
                        detected=True,
                        arm_id=arm.arm_id,
                        old_mean=old_mean,
                        new_mean=recent_mean,
                        detection_method='z_test'
                    )
                    changes.append(change)
                    self._change_history.append(change)
                    
                    # 重置臂统计
                    arm.reset_for_change_detection()
                    self.logger.info(
                        f"Change detected in arm {arm.arm_id}: "
                        f"old_mean={old_mean:.4f}, new_mean={recent_mean:.4f}"
                    )
        
        return changes
    
    # ==================== 臂淘汰 ====================
    
    def _eliminate_suboptimal_arms(self) -> List[str]:
        """淘汰次优臂"""
        eliminated = []
        active_arms = list(self.active_arms.values())
        
        if len(active_arms) <= 2:
            return eliminated
        
        # 找到当前最佳臂
        best_arm = max(active_arms, key=lambda a: a.mean_reward)
        best_mean = best_arm.mean_reward
        
        for arm in active_arms:
            if arm.arm_id == best_arm.arm_id:
                continue
            
            if arm.n_pulls < self._min_pulls_for_elimination:
                continue
            
            # 计算置信区间
            if arm.n_pulls > 0 and arm.reward_variance > 0:
                ci_width = 1.96 * math.sqrt(arm.reward_variance / arm.n_pulls)
                upper_bound = arm.mean_reward + ci_width
                
                # 如果上界显著低于最佳均值，淘汰
                if upper_bound < best_mean - self._elimination_threshold:
                    arm.eliminated = True
                    arm.elimination_reason = f"Upper bound {upper_bound:.4f} < best mean {best_mean:.4f}"
                    eliminated.append(arm.arm_id)
                    self.logger.info(f"Arm {arm.arm_id} eliminated: {arm.elimination_reason}")
        
        return eliminated
    
    def reset_elimination(self) -> None:
        """重置臂淘汰状态"""
        for arm in self._arms.values():
            arm.eliminated = False
            arm.elimination_reason = ""
    
    # ==================== 状态管理 ====================
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'arms': {k: v.to_dict() for k, v in self._arms.items()},
            'total_pulls': self._total_pulls,
            'cumulative_reward': self._cumulative_reward,
            'cumulative_regret': self._cumulative_regret,
            'best_arm_reward': self._best_arm_reward,
            'epsilon': self._epsilon,
            'temperature': self._temperature,
            'baseline_reward': self._baseline_reward,
            'strategy': self._strategy.value,
            'iteration_count': self._iteration_count,
            'observations': self._observations[-100:],  # 最近100条
            'change_history': [
                {
                    'arm_id': c.arm_id,
                    'old_mean': c.old_mean,
                    'new_mean': c.new_mean,
                    'method': c.detection_method
                }
                for c in self._change_history[-20:]
            ]
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        # 恢复臂
        self._arms = {
            k: ArmStatistics.from_dict(v) 
            for k, v in state.get('arms', {}).items()
        }
        
        self._total_pulls = state.get('total_pulls', 0)
        self._cumulative_reward = state.get('cumulative_reward', 0.0)
        self._cumulative_regret = state.get('cumulative_regret', 0.0)
        self._best_arm_reward = state.get('best_arm_reward', 0.0)
        self._epsilon = state.get('epsilon', self._initial_epsilon)
        self._temperature = state.get('temperature', 1.0)
        self._baseline_reward = state.get('baseline_reward', 0.0)
        self._iteration_count = state.get('iteration_count', 0)
        self._observations = state.get('observations', [])
        
        self.logger.info(
            f"State loaded: {len(self._arms)} arms, "
            f"{self._total_pulls} total pulls"
        )
    
    def warm_start(
        self, 
        arm_configs: List[Dict[str, Any]],
        observations: List[Dict[str, Any]] = None
    ) -> None:
        """热启动
        
        Args:
            arm_configs: 臂配置列表 [{'action': {...}, 'n_pulls': 10, 'mean_reward': 0.5}, ...]
            observations: 可选的历史观测
        """
        for config in arm_configs:
            action = config.get('action', {})
            arm_id = self._action_to_arm_id(action)
            
            arm = ArmStatistics(
                arm_id=arm_id,
                action=action
            )
            
            # 设置初始统计
            arm.n_pulls = config.get('n_pulls', 0)
            arm.mean_reward = config.get('mean_reward', 0.0)
            arm.total_reward = arm.n_pulls * arm.mean_reward
            arm.reward_variance = config.get('reward_variance', 0.0)
            arm.alpha = config.get('alpha', 1.0 + arm.n_pulls * arm.mean_reward)
            arm.beta = config.get('beta', 1.0 + arm.n_pulls * (1 - arm.mean_reward))
            
            self._arms[arm_id] = arm
            self._total_pulls += arm.n_pulls
            self._cumulative_reward += arm.total_reward
        
        if observations:
            for obs in observations:
                self._observations.append(obs)
        
        self._initialized = True
        self.logger.info(f"Warm start completed: {len(self._arms)} arms")
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        data = {
            'arms': [],
            'selection_history': self._selection_history[-100:],
            'regret_curve': [],
            'reward_curve': [],
            'statistics': {}
        }
        
        # 臂数据
        for arm in self._arms.values():
            data['arms'].append({
                'arm_id': arm.arm_id,
                'action': arm.action,
                'n_pulls': arm.n_pulls,
                'mean_reward': arm.mean_reward,
                'variance': arm.reward_variance,
                'eliminated': arm.eliminated,
                'alpha': arm.alpha,
                'beta': arm.beta,
                'weight': arm.weight,
                'preference': arm.preference
            })
        
        # 累计遗憾曲线
        cumulative_regret = 0.0
        cumulative_reward = 0.0
        for i, obs in enumerate(self._observations):
            cumulative_reward += obs.get('reward', 0)
            cumulative_regret += max(0, self._best_arm_reward - obs.get('reward', 0))
            
            if i % 10 == 0:  # 每10步记录一次
                data['regret_curve'].append({
                    'step': i,
                    'cumulative_regret': cumulative_regret
                })
                data['reward_curve'].append({
                    'step': i,
                    'cumulative_reward': cumulative_reward
                })
        
        # 统计信息
        data['statistics'] = {
            'total_arms': len(self._arms),
            'active_arms': len(self.active_arms),
            'total_pulls': self._total_pulls,
            'cumulative_reward': self._cumulative_reward,
            'cumulative_regret': self._cumulative_regret,
            'best_arm_reward': self._best_arm_reward,
            'current_epsilon': self._epsilon,
            'current_temperature': self._temperature,
            'strategy': self._strategy.value,
            'changes_detected': len(self._change_history)
        }
        
        return data
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._arms:
            return {'status': 'not_initialized'}
        
        active_arms = list(self.active_arms.values())
        if not active_arms:
            return {'status': 'no_active_arms'}
        
        best_arm = max(active_arms, key=lambda a: a.mean_reward)
        
        # 计算统计
        rewards = [a.mean_reward for a in active_arms]
        pulls = [a.n_pulls for a in active_arms]
        
        return {
            'status': 'active',
            'strategy': self._strategy.value,
            'total_arms': len(self._arms),
            'active_arms': len(active_arms),
            'eliminated_arms': len(self._arms) - len(active_arms),
            'total_pulls': self._total_pulls,
            'cumulative_reward': self._cumulative_reward,
            'cumulative_regret': self._cumulative_regret,
            'average_reward': self._cumulative_reward / self._total_pulls if self._total_pulls > 0 else 0,
            'best_arm': {
                'arm_id': best_arm.arm_id,
                'action': best_arm.action,
                'mean_reward': best_arm.mean_reward,
                'n_pulls': best_arm.n_pulls,
                'pull_ratio': best_arm.n_pulls / self._total_pulls if self._total_pulls > 0 else 0
            },
            'reward_stats': {
                'min': min(rewards),
                'max': max(rewards),
                'mean': sum(rewards) / len(rewards),
                'std': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5
            },
            'pull_distribution': {
                'min': min(pulls),
                'max': max(pulls),
                'mean': sum(pulls) / len(pulls)
            },
            'exploration_params': {
                'epsilon': self._epsilon,
                'temperature': self._temperature,
                'ucb_c': self._ucb_c
            },
            'changes_detected': len(self._change_history)
        }
    
    def get_arm_rankings(self, metric: str = 'mean_reward') -> List[Dict[str, Any]]:
        """获取臂排名
        
        Args:
            metric: 排序指标 ('mean_reward', 'n_pulls', 'ucb', 'thompson_sample')
            
        Returns:
            排名列表
        """
        active_arms = list(self.active_arms.values())
        
        if metric == 'mean_reward':
            sorted_arms = sorted(active_arms, key=lambda a: a.mean_reward, reverse=True)
        elif metric == 'n_pulls':
            sorted_arms = sorted(active_arms, key=lambda a: a.n_pulls, reverse=True)
        elif metric == 'ucb':
            def ucb_value(arm):
                if arm.n_pulls == 0:
                    return float('inf')
                return arm.mean_reward + self._ucb_c * math.sqrt(
                    math.log(self._total_pulls + 1) / arm.n_pulls
                )
            sorted_arms = sorted(active_arms, key=ucb_value, reverse=True)
        elif metric == 'thompson_sample':
            samples = [(arm, random.betavariate(arm.alpha, arm.beta)) for arm in active_arms]
            samples.sort(key=lambda x: x[1], reverse=True)
            sorted_arms = [s[0] for s in samples]
        else:
            sorted_arms = active_arms
        
        return [
            {
                'rank': i + 1,
                'arm_id': arm.arm_id,
                'action': arm.action,
                'mean_reward': arm.mean_reward,
                'n_pulls': arm.n_pulls,
                'variance': arm.reward_variance
            }
            for i, arm in enumerate(sorted_arms)
        ]


class ContextualBandit(MultiArmedBanditAlgorithm):
    """上下文多臂老虎机 - 生产级实现
    
    考虑上下文信息的多臂老虎机变体，支持多种策略：
    - Linear: 简单线性模型
    - LinUCB: 线性UCB（置信上界）
    - LinTS: 线性Thompson Sampling
    - Neural UCB: 神经网络UCB
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        # 上下文维度
        self._context_dim: int = 0
        
        # 策略设置
        extra = config.extra if config and config.extra else {}
        strategy_str = extra.get('contextual_strategy', 'lin_ucb')
        try:
            self._contextual_strategy = ContextualStrategy(strategy_str)
        except ValueError:
            self._contextual_strategy = ContextualStrategy.LIN_UCB
        
        # LinUCB参数
        self._lin_ucb_alpha = extra.get('lin_ucb_alpha', 1.0)
        self._regularization = extra.get('regularization', 1.0)
        
        # 每个臂的参数
        self._arm_A: Dict[str, np.ndarray] = {}  # A矩阵
        self._arm_b: Dict[str, np.ndarray] = {}  # b向量
        self._arm_theta: Dict[str, np.ndarray] = {}  # θ参数
        
        # 历史上下文
        self._context_history: List[Tuple[str, np.ndarray, float]] = []
        
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成上下文相关的建议"""
        import time
        start_time = time.time()
        
        # 提取上下文特征
        context_features = self._extract_context_features(context)
        context_array = np.array(context_features)
        
        if not self._initialized or self._context_dim == 0:
            self._context_dim = len(context_features)
            self.initialize(context)
            self._initialize_contextual_params()
        
        self._iteration_count += 1
        
        # 确保所有臂都初始化了参数
        for arm_id in self._arms:
            if arm_id not in self._arm_A:
                self._initialize_arm_params(arm_id)
        
        # 根据策略选择臂
        if self._contextual_strategy == ContextualStrategy.LIN_UCB:
            selected_arm, score, arm_scores = self._lin_ucb_select(context_array)
        elif self._contextual_strategy == ContextualStrategy.LIN_TS:
            selected_arm, score, arm_scores = self._lin_ts_select(context_array)
        else:
            selected_arm, score, arm_scores = self._linear_select(context_array)
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._build_result(
            action=selected_arm.action,
            confidence=min(0.95, 0.5 + selected_arm.n_pulls * 0.01),
            reasoning=f"上下文Bandit({self._contextual_strategy.value})：预测分数={score:.4f}",
            alternatives=self._generate_alternatives(exclude_arm=selected_arm),
            reasoning_steps=[
                f"策略: {self._contextual_strategy.value}",
                f"上下文维度: {self._context_dim}",
                f"选中臂: {selected_arm.arm_id}",
                f"预测分数: {score:.4f}",
                f"执行时间: {execution_time:.2f}ms"
            ],
            debug_info={
                'contextual_strategy': self._contextual_strategy.value,
                'context_dim': self._context_dim,
                'context_features': context_features,
                'arm_scores': arm_scores,
                'selected_score': score,
                'execution_time_ms': execution_time
            }
        )
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新上下文Bandit参数"""
        # 调用父类更新
        super().update(action, reward, context)
        
        # 更新上下文参数
        if context:
            context_features = self._extract_context_features(context)
            context_array = np.array(context_features)
            arm_id = self._action_to_arm_id(action)
            
            self._update_contextual_params(arm_id, context_array, reward)
            
            # 记录历史
            self._context_history.append((arm_id, context_array, reward))
    
    def _initialize_contextual_params(self) -> None:
        """初始化上下文参数"""
        for arm_id in self._arms:
            self._initialize_arm_params(arm_id)
    
    def _initialize_arm_params(self, arm_id: str) -> None:
        """初始化单个臂的参数"""
        d = self._context_dim
        self._arm_A[arm_id] = self._regularization * np.eye(d)
        self._arm_b[arm_id] = np.zeros(d)
        self._arm_theta[arm_id] = np.zeros(d)
    
    def _update_contextual_params(
        self, 
        arm_id: str, 
        context: np.ndarray, 
        reward: float
    ) -> None:
        """更新上下文参数
        
        LinUCB/LinTS的参数更新
        """
        if arm_id not in self._arm_A:
            self._initialize_arm_params(arm_id)
        
        # 更新A矩阵: A = A + x * x^T
        self._arm_A[arm_id] += np.outer(context, context)
        
        # 更新b向量: b = b + r * x
        self._arm_b[arm_id] += reward * context
        
        # 更新θ参数: θ = A^{-1} * b
        try:
            A_inv = np.linalg.inv(self._arm_A[arm_id])
            self._arm_theta[arm_id] = A_inv @ self._arm_b[arm_id]
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            self._arm_theta[arm_id] = np.linalg.pinv(self._arm_A[arm_id]) @ self._arm_b[arm_id]
    
    def _linear_select(
        self, 
        context: np.ndarray
    ) -> Tuple[ArmStatistics, float, Dict[str, float]]:
        """简单线性模型选择"""
        best_arm = None
        best_score = float('-inf')
        arm_scores = {}
        
        for arm_id, arm in self.active_arms.items():
            if arm_id not in self._arm_theta:
                self._initialize_arm_params(arm_id)
            
            # 线性预测
            score = context @ self._arm_theta[arm_id]
            arm_scores[arm_id] = float(score)
            
            if score > best_score:
                best_score = score
                best_arm = arm
        
        return best_arm, best_score, arm_scores
    
    def _lin_ucb_select(
        self, 
        context: np.ndarray
    ) -> Tuple[ArmStatistics, float, Dict[str, float]]:
        """LinUCB选择
        
        UCB = θ^T * x + α * sqrt(x^T * A^{-1} * x)
        """
        alpha = self._lin_ucb_alpha
        best_arm = None
        best_ucb = float('-inf')
        arm_scores = {}
        
        for arm_id, arm in self.active_arms.items():
            if arm_id not in self._arm_A:
                self._initialize_arm_params(arm_id)
            
            theta = self._arm_theta[arm_id]
            
            # 计算置信上界
            try:
                A_inv = np.linalg.inv(self._arm_A[arm_id])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self._arm_A[arm_id])
            
            # 预测均值
            mean = context @ theta
            
            # 置信宽度
            confidence_width = alpha * math.sqrt(context @ A_inv @ context)
            
            ucb = mean + confidence_width
            arm_scores[arm_id] = float(ucb)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        
        return best_arm, best_ucb, arm_scores
    
    def _lin_ts_select(
        self, 
        context: np.ndarray
    ) -> Tuple[ArmStatistics, float, Dict[str, float]]:
        """LinTS选择（线性Thompson Sampling）
        
        从参数后验分布采样
        """
        best_arm = None
        best_sample = float('-inf')
        arm_scores = {}
        
        for arm_id, arm in self.active_arms.items():
            if arm_id not in self._arm_A:
                self._initialize_arm_params(arm_id)
            
            theta = self._arm_theta[arm_id]
            
            # 计算协方差
            try:
                A_inv = np.linalg.inv(self._arm_A[arm_id])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self._arm_A[arm_id])
            
            # 从后验分布采样θ
            try:
                sampled_theta = np.random.multivariate_normal(theta, A_inv)
            except np.linalg.LinAlgError:
                sampled_theta = theta + np.random.randn(len(theta)) * 0.1
            
            # 计算预测奖励
            sample = context @ sampled_theta
            arm_scores[arm_id] = float(sample)
            
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        
        return best_arm, best_sample, arm_scores
    
    def _extract_context_features(self, context: AlgorithmContext) -> List[float]:
        """提取上下文特征"""
        features = []
        
        # 从inputs提取数值特征
        for key, value in sorted(context.inputs.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # 简单哈希编码（可以替换为更好的编码方式）
                features.append(hash(value) % 1000 / 1000.0)
        
        # 从constraints提取特征
        for key, value in sorted(context.constraints.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
        
        # 确保至少有一个特征
        if not features:
            features = [1.0]
        
        # 特征归一化
        features = self._normalize_features(features)
        
        return features
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """特征归一化"""
        if not features:
            return features
        
        # 简单的标准化
        mean = sum(features) / len(features)
        std = (sum((f - mean) ** 2 for f in features) / len(features)) ** 0.5
        
        if std > 0:
            return [(f - mean) / std for f in features]
        return features
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态（包含上下文参数）"""
        state = super().get_state()
        
        # 添加上下文参数
        state['contextual'] = {
            'strategy': self._contextual_strategy.value,
            'context_dim': self._context_dim,
            'arm_theta': {k: v.tolist() for k, v in self._arm_theta.items()},
            'history_size': len(self._context_history)
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复（包含上下文参数）"""
        super().load_state(state)
        
        contextual = state.get('contextual', {})
        self._context_dim = contextual.get('context_dim', 0)
        
        # 恢复θ参数
        arm_theta = contextual.get('arm_theta', {})
        for arm_id, theta_list in arm_theta.items():
            self._arm_theta[arm_id] = np.array(theta_list)
            # 重新初始化A和b（简化处理）
            if arm_id not in self._arm_A:
                self._initialize_arm_params(arm_id)


class HybridBandit(MultiArmedBanditAlgorithm):
    """混合Bandit策略
    
    动态切换多种策略，根据性能自动选择最佳策略
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        # 可用策略
        self._available_strategies = [
            BanditStrategy.UCB1,
            BanditStrategy.UCB_TUNED,
            BanditStrategy.THOMPSON_BETA,
            BanditStrategy.EPSILON_GREEDY
        ]
        
        # 策略性能追踪
        self._strategy_rewards: Dict[str, List[float]] = {
            s.value: [] for s in self._available_strategies
        }
        self._strategy_counts: Dict[str, int] = {
            s.value: 0 for s in self._available_strategies
        }
        
        # 策略切换参数
        extra = config.extra if config and config.extra else {}
        self._strategy_window = extra.get('strategy_window', 50)
        self._strategy_switch_threshold = extra.get('strategy_switch_threshold', 0.1)
        
        self._current_strategy_index = 0
        self._last_strategy_switch = 0
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成建议，可能切换策略"""
        # 每隔一定步数评估是否切换策略
        if self._total_pulls > 0 and self._total_pulls % self._strategy_window == 0:
            self._evaluate_and_switch_strategy()
        
        # 使用当前策略
        self._strategy = self._available_strategies[self._current_strategy_index]
        
        return super().suggest(context)
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新并记录策略性能"""
        super().update(action, reward, context)
        
        # 记录当前策略的奖励
        current_strategy = self._available_strategies[self._current_strategy_index].value
        self._strategy_rewards[current_strategy].append(reward)
        self._strategy_counts[current_strategy] += 1
    
    def _evaluate_and_switch_strategy(self) -> None:
        """评估策略性能并决定是否切换"""
        best_strategy_idx = self._current_strategy_index
        best_mean = -float('inf')
        
        for i, strategy in enumerate(self._available_strategies):
            rewards = self._strategy_rewards[strategy.value]
            if len(rewards) >= 10:
                # 只看最近的奖励
                recent_rewards = rewards[-self._strategy_window:]
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                
                if mean_reward > best_mean:
                    best_mean = mean_reward
                    best_strategy_idx = i
        
        # 如果最佳策略不是当前策略，且差距超过阈值，则切换
        current_rewards = self._strategy_rewards[
            self._available_strategies[self._current_strategy_index].value
        ]
        if len(current_rewards) >= 10:
            current_mean = sum(current_rewards[-self._strategy_window:]) / min(
                len(current_rewards), self._strategy_window
            )
            
            if best_mean - current_mean > self._strategy_switch_threshold:
                old_strategy = self._available_strategies[self._current_strategy_index].value
                self._current_strategy_index = best_strategy_idx
                new_strategy = self._available_strategies[self._current_strategy_index].value
                
                self.logger.info(
                    f"Strategy switched: {old_strategy} -> {new_strategy} "
                    f"(performance gain: {best_mean - current_mean:.4f})"
                )
                self._last_strategy_switch = self._total_pulls
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """获取各策略性能"""
        performance = {}
        
        for strategy in self._available_strategies:
            rewards = self._strategy_rewards[strategy.value]
            count = self._strategy_counts[strategy.value]
            
            if rewards:
                performance[strategy.value] = {
                    'count': count,
                    'total_reward': sum(rewards),
                    'mean_reward': sum(rewards) / len(rewards),
                    'recent_mean': sum(rewards[-50:]) / min(len(rewards), 50)
                }
            else:
                performance[strategy.value] = {
                    'count': 0,
                    'total_reward': 0,
                    'mean_reward': 0,
                    'recent_mean': 0
                }
        
        performance['current_strategy'] = self._available_strategies[
            self._current_strategy_index
        ].value
        
        return performance
