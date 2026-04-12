"""
算法基础模块

定义所有决策算法的基类和通用数据结构。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """算法类型枚举"""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENETIC_ALGORITHM = "genetic_algorithm"
    KNOWLEDGE_REASONING = "knowledge_reasoning"
    RULE_BASED = "rule_based"
    ENSEMBLE = "ensemble"


@dataclass
class AlgorithmConfig:
    """算法配置"""
    algorithm_type: AlgorithmType
    
    # 通用配置
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    random_seed: Optional[int] = None
    
    # 贝叶斯优化配置
    acquisition_function: str = "expected_improvement"  # ei, ucb, poi
    exploration_weight: float = 0.1
    n_initial_points: int = 5
    
    # 多臂老虎机配置
    epsilon: float = 0.1  # epsilon-greedy
    ucb_c: float = 2.0    # UCB参数
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    
    # 强化学习配置
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    
    # 遗传算法配置
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    selection_method: str = "tournament"  # tournament, roulette, rank
    
    # 知识推理配置
    inference_depth: int = 3
    confidence_threshold: float = 0.6
    
    # 额外配置
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmContext:
    """算法执行上下文"""
    # 输入数据
    inputs: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # 历史数据
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 搜索空间定义
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # 目标函数信息
    objective: str = "maximize"  # maximize, minimize
    objective_metric: str = "score"
    
    # 已知的观测点
    observations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmResult:
    """算法执行结果"""
    # 基本信息
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    algorithm_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 推荐结果
    recommended_action: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # 推理过程
    reasoning: str = ""
    reasoning_steps: List[str] = field(default_factory=list)
    
    # 备选方案
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    # 算法统计
    iterations: int = 0
    convergence: bool = False
    improvement: float = 0.0
    
    # 调试信息
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'result_id': self.result_id,
            'algorithm_type': self.algorithm_type,
            'timestamp': self.timestamp.isoformat(),
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'reasoning_steps': self.reasoning_steps,
            'alternatives': self.alternatives,
            'iterations': self.iterations,
            'convergence': self.convergence,
            'improvement': self.improvement,
            'debug_info': self.debug_info
        }


class BaseAlgorithm(ABC):
    """算法基类
    
    所有决策算法必须继承此类并实现抽象方法。
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """初始化算法
        
        Args:
            config: 算法配置
        """
        self.config = config or AlgorithmConfig(algorithm_type=AlgorithmType.RULE_BASED)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        
        # 算法状态
        self._observations: List[Dict[str, Any]] = []
        self._best_result: Optional[Dict[str, Any]] = None
        self._iteration_count = 0
    
    @property
    @abstractmethod
    def algorithm_type(self) -> AlgorithmType:
        """返回算法类型"""
        pass
    
    @property
    def name(self) -> str:
        """返回算法名称"""
        return self.algorithm_type.value
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化算法状态
        
        Args:
            context: 算法上下文
        """
        self._observations = list(context.observations)
        self._best_result = None
        self._iteration_count = 0
        self._initialized = True
        self.logger.info(f"Algorithm {self.name} initialized with {len(self._observations)} observations")
    
    @abstractmethod
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成建议
        
        Args:
            context: 算法上下文
            
        Returns:
            算法结果
        """
        pass
    
    @abstractmethod
    def update(self, action: Dict[str, Any], reward: float, 
               context: Optional[AlgorithmContext] = None) -> None:
        """更新算法状态
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            context: 可选的上下文
        """
        pass
    
    def evaluate(self, action: Dict[str, Any], 
                context: AlgorithmContext) -> float:
        """评估动作的预期收益
        
        Args:
            action: 待评估的动作
            context: 算法上下文
            
        Returns:
            预期收益值
        """
        # 默认实现：返回0
        return 0.0
    
    def get_best(self) -> Optional[Dict[str, Any]]:
        """获取当前最佳结果
        
        Returns:
            最佳结果
        """
        return self._best_result
    
    def reset(self) -> None:
        """重置算法状态"""
        self._observations = []
        self._best_result = None
        self._iteration_count = 0
        self._initialized = False
        self.logger.info(f"Algorithm {self.name} reset")
    
    def _record_observation(self, action: Dict[str, Any], reward: float,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """记录观测
        
        Args:
            action: 动作
            reward: 奖励
            metadata: 元数据
        """
        observation = {
            'action': action,
            'reward': reward,
            'timestamp': datetime.utcnow().isoformat(),
            'iteration': self._iteration_count
        }
        if metadata:
            observation['metadata'] = metadata
        
        self._observations.append(observation)
        
        # 更新最佳结果
        if self._best_result is None or reward > self._best_result.get('reward', float('-inf')):
            self._best_result = observation
    
    def _build_result(self, action: Dict[str, Any], confidence: float,
                     reasoning: str, alternatives: List[Dict[str, Any]] = None,
                     reasoning_steps: List[str] = None,
                     debug_info: Dict[str, Any] = None) -> AlgorithmResult:
        """构建算法结果
        
        Args:
            action: 推荐动作
            confidence: 置信度
            reasoning: 推理说明
            alternatives: 备选方案
            reasoning_steps: 推理步骤
            debug_info: 调试信息
            
        Returns:
            算法结果
        """
        return AlgorithmResult(
            algorithm_type=self.algorithm_type.value,
            recommended_action=action,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives or [],
            reasoning_steps=reasoning_steps or [],
            iterations=self._iteration_count,
            convergence=self._check_convergence(),
            improvement=self._calculate_improvement(),
            debug_info=debug_info or {}
        )
    
    def _check_convergence(self) -> bool:
        """检查是否收敛
        
        Returns:
            是否收敛
        """
        if len(self._observations) < 5:
            return False
        
        # 检查最近几次观测的改进幅度
        recent = self._observations[-5:]
        rewards = [o.get('reward', 0) for o in recent]
        
        if len(rewards) < 2:
            return False
        
        # 计算标准差
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5
        
        return std < self.config.convergence_threshold
    
    def _calculate_improvement(self) -> float:
        """计算改进幅度
        
        Returns:
            改进幅度
        """
        if len(self._observations) < 2:
            return 0.0
        
        first_reward = self._observations[0].get('reward', 0)
        best_reward = self._best_result.get('reward', 0) if self._best_result else 0
        
        if first_reward == 0:
            return best_reward
        
        return (best_reward - first_reward) / abs(first_reward)

