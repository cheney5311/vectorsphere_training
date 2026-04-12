"""
算法工厂 - 生产级实现

提供统一的算法创建、管理和调度接口。

核心功能：
- 算法注册和变体管理
- 智能场景映射（自动选择最佳算法）
- 动态配置生成
- 算法组合/集成执行
- 性能监控和统计
- 批量执行和并行处理
- 状态管理（保存/恢复）
- 算法切换策略

使用示例：
    # 基础使用
    algo = AlgorithmFactory.create(AlgorithmType.BAYESIAN_OPTIMIZATION)
    result = algo.suggest(context)
    
    # 场景驱动
    algo = AlgorithmFactory.create_from_scenario('hyperparameter_optimization', context)
    
    # 智能选择
    algo = AlgorithmFactory.auto_select(context)
    
    # 集成执行
    ensemble = AlgorithmFactory.create_ensemble(
        [AlgorithmType.BAYESIAN_OPTIMIZATION, AlgorithmType.GENETIC_ALGORITHM]
    )
    result = ensemble.suggest(context)
"""

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Type, List, Tuple

from .base import (
    BaseAlgorithm,
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext,
    AlgorithmResult
)
from .bayesian_optimization import BayesianOptimizationAlgorithm
from .genetic_algorithm import (
    GeneticAlgorithm,
    DifferentialEvolution,
    EvolutionStrategy,
    NSGAII
)
from .knowledge_reasoning import KnowledgeReasoningEngine
from .multi_armed_bandit import (
    MultiArmedBanditAlgorithm,
    ContextualBandit,
    HybridBandit
)
from .reinforcement_learning import (
    ReinforcementLearningAlgorithm,
    PolicyGradient,
    ActorCritic
)

logger = logging.getLogger(__name__)


# ==================== 枚举和配置 ====================

class SelectionStrategy(Enum):
    """算法选择策略"""
    FIXED = "fixed"                      # 固定算法
    SCENARIO_BASED = "scenario_based"    # 基于场景
    CONTEXT_AWARE = "context_aware"      # 上下文感知
    PERFORMANCE_BASED = "performance_based"  # 基于历史性能
    ADAPTIVE = "adaptive"                # 自适应选择


class EnsembleMethod(Enum):
    """集成方法"""
    VOTING = "voting"                    # 投票法
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    BEST_CONFIDENCE = "best_confidence"  # 最高置信度
    STACKING = "stacking"                # 堆叠
    CASCADE = "cascade"                  # 级联


@dataclass
class AlgorithmPerformance:
    """算法性能统计"""
    algorithm_type: str
    variant: str = ""
    total_calls: int = 0
    successful_calls: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    avg_confidence: float = 0.0
    avg_reward: float = 0.0
    best_reward: float = float('-inf')
    worst_reward: float = float('inf')
    last_used: Optional[datetime] = None
    scenarios: Dict[str, int] = field(default_factory=dict)
    
    def update(self, execution_time_ms: float, confidence: float, 
               reward: Optional[float] = None, scenario: str = None) -> None:
        """更新性能统计"""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_execution_time_ms += execution_time_ms
        self.avg_execution_time_ms = self.total_execution_time_ms / self.total_calls
        
        # 增量更新平均置信度
        self.avg_confidence = (
            (self.avg_confidence * (self.total_calls - 1) + confidence) / self.total_calls
        )
        
        if reward is not None:
            self.avg_reward = (
                (self.avg_reward * (self.total_calls - 1) + reward) / self.total_calls
            )
            self.best_reward = max(self.best_reward, reward)
            self.worst_reward = min(self.worst_reward, reward)
        
        self.last_used = datetime.now()
        
        if scenario:
            self.scenarios[scenario] = self.scenarios.get(scenario, 0) + 1


    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm_type': self.algorithm_type,
            'variant': self.variant,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'avg_confidence': self.avg_confidence,
            'avg_reward': self.avg_reward,
            'best_reward': self.best_reward,
            'scenarios': self.scenarios,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class AlgorithmEntry:
    """算法注册条目"""
    algorithm_class: Type[BaseAlgorithm]
    algorithm_type: AlgorithmType
    name: str
    description: str
    is_variant: bool = False
    parent_type: Optional[AlgorithmType] = None
    supported_scenarios: List[str] = field(default_factory=list)
    default_config: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


# ==================== 算法工厂 ====================

class AlgorithmFactory:
    """算法工厂 - 生产级实现
    
    负责创建和管理各种决策算法实例，提供智能调度能力。
    """
    
    # 注册的算法类
    _algorithms: Dict[AlgorithmType, AlgorithmEntry] = {}
    
    # 算法变体
    _variants: Dict[str, AlgorithmEntry] = {}
    
    # 缓存的算法实例
    _instances: Dict[str, BaseAlgorithm] = {}
    
    # 性能统计
    _performance_stats: Dict[str, AlgorithmPerformance] = {}
    
    # 线程锁
    _lock = threading.RLock()
    
    # 初始化标志
    _initialized = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """确保工厂已初始化"""
        if cls._initialized:
            return
        
        with cls._lock:
            if cls._initialized:
                return
            
            cls._register_default_algorithms()
            cls._initialized = True
    
    @classmethod
    def _register_default_algorithms(cls) -> None:
        """注册默认算法"""
        
        # 贝叶斯优化
        cls._algorithms[AlgorithmType.BAYESIAN_OPTIMIZATION] = AlgorithmEntry(
            algorithm_class=BayesianOptimizationAlgorithm,
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            name="贝叶斯优化",
            description="使用高斯过程的黑盒优化算法，适用于昂贵的函数评估",
            supported_scenarios=[
                'hyperparameter_optimization', 
                'data_preprocessing',
                'model_tuning',
                'expensive_optimization'
            ],
            default_config={
                'acquisition_function': 'expected_improvement',
                'n_initial_points': 5,
                'kernel': 'matern52'
            },
            capabilities=[
                'continuous_optimization',
                'expensive_function',
                'uncertainty_quantification',
                'batch_suggestion'
            ]
        )
        
        # 多臂老虎机
        cls._algorithms[AlgorithmType.MULTI_ARMED_BANDIT] = AlgorithmEntry(
            algorithm_class=MultiArmedBanditAlgorithm,
            algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
            name="多臂老虎机",
            description="探索-利用权衡的在线学习算法",
            supported_scenarios=[
                'model_selection',
                'hyperparameter_init',
                'ab_testing',
                'recommendation'
            ],
            default_config={
                'strategy': 'ucb1',
                'epsilon': 0.1
            },
            capabilities=[
                'online_learning',
                'exploration_exploitation',
                'discrete_actions',
                'regret_minimization'
            ]
        )
        
        # 强化学习
        cls._algorithms[AlgorithmType.REINFORCEMENT_LEARNING] = AlgorithmEntry(
            algorithm_class=ReinforcementLearningAlgorithm,
            algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
            name="强化学习",
            description="序贯决策和长期规划的学习算法",
            supported_scenarios=[
                'training_strategy',
                'resource_scheduling',
                'sequential_decision',
                'control_optimization'
            ],
            default_config={
                'method': 'q_learning',
                'exploration': 'epsilon_decay',
                'learning_rate': 0.01
            },
            capabilities=[
                'sequential_decision',
                'delayed_reward',
                'policy_learning',
                'state_value_estimation'
            ]
        )
        
        # 遗传算法
        cls._algorithms[AlgorithmType.GENETIC_ALGORITHM] = AlgorithmEntry(
            algorithm_class=GeneticAlgorithm,
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            name="遗传算法",
            description="基于进化的全局优化算法",
            supported_scenarios=[
                'resource_allocation',
                'feature_selection',
                'architecture_search',
                'combinatorial_optimization'
            ],
            default_config={
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'selection_method': 'tournament'
            },
            capabilities=[
                'global_optimization',
                'multi_objective',
                'combinatorial',
                'parallel_evaluation'
            ]
        )
        
        # 知识推理
        cls._algorithms[AlgorithmType.KNOWLEDGE_REASONING] = AlgorithmEntry(
            algorithm_class=KnowledgeReasoningEngine,
            algorithm_type=AlgorithmType.KNOWLEDGE_REASONING,
            name="知识推理",
            description="基于知识图谱和规则的智能推理引擎",
            supported_scenarios=[
                'model_architecture',
                'feature_engineering',
                'problem_diagnosis',
                'expert_recommendation'
            ],
            default_config={
                'inference_depth': 3,
                'confidence_threshold': 0.6
            },
            capabilities=[
                'rule_based_reasoning',
                'knowledge_graph',
                'explainable',
                'domain_knowledge'
            ]
        )
        
        # 注册变体
        cls._register_default_variants()
    
    @classmethod
    def _register_default_variants(cls) -> None:
        """注册默认变体"""
        
        # 上下文多臂老虎机
        cls._variants['contextual_bandit'] = AlgorithmEntry(
            algorithm_class=ContextualBandit,
            algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
            name="上下文多臂老虎机",
            description="考虑上下文信息的多臂老虎机变体",
            is_variant=True,
            parent_type=AlgorithmType.MULTI_ARMED_BANDIT,
            supported_scenarios=['personalized_recommendation', 'contextual_optimization'],
            capabilities=['context_aware', 'linear_model', 'online_learning']
        )
        
        # 混合多臂老虎机
        cls._variants['hybrid_bandit'] = AlgorithmEntry(
            algorithm_class=HybridBandit,
            algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
            name="混合多臂老虎机",
            description="自动切换策略的多臂老虎机",
            is_variant=True,
            parent_type=AlgorithmType.MULTI_ARMED_BANDIT,
            supported_scenarios=['adaptive_optimization'],
            capabilities=['adaptive_strategy', 'meta_learning']
        )
        
        # 策略梯度
        cls._variants['policy_gradient'] = AlgorithmEntry(
            algorithm_class=PolicyGradient,
            algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
            name="策略梯度",
            description="直接学习策略的强化学习方法",
            is_variant=True,
            parent_type=AlgorithmType.REINFORCEMENT_LEARNING,
            supported_scenarios=['continuous_control', 'policy_optimization'],
            capabilities=['continuous_action', 'policy_based']
        )
        
        # Actor-Critic
        cls._variants['actor_critic'] = AlgorithmEntry(
            algorithm_class=ActorCritic,
            algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
            name="Actor-Critic",
            description="结合策略和价值函数的强化学习方法",
            is_variant=True,
            parent_type=AlgorithmType.REINFORCEMENT_LEARNING,
            supported_scenarios=['continuous_control', 'stable_training'],
            capabilities=['actor_critic', 'variance_reduction']
        )
        
        # 差分进化
        cls._variants['differential_evolution'] = AlgorithmEntry(
            algorithm_class=DifferentialEvolution,
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            name="差分进化",
            description="基于差分变异的进化算法",
            is_variant=True,
            parent_type=AlgorithmType.GENETIC_ALGORITHM,
            supported_scenarios=['continuous_optimization', 'robust_optimization'],
            capabilities=['differential_mutation', 'self_adaptive']
        )
        
        # 进化策略
        cls._variants['evolution_strategy'] = AlgorithmEntry(
            algorithm_class=EvolutionStrategy,
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            name="进化策略",
            description="基于自适应变异的进化算法",
            is_variant=True,
            parent_type=AlgorithmType.GENETIC_ALGORITHM,
            supported_scenarios=['continuous_optimization', 'large_scale'],
            capabilities=['self_adaptation', 'elitist_selection']
        )
        
        # NSGA-II（多目标）
        cls._variants['nsga2'] = AlgorithmEntry(
            algorithm_class=NSGAII,
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            name="NSGA-II",
            description="非支配排序遗传算法，用于多目标优化",
            is_variant=True,
            parent_type=AlgorithmType.GENETIC_ALGORITHM,
            supported_scenarios=['multi_objective', 'pareto_optimization'],
            capabilities=['multi_objective', 'pareto_front', 'crowding_distance']
        )
    
    @classmethod
    def create(
        cls, 
        algorithm_type: AlgorithmType,
        config: Optional[AlgorithmConfig] = None,
        variant: Optional[str] = None,
        use_cache: bool = True,
        track_performance: bool = True
    ) -> BaseAlgorithm:
        """创建算法实例
        
        Args:
            algorithm_type: 算法类型
            config: 算法配置
            variant: 算法变体
            use_cache: 是否使用缓存
            track_performance: 是否跟踪性能
            
        Returns:
            算法实例
        """
        cls._ensure_initialized()
        
        # 生成缓存键
        config_hash = cls._hash_config(config) if config else "default"
        cache_key = f"{algorithm_type.value}:{variant or 'default'}:{config_hash}"
        
        with cls._lock:
            # 检查缓存
            if use_cache and cache_key in cls._instances:
                logger.debug("Using cached algorithm: %s", cache_key)
                return cls._instances[cache_key]
            
            # 确定算法类和配置
            if variant and variant in cls._variants:
                entry = cls._variants[variant]
                algo_class = entry.algorithm_class
                default_config = entry.default_config
            elif algorithm_type in cls._algorithms:
                entry = cls._algorithms[algorithm_type]
                algo_class = entry.algorithm_class
                default_config = entry.default_config
            else:
                logger.warning("Unknown algorithm type: %s, using KnowledgeReasoning", algorithm_type)
                algo_class = KnowledgeReasoningEngine
                default_config = {}
            
            # 创建配置
            if config is None:
                config = AlgorithmConfig(
                    algorithm_type=algorithm_type,
                    extra=default_config.copy()
                )
            else:
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config.extra:
                        config.extra[key] = value
            
            # 创建实例
            instance = algo_class(config)
            
            # 缓存
            if use_cache:
                cls._instances[cache_key] = instance
            
            # 初始化性能统计
            if track_performance:
                perf_key = f"{algorithm_type.value}:{variant or 'default'}"
                if perf_key not in cls._performance_stats:
                    cls._performance_stats[perf_key] = AlgorithmPerformance(
                        algorithm_type=algorithm_type.value,
                        variant=variant or ""
                    )
            
            logger.info("Created algorithm: %s (variant=%s)", algorithm_type.value, variant)
            return instance
    
    @classmethod
    def create_from_scenario(
        cls, 
        scenario: str,
        context: Optional[AlgorithmContext] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        prefer_variant: bool = True
    ) -> BaseAlgorithm:
        """根据场景创建算法
        
        Args:
            scenario: 决策场景
            context: 算法上下文
            extra_config: 额外配置
            prefer_variant: 是否优先使用变体
            
        Returns:
            算法实例
        """
        cls._ensure_initialized()
        
        # 场景到算法的映射
        scenario_mapping = {
            # 超参数和模型优化
            'hyperparameter_optimization': (AlgorithmType.BAYESIAN_OPTIMIZATION, None),
            'hyperparameter_init': (AlgorithmType.MULTI_ARMED_BANDIT, None),
            'model_tuning': (AlgorithmType.BAYESIAN_OPTIMIZATION, None),
            'model_selection': (AlgorithmType.MULTI_ARMED_BANDIT, None),
            
            # 架构和特征
            'model_architecture': (AlgorithmType.KNOWLEDGE_REASONING, None),
            'architecture_search': (AlgorithmType.GENETIC_ALGORITHM, None),
            'feature_engineering': (AlgorithmType.KNOWLEDGE_REASONING, None),
            'feature_selection': (AlgorithmType.GENETIC_ALGORITHM, None),
            
            # 资源和调度
            'resource_allocation': (AlgorithmType.GENETIC_ALGORITHM, None),
            'resource_scheduling': (AlgorithmType.REINFORCEMENT_LEARNING, None),
            
            # 训练策略
            'training_strategy': (AlgorithmType.REINFORCEMENT_LEARNING, None),
            'data_preprocessing': (AlgorithmType.BAYESIAN_OPTIMIZATION, None),
            
            # 多目标优化
            'multi_objective': (AlgorithmType.GENETIC_ALGORITHM, 'nsga2'),
            'pareto_optimization': (AlgorithmType.GENETIC_ALGORITHM, 'nsga2'),
            
            # 连续控制
            'continuous_control': (AlgorithmType.REINFORCEMENT_LEARNING, 'actor_critic'),
            'policy_optimization': (AlgorithmType.REINFORCEMENT_LEARNING, 'policy_gradient'),
            
            # 上下文感知
            'personalized_recommendation': (AlgorithmType.MULTI_ARMED_BANDIT, 'contextual_bandit'),
            'contextual_optimization': (AlgorithmType.MULTI_ARMED_BANDIT, 'contextual_bandit'),
            
            # 其他
            'ab_testing': (AlgorithmType.MULTI_ARMED_BANDIT, None),
            'recommendation': (AlgorithmType.MULTI_ARMED_BANDIT, 'contextual_bandit'),
            'expert_recommendation': (AlgorithmType.KNOWLEDGE_REASONING, None),
            'problem_diagnosis': (AlgorithmType.KNOWLEDGE_REASONING, None),
        }
        
        # 获取映射或使用默认
        mapping = scenario_mapping.get(scenario, (AlgorithmType.KNOWLEDGE_REASONING, None))
        algorithm_type, variant = mapping
        
        # 根据上下文调整选择
        if context and prefer_variant:
            variant = cls._refine_variant_selection(algorithm_type, variant, context)
        
        # 创建配置
        config = cls._create_config_for_scenario(scenario, algorithm_type, extra_config)
        
        # 创建算法
        algorithm = cls.create(algorithm_type, config, variant, use_cache=False)
        
        # 初始化
        if context:
            algorithm.initialize(context)
        
        return algorithm
    
    @classmethod
    def auto_select(
        cls, 
        context: AlgorithmContext,
        strategy: SelectionStrategy = SelectionStrategy.CONTEXT_AWARE,
        candidates: List[AlgorithmType] = None
    ) -> BaseAlgorithm:
        """智能选择最佳算法
        
        Args:
            context: 算法上下文
            strategy: 选择策略
            candidates: 候选算法类型
            
        Returns:
            最佳算法实例
        """
        cls._ensure_initialized()
        
        if candidates is None:
            candidates = list(cls._algorithms.keys())
        
        if strategy == SelectionStrategy.CONTEXT_AWARE:
            return cls._context_aware_select(context, candidates)
        elif strategy == SelectionStrategy.PERFORMANCE_BASED:
            return cls._performance_based_select(context, candidates)
        elif strategy == SelectionStrategy.ADAPTIVE:
            return cls._adaptive_select(context, candidates)
        else:
            # 默认使用知识推理
            return cls.create(AlgorithmType.KNOWLEDGE_REASONING)
    
    @classmethod
    def _context_aware_select(
        cls, 
        context: AlgorithmContext,
        candidates: List[AlgorithmType]
    ) -> BaseAlgorithm:
        """基于上下文特征选择算法"""
        scores = {}
        
        for algo_type in candidates:
            if algo_type not in cls._algorithms:
                continue
            
            entry = cls._algorithms[algo_type]
            score = 0.0
            
            # 基于搜索空间特征评分
            if context.search_space:
                n_params = len(context.search_space)
                has_continuous = any(
                    isinstance(v, dict) and 'low' in v 
                    for v in context.search_space.values()
                )
                has_discrete = any(
                    isinstance(v, dict) and 'choices' in v 
                    for v in context.search_space.values()
                )
                
                # 贝叶斯优化适合小维度连续优化
                if algo_type == AlgorithmType.BAYESIAN_OPTIMIZATION:
                    if has_continuous and n_params <= 10:
                        score += 5.0
                    elif n_params > 20:
                        score -= 3.0
                
                # 遗传算法适合大规模组合优化
                if algo_type == AlgorithmType.GENETIC_ALGORITHM:
                    if has_discrete or n_params > 10:
                        score += 4.0
                
                # 多臂老虎机适合少量离散选择
                if algo_type == AlgorithmType.MULTI_ARMED_BANDIT:
                    if has_discrete and n_params <= 3:
                        score += 5.0
            
            # 基于观测数量评分
            n_observations = len(context.observations)
            if algo_type == AlgorithmType.BAYESIAN_OPTIMIZATION:
                if n_observations < 5:
                    score += 3.0  # 样本效率高
            elif algo_type == AlgorithmType.GENETIC_ALGORITHM:
                if n_observations > 100:
                    score += 2.0  # 适合大量评估
            
            # 基于目标评分
            if context.objective == 'minimize':
                if 'continuous_optimization' in entry.capabilities:
                    score += 1.0
            
            # 基于约束评分
            if context.constraints:
                if algo_type in [AlgorithmType.GENETIC_ALGORITHM, 
                                AlgorithmType.BAYESIAN_OPTIMIZATION]:
                    score += 2.0  # 支持约束处理
            
            scores[algo_type] = score
        
        # 选择得分最高的算法
        if scores:
            best_type = max(scores, key=scores.get)
            logger.info("Auto-selected algorithm: %s (score=%.2f)", best_type.value, scores[best_type])
            return cls.create(best_type)
        
        return cls.create(AlgorithmType.KNOWLEDGE_REASONING)
    
    @classmethod
    def _performance_based_select(
        cls, 
        context: AlgorithmContext,
        candidates: List[AlgorithmType]
    ) -> BaseAlgorithm:
        """基于历史性能选择算法"""
        best_type = None
        best_score = float('-inf')
        
        for algo_type in candidates:
            perf_key = f"{algo_type.value}:default"
            if perf_key in cls._performance_stats:
                perf = cls._performance_stats[perf_key]
                # 综合评分：平均奖励 + 置信度 - 执行时间惩罚
                score = (
                    perf.avg_reward * 0.5 + 
                    perf.avg_confidence * 0.3 - 
                    perf.avg_execution_time_ms * 0.0001
                )
                if score > best_score:
                    best_score = score
                    best_type = algo_type
        
        if best_type:
            return cls.create(best_type)
        
        # 如果没有历史数据，回退到上下文感知选择
        return cls._context_aware_select(context, candidates)
    
    @classmethod
    def _adaptive_select(
        cls, 
        context: AlgorithmContext,
        candidates: List[AlgorithmType]
    ) -> BaseAlgorithm:
        """自适应选择（结合上下文和性能）"""
        # 综合上下文分析和历史性能
        context_scores = {}
        perf_scores = {}
        
        # 获取上下文分数
        for algo_type in candidates:
            if algo_type in cls._algorithms:
                context_scores[algo_type] = cls._compute_context_score(context, algo_type)
        
        # 获取性能分数
        for algo_type in candidates:
            perf_key = f"{algo_type.value}:default"
            if perf_key in cls._performance_stats:
                perf = cls._performance_stats[perf_key]
                perf_scores[algo_type] = perf.avg_reward * perf.avg_confidence
            else:
                perf_scores[algo_type] = 0.5  # 默认分数
        
        # 综合评分
        final_scores = {}
        for algo_type in candidates:
            cs = context_scores.get(algo_type, 0)
            ps = perf_scores.get(algo_type, 0.5)
            # 加权平均，性能权重随时间增加
            n_calls = cls._performance_stats.get(
                f"{algo_type.value}:default", 
                AlgorithmPerformance(algorithm_type=algo_type.value)
            ).total_calls
            perf_weight = min(0.7, 0.3 + n_calls * 0.01)
            final_scores[algo_type] = cs * (1 - perf_weight) + ps * perf_weight
        
        if final_scores:
            best_type = max(final_scores, key=final_scores.get)
            return cls.create(best_type)
        
        return cls.create(AlgorithmType.KNOWLEDGE_REASONING)
    
    @classmethod
    def _compute_context_score(cls, context: AlgorithmContext, 
                               algo_type: AlgorithmType) -> float:
        """计算上下文适配分数"""
        score = 0.5  # 基础分
        entry = cls._algorithms.get(algo_type)
        if not entry:
            return score
        
        # 搜索空间分析
        if context.search_space:
            n_params = len(context.search_space)
            
            if algo_type == AlgorithmType.BAYESIAN_OPTIMIZATION:
                score += max(0, (10 - n_params) * 0.05)
            elif algo_type == AlgorithmType.GENETIC_ALGORITHM:
                score += min(0.3, n_params * 0.02)
            elif algo_type == AlgorithmType.MULTI_ARMED_BANDIT:
                if n_params <= 3:
                    score += 0.3
        
        return min(1.0, max(0.0, score))
    
    @classmethod
    def _refine_variant_selection(
        cls, 
        algorithm_type: AlgorithmType,
        variant: Optional[str],
        context: AlgorithmContext
    ) -> Optional[str]:
        """根据上下文精化变体选择"""
        if algorithm_type == AlgorithmType.GENETIC_ALGORITHM:
            # 检查是否是多目标问题
            if context.metadata.get('multi_objective', False):
                return 'nsga2'
            # 检查是否需要差分进化
            if context.search_space and all(
                isinstance(v, dict) and 'low' in v
                for v in context.search_space.values()
            ):
                return 'differential_evolution'
        
        elif algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            # 检查是否需要连续动作
            if context.metadata.get('continuous_action', False):
                return 'actor_critic'
        
        elif algorithm_type == AlgorithmType.MULTI_ARMED_BANDIT:
            # 检查是否有上下文特征
            if context.inputs and len(context.inputs) > 3:
                return 'contextual_bandit'
        
        return variant
    
    @classmethod
    def _create_config_for_scenario(
        cls, 
        scenario: str, 
        algorithm_type: AlgorithmType,
        extra_config: Optional[Dict[str, Any]] = None
    ) -> AlgorithmConfig:
        """为场景创建优化配置"""
        config = AlgorithmConfig(algorithm_type=algorithm_type)
        
        # 场景特定配置
        scenario_configs = {
            'hyperparameter_optimization': {
                'acquisition_function': 'expected_improvement',
                'n_initial_points': 5,
                'exploration_weight': 0.05,
                'kernel': 'matern52',
                'optimize_hyperparams': True
            },
            'resource_allocation': {
                'population_size': 30,
                'mutation_rate': 0.15,
                'crossover_rate': 0.85,
                'selection_method': 'tournament',
                'elitism': True
            },
            'training_strategy': {
                'method': 'double_q_learning',
                'exploration': 'epsilon_decay',
                'learning_rate': 0.01,
                'discount_factor': 0.95,
                'use_replay_buffer': True
            },
            'model_selection': {
                'strategy': 'ucb1',
                'ucb_c': 2.5
            },
            'multi_objective': {
                'population_size': 100,
                'mutation_rate': 0.1
            },
            'continuous_control': {
                'method': 'actor_critic',
                'actor_lr': 0.001,
                'critic_lr': 0.01
            }
        }
        
        # 应用场景配置
        if scenario in scenario_configs:
            for key, value in scenario_configs[scenario].items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.extra[key] = value
        
        # 合并额外配置
        if extra_config:
            for key, value in extra_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.extra[key] = value
        
        return config
    
    @classmethod
    def create_ensemble(
        cls,
        algorithm_types: List[AlgorithmType],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        weights: List[float] = None,
        configs: Dict[AlgorithmType, AlgorithmConfig] = None
    ) -> 'AlgorithmEnsemble':
        """创建算法集成
        
        Args:
            algorithm_types: 算法类型列表
            method: 集成方法
            weights: 权重列表
            configs: 各算法配置
            
        Returns:
            算法集成实例
        """
        cls._ensure_initialized()
        
        algorithms = []
        for algo_type in algorithm_types:
            config = configs.get(algo_type) if configs else None
            algo = cls.create(algo_type, config, use_cache=False)
            algorithms.append(algo)
        
        return AlgorithmEnsemble(
            algorithms=algorithms,
            method=method,
            weights=weights
        )
    
    @classmethod
    def run_parallel(
        cls,
        algorithm_types: List[AlgorithmType],
        context: AlgorithmContext,
        max_workers: int = None
    ) -> Dict[str, AlgorithmResult]:
        """并行运行多个算法
        
        Args:
            algorithm_types: 算法类型列表
            context: 算法上下文
            max_workers: 最大工作线程数
            
        Returns:
            {算法类型: 结果} 字典
        """
        cls._ensure_initialized()
        
        results = {}
        
        def _run_single_algorithm(algo_type: AlgorithmType) -> Tuple[str, AlgorithmResult]:
            algo = cls.create(algo_type, use_cache=False)
            algo.initialize(context)
            start_time = time.time()
            result = algo.suggest(context)
            execution_time = (time.time() - start_time) * 1000
            
            # 更新性能统计
            cls._update_performance(algo_type, None, execution_time, result.confidence)
            
            return algo_type.value, result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_algorithm, algo_type): algo_type 
                for algo_type in algorithm_types
            }
            
            for future in as_completed(futures):
                try:
                    algo_name, result = future.result()
                    results[algo_name] = result
                except Exception as e:  # pylint: disable=broad-exception-caught
                    algo_type = futures[future]
                    logger.error("Algorithm %s failed: %s", algo_type.value, e)
                    results[algo_type.value] = None
        
        return results
    
    @classmethod
    def _update_performance(
        cls,
        algorithm_type: AlgorithmType,
        variant: Optional[str],
        execution_time_ms: float,
        confidence: float,
        reward: Optional[float] = None,
        scenario: str = None
    ) -> None:
        """更新算法性能统计"""
        perf_key = f"{algorithm_type.value}:{variant or 'default'}"
        
        with cls._lock:
            if perf_key not in cls._performance_stats:
                cls._performance_stats[perf_key] = AlgorithmPerformance(
                    algorithm_type=algorithm_type.value,
                    variant=variant or ""
                )
            
            cls._performance_stats[perf_key].update(
                execution_time_ms, confidence, reward, scenario
            )
    
    @classmethod
    def get_performance_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有算法的性能统计"""
        return {
            key: perf.to_dict() 
            for key, perf in cls._performance_stats.items()
        }
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """获取可用的算法列表"""
        cls._ensure_initialized()
        
        algorithms = {}
        
        for algo_type, entry in cls._algorithms.items():
            algorithms[algo_type.value] = {
                'name': entry.name,
                'class': entry.algorithm_class.__name__,
                'description': entry.description,
                'supported_scenarios': entry.supported_scenarios,
                'capabilities': entry.capabilities,
                'variants': []
            }
        
        # 添加变体信息
        for variant_name, entry in cls._variants.items():
            if entry.parent_type and entry.parent_type.value in algorithms:
                algorithms[entry.parent_type.value]['variants'].append({
                    'name': variant_name,
                    'display_name': entry.name,
                    'class': entry.algorithm_class.__name__,
                    'description': entry.description,
                    'supported_scenarios': entry.supported_scenarios
                })
        
        return algorithms
    
    @classmethod
    def clear_cache(cls) -> None:
        """清除算法缓存"""
        with cls._lock:
            cls._instances.clear()
        logger.info("Algorithm cache cleared")
    
    @classmethod
    def clear_performance_stats(cls) -> None:
        """清除性能统计"""
        with cls._lock:
            cls._performance_stats.clear()
        logger.info("Performance stats cleared")
    
    @classmethod
    def register_algorithm(
        cls, 
        algorithm_type: AlgorithmType,
        algorithm_class: Type[BaseAlgorithm],
        name: str = "",
        description: str = "",
        supported_scenarios: List[str] = None,
        capabilities: List[str] = None,
        default_config: Dict[str, Any] = None
    ) -> None:
        """注册新算法"""
        cls._ensure_initialized()
        
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"{algorithm_class} must be a subclass of BaseAlgorithm")
        
        with cls._lock:
            cls._algorithms[algorithm_type] = AlgorithmEntry(
                algorithm_class=algorithm_class,
                algorithm_type=algorithm_type,
                name=name or algorithm_class.__name__,
                description=description or algorithm_class.__doc__ or "",
                supported_scenarios=supported_scenarios or [],
                capabilities=capabilities or [],
                default_config=default_config or {}
            )
        
        logger.info("Registered algorithm: %s -> %s", algorithm_type.value, algorithm_class.__name__)
    
    @classmethod
    def register_variant(
        cls, 
        variant_name: str,
        variant_class: Type[BaseAlgorithm],
        parent_type: AlgorithmType,
        name: str = "",
        description: str = "",
        supported_scenarios: List[str] = None,
        capabilities: List[str] = None
    ) -> None:
        """注册算法变体"""
        cls._ensure_initialized()
        
        if not issubclass(variant_class, BaseAlgorithm):
            raise ValueError(f"{variant_class} must be a subclass of BaseAlgorithm")
        
        with cls._lock:
            cls._variants[variant_name] = AlgorithmEntry(
                algorithm_class=variant_class,
                algorithm_type=parent_type,
                name=name or variant_class.__name__,
                description=description or variant_class.__doc__ or "",
                is_variant=True,
                parent_type=parent_type,
                supported_scenarios=supported_scenarios or [],
                capabilities=capabilities or []
            )
        
        logger.info("Registered variant: %s -> %s", variant_name, variant_class.__name__)
    
    @classmethod
    def _hash_config(cls, config: AlgorithmConfig) -> str:
        """生成配置哈希"""
        config_dict = {
            'type': config.algorithm_type.value,
            'max_iter': config.max_iterations,
            'acq_func': config.acquisition_function,
            'epsilon': config.epsilon,
            'pop_size': config.population_size,
            'extra': str(sorted(config.extra.items()))
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @classmethod
    def save_state(cls, filepath: str) -> None:
        """保存工厂状态"""
        state = {
            'performance_stats': {
                k: v.to_dict() for k, v in cls._performance_stats.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Factory state saved to %s", filepath)
    
    @classmethod
    def load_state(cls, filepath: str) -> None:
        """加载工厂状态"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            cls._ensure_initialized()
            
            with cls._lock:
                for key, perf_dict in state.get('performance_stats', {}).items():
                    cls._performance_stats[key] = AlgorithmPerformance(
                        algorithm_type=perf_dict['algorithm_type'],
                        variant=perf_dict.get('variant', ''),
                        total_calls=perf_dict.get('total_calls', 0),
                        successful_calls=perf_dict.get('successful_calls', 0),
                        total_execution_time_ms=perf_dict.get('avg_execution_time_ms', 0) * perf_dict.get('total_calls', 0),
                        avg_execution_time_ms=perf_dict.get('avg_execution_time_ms', 0),
                        avg_confidence=perf_dict.get('avg_confidence', 0),
                        avg_reward=perf_dict.get('avg_reward', 0),
                        best_reward=perf_dict.get('best_reward', float('-inf')),
                        scenarios=perf_dict.get('scenarios', {})
                    )
            
            logger.info("Factory state loaded from %s", filepath)
        except FileNotFoundError:
            logger.warning("State file not found: %s", filepath)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load state: %s", e)


# ==================== 算法集成 ====================

class AlgorithmEnsemble(BaseAlgorithm):
    """算法集成
    
    组合多个算法的结果以提高决策质量。
    """
    
    def __init__(
        self,
        algorithms: List[BaseAlgorithm],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        weights: List[float] = None,
        config: Optional[AlgorithmConfig] = None
    ):
        super().__init__(config)
        self._algorithms = algorithms
        self._method = method
        self._weights = weights or [1.0 / len(algorithms)] * len(algorithms)
        
        # 归一化权重
        total_weight = sum(self._weights)
        self._weights = [w / total_weight for w in self._weights]
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.ENSEMBLE
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化所有子算法"""
        super().initialize(context)
        for algo in self._algorithms:
            algo.initialize(context)
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """集成建议"""
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        
        # 收集所有算法的结果
        results = []
        for algo in self._algorithms:
            try:
                result = algo.suggest(context)
                results.append(result)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Algorithm %s failed: %s", algo.name, e)
        
        if not results:
            return self._build_result(
                action={},
                confidence=0.0,
                reasoning="所有子算法都失败了"
            )
        
        # 根据集成方法合并结果
        if self._method == EnsembleMethod.BEST_CONFIDENCE:
            return self._best_confidence_combine(results)
        elif self._method == EnsembleMethod.VOTING:
            return self._voting_combine(results)
        elif self._method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_combine(results)
        elif self._method == EnsembleMethod.CASCADE:
            return self._cascade_combine(results, context)
        else:
            return self._best_confidence_combine(results)
    
    def _best_confidence_combine(self, results: List[AlgorithmResult]) -> AlgorithmResult:
        """选择置信度最高的结果"""
        best_result = max(results, key=lambda r: r.confidence)
        
        reasoning_steps = [
            "集成方法: 最高置信度选择",
            f"子算法数量: {len(results)}",
            f"选中算法: {best_result.algorithm_type}",
            f"置信度: {best_result.confidence:.4f}"
        ]
        
        return self._build_result(
            action=best_result.recommended_action,
            confidence=best_result.confidence,
            reasoning=f"选择置信度最高的建议（来自 {best_result.algorithm_type}）",
            alternatives=best_result.alternatives,
            reasoning_steps=reasoning_steps,
            debug_info={
                'method': 'best_confidence',
                'selected_algorithm': best_result.algorithm_type,
                'all_confidences': [r.confidence for r in results]
            }
        )
    
    def _voting_combine(self, results: List[AlgorithmResult]) -> AlgorithmResult:
        """投票法合并"""
        # 对动作进行投票
        action_votes: Dict[str, int] = {}
        action_map: Dict[str, Dict[str, Any]] = {}
        
        for result in results:
            action_key = json.dumps(result.recommended_action, sort_keys=True)
            action_votes[action_key] = action_votes.get(action_key, 0) + 1
            action_map[action_key] = result.recommended_action
        
        # 选择票数最多的动作
        best_key = max(action_votes, key=action_votes.get)
        best_action = action_map[best_key]
        vote_ratio = action_votes[best_key] / len(results)
        
        return self._build_result(
            action=best_action,
            confidence=vote_ratio,
            reasoning=f"投票法选择：{action_votes[best_key]}/{len(results)} 票",
            reasoning_steps=[
                "集成方法: 投票",
                f"总票数: {len(results)}",
                f"获胜票数: {action_votes[best_key]}",
                f"投票比例: {vote_ratio:.2%}"
            ]
        )
    
    def _weighted_average_combine(self, results: List[AlgorithmResult]) -> AlgorithmResult:
        """加权平均合并（适用于数值型动作）"""
        # 尝试对数值型参数进行加权平均
        combined_action = {}
        total_confidence = 0.0
        
        # 获取所有参数名
        all_params = set()
        for result in results:
            all_params.update(result.recommended_action.keys())
        
        for param in all_params:
            values = []
            weights = []
            
            for i, result in enumerate(results):
                if param in result.recommended_action:
                    value = result.recommended_action[param]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        weights.append(self._weights[i] * result.confidence)
            
            if values and weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    combined_action[param] = sum(v * w for v, w in zip(values, weights)) / total_weight
                else:
                    combined_action[param] = values[0]
            elif results:
                # 非数值型，使用最高置信度的结果
                best_result = max(results, key=lambda r: r.confidence)
                if param in best_result.recommended_action:
                    combined_action[param] = best_result.recommended_action[param]
        
        # 计算总体置信度
        for i, result in enumerate(results):
            total_confidence += self._weights[i] * result.confidence
        
        return self._build_result(
            action=combined_action,
            confidence=total_confidence,
            reasoning="加权平均集成多个算法的建议",
            reasoning_steps=[
                f"集成方法: 加权平均",
                f"子算法数量: {len(results)}",
                f"权重: {self._weights}",
                f"综合置信度: {total_confidence:.4f}"
            ]
        )
    
    def _cascade_combine(
        self, 
        results: List[AlgorithmResult],
        context: AlgorithmContext
    ) -> AlgorithmResult:
        """级联合并（按置信度阈值筛选）"""
        threshold = 0.7
        
        # 按置信度排序
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        
        for result in sorted_results:
            if result.confidence >= threshold:
                return self._build_result(
                    action=result.recommended_action,
                    confidence=result.confidence,
                    reasoning=f"级联选择：{result.algorithm_type} 置信度达到阈值",
                    reasoning_steps=[
                        f"集成方法: 级联",
                        f"阈值: {threshold}",
                        f"选中: {result.algorithm_type}",
                        f"置信度: {result.confidence:.4f}"
                    ]
                )
        
        # 如果没有达到阈值，返回最高置信度的
        return self._best_confidence_combine(results)
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新所有子算法"""
        self._record_observation(action, reward)
        
        for algo in self._algorithms:
            try:
                algo.update(action, reward, context)
            except Exception as e:
                logger.warning("Failed to update %s: %s", algo.name, e)
    
    def get_sub_algorithms(self) -> List[BaseAlgorithm]:
        """获取子算法列表"""
        return self._algorithms


# ==================== 便捷函数 ====================

def get_algorithm(
    algorithm_type: str,
    config: Optional[Dict[str, Any]] = None,
    variant: Optional[str] = None
) -> BaseAlgorithm:
    """获取算法实例的便捷函数"""
    # 转换类型
    try:
        algo_type = AlgorithmType(algorithm_type)
    except ValueError:
        logger.warning("Unknown algorithm type: %s, using knowledge_reasoning", algorithm_type)
        algo_type = AlgorithmType.KNOWLEDGE_REASONING
    
    # 转换配置
    algo_config = None
    if config:
        algo_config = AlgorithmConfig(
            algorithm_type=algo_type,
            max_iterations=config.get('max_iterations', 100),
            convergence_threshold=config.get('convergence_threshold', 0.001),
            random_seed=config.get('random_seed'),
            acquisition_function=config.get('acquisition_function', 'expected_improvement'),
            exploration_weight=config.get('exploration_weight', 0.1),
            n_initial_points=config.get('n_initial_points', 5),
            epsilon=config.get('epsilon', 0.1),
            ucb_c=config.get('ucb_c', 2.0),
            thompson_alpha=config.get('thompson_alpha', 1.0),
            thompson_beta=config.get('thompson_beta', 1.0),
            learning_rate=config.get('learning_rate', 0.01),
            discount_factor=config.get('discount_factor', 0.99),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            min_epsilon=config.get('min_epsilon', 0.01),
            population_size=config.get('population_size', 50),
            mutation_rate=config.get('mutation_rate', 0.1),
            crossover_rate=config.get('crossover_rate', 0.8),
            elite_ratio=config.get('elite_ratio', 0.1),
            selection_method=config.get('selection_method', 'tournament'),
            inference_depth=config.get('inference_depth', 3),
            confidence_threshold=config.get('confidence_threshold', 0.6),
            extra=config.get('extra', {})
        )
    
    return AlgorithmFactory.create(algo_type, algo_config, variant)


def run_algorithm(
    algorithm_type: str,
    context: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """运行算法的便捷函数"""
    # 获取算法
    algorithm = get_algorithm(algorithm_type, config)
    
    # 构建上下文
    algo_context = AlgorithmContext(
        inputs=context.get('inputs', {}),
        constraints=context.get('constraints', {}),
        history=context.get('history', []),
        search_space=context.get('search_space', {}),
        objective=context.get('objective', 'maximize'),
        objective_metric=context.get('objective_metric', 'score'),
        observations=context.get('observations', []),
        metadata=context.get('metadata', {})
    )
    
    # 初始化并运行
    algorithm.initialize(algo_context)
    
    start_time = time.time()
    result = algorithm.suggest(algo_context)
    execution_time = (time.time() - start_time) * 1000
    
    # 更新性能统计
    try:
        algo_type = AlgorithmType(algorithm_type)
        AlgorithmFactory._update_performance(
            algo_type, None, execution_time, result.confidence
        )
    except ValueError:
        pass
    
    return result.to_dict()


def auto_run(
    context: Dict[str, Any],
    strategy: str = "context_aware"
) -> Dict[str, Any]:
    """自动选择并运行算法"""
    # 构建上下文
    algo_context = AlgorithmContext(
        inputs=context.get('inputs', {}),
        constraints=context.get('constraints', {}),
        history=context.get('history', []),
        search_space=context.get('search_space', {}),
        objective=context.get('objective', 'maximize'),
        objective_metric=context.get('objective_metric', 'score'),
        observations=context.get('observations', []),
        metadata=context.get('metadata', {})
    )
    
    # 转换策略
    try:
        sel_strategy = SelectionStrategy(strategy)
    except ValueError:
        sel_strategy = SelectionStrategy.CONTEXT_AWARE
    
    # 自动选择算法
    algorithm = AlgorithmFactory.auto_select(algo_context, sel_strategy)
    
    # 初始化并运行
    algorithm.initialize(algo_context)
    result = algorithm.suggest(algo_context)
    
    return result.to_dict()


def create_ensemble(
    algorithm_types: List[str],
    method: str = "weighted_average",
    weights: List[float] = None
) -> AlgorithmEnsemble:
    """创建算法集成"""
    types = []
    for t in algorithm_types:
        try:
            types.append(AlgorithmType(t))
        except ValueError:
            logger.warning("Unknown algorithm type: %s, skipping", t)
    
    if not types:
        types = [AlgorithmType.KNOWLEDGE_REASONING]
    
    try:
        ensemble_method = EnsembleMethod(method)
    except ValueError:
        ensemble_method = EnsembleMethod.WEIGHTED_AVERAGE
    
    return AlgorithmFactory.create_ensemble(types, ensemble_method, weights)
