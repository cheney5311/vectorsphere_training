"""
智能决策算法模块

提供生产级的智能化决策算法，包括：
- 贝叶斯优化 (Bayesian Optimization)
- 多臂老虎机 (Multi-Armed Bandit)
- 强化学习 (Reinforcement Learning)
- 遗传算法 (Genetic Algorithm)
- 知识推理 (Knowledge Reasoning)
- LangGraph Agent (多步骤推理、工具调用、循环控制)

所有算法实现统一的接口，支持可插拔式调用。
"""

from .base import (
    BaseAlgorithm,
    AlgorithmResult,
    AlgorithmConfig,
    AlgorithmContext,
    AlgorithmType
)
from .bayesian_optimization import BayesianOptimizationAlgorithm
from .multi_armed_bandit import MultiArmedBanditAlgorithm
from .reinforcement_learning import ReinforcementLearningAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .knowledge_reasoning import KnowledgeReasoningEngine
from .algorithm_factory import AlgorithmFactory, get_algorithm

# LangGraph 模块
from . import langgraph

__all__ = [
    # 基础类
    'BaseAlgorithm',
    'AlgorithmResult',
    'AlgorithmConfig',
    'AlgorithmContext',
    'AlgorithmType',
    
    # 具体算法
    'BayesianOptimizationAlgorithm',
    'MultiArmedBanditAlgorithm',
    'ReinforcementLearningAlgorithm',
    'GeneticAlgorithm',
    'KnowledgeReasoningEngine',
    
    # 工厂
    'AlgorithmFactory',
    'get_algorithm',
    
    # LangGraph
    'langgraph'
]

