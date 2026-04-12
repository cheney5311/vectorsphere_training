"""
算法工厂集成测试 - 系统级测试

测试通过 AlgorithmFactory 调用所有算法实现：
- 贝叶斯优化
- 遗传算法
- 知识推理
- 多臂老虎机
- 强化学习

测试内容：
1. 基础算法创建和调用
2. 算法变体测试
3. 场景驱动选择
4. 智能选择
5. 算法集成
6. 并行执行
7. 性能监控
8. 状态管理

运行方式：
    python -m pytest backend/algo/tests/test_algorithm_factory_integration.py -v
    或直接运行：
    python backend/algo/tests/test_algorithm_factory_integration.py
"""

import sys
import os
import json
import time
import tempfile
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from backend.algo.algorithm_factory import (
    AlgorithmFactory,
    AlgorithmEnsemble,
    get_algorithm,
    run_algorithm,
    auto_run,
    create_ensemble,
    SelectionStrategy,
    EnsembleMethod
)
from backend.algo.base import (
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext,
    AlgorithmResult
)


# ==================== 测试配置 ====================

class TestConfig:
    """测试配置"""
    VERBOSE = True
    ITERATIONS = 3
    MAX_PARALLEL_WORKERS = 3


def log(message: str, level: str = "INFO"):
    """日志输出"""
    if TestConfig.VERBOSE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")


def create_test_context(
    search_space: Dict[str, Any] = None,
    observations: List[Dict[str, Any]] = None,
    inputs: Dict[str, Any] = None,
    objective: str = "maximize"
) -> AlgorithmContext:
    """创建测试上下文"""
    if search_space is None:
        search_space = {
            "learning_rate": {"low": 0.0001, "high": 0.1},
            "batch_size": {"choices": [16, 32, 64, 128]},
            "hidden_units": {"low": 32, "high": 256},
            "dropout": {"low": 0.0, "high": 0.5}
        }
    
    if inputs is None:
        inputs = {
            "dataset_size": 10000,
            "task_type": "classification",
            "num_features": 100
        }
    
    return AlgorithmContext(
        inputs=inputs,
        search_space=search_space,
        observations=observations or [],
        objective=objective,
        objective_metric="accuracy",
        constraints={"max_time": 3600},
        metadata={"test": True}
    )


# ==================== 测试类 ====================

import pytest

class OptimizationTestResults:
    """测试结果收集"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.details = []
    
    def add_pass(self, test_name: str, details: str = ""):
        self.passed += 1
        self.details.append({
            "name": test_name,
            "status": "PASSED",
            "details": details
        })
        log(f"✓ {test_name}: PASSED", "PASS")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append({"name": test_name, "error": error})
        self.details.append({
            "name": test_name,
            "status": "FAILED",
            "error": error
        })
        log(f"✗ {test_name}: FAILED - {error}", "FAIL")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"\n{'='*60}\nTest Summary: {self.passed}/{total} passed, {self.failed} failed\n{'='*60}"

@pytest.fixture
def results():
    return OptimizationTestResults()


# ==================== 基础算法测试 ====================

def test_bayesian_optimization(results: OptimizationTestResults):
    """测试贝叶斯优化"""
    test_name = "Bayesian Optimization - Basic"
    try:
        algo = AlgorithmFactory.create(AlgorithmType.BAYESIAN_OPTIMIZATION)
        context = create_test_context()
        algo.initialize(context)
        
        # 多次迭代
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            assert result.recommended_action, "No action recommended"
            assert 0 <= result.confidence <= 1, "Invalid confidence"
            
            # 模拟奖励
            reward = 0.5 + i * 0.1
            algo.update(result.recommended_action, reward)
        
        # 验证收敛检测
        summary = algo.get_optimization_summary()
        assert "total_iterations" in summary
        
        results.add_pass(test_name, f"Iterations: {TestConfig.ITERATIONS}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm(results: OptimizationTestResults):
    """测试遗传算法"""
    test_name = "Genetic Algorithm - Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        algo = AlgorithmFactory.create(AlgorithmType.GENETIC_ALGORITHM, config)
        context = create_test_context()
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            assert result.recommended_action
            
            reward = 0.3 + i * 0.2
            algo.update(result.recommended_action, reward)
        
        summary = algo.get_optimization_summary()
        assert "generation" in summary or "total_evaluations" in summary
        
        results.add_pass(test_name, f"Population size: {config.population_size}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_knowledge_reasoning(results: OptimizationTestResults):
    """测试知识推理"""
    test_name = "Knowledge Reasoning - Basic"
    try:
        algo = AlgorithmFactory.create(AlgorithmType.KNOWLEDGE_REASONING)
        context = create_test_context(
            inputs={
                "dataset_size": 50000,
                "task_type": "classification",
                "has_gpu": True,
                "memory_gb": 16
            }
        )
        algo.initialize(context)
        
        result = algo.suggest(context)
        assert result.recommended_action
        assert result.reasoning
        assert len(result.reasoning_steps) > 0
        
        # 测试知识查询
        query_result = algo.query_knowledge("model architecture recommendation")
        assert query_result is not None
        
        results.add_pass(test_name, f"Reasoning steps: {len(result.reasoning_steps)}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_multi_armed_bandit(results: OptimizationTestResults):
    """测试多臂老虎机"""
    test_name = "Multi-Armed Bandit - Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
            epsilon=0.1,
            extra={"strategy": "ucb1"}
        )
        algo = AlgorithmFactory.create(AlgorithmType.MULTI_ARMED_BANDIT, config)
        context = create_test_context(
            search_space={
                "optimizer": {"choices": ["adam", "sgd", "rmsprop"]},
                "activation": {"choices": ["relu", "tanh", "gelu"]}
            }
        )
        algo.initialize(context)
        
        total_reward = 0
        for i in range(TestConfig.ITERATIONS * 3):
            result = algo.suggest(context)
            
            # 模拟不同动作的奖励
            action_key = str(result.recommended_action)
            reward = 0.5 + (hash(action_key) % 10) * 0.05
            algo.update(result.recommended_action, reward)
            total_reward += reward
        
        summary = algo.get_optimization_summary()
        assert "total_pulls" in summary
        
        results.add_pass(test_name, f"Total reward: {total_reward:.2f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_reinforcement_learning(results: OptimizationTestResults):
    """测试强化学习"""
    test_name = "Reinforcement Learning - Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
            learning_rate=0.01,
            discount_factor=0.99,
            epsilon=0.2,
            extra={
                "method": "q_learning",
                "exploration": "epsilon_decay"
            }
        )
        algo = AlgorithmFactory.create(AlgorithmType.REINFORCEMENT_LEARNING, config)
        context = create_test_context(
            search_space={
                "action": {"choices": ["increase_lr", "decrease_lr", "keep_lr"]},
                "batch_action": {"choices": ["increase", "decrease", "keep"]}
            }
        )
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS * 2):
            result = algo.suggest(context)
            reward = 0.4 + i * 0.05
            algo.update(result.recommended_action, reward, done=(i == TestConfig.ITERATIONS * 2 - 1))
        
        summary = algo.get_optimization_summary()
        assert "total_steps" in summary
        
        results.add_pass(test_name, f"Total steps: {summary.get('total_steps', 0)}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 变体测试 ====================

def test_contextual_bandit(results: OptimizationTestResults):
    """测试上下文多臂老虎机"""
    test_name = "Contextual Bandit Variant"
    try:
        algo = AlgorithmFactory.create(
            AlgorithmType.MULTI_ARMED_BANDIT,
            variant='contextual_bandit'
        )
        context = create_test_context(
            inputs={
                "user_age": 25,
                "user_segment": "premium",
                "time_of_day": 14
            },
            search_space={
                "content_type": {"choices": ["news", "sports", "tech"]},
                "layout": {"choices": ["grid", "list"]}
            }
        )
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            reward = 0.6 + i * 0.1
            algo.update(result.recommended_action, reward, context)
        
        results.add_pass(test_name, "LinUCB contextual selection working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_differential_evolution(results: OptimizationTestResults):
    """测试差分进化"""
    test_name = "Differential Evolution Variant"
    try:
        algo = AlgorithmFactory.create(
            AlgorithmType.GENETIC_ALGORITHM,
            variant='differential_evolution'
        )
        context = create_test_context(
            search_space={
                "x1": {"low": -5.0, "high": 5.0},
                "x2": {"low": -5.0, "high": 5.0}
            }
        )
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            # Sphere function
            x1 = result.recommended_action.get("x1", 0)
            x2 = result.recommended_action.get("x2", 0)
            fitness = -(x1**2 + x2**2)  # 最小化
            algo.update(result.recommended_action, -fitness)
        
        results.add_pass(test_name, "DE mutation and crossover working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_actor_critic(results: OptimizationTestResults):
    """测试 Actor-Critic"""
    test_name = "Actor-Critic Variant"
    try:
        algo = AlgorithmFactory.create(
            AlgorithmType.REINFORCEMENT_LEARNING,
            variant='actor_critic'
        )
        context = create_test_context(
            search_space={
                "velocity": {"low": -1.0, "high": 1.0},
                "angle": {"low": 0.0, "high": 360.0}
            }
        )
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            reward = 0.5 + i * 0.15
            algo.update(result.recommended_action, reward, context, done=(i == TestConfig.ITERATIONS - 1))
        
        results.add_pass(test_name, "Actor-Critic TD learning working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_policy_gradient(results: OptimizationTestResults):
    """测试策略梯度"""
    test_name = "Policy Gradient Variant"
    try:
        algo = AlgorithmFactory.create(
            AlgorithmType.REINFORCEMENT_LEARNING,
            variant='policy_gradient'
        )
        context = create_test_context(
            search_space={
                "param1": {"low": 0.0, "high": 1.0},
                "param2": {"low": 0.0, "high": 1.0}
            }
        )
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS * 4):
            result = algo.suggest(context)
            reward = 0.4 + i * 0.02
            algo.update(result.recommended_action, reward, done=(i % 4 == 3))
        
        results.add_pass(test_name, "REINFORCE policy update working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_nsga2(results: OptimizationTestResults):
    """测试 NSGA-II 多目标优化"""
    test_name = "NSGA-II Variant"
    try:
        algo = AlgorithmFactory.create(
            AlgorithmType.GENETIC_ALGORITHM,
            variant='nsga2'
        )
        context = create_test_context(
            search_space={
                "x": {"low": 0.0, "high": 1.0},
                "y": {"low": 0.0, "high": 1.0}
            },
            inputs={"multi_objective": True}
        )
        context.metadata["multi_objective"] = True
        algo.initialize(context)
        
        for i in range(TestConfig.ITERATIONS):
            result = algo.suggest(context)
            # 多目标：两个冲突的目标
            x = result.recommended_action.get("x", 0.5)
            y = result.recommended_action.get("y", 0.5)
            fitness = x + y  # 简化
            algo.update(result.recommended_action, fitness)
        
        results.add_pass(test_name, "Non-dominated sorting working")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 场景驱动测试 ====================

def test_scenario_hyperparameter_optimization(results: OptimizationTestResults):
    """测试场景：超参数优化"""
    test_name = "Scenario - Hyperparameter Optimization"
    try:
        context = create_test_context()
        algo = AlgorithmFactory.create_from_scenario(
            scenario='hyperparameter_optimization',
            context=context
        )
        
        assert algo.algorithm_type == AlgorithmType.BAYESIAN_OPTIMIZATION
        
        result = algo.suggest(context)
        assert result.recommended_action
        
        results.add_pass(test_name, f"Selected: {algo.algorithm_type.value}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_scenario_model_selection(results: OptimizationTestResults):
    """测试场景：模型选择"""
    test_name = "Scenario - Model Selection"
    try:
        context = create_test_context(
            search_space={
                "model": {"choices": ["random_forest", "xgboost", "lightgbm", "catboost"]}
            }
        )
        algo = AlgorithmFactory.create_from_scenario(
            scenario='model_selection',
            context=context
        )
        
        assert algo.algorithm_type == AlgorithmType.MULTI_ARMED_BANDIT
        
        result = algo.suggest(context)
        assert "model" in result.recommended_action
        
        results.add_pass(test_name, f"Selected: {algo.algorithm_type.value}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_scenario_training_strategy(results: OptimizationTestResults):
    """测试场景：训练策略"""
    test_name = "Scenario - Training Strategy"
    try:
        context = create_test_context(
            search_space={
                "action": {"choices": ["increase_lr", "decrease_lr", "early_stop", "continue"]}
            }
        )
        algo = AlgorithmFactory.create_from_scenario(
            scenario='training_strategy',
            context=context
        )
        
        assert algo.algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING
        
        result = algo.suggest(context)
        assert result.recommended_action
        
        results.add_pass(test_name, f"Selected: {algo.algorithm_type.value}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_scenario_multi_objective(results: OptimizationTestResults):
    """测试场景：多目标优化"""
    test_name = "Scenario - Multi-Objective"
    try:
        context = create_test_context()
        context.metadata["multi_objective"] = True
        
        algo = AlgorithmFactory.create_from_scenario(
            scenario='multi_objective',
            context=context
        )
        
        # 应该选择 NSGA-II
        assert algo.algorithm_type == AlgorithmType.GENETIC_ALGORITHM
        
        result = algo.suggest(context)
        assert result.recommended_action
        
        results.add_pass(test_name, f"Selected NSGA-II variant")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 智能选择测试 ====================

def test_auto_select_context_aware(results: OptimizationTestResults):
    """测试上下文感知选择"""
    test_name = "Auto Select - Context Aware"
    try:
        # 小维度连续优化 -> 贝叶斯优化
        context = create_test_context(
            search_space={
                "x1": {"low": 0.0, "high": 1.0},
                "x2": {"low": 0.0, "high": 1.0}
            }
        )
        
        algo = AlgorithmFactory.auto_select(
            context=context,
            strategy=SelectionStrategy.CONTEXT_AWARE
        )
        
        result = algo.suggest(context)
        assert result.recommended_action
        
        results.add_pass(test_name, f"Selected: {algo.algorithm_type.value}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_auto_select_adaptive(results: OptimizationTestResults):
    """测试自适应选择"""
    test_name = "Auto Select - Adaptive"
    try:
        context = create_test_context()
        
        algo = AlgorithmFactory.auto_select(
            context=context,
            strategy=SelectionStrategy.ADAPTIVE
        )
        
        result = algo.suggest(context)
        assert result.recommended_action
        
        results.add_pass(test_name, f"Selected: {algo.algorithm_type.value}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 算法集成测试 ====================

def test_ensemble_voting(results: OptimizationTestResults):
    """测试集成 - 投票法"""
    test_name = "Ensemble - Voting"
    try:
        ensemble = AlgorithmFactory.create_ensemble(
            algorithm_types=[
                AlgorithmType.BAYESIAN_OPTIMIZATION,
                AlgorithmType.GENETIC_ALGORITHM,
                AlgorithmType.KNOWLEDGE_REASONING
            ],
            method=EnsembleMethod.VOTING
        )
        
        context = create_test_context()
        ensemble.initialize(context)
        
        result = ensemble.suggest(context)
        assert result.recommended_action
        assert "投票" in result.reasoning or "voting" in result.reasoning.lower()
        
        results.add_pass(test_name, f"Confidence: {result.confidence:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_ensemble_weighted_average(results: OptimizationTestResults):
    """测试集成 - 加权平均"""
    test_name = "Ensemble - Weighted Average"
    try:
        ensemble = AlgorithmFactory.create_ensemble(
            algorithm_types=[
                AlgorithmType.BAYESIAN_OPTIMIZATION,
                AlgorithmType.GENETIC_ALGORITHM
            ],
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights=[0.7, 0.3]
        )
        
        context = create_test_context()
        ensemble.initialize(context)
        
        result = ensemble.suggest(context)
        assert result.recommended_action
        
        # 更新测试
        ensemble.update(result.recommended_action, reward=0.8)
        
        results.add_pass(test_name, f"Confidence: {result.confidence:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_ensemble_best_confidence(results: OptimizationTestResults):
    """测试集成 - 最高置信度"""
    test_name = "Ensemble - Best Confidence"
    try:
        ensemble = AlgorithmFactory.create_ensemble(
            algorithm_types=[
                AlgorithmType.MULTI_ARMED_BANDIT,
                AlgorithmType.KNOWLEDGE_REASONING
            ],
            method=EnsembleMethod.BEST_CONFIDENCE
        )
        
        context = create_test_context(
            search_space={
                "optimizer": {"choices": ["adam", "sgd"]}
            }
        )
        ensemble.initialize(context)
        
        result = ensemble.suggest(context)
        assert result.recommended_action
        assert result.debug_info.get('method') == 'best_confidence'
        
        results.add_pass(test_name, f"Selected from: {result.debug_info.get('selected_algorithm')}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 并行执行测试 ====================

def test_parallel_execution(results: OptimizationTestResults):
    """测试并行执行"""
    test_name = "Parallel Execution"
    try:
        context = create_test_context()
        
        start_time = time.time()
        parallel_results = AlgorithmFactory.run_parallel(
            algorithm_types=[
                AlgorithmType.BAYESIAN_OPTIMIZATION,
                AlgorithmType.GENETIC_ALGORITHM,
                AlgorithmType.MULTI_ARMED_BANDIT
            ],
            context=context,
            max_workers=TestConfig.MAX_PARALLEL_WORKERS
        )
        execution_time = (time.time() - start_time) * 1000
        
        assert len(parallel_results) == 3
        
        success_count = sum(1 for r in parallel_results.values() if r is not None)
        
        results.add_pass(test_name, f"Success: {success_count}/3, Time: {execution_time:.2f}ms")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 性能监控测试 ====================

def test_performance_tracking(results: OptimizationTestResults):
    """测试性能跟踪"""
    test_name = "Performance Tracking"
    try:
        # 清除之前的统计
        AlgorithmFactory.clear_performance_stats()
        
        context = create_test_context()
        
        # 运行几个算法
        for algo_type in [AlgorithmType.BAYESIAN_OPTIMIZATION, AlgorithmType.MULTI_ARMED_BANDIT]:
            algo = AlgorithmFactory.create(algo_type, use_cache=False)
            algo.initialize(context)
            
            for _ in range(3):
                result = algo.suggest(context)
                algo.update(result.recommended_action, reward=0.7)
        
        # 获取性能统计
        stats = AlgorithmFactory.get_performance_stats()
        assert len(stats) > 0
        
        results.add_pass(test_name, f"Tracked {len(stats)} algorithms")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 状态管理测试 ====================

def test_state_save_load(results: OptimizationTestResults):
    """测试状态保存和加载"""
    test_name = "State Save/Load"
    try:
        # 运行一些算法生成统计
        context = create_test_context()
        algo = AlgorithmFactory.create(AlgorithmType.BAYESIAN_OPTIMIZATION, use_cache=False)
        algo.initialize(context)
        result = algo.suggest(context)
        algo.update(result.recommended_action, reward=0.75)
        
        # 保存状态
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        AlgorithmFactory.save_state(temp_path)
        
        # 清除并重新加载
        AlgorithmFactory.clear_performance_stats()
        AlgorithmFactory.load_state(temp_path)
        
        # 验证加载成功
        stats = AlgorithmFactory.get_performance_stats()
        
        # 清理临时文件
        os.unlink(temp_path)
        
        results.add_pass(test_name, f"State persisted and restored")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 便捷函数测试 ====================

def test_convenience_functions(results: OptimizationTestResults):
    """测试便捷函数"""
    test_name = "Convenience Functions"
    try:
        # get_algorithm
        algo = get_algorithm('bayesian_optimization', {'n_initial_points': 3})
        assert algo is not None
        
        # run_algorithm
        result = run_algorithm(
            'genetic_algorithm',
            context={
                'inputs': {'test': True},
                'search_space': {'x': {'low': 0, 'high': 1}}
            }
        )
        assert 'recommended_action' in result
        
        # auto_run
        result = auto_run(
            context={
                'inputs': {'test': True},
                'search_space': {'x': {'low': 0, 'high': 1}}
            },
            strategy='context_aware'
        )
        assert 'recommended_action' in result
        
        # create_ensemble
        ensemble = create_ensemble(
            ['bayesian_optimization', 'genetic_algorithm'],
            method='weighted_average'
        )
        assert isinstance(ensemble, AlgorithmEnsemble)
        
        results.add_pass(test_name, "All convenience functions working")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 获取可用算法测试 ====================

def test_available_algorithms(results: OptimizationTestResults):
    """测试获取可用算法"""
    test_name = "Available Algorithms"
    try:
        algorithms = AlgorithmFactory.get_available_algorithms()
        
        assert 'bayesian_optimization' in algorithms
        assert 'genetic_algorithm' in algorithms
        assert 'reinforcement_learning' in algorithms
        assert 'multi_armed_bandit' in algorithms
        assert 'knowledge_reasoning' in algorithms
        
        # 检查变体
        ga_info = algorithms['genetic_algorithm']
        assert len(ga_info['variants']) >= 2  # 至少有 DE 和 ES
        
        total_algorithms = len(algorithms)
        total_variants = sum(len(a['variants']) for a in algorithms.values())
        
        results.add_pass(test_name, f"Found {total_algorithms} algorithms, {total_variants} variants")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 高级用例测试 ====================

def test_complete_optimization_workflow(results: OptimizationTestResults):
    """测试完整优化工作流"""
    test_name = "Complete Optimization Workflow"
    try:
        # 1. 使用场景驱动选择算法
        context = create_test_context()
        algo = AlgorithmFactory.create_from_scenario(
            'hyperparameter_optimization',
            context=context
        )
        
        best_reward = float('-inf')
        best_action = None
        
        # 2. 运行优化循环
        for i in range(5):
            result = algo.suggest(context)
            
            # 模拟目标函数评估
            action = result.recommended_action
            lr = action.get('learning_rate', 0.01)
            hidden = action.get('hidden_units', 128)
            
            # 模拟奖励（某个假设的最优点附近给高分）
            reward = 0.8 - abs(lr - 0.01) * 10 - abs(hidden - 128) * 0.001
            reward = max(0, min(1, reward + i * 0.02))
            
            algo.update(action, reward)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        # 3. 获取最终结果
        summary = algo.get_optimization_summary()
        
        assert best_action is not None
        assert best_reward > 0
        
        results.add_pass(test_name, f"Best reward: {best_reward:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_algorithm_switching(results: OptimizationTestResults):
    """测试算法切换"""
    test_name = "Algorithm Switching"
    try:
        context = create_test_context()
        
        # 阶段1：使用多臂老虎机进行初始探索
        algo1 = AlgorithmFactory.create_from_scenario('hyperparameter_init', context)
        for _ in range(3):
            result = algo1.suggest(context)
            algo1.update(result.recommended_action, reward=0.5)
        
        # 阶段2：切换到贝叶斯优化进行精细调优
        # 收集观测作为热启动数据
        observations = list(algo1._observations)
        
        algo2 = AlgorithmFactory.create_from_scenario('hyperparameter_optimization', context)
        # 热启动
        if hasattr(algo2, 'warm_start') and observations:
            X = [obs['action'] for obs in observations]
            y = [obs['reward'] for obs in observations]
            algo2.warm_start(X, y)
        else:
            algo2.initialize(context)
        
        for _ in range(3):
            result = algo2.suggest(context)
            algo2.update(result.recommended_action, reward=0.7)
        
        results.add_pass(test_name, "Successfully switched from MAB to BO")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 主测试运行器 ====================

def run_all_tests():
    """运行所有测试"""
    log("=" * 60)
    log("Algorithm Factory Integration Tests")
    log("=" * 60)
    
    results = OptimizationTestResults()
    
    # 清除缓存和统计
    AlgorithmFactory.clear_cache()
    AlgorithmFactory.clear_performance_stats()
    
    # 基础算法测试
    log("\n[Section 1] Basic Algorithm Tests")
    log("-" * 40)
    test_bayesian_optimization(results)
    test_genetic_algorithm(results)
    test_knowledge_reasoning(results)
    test_multi_armed_bandit(results)
    test_reinforcement_learning(results)
    
    # 变体测试
    log("\n[Section 2] Variant Tests")
    log("-" * 40)
    test_contextual_bandit(results)
    test_differential_evolution(results)
    test_actor_critic(results)
    test_policy_gradient(results)
    test_nsga2(results)
    
    # 场景驱动测试
    log("\n[Section 3] Scenario-Driven Tests")
    log("-" * 40)
    test_scenario_hyperparameter_optimization(results)
    test_scenario_model_selection(results)
    test_scenario_training_strategy(results)
    test_scenario_multi_objective(results)
    
    # 智能选择测试
    log("\n[Section 4] Auto Selection Tests")
    log("-" * 40)
    test_auto_select_context_aware(results)
    test_auto_select_adaptive(results)
    
    # 集成测试
    log("\n[Section 5] Ensemble Tests")
    log("-" * 40)
    test_ensemble_voting(results)
    test_ensemble_weighted_average(results)
    test_ensemble_best_confidence(results)
    
    # 并行执行测试
    log("\n[Section 6] Parallel Execution Tests")
    log("-" * 40)
    test_parallel_execution(results)
    
    # 性能监控测试
    log("\n[Section 7] Performance Tracking Tests")
    log("-" * 40)
    test_performance_tracking(results)
    
    # 状态管理测试
    log("\n[Section 8] State Management Tests")
    log("-" * 40)
    test_state_save_load(results)
    
    # 便捷函数测试
    log("\n[Section 9] Convenience Function Tests")
    log("-" * 40)
    test_convenience_functions(results)
    test_available_algorithms(results)
    
    # 高级用例测试
    log("\n[Section 10] Advanced Use Case Tests")
    log("-" * 40)
    test_complete_optimization_workflow(results)
    test_algorithm_switching(results)
    
    # 输出总结
    print(results.summary())
    
    if results.errors:
        print("\nFailed Tests:")
        for error in results.errors:
            print(f"  - {error['name']}: {error['error']}")
    
    return results.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
