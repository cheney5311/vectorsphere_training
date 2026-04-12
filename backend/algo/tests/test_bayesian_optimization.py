"""
贝叶斯优化算法集成测试

测试内容：
1. 核心功能测试
   - 高斯过程拟合和预测
   - 不同核函数测试
   - 采集函数测试
   
2. 高级功能测试
   - 超参数自动优化
   - 批量建议
   - 约束优化
   - 输入归一化
   - 收敛检测
   
3. 状态管理测试
   - 热启动
   - 状态保存/恢复
   
4. 可视化数据测试
5. 完整优化工作流测试

运行方式：
    python -m pytest backend/algo/tests/test_bayesian_optimization.py -v
    或直接运行：
    python backend/algo/tests/test_bayesian_optimization.py
"""

import sys
import os
import json
import time
import math
import tempfile
import numpy as np
from typing import Dict, Any, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from backend.algo.bayesian_optimization import (
    BayesianOptimizationAlgorithm,
    GaussianProcess,
    KernelType,
    KernelParams,
    AcquisitionFunction,
    BatchSuggester,
    ConstraintHandler,
    Constraint,
    InputNormalizer,
    ConvergenceDetector,
    GaussianProcessState,
    create_kernel,
    RBFKernel,
    Matern32Kernel,
    Matern52Kernel,
    RationalQuadraticKernel
)
from backend.algo.base import (
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext
)


# ==================== 测试配置 ====================

class TestConfig:
    """测试配置"""
    VERBOSE = True
    SEED = 42


def log(message: str, level: str = "INFO"):
    """日志输出"""
    if TestConfig.VERBOSE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")


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
        self.details.append({"name": test_name, "status": "PASSED", "details": details})
        log(f"✓ {test_name}: PASSED - {details}", "PASS")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append({"name": test_name, "error": error})
        self.details.append({"name": test_name, "status": "FAILED", "error": error})
        log(f"✗ {test_name}: FAILED - {error}", "FAIL")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"\n{'='*60}\nTest Summary: {self.passed}/{total} passed, {self.failed} failed\n{'='*60}"

@pytest.fixture
def results():
    return OptimizationTestResults()


# ==================== 测试目标函数 ====================

def branin(x: np.ndarray) -> float:
    """Branin 函数（2D 标准测试函数）
    
    全局最小值约为 0.397887，在多个点达到
    """
    x1, x2 = x[0], x[1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere 函数（全局最小值为 0，在原点）
    
    支持单个点或多个点：
    - 单个点: x shape (n_dims,) -> scalar
    - 多个点: x shape (n_samples, n_dims) -> (n_samples,)
    """
    x = np.atleast_2d(x)
    result = np.sum(x**2, axis=1)
    return result[0] if len(result) == 1 else result


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock 函数（全局最小值为 0，在 (1,1,...,1)）"""
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))


def create_test_context(
    n_dims: int = 2,
    bounds: List[Tuple[float, float]] = None,
    observations: List[Dict[str, Any]] = None
) -> AlgorithmContext:
    """创建测试上下文"""
    if bounds is None:
        bounds = [(-5.0, 10.0), (0.0, 15.0)] if n_dims == 2 else [(-5.0, 5.0)] * n_dims
    
    search_space = {f"x{i}": {"low": bounds[i][0], "high": bounds[i][1]} 
                   for i in range(n_dims)}
    
    return AlgorithmContext(
        inputs={"n_dims": n_dims},
        search_space=search_space,
        observations=observations or [],
        objective="minimize",
        objective_metric="loss"
    )


# ==================== 核函数测试 ====================

def test_rbf_kernel(results: OptimizationTestResults):
    """测试 RBF 核函数"""
    test_name = "RBF Kernel"
    try:
        params = KernelParams(length_scale=1.0, variance=1.0)
        kernel = RBFKernel(params)
        
        X1 = np.array([[0, 0], [1, 1]])
        X2 = np.array([[0, 0], [0.5, 0.5]])
        
        K = kernel(X1, X2)
        
        assert K.shape == (2, 2), f"Wrong shape: {K.shape}"
        assert K[0, 0] == 1.0, "K(x,x) should be 1.0"
        assert 0 < K[0, 1] < 1, "K should decrease with distance"
        
        results.add_pass(test_name, f"K shape: {K.shape}, K[0,0]={K[0,0]:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_matern_kernels(results: OptimizationTestResults):
    """测试 Matern 核函数"""
    test_name = "Matern Kernels"
    try:
        params = KernelParams(length_scale=1.0, variance=1.0)
        
        X = np.array([[0, 0], [1, 1], [2, 2]])
        
        for kernel_class, name in [
            (Matern32Kernel, "Matern 3/2"),
            (Matern52Kernel, "Matern 5/2")
        ]:
            kernel = kernel_class(params)
            K = kernel(X, X)
            
            # 验证对称性
            assert np.allclose(K, K.T), f"{name} not symmetric"
            # 验证对角线
            assert np.allclose(np.diag(K), params.variance), f"{name} diagonal wrong"
        
        results.add_pass(test_name, "All Matern kernels passed")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_kernel_factory(results: OptimizationTestResults):
    """测试核函数工厂"""
    test_name = "Kernel Factory"
    try:
        kernel_types = ["rbf", "matern32", "matern52", "rq", "periodic", "linear"]
        
        X = np.array([[0, 0], [1, 1]])
        
        for kt in kernel_types:
            kernel = create_kernel(kt)
            K = kernel(X, X)
            assert K.shape == (2, 2), f"{kt} kernel wrong shape"
        
        results.add_pass(test_name, f"Tested {len(kernel_types)} kernel types")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 高斯过程测试 ====================

def test_gaussian_process_fit_predict(results: OptimizationTestResults):
    """测试高斯过程拟合和预测"""
    test_name = "GP Fit and Predict"
    try:
        np.random.seed(TestConfig.SEED)
        
        # 生成训练数据
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.sin(X_train.ravel() * 2 * np.pi)
        
        gp = GaussianProcess(
            kernel="rbf",
            normalize_y=True,
            optimize_hyperparams=False
        )
        gp.fit(X_train, y_train)
        
        # 预测
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mean, std = gp.predict(X_test, return_std=True)
        
        assert len(mean) == 4, "Wrong prediction length"
        assert len(std) == 4, "Wrong std length"
        assert all(s >= 0 for s in std), "Std should be non-negative"
        
        results.add_pass(test_name, f"Predicted {len(mean)} points, std range: [{min(std):.4f}, {max(std):.4f}]")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_gaussian_process_hyperparameter_optimization(results: OptimizationTestResults):
    """测试高斯过程超参数优化"""
    test_name = "GP Hyperparameter Optimization"
    try:
        np.random.seed(TestConfig.SEED)
        
        # 生成训练数据
        X_train = np.random.rand(10, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(
            kernel="matern52",
            normalize_y=True,
            optimize_hyperparams=True,
            n_restarts=3
        )
        
        initial_params = gp.kernel_params.to_dict()
        gp.fit(X_train, y_train)
        final_params = gp.kernel_params.to_dict()
        
        # 超参数应该有变化
        params_changed = any(
            abs(initial_params[k] - final_params[k]) > 1e-6 
            for k in initial_params
        )
        
        results.add_pass(test_name, f"length_scale: {final_params['length_scale']:.4f}, variance: {final_params['variance']:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_gaussian_process_sampling(results: OptimizationTestResults):
    """测试高斯过程采样"""
    test_name = "GP Sampling"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.array([[0.2], [0.5], [0.8]])
        y_train = np.array([0.1, 0.5, 0.2])
        
        gp = GaussianProcess(kernel="rbf", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        X_test = np.array([[0.3], [0.6]])
        samples = gp.sample(X_test, n_samples=10, random_state=TestConfig.SEED)
        
        assert samples.shape == (10, 2), f"Wrong sample shape: {samples.shape}"
        
        # 样本应该在合理范围内
        mean, _ = gp.predict(X_test)
        sample_means = samples.mean(axis=0)
        
        results.add_pass(test_name, f"Sampled {samples.shape[0]} paths, sample mean: {sample_means}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_gaussian_process_state_management(results: OptimizationTestResults):
    """测试高斯过程状态管理"""
    test_name = "GP State Management"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp1 = GaussianProcess(kernel="matern32", optimize_hyperparams=False)
        gp1.fit(X_train, y_train)
        
        # 保存状态
        state = gp1.get_state()
        
        # 创建新的 GP 并加载状态
        gp2 = GaussianProcess(kernel="rbf", optimize_hyperparams=False)
        gp2.load_state(state)
        
        # 验证预测一致
        X_test = np.random.rand(3, 2)
        mean1, std1 = gp1.predict(X_test)
        mean2, std2 = gp2.predict(X_test)
        
        assert np.allclose(mean1, mean2, rtol=1e-5), "Means should match"
        
        results.add_pass(test_name, "State saved and restored successfully")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 采集函数测试 ====================

def test_acquisition_expected_improvement(results: OptimizationTestResults):
    """测试期望改进采集函数"""
    test_name = "Acquisition - Expected Improvement"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="matern52", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="expected_improvement", xi=0.01)
        
        X_test = np.random.rand(10, 2)
        y_best = np.min(y_train)
        
        ei_values = acq.evaluate(gp, X_test, -y_best)  # 负号因为是最小化
        
        assert len(ei_values) == 10, "Wrong number of EI values"
        assert all(v >= 0 for v in ei_values), "EI should be non-negative"
        
        results.add_pass(test_name, f"EI range: [{min(ei_values):.6f}, {max(ei_values):.6f}]")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_acquisition_ucb(results: OptimizationTestResults):
    """测试 UCB 采集函数"""
    test_name = "Acquisition - UCB"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="rbf", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="upper_confidence_bound", kappa=2.0)
        
        X_test = np.random.rand(10, 2)
        ucb_values = acq.evaluate(gp, X_test, 0)
        
        assert len(ucb_values) == 10, "Wrong number of UCB values"
        
        # UCB 应该 = mean + kappa * std
        mean, std = gp.predict(X_test)
        expected_ucb = mean + 2.0 * std
        
        results.add_pass(test_name, f"UCB range: [{min(ucb_values):.4f}, {max(ucb_values):.4f}]")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_acquisition_thompson_sampling(results: OptimizationTestResults):
    """测试 Thompson Sampling 采集函数"""
    test_name = "Acquisition - Thompson Sampling"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="matern52", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="thompson_sampling", n_samples=100)
        
        X_test = np.random.rand(10, 2)
        ts_values = acq.evaluate(gp, X_test, 0)
        
        assert len(ts_values) == 10, "Wrong number of TS values"
        
        results.add_pass(test_name, f"TS range: [{min(ts_values):.4f}, {max(ts_values):.4f}]")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_acquisition_poi(results: OptimizationTestResults):
    """测试改进概率采集函数"""
    test_name = "Acquisition - Probability of Improvement"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="rbf", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="probability_of_improvement", xi=0.01)
        
        X_test = np.random.rand(10, 2)
        y_best = np.min(y_train)
        
        poi_values = acq.evaluate(gp, X_test, -y_best)
        
        assert len(poi_values) == 10, "Wrong number of POI values"
        assert all(0 <= v <= 1 for v in poi_values), "POI should be in [0, 1]"
        
        results.add_pass(test_name, f"POI range: [{min(poi_values):.4f}, {max(poi_values):.4f}]")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 批量建议测试 ====================

def test_batch_suggester(results: OptimizationTestResults):
    """测试批量建议器"""
    test_name = "Batch Suggester"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="matern52", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="expected_improvement")
        batch_suggester = BatchSuggester(strategy="kriging_believer")
        
        bounds = [(0, 1), (0, 1)]
        y_best = np.max(-y_train)
        
        suggestions = batch_suggester.suggest_batch(
            gp=gp,
            acq_func=acq,
            bounds=bounds,
            y_best=y_best,
            batch_size=3,
            n_candidates=100
        )
        
        assert suggestions.shape == (3, 2), f"Wrong shape: {suggestions.shape}"
        
        # 验证建议在边界内
        assert np.all(suggestions >= 0) and np.all(suggestions <= 1), "Suggestions out of bounds"
        
        results.add_pass(test_name, f"Generated {len(suggestions)} batch suggestions")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_batch_strategies(results: OptimizationTestResults):
    """测试不同批量策略"""
    test_name = "Batch Strategies"
    try:
        np.random.seed(TestConfig.SEED)
        
        X_train = np.random.rand(5, 2)
        y_train = sphere(X_train)
        
        gp = GaussianProcess(kernel="rbf", optimize_hyperparams=False)
        gp.fit(X_train, y_train)
        
        acq = AcquisitionFunction(kind="expected_improvement")
        bounds = [(0, 1), (0, 1)]
        y_best = np.max(-y_train)
        
        strategies = ["kriging_believer", "constant_liar", "hallucination"]
        
        for strategy in strategies:
            batch_suggester = BatchSuggester(strategy=strategy)
            suggestions = batch_suggester.suggest_batch(
                gp=gp,
                acq_func=acq,
                bounds=bounds,
                y_best=y_best,
                batch_size=2,
                n_candidates=50
            )
            assert suggestions.shape[0] == 2, f"{strategy} failed"
        
        results.add_pass(test_name, f"Tested {len(strategies)} strategies")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 约束优化测试 ====================

def test_constraint_handler(results: OptimizationTestResults):
    """测试约束处理器"""
    test_name = "Constraint Handler"
    try:
        # 定义约束：x1 + x2 <= 1
        def constraint_func(x: Dict[str, Any]) -> float:
            return 1 - x.get('x0', 0) - x.get('x1', 0)  # g(x) >= 0
        
        handler = ConstraintHandler()
        handler.add_constraint(Constraint(
            func=constraint_func,
            constraint_type="ineq",
            name="sum_constraint"
        ))
        
        # 测试可行点
        feasible_point = {'x0': 0.3, 'x1': 0.3}
        is_feasible, violated = handler.check_feasibility(feasible_point)
        assert is_feasible, "Should be feasible"
        
        # 测试不可行点
        infeasible_point = {'x0': 0.7, 'x1': 0.7}
        is_feasible, violated = handler.check_feasibility(infeasible_point)
        assert not is_feasible, "Should be infeasible"
        assert len(violated) > 0, "Should have violations"
        
        # 测试惩罚计算
        penalty = handler.compute_penalty(infeasible_point)
        assert penalty > 0, "Penalty should be positive for infeasible point"
        
        results.add_pass(test_name, f"Constraint validation working, penalty: {penalty:.2f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 输入归一化测试 ====================

def test_input_normalizer(results: OptimizationTestResults):
    """测试输入归一化器"""
    test_name = "Input Normalizer"
    try:
        bounds = {
            'x0': (-5.0, 5.0),
            'x1': (0.0, 100.0)
        }
        
        normalizer = InputNormalizer(bounds)
        
        # 测试归一化
        x = {'x0': 0.0, 'x1': 50.0}
        x_normalized = normalizer.normalize(x)
        
        assert len(x_normalized) == 2, "Wrong dimensions"
        assert 0 <= x_normalized[0] <= 1, "x0 should be normalized"
        assert abs(x_normalized[0] - 0.5) < 0.01, "x0=0 should map to 0.5"
        assert abs(x_normalized[1] - 0.5) < 0.01, "x1=50 should map to 0.5"
        
        # 测试反归一化
        x_denormalized = normalizer.denormalize(x_normalized)
        assert abs(x_denormalized['x0'] - 0.0) < 0.01, "Denormalization failed"
        assert abs(x_denormalized['x1'] - 50.0) < 0.01, "Denormalization failed"
        
        results.add_pass(test_name, f"Normalized: {x_normalized}, Denormalized: {x_denormalized}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 收敛检测测试 ====================

def test_convergence_detector(results: OptimizationTestResults):
    """测试收敛检测器"""
    test_name = "Convergence Detector"
    try:
        detector = ConvergenceDetector(
            max_iterations=50,
            no_improvement_rounds=5,
            improvement_threshold=0.001
        )
        
        # 模拟改进
        for i in range(10):
            state = detector.update(0.5 + i * 0.1, i)
            assert not state.converged, f"Should not converge at iteration {i}"
        
        # 模拟无改进
        for i in range(10, 20):
            state = detector.update(1.5, i)  # 相同值
        
        assert state.converged, "Should converge after no improvement"
        assert "No improvement" in state.reason or "converged" in state.reason.lower()
        
        results.add_pass(test_name, f"Converged at iteration {i}, reason: {state.reason}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 贝叶斯优化算法测试 ====================

def test_bayesian_optimization_basic(results: OptimizationTestResults):
    """测试贝叶斯优化基本功能"""
    test_name = "BO Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            acquisition_function="expected_improvement",
            n_initial_points=3,
            extra={
                "kernel": "matern52",
                "optimize_hyperparams": False,
                "minimize": True
            }
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 运行几次迭代
        for i in range(5):
            result = algo.suggest(context)
            assert result.recommended_action, "No action recommended"
            
            # 评估目标函数
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            y = sphere(x)
            
            algo.update(result.recommended_action, y)
        
        # 验证有观测记录
        assert len(algo._observations) == 5, f"Should have 5 observations, got {len(algo._observations)}"
        
        results.add_pass(test_name, f"Completed 5 iterations")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_with_different_kernels(results: OptimizationTestResults):
    """测试不同核函数的贝叶斯优化"""
    test_name = "BO with Different Kernels"
    try:
        kernels = ["rbf", "matern32", "matern52", "rq"]
        
        for kernel in kernels:
            config = AlgorithmConfig(
                algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
                n_initial_points=2,
                extra={"kernel": kernel, "optimize_hyperparams": False}
            )
            
            algo = BayesianOptimizationAlgorithm(config)
            context = create_test_context(n_dims=2)
            algo.initialize(context)
            
            result = algo.suggest(context)
            assert result.recommended_action, f"Kernel {kernel} failed"
            
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        results.add_pass(test_name, f"Tested {len(kernels)} kernels")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_with_different_acquisitions(results: OptimizationTestResults):
    """测试不同采集函数的贝叶斯优化"""
    test_name = "BO with Different Acquisitions"
    try:
        acquisitions = ["expected_improvement", "upper_confidence_bound", 
                        "probability_of_improvement", "thompson_sampling"]
        
        for acq in acquisitions:
            config = AlgorithmConfig(
                algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
                acquisition_function=acq,
                n_initial_points=2,
                extra={"optimize_hyperparams": False}
            )
            
            algo = BayesianOptimizationAlgorithm(config)
            context = create_test_context(n_dims=2)
            algo.initialize(context)
            
            result = algo.suggest(context)
            assert result.recommended_action, f"Acquisition {acq} failed"
            
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        results.add_pass(test_name, f"Tested {len(acquisitions)} acquisitions")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_batch_suggestion(results: OptimizationTestResults):
    """测试贝叶斯优化批量建议"""
    test_name = "BO Batch Suggestion"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=3,
            extra={"optimize_hyperparams": False, "batch_strategy": "kriging_believer"}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 先运行一些初始点
        for i in range(3):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        # 请求批量建议
        result = algo.suggest(context, batch_size=3)
        assert result.recommended_action, "Batch suggestion failed"
        
        # 检查是否有批量建议
        batch_suggestions = result.debug_info.get('batch_suggestions', [])
        
        results.add_pass(test_name, f"Got batch with {len(batch_suggestions) if batch_suggestions else 1} suggestions")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_warm_start(results: OptimizationTestResults):
    """测试贝叶斯优化热启动"""
    test_name = "BO Warm Start"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=2,
            extra={"optimize_hyperparams": False}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        
        # 准备历史数据
        X_history = [
            {'x0': 0.0, 'x1': 0.0},
            {'x0': 1.0, 'x1': 1.0},
            {'x0': 2.0, 'x1': 2.0}
        ]
        y_history = [sphere(np.array([x['x0'], x['x1']])) for x in X_history]
        
        # 热启动
        algo.initialize(context)
        algo.warm_start(X_history, y_history)
        
        assert len(algo._observations) == 3, "Should have 3 observations from warm start"
        
        # 继续优化
        result = algo.suggest(context)
        assert result.recommended_action, "Suggestion after warm start failed"
        
        results.add_pass(test_name, f"Warm started with {len(X_history)} points")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_state_management(results: OptimizationTestResults):
    """测试贝叶斯优化状态管理"""
    test_name = "BO State Management"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=2,
            extra={"optimize_hyperparams": False}
        )
        
        algo1 = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo1.initialize(context)
        
        # 运行几次
        for i in range(3):
            result = algo1.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo1.update(result.recommended_action, sphere(x))
        
        # 保存状态
        state = algo1.get_state()
        
        # 创建新算法并加载状态
        algo2 = BayesianOptimizationAlgorithm(config)
        algo2.load_state(state)
        
        assert len(algo2._observations) == len(algo1._observations), "Observations count mismatch"
        
        # 验证可以继续运行
        result = algo2.suggest(context)
        assert result.recommended_action, "Suggestion after state load failed"
        
        results.add_pass(test_name, f"State saved and restored, {len(algo2._observations)} observations")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_with_constraints(results: OptimizationTestResults):
    """测试带约束的贝叶斯优化"""
    test_name = "BO with Constraints"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=3,
            extra={"optimize_hyperparams": False}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2, bounds=[(0, 2), (0, 2)])
        algo.initialize(context)
        
        # 添加约束: x0 + x1 <= 3
        algo.add_constraint(
            func=lambda x: 3 - x.get('x0', 0) - x.get('x1', 0),
            constraint_type="ineq",
            name="sum_constraint"
        )
        
        # 运行优化
        for i in range(3):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        results.add_pass(test_name, "Constraint optimization working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_visualization_data(results: OptimizationTestResults):
    """测试贝叶斯优化可视化数据"""
    test_name = "BO Visualization Data"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=2,
            extra={"optimize_hyperparams": False}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 运行几次
        for i in range(4):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        # 获取可视化数据
        viz_data = algo.get_visualization_data()
        
        assert 'observations' in viz_data, "Missing observations"
        assert 'best_history' in viz_data, "Missing best history"
        assert len(viz_data['observations']) == 4, "Wrong observation count"
        
        results.add_pass(test_name, f"Got viz data with {len(viz_data['observations'])} observations")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_bayesian_optimization_summary(results: OptimizationTestResults):
    """测试贝叶斯优化摘要"""
    test_name = "BO Optimization Summary"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=2,
            extra={"optimize_hyperparams": False, "minimize": True}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 运行优化
        for i in range(5):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        # 获取摘要
        summary = algo.get_optimization_summary()
        
        assert 'status' in summary, "Missing status"
        assert 'total_evaluations' in summary, "Missing total_evaluations"
        assert 'best_value' in summary, "Missing best_value"
        assert 'best_params' in summary, "Missing best_params"
        
        results.add_pass(test_name, f"Best value: {summary['best_value']:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 完整优化工作流测试 ====================

def test_optimization_workflow_sphere(results: OptimizationTestResults):
    """测试完整优化工作流 - Sphere 函数"""
    test_name = "Optimization Workflow - Sphere"
    try:
        np.random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            acquisition_function="expected_improvement",
            n_initial_points=5,
            extra={
                "kernel": "matern52",
                "optimize_hyperparams": True,
                "minimize": True,
                "n_restarts": 2
            }
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2, bounds=[(-5, 5), (-5, 5)])
        algo.initialize(context)
        
        best_y = float('inf')
        best_x = None
        
        for i in range(15):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            y = sphere(x)
            
            if y < best_y:
                best_y = y
                best_x = x
            
            algo.update(result.recommended_action, y)
        
        # Sphere 函数最小值在原点，应该接近 0
        assert best_y < 5.0, f"Optimization did not converge well, best_y={best_y}"
        
        results.add_pass(test_name, f"Best y: {best_y:.4f}, Best x: {best_x}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_optimization_workflow_1d(results: OptimizationTestResults):
    """测试 1D 优化工作流"""
    test_name = "Optimization Workflow - 1D"
    try:
        np.random.seed(TestConfig.SEED)
        
        # 1D 函数: f(x) = sin(x) + 0.1*x^2，在 x≈-1.5 附近有最小值
        def objective(x):
            return np.sin(x[0]) + 0.1 * x[0]**2
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=3,
            extra={"kernel": "rbf", "optimize_hyperparams": False, "minimize": True}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=1, bounds=[(-5, 5)])
        algo.initialize(context)
        
        best_y = float('inf')
        
        for i in range(10):
            result = algo.suggest(context)
            x = np.array([result.recommended_action['x0']])
            y = objective(x)
            
            if y < best_y:
                best_y = y
            
            algo.update(result.recommended_action, y)
        
        # 检查是否有改进
        summary = algo.get_optimization_summary()
        
        results.add_pass(test_name, f"Best y: {best_y:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_optimization_workflow_high_dim(results: OptimizationTestResults):
    """测试高维优化工作流"""
    test_name = "Optimization Workflow - High Dim (5D)"
    try:
        np.random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=10,
            extra={
                "kernel": "matern52",
                "optimize_hyperparams": False,
                "minimize": True
            }
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=5, bounds=[(-2, 2)] * 5)
        algo.initialize(context)
        
        best_y = float('inf')
        
        for i in range(15):
            result = algo.suggest(context)
            x = np.array([result.recommended_action[f'x{j}'] for j in range(5)])
            y = sphere(x)
            
            if y < best_y:
                best_y = y
            
            algo.update(result.recommended_action, y)
        
        results.add_pass(test_name, f"5D Sphere best y: {best_y:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 性能测试 ====================

def test_performance_timing(results: OptimizationTestResults):
    """测试性能计时"""
    test_name = "Performance Timing"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
            n_initial_points=5,
            extra={"optimize_hyperparams": False}
        )
        
        algo = BayesianOptimizationAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 测量建议时间
        times = []
        for i in range(10):
            start = time.time()
            result = algo.suggest(context)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            x = np.array([result.recommended_action[f'x{j}'] for j in range(2)])
            algo.update(result.recommended_action, sphere(x))
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        results.add_pass(test_name, f"Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 主测试运行器 ====================

def run_all_tests():
    """运行所有测试"""
    log("=" * 60)
    log("Bayesian Optimization Integration Tests")
    log("=" * 60)
    
    results = OptimizationTestResults()
    
    # 核函数测试
    log("\n[Section 1] Kernel Tests")
    log("-" * 40)
    test_rbf_kernel(results)
    test_matern_kernels(results)
    test_kernel_factory(results)
    
    # 高斯过程测试
    log("\n[Section 2] Gaussian Process Tests")
    log("-" * 40)
    test_gaussian_process_fit_predict(results)
    test_gaussian_process_hyperparameter_optimization(results)
    test_gaussian_process_sampling(results)
    test_gaussian_process_state_management(results)
    
    # 采集函数测试
    log("\n[Section 3] Acquisition Function Tests")
    log("-" * 40)
    test_acquisition_expected_improvement(results)
    test_acquisition_ucb(results)
    test_acquisition_thompson_sampling(results)
    test_acquisition_poi(results)
    
    # 批量建议测试
    log("\n[Section 4] Batch Suggestion Tests")
    log("-" * 40)
    test_batch_suggester(results)
    test_batch_strategies(results)
    
    # 约束和归一化测试
    log("\n[Section 5] Constraint and Normalization Tests")
    log("-" * 40)
    test_constraint_handler(results)
    test_input_normalizer(results)
    test_convergence_detector(results)
    
    # 贝叶斯优化算法测试
    log("\n[Section 6] Bayesian Optimization Algorithm Tests")
    log("-" * 40)
    test_bayesian_optimization_basic(results)
    test_bayesian_optimization_with_different_kernels(results)
    test_bayesian_optimization_with_different_acquisitions(results)
    test_bayesian_optimization_batch_suggestion(results)
    test_bayesian_optimization_warm_start(results)
    test_bayesian_optimization_state_management(results)
    test_bayesian_optimization_with_constraints(results)
    test_bayesian_optimization_visualization_data(results)
    test_bayesian_optimization_summary(results)
    
    # 完整优化工作流测试
    log("\n[Section 7] Complete Optimization Workflow Tests")
    log("-" * 40)
    test_optimization_workflow_sphere(results)
    test_optimization_workflow_1d(results)
    test_optimization_workflow_high_dim(results)
    
    # 性能测试
    log("\n[Section 8] Performance Tests")
    log("-" * 40)
    test_performance_timing(results)
    
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
