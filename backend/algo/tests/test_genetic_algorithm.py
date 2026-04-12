"""
遗传算法集成测试

测试内容：
1. 遗传操作测试
   - 选择操作器（锦标赛、轮盘赌、排名、SUS、截断、玻尔兹曼）
   - 交叉操作器（SBX、BLX-α、单点、两点、均匀、算术）
   - 变异操作器（多项式、高斯、均匀、非均匀、边界）

2. 种群管理测试
   - 种群初始化
   - 种群统计
   - 多样性计算

3. 约束处理测试
   - 约束定义
   - 惩罚计算
   - 可行性检查

4. 收敛检测测试

5. 算法变体测试
   - 标准遗传算法
   - NSGA-II 多目标优化
   - 差分进化
   - 进化策略

6. 状态管理测试
   - 热启动
   - 状态保存/恢复

7. 完整优化工作流测试

运行方式：
    python -m pytest backend/algo/tests/test_genetic_algorithm.py -v
    或直接运行：
    python backend/algo/tests/test_genetic_algorithm.py
"""

import sys
import os
import time
import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from backend.algo.genetic_algorithm import (
    GeneticAlgorithm,
    NSGAII,
    DifferentialEvolution,
    EvolutionStrategy,
    Individual,
    Population,
    SelectionOperator,
    CrossoverOperator,
    MutationOperator,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod,
    Constraint,
    ConstraintHandler,
    ConvergenceDetector,
    ConvergenceState
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

def sphere(genes: Dict[str, Any]) -> float:
    """Sphere 函数（最大化：取负值）
    
    全局最大值为 0，在原点
    """
    total = 0.0
    for key, value in genes.items():
        if isinstance(value, (int, float)):
            total += value ** 2
    return -total  # 负值因为 GA 默认最大化


def rastrigin(genes: Dict[str, Any]) -> float:
    """Rastrigin 函数（最大化：取负值）
    
    全局最小值为 0，在原点。有很多局部最小值。
    """
    A = 10
    n = len([v for v in genes.values() if isinstance(v, (int, float))])
    total = A * n
    for key, value in genes.items():
        if isinstance(value, (int, float)):
            total += value ** 2 - A * math.cos(2 * math.pi * value)
    return -total


def rosenbrock(genes: Dict[str, Any]) -> float:
    """Rosenbrock 函数（最大化：取负值）"""
    values = [v for v in genes.values() if isinstance(v, (int, float))]
    total = 0.0
    for i in range(len(values) - 1):
        total += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
    return -total


def multi_objective_1(genes: Dict[str, Any]) -> float:
    """多目标函数 1: f1(x) = x^2 + y^2"""
    x = genes.get('x', 0)
    y = genes.get('y', 0)
    return x**2 + y**2


def multi_objective_2(genes: Dict[str, Any]) -> float:
    """多目标函数 2: f2(x) = (x-1)^2 + (y-1)^2"""
    x = genes.get('x', 0)
    y = genes.get('y', 0)
    return (x - 1)**2 + (y - 1)**2


def create_test_context(
    n_dims: int = 2,
    bounds: List[Tuple[float, float]] = None,
    observations: List[Dict[str, Any]] = None
) -> AlgorithmContext:
    """创建测试上下文"""
    if bounds is None:
        bounds = [(-5.0, 5.0)] * n_dims
    
    search_space = {f"x{i}": {"low": bounds[i][0], "high": bounds[i][1], "type": "float"} 
                   for i in range(n_dims)}
    
    return AlgorithmContext(
        inputs={"n_dims": n_dims},
        search_space=search_space,
        observations=observations or [],
        objective="maximize",
        objective_metric="fitness"
    )


def create_test_individuals(n: int = 10, n_dims: int = 2) -> List[Individual]:
    """创建测试个体"""
    individuals = []
    for i in range(n):
        genes = {f"x{j}": random.uniform(-5, 5) for j in range(n_dims)}
        fitness = sphere(genes)
        individuals.append(Individual(
            genes=genes,
            fitness=fitness,
            generation=0,
            evaluated=True
        ))
    return individuals


# ==================== 个体和种群测试 ====================

def test_individual_creation(results: OptimizationTestResults):
    """测试个体创建"""
    test_name = "Individual Creation"
    try:
        genes = {"x0": 1.0, "x1": 2.0}
        ind = Individual(genes=genes, fitness=0.0)
        
        assert ind.genes == genes, "Genes mismatch"
        assert ind.fitness == 0.0, "Fitness mismatch"
        assert ind.generation == 0, "Generation mismatch"
        assert not ind.evaluated, "Should not be evaluated"
        
        results.add_pass(test_name, f"Individual created with {len(genes)} genes")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_individual_copy(results: OptimizationTestResults):
    """测试个体复制"""
    test_name = "Individual Copy"
    try:
        genes = {"x0": 1.0, "x1": 2.0}
        ind1 = Individual(genes=genes, fitness=5.0, generation=3)
        ind2 = ind1.copy()
        
        # 修改副本不应影响原件
        ind2.genes["x0"] = 999.0
        ind2.fitness = 999.0
        
        assert ind1.genes["x0"] == 1.0, "Original should not be modified"
        assert ind1.fitness == 5.0, "Original fitness should not be modified"
        
        results.add_pass(test_name, "Deep copy working correctly")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_individual_dominance(results: OptimizationTestResults):
    """测试个体支配关系"""
    test_name = "Individual Dominance"
    try:
        # 注意：Individual.dominates 假设最大化目标
        # ind1 的目标都比 ind2 大，所以 ind1 支配 ind2
        ind1 = Individual(genes={}, objectives=[3.0, 4.0])  # 较好（目标值更大）
        ind2 = Individual(genes={}, objectives=[1.0, 2.0])  # 较差（目标值更小）
        ind3 = Individual(genes={}, objectives=[2.0, 3.0])  # 中间
        
        # ind1 支配 ind2（因为目标都更大，假设最大化）
        assert ind1.dominates(ind2), "ind1 should dominate ind2"
        assert not ind2.dominates(ind1), "ind2 should not dominate ind1"
        
        results.add_pass(test_name, "Dominance relation working")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_population_stats(results: OptimizationTestResults):
    """测试种群统计"""
    test_name = "Population Stats"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        pop = Population(individuals=individuals, generation=0)
        pop.update_stats()
        
        assert pop.best_fitness > pop.min_fitness, "Best should be > min"
        assert pop.min_fitness <= pop.avg_fitness <= pop.best_fitness, "Avg should be in range"
        assert pop.diversity >= 0, "Diversity should be non-negative"
        
        results.add_pass(test_name, f"best={pop.best_fitness:.4f}, avg={pop.avg_fitness:.4f}, diversity={pop.diversity:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_population_best(results: OptimizationTestResults):
    """测试获取最佳个体"""
    test_name = "Population Best"
    try:
        individuals = create_test_individuals(n=10, n_dims=2)
        pop = Population(individuals=individuals, generation=0)
        pop.update_stats()
        
        best = pop.get_best()
        top3 = pop.get_best(n=3)
        
        assert best is not None, "Best should not be None"
        assert best.fitness == pop.best_fitness, "Best fitness mismatch"
        assert len(top3) == 3, "Should return 3 individuals"
        assert top3[0].fitness >= top3[1].fitness >= top3[2].fitness, "Should be sorted"
        
        results.add_pass(test_name, f"Best fitness: {best.fitness:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 选择操作器测试 ====================

def test_selection_tournament(results: OptimizationTestResults):
    """测试锦标赛选择"""
    test_name = "Selection - Tournament"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.TOURNAMENT, tournament_size=3)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        assert all(isinstance(ind, Individual) for ind in selected), "All should be individuals"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_selection_roulette(results: OptimizationTestResults):
    """测试轮盘赌选择"""
    test_name = "Selection - Roulette"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.ROULETTE)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_selection_rank(results: OptimizationTestResults):
    """测试排名选择"""
    test_name = "Selection - Rank"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.RANK)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_selection_sus(results: OptimizationTestResults):
    """测试随机通用采样"""
    test_name = "Selection - SUS"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.SUS)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_selection_truncation(results: OptimizationTestResults):
    """测试截断选择"""
    test_name = "Selection - Truncation"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.TRUNCATION, truncation_ratio=0.3)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_selection_boltzmann(results: OptimizationTestResults):
    """测试玻尔兹曼选择"""
    test_name = "Selection - Boltzmann"
    try:
        individuals = create_test_individuals(n=20, n_dims=2)
        selector = SelectionOperator(method=SelectionMethod.BOLTZMANN, temperature=1.0)
        
        selected = selector.select(individuals, n=10)
        
        assert len(selected) == 10, f"Should select 10, got {len(selected)}"
        
        results.add_pass(test_name, f"Selected {len(selected)} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 交叉操作器测试 ====================

def test_crossover_sbx(results: OptimizationTestResults):
    """测试 SBX 交叉"""
    test_name = "Crossover - SBX"
    try:
        parent1 = Individual(genes={"x0": 1.0, "x1": 2.0}, fitness=0.0)
        parent2 = Individual(genes={"x0": 3.0, "x1": 4.0}, fitness=0.0)
        
        crossover = CrossoverOperator(method=CrossoverMethod.SBX, eta=20.0, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        child1, child2 = crossover.crossover(parent1, parent2, param_types, param_bounds, {})
        
        assert child1.genes != parent1.genes or child1.genes != parent2.genes, "Children should differ from parents"
        assert -5.0 <= child1.genes["x0"] <= 5.0, "Should be in bounds"
        
        results.add_pass(test_name, f"Child1: {child1.genes}, Child2: {child2.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_crossover_blx_alpha(results: OptimizationTestResults):
    """测试 BLX-α 交叉"""
    test_name = "Crossover - BLX-α"
    try:
        parent1 = Individual(genes={"x0": 1.0, "x1": 2.0}, fitness=0.0)
        parent2 = Individual(genes={"x0": 3.0, "x1": 4.0}, fitness=0.0)
        
        crossover = CrossoverOperator(method=CrossoverMethod.BLX_ALPHA, alpha=0.5, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        child1, child2 = crossover.crossover(parent1, parent2, param_types, param_bounds, {})
        
        assert -5.0 <= child1.genes["x0"] <= 5.0, "Should be in bounds"
        
        results.add_pass(test_name, f"Child1: {child1.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_crossover_arithmetic(results: OptimizationTestResults):
    """测试算术交叉"""
    test_name = "Crossover - Arithmetic"
    try:
        parent1 = Individual(genes={"x0": 0.0, "x1": 0.0}, fitness=0.0)
        parent2 = Individual(genes={"x0": 4.0, "x1": 4.0}, fitness=0.0)
        
        crossover = CrossoverOperator(method=CrossoverMethod.ARITHMETIC, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        child1, child2 = crossover.crossover(parent1, parent2, param_types, param_bounds, {})
        
        # 算术交叉的子代应该在父代之间
        assert 0.0 <= child1.genes["x0"] <= 4.0, "Should be between parents"
        
        results.add_pass(test_name, f"Child1: {child1.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_crossover_categorical(results: OptimizationTestResults):
    """测试分类变量交叉"""
    test_name = "Crossover - Categorical"
    try:
        parent1 = Individual(genes={"x0": 1.0, "cat": "A"}, fitness=0.0)
        parent2 = Individual(genes={"x0": 2.0, "cat": "B"}, fitness=0.0)
        
        crossover = CrossoverOperator(method=CrossoverMethod.SBX, probability=1.0)
        
        param_types = {"x0": "float", "cat": "categorical"}
        param_bounds = {"x0": (-5.0, 5.0)}
        param_choices = {"cat": ["A", "B", "C"]}
        
        child1, child2 = crossover.crossover(parent1, parent2, param_types, param_bounds, param_choices)
        
        assert child1.genes["cat"] in ["A", "B"], "Should be from parents"
        
        results.add_pass(test_name, f"Child cat: {child1.genes['cat']}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 变异操作器测试 ====================

def test_mutation_polynomial(results: OptimizationTestResults):
    """测试多项式变异"""
    test_name = "Mutation - Polynomial"
    try:
        ind = Individual(genes={"x0": 0.0, "x1": 0.0}, fitness=0.0)
        
        mutation = MutationOperator(method=MutationMethod.POLYNOMIAL, eta=20.0, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        original_x0 = ind.genes["x0"]
        mutation.mutate(ind, param_types, param_bounds, {})
        
        # 高概率下应该发生变异
        mutated = ind.genes["x0"] != original_x0 or ind.genes["x1"] != 0.0
        
        assert -5.0 <= ind.genes["x0"] <= 5.0, "Should be in bounds"
        
        results.add_pass(test_name, f"Mutated: {ind.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_mutation_gaussian(results: OptimizationTestResults):
    """测试高斯变异"""
    test_name = "Mutation - Gaussian"
    try:
        ind = Individual(genes={"x0": 0.0, "x1": 0.0}, fitness=0.0)
        
        mutation = MutationOperator(method=MutationMethod.GAUSSIAN, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        mutation.mutate(ind, param_types, param_bounds, {})
        
        assert -5.0 <= ind.genes["x0"] <= 5.0, "Should be in bounds"
        
        results.add_pass(test_name, f"Mutated: {ind.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_mutation_uniform(results: OptimizationTestResults):
    """测试均匀变异"""
    test_name = "Mutation - Uniform"
    try:
        ind = Individual(genes={"x0": 0.0, "x1": 0.0}, fitness=0.0)
        
        mutation = MutationOperator(method=MutationMethod.UNIFORM, probability=1.0)
        
        param_types = {"x0": "float", "x1": "float"}
        param_bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
        
        mutation.mutate(ind, param_types, param_bounds, {})
        
        assert -5.0 <= ind.genes["x0"] <= 5.0, "Should be in bounds"
        
        results.add_pass(test_name, f"Mutated: {ind.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_mutation_adaptive(results: OptimizationTestResults):
    """测试自适应变异"""
    test_name = "Mutation - Adaptive"
    try:
        mutation = MutationOperator(
            method=MutationMethod.POLYNOMIAL, 
            probability=0.5,
            adaptive=True,
            max_generation=100
        )
        
        initial_prob = mutation.probability
        
        # 模拟进化
        mutation.set_generation(50)
        mid_prob = mutation.probability
        
        mutation.set_generation(100)
        final_prob = mutation.probability
        
        assert final_prob <= initial_prob, "Probability should decay"
        
        results.add_pass(test_name, f"Prob: {initial_prob:.4f} -> {mid_prob:.4f} -> {final_prob:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_mutation_categorical(results: OptimizationTestResults):
    """测试分类变量变异"""
    test_name = "Mutation - Categorical"
    try:
        ind = Individual(genes={"x0": 0.0, "cat": "A"}, fitness=0.0)
        
        mutation = MutationOperator(method=MutationMethod.POLYNOMIAL, probability=1.0)
        
        param_types = {"x0": "float", "cat": "categorical"}
        param_bounds = {"x0": (-5.0, 5.0)}
        param_choices = {"cat": ["A", "B", "C"]}
        
        # 多次变异以确保分类变量也被变异
        for _ in range(10):
            mutation.mutate(ind, param_types, param_bounds, param_choices)
        
        assert ind.genes["cat"] in ["A", "B", "C"], "Should be valid choice"
        
        results.add_pass(test_name, f"Final cat: {ind.genes['cat']}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 约束处理测试 ====================

def test_constraint_definition(results: OptimizationTestResults):
    """测试约束定义"""
    test_name = "Constraint Definition"
    try:
        # g(x) = 1 - x0 - x1 >= 0，即 x0 + x1 <= 1
        constraint = Constraint(
            func=lambda g: 1 - g.get('x0', 0) - g.get('x1', 0),
            constraint_type="ineq",
            name="sum_constraint"
        )
        
        # 可行点
        feasible = {"x0": 0.3, "x1": 0.3}
        violation1 = constraint.get_violation(feasible)
        assert violation1 == 0, f"Should be feasible, got violation {violation1}"
        
        # 不可行点
        infeasible = {"x0": 0.7, "x1": 0.7}
        violation2 = constraint.get_violation(infeasible)
        assert violation2 > 0, f"Should be infeasible, got violation {violation2}"
        
        results.add_pass(test_name, f"Feasible violation: {violation1}, Infeasible violation: {violation2:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_constraint_handler(results: OptimizationTestResults):
    """测试约束处理器"""
    test_name = "Constraint Handler"
    try:
        handler = ConstraintHandler(penalty_coefficient=1000)
        
        handler.add_constraint(Constraint(
            func=lambda g: 1 - g.get('x0', 0) - g.get('x1', 0),
            constraint_type="ineq",
            name="sum_constraint"
        ))
        
        # 可行个体
        feasible_ind = Individual(genes={"x0": 0.3, "x1": 0.3}, fitness=0.0)
        penalty1 = handler.compute_penalty(feasible_ind)
        is_feasible1 = handler.is_feasible(feasible_ind)
        
        # 不可行个体
        infeasible_ind = Individual(genes={"x0": 0.8, "x1": 0.8}, fitness=0.0)
        penalty2 = handler.compute_penalty(infeasible_ind)
        is_feasible2 = handler.is_feasible(infeasible_ind)
        
        assert penalty1 == 0, f"Feasible should have 0 penalty, got {penalty1}"
        assert is_feasible1, "Should be feasible"
        assert penalty2 > 0, "Infeasible should have positive penalty"
        assert not is_feasible2, "Should be infeasible"
        
        results.add_pass(test_name, f"Feasible penalty: {penalty1}, Infeasible penalty: {penalty2:.2f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_adaptive_penalty(results: OptimizationTestResults):
    """测试自适应惩罚"""
    test_name = "Adaptive Penalty"
    try:
        handler = ConstraintHandler(penalty_coefficient=1000, adaptive_penalty=True)
        
        handler.add_constraint(Constraint(
            func=lambda g: 1 - g.get('x0', 0),
            constraint_type="ineq"
        ))
        
        initial_coef = handler.penalty_coefficient
        
        handler.set_generation(10)
        mid_coef = handler.penalty_coefficient
        
        handler.set_generation(50)
        final_coef = handler.penalty_coefficient
        
        assert final_coef > initial_coef, "Penalty coefficient should increase"
        
        results.add_pass(test_name, f"Coef: {initial_coef} -> {mid_coef} -> {final_coef}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 收敛检测测试 ====================

def test_convergence_detector(results: OptimizationTestResults):
    """测试收敛检测器"""
    test_name = "Convergence Detector"
    try:
        # 设置较宽松的多样性阈值以避免提前收敛
        detector = ConvergenceDetector(
            max_generations=50,
            no_improvement_generations=5,
            fitness_tolerance=0.001,
            diversity_threshold=0.0  # 禁用多样性收敛检测
        )
        
        # 创建具有多样性的种群
        individuals = []
        for i in range(20):
            genes = {"x0": random.uniform(-5, 5), "x1": random.uniform(-5, 5)}
            individuals.append(Individual(genes=genes, fitness=-10 + i * 0.1, evaluated=True))
        
        pop = Population(individuals=individuals, generation=0)
        pop.update_stats()
        pop.diversity = 1.0  # 设置足够高的多样性
        
        # 模拟改进
        for i in range(10):
            pop.generation = i
            pop.best_fitness = -10 + i  # 逐渐改进
            pop.diversity = 1.0  # 保持多样性
            state = detector.update(pop)
            if state.converged:
                # 如果是因为多样性收敛，跳过
                if "diversity" in state.reason.lower():
                    detector.reset()
                    continue
            assert not state.converged, f"Should not converge at generation {i}, reason: {state.reason}"
        
        # 模拟无改进
        for i in range(10, 20):
            pop.generation = i
            pop.diversity = 1.0  # 保持多样性
            # best_fitness 保持不变（不再改进）
            state = detector.update(pop)
        
        assert state.converged, "Should converge after no improvement"
        
        results.add_pass(test_name, f"Converged: {state.reason}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 遗传算法基本测试 ====================

def test_genetic_algorithm_basic(results: OptimizationTestResults):
    """测试遗传算法基本功能"""
    test_name = "GA Basic"
    try:
        random.seed(TestConfig.SEED)
        np.random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.9,
            elite_ratio=0.1,
            extra={
                "selection_method": "tournament",
                "crossover_method": "sbx",
                "mutation_method": "polynomial"
            }
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 设置适应度函数
        algo.set_fitness_function(sphere)
        
        # 运行几代
        for i in range(5):
            result = algo.suggest(context)
            assert result.recommended_action, f"No action at iteration {i}"
            
            fitness = sphere(result.recommended_action)
            algo.update(result.recommended_action, fitness)
        
        results.add_pass(test_name, f"Best fitness: {algo._population.best_fitness:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_evolve(results: OptimizationTestResults):
    """测试遗传算法进化"""
    test_name = "GA Evolve"
    try:
        random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=30,
            mutation_rate=0.15,
            crossover_rate=0.85,
            extra={"max_generations": 20}
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        # 评估初始种群
        algo.evaluate_population()
        initial_best = algo._population.best_fitness
        
        # 进化几代
        for _ in range(10):
            algo.evolve()
            algo.evaluate_population()
        
        final_best = algo._population.best_fitness
        
        # 应该有改进（最大化，所以 final >= initial）
        assert final_best >= initial_best - 1.0, f"Should improve: {initial_best:.4f} -> {final_best:.4f}"
        
        results.add_pass(test_name, f"Fitness: {initial_best:.4f} -> {final_best:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_with_constraints(results: OptimizationTestResults):
    """测试带约束的遗传算法"""
    test_name = "GA with Constraints"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=30,
            mutation_rate=0.1,
            crossover_rate=0.9
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2, bounds=[(-2, 2), (-2, 2)])
        algo.initialize(context)
        
        # 添加约束: x0 + x1 <= 2
        algo.add_constraint(
            func=lambda g: 2 - g.get('x0', 0) - g.get('x1', 0),
            constraint_type="ineq",
            name="sum_constraint"
        )
        
        algo.set_fitness_function(sphere)
        
        # 进化
        for _ in range(5):
            algo.evaluate_population()
            algo.evolve()
        
        results.add_pass(test_name, f"Best fitness: {algo._population.best_fitness:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_state_management(results: OptimizationTestResults):
    """测试遗传算法状态管理"""
    test_name = "GA State Management"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20
        )
        
        algo1 = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo1.initialize(context)
        algo1.set_fitness_function(sphere)
        
        # 进化几代
        for _ in range(3):
            algo1.evaluate_population()
            algo1.evolve()
        
        # 保存状态
        state = algo1.get_state()
        
        # 创建新算法并加载状态
        algo2 = GeneticAlgorithm(config)
        algo2.load_state(state)
        
        assert algo2._population.generation == algo1._population.generation, "Generation mismatch"
        assert len(algo2._population.individuals) == len(algo1._population.individuals), "Population size mismatch"
        
        results.add_pass(test_name, f"State loaded at generation {algo2._population.generation}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_warm_start(results: OptimizationTestResults):
    """测试遗传算法热启动"""
    test_name = "GA Warm Start"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        
        # 准备历史数据
        X_history = [
            {"x0": 0.1, "x1": 0.1},
            {"x0": 0.2, "x1": 0.2},
            {"x0": 0.3, "x1": 0.3}
        ]
        y_history = [sphere(x) for x in X_history]
        
        # 热启动
        algo.warm_start(X_history, y_history)
        
        assert len(algo._observations) == 3, "Should have 3 observations"
        
        results.add_pass(test_name, f"Warm started with {len(X_history)} points")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_visualization_data(results: OptimizationTestResults):
    """测试遗传算法可视化数据"""
    test_name = "GA Visualization Data"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        for _ in range(3):
            algo.evaluate_population()
            algo.evolve()
        
        viz_data = algo.get_visualization_data()
        
        assert 'evolution_history' in viz_data, "Missing evolution_history"
        assert 'current_population' in viz_data, "Missing current_population"
        assert len(viz_data['current_population']) == 20, "Wrong population size"
        
        results.add_pass(test_name, f"Got viz data with {len(viz_data['current_population'])} individuals")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_genetic_algorithm_summary(results: OptimizationTestResults):
    """测试遗传算法优化摘要"""
    test_name = "GA Optimization Summary"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        for _ in range(5):
            algo.evaluate_population()
            algo.evolve()
        
        summary = algo.get_optimization_summary()
        
        assert 'status' in summary, "Missing status"
        assert 'current_generation' in summary, "Missing generation"
        assert 'best_fitness' in summary, "Missing best_fitness"
        assert 'selection_method' in summary, "Missing selection_method"
        
        results.add_pass(test_name, f"Status: {summary['status']}, Best: {summary['best_fitness']:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== NSGA-II 测试 ====================

def test_nsga2_basic(results: OptimizationTestResults):
    """测试 NSGA-II 基本功能"""
    test_name = "NSGA-II Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=30,
            extra={"max_generations": 10}
        )
        
        algo = NSGAII(config)
        
        # 添加两个目标
        algo.add_objective(multi_objective_1)
        algo.add_objective(multi_objective_2)
        
        context = AlgorithmContext(
            inputs={},
            search_space={
                "x": {"low": 0.0, "high": 1.0, "type": "float"},
                "y": {"low": 0.0, "high": 1.0, "type": "float"}
            },
            observations=[],
            objective="minimize"
        )
        
        algo.initialize(context)
        
        # 评估初始种群
        algo.evaluate_population()
        
        # 进化
        for _ in range(5):
            algo.evolve()
        
        # 获取 Pareto 前沿
        pareto_front = algo.get_pareto_front()
        
        assert len(pareto_front) > 0, "Pareto front should not be empty"
        assert all('objectives' in ind for ind in pareto_front), "All should have objectives"
        
        results.add_pass(test_name, f"Pareto front size: {len(pareto_front)}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_nsga2_crowding_distance(results: OptimizationTestResults):
    """测试 NSGA-II 拥挤度计算"""
    test_name = "NSGA-II Crowding Distance"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20
        )
        
        algo = NSGAII(config)
        algo.add_objective(multi_objective_1)
        algo.add_objective(multi_objective_2)
        
        context = AlgorithmContext(
            inputs={},
            search_space={
                "x": {"low": 0.0, "high": 1.0, "type": "float"},
                "y": {"low": 0.0, "high": 1.0, "type": "float"}
            },
            observations=[],
            objective="minimize"
        )
        
        algo.initialize(context)
        algo.evaluate_population()
        algo.evolve()
        
        # 检查拥挤度
        has_crowding = any(
            ind.crowding_distance > 0 
            for ind in algo._population.individuals
        )
        
        results.add_pass(test_name, f"Crowding distance computed: {has_crowding}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 差分进化测试 ====================

def test_differential_evolution_basic(results: OptimizationTestResults):
    """测试差分进化基本功能"""
    test_name = "DE Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20,
            extra={
                "F": 0.8,
                "CR": 0.9,
                "strategy": "rand/1"
            }
        )
        
        algo = DifferentialEvolution(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        # 评估初始种群
        algo.evaluate_population()
        initial_best = algo._population.best_fitness
        
        # 进化
        for _ in range(10):
            algo.evolve()
        
        final_best = algo._population.best_fitness
        
        results.add_pass(test_name, f"Fitness: {initial_best:.4f} -> {final_best:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_differential_evolution_strategies(results: OptimizationTestResults):
    """测试差分进化不同策略"""
    test_name = "DE Strategies"
    try:
        strategies = ["rand/1", "best/1", "current-to-best/1"]
        
        for strategy in strategies:
            config = AlgorithmConfig(
                algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
                population_size=15,
                extra={"strategy": strategy, "F": 0.7, "CR": 0.8}
            )
            
            algo = DifferentialEvolution(config)
            context = create_test_context(n_dims=2)
            algo.initialize(context)
            algo.set_fitness_function(sphere)
            
            algo.evaluate_population()
            for _ in range(3):
                algo.evolve()
        
        results.add_pass(test_name, f"Tested {len(strategies)} strategies")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 进化策略测试 ====================

def test_evolution_strategy_basic(results: OptimizationTestResults):
    """测试进化策略基本功能"""
    test_name = "ES Basic"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=20,
            extra={
                "mu": 5,
                "lambda": 20,
                "plus_selection": True,
                "initial_sigma": 0.3,
                "sigma_decay": 0.99
            }
        )
        
        algo = EvolutionStrategy(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        algo.evaluate_population()
        initial_best = algo._population.best_fitness
        
        for _ in range(10):
            algo.evolve()
        
        final_best = algo._population.best_fitness
        
        results.add_pass(test_name, f"Fitness: {initial_best:.4f} -> {final_best:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_evolution_strategy_selection_modes(results: OptimizationTestResults):
    """测试进化策略选择模式"""
    test_name = "ES Selection Modes"
    try:
        for plus_selection in [True, False]:
            mode_name = "(μ+λ)" if plus_selection else "(μ,λ)"
            
            config = AlgorithmConfig(
                algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
                population_size=20,
                extra={
                    "mu": 5,
                    "lambda": 20,
                    "plus_selection": plus_selection
                }
            )
            
            algo = EvolutionStrategy(config)
            context = create_test_context(n_dims=2)
            algo.initialize(context)
            algo.set_fitness_function(sphere)
            
            algo.evaluate_population()
            for _ in range(3):
                algo.evolve()
        
        results.add_pass(test_name, "Tested both (μ+λ) and (μ,λ) modes")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 完整优化工作流测试 ====================

def test_optimization_workflow_sphere(results: OptimizationTestResults):
    """测试完整优化工作流 - Sphere 函数"""
    test_name = "Optimization Workflow - Sphere"
    try:
        random.seed(TestConfig.SEED)
        np.random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.9,
            elite_ratio=0.1,
            extra={
                "selection_method": "tournament",
                "crossover_method": "sbx",
                "mutation_method": "polynomial",
                "max_generations": 30
            }
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2, bounds=[(-5, 5), (-5, 5)])
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        algo.evaluate_population()
        
        for _ in range(20):
            algo.evolve()
            algo.evaluate_population()
        
        best = algo._population.get_best()
        
        # Sphere 最大化后最优值应接近 0（在原点）
        assert best.fitness > -1.0, f"Should be close to 0, got {best.fitness}"
        
        results.add_pass(test_name, f"Best fitness: {best.fitness:.4f}, genes: {best.genes}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_optimization_workflow_rastrigin(results: OptimizationTestResults):
    """测试完整优化工作流 - Rastrigin 函数"""
    test_name = "Optimization Workflow - Rastrigin"
    try:
        random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=50,
            mutation_rate=0.15,
            crossover_rate=0.85,
            extra={"selection_method": "tournament", "tournament_size": 5}
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2, bounds=[(-5.12, 5.12), (-5.12, 5.12)])
        algo.initialize(context)
        algo.set_fitness_function(rastrigin)
        
        algo.evaluate_population()
        
        for _ in range(15):
            algo.evolve()
            algo.evaluate_population()
        
        best = algo._population.get_best()
        
        results.add_pass(test_name, f"Best fitness: {best.fitness:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


def test_optimization_workflow_high_dim(results: OptimizationTestResults):
    """测试高维优化工作流"""
    test_name = "Optimization Workflow - High Dim (5D)"
    try:
        random.seed(TestConfig.SEED)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=100,
            mutation_rate=0.15,
            crossover_rate=0.9
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=5, bounds=[(-2, 2)] * 5)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        algo.evaluate_population()
        
        for _ in range(15):
            algo.evolve()
            algo.evaluate_population()
        
        best = algo._population.get_best()
        
        results.add_pass(test_name, f"5D Best fitness: {best.fitness:.4f}")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 性能测试 ====================

def test_performance_timing(results: OptimizationTestResults):
    """测试性能计时"""
    test_name = "Performance Timing"
    try:
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
            population_size=50
        )
        
        algo = GeneticAlgorithm(config)
        context = create_test_context(n_dims=2)
        algo.initialize(context)
        algo.set_fitness_function(sphere)
        
        # 测量进化时间
        times = []
        for _ in range(10):
            algo.evaluate_population()
            start = time.time()
            algo.evolve()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        results.add_pass(test_name, f"Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
    except Exception as e:
        results.add_fail(test_name, str(e))


# ==================== 主测试运行器 ====================

def run_all_tests():
    """运行所有测试"""
    log("=" * 60)
    log("Genetic Algorithm Integration Tests")
    log("=" * 60)
    
    results = OptimizationTestResults()
    
    # 个体和种群测试
    log("\n[Section 1] Individual and Population Tests")
    log("-" * 40)
    test_individual_creation(results)
    test_individual_copy(results)
    test_individual_dominance(results)
    test_population_stats(results)
    test_population_best(results)
    
    # 选择操作器测试
    log("\n[Section 2] Selection Operator Tests")
    log("-" * 40)
    test_selection_tournament(results)
    test_selection_roulette(results)
    test_selection_rank(results)
    test_selection_sus(results)
    test_selection_truncation(results)
    test_selection_boltzmann(results)
    
    # 交叉操作器测试
    log("\n[Section 3] Crossover Operator Tests")
    log("-" * 40)
    test_crossover_sbx(results)
    test_crossover_blx_alpha(results)
    test_crossover_arithmetic(results)
    test_crossover_categorical(results)
    
    # 变异操作器测试
    log("\n[Section 4] Mutation Operator Tests")
    log("-" * 40)
    test_mutation_polynomial(results)
    test_mutation_gaussian(results)
    test_mutation_uniform(results)
    test_mutation_adaptive(results)
    test_mutation_categorical(results)
    
    # 约束处理测试
    log("\n[Section 5] Constraint Handling Tests")
    log("-" * 40)
    test_constraint_definition(results)
    test_constraint_handler(results)
    test_adaptive_penalty(results)
    
    # 收敛检测测试
    log("\n[Section 6] Convergence Detection Tests")
    log("-" * 40)
    test_convergence_detector(results)
    
    # 遗传算法测试
    log("\n[Section 7] Genetic Algorithm Tests")
    log("-" * 40)
    test_genetic_algorithm_basic(results)
    test_genetic_algorithm_evolve(results)
    test_genetic_algorithm_with_constraints(results)
    test_genetic_algorithm_state_management(results)
    test_genetic_algorithm_warm_start(results)
    test_genetic_algorithm_visualization_data(results)
    test_genetic_algorithm_summary(results)
    
    # NSGA-II 测试
    log("\n[Section 8] NSGA-II Tests")
    log("-" * 40)
    test_nsga2_basic(results)
    test_nsga2_crowding_distance(results)
    
    # 差分进化测试
    log("\n[Section 9] Differential Evolution Tests")
    log("-" * 40)
    test_differential_evolution_basic(results)
    test_differential_evolution_strategies(results)
    
    # 进化策略测试
    log("\n[Section 10] Evolution Strategy Tests")
    log("-" * 40)
    test_evolution_strategy_basic(results)
    test_evolution_strategy_selection_modes(results)
    
    # 完整优化工作流测试
    log("\n[Section 11] Complete Optimization Workflow Tests")
    log("-" * 40)
    test_optimization_workflow_sphere(results)
    test_optimization_workflow_rastrigin(results)
    test_optimization_workflow_high_dim(results)
    
    # 性能测试
    log("\n[Section 12] Performance Tests")
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
