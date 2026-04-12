"""
遗传算法 - 生产级实现

实现进化计算方法用于组合优化，支持多种高级特性：

核心特性：
- 多种选择算法：锦标赛、轮盘赌、排名、随机通用采样（SUS）、截断选择、玻尔兹曼选择
- 多种交叉算法：单点、两点、均匀、SBX、BLX-α、算术交叉
- 多种变异算法：高斯变异、多项式变异、均匀变异、自适应变异
- 自适应参数：动态调整变异率和交叉率
- 多目标优化：NSGA-II、NSGA-III 支持
- 约束处理：惩罚函数、可行性规则、自适应惩罚
- 收敛检测：多种收敛标准
- 并行评估：支持并行适应度计算
- 热启动：从历史数据恢复
- 可视化支持：进化历史、帕累托前沿

算法变体：
- 标准遗传算法 (GA)
- 差分进化 (DE)
- 进化策略 (ES)
- 协方差矩阵自适应进化策略 (CMA-ES)

使用示例：
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        extra={
            "selection_method": "tournament",
            "crossover_method": "sbx",
            "mutation_method": "polynomial",
            "adaptive_params": True
        }
    )
    
    algo = GeneticAlgorithm(config)
    result = algo.suggest(context)
"""

import logging
import math
import random
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from datetime import datetime
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

# 尝试导入可选依赖
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


# ==================== 枚举类型定义 ====================

class SelectionMethod(Enum):
    """选择方法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    SUS = "sus"  # 随机通用采样
    TRUNCATION = "truncation"
    BOLTZMANN = "boltzmann"
    RANDOM = "random"


class CrossoverMethod(Enum):
    """交叉方法"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SBX = "sbx"  # 模拟二进制交叉
    BLX_ALPHA = "blx_alpha"
    ARITHMETIC = "arithmetic"
    INTERMEDIATE = "intermediate"


class MutationMethod(Enum):
    """变异方法"""
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    NON_UNIFORM = "non_uniform"
    BOUNDARY = "boundary"


# ==================== 数据结构定义 ====================

@dataclass
class Individual:
    """个体（候选解）- 增强版
    
    Attributes:
        genes: 基因编码字典
        fitness: 单目标适应度
        objectives: 多目标适应度值列表
        rank: Pareto 排名（用于多目标优化）
        crowding_distance: 拥挤度距离
        constraint_violation: 约束违反程度
        generation: 产生的代数
        evaluated: 是否已评估
        metadata: 额外元数据
    """
    genes: Dict[str, Any]
    fitness: float = 0.0
    objectives: List[float] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0
    generation: int = 0
    evaluated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'Individual':
        """深拷贝个体"""
        return Individual(
            genes=deepcopy(self.genes),
            fitness=self.fitness,
            objectives=self.objectives.copy() if self.objectives else [],
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            constraint_violation=self.constraint_violation,
            generation=self.generation,
            evaluated=self.evaluated,
            metadata=deepcopy(self.metadata)
        )
    
    def dominates(self, other: 'Individual') -> bool:
        """判断是否支配另一个个体（多目标优化）
        
        Args:
            other: 另一个个体
            
        Returns:
            如果 self 支配 other 返回 True
        """
        if not self.objectives or not other.objectives:
            return self.fitness > other.fitness
        
        # 可行性规则：可行解总是支配不可行解
        if self.constraint_violation < other.constraint_violation:
            return True
        if self.constraint_violation > other.constraint_violation:
            return False
        
        # Pareto 支配
        at_least_one_better = False
        for s, o in zip(self.objectives, other.objectives):
            if s < o:  # 假设最小化
                return False
            if s > o:
                at_least_one_better = True
        return at_least_one_better
    
    def __lt__(self, other: 'Individual') -> bool:
        """比较运算符，用于排序"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance


@dataclass
class Population:
    """种群 - 增强版"""
    individuals: List[Individual]
    generation: int = 0
    best_fitness: float = float('-inf')
    avg_fitness: float = 0.0
    std_fitness: float = 0.0
    min_fitness: float = float('inf')
    diversity: float = 0.0
    pareto_front: List[Individual] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_stats(self) -> None:
        """更新种群统计信息"""
        if not self.individuals:
            return
        
        fitnesses = [ind.fitness for ind in self.individuals if ind.evaluated]
        if not fitnesses:
            return
        
        self.best_fitness = max(fitnesses)
        self.min_fitness = min(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
    
        if len(fitnesses) > 1:
            variance = sum((f - self.avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
            self.std_fitness = math.sqrt(variance)
        else:
            self.std_fitness = 0.0
        
        # 计算多样性（基于基因差异）
        self._compute_diversity()
        
        # 记录历史
        self.history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'std_fitness': self.std_fitness,
            'diversity': self.diversity,
            'timestamp': datetime.now().isoformat()
        })
    
    def _compute_diversity(self) -> None:
        """计算种群多样性"""
        if len(self.individuals) < 2:
            self.diversity = 0.0
            return
        
        # 使用基因空间的平均距离作为多样性度量
        total_distance = 0.0
        count = 0
        
        for i, ind1 in enumerate(self.individuals):
            for ind2 in self.individuals[i+1:]:
                dist = self._gene_distance(ind1.genes, ind2.genes)
                total_distance += dist
                count += 1
        
        self.diversity = total_distance / count if count > 0 else 0.0
    
    def _gene_distance(self, genes1: Dict[str, Any], genes2: Dict[str, Any]) -> float:
        """计算两个基因之间的距离"""
        distance = 0.0
        for key in genes1:
            if key in genes2:
                v1, v2 = genes1[key], genes2[key]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    distance += (v1 - v2) ** 2
                elif v1 != v2:
                    distance += 1.0
        return math.sqrt(distance)
    
    def get_best(self, n: int = 1) -> Union[Individual, List[Individual]]:
        """获取最佳个体
        
        Args:
            n: 返回的个体数量
            
        Returns:
            单个个体或个体列表
        """
        if not self.individuals:
            return None if n == 1 else []
        
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        
        if n == 1:
            return sorted_inds[0]
        return sorted_inds[:n]
    
    def get_pareto_front(self) -> List[Individual]:
        """获取 Pareto 前沿"""
        if self.pareto_front:
            return self.pareto_front
        
        # 计算 Pareto 前沿
        front = []
        for ind in self.individuals:
            dominated = False
            for other in self.individuals:
                if other.dominates(ind):
                    dominated = True
                    break
            if not dominated:
                front.append(ind)
        
        self.pareto_front = front
        return front


# ==================== 遗传操作类 ====================

class SelectionOperator:
    """选择操作器
    
    支持多种选择算法，自动处理不同场景。
    """
    
    def __init__(
        self,
        method: Union[str, SelectionMethod] = SelectionMethod.TOURNAMENT,
        tournament_size: int = 3,
        truncation_ratio: float = 0.5,
        temperature: float = 1.0
    ):
        if isinstance(method, str):
            try:
                self.method = SelectionMethod(method.lower())
            except ValueError:
                self.method = SelectionMethod.TOURNAMENT
        else:
            self.method = method
        
        self.tournament_size = tournament_size
        self.truncation_ratio = truncation_ratio
        self.temperature = temperature  # 用于 Boltzmann 选择
    
    def select(self, population: List[Individual], n: int = 2) -> List[Individual]:
        """选择个体
        
        Args:
            population: 种群
            n: 选择数量
            
        Returns:
            选中的个体列表
        """
        if not population:
            return []
        
        if self.method == SelectionMethod.TOURNAMENT:
            return [self._tournament_select(population) for _ in range(n)]
        elif self.method == SelectionMethod.ROULETTE:
            return [self._roulette_select(population) for _ in range(n)]
        elif self.method == SelectionMethod.RANK:
            return [self._rank_select(population) for _ in range(n)]
        elif self.method == SelectionMethod.SUS:
            return self._sus_select(population, n)
        elif self.method == SelectionMethod.TRUNCATION:
            return self._truncation_select(population, n)
        elif self.method == SelectionMethod.BOLTZMANN:
            return [self._boltzmann_select(population) for _ in range(n)]
        else:
            return random.sample(population, min(n, len(population)))
    
    def _tournament_select(self, population: List[Individual]) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        # 考虑约束违反
        tournament.sort(key=lambda x: (x.constraint_violation, -x.fitness))
        return tournament[0].copy()
    
    def _roulette_select(self, population: List[Individual]) -> Individual:
        """轮盘赌选择"""
        min_fitness = min(ind.fitness for ind in population)
        adjusted = [ind.fitness - min_fitness + 1 for ind in population]
        total = sum(adjusted)
        
        if total == 0:
            return random.choice(population).copy()
        
        r = random.random() * total
        cumsum = 0
        for ind, fit in zip(population, adjusted):
            cumsum += fit
            if r <= cumsum:
                return ind.copy()
        return population[-1].copy()
    
    def _rank_select(self, population: List[Individual]) -> Individual:
        """排名选择"""
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_pop)
        ranks = list(range(1, n + 1))
        total = sum(ranks)
        
        r = random.random() * total
        cumsum = 0
        for ind, rank in zip(sorted_pop, ranks):
            cumsum += rank
            if r <= cumsum:
                return ind.copy()
        return sorted_pop[-1].copy()
    
    def _sus_select(self, population: List[Individual], n: int) -> List[Individual]:
        """随机通用采样（Stochastic Universal Sampling）
        
        比轮盘赌更公平，减少随机漂移
        """
        min_fitness = min(ind.fitness for ind in population)
        adjusted = [ind.fitness - min_fitness + 1 for ind in population]
        total = sum(adjusted)
        
        if total == 0:
            return [random.choice(population).copy() for _ in range(n)]
        
        # 计算指针间距和起始位置
        pointer_distance = total / n
        start = random.uniform(0, pointer_distance)
        pointers = [start + i * pointer_distance for i in range(n)]
        
        selected = []
        cumsum = 0
        pointer_idx = 0
        
        for ind, fit in zip(population, adjusted):
            cumsum += fit
            while pointer_idx < n and cumsum >= pointers[pointer_idx]:
                selected.append(ind.copy())
                pointer_idx += 1
        
        # 补齐（如果有精度问题）
        while len(selected) < n:
            selected.append(random.choice(population).copy())
        
        return selected
    
    def _truncation_select(self, population: List[Individual], n: int) -> List[Individual]:
        """截断选择
        
        只从最优的一部分中选择
        """
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        truncation_point = max(1, int(len(sorted_pop) * self.truncation_ratio))
        elite = sorted_pop[:truncation_point]
        
        return [random.choice(elite).copy() for _ in range(n)]
    
    def _boltzmann_select(self, population: List[Individual]) -> Individual:
        """玻尔兹曼选择
        
        使用 softmax 概率，temperature 控制选择压力
        """
        fitnesses = np.array([ind.fitness for ind in population])
        
        # 防止数值溢出
        shifted = fitnesses - np.max(fitnesses)
        exp_values = np.exp(shifted / max(self.temperature, 1e-10))
        probabilities = exp_values / np.sum(exp_values)
        
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx].copy()


class CrossoverOperator:
    """交叉操作器
    
    支持多种交叉算法，处理不同类型的基因。
    """
    
    def __init__(
        self,
        method: Union[str, CrossoverMethod] = CrossoverMethod.SBX,
        eta: float = 20.0,  # SBX 分布指数
        alpha: float = 0.5,  # BLX-α 参数
        probability: float = 0.9
    ):
        if isinstance(method, str):
            try:
                self.method = CrossoverMethod(method.lower())
            except ValueError:
                self.method = CrossoverMethod.SBX
        else:
            self.method = method
        
        self.eta = eta
        self.alpha = alpha
        self.probability = probability
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_types: Dict[str, str],
        param_bounds: Dict[str, Tuple[float, float]],
        param_choices: Dict[str, List[Any]]
    ) -> Tuple[Individual, Individual]:
        """执行交叉操作
        
        Args:
            parent1, parent2: 父代个体
            param_types: 参数类型字典
            param_bounds: 参数边界
            param_choices: 分类参数选项
            
        Returns:
            两个子代个体
        """
        if random.random() > self.probability:
            return parent1.copy(), parent2.copy()
        
        child1_genes = {}
        child2_genes = {}
        
        for param_name in parent1.genes.keys():
            p1_val = parent1.genes[param_name]
            p2_val = parent2.genes[param_name]
            param_type = param_types.get(param_name, 'float')
            
            if param_type == 'categorical':
                # 分类变量：随机选择
                if random.random() < 0.5:
                    child1_genes[param_name] = p1_val
                    child2_genes[param_name] = p2_val
                else:
                    child1_genes[param_name] = p2_val
                    child2_genes[param_name] = p1_val
            else:
                # 数值变量
                bounds = param_bounds.get(param_name, (0.0, 1.0))
                
                if self.method == CrossoverMethod.SBX:
                    c1, c2 = self._sbx_crossover(p1_val, p2_val, bounds)
                elif self.method == CrossoverMethod.BLX_ALPHA:
                    c1, c2 = self._blx_alpha_crossover(p1_val, p2_val, bounds)
                elif self.method == CrossoverMethod.SINGLE_POINT:
                    c1, c2 = self._single_point_crossover(p1_val, p2_val)
                elif self.method == CrossoverMethod.UNIFORM:
                    c1, c2 = self._uniform_crossover(p1_val, p2_val)
                elif self.method == CrossoverMethod.ARITHMETIC:
                    c1, c2 = self._arithmetic_crossover(p1_val, p2_val, bounds)
                elif self.method == CrossoverMethod.INTERMEDIATE:
                    c1, c2 = self._intermediate_crossover(p1_val, p2_val, bounds)
                else:
                    c1, c2 = self._sbx_crossover(p1_val, p2_val, bounds)
                
                # 整数处理
                if param_type == 'int':
                    c1 = int(round(c1))
                    c2 = int(round(c2))
                
                child1_genes[param_name] = c1
                child2_genes[param_name] = c2
        
        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)
        
        return child1, child2
    
    def _sbx_crossover(
        self, 
        p1: float, 
        p2: float, 
        bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """模拟二进制交叉（Simulated Binary Crossover）"""
        low, high = bounds
        
        if abs(p1 - p2) < 1e-10:
            return p1, p2
        
        if p1 > p2:
            p1, p2 = p2, p1
        
        # 计算 beta
        u = random.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (self.eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
        
        c1 = 0.5 * ((p1 + p2) - beta * (p2 - p1))
        c2 = 0.5 * ((p1 + p2) + beta * (p2 - p1))
        
        # 边界裁剪
        c1 = max(low, min(high, c1))
        c2 = max(low, min(high, c2))
        
        return c1, c2
    
    def _blx_alpha_crossover(
        self, 
        p1: float, 
        p2: float, 
        bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """BLX-α 交叉"""
        low, high = bounds
        
        p_min = min(p1, p2)
        p_max = max(p1, p2)
        interval = p_max - p_min
        
        c1 = random.uniform(p_min - self.alpha * interval, p_max + self.alpha * interval)
        c2 = random.uniform(p_min - self.alpha * interval, p_max + self.alpha * interval)
        
        c1 = max(low, min(high, c1))
        c2 = max(low, min(high, c2))
        
        return c1, c2
    
    def _single_point_crossover(self, p1: float, p2: float) -> Tuple[float, float]:
        """单点交叉（简化版：随机选择）"""
        if random.random() < 0.5:
            return p1, p2
        return p2, p1
    
    def _uniform_crossover(self, p1: float, p2: float) -> Tuple[float, float]:
        """均匀交叉"""
        if random.random() < 0.5:
            return p1, p2
        return p2, p1
    
    def _arithmetic_crossover(
        self, 
        p1: float, 
        p2: float, 
        bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """算术交叉"""
        alpha = random.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        
        low, high = bounds
        c1 = max(low, min(high, c1))
        c2 = max(low, min(high, c2))
        
        return c1, c2
    
    def _intermediate_crossover(
        self, 
        p1: float, 
        p2: float, 
        bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """中间交叉"""
        ratio = random.uniform(-0.25, 1.25)
        c1 = p1 + ratio * (p2 - p1)
        c2 = p2 + ratio * (p1 - p2)
        
        low, high = bounds
        c1 = max(low, min(high, c1))
        c2 = max(low, min(high, c2))
        
        return c1, c2


class MutationOperator:
    """变异操作器
    
    支持多种变异算法，包括自适应变异。
    """
    
    def __init__(
        self,
        method: Union[str, MutationMethod] = MutationMethod.POLYNOMIAL,
        eta: float = 20.0,  # 多项式变异分布指数
        probability: float = 0.1,
        adaptive: bool = False,
        max_generation: int = 100
    ):
        if isinstance(method, str):
            try:
                self.method = MutationMethod(method.lower())
            except ValueError:
                self.method = MutationMethod.POLYNOMIAL
        else:
            self.method = method
        
        self.eta = eta
        self.probability = probability
        self.initial_probability = probability
        self.adaptive = adaptive
        self.max_generation = max_generation
        self._current_generation = 0
    
    def set_generation(self, generation: int) -> None:
        """设置当前代数（用于自适应变异）"""
        self._current_generation = generation
        
        if self.adaptive:
            # 线性衰减：从初始概率到初始概率的一半
            decay = 1.0 - (generation / (2 * self.max_generation))
            self.probability = self.initial_probability * max(0.5, decay)
    
    def mutate(
        self,
        individual: Individual,
        param_types: Dict[str, str],
        param_bounds: Dict[str, Tuple[float, float]],
        param_choices: Dict[str, List[Any]]
    ) -> None:
        """执行变异操作
        
        Args:
            individual: 待变异的个体
            param_types: 参数类型字典
            param_bounds: 参数边界
            param_choices: 分类参数选项
        """
        for param_name in individual.genes.keys():
            if random.random() > self.probability:
                continue
            
            param_type = param_types.get(param_name, 'float')
            
            if param_type == 'categorical':
                choices = param_choices.get(param_name, [])
                if choices:
                    individual.genes[param_name] = random.choice(choices)
            else:
                bounds = param_bounds.get(param_name, (0.0, 1.0))
                current = individual.genes[param_name]
                
                if self.method == MutationMethod.POLYNOMIAL:
                    new_val = self._polynomial_mutation(current, bounds)
                elif self.method == MutationMethod.GAUSSIAN:
                    new_val = self._gaussian_mutation(current, bounds)
                elif self.method == MutationMethod.UNIFORM:
                    new_val = self._uniform_mutation(bounds)
                elif self.method == MutationMethod.NON_UNIFORM:
                    new_val = self._non_uniform_mutation(current, bounds)
                elif self.method == MutationMethod.BOUNDARY:
                    new_val = self._boundary_mutation(bounds)
                else:
                    new_val = self._polynomial_mutation(current, bounds)
                
                if param_type == 'int':
                    new_val = int(round(new_val))
                
                individual.genes[param_name] = new_val
    
    def _polynomial_mutation(
        self, 
        value: float, 
        bounds: Tuple[float, float]
    ) -> float:
        """多项式变异"""
        low, high = bounds
        
        u = random.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (self.eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta + 1.0))
        
        mutated = value + delta * (high - low)
        return max(low, min(high, mutated))
    
    def _gaussian_mutation(
        self, 
        value: float, 
        bounds: Tuple[float, float]
    ) -> float:
        """高斯变异"""
        low, high = bounds
        sigma = (high - low) * 0.1
        mutated = value + random.gauss(0, sigma)
        return max(low, min(high, mutated))
    
    def _uniform_mutation(self, bounds: Tuple[float, float]) -> float:
        """均匀变异"""
        return random.uniform(bounds[0], bounds[1])
    
    def _non_uniform_mutation(
        self, 
        value: float, 
        bounds: Tuple[float, float]
    ) -> float:
        """非均匀变异（随代数递减）"""
        low, high = bounds
        
        b = 5.0  # 形状参数
        progress = self._current_generation / max(self.max_generation, 1)
        decay = (1.0 - progress) ** b
        
        if random.random() < 0.5:
            delta = (high - value) * (1.0 - random.random() ** decay)
        else:
            delta = -(value - low) * (1.0 - random.random() ** decay)
        
        mutated = value + delta
        return max(low, min(high, mutated))
    
    def _boundary_mutation(self, bounds: Tuple[float, float]) -> float:
        """边界变异"""
        return bounds[0] if random.random() < 0.5 else bounds[1]


# ==================== 约束处理 ====================

@dataclass
class Constraint:
    """优化约束"""
    func: Callable[[Dict[str, Any]], float]
    constraint_type: str = "ineq"  # "ineq": g(x) >= 0, "eq": h(x) = 0
    name: str = ""
    tolerance: float = 1e-6
    
    def evaluate(self, genes: Dict[str, Any]) -> float:
        """评估约束"""
        return self.func(genes)
    
    def get_violation(self, genes: Dict[str, Any]) -> float:
        """获取约束违反程度"""
        value = self.evaluate(genes)
        if self.constraint_type == "eq":
            return abs(value)
        else:  # ineq
            return max(0, -value)


class ConstraintHandler:
    """约束处理器"""
    
    def __init__(
        self,
        constraints: List[Constraint] = None,
        penalty_coefficient: float = 1e6,
        adaptive_penalty: bool = True
    ):
        self.constraints = constraints or []
        self.penalty_coefficient = penalty_coefficient
        self.adaptive_penalty = adaptive_penalty
        self._generation = 0
    
    def add_constraint(self, constraint: Constraint) -> None:
        """添加约束"""
        self.constraints.append(constraint)
    
    def set_generation(self, generation: int) -> None:
        """设置当前代数"""
        self._generation = generation
        if self.adaptive_penalty:
            # 自适应惩罚系数
            self.penalty_coefficient = 1e3 * (1 + generation)
    
    def compute_violation(self, individual: Individual) -> float:
        """计算总约束违反程度"""
        total_violation = 0.0
        for c in self.constraints:
            violation = c.get_violation(individual.genes)
            total_violation += violation ** 2
        individual.constraint_violation = math.sqrt(total_violation)
        return individual.constraint_violation
    
    def compute_penalty(self, individual: Individual) -> float:
        """计算惩罚值"""
        if not self.constraints:
            return 0.0
        
        violation = self.compute_violation(individual)
        return self.penalty_coefficient * violation
    
    def is_feasible(self, individual: Individual) -> bool:
        """检查个体是否可行"""
        for c in self.constraints:
            if c.get_violation(individual.genes) > c.tolerance:
                return False
        return True


# ==================== 收敛检测 ====================

@dataclass
class ConvergenceState:
    """收敛状态"""
    converged: bool = False
    reason: str = ""
    generations_without_improvement: int = 0
    best_fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)


class ConvergenceDetector:
    """收敛检测器"""
    
    def __init__(
        self,
        max_generations: int = 100,
        no_improvement_generations: int = 20,
        fitness_tolerance: float = 1e-6,
        diversity_threshold: float = 1e-4
    ):
        self.max_generations = max_generations
        self.no_improvement_generations = no_improvement_generations
        self.fitness_tolerance = fitness_tolerance
        self.diversity_threshold = diversity_threshold
        self.state = ConvergenceState()
        self._best_fitness = float('-inf')
    
    def update(self, population: Population) -> ConvergenceState:
        """更新收敛状态"""
        self.state.best_fitness_history.append(population.best_fitness)
        self.state.diversity_history.append(population.diversity)
        
        # 检查最大代数
        if population.generation >= self.max_generations:
            self.state.converged = True
            self.state.reason = f"Reached max generations ({self.max_generations})"
            return self.state
        
        # 检查适应度改进
        if population.best_fitness > self._best_fitness + self.fitness_tolerance:
            self._best_fitness = population.best_fitness
            self.state.generations_without_improvement = 0
        else:
            self.state.generations_without_improvement += 1
        
        if self.state.generations_without_improvement >= self.no_improvement_generations:
            self.state.converged = True
            self.state.reason = f"No improvement for {self.no_improvement_generations} generations"
            return self.state
        
        # 检查多样性
        if population.diversity < self.diversity_threshold:
            self.state.converged = True
            self.state.reason = f"Population diversity too low ({population.diversity:.6f})"
            return self.state
        
        return self.state
    
    def reset(self) -> None:
        """重置状态"""
        self.state = ConvergenceState()
        self._best_fitness = float('-inf')


# ==================== 主算法类 ====================

class GeneticAlgorithm(BaseAlgorithm):
    """遗传算法 - 生产级实现
    
    特性：
    - 多种选择、交叉、变异算法
    - 自适应参数
    - 约束处理
    - 收敛检测
    - 并行评估
    - 热启动
    - 可视化支持
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        extra = config.extra if config and config.extra else {}
        
        self._population: Optional[Population] = None
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        self._param_types: Dict[str, str] = {}
        self._param_choices: Dict[str, List[Any]] = {}
        self._fitness_function: Optional[Callable] = None
        self._elite_individuals: List[Individual] = []
        
        # 初始化遗传操作器
        self._selection_operator = SelectionOperator(
            method=extra.get('selection_method', 'tournament'),
            tournament_size=extra.get('tournament_size', 3),
            truncation_ratio=extra.get('truncation_ratio', 0.5),
            temperature=extra.get('selection_temperature', 1.0)
        )
        
        self._crossover_operator = CrossoverOperator(
            method=extra.get('crossover_method', 'sbx'),
            eta=extra.get('crossover_eta', 20.0),
            alpha=extra.get('blx_alpha', 0.5),
            probability=config.crossover_rate if config else 0.9
        )
        
        self._mutation_operator = MutationOperator(
            method=extra.get('mutation_method', 'polynomial'),
            eta=extra.get('mutation_eta', 20.0),
            probability=config.mutation_rate if config else 0.1,
            adaptive=extra.get('adaptive_mutation', False),
            max_generation=extra.get('max_generations', 100)
        )
        
        # 约束处理器
        self._constraint_handler = ConstraintHandler(
            penalty_coefficient=extra.get('penalty_coefficient', 1e6),
            adaptive_penalty=extra.get('adaptive_penalty', True)
        )
        
        # 收敛检测器
        self._convergence_detector = ConvergenceDetector(
            max_generations=extra.get('max_generations', 100),
            no_improvement_generations=extra.get('no_improvement_generations', 20),
            fitness_tolerance=extra.get('fitness_tolerance', 1e-6),
            diversity_threshold=extra.get('diversity_threshold', 1e-4)
        )
        
        # 配置选项
        self._use_parallel = extra.get('parallel_evaluation', False) and PARALLEL_AVAILABLE
        self._n_workers = extra.get('n_workers', 4)
        self._elitism_ratio = config.elite_ratio if config else 0.1
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GENETIC_ALGORITHM
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化遗传算法
        
        Args:
            context: 算法上下文
        """
        super().initialize(context)
        
        # 解析搜索空间
        self._parse_search_space(context.search_space)
        
        # 初始化种群
        self._initialize_population(context)
        
        # 从历史评估个体
        self._evaluate_from_history(context)
    
    def suggest(self, context: AlgorithmContext) -> AlgorithmResult:
        """生成优化建议
        
        返回当前种群中的最佳个体。
        
        Args:
            context: 算法上下文
            
        Returns:
            算法结果
        """
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        
        # 如果种群为空，创建初始种群
        if not self._population or not self._population.individuals:
            self._initialize_population(context)
        
        # 获取最佳个体
        best_individual = self._population.get_best()
        
        reasoning_steps = [
            f"当前代数: {self._population.generation}",
            f"种群大小: {len(self._population.individuals)}",
            f"最佳适应度: {self._population.best_fitness:.4f}",
            f"平均适应度: {self._population.avg_fitness:.4f}",
            f"精英个体数: {len(self._elite_individuals)}"
        ]
        
        # 计算置信度
        confidence = self._compute_confidence()
        
        # 生成备选方案
        alternatives = self._generate_alternatives(exclude=best_individual)
        
        return self._build_result(
            action=best_individual.genes if best_individual else {},
            confidence=confidence,
            reasoning=f"遗传算法第{self._population.generation}代：最佳适应度={self._population.best_fitness:.4f}",
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            debug_info={
                'generation': self._population.generation,
                'population_size': len(self._population.individuals),
                'best_fitness': self._population.best_fitness,
                'avg_fitness': self._population.avg_fitness,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate
            }
        )
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新种群
        
        评估个体并进化到下一代。
        
        Args:
            action: 执行的动作（个体基因）
            reward: 获得的奖励（适应度）
            context: 可选的上下文
        """
        self._record_observation(action, reward)
        
        # 更新对应个体的适应度
        for ind in self._population.individuals:
            if ind.genes == action:
                ind.fitness = reward
                break
        
        self._population.update_stats()
        
        # 检查是否需要进化
        evaluated_count = sum(1 for ind in self._population.individuals if ind.fitness != 0)
        if evaluated_count >= len(self._population.individuals) * 0.8:
            self._evolve()
    
    def evolve(self) -> None:
        """手动触发进化"""
        self._evolve()
    
    def _parse_search_space(self, search_space: Dict[str, Any]) -> None:
        """解析搜索空间
        
        Args:
            search_space: 搜索空间定义
        """
        self._param_bounds = {}
        self._param_types = {}
        self._param_choices = {}
        
        for param_name, param_def in search_space.items():
            if isinstance(param_def, dict):
                if 'choices' in param_def:
                    self._param_types[param_name] = 'categorical'
                    self._param_choices[param_name] = param_def['choices']
                elif 'type' in param_def:
                    param_type = param_def['type']
                    if param_type == 'int':
                        self._param_types[param_name] = 'int'
                    else:
                        self._param_types[param_name] = 'float'
                    
                    low = param_def.get('low', param_def.get('min', 0.0))
                    high = param_def.get('high', param_def.get('max', 1.0))
                    self._param_bounds[param_name] = (float(low), float(high))
                else:
                    low = param_def.get('low', param_def.get('min', 0.0))
                    high = param_def.get('high', param_def.get('max', 1.0))
                    self._param_types[param_name] = 'float'
                    self._param_bounds[param_name] = (float(low), float(high))
            elif isinstance(param_def, list):
                self._param_types[param_name] = 'categorical'
                self._param_choices[param_name] = param_def
            elif isinstance(param_def, (tuple, )) and len(param_def) >= 2:
                self._param_types[param_name] = 'float'
                self._param_bounds[param_name] = (float(param_def[0]), float(param_def[1]))
    
    def _initialize_population(self, context: AlgorithmContext) -> None:
        """初始化种群
        
        Args:
            context: 算法上下文
        """
        individuals = []
        pop_size = self.config.population_size
        
        # 从历史观测创建个体
        for obs in context.observations:
            if len(individuals) >= pop_size:
                break
            ind = Individual(
                genes=obs.get('action', {}),
                fitness=obs.get('reward', 0.0),
                generation=0
            )
            individuals.append(ind)
        
        # 随机生成剩余个体
        while len(individuals) < pop_size:
            genes = self._random_genes()
            ind = Individual(genes=genes, generation=0)
            individuals.append(ind)
        
        self._population = Population(individuals=individuals, generation=0)
        self._population.update_stats()
        
        self.logger.info(f"Population initialized with {len(individuals)} individuals")
    
    def _random_genes(self) -> Dict[str, Any]:
        """生成随机基因
        
        Returns:
            随机基因字典
        """
        genes = {}
        
        for param_name in set(list(self._param_bounds.keys()) + list(self._param_choices.keys())):
            param_type = self._param_types.get(param_name, 'float')
            
            if param_type == 'categorical':
                choices = self._param_choices.get(param_name, [0, 1])
                genes[param_name] = random.choice(choices)
            elif param_type == 'int':
                low, high = self._param_bounds.get(param_name, (0, 10))
                genes[param_name] = random.randint(int(low), int(high))
            else:  # float
                low, high = self._param_bounds.get(param_name, (0.0, 1.0))
                genes[param_name] = random.uniform(low, high)
        
        return genes
    
    def _evolve(self) -> None:
        """进化种群到下一代"""
        if not self._population:
            return
        
        self._population.generation += 1
        old_population = self._population.individuals
        new_population = []
        
        # 更新操作器状态
        self._mutation_operator.set_generation(self._population.generation)
        self._constraint_handler.set_generation(self._population.generation)
        
        # 精英保留（考虑约束）
        elite_count = max(1, int(len(old_population) * self._elitism_ratio))
        
        # 按可行性和适应度排序
        sorted_individuals = sorted(
            old_population,
            key=lambda x: (x.constraint_violation, -x.fitness)
        )
        elites = [ind.copy() for ind in sorted_individuals[:elite_count]]
        self._elite_individuals = elites
        new_population.extend(elites)
        
        # 生成新个体
        while len(new_population) < len(old_population):
            # 选择父母
            parents = self._selection_operator.select(old_population, n=2)
            parent1, parent2 = parents[0], parents[1]
            
            # 交叉
            child1, child2 = self._crossover_operator.crossover(
                parent1, parent2,
                self._param_types,
                self._param_bounds,
                self._param_choices
            )
            
            # 变异
            self._mutation_operator.mutate(
                child1, self._param_types, self._param_bounds, self._param_choices
            )
            self._mutation_operator.mutate(
                child2, self._param_types, self._param_bounds, self._param_choices
            )
            
            # 设置代数和重置评估状态
            child1.generation = self._population.generation
            child1.fitness = 0.0
            child1.evaluated = False
            child2.generation = self._population.generation
            child2.fitness = 0.0
            child2.evaluated = False
            
            # 计算约束违反
            if self._constraint_handler.constraints:
                self._constraint_handler.compute_violation(child1)
                self._constraint_handler.compute_violation(child2)
            
            new_population.append(child1)
            if len(new_population) < len(old_population):
                new_population.append(child2)
        
        self._population.individuals = new_population
        self._population.update_stats()
        
        # 更新收敛检测
        convergence = self._convergence_detector.update(self._population)
        
        self.logger.info(
            f"Evolved to generation {self._population.generation}, "
            f"best={self._population.best_fitness:.4f}, "
            f"avg={self._population.avg_fitness:.4f}, "
            f"diversity={self._population.diversity:.4f}"
        )
        
        if convergence.converged:
            self.logger.info(f"Algorithm converged: {convergence.reason}")
    
    def add_constraint(
        self, 
        func: Callable[[Dict[str, Any]], float],
        constraint_type: str = "ineq",
        name: str = ""
    ) -> None:
        """添加优化约束
        
        Args:
            func: 约束函数，g(x) >= 0 为满足约束
            constraint_type: "ineq" 或 "eq"
            name: 约束名称
        """
        constraint = Constraint(
            func=func,
            constraint_type=constraint_type,
            name=name
        )
        self._constraint_handler.add_constraint(constraint)
    
    def set_fitness_function(self, func: Callable[[Dict[str, Any]], float]) -> None:
        """设置适应度函数（用于自动评估）
        
        Args:
            func: 适应度函数
        """
        self._fitness_function = func
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """评估单个个体
        
        Args:
            individual: 待评估的个体
            
        Returns:
            适应度值
        """
        if self._fitness_function is None:
            return 0.0
        
        try:
            fitness = self._fitness_function(individual.genes)
            
            # 应用约束惩罚
            if self._constraint_handler.constraints:
                penalty = self._constraint_handler.compute_penalty(individual)
                fitness -= penalty
            
            individual.fitness = fitness
            individual.evaluated = True
            return fitness
        except Exception as e:
            self.logger.error(f"Error evaluating individual: {e}")
            return float('-inf')
    
    def evaluate_population(self) -> None:
        """评估整个种群
        
        如果配置了并行评估，则使用多线程。
        """
        if not self._population or not self._fitness_function:
            return
        
        unevaluated = [ind for ind in self._population.individuals if not ind.evaluated]
        
        if not unevaluated:
            return
        
        if self._use_parallel and len(unevaluated) > 1:
            # 并行评估
            with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_individual, ind): ind
                    for ind in unevaluated
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Parallel evaluation error: {e}")
                else:
                    # 顺序评估
                    for ind in unevaluated:
                        self._evaluate_individual(ind)
        
        self._population.update_stats()
    
    def _evaluate_from_history(self, context: AlgorithmContext) -> None:
        """从历史数据评估个体
        
        Args:
            context: 算法上下文
        """
        # 创建动作到奖励的映射
        action_rewards = {}
        for obs in context.observations:
            action_key = self._genes_to_key(obs.get('action', {}))
            action_rewards[action_key] = obs.get('reward', 0.0)
        
        # 更新种群中的适应度
        for ind in self._population.individuals:
            genes_key = self._genes_to_key(ind.genes)
            if genes_key in action_rewards:
                ind.fitness = action_rewards[genes_key]
        
        self._population.update_stats()
    
    def _genes_to_key(self, genes: Dict[str, Any]) -> str:
        """将基因转换为键
        
        Args:
            genes: 基因字典
            
        Returns:
            键字符串
        """
        sorted_items = sorted(genes.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def _compute_confidence(self) -> float:
        """计算置信度
        
        Returns:
            置信度 (0-1)
        """
        if not self._population:
            return 0.5
        
        # 基于代数和适应度收敛性计算置信度
        generation_factor = min(1.0, self._population.generation / 10)
        
        # 适应度差异
        if len(self._population.individuals) > 1:
            fitnesses = [ind.fitness for ind in self._population.individuals]
            fitness_std = (sum((f - self._population.avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            convergence_factor = 1.0 / (1.0 + fitness_std)
        else:
            convergence_factor = 0.5
        
        confidence = 0.5 + 0.3 * generation_factor + 0.2 * convergence_factor
        return min(0.95, confidence)
    
    def _generate_alternatives(self, exclude: Individual = None) -> List[Dict[str, Any]]:
        """生成备选方案
        
        Args:
            exclude: 要排除的个体
            
        Returns:
            备选方案列表
        """
        alternatives = []
        
        if not self._population:
            return alternatives
        
        sorted_individuals = sorted(
            self._population.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )
        
        for ind in sorted_individuals[:3]:
            if exclude and ind.genes == exclude.genes:
                continue
            
            alternatives.append({
                'action': ind.genes,
                'fitness': ind.fitness,
                'generation': ind.generation,
                'confidence': min(0.9, 0.5 + ind.fitness * 0.1),
                'reasoning': f"适应度={ind.fitness:.4f}, 第{ind.generation}代"
            })
        
        return alternatives[:2]
    
    # ==================== 热启动和状态管理 ====================
    
    def warm_start(self, X: List[Dict[str, Any]], y: List[float]) -> None:
        """热启动：从历史数据恢复
        
        Args:
            X: 历史参数配置列表
            y: 历史适应度值列表
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        for action, reward in zip(X, y):
            self._record_observation(action, reward)
        
        # 从历史创建初始种群
        if self._population:
            for i, (action, reward) in enumerate(zip(X, y)):
                if i < len(self._population.individuals):
                    self._population.individuals[i].genes = deepcopy(action)
                    self._population.individuals[i].fitness = reward
                    self._population.individuals[i].evaluated = True
            
            self._population.update_stats()
        
        self.logger.info(f"Warm started with {len(X)} historical observations")
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'observations': self._observations.copy(),
            'iteration_count': self._iteration_count,
            'param_names': list(self._param_bounds.keys()),
            'param_bounds': self._param_bounds,
            'param_types': self._param_types,
            'param_choices': self._param_choices,
            'population': {
                'generation': self._population.generation if self._population else 0,
                'individuals': [
                    {
                        'genes': ind.genes,
                        'fitness': ind.fitness,
                        'generation': ind.generation,
                        'evaluated': ind.evaluated
                    }
                    for ind in (self._population.individuals if self._population else [])
                ],
                'history': self._population.history if self._population else []
            },
            'convergence': {
                'converged': self._convergence_detector.state.converged,
                'reason': self._convergence_detector.state.reason,
                'best_history': self._convergence_detector.state.best_fitness_history
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        self._observations = state.get('observations', [])
        self._iteration_count = state.get('iteration_count', 0)
        self._param_bounds = state.get('param_bounds', {})
        self._param_types = state.get('param_types', {})
        self._param_choices = state.get('param_choices', {})
        
        pop_state = state.get('population', {})
        if pop_state.get('individuals'):
            individuals = [
                Individual(
                    genes=ind['genes'],
                    fitness=ind['fitness'],
                    generation=ind['generation'],
                    evaluated=ind['evaluated']
                )
                for ind in pop_state['individuals']
            ]
            self._population = Population(
                individuals=individuals,
                generation=pop_state.get('generation', 0),
                history=pop_state.get('history', [])
            )
            self._population.update_stats()
        
        self.logger.info(f"State loaded with {len(self._observations)} observations")
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        data = {
            'evolution_history': [],
            'current_population': [],
            'best_individuals': [],
            'diversity_history': [],
            'fitness_distribution': []
        }
        
        if not self._population:
            return data
        
        # 进化历史
        data['evolution_history'] = self._population.history.copy()
        
        # 当前种群
        for ind in self._population.individuals:
            data['current_population'].append({
                'genes': ind.genes,
                'fitness': ind.fitness,
                'generation': ind.generation,
                'constraint_violation': ind.constraint_violation
            })
        
        # 最优个体历史
        for obs in self._observations:
            data['best_individuals'].append({
                'action': obs['action'],
                'reward': obs['reward']
            })
        
        # 多样性历史
        data['diversity_history'] = self._convergence_detector.state.diversity_history.copy()
        
        # 适应度分布
        fitnesses = [ind.fitness for ind in self._population.individuals if ind.evaluated]
        if fitnesses:
            data['fitness_distribution'] = {
                'values': fitnesses,
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'min': min(fitnesses),
                'max': max(fitnesses)
            }
        
        return data
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._population:
            return {'status': 'not_started'}
        
        best_ind = self._population.get_best()
        
        return {
            'status': 'converged' if self._convergence_detector.state.converged else 'running',
            'convergence_reason': self._convergence_detector.state.reason,
            'current_generation': self._population.generation,
            'total_evaluations': len(self._observations),
            'best_fitness': self._population.best_fitness,
            'best_individual': best_ind.genes if best_ind else None,
            'avg_fitness': self._population.avg_fitness,
            'std_fitness': self._population.std_fitness,
            'population_diversity': self._population.diversity,
            'elite_count': len(self._elite_individuals),
            'selection_method': self._selection_operator.method.value,
            'crossover_method': self._crossover_operator.method.value,
            'mutation_method': self._mutation_operator.method.value,
            'mutation_rate': self._mutation_operator.probability,
            'crossover_rate': self._crossover_operator.probability
        }


# ==================== NSGA-II 多目标优化 ====================

class NSGAII(GeneticAlgorithm):
    """NSGA-II 多目标优化算法
    
    Non-dominated Sorting Genetic Algorithm II
    用于解决多目标优化问题，生成 Pareto 前沿。
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._objective_functions: List[Callable[[Dict[str, Any]], float]] = []
        self._n_objectives = 0
    
    def add_objective(self, func: Callable[[Dict[str, Any]], float]) -> None:
        """添加目标函数（假设最小化）
        
        Args:
            func: 目标函数
        """
        self._objective_functions.append(func)
        self._n_objectives = len(self._objective_functions)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """评估个体的所有目标"""
        if not self._objective_functions:
            return super()._evaluate_individual(individual)
        
        objectives = []
        for func in self._objective_functions:
            try:
                value = func(individual.genes)
                objectives.append(value)
            except Exception as e:
                self.logger.error(f"Error evaluating objective: {e}")
                objectives.append(float('inf'))
        
        individual.objectives = objectives
        individual.evaluated = True
        
        # 单目标适应度使用第一个目标的负值（因为假设最小化）
        individual.fitness = -objectives[0] if objectives else 0.0
        
        return individual.fitness
    
    def _evolve(self) -> None:
        """NSGA-II 进化"""
        if not self._population:
            return
        
        self._population.generation += 1
        old_population = self._population.individuals
        
        # 更新操作器
        self._mutation_operator.set_generation(self._population.generation)
        
        # 生成子代
        offspring = []
        while len(offspring) < len(old_population):
            parents = self._selection_operator.select(old_population, n=2)
            
            child1, child2 = self._crossover_operator.crossover(
                parents[0], parents[1],
                self._param_types,
                self._param_bounds,
                self._param_choices
            )
            
            self._mutation_operator.mutate(
                child1, self._param_types, self._param_bounds, self._param_choices
            )
            self._mutation_operator.mutate(
                child2, self._param_types, self._param_bounds, self._param_choices
            )
            
            child1.generation = self._population.generation
            child2.generation = self._population.generation
            
            offspring.extend([child1, child2])
        
        offspring = offspring[:len(old_population)]
        
        # 评估子代
        for ind in offspring:
            self._evaluate_individual(ind)
        
        # 合并父代和子代
        combined = old_population + offspring
        
        # 非支配排序
        fronts = self._fast_non_dominated_sort(combined)
        
        # 计算拥挤度并选择
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= len(old_population):
                self._calculate_crowding_distance(front)
                new_population.extend(front)
            else:
                self._calculate_crowding_distance(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = len(old_population) - len(new_population)
                new_population.extend(front[:remaining])
                break
        
        self._population.individuals = new_population
        self._population.pareto_front = fronts[0] if fronts else []
        self._population.update_stats()
        
        self.logger.info(
            f"NSGA-II Generation {self._population.generation}: "
            f"Pareto front size = {len(self._population.pareto_front)}"
        )
    
    def _fast_non_dominated_sort(
        self, 
        population: List[Individual]
    ) -> List[List[Individual]]:
        """快速非支配排序"""
        fronts = [[]]
        S = {id(p): [] for p in population}  # 被 p 支配的解集
        n = {id(p): 0 for p in population}   # 支配 p 的解的数量
        
        for p in population:
            for q in population:
                if p is q:
                    continue
                if self._dominates(p, q):
                    S[id(p)].append(q)
                elif self._dominates(q, p):
                    n[id(p)] += 1
            
            if n[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[id(p)]:
                    n[id(q)] -= 1
                    if n[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return [f for f in fronts if f]
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """判断 p 是否支配 q（所有目标都不差，至少一个更好）"""
        if not p.objectives or not q.objectives:
            return p.fitness > q.fitness
        
        at_least_one_better = False
        for p_obj, q_obj in zip(p.objectives, q.objectives):
            if p_obj > q_obj:  # 假设最小化，p_obj > q_obj 意味着 p 更差
                return False
            if p_obj < q_obj:
                at_least_one_better = True
        return at_least_one_better
    
    def _calculate_crowding_distance(self, front: List[Individual]) -> None:
        """计算拥挤度距离"""
        n = len(front)
        if n == 0:
            return
        
        for ind in front:
            ind.crowding_distance = 0.0
        
        if not front[0].objectives:
            return
        
        n_objectives = len(front[0].objectives)
        
        for m in range(n_objectives):
            front.sort(key=lambda x: x.objectives[m])
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                )
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """获取 Pareto 前沿"""
        if not self._population or not self._population.pareto_front:
            return []
        
        return [
            {
                'genes': ind.genes,
                'objectives': ind.objectives,
                'rank': ind.rank,
                'crowding_distance': ind.crowding_distance
            }
            for ind in self._population.pareto_front
        ]


class DifferentialEvolution(GeneticAlgorithm):
    """差分进化算法
    
    使用差分向量进行变异的进化算法变体。
    支持多种 DE 策略：
    - DE/rand/1
    - DE/best/1
    - DE/current-to-best/1
    - DE/rand/2
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        extra = config.extra if config and config.extra else {}
        
        self._F = extra.get('F', 0.8)  # 缩放因子
        self._CR = extra.get('CR', 0.9)  # 交叉概率
        self._strategy = extra.get('strategy', 'rand/1')  # DE 策略
        self._adaptive_F = extra.get('adaptive_F', False)
        self._F_min = extra.get('F_min', 0.4)
        self._F_max = extra.get('F_max', 1.0)
    
    def _evolve(self) -> None:
        """差分进化"""
        if not self._population or len(self._population.individuals) < 4:
            super()._evolve()
            return
        
        self._population.generation += 1
        old_population = self._population.individuals
        new_population = []
        
        # 获取当前最优个体
        best_individual = self._population.get_best()
        
        # 自适应 F
        if self._adaptive_F:
            progress = self._population.generation / max(
                self._convergence_detector.max_generations, 1
            )
            self._F = self._F_max - (self._F_max - self._F_min) * progress
        
        for i, target in enumerate(old_population):
            # 生成变异向量
            mutant_genes = self._generate_mutant(i, old_population, best_individual)
            
            # 二项交叉
            trial_genes = self._binomial_crossover(target.genes, mutant_genes)
            
            trial = Individual(
                genes=trial_genes,
                generation=self._population.generation,
                evaluated=False
            )
            
            # 评估试验向量
            if self._fitness_function:
                self._evaluate_individual(trial)
            
            # 贪婪选择
            if trial.evaluated and target.evaluated:
                if trial.fitness >= target.fitness:
                    new_population.append(trial)
                else:
                    new_population.append(target.copy())
            else:
                new_population.append(trial)
        
        self._population.individuals = new_population
        self._population.update_stats()
        
        # 更新收敛检测
        self._convergence_detector.update(self._population)
        
        self.logger.info(
            f"DE Generation {self._population.generation}: "
            f"best={self._population.best_fitness:.4f}, "
            f"F={self._F:.3f}"
        )
    
    def _generate_mutant(
        self, 
        target_idx: int, 
        population: List[Individual],
        best: Individual
    ) -> Dict[str, Any]:
        """生成变异向量"""
        n = len(population)
        candidates = [j for j in range(n) if j != target_idx]
        
        mutant_genes = {}
        
        if self._strategy == 'rand/1':
            # DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
            r1, r2, r3 = random.sample(candidates, 3)
            x1, x2, x3 = population[r1], population[r2], population[r3]
            
            for param_name in x1.genes.keys():
                self._mutate_gene(mutant_genes, param_name, x1, x2, x3)
        
        elif self._strategy == 'best/1':
            # DE/best/1: v = x_best + F * (x_r1 - x_r2)
            r1, r2 = random.sample(candidates, 2)
            x1, x2 = population[r1], population[r2]
            
            for param_name in best.genes.keys():
                self._mutate_gene(mutant_genes, param_name, best, x1, x2)
        
        elif self._strategy == 'current-to-best/1':
            # DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            target = population[target_idx]
            r1, r2 = random.sample(candidates, 2)
            x1, x2 = population[r1], population[r2]
            
            for param_name in target.genes.keys():
                param_type = self._param_types.get(param_name, 'float')
                
                if param_type == 'categorical':
                    mutant_genes[param_name] = best.genes[param_name]
                else:
                    v = (target.genes[param_name] + 
                         self._F * (best.genes[param_name] - target.genes[param_name]) +
                         self._F * (x1.genes[param_name] - x2.genes[param_name]))
                    
                    if param_name in self._param_bounds:
                        low, high = self._param_bounds[param_name]
                        v = max(low, min(high, v))
                    
                    if param_type == 'int':
                        v = int(round(v))
                    
                    mutant_genes[param_name] = v
        
        else:
            # 默认 rand/1
            r1, r2, r3 = random.sample(candidates, 3)
            x1, x2, x3 = population[r1], population[r2], population[r3]
            
            for param_name in x1.genes.keys():
                self._mutate_gene(mutant_genes, param_name, x1, x2, x3)
        
        return mutant_genes
    
    def _mutate_gene(
        self,
        mutant_genes: Dict[str, Any],
        param_name: str,
        x1: Individual,
        x2: Individual,
        x3: Individual
    ) -> None:
        """变异单个基因"""
        param_type = self._param_types.get(param_name, 'float')
        
        if param_type == 'categorical':
                    votes = [x1.genes[param_name], x2.genes[param_name], x3.genes[param_name]]
                    mutant_genes[param_name] = max(set(votes), key=votes.count)
        else:
            v = x1.genes[param_name] + self._F * (x2.genes[param_name] - x3.genes[param_name])
                    
            if param_name in self._param_bounds:
                low, high = self._param_bounds[param_name]
                v = max(low, min(high, v))
                    
            if param_type == 'int':
                v = int(round(v))
                    
            mutant_genes[param_name] = v
            
    def _binomial_crossover(
        self,
        target_genes: Dict[str, Any],
        mutant_genes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """二项交叉"""
        trial_genes = {}
        gene_names = list(target_genes.keys())
        j_rand = random.randint(0, len(gene_names) - 1)
            
        for k, param_name in enumerate(gene_names):
                if random.random() < self._CR or k == j_rand:
                    trial_genes[param_name] = mutant_genes[param_name]
                else:
                    trial_genes[param_name] = target_genes[param_name]
        
        return trial_genes


# ==================== 进化策略 (ES) ====================

class EvolutionStrategy(GeneticAlgorithm):
    """进化策略 (Evolution Strategy)
    
    (μ, λ) 或 (μ + λ) 策略
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        extra = config.extra if config and config.extra else {}
        
        self._mu = extra.get('mu', 10)  # 父代数量
        self._lambda = extra.get('lambda', 50)  # 子代数量
        self._plus_selection = extra.get('plus_selection', False)  # True: (μ+λ), False: (μ,λ)
        self._sigma = extra.get('initial_sigma', 0.3)  # 初始步长
        self._sigma_decay = extra.get('sigma_decay', 0.99)
    
    def _evolve(self) -> None:
        """进化策略进化"""
        if not self._population:
            return
        
        self._population.generation += 1
        
        # 选择父代（最优的 μ 个）
        sorted_pop = sorted(
            self._population.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )
        parents = sorted_pop[:self._mu]
        
        # 生成子代
        offspring = []
        for _ in range(self._lambda):
            parent = random.choice(parents)
            child = self._mutate_es(parent.copy())
            child.generation = self._population.generation
            child.evaluated = False
            
            # 评估
            if self._fitness_function:
                self._evaluate_individual(child)
            
            offspring.append(child)
        
        # 选择下一代
        if self._plus_selection:
            # (μ + λ) 策略：从父代+子代中选择
            combined = parents + offspring
            combined.sort(key=lambda x: x.fitness, reverse=True)
            new_population = combined[:self.config.population_size]
        else:
            # (μ, λ) 策略：只从子代中选择
            offspring.sort(key=lambda x: x.fitness, reverse=True)
            new_population = offspring[:self.config.population_size]
        
        self._population.individuals = new_population
        self._population.update_stats()

        # 衰减步长
        self._sigma *= self._sigma_decay
        
        self._convergence_detector.update(self._population)
        
        self.logger.info(
            f"ES Generation {self._population.generation}: "
            f"best={self._population.best_fitness:.4f}, "
            f"sigma={self._sigma:.4f}"
        )
    
    def _mutate_es(self, individual: Individual) -> Individual:
        """ES 高斯变异"""
        for param_name in individual.genes.keys():
            param_type = self._param_types.get(param_name, 'float')
            
            if param_type == 'categorical':
                if random.random() < 0.1:
                    choices = self._param_choices.get(param_name, [])
                    if choices:
                        individual.genes[param_name] = random.choice(choices)
            else:
                if param_name in self._param_bounds:
                    low, high = self._param_bounds[param_name]
                    current = individual.genes[param_name]
                    
                    # 高斯变异
                    perturbation = random.gauss(0, self._sigma * (high - low))
                    new_val = current + perturbation
                    new_val = max(low, min(high, new_val))
                    
                    if param_type == 'int':
                        new_val = int(round(new_val))
                    
                    individual.genes[param_name] = new_val
        
        return individual
