# 智能决策算法模块 (Algo Module)

本模块提供生产级的智能化决策算法能力，包括传统优化算法和基于 LangGraph 的高级 Agent 推理系统。

## 目录

- [概述](#概述)
- [模块架构](#模块架构)
- [传统决策算法](#传统决策算法)
- [LangGraph Agent 模块](#langgraph-agent-模块)
- [智能决策服务](#智能决策服务)
- [使用示例](#使用示例)
- [API 参考](#api-参考)

---

## 概述

`backend/algo` 模块是 VectorSphere 智能平台的核心决策引擎，提供：

- **传统优化算法**: 贝叶斯优化、多臂老虎机、强化学习、遗传算法、知识推理
- **LangGraph Agent**: 多步骤推理、工具调用、循环控制、多 Agent 协作
- **统一接口**: 所有算法实现统一的接口，支持可插拔式调用

---

## 模块架构

```
backend/algo/
├── __init__.py                    # 模块导出
├── base.py                        # 算法基类和数据结构
├── bayesian_optimization.py       # 贝叶斯优化算法
├── multi_armed_bandit.py          # 多臂老虎机算法
├── reinforcement_learning.py      # 强化学习算法
├── genetic_algorithm.py           # 遗传算法
├── knowledge_reasoning.py         # 知识推理引擎
├── algorithm_factory.py           # 算法工厂
├── README.md                      # 本文档
│
└── langgraph/                     # LangGraph Agent 子模块
    ├── __init__.py                # 子模块导出
    ├── state.py                   # Agent 状态管理
    ├── tools.py                   # 工具定义和注册
    ├── nodes.py                   # 图节点定义
    ├── edges.py                   # 图边定义
    ├── graph.py                   # 状态图构建
    ├── agents.py                  # Agent 实现
    ├── checkpointer.py            # 状态检查点
    ├── factory.py                 # Agent 工厂
    └── builtin_tools.py           # 内置工具集
```

---

## 传统决策算法

### 算法类型

| 算法 | 类名 | 适用场景 |
|------|------|----------|
| 贝叶斯优化 | `BayesianOptimizationAlgorithm` | 超参数调优、黑盒优化 |
| 多臂老虎机 | `MultiArmedBanditAlgorithm` | 在线学习、A/B 测试 |
| 强化学习 | `ReinforcementLearningAlgorithm` | 序列决策、策略优化 |
| 遗传算法 | `GeneticAlgorithm` | 组合优化、架构搜索 |
| 知识推理 | `KnowledgeReasoningEngine` | 规则推理、知识图谱、语义搜索 |

### 基本使用

```python
from backend.algo import (
    AlgorithmFactory,
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext
)

# 创建算法配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
    n_initial_points=5,
    acquisition_function='expected_improvement'
)

# 使用工厂创建算法
algorithm = AlgorithmFactory.create(AlgorithmType.BAYESIAN_OPTIMIZATION, config)

# 定义搜索空间
context = AlgorithmContext(
    inputs={'task': 'hyperparameter_tuning'},
    search_space={
        'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.1},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
        'epochs': {'type': 'int', 'low': 10, 'high': 100}
    },
    objective='maximize',
    objective_metric='accuracy'
)

# 初始化并获取建议
algorithm.initialize(context)
result = algorithm.suggest(context)

print(f"推荐参数: {result.recommended_action}")
print(f"置信度: {result.confidence}")

# 更新算法（提供反馈）
algorithm.update(
    action=result.recommended_action,
    reward=0.95,  # 实际评估得分
    context=context
)
```

### 贝叶斯优化详细使用

贝叶斯优化是一种基于高斯过程的黑盒优化算法，特别适用于昂贵的函数评估场景（如超参数调优）。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **多种核函数** | RBF、Matern(1/2, 3/2, 5/2)、Rational Quadratic、周期性核、线性核、多项式核 |
| **自动超参数优化** | 使用最大似然估计（MLE）自动优化核函数参数 |
| **多种采集函数** | EI、UCB、LCB、POI、Thompson Sampling、Knowledge Gradient |
| **批量建议** | 支持并行评估多个点（Kriging Believer、Constant Liar 等策略） |
| **约束优化** | 支持等式和不等式约束 |
| **收敛检测** | 自动检测算法收敛 |
| **热启动** | 从历史数据恢复优化过程 |

#### 基础示例

```python
from backend.algo.bayesian_optimization import BayesianOptimizationAlgorithm
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType

# 1. 创建配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
    acquisition_function="expected_improvement",  # 采集函数
    n_initial_points=5,  # 初始随机采样点数
    extra={
        "kernel": "matern52",           # 核函数类型
        "length_scale": 1.0,            # 核函数长度尺度
        "variance": 1.0,                # 核函数方差
        "noise": 0.1,                   # 观测噪声
        "optimize_hyperparams": True,   # 是否自动优化超参数
        "minimize": False,              # True=最小化, False=最大化
        "n_candidates": 1000,           # 采集函数优化的候选点数
        "use_scipy_optimizer": True,    # 使用 scipy 优化采集函数
    }
)

# 2. 创建算法实例
algo = BayesianOptimizationAlgorithm(config)

# 3. 定义搜索空间
context = AlgorithmContext(
    search_space={
        "learning_rate": {"low": 0.0001, "high": 0.1},
        "batch_size": {"low": 16, "high": 256},
        "dropout": {"low": 0.0, "high": 0.5}
    },
    objective="maximize",
    objective_metric="accuracy"
)

# 4. 初始化
algo.initialize(context)

# 5. 优化循环
for iteration in range(20):
    # 获取建议
    result = algo.suggest(context)
    
    print(f"迭代 {iteration + 1}:")
    print(f"  建议参数: {result.action}")
    print(f"  置信度: {result.confidence:.4f}")
    print(f"  推理: {result.reasoning}")
    
    # 评估建议的参数（这里用模拟函数）
    params = result.action
    reward = evaluate_model(params)  # 您的评估函数
    
    print(f"  实际得分: {reward:.4f}")
    
    # 更新模型
    algo.update(result.action, reward, context)
    
    # 检查收敛
    if algo.convergence_detector.state.converged:
        print(f"算法已收敛: {algo.convergence_detector.state.reason}")
        break

# 6. 获取最优结果
summary = algo.get_optimization_summary()
print(f"\n最优参数: {summary['best_params']}")
print(f"最优值: {summary['best_value']:.4f}")
```

#### 核函数选择

```python
# RBF 核（高斯核）- 适用于平滑函数
config.extra["kernel"] = "rbf"

# Matern 5/2 核 - 比 RBF 更灵活，推荐默认选择
config.extra["kernel"] = "matern52"

# Matern 3/2 核 - 适用于较粗糙的函数
config.extra["kernel"] = "matern32"

# Matern 1/2 核（指数核）- 适用于非常粗糙的函数
config.extra["kernel"] = "matern12"

# Rational Quadratic 核 - RBF 的混合版本
config.extra["kernel"] = "rq"

# 周期性核 - 适用于周期性函数
config.extra["kernel"] = "periodic"
config.extra["period"] = 1.0  # 周期
```

#### 采集函数选择

```python
# Expected Improvement (EI) - 平衡探索与利用，推荐默认
config.acquisition_function = "expected_improvement"

# Upper Confidence Bound (UCB) - 显式控制探索
config.acquisition_function = "upper_confidence_bound"
config.extra["kappa"] = 2.576  # 探索参数

# Lower Confidence Bound (LCB) - 用于最小化问题
config.acquisition_function = "lower_confidence_bound"
config.extra["minimize"] = True

# Probability of Improvement (POI) - 保守的改进策略
config.acquisition_function = "probability_of_improvement"

# Thompson Sampling (TS) - 基于采样的探索
config.acquisition_function = "thompson_sampling"

# Knowledge Gradient (KG) - 考虑未来信息价值
config.acquisition_function = "knowledge_gradient"
```

#### 批量建议（并行评估）

```python
# 配置批量策略
config.extra["batch_strategy"] = "kriging_believer"  # 或 "constant_liar", "hallucination"

# 获取批量建议
result = algo.suggest(context, batch_size=4)  # 建议 4 个点

# 访问批量建议
batch_suggestions = result.debug_info.get('batch_suggestions', [])
print(f"批量建议: {batch_suggestions}")

# 并行评估所有建议
import concurrent.futures

def evaluate_params(params):
    return params, evaluate_model(params)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(evaluate_params, p) for p in batch_suggestions]
    for future in concurrent.futures.as_completed(futures):
        params, reward = future.result()
        algo.update(params, reward, context)
```

#### 添加约束

```python
# 添加不等式约束: learning_rate >= 0.001
algo.add_constraint(
    func=lambda x: x['learning_rate'] - 0.001,  # g(x) >= 0
    constraint_type="ineq",
    name="min_learning_rate"
)

# 添加不等式约束: batch_size <= 128
algo.add_constraint(
    func=lambda x: 128 - x['batch_size'],  # g(x) >= 0
    constraint_type="ineq",
    name="max_batch_size"
)

# 添加等式约束: learning_rate * batch_size = 1.0
algo.add_constraint(
    func=lambda x: x['learning_rate'] * x['batch_size'] - 1.0,  # h(x) = 0
    constraint_type="eq",
    name="lr_bs_product"
)
```

#### 热启动（从历史数据恢复）

```python
# 历史数据
historical_params = [
    {"learning_rate": 0.01, "batch_size": 32, "dropout": 0.1},
    {"learning_rate": 0.001, "batch_size": 64, "dropout": 0.2},
    {"learning_rate": 0.05, "batch_size": 128, "dropout": 0.3},
]
historical_rewards = [0.85, 0.88, 0.82]

# 热启动
algo.warm_start(historical_params, historical_rewards)

# 继续优化
result = algo.suggest(context)
```

#### 状态保存与恢复

```python
import json

# 保存状态
state = algo.get_state()
with open('bo_state.json', 'w') as f:
    json.dump(state, f)

# 恢复状态
with open('bo_state.json', 'r') as f:
    state = json.load(f)

new_algo = BayesianOptimizationAlgorithm(config)
new_algo.load_state(state)
new_algo.initialize(context)

# 继续优化
result = new_algo.suggest(context)
```

#### 可视化

```python
# 获取可视化数据
viz_data = algo.get_visualization_data()

# 观测历史
for obs in viz_data['observations']:
    print(f"迭代 {obs['iteration']}: 参数={obs['params']}, 值={obs['value']}")

# 最优值历史
for best in viz_data['best_history']:
    print(f"迭代 {best['iteration']}: 当前最优={best['best_value']}")

# 如果是 1D/2D 问题，可以获取预测网格
if viz_data['predictions']:
    pred = viz_data['predictions']
    if pred['type'] == '1d':
        # 绘制 1D 预测曲线
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(pred['x'], pred['mean'], 'b-', label='Mean')
        plt.fill_between(pred['x'], pred['lower'], pred['upper'], 
                        alpha=0.3, label='95% CI')
        plt.xlabel(pred['param_name'])
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.show()
```

#### 获取优化摘要

```python
summary = algo.get_optimization_summary()

print("=" * 50)
print("贝叶斯优化摘要")
print("=" * 50)
print(f"状态: {summary['status']}")
print(f"总迭代: {summary['total_iterations']}")
print(f"总评估: {summary['total_evaluations']}")
print(f"最优值: {summary['best_value']:.6f}")
print(f"最优参数: {summary['best_params']}")
print(f"最优出现在第 {summary['best_iteration']} 次迭代")
print(f"均值: {summary['mean_value']:.6f}")
print(f"标准差: {summary['std_value']:.6f}")
print(f"改进率: {summary['improvement_rate']:.2%}")
print(f"GP 对数似然: {summary['gp_log_likelihood']:.4f}")
print(f"核类型: {summary['kernel_info']['type']}")
print(f"核参数: {summary['kernel_info']['params']}")

if summary['status'] == 'converged':
    print(f"收敛原因: {summary['convergence_reason']}")
```

#### 完整优化示例

```python
from backend.algo.bayesian_optimization import BayesianOptimizationAlgorithm
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType
import numpy as np

# 目标函数（模拟模型训练）
def objective_function(params):
    """
    模拟一个复杂的目标函数
    实际使用时替换为真实的模型训练和评估
    """
    lr = params['learning_rate']
    bs = params['batch_size']
    dropout = params['dropout']
    
    # 模拟一个有噪声的目标函数
    value = (
        -10 * (lr - 0.01) ** 2 
        - 0.001 * (bs - 64) ** 2 
        - 5 * (dropout - 0.2) ** 2 
        + 0.95
    )
    noise = np.random.normal(0, 0.01)
    return value + noise

# 配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
    acquisition_function="expected_improvement",
    n_initial_points=5,
    extra={
        "kernel": "matern52",
        "optimize_hyperparams": True,
        "minimize": False,
        "max_iterations": 30,
        "no_improvement_rounds": 10,
    }
)

# 创建算法
algo = BayesianOptimizationAlgorithm(config)

# 搜索空间
context = AlgorithmContext(
    search_space={
        "learning_rate": {"low": 0.0001, "high": 0.1},
        "batch_size": {"low": 16, "high": 256},
        "dropout": {"low": 0.0, "high": 0.5}
    }
)

# 添加约束
algo.add_constraint(
    lambda x: x['learning_rate'] - 0.0005,
    constraint_type="ineq",
    name="min_lr"
)

# 初始化
algo.initialize(context)

# 优化循环
print("开始贝叶斯优化...")
print("=" * 60)

best_value = -np.inf
best_params = None

for i in range(30):
    # 获取建议
    result = algo.suggest(context)
    params = result.action
    
    # 评估
    value = objective_function(params)
    
    # 更新最优
    if value > best_value:
        best_value = value
        best_params = params.copy()
    
    # 打印进度
    print(f"[{i+1:2d}] lr={params['learning_rate']:.5f}, "
          f"bs={params['batch_size']:.1f}, "
          f"dropout={params['dropout']:.3f} -> "
          f"value={value:.4f} (best={best_value:.4f})")
    
    # 更新模型
    algo.update(params, value, context)
    
    # 检查收敛
    if algo.convergence_detector.state.converged:
        print(f"\n算法收敛: {algo.convergence_detector.state.reason}")
        break

# 打印最终结果
print("\n" + "=" * 60)
summary = algo.get_optimization_summary()
print(f"最优参数: {summary['best_params']}")
print(f"最优值: {summary['best_value']:.6f}")
print(f"总评估次数: {summary['total_evaluations']}")
print(f"改进率: {summary['improvement_rate']:.2%}")
```

### 遗传算法详细使用

遗传算法是一种模拟自然进化的优化算法，特别适用于组合优化和全局搜索问题。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **多种选择算法** | 锦标赛、轮盘赌、排名、SUS、截断、玻尔兹曼 |
| **多种交叉算法** | 单点、两点、均匀、SBX、BLX-α、算术交叉 |
| **多种变异算法** | 多项式、高斯、均匀、自适应、非均匀、边界变异 |
| **多目标优化** | NSGA-II 支持，Pareto 前沿 |
| **约束优化** | 惩罚函数、可行性规则 |
| **算法变体** | 差分进化（DE）、进化策略（ES） |
| **并行评估** | 支持多线程并行评估 |

#### 基础示例

```python
from backend.algo.genetic_algorithm import GeneticAlgorithm
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType

# 1. 创建配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.GENETIC_ALGORITHM,
    population_size=50,
    mutation_rate=0.1,
    crossover_rate=0.9,
    elite_ratio=0.1,
    extra={
        "selection_method": "tournament",   # 选择方法
        "crossover_method": "sbx",          # 交叉方法
        "mutation_method": "polynomial",    # 变异方法
        "tournament_size": 3,               # 锦标赛大小
        "crossover_eta": 20.0,              # SBX 分布指数
        "mutation_eta": 20.0,               # 多项式变异分布指数
        "adaptive_mutation": True,          # 自适应变异
        "max_generations": 100,             # 最大代数
        "no_improvement_generations": 20,   # 无改进代数阈值
    }
)

# 2. 创建算法实例
algo = GeneticAlgorithm(config)

# 3. 定义搜索空间
context = AlgorithmContext(
    search_space={
        "learning_rate": {"low": 0.0001, "high": 0.1},
        "batch_size": {"type": "int", "low": 16, "high": 256},
        "optimizer": {"choices": ["adam", "sgd", "rmsprop"]},
        "dropout": {"low": 0.0, "high": 0.5}
    }
)

# 4. 初始化
algo.initialize(context)

# 5. 设置适应度函数（可选，用于自动评估）
def fitness_function(params):
    # 实际训练和评估模型
    return train_and_evaluate(params)

algo.set_fitness_function(fitness_function)

# 6. 进化循环
for generation in range(100):
    # 获取当前最佳建议
    result = algo.suggest(context)
    
    print(f"代 {generation + 1}:")
    print(f"  最佳参数: {result.action}")
    print(f"  最佳适应度: {algo._population.best_fitness:.4f}")
    
    # 如果没有设置适应度函数，需要手动评估和更新
    if not algo._fitness_function:
        reward = evaluate_model(result.action)
        algo.update(result.action, reward, context)
    
    # 触发进化
    algo.evolve()
    
    # 检查收敛
    if algo._convergence_detector.state.converged:
        print(f"算法收敛: {algo._convergence_detector.state.reason}")
        break

# 7. 获取最终结果
summary = algo.get_optimization_summary()
print(f"\n最优参数: {summary['best_individual']}")
print(f"最优适应度: {summary['best_fitness']:.4f}")
```

#### 选择方法

```python
# 锦标赛选择（推荐，选择压力可控）
config.extra["selection_method"] = "tournament"
config.extra["tournament_size"] = 3

# 轮盘赌选择（适应度比例选择）
config.extra["selection_method"] = "roulette"

# 排名选择（降低选择压力）
config.extra["selection_method"] = "rank"

# 随机通用采样（减少随机漂移）
config.extra["selection_method"] = "sus"

# 截断选择（强选择压力）
config.extra["selection_method"] = "truncation"
config.extra["truncation_ratio"] = 0.5  # 只从最优 50% 中选择

# 玻尔兹曼选择（温度控制）
config.extra["selection_method"] = "boltzmann"
config.extra["selection_temperature"] = 1.0
```

#### 交叉方法

```python
# 模拟二进制交叉 SBX（推荐，保持解的分布）
config.extra["crossover_method"] = "sbx"
config.extra["crossover_eta"] = 20.0  # 分布指数，越大越接近父代

# BLX-α 交叉（扩展搜索范围）
config.extra["crossover_method"] = "blx_alpha"
config.extra["blx_alpha"] = 0.5

# 算术交叉
config.extra["crossover_method"] = "arithmetic"

# 均匀交叉
config.extra["crossover_method"] = "uniform"
```

#### 变异方法

```python
# 多项式变异（推荐，与 SBX 配合）
config.extra["mutation_method"] = "polynomial"
config.extra["mutation_eta"] = 20.0

# 高斯变异
config.extra["mutation_method"] = "gaussian"

# 自适应变异（随代数递减）
config.extra["mutation_method"] = "polynomial"
config.extra["adaptive_mutation"] = True

# 非均匀变异（强收敛性）
config.extra["mutation_method"] = "non_uniform"
```

#### 添加约束

```python
# 添加不等式约束
algo.add_constraint(
    func=lambda x: x['learning_rate'] - 0.001,  # g(x) >= 0
    constraint_type="ineq",
    name="min_lr"
)

# 添加等式约束
algo.add_constraint(
    func=lambda x: x['batch_size'] % 8,  # h(x) = 0
    constraint_type="eq",
    name="batch_divisible_8"
)

# 配置惩罚系数
config.extra["penalty_coefficient"] = 1e6
config.extra["adaptive_penalty"] = True  # 自适应惩罚
```

#### NSGA-II 多目标优化

```python
from backend.algo.genetic_algorithm import NSGAII

# 创建 NSGA-II 算法
algo = NSGAII(config)
algo.initialize(context)

# 添加多个目标函数（假设最小化）
algo.add_objective(lambda x: -accuracy(x))    # 最大化准确率
algo.add_objective(lambda x: model_size(x))   # 最小化模型大小
algo.add_objective(lambda x: inference_time(x))  # 最小化推理时间

# 进化
for gen in range(100):
    algo.evolve()
    
    # 获取 Pareto 前沿
    pareto_front = algo.get_pareto_front()
    print(f"代 {gen + 1}: Pareto 前沿大小 = {len(pareto_front)}")

# 打印 Pareto 最优解
for solution in pareto_front:
    print(f"参数: {solution['genes']}")
    print(f"目标值: {solution['objectives']}")
```

#### 差分进化 (DE)

```python
from backend.algo.genetic_algorithm import DifferentialEvolution

config.extra.update({
    "F": 0.8,           # 缩放因子
    "CR": 0.9,          # 交叉概率
    "strategy": "rand/1",  # DE 策略: rand/1, best/1, current-to-best/1
    "adaptive_F": True,    # 自适应 F
})

algo = DifferentialEvolution(config)
algo.initialize(context)
algo.set_fitness_function(fitness_function)

for gen in range(100):
    algo.evolve()
```

#### 进化策略 (ES)

```python
from backend.algo.genetic_algorithm import EvolutionStrategy

config.extra.update({
    "mu": 10,              # 父代数量
    "lambda": 50,          # 子代数量
    "plus_selection": False,  # False: (μ,λ), True: (μ+λ)
    "initial_sigma": 0.3,  # 初始步长
    "sigma_decay": 0.99,   # 步长衰减
})

algo = EvolutionStrategy(config)
algo.initialize(context)
algo.set_fitness_function(fitness_function)

for gen in range(100):
    algo.evolve()
```

#### 热启动

```python
# 从历史数据恢复
historical_params = [
    {"learning_rate": 0.01, "batch_size": 32},
    {"learning_rate": 0.001, "batch_size": 64},
]
historical_fitness = [0.85, 0.88]

algo.warm_start(historical_params, historical_fitness)
```

#### 可视化

```python
# 获取可视化数据
viz_data = algo.get_visualization_data()

# 进化历史
for record in viz_data['evolution_history']:
    print(f"代 {record['generation']}: "
          f"best={record['best_fitness']:.4f}, "
          f"avg={record['avg_fitness']:.4f}, "
          f"diversity={record['diversity']:.4f}")

# 适应度分布
dist = viz_data['fitness_distribution']
print(f"适应度分布: mean={dist['mean']:.4f}, std={dist['std']:.4f}")
```

#### 获取优化摘要

```python
summary = algo.get_optimization_summary()

print("=" * 50)
print("遗传算法优化摘要")
print("=" * 50)
print(f"状态: {summary['status']}")
print(f"当前代数: {summary['current_generation']}")
print(f"总评估次数: {summary['total_evaluations']}")
print(f"最优适应度: {summary['best_fitness']:.6f}")
print(f"最优参数: {summary['best_individual']}")
print(f"平均适应度: {summary['avg_fitness']:.6f}")
print(f"种群多样性: {summary['population_diversity']:.6f}")
print(f"选择方法: {summary['selection_method']}")
print(f"交叉方法: {summary['crossover_method']}")
print(f"变异方法: {summary['mutation_method']}")
```

### 知识推理引擎详细使用

知识推理引擎基于知识图谱和规则进行智能推理，适用于需要领域知识的决策场景。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **前向链推理** | 从事实出发，应用规则得出结论 |
| **后向链推理** | 从目标出发，逆向搜索支持的事实 |
| **概率推理** | 贝叶斯推理，处理不确定性 |
| **模糊推理** | 模糊规则匹配，处理模糊条件 |
| **路径搜索** | 知识图谱路径发现 |
| **规则学习** | 从历史数据学习新规则 |
| **推理解释** | 生成可解释的推理过程 |

#### 基础示例

```python
from backend.algo.knowledge_reasoning import (
    KnowledgeReasoningEngine,
    InferenceType,
    Entity,
    Relation,
    Rule,
    Condition,
    EntityType,
    RelationType,
    ConditionOperator
)
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType

# 1. 创建配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.KNOWLEDGE_REASONING,
    inference_depth=10,
    extra={
        "enable_fuzzy": True,           # 启用模糊推理
        "enable_probabilistic": True,   # 启用概率推理
        "similarity_threshold": 0.7,    # 相似度阈值
        "use_cache": True,              # 启用缓存
        "cache_size": 1000,             # 缓存大小
    }
)

# 2. 创建推理引擎
engine = KnowledgeReasoningEngine(config)

# 3. 定义上下文
context = AlgorithmContext(
    inputs={
        "data_type": "text",
        "data_size": "large",
        "task_type": "classification",
        "gpu_memory": 16
    },
    search_space={},
    constraints={"max_training_time": 24}
)

# 4. 初始化
engine.initialize(context)

# 5. 执行推理
result = engine.suggest(context, inference_type=InferenceType.HYBRID)

print(f"推荐: {result.action}")
print(f"置信度: {result.confidence:.4f}")
print(f"推理过程: {result.reasoning}")
```

#### 添加自定义知识

```python
# 添加实体
engine.knowledge_graph.add_entity(Entity(
    entity_id="model_bert",
    entity_type=EntityType.MODEL,
    name="BERT",
    properties={
        "architecture": "transformer",
        "suitable_tasks": ["nlp", "text_classification", "qa"],
        "min_data_size": "medium",
        "compute_requirement": "high"
    },
    confidence=0.95,
    aliases=["bert-base", "bert-large"],
    description="预训练的双向 Transformer 模型"
))

# 添加关系
engine.knowledge_graph.add_relation(Relation(
    source_id="model_bert",
    target_id="task_nlp",
    relation_type=RelationType.OPTIMAL_FOR,
    confidence=0.95,
    properties={"reason": "BERT 是 NLP 任务的首选模型"}
))

# 添加规则
engine.knowledge_graph.add_rule(Rule(
    rule_id="rule_bert_for_text",
    name="文本数据推荐 BERT",
    conditions=[
        Condition(
            field="data_type",
            operator=ConditionOperator.EQUALS,
            value="text"
        ),
        Condition(
            field="data_size",
            operator=ConditionOperator.IN,
            value=["medium", "large"]
        )
    ],
    conclusion={
        "recommendation": "bert",
        "learning_rate": 2e-5,
        "reason": "文本数据推荐使用 BERT 预训练模型"
    },
    confidence=0.9,
    priority=10,
    category="model_selection"
))
```

#### 推理类型

```python
# 前向链推理（从事实出发）
result = engine.suggest(context, inference_type=InferenceType.FORWARD_CHAINING)

# 后向链推理（从目标出发）
goal = {"recommendation": "transformer"}
result = engine.suggest(
    context, 
    inference_type=InferenceType.BACKWARD_CHAINING,
    goal=goal
)

# 概率推理
result = engine.suggest(context, inference_type=InferenceType.PROBABILISTIC)

# 混合推理（推荐）
result = engine.suggest(context, inference_type=InferenceType.HYBRID)
```

#### 知识图谱路径搜索

```python
# 查找两个实体之间的路径
path = engine.knowledge_graph.find_path(
    start_id="model_transformer",
    end_id="task_nlp",
    max_depth=5
)

if path:
    print(f"路径: {' -> '.join(path.path)}")
    print(f"关系: {path.relations}")
    print(f"置信度: {path.confidence:.4f}")

# 查找所有路径
all_paths = engine.knowledge_graph.find_all_paths(
    start_id="model_transformer",
    end_id="task_nlp",
    max_depth=5,
    max_paths=10
)
```

#### 语义搜索

```python
# 搜索相关实体
results = engine.semantic_search(
    query="transformer",
    entity_types=[EntityType.MODEL],
    top_k=5
)

for entity, score in results:
    print(f"{entity.name}: {score:.4f}")
```

#### 规则学习

```python
# 从历史观测学习新规则
learned_rules = engine.learn_rules_from_observations(
    min_support=0.1,      # 最小支持度
    min_confidence=0.7    # 最小置信度
)

print(f"学习到 {len(learned_rules)} 条新规则")
for rule in learned_rules:
    print(f"  - {rule.name} (置信度={rule.confidence:.2f})")
```

#### 推理解释

```python
# 执行推理
result = engine.suggest(context, inference_type=InferenceType.HYBRID)

# 获取推理解释
from backend.algo.knowledge_reasoning import InferenceResult

inference_result = engine._inference_history[-1]  # 获取最近的推理结果
explanation = engine.explain_inference(inference_result)

print("推理解释:")
print(f"  摘要: {explanation['summary']}")
print(f"  置信度说明: {explanation['confidence_explanation']}")
print(f"  支持证据: {explanation['supporting_evidence']}")
print(f"  触发规则: {explanation['rules_explanation']}")
print(f"  不确定因素: {explanation['uncertainty_factors']}")
```

#### 状态管理

```python
import json

# 保存状态
state = engine.get_state()
with open('reasoning_state.json', 'w') as f:
    json.dump(state, f, default=str)

# 恢复状态
with open('reasoning_state.json', 'r') as f:
    state = json.load(f)

new_engine = KnowledgeReasoningEngine(config)
new_engine.load_state(state)

# 清空缓存
engine.clear_cache()
```

#### 可视化数据

```python
# 获取可视化数据
viz_data = engine.get_visualization_data()

# 实体数据
print(f"实体数量: {len(viz_data['entities'])}")
for entity in viz_data['entities'][:5]:
    print(f"  - {entity['name']} ({entity['type']})")

# 关系数据
print(f"关系数量: {len(viz_data['relations'])}")
for rel in viz_data['relations'][:5]:
    print(f"  - {rel['source_name']} --[{rel['type']}]--> {rel['target_name']}")

# 规则数据
print(f"规则数量: {len(viz_data['rules'])}")
for rule in viz_data['rules'][:5]:
    print(f"  - {rule['name']} (触发次数={rule['fire_count']}, 成功率={rule['success_rate']:.2%})")

# 统计信息
stats = viz_data['statistics']
print(f"总实体: {stats['total_entities']}")
print(f"总关系: {stats['total_relations']}")
print(f"总规则: {stats['total_rules']}")
```

#### 获取优化摘要

```python
summary = engine.get_optimization_summary()

print("=" * 50)
print("知识推理引擎摘要")
print("=" * 50)
print(f"状态: {summary['status']}")
print(f"总推理次数: {summary['total_inferences']}")
print(f"总观测数: {summary['total_observations']}")
print(f"平均置信度: {summary['inference_stats']['avg_confidence']:.4f}")
print(f"平均执行时间: {summary['inference_stats']['avg_execution_time_ms']:.2f}ms")
print(f"缓存命中率: {summary['cache_stats']['hit_rate']:.2%}")

print("\n最常触发的规则:")
for rule in summary['top_rules']:
    print(f"  - {rule['name']}: {rule['fire_count']}次, 成功率={rule['success_rate']:.2%}")
```

### 多臂老虎机算法详细使用

多臂老虎机算法用于探索-利用权衡，适用于在线学习、A/B测试、推荐系统等场景。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **ε-Greedy** | 经典探索策略，支持衰减 |
| **UCB系列** | UCB1, UCB-Tuned, UCB-V, KL-UCB |
| **Thompson Sampling** | Beta分布、高斯分布 |
| **Softmax** | Boltzmann探索，支持温度衰减 |
| **Exp3** | 对抗性环境 |
| **Gradient Bandit** | 梯度策略 |
| **滑动窗口** | 非平稳环境支持 |
| **变异检测** | 自动检测奖励分布变化 |
| **臂淘汰** | 自动淘汰次优臂 |
| **批量选择** | 一次选择多个臂 |
| **上下文Bandit** | LinUCB, LinTS |

#### 基础示例

```python
from backend.algo.multi_armed_bandit import (
    MultiArmedBanditAlgorithm,
    ContextualBandit,
    HybridBandit,
    BanditStrategy
)
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType

# 1. 创建配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
    epsilon=0.1,
    ucb_c=2.0,
    extra={
        "strategy": "ucb_tuned",         # 策略选择
        "epsilon_decay": True,            # 启用ε衰减
        "epsilon_decay_rate": 0.995,
        "epsilon_min": 0.01,
        "use_sliding_window": True,       # 非平稳环境
        "sliding_window_size": 100,
        "enable_change_detection": True,  # 变异检测
        "enable_arm_elimination": True,   # 臂淘汰
        "batch_size": 1
    }
)

# 2. 创建算法
algo = MultiArmedBanditAlgorithm(config)

# 3. 定义上下文（搜索空间）
context = AlgorithmContext(
    inputs={"experiment": "model_selection"},
    search_space={
        "model": {"choices": ["resnet", "vgg", "efficientnet"]},
        "optimizer": {"choices": ["adam", "sgd", "adamw"]}
    },
    constraints={}
)

# 4. 初始化
algo.initialize(context)

# 5. 获取建议
result = algo.suggest(context)
print(f"推荐: {result.action}")
print(f"置信度: {result.confidence:.4f}")

# 6. 更新奖励
algo.update(result.action, reward=0.85)
```

#### 可用策略

```python
# ε-Greedy（支持衰减）
config = AlgorithmConfig(extra={"strategy": "epsilon_greedy_decay"})

# UCB-Tuned（方差感知）
config = AlgorithmConfig(extra={"strategy": "ucb_tuned"})

# UCB-V（方差上界）
config = AlgorithmConfig(extra={"strategy": "ucb_v"})

# KL-UCB（KL散度上界）
config = AlgorithmConfig(extra={"strategy": "kl_ucb"})

# Thompson Sampling (Beta)
config = AlgorithmConfig(extra={"strategy": "thompson_beta"})

# Thompson Sampling (Gaussian)
config = AlgorithmConfig(extra={"strategy": "thompson_gaussian"})

# Softmax（温度衰减）
config = AlgorithmConfig(extra={
    "strategy": "softmax",
    "temperature": 1.0,
    "temperature_decay": True,
    "temperature_decay_rate": 0.99
})

# Exp3（对抗性环境）
config = AlgorithmConfig(extra={"strategy": "exp3", "exp3_gamma": 0.1})

# Gradient Bandit
config = AlgorithmConfig(extra={"strategy": "gradient_bandit", "gradient_alpha": 0.1})
```

#### 上下文Bandit

```python
from backend.algo.multi_armed_bandit import ContextualBandit

# LinUCB配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
    extra={
        "contextual_strategy": "lin_ucb",  # lin_ucb, lin_ts, linear
        "lin_ucb_alpha": 1.0,
        "regularization": 1.0
    }
)

# 创建上下文Bandit
algo = ContextualBandit(config)

# 上下文包含特征
context = AlgorithmContext(
    inputs={
        "user_age": 25,
        "user_activity": 0.8,
        "time_of_day": 14
    },
    search_space={
        "recommendation": {"choices": ["item_a", "item_b", "item_c"]}
    }
)

# 获取建议（考虑上下文）
result = algo.suggest(context)
```

#### 混合Bandit（自动策略切换）

```python
from backend.algo.multi_armed_bandit import HybridBandit

config = AlgorithmConfig(
    algorithm_type=AlgorithmType.MULTI_ARMED_BANDIT,
    extra={
        "strategy_window": 50,           # 评估窗口
        "strategy_switch_threshold": 0.1  # 切换阈值
    }
)

algo = HybridBandit(config)

# 自动切换策略
result = algo.suggest(context)

# 查看策略性能
performance = algo.get_strategy_performance()
print(f"当前策略: {performance['current_strategy']}")
```

#### 热启动

```python
# 从历史数据热启动
algo.warm_start(
    arm_configs=[
        {"action": {"model": "resnet"}, "n_pulls": 50, "mean_reward": 0.85},
        {"action": {"model": "vgg"}, "n_pulls": 30, "mean_reward": 0.78},
        {"action": {"model": "efficientnet"}, "n_pulls": 20, "mean_reward": 0.82}
    ]
)
```

#### 批量选择

```python
# 一次选择多个臂
result = algo.suggest(context, batch_size=3)

# 结果包含主选择和批量选择
primary_action = result.action['primary']
batch_actions = result.action['batch']
```

#### 状态管理

```python
import json

# 保存状态
state = algo.get_state()
with open('bandit_state.json', 'w') as f:
    json.dump(state, f, default=str)

# 恢复状态
with open('bandit_state.json', 'r') as f:
    state = json.load(f)

new_algo = MultiArmedBanditAlgorithm(config)
new_algo.load_state(state)
```

#### 可视化数据

```python
viz_data = algo.get_visualization_data()

# 臂数据
print(f"臂数量: {len(viz_data['arms'])}")
for arm in viz_data['arms'][:5]:
    print(f"  - {arm['arm_id']}: 均值={arm['mean_reward']:.4f}, 次数={arm['n_pulls']}")

# 遗憾曲线
regret_curve = viz_data['regret_curve']
reward_curve = viz_data['reward_curve']

# 统计信息
stats = viz_data['statistics']
print(f"累计奖励: {stats['cumulative_reward']:.4f}")
print(f"累计遗憾: {stats['cumulative_regret']:.4f}")
```

#### 臂排名

```python
# 按平均奖励排名
rankings = algo.get_arm_rankings(metric='mean_reward')
for r in rankings[:5]:
    print(f"#{r['rank']}: {r['arm_id']} (均值={r['mean_reward']:.4f})")

# 按UCB值排名
rankings = algo.get_arm_rankings(metric='ucb')

# 按拉取次数排名
rankings = algo.get_arm_rankings(metric='n_pulls')
```

#### 获取优化摘要

```python
summary = algo.get_optimization_summary()

print("=" * 50)
print("多臂老虎机优化摘要")
print("=" * 50)
print(f"状态: {summary['status']}")
print(f"策略: {summary['strategy']}")
print(f"总臂数: {summary['total_arms']}")
print(f"活跃臂: {summary['active_arms']}")
print(f"总拉取次数: {summary['total_pulls']}")
print(f"累计奖励: {summary['cumulative_reward']:.4f}")
print(f"累计遗憾: {summary['cumulative_regret']:.4f}")
print(f"平均奖励: {summary['average_reward']:.4f}")

print("\n最佳臂:")
best = summary['best_arm']
print(f"  ID: {best['arm_id']}")
print(f"  均值: {best['mean_reward']:.4f}")
print(f"  拉取次数: {best['n_pulls']}")
print(f"  占比: {best['pull_ratio']:.2%}")
```

### 强化学习算法详细使用

强化学习算法用于序贯决策，适用于需要长期规划的任务。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **Q-Learning** | 标准Q-Learning，off-policy |
| **Double Q-Learning** | 减少过估计，两个Q表 |
| **SARSA** | on-policy TD学习 |
| **Expected SARSA** | 期望Q值更新 |
| **N-step TD** | 多步时序差分 |
| **经验回放** | 随机采样、优先采样 |
| **目标网络** | 稳定学习 |
| **探索策略** | ε-greedy、Boltzmann、UCB、Noisy |
| **奖励处理** | 裁剪、归一化、缩放 |
| **Actor-Critic** | 策略+价值函数 |

#### 基础示例

```python
from backend.algo.reinforcement_learning import (
    ReinforcementLearningAlgorithm,
    PolicyGradient,
    ActorCritic,
    RLMethod,
    ExplorationStrategy
)
from backend.algo.base import AlgorithmConfig, AlgorithmContext, AlgorithmType

# 1. 创建配置
config = AlgorithmConfig(
    algorithm_type=AlgorithmType.REINFORCEMENT_LEARNING,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon=0.1,
    extra={
        "method": "double_q_learning",    # 学习方法
        "exploration": "epsilon_decay",   # 探索策略
        "use_replay_buffer": True,        # 经验回放
        "replay_buffer_size": 10000,
        "prioritized_replay": True,       # 优先回放
        "batch_size": 32,
        "use_target_network": True,       # 目标网络
        "target_update_freq": 100,
        "reward_shaping": "normalize",    # 奖励归一化
        "normalize_states": True          # 状态归一化
    }
)

# 2. 创建算法
algo = ReinforcementLearningAlgorithm(config)

# 3. 定义上下文
context = AlgorithmContext(
    inputs={"state": "initial", "step": 0},
    search_space={
        "action": {"choices": ["up", "down", "left", "right"]}
    },
    constraints={}
)

# 4. 初始化
algo.initialize(context)

# 5. 获取建议
result = algo.suggest(context)
print(f"推荐动作: {result.action}")
print(f"置信度: {result.confidence:.4f}")

# 6. 更新（带回合结束标志）
algo.update(result.action, reward=1.0, context=new_context, done=False)
```

#### 学习方法

```python
# Q-Learning
config = AlgorithmConfig(extra={"method": "q_learning"})

# Double Q-Learning（减少过估计）
config = AlgorithmConfig(extra={"method": "double_q_learning"})

# SARSA（on-policy）
config = AlgorithmConfig(extra={"method": "sarsa"})

# Expected SARSA
config = AlgorithmConfig(extra={"method": "expected_sarsa"})

# N-step TD
config = AlgorithmConfig(extra={
    "method": "n_step_td",
    "n_steps": 3
})
```

#### 探索策略

```python
# ε-greedy（固定）
config = AlgorithmConfig(
    epsilon=0.1,
    extra={"exploration": "epsilon_greedy"}
)

# ε-greedy（衰减）
config = AlgorithmConfig(
    epsilon=0.5,
    min_epsilon=0.01,
    extra={
        "exploration": "epsilon_decay",
        "epsilon_decay_rate": 0.995
    }
)

# Boltzmann（Softmax）
config = AlgorithmConfig(extra={
    "exploration": "boltzmann",
    "temperature": 1.0,
    "temperature_decay": 0.995
})

# UCB探索
config = AlgorithmConfig(extra={
    "exploration": "ucb",
    "ucb_c": 2.0
})

# 噪声探索
config = AlgorithmConfig(extra={"exploration": "noisy"})
```

#### 经验回放

```python
config = AlgorithmConfig(extra={
    "use_replay_buffer": True,
    "replay_buffer_size": 10000,
    "batch_size": 32,
    "min_replay_size": 100,
    
    # 优先经验回放（PER）
    "prioritized_replay": True
})
```

#### 目标网络

```python
config = AlgorithmConfig(extra={
    "use_target_network": True,
    "target_update_freq": 100  # 每100步更新目标网络
})
```

#### 奖励处理

```python
config = AlgorithmConfig(extra={
    # 奖励裁剪
    "reward_shaping": "clip",
    "reward_clip_range": [-1.0, 1.0],
    
    # 或者奖励归一化
    "reward_shaping": "normalize",
    
    # 或者奖励缩放
    "reward_shaping": "scale",
    "reward_scale": 0.1
})
```

#### Actor-Critic算法

```python
from backend.algo.reinforcement_learning import ActorCritic

config = AlgorithmConfig(
    discount_factor=0.99,
    extra={
        "actor_lr": 0.01,
        "critic_lr": 0.1,
        "noise_std": 0.3,
        "noise_decay": 0.995
    }
)

algo = ActorCritic(config)

context = AlgorithmContext(
    inputs={"x": 0.5, "y": 0.5},
    search_space={
        "velocity": {"low": -1.0, "high": 1.0},
        "angle": {"low": 0.0, "high": 360.0}
    }
)

result = algo.suggest(context)
algo.update(result.action, reward=0.8, context=new_context, done=False)
```

#### Policy Gradient (REINFORCE)

```python
from backend.algo.reinforcement_learning import PolicyGradient

config = AlgorithmConfig(
    learning_rate=0.01,
    discount_factor=0.99,
    extra={
        "use_baseline": True,           # 使用基线减少方差
        "baseline_alpha": 0.1,
        "entropy_coef": 0.01,           # 熵正则化
        "initial_noise_std": 0.5,
        "noise_decay": 0.995,
        "grad_clip": 1.0,               # 梯度裁剪
        "update_batch_size": 10
    }
)

algo = PolicyGradient(config)
```

#### 热启动

```python
# 从历史数据热启动
algo.warm_start(
    observations=[
        {"state": {"x": 0}, "action": {"move": "right"}, "reward": 1.0},
        {"state": {"x": 1}, "action": {"move": "right"}, "reward": 0.5},
        {"state": {"x": 2}, "action": {"move": "left"}, "reward": -0.5}
    ],
    q_values={
        "state_abc123": {
            "move=right": 0.8,
            "move=left": 0.3
        }
    }
)
```

#### 状态管理

```python
import json

# 保存状态
state = algo.get_state()
with open('rl_state.json', 'w') as f:
    json.dump(state, f, default=str)

# 恢复状态
with open('rl_state.json', 'r') as f:
    state = json.load(f)

new_algo = ReinforcementLearningAlgorithm(config)
new_algo.load_state(state)
```

#### 可视化数据

```python
viz_data = algo.get_visualization_data()

# Q表统计
print(f"状态数: {viz_data['q_table_stats']['num_states']}")
print(f"Q值均值: {viz_data['q_table_stats']['q_value_mean']:.4f}")

# 学习曲线
for point in viz_data['learning_curve']:
    print(f"Episode {point['episode']}: {point['avg_reward']:.4f}")

# 统计信息
stats = viz_data['statistics']
print(f"总步数: {stats['total_steps']}")
print(f"回合数: {stats['episode_count']}")
```

#### 获取策略和价值函数

```python
# 获取策略（状态→动作概率）
policy = algo.get_policy()
for state_id, action_probs in list(policy.items())[:5]:
    print(f"State {state_id}: {action_probs}")

# 获取价值函数
value_function = algo.get_value_function()
for state_id, value in list(value_function.items())[:5]:
    print(f"State {state_id}: V={value:.4f}")
```

#### 获取优化摘要

```python
summary = algo.get_optimization_summary()

print("=" * 50)
print("强化学习优化摘要")
print("=" * 50)
print(f"状态: {summary['status']}")
print(f"方法: {summary['method']}")
print(f"探索策略: {summary['exploration']}")
print(f"总步数: {summary['total_steps']}")
print(f"回合数: {summary['episode_count']}")
print(f"总奖励: {summary['total_reward']:.4f}")
print(f"平均奖励: {summary['avg_reward']:.4f}")
print(f"最近平均奖励: {summary['recent_avg_reward']:.4f}")
print(f"最大奖励: {summary['max_reward']:.4f}")
print(f"平均TD误差: {summary['avg_td_error']:.4f}")

print("\n探索参数:")
params = summary['exploration_params']
print(f"  ε: {params['epsilon']:.4f}")
print(f"  温度: {params['temperature']:.4f}")

print("\nQ表统计:")
q_stats = summary['q_table_stats']
print(f"  状态数: {q_stats['num_states']}")
print(f"  Q值均值: {q_stats['q_value_mean']:.4f}")
```

### 算法工厂详细使用

算法工厂提供统一的算法创建、管理和调度接口。

#### 核心特性

| 特性 | 说明 |
|------|------|
| **算法注册** | 支持动态注册新算法和变体 |
| **场景映射** | 根据场景自动选择最佳算法 |
| **智能选择** | 上下文感知、性能驱动、自适应选择 |
| **算法集成** | 投票、加权平均、最高置信度、级联 |
| **并行执行** | 同时运行多个算法 |
| **性能监控** | 跟踪执行时间、置信度、奖励 |
| **状态管理** | 保存/恢复工厂状态 |
| **缓存管理** | 算法实例缓存和复用 |

#### 基础使用

```python
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
from backend.algo.base import AlgorithmType, AlgorithmConfig, AlgorithmContext

# 1. 直接创建算法
algo = AlgorithmFactory.create(AlgorithmType.BAYESIAN_OPTIMIZATION)
result = algo.suggest(context)

# 2. 使用变体
contextual_bandit = AlgorithmFactory.create(
    AlgorithmType.MULTI_ARMED_BANDIT,
    variant='contextual_bandit'
)

# 3. 使用便捷函数
algo = get_algorithm('bayesian_optimization', config={'n_initial_points': 10})
```

#### 场景驱动选择

```python
# 根据场景自动创建算法
algo = AlgorithmFactory.create_from_scenario(
    scenario='hyperparameter_optimization',
    context=context,
    extra_config={'n_initial_points': 5}
)

# 支持的场景
scenarios = [
    'hyperparameter_optimization',  # -> 贝叶斯优化
    'hyperparameter_init',          # -> 多臂老虎机
    'model_selection',              # -> 多臂老虎机
    'model_architecture',           # -> 知识推理
    'feature_engineering',          # -> 知识推理
    'feature_selection',            # -> 遗传算法
    'resource_allocation',          # -> 遗传算法
    'training_strategy',            # -> 强化学习
    'multi_objective',              # -> NSGA-II
    'continuous_control',           # -> Actor-Critic
    'personalized_recommendation',  # -> 上下文Bandit
    'ab_testing',                   # -> 多臂老虎机
]
```

#### 智能选择

```python
# 上下文感知选择（根据搜索空间、观测数量等自动选择）
algo = AlgorithmFactory.auto_select(
    context=context,
    strategy=SelectionStrategy.CONTEXT_AWARE
)

# 性能驱动选择（根据历史性能选择最佳算法）
algo = AlgorithmFactory.auto_select(
    context=context,
    strategy=SelectionStrategy.PERFORMANCE_BASED
)

# 自适应选择（综合上下文和历史性能）
algo = AlgorithmFactory.auto_select(
    context=context,
    strategy=SelectionStrategy.ADAPTIVE
)
```

#### 算法集成

```python
# 创建集成
ensemble = AlgorithmFactory.create_ensemble(
    algorithm_types=[
        AlgorithmType.BAYESIAN_OPTIMIZATION,
        AlgorithmType.GENETIC_ALGORITHM,
        AlgorithmType.KNOWLEDGE_REASONING
    ],
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights=[0.5, 0.3, 0.2]
)

# 使用集成
ensemble.initialize(context)
result = ensemble.suggest(context)

# 更新所有子算法
ensemble.update(result.recommended_action, reward=0.8)

# 集成方法
methods = [
    EnsembleMethod.VOTING,           # 投票法
    EnsembleMethod.WEIGHTED_AVERAGE, # 加权平均
    EnsembleMethod.BEST_CONFIDENCE,  # 最高置信度
    EnsembleMethod.CASCADE,          # 级联
]
```

#### 并行执行

```python
# 并行运行多个算法
results = AlgorithmFactory.run_parallel(
    algorithm_types=[
        AlgorithmType.BAYESIAN_OPTIMIZATION,
        AlgorithmType.GENETIC_ALGORITHM,
        AlgorithmType.MULTI_ARMED_BANDIT
    ],
    context=context,
    max_workers=3
)

# 获取各算法结果
for algo_name, result in results.items():
    if result:
        print(f"{algo_name}: 置信度={result.confidence:.4f}")
        print(f"  建议: {result.recommended_action}")
```

#### 性能监控

```python
# 获取性能统计
stats = AlgorithmFactory.get_performance_stats()
for algo_key, perf in stats.items():
    print(f"{algo_key}:")
    print(f"  总调用次数: {perf['total_calls']}")
    print(f"  平均执行时间: {perf['avg_execution_time_ms']:.2f}ms")
    print(f"  平均置信度: {perf['avg_confidence']:.4f}")
    print(f"  平均奖励: {perf['avg_reward']:.4f}")
    print(f"  最佳奖励: {perf['best_reward']:.4f}")
```

#### 状态管理

```python
# 保存工厂状态
AlgorithmFactory.save_state('factory_state.json')

# 加载工厂状态
AlgorithmFactory.load_state('factory_state.json')

# 清除缓存
AlgorithmFactory.clear_cache()

# 清除性能统计
AlgorithmFactory.clear_performance_stats()
```

#### 注册新算法

```python
from backend.algo.base import BaseAlgorithm, AlgorithmType

# 注册新算法类型
class MyCustomAlgorithm(BaseAlgorithm):
    pass

AlgorithmFactory.register_algorithm(
    algorithm_type=AlgorithmType.RULE_BASED,
    algorithm_class=MyCustomAlgorithm,
    name="自定义算法",
    description="我的自定义算法",
    supported_scenarios=['custom_scenario'],
    capabilities=['custom_capability'],
    default_config={'param1': 'value1'}
)

# 注册变体
class MyVariant(BaseAlgorithm):
    pass

AlgorithmFactory.register_variant(
    variant_name='my_variant',
    variant_class=MyVariant,
    parent_type=AlgorithmType.GENETIC_ALGORITHM,
    name="我的变体",
    description="遗传算法变体",
    supported_scenarios=['variant_scenario']
)
```

#### 便捷函数

```python
# 直接运行算法
result = run_algorithm(
    algorithm_type='bayesian_optimization',
    context={
        'inputs': {'param1': 0.5},
        'search_space': {'x': {'low': 0, 'high': 1}},
        'objective': 'maximize'
    },
    config={'n_initial_points': 5}
)

# 自动选择并运行
result = auto_run(
    context={
        'inputs': {'param1': 0.5},
        'search_space': {'x': {'low': 0, 'high': 1}}
    },
    strategy='context_aware'
)

# 创建集成
ensemble = create_ensemble(
    algorithm_types=['bayesian_optimization', 'genetic_algorithm'],
    method='weighted_average',
    weights=[0.6, 0.4]
)
```

#### 获取可用算法

```python
# 获取所有可用算法
algorithms = AlgorithmFactory.get_available_algorithms()

for algo_type, info in algorithms.items():
    print(f"\n{info['name']} ({algo_type})")
    print(f"  描述: {info['description']}")
    print(f"  类: {info['class']}")
    print(f"  场景: {info['supported_scenarios']}")
    print(f"  能力: {info['capabilities']}")
    
    if info['variants']:
        print(f"  变体:")
        for variant in info['variants']:
            print(f"    - {variant['name']}: {variant['display_name']}")
```

---

## LangGraph Agent 模块

LangGraph 模块提供基于状态图的 Agent 系统，支持复杂的多步骤推理。

### 核心概念

| 概念 | 说明 |
|------|------|
| **AgentState** | Agent 执行状态，包含消息历史、工具调用、迭代控制 |
| **Tool** | 可被 Agent 调用的工具 |
| **StateGraph** | 状态图，定义执行流程 |
| **Node** | 图节点，执行具体操作 |
| **Edge** | 图边，定义节点间的转换 |

### Agent 类型

| Agent | 说明 | 适用场景 |
|-------|------|----------|
| `ReActAgent` | 推理-行动循环 | 需要工具调用的任务 |
| `PlanAndExecuteAgent` | 先计划后执行 | 复杂多步骤任务 |
| `ReflexionAgent` | 自我反思改进 | 需要验证的决策 |
| `MultiAgentSystem` | 多 Agent 协作 | 需要专家协作的任务 |

### 工具定义

```python
from backend.algo.langgraph import tool, ToolCategory

# 使用装饰器定义工具
@tool(
    name="search_documents",
    description="搜索文档获取相关信息",
    category=ToolCategory.RETRIEVAL
)
def search_documents(query: str, top_k: int = 5) -> str:
    """
    搜索文档
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
    """
    # 实现搜索逻辑
    results = perform_search(query, top_k)
    return json.dumps(results)
```

### 创建 Agent

```python
from backend.algo.langgraph import (
    create_react_agent,
    create_plan_execute_agent,
    AgentConfig,
    ReflexionAgent
)

# 1. 创建 ReAct Agent
react_agent = create_react_agent(
    name="assistant",
    tools=[search_documents._tool],
    system_prompt="你是一个智能助手",
    max_iterations=10
)

# 执行
result = react_agent.invoke("帮我查找关于机器学习的资料")
print(result.final_answer)

# 2. 创建计划执行 Agent
plan_agent = create_plan_execute_agent(
    name="planner",
    tools=[search_documents._tool],
    max_iterations=15,
    enable_replanning=True
)

# 3. 创建反思 Agent
config = AgentConfig(name="reflexion", max_iterations=10)
reflexion_agent = ReflexionAgent(
    config=config,
    tools=[search_documents._tool],
    max_reflections=3
)
```

### 流式执行

```python
# 流式获取执行过程
for chunk in react_agent.stream("分析这个问题"):
    print(f"节点: {chunk['node']}")
    print(f"状态: {chunk['state'].status}")
    print(f"迭代: {chunk['state'].iteration}")
```

### 自定义图

```python
from backend.algo.langgraph import (
    GraphBuilder,
    LLMNode,
    ToolNode,
    route_after_agent
)

# 构建自定义工作流
builder = GraphBuilder(name="custom_workflow")

# 添加节点
builder.add_llm_node("agent", system_prompt="你是专家")
builder.add_tool_node("tools")

# 设置边
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    "__end__": "__end__"
})
builder.add_edge("tools", "agent")

# 编译
graph = builder.compile()

# 执行
result = graph.invoke("用户问题")
```

---

## 智能决策服务

`IntelligentDecisionService` 整合了传统算法和 LangGraph，提供统一的决策接口。

### 初始化

```python
from backend.services.intelligent_decision_service import (
    IntelligentDecisionService,
    DecisionContext,
    DecisionScenario,
    AgentTask,
    AgentMode
)

# 创建服务（内存模式用于测试）
service = IntelligentDecisionService(use_memory_storage=True)

# 创建服务（数据库模式用于生产）
service = IntelligentDecisionService(use_memory_storage=False)
```

### 传统决策

```python
# 定义决策上下文
context = DecisionContext(
    scenario=DecisionScenario.MODEL_ARCHITECTURE,
    inputs={
        'data_type': 'text',
        'task_type': 'classification',
        'data_size': 'medium'
    },
    constraints={'max_memory': 32},
    history=[]
)

# 执行智能决策
result = service.make_intelligent_decision(
    context=context,
    tenant_id='tenant_001',
    user_id='user_001'
)

print(f"决策ID: {result.decision_id}")
print(f"推荐动作: {result.recommended_action}")
print(f"置信度: {result.confidence}")
print(f"推理过程: {result.reasoning}")
print(f"备选方案: {result.alternatives}")
```

### Agent 高级推理

#### 方式一：使用任务定义

```python
# 定义任务
task = AgentTask(
    task_id='task_001',
    task_type='model_recommendation',
    description='为文本分类任务推荐最佳模型架构和超参数配置',
    inputs={
        'data_type': 'text',
        'data_size': 'large',
        'task': 'classification'
    },
    constraints={'max_training_time': 24},
    required_tools=['query_knowledge_base', 'recommend_hyperparameters'],
    max_iterations=10
)

# 执行 Agent 推理
result = service.execute_agent_reasoning(
    task=task,
    tenant_id='tenant_001',
    user_id='user_001',
    mode=AgentMode.MULTI_STEP
)

print(f"执行ID: {result.execution_id}")
print(f"状态: {result.status}")
print(f"迭代次数: {result.iterations}")
print(f"工具调用: {len(result.tool_calls)}")
print(f"最终答案: {result.final_answer}")
```

#### 方式二：简化接口

```python
# 自动推断任务类型
result = service.execute_complex_reasoning(
    query="为 NLP 任务推荐合适的模型和超参数",
    context={'data_type': 'text', 'data_size': 'large'},
    tenant_id='tenant_001',
    user_id='user_001',
    mode=AgentMode.MULTI_STEP
)
```

### Agent 执行模式

```python
# 1. 多步骤推理 (ReAct)
result = service.execute_agent_reasoning(
    task=task,
    mode=AgentMode.MULTI_STEP  # 思考 -> 行动 -> 观察 -> 重复
)

# 2. 计划执行模式
result = service.execute_agent_reasoning(
    task=task,
    mode=AgentMode.PLAN_EXECUTE  # 制定计划 -> 逐步执行 -> 评估
)

# 3. 反思模式
result = service.execute_agent_reasoning(
    task=task,
    mode=AgentMode.REFLEXION  # 生成 -> 评估 -> 反思 -> 改进
)
```

### 多 Agent 协作

```python
# 定义 Agent 角色
agent_roles = [
    {
        'name': '数据分析师',
        'specialty': 'data_analysis',
        'description': '负责分析数据特征和质量',
        'system_prompt': '你是数据分析专家，专注于数据质量和特征分析'
    },
    {
        'name': '模型架构师',
        'specialty': 'model_recommendation',
        'description': '负责推荐模型架构',
        'system_prompt': '你是深度学习架构专家，专注于模型设计'
    },
    {
        'name': '超参数专家',
        'specialty': 'hyperparameter_tuning',
        'description': '负责超参数优化',
        'system_prompt': '你是超参数调优专家，专注于模型性能优化'
    }
]

# 执行多 Agent 协作
result = service.create_multi_agent_workflow(
    task=task,
    agent_roles=agent_roles,
    tenant_id='tenant_001',
    user_id='user_001'
)
```

### 流式执行

```python
# 流式获取执行进度
for chunk in service.stream_agent_reasoning(
    task=task,
    tenant_id='tenant_001',
    mode=AgentMode.MULTI_STEP
):
    print(f"节点: {chunk['node']}")
    print(f"状态: {chunk['status']}")
    print(f"迭代: {chunk['iteration']}")
    print(f"消息数: {chunk['messages_count']}")
    
    if chunk['final_answer']:
        print(f"最终答案: {chunk['final_answer']}")
        break
```

### 工具管理

```python
# 获取可用工具列表
tools = service.get_available_tools()
for tool in tools:
    print(f"- {tool['name']}: {tool['description']}")

# 注册自定义工具
def my_custom_analyzer(data: str) -> str:
    """自定义分析工具"""
    return f"分析结果: {data}"

service.register_custom_tool(
    name='custom_analyzer',
    func=my_custom_analyzer,
    description='执行自定义数据分析',
    category='data'
)
```

---

## API 参考

### DecisionScenario 枚举

| 场景 | 值 | 说明 |
|------|-----|------|
| 数据预处理 | `DATA_PREPROCESSING` | 选择数据预处理策略 |
| 模型架构 | `MODEL_ARCHITECTURE` | 推荐模型架构 |
| 超参数初始化 | `HYPERPARAMETER_INIT` | 初始化超参数 |
| 资源分配 | `RESOURCE_ALLOCATION` | 分配计算资源 |
| 训练策略 | `TRAINING_STRATEGY` | 选择训练策略 |
| 模型选择 | `MODEL_SELECTION` | 选择最佳模型 |
| 特征工程 | `FEATURE_ENGINEERING` | 特征工程建议 |

### AgentMode 枚举

| 模式 | 值 | 说明 |
|------|-----|------|
| 单步推理 | `SINGLE_STEP` | 简单的单次推理 |
| 多步推理 | `MULTI_STEP` | ReAct 多步骤推理 |
| 计划执行 | `PLAN_EXECUTE` | 先计划后执行 |
| 反思模式 | `REFLEXION` | 自我反思改进 |
| 协作模式 | `COLLABORATIVE` | 多 Agent 协作 |

### 内置工具

| 工具名 | 类别 | 说明 |
|--------|------|------|
| `web_search` | SEARCH | 搜索互联网 |
| `knowledge_search` | RETRIEVAL | 知识库搜索 |
| `python_executor` | CODE | 执行 Python 代码 |
| `sql_executor` | DATA | 执行 SQL 查询 |
| `data_analyzer` | DATA | 数据统计分析 |
| `data_transformer` | DATA | 数据转换 |
| `calculator` | COMPUTATION | 数学计算 |
| `current_time` | SYSTEM | 获取当前时间 |
| `analyze_data_characteristics` | DATA | 分析数据特征 |
| `recommend_hyperparameters` | COMPUTATION | 推荐超参数 |
| `estimate_resources` | COMPUTATION | 估算资源需求 |
| `query_knowledge_base` | RETRIEVAL | 查询知识库 |
| `evaluate_decision` | COMPUTATION | 评估决策方案 |

---

## 最佳实践

### 1. 选择合适的算法

```python
# 超参数调优 -> 贝叶斯优化
scenario = DecisionScenario.HYPERPARAMETER_INIT

# 模型架构选择 -> 知识推理 + Agent
scenario = DecisionScenario.MODEL_ARCHITECTURE
mode = AgentMode.MULTI_STEP

# 复杂训练策略 -> 计划执行 Agent
scenario = DecisionScenario.TRAINING_STRATEGY
mode = AgentMode.PLAN_EXECUTE
```

### 2. 合理设置迭代次数

```python
task = AgentTask(
    # ...
    max_iterations=10  # 一般任务
)

task = AgentTask(
    # ...
    max_iterations=5   # 简单任务
)

task = AgentTask(
    # ...
    max_iterations=20  # 复杂任务
)
```

### 3. 使用流式执行监控进度

```python
for chunk in service.stream_agent_reasoning(task, mode=AgentMode.MULTI_STEP):
    # 实时显示进度
    update_progress_bar(chunk['iteration'], task.max_iterations)
    
    # 提前终止
    if should_stop(chunk):
        break
```

### 4. 提供反馈改进决策

```python
# 执行决策
result = service.make_intelligent_decision(context, tenant_id, user_id)

# 执行后提供反馈
service.provide_feedback(
    decision_id=result.decision_id,
    tenant_id=tenant_id,
    user_id=user_id,
    feedback_score=0.9,  # 0-1 评分
    feedback_comment="决策效果很好"
)

# 反馈会被记录为经验，用于改进后续决策
```

---

## 版本历史

- **v1.0.0**: 初始版本，包含传统优化算法
- **v2.0.0**: 添加 LangGraph Agent 支持
- **v2.1.0**: 集成多 Agent 协作系统
- **v2.2.0**: 贝叶斯优化升级为生产级实现
  - 新增多种核函数（Matern、RQ、周期性核等）
  - 新增核参数自动优化（MLE）
  - 新增 Thompson Sampling、Knowledge Gradient 等采集函数
  - 新增批量建议功能（支持并行评估）
  - 新增约束优化支持
  - 新增收敛检测、热启动、可视化支持
- **v2.3.0**: 遗传算法升级为生产级实现
  - 新增多种选择算法（锦标赛、轮盘赌、SUS、截断、玻尔兹曼）
  - 新增多种交叉算法（SBX、BLX-α、算术交叉等）
  - 新增多种变异算法（多项式、高斯、自适应变异）
  - 新增 NSGA-II 多目标优化支持
  - 新增差分进化（DE）多策略支持
  - 新增进化策略（ES）支持
  - 新增约束处理、收敛检测、热启动支持
- **v2.4.0**: 知识推理引擎升级为生产级实现
  - 新增后向链推理支持
  - 新增概率推理（贝叶斯推理）
  - 新增模糊推理和模糊匹配
  - 新增知识图谱路径搜索
  - 新增语义相似度计算
  - 新增规则学习（从历史数据学习新规则）
  - 新增推理结果缓存
  - 新增可解释推理过程生成
- **v2.5.0**: 多臂老虎机算法升级为生产级实现
  - 新增多种选择策略（UCB-Tuned, UCB-V, KL-UCB, Exp3, Gradient Bandit）
  - 新增 Thompson Sampling（Beta/Gaussian 分布）
  - 新增衰减探索参数（ε-decay, 温度衰减）
  - 新增滑动窗口统计（非平稳环境）
  - 新增变异检测和臂淘汰机制
  - 新增批量选择支持
  - 新增上下文 Bandit（LinUCB, LinTS）
  - 新增混合 Bandit（自动策略切换）
- **v2.6.0**: 强化学习算法升级为生产级实现
  - 新增 Double Q-Learning、Expected SARSA、N-step TD
  - 新增经验回放缓冲区（随机采样、优先采样）
  - 新增目标网络机制
  - 新增多种探索策略（ε-greedy、Boltzmann、UCB、Noisy）
  - 新增奖励/状态归一化
  - 新增 Actor-Critic 算法
  - 新增增强版 Policy Gradient（基线减法、熵正则化）
- **v2.7.0**: 算法工厂升级为生产级调度中心
  - 新增完整的算法注册和变体管理
  - 新增智能场景映射（自动选择最佳算法）
  - 新增上下文感知选择和性能驱动选择
  - 新增算法集成（投票、加权平均、级联）
  - 新增性能监控和统计
  - 新增并行执行支持
  - 新增状态管理（保存/恢复）

---

## 许可证

Copyright © 2024 VectorSphere. All rights reserved.

