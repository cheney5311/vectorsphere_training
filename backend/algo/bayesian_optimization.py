"""
贝叶斯优化算法

使用高斯过程回归和采集函数进行黑盒优化。

核心特性：
- 多种核函数：RBF、Matern(1/2, 3/2, 5/2)、Rational Quadratic、周期性核
- 核函数参数自动优化（最大似然估计）
- 多种采集函数：EI、UCB、POI、Thompson Sampling、Knowledge Gradient
- 批量建议：支持并行评估多个点
- 约束优化：支持等式和不等式约束
- 输入归一化：自动处理不同尺度参数
- 早停和收敛检测
- 热启动：从历史数据恢复
- 可视化支持

使用示例：
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.BAYESIAN_OPTIMIZATION,
        acquisition_function="expected_improvement",
        n_initial_points=5,
        extra={
            "kernel": "matern52",
            "length_scale": 1.0,
            "noise": 0.1,
            "optimize_hyperparams": True
        }
    )
    
    algo = BayesianOptimizationAlgorithm(config)
    result = algo.suggest(context)
"""

import logging
import math
import random
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
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

# 尝试导入 scipy 用于优化
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    from scipy.linalg import cholesky, cho_solve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info("scipy not available, using basic implementations")


# ==================== 核函数定义 ====================

class KernelType(Enum):
    """核函数类型"""
    RBF = "rbf"
    MATERN12 = "matern12"
    MATERN32 = "matern32"
    MATERN52 = "matern52"
    RATIONAL_QUADRATIC = "rq"
    PERIODIC = "periodic"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    COMPOSITE = "composite"


@dataclass
class KernelParams:
    """核函数参数"""
    length_scale: float = 1.0
    variance: float = 1.0
    noise: float = 0.1
    alpha: float = 1.0  # RQ 核的 alpha 参数
    period: float = 1.0  # 周期核的周期
    degree: int = 2  # 多项式核的阶数
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'length_scale': self.length_scale,
            'variance': self.variance,
            'noise': self.noise,
            'alpha': self.alpha,
            'period': self.period,
            'degree': self.degree
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KernelParams':
        return cls(
            length_scale=d.get('length_scale', 1.0),
            variance=d.get('variance', 1.0),
            noise=d.get('noise', 0.1),
            alpha=d.get('alpha', 1.0),
            period=d.get('period', 1.0),
            degree=int(d.get('degree', 2))
        )


class BaseKernel(ABC):
    """核函数基类"""
    
    def __init__(self, params: Optional[KernelParams] = None):
        self.params = params or KernelParams()
    
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        pass
    
    def _compute_squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算平方距离矩阵"""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=-1)
    
    def _compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算欧氏距离矩阵"""
        return np.sqrt(self._compute_squared_distance(X1, X2) + 1e-12)


class RBFKernel(BaseKernel):
    """RBF（高斯）核函数
    
    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = self._compute_squared_distance(X1, X2)
        return self.params.variance * np.exp(
            -0.5 * sq_dist / (self.params.length_scale ** 2)
        )


class Matern12Kernel(BaseKernel):
    """Matern 1/2 核函数（指数核）
    
    k(x, x') = σ² * exp(-||x - x'|| / l)
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = self._compute_distance(X1, X2)
        return self.params.variance * np.exp(-dist / self.params.length_scale)


class Matern32Kernel(BaseKernel):
    """Matern 3/2 核函数
    
    k(x, x') = σ² * (1 + √3 * r) * exp(-√3 * r), r = ||x - x'|| / l
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = self._compute_distance(X1, X2)
        r = dist / self.params.length_scale
        sqrt3_r = np.sqrt(3) * r
        return self.params.variance * (1 + sqrt3_r) * np.exp(-sqrt3_r)


class Matern52Kernel(BaseKernel):
    """Matern 5/2 核函数
    
    k(x, x') = σ² * (1 + √5 * r + 5/3 * r²) * exp(-√5 * r), r = ||x - x'|| / l
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = self._compute_distance(X1, X2)
        r = dist / self.params.length_scale
        sqrt5_r = np.sqrt(5) * r
        return self.params.variance * (1 + sqrt5_r + (5.0/3.0) * r**2) * np.exp(-sqrt5_r)


class RationalQuadraticKernel(BaseKernel):
    """Rational Quadratic 核函数
    
    k(x, x') = σ² * (1 + ||x - x'||² / (2 * α * l²))^(-α)
    
    当 α → ∞ 时，等价于 RBF 核
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = self._compute_squared_distance(X1, X2)
        base = 1 + sq_dist / (2 * self.params.alpha * self.params.length_scale ** 2)
        return self.params.variance * np.power(base, -self.params.alpha)


class PeriodicKernel(BaseKernel):
    """周期性核函数
    
    k(x, x') = σ² * exp(-2 * sin²(π * ||x - x'|| / p) / l²)
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = self._compute_distance(X1, X2)
        sin_term = np.sin(np.pi * dist / self.params.period)
        return self.params.variance * np.exp(
            -2 * sin_term ** 2 / (self.params.length_scale ** 2)
        )


class LinearKernel(BaseKernel):
    """线性核函数
    
    k(x, x') = σ² * (x · x')
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.params.variance * (X1 @ X2.T)


class PolynomialKernel(BaseKernel):
    """多项式核函数
    
    k(x, x') = σ² * (x · x' + 1)^d
    """
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.params.variance * np.power(X1 @ X2.T + 1, self.params.degree)


class CompositeKernel(BaseKernel):
    """复合核函数（核函数组合）
    
    支持加法和乘法组合
    """
    
    def __init__(self, kernels: List[BaseKernel], weights: List[float] = None,
                 operation: str = "add"):
        super().__init__()
        self.kernels = kernels
        self.weights = weights or [1.0] * len(kernels)
        self.operation = operation  # "add" or "multiply"
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.operation == "add":
            result = np.zeros((len(np.atleast_2d(X1)), len(np.atleast_2d(X2))))
            for kernel, weight in zip(self.kernels, self.weights):
                result += weight * kernel(X1, X2)
            return result
        else:  # multiply
            result = np.ones((len(np.atleast_2d(X1)), len(np.atleast_2d(X2))))
            for kernel, weight in zip(self.kernels, self.weights):
                result *= np.power(kernel(X1, X2), weight)
            return result


def create_kernel(kernel_type: Union[str, KernelType], 
                  params: Optional[KernelParams] = None) -> BaseKernel:
    """创建核函数
    
    Args:
        kernel_type: 核函数类型
        params: 核函数参数
        
    Returns:
        核函数实例
    """
    if isinstance(kernel_type, str):
        kernel_type = kernel_type.lower()
    else:
        kernel_type = kernel_type.value
    
    kernel_map = {
        "rbf": RBFKernel,
        "gaussian": RBFKernel,
        "matern12": Matern12Kernel,
        "matern1/2": Matern12Kernel,
        "exponential": Matern12Kernel,
        "matern32": Matern32Kernel,
        "matern3/2": Matern32Kernel,
        "matern52": Matern52Kernel,
        "matern5/2": Matern52Kernel,
        "rq": RationalQuadraticKernel,
        "rational_quadratic": RationalQuadraticKernel,
        "periodic": PeriodicKernel,
        "linear": LinearKernel,
        "polynomial": PolynomialKernel,
        "poly": PolynomialKernel,
    }
    
    kernel_class = kernel_map.get(kernel_type, RBFKernel)
    return kernel_class(params)


@dataclass
class GaussianProcessState:
    """高斯过程状态（用于序列化和恢复）"""
    X: List[List[float]]  # 观测点
    y: List[float]        # 观测值
    kernel_type: str      # 核函数类型
    kernel_params: Dict[str, float]  # 核函数参数
    y_mean: float         # 目标均值
    y_std: float          # 目标标准差
    log_marginal_likelihood: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'X': self.X,
            'y': self.y,
            'kernel_type': self.kernel_type,
            'kernel_params': self.kernel_params,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'log_marginal_likelihood': self.log_marginal_likelihood
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GaussianProcessState':
        return cls(
            X=d['X'],
            y=d['y'],
            kernel_type=d.get('kernel_type', 'rbf'),
            kernel_params=d.get('kernel_params', {}),
            y_mean=d.get('y_mean', 0.0),
            y_std=d.get('y_std', 1.0),
            log_marginal_likelihood=d.get('log_marginal_likelihood', 0.0)
        )


class GaussianProcess:
    """高斯过程回归
    
    特性：
    - 支持多种核函数
    - 自动超参数优化
    - 数值稳定的 Cholesky 分解
    - 增量更新支持
    - 状态序列化/恢复
    
    Args:
        kernel: 核函数类型或实例
        kernel_params: 核函数参数
        normalize_y: 是否标准化目标值
        optimize_hyperparams: 是否自动优化超参数
        n_restarts: 超参数优化重启次数
    """
    
    def __init__(
        self,
        kernel: Union[str, BaseKernel] = "rbf",
        kernel_params: Optional[KernelParams] = None,
        normalize_y: bool = True,
        optimize_hyperparams: bool = True,
        n_restarts: int = 5,
        alpha: float = 1e-10  # 数值稳定性参数
    ):
        # 初始化核函数
        if isinstance(kernel, str):
            self.kernel_type = kernel
            self.kernel_params = kernel_params or KernelParams()
            self.kernel = create_kernel(kernel, self.kernel_params)
        else:
            self.kernel_type = "custom"
            self.kernel = kernel
            self.kernel_params = kernel.params if hasattr(kernel, 'params') else KernelParams()
        
        self.normalize_y = normalize_y
        self.optimize_hyperparams = optimize_hyperparams
        self.n_restarts = n_restarts
        self.alpha = alpha
        
        # 训练数据
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_train_mean: float = 0.0
        self.y_train_std: float = 1.0
        
        # Cholesky 分解结果
        self.L_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        
        # 边际似然
        self.log_marginal_likelihood_value_: float = -np.inf
        
    def fit(self, X: Union[List[List[float]], np.ndarray], 
            y: Union[List[float], np.ndarray]) -> 'GaussianProcess':
        """拟合高斯过程
        
        Args:
            X: 训练样本 (n_samples, n_features)
            y: 目标值 (n_samples,)
            
        Returns:
            self
        """
        self.X_train = np.atleast_2d(np.array(X))
        self.y_train = np.array(y).ravel()
        
        n_samples = len(self.y_train)
        
        # 标准化目标值
        if self.normalize_y and n_samples > 1:
            self.y_train_mean = np.mean(self.y_train)
            self.y_train_std = np.std(self.y_train)
            if self.y_train_std < 1e-10:
                self.y_train_std = 1.0
            y_normalized = (self.y_train - self.y_train_mean) / self.y_train_std
        else:
            self.y_train_mean = 0.0
            self.y_train_std = 1.0
            y_normalized = self.y_train
        
        # 自动优化超参数
        if self.optimize_hyperparams and SCIPY_AVAILABLE and n_samples >= 3:
            self._optimize_hyperparameters(self.X_train, y_normalized)
        
        # 计算核矩阵并进行 Cholesky 分解
        self._compute_cholesky(self.X_train, y_normalized)
        
        return self
    
    def _compute_cholesky(self, X: np.ndarray, y: np.ndarray) -> None:
        """计算 Cholesky 分解
        
        使用数值稳定的方法计算。
        """
        n = len(X)
        K = self.kernel(X, X)
        
        # 添加噪声项和正则化
        noise_var = self.kernel_params.noise ** 2
        K_reg = K + (noise_var + self.alpha) * np.eye(n)
        
        try:
            if SCIPY_AVAILABLE:
                self.L_ = cholesky(K_reg, lower=True)
                self.alpha_ = cho_solve((self.L_, True), y)
            else:
                self.L_ = np.linalg.cholesky(K_reg)
                self.alpha_ = np.linalg.solve(
                    self.L_.T, np.linalg.solve(self.L_, y)
                )
            
            # 计算边际似然
            self.log_marginal_likelihood_value_ = self._compute_log_marginal_likelihood(
                y, K_reg, self.L_
            )
            
        except np.linalg.LinAlgError:
            # 如果 Cholesky 分解失败，添加更多正则化
            logger.warning("Cholesky decomposition failed, adding more regularization")
            for jitter in [1e-6, 1e-4, 1e-2, 1e-1]:
                try:
                    K_reg = K + (noise_var + jitter) * np.eye(n)
                    if SCIPY_AVAILABLE:
                        self.L_ = cholesky(K_reg, lower=True)
                        self.alpha_ = cho_solve((self.L_, True), y)
                    else:
                        self.L_ = np.linalg.cholesky(K_reg)
                        self.alpha_ = np.linalg.solve(
                            self.L_.T, np.linalg.solve(self.L_, y)
                        )
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                # 回退到伪逆
                logger.warning("Using pseudo-inverse as fallback")
                K_inv = np.linalg.pinv(K_reg)
                self.L_ = None
                self.alpha_ = K_inv @ y
    
    def _compute_log_marginal_likelihood(self, y: np.ndarray, K: np.ndarray,
                                         L: Optional[np.ndarray] = None) -> float:
        """计算对数边际似然
        
        log p(y|X, θ) = -1/2 * y^T * K^{-1} * y - 1/2 * log|K| - n/2 * log(2π)
        """
        n = len(y)
        
        if L is not None:
            # 使用 Cholesky 分解计算
            alpha = cho_solve((L, True), y) if SCIPY_AVAILABLE else \
                    np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
        else:
            # 直接计算
            sign, log_det = np.linalg.slogdet(K)
            alpha = np.linalg.solve(K, y)
        
        log_likelihood = -0.5 * (
            y @ alpha + log_det + n * np.log(2 * np.pi)
        )
        
        return log_likelihood
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """优化核函数超参数
        
        使用最大似然估计（MLE）优化。
        """
        def negative_log_marginal_likelihood(theta):
            """负对数边际似然（用于最小化）"""
            # 解包参数
            length_scale = np.exp(theta[0])
            variance = np.exp(theta[1])
            noise = np.exp(theta[2])
            
            # 更新核参数
            self.kernel_params.length_scale = length_scale
            self.kernel_params.variance = variance
            self.kernel_params.noise = noise
            self.kernel = create_kernel(self.kernel_type, self.kernel_params)
            
            # 计算核矩阵
            n = len(X)
            K = self.kernel(X, X) + (noise ** 2 + self.alpha) * np.eye(n)
            
            try:
                if SCIPY_AVAILABLE:
                    L = cholesky(K, lower=True)
                    alpha = cho_solve((L, True), y)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                else:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                
                nll = 0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))
                return nll
                
            except np.linalg.LinAlgError:
                return 1e10
        
        # 初始参数（对数空间）
        theta0 = np.array([
            np.log(self.kernel_params.length_scale),
            np.log(self.kernel_params.variance),
            np.log(self.kernel_params.noise)
        ])
        
        # 参数边界
        bounds = [
            (-5, 5),   # log(length_scale): [0.007, 148]
            (-5, 5),   # log(variance): [0.007, 148]
            (-10, 2)   # log(noise): [0.00005, 7.4]
        ]
        
        best_nll = np.inf
        best_theta = theta0
        
        # 多次重启优化
        for i in range(self.n_restarts + 1):
            if i == 0:
                theta_init = theta0
            else:
                theta_init = np.array([
                    np.random.uniform(b[0], b[1]) for b in bounds
                ])
            
            try:
                result = minimize(
                    negative_log_marginal_likelihood,
                    theta_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_theta = result.x
                    
            except Exception as e:
                logger.debug(f"Optimization restart {i} failed: {e}")
                continue
        
        # 应用最优参数
        self.kernel_params.length_scale = np.exp(best_theta[0])
        self.kernel_params.variance = np.exp(best_theta[1])
        self.kernel_params.noise = np.exp(best_theta[2])
        self.kernel = create_kernel(self.kernel_type, self.kernel_params)
        
        logger.debug(f"Optimized hyperparameters: length_scale={self.kernel_params.length_scale:.4f}, "
                    f"variance={self.kernel_params.variance:.4f}, noise={self.kernel_params.noise:.4f}")
    
    def predict(self, X: Union[List[List[float]], np.ndarray], 
                return_std: bool = True,
                return_cov: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """预测
        
        Args:
            X: 预测点
            return_std: 是否返回标准差
            return_cov: 是否返回完整协方差矩阵
            
        Returns:
            预测均值，以及可选的标准差或协方差
        """
        X = np.atleast_2d(np.array(X))
        
        if self.X_train is None or self.alpha_ is None:
            n = len(X)
            mean = np.zeros(n) + self.y_train_mean
            if return_cov:
                return mean, np.eye(n) * self.y_train_std ** 2
            elif return_std:
                return mean, np.ones(n) * self.y_train_std
            return mean
        
        # 计算预测核矩阵
        K_star = self.kernel(X, self.X_train)
        
        # 预测均值
        mean = K_star @ self.alpha_
        
        # 反标准化
        mean = mean * self.y_train_std + self.y_train_mean
        
        if not return_std and not return_cov:
            return mean
        
        # 计算预测方差/协方差
        K_star_star = self.kernel(X, X)
        
        if self.L_ is not None:
            if SCIPY_AVAILABLE:
                V = cho_solve((self.L_, True), K_star.T)
            else:
                V = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, K_star.T))
        else:
            # 使用伪逆的情况
            K = self.kernel(self.X_train, self.X_train)
            K += (self.kernel_params.noise ** 2 + self.alpha) * np.eye(len(self.X_train))
            V = np.linalg.solve(K, K_star.T)
        
        cov = K_star_star - K_star @ V
        
        # 确保数值稳定性
        cov = np.maximum(cov, 0)
        
        # 反标准化方差
        cov = cov * (self.y_train_std ** 2)
        
        if return_cov:
            return mean, cov
        else:
            std = np.sqrt(np.diag(cov))
        return mean, std
    
    def sample(self, X: Union[List[List[float]], np.ndarray], 
               n_samples: int = 1,
               random_state: Optional[int] = None) -> np.ndarray:
        """从后验分布采样
        
        Args:
            X: 采样点
            n_samples: 采样数量
            random_state: 随机种子
            
        Returns:
            采样结果 (n_samples, n_points)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        mean, cov = self.predict(X, return_cov=True)
        
        # 确保协方差矩阵对称且正定
        cov = (cov + cov.T) / 2
        cov += 1e-6 * np.eye(len(cov))
        
        try:
            samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵不是正定的，使用对角近似
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            samples = mean + std * np.random.randn(n_samples, len(mean))
        
        return samples
    
    def get_state(self) -> GaussianProcessState:
        """获取高斯过程状态"""
        return GaussianProcessState(
            X=self.X_train.tolist() if self.X_train is not None else [],
            y=self.y_train.tolist() if self.y_train is not None else [],
            kernel_type=self.kernel_type,
            kernel_params=self.kernel_params.to_dict(),
            y_mean=self.y_train_mean,
            y_std=self.y_train_std,
            log_marginal_likelihood=self.log_marginal_likelihood_value_
        )
    
    def load_state(self, state: GaussianProcessState) -> None:
        """从状态恢复"""
        if state.X and state.y:
            self.kernel_type = state.kernel_type
            self.kernel_params = KernelParams.from_dict(state.kernel_params)
            self.kernel = create_kernel(self.kernel_type, self.kernel_params)
            self.y_train_mean = state.y_mean
            self.y_train_std = state.y_std
            
            # 重新拟合（不优化超参数）
            old_optimize = self.optimize_hyperparams
            self.optimize_hyperparams = False
            self.fit(state.X, state.y)
            self.optimize_hyperparams = old_optimize


# ==================== 采集函数定义 ====================

class AcquisitionType(Enum):
    """采集函数类型"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"
    LOWER_CONFIDENCE_BOUND = "lower_confidence_bound"


class AcquisitionFunction:
    """采集函数 - 生产级实现
    
    支持多种采集函数：
    - expected_improvement (EI): 期望改进
    - upper_confidence_bound (UCB): 置信上界
    - probability_of_improvement (POI): 改进概率
    - thompson_sampling (TS): 汤普森采样
    - knowledge_gradient (KG): 知识梯度（近似）
    """
    
    def __init__(
        self, 
        kind: str = "expected_improvement", 
        kappa: float = 2.576, 
        xi: float = 0.01,
        minimize: bool = False,
        n_samples: int = 500
    ):
        self.kind = kind.lower() if kind else "expected_improvement"
        self.kappa = kappa
        self.xi = xi
        self.minimize = minimize
        self.n_samples = n_samples
        self._iteration = 0
        self._kappa_decay = 0.99
        self._min_kappa = 0.1
        
    def set_iteration(self, iteration: int) -> None:
        """设置当前迭代次数"""
        self._iteration = iteration
        
    def get_effective_kappa(self) -> float:
        """获取衰减后的 kappa 值"""
        return max(
            self.kappa * (self._kappa_decay ** self._iteration),
            self._min_kappa
        )
    
    def evaluate(
        self, 
        gp: GaussianProcess, 
        X: Union[List[List[float]], np.ndarray], 
        y_best: float
    ) -> np.ndarray:
        """计算采集函数值"""
        X = np.atleast_2d(np.array(X))
        mean, std = gp.predict(X)
        
        if self.minimize:
            mean = -mean
            y_best = -y_best
        
        if self.kind in ["expected_improvement", "ei"]:
            return self._expected_improvement(mean, std, y_best)
        elif self.kind in ["upper_confidence_bound", "ucb"]:
            return self._upper_confidence_bound(mean, std)
        elif self.kind in ["lower_confidence_bound", "lcb"]:
            return self._lower_confidence_bound(mean, std)
        elif self.kind in ["probability_of_improvement", "poi", "pi"]:
            return self._probability_of_improvement(mean, std, y_best)
        elif self.kind in ["thompson_sampling", "ts"]:
            return self._thompson_sampling(gp, X)
        elif self.kind in ["knowledge_gradient", "kg"]:
            return self._knowledge_gradient(gp, X, y_best)
        else:
            return self._expected_improvement(mean, std, y_best)
    
    def _expected_improvement(
        self, mean: np.ndarray, std: np.ndarray, y_best: float
    ) -> np.ndarray:
        """期望改进"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            std_safe = np.maximum(std, 1e-10)
            improvement = mean - y_best - self.xi
            Z = improvement / std_safe
            
            if SCIPY_AVAILABLE:
                ei = improvement * norm.cdf(Z) + std_safe * norm.pdf(Z)
            else:
                ei = improvement * self._norm_cdf(Z) + std_safe * self._norm_pdf(Z)
            
            ei[std < 1e-10] = 0.0
            return ei
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """置信上界"""
        kappa = self.get_effective_kappa()
        return mean + kappa * std
    
    def _lower_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """置信下界（用于最小化）"""
        kappa = self.get_effective_kappa()
        return -(mean - kappa * std)
    
    def _probability_of_improvement(
        self, mean: np.ndarray, std: np.ndarray, y_best: float
    ) -> np.ndarray:
        """改进概率"""
        std_safe = np.maximum(std, 1e-10)
        Z = (mean - y_best - self.xi) / std_safe
        
        if SCIPY_AVAILABLE:
            poi = norm.cdf(Z)
        else:
            poi = self._norm_cdf(Z)
        
        poi[std < 1e-10] = 0.0
        return poi
    
    def _thompson_sampling(self, gp: GaussianProcess, X: np.ndarray) -> np.ndarray:
        """汤普森采样"""
        try:
            samples = gp.sample(X, n_samples=self.n_samples)
            return np.mean(samples, axis=0)
        except Exception:
            mean, std = gp.predict(X)
            return self._upper_confidence_bound(mean, std)
    
    def _knowledge_gradient(
        self, gp: GaussianProcess, X: np.ndarray, y_best: float
    ) -> np.ndarray:
        """知识梯度近似"""
        mean, std = gp.predict(X)
        n = len(X)
        kg = np.zeros(n)
        
        for i in range(n):
            z_values = np.linspace(-3, 3, 20)
            expected_improvement = 0
            
            for z in z_values:
                y_new = mean[i] + z * std[i]
                potential_best = max(y_best, y_new)
                improvement = potential_best - y_best
                if SCIPY_AVAILABLE:
                    weight = norm.pdf(z)
                else:
                    weight = self._norm_pdf(np.array([z]))[0]
                expected_improvement += improvement * weight
            
            kg[i] = expected_improvement / len(z_values)
        
        return kg
    
    # 静态方法保持向后兼容
    @staticmethod
    def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                           y_best: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement (EI) - 静态方法"""
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mean - y_best - xi) / np.maximum(std, 1e-10)
            ei = (mean - y_best - xi) * AcquisitionFunction._norm_cdf(z) + \
                 std * AcquisitionFunction._norm_pdf(z)
            ei[std < 1e-10] = 0
        return ei
    
    @staticmethod
    def upper_confidence_bound(mean: np.ndarray, std: np.ndarray,
                              kappa: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound (UCB) - 静态方法"""
        return mean + kappa * std
    
    @staticmethod
    def probability_of_improvement(mean: np.ndarray, std: np.ndarray,
                                  y_best: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement (POI) - 静态方法"""
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mean - y_best - xi) / np.maximum(std, 1e-10)
            poi = AcquisitionFunction._norm_cdf(z)
            poi[std < 1e-10] = 0
        return poi
    
    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """标准正态分布CDF"""
        return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """标准正态分布PDF"""
        return np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


# ==================== 批量建议 ====================

class BatchSuggestionStrategy(Enum):
    """批量建议策略"""
    KRIGING_BELIEVER = "kriging_believer"
    CONSTANT_LIAR = "constant_liar"
    LOCAL_PENALIZATION = "local_penalization"
    HALLUCINATION = "hallucination"


class BatchSuggester:
    """批量建议器 - 支持并行评估
    
    Args:
        strategy: 批量策略
        constant_liar_value: Constant Liar 使用的值类型
    """
    
    def __init__(
        self, 
        strategy: Union[str, BatchSuggestionStrategy] = "kriging_believer",
        constant_liar_value: str = "mean"
    ):
        if isinstance(strategy, str):
            strategy = strategy.lower()
            try:
                self.strategy = BatchSuggestionStrategy(strategy)
            except ValueError:
                self.strategy = BatchSuggestionStrategy.KRIGING_BELIEVER
        else:
            self.strategy = strategy
        self.constant_liar_value = constant_liar_value
        
    def suggest_batch(
        self,
        gp: GaussianProcess,
        acq_func: AcquisitionFunction,
        bounds: List[Tuple[float, float]],
        y_best: float,
        batch_size: int,
        n_candidates: int = 1000
    ) -> np.ndarray:
        """建议一批点
        
        Args:
            gp: 高斯过程
            acq_func: 采集函数
            bounds: 参数边界
            y_best: 当前最优值
            batch_size: 批量大小
            n_candidates: 候选点数量
            
        Returns:
            建议的点 (batch_size, n_dims)
        """
        n_dims = len(bounds)
        suggested = []
        
        X_virtual = list(gp.X_train) if gp.X_train is not None else []
        y_virtual = list(gp.y_train) if gp.y_train is not None else []
        
        for i in range(batch_size):
            if i > 0 and X_virtual:
                gp_temp = GaussianProcess(
                    kernel=gp.kernel_type,
                    kernel_params=gp.kernel_params,
                    normalize_y=gp.normalize_y,
                    optimize_hyperparams=False
                )
                gp_temp.fit(X_virtual, y_virtual)
            else:
                gp_temp = gp
            
            candidates = self._generate_candidates(bounds, n_candidates)
            acq_values = acq_func.evaluate(gp_temp, candidates, y_best)
            
            best_idx = np.argmax(acq_values)
            best_point = candidates[best_idx]
            suggested.append(best_point)
            
            X_virtual.append(best_point.tolist())
            y_virtual.append(self._get_virtual_observation(gp_temp, best_point, y_virtual))
        
        return np.array(suggested)
    
    def _generate_candidates(
        self, bounds: List[Tuple[float, float]], n_candidates: int
    ) -> np.ndarray:
        """生成候选点（拉丁超立方 + 随机）"""
        n_dims = len(bounds)
        n_lhs = n_candidates // 2
        
        # Latin Hypercube Sampling
        samples_lhs = np.zeros((n_lhs, n_dims))
        for dim, (low, high) in enumerate(bounds):
            perms = np.random.permutation(n_lhs)
            samples_lhs[:, dim] = low + (high - low) * (perms + np.random.rand(n_lhs)) / n_lhs
        
        # 随机采样
        n_random = n_candidates - n_lhs
        samples_random = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(n_random)
        ])
        
        return np.vstack([samples_lhs, samples_random])
    
    def _get_virtual_observation(
        self, gp: GaussianProcess, point: np.ndarray, y_virtual: List[float]
    ) -> float:
        """获取虚拟观测值"""
        if self.strategy == BatchSuggestionStrategy.KRIGING_BELIEVER:
            mean, _ = gp.predict([point])
            return mean[0]
        
        elif self.strategy == BatchSuggestionStrategy.CONSTANT_LIAR:
            if not y_virtual:
                return 0.0
            if self.constant_liar_value == "max":
                return np.max(y_virtual)
            elif self.constant_liar_value == "min":
                return np.min(y_virtual)
            return np.mean(y_virtual)
        
        elif self.strategy == BatchSuggestionStrategy.HALLUCINATION:
            try:
                samples = gp.sample([point], n_samples=1)
                return samples[0, 0]
            except:
                mean, _ = gp.predict([point])
                return mean[0]
        
        else:
            mean, _ = gp.predict([point])
            return mean[0]


# ==================== 约束处理 ====================

@dataclass
class Constraint:
    """优化约束
    
    Args:
        func: 约束函数，接收参数字典，返回浮点数
        constraint_type: 约束类型 ("eq" 等式, "ineq" 不等式 g(x) >= 0)
        name: 约束名称
    """
    func: Callable[[Dict[str, Any]], float]
    constraint_type: str = "ineq"
    name: str = ""
    
    def evaluate(self, x: Dict[str, Any]) -> float:
        """评估约束"""
        return self.func(x)
    
    def is_satisfied(self, x: Dict[str, Any], tolerance: float = 1e-6) -> bool:
        """检查约束是否满足"""
        value = self.evaluate(x)
        if self.constraint_type == "eq":
            return abs(value) <= tolerance
        else:  # ineq: g(x) >= 0
            return value >= -tolerance


class ConstraintHandler:
    """约束处理器
    
    使用惩罚函数法和可行性概率处理约束
    """
    
    def __init__(
        self,
        constraints: List[Constraint] = None,
        penalty_weight: float = 1e6,
        use_probability: bool = True
    ):
        self.constraints = constraints or []
        self.penalty_weight = penalty_weight
        self.use_probability = use_probability
        self._constraint_gps: Dict[str, GaussianProcess] = {}
        
    def add_constraint(self, constraint: Constraint) -> None:
        """添加约束"""
        self.constraints.append(constraint)
        
    def check_feasibility(self, x: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查点的可行性
        
        Returns:
            (是否可行, 违反的约束列表)
        """
        violated = []
        for c in self.constraints:
            if not c.is_satisfied(x):
                violated.append(c.name or f"constraint_{self.constraints.index(c)}")
        return len(violated) == 0, violated
    
    def compute_penalty(self, x: Dict[str, Any]) -> float:
        """计算惩罚值"""
        penalty = 0.0
        for c in self.constraints:
            value = c.evaluate(x)
            if c.constraint_type == "eq":
                penalty += self.penalty_weight * value ** 2
            else:  # ineq
                if value < 0:
                    penalty += self.penalty_weight * value ** 2
        return penalty
    
    def compute_feasibility_probability(
        self, 
        x: np.ndarray,
        constraint_gps: Dict[str, GaussianProcess] = None
    ) -> float:
        """计算可行性概率（使用约束的GP模型）"""
        if not constraint_gps:
            return 1.0
        
        prob = 1.0
        for name, gp in constraint_gps.items():
            mean, std = gp.predict([x])
            # P(g(x) >= 0) = P(Z >= -mean/std) = 1 - Phi(-mean/std)
            if std[0] > 1e-10:
                z = mean[0] / std[0]
                if SCIPY_AVAILABLE:
                    prob *= norm.cdf(z)
                else:
                    prob *= 0.5 * (1 + math.erf(z / math.sqrt(2)))
            else:
                prob *= 1.0 if mean[0] >= 0 else 0.0
        
        return prob
    
    def filter_candidates(
        self, 
        candidates: List[Dict[str, Any]],
        return_all_if_none_feasible: bool = True
    ) -> List[Dict[str, Any]]:
        """过滤不可行的候选点"""
        feasible = []
        for x in candidates:
            is_feasible, _ = self.check_feasibility(x)
            if is_feasible:
                feasible.append(x)
        
        if not feasible and return_all_if_none_feasible:
            return candidates
        return feasible


# ==================== 输入预处理 ====================

class InputNormalizer:
    """输入归一化器
    
    将参数空间归一化到 [0, 1]^d
    """
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)
        
    def normalize(self, x: Dict[str, Any]) -> np.ndarray:
        """归一化"""
        result = np.zeros(self.n_dims)
        for i, name in enumerate(self.param_names):
            low, high = self.bounds[name]
            value = x.get(name, low)
            result[i] = (value - low) / (high - low + 1e-10)
        return result
    
    def normalize_array(self, X: np.ndarray) -> np.ndarray:
        """归一化数组"""
        X = np.atleast_2d(X)
        result = np.zeros_like(X)
        for i, name in enumerate(self.param_names):
            low, high = self.bounds[name]
            result[:, i] = (X[:, i] - low) / (high - low + 1e-10)
        return result
    
    def denormalize(self, x_normalized: np.ndarray) -> Dict[str, float]:
        """反归一化"""
        result = {}
        for i, name in enumerate(self.param_names):
            low, high = self.bounds[name]
            result[name] = x_normalized[i] * (high - low) + low
        return result
    
    def denormalize_array(self, X_normalized: np.ndarray) -> np.ndarray:
        """反归一化数组"""
        X_normalized = np.atleast_2d(X_normalized)
        result = np.zeros_like(X_normalized)
        for i, name in enumerate(self.param_names):
            low, high = self.bounds[name]
            result[:, i] = X_normalized[:, i] * (high - low) + low
        return result
    
    def get_normalized_bounds(self) -> List[Tuple[float, float]]:
        """获取归一化后的边界"""
        return [(0.0, 1.0) for _ in self.param_names]


# ==================== 收敛检测 ====================

@dataclass
class ConvergenceState:
    """收敛状态"""
    converged: bool = False
    reason: str = ""
    iterations_since_improvement: int = 0
    best_value: float = -np.inf
    best_value_history: List[float] = field(default_factory=list)
    improvement_threshold: float = 1e-4


class ConvergenceDetector:
    """收敛检测器
    
    支持多种收敛准则：
    - 最优值无改进
    - 方差收敛
    - 最大迭代次数
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        no_improvement_rounds: int = 10,
        improvement_threshold: float = 1e-4,
        variance_threshold: float = 1e-6
    ):
        self.max_iterations = max_iterations
        self.no_improvement_rounds = no_improvement_rounds
        self.improvement_threshold = improvement_threshold
        self.variance_threshold = variance_threshold
        self.state = ConvergenceState(improvement_threshold=improvement_threshold)
        
    def update(self, new_value: float, iteration: int) -> ConvergenceState:
        """更新收敛状态"""
        self.state.best_value_history.append(new_value)
        
        # 检查最大迭代
        if iteration >= self.max_iterations:
            self.state.converged = True
            self.state.reason = f"Reached max iterations ({self.max_iterations})"
            return self.state
        
        # 检查是否有改进
        improvement = new_value - self.state.best_value
        if improvement > self.improvement_threshold:
            self.state.best_value = new_value
            self.state.iterations_since_improvement = 0
        else:
            self.state.iterations_since_improvement += 1
        
        # 检查无改进轮数
        if self.state.iterations_since_improvement >= self.no_improvement_rounds:
            self.state.converged = True
            self.state.reason = f"No improvement for {self.no_improvement_rounds} rounds"
            return self.state
        
        # 检查方差收敛（最近几轮的最优值方差）
        if len(self.state.best_value_history) >= 5:
            recent_values = self.state.best_value_history[-5:]
            variance = np.var(recent_values)
            if variance < self.variance_threshold:
                self.state.converged = True
                self.state.reason = f"Values converged (variance={variance:.2e})"
                return self.state
        
        return self.state
    
    def reset(self) -> None:
        """重置状态"""
        self.state = ConvergenceState(improvement_threshold=self.improvement_threshold)


class BayesianOptimizationAlgorithm(BaseAlgorithm):
    """贝叶斯优化算法 - 生产级实现
    
    特性：
    - 多种核函数和采集函数支持
    - 批量建议（并行评估）
    - 约束优化
    - 输入归一化
    - 收敛检测
    - 热启动（从历史恢复）
    - 可视化支持
    
    使用高斯过程作为代理模型，通过采集函数选择下一个评估点。
    适用于昂贵的黑盒函数优化。
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        
        # 从配置中获取参数
        extra = config.extra if config and config.extra else {}
        
        # 初始化高斯过程
        kernel_type = extra.get('kernel', 'matern52')
        kernel_params = KernelParams(
            length_scale=extra.get('length_scale', 1.0),
            variance=extra.get('variance', 1.0),
            noise=extra.get('noise', 0.1)
        )
        
        self.gp = GaussianProcess(
            kernel=kernel_type,
            kernel_params=kernel_params,
            normalize_y=extra.get('normalize_y', True),
            optimize_hyperparams=extra.get('optimize_hyperparams', True),
            n_restarts=extra.get('n_restarts', 5)
        )
        
        # 初始化采集函数
        acq_kind = config.acquisition_function if config else "expected_improvement"
        self.acquisition = AcquisitionFunction(
            kind=acq_kind,
            kappa=extra.get('kappa', 2.576),
            xi=extra.get('xi', 0.01),
            minimize=extra.get('minimize', False)
        )
        
        # 批量建议器
        self.batch_suggester = BatchSuggester(
            strategy=extra.get('batch_strategy', 'kriging_believer')
        )
        
        # 约束处理器
        self.constraint_handler = ConstraintHandler()
        
        # 收敛检测器
        self.convergence_detector = ConvergenceDetector(
            max_iterations=extra.get('max_iterations', 100),
            no_improvement_rounds=extra.get('no_improvement_rounds', 10)
        )
        
        # 输入归一化器
        self.normalizer: Optional[InputNormalizer] = None
        
        # 参数空间
        self._param_names: List[str] = []
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        
        # 配置选项
        self._minimize = extra.get('minimize', False)
        self._n_candidates = extra.get('n_candidates', 1000)
        self._use_scipy_optimizer = extra.get('use_scipy_optimizer', True) and SCIPY_AVAILABLE
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.BAYESIAN_OPTIMIZATION
    
    def add_constraint(self, func: Callable[[Dict[str, Any]], float],
                      constraint_type: str = "ineq", name: str = "") -> None:
        """添加优化约束
        
        Args:
            func: 约束函数，返回值 >= 0 表示满足约束
            constraint_type: "ineq" (不等式) 或 "eq" (等式)
            name: 约束名称
        """
        constraint = Constraint(func=func, constraint_type=constraint_type, name=name)
        self.constraint_handler.add_constraint(constraint)
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化贝叶斯优化
        
        Args:
            context: 算法上下文
        """
        super().initialize(context)
        
        # 解析搜索空间
        self._parse_search_space(context.search_space)
        
        # 初始化归一化器
        self.normalizer = InputNormalizer(self._param_bounds)
        
        # 如果有历史观测，拟合高斯过程
        if self._observations:
            X, y = self._observations_to_arrays()
            self.gp.fit(X, y)
    
        self.logger.info(f"Bayesian optimization initialized with {len(self._param_names)} parameters")
    
    def suggest(self, context: AlgorithmContext, batch_size: int = 1) -> AlgorithmResult:
        """生成优化建议
        
        Args:
            context: 算法上下文
            batch_size: 批量大小（用于并行评估）
            
        Returns:
            算法结果
        """
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        self.acquisition.set_iteration(self._iteration_count)
        reasoning_steps = []
        
        # 检查收敛
        convergence_state = self.convergence_detector.state
        if convergence_state.converged:
            reasoning_steps.append(f"算法已收敛: {convergence_state.reason}")
            self.logger.info(f"Algorithm converged: {convergence_state.reason}")
        
        # 如果观测点不足，使用初始采样
        if len(self._observations) < self.config.n_initial_points:
            if batch_size > 1:
                actions = [self._initial_sample() for _ in range(batch_size)]
                action = actions[0]
            else:
                action = self._initial_sample()
            
            confidence = 0.5
            reasoning = f"初始探索阶段，随机采样第 {len(self._observations) + 1}/{self.config.n_initial_points} 个点"
            reasoning_steps.append("使用拉丁超立方采样进行初始探索")
        else:
            # 使用采集函数选择下一个点
            if batch_size > 1:
                # 批量建议
                action, acq_value, batch_actions = self._optimize_acquisition_batch(
                    context, batch_size
                )
                reasoning_steps.append(f"批量建议 {batch_size} 个点用于并行评估")
            else:
                action, acq_value = self._optimize_acquisition(context)
            
            # 计算置信度
            X_test = [self._action_to_array(action)]
            mean, std = self.gp.predict(X_test)
            confidence = self._compute_confidence(mean[0], std[0])
            
            # 检查约束满足情况
            is_feasible, violated = self.constraint_handler.check_feasibility(action)
            if not is_feasible:
                reasoning_steps.append(f"建议点违反约束: {violated}")
            
            reasoning_steps.extend([
                f"当前有 {len(self._observations)} 个观测点",
                f"使用 {self.config.acquisition_function} 采集函数",
                f"采集函数值: {acq_value:.4f}",
                f"预测均值: {mean[0]:.4f}, 标准差: {std[0]:.4f}",
                f"GP 核类型: {self.gp.kernel_type}",
                f"核参数: length_scale={self.gp.kernel_params.length_scale:.4f}"
            ])
            reasoning = f"基于高斯过程预测，使用{self.config.acquisition_function}采集函数选择最优点"
        
        # 生成备选方案
        alternatives = self._generate_alternatives(context, exclude=action)
        
        # 构建额外信息
        debug_info = {
            'n_observations': len(self._observations),
            'acquisition_function': self.config.acquisition_function,
            'exploration_weight': self.config.exploration_weight,
            'kernel_type': self.gp.kernel_type,
            'kernel_params': self.gp.kernel_params.to_dict(),
            'iteration': self._iteration_count,
            'converged': convergence_state.converged,
            'best_observed': self._get_best_observed()
        }
        
        if batch_size > 1 and len(self._observations) >= self.config.n_initial_points:
            debug_info['batch_suggestions'] = batch_actions
        
        return self._build_result(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            debug_info=debug_info
        )
    
    def _get_best_observed(self) -> Optional[Dict[str, Any]]:
        """获取当前最优观测"""
        if not self._observations:
            return None
        
        if self._minimize:
            best = min(self._observations, key=lambda x: x['reward'])
        else:
            best = max(self._observations, key=lambda x: x['reward'])
        
        return {
            'action': best['action'],
            'value': best['reward']
        }
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新高斯过程模型
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            context: 可选的上下文
        """
        # 如果是最小化问题，取负
        effective_reward = -reward if self._minimize else reward
        
        self._record_observation(action, reward)
        
        # 更新收敛检测器
        self.convergence_detector.update(effective_reward, self._iteration_count)
        
        # 重新拟合高斯过程
        if len(self._observations) >= 2:
            X, y = self._observations_to_arrays()
            self.gp.fit(X, y)
            
            self.logger.debug(
                f"GP updated with {len(self._observations)} observations, "
                f"best={self._get_best_observed()}"
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
        if len(self._observations) < 2:
            return 0.0
        
        X_test = [self._action_to_array(action)]
        mean, std = self.gp.predict(X_test)
        return mean[0]
    
    def _parse_search_space(self, search_space: Dict[str, Any]) -> None:
        """解析搜索空间
        
        Args:
            search_space: 搜索空间定义
        """
        self._param_names = []
        self._param_bounds = {}
        
        for param_name, param_def in search_space.items():
            if isinstance(param_def, dict):
                low = param_def.get('low', param_def.get('min', 0.0))
                high = param_def.get('high', param_def.get('max', 1.0))
            elif isinstance(param_def, (list, tuple)) and len(param_def) >= 2:
                low, high = param_def[0], param_def[1]
            else:
                low, high = 0.0, 1.0
            
            self._param_names.append(param_name)
            self._param_bounds[param_name] = (float(low), float(high))
    
    def _action_to_array(self, action: Dict[str, Any]) -> List[float]:
        """将动作转换为数组
        
        Args:
            action: 动作字典
            
        Returns:
            参数数组
        """
        result = []
        for name in self._param_names:
            value = action.get(name, 0.0)
            bounds = self._param_bounds.get(name, (0.0, 1.0))
            # 归一化到[0, 1]
            normalized = (value - bounds[0]) / (bounds[1] - bounds[0] + 1e-10)
            result.append(normalized)
        return result
    
    def _array_to_action(self, array: List[float]) -> Dict[str, Any]:
        """将数组转换为动作
        
        Args:
            array: 参数数组
            
        Returns:
            动作字典
        """
        action = {}
        for i, name in enumerate(self._param_names):
            bounds = self._param_bounds.get(name, (0.0, 1.0))
            # 反归一化
            value = array[i] * (bounds[1] - bounds[0]) + bounds[0]
            action[name] = value
        return action
    
    def _observations_to_arrays(self) -> Tuple[List[List[float]], List[float]]:
        """将观测转换为数组
        
        Returns:
            X和y数组
        """
        X = []
        y = []
        for obs in self._observations:
            X.append(self._action_to_array(obs['action']))
            y.append(obs['reward'])
        return X, y
    
    def _initial_sample(self) -> Dict[str, Any]:
        """初始采样
        
        使用拉丁超立方采样。
        
        Returns:
            采样的动作
        """
        n_samples = self.config.n_initial_points
        current_idx = len(self._observations)
        
        action = {}
        for name in self._param_names:
            bounds = self._param_bounds.get(name, (0.0, 1.0))
            # 分层采样
            segment = (bounds[1] - bounds[0]) / n_samples
            low = bounds[0] + current_idx * segment
            high = low + segment
            value = random.uniform(low, high)
            action[name] = value
        
        return action
    
    def _optimize_acquisition(self, context: AlgorithmContext) -> Tuple[Dict[str, Any], float]:
        """优化采集函数
        
        Args:
            context: 算法上下文
            
        Returns:
            最优动作和采集函数值
        """
        # 获取当前最佳值
        if self._observations:
            if self._minimize:
                y_best = -min(obs['reward'] for obs in self._observations)
            else:
                y_best = max(obs['reward'] for obs in self._observations)
        else:
            y_best = 0.0
        
        # 使用 scipy 优化器或随机搜索
        if self._use_scipy_optimizer and SCIPY_AVAILABLE:
            return self._optimize_acquisition_scipy(y_best)
        else:
            return self._optimize_acquisition_random(y_best)
    
    def _optimize_acquisition_scipy(self, y_best: float) -> Tuple[Dict[str, Any], float]:
        """使用 scipy 优化采集函数"""
        n_dims = len(self._param_names)
        bounds = [(0, 1) for _ in range(n_dims)]
        
        def neg_acquisition(x):
            x = np.atleast_2d(x)
            acq_value = self.acquisition.evaluate(self.gp, x, y_best)
            return -acq_value[0]
        
        best_x = None
        best_value = np.inf
        
        # 多次重启
        n_restarts = 5
        for i in range(n_restarts):
            if i == 0 and self._observations:
                # 从当前最优点开始
                best_obs = max(self._observations, key=lambda x: x['reward'])
                x0 = self._action_to_array(best_obs['action'])
            else:
                x0 = np.random.rand(n_dims)
            
            try:
                result = minimize(
                    neg_acquisition,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
                
                if result.fun < best_value:
                    best_value = result.fun
                    best_x = result.x
            except Exception as e:
                self.logger.debug(f"Scipy optimization failed: {e}")
                continue
        
        if best_x is None:
            return self._optimize_acquisition_random(y_best)
        
        return self._array_to_action(best_x.tolist()), float(-best_value)
    
    def _optimize_acquisition_random(self, y_best: float) -> Tuple[Dict[str, Any], float]:
        """使用随机搜索优化采集函数"""
        n_candidates = self._n_candidates
        n_dims = len(self._param_names)
        
        # 生成候选点（拉丁超立方 + 随机）
        n_lhs = n_candidates // 2
        
        # Latin Hypercube Sampling
        candidates_lhs = np.zeros((n_lhs, n_dims))
        for dim in range(n_dims):
            perms = np.random.permutation(n_lhs)
            candidates_lhs[:, dim] = (perms + np.random.rand(n_lhs)) / n_lhs
        
        # 随机采样
        n_random = n_candidates - n_lhs
        candidates_random = np.random.rand(n_random, n_dims)
        
        candidates = np.vstack([candidates_lhs, candidates_random])
        
        # 计算采集函数值
        acq_values = self.acquisition.evaluate(self.gp, candidates, y_best)
        
        # 考虑约束的可行性
        if self.constraint_handler.constraints:
            for i, candidate in enumerate(candidates):
                action = self._array_to_action(candidate.tolist())
                is_feasible, _ = self.constraint_handler.check_feasibility(action)
                if not is_feasible:
                    # 惩罚不可行点
                    penalty = self.constraint_handler.compute_penalty(action)
                    acq_values[i] -= penalty
        
        # 选择最佳候选
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]
        best_value = acq_values[best_idx]
        
        return self._array_to_action(best_candidate.tolist()), float(best_value)
    
    def _optimize_acquisition_batch(
        self, 
        context: AlgorithmContext, 
        batch_size: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """批量优化采集函数
        
        Returns:
            (首选动作, 采集函数值, 批量建议列表)
        """
        if self._observations:
            if self._minimize:
                y_best = -min(obs['reward'] for obs in self._observations)
            else:
                y_best = max(obs['reward'] for obs in self._observations)
        else:
            y_best = 0.0
        
        bounds = self.normalizer.get_normalized_bounds() if self.normalizer else \
                 [(0, 1) for _ in self._param_names]
        
        # 使用批量建议器
        suggestions = self.batch_suggester.suggest_batch(
            gp=self.gp,
            acq_func=self.acquisition,
            bounds=bounds,
            y_best=y_best,
            batch_size=batch_size,
            n_candidates=self._n_candidates
        )
        
        # 转换为动作字典
        batch_actions = [self._array_to_action(s.tolist()) for s in suggestions]
        
        # 计算首个建议的采集函数值
        acq_value = self.acquisition.evaluate(self.gp, [suggestions[0]], y_best)[0]
        
        return batch_actions[0], float(acq_value), batch_actions
    
    def _compute_confidence(self, mean: float, std: float) -> float:
        """计算置信度
        
        Args:
            mean: 预测均值
            std: 预测标准差
            
        Returns:
            置信度 (0-1)
        """
        # 基于不确定性计算置信度
        # 标准差越小，置信度越高
        if std < 0.01:
            return 0.95
        elif std < 0.1:
            return 0.85 - (std - 0.01) * 1.0
        elif std < 0.5:
            return 0.75 - (std - 0.1) * 0.5
        else:
            return max(0.5, 0.75 - std * 0.3)
    
    def _generate_alternatives(self, context: AlgorithmContext,
                              exclude: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """生成备选方案
        
        Args:
            context: 算法上下文
            exclude: 要排除的动作
            
        Returns:
            备选方案列表
        """
        alternatives = []
        
        if len(self._observations) < 2:
            return alternatives
        
        # 获取当前最佳值
        y_best = max(obs['reward'] for obs in self._observations)
        
        # 采样候选点
        n_candidates = 100
        candidates = []
        for _ in range(n_candidates):
            candidate = []
            for name in self._param_names:
                candidate.append(random.uniform(0, 1))
            candidates.append(candidate)
        
        # 预测
        mean, std = self.gp.predict(candidates)
        
        # 计算EI
        ei_values = AcquisitionFunction.expected_improvement(
            mean, std, y_best, xi=self.config.exploration_weight
        )
        
        # 选择Top-3
        top_indices = np.argsort(ei_values)[-3:][::-1]
        
        for idx in top_indices:
            action = self._array_to_action(candidates[idx])
            if exclude and action == exclude:
                continue
            alternatives.append({
                'action': action,
                'confidence': self._compute_confidence(mean[idx], std[idx]),
                'expected_improvement': float(ei_values[idx]),
                'reasoning': f"EI={ei_values[idx]:.4f}, 预测值={mean[idx]:.4f}"
            })
        
        return alternatives[:2]  # 返回最多2个备选
    
    # ==================== 热启动和状态管理 ====================
    
    def warm_start(self, X: List[Dict[str, Any]], y: List[float]) -> None:
        """热启动：从历史数据恢复
        
        Args:
            X: 历史参数配置列表
            y: 历史目标值列表
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        # 记录历史观测
        for action, reward in zip(X, y):
            self._record_observation(action, reward)
        
        # 拟合高斯过程
        if len(self._observations) >= 2:
            X_array, y_array = self._observations_to_arrays()
            self.gp.fit(X_array, y_array)
        
        self.logger.info(f"Warm started with {len(X)} historical observations")
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态（用于保存/恢复）"""
        return {
            'observations': self._observations.copy(),
            'iteration_count': self._iteration_count,
            'gp_state': self.gp.get_state().to_dict(),
            'param_names': self._param_names,
            'param_bounds': self._param_bounds,
            'convergence_state': {
                'converged': self.convergence_detector.state.converged,
                'reason': self.convergence_detector.state.reason,
                'best_value': self.convergence_detector.state.best_value,
                'history': self.convergence_detector.state.best_value_history
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        self._observations = state.get('observations', [])
        self._iteration_count = state.get('iteration_count', 0)
        self._param_names = state.get('param_names', [])
        self._param_bounds = state.get('param_bounds', {})
        
        # 恢复高斯过程状态
        gp_state_dict = state.get('gp_state', {})
        if gp_state_dict:
            gp_state = GaussianProcessState.from_dict(gp_state_dict)
            self.gp.load_state(gp_state)
        
        # 恢复归一化器
        if self._param_bounds:
            self.normalizer = InputNormalizer(self._param_bounds)
        
        self.logger.info(f"State loaded with {len(self._observations)} observations")
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据
        
        Returns:
            包含优化历史、GP预测等信息的字典
        """
        data = {
            'observations': [],
            'best_history': [],
            'predictions': None,
            'acquisition_surface': None
        }
        
        # 观测历史
        best_so_far = -np.inf if not self._minimize else np.inf
        for i, obs in enumerate(self._observations):
            reward = obs['reward']
            if self._minimize:
                best_so_far = min(best_so_far, reward)
            else:
                best_so_far = max(best_so_far, reward)
            
            data['observations'].append({
                'iteration': i + 1,
                'params': obs['action'],
                'value': reward
            })
            data['best_history'].append({
                'iteration': i + 1,
                'best_value': best_so_far
            })
        
        # 如果是 1D 或 2D 问题，生成预测网格
        if len(self._param_names) <= 2 and len(self._observations) >= 2:
            data['predictions'] = self._generate_prediction_grid()
        
        return data
    
    def _generate_prediction_grid(self, resolution: int = 50) -> Dict[str, Any]:
        """生成预测网格（用于可视化）"""
        n_dims = len(self._param_names)
        
        if n_dims == 1:
            # 1D 情况
            param_name = self._param_names[0]
            low, high = self._param_bounds[param_name]
            
            x_grid = np.linspace(0, 1, resolution).reshape(-1, 1)
            mean, std = self.gp.predict(x_grid)
            
            # 反归一化
            x_actual = x_grid * (high - low) + low
            
            return {
                'type': '1d',
                'x': x_actual.flatten().tolist(),
                'mean': mean.tolist(),
                'std': std.tolist(),
                'lower': (mean - 2 * std).tolist(),
                'upper': (mean + 2 * std).tolist(),
                'param_name': param_name
            }
        
        elif n_dims == 2:
            # 2D 情况
            x = np.linspace(0, 1, resolution)
            y = np.linspace(0, 1, resolution)
            X, Y = np.meshgrid(x, y)
            
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            mean, std = self.gp.predict(grid_points)
            
            # 反归一化
            param1, param2 = self._param_names[0], self._param_names[1]
            low1, high1 = self._param_bounds[param1]
            low2, high2 = self._param_bounds[param2]
            
            X_actual = X * (high1 - low1) + low1
            Y_actual = Y * (high2 - low2) + low2
            
            return {
                'type': '2d',
                'x': X_actual.tolist(),
                'y': Y_actual.tolist(),
                'mean': mean.reshape(resolution, resolution).tolist(),
                'std': std.reshape(resolution, resolution).tolist(),
                'param_names': [param1, param2]
            }
        
        return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._observations:
            return {'status': 'not_started'}
        
        rewards = [obs['reward'] for obs in self._observations]
        
        if self._minimize:
            best_idx = np.argmin(rewards)
            best_value = rewards[best_idx]
        else:
            best_idx = np.argmax(rewards)
            best_value = rewards[best_idx]
        
        return {
            'status': 'converged' if self.convergence_detector.state.converged else 'running',
            'convergence_reason': self.convergence_detector.state.reason,
            'total_iterations': self._iteration_count,
            'total_evaluations': len(self._observations),
            'best_value': best_value,
            'best_params': self._observations[best_idx]['action'],
            'best_iteration': best_idx + 1,
            'mean_value': float(np.mean(rewards)),
            'std_value': float(np.std(rewards)),
            'improvement_rate': self._compute_improvement_rate(),
            'gp_log_likelihood': self.gp.log_marginal_likelihood_value_,
            'kernel_info': {
                'type': self.gp.kernel_type,
                'params': self.gp.kernel_params.to_dict()
            }
        }
    
    def _compute_improvement_rate(self) -> float:
        """计算改进率"""
        if len(self._observations) < 2:
            return 0.0
        
        rewards = [obs['reward'] for obs in self._observations]
        
        # 计算累积最优
        if self._minimize:
            cumulative_best = np.minimum.accumulate(rewards)
        else:
            cumulative_best = np.maximum.accumulate(rewards)
        
        # 计算改进次数
        improvements = np.sum(np.diff(cumulative_best) != 0)
        
        return improvements / (len(self._observations) - 1)

