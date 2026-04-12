# -*- coding: utf-8 -*-
"""
混合精度训练管理

管理自动混合精度(AMP)训练。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, ContextManager, List, Tuple, Set
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from .device_types import PrecisionType

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """精度模式"""
    FP32 = "fp32"           # 全精度
    FP16 = "fp16"           # 半精度
    BF16 = "bf16"           # BFloat16
    MIXED_FP16 = "mixed_fp16"  # 混合精度FP16
    MIXED_BF16 = "mixed_bf16"  # 混合精度BF16
    
    @classmethod
    def from_string(cls, s: str) -> 'PrecisionMode':
        """从字符串创建精度模式"""
        s = s.lower().strip()
        for mode in cls:
            if mode.value == s:
                return mode
        return cls.FP32
    
    @property
    def is_mixed(self) -> bool:
        """是否是混合精度模式"""
        return self in (PrecisionMode.MIXED_FP16, PrecisionMode.MIXED_BF16)
    
    @property
    def is_low_precision(self) -> bool:
        """是否是低精度模式"""
        return self != PrecisionMode.FP32
    
    @property
    def requires_scaler(self) -> bool:
        """是否需要梯度缩放器"""
        return self in (PrecisionMode.FP16, PrecisionMode.MIXED_FP16)
    
    def to_dtype(self) -> torch.dtype:
        """转换为torch.dtype"""
        if self in (PrecisionMode.FP16, PrecisionMode.MIXED_FP16):
            return torch.float16
        elif self in (PrecisionMode.BF16, PrecisionMode.MIXED_BF16):
            return torch.bfloat16
        else:
            return torch.float32
    
    def get_memory_savings(self) -> float:
        """获取相对于FP32的内存节省比例"""
        if self.is_low_precision:
            return 0.5  # FP16/BF16约节省50%内存
        return 0.0
    
    def get_speedup_factor(self) -> float:
        """获取相对于FP32的加速比（估算）"""
        if self == PrecisionMode.FP32:
            return 1.0
        elif self in (PrecisionMode.FP16, PrecisionMode.MIXED_FP16):
            return 2.0  # FP16通常有2x加速
        elif self in (PrecisionMode.BF16, PrecisionMode.MIXED_BF16):
            return 1.8  # BF16略慢于FP16
        return 1.0
    
    def get_precision_loss_risk(self) -> str:
        """获取精度损失风险等级"""
        if self == PrecisionMode.FP32:
            return "none"
        elif self in (PrecisionMode.BF16, PrecisionMode.MIXED_BF16):
            return "low"
        elif self in (PrecisionMode.FP16, PrecisionMode.MIXED_FP16):
            return "medium"
        return "unknown"


@dataclass
class AmpConfig:
    """AMP配置"""
    enabled: bool = True
    precision: PrecisionMode = PrecisionMode.MIXED_FP16
    
    # GradScaler配置
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # 优化
    cache_enabled: bool = True
    
    # 新增：高级配置
    dynamic_loss_scale: bool = True
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    
    # 新增：白名单/黑名单（操作类型）
    whitelist_ops: Set[str] = field(default_factory=set)
    blacklist_ops: Set[str] = field(default_factory=set)
    
    # 新增：性能监控
    enable_profiling: bool = False
    log_overflow: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'precision': self.precision.value,
            'init_scale': self.init_scale,
            'growth_factor': self.growth_factor,
            'backoff_factor': self.backoff_factor,
            'growth_interval': self.growth_interval,
            'cache_enabled': self.cache_enabled,
            'dynamic_loss_scale': self.dynamic_loss_scale,
            'max_scale': self.max_scale,
            'min_scale': self.min_scale,
            'whitelist_ops': list(self.whitelist_ops),
            'blacklist_ops': list(self.blacklist_ops),
            'enable_profiling': self.enable_profiling,
            'log_overflow': self.log_overflow,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AmpConfig':
        """从字典创建配置"""
        return cls(
            enabled=data.get('enabled', True),
            precision=PrecisionMode.from_string(data.get('precision', 'mixed_fp16')),
            init_scale=data.get('init_scale', 65536.0),
            growth_factor=data.get('growth_factor', 2.0),
            backoff_factor=data.get('backoff_factor', 0.5),
            growth_interval=data.get('growth_interval', 2000),
            cache_enabled=data.get('cache_enabled', True),
            dynamic_loss_scale=data.get('dynamic_loss_scale', True),
            max_scale=data.get('max_scale', 2.0 ** 24),
            min_scale=data.get('min_scale', 1.0),
            whitelist_ops=set(data.get('whitelist_ops', [])),
            blacklist_ops=set(data.get('blacklist_ops', [])),
            enable_profiling=data.get('enable_profiling', False),
            log_overflow=data.get('log_overflow', True),
        )
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误消息列表
        """
        errors = []
        
        if self.init_scale <= 0:
            errors.append(f"init_scale must be positive, got {self.init_scale}")
        
        if self.growth_factor <= 1.0:
            errors.append(f"growth_factor must be > 1.0, got {self.growth_factor}")
        
        if self.backoff_factor <= 0 or self.backoff_factor >= 1.0:
            errors.append(f"backoff_factor must be in (0, 1), got {self.backoff_factor}")
        
        if self.growth_interval <= 0:
            errors.append(f"growth_interval must be positive, got {self.growth_interval}")
        
        if self.max_scale < self.min_scale:
            errors.append(f"max_scale ({self.max_scale}) < min_scale ({self.min_scale})")
        
        return errors
    
    def optimize_for_stability(self) -> None:
        """优化配置以提高稳定性"""
        self.init_scale = 1024.0  # 较低的初始缩放
        self.growth_factor = 1.5  # 较慢的增长
        self.growth_interval = 5000  # 较长的增长间隔
        logger.info("Config optimized for stability")
    
    def optimize_for_performance(self) -> None:
        """优化配置以提高性能"""
        self.init_scale = 65536.0  # 较高的初始缩放
        self.growth_factor = 2.0  # 较快的增长
        self.growth_interval = 1000  # 较短的增长间隔
        self.cache_enabled = True
        logger.info("Config optimized for performance")


@dataclass
class PrecisionStats:
    """精度统计信息"""
    total_steps: int = 0
    overflow_steps: int = 0
    successful_steps: int = 0
    total_forward_time: float = 0.0
    total_backward_time: float = 0.0
    scale_updates: List[Tuple[int, float]] = field(default_factory=list)  # (step, scale)
    
    @property
    def overflow_rate(self) -> float:
        """溢出率"""
        if self.total_steps == 0:
            return 0.0
        return self.overflow_steps / self.total_steps
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps
    
    @property
    def avg_forward_time(self) -> float:
        """平均前向传播时间"""
        if self.successful_steps == 0:
            return 0.0
        return self.total_forward_time / self.successful_steps
    
    @property
    def avg_backward_time(self) -> float:
        """平均反向传播时间"""
        if self.successful_steps == 0:
            return 0.0
        return self.total_backward_time / self.successful_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_steps': self.total_steps,
            'overflow_steps': self.overflow_steps,
            'successful_steps': self.successful_steps,
            'overflow_rate': self.overflow_rate,
            'success_rate': self.success_rate,
            'avg_forward_time_ms': self.avg_forward_time * 1000,
            'avg_backward_time_ms': self.avg_backward_time * 1000,
            'current_scale': self.scale_updates[-1][1] if self.scale_updates else 0.0,
        }


class PrecisionMonitor:
    """精度监控器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化精度监控器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.stats = PrecisionStats()
        self._overflow_history: List[Tuple[int, str]] = []  # (step, reason)
        self._scale_history: List[float] = []
        
    def record_step(
        self,
        step: int,
        scale: float,
        is_overflow: bool = False,
        forward_time: float = 0.0,
        backward_time: float = 0.0
    ) -> None:
        """
        记录训练步骤
        
        Args:
            step: 步骤编号
            scale: 当前缩放因子
            is_overflow: 是否发生溢出
            forward_time: 前向传播时间
            backward_time: 反向传播时间
        """
        self.stats.total_steps += 1
        
        if is_overflow:
            self.stats.overflow_steps += 1
            self._overflow_history.append((step, "gradient_overflow"))
            if len(self._overflow_history) > self.max_history:
                self._overflow_history.pop(0)
        else:
            self.stats.successful_steps += 1
            self.stats.total_forward_time += forward_time
            self.stats.total_backward_time += backward_time
        
        # 记录缩放因子
        self.stats.scale_updates.append((step, scale))
        self._scale_history.append(scale)
        if len(self.stats.scale_updates) > self.max_history:
            self.stats.scale_updates.pop(0)
        if len(self._scale_history) > self.max_history:
            self._scale_history.pop(0)
    
    def get_overflow_trend(self, window_size: int = 100) -> float:
        """
        获取溢出趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            最近窗口的溢出率
        """
        if len(self._overflow_history) == 0:
            return 0.0
        
        recent_overflows = [
            step for step, _ in self._overflow_history 
            if step >= self.stats.total_steps - window_size
        ]
        
        return len(recent_overflows) / min(window_size, self.stats.total_steps)
    
    def get_scale_stability(self) -> Dict[str, float]:
        """
        获取缩放因子稳定性指标
        
        Returns:
            稳定性指标字典
        """
        if len(self._scale_history) < 2:
            return {'stability': 1.0, 'variance': 0.0}
        
        import statistics
        
        variance = statistics.variance(self._scale_history)
        mean = statistics.mean(self._scale_history)
        
        # 变异系数（CV）作为稳定性指标
        cv = (variance ** 0.5) / mean if mean > 0 else 0.0
        stability = max(0.0, 1.0 - cv)
        
        return {
            'stability': stability,
            'variance': variance,
            'mean_scale': mean,
            'coefficient_of_variation': cv
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断精度状态
        
        Returns:
            诊断信息
        """
        issues = []
        warnings = []
        recommendations = []
        
        # 检查溢出率
        if self.stats.overflow_rate > 0.1:
            issues.append(f"High overflow rate: {self.stats.overflow_rate:.2%}")
            recommendations.append("Consider using lower initial scale or BF16 precision")
        elif self.stats.overflow_rate > 0.05:
            warnings.append(f"Moderate overflow rate: {self.stats.overflow_rate:.2%}")
        
        # 检查溢出趋势
        overflow_trend = self.get_overflow_trend()
        if overflow_trend > 0.15:
            issues.append(f"Increasing overflow trend: {overflow_trend:.2%}")
            recommendations.append("Review model architecture for numerical stability")
        
        # 检查缩放稳定性
        stability = self.get_scale_stability()
        if stability['stability'] < 0.5:
            warnings.append(f"Unstable loss scaling: {stability['stability']:.2f}")
            recommendations.append("Consider adjusting growth_factor or growth_interval")
        
        # 确定状态
        if issues:
            status = 'unstable'
        elif warnings:
            status = 'suboptimal'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'stats': self.stats.to_dict(),
            'stability': stability
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            **self.stats.to_dict(),
            **self.get_scale_stability(),
            'overflow_trend': self.get_overflow_trend()
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self.stats = PrecisionStats()
        self._overflow_history.clear()
        self._scale_history.clear()


class PrecisionProfiler:
    """精度性能分析器"""
    
    def __init__(self):
        self._enabled = False
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._current_region: Optional[Tuple[str, float]] = None
        
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    @contextmanager
    def profile_region(self, name: str):
        """
        分析代码区域
        
        Args:
            name: 区域名称
        """
        if not self._enabled:
            yield
            return
        
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._timings[name].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        stats = {}
        
        for name, times in self._timings.items():
            if times:
                import statistics
                stats[name] = {
                    'count': len(times),
                    'total_ms': sum(times) * 1000,
                    'avg_ms': statistics.mean(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
                }
        
        return stats
    
    def reset(self) -> None:
        """重置分析器"""
        self._timings.clear()
        self._current_region = None
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available")
            return
        
        print("\n" + "="*80)
        print("Precision Profiling Summary")
        print("="*80)
        
        for region, region_stats in sorted(stats.items()):
            print(f"\nRegion: {region}")
            print(f"  Count: {region_stats['count']}")
            print(f"  Total: {region_stats['total_ms']:.2f} ms")
            print(f"  Average: {region_stats['avg_ms']:.2f} ms")
            print(f"  Min: {region_stats['min_ms']:.2f} ms")
            print(f"  Max: {region_stats['max_ms']:.2f} ms")
            print(f"  Std Dev: {region_stats['std_ms']:.2f} ms")
        
        print("="*80)


class MixedPrecisionManager:
    """
    混合精度训练管理器
    
    统一管理AMP相关的所有操作。
    """
    
    def __init__(
        self, 
        config: Optional[AmpConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or AmpConfig()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 检查设备支持
        self._check_support()
        
        # 初始化GradScaler
        self.scaler = self._create_scaler() if self.config.enabled else None
        
        # 统计
        self._overflow_count = 0
        self._step_count = 0
        
        # 新增：监控和分析
        self._monitor = PrecisionMonitor()
        self._profiler = PrecisionProfiler()
        if self.config.enable_profiling:
            self._profiler.enable()
        
        # 新增：性能跟踪
        self._last_scale = self.get_scale()
        self._scale_history: List[float] = []
    
    def _check_support(self):
        """检查设备支持"""
        if not torch.cuda.is_available():
            if self.config.precision in [PrecisionMode.MIXED_FP16, PrecisionMode.MIXED_BF16]:
                logger.warning("CUDA not available, falling back to FP32")
                self.config.precision = PrecisionMode.FP32
                self.config.enabled = False
            return
        
        # 检查BF16支持
        if self.config.precision == PrecisionMode.MIXED_BF16:
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 not supported, falling back to FP16")
                self.config.precision = PrecisionMode.MIXED_FP16
    
    def _create_scaler(self) -> Optional[GradScaler]:
        """创建GradScaler"""
        if self.config.precision == PrecisionMode.MIXED_BF16:
            # BF16不需要scaler
            return None
        
        return GradScaler(
            init_scale=self.config.init_scale,
            growth_factor=self.config.growth_factor,
            backoff_factor=self.config.backoff_factor,
            growth_interval=self.config.growth_interval,
            enabled=self.config.enabled
        )
    
    @property
    def dtype(self) -> torch.dtype:
        """获取当前精度的dtype"""
        if self.config.precision in [PrecisionMode.FP16, PrecisionMode.MIXED_FP16]:
            return torch.float16
        elif self.config.precision in [PrecisionMode.BF16, PrecisionMode.MIXED_BF16]:
            return torch.bfloat16
        else:
            return torch.float32
    
    @property
    def is_enabled(self) -> bool:
        """是否启用混合精度"""
        return self.config.enabled and self.device.type == 'cuda'
    
    @contextmanager
    def autocast_context(self):
        """
        自动混合精度上下文
        
        Usage:
            with manager.autocast_context():
                output = model(input)
                loss = criterion(output, target)
        """
        if not self.is_enabled:
            yield
            return
        
        with autocast(
            device_type='cuda',
            dtype=self.dtype,
            cache_enabled=self.config.cache_enabled
        ):
            yield
    
    def scale_loss(self, loss: Tensor) -> Tensor:
        """
        缩放损失
        
        Args:
            loss: 原始损失
            
        Returns:
            缩放后的损失
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: Tensor, create_graph: bool = False):
        """
        执行反向传播
        
        Args:
            loss: 损失张量
            create_graph: 是否创建计算图
        """
        start_time = time.time()
        
        try:
            if self.scaler is not None:
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward(create_graph=create_graph)
            else:
                loss.backward(create_graph=create_graph)
            
            # 记录时间
            if hasattr(self, '_monitor') and self.config.enable_profiling:
                backward_time = time.time() - start_time
                self._monitor.record_step(
                    step=self._step_count,
                    scale=self.get_scale(),
                    backward_time=backward_time
                )
        except RuntimeError as e:
            if "inf" in str(e).lower() or "nan" in str(e).lower():
                logger.warning(f"Numerical instability detected in backward: {e}")
                self._overflow_count += 1
            raise
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        反缩放梯度
        
        在梯度裁剪前调用。
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        执行优化器步骤
        
        Args:
            optimizer: 优化器
            
        Returns:
            是否成功更新（无溢出）
        """
        self._step_count += 1
        
        if self.scaler is not None:
            # 检查溢出
            scale_before = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            
            # 判断是否溢出
            is_overflow = scale_after < scale_before
            if is_overflow:
                self._overflow_count += 1
                if self.config.log_overflow:
                    logger.warning(f"Gradient overflow detected at step {self._step_count}, "
                                 f"scale: {scale_before:.0f} -> {scale_after:.0f}")
            
            # 更新监控
            if hasattr(self, '_monitor'):
                self._monitor.record_step(
                    step=self._step_count,
                    scale=scale_after,
                    is_overflow=is_overflow
                )
            
            # 记录历史
            if hasattr(self, '_scale_history'):
                self._scale_history.append(scale_after)
                if len(self._scale_history) > 1000:
                    self._scale_history.pop(0)
            
            return not is_overflow
        else:
            optimizer.step()
            if hasattr(self, '_monitor'):
                self._monitor.record_step(
                    step=self._step_count,
                    scale=1.0,
                    is_overflow=False
                )
            return True
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'enabled': self.is_enabled,
            'precision': self.config.precision.value,
            'dtype': str(self.dtype),
            'scale': self.get_scale(),
            'step_count': self._step_count,
            'overflow_count': self._overflow_count,
            'overflow_rate': self._overflow_count / max(self._step_count, 1)
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            'config': self.config.to_dict(),
            'step_count': self._step_count,
            'overflow_count': self._overflow_count,
            'scale_history': self._scale_history[-100:] if hasattr(self, '_scale_history') else []
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        """加载状态字典"""
        self._step_count = state.get('step_count', 0)
        self._overflow_count = state.get('overflow_count', 0)
        if hasattr(self, '_scale_history'):
            self._scale_history = state.get('scale_history', [])
        if self.scaler is not None and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        logger.info(f"Loaded precision state: {self._step_count} steps, "
                   f"{self._overflow_count} overflows")
    
    # ==================== 新增方法 ====================
    
    def get_monitor_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return self._monitor.get_summary() if hasattr(self, '_monitor') else {}
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断精度状态"""
        return self._monitor.diagnose() if hasattr(self, '_monitor') else {'status': 'unknown'}
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        if hasattr(self, '_profiler'):
            self._profiler.enable()
            self.config.enable_profiling = True
            logger.info("Precision profiling enabled")
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        if hasattr(self, '_profiler'):
            self._profiler.disable()
            self.config.enable_profiling = False
            logger.info("Precision profiling disabled")
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, float]]:
        """获取性能分析统计"""
        if hasattr(self, '_profiler'):
            return self._profiler.get_stats()
        return {}
    
    def print_profiling_summary(self) -> None:
        """打印性能分析摘要"""
        if hasattr(self, '_profiler'):
            self._profiler.print_summary()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._overflow_count = 0
        self._step_count = 0
        if hasattr(self, '_scale_history'):
            self._scale_history.clear()
        if hasattr(self, '_monitor'):
            self._monitor.reset()
        if hasattr(self, '_profiler'):
            self._profiler.reset()
        logger.info("Precision stats reset")
    
    def set_scale(self, scale: float) -> None:
        """
        设置缩放因子
        
        Args:
            scale: 新的缩放因子
        """
        if self.scaler is not None:
            self.scaler._scale = torch.tensor(scale)
            logger.info(f"Loss scale manually set to {scale}")
    
    def optimize_config(self) -> None:
        """根据运行时统计优化配置"""
        if not hasattr(self, '_monitor'):
            return
        
        diagnosis = self.diagnose()
        
        if diagnosis['status'] == 'unstable':
            logger.warning("Detected unstable precision, optimizing for stability")
            self.config.optimize_for_stability()
            # 重新创建scaler
            self.scaler = self._create_scaler()
        elif diagnosis['status'] == 'healthy':
            logger.info("Precision is healthy, optimizing for performance")
            self.config.optimize_for_performance()
            self.scaler = self._create_scaler()
    
    def get_memory_savings(self) -> Dict[str, float]:
        """
        估算内存节省
        
        Returns:
            内存节省信息
        """
        savings_ratio = self.config.precision.get_memory_savings()
        
        return {
            'savings_ratio': savings_ratio,
            'savings_percent': savings_ratio * 100,
            'speedup_factor': self.config.precision.get_speedup_factor(),
            'precision_loss_risk': self.config.precision.get_precision_loss_risk()
        }
    
    @contextmanager
    def profile_region(self, name: str):
        """
        分析代码区域
        
        Args:
            name: 区域名称
        """
        if hasattr(self, '_profiler'):
            with self._profiler.profile_region(name):
                yield
        else:
            yield


class AmpContext:
    """
    AMP上下文管理器
    
    便捷的混合精度训练上下文。
    
    Usage:
        with AmpContext() as amp:
            output = model(input)
            loss = criterion(output, target)
            amp.backward(loss)
            amp.step(optimizer)
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        precision: PrecisionMode = PrecisionMode.MIXED_FP16,
        device: Optional[torch.device] = None
    ):
        config = AmpConfig(enabled=enabled, precision=precision)
        self._manager = MixedPrecisionManager(config, device)
        self._autocast_ctx = None
    
    def __enter__(self) -> 'AmpContext':
        if self._manager.is_enabled:
            self._autocast_ctx = autocast(
                device_type='cuda',
                dtype=self._manager.dtype,
                cache_enabled=self._manager.config.cache_enabled
            )
            self._autocast_ctx.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc_val, exc_tb)
    
    def backward(self, loss: Tensor, create_graph: bool = False):
        """执行反向传播"""
        self._manager.backward(loss, create_graph)
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """执行优化步骤"""
        return self._manager.step(optimizer)
    
    def unscale(self, optimizer: torch.optim.Optimizer):
        """反缩放梯度"""
        self._manager.unscale_gradients(optimizer)
    
    @property
    def scaler(self) -> Optional[GradScaler]:
        """获取GradScaler"""
        return self._manager.scaler
    
    @property
    def scale(self) -> float:
        """获取当前缩放因子"""
        return self._manager.get_scale()


# ==================== 全局实例和便捷函数 ====================

_amp_context: Optional[MixedPrecisionManager] = None


def get_amp_context(
    enabled: bool = True,
    precision: PrecisionMode = PrecisionMode.MIXED_FP16
) -> MixedPrecisionManager:
    """
    获取全局AMP上下文
    
    Args:
        enabled: 是否启用
        precision: 精度模式
    """
    global _amp_context
    if _amp_context is None:
        config = AmpConfig(enabled=enabled, precision=precision)
        _amp_context = MixedPrecisionManager(config)
    return _amp_context


@contextmanager
def amp_autocast(
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None
):
    """
    便捷的autocast上下文
    
    Usage:
        with amp_autocast():
            output = model(input)
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    with autocast(device_type='cuda', dtype=dtype):
        yield


def cast_model_to_precision(
    model: nn.Module,
    precision: PrecisionMode
) -> nn.Module:
    """
    将模型转换为指定精度
    
    Args:
        model: 模型
        precision: 目标精度
    """
    if precision == PrecisionMode.FP16:
        return model.half()
    elif precision == PrecisionMode.BF16:
        return model.bfloat16()
    else:
        return model.float()


# ==================== 新增工具函数 ====================

def convert_tensor_precision(
    tensor: Tensor,
    target_precision: PrecisionMode
) -> Tensor:
    """
    转换张量精度
    
    Args:
        tensor: 输入张量
        target_precision: 目标精度
        
    Returns:
        转换后的张量
    """
    dtype = target_precision.to_dtype()
    return tensor.to(dtype=dtype)


def analyze_model_precision(model: nn.Module) -> Dict[str, Any]:
    """
    分析模型的精度分布
    
    Args:
        model: 模型
        
    Returns:
        精度分析结果
    """
    precision_counts = defaultdict(int)
    total_params = 0
    
    for name, param in model.named_parameters():
        precision_counts[str(param.dtype)] += param.numel()
        total_params += param.numel()
    
    precision_distribution = {
        dtype: count / total_params 
        for dtype, count in precision_counts.items()
    }
    
    return {
        'total_parameters': total_params,
        'precision_counts': dict(precision_counts),
        'precision_distribution': precision_distribution,
        'mixed_precision': len(precision_counts) > 1
    }


def recommend_precision_mode(
    model_size_gb: float,
    available_memory_gb: float,
    task_type: str = "training"
) -> PrecisionMode:
    """
    推荐精度模式
    
    Args:
        model_size_gb: 模型大小（GB）
        available_memory_gb: 可用内存（GB）
        task_type: 任务类型（training/inference）
        
    Returns:
        推荐的精度模式
    """
    memory_ratio = model_size_gb / available_memory_gb
    
    # 内存紧张，必须使用混合精度
    if memory_ratio > 0.7:
        if torch.cuda.is_bf16_supported():
            return PrecisionMode.MIXED_BF16
        return PrecisionMode.MIXED_FP16
    
    # 内存充足
    elif memory_ratio < 0.3:
        return PrecisionMode.FP32
    
    # 中等内存压力
    else:
        if task_type == "training":
            # 训练时使用混合精度可以加速
            if torch.cuda.is_bf16_supported():
                return PrecisionMode.MIXED_BF16
            return PrecisionMode.MIXED_FP16
        else:
            # 推理时可以用FP32保证精度
            return PrecisionMode.FP32


def estimate_precision_speedup(
    model_flops: float,
    precision_mode: PrecisionMode
) -> Dict[str, float]:
    """
    估算精度加速比
    
    Args:
        model_flops: 模型浮点运算数
        precision_mode: 精度模式
        
    Returns:
        加速比估算
    """
    speedup = precision_mode.get_speedup_factor()
    
    return {
        'theoretical_speedup': speedup,
        'expected_training_speedup': speedup * 0.7,  # 考虑I/O等开销
        'expected_inference_speedup': speedup * 0.8,
        'memory_savings': precision_mode.get_memory_savings()
    }


def create_precision_optimizer(
    model: nn.Module,
    base_optimizer: torch.optim.Optimizer,
    precision_mode: PrecisionMode,
    **amp_config_kwargs
) -> Tuple[torch.optim.Optimizer, MixedPrecisionManager]:
    """
    创建带精度管理的优化器
    
    Args:
        model: 模型
        base_optimizer: 基础优化器
        precision_mode: 精度模式
        **amp_config_kwargs: AMP配置参数
        
    Returns:
        (优化器, 精度管理器)
    """
    config = AmpConfig(
        enabled=precision_mode.is_mixed,
        precision=precision_mode,
        **amp_config_kwargs
    )
    
    manager = MixedPrecisionManager(config)
    
    return base_optimizer, manager


def compare_precision_modes(
    model: nn.Module,
    modes: List[PrecisionMode]
) -> Dict[PrecisionMode, Dict[str, Any]]:
    """
    比较不同精度模式
    
    Args:
        model: 模型
        modes: 精度模式列表
        
    Returns:
        比较结果
    """
    results = {}
    
    for mode in modes:
        # 估算内存使用
        total_params = sum(p.numel() for p in model.parameters())
        dtype_size = mode.to_dtype().itemsize
        model_memory_mb = (total_params * dtype_size) / (1024**2)
        
        results[mode] = {
            'model_memory_mb': model_memory_mb,
            'memory_savings': mode.get_memory_savings(),
            'speedup_factor': mode.get_speedup_factor(),
            'precision_loss_risk': mode.get_precision_loss_risk(),
            'requires_scaler': mode.requires_scaler,
            'is_mixed': mode.is_mixed
        }
    
    return results


def validate_precision_config(
    config: AmpConfig,
    device: torch.device
) -> List[str]:
    """
    验证精度配置
    
    Args:
        config: AMP配置
        device: 设备
        
    Returns:
        错误消息列表
    """
    errors = config.validate()
    
    # 检查设备支持
    if device.type != 'cuda' and config.precision.is_mixed:
        errors.append(f"Mixed precision requires CUDA, but device is {device.type}")
    
    if config.precision == PrecisionMode.MIXED_BF16:
        if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            errors.append("BF16 not supported on this CUDA device")
    
    return errors


def print_precision_info(manager: MixedPrecisionManager) -> None:
    """
    打印精度信息
    
    Args:
        manager: 精度管理器
    """
    print("\n" + "="*80)
    print("Mixed Precision Training Information")
    print("="*80)
    
    stats = manager.get_stats()
    print(f"\nConfiguration:")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  Precision: {stats['precision']}")
    print(f"  DType: {stats['dtype']}")
    print(f"  Current Scale: {stats['scale']:.0f}")
    
    print(f"\nStatistics:")
    print(f"  Total Steps: {stats['step_count']}")
    print(f"  Overflow Count: {stats['overflow_count']}")
    print(f"  Overflow Rate: {stats['overflow_rate']:.2%}")
    
    savings = manager.get_memory_savings()
    print(f"\nPerformance:")
    print(f"  Memory Savings: {savings['savings_percent']:.1f}%")
    print(f"  Speedup Factor: {savings['speedup_factor']:.1f}x")
    print(f"  Precision Loss Risk: {savings['precision_loss_risk']}")
    
    # 诊断
    diagnosis = manager.diagnose()
    print(f"\nHealth Status: {diagnosis['status'].upper()}")
    if diagnosis['issues']:
        print("  Issues:")
        for issue in diagnosis['issues']:
            print(f"    - {issue}")
    if diagnosis['warnings']:
        print("  Warnings:")
        for warning in diagnosis['warnings']:
            print(f"    - {warning}")
    
    print("="*80)


def auto_select_precision(
    model: nn.Module,
    device: torch.device,
    target_batch_size: int,
    sample_memory_mb: float
) -> PrecisionMode:
    """
    自动选择精度模式
    
    Args:
        model: 模型
        device: 设备
        target_batch_size: 目标批次大小
        sample_memory_mb: 单个样本内存（MB）
        
    Returns:
        推荐的精度模式
    """
    if device.type != 'cuda':
        return PrecisionMode.FP32
    
    # 估算模型内存
    total_params = sum(p.numel() for p in model.parameters())
    model_memory_gb = (total_params * 4) / (1024**3)  # FP32
    
    # 获取可用内存
    if torch.cuda.is_available():
        available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    else:
        available_memory_gb = 8.0  # 默认假设
    
    # 估算所需内存
    batch_memory_gb = (target_batch_size * sample_memory_mb) / 1024
    total_required_gb = model_memory_gb * 3 + batch_memory_gb  # 3x for gradients and optimizer
    
    # 选择精度
    if total_required_gb > available_memory_gb * 0.9:
        # 内存不足，使用混合精度
        if torch.cuda.is_bf16_supported():
            return PrecisionMode.MIXED_BF16
        return PrecisionMode.MIXED_FP16
    elif total_required_gb < available_memory_gb * 0.5:
        # 内存充足，使用FP32
        return PrecisionMode.FP32
    else:
        # 中等内存压力，使用混合精度以加速
        if torch.cuda.is_bf16_supported():
            return PrecisionMode.MIXED_BF16
        return PrecisionMode.MIXED_FP16


@contextmanager
def managed_precision_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    precision: PrecisionMode = PrecisionMode.MIXED_FP16,
    enable_profiling: bool = False
):
    """
    托管的精度训练上下文
    
    自动处理精度管理和统计
    
    Args:
        model: 模型
        optimizer: 优化器
        precision: 精度模式
        enable_profiling: 是否启用性能分析
        
    Yields:
        精度管理器
    """
    config = AmpConfig(
        enabled=precision.is_mixed,
        precision=precision,
        enable_profiling=enable_profiling
    )
    
    manager = MixedPrecisionManager(config)
    
    try:
        yield manager
    finally:
        if enable_profiling:
            manager.print_profiling_summary()
        
        # 打印最终统计
        stats = manager.get_stats()
        if stats['step_count'] > 0:
            logger.info(f"Precision training completed: {stats['step_count']} steps, "
                       f"{stats['overflow_count']} overflows ({stats['overflow_rate']:.2%})")
