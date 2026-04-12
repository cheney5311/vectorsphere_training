# -*- coding: utf-8 -*-
"""
内存管理器

管理训练过程中的内存使用和优化。
"""

import logging
import gc
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from .device_types import DeviceType

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """内存压力等级"""
    LOW = "low"           # < 50%
    MEDIUM = "medium"     # 50-75%
    HIGH = "high"         # 75-90%
    CRITICAL = "critical" # > 90%


class OptimizationStrategy(Enum):
    """优化策略"""
    AGGRESSIVE = "aggressive"     # 激进优化
    BALANCED = "balanced"         # 平衡优化
    CONSERVATIVE = "conservative" # 保守优化


@dataclass
class MemoryStats:
    """内存统计"""
    total: int = 0
    used: int = 0
    free: int = 0
    reserved: int = 0
    allocated: int = 0
    max_allocated: int = 0
    
    # 新增：时间戳和设备信息
    timestamp: float = field(default_factory=time.time)
    device_type: str = "unknown"
    device_id: int = 0
    
    @property
    def total_gb(self) -> float:
        return self.total / (1024 ** 3)
    
    @property
    def used_gb(self) -> float:
        return self.used / (1024 ** 3)
    
    @property
    def free_gb(self) -> float:
        return self.free / (1024 ** 3)
    
    @property
    def reserved_gb(self) -> float:
        """预留内存（GB）"""
        return self.reserved / (1024 ** 3)
    
    @property
    def allocated_gb(self) -> float:
        """已分配内存（GB）"""
        return self.allocated / (1024 ** 3)
    
    @property
    def max_allocated_gb(self) -> float:
        """峰值分配内存（GB）"""
        return self.max_allocated / (1024 ** 3)
    
    @property
    def usage_percent(self) -> float:
        if self.total > 0:
            return (self.used / self.total) * 100
        return 0.0
    
    @property
    def fragmentation_ratio(self) -> float:
        """碎片化比率"""
        if self.reserved > 0:
            return (self.reserved - self.allocated) / self.reserved
        return 0.0
    
    @property
    def pressure_level(self) -> MemoryPressure:
        """内存压力等级"""
        usage = self.usage_percent
        if usage < 50:
            return MemoryPressure.LOW
        elif usage < 75:
            return MemoryPressure.MEDIUM
        elif usage < 90:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'used': self.used,
            'free': self.free,
            'reserved': self.reserved,
            'allocated': self.allocated,
            'max_allocated': self.max_allocated,
            'total_gb': self.total_gb,
            'used_gb': self.used_gb,
            'free_gb': self.free_gb,
            'reserved_gb': self.reserved_gb,
            'allocated_gb': self.allocated_gb,
            'max_allocated_gb': self.max_allocated_gb,
            'usage_percent': self.usage_percent,
            'fragmentation_ratio': self.fragmentation_ratio,
            'pressure_level': self.pressure_level.value,
            'timestamp': self.timestamp,
            'device_type': self.device_type,
            'device_id': self.device_id,
        }
    
    def get_available_for_allocation(self, safety_margin: float = 0.1) -> int:
        """
        获取可用于分配的内存
        
        Args:
            safety_margin: 安全边际（0-1）
            
        Returns:
            可用字节数
        """
        available = self.free * (1 - safety_margin)
        return max(0, int(available))
    
    def can_allocate(self, size_bytes: int, safety_margin: float = 0.1) -> bool:
        """
        检查是否可以分配指定大小的内存
        
        Args:
            size_bytes: 所需字节数
            safety_margin: 安全边际
            
        Returns:
            是否可以分配
        """
        return self.get_available_for_allocation(safety_margin) >= size_bytes
    
    def __str__(self) -> str:
        return (
            f"Memory: {self.used_gb:.2f}/{self.total_gb:.2f} GB "
            f"({self.usage_percent:.1f}% used, {self.pressure_level.value} pressure)"
        )


@dataclass
class MemoryEvent:
    """内存事件"""
    timestamp: float
    event_type: str  # allocation, deallocation, oom, cleanup
    size_bytes: int
    stats_before: Optional[MemoryStats] = None
    stats_after: Optional[MemoryStats] = None
    context: str = ""


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, max_history: int = 1000):
        """
        初始化内存监控器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self._history: List[MemoryStats] = []
        self._events: List[MemoryEvent] = []
        self._oom_count = 0
        self._cleanup_count = 0
        
    def record(self, stats: MemoryStats) -> None:
        """
        记录内存统计
        
        Args:
            stats: 内存统计
        """
        self._history.append(stats)
        if len(self._history) > self.max_history:
            self._history.pop(0)
    
    def record_event(self, event: MemoryEvent) -> None:
        """
        记录内存事件
        
        Args:
            event: 内存事件
        """
        self._events.append(event)
        if len(self._events) > self.max_history:
            self._events.pop(0)
        
        if event.event_type == 'oom':
            self._oom_count += 1
        elif event.event_type == 'cleanup':
            self._cleanup_count += 1
    
    def get_trend(self, window_size: int = 100) -> Dict[str, float]:
        """
        获取内存趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            趋势信息
        """
        if len(self._history) < 2:
            return {'trend': 0.0, 'avg_usage': 0.0}
        
        recent = self._history[-window_size:]
        usages = [s.usage_percent for s in recent]
        
        if len(usages) < 2:
            return {'trend': 0.0, 'avg_usage': usages[0] if usages else 0.0}
        
        # 简单线性趋势：最后值 - 第一值
        trend = usages[-1] - usages[0]
        avg_usage = sum(usages) / len(usages)
        
        return {
            'trend': trend,
            'avg_usage': avg_usage,
            'max_usage': max(usages),
            'min_usage': min(usages),
            'current_usage': usages[-1]
        }
    
    def get_peak_stats(self) -> Optional[MemoryStats]:
        """获取峰值内存统计"""
        if not self._history:
            return None
        
        return max(self._history, key=lambda s: s.used)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._history:
            return {}
        
        recent_stats = self._history[-100:]
        usages = [s.usage_percent for s in recent_stats]
        
        import statistics
        
        return {
            'total_records': len(self._history),
            'oom_count': self._oom_count,
            'cleanup_count': self._cleanup_count,
            'avg_usage': statistics.mean(usages) if usages else 0.0,
            'max_usage': max(usages) if usages else 0.0,
            'min_usage': min(usages) if usages else 0.0,
            'std_usage': statistics.stdev(usages) if len(usages) > 1 else 0.0,
        }
    
    def diagnose(self, current_stats: MemoryStats) -> Dict[str, Any]:
        """
        诊断内存状态
        
        Args:
            current_stats: 当前内存统计
            
        Returns:
            诊断信息
        """
        issues = []
        warnings = []
        recommendations = []
        
        # 检查内存压力
        if current_stats.pressure_level == MemoryPressure.CRITICAL:
            issues.append(f"Critical memory pressure: {current_stats.usage_percent:.1f}% used")
            recommendations.append("Enable gradient checkpointing or reduce batch size")
        elif current_stats.pressure_level == MemoryPressure.HIGH:
            warnings.append(f"High memory pressure: {current_stats.usage_percent:.1f}% used")
            recommendations.append("Consider memory optimization strategies")
        
        # 检查碎片化
        if current_stats.fragmentation_ratio > 0.3:
            warnings.append(f"High memory fragmentation: {current_stats.fragmentation_ratio:.1%}")
            recommendations.append("Consider clearing cache or restarting")
        
        # 检查OOM频率
        if self._oom_count > 0:
            issues.append(f"Out of memory errors: {self._oom_count}")
            recommendations.append("Reduce model size or enable memory optimizations")
        
        # 检查趋势
        trend = self.get_trend()
        if trend.get('trend', 0) > 10:
            warnings.append(f"Memory usage increasing: +{trend['trend']:.1f}%")
            recommendations.append("Check for memory leaks")
        
        # 确定状态
        if issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'statistics': self.get_statistics(),
            'trend': trend
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self._history.clear()
        self._events.clear()
        self._oom_count = 0
        self._cleanup_count = 0


class MemoryProfiler:
    """内存性能分析器"""
    
    def __init__(self):
        self._enabled = False
        self._allocations: Dict[str, List[int]] = defaultdict(list)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    @contextmanager
    def profile(self, name: str, device: torch.device):
        """
        分析内存使用
        
        Args:
            name: 区域名称
            device: 设备
        """
        if not self._enabled:
            yield
            return
        
        # 记录开始状态
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            start_mem = torch.cuda.memory_allocated(device)
        else:
            start_mem = 0
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # 记录结束状态
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                end_mem = torch.cuda.memory_allocated(device)
                self._allocations[name].append(end_mem - start_mem)
            
            duration = time.time() - start_time
            self._timings[name].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取统计信息"""
        import statistics
        
        stats = {}
        
        for name in set(list(self._allocations.keys()) + list(self._timings.keys())):
            stat = {}
            
            if name in self._allocations:
                allocs = self._allocations[name]
                stat['allocations'] = {
                    'count': len(allocs),
                    'total_mb': sum(allocs) / (1024**2),
                    'avg_mb': statistics.mean(allocs) / (1024**2) if allocs else 0,
                    'max_mb': max(allocs) / (1024**2) if allocs else 0,
                }
            
            if name in self._timings:
                times = self._timings[name]
                stat['timing'] = {
                    'count': len(times),
                    'total_ms': sum(times) * 1000,
                    'avg_ms': statistics.mean(times) * 1000 if times else 0,
                    'max_ms': max(times) * 1000 if times else 0,
                }
            
            stats[name] = stat
        
        return stats
    
    def reset(self) -> None:
        """重置分析器"""
        self._allocations.clear()
        self._timings.clear()
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available")
            return
        
        print("\n" + "="*80)
        print("Memory Profiling Summary")
        print("="*80)
        
        for name, stat in sorted(stats.items()):
            print(f"\nRegion: {name}")
            if 'allocations' in stat:
                alloc = stat['allocations']
                print(f"  Allocations: {alloc['count']}")
                print(f"  Total: {alloc['total_mb']:.2f} MB")
                print(f"  Average: {alloc['avg_mb']:.2f} MB")
                print(f"  Max: {alloc['max_mb']:.2f} MB")
            if 'timing' in stat:
                timing = stat['timing']
                print(f"  Timing: {timing['count']} calls")
                print(f"  Total: {timing['total_ms']:.2f} ms")
                print(f"  Average: {timing['avg_ms']:.2f} ms")
        
        print("="*80)


class MemoryManager:
    """
    内存管理器
    
    提供内存监控、清理和优化功能。
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._memory_history: List[MemoryStats] = []
        self._peak_memory: int = 0
        
        # 新增：监控和分析
        self._monitor = MemoryMonitor()
        self._profiler = MemoryProfiler()
        
        # 新增：自动清理配置
        self._auto_cleanup_enabled = False
        self._cleanup_threshold = 0.85  # 85%使用率时自动清理
    
    def get_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        if self.device.type == 'cuda':
            return self._get_cuda_stats()
        else:
            return self._get_cpu_stats()
    
    def _get_cuda_stats(self) -> MemoryStats:
        """获取CUDA内存统计"""
        device_id = self.device.index or 0
        
        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(device_id)
        allocated = torch.cuda.memory_allocated(device_id)
        max_allocated = torch.cuda.max_memory_allocated(device_id)
        
        return MemoryStats(
            total=total,
            used=allocated,
            free=total - reserved,
            reserved=reserved,
            allocated=allocated,
            max_allocated=max_allocated,
            device_type='cuda',
            device_id=device_id
        )
    
    def _get_cpu_stats(self) -> MemoryStats:
        """获取CPU内存统计"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return MemoryStats(
                total=mem.total,
                used=mem.used,
                free=mem.available,
                reserved=0,
                allocated=mem.used,
                max_allocated=0,
                device_type='cpu',
                device_id=0
            )
        except ImportError:
            return MemoryStats(device_type='cpu', device_id=0)
    
    def record_stats(self):
        """记录当前内存统计"""
        stats = self.get_stats()
        self._memory_history.append(stats)
        self._monitor.record(stats)
        
        if stats.allocated > self._peak_memory:
            self._peak_memory = stats.allocated
        
        # 自动清理检查
        if self._auto_cleanup_enabled and stats.usage_percent > self._cleanup_threshold * 100:
            logger.warning(f"Memory usage {stats.usage_percent:.1f}% exceeds threshold, auto-cleaning")
            self.clear_cache()
    
    def get_history(self) -> List[MemoryStats]:
        """获取内存历史"""
        return self._memory_history.copy()
    
    def get_peak_memory(self) -> int:
        """获取峰值内存使用"""
        return self._peak_memory
    
    def get_peak_memory_gb(self) -> float:
        """获取峰值内存使用（GB）"""
        return self._peak_memory / (1024 ** 3)
    
    def clear_cache(self):
        """清理缓存"""
        stats_before = self.get_stats()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
        gc.collect()
        
        stats_after = self.get_stats()
        
        # 记录清理事件
        freed = stats_before.used - stats_after.used
        event = MemoryEvent(
            timestamp=time.time(),
            event_type='cleanup',
            size_bytes=freed,
            stats_before=stats_before,
            stats_after=stats_after
        )
        self._monitor.record_event(event)
        
        logger.info(f"Memory cache cleared, freed {freed / (1024**2):.1f} MB")
    
    def reset_stats(self):
        """重置统计"""
        self._memory_history.clear()
        self._peak_memory = 0
        self._monitor.reset()
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        logger.info("Memory stats reset")
    
    # ==================== 新增方法 ====================
    
    def enable_auto_cleanup(self, threshold: float = 0.85) -> None:
        """
        启用自动清理
        
        Args:
            threshold: 使用率阈值（0-1）
        """
        self._auto_cleanup_enabled = True
        self._cleanup_threshold = threshold
        logger.info(f"Auto cleanup enabled at {threshold*100:.0f}% threshold")
    
    def disable_auto_cleanup(self) -> None:
        """禁用自动清理"""
        self._auto_cleanup_enabled = False
        logger.info("Auto cleanup disabled")
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        self._profiler.enable()
        logger.info("Memory profiling enabled")
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        self._profiler.disable()
        logger.info("Memory profiling disabled")
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取性能分析统计"""
        return self._profiler.get_stats()
    
    def print_profiling_summary(self) -> None:
        """打印性能分析摘要"""
        self._profiler.print_summary()
    
    @contextmanager
    def profile_region(self, name: str):
        """
        分析内存区域
        
        Args:
            name: 区域名称
        """
        with self._profiler.profile(name, self.device):
            yield
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return self._monitor.get_statistics()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断内存状态"""
        current_stats = self.get_stats()
        return self._monitor.diagnose(current_stats)
    
    def get_trend(self, window_size: int = 100) -> Dict[str, float]:
        """获取内存趋势"""
        return self._monitor.get_trend(window_size)
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "="*80)
        print("Memory Diagnosis")
        print("="*80)
        
        print(f"\nStatus: {diagnosis['status'].upper()}")
        
        if diagnosis['issues']:
            print("\n⛔ Issues:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
        
        if diagnosis['warnings']:
            print("\n⚠️  Warnings:")
            for warning in diagnosis['warnings']:
                print(f"  - {warning}")
        
        if diagnosis['recommendations']:
            print("\n💡 Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        if not diagnosis['issues'] and not diagnosis['warnings']:
            print("\n✅ Memory state is healthy")
        
        # 打印统计
        stats = diagnosis.get('statistics', {})
        if stats:
            print(f"\nStatistics:")
            print(f"  Average usage: {stats.get('avg_usage', 0):.1f}%")
            print(f"  Max usage: {stats.get('max_usage', 0):.1f}%")
            print(f"  OOM count: {stats.get('oom_count', 0)}")
            print(f"  Cleanup count: {stats.get('cleanup_count', 0)}")
        
        print("="*80)
    
    def estimate_model_memory(self, model: nn.Module) -> int:
        """估算模型内存占用"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # 估算梯度内存（假设所有参数都需要梯度）
        grad_size = param_size
        
        # 估算优化器状态（Adam约3倍参数大小）
        optimizer_size = param_size * 3
        
        total = param_size + buffer_size + grad_size + optimizer_size
        
        logger.info(f"Model memory estimate: {total / (1024**3):.2f} GB "
                   f"(params: {param_size / (1024**2):.1f} MB, "
                   f"grads: {grad_size / (1024**2):.1f} MB, "
                   f"optimizer: {optimizer_size / (1024**2):.1f} MB)")
        
        return total
    
    def can_fit_model(self, model: nn.Module, margin: float = 0.1) -> bool:
        """检查是否可以容纳模型"""
        estimated = self.estimate_model_memory(model)
        stats = self.get_stats()
        
        required = estimated * (1 + margin)  # 添加余量
        
        return stats.free > required
    
    def estimate_batch_size(
        self,
        model: nn.Module,
        sample_size_bytes: int,
        margin: float = 0.2
    ) -> int:
        """
        估算可支持的批次大小
        
        Args:
            model: 模型
            sample_size_bytes: 单个样本大小（字节）
            margin: 安全边际
            
        Returns:
            估算的批次大小
        """
        model_memory = self.estimate_model_memory(model)
        stats = self.get_stats()
        
        available = stats.free * (1 - margin)
        remaining = available - model_memory
        
        if remaining <= 0:
            return 1
        
        batch_size = int(remaining / sample_size_bytes)
        return max(1, batch_size)
    
    def print_memory_summary(self) -> None:
        """打印内存摘要"""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("Memory Summary")
        print("="*80)
        
        print(f"\nDevice: {stats.device_type}:{stats.device_id}")
        print(f"Total: {stats.total_gb:.2f} GB")
        print(f"Used: {stats.used_gb:.2f} GB ({stats.usage_percent:.1f}%)")
        print(f"Free: {stats.free_gb:.2f} GB")
        print(f"Reserved: {stats.reserved_gb:.2f} GB")
        print(f"Allocated: {stats.allocated_gb:.2f} GB")
        print(f"Peak: {self.get_peak_memory_gb():.2f} GB")
        print(f"Pressure: {stats.pressure_level.value}")
        print(f"Fragmentation: {stats.fragmentation_ratio:.1%}")
        
        # 历史统计
        if len(self._memory_history) > 1:
            trend = self.get_trend()
            print(f"\nTrend (last 100 records):")
            print(f"  Average usage: {trend.get('avg_usage', 0):.1f}%")
            print(f"  Trend: {trend.get('trend', 0):+.1f}%")
        
        print("="*80)


class MemoryOptimizer:
    """
    内存优化器
    
    提供各种内存优化策略。
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._optimizations: List[str] = []
        self._strategy = OptimizationStrategy.BALANCED
    
    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """
        设置优化策略
        
        Args:
            strategy: 优化策略
        """
        self._strategy = strategy
        logger.info(f"Memory optimization strategy set to: {strategy.value}")
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self._optimizations.append('gradient_checkpointing')
            logger.info("Enabled gradient checkpointing")
        else:
            # 尝试手动启用
            count = 0
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    count += 1
            if count > 0:
                self._optimizations.append('gradient_checkpointing')
                logger.info(f"Enabled gradient checkpointing for {count} modules")
    
    def enable_activation_checkpointing(self, modules: Optional[List[nn.Module]] = None):
        """启用激活检查点"""
        if modules is None:
            # 自动选择大型模块
            modules = [
                m for m in self.model.modules()
                if sum(p.numel() for p in m.parameters()) > 1e6
            ]
        
        self._optimizations.append('activation_checkpointing')
        logger.info(f"Enabled activation checkpointing for {len(modules)} modules")
    
    def enable_mixed_precision(self, dtype: torch.dtype = torch.float16):
        """启用混合精度"""
        self.model = self.model.to(dtype)
        self._optimizations.append(f'mixed_precision_{dtype}')
        logger.info(f"Enabled mixed precision: {dtype}")
    
    def offload_to_cpu(self, modules: Optional[List[str]] = None):
        """将部分模块卸载到CPU"""
        if modules is None:
            return
        
        count = 0
        for name, module in self.model.named_modules():
            if any(name.startswith(m) for m in modules):
                module.to('cpu')
                count += 1
        
        self._optimizations.append('cpu_offload')
        logger.info(f"Offloaded {count} modules to CPU")
    
    def optimize_memory_layout(self):
        """优化内存布局"""
        count = 0
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
                count += 1
        
        self._optimizations.append('memory_layout')
        logger.info(f"Optimized memory layout for {count} parameters")
    
    def apply_torch_compile(self):
        """应用Torch编译优化"""
        try:
            self.model = torch.compile(self.model)
            self._optimizations.append('torch_compile')
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}")
    
    def enable_cpu_pinning(self):
        """启用CPU内存固定"""
        for param in self.model.parameters():
            if param.device.type == 'cpu':
                param.data = param.data.pin_memory()
        
        self._optimizations.append('cpu_pinning')
        logger.info("Enabled CPU memory pinning")
    
    def auto_optimize(self, target_memory_gb: Optional[float] = None) -> None:
        """
        自动优化内存
        
        根据策略和目标内存自动应用优化
        
        Args:
            target_memory_gb: 目标内存限制（GB）
        """
        logger.info(f"Auto-optimizing with strategy: {self._strategy.value}")
        
        if self._strategy == OptimizationStrategy.CONSERVATIVE:
            # 保守策略：只优化布局
            self.optimize_memory_layout()
        
        elif self._strategy == OptimizationStrategy.BALANCED:
            # 平衡策略：布局 + 梯度检查点
            self.optimize_memory_layout()
            self.enable_gradient_checkpointing()
        
        elif self._strategy == OptimizationStrategy.AGGRESSIVE:
            # 激进策略：所有优化
            self.optimize_memory_layout()
            self.enable_gradient_checkpointing()
            self.enable_activation_checkpointing()
            
            # 如果内存仍然不足，启用混合精度
            if target_memory_gb:
                manager = MemoryManager(self.device)
                estimated = manager.estimate_model_memory(self.model)
                if estimated / (1024**3) > target_memory_gb:
                    self.enable_mixed_precision(torch.float16)
    
    def get_applied_optimizations(self) -> List[str]:
        """获取已应用的优化"""
        return self._optimizations.copy()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        获取优化报告
        
        Returns:
            优化报告字典
        """
        return {
            'strategy': self._strategy.value,
            'optimizations': self._optimizations,
            'count': len(self._optimizations),
        }
    
    def print_report(self) -> None:
        """打印优化报告"""
        report = self.get_optimization_report()
        
        print("\n" + "="*80)
        print("Memory Optimization Report")
        print("="*80)
        
        print(f"\nStrategy: {report['strategy']}")
        print(f"Optimizations applied: {report['count']}")
        
        if report['optimizations']:
            print("\nOptimizations:")
            for opt in report['optimizations']:
                print(f"  ✓ {opt}")
        else:
            print("\nNo optimizations applied")
        
        print("="*80)


class GradientCheckpointing:
    """
    梯度检查点管理器
    
    提供细粒度的梯度检查点控制。
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._checkpoint_modules: List[nn.Module] = []
        self._enabled = False
    
    def enable(self, modules: Optional[List[nn.Module]] = None):
        """
        启用梯度检查点
        
        Args:
            modules: 要启用检查点的模块列表，None表示自动选择
        """
        if modules is None:
            # 自动选择transformer层
            modules = [
                m for m in self.model.modules()
                if 'TransformerLayer' in type(m).__name__ or
                   'Attention' in type(m).__name__
            ]
        
        self._checkpoint_modules = modules
        self._enabled = True
        
        # 尝试使用模型内置方法
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Gradient checkpointing enabled for {len(modules)} modules")
    
    def disable(self):
        """禁用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        self._enabled = False
        logger.info("Gradient checkpointing disabled")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def checkpoint(
        self, 
        module: nn.Module, 
        *inputs
    ) -> Tensor:
        """
        对模块应用检查点
        
        Args:
            module: 要检查点的模块
            *inputs: 输入张量
        """
        from torch.utils.checkpoint import checkpoint
        
        if self._enabled:
            return checkpoint(module, *inputs, use_reentrant=False)
        else:
            return module(*inputs)

    def get_checkpointed_modules(self) -> List[nn.Module]:
        """获取检查点模块列表"""
        return self._checkpoint_modules.copy()
    
    def estimate_memory_savings(self) -> float:
        """
        估算内存节省
        
        Returns:
            估算的节省比例（0-1）
        """
        if not self._enabled or not self._checkpoint_modules:
            return 0.0
        
        # 梯度检查点通常可以节省30-70%的激活内存
        # 这里返回保守估计
        return 0.3
    
    def get_status(self) -> Dict[str, Any]:
        """获取检查点状态"""
        return {
            'enabled': self._enabled,
            'checkpointed_modules': len(self._checkpoint_modules),
            'estimated_savings': self.estimate_memory_savings()
        }


# ==================== 工具函数 ====================

def clear_memory(device: Optional[torch.device] = None):
    """
    清理内存
    
    便捷函数，清理GPU缓存和Python垃圾。
    """
    gc.collect()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    
    logger.debug("Memory cleared")


def get_memory_summary(device: Optional[torch.device] = None) -> str:
    """获取内存摘要字符串"""
    manager = MemoryManager(device)
    stats = manager.get_stats()
    
    return (
        f"Memory: {stats.used_gb:.2f}/{stats.total_gb:.2f} GB "
        f"({stats.usage_percent:.1f}% used)"
    )


def get_available_memory(device: Optional[torch.device] = None) -> int:
    """
    获取可用内存（字节）
    
    Args:
        device: 设备
        
    Returns:
        可用字节数
    """
    manager = MemoryManager(device)
    stats = manager.get_stats()
    return stats.free


def estimate_tensor_memory(tensor: Tensor) -> int:
    """
    估算张量内存占用
    
    Args:
        tensor: 张量
        
    Returns:
        字节数
    """
    return tensor.numel() * tensor.element_size()


def compare_memory_usage(
    before_stats: MemoryStats,
    after_stats: MemoryStats
) -> Dict[str, Any]:
    """
    比较内存使用
    
    Args:
        before_stats: 之前的统计
        after_stats: 之后的统计
        
    Returns:
        比较结果
    """
    return {
        'used_delta_mb': (after_stats.used - before_stats.used) / (1024**2),
        'free_delta_mb': (after_stats.free - before_stats.free) / (1024**2),
        'usage_delta_percent': after_stats.usage_percent - before_stats.usage_percent,
        'allocated_delta_mb': (after_stats.allocated - before_stats.allocated) / (1024**2),
    }


@contextmanager
def track_memory(device: Optional[torch.device] = None, name: str = "operation"):
    """
    跟踪内存使用的上下文管理器
    
    Args:
        device: 设备
        name: 操作名称
        
    Yields:
        内存管理器
    """
    manager = MemoryManager(device)
    stats_before = manager.get_stats()
    
    try:
        yield manager
    finally:
        stats_after = manager.get_stats()
        delta = compare_memory_usage(stats_before, stats_after)
        
        logger.info(f"Memory tracking for '{name}': "
                   f"allocated {delta['allocated_delta_mb']:+.1f} MB, "
                   f"usage {delta['usage_delta_percent']:+.1f}%")


@contextmanager
def managed_memory(
    device: Optional[torch.device] = None,
    enable_auto_cleanup: bool = True,
    cleanup_threshold: float = 0.85
):
    """
    托管的内存上下文
    
    自动管理内存清理和监控
    
    Args:
        device: 设备
        enable_auto_cleanup: 是否启用自动清理
        cleanup_threshold: 清理阈值
        
    Yields:
        内存管理器
    """
    manager = MemoryManager(device)
    
    if enable_auto_cleanup:
        manager.enable_auto_cleanup(cleanup_threshold)
    
    try:
        yield manager
    finally:
        # 最终清理
        manager.clear_cache()


def optimize_model_memory(
    model: nn.Module,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    device: Optional[torch.device] = None
) -> MemoryOptimizer:
    """
    优化模型内存
    
    便捷函数，自动应用内存优化
    
    Args:
        model: 模型
        strategy: 优化策略
        device: 设备
        
    Returns:
        优化器实例
    """
    optimizer = MemoryOptimizer(model, device)
    optimizer.set_strategy(strategy)
    optimizer.auto_optimize()
    return optimizer


def print_memory_report(device: Optional[torch.device] = None) -> None:
    """
    打印详细的内存报告
    
    Args:
        device: 设备
    """
    manager = MemoryManager(device)
    manager.print_memory_summary()
    manager.print_diagnosis()


def recommend_batch_size(
    model: nn.Module,
    sample_size_mb: float,
    device: Optional[torch.device] = None,
    target_usage: float = 0.8
) -> int:
    """
    推荐批次大小
    
    Args:
        model: 模型
        sample_size_mb: 单个样本大小（MB）
        device: 设备
        target_usage: 目标使用率（0-1）
        
    Returns:
        推荐的批次大小
    """
    manager = MemoryManager(device)
    model_memory = manager.estimate_model_memory(model)
    stats = manager.get_stats()
    
    # 计算可用内存
    target_memory = stats.total * target_usage
    available = target_memory - model_memory
    
    if available <= 0:
        return 1
    
    batch_size = int(available / (sample_size_mb * 1024**2))
    return max(1, batch_size)


def check_memory_health(device: Optional[torch.device] = None) -> bool:
    """
    检查内存健康状态
    
    Args:
        device: 设备
        
    Returns:
        是否健康
    """
    manager = MemoryManager(device)
    diagnosis = manager.diagnose()
    return diagnosis['status'] in ['healthy', 'warning']


@contextmanager
def emergency_memory_cleanup(device: Optional[torch.device] = None):
    """
    紧急内存清理上下文
    
    在内存不足时自动清理
    
    Args:
        device: 设备
        
    Yields:
        None
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("Out of memory detected, performing emergency cleanup")
            clear_memory(device)
            gc.collect()
            raise
        else:
            raise
