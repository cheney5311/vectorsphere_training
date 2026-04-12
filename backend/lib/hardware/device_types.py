# -*- coding: utf-8 -*-
"""
设备类型定义

定义硬件设备相关的类型和数据结构。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum
import re

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    GPU = "gpu"          # NVIDIA CUDA GPU
    NPU = "npu"          # 华为昇腾 NPU
    TPU = "tpu"          # Google TPU
    MPS = "mps"          # Apple Metal Performance Shaders
    XPU = "xpu"          # Intel GPU
    ROCM = "rocm"        # AMD ROCm GPU
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, s: str) -> 'DeviceType':
        """从字符串创建设备类型"""
        s = s.lower().strip()
        for device_type in cls:
            if device_type.value == s:
                return device_type
        return cls.UNKNOWN
    
    @property
    def supports_cuda(self) -> bool:
        """是否支持CUDA"""
        return self in (DeviceType.GPU, DeviceType.ROCM)
    
    @property
    def is_accelerator(self) -> bool:
        """是否是加速器"""
        return self != DeviceType.CPU and self != DeviceType.UNKNOWN
    
    @property
    def requires_driver(self) -> bool:
        """是否需要驱动"""
        return self.is_accelerator
    
    def get_framework_name(self) -> str:
        """获取框架名称"""
        framework_map = {
            DeviceType.GPU: 'cuda',
            DeviceType.NPU: 'torch_npu',
            DeviceType.TPU: 'torch_xla',
            DeviceType.MPS: 'mps',
            DeviceType.XPU: 'torch_xpu',
            DeviceType.ROCM: 'rocm',
            DeviceType.CPU: 'cpu'
        }
        return framework_map.get(self, 'cpu')
    
    def get_memory_efficiency(self) -> float:
        """获取内存效率评分（0-1）"""
        efficiency_map = {
            DeviceType.GPU: 0.95,
            DeviceType.NPU: 0.90,
            DeviceType.TPU: 0.98,
            DeviceType.MPS: 0.85,
            DeviceType.XPU: 0.88,
            DeviceType.ROCM: 0.93,
            DeviceType.CPU: 0.70,
            DeviceType.UNKNOWN: 0.50
        }
        return efficiency_map.get(self, 0.50)
    
    def get_compute_efficiency(self) -> float:
        """获取计算效率评分（0-1）"""
        efficiency_map = {
            DeviceType.GPU: 0.95,
            DeviceType.NPU: 0.90,
            DeviceType.TPU: 0.98,
            DeviceType.MPS: 0.80,
            DeviceType.XPU: 0.85,
            DeviceType.ROCM: 0.92,
            DeviceType.CPU: 0.60,
            DeviceType.UNKNOWN: 0.50
        }
        return efficiency_map.get(self, 0.50)


class PrecisionType(Enum):
    """精度类型"""
    FP32 = "fp32"        # 全精度
    FP16 = "fp16"        # 半精度
    BF16 = "bf16"        # Brain Float 16
    INT8 = "int8"        # 8位整数
    INT4 = "int4"        # 4位整数
    
    @classmethod
    def from_string(cls, s: str) -> 'PrecisionType':
        """从字符串创建精度类型"""
        s = s.lower().strip()
        for precision in cls:
            if precision.value == s:
                return precision
        return cls.FP32
    
    @property
    def bits(self) -> int:
        """获取位数"""
        bits_map = {
            PrecisionType.FP32: 32,
            PrecisionType.FP16: 16,
            PrecisionType.BF16: 16,
            PrecisionType.INT8: 8,
            PrecisionType.INT4: 4
        }
        return bits_map.get(self, 32)
    
    @property
    def bytes_per_element(self) -> int:
        """每个元素的字节数"""
        return self.bits // 8
    
    @property
    def is_floating_point(self) -> bool:
        """是否是浮点类型"""
        return self in (PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16)
    
    @property
    def is_integer(self) -> bool:
        """是否是整数类型"""
        return self in (PrecisionType.INT8, PrecisionType.INT4)
    
    def get_memory_multiplier(self) -> float:
        """相对于FP32的内存倍数"""
        return self.bits / 32.0
    
    def get_speed_multiplier(self) -> float:
        """相对于FP32的速度倍数（估算）"""
        multiplier_map = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 2.0,
            PrecisionType.BF16: 2.0,
            PrecisionType.INT8: 4.0,
            PrecisionType.INT4: 8.0
        }
        return multiplier_map.get(self, 1.0)
    
    def can_cast_to(self, target: 'PrecisionType') -> bool:
        """是否可以安全转换到目标精度"""
        # FP32可以转换到任何类型
        if self == PrecisionType.FP32:
            return True
        # FP16和BF16可以互转
        if self in (PrecisionType.FP16, PrecisionType.BF16) and target in (PrecisionType.FP16, PrecisionType.BF16):
            return True
        # 整数类型可以向上转换
        if self.is_integer and target.is_floating_point:
            return True
        if self == PrecisionType.INT4 and target == PrecisionType.INT8:
            return True
        return False
    
    def get_dynamic_range(self) -> Tuple[float, float]:
        """获取动态范围"""
        range_map = {
            PrecisionType.FP32: (-3.4e38, 3.4e38),
            PrecisionType.FP16: (-65504, 65504),
            PrecisionType.BF16: (-3.4e38, 3.4e38),  # 范围类似FP32
            PrecisionType.INT8: (-128, 127),
            PrecisionType.INT4: (-8, 7)
        }
        return range_map.get(self, (-1e10, 1e10))


@dataclass
class DeviceCapabilities:
    """设备能力信息"""
    # 计算能力
    compute_capability: str = ""  # e.g., "8.6" for NVIDIA
    
    # 支持的精度类型
    supported_precisions: List[PrecisionType] = field(default_factory=lambda: [PrecisionType.FP32])
    
    # 特性支持
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_tensor_cores: bool = False
    supports_flash_attention: bool = False
    
    # 并行能力
    max_threads_per_block: int = 1024
    max_shared_memory_per_block: int = 48 * 1024  # 48KB
    warp_size: int = 32
    
    # 内存
    supports_unified_memory: bool = False
    supports_memory_pool: bool = False
    
    # 新增：高级特性
    supports_dynamic_parallelism: bool = False
    supports_cooperative_groups: bool = False
    supports_async_copy: bool = False
    max_grid_size: Tuple[int, int, int] = (2147483647, 65535, 65535)
    
    # 新增：内存特性
    memory_clock_rate: int = 0  # MHz
    memory_bus_width: int = 0   # bits
    l2_cache_size: int = 0      # bytes
    
    # 新增：性能指标
    peak_flops_fp32: float = 0.0  # TFLOPS
    peak_flops_fp16: float = 0.0  # TFLOPS
    peak_memory_bandwidth: float = 0.0  # GB/s
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'compute_capability': self.compute_capability,
            'supported_precisions': [p.value for p in self.supported_precisions],
            'supports_fp16': self.supports_fp16,
            'supports_bf16': self.supports_bf16,
            'supports_int8': self.supports_int8,
            'supports_tensor_cores': self.supports_tensor_cores,
            'supports_flash_attention': self.supports_flash_attention,
            'max_threads_per_block': self.max_threads_per_block,
            'max_shared_memory_per_block': self.max_shared_memory_per_block,
            'warp_size': self.warp_size,
            'supports_unified_memory': self.supports_unified_memory,
            'supports_memory_pool': self.supports_memory_pool,
            'supports_dynamic_parallelism': self.supports_dynamic_parallelism,
            'supports_cooperative_groups': self.supports_cooperative_groups,
            'supports_async_copy': self.supports_async_copy,
            'max_grid_size': self.max_grid_size,
            'memory_clock_rate': self.memory_clock_rate,
            'memory_bus_width': self.memory_bus_width,
            'l2_cache_size': self.l2_cache_size,
            'peak_flops_fp32': self.peak_flops_fp32,
            'peak_flops_fp16': self.peak_flops_fp16,
            'peak_memory_bandwidth': self.peak_memory_bandwidth,
        }
    
    def supports_precision(self, precision: PrecisionType) -> bool:
        """检查是否支持指定精度"""
        if precision in self.supported_precisions:
            return True
        
        # 检查显式标志
        if precision == PrecisionType.FP16 and self.supports_fp16:
            return True
        if precision == PrecisionType.BF16 and self.supports_bf16:
            return True
        if precision == PrecisionType.INT8 and self.supports_int8:
            return True
        
        return False
    
    def get_compute_capability_version(self) -> Tuple[int, int]:
        """解析计算能力版本"""
        if not self.compute_capability:
            return (0, 0)
        
        try:
            parts = self.compute_capability.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        except (ValueError, IndexError):
            return (0, 0)
    
    def is_modern_gpu(self) -> bool:
        """是否是现代GPU（计算能力 >= 7.0）"""
        major, _ = self.get_compute_capability_version()
        return major >= 7
    
    def get_effective_bandwidth(self) -> float:
        """获取有效带宽（考虑ECC等因素）"""
        if self.peak_memory_bandwidth == 0:
            return 0.0
        # 假设ECC和其他因素导致约10%的带宽损失
        return self.peak_memory_bandwidth * 0.9
    
    def estimate_optimal_batch_size(self, model_size_mb: float) -> int:
        """根据设备能力估算最优batch size"""
        if self.max_shared_memory_per_block == 0:
            return 32  # 默认值
        
        # 简化的启发式算法
        memory_per_sample = model_size_mb / 100  # 粗略估算
        if memory_per_sample > 0:
            batch_size = int(self.max_shared_memory_per_block / (memory_per_sample * 1024 * 1024))
            # 限制在合理范围内
            return max(1, min(batch_size, 512))
        return 32
    
    def get_performance_score(self) -> float:
        """获取性能评分（0-100）"""
        score = 0.0
        
        # 计算能力评分（30分）
        major, minor = self.get_compute_capability_version()
        if major >= 8:
            score += 30
        elif major >= 7:
            score += 25
        elif major >= 6:
            score += 20
        else:
            score += 10
        
        # 精度支持评分（20分）
        if self.supports_tensor_cores:
            score += 10
        if self.supports_bf16:
            score += 5
        if self.supports_fp16:
            score += 5
        
        # 内存评分（20分）
        if self.peak_memory_bandwidth > 500:
            score += 20
        elif self.peak_memory_bandwidth > 300:
            score += 15
        elif self.peak_memory_bandwidth > 100:
            score += 10
        else:
            score += 5
        
        # 算力评分（20分）
        if self.peak_flops_fp32 > 20:
            score += 20
        elif self.peak_flops_fp32 > 10:
            score += 15
        elif self.peak_flops_fp32 > 5:
            score += 10
        else:
            score += 5
        
        # 高级特性评分（10分）
        if self.supports_flash_attention:
            score += 5
        if self.supports_async_copy:
            score += 3
        if self.supports_cooperative_groups:
            score += 2
        
        return min(score, 100.0)


@dataclass
class DeviceInfo:
    """设备信息"""
    # 基本信息
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0
    name: str = "Unknown"
    
    # 内存信息（字节）
    total_memory: int = 0
    available_memory: int = 0
    used_memory: int = 0
    
    # 设备能力
    capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)
    
    # 状态
    is_available: bool = True
    is_busy: bool = False
    utilization: float = 0.0  # 0-100%
    temperature: float = 0.0  # 摄氏度
    
    # 额外信息
    driver_version: str = ""
    cuda_version: str = ""
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    # 新增：性能指标
    power_usage: float = 0.0     # 瓦特
    power_limit: float = 0.0     # 瓦特
    clock_speed: int = 0         # MHz
    memory_clock_speed: int = 0  # MHz
    
    # 新增：进程信息
    processes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 新增：健康状态
    error_count: int = 0
    last_error: str = ""
    
    @property
    def memory_usage_percent(self) -> float:
        """内存使用百分比"""
        if self.total_memory > 0:
            return (self.used_memory / self.total_memory) * 100
        return 0.0
    
    @property
    def available_memory_gb(self) -> float:
        """可用内存（GB）"""
        return self.available_memory / (1024 ** 3)
    
    @property
    def total_memory_gb(self) -> float:
        """总内存（GB）"""
        return self.total_memory / (1024 ** 3)
    
    @property
    def used_memory_gb(self) -> float:
        """已用内存（GB）"""
        return self.used_memory / (1024 ** 3)
    
    @property
    def device_string(self) -> str:
        """PyTorch设备字符串"""
        if self.device_type == DeviceType.CPU:
            return "cpu"
        elif self.device_type == DeviceType.GPU:
            return f"cuda:{self.device_id}"
        elif self.device_type == DeviceType.MPS:
            return "mps"
        elif self.device_type == DeviceType.NPU:
            return f"npu:{self.device_id}"
        elif self.device_type == DeviceType.XPU:
            return f"xpu:{self.device_id}"
        else:
            return "cpu"
    
    @property
    def is_healthy(self) -> bool:
        """设备是否健康"""
        # 检查温度
        if self.temperature > 85:  # 温度过高
            return False
        # 检查错误计数
        if self.error_count > 10:
            return False
        # 检查功耗
        if self.power_limit > 0 and self.power_usage > self.power_limit * 1.1:
            return False
        return True
    
    @property
    def power_efficiency(self) -> float:
        """功耗效率（utilization/power_usage）"""
        if self.power_usage > 0:
            return self.utilization / self.power_usage
        return 0.0
    
    @property
    def memory_pressure(self) -> str:
        """内存压力等级"""
        usage = self.memory_usage_percent
        if usage < 50:
            return "low"
        elif usage < 75:
            return "medium"
        elif usage < 90:
            return "high"
        else:
            return "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device_type': self.device_type.value,
            'device_id': self.device_id,
            'name': self.name,
            'total_memory': self.total_memory,
            'available_memory': self.available_memory,
            'used_memory': self.used_memory,
            'memory_usage_percent': self.memory_usage_percent,
            'capabilities': self.capabilities.to_dict(),
            'is_available': self.is_available,
            'is_busy': self.is_busy,
            'utilization': self.utilization,
            'temperature': self.temperature,
            'driver_version': self.driver_version,
            'cuda_version': self.cuda_version,
            'power_usage': self.power_usage,
            'power_limit': self.power_limit,
            'clock_speed': self.clock_speed,
            'memory_clock_speed': self.memory_clock_speed,
            'processes': self.processes,
            'error_count': self.error_count,
            'is_healthy': self.is_healthy,
            'memory_pressure': self.memory_pressure
        }
    
    def __str__(self) -> str:
        return (
            f"{self.device_type.value}:{self.device_id} ({self.name}) - "
            f"{self.available_memory_gb:.1f}/{self.total_memory_gb:.1f}GB "
            f"({self.memory_usage_percent:.1f}% used)"
        )
    
    def can_fit_model(self, model_size_bytes: int, safety_margin: float = 0.1) -> bool:
        """
        检查设备是否能容纳指定大小的模型
        
        Args:
            model_size_bytes: 模型大小（字节）
            safety_margin: 安全边际（0-1）
            
        Returns:
            是否可以容纳
        """
        required_memory = model_size_bytes * (1 + safety_margin)
        return self.available_memory >= required_memory
    
    def estimate_batch_size(
        self,
        model_size_bytes: int,
        sample_size_bytes: int,
        overhead_factor: float = 2.0
    ) -> int:
        """
        估算可支持的批次大小
        
        Args:
            model_size_bytes: 模型大小（字节）
            sample_size_bytes: 单个样本大小（字节）
            overhead_factor: 开销因子（考虑梯度、优化器状态等）
            
        Returns:
            估算的批次大小
        """
        available = self.available_memory
        model_memory = model_size_bytes * overhead_factor
        
        if available <= model_memory:
            return 1
        
        remaining = available - model_memory
        batch_size = int(remaining / sample_size_bytes)
        
        # 确保在合理范围内
        return max(1, min(batch_size, 1024))
    
    def get_suitability_score(
        self,
        required_memory_gb: float = 0,
        required_precision: Optional[PrecisionType] = None,
        prefer_low_temp: bool = True
    ) -> float:
        """
        获取设备适用性评分（0-100）
        
        Args:
            required_memory_gb: 所需内存（GB）
            required_precision: 所需精度
            prefer_low_temp: 是否偏好低温设备
            
        Returns:
            适用性评分
        """
        score = 0.0
        
        # 可用性检查（必要条件）
        if not self.is_available or not self.is_healthy:
            return 0.0
        
        # 内存评分（40分）
        if required_memory_gb > 0:
            if self.available_memory_gb >= required_memory_gb:
                memory_ratio = self.available_memory_gb / required_memory_gb
                score += min(40, memory_ratio * 20)
            else:
                return 0.0  # 内存不足，不适用
        else:
            score += 40 * (self.available_memory_gb / max(self.total_memory_gb, 1))
        
        # 精度支持评分（20分）
        if required_precision:
            if self.capabilities.supports_precision(required_precision):
                score += 20
            else:
                return 0.0  # 不支持所需精度
        else:
            score += 20
        
        # 利用率评分（15分）- 偏好低利用率设备
        score += 15 * (1 - self.utilization / 100)
        
        # 温度评分（15分）
        if prefer_low_temp:
            if self.temperature < 60:
                score += 15
            elif self.temperature < 75:
                score += 10
            elif self.temperature < 85:
                score += 5
        else:
            score += 15
        
        # 性能评分（10分）
        perf_score = self.capabilities.get_performance_score()
        score += perf_score * 0.1
        
        return min(score, 100.0)
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断设备状态
        
        Returns:
            诊断信息字典
        """
        issues = []
        warnings = []
        recommendations = []
        
        # 检查可用性
        if not self.is_available:
            issues.append("Device is not available")
            return {
                'status': 'unavailable',
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
        
        # 检查健康状态
        if not self.is_healthy:
            issues.append("Device health check failed")
        
        # 检查温度
        if self.temperature > 85:
            issues.append(f"Temperature too high: {self.temperature}°C")
            recommendations.append("Improve cooling or reduce workload")
        elif self.temperature > 75:
            warnings.append(f"Temperature elevated: {self.temperature}°C")
        
        # 检查内存
        mem_usage = self.memory_usage_percent
        if mem_usage > 95:
            issues.append(f"Memory critically low: {mem_usage:.1f}% used")
            recommendations.append("Reduce batch size or model size")
        elif mem_usage > 85:
            warnings.append(f"Memory usage high: {mem_usage:.1f}% used")
            recommendations.append("Consider reducing memory usage")
        
        # 检查利用率
        if self.utilization > 95:
            warnings.append(f"High utilization: {self.utilization:.1f}%")
        elif self.utilization < 10 and self.is_busy:
            warnings.append(f"Low utilization: {self.utilization:.1f}%")
            recommendations.append("Check for bottlenecks in data loading or preprocessing")
        
        # 检查功耗
        if self.power_limit > 0 and self.power_usage > self.power_limit * 0.95:
            warnings.append(f"Power usage near limit: {self.power_usage:.1f}W / {self.power_limit:.1f}W")
        
        # 检查错误
        if self.error_count > 0:
            warnings.append(f"Device has {self.error_count} errors")
            if self.last_error:
                warnings.append(f"Last error: {self.last_error}")
        
        # 确定状态
        if issues:
            status = 'unhealthy'
        elif warnings:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'health_score': self.get_suitability_score()
        }
    
    def compare_to(self, other: 'DeviceInfo') -> Dict[str, Any]:
        """
        与另一个设备比较
        
        Args:
            other: 另一个设备
            
        Returns:
            比较结果
        """
        return {
            'memory_advantage': self.total_memory - other.total_memory,
            'available_memory_advantage': self.available_memory - other.available_memory,
            'utilization_diff': self.utilization - other.utilization,
            'temperature_diff': self.temperature - other.temperature,
            'performance_score_diff': (
                self.capabilities.get_performance_score() - 
                other.capabilities.get_performance_score()
            ),
            'is_better': self.get_suitability_score() > other.get_suitability_score()
        }


@dataclass
class HardwareConfig:
    """硬件配置"""
    # 设备选择
    device_type: DeviceType = DeviceType.GPU
    device_ids: List[int] = field(default_factory=lambda: [0])
    
    # 精度设置
    precision: PrecisionType = PrecisionType.FP32
    enable_amp: bool = True  # 自动混合精度
    
    # 内存设置
    max_memory_gb: Optional[float] = None  # 最大内存使用
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True
    
    # 并行设置
    num_workers: int = 4  # 数据加载线程数
    pin_memory: bool = True
    
    # 分布式设置
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # 优化设置
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    
    # 新增：编译设置
    torch_compile: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    
    # 新增：性能设置
    tf32_enabled: bool = True
    allow_tf32_cublas: bool = True
    allow_tf32_cudnn: bool = True
    
    # 新增：调试设置
    detect_anomaly: bool = False
    profile_memory: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device_type': self.device_type.value,
            'device_ids': self.device_ids,
            'precision': self.precision.value,
            'enable_amp': self.enable_amp,
            'max_memory_gb': self.max_memory_gb,
            'gradient_checkpointing': self.gradient_checkpointing,
            'memory_efficient_attention': self.memory_efficient_attention,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'distributed': self.distributed,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'cudnn_benchmark': self.cudnn_benchmark,
            'cudnn_deterministic': self.cudnn_deterministic,
            'torch_compile': self.torch_compile,
            'compile_mode': self.compile_mode,
            'tf32_enabled': self.tf32_enabled,
            'allow_tf32_cublas': self.allow_tf32_cublas,
            'allow_tf32_cudnn': self.allow_tf32_cudnn,
            'detect_anomaly': self.detect_anomaly,
            'profile_memory': self.profile_memory,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareConfig':
        """从字典创建"""
        return cls(
            device_type=DeviceType(data.get('device_type', 'gpu')),
            device_ids=data.get('device_ids', [0]),
            precision=PrecisionType(data.get('precision', 'fp32')),
            enable_amp=data.get('enable_amp', True),
            max_memory_gb=data.get('max_memory_gb'),
            gradient_checkpointing=data.get('gradient_checkpointing', False),
            memory_efficient_attention=data.get('memory_efficient_attention', True),
            num_workers=data.get('num_workers', 4),
            pin_memory=data.get('pin_memory', True),
            distributed=data.get('distributed', False),
            world_size=data.get('world_size', 1),
            rank=data.get('rank', 0),
            local_rank=data.get('local_rank', 0),
            cudnn_benchmark=data.get('cudnn_benchmark', True),
            cudnn_deterministic=data.get('cudnn_deterministic', False),
            torch_compile=data.get('torch_compile', False),
            compile_mode=data.get('compile_mode', 'default'),
            tf32_enabled=data.get('tf32_enabled', True),
            allow_tf32_cublas=data.get('allow_tf32_cublas', True),
            allow_tf32_cudnn=data.get('allow_tf32_cudnn', True),
            detect_anomaly=data.get('detect_anomaly', False),
            profile_memory=data.get('profile_memory', False),
        )
    
    @classmethod
    def auto_detect(cls) -> 'HardwareConfig':
        """自动检测并创建配置"""
        import torch
        
        config = cls()
        
        # 检测设备
        if torch.cuda.is_available():
            config.device_type = DeviceType.GPU
            config.device_ids = list(range(torch.cuda.device_count()))
            
            # 检测精度支持
            if torch.cuda.is_bf16_supported():
                config.precision = PrecisionType.BF16
            else:
                config.precision = PrecisionType.FP16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config.device_type = DeviceType.MPS
            config.device_ids = [0]
        else:
            config.device_type = DeviceType.CPU
            config.device_ids = [0]
            config.enable_amp = False
        
        return config
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误消息列表，如果为空则配置有效
        """
        errors = []
        
        # 验证设备ID
        if not self.device_ids:
            errors.append("device_ids cannot be empty")
        
        for device_id in self.device_ids:
            if device_id < 0:
                errors.append(f"Invalid device_id: {device_id}")
        
        # 验证并行设置
        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")
        
        # 验证分布式设置
        if self.distributed:
            if self.world_size < 1:
                errors.append(f"world_size must be at least 1, got {self.world_size}")
            if self.rank < 0 or self.rank >= self.world_size:
                errors.append(f"rank must be in [0, world_size), got {self.rank}")
            if self.local_rank < 0:
                errors.append(f"local_rank must be non-negative, got {self.local_rank}")
        
        # 验证内存设置
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            errors.append(f"max_memory_gb must be positive, got {self.max_memory_gb}")
        
        # 验证编译模式
        valid_compile_modes = ['default', 'reduce-overhead', 'max-autotune']
        if self.compile_mode not in valid_compile_modes:
            errors.append(f"Invalid compile_mode: {self.compile_mode}, must be one of {valid_compile_modes}")
        
        # 验证互斥设置
        if self.cudnn_benchmark and self.cudnn_deterministic:
            errors.append("cudnn_benchmark and cudnn_deterministic cannot both be True")
        
        return errors
    
    def apply(self) -> None:
        """应用配置到PyTorch"""
        import torch
        
        # 设置随机种子（如果需要确定性）
        if self.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
        
        # 设置TF32
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32_cublas
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = self.allow_tf32_cudnn
        
        # 设置异常检测
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        logger.info(f"Applied hardware config: {self.device_type.value}, precision={self.precision.value}")
    
    def get_recommended_batch_size(
        self,
        model_size_mb: float,
        sample_size_mb: float = 1.0
    ) -> int:
        """
        获取推荐的批次大小
        
        Args:
            model_size_mb: 模型大小（MB）
            sample_size_mb: 单个样本大小（MB）
            
        Returns:
            推荐的批次大小
        """
        if self.max_memory_gb:
            available_memory_mb = self.max_memory_gb * 1024
        else:
            # 假设有8GB可用内存（保守估计）
            available_memory_mb = 8 * 1024
        
        # 考虑梯度和优化器状态（约3x模型大小）
        model_overhead = model_size_mb * 3
        if self.gradient_checkpointing:
            model_overhead *= 0.7  # 梯度检查点可以节约约30%内存
        
        remaining_memory = available_memory_mb - model_overhead
        if remaining_memory <= 0:
            return 1
        
        batch_size = int(remaining_memory / sample_size_mb)
        
        # 限制在合理范围
        return max(1, min(batch_size, 512))
    
    def optimize_for_task(self, task_type: str = "training") -> None:
        """
        根据任务类型优化配置
        
        Args:
            task_type: 任务类型（training, inference, fine_tuning）
        """
        if task_type == "training":
            self.gradient_checkpointing = True
            self.cudnn_benchmark = True
            self.enable_amp = True
        elif task_type == "inference":
            self.gradient_checkpointing = False
            self.cudnn_benchmark = True
            self.torch_compile = True
            self.pin_memory = False
        elif task_type == "fine_tuning":
            self.gradient_checkpointing = True
            self.memory_efficient_attention = True
            self.enable_amp = True
        
        logger.info(f"Optimized config for {task_type}")
    
    def estimate_memory_usage(self, model_size_mb: float) -> Dict[str, float]:
        """
        估算内存使用
        
        Args:
            model_size_mb: 模型大小（MB）
            
        Returns:
            内存使用估算（MB）
        """
        usage = {
            'model': model_size_mb,
            'gradients': model_size_mb if not self.gradient_checkpointing else model_size_mb * 0.3,
            'optimizer': model_size_mb * 2,  # Adam需要2x
            'activations': model_size_mb * 0.5,
            'overhead': model_size_mb * 0.2
        }
        
        # 精度调整
        multiplier = self.precision.get_memory_multiplier()
        for key in usage:
            usage[key] *= multiplier
        
        usage['total'] = sum(usage.values())
        return usage
    
    def compare_to(self, other: 'HardwareConfig') -> Dict[str, Any]:
        """
        与另一个配置比较
        
        Args:
            other: 另一个配置
            
        Returns:
            比较结果
        """
        return {
            'device_type_same': self.device_type == other.device_type,
            'precision_diff': f"{self.precision.value} vs {other.precision.value}",
            'amp_diff': self.enable_amp != other.enable_amp,
            'memory_optimization_level': (
                int(self.gradient_checkpointing) + 
                int(self.memory_efficient_attention)
            ) - (
                int(other.gradient_checkpointing) + 
                int(other.memory_efficient_attention)
            ),
            'is_more_optimized': (
                self.cudnn_benchmark and 
                self.memory_efficient_attention and 
                not self.detect_anomaly
            )
        }


# ==================== 设备选择和比较工具 ====================

class DeviceSelector:
    """设备选择器"""
    
    def __init__(self, devices: List[DeviceInfo]):
        """
        初始化设备选择器
        
        Args:
            devices: 设备列表
        """
        self.devices = devices
    
    def select_best(
        self,
        required_memory_gb: float = 0,
        required_precision: Optional[PrecisionType] = None,
        prefer_low_utilization: bool = True,
        prefer_low_temperature: bool = True
    ) -> Optional[DeviceInfo]:
        """
        选择最佳设备
        
        Args:
            required_memory_gb: 所需内存（GB）
            required_precision: 所需精度
            prefer_low_utilization: 偏好低利用率
            prefer_low_temperature: 偏好低温度
            
        Returns:
            最佳设备，如果没有合适的设备则返回None
        """
        if not self.devices:
            return None
        
        # 过滤可用设备
        available_devices = [d for d in self.devices if d.is_available and d.is_healthy]
        
        if not available_devices:
            return None
        
        # 计算每个设备的评分
        scored_devices = []
        for device in available_devices:
            score = device.get_suitability_score(
                required_memory_gb=required_memory_gb,
                required_precision=required_precision,
                prefer_low_temp=prefer_low_temperature
            )
            if score > 0:
                scored_devices.append((device, score))
        
        if not scored_devices:
            return None
        
        # 返回评分最高的设备
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return scored_devices[0][0]
    
    def select_multiple(
        self,
        count: int,
        required_memory_gb: float = 0,
        balance_load: bool = True
    ) -> List[DeviceInfo]:
        """
        选择多个设备
        
        Args:
            count: 需要的设备数量
            required_memory_gb: 每个设备所需内存（GB）
            balance_load: 是否均衡负载
            
        Returns:
            选中的设备列表
        """
        available_devices = [d for d in self.devices if d.is_available and d.is_healthy]
        
        if not available_devices:
            return []
        
        # 过滤满足内存要求的设备
        if required_memory_gb > 0:
            available_devices = [
                d for d in available_devices 
                if d.available_memory_gb >= required_memory_gb
            ]
        
        if not available_devices:
            return []
        
        # 根据策略选择
        if balance_load:
            # 按利用率排序，选择最空闲的设备
            available_devices.sort(key=lambda d: (d.utilization, d.temperature))
        else:
            # 按性能评分排序
            available_devices.sort(
                key=lambda d: d.capabilities.get_performance_score(),
                reverse=True
            )
        
        return available_devices[:count]
    
    def group_by_type(self) -> Dict[DeviceType, List[DeviceInfo]]:
        """按设备类型分组"""
        groups: Dict[DeviceType, List[DeviceInfo]] = {}
        for device in self.devices:
            if device.device_type not in groups:
                groups[device.device_type] = []
            groups[device.device_type].append(device)
        return groups
    
    def get_total_memory(self) -> int:
        """获取所有设备的总内存"""
        return sum(d.total_memory for d in self.devices)
    
    def get_available_memory(self) -> int:
        """获取所有设备的可用内存"""
        return sum(d.available_memory for d in self.devices if d.is_available)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        available_devices = [d for d in self.devices if d.is_available]
        
        if not available_devices:
            return {
                'total_devices': len(self.devices),
                'available_devices': 0
            }
        
        return {
            'total_devices': len(self.devices),
            'available_devices': len(available_devices),
            'total_memory_gb': self.get_total_memory() / (1024 ** 3),
            'available_memory_gb': self.get_available_memory() / (1024 ** 3),
            'avg_utilization': sum(d.utilization for d in available_devices) / len(available_devices),
            'avg_temperature': sum(d.temperature for d in available_devices) / len(available_devices),
            'device_types': list(self.group_by_type().keys())
        }


class DeviceComparator:
    """设备比较器"""
    
    @staticmethod
    def compare_devices(devices: List[DeviceInfo]) -> List[Dict[str, Any]]:
        """
        比较多个设备
        
        Args:
            devices: 设备列表
            
        Returns:
            比较结果列表
        """
        if not devices:
            return []
        
        results = []
        for device in devices:
            result = {
                'device': f"{device.device_type.value}:{device.device_id}",
                'name': device.name,
                'memory_gb': device.total_memory_gb,
                'available_memory_gb': device.available_memory_gb,
                'utilization': device.utilization,
                'temperature': device.temperature,
                'performance_score': device.capabilities.get_performance_score(),
                'suitability_score': device.get_suitability_score(),
                'is_healthy': device.is_healthy
            }
            results.append(result)
        
        return results
    
    @staticmethod
    def rank_devices(
        devices: List[DeviceInfo],
        criteria: str = "performance"
    ) -> List[Tuple[DeviceInfo, float]]:
        """
        对设备进行排名
        
        Args:
            devices: 设备列表
            criteria: 排名标准（performance, memory, efficiency, suitability）
            
        Returns:
            排名后的设备列表（设备，评分）
        """
        scored_devices = []
        
        for device in devices:
            if criteria == "performance":
                score = device.capabilities.get_performance_score()
            elif criteria == "memory":
                score = device.available_memory_gb
            elif criteria == "efficiency":
                score = device.power_efficiency if device.power_usage > 0 else 0
            elif criteria == "suitability":
                score = device.get_suitability_score()
            else:
                score = 0
            
            scored_devices.append((device, score))
        
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        return scored_devices
    
    @staticmethod
    def find_bottleneck(devices: List[DeviceInfo]) -> Dict[str, Any]:
        """
        找出瓶颈设备
        
        Args:
            devices: 设备列表
            
        Returns:
            瓶颈分析结果
        """
        if not devices:
            return {'bottleneck': None, 'reason': 'No devices'}
        
        bottlenecks = []
        
        for device in devices:
            issues = []
            
            # 检查内存
            if device.memory_usage_percent > 90:
                issues.append('high_memory_usage')
            
            # 检查利用率
            if device.utilization > 95:
                issues.append('high_utilization')
            elif device.utilization < 10 and device.is_busy:
                issues.append('low_utilization')
            
            # 检查温度
            if device.temperature > 85:
                issues.append('high_temperature')
            
            # 检查健康状态
            if not device.is_healthy:
                issues.append('unhealthy')
            
            if issues:
                bottlenecks.append({
                    'device': f"{device.device_type.value}:{device.device_id}",
                    'issues': issues,
                    'severity': len(issues)
                })
        
        if not bottlenecks:
            return {'bottleneck': None, 'reason': 'No bottlenecks detected'}
        
        # 返回最严重的瓶颈
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        return {'bottleneck': bottlenecks[0], 'all_issues': bottlenecks}


# ==================== 工具函数 ====================

def parse_device_string(device_str: str) -> Tuple[DeviceType, int]:
    """
    解析设备字符串
    
    Args:
        device_str: 设备字符串，如 "cuda:0", "cpu", "npu:1"
        
    Returns:
        (设备类型, 设备ID)
    """
    device_str = device_str.lower().strip()
    
    if ':' in device_str:
        type_str, id_str = device_str.split(':', 1)
        try:
            device_id = int(id_str)
        except ValueError:
            device_id = 0
    else:
        type_str = device_str
        device_id = 0
    
    # 映射
    type_map = {
        'cuda': DeviceType.GPU,
        'gpu': DeviceType.GPU,
        'cpu': DeviceType.CPU,
        'npu': DeviceType.NPU,
        'tpu': DeviceType.TPU,
        'mps': DeviceType.MPS,
        'xpu': DeviceType.XPU,
        'rocm': DeviceType.ROCM
    }
    
    device_type = type_map.get(type_str, DeviceType.UNKNOWN)
    return device_type, device_id


def create_device_string(device_type: DeviceType, device_id: int = 0) -> str:
    """
    创建设备字符串
    
    Args:
        device_type: 设备类型
        device_id: 设备ID
        
    Returns:
        设备字符串
    """
    if device_type == DeviceType.CPU:
        return "cpu"
    elif device_type == DeviceType.GPU:
        return f"cuda:{device_id}"
    elif device_type == DeviceType.MPS:
        return "mps"
    elif device_type == DeviceType.NPU:
        return f"npu:{device_id}"
    elif device_type == DeviceType.XPU:
        return f"xpu:{device_id}"
    elif device_type == DeviceType.ROCM:
        return f"rocm:{device_id}"
    else:
        return "cpu"


def estimate_model_memory(
    num_parameters: int,
    precision: PrecisionType = PrecisionType.FP32,
    include_gradients: bool = True,
    include_optimizer: bool = True
) -> Dict[str, float]:
    """
    估算模型内存需求
    
    Args:
        num_parameters: 参数数量
        precision: 精度类型
        include_gradients: 是否包含梯度
        include_optimizer: 是否包含优化器状态
        
    Returns:
        内存需求（MB）
    """
    bytes_per_param = precision.bytes_per_element
    
    model_memory_mb = (num_parameters * bytes_per_param) / (1024 ** 2)
    
    memory = {
        'model': model_memory_mb
    }
    
    if include_gradients:
        memory['gradients'] = model_memory_mb
    
    if include_optimizer:
        # Adam需要2个state（momentum和variance）
        memory['optimizer_states'] = model_memory_mb * 2
    
    memory['total'] = sum(memory.values())
    
    return memory


def recommend_precision(
    device_capabilities: DeviceCapabilities,
    task_type: str = "training"
) -> PrecisionType:
    """
    推荐精度类型
    
    Args:
        device_capabilities: 设备能力
        task_type: 任务类型
        
    Returns:
        推荐的精度类型
    """
    # 训练任务
    if task_type == "training":
        if device_capabilities.supports_bf16:
            return PrecisionType.BF16
        elif device_capabilities.supports_fp16:
            return PrecisionType.FP16
        else:
            return PrecisionType.FP32
    
    # 推理任务
    elif task_type == "inference":
        if device_capabilities.supports_int8:
            return PrecisionType.INT8
        elif device_capabilities.supports_fp16:
            return PrecisionType.FP16
        else:
            return PrecisionType.FP32
    
    # 默认
    return PrecisionType.FP32


def validate_device_config(
    config: HardwareConfig,
    available_devices: List[DeviceInfo]
) -> List[str]:
    """
    验证设备配置
    
    Args:
        config: 硬件配置
        available_devices: 可用设备列表
        
    Returns:
        错误消息列表
    """
    errors = config.validate()
    
    # 检查设备可用性
    device_ids_set = set(config.device_ids)
    available_ids = set(d.device_id for d in available_devices if d.device_type == config.device_type)
    
    missing_devices = device_ids_set - available_ids
    if missing_devices:
        errors.append(f"Devices not available: {missing_devices}")
    
    # 检查精度支持
    for device_id in config.device_ids:
        device = next((d for d in available_devices if d.device_id == device_id and d.device_type == config.device_type), None)
        if device:
            if not device.capabilities.supports_precision(config.precision):
                errors.append(f"Device {device_id} does not support precision {config.precision.value}")
    
    return errors


def print_device_comparison(devices: List[DeviceInfo]) -> None:
    """
    打印设备比较表
    
    Args:
        devices: 设备列表
    """
    if not devices:
        print("No devices to compare")
        return
    
    print("\n" + "="*100)
    print("Device Comparison")
    print("="*100)
    print(f"{'Device':<15} {'Name':<25} {'Memory (GB)':<15} {'Util%':<8} {'Temp°C':<8} {'Score':<8} {'Status':<10}")
    print("-"*100)
    
    for device in devices:
        status = "✓" if device.is_healthy else "✗"
        print(
            f"{device.device_string:<15} "
            f"{device.name[:24]:<25} "
            f"{device.available_memory_gb:.1f}/{device.total_memory_gb:.1f}    "
            f"{device.utilization:<7.1f} "
            f"{device.temperature:<7.1f} "
            f"{device.get_suitability_score():<7.1f} "
            f"{status:<10}"
        )
    
    print("="*100)


def get_optimal_device_allocation(
    devices: List[DeviceInfo],
    num_tasks: int,
    memory_per_task_gb: float
) -> Dict[int, List[int]]:
    """
    获取最优设备分配方案
    
    Args:
        devices: 可用设备列表
        num_tasks: 任务数量
        memory_per_task_gb: 每个任务所需内存（GB）
        
    Returns:
        设备ID到任务ID列表的映射
    """
    # 过滤可用设备
    available_devices = [
        d for d in devices 
        if d.is_available and d.is_healthy and d.available_memory_gb >= memory_per_task_gb
    ]
    
    if not available_devices:
        return {}
    
    # 按可用内存排序
    available_devices.sort(key=lambda d: d.available_memory_gb, reverse=True)
    
    allocation: Dict[int, List[int]] = {}
    task_id = 0
    
    # 贪心分配
    while task_id < num_tasks and available_devices:
        for device in available_devices:
            # 检查设备是否还有足够内存
            tasks_on_device = len(allocation.get(device.device_id, []))
            required_memory = (tasks_on_device + 1) * memory_per_task_gb
            
            if device.available_memory_gb >= required_memory:
                if device.device_id not in allocation:
                    allocation[device.device_id] = []
                allocation[device.device_id].append(task_id)
                task_id += 1
                
                if task_id >= num_tasks:
                    break
        
        # 如果还有任务但没有设备可分配，结束
        if task_id < num_tasks:
            logger.warning(f"Could only allocate {task_id} out of {num_tasks} tasks")
            break
    
    return allocation

