# -*- coding: utf-8 -*-
"""
设备管理器

统一管理各种硬件设备的检测和使用。
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from .device_types import DeviceType, DeviceInfo, DeviceCapabilities, HardwareConfig

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """设备状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


class AllocationStrategy(Enum):
    """设备分配策略"""
    LEAST_LOADED = "least_loaded"  # 负载最低
    MOST_MEMORY = "most_memory"    # 内存最多
    ROUND_ROBIN = "round_robin"    # 轮询
    BALANCED = "balanced"          # 平衡


@dataclass
class DeviceMetrics:
    """设备指标"""
    device_id: str
    timestamp: float = field(default_factory=time.time)
    
    # 内存指标
    memory_used: int = 0
    memory_total: int = 0
    memory_percent: float = 0.0
    
    # 性能指标
    utilization: float = 0.0
    temperature: float = 0.0
    power_usage: float = 0.0
    
    # 操作计数
    allocation_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'timestamp': self.timestamp,
            'memory_used': self.memory_used,
            'memory_total': self.memory_total,
            'memory_percent': self.memory_percent,
            'utilization': self.utilization,
            'temperature': self.temperature,
            'power_usage': self.power_usage,
            'allocation_count': self.allocation_count,
            'error_count': self.error_count,
        }


class DeviceMonitor:
    """设备监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics_history: Dict[str, List[DeviceMetrics]] = defaultdict(list)
        self._device_status: Dict[str, DeviceStatus] = {}
        
    def record_metrics(self, metrics: DeviceMetrics) -> None:
        """记录设备指标"""
        device_id = metrics.device_id
        self._metrics_history[device_id].append(metrics)
        
        # 限制历史长度
        if len(self._metrics_history[device_id]) > self.max_history:
            self._metrics_history[device_id].pop(0)
        
        # 更新设备状态
        self._update_device_status(metrics)
    
    def _update_device_status(self, metrics: DeviceMetrics) -> None:
        """更新设备状态"""
        device_id = metrics.device_id
        
        # 检查错误
        if metrics.error_count > 10:
            self._device_status[device_id] = DeviceStatus.ERROR
        # 检查温度
        elif metrics.temperature > 85:
            self._device_status[device_id] = DeviceStatus.WARNING
        # 检查内存
        elif metrics.memory_percent > 95:
            self._device_status[device_id] = DeviceStatus.WARNING
        else:
            self._device_status[device_id] = DeviceStatus.HEALTHY
    
    def get_device_status(self, device_id: str) -> DeviceStatus:
        """获取设备状态"""
        return self._device_status.get(device_id, DeviceStatus.HEALTHY)
    
    def get_latest_metrics(self, device_id: str) -> Optional[DeviceMetrics]:
        """获取最新指标"""
        history = self._metrics_history.get(device_id, [])
        return history[-1] if history else None
    
    def get_average_metrics(self, device_id: str, window: int = 100) -> Dict[str, float]:
        """获取平均指标"""
        history = self._metrics_history.get(device_id, [])
        if not history:
            return {}
        
        recent = history[-window:]
        
        return {
            'avg_memory_percent': sum(m.memory_percent for m in recent) / len(recent),
            'avg_utilization': sum(m.utilization for m in recent) / len(recent),
            'avg_temperature': sum(m.temperature for m in recent) / len(recent),
            'total_allocations': sum(m.allocation_count for m in recent),
            'total_errors': sum(m.error_count for m in recent),
        }
    
    def get_all_status(self) -> Dict[str, DeviceStatus]:
        """获取所有设备状态"""
        return self._device_status.copy()
    
    def reset(self, device_id: Optional[str] = None) -> None:
        """重置监控数据"""
        if device_id:
            self._metrics_history.pop(device_id, None)
            self._device_status.pop(device_id, None)
        else:
            self._metrics_history.clear()
            self._device_status.clear()


class DeviceBenchmark:
    """设备性能基准测试"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._results: Dict[str, Any] = {}
    
    def run_compute_benchmark(self, size: int = 4096, iterations: int = 100) -> float:
        """
        运行计算基准测试
        
        Args:
            size: 矩阵大小
            iterations: 迭代次数
            
        Returns:
            平均时间（秒）
        """
        # 生成测试数据
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        # 预热
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        
        # 测试
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        # 计算GFLOPS
        flops = 2 * size ** 3  # 矩阵乘法的浮点运算数
        gflops = (flops / avg_time) / 1e9
        
        self._results['compute'] = {
            'avg_time': avg_time,
            'gflops': gflops,
            'size': size,
            'iterations': iterations
        }
        
        return avg_time
    
    def run_memory_benchmark(self, size_mb: int = 1024) -> Tuple[float, float]:
        """
        运行内存带宽基准测试
        
        Args:
            size_mb: 测试数据大小（MB）
            
        Returns:
            (读取带宽 GB/s, 写入带宽 GB/s)
        """
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32
        
        # 写入测试
        start_time = time.time()
        data = torch.randn(num_elements, device=self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        write_time = time.time() - start_time
        write_bandwidth = size_bytes / write_time / 1e9
        
        # 读取测试
        start_time = time.time()
        _ = data.sum()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        read_time = time.time() - start_time
        read_bandwidth = size_bytes / read_time / 1e9
        
        self._results['memory'] = {
            'read_bandwidth_gbs': read_bandwidth,
            'write_bandwidth_gbs': write_bandwidth,
            'size_mb': size_mb
        }
        
        return read_bandwidth, write_bandwidth
    
    def run_transfer_benchmark(self, size_mb: int = 100) -> Tuple[float, float]:
        """
        运行数据传输基准测试
        
        Args:
            size_mb: 测试数据大小（MB）
            
        Returns:
            (Host->Device 带宽 GB/s, Device->Host 带宽 GB/s)
        """
        if self.device.type != 'cuda':
            return 0.0, 0.0
        
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4
        
        # Host to Device
        cpu_data = torch.randn(num_elements)
        start_time = time.time()
        gpu_data = cpu_data.to(self.device)
        torch.cuda.synchronize(self.device)
        h2d_time = time.time() - start_time
        h2d_bandwidth = size_bytes / h2d_time / 1e9
        
        # Device to Host
        start_time = time.time()
        _ = gpu_data.cpu()
        torch.cuda.synchronize(self.device)
        d2h_time = time.time() - start_time
        d2h_bandwidth = size_bytes / d2h_time / 1e9
        
        self._results['transfer'] = {
            'h2d_bandwidth_gbs': h2d_bandwidth,
            'd2h_bandwidth_gbs': d2h_bandwidth,
            'size_mb': size_mb
        }
        
        return h2d_bandwidth, d2h_bandwidth
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        logger.info(f"Running benchmarks on {self.device}")
        
        self.run_compute_benchmark()
        self.run_memory_benchmark()
        self.run_transfer_benchmark()
        
        return self._results
    
    def get_results(self) -> Dict[str, Any]:
        """获取基准测试结果"""
        return self._results.copy()
    
    def print_results(self) -> None:
        """打印基准测试结果"""
        if not self._results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print(f"Benchmark Results for {self.device}")
        print("="*80)
        
        if 'compute' in self._results:
            compute = self._results['compute']
            print(f"\nCompute Performance:")
            print(f"  Average time: {compute['avg_time']*1000:.2f} ms")
            print(f"  Performance: {compute['gflops']:.1f} GFLOPS")
        
        if 'memory' in self._results:
            memory = self._results['memory']
            print(f"\nMemory Bandwidth:")
            print(f"  Read: {memory['read_bandwidth_gbs']:.1f} GB/s")
            print(f"  Write: {memory['write_bandwidth_gbs']:.1f} GB/s")
        
        if 'transfer' in self._results:
            transfer = self._results['transfer']
            print(f"\nData Transfer:")
            print(f"  Host->Device: {transfer['h2d_bandwidth_gbs']:.1f} GB/s")
            print(f"  Device->Host: {transfer['d2h_bandwidth_gbs']:.1f} GB/s")
        
        print("="*80)


class DeviceManager:
    """
    设备管理器
    
    负责硬件设备的检测、选择和管理。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        self._devices: Dict[str, DeviceInfo] = {}
        self._primary_device: Optional[torch.device] = None
        self._config = HardwareConfig()
        
        # 新增：监控和管理
        self._monitor = DeviceMonitor()
        self._benchmarks: Dict[str, DeviceBenchmark] = {}
        self._allocation_strategy = AllocationStrategy.LEAST_LOADED
        self._allocation_counts: Dict[str, int] = defaultdict(int)
        self._round_robin_index = 0
        
        self._detect_all_devices()
        self._initialized = True
    
    def _detect_all_devices(self):
        """检测所有可用设备"""
        # CPU始终可用
        self._devices['cpu'] = self._detect_cpu()
        
        # 检测GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_info = self._detect_cuda_gpu(i)
                self._devices[f'cuda:{i}'] = device_info
        
        # 检测MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._devices['mps'] = self._detect_mps()
        
        # 设置主设备
        self._primary_device = self._select_best_device()
        
        logger.info(f"Detected {len(self._devices)} devices, primary: {self._primary_device}")
    
    def _detect_cpu(self) -> DeviceInfo:
        """检测CPU信息"""
        import os
        import platform
        
        # 获取CPU信息
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            total_memory = mem_info.total
            available_memory = mem_info.available
            used_memory = mem_info.used
        except ImportError:
            total_memory = 0
            available_memory = 0
            used_memory = 0
        
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name=platform.processor() or "CPU",
            total_memory=total_memory,
            available_memory=available_memory,
            used_memory=used_memory,
            capabilities=DeviceCapabilities(
                supported_precisions=[],
                supports_fp16=False,
                supports_bf16=False
            ),
            is_available=True
        )
    
    def _detect_cuda_gpu(self, device_id: int) -> DeviceInfo:
        """检测CUDA GPU信息"""
        props = torch.cuda.get_device_properties(device_id)
        
        # 获取内存信息
        torch.cuda.set_device(device_id)
        total_memory = props.total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        allocated_memory = torch.cuda.memory_allocated(device_id)
        available_memory = total_memory - reserved_memory
        
        # 检测能力
        compute_cap = f"{props.major}.{props.minor}"
        supports_fp16 = props.major >= 7  # Volta及以上
        supports_bf16 = props.major >= 8  # Ampere及以上
        supports_tensor_cores = props.major >= 7
        
        capabilities = DeviceCapabilities(
            compute_capability=compute_cap,
            supported_precisions=[],
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_int8=props.major >= 7,
            supports_tensor_cores=supports_tensor_cores,
            supports_flash_attention=props.major >= 8,
            max_threads_per_block=props.max_threads_per_block,
            max_shared_memory_per_block=props.max_shared_memory_per_block,
            warp_size=props.warp_size
        )
        
        return DeviceInfo(
            device_type=DeviceType.GPU,
            device_id=device_id,
            name=props.name,
            total_memory=total_memory,
            available_memory=available_memory,
            used_memory=allocated_memory,
            capabilities=capabilities,
            is_available=True,
            cuda_version=torch.version.cuda or ""
        )
    
    def _detect_mps(self) -> DeviceInfo:
        """检测Apple MPS信息"""
        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_id=0,
            name="Apple Metal",
            is_available=True,
            capabilities=DeviceCapabilities(
                supports_fp16=True,
                supports_bf16=False
            )
        )
    
    def _select_best_device(self) -> torch.device:
        """选择最佳设备"""
        # 优先选择GPU
        for key, info in self._devices.items():
            if info.device_type == DeviceType.GPU and info.is_available:
                return torch.device(key)
        
        # 其次MPS
        if 'mps' in self._devices and self._devices['mps'].is_available:
            return torch.device('mps')
        
        # 默认CPU
        return torch.device('cpu')
    
    @property
    def device(self) -> torch.device:
        """获取主设备"""
        return self._primary_device
    
    @property
    def device_info(self) -> DeviceInfo:
        """获取主设备信息"""
        return self._devices.get(str(self._primary_device), self._devices['cpu'])
    
    def get_device(self, device_str: Optional[str] = None) -> torch.device:
        """
        获取设备
        
        Args:
            device_str: 设备字符串，如 'cuda:0', 'cpu', None表示自动选择
        """
        if device_str is None:
            return self._primary_device
        
        return torch.device(device_str)
    
    def get_device_info(self, device: Optional[Union[str, torch.device]] = None) -> DeviceInfo:
        """获取设备信息"""
        if device is None:
            return self.device_info
        
        key = str(device) if isinstance(device, torch.device) else device
        return self._devices.get(key, self._devices['cpu'])
    
    def list_devices(self) -> List[DeviceInfo]:
        """列出所有设备"""
        return list(self._devices.values())
    
    def get_gpu_devices(self) -> List[DeviceInfo]:
        """获取所有GPU设备"""
        return [
            info for info in self._devices.values()
            if info.device_type == DeviceType.GPU
        ]
    
    def get_memory_info(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
        """获取设备内存信息"""
        info = self.get_device_info(device)
        
        result = {
            'total': info.total_memory,
            'available': info.available_memory,
            'used': info.used_memory,
            'total_gb': info.total_memory_gb,
            'available_gb': info.available_memory_gb,
            'usage_percent': info.memory_usage_percent
        }
        
        # 如果是CUDA设备，获取实时信息
        if info.device_type == DeviceType.GPU:
            try:
                torch.cuda.set_device(info.device_id)
                result['reserved'] = torch.cuda.memory_reserved(info.device_id)
                result['allocated'] = torch.cuda.memory_allocated(info.device_id)
                result['max_allocated'] = torch.cuda.max_memory_allocated(info.device_id)
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")
        
        return result
    
    def set_device(self, device: Union[str, torch.device, int]):
        """设置主设备"""
        if isinstance(device, int):
            device = f'cuda:{device}'
        
        self._primary_device = torch.device(device)
        
        if self._primary_device.type == 'cuda':
            torch.cuda.set_device(self._primary_device)
        
        logger.info(f"Set primary device to {self._primary_device}")
    
    def to_device(
        self, 
        data: Union[Tensor, nn.Module, Dict, List, Any],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
    ) -> Any:
        """
        将数据移动到设备
        
        Args:
            data: 要移动的数据
            device: 目标设备
            non_blocking: 是否异步移动
        """
        target = device if device else self._primary_device
        
        return to_device(data, target, non_blocking)
    
    def synchronize(self, device: Optional[Union[str, torch.device]] = None):
        """同步设备操作"""
        target = device if device else self._primary_device
        
        if isinstance(target, str):
            target = torch.device(target)
        
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
    
    def empty_cache(self, device: Optional[Union[str, torch.device]] = None):
        """清空设备缓存"""
        target = device if device else self._primary_device
        
        if isinstance(target, str):
            target = torch.device(target)
        
        if target.type == 'cuda':
            torch.cuda.empty_cache()
    
    def reset_peak_memory_stats(self, device: Optional[Union[str, torch.device]] = None):
        """重置峰值内存统计"""
        target = device if device else self._primary_device
        
        if isinstance(target, str):
            target = torch.device(target)
        
        if target.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(target)
    
    def apply_config(self, config: HardwareConfig):
        """应用硬件配置"""
        self._config = config
        
        # 设置设备
        if config.device_type == DeviceType.GPU and config.device_ids:
            self.set_device(config.device_ids[0])
        
        # 设置cuDNN
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = config.cudnn_benchmark
            torch.backends.cudnn.deterministic = config.cudnn_deterministic
        
        # 设置内存限制
        if config.max_memory_gb and torch.cuda.is_available():
            max_bytes = int(config.max_memory_gb * 1024 ** 3)
            for device_id in config.device_ids:
                torch.cuda.set_per_process_memory_fraction(
                    max_bytes / torch.cuda.get_device_properties(device_id).total_memory,
                    device_id
                )
        
        logger.info(f"Applied hardware config: {config.to_dict()}")
    
    # ==================== 新增方法：监控和诊断 ====================
    
    def collect_metrics(self, device: Optional[Union[str, torch.device]] = None) -> DeviceMetrics:
        """
        收集设备指标
        
        Args:
            device: 设备
            
        Returns:
            设备指标
        """
        info = self.get_device_info(device)
        device_id = str(device) if device else str(self._primary_device)
        
        metrics = DeviceMetrics(device_id=device_id)
        
        # 内存信息
        mem_info = self.get_memory_info(device)
        metrics.memory_used = mem_info.get('used', 0)
        metrics.memory_total = mem_info.get('total', 0)
        metrics.memory_percent = mem_info.get('usage_percent', 0.0)
        
        # GPU特定信息
        if info.device_type == DeviceType.GPU:
            try:
                # 尝试获取温度和利用率
                if hasattr(torch.cuda, 'temperature'):
                    metrics.temperature = torch.cuda.temperature(info.device_id)
                
                if hasattr(torch.cuda, 'utilization'):
                    metrics.utilization = torch.cuda.utilization(info.device_id)
                
                # 分配计数
                metrics.allocation_count = self._allocation_counts.get(device_id, 0)
            except Exception as e:
                logger.debug(f"Failed to collect GPU metrics: {e}")
        
        # 记录到监控器
        self._monitor.record_metrics(metrics)
        
        return metrics
    
    def get_device_status(self, device: Optional[Union[str, torch.device]] = None) -> DeviceStatus:
        """获取设备状态"""
        device_id = str(device) if device else str(self._primary_device)
        return self._monitor.get_device_status(device_id)
    
    def check_health(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
        """
        检查设备健康状态
        
        Args:
            device: 设备
            
        Returns:
            健康检查结果
        """
        metrics = self.collect_metrics(device)
        status = self.get_device_status(device)
        info = self.get_device_info(device)
        
        issues = []
        warnings = []
        recommendations = []
        
        # 检查可用性
        if not info.is_available:
            issues.append("Device is not available")
        
        # 检查内存
        if metrics.memory_percent > 95:
            issues.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
            recommendations.append("Clear cache or reduce batch size")
        elif metrics.memory_percent > 85:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # 检查温度
        if metrics.temperature > 85:
            warnings.append(f"High temperature: {metrics.temperature:.1f}°C")
            recommendations.append("Check cooling system")
        
        # 检查错误计数
        if metrics.error_count > 0:
            warnings.append(f"Device errors detected: {metrics.error_count}")
        
        return {
            'device_id': metrics.device_id,
            'status': status.value,
            'metrics': metrics.to_dict(),
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'healthy': len(issues) == 0 and status != DeviceStatus.ERROR
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断所有设备
        
        Returns:
            诊断结果
        """
        results = {}
        all_healthy = True
        
        for device_key in self._devices.keys():
            health = self.check_health(device_key)
            results[device_key] = health
            if not health['healthy']:
                all_healthy = False
        
        return {
            'all_healthy': all_healthy,
            'devices': results,
            'primary_device': str(self._primary_device),
            'timestamp': time.time()
        }
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "="*80)
        print("Device Diagnosis")
        print("="*80)
        
        print(f"\nPrimary Device: {diagnosis['primary_device']}")
        print(f"Overall Status: {'✅ HEALTHY' if diagnosis['all_healthy'] else '⚠️ ISSUES DETECTED'}")
        
        for device_id, health in diagnosis['devices'].items():
            print(f"\n{'='*40}")
            print(f"Device: {device_id}")
            print(f"Status: {health['status'].upper()}")
            
            metrics = health['metrics']
            print(f"Memory: {metrics['memory_percent']:.1f}%")
            if metrics['temperature'] > 0:
                print(f"Temperature: {metrics['temperature']:.1f}°C")
            if metrics['utilization'] > 0:
                print(f"Utilization: {metrics['utilization']:.1f}%")
            
            if health['issues']:
                print("\n⛔ Issues:")
                for issue in health['issues']:
                    print(f"  - {issue}")
            
            if health['warnings']:
                print("\n⚠️  Warnings:")
                for warning in health['warnings']:
                    print(f"  - {warning}")
            
            if health['recommendations']:
                print("\n💡 Recommendations:")
                for rec in health['recommendations']:
                    print(f"  - {rec}")
        
        print("\n" + "="*80)
    
    # ==================== 新增方法：性能基准测试 ====================
    
    def run_benchmark(
        self,
        device: Optional[Union[str, torch.device]] = None,
        run_all: bool = True
    ) -> Dict[str, Any]:
        """
        运行设备性能基准测试
        
        Args:
            device: 设备
            run_all: 是否运行所有测试
            
        Returns:
            基准测试结果
        """
        target = device if device else self._primary_device
        if isinstance(target, str):
            target = torch.device(target)
        
        device_key = str(target)
        
        if device_key not in self._benchmarks:
            self._benchmarks[device_key] = DeviceBenchmark(target)
        
        benchmark = self._benchmarks[device_key]
        
        if run_all:
            results = benchmark.run_all_benchmarks()
        else:
            results = benchmark.get_results()
        
        return results
    
    def get_benchmark_results(
        self,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """获取基准测试结果"""
        device_key = str(device) if device else str(self._primary_device)
        
        if device_key in self._benchmarks:
            return self._benchmarks[device_key].get_results()
        
        return {}
    
    def print_benchmark_results(
        self,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """打印基准测试结果"""
        device_key = str(device) if device else str(self._primary_device)
        
        if device_key in self._benchmarks:
            self._benchmarks[device_key].print_results()
        else:
            print(f"No benchmark results for {device_key}")
    
    def compare_devices(self) -> None:
        """比较所有设备性能"""
        print("\n" + "="*80)
        print("Device Performance Comparison")
        print("="*80)
        
        # 运行所有设备的基准测试
        for device_key in self._devices.keys():
            if self._devices[device_key].is_available:
                try:
                    self.run_benchmark(device_key)
                except Exception as e:
                    logger.warning(f"Benchmark failed for {device_key}: {e}")
        
        # 打印比较
        print("\n{:<15} {:>12} {:>12} {:>15}".format(
            "Device", "Compute", "Memory", "Status"
        ))
        print("-" * 80)
        
        for device_key, benchmark in self._benchmarks.items():
            results = benchmark.get_results()
            status = self._monitor.get_device_status(device_key).value
            
            compute = results.get('compute', {}).get('gflops', 0)
            memory = results.get('memory', {}).get('read_bandwidth_gbs', 0)
            
            print("{:<15} {:>10.1f}G {:>10.1f}G/s {:>15}".format(
                device_key, compute, memory, status
            ))
        
        print("="*80)
    
    # ==================== 新增方法：设备分配和负载均衡 ====================
    
    def set_allocation_strategy(self, strategy: AllocationStrategy) -> None:
        """设置分配策略"""
        self._allocation_strategy = strategy
        logger.info(f"Device allocation strategy set to: {strategy.value}")
    
    def allocate_device(
        self,
        exclude: Optional[List[str]] = None
    ) -> torch.device:
        """
        智能分配设备
        
        根据配置的策略选择最佳设备
        
        Args:
            exclude: 排除的设备列表
            
        Returns:
            分配的设备
        """
        exclude = exclude or []
        available = [
            (k, info) for k, info in self._devices.items()
            if info.is_available and k not in exclude and info.device_type != DeviceType.CPU
        ]
        
        if not available:
            return torch.device('cpu')
        
        if self._allocation_strategy == AllocationStrategy.LEAST_LOADED:
            # 选择负载最低的设备
            selected = min(
                available,
                key=lambda x: self._allocation_counts.get(x[0], 0)
            )
        
        elif self._allocation_strategy == AllocationStrategy.MOST_MEMORY:
            # 选择可用内存最多的设备
            selected = max(
                available,
                key=lambda x: x[1].available_memory
            )
        
        elif self._allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            # 轮询选择
            selected = available[self._round_robin_index % len(available)]
            self._round_robin_index += 1
        
        else:  # BALANCED
            # 平衡考虑负载和内存
            def score(item):
                k, info = item
                load_score = 1.0 / (1 + self._allocation_counts.get(k, 0))
                memory_score = info.available_memory / info.total_memory
                return (load_score + memory_score) / 2
            
            selected = max(available, key=score)
        
        device_key = selected[0]
        self._allocation_counts[device_key] += 1
        
        return torch.device(device_key)
    
    def release_device(self, device: Union[str, torch.device]) -> None:
        """
        释放设备
        
        Args:
            device: 设备
        """
        device_key = str(device)
        if device_key in self._allocation_counts:
            self._allocation_counts[device_key] = max(
                0, self._allocation_counts[device_key] - 1
            )
    
    def get_allocation_stats(self) -> Dict[str, int]:
        """获取分配统计"""
        return self._allocation_counts.copy()
    
    def reset_allocation_stats(self) -> None:
        """重置分配统计"""
        self._allocation_counts.clear()
        self._round_robin_index = 0
    
    # ==================== 新增方法：信息和摘要 ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """获取设备管理器摘要"""
        return {
            'total_devices': len(self._devices),
            'gpu_count': len(self.get_gpu_devices()),
            'primary_device': str(self._primary_device),
            'allocation_strategy': self._allocation_strategy.value,
            'devices': {
                k: {
                    'type': info.device_type.value,
                    'name': info.name,
                    'memory_gb': info.total_memory_gb,
                    'available': info.is_available,
                    'status': self._monitor.get_device_status(k).value
                }
                for k, info in self._devices.items()
            }
        }
    
    def print_summary(self) -> None:
        """打印设备管理器摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("Device Manager Summary")
        print("="*80)
        
        print(f"\nTotal Devices: {summary['total_devices']}")
        print(f"GPU Count: {summary['gpu_count']}")
        print(f"Primary Device: {summary['primary_device']}")
        print(f"Allocation Strategy: {summary['allocation_strategy']}")
        
        print("\nDevices:")
        for device_id, device_info in summary['devices'].items():
            status_icon = "✅" if device_info['status'] == 'healthy' else "⚠️"
            print(f"  {status_icon} {device_id}: {device_info['name']} "
                  f"({device_info['memory_gb']:.1f} GB, {device_info['status']})")
        
        print("="*80)
    
    def get_device_utilization(self) -> Dict[str, float]:
        """获取所有设备的利用率"""
        utilization = {}
        
        for device_key in self._devices.keys():
            metrics = self._monitor.get_latest_metrics(device_key)
            if metrics:
                utilization[device_key] = metrics.utilization
            else:
                utilization[device_key] = 0.0
        
        return utilization
    
    def find_idle_device(self, threshold: float = 20.0) -> Optional[torch.device]:
        """
        查找空闲设备
        
        Args:
            threshold: 空闲阈值（利用率%）
            
        Returns:
            空闲设备或None
        """
        utilization = self.get_device_utilization()
        
        for device_key, util in utilization.items():
            if util < threshold and self._devices[device_key].is_available:
                if self._devices[device_key].device_type != DeviceType.CPU:
                    return torch.device(device_key)
        
        return None


# ==================== 全局实例和便捷函数 ====================

_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """获取设备管理器单例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def detect_devices() -> List[DeviceInfo]:
    """检测所有可用设备"""
    return get_device_manager().list_devices()


def select_device(preference: Optional[str] = None) -> torch.device:
    """选择设备"""
    manager = get_device_manager()
    if preference:
        return manager.get_device(preference)
    return manager.device


def to_device(
    data: Any,
    device: Union[str, torch.device],
    non_blocking: bool = False
) -> Any:
    """
    将数据移动到指定设备
    
    支持Tensor、Module、Dict、List等类型。
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(data, Tensor):
        return data.to(device, non_blocking=non_blocking)
    
    elif isinstance(data, nn.Module):
        return data.to(device)
    
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [to_device(item, device, non_blocking) for item in data]
        return type(data)(result)
    
    else:
        return data


# ==================== 新增工具函数 ====================

def get_optimal_device(
    min_memory_gb: Optional[float] = None,
    prefer_type: Optional[DeviceType] = None
) -> torch.device:
    """
    获取最优设备
    
    Args:
        min_memory_gb: 最小内存要求（GB）
        prefer_type: 优先的设备类型
        
    Returns:
        最优设备
    """
    manager = get_device_manager()
    
    devices = manager.list_devices()
    
    # 过滤
    candidates = [
        d for d in devices
        if d.is_available and
        (min_memory_gb is None or d.total_memory_gb >= min_memory_gb) and
        (prefer_type is None or d.device_type == prefer_type)
    ]
    
    if not candidates:
        return torch.device('cpu')
    
    # 选择最佳
    best = max(candidates, key=lambda d: (
        d.device_type == DeviceType.GPU,
        d.available_memory,
        d.capabilities.compute_capability or "0"
    ))
    
    return torch.device(f"{best.device_type.value}:{best.device_id}" if best.device_id else best.device_type.value)


def get_all_available_devices() -> List[torch.device]:
    """获取所有可用设备"""
    manager = get_device_manager()
    devices = manager.list_devices()
    
    return [
        torch.device(f"{d.device_type.value}:{d.device_id}" if d.device_id else d.device_type.value)
        for d in devices if d.is_available
    ]


def check_device_compatibility(
    device: Union[str, torch.device],
    precision: Optional[str] = None
) -> bool:
    """
    检查设备兼容性
    
    Args:
        device: 设备
        precision: 精度类型 (fp16, bf16, etc.)
        
    Returns:
        是否兼容
    """
    manager = get_device_manager()
    info = manager.get_device_info(device)
    
    if not info.is_available:
        return False
    
    if precision:
        caps = info.capabilities
        if precision == 'fp16' and not caps.supports_fp16:
            return False
        if precision == 'bf16' and not caps.supports_bf16:
            return False
        if precision == 'int8' and not caps.supports_int8:
            return False
    
    return True


def estimate_device_capacity(
    device: Union[str, torch.device],
    model_size_gb: float,
    batch_size: int = 1,
    sample_size_mb: float = 1.0
) -> Dict[str, Any]:
    """
    估算设备容量
    
    Args:
        device: 设备
        model_size_gb: 模型大小（GB）
        batch_size: 批次大小
        sample_size_mb: 单个样本大小（MB）
        
    Returns:
        容量估算
    """
    manager = get_device_manager()
    info = manager.get_device_info(device)
    
    available_gb = info.available_memory_gb
    required_gb = model_size_gb + (batch_size * sample_size_mb / 1024)
    
    can_fit = available_gb >= required_gb
    max_batch_size = int((available_gb - model_size_gb) * 1024 / sample_size_mb) if available_gb > model_size_gb else 0
    
    return {
        'can_fit': can_fit,
        'available_gb': available_gb,
        'required_gb': required_gb,
        'max_batch_size': max(1, max_batch_size),
        'margin_gb': available_gb - required_gb
    }


def print_device_info(device: Optional[Union[str, torch.device]] = None) -> None:
    """打印设备信息"""
    manager = get_device_manager()
    info = manager.get_device_info(device)
    
    print("\n" + "="*80)
    print("Device Information")
    print("="*80)
    
    print(f"\nDevice: {info.device_type.value}:{info.device_id}")
    print(f"Name: {info.name}")
    print(f"Available: {info.is_available}")
    
    print(f"\nMemory:")
    print(f"  Total: {info.total_memory_gb:.2f} GB")
    print(f"  Available: {info.available_memory_gb:.2f} GB")
    print(f"  Used: {info.used_memory_gb:.2f} GB")
    print(f"  Usage: {info.memory_usage_percent:.1f}%")
    
    caps = info.capabilities
    print(f"\nCapabilities:")
    print(f"  FP16: {caps.supports_fp16}")
    print(f"  BF16: {caps.supports_bf16}")
    print(f"  INT8: {caps.supports_int8}")
    if caps.compute_capability:
        print(f"  Compute Capability: {caps.compute_capability}")
    if caps.supports_tensor_cores:
        print(f"  Tensor Cores: Yes")
    
    if info.cuda_version:
        print(f"\nCUDA Version: {info.cuda_version}")
    
    print("="*80)


@contextmanager
def managed_device(
    device: Optional[Union[str, torch.device]] = None,
    empty_cache_on_exit: bool = True
):
    """
    设备管理上下文
    
    Args:
        device: 设备
        empty_cache_on_exit: 退出时清空缓存
        
    Yields:
        设备
    """
    manager = get_device_manager()
    
    if device is None:
        device = manager.device
    elif isinstance(device, str):
        device = torch.device(device)
    
    # 设置为当前设备
    original_device = manager.device
    if device != original_device:
        manager.set_device(device)
    
    try:
        yield device
    finally:
        # 恢复原设备
        if device != original_device:
            manager.set_device(original_device)
        
        # 清空缓存
        if empty_cache_on_exit:
            manager.empty_cache(device)


@contextmanager
def auto_device_allocation(
    strategy: AllocationStrategy = AllocationStrategy.BALANCED,
    release_on_exit: bool = True
):
    """
    自动设备分配上下文
    
    Args:
        strategy: 分配策略
        release_on_exit: 退出时释放设备
        
    Yields:
        分配的设备
    """
    manager = get_device_manager()
    
    # 设置策略
    original_strategy = manager._allocation_strategy
    manager.set_allocation_strategy(strategy)
    
    # 分配设备
    device = manager.allocate_device()
    
    try:
        yield device
    finally:
        # 释放设备
        if release_on_exit:
            manager.release_device(device)
        
        # 恢复策略
        manager.set_allocation_strategy(original_strategy)


@contextmanager
def device_monitoring(interval: float = 1.0, print_summary: bool = True):
    """
    设备监控上下文
    
    Args:
        interval: 监控间隔（秒）
        print_summary: 是否打印摘要
        
    Yields:
        设备管理器
    """
    manager = get_device_manager()
    
    # 收集初始指标
    manager.collect_metrics()
    
    try:
        yield manager
    finally:
        # 收集最终指标
        manager.collect_metrics()
        
        # 打印摘要
        if print_summary:
            print("\n" + "="*80)
            print("Device Monitoring Summary")
            print("="*80)
            
            for device_key in manager._devices.keys():
                avg_metrics = manager._monitor.get_average_metrics(device_key)
                if avg_metrics:
                    print(f"\n{device_key}:")
                    print(f"  Avg Memory: {avg_metrics.get('avg_memory_percent', 0):.1f}%")
                    if avg_metrics.get('avg_utilization', 0) > 0:
                        print(f"  Avg Utilization: {avg_metrics.get('avg_utilization', 0):.1f}%")
                    print(f"  Allocations: {avg_metrics.get('total_allocations', 0)}")
            
            print("="*80)


def clear_all_caches():
    """清空所有设备缓存"""
    manager = get_device_manager()
    
    for device_key, info in manager._devices.items():
        if info.device_type == DeviceType.GPU and info.is_available:
            manager.empty_cache(device_key)
    
    logger.info("Cleared all device caches")


def reset_all_devices():
    """重置所有设备统计"""
    manager = get_device_manager()
    
    for device_key, info in manager._devices.items():
        if info.device_type == DeviceType.GPU and info.is_available:
            manager.reset_peak_memory_stats(device_key)
    
    manager.reset_allocation_stats()
    manager._monitor.reset()
    
    logger.info("Reset all device statistics")


def synchronize_all_devices():
    """同步所有设备"""
    manager = get_device_manager()
    
    for device_key, info in manager._devices.items():
        if info.device_type == DeviceType.GPU and info.is_available:
            manager.synchronize(device_key)


def get_device_count(device_type: Optional[DeviceType] = None) -> int:
    """
    获取设备数量
    
    Args:
        device_type: 设备类型，None表示所有设备
        
    Returns:
        设备数量
    """
    manager = get_device_manager()
    devices = manager.list_devices()
    
    if device_type is None:
        return len(devices)
    
    return len([d for d in devices if d.device_type == device_type])


def is_device_available(device: Union[str, torch.device, DeviceType]) -> bool:
    """
    检查设备是否可用
    
    Args:
        device: 设备
        
    Returns:
        是否可用
    """
    if isinstance(device, DeviceType):
        if device == DeviceType.CPU:
            return True
        elif device == DeviceType.GPU:
            return torch.cuda.is_available()
        elif device == DeviceType.MPS:
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        return False
    
    manager = get_device_manager()
    info = manager.get_device_info(device)
    return info.is_available
