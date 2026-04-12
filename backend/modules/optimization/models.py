"""优化配置模型

定义资源优化的各种配置参数、策略和阈值。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime


class OptimizationStrategy(Enum):
    """优化策略枚举"""
    BALANCED = "balanced"  # 平衡策略
    PERFORMANCE_FIRST = "performance_first"  # 性能优先
    COST_EFFICIENT = "cost_efficient"  # 成本效率优先
    ENERGY_SAVING = "energy_saving"  # 节能模式


class OptimizationMode(Enum):
    """优化模式枚举"""
    DEVELOPMENT = "development"  # 开发模式
    TESTING = "testing"  # 测试模式
    PRODUCTION = "production"  # 生产模式
    HIGH_PERFORMANCE = "high_performance"  # 高性能模式
    COST_SAVING = "cost_saving"  # 成本节约模式


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    BANDWIDTH = "bandwidth"


@dataclass
class ThresholdConfig:
    """阈值配置"""
    # CPU阈值
    cpu_utilization_low: float = 0.2  # CPU利用率过低阈值
    cpu_utilization_high: float = 0.85  # CPU利用率过高阈值
    cpu_utilization_critical: float = 0.95  # CPU利用率临界阈值

    # 内存阈值
    memory_utilization_low: float = 0.3  # 内存利用率过低阈值
    memory_utilization_high: float = 0.8  # 内存利用率过高阈值
    memory_utilization_critical: float = 0.9  # 内存利用率临界阈值

    # GPU阈值
    gpu_utilization_low: float = 0.25  # GPU利用率过低阈值
    gpu_utilization_high: float = 0.85  # GPU利用率过高阈值
    gpu_utilization_critical: float = 0.95  # GPU利用率临界阈值

    gpu_memory_utilization_high: float = 0.8  # GPU内存利用率过高阈值
    gpu_memory_utilization_critical: float = 0.9  # GPU内存利用率临界阈值

    gpu_temperature_high: float = 80.0  # GPU温度过高阈值(°C)
    gpu_temperature_critical: float = 90.0  # GPU温度临界阈值(°C)

    # 存储阈值
    disk_utilization_high: float = 0.8  # 磁盘利用率过高阈值
    disk_utilization_critical: float = 0.9  # 磁盘利用率临界阈值

    disk_io_high_mbps: float = 500.0  # 磁盘IO过高阈值(MB/s)
    disk_queue_depth_high: float = 10.0  # 磁盘队列深度过高阈值

    # 网络阈值
    network_utilization_high: float = 0.8  # 网络利用率过高阈值
    network_latency_high_ms: float = 100.0  # 网络延迟过高阈值(ms)

    # 性能阈值
    task_latency_multiplier: float = 1.5  # 任务延迟倍数阈值
    error_rate_high: float = 0.05  # 错误率过高阈值(5%)
    throughput_drop_ratio: float = 0.2  # 吞吐量下降比例阈值(20%)


@dataclass
class SchedulingConfig:
    """调度配置"""
    # 调度策略权重
    performance_weight: float = 0.4  # 性能权重
    cost_weight: float = 0.3  # 成本权重
    availability_weight: float = 0.2  # 可用性权重
    energy_weight: float = 0.1  # 能耗权重

    # GPU调度参数
    gpu_allocation_strategy: str = "balanced"  # GPU分配策略: balanced, performance, cost, energy
    gpu_sharing_enabled: bool = True  # 是否启用GPU共享
    gpu_max_tasks_per_gpu: int = 4  # 每个GPU最大任务数
    gpu_memory_fragmentation_threshold: float = 0.1  # GPU内存碎片化阈值

    # 内存调度参数
    memory_allocation_strategy: str = "first_fit"  # 内存分配策略: first_fit, best_fit, worst_fit
    memory_overcommit_ratio: float = 1.2  # 内存超分比例
    memory_swap_threshold: float = 0.8  # 内存交换阈值

    # 任务调度参数
    task_priority_levels: int = 5  # 任务优先级级别数
    task_queue_max_size: int = 1000  # 任务队列最大大小
    task_timeout_seconds: int = 3600  # 任务超时时间(秒)
    task_retry_max_attempts: int = 3  # 任务最大重试次数

    # 负载均衡参数
    load_balance_algorithm: str = "round_robin"  # 负载均衡算法: round_robin, least_connections, weighted
    load_balance_check_interval: int = 30  # 负载均衡检查间隔(秒)
    node_health_check_interval: int = 60  # 节点健康检查间隔(秒)


@dataclass
class CacheConfig:
    """缓存配置"""
    # 缓存策略
    default_cache_policy: str = "lru"  # 默认缓存策略: lru, lfu, fifo, random
    cache_size_mb: int = 2048  # 缓存大小(MB)
    cache_ttl_seconds: int = 3600  # 缓存TTL(秒)

    # 分层缓存
    l1_cache_size_mb: int = 512  # L1缓存大小(MB)
    l2_cache_size_mb: int = 1536  # L2缓存大小(MB)

    # 预取配置
    prefetch_enabled: bool = True  # 是否启用预取
    prefetch_size_mb: int = 100  # 预取大小(MB)
    prefetch_threshold: float = 0.7  # 预取阈值

    # 缓存命中率阈值
    cache_hit_rate_low: float = 0.6  # 缓存命中率过低阈值
    cache_hit_rate_target: float = 0.8  # 缓存命中率目标

    # 缓存清理
    cache_cleanup_interval: int = 300  # 缓存清理间隔(秒)
    cache_cleanup_threshold: float = 0.9  # 缓存清理阈值


@dataclass
class PredictionConfig:
    """预测配置"""
    # 预测模型参数
    prediction_window_minutes: int = 30  # 预测窗口(分钟)
    history_window_hours: int = 24  # 历史窗口(小时)
    min_data_points: int = 10  # 最少数据点数

    # 预测算法
    cpu_prediction_algorithm: str = "moving_average"  # CPU预测算法
    memory_prediction_algorithm: str = "exponential_smoothing"  # 内存预测算法
    gpu_prediction_algorithm: str = "linear_regression"  # GPU预测算法

    # 预测精度
    prediction_confidence_threshold: float = 0.7  # 预测置信度阈值
    prediction_accuracy_target: float = 0.8  # 预测精度目标

    # 异常检测
    anomaly_detection_enabled: bool = True  # 是否启用异常检测
    anomaly_threshold_sigma: float = 2.0  # 异常检测阈值(标准差倍数)
    anomaly_window_size: int = 20  # 异常检测窗口大小


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 基本配置
    mode: OptimizationMode = OptimizationMode.PRODUCTION
    enabled: bool = True

    # 优化间隔
    optimization_interval_seconds: int = 60  # 优化间隔(秒)
    metrics_collection_interval_seconds: int = 10  # 指标收集间隔(秒)

    # 优化限制
    max_recommendations_per_cycle: int = 5  # 每次最大建议数
    max_concurrent_optimizations: int = 3  # 最大并发优化数
    optimization_cooldown_seconds: int = 300  # 优化冷却时间(秒)

    # 子配置
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    # 资源限制
    total_memory_mb: int = 32768  # 总内存(MB)
    total_cpu_cores: int = 16  # 总CPU核心数
    total_gpu_count: int = 8  # 总GPU数量
    total_disk_gb: int = 1024  # 总磁盘空间(GB)

    # 环境特定配置
    environment_overrides: Dict[str, Any] = field(default_factory=dict)

    # 日志配置
    log_level: str = "INFO"
    log_optimization_decisions: bool = True
    log_metrics: bool = False  # 是否记录详细指标

    # 安全配置
    enable_resource_limits: bool = True  # 是否启用资源限制
    max_resource_allocation_ratio: float = 0.9  # 最大资源分配比例
    emergency_resource_reserve_ratio: float = 0.1  # 紧急资源预留比例