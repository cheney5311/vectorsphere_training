"""分布式训练模块

提供高性能分布式训练架构，支持大规模GPU集群管理、任务调度、资源分配等功能。
"""

from .cluster_manager import ClusterManager, NodeInfo, ClusterStatus
from .task_scheduler import TaskScheduler, TrainingTask, TaskStatus, SchedulingStrategy
from .resource_allocator import ResourceAllocator, ResourceAllocation
from .task_scheduler import ResourceRequirement
from .distributed_trainer import DistributedTrainer, launch_distributed_training
from .communication_backend import CommunicationBackend
from .fault_tolerance import FaultToleranceManager as FaultTolerance, CheckpointManager, RecoveryStrategy
from .load_balancer import LoadBalancer, LoadBalancingStrategy
from .metrics_collector import MetricsCollector
from .distributed_training_config import DistributedTrainingConfig

__all__ = [
    # 集群管理
    'ClusterManager',
    'NodeInfo',
    'ClusterStatus',
    
    # 任务调度
    'TaskScheduler',
    'TrainingTask',
    'TaskStatus',
    'SchedulingStrategy',
    
    # 资源分配
    'ResourceAllocator',
    'ResourceRequirement',
    'ResourceAllocation',
    
    # 分布式训练
    'DistributedTrainer',
    'DistributedTrainingConfig',
    'launch_distributed_training',
    
    # 通信后端
    'CommunicationBackend',
    
    # 容错机制
    'FaultTolerance',
    'CheckpointManager',
    'RecoveryStrategy',
    
    # 负载均衡
    'LoadBalancer',
    'LoadBalancingStrategy',
    
    # 指标收集
    'MetricsCollector',
]