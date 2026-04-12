"""分布式训练器

支持多种分布式训练策略。
"""

import asyncio
import json
import time
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging

from .cluster_manager import NodeInfo
from .task_scheduler import TrainingTask
from .resource_allocator import ResourceAllocation

logger = logging.getLogger(__name__)

# 便捷函数：合并 training.distributed 的入口，使用本模块统一实现
def launch_distributed_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    启动分布式训练（统一入口）
    - 尝试构造 DistributedTrainingConfig 并校验参数
    - 如果运行环境支持 torch.distributed 且 world_size>1，则返回可执行命令或触发真实启动（非阻塞）
    - 否则降级为安全的模拟模式（不会尝试直接 fork/exec），返回结构化的启动元信息
    - 保留扩展点：heartbeat/lease/idempotency 元数据便于上层编排器做容错与重试
    """
    from .distributed_training_config import DistributedTrainingConfig  # 延迟导入避免循环
    import time
    try:
        training_config = DistributedTrainingConfig(**config)
    except Exception as e:
        return {"success": False, "error": f"配置错误: {e}"}

    # 从传入配置中提取 runtime_allocation（若有）以便附带到返回结果，供上层编排/训练器消费
    runtime_allocation = None
    try:
        runtime_allocation = config.get('runtime_allocation') if isinstance(config, dict) else None
    except Exception:
        runtime_allocation = None

    # 基本返回骨架
    result = {
        "success": False,
        "distributed_config": training_config.to_dict() if hasattr(training_config, 'to_dict') else config,
        "orchestration": {},
        "env_info": {},
        "training_result": None,
        "runtime_allocation": runtime_allocation
    }

    # 填充环境信息（尽量宽容，不抛异常）
    try:
        detect_gpus = []
        if hasattr(training_config, 'detect_gpus'):
            try:
                detect_gpus = training_config.detect_gpus()
            except Exception:
                detect_gpus = []
        result['env_info'] = {
            'master_addr': getattr(training_config, 'master_addr', 'localhost'),
            'master_port': getattr(training_config, 'master_port', '12355'),
            'available_gpus': detect_gpus,
            'cuda_available': len(detect_gpus) > 0
        }
    except Exception:
        result['env_info'] = {}

    # 生成 orchestration metadata
    # orchestration metadata（支持外部传入 orchestration_id 以实现幂等启动）
    orchestration_id = config.get('orchestration_id') if isinstance(config, dict) else None
    if not orchestration_id:
        orchestration_id = f"dist-{int(time.time())}-{training_config.world_size}"

    orchestration = {
        'launch_time': time.time(),
        'heartbeat_interval_sec': 10,
        'lease_seconds': 60,
        'orchestration_id': orchestration_id
    }

    # 幂等性：检查本地注册表（/tmp/vectorsphere_orchestrations.json），避免重复启动
    try:
        registry_path = '/tmp/vectorsphere_orchestrations.json'
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, 'r', encoding='utf-8') as rf:
                try:
                    registry = json.load(rf)
                except Exception:
                    registry = {}
        if orchestration_id in registry:
            # 已存在相同 orchestration_id，返回现有记录（幂等）
            result['orchestration'] = registry[orchestration_id]
            result['success'] = True
            result['training_result'] = result.get('training_result', None)
            return result
        else:
            registry[orchestration_id] = orchestration
            try:
                with open(registry_path, 'w', encoding='utf-8') as wf:
                    json.dump(registry, wf, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception:
        pass

    result['orchestration'] = orchestration

    # 当真实的 torch.distributed 可用并且配置指明需要多进程时，尝试构造启动命令（不直接执行）
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    try:
        if has_torch and training_config.world_size and training_config.world_size > 1:
            # 推荐使用 torchrun / torch.distributed.launch，构造建议命令以供上层调度器执行
            nproc = getattr(training_config, 'nproc_per_node', getattr(training_config, 'gpus_per_node', 1))
            cmd = (
                f"torchrun --nproc_per_node={nproc} --nnodes={training_config.num_nodes} "
                f"--master_addr={training_config.master_addr} --master_port={training_config.master_port} "
                f"train.py --world_size {training_config.world_size} --backend {getattr(training_config, 'backend', 'nccl')}"
            )
            result['training_result'] = {
                'mode': 'command_preview',
                'command': cmd,
                'note': '环境中检测到 torch，可由上层调度器使用该命令来启动真实分布式训练'
            }
            result['success'] = True
        else:
            # 降级模拟模式：在无法执行真实分布式训练时返回安全的模拟结果
            result['training_result'] = {
                'mode': 'simulation',
                'message': '分布式训练已降级为模拟模式（无 torch 或 world_size<=1）',
                'process_id': f"sim-{int(time.time())}"
            }
            # 返回部分最终结果占位（上层需根据 real job 更新）
            result['final_result'] = {
                'training_completed': False,
                'model_saved': False,
                'metrics': None
            }
            result['success'] = True
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)

    return result


class DistributedStrategy(Enum):
    """分布式策略枚举"""
    DATA_PARALLEL = "data_parallel"      # 数据并行
    MODEL_PARALLEL = "model_parallel"    # 模型并行
    PIPELINE_PARALLEL = "pipeline_parallel"  # 流水线并行
    TENSOR_PARALLEL = "tensor_parallel"  # 张量并行
    ZERO_PARALLEL = "zero_parallel"      # ZeRO并行


class CommunicationBackend(Enum):
    """通信后端枚举"""
    NCCL = "nccl"      # NVIDIA Collective Communications Library
    GLOO = "gloo"      # Facebook的Gloo库
    MPI = "mpi"        # Message Passing Interface
    UCX = "ucx"        # Unified Communication X


@dataclass
class DistributedConfig:
    """分布式配置"""
    strategy: DistributedStrategy = DistributedStrategy.DATA_PARALLEL
    backend: CommunicationBackend = CommunicationBackend.NCCL
    world_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 12355
    rank: int = 0
    local_rank: int = 0
    num_nodes: int = 1
    gpus_per_node: int = 1
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = False
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True


class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.is_training = False
        self.training_task: Optional[TrainingTask] = None
        self.resource_allocation: Optional[ResourceAllocation] = None
        self._training_task: Optional[asyncio.Task] = None
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._completion_callback: Optional[Callable[[bool, Optional[str]], None]] = None
        # 心跳/lease 与重试配置
        self._lease_id: Optional[str] = None
        self._retry_config = {
            'max_retries': int(os.getenv('DIST_RETRY_MAX', '3')),
            'initial_delay': float(os.getenv('DIST_RETRY_INITIAL_DELAY', '1.0')),
            'backoff_base': float(os.getenv('DIST_RETRY_BACKOFF_BASE', '2.0')),
            'jitter': float(os.getenv('DIST_RETRY_JITTER', '0.5')),
        }
        self._retries_attempted: int = 0
    
    async def initialize(self, nodes: List[NodeInfo], 
                        allocation: ResourceAllocation,
                        allocation_id: Optional[str] = None) -> bool:
        """初始化分布式训练环境"""
        try:
            self.resource_allocation = allocation
            self.allocation_id = allocation_id
            
            # 初始化分布式环境
            await self._setup_distributed_environment(nodes)
            
            self.is_initialized = True
            logger.info("Distributed trainer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed trainer: {e}")
            return False
    
    async def _setup_distributed_environment(self, nodes: List[NodeInfo]):
        """设置分布式环境"""
        # 这里应该实现分布式环境的设置
        # 包括初始化通信后端、设置主节点地址等
        logger.info(f"Setting up distributed environment with {len(nodes)} nodes")
        
        # 模拟初始化过程
        await asyncio.sleep(0.1)
    
    async def start_training(self, task: TrainingTask, 
                           progress_callback: Optional[Callable[[float], None]] = None,
                           completion_callback: Optional[Callable[[bool, Optional[str]], None]] = None) -> bool:
        """开始训练（带租约与心跳）"""
        if not self.is_initialized:
            logger.error("Distributed trainer not initialized")
            return False
        
        if self.is_training:
            logger.warning("Training already in progress")
            return False
        
        try:
            self.training_task = task
            self._progress_callback = progress_callback
            self._completion_callback = completion_callback

            # 为任务创建租约（lease），用于心跳与过期容错
            try:
                from backend.modules.distributed.lease_manager import get_lease_manager
                lease_mgr = get_lease_manager()
                self._lease_id = f"task-lease-{task.task_id}-{int(time.time())}"
                # TTL 可按需调整；这里取 120s
                await lease_mgr.create_lease(lease_id=self._lease_id, owner_id=task.task_id, ttl_seconds=120, metadata={"task_id": task.task_id})
                # 将 lease_id 写回 task.config 以便调度器监控
                try:
                    task.config['lease_id'] = self._lease_id
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Lease creation skipped or failed: {e}")
                self._lease_id = None
            
            # 启动训练任务（含重试策略）
            self._training_task = asyncio.create_task(self._training_loop_with_retry())
            self.is_training = True
            
            logger.info(f"Started distributed training for task: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start distributed training: {e}")
            return False
    
    async def _training_loop(self):
        """训练循环（携带心跳）"""
        try:
            # 模拟训练过程
            for epoch in range(10):
                for batch in range(100):
                    # 模拟训练步骤
                    await asyncio.sleep(0.01)

                    # 心跳：续租任务 lease
                    try:
                        if self._lease_id:
                            from backend.modules.distributed.lease_manager import get_lease_manager
                            lease_mgr = get_lease_manager()
                            await lease_mgr.heartbeat(self._lease_id)
                    except Exception:
                        pass
                    
                    # 更新进度
                    progress = (epoch * 100 + batch + 1) / 1000.0
                    if self._progress_callback:
                        self._progress_callback(progress)
                    
                    # 检查是否需要停止
                    if not self.is_training:
                        break
                
                if not self.is_training:
                    break
            
            # 训练完成
            await self._on_training_completed(True, None)
            
        except Exception as e:
            logger.error(f"Training loop error: {e}")
            await self._on_training_completed(False, str(e))

    async def _training_loop_with_retry(self):
        """带指数回退重试的训练包装循环"""
        max_retries = self._retry_config['max_retries']
        delay = self._retry_config['initial_delay']
        base = self._retry_config['backoff_base']
        jitter = self._retry_config['jitter']
        self._retries_attempted = 0

        while True:
            try:
                await self._training_loop()
                break
            except Exception as e:
                # _training_loop 自身捕获异常并调用 _on_training_completed；这里仅在未成功时考虑重试
                if self._retries_attempted >= max_retries:
                    logger.error(f"Training failed after {self._retries_attempted} retries: {e}")
                    break
                self._retries_attempted += 1
                sleep_for = delay + (jitter * (0.5))
                logger.warning(f"Retrying training (attempt {self._retries_attempted}/{max_retries}) after {sleep_for:.2f}s")
                await asyncio.sleep(sleep_for)
                delay *= base
    
    async def _on_training_completed(self, success: bool, error_message: Optional[str]):
        """训练完成回调"""
        self.is_training = False
        
        if self._completion_callback:
            self._completion_callback(success, error_message)
        
        logger.info(f"Training completed. Success: {success}, Error: {error_message}")
    
    async def stop_training(self) -> bool:
        """停止训练"""
        if not self.is_training:
            logger.warning("No training in progress")
            return False
        
        try:
            self.is_training = False
            
            if self._training_task:
                self._training_task.cancel()
                try:
                    await self._training_task
                except asyncio.CancelledError:
                    pass

            # 回收租约
            try:
                if self._lease_id:
                    from backend.modules.distributed.lease_manager import get_lease_manager
                    lease_mgr = get_lease_manager()
                    await lease_mgr.revoke(self._lease_id)
            except Exception:
                pass
            
            logger.info("Training stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return False
    
    async def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            "is_initialized": self.is_initialized,
            "is_training": self.is_training,
            "task_id": self.training_task.task_id if self.training_task else None,
            "config": {
                "strategy": self.config.strategy.value,
                "backend": self.config.backend.value,
                "world_size": self.config.world_size
            },
            "allocation_id": self.allocation_id,
            "resource_allocation": {
                "node_id": self.resource_allocation.node_id if self.resource_allocation else None,
                "cpu_cores": self.resource_allocation.cpu_cores if self.resource_allocation else 0,
                "memory_mb": self.resource_allocation.memory_mb if self.resource_allocation else 0,
                "gpus": self.resource_allocation.gpus if self.resource_allocation else []
            } if self.resource_allocation else None
        }
    
    async def get_training_progress(self) -> float:
        """获取训练进度"""
        if self.training_task:
            return self.training_task.progress
        return 0.0
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.is_training:
                await self.stop_training()
            
            self.is_initialized = False
            self.training_task = None
            self.resource_allocation = None
            self._training_task = None
            self._progress_callback = None
            self._completion_callback = None
            
            logger.info("Distributed trainer cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup distributed trainer: {e}")


# 全局分布式训练器工厂
class DistributedTrainerFactory:
    """分布式训练器工厂"""
    
    @staticmethod
    def create_trainer(config: DistributedConfig) -> DistributedTrainer:
        """创建分布式训练器"""
        return DistributedTrainer(config)
    
    @staticmethod
    async def create_and_initialize(config: DistributedConfig, 
                                  nodes: List[NodeInfo],
                                  allocation: ResourceAllocation,
                                  allocation_id: Optional[str] = None) -> Optional[DistributedTrainer]:
        """创建并初始化分布式训练器"""
        trainer = DistributedTrainer(config)
        success = await trainer.initialize(nodes, allocation, allocation_id=allocation_id)
        return trainer if success else None


# 全局分布式训练器管理器
class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self):
        self.trainers: Dict[str, DistributedTrainer] = {}
        self._lock = asyncio.Lock()
    
    async def create_trainer(self, task_id: str, config: DistributedConfig) -> Optional[DistributedTrainer]:
        """创建训练器"""
        async with self._lock:
            if task_id in self.trainers:
                logger.warning(f"Trainer for task {task_id} already exists")
                return self.trainers[task_id]
            
            trainer = DistributedTrainer(config)
            self.trainers[task_id] = trainer
            logger.info(f"Created trainer for task: {task_id}")
            return trainer
    
    async def get_trainer(self, task_id: str) -> Optional[DistributedTrainer]:
        """获取训练器"""
        async with self._lock:
            return self.trainers.get(task_id)
    
    async def remove_trainer(self, task_id: str) -> bool:
        """移除训练器"""
        async with self._lock:
            if task_id in self.trainers:
                trainer = self.trainers[task_id]
                await trainer.cleanup()
                del self.trainers[task_id]
                logger.info(f"Removed trainer for task: {task_id}")
                return True
            return False
    
    async def list_trainers(self) -> List[str]:
        """列出所有训练器"""
        async with self._lock:
            return list(self.trainers.keys())


# 全局分布式训练管理器实例
_distributed_training_manager: Optional[DistributedTrainingManager] = None


def get_distributed_training_manager() -> DistributedTrainingManager:
    """获取全局分布式训练管理器实例"""
    global _distributed_training_manager
    if _distributed_training_manager is None:
        _distributed_training_manager = DistributedTrainingManager()
    return _distributed_training_manager


def set_distributed_training_manager(manager: DistributedTrainingManager):
    """设置全局分布式训练管理器实例"""
    global _distributed_training_manager
    _distributed_training_manager = manager