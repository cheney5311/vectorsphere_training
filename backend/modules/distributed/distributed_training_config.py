"""分布式训练配置模型（合并自 training.distributed.distributed_config）"""

from dataclasses import dataclass
from typing import List

import subprocess

# 尝试使用项目内的标准异常定义
try:
    from backend.core.exceptions import ValidationError  # 标准位置
except Exception:
    # 回退：若不可用，提供一个轻量替代，避免服务启动失败
    class ValidationError(Exception):
        def __init__(self, message: str, field: str = ""):
            super().__init__(message)
            self.field = field


@dataclass
class DistributedTrainingConfig:
    """分布式训练配置"""
    # 基础配置
    model_name: str = "gpt2"
    data_path: str = "./data/train.txt"
    output_dir: str = "./outputs/distributed"

    # 分布式配置
    world_size: int = 1  # 总进程数
    num_nodes: int = 1   # 节点数
    node_rank: int = 0   # 当前节点排名
    nproc_per_node: int = 1  # 每个节点的进程数（通常等于GPU数）
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # nccl, gloo, mpi

    # 训练参数
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 8  # 每个GPU的batch size
    gradient_accumulation_steps: int = 1
    max_length: int = 512

    # 优化参数
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    fp16: bool = True
    gradient_clipping: float = 1.0

    # 数据并行配置
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # 模型并行配置（可选）
    model_parallel: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1

    def __post_init__(self):
        """配置验证"""
        self.validate()

    def validate(self) -> None:
        """验证配置参数"""
        if self.world_size <= 0:
            raise ValidationError("world_size必须大于0", field="world_size")

        if self.num_nodes <= 0:
            raise ValidationError("num_nodes必须大于0", field="num_nodes")

        if self.nproc_per_node <= 0:
            raise ValidationError("nproc_per_node必须大于0", field="nproc_per_node")

        if self.node_rank < 0:
            raise ValidationError("node_rank不能为负数", field="node_rank")

        if self.learning_rate <= 0:
            raise ValidationError("learning_rate必须大于0", field="learning_rate")

        if self.num_epochs <= 0:
            raise ValidationError("num_epochs必须大于0", field="num_epochs")

        if self.batch_size <= 0:
            raise ValidationError("batch_size必须大于0", field="batch_size")

    def detect_gpus(self) -> List[int]:
        """检测可用的GPU"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                gpu_count = len([line for line in gpu_lines if 'GPU' in line])
                return list(range(gpu_count))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return []  # 没有GPU，使用CPU

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'data_path': self.data_path,
            'output_dir': self.output_dir,
            'world_size': self.world_size,
            'num_nodes': self.num_nodes,
            'node_rank': self.node_rank,
            'nproc_per_node': self.nproc_per_node,
            'master_addr': self.master_addr,
            'master_port': self.master_port,
            'backend': self.backend,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_length': self.max_length,
            'warmup_steps': self.warmup_steps,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'fp16': self.fp16,
            'gradient_clipping': self.gradient_clipping,
            'dataloader_num_workers': self.dataloader_num_workers,
            'pin_memory': self.pin_memory,
            'model_parallel': self.model_parallel,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'tensor_parallel_size': self.tensor_parallel_size
        }