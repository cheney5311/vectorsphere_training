# -*- coding: utf-8 -*-
"""
适配器管理器 - 生产级实现

统一管理模态编码器、融合模块、对齐模块和模型适配器。

生产级特性：
- 组件缓存和预热
- 指标收集和监控
- 健康检查和诊断
- 批量处理支持
- 内存管理
- 检查点保存/加载
- 配置验证
- 错误恢复
"""

import logging
import time
import os
import json
import gc
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from enum import Enum
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn

from .modality_encoders import (
    ModalityType, ModalityEncoder, EncoderConfig, EncoderFactory,
    create_encoder as _create_encoder,
    # 新增类型
    EncoderStatus, PoolingMethod as EncoderPoolingMethod, AugmentationType as EncoderAugmentationType,
    EncoderMetrics, EncoderConfigBuilder, build_encoder_config,
    # 新增编码器
    GraphEncoder, PointCloudEncoder, MultiModalEncoder,
    # 工具
    EncoderQualityAnalyzer, create_multimodal_encoder, encoder_factory_health_check
)
from .fusion_modules import (
    FusionMethod, FusionModule, FusionConfig, FusionFactory,
    create_fusion as _create_fusion,
    # 新增类型
    FusionStatus, PoolingType as FusionPoolingType, NormType,
    FusionMetrics, FusionConfigBuilder, build_fusion_config,
    # 新增融合模块
    TensorFusion, BilinearFusion, HybridFusion,
    # 工具
    FusionQualityAnalyzer, create_hybrid_fusion
)
from .alignment_modules import (
    AlignmentMethod, AlignmentModule, AlignmentConfig, AlignmentFactory,
    create_alignment as _create_alignment,
    # 新增类型
    AlignmentStatus, PoolingMethod as AlignmentPoolingMethod, LossType, AugmentationType as AlignmentAugmentationType,
    AlignmentMetrics, AlignmentConfigBuilder, build_alignment_config,
    # 新增对齐模块
    CCAAlignment, HybridAlignment,
    # 工具
    QualityMetric as AlignmentQualityMetric
)
from .model_adapters import (
    AdapterType, ModelAdapter, AdapterConfig, AdapterFactory,
    create_adapter as _create_adapter,
    # 新增类型
    AdapterStatus, MergeStrategy, InitStrategy,
    AdapterMetrics, AdapterConfigBuilder, build_adapter_config,
    # 新增适配器
    AdapterLayersAdapter, BitFitAdapter, IA3Adapter, CompacterAdapter,
    LoRALayer, LoRAAdapter,
    # 融合和合并
    AdapterFusion, AdapterMerger,
    # 工具
    AdapterQualityAnalyzer, create_lora_adapter, create_adapter_fusion,
    merge_adapters, adapter_factory_health_check, get_adapter_summary
)

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class ComponentType(Enum):
    """组件类型"""
    ENCODER = "encoder"
    FUSION = "fusion"
    ALIGNMENT = "alignment"
    ADAPTER = "adapter"
    PIPELINE = "pipeline"


class ManagerStatus(Enum):
    """管理器状态"""
    IDLE = "idle"
    RUNNING = "running"
    WARMING_UP = "warming_up"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ==================== 数据类 ====================

@dataclass
class ComponentMetrics:
    """组件指标"""
    component_type: str
    component_key: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    
    def record_use(self, processing_time: float = 0.0):
        """记录使用"""
        self.last_used = datetime.now()
        self.use_count += 1
        self.total_processing_time += processing_time
        if self.use_count > 0:
            self.avg_processing_time = self.total_processing_time / self.use_count
    
    def record_error(self):
        """记录错误"""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "component_key": self.component_key,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "use_count": self.use_count,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": self.avg_processing_time,
            "error_count": self.error_count,
            "memory_usage_mb": self.memory_usage_mb
        }


@dataclass
class PipelineMetrics:
    """管道指标"""
    pipeline_id: str
    modalities: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    total_inferences: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    errors: int = 0
    
    def record_inference(self, time_taken: float):
        """记录推理"""
        self.total_inferences += 1
        self.total_time += time_taken
        self.avg_time = self.total_time / self.total_inferences
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "modalities": self.modalities,
            "created_at": self.created_at.isoformat(),
            "total_inferences": self.total_inferences,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "errors": self.errors
        }


@dataclass
class AdapterManagerConfig:
    """适配器管理器配置"""
    # 默认隐藏层大小
    default_hidden_size: int = 768
    
    # 默认组件配置
    default_modalities: List[str] = field(default_factory=lambda: ["text"])
    default_fusion_method: str = "cross_attention"
    default_alignment_method: str = "contrastive"
    default_adapter_type: str = "lora"
    
    # 缓存配置
    cache_encoders: bool = True
    cache_fusions: bool = True
    cache_alignments: bool = True
    cache_adapters: bool = True
    max_cache_size: int = 100
    
    # 设备配置
    device: str = "auto"  # auto, cuda, cpu
    
    # 指标配置
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    
    # 预热配置
    enable_warmup: bool = False
    warmup_modalities: List[str] = field(default_factory=lambda: ["text"])
    
    # 内存管理
    max_memory_gb: float = 8.0
    memory_cleanup_threshold: float = 0.9
    
    # 检查点配置
    checkpoint_dir: str = "./adapter_checkpoints"
    auto_checkpoint: bool = False
    checkpoint_interval_minutes: int = 30
    
    # 健康检查
    health_check_interval_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdapterManagerConfig':
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


# 全局管理器实例
_adapter_manager: Optional['AdapterManager'] = None
_manager_lock = threading.Lock()


class AdapterManager:
    """
    适配器管理器 - 生产级实现
    
    统一管理所有适配器组件的创建、配置和缓存。
    
    特性：
    - 组件缓存和预热
    - 指标收集和监控
    - 健康检查和诊断
    - 批量处理支持
    - 内存管理
    - 检查点保存/加载
    """
    
    def __init__(self, config: Optional[AdapterManagerConfig] = None):
        self.config = config or AdapterManagerConfig()
        
        # 组件缓存
        self._encoder_cache: Dict[str, ModalityEncoder] = {}
        self._fusion_cache: Dict[str, FusionModule] = {}
        self._alignment_cache: Dict[str, AlignmentModule] = {}
        self._adapter_cache: Dict[str, ModelAdapter] = {}
        
        # 管道缓存
        self._pipeline_cache: Dict[str, Dict[str, Any]] = {}
        
        # 指标收集
        self._component_metrics: Dict[str, ComponentMetrics] = {}
        self._pipeline_metrics: Dict[str, PipelineMetrics] = {}
        self._global_metrics = {
            "total_components_created": 0,
            "total_inferences": 0,
            "total_errors": 0,
            "startup_time": datetime.now()
        }
        
        # 状态
        self._status = ManagerStatus.IDLE
        self._health_status = HealthStatus.HEALTHY
        self._last_health_check: Optional[datetime] = None
        
        # 线程锁
        self._cache_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        
        # 设备
        self._device = self._resolve_device()
        
        # 配置验证
        self._validate_config()
        
        # 预热（如果启用）
        if self.config.enable_warmup:
            self._warmup_components()
        
        # 初始健康检查
        self._perform_health_check()
        
        logger.info(f"AdapterManager initialized: device={self._device}, status={self._status.value}")
    
    def _resolve_device(self) -> torch.device:
        """解析设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _validate_config(self) -> None:
        """验证配置"""
        errors = []
        
        if self.config.default_hidden_size <= 0:
            errors.append("default_hidden_size must be positive")
        
        if self.config.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        
        if not 0 < self.config.memory_cleanup_threshold <= 1:
            errors.append("memory_cleanup_threshold must be between 0 and 1")
        
        # 验证默认模态
        valid_modalities = [m.value for m in ModalityType]
        for modality in self.config.default_modalities:
            if modality not in valid_modalities:
                errors.append(f"Invalid modality: {modality}")
        
        # 验证默认方法
        valid_fusion = [f.value for f in FusionMethod]
        if self.config.default_fusion_method not in valid_fusion:
            errors.append(f"Invalid fusion method: {self.config.default_fusion_method}")
        
        valid_alignment = [a.value for a in AlignmentMethod]
        if self.config.default_alignment_method not in valid_alignment:
            errors.append(f"Invalid alignment method: {self.config.default_alignment_method}")
        
        valid_adapter = [t.value for t in AdapterType]
        if self.config.default_adapter_type not in valid_adapter:
            errors.append(f"Invalid adapter type: {self.config.default_adapter_type}")
        
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
    
    def _warmup_components(self) -> None:
        """预热组件"""
        self._status = ManagerStatus.WARMING_UP
        logger.info("Starting component warmup...")
        
        start_time = time.time()
        warmed_up = []
        
        try:
            # 预热编码器
            for modality in self.config.warmup_modalities:
                try:
                    self.get_encoder(modality)
                    warmed_up.append(f"encoder:{modality}")
                except Exception as e:
                    logger.warning(f"Failed to warmup encoder {modality}: {e}")
            
            # 预热默认融合模块
            try:
                self.get_fusion(self.config.default_fusion_method)
                warmed_up.append(f"fusion:{self.config.default_fusion_method}")
            except Exception as e:
                logger.warning(f"Failed to warmup fusion: {e}")
            
            warmup_time = time.time() - start_time
            logger.info(f"Component warmup completed in {warmup_time:.2f}s, warmed: {warmed_up}")
            
        finally:
            self._status = ManagerStatus.RUNNING
    
    def _perform_health_check(self) -> HealthStatus:
        """执行健康检查"""
        issues = []
        
        # 检查设备
        try:
            if self._device.type == "cuda":
                if not torch.cuda.is_available():
                    issues.append("CUDA not available")
                else:
                    # 检查GPU内存
                    memory_used = torch.cuda.memory_allocated() / (1024**3)
                    if memory_used > self.config.max_memory_gb * self.config.memory_cleanup_threshold:
                        issues.append(f"High GPU memory usage: {memory_used:.2f}GB")
        except Exception as e:
            issues.append(f"Device check failed: {e}")
        
        # 检查缓存大小
        total_cached = (len(self._encoder_cache) + len(self._fusion_cache) + 
                       len(self._alignment_cache) + len(self._adapter_cache))
        if total_cached > self.config.max_cache_size:
            issues.append(f"Cache size exceeded: {total_cached}/{self.config.max_cache_size}")
        
        # 检查错误率
        if self._global_metrics["total_inferences"] > 100:
            error_rate = self._global_metrics["total_errors"] / self._global_metrics["total_inferences"]
            if error_rate > 0.1:
                issues.append(f"High error rate: {error_rate:.2%}")
        
        # 确定健康状态
        if len(issues) == 0:
            self._health_status = HealthStatus.HEALTHY
        elif len(issues) <= 2:
            self._health_status = HealthStatus.DEGRADED
        else:
            self._health_status = HealthStatus.UNHEALTHY
        
        self._last_health_check = datetime.now()
        
        if issues:
            logger.warning(f"Health check issues: {issues}")
        
        return self._health_status
    
    def _record_component_metric(self, component_type: str, component_key: str,
                                processing_time: float = 0.0, is_error: bool = False) -> None:
        """记录组件指标"""
        if not self.config.enable_metrics:
            return
        
        with self._metrics_lock:
            metric_key = f"{component_type}:{component_key}"
            
            if metric_key not in self._component_metrics:
                self._component_metrics[metric_key] = ComponentMetrics(
                    component_type=component_type,
                    component_key=component_key
                )
            
            metric = self._component_metrics[metric_key]
            
            if is_error:
                metric.record_error()
                self._global_metrics["total_errors"] += 1
            else:
                metric.record_use(processing_time)
                self._global_metrics["total_inferences"] += 1
    
    def _check_memory_and_cleanup(self) -> bool:
        """检查内存并在需要时清理"""
        if self._device.type != "cuda":
            return False
        
        try:
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            threshold = self.config.max_memory_gb * self.config.memory_cleanup_threshold
            
            if memory_used > threshold:
                logger.warning(f"Memory threshold exceeded ({memory_used:.2f}GB > {threshold:.2f}GB), cleaning up...")
                self._cleanup_least_used_components()
                torch.cuda.empty_cache()
                gc.collect()
                return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
        
        return False
    
    def _cleanup_least_used_components(self, keep_count: int = 5) -> int:
        """清理最少使用的组件"""
        cleaned = 0
        
        with self._cache_lock:
            # 按使用时间排序并清理
            for cache_name, cache in [
                ("encoder", self._encoder_cache),
                ("fusion", self._fusion_cache),
                ("alignment", self._alignment_cache),
                ("adapter", self._adapter_cache)
            ]:
                if len(cache) <= keep_count:
                    continue
                
                # 获取指标并排序
                items_with_metrics = []
                for key in list(cache.keys()):
                    metric_key = f"{cache_name}:{key}"
                    metric = self._component_metrics.get(metric_key)
                    last_used = metric.last_used if metric else datetime.min
                    items_with_metrics.append((key, last_used))
                
                # 按最后使用时间排序
                items_with_metrics.sort(key=lambda x: x[1])
                
                # 清理最旧的
                to_remove = items_with_metrics[:-keep_count]
                for key, _ in to_remove:
                    del cache[key]
                    cleaned += 1
                    logger.debug(f"Cleaned up {cache_name}:{key}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} components")
        
        return cleaned
    
    @contextmanager
    def _timed_operation(self, component_type: str, component_key: str):
        """计时操作上下文管理器"""
        start_time = time.time()
        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            self._record_component_metric(component_type, component_key, is_error=True)
            raise
        finally:
            if not error_occurred:
                elapsed = time.time() - start_time
                self._record_component_metric(component_type, component_key, elapsed)
    
    @property
    def device(self) -> torch.device:
        """获取设备"""
        return self._device
    
    @property
    def status(self) -> ManagerStatus:
        """获取管理器状态"""
        return self._status
    
    @property
    def health(self) -> HealthStatus:
        """获取健康状态"""
        return self._health_status
    
    # ==================== 编码器管理 ====================
    
    def get_encoder(
        self,
        modality: Union[ModalityType, str],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> ModalityEncoder:
        """
        获取模态编码器
        
        Args:
            modality: 模态类型
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
        """
        if isinstance(modality, str):
            modality = ModalityType(modality)
        
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 生成缓存key
        cache_key = f"{modality.value}_{hidden_size}"
        
        # 检查缓存
        with self._cache_lock:
            if self.config.cache_encoders and cache_key in self._encoder_cache:
                self._record_component_metric("encoder", cache_key, 0.0)
                return self._encoder_cache[cache_key]
        
        # 检查内存
        self._check_memory_and_cleanup()
        
        # 创建编码器
        with self._timed_operation("encoder", cache_key):
            encoder = _create_encoder(modality, hidden_size=hidden_size, **kwargs)
            encoder = encoder.to(self._device)
            
            # 更新内存使用
            self._update_component_memory("encoder", cache_key, encoder)
            
            # 缓存
            with self._cache_lock:
                if self.config.cache_encoders:
                    self._encoder_cache[cache_key] = encoder
                    self._global_metrics["total_components_created"] += 1
        
        logger.debug(f"Created encoder: {cache_key}")
        return encoder
    
    def _update_component_memory(self, component_type: str, component_key: str, 
                                module: nn.Module) -> None:
        """更新组件内存使用"""
        if not self.config.enable_metrics:
            return
        
        try:
            param_size = sum(p.numel() * p.element_size() for p in module.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
            memory_mb = (param_size + buffer_size) / (1024 * 1024)
            
            metric_key = f"{component_type}:{component_key}"
            if metric_key in self._component_metrics:
                self._component_metrics[metric_key].memory_usage_mb = memory_mb
        except Exception:
            pass
    
    def create_multimodal_encoders(
        self,
        modalities: List[str],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, ModalityEncoder]:
        """
        创建多模态编码器集合
        
        Args:
            modalities: 模态列表
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
        """
        encoders = {}
        errors = []
        
        for modality in modalities:
            try:
                encoders[modality] = self.get_encoder(modality, hidden_size, **kwargs)
            except Exception as e:
                errors.append(f"{modality}: {e}")
                logger.error(f"Failed to create encoder for {modality}: {e}")
        
        if errors and not encoders:
            raise RuntimeError(f"Failed to create any encoders: {errors}")
        
        return encoders
    
    def encode_batch(
        self,
        modality: Union[ModalityType, str],
        inputs: List[torch.Tensor],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        批量编码
        
        Args:
            modality: 模态类型
            inputs: 输入张量列表
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            编码结果列表
        """
        encoder = self.get_encoder(modality, hidden_size, **kwargs)
        
        results = []
        for inp in inputs:
            with torch.no_grad():
                result = encoder(inp.to(self._device))
                results.append(result)
        
        return results
    
    # ==================== 融合模块管理 ====================
    
    def get_fusion(
        self,
        method: Union[FusionMethod, str],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> FusionModule:
        """
        获取融合模块
        
        Args:
            method: 融合方法
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
        """
        if isinstance(method, str):
            method = FusionMethod(method)
        
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 生成缓存key
        cache_key = f"{method.value}_{hidden_size}"
        
        # 检查缓存
        with self._cache_lock:
            if self.config.cache_fusions and cache_key in self._fusion_cache:
                self._record_component_metric("fusion", cache_key, 0.0)
                return self._fusion_cache[cache_key]
        
        # 检查内存
        self._check_memory_and_cleanup()
        
        # 创建融合模块
        with self._timed_operation("fusion", cache_key):
            fusion = _create_fusion(method, hidden_size=hidden_size, **kwargs)
            fusion = fusion.to(self._device)
            
            # 更新内存使用
            self._update_component_memory("fusion", cache_key, fusion)
            
            # 缓存
            with self._cache_lock:
                if self.config.cache_fusions:
                    self._fusion_cache[cache_key] = fusion
                    self._global_metrics["total_components_created"] += 1
        
        logger.debug(f"Created fusion: {cache_key}")
        return fusion
    
    def fuse_features(
        self,
        features: List[torch.Tensor],
        method: Union[FusionMethod, str] = None,
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            features: 特征列表
            method: 融合方法
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            融合后的特征
        """
        method = method or self.config.default_fusion_method
        fusion = self.get_fusion(method, hidden_size, **kwargs)
        
        with torch.no_grad():
            return fusion(features)
    
    # ==================== 对齐模块管理 ====================
    
    def get_alignment(
        self,
        method: Union[AlignmentMethod, str],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> AlignmentModule:
        """
        获取对齐模块
        
        Args:
            method: 对齐方法
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
        """
        if isinstance(method, str):
            method = AlignmentMethod(method)
        
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 生成缓存key
        cache_key = f"{method.value}_{hidden_size}"
        
        # 检查缓存
        with self._cache_lock:
            if self.config.cache_alignments and cache_key in self._alignment_cache:
                self._record_component_metric("alignment", cache_key, 0.0)
                return self._alignment_cache[cache_key]
        
        # 检查内存
        self._check_memory_and_cleanup()
        
        # 创建对齐模块
        with self._timed_operation("alignment", cache_key):
            alignment = _create_alignment(method, hidden_size=hidden_size, **kwargs)
            alignment = alignment.to(self._device)
            
            # 更新内存使用
            self._update_component_memory("alignment", cache_key, alignment)
            
            # 缓存
            with self._cache_lock:
                if self.config.cache_alignments:
                    self._alignment_cache[cache_key] = alignment
                    self._global_metrics["total_components_created"] += 1
        
        logger.debug(f"Created alignment: {cache_key}")
        return alignment
    
    def align_features(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        method: Union[AlignmentMethod, str] = None,
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐两个模态的特征
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            method: 对齐方法
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            对齐后的特征元组
        """
        method = method or self.config.default_alignment_method
        alignment = self.get_alignment(method, hidden_size, **kwargs)
        
        with torch.no_grad():
            return alignment(features_a, features_b)
    
    # ==================== 适配器管理 ====================
    
    def get_adapter(
        self,
        adapter_type: Union[AdapterType, str],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> ModelAdapter:
        """
        获取模型适配器
        
        Args:
            adapter_type: 适配器类型
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
        """
        if isinstance(adapter_type, str):
            adapter_type = AdapterType(adapter_type)
        
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 生成缓存key
        cache_key = f"{adapter_type.value}_{hidden_size}"
        
        # 检查缓存
        with self._cache_lock:
            if self.config.cache_adapters and cache_key in self._adapter_cache:
                self._record_component_metric("adapter", cache_key, 0.0)
                return self._adapter_cache[cache_key]
        
        # 创建适配器
        with self._timed_operation("adapter", cache_key):
            adapter = _create_adapter(adapter_type, hidden_size=hidden_size, **kwargs)
            
            # 缓存
            with self._cache_lock:
                if self.config.cache_adapters:
                    self._adapter_cache[cache_key] = adapter
                    self._global_metrics["total_components_created"] += 1
        
        logger.debug(f"Created adapter: {cache_key}")
        return adapter
    
    def adapt_model(
        self,
        model: nn.Module,
        adapter_type: Union[AdapterType, str] = None,
        **kwargs
    ) -> nn.Module:
        """
        适配模型
        
        Args:
            model: 原始模型
            adapter_type: 适配器类型
            **kwargs: 其他配置
        """
        adapter_type = adapter_type or self.config.default_adapter_type
        adapter = self.get_adapter(adapter_type, **kwargs)
        
        adapted_model = adapter.adapt(model)
        adapted_model = adapted_model.to(self._device)
        
        return adapted_model
    
    def get_trainable_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """
        获取模型可训练参数统计
        
        Args:
            model: 模型
            
        Returns:
            (可训练参数数, 总参数数)
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total
    
    # ==================== 组合功能 ====================
    
    def create_multimodal_pipeline(
        self,
        modalities: List[str],
        fusion_method: str = None,
        alignment_method: str = None,
        hidden_size: int = None,
        pipeline_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建多模态处理管道
        
        Args:
            modalities: 模态列表
            fusion_method: 融合方法
            alignment_method: 对齐方法
            hidden_size: 隐藏层大小
            pipeline_id: 管道ID（用于缓存）
            **kwargs: 其他配置
            
        Returns:
            包含所有组件的字典
        """
        fusion_method = fusion_method or self.config.default_fusion_method
        alignment_method = alignment_method or self.config.default_alignment_method
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 生成管道ID
        pipeline_id = pipeline_id or f"pipeline_{'_'.join(modalities)}_{fusion_method}"
        
        # 检查缓存
        if pipeline_id in self._pipeline_cache:
            logger.debug(f"Pipeline cache hit: {pipeline_id}")
            return self._pipeline_cache[pipeline_id]
        
        start_time = time.time()
        
        try:
            pipeline = {
                'id': pipeline_id,
                'encoders': self.create_multimodal_encoders(modalities, hidden_size),
                'fusion': self.get_fusion(fusion_method, hidden_size),
                'alignment': self.get_alignment(alignment_method, hidden_size) if len(modalities) > 1 else None,
                'config': {
                    'modalities': modalities,
                    'fusion_method': fusion_method,
                    'alignment_method': alignment_method,
                    'hidden_size': hidden_size
                },
                'created_at': datetime.now().isoformat()
            }
            
            # 缓存管道
            self._pipeline_cache[pipeline_id] = pipeline
            
            # 记录指标
            if self.config.enable_metrics:
                self._pipeline_metrics[pipeline_id] = PipelineMetrics(
                    pipeline_id=pipeline_id,
                    modalities=modalities
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Created multimodal pipeline: {pipeline_id} in {elapsed:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline_id}: {e}")
            raise
    
    def run_pipeline(
        self,
        pipeline: Dict[str, Any],
        inputs: Dict[str, torch.Tensor],
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        运行多模态管道
        
        Args:
            pipeline: 管道字典
            inputs: 各模态输入字典 {modality: tensor}
            return_intermediate: 是否返回中间结果
            
        Returns:
            融合结果或包含中间结果的字典
        """
        start_time = time.time()
        pipeline_id = pipeline.get('id', 'unknown')
        
        try:
            # 编码各模态
            encoded_features = {}
            for modality, encoder in pipeline['encoders'].items():
                if modality in inputs:
                    inp = inputs[modality].to(self._device)
                    with torch.no_grad():
                        encoded_features[modality] = encoder(inp)
            
            # 对齐（如果有多个模态）
            aligned_features = {}
            if pipeline['alignment'] and len(encoded_features) >= 2:
                modality_list = list(encoded_features.keys())
                for i in range(len(modality_list) - 1):
                    m1, m2 = modality_list[i], modality_list[i + 1]
                    aligned_a, aligned_b = pipeline['alignment'](
                        encoded_features[m1], encoded_features[m2]
                    )
                    aligned_features[m1] = aligned_a
                    aligned_features[m2] = aligned_b
            else:
                aligned_features = encoded_features
            
            # 融合
            features_list = list(aligned_features.values())
            with torch.no_grad():
                fused = pipeline['fusion'](features_list)
            
            # 记录指标
            elapsed = time.time() - start_time
            if pipeline_id in self._pipeline_metrics:
                self._pipeline_metrics[pipeline_id].record_inference(elapsed)
            
            if return_intermediate:
                return {
                    'fused': fused,
                    'encoded': encoded_features,
                    'aligned': aligned_features,
                    'time': elapsed
                }
            
            return fused
            
        except Exception as e:
            if pipeline_id in self._pipeline_metrics:
                self._pipeline_metrics[pipeline_id].errors += 1
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的管道"""
        return self._pipeline_cache.get(pipeline_id)
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """删除管道"""
        if pipeline_id in self._pipeline_cache:
            del self._pipeline_cache[pipeline_id]
            if pipeline_id in self._pipeline_metrics:
                del self._pipeline_metrics[pipeline_id]
            logger.info(f"Deleted pipeline: {pipeline_id}")
            return True
        return False
    
    # ==================== 工具方法 ====================
    
    def clear_cache(self, component_type: Optional[str] = None) -> int:
        """
        清除缓存
        
        Args:
            component_type: 可选，指定清除的组件类型（encoder/fusion/alignment/adapter/pipeline）
                          如果为None则清除所有
        
        Returns:
            清除的组件数量
        """
        cleared = 0
        
        with self._cache_lock:
            if component_type is None or component_type == "encoder":
                cleared += len(self._encoder_cache)
                self._encoder_cache.clear()
            
            if component_type is None or component_type == "fusion":
                cleared += len(self._fusion_cache)
                self._fusion_cache.clear()
            
            if component_type is None or component_type == "alignment":
                cleared += len(self._alignment_cache)
                self._alignment_cache.clear()
            
            if component_type is None or component_type == "adapter":
                cleared += len(self._adapter_cache)
                self._adapter_cache.clear()
            
            if component_type is None or component_type == "pipeline":
                cleared += len(self._pipeline_cache)
                self._pipeline_cache.clear()
        
        # 清理GPU内存
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        
        logger.info(f"AdapterManager cache cleared: {cleared} components")
        return cleared
    
    def get_info(self) -> Dict[str, Any]:
        """获取管理器信息"""
        # 执行健康检查
        health_status = self._perform_health_check()
        
        return {
            'device': str(self._device),
            'status': self._status.value,
            'health': health_status.value,
            'cached_components': {
                'encoders': list(self._encoder_cache.keys()),
                'fusions': list(self._fusion_cache.keys()),
                'alignments': list(self._alignment_cache.keys()),
                'adapters': list(self._adapter_cache.keys()),
                'pipelines': list(self._pipeline_cache.keys())
            },
            'cache_sizes': {
                'encoders': len(self._encoder_cache),
                'fusions': len(self._fusion_cache),
                'alignments': len(self._alignment_cache),
                'adapters': len(self._adapter_cache),
                'pipelines': len(self._pipeline_cache),
                'total': (len(self._encoder_cache) + len(self._fusion_cache) + 
                         len(self._alignment_cache) + len(self._adapter_cache) +
                         len(self._pipeline_cache))
            },
            'config': self.config.to_dict(),
            'metrics_enabled': self.config.enable_metrics,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取详细指标"""
        metrics = {
            'global': {
                **self._global_metrics,
                'startup_time': self._global_metrics['startup_time'].isoformat(),
                'uptime_seconds': (datetime.now() - self._global_metrics['startup_time']).total_seconds()
            },
            'components': {},
            'pipelines': {}
        }
        
        # 组件指标
        for key, metric in self._component_metrics.items():
            metrics['components'][key] = metric.to_dict()
        
        # 管道指标
        for key, metric in self._pipeline_metrics.items():
            metrics['pipelines'][key] = metric.to_dict()
        
        # 内存指标
        if self._device.type == "cuda":
            metrics['memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_memory_gb': self.config.max_memory_gb
            }
        
        return metrics
    
    def get_component_metrics(self, component_type: str, component_key: str) -> Optional[Dict[str, Any]]:
        """获取特定组件的指标"""
        metric_key = f"{component_type}:{component_key}"
        metric = self._component_metrics.get(metric_key)
        return metric.to_dict() if metric else None
    
    def health_check(self) -> Dict[str, Any]:
        """执行健康检查并返回详细报告"""
        status = self._perform_health_check()
        
        report = {
            'status': status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # 设备检查
        report['checks']['device'] = {
            'type': self._device.type,
            'available': True
        }
        
        if self._device.type == "cuda":
            try:
                report['checks']['device']['cuda_available'] = torch.cuda.is_available()
                report['checks']['device']['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                report['checks']['device']['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            except Exception as e:
                report['checks']['device']['error'] = str(e)
        
        # 缓存检查
        total_cached = (len(self._encoder_cache) + len(self._fusion_cache) + 
                       len(self._alignment_cache) + len(self._adapter_cache))
        report['checks']['cache'] = {
            'total_cached': total_cached,
            'max_size': self.config.max_cache_size,
            'utilization': total_cached / self.config.max_cache_size if self.config.max_cache_size > 0 else 0
        }
        
        # 错误率检查
        if self._global_metrics["total_inferences"] > 0:
            error_rate = self._global_metrics["total_errors"] / self._global_metrics["total_inferences"]
            report['checks']['error_rate'] = {
                'rate': error_rate,
                'total_errors': self._global_metrics["total_errors"],
                'total_inferences': self._global_metrics["total_inferences"]
            }
        
        return report
    
    @staticmethod
    def list_available_components() -> Dict[str, List[str]]:
        """列出所有可用组件"""
        return {
            'modalities': [m.value for m in ModalityType],
            'fusion_methods': [f.value for f in FusionMethod],
            'alignment_methods': [a.value for a in AlignmentMethod],
            'adapter_types': [t.value for t in AdapterType]
        }
    
    # ==================== 检查点管理 ====================
    
    def save_checkpoint(self, checkpoint_path: str = None) -> str:
        """
        保存检查点
        
        Args:
            checkpoint_path: 检查点路径，如果为None则使用默认路径
            
        Returns:
            保存的检查点路径
        """
        if checkpoint_path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"adapter_manager_{timestamp}.pt")
        
        checkpoint = {
            'config': self.config.to_dict(),
            'encoders': {},
            'fusions': {},
            'alignments': {},
            'adapters': {},
            'metrics': {
                'global': {
                    **self._global_metrics,
                    'startup_time': self._global_metrics['startup_time'].isoformat()
                },
                'components': {k: v.to_dict() for k, v in self._component_metrics.items()},
                'pipelines': {k: v.to_dict() for k, v in self._pipeline_metrics.items()}
            }
        }
        
        # 保存组件状态
        for key, encoder in self._encoder_cache.items():
            checkpoint['encoders'][key] = encoder.state_dict()
        
        for key, fusion in self._fusion_cache.items():
            checkpoint['fusions'][key] = fusion.state_dict()
        
        for key, alignment in self._alignment_cache.items():
            checkpoint['alignments'][key] = alignment.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            是否成功加载
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            
            # 恢复配置
            if 'config' in checkpoint:
                self.config = AdapterManagerConfig.from_dict(checkpoint['config'])
            
            # 注意：实际恢复组件状态需要先创建组件，这里仅记录日志
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Cached components in checkpoint: encoders={len(checkpoint.get('encoders', {}))}, "
                       f"fusions={len(checkpoint.get('fusions', {}))}, "
                       f"alignments={len(checkpoint.get('alignments', {}))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    # ==================== 生命周期管理 ====================
    
    def shutdown(self) -> None:
        """关闭管理器"""
        self._status = ManagerStatus.SHUTTING_DOWN
        logger.info("AdapterManager shutting down...")
        
        # 保存检查点（如果启用）
        if self.config.auto_checkpoint:
            try:
                self.save_checkpoint()
            except Exception as e:
                logger.error(f"Failed to save checkpoint on shutdown: {e}")
        
        # 清理缓存
        self.clear_cache()
        
        self._status = ManagerStatus.IDLE
        logger.info("AdapterManager shutdown complete")
    
    # ==================== 质量分析功能 ====================
    
    def analyze_encoder_quality(
        self,
        modality: Union[ModalityType, str],
        inputs: torch.Tensor,
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析编码器质量
        
        Args:
            modality: 模态类型
            inputs: 输入张量
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            质量分析报告
        """
        encoder = self.get_encoder(modality, hidden_size, **kwargs)
        
        # 使用编码器质量分析器
        analyzer = EncoderQualityAnalyzer()
        analysis = analyzer.analyze(encoder, inputs.to(self._device), **kwargs)
        
        # 记录指标
        self._record_component_metric("encoder_analysis", str(modality), 0.0)
        
        return analysis
    
    def analyze_fusion_quality(
        self,
        features: List[torch.Tensor],
        method: Union[FusionMethod, str] = None,
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析融合质量
        
        Args:
            features: 特征列表
            method: 融合方法
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            质量分析报告
        """
        method = method or self.config.default_fusion_method
        fusion = self.get_fusion(method, hidden_size, **kwargs)
        
        # 将特征移到设备上
        features_device = [f.to(self._device) for f in features]
        
        # 执行融合
        with torch.no_grad():
            fused = fusion(features_device)
        
        # 使用融合质量分析器
        analyzer = FusionQualityAnalyzer()
        analysis = analyzer.analyze(features_device, fused)
        
        # 记录指标
        self._record_component_metric("fusion_analysis", str(method), 0.0)
        
        return analysis
    
    def analyze_alignment_quality(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        method: Union[AlignmentMethod, str] = None,
        hidden_size: Optional[int] = None,
        metric: AlignmentQualityMetric = AlignmentQualityMetric.COSINE_SIMILARITY,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析对齐质量
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            method: 对齐方法
            hidden_size: 隐藏层大小
            metric: 质量指标类型
            **kwargs: 其他配置
            
        Returns:
            质量分析报告
        """
        method = method or self.config.default_alignment_method
        alignment = self.get_alignment(method, hidden_size, **kwargs)
        
        # 将特征移到设备上
        fa = features_a.to(self._device)
        fb = features_b.to(self._device)
        
        # 评估对齐质量
        quality = alignment.evaluate_quality(fa, fb, metric)
        
        # 获取对齐模块的指标
        alignment_metrics = alignment.get_metrics()
        
        # 记录指标
        self._record_component_metric("alignment_analysis", str(method), 0.0)
        
        return {
            "quality": quality,
            "alignment_metrics": alignment_metrics,
            "method": method.value if isinstance(method, AlignmentMethod) else method,
            "metric_type": metric.value
        }
    
    def analyze_adapter_quality(
        self,
        adapter: ModelAdapter
    ) -> Dict[str, Any]:
        """
        分析适配器质量
        
        Args:
            adapter: 适配器实例
            
        Returns:
            质量分析报告
        """
        analyzer = AdapterQualityAnalyzer()
        analysis = analyzer.analyze_adapter(adapter)
        
        # 添加详细摘要
        summary = get_adapter_summary(adapter)
        analysis["detailed_summary"] = summary
        
        return analysis
    
    # ==================== 适配器融合和合并功能 ====================
    
    def create_adapter_fusion(
        self,
        adapter_types: List[Union[AdapterType, str]],
        fusion_method: str = "attention",
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> AdapterFusion:
        """
        创建适配器融合
        
        Args:
            adapter_types: 适配器类型列表
            fusion_method: 融合方法（attention/gated/average）
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            适配器融合模块
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 创建适配器列表
        adapters = []
        for adapter_type in adapter_types:
            adapter = self.get_adapter(adapter_type, hidden_size, **kwargs)
            adapters.append(adapter)
        
        # 创建融合
        fusion = create_adapter_fusion(adapters, fusion_method, hidden_size)
        
        logger.info(f"Created adapter fusion with {len(adapters)} adapters, method={fusion_method}")
        return fusion
    
    def merge_adapters(
        self,
        adapter_types: List[Union[AdapterType, str]],
        weights: Optional[List[float]] = None,
        strategy: MergeStrategy = MergeStrategy.LINEAR,
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        合并多个适配器
        
        Args:
            adapter_types: 适配器类型列表
            weights: 权重列表
            strategy: 合并策略
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            合并后的参数字典
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 创建适配器列表
        adapters = []
        for adapter_type in adapter_types:
            adapter = self.get_adapter(adapter_type, hidden_size, **kwargs)
            adapters.append(adapter)
        
        # 执行合并
        merged = merge_adapters(adapters, weights, strategy)
        
        logger.info(f"Merged {len(adapters)} adapters using {strategy.value} strategy")
        return merged
    
    def compare_adapters(
        self,
        adapter_types: List[Union[AdapterType, str]],
        hidden_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        比较多个适配器
        
        Args:
            adapter_types: 适配器类型列表
            hidden_size: 隐藏层大小
            **kwargs: 其他配置
            
        Returns:
            比较结果
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        adapters = []
        for adapter_type in adapter_types:
            adapter = self.get_adapter(adapter_type, hidden_size, **kwargs)
            adapters.append(adapter)
        
        analyzer = AdapterQualityAnalyzer()
        comparison = analyzer.compare_adapters(adapters)
        
        return comparison
    
    # ==================== 高级管道功能 ====================
    
    def create_multimodal_encoder(
        self,
        modalities: List[Union[ModalityType, str]],
        hidden_size: Optional[int] = None,
        fusion_method: str = "attention",
        **kwargs
    ) -> MultiModalEncoder:
        """
        创建多模态编码器
        
        Args:
            modalities: 模态列表
            hidden_size: 隐藏层大小
            fusion_method: 融合方法
            **kwargs: 其他配置
            
        Returns:
            多模态编码器
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        # 为每个模态创建配置
        modality_configs = {}
        for modality in modalities:
            if isinstance(modality, str):
                modality = ModalityType(modality)
            
            config = (build_encoder_config(modality)
                .hidden_size(hidden_size)
                .enable_metrics(self.config.enable_metrics)
                .build())
            modality_configs[modality] = config
        
        # 创建多模态编码器
        encoder = create_multimodal_encoder(
            modalities, 
            hidden_size, 
            fusion_method, 
            **kwargs
        )
        encoder = encoder.to(self._device)
        
        logger.info(f"Created multimodal encoder: modalities={[str(m) for m in modalities]}, fusion={fusion_method}")
        return encoder
    
    def create_hybrid_fusion_pipeline(
        self,
        modalities: List[str],
        fusion_methods: List[str],
        fusion_weights: Optional[List[float]] = None,
        alignment_method: str = None,
        hidden_size: int = None,
        pipeline_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建混合融合管道
        
        Args:
            modalities: 模态列表
            fusion_methods: 融合方法列表（用于混合融合）
            fusion_weights: 融合权重
            alignment_method: 对齐方法
            hidden_size: 隐藏层大小
            pipeline_id: 管道ID
            **kwargs: 其他配置
            
        Returns:
            管道字典
        """
        alignment_method = alignment_method or self.config.default_alignment_method
        hidden_size = hidden_size or self.config.default_hidden_size
        pipeline_id = pipeline_id or f"hybrid_{'_'.join(modalities)}_{'_'.join(fusion_methods)}"
        
        # 检查缓存
        if pipeline_id in self._pipeline_cache:
            logger.debug(f"Hybrid pipeline cache hit: {pipeline_id}")
            return self._pipeline_cache[pipeline_id]
        
        start_time = time.time()
        
        try:
            # 创建编码器
            encoders = self.create_multimodal_encoders(modalities, hidden_size)
            
            # 创建混合融合
            hybrid_fusion = create_hybrid_fusion(
                fusion_methods, 
                fusion_weights, 
                hidden_size, 
                **kwargs
            )
            hybrid_fusion = hybrid_fusion.to(self._device)
            
            # 创建对齐模块
            alignment = self.get_alignment(alignment_method, hidden_size) if len(modalities) > 1 else None
            
            pipeline = {
                'id': pipeline_id,
                'encoders': encoders,
                'fusion': hybrid_fusion,
                'fusion_type': 'hybrid',
                'fusion_methods': fusion_methods,
                'alignment': alignment,
                'config': {
                    'modalities': modalities,
                    'fusion_methods': fusion_methods,
                    'fusion_weights': fusion_weights,
                    'alignment_method': alignment_method,
                    'hidden_size': hidden_size
                },
                'created_at': datetime.now().isoformat()
            }
            
            # 缓存
            self._pipeline_cache[pipeline_id] = pipeline
            
            # 记录指标
            if self.config.enable_metrics:
                self._pipeline_metrics[pipeline_id] = PipelineMetrics(
                    pipeline_id=pipeline_id,
                    modalities=modalities
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Created hybrid fusion pipeline: {pipeline_id} in {elapsed:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create hybrid pipeline {pipeline_id}: {e}")
            raise
    
    def create_advanced_adapter_pipeline(
        self,
        modalities: List[str],
        adapter_types: List[Union[AdapterType, str]],
        fusion_method: str = None,
        adapter_fusion_method: str = "attention",
        hidden_size: int = None,
        pipeline_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建高级适配器管道（包含适配器融合）
        
        Args:
            modalities: 模态列表
            adapter_types: 适配器类型列表
            fusion_method: 特征融合方法
            adapter_fusion_method: 适配器融合方法
            hidden_size: 隐藏层大小
            pipeline_id: 管道ID
            **kwargs: 其他配置
            
        Returns:
            管道字典
        """
        fusion_method = fusion_method or self.config.default_fusion_method
        hidden_size = hidden_size or self.config.default_hidden_size
        adapter_types_str = '_'.join([str(a) for a in adapter_types])
        pipeline_id = pipeline_id or f"advanced_{'_'.join(modalities)}_{adapter_types_str}"
        
        start_time = time.time()
        
        try:
            # 创建编码器
            encoders = self.create_multimodal_encoders(modalities, hidden_size)
            
            # 创建特征融合
            fusion = self.get_fusion(fusion_method, hidden_size, **kwargs)
            
            # 创建适配器融合
            adapter_fusion = self.create_adapter_fusion(
                adapter_types, 
                adapter_fusion_method, 
                hidden_size, 
                **kwargs
            )
            
            pipeline = {
                'id': pipeline_id,
                'encoders': encoders,
                'fusion': fusion,
                'adapter_fusion': adapter_fusion,
                'adapters': list(adapter_fusion.adapters),
                'config': {
                    'modalities': modalities,
                    'adapter_types': [str(a) for a in adapter_types],
                    'fusion_method': fusion_method,
                    'adapter_fusion_method': adapter_fusion_method,
                    'hidden_size': hidden_size
                },
                'created_at': datetime.now().isoformat()
            }
            
            # 缓存
            self._pipeline_cache[pipeline_id] = pipeline
            
            elapsed = time.time() - start_time
            logger.info(f"Created advanced adapter pipeline: {pipeline_id} in {elapsed:.2f}s")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create advanced adapter pipeline {pipeline_id}: {e}")
            raise
    
    # ==================== 综合诊断功能 ====================
    
    def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """
        运行综合诊断
        
        Returns:
            诊断报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "manager_status": self._status.value,
            "health_status": self._health_status.value,
            "device": str(self._device),
            "diagnostics": {}
        }
        
        # 1. 编码器工厂诊断
        try:
            encoder_health = encoder_factory_health_check()
            report["diagnostics"]["encoder_factory"] = encoder_health
        except Exception as e:
            report["diagnostics"]["encoder_factory"] = {"error": str(e)}
        
        # 2. 融合工厂诊断
        try:
            fusion_metrics = FusionFactory.get_factory_metrics()
            report["diagnostics"]["fusion_factory"] = {
                "metrics": fusion_metrics,
                "available_methods": FusionFactory.get_available_methods()
            }
        except Exception as e:
            report["diagnostics"]["fusion_factory"] = {"error": str(e)}
        
        # 3. 对齐工厂诊断
        try:
            alignment_metrics = AlignmentFactory.get_factory_metrics()
            report["diagnostics"]["alignment_factory"] = {
                "metrics": alignment_metrics,
                "available_methods": AlignmentFactory.get_available_methods()
            }
        except Exception as e:
            report["diagnostics"]["alignment_factory"] = {"error": str(e)}
        
        # 4. 适配器工厂诊断
        try:
            adapter_health = adapter_factory_health_check()
            report["diagnostics"]["adapter_factory"] = adapter_health
        except Exception as e:
            report["diagnostics"]["adapter_factory"] = {"error": str(e)}
        
        # 5. 缓存诊断
        report["diagnostics"]["cache"] = {
            "encoder_cache_size": len(self._encoder_cache),
            "fusion_cache_size": len(self._fusion_cache),
            "alignment_cache_size": len(self._alignment_cache),
            "adapter_cache_size": len(self._adapter_cache),
            "pipeline_cache_size": len(self._pipeline_cache),
            "max_cache_size": self.config.max_cache_size
        }
        
        # 6. 内存诊断
        if self._device.type == "cuda":
            try:
                report["diagnostics"]["memory"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "max_memory_gb": self.config.max_memory_gb,
                    "utilization": torch.cuda.memory_allocated() / (1024**3) / self.config.max_memory_gb
                }
            except Exception as e:
                report["diagnostics"]["memory"] = {"error": str(e)}
        
        # 7. 组件指标汇总
        report["diagnostics"]["component_metrics_summary"] = {
            "total_components": len(self._component_metrics),
            "total_pipelines": len(self._pipeline_metrics),
            "global_metrics": {
                **self._global_metrics,
                "startup_time": self._global_metrics["startup_time"].isoformat()
            }
        }
        
        return report
    
    def get_component_health(self, component_type: str, component_key: str) -> Dict[str, Any]:
        """
        获取特定组件的健康状态
        
        Args:
            component_type: 组件类型（encoder/fusion/alignment/adapter）
            component_key: 组件键
            
        Returns:
            组件健康状态
        """
        health = {
            "component_type": component_type,
            "component_key": component_key,
            "status": "unknown",
            "details": {}
        }
        
        # 获取组件指标
        metrics = self.get_component_metrics(component_type, component_key)
        if metrics:
            health["details"]["metrics"] = metrics
            
            # 评估健康状态
            error_rate = metrics.get("error_count", 0) / max(metrics.get("use_count", 1), 1)
            if error_rate == 0:
                health["status"] = "healthy"
            elif error_rate < 0.1:
                health["status"] = "degraded"
            else:
                health["status"] = "unhealthy"
            
            health["details"]["error_rate"] = error_rate
        
        # 获取缓存中的组件
        component = None
        if component_type == "encoder" and component_key in self._encoder_cache:
            component = self._encoder_cache[component_key]
            health["details"]["cached"] = True
            health["details"]["status"] = component.get_status().value if hasattr(component, 'get_status') else "unknown"
        elif component_type == "fusion" and component_key in self._fusion_cache:
            component = self._fusion_cache[component_key]
            health["details"]["cached"] = True
            health["details"]["status"] = component.status.value if hasattr(component, 'status') else "unknown"
        elif component_type == "alignment" and component_key in self._alignment_cache:
            component = self._alignment_cache[component_key]
            health["details"]["cached"] = True
            health["details"]["status"] = component.status.value if hasattr(component, 'status') else "unknown"
        elif component_type == "adapter" and component_key in self._adapter_cache:
            component = self._adapter_cache[component_key]
            health["details"]["cached"] = True
            health["details"]["status"] = component.get_status().value if hasattr(component, 'get_status') else "unknown"
        else:
            health["details"]["cached"] = False
        
        return health
    
    def optimize_cache(self, target_utilization: float = 0.8) -> Dict[str, Any]:
        """
        优化缓存
        
        Args:
            target_utilization: 目标缓存利用率
            
        Returns:
            优化结果
        """
        result = {
            "before": {
                "total_cached": (len(self._encoder_cache) + len(self._fusion_cache) + 
                               len(self._alignment_cache) + len(self._adapter_cache)),
                "max_size": self.config.max_cache_size
            },
            "actions": [],
            "after": {}
        }
        
        current_total = result["before"]["total_cached"]
        target_size = int(self.config.max_cache_size * target_utilization)
        
        if current_total > target_size:
            # 需要清理
            to_remove = current_total - target_size
            cleaned = self._cleanup_least_used_components(keep_count=target_size // 4)
            result["actions"].append(f"Cleaned {cleaned} least-used components")
        
        # 检查内存
        if self._check_memory_and_cleanup():
            result["actions"].append("Performed memory cleanup")
        
        result["after"] = {
            "total_cached": (len(self._encoder_cache) + len(self._fusion_cache) + 
                           len(self._alignment_cache) + len(self._adapter_cache)),
            "max_size": self.config.max_cache_size
        }
        
        return result
    
    # ==================== 特定组件创建方法 ====================
    
    def create_graph_encoder(
        self,
        hidden_size: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 8,
        **kwargs
    ) -> GraphEncoder:
        """
        创建图编码器
        
        Args:
            hidden_size: 隐藏层大小
            num_layers: 层数
            num_heads: 注意力头数
            **kwargs: 其他配置
            
        Returns:
            图编码器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = EncoderConfig(
            modality=ModalityType.GRAPH,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            enable_metrics=self.config.enable_metrics
        )
        
        encoder = GraphEncoder(config)
        encoder = encoder.to(self._device)
        
        # 记录状态
        status = EncoderStatus.READY
        logger.debug(f"Created GraphEncoder: hidden_size={hidden_size}, status={status.value}")
        
        return encoder
    
    def create_point_cloud_encoder(
        self,
        hidden_size: Optional[int] = None,
        num_points: int = 1024,
        point_dim: int = 3,
        **kwargs
    ) -> PointCloudEncoder:
        """
        创建点云编码器
        
        Args:
            hidden_size: 隐藏层大小
            num_points: 点数
            point_dim: 点维度
            **kwargs: 其他配置
            
        Returns:
            点云编码器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = EncoderConfig(
            modality=ModalityType.POINT_CLOUD,
            hidden_size=hidden_size,
            num_points=num_points,
            point_dim=point_dim,
            enable_metrics=self.config.enable_metrics
        )
        
        encoder = PointCloudEncoder(config)
        encoder = encoder.to(self._device)
        
        logger.debug(f"Created PointCloudEncoder: hidden_size={hidden_size}, num_points={num_points}")
        return encoder
    
    def create_tensor_fusion(
        self,
        hidden_size: Optional[int] = None,
        tensor_rank: int = 16,
        num_modalities: int = 2,
        **kwargs
    ) -> TensorFusion:
        """
        创建张量融合模块
        
        Args:
            hidden_size: 隐藏层大小
            tensor_rank: 张量秩
            num_modalities: 模态数量
            **kwargs: 其他配置
            
        Returns:
            张量融合模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = FusionConfig(
            method=FusionMethod.TENSOR,
            hidden_size=hidden_size,
            tensor_rank=tensor_rank,
            enable_metrics=self.config.enable_metrics
        )
        
        fusion = TensorFusion(config, num_modalities=num_modalities)
        fusion = fusion.to(self._device)
        
        # 记录状态
        status = FusionStatus.READY
        logger.debug(f"Created TensorFusion: rank={tensor_rank}, status={status.value}")
        
        return fusion
    
    def create_bilinear_fusion(
        self,
        hidden_size: Optional[int] = None,
        output_dim: int = 256,
        use_low_rank: bool = False,
        **kwargs
    ) -> BilinearFusion:
        """
        创建双线性融合模块
        
        Args:
            hidden_size: 隐藏层大小
            output_dim: 输出维度
            use_low_rank: 是否使用低秩分解
            **kwargs: 其他配置
            
        Returns:
            双线性融合模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = FusionConfig(
            method=FusionMethod.BILINEAR,
            hidden_size=hidden_size,
            bilinear_output_dim=output_dim,
            enable_metrics=self.config.enable_metrics
        )
        
        fusion = BilinearFusion(config, use_low_rank=use_low_rank)
        fusion = fusion.to(self._device)
        
        logger.debug(f"Created BilinearFusion: output_dim={output_dim}, low_rank={use_low_rank}")
        return fusion
    
    def create_hybrid_fusion_module(
        self,
        hidden_size: Optional[int] = None,
        methods: List[str] = None,
        weights: List[float] = None,
        **kwargs
    ) -> HybridFusion:
        """
        创建混合融合模块
        
        Args:
            hidden_size: 隐藏层大小
            methods: 融合方法列表
            weights: 权重列表
            **kwargs: 其他配置
            
        Returns:
            混合融合模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        methods = methods or ["early", "late"]
        
        config = FusionConfig(
            hidden_size=hidden_size,
            enable_metrics=self.config.enable_metrics
        )
        
        fusion = HybridFusion(config, methods=methods, weights=weights)
        fusion = fusion.to(self._device)
        
        logger.debug(f"Created HybridFusion: methods={methods}")
        return fusion
    
    def create_cca_alignment(
        self,
        hidden_size: Optional[int] = None,
        cca_components: int = 64,
        cca_reg: float = 1e-4,
        **kwargs
    ) -> CCAAlignment:
        """
        创建CCA对齐模块
        
        Args:
            hidden_size: 隐藏层大小
            cca_components: CCA分量数
            cca_reg: 正则化系数
            **kwargs: 其他配置
            
        Returns:
            CCA对齐模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = AlignmentConfig(
            method=AlignmentMethod.CCA,
            hidden_size=hidden_size,
            cca_components=cca_components,
            cca_reg=cca_reg,
            enable_metrics=self.config.enable_metrics
        )
        
        alignment = CCAAlignment(config)
        alignment = alignment.to(self._device)
        
        # 记录状态
        status = AlignmentStatus.READY
        logger.debug(f"Created CCAAlignment: components={cca_components}, status={status.value}")
        
        return alignment
    
    def create_hybrid_alignment(
        self,
        hidden_size: Optional[int] = None,
        methods: List[str] = None,
        weights: List[float] = None,
        **kwargs
    ) -> HybridAlignment:
        """
        创建混合对齐模块
        
        Args:
            hidden_size: 隐藏层大小
            methods: 对齐方法列表
            weights: 权重列表
            **kwargs: 其他配置
            
        Returns:
            混合对齐模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        methods = methods or ["contrastive", "explicit"]
        weights = weights or [0.5, 0.5]
        
        config = AlignmentConfig(
            method=AlignmentMethod.HYBRID,
            hidden_size=hidden_size,
            hybrid_methods=methods,
            hybrid_weights=weights,
            enable_metrics=self.config.enable_metrics
        )
        
        alignment = HybridAlignment(config)
        alignment = alignment.to(self._device)
        
        logger.debug(f"Created HybridAlignment: methods={methods}")
        return alignment
    
    def create_lora_adapter(
        self,
        hidden_size: Optional[int] = None,
        rank: int = 8,
        alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        use_dora: bool = False,
        use_rslora: bool = False,
        **kwargs
    ) -> LoRAAdapter:
        """
        创建LoRA适配器
        
        Args:
            hidden_size: 隐藏层大小
            rank: LoRA秩
            alpha: LoRA alpha
            target_modules: 目标模块
            use_dora: 是否使用DoRA
            use_rslora: 是否使用RS-LoRA
            **kwargs: 其他配置
            
        Returns:
            LoRA适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        adapter = create_lora_adapter(
            hidden_size=hidden_size,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            use_dora=use_dora,
            use_rslora=use_rslora,
            **kwargs
        )
        
        # 记录状态
        status = AdapterStatus.READY
        logger.debug(f"Created LoRAAdapter: rank={rank}, alpha={alpha}, status={status.value}")
        
        return adapter
    
    def create_adapter_layers(
        self,
        hidden_size: Optional[int] = None,
        bottleneck_size: int = 64,
        **kwargs
    ) -> AdapterLayersAdapter:
        """
        创建Adapter Layers适配器
        
        Args:
            hidden_size: 隐藏层大小
            bottleneck_size: 瓶颈大小
            **kwargs: 其他配置
            
        Returns:
            Adapter Layers适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = AdapterConfig(
            adapter_type=AdapterType.ADAPTER,
            hidden_size=hidden_size,
            adapter_bottleneck=bottleneck_size,
            enable_metrics=self.config.enable_metrics
        )
        
        adapter = AdapterLayersAdapter(config)
        logger.debug(f"Created AdapterLayersAdapter: bottleneck={bottleneck_size}")
        
        return adapter
    
    def create_bitfit_adapter(
        self,
        hidden_size: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ) -> BitFitAdapter:
        """
        创建BitFit适配器
        
        Args:
            hidden_size: 隐藏层大小
            target_modules: 目标模块
            **kwargs: 其他配置
            
        Returns:
            BitFit适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = AdapterConfig(
            adapter_type=AdapterType.BITFIT,
            hidden_size=hidden_size,
            enable_metrics=self.config.enable_metrics
        )
        
        adapter = BitFitAdapter(config, target_modules=target_modules)
        logger.debug(f"Created BitFitAdapter: hidden_size={hidden_size}")
        
        return adapter
    
    def create_ia3_adapter(
        self,
        hidden_size: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ) -> IA3Adapter:
        """
        创建IA3适配器
        
        Args:
            hidden_size: 隐藏层大小
            target_modules: 目标模块
            **kwargs: 其他配置
            
        Returns:
            IA3适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = AdapterConfig(
            adapter_type=AdapterType.IA3,
            hidden_size=hidden_size,
            ia3_target_modules=target_modules,
            enable_metrics=self.config.enable_metrics
        )
        
        adapter = IA3Adapter(config)
        logger.debug(f"Created IA3Adapter: hidden_size={hidden_size}")
        
        return adapter
    
    def create_compacter_adapter(
        self,
        hidden_size: Optional[int] = None,
        rank: int = 4,
        **kwargs
    ) -> CompacterAdapter:
        """
        创建Compacter适配器
        
        Args:
            hidden_size: 隐藏层大小
            rank: 秩
            **kwargs: 其他配置
            
        Returns:
            Compacter适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        config = AdapterConfig(
            adapter_type=AdapterType.COMPACTER,
            hidden_size=hidden_size,
            enable_metrics=self.config.enable_metrics
        )
        
        adapter = CompacterAdapter(config, rank=rank)
        logger.debug(f"Created CompacterAdapter: rank={rank}")
        
        return adapter
    
    # ==================== 详细指标获取方法 ====================
    
    def get_encoder_detailed_metrics(self, encoder: ModalityEncoder) -> Dict[str, Any]:
        """
        获取编码器详细指标
        
        Args:
            encoder: 编码器实例
            
        Returns:
            详细指标字典
        """
        metrics = encoder.get_metrics()
        
        # 使用 EncoderMetrics
        if hasattr(encoder, '_metrics') and encoder._metrics:
            raw_metrics: EncoderMetrics = encoder._metrics
            metrics["encoder_metrics"] = {
                "total_encodings": raw_metrics.total_encodings,
                "avg_time": raw_metrics.avg_time,
                "error_count": raw_metrics.error_count,
                "recent_quality": raw_metrics.get_recent_quality(),
                "feature_stats": raw_metrics.feature_stats
            }
        
        # 添加参数统计
        metrics["parameter_count"] = encoder.get_parameter_count()
        metrics["memory_usage"] = encoder.get_memory_usage()
        
        return metrics
    
    def get_fusion_detailed_metrics(self, fusion: FusionModule) -> Dict[str, Any]:
        """
        获取融合模块详细指标
        
        Args:
            fusion: 融合模块实例
            
        Returns:
            详细指标字典
        """
        metrics = fusion.get_metrics()
        
        # 使用 FusionMetrics
        if hasattr(fusion, '_metrics') and fusion._metrics:
            raw_metrics: FusionMetrics = fusion._metrics
            metrics["fusion_metrics"] = {
                "total_fusions": raw_metrics.total_fusions,
                "avg_time": raw_metrics.avg_time,
                "error_count": raw_metrics.error_count,
                "modality_counts": raw_metrics.modality_counts
            }
        
        # 添加参数统计
        metrics["parameter_count"] = fusion.get_parameter_count()
        metrics["memory_usage"] = fusion.get_memory_usage()
        
        return metrics
    
    def get_alignment_detailed_metrics(self, alignment: AlignmentModule) -> Dict[str, Any]:
        """
        获取对齐模块详细指标
        
        Args:
            alignment: 对齐模块实例
            
        Returns:
            详细指标字典
        """
        metrics = alignment.get_metrics()
        
        # 使用 AlignmentMetrics
        if hasattr(alignment, '_metrics') and alignment._metrics:
            raw_metrics: AlignmentMetrics = alignment._metrics
            metrics["alignment_metrics"] = {
                "total_alignments": raw_metrics.total_alignments,
                "avg_time": raw_metrics.avg_time,
                "avg_similarity": raw_metrics.avg_similarity,
                "avg_loss": raw_metrics.avg_loss,
                "error_count": raw_metrics.error_count,
                "recent_quality": raw_metrics.get_recent_quality()
            }
        
        return metrics
    
    def get_adapter_detailed_metrics(self, adapter: ModelAdapter) -> Dict[str, Any]:
        """
        获取适配器详细指标
        
        Args:
            adapter: 适配器实例
            
        Returns:
            详细指标字典
        """
        metrics = adapter.get_metrics()
        
        # 使用 AdapterMetrics
        if hasattr(adapter, '_metrics') and adapter._metrics:
            raw_metrics: AdapterMetrics = adapter._metrics
            metrics["adapter_metrics"] = {
                "total_adaptations": raw_metrics.total_adaptations,
                "avg_time": raw_metrics.avg_time,
                "error_count": raw_metrics.error_count,
                "parameter_stats": raw_metrics.parameter_stats,
                "gradient_stats": raw_metrics.gradient_stats
            }
        
        # 使用 get_adapter_summary
        metrics["summary"] = get_adapter_summary(adapter)
        
        return metrics
    
    # ==================== 配置构建方法 ====================
    
    def build_encoder_with_builder(
        self,
        modality: Union[ModalityType, str],
        hidden_size: Optional[int] = None,
        pooling: Union[EncoderPoolingMethod, str] = EncoderPoolingMethod.MEAN,
        augmentation: Union[EncoderAugmentationType, str] = EncoderAugmentationType.NONE,
        **kwargs
    ) -> ModalityEncoder:
        """
        使用配置构建器创建编码器
        
        Args:
            modality: 模态类型
            hidden_size: 隐藏层大小
            pooling: 池化方法
            augmentation: 数据增强类型
            **kwargs: 其他配置
            
        Returns:
            编码器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        if isinstance(modality, str):
            modality = ModalityType(modality)
        if isinstance(pooling, str):
            pooling = EncoderPoolingMethod(pooling)
        if isinstance(augmentation, str):
            augmentation = EncoderAugmentationType(augmentation)
        
        # 使用 EncoderConfigBuilder
        builder = build_encoder_config(modality)
        builder.hidden_size(hidden_size)
        builder.pooling(pooling)
        builder.augmentation(augmentation)
        builder.enable_metrics(self.config.enable_metrics)
        
        config = builder.build()
        
        # 使用 EncoderFactory
        encoder = EncoderFactory.create(modality, config)
        encoder = encoder.to(self._device)
        
        logger.debug(f"Built encoder with builder: modality={modality.value}")
        return encoder
    
    def build_fusion_with_builder(
        self,
        method: Union[FusionMethod, str],
        hidden_size: Optional[int] = None,
        pooling: Union[FusionPoolingType, str] = FusionPoolingType.MEAN,
        norm: Union[NormType, str] = NormType.LAYER_NORM,
        **kwargs
    ) -> FusionModule:
        """
        使用配置构建器创建融合模块
        
        Args:
            method: 融合方法
            hidden_size: 隐藏层大小
            pooling: 池化类型
            norm: 归一化类型
            **kwargs: 其他配置
            
        Returns:
            融合模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        if isinstance(method, str):
            method = FusionMethod(method)
        if isinstance(pooling, str):
            pooling = FusionPoolingType(pooling)
        if isinstance(norm, str):
            norm = NormType(norm)
        
        # 使用 FusionConfigBuilder
        builder = (build_fusion_config()
            .with_method(method)
            .with_hidden_size(hidden_size)
            .with_pooling(pooling)
            .with_norm(norm)
            .with_metrics(self.config.enable_metrics))
        
        config = builder.build()
        
        # 使用 FusionFactory
        fusion = FusionFactory.create(method, config)
        fusion = fusion.to(self._device)
        
        logger.debug(f"Built fusion with builder: method={method.value}")
        return fusion
    
    def build_alignment_with_builder(
        self,
        method: Union[AlignmentMethod, str],
        hidden_size: Optional[int] = None,
        pooling: Union[AlignmentPoolingMethod, str] = AlignmentPoolingMethod.MEAN,
        loss_type: Union[LossType, str] = LossType.INFONCE,
        augmentation: Union[AlignmentAugmentationType, str] = AlignmentAugmentationType.NONE,
        **kwargs
    ) -> AlignmentModule:
        """
        使用配置构建器创建对齐模块
        
        Args:
            method: 对齐方法
            hidden_size: 隐藏层大小
            pooling: 池化方法
            loss_type: 损失类型
            augmentation: 数据增强类型
            **kwargs: 其他配置
            
        Returns:
            对齐模块实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        if isinstance(method, str):
            method = AlignmentMethod(method)
        if isinstance(pooling, str):
            pooling = AlignmentPoolingMethod(pooling)
        if isinstance(loss_type, str):
            loss_type = LossType(loss_type)
        if isinstance(augmentation, str):
            augmentation = AlignmentAugmentationType(augmentation)
        
        # 使用 AlignmentConfigBuilder
        builder = (build_alignment_config()
            .with_method(method)
            .with_hidden_size(hidden_size)
            .with_pooling(pooling)
            .with_augmentation(augmentation)
            .with_metrics(self.config.enable_metrics))
        
        config = builder.build()
        
        # 使用 AlignmentFactory
        alignment = AlignmentFactory.create(method, config)
        alignment = alignment.to(self._device)
        
        logger.debug(f"Built alignment with builder: method={method.value}")
        return alignment
    
    def build_adapter_with_builder(
        self,
        adapter_type: Union[AdapterType, str],
        hidden_size: Optional[int] = None,
        init_strategy: Union[InitStrategy, str] = InitStrategy.KAIMING,
        **kwargs
    ) -> ModelAdapter:
        """
        使用配置构建器创建适配器
        
        Args:
            adapter_type: 适配器类型
            hidden_size: 隐藏层大小
            init_strategy: 初始化策略
            **kwargs: 其他配置
            
        Returns:
            适配器实例
        """
        hidden_size = hidden_size or self.config.default_hidden_size
        
        if isinstance(adapter_type, str):
            adapter_type = AdapterType(adapter_type)
        if isinstance(init_strategy, str):
            init_strategy = InitStrategy(init_strategy)
        
        # 使用 AdapterConfigBuilder
        builder = (build_adapter_config(adapter_type)
            .hidden_size(hidden_size)
            .init_strategy(init_strategy)
            .enable_metrics(self.config.enable_metrics))
        
        config = builder.build()
        
        # 使用 AdapterFactory
        adapter = AdapterFactory.create(adapter_type, config)
        
        logger.debug(f"Built adapter with builder: type={adapter_type.value}")
        return adapter
    
    # ==================== LoRA层分析方法 ====================
    
    def analyze_lora_layers(self, lora_adapter: LoRAAdapter) -> Dict[str, Any]:
        """
        分析LoRA适配器的层
        
        Args:
            lora_adapter: LoRA适配器实例
            
        Returns:
            层分析结果
        """
        analysis = {
            "num_lora_layers": len(lora_adapter.lora_layers),
            "layers": {},
            "total_params": 0
        }
        
        for name, lora_layer in lora_adapter.lora_layers.items():
            # 使用 LoRALayer 的方法
            layer: LoRALayer = lora_layer
            layer_info = {
                "rank": layer.rank,
                "alpha": layer.alpha,
                "scaling": layer.scaling,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "use_dora": layer.use_dora,
                "use_rslora": layer.use_rslora,
                "effective_rank": layer.get_effective_rank(),
                "layer_analysis": layer.get_layer_analysis()
            }
            analysis["layers"][name] = layer_info
            analysis["total_params"] += (layer.in_features * layer.rank + layer.rank * layer.out_features)
        
        return analysis
    
    def merge_lora_adapters(
        self,
        adapters: List[LoRAAdapter],
        weights: Optional[List[float]] = None,
        strategy: MergeStrategy = MergeStrategy.LINEAR
    ) -> Dict[str, torch.Tensor]:
        """
        合并多个LoRA适配器
        
        Args:
            adapters: LoRA适配器列表
            weights: 权重列表
            strategy: 合并策略
            
        Returns:
            合并后的参数字典
        """
        if not adapters:
            return {}
        
        # 使用 AdapterMerger
        if strategy == MergeStrategy.LINEAR:
            weights = weights or [1.0 / len(adapters)] * len(adapters)
            return AdapterMerger.merge_linear(adapters, weights)
        elif strategy == MergeStrategy.TIES:
            return AdapterMerger.merge_ties(adapters)
        elif strategy == MergeStrategy.TASK_ARITHMETIC and len(adapters) > 1:
            task_weights = weights[1:] if weights else None
            return AdapterMerger.merge_task_arithmetic(adapters[0], adapters[1:], task_weights)
        else:
            weights = weights or [1.0 / len(adapters)] * len(adapters)
            return AdapterMerger.merge_linear(adapters, weights)


# ==================== 全局函数 ====================

def get_adapter_manager(config: Optional[AdapterManagerConfig] = None) -> AdapterManager:
    """获取全局适配器管理器实例"""
    global _adapter_manager
    
    with _manager_lock:
        if _adapter_manager is None:
            _adapter_manager = AdapterManager(config)
        return _adapter_manager


def reset_adapter_manager() -> None:
    """重置全局适配器管理器"""
    global _adapter_manager
    
    with _manager_lock:
        if _adapter_manager is not None:
            _adapter_manager.shutdown()
            _adapter_manager = None


# ==================== 便捷函数 ====================

def get_managed_encoder(
    modality: Union[ModalityType, str],
    hidden_size: int = 768,
    **kwargs
) -> ModalityEncoder:
    """
    便捷函数：通过管理器获取模态编码器（带缓存和指标）
    
    Args:
        modality: 模态类型
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        模态编码器实例
    """
    manager = get_adapter_manager()
    return manager.get_encoder(modality, hidden_size, **kwargs)


def get_managed_fusion(
    method: Union[FusionMethod, str] = "cross_attention",
    hidden_size: int = 768,
    **kwargs
) -> FusionModule:
    """
    便捷函数：通过管理器获取融合模块（带缓存和指标）
    
    Args:
        method: 融合方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        融合模块实例
    """
    manager = get_adapter_manager()
    return manager.get_fusion(method, hidden_size, **kwargs)


def get_managed_alignment(
    method: Union[AlignmentMethod, str] = "contrastive",
    hidden_size: int = 768,
    **kwargs
) -> AlignmentModule:
    """
    便捷函数：通过管理器获取对齐模块（带缓存和指标）
    
    Args:
        method: 对齐方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        对齐模块实例
    """
    manager = get_adapter_manager()
    return manager.get_alignment(method, hidden_size, **kwargs)


def get_managed_adapter(
    adapter_type: Union[AdapterType, str] = "lora",
    hidden_size: int = 768,
    **kwargs
) -> ModelAdapter:
    """
    便捷函数：通过管理器获取模型适配器（带缓存和指标）
    
    Args:
        adapter_type: 适配器类型
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        模型适配器实例
    """
    manager = get_adapter_manager()
    return manager.get_adapter(adapter_type, hidden_size, **kwargs)


def create_multimodal_pipeline(
    modalities: List[str],
    fusion_method: str = None,
    alignment_method: str = None,
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：创建多模态管道
    
    Args:
        modalities: 模态列表
        fusion_method: 融合方法
        alignment_method: 对齐方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        管道字典
    """
    manager = get_adapter_manager()
    return manager.create_multimodal_pipeline(
        modalities, fusion_method, alignment_method, hidden_size, **kwargs
    )


def run_multimodal_pipeline(
    pipeline: Dict[str, Any],
    inputs: Dict[str, torch.Tensor],
    return_intermediate: bool = False
) -> Union[torch.Tensor, Dict[str, Any]]:
    """
    便捷函数：运行多模态管道
    
    Args:
        pipeline: 管道字典
        inputs: 各模态输入
        return_intermediate: 是否返回中间结果
        
    Returns:
        融合结果
    """
    manager = get_adapter_manager()
    return manager.run_pipeline(pipeline, inputs, return_intermediate)


def adapter_manager_health_check() -> Dict[str, Any]:
    """
    便捷函数：执行健康检查
    
    Returns:
        健康检查报告
    """
    manager = get_adapter_manager()
    return manager.health_check()


def adapter_manager_metrics() -> Dict[str, Any]:
    """
    便捷函数：获取指标
    
    Returns:
        指标字典
    """
    manager = get_adapter_manager()
    return manager.get_metrics()


# ==================== 配置构建器 ====================

class AdapterManagerConfigBuilder:
    """适配器管理器配置构建器"""
    
    def __init__(self):
        self._config = {}
    
    def with_hidden_size(self, size: int) -> 'AdapterManagerConfigBuilder':
        """设置默认隐藏层大小"""
        self._config['default_hidden_size'] = size
        return self
    
    def with_modalities(self, modalities: List[str]) -> 'AdapterManagerConfigBuilder':
        """设置默认模态"""
        self._config['default_modalities'] = modalities
        return self
    
    def with_fusion_method(self, method: str) -> 'AdapterManagerConfigBuilder':
        """设置默认融合方法"""
        self._config['default_fusion_method'] = method
        return self
    
    def with_alignment_method(self, method: str) -> 'AdapterManagerConfigBuilder':
        """设置默认对齐方法"""
        self._config['default_alignment_method'] = method
        return self
    
    def with_adapter_type(self, adapter_type: str) -> 'AdapterManagerConfigBuilder':
        """设置默认适配器类型"""
        self._config['default_adapter_type'] = adapter_type
        return self
    
    def with_device(self, device: str) -> 'AdapterManagerConfigBuilder':
        """设置设备"""
        self._config['device'] = device
        return self
    
    def with_caching(self, encoders: bool = True, fusions: bool = True,
                    alignments: bool = True, adapters: bool = True,
                    max_size: int = 100) -> 'AdapterManagerConfigBuilder':
        """设置缓存配置"""
        self._config['cache_encoders'] = encoders
        self._config['cache_fusions'] = fusions
        self._config['cache_alignments'] = alignments
        self._config['cache_adapters'] = adapters
        self._config['max_cache_size'] = max_size
        return self
    
    def with_metrics(self, enabled: bool = True, 
                    retention_hours: int = 24) -> 'AdapterManagerConfigBuilder':
        """设置指标配置"""
        self._config['enable_metrics'] = enabled
        self._config['metrics_retention_hours'] = retention_hours
        return self
    
    def with_warmup(self, enabled: bool = True,
                   modalities: List[str] = None) -> 'AdapterManagerConfigBuilder':
        """设置预热配置"""
        self._config['enable_warmup'] = enabled
        if modalities:
            self._config['warmup_modalities'] = modalities
        return self
    
    def with_memory_limit(self, max_gb: float = 8.0,
                         cleanup_threshold: float = 0.9) -> 'AdapterManagerConfigBuilder':
        """设置内存限制"""
        self._config['max_memory_gb'] = max_gb
        self._config['memory_cleanup_threshold'] = cleanup_threshold
        return self
    
    def with_checkpoint(self, enabled: bool = False, 
                       directory: str = "./adapter_checkpoints",
                       interval_minutes: int = 30) -> 'AdapterManagerConfigBuilder':
        """设置检查点配置"""
        self._config['auto_checkpoint'] = enabled
        self._config['checkpoint_dir'] = directory
        self._config['checkpoint_interval_minutes'] = interval_minutes
        return self
    
    def build(self) -> AdapterManagerConfig:
        """构建配置"""
        return AdapterManagerConfig(**self._config)


def build_adapter_manager_config() -> AdapterManagerConfigBuilder:
    """
    便捷函数：获取配置构建器
    
    Returns:
        配置构建器实例
        
    使用示例:
        config = (build_adapter_manager_config()
            .with_hidden_size(1024)
            .with_modalities(["text", "image"])
            .with_fusion_method("cross_attention")
            .with_caching(encoders=True, fusions=True)
            .with_metrics(enabled=True)
            .build())
    """
    return AdapterManagerConfigBuilder()


# ==================== 高级便捷函数 ====================

def analyze_encoder(
    modality: Union[ModalityType, str],
    inputs: torch.Tensor,
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：分析编码器质量
    
    Args:
        modality: 模态类型
        inputs: 输入张量
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        质量分析报告
    """
    manager = get_adapter_manager()
    return manager.analyze_encoder_quality(modality, inputs, hidden_size, **kwargs)


def analyze_fusion(
    features: List[torch.Tensor],
    method: Union[FusionMethod, str] = "cross_attention",
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：分析融合质量
    
    Args:
        features: 特征列表
        method: 融合方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        质量分析报告
    """
    manager = get_adapter_manager()
    return manager.analyze_fusion_quality(features, method, hidden_size, **kwargs)


def analyze_alignment(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    method: Union[AlignmentMethod, str] = "contrastive",
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：分析对齐质量
    
    Args:
        features_a: 模态A特征
        features_b: 模态B特征
        method: 对齐方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        质量分析报告
    """
    manager = get_adapter_manager()
    return manager.analyze_alignment_quality(features_a, features_b, method, hidden_size, **kwargs)


def create_managed_adapter_fusion(
    adapter_types: List[Union[AdapterType, str]],
    fusion_method: str = "attention",
    hidden_size: int = 768,
    **kwargs
) -> AdapterFusion:
    """
    便捷函数：通过管理器创建适配器融合
    
    Args:
        adapter_types: 适配器类型列表
        fusion_method: 融合方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        适配器融合模块
    """
    manager = get_adapter_manager()
    return manager.create_adapter_fusion(adapter_types, fusion_method, hidden_size, **kwargs)


def create_managed_multimodal_encoder(
    modalities: List[Union[ModalityType, str]],
    hidden_size: int = 768,
    fusion_method: str = "attention",
    **kwargs
) -> MultiModalEncoder:
    """
    便捷函数：通过管理器创建多模态编码器
    
    Args:
        modalities: 模态列表
        hidden_size: 隐藏层大小
        fusion_method: 融合方法
        **kwargs: 其他配置
        
    Returns:
        多模态编码器
    """
    manager = get_adapter_manager()
    return manager.create_multimodal_encoder(modalities, hidden_size, fusion_method, **kwargs)


def create_hybrid_pipeline(
    modalities: List[str],
    fusion_methods: List[str],
    fusion_weights: Optional[List[float]] = None,
    alignment_method: str = "contrastive",
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：创建混合融合管道
    
    Args:
        modalities: 模态列表
        fusion_methods: 融合方法列表
        fusion_weights: 融合权重
        alignment_method: 对齐方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        管道字典
    """
    manager = get_adapter_manager()
    return manager.create_hybrid_fusion_pipeline(
        modalities, fusion_methods, fusion_weights, alignment_method, hidden_size, **kwargs
    )


def create_advanced_pipeline(
    modalities: List[str],
    adapter_types: List[Union[AdapterType, str]],
    fusion_method: str = "cross_attention",
    adapter_fusion_method: str = "attention",
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：创建高级适配器管道
    
    Args:
        modalities: 模态列表
        adapter_types: 适配器类型列表
        fusion_method: 特征融合方法
        adapter_fusion_method: 适配器融合方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        管道字典
    """
    manager = get_adapter_manager()
    return manager.create_advanced_adapter_pipeline(
        modalities, adapter_types, fusion_method, adapter_fusion_method, hidden_size, **kwargs
    )


def run_diagnostics() -> Dict[str, Any]:
    """
    便捷函数：运行综合诊断
    
    Returns:
        诊断报告
    """
    manager = get_adapter_manager()
    return manager.run_comprehensive_diagnostics()


def get_component_health_status(
    component_type: str,
    component_key: str
) -> Dict[str, Any]:
    """
    便捷函数：获取组件健康状态
    
    Args:
        component_type: 组件类型
        component_key: 组件键
        
    Returns:
        组件健康状态
    """
    manager = get_adapter_manager()
    return manager.get_component_health(component_type, component_key)


def optimize_manager_cache(target_utilization: float = 0.8) -> Dict[str, Any]:
    """
    便捷函数：优化管理器缓存
    
    Args:
        target_utilization: 目标缓存利用率
        
    Returns:
        优化结果
    """
    manager = get_adapter_manager()
    return manager.optimize_cache(target_utilization)


def merge_managed_adapters(
    adapter_types: List[Union[AdapterType, str]],
    weights: Optional[List[float]] = None,
    strategy: MergeStrategy = MergeStrategy.LINEAR,
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：通过管理器合并适配器
    
    Args:
        adapter_types: 适配器类型列表
        weights: 权重列表
        strategy: 合并策略
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        合并后的参数字典
    """
    manager = get_adapter_manager()
    return manager.merge_adapters(adapter_types, weights, strategy, hidden_size, **kwargs)


def compare_managed_adapters(
    adapter_types: List[Union[AdapterType, str]],
    hidden_size: int = 768,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：通过管理器比较适配器
    
    Args:
        adapter_types: 适配器类型列表
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        比较结果
    """
    manager = get_adapter_manager()
    return manager.compare_adapters(adapter_types, hidden_size, **kwargs)

