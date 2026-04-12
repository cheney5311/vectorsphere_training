"""统一性能分析器

提供系统性能分析、瓶颈识别、优化建议等功能。
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np

from .models import SystemMetrics, GPUMetrics
from .exceptions import MonitoringError

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """分析类型枚举"""
    FULL = "full"  # 全面分析
    CPU = "cpu"  # CPU分析
    MEMORY = "memory"  # 内存分析
    GPU = "gpu"  # GPU分析
    IO = "io"  # I/O分析
    NETWORK = "network"  # 网络分析


class BottleneckType(Enum):
    """瓶颈类型枚举"""
    CPU_BOUND = "cpu_bound"  # CPU瓶颈
    MEMORY_BOUND = "memory_bound"  # 内存瓶颈
    IO_BOUND = "io_bound"  # I/O瓶颈
    GPU_BOUND = "gpu_bound"  # GPU瓶颈
    NETWORK_BOUND = "network_bound"  # 网络瓶颈
    DISK_BOUND = "disk_bound"  # 磁盘瓶颈


class SeverityLevel(Enum):
    """严重程度枚举"""
    LOW = "low"  # 低
    MEDIUM = "medium"  # 中
    HIGH = "high"  # 高
    CRITICAL = "critical"  # 严重


@dataclass
class PerformanceBottleneck:
    """性能瓶颈信息"""
    bottleneck_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # 瓶颈基本信息
    type: BottleneckType = BottleneckType.CPU_BOUND
    severity: SeverityLevel = SeverityLevel.LOW
    description: str = ""

    # 相关指标
    metrics: Dict[str, float] = field(default_factory=dict)

    # 影响评估
    impact_score: float = 0.0  # 影响分数 0-100
    affected_components: List[str] = field(default_factory=list)

    # 建议解决方案
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class PerformanceRecommendation:
    """性能优化建议"""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # 建议基本信息
    title: str = ""
    description: str = ""
    category: str = ""  # 类别：cpu, memory, gpu, io, network

    # 优先级和影响
    priority: int = 1  # 优先级 1-5，5最高
    estimated_impact: float = 0.0  # 预期影响 0-100
    confidence: float = 0.0  # 置信度 0-1

    # 实施信息
    implementation_effort: str = "low"  # low, medium, high
    estimated_time_hours: float = 0.0

    # 相关资源
    related_metrics: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """性能分析报告"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # 分析基本信息
    analysis_type: AnalysisType = AnalysisType.FULL
    target_id: Optional[str] = None
    duration_seconds: float = 0.0

    # 分析结果
    summary: str = ""
    overall_score: float = 0.0  # 整体性能分数 0-100

    # 瓶颈和建议
    bottlenecks: List[PerformanceBottleneck] = field(default_factory=list)
    recommendations: List[PerformanceRecommendation] = field(default_factory=list)

    # 详细指标
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化性能分析器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 分析阈值配置
        self.thresholds = {
            'cpu_high': self.config.get('cpu_high_threshold', 80.0),
            'memory_high': self.config.get('memory_high_threshold', 85.0),
            'gpu_high': self.config.get('gpu_high_threshold', 90.0),
            'io_high': self.config.get('io_high_threshold', 80.0),
            'disk_high': self.config.get('disk_high_threshold', 90.0)
        }

        logger.info("性能分析器初始化完成")

    def analyze(self, metrics: SystemMetrics, 
                gpu_metrics: Optional[List[GPUMetrics]] = None,
                analysis_type: AnalysisType = AnalysisType.FULL) -> PerformanceReport:
        """执行性能分析

        Args:
            metrics: 系统指标
            gpu_metrics: GPU指标列表
            analysis_type: 分析类型

        Returns:
            PerformanceReport: 性能分析报告
        """
        start_time = time.time()

        try:
            # 识别性能瓶颈
            bottlenecks = self._identify_bottlenecks(metrics, gpu_metrics, analysis_type)
            recommendations = self._generate_recommendations(bottlenecks, metrics)

            # 计算整体性能分数
            overall_score = self._calculate_overall_score(metrics, gpu_metrics, bottlenecks)

            # 生成报告
            report = PerformanceReport(
                analysis_type=analysis_type,
                duration_seconds=time.time() - start_time,
                overall_score=overall_score,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                metrics_snapshot=self._metrics_to_dict(metrics, gpu_metrics),
                summary=self._generate_summary(bottlenecks, overall_score)
            )

            logger.info(f"性能分析完成，整体分数: {overall_score:.1f}")
            return report

        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            raise MonitoringError(f"性能分析失败: {e}")

    def _identify_bottlenecks(self, metrics: SystemMetrics,
                              gpu_metrics: Optional[List[GPUMetrics]],
                              analysis_type: AnalysisType) -> List[PerformanceBottleneck]:
        """识别性能瓶颈

        Args:
            metrics: 系统指标
            gpu_metrics: GPU指标列表
            analysis_type: 分析类型

        Returns:
            List[PerformanceBottleneck]: 性能瓶颈列表
        """
        bottlenecks = []

        # CPU瓶颈检查
        if metrics.cpu_percent > self.thresholds['cpu_high']:
            bottlenecks.append(PerformanceBottleneck(
                type=BottleneckType.CPU_BOUND,
                severity=self._determine_severity(metrics.cpu_percent, self.thresholds['cpu_high']),
                description=f"CPU利用率过高: {metrics.cpu_percent:.1f}%",
                metrics={'cpu_utilization': metrics.cpu_percent},
                impact_score=metrics.cpu_percent,
                affected_components=['CPU'],
                suggested_actions=[
                    "优化算法复杂度",
                    "增加CPU核心数",
                    "检查是否有死循环或无限等待"
                ]
            ))

        # 内存瓶颈检查
        if metrics.memory_percent > self.thresholds['memory_high']:
            bottlenecks.append(PerformanceBottleneck(
                type=BottleneckType.MEMORY_BOUND,
                severity=self._determine_severity(metrics.memory_percent, self.thresholds['memory_high']),
                description=f"内存利用率过高: {metrics.memory_percent:.1f}%",
                metrics={'memory_utilization': metrics.memory_percent},
                impact_score=metrics.memory_percent,
                affected_components=['Memory'],
                suggested_actions=[
                    "优化内存使用",
                    "增加内存容量",
                    "检查内存泄漏"
                ]
            ))

        # 磁盘瓶颈检查
        if metrics.disk_percent > self.thresholds['disk_high']:
            bottlenecks.append(PerformanceBottleneck(
                type=BottleneckType.DISK_BOUND,
                severity=self._determine_severity(metrics.disk_percent, self.thresholds['disk_high']),
                description=f"磁盘使用率过高: {metrics.disk_percent:.1f}%",
                metrics={'disk_utilization': metrics.disk_percent},
                impact_score=metrics.disk_percent,
                affected_components=['Disk'],
                suggested_actions=[
                    "清理磁盘空间",
                    "增加磁盘容量",
                    "优化磁盘使用"
                ]
            ))

        # GPU瓶颈检查
        if gpu_metrics:
            for gpu_metric in gpu_metrics:
                if gpu_metric.gpu_utilization > self.thresholds['gpu_high']:
                    bottlenecks.append(PerformanceBottleneck(
                        type=BottleneckType.GPU_BOUND,
                        severity=self._determine_severity(gpu_metric.gpu_utilization, self.thresholds['gpu_high']),
                        description=f"GPU {gpu_metric.gpu_id} 利用率过高: {gpu_metric.gpu_utilization:.1f}%",
                        metrics={f'gpu_{gpu_metric.gpu_id}_utilization': gpu_metric.gpu_utilization},
                        impact_score=gpu_metric.gpu_utilization,
                        affected_components=[f'GPU_{gpu_metric.gpu_id}'],
                        suggested_actions=[
                            "优化GPU计算",
                            "增加GPU数量",
                            "检查是否有不必要的GPU计算"
                        ]
                    ))

        return bottlenecks

    def _generate_recommendations(self, bottlenecks: List[PerformanceBottleneck],
                                  metrics: SystemMetrics) -> List[PerformanceRecommendation]:
        """生成性能优化建议

        Args:
            bottlenecks: 性能瓶颈列表
            metrics: 系统指标

        Returns:
            List[PerformanceRecommendation]: 性能优化建议列表
        """
        recommendations = []

        for bottleneck in bottlenecks:
            # 根据瓶颈类型生成建议
            if bottleneck.type == BottleneckType.CPU_BOUND:
                recommendations.append(PerformanceRecommendation(
                    title="CPU性能优化建议",
                    description="系统存在CPU瓶颈，建议采取以下措施优化CPU性能",
                    category="cpu",
                    priority=3 if bottleneck.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 2,
                    estimated_impact=bottleneck.impact_score * 0.8,
                    confidence=0.8,
                    implementation_effort="medium",
                    estimated_time_hours=4.0,
                    related_metrics=list(bottleneck.metrics.keys()),
                    suggested_actions=bottleneck.suggested_actions
                ))
            elif bottleneck.type == BottleneckType.MEMORY_BOUND:
                recommendations.append(PerformanceRecommendation(
                    title="内存性能优化建议",
                    description="系统存在内存瓶颈，建议采取以下措施优化内存使用",
                    category="memory",
                    priority=3 if bottleneck.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 2,
                    estimated_impact=bottleneck.impact_score * 0.7,
                    confidence=0.75,
                    implementation_effort="medium",
                    estimated_time_hours=3.0,
                    related_metrics=list(bottleneck.metrics.keys()),
                    suggested_actions=bottleneck.suggested_actions
                ))
            elif bottleneck.type == BottleneckType.GPU_BOUND:
                recommendations.append(PerformanceRecommendation(
                    title="GPU性能优化建议",
                    description="系统存在GPU瓶颈，建议采取以下措施优化GPU使用",
                    category="gpu",
                    priority=3 if bottleneck.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 2,
                    estimated_impact=bottleneck.impact_score * 0.85,
                    confidence=0.85,
                    implementation_effort="high",
                    estimated_time_hours=6.0,
                    related_metrics=list(bottleneck.metrics.keys()),
                    suggested_actions=bottleneck.suggested_actions
                ))
            elif bottleneck.type == BottleneckType.DISK_BOUND:
                recommendations.append(PerformanceRecommendation(
                    title="磁盘性能优化建议",
                    description="系统存在磁盘瓶颈，建议采取以下措施优化磁盘使用",
                    category="disk",
                    priority=2,
                    estimated_impact=bottleneck.impact_score * 0.6,
                    confidence=0.7,
                    implementation_effort="low",
                    estimated_time_hours=2.0,
                    related_metrics=list(bottleneck.metrics.keys()),
                    suggested_actions=bottleneck.suggested_actions
                ))

        return recommendations

    def _calculate_overall_score(self, metrics: SystemMetrics,
                                 gpu_metrics: Optional[List[GPUMetrics]],
                                 bottlenecks: List[PerformanceBottleneck]) -> float:
        """计算整体性能分数

        Args:
            metrics: 系统指标
            gpu_metrics: GPU指标列表
            bottlenecks: 性能瓶颈列表

        Returns:
            float: 整体性能分数 (0-100)
        """
        # 基础分数基于资源利用率
        cpu_score = max(0, 100 - metrics.cpu_percent)
        memory_score = max(0, 100 - metrics.memory_percent)
        disk_score = max(0, 100 - metrics.disk_percent)

        # GPU分数计算
        gpu_scores = []
        if gpu_metrics:
            for gpu_metric in gpu_metrics:
                gpu_scores.append(max(0, 100 - gpu_metric.gpu_utilization))
            gpu_score = sum(gpu_scores) / len(gpu_scores) if gpu_scores else 100
        else:
            gpu_score = 100

        # 基础分数是各项资源分数的加权平均
        base_score = (cpu_score * 0.3 + memory_score * 0.2 + disk_score * 0.2 + gpu_score * 0.3)

        # 根据瓶颈严重程度调整分数
        penalty = 0
        for bottleneck in bottlenecks:
            if bottleneck.severity == SeverityLevel.CRITICAL:
                penalty += 20
            elif bottleneck.severity == SeverityLevel.HIGH:
                penalty += 10
            elif bottleneck.severity == SeverityLevel.MEDIUM:
                penalty += 5

        final_score = max(0, base_score - penalty)
        return final_score

    def _generate_summary(self, bottlenecks: List[PerformanceBottleneck], overall_score: float) -> str:
        """生成分析摘要

        Args:
            bottlenecks: 性能瓶颈列表
            overall_score: 整体性能分数

        Returns:
            str: 分析摘要
        """
        if not bottlenecks:
            return f"系统性能良好，整体分数: {overall_score:.1f}"

        critical_bottlenecks = [b for b in bottlenecks if b.severity == SeverityLevel.CRITICAL]
        high_bottlenecks = [b for b in bottlenecks if b.severity == SeverityLevel.HIGH]

        if critical_bottlenecks:
            return f"系统存在{len(critical_bottlenecks)}个严重瓶颈，整体分数: {overall_score:.1f}"
        elif high_bottlenecks:
            return f"系统存在{len(high_bottlenecks)}个高优先级瓶颈，整体分数: {overall_score:.1f}"
        else:
            return f"系统存在{len(bottlenecks)}个性能瓶颈，整体分数: {overall_score:.1f}"

    def _determine_severity(self, utilization: float, threshold: float) -> SeverityLevel:
        """确定严重程度

        Args:
            utilization: 资源利用率
            threshold: 阈值

        Returns:
            SeverityLevel: 严重程度
        """
        if utilization > threshold * 1.2:  # 超过阈值20%
            return SeverityLevel.CRITICAL
        elif utilization > threshold * 1.1:  # 超过阈值10%
            return SeverityLevel.HIGH
        elif utilization > threshold:  # 超过阈值
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _metrics_to_dict(self, metrics: SystemMetrics, 
                         gpu_metrics: Optional[List[GPUMetrics]]) -> Dict[str, Any]:
        """将指标转换为字典

        Args:
            metrics: 系统指标
            gpu_metrics: GPU指标列表

        Returns:
            Dict[str, Any]: 指标字典
        """
        result = {
            'cpu_utilization': metrics.cpu_percent,
            'memory_utilization': metrics.memory_percent,
            'disk_utilization': metrics.disk_percent,
            'network_rx_mbps': metrics.network_recv_mb,
            'network_tx_mbps': metrics.network_sent_mb
        }
        
        if gpu_metrics:
            result['gpu_utilization'] = {gpu.gpu_id: gpu.gpu_utilization for gpu in gpu_metrics}
            
        return result


# 全局性能分析器实例
_global_analyzer: Optional[PerformanceAnalyzer] = None


def get_performance_analyzer(config: Optional[Dict[str, Any]] = None) -> PerformanceAnalyzer:
    """获取全局性能分析器实例

    Args:
        config: 配置参数

    Returns:
        PerformanceAnalyzer: 性能分析器实例
    """
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = PerformanceAnalyzer(config)
    return _global_analyzer