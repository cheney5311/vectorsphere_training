"""
算子融合模块
提供生产级的模型算子融合优化功能
"""

from .fusion_optimizer import (
    FusionOptimizer,
    FusionAnalyzer,
    FusionPatterns,
    FusionPattern,
    ConvBatchNormFusion,
    ConvReluFusion,
    LinearReluFusion,
    MatMulAddFusion,
    FusionType,
    Operator,
    FusionCandidate,
    FusionResult,
)

__all__ = [
    'FusionOptimizer',
    'FusionAnalyzer',
    'FusionPatterns',
    'FusionPattern',
    'ConvBatchNormFusion',
    'ConvReluFusion',
    'LinearReluFusion',
    'MatMulAddFusion',
    'FusionType',
    'Operator',
    'FusionCandidate',
    'FusionResult',
]
