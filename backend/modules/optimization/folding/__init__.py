"""
常量折叠模块
提供生产级的模型常量折叠优化功能
"""

from .folding_optimizer import (
    FoldingOptimizer,
    ConstantAnalyzer,
    FoldingTransformer,
    FoldingStrategy,
    ArithmeticFolding,
    ConstantPropagation,
    DeadCodeElimination,
    ShapeFolding,
    ConstantType,
    ConstantInfo,
    FoldingResult,
)

__all__ = [
    'FoldingOptimizer',
    'ConstantAnalyzer',
    'FoldingTransformer',
    'FoldingStrategy',
    'ArithmeticFolding',
    'ConstantPropagation',
    'DeadCodeElimination',
    'ShapeFolding',
    'ConstantType',
    'ConstantInfo',
    'FoldingResult',
]
