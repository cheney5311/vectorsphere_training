"""融合模式定义"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FusionPatterns:
    """融合模式集合"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """初始化融合模式"""
        # 基础融合模式
        self.patterns['conv_bn'] = {
            'name': 'ConvBatchNorm',
            'description': '卷积层与批归一化层融合',
            'pattern': ['conv2d', 'batch_norm'],
            'performance_gain': 0.15,
            'memory_reduction': 0.10
        }
        
        self.patterns['conv_relu'] = {
            'name': 'ConvReLU',
            'description': '卷积层与ReLU激活函数融合',
            'pattern': ['conv2d', 'relu'],
            'performance_gain': 0.12,
            'memory_reduction': 0.08
        }
        
        self.patterns['linear_relu'] = {
            'name': 'LinearReLU',
            'description': '线性层与ReLU激活函数融合',
            'pattern': ['linear', 'relu'],
            'performance_gain': 0.10,
            'memory_reduction': 0.06
        }
        
        self.patterns['matmul_add'] = {
            'name': 'MatMulAdd',
            'description': '矩阵乘法与加法融合',
            'pattern': ['matmul', 'add'],
            'performance_gain': 0.18,
            'memory_reduction': 0.12
        }
        
        # 复合融合模式
        self.patterns['conv_bn_relu'] = {
            'name': 'ConvBatchNormReLU',
            'description': '卷积、批归一化、ReLU三层融合',
            'pattern': ['conv2d', 'batch_norm', 'relu'],
            'performance_gain': 0.25,
            'memory_reduction': 0.18
        }
        
        self.patterns['attention_qkv'] = {
            'name': 'AttentionQKV',
            'description': '注意力机制中Q、K、V计算融合',
            'pattern': ['linear', 'linear', 'linear'],
            'performance_gain': 0.20,
            'memory_reduction': 0.15
        }
        
        self.logger.info(f"已初始化 {len(self.patterns)} 个融合模式")
    
    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """获取指定的融合模式"""
        return self.patterns.get(pattern_name)
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有融合模式"""
        return self.patterns.copy()
    
    def get_patterns_by_type(self, operator_type: str) -> List[Dict[str, Any]]:
        """根据算子类型获取相关的融合模式"""
        matching_patterns = []
        
        for pattern_name, pattern_info in self.patterns.items():
            if operator_type in pattern_info['pattern']:
                matching_patterns.append({
                    'name': pattern_name,
                    **pattern_info
                })
        
        return matching_patterns
    
    def match_pattern(self, operators: List[str]) -> List[Dict[str, Any]]:
        """匹配操作序列与融合模式"""
        matched_patterns = []
        
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            
            # 检查是否匹配
            if self._is_pattern_match(operators, pattern):
                matched_patterns.append({
                    'name': pattern_name,
                    'match_length': len(pattern),
                    **pattern_info
                })
        
        # 按匹配长度排序，优先选择更长的模式
        matched_patterns.sort(key=lambda x: x['match_length'], reverse=True)
        
        return matched_patterns
    
    def _is_pattern_match(self, operators: List[str], pattern: List[str]) -> bool:
        """检查操作序列是否匹配模式"""
        if len(operators) < len(pattern):
            return False
        
        # 检查是否有连续的子序列匹配模式
        for i in range(len(operators) - len(pattern) + 1):
            if operators[i:i+len(pattern)] == pattern:
                return True
        
        return False
    
    def add_pattern(self, name: str, pattern_info: Dict[str, Any]):
        """添加新的融合模式"""
        required_fields = ['name', 'description', 'pattern', 'performance_gain', 'memory_reduction']
        
        for field in required_fields:
            if field not in pattern_info:
                raise ValueError(f"融合模式缺少必需字段: {field}")
        
        self.patterns[name] = pattern_info
        self.logger.info(f"已添加融合模式: {name}")
    
    def remove_pattern(self, name: str) -> bool:
        """移除融合模式"""
        if name in self.patterns:
            del self.patterns[name]
            self.logger.info(f"已移除融合模式: {name}")
            return True
        return False
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """获取融合模式统计信息"""
        total_patterns = len(self.patterns)
        avg_performance_gain = sum(p['performance_gain'] for p in self.patterns.values()) / total_patterns
        avg_memory_reduction = sum(p['memory_reduction'] for p in self.patterns.values()) / total_patterns
        
        # 按算子类型分组统计
        operator_counts = {}
        for pattern_info in self.patterns.values():
            for op in pattern_info['pattern']:
                operator_counts[op] = operator_counts.get(op, 0) + 1
        
        return {
            'total_patterns': total_patterns,
            'average_performance_gain': avg_performance_gain,
            'average_memory_reduction': avg_memory_reduction,
            'operator_distribution': operator_counts,
            'most_common_operators': sorted(operator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }