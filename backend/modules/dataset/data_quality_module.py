"""数据质量模块

实现数据质量评估、问题检测、数据清理等核心功能。
"""

import logging
import os
import random
import re
import statistics
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    QualityAssessmentFailedError,
    IssueDetectionError,
    DataCleaningError,
    DataProfilingError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类型定义
# ============================================================================

@dataclass
class ColumnProfile:
    """列剖析结果"""
    column_name: str
    data_type: str  # numeric/categorical/text/datetime/boolean/unknown
    
    # 基本统计
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    
    # 数值统计
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    
    # 文本统计
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # 分布信息
    value_distribution: Dict[str, int] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    # 异常检测
    outliers_count: int = 0
    outliers_indices: List[int] = field(default_factory=list)
    
    # 模式识别
    detected_pattern: Optional[str] = None
    pattern_match_rate: float = 0.0


@dataclass
class DatasetProfile:
    """数据集剖析结果"""
    dataset_id: str
    total_rows: int = 0
    total_columns: int = 0
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    duplicate_rows_count: int = 0
    duplicate_row_indices: List[int] = field(default_factory=list)
    memory_usage_bytes: int = 0
    profiled_at: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# 数据剖析器
# ============================================================================

class DataProfiler:
    """数据剖析器
    
    对数据集进行全面的统计分析和剖析。
    """
    
    def __init__(self):
        """初始化数据剖析器"""
        self.type_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s]+$',
            'phone': r'^[\d\-\+\(\)\s]{7,}$',
            'date': r'^\d{4}[-/]\d{2}[-/]\d{2}$',
            'datetime': r'^\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            'credit_card': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
        }
    
    def profile_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        sample_size: Optional[int] = None
    ) -> DatasetProfile:
        """剖析数据集
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            sample_size: 采样大小（None表示全量分析）
            
        Returns:
            DatasetProfile: 数据集剖析结果
        """
        try:
            if not data:
                return DatasetProfile(dataset_id=dataset_id)
            
            # 采样
            if sample_size and len(data) > sample_size:
                sample_data = random.sample(data, sample_size)
            else:
                sample_data = data
            
            total_rows = len(data)
            columns = list(data[0].keys()) if data else []
            
            # 剖析每一列
            column_profiles = []
            for col in columns:
                profile = self._profile_column(sample_data, col)
                column_profiles.append(profile)
            
            # 检测重复行
            duplicate_count, duplicate_indices = self._detect_duplicate_rows(data)
            
            # 估算内存使用
            memory_usage = self._estimate_memory_usage(data)
            
            return DatasetProfile(
                dataset_id=dataset_id,
                total_rows=total_rows,
                total_columns=len(columns),
                column_profiles=column_profiles,
                duplicate_rows_count=duplicate_count,
                duplicate_row_indices=duplicate_indices[:100],  # 只保留前100个
                memory_usage_bytes=memory_usage,
                profiled_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to profile dataset {dataset_id}: {e}")
            raise DataProfilingError(dataset_id, str(e))
    
    def _profile_column(self, data: List[Dict], column_name: str) -> ColumnProfile:
        """剖析单列
        
        Args:
            data: 数据列表
            column_name: 列名
            
        Returns:
            ColumnProfile: 列剖析结果
        """
        values = [row.get(column_name) for row in data]
        non_null_values = [v for v in values if v is not None and v != '']
        
        total_count = len(values)
        null_count = total_count - len(non_null_values)
        
        # 推断数据类型
        data_type = self._infer_column_type(non_null_values)
        
        profile = ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            total_count=total_count,
            null_count=null_count,
            distinct_count=len(set(str(v) for v in non_null_values))
        )
        
        if not non_null_values:
            return profile
        
        # 根据类型计算统计信息
        if data_type == 'numeric':
            numeric_values = [self._to_numeric(v) for v in non_null_values if self._to_numeric(v) is not None]
            if numeric_values:
                profile.min_value = min(numeric_values)
                profile.max_value = max(numeric_values)
                profile.mean = statistics.mean(numeric_values)
                if len(numeric_values) > 1:
                    profile.median = statistics.median(numeric_values)
                    profile.std = statistics.stdev(numeric_values)
                
                # 计算百分位数
                sorted_values = sorted(numeric_values)
                n = len(sorted_values)
                profile.percentiles = {
                    'p25': sorted_values[int(n * 0.25)] if n > 0 else None,
                    'p50': sorted_values[int(n * 0.5)] if n > 0 else None,
                    'p75': sorted_values[int(n * 0.75)] if n > 0 else None,
                    'p95': sorted_values[int(n * 0.95)] if n > 0 else None,
                }
                
                # 检测异常值
                outliers, indices = self._detect_outliers_zscore(numeric_values)
                profile.outliers_count = len(outliers)
                profile.outliers_indices = indices[:100]
                
        elif data_type == 'text':
            str_values = [str(v) for v in non_null_values]
            lengths = [len(s) for s in str_values]
            profile.min_length = min(lengths)
            profile.max_length = max(lengths)
            profile.avg_length = statistics.mean(lengths)
            
            # 检测模式
            pattern, match_rate = self._detect_pattern(str_values)
            profile.detected_pattern = pattern
            profile.pattern_match_rate = match_rate
        
        # 值分布（取Top 10）
        value_counts = Counter(str(v) for v in non_null_values)
        profile.value_distribution = dict(value_counts.most_common(10))
        
        return profile
    
    def _infer_column_type(self, values: List[Any]) -> str:
        """推断列的数据类型
        
        Args:
            values: 非空值列表
            
        Returns:
            str: 数据类型
        """
        if not values:
            return 'unknown'
        
        # 采样检测
        sample = values[:100] if len(values) > 100 else values
        
        # 检测数值类型
        numeric_count = sum(1 for v in sample if self._is_numeric(v))
        if numeric_count / len(sample) > 0.8:
            return 'numeric'
        
        # 检测布尔类型
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f'}
        bool_count = sum(1 for v in sample if str(v).lower() in bool_values)
        if bool_count / len(sample) > 0.9:
            return 'boolean'
        
        # 检测日期时间
        datetime_count = sum(1 for v in sample if self._is_datetime(str(v)))
        if datetime_count / len(sample) > 0.8:
            return 'datetime'
        
        # 检测分类变量（唯一值比例低）
        unique_ratio = len(set(str(v) for v in sample)) / len(sample)
        if unique_ratio < 0.1:
            return 'categorical'
        
        return 'text'
    
    def _is_numeric(self, value: Any) -> bool:
        """检查是否为数值"""
        if isinstance(value, (int, float)):
            return True
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False
    
    def _to_numeric(self, value: Any) -> Optional[float]:
        """转换为数值"""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    def _is_datetime(self, value: str) -> bool:
        """检查是否为日期时间格式"""
        datetime_patterns = [
            r'^\d{4}[-/]\d{2}[-/]\d{2}$',
            r'^\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}',
            r'^\d{2}[-/]\d{2}[-/]\d{4}$',
        ]
        return any(re.match(p, value) for p in datetime_patterns)
    
    def _detect_pattern(self, values: List[str]) -> Tuple[Optional[str], float]:
        """检测值的模式
        
        Args:
            values: 字符串值列表
            
        Returns:
            (模式名称, 匹配率)
        """
        if not values:
            return None, 0.0
        
        sample = values[:100] if len(values) > 100 else values
        
        for pattern_name, pattern in self.type_patterns.items():
            match_count = sum(1 for v in sample if re.match(pattern, v, re.IGNORECASE))
            match_rate = match_count / len(sample)
            if match_rate > 0.8:
                return pattern_name, match_rate
        
        return None, 0.0
    
    def _detect_outliers_zscore(
        self,
        values: List[float],
        threshold: float = 3.0
    ) -> Tuple[List[float], List[int]]:
        """使用Z-score检测异常值
        
        Args:
            values: 数值列表
            threshold: Z-score阈值
            
        Returns:
            (异常值列表, 异常值索引)
        """
        if len(values) < 2:
            return [], []
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return [], []
        
        outliers = []
        indices = []
        for i, v in enumerate(values):
            z_score = abs((v - mean) / std)
            if z_score > threshold:
                outliers.append(v)
                indices.append(i)
        
        return outliers, indices
    
    def _detect_duplicate_rows(
        self,
        data: List[Dict[str, Any]]
    ) -> Tuple[int, List[int]]:
        """检测重复行
        
        Args:
            data: 数据列表
            
        Returns:
            (重复行数量, 重复行索引)
        """
        seen = {}
        duplicates = []
        
        for i, row in enumerate(data):
            # 将行转换为可哈希的元组
            key = tuple(sorted((k, str(v)) for k, v in row.items()))
            if key in seen:
                duplicates.append(i)
            else:
                seen[key] = i
        
        return len(duplicates), duplicates
    
    def _estimate_memory_usage(self, data: List[Dict[str, Any]]) -> int:
        """估算内存使用
        
        Args:
            data: 数据列表
            
        Returns:
            int: 估算的字节数
        """
        if not data:
            return 0
        
        # 简单估算：每行的字符串长度 * 行数
        sample_row = str(data[0])
        avg_row_size = len(sample_row.encode('utf-8'))
        return avg_row_size * len(data)


# ============================================================================
# 质量评估器
# ============================================================================

class QualityAssessor:
    """质量评估器
    
    评估数据质量的各个维度。
    """
    
    def __init__(self, profiler: Optional[DataProfiler] = None):
        """初始化质量评估器
        
        Args:
            profiler: 数据剖析器实例
        """
        self.profiler = profiler or DataProfiler()
    
    def assess(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """评估数据质量
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            dimensions: 要评估的维度列表
            
        Returns:
            Dict[str, Any]: 质量评估结果
        """
        try:
            if not data:
                return self._empty_result(dataset_id)
            
            # 获取数据剖析结果
            profile = self.profiler.profile_dataset(data, dataset_id)
            
            # 默认评估所有维度
            all_dimensions = ['completeness', 'consistency', 'accuracy', 
                             'uniqueness', 'validity', 'timeliness', 'integrity']
            dimensions = dimensions or all_dimensions
            
            # 计算各维度评分
            dimension_scores = {}
            for dim in dimensions:
                score, details = self._assess_dimension(dim, data, profile)
                dimension_scores[dim] = {
                    'score': score,
                    'details': details
                }
            
            # 计算总体评分（加权平均）
            weights = {
                'completeness': 0.2,
                'consistency': 0.15,
                'accuracy': 0.2,
                'uniqueness': 0.15,
                'validity': 0.15,
                'timeliness': 0.1,
                'integrity': 0.05
            }
            
            total_weight = sum(weights.get(d, 0.1) for d in dimension_scores)
            overall_score = sum(
                dimension_scores[d]['score'] * weights.get(d, 0.1)
                for d in dimension_scores
            ) / total_weight if total_weight > 0 else 0.0
            
            # 列级质量指标
            column_metrics = []
            for col_profile in profile.column_profiles:
                col_score = self._calculate_column_quality_score(col_profile)
                column_metrics.append({
                    'column_name': col_profile.column_name,
                    'data_type': col_profile.data_type,
                    'total_count': col_profile.total_count,
                    'null_count': col_profile.null_count,
                    'null_rate': col_profile.null_count / col_profile.total_count if col_profile.total_count > 0 else 0,
                    'distinct_count': col_profile.distinct_count,
                    'distinct_rate': col_profile.distinct_count / col_profile.total_count if col_profile.total_count > 0 else 0,
                    'min_value': col_profile.min_value,
                    'max_value': col_profile.max_value,
                    'mean_value': col_profile.mean,
                    'median_value': col_profile.median,
                    'std_value': col_profile.std,
                    'outlier_count': col_profile.outliers_count,
                    'outlier_rate': col_profile.outliers_count / col_profile.total_count if col_profile.total_count > 0 else 0,
                    'pattern_consistency': col_profile.pattern_match_rate,
                    'quality_score': col_score
                })
            
            return {
                'dataset_id': dataset_id,
                'total_records': profile.total_rows,
                'total_columns': profile.total_columns,
                'missing_values_count': sum(cp.null_count for cp in profile.column_profiles),
                'missing_values_rate': self._calculate_missing_rate(profile),
                'duplicate_records_count': profile.duplicate_rows_count,
                'duplicate_records_rate': profile.duplicate_rows_count / profile.total_rows if profile.total_rows > 0 else 0,
                'outliers_count': sum(cp.outliers_count for cp in profile.column_profiles),
                'outliers_rate': self._calculate_outliers_rate(profile),
                'dimension_scores': dimension_scores,
                'column_metrics': column_metrics,
                'overall_score': round(overall_score, 4),
                'assessed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed for dataset {dataset_id}: {e}")
            raise QualityAssessmentFailedError(dataset_id, str(e))
    
    def _empty_result(self, dataset_id: str) -> Dict[str, Any]:
        """返回空数据集的评估结果"""
        return {
            'dataset_id': dataset_id,
            'total_records': 0,
            'total_columns': 0,
            'missing_values_count': 0,
            'missing_values_rate': 0.0,
            'duplicate_records_count': 0,
            'duplicate_records_rate': 0.0,
            'outliers_count': 0,
            'outliers_rate': 0.0,
            'dimension_scores': {},
            'column_metrics': [],
            'overall_score': 0.0,
            'assessed_at': datetime.utcnow().isoformat()
        }
    
    def _assess_dimension(
        self,
        dimension: str,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估单个质量维度
        
        Args:
            dimension: 维度名称
            data: 数据列表
            profile: 数据集剖析结果
            
        Returns:
            (评分, 详细信息)
        """
        assessors = {
            'completeness': self._assess_completeness,
            'consistency': self._assess_consistency,
            'accuracy': self._assess_accuracy,
            'uniqueness': self._assess_uniqueness,
            'validity': self._assess_validity,
            'timeliness': self._assess_timeliness,
            'integrity': self._assess_integrity,
        }
        
        assessor = assessors.get(dimension)
        if assessor:
            return assessor(data, profile)
        return 0.5, {'error': 'Unknown dimension'}
    
    def _assess_completeness(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估完整性"""
        if profile.total_rows == 0:
            return 0.0, {'total_cells': 0, 'non_null_cells': 0}
        
        total_cells = profile.total_rows * profile.total_columns
        null_cells = sum(cp.null_count for cp in profile.column_profiles)
        non_null_cells = total_cells - null_cells
        
        score = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        # 按列统计
        column_completeness = {
            cp.column_name: 1 - (cp.null_count / cp.total_count) if cp.total_count > 0 else 0
            for cp in profile.column_profiles
        }
        
        return score, {
            'total_cells': total_cells,
            'non_null_cells': non_null_cells,
            'null_cells': null_cells,
            'column_completeness': column_completeness
        }
    
    def _assess_consistency(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估一致性"""
        if not profile.column_profiles:
            return 1.0, {'type_consistency': {}}
        
        type_consistency = {}
        total_score = 0
        
        for cp in profile.column_profiles:
            # 检查类型一致性（通过模式匹配率）
            if cp.detected_pattern:
                consistency = cp.pattern_match_rate
            elif cp.data_type == 'numeric':
                # 数值类型检查格式一致性
                consistency = 0.95  # 假设大部分数值格式一致
            else:
                consistency = 0.9  # 默认一致性
            
            type_consistency[cp.column_name] = consistency
            total_score += consistency
        
        score = total_score / len(profile.column_profiles) if profile.column_profiles else 0.0
        
        return score, {'type_consistency': type_consistency}
    
    def _assess_accuracy(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估准确性"""
        if not profile.column_profiles:
            return 1.0, {'outlier_rate': 0}
        
        # 基于异常值比例评估准确性
        total_values = sum(cp.total_count - cp.null_count for cp in profile.column_profiles)
        total_outliers = sum(cp.outliers_count for cp in profile.column_profiles)
        
        outlier_rate = total_outliers / total_values if total_values > 0 else 0
        score = 1.0 - min(outlier_rate * 5, 1.0)  # 异常值比例越高，准确性越低
        
        return score, {
            'outlier_rate': outlier_rate,
            'total_outliers': total_outliers,
            'column_outliers': {
                cp.column_name: cp.outliers_count for cp in profile.column_profiles
            }
        }
    
    def _assess_uniqueness(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估唯一性"""
        if profile.total_rows == 0:
            return 1.0, {'duplicate_rate': 0}
        
        duplicate_rate = profile.duplicate_rows_count / profile.total_rows
        score = 1.0 - duplicate_rate
        
        return score, {
            'duplicate_rate': duplicate_rate,
            'duplicate_count': profile.duplicate_rows_count,
            'unique_count': profile.total_rows - profile.duplicate_rows_count
        }
    
    def _assess_validity(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估有效性"""
        if not profile.column_profiles:
            return 1.0, {'validity_by_column': {}}
        
        validity_scores = {}
        total_score = 0
        
        for cp in profile.column_profiles:
            valid_count = cp.total_count - cp.null_count - cp.outliers_count
            validity = valid_count / cp.total_count if cp.total_count > 0 else 0
            validity_scores[cp.column_name] = validity
            total_score += validity
        
        score = total_score / len(profile.column_profiles)
        
        return score, {'validity_by_column': validity_scores}
    
    def _assess_timeliness(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估时效性"""
        # 检查是否有日期时间列
        datetime_columns = [
            cp for cp in profile.column_profiles
            if cp.data_type == 'datetime'
        ]
        
        if not datetime_columns:
            return 0.9, {'has_datetime_columns': False}  # 默认较高分数
        
        # 这里可以检查数据的时间戳是否在合理范围内
        # 简化实现：返回默认分数
        return 0.85, {
            'has_datetime_columns': True,
            'datetime_column_count': len(datetime_columns)
        }
    
    def _assess_integrity(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> Tuple[float, Dict[str, Any]]:
        """评估完整性约束"""
        # 检查数据完整性（如非空约束、唯一约束等）
        # 简化实现
        completeness_score = 1.0 - self._calculate_missing_rate(profile)
        uniqueness_score = 1.0 - (profile.duplicate_rows_count / profile.total_rows if profile.total_rows > 0 else 0)
        
        score = (completeness_score + uniqueness_score) / 2
        
        return score, {
            'completeness_contribution': completeness_score,
            'uniqueness_contribution': uniqueness_score
        }
    
    def _calculate_missing_rate(self, profile: DatasetProfile) -> float:
        """计算缺失率"""
        total_cells = profile.total_rows * profile.total_columns
        if total_cells == 0:
            return 0.0
        null_cells = sum(cp.null_count for cp in profile.column_profiles)
        return null_cells / total_cells
    
    def _calculate_outliers_rate(self, profile: DatasetProfile) -> float:
        """计算异常值率"""
        total_values = sum(cp.total_count - cp.null_count for cp in profile.column_profiles)
        if total_values == 0:
            return 0.0
        total_outliers = sum(cp.outliers_count for cp in profile.column_profiles)
        return total_outliers / total_values
    
    def _calculate_column_quality_score(self, profile: ColumnProfile) -> float:
        """计算列的质量评分"""
        scores = []
        
        # 完整性得分
        completeness = 1 - (profile.null_count / profile.total_count) if profile.total_count > 0 else 0
        scores.append(completeness * 0.3)
        
        # 异常值得分
        outlier_rate = profile.outliers_count / profile.total_count if profile.total_count > 0 else 0
        accuracy = 1 - min(outlier_rate * 5, 1.0)
        scores.append(accuracy * 0.3)
        
        # 唯一性得分
        unique_rate = profile.distinct_count / profile.total_count if profile.total_count > 0 else 0
        # 唯一性得分取决于数据类型
        if profile.data_type == 'categorical':
            uniqueness = 0.8  # 分类变量唯一值比例低是正常的
        else:
            uniqueness = min(unique_rate * 2, 1.0)
        scores.append(uniqueness * 0.2)
        
        # 模式一致性得分
        pattern_score = profile.pattern_match_rate if profile.detected_pattern else 0.8
        scores.append(pattern_score * 0.2)
        
        return round(sum(scores), 4)


# ============================================================================
# 问题检测器
# ============================================================================

class IssueDetector:
    """问题检测器
    
    检测数据中的各种质量问题。
    """
    
    def __init__(self, profiler: Optional[DataProfiler] = None):
        """初始化问题检测器"""
        self.profiler = profiler or DataProfiler()
    
    def detect(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        issue_types: Optional[List[str]] = None,
        severity_threshold: str = 'low',
        max_issues: int = 100
    ) -> Dict[str, Any]:
        """检测数据问题
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            issue_types: 要检测的问题类型
            severity_threshold: 严重程度阈值
            max_issues: 最大返回问题数
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        try:
            if not data:
                return self._empty_result(dataset_id)
            
            profile = self.profiler.profile_dataset(data, dataset_id)
            
            # 默认检测所有类型
            all_types = [
                'missing_values', 'duplicate_records', 'outliers',
                'inconsistent_format', 'invalid_values', 'data_type_mismatch'
            ]
            issue_types = issue_types or all_types
            
            issues = []
            
            # 检测各类问题
            if 'missing_values' in issue_types:
                issues.extend(self._detect_missing_values(data, profile))
            
            if 'duplicate_records' in issue_types:
                issues.extend(self._detect_duplicates(data, profile))
            
            if 'outliers' in issue_types:
                issues.extend(self._detect_outliers(data, profile))
            
            if 'inconsistent_format' in issue_types:
                issues.extend(self._detect_format_issues(data, profile))
            
            if 'invalid_values' in issue_types:
                issues.extend(self._detect_invalid_values(data, profile))
            
            if 'data_type_mismatch' in issue_types:
                issues.extend(self._detect_type_mismatch(data, profile))
            
            # 按严重程度过滤
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
            threshold_level = severity_order.get(severity_threshold, 4)
            issues = [
                i for i in issues
                if severity_order.get(i['severity'], 4) <= threshold_level
            ]
            
            # 按严重程度排序
            issues.sort(key=lambda x: severity_order.get(x['severity'], 4))
            
            # 限制返回数量
            issues = issues[:max_issues]
            
            # 统计
            summary_by_type = Counter(i['issue_type'] for i in issues)
            severity_counts = Counter(i['severity'] for i in issues)
            
            return {
                'dataset_id': dataset_id,
                'total_issues': len(issues),
                'critical_count': severity_counts.get('critical', 0),
                'high_count': severity_counts.get('high', 0),
                'medium_count': severity_counts.get('medium', 0),
                'low_count': severity_counts.get('low', 0),
                'issues': issues,
                'summary_by_type': dict(summary_by_type),
                'detected_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Issue detection failed for dataset {dataset_id}: {e}")
            raise IssueDetectionError(dataset_id, 'general', str(e))
    
    def _empty_result(self, dataset_id: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'dataset_id': dataset_id,
            'total_issues': 0,
            'critical_count': 0,
            'high_count': 0,
            'medium_count': 0,
            'low_count': 0,
            'issues': [],
            'summary_by_type': {},
            'detected_at': datetime.utcnow().isoformat()
        }
    
    def _detect_missing_values(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测缺失值问题"""
        issues = []
        
        for cp in profile.column_profiles:
            if cp.null_count == 0:
                continue
            
            null_rate = cp.null_count / cp.total_count if cp.total_count > 0 else 0
            
            # 根据缺失率确定严重程度
            if null_rate > 0.5:
                severity = 'critical'
            elif null_rate > 0.2:
                severity = 'high'
            elif null_rate > 0.05:
                severity = 'medium'
            else:
                severity = 'low'
            
            issues.append({
                'issue_id': f'missing_{cp.column_name}_{str(uuid.uuid4())[:8]}',
                'issue_type': 'missing_values',
                'severity': severity,
                'column_name': cp.column_name,
                'description': f"列 '{cp.column_name}' 存在 {cp.null_count} 个缺失值 ({null_rate:.1%})",
                'affected_count': cp.null_count,
                'affected_rate': round(null_rate, 4),
                'sample_values': [],
                'recommendation': self._get_missing_recommendation(cp, null_rate),
                'auto_fixable': True,
                'detected_at': datetime.utcnow().isoformat()
            })
        
        return issues
    
    def _detect_duplicates(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测重复记录"""
        if profile.duplicate_rows_count == 0:
            return []
        
        dup_rate = profile.duplicate_rows_count / profile.total_rows if profile.total_rows > 0 else 0
        
        if dup_rate > 0.2:
            severity = 'high'
        elif dup_rate > 0.05:
            severity = 'medium'
        else:
            severity = 'low'
        
        return [{
            'issue_id': f'duplicates_{str(uuid.uuid4())[:8]}',
            'issue_type': 'duplicate_records',
            'severity': severity,
            'column_name': None,
            'description': f"发现 {profile.duplicate_rows_count} 条重复记录 ({dup_rate:.1%})",
            'affected_count': profile.duplicate_rows_count,
            'affected_rate': round(dup_rate, 4),
            'sample_values': profile.duplicate_row_indices[:5],
            'recommendation': '建议删除重复记录以提高数据质量',
            'auto_fixable': True,
            'detected_at': datetime.utcnow().isoformat()
        }]
    
    def _detect_outliers(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测异常值"""
        issues = []
        
        for cp in profile.column_profiles:
            if cp.data_type != 'numeric' or cp.outliers_count == 0:
                continue
            
            outlier_rate = cp.outliers_count / cp.total_count if cp.total_count > 0 else 0
            
            if outlier_rate > 0.1:
                severity = 'high'
            elif outlier_rate > 0.05:
                severity = 'medium'
            else:
                severity = 'low'
            
            issues.append({
                'issue_id': f'outliers_{cp.column_name}_{str(uuid.uuid4())[:8]}',
                'issue_type': 'outliers',
                'severity': severity,
                'column_name': cp.column_name,
                'description': f"列 '{cp.column_name}' 存在 {cp.outliers_count} 个异常值 ({outlier_rate:.1%})",
                'affected_count': cp.outliers_count,
                'affected_rate': round(outlier_rate, 4),
                'sample_values': cp.outliers_indices[:5],
                'recommendation': f"检查数据录入错误或使用统计方法处理异常值（均值: {cp.mean:.2f}, 标准差: {cp.std:.2f}）" if cp.mean and cp.std else "检查数据录入错误",
                'auto_fixable': True,
                'detected_at': datetime.utcnow().isoformat()
            })
        
        return issues
    
    def _detect_format_issues(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测格式不一致问题"""
        issues = []
        
        for cp in profile.column_profiles:
            if cp.detected_pattern and cp.pattern_match_rate < 0.95:
                mismatch_count = int(cp.total_count * (1 - cp.pattern_match_rate))
                
                if cp.pattern_match_rate < 0.7:
                    severity = 'high'
                elif cp.pattern_match_rate < 0.9:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                issues.append({
                    'issue_id': f'format_{cp.column_name}_{str(uuid.uuid4())[:8]}',
                    'issue_type': 'inconsistent_format',
                    'severity': severity,
                    'column_name': cp.column_name,
                    'description': f"列 '{cp.column_name}' 的格式不一致，预期模式: {cp.detected_pattern}，匹配率: {cp.pattern_match_rate:.1%}",
                    'affected_count': mismatch_count,
                    'affected_rate': round(1 - cp.pattern_match_rate, 4),
                    'sample_values': [],
                    'recommendation': f"建议统一格式为 {cp.detected_pattern}",
                    'auto_fixable': False,
                    'detected_at': datetime.utcnow().isoformat()
                })
        
        return issues
    
    def _detect_invalid_values(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测无效值"""
        issues = []
        
        for cp in profile.column_profiles:
            # 检查数值列的范围问题
            if cp.data_type == 'numeric':
                if cp.min_value is not None and cp.max_value is not None:
                    range_size = cp.max_value - cp.min_value
                    if cp.std and range_size > 0:
                        # 如果标准差很大，可能有异常值导致的范围过大
                        cv = cp.std / abs(cp.mean) if cp.mean and cp.mean != 0 else 0
                        if cv > 2:  # 变异系数过大
                            issues.append({
                                'issue_id': f'range_{cp.column_name}_{str(uuid.uuid4())[:8]}',
                                'issue_type': 'invalid_values',
                                'severity': 'medium',
                                'column_name': cp.column_name,
                                'description': f"列 '{cp.column_name}' 的值范围过大 (min: {cp.min_value}, max: {cp.max_value})，可能存在无效值",
                                'affected_count': cp.outliers_count,
                                'affected_rate': cp.outliers_count / cp.total_count if cp.total_count > 0 else 0,
                                'sample_values': [],
                                'recommendation': '检查是否有数据录入错误或异常极值',
                                'auto_fixable': False,
                                'detected_at': datetime.utcnow().isoformat()
                            })
        
        return issues
    
    def _detect_type_mismatch(
        self,
        data: List[Dict],
        profile: DatasetProfile
    ) -> List[Dict[str, Any]]:
        """检测数据类型不匹配"""
        # 简化实现：在实际场景中，可以与预定义的schema进行对比
        return []
    
    def _get_missing_recommendation(
        self,
        profile: ColumnProfile,
        null_rate: float
    ) -> str:
        """获取缺失值处理建议"""
        if null_rate > 0.7:
            return '缺失率过高，建议删除此列或调查数据来源'
        elif profile.data_type == 'numeric':
            return '建议使用均值或中位数填充'
        elif profile.data_type == 'categorical':
            return '建议使用众数填充或创建"未知"类别'
        else:
            return '建议删除包含缺失值的记录或使用适当方法填充'


# ============================================================================
# 数据清理器
# ============================================================================

class DataCleaner:
    """数据清理器
    
    执行数据清理操作。
    """
    
    def __init__(self):
        """初始化数据清理器"""
        pass
    
    def clean(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
        dataset_id: str
    ) -> Dict[str, Any]:
        """执行数据清理
        
        Args:
            data: 原始数据
            config: 清理配置
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 清理结果
        """
        try:
            start_time = datetime.utcnow()
            original_count = len(data)
            cleaned_data = data.copy()
            operation_results = []
            
            # 执行配置的清理操作
            operations = config.get('operations', [])
            
            for op in operations:
                if not op.get('enabled', True):
                    continue
                
                strategy = op.get('strategy')
                target_column = op.get('target_column')
                params = op.get('parameters', {})
                
                result = self._execute_operation(
                    cleaned_data, strategy, target_column, params
                )
                
                if result['success']:
                    cleaned_data = result['data']
                
                operation_results.append({
                    'operation_id': op.get('operation_id', str(uuid.uuid4())),
                    'strategy': strategy,
                    'target_column': target_column,
                    'success': result['success'],
                    'records_affected': result['affected_count'],
                    'execution_time_ms': result['execution_time_ms'],
                    'error_message': result.get('error')
                })
            
            # 全局清理选项
            if config.get('remove_duplicates', False):
                result = self._remove_duplicates(cleaned_data)
                cleaned_data = result['data']
                operation_results.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': 'drop_duplicates',
                    'target_column': None,
                    'success': True,
                    'records_affected': result['affected_count'],
                    'execution_time_ms': 0
                })
            
            if config.get('handle_missing_values', False):
                strategy = config.get('missing_value_strategy', 'fill_median')
                result = self._handle_all_missing_values(
                    cleaned_data, strategy, config.get('missing_threshold', 0.7)
                )
                cleaned_data = result['data']
                operation_results.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': strategy,
                    'target_column': None,
                    'success': True,
                    'records_affected': result['affected_count'],
                    'execution_time_ms': 0
                })
            
            if config.get('handle_outliers', False):
                strategy = config.get('outlier_strategy', 'clip_values')
                threshold = config.get('outlier_std_threshold', 3.0)
                result = self._handle_all_outliers(cleaned_data, strategy, threshold)
                cleaned_data = result['data']
                operation_results.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': strategy,
                    'target_column': None,
                    'success': True,
                    'records_affected': result['affected_count'],
                    'execution_time_ms': 0
                })
            
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'dataset_id': dataset_id,
                'cleaned_data': cleaned_data,
                'original_record_count': original_count,
                'cleaned_record_count': len(cleaned_data),
                'total_records_affected': sum(r['records_affected'] for r in operation_results),
                'operation_results': operation_results,
                'execution_time_ms': execution_time_ms,
                'cleaned_at': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data cleaning failed for dataset {dataset_id}: {e}")
            raise DataCleaningError(dataset_id, 'general', str(e))
    
    def _execute_operation(
        self,
        data: List[Dict],
        strategy: str,
        target_column: Optional[str],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行单个清理操作"""
        start_time = datetime.utcnow()
        
        try:
            handlers = {
                'drop_rows': self._drop_rows_with_missing,
                'fill_mean': lambda d, c, p: self._fill_missing(d, c, 'mean'),
                'fill_median': lambda d, c, p: self._fill_missing(d, c, 'median'),
                'fill_mode': lambda d, c, p: self._fill_missing(d, c, 'mode'),
                'fill_constant': lambda d, c, p: self._fill_constant(d, c, p.get('value')),
                'drop_duplicates': lambda d, c, p: self._remove_duplicates(d),
                'clip_values': lambda d, c, p: self._clip_values(d, c, p.get('min'), p.get('max')),
                'remove_outliers': lambda d, c, p: self._remove_outliers(d, c, p.get('threshold', 3.0)),
            }
            
            handler = handlers.get(strategy)
            if not handler:
                return {
                    'success': False,
                    'data': data,
                    'affected_count': 0,
                    'execution_time_ms': 0,
                    'error': f'Unknown strategy: {strategy}'
                }
            
            result = handler(data, target_column, params)
            end_time = datetime.utcnow()
            
            return {
                'success': True,
                'data': result['data'],
                'affected_count': result['affected_count'],
                'execution_time_ms': (end_time - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                'success': False,
                'data': data,
                'affected_count': 0,
                'execution_time_ms': 0,
                'error': str(e)
            }
    
    def _drop_rows_with_missing(
        self,
        data: List[Dict],
        column: Optional[str],
        params: Dict
    ) -> Dict[str, Any]:
        """删除包含缺失值的行"""
        original_count = len(data)
        
        if column:
            cleaned = [row for row in data if row.get(column) is not None and row.get(column) != '']
        else:
            cleaned = [row for row in data if all(v is not None and v != '' for v in row.values())]
        
        return {
            'data': cleaned,
            'affected_count': original_count - len(cleaned)
        }
    
    def _fill_missing(
        self,
        data: List[Dict],
        column: str,
        method: str
    ) -> Dict[str, Any]:
        """填充缺失值"""
        if not column:
            return {'data': data, 'affected_count': 0}
        
        # 获取非空值
        non_null_values = [
            row[column] for row in data
            if column in row and row[column] is not None and row[column] != ''
        ]
        
        if not non_null_values:
            return {'data': data, 'affected_count': 0}
        
        # 计算填充值
        try:
            numeric_values = [float(v) for v in non_null_values]
            if method == 'mean':
                fill_value = statistics.mean(numeric_values)
            elif method == 'median':
                fill_value = statistics.median(numeric_values)
            else:  # mode
                fill_value = statistics.mode(numeric_values)
        except (ValueError, statistics.StatisticsError):
            # 非数值类型，使用众数
            fill_value = max(set(non_null_values), key=non_null_values.count)
        
        # 填充
        affected_count = 0
        for row in data:
            if column in row and (row[column] is None or row[column] == ''):
                row[column] = fill_value
                affected_count += 1
        
        return {
            'data': data,
            'affected_count': affected_count
        }
    
    def _fill_constant(
        self,
        data: List[Dict],
        column: str,
        value: Any
    ) -> Dict[str, Any]:
        """使用常量填充缺失值"""
        if not column:
            return {'data': data, 'affected_count': 0}
        
        affected_count = 0
        for row in data:
            if column in row and (row[column] is None or row[column] == ''):
                row[column] = value
                affected_count += 1
        
        return {
            'data': data,
            'affected_count': affected_count
        }
    
    def _remove_duplicates(self, data: List[Dict]) -> Dict[str, Any]:
        """删除重复记录"""
        seen = set()
        unique_data = []
        
        for row in data:
            key = tuple(sorted((k, str(v)) for k, v in row.items()))
            if key not in seen:
                seen.add(key)
                unique_data.append(row)
        
        return {
            'data': unique_data,
            'affected_count': len(data) - len(unique_data)
        }
    
    def _clip_values(
        self,
        data: List[Dict],
        column: str,
        min_val: Optional[float],
        max_val: Optional[float]
    ) -> Dict[str, Any]:
        """截断值到指定范围"""
        if not column:
            return {'data': data, 'affected_count': 0}
        
        affected_count = 0
        for row in data:
            if column in row and row[column] is not None:
                try:
                    val = float(row[column])
                    original_val = val
                    if min_val is not None and val < min_val:
                        val = min_val
                    if max_val is not None and val > max_val:
                        val = max_val
                    if val != original_val:
                        row[column] = val
                        affected_count += 1
                except (ValueError, TypeError):
                    pass
        
        return {
            'data': data,
            'affected_count': affected_count
        }
    
    def _remove_outliers(
        self,
        data: List[Dict],
        column: str,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """删除异常值"""
        if not column:
            return {'data': data, 'affected_count': 0}
        
        # 计算统计量
        values = []
        for row in data:
            if column in row and row[column] is not None:
                try:
                    values.append(float(row[column]))
                except (ValueError, TypeError):
                    pass
        
        if len(values) < 2:
            return {'data': data, 'affected_count': 0}
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return {'data': data, 'affected_count': 0}
        
        # 过滤异常值
        original_count = len(data)
        cleaned = []
        for row in data:
            if column in row and row[column] is not None:
                try:
                    val = float(row[column])
                    z_score = abs((val - mean) / std)
                    if z_score <= threshold:
                        cleaned.append(row)
                except (ValueError, TypeError):
                    cleaned.append(row)
            else:
                cleaned.append(row)
        
        return {
            'data': cleaned,
            'affected_count': original_count - len(cleaned)
        }
    
    def _handle_all_missing_values(
        self,
        data: List[Dict],
        strategy: str,
        threshold: float
    ) -> Dict[str, Any]:
        """处理所有列的缺失值"""
        if not data:
            return {'data': data, 'affected_count': 0}
        
        columns = list(data[0].keys())
        total_affected = 0
        
        for column in columns:
            # 计算该列的缺失率
            null_count = sum(1 for row in data if row.get(column) is None or row.get(column) == '')
            null_rate = null_count / len(data) if data else 0
            
            if null_rate > threshold:
                # 缺失率过高，删除该列
                for row in data:
                    if column in row:
                        del row[column]
                        total_affected += 1
            elif null_count > 0:
                # 使用策略填充
                method_map = {
                    'fill_mean': 'mean',
                    'fill_median': 'median',
                    'fill_mode': 'mode'
                }
                method = method_map.get(strategy, 'median')
                result = self._fill_missing(data, column, method)
                total_affected += result['affected_count']
        
        return {
            'data': data,
            'affected_count': total_affected
        }
    
    def _handle_all_outliers(
        self,
        data: List[Dict],
        strategy: str,
        threshold: float
    ) -> Dict[str, Any]:
        """处理所有数值列的异常值"""
        if not data:
            return {'data': data, 'affected_count': 0}
        
        columns = list(data[0].keys())
        total_affected = 0
        
        for column in columns:
            # 检查是否为数值列
            sample_values = [row.get(column) for row in data[:100] if row.get(column) is not None]
            numeric_count = sum(1 for v in sample_values if self._is_numeric(v))
            
            if numeric_count / len(sample_values) > 0.8 if sample_values else False:
                if strategy == 'clip_values':
                    # 计算边界
                    values = [float(row[column]) for row in data if row.get(column) is not None and self._is_numeric(row[column])]
                    if len(values) >= 2:
                        mean = statistics.mean(values)
                        std = statistics.stdev(values)
                        min_val = mean - threshold * std
                        max_val = mean + threshold * std
                        result = self._clip_values(data, column, min_val, max_val)
                        total_affected += result['affected_count']
                elif strategy == 'remove_outliers':
                    result = self._remove_outliers(data, column, threshold)
                    data = result['data']
                    total_affected += result['affected_count']
        
        return {
            'data': data,
            'affected_count': total_affected
        }
    
    def _is_numeric(self, value: Any) -> bool:
        """检查是否为数值"""
        if isinstance(value, (int, float)):
            return True
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False


# ============================================================================
# 质量规则验证器
# ============================================================================

class QualityRuleValidator:
    """质量规则验证器
    
    验证数据是否符合定义的质量规则。
    """
    
    def __init__(self):
        """初始化规则验证器"""
        pass
    
    def validate_rules(
        self,
        data: List[Dict[str, Any]],
        rules: List[Dict[str, Any]],
        dataset_id: str,
        stop_on_failure: bool = False
    ) -> Dict[str, Any]:
        """验证规则集
        
        Args:
            data: 数据列表
            rules: 规则列表
            dataset_id: 数据集ID
            stop_on_failure: 是否在失败时停止
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = []
        passed_count = 0
        
        for rule in rules:
            if not rule.get('enabled', True):
                continue
            
            try:
                result = self._validate_single_rule(data, rule)
                results.append(result)
                
                if result['passed']:
                    passed_count += 1
                elif stop_on_failure:
                    break
                    
            except Exception as e:
                results.append({
                    'rule_id': rule.get('rule_id'),
                    'rule_name': rule.get('name', 'Unknown'),
                    'passed': False,
                    'violations_count': len(data),
                    'violation_rate': 1.0,
                    'sample_violations': [],
                    'execution_time_ms': 0,
                    'error': str(e)
                })
                if stop_on_failure:
                    break
        
        total_rules = len([r for r in results])
        pass_rate = passed_count / total_rules if total_rules > 0 else 0
        
        return {
            'dataset_id': dataset_id,
            'total_rules': total_rules,
            'passed_rules': passed_count,
            'failed_rules': total_rules - passed_count,
            'pass_rate': round(pass_rate, 4),
            'results': results,
            'validated_at': datetime.utcnow().isoformat()
        }
    
    def _validate_single_rule(
        self,
        data: List[Dict],
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证单个规则"""
        start_time = datetime.utcnow()
        
        rule_type = rule.get('rule_type')
        target_column = rule.get('target_column')
        condition = rule.get('condition', '')
        params = rule.get('parameters', {})
        
        validators = {
            'completeness_rule': self._validate_completeness,
            'range_rule': self._validate_range,
            'pattern_rule': self._validate_pattern,
            'uniqueness_rule': self._validate_uniqueness,
            'custom_rule': self._validate_custom,
        }
        
        validator = validators.get(rule_type, self._validate_custom)
        violations = validator(data, target_column, condition, params)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        violation_count = len(violations)
        violation_rate = violation_count / len(data) if data else 0
        
        return {
            'rule_id': rule.get('rule_id'),
            'rule_name': rule.get('name', 'Unknown'),
            'passed': violation_count == 0,
            'violations_count': violation_count,
            'violation_rate': round(violation_rate, 4),
            'sample_violations': violations[:10],
            'execution_time_ms': round(execution_time, 2)
        }
    
    def _validate_completeness(
        self,
        data: List[Dict],
        column: str,
        condition: str,
        params: Dict
    ) -> List[Any]:
        """验证完整性规则"""
        min_completeness = params.get('min_completeness', 1.0)
        
        null_count = sum(
            1 for row in data
            if column not in row or row[column] is None or row[column] == ''
        )
        
        completeness = 1 - (null_count / len(data)) if data else 0
        
        if completeness >= min_completeness:
            return []
        
        return [i for i, row in enumerate(data) if column not in row or row[column] is None or row[column] == '']
    
    def _validate_range(
        self,
        data: List[Dict],
        column: str,
        condition: str,
        params: Dict
    ) -> List[Any]:
        """验证范围规则"""
        min_val = params.get('min')
        max_val = params.get('max')
        
        violations = []
        for i, row in enumerate(data):
            if column not in row or row[column] is None:
                continue
            
            try:
                val = float(row[column])
                if min_val is not None and val < min_val:
                    violations.append({'index': i, 'value': val, 'reason': f'< {min_val}'})
                if max_val is not None and val > max_val:
                    violations.append({'index': i, 'value': val, 'reason': f'> {max_val}'})
            except (ValueError, TypeError):
                violations.append({'index': i, 'value': row[column], 'reason': 'not numeric'})
        
        return violations
    
    def _validate_pattern(
        self,
        data: List[Dict],
        column: str,
        condition: str,
        params: Dict
    ) -> List[Any]:
        """验证模式规则"""
        pattern = params.get('pattern', condition)
        
        if not pattern:
            return []
        
        violations = []
        for i, row in enumerate(data):
            if column not in row or row[column] is None:
                continue
            
            val = str(row[column])
            if not re.match(pattern, val):
                violations.append({'index': i, 'value': val})
        
        return violations
    
    def _validate_uniqueness(
        self,
        data: List[Dict],
        column: str,
        condition: str,
        params: Dict
    ) -> List[Any]:
        """验证唯一性规则"""
        seen = {}
        violations = []
        
        for i, row in enumerate(data):
            if column not in row:
                continue
            
            val = row[column]
            if val in seen:
                violations.append({'index': i, 'value': val, 'duplicate_of': seen[val]})
            else:
                seen[val] = i
        
        return violations
    
    def _validate_custom(
        self,
        data: List[Dict],
        column: str,
        condition: str,
        params: Dict
    ) -> List[Any]:
        """验证自定义规则"""
        # 简化实现：自定义规则暂不支持
        return []


# ============================================================================
# 数据质量模块主类
# ============================================================================

class DataQualityModule:
    """数据质量模块
    
    整合数据质量评估、问题检测、数据清理等功能。
    """
    
    def __init__(self):
        """初始化数据质量模块"""
        self.profiler = DataProfiler()
        self.assessor = QualityAssessor(self.profiler)
        self.detector = IssueDetector(self.profiler)
        self.cleaner = DataCleaner()
        self.validator = QualityRuleValidator()
        logger.info("DataQualityModule initialized")
    
    def profile_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        sample_size: Optional[int] = None
    ) -> DatasetProfile:
        """剖析数据集
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            sample_size: 采样大小
            
        Returns:
            DatasetProfile: 剖析结果
        """
        return self.profiler.profile_dataset(data, dataset_id, sample_size)
    
    def assess_quality(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """评估数据质量
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            dimensions: 要评估的维度
            
        Returns:
            Dict[str, Any]: 质量评估结果
        """
        return self.assessor.assess(data, dataset_id, dimensions)
    
    def detect_issues(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        issue_types: Optional[List[str]] = None,
        severity_threshold: str = 'low',
        max_issues: int = 100
    ) -> Dict[str, Any]:
        """检测数据问题
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            issue_types: 要检测的问题类型
            severity_threshold: 严重程度阈值
            max_issues: 最大返回问题数
            
        Returns:
            Dict[str, Any]: 问题检测结果
        """
        return self.detector.detect(
            data, dataset_id, issue_types, severity_threshold, max_issues
        )
    
    def clean_data(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
        dataset_id: str
    ) -> Dict[str, Any]:
        """清理数据
        
        Args:
            data: 原始数据
            config: 清理配置
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 清理结果
        """
        return self.cleaner.clean(data, config, dataset_id)
    
    def validate_rules(
        self,
        data: List[Dict[str, Any]],
        rules: List[Dict[str, Any]],
        dataset_id: str,
        stop_on_failure: bool = False
    ) -> Dict[str, Any]:
        """验证质量规则
        
        Args:
            data: 数据列表
            rules: 规则列表
            dataset_id: 数据集ID
            stop_on_failure: 是否在失败时停止
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        return self.validator.validate_rules(data, rules, dataset_id, stop_on_failure)
    
    def generate_report(
        self,
        data: List[Dict[str, Any]],
        dataset_id: str,
        dataset_name: str = ""
    ) -> Dict[str, Any]:
        """生成质量报告
        
        Args:
            data: 数据列表
            dataset_id: 数据集ID
            dataset_name: 数据集名称
            
        Returns:
            Dict[str, Any]: 质量报告
        """
        # 评估质量
        quality_metrics = self.assess_quality(data, dataset_id)
        
        # 检测问题
        issues = self.detect_issues(data, dataset_id)
        
        # 生成清理建议
        recommendations = self._generate_recommendations(issues)
        
        return {
            'report_id': str(uuid.uuid4()),
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'total_records': len(data),
            'total_columns': len(data[0].keys()) if data else 0,
            'quality_metrics': quality_metrics,
            'overall_score': quality_metrics.get('overall_score', 0),
            'issue_detection': issues,
            'cleaning_recommendations': recommendations,
            'summary': self._generate_summary(quality_metrics, issues),
            'recommendations': self._generate_text_recommendations(quality_metrics, issues),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(
        self,
        issues: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成清理建议操作"""
        recommendations = []
        
        for issue in issues.get('issues', []):
            issue_type = issue.get('issue_type')
            column = issue.get('column_name')
            
            if issue_type == 'missing_values':
                recommendations.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': 'fill_median',
                    'target_column': column,
                    'parameters': {},
                    'priority': 1
                })
            elif issue_type == 'duplicate_records':
                recommendations.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': 'drop_duplicates',
                    'target_column': None,
                    'parameters': {},
                    'priority': 0
                })
            elif issue_type == 'outliers':
                recommendations.append({
                    'operation_id': str(uuid.uuid4()),
                    'strategy': 'clip_values',
                    'target_column': column,
                    'parameters': {},
                    'priority': 2
                })
        
        return recommendations
    
    def _generate_summary(
        self,
        metrics: Dict[str, Any],
        issues: Dict[str, Any]
    ) -> str:
        """生成报告摘要"""
        score = metrics.get('overall_score', 0)
        total_issues = issues.get('total_issues', 0)
        
        if score >= 0.9:
            quality_level = '优秀'
        elif score >= 0.7:
            quality_level = '良好'
        elif score >= 0.5:
            quality_level = '一般'
        else:
            quality_level = '较差'
        
        return f"数据质量评分: {score:.2%}（{quality_level}），共发现 {total_issues} 个质量问题。"
    
    def _generate_text_recommendations(
        self,
        metrics: Dict[str, Any],
        issues: Dict[str, Any]
    ) -> List[str]:
        """生成文本建议"""
        recommendations = []
        
        critical_count = issues.get('critical_count', 0)
        high_count = issues.get('high_count', 0)
        
        if critical_count > 0:
            recommendations.append(f"发现 {critical_count} 个严重问题，建议优先处理")
        
        if high_count > 0:
            recommendations.append(f"发现 {high_count} 个高优先级问题，建议尽快处理")
        
        missing_rate = metrics.get('missing_values_rate', 0)
        if missing_rate > 0.1:
            recommendations.append(f"缺失值率 {missing_rate:.1%} 较高，建议使用适当方法填充")
        
        duplicate_rate = metrics.get('duplicate_records_rate', 0)
        if duplicate_rate > 0.01:
            recommendations.append(f"存在 {duplicate_rate:.1%} 的重复记录，建议去重")
        
        if not recommendations:
            recommendations.append("数据质量良好，无需立即处理")
        
        return recommendations
