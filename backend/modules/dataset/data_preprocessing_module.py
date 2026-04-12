"""数据预处理核心模块

实现数据预处理的底层核心逻辑，包括数据清洗、特征工程、数据增强、数据分割等功能。
"""

import copy
import hashlib
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    DataSplitRatioError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class ColumnStats:
    """列统计信息"""
    name: str
    dtype: str
    count: int = 0
    null_count: int = 0
    unique_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    q1: Optional[float] = None
    q3: Optional[float] = None


@dataclass
class PreprocessingContext:
    """预处理上下文"""
    dataset_id: str
    user_id: str
    tenant_id: Optional[str] = None
    data: List[Dict[str, Any]] = field(default_factory=list)
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    operation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationResult:
    """操作结果"""
    operation_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    rows_affected: int = 0
    columns_affected: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 数据清洗处理器
# ============================================================================

class DataCleaningProcessor:
    """数据清洗处理器
    
    实现数据清洗相关的操作，包括去重、处理缺失值、处理异常值等。
    """
    
    def __init__(self):
        """初始化数据清洗处理器"""
        self._stats_cache: Dict[str, ColumnStats] = {}
    
    def remove_duplicates(
        self,
        data: List[Dict[str, Any]],
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """去除重复数据
        
        Args:
            data: 数据列表
            subset: 用于判断重复的列（None表示所有列）
            keep: 保留策略：first/last/none
            
        Returns:
            去重后的数据和删除的行数
        """
        if not data:
            return data, 0
        
        seen = set()
        result = []
        duplicates_count = 0
        
        # 如果是keep=last，需要反向处理
        items = list(reversed(data)) if keep == "last" else data
        
        for row in items:
            # 生成用于比较的键
            if subset:
                key_values = tuple(row.get(col) for col in subset)
            else:
                key_values = tuple(sorted(row.items()))
            
            key = hashlib.md5(str(key_values).encode()).hexdigest()
            
            if key not in seen:
                seen.add(key)
                if keep != "none":
                    result.append(row)
            else:
                duplicates_count += 1
        
        # 如果是last，需要再次反转
        if keep == "last":
            result = list(reversed(result))
        
        logger.info(f"Removed {duplicates_count} duplicate rows")
        return result, duplicates_count
    
    def handle_missing_values(
        self,
        data: List[Dict[str, Any]],
        strategy: str,
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None,
        threshold: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """处理缺失值
        
        Args:
            data: 数据列表
            strategy: 处理策略
            columns: 目标列
            fill_value: 填充值
            threshold: 缺失比例阈值
            
        Returns:
            处理后的数据和各列处理数量
        """
        if not data:
            return data, {}
        
        result = copy.deepcopy(data)
        affected_counts: Dict[str, int] = {}
        
        # 确定要处理的列
        target_columns = columns or list(data[0].keys())
        
        for col in target_columns:
            # 计算缺失值
            null_indices = []
            non_null_values = []
            
            for i, row in enumerate(result):
                val = row.get(col)
                if val is None or (isinstance(val, str) and val.strip() == ''):
                    null_indices.append(i)
                else:
                    non_null_values.append(val)
            
            if not null_indices:
                continue
            
            # 检查缺失比例
            null_ratio = len(null_indices) / len(result)
            if null_ratio > threshold:
                logger.warning(f"Column {col} has {null_ratio:.2%} missing values, exceeds threshold")
                # 可以选择删除该列或继续处理
            
            # 根据策略处理
            fill_val = None
            
            if strategy == "drop":
                # 删除包含缺失值的行
                result = [row for i, row in enumerate(result) if i not in null_indices]
                affected_counts[col] = len(null_indices)
                continue
            
            elif strategy == "fill_mean":
                if non_null_values:
                    try:
                        fill_val = sum(float(v) for v in non_null_values) / len(non_null_values)
                    except (TypeError, ValueError):
                        fill_val = fill_value
            
            elif strategy == "fill_median":
                if non_null_values:
                    try:
                        sorted_vals = sorted(float(v) for v in non_null_values)
                        n = len(sorted_vals)
                        fill_val = sorted_vals[n // 2] if n % 2 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
                    except (TypeError, ValueError):
                        fill_val = fill_value
            
            elif strategy == "fill_mode":
                if non_null_values:
                    from collections import Counter
                    counter = Counter(non_null_values)
                    fill_val = counter.most_common(1)[0][0] if counter else fill_value
            
            elif strategy == "fill_constant":
                fill_val = fill_value
            
            elif strategy == "fill_forward":
                # 前向填充
                last_valid = None
                for i, row in enumerate(result):
                    val = row.get(col)
                    if val is not None and not (isinstance(val, str) and val.strip() == ''):
                        last_valid = val
                    elif last_valid is not None:
                        result[i][col] = last_valid
                affected_counts[col] = len(null_indices)
                continue
            
            elif strategy == "fill_backward":
                # 后向填充
                next_valid = None
                for i in range(len(result) - 1, -1, -1):
                    val = result[i].get(col)
                    if val is not None and not (isinstance(val, str) and val.strip() == ''):
                        next_valid = val
                    elif next_valid is not None:
                        result[i][col] = next_valid
                affected_counts[col] = len(null_indices)
                continue
            
            # 应用填充值
            if fill_val is not None:
                for i in null_indices:
                    if i < len(result):
                        result[i][col] = fill_val
                affected_counts[col] = len(null_indices)
        
        logger.info(f"Handled missing values: {affected_counts}")
        return result, affected_counts
    
    def handle_outliers(
        self,
        data: List[Dict[str, Any]],
        method: str,
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        action: str = "remove"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """处理异常值
        
        Args:
            data: 数据列表
            method: 检测方法
            columns: 目标列
            threshold: 阈值
            action: 处理方式
            
        Returns:
            处理后的数据和各列处理数量
        """
        if not data:
            return data, {}
        
        result = copy.deepcopy(data)
        affected_counts: Dict[str, int] = {}
        outlier_indices: set = set()
        
        # 确定要处理的列（只处理数值列）
        target_columns = columns or list(data[0].keys())
        
        for col in target_columns:
            # 获取数值
            values = []
            indices = []
            
            for i, row in enumerate(result):
                val = row.get(col)
                if val is not None:
                    try:
                        values.append((i, float(val)))
                        indices.append(i)
                    except (TypeError, ValueError):
                        continue
            
            if len(values) < 3:
                continue
            
            numeric_values = [v[1] for v in values]
            col_outlier_indices = []
            
            if method == "iqr":
                # IQR方法
                sorted_vals = sorted(numeric_values)
                n = len(sorted_vals)
                q1 = sorted_vals[n // 4]
                q3 = sorted_vals[3 * n // 4]
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                
                for idx, val in values:
                    if val < lower or val > upper:
                        col_outlier_indices.append(idx)
            
            elif method == "zscore":
                # Z分数方法
                mean_val = sum(numeric_values) / len(numeric_values)
                std_val = (sum((x - mean_val) ** 2 for x in numeric_values) / len(numeric_values)) ** 0.5
                
                if std_val > 0:
                    for idx, val in values:
                        z = abs(val - mean_val) / std_val
                        if z > threshold:
                            col_outlier_indices.append(idx)
            
            elif method == "percentile":
                # 百分位数方法
                sorted_vals = sorted(numeric_values)
                n = len(sorted_vals)
                lower_idx = int(n * (threshold / 100))
                upper_idx = int(n * (1 - threshold / 100))
                lower = sorted_vals[lower_idx]
                upper = sorted_vals[upper_idx]
                
                for idx, val in values:
                    if val < lower or val > upper:
                        col_outlier_indices.append(idx)
            
            affected_counts[col] = len(col_outlier_indices)
            outlier_indices.update(col_outlier_indices)
            
            # 根据action处理
            if action == "cap":
                # 封顶处理
                sorted_vals = sorted(numeric_values)
                n = len(sorted_vals)
                lower_cap = sorted_vals[int(n * 0.01)]
                upper_cap = sorted_vals[int(n * 0.99)]
                
                for idx, val in values:
                    if val < lower_cap:
                        result[idx][col] = lower_cap
                    elif val > upper_cap:
                        result[idx][col] = upper_cap
        
        # 如果action是remove，删除包含异常值的行
        if action == "remove":
            result = [row for i, row in enumerate(result) if i not in outlier_indices]
        
        logger.info(f"Handled outliers: {affected_counts}, action={action}")
        return result, affected_counts
    
    def filter_rows(
        self,
        data: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]],
        logic: str = "and"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """过滤行
        
        Args:
            data: 数据列表
            conditions: 过滤条件
            logic: 条件逻辑
            
        Returns:
            过滤后的数据和过滤的行数
        """
        if not data or not conditions:
            return data, 0
        
        result = []
        
        for row in data:
            condition_results = []
            
            for cond in conditions:
                col = cond.get("column")
                op = cond.get("operator", "==")
                value = cond.get("value")
                
                row_val = row.get(col)
                
                try:
                    if op == "==":
                        match = row_val == value
                    elif op == "!=":
                        match = row_val != value
                    elif op == ">":
                        match = float(row_val) > float(value)
                    elif op == ">=":
                        match = float(row_val) >= float(value)
                    elif op == "<":
                        match = float(row_val) < float(value)
                    elif op == "<=":
                        match = float(row_val) <= float(value)
                    elif op == "in":
                        match = row_val in value
                    elif op == "not_in":
                        match = row_val not in value
                    elif op == "contains":
                        match = value in str(row_val) if row_val else False
                    elif op == "not_contains":
                        match = value not in str(row_val) if row_val else True
                    elif op == "is_null":
                        match = row_val is None or (isinstance(row_val, str) and row_val.strip() == '')
                    elif op == "not_null":
                        match = row_val is not None and not (isinstance(row_val, str) and row_val.strip() == '')
                    else:
                        match = True
                except (TypeError, ValueError):
                    match = False
                
                condition_results.append(match)
            
            # 应用逻辑
            if logic == "and":
                keep = all(condition_results)
            else:
                keep = any(condition_results)
            
            if keep:
                result.append(row)
        
        filtered_count = len(data) - len(result)
        logger.info(f"Filtered {filtered_count} rows")
        return result, filtered_count


# ============================================================================
# 数据转换处理器
# ============================================================================

class DataTransformProcessor:
    """数据转换处理器
    
    实现数据转换相关的操作，包括标准化、编码、分词等。
    """
    
    def __init__(self):
        """初始化数据转换处理器"""
        self._encoders: Dict[str, Dict[str, int]] = {}
        self._scalers: Dict[str, Dict[str, float]] = {}
    
    def normalize(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        method: str = "min_max",
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """标准化数据
        
        Args:
            data: 数据列表
            columns: 目标列
            method: 标准化方法
            feature_range: 目标范围
            
        Returns:
            标准化后的数据和标准化参数
        """
        if not data or not columns:
            return data, {}
        
        result = copy.deepcopy(data)
        scaler_params: Dict[str, Any] = {}
        
        for col in columns:
            # 获取数值
            values = []
            for row in result:
                val = row.get(col)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        continue
            
            if not values:
                continue
            
            if method == "min_max":
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val != min_val else 1
                
                scaler_params[col] = {
                    "method": method,
                    "min": min_val,
                    "max": max_val,
                    "feature_range": feature_range
                }
                
                for row in result:
                    val = row.get(col)
                    if val is not None:
                        try:
                            normalized = (float(val) - min_val) / range_val
                            # 映射到feature_range
                            scaled = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
                            row[col] = round(scaled, 6)
                        except (TypeError, ValueError):
                            pass
            
            elif method == "z_score":
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                std_val = std_val if std_val > 0 else 1
                
                scaler_params[col] = {
                    "method": method,
                    "mean": mean_val,
                    "std": std_val
                }
                
                for row in result:
                    val = row.get(col)
                    if val is not None:
                        try:
                            normalized = (float(val) - mean_val) / std_val
                            row[col] = round(normalized, 6)
                        except (TypeError, ValueError):
                            pass
            
            elif method == "max_abs":
                max_abs = max(abs(v) for v in values)
                max_abs = max_abs if max_abs > 0 else 1
                
                scaler_params[col] = {
                    "method": method,
                    "max_abs": max_abs
                }
                
                for row in result:
                    val = row.get(col)
                    if val is not None:
                        try:
                            normalized = float(val) / max_abs
                            row[col] = round(normalized, 6)
                        except (TypeError, ValueError):
                            pass
        
        self._scalers.update(scaler_params)
        logger.info(f"Normalized columns: {columns}, method={method}")
        return result, scaler_params
    
    def encode_categorical(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        method: str = "label",
        handle_unknown: str = "ignore"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """编码分类变量
        
        Args:
            data: 数据列表
            columns: 目标列
            method: 编码方法
            handle_unknown: 未知值处理方式
            
        Returns:
            编码后的数据和编码映射
        """
        if not data or not columns:
            return data, {}
        
        result = copy.deepcopy(data)
        encoder_mappings: Dict[str, Any] = {}
        
        for col in columns:
            # 获取唯一值
            unique_values = set()
            for row in result:
                val = row.get(col)
                if val is not None:
                    unique_values.add(val)
            
            unique_list = sorted(list(unique_values), key=str)
            
            if method == "label":
                # 标签编码
                mapping = {v: i for i, v in enumerate(unique_list)}
                encoder_mappings[col] = {
                    "method": method,
                    "mapping": mapping,
                    "inverse_mapping": {i: v for v, i in mapping.items()}
                }
                
                for row in result:
                    val = row.get(col)
                    if val is not None:
                        row[col] = mapping.get(val, -1 if handle_unknown == "ignore" else None)
            
            elif method == "one_hot":
                # 独热编码
                encoder_mappings[col] = {
                    "method": method,
                    "categories": unique_list
                }
                
                for row in result:
                    val = row.get(col)
                    # 创建独热编码列
                    for cat in unique_list:
                        new_col = f"{col}_{cat}"
                        row[new_col] = 1 if val == cat else 0
                    # 删除原列
                    del row[col]
            
            elif method == "frequency":
                # 频率编码
                from collections import Counter
                value_counts = Counter(row.get(col) for row in result if row.get(col) is not None)
                total = sum(value_counts.values())
                freq_mapping = {v: count / total for v, count in value_counts.items()}
                
                encoder_mappings[col] = {
                    "method": method,
                    "mapping": freq_mapping
                }
                
                for row in result:
                    val = row.get(col)
                    if val is not None:
                        row[col] = round(freq_mapping.get(val, 0), 6)
        
        self._encoders.update(encoder_mappings)
        logger.info(f"Encoded columns: {columns}, method={method}")
        return result, encoder_mappings
    
    def tokenize_text(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        language: str = "zh",
        remove_stopwords: bool = True,
        lowercase: bool = True,
        min_token_length: int = 1
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """文本分词
        
        Args:
            data: 数据列表
            columns: 目标列
            language: 语言
            remove_stopwords: 是否去除停用词
            lowercase: 是否转小写
            min_token_length: 最小词长度
            
        Returns:
            分词后的数据和词汇统计
        """
        if not data or not columns:
            return data, {}
        
        result = copy.deepcopy(data)
        vocab_stats: Dict[str, Any] = {}
        
        # 简单的停用词列表
        chinese_stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        english_stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'although', 'this', 'that', 'these', 'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom'}
        
        stopwords = chinese_stopwords if language == "zh" else english_stopwords
        
        for col in columns:
            col_vocab: Dict[str, int] = {}
            
            for row in result:
                text = row.get(col)
                if text is None:
                    row[f"{col}_tokens"] = []
                    continue
                
                text = str(text)
                if lowercase:
                    text = text.lower()
                
                # 简单分词（实际应用中应该使用专业分词库）
                if language == "zh":
                    # 中文：按字符分词（简化版本）
                    tokens = list(text)
                else:
                    # 英文：按空格和标点分词
                    import re
                    tokens = re.findall(r'\b\w+\b', text)
                
                # 过滤
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stopwords]
                
                tokens = [t for t in tokens if len(t) >= min_token_length]
                
                # 更新词汇统计
                for token in tokens:
                    col_vocab[token] = col_vocab.get(token, 0) + 1
                
                row[f"{col}_tokens"] = tokens
            
            vocab_stats[col] = {
                "vocab_size": len(col_vocab),
                "total_tokens": sum(col_vocab.values()),
                "top_tokens": sorted(col_vocab.items(), key=lambda x: x[1], reverse=True)[:20]
            }
        
        logger.info(f"Tokenized columns: {columns}")
        return result, vocab_stats


# ============================================================================
# 特征工程处理器
# ============================================================================

class FeatureEngineeringProcessor:
    """特征工程处理器
    
    实现特征创建、选择、转换等操作。
    """
    
    def create_feature(
        self,
        data: List[Dict[str, Any]],
        feature_name: str,
        expression: str
    ) -> List[Dict[str, Any]]:
        """创建新特征
        
        Args:
            data: 数据列表
            feature_name: 新特征名称
            expression: 计算表达式
            
        Returns:
            添加新特征后的数据
        """
        if not data:
            return data
        
        result = copy.deepcopy(data)
        
        for row in result:
            try:
                # 安全的表达式求值
                # 注意：实际生产环境需要更严格的安全检查
                local_vars = {k: v for k, v in row.items() if isinstance(v, (int, float))}
                # 添加数学函数
                local_vars.update({
                    'abs': abs,
                    'round': round,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'len': len,
                    'sqrt': math.sqrt,
                    'log': math.log,
                    'log10': math.log10,
                    'exp': math.exp,
                    'pow': math.pow,
                })
                
                value = eval(expression, {"__builtins__": {}}, local_vars)
                row[feature_name] = value
            except Exception as e:
                logger.warning(f"Error creating feature {feature_name}: {e}")
                row[feature_name] = None
        
        logger.info(f"Created feature: {feature_name}")
        return result
    
    def select_features_by_variance(
        self,
        data: List[Dict[str, Any]],
        threshold: float = 0.0,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """基于方差选择特征
        
        Args:
            data: 数据列表
            threshold: 方差阈值
            exclude_columns: 排除的列
            
        Returns:
            选择后的数据、选中的特征、移除的特征
        """
        if not data:
            return data, [], []
        
        exclude_columns = exclude_columns or []
        selected_features = []
        removed_features = []
        
        # 计算各列方差
        all_columns = list(data[0].keys())
        
        for col in all_columns:
            if col in exclude_columns:
                selected_features.append(col)
                continue
            
            # 获取数值
            values = []
            for row in data:
                val = row.get(col)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        break
            else:
                if values:
                    mean_val = sum(values) / len(values)
                    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                    
                    if variance > threshold:
                        selected_features.append(col)
                    else:
                        removed_features.append(col)
                        continue
                else:
                    selected_features.append(col)
                continue
            
            # 非数值列保留
            selected_features.append(col)
        
        # 过滤数据
        result = []
        for row in data:
            new_row = {k: v for k, v in row.items() if k in selected_features}
            result.append(new_row)
        
        logger.info(f"Selected {len(selected_features)} features, removed {len(removed_features)}")
        return result, selected_features, removed_features
    
    def transform_feature(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        transform_type: str
    ) -> List[Dict[str, Any]]:
        """转换特征
        
        Args:
            data: 数据列表
            columns: 目标列
            transform_type: 转换类型
            
        Returns:
            转换后的数据
        """
        if not data or not columns:
            return data
        
        result = copy.deepcopy(data)
        
        for col in columns:
            for row in result:
                val = row.get(col)
                if val is None:
                    continue
                
                try:
                    num_val = float(val)
                    
                    if transform_type == "log":
                        row[col] = math.log(num_val + 1)  # +1避免log(0)
                    elif transform_type == "log10":
                        row[col] = math.log10(num_val + 1)
                    elif transform_type == "sqrt":
                        row[col] = math.sqrt(abs(num_val))
                    elif transform_type == "square":
                        row[col] = num_val ** 2
                    elif transform_type == "exp":
                        row[col] = math.exp(min(num_val, 700))  # 防止溢出
                    elif transform_type == "reciprocal":
                        row[col] = 1 / num_val if num_val != 0 else 0
                except (TypeError, ValueError, OverflowError) as e:
                    logger.warning(f"Error transforming {col}: {e}")
        
        logger.info(f"Transformed columns: {columns}, type={transform_type}")
        return result
    
    def reduce_dimensions(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        n_components: int,
        method: str = "pca"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """降维
        
        Args:
            data: 数据列表
            columns: 目标列
            n_components: 目标维度
            method: 降维方法
            
        Returns:
            降维后的数据和降维信息
        """
        if not data or not columns:
            return data, {}
        
        # 简化版PCA实现（实际应使用sklearn）
        # 这里只是演示逻辑，实际生产环境应该使用专业库
        
        # 提取数值矩阵
        matrix = []
        for row in data:
            row_values = []
            for col in columns:
                val = row.get(col)
                try:
                    row_values.append(float(val) if val is not None else 0.0)
                except (TypeError, ValueError):
                    row_values.append(0.0)
            matrix.append(row_values)
        
        # 简化：只保留前n_components列作为主成分
        result = copy.deepcopy(data)
        
        # 移除原列，添加新的主成分列
        for i, row in enumerate(result):
            for col in columns:
                if col in row:
                    del row[col]
            
            # 添加主成分（简化版本）
            for j in range(min(n_components, len(columns))):
                row[f"PC{j+1}"] = matrix[i][j] if j < len(matrix[i]) else 0.0
        
        reduction_info = {
            "method": method,
            "original_features": columns,
            "n_components": n_components,
            "new_features": [f"PC{i+1}" for i in range(n_components)]
        }
        
        logger.info(f"Reduced dimensions: {len(columns)} -> {n_components}")
        return result, reduction_info


# ============================================================================
# 数据增强处理器
# ============================================================================

class DataAugmentationProcessor:
    """数据增强处理器
    
    实现数据增强相关的操作。
    """
    
    def oversample(
        self,
        data: List[Dict[str, Any]],
        target_column: str,
        sampling_strategy: Union[str, Dict[str, int]] = "auto"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """过采样
        
        Args:
            data: 数据列表
            target_column: 目标列
            sampling_strategy: 采样策略
            
        Returns:
            过采样后的数据和各类别数量
        """
        if not data:
            return data, {}
        
        # 统计各类别数量
        from collections import Counter
        class_counts = Counter(row.get(target_column) for row in data)
        
        if not class_counts:
            return data, {}
        
        # 确定目标数量
        max_count = max(class_counts.values())
        
        if sampling_strategy == "auto" or sampling_strategy == "minority":
            target_counts = {cls: max_count for cls in class_counts}
        elif isinstance(sampling_strategy, dict):
            target_counts = sampling_strategy
        else:
            target_counts = {cls: max_count for cls in class_counts}
        
        # 按类别分组
        class_data: Dict[Any, List[Dict[str, Any]]] = {}
        for row in data:
            cls = row.get(target_column)
            if cls not in class_data:
                class_data[cls] = []
            class_data[cls].append(row)
        
        # 过采样
        result = list(data)  # 保留原数据
        
        for cls, samples in class_data.items():
            current_count = len(samples)
            target_count = target_counts.get(cls, current_count)
            
            if target_count > current_count:
                # 随机复制样本
                additional = target_count - current_count
                for _ in range(additional):
                    sample = random.choice(samples)
                    result.append(copy.deepcopy(sample))
        
        # 统计结果
        final_counts = Counter(row.get(target_column) for row in result)
        
        logger.info(f"Oversampled: {dict(class_counts)} -> {dict(final_counts)}")
        return result, dict(final_counts)
    
    def undersample(
        self,
        data: List[Dict[str, Any]],
        target_column: str,
        sampling_strategy: Union[str, Dict[str, int]] = "auto"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """欠采样
        
        Args:
            data: 数据列表
            target_column: 目标列
            sampling_strategy: 采样策略
            
        Returns:
            欠采样后的数据和各类别数量
        """
        if not data:
            return data, {}
        
        # 统计各类别数量
        from collections import Counter
        class_counts = Counter(row.get(target_column) for row in data)
        
        if not class_counts:
            return data, {}
        
        # 确定目标数量
        min_count = min(class_counts.values())
        
        if sampling_strategy == "auto" or sampling_strategy == "majority":
            target_counts = {cls: min_count for cls in class_counts}
        elif isinstance(sampling_strategy, dict):
            target_counts = sampling_strategy
        else:
            target_counts = {cls: min_count for cls in class_counts}
        
        # 按类别分组
        class_data: Dict[Any, List[Dict[str, Any]]] = {}
        for row in data:
            cls = row.get(target_column)
            if cls not in class_data:
                class_data[cls] = []
            class_data[cls].append(row)
        
        # 欠采样
        result = []
        
        for cls, samples in class_data.items():
            target_count = target_counts.get(cls, len(samples))
            if target_count < len(samples):
                # 随机选择样本
                selected = random.sample(samples, target_count)
                result.extend(selected)
            else:
                result.extend(samples)
        
        # 打乱顺序
        random.shuffle(result)
        
        # 统计结果
        final_counts = Counter(row.get(target_column) for row in result)
        
        logger.info(f"Undersampled: {dict(class_counts)} -> {dict(final_counts)}")
        return result, dict(final_counts)
    
    def augment_text(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        methods: List[str],
        augment_ratio: float = 0.3,
        num_augment: int = 1
    ) -> Tuple[List[Dict[str, Any]], int]:
        """文本增强
        
        Args:
            data: 数据列表
            columns: 目标列
            methods: 增强方法
            augment_ratio: 增强比例
            num_augment: 每条增强次数
            
        Returns:
            增强后的数据和生成的样本数
        """
        if not data or not columns:
            return data, 0
        
        result = list(data)
        generated = 0
        
        # 选择要增强的样本
        num_to_augment = int(len(data) * augment_ratio)
        samples_to_augment = random.sample(data, min(num_to_augment, len(data)))
        
        for sample in samples_to_augment:
            for _ in range(num_augment):
                new_sample = copy.deepcopy(sample)
                
                for col in columns:
                    text = new_sample.get(col)
                    if not text or not isinstance(text, str):
                        continue
                    
                    for method in methods:
                        text = self._apply_text_augmentation(text, method)
                    
                    new_sample[col] = text
                
                result.append(new_sample)
                generated += 1
        
        logger.info(f"Generated {generated} augmented text samples")
        return result, generated
    
    def _apply_text_augmentation(self, text: str, method: str) -> str:
        """应用文本增强方法
        
        Args:
            text: 原始文本
            method: 增强方法
            
        Returns:
            增强后的文本
        """
        if method == "random_deletion":
            # 随机删除
            words = list(text)
            if len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
            return ''.join(words)
        
        elif method == "random_swap":
            # 随机交换
            words = list(text)
            if len(words) > 1:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return ''.join(words)
        
        elif method == "random_insertion":
            # 随机插入（重复一个字符）
            words = list(text)
            if words:
                char = random.choice(words)
                idx = random.randint(0, len(words))
                words.insert(idx, char)
            return ''.join(words)
        
        return text


# ============================================================================
# 数据分割处理器
# ============================================================================

class DataSplitProcessor:
    """数据分割处理器
    
    实现数据集分割相关的操作。
    """
    
    def train_val_test_split(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_column: Optional[str] = None,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """训练/验证/测试分割
        
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify_column: 分层采样目标列
            shuffle: 是否打乱
            random_state: 随机种子
            
        Returns:
            分割结果
        """
        if not data:
            return {
                "train": [], "val": [], "test": [],
                "train_count": 0, "val_count": 0, "test_count": 0
            }
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise DataSplitRatioError(train_ratio, val_ratio, test_ratio)
        
        # 设置随机种子
        random.seed(random_state)
        
        if stratify_column:
            # 分层采样
            return self._stratified_split(
                data, train_ratio, val_ratio, test_ratio, stratify_column, shuffle
            )
        
        # 普通分割
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
            "total_count": n
        }
    
    def _stratified_split(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify_column: str,
        shuffle: bool
    ) -> Dict[str, Any]:
        """分层分割
        
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify_column: 分层列
            shuffle: 是否打乱
            
        Returns:
            分割结果
        """
        # 按类别分组
        class_data: Dict[Any, List[Dict[str, Any]]] = {}
        for row in data:
            cls = row.get(stratify_column)
            if cls not in class_data:
                class_data[cls] = []
            class_data[cls].append(row)
        
        train_data = []
        val_data = []
        test_data = []
        
        for cls, samples in class_data.items():
            if shuffle:
                random.shuffle(samples)
            
            n = len(samples)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_data.extend(samples[:train_end])
            val_data.extend(samples[train_end:val_end])
            test_data.extend(samples[val_end:])
        
        # 打乱各集合
        if shuffle:
            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)
        
        # 统计各类别分布
        from collections import Counter
        train_dist = Counter(row.get(stratify_column) for row in train_data)
        val_dist = Counter(row.get(stratify_column) for row in val_data)
        test_dist = Counter(row.get(stratify_column) for row in test_data)
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
            "total_count": len(data),
            "train_distribution": dict(train_dist),
            "val_distribution": dict(val_dist),
            "test_distribution": dict(test_dist)
        }
    
    def k_fold_split(
        self,
        data: List[Dict[str, Any]],
        n_folds: int = 5,
        stratify_column: Optional[str] = None,
        shuffle: bool = True,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """K折交叉验证分割
        
        Args:
            data: 数据列表
            n_folds: 折数
            stratify_column: 分层列
            shuffle: 是否打乱
            random_state: 随机种子
            
        Returns:
            各折信息列表
        """
        if not data or n_folds < 2:
            return []
        
        random.seed(random_state)
        
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        
        n = len(data)
        fold_size = n // n_folds
        
        folds = []
        
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else n
            
            val_indices = set(range(start, end))
            train_data = [data[j] for j in range(n) if j not in val_indices]
            val_data = [data[j] for j in val_indices]
            
            folds.append({
                "fold": i + 1,
                "train": train_data,
                "val": val_data,
                "train_count": len(train_data),
                "val_count": len(val_data)
            })
        
        logger.info(f"Created {n_folds} folds")
        return folds


# ============================================================================
# 预处理引擎
# ============================================================================

class DataPreprocessingEngine:
    """数据预处理引擎
    
    协调各种预处理操作的执行。
    """
    
    def __init__(self):
        """初始化预处理引擎"""
        self.cleaning_processor = DataCleaningProcessor()
        self.transform_processor = DataTransformProcessor()
        self.feature_processor = FeatureEngineeringProcessor()
        self.augmentation_processor = DataAugmentationProcessor()
        self.split_processor = DataSplitProcessor()
    
    def execute_pipeline(
        self,
        data: List[Dict[str, Any]],
        operations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[OperationResult]]:
        """执行预处理流水线
        
        Args:
            data: 原始数据
            operations: 操作列表
            
        Returns:
            处理后的数据和操作结果列表
        """
        results = []
        current_data = data
        
        # 按顺序排序操作
        sorted_ops = sorted(operations, key=lambda x: x.get("order", 0))
        
        for i, op in enumerate(sorted_ops):
            if not op.get("enabled", True):
                continue
            
            op_type = op.get("operation_type")
            config = op.get("config", {})
            
            started_at = datetime.utcnow()
            result = OperationResult(
                operation_type=op_type,
                status="processing",
                started_at=started_at
            )
            
            try:
                current_data, affected = self._execute_operation(
                    current_data, op_type, config
                )
                
                result.status = "completed"
                result.rows_affected = affected.get("rows_affected", 0)
                result.columns_affected = affected.get("columns_affected", [])
                result.details = affected
                
            except Exception as e:
                logger.error(f"Operation {op_type} failed: {e}")
                result.status = "failed"
                result.error_message = str(e)
            
            result.completed_at = datetime.utcnow()
            result.duration_ms = int(
                (result.completed_at - started_at).total_seconds() * 1000
            )
            
            results.append(result)
            
            if result.status == "failed":
                break
        
        return current_data, results
    
    def _execute_operation(
        self,
        data: List[Dict[str, Any]],
        op_type: str,
        config: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """执行单个操作
        
        Args:
            data: 数据
            op_type: 操作类型
            config: 操作配置
            
        Returns:
            处理后的数据和影响信息
        """
        affected = {"rows_affected": 0, "columns_affected": []}
        
        if op_type == "remove_duplicates":
            data, count = self.cleaning_processor.remove_duplicates(
                data,
                subset=config.get("subset"),
                keep=config.get("keep", "first")
            )
            affected["rows_affected"] = count
        
        elif op_type == "handle_missing":
            data, counts = self.cleaning_processor.handle_missing_values(
                data,
                strategy=config.get("strategy", "drop"),
                columns=config.get("columns"),
                fill_value=config.get("fill_value"),
                threshold=config.get("threshold", 0.5)
            )
            affected["rows_affected"] = sum(counts.values())
            affected["columns_affected"] = list(counts.keys())
        
        elif op_type == "remove_outliers":
            data, counts = self.cleaning_processor.handle_outliers(
                data,
                method=config.get("method", "iqr"),
                columns=config.get("columns"),
                threshold=config.get("threshold", 1.5),
                action=config.get("action", "remove")
            )
            affected["rows_affected"] = sum(counts.values())
            affected["columns_affected"] = list(counts.keys())
        
        elif op_type == "filter_rows":
            data, count = self.cleaning_processor.filter_rows(
                data,
                conditions=config.get("conditions", []),
                logic=config.get("logic", "and")
            )
            affected["rows_affected"] = count
        
        elif op_type == "normalize":
            data, params = self.transform_processor.normalize(
                data,
                columns=config.get("columns", []),
                method=config.get("method", "min_max"),
                feature_range=config.get("feature_range", (0, 1))
            )
            affected["columns_affected"] = list(params.keys())
        
        elif op_type == "encode_categorical":
            data, mappings = self.transform_processor.encode_categorical(
                data,
                columns=config.get("columns", []),
                method=config.get("method", "label"),
                handle_unknown=config.get("handle_unknown", "ignore")
            )
            affected["columns_affected"] = list(mappings.keys())
        
        elif op_type == "tokenize":
            data, stats = self.transform_processor.tokenize_text(
                data,
                columns=config.get("columns", []),
                language=config.get("language", "zh"),
                remove_stopwords=config.get("remove_stopwords", True),
                lowercase=config.get("lowercase", True),
                min_token_length=config.get("min_token_length", 1)
            )
            affected["columns_affected"] = config.get("columns", [])
        
        elif op_type == "create_feature":
            data = self.feature_processor.create_feature(
                data,
                feature_name=config.get("name"),
                expression=config.get("expression")
            )
            affected["columns_affected"] = [config.get("name")]
        
        elif op_type == "select_feature":
            data, selected, removed = self.feature_processor.select_features_by_variance(
                data,
                threshold=config.get("threshold", 0.0),
                exclude_columns=config.get("exclude_columns")
            )
            affected["columns_affected"] = removed
            affected["details"] = {"selected": selected, "removed": removed}
        
        elif op_type == "transform_feature":
            data = self.feature_processor.transform_feature(
                data,
                columns=config.get("columns", []),
                transform_type=config.get("transform_type", "log")
            )
            affected["columns_affected"] = config.get("columns", [])
        
        elif op_type == "reduce_dimension":
            data, info = self.feature_processor.reduce_dimensions(
                data,
                columns=config.get("columns", []),
                n_components=config.get("n_components", 2),
                method=config.get("method", "pca")
            )
            affected["details"] = info
        
        elif op_type == "oversample":
            data, counts = self.augmentation_processor.oversample(
                data,
                target_column=config.get("target_column"),
                sampling_strategy=config.get("sampling_strategy", "auto")
            )
            affected["rows_affected"] = len(data) - sum(counts.values())
        
        elif op_type == "undersample":
            data, counts = self.augmentation_processor.undersample(
                data,
                target_column=config.get("target_column"),
                sampling_strategy=config.get("sampling_strategy", "auto")
            )
            affected["rows_affected"] = sum(counts.values())
        
        elif op_type == "augment_text":
            data, generated = self.augmentation_processor.augment_text(
                data,
                columns=config.get("columns", []),
                methods=config.get("methods", ["random_deletion"]),
                augment_ratio=config.get("augment_ratio", 0.3),
                num_augment=config.get("num_augment", 1)
            )
            affected["rows_affected"] = generated
        
        return data, affected
