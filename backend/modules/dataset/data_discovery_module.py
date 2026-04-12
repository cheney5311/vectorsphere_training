"""数据发现核心模块

实现数据发现的底层核心逻辑，包括数据源连接、扫描、格式检测、模式推断等功能。
"""

import json
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Generator

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    DataSourceConnectionError,
    UnsupportedDataFormatError,
    UnsupportedDataSourceError,
    SchemaInferenceError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类型推断器
# ============================================================================

class DataTypeInferrer:
    """数据类型推断器
    
    根据采样数据推断列的数据类型。
    """
    
    # 支持的数据类型
    SUPPORTED_TYPES = [
        'integer', 'float', 'boolean', 'string', 'datetime', 
        'date', 'time', 'json', 'array', 'binary', 'null'
    ]
    
    @classmethod
    def infer_type(cls, values: List[Any]) -> Dict[str, Any]:
        """推断值列表的数据类型
        
        Args:
            values: 值列表
            
        Returns:
            包含推断类型和置信度的字典
        """
        if not values:
            return {'type': 'null', 'confidence': 1.0, 'nullable': True}
        
        # 统计各类型的匹配数量
        type_counts = {t: 0 for t in cls.SUPPORTED_TYPES}
        null_count = 0
        total = len(values)
        
        for value in values:
            if value is None or (isinstance(value, str) and value.strip() == ''):
                null_count += 1
                type_counts['null'] += 1
                continue
            
            detected = cls._detect_single_value_type(value)
            type_counts[detected] += 1
        
        # 排除null找出最可能的类型
        non_null_counts = {k: v for k, v in type_counts.items() if k != 'null'}
        if sum(non_null_counts.values()) == 0:
            return {'type': 'null', 'confidence': 1.0, 'nullable': True}
        
        inferred_type = max(non_null_counts, key=non_null_counts.get)
        confidence = non_null_counts[inferred_type] / max(1, total - null_count)
        
        return {
            'type': inferred_type,
            'confidence': round(confidence, 4),
            'nullable': null_count > 0,
            'null_percentage': round(null_count / total * 100, 2) if total > 0 else 0
        }
    
    @classmethod
    def _detect_single_value_type(cls, value: Any) -> str:
        """检测单个值的类型"""
        if value is None:
            return 'null'
        
        if isinstance(value, bool):
            return 'boolean'
        
        if isinstance(value, int):
            return 'integer'
        
        if isinstance(value, float):
            return 'float'
        
        if isinstance(value, (list, tuple)):
            return 'array'
        
        if isinstance(value, dict):
            return 'json'
        
        if isinstance(value, bytes):
            return 'binary'
        
        if isinstance(value, str):
            return cls._infer_string_type(value)
        
        return 'string'
    
    @classmethod
    def _infer_string_type(cls, value: str) -> str:
        """推断字符串值的实际类型"""
        value = value.strip()
        
        # 检查是否为布尔值
        if value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
            return 'boolean'
        
        # 检查是否为整数
        try:
            int(value)
            return 'integer'
        except ValueError:
            pass
        
        # 检查是否为浮点数
        try:
            float(value)
            return 'float'
        except ValueError:
            pass
        
        # 检查是否为日期时间
        if cls._is_datetime(value):
            return 'datetime'
        
        if cls._is_date(value):
            return 'date'
        
        if cls._is_time(value):
            return 'time'
        
        # 检查是否为JSON
        if cls._is_json(value):
            return 'json'
        
        return 'string'
    
    @classmethod
    def _is_datetime(cls, value: str) -> bool:
        """检查是否为日期时间格式"""
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
        ]
        for fmt in datetime_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False
    
    @classmethod
    def _is_date(cls, value: str) -> bool:
        """检查是否为日期格式"""
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
        ]
        for fmt in date_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False
    
    @classmethod
    def _is_time(cls, value: str) -> bool:
        """检查是否为时间格式"""
        time_formats = ['%H:%M:%S', '%H:%M', '%H:%M:%S.%f']
        for fmt in time_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False
    
    @classmethod
    def _is_json(cls, value: str) -> bool:
        """检查是否为JSON格式"""
        if not (value.startswith('{') or value.startswith('[')):
            return False
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


# ============================================================================
# 列统计信息计算器
# ============================================================================

class ColumnStatisticsCalculator:
    """列统计信息计算器
    
    计算数据列的统计信息，如最小值、最大值、均值、标准差等。
    """
    
    @classmethod
    def calculate(cls, values: List[Any], data_type: str) -> Dict[str, Any]:
        """计算列的统计信息
        
        Args:
            values: 值列表
            data_type: 数据类型
            
        Returns:
            统计信息字典
        """
        if not values:
            return {}
        
        # 过滤掉None值
        non_null_values = [v for v in values if v is not None]
        
        stats = {
            'total_count': len(values),
            'null_count': len(values) - len(non_null_values),
            'non_null_count': len(non_null_values),
        }
        
        if not non_null_values:
            return stats
        
        # 根据数据类型计算不同的统计信息
        if data_type in ('integer', 'float'):
            stats.update(cls._calculate_numeric_stats(non_null_values))
        elif data_type == 'string':
            stats.update(cls._calculate_string_stats(non_null_values))
        elif data_type in ('datetime', 'date'):
            stats.update(cls._calculate_datetime_stats(non_null_values, data_type))
        elif data_type == 'boolean':
            stats.update(cls._calculate_boolean_stats(non_null_values))
        
        # 计算唯一值数量
        try:
            unique_values = set(str(v) for v in non_null_values)
            stats['unique_count'] = len(unique_values)
            stats['is_unique'] = len(unique_values) == len(non_null_values)
        except (TypeError, ValueError):
            pass
        
        return stats
    
    @classmethod
    def _calculate_numeric_stats(cls, values: List) -> Dict[str, Any]:
        """计算数值类型的统计信息"""
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if not numeric_values:
                return {}
            
            n = len(numeric_values)
            mean_val = sum(numeric_values) / n
            
            # 计算标准差
            variance = sum((x - mean_val) ** 2 for x in numeric_values) / n
            std_val = variance ** 0.5
            
            sorted_values = sorted(numeric_values)
            
            return {
                'min_value': min(numeric_values),
                'max_value': max(numeric_values),
                'mean_value': round(mean_val, 4),
                'std_value': round(std_val, 4),
                'median_value': sorted_values[n // 2] if n > 0 else None,
                'sum_value': sum(numeric_values),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to calculate numeric stats: {e}")
            return {}
    
    @classmethod
    def _calculate_string_stats(cls, values: List[str]) -> Dict[str, Any]:
        """计算字符串类型的统计信息"""
        try:
            lengths = [len(str(v)) for v in values]
            return {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_length': round(sum(lengths) / len(lengths), 2),
                'total_length': sum(lengths),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to calculate string stats: {e}")
            return {}
    
    @classmethod
    def _calculate_datetime_stats(cls, values: List, data_type: str) -> Dict[str, Any]:
        """计算日期时间类型的统计信息"""
        try:
            # 尝试将字符串转换为datetime对象
            datetime_values = []
            for v in values:
                if isinstance(v, datetime):
                    datetime_values.append(v)
                elif isinstance(v, str):
                    # 尝试多种格式解析
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                        try:
                            datetime_values.append(datetime.strptime(v, fmt))
                            break
                        except ValueError:
                            continue
            
            if not datetime_values:
                return {}
            
            return {
                'min_datetime': min(datetime_values).isoformat(),
                'max_datetime': max(datetime_values).isoformat(),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to calculate datetime stats: {e}")
            return {}
    
    @classmethod
    def _calculate_boolean_stats(cls, values: List) -> Dict[str, Any]:
        """计算布尔类型的统计信息"""
        try:
            true_count = sum(1 for v in values if str(v).lower() in ('true', 'yes', '1'))
            false_count = len(values) - true_count
            return {
                'true_count': true_count,
                'false_count': false_count,
                'true_percentage': round(true_count / len(values) * 100, 2) if values else 0,
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to calculate boolean stats: {e}")
            return {}


# ============================================================================
# 数据格式检测器
# ============================================================================

class DataFormatDetector:
    """数据格式检测器
    
    根据文件扩展名、MIME类型和内容检测数据格式。
    """
    
    # 文件扩展名到数据格式的映射
    EXTENSION_MAP = {
        '.csv': 'csv',
        '.tsv': 'csv',
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.ndjson': 'jsonl',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.avro': 'avro',
        '.xml': 'xml',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.txt': 'text',
        '.sql': 'sql',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.gif': 'image',
        '.bmp': 'image',
        '.mp3': 'audio',
        '.wav': 'audio',
        '.flac': 'audio',
        '.mp4': 'video',
        '.avi': 'video',
        '.mkv': 'video',
    }
    
    # MIME类型到数据格式的映射
    MIME_MAP = {
        'text/csv': 'csv',
        'application/json': 'json',
        'application/x-parquet': 'parquet',
        'application/avro': 'avro',
        'application/xml': 'xml',
        'text/xml': 'xml',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
        'application/vnd.ms-excel': 'excel',
        'text/plain': 'text',
        'image/jpeg': 'image',
        'image/png': 'image',
        'audio/mpeg': 'audio',
        'video/mp4': 'video',
    }
    
    @classmethod
    def detect_from_path(cls, file_path: str) -> Optional[str]:
        """根据文件路径检测数据格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据格式字符串，无法识别则返回None
        """
        ext = Path(file_path).suffix.lower()
        return cls.EXTENSION_MAP.get(ext)
    
    @classmethod
    def detect_from_content(cls, content: bytes, sample_size: int = 8192) -> Optional[str]:
        """根据文件内容检测数据格式
        
        Args:
            content: 文件内容（字节）
            sample_size: 采样大小
            
        Returns:
            数据格式字符串
        """
        sample = content[:sample_size]
        
        # 尝试检测JSON
        try:
            text = sample.decode('utf-8').strip()
            if text.startswith('{') or text.startswith('['):
                json.loads(text if len(text) < 1000 else text[:1000] + '...')
                return 'json'
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # 尝试检测JSONL
        try:
            text = sample.decode('utf-8').strip()
            lines = text.split('\n')
            if len(lines) > 1:
                # 检查每行是否都是有效的JSON
                valid_count = sum(1 for line in lines[:10] if cls._is_valid_json_line(line))
                if valid_count >= len(lines[:10]) * 0.8:
                    return 'jsonl'
        except UnicodeDecodeError:
            pass
        
        # 检测CSV
        try:
            text = sample.decode('utf-8')
            lines = text.split('\n')
            if len(lines) > 1:
                # 检查是否有一致的分隔符
                for delimiter in [',', '\t', ';', '|']:
                    counts = [line.count(delimiter) for line in lines[:5] if line.strip()]
                    if counts and all(c == counts[0] and c > 0 for c in counts):
                        return 'csv'
        except UnicodeDecodeError:
            pass
        
        # 检测XML
        try:
            text = sample.decode('utf-8').strip()
            if text.startswith('<?xml') or (text.startswith('<') and '>' in text):
                return 'xml'
        except UnicodeDecodeError:
            pass
        
        # 检测Parquet（魔数）
        if sample[:4] == b'PAR1':
            return 'parquet'
        
        # 检测Avro（魔数）
        if sample[:4] == b'Obj\x01':
            return 'avro'
        
        return None
    
    @classmethod
    def _is_valid_json_line(cls, line: str) -> bool:
        """检查是否为有效的JSON行"""
        line = line.strip()
        if not line:
            return True
        try:
            json.loads(line)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


# ============================================================================
# 数据源连接器基类
# ============================================================================

@dataclass
class ScanResult:
    """扫描结果数据类"""
    source_id: str
    source_type: str
    location: str
    status: str
    files_found: int = 0
    tables_found: int = 0
    total_size_bytes: int = 0
    scanned_at: datetime = field(default_factory=datetime.utcnow)
    scan_duration_ms: int = 0
    error_message: Optional[str] = None
    discovered_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiscoveredItem:
    """发现的数据项"""
    item_id: str
    name: str
    path: str
    format: str
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_info: Optional[List[Dict[str, Any]]] = None
    preview_data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSourceConnector(ABC):
    """数据源连接器基类
    
    定义数据源连接、扫描和数据读取的抽象接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化连接器
        
        Args:
            config: 数据源配置
        """
        self.config = config
        self.location = config.get('location', '')
        self.credentials = config.get('credentials', {})
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """建立与数据源的连接
        
        Returns:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开与数据源的连接"""
        pass
    
    @abstractmethod
    def scan(self) -> ScanResult:
        """扫描数据源，发现可用数据
        
        Returns:
            扫描结果
        """
        pass
    
    @abstractmethod
    def read_sample(self, item_path: str, sample_size: int = 100) -> List[Dict[str, Any]]:
        """读取数据样本
        
        Args:
            item_path: 数据项路径
            sample_size: 采样大小
            
        Returns:
            样本数据列表
        """
        pass
    
    @abstractmethod
    def get_schema(self, item_path: str) -> List[Dict[str, Any]]:
        """获取数据模式
        
        Args:
            item_path: 数据项路径
            
        Returns:
            模式信息列表
        """
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ============================================================================
# 文件系统连接器
# ============================================================================

class FileSystemConnector(DataSourceConnector):
    """文件系统数据源连接器
    
    支持本地文件系统的数据发现和读取。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.include_patterns = config.get('include_patterns', ['*'])
        self.exclude_patterns = config.get('exclude_patterns', [])
        self.recursive = config.get('recursive', True)
        self.max_depth = config.get('max_depth', 10)
        self.max_files = config.get('max_files', 1000)
    
    def connect(self) -> bool:
        """连接到文件系统"""
        if not os.path.exists(self.location):
            raise DataSourceConnectionError(
                'file_system', self.location,
                f"Path does not exist: {self.location}"
            )
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
    
    def scan(self) -> ScanResult:
        """扫描文件系统目录"""
        start_time = datetime.utcnow()
        discovered_items = []
        total_size = 0
        files_found = 0
        
        try:
            for item in self._walk_directory(self.location, depth=0):
                if files_found >= self.max_files:
                    break
                
                file_path = item['path']
                file_size = item['size']
                
                # 检测数据格式
                data_format = DataFormatDetector.detect_from_path(file_path)
                if not data_format:
                    # 尝试根据内容检测
                    try:
                        with open(file_path, 'rb') as f:
                            data_format = DataFormatDetector.detect_from_content(f.read(8192))
                    except (IOError, OSError):
                        continue
                
                if data_format:
                    discovered_items.append({
                        'item_id': str(uuid.uuid4()),
                        'name': os.path.basename(file_path),
                        'path': file_path,
                        'format': data_format,
                        'size_bytes': file_size,
                        'modified_at': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    })
                    total_size += file_size
                    files_found += 1
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ScanResult(
                source_id=str(uuid.uuid4()),
                source_type='file_system',
                location=self.location,
                status='discovered',
                files_found=files_found,
                total_size_bytes=total_size,
                scanned_at=start_time,
                scan_duration_ms=duration_ms,
                discovered_items=discovered_items
            )
        
        except Exception as e:
            logger.error(f"Error scanning file system: {e}")
            return ScanResult(
                source_id=str(uuid.uuid4()),
                source_type='file_system',
                location=self.location,
                status='failed',
                error_message=str(e),
                scanned_at=start_time,
            )
    
    def _walk_directory(self, path: str, depth: int) -> Generator[Dict[str, Any], None, None]:
        """递归遍历目录"""
        if depth > self.max_depth:
            return
        
        try:
            entries = os.listdir(path)
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot access directory {path}: {e}")
            return
        
        for entry in entries:
            full_path = os.path.join(path, entry)
            
            # 检查是否应该排除
            if self._should_exclude(entry):
                continue
            
            if os.path.isfile(full_path):
                if self._should_include(entry):
                    try:
                        stat_info = os.stat(full_path)
                        yield {
                            'path': full_path,
                            'size': stat_info.st_size,
                            'name': entry,
                        }
                    except (OSError, IOError):
                        continue
            
            elif os.path.isdir(full_path) and self.recursive:
                yield from self._walk_directory(full_path, depth + 1)
    
    def _should_include(self, filename: str) -> bool:
        """检查文件是否应该包含"""
        if not self.include_patterns or '*' in self.include_patterns:
            return True
        
        import fnmatch
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.include_patterns)
    
    def _should_exclude(self, filename: str) -> bool:
        """检查文件是否应该排除"""
        if not self.exclude_patterns:
            return False
        
        import fnmatch
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.exclude_patterns)
    
    def read_sample(self, item_path: str, sample_size: int = 100) -> List[Dict[str, Any]]:
        """读取文件样本数据"""
        data_format = DataFormatDetector.detect_from_path(item_path)
        
        if data_format == 'csv':
            return self._read_csv_sample(item_path, sample_size)
        elif data_format == 'json':
            return self._read_json_sample(item_path, sample_size)
        elif data_format == 'jsonl':
            return self._read_jsonl_sample(item_path, sample_size)
        else:
            raise UnsupportedDataFormatError(data_format or 'unknown')
    
    def _read_csv_sample(self, file_path: str, sample_size: int) -> List[Dict[str, Any]]:
        """读取CSV文件样本"""
        import csv
        
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # 尝试检测分隔符
                sample = f.read(8192)
                f.seek(0)
                
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
                
                for i, row in enumerate(reader):
                    if i >= sample_size:
                        break
                    samples.append(dict(row))
        except Exception as e:
            logger.warning(f"Error reading CSV file {file_path}: {e}")
            # 尝试使用默认设置
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= sample_size:
                            break
                        samples.append(dict(row))
            except Exception as e2:
                logger.error(f"Failed to read CSV file {file_path}: {e2}")
        
        return samples
    
    def _read_json_sample(self, file_path: str, sample_size: int) -> List[Dict[str, Any]]:
        """读取JSON文件样本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    return data[:sample_size]
                elif isinstance(data, dict):
                    return [data]
                else:
                    return [{'value': data}]
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []
    
    def _read_jsonl_sample(self, file_path: str, sample_size: int) -> List[Dict[str, Any]]:
        """读取JSONL文件样本"""
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {e}")
        
        return samples
    
    def get_schema(self, item_path: str) -> List[Dict[str, Any]]:
        """获取文件数据模式"""
        # 读取样本数据
        samples = self.read_sample(item_path, sample_size=1000)
        
        if not samples:
            return []
        
        # 收集所有列名
        all_columns = set()
        for sample in samples:
            all_columns.update(sample.keys())
        
        schema = []
        for column in sorted(all_columns):
            # 收集该列的所有值
            values = [sample.get(column) for sample in samples]
            
            # 推断类型
            type_info = DataTypeInferrer.infer_type(values)
            
            # 计算统计信息
            stats = ColumnStatisticsCalculator.calculate(values, type_info['type'])
            
            schema.append({
                'name': column,
                'data_type': type_info['type'],
                'nullable': type_info.get('nullable', True),
                'confidence': type_info.get('confidence', 0),
                **stats,
                'sample_values': [v for v in values[:5] if v is not None],
            })
        
        return schema


# ============================================================================
# 数据源连接器工厂
# ============================================================================

class DataSourceConnectorFactory:
    """数据源连接器工厂
    
    根据数据源类型创建对应的连接器实例。
    """
    
    # 注册的连接器类
    _connectors: Dict[str, Type[DataSourceConnector]] = {
        'file_system': FileSystemConnector,
    }
    
    @classmethod
    def register(cls, source_type: str, connector_class: Type[DataSourceConnector]) -> None:
        """注册新的连接器类
        
        Args:
            source_type: 数据源类型
            connector_class: 连接器类
        """
        cls._connectors[source_type] = connector_class
    
    @classmethod
    def create(cls, source_type: str, config: Dict[str, Any]) -> DataSourceConnector:
        """创建连接器实例
        
        Args:
            source_type: 数据源类型
            config: 数据源配置
            
        Returns:
            连接器实例
            
        Raises:
            UnsupportedDataSourceError: 不支持的数据源类型
        """
        connector_class = cls._connectors.get(source_type)
        if not connector_class:
            raise UnsupportedDataSourceError(
                source_type,
                supported_types=list(cls._connectors.keys())
            )
        
        return connector_class(config)
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """获取支持的数据源类型列表"""
        return list(cls._connectors.keys())


# ============================================================================
# 数据发现引擎
# ============================================================================

class DataDiscoveryEngine:
    """数据发现引擎
    
    协调数据源扫描、格式检测、模式推断等操作。
    """
    
    def __init__(self):
        """初始化数据发现引擎"""
        self.type_inferrer = DataTypeInferrer()
        self.stats_calculator = ColumnStatisticsCalculator()
        self.format_detector = DataFormatDetector()
        self.connector_factory = DataSourceConnectorFactory()
    
    def scan_source(self, source_config: Dict[str, Any]) -> ScanResult:
        """扫描单个数据源
        
        Args:
            source_config: 数据源配置
            
        Returns:
            扫描结果
        """
        source_type = source_config.get('source_type', 'file_system')
        
        try:
            connector = self.connector_factory.create(source_type, source_config)
            with connector:
                return connector.scan()
        except Exception as e:
            logger.error(f"Error scanning source: {e}")
            return ScanResult(
                source_id=str(uuid.uuid4()),
                source_type=source_type,
                location=source_config.get('location', ''),
                status='failed',
                error_message=str(e),
                scanned_at=datetime.utcnow(),
            )
    
    def scan_sources(self, source_configs: List[Dict[str, Any]], parallel: bool = False) -> List[ScanResult]:
        """批量扫描数据源
        
        Args:
            source_configs: 数据源配置列表
            parallel: 是否并行扫描
            
        Returns:
            扫描结果列表
        """
        results = []
        
        if parallel and len(source_configs) > 1:
            # 并行扫描（使用线程池）
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=min(len(source_configs), 5)) as executor:
                futures = {
                    executor.submit(self.scan_source, config): config
                    for config in source_configs
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        config = futures[future]
                        results.append(ScanResult(
                            source_id=str(uuid.uuid4()),
                            source_type=config.get('source_type', 'unknown'),
                            location=config.get('location', ''),
                            status='failed',
                            error_message=str(e),
                            scanned_at=datetime.utcnow(),
                        ))
        else:
            # 顺序扫描
            for config in source_configs:
                results.append(self.scan_source(config))
        
        return results
    
    def infer_schema(
        self,
        connector: DataSourceConnector,
        item_path: str,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """推断数据项的模式
        
        Args:
            connector: 数据源连接器
            item_path: 数据项路径
            sample_size: 采样大小
            
        Returns:
            模式信息字典
        """
        try:
            schema = connector.get_schema(item_path)
            
            return {
                'inferred_at': datetime.utcnow().isoformat(),
                'columns': schema,
                'column_count': len(schema),
                'sample_size_used': sample_size,
                'confidence_score': sum(c.get('confidence', 0) for c in schema) / max(len(schema), 1) * 100,
            }
        except Exception as e:
            raise SchemaInferenceError(item_path, str(e))
    
    def calculate_data_quality(self, schema: List[Dict[str, Any]], row_count: int) -> Dict[str, Any]:
        """计算数据质量评分
        
        Args:
            schema: 模式信息
            row_count: 行数
            
        Returns:
            数据质量信息
        """
        if not schema:
            return {'quality_score': 0, 'completeness': 0}
        
        # 计算完整性（基于非空率）
        null_rates = []
        for column in schema:
            total = column.get('total_count', row_count)
            null_count = column.get('null_count', 0)
            if total > 0:
                null_rates.append(1 - null_count / total)
        
        completeness = sum(null_rates) / len(null_rates) * 100 if null_rates else 0
        
        # 计算总体质量评分（简化版本）
        quality_score = completeness * 0.6  # 完整性占60%
        
        # 类型推断置信度占40%
        confidences = [c.get('confidence', 0) for c in schema]
        if confidences:
            quality_score += (sum(confidences) / len(confidences)) * 40
        
        return {
            'quality_score': round(quality_score, 2),
            'completeness': round(completeness, 2),
            'column_count': len(schema),
            'columns_with_nulls': sum(1 for c in schema if c.get('null_count', 0) > 0),
        }
