"""数据加载器

提供各种数据格式的加载和处理功能。
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


def load_json_data(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """加载JSON数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的数据
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON格式错误: {e}", e.doc, e.pos)


def load_csv_data(file_path: str, delimiter: str = ',') -> List[Dict[str, Any]]:
    """加载CSV数据
    
    Args:
        file_path: CSV文件路径
        delimiter: 分隔符
        
    Returns:
        解析后的数据列表
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")


def load_text_data(file_path: str) -> str:
    """加载文本数据
    
    Args:
        file_path: 文本文件路径
        
    Returns:
        文件内容字符串
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文本文件不存在: {file_path}")


def load_data(file_path: str) -> Union[Dict[str, Any], List[Any], str]:
    """根据文件扩展名自动加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        解析后的数据
        
    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if path.suffix.lower() == '.json':
        return load_json_data(file_path)
    elif path.suffix.lower() == '.csv':
        return load_csv_data(file_path)
    elif path.suffix.lower() in ['.txt', '.md', '.log']:
        return load_text_data(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def save_json_data(data: Union[Dict[str, Any], List[Any]], file_path: str, indent: int = 2) -> None:
    """保存数据为JSON格式
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
        indent: 缩进空格数
        
    Raises:
        IOError: 文件写入失败
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"保存JSON文件失败: {e}")


def save_csv_data(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> None:
    """保存数据为CSV格式
    
    Args:
        data: 要保存的数据列表
        file_path: 保存路径
        fieldnames: 字段名列表（可选）
        
    Raises:
        IOError: 文件写入失败
    """
    try:
        if not data:
            raise ValueError("数据为空")
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except IOError as e:
        raise IOError(f"保存CSV文件失败: {e}")


def convert_to_dataframe(data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """将数据转换为DataFrame
    
    Args:
        data: 输入数据
        
    Returns:
        DataFrame对象
    """
    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"数据转换为DataFrame失败: {e}")


def validate_data_schema(data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
    """验证数据是否包含必需字段
    
    Args:
        data: 数据列表
        required_fields: 必需字段列表
        
    Returns:
        验证结果
    """
    if not data:
        return False
    
    for field in required_fields:
        if field not in data[0]:
            return False
    
    return True