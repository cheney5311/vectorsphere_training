"""数据验证工具

提供通用的数据验证功能。
"""

from flask import request
from typing import Dict, Any, List, Optional
import json
import jsonschema
from jsonschema import validate, ValidationError


def validate_json(request_obj) -> Dict[str, Any]:
    """
    验证并解析JSON请求数据
    
    Args:
        request_obj: Flask请求对象
        
    Returns:
        解析后的JSON数据
        
    Raises:
        ValueError: JSON格式错误
    """
    try:
        if not request_obj.is_json:
            raise ValueError("请求必须是JSON格式")
        
        data = request_obj.get_json()
        if data is None:
            raise ValueError("JSON数据为空")
        
        return data
    except Exception as e:
        raise ValueError(f"JSON解析失败: {str(e)}")


def validate_agent_config(config_data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    验证智能体配置数据
    
    Args:
        config_data: 配置数据
        schema: 验证模式（可选）
        
    Returns:
        验证是否通过
        
    Raises:
        ValidationError: 验证失败
    """
    if schema is None:
        # 默认智能体配置模式
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "parameters": {"type": "object"},
                "prompts": {"type": "object"}
            },
            "required": ["name"]
        }
    
    try:
        validate(instance=config_data, schema=schema)
        return True
    except ValidationError as e:
        raise ValidationError(f"配置验证失败: {e.message}")


def validate_workflow_config(config_data: Dict[str, Any]) -> bool:
    """
    验证工作流配置数据
    
    Args:
        config_data: 工作流配置数据
        
    Returns:
        验证是否通过
        
    Raises:
        ValidationError: 验证失败
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "workflow_type": {
                "type": "string",
                "enum": ["sequential", "parallel", "conditional", "custom"]
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "node_type": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "config": {"type": "object"}
                    },
                    "required": ["node_id", "node_type"]
                }
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "edge_id": {"type": "string"},
                        "source_node": {"type": "string"},
                        "target_node": {"type": "string"},
                        "condition": {"type": "object"}
                    },
                    "required": ["edge_id", "source_node", "target_node"]
                }
            }
        },
        "required": ["name", "workflow_type"]
    }
    
    try:
        validate(instance=config_data, schema=schema)
        return True
    except ValidationError as e:
        raise ValidationError(f"工作流配置验证失败: {e.message}")


def validate_pagination_params(page: str, limit: str) -> tuple[int, int]:
    """
    验证分页参数
    
    Args:
        page: 页码字符串
        limit: 限制字符串
        
    Returns:
        验证后的页码和限制
        
    Raises:
        ValueError: 参数无效
    """
    try:
        page_int = int(page) if page else 1
        limit_int = int(limit) if limit else 20
        
        if page_int < 1:
            raise ValueError("页码必须大于0")
        if limit_int < 1 or limit_int > 100:
            raise ValueError("每页限制必须在1-100之间")
        
        return page_int, limit_int
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("页码和限制必须是有效数字")
        raise


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    验证必需字段
    
    Args:
        data: 数据字典
        required_fields: 必需字段列表
        
    Raises:
        ValueError: 缺少必需字段
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"缺少必需字段: {', '.join(missing_fields)}")


def validate_string_length(value: str, field_name: str, min_length: int = 1, max_length: int = 255) -> None:
    """
    验证字符串长度
    
    Args:
        value: 字符串值
        field_name: 字段名称
        min_length: 最小长度
        max_length: 最大长度
        
    Raises:
        ValueError: 长度不符合要求
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name}必须是字符串")
    
    if len(value) < min_length:
        raise ValueError(f"{field_name}长度不能少于{min_length}个字符")
    
    if len(value) > max_length:
        raise ValueError(f"{field_name}长度不能超过{max_length}个字符")


def validate_enum_value(value: Any, field_name: str, allowed_values: List[Any]) -> None:
    """
    验证枚举值
    
    Args:
        value: 要验证的值
        field_name: 字段名称
        allowed_values: 允许的值列表
        
    Raises:
        ValueError: 如果值不在允许的值列表中
    """
    if value not in allowed_values:
        raise ValueError(f"{field_name} must be one of {allowed_values}, got {value}")


def validate_phone_format(phone: str) -> None:
    """
    验证手机号格式
    
    Args:
        phone: 手机号
        
    Raises:
        ValueError: 如果手机号格式不正确
    """
    import re
    
    if not phone:
        raise ValueError("手机号不能为空")
    
    # 简单的中国手机号验证
    pattern = r'^1[3-9]\d{9}$'
    if not re.match(pattern, phone):
        raise ValueError("手机号格式不正确")


def validate_email_format(email: str) -> None:
    """
    验证邮箱格式
    
    Args:
        email: 邮箱地址
        
    Raises:
        ValueError: 如果邮箱格式不正确
    """
    import re
    
    if not email:
        raise ValueError("邮箱不能为空")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError("邮箱格式不正确")


def validate_id(value: str, field_name: str) -> None:
    """验证ID格式
    
    Args:
        value: ID值
        field_name: 字段名称
        
    Raises:
        ValueError: ID格式不正确时抛出
    """
    if not value:
        raise ValueError(f"{field_name}不能为空")
    
    # 验证UUID格式
    try:
        import uuid
        uuid.UUID(value)
    except ValueError:
        raise ValueError(f"{field_name}格式不正确")