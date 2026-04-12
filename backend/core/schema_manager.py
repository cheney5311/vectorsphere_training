"""JSON Schema管理模块

提供schema加载、缓存、版本控制和兼容性检查功能。
"""
import json
import os
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from functools import lru_cache
import jsonschema
from jsonschema import Draft7Validator, ValidationError
import threading

from .errors import make_error

logger = logging.getLogger(__name__)


class SchemaManager:
    """JSON Schema管理器"""
    
    def __init__(self, schema_dir: str = None):
        """初始化Schema管理器
        
        Args:
            schema_dir: schema文件目录路径
        """
        self.schema_dir = Path(schema_dir) if schema_dir else Path(__file__).parent.parent / "api" / "schemas"
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._schema_versions: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        
        # 确保schema目录存在
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化时加载所有schema
        self._load_all_schemas()
    
    def _load_all_schemas(self):
        """加载所有schema文件"""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
        
        for schema_file in self.schema_dir.rglob("*.json"):
            try:
                self._load_schema_file(schema_file)
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def _load_schema_file(self, schema_file: Path):
        """加载单个schema文件"""
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # 从文件路径解析schema名称和版本
            relative_path = schema_file.relative_to(self.schema_dir)
            schema_name = str(relative_path.with_suffix(''))
            
            # 支持版本化schema: schemas/v1/training.json -> training:v1
            parts = schema_name.split('/')
            if len(parts) > 1 and parts[0].startswith('v'):
                version = parts[0]
                name = '/'.join(parts[1:])
                schema_key = f"{name}:{version}"
            else:
                schema_key = schema_name
                version = "latest"
            
            # 验证schema本身的有效性
            try:
                Draft7Validator.check_schema(schema_data)
            except jsonschema.SchemaError as e:
                logger.error(f"Invalid schema {schema_key}: {e}")
                return
            
            with self._lock:
                self._schema_cache[schema_key] = schema_data
                
                # 更新版本信息
                base_name = schema_key.split(':')[0]
                if base_name not in self._schema_versions:
                    self._schema_versions[base_name] = []
                if version not in self._schema_versions[base_name]:
                    self._schema_versions[base_name].append(version)
            
            logger.info(f"Loaded schema: {schema_key}")
            
        except Exception as e:
            logger.error(f"Failed to load schema file {schema_file}: {e}")
            raise
    
    def get_schema(self, schema_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """获取指定的schema
        
        Args:
            schema_name: schema名称
            version: schema版本
            
        Returns:
            schema字典，如果不存在返回None
        """
        schema_key = f"{schema_name}:{version}" if version != "latest" else schema_name
        
        with self._lock:
            schema = self._schema_cache.get(schema_key)
            if schema is None and version == "latest":
                # 尝试获取最新版本
                versions = self._schema_versions.get(schema_name, [])
                if versions:
                    # 按版本号排序，获取最新版本
                    latest_version = self._get_latest_version(versions)
                    schema_key = f"{schema_name}:{latest_version}"
                    schema = self._schema_cache.get(schema_key)
        
        return schema
    
    def _get_latest_version(self, versions: List[str]) -> str:
        """获取最新版本号"""
        if "latest" in versions:
            return "latest"
        
        # 简单的版本排序，支持 v1, v2, v1.0, v1.1 等格式
        def version_key(v):
            if v.startswith('v'):
                v = v[1:]
            parts = v.split('.')
            return tuple(int(p) if p.isdigit() else 0 for p in parts)
        
        try:
            return sorted(versions, key=version_key, reverse=True)[0]
        except:
            return versions[-1]  # 如果排序失败，返回最后一个
    
    def validate_data(self, data: Any, schema_name: str, version: str = "latest") -> Tuple[bool, Optional[Dict[str, Any]]]:
        """验证数据是否符合schema
        
        Args:
            data: 要验证的数据
            schema_name: schema名称
            version: schema版本
            
        Returns:
            (是否有效, 错误信息)
        """
        schema = self.get_schema(schema_name, version)
        if schema is None:
            error = make_error(
                'SCHEMA_LOAD_FAILED',
                f'Schema not found: {schema_name}:{version}',
                details={'schema_name': schema_name, 'version': version}
            )
            return False, error
        
        try:
            validator = Draft7Validator(schema)
            validator.validate(data)
            return True, None
        except ValidationError as e:
            error = make_error(
                'VALIDATION_SCHEMA_FAILED',
                f'Validation failed: {e.message}',
                details={
                    'schema_name': schema_name,
                    'version': version,
                    'validation_path': list(e.absolute_path),
                    'failed_value': e.instance,
                    'schema_path': list(e.schema_path)
                }
            )
            return False, error
        except Exception as e:
            error = make_error(
                'INTERNAL_ERROR',
                f'Validation error: {str(e)}',
                details={'schema_name': schema_name, 'version': version}
            )
            return False, error
    
    def get_available_schemas(self) -> Dict[str, List[str]]:
        """获取所有可用的schema及其版本"""
        with self._lock:
            return dict(self._schema_versions)
    
    def reload_schemas(self):
        """重新加载所有schema"""
        with self._lock:
            self._schema_cache.clear()
            self._schema_versions.clear()
        self._load_all_schemas()
    
    def add_schema(self, schema_name: str, schema_data: Dict[str, Any], version: str = "latest"):
        """动态添加schema
        
        Args:
            schema_name: schema名称
            schema_data: schema数据
            version: schema版本
        """
        # 验证schema有效性
        try:
            Draft7Validator.check_schema(schema_data)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema: {e}")
        
        schema_key = f"{schema_name}:{version}" if version != "latest" else schema_name
        
        with self._lock:
            self._schema_cache[schema_key] = schema_data
            
            if schema_name not in self._schema_versions:
                self._schema_versions[schema_name] = []
            if version not in self._schema_versions[schema_name]:
                self._schema_versions[schema_name].append(version)
        
        logger.info(f"Added schema: {schema_key}")
    
    def check_compatibility(self, schema_name: str, old_version: str, new_version: str) -> Dict[str, Any]:
        """检查schema版本兼容性
        
        Args:
            schema_name: schema名称
            old_version: 旧版本
            new_version: 新版本
            
        Returns:
            兼容性检查结果
        """
        old_schema = self.get_schema(schema_name, old_version)
        new_schema = self.get_schema(schema_name, new_version)
        
        if old_schema is None or new_schema is None:
            return {
                'compatible': False,
                'reason': 'Schema not found',
                'breaking_changes': []
            }
        
        # 简单的兼容性检查
        breaking_changes = []
        
        # 检查必需字段是否增加
        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))
        added_required = new_required - old_required
        if added_required:
            breaking_changes.append(f"Added required fields: {list(added_required)}")
        
        # 检查字段类型是否改变
        old_props = old_schema.get('properties', {})
        new_props = new_schema.get('properties', {})
        for field in old_props:
            if field in new_props:
                old_type = old_props[field].get('type')
                new_type = new_props[field].get('type')
                if old_type != new_type:
                    breaking_changes.append(f"Changed type of field '{field}': {old_type} -> {new_type}")
        
        return {
            'compatible': len(breaking_changes) == 0,
            'breaking_changes': breaking_changes,
            'added_fields': list(set(new_props.keys()) - set(old_props.keys())),
            'removed_fields': list(set(old_props.keys()) - set(new_props.keys()))
        }


# 全局schema管理器实例
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """获取全局schema管理器实例"""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager


@lru_cache(maxsize=128)
def get_cached_schema(schema_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
    """获取缓存的schema（带LRU缓存）"""
    return get_schema_manager().get_schema(schema_name, version)


def validate_json_data(data: Any, schema_name: str, version: str = "latest") -> Tuple[bool, Optional[Dict[str, Any]]]:
    """验证JSON数据的便捷函数"""
    return get_schema_manager().validate_data(data, schema_name, version)