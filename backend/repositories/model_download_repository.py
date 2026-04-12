#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型下载数据访问层

提供模型下载记录的数据持久化操作：
- 下载记录CRUD
- 下载统计
- 令牌管理
- 过期清理
"""

import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DownloadRecordRepository:
    """下载记录仓库
    
    管理下载记录的存储和查询
    """
    
    def __init__(self, use_memory: bool = True):
        """初始化仓库
        
        Args:
            use_memory: 是否使用内存存储
        """
        self._use_memory = use_memory
        self._lock = threading.RLock()
        
        # 内存存储
        self._records: Dict[str, Dict[str, Any]] = {}
        self._records_by_model: Dict[str, List[str]] = defaultdict(list)
        self._records_by_user: Dict[str, List[str]] = defaultdict(list)
        self._records_by_token: Dict[str, str] = {}
        
        # 数据库服务
        self._db_service = None
        if not use_memory:
            self._init_db_service()
        
        logger.info(f"DownloadRecordRepository initialized (memory={use_memory})")
    
    def _init_db_service(self):
        """初始化数据库服务"""
        try:
            from backend.modules.database.service import DatabaseService
            self._db_service = DatabaseService()
            logger.info("Database service initialized for DownloadRecordRepository")
        except ImportError as e:
            logger.warning(f"Failed to import DatabaseService: {e}")
            self._use_memory = True
    
    # ==================== 记录CRUD ====================
    
    def create_record(
        self,
        model_id: str,
        user_id: str,
        download_format: str = 'pytorch',
        download_source: str = 'model',
        version_id: Optional[str] = None,
        training_session_id: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size: int = 0,
        file_name: Optional[str] = None,
        checksum: Optional[str] = None,
        download_token: Optional[str] = None,
        expire_at: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建下载记录
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            download_format: 下载格式
            download_source: 下载来源
            version_id: 版本ID
            training_session_id: 训练会话ID
            file_path: 文件路径
            file_size: 文件大小
            file_name: 文件名
            checksum: 校验和
            download_token: 下载令牌
            expire_at: 过期时间
            tenant_id: 租户ID
            metadata: 元数据
            
        Returns:
            创建的记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        record = {
            'id': record_id,
            'model_id': model_id,
            'user_id': user_id,
            'download_format': download_format,
            'download_source': download_source,
            'version_id': version_id,
            'training_session_id': training_session_id,
            'status': 'pending',
            'file_path': file_path,
            'file_size': file_size,
            'file_name': file_name,
            'checksum': checksum,
            'download_url': None,
            'download_token': download_token,
            'expire_at': expire_at.isoformat() if expire_at else None,
            'download_count': 0,
            'last_download_at': None,
            'download_ip': None,
            'user_agent': None,
            'tenant_id': tenant_id,
            'metadata': metadata or {},
            'created_at': now.isoformat(),
            'updated_at': now.isoformat()
        }
        
        if self._use_memory:
            with self._lock:
                self._records[record_id] = record
                self._records_by_model[model_id].append(record_id)
                self._records_by_user[user_id].append(record_id)
                if download_token:
                    self._records_by_token[download_token] = record_id
        else:
            # 数据库存储
            self._save_to_db(record)
        
        logger.info(f"Created download record: {record_id} for model {model_id}")
        return record
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            记录数据
        """
        if self._use_memory:
            with self._lock:
                return self._records.get(record_id, {}).copy() if record_id in self._records else None
        else:
            return self._get_from_db(record_id)
    
    def get_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """根据令牌获取记录
        
        Args:
            token: 下载令牌
            
        Returns:
            记录数据
        """
        if self._use_memory:
            with self._lock:
                record_id = self._records_by_token.get(token)
                if record_id:
                    return self._records.get(record_id, {}).copy()
                return None
        else:
            return self._get_by_token_from_db(token)
    
    def update_record(self, record_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新记录
        
        Args:
            record_id: 记录ID
            updates: 更新字段
            
        Returns:
            更新后的记录
        """
        if self._use_memory:
            with self._lock:
                if record_id not in self._records:
                    return None
                
                record = self._records[record_id]
                record.update(updates)
                record['updated_at'] = datetime.utcnow().isoformat()
                
                # 更新令牌索引
                if 'download_token' in updates:
                    old_token = record.get('download_token')
                    new_token = updates['download_token']
                    if old_token and old_token in self._records_by_token:
                        del self._records_by_token[old_token]
                    if new_token:
                        self._records_by_token[new_token] = record_id
                
                return record.copy()
        else:
            return self._update_in_db(record_id, updates)
    
    def delete_record(self, record_id: str) -> bool:
        """删除记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否成功
        """
        if self._use_memory:
            with self._lock:
                if record_id not in self._records:
                    return False
                
                record = self._records[record_id]
                model_id = record.get('model_id')
                user_id = record.get('user_id')
                token = record.get('download_token')
                
                del self._records[record_id]
                
                if model_id and record_id in self._records_by_model.get(model_id, []):
                    self._records_by_model[model_id].remove(record_id)
                if user_id and record_id in self._records_by_user.get(user_id, []):
                    self._records_by_user[user_id].remove(record_id)
                if token and token in self._records_by_token:
                    del self._records_by_token[token]
                
                return True
        else:
            return self._delete_from_db(record_id)
    
    # ==================== 查询方法 ====================
    
    def list_by_model(
        self,
        model_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型的下载记录
        
        Args:
            model_id: 模型ID
            limit: 返回数量
            offset: 偏移量
            status: 状态过滤
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            with self._lock:
                record_ids = self._records_by_model.get(model_id, [])
                records = [self._records[rid].copy() for rid in record_ids if rid in self._records]
                
                if status:
                    records = [r for r in records if r.get('status') == status]
                
                # 按创建时间倒序
                records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                total = len(records)
                records = records[offset:offset + limit]
                
                return records, total
        else:
            return self._list_by_model_from_db(model_id, limit, offset, status)
    
    def list_by_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        model_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的下载记录
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            offset: 偏移量
            model_id: 模型ID过滤
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            with self._lock:
                record_ids = self._records_by_user.get(user_id, [])
                records = [self._records[rid].copy() for rid in record_ids if rid in self._records]
                
                if model_id:
                    records = [r for r in records if r.get('model_id') == model_id]
                
                records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                total = len(records)
                records = records[offset:offset + limit]
                
                return records, total
        else:
            return self._list_by_user_from_db(user_id, limit, offset, model_id)
    
    def get_active_download(self, model_id: str, user_id: str, download_format: str) -> Optional[Dict[str, Any]]:
        """获取有效的下载记录（未过期）
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            download_format: 下载格式
            
        Returns:
            有效的下载记录
        """
        now = datetime.utcnow()
        
        if self._use_memory:
            with self._lock:
                record_ids = self._records_by_model.get(model_id, [])
                
                for rid in record_ids:
                    record = self._records.get(rid)
                    if not record:
                        continue
                    
                    if record.get('user_id') != user_id:
                        continue
                    if record.get('download_format') != download_format:
                        continue
                    if record.get('status') not in ['ready', 'pending']:
                        continue
                    
                    expire_at_str = record.get('expire_at')
                    if expire_at_str:
                        expire_at = datetime.fromisoformat(expire_at_str)
                        if expire_at > now:
                            return record.copy()
                
                return None
        else:
            return self._get_active_from_db(model_id, user_id, download_format)
    
    # ==================== 统计方法 ====================
    
    def increment_download_count(
        self,
        record_id: str,
        download_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """增加下载计数
        
        Args:
            record_id: 记录ID
            download_ip: 下载IP
            user_agent: 用户代理
            
        Returns:
            更新后的记录
        """
        now = datetime.utcnow()
        
        updates = {
            'download_count': None,  # 将在更新时增加
            'last_download_at': now.isoformat(),
            'download_ip': download_ip,
            'user_agent': user_agent
        }
        
        if self._use_memory:
            with self._lock:
                if record_id not in self._records:
                    return None
                
                record = self._records[record_id]
                record['download_count'] = record.get('download_count', 0) + 1
                record['last_download_at'] = now.isoformat()
                if download_ip:
                    record['download_ip'] = download_ip
                if user_agent:
                    record['user_agent'] = user_agent
                record['updated_at'] = now.isoformat()
                
                return record.copy()
        else:
            return self._increment_count_in_db(record_id, download_ip, user_agent)
    
    def get_model_download_stats(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取模型下载统计
        
        Args:
            model_id: 模型ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            统计数据
        """
        if self._use_memory:
            with self._lock:
                record_ids = self._records_by_model.get(model_id, [])
                records = [self._records[rid] for rid in record_ids if rid in self._records]
                
                # 过滤日期范围
                if start_date or end_date:
                    filtered = []
                    for r in records:
                        created_at_str = r.get('created_at')
                        if not created_at_str:
                            continue
                        created_at = datetime.fromisoformat(created_at_str)
                        if start_date and created_at < start_date:
                            continue
                        if end_date and created_at > end_date:
                            continue
                        filtered.append(r)
                    records = filtered
                
                # 计算统计
                total_downloads = sum(r.get('download_count', 0) for r in records)
                unique_users = len(set(r.get('user_id') for r in records))
                total_size = sum(r.get('file_size', 0) * r.get('download_count', 0) for r in records)
                
                # 格式分布
                format_breakdown = defaultdict(int)
                for r in records:
                    fmt = r.get('download_format', 'unknown')
                    format_breakdown[fmt] += r.get('download_count', 0)
                
                # 来源分布
                source_breakdown = defaultdict(int)
                for r in records:
                    src = r.get('download_source', 'unknown')
                    source_breakdown[src] += r.get('download_count', 0)
                
                return {
                    'model_id': model_id,
                    'total_downloads': total_downloads,
                    'unique_users': unique_users,
                    'total_size_bytes': total_size,
                    'format_breakdown': dict(format_breakdown),
                    'source_breakdown': dict(source_breakdown),
                    'record_count': len(records)
                }
        else:
            return self._get_stats_from_db(model_id, start_date, end_date)
    
    def get_user_download_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户下载统计
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计数据
        """
        if self._use_memory:
            with self._lock:
                record_ids = self._records_by_user.get(user_id, [])
                records = [self._records[rid] for rid in record_ids if rid in self._records]
                
                total_downloads = sum(r.get('download_count', 0) for r in records)
                total_size = sum(r.get('file_size', 0) * r.get('download_count', 0) for r in records)
                unique_models = len(set(r.get('model_id') for r in records))
                
                return {
                    'user_id': user_id,
                    'total_downloads': total_downloads,
                    'total_size_bytes': total_size,
                    'unique_models': unique_models,
                    'record_count': len(records)
                }
        else:
            return self._get_user_stats_from_db(user_id)
    
    # ==================== 清理方法 ====================
    
    def cleanup_expired_records(self, before_date: Optional[datetime] = None) -> int:
        """清理过期记录
        
        Args:
            before_date: 清理此日期之前的记录
            
        Returns:
            清理的记录数
        """
        if before_date is None:
            before_date = datetime.utcnow() - timedelta(days=30)
        
        cleaned = 0
        
        if self._use_memory:
            with self._lock:
                to_delete = []
                
                for record_id, record in self._records.items():
                    expire_at_str = record.get('expire_at')
                    if expire_at_str:
                        expire_at = datetime.fromisoformat(expire_at_str)
                        if expire_at < before_date:
                            to_delete.append(record_id)
                
                for record_id in to_delete:
                    self.delete_record(record_id)
                    cleaned += 1
        else:
            cleaned = self._cleanup_expired_in_db(before_date)
        
        logger.info(f"Cleaned up {cleaned} expired download records")
        return cleaned
    
    def mark_expired(self, record_id: str) -> Optional[Dict[str, Any]]:
        """标记记录为过期
        
        Args:
            record_id: 记录ID
            
        Returns:
            更新后的记录
        """
        return self.update_record(record_id, {'status': 'expired'})
    
    # ==================== 私有方法（数据库操作占位） ====================
    
    def _save_to_db(self, record: Dict[str, Any]) -> bool:
        """保存到数据库"""
        # 实际实现需要使用 SQLAlchemy
        return True
    
    def _get_from_db(self, record_id: str) -> Optional[Dict[str, Any]]:
        """从数据库获取"""
        return None
    
    def _get_by_token_from_db(self, token: str) -> Optional[Dict[str, Any]]:
        """从数据库按令牌获取"""
        return None
    
    def _update_in_db(self, record_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新数据库记录"""
        return None
    
    def _delete_from_db(self, record_id: str) -> bool:
        """从数据库删除"""
        return False
    
    def _list_by_model_from_db(self, model_id: str, limit: int, offset: int, status: str) -> Tuple[List, int]:
        """从数据库按模型查询"""
        return [], 0
    
    def _list_by_user_from_db(self, user_id: str, limit: int, offset: int, model_id: str) -> Tuple[List, int]:
        """从数据库按用户查询"""
        return [], 0
    
    def _get_active_from_db(self, model_id: str, user_id: str, download_format: str) -> Optional[Dict[str, Any]]:
        """从数据库获取有效记录"""
        return None
    
    def _increment_count_in_db(self, record_id: str, download_ip: str, user_agent: str) -> Optional[Dict[str, Any]]:
        """在数据库中增加计数"""
        return None
    
    def _get_stats_from_db(self, model_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """从数据库获取统计"""
        return {}
    
    def _get_user_stats_from_db(self, user_id: str) -> Dict[str, Any]:
        """从数据库获取用户统计"""
        return {}
    
    def _cleanup_expired_in_db(self, before_date: datetime) -> int:
        """在数据库中清理过期记录"""
        return 0


class ModelDownloadRepository:
    """模型下载仓库
    
    整合下载记录管理的统一入口
    """
    
    def __init__(self, use_memory: bool = True):
        """初始化仓库
        
        Args:
            use_memory: 是否使用内存存储
        """
        self.records = DownloadRecordRepository(use_memory)
        self._use_memory = use_memory
        
        logger.info("ModelDownloadRepository initialized")
    
    def create_download_record(self, **kwargs) -> Dict[str, Any]:
        """创建下载记录"""
        return self.records.create_record(**kwargs)
    
    def get_download_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取下载记录"""
        return self.records.get_by_id(record_id)
    
    def get_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """根据令牌获取记录"""
        return self.records.get_by_token(token)
    
    def update_download_record(self, record_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新下载记录"""
        return self.records.update_record(record_id, kwargs)
    
    def delete_download_record(self, record_id: str) -> bool:
        """删除下载记录"""
        return self.records.delete_record(record_id)
    
    def list_model_downloads(
        self,
        model_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型下载记录"""
        return self.records.list_by_model(model_id, limit, offset, status)
    
    def list_user_downloads(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        model_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户下载记录"""
        return self.records.list_by_user(user_id, limit, offset, model_id)
    
    def get_active_download(
        self,
        model_id: str,
        user_id: str,
        download_format: str
    ) -> Optional[Dict[str, Any]]:
        """获取有效的下载记录"""
        return self.records.get_active_download(model_id, user_id, download_format)
    
    def increment_download_count(
        self,
        record_id: str,
        download_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """增加下载计数"""
        return self.records.increment_download_count(record_id, download_ip, user_agent)
    
    def get_model_statistics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取模型下载统计"""
        return self.records.get_model_download_stats(model_id, start_date, end_date)
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户下载统计"""
        return self.records.get_user_download_stats(user_id)
    
    def cleanup_expired(self, before_date: Optional[datetime] = None) -> int:
        """清理过期记录"""
        return self.records.cleanup_expired_records(before_date)


# ==================== 全局实例 ====================

_download_repository: Optional[ModelDownloadRepository] = None


def get_download_repository(use_memory: bool = True) -> ModelDownloadRepository:
    """获取下载仓库实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        ModelDownloadRepository 实例
    """
    global _download_repository
    
    if _download_repository is None:
        _download_repository = ModelDownloadRepository(use_memory)
    
    return _download_repository


def reset_download_repository():
    """重置下载仓库实例"""
    global _download_repository
    _download_repository = None


__all__ = [
    'DownloadRecordRepository',
    'ModelDownloadRepository',
    'get_download_repository',
    'reset_download_repository',
]
