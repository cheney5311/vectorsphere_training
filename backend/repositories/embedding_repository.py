#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入向量数据访问层

提供嵌入向量模块的数据持久化操作：
- 嵌入记录的 CRUD
- 嵌入任务管理
- 模型配置管理
- 相似度搜索记录
- 缓存管理

架构调用关系：
Service层 (embedding_service.py)
    -> Repository层 (本模块)
        -> Database层 (DatabaseService)
"""

import logging
import uuid
import hashlib
import struct
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """嵌入向量数据访问层
    
    提供嵌入记录、任务、模型配置的数据持久化操作。
    支持内存存储模式（用于测试）和数据库存储模式。
    """
    
    def __init__(self, use_memory: bool = False):
        """初始化仓库
        
        Args:
            use_memory: 是否使用内存存储（用于测试）
        """
        self._use_memory = use_memory
        self._db_service = None
        
        # 内存存储
        self._embedding_records: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}  # content_hash -> embedding
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._search_records: Dict[str, Dict[str, Any]] = {}
        
        # 统计信息
        self._stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'total_tokens': 0,
            'models_used': {}
        }
        
        if not use_memory:
            self._init_database()
        
        # 初始化默认模型配置
        self._init_default_models()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.modules.database.service import DatabaseService
            self._db_service = DatabaseService()
            logger.info("EmbeddingRepository: Database service initialized")
        except Exception as e:
            logger.warning(f"EmbeddingRepository: Failed to init database: {e}")
            self._use_memory = True
    
    def _init_default_models(self):
        """初始化默认模型配置"""
        default_models = [
            {
                'model_type': 'sentence-transformers',
                'model_name': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'max_tokens': 256,
                'is_default': True,
                'is_active': True,
                'description': 'Sentence Transformers 轻量级模型，适合一般文本'
            },
            {
                'model_type': 'bge',
                'model_name': 'bge-large-zh-v1.5',
                'dimension': 1024,
                'max_tokens': 512,
                'is_default': False,
                'is_active': True,
                'description': 'BGE 中文大模型，适合中文文本'
            },
            {
                'model_type': 'm3e',
                'model_name': 'm3e-base',
                'dimension': 768,
                'max_tokens': 512,
                'is_default': False,
                'is_active': True,
                'description': 'M3E 中文通用模型'
            },
            {
                'model_type': 'openai',
                'model_name': 'text-embedding-ada-002',
                'dimension': 1536,
                'max_tokens': 8191,
                'is_default': False,
                'is_active': False,
                'description': 'OpenAI Ada 嵌入模型（需要API密钥）'
            },
            {
                'model_type': 'tfidf',
                'model_name': 'tfidf-default',
                'dimension': 384,
                'max_tokens': 10000,
                'is_default': False,
                'is_active': True,
                'description': 'TF-IDF 统计模型，无需深度学习'
            },
            {
                'model_type': 'word2vec',
                'model_name': 'word2vec-default',
                'dimension': 384,
                'max_tokens': 10000,
                'is_default': False,
                'is_active': True,
                'description': 'Word2Vec 词向量平均模型'
            },
        ]
        
        for model_config in default_models:
            model_id = str(uuid.uuid4())
            model_config['id'] = model_id
            model_config['created_at'] = datetime.utcnow().isoformat()
            model_config['tenant_id'] = 'default'
            self._model_configs[model_id] = model_config
    
    # ==========================================================================
    # 嵌入记录操作
    # ==========================================================================
    
    def create_embedding_record(
        self,
        user_id: str,
        content_hash: str,
        embedding: np.ndarray,
        model_type: str,
        content_preview: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: float = 0.0,
        token_count: int = 0,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建嵌入记录
        
        Args:
            user_id: 用户ID
            content_hash: 内容哈希值
            embedding: 嵌入向量
            model_type: 模型类型
            content_preview: 内容预览
            metadata: 元数据
            processing_time_ms: 处理时间
            token_count: 令牌数量
            tenant_id: 租户ID
            
        Returns:
            创建的嵌入记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # 将嵌入向量转换为二进制
        embedding_blob = self._embedding_to_bytes(embedding)
        
        record = {
            'id': record_id,
            'user_id': user_id,
            'content_hash': content_hash,
            'content_type': 'text',
            'content_preview': content_preview,
            'model_type': model_type,
            'model_name': self._get_model_name(model_type),
            'dimension': len(embedding),
            'embedding_blob': embedding_blob,
            'metadata': metadata or {},
            'processing_time_ms': processing_time_ms,
            'token_count': token_count,
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._embedding_records[record_id] = record
            # 同时缓存嵌入向量
            cache_key = f"{model_type}:{content_hash}"
            self._embedding_cache[cache_key] = embedding
        else:
            try:
                from backend.schemas.embedding_models import EmbeddingRecordDB
                record_db = EmbeddingRecordDB(
                    id=uuid.UUID(record_id),
                    user_id=user_id,
                    content_hash=content_hash,
                    content_type='text',
                    content_preview=content_preview,
                    model_type=model_type,
                    model_name=self._get_model_name(model_type),
                    dimension=len(embedding),
                    embedding_blob=embedding_blob,
                    metadata_json=metadata or {},
                    processing_time_ms=processing_time_ms,
                    token_count=token_count,
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(record_db)
                record = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create embedding record: {e}")
                self._embedding_records[record_id] = record
                cache_key = f"{model_type}:{content_hash}"
                self._embedding_cache[cache_key] = embedding
        
        # 更新统计
        self._update_stats(model_type, processing_time_ms, token_count)
        
        logger.debug(f"Created embedding record: {record_id}")
        return record
    
    def get_embedding_by_hash(
        self,
        content_hash: str,
        model_type: str
    ) -> Optional[np.ndarray]:
        """通过内容哈希获取缓存的嵌入向量
        
        Args:
            content_hash: 内容哈希值
            model_type: 模型类型
            
        Returns:
            嵌入向量或None
        """
        cache_key = f"{model_type}:{content_hash}"
        
        # 先检查内存缓存
        if cache_key in self._embedding_cache:
            self._stats['cache_hits'] += 1
            return self._embedding_cache[cache_key]
        
        # 再检查数据库
        if not self._use_memory:
            try:
                from backend.schemas.embedding_models import EmbeddingRecordDB
                records = self._db_service.filter_by(
                    EmbeddingRecordDB,
                    content_hash=content_hash,
                    model_type=model_type
                )
                if records:
                    record = records[0]
                    if record.embedding_blob:
                        embedding = self._bytes_to_embedding(record.embedding_blob)
                        # 加入内存缓存
                        self._embedding_cache[cache_key] = embedding
                        self._stats['cache_hits'] += 1
                        return embedding
            except Exception as e:
                logger.error(f"Failed to get embedding by hash: {e}")
        
        self._stats['cache_misses'] += 1
        return None
    
    def get_embedding_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取嵌入记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            嵌入记录
        """
        if self._use_memory:
            return self._embedding_records.get(record_id)
        
        try:
            from backend.schemas.embedding_models import EmbeddingRecordDB
            record_db = self._db_service.get_by_id(EmbeddingRecordDB, record_id)
            return record_db.to_dict() if record_db else None
        except Exception as e:
            logger.error(f"Failed to get embedding record: {e}")
            return self._embedding_records.get(record_id)
    
    def list_user_embeddings(
        self,
        user_id: str,
        model_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的嵌入记录列表
        
        Args:
            user_id: 用户ID
            model_type: 模型类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            records = [r for r in self._embedding_records.values() 
                      if r.get('user_id') == user_id]
            if model_type:
                records = [r for r in records if r.get('model_type') == model_type]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        
        try:
            from backend.schemas.embedding_models import EmbeddingRecordDB
            records_db = self._db_service.filter_by(EmbeddingRecordDB, user_id=user_id)
            records = [r.to_dict() for r in records_db]
            if model_type:
                records = [r for r in records if r.get('model_type') == model_type]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to list user embeddings: {e}")
            return [], 0
    
    def delete_embedding_record(self, record_id: str) -> bool:
        """删除嵌入记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory:
            if record_id in self._embedding_records:
                record = self._embedding_records.pop(record_id)
                # 同时删除缓存
                cache_key = f"{record['model_type']}:{record['content_hash']}"
                self._embedding_cache.pop(cache_key, None)
                return True
            return False
        
        try:
            from backend.schemas.embedding_models import EmbeddingRecordDB
            record_db = self._db_service.get_by_id(EmbeddingRecordDB, record_id)
            if record_db:
                self._db_service.delete(record_db)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete embedding record: {e}")
            return False
    
    # ==========================================================================
    # 模型配置操作
    # ==========================================================================
    
    def get_model_config(
        self,
        model_type: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取模型配置
        
        Args:
            model_type: 模型类型
            model_name: 模型名称（可选）
            
        Returns:
            模型配置
        """
        for config in self._model_configs.values():
            if config.get('model_type') == model_type:
                if model_name is None or config.get('model_name') == model_name:
                    return config
        return None
    
    def list_model_configs(
        self,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """获取模型配置列表
        
        Args:
            active_only: 是否只返回激活的模型
            
        Returns:
            模型配置列表
        """
        configs = list(self._model_configs.values())
        if active_only:
            configs = [c for c in configs if c.get('is_active', True)]
        return configs
    
    def get_default_model(self) -> Optional[Dict[str, Any]]:
        """获取默认模型配置
        
        Returns:
            默认模型配置
        """
        for config in self._model_configs.values():
            if config.get('is_default', False):
                return config
        
        # 如果没有默认模型，返回第一个激活的模型
        active_models = [c for c in self._model_configs.values() if c.get('is_active', True)]
        return active_models[0] if active_models else None
    
    def update_model_stats(
        self,
        model_type: str,
        requests: int = 1,
        tokens: int = 0,
        latency_ms: float = 0.0
    ):
        """更新模型统计信息
        
        Args:
            model_type: 模型类型
            requests: 请求数
            tokens: 令牌数
            latency_ms: 延迟(毫秒)
        """
        for config_id, config in self._model_configs.items():
            if config.get('model_type') == model_type:
                config['total_requests'] = config.get('total_requests', 0) + requests
                config['total_tokens'] = config.get('total_tokens', 0) + tokens
                
                # 更新平均延迟
                total_requests = config['total_requests']
                old_avg = config.get('avg_latency_ms', 0.0)
                config['avg_latency_ms'] = (old_avg * (total_requests - 1) + latency_ms) / total_requests
                break
    
    # ==========================================================================
    # 任务管理操作
    # ==========================================================================
    
    def create_task(
        self,
        user_id: str,
        model_type: str,
        total_items: int,
        task_type: str = 'batch_generate',
        priority: int = 5,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建嵌入任务
        
        Args:
            user_id: 用户ID
            model_type: 模型类型
            total_items: 总项数
            task_type: 任务类型
            priority: 优先级
            tenant_id: 租户ID
            
        Returns:
            创建的任务
        """
        task_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        task = {
            'id': task_id,
            'user_id': user_id,
            'task_type': task_type,
            'status': 'pending',
            'priority': priority,
            'model_type': model_type,
            'model_name': self._get_model_name(model_type),
            'total_items': total_items,
            'processed_items': 0,
            'failed_items': 0,
            'progress': 0.0,
            'started_at': None,
            'completed_at': None,
            'total_tokens': 0,
            'processing_time_ms': 0.0,
            'error_message': None,
            'result_summary': {},
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._tasks[task_id] = task
        else:
            try:
                from backend.schemas.embedding_models import EmbeddingTaskDB
                task_db = EmbeddingTaskDB(
                    id=uuid.UUID(task_id),
                    user_id=user_id,
                    task_type=task_type,
                    status='pending',
                    priority=priority,
                    model_type=model_type,
                    model_name=self._get_model_name(model_type),
                    total_items=total_items,
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(task_db)
                task = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create task: {e}")
                self._tasks[task_id] = task
        
        logger.info(f"Created embedding task: {task_id}")
        return task
    
    def update_task(
        self,
        task_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """更新任务
        
        Args:
            task_id: 任务ID
            **updates: 更新字段
            
        Returns:
            更新后的任务
        """
        if self._use_memory:
            if task_id in self._tasks:
                self._tasks[task_id].update(updates)
                return self._tasks[task_id]
            return None
        
        try:
            from backend.schemas.embedding_models import EmbeddingTaskDB
            task_db = self._db_service.get_by_id(EmbeddingTaskDB, task_id)
            if task_db:
                updated = self._db_service.update(task_db, updates)
                return updated.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to update task: {e}")
            return None
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息
        """
        if self._use_memory:
            return self._tasks.get(task_id)
        
        try:
            from backend.schemas.embedding_models import EmbeddingTaskDB
            task_db = self._db_service.get_by_id(EmbeddingTaskDB, task_id)
            return task_db.to_dict() if task_db else None
        except Exception as e:
            logger.error(f"Failed to get task: {e}")
            return self._tasks.get(task_id)
    
    def list_user_tasks(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户任务列表
        
        Args:
            user_id: 用户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (任务列表, 总数)
        """
        if self._use_memory:
            tasks = [t for t in self._tasks.values() 
                    if t.get('user_id') == user_id]
            if status:
                tasks = [t for t in tasks if t.get('status') == status]
            tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(tasks)
            return tasks[offset:offset + limit], total
        
        try:
            from backend.schemas.embedding_models import EmbeddingTaskDB
            tasks_db = self._db_service.filter_by(EmbeddingTaskDB, user_id=user_id)
            tasks = [t.to_dict() for t in tasks_db]
            if status:
                tasks = [t for t in tasks if t.get('status') == status]
            tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(tasks)
            return tasks[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to list user tasks: {e}")
            return [], 0
    
    # ==========================================================================
    # 相似度搜索记录
    # ==========================================================================
    
    def create_search_record(
        self,
        user_id: str,
        query_hash: str,
        query_preview: str,
        model_type: str,
        similarity_metric: str,
        top_k: int,
        results: List[Dict[str, Any]],
        search_time_ms: float,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建搜索记录
        
        Args:
            user_id: 用户ID
            query_hash: 查询哈希
            query_preview: 查询预览
            model_type: 模型类型
            similarity_metric: 相似度方法
            top_k: 返回数量
            results: 搜索结果
            search_time_ms: 搜索时间
            tenant_id: 租户ID
            
        Returns:
            创建的搜索记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        record = {
            'id': record_id,
            'user_id': user_id,
            'query_hash': query_hash,
            'query_preview': query_preview,
            'model_type': model_type,
            'similarity_metric': similarity_metric,
            'top_k': top_k,
            'result_count': len(results),
            'results': results,
            'search_time_ms': search_time_ms,
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._search_records[record_id] = record
        
        logger.debug(f"Created search record: {record_id}")
        return record
    
    # ==========================================================================
    # 缓存管理
    # ==========================================================================
    
    def clear_cache(self):
        """清空缓存"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._embedding_cache)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self._embedding_cache),
            'hits': self._stats['cache_hits'],
            'misses': self._stats['cache_misses'],
            'hit_rate': self._stats['cache_hits'] / max(
                self._stats['cache_hits'] + self._stats['cache_misses'], 1
            )
        }
    
    # ==========================================================================
    # 统计信息
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        
        return {
            'total_embeddings': self._stats['total_embeddings'],
            'total_tokens': self._stats['total_tokens'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': self._stats['cache_hits'] / max(total_requests, 1),
            'avg_processing_time_ms': (
                self._stats['total_processing_time'] / 
                max(self._stats['total_embeddings'], 1)
            ),
            'total_processing_time_ms': self._stats['total_processing_time'],
            'cache_size': len(self._embedding_cache),
            'models_used': self._stats['models_used'],
        }
    
    def _update_stats(
        self,
        model_type: str,
        processing_time_ms: float,
        token_count: int
    ):
        """更新统计信息"""
        self._stats['total_embeddings'] += 1
        self._stats['total_processing_time'] += processing_time_ms
        self._stats['total_tokens'] += token_count
        
        if model_type not in self._stats['models_used']:
            self._stats['models_used'][model_type] = 0
        self._stats['models_used'][model_type] += 1
        
        # 更新模型统计
        self.update_model_stats(model_type, tokens=token_count, latency_ms=processing_time_ms)
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'total_tokens': 0,
            'models_used': {}
        }
    
    # ==========================================================================
    # 工具方法
    # ==========================================================================
    
    def _get_model_name(self, model_type: str) -> str:
        """获取模型名称"""
        config = self.get_model_config(model_type)
        if config:
            return config.get('model_name', model_type)
        return model_type
    
    def _embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """将嵌入向量转换为字节"""
        if embedding is None:
            return b''
        # 确保是float32类型
        embedding = embedding.astype(np.float32)
        # 转换为字节
        return embedding.tobytes()
    
    def _bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """将字节转换为嵌入向量"""
        if not data:
            return np.array([], dtype=np.float32)
        # 从字节恢复数组
        return np.frombuffer(data, dtype=np.float32)


# ==================== 全局单例 ====================

_global_repository: Optional[EmbeddingRepository] = None


def get_embedding_repository(use_memory: bool = False) -> EmbeddingRepository:
    """获取嵌入仓库实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        EmbeddingRepository 实例
    """
    global _global_repository
    
    if _global_repository is None:
        _global_repository = EmbeddingRepository(use_memory=use_memory)
    
    return _global_repository


def reset_embedding_repository():
    """重置全局仓库实例（用于测试）"""
    global _global_repository
    _global_repository = None


# ==================== 导出 ====================

__all__ = [
    'EmbeddingRepository',
    'get_embedding_repository',
    'reset_embedding_repository',
]
