#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入向量业务逻辑层

提供嵌入向量的核心业务逻辑：
- 嵌入向量生成（单个和批量）
- 相似度计算
- 相似度搜索
- 模型管理
- 统计和监控

架构调用关系：
API层 (api.py)
    -> Service层 (本模块)
        -> Repository层 (embedding_repository.py)
        -> EmbeddingManager (manager.py)

支持的嵌入模型：
- sentence-transformers: Sentence Transformers模型
- bge: BGE中文模型
- m3e: M3E模型
- openai: OpenAI Ada嵌入模型
- tfidf: TF-IDF统计模型
- word2vec: Word2Vec词向量模型
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# 异常类
try:
    from backend.core.exceptions import ValidationError, BusinessLogicError
except ImportError:
    class ValidationError(Exception):
        def __init__(self, message: str, field: str = None):
            self.message = message
            self.field = field
            super().__init__(message)
    
    class BusinessLogicError(Exception):
        def __init__(self, message: str, operation: str = None):
            self.message = message
            self.operation = operation
            super().__init__(message)


def _euclidean_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """欧几里得相似度（1 / (1 + distance)）"""
    distance = np.linalg.norm(emb1 - emb2)
    return 1.0 / (1.0 + distance)


def _manhattan_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """曼哈顿相似度（1 / (1 + distance)）"""
    distance = np.sum(np.abs(emb1 - emb2))
    return 1.0 / (1.0 + distance)


def _dot_product_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """点积相似度"""
    return float(np.dot(emb1, emb2))


class EmbeddingService:
    """嵌入向量业务逻辑层
    
    整合 EmbeddingManager 和 Repository 层，提供完整的嵌入向量业务逻辑。
    支持多种嵌入模型、缓存管理、批量处理和统计监控。
    """
    
    # 支持的模型类型
    SUPPORTED_MODELS = [
        'sentence-transformers', 'bge', 'm3e', 'openai', 
        'cohere', 'huggingface', 'tfidf', 'word2vec', 
        'fasttext', 'custom', 'default'
    ]
    
    # 支持的相似度计算方法
    SUPPORTED_METRICS = ['cosine', 'euclidean', 'dot_product', 'manhattan']
    
    # 默认配置
    DEFAULT_MODEL = 'sentence-transformers'
    DEFAULT_DIMENSION = 384
    MAX_BATCH_SIZE = 1000
    MAX_TEXT_LENGTH = 10000
    
    def __init__(self, use_memory: bool = False):
        """初始化服务
        
        Args:
            use_memory: 是否使用内存存储（用于测试）
        """
        self._use_memory = use_memory
        self._lock = threading.RLock()
        
        # 延迟初始化依赖
        self._repository = None
        self._embedding_manager = None
        
        # 批处理线程池
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self._init_dependencies()
    
    def _init_dependencies(self):
        """初始化依赖服务"""
        # 初始化 Repository
        try:
            from backend.repositories.embedding_repository import get_embedding_repository
            self._repository = get_embedding_repository(use_memory=self._use_memory)
            logger.info("EmbeddingService: Repository initialized")
        except Exception as e:
            logger.warning(f"EmbeddingService: Failed to init repository: {e}")
        
        # 初始化 EmbeddingManager
        try:
            from backend.modules.embeddings.manager import get_embedding_manager
            self._embedding_manager = get_embedding_manager()
            logger.info("EmbeddingService: EmbeddingManager initialized")
        except Exception as e:
            logger.warning(f"EmbeddingService: Failed to init EmbeddingManager: {e}")
    
    # ==========================================================================
    # 嵌入向量生成
    # ==========================================================================
    
    def generate_embedding(
        self,
        text: str,
        model_type: str = 'sentence-transformers',
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """生成单个文本的嵌入向量
        
        Args:
            text: 文本内容
            model_type: 模型类型
            user_id: 用户ID
            metadata: 元数据
            use_cache: 是否使用缓存
            
        Returns:
            嵌入结果字典
        """
        # 验证参数
        self._validate_text(text)
        self._validate_model_type(model_type)
        
        start_time = time.time()
        cached = False
        
        # 计算内容哈希
        from backend.schemas.embedding_models import compute_content_hash, estimate_token_count, truncate_text
        content_hash = compute_content_hash(text, model_type)
        
        # 检查缓存
        if use_cache and self._repository:
            cached_embedding = self._repository.get_embedding_by_hash(content_hash, model_type)
            if cached_embedding is not None:
                processing_time = (time.time() - start_time) * 1000
                return {
                    'embedding': cached_embedding.tolist(),
                    'dimension': len(cached_embedding),
                    'modelType': model_type,
                    'processingTimeMs': processing_time,
                    'tokenCount': estimate_token_count(text),
                    'cached': True
                }
        
        # 生成嵌入向量
        try:
            if self._embedding_manager:
                embedding = self._embedding_manager.generate_embedding(text, model_type)
            else:
                # 降级到简单实现
                embedding = self._generate_fallback_embedding(text, model_type)
            
            processing_time = (time.time() - start_time) * 1000
            token_count = estimate_token_count(text)
            
            # 保存到仓库
            if self._repository and user_id:
                self._repository.create_embedding_record(
                    user_id=user_id,
                    content_hash=content_hash,
                    embedding=embedding,
                    model_type=model_type,
                    content_preview=truncate_text(text, 100),
                    metadata=metadata,
                    processing_time_ms=processing_time,
                    token_count=token_count
                )
            
            return {
                'embedding': embedding.tolist(),
                'dimension': len(embedding),
                'modelType': model_type,
                'processingTimeMs': processing_time,
                'tokenCount': token_count,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise BusinessLogicError(f"生成嵌入向量失败: {e}", operation="generate_embedding")
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        model_type: str = 'sentence-transformers',
        user_id: Optional[str] = None,
        use_cache: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """批量生成嵌入向量
        
        Args:
            texts: 文本列表
            model_type: 模型类型
            user_id: 用户ID
            use_cache: 是否使用缓存
            parallel: 是否并行处理
            
        Returns:
            批量嵌入结果字典
        """
        # 验证参数
        if not texts:
            raise ValidationError("文本列表不能为空")
        if len(texts) > self.MAX_BATCH_SIZE:
            raise ValidationError(f"单次批量请求最多支持{self.MAX_BATCH_SIZE}条文本")
        
        self._validate_model_type(model_type)
        
        start_time = time.time()
        embeddings = []
        success_count = 0
        failed_count = 0
        total_tokens = 0

        # 创建任务记录
        task_id = None
        if self._repository and user_id:
            task = self._repository.create_task(
                user_id=user_id,
                model_type=model_type,
                total_items=len(texts),
                task_type='batch_generate'
            )
            task_id = task['id']
            
            # 更新任务状态为处理中
            self._repository.update_task(
                task_id,
                status='processing',
                started_at=datetime.utcnow().isoformat()
            )
        
        try:
            if parallel and len(texts) > 10:
                # 并行处理
                embeddings, success_count, failed_count, total_tokens = \
                    self._generate_batch_parallel(texts, model_type, user_id, use_cache)
            else:
                # 串行处理
                for text in texts:
                    try:
                        result = self.generate_embedding(
                            text, model_type, user_id, use_cache=use_cache
                        )
                        embeddings.append(result['embedding'])
                        success_count += 1
                        total_tokens += result.get('tokenCount', 0)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for text: {e}")
                        embeddings.append([0.0] * self.DEFAULT_DIMENSION)
                        failed_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            # 更新任务状态为完成
            if task_id and self._repository:
                self._repository.update_task(
                    task_id,
                    status='completed',
                    processed_items=success_count,
                    failed_items=failed_count,
                    progress=100.0,
                    completed_at=datetime.utcnow().isoformat(),
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time,
                    result_summary={
                        'success': success_count,
                        'failed': failed_count
                    }
                )
            
            # 获取维度
            dimension = len(embeddings[0]) if embeddings else self.DEFAULT_DIMENSION
            
            return {
                'embeddings': embeddings,
                'dimension': dimension,
                'modelType': model_type,
                'totalCount': len(texts),
                'successCount': success_count,
                'failedCount': failed_count,
                'processingTimeMs': processing_time,
                'totalTokens': total_tokens,
                'taskId': task_id
            }
            
        except Exception as e:
            # 更新任务状态为失败
            if task_id and self._repository:
                self._repository.update_task(
                    task_id,
                    status='failed',
                    error_message=str(e)
                )
            
            logger.error(f"Batch embedding failed: {e}")
            raise BusinessLogicError(f"批量生成嵌入向量失败: {e}", operation="generate_batch_embeddings")
    
    def _generate_batch_parallel(
        self,
        texts: List[str],
        model_type: str,
        user_id: Optional[str],
        use_cache: bool
    ) -> tuple[list[None], int, int, int]:
        """并行生成批量嵌入
        
        Args:
            texts: 文本列表
            model_type: 模型类型
            user_id: 用户ID
            use_cache: 是否使用缓存
            
        Returns:
            (嵌入列表, 成功数, 失败数, 总令牌数)
        """
        embeddings = [None] * len(texts)
        success_count = 0
        failed_count = 0
        total_tokens = 0
        
        def process_text(idx_text):
            idx, text = idx_text
            try:
                result = self.generate_embedding(
                    text, model_type, user_id, use_cache=use_cache
                )
                return idx, result['embedding'], result.get('tokenCount', 0), True
            except Exception as e:
                logger.warning(f"Failed to process text {idx}: {e}")
                return idx, [0.0] * self.DEFAULT_DIMENSION, 0, False
        
        # 使用线程池并行处理
        futures = []
        for idx, text in enumerate(texts):
            future = self._executor.submit(process_text, (idx, text))
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                idx, embedding, tokens, success = future.result()
                embeddings[idx] = embedding
                total_tokens += tokens
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Future execution error: {e}")
                failed_count += 1
        
        # 填充空值
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = [0.0] * self.DEFAULT_DIMENSION
        
        return embeddings, success_count, failed_count, total_tokens
    
    def _generate_fallback_embedding(
        self,
        text: str,
        model_type: str
    ) -> np.ndarray:
        """降级嵌入生成
        
        当 EmbeddingManager 不可用时使用
        
        Args:
            text: 文本内容
            model_type: 模型类型
            
        Returns:
            嵌入向量
        """
        import hashlib
        
        # 基于文本内容生成确定性向量
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 将哈希转换为浮点数向量
        embedding = []
        for i in range(0, min(len(text_hash), 32), 2):
            val = int(text_hash[i:i+2], 16) / 255.0 - 0.5
            embedding.append(val)
        
        # 填充到目标维度
        while len(embedding) < self.DEFAULT_DIMENSION:
            embedding.append(0.0)
        
        embedding = np.array(embedding[:self.DEFAULT_DIMENSION], dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    # ==========================================================================
    # 相似度计算
    # ==========================================================================
    
    def calculate_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
        metric: str = 'cosine'
    ) -> Dict[str, Any]:
        """计算两个嵌入向量的相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            metric: 相似度计算方法
            
        Returns:
            相似度结果字典
        """
        start_time = time.time()
        
        # 验证参数
        if metric not in self.SUPPORTED_METRICS:
            raise ValidationError(f"不支持的相似度计算方法: {metric}")
        
        # 转换为numpy数组
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)
        
        # 检查维度
        if emb1.shape != emb2.shape:
            raise ValidationError("两个嵌入向量的维度不一致")
        
        # 计算相似度
        try:
            if metric == 'cosine':
                similarity = self._cosine_similarity(emb1, emb2)
            elif metric == 'euclidean':
                similarity = _euclidean_similarity(emb1, emb2)
            elif metric == 'dot_product':
                similarity = _dot_product_similarity(emb1, emb2)
            elif metric == 'manhattan':
                similarity = _manhattan_similarity(emb1, emb2)
            else:
                similarity = self._cosine_similarity(emb1, emb2)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'similarity': float(similarity),
                'metric': metric,
                'processingTimeMs': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            raise BusinessLogicError(f"计算相似度失败: {e}", operation="calculate_similarity")
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """余弦相似度"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def calculate_text_similarity(
        self,
        text1: str,
        text2: str,
        model_type: str = 'sentence-transformers',
        metric: str = 'cosine'
    ) -> Dict[str, Any]:
        """计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            model_type: 模型类型
            metric: 相似度计算方法
            
        Returns:
            相似度结果字典
        """
        start_time = time.time()
        
        # 生成嵌入向量
        result1 = self.generate_embedding(text1, model_type)
        result2 = self.generate_embedding(text2, model_type)
        
        # 计算相似度
        similarity_result = self.calculate_similarity(
            result1['embedding'],
            result2['embedding'],
            metric
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'text1Preview': text1[:100] + '...' if len(text1) > 100 else text1,
            'text2Preview': text2[:100] + '...' if len(text2) > 100 else text2,
            'similarity': similarity_result['similarity'],
            'metric': metric,
            'modelType': model_type,
            'processingTimeMs': processing_time
        }
    
    # ==========================================================================
    # 相似度搜索
    # ==========================================================================
    
    def similarity_search(
        self,
        query: str,
        candidates: List[str],
        model_type: str = 'sentence-transformers',
        top_k: int = 10,
        threshold: Optional[float] = None,
        metric: str = 'cosine',
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """相似度搜索
        
        在候选文本列表中搜索与查询最相似的文本
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            model_type: 模型类型
            top_k: 返回数量
            threshold: 相似度阈值
            metric: 相似度计算方法
            user_id: 用户ID
            
        Returns:
            搜索结果字典
        """
        start_time = time.time()
        
        # 验证参数
        self._validate_text(query)
        if not candidates:
            raise ValidationError("候选文本列表不能为空")
        if top_k < 1:
            raise ValidationError("top_k必须大于0")
        
        # 生成查询嵌入
        query_result = self.generate_embedding(query, model_type, user_id)
        query_embedding = np.array(query_result['embedding'], dtype=np.float32)
        
        # 批量生成候选嵌入
        batch_result = self.generate_batch_embeddings(
            candidates, model_type, user_id, parallel=True
        )
        candidate_embeddings = [np.array(e, dtype=np.float32) for e in batch_result['embeddings']]
        
        # 计算相似度
        similarities = []
        for i, emb in enumerate(candidate_embeddings):
            sim_result = self.calculate_similarity(query_embedding, emb, metric)
            similarities.append({
                'index': i,
                'text': candidates[i],
                'similarity': sim_result['similarity']
            })
        
        # 排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 应用阈值过滤
        if threshold is not None:
            similarities = [s for s in similarities if s['similarity'] >= threshold]
        
        # 取top_k
        results = similarities[:top_k]
        
        search_time = (time.time() - start_time) * 1000
        
        # 记录搜索
        from backend.schemas.embedding_models import compute_content_hash, truncate_text
        if self._repository and user_id:
            self._repository.create_search_record(
                user_id=user_id,
                query_hash=compute_content_hash(query, model_type),
                query_preview=truncate_text(query, 100),
                model_type=model_type,
                similarity_metric=metric,
                top_k=top_k,
                results=results,
                search_time_ms=search_time
            )
        
        return {
            'query': query,
            'results': results,
            'totalCount': len(results),
            'searchTimeMs': search_time,
            'modelType': model_type,
            'metric': metric,
            'threshold': threshold
        }
    
    # ==========================================================================
    # 模型管理
    # ==========================================================================
    
    def list_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取支持的嵌入模型列表
        
        Args:
            active_only: 是否只返回激活的模型
            
        Returns:
            模型列表
        """
        if self._repository:
            return self._repository.list_model_configs(active_only)
        
        # 返回默认模型列表
        return [
            {
                'modelType': 'sentence-transformers',
                'modelName': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'maxTokens': 256,
                'isActive': True,
                'isDefault': True,
                'description': 'Sentence Transformers 轻量级模型'
            },
            {
                'modelType': 'tfidf',
                'modelName': 'tfidf-default',
                'dimension': 384,
                'maxTokens': 10000,
                'isActive': True,
                'isDefault': False,
                'description': 'TF-IDF 统计模型'
            },
            {
                'modelType': 'word2vec',
                'modelName': 'word2vec-default',
                'dimension': 384,
                'maxTokens': 10000,
                'isActive': True,
                'isDefault': False,
                'description': 'Word2Vec 词向量模型'
            },
        ]
    
    def get_model_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取模型详情
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型信息
        """
        if self._repository:
            return self._repository.get_model_config(model_type)
        return None
    
    # ==========================================================================
    # 统计和监控
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        if self._repository:
            return self._repository.get_stats()
        
        if self._embedding_manager:
            return self._embedding_manager.get_stats()
        
        return {
            'totalEmbeddings': 0,
            'cacheHits': 0,
            'cacheMisses': 0,
            'cacheHitRate': 0.0,
            'avgProcessingTimeMs': 0.0,
            'cacheSize': 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self._repository:
            self._repository.clear_cache()
        
        if self._embedding_manager:
            self._embedding_manager.clear_cache()
        
        logger.info("Embedding cache cleared")
    
    def get_user_embeddings(
        self,
        user_id: str,
        model_type: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """获取用户的嵌入记录
        
        Args:
            user_id: 用户ID
            model_type: 模型类型过滤
            page: 页码
            limit: 每页数量
            
        Returns:
            嵌入记录列表
        """
        offset = (page - 1) * limit
        
        if self._repository:
            records, total = self._repository.list_user_embeddings(
                user_id, model_type, limit, offset
            )
            return {
                'embeddings': records,
                'total': total,
                'page': page,
                'limit': limit
            }
        
        return {
            'embeddings': [],
            'total': 0,
            'page': page,
            'limit': limit
        }
    
    def get_user_tasks(
        self,
        user_id: str,
        status: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """获取用户的任务列表
        
        Args:
            user_id: 用户ID
            status: 状态过滤
            page: 页码
            limit: 每页数量
            
        Returns:
            任务列表
        """
        offset = (page - 1) * limit
        
        if self._repository:
            tasks, total = self._repository.list_user_tasks(
                user_id, status, limit, offset
            )
            return {
                'tasks': tasks,
                'total': total,
                'page': page,
                'limit': limit
            }
        
        return {
            'tasks': [],
            'total': 0,
            'page': page,
            'limit': limit
        }
    
    def get_task_status(self, task_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            user_id: 用户ID
            
        Returns:
            任务状态
        """
        if self._repository:
            task = self._repository.get_task(task_id)
            if task and task.get('user_id') == user_id:
                return task
        return None
    
    # ==========================================================================
    # 验证方法
    # ==========================================================================
    
    def _validate_text(self, text: str):
        """验证文本参数"""
        if not text:
            raise ValidationError("文本内容不能为空")
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValidationError(f"文本长度不能超过{self.MAX_TEXT_LENGTH}字符")
    
    def _validate_model_type(self, model_type: str):
        """验证模型类型"""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValidationError(
                f"不支持的模型类型: {model_type}，"
                f"支持的类型: {', '.join(self.SUPPORTED_MODELS)}"
            )
    
    def __del__(self):
        """清理资源"""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass


# ==================== 全局单例 ====================

_global_service: Optional[EmbeddingService] = None


def get_embedding_service(use_memory: bool = False) -> EmbeddingService:
    """获取嵌入服务实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        EmbeddingService 实例
    """
    global _global_service
    
    if _global_service is None:
        _global_service = EmbeddingService(use_memory=use_memory)
    
    return _global_service


def reset_embedding_service():
    """重置全局服务实例（用于测试）"""
    global _global_service
    _global_service = None


# ==================== 导出 ====================

__all__ = [
    'EmbeddingService',
    'get_embedding_service',
    'reset_embedding_service',
]
