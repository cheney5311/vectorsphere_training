#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入API接口

提供嵌入向量生成和管理的REST API接口。

支持功能：
- 单个文本嵌入生成
- 批量文本嵌入生成
- 向量相似度计算
- 文本相似度计算
- 相似度搜索
- 嵌入缓存管理
- 模型管理
- 统计信息

架构调用关系：
API层 (本模块)
    -> Service层 (EmbeddingService)
        -> Repository层 (EmbeddingRepository)
        -> EmbeddingManager
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.utils.response import success_response, error_response
from backend.core.exceptions import ValidationError, BusinessLogicError

# 创建蓝图
embeddings_bp = Blueprint('embeddings', __name__, url_prefix='/api/v1/embeddings')
logger = logging.getLogger(__name__)

# ==================== 服务初始化 ====================

_embedding_service = None


def get_embedding_service():
    """获取嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        try:
            from backend.services.embedding_service import get_embedding_service as get_svc
            _embedding_service = get_svc()
            logger.info("EmbeddingsAPI: EmbeddingService initialized")
        except Exception as e:
            logger.warning(f"EmbeddingsAPI: Failed to init service: {e}")
    return _embedding_service


def init_embeddings_api(app=None):
    """初始化嵌入API模块
    
    Args:
        app: Flask应用实例
    """
    global _embedding_service
    _embedding_service = get_embedding_service()
    
    if app:
        app.register_blueprint(embeddings_bp)
        logger.info("EmbeddingsAPI: Blueprint registered")


# ==================== 嵌入生成接口 ====================

@embeddings_bp.route('/generate', methods=['POST'])
@jwt_required()
def generate_embedding():
    """生成文本嵌入向量
    
    Request Body:
        {
            "text": "string",
            "model_type": "string",  // 可选: sentence-transformers, bge, m3e, tfidf, word2vec, default
            "use_cache": boolean,    // 可选: 是否使用缓存 (默认: true)
            "metadata": {}           // 可选: 元数据
        }
    
    Returns:
        {
            "embedding": [float],
            "dimension": integer,
            "modelType": "string",
            "processingTimeMs": float,
            "tokenCount": integer,
            "cached": boolean
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 获取参数
        text = data.get('text')
        if not text:
            return error_response("缺少必需字段: text", 400)
        
        model_type = data.get('model_type', 'sentence-transformers')
        use_cache = data.get('use_cache', True)
        metadata = data.get('metadata', {})
        
        # 调用服务层
        service = get_embedding_service()
        if service:
            result = service.generate_embedding(
                text=text,
                model_type=model_type,
                user_id=user_id,
                metadata=metadata,
                use_cache=use_cache
            )
            return success_response(result, "嵌入向量生成成功")
        
        # 降级到原有管理器
        from backend.modules.embeddings.manager import get_embedding_manager
        embedding_manager = get_embedding_manager()
        embedding = embedding_manager.generate_embedding(text, model_type)
        
        return success_response({
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "modelType": model_type,
            "cached": False
        }, "嵌入向量生成成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"生成嵌入向量失败: {str(e)}")
        return error_response(f"生成嵌入向量失败: {str(e)}", 500)


@embeddings_bp.route('/batch-generate', methods=['POST'])
@jwt_required()
def generate_batch_embeddings():
    """批量生成文本嵌入向量
    
    Request Body:
        {
            "texts": ["string"],
            "model_type": "string",  // 可选: sentence-transformers, bge, m3e, tfidf, word2vec
            "use_cache": boolean,    // 可选: 是否使用缓存 (默认: true)
            "parallel": boolean      // 可选: 是否并行处理 (默认: true)
        }
    
    Returns:
        {
            "embeddings": [[float]],
            "dimension": integer,
            "modelType": "string",
            "totalCount": integer,
            "successCount": integer,
            "failedCount": integer,
            "processingTimeMs": float,
            "totalTokens": integer,
            "taskId": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 获取参数
        texts = data.get('texts')
        if not texts or not isinstance(texts, list):
            return error_response("缺少必需字段: texts 或 texts 不是列表", 400)
        
        model_type = data.get('model_type', 'sentence-transformers')
        use_cache = data.get('use_cache', True)
        parallel = data.get('parallel', True)
        
        # 调用服务层
        service = get_embedding_service()
        if service:
            result = service.generate_batch_embeddings(
                texts=texts,
                model_type=model_type,
                user_id=user_id,
                use_cache=use_cache,
                parallel=parallel
            )
            return success_response(result, "批量嵌入向量生成成功")
        
        # 降级到原有管理器
        from backend.modules.embeddings.manager import get_embedding_manager
        embedding_manager = get_embedding_manager()
        embeddings = embedding_manager.generate_batch_embeddings(texts, model_type)
        
        return success_response({
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
            "modelType": model_type,
            "totalCount": len(texts),
            "successCount": len(texts),
            "failedCount": 0
        }, "批量嵌入向量生成成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"批量生成嵌入向量失败: {str(e)}")
        return error_response(f"批量生成嵌入向量失败: {str(e)}", 500)


# ==================== 相似度计算接口 ====================

@embeddings_bp.route('/similarity', methods=['POST'])
@jwt_required()
def calculate_similarity():
    """计算两个嵌入向量的相似度
    
    Request Body:
        {
            "embedding1": [float],
            "embedding2": [float],
            "metric": "string"  // 可选: cosine, euclidean, dot_product, manhattan (默认: cosine)
        }
    
    Returns:
        {
            "similarity": float,
            "metric": "string",
            "processingTimeMs": float
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 获取参数
        embedding1 = data.get('embedding1')
        embedding2 = data.get('embedding2')
        
        if not embedding1 or not embedding2:
            return error_response("缺少必需字段: embedding1 或 embedding2", 400)
        
        if not isinstance(embedding1, list) or not isinstance(embedding2, list):
            return error_response("embedding1 和 embedding2 必须是列表", 400)
        
        metric = data.get('metric', 'cosine')
        
        # 调用服务层
        service = get_embedding_service()
        if service:
            result = service.calculate_similarity(
                embedding1=embedding1,
                embedding2=embedding2,
                metric=metric
            )
            return success_response(result, "相似度计算成功")
        
        # 降级到原有管理器
        from backend.modules.embeddings.manager import get_embedding_manager
        embedding_manager = get_embedding_manager()
        
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)
        
        if emb1.shape != emb2.shape:
            return error_response("两个嵌入向量的维度不一致", 400)
        
        similarity = embedding_manager.calculate_similarity(emb1, emb2)
        
        return success_response({
            "similarity": similarity,
            "metric": "cosine"
        }, "相似度计算成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"计算相似度失败: {str(e)}")
        return error_response(f"计算相似度失败: {str(e)}", 500)


@embeddings_bp.route('/text-similarity', methods=['POST'])
@jwt_required()
def calculate_text_similarity():
    """计算两个文本的相似度
    
    Request Body:
        {
            "text1": "string",
            "text2": "string",
            "model_type": "string",  // 可选: sentence-transformers等 (默认: sentence-transformers)
            "metric": "string"       // 可选: cosine等 (默认: cosine)
        }
    
    Returns:
        {
            "text1Preview": "string",
            "text2Preview": "string",
            "similarity": float,
            "metric": "string",
            "modelType": "string",
            "processingTimeMs": float
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        if not text1 or not text2:
            return error_response("缺少必需字段: text1 或 text2", 400)
        
        model_type = data.get('model_type', 'sentence-transformers')
        metric = data.get('metric', 'cosine')
        
        # 调用服务层
        service = get_embedding_service()
        if service:
            result = service.calculate_text_similarity(
                text1=text1,
                text2=text2,
                model_type=model_type,
                metric=metric
            )
            return success_response(result, "文本相似度计算成功")
        
        return error_response("嵌入服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"计算文本相似度失败: {str(e)}")
        return error_response(f"计算文本相似度失败: {str(e)}", 500)


# ==================== 相似度搜索接口 ====================

@embeddings_bp.route('/search', methods=['POST'])
@jwt_required()
def similarity_search():
    """相似度搜索
    
    在候选文本列表中搜索与查询最相似的文本
    
    Request Body:
        {
            "query": "string",
            "candidates": ["string"],
            "model_type": "string",    // 可选 (默认: sentence-transformers)
            "top_k": integer,          // 可选: 返回数量 (默认: 10)
            "threshold": float,        // 可选: 相似度阈值
            "metric": "string"         // 可选 (默认: cosine)
        }
    
    Returns:
        {
            "query": "string",
            "results": [
                {
                    "index": integer,
                    "text": "string",
                    "similarity": float
                }
            ],
            "totalCount": integer,
            "searchTimeMs": float,
            "modelType": "string",
            "metric": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        query = data.get('query')
        candidates = data.get('candidates')
        
        if not query:
            return error_response("缺少必需字段: query", 400)
        if not candidates or not isinstance(candidates, list):
            return error_response("缺少必需字段: candidates 或格式错误", 400)
        
        model_type = data.get('model_type', 'sentence-transformers')
        top_k = data.get('top_k', 10)
        threshold = data.get('threshold')
        metric = data.get('metric', 'cosine')
        
        # 调用服务层
        service = get_embedding_service()
        if service:
            result = service.similarity_search(
                query=query,
                candidates=candidates,
                model_type=model_type,
                top_k=top_k,
                threshold=threshold,
                metric=metric,
                user_id=user_id
            )
            return success_response(result, "相似度搜索完成")
        
        return error_response("嵌入服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"相似度搜索失败: {str(e)}")
        return error_response(f"相似度搜索失败: {str(e)}", 500)


# ==================== 统计信息接口 ====================

@embeddings_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_embedding_stats():
    """获取嵌入管理器统计信息
    
    Returns:
        {
            "stats": {
                "totalEmbeddings": integer,
                "totalTokens": integer,
                "cacheHits": integer,
                "cacheMisses": integer,
                "cacheHitRate": float,
                "avgProcessingTimeMs": float,
                "cacheSize": integer,
                "modelsUsed": {}
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_embedding_service()
        if service:
            stats = service.get_stats()
            return success_response({"stats": stats}, "获取统计信息成功")
        
        # 降级到原有管理器
        from backend.modules.embeddings.manager import get_embedding_manager
        embedding_manager = get_embedding_manager()
        stats = embedding_manager.get_stats()
        
        return success_response({"stats": stats}, "获取统计信息成功")
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return error_response(f"获取统计信息失败: {str(e)}", 500)


# ==================== 缓存管理接口 ====================

@embeddings_bp.route('/cache/clear', methods=['POST'])
@jwt_required()
def clear_embedding_cache():
    """清空嵌入缓存
    
    Returns:
        {
            "message": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_embedding_service()
        if service:
            service.clear_cache()
            return success_response(None, "嵌入缓存清空成功")
        
        # 降级到原有管理器
        from backend.modules.embeddings.manager import get_embedding_manager
        embedding_manager = get_embedding_manager()
        embedding_manager.clear_cache()
        
        return success_response(None, "嵌入缓存清空成功")
        
    except Exception as e:
        logger.error(f"清空嵌入缓存失败: {str(e)}")
        return error_response(f"清空嵌入缓存失败: {str(e)}", 500)


@embeddings_bp.route('/cache/stats', methods=['GET'])
@jwt_required()
def get_cache_stats():
    """获取缓存统计信息
    
    Returns:
        {
            "size": integer,
            "hits": integer,
            "misses": integer,
            "hitRate": float
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_embedding_service()
        if service and service._repository:
            stats = service._repository.get_cache_stats()
            return success_response(stats, "获取缓存统计成功")
        
        return success_response({
            'size': 0,
            'hits': 0,
            'misses': 0,
            'hitRate': 0.0
        }, "获取缓存统计成功")
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {str(e)}")
        return error_response(f"获取缓存统计失败: {str(e)}", 500)


# ==================== 模型管理接口 ====================

@embeddings_bp.route('/models', methods=['GET'])
@jwt_required()
def list_embedding_models():
    """获取支持的嵌入模型列表
    
    Query Parameters:
        - active_only: boolean (默认: true) - 是否只返回激活的模型
    
    Returns:
        {
            "models": [
                {
                    "modelType": "string",
                    "modelName": "string",
                    "dimension": integer,
                    "maxTokens": integer,
                    "isActive": boolean,
                    "isDefault": boolean,
                    "description": "string"
                }
            ]
        }
    """
    try:
        user_id = get_jwt_identity()
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        
        service = get_embedding_service()
        if service:
            models = service.list_models(active_only)
            return success_response({"models": models}, "获取模型列表成功")
        
        # 返回默认列表
        models = [
            {
                "modelType": "sentence-transformers",
                "modelName": "all-MiniLM-L6-v2",
                "dimension": 384,
                "maxTokens": 256,
                "isActive": True,
                "isDefault": True,
                "description": "Sentence Transformers 轻量级模型"
            },
            {
                "modelType": "bge",
                "modelName": "bge-large-zh-v1.5",
                "dimension": 1024,
                "maxTokens": 512,
                "isActive": True,
                "isDefault": False,
                "description": "BGE 中文大模型"
            },
            {
                "modelType": "m3e",
                "modelName": "m3e-base",
                "dimension": 768,
                "maxTokens": 512,
                "isActive": True,
                "isDefault": False,
                "description": "M3E 中文通用模型"
            },
            {
                "modelType": "tfidf",
                "modelName": "tfidf-default",
                "dimension": 384,
                "maxTokens": 10000,
                "isActive": True,
                "isDefault": False,
                "description": "TF-IDF 统计模型"
            },
            {
                "modelType": "word2vec",
                "modelName": "word2vec-default",
                "dimension": 384,
                "maxTokens": 10000,
                "isActive": True,
                "isDefault": False,
                "description": "Word2Vec 词向量模型"
            }
        ]
        
        return success_response({"models": models}, "获取模型列表成功")
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return error_response(f"获取模型列表失败: {str(e)}", 500)


@embeddings_bp.route('/models/<model_type>', methods=['GET'])
@jwt_required()
def get_model_info(model_type: str):
    """获取指定模型的详细信息
    
    Path Parameters:
        - model_type: 模型类型
    
    Returns:
        {
            "modelType": "string",
            "modelName": "string",
            "dimension": integer,
            "maxTokens": integer,
            "isActive": boolean,
            "isDefault": boolean,
            "totalRequests": integer,
            "totalTokens": integer,
            "avgLatencyMs": float,
            "config": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_embedding_service()
        if service:
            model_info = service.get_model_info(model_type)
            if model_info:
                return success_response(model_info, "获取模型信息成功")
            return error_response(f"模型 {model_type} 不存在", 404)
        
        return error_response("嵌入服务不可用", 503)
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return error_response(f"获取模型信息失败: {str(e)}", 500)


# ==================== 用户记录接口 ====================

@embeddings_bp.route('/records', methods=['GET'])
@jwt_required()
def get_user_embeddings():
    """获取用户的嵌入记录列表
    
    Query Parameters:
        - model_type: string (可选) - 模型类型过滤
        - page: integer (默认: 1) - 页码
        - limit: integer (默认: 20) - 每页数量
    
    Returns:
        {
            "embeddings": [...],
            "total": integer,
            "page": integer,
            "limit": integer
        }
    """
    try:
        user_id = get_jwt_identity()
        
        model_type = request.args.get('model_type')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = get_embedding_service()
        if service:
            result = service.get_user_embeddings(
                user_id=user_id,
                model_type=model_type,
                page=page,
                limit=limit
            )
            return success_response(result, "获取嵌入记录成功")
        
        return success_response({
            'embeddings': [],
            'total': 0,
            'page': page,
            'limit': limit
        }, "获取嵌入记录成功")
        
    except Exception as e:
        logger.error(f"获取嵌入记录失败: {str(e)}")
        return error_response(f"获取嵌入记录失败: {str(e)}", 500)


# ==================== 任务管理接口 ====================

@embeddings_bp.route('/tasks', methods=['GET'])
@jwt_required()
def get_user_tasks():
    """获取用户的嵌入任务列表
    
    Query Parameters:
        - status: string (可选) - 状态过滤 (pending, processing, completed, failed)
        - page: integer (默认: 1) - 页码
        - limit: integer (默认: 20) - 每页数量
    
    Returns:
        {
            "tasks": [...],
            "total": integer,
            "page": integer,
            "limit": integer
        }
    """
    try:
        user_id = get_jwt_identity()
        
        status = request.args.get('status')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = get_embedding_service()
        if service:
            result = service.get_user_tasks(
                user_id=user_id,
                status=status,
                page=page,
                limit=limit
            )
            return success_response(result, "获取任务列表成功")
        
        return success_response({
            'tasks': [],
            'total': 0,
            'page': page,
            'limit': limit
        }, "获取任务列表成功")
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        return error_response(f"获取任务列表失败: {str(e)}", 500)


@embeddings_bp.route('/tasks/<task_id>', methods=['GET'])
@jwt_required()
def get_task_status(task_id: str):
    """获取指定任务的状态
    
    Path Parameters:
        - task_id: 任务ID
    
    Returns:
        {
            "id": "string",
            "status": "string",
            "progress": float,
            "processedItems": integer,
            "totalItems": integer,
            ...
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_embedding_service()
        if service:
            task = service.get_task_status(task_id, user_id)
            if task:
                return success_response(task, "获取任务状态成功")
            return error_response("任务不存在", 404)
        
        return error_response("嵌入服务不可用", 503)
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        return error_response(f"获取任务状态失败: {str(e)}", 500)


# ==================== 健康检查接口 ====================

@embeddings_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    Returns:
        {
            "status": "healthy",
            "service": "embeddings",
            "components": {
                "embedding_service": boolean,
                "embedding_manager": boolean,
                "repository": boolean
            }
        }
    """
    try:
        service = get_embedding_service()
        
        components = {
            'embedding_service': service is not None,
            'embedding_manager': service._embedding_manager is not None if service else False,
            'repository': service._repository is not None if service else False
        }
        
        all_healthy = all(components.values())
        
        return success_response({
            'status': 'healthy' if all_healthy else 'degraded',
            'service': 'embeddings',
            'components': components
        }, "健康检查完成")
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return error_response(f"健康检查失败: {str(e)}", 500)


# ==================== 导出 ====================

__all__ = [
    'embeddings_bp',
    'init_embeddings_api',
    'get_embedding_service',
]
