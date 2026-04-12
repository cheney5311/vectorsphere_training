"""
向量数据库模块

提供与 VectorSphere 向量数据库的集成，支持：
- 语义搜索（Semantic Search）
- 向量相似度搜索（Vector Similarity Search）
- 混合搜索（Hybrid Search）
- 知识库管理

生产级特性：
- 连接池管理
- 自动重试和熔断
- 异步支持
- 批量操作优化
- 健康检查
"""

from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    SearchResult,
    VectorRecord,
    get_vector_store,
    set_vector_store,
    reset_vector_store,
)

__all__ = [
    'VectorStore',
    'VectorStoreConfig',
    'SearchResult',
    'VectorRecord',
    'get_vector_store',
    'set_vector_store',
    'reset_vector_store',
]
