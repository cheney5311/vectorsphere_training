"""
向量数据库存储模块

提供生产级的向量数据库客户端，对接 VectorSphere 向量数据库平台。

功能特性：
- 集合管理（创建、删除、配置）
- 向量 CRUD 操作
- 语义搜索和向量相似度搜索
- 混合搜索（支持元数据过滤）
- 批量操作优化
- 连接池和自动重试
- 熔断器保护
- 健康检查

使用示例：
    from backend.modules.vectordb import get_vector_store
    
    # 获取全局实例
    store = get_vector_store()
    
    # 语义搜索
    results = store.search(
        query="人工智能的应用",
        collection="knowledge_base",
        top_k=5
    )
    
    # 向量相似度搜索
    results = store.similarity_search(
        query_vector=[0.1, 0.2, ...],
        collection="articles",
        k=10,
        filters={"category": "tech"}
    )
"""

import json
import logging
import threading
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os

import numpy as np

logger = logging.getLogger(__name__)


# ==================== 配置和数据模型 ====================

class MetricType(Enum):
    """距离度量类型"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    IP = "ip"  # Inner Product


class IndexType(Enum):
    """索引类型"""
    HNSW = "hnsw"
    IVF = "ivf"
    FLAT = "flat"
    LSH = "lsh"
    PQ = "pq"


class SearchType(Enum):
    """搜索类型"""
    SEMANTIC = "semantic"      # 语义搜索（需要文本转向量）
    VECTOR = "vector"          # 直接向量搜索
    HYBRID = "hybrid"          # 混合搜索
    KEYWORD = "keyword"        # 关键词搜索


@dataclass
class VectorStoreConfig:
    """向量存储配置
    
    Attributes:
        base_url: VectorSphere API 地址
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        connection_pool_size: 连接池大小
        enable_circuit_breaker: 是否启用熔断器
        circuit_breaker_threshold: 熔断阈值（连续失败次数）
        circuit_breaker_timeout: 熔断恢复时间（秒）
        default_collection: 默认集合名称
        default_dimension: 默认向量维度
        default_metric: 默认距离度量
        embedding_model: 嵌入模型类型
        cache_enabled: 是否启用缓存
        cache_ttl: 缓存过期时间（秒）
    """
    base_url: str = "http://localhost:8080"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_pool_size: int = 10
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    default_collection: str = "default"
    default_dimension: int = 384
    default_metric: str = "cosine"
    embedding_model: str = "sentence-transformers"
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    @classmethod
    def from_env(cls) -> 'VectorStoreConfig':
        """从环境变量创建配置"""
        return cls(
            base_url=os.environ.get('VECTORSPHERE_URL', 'http://localhost:8080'),
            timeout=float(os.environ.get('VECTORSPHERE_TIMEOUT', '30.0')),
            max_retries=int(os.environ.get('VECTORSPHERE_MAX_RETRIES', '3')),
            default_collection=os.environ.get('VECTORSPHERE_DEFAULT_COLLECTION', 'default'),
            default_dimension=int(os.environ.get('VECTORSPHERE_DEFAULT_DIMENSION', '384')),
            default_metric=os.environ.get('VECTORSPHERE_DEFAULT_METRIC', 'cosine'),
            embedding_model=os.environ.get('VECTORSPHERE_EMBEDDING_MODEL', 'sentence-transformers'),
        )


@dataclass
class VectorRecord:
    """向量记录
    
    Attributes:
        id: 唯一标识
        vector: 向量数据
        content: 原始文本内容（可选）
        metadata: 元数据字典
    """
    id: str
    vector: List[float]
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "id": self.id,
            "vector": self.vector,
        }
        if self.metadata:
            result["metadata"] = self.metadata
            # 如果有 content，也放入 metadata
            if self.content and "content" not in self.metadata:
                result["metadata"]["content"] = self.content
        elif self.content:
            result["metadata"] = {"content": self.content}
        return result


@dataclass
class SearchResult:
    """搜索结果
    
    Attributes:
        id: 向量 ID
        score: 相似度分数
        distance: 距离值
        content: 文本内容
        metadata: 元数据
        vector: 原始向量（可选）
    """
    id: str
    score: float
    distance: float = 0.0
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class CollectionInfo:
    """集合信息"""
    name: str
    dimension: int
    metric: str
    index_type: str
    description: str = ""
    vector_count: int = 0
    created_at: Optional[str] = None
    index_config: Dict[str, Any] = field(default_factory=dict)


# ==================== 熔断器 ====================

class CircuitBreaker:
    """熔断器实现
    
    状态：
    - CLOSED: 正常状态，请求正常通过
    - OPEN: 熔断状态，请求直接失败
    - HALF_OPEN: 半开状态，允许少量请求测试
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()
    
    def record_success(self):
        """记录成功"""
        with self.lock:
            self.failure_count = 0
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                logger.info("Circuit breaker closed (recovered)")
    
    def record_failure(self):
        """记录失败"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold:
                if self.state != self.OPEN:
                    self.state = self.OPEN
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        with self.lock:
            if self.state == self.CLOSED:
                return True
            
            if self.state == self.OPEN:
                # 检查是否超过熔断时间
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker half-open (testing)")
                    return True
                return False
            
            # HALF_OPEN 状态允许请求
            return True
    
    def reset(self):
        """重置熔断器"""
        with self.lock:
            self.state = self.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0


# ==================== HTTP 客户端 ====================

class HTTPClient:
    """HTTP 客户端封装
    
    支持连接池、重试、熔断等特性
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        ) if config.enable_circuit_breaker else None
        
        # 尝试使用 httpx（更好的异步支持和连接池）
        self._httpx_available = False
        self._client = None
        
        try:
            import httpx
            self._httpx_available = True
            self._client = httpx.Client(
                base_url=config.base_url,
                timeout=config.timeout,
                limits=httpx.Limits(
                    max_connections=config.connection_pool_size,
                    max_keepalive_connections=config.connection_pool_size // 2
                )
            )
            logger.debug("Using httpx client with connection pooling")
        except ImportError:
            logger.info("httpx not available, using urllib")
    
    def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送 HTTP 请求
        
        Args:
            method: HTTP 方法
            path: 请求路径
            data: 请求体数据
            params: 查询参数
            
        Returns:
            响应数据
            
        Raises:
            Exception: 请求失败
        """
        # 熔断检查
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open, request blocked")
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self._do_request(method, path, data, params)
                
                # 记录成功
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                # 记录失败
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # 重试延迟
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise Exception(f"Request failed after {self.config.max_retries} attempts: {last_error}")
    
    def _do_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """执行 HTTP 请求"""
        url = f"{self.config.base_url}{path}"
        
        if self._httpx_available and self._client:
            return self._do_httpx_request(method, url, data, params)
        else:
            return self._do_urllib_request(method, url, data, params)
    
    def _do_httpx_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """使用 httpx 发送请求"""
        import httpx
        
        kwargs = {"params": params}
        
        if data is not None:
            kwargs["json"] = data
        
        response = self._client.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        try:
            return response.json()
        except:
            return {"raw": response.text}
    
    def _do_urllib_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """使用 urllib 发送请求"""
        import urllib.request
        import urllib.parse
        import urllib.error
        
        # 添加查询参数
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"
        
        # 准备请求
        request_data = None
        if data is not None:
            request_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=request_data,
            method=method,
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_data = response.read().decode('utf-8')
                try:
                    return json.loads(response_data)
                except:
                    return {"raw": response_data}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            raise Exception(f"HTTP {e.code}: {error_body}")
    
    def close(self):
        """关闭客户端"""
        if self._client:
            self._client.close()
            self._client = None


# ==================== 向量存储主类 ====================

class VectorStore:
    """向量数据库存储客户端
    
    生产级的向量数据库客户端，提供：
    - 集合管理
    - 向量 CRUD
    - 语义搜索和向量搜索
    - 批量操作
    - 缓存支持
    
    使用示例：
        config = VectorStoreConfig(base_url="http://localhost:8080")
        store = VectorStore(config)
        
        # 创建集合
        store.create_collection("articles", dimension=384)
        
        # 添加向量
        store.upsert("articles", [
            VectorRecord(id="1", vector=[0.1, 0.2, ...], content="文章内容")
        ])
        
        # 搜索
        results = store.search("人工智能", collection="articles", top_k=5)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """初始化向量存储
        
        Args:
            config: 配置对象，不传则使用环境变量配置
        """
        self.config = config or VectorStoreConfig.from_env()
        self._client = HTTPClient(self.config)
        self._embedding_manager = None
        self._embedding_lock = threading.Lock()
        
        # 搜索结果缓存
        self._cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self._cache_lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "total_searches": 0,
            "total_upserts": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
        
        logger.info(f"VectorStore initialized: {self.config.base_url}")
    
    def _get_embedding_manager(self):
        """获取嵌入管理器（懒加载）"""
        if self._embedding_manager is None:
            with self._embedding_lock:
                if self._embedding_manager is None:
                    try:
                        from backend.modules.embeddings.manager import get_embedding_manager
                        self._embedding_manager = get_embedding_manager({
                            'default_model': self.config.embedding_model,
                            'embedding_dim': self.config.default_dimension,
                        })
                        logger.info(f"Embedding manager loaded: {self.config.embedding_model}")
                    except ImportError as e:
                        logger.warning(f"Failed to load embedding manager: {e}")
                        raise
        return self._embedding_manager
    
    def _text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            向量列表
        """
        manager = self._get_embedding_manager()
        embedding = manager.generate_embedding(text, model_type=self.config.embedding_model)
        return embedding.tolist()
    
    def _get_cache_key(self, query: str, collection: str, top_k: int, filters: Optional[Dict]) -> str:
        """生成缓存键"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        content = f"{collection}:{query}:{top_k}:{filter_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """从缓存获取结果"""
        if not self.config.cache_enabled:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                results, timestamp = self._cache[cache_key]
                # 检查过期
                if time.time() - timestamp < self.config.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return results
                else:
                    # 过期，删除
                    del self._cache[cache_key]
        
        self.stats["cache_misses"] += 1
        return None
    
    def _set_cache(self, cache_key: str, results: List[SearchResult]):
        """设置缓存"""
        if not self.config.cache_enabled:
            return
        
        with self._cache_lock:
            self._cache[cache_key] = (results, time.time())
            
            # 缓存清理（保持最大 1000 条）
            if len(self._cache) > 1000:
                # 删除最旧的 100 条
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )
                for key in sorted_keys[:100]:
                    del self._cache[key]
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态信息
        """
        try:
            response = self._client.request("GET", "/health")
            return {
                "status": "healthy",
                "vectorsphere": response,
                "client_stats": self.stats.copy()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_stats": self.stats.copy()
            }
    
    # ==================== 集合管理 ====================
    
    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        index_type: str = "hnsw",
        index_config: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> CollectionInfo:
        """创建集合
        
        Args:
            name: 集合名称
            dimension: 向量维度
            metric: 距离度量（cosine, euclidean, dot）
            index_type: 索引类型（hnsw, ivf, flat）
            index_config: 索引配置
            description: 集合描述
            
        Returns:
            集合信息
        """
        dimension = dimension or self.config.default_dimension
        metric = metric or self.config.default_metric
        index_config = index_config or {}
        
        data = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index_type": index_type,
            "index_config": index_config,
            "description": description
        }
        
        response = self._client.request("POST", "/api/v1/collections", data=data)
        
        logger.info(f"Collection created: {name} (dim={dimension}, metric={metric})")
        
        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            description=description,
            index_config=index_config
        )
    
    def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """获取集合信息
        
        Args:
            name: 集合名称
            
        Returns:
            集合信息，不存在返回 None
        """
        try:
            response = self._client.request("GET", f"/api/v1/collections/{name}")
            
            return CollectionInfo(
                name=response.get("name", name),
                dimension=response.get("dimension", 0),
                metric=response.get("metric", ""),
                index_type=response.get("index_type", ""),
                description=response.get("description", ""),
                vector_count=response.get("vector_count", 0),
                created_at=response.get("created_at"),
                index_config=response.get("index_config", {})
            )
        except Exception as e:
            logger.debug(f"Collection not found: {name}, error: {e}")
            return None
    
    def list_collections(self) -> List[CollectionInfo]:
        """列出所有集合
        
        Returns:
            集合列表
        """
        response = self._client.request("GET", "/api/v1/collections")
        collections = response.get("collections", [])
        
        return [
            CollectionInfo(
                name=c.get("name", ""),
                dimension=c.get("dimension", 0),
                metric=c.get("metric", ""),
                index_type=c.get("index_type", ""),
                description=c.get("description", ""),
                vector_count=c.get("vector_count", 0),
            )
            for c in collections
        ]
    
    def delete_collection(self, name: str) -> bool:
        """删除集合
        
        Args:
            name: 集合名称
            
        Returns:
            是否成功
        """
        try:
            self._client.request("DELETE", f"/api/v1/collections/{name}")
            logger.info(f"Collection deleted: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """检查集合是否存在
        
        Args:
            name: 集合名称
            
        Returns:
            是否存在
        """
        return self.get_collection(name) is not None
    
    def ensure_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        **kwargs
    ) -> CollectionInfo:
        """确保集合存在（不存在则创建）
        
        Args:
            name: 集合名称
            dimension: 向量维度
            **kwargs: 其他创建参数
            
        Returns:
            集合信息
        """
        info = self.get_collection(name)
        if info:
            return info
        return self.create_collection(name, dimension=dimension, **kwargs)
    
    # ==================== 向量操作 ====================
    
    def upsert(
        self,
        collection: str,
        records: List[Union[VectorRecord, Dict[str, Any]]],
        auto_create_collection: bool = True
    ) -> Dict[str, Any]:
        """插入或更新向量
        
        Args:
            collection: 集合名称
            records: 向量记录列表
            auto_create_collection: 集合不存在时是否自动创建
            
        Returns:
            操作结果
        """
        # 确保集合存在
        if auto_create_collection:
            self.ensure_collection(collection)
        
        # 转换记录格式
        formatted_records = []
        for record in records:
            if isinstance(record, VectorRecord):
                formatted_records.append(record.to_dict())
            elif isinstance(record, dict):
                formatted_records.append(record)
            else:
                raise ValueError(f"Invalid record type: {type(record)}")
        
        data = {"records": formatted_records}
        
        response = self._client.request(
            "POST",
            f"/api/v1/collections/{collection}/vectors",
            data=data
        )
        
        self.stats["total_upserts"] += len(records)
        logger.debug(f"Upserted {len(records)} vectors to {collection}")
        
        return response
    
    def upsert_texts(
        self,
        collection: str,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """插入文本（自动向量化）
        
        Args:
            collection: 集合名称
            texts: 文本列表
            ids: ID 列表（可选，自动生成）
            metadatas: 元数据列表
            batch_size: 批处理大小
            
        Returns:
            操作结果
        """
        if not texts:
            return {"message": "No texts to upsert"}
        
        # 生成 ID
        if ids is None:
            ids = [
                hashlib.md5(f"{text}_{i}".encode()).hexdigest()[:16]
                for i, text in enumerate(texts)
            ]
        
        # 默认元数据
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 批量生成向量
        manager = self._get_embedding_manager()
        
        total_upserted = 0
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # 批量向量化
            vectors = manager.generate_batch_embeddings(
                batch_texts,
                model_type=self.config.embedding_model,
                batch_size=batch_size
            )
            
            # 构建记录
            records = []
            for j, (text, vid, vec, meta) in enumerate(zip(batch_texts, batch_ids, vectors, batch_metadatas)):
                meta_copy = meta.copy()
                meta_copy["content"] = text
                meta_copy["created_at"] = datetime.utcnow().isoformat()
                
                records.append(VectorRecord(
                    id=vid,
                    vector=vec.tolist(),
                    content=text,
                    metadata=meta_copy
                ))
            
            # 插入
            result = self.upsert(collection, records)
            results.append(result)
            total_upserted += len(records)
        
        return {
            "message": "upsert successful",
            "total": total_upserted,
            "batches": len(results)
        }
    
    def get(
        self,
        collection: str,
        ids: List[str]
    ) -> List[VectorRecord]:
        """获取向量
        
        Args:
            collection: 集合名称
            ids: ID 列表
            
        Returns:
            向量记录列表
        """
        response = self._client.request(
            "GET",
            f"/api/v1/collections/{collection}/vectors",
            params={"ids": ",".join(ids)}
        )
        
        vectors = response.get("vectors", [])
        
        return [
            VectorRecord(
                id=v.get("id", ""),
                vector=v.get("vector", []),
                content=v.get("metadata", {}).get("content", ""),
                metadata=v.get("metadata", {})
            )
            for v in vectors
        ]
    
    def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """删除向量
        
        Args:
            collection: 集合名称
            ids: 要删除的 ID 列表
            filters: 过滤条件
            
        Returns:
            删除结果
        """
        data = {}
        if ids:
            data["ids"] = ids
        if filters:
            data["filter"] = self._build_filter(filters)
        
        response = self._client.request(
            "DELETE",
            f"/api/v1/collections/{collection}/vectors",
            data=data if data else None
        )
        
        logger.debug(f"Deleted vectors from {collection}")
        return response
    
    # ==================== 搜索功能 ====================
    
    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        include_metadata: bool = True,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """语义搜索
        
        将查询文本转换为向量后进行相似度搜索。
        
        Args:
            query: 搜索查询文本
            collection: 集合名称
            top_k: 返回结果数量
            filters: 元数据过滤条件
            search_type: 搜索类型（semantic, hybrid）
            include_metadata: 是否包含元数据
            score_threshold: 最小分数阈值
            
        Returns:
            搜索结果列表
        """
        collection = collection or self.config.default_collection
        
        # 检查缓存
        cache_key = self._get_cache_key(query, collection, top_k, filters)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # 将查询文本转换为向量
            query_vector = self._text_to_vector(query)
            
            # 执行向量搜索
            results = self.similarity_search(
                query_vector=query_vector,
                collection=collection,
                k=top_k,
                filters=filters,
                include_metadata=include_metadata,
                score_threshold=score_threshold
            )
            
            # 更新统计
            self.stats["total_searches"] += 1
            
            # 缓存结果
            self._set_cache(cache_key, results)
            
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Search failed: {e}")
            raise
    
    def similarity_search(
        self,
        query_vector: Optional[List[float]] = None,
        query: Optional[str] = None,
        collection: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """向量相似度搜索
        
        直接使用向量进行相似度搜索。
        
        Args:
            query_vector: 查询向量
            query: 查询文本（如果没有 query_vector，则使用文本转向量）
            collection: 集合名称
            k: 返回结果数量
            filters: 元数据过滤条件
            include_metadata: 是否包含元数据
            score_threshold: 最小分数阈值
            
        Returns:
            搜索结果列表
        """
        collection = collection or self.config.default_collection
        
        # 如果没有向量，从文本生成
        if query_vector is None:
            if query is None:
                raise ValueError("Must provide either query_vector or query")
            query_vector = self._text_to_vector(query)
        
        # 构建请求
        data = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": include_metadata
        }
        
        # 添加过滤条件
        if filters:
            data["filter"] = self._build_filter(filters)
        
        # 执行搜索
        response = self._client.request(
            "POST",
            f"/api/v1/collections/{collection}/vectors/query",
            data=data
        )
        
        # 解析结果
        results = self._parse_search_response(response, score_threshold)
        
        self.stats["total_searches"] += 1
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        collection: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """批量语义搜索
        
        Args:
            queries: 查询文本列表
            collection: 集合名称
            top_k: 每个查询返回结果数量
            filters: 元数据过滤条件
            
        Returns:
            搜索结果列表的列表
        """
        collection = collection or self.config.default_collection
        
        # 批量向量化
        manager = self._get_embedding_manager()
        vectors = manager.generate_batch_embeddings(
            queries,
            model_type=self.config.embedding_model
        )
        
        # 构建批量请求
        data = {
            "vectors": [v.tolist() for v in vectors],
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filters:
            data["filter"] = self._build_filter(filters)
        
        # 执行批量搜索
        response = self._client.request(
            "POST",
            f"/api/v1/collections/{collection}/vectors/query/batch",
            data=data
        )
        
        # 解析批量结果
        all_results = []
        data_results = response.get("data", {}).get("results", [])
        
        for result_group in data_results:
            results = self._parse_single_result_group(result_group)
            all_results.append(results)
        
        self.stats["total_searches"] += len(queries)
        
        return all_results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """构建过滤条件
        
        支持的过滤格式：
        - 简单等值: {"field": "value"}
        - 复杂条件: {"field": {"$gt": 10, "$lt": 100}}
        - 逻辑组合: {"$and": [...], "$or": [...]}
        
        Args:
            filters: 过滤条件字典
            
        Returns:
            VectorSphere 格式的过滤条件
        """
        conditions = []
        
        for key, value in filters.items():
            if key.startswith("$"):
                # 逻辑操作符
                if key == "$and":
                    return {
                        "op": "AND",
                        "conditions": [
                            self._build_single_condition(k, v)
                            for item in value
                            for k, v in item.items()
                        ]
                    }
                elif key == "$or":
                    return {
                        "op": "OR",
                        "conditions": [
                            self._build_single_condition(k, v)
                            for item in value
                            for k, v in item.items()
                        ]
                    }
            else:
                conditions.append(self._build_single_condition(key, value))
        
        if len(conditions) == 1:
            return {"op": "AND", "conditions": conditions}
        elif len(conditions) > 1:
            return {"op": "AND", "conditions": conditions}
        
        return {}
    
    def _build_single_condition(self, field: str, value: Any) -> Dict[str, Any]:
        """构建单个过滤条件"""
        if isinstance(value, dict):
            # 复杂条件
            for op, val in value.items():
                operator_map = {
                    "$eq": "eq",
                    "$ne": "ne",
                    "$gt": "gt",
                    "$gte": "gte",
                    "$lt": "lt",
                    "$lte": "lte",
                    "$in": "in",
                    "$nin": "nin",
                    "$contains": "contains",
                    "$prefix": "prefix",
                    "$between": "between",
                }
                mapped_op = operator_map.get(op, op.replace("$", ""))
                return {"field": field, "operator": mapped_op, "value": val}
        else:
            # 简单等值
            return {"field": field, "operator": "eq", "value": value}
    
    def _parse_search_response(
        self,
        response: Dict[str, Any],
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """解析搜索响应
        
        Args:
            response: API 响应
            score_threshold: 分数阈值
            
        Returns:
            搜索结果列表
        """
        results = []
        
        # 处理响应格式
        data = response.get("data", response)
        result_groups = data.get("results", [])
        
        for group in result_groups:
            results.extend(self._parse_single_result_group(group, score_threshold))
        
        return results
    
    def _parse_single_result_group(
        self,
        group: Dict[str, Any],
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """解析单个结果组"""
        results = []
        
        ids = group.get("ids", [])
        distances = group.get("distances", [])
        fields = group.get("fields", [])
        
        for i, vid in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            metadata = fields[i] if i < len(fields) else {}
            
            # 计算相似度分数（距离越小越相似）
            # 对于 cosine 距离，score = 1 - distance
            score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
            
            if score < score_threshold:
                continue
            
            content = metadata.get("content", "") if metadata else ""
            
            results.append(SearchResult(
                id=vid,
                score=round(score, 4),
                distance=round(distance, 4),
                content=content,
                metadata=metadata or {}
            ))
        
        return results
    
    # ==================== 辅助方法 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["cache_size"] = len(self._cache)
        
        if self._client.circuit_breaker:
            cb = self._client.circuit_breaker
            stats["circuit_breaker"] = {
                "state": cb.state,
                "failure_count": cb.failure_count
            }
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Search cache cleared")
    
    def close(self):
        """关闭连接"""
        self._client.close()
        logger.info("VectorStore connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 全局实例管理 ====================

_vector_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """获取全局向量存储实例
    
    Args:
        config: 配置对象（仅首次调用时有效）
        
    Returns:
        VectorStore 实例
    """
    global _vector_store
    
    with _store_lock:
        if _vector_store is None:
            try:
                _vector_store = VectorStore(config)
            except Exception as e:
                logger.error(f"Failed to create VectorStore: {e}")
                raise
    
    return _vector_store


def set_vector_store(store: VectorStore):
    """设置全局向量存储实例
    
    Args:
        store: VectorStore 实例
    """
    global _vector_store
    
    with _store_lock:
        _vector_store = store


def reset_vector_store():
    """重置全局向量存储实例（用于测试）"""
    global _vector_store
    
    with _store_lock:
        if _vector_store:
            _vector_store.close()
        _vector_store = None


# ==================== 导出 ====================

__all__ = [
    'VectorStore',
    'VectorStoreConfig',
    'VectorRecord',
    'SearchResult',
    'CollectionInfo',
    'MetricType',
    'IndexType',
    'SearchType',
    'get_vector_store',
    'set_vector_store',
    'reset_vector_store',
]
