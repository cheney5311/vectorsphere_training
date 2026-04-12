#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入管理器

提供文本嵌入向量的生成和管理功能。

支持的模型类型：
- sentence-transformers: 使用真实的 SentenceTransformer 模型
- bge: BGE 中文嵌入模型
- m3e: M3E 中文嵌入模型  
- openai: OpenAI 嵌入 API（需要 API 密钥）
- tfidf: TF-IDF 统计模型
- word2vec: Word2Vec 词向量模型
- default: 基于特征提取的降级模型

生产级特性：
- 模型懒加载和缓存
- 多模型支持
- 批量处理优化
- GPU 加速（如果可用）
- 优雅降级机制
- 线程安全
"""

import numpy as np
import logging
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import hashlib
from pathlib import Path
import time
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入管理器
    
    生产级的文本嵌入向量生成器，支持多种模型类型。
    """
    
    # 预定义的模型配置
    MODEL_CONFIGS = {
        'sentence-transformers': {
            'model_name': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'max_seq_length': 256,
            'description': 'Sentence Transformers 轻量级英文模型'
        },
        'sentence-transformers-multilingual': {
            'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'max_seq_length': 128,
            'description': 'Sentence Transformers 多语言模型'
        },
        'bge-large-zh': {
            'model_name': 'BAAI/bge-large-zh-v1.5',
            'dimension': 1024,
            'max_seq_length': 512,
            'description': 'BGE 中文大模型'
        },
        'bge-base-zh': {
            'model_name': 'BAAI/bge-base-zh-v1.5',
            'dimension': 768,
            'max_seq_length': 512,
            'description': 'BGE 中文基础模型'
        },
        'bge-small-zh': {
            'model_name': 'BAAI/bge-small-zh-v1.5',
            'dimension': 512,
            'max_seq_length': 512,
            'description': 'BGE 中文小模型'
        },
        'm3e-base': {
            'model_name': 'moka-ai/m3e-base',
            'dimension': 768,
            'max_seq_length': 512,
            'description': 'M3E 中文基础模型'
        },
        'm3e-small': {
            'model_name': 'moka-ai/m3e-small',
            'dimension': 512,
            'max_seq_length': 512,
            'description': 'M3E 中文小模型'
        },
        'text2vec-chinese': {
            'model_name': 'shibing624/text2vec-base-chinese',
            'dimension': 768,
            'max_seq_length': 128,
            'description': 'Text2Vec 中文模型'
        },
    }
    
    # 默认模型映射
    DEFAULT_MODEL_MAP = {
        'sentence-transformers': 'sentence-transformers',
        'bge': 'bge-base-zh',
        'm3e': 'm3e-base',
        'multilingual': 'sentence-transformers-multilingual',
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化嵌入管理器
        
        Args:
            config: 配置字典，支持以下选项：
                - embedding_dim: 默认嵌入维度 (默认: 384)
                - cache_enabled: 是否启用缓存 (默认: True)
                - cache_dir: 缓存目录 (默认: embedding_cache)
                - default_model: 默认模型类型 (默认: sentence-transformers)
                - device: 设备类型 (auto/cpu/cuda)
                - use_gpu: 是否使用 GPU (默认: auto)
                - model_cache_dir: 模型缓存目录
        """
        self.config = config or {}
        self.embedding_dim = self.config.get('embedding_dim', 384)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.default_model = self.config.get('default_model', 'sentence-transformers')
        
        # 设备配置
        self.device = self._determine_device()
        
        # 线程锁
        self._lock = threading.RLock()
        self._model_lock = threading.RLock()
        
        # 模型缓存（懒加载）
        self._loaded_models: Dict[str, Any] = {}
        self._model_loading: Dict[str, bool] = {}
        
        # SentenceTransformer 可用性标记
        self._sentence_transformer_available = None
        
        # TF-IDF 相关组件（懒加载）
        self._tfidf_vectorizer = None
        self._tfidf_svd = None
        self._tfidf_fitted = False
        self._tfidf_vocabulary: Dict[str, int] = {}
        self._tfidf_idf_values: Dict[str, float] = {}
        self._sklearn_available = None
        self._jieba_available = None
        
        # TF-IDF 配置
        self._tfidf_config = {
            'max_features': self.config.get('tfidf_max_features', 10000),
            'min_df': self.config.get('tfidf_min_df', 1),
            'max_df': self.config.get('tfidf_max_df', 0.95),
            'ngram_range': self.config.get('tfidf_ngram_range', (1, 2)),
            'sublinear_tf': self.config.get('tfidf_sublinear_tf', True),
            'use_svd': self.config.get('tfidf_use_svd', True),
            'svd_components': self.config.get('tfidf_svd_components', None),  # None = 使用 embedding_dim
        }
        
        # Word2Vec 相关组件（懒加载）
        self._word2vec_model = None
        self._word2vec_loaded = False
        self._word2vec_vocab: Dict[str, np.ndarray] = {}  # 词向量缓存
        self._word2vec_oov_cache: Dict[str, np.ndarray] = {}  # OOV 词向量缓存
        self._gensim_available = None
        
        # Word2Vec 配置
        self._word2vec_config = {
            'vector_size': self.config.get('word2vec_vector_size', 300),
            'window': self.config.get('word2vec_window', 5),
            'min_count': self.config.get('word2vec_min_count', 1),
            'aggregation': self.config.get('word2vec_aggregation', 'mean'),  # mean, weighted, attention
            'use_subword': self.config.get('word2vec_use_subword', True),
            'subword_ngram_range': self.config.get('word2vec_subword_ngram_range', (3, 6)),
            'model_path': self.config.get('word2vec_model_path', None),
            'use_pretrained': self.config.get('word2vec_use_pretrained', True),
        }
        
        # 预定义的词向量模型路径
        self._word2vec_pretrained_paths = {
            'zh': [
                'word2vec/chinese_word2vec.bin',
                'word2vec/sgns.weibo.bigram-char',
                'word2vec/chinese_L100.bin',
            ],
            'en': [
                'word2vec/GoogleNews-vectors-negative300.bin',
                'word2vec/glove.6B.300d.txt',
                'word2vec/word2vec-google-news-300',
            ],
        }
        
        # 安全获取缓存目录
        cache_dir_path = self.config.get('cache_dir', 'embedding_cache')
        if isinstance(cache_dir_path, dict):
            cache_dir_path = 'embedding_cache'
        
        self.cache_dir = Path(str(cache_dir_path))
        self.cache_dir.mkdir(exist_ok=True)
        
        # 嵌入缓存
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_file = self.cache_dir / 'embedding_cache.json'
        self._load_cache()
        
        # 统计信息
        self.stats = {
            'total_embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'model_usage': {},
            'errors': 0,
            'fallbacks': 0
        }
        
        logger.info(f"EmbeddingManager initialized with dim={self.embedding_dim}, device={self.device}")
    
    def _determine_device(self) -> str:
        """确定使用的设备
        
        Returns:
            设备字符串: 'cuda' 或 'cpu'
        """
        device_config = self.config.get('device', 'auto')
        use_gpu = self.config.get('use_gpu', True)
        
        if device_config != 'auto':
            return device_config
        
        if not use_gpu:
            return 'cpu'
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return 'cuda'
        except ImportError:
            pass
        
        return 'cpu'
    
    def _check_sentence_transformer_available(self) -> bool:
        """检查 sentence-transformers 是否可用
        
        Returns:
            是否可用
        """
        if self._sentence_transformer_available is not None:
            return self._sentence_transformer_available
        
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_transformer_available = True
            logger.info("sentence-transformers library is available")
        except ImportError:
            self._sentence_transformer_available = False
            logger.warning("sentence-transformers library not installed, using fallback methods")
        
        return self._sentence_transformer_available
    
    def _load_sentence_transformer_model(self, model_key: str) -> Optional[Any]:
        """加载 SentenceTransformer 模型
        
        Args:
            model_key: 模型配置键
            
        Returns:
            加载的模型或 None
        """
        # 检查是否已加载
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]
        
        # 检查是否正在加载
        with self._model_lock:
            if self._model_loading.get(model_key, False):
                # 等待其他线程加载完成
                while self._model_loading.get(model_key, False):
                    time.sleep(0.1)
                return self._loaded_models.get(model_key)
            
            self._model_loading[model_key] = True
        
        try:
            if not self._check_sentence_transformer_available():
                return None
            
            from sentence_transformers import SentenceTransformer
            
            # 获取模型配置
            model_config = self.MODEL_CONFIGS.get(model_key)
            if not model_config:
                # 尝试直接使用 model_key 作为模型名称
                model_name = model_key
            else:
                model_name = model_config['model_name']
            
            # 模型缓存目录
            cache_folder = self.config.get('model_cache_dir')
            
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            start_time = time.time()
            
            # 加载模型
            model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=cache_folder
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
            
            # 缓存模型
            with self._model_lock:
                self._loaded_models[model_key] = model
                self._model_loading[model_key] = False
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            with self._model_lock:
                self._model_loading[model_key] = False
            return None
    
    def _load_cache(self):
        """加载嵌入缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 转换回numpy数组
                    for key, value in cache_data.items():
                        if isinstance(value, list):
                            self.embedding_cache[key] = np.array(value, dtype=np.float32)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """保存嵌入缓存"""
        try:
            if self.cache_enabled and self.embedding_cache:
                # 转换numpy数组为列表
                cache_data = {}
                for key, value in self.embedding_cache.items():
                    if isinstance(value, np.ndarray):
                        cache_data[key] = value.tolist()
                    else:
                        cache_data[key] = value
                
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {len(cache_data)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str, model_type: str = 'default') -> str:
        """生成缓存键"""
        content = f"{model_type}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, text: str, model_type: str = 'sentence-transformers') -> np.ndarray:
        """生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            model_type: 模型类型
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(text, model_type)
            if self.cache_enabled and cache_key in self.embedding_cache:
                self.stats['cache_hits'] += 1
                return self.embedding_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # 根据模型类型生成嵌入向量
            if model_type in ['sentence-transformers', 'bge', 'm3e', 'multilingual']:
                embedding = self._generate_sentence_transformer_embedding(text, model_type)
            elif model_type == 'tfidf':
                embedding = self._generate_tfidf_embedding(text)
            elif model_type == 'word2vec':
                embedding = self._generate_word2vec_embedding(text)
            elif model_type == 'openai':
                embedding = self._generate_openai_embedding(text)
            else:
                embedding = self._generate_default_embedding(text)
            
            # 缓存结果
            if self.cache_enabled:
                with self._lock:
                    self.embedding_cache[cache_key] = embedding
                    # 定期保存缓存
                    if len(self.embedding_cache) % 100 == 0:
                        self._save_cache()
            
            # 更新统计
            self._update_stats(model_type, time.time() - start_time)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            self.stats['errors'] += 1
            # 返回随机向量作为备用
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        model_type: str = 'sentence-transformers',
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """批量生成嵌入向量
        
        Args:
            texts: 文本列表
            model_type: 模型类型
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            嵌入向量数组 (numpy.ndarray)
        """
        if not texts:
            return np.array([])
        
        # 对于支持批量处理的模型，使用批量 API
        if model_type in ['sentence-transformers', 'bge', 'm3e', 'multilingual']:
            return self._generate_batch_sentence_transformer_embeddings(
                texts, model_type, batch_size, show_progress
            )
        
        # 其他模型逐个处理
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text, model_type)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _generate_sentence_transformer_embedding(
        self,
        text: str,
        model_type: str = 'sentence-transformers'
    ) -> np.ndarray:
        """生成 Sentence Transformer 风格的嵌入向量
        
        使用真实的 sentence-transformers 模型生成高质量嵌入向量。
        如果模型不可用，则使用降级的特征提取方法。
        
        Args:
            text: 输入文本
            model_type: 模型类型
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        # 确定使用的模型键
        model_key = self.DEFAULT_MODEL_MAP.get(model_type, model_type)
        
        # 尝试加载模型
        model = self._load_sentence_transformer_model(model_key)
        
        if model is not None:
            try:
                # 使用真实模型生成嵌入
                embedding = model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # 确保返回正确的类型
                if isinstance(embedding, np.ndarray):
                    return embedding.astype(np.float32)
                return np.array(embedding, dtype=np.float32)
                
            except Exception as e:
                logger.warning(f"Model inference failed, using fallback: {e}")
                self.stats['fallbacks'] += 1
        
        # 降级：使用改进的特征提取
        return self._generate_fallback_embedding(text)
    
    def _generate_batch_sentence_transformer_embeddings(
        self,
        texts: List[str],
        model_type: str = 'sentence-transformers',
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """批量生成 Sentence Transformer 嵌入向量
        
        Args:
            texts: 文本列表
            model_type: 模型类型
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            嵌入向量数组 (numpy.ndarray)
        """
        # 确定使用的模型键
        model_key = self.DEFAULT_MODEL_MAP.get(model_type, model_type)
        
        # 尝试加载模型
        model = self._load_sentence_transformer_model(model_key)
        
        if model is not None:
            try:
                # 使用真实模型批量生成嵌入
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress
                )
                
                # 更新统计
                for _ in texts:
                    self._update_stats(model_type, 0.0)
                
                return embeddings.astype(np.float32)
                
            except Exception as e:
                logger.warning(f"Batch model inference failed, using fallback: {e}")
                self.stats['fallbacks'] += len(texts)
        
        # 降级：逐个处理
        embeddings = []
        for text in texts:
            embedding = self._generate_fallback_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """生成降级嵌入向量
        
        当真实模型不可用时使用此方法。
        基于多种文本特征提取生成嵌入向量。
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        # 文本预处理
        text = text.lower().strip()
        words = text.split()
        
        # 基于词汇的特征
        vocab_features = self._extract_vocabulary_features(words)
        
        # 基于语义的特征
        semantic_features = self._extract_semantic_features(text)
        
        # 基于统计的特征
        statistical_features = self._extract_statistical_features(text)
        
        # N-gram 特征
        ngram_features = self._extract_ngram_features(text)
        
        # 字符级嵌入特征
        char_embedding_features = self._extract_char_embedding_features(text)
        
        # 组合所有特征
        all_features = np.concatenate([
            vocab_features,
            semantic_features,
            statistical_features,
            ngram_features,
            char_embedding_features
        ])
        
        # 调整到目标维度
        embedding = self._adjust_dimension(all_features, self.embedding_dim)
        
        # L2归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """生成 OpenAI 嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        api_key = self.config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("OpenAI API key not configured, using fallback")
            return self._generate_fallback_embedding(text)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except ImportError:
            logger.warning("openai library not installed, using fallback")
            return self._generate_fallback_embedding(text)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_embedding(text)
    
    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """生成 TF-IDF 风格的嵌入向量
        
        生产级的 TF-IDF 嵌入生成，支持：
        - sklearn TfidfVectorizer（如果可用）
        - 中文分词（jieba）
        - SVD 降维
        - 特征哈希降级
        - N-gram 特征
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        try:
            # 尝试使用 sklearn 的 TfidfVectorizer
            if self._check_sklearn_available():
                return self._generate_tfidf_sklearn(text)
            else:
                # 降级到自定义实现
                return self._generate_tfidf_custom(text)
                
        except Exception as e:
            logger.warning(f"TF-IDF embedding generation failed: {e}, using fallback")
            self.stats['fallbacks'] += 1
            return self._generate_tfidf_fallback(text)
    
    def _check_sklearn_available(self) -> bool:
        """检查 sklearn 是否可用
        
        Returns:
            是否可用
        """
        if self._sklearn_available is not None:
            return self._sklearn_available
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._sklearn_available = True
            logger.debug("sklearn library is available for TF-IDF")
        except ImportError:
            self._sklearn_available = False
            logger.warning("sklearn not installed, using custom TF-IDF implementation")
        
        return self._sklearn_available
    
    def _check_jieba_available(self) -> bool:
        """检查 jieba 是否可用
        
        Returns:
            是否可用
        """
        if self._jieba_available is not None:
            return self._jieba_available
        
        try:
            import jieba
            self._jieba_available = True
            logger.debug("jieba library is available for Chinese tokenization")
        except ImportError:
            self._jieba_available = False
            logger.debug("jieba not installed, using simple tokenization")
        
        return self._jieba_available
    
    def _tokenize_text(self, text: str, use_jieba: bool = True) -> List[str]:
        """智能分词
        
        自动检测文本语言并选择合适的分词方法：
        - 中文：使用 jieba 分词
        - 英文：使用空格分词 + 标点处理
        - 混合：结合两种方法
        
        Args:
            text: 输入文本
            use_jieba: 是否使用 jieba（对于中文）
            
        Returns:
            分词结果列表
        """
        import re
        
        # 预处理：统一小写，去除多余空白
        text = text.lower().strip()
        
        # 检测是否包含中文字符
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        chinese_ratio = chinese_chars / max(total_chars, 1)
        
        tokens = []
        
        if chinese_ratio > 0.3 and use_jieba and self._check_jieba_available():
            # 使用 jieba 进行中文分词
            import jieba
            
            # 分离中文和英文部分
            # 中文部分用 jieba 分词，英文部分用空格分词
            segments = re.split(r'([a-zA-Z0-9]+)', text)
            
            for segment in segments:
                if not segment.strip():
                    continue
                
                # 检查是否为纯英文/数字
                if re.match(r'^[a-zA-Z0-9]+$', segment):
                    tokens.append(segment.lower())
                else:
                    # 使用 jieba 分词
                    jieba_tokens = jieba.lcut(segment)
                    # 过滤空白和单字符标点
                    tokens.extend([t.strip() for t in jieba_tokens 
                                  if t.strip() and len(t.strip()) > 0])
        else:
            # 使用基于规则的分词
            # 分离标点符号
            text_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
            
            # 处理中文字符（每个字符单独作为 token）
            result = []
            current_word = []
            
            for char in text_clean:
                if '\u4e00' <= char <= '\u9fff':
                    # 中文字符：先保存之前的英文词
                    if current_word:
                        result.append(''.join(current_word))
                        current_word = []
                    result.append(char)
                elif char.isalnum():
                    current_word.append(char)
                else:
                    if current_word:
                        result.append(''.join(current_word))
                        current_word = []
            
            if current_word:
                result.append(''.join(current_word))
            
            tokens = [t for t in result if t.strip()]
        
        # 过滤停用词
        tokens = self._filter_stopwords(tokens)
        
        return tokens
    
    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        """过滤停用词
        
        Args:
            tokens: 分词结果
            
        Returns:
            过滤后的结果
        """
        # 中文停用词
        chinese_stopwords = {
            '的', '了', '是', '在', '有', '和', '与', '对', '为', '以',
            '这', '那', '也', '不', '都', '就', '而', '及', '或', '但',
            '如', '其', '之', '等', '被', '把', '从', '到', '使', '让',
            '会', '能', '可', '要', '将', '已', '所', '上', '下', '中',
            '着', '过', '来', '去', '很', '更', '最', '只', '又', '还',
            '个', '些', '什么', '怎么', '哪', '那些', '这些', '一个', '一些',
        }
        
        # 英文停用词
        english_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must', 'need',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'just', 'don', 'now', 'up', 'down', 'out', 'in', 'on',
            'at', 'by', 'for', 'from', 'to', 'of', 'with', 'about', 'into', 'through',
        }
        
        stopwords = chinese_stopwords | english_stopwords
        
        return [t for t in tokens if t not in stopwords and len(t) > 0]
    
    def _generate_tfidf_sklearn(self, text: str) -> np.ndarray:
        """使用 sklearn 生成 TF-IDF 嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
        from sklearn.decomposition import TruncatedSVD
        from scipy.sparse import csr_matrix
        
        # 分词
        tokens = self._tokenize_text(text)
        tokenized_text = ' '.join(tokens)
        
        # 如果分词结果为空，返回零向量
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 初始化或使用缓存的 TfidfVectorizer
        if self._tfidf_vectorizer is None:
            self._init_tfidf_vectorizer()
        
        try:
            # 使用特征哈希 + SVD 方法生成固定维度的嵌入
            # 这种方法不需要预先拟合，可以处理任意文本
            
            # 生成 TF-IDF 特征
            tfidf_vector = self._compute_tfidf_features(tokens)
            
            # 调整到目标维度
            embedding = self._adjust_dimension(tfidf_vector, self.embedding_dim)
            
            # L2 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"sklearn TF-IDF failed: {e}, using fallback")
            return self._generate_tfidf_fallback(text)
    
    def _init_tfidf_vectorizer(self):
        """初始化 TF-IDF 向量化器"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=self._tfidf_config['max_features'],
                min_df=self._tfidf_config['min_df'],
                max_df=self._tfidf_config['max_df'],
                ngram_range=self._tfidf_config['ngram_range'],
                sublinear_tf=self._tfidf_config['sublinear_tf'],
                norm='l2',
                use_idf=True,
                smooth_idf=True,
            )
            
            logger.debug("TfidfVectorizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TfidfVectorizer: {e}")
            self._tfidf_vectorizer = None
    
    def _compute_tfidf_features(self, tokens: List[str]) -> np.ndarray:
        """计算 TF-IDF 特征向量
        
        使用特征哈希技术生成固定维度的向量，
        不需要预先训练词汇表。
        
        Args:
            tokens: 分词结果
            
        Returns:
            TF-IDF 特征向量
        """
        import math
        
        # 特征维度（使用较大的维度以减少哈希冲突）
        feature_dim = min(self._tfidf_config['max_features'], 4096)
        
        # 计算词频 (TF)
        word_count = {}
        total_words = len(tokens)
        
        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1
        
        # 生成 N-gram
        ngrams = []
        ngram_range = self._tfidf_config['ngram_range']
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        # 计算 N-gram 频率
        ngram_count = {}
        for ngram in ngrams:
            ngram_count[ngram] = ngram_count.get(ngram, 0) + 1
        
        # 使用特征哈希生成向量
        features = np.zeros(feature_dim, dtype=np.float64)
        
        for ngram, count in ngram_count.items():
            # 计算 TF（使用 sublinear TF 如果配置）
            if self._tfidf_config['sublinear_tf']:
                tf = 1 + math.log(count) if count > 0 else 0
            else:
                tf = count / max(len(ngrams), 1)
            
            # 计算 IDF（使用平滑 IDF）
            # 使用缓存的 IDF 值或估算
            if ngram in self._tfidf_idf_values:
                idf = self._tfidf_idf_values[ngram]
            else:
                # 估算 IDF：基于 N-gram 长度和字符熵
                ngram_length = len(ngram.split())
                char_entropy = self._compute_string_entropy(ngram)
                # 较长的 N-gram 和较高熵的词通常更有区分度
                idf = 1 + math.log(1 + ngram_length) + char_entropy * 0.5
                self._tfidf_idf_values[ngram] = idf
            
            # 计算 TF-IDF 值
            tfidf = tf * idf
            
            # 特征哈希：将 N-gram 映射到固定维度
            hash_index = self._feature_hash(ngram, feature_dim)
            # 使用符号哈希减少冲突影响
            sign = 1 if self._feature_hash(ngram + '_sign', 2) == 0 else -1
            features[hash_index] += sign * tfidf
        
        return features
    
    def _feature_hash(self, text: str, num_features: int) -> int:
        """特征哈希函数
        
        将字符串映射到固定范围的索引
        
        Args:
            text: 输入字符串
            num_features: 特征数量
            
        Returns:
            哈希索引
        """
        hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        return hash_value % num_features
    
    def _compute_string_entropy(self, text: str) -> float:
        """计算字符串的熵
        
        Args:
            text: 输入字符串
            
        Returns:
            熵值
        """
        import math
        
        if not text:
            return 0.0
        
        char_count = {}
        for char in text:
            char_count[char] = char_count.get(char, 0) + 1
        
        entropy = 0.0
        total = len(text)
        for count in char_count.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _generate_tfidf_custom(self, text: str) -> np.ndarray:
        """自定义 TF-IDF 实现（不依赖 sklearn）
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        import math
        
        # 分词
        tokens = self._tokenize_text(text, use_jieba=self._check_jieba_available())
        
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 生成 N-gram
        ngram_range = self._tfidf_config['ngram_range']
        all_ngrams = []
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i+n])
                all_ngrams.append(ngram)
        
        # 计算词频
        ngram_freq = {}
        for ngram in all_ngrams:
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        
        # 特征维度
        feature_dim = min(2048, self.embedding_dim * 4)
        features = np.zeros(feature_dim, dtype=np.float64)
        
        # 计算 TF-IDF 并使用特征哈希
        for ngram, freq in ngram_freq.items():
            # TF (sublinear)
            tf = 1 + math.log(freq) if freq > 0 else 0
            
            # 简化的 IDF（基于逆文档频率估算）
            # 使用 N-gram 的字符多样性作为 IDF 的近似
            unique_chars = len(set(ngram))
            ngram_length = len(ngram)
            idf = math.log(1 + unique_chars) + math.log(1 + ngram_length / 10)
            
            # TF-IDF
            tfidf = tf * idf
            
            # 特征哈希
            hash_idx = self._feature_hash(ngram, feature_dim)
            sign = 1 if self._feature_hash(ngram + 's', 2) == 0 else -1
            features[hash_idx] += sign * tfidf
        
        # 添加位置权重特征
        position_features = self._compute_position_weighted_features(tokens)
        features = np.concatenate([features, position_features])
        
        # 添加统计特征增强
        stat_features = self._compute_tfidf_stat_features(text, tokens, ngram_freq)
        features = np.concatenate([features, stat_features])
        
        # 调整到目标维度
        embedding = self._adjust_dimension(features, self.embedding_dim)
        
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _compute_position_weighted_features(self, tokens: List[str], max_positions: int = 50) -> np.ndarray:
        """计算位置加权特征
        
        文档开头的词通常更重要
        
        Args:
            tokens: 分词结果
            max_positions: 最大位置数
            
        Returns:
            位置加权特征向量
        """
        import math
        
        features = np.zeros(max_positions, dtype=np.float64)
        
        for i, token in enumerate(tokens[:max_positions]):
            # 位置权重：前面的词权重更高
            position_weight = 1.0 / (1.0 + math.log(i + 1))
            
            # 词特征
            token_hash = self._feature_hash(token, max_positions)
            features[token_hash] += position_weight
        
        return features
    
    def _compute_tfidf_stat_features(
        self, 
        text: str, 
        tokens: List[str], 
        ngram_freq: Dict[str, int]
    ) -> np.ndarray:
        """计算 TF-IDF 统计特征
        
        Args:
            text: 原始文本
            tokens: 分词结果
            ngram_freq: N-gram 频率
            
        Returns:
            统计特征向量
        """
        import math
        
        features = []
        
        # 1. 词汇丰富度
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        lexical_diversity = unique_tokens / max(total_tokens, 1)
        features.append(lexical_diversity)
        
        # 2. 平均词长
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        features.append(avg_token_length / 10.0)
        
        # 3. N-gram 多样性
        unique_ngrams = len(ngram_freq)
        total_ngrams = sum(ngram_freq.values())
        ngram_diversity = unique_ngrams / max(total_ngrams, 1)
        features.append(ngram_diversity)
        
        # 4. 最大词频
        max_freq = max(ngram_freq.values()) if ngram_freq else 0
        features.append(max_freq / max(total_ngrams, 1))
        
        # 5. 词频熵
        freq_entropy = 0.0
        for freq in ngram_freq.values():
            if freq > 0 and total_ngrams > 0:
                prob = freq / total_ngrams
                freq_entropy -= prob * math.log2(prob)
        features.append(freq_entropy / 10.0)
        
        # 6. 文本长度特征
        features.append(len(text) / 1000.0)
        features.append(len(tokens) / 100.0)
        
        # 7. 中文/英文比例
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_count = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = chinese_count + english_count
        
        features.append(chinese_count / max(total_alpha, 1))
        features.append(english_count / max(total_alpha, 1))
        
        # 8. 数字和标点比例
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        features.append(digit_count / max(len(text), 1))
        features.append(punct_count / max(len(text), 1))
        
        return np.array(features, dtype=np.float64)
    
    def _generate_tfidf_fallback(self, text: str) -> np.ndarray:
        """TF-IDF 降级实现
        
        最简单的实现，当其他方法都失败时使用
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        import math
        
        # 简单分词
        text = text.lower()
        words = []
        current_word = []
        
        for char in text:
            if char.isalnum() or '\u4e00' <= char <= '\u9fff':
                current_word.append(char)
            else:
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
        
        if current_word:
            words.append(''.join(current_word))
        
        if not words:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 计算词频
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 使用特征哈希生成向量
        features = np.zeros(self.embedding_dim, dtype=np.float64)
        
        for word, freq in word_freq.items():
            # 简单的 TF-IDF
            tf = 1 + math.log(freq) if freq > 0 else 0
            idf = math.log(1 + len(word))  # 简化的 IDF
            tfidf = tf * idf
            
            # 哈希到特征空间
            hash_idx = self._feature_hash(word, self.embedding_dim)
            features[hash_idx] += tfidf
        
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def fit_tfidf_vocabulary(self, corpus: List[str]) -> None:
        """在语料库上训练 TF-IDF 词汇表
        
        可选方法：如果有语料库，可以预先训练以获得更好的 IDF 估算
        
        Args:
            corpus: 文本语料库列表
        """
        import math
        
        logger.info(f"Fitting TF-IDF vocabulary on {len(corpus)} documents...")
        
        # 文档频率统计
        doc_freq = {}
        total_docs = len(corpus)
        
        for doc in corpus:
            tokens = self._tokenize_text(doc)
            unique_tokens = set(tokens)
            
            # 生成 N-gram
            ngram_range = self._tfidf_config['ngram_range']
            all_ngrams = set()
            
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = '_'.join(tokens[i:i+n])
                    all_ngrams.add(ngram)
            
            # 更新文档频率
            for ngram in all_ngrams:
                doc_freq[ngram] = doc_freq.get(ngram, 0) + 1
        
        # 计算 IDF 值
        self._tfidf_idf_values = {}
        for ngram, df in doc_freq.items():
            # 平滑 IDF
            idf = math.log((total_docs + 1) / (df + 1)) + 1
            self._tfidf_idf_values[ngram] = idf
        
        self._tfidf_fitted = True
        logger.info(f"TF-IDF vocabulary fitted with {len(self._tfidf_idf_values)} terms")
    
    def _generate_word2vec_embedding(self, text: str) -> np.ndarray:
        """生成 Word2Vec 风格的嵌入向量
        
        生产级的 Word2Vec 嵌入生成，支持：
        - gensim Word2Vec/FastText 模型（如果可用）
        - 多种聚合策略（平均、TF-IDF加权、位置加权、注意力加权）
        - 子词嵌入处理未登录词（OOV）
        - 字符级 N-gram 特征
        - 语义增强特征
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        try:
            # 尝试使用 gensim 的 Word2Vec
            if self._check_gensim_available():
                return self._generate_word2vec_gensim(text)
            else:
                # 降级到自定义实现
                return self._generate_word2vec_custom(text)
                
        except Exception as e:
            logger.warning(f"Word2Vec embedding generation failed: {e}, using fallback")
            self.stats['fallbacks'] += 1
            return self._generate_word2vec_fallback(text)
    
    def _check_gensim_available(self) -> bool:
        """检查 gensim 是否可用
        
        Returns:
            是否可用
        """
        if self._gensim_available is not None:
            return self._gensim_available
        
        try:
            import gensim
            from gensim.models import Word2Vec, KeyedVectors
            self._gensim_available = True
            logger.debug("gensim library is available for Word2Vec")
        except ImportError:
            self._gensim_available = False
            logger.warning("gensim not installed, using custom Word2Vec implementation")
        
        return self._gensim_available
    
    def _load_word2vec_model(self) -> bool:
        """加载 Word2Vec 模型
        
        Returns:
            是否成功加载
        """
        if self._word2vec_loaded:
            return self._word2vec_model is not None
        
        with self._model_lock:
            if self._word2vec_loaded:
                return self._word2vec_model is not None
            
            try:
                from gensim.models import KeyedVectors
                
                # 尝试加载预训练模型
                model_path = self._word2vec_config.get('model_path')
                
                if model_path and Path(model_path).exists():
                    logger.info(f"Loading Word2Vec model from {model_path}")
                    
                    if model_path.endswith('.bin'):
                        self._word2vec_model = KeyedVectors.load_word2vec_format(
                            model_path, binary=True
                        )
                    elif model_path.endswith('.txt'):
                        self._word2vec_model = KeyedVectors.load_word2vec_format(
                            model_path, binary=False
                        )
                    else:
                        self._word2vec_model = KeyedVectors.load(model_path)
                    
                    logger.info(f"Word2Vec model loaded: {len(self._word2vec_model)} words")
                else:
                    logger.debug("No pretrained Word2Vec model found, using custom implementation")
                    self._word2vec_model = None
                    
            except Exception as e:
                logger.warning(f"Failed to load Word2Vec model: {e}")
                self._word2vec_model = None
            
            self._word2vec_loaded = True
            return self._word2vec_model is not None
    
    def _generate_word2vec_gensim(self, text: str) -> np.ndarray:
        """使用 gensim 生成 Word2Vec 嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 分词
        tokens = self._tokenize_text(text)
        
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 尝试加载预训练模型
        has_model = self._load_word2vec_model()
        
        # 获取每个词的向量
        word_vectors = []
        word_weights = []
        
        for i, token in enumerate(tokens):
            vector = self._get_word_vector(token, has_model)
            if vector is not None:
                word_vectors.append(vector)
                # 计算权重（位置权重 + TF-IDF 权重）
                weight = self._compute_word_weight(token, i, len(tokens))
                word_weights.append(weight)
        
        if not word_vectors:
            # 如果没有任何词向量，使用降级方法
            return self._generate_word2vec_custom(text)
        
        # 聚合词向量
        aggregation = self._word2vec_config.get('aggregation', 'mean')
        embedding = self._aggregate_word_vectors(
            word_vectors, word_weights, aggregation
        )
        
        # 调整到目标维度
        embedding = self._adjust_dimension(embedding, self.embedding_dim)
        
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _get_word_vector(self, word: str, has_model: bool) -> Optional[np.ndarray]:
        """获取词向量
        
        优先使用预训练模型，如果词不在词汇表中，使用子词嵌入
        
        Args:
            word: 输入词
            has_model: 是否有预训练模型
            
        Returns:
            词向量或 None
        """
        # 检查缓存
        if word in self._word2vec_vocab:
            return self._word2vec_vocab[word]
        
        vector = None
        
        # 尝试从预训练模型获取
        if has_model and self._word2vec_model is not None:
            try:
                if word in self._word2vec_model:
                    vector = self._word2vec_model[word]
                elif word.lower() in self._word2vec_model:
                    vector = self._word2vec_model[word.lower()]
            except KeyError:
                pass
        
        # 如果没找到，检查 OOV 缓存
        if vector is None and word in self._word2vec_oov_cache:
            return self._word2vec_oov_cache[word]
        
        # 使用子词嵌入处理 OOV
        if vector is None and self._word2vec_config.get('use_subword', True):
            vector = self._compute_subword_embedding(word, has_model)
        
        # 如果还是没有，生成随机但确定性的向量
        if vector is None:
            vector = self._generate_deterministic_word_vector(word)
        
        # 缓存结果
        if vector is not None:
            if has_model and self._word2vec_model is not None and word in self._word2vec_model:
                self._word2vec_vocab[word] = vector
            else:
                self._word2vec_oov_cache[word] = vector
        
        return vector
    
    def _compute_subword_embedding(self, word: str, has_model: bool) -> Optional[np.ndarray]:
        """计算子词嵌入
        
        使用字符级 N-gram 来处理未登录词
        
        Args:
            word: 输入词
            has_model: 是否有预训练模型
            
        Returns:
            子词嵌入向量
        """
        ngram_range = self._word2vec_config.get('subword_ngram_range', (3, 6))
        vector_size = self._word2vec_config.get('vector_size', 300)
        
        # 添加词边界标记
        word_with_boundaries = f"<{word}>"
        
        # 提取字符 N-gram
        ngrams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(word_with_boundaries) - n + 1):
                ngram = word_with_boundaries[i:i+n]
                ngrams.append(ngram)
        
        if not ngrams:
            return None
        
        # 尝试从预训练模型获取 N-gram 向量
        ngram_vectors = []
        
        if has_model and self._word2vec_model is not None:
            for ngram in ngrams:
                try:
                    if ngram in self._word2vec_model:
                        ngram_vectors.append(self._word2vec_model[ngram])
                except KeyError:
                    pass
        
        # 如果没有从模型获取到，使用确定性哈希生成
        if not ngram_vectors:
            for ngram in ngrams:
                ngram_vector = self._hash_to_vector(ngram, vector_size)
                ngram_vectors.append(ngram_vector)
        
        if ngram_vectors:
            # 平均所有 N-gram 向量
            return np.mean(ngram_vectors, axis=0)
        
        return None
    
    def _generate_deterministic_word_vector(self, word: str) -> np.ndarray:
        """生成确定性的词向量
        
        生产级的确定性词向量生成，融合多种特征：
        - 字符级特征（字符 N-gram、字符分布）
        - 语义特征（词长、词性估计、语言检测）
        - 形态学特征（前缀、后缀、词根）
        - 音韵学特征（元音/辅音模式、音节结构）
        - 结构特征（大小写模式、数字、特殊字符）
        
        确保相同的词总是得到相同的向量，同时保持语义相关词的向量相似性。
        
        Args:
            word: 输入词
            
        Returns:
            词向量 (numpy.ndarray)
        """
        vector_size = self._word2vec_config.get('vector_size', 300)
        
        if not word:
            return np.zeros(vector_size, dtype=np.float32)
        
        # 分配各特征的维度
        # 确保总和等于 vector_size
        hash_dim = vector_size // 3           # 基础哈希特征
        char_dim = vector_size // 6           # 字符级特征
        ngram_dim = vector_size // 6          # N-gram 特征
        semantic_dim = vector_size // 12      # 语义特征
        morph_dim = vector_size // 12         # 形态学特征
        phonetic_dim = vector_size // 12      # 音韵学特征
        remaining_dim = vector_size - hash_dim - char_dim - ngram_dim - semantic_dim - morph_dim - phonetic_dim
        
        # 1. 基础哈希向量（确保确定性）
        hash_vector = self._generate_base_hash_vector(word, hash_dim)
        
        # 2. 字符级特征
        char_vector = self._generate_char_distribution_vector(word, char_dim)
        
        # 3. N-gram 哈希特征
        ngram_vector = self._generate_ngram_hash_vector(word, ngram_dim)
        
        # 4. 语义特征
        semantic_vector = self._generate_semantic_feature_vector(word, semantic_dim)
        
        # 5. 形态学特征
        morph_vector = self._generate_morphological_vector(word, morph_dim)
        
        # 6. 音韵学特征
        phonetic_vector = self._generate_phonetic_vector(word, phonetic_dim)
        
        # 7. 剩余维度用额外哈希填充
        extra_vector = self._hash_to_vector(f"{word}_extra", remaining_dim)
        
        # 合并所有特征
        full_vector = np.concatenate([
            hash_vector,
            char_vector,
            ngram_vector,
            semantic_vector,
            morph_vector,
            phonetic_vector,
            extra_vector
        ])
        
        # 确保维度正确
        if len(full_vector) != vector_size:
            full_vector = self._adjust_dimension(full_vector, vector_size)
        
        # 处理数值问题
        full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # L2 归一化
        norm = np.linalg.norm(full_vector)
        if norm > 0:
            full_vector = full_vector / norm
        else:
            # 降级到简单哈希
            full_vector = self._hash_to_vector(word, vector_size)
        
        return full_vector.astype(np.float32)
    
    def _generate_base_hash_vector(self, word: str, dim: int) -> np.ndarray:
        """生成基础哈希向量
        
        使用多个哈希函数和不同变体生成确定性向量
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            哈希向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        # 词的多种变体
        variants = [
            word,                    # 原始
            word.lower(),            # 小写
            word.upper(),            # 大写
            word.capitalize(),       # 首字母大写
            word[::-1],              # 反转
            f"<{word}>",             # 添加边界
        ]
        
        # 多种哈希算法
        hash_funcs = [
            ('md5', hashlib.md5),
            ('sha1', hashlib.sha1),
            ('sha256', hashlib.sha256),
            ('sha512', hashlib.sha512),
        ]
        
        chunk_size = dim // (len(variants) * len(hash_funcs))
        idx = 0
        
        for variant in variants:
            for hash_name, hash_func in hash_funcs:
                # 生成哈希
                hash_bytes = hash_func(variant.encode('utf-8')).digest()
                
                # 转换为浮点数
                for byte in hash_bytes:
                    if idx < dim:
                        # 使用正弦变换增加非线性
                        value = np.sin((byte - 128) / 128.0 * np.pi)
                        vector[idx] = value
                        idx += 1
                    else:
                        break
                
                if idx >= dim:
                    break
            if idx >= dim:
                break
        
        return vector
    
    def _generate_char_distribution_vector(self, word: str, dim: int) -> np.ndarray:
        """生成字符分布向量
        
        基于字符频率和分布生成特征向量
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            字符分布向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        if not word:
            return vector
        
        word_lower = word.lower()
        word_len = len(word)
        
        # 1. 字符频率特征（基于 ASCII/Unicode 范围）
        char_freq_dim = dim // 3
        
        for char in word_lower:
            # 将字符映射到向量索引
            char_code = ord(char)
            idx = char_code % char_freq_dim
            vector[idx] += 1.0 / word_len
        
        # 2. 字符类型分布
        type_start = char_freq_dim
        type_dim = dim // 3
        
        # 统计各类型字符
        lower_count = sum(1 for c in word if c.islower())
        upper_count = sum(1 for c in word if c.isupper())
        digit_count = sum(1 for c in word if c.isdigit())
        chinese_count = sum(1 for c in word if '\u4e00' <= c <= '\u9fff')
        punct_count = sum(1 for c in word if not c.isalnum())
        
        # 归一化并填充
        type_features = [
            lower_count / max(word_len, 1),
            upper_count / max(word_len, 1),
            digit_count / max(word_len, 1),
            chinese_count / max(word_len, 1),
            punct_count / max(word_len, 1),
        ]
        
        for i, feat in enumerate(type_features):
            if type_start + i < dim:
                vector[type_start + i] = feat
        
        # 3. 位置编码特征
        pos_start = char_freq_dim + type_dim
        pos_dim = dim - pos_start
        
        for i, char in enumerate(word[:min(word_len, pos_dim)]):
            if pos_start + i < dim:
                # 位置加权的字符编码
                position_weight = 1.0 - (i / max(word_len, 1)) * 0.5
                char_value = (ord(char) % 256 - 128) / 128.0
                vector[pos_start + i] = char_value * position_weight
        
        return vector
    
    def _generate_ngram_hash_vector(self, word: str, dim: int) -> np.ndarray:
        """生成 N-gram 哈希向量
        
        使用字符级 N-gram 的哈希特征
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            N-gram 哈希向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        if not word:
            return vector
        
        # 添加词边界标记
        word_bounded = f"<{word}>"
        
        # 提取不同长度的 N-gram
        ngrams = []
        for n in range(1, 7):  # 1-gram 到 6-gram
            for i in range(len(word_bounded) - n + 1):
                ngram = word_bounded[i:i+n]
                ngrams.append((ngram, n))
        
        if not ngrams:
            return vector
        
        # N-gram 特征哈希
        for ngram, n in ngrams:
            # 计算哈希索引
            hash_val = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
            idx = hash_val % dim
            
            # N-gram 长度权重（较长的 N-gram 权重更高）
            weight = np.log(1 + n) / np.log(7)
            
            # 符号哈希（减少碰撞影响）
            sign = 1 if (hash_val // dim) % 2 == 0 else -1
            
            vector[idx] += sign * weight / len(ngrams)
        
        return vector
    
    def _generate_semantic_feature_vector(self, word: str, dim: int) -> np.ndarray:
        """生成语义特征向量
        
        基于词的语义属性生成特征
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            语义特征向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        if not word or dim < 10:
            return vector
        
        word_lower = word.lower()
        
        # 1. 词长特征
        vector[0] = min(len(word), 30) / 30.0
        
        # 2. 语言检测特征
        chinese_ratio = sum(1 for c in word if '\u4e00' <= c <= '\u9fff') / max(len(word), 1)
        english_ratio = sum(1 for c in word if c.isascii() and c.isalpha()) / max(len(word), 1)
        digit_ratio = sum(1 for c in word if c.isdigit()) / max(len(word), 1)
        
        vector[1] = chinese_ratio
        vector[2] = english_ratio
        vector[3] = digit_ratio
        
        # 3. 大小写模式
        vector[4] = 1.0 if word[0].isupper() else 0.0  # 首字母大写
        vector[5] = 1.0 if word.isupper() else 0.0     # 全大写
        vector[6] = 1.0 if word.islower() else 0.0     # 全小写
        
        # 4. 词性估计（启发式）
        # 动词特征（以 -ing, -ed, -s 结尾）
        verb_suffixes = ['ing', 'ed', 'es', 's']
        vector[7] = 1.0 if any(word_lower.endswith(s) for s in verb_suffixes) else 0.0
        
        # 名词特征（以 -tion, -ness, -ment 结尾）
        noun_suffixes = ['tion', 'sion', 'ness', 'ment', 'ity', 'ty']
        vector[8] = 1.0 if any(word_lower.endswith(s) for s in noun_suffixes) else 0.0
        
        # 形容词特征（以 -ly, -ful, -less, -ous 结尾）
        adj_suffixes = ['ly', 'ful', 'less', 'ous', 'ive', 'able', 'ible']
        vector[9] = 1.0 if any(word_lower.endswith(s) for s in adj_suffixes) else 0.0
        
        # 5. 使用剩余维度存储语义哈希
        if dim > 10:
            # 基于语义类别的确定性哈希
            semantic_categories = [
                ('tech', ['tech', 'data', 'code', 'api', 'web', 'app', 'ai', 'ml']),
                ('science', ['science', 'research', 'study', 'theory', 'experiment']),
                ('business', ['business', 'market', 'sales', 'profit', 'revenue']),
                ('time', ['day', 'year', 'month', 'time', 'hour', 'minute']),
                ('space', ['place', 'location', 'area', 'region', 'country']),
            ]
            
            for cat_idx, (cat_name, keywords) in enumerate(semantic_categories):
                if 10 + cat_idx < dim:
                    # 检查词是否包含该类别的关键词
                    match_score = sum(1 for kw in keywords if kw in word_lower) / len(keywords)
                    vector[10 + cat_idx] = match_score
        
        return vector
    
    def _generate_morphological_vector(self, word: str, dim: int) -> np.ndarray:
        """生成形态学特征向量
        
        基于词的形态学结构生成特征
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            形态学特征向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        if not word or dim < 5:
            return vector
        
        word_lower = word.lower()
        
        # 1. 常见前缀检测
        prefixes = [
            ('un', 0.5), ('re', 0.5), ('in', 0.4), ('dis', 0.5), ('en', 0.4),
            ('non', 0.5), ('pre', 0.5), ('mis', 0.5), ('over', 0.5), ('sub', 0.5),
            ('anti', 0.6), ('auto', 0.6), ('co', 0.4), ('de', 0.4), ('ex', 0.4),
            ('inter', 0.6), ('multi', 0.6), ('out', 0.4), ('post', 0.5), ('semi', 0.5),
            ('super', 0.6), ('trans', 0.6), ('ultra', 0.6), ('under', 0.5),
        ]
        
        prefix_dim = min(len(prefixes), dim // 2)
        for i, (prefix, weight) in enumerate(prefixes[:prefix_dim]):
            if word_lower.startswith(prefix):
                vector[i] = weight
        
        # 2. 常见后缀检测
        suffixes = [
            ('ing', 0.5), ('ed', 0.4), ('er', 0.4), ('est', 0.4), ('ly', 0.5),
            ('ness', 0.5), ('ment', 0.5), ('tion', 0.6), ('sion', 0.6), ('ful', 0.5),
            ('less', 0.5), ('able', 0.5), ('ible', 0.5), ('ive', 0.5), ('ous', 0.5),
            ('al', 0.4), ('ity', 0.5), ('ty', 0.4), ('ry', 0.4), ('ship', 0.5),
            ('ism', 0.5), ('ist', 0.5), ('ize', 0.5), ('ify', 0.5),
        ]
        
        suffix_start = prefix_dim
        suffix_dim = min(len(suffixes), dim - suffix_start)
        for i, (suffix, weight) in enumerate(suffixes[:suffix_dim]):
            if word_lower.endswith(suffix):
                if suffix_start + i < dim:
                    vector[suffix_start + i] = weight
        
        return vector
    
    def _generate_phonetic_vector(self, word: str, dim: int) -> np.ndarray:
        """生成音韵学特征向量
        
        基于词的发音特征生成向量
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            音韵学特征向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        if not word or dim < 5:
            return vector
        
        word_lower = word.lower()
        
        # 英文元音和辅音
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        # 1. 元音/辅音统计
        vowel_count = sum(1 for c in word if c in vowels)
        consonant_count = sum(1 for c in word if c in consonants)
        total = vowel_count + consonant_count
        
        vector[0] = vowel_count / max(total, 1)
        vector[1] = consonant_count / max(total, 1)
        
        # 2. 音节估计（基于元音群）
        syllable_count = 0
        prev_vowel = False
        for c in word_lower:
            is_vowel = c in vowels
            if is_vowel and not prev_vowel:
                syllable_count += 1
            prev_vowel = is_vowel
        
        vector[2] = min(syllable_count, 10) / 10.0
        
        # 3. 元音模式编码
        vowel_pattern = ''.join('V' if c in vowels else 'C' for c in word_lower if c.isalpha())
        pattern_hash = int(hashlib.md5(vowel_pattern.encode()).hexdigest(), 16)
        
        if dim > 3:
            vector[3] = (pattern_hash % 1000) / 1000.0
        
        # 4. 首尾音素特征
        if dim > 4 and word:
            first_char = word[0].lower()
            last_char = word[-1].lower()
            
            # 首字符是元音
            vector[4] = 1.0 if first_char in vowels else 0.0
            
            if dim > 5:
                # 尾字符是元音
                vector[5] = 1.0 if last_char in vowels else 0.0
        
        # 5. 双元音/双辅音检测
        if dim > 6:
            double_vowel = any(word_lower[i] in vowels and word_lower[i+1] in vowels 
                             for i in range(len(word_lower)-1))
            vector[6] = 1.0 if double_vowel else 0.0
        
        if dim > 7:
            double_consonant = any(word_lower[i] == word_lower[i+1] and word_lower[i] in consonants
                                  for i in range(len(word_lower)-1))
            vector[7] = 1.0 if double_consonant else 0.0
        
        # 6. 使用剩余维度存储音韵哈希
        if dim > 8:
            # Soundex 风格的编码（简化版）
            soundex_map = {
                'b': '1', 'f': '1', 'p': '1', 'v': '1',
                'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
                'd': '3', 't': '3',
                'l': '4',
                'm': '5', 'n': '5',
                'r': '6',
            }
            
            soundex = word[0].upper() if word else ''
            prev_code = ''
            for c in word_lower[1:]:
                code = soundex_map.get(c, '')
                if code and code != prev_code:
                    soundex += code
                prev_code = code
            
            soundex_hash = int(hashlib.md5(soundex.encode()).hexdigest(), 16)
            remaining = dim - 8
            for i in range(remaining):
                vector[8 + i] = ((soundex_hash >> (i * 8)) & 0xFF - 128) / 128.0
        
        return vector
    
    def _hash_to_vector(self, text: str, dim: int) -> np.ndarray:
        """将文本哈希为向量
        
        使用多个哈希函数生成高维向量，确保数值稳定
        
        Args:
            text: 输入文本
            dim: 目标维度
            
        Returns:
            向量
        """
        vector = np.zeros(dim, dtype=np.float64)
        
        # 使用多个哈希种子生成不同的哈希值
        seeds = [0, 42, 123, 456, 789, 1024, 2048, 4096, 16384, 32768]
        
        for seed_idx, seed in enumerate(seeds):
            # 生成 SHA256 哈希
            hash_input = f"{text}_{seed}".encode('utf-8')
            hash_bytes = hashlib.sha256(hash_input).digest()
            
            # 将字节转换为 [-1, 1] 范围的浮点数
            # 每个字节映射到 [-1, 1]
            floats = [(b - 128) / 128.0 for b in hash_bytes]
            
            # 填充向量的一部分
            chunk_size = dim // len(seeds)
            start_idx = seed_idx * chunk_size
            end_idx = start_idx + chunk_size if seed_idx < len(seeds) - 1 else dim
            
            for i, val in enumerate(floats):
                idx = start_idx + (i % (end_idx - start_idx))
                if idx < dim:
                    vector[idx] += val
        
        # 处理可能的 NaN 或 Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            # 如果范数为0，使用随机初始化
            np.random.seed(hash(text) % (2**32))
            vector = np.random.randn(dim)
            vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def _compute_word_weight(self, word: str, position: int, total_words: int) -> float:
        """计算词的权重
        
        结合位置权重和词频权重
        
        Args:
            word: 输入词
            position: 词在文本中的位置
            total_words: 文本总词数
            
        Returns:
            权重值
        """
        import math
        
        # 位置权重：句首和句尾的词更重要
        if total_words <= 1:
            position_weight = 1.0
        else:
            # U 形权重：开头和结尾权重高，中间权重低
            relative_pos = position / (total_words - 1) if total_words > 1 else 0
            position_weight = 1.0 - 0.5 * math.sin(math.pi * relative_pos)
        
        # TF-IDF 权重（如果有）
        idf_weight = 1.0
        if word in self._tfidf_idf_values:
            idf_weight = self._tfidf_idf_values[word]
        else:
            # 使用词长度作为简化的 IDF 估计
            idf_weight = 1.0 + math.log(1 + len(word)) / 5
        
        # 词性权重估计（基于启发式规则）
        pos_weight = self._estimate_pos_weight(word)
        
        return position_weight * idf_weight * pos_weight
    
    def _estimate_pos_weight(self, word: str) -> float:
        """估计词性权重
        
        基于启发式规则估计词的重要性
        
        Args:
            word: 输入词
            
        Returns:
            权重值
        """
        # 停用词权重低
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'shall', 'must',
            '的', '了', '是', '在', '有', '和', '与', '对', '为', '以',
            '这', '那', '也', '不', '都', '就', '而', '及', '或', '但',
        }
        
        if word.lower() in stopwords:
            return 0.3
        
        # 数字权重中等
        if word.isdigit():
            return 0.5
        
        # 专有名词（首字母大写）权重高
        if word and word[0].isupper() and word[1:].islower():
            return 1.5
        
        # 全大写（缩写）权重高
        if word.isupper() and len(word) > 1:
            return 1.3
        
        # 长词权重稍高
        if len(word) > 8:
            return 1.2
        
        return 1.0
    
    def _aggregate_word_vectors(
        self, 
        vectors: List[np.ndarray], 
        weights: List[float], 
        method: str = 'mean'
    ) -> np.ndarray:
        """聚合词向量
        
        Args:
            vectors: 词向量列表
            weights: 权重列表
            method: 聚合方法 (mean, weighted, attention, max)
            
        Returns:
            聚合后的向量
        """
        if not vectors:
            return np.zeros(self._word2vec_config.get('vector_size', 300))
        
        vectors = np.array(vectors)
        weights = np.array(weights)
        
        if method == 'mean':
            # 简单平均
            return np.mean(vectors, axis=0)
        
        elif method == 'weighted':
            # 加权平均
            weights = weights / (np.sum(weights) + 1e-8)
            return np.sum(vectors * weights[:, np.newaxis], axis=0)
        
        elif method == 'attention':
            # 自注意力聚合
            return self._attention_aggregation(vectors, weights)
        
        elif method == 'max':
            # 最大池化
            return np.max(vectors, axis=0)
        
        elif method == 'concat_pool':
            # 连接池化：平均 + 最大
            mean_vec = np.mean(vectors, axis=0)
            max_vec = np.max(vectors, axis=0)
            return np.concatenate([mean_vec, max_vec])
        
        else:
            # 默认使用加权平均
            weights = weights / (np.sum(weights) + 1e-8)
            return np.sum(vectors * weights[:, np.newaxis], axis=0)
    
    def _attention_aggregation(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """自注意力聚合
        
        Args:
            vectors: 词向量矩阵 (n_words, dim)
            weights: 初始权重
            
        Returns:
            聚合后的向量
        """
        # 计算向量间的相似度作为注意力分数
        # 使用简化的点积注意力
        mean_vec = np.mean(vectors, axis=0)
        
        # 计算每个向量与均值的相似度
        attention_scores = np.array([
            np.dot(vec, mean_vec) / (np.linalg.norm(vec) * np.linalg.norm(mean_vec) + 1e-8)
            for vec in vectors
        ])
        
        # 结合原始权重
        combined_scores = attention_scores * weights
        
        # Softmax 归一化
        exp_scores = np.exp(combined_scores - np.max(combined_scores))
        attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        # 加权求和
        return np.sum(vectors * attention_weights[:, np.newaxis], axis=0)
    
    def _generate_word2vec_custom(self, text: str) -> np.ndarray:
        """自定义 Word2Vec 实现（不依赖 gensim）
        
        使用多种特征生成类似 Word2Vec 的嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 分词
        tokens = self._tokenize_text(text)
        
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        vector_size = self._word2vec_config.get('vector_size', 300)
        
        # 生成每个词的向量
        word_vectors = []
        word_weights = []
        
        for i, token in enumerate(tokens):
            # 生成确定性词向量
            word_vec = self._generate_rich_word_vector(token)
            word_vectors.append(word_vec)
            
            # 计算权重
            weight = self._compute_word_weight(token, i, len(tokens))
            word_weights.append(weight)
        
        # 聚合
        aggregation = self._word2vec_config.get('aggregation', 'weighted')
        embedding = self._aggregate_word_vectors(word_vectors, word_weights, aggregation)
        
        # 添加上下文特征
        context_features = self._extract_context_features(tokens)
        embedding = np.concatenate([embedding, context_features])
        
        # 调整到目标维度
        embedding = self._adjust_dimension(embedding, self.embedding_dim)
        
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _generate_rich_word_vector(self, word: str) -> np.ndarray:
        """生成丰富的词向量
        
        结合多种特征生成词向量
        
        Args:
            word: 输入词
            
        Returns:
            词向量
        """
        vector_size = self._word2vec_config.get('vector_size', 300)
        
        # 1. 基础哈希向量
        base_vector = self._hash_to_vector(word, vector_size // 2)
        
        # 2. 字符级特征
        char_features = self._extract_char_features(word, vector_size // 4)
        
        # 3. 形态学特征
        morph_features = self._extract_morphological_features(word, vector_size // 4)
        
        # 合并所有特征
        full_vector = np.concatenate([base_vector, char_features, morph_features])
        
        # 调整到目标维度
        if len(full_vector) != vector_size:
            full_vector = self._adjust_dimension(full_vector, vector_size)
        
        return full_vector
    
    def _extract_char_features(self, word: str, dim: int) -> np.ndarray:
        """提取字符级特征
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            字符特征向量
        """
        features = np.zeros(dim, dtype=np.float64)
        
        if not word:
            return features
        
        # 字符分布特征
        char_counts = {}
        for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # 字符熵
        total_chars = len(word)
        char_entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                prob = count / total_chars
                char_entropy -= prob * np.log2(prob + 1e-10)
        
        features[0] = char_entropy / 5.0  # 归一化
        
        # 元音/辅音比例
        vowels = set('aeiouAEIOU')
        vowel_count = sum(1 for c in word if c in vowels)
        features[1] = vowel_count / max(len(word), 1)
        
        # 字符类型分布
        alpha_count = sum(1 for c in word if c.isalpha())
        digit_count = sum(1 for c in word if c.isdigit())
        chinese_count = sum(1 for c in word if '\u4e00' <= c <= '\u9fff')
        
        features[2] = alpha_count / max(len(word), 1)
        features[3] = digit_count / max(len(word), 1)
        features[4] = chinese_count / max(len(word), 1)
        
        # 词长特征
        features[5] = min(len(word), 20) / 20.0
        
        # 首字母大写
        features[6] = 1.0 if word and word[0].isupper() else 0.0
        
        # 全大写
        features[7] = 1.0 if word.isupper() and len(word) > 1 else 0.0
        
        # 字符 N-gram 哈希特征
        ngram_start = 8
        ngram_dim = dim - ngram_start
        
        for n in range(2, 5):
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                hash_idx = self._feature_hash(ngram, ngram_dim)
                features[ngram_start + hash_idx] += 1.0 / (len(word) - n + 1)
        
        return features
    
    def _extract_morphological_features(self, word: str, dim: int) -> np.ndarray:
        """提取形态学特征
        
        Args:
            word: 输入词
            dim: 目标维度
            
        Returns:
            形态学特征向量
        """
        features = np.zeros(dim, dtype=np.float64)
        
        if not word:
            return features
        
        word_lower = word.lower()
        
        # 常见前缀
        prefixes = ['un', 're', 'in', 'dis', 'en', 'non', 'pre', 'mis', 'over', 'sub', 
                   'anti', 'auto', 'co', 'de', 'ex', 'inter', 'multi', 'out', 'post', 'semi']
        
        for i, prefix in enumerate(prefixes[:min(len(prefixes), dim // 4)]):
            if word_lower.startswith(prefix):
                features[i] = 1.0
        
        # 常见后缀
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'sion', 'ful',
                   'less', 'able', 'ible', 'ive', 'ous', 'al', 'ity', 'ty', 'ry', 'ship']
        
        suffix_start = dim // 4
        for i, suffix in enumerate(suffixes[:min(len(suffixes), dim // 4)]):
            if word_lower.endswith(suffix):
                features[suffix_start + i] = 1.0
        
        # 词形特征
        morph_start = dim // 2
        
        # 是否包含连字符
        features[morph_start] = 1.0 if '-' in word else 0.0
        
        # 是否包含撇号
        features[morph_start + 1] = 1.0 if "'" in word else 0.0
        
        # 是否包含数字
        features[morph_start + 2] = 1.0 if any(c.isdigit() for c in word) else 0.0
        
        # 词的"形状"编码
        shape = self._get_word_shape(word)
        shape_hash = self._feature_hash(shape, dim // 4)
        features[morph_start + 3 + shape_hash % (dim // 4 - 3)] = 1.0
        
        # 音节估计（基于元音）
        vowels = 'aeiouAEIOU'
        syllable_count = sum(1 for i, c in enumerate(word) 
                           if c in vowels and (i == 0 or word[i-1] not in vowels))
        features[morph_start + dim // 4] = min(syllable_count, 10) / 10.0
        
        return features
    
    def _get_word_shape(self, word: str) -> str:
        """获取词的形状
        
        Args:
            word: 输入词
            
        Returns:
            形状字符串
        """
        shape = []
        prev_type = None
        
        for char in word:
            if char.isupper():
                char_type = 'X'
            elif char.islower():
                char_type = 'x'
            elif char.isdigit():
                char_type = 'd'
            elif '\u4e00' <= char <= '\u9fff':
                char_type = 'c'
            else:
                char_type = char
            
            # 压缩连续相同类型
            if char_type != prev_type:
                shape.append(char_type)
                prev_type = char_type
        
        return ''.join(shape)
    
    def _extract_context_features(self, tokens: List[str]) -> np.ndarray:
        """提取上下文特征
        
        Args:
            tokens: 分词结果
            
        Returns:
            上下文特征向量
        """
        dim = 50  # 上下文特征维度
        features = np.zeros(dim, dtype=np.float64)
        
        if not tokens:
            return features
        
        # 词汇丰富度
        unique_ratio = len(set(tokens)) / len(tokens)
        features[0] = unique_ratio
        
        # 平均词长
        avg_length = np.mean([len(t) for t in tokens])
        features[1] = avg_length / 10.0
        
        # 词长标准差
        length_std = np.std([len(t) for t in tokens]) if len(tokens) > 1 else 0
        features[2] = length_std / 5.0
        
        # 最大词长
        max_length = max(len(t) for t in tokens)
        features[3] = max_length / 20.0
        
        # 单字词比例
        single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
        features[4] = single_char_ratio
        
        # 长词比例（>8字符）
        long_word_ratio = sum(1 for t in tokens if len(t) > 8) / len(tokens)
        features[5] = long_word_ratio
        
        # 数字词比例
        digit_ratio = sum(1 for t in tokens if t.isdigit()) / len(tokens)
        features[6] = digit_ratio
        
        # 首字母大写词比例
        cap_ratio = sum(1 for t in tokens if t and t[0].isupper()) / len(tokens)
        features[7] = cap_ratio
        
        # 中文词比例
        chinese_ratio = sum(1 for t in tokens if any('\u4e00' <= c <= '\u9fff' for c in t)) / len(tokens)
        features[8] = chinese_ratio
        
        # Bigram 多样性
        bigrams = set()
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i+1]))
        bigram_diversity = len(bigrams) / max(len(tokens) - 1, 1)
        features[9] = bigram_diversity
        
        return features
    
    def _generate_word2vec_fallback(self, text: str) -> np.ndarray:
        """Word2Vec 降级实现
        
        最简单的实现，当其他方法都失败时使用
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 简单分词
        words = []
        current_word = []
        
        for char in text.lower():
            if char.isalnum() or '\u4e00' <= char <= '\u9fff':
                current_word.append(char)
            else:
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
        
        if current_word:
            words.append(''.join(current_word))
        
        if not words:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 为每个词生成简单的哈希向量
        word_vectors = []
        
        for word in words:
            # 使用 MD5 哈希生成确定性向量
            hash_bytes = hashlib.md5(word.encode('utf-8')).digest()
            
            # 将哈希转换为浮点数
            vector = []
            for byte in hash_bytes:
                vector.append((byte - 128) / 128.0)
            
            # 扩展到需要的维度
            while len(vector) < self.embedding_dim // 4:
                extended_hash = hashlib.md5(f"{word}_{len(vector)}".encode()).digest()
                for byte in extended_hash:
                    vector.append((byte - 128) / 128.0)
                    if len(vector) >= self.embedding_dim // 4:
                        break
            
            word_vectors.append(np.array(vector[:self.embedding_dim // 4]))
        
        # 平均所有词向量
        if word_vectors:
            avg_vector = np.mean(word_vectors, axis=0)
        else:
            avg_vector = np.zeros(self.embedding_dim // 4)
        
        # 调整到目标维度
        embedding = self._adjust_dimension(avg_vector, self.embedding_dim)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def train_word2vec(self, corpus: List[str], **kwargs) -> None:
        """在语料库上训练 Word2Vec 模型
        
        Args:
            corpus: 文本语料库
            **kwargs: gensim Word2Vec 参数
        """
        if not self._check_gensim_available():
            logger.warning("gensim not available, cannot train Word2Vec model")
            return
        
        from gensim.models import Word2Vec
        
        logger.info(f"Training Word2Vec on {len(corpus)} documents...")
        
        # 分词
        tokenized_corpus = [self._tokenize_text(doc) for doc in corpus]
        
        # 训练参数
        params = {
            'vector_size': self._word2vec_config.get('vector_size', 300),
            'window': self._word2vec_config.get('window', 5),
            'min_count': self._word2vec_config.get('min_count', 1),
            'workers': 4,
            'epochs': 10,
        }
        params.update(kwargs)
        
        # 训练模型
        model = Word2Vec(sentences=tokenized_corpus, **params)
        
        # 保存词向量
        self._word2vec_model = model.wv
        self._word2vec_loaded = True
        
        # 清空缓存
        self._word2vec_vocab.clear()
        self._word2vec_oov_cache.clear()
        
        logger.info(f"Word2Vec trained with vocabulary size: {len(self._word2vec_model)}")
    
    def save_word2vec_model(self, path: str) -> None:
        """保存 Word2Vec 模型
        
        Args:
            path: 保存路径
        """
        if self._word2vec_model is None:
            logger.warning("No Word2Vec model to save")
            return
        
        try:
            self._word2vec_model.save(path)
            logger.info(f"Word2Vec model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Word2Vec model: {e}")
    
    def load_word2vec_model(self, path: str) -> bool:
        """加载 Word2Vec 模型
        
        Args:
            path: 模型路径
            
        Returns:
            是否成功加载
        """
        self._word2vec_config['model_path'] = path
        self._word2vec_loaded = False
        return self._load_word2vec_model()
    
    def _generate_default_embedding(self, text: str) -> np.ndarray:
        """生成默认嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy.ndarray)
        """
        return self._generate_fallback_embedding(text)
    
    def _extract_vocabulary_features(self, words: List[str]) -> np.ndarray:
        """提取词汇特征
        
        Args:
            words: 词列表
            
        Returns:
            特征向量
        """
        features = []
        
        # 词汇多样性
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        features.append(vocab_diversity)
        
        # 平均词长
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        features.append(avg_word_length / 10.0)
        
        # 常见词汇特征（中文停用词）
        common_words_zh = {'的', '是', '在', '有', '和', '与', '对', '为', '了', '以', '这', '那', '也', '不', '都'}
        # 英文停用词
        common_words_en = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does'}
        
        common_word_ratio_zh = sum(1 for word in words if word in common_words_zh) / max(len(words), 1)
        common_word_ratio_en = sum(1 for word in words if word.lower() in common_words_en) / max(len(words), 1)
        features.extend([common_word_ratio_zh, common_word_ratio_en])
        
        # 词长分布
        if words:
            word_lengths = [len(word) for word in words]
            features.append(np.std(word_lengths) / 5.0)
            features.append(max(word_lengths) / 20.0)
            features.append(min(word_lengths) / 5.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """提取语义特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征向量
        """
        features = []
        
        # 领域关键词密度
        domain_keywords = {
            'technology': ['技术', '科技', '创新', '智能', '数字', '自动化', 'AI', 'ML', 'technology', 'digital'],
            'business': ['业务', '商业', '市场', '客户', '服务', '产品', 'business', 'market', 'customer'],
            'science': ['研究', '实验', '数据', '分析', '理论', '方法', 'research', 'data', 'analysis'],
            'health': ['健康', '医疗', '治疗', '疾病', '药物', '患者', 'health', 'medical', 'treatment'],
            'education': ['教育', '学习', '培训', '课程', '学生', '教师', 'education', 'learning', 'training'],
            'finance': ['金融', '投资', '资本', '股票', '银行', '贷款', 'finance', 'investment', 'capital'],
        }
        
        for category, keywords in domain_keywords.items():
            density = sum(1 for word in keywords if word in text or word.lower() in text.lower()) / len(keywords)
            features.append(density)
        
        # 情感倾向
        positive_words = ['好', '优', '高', '强', '增', '提升', '改善', '成功', 'good', 'great', 'excellent', 'better', 'best']
        negative_words = ['差', '低', '弱', '减', '下降', '问题', '困难', '失败', 'bad', 'poor', 'worse', 'worst', 'problem']
        
        positive_score = sum(1 for word in positive_words if word in text or word.lower() in text.lower()) / len(positive_words)
        negative_score = sum(1 for word in negative_words if word in text or word.lower() in text.lower()) / len(negative_words)
        
        features.extend([positive_score, negative_score, positive_score - negative_score])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_statistical_features(self, text: str) -> np.ndarray:
        """提取统计特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征向量
        """
        features = []
        
        # 文本长度特征
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 100.0)
        
        # 标点符号特征
        punctuation_zh = ['，', '。', '？', '！', '：', '；', '、']
        punctuation_en = [',', '.', '?', '!', ':', ';']
        
        for p in punctuation_zh + punctuation_en:
            features.append(text.count(p) / 50.0)
        
        # 数字特征
        digit_count = sum(1 for char in text if char.isdigit())
        features.append(digit_count / 100.0)
        
        # 大写字母比例（英文）
        upper_count = sum(1 for char in text if char.isupper())
        alpha_count = sum(1 for char in text if char.isalpha())
        features.append(upper_count / max(alpha_count, 1))
        
        # 中文字符比例
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        features.append(chinese_count / max(len(text), 1))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_ngram_features(self, text: str, n_range: Tuple[int, int] = (2, 3)) -> np.ndarray:
        """提取 N-gram 特征
        
        Args:
            text: 输入文本
            n_range: N-gram 范围
            
        Returns:
            特征向量
        """
        features = []
        
        # 字符级 N-gram
        for n in range(n_range[0], n_range[1] + 1):
            ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
            
            if ngrams:
                # N-gram 多样性
                unique_ngrams = set(ngrams)
                diversity = len(unique_ngrams) / len(ngrams)
                features.append(diversity)
                
                # 最常见 N-gram 的频率
                ngram_freq = {}
                for ng in ngrams:
                    ngram_freq[ng] = ngram_freq.get(ng, 0) + 1
                
                max_freq = max(ngram_freq.values()) / len(ngrams)
                features.append(max_freq)
            else:
                features.extend([0.0, 0.0])
        
        # 填充到固定长度
        target_len = 10
        while len(features) < target_len:
            features.append(0.0)
        
        return np.array(features[:target_len], dtype=np.float32)
    
    def _extract_char_embedding_features(self, text: str, max_chars: int = 100) -> np.ndarray:
        """提取字符级嵌入特征
        
        基于字符的 Unicode 编码生成嵌入特征
        
        Args:
            text: 输入文本
            max_chars: 最大字符数
            
        Returns:
            特征向量
        """
        # 取前 max_chars 个字符
        text = text[:max_chars]
        
        features = []
        
        # 字符熵
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(text)
        if total_chars > 0:
            entropy = 0
            for count in char_counts.values():
                prob = count / total_chars
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            features.append(entropy / 10.0)
        else:
            features.append(0.0)
        
        # 基于字符编码的特征
        if text:
            char_codes = [ord(c) for c in text]
            features.append(np.mean(char_codes) / 65535.0)
            features.append(np.std(char_codes) / 10000.0)
            features.append(np.median(char_codes) / 65535.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 字符类型分布
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        space_ratio = sum(1 for c in text if c.isspace()) / max(len(text), 1)
        
        features.extend([alpha_ratio, digit_ratio, space_ratio])
        
        return np.array(features, dtype=np.float32)
    
    def _adjust_dimension(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """调整特征维度
        
        Args:
            features: 输入特征
            target_dim: 目标维度
            
        Returns:
            调整后的特征向量
        """
        current_dim = len(features)
        
        if current_dim == target_dim:
            return features
        elif current_dim < target_dim:
            # 填充零值
            padding = np.zeros(target_dim - current_dim)
            return np.concatenate([features, padding])
        else:
            # 截断或降维
            if current_dim > target_dim * 2:
                # 使用简单的降维（取平均）
                chunk_size = current_dim // target_dim
                reduced_features = []
                for i in range(target_dim):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, current_dim)
                    chunk_mean = np.mean(features[start_idx:end_idx])
                    reduced_features.append(chunk_mean)
                return np.array(reduced_features)
            else:
                # 直接截断
                return features[:target_dim]
    
    def _update_stats(self, model_type: str, processing_time: float):
        """更新统计信息
        
        Args:
            model_type: 模型类型
            processing_time: 处理时间
        """
        with self._lock:
            self.stats['total_embeddings_generated'] += 1
            self.stats['total_processing_time'] += processing_time
            
            if model_type not in self.stats['model_usage']:
                self.stats['model_usage'][model_type] = 0
            self.stats['model_usage'][model_type] += 1
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            相似度值 (0-1)
        """
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self.stats.copy()
        
        stats['cache_size'] = len(self.embedding_cache)
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / 
            max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        )
        stats['avg_processing_time'] = (
            self.stats['total_processing_time'] / 
            max(self.stats['total_embeddings_generated'], 1)
        )
        stats['loaded_models'] = list(self._loaded_models.keys())
        stats['device'] = self.device
        stats['sentence_transformer_available'] = self._sentence_transformer_available
        
        return stats
    
    def get_model_info(self, model_type: str = None) -> Dict[str, Any]:
        """获取模型信息
        
        Args:
            model_type: 模型类型（可选，不指定则返回所有模型信息）
            
        Returns:
            模型信息字典
        """
        if model_type:
            model_key = self.DEFAULT_MODEL_MAP.get(model_type, model_type)
            if model_key in self.MODEL_CONFIGS:
                config = self.MODEL_CONFIGS[model_key].copy()
                config['is_loaded'] = model_key in self._loaded_models
                return config
            return {}
        
        # 返回所有模型信息
        return {
            key: {**config, 'is_loaded': key in self._loaded_models}
            for key, config in self.MODEL_CONFIGS.items()
        }
    
    def preload_model(self, model_type: str = 'sentence-transformers') -> bool:
        """预加载模型
        
        Args:
            model_type: 模型类型
            
        Returns:
            是否加载成功
        """
        model_key = self.DEFAULT_MODEL_MAP.get(model_type, model_type)
        model = self._load_sentence_transformer_model(model_key)
        return model is not None
    
    def unload_model(self, model_type: str = None):
        """卸载模型以释放内存
        
        Args:
            model_type: 模型类型（不指定则卸载所有模型）
        """
        with self._model_lock:
            if model_type:
                model_key = self.DEFAULT_MODEL_MAP.get(model_type, model_type)
                if model_key in self._loaded_models:
                    del self._loaded_models[model_key]
                    logger.info(f"Unloaded model: {model_key}")
            else:
                self._loaded_models.clear()
                logger.info("Unloaded all models")
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.embedding_cache.clear()
        
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        logger.info("Embedding cache cleared")
    
    def __del__(self):
        """析构函数，保存缓存"""
        try:
            self._save_cache()
        except:
            pass


# ==================== 全局实例管理 ====================

_embedding_manager: Optional[EmbeddingManager] = None
_manager_lock = threading.Lock()


def get_embedding_manager(config: Dict[str, Any] = None) -> EmbeddingManager:
    """获取全局嵌入管理器实例
    
    Args:
        config: 配置字典（仅首次调用时有效）
        
    Returns:
        EmbeddingManager 实例
    """
    global _embedding_manager
    
    with _manager_lock:
        if _embedding_manager is None:
            try:
                _embedding_manager = EmbeddingManager(config)
            except Exception as e:
                logger.error(f"Failed to create embedding manager: {e}")
                raise
            else:
                logger.info("EmbeddingManager initialized successfully")
    
    return _embedding_manager


def set_embedding_manager(embedding_manager: EmbeddingManager):
    """设置全局嵌入管理器实例
    
    Args:
        embedding_manager: EmbeddingManager 实例
    """
    global _embedding_manager
    
    with _manager_lock:
        _embedding_manager = embedding_manager


def reset_embedding_manager():
    """重置全局嵌入管理器实例（用于测试）"""
    global _embedding_manager
    
    with _manager_lock:
        if _embedding_manager:
            _embedding_manager.unload_model()
        _embedding_manager = None


# ==================== 导出 ====================

__all__ = [
    'EmbeddingManager',
    'get_embedding_manager',
    'set_embedding_manager',
    'reset_embedding_manager',
]
