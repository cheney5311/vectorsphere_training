# -*- coding: utf-8 -*-
"""
生产级数据工程模块

提供多模态数据处理能力：
- 数据去重（Perceptual Hash / MinHash）
- 噪声过滤（NSFW、错配）
- 模态一致性校验
- 版权与合规扫描
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Set, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json
import os
from collections import defaultdict

import torch
import numpy as np

from .production_multimodal_config import (
    DataEngineeringConfig,
    DataDeduplicationConfig,
    DataFilterConfig,
    DataAugmentationConfig,
    DataSourceType
)

logger = logging.getLogger(__name__)


# ==================== 数据样本 ====================

@dataclass
class MultiModalSample:
    """多模态数据样本"""
    sample_id: str
    modalities: Dict[str, Any] = field(default_factory=dict)  # {modality: data}
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 质量标记
    is_valid: bool = True
    quality_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    
    # 去重标记
    content_hash: Optional[str] = None
    is_duplicate: bool = False


# ==================== 数据去重 ====================

class DataDeduplicator:
    """数据去重器"""
    
    def __init__(self, config: DataDeduplicationConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.hash_to_samples: Dict[str, List[str]] = defaultdict(list)
    
    def compute_hash(self, sample: MultiModalSample) -> str:
        """计算样本哈希"""
        if self.config.method == "perceptual_hash":
            return self._perceptual_hash(sample)
        elif self.config.method == "minhash":
            return self._minhash(sample)
        elif self.config.method == "simhash":
            return self._simhash(sample)
        else:
            return self._content_hash(sample)
    
    def _perceptual_hash(self, sample: MultiModalSample) -> str:
        """感知哈希（主要用于图像）"""
        hashes = []
        
        # 图像感知哈希
        if 'image' in sample.modalities:
            image = sample.modalities['image']
            if isinstance(image, np.ndarray):
                # 简化的pHash实现
                # 1. 缩放到固定大小
                # 2. 转灰度
                # 3. DCT变换
                # 4. 取低频分量
                # 5. 二值化生成哈希
                h = hashlib.md5(image.tobytes()).hexdigest()
                hashes.append(f"img:{h[:16]}")
            elif isinstance(image, torch.Tensor):
                h = hashlib.md5(image.numpy().tobytes()).hexdigest()
                hashes.append(f"img:{h[:16]}")
        
        # 文本哈希
        if 'text' in sample.modalities:
            text = str(sample.modalities['text'])
            h = hashlib.md5(text.encode()).hexdigest()
            hashes.append(f"txt:{h[:16]}")
        
        return "|".join(hashes) if hashes else hashlib.md5(
            json.dumps(sample.modalities, default=str).encode()
        ).hexdigest()
    
    def _minhash(self, sample: MultiModalSample) -> str:
        """MinHash（主要用于文本）"""
        # 简化实现：使用多个哈希函数
        text = ""
        for modality, data in sample.modalities.items():
            if isinstance(data, str):
                text += data
            elif isinstance(data, list):
                text += " ".join(str(x) for x in data)
        
        # 生成n-gram
        ngrams = set()
        words = text.split()
        for i in range(len(words) - 2):
            ngrams.add(" ".join(words[i:i+3]))
        
        # 计算MinHash签名
        min_hashes = []
        for seed in range(self.config.num_perm):
            min_hash = float('inf')
            for ngram in ngrams:
                h = hash(ngram + str(seed))
                if h < min_hash:
                    min_hash = h
            min_hashes.append(min_hash)
        
        # 转换为字符串哈希
        return hashlib.md5(str(min_hashes).encode()).hexdigest()
    
    def _simhash(self, sample: MultiModalSample) -> str:
        """SimHash"""
        # 简化实现
        text = ""
        for modality, data in sample.modalities.items():
            if isinstance(data, str):
                text += data
        
        if not text:
            return self._content_hash(sample)
        
        # 分词并计算特征向量
        words = text.split()
        v = [0] * 64
        
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(64):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # 生成SimHash
        simhash = 0
        for i in range(64):
            if v[i] > 0:
                simhash |= (1 << i)
        
        return format(simhash, '016x')
    
    def _content_hash(self, sample: MultiModalSample) -> str:
        """内容哈希"""
        content = json.dumps(sample.modalities, default=str, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_duplicate(self, sample: MultiModalSample) -> bool:
        """检查是否重复"""
        content_hash = self.compute_hash(sample)
        sample.content_hash = content_hash
        
        if content_hash in self.seen_hashes:
            sample.is_duplicate = True
            return True
        
        self.seen_hashes.add(content_hash)
        self.hash_to_samples[content_hash].append(sample.sample_id)
        return False
    
    def deduplicate_batch(self, samples: List[MultiModalSample]) -> List[MultiModalSample]:
        """批量去重"""
        unique_samples = []
        for sample in samples:
            if not self.is_duplicate(sample):
                unique_samples.append(sample)
        
        logger.info(f"Deduplicated: {len(samples)} -> {len(unique_samples)}")
        return unique_samples


# ==================== 数据过滤 ====================

class DataFilter:
    """数据过滤器"""
    
    def __init__(self, config: DataFilterConfig):
        self.config = config
        self.nsfw_detector = None
        self.language_detector = None
    
    def filter(self, sample: MultiModalSample) -> bool:
        """过滤样本，返回是否保留"""
        issues = []
        
        # 文本长度检查
        if 'text' in sample.modalities:
            text = str(sample.modalities['text'])
            if len(text) < self.config.min_text_length:
                issues.append(f"Text too short: {len(text)}")
            if len(text) > self.config.max_text_length:
                issues.append(f"Text too long: {len(text)}")
        
        # 图像尺寸检查
        if 'image' in sample.modalities:
            image = sample.modalities['image']
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            elif isinstance(image, torch.Tensor):
                h, w = image.shape[-2:]
            else:
                h, w = self.config.min_image_size, self.config.min_image_size
            
            if h < self.config.min_image_size or w < self.config.min_image_size:
                issues.append(f"Image too small: {h}x{w}")
            if h > self.config.max_image_size or w > self.config.max_image_size:
                issues.append(f"Image too large: {h}x{w}")
        
        # NSFW过滤（简化实现）
        if self.config.nsfw_filter:
            nsfw_score = self._check_nsfw(sample)
            if nsfw_score > self.config.nsfw_threshold:
                issues.append(f"NSFW content detected: {nsfw_score:.2f}")
        
        # 模态一致性检查
        if self.config.consistency_check:
            consistency_score = self._check_consistency(sample)
            if consistency_score < self.config.consistency_threshold:
                issues.append(f"Low consistency: {consistency_score:.2f}")
        
        # 语言过滤
        if self.config.language_filter and 'text' in sample.modalities:
            lang = self._detect_language(sample.modalities['text'])
            if lang not in self.config.language_filter:
                issues.append(f"Language mismatch: {lang}")
        
        sample.issues = issues
        sample.is_valid = len(issues) == 0
        sample.quality_score = 1.0 - len(issues) * 0.2
        
        return sample.is_valid
    
    def _check_nsfw(self, sample: MultiModalSample) -> float:
        """NSFW检测（简化实现）"""
        # 实际应使用专门的NSFW检测模型
        return 0.0
    
    def _check_consistency(self, sample: MultiModalSample) -> float:
        """模态一致性检查"""
        # 检查图文匹配度等
        # 实际应使用CLIP等模型计算相似度
        if len(sample.modalities) < 2:
            return 1.0
        
        # 简化：检查是否都有内容
        has_content = [
            bool(data) for data in sample.modalities.values()
        ]
        return sum(has_content) / len(has_content)
    
    def _detect_language(self, text: str) -> str:
        """语言检测"""
        # 简化实现
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        return 'en'
    
    def filter_batch(self, samples: List[MultiModalSample]) -> List[MultiModalSample]:
        """批量过滤"""
        valid_samples = []
        for sample in samples:
            if self.filter(sample):
                valid_samples.append(sample)
        
        logger.info(f"Filtered: {len(samples)} -> {len(valid_samples)}")
        return valid_samples


# ==================== 数据增强 ====================

class DataAugmentor:
    """数据增强器"""
    
    def __init__(self, config: DataAugmentationConfig):
        self.config = config
    
    def augment(self, sample: MultiModalSample) -> List[MultiModalSample]:
        """数据增强"""
        augmented = [sample]  # 原始样本
        
        # 图像增强
        if 'image' in sample.modalities and isinstance(
            sample.modalities['image'], (np.ndarray, torch.Tensor)
        ):
            augmented.extend(self._augment_image(sample))
        
        # 文本增强
        if 'text' in sample.modalities and self.config.text_synonym_replace:
            augmented.extend(self._augment_text(sample))
        
        return augmented
    
    def _augment_image(self, sample: MultiModalSample) -> List[MultiModalSample]:
        """图像增强"""
        augmented = []
        image = sample.modalities['image']
        
        # 水平翻转
        if self.config.image_flip:
            new_sample = MultiModalSample(
                sample_id=f"{sample.sample_id}_flip",
                modalities={**sample.modalities},
                metadata={**sample.metadata, 'augment': 'flip'}
            )
            if isinstance(image, np.ndarray):
                new_sample.modalities['image'] = np.fliplr(image)
            elif isinstance(image, torch.Tensor):
                new_sample.modalities['image'] = torch.flip(image, dims=[-1])
            augmented.append(new_sample)
        
        return augmented
    
    def _augment_text(self, sample: MultiModalSample) -> List[MultiModalSample]:
        """文本增强"""
        # 简化实现
        return []
    
    def augment_batch(self, samples: List[MultiModalSample]) -> List[MultiModalSample]:
        """批量增强"""
        all_augmented = []
        for sample in samples:
            all_augmented.extend(self.augment(sample))
        
        logger.info(f"Augmented: {len(samples)} -> {len(all_augmented)}")
        return all_augmented


# ==================== 数据管道 ====================

class MultiModalDataPipeline:
    """多模态数据处理管道"""
    
    def __init__(self, config: DataEngineeringConfig):
        self.config = config
        
        # 初始化各处理器
        self.deduplicator = DataDeduplicator(config.deduplication)
        self.filter = DataFilter(config.filtering)
        self.augmentor = DataAugmentor(config.augmentation)
        
        # 统计
        self.stats = {
            'total_input': 0,
            'after_dedup': 0,
            'after_filter': 0,
            'after_augment': 0
        }
    
    def process(self, samples: List[MultiModalSample]) -> List[MultiModalSample]:
        """处理数据管道"""
        self.stats['total_input'] = len(samples)
        
        # 1. 去重
        if self.config.deduplication.enabled:
            samples = self.deduplicator.deduplicate_batch(samples)
        self.stats['after_dedup'] = len(samples)
        
        # 2. 过滤
        if self.config.filtering.enabled:
            samples = self.filter.filter_batch(samples)
        self.stats['after_filter'] = len(samples)
        
        # 3. 增强
        if self.config.augmentation.enabled:
            samples = self.augmentor.augment_batch(samples)
        self.stats['after_augment'] = len(samples)
        
        logger.info(f"Pipeline stats: {self.stats}")
        
        return samples
    
    def process_stream(self, 
                      sample_iterator: Iterator[MultiModalSample],
                      batch_size: int = 1000) -> Iterator[MultiModalSample]:
        """流式处理"""
        batch = []
        
        for sample in sample_iterator:
            batch.append(sample)
            
            if len(batch) >= batch_size:
                processed = self.process(batch)
                for s in processed:
                    yield s
                batch = []
        
        # 处理剩余样本
        if batch:
            processed = self.process(batch)
            for s in processed:
                yield s
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'dedup_rate': 1 - self.stats['after_dedup'] / max(self.stats['total_input'], 1),
            'filter_rate': 1 - self.stats['after_filter'] / max(self.stats['after_dedup'], 1),
            'augment_rate': self.stats['after_augment'] / max(self.stats['after_filter'], 1) - 1
        }


# ==================== 数据集构建器 ====================

class MultiModalDatasetBuilder:
    """多模态数据集构建器"""
    
    def __init__(self, config: DataEngineeringConfig):
        self.config = config
        self.pipeline = MultiModalDataPipeline(config)
        self.samples: List[MultiModalSample] = []
    
    def add_samples(self, samples: List[MultiModalSample]):
        """添加样本"""
        self.samples.extend(samples)
    
    def add_from_source(self, source: DataSourceType, data_path: str):
        """从数据源添加"""
        logger.info(f"Loading from {source.value}: {data_path}")
        
        if source == DataSourceType.WEB_IMAGE_TEXT:
            self._load_web_image_text(data_path)
        elif source == DataSourceType.VIDEO_ASR:
            self._load_video_asr(data_path)
        elif source == DataSourceType.OCR_DOCUMENT:
            self._load_ocr_document(data_path)
        # ... 其他数据源
    
    def _load_web_image_text(self, data_path: str):
        """加载图文网页数据"""
        # 实现具体的加载逻辑
        pass
    
    def _load_video_asr(self, data_path: str):
        """加载视频+ASR数据"""
        pass
    
    def _load_ocr_document(self, data_path: str):
        """加载OCR文档数据"""
        pass
    
    def build(self) -> List[MultiModalSample]:
        """构建数据集"""
        logger.info(f"Building dataset with {len(self.samples)} samples")
        
        # 通过管道处理
        processed_samples = self.pipeline.process(self.samples)
        
        logger.info(f"Dataset built: {len(processed_samples)} samples")
        return processed_samples
    
    def save(self, output_path: str):
        """保存数据集"""
        processed_samples = self.build()
        
        os.makedirs(output_path, exist_ok=True)
        
        # 保存样本
        for i, sample in enumerate(processed_samples):
            sample_path = os.path.join(output_path, f"sample_{i:08d}.json")
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'sample_id': sample.sample_id,
                    'modalities': {
                        k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
                        for k, v in sample.modalities.items()
                    },
                    'metadata': sample.metadata,
                    'quality_score': sample.quality_score
                }, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats_path = os.path.join(output_path, "stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline.get_statistics(), f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")

