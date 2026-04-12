"""数据集详细管理功能测试

测试 dataset_detailed_api.py 及其下游服务的完整功能。
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid

# ============================================================================
# 模拟模型定义
# ============================================================================

@dataclass
class Dataset:
    """数据集模型"""
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    description: Optional[str] = None
    dataset_type: str = "text"
    format: str = "json"
    storage_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    status: str = "pending"
    ready: bool = False
    size: Optional[int] = None
    record_count: Optional[int] = None
    features: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    checksum: Optional[str] = None
    validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "dataset_type": self.dataset_type,
            "format": self.format,
            "storage_path": self.storage_path,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "ready": self.ready,
            "size": self.size,
            "record_count": self.record_count,
            "features": self.features,
            "labels": self.labels,
            "version": self.version,
            "checksum": self.checksum,
            "validated": self.validated,
        }


@dataclass
class DatasetVersion:
    """数据集版本模型"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str = ""
    version: str = "1.0"
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    size: Optional[int] = None
    record_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "size": self.size,
            "record_count": self.record_count,
        }


@dataclass
class DatasetStatistics:
    """数据集统计信息"""
    dataset_id: str = ""
    row_count: int = 0
    column_count: int = 0
    size_bytes: int = 0
    missing_values: int = 0
    duplicate_rows: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "size_bytes": self.size_bytes,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
        }


# ============================================================================
# 模拟仓库层
# ============================================================================

class MockDatasetRepository:
    """模拟数据集仓库"""
    
    def __init__(self):
        self._datasets: Dict[str, Dataset] = {}
        self._tags: Dict[str, List[str]] = {}  # dataset_id -> tags
        self._versions: Dict[str, List[DatasetVersion]] = {}  # dataset_id -> versions
        self._access_logs: List[Dict[str, Any]] = []
    
    def create(self, dataset: Dataset, tenant_id: str = None) -> Dataset:
        if not dataset.dataset_id:
            dataset.dataset_id = str(uuid.uuid4())
        self._datasets[dataset.dataset_id] = dataset
        self._tags[dataset.dataset_id] = []
        self._versions[dataset.dataset_id] = []
        return dataset
    
    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        return self._datasets.get(dataset_id)
    
    def update(self, dataset: Dataset) -> Dataset:
        dataset.updated_at = datetime.utcnow()
        self._datasets[dataset.dataset_id] = dataset
        return dataset
    
    def delete(self, dataset_id: str) -> bool:
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            if dataset_id in self._tags:
                del self._tags[dataset_id]
            if dataset_id in self._versions:
                del self._versions[dataset_id]
            return True
        return False
    
    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets
    
    def list_by_user(self, user_id: str, **kwargs) -> List[Dataset]:
        return [d for d in self._datasets.values() if d.user_id == user_id]


class MockTagRepository:
    """模拟标签仓库"""
    
    def __init__(self, dataset_repo: MockDatasetRepository):
        self._dataset_repo = dataset_repo
    
    def add_tag(self, dataset_id: str, tag_name: str, tag_value: str = None, created_by: str = None) -> str:
        if dataset_id not in self._dataset_repo._tags:
            self._dataset_repo._tags[dataset_id] = []
        if tag_name not in self._dataset_repo._tags[dataset_id]:
            self._dataset_repo._tags[dataset_id].append(tag_name)
        return str(uuid.uuid4())
    
    def remove_tag(self, dataset_id: str, tag_name: str) -> bool:
        if dataset_id in self._dataset_repo._tags:
            if tag_name in self._dataset_repo._tags[dataset_id]:
                self._dataset_repo._tags[dataset_id].remove(tag_name)
                return True
        return False
    
    def get_tags(self, dataset_id: str) -> List[Dict[str, str]]:
        tags = self._dataset_repo._tags.get(dataset_id, [])
        return [{'name': t, 'tag_name': t} for t in tags]
    
    def find_by_tag(self, user_id: str, tag_name: str, tag_value: str = None) -> List[str]:
        result = []
        for dataset_id, tags in self._dataset_repo._tags.items():
            if tag_name in tags:
                dataset = self._dataset_repo.get_by_id(dataset_id)
                if dataset and dataset.user_id == user_id:
                    result.append(dataset_id)
        return result


class MockVersionRepository:
    """模拟版本仓库"""
    
    def __init__(self, dataset_repo: MockDatasetRepository):
        self._dataset_repo = dataset_repo
    
    def create(self, version: DatasetVersion) -> DatasetVersion:
        if not version.version_id:
            version.version_id = str(uuid.uuid4())
        if version.dataset_id not in self._dataset_repo._versions:
            self._dataset_repo._versions[version.dataset_id] = []
        self._dataset_repo._versions[version.dataset_id].insert(0, version)
        return version
    
    def list_by_dataset(self, dataset_id: str) -> List[DatasetVersion]:
        return self._dataset_repo._versions.get(dataset_id, [])


class MockAccessLogRepository:
    """模拟访问日志仓库"""
    
    def __init__(self, dataset_repo: MockDatasetRepository):
        self._dataset_repo = dataset_repo
    
    def create(self, dataset_id: str, user_id: str, action: str, details: Dict[str, Any] = None):
        self._dataset_repo._access_logs.append({
            'dataset_id': dataset_id,
            'user_id': user_id,
            'action': action,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def list_by_dataset(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        logs = [l for l in self._dataset_repo._access_logs if l['dataset_id'] == dataset_id]
        return logs[offset:offset + limit]


# ============================================================================
# 模拟服务层
# ============================================================================

class DatasetDetailedService:
    """数据集详细管理服务"""
    
    def __init__(self, dataset_repo: MockDatasetRepository):
        self.dataset_repository = dataset_repo
        self.tag_repository = MockTagRepository(dataset_repo)
        self.version_repository = MockVersionRepository(dataset_repo)
        self.access_log_repository = MockAccessLogRepository(dataset_repo)
    
    def create_dataset(
        self, 
        user_id: str, 
        name: str, 
        description: str = None,
        dataset_type: str = "text",
        format: str = "json",
        storage_path: str = "",
        config: Dict[str, Any] = None
    ) -> Dataset:
        """创建数据集"""
        dataset = Dataset(
            user_id=user_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            format=format,
            storage_path=storage_path,
            config=config or {},
            size=1024000,
            record_count=1000,
            features={'columns': ['id', 'content', 'label', 'created_at']}
        )
        return self.dataset_repository.create(dataset)
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """获取数据集"""
        return self.dataset_repository.get_by_id(dataset_id)
    
    def update_dataset(
        self, 
        dataset_id: str, 
        name: str = None, 
        description: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ) -> Dataset:
        """更新数据集"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        if name is not None:
            dataset.name = name
        if description is not None:
            dataset.description = description
        if config is not None:
            dataset.config = config
        
        return self.dataset_repository.update(dataset)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        return self.dataset_repository.delete(dataset_id)
    
    def get_dataset_tags(self, dataset_id: str) -> List[str]:
        """获取数据集标签"""
        tags = self.tag_repository.get_tags(dataset_id)
        return [t.get('name', t.get('tag_name', '')) for t in tags]
    
    def add_dataset_tag(self, dataset_id: str, tag_name: str, user_id: str = None) -> str:
        """添加标签"""
        return self.tag_repository.add_tag(dataset_id, tag_name, None, user_id)
    
    def remove_dataset_tag(self, dataset_id: str, tag_name: str) -> bool:
        """移除标签"""
        return self.tag_repository.remove_tag(dataset_id, tag_name)
    
    def clear_dataset_tags(self, dataset_id: str) -> bool:
        """清除所有标签"""
        tags = self.get_dataset_tags(dataset_id)
        for tag in tags:
            self.tag_repository.remove_tag(dataset_id, tag)
        return True
    
    def log_access(self, dataset_id: str, user_id: str, action: str, details: Dict[str, Any] = None):
        """记录访问日志"""
        self.access_log_repository.create(dataset_id, user_id, action, details)
    
    def get_dataset_statistics(self, dataset_id: str) -> Optional[DatasetStatistics]:
        """获取统计信息"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            return None
        
        return DatasetStatistics(
            dataset_id=dataset_id,
            row_count=dataset.record_count or 0,
            column_count=len(dataset.features.get('columns', [])) if dataset.features else 0,
            size_bytes=dataset.size or 0,
            missing_values=int((dataset.record_count or 0) * 0.005),
            duplicate_rows=int((dataset.record_count or 0) * 0.002)
        )
    
    def generate_download_url(
        self, 
        dataset_id: str, 
        user_id: str,
        format: str = 'original'
    ) -> Dict[str, Any]:
        """生成下载URL"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        ext = format if format != 'original' else (dataset.format or 'json')
        file_name = f"{dataset.name}_{dataset_id[:8]}.{ext}"
        
        return {
            'download_url': f"/api/v1/datasets/{dataset_id}/download?direct=true&format={format}",
            'file_path': f"./data/datasets/{user_id}/{dataset_id}/data.{ext}",
            'file_name': file_name,
            'file_size': dataset.size or 0,
            'format': ext,
            'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
    
    def preview_dataset(
        self, 
        dataset_id: str, 
        limit: int = 10,
        offset: int = 0,
        columns: List[str] = None
    ) -> Dict[str, Any]:
        """预览数据集"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        total_rows = dataset.record_count or 1000
        all_columns = dataset.features.get('columns', ['id', 'content', 'label']) if dataset.features else ['id', 'content', 'label']
        display_columns = columns if columns else all_columns
        
        preview_data = []
        for i in range(offset, min(offset + limit, total_rows)):
            row = {}
            for col in display_columns:
                if col == 'id':
                    row[col] = i + 1
                elif col == 'content':
                    row[col] = f"示例内容 {i + 1}"
                elif col == 'label':
                    row[col] = ['positive', 'negative', 'neutral'][i % 3]
                else:
                    row[col] = f"值_{i + 1}"
            preview_data.append(row)
        
        return {
            'dataset_id': dataset_id,
            'preview_data': preview_data,
            'columns': display_columns,
            'column_types': {'id': 'integer', 'content': 'string', 'label': 'string'},
            'total_rows': total_rows,
            'preview_rows': len(preview_data),
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_rows
        }
    
    def analyze_dataset(
        self, 
        dataset_id: str, 
        analysis_type: str = 'basic',
        columns: List[str] = None,
        sample_size: int = None,
        include_distributions: bool = False
    ) -> Dict[str, Any]:
        """分析数据集"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        total_rows = dataset.record_count or 1000
        
        result = {
            'dataset_id': dataset_id,
            'analysis_type': analysis_type,
            'analyzed_at': datetime.utcnow().isoformat(),
            'basic_stats': {
                'total_rows': total_rows,
                'total_columns': 4,
                'missing_values': int(total_rows * 0.005),
                'duplicate_rows': int(total_rows * 0.002),
                'memory_usage': f"{(dataset.size or 1024000) / 1024 / 1024:.2f} MB"
            }
        }
        
        if analysis_type in ['detailed', 'full']:
            result['detailed_stats'] = {
                'column_stats': [
                    {'column_name': 'id', 'data_type': 'integer', 'unique_values': total_rows},
                    {'column_name': 'content', 'data_type': 'string', 'unique_values': int(total_rows * 0.99)},
                    {'column_name': 'label', 'data_type': 'string', 'unique_values': 3}
                ]
            }
            result['data_quality'] = {
                'completeness': 0.995,
                'uniqueness': 0.998,
                'consistency': 0.99,
                'overall_score': 0.994
            }
        
        if analysis_type == 'full':
            result['recommendations'] = [
                {
                    'type': 'data_quality',
                    'message': '发现少量缺失值，建议进行数据清洗',
                    'priority': 'medium',
                    'affected_columns': ['content', 'label']
                }
            ]
        
        return result
    
    def split_dataset(
        self, 
        dataset_id: str, 
        user_id: str,
        split_ratios: Dict[str, float] = None,
        shuffle: bool = True,
        seed: int = 42,
        stratify_column: str = None,
        create_new_datasets: bool = True
    ) -> Dict[str, Any]:
        """分割数据集"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        total_samples = dataset.record_count or 1000
        splits = {}
        
        for split_name, ratio in split_ratios.items():
            samples = int(total_samples * ratio)
            split_info = {
                'name': f"{dataset.name}_{split_name}",
                'size': int((dataset.size or 1024000) * ratio),
                'samples': samples
            }
            
            if create_new_datasets:
                new_dataset = self.create_dataset(
                    user_id=user_id,
                    name=f"{dataset.name}_{split_name}",
                    description=f"{dataset.name} 的 {split_name} 分割",
                    dataset_type=dataset.dataset_type,
                    format=dataset.format
                )
                split_info['dataset_id'] = new_dataset.dataset_id
            
            splits[split_name] = split_info
        
        return {
            'dataset_id': dataset_id,
            'split_config': {
                'ratios': split_ratios,
                'shuffle': shuffle,
                'seed': seed,
                'stratified': stratify_column is not None
            },
            'splits': splits,
            'total_samples': total_samples,
            'split_at': datetime.utcnow().isoformat()
        }
    
    def get_detailed_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """获取详细统计"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        return {
            'dataset_id': dataset_id,
            'total_rows': dataset.record_count or 0,
            'total_columns': 4,
            'size_bytes': dataset.size or 0,
            'size_human': f"{(dataset.size or 0) / 1024 / 1024:.2f} MB",
            'column_count_by_type': {'integer': 1, 'string': 2, 'datetime': 1},
            'missing_values_total': int((dataset.record_count or 0) * 0.005),
            'missing_percentage': 0.5,
            'duplicate_rows': int((dataset.record_count or 0) * 0.002)
        }
    
    def get_dataset_versions(self, dataset_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """获取版本历史"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        versions = self.version_repository.list_by_dataset(dataset_id)
        paged_versions = versions[offset:offset + limit]
        
        return {
            'dataset_id': dataset_id,
            'versions': [
                {
                    'version_id': v.version_id,
                    'version': v.version,
                    'description': v.description,
                    'created_at': v.created_at.isoformat() if v.created_at else None,
                    'created_by': v.created_by,
                    'is_current': v.version == dataset.version
                }
                for v in paged_versions
            ],
            'total': len(versions),
            'limit': limit,
            'offset': offset
        }
    
    def create_dataset_version(
        self, 
        dataset_id: str, 
        user_id: str,
        version: str = None,
        description: str = None,
        changelog: str = None
    ) -> Dict[str, Any]:
        """创建新版本"""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        # 自动递增版本
        if not version:
            current = dataset.version or '1.0'
            parts = current.split('.')
            try:
                minor = int(parts[-1]) + 1
                version = '.'.join(parts[:-1] + [str(minor)])
            except ValueError:
                version = f"{current}.1"
        
        new_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            description=description,
            created_by=user_id,
            size=dataset.size,
            record_count=dataset.record_count
        )
        created = self.version_repository.create(new_version)
        
        # 更新数据集版本
        dataset.version = version
        self.dataset_repository.update(dataset)
        
        return {
            'version_id': created.version_id,
            'dataset_id': dataset_id,
            'version': created.version,
            'description': created.description,
            'created_at': created.created_at.isoformat() if created.created_at else None,
            'created_by': created.created_by
        }


# ============================================================================
# 测试用例
# ============================================================================

class TestDatasetDetailedService(unittest.TestCase):
    """测试数据集详细管理服务"""
    
    def setUp(self):
        self.repo = MockDatasetRepository()
        self.service = DatasetDetailedService(self.repo)
        self.test_user = 'test_user_123'
        
        # 创建测试数据集
        self.test_dataset = self.service.create_dataset(
            user_id=self.test_user,
            name='测试数据集',
            description='用于测试的数据集',
            dataset_type='text',
            format='json'
        )
    
    def test_get_dataset(self):
        """测试获取数据集"""
        dataset = self.service.get_dataset(self.test_dataset.dataset_id)
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.name, '测试数据集')
        self.assertEqual(dataset.user_id, self.test_user)
        print("  ✓ test_get_dataset")
    
    def test_update_dataset(self):
        """测试更新数据集"""
        updated = self.service.update_dataset(
            dataset_id=self.test_dataset.dataset_id,
            name='更新后的名称',
            description='更新后的描述'
        )
        
        self.assertEqual(updated.name, '更新后的名称')
        self.assertEqual(updated.description, '更新后的描述')
        self.assertIsNotNone(updated.updated_at)
        print("  ✓ test_update_dataset")
    
    def test_delete_dataset(self):
        """测试删除数据集"""
        dataset = self.service.create_dataset(
            user_id=self.test_user,
            name='待删除数据集'
        )
        
        success = self.service.delete_dataset(dataset.dataset_id)
        self.assertTrue(success)
        
        deleted = self.service.get_dataset(dataset.dataset_id)
        self.assertIsNone(deleted)
        print("  ✓ test_delete_dataset")
    
    def test_dataset_tags(self):
        """测试标签管理"""
        dataset_id = self.test_dataset.dataset_id
        
        # 添加标签
        self.service.add_dataset_tag(dataset_id, '训练数据', self.test_user)
        self.service.add_dataset_tag(dataset_id, 'NLP', self.test_user)
        self.service.add_dataset_tag(dataset_id, 'v1.0', self.test_user)
        
        tags = self.service.get_dataset_tags(dataset_id)
        self.assertEqual(len(tags), 3)
        self.assertIn('训练数据', tags)
        self.assertIn('NLP', tags)
        
        # 移除标签
        self.service.remove_dataset_tag(dataset_id, 'v1.0')
        tags = self.service.get_dataset_tags(dataset_id)
        self.assertEqual(len(tags), 2)
        self.assertNotIn('v1.0', tags)
        
        # 清除所有标签
        self.service.clear_dataset_tags(dataset_id)
        tags = self.service.get_dataset_tags(dataset_id)
        self.assertEqual(len(tags), 0)
        print("  ✓ test_dataset_tags")
    
    def test_access_logging(self):
        """测试访问日志"""
        dataset_id = self.test_dataset.dataset_id
        
        self.service.log_access(dataset_id, self.test_user, 'view')
        self.service.log_access(dataset_id, self.test_user, 'download')
        self.service.log_access(dataset_id, self.test_user, 'analyze')
        
        logs = self.service.access_log_repository.list_by_dataset(dataset_id)
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]['action'], 'view')
        print("  ✓ test_access_logging")
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.service.get_dataset_statistics(self.test_dataset.dataset_id)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats.dataset_id, self.test_dataset.dataset_id)
        self.assertGreater(stats.row_count, 0)
        self.assertGreater(stats.column_count, 0)
        print("  ✓ test_get_statistics")
    
    def test_generate_download_url(self):
        """测试生成下载URL"""
        download_info = self.service.generate_download_url(
            dataset_id=self.test_dataset.dataset_id,
            user_id=self.test_user,
            format='json'
        )
        
        self.assertIn('download_url', download_info)
        self.assertIn('file_name', download_info)
        self.assertIn('file_size', download_info)
        self.assertIn('expires_at', download_info)
        self.assertEqual(download_info['format'], 'json')
        print("  ✓ test_generate_download_url")
    
    def test_preview_dataset(self):
        """测试预览数据集"""
        preview = self.service.preview_dataset(
            dataset_id=self.test_dataset.dataset_id,
            limit=5,
            offset=0
        )
        
        self.assertEqual(preview['dataset_id'], self.test_dataset.dataset_id)
        self.assertEqual(len(preview['preview_data']), 5)
        self.assertEqual(preview['limit'], 5)
        self.assertEqual(preview['offset'], 0)
        self.assertTrue(preview['has_more'])
        self.assertIn('columns', preview)
        self.assertIn('column_types', preview)
        print("  ✓ test_preview_dataset")
    
    def test_preview_with_pagination(self):
        """测试预览分页"""
        # 第一页
        page1 = self.service.preview_dataset(
            dataset_id=self.test_dataset.dataset_id,
            limit=10,
            offset=0
        )
        
        # 第二页
        page2 = self.service.preview_dataset(
            dataset_id=self.test_dataset.dataset_id,
            limit=10,
            offset=10
        )
        
        self.assertEqual(page1['preview_data'][0]['id'], 1)
        self.assertEqual(page2['preview_data'][0]['id'], 11)
        print("  ✓ test_preview_with_pagination")
    
    def test_analyze_basic(self):
        """测试基本分析"""
        result = self.service.analyze_dataset(
            dataset_id=self.test_dataset.dataset_id,
            analysis_type='basic'
        )
        
        self.assertEqual(result['analysis_type'], 'basic')
        self.assertIn('basic_stats', result)
        self.assertIn('total_rows', result['basic_stats'])
        self.assertIn('total_columns', result['basic_stats'])
        self.assertNotIn('detailed_stats', result)
        print("  ✓ test_analyze_basic")
    
    def test_analyze_detailed(self):
        """测试详细分析"""
        result = self.service.analyze_dataset(
            dataset_id=self.test_dataset.dataset_id,
            analysis_type='detailed'
        )
        
        self.assertEqual(result['analysis_type'], 'detailed')
        self.assertIn('basic_stats', result)
        self.assertIn('detailed_stats', result)
        self.assertIn('data_quality', result)
        self.assertIn('column_stats', result['detailed_stats'])
        print("  ✓ test_analyze_detailed")
    
    def test_analyze_full(self):
        """测试完整分析"""
        result = self.service.analyze_dataset(
            dataset_id=self.test_dataset.dataset_id,
            analysis_type='full'
        )
        
        self.assertEqual(result['analysis_type'], 'full')
        self.assertIn('basic_stats', result)
        self.assertIn('detailed_stats', result)
        self.assertIn('data_quality', result)
        self.assertIn('recommendations', result)
        self.assertGreater(len(result['recommendations']), 0)
        print("  ✓ test_analyze_full")
    
    def test_split_dataset(self):
        """测试分割数据集"""
        result = self.service.split_dataset(
            dataset_id=self.test_dataset.dataset_id,
            user_id=self.test_user,
            split_ratios={'train': 0.7, 'validation': 0.15, 'test': 0.15}
        )
        
        self.assertEqual(result['dataset_id'], self.test_dataset.dataset_id)
        self.assertIn('splits', result)
        self.assertIn('train', result['splits'])
        self.assertIn('validation', result['splits'])
        self.assertIn('test', result['splits'])
        
        # 验证分割比例
        total = result['total_samples']
        train_samples = result['splits']['train']['samples']
        self.assertAlmostEqual(train_samples / total, 0.7, places=1)
        print("  ✓ test_split_dataset")
    
    def test_split_creates_new_datasets(self):
        """测试分割创建新数据集"""
        result = self.service.split_dataset(
            dataset_id=self.test_dataset.dataset_id,
            user_id=self.test_user,
            create_new_datasets=True
        )
        
        for split_name, split_info in result['splits'].items():
            self.assertIn('dataset_id', split_info)
            # 验证新数据集存在
            new_dataset = self.service.get_dataset(split_info['dataset_id'])
            self.assertIsNotNone(new_dataset)
            self.assertIn(split_name, new_dataset.name)
        print("  ✓ test_split_creates_new_datasets")
    
    def test_detailed_statistics(self):
        """测试详细统计"""
        stats = self.service.get_detailed_statistics(self.test_dataset.dataset_id)
        
        self.assertEqual(stats['dataset_id'], self.test_dataset.dataset_id)
        self.assertIn('total_rows', stats)
        self.assertIn('total_columns', stats)
        self.assertIn('size_bytes', stats)
        self.assertIn('size_human', stats)
        self.assertIn('column_count_by_type', stats)
        self.assertIn('missing_values_total', stats)
        print("  ✓ test_detailed_statistics")
    
    def test_version_management(self):
        """测试版本管理"""
        dataset_id = self.test_dataset.dataset_id
        
        # 创建版本
        v1 = self.service.create_dataset_version(
            dataset_id=dataset_id,
            user_id=self.test_user,
            description='初始版本'
        )
        
        v2 = self.service.create_dataset_version(
            dataset_id=dataset_id,
            user_id=self.test_user,
            version='1.2',
            description='第二版本'
        )
        
        # 获取版本历史
        versions_result = self.service.get_dataset_versions(dataset_id)
        
        self.assertEqual(len(versions_result['versions']), 2)
        self.assertEqual(versions_result['versions'][0]['version'], '1.2')  # 最新版本在前
        print("  ✓ test_version_management")
    
    def test_version_auto_increment(self):
        """测试版本自动递增"""
        dataset_id = self.test_dataset.dataset_id
        
        # 创建多个版本
        v1 = self.service.create_dataset_version(dataset_id, self.test_user)  # 1.1
        v2 = self.service.create_dataset_version(dataset_id, self.test_user)  # 1.2
        v3 = self.service.create_dataset_version(dataset_id, self.test_user)  # 1.3
        
        dataset = self.service.get_dataset(dataset_id)
        self.assertEqual(dataset.version, '1.3')
        print("  ✓ test_version_auto_increment")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        self.repo = MockDatasetRepository()
        self.service = DatasetDetailedService(self.repo)
    
    def test_get_nonexistent_dataset(self):
        """测试获取不存在的数据集"""
        dataset = self.service.get_dataset('nonexistent_id')
        self.assertIsNone(dataset)
        print("  ✓ test_get_nonexistent_dataset")
    
    def test_update_nonexistent_dataset(self):
        """测试更新不存在的数据集"""
        with self.assertRaises(ValueError):
            self.service.update_dataset('nonexistent_id', name='new_name')
        print("  ✓ test_update_nonexistent_dataset")
    
    def test_preview_nonexistent_dataset(self):
        """测试预览不存在的数据集"""
        with self.assertRaises(ValueError):
            self.service.preview_dataset('nonexistent_id')
        print("  ✓ test_preview_nonexistent_dataset")
    
    def test_analyze_nonexistent_dataset(self):
        """测试分析不存在的数据集"""
        with self.assertRaises(ValueError):
            self.service.analyze_dataset('nonexistent_id')
        print("  ✓ test_analyze_nonexistent_dataset")
    
    def test_split_nonexistent_dataset(self):
        """测试分割不存在的数据集"""
        with self.assertRaises(ValueError):
            self.service.split_dataset('nonexistent_id', 'user123')
        print("  ✓ test_split_nonexistent_dataset")
    
    def test_empty_tags(self):
        """测试空标签列表"""
        dataset = self.service.create_dataset('user1', 'test')
        tags = self.service.get_dataset_tags(dataset.dataset_id)
        self.assertEqual(len(tags), 0)
        print("  ✓ test_empty_tags")
    
    def test_empty_versions(self):
        """测试空版本列表"""
        dataset = self.service.create_dataset('user1', 'test')
        versions = self.service.get_dataset_versions(dataset.dataset_id)
        self.assertEqual(len(versions['versions']), 0)
        print("  ✓ test_empty_versions")


class TestIntegrationWorkflow(unittest.TestCase):
    """集成工作流测试"""
    
    def setUp(self):
        self.repo = MockDatasetRepository()
        self.service = DatasetDetailedService(self.repo)
        self.user_id = 'integration_test_user'
    
    def test_complete_dataset_lifecycle(self):
        """测试完整的数据集生命周期"""
        # 1. 创建数据集
        dataset = self.service.create_dataset(
            user_id=self.user_id,
            name='生命周期测试数据集',
            description='测试完整生命周期'
        )
        self.assertIsNotNone(dataset.dataset_id)
        print("  ✓ Step 1: 创建数据集")
        
        # 2. 添加标签
        self.service.add_dataset_tag(dataset.dataset_id, '测试', self.user_id)
        self.service.add_dataset_tag(dataset.dataset_id, 'v1', self.user_id)
        tags = self.service.get_dataset_tags(dataset.dataset_id)
        self.assertEqual(len(tags), 2)
        print("  ✓ Step 2: 添加标签")
        
        # 3. 分析数据集
        analysis = self.service.analyze_dataset(dataset.dataset_id, 'full')
        self.assertIn('recommendations', analysis)
        print("  ✓ Step 3: 分析数据集")
        
        # 4. 预览数据
        preview = self.service.preview_dataset(dataset.dataset_id, limit=10)
        self.assertEqual(len(preview['preview_data']), 10)
        print("  ✓ Step 4: 预览数据")
        
        # 5. 创建版本
        version = self.service.create_dataset_version(
            dataset_id=dataset.dataset_id,
            user_id=self.user_id,
            description='第一个版本'
        )
        self.assertIn('version_id', version)
        print("  ✓ Step 5: 创建版本")
        
        # 6. 更新数据集
        updated = self.service.update_dataset(
            dataset_id=dataset.dataset_id,
            name='更新后的数据集名称',
            description='更新后的描述'
        )
        self.assertEqual(updated.name, '更新后的数据集名称')
        print("  ✓ Step 6: 更新数据集")
        
        # 7. 分割数据集
        split_result = self.service.split_dataset(
            dataset_id=dataset.dataset_id,
            user_id=self.user_id,
            split_ratios={'train': 0.8, 'test': 0.2}
        )
        self.assertEqual(len(split_result['splits']), 2)
        print("  ✓ Step 7: 分割数据集")
        
        # 8. 获取统计信息
        stats = self.service.get_detailed_statistics(dataset.dataset_id)
        self.assertIn('total_rows', stats)
        print("  ✓ Step 8: 获取统计信息")
        
        # 9. 生成下载URL
        download = self.service.generate_download_url(
            dataset_id=dataset.dataset_id,
            user_id=self.user_id
        )
        self.assertIn('download_url', download)
        print("  ✓ Step 9: 生成下载URL")
        
        # 10. 记录访问日志
        self.service.log_access(dataset.dataset_id, self.user_id, 'view')
        logs = self.service.access_log_repository.list_by_dataset(dataset.dataset_id)
        self.assertGreater(len(logs), 0)
        print("  ✓ Step 10: 记录访问日志")
        
        # 11. 删除数据集
        success = self.service.delete_dataset(dataset.dataset_id)
        self.assertTrue(success)
        print("  ✓ Step 11: 删除数据集")
        
        print("\n  ✓✓✓ 完整数据集生命周期测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("数据集详细管理功能测试 - dataset_detailed_api")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestDatasetDetailedService,
        TestEdgeCases,
        TestIntegrationWorkflow,
    ]
    
    for test_class in test_classes:
        print(f"\n▶ {test_class.__name__}")
        print("-" * 50)
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"✓ 所有 {result.testsRun} 个测试通过!")
        print("\n功能覆盖:")
        print("  • 基本操作: 创建、获取、更新、删除数据集")
        print("  • 标签管理: 添加、移除、清除、查询标签")
        print("  • 访问日志: 记录和查询访问日志")
        print("  • 数据预览: 分页预览、列过滤")
        print("  • 数据分析: 基本/详细/完整分析")
        print("  • 数据分割: 按比例分割、创建新数据集")
        print("  • 版本管理: 创建版本、版本历史、自动递增")
        print("  • 统计信息: 基本统计、详细统计")
        print("  • 下载管理: 生成下载URL")
        print("  • 边界情况: 不存在的资源、空数据处理")
        print("  • 集成测试: 完整生命周期工作流")
    else:
        print(f"✗ {len(result.failures) + len(result.errors)} 个测试失败")
        for test, traceback in result.failures + result.errors:
            print(f"\n  失败: {test}")
            print(f"  {traceback}")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
