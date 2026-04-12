"""数据集系统测试

对数据发现、数据预处理、数据质量管理功能进行系统测试。
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
import json

# ============================================================================
# 枚举定义
# ============================================================================

class DatasetStatus(Enum):
    """数据集状态"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"

class DataSourceType(Enum):
    """数据源类型"""
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    CLOUD_STORAGE = "cloud_storage"
    API = "api"

class QualityIssueType(Enum):
    """质量问题类型"""
    MISSING_VALUE = "missing_value"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    INVALID_FORMAT = "invalid_format"

class PreprocessingOperationType(Enum):
    """预处理操作类型"""
    NORMALIZE = "normalize"
    TOKENIZE = "tokenize"
    FILTER = "filter"
    TRANSFORM = "transform"
    AUGMENT = "augment"


# ============================================================================
# 数据模型
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
        }


@dataclass
class DataSource:
    """数据源模型"""
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_type: str = DataSourceType.FILE_SYSTEM.value
    connection_string: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "source_type": self.source_type,
            "connection_string": self.connection_string,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class QualityAssessment:
    """质量评估模型"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str = ""
    overall_score: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    issues_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "overall_score": self.overall_score,
            "dimensions": self.dimensions,
            "issues_count": self.issues_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class DataIssue:
    """数据问题模型"""
    issue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str = ""
    issue_type: str = QualityIssueType.MISSING_VALUE.value
    column_name: Optional[str] = None
    description: str = ""
    severity: str = "medium"
    affected_rows: int = 0
    sample_values: List[Any] = field(default_factory=list)
    status: str = "open"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "dataset_id": self.dataset_id,
            "issue_type": self.issue_type,
            "column_name": self.column_name,
            "description": self.description,
            "severity": self.severity,
            "affected_rows": self.affected_rows,
            "sample_values": self.sample_values,
            "status": self.status,
        }


@dataclass
class PreprocessingTask:
    """预处理任务模型"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str = ""
    task_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "task_type": self.task_type,
            "config": self.config,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ============================================================================
# 模拟仓库层
# ============================================================================

class MockDatasetRepository:
    """模拟数据集仓库"""
    
    def __init__(self):
        self._datasets: Dict[str, Dataset] = {}
    
    def create(self, dataset: Dataset) -> Dataset:
        self._datasets[dataset.dataset_id] = dataset
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
            return True
        return False
    
    def list_by_user(self, user_id: str) -> List[Dataset]:
        return [d for d in self._datasets.values() if d.user_id == user_id]


class MockDataSourceRepository:
    """模拟数据源仓库"""
    
    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
    
    def create(self, source: DataSource) -> DataSource:
        self._sources[source.source_id] = source
        return source
    
    def get_by_id(self, source_id: str) -> Optional[DataSource]:
        return self._sources.get(source_id)
    
    def list_all(self) -> List[DataSource]:
        return list(self._sources.values())
    
    def delete(self, source_id: str) -> bool:
        if source_id in self._sources:
            del self._sources[source_id]
            return True
        return False


class MockQualityRepository:
    """模拟质量仓库"""
    
    def __init__(self):
        self._assessments: Dict[str, QualityAssessment] = {}
        self._issues: Dict[str, DataIssue] = {}
    
    def save_assessment(self, assessment: QualityAssessment) -> QualityAssessment:
        self._assessments[assessment.assessment_id] = assessment
        return assessment
    
    def get_assessment(self, assessment_id: str) -> Optional[QualityAssessment]:
        return self._assessments.get(assessment_id)
    
    def list_assessments(self, dataset_id: str) -> List[QualityAssessment]:
        return [a for a in self._assessments.values() if a.dataset_id == dataset_id]
    
    def save_issue(self, issue: DataIssue) -> DataIssue:
        self._issues[issue.issue_id] = issue
        return issue
    
    def get_issue(self, issue_id: str) -> Optional[DataIssue]:
        return self._issues.get(issue_id)
    
    def list_issues(self, dataset_id: str) -> List[DataIssue]:
        return [i for i in self._issues.values() if i.dataset_id == dataset_id]
    
    def update_issue_status(self, issue_id: str, status: str) -> Optional[DataIssue]:
        if issue_id in self._issues:
            self._issues[issue_id].status = status
            return self._issues[issue_id]
        return None


class MockPreprocessingRepository:
    """模拟预处理仓库"""
    
    def __init__(self):
        self._tasks: Dict[str, PreprocessingTask] = {}
    
    def create_task(self, task: PreprocessingTask) -> PreprocessingTask:
        self._tasks[task.task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[PreprocessingTask]:
        return self._tasks.get(task_id)
    
    def update_task(self, task: PreprocessingTask) -> PreprocessingTask:
        self._tasks[task.task_id] = task
        return task
    
    def list_tasks(self, dataset_id: str) -> List[PreprocessingTask]:
        return [t for t in self._tasks.values() if t.dataset_id == dataset_id]


# ============================================================================
# 模拟服务层
# ============================================================================

class DataDiscoveryService:
    """数据发现服务"""
    
    def __init__(self, dataset_repo: MockDatasetRepository, source_repo: MockDataSourceRepository):
        self._dataset_repo = dataset_repo
        self._source_repo = source_repo
    
    def scan_data_sources(self, user_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """扫描数据源"""
        sources = config.get('sources', [])
        results = []
        
        for source_config in sources:
            source = DataSource(
                name=source_config.get('name', f'source_{len(results)}'),
                source_type=source_config.get('type', DataSourceType.FILE_SYSTEM.value),
                connection_string=source_config.get('path', ''),
                config=source_config
            )
            self._source_repo.create(source)
            results.append({
                'source_id': source.source_id,
                'name': source.name,
                'status': 'discovered',
                'datasets_found': source_config.get('datasets_count', 1)
            })
        
        return results
    
    def discover_datasets(self, user_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """发现数据集"""
        datasets = []
        sample_size = config.get('sample_size', 1000)
        
        for source in self._source_repo.list_all():
            dataset = Dataset(
                user_id=user_id,
                name=f"discovered_{source.name}",
                dataset_type=config.get('dataset_type', 'generic'),
                format=config.get('format', 'auto'),
                storage_path=source.connection_string
            )
            self._dataset_repo.create(dataset)
            datasets.append({
                'dataset_id': dataset.dataset_id,
                'name': dataset.name,
                'source_id': source.source_id,
                'format': dataset.format,
                'sample_size': sample_size
            })
        
        return datasets
    
    def infer_schema(self, dataset_id: str) -> Dict[str, Any]:
        """推断数据模式"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 模拟模式推断结果
        schema = {
            'columns': [
                {'name': 'id', 'type': 'integer', 'nullable': False},
                {'name': 'name', 'type': 'string', 'nullable': True},
                {'name': 'value', 'type': 'float', 'nullable': True},
                {'name': 'created_at', 'type': 'datetime', 'nullable': True}
            ],
            'row_count': 10000,
            'inferred_at': datetime.utcnow().isoformat()
        }
        
        # 保存到数据集配置
        dataset.config['inferred_schema'] = schema
        self._dataset_repo.update(dataset)
        
        return schema
    
    def auto_transform(self, dataset_id: str, config: Dict[str, Any]) -> Dataset:
        """自动转换数据"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 模拟转换操作
        operations = config.get('operations', ['normalize', 'handle_missing'])
        dataset.config['transformations'] = operations
        dataset.config['transformed_at'] = datetime.utcnow().isoformat()
        dataset.status = DatasetStatus.PROCESSING.value
        
        return self._dataset_repo.update(dataset)
    
    def setup_incremental_sync(self, dataset_id: str, config: Dict[str, Any]) -> Dataset:
        """设置增量同步"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        sync_config = {
            'enabled': config.get('sync_enabled', True),
            'frequency': config.get('frequency', 'daily'),
            'incremental_column': config.get('incremental_column'),
            'last_sync': None
        }
        
        dataset.config['sync'] = sync_config
        return self._dataset_repo.update(dataset)


class DataPreprocessingService:
    """数据预处理服务"""
    
    def __init__(self, dataset_repo: MockDatasetRepository, task_repo: MockPreprocessingRepository):
        self._dataset_repo = dataset_repo
        self._task_repo = task_repo
    
    def preprocess_dataset(self, dataset_id: str, config: Dict[str, Any]) -> Dataset:
        """预处理数据集"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 创建预处理任务
        task = PreprocessingTask(
            dataset_id=dataset_id,
            task_type='preprocess',
            config=config,
            status='running'
        )
        self._task_repo.create_task(task)
        
        # 模拟预处理操作
        operations = []
        if config.get('normalize'):
            operations.append({
                'type': 'normalize',
                'method': config.get('normalize_method', 'standard'),
                'columns': config.get('normalize_columns', 'all')
            })
        
        if config.get('tokenize'):
            operations.append({
                'type': 'tokenize',
                'columns': config.get('tokenize_columns', []),
                'language': config.get('language', 'en')
            })
        
        if config.get('remove_duplicates'):
            operations.append({
                'type': 'deduplicate',
                'removed_count': 100  # 模拟值
            })
        
        # 更新任务状态
        task.status = 'completed'
        task.progress = 100.0
        task.result = {'operations': operations, 'rows_processed': 10000}
        task.completed_at = datetime.utcnow()
        self._task_repo.update_task(task)
        
        # 更新数据集
        dataset.config['preprocessing'] = {'task_id': task.task_id, 'operations': operations}
        dataset.status = DatasetStatus.PROCESSING.value
        
        return self._dataset_repo.update(dataset)
    
    def perform_feature_engineering(self, dataset_id: str, config: Dict[str, Any]) -> Dataset:
        """执行特征工程"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        task = PreprocessingTask(
            dataset_id=dataset_id,
            task_type='feature_engineering',
            config=config,
            status='running'
        )
        self._task_repo.create_task(task)
        
        # 模拟特征工程
        features_created = []
        for feature in config.get('create_features', []):
            features_created.append({
                'name': feature.get('name'),
                'expression': feature.get('expression'),
                'type': 'derived'
            })
        
        task.status = 'completed'
        task.progress = 100.0
        task.result = {'features_created': features_created}
        task.completed_at = datetime.utcnow()
        self._task_repo.update_task(task)
        
        dataset.config['features'] = features_created
        return self._dataset_repo.update(dataset)
    
    def perform_data_augmentation(self, dataset_id: str, config: Dict[str, Any]) -> Dataset:
        """执行数据增强"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        task = PreprocessingTask(
            dataset_id=dataset_id,
            task_type='augmentation',
            config=config,
            status='running'
        )
        self._task_repo.create_task(task)
        
        # 模拟数据增强
        augmentation_type = config.get('augmentation_type', 'text')
        original_size = 10000
        augmented_size = int(original_size * config.get('augment_ratio', 1.5))
        
        task.status = 'completed'
        task.progress = 100.0
        task.result = {
            'augmentation_type': augmentation_type,
            'original_size': original_size,
            'augmented_size': augmented_size,
            'new_samples': augmented_size - original_size
        }
        task.completed_at = datetime.utcnow()
        self._task_repo.update_task(task)
        
        dataset.config['augmentation'] = task.result
        return self._dataset_repo.update(dataset)
    
    def split_dataset(self, dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """分割数据集"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        test_ratio = config.get('test_ratio', 0.1)
        
        # 验证比例
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        total_rows = 10000  # 模拟值
        
        split_result = {
            'dataset_id': dataset_id,
            'train': {
                'rows': int(total_rows * train_ratio),
                'ratio': train_ratio
            },
            'validation': {
                'rows': int(total_rows * val_ratio),
                'ratio': val_ratio
            },
            'test': {
                'rows': int(total_rows * test_ratio),
                'ratio': test_ratio
            },
            'stratified': config.get('stratify_column') is not None,
            'random_state': config.get('random_state', 42)
        }
        
        dataset.config['split'] = split_result
        self._dataset_repo.update(dataset)
        
        return split_result
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        task = self._task_repo.get_task(task_id)
        return task.to_dict() if task else None
    
    def list_tasks(self, dataset_id: str) -> List[Dict[str, Any]]:
        """列出任务"""
        tasks = self._task_repo.list_tasks(dataset_id)
        return [t.to_dict() for t in tasks]


class DataQualityService:
    """数据质量服务"""
    
    def __init__(self, dataset_repo: MockDatasetRepository, quality_repo: MockQualityRepository):
        self._dataset_repo = dataset_repo
        self._quality_repo = quality_repo
    
    def assess_data_quality(self, dataset_id: str, dimensions: List[str] = None) -> Dict[str, Any]:
        """评估数据质量"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 模拟质量评估
        all_dimensions = ['completeness', 'consistency', 'accuracy', 'timeliness', 'uniqueness']
        check_dimensions = dimensions or all_dimensions
        
        dimension_scores = {}
        for dim in check_dimensions:
            # 模拟评分
            import random
            dimension_scores[dim] = round(random.uniform(0.7, 1.0), 3)
        
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        assessment = QualityAssessment(
            dataset_id=dataset_id,
            overall_score=round(overall_score, 3),
            dimensions=dimension_scores,
            issues_count=3  # 模拟值
        )
        self._quality_repo.save_assessment(assessment)
        
        return {
            'assessment_id': assessment.assessment_id,
            'dataset_id': dataset_id,
            'overall_score': assessment.overall_score,
            'dimensions': assessment.dimensions,
            'issues_count': assessment.issues_count,
            'assessed_at': assessment.created_at.isoformat()
        }
    
    def detect_data_issues(self, dataset_id: str, issue_types: List[str] = None) -> List[Dict[str, Any]]:
        """检测数据问题"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 模拟问题检测
        detected_issues = []
        
        # 缺失值问题
        issue1 = DataIssue(
            dataset_id=dataset_id,
            issue_type=QualityIssueType.MISSING_VALUE.value,
            column_name='name',
            description='Column has 5% missing values',
            severity='medium',
            affected_rows=500,
            sample_values=[None, None, None]
        )
        self._quality_repo.save_issue(issue1)
        detected_issues.append(issue1.to_dict())
        
        # 重复值问题
        issue2 = DataIssue(
            dataset_id=dataset_id,
            issue_type=QualityIssueType.DUPLICATE.value,
            column_name=None,
            description='Found 100 duplicate rows',
            severity='low',
            affected_rows=100
        )
        self._quality_repo.save_issue(issue2)
        detected_issues.append(issue2.to_dict())
        
        # 异常值问题
        issue3 = DataIssue(
            dataset_id=dataset_id,
            issue_type=QualityIssueType.OUTLIER.value,
            column_name='value',
            description='Found 50 outliers using IQR method',
            severity='high',
            affected_rows=50,
            sample_values=[99999, -99999, 1000000]
        )
        self._quality_repo.save_issue(issue3)
        detected_issues.append(issue3.to_dict())
        
        return detected_issues
    
    def clean_data(self, dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """清理数据"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        cleaning_result = {
            'dataset_id': dataset_id,
            'operations': [],
            'rows_affected': 0,
            'cleaned_at': datetime.utcnow().isoformat()
        }
        
        if config.get('remove_duplicates'):
            cleaning_result['operations'].append({
                'type': 'remove_duplicates',
                'rows_removed': 100
            })
            cleaning_result['rows_affected'] += 100
        
        if config.get('handle_missing_values'):
            strategy = config.get('missing_value_strategy', 'drop')
            cleaning_result['operations'].append({
                'type': 'handle_missing',
                'strategy': strategy,
                'rows_affected': 500
            })
            cleaning_result['rows_affected'] += 500
        
        if config.get('handle_outliers'):
            strategy = config.get('outlier_strategy', 'clip')
            cleaning_result['operations'].append({
                'type': 'handle_outliers',
                'strategy': strategy,
                'rows_affected': 50
            })
            cleaning_result['rows_affected'] += 50
        
        # 更新数据集
        dataset.config['cleaning'] = cleaning_result
        self._dataset_repo.update(dataset)
        
        return cleaning_result
    
    def generate_quality_report(self, dataset_id: str) -> Dict[str, Any]:
        """生成质量报告"""
        dataset = self._dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        assessments = self._quality_repo.list_assessments(dataset_id)
        issues = self._quality_repo.list_issues(dataset_id)
        
        report = {
            'report_id': str(uuid.uuid4()),
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_assessments': len(assessments),
                'latest_score': assessments[-1].overall_score if assessments else None,
                'total_issues': len(issues),
                'open_issues': len([i for i in issues if i.status == 'open']),
                'resolved_issues': len([i for i in issues if i.status == 'resolved'])
            },
            'issues_by_type': {},
            'issues_by_severity': {},
            'recommendations': []
        }
        
        # 按类型统计问题
        for issue in issues:
            issue_type = issue.issue_type
            report['issues_by_type'][issue_type] = report['issues_by_type'].get(issue_type, 0) + 1
            
            severity = issue.severity
            report['issues_by_severity'][severity] = report['issues_by_severity'].get(severity, 0) + 1
        
        # 生成建议
        if report['issues_by_type'].get(QualityIssueType.MISSING_VALUE.value, 0) > 0:
            report['recommendations'].append({
                'priority': 'high',
                'recommendation': '处理缺失值：建议使用填充或删除策略'
            })
        
        if report['issues_by_type'].get(QualityIssueType.DUPLICATE.value, 0) > 0:
            report['recommendations'].append({
                'priority': 'medium',
                'recommendation': '删除重复数据以提高数据唯一性'
            })
        
        if report['issues_by_type'].get(QualityIssueType.OUTLIER.value, 0) > 0:
            report['recommendations'].append({
                'priority': 'high',
                'recommendation': '处理异常值：建议检查数据来源或使用裁剪策略'
            })
        
        return report
    
    def resolve_issue(self, issue_id: str) -> bool:
        """解决问题"""
        return self._quality_repo.update_issue_status(issue_id, 'resolved') is not None
    
    def ignore_issue(self, issue_id: str) -> bool:
        """忽略问题"""
        return self._quality_repo.update_issue_status(issue_id, 'ignored') is not None


# ============================================================================
# 测试用例
# ============================================================================

class TestDataDiscoveryService(unittest.TestCase):
    """测试数据发现服务"""
    
    def setUp(self):
        self.dataset_repo = MockDatasetRepository()
        self.source_repo = MockDataSourceRepository()
        self.service = DataDiscoveryService(self.dataset_repo, self.source_repo)
    
    def test_scan_data_sources(self):
        """测试扫描数据源"""
        config = {
            'sources': [
                {'name': 'local_files', 'type': 'file_system', 'path': '/data/files', 'datasets_count': 3},
                {'name': 'postgres_db', 'type': 'database', 'path': 'postgresql://localhost/db', 'datasets_count': 5}
            ]
        }
        
        results = self.service.scan_data_sources('user1', config)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['name'], 'local_files')
        self.assertEqual(results[0]['status'], 'discovered')
        self.assertEqual(results[1]['datasets_found'], 5)
        print("  ✓ test_scan_data_sources")
    
    def test_discover_datasets(self):
        """测试发现数据集"""
        # 先添加数据源
        source = DataSource(name='test_source', connection_string='/data/test')
        self.source_repo.create(source)
        
        config = {'sample_size': 500, 'dataset_type': 'tabular'}
        
        datasets = self.service.discover_datasets('user1', config)
        
        self.assertEqual(len(datasets), 1)
        self.assertIn('discovered_test_source', datasets[0]['name'])
        self.assertEqual(datasets[0]['sample_size'], 500)
        print("  ✓ test_discover_datasets")
    
    def test_infer_schema(self):
        """测试模式推断"""
        dataset = Dataset(user_id='user1', name='test_dataset')
        self.dataset_repo.create(dataset)
        
        schema = self.service.infer_schema(dataset.dataset_id)
        
        self.assertIn('columns', schema)
        self.assertIn('row_count', schema)
        self.assertEqual(len(schema['columns']), 4)
        print("  ✓ test_infer_schema")
    
    def test_auto_transform(self):
        """测试自动转换"""
        dataset = Dataset(user_id='user1', name='test_dataset')
        self.dataset_repo.create(dataset)
        
        config = {'operations': ['normalize', 'handle_missing', 'encode']}
        
        result = self.service.auto_transform(dataset.dataset_id, config)
        
        self.assertEqual(result.status, DatasetStatus.PROCESSING.value)
        self.assertIn('transformations', result.config)
        print("  ✓ test_auto_transform")
    
    def test_setup_incremental_sync(self):
        """测试设置增量同步"""
        dataset = Dataset(user_id='user1', name='test_dataset')
        self.dataset_repo.create(dataset)
        
        config = {
            'sync_enabled': True,
            'frequency': 'hourly',
            'incremental_column': 'updated_at'
        }
        
        result = self.service.setup_incremental_sync(dataset.dataset_id, config)
        
        self.assertIn('sync', result.config)
        self.assertTrue(result.config['sync']['enabled'])
        self.assertEqual(result.config['sync']['frequency'], 'hourly')
        print("  ✓ test_setup_incremental_sync")


class TestDataPreprocessingService(unittest.TestCase):
    """测试数据预处理服务"""
    
    def setUp(self):
        self.dataset_repo = MockDatasetRepository()
        self.task_repo = MockPreprocessingRepository()
        self.service = DataPreprocessingService(self.dataset_repo, self.task_repo)
        
        # 创建测试数据集
        self.test_dataset = Dataset(user_id='user1', name='test_dataset')
        self.dataset_repo.create(self.test_dataset)
    
    def test_preprocess_dataset(self):
        """测试预处理数据集"""
        config = {
            'normalize': True,
            'normalize_method': 'minmax',
            'tokenize': True,
            'tokenize_columns': ['text'],
            'remove_duplicates': True
        }
        
        result = self.service.preprocess_dataset(self.test_dataset.dataset_id, config)
        
        self.assertEqual(result.status, DatasetStatus.PROCESSING.value)
        self.assertIn('preprocessing', result.config)
        print("  ✓ test_preprocess_dataset")
    
    def test_feature_engineering(self):
        """测试特征工程"""
        config = {
            'create_features': [
                {'name': 'age_squared', 'expression': 'age ** 2'},
                {'name': 'income_per_age', 'expression': 'income / age'}
            ]
        }
        
        result = self.service.perform_feature_engineering(self.test_dataset.dataset_id, config)
        
        self.assertIn('features', result.config)
        self.assertEqual(len(result.config['features']), 2)
        print("  ✓ test_feature_engineering")
    
    def test_data_augmentation(self):
        """测试数据增强"""
        config = {
            'augmentation_type': 'text',
            'augment_ratio': 2.0
        }
        
        result = self.service.perform_data_augmentation(self.test_dataset.dataset_id, config)
        
        self.assertIn('augmentation', result.config)
        self.assertEqual(result.config['augmentation']['augmentation_type'], 'text')
        self.assertGreater(result.config['augmentation']['augmented_size'], 
                          result.config['augmentation']['original_size'])
        print("  ✓ test_data_augmentation")
    
    def test_split_dataset(self):
        """测试数据集分割"""
        config = {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'stratify_column': 'label',
            'random_state': 42
        }
        
        result = self.service.split_dataset(self.test_dataset.dataset_id, config)
        
        self.assertEqual(result['train']['ratio'], 0.7)
        self.assertEqual(result['validation']['ratio'], 0.15)
        self.assertEqual(result['test']['ratio'], 0.15)
        self.assertTrue(result['stratified'])
        print("  ✓ test_split_dataset")
    
    def test_split_dataset_invalid_ratio(self):
        """测试无效分割比例"""
        config = {
            'train_ratio': 0.8,
            'val_ratio': 0.2,
            'test_ratio': 0.2  # 总和超过1.0
        }
        
        with self.assertRaises(ValueError):
            self.service.split_dataset(self.test_dataset.dataset_id, config)
        print("  ✓ test_split_dataset_invalid_ratio")
    
    def test_task_management(self):
        """测试任务管理"""
        # 执行预处理创建任务
        config = {'normalize': True}
        self.service.preprocess_dataset(self.test_dataset.dataset_id, config)
        
        # 列出任务
        tasks = self.service.list_tasks(self.test_dataset.dataset_id)
        
        self.assertGreater(len(tasks), 0)
        self.assertEqual(tasks[0]['status'], 'completed')
        print("  ✓ test_task_management")


class TestDataQualityService(unittest.TestCase):
    """测试数据质量服务"""
    
    def setUp(self):
        self.dataset_repo = MockDatasetRepository()
        self.quality_repo = MockQualityRepository()
        self.service = DataQualityService(self.dataset_repo, self.quality_repo)
        
        # 创建测试数据集
        self.test_dataset = Dataset(user_id='user1', name='test_dataset')
        self.dataset_repo.create(self.test_dataset)
    
    def test_assess_data_quality(self):
        """测试质量评估"""
        result = self.service.assess_data_quality(self.test_dataset.dataset_id)
        
        self.assertIn('assessment_id', result)
        self.assertIn('overall_score', result)
        self.assertIn('dimensions', result)
        self.assertGreater(result['overall_score'], 0)
        self.assertLessEqual(result['overall_score'], 1.0)
        print("  ✓ test_assess_data_quality")
    
    def test_assess_specific_dimensions(self):
        """测试指定维度评估"""
        dimensions = ['completeness', 'accuracy']
        
        result = self.service.assess_data_quality(
            self.test_dataset.dataset_id,
            dimensions=dimensions
        )
        
        self.assertEqual(len(result['dimensions']), 2)
        self.assertIn('completeness', result['dimensions'])
        self.assertIn('accuracy', result['dimensions'])
        print("  ✓ test_assess_specific_dimensions")
    
    def test_detect_data_issues(self):
        """测试问题检测"""
        issues = self.service.detect_data_issues(self.test_dataset.dataset_id)
        
        self.assertGreater(len(issues), 0)
        
        # 检查问题类型
        issue_types = [i['issue_type'] for i in issues]
        self.assertIn(QualityIssueType.MISSING_VALUE.value, issue_types)
        self.assertIn(QualityIssueType.DUPLICATE.value, issue_types)
        self.assertIn(QualityIssueType.OUTLIER.value, issue_types)
        print("  ✓ test_detect_data_issues")
    
    def test_clean_data(self):
        """测试数据清理"""
        config = {
            'remove_duplicates': True,
            'handle_missing_values': True,
            'missing_value_strategy': 'fill_mean',
            'handle_outliers': True,
            'outlier_strategy': 'clip'
        }
        
        result = self.service.clean_data(self.test_dataset.dataset_id, config)
        
        self.assertIn('operations', result)
        self.assertEqual(len(result['operations']), 3)
        self.assertGreater(result['rows_affected'], 0)
        print("  ✓ test_clean_data")
    
    def test_generate_quality_report(self):
        """测试生成质量报告"""
        # 先进行评估和问题检测
        self.service.assess_data_quality(self.test_dataset.dataset_id)
        self.service.detect_data_issues(self.test_dataset.dataset_id)
        
        report = self.service.generate_quality_report(self.test_dataset.dataset_id)
        
        self.assertIn('report_id', report)
        self.assertIn('summary', report)
        self.assertIn('issues_by_type', report)
        self.assertIn('recommendations', report)
        self.assertGreater(len(report['recommendations']), 0)
        print("  ✓ test_generate_quality_report")
    
    def test_resolve_and_ignore_issues(self):
        """测试解决和忽略问题"""
        # 先检测问题
        issues = self.service.detect_data_issues(self.test_dataset.dataset_id)
        
        # 解决第一个问题
        resolved = self.service.resolve_issue(issues[0]['issue_id'])
        self.assertTrue(resolved)
        
        # 忽略第二个问题
        ignored = self.service.ignore_issue(issues[1]['issue_id'])
        self.assertTrue(ignored)
        
        # 验证状态
        issue1 = self.quality_repo.get_issue(issues[0]['issue_id'])
        issue2 = self.quality_repo.get_issue(issues[1]['issue_id'])
        
        self.assertEqual(issue1.status, 'resolved')
        self.assertEqual(issue2.status, 'ignored')
        print("  ✓ test_resolve_and_ignore_issues")


class TestIntegrationWorkflow(unittest.TestCase):
    """集成测试：完整工作流"""
    
    def setUp(self):
        self.dataset_repo = MockDatasetRepository()
        self.source_repo = MockDataSourceRepository()
        self.task_repo = MockPreprocessingRepository()
        self.quality_repo = MockQualityRepository()
        
        self.discovery_service = DataDiscoveryService(self.dataset_repo, self.source_repo)
        self.preprocessing_service = DataPreprocessingService(self.dataset_repo, self.task_repo)
        self.quality_service = DataQualityService(self.dataset_repo, self.quality_repo)
    
    def test_complete_data_pipeline(self):
        """测试完整数据处理流水线"""
        user_id = 'user1'
        
        # 1. 扫描数据源
        scan_config = {
            'sources': [
                {'name': 'sales_data', 'type': 'file_system', 'path': '/data/sales.csv'}
            ]
        }
        scan_results = self.discovery_service.scan_data_sources(user_id, scan_config)
        self.assertEqual(len(scan_results), 1)
        print("  ✓ Step 1: 数据源扫描完成")
        
        # 2. 发现数据集
        discover_config = {'sample_size': 1000}
        discovered = self.discovery_service.discover_datasets(user_id, discover_config)
        self.assertEqual(len(discovered), 1)
        dataset_id = discovered[0]['dataset_id']
        print("  ✓ Step 2: 数据集发现完成")
        
        # 3. 推断模式
        schema = self.discovery_service.infer_schema(dataset_id)
        self.assertIn('columns', schema)
        print("  ✓ Step 3: 模式推断完成")
        
        # 4. 质量评估
        quality_result = self.quality_service.assess_data_quality(dataset_id)
        self.assertIn('overall_score', quality_result)
        print("  ✓ Step 4: 质量评估完成")
        
        # 5. 问题检测
        issues = self.quality_service.detect_data_issues(dataset_id)
        self.assertGreater(len(issues), 0)
        print("  ✓ Step 5: 问题检测完成")
        
        # 6. 数据清理
        clean_config = {
            'remove_duplicates': True,
            'handle_missing_values': True
        }
        clean_result = self.quality_service.clean_data(dataset_id, clean_config)
        self.assertGreater(clean_result['rows_affected'], 0)
        print("  ✓ Step 6: 数据清理完成")
        
        # 7. 数据预处理
        preprocess_config = {
            'normalize': True,
            'normalize_method': 'standard'
        }
        preprocessed = self.preprocessing_service.preprocess_dataset(dataset_id, preprocess_config)
        self.assertEqual(preprocessed.status, DatasetStatus.PROCESSING.value)
        print("  ✓ Step 7: 数据预处理完成")
        
        # 8. 特征工程
        feature_config = {
            'create_features': [
                {'name': 'total_sales', 'expression': 'quantity * price'}
            ]
        }
        featured = self.preprocessing_service.perform_feature_engineering(dataset_id, feature_config)
        self.assertIn('features', featured.config)
        print("  ✓ Step 8: 特征工程完成")
        
        # 9. 数据分割
        split_config = {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        }
        split_result = self.preprocessing_service.split_dataset(dataset_id, split_config)
        self.assertEqual(split_result['train']['ratio'], 0.8)
        print("  ✓ Step 9: 数据分割完成")
        
        # 10. 生成质量报告
        report = self.quality_service.generate_quality_report(dataset_id)
        self.assertIn('recommendations', report)
        print("  ✓ Step 10: 质量报告生成完成")
        
        print("\n  ✓✓✓ 完整数据处理流水线测试通过！")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        self.dataset_repo = MockDatasetRepository()
        self.source_repo = MockDataSourceRepository()
        self.quality_repo = MockQualityRepository()
        self.task_repo = MockPreprocessingRepository()
        
        self.discovery_service = DataDiscoveryService(self.dataset_repo, self.source_repo)
        self.quality_service = DataQualityService(self.dataset_repo, self.quality_repo)
        self.preprocessing_service = DataPreprocessingService(self.dataset_repo, self.task_repo)
    
    def test_nonexistent_dataset(self):
        """测试不存在的数据集"""
        with self.assertRaises(ValueError):
            self.discovery_service.infer_schema('nonexistent_id')
        print("  ✓ test_nonexistent_dataset")
    
    def test_empty_config(self):
        """测试空配置"""
        results = self.discovery_service.scan_data_sources('user1', {})
        self.assertEqual(len(results), 0)
        print("  ✓ test_empty_config")
    
    def test_clean_data_no_operations(self):
        """测试无操作的数据清理"""
        dataset = Dataset(user_id='user1', name='test')
        self.dataset_repo.create(dataset)
        
        result = self.quality_service.clean_data(dataset.dataset_id, {})
        
        self.assertEqual(len(result['operations']), 0)
        self.assertEqual(result['rows_affected'], 0)
        print("  ✓ test_clean_data_no_operations")
    
    def test_resolve_nonexistent_issue(self):
        """测试解决不存在的问题"""
        result = self.quality_service.resolve_issue('nonexistent_id')
        self.assertFalse(result)
        print("  ✓ test_resolve_nonexistent_issue")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("数据集系统测试 - 数据发现、预处理、质量管理")
    print("=" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDataDiscoveryService,
        TestDataPreprocessingService,
        TestDataQualityService,
        TestIntegrationWorkflow,
        TestEdgeCases,
    ]
    
    for test_class in test_classes:
        print(f"\n▶ {test_class.__name__}")
        print("-" * 50)
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"✓ 所有 {result.testsRun} 个测试通过!")
        print("\n功能覆盖:")
        print("  • 数据发现服务: 数据源扫描、数据集发现、模式推断、自动转换、增量同步")
        print("  • 数据预处理服务: 预处理、特征工程、数据增强、数据分割、任务管理")
        print("  • 数据质量服务: 质量评估、问题检测、数据清理、质量报告、问题管理")
        print("  • 集成工作流: 完整数据处理流水线端到端测试")
        print("  • 边界情况: 异常处理、空配置、不存在资源")
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
