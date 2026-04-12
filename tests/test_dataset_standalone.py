"""数据集功能独立测试

独立测试数据集DTO和请求模型的功能，不依赖外部模块。
"""

import sys
import os
import unittest
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# 模拟Dataset DTO类进行测试
@dataclass
class Dataset:
    """数据集数据传输对象"""
    dataset_id: str = field(default_factory=lambda: "")
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
    version: Optional[str] = "1.0"
    checksum: Optional[str] = None
    validated: bool = False
    validation_result: Optional[Dict[str, Any]] = None

    @property
    def id(self) -> str:
        return self.dataset_id

    def process(self) -> None:
        self.status = "processing"
        self.updated_at = datetime.utcnow()

    def mark_ready(self) -> None:
        self.ready = True
        self.status = "ready"
        self.updated_at = datetime.utcnow()

    def mark_error(self, error_message: str = None) -> None:
        self.status = "error"
        self.ready = False
        self.updated_at = datetime.utcnow()
        if error_message and self.config is not None:
            self.config['error_message'] = error_message

    def validate_dataset(self, validation_result: Dict[str, Any]) -> None:
        self.validated = True
        self.validation_result = validation_result
        self.updated_at = datetime.utcnow()

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
            "validation_result": self.validation_result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        elif created_at is None:
            created_at = datetime.utcnow()
            
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            
        return cls(
            dataset_id=data.get('dataset_id', ''),
            user_id=data.get('user_id', ''),
            name=data.get('name', ''),
            description=data.get('description'),
            dataset_type=data.get('dataset_type', 'text'),
            format=data.get('format', 'json'),
            storage_path=data.get('storage_path', ''),
            config=data.get('config', {}),
            created_at=created_at,
            updated_at=updated_at,
            status=data.get('status', 'pending'),
            ready=data.get('ready', False),
            size=data.get('size'),
            record_count=data.get('record_count'),
            features=data.get('features'),
            labels=data.get('labels'),
            version=data.get('version', '1.0'),
            checksum=data.get('checksum'),
            validated=data.get('validated', False),
            validation_result=data.get('validation_result')
        )


@dataclass
class CreateDatasetRequest:
    """创建数据集请求模型"""
    name: str
    description: Optional[str] = None
    dataset_type: str = "text"
    format: str = "json"
    storage_path: str = ""
    config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        errors = []
        if not self.name or len(self.name.strip()) == 0:
            errors.append("数据集名称不能为空")
        if len(self.name) > 200:
            errors.append("数据集名称长度不能超过200个字符")
        
        valid_types = ['text', 'image', 'audio', 'video', 'tabular', 'mixed']
        if self.dataset_type not in valid_types:
            errors.append(f"无效的数据集类型，支持: {', '.join(valid_types)}")
            
        valid_formats = ['json', 'csv', 'parquet', 'tfrecord', 'arrow', 'custom']
        if self.format not in valid_formats:
            errors.append(f"无效的数据格式，支持: {', '.join(valid_formats)}")
            
        return errors


@dataclass
class UpdateDatasetRequest:
    """更新数据集请求模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = []
        if self.name is not None:
            if len(self.name.strip()) == 0:
                errors.append("数据集名称不能为空")
            if len(self.name) > 200:
                errors.append("数据集名称长度不能超过200个字符")
                
        if self.status is not None:
            valid_statuses = ['pending', 'uploading', 'processing', 'ready', 'error', 'archived']
            if self.status not in valid_statuses:
                errors.append(f"无效的状态，支持: {', '.join(valid_statuses)}")
                
        return errors


# ============================================================================
# 测试用例
# ============================================================================

class TestDatasetDTO(unittest.TestCase):
    """测试数据集DTO"""
    
    def test_create_dataset_dto(self):
        """测试创建数据集DTO"""
        dataset = Dataset(
            user_id="user123",
            name="test_dataset",
            description="测试数据集",
            dataset_type="text",
            format="json"
        )
        
        self.assertEqual(dataset.name, "test_dataset")
        self.assertEqual(dataset.user_id, "user123")
        self.assertEqual(dataset.dataset_type, "text")
        self.assertEqual(dataset.status, "pending")
        self.assertFalse(dataset.ready)
        print("  ✓ test_create_dataset_dto")
        
    def test_dataset_to_dict(self):
        """测试数据集转换为字典"""
        dataset = Dataset(
            dataset_id="test-id-123",
            user_id="user123",
            name="test_dataset"
        )
        
        data = dataset.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['dataset_id'], "test-id-123")
        self.assertEqual(data['name'], "test_dataset")
        self.assertIn('created_at', data)
        print("  ✓ test_dataset_to_dict")
        
    def test_dataset_from_dict(self):
        """测试从字典创建数据集"""
        data = {
            'dataset_id': 'test-id-123',
            'user_id': 'user123',
            'name': 'test_dataset',
            'dataset_type': 'image',
            'status': 'ready'
        }
        
        dataset = Dataset.from_dict(data)
        
        self.assertEqual(dataset.dataset_id, 'test-id-123')
        self.assertEqual(dataset.dataset_type, 'image')
        self.assertEqual(dataset.status, 'ready')
        print("  ✓ test_dataset_from_dict")
        
    def test_dataset_process(self):
        """测试数据集处理状态变更"""
        dataset = Dataset(name="test")
        
        self.assertEqual(dataset.status, "pending")
        
        dataset.process()
        
        self.assertEqual(dataset.status, "processing")
        self.assertIsNotNone(dataset.updated_at)
        print("  ✓ test_dataset_process")
        
    def test_dataset_mark_ready(self):
        """测试标记数据集就绪"""
        dataset = Dataset(name="test")
        
        self.assertFalse(dataset.ready)
        
        dataset.mark_ready()
        
        self.assertTrue(dataset.ready)
        self.assertEqual(dataset.status, "ready")
        print("  ✓ test_dataset_mark_ready")
        
    def test_dataset_mark_error(self):
        """测试标记数据集错误"""
        dataset = Dataset(name="test")
        
        dataset.mark_error("处理失败")
        
        self.assertEqual(dataset.status, "error")
        self.assertFalse(dataset.ready)
        self.assertEqual(dataset.config.get('error_message'), "处理失败")
        print("  ✓ test_dataset_mark_error")
        
    def test_dataset_validate(self):
        """测试数据集验证"""
        dataset = Dataset(name="test")
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': ['warning1']
        }
        
        dataset.validate_dataset(validation_result)
        
        self.assertTrue(dataset.validated)
        self.assertEqual(dataset.validation_result, validation_result)
        print("  ✓ test_dataset_validate")
        
    def test_dataset_id_property(self):
        """测试id属性"""
        dataset = Dataset(dataset_id="test-123", name="test")
        
        self.assertEqual(dataset.id, "test-123")
        print("  ✓ test_dataset_id_property")


class TestCreateDatasetRequest(unittest.TestCase):
    """测试创建数据集请求"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = CreateDatasetRequest(
            name="valid_name",
            dataset_type="text",
            format="json"
        )
        
        errors = request.validate()
        
        self.assertEqual(len(errors), 0)
        print("  ✓ test_valid_request")
        
    def test_empty_name(self):
        """测试空名称"""
        request = CreateDatasetRequest(name="")
        
        errors = request.validate()
        
        self.assertIn("数据集名称不能为空", errors)
        print("  ✓ test_empty_name")
        
    def test_name_too_long(self):
        """测试名称过长"""
        request = CreateDatasetRequest(name="a" * 201)
        
        errors = request.validate()
        
        self.assertIn("数据集名称长度不能超过200个字符", errors)
        print("  ✓ test_name_too_long")
        
    def test_invalid_type(self):
        """测试无效类型"""
        request = CreateDatasetRequest(
            name="test",
            dataset_type="invalid_type"
        )
        
        errors = request.validate()
        
        self.assertTrue(any("无效的数据集类型" in e for e in errors))
        print("  ✓ test_invalid_type")
        
    def test_invalid_format(self):
        """测试无效格式"""
        request = CreateDatasetRequest(
            name="test",
            format="invalid_format"
        )
        
        errors = request.validate()
        
        self.assertTrue(any("无效的数据格式" in e for e in errors))
        print("  ✓ test_invalid_format")
        
    def test_all_valid_types(self):
        """测试所有有效类型"""
        valid_types = ['text', 'image', 'audio', 'video', 'tabular', 'mixed']
        
        for dtype in valid_types:
            request = CreateDatasetRequest(name="test", dataset_type=dtype)
            errors = request.validate()
            self.assertEqual(len(errors), 0, f"类型 {dtype} 验证失败")
        
        print("  ✓ test_all_valid_types")
        
    def test_all_valid_formats(self):
        """测试所有有效格式"""
        valid_formats = ['json', 'csv', 'parquet', 'tfrecord', 'arrow', 'custom']
        
        for fmt in valid_formats:
            request = CreateDatasetRequest(name="test", format=fmt)
            errors = request.validate()
            self.assertEqual(len(errors), 0, f"格式 {fmt} 验证失败")
        
        print("  ✓ test_all_valid_formats")


class TestUpdateDatasetRequest(unittest.TestCase):
    """测试更新数据集请求"""
    
    def test_valid_update(self):
        """测试有效更新"""
        request = UpdateDatasetRequest(
            name="new_name",
            status="ready"
        )
        
        errors = request.validate()
        
        self.assertEqual(len(errors), 0)
        print("  ✓ test_valid_update")
        
    def test_empty_name_update(self):
        """测试空名称更新"""
        request = UpdateDatasetRequest(name="")
        
        errors = request.validate()
        
        self.assertIn("数据集名称不能为空", errors)
        print("  ✓ test_empty_name_update")
        
    def test_invalid_status(self):
        """测试无效状态"""
        request = UpdateDatasetRequest(status="invalid")
        
        errors = request.validate()
        
        self.assertTrue(any("无效的状态" in e for e in errors))
        print("  ✓ test_invalid_status")
        
    def test_all_valid_statuses(self):
        """测试所有有效状态"""
        valid_statuses = ['pending', 'uploading', 'processing', 'ready', 'error', 'archived']
        
        for status in valid_statuses:
            request = UpdateDatasetRequest(status=status)
            errors = request.validate()
            self.assertEqual(len(errors), 0, f"状态 {status} 验证失败")
        
        print("  ✓ test_all_valid_statuses")
        
    def test_none_values_valid(self):
        """测试None值应该有效"""
        request = UpdateDatasetRequest()
        
        errors = request.validate()
        
        self.assertEqual(len(errors), 0)
        print("  ✓ test_none_values_valid")


class TestDatasetWorkflow(unittest.TestCase):
    """测试数据集工作流"""
    
    def test_complete_workflow(self):
        """测试完整工作流"""
        # 1. 创建
        dataset = Dataset(
            dataset_id="test-123",
            user_id="user123",
            name="workflow_test"
        )
        self.assertEqual(dataset.status, "pending")
        
        # 2. 开始处理
        dataset.process()
        self.assertEqual(dataset.status, "processing")
        
        # 3. 验证
        validation_result = {"is_valid": True, "errors": []}
        dataset.validate_dataset(validation_result)
        self.assertTrue(dataset.validated)
        
        # 4. 标记就绪
        dataset.mark_ready()
        self.assertTrue(dataset.ready)
        self.assertEqual(dataset.status, "ready")
        
        print("  ✓ test_complete_workflow")
        
    def test_error_workflow(self):
        """测试错误工作流"""
        dataset = Dataset(
            dataset_id="test-456",
            user_id="user123",
            name="error_test"
        )
        
        # 开始处理
        dataset.process()
        self.assertEqual(dataset.status, "processing")
        
        # 处理失败
        dataset.mark_error("数据格式错误")
        self.assertEqual(dataset.status, "error")
        self.assertFalse(dataset.ready)
        
        print("  ✓ test_error_workflow")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("数据集API功能测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetDTO))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateDatasetRequest))
    suite.addTests(loader.loadTestsFromTestCase(TestUpdateDatasetRequest))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetWorkflow))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"✓ 所有 {result.testsRun} 个测试通过!")
    else:
        print(f"✗ {len(result.failures) + len(result.errors)} 个测试失败")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}: {traceback}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
