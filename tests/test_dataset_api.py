"""数据集API测试

测试数据集API的所有功能接口。

测试范围:
    - 基础CRUD操作
    - 状态管理
    - 搜索和统计
    - 批量操作
    - 标签管理
    - 版本管理

运行方式:
    pytest tests/test_dataset_api.py -v
    或
    python tests/test_dataset_api.py
"""

import sys
import os
import unittest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas.dataset import (
    Dataset,
    DatasetVersion,
    CreateDatasetRequest,
    UpdateDatasetRequest,
    DatasetListResponse
)
from backend.repositories.dataset_repository import (
    DatasetRepository,
    DatasetVersionRepository,
    DatasetTagRepository
)
from backend.services.dataset_service import DatasetService
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    DatasetBusinessLogicError
)


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
        
    def test_dataset_process(self):
        """测试数据集处理状态变更"""
        dataset = Dataset(name="test")
        
        self.assertEqual(dataset.status, "pending")
        
        dataset.process()
        
        self.assertEqual(dataset.status, "processing")
        
    def test_dataset_mark_ready(self):
        """测试标记数据集就绪"""
        dataset = Dataset(name="test")
        
        self.assertFalse(dataset.ready)
        
        dataset.mark_ready()
        
        self.assertTrue(dataset.ready)
        self.assertEqual(dataset.status, "ready")
        
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
        
    def test_empty_name(self):
        """测试空名称"""
        request = CreateDatasetRequest(name="")
        
        errors = request.validate()
        
        self.assertIn("数据集名称不能为空", errors)
        
    def test_name_too_long(self):
        """测试名称过长"""
        request = CreateDatasetRequest(name="a" * 201)
        
        errors = request.validate()
        
        self.assertIn("数据集名称长度不能超过200个字符", errors)
        
    def test_invalid_type(self):
        """测试无效类型"""
        request = CreateDatasetRequest(
            name="test",
            dataset_type="invalid_type"
        )
        
        errors = request.validate()
        
        self.assertTrue(any("无效的数据集类型" in e for e in errors))
        
    def test_invalid_format(self):
        """测试无效格式"""
        request = CreateDatasetRequest(
            name="test",
            format="invalid_format"
        )
        
        errors = request.validate()
        
        self.assertTrue(any("无效的数据格式" in e for e in errors))
        
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'name': 'test',
            'type': 'image',
            'format': 'csv'
        }
        
        request = CreateDatasetRequest.from_dict(data)
        
        self.assertEqual(request.name, 'test')
        self.assertEqual(request.dataset_type, 'image')
        self.assertEqual(request.format, 'csv')


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
        
    def test_empty_name_update(self):
        """测试空名称更新"""
        request = UpdateDatasetRequest(name="")
        
        errors = request.validate()
        
        self.assertIn("数据集名称不能为空", errors)
        
    def test_invalid_status(self):
        """测试无效状态"""
        request = UpdateDatasetRequest(status="invalid")
        
        errors = request.validate()
        
        self.assertTrue(any("无效的状态" in e for e in errors))


class TestDatasetServiceUnit(unittest.TestCase):
    """测试数据集服务单元测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_repository = Mock(spec=DatasetRepository)
        self.service = DatasetService(self.mock_repository)
        
    def test_create_dataset_success(self):
        """测试创建数据集成功"""
        # 准备
        expected_dataset = Dataset(
            dataset_id=str(uuid.uuid4()),
            user_id="user123",
            name="test_dataset",
            status="pending"
        )
        self.mock_repository.create.return_value = expected_dataset
        
        # 执行
        with patch.object(self.service, '_log_access'):
            result = self.service.create_dataset(
                user_id="user123",
                name="test_dataset",
                dataset_type="text"
            )
        
        # 验证
        self.mock_repository.create.assert_called_once()
        self.assertEqual(result.name, "test_dataset")
        
    def test_create_dataset_validation_error(self):
        """测试创建数据集验证失败"""
        with self.assertRaises(DatasetValidationError):
            self.service.create_dataset(
                user_id="user123",
                name="",  # 空名称
                dataset_type="text"
            )
            
    def test_get_dataset_exists(self):
        """测试获取存在的数据集"""
        expected_dataset = Dataset(
            dataset_id="test-id",
            user_id="user123",
            name="test_dataset"
        )
        self.mock_repository.get_by_id.return_value = expected_dataset
        
        with patch.object(self.service, '_log_access'):
            result = self.service.get_dataset("test-id")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.dataset_id, "test-id")
        
    def test_get_dataset_not_found(self):
        """测试获取不存在的数据集"""
        self.mock_repository.get_by_id.return_value = None
        
        with patch.object(self.service, '_log_access'):
            result = self.service.get_dataset("non-existent-id")
        
        self.assertIsNone(result)
        
    def test_list_datasets(self):
        """测试获取数据集列表"""
        expected_datasets = [
            Dataset(dataset_id="1", name="ds1", user_id="user123"),
            Dataset(dataset_id="2", name="ds2", user_id="user123")
        ]
        self.mock_repository.list_by_user.return_value = expected_datasets
        
        result = self.service.list_datasets(user_id="user123", limit=10)
        
        self.assertEqual(len(result), 2)
        self.mock_repository.list_by_user.assert_called_once()
        
    def test_list_datasets_invalid_limit(self):
        """测试无效的limit参数"""
        with self.assertRaises(DatasetValidationError):
            self.service.list_datasets(user_id="user123", limit=0)
            
        with self.assertRaises(DatasetValidationError):
            self.service.list_datasets(user_id="user123", limit=101)
            
    def test_list_datasets_invalid_offset(self):
        """测试无效的offset参数"""
        with self.assertRaises(DatasetValidationError):
            self.service.list_datasets(user_id="user123", offset=-1)
            
    def test_update_dataset_success(self):
        """测试更新数据集成功"""
        existing_dataset = Dataset(
            dataset_id="test-id",
            user_id="user123",
            name="old_name"
        )
        updated_dataset = Dataset(
            dataset_id="test-id",
            user_id="user123",
            name="new_name"
        )
        
        self.mock_repository.get_by_id.return_value = existing_dataset
        self.mock_repository.update.return_value = updated_dataset
        
        with patch.object(self.service, '_log_access'):
            result = self.service.update_dataset(
                dataset_id="test-id",
                name="new_name"
            )
        
        self.assertEqual(result.name, "new_name")
        
    def test_update_dataset_not_found(self):
        """测试更新不存在的数据集"""
        self.mock_repository.get_by_id.return_value = None
        
        with self.assertRaises(DatasetNotFoundError):
            self.service.update_dataset(
                dataset_id="non-existent",
                name="new_name"
            )
            
    def test_delete_dataset_success(self):
        """测试删除数据集成功"""
        dataset = Dataset(dataset_id="test-id", status="ready", user_id="user123")
        self.mock_repository.get_by_id.return_value = dataset
        self.mock_repository.delete.return_value = True
        
        with patch.object(self.service, '_log_access'):
            result = self.service.delete_dataset("test-id", "user123")
        
        self.assertTrue(result)
        
    def test_delete_dataset_processing(self):
        """测试删除处理中的数据集"""
        dataset = Dataset(dataset_id="test-id", status="processing", user_id="user123")
        self.mock_repository.get_by_id.return_value = dataset
        
        with self.assertRaises(DatasetBusinessLogicError):
            self.service.delete_dataset("test-id", "user123")
            
    def test_process_dataset(self):
        """测试处理数据集"""
        dataset = Dataset(dataset_id="test-id", status="pending", user_id="user123")
        processed_dataset = Dataset(dataset_id="test-id", status="processing", user_id="user123")
        
        self.mock_repository.get_by_id.return_value = dataset
        self.mock_repository.update.return_value = processed_dataset
        
        with patch.object(self.service, '_log_access'):
            result = self.service.process_dataset("test-id", "user123")
        
        self.assertEqual(result.status, "processing")
        
    def test_process_dataset_already_processing(self):
        """测试处理已在处理中的数据集"""
        dataset = Dataset(dataset_id="test-id", status="processing", user_id="user123")
        self.mock_repository.get_by_id.return_value = dataset
        
        with self.assertRaises(DatasetBusinessLogicError):
            self.service.process_dataset("test-id", "user123")
            
    def test_mark_dataset_ready(self):
        """测试标记数据集就绪"""
        dataset = Dataset(dataset_id="test-id", status="processing", user_id="user123")
        ready_dataset = Dataset(dataset_id="test-id", status="ready", ready=True, user_id="user123")
        
        self.mock_repository.get_by_id.return_value = dataset
        self.mock_repository.update.return_value = ready_dataset
        
        result = self.service.mark_dataset_ready("test-id")
        
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.ready)


class TestDatasetListResponse(unittest.TestCase):
    """测试数据集列表响应"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        datasets = [
            Dataset(dataset_id="1", name="ds1", user_id="user123"),
            Dataset(dataset_id="2", name="ds2", user_id="user123")
        ]
        
        response = DatasetListResponse(
            datasets=datasets,
            total=10,
            limit=50,
            offset=0,
            has_more=True
        )
        
        data = response.to_dict()
        
        self.assertEqual(len(data['datasets']), 2)
        self.assertEqual(data['total'], 10)
        self.assertTrue(data['has_more'])


class TestDatasetVersion(unittest.TestCase):
    """测试数据集版本"""
    
    def test_create_version(self):
        """测试创建版本"""
        version = DatasetVersion(
            dataset_id="ds-123",
            version="1.0",
            description="初始版本",
            created_by="user123"
        )
        
        self.assertEqual(version.version, "1.0")
        self.assertEqual(version.dataset_id, "ds-123")
        
    def test_version_to_dict(self):
        """测试版本转换为字典"""
        version = DatasetVersion(
            version_id="v-123",
            dataset_id="ds-123",
            version="2.0",
            description="更新版本"
        )
        
        data = version.to_dict()
        
        self.assertEqual(data['version'], "2.0")
        self.assertIn('created_at', data)


def run_integration_test():
    """运行集成测试(需要数据库连接)
    
    注意: 此测试需要配置数据库环境
    """
    print("\n" + "=" * 60)
    print("数据集API集成测试")
    print("=" * 60)
    
    try:
        # 初始化服务
        repository = DatasetRepository()
        service = DatasetService(repository)
        
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # 1. 测试创建数据集
        print("\n[1] 测试创建数据集...")
        dataset = service.create_dataset(
            user_id=test_user_id,
            name="integration_test_dataset",
            description="集成测试数据集",
            dataset_type="text",
            format="json"
        )
        print(f"   ✓ 创建成功: {dataset.dataset_id}")
        dataset_id = dataset.dataset_id
        
        # 2. 测试获取数据集
        print("\n[2] 测试获取数据集...")
        fetched = service.get_dataset(dataset_id)
        assert fetched is not None, "获取数据集失败"
        assert fetched.name == "integration_test_dataset"
        print(f"   ✓ 获取成功: {fetched.name}")
        
        # 3. 测试更新数据集
        print("\n[3] 测试更新数据集...")
        updated = service.update_dataset(
            dataset_id=dataset_id,
            name="updated_test_dataset",
            description="更新后的描述"
        )
        assert updated.name == "updated_test_dataset"
        print(f"   ✓ 更新成功: {updated.name}")
        
        # 4. 测试列表查询
        print("\n[4] 测试列表查询...")
        datasets = service.list_datasets(test_user_id, limit=10)
        assert len(datasets) >= 1
        print(f"   ✓ 查询成功: 找到 {len(datasets)} 个数据集")
        
        # 5. 测试处理数据集
        print("\n[5] 测试处理数据集...")
        processed = service.process_dataset(dataset_id)
        assert processed.status == "processing"
        print(f"   ✓ 处理成功: 状态={processed.status}")
        
        # 6. 测试标记就绪
        print("\n[6] 测试标记就绪...")
        ready = service.mark_dataset_ready(dataset_id)
        assert ready.ready == True
        assert ready.status == "ready"
        print(f"   ✓ 标记成功: ready={ready.ready}")
        
        # 7. 测试统计信息
        print("\n[7] 测试统计信息...")
        stats = service.get_statistics(test_user_id)
        print(f"   ✓ 统计成功: total={stats.get('total', 0)}")
        
        # 8. 测试删除数据集
        print("\n[8] 测试删除数据集...")
        deleted = service.delete_dataset(dataset_id)
        assert deleted == True
        print(f"   ✓ 删除成功")
        
        # 验证删除
        verify = service.get_dataset(dataset_id)
        assert verify is None, "数据集应该已被删除"
        print(f"   ✓ 验证删除: 数据集已不存在")
        
        print("\n" + "=" * 60)
        print("✓ 所有集成测试通过!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集API测试')
    parser.add_argument('--integration', action='store_true', 
                        help='运行集成测试(需要数据库)')
    parser.add_argument('--unit', action='store_true',
                        help='只运行单元测试')
    args = parser.parse_args()
    
    if args.integration:
        # 运行集成测试
        run_integration_test()
    elif args.unit:
        # 只运行单元测试
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # 默认运行单元测试
        print("运行单元测试...")
        print("提示: 使用 --integration 参数运行集成测试")
        print()
        unittest.main(argv=[''], exit=False, verbosity=2)
