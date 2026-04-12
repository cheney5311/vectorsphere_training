"""数据集管理API测试

测试 dataset_management_api.py 的所有功能接口。
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import uuid

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatasetManagementService(unittest.TestCase):
    """测试数据集管理服务"""
    
    def setUp(self):
        """测试准备"""
        # 模拟 dataset 对象
        self.mock_dataset = Mock()
        self.mock_dataset.dataset_id = str(uuid.uuid4())
        self.mock_dataset.user_id = "test_user_123"
        self.mock_dataset.name = "测试数据集"
        self.mock_dataset.description = "测试描述"
        self.mock_dataset.dataset_type = "text"
        self.mock_dataset.format = "json"
        self.mock_dataset.status = "ready"
        self.mock_dataset.size = 1024000
        self.mock_dataset.record_count = 1000
        self.mock_dataset.storage_path = "/data/test"
        self.mock_dataset.version = "1.0"
        self.mock_dataset.ready = True
        self.mock_dataset.validated = True
        self.mock_dataset.features = {"columns": ["id", "content", "label"]}
        self.mock_dataset.config = {}
        self.mock_dataset.created_at = datetime.utcnow()
        self.mock_dataset.updated_at = datetime.utcnow()
        self.mock_dataset.checksum = "abc123"
        
        self.mock_dataset.to_dict = Mock(return_value={
            "dataset_id": self.mock_dataset.dataset_id,
            "user_id": self.mock_dataset.user_id,
            "name": self.mock_dataset.name,
            "description": self.mock_dataset.description,
            "dataset_type": self.mock_dataset.dataset_type,
            "format": self.mock_dataset.format,
            "status": self.mock_dataset.status,
            "size": self.mock_dataset.size,
            "record_count": self.mock_dataset.record_count,
            "created_at": self.mock_dataset.created_at.isoformat(),
            "updated_at": self.mock_dataset.updated_at.isoformat()
        })
        
        # 模拟仓库
        self.mock_repository = Mock()
        self.mock_repository.get_by_id.return_value = self.mock_dataset
        self.mock_repository.create.return_value = self.mock_dataset
        self.mock_repository.update.return_value = self.mock_dataset
        self.mock_repository.delete.return_value = True
        self.mock_repository.list_by_user.return_value = [self.mock_dataset]
        self.mock_repository.count_by_user.return_value = 1
        self.mock_repository.exists.return_value = True
        
    def test_create_dataset_service(self):
        """测试创建数据集服务"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        
        # 验证初始化
        self.assertIsNotNone(service.dataset_repository)
        self.assertEqual(service.dataset_repository, self.mock_repository)
        
    def test_get_user_statistics(self):
        """测试获取用户统计信息"""
        from backend.services.dataset_service import DatasetService
        
        # 创建多个模拟数据集
        datasets = []
        for i in range(5):
            ds = Mock()
            ds.dataset_id = str(uuid.uuid4())
            ds.user_id = "test_user"
            ds.name = f"数据集_{i}"
            ds.dataset_type = ["text", "image", "tabular"][i % 3]
            ds.status = ["ready", "processing", "pending"][i % 3]
            ds.size = 1024 * 1024 * (i + 1)
            ds.record_count = 1000 * (i + 1)
            ds.created_at = datetime.utcnow() - timedelta(days=i)
            datasets.append(ds)
        
        self.mock_repository.list_by_user.return_value = datasets
        
        service = DatasetService(self.mock_repository)
        stats = service.get_user_statistics("test_user")
        
        # 验证统计信息
        self.assertEqual(stats['total_datasets'], 5)
        self.assertIn('total_size_bytes', stats)
        self.assertIn('total_size_human', stats)
        self.assertIn('datasets_by_type', stats)
        self.assertIn('datasets_by_status', stats)
        self.assertIn('recent_uploads', stats)
        
    def test_search_datasets(self):
        """测试搜索数据集"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        
        # 模拟标签仓库
        service.tag_repository = Mock()
        service.tag_repository.get_tags.return_value = [{"name": "NLP"}]
        
        result = service.search_datasets(
            user_id="test_user",
            dataset_type="text",
            status="ready",
            search="测试",
            limit=10,
            offset=0
        )
        
        # 验证结果
        self.assertIn('datasets', result)
        self.assertIn('total_count', result)
        self.assertIn('filtered_count', result)
        self.assertIn('has_more', result)
        
    def test_advanced_search(self):
        """测试高级搜索"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        service.tag_repository = Mock()
        service.tag_repository.get_tags.return_value = []
        
        search_params = {
            'q': '测试',
            'type': 'text',
            'status': 'ready',
            'min_size': 1000,
            'max_size': 10000000
        }
        
        result = service.advanced_search(
            user_id="test_user",
            search_params=search_params,
            limit=20
        )
        
        # 验证结果
        self.assertIn('datasets', result)
        self.assertIn('total_count', result)
        self.assertIn('query', result)
        self.assertEqual(result['query'], search_params)
        
    def test_transfer_dataset(self):
        """测试转移数据集所有权"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        service.access_log_repository = Mock()
        service.access_log_repository.create.return_value = None
        
        result = service.transfer_dataset(
            dataset_id=self.mock_dataset.dataset_id,
            from_user_id="test_user_123",
            to_user_id="new_user_456"
        )
        
        # 验证更新被调用
        self.mock_repository.update.assert_called_once()
        
        # 验证用户ID被更新
        updated_dataset = self.mock_repository.update.call_args[0][0]
        self.assertEqual(updated_dataset.user_id, "new_user_456")
        
    def test_transfer_dataset_to_self(self):
        """测试转移给自己应该失败"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        
        with self.assertRaises(ValueError):
            service.transfer_dataset(
                dataset_id=self.mock_dataset.dataset_id,
                from_user_id="test_user_123",
                to_user_id="test_user_123"  # 相同用户
            )
            
    def test_merge_datasets(self):
        """测试合并数据集"""
        from backend.services.dataset_service import DatasetService
        
        # 创建源数据集
        source_dataset = Mock()
        source_dataset.dataset_id = str(uuid.uuid4())
        source_dataset.user_id = "test_user"
        source_dataset.name = "源数据集"
        source_dataset.dataset_type = "text"
        source_dataset.format = "json"
        source_dataset.size = 512000
        source_dataset.record_count = 500
        
        # 设置 get_by_id 返回不同的数据集
        def get_by_id_side_effect(dataset_id):
            if dataset_id == self.mock_dataset.dataset_id:
                return self.mock_dataset
            elif dataset_id == source_dataset.dataset_id:
                return source_dataset
            return None
        
        self.mock_repository.get_by_id.side_effect = get_by_id_side_effect
        
        service = DatasetService(self.mock_repository)
        service.access_log_repository = Mock()
        service.access_log_repository.create.return_value = None
        
        result = service.merge_datasets(
            target_dataset_id=self.mock_dataset.dataset_id,
            source_dataset_ids=[source_dataset.dataset_id],
            user_id="test_user"
        )
        
        # 验证结果
        self.assertEqual(result['target_dataset_id'], self.mock_dataset.dataset_id)
        self.assertEqual(result['merged_count'], 1)
        self.assertIn('total_records', result)
        self.assertIn('merged_at', result)
        
    def test_restore_dataset(self):
        """测试恢复归档数据集"""
        from backend.services.dataset_service import DatasetService
        
        # 设置数据集为归档状态
        self.mock_dataset.status = "archived"
        
        service = DatasetService(self.mock_repository)
        service.access_log_repository = Mock()
        service.access_log_repository.create.return_value = None
        
        result = service.restore_dataset(
            dataset_id=self.mock_dataset.dataset_id,
            user_id="test_user"
        )
        
        # 验证更新被调用
        self.mock_repository.update.assert_called()
        
    def test_restore_non_archived_dataset(self):
        """测试恢复非归档数据集应该失败"""
        from backend.services.dataset_service import DatasetService
        from backend.modules.dataset.dataset_exceptions import DatasetBusinessLogicError
        
        # 设置数据集为非归档状态
        self.mock_dataset.status = "ready"
        
        service = DatasetService(self.mock_repository)
        
        with self.assertRaises(DatasetBusinessLogicError):
            service.restore_dataset(
                dataset_id=self.mock_dataset.dataset_id,
                user_id="test_user"
            )
            
    def test_get_recent_datasets(self):
        """测试获取最近访问的数据集"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        
        # 模拟访问日志仓库
        service.access_log_repository = Mock()
        service.access_log_repository.get_recent_by_user.return_value = [
            {
                'dataset_id': self.mock_dataset.dataset_id,
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        result = service.get_recent_datasets("test_user_123", limit=5)
        
        # 验证结果
        self.assertIsInstance(result, list)
        
    def test_clone_dataset(self):
        """测试克隆数据集"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        service.access_log_repository = Mock()
        service.access_log_repository.create.return_value = None
        
        result = service.clone_dataset(
            source_dataset_id=self.mock_dataset.dataset_id,
            user_id="test_user",
            new_name="克隆的数据集"
        )
        
        # 验证 create 被调用
        self.mock_repository.create.assert_called()
        
    def test_generate_download_url(self):
        """测试生成下载URL"""
        from backend.services.dataset_service import DatasetService
        
        service = DatasetService(self.mock_repository)
        
        result = service.generate_download_url(
            dataset_id=self.mock_dataset.dataset_id,
            user_id="test_user",
            format="json"
        )
        
        # 验证结果
        self.assertIn('download_url', result)
        self.assertIn('file_name', result)
        self.assertIn('file_size', result)
        self.assertIn('expires_at', result)


class TestDatasetManagementServiceIntegration(unittest.TestCase):
    """数据集管理服务集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        from backend.services.dataset_service import DatasetService
        
        # 模拟仓库
        mock_repo = Mock()
        datasets = {}
        
        def create_dataset(dataset):
            dataset.dataset_id = str(uuid.uuid4())
            datasets[dataset.dataset_id] = dataset
            return dataset
        
        def get_by_id(dataset_id):
            return datasets.get(dataset_id)
        
        def update_dataset(dataset):
            datasets[dataset.dataset_id] = dataset
            return dataset
        
        def delete_dataset(dataset_id):
            if dataset_id in datasets:
                del datasets[dataset_id]
                return True
            return False
        
        def list_by_user(user_id, **kwargs):
            return [ds for ds in datasets.values() if ds.user_id == user_id]
        
        mock_repo.create.side_effect = create_dataset
        mock_repo.get_by_id.side_effect = get_by_id
        mock_repo.update.side_effect = update_dataset
        mock_repo.delete.side_effect = delete_dataset
        mock_repo.list_by_user.side_effect = list_by_user
        mock_repo.exists.return_value = True
        
        service = DatasetService(mock_repo)
        service.tag_repository = Mock()
        service.tag_repository.add_tag.return_value = "tag_id"
        service.tag_repository.get_tags.return_value = []
        service.access_log_repository = Mock()
        service.access_log_repository.create.return_value = None
        service.access_log_repository.get_recent_by_user.return_value = []
        service.version_repository = Mock()
        service.version_repository.list_by_dataset.return_value = []
        
        # 1. 创建数据集
        user_id = "integration_test_user"
        
        # 模拟 Dataset 类
        with patch('backend.services.dataset_service.Dataset') as MockDataset:
            mock_ds = Mock()
            mock_ds.dataset_id = str(uuid.uuid4())
            mock_ds.user_id = user_id
            mock_ds.name = "集成测试数据集"
            mock_ds.description = "测试描述"
            mock_ds.dataset_type = "text"
            mock_ds.format = "json"
            mock_ds.status = "pending"
            mock_ds.ready = False
            mock_ds.config = {}
            mock_ds.features = {}
            mock_ds.size = 0
            mock_ds.record_count = 0
            mock_ds.storage_path = ""
            mock_ds.version = "1.0"
            mock_ds.created_at = datetime.utcnow()
            mock_ds.updated_at = datetime.utcnow()
            mock_ds.to_dict = Mock(return_value={
                "dataset_id": mock_ds.dataset_id,
                "name": mock_ds.name,
                "status": mock_ds.status
            })
            
            MockDataset.return_value = mock_ds
            
            # 模拟 CreateDatasetRequest
            with patch('backend.services.dataset_service.CreateDatasetRequest') as MockRequest:
                mock_request = Mock()
                mock_request.validate.return_value = []
                MockRequest.return_value = mock_request
                
                dataset = service.create_dataset(
                    user_id=user_id,
                    name="集成测试数据集",
                    description="测试描述",
                    dataset_type="text",
                    format="json"
                )
                
                self.assertIsNotNone(dataset)
                self.assertEqual(dataset.name, "集成测试数据集")
        
        print("✓ 完整工作流测试通过")


class TestDatasetManagementAPIRoutes(unittest.TestCase):
    """测试API路由定义"""
    
    def test_api_endpoints_defined(self):
        """测试API端点已定义"""
        from backend.api.dataset.dataset_management_api import dataset_management_bp
        
        # 验证蓝图存在
        self.assertIsNotNone(dataset_management_bp)
        self.assertEqual(dataset_management_bp.url_prefix, '/api/v1/datasets')
        
        # 获取所有路由规则
        # 注意：蓝图需要注册到应用才能完全检查路由
        print("✓ API蓝图配置正确")
        
    def test_blueprint_name(self):
        """测试蓝图名称"""
        from backend.api.dataset.dataset_management_api import dataset_management_bp
        
        self.assertEqual(dataset_management_bp.name, 'dataset_management')


def run_service_tests():
    """运行服务层测试"""
    print("\n" + "=" * 60)
    print("数据集管理服务测试")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetManagementService))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetManagementServiceIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetManagementAPIRoutes))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_functional_tests():
    """运行功能测试"""
    print("\n" + "=" * 60)
    print("功能测试")
    print("=" * 60)
    
    # 测试1: 验证服务初始化
    print("\n[1] 测试服务初始化...")
    try:
        from backend.services.dataset_service import DatasetService
        from backend.repositories.dataset_repository import DatasetRepository
        
        repo = DatasetRepository()
        service = DatasetService(repo)
        
        assert service is not None
        assert service.dataset_repository is not None
        print("   ✓ 服务初始化成功")
    except Exception as e:
        print(f"   ✗ 服务初始化失败: {e}")
        
    # 测试2: 验证API蓝图加载
    print("\n[2] 测试API蓝图加载...")
    try:
        from backend.api.dataset.dataset_management_api import dataset_management_bp
        
        assert dataset_management_bp is not None
        assert dataset_management_bp.name == 'dataset_management'
        print("   ✓ API蓝图加载成功")
    except Exception as e:
        print(f"   ✗ API蓝图加载失败: {e}")
        
    # 测试3: 验证统计功能
    print("\n[3] 测试统计功能...")
    try:
        from backend.services.dataset_service import DatasetService
        
        mock_repo = Mock()
        mock_repo.list_by_user.return_value = []
        
        service = DatasetService(mock_repo)
        stats = service.get_user_statistics("test_user")
        
        assert 'total_datasets' in stats
        assert 'total_size_bytes' in stats
        assert 'datasets_by_type' in stats
        print("   ✓ 统计功能正常")
    except Exception as e:
        print(f"   ✗ 统计功能测试失败: {e}")
        
    # 测试4: 验证搜索功能
    print("\n[4] 测试搜索功能...")
    try:
        from backend.services.dataset_service import DatasetService
        
        mock_repo = Mock()
        mock_repo.list_by_user.return_value = []
        
        service = DatasetService(mock_repo)
        service.tag_repository = Mock()
        service.tag_repository.get_tags.return_value = []
        
        result = service.search_datasets(
            user_id="test_user",
            dataset_type="text",
            search="test"
        )
        
        assert 'datasets' in result
        assert 'total_count' in result
        print("   ✓ 搜索功能正常")
    except Exception as e:
        print(f"   ✗ 搜索功能测试失败: {e}")
        
    # 测试5: 验证高级搜索
    print("\n[5] 测试高级搜索...")
    try:
        from backend.services.dataset_service import DatasetService
        
        mock_repo = Mock()
        mock_repo.list_by_user.return_value = []
        
        service = DatasetService(mock_repo)
        service.tag_repository = Mock()
        service.tag_repository.get_tags.return_value = []
        
        result = service.advanced_search(
            user_id="test_user",
            search_params={'q': 'test', 'type': 'text'},
            limit=10
        )
        
        assert 'datasets' in result
        assert 'query' in result
        print("   ✓ 高级搜索功能正常")
    except Exception as e:
        print(f"   ✗ 高级搜索测试失败: {e}")
        
    print("\n" + "=" * 60)
    print("功能测试完成")
    print("=" * 60)


if __name__ == '__main__':
    # 运行单元测试
    result = run_service_tests()
    
    # 运行功能测试
    run_functional_tests()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total - failures - errors
    
    print(f"总测试数: {total}")
    print(f"成功: {success}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
        
    sys.exit(0 if failures == 0 and errors == 0 else 1)
